"""Build an offline BFCL mechanism audit from already-recorded artifacts.

The output deliberately separates a proxy-verified conversation state from an
exactly replayable HTTP request.  The proxy fingerprint excludes transport-only
message fields and tools, so a matching fingerprint is not sufficient evidence
for an exact replay payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "c2kv.mechanism_case.v1"
LABEL = "preliminary, n=1"
CATEGORY = "multi_turn_base"
PINNED_BFCL_SHA = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
ASSISTANT_CONTENT_CANDIDATES: Tuple[Optional[str], ...] = (
    None,
    "",
    "Action:",
    "Action:\n",
    "Action:\n\n",
    "\nAction:\n",
)


class AuditError(ValueError):
    """An artifact violates the fixed-corpus contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error


def _read_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise AuditError(f"{path}:{line_number}: expected JSON object")
                rows.append((line_number, value))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _one(paths: Iterable[Path], description: str) -> Path:
    found = sorted(path for path in paths if path.is_file())
    if len(found) != 1:
        raise AuditError(f"expected one {description}, found {len(found)}: {found}")
    return found[0]


def _source(path: Path, line: int, **fields: Any) -> Dict[str, Any]:
    return {"path": str(path.resolve()), "line": line, **fields}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    canonical: List[Dict[str, Any]] = []
    for message in messages:
        item: Dict[str, Any] = {
            "role": message.get("role") or "user",
            "content": _canonical_content(message.get("content")),
        }
        calls = message.get("tool_calls")
        if calls:
            canonical_calls = []
            for call in calls:
                function = call.get("function") or {}
                arguments_raw = function.get("arguments") or "{}"
                try:
                    arguments = json.loads(arguments_raw)
                except (TypeError, json.JSONDecodeError):
                    arguments = arguments_raw
                canonical_calls.append({
                    "name": function.get("name"), "arguments": arguments,
                })
            item["tool_calls"] = canonical_calls
        canonical.append(item)
    return canonical


def _messages_fingerprint(messages: Sequence[Mapping[str, Any]]) -> str:
    return _digest(_canonical_messages(messages))


def _canonical_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or "", ensure_ascii=False, sort_keys=True)


def _context_key(context: Mapping[str, Any]) -> Tuple[str, int, int, int]:
    task_id = context.get("task_id")
    user_turn = context.get("user_turn")
    step = context.get("step")
    attempt = context.get("attempt", 0)
    if not isinstance(task_id, str):
        raise AuditError(f"invalid eval_context task_id: {context!r}")
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (user_turn, step, attempt)):
        raise AuditError(f"invalid eval_context coordinates: {context!r}")
    return task_id, user_turn, step, attempt


def _task_number(task_id: str) -> int:
    match = re.search(r"(\d+)$", task_id)
    if not match:
        raise AuditError(f"task id has no numeric suffix: {task_id!r}")
    return int(match.group(1))


def _selection(run_dir: Path) -> Tuple[List[str], Path]:
    path = run_dir / "test_case_ids_to_generate.json"
    value = _read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get(CATEGORY), list):
        raise AuditError(f"{path}: missing list {CATEGORY!r}")
    task_ids = value[CATEGORY]
    if not task_ids or any(not isinstance(task_id, str) for task_id in task_ids):
        raise AuditError(f"{path}: invalid selected task ids")
    if len(set(task_ids)) != len(task_ids):
        raise AuditError(f"{path}: duplicate selected task ids")
    return task_ids, path


def _indexed_results(path: Path, expected_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for line_number, row in _read_jsonl(path):
        task_id = row.get("id")
        if not isinstance(task_id, str):
            raise AuditError(f"{path}:{line_number}: missing string id")
        if task_id in indexed:
            raise AuditError(f"{path}:{line_number}: duplicate id {task_id!r}")
        indexed[task_id] = {"row": row, "line": line_number}
    expected = set(expected_ids)
    actual = set(indexed)
    if actual != expected:
        raise AuditError(
            f"{path}: result id mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return indexed


def _turn_logs(result: Mapping[str, Any], path: Path, line: int) -> List[Dict[str, Any]]:
    inference_log = result.get("inference_log")
    if not isinstance(inference_log, list):
        raise AuditError(f"{path}:{line}: missing inference_log list")
    turns = [value for value in inference_log
             if isinstance(value, dict) and "begin_of_turn_query" in value]
    if not turns:
        raise AuditError(f"{path}:{line}: inference_log has no turn objects")
    return turns


def _action_from_result(indexed: Mapping[str, Dict[str, Any]], result_path: Path,
                        task_id: str, user_turn: int, step: int) -> Dict[str, Any]:
    record = indexed[task_id]
    turns = _turn_logs(record["row"], result_path, record["line"])
    if user_turn >= len(turns):
        raise AuditError(
            f"{result_path}:{record['line']}: no user_turn {user_turn} for {task_id}"
        )
    step_key = f"step_{step}"
    entries = turns[user_turn].get(step_key)
    if not isinstance(entries, list):
        raise AuditError(
            f"{result_path}:{record['line']}: no {step_key} for {task_id} turn {user_turn}"
        )
    assistants = [entry for entry in entries
                  if isinstance(entry, dict) and entry.get("role") == "assistant"]
    if len(assistants) != 1:
        raise AuditError(
            f"{result_path}:{record['line']}: expected one assistant action at "
            f"{task_id} turn {user_turn} step {step}, found {len(assistants)}"
        )
    content = assistants[0].get("content")
    source = _source(
        result_path, record["line"], task_id=task_id,
        user_turn=user_turn, step=step,
    )
    if isinstance(content, str):
        return {
            "kind": "text", "tool_calls": [], "text": content,
            "source": source,
        }
    if not isinstance(content, list) or not content:
        raise AuditError(
            f"{result_path}:{record['line']}: unsupported action content at "
            f"{task_id} turn {user_turn} step {step}"
        )
    calls: List[Dict[str, Any]] = []
    for index, call in enumerate(content):
        if not isinstance(call, dict) or len(call) != 1:
            raise AuditError(
                f"{result_path}:{record['line']}: malformed call {index} at "
                f"{task_id} turn {user_turn} step {step}"
            )
        name, arguments_raw = next(iter(call.items()))
        if not isinstance(name, str):
            raise AuditError(f"{result_path}:{record['line']}: non-string tool name")
        valid = True
        parsed: Any = None
        if isinstance(arguments_raw, str):
            try:
                parsed = json.loads(arguments_raw)
            except json.JSONDecodeError:
                valid = False
        else:
            valid = False
        calls.append({
            "name": name,
            "arguments_raw": arguments_raw,
            "arguments": parsed,
            "arguments_json_valid": valid,
        })
    return {
        "kind": "tool_calls", "tool_calls": calls, "text": None,
        "source": source,
    }


def _semantic_action(action: Mapping[str, Any]) -> Any:
    if action["kind"] == "text":
        return "text", action["text"]
    calls = []
    for call in action["tool_calls"]:
        arguments = (call["arguments"] if call["arguments_json_valid"]
                     else call["arguments_raw"])
        calls.append((call["name"], json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )))
    return "tool_calls", tuple(calls)


def _raw_action(action: Mapping[str, Any]) -> Any:
    if action["kind"] == "text":
        return "text", action["text"]
    return "tool_calls", tuple(
        (call["name"], call["arguments_raw"]) for call in action["tool_calls"]
    )


def _proxy_rows(path: Path, expected_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = {task_id: [] for task_id in expected_ids}
    seen_contexts: set[Tuple[str, int, int, int]] = set()
    for line_number, row in _read_jsonl(path):
        context = row.get("eval_context")
        if not isinstance(context, dict):
            raise AuditError(f"{path}:{line_number}: missing eval_context")
        key = _context_key(context)
        if context.get("benchmark") != "bfcl":
            raise AuditError(f"{path}:{line_number}: non-BFCL eval_context")
        if key[0] not in by_task:
            raise AuditError(f"{path}:{line_number}: unexpected task id {key[0]!r}")
        if key in seen_contexts:
            raise AuditError(f"{path}:{line_number}: duplicate eval_context {context!r}")
        seen_contexts.add(key)
        fp = row.get("fp")
        if not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{64}", fp):
            raise AuditError(f"{path}:{line_number}: invalid fp")
        by_task[key[0]].append({
            "row": row, "line": line_number, "context": context,
            "context_key": key,
        })
    missing = [task_id for task_id, rows in by_task.items() if not rows]
    if missing:
        raise AuditError(f"{path}: selected tasks without proxy rows: {missing}")
    return by_task


def _proxy_at(rows: Sequence[Mapping[str, Any]], user_turn: int, step: int,
              attempt: int = 0) -> Optional[Mapping[str, Any]]:
    matches = [item for item in rows
               if item["context_key"][1:] == (user_turn, step, attempt)]
    if len(matches) > 1:
        raise AuditError(
            f"duplicate proxy rows at user_turn={user_turn}, step={step}, attempt={attempt}"
        )
    return matches[0] if matches else None


def _score_success(score_path: Path, result_index: Mapping[str, Dict[str, Any]],
                   result_path: Path) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rows = _read_jsonl(score_path)
    if not rows:
        raise AuditError(f"{score_path}: empty score file")
    header_line, header = rows[0]
    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if (not isinstance(total, int) or isinstance(total, bool)
            or not isinstance(correct, int) or isinstance(correct, bool)
            or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool)
            or total != len(result_index) or not 0 <= correct <= total
            or abs(float(accuracy) - correct / total) > 1e-12):
        raise AuditError(f"{score_path}:{header_line}: invalid aggregate header")
    known: Dict[str, bool] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for line_number, row in rows[1:]:
        task_id = row.get("id")
        valid = row.get("valid")
        if task_id not in result_index or not isinstance(valid, bool):
            raise AuditError(f"{score_path}:{line_number}: invalid score detail row")
        if task_id in known:
            raise AuditError(f"{score_path}:{line_number}: duplicate score id {task_id!r}")
        known[task_id] = valid
        sources[task_id] = _source(
            score_path, line_number, task_id=task_id, method="explicit_detail",
        )
    unknown = set(result_index) - set(known)
    required_correct = correct - sum(known.values())
    required_incorrect = (total - correct) - (len(known) - sum(known.values()))
    if required_correct < 0 or required_incorrect < 0 \
            or required_correct + required_incorrect != len(unknown):
        raise AuditError(f"{score_path}: details contradict aggregate header")
    if unknown and required_incorrect == 0:
        inferred = True
    elif unknown and required_correct == 0:
        inferred = False
    elif unknown:
        raise AuditError(f"{score_path}: aggregate leaves task success ambiguous")
    else:
        inferred = False
    for task_id in unknown:
        known[task_id] = inferred
        sources[task_id] = {
            **_source(
                score_path, header_line, task_id=task_id,
                method="inferred_from_validated_failure_only_score",
            ),
            "prediction_source": _source(
                result_path, result_index[task_id]["line"], task_id=task_id,
            ),
        }
    if len(known) != total or sum(known.values()) != correct:
        raise AuditError(f"{score_path}: reconstructed success map contradicts header")
    return known, sources, {
        "correct_count": correct, "total_count": total,
        "accuracy": float(accuracy), "source": _source(score_path, header_line),
    }


def _command_model(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "command.json"
    command = _read_json(path)
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        raise AuditError(f"{path}: expected command argv list")
    values = [command[index + 1] for index, value in enumerate(command[:-1])
              if value == "--model"]
    if len(values) != 1:
        raise AuditError(f"{path}: expected exactly one --model")
    return {"value": values[0], "source": _source(path, 1)}


def _checkpoint(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "checkpoint_profile.resolved.json"
    value = _read_json(path)
    checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
    if not isinstance(checkpoint, dict) or not isinstance(value.get("profile_fingerprint"), str):
        raise AuditError(f"{path}: invalid checkpoint profile")
    return {
        "profile_fingerprint": value["profile_fingerprint"],
        "checkpoint_name": checkpoint.get("name"),
        "checkpoint_config_sha256": checkpoint.get("config_sha256"),
        "source": _source(path, 1),
    }


def _tool_task_lines(path: Path, task_ids: Sequence[str]) -> Dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found: Dict[str, int] = {}
    for task_id in task_ids:
        token = json.dumps(task_id, ensure_ascii=False)
        hits = [index for index, line in enumerate(lines, 1) if token in line]
        if len(hits) != 1:
            raise AuditError(f"{path}: expected one source line for task {task_id!r}")
        found[task_id] = hits[0]
    return found


def _tools_snapshot(path: Path, task_ids: Sequence[str]) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, List[Any]]]:
    value = _read_json(path)
    tasks = value.get("tasks") if isinstance(value, dict) else None
    if not isinstance(tasks, list):
        raise AuditError(f"{path}: missing tasks list")
    if value.get("bfcl_git_sha") not in (None, PINNED_BFCL_SHA):
        raise AuditError(
            f"{path}: BFCL sha {value.get('bfcl_git_sha')!r} != pinned {PINNED_BFCL_SHA}"
        )
    indexed: Dict[str, List[Any]] = {}
    for item in tasks:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str) \
                or not isinstance(item.get("tools"), list):
            raise AuditError(f"{path}: invalid task tools entry")
        task_id = item["task_id"]
        if task_id in indexed:
            raise AuditError(f"{path}: duplicate tools task {task_id!r}")
        indexed[task_id] = item["tools"]
    if set(indexed) != set(task_ids):
        raise AuditError(
            f"{path}: tools task mismatch; missing={sorted(set(task_ids)-set(indexed))}, "
            f"unexpected={sorted(set(indexed)-set(task_ids))}"
        )
    task_lines = _tool_task_lines(path, task_ids)
    digest = _sha256_file(path)
    references = {
        task_id: {
            "path": str(path.resolve()), "sha256": digest,
            "task_id": task_id, "task_line": task_lines[task_id],
            "tool_count": len(indexed[task_id]), "bfcl_git_sha": value.get("bfcl_git_sha"),
        }
        for task_id in task_ids
    }
    return references, indexed


def _canonical_calls(action: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if action["kind"] != "tool_calls":
        return []
    return [
        {
            "name": call["name"],
            "arguments": (call["arguments"] if call["arguments_json_valid"]
                          else call["arguments_raw"]),
        }
        for call in action["tool_calls"]
    ]


def _prior_tool_messages(indexed: Mapping[str, Dict[str, Any]], result_path: Path,
                         task_id: str, user_turn: int, step: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    record = indexed[task_id]
    turns = _turn_logs(record["row"], result_path, record["line"])
    turn = turns[user_turn]
    query = turn.get("begin_of_turn_query")
    previous = turn.get(f"step_{step - 1}") if step > 0 else None
    if not isinstance(query, list) or not isinstance(previous, list):
        return [], []
    query_messages = [
        {"role": item.get("role") or "user", "content": _canonical_content(item.get("content"))}
        for item in query if isinstance(item, dict)
    ]
    tool_messages = [
        {"role": "tool", "content": _canonical_content(item.get("content"))}
        for item in previous if isinstance(item, dict) and item.get("role") == "tool"
    ]
    return query_messages, tool_messages


def _turn_query_and_tool_results(indexed: Mapping[str, Dict[str, Any]],
                                 result_path: Path, task_id: str,
                                 user_turn: int, prior_step: int) -> Tuple[
                                     List[Dict[str, Any]], List[Dict[str, Any]]]:
    record = indexed[task_id]
    turns = _turn_logs(record["row"], result_path, record["line"])
    if user_turn >= len(turns):
        return [], []
    turn = turns[user_turn]
    query = turn.get("begin_of_turn_query")
    transition = turn.get(f"step_{prior_step}")
    if not isinstance(query, list) or not isinstance(transition, list):
        return [], []
    query_messages: List[Dict[str, Any]] = []
    for item in query:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) \
                or "content" not in item:
            return [], []
        query_messages.append({"role": item["role"], "content": item["content"]})
    tool_results: List[Dict[str, Any]] = []
    for item in transition:
        if isinstance(item, dict) and item.get("role") == "tool":
            if "content" not in item:
                return [], []
            tool_results.append({"role": "tool", "content": item["content"]})
    return query_messages, tool_results


def _verify_second_prefix(indexed: Mapping[str, Dict[str, Any]], result_path: Path,
                          task_id: str, context: Mapping[str, Any], fp: str,
                          prior_action: Mapping[str, Any]) -> Dict[str, Any]:
    user_turn = context["user_turn"]
    step = context["step"]
    if step <= 0 or prior_action["kind"] != "tool_calls":
        return {"verified": False, "canonical_messages": None, "candidate": None}
    query, tool_messages = _prior_tool_messages(
        indexed, result_path, task_id, user_turn, step,
    )
    if not query or not tool_messages:
        return {"verified": False, "canonical_messages": None, "candidate": None}
    matches: List[Tuple[Optional[str], List[Dict[str, Any]]]] = []
    for candidate in ASSISTANT_CONTENT_CANDIDATES:
        messages = [*query, {
            "role": "assistant", "content": _canonical_content(candidate),
            "tool_calls": _canonical_calls(prior_action),
        }, *tool_messages]
        if _digest(messages) == fp:
            matches.append((candidate, messages))
    if len(matches) > 1:
        raise AuditError(f"multiple canonical prefix candidates match {task_id} {context}")
    if not matches:
        return {"verified": False, "canonical_messages": None, "candidate": None}
    candidate, messages = matches[0]
    return {"verified": True, "canonical_messages": messages, "candidate": candidate}


def _load_run(run_dir: Path, expected_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    task_ids, selection_path = _selection(run_dir)
    if expected_ids is not None and list(expected_ids) != task_ids:
        raise AuditError(
            f"{selection_path}: selected ids/order differ from reference selection"
        )
    result_path = _one(
        run_dir.glob(f"result/**/BFCL_v4_{CATEGORY}_result.json"),
        f"{CATEGORY} result under {run_dir}",
    )
    score_path = _one(
        run_dir.glob(f"score/**/BFCL_v4_{CATEGORY}_score.json"),
        f"{CATEGORY} score under {run_dir}",
    )
    proxy_path = _one(run_dir.glob("logs/proxy_*.jsonl"), f"proxy log under {run_dir}")
    results = _indexed_results(result_path, task_ids)
    proxy = _proxy_rows(proxy_path, task_ids)
    success, success_sources, aggregate = _score_success(
        score_path, results, result_path,
    )
    return {
        "run_dir": run_dir, "task_ids": task_ids,
        "selection_path": selection_path, "result_path": result_path,
        "score_path": score_path, "proxy_path": proxy_path,
        "results": results, "proxy": proxy, "success": success,
        "success_sources": success_sources, "aggregate": aggregate,
        "model": _command_model(run_dir), "checkpoint": _checkpoint(run_dir),
    }


def _request_parameters(full: Mapping[str, Any]) -> Dict[str, Any]:
    adapter_path = Path(__file__).resolve().parent / "adapters" / "bfcl_adapter.py"
    lines = adapter_path.read_text(encoding="utf-8").splitlines()

    def locate(fragment: str) -> int:
        hits = [index for index, line in enumerate(lines, 1) if fragment in line]
        if len(hits) != 1:
            raise AuditError(f"{adapter_path}: expected one source line containing {fragment!r}")
        return hits[0]

    return {
        "model": {"status": "recorded_command", **full["model"]},
        "temperature": {
            "status": "pinned_bfcl_default_source", "value": 0.001,
            "source": {
                "url": (
                    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
                    f"{PINNED_BFCL_SHA}/berkeley-function-call-leaderboard/"
                    "bfcl_eval/__main__.py"
                ),
                "lines": "100-102", "bfcl_git_sha": PINNED_BFCL_SHA,
                "adapter_forwarding_source": _source(
                    adapter_path, locate('"temperature": self.temperature')
                ),
            },
        },
        "store": {
            "status": "adapter_source_constant", "value": False,
            "source": _source(adapter_path, locate('"store": False')),
        },
        "max_completion_tokens": {
            "status": "adapter_source_constant", "value": 4096,
            "source": _source(adapter_path, locate('"max_completion_tokens": 4096')),
        },
        "seed": {
            "status": "omitted_by_adapter", "value": None,
            "source": _source(adapter_path, locate("kwargs = {")),
        },
    }


def _generation_values(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": parameters["model"]["value"],
        "temperature": parameters["temperature"]["value"],
        "store": parameters["store"]["value"],
        "max_completion_tokens": parameters["max_completion_tokens"]["value"],
    }


def _first_request(full: Mapping[str, Any], c2kv: Mapping[str, Any],
                   task_id: str, tools_reference: Mapping[str, Any],
                   task_tools: Sequence[Any], parameters: Mapping[str, Any]) -> Dict[str, Any]:
    f_first = full["proxy"][task_id][0]
    c_first = c2kv["proxy"][task_id][0]
    if f_first["context"] != c_first["context"]:
        raise AuditError(f"first request context mismatch for {task_id}")
    if f_first["row"]["fp"] != c_first["row"]["fp"]:
        raise AuditError(f"first request fp mismatch for {task_id}")
    record = full["results"][task_id]
    turns = _turn_logs(record["row"], full["result_path"], record["line"])
    query = turns[f_first["context"]["user_turn"]].get("begin_of_turn_query")
    if not isinstance(query, list) or len(query) != f_first["row"].get("n_messages"):
        raise AuditError(f"cannot reconstruct first request messages for {task_id}")
    messages = []
    for item in query:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise AuditError(f"unexpected first request message shape for {task_id}")
        messages.append({"role": item["role"], "content": item["content"]})
    if _messages_fingerprint(messages) != f_first["row"]["fp"]:
        raise AuditError(f"first request messages fail fp verification for {task_id}")
    context = f_first["context"]
    payload = {
        **_generation_values(parameters), "messages": messages,
        "tools": list(task_tools), "c2kv_eval_context": context,
    }
    return {
        "schema": "c2kv.mechanism_first_request.v1", "label": LABEL,
        "case_id": f"{task_id}:request_0:{f_first['row']['fp'][:12]}",
        "task_id": task_id, "task_numeric_id": _task_number(task_id),
        "request_ordinal": 0, "eval_context": context,
        "fingerprint": f_first["row"]["fp"],
        "full_task_success": full["success"][task_id],
        "request": {
            "status": "exact", "payload": payload,
            "exactness_scope": (
                "request fields reconstructed from the pinned handler source, command, "
                "per-task tools snapshot, and messages verified by the proxy fingerprint; "
                "does not claim byte-identical OpenAI SDK HTTP serialization"
            ),
            "tools_snapshot": dict(tools_reference),
            "generation_parameters": dict(parameters),
        },
        "provenance": {
            "full_proxy": _source(
                full["proxy_path"], f_first["line"], task_id=task_id,
                user_turn=context["user_turn"], step=context["step"],
            ),
            "c2kv_proxy": _source(
                c2kv["proxy_path"], c_first["line"], task_id=task_id,
                user_turn=context["user_turn"], step=context["step"],
            ),
            "full_result": _source(
                full["result_path"], record["line"], task_id=task_id,
                user_turn=context["user_turn"], step=context["step"],
            ),
        },
    }


def _constructed_case(case: Mapping[str, Any], full: Mapping[str, Any],
                      task_tools: Sequence[Any], parameters: Mapping[str, Any]) -> Dict[str, Any]:
    task_id = case["task_id"]
    context = case["eval_context"]
    second = _proxy_at(
        full["proxy"][task_id], context["user_turn"], context["step"],
        context.get("attempt", 0),
    )
    first = _proxy_at(full["proxy"][task_id], 0, 0, 0)
    if second is None or first is None:
        raise AuditError(f"constructed case lost required proxy state for {task_id}")
    prior_step = context["step"] - 1
    prior_action = _action_from_result(
        full["results"], full["result_path"], task_id,
        context["user_turn"], prior_step,
    ) if prior_step >= 0 else None
    query, results = _turn_query_and_tool_results(
        full["results"], full["result_path"], task_id,
        context["user_turn"], prior_step,
    ) if prior_step >= 0 else ([], [])
    calls = prior_action["tool_calls"] if prior_action \
        and prior_action["kind"] == "tool_calls" else []
    complete = (
        bool(query) and bool(calls) and len(calls) == len(results)
        and all(call["arguments_json_valid"] and isinstance(call["arguments_raw"], str)
                for call in calls)
    )
    messages: Optional[List[Dict[str, Any]]] = None
    constructed_fp: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    if complete:
        call_messages = []
        result_messages = []
        for index, (call, result) in enumerate(zip(calls, results)):
            call_id = f"c2kv_constructed_{_task_number(task_id)}_r1_{index}"
            call_messages.append({
                "id": call_id, "type": "function",
                "function": {
                    "name": call["name"], "arguments": call["arguments_raw"],
                },
            })
            result_messages.append({
                "role": "tool", "content": result["content"],
                "tool_call_id": call_id,
            })
        messages = [
            *query,
            {"role": "assistant", "content": "", "tool_calls": call_messages},
            *result_messages,
        ]
        constructed_fp = _messages_fingerprint(messages)
        payload = {
            **_generation_values(parameters), "messages": messages,
            "tools": list(task_tools), "c2kv_eval_context": context,
        }
    status = ("constructed_from_logged_transition_v1" if complete
              else "missing_incomplete_logged_transition")
    request = {
        "status": status, "payload": payload,
        "original_message_fingerprint": case["fingerprint"],
        "constructed_message_fingerprint": constructed_fp,
        "non_equivalence": (
            "Constructed from logged decoded calls and tool results; original assistant "
            "text and transport ids were not recorded, so this payload is not a historical replay."
        ),
        "construction": {
            "assistant_content": "",
            "tool_call_id_scheme": (
                "c2kv_constructed_<task_numeric_id>_r1_<zero_based_call_index>"
            ),
            "call_count": len(calls), "tool_result_count": len(results),
        },
        "tools_snapshot": case["request"]["tools_snapshot"],
        "sources": {
            "full_first_proxy": _source(
                full["proxy_path"], first["line"], task_id=task_id,
                user_turn=first["context"]["user_turn"], step=first["context"]["step"],
            ),
            "full_second_proxy": _source(
                full["proxy_path"], second["line"], task_id=task_id,
                user_turn=context["user_turn"], step=context["step"],
            ),
            "full_result_transition": _source(
                full["result_path"], prior_action["source"]["line"] if prior_action else 1,
                task_id=task_id, user_turn=context["user_turn"], step=prior_step,
            ),
        },
    }
    return {
        **{key: value for key, value in case.items() if key != "request"},
        "schema": "c2kv.mechanism_constructed_case.v1", "request": request,
    }


def collect(full_run: Path, c2kv_run: Path, tools_path: Path) -> Dict[str, Any]:
    full = _load_run(full_run)
    c2kv = _load_run(c2kv_run, full["task_ids"])
    task_ids = full["task_ids"]
    tools, tool_payloads = _tools_snapshot(tools_path, task_ids)
    request_parameters = _request_parameters(full)
    comparisons: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    first_requests: List[Dict[str, Any]] = []
    constructed_cases: List[Dict[str, Any]] = []

    first_counts = Counter()
    second_counts = Counter()
    ordinal_second_counts = Counter()
    all_counts = Counter()
    request_statuses = Counter()
    second_kind_matrix = Counter()
    full_success_cases = Counter()

    for task_id in task_ids:
        full_rows = full["proxy"][task_id]
        c2kv_rows = c2kv["proxy"][task_id]
        full_by_context = {item["context_key"][1:]: (ordinal, item)
                           for ordinal, item in enumerate(full_rows)}
        c2kv_by_context = {item["context_key"][1:]: (ordinal, item)
                           for ordinal, item in enumerate(c2kv_rows)}
        for coordinates in sorted(set(full_by_context) | set(c2kv_by_context)):
            f_pair = full_by_context.get(coordinates)
            c_pair = c2kv_by_context.get(coordinates)
            f_action = (_action_from_result(
                full["results"], full["result_path"], task_id,
                coordinates[0], coordinates[1],
            ) if f_pair else None)
            c_action = (_action_from_result(
                c2kv["results"], c2kv["result_path"], task_id,
                coordinates[0], coordinates[1],
            ) if c_pair else None)
            both = f_pair is not None and c_pair is not None
            fp_match = (both and f_pair[1]["row"]["fp"] == c_pair[1]["row"]["fp"])
            action_match = (both and _semantic_action(f_action) == _semantic_action(c_action))
            action_raw_match = (both and _raw_action(f_action) == _raw_action(c_action))
            all_counts["union_contexts"] += 1
            all_counts["paired_contexts"] += int(both)
            all_counts["common_fingerprints"] += int(fp_match)
            all_counts["action_matches"] += int(action_match)
            comparisons.append({
                "label": LABEL, "task_id": task_id,
                "task_numeric_id": _task_number(task_id),
                "user_turn": coordinates[0], "step": coordinates[1],
                "attempt": coordinates[2],
                "full_request_ordinal": f_pair[0] if f_pair else None,
                "c2kv_request_ordinal": c_pair[0] if c_pair else None,
                "context_in_full": bool(f_pair), "context_in_c2kv": bool(c_pair),
                "full_fp": f_pair[1]["row"]["fp"] if f_pair else None,
                "c2kv_fp": c_pair[1]["row"]["fp"] if c_pair else None,
                "fp_match": fp_match if both else None,
                "full_action_kind": f_action["kind"] if f_action else None,
                "c2kv_action_kind": c_action["kind"] if c_action else None,
                "action_match": action_match if both else None,
                "action_raw_match": action_raw_match if both else None,
                "full_task_success": full["success"][task_id],
                "c2kv_task_success": c2kv["success"][task_id],
                "full_proxy_path": str(full["proxy_path"].resolve()) if f_pair else None,
                "full_proxy_line": f_pair[1]["line"] if f_pair else None,
                "c2kv_proxy_path": str(c2kv["proxy_path"].resolve()) if c_pair else None,
                "c2kv_proxy_line": c_pair[1]["line"] if c_pair else None,
                "full_result_path": str(full["result_path"].resolve()) if f_action else None,
                "full_result_line": f_action["source"]["line"] if f_action else None,
                "c2kv_result_path": str(c2kv["result_path"].resolve()) if c_action else None,
                "c2kv_result_line": c_action["source"]["line"] if c_action else None,
            })

        f_first, c_first = full_rows[0], c2kv_rows[0]
        first_requests.append(_first_request(
            full, c2kv, task_id, tools[task_id], tool_payloads[task_id],
            request_parameters,
        ))
        first_counts["task_count"] += 1
        first_context_match = f_first["context"] == c_first["context"]
        first_fp_match = f_first["row"]["fp"] == c_first["row"]["fp"]
        first_counts["context_match"] += int(first_context_match)
        first_counts["fp_match"] += int(first_context_match and first_fp_match)
        f_first_action = _action_from_result(
            full["results"], full["result_path"], task_id,
            f_first["context"]["user_turn"], f_first["context"]["step"],
        )
        c_first_action = _action_from_result(
            c2kv["results"], c2kv["result_path"], task_id,
            c_first["context"]["user_turn"], c_first["context"]["step"],
        )
        first_counts["action_match"] += int(
            first_context_match and _semantic_action(f_first_action) == _semantic_action(c_first_action)
        )

        ordinal_second_counts["task_count"] += 1
        if len(full_rows) >= 2 and len(c2kv_rows) >= 2:
            ordinal_second_counts["both_present"] += 1
            ordinal_context_match = full_rows[1]["context"] == c2kv_rows[1]["context"]
            ordinal_second_counts["context_match"] += int(ordinal_context_match)
            ordinal_second_counts["fp_match"] += int(
                ordinal_context_match
                and full_rows[1]["row"]["fp"] == c2kv_rows[1]["row"]["fp"]
            )
        else:
            ordinal_second_counts["missing_full"] += int(len(full_rows) < 2)
            ordinal_second_counts["missing_c2kv"] += int(len(c2kv_rows) < 2)

        second_counts["task_count"] += 1
        f_second = _proxy_at(full_rows, 0, 1, 0)
        c_second = _proxy_at(c2kv_rows, 0, 1, 0)
        if f_second is None:
            second_counts["missing_full"] += 1
        if c_second is None:
            second_counts["missing_c2kv"] += 1
        if f_second is None or c_second is None:
            continue
        second_counts["both_present"] += 1
        context_match = f_second["context"] == c_second["context"]
        second_counts["context_match"] += int(context_match)
        if not context_match:
            continue
        fp_match = f_second["row"]["fp"] == c_second["row"]["fp"]
        second_counts["fp_match"] += int(fp_match)
        if not fp_match:
            continue
        context = f_second["context"]
        f_action = _action_from_result(
            full["results"], full["result_path"], task_id,
            context["user_turn"], context["step"],
        )
        c_action = _action_from_result(
            c2kv["results"], c2kv["result_path"], task_id,
            context["user_turn"], context["step"],
        )
        f_prior = _action_from_result(
            full["results"], full["result_path"], task_id,
            context["user_turn"], context["step"] - 1,
        ) if context["step"] > 0 else f_first_action
        c_prior = _action_from_result(
            c2kv["results"], c2kv["result_path"], task_id,
            context["user_turn"], context["step"] - 1,
        ) if context["step"] > 0 else c_first_action
        f_reconstruction = _verify_second_prefix(
            full["results"], full["result_path"], task_id,
            context, f_second["row"]["fp"], f_prior,
        )
        c_reconstruction = _verify_second_prefix(
            c2kv["results"], c2kv["result_path"], task_id,
            context, c_second["row"]["fp"], c_prior,
        )
        canonical_verified = (
            f_reconstruction["verified"] and c_reconstruction["verified"]
            and f_reconstruction["canonical_messages"]
            == c_reconstruction["canonical_messages"]
        )
        request_status = "missing_exact_replay_request"
        request_statuses[request_status] += 1
        action_match = _semantic_action(f_action) == _semantic_action(c_action)
        action_raw_match = _raw_action(f_action) == _raw_action(c_action)
        second_counts["action_match"] += int(action_match)
        second_counts["action_raw_match"] += int(action_raw_match)
        second_counts["canonical_prefix_verified"] += int(canonical_verified)
        second_kind_matrix[f"{f_action['kind']}__{c_action['kind']}"] += 1
        full_success_cases[str(full["success"][task_id]).lower()] += 1
        missing_request_fields = [
            "transport-only tool call ids and tool_call_id links",
            "original OpenAI HTTP request body",
        ]
        if not canonical_verified:
            missing_request_fields.insert(
                0, "original assistant content for the prior tool-call response"
            )
        case = {
            "schema": SCHEMA, "label": LABEL,
            "case_id": f"{task_id}:request_1:{f_second['row']['fp'][:12]}",
            "task_id": task_id, "task_numeric_id": _task_number(task_id),
            "request_ordinal": 1, "eval_context": context,
            "fingerprint": f_second["row"]["fp"],
            "full_task_success": full["success"][task_id],
            "full_task_success_source": full["success_sources"][task_id],
            "actions": {"full_r1": f_action, "c2kv4": c_action},
            "action_match": action_match, "action_raw_match": action_raw_match,
            "request": {
                "status": request_status, "payload": None,
                "canonical_messages": (
                    f_reconstruction["canonical_messages"] if canonical_verified else None
                ),
                "canonical_fingerprint_verified": canonical_verified,
                "assistant_content_candidate": (
                    f_reconstruction["candidate"] if canonical_verified else None
                ),
                "tools_snapshot": tools[task_id],
                "generation_parameters": request_parameters,
                "missing": missing_request_fields,
            },
            "provenance": {
                "full_proxy": _source(
                    full["proxy_path"], f_second["line"], task_id=task_id,
                    user_turn=context["user_turn"], step=context["step"],
                ),
                "c2kv_proxy": _source(
                    c2kv["proxy_path"], c_second["line"], task_id=task_id,
                    user_turn=context["user_turn"], step=context["step"],
                ),
                "full_checkpoint": full["checkpoint"],
                "c2kv_checkpoint": c2kv["checkpoint"],
            },
        }
        cases.append(case)
        constructed_cases.append(_constructed_case(
            case, full, tool_payloads[task_id], request_parameters,
        ))

    cases.sort(key=lambda case: case["task_numeric_id"])
    first_requests.sort(key=lambda case: case["task_numeric_id"])
    constructed_cases.sort(key=lambda case: case["task_numeric_id"])
    comparisons.sort(key=lambda row: (
        row["task_numeric_id"], row["user_turn"], row["step"], row["attempt"]
    ))
    same_input_by_success: Dict[str, Dict[str, int]] = {}
    for success in (False, True):
        subset = [case for case in cases if case["full_task_success"] is success]
        matches = sum(case["action_match"] for case in subset)
        same_input_by_success[str(success).lower()] = {
            "denominator": len(subset), "action_match": matches,
            "action_mismatch": len(subset) - matches,
        }
    same_input_matches = sum(case["action_match"] for case in cases)
    same_input_raw_matches = sum(case["action_raw_match"] for case in cases)
    verified_prefixes = [
        case for case in cases if case["request"]["canonical_fingerprint_verified"]
    ]
    canonical_candidates = Counter(
        ("null" if case["request"]["assistant_content_candidate"] is None
         else json.dumps(
             case["request"]["assistant_content_candidate"], ensure_ascii=False
         ))
        for case in verified_prefixes
    )
    summary = {
        "schema": "c2kv.mechanism_audit.v1", "label": LABEL,
        "fixed_task_count": len(task_ids), "case_count": len(cases),
        "full_score": full["aggregate"], "c2kv_score": c2kv["aggregate"],
        "first_request": dict(first_counts),
        "first_turn_step_1": {
            "definition": "eval_context user_turn=0, step=1, attempt=0",
            **dict(second_counts),
        },
        "overall_request_ordinal_1": {
            "definition": "second proxy request observed for each task regardless of user_turn",
            **dict(ordinal_second_counts),
        },
        "all_turn_step_contexts": dict(all_counts),
        "case_request_statuses": dict(request_statuses),
        "historical_canonical_prefix": {
            "denominator": len(cases), "verified": len(verified_prefixes),
            "missing": len(cases) - len(verified_prefixes),
            "assistant_content_candidates": dict(canonical_candidates),
            "scope": (
                "proxy-fingerprint-verified canonical messages; excludes original "
                "transport ids and byte-level HTTP serialization"
            ),
        },
        "same_input_action_comparison": {
            "definition": (
                "only first-turn step-1 states with equal full_r1 and c2kv4 proxy fp"
            ),
            "denominator": len(cases), "action_match": same_input_matches,
            "action_mismatch": len(cases) - same_input_matches,
            "action_raw_match": same_input_raw_matches,
            "action_raw_mismatch": len(cases) - same_input_raw_matches,
            "action_kind_matrix": dict(second_kind_matrix),
            "by_full_task_success": same_input_by_success,
        },
        "first_request_exact_count": sum(
            item["request"]["status"] == "exact" for item in first_requests
        ),
        "constructed_case_statuses": dict(Counter(
            item["request"]["status"] for item in constructed_cases
        )),
        "case_action_kind_matrix": dict(second_kind_matrix),
        "case_full_task_success": dict(full_success_cases),
        "inputs": {
            "full_run": str(full_run.resolve()),
            "c2kv_run": str(c2kv_run.resolve()),
            "full_selection": str(full["selection_path"].resolve()),
            "c2kv_selection": str(c2kv["selection_path"].resolve()),
            "tools_snapshot": str(tools_path.resolve()),
            "tools_snapshot_sha256": _sha256_file(tools_path),
        },
        "gaps": [
            f"Canonical message prefixes are proxy-fingerprint verified for "
            f"{len(verified_prefixes)}/{len(cases)} common states; the remaining "
            f"{len(cases) - len(verified_prefixes)} lack a verified original assistant content.",
            "No exact historical replay payload is emitted for the 35 common states because "
            "transport ids and the byte-level HTTP body are not present in the local artifacts.",
            "A common proxy fingerprint proves equality of canonical pre-assembly messages; "
            "it does not prove byte-identical request transport or tool payload equality.",
        ],
    }
    return {
        "summary": summary, "cases": cases, "first_requests": first_requests,
        "constructed_cases": constructed_cases, "comparisons": comparisons,
    }


SUMMARY_CSV_FIELDS = [
    "label", "fixed_task_count", "case_count", "full_correct_count",
    "full_total_count", "c2kv_correct_count", "c2kv_total_count",
    "first_context_match", "first_fp_match", "first_action_match",
    "turn0_step1_both_present", "turn0_step1_context_match", "turn0_step1_fp_match",
    "turn0_step1_action_match", "turn0_step1_canonical_prefix_verified",
    "ordinal1_both_present", "ordinal1_context_match", "ordinal1_fp_match",
    "same_input_denominator", "same_input_action_match", "same_input_action_mismatch",
    "same_input_full_tool_c2kv_text", "same_input_full_text_c2kv_text",
    "same_input_full_tool_c2kv_tool",
    "all_paired_contexts", "all_common_fingerprints",
    "historical_exact_request_count", "historical_missing_exact_request_count",
    "first_request_exact_count", "constructed_request_count",
    "constructed_missing_count",
]


def _summary_csv_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    first = summary["first_request"]
    second = summary["first_turn_step_1"]
    ordinal_second = summary["overall_request_ordinal_1"]
    all_contexts = summary["all_turn_step_contexts"]
    statuses = summary["case_request_statuses"]
    constructed = summary["constructed_case_statuses"]
    same_input = summary["same_input_action_comparison"]
    kind_matrix = same_input["action_kind_matrix"]
    return {
        "label": LABEL, "fixed_task_count": summary["fixed_task_count"],
        "case_count": summary["case_count"],
        "full_correct_count": summary["full_score"]["correct_count"],
        "full_total_count": summary["full_score"]["total_count"],
        "c2kv_correct_count": summary["c2kv_score"]["correct_count"],
        "c2kv_total_count": summary["c2kv_score"]["total_count"],
        "first_context_match": first.get("context_match", 0),
        "first_fp_match": first.get("fp_match", 0),
        "first_action_match": first.get("action_match", 0),
        "turn0_step1_both_present": second.get("both_present", 0),
        "turn0_step1_context_match": second.get("context_match", 0),
        "turn0_step1_fp_match": second.get("fp_match", 0),
        "turn0_step1_action_match": second.get("action_match", 0),
        "turn0_step1_canonical_prefix_verified": second.get("canonical_prefix_verified", 0),
        "ordinal1_both_present": ordinal_second.get("both_present", 0),
        "ordinal1_context_match": ordinal_second.get("context_match", 0),
        "ordinal1_fp_match": ordinal_second.get("fp_match", 0),
        "same_input_denominator": same_input["denominator"],
        "same_input_action_match": same_input["action_match"],
        "same_input_action_mismatch": same_input["action_mismatch"],
        "same_input_full_tool_c2kv_text": kind_matrix.get("tool_calls__text", 0),
        "same_input_full_text_c2kv_text": kind_matrix.get("text__text", 0),
        "same_input_full_tool_c2kv_tool": kind_matrix.get("tool_calls__tool_calls", 0),
        "all_paired_contexts": all_contexts.get("paired_contexts", 0),
        "all_common_fingerprints": all_contexts.get("common_fingerprints", 0),
        "historical_exact_request_count": statuses.get("exact", 0),
        "historical_missing_exact_request_count": statuses.get(
            "missing_exact_replay_request", 0
        ),
        "first_request_exact_count": summary["first_request_exact_count"],
        "constructed_request_count": constructed.get(
            "constructed_from_logged_transition_v1", 0
        ),
        "constructed_missing_count": constructed.get(
            "missing_incomplete_logged_transition", 0
        ),
    }


def write_outputs(out_dir: Path, audit: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(audit["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for case in audit["cases"]:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    for name, key in (("first_requests.jsonl", "first_requests"),
                      ("constructed_cases.jsonl", "constructed_cases")):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for case in audit[key]:
                handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(_summary_csv_row(audit["summary"]))
    comparison_fields = list(audit["comparisons"][0]) if audit["comparisons"] else []
    with (out_dir / "action_comparison.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        if comparison_fields:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(audit["comparisons"])


def default_paths(repo_root: Path) -> Tuple[Path, Path, Path, Path]:
    base = repo_root / "outputs" / "eval_20260906" / "core" / "bfcl_base"
    return (
        base / "full_r1",
        base / "c2kv4",
        base / "protocol_fidelity" / "bfcl_base40_advertised_tools.json",
        repo_root / "outputs" / "mechanism_20260906" / "offline_v1",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    full_default, c2kv_default, tools_default, out_default = default_paths(repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, default=full_default)
    parser.add_argument("--c2kv-run", type=Path, default=c2kv_default)
    parser.add_argument("--tools-snapshot", type=Path, default=tools_default)
    parser.add_argument("--out", type=Path, default=out_default)
    args = parser.parse_args(argv)
    audit = collect(
        args.full_run.resolve(), args.c2kv_run.resolve(),
        args.tools_snapshot.resolve(),
    )
    write_outputs(args.out.resolve(), audit)
    print(json.dumps(_summary_csv_row(audit["summary"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
