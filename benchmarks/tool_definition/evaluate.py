"""Five-method, recorded next-action tool-definition evaluation on one T0 base."""
from __future__ import annotations

import hashlib
import json
import os
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from benchmarks import toolmemory
from benchmarks.backends.sglang import SglangBackend
from .core import FULL_CONTROL_POLICY, canonical_calls, parse_calls, sha256_file
from .prepare import LAYOUTS, SCHEMA

METHODS = ("c2kv", "streamingllm", "h2o", "snapkv", "pyramidkv")


class UpstreamUnavailable(RuntimeError):
    """The inference service cannot accept further requests."""


def read_manifest(path: Path, checkpoint: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, dict]]]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("purpose") != "recorded_next_action_tool_definition":
        raise ValueError("not a recorded tool-definition manifest")
    interface_policy = manifest.get("interface_policy", "none")
    toolmemory.ToolMemorySpec(ratio=8, interface_policy=interface_policy).validate()
    if interface_policy == "schema":
        if manifest.get("interface_render_profile") != toolmemory.INTERFACE_RENDER_PROFILE:
            raise ValueError("prepared tool interface render profile differs from this evaluator")
        if manifest.get("full_control_policy") != FULL_CONTROL_POLICY:
            raise ValueError("prepared Full control policy differs from this evaluator")
    elif "interface_render_profile" in manifest or "full_control_policy" in manifest:
        raise ValueError("default tool interface policy must not declare a render profile or Full control policy")
    if manifest["checkpoint"]["config_sha256"] != sha256_file(checkpoint / "config.json"):
        raise ValueError("T0 checkpoint config differs from preparation")
    model_files = manifest["checkpoint"].get("model_files_sha256")
    observed_model_files = {item.name for item in checkpoint.glob("model*.safetensors")}
    if not isinstance(model_files, dict) or not model_files or set(model_files) != observed_model_files:
        raise ValueError("T0 checkpoint weight file set differs from preparation")
    if any(sha256_file(checkpoint / name) != digest for name, digest in model_files.items()):
        raise ValueError("T0 checkpoint weights differ from preparation")
    for ratio in manifest["ratios"]:
        toolmemory.load_tool_checkpoint_contract(checkpoint, toolmemory.ToolMemorySpec(ratio=ratio))
    tokenizer_files = manifest.get("tokenizer_files_sha256")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files.get("tokenizer.json"):
        raise ValueError("prepared manifest lacks tokenizer identity")
    if any(not (checkpoint / name).is_file() or sha256_file(checkpoint / name) != digest
           for name, digest in tokenizer_files.items()):
        raise ValueError("checkpoint tokenizer differs from preparation")
    observed_tokenizer_files = {
        name for name in ("tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "added_tokens.json",
                          "chat_template.jinja")
        if (checkpoint / name).is_file()
    }
    if set(tokenizer_files) != observed_tokenizer_files:
        raise ValueError("checkpoint tokenizer file set differs from preparation")
    info = manifest["records"]
    records_path = (path.parent / info["path"]).resolve()
    if records_path.parent != path.parent or not records_path.is_file():
        raise ValueError("records must sit next to manifest")
    if records_path.stat().st_size != info["bytes"] or sha256_file(records_path) != info["sha256"]:
        raise ValueError("recorded decisions differ from manifest hash/size")
    qualification = manifest.get("qualification")
    if qualification is not None:
        excluded_path = (path.parent / qualification["path"]).resolve()
        if (excluded_path.parent != path.parent or not excluded_path.is_file() or
                excluded_path.stat().st_size != qualification["bytes"] or
                sha256_file(excluded_path) != qualification["sha256"]):
            raise ValueError("qualified exclusions differ from manifest")
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    count = 0
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("schema") != SCHEMA or record.get("layout") not in LAYOUTS:
                raise ValueError("invalid tool-definition record")
            if record.get("interface_policy", "none") != interface_policy:
                raise ValueError("recorded tool interface policy differs from manifest")
            if (interface_policy == "schema" and
                    record.get("interface_render_profile") != toolmemory.INTERFACE_RENDER_PROFILE):
                raise ValueError("recorded tool interface render profile differs from manifest")
            if interface_policy == "schema" and record.get("full_control_policy") != FULL_CONTROL_POLICY:
                raise ValueError("recorded Full control policy differs from manifest")
            if interface_policy == "none" and ("interface_render_profile" in record or
                                               "full_control_policy" in record):
                raise ValueError("recorded tool interface render profile or Full control policy differs from manifest")
            key = (record["decision_id"], record["ratio"])
            layouts = groups.setdefault(key, {})
            if record["layout"] in layouts:
                raise ValueError(f"duplicate layout for {key}")
            layouts[record["layout"]] = record
            count += 1
    if count != info["count"] or len(groups) != manifest["decisions"] * len(manifest["ratios"]):
        raise ValueError("prepared decision count differs from manifest")
    for key, layouts in groups.items():
        if set(layouts) != set(LAYOUTS):
            raise ValueError(f"incomplete paired layouts for {key}")
        if len({record["prompt_sha256"] for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different visible source for {key}")
        if len({json.dumps(record["gold_tool_calls"], sort_keys=True) for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different gold action for {key}")
        if len({record["base_prompt_tokens_without_tool_protocol"] for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different base prompt length for {key}")
    return manifest, groups


class SglangClient:
    """Small HTTP client for the same SGLang server used by paper benchmarks."""

    def __init__(self, upstream: str):
        if not upstream.startswith(("http://", "https://")):
            raise ValueError("upstream must be an HTTP(S) base URL")
        self.upstream = upstream.rstrip("/")
        self.backend = SglangBackend(self.post_json)

    def _json(self, method: str, path: str, payload: Mapping[str, Any] | None = None,
              timeout: int = 600) -> dict[str, Any]:
        body = (None if payload is None else
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        request = urllib.request.Request(
            self.upstream + path, data=body, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"SGLang {path} returned HTTP {error.code}: {detail[:1000]}"
            if error.code in {502, 503, 504}:
                raise UpstreamUnavailable(message) from error
            raise RuntimeError(message) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise UpstreamUnavailable(f"SGLang {path} is unavailable: {error}") from error
        if not isinstance(result, dict):
            raise ValueError(f"SGLang {path} returned a non-object JSON body")
        return result

    def post_json(self, path: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
        return self._json("POST", path, payload, timeout)

    def model_id(self, requested: str | None) -> str:
        models = self._json("GET", "/v1/models").get("data")
        ids = [item.get("id") for item in models or [] if isinstance(item, dict)]
        ids = [item for item in ids if isinstance(item, str) and item]
        if requested is not None:
            if requested not in ids:
                raise ValueError(f"model {requested!r} is absent from SGLang /v1/models")
            return requested
        if len(ids) != 1:
            raise ValueError("SGLang must expose one model, or --model must select one")
        return ids[0]


def _source_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source = manifest["source"]
    path = Path(source["path"]).resolve()
    if not path.is_file() or sha256_file(path) != source["sha256"]:
        raise ValueError("original recorded-decision source differs from preparation")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decision_id = row["decision_id"]
            if decision_id in rows:
                raise ValueError(f"duplicate source decision_id: {decision_id}")
            rows[decision_id] = row
    return rows


def _verified_source(record: Mapping[str, Any], sources: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    row = sources[record["decision_id"]]
    fingerprint = hashlib.sha256(json.dumps(
        {"messages": row["messages"], "tools": row["tools"]}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if fingerprint != record["prompt_sha256"]:
        raise ValueError(f"prepared prompt differs from source for {record['decision_id']}")
    if canonical_calls(row["gold_tool_calls"]) != canonical_calls(record["gold_tool_calls"]):
        raise ValueError(f"prepared gold differs from source for {record['decision_id']}")
    return row


def _outcome_http(gold: list[dict], content: str | None, tool_calls: Any) -> dict[str, Any]:
    try:
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("SGLang tool_calls is not a list")
        predicted = (canonical_calls(tool_calls) if tool_calls else
                     parse_calls(content or ""))
        malformed = False
    except (ValueError, TypeError, KeyError):
        predicted, malformed = [], True
    return {
        "tool_decision": bool(gold),
        "strict_ordered_call_correct": (not malformed and predicted == gold) if gold else None,
        "first_tool_name_correct": (bool(predicted) and predicted[0]["name"] == gold[0]["name"]) if gold else None,
        "false_tool_call": (malformed or bool(predicted)) if not gold else None,
        "malformed": malformed,
        "predicted_tool_calls": predicted,
    }


def _tool_eviction_report(response: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = response.get("metadata") or {}
    runtime = (metadata.get("sglang_runtime") or {}) if isinstance(metadata, dict) else {}
    for container in (response, metadata, runtime):
        if not isinstance(container, dict):
            continue
        report = container.get("kv_memory_report") or {}
        if isinstance(report, dict) and isinstance(report.get("tool_kv_eviction"), dict):
            return report["tool_kv_eviction"]
    return None


def _paper_measurement_request(staged: Mapping[str, Any],
                               full_messages: list[dict[str, Any]],
                               tools: list[dict[str, Any]], *, full: bool) -> dict[str, Any]:
    """Give the server the completed-history boundary and an exact Full source."""
    out = dict(staged)
    messages = out["messages"]
    original = [index for index, message in enumerate(messages)
                if not message.get("c2kv_key_hash")]
    if not original:
        raise ValueError("staged request contains no original messages")
    current = original[-1]
    history = [index for index in original[:-1]
               if messages[index].get("role") != "system"]
    history_start, history_end = (min(history), current) if history else (0, 0)
    hint = dict(out.get("c2kv_kv_memory_hint") or {})
    hint["paper_measurement"] = {
        "history_start_message_count": history_start,
        "history_message_count": history_end,
        "canonical_full_source": full,
        "canonical_source_messages": full_messages,
        "canonical_source_tools": tools,
    }
    out["c2kv_kv_memory_hint"] = hint
    return out


def _measured_kv(response: Mapping[str, Any], normalized: Mapping[str, Any],
                 method: str, target: int | None) -> dict[str, Any]:
    cost = normalized.get("cost") or {}
    measurement = cost.get("server_measurement") or {}
    if not isinstance(measurement, dict):
        raise ValueError("SGLang server_measurement is not an object")
    active = measurement.get("generation_active_kv_tokens")
    active_bytes = measurement.get("generation_active_kv_bytes")
    if type(active) is not int or active <= 0 or type(active_bytes) is not int or active_bytes <= 0:
        raise ValueError("SGLang response lacks generation-start active KV tokens/bytes; "
                         "launch the server with C2KV_PAPER_TELEMETRY=1")
    eviction = _tool_eviction_report(response)
    per_layer = None
    if method != "c2kv":
        if eviction is None or eviction.get("success") is not True or eviction.get("method") != method:
            raise ValueError(f"SGLang response lacks successful {method} tool-KV receipt")
        if eviction.get("history_untouched") is not True:
            raise ValueError("tool-KV eviction changed the Full history")
        no_op = eviction.get("no_op") is True
        if no_op:
            full_prompt = eviction.get("full_prompt_tokens")
            if (type(full_prompt) is not int or full_prompt <= 0 or
                    target is None or target < full_prompt):
                raise ValueError("tool-KV no-op lacks a full-prompt matched allowance")
        elif eviction.get("first_token_after_selection") is not True:
            raise ValueError("first action token was not generated after tool-KV selection")
        per_layer = eviction.get("resident_tokens_by_layer")
        if (not isinstance(per_layer, list) or not per_layer or
                any(type(value) is not int or value <= 0 for value in per_layer)):
            raise ValueError("tool-KV receipt lacks measured per-layer resident tokens")
        if (type(eviction.get("logical_kv_bytes")) is not int or
                eviction["logical_kv_bytes"] <= 0):
            raise ValueError("tool-KV receipt lacks logical KV bytes")
        if no_op and any(value != full_prompt for value in per_layer):
            raise ValueError("tool-KV no-op did not retain the Full prompt")
        if target is not None and eviction.get("target_resident_tokens_per_layer") != target:
            raise ValueError("SGLang applied a different matched total-KV allowance")
    return {"server_measurement": measurement,
            "generation_active_kv_tokens": active,
            "generation_active_kv_bytes": active_bytes,
            "tool_kv_eviction": eviction,
            "full_equivalent_no_op": bool(eviction and eviction.get("no_op") is True),
            "resident_kv_tokens_by_layer": per_layer}


def _http_result(record: Mapping[str, Any], *, method: str, layout: str,
                 response: dict[str, Any], normalized: dict[str, Any],
                 plan: Any, target: int | None,
                 full_active_tokens: int, full_active_bytes: int) -> dict[str, Any]:
    content = normalized.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("SGLang content must be text or null")
    gold = canonical_calls(record["gold_tool_calls"])
    measured = _measured_kv(response, normalized, method, target)
    native = list(plan.info["native_indices"])
    if native != record["native_indices"]:
        raise ValueError(f"server plan native tool indices differ from frozen {layout} layout")
    base = int(record["base_prompt_tokens_without_tool_protocol"])
    full_marginal = full_active_tokens - base
    resident_marginal = measured["generation_active_kv_tokens"] - base
    if full_marginal <= 0 or resident_marginal <= 0:
        raise ValueError("measured KV does not exceed the prepared no-tool prompt")
    budget_status = ("not_matched" if target is None else
                     "within_allowance" if measured["generation_active_kv_tokens"] <= target else
                     "exceeds_allowance")
    return {
        "schema": "c2kv-paper-tool-definition-result-v2",
        "decision_id": record["decision_id"], "source": record["source"],
        "ratio": record["ratio"], "method": method, "layout": layout,
        **({"interface_policy": record["interface_policy"]}
           if "interface_policy" in record else {}),
        **({"interface_render_profile": record["interface_render_profile"]}
           if "interface_render_profile" in record else {}),
        **({"full_control_policy": record["full_control_policy"]}
           if "full_control_policy" in record else {}),
        "k": record["k"], "seed": record["seed"],
        "prompt_sha256": record["prompt_sha256"],
        "gold_tool_calls": gold,
        "generated_text": content,
        "finish_reason": normalized.get("finish_reason"),
        "outcome": _outcome_http(gold, content, normalized.get("tool_calls")),
        "usage": normalized.get("usage"),
        "plan_info": dict(plan.info),
        "native_tool_indices": native,
        "planned_resident_kv_tokens": record["resident_kv_tokens"],
        "matched_allowance_tokens_per_layer": target,
        "budget_status": budget_status,
        "base_prompt_tokens_without_tool_protocol": base,
        "full_generation_active_kv_tokens": full_active_tokens,
        "full_generation_active_kv_bytes": full_active_bytes,
        "R_tool_numerator_tokens": full_marginal,
        "R_tool_denominator_tokens": resident_marginal,
        "R_tool": full_marginal / resident_marginal,
        "tool_kv_retention_fraction": resident_marginal / full_marginal,
        "R_tool_accounting": "server predecode resident token-equivalents minus prepared original-message no-tool tokens",
        "generation_active_kv_bytes_scope": "server predecode resident bytes including reference-position metadata",
        "tool_kv_eviction_logical_bytes_scope": "raw eviction receipt excludes reference-position metadata",
        **measured,
        "raw_response": response,
        "scope": "recorded next-action proxy; no tool execution or official task success",
    }


def _http_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["method"], row["layout"], row["ratio"])
        item = groups.setdefault(key, {
            "method": key[0], "layout": key[1], "ratio": key[2], "rows": 0,
            "gold_tool_rows": 0, "strict_ordered_call_correct": 0,
            "first_tool_name_correct": 0, "gold_no_call_rows": 0,
            "false_tool_calls": 0, "R_tool_numerator_tokens": 0,
            "R_tool_denominator_tokens": 0, "budget_exceeded_rows": 0,
        })
        item["rows"] += 1
        item["R_tool_numerator_tokens"] += row["R_tool_numerator_tokens"]
        item["R_tool_denominator_tokens"] += row["R_tool_denominator_tokens"]
        item["budget_exceeded_rows"] += int(row["budget_status"] == "exceeds_allowance")
        outcome = row["outcome"]
        if outcome["tool_decision"]:
            item["gold_tool_rows"] += 1
            item["strict_ordered_call_correct"] += int(outcome["strict_ordered_call_correct"])
            item["first_tool_name_correct"] += int(outcome["first_tool_name_correct"])
        else:
            item["gold_no_call_rows"] += 1
            item["false_tool_calls"] += int(outcome["false_tool_call"])
    result = []
    for key in sorted(groups):
        item = groups[key]
        item["strict_ordered_call_accuracy"] = (item["strict_ordered_call_correct"] /
                                                item["gold_tool_rows"] if item["gold_tool_rows"] else None)
        item["first_tool_name_accuracy"] = (item["first_tool_name_correct"] /
                                            item["gold_tool_rows"] if item["gold_tool_rows"] else None)
        item["false_tool_call_rate"] = (item["false_tool_calls"] /
                                        item["gold_no_call_rows"] if item["gold_no_call_rows"] else None)
        denominator = item["R_tool_denominator_tokens"]
        item["R_tool"] = item["R_tool_numerator_tokens"] / denominator if denominator > 0 else None
        result.append(item)
    return result


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"resume output lacks {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return row["decision_id"], row["ratio"], row["method"], row["layout"]


def evaluate(manifest_path: Path, checkpoint: Path, output: Path, *, upstream: str,
             max_new_tokens: int, model: str | None = None,
             methods: tuple[str, ...] = METHODS, layouts: tuple[str, ...] = LAYOUTS,
             limit: int | None = None, resume: bool = False,
             interface_policy: str = "none") -> dict[str, Any]:
    """Evaluate each frozen request; resume only under the identical run contract."""
    if (max_new_tokens < 1 or not methods or len(set(methods)) != len(methods)
            or any(item not in METHODS for item in methods)):
        raise ValueError("invalid methods or max_new_tokens")
    if not layouts or len(set(layouts)) != len(layouts) or any(item not in LAYOUTS for item in layouts):
        raise ValueError("unknown, repeated, or empty layout set")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest_path, checkpoint = manifest_path.resolve(), checkpoint.resolve()
    manifest, groups = read_manifest(manifest_path, checkpoint)
    if interface_policy != manifest.get("interface_policy", "none"):
        raise ValueError("evaluation interface policy differs from prepared manifest")
    sources = _source_rows(manifest)
    if output.exists() and not resume:
        raise FileExistsError(output)
    client = SglangClient(upstream)
    model_id = client.model_id(model)
    server_model_info = client._json("GET", "/model_info")
    capability = server_model_info.get("c2kv_native_packed") or {}
    tool_gist = capability.get("tool_gist") or {}
    if (tool_gist.get("enabled") is not True or
            tool_gist.get("config_sha256") != manifest["checkpoint"]["config_sha256"] or
            not tool_gist.get("identity")):
        raise ValueError("SGLang tool gist is absent or its config differs from frozen T0")
    selected = {key: groups[key] for key in sorted(groups)[:limit]}
    source_files = {
        "benchmarks/tool_definition/evaluate.py": Path(__file__),
        "benchmarks/tool_definition/core.py": Path(__file__).with_name("core.py"),
        "benchmarks/tool_definition/prepare.py": Path(__file__).with_name("prepare.py"),
        "benchmarks/toolmemory.py": Path(toolmemory.__file__),
        "benchmarks/backends/sglang.py": Path(__file__).resolve().parents[1] / "backends" / "sglang.py",
    }
    if interface_policy == "schema":
        source_files.update({
            "benchmarks/toolinterface.py": Path(__file__).resolve().parents[1] / "toolinterface.py",
            "benchmarks/toolmemory_joint.py": Path(__file__).resolve().parents[1] / "toolmemory_joint.py",
        })
    contract = {
        "schema": "c2kv-paper-tool-definition-run-contract-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_config_sha256": manifest["checkpoint"]["config_sha256"],
        "checkpoint_model_files_sha256": manifest["checkpoint"]["model_files_sha256"],
        "upstream": upstream, "server_model": model_id,
        "server_model_info": server_model_info,
        "client_source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "max_new_tokens": max_new_tokens,
        "methods": list(methods), "layouts": list(layouts), "limit": limit,
        **({"interface_policy": interface_policy,
            "interface_render_profile": toolmemory.INTERFACE_RENDER_PROFILE,
            "full_control_policy": FULL_CONTROL_POLICY}
           if interface_policy == "schema" else {}),
        "selected_decision_ratio_groups": len(selected),
        "selected_unique_decisions": len({key[0] for key in selected}),
    }
    contract_hash = hashlib.sha256(json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    contract["evaluation_config_sha256"] = contract_hash
    contract_path = output / "run_contract.json"
    rows_path, raw_path, errors_path = (
        output / "results.jsonl", output / "raw_responses.jsonl", output / "errors.jsonl")
    expected_by_pair = {
        key: {(key[0], key[1], method, layout)
              for method in methods for layout in layouts
              if method == "c2kv" or layout not in {"full", "retrieval"}}
        for key in selected
    }
    expected = set().union(*expected_by_pair.values()) if expected_by_pair else set()
    if output.exists():
        if not contract_path.is_file() or json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("resume run contract differs from frozen manifest, model, or request config")
        if not raw_path.is_file():
            raise ValueError("resume output lacks raw_responses.jsonl")
        rows_count, errors_count = 0, 0
        done: set[tuple[str, int, str, str]] = set()
        full_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for row in _iter_jsonl(rows_path):
            key = _row_key(row)
            if (row.get("evaluation_config_sha256") != contract_hash or
                    row.get("schema") != "c2kv-paper-tool-definition-result-v2" or
                    key not in expected or key in done or
                    row.get("prompt_sha256") != selected[key[:2]][key[3]]["prompt_sha256"]):
                raise ValueError(f"resume results contain mismatched or duplicate request {key}")
            done.add(key)
            rows_count += 1
            if key[2:] == ("c2kv", "full"):
                full_cache[key] = row["raw_response"]
        for error in _iter_jsonl(errors_path):
            key = _row_key(error)
            if (error.get("evaluation_config_sha256") != contract_hash or
                    (key not in expected and key != (key[0], key[1], "c2kv", "full")) or
                    key[:2] not in selected):
                raise ValueError("resume errors differ from the run contract")
            errors_count += 1
        full_cache = {key: response for key, response in full_cache.items()
                      if not expected_by_pair[key[:2]] <= done}
    else:
        output.mkdir(parents=True)
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        for path in (rows_path, raw_path, errors_path):
            path.touch()
        rows_count, errors_count, done, full_cache = 0, 0, set(), {}
    adapters: dict[tuple[str, str, int], toolmemory.ToolMemory] = {}
    failure: BaseException | None = None
    try:
        with (rows_path.open("a", encoding="utf-8") as sink,
              raw_path.open("a", encoding="utf-8") as raw_sink,
              errors_path.open("a", encoding="utf-8") as error_sink):
            def record_error(record: Mapping[str, Any], method: str, layout: str,
                             phase: str, exc: Exception) -> None:
                nonlocal errors_count
                item = {
                    "schema": "c2kv-paper-tool-definition-error-v1",
                    "decision_id": record["decision_id"], "ratio": record["ratio"],
                    "method": method, "layout": layout,
                    "evaluation_config_sha256": contract_hash,
                    **({"interface_policy": record["interface_policy"],
                        "interface_render_profile": record["interface_render_profile"],
                        "full_control_policy": record["full_control_policy"]}
                       if interface_policy == "schema" else {}),
                    "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                }
                _append_jsonl(error_sink, item)
                errors_count += 1

            def record_response(record: Mapping[str, Any], method: str, layout: str,
                                response: dict[str, Any]) -> None:
                _append_jsonl(raw_sink, {
                    "decision_id": record["decision_id"], "ratio": record["ratio"],
                    "method": method, "layout": layout,
                    "evaluation_config_sha256": contract_hash, "response": response,
                    **({"interface_policy": record["interface_policy"],
                        "interface_render_profile": record["interface_render_profile"],
                        "full_control_policy": record["full_control_policy"]}
                       if interface_policy == "schema" else {}),
                })

            for key, pair in selected.items():
                requested = expected_by_pair[key]
                if requested <= done:
                    continue
                source = _verified_source(pair["full"], sources)
                payload = {"model": model_id, "messages": source["messages"],
                           "tools": source["tools"], "temperature": 0,
                           "max_tokens": max_new_tokens, "stream": False}
                full_key = (key[0], key[1], "c2kv", "full")
                cached_full = full_cache.pop(full_key, None)
                try:
                    full_spec = toolmemory.ToolMemorySpec(ratio=pair["full"]["ratio"])
                    full_adapter_key = ("c2kv", "full", pair["full"]["ratio"])
                    if full_adapter_key not in adapters:
                        adapters[full_adapter_key] = toolmemory.ToolMemory(
                            full_spec, checkpoint, client.backend.extract_tokens)
                    full_staged, full_plan = adapters[full_adapter_key].prepare_full_history_request(
                        payload, native_override=list(range(len(source["tools"]))),
                        retrieval_only=True)
                    if full_plan is None:
                        raise ValueError(f"tool-memory adapter produced no Full plan for {key}")
                    full_staged = _paper_measurement_request(
                        full_staged, full_staged["messages"], source["tools"], full=True)
                    if cached_full is None:
                        full_response = client.post_json("/v1/chat/completions", full_staged)
                        record_response(pair["full"], "c2kv", "full", full_response)
                        full_normalized = client.backend.normalize_response(full_response)
                    else:
                        full_response = cached_full
                        full_normalized = client.backend.normalize_response(full_response)
                    full_measurement = _measured_kv(full_response, full_normalized, "c2kv", None)
                    if list(full_plan.info["native_indices"]) != pair["full"]["native_indices"]:
                        raise ValueError(f"Full native tool indices differ from frozen layout for {key}")
                except Exception as exc:
                    record_error(pair["full"], "c2kv", "full", "full_anchor", exc)
                    if isinstance(exc, UpstreamUnavailable):
                        raise
                    continue
                for method in methods:
                    for layout in layouts:
                        request_key = (key[0], key[1], method, layout)
                        if request_key not in expected or request_key in done:
                            continue
                        record = pair[layout]
                        try:
                            spec_layout = "hybrid" if layout in {"hybrid", "random"} else "uniform"
                            spec = toolmemory.ToolMemorySpec(
                                ratio=record["ratio"], layout=spec_layout,
                                top_k=record["k"] if spec_layout == "hybrid" else 0,
                                encoder="t0" if method == "c2kv" else method,
                                interface_policy=interface_policy,
                            )
                            adapter_key = (method, layout, record["ratio"])
                            if adapter_key not in adapters:
                                adapters[adapter_key] = toolmemory.ToolMemory(
                                    spec, checkpoint, client.backend.extract_tokens)
                            adapter = adapters[adapter_key]
                            override = (list(range(len(source["tools"]))) if layout == "full" else
                                        record["native_indices"] if layout in {"random", "retrieval"} else None)
                            target = (pair["hybrid"]["resident_kv_tokens"]
                                      if layout in {"random", "retrieval"} else
                                      pair["uniform"]["resident_kv_tokens"] if method != "c2kv" and layout == "uniform" else
                                      pair["hybrid"]["resident_kv_tokens"] if method != "c2kv" else None)
                            if method == "c2kv" and layout == "full":
                                response, normalized, plan = full_response, full_normalized, full_plan
                            else:
                                staged, plan = adapter.prepare_full_history_request(
                                    payload, native_override=override,
                                    retrieval_only=layout in {"full", "retrieval"},
                                    target_resident_tokens=target)
                                if plan is None:
                                    raise ValueError(f"tool-memory adapter produced no plan for {key}/{method}/{layout}")
                                staged = _paper_measurement_request(
                                    staged, full_staged["messages"], source["tools"],
                                    full=method != "c2kv")
                                response = client.post_json("/v1/chat/completions", staged)
                                record_response(record, method, layout, response)
                                normalized = client.backend.normalize_response(response)
                            result = _http_result(record, method=method, layout=layout,
                                                  response=response, normalized=normalized,
                                                  plan=plan, target=target,
                                                  full_active_tokens=full_measurement["generation_active_kv_tokens"],
                                                  full_active_bytes=full_measurement["generation_active_kv_bytes"])
                            result["evaluation_config_sha256"] = contract_hash
                            _append_jsonl(sink, result)
                            rows_count += 1
                            done.add(request_key)
                        except Exception as exc:
                            record_error(record, method, layout, "request", exc)
                            if isinstance(exc, UpstreamUnavailable):
                                raise
    except BaseException as exc:
        failure = exc
    unresolved = expected - done
    coverage: dict[tuple[str, str, int], dict[str, Any]] = {}
    for decision_id, ratio, method, layout in expected:
        item = coverage.setdefault((method, layout, ratio), {
            "method": method, "layout": layout, "ratio": ratio,
            "expected_rows": 0, "result_rows": 0,
        })
        item["expected_rows"] += 1
        item["result_rows"] += int((decision_id, ratio, method, layout) in done)
    report = {
        "schema": "c2kv-paper-tool-definition-evaluation-v2",
        "status": ("interrupted" if isinstance(failure, UpstreamUnavailable) else
                   "failed" if failure is not None else
                   "completed" if not unresolved else "completed_with_errors"),
        "transport": "sglang_http", **contract,
        "server_checkpoint_identity": "tool_config_matched_weights_unverified",
        "server_identity_scope": "queried_model_and_capability_metadata; engine_source_revision_unverified",
        "expected_rows": len(expected), "result_rows": rows_count,
        "unresolved_rows": len(unresolved), "request_error_events": errors_count,
        "group_coverage": [coverage[key] for key in sorted(coverage)],
        "results_sha256": sha256_file(rows_path),
        "raw_responses_sha256": sha256_file(raw_path),
        "errors_sha256": sha256_file(errors_path),
        "grouped_metrics": _http_summaries(_iter_jsonl(rows_path)) if not unresolved else [],
        "grouped_metrics_scope": ("complete_cohort" if not unresolved else
                                  "unavailable_until_all_requests_succeed"),
        "kv_measurement": "SGLang predecode resident token-equivalents and bytes (including reference-position metadata); raw eviction receipt preserves per-layer counts and logical KV bytes",
        "result_scope": "recorded next-action proxy; no tool execution or official task success",
    }
    if failure is not None:
        report["error"] = "".join(traceback.format_exception_only(type(failure), failure)).strip()
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    return report
