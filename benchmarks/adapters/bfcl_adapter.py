"""BFCL adapter (server-side verified version).

The working runner lives at ~/bench_results/bfcl_arm.py on the NPU server
(registers an OpenAI handler under model name ``c2kv-hf`` pointed at the
hf_server or the arm proxy, then drives the official bfcl CLI in-process).
This module mirrors it so the recipe is version-controlled; it must run in
the ``bench`` venv with cwd inside the gorilla checkout.

Usage:
    python benchmarks/adapters/bfcl_adapter.py \
        --base-url http://127.0.0.1:34000/v1 --categories multi_turn_base

ARGV CHANGE (2026-09-05), this standalone CLI only: it used to build its own
argv and append ``--run-ids`` to BOTH generate and evaluate; it now shares
``run_bfcl`` with ``run(ctx)``, so a subset run evaluates with
``--partial-eval`` (this vintage's evaluate has no ``--run-ids`` and scored
the whole category instead) and the terminal-state gate can exit non-zero.
Both entrypoints now keep the embedded Typer CLI alive through generation,
evaluation and terminal checks; run.py also isolates BFCL's result, score
and subset-id files under its output directory. A subset number from the OLD
standalone recipe is a full-category score — do not compare it with a new
one (README "BFCL standalone-CLI argv change").

Env fixes the bench venv needed: anthropic>=new, openai>=1.66, soundfile,
tree-sitter==0.21.3 + tree-sitter-java.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.base import RunContext, v1  # noqa: E402
from bfcl_completion import bfcl_row_is_terminal, terminal_failure_kind  # noqa: E402
from measurement.telemetry import HarnessTelemetry, current_episode  # noqa: E402

NAME = "bfcl"
MODEL_NAME = "c2kv-hf"  # BFCL handler key / result-dir name (stable layout)
SERVED_MODEL = "c2kv-agent"  # default served model name at the endpoint
HARNESS_TELEMETRY_PATH: Optional[Path] = None


def _response_dict(response: Any) -> Any:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    return response if isinstance(response, (dict, list, str, int, float, bool, type(None))) else repr(response)


def _proxy_request_id(response: Any) -> Optional[str]:
    extra = getattr(response, "model_extra", None)
    proxy = extra.get("c2kv_proxy") if isinstance(extra, dict) else None
    if not isinstance(proxy, dict):
        dumped = _response_dict(response)
        proxy = dumped.get("c2kv_proxy") if isinstance(dumped, dict) else None
    value = proxy.get("request_id") if isinstance(proxy, dict) else None
    return str(value) if value else None


def _install_timed_executor(telemetry: HarnessTelemetry) -> None:
    """Wrap the symbol imported by BaseHandler, preserving BFCL state/order."""
    import bfcl_eval.model_handler.base_handler as base_handler

    current = base_handler.execute_multi_turn_func_call
    original = getattr(current, "_c2kv_original_execute", current)

    def timed_execute(func_call_list, *args, **kwargs):
        if not func_call_list:
            return original(func_call_list, *args, **kwargs)
        results = []
        involved_instances = {}
        for index, action in enumerate(func_call_list):
            start_unix = time.time_ns()
            start_perf = time.perf_counter_ns()
            error = None
            try:
                single_results, involved_instances = original(
                    [action], *args, **kwargs)
                outcome = single_results[0] if single_results else None
                status = ("error" if isinstance(outcome, str)
                          and outcome.startswith("Error during execution:") else "ok")
                results.extend(single_results)
            except BaseException as exc:
                outcome = None
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                telemetry.record_action(
                    action=action, outcome=outcome,
                    start_unix_ns=start_unix,
                    duration_ns=time.perf_counter_ns() - start_perf,
                    action_index=index, status=status, error=error,
                    metadata={"harness": "bfcl"},
                )
        return results, involved_instances

    timed_execute._c2kv_original_execute = original  # type: ignore[attr-defined]
    base_handler.execute_multi_turn_func_call = timed_execute


def add_arguments(parser) -> None:
    """BFCL-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--run-ids", default="",
                        help="bfcl: comma-separated official case ids for a subset run")
    parser.add_argument(
        "--bfcl-refill-rounds", type=int, default=0,
        help=("explicit bounded extra generation rounds for missing/incomplete "
              "BFCL rows; disabled by default"),
    )


def default_bfcl_dir() -> str:
    """Checkout containing the installed BFCL package data."""
    return os.environ.get("BENCH_BFCL_DIR") or str(
        Path.home() / "benchmarks" / "gorilla"
        / "berkeley-function-call-leaderboard")


def handler_key(arm: str) -> str:
    """BFCL model key = result-dir name, one per arm so runs never overwrite
    each other.  eval_runner.py:782 un-escapes the result dir with
    ``replace("_", "/")`` — underscores in arm names would corrupt the path,
    so the key uses dashes."""
    return f"c2kv-{(arm or 'full').replace('_', '-')}"


def generate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None,
                  num_threads: int = 1) -> List[str]:
    """``bfcl generate`` argv (PINNED; driven in-process by run_cli)."""
    if num_threads != 1:
        raise ValueError(
            "paper measurement requires BFCL num_threads=1 for single-flight telemetry")
    argv = [
        "generate", "--model", handler_name, "--test-category", categories,
        "--num-threads", "1",
    ]
    if run_ids:
        argv.append("--run-ids")
    return argv


def evaluate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None) -> List[str]:
    """``bfcl evaluate`` argv.  A subset EVALUATE needs --partial-eval, NOT
    --run-ids (this vintage scores the full category otherwise)."""
    argv = ["evaluate", "--model", handler_name, "--test-category", categories]
    if run_ids:
        argv.append("--partial-eval")
    return argv


def install_handler(base_url: str, model: str = SERVED_MODEL,
                    handler_name: "str | None" = None) -> None:
    # NOTE: default resolved at CALL time — binding the default to
    # MODEL_NAME at def time made monkeypatched names register the
    # wrong key (val20 evaluate failure)
    handler_name = handler_name or MODEL_NAME
    import httpx
    from openai import OpenAI
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    telemetry = HarnessTelemetry(
        HARNESS_TELEMETRY_PATH or Path("measurement/harness_events.jsonl"),
        "bfcl",
    )
    _install_timed_executor(telemetry)

    def normalize_native_calls(response):
        # Preserve native JSON tool blocks when the engine has no FC parser.
        # Malformed drafts remain model errors; never infer or repair arguments.
        from experiments.history_system.runtime.benchmarks.memory_runtime.event_native_draft import parse_native_draft
        for choice in response.choices:
            message = choice.message
            if message.tool_calls or not isinstance(message.content, str):
                continue
            draft = parse_native_draft(message.content, call_id_prefix=f"bfcl_native_{response.id}")
            if draft.status == "tool_calls":
                values = message.model_dump()
                values.update(tool_calls=list(draft.tool_calls), content=draft.content or None)
                choice.message = type(message).model_validate(values)
                choice.finish_reason = "tool_calls"
        return response

    class C2KVHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            return {
                "api_key": "EMPTY",
                "base_url": base_url,
                "timeout": httpx.Timeout(timeout=600.0, connect=8.0),
            }

        def _query_FC(self, inference_data: dict):
            kwargs = {
                "messages": inference_data["message"],
                "model": model,
                "temperature": self.temperature,
                "store": False,
                "max_completion_tokens": 4096,
            }
            if inference_data.get("tools"):
                kwargs["tools"] = inference_data["tools"]
            episode = current_episode()
            if episode:
                kwargs["extra_body"] = {
                    "c2kv_measurement_session_id": episode["episode_id"],
                }
            inference_data["inference_input_log"] = {
                "message": repr(inference_data["message"]),
                "tools": inference_data["tools"],
            }
            start_unix = time.time_ns()
            start_perf = time.perf_counter_ns()
            response = None
            try:
                response = self.client.chat.completions.create(**kwargs)
            except BaseException as exc:
                duration = time.perf_counter_ns() - start_perf
                telemetry.record_decision(
                    request_id=None, start_unix_ns=start_unix,
                    duration_ns=duration, response=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            duration = time.perf_counter_ns() - start_perf
            telemetry.record_decision(
                request_id=_proxy_request_id(response),
                start_unix_ns=start_unix, duration_ns=duration,
                response=_response_dict(response),
            )
            return normalize_native_calls(response), duration / 1e9

        def inference(self, test_entry: dict, include_input_log: bool,
                      exclude_state_log: bool):
            episode_id = str(test_entry.get("id") or "unknown")
            with telemetry.episode(
                episode_id,
                metadata={"category": episode_id.rsplit("_", 1)[0]},
            ):
                return super().inference(
                    test_entry, include_input_log, exclude_state_log)

    MODEL_CONFIG_MAPPING[handler_name] = ModelConfig(
        model_name=model,
        display_name="c2kv-hf",
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def official_category_ids(category: str) -> Dict[str, List[str]]:
    """Resolve a BFCL category/collection with the pinned official loader.

    A collection such as ``multi_turn`` is not a data filename. Memory and
    web-search concrete categories also share source files and are rewritten
    by the official loader, so counting ``BFCL_v4_<argument>.json`` is not a
    general denominator rule.
    """
    try:
        from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument
    except ImportError as error:
        raise RuntimeError(
            "BFCL expected-count resolution requires the pinned bfcl_eval package"
        ) from error

    concrete = parse_test_category_argument([category])
    if not concrete:
        raise ValueError(f"BFCL category resolves to no concrete categories: {category}")
    if "format_sensitivity" in concrete:
        raise ValueError(
            "BFCL format_sensitivity is non-scoring and cannot produce a scored summary"
        )

    selected: Dict[str, List[str]] = {}
    for name in concrete:
        entries = load_dataset_entry(
            name, include_prereq=False, include_language_specific_hint=False)
        raw_ids = [entry.get("id") if isinstance(entry, dict) else None
                   for entry in entries]
        if not raw_ids or any(item is None for item in raw_ids):
            raise RuntimeError(f"BFCL category has no complete official id set: {name}")
        ids = [str(item) for item in raw_ids]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"BFCL category has duplicate official ids: {name}")
        selected[name] = ids
    return selected


def official_category_counts(category: str) -> Dict[str, int]:
    return {name: len(ids) for name, ids in official_category_ids(category).items()}


def expected_count(category: str) -> int:
    """Number of scored tasks after official collection expansion."""
    return sum(official_category_counts(category).values())


def _score_header(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            header = next((json.loads(line) for line in handle if line.strip()), None)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid BFCL score file {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"BFCL score file has no JSON header: {path}")

    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if (not isinstance(total, int) or isinstance(total, bool) or total <= 0
            or not isinstance(correct, int) or isinstance(correct, bool)
            or correct < 0 or correct > total
            or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool)
            or not math.isfinite(float(accuracy))):
        raise RuntimeError(f"invalid BFCL score header in {path}: {header}")
    derived = correct / total
    if not math.isclose(float(accuracy), derived, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"BFCL score header accuracy mismatch in {path}: "
            f"accuracy={accuracy} correct_count={correct} total_count={total}")
    return header


def collect_score_summary(project_root: "Path | str", handler_name: str,
                          selected_counts: Mapping[str, int]) -> Dict[str, Any]:
    """Read and validate official per-category score headers without inference."""
    score_root = Path(project_root) / "score" / handler_name
    if not selected_counts:
        raise ValueError("BFCL selected category counts are empty")

    headers = []
    total = 0
    correct = 0
    for category, expected in selected_counts.items():
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"invalid selected count for BFCL/{category}: {expected!r}")
        filename = f"BFCL_v4_{category}_score.json"
        hits = sorted(score_root.rglob(filename)) if score_root.is_dir() else []
        if len(hits) != 1:
            raise RuntimeError(
                f"expected one BFCL score file for {category} under {score_root}, "
                f"found {len(hits)}")
        header = _score_header(hits[0])
        if header["total_count"] != expected:
            raise RuntimeError(
                f"BFCL scorer denominator mismatch for {category}: "
                f"total_count={header['total_count']} selected_expected={expected}")
        total += header["total_count"]
        correct += header["correct_count"]
        headers.append({
            "category": category,
            "path": str(hits[0]),
            "accuracy": float(header["accuracy"]),
            "correct_count": header["correct_count"],
            "total_count": header["total_count"],
        })

    selected_expected = sum(selected_counts.values())
    if total != selected_expected:
        raise RuntimeError(
            f"BFCL scorer coverage mismatch: n_scored={total} n={selected_expected}")
    return {
        "n": selected_expected,
        "n_total": selected_expected,
        "n_scored": total,
        "correct_count": correct,
        "semantic_score": correct / total,
        "official_score_headers": headers,
        "scored": True,
    }


def _result_path(project_root: Path, handler_name: str,
                 category: str) -> Optional[Path]:
    root = project_root / "result" / handler_name
    hits = sorted(root.rglob(f"BFCL_v4_{category}_result.json")) if root.is_dir() else []
    if len(hits) > 1:
        raise RuntimeError(
            f"expected at most one BFCL result file for {category} under {root}, "
            f"found {len(hits)}")
    return hits[0] if hits else None


def _completion_ledger(project_root: Path, handler_name: str,
                       selected_ids: Mapping[str, List[str]],
                       prior: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Canonical valid completions across the active official result files.

    Recognized terminal context/method failures remain official scored rows.
    Missing outputs and unclassified transport/engine failures remain incomplete.
    """
    entries: Dict[str, List[Dict[str, Any]]] = {
        task_id: [] for ids in selected_ids.values() for task_id in ids}
    category_of = {
        task_id: category for category, ids in selected_ids.items()
        for task_id in ids}
    if len(category_of) != sum(len(ids) for ids in selected_ids.values()):
        raise ValueError("BFCL concrete categories contain overlapping task IDs")
    if prior:
        for task_id, row in prior["canonical"].items():
            entries[task_id].append(dict(row))
    foreign = []
    invalid_rows = 0
    paths: Dict[str, Path] = dict(prior["paths"]) if prior else {}
    for category in selected_ids:
        path = _result_path(project_root, handler_name, category)
        if path is None:
            continue
        paths[category] = path
        mtime_ns = path.stat().st_mtime_ns
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(row, dict) or row.get("id") is None:
                invalid_rows += 1
                continue
            task_id = str(row["id"])
            if category_of.get(task_id) != category:
                foreign.append({"task_id": task_id, "category": category,
                                "path": str(path), "line": line_number})
                continue
            entries[task_id].append({
                "task_id": task_id,
                "category": category,
                "valid": bfcl_row_is_terminal(row),
                "path": str(path),
                "line": line_number,
                "mtime_ns": mtime_ns,
                "row": row,
            })
            if not bfcl_row_is_terminal(row):
                invalid_rows += 1
    if foreign:
        shown = ", ".join(item["task_id"] for item in foreign[:20])
        raise RuntimeError(f"BFCL result contains unrequested task IDs: {shown}")
    canonical = {}
    for task_id, rows in entries.items():
        valid = [row for row in rows if row["valid"]]
        if valid:
            canonical[task_id] = max(
                valid, key=lambda row: (row["mtime_ns"], row["path"], row["line"]))
    requested = [task_id for ids in selected_ids.values() for task_id in ids]
    remaining = [task_id for task_id in requested if task_id not in canonical]
    return {
        "requested": requested,
        "valid_unique": list(canonical),
        "remaining": remaining,
        "duplicate_rows": sum(max(0, len(rows) - 1) for rows in entries.values()),
        "invalid_rows": invalid_rows,
        "canonical": canonical,
        "paths": paths,
        "terminal_failures": {task_id: terminal_failure_kind(item["row"])
                              for task_id, item in canonical.items()
                              if terminal_failure_kind(item["row"])},
    }


def _snapshot_completion_round(project_root: Path, invocation_root: Path,
                               round_index: int,
                               ledger: Mapping[str, Any]) -> Path:
    round_root = invocation_root / f"round-{round_index:03d}"
    raw_root = round_root / "raw"
    for path in ledger["paths"].values():
        if not path.exists():
            continue
        relative = path.relative_to(project_root)
        destination = raw_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    receipt = {
        "schema": "c2kv.bfcl.completion-round.v1",
        "round": round_index,
        "requested": ledger["requested"],
        "valid_unique": ledger["valid_unique"],
        "remaining": ledger["remaining"],
        "duplicate_rows": ledger["duplicate_rows"],
        "invalid_rows": ledger["invalid_rows"],
        "terminal_failures": ledger["terminal_failures"],
    }
    receipt_path = round_root / "ledger.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def _write_canonical_results(selected_ids: Mapping[str, List[str]],
                             ledger: Mapping[str, Any]) -> None:
    """Expose exactly one valid row per completed task to the official scorer."""
    canonical = ledger["canonical"]
    for category, task_ids in selected_ids.items():
        path = ledger["paths"].get(category)
        if path is None:
            continue
        rows = [canonical[task_id]["row"] for task_id in task_ids
                if task_id in canonical]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
        temporary.replace(path)


def _write_selected_ids(project_root: Path,
                        selected_ids: Mapping[str, List[str]]) -> None:
    id_file = project_root / "test_case_ids_to_generate.json"
    temporary = id_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selected_ids), encoding="utf-8")
    temporary.replace(id_file)


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: run the official CLI from inside the gorilla checkout.

    BFCL reads ``BFCL_PROJECT_ROOT`` when its modules are first imported.
    Point it at this run's output directory before ``run_bfcl`` imports BFCL,
    while keeping cwd in the checkout so the installed package data resolves.
    The handler expects an OpenAI base_url WITH ``/v1``.

    No cost join: see ``COST_JOIN`` below.
    """
    project_root = ctx.out_dir.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.chdir(ctx.opt("bfcl_dir") or default_bfcl_dir())
    try:
        summary = run_bfcl(
            v1(ctx.base_url),
            categories=ctx.opt("categories", "multi_turn_base"),
            mode=ctx.opt("mode", "both"),
            run_ids=ctx.opt("run_ids"),
            model=ctx.model,
            handler_name=handler_key(ctx.arm),
            project_root=project_root,
            num_threads=int(ctx.opt("num_workers", 1)),
            max_refill_rounds=int(ctx.opt("bfcl_refill_rounds", 0)),
        )
    finally:
        os.chdir(prev_cwd)
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root
    summary["cost_join"] = COST_JOIN
    return summary


# Why BFCL gets no per-task cost columns.  ``proxy.conversation_id`` shifts
# once per entry: request 1 carries only question[0] (the data file's turn-0
# user message), every later request also carries the FIRST ASSISTANT
# MESSAGE (proxy.py:434-447).  That assistant message is the raw OpenAI
# message object appended by _add_assistant_message_FC; the result file
# stores only the decoded ``model_responses`` (base_handler.py:243-253), and
# the verbatim wire payload is written only under ``--include-input-log``
# (base_handler.py:219-225) — a flag the pinned generate argv does not pass.
# Keying on the first id alone would attribute one request per entry, which
# is a wrong cost column rather than a missing one.
COST_JOIN = ("not joinable: the steady-state conversation id needs the first "
             "assistant message verbatim, which the BFCL result file does not "
             "store (see adapters/bfcl_adapter.py)")


def run_bfcl(base_url: str, categories: str = "multi_turn_base",
             mode: str = "both", run_ids: "list[str] | str | None" = None,
             model: str = SERVED_MODEL,
             handler_name: str = MODEL_NAME,
             project_root: "Path | str | None" = None,
             num_threads: int = 1,
             max_refill_rounds: int = 0) -> Dict[str, Any]:
    """Register the handler and drive the official generate/evaluate CLI
    in-process.

    ``handler_name`` is the BFCL model key (= result-dir name); ``run``
    passes ``handler_key(arm)`` so arms never overwrite each other's results.

    ``run_ids`` subsets the category: this BFCL vintage implements subsetting
    through <project-root>/test_case_ids_to_generate.json ({"<category>":
    [ids]}) + the boolean --run-ids flag, NOT a CLI value, so the ids are
    written to that file and the flag is passed (comma string or list).

    ``project_root`` is the isolated output root used by BFCL's official
    RESULT_PATH, SCORE_PATH and TEST_IDS_TO_GENERATE_PATH.  It must be set
    before the first official import.

    Terminal-state check (acceptance 1): every selected entry must have a
    result row and, after evaluation, an official score row. Generation-only
    summaries deliberately contain no ``n_scored`` or ``semantic_score``."""
    if mode not in ("generate", "evaluate", "both"):
        raise ValueError(f"invalid BFCL mode: {mode}")
    if (not isinstance(max_refill_rounds, int)
            or isinstance(max_refill_rounds, bool) or max_refill_rounds < 0):
        raise ValueError("BFCL max_refill_rounds must be a non-negative integer")
    global HARNESS_TELEMETRY_PATH
    project_root = Path(
        project_root or os.environ.get("BFCL_PROJECT_ROOT") or Path.cwd()
    ).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    HARNESS_TELEMETRY_PATH = project_root / "measurement" / "harness_events.jsonl"
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    try:
        install_handler(base_url, model=model, handler_name=handler_name)
        category_ids = official_category_ids(categories)
        ids: Optional[List[str]] = None
        if run_ids:
            raw_ids = ([i.strip() for i in run_ids.split(",") if i.strip()]
                       if isinstance(run_ids, str) else [str(i).strip() for i in run_ids])
            # BFCL's boolean --run-ids flag reads a JSON list.  Normalize it
            # once here so repeated CLI values cannot inflate the denominator.
            ids = list(dict.fromkeys(item for item in raw_ids if item))
            if len(category_ids) != 1:
                raise ValueError(
                    "BFCL --run-ids requires one concrete category, not collection "
                    f"{categories!r} -> {sorted(category_ids)}")
            if not ids:
                raise ValueError("BFCL --run-ids resolved to an empty id list")
            category = next(iter(category_ids))
            unknown = [task_id for task_id in ids
                       if task_id not in set(category_ids[category])]
            if unknown:
                raise ValueError(
                    "BFCL --run-ids contains IDs outside the official category: "
                    + ",".join(unknown[:20]))
            selected_ids = {category: ids}
            _write_selected_ids(project_root, selected_ids)
        else:
            selected_ids = category_ids
        selected_counts = {name: len(values) for name, values in selected_ids.items()}
        expected = sum(selected_counts.values())
        completion = None
        refill_receipts = []
        invocation_root = (project_root / "measurement" / "bfcl_completion" /
                           f"{time.time_ns()}-{os.getpid()}")
        invocation_root.mkdir(parents=True, exist_ok=False)
        if mode in ("generate", "both"):
            remaining = selected_ids
            for round_index in range(max_refill_rounds + 1):
                if round_index:
                    _write_selected_ids(project_root, remaining)
                use_subset = ids is not None or round_index > 0
                run_cli(generate_argv(
                    handler_name, categories,
                    [task for values in remaining.values() for task in values]
                    if use_subset else None,
                    num_threads=num_threads))
                completion = _completion_ledger(
                    project_root, handler_name, selected_ids, prior=completion)
                refill_receipts.append(str(_snapshot_completion_round(
                    project_root, invocation_root, round_index, completion)))
                _write_canonical_results(selected_ids, completion)
                if not completion["remaining"]:
                    break
                if round_index == max_refill_rounds:
                    raise RuntimeError(
                        "BFCL generation lacks valid unique completions after "
                        f"{round_index + 1} bounded round(s): "
                        + ",".join(completion["remaining"][:20]))
                remaining = {
                    category: [task_id for task_id in task_ids
                               if task_id in set(completion["remaining"])]
                    for category, task_ids in selected_ids.items()
                }
                remaining = {category: task_ids for category, task_ids in remaining.items()
                             if task_ids}
        else:
            completion = _completion_ledger(project_root, handler_name, selected_ids)
            refill_receipts.append(str(_snapshot_completion_round(
                project_root, invocation_root, 0, completion)))
            if completion["remaining"]:
                raise RuntimeError(
                    "BFCL evaluation requires valid unique completions for: "
                    + ",".join(completion["remaining"][:20]))
            _write_canonical_results(selected_ids, completion)
        # A refill temporarily narrows BFCL's global subset file.  Restore the
        # original selection before evaluation so the official denominator
        # cannot silently collapse to the final gap.
        _write_selected_ids(project_root, selected_ids)
        if mode in ("evaluate", "both"):
            run_cli(evaluate_argv(handler_name, categories, ids))
        import terminal_check  # noqa: E402  (sibling module, sys.path has parent)

        for category, selected in selected_counts.items():
            ids_str = ",".join(selected_ids[category])
            code = terminal_check.check_bfcl(
                selected, ids_str, handler=handler_name,
                category=category, root=project_root)
            if code != 0:
                raise SystemExit(
                    f"FATAL: bfcl terminal-state check failed for {category} (rc={code})")

        summary: Dict[str, Any] = {
            "benchmark": "bfcl", "categories": categories, "mode": mode,
            "n_total": expected,
            "bfcl_project_root": str(project_root),
            "harness_telemetry": str(HARNESS_TELEMETRY_PATH),
            "num_threads": num_threads,
            "completion_ledger": {
                "valid_unique": len(completion["valid_unique"]),
                "duplicate_rows": completion["duplicate_rows"],
                "invalid_rows": completion["invalid_rows"],
                "remaining": completion["remaining"],
                "terminal_failures": completion["terminal_failures"],
                "max_refill_rounds": max_refill_rounds,
                "refill_rounds_used": max(0, len(refill_receipts) - 1),
                "round_receipts": refill_receipts,
            },
        }
        if mode == "generate":
            summary.update({"n_generated": expected, "scored": False})
        else:
            summary.update(collect_score_summary(
                project_root, handler_name, selected_counts))
            if mode == "both":
                summary["n_generated"] = expected
        return summary
    finally:
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root


def run_cli(argv):
    from bfcl_eval.__main__ import cli

    # Typer's standalone mode raises SystemExit(0) after generate, which
    # would skip evaluate, terminal checks and the unified run summary.
    cli(args=argv, prog_name="bfcl", standalone_mode=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:34000/v1")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--model", default=SERVED_MODEL,
                        help="served model name at the endpoint")
    parser.add_argument("--handler-name", default=MODEL_NAME,
                        help="BFCL model key / result-dir name")
    parser.add_argument("--bfcl-refill-rounds", type=int, default=0)
    args = parser.parse_args(argv)
    # one code path: the CLI used to re-implement run() and had drifted
    # (evaluate got --run-ids instead of --partial-eval, no terminal gate)
    ids = [i.strip() for i in args.run_ids.split(",") if i.strip()] or None
    summary = run_bfcl(args.base_url, categories=args.categories, mode=args.mode,
                       run_ids=ids, model=args.model,
                       handler_name=args.handler_name,
                       max_refill_rounds=args.bfcl_refill_rounds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
