"""Build decision-local BFCL labels from returned shadow task shards."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LABEL_SCHEMA = "a-history-r002-decision-label-v1"
EXACT_LABEL_KIND = "single_decision_official_prefix_error"
EXACT_LABEL_TARGET = "current_decision_official_prefix_outcome"
SEGMENT_LABEL_KIND = "single_action_then_stop_prefix_error"
SEGMENT_LABEL_TARGET = "short_segment_outcome_from_action_prefill"
LABEL_SOURCE = (
    "runtime/benchmarks/bfcl_gold_recovery.py::official_prefix_check"
)
_DECISION_RE = re.compile(r"^turn-(\d+)/step-(\d+)$")
_DECODE_ERROR_PREFIX = "Error decoding the model response."
CHECKPOINT_HASH_KIND = "config_json"
SUPPORTED_RATIO = 8


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = value if isinstance(value, list) else [value]
    if any(not isinstance(item, dict) for item in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def load_data_roles(path: Path) -> dict[str, dict[str, Any]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "a-history-r002-detector-data-plan-v1":
        raise ValueError("unsupported r002 data-plan schema")
    roles: dict[str, dict[str, Any]] = {}
    for split, key in (("train", "train_groups"), ("calibration", "calibration_groups")):
        groups = plan.get(key)
        if not isinstance(groups, list):
            raise ValueError(f"data plan {key} must be a list")
        for group in groups:
            if not isinstance(group, Mapping) or type(group.get("row_ordinal")) is not int:
                raise ValueError(f"data plan {key} contains an invalid group")
            task_ids = group.get("task_ids")
            if not isinstance(task_ids, list) or not task_ids:
                raise ValueError(f"data plan {key} contains invalid task_ids")
            for task_id in task_ids:
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(f"data plan {key} contains an invalid task id")
                if task_id in roles:
                    raise ValueError(f"data plan repeats task id {task_id}")
                roles[task_id] = {
                    "split": split,
                    "group_ordinal": group["row_ordinal"],
                }
    return roles


def load_checkpoint_binding(
    value: Path | Mapping[str, Any], *, ratio: int = SUPPORTED_RATIO
) -> dict[str, Any]:
    """Validate runtime identity without conflating a config digest with weights."""

    if isinstance(value, Path):
        source_bytes = value.read_bytes()
        raw = json.loads(source_bytes.decode("utf-8"))
        source_path = str(value)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    else:
        raw = value
        source_path = None
        source_sha256 = None
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint binding must be a JSON object")
    required = {"status", "path", "config_sha256", "selected_arm", "selected_step"}
    allowed = required | {
        "ratio",
        "selection_basis",
        "historical_ratio8",
        "historical_ratio4",
        "sample_label",
        "c500",
        "new_comparison_outcomes_used",
    }
    if not required.issubset(raw) or set(raw) - allowed:
        raise ValueError(
            "checkpoint binding requires status, path, config_sha256, selected_arm, "
            "and selected_step, with only recognized selection provenance fields"
        )
    if raw["status"] not in {"candidate_for_selection", "selected"}:
        raise ValueError("checkpoint binding status must be candidate_for_selection or selected")
    if not isinstance(raw["path"], str) or not raw["path"]:
        raise ValueError("checkpoint binding path must be nonempty")
    digest = raw["config_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint binding config_sha256 must be a SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("checkpoint binding config_sha256 must be hexadecimal") from error
    if not isinstance(raw["selected_arm"], str) or not raw["selected_arm"]:
        raise ValueError("checkpoint binding selected_arm must be nonempty")
    if type(raw["selected_step"]) is not int or raw["selected_step"] < 0:
        raise ValueError("checkpoint binding selected_step must be a nonnegative integer")
    if ratio != SUPPORTED_RATIO or type(ratio) is not int:
        raise ValueError(f"this r002 detector slice requires ratio={SUPPORTED_RATIO}")
    if "ratio" in raw and raw["ratio"] != ratio:
        raise ValueError("checkpoint binding ratio differs from requested ratio")
    return {
        "status": raw["status"],
        "checkpoint_binding_status": raw["status"],
        "checkpoint_path": raw["path"],
        "checkpoint_hash": digest,
        "checkpoint_hash_kind": CHECKPOINT_HASH_KIND,
        "checkpoint_config_sha256": digest,
        "checkpoint_selected_arm": raw["selected_arm"],
        "checkpoint_selected_step": raw["selected_step"],
        "ratio": ratio,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }


def _discover_task_shards(
    returned_roots: Sequence[Path],
    roles: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Path], list[str]]:
    found: dict[str, Path] = {}
    ignored: list[str] = []
    for root in returned_roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for steps_path in root.rglob("steps.jsonl"):
            if steps_path.parent.name != "server":
                continue
            task_dir = steps_path.parent.parent
            task_id = task_dir.name
            if task_id not in roles:
                ignored.append(task_id)
                continue
            if task_id in found and found[task_id] != task_dir:
                raise ValueError(f"duplicate returned task shard for {task_id}")
            found[task_id] = task_dir
    return found, sorted(set(ignored))


def _result_path(task_dir: Path) -> Path | None:
    candidates = [
        path
        for path in task_dir.rglob("BFCL_v4_multi_turn*_result.json")
        if "result" in path.parts
    ]
    if len(candidates) > 1:
        raise ValueError(f"multiple official result files under {task_dir}")
    return candidates[0] if candidates else None


def _task_result(path: Path, task_id: str) -> dict[str, Any]:
    matches = [row for row in _read_records(path) if str(row.get("id")) == task_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one official result row for {task_id}, found {len(matches)}"
        )
    return matches[0]


def _step_batch(
    step: Any,
) -> tuple[list[str] | None, str | None, str | None, dict[str, Any] | None]:
    if not isinstance(step, list):
        return None, "official_step_not_list", None, None
    assistants = [
        item for item in step
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    if len(assistants) != 1:
        return None, "official_step_assistant_count", None, None
    handlers = [
        item for item in step
        if isinstance(item, Mapping) and item.get("role") == "handler_log"
    ]
    if len(handlers) != 1:
        return None, "official_step_handler_log_count", None, None
    handler = handlers[0]
    decoded = handler.get("model_response_decoded")
    if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
        kind = "decoded_nonempty" if decoded else "decoded_empty"
        return list(decoded), None, kind, dict(handler)
    content = handler.get("content")
    if isinstance(content, str) and content.startswith(_DECODE_ERROR_PREFIX):
        # BFCL's own controller feeds an empty decoded batch to the checker.
        return [], None, "text_decode_to_empty", dict(handler)
    return None, "official_decoded_batch_unavailable", None, dict(handler)


def decoded_turns_from_inference_log(
    inference_log: Any,
) -> list[dict[str, Any]]:
    if not isinstance(inference_log, list):
        return []
    turns = []
    for item in inference_log:
        if not isinstance(item, Mapping):
            continue
        step_items = []
        for key, value in item.items():
            if not isinstance(key, str) or not key.startswith("step_"):
                continue
            suffix = key.removeprefix("step_")
            if not suffix.isdigit():
                return []
            step_items.append((int(suffix), value))
        if not step_items:
            continue
        step_items.sort(key=lambda pair: pair[0])
        if [index for index, _ in step_items] != list(range(len(step_items))):
            turns.append({"complete": False, "reason": "official_steps_not_contiguous"})
            continue
        batches = []
        batch_kinds = []
        handler_logs = []
        reason = None
        for _index, step in step_items:
            batch, reason, kind, handler = _step_batch(step)
            if reason is not None:
                break
            batches.append(batch)
            batch_kinds.append(kind)
            handler_logs.append(handler)
        turns.append(
            {
                "complete": reason is None,
                "reason": reason,
                "decoded_batches": batches if reason is None else None,
                "batch_kinds": batch_kinds if reason is None else None,
                "handler_logs": handler_logs if reason is None else None,
            }
        )
    return turns


def _default_task_loader(task_id: str):
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    category = task_id.rsplit("_", 1)[0]
    tests = [
        row for row in load_dataset_entry(category)
        if isinstance(row, dict) and str(row.get("id")) == task_id
    ]
    answers = [
        row for row in load_ground_truth_entry(category)
        if isinstance(row, dict) and str(row.get("id")) == task_id
    ]
    if len(tests) != 1 or len(answers) != 1:
        raise RuntimeError(
            f"official BFCL rows for {task_id}: tests={len(tests)}, answers={len(answers)}"
        )
    ground_truth = answers[0].get("ground_truth")
    if not isinstance(ground_truth, list) or any(
        not isinstance(turn, list) for turn in ground_truth
    ):
        raise RuntimeError(f"invalid official BFCL ground truth for {task_id}")
    return tests[0], ground_truth


def _default_prefix_checker(decoded_turns, ground_truth, test_entry):
    from experiments.history_system.runtime.benchmarks.bfcl_gold_recovery import (
        official_prefix_check,
    )

    return official_prefix_check(decoded_turns, ground_truth, test_entry)


def _feature_fields(step: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "shadow_schema": None,
        "prefill_layer": None,
        "prefill_stored_dtype": None,
        "prefill_hidden": None,
        "prefill_feature_status": "unavailable",
        "first_name_top2_logprob_margin": None,
        "margin_feature_status": "unavailable",
        "model_binding": None,
        "tokenizer_binding": None,
    }
    generations = step.get("generation_trace")
    if not isinstance(generations, list):
        return result
    drafts = [
        item for item in generations
        if isinstance(item, Mapping)
        and item.get("phase") == "draft"
        and item.get("status") == "completed"
    ]
    if len(drafts) != 1:
        return result
    generation = drafts[0].get("generation")
    stats = generation.get("stats") if isinstance(generation, Mapping) else None
    shadow = stats.get("shadow_features") if isinstance(stats, Mapping) else None
    if not isinstance(shadow, Mapping):
        return result
    result["shadow_schema"] = shadow.get("schema")
    bindings = shadow.get("bindings")
    if isinstance(bindings, Mapping):
        result["model_binding"] = bindings.get("model")
        result["tokenizer_binding"] = bindings.get("tokenizer")
    prefill = shadow.get("prefill")
    if isinstance(prefill, Mapping):
        result["prefill_layer"] = prefill.get("layer")
        result["prefill_stored_dtype"] = prefill.get("stored_dtype")
        hidden = prefill.get("hidden")
        position = prefill.get("position")
        if (
            shadow.get("schema") == "event-native-shadow-features-v1"
            and prefill.get("status") == "captured"
            and isinstance(position, Mapping)
            and position.get("kind") == "prompt_last"
            and prefill.get("readout") == "decoder_layer_output"
            and isinstance(hidden, list)
            and hidden
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in hidden
            )
        ):
            result["prefill_hidden"] = [float(item) for item in hidden]
            result["prefill_feature_status"] = "available"
    tool_name = shadow.get("tool_name")
    signals = shadow.get("signals")
    if (
        isinstance(tool_name, Mapping)
        and tool_name.get("status") == "located"
        and isinstance(signals, Mapping)
    ):
        margin = signals.get("first_name_top2_logprob_margin")
        if (
            not isinstance(margin, bool)
            and isinstance(margin, (int, float))
            and math.isfinite(float(margin))
            and margin >= 0
        ):
            result["first_name_top2_logprob_margin"] = float(margin)
            result["margin_feature_status"] = "available"
    return result


def _base_row(
    step: Mapping[str, Any],
    *,
    task_id: str,
    role: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    decision_key = step.get("decision_key")
    match = _DECISION_RE.fullmatch(decision_key) if isinstance(decision_key, str) else None
    turn = int(match.group(1)) if match else None
    decision_step = int(match.group(2)) if match else None
    generations = step.get("generation_trace")
    draft = generations[0] if isinstance(generations, list) and generations else None
    prepared = draft.get("prepared_input") if isinstance(draft, Mapping) else None
    return {
        "schema": LABEL_SCHEMA,
        "task_id": task_id,
        "split": role["split"],
        "group_ordinal": role["group_ordinal"],
        "decision_key": decision_key,
        "turn_index": turn,
        "step_index": decision_step,
        "prefix_hash": None,
        "visible_input_hash": _sha256_json(prepared) if prepared is not None else None,
        "checkpoint_hash": checkpoint_binding["checkpoint_hash"],
        "checkpoint_hash_kind": checkpoint_binding["checkpoint_hash_kind"],
        "checkpoint_config_sha256": checkpoint_binding["checkpoint_config_sha256"],
        "checkpoint_binding_status": checkpoint_binding["checkpoint_binding_status"],
        "checkpoint_path": checkpoint_binding["checkpoint_path"],
        "checkpoint_selected_arm": checkpoint_binding["checkpoint_selected_arm"],
        "checkpoint_selected_step": checkpoint_binding["checkpoint_selected_step"],
        "ratio": checkpoint_binding["ratio"],
        "label": None,
        "label_kind": None,
        "label_target": None,
        "label_source": LABEL_SOURCE,
        "segment_stop_bridge": None,
        "adjudication_reason": "not_adjudicated",
        **_feature_fields(step),
    }


def _checker_valid(
    checker: Callable[..., Mapping[str, Any]],
    decoded: Sequence[Sequence[Sequence[str]]],
    ground_truth: Sequence[Sequence[str]],
    test_entry: Mapping[str, Any],
) -> tuple[bool | None, str | None]:
    try:
        outcome = checker(decoded, ground_truth, test_entry)
    except Exception as error:
        return None, f"official_prefix_check_error:{type(error).__name__}"
    if not isinstance(outcome, Mapping) or type(outcome.get("valid")) is not bool:
        return None, "official_prefix_check_invalid_output"
    return outcome["valid"], None


def _decision_record_complete(step: Mapping[str, Any]) -> bool:
    generations = step.get("generation_trace")
    return bool(
        step.get("status") == "ok"
        and isinstance(generations, list)
        and len(generations) == 1
        and isinstance(generations[0], Mapping)
        and generations[0].get("phase") == "draft"
        and generations[0].get("status") == "completed"
        and isinstance(step.get("response"), Mapping)
    )


def _segment_shape(
    *,
    turn_indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    official_result_sha256: str,
) -> tuple[int | None, int | None, dict[str, Any] | None, str | None]:
    """Locate one action followed by one verified text-stop bridge."""

    if len(turn_indices) != 2:
        return None, None, None, "current_turn_multiple_model_decisions"
    ordered = sorted(turn_indices, key=lambda index: rows[index]["step_index"])
    if [rows[index]["step_index"] for index in ordered] != [0, 1]:
        return None, None, None, "segment_server_steps_not_exactly_zero_one"
    batches = current.get("decoded_batches")
    kinds = current.get("batch_kinds")
    handlers = current.get("handler_logs")
    if not (
        isinstance(batches, list)
        and isinstance(kinds, list)
        and isinstance(handlers, list)
        and len(batches) == len(kinds) == len(handlers) == 2
    ):
        return None, None, None, "segment_official_steps_not_exactly_two"
    if kinds != ["decoded_nonempty", "text_decode_to_empty"]:
        return None, None, None, "segment_official_action_stop_shape_mismatch"
    if not batches[0] or batches[1] != []:
        return None, None, None, "segment_official_action_stop_shape_mismatch"
    action_index, stop_index = ordered
    action_step, stop_step = steps[action_index], steps[stop_index]
    if not _decision_record_complete(action_step) or not _decision_record_complete(stop_step):
        return None, None, None, "segment_decision_record_incomplete"
    action_response = action_step["response"]
    stop_response = stop_step["response"]
    if not (
        action_response.get("native_parse_status") == "tool_calls"
        and isinstance(action_response.get("tool_calls"), list)
        and action_response["tool_calls"]
        and action_response.get("finish_reason") == "stop"
    ):
        return None, None, None, "segment_server_action_parse_mismatch"
    if not (
        stop_response.get("native_parse_status") == "text"
        and stop_response.get("tool_calls") == []
        and isinstance(stop_response.get("content"), str)
        and stop_response.get("finish_reason") == "stop"
    ):
        return None, None, None, "segment_server_stop_parse_mismatch"
    handler = handlers[1]
    bridge = {
        "schema": "bfcl-normal-text-stop-to-empty-bridge-v1",
        "server_response": {
            "status": stop_step.get("status"),
            "native_parse_status": stop_response.get("native_parse_status"),
            "tool_calls": stop_response.get("tool_calls"),
            "finish_reason": stop_response.get("finish_reason"),
        },
        "official_handler_log": {
            "content": handler.get("content"),
            "error": handler.get("error"),
            "model_response_decoded_present": "model_response_decoded" in handler,
        },
        "official_handler_log_sha256": _sha256_json(handler),
        "server_stop_step_sha256": _sha256_json(stop_step),
        "official_result_sha256": official_result_sha256,
    }
    return action_index, stop_index, bridge, None


def collect_decision_rows(
    *,
    returned_roots: Sequence[Path],
    data_plan_path: Path,
    checkpoint_binding: Path | Mapping[str, Any],
    ratio: int = SUPPORTED_RATIO,
    task_loader: Callable[[str], tuple[Mapping[str, Any], Sequence[Sequence[str]]]] | None = None,
    prefix_checker: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect labels and features without copying terminal outcomes to steps."""

    binding = load_checkpoint_binding(checkpoint_binding, ratio=ratio)
    roles = load_data_roles(data_plan_path)
    task_dirs, ignored = _discover_task_shards(returned_roots, roles)
    task_loader = task_loader or _default_task_loader
    prefix_checker = prefix_checker or _default_prefix_checker
    rows: list[dict[str, Any]] = []

    for task_id in sorted(
        task_dirs,
        key=lambda item: (roles[item]["split"], roles[item]["group_ordinal"], item),
    ):
        task_dir = task_dirs[task_id]
        steps_path = task_dir / "server" / "steps.jsonl"
        steps = _read_records(steps_path)
        server_steps_sha256 = hashlib.sha256(steps_path.read_bytes()).hexdigest()
        base_rows = [
            _base_row(
                step,
                task_id=task_id,
                role=roles[task_id],
                checkpoint_binding=binding,
            )
            for step in steps
        ]
        for row in base_rows:
            row["server_steps_sha256"] = server_steps_sha256
            row["official_result_sha256"] = None
        by_turn: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(base_rows):
            if row["turn_index"] is not None:
                by_turn[row["turn_index"]].append(index)

        result_path = _result_path(task_dir)
        if result_path is None:
            for row in base_rows:
                row["adjudication_reason"] = "official_result_missing"
            rows.extend(base_rows)
            continue
        try:
            official_result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
            task_result = _task_result(result_path, task_id)
            decoded_records = decoded_turns_from_inference_log(
                task_result.get("inference_log")
            )
            test_entry, ground_truth = task_loader(task_id)
        except Exception as error:
            reason = f"official_artifact_error:{type(error).__name__}"
            for row in base_rows:
                row["adjudication_reason"] = reason
            rows.extend(base_rows)
            continue
        for row in base_rows:
            row["official_result_sha256"] = official_result_sha256

        for index, row in enumerate(base_rows):
            turn = row["turn_index"]
            if turn is None or row["step_index"] is None:
                row["adjudication_reason"] = "decision_key_invalid"
                continue
            prior = decoded_records[:turn]
            if all(item.get("complete") for item in prior):
                decoded_prior = [item["decoded_batches"] for item in prior]
                row["prefix_hash"] = _sha256_json(decoded_prior)
            else:
                row["adjudication_reason"] = "previous_decoded_prefix_incomplete"
                continue
            if turn >= len(decoded_records):
                row["adjudication_reason"] = "current_decoded_turn_missing"
                continue
            current = decoded_records[turn]
            batches = current.get("decoded_batches") if current.get("complete") else None
            if not isinstance(batches, list):
                row["adjudication_reason"] = (
                    current.get("reason") or "current_decoded_turn_incomplete"
                )
                continue
            if turn >= len(ground_truth):
                row["adjudication_reason"] = "official_ground_truth_turn_missing"
                continue
            if turn > 0:
                previous_valid, reason = _checker_valid(
                    prefix_checker,
                    decoded_prior,
                    ground_truth[:turn],
                    test_entry,
                )
                if reason is not None:
                    row["adjudication_reason"] = reason
                    continue
                if not previous_valid:
                    row["adjudication_reason"] = "previous_official_prefix_invalid"
                    continue
            if len(by_turn[turn]) == 1 and row["step_index"] == 0:
                if len(batches) != 1:
                    row["adjudication_reason"] = "current_turn_decoded_batch_count"
                    continue
                if not _decision_record_complete(steps[index]):
                    row["adjudication_reason"] = "decision_record_incomplete"
                    continue
                label_kind = EXACT_LABEL_KIND
                label_target = EXACT_LABEL_TARGET
            else:
                action_index, stop_index, bridge, segment_reason = _segment_shape(
                    turn_indices=by_turn[turn],
                    rows=base_rows,
                    steps=steps,
                    current=current,
                    official_result_sha256=official_result_sha256,
                )
                if segment_reason is not None:
                    row["adjudication_reason"] = segment_reason
                    continue
                row["segment_stop_bridge"] = bridge
                if index == stop_index:
                    row["adjudication_reason"] = "single_action_then_stop_stop_decision_unknown"
                    continue
                if index != action_index:
                    row["adjudication_reason"] = "segment_action_decision_not_identified"
                    continue
                label_kind = SEGMENT_LABEL_KIND
                label_target = SEGMENT_LABEL_TARGET
            current_valid, reason = _checker_valid(
                prefix_checker,
                decoded_prior + [batches],
                ground_truth[: turn + 1],
                test_entry,
            )
            if reason is not None:
                row["adjudication_reason"] = reason
                continue
            row["label"] = 0 if current_valid else 1
            row["label_kind"] = label_kind
            row["label_target"] = label_target
            row["adjudication_reason"] = (
                f"{label_kind}:official_prefix_valid"
                if current_valid
                else f"{label_kind}:official_prefix_invalid"
            )
        rows.extend(base_rows)

    label_counts = Counter(
        "unknown" if row["label"] is None else str(row["label"])
        for row in rows
    )
    summary = {
        "schema": "a-history-r002-label-collection-summary-v1",
        "status": "completed",
        "data_plan": str(data_plan_path),
        "data_plan_sha256": hashlib.sha256(data_plan_path.read_bytes()).hexdigest(),
        "checkpoint_binding": {
            key: binding[key]
            for key in (
                "status",
                "checkpoint_binding_status",
                "checkpoint_path",
                "checkpoint_hash",
                "checkpoint_hash_kind",
                "checkpoint_config_sha256",
                "checkpoint_selected_arm",
                "checkpoint_selected_step",
                "ratio",
            )
        },
        "checkpoint_binding_source": {
            "path": binding["source_path"],
            "sha256": binding["source_sha256"],
        },
        "expected_tasks": len(roles),
        "returned_tasks": len(task_dirs),
        "missing_tasks": sorted(set(roles) - set(task_dirs)),
        "ignored_tasks": ignored,
        "decision_rows": len(rows),
        "labels": dict(sorted(label_counts.items())),
        "known_label_kinds": dict(
            sorted(Counter(
                row["label_kind"] for row in rows if row["label"] in {0, 1}
            ).items())
        ),
        "known_label_targets": dict(
            sorted(Counter(
                row["label_target"] for row in rows if row["label"] in {0, 1}
            ).items())
        ),
        "prefill_available": sum(
            row["prefill_feature_status"] == "available" for row in rows
        ),
        "margin_available": sum(
            row["margin_feature_status"] == "available" for row in rows
        ),
        "adjudication_reasons": dict(
            sorted(Counter(row["adjudication_reason"] for row in rows).items())
        ),
        "terminal_task_outcome_used_as_step_label": False,
    }
    return rows, summary


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def add_bfcl_root(path: Path | None) -> None:
    if path is not None:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
