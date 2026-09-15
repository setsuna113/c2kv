from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.history_system.detectors.labels import (
    collect_decision_rows,
    load_checkpoint_binding,
)
from experiments.history_system.detectors.prefill_head import (
    calibrate_exact_fraction,
    fit_prefill_bundle,
    load_prefill_head_bundle,
    recalibrate_prefill_bundle,
)


CONFIG_SHA = "15e14bfa5853ce7e74ef3cd6d65fc4b4411378c3111aebcfdc8ed272f5424655"
BINDING = {
    "status": "candidate_for_selection",
    "path": "/checkpoints/b_history/arm-C/seed-42/checkpoint-1000",
    "config_sha256": CONFIG_SHA,
    "selected_arm": "C",
    "selected_step": 1000,
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _shadow(hidden=(1.0, 2.0), margin=0.25):
    return {
        "schema": "event-native-shadow-features-v1",
        "bindings": {"model": BINDING["path"], "tokenizer": "a" * 64},
        "tool_name": {"status": "located"},
        "signals": {"first_name_top2_logprob_margin": margin},
        "prefill": {
            "status": "captured",
            "layer": 34,
            "position": {"kind": "prompt_last", "logical_position": 9},
            "readout": "decoder_layer_output",
            "stored_dtype": "float16",
            "hidden": list(hidden),
        },
    }


def _server_step(
    turn: int, step: int = 0, *, terminal_ok: bool = True, response=None
):
    return {
        "decision_key": f"turn-{turn}/step-{step}",
        "status": "ok",
        "response": response or {"terminal_task_ok": terminal_ok},
        "generation_trace": [
            {
                "phase": "draft",
                "status": "completed",
                "prepared_input": {"turn": turn},
                "generation": {"stats": {"shadow_features": _shadow()}},
            }
        ],
    }


def _official_step(decoded):
    handler = (
        {"role": "handler_log", "model_response_decoded": decoded}
        if decoded is not None
        else {
            "role": "handler_log",
            "content": "Error decoding the model response. Proceed to next turn.",
        }
    )
    return [{"role": "assistant", "content": "raw"}, handler]


def _make_task(root: Path, task_id: str, server_steps, decoded_turns) -> None:
    task = root / task_id
    steps_path = task / "server" / "steps.jsonl"
    steps_path.parent.mkdir(parents=True)
    steps_path.write_text(
        "".join(json.dumps(row) + "\n" for row in server_steps), encoding="utf-8"
    )
    result_path = (
        task / "bfcl" / "bfcl" / "result" / "model" / "multi_turn"
        / "BFCL_v4_multi_turn_base_result.json"
    )
    _write_json(
        result_path,
        {
            "id": task_id,
            "result": False,
            "inference_log": [
                (
                    {
                        f"step_{index}": _official_step(decoded)
                        for index, decoded in enumerate(turn)
                    }
                    if isinstance(turn, tuple)
                    else {"step_0": _official_step(turn)}
                )
                for turn in decoded_turns
            ],
        },
    )


def test_label_collection_is_decision_local_and_keeps_config_digest_kind(tmp_path):
    task_ids = [
        "multi_turn_base_good",
        "multi_turn_base_decode",
        "multi_turn_base_prior",
        "multi_turn_base_multi",
    ]
    data_plan = {
        "schema": "a-history-r002-detector-data-plan-v1",
        "train_groups": [
            {"row_ordinal": index, "task_ids": [task_id]}
            for index, task_id in enumerate(task_ids)
        ],
        "calibration_groups": [],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, data_plan)
    returned = tmp_path / "returned"
    _make_task(returned, task_ids[0], [_server_step(0, terminal_ok=False)], [["ok"]])
    _make_task(returned, task_ids[1], [_server_step(0)], [None])
    _make_task(returned, task_ids[2], [_server_step(0), _server_step(1)], [["bad"], ["ok"]])
    _make_task(
        returned,
        task_ids[3],
        [_server_step(0, 0), _server_step(0, 1)],
        [["ok"]],
    )

    def checker(decoded, _ground_truth, _test_entry):
        return {"valid": bool(decoded and decoded[-1] and decoded[-1][0] == ["ok"])}

    rows, summary = collect_decision_rows(
        returned_roots=[returned],
        data_plan_path=plan_path,
        checkpoint_binding=BINDING,
        ratio=8,
        task_loader=lambda _task: ({"id": _task}, [["gold"], ["gold"]]),
        prefix_checker=checker,
    )
    by_key = {(row["task_id"], row["decision_key"]): row for row in rows}
    assert by_key[(task_ids[0], "turn-0/step-0")]["label"] == 0
    assert by_key[(task_ids[0], "turn-0/step-0")]["label_kind"] == (
        "single_decision_official_prefix_error"
    )
    assert by_key[(task_ids[1], "turn-0/step-0")]["label"] == 1
    assert by_key[(task_ids[2], "turn-0/step-0")]["label"] == 1
    assert by_key[(task_ids[2], "turn-1/step-0")]["label"] is None
    assert by_key[(task_ids[3], "turn-0/step-0")]["label"] is None
    assert by_key[(task_ids[3], "turn-0/step-1")]["label"] is None
    assert summary["terminal_task_outcome_used_as_step_label"] is False
    assert summary["labels"] == {"0": 1, "1": 2, "unknown": 3}
    assert all(row["checkpoint_hash"] == CONFIG_SHA for row in rows)
    assert all(row["checkpoint_hash_kind"] == "config_json" for row in rows)
    assert all(row["checkpoint_config_sha256"] == CONFIG_SHA for row in rows)
    assert all(row["checkpoint_binding_status"] == "candidate_for_selection" for row in rows)
    assert all(row["checkpoint_selected_arm"] == "C" for row in rows)
    assert all(row["checkpoint_selected_step"] == 1000 for row in rows)
    assert all(row["ratio"] == 8 for row in rows)


def test_single_action_then_verified_text_stop_labels_only_action(tmp_path):
    task_id = "multi_turn_base_segment"
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": "a-history-r002-detector-data-plan-v1",
            "train_groups": [{"row_ordinal": 7, "task_ids": [task_id]}],
            "calibration_groups": [],
        },
    )
    action_response = {
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
        "native_parse_status": "tool_calls",
        "finish_reason": "stop",
    }
    stop_response = {
        "content": "Done.",
        "tool_calls": [],
        "native_parse_status": "text",
        "finish_reason": "stop",
    }
    _make_task(
        tmp_path / "returned",
        task_id,
        [
            _server_step(0, 0, response=action_response),
            _server_step(0, 1, response=stop_response),
        ],
        [(["ok"], None)],
    )
    rows, summary = collect_decision_rows(
        returned_roots=[tmp_path / "returned"],
        data_plan_path=plan_path,
        checkpoint_binding=BINDING,
        ratio=8,
        task_loader=lambda _task: ({"id": _task}, [["gold"]]),
        prefix_checker=lambda decoded, _gold, _test: {
            "valid": decoded[-1][0] == ["ok"] and decoded[-1][1] == []
        },
    )
    action, stop = rows
    assert action["label"] == 0
    assert action["label_kind"] == "single_action_then_stop_prefix_error"
    assert action["label_target"] == "short_segment_outcome_from_action_prefill"
    assert action["segment_stop_bridge"]["official_handler_log"]["content"].startswith(
        "Error decoding the model response."
    )
    assert len(action["segment_stop_bridge"]["official_handler_log_sha256"]) == 64
    assert stop["label"] is None
    assert stop["label_kind"] is None
    assert stop["adjudication_reason"] == "single_action_then_stop_stop_decision_unknown"
    assert summary["known_label_kinds"] == {"single_action_then_stop_prefix_error": 1}


def test_checkpoint_binding_rejects_unsupported_ratio_and_ambiguous_fields():
    with pytest.raises(ValueError, match="ratio=8"):
        load_checkpoint_binding(BINDING, ratio=4)
    with pytest.raises(ValueError, match="requires status"):
        load_checkpoint_binding({**BINDING, "weight_sha256": "b" * 64}, ratio=8)


def _fit_row(split: str, group: int, label, hidden, margin: float):
    return {
        "schema": "a-history-r002-decision-label-v1",
        "task_id": f"task-{split}-{group}-{label}",
        "decision_key": f"turn-{group}/step-0",
        "split": split,
        "group_ordinal": group,
        "label": label,
        "label_kind": (
            "single_action_then_stop_prefix_error" if label in {0, 1} else None
        ),
        "label_target": (
            "short_segment_outcome_from_action_prefill" if label in {0, 1} else None
        ),
        "shadow_schema": "event-native-shadow-features-v1",
        "prefill_feature_status": "available",
        "prefill_layer": 34,
        "prefill_hidden": list(hidden),
        "prefill_stored_dtype": "float16",
        "margin_feature_status": "available",
        "first_name_top2_logprob_margin": margin,
        "model_binding": BINDING["path"],
        "tokenizer_binding": "a" * 64,
        "checkpoint_hash": CONFIG_SHA,
        "checkpoint_hash_kind": "config_json",
        "checkpoint_config_sha256": CONFIG_SHA,
        "checkpoint_binding_status": "selected",
        "checkpoint_path": BINDING["path"],
        "checkpoint_selected_arm": "C",
        "checkpoint_selected_step": 1000,
        "ratio": 8,
    }


def _fit_rows():
    rows = []
    for group in range(6):
        rows.append(_fit_row("train", group, 0, (-2.0 - group / 10, 0.1), 1 + group))
        rows.append(_fit_row("train", group, 1, (2.0 + group / 10, -0.1), 2 + group))
    for index in range(10):
        row = _fit_row(
            "calibration",
            100 + index // 2,
            None,
            (-2.5 + index * 0.55, 0.05 * index),
            float(index + 1),
        )
        row["task_id"] = f"task-calibration-{index}"
        row["decision_key"] = "turn-0/step-0"
        rows.append(row)
    rows[0]["label_kind"] = "single_decision_official_prefix_error"
    rows[0]["label_target"] = "current_decision_official_prefix_outcome"
    return rows


def test_fit_exports_runtime_head_and_semantically_validates_bundle(tmp_path):
    manifest = fit_prefill_bundle(_fit_rows(), tmp_path, dataset_sha256="b" * 64)
    assert manifest["status"] == "fitted_and_calibrated"
    assert manifest["mixed_label_kinds"] is True
    assert manifest["train_label_kind_counts"] == {
        "single_action_then_stop_prefix_error": 11,
        "single_decision_official_prefix_error": 1,
    }
    head = load_prefill_head_bundle(tmp_path)
    assert set(head) == {
        "schema", "feature", "layer", "weights", "bias", "input_mean",
        "input_scale", "score_transform", "direction", "threshold",
        "artifact_sha256",
    }
    assert head["layer"] == 34
    assert len(head["weights"]) == 2
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["prefill"]["calibration_selected_count"] == 2
    assert calibration["margin"]["calibration_selected_count"] == 2
    margin = json.loads((tmp_path / "margin_calibration.json").read_text())
    assert margin["direction"] == "at_or_below"
    assert margin["threshold"] == 2.5


def test_calibration_labels_do_not_change_fit_or_selected_c(tmp_path):
    rows = _fit_rows()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = fit_prefill_bundle(rows, first, dataset_sha256="c" * 64)
    changed = [
        {
            **row,
            "label": index % 2,
            "label_kind": "single_action_then_stop_prefix_error",
            "label_target": "short_segment_outcome_from_action_prefill",
        }
        if row["split"] == "calibration"
        else row
        for index, row in enumerate(rows)
    ]
    second_manifest = fit_prefill_bundle(changed, second, dataset_sha256="c" * 64)
    with np.load(first / "head.npz") as left, np.load(second / "head.npz") as right:
        for name in left.files:
            assert np.array_equal(left[name], right[name])
    assert first_manifest["selected_C"] == second_manifest["selected_C"]


def test_missing_class_writes_honest_not_fitted_receipt(tmp_path):
    rows = [row for row in _fit_rows() if row["label"] != 1]
    manifest = fit_prefill_bundle(rows, tmp_path, dataset_sha256="d" * 64)
    assert manifest["status"] == "not_fitted"
    assert manifest["reason"] == "training_rows_lack_both_classes"
    assert not (tmp_path / "head.npz").exists()
    assert not (tmp_path / "prefill_head.json").exists()
    assert (tmp_path / "margin_calibration.json").exists()


def test_scalar_calibration_records_seeded_boundary_ties_without_blocking_head():
    result = calibrate_exact_fraction([0.0, 0.0, 0.0, 1.0, 2.0], direction="at_or_below")
    assert result["status"] == "available"
    assert result["calibration_selected_count"] == 1
    assert result["threshold_eligible_count"] == 3
    assert result["threshold_realizes_target_without_tie_break"] is False
    assert result["runtime_reproduces_seeded_boundary_subset"] is False
    assert result["calibration_selected_count_is_not_deploy_trigger_count"] is True
    assert result["boundary_tie_break"]["seed"] == 0
    assert len(result["boundary_tie_break"]["selected_boundary_keys"]) == 1


def test_bundle_loader_rejects_metadata_tampering(tmp_path):
    fit_prefill_bundle(_fit_rows(), tmp_path, dataset_sha256="e" * 64)
    contract_path = tmp_path / "feature_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["dimension"] = 3
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="semantic artifact digest mismatch"):
        load_prefill_head_bundle(tmp_path)


def test_recalibration_reuses_head_and_only_finalizes_complete_calibration(tmp_path):
    rows = _fit_rows()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_manifest = fit_prefill_bundle(
        rows[:-1], source, dataset_sha256="1" * 64
    )
    assert source_manifest["status"] == "fitted_provisional_calibration"
    assert (source / "prefill_head.provisional.json").exists()

    result = recalibrate_prefill_bundle(
        rows,
        source,
        output,
        calibration_dataset_sha256="2" * 64,
    )
    assert result["status"] == "fitted_and_calibrated"
    assert result["deployment_status"] == "runtime_head_ready"
    assert result["training_head_reused_without_refit"] is True
    assert result["source_artifact_sha256"] == source_manifest["artifact_sha256"]
    assert result["source_dataset_sha256"] == "1" * 64
    assert result["recalibration_dataset_sha256"] == "2" * 64
    assert (output / "prefill_head.json").exists()
    assert not (output / "prefill_head.provisional.json").exists()
    with np.load(source / "head.npz") as before, np.load(output / "head.npz") as after:
        for name in before.files:
            assert np.array_equal(before[name], after[name])
    load_prefill_head_bundle(output)
