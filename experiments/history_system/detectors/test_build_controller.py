"""CPU integration tests for Prefill controller materialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.history_system.detectors.build_controller import materialize_controller
from experiments.history_system.detectors.prefill_head import fit_prefill_bundle


CONFIG_SHA = "15e14bfa5853ce7e74ef3cd6d65fc4b4411378c3111aebcfdc8ed272f5424655"
CHECKPOINT_PATH = "/checkpoints/b_history/arm-C/seed-42/checkpoint-1000"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _row(split: str, group: int, index: int, label, hidden) -> dict:
    return {
        "schema": "a-history-r002-decision-label-v1",
        "task_id": f"task-{split}-{group}-{index}",
        "decision_key": "turn-0/step-0",
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
        "first_name_top2_logprob_margin": float(index + 1),
        "model_binding": CHECKPOINT_PATH,
        "tokenizer_binding": "a" * 64,
        "checkpoint_hash": CONFIG_SHA,
        "checkpoint_hash_kind": "config_json",
        "checkpoint_config_sha256": CONFIG_SHA,
        "checkpoint_binding_status": "selected",
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_selected_arm": "C",
        "checkpoint_selected_step": 1000,
        "ratio": 8,
    }


def _rows(*, complete_calibration: bool) -> list[dict]:
    rows = []
    for group in range(6):
        rows.append(_row("train", group, 0, 0, (-2.0 - group / 10, 0.1)))
        rows.append(_row("train", group, 1, 1, (2.0 + group / 10, -0.1)))
    count = 10 if complete_calibration else 2
    for index in range(count):
        rows.append(
            _row(
                "calibration",
                100 + index // 2,
                index,
                None,
                (-2.5 + index * 0.55, 0.05 * index),
            )
        )
    return rows


def _inputs(tmp_path: Path, *, complete_calibration: bool):
    bundle = tmp_path / "head"
    manifest = fit_prefill_bundle(
        _rows(complete_calibration=complete_calibration),
        bundle,
        dataset_sha256="b" * 64,
    )
    base = tmp_path / "c0.json"
    _write(
        base,
        {
            "source_index_max_events": 12,
            "predictor_prompt_token_cap": 2048,
            "predictor_completion_token_cap": 256,
            "latest_complete_tool_protection": "budgeted",
            "observed_entity_slot_policy": (
                "same-complete-event-reference-bridge-only-v1"
            ),
        },
    )
    checkpoint = tmp_path / "checkpoint.json"
    _write(
        checkpoint,
        {
            "status": "selected",
            "path": CHECKPOINT_PATH,
            "config_sha256": CONFIG_SHA,
            "selected_arm": "C",
            "selected_step": 1000,
            "ratio": 8,
        },
    )
    shadow = tmp_path / "shadow.json"
    _write(shadow, {"enabled": True, "prefill_layer": -2, "memgen_layer": -2})
    return manifest, bundle, base, checkpoint, shadow


class _BaseController:
    policy_config = {}
    kv_bytes_per_token = 1
    tokenizer = object()
    packing = {}
    model_context = 40960

    def prepare(self, *_args, **_kwargs):
        raise AssertionError("not used by this gate integration test")

    def reconsider(self, *_args, **_kwargs):
        raise AssertionError("not used by this gate integration test")


def test_real_export_bundle_materializes_and_scores_the_same_shadow_feature(tmp_path):
    manifest, bundle, base, checkpoint, shadow = _inputs(
        tmp_path, complete_calibration=True
    )
    assert manifest["status"] == "fitted_and_calibrated"
    controller_path = tmp_path / "controller.json"
    receipt_path = tmp_path / "receipt.json"
    receipt = materialize_controller(
        base_controller_path=base,
        head_bundle=bundle,
        checkpoint_binding_path=checkpoint,
        shadow_feature_config_path=shadow,
        deployment_profile="d3_prefill_event",
        controller_output=controller_path,
        receipt_output=receipt_path,
    )

    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    recovery_config = controller["post_draft_recovery"]
    assert recovery_config["gate"] == "prefill_linear_head"
    assert receipt["status"] == "materialized_not_launched"
    assert receipt["deployment_profile"] == "d3_prefill_event"
    assert receipt["model_calls"] == receipt["network_calls"] == 0

    from experiments.history_system.runtime.benchmarks.memory_runtime.event_native_recovery import (
        EventNativeRecoveryController,
    )

    runtime = EventNativeRecoveryController(_BaseController(), recovery_config)
    hidden = [0.75, -0.25]
    shadow_trace = {
        "schema": "event-native-shadow-features-v1",
        "prefill": {
            "status": "captured",
            "layer": recovery_config["prefill_head"]["layer"],
            "position": {"kind": "prompt_last", "logical_position": 7},
            "readout": "decoder_layer_output",
            "stored_dtype": "float16",
            "hidden": hidden,
        },
    }
    gate = runtime._gate(SimpleNamespace(_shadow_features=shadow_trace))
    head = recovery_config["prefill_head"]
    normalized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            hidden, head["input_mean"], head["input_scale"], strict=True
        )
    ]
    logit = math.fsum(
        weight * value
        for weight, value in zip(head["weights"], normalized, strict=True)
    ) + head["bias"]
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert gate["score"] == pytest.approx(expected)
    assert gate["head_artifact_sha256"] == receipt["sources"]["head_bundle"][
        "artifact_sha256"
    ]


def test_provisional_calibration_bundle_is_not_materialized(tmp_path):
    manifest, bundle, base, checkpoint, shadow = _inputs(
        tmp_path, complete_calibration=False
    )
    assert manifest["deployment_status"] == "provisional_not_for_deployment"
    assert (bundle / "prefill_head.provisional.json").exists()
    with pytest.raises(ValueError, match="rejects provisional"):
        materialize_controller(
            base_controller_path=base,
            head_bundle=bundle,
            checkpoint_binding_path=checkpoint,
            shadow_feature_config_path=shadow,
            deployment_profile="d3_prefill_event",
            controller_output=tmp_path / "controller.json",
            receipt_output=tmp_path / "receipt.json",
        )
    assert not (tmp_path / "controller.json").exists()
    assert not (tmp_path / "receipt.json").exists()
