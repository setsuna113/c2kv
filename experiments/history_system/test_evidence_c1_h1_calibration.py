from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import evidence_c1_h1_calibration as calibration


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path) -> tuple[Path, dict]:
    set_models = calibration._SET_MODELS

    prefill_contract = {
        "layer": 12,
        "readout": "decoder_layer_output",
        "position": {"kind": "prompt_last"},
        "bindings": {"model": "fixture-model", "tokenizer": "fixture-tokenizer"},
    }
    artifact = {
        "schema": set_models.ARTIFACT_SCHEMA,
        "model_kind": "c1_risk_logistic",
        "target": "c1_risk_label",
        "feature_contract": {
            "schema": set_models.C1_FEATURE_SCHEMA,
            "feature_names": list(set_models.C1_FEATURE_NAMES),
            "dimension": len(set_models.C1_FEATURE_NAMES),
            "prefill_contract": prefill_contract,
            "draft_logprobs_required": True,
            "missing_feature_policy": "unavailable_and_abstain",
        },
        "components": {
            "hidden_dimension": 8,
            "pca": {
                "n_components": 8,
                "mean": [0.0] * 8,
                "components": [
                    [1.0 if row == column else 0.0 for column in range(8)]
                    for row in range(8)
                ],
            },
            "scaler": {"mean": [0.0] * 11, "scale": [1.0] * 11},
            "weights": [1.0] + [0.0] * 10,
            "intercept": 0.0,
        },
        "fit": {"fixture": True},
        "provenance": {"fixture": True},
    }
    artifact["artifact_sha256"] = set_models.artifact_sha256(artifact)
    _write(path, artifact)
    return path, prefill_contract


def _context(score: float, prefill_contract: dict) -> dict:
    logit = math.log(score / (1.0 - score))
    return {
        "prefill_hidden": [logit] + [0.0] * 7,
        "draft_logprobs": [-0.3, -0.5],
        "is_stop": False,
        "parse_ok": True,
        "prefill_contract": prefill_contract,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    artifact_path, prefill_contract = _artifact(tmp_path / "c1.json")
    gp_path = tmp_path / "h1.gp.json"
    _write(gp_path, {"schema": "a-history-gp-v1", "G": "record_bound"})
    gp = calibration.json_file_binding(gp_path)
    rows = []
    evidence_documents = []
    examples = [
        ("s0", "multi_turn_base_0", 0.15, 0),
        ("s1", "multi_turn_miss_func_1", 0.35, 1),
        ("s2", "multi_turn_long_context_2", 0.55, 0),
        ("s3", "multi_turn_miss_param_3", 0.75, 1),
        ("s4", "multi_turn_base_4", 0.45, None),
    ]
    for index, (state_id, task_id, score, label) in enumerate(examples):
        group = calibration.t02.canonical_task_group_id(task_id)
        snapshot_id = f"snapshot-{index}"
        component_digests = {
            "environment": f"environment-{index}",
            "actor_kv": f"actor-kv-{index}",
        }
        context = _context(score, prefill_contract)
        observation = {
            "schema": calibration.OBSERVATION_SCHEMA,
            "state_id": state_id,
            "task_id": task_id,
            "task_group_id": group,
            "source_history": "H1",
            "G": "record_bound",
            "source_gp_file_sha256": gp["file_sha256"],
            "source_gp_payload_sha256": gp["payload_sha256"],
            "evidence_mode": "synthetic_fixture",
            "context": context,
            "snapshot": {
                "state_id": state_id,
                "snapshot_id": snapshot_id,
                "component_digests": component_digests,
                "adapter": "fixture",
            },
        }
        known = label is not None
        turn_success = bool(1 - label) if known else None
        outcome = {
            "schema": calibration.OUTCOME_SCHEMA,
            "state_id": state_id,
            "task_id": task_id,
            "task_group_id": group,
            "source_history": "H1",
            "G": "record_bound",
            "source_gp_file_sha256": gp["file_sha256"],
            "source_gp_payload_sha256": gp["payload_sha256"],
            "evidence_mode": "synthetic_fixture",
            "branch_id": "A0",
            "candidate_ids": [],
            "snapshot_id": snapshot_id,
            "restore_receipt": {
                "restored": True,
                "state_id": state_id,
                "snapshot_id": snapshot_id,
                "component_digests": component_digests,
            },
            "turn_success": turn_success,
            "turn_checker_result": {"fixture": True, "passed": turn_success},
            "previous_turn_valid": True if known else None,
        }
        observation_path = tmp_path / "evidence" / f"{state_id}.observation.json"
        outcome_path = tmp_path / "evidence" / f"{state_id}.outcome.json"
        _write(observation_path, observation)
        _write(outcome_path, outcome)
        observation_binding = calibration.json_file_binding(observation_path)
        outcome_binding = calibration.json_file_binding(outcome_path)
        row = {
            "schema": calibration.ROW_SCHEMA,
            "state_id": state_id,
            "task_id": task_id,
            "task_group_id": group,
            "split": "calibration",
            "context": context,
            "c1_risk_label": label,
            "c1_label_status": "known" if known else "unknown",
            "evidence": {
                "observation": {
                    "path": str(observation_path),
                    "file_sha256": observation_binding["file_sha256"],
                    "payload_sha256": observation_binding["payload_sha256"],
                },
                "official_outcome": {
                    "path": str(outcome_path),
                    "file_sha256": outcome_binding["file_sha256"],
                    "payload_sha256": outcome_binding["payload_sha256"],
                },
            },
        }
        rows.append(row)
        evidence_documents.append(
            {
                "observation_path": observation_path,
                "observation": observation,
                "outcome_path": outcome_path,
                "outcome": outcome,
            }
        )
    dataset = {
        "schema": calibration.DATASET_SCHEMA,
        "status": "completed",
        "evidence_mode": "synthetic_fixture",
        "history_binding": {
            "source_history": "H1",
            "G": "record_bound",
            "gp_path": str(gp_path),
            "gp_file_sha256": gp["file_sha256"],
            "gp_payload_sha256": gp["payload_sha256"],
            "h1_source_hashes": calibration.H1_SOURCE_HASHES,
        },
        "excluded_group_ids": {
            "training": ["train-0", "train-1"],
            "evaluation": ["eval-0", "eval-1"],
        },
        "rows": rows,
    }
    input_path = tmp_path / "input.json"
    _write(input_path, dataset)
    return artifact_path, input_path, evidence_documents


def test_calibrates_fixed_grid_with_strict_runtime_rule_and_higher_tie(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    before = artifact.read_bytes()
    receipt = calibration.build_calibration_receipt(artifact, input_path)

    assert artifact.read_bytes() == before
    assert receipt["status"] == "synthetic_fixture_only"
    assert receipt["production_eligible"] is False
    assert receipt["selected_threshold"] == 0.7
    assert receipt["selection_rule"] == {
        "threshold_grid": list(calibration.THRESHOLD_GRID),
        "metric": "risk_balanced_accuracy",
        "prediction_rule": "score > threshold",
        "tie_break": "higher_threshold",
        "unknown_labels": "excluded",
        "weights_refit": False,
    }
    assert receipt["calibration"]["known_label_count"] == 4
    assert receipt["calibration"]["unknown_label_count"] == 1
    assert receipt["calibration"]["excluded_unknown_state_ids"] == ["s4"]
    assert len(receipt["grid_results"]) == 9
    assert receipt["source_artifact"]["bytes_unchanged"] is True


def test_threshold_comparison_is_strictly_greater_than():
    result = calibration._confusion(
        [
            {"risk_score": 0.7, "label": 1},
            {"risk_score": 0.0, "label": 0},
        ],
        0.7,
    )
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["comparison"] == "score > threshold"


def test_verify_recomputes_receipt_and_rejects_fixture_for_production(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = calibration.build_calibration_receipt(artifact, input_path)
    _write(receipt_path, receipt)

    verified = calibration.verify_calibration_receipt(
        receipt_path,
        artifact_path=artifact,
        input_path=input_path,
        require_production=False,
    )
    assert verified["selected_threshold"] == 0.7
    with pytest.raises(ValueError, match="synthetic fixture"):
        calibration.verify_calibration_receipt(
            receipt_path, artifact_path=artifact, input_path=input_path
        )


def test_rejects_risk_label_not_derived_from_official_a0_turn(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["rows"][0]["c1_risk_label"] = 1
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="1 - A0 turn_success"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_rejects_h0_observation_even_when_reference_hash_is_updated(tmp_path):
    artifact, input_path, documents = _fixture(tmp_path)
    first = documents[0]
    first["observation"]["source_history"] = "H0"
    _write(first["observation_path"], first["observation"])
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    binding = calibration.json_file_binding(first["observation_path"])
    dataset["rows"][0]["evidence"]["observation"].update(
        file_sha256=binding["file_sha256"],
        payload_sha256=binding["payload_sha256"],
    )
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="not an H1/record_bound"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_rejects_non_calibration_or_excluded_group(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["rows"][0]["split"] = "train"
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="calibration rows only"):
        calibration.build_calibration_receipt(artifact, input_path)

    artifact, input_path, _ = _fixture(tmp_path / "overlap")
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["excluded_group_ids"]["training"].append("bfcl_pair_0")
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="overlaps a training or evaluation group"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_rejects_partial_collector_dataset_without_selecting_around_failures(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["status"] = "partial_failed"
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="completed collector dataset"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_rejects_context_or_gp_provenance_drift(tmp_path):
    artifact, input_path, documents = _fixture(tmp_path)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["rows"][0]["context"]["is_stop"] = True
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="observation context differs"):
        calibration.build_calibration_receipt(artifact, input_path)

    artifact, input_path, documents = _fixture(tmp_path / "gp")
    first = documents[0]
    first["outcome"]["source_gp_payload_sha256"] = "0" * 64
    _write(first["outcome_path"], first["outcome"])
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    binding = calibration.json_file_binding(first["outcome_path"])
    dataset["rows"][0]["evidence"]["official_outcome"].update(
        file_sha256=binding["file_sha256"],
        payload_sha256=binding["payload_sha256"],
    )
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="different source GP"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_unknown_label_allows_null_previous_turn_valid(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    receipt = calibration.build_calibration_receipt(artifact, input_path)
    assert receipt["calibration"]["excluded_unknown_state_ids"] == ["s4"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("restored", False),
        ("state_id", "different-state"),
        ("snapshot_id", "different-snapshot"),
        ("component_digests", {"environment": "different"}),
    ],
)
def test_rejects_restore_receipt_that_does_not_match_snapshot(
    tmp_path, field, replacement
):
    artifact, input_path, documents = _fixture(tmp_path)
    first = documents[0]
    first["outcome"]["restore_receipt"][field] = replacement
    _write(first["outcome_path"], first["outcome"])
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    binding = calibration.json_file_binding(first["outcome_path"])
    dataset["rows"][0]["evidence"]["official_outcome"].update(
        file_sha256=binding["file_sha256"],
        payload_sha256=binding["payload_sha256"],
    )
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="exact snapshot restoration"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_rejects_noncanonical_task_group_id(tmp_path):
    artifact, input_path, _ = _fixture(tmp_path)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    dataset["rows"][0]["task_group_id"] = "bfcl_pair_999"
    _write(input_path, dataset)
    with pytest.raises(ValueError, match="not canonical"):
        calibration.build_calibration_receipt(artifact, input_path)


def test_cli_writes_only_a_fixture_receipt_for_synthetic_input(tmp_path, capsys):
    artifact, input_path, _ = _fixture(tmp_path)
    output = tmp_path / "receipt.json"
    assert calibration.main(
        [
            "calibrate",
            "--artifact",
            str(artifact),
            "--input",
            str(input_path),
            "--output",
            str(output),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed["production_eligible"] is False
    assert saved["status"] == "synthetic_fixture_only"
