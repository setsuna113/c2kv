"""Calibrate the frozen C1 risk threshold on independent H1 observations.

This module never fits or mutates the C1 artifact.  It evaluates the exact
runtime C1 score, selects one threshold from a fixed grid by balanced accuracy,
and emits a provenance-bound receipt.  Synthetic fixtures remain explicitly
ineligible for production use.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import t02


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
SET_MODELS_PATH = RUNTIME / "benchmarks/memory_runtime/recovery/set_models.py"
_RECOVERY_PACKAGE_NAME = "_c2kv_c1_h1_recovery"
_RECOVERY_PACKAGE = types.ModuleType(_RECOVERY_PACKAGE_NAME)
_RECOVERY_PACKAGE.__path__ = [str(SET_MODELS_PATH.parent)]
sys.modules.setdefault(_RECOVERY_PACKAGE_NAME, _RECOVERY_PACKAGE)
_SET_MODELS_SPEC = importlib.util.spec_from_file_location(
    f"{_RECOVERY_PACKAGE_NAME}.set_models", SET_MODELS_PATH
)
if _SET_MODELS_SPEC is None or _SET_MODELS_SPEC.loader is None:
    raise ImportError(f"Cannot load C1 runtime model contract: {SET_MODELS_PATH}")
_SET_MODELS = importlib.util.module_from_spec(_SET_MODELS_SPEC)
sys.modules[_SET_MODELS_SPEC.name] = _SET_MODELS
_SET_MODELS_SPEC.loader.exec_module(_SET_MODELS)
C1RiskArtifact = _SET_MODELS.C1RiskArtifact
artifact_sha256 = _SET_MODELS.artifact_sha256


DATASET_SCHEMA = "c1-h1-calibration-dataset-v1"
ROW_SCHEMA = "c1-h1-calibration-row-v1"
OBSERVATION_SCHEMA = "c1-h1-calibration-observation-v1"
OUTCOME_SCHEMA = "c1-h1-a0-turn-outcome-v1"
RECEIPT_SCHEMA = "c1-h1-threshold-calibration-receipt-v1"
THRESHOLD_GRID = tuple(index / 10 for index in range(1, 10))
SOURCE_HISTORY = "H1"
HISTORY_ENCODING = "record_bound"
H1_SOURCE_HASHES = {
    "python/history_memory/encoding_scope.py": (
        "81036aba03bbdc054e7c9189922ff3cd4c42fda6cff16469a302f81a6b3a63ce"
    ),
    "python/history_memory/packing.py": (
        "064e0757d1adfea5ccbb819e41873cd8e41ca5f4d3ab9cdcbb037f473b1faa2f"
    ),
}


def canonical_payload_sha256(value: Any) -> str:
    """Return the canonical JSON payload digest used by H1 provenance."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_file_binding(path: str | Path) -> dict[str, str]:
    """Hash both exact JSON bytes and its canonical decoded payload."""

    source = Path(path).resolve()
    payload = source.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON evidence is invalid: {source}") from error
    return {
        "path": str(source),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_sha256": canonical_payload_sha256(value),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} root must be a JSON object")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return list(value)


def _resolve_reference(
    value: Any, *, input_path: Path, label: str
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} reference must be an object")
    raw_path = value.get("path")
    expected_file = value.get("file_sha256")
    expected_payload = value.get("payload_sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not _valid_sha256(expected_file)
        or not _valid_sha256(expected_payload)
    ):
        raise ValueError(
            f"{label} reference requires path, file_sha256, and payload_sha256"
        )
    path = Path(raw_path)
    path = path if path.is_absolute() else input_path.parent / path
    path = path.resolve()
    binding = json_file_binding(path)
    if (
        binding["file_sha256"] != expected_file
        or binding["payload_sha256"] != expected_payload
    ):
        raise ValueError(f"{label} differs from its bound file/payload SHA")
    return path, _read_json(path, label), binding


def _extract_gp(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(document.get("gp_experiments"), Mapping):
        return document["gp_experiments"]
    resolved = document.get("resolved_configs")
    if isinstance(resolved, Mapping):
        controller = resolved.get("controller")
        if isinstance(controller, Mapping) and isinstance(
            controller.get("gp_experiments"), Mapping
        ):
            return controller["gp_experiments"]
    return document


def _history_binding(
    dataset: Mapping[str, Any], input_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    binding = dataset.get("history_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("history_binding must be an object")
    required = {
        "source_history": SOURCE_HISTORY,
        "G": HISTORY_ENCODING,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise ValueError("calibration input must be bound to H1/record_bound")
    hashes = binding.get("h1_source_hashes")
    if hashes != H1_SOURCE_HASHES:
        raise ValueError("calibration input does not bind the frozen H1 source hashes")
    gp_value = binding.get("gp_path")
    if not isinstance(gp_value, str) or not gp_value:
        raise ValueError("history_binding.gp_path is required")
    gp_path = Path(gp_value)
    gp_path = gp_path if gp_path.is_absolute() else input_path.parent / gp_path
    gp_path = gp_path.resolve()
    gp_document = _read_json(gp_path, "source GP")
    gp_binding = json_file_binding(gp_path)
    if (
        binding.get("gp_file_sha256") != gp_binding["file_sha256"]
        or binding.get("gp_payload_sha256") != gp_binding["payload_sha256"]
    ):
        raise ValueError("source GP differs from the H1 history binding")
    if _extract_gp(gp_document).get("G") != HISTORY_ENCODING:
        raise ValueError("source GP does not configure G=record_bound")
    return dict(binding), gp_binding


def _require_identity(
    document: Mapping[str, Any], row: Mapping[str, Any], label: str
) -> None:
    for key in ("state_id", "task_id", "task_group_id"):
        if document.get(key) != row.get(key):
            raise ValueError(f"{label} {key} differs from its calibration row")
    if (
        document.get("source_history") != SOURCE_HISTORY
        or document.get("G") != HISTORY_ENCODING
    ):
        raise ValueError(f"{label} is not an H1/record_bound artifact")


def _turn_success(value: Any, *, allow_null: bool) -> bool | None:
    if value is None and allow_null:
        return None
    if type(value) is not bool:
        raise ValueError("turn_success must be boolean or null")
    return value


def _validate_row(
    row: Mapping[str, Any],
    *,
    input_path: Path,
    gp_binding: Mapping[str, str],
    expected_prefill: Mapping[str, Any],
    excluded_groups: set[str],
    evidence_mode: str,
    model: C1RiskArtifact,
) -> dict[str, Any]:
    if row.get("schema") != ROW_SCHEMA:
        raise ValueError("calibration row schema is unsupported")
    if row.get("split") != "calibration":
        raise ValueError("C1 H1 threshold calibration accepts calibration rows only")
    for key in ("state_id", "task_id", "task_group_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"calibration row requires nonempty {key}")
    if row["task_group_id"] != t02.canonical_task_group_id(row["task_id"]):
        raise ValueError("task_group_id is not canonical for task_id")
    if row["task_group_id"] in excluded_groups:
        raise ValueError("calibration row overlaps a training or evaluation group")
    context = row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("calibration row context must be an object")
    if canonical_payload_sha256(context.get("prefill_contract")) != canonical_payload_sha256(
        expected_prefill
    ):
        raise ValueError("calibration row prefill_contract differs from C1 artifact")

    evidence = row.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("calibration row requires evidence references")
    observation_path, observation, observation_binding = _resolve_reference(
        evidence.get("observation"), input_path=input_path, label="raw observation"
    )
    outcome_path, outcome, outcome_binding = _resolve_reference(
        evidence.get("official_outcome"),
        input_path=input_path,
        label="official A0 turn outcome",
    )
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("raw observation schema is unsupported")
    if outcome.get("schema") != OUTCOME_SCHEMA:
        raise ValueError("official A0 turn outcome schema is unsupported")
    _require_identity(observation, row, "raw observation")
    _require_identity(outcome, row, "official A0 turn outcome")
    for document, label in (
        (observation, "raw observation"),
        (outcome, "official A0 turn outcome"),
    ):
        if (
            document.get("source_gp_file_sha256") != gp_binding["file_sha256"]
            or document.get("source_gp_payload_sha256")
            != gp_binding["payload_sha256"]
        ):
            raise ValueError(f"{label} is bound to a different source GP")
        if document.get("evidence_mode") != evidence_mode:
            raise ValueError(f"{label} evidence_mode differs from the dataset")
    if canonical_payload_sha256(observation.get("context")) != canonical_payload_sha256(
        context
    ):
        raise ValueError("raw observation context differs from calibration row context")
    snapshot = observation.get("snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("state_id") != row["state_id"]
        or not isinstance(snapshot.get("snapshot_id"), str)
        or not snapshot["snapshot_id"]
        or not isinstance(snapshot.get("component_digests"), Mapping)
        or not snapshot["component_digests"]
    ):
        raise ValueError(
            "raw observation requires matching state_id, snapshot_id, and "
            "component_digests"
        )
    snapshot_id = snapshot["snapshot_id"]
    if outcome.get("snapshot_id") != snapshot_id:
        raise ValueError("official A0 outcome is bound to a different snapshot")
    if outcome.get("branch_id") != "A0" or outcome.get("candidate_ids") != []:
        raise ValueError("official outcome must be the A0 no-recovery branch")
    restore = outcome.get("restore_receipt")
    if not isinstance(restore, Mapping) or not restore:
        raise ValueError("official A0 outcome requires a nonempty restore_receipt")
    if (
        restore.get("restored") is not True
        or restore.get("state_id") != snapshot["state_id"]
        or restore.get("snapshot_id") != snapshot_id
        or restore.get("component_digests") != snapshot["component_digests"]
    ):
        raise ValueError("restore_receipt does not prove exact snapshot restoration")
    if outcome.get("turn_checker_result") is None:
        raise ValueError("official A0 outcome requires turn_checker_result")
    previous_turn_valid = outcome.get("previous_turn_valid")
    if previous_turn_valid is not None and type(previous_turn_valid) is not bool:
        raise ValueError("previous_turn_valid must be boolean or null")

    status = row.get("c1_label_status")
    label = row.get("c1_risk_label")
    success = _turn_success(outcome.get("turn_success"), allow_null=True)
    if status == "known":
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("known C1 risk label must be integer 0 or 1")
        if outcome["previous_turn_valid"] is not True or success is None:
            raise ValueError("known C1 risk label requires a valid previous turn outcome")
        if label != int(not success):
            raise ValueError("C1 risk label must equal 1 - A0 turn_success")
    elif status == "unknown":
        if label is not None or success is not None:
            raise ValueError("unknown C1 risk label and turn_success must remain null")
    else:
        raise ValueError("c1_label_status must be known or unknown")

    prediction = model.predict_risk(context)
    if not prediction.available or prediction.score is None:
        raise ValueError(f"C1 feature contract rejected row: {prediction.reason}")
    score = float(prediction.score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("C1 risk score must be finite and in [0, 1]")
    return {
        "state_id": row["state_id"],
        "task_id": row["task_id"],
        "task_group_id": row["task_group_id"],
        "label_status": status,
        "label": label,
        "risk_score": score,
        "snapshot_id": snapshot_id,
        "observation": {
            **observation_binding,
            "path": str(observation_path),
        },
        "official_outcome": {
            **outcome_binding,
            "path": str(outcome_path),
        },
    }


def _confusion(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        predicted = row["risk_score"] > threshold
        actual = row["label"] == 1
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    positives = tp + fn
    negatives = tn + fp
    if positives == 0 or negatives == 0:
        raise ValueError("balanced accuracy requires both risk classes")
    balanced = (Fraction(tp, positives) + Fraction(tn, negatives)) / 2
    return {
        "threshold": threshold,
        "comparison": "score > threshold",
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "true_positive_rate": tp / positives,
        "true_negative_rate": tn / negatives,
        "balanced_accuracy": float(balanced),
        "balanced_accuracy_fraction": f"{balanced.numerator}/{balanced.denominator}",
        "_balanced_fraction": balanced,
    }


def build_calibration_receipt(
    artifact_path: str | Path, input_path: str | Path
) -> dict[str, Any]:
    """Build a deterministic receipt without changing the source artifact."""

    artifact_path = Path(artifact_path).resolve()
    input_path = Path(input_path).resolve()
    artifact_before = artifact_path.read_bytes()
    artifact_document = _read_json(artifact_path, "C1 artifact")
    model = C1RiskArtifact(artifact_document)
    payload_sha = artifact_sha256(artifact_document)
    if artifact_document.get("artifact_sha256") != payload_sha:
        raise ValueError("C1 artifact payload SHA differs from its embedded hash")
    dataset = _read_json(input_path, "C1 H1 calibration input")
    if dataset.get("schema") != DATASET_SCHEMA:
        raise ValueError("C1 H1 calibration input schema is unsupported")
    if dataset.get("status") != "completed":
        raise ValueError("C1 H1 calibration requires a completed collector dataset")
    mode = dataset.get("evidence_mode")
    if mode not in {"production", "synthetic_fixture"}:
        raise ValueError("evidence_mode must be production or synthetic_fixture")
    history, gp_binding = _history_binding(dataset, input_path)
    excluded = dataset.get("excluded_group_ids")
    if not isinstance(excluded, Mapping):
        raise ValueError("excluded_group_ids must bind training and evaluation groups")
    training_groups = _strings(excluded.get("training"), "training group IDs")
    evaluation_groups = _strings(excluded.get("evaluation"), "evaluation group IDs")
    if set(training_groups) & set(evaluation_groups):
        raise ValueError("training and evaluation group exclusions overlap")
    excluded_groups = set(training_groups) | set(evaluation_groups)
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("C1 H1 calibration input requires nonempty rows")
    expected_prefill = artifact_document["feature_contract"].get("prefill_contract")
    if not isinstance(expected_prefill, Mapping) or not expected_prefill:
        raise ValueError("C1 artifact lacks a prefill_contract")
    parsed = [
        _validate_row(
            row,
            input_path=input_path,
            gp_binding=gp_binding,
            expected_prefill=expected_prefill,
            excluded_groups=excluded_groups,
            evidence_mode=mode,
            model=model,
        )
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(parsed) != len(rows):
        raise ValueError("all calibration rows must be JSON objects")
    state_ids = [row["state_id"] for row in parsed]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("calibration input repeats a state_id")
    known = [row for row in parsed if row["label_status"] == "known"]
    unknown = [row for row in parsed if row["label_status"] == "unknown"]
    metrics = [_confusion(known, threshold) for threshold in THRESHOLD_GRID]
    selected = max(
        metrics,
        key=lambda row: (row["_balanced_fraction"], row["threshold"]),
    )
    metric_rows = [
        {key: value for key, value in row.items() if key != "_balanced_fraction"}
        for row in metrics
    ]
    artifact_after = artifact_path.read_bytes()
    if artifact_after != artifact_before:
        raise RuntimeError("C1 source artifact changed during threshold calibration")
    input_binding = json_file_binding(input_path)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "completed" if mode == "production" else "synthetic_fixture_only",
        "production_eligible": mode == "production",
        "scope": "C1 H1 risk-threshold calibration only; no weight refit or task-quality claim",
        "source_artifact": {
            "path": str(artifact_path),
            "file_sha256": hashlib.sha256(artifact_before).hexdigest(),
            "payload_sha256": payload_sha,
            "payload_sha256_excludes_field": "artifact_sha256",
            "bytes_unchanged": True,
        },
        "input": input_binding,
        "history_binding": {
            **history,
            "gp_path": gp_binding["path"],
            "gp_file_sha256": gp_binding["file_sha256"],
            "gp_payload_sha256": gp_binding["payload_sha256"],
        },
        "prefill_contract": json.loads(
            json.dumps(expected_prefill, ensure_ascii=False, allow_nan=False)
        ),
        "calibration": {
            "split": "calibration",
            "task_group_ids": sorted({row["task_group_id"] for row in parsed}),
            "task_ids": sorted({row["task_id"] for row in parsed}),
            "state_count": len(parsed),
            "known_label_count": len(known),
            "unknown_label_count": len(unknown),
            "known_state_ids": [row["state_id"] for row in known],
            "excluded_unknown_state_ids": [row["state_id"] for row in unknown],
            "excluded_group_ids": {
                "training": sorted(training_groups),
                "evaluation": sorted(evaluation_groups),
            },
            "evidence": [
                {
                    "state_id": row["state_id"],
                    "snapshot_id": row["snapshot_id"],
                    "risk_score": row["risk_score"],
                    "label_status": row["label_status"],
                    "label": row["label"],
                    "observation": row["observation"],
                    "official_outcome": row["official_outcome"],
                }
                for row in parsed
            ],
        },
        "selection_rule": {
            "threshold_grid": list(THRESHOLD_GRID),
            "metric": "risk_balanced_accuracy",
            "prediction_rule": "score > threshold",
            "tie_break": "higher_threshold",
            "unknown_labels": "excluded",
            "weights_refit": False,
        },
        "grid_results": metric_rows,
        "selected_threshold": selected["threshold"],
        "selected_balanced_accuracy": selected["balanced_accuracy"],
        "selected_balanced_accuracy_fraction": selected[
            "balanced_accuracy_fraction"
        ],
    }


def verify_calibration_receipt(
    receipt_path: str | Path,
    *,
    artifact_path: str | Path,
    input_path: str | Path,
    require_production: bool = True,
) -> dict[str, Any]:
    """Recompute and verify a receipt for an H1+C1 package builder."""

    receipt_path = Path(receipt_path).resolve()
    observed = _read_json(receipt_path, "C1 H1 calibration receipt")
    expected = build_calibration_receipt(artifact_path, input_path)
    if canonical_payload_sha256(observed) != canonical_payload_sha256(expected):
        raise ValueError("C1 H1 calibration receipt differs from recomputed evidence")
    if require_production and observed.get("production_eligible") is not True:
        raise ValueError("synthetic fixture receipt is not eligible for H1+C1 execution")
    return observed


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Refusing to overwrite a different receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--artifact", type=Path, required=True)
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--allow-synthetic-fixture", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        receipt = build_calibration_receipt(args.artifact, args.input)
        _write_once(args.output, receipt)
    else:
        receipt = verify_calibration_receipt(
            args.receipt,
            artifact_path=args.artifact,
            input_path=args.input,
            require_production=not args.allow_synthetic_fixture,
        )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "status": receipt["status"],
                "production_eligible": receipt["production_eligible"],
                "selected_threshold": receipt["selected_threshold"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_SCHEMA",
    "H1_SOURCE_HASHES",
    "OBSERVATION_SCHEMA",
    "OUTCOME_SCHEMA",
    "RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "THRESHOLD_GRID",
    "build_calibration_receipt",
    "canonical_payload_sha256",
    "json_file_binding",
    "verify_calibration_receipt",
]
