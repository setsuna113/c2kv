"""Offline detector-threshold calibration from observed development receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CALIBRATION_SCHEMA = "a-history-detector-exposure-calibration-v1"
OBSERVATION_SCHEMA = "a-history-detector-feasibility-observation-v1"
DEFAULT_TARGET_EXPOSURES = (0.4, 0.6)
_DEVELOPMENT_SPLITS = frozenset({"dev", "development", "calibration"})


def calibrate_detector_thresholds(
    log_path: str | Path,
    *,
    target_exposures: Sequence[float] = DEFAULT_TARGET_EXPOSURES,
) -> dict[str, Any]:
    """Calibrate scalar thresholds over feasible development states only.

    The input must be an existing JSON or JSONL log. Every included decision
    needs an observed detector score, detector direction, explicit split, and
    the runtime-produced legal-candidate feasibility marker. No score or
    feasibility value is inferred from outcomes or labels.
    """

    path = Path(log_path)
    payload = path.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    rows, source_format = _load_rows(payload, path)
    observations = [_parse_observation(row, index) for index, row in enumerate(rows)]
    if not observations:
        raise ValueError("development risk log contains no detector observations")

    directions = {row["direction"] for row in observations}
    if len(directions) != 1:
        raise ValueError("development risk log mixes detector directions")
    detector_types = {row["detector_type"] for row in observations}
    features = {row["feature"] for row in observations}
    if len(detector_types) != 1:
        raise ValueError("development risk log mixes detector types")
    if len(features) != 1:
        raise ValueError("development risk log mixes detector features")
    feasible = [row for row in observations if row["candidate_feasible"]]
    if not feasible:
        raise ValueError("development risk log contains no feasible candidate states")

    targets = [_target(value) for value in target_exposures]
    if len(set(targets)) != len(targets):
        raise ValueError("target exposures must be unique")
    direction = directions.pop()
    calibrations = [
        _calibrate([row["score"] for row in feasible], target, direction)
        for target in targets
    ]
    splits = sorted({row["split"] for row in observations})
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "calibrated_offline",
        "source": {
            "path": str(path.resolve()),
            "sha256": source_sha256,
            "format": source_format,
            "parsed_observation_count": len(observations),
        },
        "data_contract": {
            "accepted_splits": sorted(_DEVELOPMENT_SPLITS),
            "observed_splits": splits,
            "test_split_rows_used": 0,
            "requires_observed_detector_score": True,
            "requires_runtime_candidate_feasibility": True,
            "feasibility_stage": (
                "single_candidate_expansion_after_real_b0_admission"
            ),
        },
        "detector": {
            "types": sorted(detector_types),
            "features": sorted(features),
            "direction": direction,
        },
        "observations": {
            "all_development_states": len(observations),
            "feasible_candidate_states": len(feasible),
            "infeasible_candidate_states_excluded": len(observations) - len(feasible),
        },
        "calibrations": calibrations,
        "execution": {
            "model_calls": 0,
            "supervised_training_runs": 0,
            "uses_outcome_labels": False,
            "offline_only": True,
        },
    }


def _load_rows(payload: bytes, path: Path) -> tuple[list[Mapping[str, Any]], str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("development risk log must be UTF-8 JSON or JSONL") from exc
    stripped = text.strip()
    if not stripped:
        raise ValueError("development risk log is empty")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed_rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
        parsed = parsed_rows
        source_format = "jsonl"
    else:
        source_format = "json"
    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, Mapping) and isinstance(parsed.get("records"), list):
        records = parsed["records"]
    elif isinstance(parsed, Mapping):
        records = [parsed]
    else:
        raise ValueError("development risk log must contain objects")

    flattened = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("development risk log rows must be objects")
        flattened.extend(_decision_rows(record))
    return flattened, source_format


def _decision_rows(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    inherited = {
        key: record[key]
        for key in ("split", "session_id", "decision_key")
        if key in record
    }
    checks = record.get("recovery_checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, Mapping):
                raise ValueError("recovery_checks entries must be objects")
            yield {**inherited, "decision": check}
        return
    decision = record.get("decision")
    if isinstance(decision, Mapping):
        yield record
        return
    exact = record.get("exact_recovery")
    if isinstance(exact, Mapping):
        yield {**inherited, "decision": exact}
        return
    yield record


def _parse_observation(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    split = row.get("split")
    if not isinstance(split, str):
        raise ValueError(f"detector observation {index} lacks an explicit split")
    normalized_split = split.strip().lower()
    if normalized_split not in _DEVELOPMENT_SPLITS:
        raise ValueError(
            f"detector observation {index} uses non-development split {split!r}"
        )
    decision = row.get("decision")
    container = decision if isinstance(decision, Mapping) else row
    telemetry = container.get("calibration_telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError(
            f"detector observation {index} lacks runtime calibration telemetry"
        )
    score = telemetry.get("detector_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"detector observation {index} lacks a real detector score")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"detector observation {index} has a non-finite detector score")
    feasible = telemetry.get("candidate_feasible")
    if type(feasible) is not bool:
        raise ValueError(
            f"detector observation {index} lacks a boolean candidate feasibility marker"
        )
    if telemetry.get("feasibility_stage") != (
        "single_candidate_expansion_after_real_b0_admission"
    ):
        raise ValueError(
            f"detector observation {index} lacks explicit B0 admission feasibility"
        )
    direction = telemetry.get("detector_direction")
    if direction not in {"at_or_above", "at_or_below"}:
        raise ValueError(f"detector observation {index} lacks a supported direction")
    detector_type = telemetry.get("detector_type")
    feature = telemetry.get("detector_feature")
    if not isinstance(detector_type, str) or not detector_type:
        raise ValueError(f"detector observation {index} lacks detector type")
    if not isinstance(feature, str) or not feature:
        raise ValueError(f"detector observation {index} lacks detector feature")
    return {
        "split": normalized_split,
        "score": score,
        "candidate_feasible": feasible,
        "direction": direction,
        "detector_type": detector_type,
        "feature": feature,
    }


def _target(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("target exposures must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError("target exposures must be in (0, 1]")
    return result


def _calibrate(scores: Sequence[float], target: float, direction: str) -> dict[str, Any]:
    ordered = sorted(scores, reverse=direction == "at_or_above")
    target_count = math.ceil(len(ordered) * target)
    boundary_score = ordered[target_count - 1]
    next_score = ordered[target_count] if target_count < len(ordered) else None
    if next_score is not None and next_score != boundary_score:
        threshold = (boundary_score + next_score) / 2.0
        threshold_uses_midpoint = True
    else:
        threshold = boundary_score
        threshold_uses_midpoint = False
    if direction == "at_or_above":
        actual_count = sum(score >= threshold for score in scores)
    else:
        actual_count = sum(score <= threshold for score in scores)
    actual_exposure = actual_count / len(scores)
    boundary_tie_count = sum(score == boundary_score for score in scores)
    return {
        "target_exposure": target,
        "target_count": target_count,
        "target_count_rounding": "ceil",
        "threshold": threshold,
        "direction": direction,
        "feasible_state_count": len(scores),
        "actual_trigger_count_including_ties": actual_count,
        "actual_exposure_including_ties": actual_exposure,
        "actual_exposure_differs_from_target": actual_exposure != target,
        "boundary": {
            "score": boundary_score,
            "score_tie_count": boundary_tie_count,
            "next_score": next_score,
            "threshold_uses_midpoint": threshold_uses_midpoint,
            "ties_are_all_triggered_by_scalar_runtime_threshold": True,
        },
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate 40% and 60% detector exposure from development receipts."
    )
    parser.add_argument("risk_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    artifact = calibrate_detector_thresholds(args.risk_log)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
