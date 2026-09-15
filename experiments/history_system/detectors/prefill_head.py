"""Fit and export the fixed r002 Prefill detector from decision-local labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .labels import (
    CHECKPOINT_HASH_KIND,
    EXACT_LABEL_KIND,
    EXACT_LABEL_TARGET,
    LABEL_SCHEMA,
    SEGMENT_LABEL_KIND,
    SEGMENT_LABEL_TARGET,
    SUPPORTED_RATIO,
    add_bfcl_root,
    collect_decision_rows,
    write_json,
    write_jsonl,
)


HEAD_SCHEMA = "event-native-prefill-linear-head-v1"
FEATURE = "prefill.prompt_last.decoder_layer_output"
SHADOW_SCHEMA = "event-native-shadow-features-v1"
REQUESTED_LAYER = -2
FIT_SEED = 20260905
INNER_C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1.0)
TARGET_FIRE_FRACTION = {"numerator": 1, "denominator": 5}
EXPECTED_TRAIN_TASKS = 40
EXPECTED_CALIBRATION_TASKS = 10
EXPECTED_TRAIN_GROUPS = 20
EXPECTED_CALIBRATION_GROUPS = 5
_OUTPUT_FILES = (
    "head.npz",
    "prefill_head.json",
    "prefill_head.provisional.json",
    "margin_calibration.json",
    "feature_contract.json",
    "fit_manifest.json",
    "calibration.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


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


def _finite_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "checkpoint_binding_status",
        "checkpoint_hash",
        "checkpoint_hash_kind",
        "checkpoint_config_sha256",
        "checkpoint_path",
        "checkpoint_selected_arm",
        "checkpoint_selected_step",
        "ratio",
    )
    if not rows:
        raise ValueError("decision dataset is empty")
    values = {tuple(row.get(field) for field in fields) for row in rows}
    if len(values) != 1:
        raise ValueError("decision rows mix checkpoint or ratio bindings")
    result = dict(zip(fields, next(iter(values)), strict=True))
    if result["checkpoint_binding_status"] != "selected":
        raise ValueError("r002 fitting requires an explicitly selected checkpoint binding")
    if result["checkpoint_hash_kind"] != CHECKPOINT_HASH_KIND:
        raise ValueError("checkpoint_hash must be identified as config_json")
    digest = result["checkpoint_hash"]
    if result["checkpoint_config_sha256"] != digest:
        raise ValueError("checkpoint config digest fields disagree")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint config digest is invalid")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("checkpoint config digest is not hexadecimal") from error
    if result["ratio"] != SUPPORTED_RATIO:
        raise ValueError(f"r002 detector rows require ratio={SUPPORTED_RATIO}")
    if not isinstance(result["checkpoint_path"], str) or not result["checkpoint_path"]:
        raise ValueError("checkpoint path binding is missing")
    if not isinstance(result["checkpoint_selected_arm"], str):
        raise ValueError("checkpoint arm binding is missing")
    if type(result["checkpoint_selected_step"]) is not int:
        raise ValueError("checkpoint step binding is missing")
    if result["checkpoint_selected_arm"] != "C" or result["checkpoint_selected_step"] != 1000:
        raise ValueError("r002 detector rows require arm C checkpoint step 1000")
    return result


def _prefill_rows(
    rows: Sequence[Mapping[str, Any]], split: str, *, require_label: bool
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("split") != split or row.get("prefill_feature_status") != "available":
            continue
        if row.get("schema") != LABEL_SCHEMA or row.get("shadow_schema") != SHADOW_SCHEMA:
            continue
        if require_label and row.get("label") not in {0, 1}:
            continue
        hidden = _finite_vector(row.get("prefill_hidden"), "prefill_hidden")
        selected.append({**dict(row), "prefill_hidden": hidden})
    return selected


def _margin_rows(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("split") != split or row.get("margin_feature_status") != "available":
            continue
        value = row.get("first_name_top2_logprob_margin")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            continue
        selected.append(dict(row))
    return selected


def _feature_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no Prefill feature rows")
    dimensions = {len(row["prefill_hidden"]) for row in rows}
    layers = {row.get("prefill_layer") for row in rows}
    models = {row.get("model_binding") for row in rows}
    tokenizers = {row.get("tokenizer_binding") for row in rows}
    stored_dtypes = {row.get("prefill_stored_dtype") for row in rows}
    if len(dimensions) != 1 or len(layers) != 1:
        raise ValueError("Prefill rows mix layer or hidden dimension")
    layer = next(iter(layers))
    if type(layer) is not int or layer < 0:
        raise ValueError("Prefill layer must be a normalized nonnegative index")
    if len(models) != 1 or not isinstance(next(iter(models)), str):
        raise ValueError("Prefill rows require one explicit model binding")
    if len(tokenizers) != 1 or not isinstance(next(iter(tokenizers)), str):
        raise ValueError("Prefill rows require one explicit tokenizer binding")
    tokenizer = next(iter(tokenizers))
    if len(tokenizer) != 64:
        raise ValueError("Prefill tokenizer binding must be a SHA-256 digest")
    try:
        int(tokenizer, 16)
    except ValueError as error:
        raise ValueError("Prefill tokenizer binding must be hexadecimal") from error
    if stored_dtypes != {"float16"}:
        raise ValueError("Prefill rows require the fixed float16 stored input dtype")
    return {
        "resolved_layer": layer,
        "dimension": next(iter(dimensions)),
        "model_binding": next(iter(models)),
        "tokenizer_binding": tokenizer,
        "stored_input_dtype": "float16",
    }


def _group_folds(groups: Sequence[int], *, seed: int) -> list[set[int]]:
    unique = sorted(set(groups))
    if len(unique) < 3:
        return []
    random.Random(seed).shuffle(unique)
    folds = [set(unique[offset::3]) for offset in range(3)]
    return folds if all(folds) else []


def _select_c(x, y, groups):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    folds = _group_folds([int(value) for value in groups], seed=FIT_SEED)
    if not folds:
        return None, [], "fewer_than_three_training_groups"
    cv_rows = []
    for c_value in INNER_C_GRID:
        fold_scores = []
        for fold_index, validation_groups in enumerate(folds):
            validation = np.asarray(
                [group in validation_groups for group in groups], dtype=bool
            )
            training = ~validation
            if (
                len(set(y[validation].tolist())) != 2
                or len(set(y[training].tolist())) != 2
            ):
                return None, cv_rows, "grouped_fold_lacks_class_coverage"
            scaler = StandardScaler().fit(x[training])
            model = LogisticRegression(
                C=c_value,
                solver="liblinear",
                max_iter=2000,
                random_state=FIT_SEED,
            ).fit(scaler.transform(x[training]), y[training])
            score = model.predict_proba(scaler.transform(x[validation]))[:, 1]
            fold_scores.append(float(roc_auc_score(y[validation], score)))
        cv_rows.append(
            {
                "C": c_value,
                "fold_auroc": fold_scores,
                "mean_auroc": float(np.mean(fold_scores)),
            }
        )
    selected = max(cv_rows, key=lambda row: (row["mean_auroc"], -row["C"]))
    return float(selected["C"]), cv_rows, None


def calibrate_exact_fraction(
    values: Sequence[float],
    *,
    direction: str,
    keys: Sequence[str] | None = None,
    tie_break_seed: int = 0,
) -> dict[str, Any]:
    """Choose a 0.2 quantile threshold with a recorded seed-0 boundary split."""

    if direction not in {"at_or_above", "at_or_below"}:
        raise ValueError("unsupported calibration direction")
    if keys is None:
        keys = [str(index) for index in range(len(values))]
    if len(keys) != len(values) or len(set(keys)) != len(keys):
        raise ValueError("calibration keys must be unique and align with values")
    clean = [
        (float(value), str(key))
        for value, key in zip(values, keys, strict=True)
        if math.isfinite(float(value))
    ]
    if not clean:
        return {"status": "unavailable", "reason": "no_calibration_features", "n": 0}
    ordered = sorted(
        clean,
        key=lambda item: item[0],
        reverse=direction == "at_or_above",
    )
    fire_count = math.ceil(len(ordered) / 5)
    included = ordered[fire_count - 1][0]
    boundary = [key for value, key in clean if value == included]
    strict = [
        key
        for value, key in clean
        if (value > included if direction == "at_or_above" else value < included)
    ]
    boundary_needed = fire_count - len(strict)
    ranked_boundary = sorted(
        boundary,
        key=lambda key: hashlib.sha256(
            f"seed={tie_break_seed}:{key}".encode("utf-8")
        ).hexdigest(),
    )
    selected_boundary = ranked_boundary[:boundary_needed]
    if fire_count == len(ordered):
        threshold = included
    else:
        excluded = ordered[fire_count][0]
        if included == excluded:
            threshold = included
        else:
            threshold = (included + excluded) / 2.0
    threshold_eligible = sum(
        value >= threshold if direction == "at_or_above" else value <= threshold
        for value, _key in clean
    )
    return {
        "status": "available",
        "direction": direction,
        "threshold": threshold,
        "n": len(clean),
        "target_fire_fraction": TARGET_FIRE_FRACTION,
        "target_fire_count": fire_count,
        "calibration_selected_count": len(strict) + len(selected_boundary),
        "threshold_eligible_count": threshold_eligible,
        "threshold_realizes_target_without_tie_break": threshold_eligible == fire_count,
        "runtime_reproduces_seeded_boundary_subset": threshold_eligible == fire_count,
        "calibration_selected_count_is_not_deploy_trigger_count": True,
        "boundary_tie_break": {
            "seed": tie_break_seed,
            "key": "stable_calibration_item_key",
            "boundary_value": included,
            "boundary_candidate_count": len(boundary),
            "boundary_selected_count": len(selected_boundary),
            "selected_boundary_keys": selected_boundary,
        },
        "online_quota_contract": "ceil(0.2 * decisions_seen) remains the recovery cap",
    }


def _array_manifest(arrays: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = {}
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name], dtype="<f8"))
        result[name] = {
            "dtype": "float64-little-endian",
            "shape": list(array.shape),
            "data_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def _bundle_core(
    arrays: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    fit_manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "a-history-r002-prefill-bundle-digest-v1",
        "digest_scope": "canonical JSON metadata plus semantic little-endian float64 arrays",
        "head_arrays": _array_manifest(arrays),
        "feature_contract": dict(feature_contract),
        "fit_manifest": dict(fit_manifest),
        "calibration": dict(calibration),
    }


def _remove_stale(output_dir: Path) -> None:
    for name in _OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _write_margin_artifact(
    output_dir: Path,
    margin_calibration: dict[str, Any],
    *,
    dataset_sha256: str,
    provenance: Mapping[str, Any],
) -> None:
    if margin_calibration["status"] != "available":
        return
    digest = _sha256_json(
        {
            "schema": "a-history-r002-margin-calibration-artifact-v1",
            "dataset_sha256": dataset_sha256,
            "checkpoint_binding": dict(provenance),
            "calibration": margin_calibration,
        }
    )
    margin_calibration["artifact_sha256"] = digest
    write_json(
        output_dir / "margin_calibration.json",
        {
            "feature": "first_name_top2_logprob_margin",
            "direction": "at_or_below",
            "threshold": margin_calibration["threshold"],
            "artifact_sha256": digest,
        },
    )


def fit_prefill_bundle(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Fit with grouped train CV and calibrate only on held calibration groups."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale(output_dir)
    provenance = _provenance(rows)
    label_counts = Counter(
        "unknown" if row.get("label") is None else str(row.get("label"))
        for row in rows
    )
    train = _prefill_rows(rows, "train", require_label=True)
    calibration_rows = _prefill_rows(rows, "calibration", require_label=False)
    margin_rows = _margin_rows(rows, "calibration")
    train_tasks = {row.get("task_id") for row in rows if row.get("split") == "train"}
    calibration_tasks = {
        row.get("task_id") for row in rows if row.get("split") == "calibration"
    }
    train_groups_seen = {
        row.get("group_ordinal") for row in rows if row.get("split") == "train"
    }
    calibration_groups_seen = {
        row.get("group_ordinal")
        for row in rows
        if row.get("split") == "calibration"
    }
    training_collection_complete = bool(
        len(train_tasks) == EXPECTED_TRAIN_TASKS
        and len(train_groups_seen) == EXPECTED_TRAIN_GROUPS
    )
    calibration_collection_complete = bool(
        len(calibration_tasks) == EXPECTED_CALIBRATION_TASKS
        and len(calibration_groups_seen) == EXPECTED_CALIBRATION_GROUPS
    )
    collection_complete = training_collection_complete and calibration_collection_complete
    base_manifest = {
        "schema": "a-history-r002-prefill-fit-manifest-v1",
        "dataset_sha256": dataset_sha256,
        "fit_seed": FIT_SEED,
        "inner_C_grid": list(INNER_C_GRID),
        "group_key": "group_ordinal",
        "inner_folds": 3,
        "train_known_prefill_rows": len(train),
        "calibration_prefill_rows": len(calibration_rows),
        "calibration_margin_rows": len(margin_rows),
        "all_label_counts": dict(sorted(label_counts.items())),
        "known_label_kind_counts": dict(sorted(Counter(
            str(row.get("label_kind")) for row in rows if row.get("label") in {0, 1}
        ).items())),
        "known_label_target_counts": dict(sorted(Counter(
            str(row.get("label_target")) for row in rows if row.get("label") in {0, 1}
        ).items())),
        "detector_scope": (
            "short-horizon official-prefix risk from eligible decision Prefill; "
            "segment labels attach only to the action decision"
        ),
        "collection_scope": (
            "complete_r002_data_plan"
            if collection_complete
            else "current_completed_task_snapshot"
        ),
        "collection_complete": collection_complete,
        "training_collection_complete": training_collection_complete,
        "calibration_collection_complete": calibration_collection_complete,
        "observed_tasks": {
            "train": len(train_tasks),
            "calibration": len(calibration_tasks),
        },
        "expected_tasks": {
            "train": EXPECTED_TRAIN_TASKS,
            "calibration": EXPECTED_CALIBRATION_TASKS,
        },
        "observed_groups": {
            "train": len(train_groups_seen),
            "calibration": len(calibration_groups_seen),
        },
        "expected_groups": {
            "train": EXPECTED_TRAIN_GROUPS,
            "calibration": EXPECTED_CALIBRATION_GROUPS,
        },
        "checkpoint_binding": provenance,
        "terminal_task_outcome_used_as_step_label": False,
    }
    margin_values = [row["first_name_top2_logprob_margin"] for row in margin_rows]
    margin_calibration = calibrate_exact_fraction(
        margin_values,
        direction="at_or_below",
        keys=[f"{row['task_id']}/{row['decision_key']}" for row in margin_rows],
    )
    _write_margin_artifact(
        output_dir,
        margin_calibration,
        dataset_sha256=dataset_sha256,
        provenance=provenance,
    )

    def stop(reason: str) -> dict[str, Any]:
        manifest = {**base_manifest, "status": "not_fitted", "reason": reason}
        write_json(output_dir / "fit_manifest.json", manifest)
        write_json(
            output_dir / "calibration.json",
            {
                "schema": "a-history-r002-detector-calibration-v1",
                "prefill": {"status": "unavailable", "reason": "head_not_fitted"},
                "margin": margin_calibration,
            },
        )
        return manifest

    if not train:
        return stop("no_known_training_rows_with_prefill")
    if {row["label"] for row in train} != {0, 1}:
        return stop("training_rows_lack_both_classes")
    allowed_semantics = {
        EXACT_LABEL_KIND: EXACT_LABEL_TARGET,
        SEGMENT_LABEL_KIND: SEGMENT_LABEL_TARGET,
    }
    if any(
        allowed_semantics.get(row.get("label_kind")) != row.get("label_target")
        for row in train
    ):
        return stop("training_rows_have_unsupported_label_semantics")
    try:
        identity = _feature_identity(train + calibration_rows)
    except ValueError as error:
        return stop(f"feature_contract_error:{error}")
    if identity["model_binding"] != provenance["checkpoint_path"]:
        return stop("feature_contract_error:model binding differs from checkpoint path")
    x = np.asarray([row["prefill_hidden"] for row in train], dtype=np.float64)
    y = np.asarray([row["label"] for row in train], dtype=np.int64)
    groups = np.asarray([row["group_ordinal"] for row in train], dtype=np.int64)
    selected_c, cv_rows, cv_error = _select_c(x, y, groups)
    if cv_error is not None:
        return stop(cv_error)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=selected_c,
        solver="liblinear",
        max_iter=2000,
        random_state=FIT_SEED,
    ).fit(scaler.transform(x), y)
    arrays = {
        "weights": np.asarray(model.coef_[0], dtype=np.float64),
        "bias": np.asarray(model.intercept_[0], dtype=np.float64),
        "input_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "input_scale": np.asarray(scaler.scale_, dtype=np.float64),
    }
    if calibration_rows:
        cal_x = np.asarray(
            [row["prefill_hidden"] for row in calibration_rows], dtype=np.float64
        )
        cal_scores = model.predict_proba(scaler.transform(cal_x))[:, 1].tolist()
    else:
        cal_scores = []
    prefill_calibration = calibrate_exact_fraction(
        cal_scores,
        direction="at_or_above",
        keys=[
            f"{row['task_id']}/{row['decision_key']}" for row in calibration_rows
        ],
    )
    feature_contract = {
        "schema": "a-history-r002-prefill-feature-contract-v1",
        "shadow_schema": SHADOW_SCHEMA,
        "feature": FEATURE,
        "position": "prompt_last",
        "readout": "decoder_layer_output",
        "requested_layer": REQUESTED_LAYER,
        "resolved_layer": identity["resolved_layer"],
        "dimension": identity["dimension"],
        "stored_input_dtype": identity["stored_input_dtype"],
        "fit_dtype": "float64",
        "model_binding": identity["model_binding"],
        "tokenizer_binding": identity["tokenizer_binding"],
        "checkpoint_binding": provenance,
        "training_label_scope": (
            "short-horizon official-prefix risk from eligible decision Prefill; "
            "segment labels attach only to the action decision"
        ),
        "training_label_kinds": sorted({row["label_kind"] for row in train}),
    }
    fit_manifest = {
        **base_manifest,
        "status": (
            "fitted_and_calibrated"
            if (
                prefill_calibration["status"] == "available"
                and calibration_collection_complete
            )
            else "fitted_provisional_calibration"
            if prefill_calibration["status"] == "available"
            else "fitted_calibration_unavailable"
        ),
        "deployment_status": (
            "runtime_head_ready"
            if (
                prefill_calibration["status"] == "available"
                and calibration_collection_complete
            )
            else "provisional_not_for_deployment"
        ),
        "selected_C": selected_c,
        "inner_cv": cv_rows,
        "train_label_counts": dict(sorted(Counter(map(str, y.tolist())).items())),
        "training_groups": len(set(groups.tolist())),
        "train_label_kind_counts": dict(sorted(Counter(
            row["label_kind"] for row in train
        ).items())),
        "train_label_target_counts": dict(sorted(Counter(
            row["label_target"] for row in train
        ).items())),
        "mixed_label_kinds": len({row["label_kind"] for row in train}) > 1,
        "calibration_known_labels": sum(
            row.get("label") in {0, 1} for row in calibration_rows
        ),
        "calibration_unknown_labels": sum(
            row.get("label") is None for row in calibration_rows
        ),
    }
    calibration = {
        "schema": "a-history-r002-detector-calibration-v1",
        "prefill": prefill_calibration,
        "margin": margin_calibration,
    }
    core = _bundle_core(arrays, feature_contract, fit_manifest, calibration)
    artifact_sha256 = _sha256_json(core)
    fit_manifest["artifact_sha256"] = artifact_sha256
    fit_manifest["artifact_digest_contract"] = core["digest_scope"]
    calibration["artifact_sha256"] = artifact_sha256
    feature_contract["artifact_sha256"] = artifact_sha256
    np.savez(output_dir / "head.npz", **arrays)
    write_json(output_dir / "feature_contract.json", feature_contract)
    write_json(output_dir / "fit_manifest.json", fit_manifest)
    write_json(output_dir / "calibration.json", calibration)
    if prefill_calibration["status"] == "available":
        runtime_head = {
            "schema": HEAD_SCHEMA,
            "feature": FEATURE,
            "layer": identity["resolved_layer"],
            "weights": arrays["weights"].tolist(),
            "bias": float(arrays["bias"]),
            "input_mean": arrays["input_mean"].tolist(),
            "input_scale": arrays["input_scale"].tolist(),
            "score_transform": "sigmoid",
            "direction": "at_or_above",
            "threshold": prefill_calibration["threshold"],
            "artifact_sha256": artifact_sha256,
        }
        runtime_name = (
            "prefill_head.json"
            if calibration_collection_complete
            else "prefill_head.provisional.json"
        )
        write_json(output_dir / runtime_name, runtime_head)
    return fit_manifest


def load_prefill_head_bundle(path: Path) -> dict[str, Any]:
    """Validate a bundle's semantic digest and return the exact runtime mapping."""

    import numpy as np

    feature_contract = json.loads((path / "feature_contract.json").read_text(encoding="utf-8"))
    fit_manifest = json.loads((path / "fit_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
    runtime_paths = [
        candidate
        for candidate in (
            path / "prefill_head.json",
            path / "prefill_head.provisional.json",
        )
        if candidate.exists()
    ]
    if len(runtime_paths) != 1:
        raise ValueError("bundle requires exactly one final or provisional Prefill head")
    runtime_head = json.loads(runtime_paths[0].read_text(encoding="utf-8"))
    with np.load(path / "head.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"weights", "bias", "input_mean", "input_scale"}:
            raise ValueError("head.npz fields do not match the bundle schema")
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    for document in (feature_contract, fit_manifest, calibration):
        if document.get("artifact_sha256") != runtime_head.get("artifact_sha256"):
            raise ValueError("bundle artifact digests disagree")
    core = _bundle_core(
        arrays,
        {key: value for key, value in feature_contract.items() if key != "artifact_sha256"},
        {
            key: value
            for key, value in fit_manifest.items()
            if key not in {"artifact_sha256", "artifact_digest_contract"}
        },
        {key: value for key, value in calibration.items() if key != "artifact_sha256"},
    )
    actual_digest = _sha256_json(core)
    if actual_digest != runtime_head.get("artifact_sha256"):
        raise ValueError("bundle semantic artifact digest mismatch")
    required = {
        "schema", "feature", "layer", "weights", "bias", "input_mean",
        "input_scale", "score_transform", "direction", "threshold",
        "artifact_sha256",
    }
    if set(runtime_head) != required or runtime_head.get("schema") != HEAD_SCHEMA:
        raise ValueError("prefill_head.json does not match the runtime schema")
    for name in ("weights", "input_mean", "input_scale"):
        if not np.array_equal(np.asarray(runtime_head[name], dtype=np.float64), arrays[name]):
            raise ValueError(f"prefill_head.json {name} differs from head.npz")
    if float(runtime_head["bias"]) != float(arrays["bias"]):
        raise ValueError("prefill_head.json bias differs from head.npz")
    return runtime_head


def recalibrate_prefill_bundle(
    rows: Sequence[Mapping[str, Any]],
    source_dir: Path,
    output_dir: Path,
    *,
    calibration_dataset_sha256: str,
) -> dict[str, Any]:
    """Recalibrate a fitted head without changing its weights or normalizer."""

    import numpy as np

    source_head = load_prefill_head_bundle(source_dir)
    source_contract = json.loads(
        (source_dir / "feature_contract.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (source_dir / "fit_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") not in {
        "fitted_provisional_calibration",
        "fitted_and_calibrated",
    }:
        raise ValueError("source bundle does not contain a fitted Prefill head")
    with np.load(source_dir / "head.npz", allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in ("weights", "bias", "input_mean", "input_scale")
        }
    provenance = _provenance(rows)
    if provenance != source_contract.get("checkpoint_binding"):
        raise ValueError("recalibration checkpoint binding differs from fitted head")
    calibration_rows = _prefill_rows(rows, "calibration", require_label=False)
    margin_rows = _margin_rows(rows, "calibration")
    if not calibration_rows:
        raise ValueError("recalibration rows contain no Prefill calibration features")
    identity = _feature_identity(calibration_rows)
    expected_identity = {
        "resolved_layer": source_contract.get("resolved_layer"),
        "dimension": source_contract.get("dimension"),
        "model_binding": source_contract.get("model_binding"),
        "tokenizer_binding": source_contract.get("tokenizer_binding"),
        "stored_input_dtype": source_contract.get("stored_input_dtype"),
    }
    if identity != expected_identity:
        raise ValueError("recalibration feature identity differs from fitted head")
    if any(
        arrays[name].shape != (identity["dimension"],)
        for name in ("weights", "input_mean", "input_scale")
    ) or arrays["bias"].shape != ():
        raise ValueError("source head array shapes differ from the feature contract")
    if np.any(arrays["input_scale"] <= 0):
        raise ValueError("source head input_scale must be positive")

    cal_x = np.asarray(
        [row["prefill_hidden"] for row in calibration_rows], dtype=np.float64
    )
    logits = (
        ((cal_x - arrays["input_mean"]) / arrays["input_scale"])
        @ arrays["weights"]
        + float(arrays["bias"])
    )
    cal_scores = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0
    cal_scores[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    cal_scores[~nonnegative] = exponent / (1.0 + exponent)
    prefill_calibration = calibrate_exact_fraction(
        cal_scores.tolist(),
        direction="at_or_above",
        keys=[
            f"{row['task_id']}/{row['decision_key']}" for row in calibration_rows
        ],
    )
    margin_calibration = calibrate_exact_fraction(
        [row["first_name_top2_logprob_margin"] for row in margin_rows],
        direction="at_or_below",
        keys=[f"{row['task_id']}/{row['decision_key']}" for row in margin_rows],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale(output_dir)
    _write_margin_artifact(
        output_dir,
        margin_calibration,
        dataset_sha256=calibration_dataset_sha256,
        provenance=provenance,
    )
    calibration_tasks = {
        row.get("task_id") for row in rows if row.get("split") == "calibration"
    }
    calibration_groups = {
        row.get("group_ordinal")
        for row in rows
        if row.get("split") == "calibration"
    }
    calibration_complete = bool(
        len(calibration_tasks) == EXPECTED_CALIBRATION_TASKS
        and len(calibration_groups) == EXPECTED_CALIBRATION_GROUPS
    )
    deployment_ready = (
        prefill_calibration["status"] == "available" and calibration_complete
    )

    feature_contract = {
        key: value
        for key, value in source_contract.items()
        if key != "artifact_sha256"
    }
    fit_manifest = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"artifact_sha256", "artifact_digest_contract"}
    }
    fit_manifest.update(
        {
            "status": (
                "fitted_and_calibrated"
                if deployment_ready
                else "fitted_provisional_calibration"
            ),
            "deployment_status": (
                "runtime_head_ready"
                if deployment_ready
                else "provisional_not_for_deployment"
            ),
            "calibration_collection_complete": calibration_complete,
            "calibration_prefill_rows": len(calibration_rows),
            "calibration_margin_rows": len(margin_rows),
            "observed_tasks": {
                **dict(source_manifest.get("observed_tasks", {})),
                "calibration": len(calibration_tasks),
            },
            "observed_groups": {
                **dict(source_manifest.get("observed_groups", {})),
                "calibration": len(calibration_groups),
            },
            "calibration_known_labels": sum(
                row.get("label") in {0, 1} for row in calibration_rows
            ),
            "calibration_unknown_labels": sum(
                row.get("label") is None for row in calibration_rows
            ),
            "training_head_reused_without_refit": True,
            "source_artifact_sha256": source_head["artifact_sha256"],
            "source_dataset_sha256": source_manifest["dataset_sha256"],
            "recalibration_dataset_sha256": calibration_dataset_sha256,
            "recalibration_label_counts": dict(sorted(Counter(
                "unknown" if row.get("label") is None else str(row.get("label"))
                for row in rows
            ).items())),
            "recalibration_known_label_kind_counts": dict(sorted(Counter(
                str(row.get("label_kind"))
                for row in rows
                if row.get("label") in {0, 1}
            ).items())),
        }
    )
    fit_manifest["collection_complete"] = bool(
        fit_manifest.get("training_collection_complete") and calibration_complete
    )
    fit_manifest["collection_scope"] = (
        "complete_r002_data_plan"
        if fit_manifest["collection_complete"]
        else "fixed_training_snapshot_with_complete_calibration"
        if calibration_complete
        else "fixed_training_snapshot_with_partial_calibration"
    )
    calibration = {
        "schema": "a-history-r002-detector-calibration-v1",
        "prefill": prefill_calibration,
        "margin": margin_calibration,
        "recalibration": {
            "weights_refit": False,
            "source_artifact_sha256": source_head["artifact_sha256"],
            "calibration_dataset_sha256": calibration_dataset_sha256,
        },
    }
    core = _bundle_core(arrays, feature_contract, fit_manifest, calibration)
    artifact_sha256 = _sha256_json(core)
    fit_manifest["artifact_sha256"] = artifact_sha256
    fit_manifest["artifact_digest_contract"] = core["digest_scope"]
    feature_contract["artifact_sha256"] = artifact_sha256
    calibration["artifact_sha256"] = artifact_sha256

    np.savez(output_dir / "head.npz", **arrays)
    write_json(output_dir / "feature_contract.json", feature_contract)
    write_json(output_dir / "fit_manifest.json", fit_manifest)
    write_json(output_dir / "calibration.json", calibration)
    runtime_head = {
        **source_head,
        "threshold": prefill_calibration["threshold"],
        "artifact_sha256": artifact_sha256,
    }
    runtime_name = (
        "prefill_head.json"
        if deployment_ready
        else "prefill_head.provisional.json"
    )
    write_json(output_dir / runtime_name, runtime_head)
    return fit_manifest


def _collect_command(args: argparse.Namespace) -> int:
    add_bfcl_root(args.bfcl_root)
    rows, summary = collect_decision_rows(
        returned_roots=args.returned_root,
        data_plan_path=args.data_plan,
        checkpoint_binding=args.checkpoint_binding,
        ratio=args.ratio,
    )
    write_jsonl(args.output / "decision_rows.jsonl", rows)
    write_json(args.output / "label_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _fit_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.rows)
    result = fit_prefill_bundle(
        rows,
        args.output,
        dataset_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] != "not_fitted" else 2


def _recalibrate_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.rows)
    result = recalibrate_prefill_bundle(
        rows,
        args.source,
        args.output,
        calibration_dataset_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="label returned r002 task shards")
    collect.add_argument("--returned-root", action="append", type=Path, required=True)
    collect.add_argument("--data-plan", type=Path, required=True)
    collect.add_argument("--checkpoint-binding", type=Path, required=True)
    collect.add_argument("--ratio", type=int, choices=[SUPPORTED_RATIO], required=True)
    collect.add_argument("--bfcl-root", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.set_defaults(run=_collect_command)
    fit = commands.add_parser("fit", help="fit and calibrate the Prefill detector")
    fit.add_argument("--rows", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(run=_fit_command)
    recalibrate = commands.add_parser(
        "recalibrate", help="calibrate an existing fitted head without refitting"
    )
    recalibrate.add_argument("--rows", type=Path, required=True)
    recalibrate.add_argument("--source", type=Path, required=True)
    recalibrate.add_argument("--output", type=Path, required=True)
    recalibrate.set_defaults(run=_recalibrate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
