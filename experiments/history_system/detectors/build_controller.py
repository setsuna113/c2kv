"""Materialize a deployable Prefill recovery controller from a fitted bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .prefill_head import HEAD_SCHEMA, load_prefill_head_bundle


CONTROLLER_RECEIPT_SCHEMA = "a-history-r002-controller-materialization-v1"
FINAL_FIT_STATUS = "fitted_and_calibrated"
FINAL_DEPLOYMENT_STATUS = "runtime_head_ready"
FINAL_HEAD_NAME = "prefill_head.json"
PROVISIONAL_HEAD_NAME = "prefill_head.provisional.json"
FEATURE_CONTRACT_SCHEMA = "a-history-r002-prefill-feature-contract-v1"
FIT_MANIFEST_SCHEMA = "a-history-r002-prefill-fit-manifest-v1"
CALIBRATION_SCHEMA = "a-history-r002-detector-calibration-v1"
SHADOW_SCHEMA = "event-native-shadow-features-v1"
CHECKPOINT_HASH_KIND = "config_json"
REQUIRED_CHECKPOINT = {"selected_arm": "C", "selected_step": 1000, "ratio": 8}
REQUIRED_BASE_CONTROLLER = {
    "source_index_max_events": 12,
    "predictor_prompt_token_cap": 2048,
    "predictor_completion_token_cap": 256,
    "latest_complete_tool_protection": "budgeted",
    "observed_entity_slot_policy": "same-complete-event-reference-bridge-only-v1",
}
ONE_FIFTH = {"numerator": 1, "denominator": 5}
TASK_GENERATION_LIMIT = 96


def _runtime_module():
    history_system = Path(__file__).resolve().parents[1]
    history_memory_source = str(history_system / "runtime" / "python")
    if history_memory_source not in sys.path:
        sys.path.insert(0, history_memory_source)
    from experiments.history_system.runtime.benchmarks.memory_runtime import (
        event_native_recovery,
    )

    return event_native_recovery


def _read_json(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _validate_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_checkpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("status") != "selected":
        raise ValueError("checkpoint binding must have status=selected")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("checkpoint binding requires a nonempty path")
    digest = _validate_digest(value.get("config_sha256"), "checkpoint config_sha256")
    for name, expected in REQUIRED_CHECKPOINT.items():
        if value.get(name) != expected:
            raise ValueError(f"checkpoint binding must keep {name}={expected!r}")
    return {
        "status": "selected",
        "path": path,
        "config_sha256": digest,
        **REQUIRED_CHECKPOINT,
    }


def _validate_base_controller(value: Mapping[str, Any]) -> None:
    if "post_draft_recovery" in value:
        raise ValueError("base C0 controller already contains post_draft_recovery")
    for name, expected in REQUIRED_BASE_CONTROLLER.items():
        if value.get(name) != expected:
            raise ValueError(f"base C0 controller must keep {name}={expected!r}")


def _validate_deployment_bundle(
    bundle: Path,
    runtime_head: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_head = bundle / FINAL_HEAD_NAME
    provisional_head = bundle / PROVISIONAL_HEAD_NAME
    if not final_head.is_file() or provisional_head.exists():
        raise ValueError("deployment requires final prefill_head.json and rejects provisional heads")

    feature = _read_json(bundle / "feature_contract.json", "feature contract")
    fit = _read_json(bundle / "fit_manifest.json", "fit manifest")
    calibration = _read_json(bundle / "calibration.json", "calibration")

    if feature.get("schema") != FEATURE_CONTRACT_SCHEMA:
        raise ValueError("feature contract schema is unsupported")
    if fit.get("schema") != FIT_MANIFEST_SCHEMA:
        raise ValueError("fit manifest schema is unsupported")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("calibration schema is unsupported")
    if fit.get("status") != FINAL_FIT_STATUS:
        raise ValueError(f"fit manifest must have status={FINAL_FIT_STATUS}")
    if fit.get("deployment_status") != FINAL_DEPLOYMENT_STATUS:
        raise ValueError(
            f"fit manifest must have deployment_status={FINAL_DEPLOYMENT_STATUS}"
        )
    if fit.get("calibration_collection_complete") is not True:
        raise ValueError("fit manifest calibration collection is incomplete")

    prefill_calibration = calibration.get("prefill")
    if not isinstance(prefill_calibration, Mapping):
        raise ValueError("calibration.prefill must be an object")
    if prefill_calibration.get("status") != "available":
        raise ValueError("calibration.prefill must be available")
    if prefill_calibration.get("direction") != "at_or_above":
        raise ValueError("calibration.prefill direction must be at_or_above")
    calibration_threshold = prefill_calibration.get("threshold")
    if (
        isinstance(calibration_threshold, bool)
        or not isinstance(calibration_threshold, (int, float))
        or not math.isfinite(float(calibration_threshold))
        or float(calibration_threshold) != float(runtime_head["threshold"])
    ):
        raise ValueError("calibration.prefill threshold differs from the runtime head")

    if runtime_head.get("schema") != HEAD_SCHEMA:
        raise ValueError("runtime head schema is unsupported")
    if feature.get("shadow_schema") != SHADOW_SCHEMA:
        raise ValueError("feature contract shadow schema is unsupported")
    if feature.get("feature") != runtime_head.get("feature"):
        raise ValueError("feature contract differs from the runtime head feature")
    if feature.get("resolved_layer") != runtime_head.get("layer"):
        raise ValueError("feature contract differs from the runtime head layer")
    if feature.get("dimension") != len(runtime_head["weights"]):
        raise ValueError("feature contract differs from the runtime head dimension")

    if shadow.get("enabled") is not True:
        raise ValueError("shadow feature config must be enabled")
    requested_layer = shadow.get("prefill_layer")
    if type(requested_layer) is not int:
        raise ValueError("shadow feature config requires an integer prefill_layer")
    if feature.get("requested_layer") != requested_layer:
        raise ValueError("shadow prefill_layer differs from the fitted feature contract")

    expected_provenance = {
        "checkpoint_binding_status": "selected",
        "checkpoint_hash": checkpoint["config_sha256"],
        "checkpoint_hash_kind": CHECKPOINT_HASH_KIND,
        "checkpoint_config_sha256": checkpoint["config_sha256"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_selected_arm": checkpoint["selected_arm"],
        "checkpoint_selected_step": checkpoint["selected_step"],
        "ratio": checkpoint["ratio"],
    }
    for document_name, document in (
        ("fit manifest", fit),
        ("feature contract", feature),
    ):
        binding = document.get("checkpoint_binding")
        if not isinstance(binding, Mapping) or dict(binding) != expected_provenance:
            raise ValueError(f"{document_name} checkpoint binding differs from deployment")
    if feature.get("model_binding") != checkpoint["path"]:
        raise ValueError("feature contract model binding differs from deployment checkpoint")
    if shadow.get("model_binding") is not None and shadow["model_binding"] != checkpoint["path"]:
        raise ValueError("shadow model binding differs from deployment checkpoint")
    tokenizer = shadow.get("tokenizer_binding")
    if tokenizer is not None and tokenizer != feature.get("tokenizer_binding"):
        raise ValueError("shadow tokenizer binding differs from the fitted feature contract")
    return feature, fit, calibration


def materialize_controller(
    *,
    base_controller_path: Path,
    head_bundle: Path,
    checkpoint_binding_path: Path,
    shadow_feature_config_path: Path,
    deployment_profile: str,
    controller_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Validate all deployment sources and write one C0 plus Prefill controller."""

    if not isinstance(deployment_profile, str) or not deployment_profile.strip():
        raise ValueError("deployment_profile must be a nonempty explicit string")
    paths = {
        "base_controller": base_controller_path.resolve(),
        "head_bundle": head_bundle.resolve(),
        "checkpoint_binding": checkpoint_binding_path.resolve(),
        "shadow_feature_config": shadow_feature_config_path.resolve(),
        "controller_output": controller_output.resolve(),
        "receipt_output": receipt_output.resolve(),
    }
    if paths["controller_output"] == paths["receipt_output"]:
        raise ValueError("controller and receipt outputs must be different files")
    source_files = {
        paths["base_controller"],
        paths["checkpoint_binding"],
        paths["shadow_feature_config"],
        *(paths["head_bundle"] / name for name in (
            "head.npz",
            "feature_contract.json",
            "fit_manifest.json",
            "calibration.json",
            FINAL_HEAD_NAME,
            PROVISIONAL_HEAD_NAME,
        )),
    }
    if paths["controller_output"] in source_files or paths["receipt_output"] in source_files:
        raise ValueError("materializer outputs must not overwrite deployment sources")

    base = _read_json(paths["base_controller"], "base C0 controller")
    checkpoint = _validate_checkpoint(
        _read_json(paths["checkpoint_binding"], "checkpoint binding")
    )
    shadow = _read_json(paths["shadow_feature_config"], "shadow feature config")
    runtime_head = load_prefill_head_bundle(paths["head_bundle"])
    feature, fit, calibration = _validate_deployment_bundle(
        paths["head_bundle"], runtime_head, checkpoint, shadow
    )
    _validate_base_controller(base)

    recovery_module = _runtime_module()
    recovery = {
        "schema": recovery_module.E1_RECOVERY_VERSION,
        "gate": "prefill_linear_head",
        "random_seed": 0,
        "random_probability": dict(ONE_FIFTH),
        "quota": dict(ONE_FIFTH),
        "task_generation_limit": TASK_GENERATION_LIMIT,
        "prefill_head": dict(runtime_head),
    }
    recovery_module._parse_config(recovery)
    controller = {**base, recovery_module.E1_RECOVERY_CONFIG_KEY: recovery}
    _write_json(paths["controller_output"], controller)

    bundle_names = (
        "head.npz",
        "feature_contract.json",
        "fit_manifest.json",
        "calibration.json",
        FINAL_HEAD_NAME,
    )
    receipt = {
        "schema": CONTROLLER_RECEIPT_SCHEMA,
        "status": "materialized_not_launched",
        "deployment_profile": deployment_profile.strip(),
        "controller": {
            "path": str(paths["controller_output"]),
            "sha256": _sha256(paths["controller_output"]),
        },
        "sources": {
            "base_controller": {
                "path": str(paths["base_controller"]),
                "sha256": _sha256(paths["base_controller"]),
            },
            "head_bundle": {
                "path": str(paths["head_bundle"]),
                "artifact_sha256": runtime_head["artifact_sha256"],
                "files": {
                    name: _sha256(paths["head_bundle"] / name) for name in bundle_names
                },
            },
            "checkpoint_binding": {
                "path": str(paths["checkpoint_binding"]),
                "sha256": _sha256(paths["checkpoint_binding"]),
                "binding": checkpoint,
            },
            "shadow_feature_config": {
                "path": str(paths["shadow_feature_config"]),
                "sha256": _sha256(paths["shadow_feature_config"]),
            },
        },
        "validated": {
            "bundle_semantic_digest": True,
            "final_calibration": True,
            "checkpoint_binding": True,
            "shadow_feature_binding": True,
            "event_native_recovery_config": True,
            "feature": feature["feature"],
            "requested_layer": feature["requested_layer"],
            "resolved_layer": feature["resolved_layer"],
            "dimension": feature["dimension"],
            "fit_status": fit["status"],
            "calibration_status": calibration["prefill"]["status"],
        },
        "model_calls": 0,
        "scorer_calls": 0,
        "network_calls": 0,
    }
    _write_json(paths["receipt_output"], receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-controller", type=Path, required=True)
    parser.add_argument("--head-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--shadow-feature-config", type=Path, required=True)
    parser.add_argument("--deployment-profile", required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize_controller(
        base_controller_path=args.base_controller,
        head_bundle=args.head_bundle,
        checkpoint_binding_path=args.checkpoint_binding,
        shadow_feature_config_path=args.shadow_feature_config,
        deployment_profile=args.deployment_profile,
        controller_output=args.controller_output,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
