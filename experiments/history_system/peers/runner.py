"""Freeze, preview, or execute the four r001 peer-completion manifests.

The production command construction and task-terminalization logic comes from
the hash-bound audited R5 templates.  The only template-source change replaces
the independent-holdout acceptance gate with a development-search stage gate.
No action in this module starts a model except the explicit ``run`` action.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG_SCHEMA = "a-history-system-peer-completion-config-v1"
FREEZE_SCHEMA = "a-history-system-peer-completion-freeze-v1"
INDEX_SCHEMA = "a-history-system-peer-r001-index-design-v1"
SAMPLE_LABEL = "preliminary, n=1"

R5_GATE = (
    '    if design.get("evaluation_stage") != "independent_holdout" or '
    'design.get("acceptance_parameters_frozen") is not True:\n'
    '        raise ValueError("R5 requires frozen acceptance parameters before execution")'
)
DEVELOPMENT_GATE = (
    '    if design.get("evaluation_stage") != "development_search":\n'
    '        raise ValueError("peer completion requires development_search")'
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def set_repo_root(path: Path) -> None:
    """Set the root against which all hash-bound config paths are resolved."""
    global REPO_ROOT
    REPO_ROOT = path.resolve()
    if not REPO_ROOT.is_dir():
        raise FileNotFoundError(f"peer repo root does not exist: {REPO_ROOT}")


def _bound_path(binding: Mapping[str, Any], label: str) -> Path:
    path_value = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{label} requires path and sha256")
    path = _repo_path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"hash changed for {label}: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.resolve())
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected peer completion config schema")
    if config.get("evaluation_stage") != "development_search":
        raise ValueError("peer completion config must be development_search")
    if config.get("method") not in {"raw", "text", "full", "hiagent"}:
        raise ValueError("unknown peer method")
    for key in (
        "source_design",
        "source_runner",
        "audited_template",
        "template_cpu_validation",
        "base_task_manifest",
        "r001_task_manifest",
        "remote_scorer_lineage",
    ):
        _bound_path(config[key], key)
    if config.get("template_patch") != {
        "remove_exactly": R5_GATE,
        "add_exactly": DEVELOPMENT_GATE,
    }:
        raise ValueError("peer config template patch is not the one audited change")
    return config


def _patch_template(source: str) -> str:
    if source.count(R5_GATE) != 1:
        raise ValueError("audited template must contain exactly one R5 acceptance gate")
    patched = source.replace(R5_GATE, DEVELOPMENT_GATE)
    if R5_GATE in patched or patched.count(DEVELOPMENT_GATE) != 1:
        raise ValueError("development-search gate patch failed")
    return patched


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_template(
    config: Mapping[str, Any], *, frozen_template: Path | None = None
) -> tuple[types.ModuleType, str]:
    original_runner = _bound_path(config["source_runner"], "source_runner")
    source_template = _bound_path(config["audited_template"], "audited_template")
    patched_source = _patch_template(source_template.read_text(encoding="utf-8"))
    if frozen_template is not None:
        if frozen_template.read_text(encoding="utf-8") != patched_source:
            raise ValueError("frozen template differs from the audited one-change patch")
        module_path = frozen_template
    else:
        module_path = source_template

    original = _load_module(
        original_runner,
        f"_peer_original_{config['method']}_{hashlib.sha1(str(original_runner).encode()).hexdigest()}",
    )
    module = types.ModuleType(f"_peer_template_{config['method']}")
    module.__file__ = str(module_path)
    module.__name__ = f"_peer_template_{config['method']}"
    exec(compile(patched_source, str(module_path), "exec"), module.__dict__)
    module.ROOT = original.ROOT
    module.HERE = original.HERE
    return module, patched_source


def _task_manifest(config: Mapping[str, Any], binding: str) -> tuple[Path, dict[str, Any]]:
    path = _bound_path(config[binding], binding)
    manifest = _read_json(path)
    tasks = manifest.get("task_ids")
    if not isinstance(tasks, list) or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid task IDs in {binding}")
    if manifest.get("fixed_denominator") != len(tasks):
        raise ValueError(f"fixed denominator changed in {binding}")
    return path, manifest


def build_design(
    config: Mapping[str, Any], *, selection_path: Path | None = None
) -> dict[str, Any]:
    source_path = _bound_path(config["source_design"], "source_design")
    source = _read_json(source_path)
    manifest_path, manifest = _task_manifest(config, "base_task_manifest")
    task_ids = manifest["task_ids"]
    if manifest.get("stage") != "development_search" or any(
        not isinstance(task, str) or not task.startswith("multi_turn_base_")
        for task in task_ids
    ):
        raise ValueError("peer completion manifest must contain only r001 Base tasks")
    expected = [
        task
        for task in _task_manifest(config, "r001_task_manifest")[1]["task_ids"]
        if task.startswith("multi_turn_base_")
    ]
    if task_ids != expected:
        raise ValueError("peer completion tasks differ from the ordered r001 Base subset")

    design = copy.deepcopy(source)
    design["schema"] = config["template_design_schema"]
    design["candidate_id"] = config["candidate_id"]
    design["run_id_template"] = config["run_id"]
    design["status"] = "frozen"
    design["evaluation_stage"] = "development_search"
    design["acceptance_parameters_frozen"] = False
    design["task_ids"] = task_ids
    design["limits"]["tasks"] = len(task_ids)
    design["task_manifest_sha256"] = _sha256(manifest_path)
    lineage = design["task_and_scorer_lineage"]
    resolved_selection = (selection_path or manifest_path).resolve()
    lineage["task_selection"] = str(resolved_selection)
    lineage["task_selection_sha256"] = _sha256(resolved_selection)
    lineage["selection_source_sha256"] = _sha256(manifest_path)
    lineage["selection_rule"] = manifest["selection"]
    lineage["source_bindings"] = copy.deepcopy(config["base_scorer_bindings"])
    design["peer_completion"] = {
        "schema": "a-history-system-r001-peer-completion-v1",
        "method": config["method"],
        "stage": "development_search",
        "sample_label": SAMPLE_LABEL,
        "missing_variant": "base",
        "new_cells": len(task_ids),
        "reused_variant": "long",
        "reused_cells": len(config["reused_long_task_ids"]),
        "source_design": config["source_design"],
        "audited_template": config["audited_template"],
        "template_cpu_validation": config["template_cpu_validation"],
        "comparison_scope": (
            "descriptive whole-system comparison after B500, greedy sampling, official "
            "scorer, and exact r001 manifest are matched"
        ),
        "strict_causal_claim": False,
    }
    return design


def build_index_design(config: Mapping[str, Any]) -> dict[str, Any]:
    source_path = _bound_path(config["source_design"], "source_design")
    source = _read_json(source_path)
    mixed_path, mixed = _task_manifest(config, "r001_task_manifest")
    index: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "candidate_id": config["candidate_id"],
        "peer_method": config["method"],
        "status": "frozen_index_only",
        "evaluation_stage": "development_search",
        "sample_label": SAMPLE_LABEL,
        "task_ids": mixed["task_ids"],
        "task_manifest_sha256": _sha256(mixed_path),
        "fixed_denominator": len(mixed["task_ids"]),
        "source_design": config["source_design"],
        "source_design_sha256": config["source_design"]["sha256"],
        "remote_scorer_lineage": config["remote_scorer_lineage"],
        "scorer_bindings": copy.deepcopy(config["r001_scorer_bindings"]),
        "result_composition": {
            "reused_long_result_roots": config["reused_long_result_roots"],
            "new_base_result_root": "<new-base10-returned-root>",
            "reused_long_cells": len(config["reused_long_task_ids"]),
            "new_base_cells": len(_task_manifest(config, "base_task_manifest")[1]["task_ids"]),
            "missing_is_unknown_not_failure": True,
        },
        "comparison_scope": {
            "kind": "descriptive_whole_system",
            "matched": [
                "B500 checkpoint identity",
                "greedy temperature=0 seed=0 sampling",
                "official scorer lineage",
                "exact ordered r001 mixed20 manifest",
            ],
            "strict_causal_claim": False,
        },
        "resource_accounting": copy.deepcopy(config["resource_accounting"]),
    }
    for key in ("checkpoint_selection", "checkpoint", "sampling", "policy_sampling"):
        if key in source:
            index[key] = copy.deepcopy(source[key])
    if "b0_contract" in source:
        index["b0_contract"] = copy.deepcopy(source["b0_contract"])
    return index


def _arguments(config: Mapping[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    defaults = config["preview_defaults"]
    if config["runner_kind"] == "baseline":
        return argparse.Namespace(
            checkpoint=args.checkpoint,
            output=args.output,
            benchmark_dir=args.benchmark_dir,
            python=args.python,
            bfcl_python=args.bfcl_python,
            port_base=args.port_base if args.port_base is not None else defaults["port_base"],
        )
    return argparse.Namespace(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        output=args.output,
        benchmark_dir=args.benchmark_dir,
        policy_sampling=args.policy_sampling,
        server_python=args.server_python,
        bfcl_python=args.bfcl_python,
        proxy_python=args.proxy_python,
        device=args.device,
        server_port_base=(
            args.server_port_base
            if args.server_port_base is not None
            else defaults["server_port_base"]
        ),
        proxy_port_base=(
            args.proxy_port_base
            if args.proxy_port_base is not None
            else defaults["proxy_port_base"]
        ),
    )


def preview(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(config_path)
    design = build_design(config)
    module, _ = _load_template(config)
    result = module.preview(design, _arguments(config, args))
    result["peer_completion"] = {
        "method": config["method"],
        "evaluation_stage": "development_search",
        "template_patch": "R5 acceptance gate replaced by development_search gate only",
        "source_design_sha256": config["source_design"]["sha256"],
        "model_requests": 0,
        "scorer_calls": 0,
        "network_calls": 0,
        "launches": 0,
    }
    return result


def freeze(config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"freeze output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        task_source = _bound_path(config["base_task_manifest"], "base_task_manifest")
        tasks_path = output_dir / "tasks.json"
        shutil.copyfile(task_source, tasks_path)
        design = build_design(config, selection_path=tasks_path)
        module, patched_source = _load_template(config)
        preview_result = module.preview(design, _arguments(config, args))

        template_path = output_dir / "runner.template.py"
        template_path.write_text(patched_source, encoding="utf-8", newline="\n")
        design_path = output_dir / "design.json"
        index_path = output_dir / "index.design.json"
        config_snapshot = output_dir / "config.snapshot.json"
        preview_path = output_dir / "preview.json"
        shutil.copyfile(config_path, config_snapshot)
        _save_json(design_path, design)
        _save_json(index_path, build_index_design(config))
        _save_json(preview_path, preview_result)
        receipt = {
            "schema": FREEZE_SCHEMA,
            "status": "frozen_cpu_preview_only_no_model_scorer_network_or_launch",
            "method": config["method"],
            "evaluation_stage": "development_search",
            "task_ids": design["task_ids"],
            "whole_task_denominator": len(design["task_ids"]),
            "source_design": config["source_design"],
            "source_runner": config["source_runner"],
            "audited_template": config["audited_template"],
            "template_cpu_validation": config["template_cpu_validation"],
            "template_change": {
                "removed": R5_GATE,
                "added": DEVELOPMENT_GATE,
                "other_template_source_changes": 0,
            },
            "files": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in (
                    tasks_path,
                    design_path,
                    index_path,
                    config_snapshot,
                    preview_path,
                    template_path,
                )
            },
            "model_requests": 0,
            "scorer_calls": 0,
            "network_calls": 0,
            "launches": 0,
        }
        _save_json(output_dir / "freeze.json", receipt)
        return receipt
    except BaseException as exc:
        partial_files = {}
        for path in output_dir.iterdir():
            if path.is_file():
                try:
                    partial_files[path.name] = {
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                except OSError:
                    pass
        failure = {
            "schema": "a-history-system-peer-completion-freeze-failure-v1",
            "status": "freeze_failed_partial_directory_preserved",
            "method": config.get("method"),
            "evaluation_stage": "development_search",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "partial_files": partial_files,
            "model_requests": 0,
            "scorer_calls": 0,
            "network_calls": 0,
            "launches": 0,
        }
        try:
            _save_json(output_dir / "freeze.failed.json", failure)
        except OSError:
            pass
        raise


def _load_frozen(directory: Path) -> tuple[dict[str, Any], dict[str, Any], types.ModuleType]:
    directory = directory.resolve()
    receipt = _read_json(directory / "freeze.json")
    if receipt.get("schema") != FREEZE_SCHEMA or receipt.get("status") != (
        "frozen_cpu_preview_only_no_model_scorer_network_or_launch"
    ):
        raise ValueError("invalid peer freeze receipt")
    for name, binding in receipt["files"].items():
        path = directory / name
        if _sha256(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            raise ValueError(f"frozen peer file changed: {name}")
    config = load_config(directory / "config.snapshot.json")
    design = _read_json(directory / "design.json")
    if design.get("evaluation_stage") != "development_search":
        raise ValueError("frozen peer design is not development_search")
    module, _ = _load_template(config, frozen_template=directory / "runner.template.py")
    return config, design, module


def run(frozen_dir: Path, args: argparse.Namespace) -> int:
    config, design, module = _load_frozen(frozen_dir)
    forwarded = _arguments(config, args)
    if config["runner_kind"] == "baseline":
        required = (forwarded.checkpoint, forwarded.output, forwarded.benchmark_dir)
    else:
        required = (
            forwarded.checkpoint,
            forwarded.output,
            forwarded.benchmark_dir,
            forwarded.policy_sampling,
            forwarded.server_python,
            forwarded.bfcl_python,
            forwarded.proxy_python,
            forwarded.device,
        )
    if not all(required):
        raise ValueError("run requires every method-specific execution argument")
    return int(module.run(design, forwarded))


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="root containing every relative hash-bound config path",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--python")
    parser.add_argument("--bfcl-python")
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--source-root")
    parser.add_argument("--policy-sampling")
    parser.add_argument("--server-python")
    parser.add_argument("--proxy-python")
    parser.add_argument("--device")
    parser.add_argument("--server-port-base", type=int)
    parser.add_argument("--proxy-port-base", type=int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("--config", type=Path, required=True)
    preview_parser.add_argument("--output-json", type=Path)
    _add_runtime_arguments(preview_parser)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    _add_runtime_arguments(freeze_parser)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--frozen-dir", type=Path, required=True)
    _add_runtime_arguments(run_parser)
    args = parser.parse_args(argv)
    if args.repo_root is not None:
        set_repo_root(args.repo_root)
    if args.action in {"preview", "freeze"}:
        args.config = _repo_path(str(args.config))

    if args.action == "preview":
        result = preview(args.config, args)
        if args.output_json is not None:
            _save_json(args.output_json, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.action == "freeze":
        result = freeze(args.config, args.output_dir, args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    return run(args.frozen_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())
