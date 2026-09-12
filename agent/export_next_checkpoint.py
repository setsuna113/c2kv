#!/usr/bin/env python3
"""Export a lightweight gist package or materialize it with the pinned base."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from next_compression.common import TRAINING_PROFILE, VARIANTS, sha256_file


EXPORT_SCHEMA = "next-compression-gist-export-v1"
MATERIALIZATION_SCHEMA = "next-compression-materialization-v1"
BASE_REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GIST_FILE = "c2kv-gist.safetensors"
MANIFEST_FILE = "manifest.json"
BASE_RECEIPT_FILE = "C2KV_SOURCE_REVISION.json"
MATERIALIZATION_RECEIPT_FILE = "C2KV_MATERIALIZATION.json"
_GIST_PROJECTIONS = (".gist_q_proj.", ".gist_k_proj.", ".gist_v_proj.")
_BASE_ARCHITECTURE_FIELDS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "tie_word_embeddings",
)


def _is_gist_parameter(name: str) -> bool:
    return name.startswith("model.gist_embed_tokens.") or any(
        marker in name for marker in _GIST_PROJECTIONS
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _empty_destination(path: str | Path) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _pending_destination(destination: Path) -> Path:
    pending = destination.with_name(destination.name + f".pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"Stale pending destination exists: {pending}")
    pending.mkdir()
    return pending


def _validate_checkpoint_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("history_memory_training_profile") != TRAINING_PROFILE:
        raise ValueError("Checkpoint is not a next-compression training profile")
    variant = config.get("history_memory_variant")
    if variant not in VARIANTS:
        raise ValueError("Checkpoint has no valid next-compression variant")
    domain = "history" if str(variant).startswith("H") else "tool"
    expected = {
        "history_memory_compression_domain": domain,
        "history_memory_supported_ratios": [8, 12],
        "history_memory_normal_query": "base",
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_residual_type": "embed-mean",
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Checkpoint has incompatible {field}")
    required_metadata = (
        "history_memory_corpus_identity",
        "history_memory_render_profile",
        "history_memory_loss_profile",
        "history_memory_initialization_id",
        "history_memory_initialization_group",
        "history_memory_initialization_kind",
    )
    if any(not isinstance(config.get(field), str) or not config[field] for field in required_metadata):
        raise ValueError("Checkpoint is missing next-compression provenance metadata")
    if config["history_memory_initialization_group"] != domain:
        raise ValueError("Checkpoint initialization group differs from its variant")
    if config["history_memory_initialization_kind"] not in {
        "fresh-base",
        "warm-start",
    }:
        raise ValueError("Checkpoint has an invalid initialization kind")
    expected_loss = (
        "decision-mean-critical-token-weighted-ce-v1"
        if variant == "H3"
        else "decision-mean-complete-ce-v1"
    )
    if config["history_memory_loss_profile"] != expected_loss:
        raise ValueError("Checkpoint loss profile differs from its variant")
    return {
        "training_profile": TRAINING_PROFILE,
        "variant": variant,
        "compression_domain": domain,
        "ratios": [8, 12],
        "normal_query": "base",
        "corpus_identity": config["history_memory_corpus_identity"],
        "render_profile": config["history_memory_render_profile"],
        "loss_profile": config["history_memory_loss_profile"],
        "initialization_id": config["history_memory_initialization_id"],
        "initialization_group": config["history_memory_initialization_group"],
        "initialization_kind": config["history_memory_initialization_kind"],
    }


def _safetensor_locations(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weight map: {index_path}")
        locations = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise ValueError("Safetensors weight map must contain string entries")
            path = (root / filename).resolve()
            if path.parent != root.resolve() or not path.is_file():
                raise ValueError(f"Invalid safetensors shard path: {filename!r}")
            locations[name] = path
        return locations
    single = root / "model.safetensors"
    if not single.is_file():
        raise ValueError(f"No safetensors model found in {root}")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single.resolve() for name in handle.keys()}


def _load_selected_tensors(
    locations: Mapping[str, Path], predicate
) -> dict[str, Any]:
    from safetensors import safe_open

    selected = {name: path for name, path in locations.items() if predicate(name)}
    tensors = {}
    for path in sorted(set(selected.values())):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name, location in selected.items():
                if location == path:
                    tensors[name] = handle.get_tensor(name)
    return tensors


def _file_integrity(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in excluded
    }


def _copy_tokenizer(checkpoint: Path, destination: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.save_pretrained(destination)


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    base_repository: str = BASE_REPOSITORY,
    base_revision: str = BASE_REVISION,
) -> dict[str, Any]:
    """Create a model-weight-light package containing only FP32 gist weights."""

    import torch
    from safetensors.torch import save_file

    checkpoint = Path(checkpoint).resolve()
    destination = _empty_destination(output_dir)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint config is missing: {config_path}")
    config = _read_json(config_path)
    metadata = _validate_checkpoint_config(config)
    locations = _safetensor_locations(checkpoint)
    gist = _load_selected_tensors(locations, _is_gist_parameter)
    if not gist:
        raise ValueError("Checkpoint has no gist tensors")

    pending = _pending_destination(destination)
    try:
        shutil.copy2(config_path, pending / "config.json")
        generation_config = checkpoint / "generation_config.json"
        if generation_config.is_file():
            shutil.copy2(generation_config, pending / generation_config.name)
        _copy_tokenizer(checkpoint, pending)
        fp32_gist = {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, tensor in gist.items()
        }
        save_file(fp32_gist, pending / GIST_FILE)
        base_receipt = {"repository": base_repository, "revision": base_revision}
        (pending / BASE_RECEIPT_FILE).write_text(
            json.dumps(base_receipt, indent=2) + "\n", encoding="utf-8"
        )
        files = _file_integrity(pending)
        manifest = {
            "schema": EXPORT_SCHEMA,
            "base": base_receipt,
            "checkpoint": metadata,
            "config_sha256": files["config.json"]["sha256"],
            "gist": {
                "path": GIST_FILE,
                "dtype": "float32",
                "parameter_names": sorted(fp32_gist),
                "parameter_count": sum(tensor.numel() for tensor in fp32_gist.values()),
                **files[GIST_FILE],
            },
            "files": files,
        }
        (pending / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        pending.rename(destination)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return manifest


def _validate_export(package: Path) -> dict[str, Any]:
    manifest = _read_json(package / MANIFEST_FILE)
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Unrecognized next-compression gist export")
    if manifest.get("base") != {
        "repository": BASE_REPOSITORY,
        "revision": BASE_REVISION,
    }:
        raise ValueError("Gist export does not bind the pinned base revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Gist export has no file integrity contract")
    for name, expected in files.items():
        path = (package / name).resolve()
        if path.parent != package.resolve() or not path.is_file():
            raise ValueError(f"Invalid export file path: {name!r}")
        actual = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if actual != expected:
            raise ValueError(f"Gist export file integrity failed: {name}")
    config = _read_json(package / "config.json")
    metadata = _validate_checkpoint_config(config)
    if metadata != manifest.get("checkpoint"):
        raise ValueError("Export manifest differs from config metadata")
    gist_info = manifest.get("gist")
    if not isinstance(gist_info, dict) or gist_info.get("dtype") != "float32":
        raise ValueError("Export must declare FP32 gist tensors")
    locations = {name: package / GIST_FILE for name in gist_info.get("parameter_names", ())}
    gist = _load_selected_tensors(locations, lambda _name: True)
    if set(gist) != set(gist_info.get("parameter_names", ())):
        raise ValueError("Export gist parameter list differs from its shard")
    if any(str(tensor.dtype) != "torch.float32" for tensor in gist.values()):
        raise ValueError("Export gist shard contains a non-FP32 tensor")
    return manifest


def _validate_base(
    base: Path,
    expected: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> dict[str, Path]:
    receipt = _read_json(base / BASE_RECEIPT_FILE)
    if receipt != expected:
        raise ValueError(f"Base source receipt differs: {receipt!r} vs {dict(expected)!r}")
    config = _read_json(base / "config.json")
    if config.get("model_type") != "qwen3" or config.get("architectures") != [
        "Qwen3ForCausalLM"
    ]:
        raise ValueError("Base directory is not the pinned Qwen3 causal LM")
    if "gist_param" in config or "history_memory_training_profile" in config:
        raise ValueError("Base directory must contain original non-gist config")
    for field in _BASE_ARCHITECTURE_FIELDS:
        if config.get(field) != checkpoint_config.get(field):
            raise ValueError(f"Base architecture differs for {field}")
    locations = _safetensor_locations(base)
    if any(_is_gist_parameter(name) for name in locations):
        raise ValueError("Base directory already contains gist tensors")
    return locations


def _link_or_copy(source: Path, destination: Path) -> tuple[str, int]:
    try:
        os.link(source, destination)
        return "hardlink", 0
    except OSError:
        shutil.copy2(source, destination)
        return "copy", destination.stat().st_size


def materialize_checkpoint(
    package_dir: str | Path,
    base_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Combine a validated gist export with the exact pinned base snapshot."""

    package = Path(package_dir).resolve()
    base = Path(base_dir).resolve()
    manifest = _validate_export(package)
    checkpoint_config = _read_json(package / "config.json")
    base_locations = _validate_base(base, manifest["base"], checkpoint_config)
    destination = _empty_destination(output_dir)
    pending = _pending_destination(destination)
    transfers = []
    copied_bytes = 0
    try:
        base_file_names: dict[Path, str] = {}
        for source in sorted(set(base_locations.values())):
            name = source.name
            if name in {GIST_FILE, "model.safetensors.index.json", "model.safetensors"}:
                name = "c2kv-base-" + name
            mode, copied = _link_or_copy(source, pending / name)
            copied_bytes += copied
            transfers.append({"source": source.name, "destination": name, "mode": mode})
            base_file_names[source] = name

        gist_source = package / GIST_FILE
        mode, copied = _link_or_copy(gist_source, pending / GIST_FILE)
        copied_bytes += copied
        transfers.append(
            {"source": GIST_FILE, "destination": GIST_FILE, "mode": mode}
        )

        for name in manifest["files"]:
            if name in {GIST_FILE, MANIFEST_FILE}:
                continue
            shutil.copy2(package / name, pending / name)

        gist_names = manifest["gist"]["parameter_names"]
        weight_map = {
            name: base_file_names[path] for name, path in base_locations.items()
        }
        overlap = set(weight_map) & set(gist_names)
        if overlap:
            raise ValueError(f"Base and gist export overlap: {sorted(overlap)}")
        weight_map.update({name: GIST_FILE for name in gist_names})
        total_size = sum(path.stat().st_size for path in set(base_locations.values()))
        total_size += gist_source.stat().st_size
        (pending / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema": MATERIALIZATION_SCHEMA,
            "base": manifest["base"],
            "export_manifest_sha256": sha256_file(package / MANIFEST_FILE),
            "output_config_sha256": sha256_file(pending / "config.json"),
            "transfers": transfers,
            "copy_fallback_bytes": copied_bytes,
            "weight_map_sha256": _canonical_sha256(weight_map),
        }
        (pending / MATERIALIZATION_RECEIPT_FILE).write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        pending.rename(destination)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return receipt


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Write a lightweight FP32 gist package")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output-dir", required=True)
    materialize_parser = commands.add_parser(
        "materialize", help="Combine a gist package with the exact pinned base"
    )
    materialize_parser.add_argument("--package-dir", required=True)
    materialize_parser.add_argument("--base-dir", required=True)
    materialize_parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = arguments(argv)
    if args.command == "export":
        result = export_checkpoint(args.checkpoint, args.output_dir)
    else:
        result = materialize_checkpoint(
            args.package_dir, args.base_dir, args.output_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
