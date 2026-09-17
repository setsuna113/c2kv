#!/usr/bin/env python3
"""Validate one next-compression checkpoint and launch its pinned SGLang engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TOOL_ROOT = REPOSITORY_ROOT / "scripts" / "next_compression"
sys.path.insert(0, str(SOURCE_TOOL_ROOT))

from sglang_source import verify_source  # noqa: E402


TRAINING_PROFILE = "next-compression-base-query-v1"
VARIANTS = ("H0", "H1", "H2", "H3", "T0", "T1")
RENDER_PROFILES = {
    "H0": "event-native-evidence-v1",
    "H1": "event-native-evidence-v1",
    "H2": "a-event-native-s0-v1",
    "H3": "a-event-native-s0-v1",
    "T0": "next-compression-tool-explicit-protocol-v2",
    "T1": "next-compression-tool-explicit-protocol-v2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _manifest_profile(manifest: Mapping[str, Any], field: str) -> Any:
    value = manifest.get(field)
    preparation = manifest.get("preparation")
    if value is None and isinstance(preparation, Mapping):
        value = preparation.get(field)
    return value


def validate_checkpoint(
    checkpoint: str | Path,
    training_manifest: str | Path,
) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    manifest_path = Path(training_manifest).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
    config = read_json(config_path)
    manifest = read_json(manifest_path)
    variant = config.get("history_memory_variant")
    if variant not in VARIANTS:
        raise ValueError("Checkpoint has no supported next-compression variant")
    domain = "history" if str(variant).startswith("H") else "tool"
    expected_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "history_memory_training_profile": TRAINING_PROFILE,
        "history_memory_compression_domain": domain,
        "history_memory_supported_ratios": [8, 12],
        "history_memory_normal_query": "base",
        "history_memory_render_profile": RENDER_PROFILES[variant],
        "history_memory_loss_profile": (
            "decision-mean-critical-token-weighted-ce-v1"
            if variant == "H3"
            else "decision-mean-complete-ce-v1"
        ),
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_residual_type": "embed-mean",
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise ValueError(
                f"Checkpoint has incompatible {field}: {config.get(field)!r} != {expected!r}"
            )
    if config.get("pic_enabled", False) is not False:
        raise ValueError("Next-compression SGLang serving does not support PIC checkpoints")
    max_position_embeddings = config.get("max_position_embeddings")
    if (
        isinstance(max_position_embeddings, bool)
        or not isinstance(max_position_embeddings, int)
        or max_position_embeddings <= 0
    ):
        raise ValueError("Checkpoint has no positive integer max_position_embeddings")
    corpus_identity = config.get("history_memory_corpus_identity")
    if not isinstance(corpus_identity, str) or len(corpus_identity) != 64:
        raise ValueError("Checkpoint is missing a full training-manifest identity")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != corpus_identity:
        raise ValueError("Training manifest SHA256 differs from checkpoint corpus identity")
    expected_manifest = {
        "training_profile": TRAINING_PROFILE,
        "variant": variant,
        "compression_domain": domain,
        "ratios": [8, 12],
        "render_profile": config["history_memory_render_profile"],
        "loss_profile": config["history_memory_loss_profile"],
    }
    for field, expected in expected_manifest.items():
        actual = _manifest_profile(manifest, field)
        if actual != expected:
            raise ValueError(
                f"Training manifest differs from checkpoint for {field}: "
                f"{actual!r} != {expected!r}"
            )

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file() and single_path.is_file():
        raise ValueError("Checkpoint has ambiguous single and sharded safetensors layouts")
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no weight_map")
        shards: set[str] = set()
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(filename, str):
                raise ValueError("Safetensors weight_map entries must be strings")
            relative = Path(filename)
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(f"Invalid safetensors shard path: {filename!r}")
            shard = root / relative
            if not shard.is_file():
                raise FileNotFoundError(f"Safetensors shard is missing: {shard}")
            shards.add(filename)
        weights = [str((root / filename).resolve()) for filename in sorted(shards)]
        layout = "sharded"
    elif single_path.is_file():
        weights = [str(single_path.resolve())]
        layout = "single"
    else:
        raise FileNotFoundError("Checkpoint has no single or indexed safetensors weights")
    if not (root / "tokenizer_config.json").is_file() or not any(
        (root / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise FileNotFoundError("Checkpoint is missing its local tokenizer files")
    config_sha256 = sha256_file(config_path)
    return {
        "checkpoint": str(root),
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": config_sha256,
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": manifest_sha256,
        "training_profile": TRAINING_PROFILE,
        "variant": variant,
        "compression_domain": domain,
        "supported_ratios": [8, 12],
        "max_position_embeddings": max_position_embeddings,
        "weight_layout": layout,
        "weight_files": weights,
        "weight_version": f"next-compression:{config_sha256}",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = validate_checkpoint(args.checkpoint, args.training_manifest)
    source = verify_source(args.sglang_source)
    python = Path(args.python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"SGLang Python interpreter does not exist: {python}")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.tp_size <= 0 or args.max_running_requests <= 0:
        raise ValueError("--tp-size and --max-running-requests must be positive")
    for name in ("mem_fraction_static", "c2kv_pool_fraction"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in ("c2kv_max_tokens", "max_total_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    context_length = (
        checkpoint["max_position_embeddings"]
        if args.context_length is None
        else args.context_length
    )
    if context_length <= 0:
        raise ValueError("--context-length must be positive")

    if args.device == "cuda":
        cuda_execution = args.cuda_execution or "flashinfer-graph"
        attention_backend = (
            "flashinfer" if cuda_execution == "flashinfer-graph" else "torch_native"
        )
        cuda_graph = cuda_execution == "flashinfer-graph"
        page_size = 1
    else:
        if args.cuda_execution is not None:
            raise ValueError("--cuda-execution is only valid with --device cuda")
        cuda_execution = None
        attention_backend = "ascend"
        cuda_graph = False
        page_size = 128
    command = [
        str(python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        checkpoint["checkpoint"],
        "--tokenizer-path",
        checkpoint["checkpoint"],
        "--served-model-name",
        args.served_model_name,
        "--weight-version",
        checkpoint["weight_version"],
        "--model-impl",
        "sglang",
        "--device",
        args.device,
        "--attention-backend",
        attention_backend,
        "--dtype",
        args.dtype,
        "--tp-size",
        str(args.tp_size),
        "--enable-c2kv",
        "--c2kv-gist-type",
        "dynamic-interleave",
        "--c2kv-gist-param",
        "qkv",
        "--c2kv-query-proj",
        "base",
        "--c2kv-pool-fraction",
        str(args.c2kv_pool_fraction),
        "--c2kv-max-tokens",
        str(args.c2kv_max_tokens),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(context_length),
        "--max-running-requests",
        str(args.max_running_requests),
        "--page-size",
        str(page_size),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--disable-radix-cache",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not cuda_graph:
        command.append("--disable-cuda-graph")
    if args.device == "cuda":
        command.extend(["--disable-overlap-schedule", "--disable-piecewise-cuda-graph"])
    if args.c2kv_shadow_feature_layer is not None:
        command.extend(
            [
                "--c2kv-shadow-feature-layer",
                str(args.c2kv_shadow_feature_layer),
                "--enable-return-hidden-states",
            ]
        )
    environment = {
        "PYTHONPATH": str(Path(args.sglang_source).resolve() / "python"),
        "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return {
        "schema": "next-compression-sglang-launch-v1",
        "status": "ready_to_exec" if args.run else "dry_run",
        "checkpoint": checkpoint,
        "source": source,
        "server": {
            "base_url": f"http://{args.host}:{args.port}",
            "openai_base_url": f"http://{args.host}:{args.port}/v1",
            "served_model_name": args.served_model_name,
            "device": args.device,
            "cuda_execution": cuda_execution,
            "attention_backend": attention_backend,
            "page_size": page_size,
            "context_length": context_length,
            "context_length_source": (
                "checkpoint.max_position_embeddings"
                if args.context_length is None
                else "command_line"
            ),
            "cuda_graph": cuda_graph,
            "foreground_exec": True,
        },
        "environment": environment,
        "command": command,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument(
        "--cuda-execution",
        choices=("flashinfer-graph", "torch-native-eager"),
        help=(
            "CUDA-only execution profile; defaults to flashinfer-graph. "
            "torch-native-eager is the explicit slow reference path."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=34010)
    parser.add_argument("--served-model-name", default="c2kv-next")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--c2kv-pool-fraction", type=float, default=0.05)
    parser.add_argument("--c2kv-max-tokens", type=int, default=65536)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--chunked-prefill-size", type=int, default=256)
    parser.add_argument("--c2kv-shadow-feature-layer", type=int)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    if not args.run:
        return 0
    environment = os.environ.copy()
    for key, value in plan["environment"].items():
        if key == "PYTHONPATH" and environment.get(key):
            environment[key] = value + os.pathsep + environment[key]
        else:
            environment[key] = value
    os.chdir(Path(args.sglang_source).resolve())
    os.execvpe(plan["command"][0], plan["command"], environment)
    raise AssertionError("os.execvpe returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
