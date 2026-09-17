"""CPU-only contracts for the dedicated next-compression SGLang launcher."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import launch_next_sglang as launcher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "prepared" / "H3" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "training_profile": launcher.TRAINING_PROFILE,
                "variant": "H3",
                "compression_domain": "history",
                "ratios": [8, 12],
                "render_profile": launcher.RENDER_PROFILES["H3"],
                "loss_profile": "decision-mean-critical-token-weighted-ce-v1",
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32768,
        "history_memory_training_profile": launcher.TRAINING_PROFILE,
        "history_memory_variant": "H3",
        "history_memory_compression_domain": "history",
        "history_memory_supported_ratios": [8, 12],
        "history_memory_normal_query": "base",
        "history_memory_corpus_identity": _sha(manifest),
        "history_memory_render_profile": launcher.RENDER_PROFILES["H3"],
        "history_memory_loss_profile": "decision-mean-critical-token-weighted-ce-v1",
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_residual_type": "embed-mean",
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "part-1.safetensors").write_bytes(b"one")
    (checkpoint / "part-2.safetensors").write_bytes(b"two")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "part-1.safetensors",
                    "model.layers.0.gist_qkv_proj.weight": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, manifest


@pytest.mark.parametrize(
    ("device", "backend", "page_size", "cuda_graph", "cuda_execution"),
    (
        ("cuda", "flashinfer", "1", True, "flashinfer-graph"),
        ("npu", "ascend", "128", False, None),
    ),
)
def test_plan_binds_checkpoint_identity_and_platform_flags(
    tmp_path, monkeypatch, device, backend, page_size, cuda_graph, cuda_execution
):
    checkpoint, manifest = _fixture(tmp_path)
    source = tmp_path / "sglang"
    source.mkdir()
    monkeypatch.setattr(
        launcher,
        "verify_source",
        lambda path: {
            "status": "verified",
            "source": str(Path(path).resolve()),
            "base_revision": "8" * 40,
        },
    )
    args = launcher.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--training-manifest",
            str(manifest),
            "--sglang-source",
            str(source),
            "--python",
            sys.executable,
            "--device",
            device,
        ]
    )
    plan = launcher.build_plan(args)
    command = plan["command"]
    config_sha = _sha(checkpoint / "config.json")
    assert plan["status"] == "dry_run"
    assert plan["checkpoint"]["weight_layout"] == "sharded"
    assert plan["checkpoint"]["weight_version"] == f"next-compression:{config_sha}"
    assert command[command.index("--weight-version") + 1] == f"next-compression:{config_sha}"
    assert command[command.index("--attention-backend") + 1] == backend
    assert command[command.index("--page-size") + 1] == page_size
    assert plan["server"]["cuda_execution"] == cuda_execution
    assert plan["server"]["cuda_graph"] is cuda_graph
    assert command[command.index("--context-length") + 1] == "32768"
    assert plan["server"]["context_length"] == 32768
    assert plan["server"]["context_length_source"] == "checkpoint.max_position_embeddings"
    assert ("--disable-cuda-graph" in command) is (not cuda_graph)
    assert "--c2kv-query-proj" in command
    assert command[command.index("--c2kv-query-proj") + 1] == "base"
    assert plan["environment"]["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] == "1"
    if device == "cuda":
        assert "--disable-overlap-schedule" in command
        assert "--disable-piecewise-cuda-graph" in command
    else:
        assert "--disable-overlap-schedule" not in command


def test_cuda_torch_native_eager_is_an_explicit_reference_profile(tmp_path, monkeypatch):
    checkpoint, manifest = _fixture(tmp_path)
    source = tmp_path / "sglang"
    source.mkdir()
    monkeypatch.setattr(launcher, "verify_source", lambda path: {"status": "verified"})
    args = launcher.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--training-manifest",
            str(manifest),
            "--sglang-source",
            str(source),
            "--python",
            sys.executable,
            "--device",
            "cuda",
            "--cuda-execution",
            "torch-native-eager",
        ]
    )
    plan = launcher.build_plan(args)
    command = plan["command"]
    assert plan["server"]["cuda_execution"] == "torch-native-eager"
    assert plan["server"]["attention_backend"] == "torch_native"
    assert plan["server"]["cuda_graph"] is False
    assert command[command.index("--attention-backend") + 1] == "torch_native"
    assert "--disable-cuda-graph" in command


def test_cuda_execution_profile_is_rejected_for_npu(tmp_path, monkeypatch):
    checkpoint, manifest = _fixture(tmp_path)
    source = tmp_path / "sglang"
    source.mkdir()
    monkeypatch.setattr(launcher, "verify_source", lambda path: {"status": "verified"})
    args = launcher.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--training-manifest",
            str(manifest),
            "--sglang-source",
            str(source),
            "--python",
            sys.executable,
            "--device",
            "npu",
            "--cuda-execution",
            "torch-native-eager",
        ]
    )
    with pytest.raises(ValueError, match="only valid with --device cuda"):
        launcher.build_plan(args)


def test_explicit_context_length_override_is_recorded(tmp_path, monkeypatch):
    checkpoint, manifest = _fixture(tmp_path)
    source = tmp_path / "sglang"
    source.mkdir()
    monkeypatch.setattr(launcher, "verify_source", lambda path: {"status": "verified"})
    args = launcher.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--training-manifest",
            str(manifest),
            "--sglang-source",
            str(source),
            "--python",
            sys.executable,
            "--device",
            "cuda",
            "--context-length",
            "65536",
        ]
    )
    plan = launcher.build_plan(args)
    assert plan["server"]["context_length"] == 65536
    assert plan["server"]["context_length_source"] == "command_line"
    command = plan["command"]
    assert command[command.index("--context-length") + 1] == "65536"


def test_checkpoint_requires_original_manifest_and_accepts_single_weights(tmp_path):
    checkpoint, manifest = _fixture(tmp_path)
    (checkpoint / "model.safetensors.index.json").unlink()
    (checkpoint / "part-1.safetensors").unlink()
    (checkpoint / "part-2.safetensors").unlink()
    (checkpoint / "model.safetensors").write_bytes(b"single")
    profile = launcher.validate_checkpoint(checkpoint, manifest)
    assert profile["weight_layout"] == "single"
    manifest.write_text('{"variant":"H3","changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256"):
        launcher.validate_checkpoint(checkpoint, manifest)


def test_sharded_checkpoint_rejects_missing_shard(tmp_path):
    checkpoint, manifest = _fixture(tmp_path)
    (checkpoint / "part-2.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="shard is missing"):
        launcher.validate_checkpoint(checkpoint, manifest)
