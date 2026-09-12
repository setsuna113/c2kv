"""Lightweight gist export and pinned-base materialization contracts."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open
from safetensors.torch import load_file, save_file


AGENT_ROOT = Path(__file__).resolve().parents[2] / "agent"


def _load_entry(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, AGENT_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_entry("next_compression_train_for_export", "train_next_compression.py")
export_entry = _load_entry("next_compression_export", "export_next_checkpoint.py")


def _trained_checkpoint(path: Path) -> Path:
    train_entry.main(
        [
            "--cpu_smoke",
            "--variant",
            "H3",
            "--output_dir",
            str(path),
            "--max_steps",
            "1",
            "--save_steps",
            "1",
        ]
    )
    return path / "checkpoint-1"


def _original_base_from_checkpoint(checkpoint: Path, destination: Path) -> Path:
    from transformers import AutoTokenizer

    destination.mkdir()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    for key in tuple(config):
        if key.startswith("history_memory_") or key in {
            "gist_type",
            "gist_param",
            "gist_token_id",
            "gist_residual_type",
        }:
            config.pop(key)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.save_pretrained(destination)
    weights = load_file(checkpoint / "model.safetensors")
    save_file(
        {
            name: tensor
            for name, tensor in weights.items()
            if not export_entry._is_gist_parameter(name)
        },
        destination / "model.safetensors",
    )
    (destination / export_entry.BASE_RECEIPT_FILE).write_text(
        json.dumps(
            {
                "repository": export_entry.BASE_REPOSITORY,
                "revision": export_entry.BASE_REVISION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def test_export_materialize_exact_model_and_warm_start(tmp_path):
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    checkpoint = _trained_checkpoint(tmp_path / "trained")
    base = _original_base_from_checkpoint(checkpoint, tmp_path / "base")
    package = tmp_path / "gist-package"
    materialized = tmp_path / "materialized"

    manifest = export_entry.export_checkpoint(checkpoint, package)
    assert manifest["checkpoint"]["variant"] == "H3"
    assert manifest["gist"]["dtype"] == "float32"
    assert not (package / "optimizer.pt").exists()
    assert [path.name for path in package.glob("*.safetensors")] == [
        export_entry.GIST_FILE
    ]
    with safe_open(package / export_entry.GIST_FILE, framework="pt", device="cpu") as handle:
        assert all(str(handle.get_tensor(name).dtype) == "torch.float32" for name in handle.keys())

    receipt = export_entry.materialize_checkpoint(package, base, materialized)
    assert receipt["schema"] == export_entry.MATERIALIZATION_SCHEMA
    assert all(item["mode"] in {"hardlink", "copy"} for item in receipt["transfers"])
    assert (materialized / "model.safetensors.index.json").is_file()

    expected = Qwen3ForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    actual = Qwen3ForCausalLM.from_pretrained(materialized, local_files_only=True)
    expected_state, actual_state = expected.state_dict(), actual.state_dict()
    assert expected_state.keys() == actual_state.keys()
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)

    warm_args = train_entry.arguments(
        [
            "--device",
            "cpu",
            "--data_path",
            str(tmp_path / "unused-data"),
            "--variant",
            "H3",
            "--warm_start_checkpoint",
            str(materialized),
            "--output_dir",
            str(tmp_path / "warm-output"),
            "--no-gradient_checkpointing",
            "--wandb_mode",
            "disabled",
        ]
    )
    warmed, _ = train_entry.build_model(warm_args, torch.device("cpu"))
    exported_gist = load_file(package / export_entry.GIST_FILE)
    for name, tensor in warmed.state_dict().items():
        if export_entry._is_gist_parameter(name):
            assert tensor.dtype == torch.float32
            torch.testing.assert_close(tensor, exported_gist[name], rtol=0, atol=0)


def test_export_rejects_old_profile_and_materialize_rejects_wrong_base(tmp_path):
    checkpoint = _trained_checkpoint(tmp_path / "trained")
    old_profile = tmp_path / "old-profile"
    shutil.copytree(checkpoint, old_profile)
    config_path = old_profile / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["history_memory_training_profile"] = "history-event-base-query-v1"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not a next-compression"):
        export_entry.export_checkpoint(old_profile, tmp_path / "bad-export")

    package = tmp_path / "gist-package"
    export_entry.export_checkpoint(checkpoint, package)
    base = _original_base_from_checkpoint(checkpoint, tmp_path / "base")
    receipt_path = base / export_entry.BASE_RECEIPT_FILE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["revision"] = "wrong-revision"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Base source receipt differs"):
        export_entry.materialize_checkpoint(package, base, tmp_path / "rejected")

    receipt["revision"] = export_entry.BASE_REVISION
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    base_config_path = base / "config.json"
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    base_config["hidden_size"] += 1
    base_config_path.write_text(json.dumps(base_config), encoding="utf-8")
    with pytest.raises(ValueError, match="Base architecture differs"):
        export_entry.materialize_checkpoint(
            package, base, tmp_path / "rejected-architecture"
        )
