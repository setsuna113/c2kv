from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.checkpoint_profile import (
    PROFILE_FILENAME,
    ProfileError,
    build_training_profile,
    profile_run_args,
    resolve_checkpoint_profile,
    write_training_profile,
)
from benchmarks import sglang_smoke


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _checkpoint(tmp_path: Path, name: str = "checkpoint-588") -> Path:
    checkpoint = tmp_path / "run" / name
    _write_json(
        checkpoint / "config.json",
        {
            "model_type": "qwen3",
            "gist_param": "qkv",
            "gist_type": "dynamic-interleave",
            "gist_overlap": 64,
            "gist_residual_type": "embed-mean",
        },
    )
    return checkpoint


def test_legacy_parent_artifacts_resolve_g_history_contract(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {
            "doc_mode": "history_only",
            "tools_in_system": "True",
            "max_doc_length": "768",
            "max_doc_num": "16",
            "ratios": "8,8,4,16",
        },
    )
    _write_json(
        checkpoint.parent / "train_manifest_used.json",
        {"doc_mode": "history_only", "tools_in_system": True},
    )

    profile = resolve_checkpoint_profile(
        checkpoint, query_projection="gist", require_serving_e2e=True
    )

    assert profile["profile_kind"] == "legacy_artifacts"
    assert profile["training"] == {
        "doc_mode": "history_only",
        "tools_in_system": True,
        "doc_packing": "turn",
        "max_doc_length": 768,
        "max_doc_num": 16,
        "compression_ratios": [8, 4, 16],
        "history_selection": None,
        "hybrid_tail_choices": None,
    }
    assert profile["serving"]["query_projection"] == "gist"
    assert profile_run_args(profile) == [
        "--doc-packing", "turn",
        "--max-doc-length", "768",
        "--max-doc-num", "16",
    ]
    artifacts = profile["provenance"]["artifacts"]
    assert {item["kind"] for item in artifacts} == {"run_config", "train_manifest"}
    assert all(len(item["sha256"]) == 64 for item in artifacts)
    assert len(profile["profile_fingerprint"]) == 64


def test_legacy_query_projection_must_be_explicit(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "max_doc_length": 768,
            "max_doc_num": 16,
            "ratios": "8",
        },
    )
    with pytest.raises(ProfileError, match="query projection is absent"):
        resolve_checkpoint_profile(checkpoint)


def test_legacy_artifact_conflict_is_not_silently_preferred(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {"doc_mode": "history_only", "tools_in_system": True},
    )
    _write_json(
        checkpoint.parent / "train_manifest_used.json",
        {"doc_mode": "joint", "tools_in_system": True},
    )
    with pytest.raises(ProfileError, match="conflicting doc_mode"):
        resolve_checkpoint_profile(checkpoint, query_projection="gist")


def test_joint_checkpoint_cannot_enter_history_serving(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {
            "doc_mode": "joint",
            "tools_in_system": False,
            "max_doc_length": 1024,
            "max_doc_num": 24,
            "ratios": "8",
        },
    )
    with pytest.raises(ProfileError, match="history_only checkpoint"):
        resolve_checkpoint_profile(
            checkpoint, query_projection="gist", require_serving_e2e=True
        )


def test_checkpoint_1088_is_explicitly_reference_not_as_trained(tmp_path):
    checkpoint = _checkpoint(tmp_path, "checkpoint-1088")
    with pytest.raises(ProfileError, match="no c2kv_checkpoint_profile"):
        resolve_checkpoint_profile(checkpoint)

    profile = resolve_checkpoint_profile(
        checkpoint,
        reference_profile="checkpoint-1088",
        require_serving_e2e=True,
    )
    assert profile["profile_kind"] == "reference"
    assert profile["serving"] == {
        "compatible": True,
        "compatibility_reason": "explicit checkpoint-1088 reference serving contract",
        "query_projection": "base",
        "doc_packing": "turn",
        "max_doc_length": 512,
        "max_doc_num": 12,
        "compression_ratios": [4, 8, 16],
    }
    assert "not an as-trained record" in profile["provenance"]["claim"]
    assert "resolved_training_doc_geometry" in profile["missing"]


def test_explicit_reference_is_not_shadowed_by_a_parent_profile(tmp_path):
    checkpoint = _checkpoint(tmp_path, "checkpoint-1088")
    _write_json(checkpoint.parent / PROFILE_FILENAME, {"schema_version": 999})
    profile = resolve_checkpoint_profile(
        checkpoint, reference_profile="checkpoint-1088", require_serving_e2e=True
    )
    assert profile["profile_kind"] == "reference"


def test_reference_profile_rejects_a_different_checkpoint_name(tmp_path):
    checkpoint = _checkpoint(tmp_path, "checkpoint-500")
    with pytest.raises(ProfileError, match="path named checkpoint-1088"):
        resolve_checkpoint_profile(checkpoint, reference_profile="checkpoint-1088")


@dataclass
class _ModelArgs:
    model_name_or_path: str = "/models/base"
    gist_param: str = "qkv"


@dataclass
class _DataArgs:
    doc_mode: str = "history_only"
    tools_in_system: bool = True
    max_doc_length: int = 768
    max_doc_num: int = 16
    history_selection: str = "tail"
    hybrid_tail_choices: str | None = None
    max_source_tokens: int = 1234


class _TrainingArgs:
    def to_dict(self):
        return {
            "output_dir": "/tmp/out",
            "learning_rate": 5e-5,
            "max_new_tokens": 128,
            "push_to_hub_token": "do-not-write-me",
        }


def test_new_training_profile_is_self_describing_and_secret_safe(tmp_path):
    profile = build_training_profile(
        output_dir=tmp_path / "run",
        repo_root=tmp_path,
        model_args=_ModelArgs(),
        training_args=_TrainingArgs(),
        data_args=_DataArgs(),
        model_config={
            "gist_param": "qkv",
            "gist_type": "dynamic-interleave",
            "gist_overlap": 64,
            "gist_residual_type": "embed-mean",
        },
        argv=["train.py", "--do_train", "True", "--hub_token", "private-value", "--api-key=another-private-value"],
        environ={"C2KV_GIST_TRAIN_RATIOS": "8,8,4,16"},
    )
    assert profile["profile_kind"] == "as_trained"
    assert profile["serving"]["query_projection"] == "gist"
    assert profile["serving"]["max_doc_length"] == 768
    assert profile["training"]["compression_ratios"] == [8, 4, 16]
    assert profile["training"]["compression_ratio_sampling"] == [8, 8, 4, 16]
    assert "private-value" not in json.dumps(profile)
    resolved_training = profile["training"]["resolved_args"]["training"]
    assert resolved_training["max_new_tokens"] == 128
    assert resolved_training["push_to_hub_token"] == "<redacted>"
    assert profile["provenance"]["source"]["missing"] == ["git_commit", "git_dirty"]


def test_written_training_profile_is_discovered_from_checkpoint_child(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setenv("C2KV_GIST_TRAIN_RATIOS", "8")
    path = write_training_profile(
        output_dir=run,
        repo_root=tmp_path,
        model_args=_ModelArgs(),
        training_args=_TrainingArgs(),
        data_args=_DataArgs(),
        model_config={"gist_param": "qkv", "gist_type": "dynamic-interleave"},
        argv=["train.py"],
    )
    assert path == run / PROFILE_FILENAME
    checkpoint = _checkpoint(tmp_path)
    profile = resolve_checkpoint_profile(checkpoint, require_serving_e2e=True)
    assert profile["profile_kind"] == "as_trained"
    assert profile["serving"]["max_doc_num"] == 16
    assert any(
        item["kind"] == "checkpoint_profile"
        for item in profile["provenance"]["artifacts"]
    )


def test_resolved_profile_is_stable_and_bound_to_checkpoint(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "max_doc_length": 768,
            "max_doc_num": 16,
            "ratios": "8",
        },
    )
    profile = resolve_checkpoint_profile(checkpoint, query_projection="gist")
    resolved_path = _write_json(tmp_path / "resolved.json", profile)
    again = resolve_checkpoint_profile(checkpoint, profile_path=resolved_path)
    assert again["profile_fingerprint"] == profile["profile_fingerprint"]

    other = _checkpoint(tmp_path / "other")
    with pytest.raises(ProfileError, match="does not match"):
        resolve_checkpoint_profile(other, profile_path=resolved_path)
    config = json.loads((checkpoint / "config.json").read_text())
    config["gist_param"] = "QkV"
    _write_json(checkpoint / "config.json", config)
    with pytest.raises(ProfileError, match="config.json changed"):
        resolve_checkpoint_profile(checkpoint, profile_path=resolved_path)


def test_profile_preserves_supported_default_training_ratios(tmp_path):
    profile = build_training_profile(output_dir=tmp_path, repo_root=tmp_path,
        model_args=_ModelArgs(), training_args=_TrainingArgs(), data_args=_DataArgs(),
        model_config={"gist_param": "qkv", "gist_type": "dynamic-interleave"}, environ={})
    assert profile["training"]["compression_ratio_sampling"] == [2, 4, 8]
    assert "default" in profile["provenance"]["field_sources"]["training.compression_ratio_sampling"]


def test_fixed_ratio_profile_ignores_dynamic_environment(tmp_path):
    profile = build_training_profile(output_dir=tmp_path, repo_root=tmp_path,
        model_args=_ModelArgs(), training_args=_TrainingArgs(), data_args=_DataArgs(),
        model_config={"gist_param": "qkv", "gist_type": "interleave-4"},
        environ={"C2KV_GIST_TRAIN_RATIOS": "8,16"})
    assert profile["training"]["compression_ratio_sampling"] == [4]


def test_smoke_gate_uses_resolved_profile_instead_of_proxy_defaults(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _write_json(
        checkpoint.parent / "run_config.json",
        {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "max_doc_length": 768,
            "max_doc_num": 16,
            "ratios": "8",
        },
    )
    resolved = resolve_checkpoint_profile(checkpoint, query_projection="gist")
    profile_path = _write_json(tmp_path / "resolved.json", resolved)
    args = argparse.Namespace(
        checkpoint=checkpoint,
        checkpoint_profile=profile_path,
        reference_profile=None,
        query_projection=None,
    )
    regime = sglang_smoke._proxy_regime(args)
    assert regime == {
        "doc_packing": "turn",
        "max_doc_num": 16,
        "max_doc_length": 768,
        "query_projection": "gist",
        "profile_fingerprint": resolved["profile_fingerprint"],
    }
    assert sglang_smoke._proxy_regime_flags(regime) == [
        "--doc-packing", "turn",
        "--max-doc-num", "16",
        "--max-doc-length", "768",
        "--query-projection", "gist",
    ]


def test_prologue_token_count_handles_transformers5_batch_encoding(tmp_path, monkeypatch):
    class _Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            tools = kwargs["tools"]
            assert all(tool["function"]["strict"] is False for tool in tools)
            assert all("response" not in tool["function"] for tool in tools)
            return {"input_ids": [list(range(318))], "attention_mask": [[1] * 318]}

    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _Tokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert sglang_smoke._prologue_token_len(tmp_path) == 318


def test_h200_wrapper_delegates_cells_to_generic_matrix():
    text = (Path(__file__).with_name("run_matrix_h200.sh")).read_text(encoding="utf-8")
    assert '"$REPO_ROOT/benchmarks/matrix.py"' in text
    assert "--execute" in text
    assert '"$REPO_ROOT/benchmarks/run.py"' not in text
    assert '"profile": {**profile, "path": profile_path}' in text
