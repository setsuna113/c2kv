"""Pure-part tests for agent/t34_model_loader.py (the torch load is server-only)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t34_model_loader as L  # noqa: E402


def test_checkpoint_mode_detects_gist_config(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    gist = tmp_path / "gist"
    gist.mkdir()
    (gist / "config.json").write_text(json.dumps({"model_type": "qwen3", "gist_param": "qkv",
                                                  "num_gist_tokens": 8}), encoding="utf-8")
    assert L.checkpoint_mode(str(plain)) == "full"
    assert L.checkpoint_mode(str(gist)) == "c2kv"
    assert L.checkpoint_mode(str(tmp_path / "missing")) == "full"      # hub id / no config
    assert L.checkpoint_mode(str(gist), "full") == "full"               # explicit wins
    assert L.checkpoint_mode(str(plain), "c2kv") == "c2kv"
    with pytest.raises(ValueError):
        L.checkpoint_mode(str(plain), "weird")


def test_harness_namespace_carries_the_frozen_recipe_and_overrides():
    ns = L.harness_namespace("/ckpt/fixed_joint", mode="c2kv", tokenizer_path="/tok")
    assert ns.mode == "c2kv" and ns.model == "/ckpt/fixed_joint" and ns.tokenizer == "/tok"
    assert ns.base_model is None
    assert (ns.max_doc_length, ns.max_doc_num, ns.override_ratio, ns.dtype) == (768, 16, 8, "bf16")
    assert ns.generate_attn_impl == "eager" == ns.gist_attn_impl == ns.system_attn_impl
    ns2 = L.harness_namespace("/models/base", mode="full", base_model="/models/base", attn_impl="eager")
    assert ns2.mode == "full" and ns2.base_model == "/models/base"
    assert ns2.tokenizer == "/models/base"                              # defaults to the model path
