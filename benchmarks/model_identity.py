"""Verify the loaded architecture, not an OpenAI served-name alias.

Dimensions are from Qwen/Qwen3-4B's published config.json. This verifies
the architecture/size; checkpoint provenance remains the deployment profile.
"""
QWEN3_4B = {"hidden_size": 2560, "intermediate_size": 9728,
            "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936}


def require_qwen3_4b(info):
    if info.get("model_type") != "qwen3":
        raise ValueError("upstream must load a Qwen3 dense model")
    actual = info.get("model_dimensions") or {}
    mismatches = {k: actual.get(k) for k, v in QWEN3_4B.items() if actual.get(k) != v}
    if mismatches:
        raise ValueError(f"upstream does not verify as Qwen3-4B: {mismatches}")
    if not info.get("model_path"):
        raise ValueError("upstream is missing its loaded checkpoint path")
    return {"model_path": info["model_path"], "tokenizer_path": info.get("tokenizer_path"),
            "model_dimensions": dict(actual), "weight_version": info.get("weight_version")}
