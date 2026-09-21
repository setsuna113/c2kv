"""Materialize an explicit, checkpoint-bound native history budget."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


POLICY_FILENAME = "history_budget_eval_policy.json"
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8"))


def _policy_bytes(policy: Mapping[str, Any]) -> bytes:
    return (json.dumps(policy, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _kv_bytes_per_token(config: Mapping[str, Any], dtype: str) -> int:
    element_bytes = DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise ValueError(f"unsupported native history budget dtype: {dtype!r}")
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden, heads = config.get("hidden_size"), config.get("num_attention_heads")
        if type(hidden) is not int or type(heads) is not int or heads <= 0 or hidden % heads:
            raise ValueError("checkpoint lacks a valid KV head dimension")
        head_dim = hidden // heads
    dimensions = (config.get("num_hidden_layers"), config.get("num_key_value_heads"), head_dim)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("checkpoint lacks positive KV geometry")
    required = 2 * dimensions[0] * dimensions[1] * dimensions[2] * element_bytes
    policy = config.get("history_memory_policy")
    if not isinstance(policy, Mapping) or policy.get("kv_bytes_per_token") != required:
        raise ValueError("checkpoint policy disagrees with native KV byte geometry")
    return required


def resolve_override(tokens: int, checkpoint: Path, design: Mapping[str, Any],
                     runtime_root: Path, output: Path) -> dict[str, Any]:
    """Resolve B tokens to the two policy byte caps without loading model weights."""
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("--history-budget-tokens must be a positive integer")
    config_path = checkpoint / "config.json"
    config_bytes = config_path.read_bytes()
    config_sha256 = _sha256(config_bytes)
    expected = design["checkpoint_selection"]["config_sha256"]
    if config_sha256 != expected:
        raise ValueError("native history budget requires the selected C1000 checkpoint config")
    config = json.loads(config_bytes)
    dtype = design["runtime"]["dtype"]
    kv_bytes = _kv_bytes_per_token(config, dtype)
    base_path = (runtime_root / design["runtime"]["eval_policy"]).resolve()
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    if (base.get("schema") != "a-event-native-eval-policy-v1"
            or not isinstance(base.get("policy"), dict)
            or type(base["policy"].get("history_budget_bytes")) is not int
            or type(base["policy"].get("workspace_budget_bytes")) is not int):
        raise ValueError("frozen native evaluation policy has an unexpected contract")
    budget_bytes = tokens * kv_bytes
    override = copy.deepcopy(base)
    override["policy_id"] = f"{base['policy_id']}-native-history-b{tokens}-v1"
    override["policy"]["history_budget_bytes"] = budget_bytes
    override["policy"]["workspace_budget_bytes"] = budget_bytes
    target = (output / POLICY_FILENAME).resolve()
    return {
        "schema": "c2kv-native-history-budget-override-v1",
        "requested_tokens": tokens,
        "kv_bytes_per_token": kv_bytes,
        "inference_dtype": dtype,
        "history_budget_bytes": budget_bytes,
        "workspace_budget_bytes": budget_bytes,
        "checkpoint_config_sha256": config_sha256,
        "base_eval_policy_path": str(base_path),
        "base_eval_policy_sha256": _sha256(base_bytes),
        "base_policy_id": base["policy_id"],
        "base_history_budget_bytes": base["policy"]["history_budget_bytes"],
        "base_workspace_budget_bytes": base["policy"]["workspace_budget_bytes"],
        "override_eval_policy_path": str(target),
        "override_eval_policy_sha256": _sha256(_policy_bytes(override)),
        "override_policy_sha256": _canonical_sha256(override),
        "eval_policy": override,
    }


def materialize(receipt: Mapping[str, Any]) -> Path:
    """Write once, refusing to replace a different policy in an existing run."""
    target = Path(receipt["override_eval_policy_path"])
    contents = _policy_bytes(receipt["eval_policy"])
    if _sha256(contents) != receipt["override_eval_policy_sha256"]:
        raise ValueError("native history budget policy receipt differs from its bytes")
    if target.exists():
        if target.read_bytes() != contents:
            raise FileExistsError(f"Different native history budget policy already exists: {target}")
    else:
        target.write_bytes(contents)
    return target
