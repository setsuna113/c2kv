"""hf_server backend (in-repo Flask stack, the pre-migration serving path).

Kept as the reference/contrast implementation after the SGLang migration:
repair is request-level (``c2kv_repair`` resolved server-side over its own
doc/chunk ledger), the cost block rides on the response ``c2kv`` field.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Backend


class HfServerBackend(Backend):
    name = "hfserver"

    def __init__(self, post_json):
        # post_json(base_url, path, payload, timeout) -> dict (retries incl.)
        self._post_json = post_json

    def extract(self, text: str, role: str, ratio: int) -> Dict[str, Any]:
        result = self._post_json(
            "/v1/c2kv/extract",
            {
                "text": text,
                "compression_ratio": ratio,
                "role": role,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            600,
        )
        if not result.get("success", True) or not result.get("key_hash"):
            raise RuntimeError(
                f"c2kv extract failed: {result.get('error') or result}")
        return result

    def repair_extract(self, text: str, role: str, span_start: int,
                       span_end: Optional[int], position_offset: int,
                       source_doc_index: int) -> Dict[str, Any]:
        # no message-level primitive: repair rides on the chat request
        return {}

    def prepare_chat(self, payload: Dict[str, Any], arm,
                     repair_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(payload)
        if arm.constrain_tools:
            out["constrain_tools"] = True
        if arm.repair:
            # docs/hybrid_spec.md "Repair interaction": the server resolves
            # the policy over the compressed docs it assembles
            out["c2kv_repair"] = dict(arm.repair)
        else:
            out.pop("c2kv_repair", None)
        return out

    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("object") == "error" or "error" in data and data.get("error"):
            raise RuntimeError(f"hf_server error body: {data.get('error')}")
        choice = (data.get("choices") or [{}])[0]
        c2kv = data.get("c2kv") or {}
        return {
            "content": (choice.get("message") or {}).get("content"),
            "tool_calls": (choice.get("message") or {}).get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "cost": {k: c2kv[k] for k in (
                "cache_tokens", "logical_tokens", "prompt_tokens",
                "system_len", "repair_policy", "repair_block_tokens",
                "repair_doc_index", "repair_prefill_sec") if k in c2kv},
        }
