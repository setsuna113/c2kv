"""SGLang c2kv fork backend (kvoffload-sglang-c2kv, branch c2kv-sglang-bfcl).

Wire protocol (verified live against 22fbf3146 on NPU, docs/sglang_migration.md):
* ``POST /v1/c2kv/extract`` — same request/response shape as hf_server
  (``text``/``compression_ratio``/``role`` → ``key_hash``/``gist_len``/
  ``original_seq_len``); failures are HTTP 200 with ``success=false``.
* ``POST /v1/c2kv/repair_extract`` — stores the raw KV of a span; the
  d_corr entry bakes the CALLER-SUPPLIED ``position_offset`` into the
  absolute RoPE phase at capture time and is copied verbatim at injection
  (``inject_c2kv_stored_kv``: already_rotated entries are not re-rotated).
  The offset must therefore be the span's true logical position in the
  ORIGINAL uncompressed conversation: system tokens + Σ original_seq_len
  of every message before it.  The proxy ledger supplies it.
* message-level ``c2kv_key_hash`` + ``c2kv_repair_key_hashes`` on chat
  messages; injection follows the message's own gist block.
* constrained decoding via ``response_format={"type": "structural_tag"}``;
  tool schemas need xgrammar-safe repair (``_normalize_tool_schema``,
  moved verbatim from hf_server.py).
* per-request KV accounting in ``metadata.sglang_runtime``
  (kv_resident_tokens / kv_peak_resident_tokens / kv_pool_size).

Accepted regime differences (decided 2026-08-30, docs/sglang_migration.md):
no use_gist global rule (bench face is its own regime, all numbers
rebaselined); whole-message chunking (no <=768 split simulation); repair
injects after the target's own gist block, not at the prefix end.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import Backend, BackendError


def _normalize_tool_schema(value: Any, defs: Dict[str, Any] = None,
                           seen: frozenset = frozenset()) -> Any:
    """Schema repair before grammar compilation (moved verbatim from
    benchmarks/hf_server.py — the SGLang/xgrammar path needs it just as
    much as the old server did).

    Benchmark tool schemas use loose types ("type": "dict"/"any"/...,
    {"type": "list"} without items) and $ref/$defs indirection, neither
    of which xgrammar's JSON-schema converter accepts.  Repair applies to
    the grammar input only, never to the tool definition the model sees:
    $refs are inlined (cycle-guarded), then types are mapped and
    unsupported keywords stripped.
    """
    if isinstance(value, dict):
        if "$ref" in value and defs is not None:
            name = str(value["$ref"]).split("/")[-1]
            if name in defs and name not in seen:
                return _normalize_tool_schema(
                    defs[name], defs, seen | {name}
                )
            return {"type": "object"}
        out = {}
        for key, item in value.items():
            if key in ("$schema", "$id", "$defs", "$ref"):
                continue
            if key == "type":
                if isinstance(item, str):
                    item = {
                        "dict": "object",
                        "any": "object",
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "tuple": "array",
                        "None": "null",
                    }.get(item, item)
                elif isinstance(item, list):
                    mapped = []
                    for t in item:
                        t = {
                            "dict": "object", "any": "object", "int": "integer",
                            "float": "number", "str": "string", "bool": "boolean",
                            "list": "array", "tuple": "array", "None": "null",
                        }.get(t, t)
                        if t not in mapped:
                            mapped.append(t)
                    if not mapped:
                        continue
                    item = mapped
            out[key] = _normalize_tool_schema(item, defs, seen)
        return out
    if isinstance(value, list):
        return [_normalize_tool_schema(v, defs, seen) for v in value]
    return value


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return _normalize_tool_schema(schema, defs)


class SglangBackend(Backend):
    name = "sglang"
    needs_repair_plan = True

    def __init__(self, post_json):
        self._post_json = post_json  # (path, payload, timeout) -> dict

    # ---- primitives ----
    def extract(self, text: str, role: str, ratio: int,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload = {
            "text": text,
            "compression_ratio": ratio,
            "role": role,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            # the server renders tools into the chat template exactly as
            # the serving path does, so original_seq_len measures the TRUE
            # system block (fixes the position_offset short-by-tools bug)
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "extract_failed",
                f"c2kv extract failed: {result.get('error') or json.dumps(result)[:500]}")
        return result

    def repair_extract(self, text: str, role: str, span_start: int,
                       span_end: Optional[int], position_offset: int,
                       source_doc_index: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text": text,
            "role": role,
            "chat_template_kwargs": {"enable_thinking": False},
            "span_start": span_start,
            "position_offset": position_offset,
            "repair_mode": "d_corr",
            "source_doc_index": source_doc_index,
        }
        if span_end is not None:
            payload["span_end"] = span_end
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "repair_failed",
                f"c2kv repair_extract failed: {result.get('error') or json.dumps(result)[:500]}")
        return result

    # ---- chat shaping ----
    def prepare_chat(self, payload: Dict[str, Any], arm,
                     repair_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(payload)
        out.pop("c2kv_repair", None)  # request-level repair is hf_server-only
        messages = list(out.get("messages") or [])
        if arm.constrain_tools:
            # structural_tag constrained decoding; the grammar input needs
            # xgrammar-safe schemas (repair only the copy the server compiles)
            out["response_format"] = {"type": "structural_tag"}
            tools = out.get("tools")
            if tools:
                out["tools"] = [
                    {**tool, "function": {
                        **tool.get("function", {}),
                        "parameters": _inline_refs(
                            tool.get("function", {}).get("parameters")
                            or {"type": "object"}),
                    }} if isinstance(tool, dict) else tool
                    for tool in tools
                ]
        if repair_plan and arm.repair:
            index = repair_plan.get("message_index")
            if index is None or not 0 <= index < len(messages):
                raise BackendError(
                    "repair_failed",
                    f"repair plan message_index {index!r} out of range")
            message = dict(messages[index])
            # the repair hashes are read only on messages that also carry
            # their own gist reference (scheduler walks c2kv segments)
            if not message.get("c2kv_key_hash"):
                raise BackendError(
                    "repair_failed",
                    f"repair target message {index} has no c2kv_key_hash")
            hashes = list(message.get("c2kv_repair_key_hashes") or [])
            hashes.append(repair_plan["repair_key_hash"])
            message["c2kv_repair_key_hashes"] = hashes
            messages[index] = message
        out["messages"] = messages
        return out

    # ---- response normalization ----
    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("object") == "error" or (data.get("error") and not data.get("choices")):
            raise BackendError("upstream", f"sglang error body: {data.get('error')}")
        choice = (data.get("choices") or [{}])[0] or {}
        finish = choice.get("finish_reason")
        if finish == "abort":
            raise BackendError("finish_abort", json.dumps(data)[:500])
        message = choice.get("message") or {}
        runtime = ((data.get("metadata") or {}).get("sglang_runtime")) or {}
        cost = {k: runtime[k] for k in (
            "kv_resident_tokens", "kv_peak_resident_tokens", "kv_pool_size")
            if k in runtime}
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": finish,
            "usage": data.get("usage"),
            "cost": cost,
        }
