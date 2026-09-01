"""SGLang c2kv fork backend (kvoffload-sglang-c2kv, branch c2kv-sglang-bfcl).

Wire protocol (verified live against 22fbf3146 on NPU, docs/sglang_migration.md):
* ``POST /v1/c2kv/extract`` — same request/response shape as hf_server
  (``text``/``compression_ratio``/``role`` → ``key_hash``/``gist_len``/
  ``original_seq_len``); failures are HTTP 200 with ``success=false``.
* ``POST /v1/c2kv/repair_extract`` — full-context form (server branch
  task/c2kv-serve-align, c2kv/c2kv_serving_semantics.md): ``messages`` +
  ``target_index`` + ``tools``; the server renders the prefix like a chat
  request and stores the target message's raw K/V (pre-RoPE) with its
  absolute positions in that rendering.  The legacy ``text``+``role`` form
  (standalone encoding, caller-supplied ``position_offset``) is kept only
  for A/B.
* message-level ``c2kv_key_hash`` (gist), ``c2kv_repair_key_hashes`` /
  ``c2kv_repair_only_key_hashes`` (raw KV) and ``c2kv_repair_placement``
  (in_place / append_keep_ledger / append_tail) on chat messages.
* every response carries ``metadata.sglang_runtime.c2kv_query_proj`` (which
  projection the server used for post-gist tokens) and ``c2kv_layout`` (the
  injections with their RoPE positions) for provenance / frame checks.
* constrained decoding via ``response_format={"type": "structural_tag"}``;
  tool schemas need xgrammar-safe repair (``_normalize_tool_schema``,
  moved verbatim from hf_server.py).
* per-request KV accounting in ``metadata.sglang_runtime``
  (kv_resident_tokens / kv_peak_resident_tokens / kv_pool_size).

Regime notes (superseding the 2026-08-30 decisions in docs/sglang_migration.md):
the server's ``--c2kv-query-proj gist`` restores the training use_gist rule
for post-gist tokens; docs are packed by the proxy in the TRAINING turn
format (``--doc-packing turn``); repair placement is explicit.
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

    def repair_extract_messages(self, messages: List[Dict[str, Any]],
                                target_index: int,
                                tools: Optional[List[Dict[str, Any]]],
                                source_doc_index: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [
                {"role": m.get("role") or "user", "content": m.get("content") or ""}
                for m in messages
            ],
            "target_index": int(target_index),
            "chat_template_kwargs": {"enable_thinking": False},
            "repair_mode": "d_corr",
            "source_doc_index": source_doc_index,
        }
        if tools:
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "repair_failed",
                f"c2kv repair_extract(messages) failed: "
                f"{result.get('error') or json.dumps(result)[:500]}")
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
            placement = str(repair_plan.get("placement") or "append_keep_ledger")
            key_hash = repair_plan["repair_key_hash"]
            index = repair_plan.get("target_out_index", repair_plan.get("message_index"))
            if index is None or not 0 <= index < len(messages):
                raise BackendError(
                    "repair_failed",
                    f"repair plan target index {index!r} out of range")
            if placement == "in_place":
                # the raw span REPLACES the target doc's gist: the message
                # becomes repair-only (no gist), the server re-anchors the
                # query to the span's absolute end
                message = dict(messages[index])
                if not message.get("c2kv_key_hash"):
                    raise BackendError(
                        "repair_failed",
                        f"in_place repair target {index} has no c2kv_key_hash")
                message.pop("c2kv_key_hash", None)
                message["c2kv_repair_only_key_hashes"] = [key_hash]
                message["c2kv_repair_placement"] = placement
                messages[index] = message
            elif placement in ("append_keep_ledger", "append_tail"):
                # the raw span is appended to the end of history (after all
                # gists and the raw hybrid tail, before the current turn) as a
                # repair-only message; the gist of the target doc stays
                insert_at = repair_plan.get("current_start_out_index")
                if insert_at is None or not 0 <= insert_at <= len(messages):
                    insert_at = len(messages)
                messages.insert(int(insert_at), {
                    "role": "user", "content": "",
                    "c2kv_repair_only_key_hashes": [key_hash],
                    "c2kv_repair_placement": placement,
                })
            else:
                raise BackendError("repair_failed", f"unknown placement {placement!r}")
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
        error_text = str(data.get("error") or "")
        if "C2KV_CACHE_MISS" in error_text or "C2KV cache miss" in error_text:
            raise BackendError("cache_miss", error_text)
        runtime = ((data.get("metadata") or {}).get("sglang_runtime")) or {}
        cost = {k: runtime[k] for k in (
            "kv_resident_tokens", "kv_peak_resident_tokens", "kv_pool_size",
            "c2kv_query_proj", "c2kv_gist_seen", "c2kv_position_correction",
            "c2kv_layout")
            if k in runtime}
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": finish,
            "usage": data.get("usage"),
            "cost": cost,
        }
