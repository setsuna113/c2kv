"""Compose schema-protected raw tool KV with an independent history policy.

The chat server rejects two eviction hints because both would mutate the same
request KV state.  Here the tool protocol is selected first into a repair-only
KV entry; the history hint remains in charge of the rest of the request.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence


class JointToolMemoryError(ValueError):
    """An exact-frame or separate-budget requirement could not be met."""


def _render(messages: Sequence[Mapping[str, Any]], tokenizer: Any,
            request: Mapping[str, Any]) -> tuple[str, tuple[int, ...], list[tuple[int, int]], list[int]]:
    native = tokenizer._load() if hasattr(tokenizer, "_load") else tokenizer
    visible = [copy.deepcopy(dict(message)) for message in messages if not (
        message.get("c2kv_key_hash") or message.get("c2kv_repair_key_hashes")
        or message.get("c2kv_repair_only_key_hashes"))]
    for message in visible:
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message.get("content"), str):
            raise JointToolMemoryError("TOOL_KV_JOINT_TEXT_MESSAGES_REQUIRED")
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
    kwargs = dict(request.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", False)
    if request.get("reasoning_effort") is not None:
        kwargs.setdefault("reasoning_effort", request["reasoning_effort"])
    rendered = native.apply_chat_template(
        visible, tools=None, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = native(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    native_ids = native.apply_chat_template(
        visible, tools=None, tokenize=True, add_generation_prompt=True, **kwargs)
    if hasattr(native_ids, "input_ids"):
        native_ids = native_ids.input_ids
    if tuple(ids) != tuple(native_ids):
        raise JointToolMemoryError("TOOL_KV_JOINT_RENDERED_TOKEN_MISMATCH")
    starts = []
    cursor = 0
    for message in visible:
        content = message.get("content")
        start = rendered.find(content, cursor) if content else cursor
        if start < 0:
            raise JointToolMemoryError("TOOL_KV_JOINT_MESSAGE_CONTENT_NOT_RENDERED")
        starts.append(start)
        cursor = start + len(content)
    return rendered, tuple(int(item) for item in ids), [tuple(pair) for pair in offsets], starts


def _indices(span: Mapping[str, Any], messages: Sequence[Mapping[str, Any]],
             offsets: Sequence[tuple[int, int]], starts: Sequence[int], *,
             overlap: bool) -> list[int]:
    index = int(span["message_index"])
    if not 0 <= index < len(messages):
        raise JointToolMemoryError("TOOL_KV_JOINT_SCHEMA_MESSAGE_INDEX_INVALID")
    content = messages[index].get("content")
    left, right = int(span["start"]), int(span["end"])
    if (not isinstance(content, str) or not 0 <= left < right <= len(content)
            or content[left:right] != span.get("text")):
        raise JointToolMemoryError("TOOL_KV_JOINT_SCHEMA_TEXT_MISMATCH")
    left += starts[index]
    right += starts[index]
    result = [i for i, (a, b) in enumerate(offsets) if a < b and (
        (a < right and b > left) if overlap else (a >= left and b <= right))]
    if not result:
        raise JointToolMemoryError("TOOL_KV_JOINT_SCHEMA_HAS_NO_TOKENS")
    return result


def _retained_source_tokens(messages: Sequence[Mapping[str, Any]],
                            source_spans: Sequence[Mapping[str, Any]],
                            tokenizer: Any, request: Mapping[str, Any]) -> tuple[int, set[int]]:
    if not source_spans:
        return 0, set()
    _, _, offsets, starts = _render(messages, tokenizer, request)
    visible = [m for m in messages if not (m.get("c2kv_key_hash")
               or m.get("c2kv_repair_key_hashes") or m.get("c2kv_repair_only_key_hashes"))]
    covered: set[int] = set()
    found: set[int] = set()
    for span in source_spans:
        text = str(span["text"])
        for index, message in enumerate(visible):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            cursor = 0
            while (position := content.find(text, cursor)) >= 0:
                # The protected protocol contains interface copies.  They are
                # counted in the protocol, never again as source documents.
                protocol = str(request.get("_joint_protocol") or "")
                protocol_start = content.find(protocol) if index == 0 and protocol else -1
                inside_protocol = (protocol_start >= 0 and position < protocol_start + len(protocol)
                                   and position + len(text) > protocol_start)
                if not inside_protocol:
                    left, right = starts[index] + position, starts[index] + position + len(text)
                    covered.update(i for i, (a, b) in enumerate(offsets)
                                   if a < b and a < right and b > left)
                    found.add(int(span["schema_index"]))
                cursor = position + len(text)
    return len(covered), found


def prepare_joint_raw_tool_history(
    staged: Mapping[str, Any], prepared: Mapping[str, Any], plan: Any,
    tokenizer: Any, backend: Any, *,
    source_messages: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Pre-extract one exact raw tool protocol and keep history's own hint.

    ``staged`` is the assembled request before backend history shaping.
    ``source_messages`` provides original full history when ``staged`` already
    contains C2KV gist carriers. The extractor observes that query-bearing
    source; the inserted carrier and history algorithm operate separately.
    """
    staged_hint = (staged.get("c2kv_kv_memory_hint") or {}).get("tool_kv_eviction")
    staged_hint = staged_hint if isinstance(staged_hint, dict) else {}
    if not (staged_hint.get("joint_history_assembly")
            or plan.info.get("joint_history_assembly")
            or source_messages is not None):
        return dict(prepared)
    if getattr(plan.spec, "interface_policy", None) != "schema":
        raise JointToolMemoryError("TOOL_KV_JOINT_REQUIRES_SCHEMA_POLICY")
    if prepared.get("c2kv_tools_in_prompt") is not False:
        raise JointToolMemoryError("TOOL_KV_JOINT_REQUIRES_EXPLICIT_PROTOCOL")
    if prepared.get("continue_final_message"):
        raise JointToolMemoryError("TOOL_KV_JOINT_CONTINUE_FINAL_UNSUPPORTED")
    messages = list(source_messages if source_messages is not None
                    else staged.get("messages") or [])
    final_messages = list(prepared.get("messages") or [])
    protocol = staged_hint.get("tool_protocol_span")
    hint = dict(prepared.get("c2kv_kv_memory_hint") or {})
    hint.pop("tool_kv_eviction", None)
    if not protocol:
        # A source-only catalog has no structured schema block to select.
        # Its protected copies remain raw in the system prefix; count those
        # slots plus any separately retained original source text explicitly.
        system = final_messages[0].get("content") if final_messages else None
        raw_protocol = str(plan.protocol or "")
        start_char = system.rfind(raw_protocol) if isinstance(system, str) and raw_protocol else -1
        if start_char < 0:
            raise JointToolMemoryError("TOOL_KV_JOINT_SOURCE_INTERFACE_MOVED")
        _, _, offsets, starts = _render(final_messages, tokenizer, prepared)
        visible = [m for m in final_messages if not (m.get("c2kv_key_hash")
                   or m.get("c2kv_repair_key_hashes") or m.get("c2kv_repair_only_key_hashes"))]
        protocol_indices = _indices({"message_index": 0, "start": start_char,
                                     "end": start_char + len(raw_protocol),
                                     "text": raw_protocol}, visible, offsets, starts,
                                    overlap=True)
        source_spans = (staged_hint.get("schema_spans") or
                        plan.assembled_schema_spans or ())
        source_tokens, source_indices = _retained_source_tokens(
            final_messages, source_spans, tokenizer,
            {**prepared, "_joint_protocol": raw_protocol})
        protocol_tokens = len(protocol_indices)
        resident = protocol_tokens + source_tokens
        budget = staged_hint.get("max_resident_tool_tokens", plan.info.get("budget_tokens"))
        if budget is not None and resident > int(budget):
            raise JointToolMemoryError("TOOL_KV_JOINT_PROTECTED_INTERFACE_EXCEEDS_BUDGET")
        audit = {"method": plan.spec.encoder,
                 "selection_backend": "raw_full_no_repair",
                 "history_composition": ("gist_carriers" if source_messages is not None
                                         else "raw_history_kv"),
                 "source_protocol_tokens": protocol_tokens,
                 "protected_protocol_tokens": protocol_tokens,
                 "selectable_protocol_tokens": 0,
                 "active_tool_protocol_tokens": protocol_tokens,
                 "retained_source_tool_tokens": source_tokens,
                 "retained_source_span_count": len(source_indices),
                 "raw_source_fallback_count": len(source_indices & set(
                     int(i) for i in plan.compressed_source_indices)),
                 "reference_tool_tokens": int(plan.info["matched_resident_tool_tokens"]),
                 "resident_tool_tokens": resident,
                 "resident_tool_tokens_accounting": (
                     "pre_history_eviction_upper_bound" if source_tokens and
                     isinstance(hint.get("history_kv_eviction"), dict) else "exact")}
        hint["joint_tool_memory"] = audit
        plan.info.update(audit)
        return {**prepared, "c2kv_kv_memory_hint": hint}
    rendered, ids, offsets, starts = _render(messages, tokenizer, staged)
    visible = [m for m in messages if not (m.get("c2kv_key_hash")
               or m.get("c2kv_repair_key_hashes") or m.get("c2kv_repair_only_key_hashes"))]
    if len(visible) != len(messages):
        raise JointToolMemoryError("TOOL_KV_JOINT_STAGED_CARRIERS_UNSUPPORTED")
    protocol_indices = _indices(protocol, visible, offsets, starts, overlap=True)
    start, end = protocol_indices[0], protocol_indices[-1] + 1
    if protocol_indices != list(range(start, end)):
        raise JointToolMemoryError("TOOL_KV_JOINT_PROTOCOL_DISCONTIGUOUS")
    n_structured = len(staged.get("tools") or [])
    native = set(int(i) for i in staged_hint.get("protected_schema_indices") or [])
    selectable: set[int] = set()
    source_spans = []
    for span in staged_hint.get("schema_spans") or []:
        if int(span["schema_index"]) >= n_structured:
            source_spans.append(span)
            continue
        if int(span["schema_index"]) in native:
            continue
        for index in _indices(span, visible, offsets, starts, overlap=False):
            if not start <= index < end:
                raise JointToolMemoryError("TOOL_KV_JOINT_SCHEMA_OUTSIDE_PROTOCOL")
            selectable.add(index - start)
    final_system = final_messages[0].get("content") if final_messages else None
    if (not isinstance(final_system, str) or int(protocol["message_index"]) != 0
            or final_system[int(protocol["start"]):int(protocol["end"])] != protocol["text"]):
        raise JointToolMemoryError("TOOL_KV_JOINT_PROTOCOL_MOVED_BY_HISTORY")
    source_tokens, source_indices = _retained_source_tokens(
        final_messages, source_spans, tokenizer,
        {**prepared, "_joint_protocol": protocol["text"]})
    source_count = len(source_indices)
    compressed_source = {n_structured + int(i) for i in plan.compressed_source_indices}
    native_source = source_indices - compressed_source
    length = end - start
    mandatory = length - len(selectable)
    reference = int(staged_hint.get("joint_tool_target_tokens_per_layer")
                    or plan.info["matched_resident_tool_tokens"])
    target = min(length, max(mandatory, reference - source_tokens))
    budget = staged_hint.get("max_resident_tool_tokens")
    nominal = target + source_tokens
    if budget is not None and nominal > int(budget):
        raise JointToolMemoryError("TOOL_KV_JOINT_PROTECTED_INTERFACE_EXCEEDS_BUDGET")
    audit = {"method": staged_hint["method"], "source_protocol_tokens": length,
             "source_protocol_token_sha256": hashlib.sha256(
                 json.dumps(list(ids[:end]), separators=(",", ":")).encode("ascii")).hexdigest(),
             "protected_protocol_tokens": mandatory,
             "selectable_protocol_tokens": len(selectable),
             "retained_source_tool_tokens": source_tokens,
             "retained_source_span_count": source_count,
             "native_source_span_count": len(native_source),
             "reference_tool_tokens": reference, "target_protocol_tokens": target,
             "history_composition": ("gist_carriers" if source_messages is not None
                                     else "raw_history_kv"),
             "nominal_resident_tool_tokens": nominal,
             "raw_source_fallback_count": len(source_indices & compressed_source),
             "resident_tool_tokens_accounting": (
                 "pre_history_eviction_upper_bound" if source_tokens and
                 isinstance(hint.get("history_kv_eviction"), dict) else "exact"),
             "selection_backend": "raw_full_no_repair"}
    if not selectable or target == length:
        audit["resident_tool_tokens"] = length + source_tokens
        audit["active_tool_protocol_tokens"] = length
        hint["joint_tool_memory"] = audit
        plan.info.update(audit)
        return {**prepared, "c2kv_kv_memory_hint": hint}

    # Removing the protocol must leave exactly the source's token suffix and
    # prefix.  A BPE token that straddles the boundary cannot be represented
    # by an in-place KV carrier; reject it instead of shifting the frame.
    reduced_system = final_system[:int(protocol["start"])] + final_system[int(protocol["end"]):]
    source_reduced = [dict(item) for item in messages]
    source_reduced[0] = {**source_reduced[0], "content": (
        str(source_reduced[0]["content"])[:int(protocol["start"])]
        + str(source_reduced[0]["content"])[int(protocol["end"]):])}
    _, reduced_ids, _, _ = _render(source_reduced, tokenizer, staged)
    if reduced_ids[:start] != ids[:start] or reduced_ids[start:] != ids[end:]:
        raise JointToolMemoryError("TOOL_KV_JOINT_PROTOCOL_TOKEN_BOUNDARY_CHANGED")
    final_reduced = [dict(item) for item in final_messages]
    final_reduced[0]["content"] = reduced_system
    _, final_ids, _, _ = _render(final_reduced, tokenizer, prepared)
    if final_ids[:start] != ids[:start]:
        raise JointToolMemoryError("TOOL_KV_JOINT_FINAL_PREFIX_CHANGED")
    receipt = backend.repair_extract_tool_protocol(
        ids, span_start=start, span_end=end, method=str(staged_hint["method"]),
        target_tokens=target, selectable_relative_indices=sorted(selectable),
        recent_window=int(staged_hint["recent_window"]),
        kernel_size=int(staged_hint["kernel_size"]),
        pooling=str(staged_hint["pooling"]),
        h2o_recent_fraction=float(staged_hint["h2o_recent_fraction"]))
    actual = int(receipt["token_len"]) + source_tokens
    if budget is not None and actual > int(budget):
        raise JointToolMemoryError("TOOL_KV_JOINT_PHYSICAL_STORAGE_EXCEEDS_BUDGET")
    carrier = {"role": "user", "content": "",
               "c2kv_repair_only_key_hashes": [receipt["key_hash"]],
               "c2kv_repair_placement": "in_place", "c2kv_region": "tool",
               "c2kv_repair_token_start": start,
               "c2kv_source_token_count": length,
               "c2kv_source_token_end": end,
               "c2kv_use_gist_projection": False}
    # The explicit token start lies inside the system message.  Carrier order
    # must follow token order: any history gist starts after this protocol.
    final_reduced.insert(1, carrier)
    for key in ("history_kv_eviction", "history_kv_reference_config"):
        config = hint.get(key)
        if isinstance(config, dict):
            config = dict(config)
            for count_key in ("history_start_message_count", "history_message_count"):
                if count_key in config and int(config[count_key]) >= 1:
                    config[count_key] = int(config[count_key]) + 1
            hint[key] = config
    audit.update(selection_backend="sglang_global_tool_repair",
                 repair_key_hash=receipt["key_hash"],
                 selected_protocol_tokens=int(receipt["token_len"]),
                 active_tool_protocol_tokens=int(receipt["token_len"]),
                 resident_tool_tokens=actual,
                 source_token_start=start, source_token_end=end,
                 repair_extract_prompt_tokens=len(ids))
    hint["joint_tool_memory"] = audit
    plan.info.update(audit)
    return {**prepared, "messages": final_reduced,
            "c2kv_kv_memory_hint": hint}
