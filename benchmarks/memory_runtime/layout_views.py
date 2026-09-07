"""Build the six fixed, CPU-only prompt views for the layout diagnostic."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .tokenization import TOOL_SCHEMA_PROFILE

_shared_path = str(Path(__file__).resolve().parents[2] / "python")
sys.path.insert(0, _shared_path)
try:
    from history_memory.evidence import evidence_message
    from history_memory.events import EventStore
    from history_memory.packing import visible_message
finally:
    sys.path.remove(_shared_path)


def _is_gist(message: Mapping[str, Any]) -> bool:
    return bool(message.get("c2kv_key_hash"))


def _index(counts: Mapping[str, Any], size: int, name: str) -> int:
    value = counts.get(name)
    if type(value) is not int or not 0 <= value <= size:
        raise ValueError(f"{name} must index the supplied message view")
    return value


def _count(
    token_counter: Callable[[list[dict[str, Any]], Any], int],
    messages: list[dict[str, Any]],
    tools: Any,
) -> int:
    value = token_counter(messages, tools)
    if type(value) is not int or value < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    return value


def _records_for(
    messages: list[dict[str, Any]],
    counts: Mapping[str, Any],
    *,
    allow_removed: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = copy.deepcopy(list(counts.get("compressed_records") or []))
    positions: dict[str, int] = {}
    for index, message in enumerate(messages):
        key = message.get("c2kv_key_hash")
        if not key:
            continue
        if not isinstance(key, str) or key in positions:
            raise ValueError("Each gist carrier needs one unique string key")
        positions[key] = index
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            key = record["record"]["key_hash"]
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed compressed_records entry") from error
        if not isinstance(key, str) or key in by_key:
            raise ValueError("compressed_records needs unique string key_hash values")
        by_key[key] = record
    if not allow_removed and set(positions) != set(by_key):
        raise ValueError("Gist carriers and compressed_records disagree")
    if set(positions) - set(by_key):
        raise ValueError("A gist carrier has no compressed record")
    retained = []
    for record in records:
        key = record["record"]["key_hash"]
        if key in positions:
            record["out_index"] = positions[key]
            retained.append(record)
    removed = [key for key in by_key if key not in positions]
    return retained, removed


def _measured_view(
    *,
    label: str,
    messages: list[dict[str, Any]],
    source_counts: Mapping[str, Any],
    current_start: int,
    evidence_indices: tuple[int, ...],
    selected_event_ids: tuple[str, ...],
    quoted_event_ids: tuple[str, ...],
    common_tokens: int,
    token_counter: Callable[[list[dict[str, Any]], Any], int],
    tools: Any,
    store: EventStore,
    bytes_per_kv_token: int,
    history_budget_bytes: int,
    workspace_budget_bytes: int,
    note: str,
    allow_removed_gists: bool = False,
) -> dict[str, Any]:
    view = copy.deepcopy(messages)
    if any(type(index) is not int or not 0 <= index < len(view) for index in evidence_indices):
        raise ValueError("Evidence index is outside its message view")
    records, removed_gists = _records_for(
        view, source_counts, allow_removed=allow_removed_gists
    )
    gist_tokens = sum(int(record["record"]["gist_len"]) for record in records)
    original_tokens = sum(
        int(record["record"].get("original_seq_len", 0)) for record in records
    )
    raw = [message for message in view if not _is_gist(message)]
    total_raw_tokens = _count(token_counter, raw, tools)
    raw_history_tokens = total_raw_tokens - common_tokens
    if raw_history_tokens < 0:
        raise ValueError(f"{label} raw history token delta became negative")
    evidence_free = [
        message for index, message in enumerate(view)
        if not _is_gist(message) and index not in evidence_indices
    ]
    evidence_tokens = total_raw_tokens - _count(token_counter, evidence_free, tools)
    if evidence_tokens < 0:
        raise ValueError(f"{label} evidence renderer produced a negative token delta")
    evidence_bytes = evidence_tokens * bytes_per_kv_token
    active_history_bytes = (gist_tokens + raw_history_tokens) * bytes_per_kv_token
    if label not in {"full", "full_aux"} and active_history_bytes > history_budget_bytes:
        raise ValueError(f"{label} exceeds history_budget_bytes")
    if evidence_indices and evidence_bytes > workspace_budget_bytes:
        raise ValueError(f"{label} exceeds workspace_budget_bytes")

    updated = copy.deepcopy(dict(source_counts))
    metadata = copy.deepcopy(updated.get("memory_runtime") or {})
    source_mode = metadata.get("mode")
    for stale in (
        "backend_bytes_per_kv_token", "backend_raw_prompt_tokens", "backend_tools_dump",
        "byte_geometry_error", "byte_geometry_verification",
    ):
        metadata.pop(stale, None)
    selected_sources = {
        index for event_id in selected_event_ids
        for index in store.event(event_id).source_indices
    }
    old_blocks = {
        block.get("key_hash"): block
        for block in metadata.get("block_refs") or [] if isinstance(block, dict)
    }
    metadata.update({
        "mode": label,
        "source_mode": source_mode,
        "layout_probe": True,
        "tool_schema_profile": TOOL_SCHEMA_PROFILE,
        "c2kv_tools_dump_expected": "full",
        "bytes_per_kv_token": bytes_per_kv_token,
        "byte_geometry_verified_by_backend": False,
        "raw_prompt_tokens_verified_by_backend": False,
        "history_budget_bytes": history_budget_bytes,
        "workspace_budget_bytes": workspace_budget_bytes,
        "budget_applies": label not in {"full", "full_aux"},
        "active_history_bytes": active_history_bytes,
        "evidence_bytes": evidence_bytes,
        "gist_tokens": gist_tokens,
        "raw_history_tokens": raw_history_tokens,
        "common_raw_prompt_tokens": common_tokens,
        "total_raw_prompt_tokens": total_raw_tokens,
        "selected_event_ids": list(selected_event_ids),
        "quoted_evidence_event_ids": list(quoted_event_ids),
        "evidence_out_index": evidence_indices[0] if quoted_event_ids else None,
        "block_refs": [copy.deepcopy(old_blocks[key]) for key in (
            record["record"]["key_hash"] for record in records
        ) if key in old_blocks],
        "overlapping_gist_keys": [
            key for record in records
            if (key := record["record"]["key_hash"]) in old_blocks
            and selected_sources.intersection(old_blocks[key].get("source_indices") or [])
        ],
        "layout_removed_gist_keys": removed_gists,
    })
    updated.update({
        "memory_runtime": metadata,
        "compressed_records": records,
        "gist_tokens": gist_tokens,
        "original_tokens": original_tokens,
        "n_gist_messages": len(records),
        "compressed": len(records),
        "n_docs": len(records),
        "dropped_docs": int(source_counts.get("dropped_docs", 0)) + len(removed_gists),
        "current_start_out_index": current_start,
        "history_raw": sum(
            message.get("role") != "system" and not _is_gist(message)
            for message in view[:current_start]
        ),
    })
    denominator = updated.get("history_packed_original_tokens")
    if denominator is not None:
        updated["history_dropped_original_tokens"] = int(denominator) - original_tokens
        updated["history_retained_fraction"] = (
            original_tokens / int(denominator) if int(denominator) else None
        )
    return {"label": label, "messages": view, "counts": updated, "notes": note}


def build_layout_views(
    store: EventStore,
    full_messages: Sequence[Mapping[str, Any]],
    full_counts: Mapping[str, Any],
    legacy_messages: Sequence[Mapping[str, Any]],
    legacy_counts: Mapping[str, Any],
    protect_messages: Sequence[Mapping[str, Any]],
    protect_counts: Mapping[str, Any],
    token_counter: Callable[[list[dict[str, Any]], Any], int],
    tools: Any,
    bytes_per_kv_token: int,
    history_budget_bytes: int,
    workspace_budget_bytes: int,
) -> list[dict[str, Any]]:
    """Return fixed views without selecting evidence or changing production policy."""
    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if type(bytes_per_kv_token) is not int or bytes_per_kv_token <= 0:
        raise ValueError("bytes_per_kv_token must be a positive integer")
    if any(type(value) is not int or value < 0 for value in (
        history_budget_bytes, workspace_budget_bytes
    )):
        raise ValueError("history and workspace budgets must be nonnegative integers")
    full = [copy.deepcopy(dict(message)) for message in full_messages]
    legacy = [copy.deepcopy(dict(message)) for message in legacy_messages]
    protect = [copy.deepcopy(dict(message)) for message in protect_messages]
    full_start = _index(full_counts, len(full), "current_start_out_index")
    legacy_start = _index(legacy_counts, len(legacy), "current_start_out_index")
    protect_start = _index(protect_counts, len(protect), "current_start_out_index")
    if not (full[full_start:] == legacy[legacy_start:] == protect[protect_start:]):
        raise ValueError("All supplied views must have the same current raw suffix")

    protect_meta = protect_counts.get("memory_runtime")
    if not isinstance(protect_meta, Mapping):
        raise ValueError("protect_counts needs memory_runtime metadata")
    selected = protect_meta.get("selected_event_ids")
    if not isinstance(selected, list) or not selected or any(
        not isinstance(event_id, str) for event_id in selected
    ):
        raise ValueError("Protection view needs a nonempty selected_event_ids list")
    selected_ids = tuple(selected)
    evidence_index = protect_meta.get("evidence_out_index")
    if type(evidence_index) is not int or evidence_index != protect_start - 1:
        raise ValueError("Protection evidence must immediately precede the current raw suffix")
    packet = evidence_message(store, selected_ids)
    if packet is None or protect[evidence_index] != packet:
        raise ValueError("Protection evidence packet does not match its selected events")

    users = [event for event in store.events if event.kind == "user"]
    if not users or users[-1].event_id not in selected_ids:
        raise ValueError("Protection evidence must contain the current user event")
    current_user = users[-1]
    current_messages = store.event_messages(current_user.event_id)
    if len(current_messages) != 1:
        raise ValueError("The current user event must contain exactly one message")
    active_user = visible_message(current_messages[0])
    remaining_ids = tuple(
        event_id for event_id in selected_ids if event_id != current_user.event_id
    )
    remaining_packet = evidence_message(store, remaining_ids)

    common = [message for message in legacy if not _is_gist(message)]
    common_tokens = _count(token_counter, common, tools)
    full_aux = full[:full_start] + [copy.deepcopy(packet)] + full[full_start:]
    evidence_only = [message for message in protect if not _is_gist(message)]
    evidence_only_index = sum(
        not _is_gist(message) for message in protect[:evidence_index]
    )
    evidence_only_start = sum(
        not _is_gist(message) for message in protect[:protect_start]
    )
    active_prefix = protect[:evidence_index]
    active_insert = ([] if remaining_packet is None else [remaining_packet]) + [active_user]
    active_query_raw = active_prefix + active_insert + protect[protect_start:]
    active_evidence_indices = tuple(
        range(len(active_prefix), len(active_prefix) + len(active_insert))
    )

    specs = [
        ("full", full, full_counts, full_start, (), (), (),
         "Unmodified full-raw identity control.", False),
        ("full_aux", full_aux, full_counts, full_start + 1, (full_start,),
         selected_ids, selected_ids, "Full raw plus the exact protection evidence packet.", False),
        ("legacy", legacy, legacy_counts, legacy_start, (), (), (),
         "Unmodified legacy gist view.", False),
        ("protect", protect, protect_counts, protect_start, (evidence_index,),
         selected_ids, selected_ids, "Unmodified production protection view.", False),
        ("evidence_only", evidence_only, protect_counts, evidence_only_start,
         (evidence_only_index,), selected_ids, selected_ids,
         "Protection evidence without gist carriers; deliberately smaller, not a fair NoGist baseline.", True),
        ("active_query_raw", active_query_raw, protect_counts,
         len(active_prefix) + len(active_insert), active_evidence_indices,
         selected_ids, remaining_ids,
         "Same evidence set with the current user restored as an actual user message.", False),
    ]
    return [
        _measured_view(
            label=label, messages=messages, source_counts=counts,
            current_start=current_start, evidence_indices=evidence_indices,
            selected_event_ids=event_ids, quoted_event_ids=quoted_ids,
            common_tokens=common_tokens, token_counter=token_counter, tools=tools,
            store=store, bytes_per_kv_token=bytes_per_kv_token,
            history_budget_bytes=history_budget_bytes,
            workspace_budget_bytes=workspace_budget_bytes, note=note,
            allow_removed_gists=allow_removed,
        )
        for (
            label, messages, counts, current_start, evidence_indices,
            event_ids, quoted_ids, note, allow_removed,
        ) in specs
    ]
