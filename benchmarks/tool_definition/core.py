"""Portable, target-independent layouts for recorded tool decisions.

The T0 document envelope, protocol, chunking and lexical ranker are shared
with the live paper proxy.  A row's gold action is never read by a selector.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks import toolmemory

RUNTIME_PYTHON = Path(__file__).resolve().parents[2] / "experiments" / "history_system" / "runtime" / "python"


def runtime_modules():
    # The bundled history_memory and Qwen3 implementation are part of the
    # paper checkout.  No mutable training checkout is imported at runtime.
    if str(RUNTIME_PYTHON) not in sys.path:
        sys.path.insert(0, str(RUNTIME_PYTHON))
    from history_memory.events import EventStore
    from history_memory.packing import EncoderChunk, MemoryView, PackedMemory, native_ids
    return EventStore, EncoderChunk, MemoryView, PackedMemory, native_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_calls(calls: Any) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        raise ValueError("gold_tool_calls must be a list")
    result = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("tool call must be an object")
        function = call.get("function", call)
        if not isinstance(function, Mapping):
            raise ValueError("tool function must be an object")
        name, arguments = function.get("name"), function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("tool call needs a name and object arguments")
        result.append({"name": name, "arguments": arguments})
    return result


def parse_calls(text: str) -> list[dict[str, Any]]:
    """Parse complete Qwen tool-call objects, including tags inside strings."""
    if not isinstance(text, str):
        raise ValueError("generated action must be text")
    calls, cursor = [], 0
    decoder = json.JSONDecoder()
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            if "</tool_call>" in text[cursor:]:
                raise ValueError("unmatched closing tool-call tag")
            return canonical_calls(calls)
        payload = text[start + len("<tool_call>"):].lstrip()
        value, end = decoder.raw_decode(payload)
        remainder = payload[end:].lstrip()
        if not remainder.startswith("</tool_call>") or not isinstance(value, dict):
            raise ValueError("incomplete tool call")
        calls.append(value)
        cursor = len(text) - len(remainder) + len("</tool_call>")


def random_rank(count: int, *, seed: int, decision_id: str) -> tuple[int, ...]:
    order = list(range(count))
    random.Random(f"{seed}:{decision_id}").shuffle(order)
    return tuple(order)


def _render(messages: Sequence[Mapping[str, Any]], native_tools: Sequence[Any], tokenizer: Any):
    _, _, _, _, native_ids = runtime_modules()
    protocol = toolmemory.protocol_block(native_tools)
    visible = toolmemory.with_protocol_system(messages, protocol)
    full_ids = native_ids(tokenizer, visible, generation=True)
    system = []
    for message in visible:
        if message["role"] != "system":
            break
        system.append(message)
    dummy = {"role": "user", "content": ""}
    dummy_ids = native_ids(tokenizer, [dummy])
    combined = native_ids(tokenizer, [*system, dummy])
    if not dummy_ids or combined[-len(dummy_ids):] != dummy_ids:
        raise ValueError("native tool protocol has no separable system prefix")
    prefix = combined[:-len(dummy_ids)]
    if full_ids[:len(prefix)] != prefix or len(full_ids) == len(prefix):
        raise ValueError("native tool prefix changed with workspace")
    return prefix, full_ids[len(prefix):]


def pack_layout(
    row: Mapping[str, Any], tokenizer: Any, *, layout: str, ratio: int,
    k: int = 3, seed: int = 42, native_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Pack Full, T0, or a selected native subset on the same visible prefix."""
    EventStore, EncoderChunk, MemoryView, PackedMemory, native_ids = runtime_modules()
    decision_id = row.get("decision_id")
    messages, tools = row.get("messages"), row.get("tools")
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError("decision_id must be nonempty")
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(item, Mapping) for item in messages)):
        raise ValueError("messages must be a nonempty list of objects")
    if (not isinstance(tools, list) or not tools
            or any(not isinstance(item, Mapping) for item in tools)):
        raise ValueError("tools must be a nonempty list of objects")
    if ratio not in toolmemory.SUPPORTED_RATIOS:
        raise ValueError("T0 ratio must be 8 or 12")
    if layout not in {"full", "uniform", "hybrid", "random", "retrieval"}:
        raise ValueError("unknown tool layout")
    snapshots = [toolmemory.tool_snapshot(item) for item in tools]
    rank = toolmemory.lexical_rank(snapshots, toolmemory.query_text(messages))
    if native_override is not None:
        native = tuple(sorted(int(index) for index in native_override))
    elif layout == "full":
        native = tuple(range(len(tools)))
    elif layout == "uniform":
        native = ()
    elif layout == "random":
        native = tuple(sorted(random_rank(len(tools), seed=seed, decision_id=decision_id)[:k]))
    else:
        native = tuple(sorted(rank[:k]))
    if len(set(native)) != len(native) or any(not 0 <= index < len(tools) for index in native):
        raise ValueError("native indices are invalid")
    native_set = set(native)
    compressed = tuple(index for index in range(len(tools)) if index not in native_set)
    if layout in {"full", "retrieval"}:
        compressed = ()
    prefix, workspace = _render(messages, [snapshots[index] for index in native], tokenizer)
    chunks = ()
    if compressed:
        spec = toolmemory.ToolMemorySpec(ratio=ratio)
        source_chunks = toolmemory.document_chunks(
            lambda value: native_ids(tokenizer, value),
            toolmemory.t0_documents(snapshots, compressed), spec,
        )
        chunks = tuple(EncoderChunk(chunk.event_id, chunk.part_index, (),
                                    chunk.source_token_start, chunk.source_token_end,
                                    chunk.token_ids) for chunk in source_chunks)
    store = EventStore.from_messages(str(row.get("session_key") or decision_id), messages)
    view = MemoryView((), tuple(event.event_id for event in store.events))
    memory = PackedMemory(view, prefix, workspace, tuple(range(len(messages))), chunks)
    costs = memory.costs(ratio)
    return {
        "decision_id": decision_id, "session_key": store.session_id,
        "source": row.get("source", "recorded_decision"), "layout": layout,
        "ratio": ratio, "k": k if layout in {"hybrid", "random", "retrieval"} else None,
        "seed": seed if layout == "random" else None,
        "gold_tool_calls": canonical_calls(row["gold_tool_calls"]),
        "native_indices": list(native), "lexical_rank": list(rank),
        "memory": asdict(memory), "resident_kv_tokens": costs["resident_kv_tokens"],
        "tool_gist_tokens": costs["gist_tokens"],
        "prompt_sha256": hashlib.sha256(json.dumps({"messages": messages, "tools": tools},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest(),
        "tool_names": [toolmemory.tool_name(tool) for tool in snapshots],
    }


def retrieval_layout(row: Mapping[str, Any], tokenizer: Any, *, ratio: int,
                     allowance_tokens: int, k: int = 3) -> dict[str, Any]:
    """Admit lexical-ranked schemas while the total resident KV fits hybrid."""
    chosen: list[int] = []
    rank = toolmemory.lexical_rank(row["tools"], toolmemory.query_text(row["messages"]))
    for index in rank:
        candidate = sorted([*chosen, index])
        packed = pack_layout(row, tokenizer, layout="retrieval", ratio=ratio,
                             k=k, native_override=candidate)
        if packed["resident_kv_tokens"] <= allowance_tokens:
            chosen = candidate
    result = pack_layout(row, tokenizer, layout="retrieval", ratio=ratio,
                         k=k, native_override=chosen)
    result["allowance_tokens"] = allowance_tokens
    return result


def deserialize_memory(value: Mapping[str, Any]):
    _, EncoderChunk, MemoryView, PackedMemory, _ = runtime_modules()
    view = MemoryView(**{key: tuple(value["view"].get(key, ())) for key in
                         ("gist_event_ids", "raw_event_ids", "evidence_event_ids")})
    chunks = tuple(EncoderChunk(
        str(chunk["event_id"]), int(chunk["part_index"]),
        tuple(chunk["source_indices"]), int(chunk["source_token_start"]),
        int(chunk["source_token_end"]), tuple(chunk["token_ids"]),
        chunk.get("projection_set"), chunk.get("compression_ratio"),
    ) for chunk in value["chunks"])
    return PackedMemory(view, tuple(value["system_input_ids"]),
                        tuple(value["workspace_input_ids"]),
                        tuple(value["raw_source_indices"]), chunks,
                        value.get("raw_layout_profile", "event-native-evidence-v1"))


def full_tool_spans(row: Mapping[str, Any], tokenizer: Any, *, ratio: int) -> tuple[tuple[int, int], ...]:
    """Partition the native catalog region by tool in original catalog order."""
    tools = row["tools"]
    prefixes = [tuple(pack_layout(row, tokenizer, layout="full", ratio=ratio,
                   native_override=range(count))["memory"]["system_input_ids"])
                for count in range(len(tools) + 1)]
    full, empty = prefixes[-1], prefixes[0]
    common_head = 0
    while common_head < min(len(full), len(empty)) and full[common_head] == empty[common_head]:
        common_head += 1
    common_tail = 0
    while (common_tail < min(len(full), len(empty))
           and full[-1 - common_tail] == empty[-1 - common_tail]):
        common_tail += 1
    end = len(full) - common_tail
    lengths = [len(right) - len(left) for left, right in zip(prefixes, prefixes[1:])]
    start = end - sum(lengths)
    if any(length <= 0 for length in lengths) or start < 0 or start > common_head:
        raise ValueError("native tool spans cannot be separated by the checkpoint tokenizer")
    spans = []
    for length in lengths:
        spans.append((start, start + length))
        start += length
    return tuple(spans)
