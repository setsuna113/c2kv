"""Optional visible-tool planning around the unchanged history controller.

The tool catalog lives in the paper checkout. This adapter keeps its plan on
each prepared decision, including decisions revisited for recovery. OFF does
not construct this adapter and therefore preserves the original request path.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from history_memory.events import EventStore
from history_memory.packing import (
    EncoderChunk, PackedMemory, native_ids, raw_workspace_messages, visible_message,
)
from .event_native_raw import RuntimeMemoryView, render_raw_control_messages


_CATALOG_NAME = "c2kv_shared_toolmemory"


def shared_tool_catalog():
    module = sys.modules.get(_CATALOG_NAME)
    if module is not None:
        return module
    path = Path(__file__).resolve().parents[5] / "benchmarks" / "toolmemory.py"
    spec = importlib.util.spec_from_file_location(_CATALOG_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared tool catalog at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CATALOG_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_CATALOG_NAME, None)
        raise
    return module


def parse_native_tool_spec(value: str | None):
    if value is None or value.strip().lower() in {"", "none", "full", "raw"}:
        return None
    return shared_tool_catalog().parse_tool_memory_spec(value)


def validate_ready_tool_contract(manifest: Mapping[str, Any], tool_memory: str | None,
                                 checkpoint: str | Path | None = None,
                                 budget_tokens: int | None = None) -> None:
    shared_tool_catalog().validate_ready_tool_contract(
        manifest, tool_memory, checkpoint, budget_tokens)


class _TokenizerView:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def native_ids(self, messages):
        return native_ids(self.tokenizer, messages)


@dataclass(frozen=True)
class ToolPrepared:
    inner: Any
    memory: PackedMemory
    metadata: dict[str, Any]
    plan: Any
    eligible_chunks: tuple[EncoderChunk, ...] | None
    ratio: int
    max_new_tokens: int
    source_payload: dict[str, Any]

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class ToolRegionController:
    """Preserve a single visible catalog through draft and every recovery view."""

    def __init__(self, inner: Any, tokenizer: Any, spec: Any, *,
                 model_context: int, generator: Any,
                 tool_budget_tokens: int | None = None,
                 tool_checkpoint_contract: Mapping[str, Any] | None = None):
        self.inner = inner
        self.tokenizer = tokenizer
        self.spec = spec
        self.model_context = model_context
        self.generator = generator
        self.tool_budget_tokens = tool_budget_tokens
        self.tool_checkpoint_contract = dict(tool_checkpoint_contract or {})
        self._original_tools: dict[str, str] = {}

    def __getattr__(self, name: str):
        if name == "advance_recovery":
            advance = getattr(self.inner, name)
            return lambda prepared, **kwargs: advance(prepared.inner, **kwargs)
        return getattr(self.inner, name)

    def _plan(self, payload: Mapping[str, Any]):
        catalog = shared_tool_catalog()
        plan = catalog.plan_visible_tool_memory(
            payload, self.spec,
            _TokenizerView(self.tokenizer) if self.spec.encoder == "t0" else None,
        )
        if plan is not None and self.spec.encoder == "t0":
            catalog.enforce_tool_budget(plan, self.tool_budget_tokens)
        return plan

    def _controller_payload(self, payload: Mapping[str, Any], plan: Any):
        catalog = shared_tool_catalog()
        visible = dict(catalog.strip_request_annotations(payload))
        if plan is not None and self.spec.encoder == "t0":
            visible["messages"] = plan.messages
            if visible.get("tools"):
                visible["tools"] = []
        return visible

    def _whole_full_tokens(self, payload: Mapping[str, Any]) -> int:
        store = EventStore.from_messages(payload["session_id"], payload["messages"])
        full_view = RuntimeMemoryView(
            gist_event_ids=(),
            raw_event_ids=tuple(event.event_id for event in store.events),
            evidence_event_ids=(),
            raw_control_layout="full-original-native-v1",
        )
        messages = render_raw_control_messages(store, full_view)
        return len(native_ids(self.tokenizer, messages,
                              tools=payload.get("tools"), generation=True))

    def _augment(self, memory: PackedMemory, plan: Any, payload: Mapping[str, Any], *,
                 ratio: int, max_new_tokens: int):
        if plan is None:
            return memory
        if self.spec.encoder != "t0":
            return self._raw_tool_memory(memory, plan, payload)
        converted = tuple(EncoderChunk(
            event_id=chunk.event_id,
            part_index=chunk.part_index,
            source_indices=(0,),
            source_token_start=chunk.source_token_start,
            source_token_end=chunk.source_token_end,
            token_ids=tuple(chunk.token_ids),
            projection_set="tool",
            compression_ratio=self.spec.ratio,
        ) for chunk in plan.chunks)
        if not converted:
            return memory
        by_catalog = {}
        for original, chunk in zip(plan.chunks, converted, strict=True):
            by_catalog.setdefault(original.catalog_index, []).append(chunk)
        memory, locations = self._anchor_token_spans(memory, plan.carrier_anchors)
        anchored = []
        anchored_catalog = set()
        for anchor in plan.carrier_anchors:
            group = tuple(by_catalog[anchor["catalog_index"]])
            location = locations.get(anchor["catalog_index"])
            if location is None:
                continue
            anchored.append({"token_start": location[0], "token_end": location[1],
                             "chunks": group})
            anchored_catalog.add(anchor["catalog_index"])
        prefix_chunks = tuple(chunk for original, chunk in zip(plan.chunks, converted, strict=True)
                              if original.catalog_index not in anchored_catalog)
        prefix_source_tokens = sum(len(chunk.token_ids) for chunk in prefix_chunks)
        if prefix_source_tokens:
            for segment in anchored:
                if segment["token_start"] >= len(memory.system_input_ids):
                    segment["token_start"] += prefix_source_tokens
                    segment["token_end"] += prefix_source_tokens
        augmented = replace(memory, chunks=prefix_chunks + memory.chunks,
                            tool_gist_segments=tuple(anchored))
        anchor_delta = sum(sum(len(chunk.token_ids) for chunk in segment["chunks"])
                           - (segment["token_end"] - segment["token_start"])
                           for segment in anchored)
        logical_end = (augmented.workspace_position_start
                       + len(augmented.workspace_input_ids) + anchor_delta)
        resident_end = augmented.costs(ratio)["resident_kv_tokens"]
        if max(logical_end, resident_end) + max_new_tokens > self.model_context:
            raise ValueError("Tool and history packing exceed the model context")
        return augmented

    def _anchor_token_spans(self, memory: PackedMemory, anchors):
        """Isolate inline placeholders without removing neighboring source text.

        A BPE token can cross an ACE/AppWorld JSON-string boundary. In that
        case, encode the unchanged text on each side separately so the tool
        replacement occupies whole tokens and every other character survives.
        """
        regions = {}
        found = {}
        for name in ("system_input_ids", "workspace_input_ids"):
            original = tuple(getattr(memory, name))
            text = self.tokenizer.decode(
                list(original), skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            matches = []
            for anchor in anchors:
                placeholder = anchor["placeholder"]
                start = text.find(placeholder)
                if start < 0:
                    continue
                if text.find(placeholder, start + 1) >= 0:
                    raise ValueError("A source tool anchor appears more than once in the packed prompt")
                catalog_index = anchor["catalog_index"]
                if catalog_index in found:
                    raise ValueError("A source tool anchor crosses the packed system boundary")
                found[catalog_index] = name
                matches.append((start, start + len(placeholder), catalog_index, placeholder))
            matches.sort()
            if any(left[1] > right[0] for left, right in zip(matches, matches[1:])):
                raise ValueError("Source tool anchors overlap")
            if not matches:
                regions[name] = (original, {})
                continue
            encoded = self.tokenizer(text, add_special_tokens=False,
                                     return_offsets_mapping=True)
            offsets = tuple(tuple(pair) for pair in encoded["offset_mapping"])
            aligned = tuple(encoded["input_ids"]) == original
            positions = {}
            if aligned:
                for start, end, catalog_index, _ in matches:
                    overlap = [index for index, (left, right) in enumerate(offsets)
                               if right > start and left < end]
                    if (not overlap or offsets[overlap[0]][0] != start
                            or offsets[overlap[-1]][1] != end):
                        aligned = False
                        break
                    positions[catalog_index] = (overlap[0], overlap[-1] + 1)
            if aligned:
                regions[name] = (original, positions)
                continue
            pieces = []
            cursor = 0
            for start, end, catalog_index, placeholder in matches:
                pieces.extend(self.tokenizer(text[cursor:start], add_special_tokens=False)["input_ids"])
                token_start = len(pieces)
                pieces.extend(self.tokenizer(placeholder, add_special_tokens=False)["input_ids"])
                positions[catalog_index] = (token_start, len(pieces))
                cursor = end
            pieces.extend(self.tokenizer(text[cursor:], add_special_tokens=False)["input_ids"])
            recoded = tuple(int(token) for token in pieces)
            if self.tokenizer.decode(
                    list(recoded), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) != text:
                raise ValueError("Cannot preserve source text while aligning tool anchors")
            regions[name] = (recoded, positions)
        system, system_positions = regions["system_input_ids"]
        workspace, workspace_positions = regions["workspace_input_ids"]
        offset = len(system) + sum(len(chunk.token_ids) for chunk in memory.chunks)
        locations = dict(system_positions)
        locations.update({index: (start + offset, end + offset)
                          for index, (start, end) in workspace_positions.items()})
        return replace(memory, system_input_ids=system,
                       workspace_input_ids=workspace), locations

    def _raw_tool_memory(self, memory: PackedMemory, plan: Any,
                         payload: Mapping[str, Any]) -> PackedMemory:
        logical_ids = (tuple(memory.system_input_ids)
                       + tuple(token for chunk in memory.chunks for token in chunk.token_ids)
                       + tuple(memory.workspace_input_ids))
        spans = []
        tools = payload.get("tools") or []
        if tools:
            prefix_messages = []
            for message in payload["messages"]:
                if message.get("role") != "system":
                    break
                prefix_messages.append(message)
            dummy = {"role": "user", "content": ""}
            with_tools = native_ids(self.tokenizer, prefix_messages + [dummy], tools=tools)
            without = native_ids(self.tokenizer, prefix_messages + [dummy])
            dummy_ids = native_ids(self.tokenizer, [dummy])
            if (not dummy_ids or with_tools[-len(dummy_ids):] != dummy_ids
                    or without[-len(dummy_ids):] != dummy_ids):
                raise ValueError("Cannot separate the native tool prologue")
            with_prefix = with_tools[:-len(dummy_ids)]
            without_prefix = without[:-len(dummy_ids)]
            if tuple(memory.system_input_ids) != with_prefix:
                raise ValueError("Tool prologue changed during history packing")
            left = 0
            while left < min(len(with_prefix), len(without_prefix)) and with_prefix[left] == without_prefix[left]:
                left += 1
            right = 0
            while (right < min(len(with_prefix) - left, len(without_prefix) - left)
                   and with_prefix[-right - 1] == without_prefix[-right - 1]):
                right += 1
            end = len(with_prefix) - right
            if left >= end:
                raise ValueError("Visible structured tools have no native token span")
            spans.append((left, end, "structured_tools"))
        if plan.source_spans:
            spans.extend(self._source_token_spans(memory, payload, plan))
        spans.sort()
        if any(first[1] > second[0] for first, second in zip(spans, spans[1:])):
            raise ValueError("Visible tool regions overlap after tokenization")
        joined = []
        for start, end, source in spans:
            if joined and joined[-1][1] == start:
                joined[-1] = (joined[-1][0], end, joined[-1][2] + "+" + source)
            else:
                joined.append((start, end, source))
        spans = joined
        if len(spans) > 1:
            raise ValueError(
                "Global H2O/SnapKV tool selection requires one contiguous visible tool region")
        requested = sum((end - start + self.spec.ratio - 1) // self.spec.ratio
                        for start, end, _ in spans)
        if self.tool_budget_tokens is not None and requested > self.tool_budget_tokens:
            raise ValueError("Tool retained-token target exceeds its separate budget")
        segments = []
        for start, end, source in spans:
            target = (end - start + self.spec.ratio - 1) // self.spec.ratio
            receipt = self.generator.repair_tool_span(
                logical_ids, span_start=start, span_end=end,
                method=self.spec.encoder, target_tokens=target)
            segments.append({
                "token_start": start, "token_end": end,
                "repair_key_hashes": [receipt["key_hash"]],
                "token_len": receipt["token_len"],
                "repair_placement": "in_place",
                "source": source,
            })
        # The engine accepts only the transport fields; source remains in the
        # controller receipt and never enters the native packed request.
        return replace(memory, raw_tool_segments=tuple({
            key: value for key, value in segment.items() if key != "source"
        } for segment in segments))

    def _source_token_spans(self, memory: PackedMemory, payload: Mapping[str, Any],
                            plan: Any) -> list[tuple[int, int, str]]:
        store = EventStore.from_messages(payload["session_id"], payload["messages"])
        if isinstance(memory.view, RuntimeMemoryView):
            rendered_messages = list(render_raw_control_messages(store, memory.view))
        else:
            rendered_messages = list(raw_workspace_messages(store, memory.view))
        source_to_rendered = {}
        cursor = 0
        for source_index in memory.raw_source_indices:
            source = visible_message(store.messages[source_index])
            for index in range(cursor, len(rendered_messages)):
                if rendered_messages[index] == source:
                    source_to_rendered[source_index] = index
                    cursor = index + 1
                    break
        tools = payload.get("tools") or None
        template = self.tokenizer.apply_chat_template
        full_text = template(rendered_messages, tools=tools, tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(full_text, add_special_tokens=False,
                                 return_offsets_mapping=True)
        ids = tuple(encoded["input_ids"])
        offsets = tuple(tuple(pair) for pair in encoded["offset_mapping"])
        if ids != tuple(memory.system_input_ids) + tuple(memory.workspace_input_ids):
            raise ValueError("Cannot align visible source spans with the native packed prompt")
        chunk_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
        spans = []
        for source_index in plan.compressed_source_indices:
            source = plan.source_spans[source_index]
            rendered_index = source_to_rendered.get(source.message_index)
            if rendered_index is None:
                continue  # Already represented by the selected history view.
            current = template(rendered_messages[:rendered_index + 1], tools=tools,
                               tokenize=False, add_generation_prompt=False,
                               enable_thinking=False)
            previous = (template(rendered_messages[:rendered_index], tools=tools,
                                 tokenize=False, add_generation_prompt=False,
                                 enable_thinking=False)
                        if rendered_index else "")
            content_start = current.find(rendered_messages[rendered_index]["content"], len(previous))
            if content_start < 0:
                raise ValueError("Cannot locate the annotated source message in native rendering")
            char_start = content_start + source.start
            char_end = content_start + source.end
            overlapping = [index for index, (start, end) in enumerate(offsets)
                           if end > char_start and start < char_end]
            if (not overlapping or offsets[overlapping[0]][0] != char_start
                    or offsets[overlapping[-1]][1] != char_end):
                raise ValueError("Tool source span is not aligned to tokenizer boundaries")
            token_start, token_end = overlapping[0], overlapping[-1] + 1
            if token_start >= len(memory.system_input_ids):
                token_start += chunk_tokens
                token_end += chunk_tokens
            elif token_end > len(memory.system_input_ids):
                raise ValueError("Source tool span crosses the packed system boundary")
            spans.append((token_start, token_end, source.source))
        return spans

    def prepare(self, payload: Mapping[str, Any], *, ratio: int, max_new_tokens: int):
        session_id = payload["session_id"]
        tools_json = json.dumps(payload.get("tools") or [], sort_keys=True,
                                ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        previous = self._original_tools.get(session_id)
        if previous is not None and previous != tools_json:
            raise ValueError("Tools changed within a session; use a new explicit session_id")
        whole_full_tokens = self._whole_full_tokens(payload)
        plan = self._plan(payload)
        base = self.inner.prepare(
            self._controller_payload(payload, plan), ratio=ratio,
            max_new_tokens=max_new_tokens)
        self._original_tools.setdefault(session_id, tools_json)
        memory = self._augment(base.memory, plan, payload, ratio=ratio,
                               max_new_tokens=max_new_tokens)
        metadata = copy.deepcopy(base.metadata)
        metadata["paper_whole_full_kv_tokens"] = whole_full_tokens
        metadata["history_only_resident_kv_tokens"] = base.memory.costs(ratio)["resident_kv_tokens"]
        metadata["history_only_gist_tokens"] = base.memory.costs(ratio)["gist_tokens"]
        metadata["tool_memory"] = (
            {"status": "no_visible_tool_region", "spec": self.spec.name}
            if plan is None else copy.deepcopy(plan.info)
        )
        if self.tool_checkpoint_contract:
            metadata["tool_checkpoint_contract"] = copy.deepcopy(self.tool_checkpoint_contract)
        metadata["tool_memory"]["anchored_segments"] = len(memory.tool_gist_segments)
        metadata["tool_memory"]["prefix_tool_chunks"] = sum(
            chunk.projection_set == "tool" for chunk in memory.chunks)
        if memory.raw_tool_segments:
            metadata["tool_memory"]["raw_tool_segments"] = [
                dict(item) for item in memory.raw_tool_segments]
        eligible = getattr(base, "eligible_chunks", None)
        if eligible is not None:
            tool_prefix = memory.chunks[:len(memory.chunks) - len(base.memory.chunks)]
            anchored_chunks = tuple(chunk for segment in memory.tool_gist_segments
                                    for chunk in segment["chunks"])
            eligible = tuple(tool_prefix) + anchored_chunks + tuple(eligible)
        return ToolPrepared(base, memory, metadata, plan, eligible, ratio,
                            max_new_tokens, copy.deepcopy(dict(payload)))

    def reconsider(self, prepared: ToolPrepared, draft_tool_calls, *,
                   draft_text: str, parse_error: str | None = None):
        if not isinstance(prepared, ToolPrepared):
            raise TypeError("Expected a tool-planned prepared decision")
        value = self.inner.reconsider(
            prepared.inner, draft_tool_calls, draft_text=draft_text,
            parse_error=parse_error)
        result = dict(value)
        result["memory"] = self._augment(value["memory"], prepared.plan,
                                          prepared.source_payload,
                                          ratio=prepared.ratio,
                                          max_new_tokens=prepared.max_new_tokens)
        result["metadata"] = copy.deepcopy(value["metadata"])
        result["metadata"]["paper_whole_full_kv_tokens"] = prepared.metadata[
            "paper_whole_full_kv_tokens"]
        result["metadata"]["history_only_resident_kv_tokens"] = value["memory"].costs(
            prepared.ratio)["resident_kv_tokens"]
        result["metadata"]["history_only_gist_tokens"] = value["memory"].costs(
            prepared.ratio)["gist_tokens"]
        result["metadata"]["tool_memory"] = copy.deepcopy(prepared.metadata["tool_memory"])
        result["metadata"]["tool_memory"]["anchored_segments"] = len(
            result["memory"].tool_gist_segments)
        result["metadata"]["tool_memory"]["prefix_tool_chunks"] = sum(
            chunk.projection_set == "tool" for chunk in result["memory"].chunks)
        return result
