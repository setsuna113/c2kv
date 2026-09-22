"""Bind one frozen tool plan to persistent normal-chat history requests."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from history_memory.packing import native_ids

from ..event_native_tool import shared_tool_catalog


BINDING_SCHEMA = "racer-tool-plan-binding-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class _TokenizerView:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def native_ids(self, messages, **kwargs):
        return native_ids(self.tokenizer, messages, **kwargs)


def _chunk_identity(chunk: Any) -> dict[str, Any]:
    return {
        "event_id": chunk.event_id,
        "part_index": chunk.part_index,
        "source_token_start": chunk.source_token_start,
        "source_token_end": chunk.source_token_end,
        "catalog_index": chunk.catalog_index,
        "token_ids_sha256": _sha256(list(chunk.token_ids)),
    }


def _frozen_plan_identity(plan: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(list(plan.messages))
    tools = copy.deepcopy(list(payload.get("tools") or ()))
    selector = {
        key: copy.deepcopy(plan.info[key])
        for key in (
            "selector_policy", "selector_version", "selector_scores",
            "selector_rank", "selector_query_sha256", "selector_latest_io_present",
            "score_selected_native_indices", "interface_forced_native_indices",
            "native_indices", "top_k",
        )
        if key in plan.info
    }
    material = {
        "spec": plan.spec.as_dict(),
        "messages_sha256": _sha256(messages),
        "tools_sha256": _sha256(tools),
        "protocol_sha256": hashlib.sha256(str(plan.protocol).encode("utf-8")).hexdigest(),
        "chunks": [_chunk_identity(chunk) for chunk in plan.chunks],
        "selector": selector,
        "carrier_anchors": copy.deepcopy(list(plan.carrier_anchors or ())),
    }
    return {
        **material,
        "binding_id": hashlib.sha256(_canonical(material)).hexdigest(),
    }


def _source_index_map(source: Sequence[Mapping[str, Any]],
                      assembled: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Map an exact source subsequence into a request with internal notes."""
    result: dict[int, int] = {}
    cursor = 0
    for source_index, message in enumerate(source):
        target = _canonical(message)
        while cursor < len(assembled) and _canonical(assembled[cursor]) != target:
            cursor += 1
        if cursor >= len(assembled):
            raise ValueError("Persistent request moved or changed a frozen tool source message")
        result[source_index] = cursor
        cursor += 1
    return result


def _shifted_index_map(length: int, insertions: Sequence[int]) -> dict[int, int]:
    return {
        index: index + sum(int(insert_at) <= index for insert_at in insertions)
        for index in range(length)
    }


def _shift_boundary(boundary: int, insertions: Sequence[int]) -> int:
    return boundary + sum(int(insert_at) <= boundary for insert_at in insertions)


class PersistentToolBinder:
    """Own extracted tool receipts while memories carry JSON-safe identities."""

    def __init__(self, backend: Any, tokenizer: Any, *,
                 checkpoint: str | Path | None = None,
                 budget_tokens: int | None = None):
        if not callable(getattr(backend, "extract_tokens", None)):
            raise TypeError("tool backend must expose extract_tokens(token_ids, ratio, projection_set)")
        self.backend = backend
        self.tokenizer = tokenizer
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.budget_tokens = budget_tokens
        self._managers: dict[str, Any] = {}
        self._plans: dict[str, tuple[Any, Any]] = {}

    def _manager(self, spec: Any):
        key = spec.name
        manager = self._managers.get(key)
        if manager is not None:
            return manager
        if spec.encoder == "t0" and self.checkpoint is None:
            raise ValueError("T0 persistent tool binding requires a tool checkpoint")
        catalog = shared_tool_catalog()
        manager = catalog.ToolMemory(
            spec,
            self.checkpoint or Path("."),
            self.backend.extract_tokens,
            tokenizer=_TokenizerView(self.tokenizer),
            budget_tokens=self.budget_tokens,
        )
        self._managers[key] = manager
        return manager

    def bind_tool_plan(self, memory: Any, plan: Any,
                       payload: Mapping[str, Any]):
        """Attach one frozen plan; repeated recovery binding performs no reselection."""
        if plan is None:
            return memory
        for field in ("source_messages", "source_tools", "tool_plan"):
            if not hasattr(memory, field):
                raise TypeError(f"Persistent tool binding needs memory.{field}")
        identity = _frozen_plan_identity(plan, payload)
        binding_id = identity["binding_id"]
        bound = self._plans.get(binding_id)
        if bound is None:
            manager = self._manager(plan.spec)
            actual = manager.materialize_visible_plan(payload, copy.deepcopy(plan))
            self._plans[binding_id] = (manager, actual)
        else:
            manager, actual = bound
            if actual.spec != plan.spec:
                raise ValueError("Frozen tool binding collided across specs")
        selector = {
            key: copy.deepcopy(actual.info[key])
            for key in (
                "selector_policy", "selector_version", "selector_query_sha256",
                "selector_latest_io_present", "score_selected_native_indices",
                "interface_forced_native_indices", "native_indices", "top_k",
            )
            if key in actual.info
        }
        costs = {
            key: copy.deepcopy(actual.info[key])
            for key in (
                "resident_tool_tokens", "expected_gist_tokens", "gist_tokens",
                "protocol_prefix_tokens", "native_source_tokens",
                "presented_encoder_tokens", "budget_tokens",
                "checkpoint", "checkpoint_config_sha256",
            )
            if key in actual.info
        }
        receipt = {
            "schema": BINDING_SCHEMA,
            "binding_id": binding_id,
            "spec": actual.spec.as_dict(),
            "source_messages_sha256": identity["messages_sha256"],
            "source_tools_sha256": identity["tools_sha256"],
            "protocol_sha256": identity["protocol_sha256"],
            "selector": selector,
            "costs": costs,
        }
        return replace(
            memory,
            source_messages=tuple(copy.deepcopy(list(plan.messages))),
            source_tools=tuple(copy.deepcopy(list(payload.get("tools") or ()))),
            tool_plan=receipt,
        )

    def resolve_plan(self, memory: Any) -> Any:
        receipt = getattr(memory, "tool_plan", None)
        if not isinstance(receipt, Mapping) or receipt.get("schema") != BINDING_SCHEMA:
            raise ValueError("Persistent memory has no valid frozen tool binding")
        bound = self._plans.get(receipt.get("binding_id"))
        if bound is None:
            raise ValueError("Persistent tool binding is not owned by this generator")
        return bound[1]

    def refresh(self, memory: Any) -> Any:
        """Refresh T0 extraction receipts without changing the selected plan."""
        receipt = getattr(memory, "tool_plan", None)
        plan = self.resolve_plan(memory)
        manager, _ = self._plans[receipt["binding_id"]]
        catalog = shared_tool_catalog()
        if isinstance(plan, catalog.ToolMemoryPlan):
            manager.refresh(plan)
        return plan

    def stage_request(
        self,
        memory: Any,
        payload: Mapping[str, Any],
        *,
        history_message_count: int,
        history_start_message_count: int,
        source_message_indices: Sequence[int] | None = None,
    ) -> tuple[dict[str, Any], int, int, dict[int, int]]:
        """Insert frozen carriers after history/recovery assembly and remap boundaries."""
        if (type(history_message_count) is not int
                or type(history_start_message_count) is not int
                or not 0 <= history_start_message_count <= history_message_count):
            raise ValueError("Persistent history message boundaries are invalid")
        receipt = getattr(memory, "tool_plan", None)
        plan = self.resolve_plan(memory)
        manager, _ = self._plans[receipt["binding_id"]]
        messages = copy.deepcopy(list(payload.get("messages") or ()))
        source = list(getattr(memory, "source_messages", ()))
        if _sha256(source) != receipt["source_messages_sha256"]:
            raise ValueError("Persistent tool source messages differ from their binding receipt")
        if source_message_indices is None:
            source_map = _source_index_map(source, messages)
        else:
            indices = [int(index) for index in source_message_indices]
            if (len(indices) != len(source) or indices != sorted(set(indices))
                    or any(not 0 <= index < len(messages) for index in indices)):
                raise ValueError("Persistent source message indices are incomplete or invalid")
            source_map = dict(enumerate(indices))
            protocol = str(plan.protocol or "")
            if protocol:
                locations = [
                    index for index, message in enumerate(plan.messages)
                    if isinstance(message.get("content"), str)
                    and protocol in message["content"]
                ]
                if len(locations) != 1:
                    raise ValueError("Frozen tool protocol has no unique source message")
                target = messages[source_map[locations[0]]].get("content")
                if not isinstance(target, str) or target.count(protocol) != 1:
                    raise ValueError("Persistent request changed the frozen tool protocol")
        tools = copy.deepcopy(list(getattr(memory, "source_tools", ())))
        if _sha256(tools) != receipt["source_tools_sha256"]:
            raise ValueError("Persistent tool source catalog differs from its binding receipt")

        catalog = shared_tool_catalog()
        insertions: list[int] = []
        staged_plan = plan
        if isinstance(plan, catalog.ToolMemoryPlan):
            anchors = {
                anchor["message_index"]: source_map[anchor["rewritten_message_index"]]
                for anchor in (plan.carrier_anchors or ())
            }
            messages, counts = catalog.insert_carriers(
                messages, {}, plan.carriers(), source_out_indices=anchors or None)
            insertions = [int(value) for value in counts.get("tool_memory_insertions") or ()]
            messages = [
                catalog.strip_carrier_fields(message) if catalog.is_carrier(message) else message
                for message in messages
            ]
        else:
            staged_plan = copy.deepcopy(plan)
            for name in ("raw_schema_spans", "assembled_schema_spans", "interface_spans"):
                spans = getattr(staged_plan, name, None)
                if spans is None:
                    continue
                setattr(staged_plan, name, tuple({
                    **copy.deepcopy(span),
                    "message_index": source_map[int(span["message_index"])],
                } for span in spans))
            protocol_span = staged_plan.info.get("tool_protocol_span")
            if isinstance(protocol_span, Mapping):
                staged_plan.info["tool_protocol_span"] = {
                    **copy.deepcopy(dict(protocol_span)),
                    "message_index": source_map[int(protocol_span["message_index"])],
                }

        staged = dict(payload)
        staged["messages"] = messages
        staged["tools"] = tools
        staged = manager.stage_request(staged, staged_plan)
        index_map = _shifted_index_map(len(payload.get("messages") or ()), insertions)
        return (
            staged,
            _shift_boundary(history_message_count, insertions),
            _shift_boundary(history_start_message_count, insertions),
            index_map,
        )
