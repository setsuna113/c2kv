"""Request source protection without changing native input or recovery policy.

Selection only names existing historical messages. The serving backend decides
whether their original KV can be retained within its native mandatory quotas.
This request never supplies text to re-prefill an evicted source.
"""
from __future__ import annotations

from dataclasses import replace

from .allocator import PersistentHistoryAllocator


class NativeProtectionAllocator(PersistentHistoryAllocator):
    """Keep the v1 view and add an optional, backend-admitted source request."""

    def _prepare_view(self, store, tools, *, ratio, max_new_tokens, decision_key, decision_index):
        prepared = super()._prepare_view(
            store, tools, ratio=ratio, max_new_tokens=max_new_tokens,
            decision_key=decision_key, decision_index=decision_index)
        memory = prepared.memory
        historical = set(range(memory.history_start_message_count, memory.history_message_count))
        users = [event for event in store.events if event.kind == "user"]
        complete_tools = [event for event in store.events
                          if event.kind == "tool_event" and event.complete]
        requested = {event.event_id for event in store.events
                     if not event.complete or event.kind == "instruction"}
        if users:
            requested.add(users[-1].event_id)
        if complete_tools:
            requested.add(complete_tools[-1].event_id)
        if self.benchmark == "acon_appworld" and users:
            requested.add(users[0].event_id)
        lexical, _, lexical_receipt = self._lexical_request(
            store, tools, requested, recent_tool_event_visible=True)
        requested.update(lexical.source_ids)
        events = tuple(event for event in store.events
                       if event.event_id in requested and historical.intersection(event.source_indices))
        indices = tuple(sorted({index for event in events for index in event.source_indices
                                if index in historical and store.messages[index].role != "system"}))
        prepared.memory = replace(memory, protection_source_indices=indices,
                                  protection_event_ids=tuple(event.event_id for event in events))
        prepared.metadata["native_protection_request"] = {
            "schema": "racer-native-protection-v1", "enabled": True,
            "event_ids": list(prepared.memory.protection_event_ids),
            "source_indices": list(indices), "lexical": lexical_receipt,
            "status": "requires_engine_admission", "input_rewritten": False,
            "original_history_budget_tokens": memory.history_budget_tokens,
        }
        return prepared
