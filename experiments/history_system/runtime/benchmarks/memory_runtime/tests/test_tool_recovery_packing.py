"""CPU contracts for native C2KV tool-plan re-rendering."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT.parents[2] / "benchmarks")]

from toolmemory import parse_tool_memory_spec, plan_visible_tool_memory
from history_memory.events import EventStore, RenderedMessages
from history_memory.packing import native_ids, pack_memory
from history_memory.source_packing import make_source_view, pack_source_memory
from benchmarks.memory_runtime.event_native_raw import RuntimeMemoryView, _pack_raw_control
from benchmarks.memory_runtime.tool_recovery_packing import (
    ToolRecoveryContextExceeded,
    rebuild_tool_history_memory,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False,
                            tokenize=True, **kwargs):
        rendered = "".join(
            f"<{message['role']}>" + json.dumps(message, sort_keys=True)
            for message in messages
        )
        if tools:
            rendered = "<native-tools>" + json.dumps(tools) + rendered
        if add_generation_prompt:
            rendered += "<assistant>"
        return tuple(map(ord, rendered)) if tokenize else rendered

    def native_ids(self, messages, *, tools=None, generation=False):
        return native_ids(self, messages, tools=tools, generation=generation)

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def _call():
    return {"id": "call-1", "type": "function", "function": {
        "name": "displayCarStatus", "arguments": '{"option":"fuelLevel"}',
    }}


def _payload():
    return {
        "session_id": "tool-rebuild",
        "messages": [
            {"role": "system", "content": "Use the visible tools."},
            {"role": "user", "content": "Earlier car question"},
            {"role": "assistant", "content": "Checking", "tool_calls": [_call()]},
            {"role": "tool", "tool_call_id": "call-1", "name": "displayCarStatus",
             "content": "invalid option"},
            {"role": "user", "content": "Check the current car status"},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "displayCarStatus",
                "description": "Return the complete status; option must be one of speed or fuel.",
                "parameters": {"type": "object", "properties": {
                    "option": {"type": "string", "description": "Use speed or fuel, never fuelLevel."},
                }, "required": ["option"]},
            }},
            {"type": "function", "function": {
                "name": "estimate_distance",
                "description": "Estimate distance between two city zipcodes.",
                "parameters": {"type": "object", "properties": {
                    "cityA": {"type": "string", "description": "Origin zipcode"},
                    "cityB": {"type": "string", "description": "Destination zipcode"},
                }, "required": ["cityA", "cityB"]},
            }},
        ],
    }


def _plans(payload, tokenizer):
    spec = parse_tool_memory_spec("t0:r8:uniform:schema")
    old = plan_visible_tool_memory(payload, spec, tokenizer)
    new = plan_visible_tool_memory(payload, spec, tokenizer, native_override=[0, 1])
    assert old is not None and new is not None
    return old, new


def _store(payload, plan):
    return EventStore.from_messages(
        payload["session_id"], RenderedMessages(plan.messages, source=payload["messages"]),
    )


class ToolRecoveryPackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.payload = _payload()
        self.old, self.new = _plans(self.payload, self.tokenizer)
        self.store = _store(self.payload, self.old)
        self.derived = [{"role": "user", "content": "Observed task reminder"}]

    def rebuild(self, memory, metadata=None, new_messages=None, **kwargs):
        return rebuild_tool_history_memory(
            self.tokenizer, memory, metadata or {}, self.payload,
            self.old.messages, self.new.messages if new_messages is None else new_messages,
            ratio=8, max_new_tokens=32, model_context=kwargs.get("model_context", 100000),
        )

    def test_all_raw_tool_protocol_changes_system_and_preserves_event_history(self):
        events = self.store.events
        view = RuntimeMemoryView(
            gist_event_ids=(events[1].event_id, events[2].event_id),
            raw_event_ids=(events[0].event_id, events[3].event_id),
            omitted_event_ids=(),
            mandatory_raw_event_ids=(events[0].event_id, events[3].event_id),
            raw_control_layout="event-native-always-compress-gist-v1",
        )
        selected = pack_memory(self.store, view, self.tokenizer,
                               derived_workspace_prefix_messages=self.derived)
        rebuilt = self.rebuild(selected, {"derived_workspace_prefix_messages": self.derived})
        old_system = self.tokenizer.decode(selected.system_input_ids)
        new_system = self.tokenizer.decode(rebuilt.system_input_ids)
        self.assertNotIn("never fuelLevel", old_system)
        self.assertIn("never fuelLevel", new_system)
        self.assertIn("Destination zipcode", new_system)
        self.assertEqual(rebuilt.view, selected.view)
        self.assertIs(rebuilt.chunks, selected.chunks)
        self.assertEqual(rebuilt.raw_source_indices, selected.raw_source_indices)
        self.assertEqual(rebuilt.workspace_input_ids, selected.workspace_input_ids)
        self.assertIn("Observed task reminder", self.tokenizer.decode(rebuilt.workspace_input_ids))
        self.assertEqual(self.payload["messages"][0]["content"], "Use the visible tools.")

    def test_source_partition_and_full_original_are_supported(self):
        source_view = make_source_view(self.store, (0, 3, 4), (1, 2), (0, 4))
        source_memory = pack_source_memory(self.store, source_view, self.tokenizer)
        source_rebuilt = self.rebuild(source_memory)
        self.assertEqual(source_rebuilt.view, source_view)
        self.assertIs(source_rebuilt.chunks, source_memory.chunks)
        self.assertEqual(source_rebuilt.workspace_input_ids, source_memory.workspace_input_ids)
        self.assertIn("never fuelLevel", self.tokenizer.decode(source_rebuilt.system_input_ids))

        all_event_ids = tuple(event.event_id for event in self.store.events)
        full_view = RuntimeMemoryView(
            gist_event_ids=(), raw_event_ids=all_event_ids,
            mandatory_raw_event_ids=all_event_ids,
            raw_control_layout="full-original-native-v1",
        )
        full_memory = pack_memory(self.store, full_view, self.tokenizer)
        full_rebuilt = self.rebuild(full_memory)
        self.assertEqual(full_rebuilt.view, full_view)
        self.assertEqual(full_rebuilt.chunks, ())
        self.assertEqual(full_rebuilt.workspace_input_ids, full_memory.workspace_input_ids)
        self.assertIn("Destination zipcode", self.tokenizer.decode(full_rebuilt.system_input_ids))

    def test_projected_workspace_and_derived_message_are_preserved(self):
        events = self.store.events
        view = RuntimeMemoryView(
            gist_event_ids=(events[1].event_id,),
            raw_event_ids=(events[0].event_id, events[2].event_id, events[3].event_id),
            mandatory_raw_event_ids=(events[0].event_id, events[3].event_id),
            raw_control_layout="event-native-always-compress-gist-v1",
        )
        projected = self.store.messages[2].to_dict()
        projected["content"] = None
        selected = pack_memory(
            self.store, view, self.tokenizer,
            raw_source_message_overrides={2: projected},
            derived_workspace_prefix_messages=self.derived,
        )
        receipt = {"applied": True, "narrative_projections": [{
            "event_id": events[2].event_id,
            "source_index": 2,
            "removed_content_sha256": hashlib.sha256(b"Checking").hexdigest(),
        }], "argument_projections": [{"source_index": 2, "field": "option"}]}
        rebuilt = self.rebuild(selected, {
            "capacity_fallback": receipt,
            "derived_workspace_prefix_messages": self.derived,
        })
        workspace = self.tokenizer.decode(rebuilt.workspace_input_ids)
        self.assertNotIn("Checking", workspace)
        self.assertIn("displayCarStatus", workspace)
        self.assertIn("Observed task reminder", workspace)
        self.assertIs(rebuilt.workspace_input_ids, selected.workspace_input_ids)
        self.assertIs(rebuilt.chunks, selected.chunks)

    def test_context_rejection_and_changed_source_are_explicit(self):
        all_event_ids = tuple(event.event_id for event in self.store.events)
        view = RuntimeMemoryView(
            gist_event_ids=(), raw_event_ids=all_event_ids,
            raw_control_layout="full-original-native-v1",
        )
        selected = pack_memory(self.store, view, self.tokenizer)
        with self.assertRaises(ToolRecoveryContextExceeded):
            self.rebuild(selected, model_context=1)
        changed = copy.deepcopy(self.new.messages)
        changed[-1]["content"] = "A different request"
        with self.assertRaisesRegex(ValueError, "non-system source message"):
            self.rebuild(selected, new_messages=changed)
        with self.assertRaisesRegex(ValueError, "original tool prefix"):
            self.rebuild(replace(selected, system_input_ids=(1,)))

    def test_full_shared_evidence_workspace_is_preserved(self):
        all_event_ids = tuple(event.event_id for event in self.store.events)
        view = RuntimeMemoryView(
            gist_event_ids=(), raw_event_ids=all_event_ids,
            evidence_event_ids=(self.store.events[2].event_id,),
            raw_control_layout="full-shared-duplicate-evidence-v1",
        )
        selected = _pack_raw_control(self.store, view, self.tokenizer, None)
        rebuilt = self.rebuild(selected)
        self.assertIs(rebuilt.workspace_input_ids, selected.workspace_input_ids)
        self.assertIn("history-evidence-v1", self.tokenizer.decode(rebuilt.workspace_input_ids))


if __name__ == "__main__":
    unittest.main()
