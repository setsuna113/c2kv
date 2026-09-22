"""CPU lifecycle contracts for event-aware native-tool selection."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

from history_memory.packing import MemoryView, PackedMemory
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.event_native_tool import (
    ToolRegionController,
    parse_native_tool_spec,
)
from benchmarks.memory_runtime.tests.test_event_native_tool import (
    CharacterTokenizer,
    HistoryController,
    RecordingGenerator,
)


def _tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": {},
    }}


TOOLS = [
    _tool("account_profile", "alpha"),
    _tool("read_inventory", "bravo"),
    _tool("compare_quotes", "charlie"),
    _tool("archive_record", "delta"),
    _tool("cancel_order", "echo"),
    _tool("send_alert", "foxtrot"),
]


def _call(name: str, call_id: str) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"}}


def _tool_draft(name: str) -> str:
    return '<tool_call>{"name":"' + name + '","arguments":{}}</tool_call>'


class LifecycleTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, *, tools=None, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        parts = []
        if tools:
            parts.append("<tools>" + json.dumps(
                tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "</tools>")
        for message in messages:
            parts.append(f"<{message['role']}>")
            if message.get("content") is not None:
                parts.append(str(message["content"]))
            if message.get("tool_calls"):
                parts.append(json.dumps(message["tool_calls"], ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")))
            if message.get("tool_call_id"):
                parts.append(f"[{message['tool_call_id']}]")
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        return list(map(ord, rendered)) if tokenize else rendered


class MultiMessageHistoryController(HistoryController):
    def prepare(self, payload, *, ratio, max_new_tokens):
        self.seen.append(payload)
        messages = list(payload["messages"])
        if messages[0].get("role") == "system":
            system_messages, workspace_messages = messages[:1], messages[1:]
        else:
            system_messages = [{"role": "system", "content": ""}]
            workspace_messages = messages
        system_ids = tuple(self.tokenizer.apply_chat_template(system_messages, tokenize=True))
        workspace_ids = tuple(self.tokenizer.apply_chat_template(workspace_messages, tokenize=True))
        memory = PackedMemory(MemoryView((), ()), system_ids, workspace_ids,
                              tuple(range(len(messages))), ())
        metadata = {"decision_index": len(self.seen),
                    "common_raw_prompt_tokens": len(system_ids) + len(workspace_ids),
                    "actual_history_bytes": 0}
        return SimpleNamespace(memory=memory, metadata=metadata, eligible_chunks=())


class ScriptedGenerator(RecordingGenerator):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)

    def generate(self, memory, **kwargs):
        self.memories.append(memory)
        self.requests.append(kwargs)
        text = self.outputs.pop(0)
        return SimpleNamespace(token_ids=tuple(map(ord, text)), finish_reason="stop",
                               token_logprobs=tuple(0.0 for _ in text),
                               stats={"eos_token_ids": ()})


def _payload() -> dict:
    return {
        "session_id": "selector-lifecycle", "decision_key": "turn-0/step-1",
        "messages": [
            {"role": "system", "content": "Use the available APIs."},
            {"role": "user", "content": "account_profile"},
            {"role": "assistant", "content": None,
             "tool_calls": [_call("read_inventory", "actual-1")]},
            {"role": "tool", "tool_call_id": "actual-1",
             "content": "compare_quotes"},
        ],
        "tools": copy.deepcopy(TOOLS),
    }


def _controller(generator):
    tokenizer = LifecycleTokenizer()
    inner = MultiMessageHistoryController(tokenizer)
    controller = ToolRegionController(
        inner, tokenizer,
        parse_native_tool_spec("t0:r8:hybrid3:schema:selector=latest_event_topk_v1"),
        model_context=100_000, generator=generator)
    return tokenizer, controller


def _selection_plan_hash(plan) -> str:
    keys = (
        "selector_policy", "selector_version", "selector_scores", "selector_rank",
        "selector_query_sha256", "selector_latest_io_present",
        "score_selected_native_indices", "native_indices", "top_k",
    )
    material = {"protocol": plan.protocol,
                "selector": {key: plan.info[key] for key in keys}}
    return hashlib.sha256(json.dumps(
        material, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def test_latest_event_top3_changes_only_after_complete_observation():
    _, controller = _controller(RecordingGenerator())
    initial = _payload()
    first = controller.prepare(initial, ratio=8, max_new_tokens=32)
    first_hash = _selection_plan_hash(first.plan)
    assert first.plan.info["native_indices"] == [0, 1, 2]

    incomplete_payload = copy.deepcopy(initial)
    incomplete_payload["decision_key"] = "turn-0/step-2"
    incomplete_payload["messages"].append({
        "role": "assistant", "content": None,
        "tool_calls": [_call("send_alert", "actual-2")],
    })
    incomplete = controller.prepare(incomplete_payload, ratio=8, max_new_tokens=32)
    assert _selection_plan_hash(incomplete.plan) == first_hash

    complete_payload = copy.deepcopy(incomplete_payload)
    complete_payload["decision_key"] = "turn-0/step-3"
    complete_payload["messages"].append({
        "role": "tool", "tool_call_id": "actual-2", "content": "archive_record",
    })
    complete = controller.prepare(complete_payload, ratio=8, max_new_tokens=32)
    assert complete.plan.info["native_indices"] == [0, 3, 5]
    assert _selection_plan_hash(complete.plan) != first_hash


def test_decision_runner_freezes_selector_plan_and_memory_through_recovery(tmp_path):
    generator = ScriptedGenerator([
        _tool_draft("cancel_order"), _tool_draft("send_alert"),
    ])
    tokenizer, controller = _controller(generator)
    plan_hashes = []
    original_plan = controller._plan

    def observed_plan(payload):
        plan = original_plan(payload)
        plan_hashes.append(_selection_plan_hash(plan))
        return plan

    controller._plan = observed_plan
    runner = EventNativeDecisionRunner(
        controller, generator, tokenizer, ratio=8, max_new_tokens=32,
        max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(_payload())
    assert len(plan_hashes) == 1
    assert [trace["phase"] for trace in record["generation_trace"]] == [
        "draft", "regeneration"]
    receipts = [trace["controller"]["tool_memory"]
                for trace in record["generation_trace"]]
    assert [receipt["native_indices"] for receipt in receipts] == [[0, 1, 2], [0, 1, 2]]
    assert receipts[0]["selector_query_sha256"] == receipts[1]["selector_query_sha256"]
    assert record["generation_trace"][0]["prepared_input"] == record[
        "generation_trace"][1]["prepared_input"]
    assert generator.memories[0] == generator.memories[1]
