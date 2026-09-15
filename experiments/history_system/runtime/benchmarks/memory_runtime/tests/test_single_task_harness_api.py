"""Verify task isolation and final-action transport for ordinary harnesses."""

import copy
import json
import time
from pathlib import Path

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPIError
from benchmarks.memory_runtime.single_task_harness_api import SingleTaskHarnessAPI


class Runner:
    generation_calls = 0
    max_generation_calls = 96

    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(copy.deepcopy(payload))
        self.generation_calls += 1
        return {
            "status": "ok", "session_id": payload["session_id"],
            "decision_key": payload["decision_key"],
            "generation_trace": [
                {"native_draft": {"text": "discarded secret draft"}, "discarded": True},
                {"native_draft": {"text": "```python\nprint('final')\n```"}, "discarded": False}],
            "response": {"role": "assistant", "content": "decoded content",
                         "tool_calls": [], "finish_reason": "stop"},
            "generation_usage_total": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        }


FIXTURES = Path(__file__).resolve().parents[4] / "multibench" / "fixtures"


def api(tmp_path, benchmark="acon_appworld", tasks=("fixed-task",),
        model="new-algorithm", max_new_tokens=32):
    runner = Runner()
    transport = SingleTaskHarnessAPI(
        runner, run_id="test", model_name=model, view_mode="full_original",
        max_new_tokens=max_new_tokens, allowed_task_ids=tasks, max_decisions=96,
        deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / "steps.jsonl",
        benchmark=benchmark)
    return transport, runner


def request():
    return {"model": "new-algorithm", "max_tokens": 32,
            "messages": [{"role": "system", "content": "Execute Python."},
                         {"role": "user", "content": "Read my calendar."}]}


def test_plain_harness_gets_only_final_original_code_and_stable_identity(tmp_path):
    transport, runner = api(tmp_path)
    wire = request()
    first = transport.handle_chat(wire)
    second = transport.handle_chat(wire)
    assert first == second
    assert len(runner.calls) == 1
    assert runner.calls[0]["session_id"] == "acon_appworld/fixed-task/attempt-0"
    assert runner.calls[0]["messages"] == wire["messages"]
    final = first["choices"][0]["message"]["content"]
    assert final == "```python\nprint('final')\n```"
    assert "discarded" not in final


def test_observation_changes_decision_without_changing_frozen_task(tmp_path):
    transport, runner = api(tmp_path)
    wire = request()
    transport.handle_chat(wire)
    wire["messages"].extend([{"role": "assistant", "content": "print('one')"},
                             {"role": "user", "content": "one"}])
    transport.handle_chat(wire)
    assert len(runner.calls) == 2
    assert runner.calls[0]["session_id"] == runner.calls[1]["session_id"]
    assert runner.calls[0]["decision_key"] != runner.calls[1]["decision_key"]


def test_real_toolsandbox_history_and_tools_reach_native_runner_unchanged(tmp_path):
    transport, runner = api(
        tmp_path, benchmark="toolsandbox",
        tasks=("send_message_with_contact_content_cellular_off_multiple_user_turn",),
        model="gpt-4o-2024-05-13",
    )
    archived = json.loads((FIXTURES / "toolsandbox_conversation.json").read_text())
    # ToolSandbox adds scorer annotations to its archived tool messages after
    # the OpenAI request. The source builder sends exactly this allowed-field projection.
    fields = {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
    messages = [{key: copy.deepcopy(value) for key, value in message.items() if key in fields}
                for message in archived[:8]]
    tools = json.loads((FIXTURES / "toolsandbox_tools.json").read_text())
    transport.handle_chat({"model": "gpt-4o-2024-05-13",
                           "messages": messages, "tools": tools})
    assert runner.calls[0]["messages"] == messages
    assert runner.calls[0]["tools"] == tools
    assert runner.calls[0]["session_id"].endswith(
        "/send_message_with_contact_content_cellular_off_multiple_user_turn/attempt-0")


def test_real_appworld_wire_defaults_are_recorded_then_server_normalized(tmp_path):
    transport, runner = api(
        tmp_path, benchmark="acon_appworld", tasks=("3d9a636_1",),
        model="c2kv-agent", max_new_tokens=2048,
    )
    sessions = json.loads((FIXTURES / "appworld_llm_history.json").read_text())
    messages = sessions[0]
    wire = {
        "model": "c2kv-agent", "messages": messages, "max_tokens": 2048,
        "temperature": 0, "top_p": 1.0, "n": 1, "seed": 42,
        "presence_penalty": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    transport.handle_chat(wire)
    assert runner.calls[0]["messages"] == messages
    step = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[0])
    receipt = step["harness_transport_normalization"]
    assert receipt["client_sampling_fields"]["presence_penalty"] == 0.5
    assert receipt["client_sampling_fields"]["seed"] == 42
    assert receipt["normalized_server_fields"] == {
        "temperature": 0, "seed": 0, "max_completion_tokens": 2048}
    assert set(receipt["normalized_away_fields"]) >= {
        "presence_penalty", "seed", "chat_template_kwargs"}


def test_acebench_client_context_is_ignored_in_favour_of_server_task(tmp_path):
    transport, runner = api(tmp_path, benchmark="acebench", tasks=("agent_multi_step_0",))
    wire = request()
    wire["c2kv_eval_context"] = {
        "benchmark": "wrong", "task_id": "wrong", "user_turn": 900,
        "step": 900, "attempt": 7,
    }
    transport.handle_chat(wire)
    assert runner.calls[0]["session_id"] == "acebench/agent_multi_step_0/attempt-0"
    step = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[0])
    assert step["harness_transport_normalization"]["client_context_ignored"] is True


def test_multi_task_server_is_not_accepted_without_harness_task_binding(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        api(tmp_path, tasks=("task-a", "task-b"))


@pytest.mark.parametrize("override", [{"max_tokens": 64}, {"temperature": 1},
                                      {"stream": True}, {"gold_answer": "secret"}])
def test_unsupported_contract_cannot_reserve_a_generation(tmp_path, override):
    transport, runner = api(tmp_path)
    wire = request()
    wire.update(override)
    with pytest.raises(EventNativeAPIError):
        transport.handle_chat(wire)
    assert runner.calls == []
    assert transport.decisions_reserved == 0


def test_tau2_task_seed_is_recorded_and_normalized_for_frozen_greedy(tmp_path):
    transport, runner = api(tmp_path, benchmark="tau2", tasks=("0",), model="c2kv-agent")
    wire = {"model": "c2kv-agent", "messages": [{"role": "user", "content": "Hello"}], "seed": 42}
    transport.handle_chat(wire)
    assert len(runner.calls) == 1
    step = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[0])
    receipt = step["harness_transport_normalization"]
    assert receipt["client_sampling_fields"]["seed"] == 42
    assert receipt["normalized_server_fields"]["seed"] == 0


def test_acebench_text_execution_preserves_source_roles_and_binds_action():
    from history_memory.events import EventStore
    messages = [{"role": "system", "content": "APIs"},
                {"role": "user", "content": "Book a ride"},
                {"role": "assistant", "content": "[get_location()]"},
                {"role": "tool", "tool_call_id": "acebench-execution-2", "content": "location=A"}]
    store = EventStore.from_messages("ace", messages)
    assert [m.to_dict() for m in store.messages] == messages
    assert store.events[-1].source_indices == (2, 3)
    assert store.events[-1].kind == "tool_event"
    assert store.events[-1].complete
    assert store.events[-1].tool_call_ids == ()
    messages[-1]["tool_call_id"] = "unmatched-native-call"
    with pytest.raises(ValueError, match="Unmatched"):
        EventStore.from_messages("ace", messages)


def test_acebench_text_execution_rejects_missing_adjacent_action():
    from history_memory.events import EventStore
    with pytest.raises(ValueError, match="adjacent assistant"):
        EventStore.from_messages("ace", [{"role": "user", "content": "goal"},
            {"role": "tool", "tool_call_id": "acebench-execution-0", "content": "output"}])


@pytest.mark.parametrize("benchmark", ["acebench", "tau2"])
def test_text_execution_source_protocol_is_scoped_to_acebench(tmp_path, benchmark):
    transport, runner = api(tmp_path, benchmark=benchmark)
    wire = request()
    wire["messages"].extend([{"role": "assistant", "content": "[get_location()]"},
                             {"role": "tool", "tool_call_id": "acebench-execution-2", "content": "A"}])
    if benchmark == "acebench":
        transport.handle_chat(wire)
        assert runner.calls[0]["messages"] == wire["messages"]
    else:
        with pytest.raises(EventNativeAPIError, match="ACEBench source protocol"):
            transport.handle_chat(wire)
        assert not runner.calls
