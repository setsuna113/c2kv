"""CPU tests for deployable post-draft E1 recovery."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_recovery import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.same_event_bridge_only import SAME_EVENT_BRIDGE_ONLY_POLICY
from benchmarks.memory_runtime.tests.test_same_event_bridge_only import (
    Tokenizer,
    packing,
    policy,
    s0_config,
)


def messages():
    result = [
        {"role": "system", "content": "Use observed tool results."},
        {"role": "user", "content": "Find the requested record."},
    ]
    rows = [
        ("old", "lookup", {"id": "item-17"}, {"value": "violet"}),
        ("a", "ping", {"id": "other-a"}, {"ok": 1}),
        ("b", "ping", {"id": "other-b"}, {"ok": 2}),
        ("c", "ping", {"id": "other-c"}, {"ok": 3}),
    ]
    for call_id, name, arguments, output in rows:
        result.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(output),
                },
            ]
        )
    result.append({"role": "user", "content": "Continue."})
    return result


def draft_call():
    return {
        "id": "held-0",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"id": "item-17"})},
    }


def recovery_config(gate, **extra):
    return {"schema": E1_RECOVERY_VERSION, "gate": gate, **extra}


def controller(gate, **extra):
    config = s0_config()
    config["observed_entity_slot_policy"] = SAME_EVENT_BRIDGE_ONLY_POLICY
    config["post_draft_recovery"] = recovery_config(gate, **extra)
    return build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=config,
    )


def request(decision_key="d1"):
    return {
        "session_id": "task-1",
        "decision_key": decision_key,
        "messages": messages(),
        "tools": [],
    }


def margin_trace(value):
    return {
        "schema": "event-native-shadow-features-v1",
        "tool_name": {"status": "located"},
        "signals": {"first_name_top2_logprob_margin": value},
    }


def margin_calibration(threshold=0.5):
    return {
        "feature": "first_name_top2_logprob_margin",
        "direction": "at_or_below",
        "threshold": threshold,
        "artifact_sha256": "1" * 64,
    }


def test_disabled_is_an_exact_one_generation_memory_reference():
    base_config = s0_config()
    base_config["observed_entity_slot_policy"] = SAME_EVENT_BRIDGE_ONLY_POLICY
    base = build_event_native_controller(
        Tokenizer(), packing=packing(), policy=policy(), view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY, s0_config=base_config,
    )
    disabled = controller("disabled")
    base_prepared = base.prepare(request(), ratio=4, max_new_tokens=32)
    prepared = disabled.prepare(request(), ratio=4, max_new_tokens=32)
    assert prepared.memory == base_prepared.memory
    disabled.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.0)
    )
    result = disabled.reconsider(
        prepared, [draft_call()], draft_text="held draft", parse_error=None
    )
    assert result["regenerate"] is False
    assert result["memory"] == base_prepared.memory
    assert result["decision"]["reason"] == "recovery_disabled"


def test_margin_gate_uses_shadow_signal_and_admits_one_complete_event_under_b0():
    recovery = controller(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    assert "task-1:m2" not in prepared.memory.view.raw_event_ids
    recovery.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.1)
    )
    result = recovery.reconsider(
        prepared, [draft_call()], draft_text="held draft", parse_error=None
    )
    assert result["regenerate"] is True
    assert result["decision"]["candidate_event_id"] == "task-1:m2"
    assert result["decision"]["judges_action_correctness"] is False
    assert "task-1:m2" in result["memory"].view.raw_event_ids
    assert result["metadata"]["actual_history_bytes"] <= min(
        recovery.policy_config.history_budget_bytes,
        recovery.policy_config.workspace_budget_bytes,
    )
    assert result["decision"]["allocation"]["b0_rechecked_after_all_changes"]


def test_text_draft_recovers_complete_user_observation_without_tool_calls():
    recovery = controller(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    text_request = {
        "session_id": "text-task",
        "decision_key": "d1",
        "tools": [],
        "messages": [
            {"role": "system", "content": "Write Python code."},
            {"role": "user", "content": "Make Venmo friends match phone contacts."},
            {"role": "assistant", "content": 'phone_contact_ids = ["u17", "u18"]'},
            {
                "role": "user",
                "content": "Execution result: phone_contact_ids contains u17 and u18.",
            },
            {"role": "assistant", "content": "venmo_friend_ids = []"},
            {"role": "user", "content": "Execution result: venmo_friend_ids is empty."},
        ],
    }
    prepared = recovery.prepare(text_request, ratio=4, max_new_tokens=32)
    recovery.observe_draft_features(
        session_id="text-task", decision_key="d1", shadow_features=margin_trace(0.1)
    )
    result = recovery.reconsider(
        prepared,
        [],
        draft_text=(
            "to_add = [uid for uid in phone_contact_ids "
            "if uid not in venmo_friend_ids]"
        ),
        parse_error=None,
    )
    selected = result["decision"]["source"]["selected_event"]
    assert result["regenerate"] is True
    assert selected["kind"] == "user" and selected["source_roles"] == ["user"]
    assert selected["event_id"] == "text-task:m3"
    assert selected["event_id"] in result["memory"].view.raw_event_ids
    assert selected["event_id"] not in prepared.memory.view.raw_event_ids
    assert result["decision"]["source"]["fabricated_tool_execution"] is False
    assert result["decision"]["allocation"]["b0_rechecked_after_all_changes"]


def test_margin_requires_explicit_calibration_and_missing_signal_abstains():
    with pytest.raises(ValueError, match="explicit margin_calibration"):
        controller("first_name_margin")
    recovery = controller(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    recovery.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=None
    )
    result = recovery.reconsider(
        prepared, [draft_call()], draft_text="held draft", parse_error=None
    )
    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "shadow_features_unavailable"


def test_online_quota_counts_only_admitted_recoveries():
    recovery = controller(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    first = recovery.prepare(request("d1"), ratio=4, max_new_tokens=32)
    recovery.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.1)
    )
    assert recovery.reconsider(
        first, [draft_call()], draft_text="held draft", parse_error=None
    )["regenerate"]

    second_request = request("d2")
    second_request["messages"].extend(
        [
            {"role": "assistant", "content": "Observed the first response."},
            {"role": "user", "content": "Continue."},
        ]
    )
    second = recovery.prepare(second_request, ratio=4, max_new_tokens=32)
    recovery.observe_draft_features(
        session_id="task-1", decision_key="d2", shadow_features=margin_trace(0.1)
    )
    result = recovery.reconsider(
        second, [draft_call()], draft_text="held draft", parse_error=None
    )
    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "online_recovery_quota_exhausted"
    assert result["decision"]["quota"]["recovery_limit_at_decision"] == 1


def test_seeded_random_gate_is_online_and_reproducible():
    chosen = None
    for index in range(100):
        key = f"random-{index}"
        payload = json.dumps(
            [E1_RECOVERY_VERSION, 0, "task-1", key, "gate"],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        if int(hashlib.sha256(payload.encode()).hexdigest(), 16) * 5 < 1 << 256:
            chosen = key
            break
    assert chosen is not None
    first = controller("seeded_random")
    second = controller("seeded_random")
    results = []
    for recovery in (first, second):
        prepared = recovery.prepare(request(chosen), ratio=4, max_new_tokens=32)
        result = recovery.reconsider(
            prepared, [draft_call()], draft_text="held draft", parse_error=None
        )
        results.append(result)
    assert all(result["regenerate"] for result in results)
    assert results[0]["decision"]["gate"] == results[1]["decision"]["gate"]
    assert results[0]["decision"]["gate"]["online_only"] is True


def prefill_head():
    return {
        "schema": "event-native-prefill-linear-head-v1",
        "feature": "prefill.prompt_last.decoder_layer_output",
        "layer": 2,
        "weights": [1.0, -1.0],
        "bias": 0.0,
        "input_mean": [0.0, 0.0],
        "input_scale": [1.0, 1.0],
        "score_transform": "sigmoid",
        "direction": "at_or_above",
        "threshold": 0.8,
        "artifact_sha256": "2" * 64,
    }


def test_prefill_linear_head_requires_export_and_validates_feature_binding():
    with pytest.raises(ValueError, match="explicit exported prefill_head"):
        controller("prefill_linear_head")
    recovery = controller("prefill_linear_head", prefill_head=prefill_head())
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    recovery.observe_draft_features(
        session_id="task-1",
        decision_key="d1",
        shadow_features={
            "schema": "event-native-shadow-features-v1",
            "prefill": {
                "status": "captured",
                "reason": None,
                "layer": 2,
                "position": {"kind": "prompt_last", "logical_position": 100},
                "readout": "decoder_layer_output",
                "stored_dtype": "float16",
                "hidden": [3.0, 0.0],
            },
        },
    )
    result = recovery.reconsider(
        prepared, [draft_call()], draft_text="held draft", parse_error=None
    )
    assert result["regenerate"] is True
    assert result["decision"]["gate"]["score"] > 0.9
    assert "weights" not in result["metadata"]["post_draft_recovery_config"]["prefill_head"]


def test_selected_d3_controller_config_loads_through_public_entry_point():
    runtime_root = Path(__file__).resolve().parents[3]
    selected = json.loads(
        (runtime_root / "configs" / "controller.json").read_text(encoding="utf-8")
    )
    recovery = build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=selected,
    )
    head = recovery.config["prefill_head"]
    assert recovery.config["gate"] == "prefill_linear_head"
    assert "source_admission_policy" not in recovery.config
    assert len(head["weights"]) == len(head["input_mean"]) == len(head["input_scale"])


def tool_text(value):
    return "<tool_call>" + json.dumps(
        {"name": "lookup", "arguments": {"id": value}}
    ) + "</tool_call>"


class Generator:
    def __init__(self, journal_path):
        self.journal_path = journal_path
        self.calls = 0

    def decision_scope(self, *, session_id=None):
        return nullcontext()

    def session_cache_info(self):
        return {"scope": "cpu-test"}

    def close_session(self):
        pass

    def generate(self, memory, **kwargs):
        self.calls += 1
        text = tool_text("item-17") if self.calls == 1 else "Done."
        token_ids = tuple(map(ord, text))
        stats = {"shadow_features": margin_trace(0.1)} if self.calls == 1 else {}
        return SimpleNamespace(
            token_ids=token_ids,
            token_logprobs=(0.0,) * len(token_ids),
            finish_reason="stop",
            stats=stats,
        )


def test_runner_wires_draft_shadow_features_before_one_regeneration(tmp_path):
    recovery = controller(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    journal_path = tmp_path / "attempts.jsonl"
    generator = Generator(journal_path)
    runner = EventNativeDecisionRunner(
        recovery,
        generator,
        Tokenizer(),
        ratio=4,
        max_new_tokens=32,
        max_generation_calls=2,
        journal=AttemptJournal(journal_path),
    )
    result = runner.run(request())
    assert result["generation_attempts"] == generator.calls == 2
    assert result["generation_trace"][0]["discarded"] is True
    assert result["exact_recovery"]["gate"]["value"] == 0.1
    assert result["response"]["content"] == "Done."
