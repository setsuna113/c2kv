"""CPU behavior checks for integrated G--P recovery and source lifetimes."""
from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.recovery.experiment_config import configure_controller, parse_gp_config
from benchmarks.memory_runtime.tests.test_event_native_recovery import (
    messages, request, draft_call, recovery_config, margin_calibration, margin_trace,
)
from benchmarks.memory_runtime.tests.test_event_native_step import Generator
from benchmarks.memory_runtime.tests.test_same_event_bridge_only import Tokenizer, packing, policy, s0_config


class UnitTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def make_controller(**switches):
    config = s0_config()
    config["post_draft_recovery"] = recovery_config(
        "first_name_margin", margin_calibration=margin_calibration())
    return build_event_native_controller(
        UnitTokenizer(), packing=packing(), policy=policy(),
        view_mode="ac_native_s0_lexical_raw_reserve_failed_operation",
        compression_policy="always-compress-v1",
        s0_config=configure_controller(config, {"D": "candidate_rule", **switches}))


def recover(controller, payload=None):
    prepared = controller.prepare(payload or request(), ratio=4, max_new_tokens=32)
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
    return prepared, result


def test_append_keeps_existing_gist_and_raw_and_binds_source_receipts():
    controller = make_controller(U="field", K=2)
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    original = prepared.memory
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
    assert result["regenerate"]
    assert result["memory"].chunks == original.chunks
    assert result["memory"].view.raw_event_ids == original.view.raw_event_ids
    assert result["memory"].view.gist_event_ids == original.view.gist_event_ids
    assert result["metadata"]["actual_history_bytes"] > prepared._base_prepared.metadata["actual_history_bytes"]
    assert all("tool_calls" not in message for message in result["metadata"]["derived_workspace_prefix_messages"])


@pytest.mark.parametrize("scope", ["event", "record", "adjacent_pair"])
def test_g_encoding_combines_with_field_append_and_protection(scope):
    controller = make_controller(G=scope, U="field", K=2, L="two_decisions", P="structured")
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    chunks = prepared.memory.chunks
    assert controller._packer.encoding_scope == scope
    assert all(chunk.source_token_start == 0 for chunk in prepared.eligible_chunks)
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
    assert result["regenerate"]
    assert result["memory"].chunks == chunks
    next_decision = controller.prepare(request("d2"), ratio=4, max_new_tokens=32)
    assert next_decision.metadata["gp_lifecycle"]["protected_unit_ids"]
    assert next_decision.metadata["actual_history_bytes"] <= controller.policy_config.history_budget_bytes


@pytest.mark.parametrize("rule,protected_until", [("next_decision", 1), ("two_decisions", 3), ("task", 10)])
def test_static_lifetime_and_repeated_recovery(rule, protected_until):
    controller = make_controller(L=rule)
    first, result = recover(controller)
    assert result["regenerate"]
    selected = result["decision"]["appended_unit_ids"]
    for index in range(2, 5):
        payload = request(f"d{index}")
        prepared = controller.prepare(payload, ratio=4, max_new_tokens=32)
        protected = prepared.metadata["gp_lifecycle"]["protected_unit_ids"]
        assert bool(set(selected) & set(protected)) == (index <= protected_until)
    if rule != "task":
        again = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
        assert again["regenerate"]
        assert again["decision"]["selected_unit_ids"] == result["decision"]["selected_unit_ids"]
        assert len(controller._leases["task-1"]) == len(selected)


def test_user_turn_lifetime_uses_explicit_adapter_turn_even_with_feedback_users():
    controller = make_controller(L="user_turn")
    payload = request("turn-0/step-0")
    _, first = recover(controller, payload)
    selected = first["decision"]["appended_unit_ids"]
    payload["decision_key"] = "turn-0/step-1"
    payload["messages"].append({"role": "user", "content": "Observed execution feedback."})
    same = controller.prepare(payload, ratio=4, max_new_tokens=32)
    assert same.metadata["gp_lifecycle"]["protected_unit_ids"] == selected
    payload["decision_key"] = "turn-1/step-0"
    payload["messages"].append({"role": "user", "content": "New request."})
    next_turn = controller.prepare(payload, ratio=4, max_new_tokens=32)
    assert next_turn.metadata["gp_lifecycle"]["protected_unit_ids"] == []


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_runner_checks_every_draft_and_returns_only_last_action(tmp_path, rounds):
    controller = make_controller(U="field", R=rounds)
    journal = tmp_path / "attempts.jsonl"
    # Include multiple historical sources in every held draft, including text stops.
    draft = "lookup ping violet item-17 other-a other-b other-c value ok"
    generator = Generator(journal, [draft] * rounds + ["Final response."])
    runner = EventNativeDecisionRunner(
        controller, generator, UnitTokenizer(), ratio=4, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(journal))
    record = runner.run(request())
    assert record["generation_attempts"] == rounds + 1
    assert len(record["recovery_rounds"]) == rounds
    assert record["response"]["content"] == "Final response."
    assert all(row["discarded"] for row in record["generation_trace"][:-1])
    ids = [unit for row in record["recovery_rounds"] for unit in row["appended_unit_ids"]]
    assert len(ids) == len(set(ids))
    assert all(check["status"] == "passed" for check in record["pre_generation_budget_checks"])


def test_no_new_evidence_stops_without_another_generation(tmp_path):
    controller = make_controller(R=3)
    journal = tmp_path / "attempts.jsonl"
    generator = Generator(journal, ["lookup violet item-17", "No matching history words."])
    runner = EventNativeDecisionRunner(
        controller, generator, UnitTokenizer(), ratio=4, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(journal))
    record = runner.run(request())
    assert record["generation_attempts"] == 2
    assert len(record["recovery_rounds"]) == 1
    assert len(record["recovery_checks"]) == 2
    assert record["recovery_checks"][-1]["appended_unit_count"] == 0


def test_detector_still_gates_stop_decisions():
    controller = make_controller(D="detector")
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9))
    result = controller.reconsider(prepared, [], draft_text="lookup complete")
    assert not result["regenerate"]
    assert result["decision"]["gate"]["triggered"] is False


def test_failed_append_preserves_draft_and_does_not_create_lease(monkeypatch):
    controller = make_controller()
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    original = prepared.memory
    monkeypatch.setattr(controller._packer, "_try_measure", lambda *args, **kwargs: None)
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
    assert not result["regenerate"]
    assert result["memory"] is original
    assert controller._leases["task-1"] == []


def test_prepare_and_reconsider_remain_idempotent_after_creating_task_lease():
    controller = make_controller(L="task")
    prepared, first = recover(controller)
    assert first["regenerate"]
    repeated = controller.prepare(request(), ratio=4, max_new_tokens=32)
    assert repeated is prepared
    again = controller.reconsider(repeated, [draft_call()], draft_text="lookup violet item-17")
    assert again == first
    assert controller._recovery_counts["task-1"] == 1


def test_config_overlay_does_not_modify_selected_d3():
    original = {"post_draft_recovery": recovery_config("disabled")}
    saved = copy.deepcopy(original)
    configured = configure_controller(original, {"G": "adjacent_pair", "K": 4, "R": 3})
    assert original == saved
    assert configured["gp_experiments"]["K"] == 4
    with pytest.raises(ValueError, match="Unknown"):
        parse_gp_config({"replace": True})
    with pytest.raises(ValueError, match="R"):
        parse_gp_config({"R": True})
    with pytest.raises(ValueError, match="api_key_env"):
        parse_gp_config({"backend": {"api_key": "not-a-real-key"}})
