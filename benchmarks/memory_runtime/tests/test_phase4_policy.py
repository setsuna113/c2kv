"""Observable behavior of the optional Phase4 development controls."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.events import EventStore
from memory_runtime.adapter import PreparedExact
from memory_runtime.phase4_policy import Phase4RuntimeAdapter, VERSION
from memory_runtime.policy import PolicyInputError


def _runtime(trigger="conservative", retention="finite", numerator=1, denominator=1):
    return Phase4RuntimeAdapter({
        "mode": "capacity_exact_persistent", "run_id": "test", "bytes_per_kv_token": 1,
        "history_budget_bytes": 8, "workspace_budget_bytes": 8,
        "lease_decisions": 3, "max_retrieved_events": 1,
        "phase4_policy": {"schema": VERSION, "trigger": trigger, "retention": retention,
            "random_probability": {"numerator": numerator, "denominator": denominator},
            "random_seed": 0},
    }, lambda messages, tools: len(messages))


def _calls(value="item-17"):
    return [{"type": "function", "id": "unsubmitted", "function": {
        "name": "read", "arguments": {"id": value}}}]


def _prefix():
    return [{"role": "user", "content": "Remember item-17 from the archive."},
            {"role": "assistant", "content": "Recorded."},
            {"role": "user", "content": "Continue."}]


def _prepare(runtime, messages, index, *, full_visible=False, cost=len):
    store = EventStore.from_messages("s", messages)
    memory = runtime._exact_memory(("test", "s", "0"), "s")
    visible_indices = set(range(len(messages))) if full_visible else {len(messages) - 1}
    visible_ids = {event.event_id for event in store.events
                   if set(event.source_indices) <= visible_indices}
    handle = memory.prepare_decision(store, cost, visible_ids, str(index))
    def render(selection, started):
        del started
        return [], {"memory_runtime": {
            "selected_event_ids": list(selection.selected_event_ids),
            "policy": selection.metadata}}
    _, counts = render(handle.selection, 0)
    prepared = PreparedExact(runtime, memory, handle, len(messages) - 1,
        frozenset(visible_indices | {source for event_id in handle.selection.selected_event_ids
                  for source in store.event(event_id).source_indices}),
        None if full_visible else render, [], counts)
    return prepared


def test_random_gate_changes_timing_but_uses_same_proposed_gap_source():
    results = {}
    for trigger, numerator in (("off", 1), ("conservative", 1), ("random", 1), ("random", 0)):
        runtime = _runtime(trigger, numerator=numerator)
        prepared = _prepare(runtime, _prefix(), 1)
        result = runtime.reconsider(prepared, _calls())
        assert runtime.reconsider(prepared, copy.deepcopy(_calls())) is result
        assert prepared.decision.decision_index == 1
        with pytest.raises(PolicyInputError, match="second different draft"):
            runtime.reconsider(prepared, _calls("different"))
        results[(trigger, numerator)] = result
    assert not results[("off", 1)]["regenerate"]
    assert not results[("random", 0)]["regenerate"]
    for trigger in ("conservative", "random"):
        assert results[(trigger, 1)]["regenerate"]
        assert results[(trigger, 1)]["decision"]["upgraded_event_id"] == "s:m0"


def test_random_false_trigger_has_visible_binding_and_unique_hidden_source():
    messages = _prefix()
    messages[-1]["content"] = "Continue using item-17."
    conservative = _runtime()
    prepared_c = _prepare(conservative, messages, 1)
    assert not conservative.reconsider(prepared_c, _calls())["regenerate"]
    random = _runtime("random")
    prepared_r = _prepare(random, messages, 1)
    assert random.proposed_source(prepared_r, _calls()) == "s:m0"
    assert random.reconsider(prepared_r, _calls())["regenerate"]
    assert random.commit_final(prepared_r, _calls())["renewals"] == []


@pytest.mark.parametrize("case", ["missing", "ambiguous", "full_visible", "over_budget", "malformed"])
def test_random_cannot_invent_a_source_or_bypass_admission(case):
    messages = _prefix()
    calls = _calls()
    if case == "missing":
        calls = _calls("not-in-prefix")
    elif case == "ambiguous":
        messages.insert(1, {"role": "user", "content": "Second mention of item-17."})
    elif case == "malformed":
        calls[0]["function"]["arguments"] = '{"id":'
    runtime = _runtime("random")
    prepared = _prepare(runtime, messages, 1, full_visible=case == "full_visible",
                        cost=(lambda ids: 9 * len(ids)) if case == "over_budget" else len)
    assert runtime.proposed_source(prepared, calls) is None
    assert runtime.reconsider(prepared, calls)["regenerate"] is False
    assert not prepared.memory._leases


def _advance(runtime, messages, index, final_calls, *, full_visible=False, cost=len):
    messages += [{"role": "assistant", "content": "continue"},
                 {"role": "user", "content": f"Next operation {index}."}]
    prepared = _prepare(runtime, messages, index, full_visible=full_visible, cost=cost)
    runtime.reconsider(prepared, None)
    result = runtime.commit_final(prepared, final_calls)
    return prepared, result


def test_final_reference_extends_inactivity_ttl_and_then_expires_without_reference():
    observations = {}
    for retention in ("finite", "final_reference"):
        runtime = _runtime(retention=retention)
        messages = _prefix()
        first = _prepare(runtime, messages, 1)
        assert runtime.reconsider(first, _calls())["regenerate"]
        runtime.commit_final(first, _calls())
        second, committed = _advance(runtime, messages, 2, _calls())
        assert runtime.commit_final(second, copy.deepcopy(_calls())) is committed
        with pytest.raises(PolicyInputError, match="different final draft"):
            runtime.commit_final(second, None)
        _advance(runtime, messages, 3, None)
        fourth, _ = _advance(runtime, messages, 4, None)
        fifth, _ = _advance(runtime, messages, 5, _calls())
        observations[retention] = (committed, fourth, fifth)
    assert observations["finite"][0]["renewals"] == []
    assert observations["final_reference"][0]["renewals"] == [
        {"event_id": "s:m0", "previous_expiry": 4, "new_expiry": 5}]
    assert "s:m0" not in observations["finite"][1].decision.selection.selected_event_ids
    assert "s:m0" in observations["final_reference"][1].decision.selection.retained_event_ids
    assert observations["final_reference"][2].decision.selection.metadata["expired_lease_event_ids"] == ("s:m0",)
    assert not observations["final_reference"][2].memory._leases


def test_discarded_reference_and_invisible_budget_skipped_lease_do_not_renew():
    runtime = _runtime(retention="final_reference")
    messages = _prefix()
    first = _prepare(runtime, messages, 1)
    runtime.reconsider(first, _calls())
    runtime.commit_final(first, None)
    messages += [{"role": "assistant", "content": "done"}, {"role": "user", "content": "Next."}]
    second = _prepare(runtime, messages, 2)
    runtime.reconsider(second, _calls())
    assert runtime.commit_final(second, None)["renewals"] == []
    assert second.memory._leases["s:m0"].expires_at_decision == 4
    third, commit = _advance(runtime, messages, 3, _calls(), cost=lambda ids: 9 * len(ids))
    assert "s:m0" not in third.counts["memory_runtime"]["selected_event_ids"]
    assert commit["renewals"] == []
    assert third.memory._leases["s:m0"].expires_at_decision == 4


def test_full_raw_visibility_can_renew_but_explicit_revision_cannot():
    runtime = _runtime(retention="final_reference")
    messages = _prefix()
    first = _prepare(runtime, messages, 1)
    runtime.reconsider(first, _calls())
    runtime.commit_final(first, _calls())
    _, second = _advance(runtime, messages, 2, _calls(), full_visible=True)
    assert second["renewals"][0]["new_expiry"] == 5
    messages += [{"role": "assistant", "content": "done"},
                 {"role": "user", "content": "Actually, cancel the previous request."}]
    third = _prepare(runtime, messages, 3)
    runtime.reconsider(third, None)
    assert third.decision.selection.metadata["revision_cancelled_event_ids"] == ("s:m0",)
    assert runtime.commit_final(third, _calls())["renewals"] == []
    assert not third.memory._leases


def test_native_parse_failure_and_new_binding_cannot_create_retention_witnesses():
    runtime = _runtime(retention="final_reference")
    messages = _prefix()
    first = _prepare(runtime, messages, 1)
    runtime.reconsider(first, _calls())
    runtime.commit_final(first, _calls())
    _, second = _advance(runtime, messages, 2, _calls("other-value"))
    assert second["renewals"] == []
    bad = _calls()
    bad[0]["function"]["arguments"] = '{"id":"item-17", "id":"item-17"}'
    _, third = _advance(runtime, messages, 3, bad)
    assert third["parse_status"] == "malformed_arguments"
    assert third["renewals"] == []


def test_development_parameters_reject_invalid_rate_and_unplanned_policy_pair():
    with pytest.raises(ValueError, match="exact rational"):
        _runtime("random", numerator=2, denominator=1)
    with pytest.raises(ValueError, match="exact rational"):
        _runtime("random", denominator=0)
    with pytest.raises(ValueError, match="conservative acquisition"):
        _runtime("random", "final_reference")
