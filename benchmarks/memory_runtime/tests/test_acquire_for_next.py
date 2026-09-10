"""Deferred acquisition keeps the consumed view and finite lease contract."""
import copy

import pytest
from benchmarks.memory_runtime.pre_b_design import CANONICAL_MODE_BY_VARIANT

from benchmarks.memory_runtime.tests.test_exact_adapter import (
    _adapter, _source, _full, _compressed_fixture, _context, _draft,
)


def prepare(runtime, decision):
    source = _source()
    return runtime.prepare_exact(
        source, *_full(source), _context(decision), [],
        render_compressed=lambda _: copy.deepcopy(_compressed_fixture()))


def test_deferred_admission_does_not_render_or_change_consumed_view():
    runtime = _adapter("ac_acquire_for_next")
    messages, counts, prepared = prepare(runtime, "d1")
    before_messages, before_meta = copy.deepcopy(messages), copy.deepcopy(counts["memory_runtime"])
    prepared.render = lambda *_: (_ for _ in ()).throw(AssertionError("unused view rendered"))
    result = runtime.reconsider(prepared, _draft())
    assert runtime.reconsider(prepared, _draft()) is result
    assert result["regenerate"] is False
    assert result["messages"] == before_messages
    current = result["counts"]["memory_runtime"]
    assert {k:v for k,v in current.items() if k != "exact_recovery"} == {
        k:v for k,v in before_meta.items() if k != "exact_recovery"}
    trace = result["decision"]
    assert trace["deferred_lease_acquisition_count"] == 1
    assert trace["upgrade_count"] == trace["actual_regeneration_count"] == 0
    assert trace["admitted_event_id"] == "synthetic:m0"
    assert trace["lease_expires_at_decision"] == 4
    assert prepared.decision.decision_index == 1
    assert current["selected_event_ids"] == ["synthetic:m3"]
    for index in (2, 3, 4):
        _, after, handle = prepare(runtime, f"d{index}")
        meta = after["memory_runtime"]
        assert handle.decision.decision_index == index
        assert ("synthetic:m0" in meta["retained_event_ids"]) is (index < 4)
        assert meta["active_history_bytes"] <= 350 and meta["gist_tokens"] > 0


def test_deferred_budget_rejection_does_not_create_a_lease():
    runtime = _adapter("ac_acquire_for_next", workspace=250)
    messages, _, prepared = prepare(runtime, "d1")
    result = runtime.reconsider(prepared, _draft())
    assert result["regenerate"] is False and result["messages"] == messages
    assert result["decision"]["reason"] == "budget_exhausted"
    assert not prepared.memory._leases


def test_candidate_and_incumbent_use_the_same_first_view_and_admission():
    observations = []
    for mode in ("ac_exact_persistent", "ac_acquire_for_next"):
        runtime = _adapter(mode)
        messages, counts, prepared = prepare(runtime, "d1")
        assert counts["memory_runtime"]["mode"] == CANONICAL_MODE_BY_VARIANT[mode]
        assert counts["memory_runtime"]["route_mode"] == mode
        result = runtime.reconsider(prepared, _draft())
        observations.append((messages, prepared.decision.selection, prepared.memory._leases))
        assert result["regenerate"] is (mode == "ac_exact_persistent")
    assert observations[0] == observations[1]


def test_different_second_draft_is_rejected_without_renewing_lease():
    runtime = _adapter("ac_acquire_for_next")
    _, _, prepared = prepare(runtime, "d1")
    runtime.reconsider(prepared, _draft())
    before = copy.deepcopy(prepared.memory._leases)
    with pytest.raises(ValueError, match="second different draft"):
        runtime.reconsider(prepared, [])
    assert prepared.memory._leases == before
