"""Synthetic adapter integration of shared E and the NoGist raw body."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.tests.test_exact_adapter import (
    _source, _full, _compressed_fixture, _context, _count, _draft,
)


def _runtime(mode, *, history=350, workspace=350):
    return RuntimeAdapter(dict(mode=mode, run_id="test", bytes_per_kv_token=1,
        history_budget_bytes=history, workspace_budget_bytes=workspace,
        lease_decisions=3, max_retrieved_events=1), _count)


def _prepare(runtime, decision="d1"):
    source = _source()
    full, counts = _full(source)

    def forbidden(_):
        raise AssertionError("NoGist cannot call the compressor")

    return runtime.prepare_exact(source, full, counts, _context(decision), [],
                                 render_compressed=forbidden)


def test_no_gist_keeps_shared_e_priority_and_spends_remaining_budget_on_raw():
    source = _source()
    full, counts = _full(source)
    _, reference, _ = _runtime("capacity_exact_persistent").prepare_exact(
        source, full, counts, _context("d1"), [],
        render_compressed=lambda _: copy.deepcopy(_compressed_fixture()))
    runtime = _runtime("capacity_exact_no_gist", workspace=250)
    out, counts, prepared = _prepare(runtime)
    metadata = counts["memory_runtime"]
    assert metadata["selected_event_ids"] == reference["memory_runtime"]["selected_event_ids"]
    assert metadata["auxiliary_selection_bytes"] == reference["memory_runtime"]["evidence_bytes"]
    assert metadata["raw_history_event_ids"] == ["synthetic:m2"]
    assert metadata["active_history_bytes"] > metadata["workspace_budget_bytes"]
    assert metadata["active_history_bytes"] <= metadata["history_budget_bytes"]
    assert metadata["evidence_bytes"] <= metadata["workspace_budget_bytes"]
    assert counts["gist_tokens"] == 0 and counts["compressed_records"] == []
    assert not any(message.get("c2kv_key_hash") for message in out)
    draft = [{"function": {"name": "synthetic", "arguments": json.dumps({"query": "Check the clock."})}}]
    decision = runtime.reconsider(prepared, draft)
    assert decision["regenerate"] is False
    assert decision["decision"]["reason"] == "all_bindings_visible"
    assert prepared.decision.selection.retrieved_event_ids == ()


def test_no_gist_upgrade_refills_raw_body_and_preserves_the_decision_clock():
    runtime = _runtime("capacity_exact_no_gist")
    _, counts, prepared = _prepare(runtime)
    assert counts["memory_runtime"]["raw_history_event_ids"] == ["synthetic:m2"]
    decision = runtime.reconsider(prepared, _draft())
    metadata = decision["counts"]["memory_runtime"]
    assert decision["regenerate"] is True
    assert metadata["retrieved_event_ids"] == ["synthetic:m0"]
    assert metadata["raw_history_event_ids"] == []
    assert metadata["raw_body_evicted_on_upgrade"] == ["synthetic:m2"]
    assert metadata["active_history_bytes"] <= metadata["history_budget_bytes"]
    assert metadata["policy"]["decision_index"] == 1
    assert runtime.reconsider(prepared, _draft()) is decision


def test_no_gist_rejects_oversize_e_without_discarding_existing_raw():
    runtime = _runtime("capacity_exact_no_gist", workspace=250)
    out, counts, prepared = _prepare(runtime)
    decision = runtime.reconsider(prepared, _draft())
    assert decision["regenerate"] is False
    assert decision["decision"]["reason"] == "budget_exhausted"
    assert decision["messages"] == out and decision["counts"] is counts
    assert counts["memory_runtime"]["raw_history_event_ids"] == ["synthetic:m2"]
    assert counts["memory_runtime"]["policy"]["upgrade"]["upgrade_count"] == 0


def test_no_gist_below_budget_is_full_and_uses_the_full_proxy_arm():
    runtime = _runtime("capacity_exact_no_gist", history=10_000)
    out, counts, prepared = _prepare(runtime)
    assert out == _full(_source())[0]
    assert counts["memory_runtime"]["capacity_gate"]["compression_activated"] is False
    assert prepared.decision.decision_index == 1
    assert runtime.reconsider(prepared, _draft())["regenerate"] is False
    proxy._validate_memory_runtime_arm(runtime, get_arm("full"))
