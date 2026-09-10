"""Behavioral contract tests for the additive pre-B policy."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.always_compress import coverage_accounting, ratio_accounting
from memory_runtime.generation_budget import GenerationBudget, TaskGenerationBudgetExceeded


def count(messages, tools):
    return len(json.dumps(messages, separators=(",", ":")))


def setup_runtime(monkeypatch, mode="ac_gist_static", budget=5000, protocol="fixed-budget-main"):
    runtime = RuntimeAdapter(dict(
        mode=mode, run_id="test", bytes_per_kv_token=1,
        history_budget_bytes=budget, workspace_budget_bytes=budget,
        history_view_protocol=protocol), count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 12)
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 512)
    calls = []

    def extract(role, text, ratio, timeout):
        calls.append(text)
        return dict(key_hash=f"g{len(calls)}", gist_len=20, original_seq_len=80)

    monkeypatch.setattr(proxy, "_extract", extract)
    return runtime, calls


def source():
    return [{"role": "user", "content": "Keep item-17."},
            {"role": "assistant", "content": "Recorded."},
            {"role": "user", "content": "Continue."}]


def prepare(messages):
    return proxy._prepare_memory_input(
        messages, dict(task_id="t", attempt=0, decision_id="d1"), [])


@pytest.mark.parametrize("mode", ["ac_gist_static", "ac_protect"])
def test_full_fits_but_new_route_still_extracts_and_reserves_gist(monkeypatch, mode):
    runtime, calls = setup_runtime(monkeypatch, mode)
    out, counts = prepare(source())
    meta = counts["memory_runtime"]
    assert calls and any(m.get("c2kv_key_hash") for m in out)
    assert meta["capacity_gate"]["full_history_bytes"] < 5000
    assert meta["capacity_gate"]["compression_activated"] is True
    assert meta["mode"] == runtime.mode and meta["route_mode"] == mode
    assert meta["compression_policy"] == "always-compress-v1"
    assert meta["gist_reservation"]["reserved_gist_bytes"] == 20
    assert meta["gist_reservation"]["satisfied"] is True
    assert meta["source_coverage"]["complete_history_coverage"] is True
    assert meta["compression_ratio"]["includes_coverage_loss"] is False
    assert meta["eligible_history_count"] == 2


def test_first_input_has_no_extraction_and_explicit_reason(monkeypatch):
    _, calls = setup_runtime(monkeypatch)
    out, counts = prepare([{"role": "user", "content": "Hello."}])
    meta = counts["memory_runtime"]
    assert not calls
    assert meta["no_eligible_history"] is True
    assert meta["policy"]["selection"] == "no_eligible_history"
    assert meta["gist_reservation"]["required"] is False
    assert meta["gist_tokens"] == 0


def test_too_small_budget_is_infeasible_not_raw_fallback(monkeypatch):
    setup_runtime(monkeypatch, budget=19)
    with pytest.raises(proxy.MemoryRuntimeError) as error:
        prepare(source())
    assert error.value.kind == "capacity_infeasible"


def test_capacity_failure_logs_the_task_and_zero_actual_generations(monkeypatch, tmp_path):
    from test_memory_runtime_proxy import _Backend, _drive, _payload
    runtime, _ = setup_runtime(monkeypatch, budget=19)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    budget = GenerationBudget(96, per_task_limit=96)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", budget)
    path = tmp_path / "infeasible.jsonl"
    sent, posts = _drive(monkeypatch, runtime, get_arm("c2kv4"), _Backend(),
                        _payload(messages=source(), c2kv_eval_context=dict(
                            task_id="t", attempt=0, decision_id="d1")), [], str(path))
    assert not posts and budget.consumed == 0
    assert sent[0][0] >= 400
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["status"] == "capacity_infeasible"
    assert row["generation_budget"] == dict(
        limit=96, consumed_before=0, consumed_after=0, attempt_indices=[],
        per_task_limit=96, task_id="t", task_consumed_before=0, task_consumed_after=0)


def test_native_packing_drops_remain_visible_in_coverage_ledger(monkeypatch):
    setup_runtime(monkeypatch)
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    messages = source()[:2] + [
        {"role": "user", "content": "Second topic."},
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": "Continue."}]
    _, counts = prepare(messages)
    coverage = counts["memory_runtime"]["source_coverage"]
    assert coverage["eligible_source_indices"] == [0, 1, 2, 3]
    assert coverage["fitted_fragment_count"] == 2
    assert coverage["retained_fragment_count"] == 1
    assert coverage["unrepresented_source_indices"] == [0, 1]
    assert coverage["complete_history_coverage"] is False


def test_coverage_diagnostic_rejects_native_preselection_loss(monkeypatch):
    setup_runtime(monkeypatch, protocol="coverage-preserving-diagnostic")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    messages = source()[:2] + source()
    with pytest.raises(proxy.MemoryRuntimeError) as error:
        prepare(messages)
    assert error.value.kind == "unsupported_by_native_packing"


def test_coverage_diagnostic_keeps_all_gists_without_fixed_b_claim(monkeypatch):
    setup_runtime(monkeypatch, budget=1, protocol="coverage-preserving-diagnostic")
    _, counts = prepare(source())
    meta = counts["memory_runtime"]
    assert meta["gist_tokens"] == 20 and meta["active_history_bytes"] > 1
    assert meta["budget_applies"] is False
    assert meta["source_coverage"]["complete_history_coverage"] is True


def test_source_split_needs_all_fragments_or_complete_raw_occurrence():
    fragments = [dict(fragment_id=i, source_indices=[4], encoder_input_tokens=100)
                 for i in (0, 1)]
    block = dict(packing_fragment_id=0, source_indices=[4])
    partial = coverage_accounting(eligible_sources=frozenset({4}), raw_sources=set(),
                                  retained_blocks=[block], packing_fragments=fragments)
    assert partial["gist_touched_source_indices"] == [4]
    assert partial["gist_fully_represented_source_indices"] == []
    assert partial["fitted_fragment_retained_token_fraction"] == 0.5
    raw = coverage_accounting(eligible_sources=frozenset({4}), raw_sources={4},
                              retained_blocks=[block], packing_fragments=fragments)
    assert raw["complete_history_coverage"] is True
    assert raw["raw_gist_overlap_source_indices"] == [4]


def test_ratio_uses_same_common_boundary_and_counts_raw_plus_gist():
    full = dict(active_history_bytes=400, common_raw_prompt_tokens=50)
    runtime = dict(bytes_per_kv_token=2, active_history_bytes=200,
                   common_raw_prompt_tokens=50, gist_tokens=60, raw_history_tokens=40)
    ratio = ratio_accounting(full, runtime)
    assert ratio["n_history"] == 2
    assert ratio["n_total"] == 500 / 300
    assert ratio["active_gist_bytes"] == 120 and ratio["active_raw_history_bytes"] == 80
    with pytest.raises(ValueError, match="common-input boundary"):
        ratio_accounting(full, {**runtime, "common_raw_prompt_tokens": 51})


def test_proxy_reserves_actual_generations_per_explicit_task(monkeypatch):
    budget = GenerationBudget(10, per_task_limit=2)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", budget)
    token = proxy._ATTEMPT_REQUEST.set({"eval_context": {"task_id": "t"}})
    try:
        assert proxy._reserve_generation_attempt("/v1/chat/completions") == 1
        assert proxy._reserve_generation_attempt("/v1/chat/completions") == 2
        with pytest.raises(TaskGenerationBudgetExceeded):
            proxy._reserve_generation_attempt("/v1/chat/completions")
        assert budget.consumed == 2
        proxy._ATTEMPT_REQUEST.get()["eval_context"]["task_id"] = "next"
        assert proxy._reserve_generation_attempt("/v1/chat/completions") == 3
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)


@pytest.mark.parametrize("mode", ["ac_exact_once", "ac_exact_persistent"])
def test_exact_upgrade_cannot_spend_the_last_gist_reservation(mode):
    from memory_runtime.tests.test_exact_adapter import (
        _source, _full, _compressed_fixture, _count, _context, _draft)
    runtime = RuntimeAdapter(dict(
        mode=mode, run_id="test", bytes_per_kv_token=1,
        history_budget_bytes=350, workspace_budget_bytes=350), _count)
    messages = _source()
    compressed, counts = _compressed_fixture()
    counts["compressed_records"][0]["record"]["gist_len"] = 100
    _, initial, prepared = runtime.prepare_exact(
        messages, *_full(messages), _context("d1"), [],
        render_compressed=lambda _: (compressed, counts))
    assert initial["memory_runtime"]["evidence_bytes"] == 200
    assert initial["memory_runtime"]["gist_reservation"]["selection_budget_bytes"] == 250
    result = runtime.reconsider(prepared, _draft())
    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "budget_exhausted"
    assert result["counts"]["memory_runtime"]["gist_tokens"] == 100


def test_min_gist_skips_large_anchor_to_fit_mandatory_query():
    from memory_runtime.tests.test_exact_adapter import (
        _source, _full, _compressed_fixture, _count, _context)
    runtime = RuntimeAdapter(dict(
        mode="ac_protect", run_id="test", bytes_per_kv_token=1,
        history_budget_bytes=250, workspace_budget_bytes=250), _count)
    messages = _source()[:-1]
    compressed, counts = _compressed_fixture()
    compressed = compressed[:-1]
    compressed.insert(2, dict(role="user", content="second gist", c2kv_key_hash="g1"))
    counts["current_start_out_index"] = 3
    counts["compressed_records"][0]["record"]["gist_len"] = 100
    counts["compressed_records"].append(dict(
        out_index=2, source_indices=[3, 4],
        record=dict(key_hash="g1", gist_len=50, original_seq_len=100)))
    out, result = runtime.apply(
        messages, *_full(messages), _context("d1"), [],
        render_compressed=lambda _: (compressed, counts))
    meta = result["memory_runtime"]
    assert meta["protected_event_ids"] == ["synthetic:m2"]
    assert meta["evidence_bytes"] == 200 and meta["gist_tokens"] == 50
    assert meta["active_history_bytes"] == 250
    assert [m["c2kv_key_hash"] for m in out if m.get("c2kv_key_hash")] == ["g1"]
