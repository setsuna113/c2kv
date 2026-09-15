"""Integration contract for Full visibility with a matched exact controller."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

import proxy  # noqa: E402
from arms import get_arm  # noqa: E402
from memory_runtime.adapter import RuntimeAdapter  # noqa: E402
from memory_runtime.tests.test_exact_adapter import (  # noqa: E402
    HISTORY_BUDGET,
    _adapter,
    _compressed_fixture,
    _context,
    _count,
    _draft,
    _full,
    _source,
)
from test_exact_recovery_proxy import (  # noqa: E402
    _ExactRuntime,
    _synthetic_body,
)
from test_memory_runtime_proxy import _Backend, _drive, _payload  # noqa: E402


def test_below_capacity_keeps_full_identity_and_ticks_persistent_clock():
    runtime = RuntimeAdapter(
        {
            "mode": "full_exact_shared",
            "run_id": "test",
            "bytes_per_kv_token": 1,
            "history_budget_bytes": 10_000,
            "workspace_budget_bytes": HISTORY_BUDGET,
            "lease_decisions": 3,
            "max_retrieved_events": 1,
        },
        _count,
    )
    source = _source()
    full, full_counts = _full(source)

    def forbidden_compression(_messages):
        raise AssertionError("Full exact shared must not render compressed history")

    for index in (1, 2):
        out, counts, prepared = runtime.prepare_exact(
            source,
            copy.deepcopy(full),
            copy.deepcopy(full_counts),
            _context(f"d{index}"),
            [],
            render_compressed=forbidden_compression,
        )
        metadata = counts["memory_runtime"]
        assert out == full
        assert metadata["auxiliary_gate"]["auxiliary_activated"] is False
        assert metadata["evidence_bytes"] == 0
        assert metadata["budget_applies"] is False
        assert metadata["policy"]["mode"] == "persistent"
        assert metadata["policy"]["decision_index"] == index
        assert metadata["policy"]["pre_draft_retrieval"] is False
        assert prepared.visible_source_indices == frozenset(range(len(source)))


def test_above_capacity_matches_exact_selection_but_keeps_all_full_raw_visible():
    source = _source()
    full, full_counts = _full(source)
    untouched = copy.deepcopy(full)

    reference_calls = []

    def render_compressed(messages):
        reference_calls.append(copy.deepcopy(messages))
        return tuple(copy.deepcopy(value) for value in _compressed_fixture())

    _, reference_counts, _ = _adapter("capacity_exact_persistent").prepare_exact(
        source,
        copy.deepcopy(full),
        copy.deepcopy(full_counts),
        _context("d1"),
        [],
        render_compressed=render_compressed,
    )

    def forbidden_compression(_messages):
        raise AssertionError("Full exact shared must not render compressed history")

    runtime = _adapter("full_exact_shared")
    out, counts, prepared = runtime.prepare_exact(
        source,
        copy.deepcopy(full),
        copy.deepcopy(full_counts),
        _context("d1"),
        [],
        render_compressed=forbidden_compression,
    )
    metadata = counts["memory_runtime"]
    reference = reference_counts["memory_runtime"]
    evidence_index = metadata["evidence_out_index"]

    assert reference_calls == [source]
    assert metadata["auxiliary_gate"]["auxiliary_activated"] is True
    assert metadata["auxiliary_reference_mode"] == "capacity_exact_persistent"
    assert metadata["selected_event_ids"] == reference["selected_event_ids"]
    assert metadata["protected_event_ids"] == reference["protected_event_ids"]
    assert metadata["auxiliary_selection_bytes"] == reference["evidence_bytes"]
    assert metadata["policy"]["selected_cost_bytes"] == reference["policy"][
        "selected_cost_bytes"
    ]
    assert metadata["policy"]["pre_draft_retrieval"] is False
    assert metadata["retrieved_event_ids"] == []
    assert out[:evidence_index] + out[evidence_index + 1 :] == full == untouched
    assert metadata["active_history_bytes"] > HISTORY_BUDGET
    assert metadata["budget_applies"] is False
    assert metadata["workspace_budget_applies"] is True
    assert metadata["evidence_bytes"] <= metadata["workspace_budget_bytes"]
    assert metadata["gist_tokens"] == 0
    assert counts["compressed_records"] == []
    assert prepared.visible_source_indices == frozenset(range(len(source)))

    reconsidered = runtime.reconsider(prepared, _draft())
    assert reconsidered["regenerate"] is False
    assert reconsidered["messages"] == out
    assert reconsidered["decision"]["status"] == "no_op"
    assert reconsidered["decision"]["reason"] == "all_bindings_visible"
    assert reconsidered["decision"]["upgrade_count"] == 0
    assert reconsidered["decision"]["regeneration_allowed"] is False


def test_actual_auxiliary_insertion_must_fit_workspace_budget():
    source = [
        {"role": "assistant", "content": "historical marker " * 1000},
        *_source(),
    ]
    full, full_counts = _full(source)

    def nonadditive_count(messages, tools):
        text = json.dumps(messages)
        return _count(messages, tools) + (
            10_000
            if "historical marker" in text and "source_indices" in text
            else 0
        )

    runtime = RuntimeAdapter(
        {
            "mode": "full_exact_shared",
            "run_id": "test",
            "bytes_per_kv_token": 1,
            "history_budget_bytes": HISTORY_BUDGET,
            "workspace_budget_bytes": HISTORY_BUDGET,
            "lease_decisions": 3,
            "max_retrieved_events": 1,
        },
        nonadditive_count,
    )
    with pytest.raises(ValueError, match="measured workspace byte cap"):
        runtime.prepare_exact(
            source,
            full,
            full_counts,
            _context("d1"),
            [],
            render_compressed=lambda _messages: (_ for _ in ()).throw(
                AssertionError("Full exact shared must not render compressed history")
            ),
        )


def test_developer_role_is_rejected_until_the_reference_renderer_supports_it():
    runtime = _adapter("full_exact_shared")
    source = [
        {"role": "developer", "content": "Keep this exact instruction."},
        *_source(),
    ]
    full, full_counts = _full(source)
    with pytest.raises(ValueError, match="developer-role"):
        runtime.prepare_exact(
            source,
            full,
            full_counts,
            _context("d1"),
            [],
            render_compressed=lambda _messages: (_ for _ in ()).throw(
                AssertionError("Full exact shared must not render compressed history")
            ),
        )


class _FullExactRuntime(_ExactRuntime):
    mode = "full_exact_shared"


def test_proxy_uses_plain_full_and_records_the_exact_draft_trace(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)

    runtime = _FullExactRuntime("no_op")
    body = _synthetic_body(
        "SYNTHETIC_FULL_EXACT_FINAL",
        "SYNTHETIC_FULL_EXACT_TOOL_CALL",
        {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )
    log_path = tmp_path / "synthetic_full_exact_shared.jsonl"
    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("full"),
        _Backend(),
        _payload(),
        [body],
        str(log_path),
    )

    assert len(posts) == 1
    assert len(runtime.prepare_calls) == len(runtime.reconsider_calls) == 1
    assert len(sent) == 1 and sent[0][0] == 200
    assert sent[0][1]["c2kv_proxy"]["generation_attempts"] == 1
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(row["generation_trace"]) == 1
    assert row["generation_trace"][0]["phase"] == "draft"
    assert row["generation_trace"][0]["discarded"] is False
    assert row["generation_trace"][0]["memory_runtime"]["mode"] == (
        "full_exact_shared"
    )

    with pytest.raises(proxy.MemoryRuntimeError, match="requires plain arm 'full'"):
        proxy._validate_memory_runtime_arm(runtime, get_arm("c2kv4"))
