"""Synthetic integration fixtures for capacity-gated exact recovery."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

from memory_runtime.adapter import RuntimeAdapter, raw_source_cutoff  # noqa: E402


FIXTURE_ID = "synthetic_exact_adapter_v1"
HISTORY_BUDGET = 350


def _count(messages, tools):
    del tools
    if any(
        isinstance(message.get("content"), str)
        and "SYNTHETIC_FULL_VISIBLE_GATE" in message["content"]
        for message in messages
    ):
        return 1000
    tokens = len(messages) * 100
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(
            "Historical evidence from this conversation."
        ):
            continue
        payload = json.loads(content.split("\n", 1)[1])
        tokens += len(payload["events"]) * 100
    return tokens


def _adapter(mode, *, workspace=HISTORY_BUDGET):
    return RuntimeAdapter(
        {
            "mode": mode,
            "run_id": "test",
            "bytes_per_kv_token": 1,
            "history_budget_bytes": HISTORY_BUDGET,
            "workspace_budget_bytes": workspace,
            "lease_decisions": 3,
            "max_retrieved_events": 1,
        },
        _count,
    )


def _context(decision):
    return {
        "task_id": "synthetic",
        "attempt": 0,
        "decision_id": decision,
        "fixture_id": FIXTURE_ID,
    }


def _source():
    return [
        {"role": "user", "content": "Remember item-17 from archive."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "user", "content": "Check the clock."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "recent",
                "type": "function",
                "function": {
                    "name": "clock",
                    "arguments": '{"city":"London"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "recent", "content": "09:30"},
        {"role": "user", "content": "Continue with the archived item."},
    ]


def _full(source):
    messages = [{"role": "system", "content": "system"}] + copy.deepcopy(source)
    return messages, {
        "current_start_out_index": 1 + raw_source_cutoff(source),
        "compressed_records": [],
    }


def _compressed_fixture():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "synthetic gist", "c2kv_key_hash": "g0"},
        {"role": "user", "content": "09:30"},
        {"role": "user", "content": "Continue with the archived item."},
    ]
    counts = {
        "current_start_out_index": 2,
        "current_raw": 2,
        "history_raw": 0,
        "system_raw": 1,
        "gist_tokens": 50,
        "n_docs": 1,
        "dropped_docs": 0,
        "history_packed_original_tokens": 200,
        "compressed_records": [{
            "out_index": 1,
            "source_indices": [1, 2, 3, 4],
            "record": {
                "key_hash": "g0",
                "gist_len": 50,
                "original_seq_len": 200,
            },
        }],
    }
    return messages, counts


def _renderer(source, calls):
    expected = _source()

    def render(messages):
        assert messages == source == expected
        calls.append(copy.deepcopy(messages))
        return tuple(copy.deepcopy(value) for value in _compressed_fixture())

    return render


def _draft():
    return [{
        "id": "unsubmitted-draft",
        "type": "function",
        "function": {
            "name": "unverified_tool",
            "arguments": '{"id":"item-17"}',
        },
    }]


def test_first_exact_acquisition_matches_once_and_persistent_without_reextract():
    source = _source()
    full, full_counts = _full(source)
    observations = {}

    for mode in ("capacity_exact_once", "capacity_exact_persistent"):
        runtime = _adapter(mode)
        render_calls = []
        initial, initial_counts, prepared = runtime.prepare_exact(
            source,
            copy.deepcopy(full),
            copy.deepcopy(full_counts),
            _context("d1"),
            [],
            render_compressed=_renderer(source, render_calls),
        )
        initial_ids = tuple(initial_counts["memory_runtime"]["selected_event_ids"])
        initial_cost = initial_counts["memory_runtime"]["evidence_bytes"]
        assert initial_ids == ("synthetic:m3",)
        assert initial_cost == 200
        assert prepared.decision.decision_index == 1
        assert prepared.decision.decision_key == "d1"
        assert prepared.decision.store.session_id == "synthetic"
        assert prepared.decision.selection.selected_event_ids == initial_ids
        assert prepared.decision.visible_event_ids == frozenset({"synthetic:m5"})
        assert len(render_calls) == 1

        reconsidered = runtime.reconsider(prepared, _draft())
        repeated = runtime.reconsider(prepared, _draft())
        meta = reconsidered["counts"]["memory_runtime"]
        assert repeated is reconsidered
        assert reconsidered["regenerate"] is True
        assert tuple(meta["selected_event_ids"]) == (
            "synthetic:m0", "synthetic:m3"
        )
        assert tuple(meta["retrieved_event_ids"]) == ("synthetic:m0",)
        assert meta["evidence_bytes"] == 300
        assert meta["policy"]["decision_index"] == 1
        assert meta["exact_recovery"]["decision_index"] == 1
        assert meta["exact_recovery"]["upgrade_count"] == 1
        assert meta["exact_recovery"]["candidate_event_id"] == "synthetic:m0"
        assert len(render_calls) == 1
        assert sum(
            isinstance(message.get("content"), str)
            and message["content"].startswith(
                "Historical evidence from this conversation."
            )
            for message in reconsidered["messages"]
        ) == 1
        observations[mode] = {
            "initial": initial,
            "initial_ids": initial_ids,
            "initial_cost": initial_cost,
            "upgraded": reconsidered["messages"],
            "upgraded_ids": tuple(meta["selected_event_ids"]),
            "upgraded_cost": meta["evidence_bytes"],
        }

    assert observations["capacity_exact_once"] == observations[
        "capacity_exact_persistent"
    ]


def test_below_budget_full_visible_decisions_tick_and_expire_persistent_lease():
    runtime = _adapter("capacity_exact_persistent")
    source = _source()
    full, full_counts = _full(source)
    _, _, prepared = runtime.prepare_exact(
        source,
        full,
        full_counts,
        _context("d1"),
        [],
        render_compressed=_renderer(source, []),
    )
    assert runtime.reconsider(prepared, _draft())["regenerate"] is True

    below = []
    for decision in range(2, 5):
        source = source + [
            {"role": "assistant", "content": f"finished decision {decision}"},
            {
                "role": "user",
                "content": f"SYNTHETIC_FULL_VISIBLE_GATE decision {decision}",
            },
        ]
        full, full_counts = _full(source)

        def forbidden_compression(_):
            raise AssertionError("below-B Full must not render compressed history")

        out, counts, current = runtime.prepare_exact(
            source,
            full,
            full_counts,
            _context(f"d{decision}"),
            [],
            render_compressed=forbidden_compression,
        )
        policy = counts["memory_runtime"]["policy"]
        assert out == full
        assert counts["memory_runtime"]["capacity_gate"]["compression_activated"] is False
        assert counts["memory_runtime"]["evidence_bytes"] == 0
        assert counts["memory_runtime"]["evidence_out_index"] is None
        assert policy["decision_index"] == decision
        assert policy["selected_cost_bytes"] == 0
        assert policy["pre_draft_retrieval"] is False
        if decision == 2:
            no_op = runtime.reconsider(current, _draft())
            assert no_op["regenerate"] is False
            assert no_op["decision"]["reason"] == "all_bindings_visible"
        below.append((counts, current))

    assert below[0][0]["memory_runtime"]["policy"]["expired_lease_event_ids"] == ()
    assert below[1][0]["memory_runtime"]["policy"]["expired_lease_event_ids"] == ()
    assert below[2][0]["memory_runtime"]["policy"]["expired_lease_event_ids"] == (
        "synthetic:m0",
    )


def test_upgrade_budget_failure_abstains_without_regeneration_or_rerender():
    runtime = _adapter("capacity_exact_once", workspace=250)
    source = _source()
    full, full_counts = _full(source)
    render_calls = []
    initial, _, prepared = runtime.prepare_exact(
        source,
        full,
        full_counts,
        _context("d1"),
        [],
        render_compressed=_renderer(source, render_calls),
    )

    reconsidered = runtime.reconsider(prepared, _draft())
    repeated = runtime.reconsider(prepared, _draft())

    assert repeated is reconsidered
    assert reconsidered["regenerate"] is False
    assert reconsidered["messages"] == initial
    assert reconsidered["decision"]["status"] == "abstain"
    assert reconsidered["decision"]["reason"] == "budget_exhausted"
    assert reconsidered["decision"]["required_cost_bytes"] == 300
    assert reconsidered["decision"]["budget_bytes"] == 250
    assert reconsidered["decision"]["upgrade_count"] == 0
    assert prepared.decision.decision_index == 1
    assert prepared.decision.selection.metadata["upgrade"]["status"] == "not_requested"
    assert len(render_calls) == 1
