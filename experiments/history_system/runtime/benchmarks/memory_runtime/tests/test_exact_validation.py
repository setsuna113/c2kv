"""CPU-only validation tests for captured exact-recovery requests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

import proxy  # noqa: E402
from arms import get_arm  # noqa: E402
from memory_runtime.adapter import EventStore, evidence_message, raw_source_cutoff  # noqa: E402
from memory_runtime.exact_validation import validate_exact_request  # noqa: E402


SOURCE = [
    {"role": "system", "content": "Use tools carefully."},
    {"role": "user", "content": "Find alpha."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "alpha-result"},
    {"role": "assistant", "content": "The previous lookup is complete."},
    {"role": "user", "content": "Now answer the current question."},
]


def _config(mode):
    return {
        "mode": mode,
        "run_id": "validation-test",
        "history_budget_bytes": 100,
        "workspace_budget_bytes": 100,
    }


def _policy(decision_index=1):
    return {"decision_index": decision_index, "pre_draft_retrieval": False}


def _base_metadata(mode, *, activated, decision_index=1):
    gate_name = "auxiliary_gate" if mode == "full_exact_shared" else "capacity_gate"
    activation_name = (
        "auxiliary_activated" if mode == "full_exact_shared" else "compression_activated"
    )
    return {
        "mode": mode,
        "run_id": "validation-test",
        "task_id": "synthetic",
        "history_budget_bytes": 100,
        "workspace_budget_bytes": 100,
        "policy": _policy(decision_index),
        gate_name: {
            "full_history_bytes": 101 if activated else 100,
            activation_name: activated,
        },
    }


def _exact(status, decision_index=1):
    gap = status == "gap"
    value = {
        "version": "exact-source-gap-v1",
        "status": status,
        "decision_index": decision_index,
        "judges_action_correctness": False,
        "upgrade_count": int(gap),
        "regeneration_allowed": gap,
    }
    if gap:
        value["upgraded_event_id"] = "synthetic:m2"
    return value


def _full():
    return proxy._assemble(copy.deepcopy(SOURCE), get_arm("full"))[0]


def _no_gist_attempt(selected_ids):
    store = EventStore.from_messages("synthetic", SOURCE)
    full = _full()
    cutoff = raw_source_cutoff(SOURCE)
    raw_ids = ["synthetic:m4"]
    raw_events = [store.event(event_id) for event_id in raw_ids]
    evidence_events = [store.event(event_id) for event_id in selected_ids]
    raw_sources = {index for event in raw_events for index in event.source_indices}
    evidence_sources = {index for event in evidence_events for index in event.source_indices}
    common_positions = {
        index
        for index, message in enumerate(full)
        if index >= cutoff or message.get("role") in {"system", "developer"}
    }
    common_sources = set(common_positions)
    positions = common_positions | raw_sources
    before = [full[index] for index in sorted(positions) if index < cutoff]
    suffix = [full[index] for index in sorted(positions) if index >= cutoff]
    packet = evidence_message(store, selected_ids)
    messages = before + [packet] + suffix
    metadata = _base_metadata("capacity_exact_no_gist", activated=True)
    metadata.update({
        "gist_tokens": 0,
        "block_refs": [],
        "source_cutoff": cutoff,
        "selected_event_ids": list(selected_ids),
        "raw_history_event_ids": raw_ids,
        "raw_history_source_indices": sorted(raw_sources),
        "common_source_indices": sorted(common_sources),
        "visible_source_indices": sorted(raw_sources | common_sources | evidence_sources),
        "evidence_out_index": len(before),
    })
    return messages, metadata


def _row(mode, attempts, *, status="no_op"):
    trace = []
    forwarded = []
    gap = status == "gap"
    for index, (messages, metadata) in enumerate(attempts):
        if gap and index == len(attempts) - 1:
            metadata = copy.deepcopy(metadata)
            metadata["exact_recovery"] = _exact(status)
        trace.append({
            "phase": "draft" if index == 0 else "regeneration",
            "status": "completed",
            "discarded": gap and index == 0,
            "backend_verified": True,
            "forwarded_request_index": index,
            "memory_runtime": metadata,
        })
        forwarded.append({"messages": messages})
    final = copy.deepcopy(trace[-1]["memory_runtime"])
    final["exact_recovery"] = _exact(status)
    return {
        "eval_context": {"task_id": "synthetic", "run_id": "validation-test"},
        "request_view": {"messages": copy.deepcopy(SOURCE)},
        "forwarded_request_views": forwarded,
        "generation_trace": trace,
        "memory_runtime": final,
    }


def _full_above_row():
    store = EventStore.from_messages("synthetic", SOURCE)
    full = _full()
    selected = ["synthetic:m2"]
    packet = evidence_message(store, selected)
    metadata = _base_metadata("full_exact_shared", activated=True)
    metadata.update({
        "gist_tokens": 0,
        "block_refs": [],
        "budget_applies": False,
        "selected_event_ids": selected,
        "evidence_out_index": len(full) - 1,
        "full_raw_visible_event_ids": [event.event_id for event in store.events],
    })
    messages = full[:-1] + [packet] + full[-1:]
    return _row("full_exact_shared", [(messages, metadata)])


def test_ordinary_mode_is_ignored_without_capture_fields():
    validate_exact_request({}, {"mode": "capacity_protect"})


def test_full_exact_accepts_eventstore_packet_over_true_full_renderer():
    validate_exact_request(_full_above_row(), _config("full_exact_shared"))


@pytest.mark.parametrize("version", ["exact-source-gap-v1", "exact-source-gap-v2"])
def test_known_exact_matcher_versions_share_the_record_contract(version):
    row = _full_above_row()
    row["memory_runtime"]["exact_recovery"]["version"] = version
    validate_exact_request(row, _config("full_exact_shared"))


def test_unknown_exact_matcher_version_is_rejected():
    row = _full_above_row()
    row["memory_runtime"]["exact_recovery"]["version"] = "unknown-matcher"
    with pytest.raises(ValueError, match="version is invalid"):
        validate_exact_request(row, _config("full_exact_shared"))


def test_official_context_may_omit_run_id_but_runtime_must_bind_config():
    row = _full_above_row()
    del row["eval_context"]["run_id"]
    validate_exact_request(row, _config("full_exact_shared"))
    row["memory_runtime"]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_exact_request(row, _config("full_exact_shared"))


@pytest.mark.parametrize("wrong_run_id", [None, "another-run"])
def test_explicit_context_run_id_must_still_match_config(wrong_run_id):
    row = _full_above_row()
    row["eval_context"]["run_id"] = wrong_run_id
    with pytest.raises(ValueError, match="context run_id differs"):
        validate_exact_request(row, _config("full_exact_shared"))


def test_full_exact_rejects_cropped_source_even_when_visibility_metadata_claims_full():
    row = _full_above_row()
    row["forwarded_request_views"][0]["messages"].pop(1)
    row["generation_trace"][0]["memory_runtime"]["evidence_out_index"] -= 1
    with pytest.raises(ValueError, match="cropped or rewrote"):
        validate_exact_request(row, _config("full_exact_shared"))


def test_no_gist_accepts_reconstructed_wire_and_one_complete_event_upgrade():
    first = _no_gist_attempt(["synthetic:m1"])
    second = _no_gist_attempt(["synthetic:m1", "synthetic:m2"])
    row = _row("capacity_exact_no_gist", [first, second], status="gap")
    validate_exact_request(row, _config("capacity_exact_no_gist"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("visible_source_indices", [0, 1, 2, 3, 4, 5], "visibility metadata"),
        ("raw_history_source_indices", [], "raw event IDs"),
    ],
)
def test_no_gist_rejects_visibility_claims_not_derived_from_events(field, value, message):
    attempt = _no_gist_attempt(["synthetic:m1"])
    attempt[1][field] = value
    row = _row("capacity_exact_no_gist", [attempt])
    with pytest.raises(ValueError, match=message):
        validate_exact_request(row, _config("capacity_exact_no_gist"))


def test_no_gist_rejects_wire_that_omits_a_declared_raw_event():
    attempt = _no_gist_attempt(["synthetic:m1"])
    attempt[0].pop(1)
    row = _row("capacity_exact_no_gist", [attempt])
    with pytest.raises(ValueError, match="forwarded wire"):
        validate_exact_request(row, _config("capacity_exact_no_gist"))


def test_exact_state_machine_rejects_unmatched_gap_trace():
    first = _no_gist_attempt(["synthetic:m1"])
    row = _row("capacity_exact_no_gist", [first])
    row["memory_runtime"]["exact_recovery"] = _exact("gap")
    with pytest.raises(ValueError, match="generation count"):
        validate_exact_request(row, _config("capacity_exact_no_gist"))


def test_upgrade_must_preserve_first_evidence_and_add_one_complete_event():
    first = _no_gist_attempt(["synthetic:m1"])
    second = _no_gist_attempt(["synthetic:m2"])
    row = _row("capacity_exact_no_gist", [first, second], status="gap")
    with pytest.raises(ValueError, match="dropped first-round evidence"):
        validate_exact_request(row, _config("capacity_exact_no_gist"))


def test_below_budget_no_gist_requires_full_identity():
    full = _full()
    metadata = _base_metadata("capacity_exact_no_gist", activated=False)
    metadata.update({
        "gist_tokens": 0,
        "block_refs": [],
        "selected_source_indices": list(range(len(SOURCE))),
        "evidence_out_index": None,
    })
    row = _row("capacity_exact_no_gist", [(full, metadata)])
    validate_exact_request(row, _config("capacity_exact_no_gist"))
    row["forwarded_request_views"][0]["messages"][1]["content"] = "rewritten"
    with pytest.raises(ValueError, match="Full renderer identity"):
        validate_exact_request(row, _config("capacity_exact_no_gist"))


def test_capacity_exact_upgrade_state_machine_needs_no_control_wire_replay():
    first = _base_metadata("capacity_exact_once", activated=True)
    first["selected_event_ids"] = ["synthetic:m1"]
    second = copy.deepcopy(first)
    second["selected_event_ids"] = ["synthetic:m1", "synthetic:m2"]
    row = _row(
        "capacity_exact_once",
        [([{"role": "user", "content": "draft"}], first),
         ([{"role": "user", "content": "regenerated"}], second)],
        status="gap",
    )
    validate_exact_request(row, _config("capacity_exact_once"))
