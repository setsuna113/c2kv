"""CPU contracts for the independent RACER v4 retrieval query switch."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import c1_v2_fields
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.racer.config import BACKENDS
from benchmarks.memory_runtime.recovery.source import select_source_event
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_static_extensions import proof_payload
from benchmarks.memory_runtime.tests.test_verified_binding import _call


def _prepared_source(rows, *, excluded=()):
    store = EventStore.from_messages("query-fixture", rows)
    return SimpleNamespace(
        _store=store,
        memory=SimpleNamespace(view=SimpleNamespace(
            raw_event_ids=tuple(excluded), mandatory_raw_event_ids=tuple(excluded))),
        metadata={"eligible_extraction": {"eligible_event_ids": [
            event.event_id for event in store.events]}})


def _call_with(name, arguments):
    return [{"id": "held", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}]


@pytest.mark.parametrize("draft_text,calls,needle", [
    ("violetmonsoon", [], "violetmonsoon"),
    ("", _call_with("lookup_citrine", {}), "lookup_citrine"),
    ("", _call_with("lookup", {"key": "cobalt-788"}), "cobalt-788"),
])
def test_off_removes_draft_text_tool_name_and_argument_from_query(draft_text, calls, needle):
    prepared = _prepared_source([
        {"role": "user", "content": f"Archived {needle}."},
        {"role": "user", "content": "Proceed."},
    ])
    target = prepared._store.events[0].event_id
    kwargs = dict(draft_text=draft_text, include_latest_complete_observation=True,
                  explicit_revision_abstain=False, allow_empty_draft_query=True)
    on, on_receipt = select_source_event(prepared, calls, include_draft=True, **kwargs)
    off, off_receipt = select_source_event(prepared, calls, include_draft=False, **kwargs)
    assert on == target and on_receipt["ranked_candidate_event_ids"] == [target]
    assert off is None and off_receipt["ranked_candidate_event_ids"] == []
    assert on_receipt["draft_in_retrieval_query"] is True
    assert off_receipt["draft_in_retrieval_query"] is False
    assert needle not in json.dumps(off_receipt)


def test_off_removes_exact_argument_anchor_but_keeps_goal_and_latest_observation():
    prepared = _prepared_source([
        {"role": "user", "content": "Archive SAPPHIRE 42."},
        {"role": "user", "content": "Archive 42 SAPPHIRE."},
        {"role": "user", "content": "Find SAPPHIRE archive."},
    ])
    older, newer = (event.event_id for event in prepared._store.events[:2])
    calls = _call_with("lookup", {"key": "SAPPHIRE 42"})
    kwargs = dict(draft_text="", explicit_revision_abstain=False,
                  allow_empty_draft_query=True)
    on, _ = select_source_event(prepared, calls, include_draft=True, **kwargs)
    off, off_receipt = select_source_event(prepared, calls, include_draft=False, **kwargs)
    assert on == older
    assert off == newer
    assert off_receipt["ranked_candidate_event_ids"] == [newer, older]

    latest = _prepared_source([
        {"role": "user", "content": "Archived lilac-handoff."},
        {"role": "assistant", "content": None, "tool_calls": _call_with("status", {})},
        {"role": "tool", "tool_call_id": "held", "content": "lilac-handoff"},
        {"role": "user", "content": "Proceed."},
    ])
    events = latest._store.events
    latest.memory.view.raw_event_ids = (events[1].event_id,)
    latest.memory.view.mandatory_raw_event_ids = (events[1].event_id,)
    selected, receipt = select_source_event(
        latest, [], draft_text="unrelated-draft", include_draft=False,
        include_latest_complete_observation=True, explicit_revision_abstain=False,
        allow_empty_draft_query=True)
    assert selected == events[0].event_id
    assert receipt["latest_complete_observation_in_query"] is True
    assert receipt["latest_complete_observation_event_id"] == events[1].event_id


def _controller(backend, retrieval_draft, *, extra="off", declared=True):
    racer_backend = {
        "schema": "racer-backend-v4", "backend": backend,
        "policy": "c1_v2_verified", "history_budget_tokens": 4096,
        "detector_calibration": ("reference" if backend == "c2kv" else
                                 "frozen_c2kv_unvalidated_transfer"),
        "allocation": "c2kv_s0" if backend == "c2kv" else "backend_native_persistent",
        "extra_protection": extra,
    }
    if declared:
        racer_backend["retrieval_draft"] = retrieval_draft
    candidate = {"variant": "c1_v2_verified", "risk_threshold": 0.5,
                 "risk_artifact": "fixture", **c1_v2_fields("c1_v2_verified")}
    config = {**S0_CONFIG_DEFAULTS, "racer_backend": racer_backend,
              "candidate_algorithm": candidate}
    return build_event_native_controller(
        Tokenizer(), packing={**packing(), "ratios": [8]}, policy=policy(4096),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=config, model_context=100_000, benchmark="bfcl")


def _prepare(controller, payload=None):
    return controller.prepare(payload or {
        "session_id": "query-parity", "decision_key": "turn-0/step-0",
        "messages": messages(), "tools": []}, ratio=8, max_new_tokens=32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_v4_initial_memory_and_default_receipt_identity_are_independent_of_query(backend, monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1))
    default, on, off = (_controller(backend, "on", declared=False),
                        _controller(backend, "on"), _controller(backend, "off"))
    original, included, excluded = (_prepare(default), _prepare(on), _prepare(off))
    assert original.memory == included.memory == excluded.memory
    assert original.eligible_chunks == included.eligible_chunks == excluded.eligible_chunks
    receipts = [prepared.metadata["racer_backend"] for prepared in
                (original, included, excluded)]
    assert receipts[0] == receipts[1]
    assert "retrieval_draft" not in receipts[0]
    assert receipts[2]["retrieval_draft"] == "off"
    assert receipts[2]["identity"] == receipts[0]["identity"].replace(
        ":b4096", ":retrieval_draft_off:b4096")


class _RecordingRisk(Risk):
    def __init__(self, score):
        super().__init__(score)
        self.contexts = []

    def predict_risk(self, context):
        self.contexts.append(copy.deepcopy(context))
        return super().predict_risk(context)


@pytest.mark.parametrize("backend,extra", [("c2kv", "off"), ("h2o", "off"),
                                            ("h2o", "on")])
def test_query_switch_preserves_detector_input_and_verified_commit(
    backend, extra, monkeypatch,
):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1))
    controls = {mode: _controller(backend, mode, extra=extra) for mode in ("on", "off")}
    risks = {mode: _RecordingRisk(0.1) for mode in controls}
    for mode, control in controls.items():
        control.inner.risk_model = risks[mode]
    prepared = {mode: _prepare(control, proof_payload()) for mode, control in controls.items()}
    assert prepared["on"].memory == prepared["off"].memory
    draft = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    for mode, control in controls.items():
        control.observe_selection_draft(
            session_id=prepared[mode]._store.session_id,
            decision_key=prepared[mode].metadata["decision_key"],
            token_logprobs=(-0.2, -0.4))
    results = {mode: control.reconsider(prepared[mode], draft, draft_text="call")
               for mode, control in controls.items()}
    assert risks["on"].contexts == risks["off"].contexts
    assert risks["on"].contexts[0]["draft_tool_calls"] == draft
    assert risks["on"].contexts[0]["draft_text"] == "call"
    assert risks["on"].contexts[0]["draft_logprobs"] == [-0.2, -0.4]
    assert all(not result["regenerate"] for result in results.values())
    for mode, result in results.items():
        assert result["metadata"]["racer_backend"] == controls[mode].backend.receipt()
        assert result["decision"]["racer_backend"] == controls[mode].backend.receipt()
    assert results["on"]["decision"]["verified_binding"] == results["off"]["decision"]["verified_binding"]
    finalized = {}
    for mode, control in controls.items():
        control.validate_commit(prepared[mode], draft, draft_text="call")
        finalized[mode] = control.finalize_commit(prepared[mode], draft)
    assert finalized["on"] == finalized["off"]
    changed, receipt = finalized["off"]
    assert receipt["changed"] and receipt["additional_generations"] == 0
    assert json.loads(changed[0]["function"]["arguments"])["access_token"] == "ABCDE12345"


@pytest.mark.parametrize("backend,extra", [("c2kv", "off"), ("h2o", "off"),
                                            ("h2o", "on")])
def test_real_controller_source_receipt_reflects_query_mode(backend, extra, monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.9))
    rows = messages()
    rows[1]["content"] = "Archived violetmonsoon."
    rows[4]["content"] = "Proceed."
    payload = {"session_id": "query-routing", "decision_key": "turn-0/step-0",
               "messages": rows, "tools": []}
    sources = {}
    for mode in ("default", "on", "off"):
        control = _controller(backend, "on" if mode == "default" else mode,
                              extra=extra, declared=mode != "default")
        prepared = _prepare(control, payload)
        target = next(event.event_id for event in prepared._store.events
                      if 1 in event.source_indices)
        result = control.reconsider(prepared, [], draft_text="violetmonsoon")
        assert result["metadata"]["racer_backend"] == control.backend.receipt()
        assert result["decision"]["gate"]["triggered"] is True
        source = result["decision"]["source"]
        assert source["draft_in_retrieval_query"] is (mode != "off")
        assert source["latest_complete_observation_in_query"] is True
        sources[mode] = source
    assert sources["default"] == sources["on"]
    assert target in sources["on"]["ranked_candidate_event_ids"]
    assert target not in sources["off"]["ranked_candidate_event_ids"]
