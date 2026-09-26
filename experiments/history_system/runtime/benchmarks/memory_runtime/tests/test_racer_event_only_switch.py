"""The event-only switch removes argument correction and nothing else from C1 v2."""
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import (
    ARGUMENT_CORRECTION_ENV, c1_v2_fields,
)
from benchmarks.memory_runtime.candidate_algorithms.verified_commit import CoreCommitPolicy
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_static_extensions import proof_payload
from benchmarks.memory_runtime.tests.test_verified_binding import _call

SESSION = "bfcl/multi_turn_base_7/attempt-0"


def factory(monkeypatch, correction, *, score=0.1, budget=4096):
    monkeypatch.setenv(ARGUMENT_CORRECTION_ENV, correction)
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
                        lambda artifact: Risk(score))
    runtime = Path(__file__).resolve().parents[3]
    frozen = json.loads((runtime / "configs/controller.json").read_text(encoding="utf-8"))
    frozen.pop("post_draft_recovery")
    candidate = {"variant": "c1_v2_verified", "risk_artifact": {"fixture": True},
                 "risk_threshold": 0.5, **c1_v2_fields("c1_v2_verified")}
    return build_event_native_controller(
        Tokenizer(), packing={**packing(), "ratios": [8]}, policy=policy(budget),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**frozen, "candidate_algorithm": candidate})


def history_rows():
    rows = [{"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Look up the order totals for keys 1 through 6."}]
    for number in range(1, 7):
        rows += tool_pair(number, result={"key": number, "total": 1000 + number,
                                          "note": "archived order record " * 20})
    rows.append({"role": "user", "content": "Now refund the order with key 2."})
    return rows


def triggered(monkeypatch, correction):
    c = factory(monkeypatch, correction, score=0.9, budget=600)
    prepared = c.prepare({"session_id": SESSION, "decision_key": "d1", "messages": history_rows(),
                          "tools": []}, ratio=8, max_new_tokens=32)
    calls = [_call("lookup", {"key": 2})]
    return c, prepared, c.reconsider(prepared, calls, draft_text="lookup key 2")


def test_switch_rejects_unknown_values(monkeypatch):
    with pytest.raises(ValueError, match=ARGUMENT_CORRECTION_ENV):
        factory(monkeypatch, "false")


def test_off_keeps_allocation_and_parse_fallback_but_never_corrects(monkeypatch):
    calls = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    outputs, memories = [], []
    for correction in ("on", "off"):
        c = factory(monkeypatch, correction)
        p = c.prepare(proof_payload(), ratio=8, max_new_tokens=32)
        memories.append((p.memory, p.eligible_chunks))
        assert not c.reconsider(p, calls, draft_text="call")["regenerate"]
        assert c.validate_commit(p, calls, draft_text="call")["accepted"]
        outputs.append(c.finalize_commit(p, calls))
    assert memories[0] == memories[1]
    (_, full_receipt), (off_output, off_receipt) = outputs
    assert full_receipt["changed"]
    assert list(off_output) == calls and off_receipt["changed"] is False
    assert off_receipt["status"] == CoreCommitPolicy.STATUS
    p = c.prepare(proof_payload() | {"decision_key": "d2"}, ratio=8, max_new_tokens=32)
    c.reconsider(p, calls, draft_text="call")
    verdict = c.validate_commit(p, calls, draft_text="call", parse_error="bad json")
    assert verdict["accepted"] is False and verdict["reason"] == "malformed_selected_draft"


def test_off_recovery_decision_matches_the_full_system(monkeypatch):
    _, full_prepared, full = triggered(monkeypatch, "on")
    _, off_prepared, off = triggered(monkeypatch, "off")
    assert off["regenerate"] and off["memory"] == full["memory"]
    for key in ("reason", "gate", "source", "candidate_event_id", "candidate_trials"):
        assert off["decision"][key] == full["decision"][key]
    assert off["decision"]["verified_binding"]["status"] == CoreCommitPolicy.STATUS
    assert full_prepared.metadata["candidate_algorithm"] == off_prepared.metadata["candidate_algorithm"]
