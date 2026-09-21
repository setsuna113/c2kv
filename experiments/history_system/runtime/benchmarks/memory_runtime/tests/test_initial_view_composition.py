"""Static initial allocation composed with the existing Goal/Pending recovery."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms import (
    GOAL_VERSION, INITIAL_VIEW_VERSION, VERSION,
)
from benchmarks.memory_runtime.candidate_algorithms.allocation import CandidateAllocator
from benchmarks.memory_runtime.candidate_algorithms.initial_view import (
    STATIC_INITIAL_VIEW, validate_initial_view_config,
)
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_server import _candidate_ready_contract
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer, messages, packing, policy, tool_pair,
)
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.budget_guard import history_budget_receipt


def candidate(variant):
    return {
        "variant": variant,
        "risk_artifact": {"fixture": True},
        "risk_threshold": 0.5,
        "recovery_backbone": {
            "goal_static": "goal_rescue", "pending_static": "goal_pending",
        }[variant],
        "initial_view": copy.deepcopy(STATIC_INITIAL_VIEW),
    }


def controller(monkeypatch, variant, *, score=0.1):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(score),
    )
    geometry = packing()
    geometry["ratios"] = [8]
    return build_event_native_controller(
        Tokenizer(), packing=geometry, policy=policy(4096),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**S0_CONFIG_DEFAULTS, "candidate_algorithm": candidate(variant)},
    )


def request(session="static-view", key="d1", rows=None):
    return {
        "session_id": session, "decision_key": key,
        "messages": copy.deepcopy(rows if rows is not None else messages()),
        "tools": [],
    }


@pytest.mark.parametrize("variant", ["goal_static", "pending_static"])
def test_first_view_matches_unchanged_static_allocator_on_each_prefix(monkeypatch, variant):
    composed = controller(monkeypatch, variant)
    geometry = packing()
    geometry["ratios"] = [8]
    baseline = CandidateAllocator(
        Tokenizer(), packing=geometry, policy=policy(4096), variant="static_t02"
    )
    first = request()
    second = request(
        key="d2",
        rows=[*first["messages"], {"role": "user", "content": "Check again."},
              *tool_pair(4, result={"value": "new"})],
    )
    for payload in (first, second):
        prepared = composed.prepare(payload, ratio=8, max_new_tokens=32)
        static = baseline.prepare(payload, ratio=8, max_new_tokens=32)
        assert prepared.memory == static.memory
        assert prepared.memory.view.raw_event_ids == static.memory.view.raw_event_ids
        assert prepared.memory.view.gist_event_ids == static.memory.view.gist_event_ids
        assert not (set(prepared.memory.view.raw_event_ids)
                    & set(prepared.memory.view.gist_event_ids))
        assert prepared.metadata["lease_decisions"] == 0
        assert prepared.metadata["candidate_algorithm"]["variant"] == variant
        assert prepared.metadata["candidate_algorithm"]["initial_view"] == STATIC_INITIAL_VIEW
        assert prepared.metadata["route"]["baseline_identity"] == (
            f"{INITIAL_VIEW_VERSION}:{variant}"
        )
        assert history_budget_receipt(
            prepared.memory, prepared.metadata, composed, ratio=8, phase="draft"
        )["status"] == "passed"


@pytest.mark.parametrize("variant", ["goal_static", "pending_static"])
def test_goal_review_and_budgeted_recovery_keep_public_identity(monkeypatch, variant):
    composed = controller(monkeypatch, variant)
    rows = [
        {"role": "user", "content": "Find the price, then buy one item."},
        *tool_pair(1, result={"price": 12}),
    ]
    prepared = composed.prepare(request(rows=rows), ratio=8, max_new_tokens=32)
    result = composed.reconsider(prepared, [], draft_text="The price is 12.")
    assert result["regenerate"]
    assert result["decision"]["reason"] == "stop_completion_review"
    assert result["decision"]["version"] == INITIAL_VIEW_VERSION
    assert result["decision"]["variant"] == variant
    assert result["decision"]["recovery_backbone"] == candidate(variant)["recovery_backbone"]
    assert result["decision"]["backbone_decision_version"] == (
        GOAL_VERSION if variant == "pending_static" else VERSION
    )
    assert result["metadata"]["exact_recovery"] == result["decision"]
    assert result["metadata"]["candidate_algorithm"]["variant"] == variant
    assert result["metadata"]["route"]["baseline_identity"] == (
        f"{INITIAL_VIEW_VERSION}:{variant}"
    )
    assert not (set(result["memory"].view.raw_event_ids)
                & set(result["memory"].view.gist_event_ids))
    assert history_budget_receipt(
        result["memory"], result["metadata"], composed,
        ratio=8, phase="regeneration",
    )["status"] == "passed"
    assert composed.reconsider(prepared, [], draft_text="The price is 12.") == result
    if variant == "pending_static":
        assert result["decision"]["goal_review"]["composition"] == "pending_receipts"
        assert composed.validate_commit(
            prepared, [], draft_text="The price is 12."
        )["accepted"]
        calls, receipt = composed.finalize_commit(prepared, [])
        assert calls == ()
        assert receipt["variant"] == "goal_pending"


@pytest.mark.parametrize("variant", ["goal_static", "pending_static"])
def test_ready_contract_records_both_modules(variant):
    identity, baseline = _candidate_ready_contract(candidate(variant))
    assert identity["variant"] == variant
    assert identity["initial_view"] == STATIC_INITIAL_VIEW
    assert identity["recovery_backbone"] == candidate(variant)["recovery_backbone"]
    assert baseline == f"{INITIAL_VIEW_VERSION}:{variant}"


def test_legacy_ready_identity_is_unchanged():
    assert _candidate_ready_contract({"variant": "goal_rescue"}) == (
        {"variant": "goal_rescue", "stable_call_ids": True,
         "recovery_rounds_per_decision": 1},
        f"{VERSION}:goal_rescue",
    )
    assert _candidate_ready_contract({"variant": "goal_pending"})[1] == (
        f"{GOAL_VERSION}:goal_pending"
    )


@pytest.mark.parametrize("variant", ["goal_static", "pending_static"])
def test_frozen_controller_config_constructs_static_composition(monkeypatch, variant):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1),
    )
    frozen = json.loads((ROOT / "configs" / "controller.json").read_text(encoding="utf-8"))
    assert "observed_entity_slot_policy" in frozen
    for key in ("gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"):
        frozen.pop(key, None)
    geometry = packing()
    geometry["ratios"] = [8]
    new_config = {**frozen, "candidate_algorithm": candidate(variant)}
    old_config = {**frozen, "candidate_algorithm": {
        "variant": "static_t02", "risk_artifact": {"fixture": True},
        "risk_threshold": 0.5,
    }}
    def build(config):
        return build_event_native_controller(
            Tokenizer(), packing=geometry, policy=policy(4096),
            view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
            s0_config=config,
        )
    composed = build(new_config)
    static = build(old_config)
    payload = request(session=f"frozen-{variant}")
    prepared = composed.prepare(payload, ratio=8, max_new_tokens=32)
    reference = static.prepare(payload, ratio=8, max_new_tokens=32)
    assert prepared.memory == reference.memory
    assert composed.base.s0_config == static.base.s0_config
    assert set(composed.base.s0_config) == set(S0_CONFIG_DEFAULTS)
    assert prepared.metadata["candidate_algorithm"]["variant"] == variant


def test_missing_or_mismatched_composition_is_rejected(monkeypatch):
    for variant in ("goal_static", "pending_static"):
        good = candidate(variant)
        for field, bad in (
            ("recovery_backbone", "goal_pending" if variant == "goal_static" else "goal_rescue"),
            ("initial_view", {"policy": "s0", "version": "legacy"}),
            ("risk_threshold", 0.6),
        ):
            invalid = {**good, field: bad}
            with pytest.raises(ValueError):
                validate_initial_view_config(invalid)
        for field in ("recovery_backbone", "initial_view", "risk_artifact", "risk_threshold"):
            invalid = {key: value for key, value in good.items() if key != field}
            with pytest.raises(ValueError):
                validate_initial_view_config(invalid)

    geometry = packing()
    geometry["ratios"] = [8]
    with pytest.raises(ValueError, match="do not accept initial_view"):
        build_event_native_controller(
            Tokenizer(), packing=geometry, policy=policy(4096),
            view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
            s0_config={**S0_CONFIG_DEFAULTS, "candidate_algorithm": {
                "variant": "goal_rescue", "initial_view": STATIC_INITIAL_VIEW,
            }},
        )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1),
    )
    legacy = build_event_native_controller(
        Tokenizer(), packing=geometry, policy=policy(4096),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**S0_CONFIG_DEFAULTS, "candidate_algorithm": {
            "variant": "goal_rescue", "risk_artifact": {"fixture": True},
        }},
    )
    prepared = legacy.prepare(request(session="legacy"), ratio=8, max_new_tokens=32)
    assert prepared.metadata["candidate_algorithm"]["variant"] == "goal_rescue"
    assert prepared.metadata["candidate_algorithm"]["schema"] == VERSION
    assert prepared.metadata["route"]["baseline_identity"] == (
        f"{VERSION}:goal_rescue"
    )
