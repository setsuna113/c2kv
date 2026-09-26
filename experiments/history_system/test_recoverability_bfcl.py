"""CPU fixtures for the opt-in live BFCL recoverability bridge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))

import recoverability
import recoverability_bfcl
import recoverability_runtime
import t02
import t02_bfcl
from test_recoverability_runtime import annotations
from test_t02_bfcl import FakeActor, FakeBindings, manifests, task
from test_t02_runtime import make_actor
from benchmarks.memory_runtime.tests.test_event_native_recovery import request


def _fake_live_state(tmp_path):
    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="recoverability-retention"
    )
    actor = FakeActor()
    policy = {"name": "frozen-C0", "version": 1}
    adapter = recoverability_bfcl.RecoverabilityBFCLAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official"
    )
    state = adapter.discover_task(environment, actor, seed=0, max_states=1)[0]
    return adapter, actor, state, policy


def _real_cpu_live_state(tmp_path):
    """Use the real controller/actor with a scripted CPU-only generator."""
    bindings = FakeBindings()
    entry, _ = task()
    entry["question"] = [request()["messages"]]
    entry["function"] = []
    ground_truth = [["unverified()"]]
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, ground_truth, bindings=bindings, namespace="recoverability-runtime"
    )
    actor, generator = make_actor(tmp_path / "actor")
    policy = {"name": "frozen-C0", "version": 1}
    adapter = recoverability_bfcl.RecoverabilityBFCLAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official"
    )

    raw = actor.hold(environment.next_payload())
    state = adapter._branchable(adapter._enrich_state(raw, environment), seed=0)
    assert state is not None
    # Each diagnostic regeneration is followed by a scripted continuation;
    # the real frozen controller fixture may spend its one recovery round.
    generator.outputs.extend(["done", "done"])
    environment_snapshot = environment.capture()
    actor_snapshot = actor.capture()
    adapter.register_state(
        state,
        environment=environment,
        actor=actor,
        environment_snapshot=environment_snapshot,
        actor_snapshot=actor_snapshot,
    )
    adapter.retain_for_diagnostic([state["state_id"]])
    snapshot = adapter.capture_state(state, frozen_policy=policy)
    restore = adapter.restore_state(snapshot, frozen_policy=policy)
    return adapter, actor, generator, state, policy, snapshot, restore


def test_retention_defers_parent_three_branch_release_until_explicit_release(tmp_path):
    adapter, actor, state, policy = _fake_live_state(tmp_path)
    adapter.retain_for_diagnostic([state["state_id"]])
    plan = t02.build_smoke_plan(
        state,
        forbidden_manifest_paths=manifests(tmp_path),
        frozen_subsequent_policy=policy,
        seed=0,
    )

    results = t02.execute_plan(plan, adapter)

    assert results["complete_branch_executions"] == 3
    assert state["state_id"] in adapter._states
    assert state["state_id"] in adapter._diagnostic_deferred_releases
    assert actor.closed is False
    assert adapter.capture_state(state, frozen_policy=policy) == results["snapshots"][0]
    adapter.release_diagnostic(state["state_id"])
    assert state["state_id"] not in adapter._states
    assert actor.closed is True


def test_review_packet_binds_official_turn_start_to_exact_snapshot(tmp_path):
    adapter, actor, _, state, _, snapshot, _ = _real_cpu_live_state(tmp_path)

    packet = adapter.review_packet_for_state(state)
    prefix = packet["prefix_validity"]
    artifact = json.loads(Path(prefix["artifact"]).read_text(encoding="utf-8"))

    assert packet["support_annotations"] == []
    assert packet["snapshot_id"] == prefix["snapshot_id"] == snapshot["snapshot_id"]
    assert prefix["component_digests"] == snapshot["component_digests"]
    assert prefix["all_previous_turns_success"] is True
    assert prefix["current_turn_step"] == 0
    assert prefix["decision_key"] == state["decision_key"]
    for key in (
        "source", "all_previous_turns_success", "current_turn_step",
        "decision_key", "snapshot_id", "component_digests",
    ):
        assert artifact[key] == prefix[key]
    assert artifact["current_turn_response_empty"] is True
    assert artifact["future_ground_truth_entered_actor_input"] is False
    assert t02._file_digest(Path(prefix["artifact"])) == prefix["artifact_sha256"]
    adapter.release_diagnostic(state["state_id"])
    assert actor.runner.generator.saved is None


def test_known_support_and_full_history_use_official_results_and_separate_cost(tmp_path):
    adapter, actor, _, state, policy, snapshot, _ = _real_cpu_live_state(tmp_path)
    reviewed = annotations(actor)
    support = adapter.prepare_support(state, reviewed)
    assert support["receipt"]["status"] == "admitted"

    known_branch = {
        "branch_id": "known_support",
        "candidate_ids": support["candidate_ids"],
        "support_receipt": support["receipt"],
        "continuation_memory_policy": "frozen_T02_policy",
    }
    known_restore = adapter.restore_state(snapshot, frozen_policy=policy)
    known = adapter.run_diagnostic(
        state, known_branch, frozen_policy=policy, restore_receipt=known_restore
    )

    full_branch = {
        "branch_id": "full_history",
        "candidate_ids": [],
        "continuation_memory_policy": recoverability.FULL_POLICY,
    }
    full_restore = adapter.restore_state(snapshot, frozen_policy=policy)
    full = adapter.run_diagnostic(
        state, full_branch, frozen_policy=policy, restore_receipt=full_restore
    )

    assert known["schema"] == full["schema"] == recoverability.BRANCH_SCHEMA
    assert known["execution_receipt"] == recoverability._branch_receipt(known_branch)
    assert full["execution_receipt"] == recoverability._branch_receipt(full_branch)
    assert known["official_outcome"]["turn_success"] is True
    assert full["official_outcome"]["task_success"] is True
    assert known["cost"]["generation_calls"] == 3
    assert full["cost"]["generation_calls"] == 2
    assert known["cost"]["diagnostic_generation_calls_total"] == 3
    assert full["cost"]["diagnostic_generation_calls_total"] == 5
    assert adapter.cost_summary()["branch_generation_calls"] == 0
    assert adapter.diagnostic_cost_summary()["branch_attempts"] == 2
    for result in (known, full):
        path = Path(result["official_outcome"]["artifact"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["snapshot_id"] == snapshot["snapshot_id"]
        assert payload["component_digests"] == snapshot["component_digests"]
        assert t02._file_digest(path) == result["official_outcome"]["artifact_sha256"]
    adapter.release_diagnostic(state["state_id"])
    assert actor.runner.generator.saved is None


def test_failed_generation_is_costed_and_same_snapshot_branch_cannot_retry(tmp_path, monkeypatch):
    adapter, actor, _, state, policy, snapshot, restore = _real_cpu_live_state(tmp_path)

    def fail_after_submission(active_actor):
        active_actor.total_generation_calls += 1
        raise RuntimeError("fixture backend failure after submission")

    monkeypatch.setattr(recoverability_runtime, "submit_full_history", fail_after_submission)
    branch = {
        "branch_id": "full_history",
        "candidate_ids": [],
        "continuation_memory_policy": recoverability.FULL_POLICY,
    }
    with pytest.raises(RuntimeError, match="fixture backend failure"):
        adapter.run_diagnostic(
            state, branch, frozen_policy=policy, restore_receipt=restore
        )

    summary = adapter.diagnostic_cost_summary()
    assert summary["branch_attempts"] == 1
    assert summary["generation_calls"] == 1
    assert summary["parent_t02_branch_generation_calls"] == 0
    retry_restore = adapter.restore_state(snapshot, frozen_policy=policy)
    with pytest.raises(ValueError, match="cannot be retried"):
        adapter.run_diagnostic(
            state, branch, frozen_policy=policy, restore_receipt=retry_restore
        )
    assert adapter.diagnostic_cost_summary()["branch_attempts"] == 1
    adapter.close()
    assert actor.runner.generator.saved is None


def test_retention_and_attempt_caps_are_hard_limits(tmp_path):
    adapter, _, state, _ = _fake_live_state(tmp_path)
    with pytest.raises(ValueError, match="at most 12"):
        adapter.retain_for_diagnostic([f"state-{index}" for index in range(13)])
    adapter.retain_for_diagnostic([state["state_id"]])
    adapter.diagnostic_branch_attempts = recoverability_bfcl.MAX_DIAGNOSTIC_ATTEMPTS
    snapshot = adapter.capture_state(state, frozen_policy=adapter.frozen_policy)
    restore = adapter.restore_state(snapshot, frozen_policy=adapter.frozen_policy)
    with pytest.raises(ValueError, match="budget exhausted"):
        adapter.run_diagnostic(
            state,
            {"branch_id": "full_history", "candidate_ids": [],
             "continuation_memory_policy": recoverability.FULL_POLICY},
            frozen_policy=adapter.frozen_policy,
            restore_receipt=restore,
        )
    adapter.close()
