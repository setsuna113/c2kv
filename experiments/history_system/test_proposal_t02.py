"""CPU contracts for H0 proposal-matched T02."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))

import proposal_t02 as proposal
import proposal_t02_bfcl
import t02


def _state(index: int, *, source_mode: str = "distinct") -> dict:
    first, second = f"u{index}-0", f"u{index}-1"
    parameter = f"secret-{index}"
    q = {
        "schema": "recovery-selection-context-v1",
        "session_id": f"bfcl/multi_turn_base_{index}/attempt-0",
        "decision_key": "turn-0/step-0",
        "goal": "do the task",
        "raw_visible": [],
        "draft_text": "call",
        "draft_tool_calls": [{"function": {"name": "lookup", "arguments": {"id": parameter}}}],
        "parse_ok": True,
        "is_stop": False,
    }
    if source_mode == "empty":
        q["draft_tool_calls"] = []
        q["is_stop"] = True
    first_text = parameter if source_mode == "same" else "unrelated first"
    second_text = parameter if source_mode == "distinct" else "unrelated second"
    return {
        "schema": t02.CANDIDATE_SCHEMA,
        "state_id": f"state-{index}",
        "benchmark": "bfcl",
        "task_id": f"multi_turn_base_{index}",
        "task_group_id": f"bfcl_pair_{index}",
        "decision_key": "turn-0/step-0",
        "draft_kind": "call",
        "risk_bucket": "unknown",
        "previous_turn_valid": True,
        "q": q,
        "draft": {"parse_ok": True, "text": "call", "tool_calls": q["draft_tool_calls"]},
        "candidates": [
            {"candidate_id": first, "unit_id": first, "source_id": f"e{index}-0",
             "event_id": f"e{index}-0", "rank": 0, "feasible": True,
             "text": first_text, "text_sha256": hashlib.sha256(first_text.encode()).hexdigest(),
             "token_count": 2, "source_indices": [0],
             "provenance": [{"event_id": f"e{index}-0", "source_index": 0,
                             "container_path": [], "field_path": [], "char_range": [0, len(first_text)],
                             "sha256": hashlib.sha256(first_text.encode()).hexdigest()}]},
            {"candidate_id": second, "unit_id": second, "source_id": f"e{index}-1",
             "event_id": f"e{index}-1", "rank": 1, "feasible": True,
             "text": second_text, "text_sha256": hashlib.sha256(second_text.encode()).hexdigest(),
             "token_count": 3, "source_indices": [1],
             "provenance": [{"event_id": f"e{index}-1", "source_index": 1,
                             "container_path": [], "field_path": [], "char_range": [0, len(second_text)],
                             "sha256": hashlib.sha256(second_text.encode()).hexdigest()}]},
        ],
        "allowed_actions": [
            {"action_id": 0, "candidate_ids": []},
            {"action_id": 1, "candidate_ids": [first]},
            {"action_id": 2, "candidate_ids": [second]},
            {"action_id": 3, "candidate_ids": [first, second]},
        ],
    }


def _manifests(tmp_path: Path, *, forbidden: str = "unused") -> dict[str, Path]:
    d128, f128 = tmp_path / "d128.json", tmp_path / "f128.json"
    d128.write_text(json.dumps({"task_ids": [forbidden]}), encoding="utf-8")
    f128.write_text(json.dumps({"task_ids": ["also-unused"]}), encoding="utf-8")
    return {"D128": d128, "F128": f128}


def _policy() -> dict:
    return {"name": "old-H0-C0", "history_profile": "H0",
            "sampling": {"mode": "greedy", "temperature": 0, "seed": 0}}


def _official(state_id: str, branch_id: str, turn: bool, task: bool) -> dict:
    return {
        "source": "official", "scorer": "BFCL_v4_multi_turn_checker",
        "artifact": f"official/{state_id}/{branch_id}.json",
        "artifact_sha256": hashlib.sha256(f"{state_id}:{branch_id}".encode()).hexdigest(),
        "turn_success": turn, "task_success": task,
    }


def test_proposals_are_exact_deduplicated_and_empty_source_is_omitted():
    distinct, receipt = proposal.proposal_branches(t02._normalize_candidate_state(_state(1)))
    assert [row["branch_id"] for row in distinct] == ["A0", "A1lex", "A2src"]
    assert distinct[1]["candidate_ids"] == ["u1-0"]
    assert distinct[2]["candidate_ids"] == ["u1-1"]
    assert receipt["distinct"] is True

    same, receipt = proposal.proposal_branches(t02._normalize_candidate_state(_state(2, source_mode="same")))
    assert [row["branch_id"] for row in same] == ["A0", "A1lex"]
    assert same[1]["proposal_origins"] == ["Slex", "Ssrc"]
    assert receipt["distinct"] is False

    empty, receipt = proposal.proposal_branches(t02._normalize_candidate_state(_state(3, source_mode="empty")))
    assert [row["branch_id"] for row in empty] == ["A0", "A1lex"]
    assert receipt["fallback"] == {"applied": True, "from": "Ssrc", "to": "Slex",
                                   "reason": "source_proposal_empty"}


def test_plan_enforces_60_180_budget_split_locks_and_union_group_exclusion(tmp_path):
    states = [_state(index, source_mode="distinct") for index in range(61)]
    locks = {"bfcl_pair_0": "calibration", "bfcl_pair_1": "train"}
    plan = proposal.build_plan(
        states, forbidden_manifest_paths=_manifests(tmp_path, forbidden="multi_turn_long_context_60"),
        frozen_subsequent_policy=_policy(), target_states=60, train_states=40,
        split_bindings=locks,
    )
    receipt = proposal.check_plan(plan)
    assert receipt["state_count"] == 60
    assert receipt["continuation_count"] == 180
    by_group = {row["task_group_id"]: row["split"] for row in plan["states"]}
    assert by_group["bfcl_pair_0"] == "calibration"
    assert by_group["bfcl_pair_1"] == "train"
    assert "bfcl_pair_60" not in by_group
    assert plan["old_359_360_budget_affected"] is False
    with pytest.raises(proposal.ProposalT02Error, match="\[1, 60\]"):
        proposal.build_plan(
            states, forbidden_manifest_paths=_manifests(tmp_path),
            frozen_subsequent_policy=_policy(), target_states=61,
        )


class FakeAdapter:
    def __init__(self):
        self.active = None
        self.finished = []

    def capabilities(self):
        return {"schema": t02.CAPABILITY_SCHEMA, "exact_same_state_restore": True,
                "components": list(t02.REQUIRED_COMPONENTS), "frozen_subsequent_policy": True,
                "official_turn_outcome": True, "official_task_outcome": True}

    def capture_state(self, state, *, frozen_policy):
        self.active = state["state_id"]
        return {"schema": t02.SNAPSHOT_SCHEMA, "state_id": state["state_id"],
                "snapshot_id": "snapshot-" + state["state_id"],
                "frozen_policy_sha256": t02._digest(frozen_policy),
                "component_digests": {name: hashlib.sha256(
                    f"{state['state_id']}:{name}".encode()).hexdigest()
                    for name in t02.REQUIRED_COMPONENTS}}

    def restore_state(self, snapshot, *, frozen_policy):
        return {**copy.deepcopy(snapshot), "schema": t02.RESTORE_SCHEMA, "restored": True}

    def run_branch(self, state, branch, *, frozen_policy, restore_receipt):
        success = branch["branch_id"] != "A0"
        return {"schema": t02.RESULT_SCHEMA, "state_id": state["state_id"],
                "branch_id": branch["branch_id"], "candidate_ids": branch["candidate_ids"],
                "snapshot_id": restore_receipt["snapshot_id"],
                "restore_receipt_sha256": t02._digest(restore_receipt),
                "frozen_policy_sha256": t02._digest(frozen_policy),
                "execution_status": "complete",
                "execution_receipt": {"submitted_original_draft": branch["submit_original_draft"],
                    "recovery_candidate_ids": branch["candidate_ids"],
                    "regeneration_count": 1 if branch["regenerate"] else 0,
                    "continued_with_frozen_policy": True,
                    "observations_replayed_from_other_branch": False},
                "official_outcome": _official(state["state_id"], branch["branch_id"], success, success)}

    def finish_state(self, state_id, *, completed_branches):
        assert self.active == state_id
        self.finished.append((state_id, completed_branches))
        self.active = None


def test_execute_label_and_tamper_checks_variable_branch_count(tmp_path):
    states = [_state(1, source_mode="distinct"), _state(2, source_mode="same")]
    plan = proposal.build_plan(
        states, forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy=_policy(), target_states=2, train_states=1,
    )
    adapter = FakeAdapter()
    results = proposal.execute_plan(plan, adapter)
    assert results["complete_continuation_executions"] == 5
    assert sorted(count for _, count in adapter.finished) == [2, 3]
    labels = proposal.label_results(plan, results)
    assert labels["action_example_count"] == 3
    assert proposal.check_labeled(labels)["state_count"] == 2
    tampered = copy.deepcopy(results)
    tampered["results"][0]["candidate_ids"] = ["invented"]
    with pytest.raises(proposal.ProposalT02Error, match="snapshot, policy, action"):
        proposal.label_results(plan, tampered)
    extra = copy.deepcopy(results)
    extra["results"].append(copy.deepcopy(extra["results"][0]))
    extra["complete_continuation_executions"] += 1
    with pytest.raises(proposal.ProposalT02Error, match="partial or contain extra"):
        proposal.label_results(plan, extra)


def _old_row(state: dict) -> dict:
    normalized = t02._normalize_candidate_state(state)
    proposal_branches, _ = proposal.proposal_branches(normalized)
    old_branches, outcomes, labels = [], {}, {}
    for index, new in enumerate(proposal_branches):
        old_id = ("A0", "A1", "A2")[index]
        branch = {**copy.deepcopy(new), "branch_id": old_id}
        old_branches.append(branch)
        success = old_id != "A0"
        official = _official(state["state_id"], old_id, success, success)
        outcomes[old_id] = {"execution_status": "complete", "turn_success": success,
                            "task_success": success, "official_outcome": official}
        if old_id != "A0":
            labels[old_id] = {"candidate_ids": branch["candidate_ids"],
                              "delta_turn": 1, "delta_task": 1,
                              "turn_label_status": "known", "task_label_status": "known"}
    return {"schema": t02.LABELED_STATE_SCHEMA, **copy.deepcopy(normalized),
            "split": "train", "branches": old_branches, "outcomes": outcomes,
            "labels": labels, "untested_actions": [], "statistical_unit": "state",
            "branches_are_independent_states": False}


def _binding() -> dict:
    return {"schema": proposal.SOURCE_BINDING_SCHEMA,
            "files": {"labels": {"sha256": "a" * 64}},
            "labels_canonical_sha256": "b" * 64}


def test_reuse_requires_exact_observation_source_action_and_continuation(tmp_path):
    state = _state(7, source_mode="distinct")
    old = _old_row(state)
    labels = {"rows": [old]}
    reused = proposal.reuse_old_labels(
        labels, binding=_binding(), source_policy=_policy(), target_policy=_policy(),
        forbidden_manifest_paths=_manifests(tmp_path), target_states=[state],
    )
    assert reused["state_count"] == 1
    assert reused["new_continuation_executions"] == 0
    assert {row["branch_id"] for row in reused["rows"][0]["proposal_actions"]} == {"A1lex", "A2src"}

    changed = copy.deepcopy(state)
    changed["q"]["goal"] = "different"
    rejected = proposal.reuse_old_labels(
        labels, binding=_binding(), source_policy=_policy(), target_policy=_policy(),
        forbidden_manifest_paths=_manifests(tmp_path), target_states=[changed],
    )
    assert rejected["state_count"] == 0
    assert "q/d snapshot" in rejected["rejected"][0]["reason"]

    changed = copy.deepcopy(state)
    changed["candidates"][1]["provenance"][0]["char_range"] = [1, 4]
    rejected = proposal.reuse_old_labels(
        labels, binding=_binding(), source_policy=_policy(), target_policy=_policy(),
        forbidden_manifest_paths=_manifests(tmp_path), target_states=[changed],
    )
    assert rejected["state_count"] == 0

    other_policy = {**_policy(), "sampling": {"mode": "sample", "temperature": 0.7, "seed": 0}}
    with pytest.raises(proposal.ProposalT02Error, match="contracts differ"):
        proposal.reuse_old_labels(
            labels, binding=_binding(), source_policy=_policy(), target_policy=other_policy,
            forbidden_manifest_paths=_manifests(tmp_path), target_states=[state],
        )


def test_reuse_rejects_untested_proposal_and_group_leakage(tmp_path):
    state = _state(9, source_mode="distinct")
    old = _old_row(state)
    old["branches"].pop()
    old["outcomes"].pop("A2")
    old["labels"].pop("A2")
    rejected = proposal.reuse_old_labels(
        {"rows": [old]}, binding=_binding(), source_policy=_policy(), target_policy=_policy(),
        forbidden_manifest_paths=_manifests(tmp_path), target_states=[state],
    )
    assert rejected["state_count"] == 0
    assert "not actually tested" in rejected["rejected"][0]["reason"]

    forbidden = proposal.reuse_old_labels(
        {"rows": [_old_row(state)]}, binding=_binding(), source_policy=_policy(), target_policy=_policy(),
        forbidden_manifest_paths=_manifests(tmp_path, forbidden="multi_turn_long_context_9"),
        target_states=[state],
    )
    assert forbidden["state_count"] == 0
    assert "D128/F128" in forbidden["rejected"][0]["reason"]


def test_live_cli_defaults_source_acquisition_to_zero(tmp_path):
    required = [
        "--bfcl-root", str(tmp_path), "--task-manifest", str(tmp_path / "tasks.json"),
        "--family-audit", str(tmp_path / "audit.json"), "--d128-manifest", str(tmp_path / "d.json"),
        "--f128-manifest", str(tmp_path / "f.json"), "--design", str(tmp_path / "design.json"),
        "--source-plan", str(tmp_path / "plan.json"), "--checkpoint", str(tmp_path / "ckpt"),
        "--backend-url", "http://127.0.0.1:30000", "--out", str(tmp_path / "out"),
        "--resource-coordination-approved",
    ]
    with pytest.raises(ValueError, match="defaults to zero"):
        proposal_t02_bfcl.main(required)
