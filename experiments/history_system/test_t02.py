"""CPU-only contracts for the bounded T02 offline collection pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(RUNTIME / "python"))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(HERE))
import t02


def _sha(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state(index, *, risk=None, draft_kind=None, pair=None):
    task = f"train_task_{index // 2}"
    first = f"candidate-{index}-0"
    second = f"candidate-{index}-1"
    if risk is None:
        risk = "high" if index % 2 else "low"
    if draft_kind is None:
        draft_kind = "call" if index % 2 else "stop"
    if pair is None:
        pair = index % 3 == 0
    return {
        "schema": t02.CANDIDATE_SCHEMA,
        "state_id": f"state-{index}",
        "benchmark": "bfcl",
        "task_id": task,
        "task_group_id": task,
        "decision_key": f"turn-{index}/step-{index}",
        "set_selector": "local_llm",
        "draft_kind": draft_kind,
        "risk_bucket": risk,
        "previous_turn_valid": True if index < 4 else None,
        "q": {"task": f"do {index}", "recent_feedback": "observed"},
        "draft": {
            "kind": draft_kind,
            "text": "" if draft_kind == "call" else "done",
            "tool_calls": [{"name": "lookup", "arguments": {"id": index}}]
            if draft_kind == "call"
            else [],
            "parse_ok": True,
            "token_logprobs": [-0.1, -0.3],
        },
        "candidates": [
            {
                "candidate_id": first,
                "source_id": f"source-{index}-0",
                "rank": 0,
                "feasible": True,
                "text": "first evidence",
            },
            {
                "candidate_id": second,
                "source_id": f"source-{index}-1",
                "rank": 1,
                "feasible": True,
                "text": "second evidence",
            },
        ],
        "allowed_actions": [
            {"action_id": "none", "candidate_ids": []},
            {"action_id": "top", "candidate_ids": [first]},
            {"action_id": "other", "candidate_ids": [second]},
            {"action_id": "pair", "candidate_ids": [first, second]},
        ],
        "local_llm_selected_ids": [first, second] if pair else [second],
    }


def _manifests(tmp_path):
    d128 = tmp_path / "d128.json"
    f128 = tmp_path / "f128.jsonl"
    d128.write_text(json.dumps({"task_ids": ["d-task"]}), encoding="utf-8")
    f128.write_text(json.dumps({"task_id": "f-task"}) + "\n", encoding="utf-8")
    return {"D128": d128, "F128": f128}


def _plan(tmp_path):
    return t02.build_plan(
        [_state(index) for index in range(6)],
        forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy={"name": "frozen-h0-continuation", "version": 1},
        seed=17,
        target_states=6,
        train_states=4,
        min_task_groups=3,
    )


def test_independent_a2_proposal_does_not_replace_c0_continuation():
    state = _state(0)
    state["set_selector"] = "candidate_rule"
    with pytest.raises(t02.T02Error, match="bound local_llm proposal"):
        t02._normalize_candidate_state(state)
    state["local_llm_proposal"] = {"selector": "local_llm",
        "candidate_ids": state["local_llm_selected_ids"]}
    normalized = t02._normalize_candidate_state(state)
    assert normalized["set_selector"] == "candidate_rule"
    assert t02._branch_actions(normalized, seed=0)[2]["candidate_ids"] == state["local_llm_selected_ids"]


def test_plan_is_task_disjoint_stratified_and_bounded(tmp_path):
    plan = _plan(tmp_path)
    receipt = t02.check_plan(plan)
    assert receipt["state_count"] == 6
    assert receipt["task_group_count"] == 3
    assert receipt["split_counts"] == {"train": 4, "calibration": 2}
    assert receipt["complete_branch_cap"] == 18
    split_by_task = {}
    for state in plan["states"]:
        split_by_task.setdefault(state["task_group_id"], state["split"])
        assert split_by_task[state["task_group_id"]] == state["split"]
        assert [branch["branch_id"] for branch in state["branches"]] == ["A0", "A1", "A2"]
        assert state["branches"][0]["candidate_ids"] == []
        assert len(state["branches"][1]["candidate_ids"]) == 1
        assert state["branches"][2]["candidate_ids"] != state["branches"][1]["candidate_ids"]
    assert {len(state["branches"][2]["candidate_ids"]) for state in plan["states"]} == {1, 2}


def test_plan_rejects_d128_or_f128_overlap_and_future_outcomes(tmp_path):
    forbidden = _state(0)
    forbidden["task_id"] = forbidden["task_group_id"] = "d-task"
    with pytest.raises(t02.T02Error, match="need 2 eligible"):
        t02.build_plan(
            [forbidden, _state(1)],
            forbidden_manifest_paths=_manifests(tmp_path),
            frozen_subsequent_policy={"name": "fixed"},
            target_states=2,
            train_states=1,
            min_task_groups=1,
        )

    leaked = _state(1)
    leaked["draft"]["turn_success"] = True
    with pytest.raises(t02.T02Error, match="future outcome"):
        t02.build_plan(
            [leaked, _state(2)],
            forbidden_manifest_paths=_manifests(tmp_path),
            frozen_subsequent_policy={"name": "fixed"},
            target_states=2,
            train_states=1,
            min_task_groups=1,
        )


def test_bfcl_base_long_pair_is_one_group_and_one_exclusion_unit(tmp_path):
    d128 = tmp_path / "d128.json"
    f128 = tmp_path / "f128.json"
    d128.write_text(json.dumps({"task_ids": ["multi_turn_base_13"]}), encoding="utf-8")
    f128.write_text(json.dumps({"task_ids": ["multi_turn_long_context_99"]}), encoding="utf-8")
    state = _state(90)
    state["task_id"] = state["task_group_id"] = "multi_turn_long_context_13"
    with pytest.raises(t02.T02Error, match="smoke state belongs"):
        t02.build_smoke_plan(
            state,
            forbidden_manifest_paths={"D128": d128, "F128": f128},
            frozen_subsequent_policy={"name": "fixed"},
        )
    state["task_id"] = state["task_group_id"] = "multi_turn_long_context_14"
    normalized = t02._normalize_candidate_state(state)
    assert normalized["task_group_id"] == "bfcl_pair_14"


def test_expanded_120_state_budget_and_family_variant_caps(tmp_path):
    categories = ("base", "long_context", "miss_func", "miss_param")
    states = []
    index = 0
    for group in range(26):
        for category in categories:
            for draft_kind in ("call", "stop"):
                state = _state(index, draft_kind=draft_kind)
                state["task_id"] = f"multi_turn_{category}_{group}"
                state["task_group_id"] = f"bfcl_pair_{group}"
                states.append(state)
                index += 1
    plan = t02.build_plan(
        states,
        forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy={"name": "expanded-C0"},
        seed=0,
        target_states=120,
        train_states=80,
        min_task_groups=26,
        max_states_per_task_group=8,
        max_states_per_task=2,
        max_complete_branch_executions=360,
    )
    receipt = t02.check_plan(plan)
    assert receipt["state_count"] == 120
    assert receipt["task_group_count"] == 26
    assert receipt["complete_branch_cap"] == 360
    groups = {}
    variants = {}
    for state in plan["states"]:
        groups[state["task_group_id"]] = groups.get(state["task_group_id"], 0) + 1
        variants[state["task_id"]] = variants.get(state["task_id"], 0) + 1
    assert max(groups.values()) <= 8
    assert max(variants.values()) <= 2
    assert receipt["split_counts"] == {"train": 80, "calibration": 40}

def test_lift_server_steps_uses_only_exported_selection_state():
    state = _state(2)
    selection = {key: copy.deepcopy(value) for key, value in state.items() if key not in {
        "schema", "state_id", "benchmark", "task_id", "task_group_id", "decision_key"
    }}
    selection["draft_kind"] = "STOP"
    selection["actions"] = selection.pop("allowed_actions")
    for index, action in enumerate(selection["actions"]):
        action["action_id"] = index
    rows = [{
        "schema": "a-event-native-exact-step-v1",
        "session_id": "bfcl/multi_turn_base_7/attempt-0",
        "decision_key": "turn-1/step-3",
        "recovery_checks": [{"selection_state": selection}],
        "response": {"content": "not inspected by T02 planning"},
    }]
    lifted = t02.lift_server_steps(rows)
    assert len(lifted) == 1
    assert lifted[0]["task_id"] == "multi_turn_base_7"
    assert lifted[0]["task_group_id"] == "bfcl_pair_7"
    assert lifted[0]["decision_key"] == "turn-1/step-3"
    assert lifted[0]["draft_kind"] == "stop"
    assert lifted[0]["allowed_actions"][0]["action_id"] == 0
    assert lifted[0]["allowed_actions"][0]["candidate_ids"] == []


def test_existing_bfcl_prefix_gate_is_recorded_as_prefix_only():
    context = {
        "benchmark": "bfcl",
        "task_id": "multi_turn_base_7",
        "user_turn": 0,
        "step": 0,
        "attempt": 0,
    }
    receipt = t02.inspect_bfcl_prefix_replay({
        "task_id": "multi_turn_base_7",
        "replay": [],
        "branch": {"context": context, "request_messages": [{"role": "user", "content": "x"}], "request_tools": []},
    })
    assert receipt["request_prefix_replay"] is True
    assert receipt["exact_actor_backend_restore"] is False
    assert receipt["acceptable_as_t02_snapshot"] is False


def test_capabilities_cli_reports_implemented_unvalidated_live_boundary(capsys):
    assert t02.main(["capabilities"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == t02.RUNTIME_CAPABILITY_SCHEMA
    assert report["offline_planning"] is True
    assert report["offline_labeling"] is True
    assert report["exact_branch_adapter_available"] is True
    assert report["status"] == "implemented_requires_live_backend_validation"
    assert report["real_backend_validated"] is False
    assert report["missing_exact_restore_components"] == []
    assert report["execution_behavior_without_adapter"] == "fail_closed"


class ExactFakeAdapter:
    def __init__(self):
        self.restores = []

    def capabilities(self):
        return {
            "schema": t02.CAPABILITY_SCHEMA,
            "exact_same_state_restore": True,
            "components": list(t02.REQUIRED_COMPONENTS),
            "frozen_subsequent_policy": True,
            "official_turn_outcome": True,
            "official_task_outcome": True,
        }

    def capture_state(self, state, *, frozen_policy):
        return {
            "schema": t02.SNAPSHOT_SCHEMA,
            "state_id": state["state_id"],
            "snapshot_id": f"snapshot/{state['state_id']}",
            "frozen_policy_sha256": _sha(frozen_policy),
            "component_digests": {
                component: hashlib.sha256(f"{state['state_id']}:{component}".encode()).hexdigest()
                for component in t02.REQUIRED_COMPONENTS
            },
        }

    def restore_state(self, snapshot, *, frozen_policy):
        receipt = {
            "schema": t02.RESTORE_SCHEMA,
            "state_id": snapshot["state_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "frozen_policy_sha256": _sha(frozen_policy),
            "component_digests": copy.deepcopy(snapshot["component_digests"]),
            "restored": True,
        }
        self.restores.append(copy.deepcopy(receipt))
        return receipt

    def run_branch(self, state, branch, *, frozen_policy, restore_receipt):
        outcome = {
            "A0": (False, False),
            "A1": (True, False),
            "A2": (None, True),
        }[branch["branch_id"]]
        return {
            "schema": t02.RESULT_SCHEMA,
            "state_id": state["state_id"],
            "branch_id": branch["branch_id"],
            "candidate_ids": branch["candidate_ids"],
            "snapshot_id": restore_receipt["snapshot_id"],
            "restore_receipt_sha256": _sha(restore_receipt),
            "frozen_policy_sha256": _sha(frozen_policy),
            "execution_status": "complete",
            "execution_receipt": {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": branch["candidate_ids"],
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            },
            "official_outcome": {
                "source": "official",
                "scorer": "bfcl-multi-turn-checker",
                "artifact": f"official/{state['state_id']}/{branch['branch_id']}.json",
                "artifact_sha256": hashlib.sha256(
                    f"{state['state_id']}:{branch['branch_id']}".encode()
                ).hexdigest(),
                "turn_success": outcome[0],
                "task_success": outcome[1],
            },
        }


def test_execute_restores_every_branch_and_labels_only_official_outcomes(tmp_path):
    plan = _plan(tmp_path)
    adapter = ExactFakeAdapter()
    results = t02.execute_plan(plan, adapter)
    assert results["complete_branch_executions"] == 18
    assert len(adapter.restores) == 18
    for state in plan["states"]:
        restores = [row for row in adapter.restores if row["state_id"] == state["state_id"]]
        assert len(restores) == 3
        assert restores[0]["component_digests"] == restores[1]["component_digests"] == restores[2]["component_digests"]

    labeled = t02.label_results(plan, results)
    assert labeled["state_count"] == 6
    row = labeled["rows"][0]
    assert row["labels"]["A1"]["delta_turn"] == 1
    assert row["labels"]["A1"]["delta_task"] == 0
    assert row["labels"]["A2"]["delta_turn"] is None
    assert row["labels"]["A2"]["turn_label_status"] == "unknown"
    assert row["labels"]["A2"]["delta_task"] == 1
    assert row["c1_risk_label"] == 1
    assert row["labels"]["A1"]["candidate_ids"] == row["branches"][1]["candidate_ids"]
    assert row["branches_are_independent_states"] is False
    tested = {tuple(branch["candidate_ids"]) for branch in row["branches"]}
    assert all(tuple(action["candidate_ids"]) not in tested for action in row["untested_actions"])
    assert all(action["delta_turn"] is None and action["delta_task"] is None
               for action in row["untested_actions"])


def test_execute_fails_closed_without_exact_backend_restore(tmp_path):
    class PrefixOnly(ExactFakeAdapter):
        def capabilities(self):
            value = super().capabilities()
            value["exact_same_state_restore"] = False
            value["prefix_replay_schema"] = "bfcl-frozen-prefix-replay-v1"
            return value

    with pytest.raises(t02.T02CapabilityError, match="exact_same_state_restore"):
        t02.execute_plan(_plan(tmp_path), PrefixOnly())


def test_missing_branch_is_unknown_not_zero(tmp_path):
    plan = _plan(tmp_path)
    full = t02.execute_plan(plan, ExactFakeAdapter())
    full["results"] = [
        row
        for row in full["results"]
        if not (row["state_id"] == plan["states"][0]["state_id"] and row["branch_id"] == "A2")
    ]
    full["complete_branch_executions"] -= 1
    labeled = t02.label_results(plan, full)
    row = next(item for item in labeled["rows"] if item["state_id"] == plan["states"][0]["state_id"])
    assert row["outcomes"]["A2"]["execution_status"] == "missing"
    assert row["labels"]["A2"]["delta_turn"] is None
    assert row["labels"]["A2"]["delta_task"] is None


def test_cli_plan_check_and_label_round_trip(tmp_path, capsys):
    states_path = tmp_path / "states.jsonl"
    states_path.write_text(
        "".join(json.dumps(_state(index)) + "\n" for index in range(6)),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"name": "fixed"}), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    manifests = _manifests(tmp_path)
    assert t02.main([
        "plan", "--states", str(states_path), "--frozen-policy", str(policy_path),
        "--forbidden-manifest", f"D128={manifests['D128']}",
        "--forbidden-manifest", f"F128={manifests['F128']}",
        "--output", str(plan_path), "--seed", "3", "--target-states", "6",
        "--train-states", "4", "--min-task-groups", "3",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert t02.main(["check", "--artifact", str(plan_path)]) == 0
    assert json.loads(capsys.readouterr().out)["state_count"] == 6

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(t02.execute_plan(plan, ExactFakeAdapter())), encoding="utf-8"
    )
    labels_path = tmp_path / "labels.json"
    assert t02.main([
        "label", "--plan", str(plan_path), "--results", str(results_path),
        "--output", str(labels_path),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["state_count"] == 6
    assert json.loads(labels_path.read_text(encoding="utf-8"))["schema"] == t02.LABELED_SET_SCHEMA


class FrozenFeatureModels:
    """Deterministic CPU feature backends; no server or model process is used."""

    def embed(self, *, texts, purpose, config):
        del purpose, config
        return [
            [float(max(len(text), 1)), float(text.count("violet") + 1),
             float(text.count("lookup") + 1)]
            for text in texts
        ]

    def rerank(self, *, query, documents):
        return [0.75 + 0.01 * float("violet" in document)
                + 0.001 * float(bool(query)) for document in documents]

    def rerank_retrieval_candidates(
        self, *, task, draft, documents, overflow_policy
    ):
        if overflow_policy not in {
            "error", "task_head_tail_preserve_draft_v1"
        }:
            raise ValueError("unsupported retrieval query overflow policy")
        return self.rerank(query=task + "\n" + draft, documents=documents)

    def drain_receipts(self):
        return []

    def public_config(self):
        return {
            "embedding": {"backend": "cpu-test", "model": "frozen-test-embedding"},
            "reranker": {"backend": "cpu-test", "model": "frozen-test-reranker"},
        }


def _real_exported_server_steps():
    from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
    from benchmarks.memory_runtime.tests.test_evidence_sets import LocalModels
    from benchmarks.memory_runtime.tests.test_event_native_recovery import draft_call, request
    from benchmarks.memory_runtime.tests.test_gp_recovery import make_controller

    steps = []
    for index in range(8):
        gp = parse_gp_config({
            "selection_protocol": "evidence_sets_v1",
            "D": "candidate_rule",
            "Q": "lexical",
            "U": "tokens_1024",
            "K": 2,
            "candidate_limit": 8,
            "selector_max_units": 2,
            "set_selector": "local_llm",
            "R": 1,
            "export_selection_state": True,
        })
        controller = make_controller(**gp)
        # Legal actions are empty, three singletons, then distinct-source pairs.
        controller.backends = LocalModels(action=4 if index % 2 == 0 else 2)
        payload = request(f"turn-{index % 2}/step-0")
        payload["session_id"] = f"bfcl/integration_task_{index // 2}/attempt-0"
        prepared = controller.prepare(payload, ratio=4, max_new_tokens=32)
        prepared._set_draft_logprobs = (-0.1 - index * 0.01, -0.2)
        controller._detector_gate = lambda prepared, high=index % 2 == 0: {
            "triggered": high,
            "score": 0.9 if high else 0.1,
            "reason": "deterministic_cpu_test_detector",
        }
        calls = [draft_call()] if index % 2 == 0 else []
        result = controller.reconsider(
            prepared,
            calls,
            draft_text="lookup violet other-a" if calls else "done violet other-a",
        )
        decision = result["decision"]
        assert decision["selection_state"]["schema"] == t02.CANDIDATE_SCHEMA
        steps.append({
            "schema": "a-event-native-exact-step-v1",
            "session_id": payload["session_id"],
            "decision_key": payload["decision_key"],
            "recovery_checks": [decision],
        })
    return steps


def test_real_controller_export_joins_t02_pipeline_and_c4_trainer(tmp_path):
    from benchmarks.memory_runtime.recovery.set_training import train_c4_models
    from benchmarks.memory_runtime.tests.test_gp_recovery import UnitTokenizer

    states = t02.lift_server_steps(_real_exported_server_steps())
    assert len(states) == 8
    assert {state["set_selector"] for state in states} == {"local_llm"}
    assert {state["draft_kind"] for state in states} == {"call", "stop"}
    assert {state["risk_bucket"] for state in states} == {"high", "low"}
    assert all(state["local_llm_selected_ids"] for state in states)
    assert all(state["state_scope"] == "observations_only_not_backend_snapshot"
               for state in states)

    plan = t02.build_plan(
        states,
        forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy={
            "name": "integration-test-frozen-continuation",
            "generation": "deterministic_cpu_fake",
        },
        seed=23,
        target_states=8,
        train_states=6,
        min_task_groups=4,
    )
    assert {len(state["branches"][2]["candidate_ids"])
            for state in plan["states"]} == {1, 2}
    assert t02.check_plan(plan)["split_counts"] == {"train": 6, "calibration": 2}

    results = t02.execute_plan(plan, ExactFakeAdapter())
    labeled = t02.label_results(plan, results)
    assert labeled["state_count"] == 8
    assert all(row["schema"] == t02.LABELED_STATE_SCHEMA for row in labeled["rows"])
    artifacts = train_c4_models(
        labeled["rows"],
        tokenizer=UnitTokenizer(),
        provenance={
            "source": "actual_GPRecoveryController.selection_state",
            "collection": "cpu_integration_test",
        },
        models=FrozenFeatureModels(),
    )
    assert set(artifacts) == {"delta_turn", "delta_task"}
    assert artifacts["delta_turn"]["fit"]["group_count"] == 3
    assert artifacts["delta_turn"]["fit"]["known_example_count"] == 6
    assert artifacts["delta_turn"]["fit"]["unknown_label_count"] == 6
    assert artifacts["delta_task"]["fit"]["known_example_count"] == 12
    assert artifacts["delta_task"]["fit"]["calibration_state_count_excluded"] == 2
