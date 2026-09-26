"""CPU contracts for the exact BFCL T02 environment/actor adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import t02
import t02_bfcl


class FakeUtils:
    @staticmethod
    def is_empty_execute_response(value):
        return not value


class FakeHandler:
    def _pre_query_processing_FC(self, data, task):
        data["message"] = []
        return data

    def _compile_tools(self, data, task):
        data["tools"] = copy.deepcopy(task["function"])
        return data

    def add_first_turn_message_FC(self, data, message):
        data["message"].extend(copy.deepcopy(message))
        return data

    def _add_next_turn_user_message_FC(self, data, message):
        data["message"].extend(copy.deepcopy(message))
        return data

    def _add_assistant_message_FC(self, data, response):
        data["message"].append(copy.deepcopy(response["model_responses_message_for_chat_history"]))
        return data

    def _add_execution_results_FC(self, data, results, response):
        for value, call_id in zip(results, response["tool_call_ids"]):
            data["message"].append({"role": "tool", "content": value, "tool_call_id": call_id})
        return data

    def decode_execute(self, response, has_tool_call_tag=False):
        if not isinstance(response, list):
            return []
        return [next(iter(row)) for row in response]


class FakeBindings:
    default_holdout_prompt = "new tool"
    maximum_step_limit = 20

    def __init__(self):
        self.multi_turn_utils = FakeUtils()
        self.instances = {}

    def make_handler(self, namespace):
        return FakeHandler()

    def clear_instances(self, task, namespace):
        self.instances.pop(namespace, None)

    def initialize_environment(self, task, namespace):
        self.instances[namespace] = {"executed": []}

    def capture_instances(self, task, namespace):
        return {namespace: copy.deepcopy(self.instances[namespace])}

    def restore_instances(self, values):
        for namespace, value in values.items():
            self.instances[namespace] = copy.deepcopy(value)

    def execute(self, calls, task, namespace):
        self.instances[namespace]["executed"].extend(calls)
        return [f"ok:{call}" for call in calls]

    def score_turn_prefix(self, handler, raw_result, ground_truth, task, turn_index):
        return {"valid": all(
            any(isinstance(step, list) and step for step in turn)
            for turn in raw_result[: turn_index + 1]
        )}

    def score_task(self, handler, raw_result, ground_truth, task):
        return {"valid": len(raw_result) == len(ground_truth) and all(
            any(isinstance(step, list) and step for step in turn)
            for turn in raw_result
        )}

    @staticmethod
    def make_json_serializable(value):
        return value


def response(name=None):
    calls = [] if name is None else [{
        "id": f"call-{name}", "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }]
    return {"role": "assistant", "content": "" if calls else "done",
            "tool_calls": calls, "finish_reason": "tool_calls" if calls else "stop"}


class FakeActor:
    def __init__(self):
        self.held = False
        self.snapshots = {}
        self.generated = 0
        self.total_generation_calls = 0
        self.closed = False

    def hold(self, payload):
        self.held = True
        self.last_state = {
            "schema": t02.CANDIDATE_SCHEMA,
            "state_id": "live-state-0",
            "benchmark": "bfcl",
            "task_id": "multi_turn_base_3",
            "task_group_id": "multi_turn_base_3",
            "decision_key": payload["decision_key"],
            "draft_kind": "stop",
            "set_selector": "candidate_rule",
            "risk_bucket": "low",
            "q": {
                "goal": "do it",
                "draft_logprobs": [-0.1, -0.2],
                "prefill_hidden": [float(index) for index in range(8)],
                "prefill_contract": {
                    "layer": 16, "readout": "last_prompt_token",
                    "position_kind": "last_prompt_token",
                    "bindings": {"model": "cpu-fake", "tokenizer": "cpu-fake"},
                },
            },
            "draft": {"text": "done", "tool_calls": [], "parse_ok": True},
            "candidates": [
                {"candidate_id": "c1", "source_id": "e1", "rank": 1,
                 "feasible": True, "text": "one"},
                {"candidate_id": "c2", "source_id": "e2", "rank": 2,
                 "feasible": True, "text": "two"},
            ],
            "allowed_actions": [
                {"action_id": 0, "candidate_ids": []},
                {"action_id": 1, "candidate_ids": ["c1"]},
                {"action_id": 2, "candidate_ids": ["c2"]},
            ],
            "selected_ids": [],
            "held_draft_response": response(),
        }
        self.total_generation_calls += 1
        return copy.deepcopy(self.last_state)

    def propose_alternative(self):
        state = copy.deepcopy(self.last_state)
        state["local_llm_selected_ids"] = ["c2"]
        state["local_llm_proposal"] = {
            "selector": "local_llm", "candidate_ids": ["c2"],
            "selection": {"reason": "fake"}, "model_calls": [],
        }
        return state

    def capture(self):
        snapshot = {"snapshot_id": f"actor-snapshot-{len(self.snapshots)}", "component_digests": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("actor_kv", "actor_positions", "backend_stats", "rng")
        }}
        self.snapshots[snapshot["snapshot_id"]] = copy.deepcopy(snapshot)
        return snapshot

    def restore(self, snapshot):
        assert self.snapshots[snapshot["snapshot_id"]] == snapshot
        self.held = True
        return copy.deepcopy(snapshot)

    def submit_held(self, candidate_ids):
        assert self.held
        self.held = False
        self.total_generation_calls += int(bool(candidate_ids))
        return response(candidate_ids[0] if candidate_ids else None)

    def generate(self, payload):
        self.generated += 1
        self.total_generation_calls += 1
        return response()

    def release(self, snapshot):
        self.snapshots.pop(snapshot["snapshot_id"])

    def close(self):
        self.closed = True


class SequencedActor(FakeActor):
    def __init__(self):
        super().__init__()
        self.holds = 0

    def hold(self, payload):
        value = super().hold(payload)
        self.holds += 1
        value["state_id"] = f"live-state-{self.holds}"
        if self.holds < 3:
            value["draft_kind"] = "call"
            value["draft"] = {"text": "", "tool_calls": response("c1")["tool_calls"], "parse_ok": True}
        self.last_state = copy.deepcopy(value)
        return value


def task():
    return {
        "id": "multi_turn_base_3",
        "question": [[{"role": "user", "content": "do it"}]],
        "function": [{"name": "c1"}, {"name": "c2"}],
        "involved_classes": ["Fake"],
        "initial_config": {},
    }, [["c1()"]]


def manifests(tmp_path):
    paths = {}
    for name, task_id in (("D128", "multi_turn_base_8"), ("F128", "multi_turn_base_9")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"task_ids": [task_id]}), encoding="utf-8")
        paths[name] = path
    return paths


def test_environment_restores_live_execution_and_message_state():
    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(entry, gold, bindings=bindings, namespace="live")
    payload = environment.next_payload()
    snapshot = environment.capture()
    environment.commit_response(response("c1"))
    assert bindings.instances["live"]["executed"] == ["c1"]
    environment.restore(snapshot)
    assert bindings.instances["live"]["executed"] == []
    assert environment._awaiting_response is True
    assert environment.inference_data["message"] == payload["messages"]
    environment.commit_response(response("c1"))
    assert bindings.instances["live"]["executed"] == ["c1"]


def test_current_turn_is_unknown_when_official_previous_prefix_failed():
    bindings = FakeBindings()
    entry, gold = task()
    entry["question"].append([{"role": "user", "content": "try again"}])
    gold.append(["c1()"])
    environment = t02_bfcl.BFCLTaskEnvironment(entry, gold, bindings=bindings, namespace="past-fail")
    environment.next_payload()
    environment.commit_response(response())
    assert environment.turn_index == 1
    assert environment.previous_turn_valid() is False
    environment.next_payload()
    snapshot = environment.capture()
    environment.commit_response(response("c1"))
    environment.next_payload()
    environment.commit_response(response())
    outcome = environment.official_outcomes(1)
    assert outcome["turn_success"] is None
    assert outcome["turn_checker_result"]["status"] == "unknown_previous_prefix_invalid"
    environment.restore(snapshot)
    assert environment.previous_turn_valid() is False


def test_official_outcomes_use_bfcl_serialization_for_directory():
    pytest.importorskip("bfcl_eval", reason="Official BFCL is an optional runtime dependency")
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import (
        Directory,
    )
    from bfcl_eval.utils import make_json_serializable as official_make_json_serializable

    class DirectoryBindings(FakeBindings):
        def __init__(self):
            super().__init__()
            self.directory = Directory("/", None)
            self.directory._add_directory("workspace")
            self.directory.contents["workspace"]._add_file("note.txt", "kept")

        def score_turn_prefix(self, handler, raw_result, ground_truth, task, turn_index):
            return {"valid": True, "state": self.directory}

        def score_task(self, handler, raw_result, ground_truth, task):
            return {"valid": True, "state": self.directory}

        make_json_serializable = staticmethod(official_make_json_serializable)

    bindings = DirectoryBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="directory-result"
    )
    environment.next_payload()
    environment.commit_response(response())

    outcome = environment.official_outcomes(0)

    assert outcome["turn_success"] is True
    assert outcome["task_success"] is True
    assert "Directory: /" in outcome["turn_checker_result"]["state"]
    assert "note.txt" in outcome["task_checker_result"]["state"]
    assert "kept" in outcome["task_checker_result"]["state"]
    json.dumps(outcome, ensure_ascii=False, allow_nan=False)


def test_json_copy_rejects_normalized_key_collisions_and_unsupported_keys():
    with pytest.raises(ValueError, match=r"JSON object key collision at \$\.orders"):
        t02_bfcl._json_copy({"orders": {12446: {"status": "open"}, "12446": "duplicate"}})
    with pytest.raises(TypeError, match="keys must be str, int, float, bool or None"):
        t02_bfcl._json_copy({"orders": {(12446, 12447): "unsupported"}})


def test_mixed_orders_checker_result_survives_branch_artifact_and_labeling(tmp_path):
    mismatch = {
        "valid": False,
        "error_type": "multi_turn:instance_state_mismatch",
        "error_message": (
            "Model instance for TradingBot does not match the state with "
            "ground truth instance."
        ),
        "details": {
            "model_instance_state": {"orders": {"order_type": "Buy"}},
            "ground_truth_instance_state": {
                "orders": {
                    "order_type": "Buy",
                    12446: {
                        "amount": 100,
                        "id": 12446,
                        "order_type": "Buy",
                        "price": 150.0,
                        "status": "Open",
                        "symbol": "XYZ",
                    },
                }
            },
        },
    }

    class MixedOrdersBindings(FakeBindings):
        def _score(self, raw_result):
            valid = bool(raw_result) and all(
                any(isinstance(step, list) and step for step in turn)
                for turn in raw_result
            )
            return {"valid": True} if valid else copy.deepcopy(mismatch)

        def score_turn_prefix(self, handler, raw_result, ground_truth, task, turn_index):
            return self._score(raw_result)

        def score_task(self, handler, raw_result, ground_truth, task):
            return self._score(raw_result)

    bindings = MixedOrdersBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="mixed-orders"
    )
    actor = FakeActor()
    policy = {"name": "frozen-C0", "version": 1}
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official"
    )
    candidates = adapter.discover_task(environment, actor, seed=0, max_states=1)
    plan = t02.build_smoke_plan(
        candidates[0], forbidden_manifest_paths=manifests(tmp_path),
        frozen_subsequent_policy=policy, seed=0,
    )
    adapter.prune([candidates[0]["state_id"]])

    results = t02.execute_plan(plan, adapter)
    labels = t02.label_results(plan, results)

    a0 = next(row for row in results["results"] if row["branch_id"] == "A0")
    artifact = json.loads(Path(a0["official_outcome"]["artifact"]).read_text())
    checker = artifact["outcomes"]["turn_checker_result"]
    assert checker["error_type"] == "multi_turn:instance_state_mismatch"
    assert checker["error_message"] == (
        "Model instance for TradingBot does not match the state with ground truth instance."
    )
    assert checker["details"]["model_instance_state"]["orders"] == {"order_type": "Buy"}
    assert checker["details"]["ground_truth_instance_state"]["orders"] == {
        "12446": {
            "amount": 100,
            "id": 12446,
            "order_type": "Buy",
            "price": 150.0,
            "status": "Open",
            "symbol": "XYZ",
        },
        "order_type": "Buy",
    }
    assert labels["rows"][0]["c1_risk_label"] == 1
    assert labels["rows"][0]["labels"]["A1"]["delta_turn"] == 1
    assert labels["rows"][0]["labels"]["A1"]["delta_task"] == 1
    assert labels["rows"][0]["labels"]["A2"]["delta_turn"] == 1
    assert labels["rows"][0]["labels"]["A2"]["delta_task"] == 1


def test_discovery_keeps_first_call_and_first_stop_not_two_calls(tmp_path):
    bindings = FakeBindings()
    entry, gold = task()
    entry["question"] = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
        [{"role": "user", "content": "three"}],
        [{"role": "user", "content": "unused tail"}],
    ]
    gold = [["c1()"], ["c1()"], ["c1()"], ["c1()"]]
    environment = t02_bfcl.BFCLTaskEnvironment(entry, gold, bindings=bindings, namespace="strata")
    actor = SequencedActor()
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy={"name": "C0"}, artifact_dir=tmp_path / "official")
    states = adapter.discover_task(environment, actor, seed=0, max_states=2)
    assert [state["draft_kind"] for state in states] == ["call", "stop"]
    assert actor.holds == 3
    assert environment.finished is False
    assert adapter.cost_summary()["source_collection_generation_calls"] == 3
    adapter.close()


def test_discovery_capacity_terminal_preserves_captured_state(tmp_path):
    class SourceCapacityActor(SequencedActor):
        def hold(self, payload):
            if self.holds:
                self.total_generation_calls += 1
                raise t02_bfcl.CapacityInfeasible("history exceeds declared capacity")
            return super().hold(payload)

    bindings = FakeBindings()
    entry, gold = task()
    entry["question"].append([{"role": "user", "content": "second turn"}])
    gold.append(["c1()"])
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="source-capacity"
    )
    actor = SourceCapacityActor()
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy={"name": "C0"}, artifact_dir=tmp_path / "official"
    )

    states = adapter.discover_task(environment, actor, seed=0, max_states=2)

    assert [state["state_id"] for state in states] == ["live-state-1"]
    assert environment.finished is True
    terminal = adapter.source_task_terminal(entry["id"])
    assert terminal["reason"] == "capacity_infeasible"
    assert terminal["terminal_status"] == "budget_terminated"
    assert terminal["stage"] == "source_collection"
    assert terminal["response_fabricated"] is False
    assert adapter.cost_summary()["source_collection_generation_calls"] == 2
    adapter.prune([states[0]["state_id"]])
    adapter.close()


def test_discovery_unexpected_error_still_propagates(tmp_path):
    class BrokenActor(FakeActor):
        def hold(self, payload):
            del payload
            self.total_generation_calls += 1
            raise RuntimeError("unexpected backend failure")

    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="unexpected-error"
    )
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy={"name": "C0"}, artifact_dir=tmp_path / "official"
    )

    with pytest.raises(RuntimeError, match="unexpected backend failure"):
        adapter.discover_task(environment, BrokenActor(), seed=0, max_states=1)

    assert adapter.source_task_terminal(entry["id"]) is None


def test_smoke_executes_three_exact_branches_and_official_labels(tmp_path):
    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(entry, gold, bindings=bindings, namespace="live")
    actor = FakeActor()
    policy = {"name": "frozen-C0", "version": 1}
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official")
    candidates = adapter.discover_task(environment, actor, seed=0, max_states=1)
    assert len(candidates) == 1
    plan = t02.build_smoke_plan(
        candidates[0], forbidden_manifest_paths=manifests(tmp_path),
        frozen_subsequent_policy=policy, seed=0)
    adapter.prune([candidates[0]["state_id"]])
    results = t02.execute_plan(plan, adapter)
    assert results["complete_branch_executions"] == 3
    assert [row["branch_id"] for row in results["results"]] == ["A0", "A1", "A2"]
    assert [row["official_outcome"]["task_success"] for row in results["results"]] == [False, True, True]
    assert actor.generated == 2
    assert adapter.cost_summary()["branch_generation_calls"] == 4
    assert actor.closed is True
    labels = t02.label_results(plan, results)
    assert labels["rows"][0]["labels"]["A1"]["delta_task"] == 1
    assert labels["rows"][0]["labels"]["A2"]["delta_task"] == 1
    assert len(list((tmp_path / "official").rglob("*.json"))) == 3


def test_branch_capacity_terminal_is_officially_unknown_without_fake_response(tmp_path):
    class BranchCapacityActor(FakeActor):
        def generate(self, payload):
            del payload
            self.generated += 1
            self.total_generation_calls += 1
            raise t02_bfcl.CapacityInfeasible("continuation exceeds declared capacity")

    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="branch-capacity"
    )
    actor = BranchCapacityActor()
    policy = {"name": "frozen-C0", "version": 1}
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official"
    )
    candidates = adapter.discover_task(environment, actor, seed=0, max_states=1)
    plan = t02.build_smoke_plan(
        candidates[0], forbidden_manifest_paths=manifests(tmp_path),
        frozen_subsequent_policy=policy, seed=0
    )
    adapter.prune([candidates[0]["state_id"]])

    results = t02.execute_plan(plan, adapter)

    by_branch = {row["branch_id"]: row for row in results["results"]}
    assert by_branch["A0"]["official_outcome"]["task_success"] is False
    for branch_id in ("A1", "A2"):
        row = by_branch[branch_id]
        assert row["execution_status"] == "complete"
        assert row["official_outcome"]["turn_success"] is None
        assert row["official_outcome"]["task_success"] is None
        terminal = row["official_outcome"]["capacity_termination"]
        assert terminal["stage"] == "branch_continuation"
        assert terminal["terminal_status"] == "budget_terminated"
        assert terminal["response_fabricated"] is False
        artifact = json.loads(Path(row["official_outcome"]["artifact"]).read_text())
        expected_call = "c1" if branch_id == "A1" else "c2"
        assert artifact["model_result"] == [[[{expected_call: "{}"}]]]
    assert actor.generated == 2
    labels = t02.label_results(plan, results)
    assert labels["rows"][0]["labels"]["A1"]["delta_task"] is None
    assert labels["rows"][0]["labels"]["A2"]["delta_task"] is None


def test_branch_submit_capacity_error_is_audited_terminal_outcome(tmp_path):
    class SubmitCapacityActor(FakeActor):
        def submit_held(self, candidate_ids):
            if candidate_ids:
                self.total_generation_calls += 1
                raise t02_bfcl.CapacityInfeasible("intervention submission failed")
            return super().submit_held(candidate_ids)

    bindings = FakeBindings()
    entry, gold = task()
    environment = t02_bfcl.BFCLTaskEnvironment(
        entry, gold, bindings=bindings, namespace="submit-capacity"
    )
    actor = SubmitCapacityActor()
    policy = {"name": "frozen-C0", "version": 1}
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=policy, artifact_dir=tmp_path / "official"
    )
    candidates = adapter.discover_task(environment, actor, seed=0, max_states=1)
    plan = t02.build_smoke_plan(
        candidates[0], forbidden_manifest_paths=manifests(tmp_path),
        frozen_subsequent_policy=policy, seed=0
    )
    adapter.prune([candidates[0]["state_id"]])

    results = t02.execute_plan(plan, adapter)

    by_branch = {row["branch_id"]: row for row in results["results"]}
    assert by_branch["A0"]["official_outcome"]["task_success"] is False
    for branch_id in ("A1", "A2"):
        row = by_branch[branch_id]
        assert row["execution_status"] == "complete"
        assert row["official_outcome"]["turn_success"] is None
        assert row["official_outcome"]["task_success"] is None
        terminal = row["official_outcome"]["capacity_termination"]
        assert terminal["stage"] == "branch_intervention"
        assert terminal["terminal_status"] == "budget_terminated"
        assert terminal["response_fabricated"] is False
        artifact = json.loads(Path(row["official_outcome"]["artifact"]).read_text())
        assert artifact["model_result"] == [[]]


def test_expanded_manifest_family_bindings_and_round_robin_cover_all_families():
    root = HERE.parents[1]
    evidence = root / "outputs" / "history_system_search" / "evidence_sets_v1"
    task_ids, bindings, receipt = t02_bfcl.load_family_bindings(
        evidence / "tasks.t02.expanded104.json",
        evidence / "data_expansion_audit.json",
    )
    ordered = t02_bfcl.round_robin_family_tasks(task_ids, bindings)
    assert receipt["task_count"] == 104
    assert receipt["source_family_count"] == 26
    assert len({bindings[task_id] for task_id in ordered[:26]}) == 26
    assert set(ordered) == set(task_ids)
    assert len(ordered) == len(task_ids)


def test_collection_source_budget_stops_before_starting_another_actor(tmp_path):
    bindings = FakeBindings()
    bindings.load_task = lambda task_id: task()
    actors, progress = [], []

    def actor_factory(task_id, task_index):
        actor = FakeActor()
        actors.append(actor)
        return actor

    with pytest.raises(t02.T02Error, match="could not build T02 plan"):
        t02_bfcl.collect_live_plan(
            task_ids=["multi_turn_base_3", "multi_turn_base_4"],
            actor_factory=actor_factory, bindings=bindings,
            forbidden_manifest_paths=manifests(tmp_path),
            frozen_policy={"name": "C0"}, artifact_dir=tmp_path / "official",
            seed=0, target_states=2, train_states=1, min_task_groups=1,
            max_states_per_task=1, candidate_snapshot_cap=2,
            max_source_task_starts=1, on_progress=progress.append,
        )
    assert len(actors) == 1
    assert actors[0].closed is True
    assert progress[-1]["source_task_starts"] == 1
    assert progress[-1]["candidate_count"] == 1


def test_training_feature_preflight_requires_finite_compatible_hidden_state():
    actor = FakeActor()
    state = actor.hold({"decision_key": "turn-0/step-0"})
    receipt = t02_bfcl.validate_training_feature_contract([state])
    assert receipt["prefill_hidden_dimension"] == 8
    broken = copy.deepcopy(state)
    broken["state_id"] = "broken"
    broken["q"]["draft_logprobs"] = [float("nan")]
    try:
        t02_bfcl.validate_training_feature_contract([broken])
    except t02.T02Error as error:
        assert "draft_logprobs" in str(error)
    else:
        raise AssertionError("nonfinite draft logprobs passed preflight")


def test_four_official_categories_share_one_source_family():
    task_ids = [
        "multi_turn_base_3", "multi_turn_long_context_3",
        "multi_turn_miss_func_3", "multi_turn_miss_param_3",
    ]
    assert {t02_bfcl.OfficialBFCLBindings.category(task_id) for task_id in task_ids} == {
        "multi_turn_base", "multi_turn_long_context",
        "multi_turn_miss_func", "multi_turn_miss_param",
    }
    assert {t02.canonical_task_group_id(task_id) for task_id in task_ids} == {"bfcl_pair_3"}


def test_generation_cost_is_counted_when_actor_call_raises(tmp_path):
    class FailingActor:
        total_generation_calls = 0

        def generate(self, payload):
            del payload
            self.total_generation_calls += 1
            raise RuntimeError("after submission")

    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy={"name": "C0"}, artifact_dir=tmp_path / "official"
    )
    actor = FailingActor()
    try:
        adapter._invoke_counted(actor, "branch_generation_calls", "generate", {})
    except RuntimeError as error:
        assert str(error) == "after submission"
    else:
        raise AssertionError("failing actor call unexpectedly returned")
    assert adapter.cost_summary()["branch_generation_calls"] == 1
