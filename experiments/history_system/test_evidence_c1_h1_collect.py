import copy
import json

import pytest

import evidence_c1_h1_collect as collect
from test_t02_bfcl import FakeActor, FakeBindings, response, task


def history(tmp_path):
    gp = {"G": "record_bound", "R": 1, "set_selector": "candidate_rule"}
    path = tmp_path / "gp.json"
    path.write_text(json.dumps(gp), encoding="utf-8")
    return {"source_history": "H1", "G": "record_bound", "gp_path": str(path),
            "gp_file_sha256": collect._sha(path),
            "gp_payload_sha256": collect.exact._json_digest(gp)}


class CallActor(FakeActor):
    def propose_alternative(self):
        raise AssertionError("Risk calibration must not invoke an A2 selector")

    def hold(self, payload):
        state = super().hold(payload)
        state.update(draft_kind="call", draft={"text": "", "parse_ok": True,
                     "tool_calls": response("c1")["tool_calls"]})
        return state

    def submit_held(self, candidate_ids):
        assert not candidate_ids
        self.held = False
        return response("c1")


def test_a0_stops_at_current_turn_and_keeps_exact_restore_evidence(tmp_path):
    binding = history(tmp_path)
    actor, official = CallActor(), FakeBindings()
    entry, gold = task()
    entry["question"].append([{"role": "user", "content": "second turn"}])
    gold.append(["c2()"])
    env = collect.exact.BFCLTaskEnvironment(entry, gold, bindings=official)
    adapter = collect.H1RiskAdapter(frozen_policy={"gp_file_sha256": binding["gp_file_sha256"]},
                                   artifact_dir=tmp_path / "evidence")
    states = adapter.discover_task(env, actor, max_states=1)
    row = adapter.run_a0_turn(states[0], history_binding=binding, evidence_mode="synthetic_fixture")
    outcome = json.loads(open(row["evidence"]["official_outcome"]["path"]).read())
    assert row["c1_risk_label"] == 0
    assert outcome["candidate_ids"] == []
    assert outcome["restore_receipt"]["restored"] is True
    assert outcome["turn_success"] is True
    assert len(outcome["model_result_prefix"]) == 1
    assert outcome["task_success"] is None
    assert actor.generated == 1
    assert not adapter._states and actor.closed


def test_failed_previous_turn_retains_unknown_risk(tmp_path):
    binding = history(tmp_path)
    entry, gold = task()
    actor = FakeActor()
    env = collect.exact.BFCLTaskEnvironment(entry, gold, bindings=FakeBindings())
    adapter = collect.H1RiskAdapter(frozen_policy={}, artifact_dir=tmp_path / "evidence")
    state = adapter.discover_task(env, actor, max_states=1)[0]
    # Simulate the same snapshot at a state with a known invalid earlier prefix.
    state["previous_turn_valid"] = False
    adapter._states[state["state_id"]]["state"] = copy.deepcopy(state)
    row = adapter.run_a0_turn(state, history_binding=binding, evidence_mode="synthetic_fixture")
    assert row["c1_risk_label"] is None and row["c1_label_status"] == "unknown"


def test_official_checker_objects_are_normalized_before_persistence(tmp_path):
    sentinel = object()
    class ObjectBindings(FakeBindings):
        def score_turn_prefix(self, *args):
            return {"valid": True, "details": {1: sentinel, "path": "/example"}}

        @staticmethod
        def make_json_serializable(value):
            return {"valid": value["valid"], "details": {"1": "Directory(example)", "path": "/example"}}

    entry, gold = task()
    actor = CallActor()
    binding = history(tmp_path)
    env = collect.exact.BFCLTaskEnvironment(entry, gold, bindings=ObjectBindings())
    adapter = collect.H1RiskAdapter(frozen_policy={}, artifact_dir=tmp_path / "evidence")
    state = adapter.discover_task(env, actor, max_states=1)[0]
    row = adapter.run_a0_turn(state, history_binding=binding, evidence_mode="synthetic_fixture")
    outcome = json.loads(open(row["evidence"]["official_outcome"]["path"]).read())
    assert row["c1_risk_label"] == 0
    assert outcome["turn_checker_result"]["details"]["1"] == "Directory(example)"


def test_complete_synthetic_collection_is_not_a_production_receipt(tmp_path):
    binding = history(tmp_path)
    entry, gold = task()
    output = tmp_path / "result"
    args = dict(task_ids=[entry["id"]], calibration_groups=["bfcl_pair_3"],
                excluded_groups={"training": ["bfcl_pair_4"], "evaluation": ["bfcl_pair_5"]},
                history_binding=binding, frozen_policy={"gp_file_sha256": binding["gp_file_sha256"]},
                output=output, max_states=1, evidence_mode="synthetic_fixture",
                environment_factory=lambda _: collect.exact.BFCLTaskEnvironment(entry, gold, bindings=FakeBindings()),
                actor_factory=lambda *_: FakeActor())
    result = collect.collect_sources(**args)
    assert result["status"] == "completed" and result["state_count"] == 1
    assert result["evidence_mode"] == "synthetic_fixture"
    assert result["complete_task_T02_branches"] == 0
    assert result["task_receipts"][0]["selected_state_ids_before_outcomes"] == ["live-state-0"]
    with pytest.raises(FileExistsError, match="overwrite"):
        collect.collect_sources(**args)


def test_h0_or_evaluation_group_leak_is_rejected_before_model_calls(tmp_path):
    binding = history(tmp_path)
    args = dict(task_ids=["multi_turn_base_3"], calibration_groups=["bfcl_pair_3"],
                excluded_groups={"training": [], "evaluation": ["bfcl_pair_3"]},
                history_binding=binding, frozen_policy={}, output=tmp_path / "result",
                environment_factory=lambda _: pytest.fail("model work must not start"),
                actor_factory=lambda *_: pytest.fail("model work must not start"))
    with pytest.raises(ValueError, match="independent group"):
        collect.collect_sources(**args)
    args["excluded_groups"]["evaluation"] = []
    binding["source_history"] = "H0"
    with pytest.raises(ValueError, match="frozen H1"):
        collect.collect_sources(**args)


def test_collector_rows_feed_calibrator_without_becoming_production(tmp_path):
    import evidence_c1_h1_calibration as calibration
    from test_evidence_c1_h1_calibration import _artifact

    artifact_path, _ = _artifact(tmp_path / "head.json")
    artifact = json.loads(artifact_path.read_text())
    sample = FakeActor().hold({"decision_key": "fixture"})
    artifact["feature_contract"]["prefill_contract"] = sample["q"]["prefill_contract"]
    artifact["artifact_sha256"] = calibration._SET_MODELS.artifact_sha256(artifact)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_before = artifact_path.read_bytes()
    binding = history(tmp_path)
    binding["h1_source_hashes"] = calibration.H1_SOURCE_HASHES

    def environment(task_id):
        entry, gold = task()
        entry["id"] = task_id
        return collect.exact.BFCLTaskEnvironment(entry, gold, bindings=FakeBindings())

    actors = []
    def actor(task_id, index):
        value = CallActor() if index == 0 else FakeActor()
        hold = value.hold
        def unique(payload):
            state = hold(payload)
            state["state_id"] = f"{task_id}-{payload['decision_key']}"
            return state
        value.hold = unique
        actors.append(value)
        return value

    result = collect.collect_sources(
        task_ids=["multi_turn_base_3", "multi_turn_base_6"],
        calibration_groups=["bfcl_pair_3", "bfcl_pair_6"],
        excluded_groups={"training": ["bfcl_pair_4"], "evaluation": ["bfcl_pair_5"]},
        history_binding=binding, frozen_policy={"gp_file_sha256": binding["gp_file_sha256"]},
        output=tmp_path / "collected", max_states=2, evidence_mode="synthetic_fixture",
        environment_factory=environment, actor_factory=actor)
    assert {row["c1_risk_label"] for row in result["rows"]} == {0, 1}
    dataset = tmp_path / "collected/dataset.json"
    receipt = calibration.build_calibration_receipt(artifact_path, dataset)
    assert receipt["production_eligible"] is False
    assert receipt["selected_threshold"] == 0.9
    assert artifact_path.read_bytes() == artifact_before
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        calibration.verify_calibration_receipt(receipt_path, artifact_path=artifact_path, input_path=dataset)


def test_failed_selected_state_consumes_cap_without_outcome_based_replacement(monkeypatch, tmp_path):
    binding = history(tmp_path)
    actor = FakeActor()
    started = []
    def environment(task_id):
        started.append(task_id)
        entry, gold = task()
        entry["id"] = task_id
        return collect.exact.BFCLTaskEnvironment(entry, gold, bindings=FakeBindings())
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic continuation failure")
    monkeypatch.setattr(collect.H1RiskAdapter, "run_a0_turn", failed)
    result = collect.collect_sources(
        task_ids=["multi_turn_base_3", "multi_turn_base_6"],
        calibration_groups=["bfcl_pair_3", "bfcl_pair_6"],
        excluded_groups={"training": [], "evaluation": []},
        history_binding=binding, frozen_policy={"gp_file_sha256": binding["gp_file_sha256"]},
        output=tmp_path / "failure", max_states=1, evidence_mode="synthetic_fixture",
        environment_factory=environment, actor_factory=lambda *_: actor)
    assert started == ["multi_turn_base_3"]
    assert result["status"] == "partial_failed"
    assert result["selected_state_count"] == 1 and result["state_count"] == 0
    assert actor.closed and not actor.snapshots


def collection_spec(tmp_path):
    from evidence_c1_h1_calibration import json_file_binding
    binding = history(tmp_path)
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return json_file_binding(path)
    rows = [{"state_id": f"s{i}", "task_id": f"multi_turn_base_{3 if i < 80 else 6}",
             "task_group_id": f"bfcl_pair_{3 if i < 80 else 6}",
             "split": "train" if i < 80 else "calibration"} for i in range(118)]
    labels = {"schema": "t02-multi-plan-labeled-set-v1", "state_count": 118,
              "rows": rows, "rows_sha256": collect.exact._json_digest(rows)}
    design = {"status": "frozen", "resolved_configs": {"controller": {
        "gp_experiments": json.loads((tmp_path / "gp.json").read_text())}}}
    tasks = ["multi_turn_base_6", "multi_turn_long_context_6"]
    spec = {"schema": "c1-h1-calibration-collection-spec-v1",
            "source_design": save("design.json", design),
            "t02_labels": save("labels.json", labels),
            "t02_summary": save("summary.json", {
                "schema": "t02-streaming-recovery-summary-v1", "phase": "labels_completed",
                "training_allowed": True, "combined_exact_state_count": 118,
                "actual_split_counts": {"train": 80, "calibration": 38},
                "combined_labels_sha256": collect.exact._json_digest(labels)}),
            "task_manifest": save("tasks.json", {"task_ids": ["multi_turn_base_3"] + tasks}),
            "d128_manifest": save("d128.json", {"task_ids": ["multi_turn_base_10"]}),
            "f128_manifest": save("f128.json", {"task_ids": ["multi_turn_base_11"]}),
            "history_binding": binding, "max_states": 38, "automatic_retries": 0,
            "task_ids": collect.exact.round_robin_family_tasks(tasks, {task: "bfcl_pair_6" for task in tasks})}
    spec["frozen_policy"] = {"design_sha256": spec["source_design"]["file_sha256"],
                             "gp_file_sha256": binding["gp_file_sha256"]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_collection_spec_binds_completed_labels_and_source_order(tmp_path):
    path, spec = collection_spec(tmp_path)
    resolved = collect.load_collection_spec(path)
    assert resolved["calibration_groups"] == ["bfcl_pair_6"]
    assert resolved["excluded_groups"]["training"] == ["bfcl_pair_3"]
    spec["task_ids"].reverse()
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="source order"):
        collect.load_collection_spec(path)


def test_collection_spec_rejects_partial_labels_even_with_rehashed_reference(tmp_path):
    from evidence_c1_h1_calibration import json_file_binding
    path, spec = collection_spec(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels = json.loads(labels_path.read_text())
    labels["rows"].pop()
    labels["rows_sha256"] = collect.exact._json_digest(labels["rows"])
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    spec["t02_labels"] = json_file_binding(labels_path)
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="completed T02"):
        collect.load_collection_spec(path)


def test_collection_spec_rejects_evaluation_overlap(tmp_path):
    from evidence_c1_h1_calibration import json_file_binding
    path, spec = collection_spec(tmp_path)
    evaluation = tmp_path / "d128.json"
    evaluation.write_text(json.dumps({"task_ids": ["multi_turn_miss_func_6"]}), encoding="utf-8")
    spec["d128_manifest"] = json_file_binding(evaluation)
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap evaluation"):
        collect.load_collection_spec(path)
