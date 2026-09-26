import copy
import hashlib
import json
from pathlib import Path

import pytest

import evidence_eval_proposals as proposals


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _source(tmp_path: Path, method: str = "E07") -> tuple[Path, Path]:
    spec = proposals.METHOD_SPECS[method]
    source = tmp_path / f"source_{method}"
    lane = source / "lanes" / spec["source_lane"]
    runtime = lane / "runtime"
    (runtime / "configs").mkdir(parents=True)
    (runtime / "benchmarks/memory_runtime/recovery").mkdir(parents=True)
    (source / "history_system").mkdir(parents=True)
    (source / "sglang").mkdir(parents=True)
    (source / "sglang/dummy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "evidence_eval.py").write_text(
        "def verify_package(package, require_sglang=True):\n"
        "    return {'status': 'passed'}\n"
        "def _assert_lane_free(lane):\n"
        "    return {'status': 'free'}\n"
        "def run_lane(package, lane):\n"
        "    return 0\n",
        encoding="utf-8",
    )
    runner = "def run():\n" + proposals.RUNNER_STOP_BLOCK
    (source / "history_system/runner.py").write_text(runner, encoding="utf-8")
    controller = {"gp_experiments": {
        "set_selector": "risk" if method == "E07" else "gain_turn",
        "local_models": {"reranker": {"batch_size": 1}},
        "semantic_query_overflow_policy": "task_head_tail_preserve_draft_v1",
        "export_selection_state": False,
        "selector_threshold": 0.5,
        "gain_delta": 0.0,
        "selection_protocol": "evidence_sets_v1",
        "G": "current", "R": 1, "candidate_limit": 8, "K": 4,
        "retrieval_limit": 24, "retrieval_route_limit": 24,
        "fallback_unit": None, "recovery_reserve_tokens": 0,
    }}
    _save(runtime / "configs/controller.json", controller)
    (runtime / "benchmarks/memory_runtime/recovery/experiment.py").write_text(
        "OLD = True\n", encoding="utf-8")
    _save(lane / "lane.json", {
        "name": spec["source_lane"], "selector": "old", "physical_device": 0,
        "engine_port": 30000, "task_port_base": 30100,
        "task_ports": list(range(30100, 30120)), "model_smokes": [],
    })
    _save(lane / "design.json", {
        "schema": "a-history-system-candidate-design-v1",
        "candidate_id": "old", "run_id_template": "old",
        "task_ids": [f"multi_turn_base_{index}" for index in range(20)],
        "task_manifest_sha256": "0" * 64,
        "limits": {"tasks": 20}, "runtime": {"sglang_backend_url": "old"},
        "resolved_configs": {"controller": controller},
        "search_contract": {"history": "H0", "recovery_attempts": 1},
        "automatic_reruns": 0,
    })
    artifact = {"schema": "test-artifact", "model_kind": spec["artifact_kind"],
                "fit": {"ok": True}}
    if spec["requires_proposal_protocol"]:
        artifact["proposal_protocol"] = proposals.PROPOSAL_PROTOCOL
    artifact_path = lane / "trained_artifact.json"
    _save(artifact_path, artifact)
    _save(source / "launch_contract.json", {
        "automatic_reruns": 0,
        "required_environment": proposals.STRICT_ENVIRONMENT,
    })
    _save(source / "sglang_files.json", {
        "schema": "test-sglang", "file_count": 1,
        "files": {"sglang/dummy.py": proposals._sha(source / "sglang/dummy.py")},
    })
    _save(source / "static_files.json", proposals._manifest(
        source, "test-static", skip_sglang=True, excluded=("static_files.json",)))
    return source, artifact_path


def _schedule(method: str, port: int = 31000):
    return [{
        "shard_id": f"{method}_part{part}", "device": device,
        "engine_port": port + part * 30,
        "task_port_base": port + part * 30 + 1,
    } for part, device in enumerate(proposals.DEVICES)]


def _fixture(tmp_path: Path, monkeypatch, method: str = "E07"):
    monkeypatch.setattr(
        proposals, "_runtime_bundle_validation",
        lambda runtime, artifact, controller, selected: {
            "status": "passed", "selector": proposals.METHOD_SPECS[selected]["selector"],
            "artifact_kind": proposals.METHOD_SPECS[selected]["artifact_kind"],
        })
    tasks = [f"multi_turn_base_{index}" for index in range(128)]
    manifest = tmp_path / "D128.json"
    _save(manifest, {
        "schema": "a-history-system-task-manifest-v1", "manifest_id": "R2D128",
        "stage": "development_search", "task_ids": tasks,
        "fixed_denominator": 128, "automatic_reruns": 0,
    })
    monkeypatch.setattr(proposals, "EXPECTED_D128_SHA256", proposals._sha(manifest))
    source, artifact = _source(tmp_path, method)
    runtime = tmp_path / "new_runtime"
    overlay = runtime / "benchmarks/memory_runtime/recovery/experiment.py"
    overlay.parent.mkdir(parents=True)
    overlay.write_text("NEW = True\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    methods = {
        method: {"source_package": str(source),
                 "source_lane": proposals.METHOD_SPECS[method]["source_lane"],
                 "artifact_path": str(artifact), "shards": _schedule(method)},
    }
    if method != "E09":
        # Deliberately unresolved: an E07-only build must ignore it.
        methods["E09"] = {"source_package": "pending", "artifact_path": "pending"}
    _save(catalog, {"schema": proposals.SOURCE_CATALOG_SCHEMA, "methods": methods})
    return tasks, manifest, source, artifact, runtime, catalog


def _build(tmp_path: Path, monkeypatch, method: str = "E07"):
    tasks, manifest, source, artifact, runtime, catalog = _fixture(
        tmp_path, monkeypatch, method)
    design_path = tmp_path / "design.json"
    receipt = proposals.build_design(
        design_path, source_catalog_path=catalog,
        canonical_manifest_path=manifest, runtime_source_root=runtime,
        overlay_files=["benchmarks/memory_runtime/recovery/experiment.py"],
        methods=[method])
    return tasks, design_path, receipt


def test_build_design_derives_hashes_and_ignores_pending_unselected_method(
        tmp_path, monkeypatch):
    tasks, design_path, receipt = _build(tmp_path, monkeypatch)
    design = proposals._read(design_path)
    assert receipt["methods"] == ["E07"]
    assert design["canonical_manifest"]["task_ids"] == tasks
    assert design["new_task_executions"] == 128
    assert design["campaign_task_execution_ceiling"] == 384
    assert design["methods"][0]["runtime_overlay"][0]["source_sha256"]
    assert design["methods"][0]["runtime_overlay"][0]["target_sha256"]


def test_methods_are_an_ordered_unique_subset():
    with pytest.raises(ValueError, match="ordered subset"):
        proposals._normal_methods(["E09", "E07"])
    with pytest.raises(ValueError, match="ordered subset"):
        proposals._normal_methods(["E07", "E07"])


def test_new_gain_artifact_requires_proposal_protocol(tmp_path, monkeypatch):
    _, manifest, _, artifact, runtime, catalog = _fixture(tmp_path, monkeypatch, "E09")
    value = proposals._read(artifact)
    value.pop("proposal_protocol")
    _save(artifact, value)
    source = artifact.parents[2]
    _save(source / "static_files.json", proposals._manifest(
        source, "test-static", skip_sglang=True, excluded=("static_files.json",)))
    with pytest.raises(ValueError, match="not trained for the proposal protocol"):
        proposals.build_design_document(
            source_catalog_path=catalog, canonical_manifest_path=manifest,
            runtime_source_root=runtime,
            overlay_files=["benchmarks/memory_runtime/recovery/experiment.py"],
            methods=["E09"])


def test_runner_patch_is_exact_and_removes_stop(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text("def run():\n" + proposals.RUNNER_STOP_BLOCK, encoding="utf-8")
    patched = proposals._patched_runner_bytes(runner).decode()
    assert proposals.RUNNER_CONTINUE_MARKER in patched
    assert "return 6" not in patched
    assert "dispatch_stop_reason" not in patched
    bad = tmp_path / "bad.py"
    bad.write_text("def run():\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact first-failure stop seam"):
        proposals._patched_runner_bytes(bad)


def test_controller_preserves_frozen_payload_and_model_policy():
    source = {"gp_experiments": {
        "set_selector": "gain_turn",
        "selector_artifact": {"old": True},
        "proposal_protocol": None,
        "semantic_query_overflow_policy": "task_head_tail_preserve_draft_v1",
        "export_selection_state": False,
        "local_models": {"reranker": {"batch_size": 1, "model": "frozen"}},
        "selector_threshold": 0.5,
        "gain_delta": 0.0,
        "candidate_limit": 8,
        "K": 4,
        "selection_protocol": "evidence_sets_v1",
        "G": "current", "R": 1,
        "retrieval_limit": 24, "retrieval_route_limit": 24,
        "fallback_unit": None, "recovery_reserve_tokens": 0,
    }}
    original = copy.deepcopy(source)
    artifact = {"model_kind": "c4_gain_turn",
                "proposal_protocol": proposals.PROPOSAL_PROTOCOL}
    controller, gp = proposals._controller(source, "E09", artifact)
    assert source == original
    assert gp["set_selector"] == "gain_turn_proposals"
    assert gp["selector_artifact"] == artifact
    assert gp["proposal_protocol"] == proposals.PROPOSAL_PROTOCOL
    for key in ("semantic_query_overflow_policy", "export_selection_state",
                "local_models", "selector_threshold", "gain_delta",
                "candidate_limit", "K", "selection_protocol", "G", "R",
                "retrieval_limit", "retrieval_route_limit", "fallback_unit",
                "recovery_reserve_tokens"):
        assert gp[key] == original["gp_experiments"][key]
    assert controller["gp_experiments"] == gp


def test_controller_rejects_silent_source_policy_drift():
    source = {"gp_experiments": {
        "semantic_query_overflow_policy": "error",
        "export_selection_state": False,
        "local_models": {"reranker": {"batch_size": 1}},
        "selector_threshold": 0.5,
    }}
    with pytest.raises(ValueError, match="source model/payload policy differs"):
        proposals._controller(source, "E07", {"model_kind": "c1_risk_logistic"})


def test_prepare_verifies_exact_d128_partition_overlay_artifact_and_runner(
        tmp_path, monkeypatch):
    tasks, design_path, _ = _build(tmp_path, monkeypatch)
    package = tmp_path / "package"
    receipt = proposals.prepare(package, design_path)
    assert receipt["verification"]["new_task_executions"] == 128
    contract = proposals._read(package / "proposal_contract.json")
    assert [row["task_budget"] for row in contract["shards"]] == list(
        proposals.SHARD_SIZES)
    assert {task for row in contract["shards"] for task in row["task_ids"]} == set(tasks)
    lane = package / "shards/E07_part0/lanes/E07_part0"
    assert proposals._read(lane / "gp.json")["set_selector"] == "risk_source_proposal"
    assert proposals._read(lane / "design.json")["search_contract"][
        "original_d20_reuse_allowed"] is False
    assert proposals.RUNNER_CONTINUE_MARKER in (
        package / "shards/E07_part0/history_system/runner.py").read_text()
    assert proposals.verify_package(package)["status"] == "passed"


def test_verify_rejects_tampered_overlay(tmp_path, monkeypatch):
    _, design_path, _ = _build(tmp_path, monkeypatch)
    package = tmp_path / "package"
    proposals.prepare(package, design_path)
    target = (package / "shards/E07_part0/lanes/E07_part0/runtime/"
              "benchmarks/memory_runtime/recovery/experiment.py")
    target.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen proposal package changed"):
        proposals.verify_package(package)


def test_run_refuses_existing_state_before_resource_acquisition(tmp_path, monkeypatch):
    _, design_path, _ = _build(tmp_path, monkeypatch)
    package = tmp_path / "package"
    proposals.prepare(package, design_path)
    expected = proposals.authorization_requirements(package)
    authorization = {**expected, "status": "authorized", "launch_authorized": True,
                     "authorized_by": "root"}
    auth_path = tmp_path / "auth.json"
    _save(auth_path, authorization)
    (package / "shards/E07_part0/lanes/E07_part0/run").mkdir()
    with pytest.raises(FileExistsError, match="Refusing rerun"):
        proposals.run_shard(package, "E07_part0", auth_path)


def _official(correct=1):
    return {"scored": True, "n_total": 1, "n_scored": 1,
            "correct_count": correct, "semantic_score": float(correct)}


def test_summary_separates_normal_unknown_and_pending(tmp_path, monkeypatch):
    _, design_path, _ = _build(tmp_path, monkeypatch)
    package = tmp_path / "package"
    proposals.prepare(package, design_path)
    row = proposals._read(package / "proposal_contract.json")["shards"][0]
    lane = package / "shards/E07_part0/lanes/E07_part0"
    normal, failed = row["task_ids"][:2]
    _save(lane / "results/stage_manifest.json", {
        "candidate_id": "proposal_h0_d128_e07",
        "task_outcomes": [
            {"task_id": normal, "outcome": "official_completed",
             "runtime_completed": True, "worker_returncode": 0, "server_returncode": 0},
            {"task_id": failed, "outcome": "runtime_failure_in_denominator",
             "in_fixed_denominator": True, "runtime_completed": False,
             "worker_returncode": 0, "server_returncode": 1},
        ],
    })
    _save(lane / f"results/task_shards/{normal}/server/final.json", {"status": "ok"})
    _save(lane / f"results/task_shards/{normal}/bfcl/official_summary.json", _official())
    summary = proposals.summarize(package)
    method = summary["methods"][0]
    assert {key: method[key] for key in
            ("normal", "audited_failed", "unknown", "pending")} == {
                "normal": 1, "audited_failed": 0, "unknown": 1, "pending": 126}
    assert method["status"] == "incomplete"


def test_summary_recomputes_audit_instead_of_trusting_classification(
        tmp_path, monkeypatch):
    _, design_path, _ = _build(tmp_path, monkeypatch)
    package = tmp_path / "package"
    proposals.prepare(package, design_path)
    audit_path = tmp_path / "audit.json"
    helper = proposals.HERE / "evidence_d128_failure_audit.py"
    _save(audit_path, {
        "schema": proposals.AUDIT_SCHEMA,
        "audit_helper": proposals._binding(helper),
        "package": {
            "path": str(package.resolve()), "stage": "H0_R1",
            "static_files": {"sha256": proposals._sha(package / "static_files.json")},
            "contract": {"sha256": proposals._sha(package / "proposal_contract.json")},
        },
        "failures": [{"classification": {"status": "audited_in_contract"}}],
    })
    monkeypatch.setattr(proposals, "collect_failure_audit",
                        lambda package, **kwargs: {"failures": []})
    with pytest.raises(ValueError, match="differ from current raw runtime evidence"):
        proposals.summarize(package, failure_audit=audit_path)


def test_prepare_rejects_changed_source_catalog(tmp_path, monkeypatch):
    _, design_path, _ = _build(tmp_path, monkeypatch)
    design = proposals._read(design_path)
    Path(design["source_catalog"]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source catalog changed"):
        proposals.prepare(tmp_path / "package", design_path)
