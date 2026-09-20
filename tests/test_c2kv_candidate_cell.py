from __future__ import annotations

import hashlib
import sys
import types

import pytest

from generality import candidate_cell


def source_cell(tmp_path):
    return {
        "cell_id": "bfcl_base__c2kv__K0__compression_full_budget",
        "cell_dir": str(tmp_path / "bfcl_base" / "c2kv" / "K0" / "compression_full_budget"),
        "backend": "c2kv", "benchmark": "bfcl", "benchmark_key": "bfcl_base",
        "condition": "compression_full_budget", "ratio": 4,
        "working_point": "K0", "task_ids": ["multi_turn_base_0"],
        "sglang_backend_url": None,
    }


@pytest.mark.parametrize("variant", candidate_cell.VARIANTS)
def test_candidate_cell_is_explicit_ratio8_and_isolated(tmp_path, variant):
    source = source_cell(tmp_path)
    result = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200")
    assert result["ratio"] == 8
    assert result["candidate_algorithm"] == variant
    if variant in candidate_cell.VERIFIED_VARIANTS:
        assert result["threshold"] == 0.5
        assert result["candidate_protocol"] == candidate_cell.VERIFIED_VERSION
        assert result["proof_registry_version"] == candidate_cell.PROOF_REGISTRY_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v4"
    elif variant in candidate_cell.GOAL_VARIANTS:
        assert result["threshold"] == 0.5
        assert result["candidate_protocol"] == candidate_cell.GOAL_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v3"
    elif variant in candidate_cell.REPAIR_VARIANTS:
        assert "threshold" not in result
        assert result["candidate_protocol"] == candidate_cell.REPAIR_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v2"
    else:
        assert result["threshold"] == 0.5
        assert result["schema"] == "c2kv-generality-candidate-cell-v1"
    assert result["candidate_source_cell_id"] == source["cell_id"]
    assert result["candidate_budget_source"] == "working_point.common_cap_bytes"
    assert result["cell_dir"].endswith(f"candidate_algorithms/{variant}") or result["cell_dir"].endswith(f"candidate_algorithms\\{variant}")
    assert result["cell_dir"] != source["cell_dir"]
    assert source["ratio"] == 4 and source["sglang_backend_url"] is None


@pytest.mark.parametrize("change", [
    {"benchmark_key": "bfcl_long_context"},
    {"benchmark": "acon_appworld", "benchmark_key": "appworld"},
    {"condition": "tracer_history"},
    {"ratio": 8},
])
def test_candidate_cell_rejects_unselected_source_panels(tmp_path, change):
    source = source_cell(tmp_path) | change
    with pytest.raises(ValueError):
        candidate_cell.candidate_cell_from_source(
            source, "static_t02", "http://127.0.0.1:36200")


def test_candidate_controller_binds_pinned_artifact_and_strips_legacy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "risk.json"
    artifact_path.write_text('{"model_kind":"c1_risk_logistic"}', encoding="utf-8")
    monkeypatch.setattr(
        candidate_cell, "RISK_ARTIFACT_SHA256",
        hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), "goal_rescue", "http://127.0.0.1:36200")
    cell["checkpoint"] = str(checkpoint)
    base = {"latest_complete_tool_protection": "budgeted",
            "observed_entity_slot_policy": "same-event-bridge-only-v1",
            "gp_experiments": {}, "post_draft_recovery": {}, "d3_hybrid_recovery": True}
    bound = {"model_kind": "c1_risk_logistic", "bound": True}
    controller, receipt = candidate_cell.controller_with_binding(
        cell, base_controller=base,
        selected={"checkpoint_selection": {"config_sha256":
                  hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()}},
        risk_artifact_path=artifact_path,
        bind_risk_artifact=lambda artifact, path: (bound, {"checkpoint": str(path)}))
    assert controller["candidate_algorithm"] == {
        "variant": "goal_rescue", "risk_artifact": bound, "risk_threshold": 0.5}
    assert controller["observed_entity_slot_policy"] == base["observed_entity_slot_policy"]
    assert not {"gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"} & controller.keys()
    assert receipt["checkpoint"] == str(checkpoint)
    assert "candidate_algorithm" not in base


@pytest.mark.parametrize("variant", candidate_cell.REPAIR_VARIANTS)
def test_repair_candidate_keeps_c1000_without_loading_t02(tmp_path, variant):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    cell["checkpoint"] = str(checkpoint)
    base = {"view_mode": "native_s0", "gp_experiments": {},
            "post_draft_recovery": {}, "d3_hybrid_recovery": True}
    controller, receipt = candidate_cell.controller_with_binding(
        cell, base_controller=base,
        selected={"checkpoint_selection": {"config_sha256":
                  hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()}},
        risk_artifact_path=tmp_path / "absent-risk-artifact.json",
        bind_risk_artifact=lambda *_: pytest.fail("T02 binding must not run"))
    assert controller == {"view_mode": "native_s0",
                          "candidate_algorithm": {"variant": variant}}
    assert receipt is None
    assert "candidate_algorithm" not in base


@pytest.mark.parametrize("variant", candidate_cell.GOAL_VARIANTS + candidate_cell.VERIFIED_VARIANTS)
def test_goal_candidate_binds_frozen_t02_with_new_resume_identity(tmp_path, monkeypatch, variant):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "risk.json"
    artifact_path.write_text('{"model_kind":"c1_risk_logistic"}', encoding="utf-8")
    monkeypatch.setattr(candidate_cell, "RISK_ARTIFACT_SHA256",
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    cell["checkpoint"] = str(checkpoint)
    base = {"view_mode": "native_s0", "gp_experiments": {},
            "post_draft_recovery": {}, "d3_hybrid_recovery": True}
    bound = {"model_kind": "c1_risk_logistic", "bound": True}
    controller, receipt = candidate_cell.controller_with_binding(
        cell, base_controller=base,
        selected={"checkpoint_selection": {"config_sha256":
                  hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()}},
        risk_artifact_path=artifact_path,
        bind_risk_artifact=lambda artifact, path: (bound, {"checkpoint": str(path)}))
    expected = {"variant": variant, "risk_artifact": bound, "risk_threshold": 0.5}
    if variant in candidate_cell.VERIFIED_VARIANTS:
        expected["proof_registry_version"] = candidate_cell.PROOF_REGISTRY_VERSION
    assert controller == {"view_mode": "native_s0", "candidate_algorithm": expected}
    assert receipt == {"checkpoint": str(checkpoint)}
    assert cell["cell_id"].endswith("__candidate_" + variant)
    assert cell["candidate_protocol"] == (candidate_cell.VERIFIED_VERSION
        if variant in candidate_cell.VERIFIED_VARIANTS else candidate_cell.GOAL_VERSION)
    assert "candidate_algorithm" not in base


@pytest.mark.parametrize("variant", ("goal_pending",) + candidate_cell.VERIFIED_VARIANTS)
def test_goal_candidate_requires_shadow_feature_launcher_config(tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    monkeypatch.setattr(c2kv_cell.current, "load_config", lambda: {
        "route": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "decode_strategy": "incremental", "prefill_chunk_size": 256,
    }, raising=False)
    common = {
        "python_sgl": "python", "checkpoint": "/checkpoint", "cell_id": "candidate-test",
        "model_name": "candidate-test", "benchmark": "bfcl", "ratio": 8,
        "caps": {"max_completion_tokens": 128, "generation_attempts_per_task": 2,
                 "extraction_calls_per_task": 4, "task_timeout": 60},
        "eval_policy_path": "/eval.json", "controller_path": "/controller.json",
        "sglang_backend_url": "http://127.0.0.1:36200",
        "condition": "candidate_algorithm",
    }
    for selected in (variant, "goal_rescue", "no_progress"):
        command = c2kv_cell.server_command(
            {**common, "candidate_algorithm": selected}, ["task-1"], tmp_path, 36300)
        assert ("--shadow-feature-config" in command) is (selected == variant)


def test_candidate_policy_uses_same_common_cap_without_recovery_reserve(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality.c2kv_cell import build_eval_policy

    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), "dependency_first", "http://127.0.0.1:36200")
    policy = build_eval_policy(cell, {"working_points": {"K0": {
        "history_allowance_bytes": 100, "common_cap_bytes": 300}}})["policy"]
    assert policy["history_budget_bytes"] == 300
    assert policy["workspace_budget_bytes"] == 300
    assert "recovery_history_bytes" not in policy
    assert "recovery_workspace_bytes" not in policy


def test_candidate_cli_selects_variant_before_any_inference(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    seen = []
    monkeypatch.setattr(c2kv_cell, "load_cell", lambda path: source_cell(tmp_path))
    monkeypatch.setattr(c2kv_cell, "_write_bfcl_completion",
                        lambda cell, task_ids: seen.append((cell, task_ids)) or {"valid_count": 0})
    monkeypatch.setattr(c2kv_cell, "completion_receipt", lambda completion: completion)
    rc = c2kv_cell.main([
        "--cell", str(tmp_path / "source.json"),
        "--budgets", str(tmp_path / "budgets.json"),
        "--candidate-algorithm", "turn_c1",
        "--sglang-backend-url", "http://127.0.0.1:36200",
        "--audit-results-only",
    ])
    assert rc == 0 and len(seen) == 1
    assert seen[0][0]["candidate_algorithm"] == "turn_c1"
    assert seen[0][0]["ratio"] == 8
    assert seen[0][1] == ["multi_turn_base_0"]
    assert "valid_count" in capsys.readouterr().out


@pytest.mark.parametrize("variant", candidate_cell.VERIFIED_VARIANTS)
def test_verified_cell_rejects_stale_registry_before_checkpoint_or_inference(tmp_path, variant):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    cell["proof_registry_version"] = "stale-proof-rules"
    with pytest.raises(ValueError, match="proof registry"):
        candidate_cell.controller_with_binding(
            cell, base_controller={}, selected={},
            risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)
