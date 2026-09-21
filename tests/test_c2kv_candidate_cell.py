from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

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


def toolsandbox_source_cell(tmp_path):
    source = source_cell(tmp_path)
    task_ids = [f"scenario_{index}_3_distraction_tools" for index in range(129)]
    source.update(
        cell_id="toolsandbox__c2kv__K0__compression_full_budget",
        cell_dir=str(tmp_path / "toolsandbox" / "c2kv" / "K0" /
                     "compression_full_budget"),
        benchmark="toolsandbox", benchmark_key="toolsandbox",
        task_ids=task_ids, heldout_task_ids=task_ids[:],
        budget_bytes={"K": 100, "R_max": 200, "B": 300},
        budget_tokens={"K": 768, "R": 256, "B": 1024},
    )
    return source


@pytest.mark.parametrize("variant", candidate_cell.VARIANTS)
def test_candidate_cell_is_explicit_ratio8_and_isolated(tmp_path, variant):
    source = source_cell(tmp_path)
    result = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200")
    assert result["ratio"] == 8
    assert result["candidate_algorithm"] == variant
    if variant in candidate_cell.C1_V2_VARIANTS:
        assert result["threshold"] == 0.5
        assert result["candidate_protocol"] == candidate_cell.C1_V2_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v7"
        assert {key: result[key] for key in candidate_cell.c1_v2_contract(variant)} == (
            candidate_cell.c1_v2_contract(variant))
    elif variant in candidate_cell.STATIC_EXTENSION_VARIANTS:
        assert result["threshold"] == 0.5
        assert result["candidate_protocol"] == candidate_cell.STATIC_EXTENSION_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v6"
        assert {key: result[key] for key in candidate_cell.static_contract(variant)} == (
            candidate_cell.static_contract(variant))
    elif variant in candidate_cell.STATIC_VARIANTS:
        assert result["threshold"] == 0.5
        assert result["candidate_protocol"] == candidate_cell.STATIC_VERSION
        assert result["schema"] == "c2kv-generality-candidate-cell-v5"
        assert {key: result[key] for key in candidate_cell.static_contract(variant)} == (
            candidate_cell.static_contract(variant))
    elif variant in candidate_cell.VERIFIED_VARIANTS:
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
    if variant not in (candidate_cell.STATIC_VARIANTS
                       + candidate_cell.STATIC_EXTENSION_VARIANTS
                       + candidate_cell.C1_V2_VARIANTS):
        assert "initial_view" not in result and "recovery_backbone" not in result
    elif variant in candidate_cell.C1_V2_VARIANTS:
        assert (result["proof_registry_version"]
                == candidate_cell.PROOF_REGISTRY_VERSION)
    elif "proof_registry_version" not in candidate_cell.static_contract(variant):
        assert "proof_registry_version" not in result
    assert result["candidate_source_cell_id"] == source["cell_id"]
    assert result["candidate_budget_source"] == "working_point.common_cap_bytes"
    assert result["cell_dir"].endswith(f"candidate_algorithms/{variant}") or result["cell_dir"].endswith(f"candidate_algorithms\\{variant}")
    assert result["cell_dir"] != source["cell_dir"]
    assert source["ratio"] == 4 and source["sglang_backend_url"] is None


def test_existing_static_contracts_keep_their_exact_fields():
    expected_view = {"policy": "static_gist",
                     "version": candidate_cell.STATIC_INITIAL_VIEW_VERSION}
    assert candidate_cell.static_contract("goal_static") == {
        "recovery_backbone": "goal_rescue", "initial_view": expected_view}
    assert candidate_cell.static_contract("pending_static") == {
        "recovery_backbone": "goal_pending", "initial_view": expected_view}


def test_legacy_static_t02_contract_is_unchanged(tmp_path):
    result = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), "static_t02", "http://127.0.0.1:36200")
    assert result["schema"] == "c2kv-generality-candidate-cell-v1"
    assert result["ratio"] == 8 and result["threshold"] == 0.5
    assert "candidate_protocol" not in result
    assert "initial_view" not in result
    assert "recovery_backbone" not in result
    assert "commit_policy" not in result


def test_c1_v2_contract_is_explicit_and_keeps_frozen_proof_registry():
    assert candidate_cell.C1_V2_VARIANTS == ("c1_v2_verified",)
    assert candidate_cell.C1_V2_VERSION == "c2kv-c1-v2-verified-v1"
    assert candidate_cell.c1_v2_contract("c1_v2_verified") == {
        "initial_view": {
            "policy": "s0_capacity_fallback",
            "version": "c2kv-s0-capacity-fallback-v1",
        },
        "recovery_backbone": "t02_complete_event",
        "completion_review": False,
        "proof_registry_version": candidate_cell.PROOF_REGISTRY_VERSION,
    }
    with pytest.raises(ValueError, match="Unknown C1 v2"):
        candidate_cell.c1_v2_contract("static_verified_v2")


def test_static_extension_contracts_are_explicit_and_versioned():
    common = {
        "recovery_backbone": "static_t02",
        "initial_view": {"policy": "static_gist",
                         "version": candidate_cell.STATIC_INITIAL_VIEW_VERSION},
    }
    assert candidate_cell.STATIC_EXTENSION_VARIANTS == (
        "static_verified", "static_action_ledger", "static_verified_v2")
    assert candidate_cell.STATIC_EXTENSION_VERSION == "c2kv-static-extension-v1"
    assert candidate_cell.static_contract("static_verified") == {
        **common,
        "commit_policy": "verified_binding",
        "proof_registry_version": candidate_cell.PROOF_REGISTRY_VERSION,
    }
    assert candidate_cell.static_contract("static_action_ledger") == {
        **common,
        "commit_policy": "action_ledger",
        "action_ledger_version": "static-action-ledger-v1",
        "action_rules_version": "action-ledger-rules-v1",
    }
    assert candidate_cell.static_contract("static_verified_v2") == {
        **common,
        "commit_policy": "verified_binding_v2",
        "proof_registry_version": candidate_cell.RELATIONAL_PROOF_REGISTRY_VERSION,
        "base_proof_registry_version": candidate_cell.PROOF_REGISTRY_VERSION,
    }


def test_static_verified_v2_cell_and_controller_bind_the_relational_contract(
        tmp_path, monkeypatch):
    variant = "static_verified_v2"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "risk.json"
    artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        candidate_cell, "RISK_ARTIFACT_SHA256",
        hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path) | {"checkpoint": str(checkpoint)},
        variant, "http://127.0.0.1:36200")
    controller, _ = candidate_cell.controller_with_binding(
        cell, base_controller={"view_mode": "native_s0"},
        selected={"checkpoint_selection": {"config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()).hexdigest()}},
        risk_artifact_path=artifact_path,
        bind_risk_artifact=lambda artifact, path: ({"bound": True}, {}))

    contract = candidate_cell.static_contract(variant)
    assert cell["schema"] == "c2kv-generality-candidate-cell-v6"
    assert cell["candidate_protocol"] == candidate_cell.STATIC_EXTENSION_VERSION
    assert cell["ratio"] == 8 and cell["threshold"] == 0.5
    assert {key: cell[key] for key in contract} == contract
    assert controller["candidate_algorithm"] == {
        "variant": variant,
        "risk_artifact": {"bound": True},
        "risk_threshold": 0.5,
        **contract,
    }


@pytest.mark.parametrize("field", [
    "commit_policy", "proof_registry_version", "base_proof_registry_version",
])
@pytest.mark.parametrize("replacement", [None, "stale-version"])
def test_static_verified_v2_rejects_missing_or_stale_contract_before_checkpoint(
        tmp_path, field, replacement):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), "static_verified_v2", "http://127.0.0.1:36200")
    if replacement is None:
        cell.pop(field)
    else:
        cell[field] = replacement
    with pytest.raises(ValueError, match="matching version and contract"):
        candidate_cell.controller_with_binding(
            cell, base_controller={}, selected={},
            risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)


@pytest.mark.parametrize("variant", candidate_cell.VARIANTS)
def test_toolsandbox_candidate_preserves_source_task_and_budget_identity(tmp_path, variant):
    source = toolsandbox_source_cell(tmp_path)
    result = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200")
    assert result["benchmark"] == result["benchmark_key"] == "toolsandbox"
    assert result["candidate_source_cell_id"] == source["cell_id"]
    assert result["task_ids"] == source["task_ids"]
    assert result["heldout_task_ids"] == source["heldout_task_ids"]
    assert result["budget_bytes"] == source["budget_bytes"]
    assert result["budget_tokens"] == source["budget_tokens"]
    assert result["ratio"] == 8
    assert "history_budget_tokens" not in result
    assert result["candidate_budget_source"] == "working_point.common_cap_bytes"
    assert result["cell_id"] == source["cell_id"] + f"__candidate_{variant}"
    assert Path(result["cell_dir"]).parts[-2:] == ("candidate_algorithms", variant)
    assert source["ratio"] == 4 and source["sglang_backend_url"] is None


@pytest.mark.parametrize("variant, backbone", [
    ("goal_verified_static", "goal_verified"),
    ("pending_verified_static", "pending_verified"),
])
def test_verified_static_contract_keeps_both_components(variant, backbone):
    assert candidate_cell.static_contract(variant) == {
        "recovery_backbone": backbone,
        "initial_view": {"policy": "static_gist",
                         "version": candidate_cell.STATIC_INITIAL_VIEW_VERSION},
        "proof_registry_version": candidate_cell.PROOF_REGISTRY_VERSION,
    }


def _checkpoint_with_geometry(path, *, declared_bytes=147456):
    from controller_runtime.benchmarks.memory_runtime.event_native import EXPECTED_PROFILE

    path.mkdir()
    config = {
        **EXPECTED_PROFILE,
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "history_memory_supported_ratios": [4, 8],
        "gist_token_id": 1, "vocab_size": 10,
        "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128,
        "history_memory_policy": {"kv_bytes_per_token": declared_bytes},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("variant", ("goal_pending",) +
                         candidate_cell.VERIFIED_STATIC_VARIANTS +
                         candidate_cell.STATIC_EXTENSION_VARIANTS)
def test_explicit_native_budget_has_independent_cell_policy_and_profile(
        tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    checkpoint = tmp_path / "checkpoint"
    _checkpoint_with_geometry(checkpoint)
    source = source_cell(tmp_path) | {
        "checkpoint": str(checkpoint), "model_name": "gen_c2kv_K0_compression_full_budget",
    }
    legacy = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200")
    explicit = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200", 768)
    another = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200", 1024)
    assert legacy["cell_id"] == source["cell_id"] + "__candidate_" + variant
    assert "history_budget_tokens" not in legacy
    assert explicit["cell_id"] == legacy["cell_id"] + "__b768"
    assert another["cell_id"] == legacy["cell_id"] + "__b1024"
    assert len({legacy["cell_dir"], explicit["cell_dir"], another["cell_dir"]}) == 3
    assert explicit["model_name"].endswith("__candidate_" + variant + "__b768")
    assert explicit["candidate_budget_source"] == "explicit.native_history_budget_tokens"

    monkeypatch.setattr(c2kv_cell, "_controller_with_binding", lambda cell: ({}, None))
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 100, "common_cap_bytes": 300}}}
    prepared = c2kv_cell.prepare_cell_files(explicit, budgets)
    profile = prepared["native_history_budget"]
    policy_path = Path(prepared["cell_dir"]) / "eval_policy.json"
    policy = json.loads(policy_path.read_text())
    frozen_cell = json.loads((Path(prepared["cell_dir"]) / "cell.json").read_text())
    assert profile["schema"] == "c2kv-native-history-budget-override-v1"
    assert profile["requested_tokens"] == 768
    assert profile["kv_bytes_per_token"] == 147456
    assert profile["history_budget_bytes"] == 768 * 147456
    assert profile["workspace_budget_bytes"] == 768 * 147456
    assert profile["base_history_budget_bytes"] == 300
    assert policy["policy_id"] == "generality-" + explicit["cell_id"]
    assert policy["policy"]["history_budget_bytes"] == 768 * 147456
    assert policy["policy"]["workspace_budget_bytes"] == 768 * 147456
    assert profile["eval_policy"] == policy
    assert frozen_cell["native_history_budget"] == profile
    if variant in candidate_cell.VERIFIED_STATIC_VARIANTS:
        assert frozen_cell["proof_registry_version"] == candidate_cell.PROOF_REGISTRY_VERSION
    assert profile["override_eval_policy_sha256"] == hashlib.sha256(
        policy_path.read_bytes()).hexdigest()
    assert profile["override_policy_sha256"] == c2kv_cell._json_sha256(policy)
    assert profile["base_eval_policy_source"] == "virtual_working_point_candidate_policy"
    legacy_policy = c2kv_cell.build_eval_policy(legacy, budgets)
    assert profile["base_eval_policy_sha256"] == hashlib.sha256(
        c2kv_cell._json_file_bytes(legacy_policy)).hexdigest()
    from controller_runtime.benchmarks.memory_runtime.event_native_eval_policy import load_eval_policy
    assert load_eval_policy(policy_path) == policy
    monkeypatch.setattr(c2kv_cell.current, "load_config", lambda: {
        "route": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "decode_strategy": "incremental", "prefill_chunk_size": 256,
    }, raising=False)
    launch_cell = prepared | {
        "python_sgl": "python", "caps": {
            "max_completion_tokens": 128, "generation_attempts_per_task": 2,
            "extraction_calls_per_task": 4, "task_timeout": 60},
    }
    command = c2kv_cell.server_command(launch_cell, ["multi_turn_base_0"],
                                       tmp_path / "attempt", 36300)
    assert command[command.index("--eval-policy") + 1] == str(policy_path)
    assert command[command.index("--model-name") + 1] == prepared["model_name"]
    assert explicit["cell_id"] in command[command.index("--run-id") + 1]
    assert legacy_policy["policy"]["history_budget_bytes"] == 300


@pytest.mark.parametrize("budget", [0, -1, True, 768.0, "768"])
def test_candidate_rejects_invalid_explicit_native_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="positive integer"):
        candidate_cell.candidate_cell_from_source(
            source_cell(tmp_path), "goal_pending", "http://127.0.0.1:36200", budget)


def test_geometry_mismatch_rejected_before_writing_explicit_cell(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    checkpoint = tmp_path / "checkpoint"
    _checkpoint_with_geometry(checkpoint, declared_bytes=123)
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path) | {"checkpoint": str(checkpoint)},
        "goal_pending", "http://127.0.0.1:36200", 768)
    with pytest.raises(ValueError, match="different byte contract"):
        c2kv_cell.prepare_cell_files(cell, {"working_points": {"K0": {
            "history_allowance_bytes": 100, "common_cap_bytes": 300}}})
    assert not Path(cell["cell_dir"]).exists()


def test_explicit_budget_cli_keeps_candidate_result_identity(tmp_path, monkeypatch):
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
    args = ["--cell", str(tmp_path / "source.json"),
            "--budgets", str(tmp_path / "budgets.json"),
            "--candidate-algorithm", "goal_pending",
            "--sglang-backend-url", "http://127.0.0.1:36200",
            "--history-budget-tokens", "768", "--audit-results-only"]
    assert c2kv_cell.main(args) == 0
    assert seen[0][0]["cell_id"].endswith("__candidate_goal_pending__b768")
    assert seen[0][0]["history_budget_tokens"] == 768
    with pytest.raises(SystemExit):
        c2kv_cell.main(["--cell", "source.json", "--budgets", "budgets.json",
                        "--history-budget-tokens", "768"])
    with pytest.raises(SystemExit):
        c2kv_cell.main(args[:args.index("--history-budget-tokens")] +
                       ["--history-budget-tokens", "0", "--audit-results-only"])


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


def test_candidate_cell_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown candidate algorithm"):
        candidate_cell.candidate_cell_from_source(
            source_cell(tmp_path), "unknown_static_extension",
            "http://127.0.0.1:36200")


@pytest.mark.parametrize("change", [
    {"benchmark_key": "bfcl_base"},
    {"benchmark": "bfcl"},
    {"backend": "h2o"},
    {"condition": "tracer_history"},
    {"ratio": 8},
])
def test_toolsandbox_candidate_rejects_mismatched_source_identity(tmp_path, change):
    with pytest.raises(ValueError):
        candidate_cell.candidate_cell_from_source(
            toolsandbox_source_cell(tmp_path) | change,
            "pending_verified", "http://127.0.0.1:36200")


def test_toolsandbox_candidate_rejects_explicit_native_budget(tmp_path):
    with pytest.raises(ValueError, match="ToolSandbox candidates use the source B-budget"):
        candidate_cell.candidate_cell_from_source(
            toolsandbox_source_cell(tmp_path), "pending_verified",
            "http://127.0.0.1:36200", 768)


@pytest.mark.parametrize("variant", ("pending_verified", "pending_verified_static"))
def test_toolsandbox_candidate_prepares_proof_budget_and_worker_route(
        tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    checkpoint = tmp_path / "checkpoint"
    _checkpoint_with_geometry(checkpoint)
    artifact_path = tmp_path / "risk.json"
    artifact_path.write_text('{"model_kind":"c1_risk_logistic"}', encoding="utf-8")
    monkeypatch.setattr(candidate_cell, "RISK_ARTIFACT_SHA256",
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    monkeypatch.setattr(c2kv_cell, "RISK_ARTIFACT", artifact_path)
    monkeypatch.setattr(c2kv_cell.evidence_sets, "_base_controller", lambda: {
        "view_mode": "native_s0", "gp_experiments": {},
        "post_draft_recovery": {}, "d3_hybrid_recovery": True,
    }, raising=False)
    monkeypatch.setattr(c2kv_cell, "bind_risk_artifact",
                        lambda artifact, path: (artifact | {"bound": True},
                                                {"checkpoint": str(path)}))
    monkeypatch.setattr(c2kv_cell.current, "load_config", lambda: {
        "checkpoint_selection": {"config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()).hexdigest()},
        "route": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "decode_strategy": "incremental", "prefill_chunk_size": 256,
    }, raising=False)
    source = toolsandbox_source_cell(tmp_path) | {
        "checkpoint": str(checkpoint), "model_name": "toolsandbox-model",
    }
    cell = candidate_cell.candidate_cell_from_source(
        source, variant, "http://127.0.0.1:36200")
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 100, "common_cap_bytes": 300}}}
    prepared = c2kv_cell.prepare_cell_files(cell, budgets)
    cell_dir = Path(prepared["cell_dir"])
    controller = json.loads((cell_dir / "controller.json").read_text())
    policy = json.loads((cell_dir / "eval_policy.json").read_text())
    assert controller["candidate_algorithm"]["variant"] == variant
    assert controller["candidate_algorithm"]["proof_registry_version"] == (
        candidate_cell.PROOF_REGISTRY_VERSION)
    assert controller["candidate_algorithm"]["risk_artifact"]["bound"] is True
    assert not {"gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"} & controller.keys()
    assert prepared["proof_registry_version"] == candidate_cell.PROOF_REGISTRY_VERSION
    assert "native_history_budget" not in prepared
    assert policy["policy"]["history_budget_bytes"] == 300
    assert policy["policy"]["workspace_budget_bytes"] == 300
    assert json.loads((cell_dir / "risk_artifact_binding.json").read_text()) == {
        "checkpoint": str(checkpoint)}
    launch = prepared | {"python_sgl": "python", "python_bench": "bench-python",
                         "benchmark_dir": "/toolsandbox", "caps": {
                             "max_completion_tokens": 128,
                             "generation_attempts_per_task": 2,
                             "extraction_calls_per_task": 4, "task_timeout": 60}}
    task_id = source["task_ids"][0]
    command = c2kv_cell.server_command(launch, [task_id], tmp_path / "attempt", 36300)
    assert command[command.index("--benchmark") + 1] == "toolsandbox"
    assert command[command.index("--source-profile") + 1] == "openai-single-task-v1"
    assert command[command.index("--task-ids") + 1] == task_id
    assert command[command.index("--eval-policy") + 1] == str(cell_dir / "eval_policy.json")
    assert "--shadow-feature-config" in command
    worker = c2kv_cell.toolsandbox_worker_command(
        launch, task_id, "http://127.0.0.1:36300/v1",
        "http://127.0.0.1:36200/v1", tmp_path / "worker")
    assert worker[worker.index("--task-id") + 1] == task_id
    assert worker[worker.index("--model") + 1] == prepared["model_name"]
    summary = c2kv_cell.toolsandbox_score_summary(prepared)
    assert summary["n_total"] == summary["score_denominator"] == 129
    assert summary["pending_task_ids"] == source["task_ids"]


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


def test_c1_v2_controller_preserves_full_s0_and_binds_frozen_contract(
        tmp_path, monkeypatch):
    variant = "c1_v2_verified"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "risk.json"
    artifact_path.write_text('{"model_kind":"c1_risk_logistic"}', encoding="utf-8")
    monkeypatch.setattr(candidate_cell, "RISK_ARTIFACT_SHA256",
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path) | {"checkpoint": str(checkpoint)},
        variant, "http://127.0.0.1:36200")
    base = {
        "view_mode": "native_s0",
        "latest_complete_tool_protection": "budgeted",
        "observed_entity_slot_policy":
            "same-complete-event-reference-bridge-only-v1",
        "gp_experiments": {}, "post_draft_recovery": {},
        "d3_hybrid_recovery": True,
    }
    bound = {"model_kind": "c1_risk_logistic", "bound": True}
    controller, receipt = candidate_cell.controller_with_binding(
        cell, base_controller=base,
        selected={"checkpoint_selection": {"config_sha256":
                  hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()}},
        risk_artifact_path=artifact_path,
        bind_risk_artifact=lambda artifact, path: (
            bound, {"checkpoint": str(path)}))
    assert controller == {
        "view_mode": "native_s0",
        "latest_complete_tool_protection": "budgeted",
        "observed_entity_slot_policy":
            "same-complete-event-reference-bridge-only-v1",
        "candidate_algorithm": {
            "variant": variant, "risk_artifact": bound, "risk_threshold": 0.5,
            **candidate_cell.c1_v2_contract(variant),
        },
    }
    assert receipt == {"checkpoint": str(checkpoint)}
    assert "candidate_algorithm" not in base


@pytest.mark.parametrize("field,replacement", [
    ("candidate_protocol", "c2kv-c1-v2-verified-v0"),
    ("schema", "c2kv-generality-candidate-cell-v6"),
    ("initial_view", {"policy": "s0_capacity_fallback", "version": "stale"}),
    ("recovery_backbone", "goal_rescue"),
    ("completion_review", True),
    ("proof_registry_version", "verified-binding-relations-v2"),
])
def test_c1_v2_rejects_version_or_contract_mismatch_before_checkpoint(
        tmp_path, field, replacement):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), "c1_v2_verified", "http://127.0.0.1:36200")
    cell[field] = replacement
    with pytest.raises(ValueError, match="matching version and contract"):
        candidate_cell.controller_with_binding(
            cell, base_controller={}, selected={},
            risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)


def test_c1_v2_ready_manifest_binds_version_controller_and_route(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    variant = "c1_v2_verified"
    contract = candidate_cell.c1_v2_contract(variant)
    controller = {"view_mode": "native_s0", "candidate_algorithm": {
        "variant": variant, "risk_artifact": {"model_kind": "c1_risk_logistic"},
        "risk_threshold": 0.5, **contract}}
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller), encoding="utf-8")
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200") | {
            "controller_path": str(controller_path)}
    ready = {
        "status": "ready",
        "s0_controller_contract": {
            "source": str(controller_path.resolve()), "config": controller,
            "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        },
        "candidate_algorithm": {
            "variant": variant, "stable_call_ids": True,
            "recovery_rounds_per_decision": 1, **contract,
        },
        "route_contract": {
            "baseline_identity": candidate_cell.C1_V2_VERSION + ":" + variant,
            "recovery_enabled": True, "max_generations_per_decision": 2,
        },
    }
    ready_path = tmp_path / "ready.json"

    def validate(cell_value=cell, ready_value=ready):
        ready_path.write_text(json.dumps(ready_value), encoding="utf-8")
        c2kv_cell.validate_static_ready_manifest(cell_value, ready_path)

    validate()
    with pytest.raises(RuntimeError, match="differs from frozen controller"):
        validate(cell | {"candidate_protocol": "c2kv-c1-v2-verified-v0"})
    changed_ready = json.loads(json.dumps(ready))
    changed_ready["route_contract"]["baseline_identity"] = (
        "c2kv-c1-v2-verified-v0:" + variant)
    with pytest.raises(RuntimeError, match="differs from frozen controller"):
        validate(ready_value=changed_ready)
    changed_ready = json.loads(json.dumps(ready))
    changed_ready["candidate_algorithm"]["completion_review"] = True
    with pytest.raises(RuntimeError, match="differs from frozen controller"):
        validate(ready_value=changed_ready)


@pytest.mark.parametrize("budget", [256, 128, 64])
def test_c1_v2_native_budget_uses_generic_override(tmp_path, monkeypatch, budget):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    checkpoint = tmp_path / "checkpoint"
    _checkpoint_with_geometry(checkpoint)
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path) | {
            "checkpoint": str(checkpoint),
            "model_name": "gen_c2kv_K0_compression_full_budget",
        },
        "c1_v2_verified", "http://127.0.0.1:36200", budget)
    monkeypatch.setattr(c2kv_cell, "_controller_with_binding", lambda _: ({}, None))
    prepared = c2kv_cell.prepare_cell_files(cell, {"working_points": {"K0": {
        "history_allowance_bytes": 100, "common_cap_bytes": 300}}})
    profile = prepared["native_history_budget"]
    assert prepared["cell_id"].endswith(f"__candidate_c1_v2_verified__b{budget}")
    assert profile["requested_tokens"] == budget
    assert profile["history_budget_bytes"] == budget * 147456
    assert profile["workspace_budget_bytes"] == budget * 147456


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


@pytest.mark.parametrize("variant", candidate_cell.GOAL_VARIANTS +
                         candidate_cell.VERIFIED_VARIANTS + candidate_cell.STATIC_VARIANTS +
                         candidate_cell.STATIC_EXTENSION_VARIANTS +
                         candidate_cell.C1_V2_VARIANTS)
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
    elif variant in candidate_cell.C1_V2_VARIANTS:
        expected.update(candidate_cell.c1_v2_contract(variant))
    elif variant in candidate_cell.STATIC_VARIANTS + candidate_cell.STATIC_EXTENSION_VARIANTS:
        expected.update(candidate_cell.static_contract(variant))
    assert controller == {"view_mode": "native_s0", "candidate_algorithm": expected}
    assert receipt == {"checkpoint": str(checkpoint)}
    assert cell["cell_id"].endswith("__candidate_" + variant)
    assert cell["candidate_protocol"] == (candidate_cell.STATIC_EXTENSION_VERSION
        if variant in candidate_cell.STATIC_EXTENSION_VARIANTS else candidate_cell.STATIC_VERSION
        if variant in candidate_cell.STATIC_VARIANTS else candidate_cell.VERIFIED_VERSION
        if variant in candidate_cell.VERIFIED_VARIANTS else candidate_cell.C1_V2_VERSION
        if variant in candidate_cell.C1_V2_VARIANTS else candidate_cell.GOAL_VERSION)
    assert "candidate_algorithm" not in base


@pytest.mark.parametrize("variant", ("goal_pending",) + candidate_cell.VERIFIED_VARIANTS +
                         candidate_cell.STATIC_VARIANTS +
                         candidate_cell.STATIC_EXTENSION_VARIANTS +
                         candidate_cell.C1_V2_VARIANTS)
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


@pytest.mark.parametrize("variant", ("turn_c1",) + candidate_cell.STATIC_VARIANTS +
                         candidate_cell.STATIC_EXTENSION_VARIANTS)
def test_candidate_cli_selects_variant_before_any_inference(tmp_path, monkeypatch, capsys, variant):
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
        "--candidate-algorithm", variant,
        "--sglang-backend-url", "http://127.0.0.1:36200",
        "--audit-results-only",
    ])
    assert rc == 0 and len(seen) == 1
    assert seen[0][0]["candidate_algorithm"] == variant
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


@pytest.mark.parametrize("variant", candidate_cell.STATIC_VARIANTS)
@pytest.mark.parametrize("change", [
    {"recovery_backbone": "wrong_backbone"},
    {"initial_view": {"policy": "dynamic", "version": "c2kv-static-initial-view-v1"}},
    {"initial_view": {"policy": "static_gist", "version": "stale-version"}},
    {"candidate_protocol": "c2kv-goal-composition-v1"},
    {"schema": "c2kv-generality-candidate-cell-v4"},
])
def test_static_cell_rejects_view_backbone_and_protocol_mismatch(tmp_path, variant, change):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200") | change
    with pytest.raises(ValueError, match="initial view and recovery backbone"):
        candidate_cell.controller_with_binding(
            cell, base_controller={}, selected={},
            risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)


@pytest.mark.parametrize("variant", candidate_cell.VERIFIED_STATIC_VARIANTS)
@pytest.mark.parametrize("proof", [None, "stale-proof-rules"])
def test_verified_static_cell_rejects_missing_or_wrong_proof_before_checkpoint(
        tmp_path, variant, proof):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    if proof is None:
        cell.pop("proof_registry_version")
    else:
        cell["proof_registry_version"] = proof
    with pytest.raises(ValueError, match="proof registry"):
        candidate_cell.controller_with_binding(
            cell, base_controller={}, selected={},
            risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)


@pytest.mark.parametrize("variant", candidate_cell.STATIC_EXTENSION_VARIANTS)
def test_static_extension_cell_rejects_missing_or_wrong_contract_before_checkpoint(
        tmp_path, variant):
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    required = {
        "schema": "c2kv-generality-candidate-cell-v6",
        "candidate_protocol": candidate_cell.STATIC_EXTENSION_VERSION,
        "threshold": 0.5,
        **candidate_cell.static_contract(variant),
    }
    for field, expected in required.items():
        for changed in (None, "stale-version"):
            invalid = json.loads(json.dumps(cell))
            if changed is None:
                invalid.pop(field)
            else:
                invalid[field] = changed
            with pytest.raises(ValueError, match="matching version and contract"):
                candidate_cell.controller_with_binding(
                    invalid, base_controller={}, selected={},
                    risk_artifact_path=tmp_path / "unused.json", bind_risk_artifact=None)
    assert required["schema"] == cell["schema"]
    assert required["candidate_protocol"] == cell["candidate_protocol"]


@pytest.mark.parametrize("variant", candidate_cell.STATIC_VARIANTS)
def test_static_ready_manifest_binds_loaded_controller_and_route(tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    contract = candidate_cell.static_contract(variant)
    controller = {"view_mode": "native_s0", "candidate_algorithm": {
        "variant": variant, "risk_artifact": {"model_kind": "c1_risk_logistic"},
        "risk_threshold": 0.5, **contract}}
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller), encoding="utf-8")
    cell = {"candidate_algorithm": variant, "controller_path": str(controller_path)}
    if variant in candidate_cell.VERIFIED_STATIC_VARIANTS:
        cell["proof_registry_version"] = contract["proof_registry_version"]
    loaded = {"source": str(controller_path.resolve()), "config": controller,
              "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest()}
    ready = {"status": "ready", "s0_controller_contract": loaded,
             "candidate_algorithm": {"variant": variant, "stable_call_ids": True,
                                     **contract},
             "route_contract": {"baseline_identity":
                                candidate_cell.STATIC_VERSION + ":" + variant,
                                "recovery_enabled": True,
                                "max_generations_per_decision": 2}}
    ready_path = tmp_path / "ready.json"

    def check(value):
        ready_path.write_text(json.dumps(value), encoding="utf-8")
        c2kv_cell.validate_static_ready_manifest(cell, ready_path)

    check(ready)
    if variant in candidate_cell.VERIFIED_STATIC_VARIANTS:
        for proof in (None, "stale-proof-rules"):
            changed_cell = dict(cell)
            if proof is None:
                changed_cell.pop("proof_registry_version")
            else:
                changed_cell["proof_registry_version"] = proof
            ready_path.write_text(json.dumps(ready), encoding="utf-8")
            with pytest.raises(RuntimeError, match="differs from frozen controller"):
                c2kv_cell.validate_static_ready_manifest(changed_cell, ready_path)
            changed_ready = json.loads(json.dumps(ready))
            if proof is None:
                changed_ready["candidate_algorithm"].pop("proof_registry_version")
            else:
                changed_ready["candidate_algorithm"]["proof_registry_version"] = proof
            ready_path.write_text(json.dumps(changed_ready), encoding="utf-8")
            with pytest.raises(RuntimeError, match="differs from frozen controller"):
                c2kv_cell.validate_static_ready_manifest(cell, ready_path)
            changed_controller = json.loads(json.dumps(controller))
            if proof is None:
                changed_controller["candidate_algorithm"].pop("proof_registry_version")
            else:
                changed_controller["candidate_algorithm"]["proof_registry_version"] = proof
            controller_path.write_text(json.dumps(changed_controller), encoding="utf-8")
            changed_ready = json.loads(json.dumps(ready))
            changed_ready["s0_controller_contract"]["config"] = changed_controller
            changed_ready["s0_controller_contract"]["sha256"] = hashlib.sha256(
                controller_path.read_bytes()).hexdigest()
            ready_path.write_text(json.dumps(changed_ready), encoding="utf-8")
            with pytest.raises(RuntimeError, match="differs from frozen controller"):
                c2kv_cell.validate_static_ready_manifest(cell, ready_path)
            controller_path.write_text(json.dumps(controller), encoding="utf-8")
    for section, field, wrong in (
            ("candidate_algorithm", "recovery_backbone", "goal_joint"),
            ("candidate_algorithm", "initial_view", {"policy": "none"}),
            ("route_contract", "baseline_identity", "c2kv-goal-composition-v1:" + variant),
            ("s0_controller_contract", "source", str(tmp_path / "other.json")),
            ("s0_controller_contract", "sha256", "0" * 64),
            ("s0_controller_contract", "config", {})):
        changed = json.loads(json.dumps(ready))
        changed[section][field] = wrong
        ready_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            c2kv_cell.validate_static_ready_manifest(cell, ready_path)
    # Existing candidates retain their original ready-manifest behavior.
    c2kv_cell.validate_static_ready_manifest(
        {**cell, "candidate_algorithm": "goal_pending"}, tmp_path / "absent.json")


@pytest.mark.parametrize("variant", candidate_cell.STATIC_EXTENSION_VARIANTS)
def test_static_extension_ready_binds_cell_controller_fields_and_version(
        tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    contract = candidate_cell.static_contract(variant)
    controller = {"view_mode": "native_s0", "candidate_algorithm": {
        "variant": variant, "risk_artifact": {"model_kind": "c1_risk_logistic"},
        "risk_threshold": 0.5, **contract}}
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller), encoding="utf-8")
    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200") | {
            "controller_path": str(controller_path)}
    ready = {
        "status": "ready",
        "s0_controller_contract": {
            "source": str(controller_path.resolve()), "config": controller,
            "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        },
        "candidate_algorithm": {"variant": variant, "stable_call_ids": True, **contract},
        "route_contract": {
            "baseline_identity": candidate_cell.STATIC_EXTENSION_VERSION + ":" + variant,
            "recovery_enabled": True, "max_generations_per_decision": 2,
        },
    }
    ready_path = tmp_path / "ready.json"

    def validate(cell_value=cell, ready_value=ready):
        ready_path.write_text(json.dumps(ready_value), encoding="utf-8")
        c2kv_cell.validate_static_ready_manifest(cell_value, ready_path)

    validate()
    for field in contract:
        changed_cell = json.loads(json.dumps(cell))
        changed_cell[field] = "stale-version"
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            validate(changed_cell)

        changed_controller = json.loads(json.dumps(controller))
        changed_controller["candidate_algorithm"][field] = "stale-version"
        controller_path.write_text(json.dumps(changed_controller), encoding="utf-8")
        changed_ready = json.loads(json.dumps(ready))
        changed_ready["s0_controller_contract"]["config"] = changed_controller
        changed_ready["s0_controller_contract"]["sha256"] = hashlib.sha256(
            controller_path.read_bytes()).hexdigest()
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            validate(ready_value=changed_ready)
        controller_path.write_text(json.dumps(controller), encoding="utf-8")

        changed_ready = json.loads(json.dumps(ready))
        changed_ready["candidate_algorithm"][field] = "stale-version"
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            validate(ready_value=changed_ready)

    for field, wrong in (("variant", "static_t02"), ("risk_threshold", 0.6)):
        changed_controller = json.loads(json.dumps(controller))
        changed_controller["candidate_algorithm"][field] = wrong
        controller_path.write_text(json.dumps(changed_controller), encoding="utf-8")
        changed_ready = json.loads(json.dumps(ready))
        changed_ready["s0_controller_contract"]["config"] = changed_controller
        changed_ready["s0_controller_contract"]["sha256"] = hashlib.sha256(
            controller_path.read_bytes()).hexdigest()
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            validate(ready_value=changed_ready)
        controller_path.write_text(json.dumps(controller), encoding="utf-8")

    for field, changed in (
            ("schema", "c2kv-generality-candidate-cell-v5"),
            ("candidate_protocol", candidate_cell.STATIC_VERSION),
            ("ratio", 4),
            ("threshold", 0.6)):
        with pytest.raises(RuntimeError, match="differs from frozen controller"):
            validate(cell | {field: changed})
    changed_ready = json.loads(json.dumps(ready))
    changed_ready["route_contract"]["baseline_identity"] = (
        candidate_cell.STATIC_VERSION + ":" + variant)
    with pytest.raises(RuntimeError, match="differs from frozen controller"):
        validate(ready_value=changed_ready)


@pytest.mark.parametrize("variant", candidate_cell.STATIC_VARIANTS +
                         candidate_cell.STATIC_EXTENSION_VARIANTS)
def test_static_resume_rejects_changed_frozen_view_or_backbone(tmp_path, monkeypatch, variant):
    monkeypatch.setitem(sys.modules, "current", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "evidence_sets", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    cell = candidate_cell.candidate_cell_from_source(
        source_cell(tmp_path), variant, "http://127.0.0.1:36200")
    monkeypatch.setattr(c2kv_cell, "_controller_with_binding", lambda _: ({}, None))
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 100, "common_cap_bytes": 300}}}
    prepared = c2kv_cell.prepare_cell_files(cell, budgets)
    (Path(prepared["cell_dir"]) / "batches" / "attempt").mkdir(parents=True)
    for change in ({"recovery_backbone": "wrong"},
                   {"initial_view": {"policy": "static_gist", "version": "stale"}}):
        with pytest.raises(ValueError, match="different frozen cell.json"):
            c2kv_cell.prepare_cell_files(cell | change, budgets)
