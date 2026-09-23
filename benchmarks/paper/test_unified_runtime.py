"""Contract checks for marked unified native history rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.paper import c1_appworld, history_kv_client, native_extra, runner
from benchmarks.paper.racer_matrix import (
    RACER_BACKENDS, resolve_unified_runtime_methods, with_racer_methods,
)


KV_SOURCES = {
    "h2o": ("H2O", "history_kv_h2o_r25_persistent"),
    "snapkv": ("SnapKV", "history_kv_snapkv_r25_persistent"),
    "pyramidkv": ("PyramidKV", "history_kv_pyramidkv_r25_persistent"),
    "commitkv": ("CommitKV", "commitkv"),
    "agentkv": ("AgentKV", "agentkv"),
    "streamingllm": ("StreamingLLM", "history_kv_streamingllm_r25_persistent"),
}


def unified_config(budget=384):
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["history_kv_budget_tokens"] = budget
    scope = ["bfcl_base", "tau2"]
    config["methods"] = [{
        "method": "C2KV", "arm": "c2kv_native_r8", "group": "main", "ratio": 8,
        "history_runtime": "racer", "history_backend": "c2kv",
        "recovery_policy": "off", "compression_ratio": 8,
        "history_budget_tokens": "shared", "benchmarks": scope,
    }]
    config["methods"] += [{
        "method": name, "arm": arm, "group": "main", "history_runtime": "racer",
        "history_backend": backend, "recovery_policy": "off",
        "history_budget_tokens": "shared", "benchmarks": scope,
        "tool_contexts": ["raw"],
    } for backend, (name, arm) in KV_SOURCES.items()]
    return config


def test_marked_primaries_use_one_explicit_b_and_keep_bare_c2kv_identity():
    original = unified_config()
    saved = copy.deepcopy(original)
    resolved = resolve_unified_runtime_methods(original)
    assert original == saved
    assert len(resolved["methods"]) == 1 + len(KV_SOURCES)
    bare = resolved["methods"][0]
    assert bare["arm"] == "c2kv_native_r8"
    assert bare["history_allocation"] == "c2kv_bare"
    assert bare["history_budget_tokens"] == 384
    assert bare["compression_ratio"] == 8
    assert "racer_c2kv_off_b384" not in {row["arm"] for row in resolved["methods"]}
    for backend in KV_SOURCES:
        row = next(row for row in resolved["methods"]
                   if row["arm"] == f"racer_v2_{backend}_bare_b384")
        assert row["history_budget_tokens"] == 384
        assert row["racer_backend"]["backend_config"]["target_tokens"] == 384
        assert row["recovery_policy"] == "off"
        assert row["benchmarks"] == ["bfcl_base", "tau2"]
    assert set(KV_SOURCES) <= set(RACER_BACKENDS)


def test_selected_recovery_reuses_native_off_and_preserves_scope():
    resolved = resolve_unified_runtime_methods(unified_config())
    paired = with_racer_methods(resolved, ("c2kv", "h2o", "agentkv", "streamingllm"),
                                ("off", "pending_verified"), 384)
    arms = [row["arm"] for row in paired["methods"]]
    assert len(arms) == len(set(arms))
    assert "racer_c2kv_off_b384" not in arms
    assert "c2kv_native_r8" in arms
    for backend in ("h2o", "agentkv", "streamingllm"):
        off = next(row for row in paired["methods"]
                   if row["arm"] == f"racer_v2_{backend}_bare_b384")
        protected = next(row for row in paired["methods"]
                         if row["arm"] == f"racer_v2_{backend}_pending_verified_protected_off_b384")
        on = next(row for row in paired["methods"]
                  if row["arm"] == f"racer_v2_{backend}_pending_verified_b384")
        assert off["benchmarks"] == on["benchmarks"]
        assert protected["benchmarks"] == on["benchmarks"]
        assert protected["recovery_policy"] == "off"
        assert protected["racer_backend"]["allocation"] == "racer_s0"
        assert on["racer_backend"]["allocation"] == "racer_s0"
        assert off["tool_contexts"] == on["tool_contexts"]
        assert off["racer_backend"]["backend_config"] == on["racer_backend"]["backend_config"]
    c2kv_on = next(row for row in paired["methods"]
                   if row["arm"] == "racer_v2_c2kv_pending_verified_b384")
    assert c2kv_on["compression_ratio"] == 8
    assert c2kv_on["benchmarks"] == ["bfcl_base", "tau2"]
    assert with_racer_methods(paired, ("c2kv", "h2o", "agentkv", "streamingllm"),
                              ("off", "pending_verified"), 384) == paired


def test_mismatched_existing_off_or_missing_global_budget_fails():
    with pytest.raises(ValueError, match="history_kv_budget_tokens"):
        resolve_unified_runtime_methods(unified_config(None))
    resolved = resolve_unified_runtime_methods(unified_config())
    existing = next(row for row in resolved["methods"]
                    if row["arm"] == "racer_v2_h2o_bare_b384")
    existing["benchmarks"] = ["bfcl_base"]
    paired = with_racer_methods(resolved, ("h2o",), ("pending_verified",), 384)
    assert next(row for row in paired["methods"]
                if row["arm"] == "racer_v2_h2o_pending_verified_b384")["benchmarks"] == ["bfcl_base"]
    bad = copy.deepcopy(paired)
    on = next(row for row in bad["methods"]
              if row["arm"] == "racer_v2_h2o_pending_verified_b384")
    on["benchmarks"] = ["tau2"]
    with pytest.raises(ValueError, match="paired contract"):
        with_racer_methods(bad, ("h2o",), ("pending_verified",), 384)


def test_unmarked_frozen_legacy_methods_keep_their_arms():
    frozen = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    frozen["methods"] = [{
        "method": "H2O", "arm": "history_kv_h2o_r25_persistent",
        "group": "main", "history_budget_tokens": "shared",
        "benchmarks": ["bfcl_base"],
    }]
    frozen["history_kv_budget_tokens"] = 384
    assert resolve_unified_runtime_methods(frozen) is frozen
    resolved = runner.resolve_history_kv_budgets(frozen)
    assert resolved["methods"][0]["arm"] == "history_kv_h2o_r25_persistent"


def test_marked_primary_rebinds_shared_b_and_keeps_independent_sweep():
    config = runner.resolve_history_kv_budgets(unified_config(384))
    swept = runner.with_history_kv_budget(config, "commitkv", 1024)
    assert {row["arm"] for row in swept["methods"] if row.get("history_backend") == "commitkv"} == {
        "racer_v2_commitkv_bare_b384", "racer_v2_commitkv_bare_b1024"}
    swept["history_kv_budget_tokens"] = 768
    rebound = runner.resolve_history_kv_budgets(swept)
    assert {row["arm"] for row in rebound["methods"] if row.get("history_backend") == "commitkv"} == {
        "racer_v2_commitkv_bare_b768", "racer_v2_commitkv_bare_b1024"}
    assert next(row for row in rebound["methods"]
                if row["arm"] == "c2kv_native_r8")["history_budget_tokens"] == 768


def test_explicit_ratio4_overlay_stays_unmarked_legacy_bare():
    config = runner.with_native_ratios(unified_config(), [4])
    ratio4 = next(row for row in config["methods"] if row["arm"] == "c2kv_native_r4")
    assert "history_runtime" not in ratio4
    assert "history_budget_tokens" not in ratio4
    assert "history_backend" not in ratio4
    resolved = runner.resolve_history_kv_budgets(config)
    assert next(row for row in resolved["methods"]
                if row["arm"] == "c2kv_native_r4") == ratio4


@pytest.mark.parametrize("overlay", [runner.with_history_kv_budget,
                                    runner.with_native_history_budget])
def test_marked_native_r8_supports_independent_absolute_b_cells(overlay, tmp_path):
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["history_kv_budget_tokens"] = 128
    config = runner.resolve_history_kv_budgets(config)
    swept = overlay(config, "c2kv_native_r8", 256)
    cells = [row for row in runner.cells(swept)
             if row["arm"] == "c2kv_native_r8" and row["benchmark"] == "bfcl_base"]
    assert {row["cell_id"] for row in cells} == {
        "bfcl_base__c2kv_native_r8_b128", "bfcl_base__c2kv_native_r8_b256"}
    assert next(row for row in cells if row["history_budget_tokens"] == 256)["group"] == "budget"
    with pytest.raises(ValueError, match="already exists"):
        overlay(swept, "c2kv_native_r8", 256)
    resolved, cell, command, _ = history_kv_client.plan(
        config, "bfcl_base", "c2kv_native_r8=256", tmp_path,
        "http://localhost:36200")
    assert resolved["history_kv_budget_tokens"] == 128
    assert cell["cell_id"] == "bfcl_base__c2kv_native_r8_b256"
    assert command[command.index("--history-budget-tokens") + 1] == "256"


def test_single_cell_client_routes_marked_legacy_name_to_native_off(tmp_path):
    resolved, cell, command, _ = history_kv_client.plan(
        unified_config(None), "bfcl_base", "history_kv_h2o_r25_persistent=384", tmp_path,
        "http://localhost:36200")
    assert resolved["history_kv_budget_tokens"] == 384
    assert cell["arm"] == "racer_v2_h2o_bare_b384"
    assert cell["cell_id"] == "bfcl_base__racer_v2_h2o_bare_b384"
    assert command[command.index("--arm") + 1] == cell["arm"]
    assert "--shared-engine" not in command
    assert "--history-kv-target-tokens" not in command


def test_single_cell_client_preserves_global_b_for_native_sweep_and_frozen_alias(tmp_path):
    source_arm = "history_kv_streamingllm_r25_persistent"
    for config in (unified_config(384),
                   runner.resolve_history_kv_budgets(unified_config(384))):
        resolved, cell, command, _ = history_kv_client.plan(
            config, "bfcl_base", f"{source_arm}=1024", tmp_path,
            "http://localhost:36200")
        assert resolved["history_kv_budget_tokens"] == 384
        assert cell["arm"] == "racer_v2_streamingllm_bare_b1024"
        assert cell["history_budget_tokens"] == 1024
        assert "--shared-engine" not in command
        assert "--history-kv-target-tokens" not in command


def test_default_prepare_uses_one_global_b_without_duplicate_off(tmp_path):
    output = tmp_path / "paper"
    runner.main([
        "prepare", "--history-kv-budget-tokens", "384", "--output", str(output),
        "--sglang-source", str(tmp_path / "engine"),
        "--racer-backends", "c2kv,h2o,streamingllm",
        "--racer-policies", "pending_verified",
    ])
    config = json.loads((output / "config.resolved.json").read_text(encoding="utf-8"))
    methods = config["methods"]
    assert sum(row["arm"] == "c2kv_native_r8" for row in methods) == 1
    assert not any(row["arm"] == "racer_c2kv_off_b384" for row in methods)
    for backend in ("h2o", "streamingllm"):
        assert sum(row["arm"] == f"racer_v2_{backend}_bare_b384" for row in methods) == 1
        assert sum(row["arm"] == f"racer_v2_{backend}_pending_verified_protected_off_b384"
                   for row in methods) == 1
        assert sum(row["arm"] == f"racer_v2_{backend}_pending_verified_b384"
                   for row in methods) == 1
    assert sum(row["arm"] == "racer_v2_c2kv_pending_verified_b384"
               for row in methods) == 1
    plan = json.loads((output / "commands.json").read_text(encoding="utf-8"))
    assert any(row["cell_id"] == "bfcl_base__c2kv_native_r8_b384" for row in plan)
    assert all(row["history_budget_tokens"] == 384 for row in plan
               if row.get("history_runtime") == "racer")


def test_explicit_budget_overlay_uses_native_off_cell(tmp_path):
    output = tmp_path / "sweep"
    runner.main([
        "prepare", "--history-kv-budget-tokens", "384", "--output", str(output),
        "--sglang-source", str(tmp_path / "engine"),
        "--history-kv-budget", "commitkv=1024",
    ])
    rows = json.loads((output / "commands.json").read_text(encoding="utf-8"))
    ids = {row["cell_id"] for row in rows}
    assert "bfcl_base__racer_v2_commitkv_bare_b384" in ids
    assert "bfcl_base__racer_v2_commitkv_bare_b1024" in ids
    assert "bfcl_base__commitkv_b1024" not in ids


@pytest.mark.parametrize("benchmark", ["acebench_agent", "toolsandbox", "tau2", "appworld"])
def test_native_extra_ready_binds_exact_racer_backend_and_controller(tmp_path, benchmark):
    config = resolve_unified_runtime_methods(unified_config())
    config["native_arm"] = "racer_v2_agentkv_bare_b384"
    identity = native_extra.arm_identity(config)
    racer = identity["racer_backend"]
    controller = tmp_path / "controller.json"
    controller.write_text(json.dumps({"racer_backend": racer}), encoding="utf-8")
    ready = {
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": native_extra.BENCHMARKS[benchmark][0],
        "source_profile": native_extra.BENCHMARKS[benchmark][1],
        "allowed_task_ids": ["task_1"], "model_name": identity["model_name"],
        "view_mode": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "ratio": identity["ratio"], "generation_backend": "sglang",
        "s0_controller_contract": {
            "source": str(controller.resolve()),
            "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
            "config": {"racer_backend": racer},
        },
        "racer_backend": dict(racer, identity="racer:v2:agentkv:bare:off:b384",
                              quality_validated=False),
        "route_contract": {
            "baseline_identity": "racer:v2:agentkv:bare:off:b384",
            "history_allocation": "backend_native_persistent",
            "recovery_enabled": False,
        },
    }
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(ready), encoding="utf-8")
    native_extra.validate_ready_manifest(config, benchmark, "task_1", path, controller)
    ready["racer_backend"]["identity"] = "racer:agentkv:t02:b384"
    path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RACER backend identity"):
        native_extra.validate_ready_manifest(config, benchmark, "task_1", path, controller)


def test_native_budget_helper_uses_racer_arm_b_and_materializes_once(tmp_path, monkeypatch):
    calls = []
    receipt = {"override_eval_policy_path": str(tmp_path / "history_budget_eval_policy.json")}
    fake = SimpleNamespace(
        resolve_override=lambda tokens, checkpoint, design, runtime, output: (
            calls.append((tokens, checkpoint, runtime, output)) or receipt),
        materialize=lambda resolved: calls.append(resolved),
    )
    monkeypatch.setattr(c1_appworld, "_load_module", lambda path, stem: fake)
    config = {"native_arm": "racer_v2_h2o_bare_b384", "checkpoint": str(tmp_path / "checkpoint")}
    design = {"runtime": {"eval_policy": "configs/eval_policy.json"}}
    actual = c1_appworld.apply_native_history_budget(
        config, design, tmp_path / "delivery", tmp_path)
    assert actual == receipt
    assert calls[0][0] == 384
    assert calls[0][1] == tmp_path / "checkpoint"
    assert calls[0][2] == tmp_path / "delivery" / "runtime"
    assert calls[1] == receipt
    assert design["runtime"]["eval_policy"] == receipt["override_eval_policy_path"]
    config["native_history_budget_tokens"] = 256
    with pytest.raises(ValueError, match="budgets disagree"):
        c1_appworld.apply_native_history_budget(config, design, tmp_path / "delivery", tmp_path)


def test_appworld_racer_design_matches_off_on_shadow_policy(tmp_path, monkeypatch):
    config = with_racer_methods(
        resolve_unified_runtime_methods(unified_config()),
        ("h2o",), ("pending_verified",), 384)
    config["generation_timeout"] = 7.5
    config["native_arm"] = "racer_v2_h2o_bare_b384"
    budget_calls = []
    monkeypatch.setattr(c1_appworld, "apply_native_history_budget",
                        lambda *args: budget_calls.append(args))
    controller = tmp_path / "controller.json"
    design = c1_appworld._resolved_design(
        config, Path(__file__).resolve().parents[2] / "experiments" / "history_system",
        controller, tmp_path)
    assert design["candidate_id"] == "racer_v2_h2o_bare_b384"
    assert design["runtime"]["controller"] == str(controller.resolve())
    assert "shadow_feature_config" not in design["runtime"]
    assert design["runtime"]["sglang_timeout_seconds"] == 7.5
    config["native_arm"] = "racer_v2_h2o_pending_verified_b384"
    on = c1_appworld._resolved_design(
        config, Path(__file__).resolve().parents[2] / "experiments" / "history_system",
        controller, tmp_path)
    assert on["candidate_id"] == "racer_v2_h2o_pending_verified_b384"
    assert "shadow_feature_config" in on["runtime"]
    assert on["runtime"]["sglang_timeout_seconds"] == 7.5
    assert len(budget_calls) == 2


def test_portable_racer_design_matches_off_on_shadow_policy(tmp_path, monkeypatch):
    config = with_racer_methods(
        resolve_unified_runtime_methods(unified_config()),
        ("h2o",), ("pending_verified",), 384)
    config["generation_timeout"] = 7.5
    designs = []

    class FakeRunner:
        def server_command(self, design, **kwargs):
            designs.append(copy.deepcopy(design))
            return ["server"]

    monkeypatch.setattr(c1_appworld, "_delivery_runner", lambda _: FakeRunner())
    monkeypatch.setattr(c1_appworld, "apply_native_history_budget", lambda *args: None)
    delivery = Path(__file__).resolve().parents[2] / "experiments" / "history_system"
    controller = tmp_path / "controller.json"
    for arm in ("racer_v2_h2o_bare_b384", "racer_v2_h2o_pending_verified_b384"):
        config["native_arm"] = arm
        native_extra.server_command(config, "tau2", "task_1", tmp_path,
                                    delivery, controller)
    assert "shadow_feature_config" not in designs[0]["runtime"]
    assert "shadow_feature_config" in designs[1]["runtime"]
    assert all(design["runtime"]["sglang_timeout_seconds"] == 7.5
               for design in designs)


@pytest.mark.parametrize("invalid", [True, 0, -1, float("inf"), "7.5"])
def test_native_generation_timeout_rejects_invalid_config(invalid):
    with pytest.raises(ValueError, match="generation_timeout"):
        c1_appworld.apply_native_generation_timeout(
            {"generation_timeout": invalid}, {"runtime": {}})
