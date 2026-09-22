"""CPU-only registry and CLI checks for modular RACER paper cells."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks.arms import get_arm, history_kv_spec
from benchmarks.paper import c1, runner
from benchmarks.paper.candidate_matrix import VARIANT_TO_ARM
from benchmarks.paper.racer_matrix import (
    RACER_BACKENDS, RACER_POLICIES, parse_racer_arm_name,
    parse_racer_backends, parse_racer_policies, resolve_racer_backend,
    with_racer_methods,
)


def base_config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_policy_parser_keeps_exact_candidate_identity_and_adds_off_pair():
    assert parse_racer_backends("all") == RACER_BACKENDS
    assert parse_racer_policies("pending_verified") == ("off", "pending_verified")
    assert parse_racer_policies("all") == RACER_POLICIES
    assert set(RACER_POLICIES) == {"off", "t02", *VARIANT_TO_ARM}
    with pytest.raises(ValueError, match="unknown RACER policies"):
        parse_racer_policies("pending-verified")
    assert resolve_racer_backend("h2o", "request_contract", 256)[
        "detector_calibration"] == "not_used"


def test_overlay_is_paired_budgeted_and_independent_from_tool_contexts():
    original = base_config()
    saved = copy.deepcopy(original)
    config = with_racer_methods(
        original, ("c2kv", "h2o"), parse_racer_policies("t02"), 256)
    assert original == saved

    racer_methods = [method for method in config["methods"] if method["group"] == "racer"]
    assert len(racer_methods) == 4
    assert {method["racer_backend"]["policy"] for method in racer_methods} == {"off", "t02"}
    assert all(method["history_budget_tokens"] == 256 for method in racer_methods)
    assert all("ratio" not in method and "retention" not in method for method in racer_methods)

    raw_rows = [row for row in runner.cells(config) if row["group"] == "racer"]
    assert len(raw_rows) == 4 * len(config["benchmarks"])
    assert len({row["cell_id"] for row in raw_rows}) == len(raw_rows)
    h2o_t02 = next(row for row in raw_rows
                    if row["arm"] == "racer_h2o_t02_b256"
                    and row["benchmark"] == "bfcl_base")
    assert h2o_t02["tool_context"] == "raw"
    assert h2o_t02["calibration_status"] == "frozen_c2kv_unvalidated_transfer"
    assert h2o_t02["history_allocation"] == "backend_native_persistent"

    with_tools = runner.with_tool_contexts(config, ["t0_r8"])
    tool_rows = [row for row in runner.cells(with_tools)
                 if row["arm"] == "racer_h2o_t02_b256"
                 and row["benchmark"] == "bfcl_base"]
    assert [row["tool_context"] for row in tool_rows] == ["raw", "t0_r8"]
    compressed = tool_rows[1]
    catalog = next(item for item in original["tool_contexts"] if item["name"] == "t0_r8")
    assert compressed["tool_memory"] == catalog["spec"]
    assert compressed["tool_checkpoint"] == catalog["checkpoint"]
    selected = runner.with_tool_contexts(config, ["t0_r8_hybrid3_schema_latest_event"])
    selected_rows = [row for row in runner.cells(selected)
                     if row["arm"] == "racer_h2o_t02_b256"
                     and row["benchmark"] == "bfcl_base"]
    assert [row["tool_context"] for row in selected_rows] == [
        "raw", "t0_r8_hybrid3_schema_latest_event"]
    assert selected_rows[1]["tool_memory"] == "t0:r8:hybrid3:schema:selector=latest_event_topk_v1"


def test_dynamic_arm_and_server_keep_persistent_transaction_flags():
    arm = get_arm("racer_commitkv_pending_verified_b384")
    assert arm.native_controller == "racer"
    assert arm.compress_history is False
    spec = history_kv_spec(arm)
    assert spec["method"] == "commitkv"
    assert spec["target_tokens"] == 384
    assert spec["persistent_session"] is True

    config = dict(base_config(), radix_cache_arms=["*"])
    command = runner.server_command(
        config, Path("engine"), arm.name, "bfcl_base")
    for flag in ("--disable-radix-cache", "--disable-overlap-schedule",
                 "--enable-streaming-session", "--enable-return-hidden-states"):
        assert flag in command

    identity = c1.racer_source_identity()
    assert identity["schema"] == "racer-source-identity-v1"
    assert "benchmarks/toolselection.py" in identity["files"]
    assert "benchmarks/fixtures/tool_selector_bfcl_source.json" in identity["files"]
    assert {name.rsplit("/", 1)[-1] for name in identity["files"]
            if "/racer/" in name} >= {
                "__init__.py", "allocator.py", "config.py", "generator.py",
                "policies.py", "tools.py",
            }


def test_prepare_and_cli_route_racer_through_native_c1_without_ratio_budget_alias(tmp_path):
    output = tmp_path / "results"
    runner.main([
        "prepare", "--output", str(output), "--sglang-source", str(tmp_path / "engine"),
        "--racer-backends", "h2o", "--racer-policies", "pending_verified",
        "--racer-history-budget", "512", "--tool-contexts", "t0_r8",
    ])
    resolved = json.loads((output / "config.resolved.json").read_text(encoding="utf-8"))
    methods = [method for method in resolved["methods"] if method.get("group") == "racer"]
    assert [method["racer_backend"]["policy"] for method in methods] == [
        "off", "pending_verified"]
    plan = json.loads((output / "commands.json").read_text(encoding="utf-8"))
    cells = [row for row in plan if row["arm"] == "racer_h2o_pending_verified_b512"]
    assert {row["tool_context"] for row in cells} == {"raw", "t0_r8"}
    raw = next(row for row in cells
               if row["benchmark"] == "bfcl_base" and row["tool_context"] == "raw")
    assert raw["command"][raw["command"].index("--arm") + 1] == raw["arm"]
    assert "--history-budget-tokens" not in raw["command"]
    assert raw["history_budget_tokens"] == 512
    assert "ratio" not in raw


@pytest.mark.parametrize("policy,method,candidate,detector", [
    ("off", "c2kv_only", None, None),
    ("t02", "proposed", None, "t02_risk"),
    ("pending_verified", "proposed", "pending_verified", None),
])
def test_paper_delivery_preserves_policy_factory_and_controller_schema(
        tmp_path, monkeypatch, policy, method, candidate, detector):
    config = with_racer_methods(base_config(), ("h2o",), (policy,), 256)
    config["sglang_source"] = str(tmp_path / "engine")
    arm = f"racer_h2o_{policy}_b256"
    previous = c1.ARM
    try:
        c1.select_arm(arm)
        delivery = c1.load_delivery()
        args = c1.delivery_args(config, "bfcl_base", tmp_path / policy, [], delivery)
        assert args.method == method
        assert args.candidate_algorithm == candidate
        if detector is not None:
            assert args.detector == detector
        assert args.racer_backend_config == resolve_racer_backend("h2o", policy, 256)

        monkeypatch.setattr(delivery, "_build_profile_unbudgeted",
                            lambda _args: ({"policy_config": policy}, {"ratio": 8}))
        monkeypatch.setattr(delivery, "_history_budget_override", lambda _args: None)
        controller, profile = delivery.build_profile(SimpleNamespace(
            racer_backend_config=args.racer_backend_config,
            history_budget_tokens=None,
        ))
        assert controller["racer_backend"] == args.racer_backend_config
        assert profile["racer_backend"] == args.racer_backend_config
        assert profile["composition_identity"] == arm
        assert profile["calibration_status"] == (
            "not_used" if policy == "off" else "frozen_c2kv_unvalidated_transfer")
        assert "ratio" not in profile
    finally:
        c1.select_arm(previous)


@pytest.mark.parametrize("kwargs", [
    {"backends": (), "policies": ("off",), "budget": 256},
    {"backends": ("h2o",), "policies": (), "budget": 256},
    {"backends": ("h2o",), "policies": ("off",), "budget": None},
    {"backends": ("h2o",), "policies": ("off",), "budget": 0},
])
def test_partial_or_invalid_overlay_is_rejected(kwargs):
    with pytest.raises(ValueError):
        with_racer_methods(
            base_config(), kwargs["backends"], kwargs["policies"], kwargs["budget"])


def test_arm_parser_rejects_nominal_ratio_or_aliased_policy_identity():
    assert parse_racer_arm_name("racer_streamingllm_static_verified_v2_b1024") == (
        "streamingllm", "static_verified_v2", 1024)
    with pytest.raises(ValueError):
        parse_racer_arm_name("racer_h2o_t02_r8")
    with pytest.raises(ValueError):
        parse_racer_arm_name("racer_h2o_pending-verified_b256")


def test_racer_budget_override_applies_to_every_portable_benchmark(tmp_path, monkeypatch):
    delivery = c1.load_delivery()
    racer = resolve_racer_backend("h2o", "off", 384)
    calls = []

    def resolve(tokens, checkpoint, design, runtime, output):
        calls.append((tokens, checkpoint, design, runtime, output))
        return {"requested_tokens": tokens, "history_budget_bytes": tokens * 2,
                "workspace_budget_bytes": tokens * 2}

    monkeypatch.setitem(sys.modules, "history_budget", SimpleNamespace(resolve_override=resolve))
    args = SimpleNamespace(
        history_budget_tokens=None, racer_backend_config=racer,
        method="c2kv_only", benchmark="tau2", checkpoint=tmp_path / "checkpoint",
        out=tmp_path / "out",
    )
    receipt = delivery._history_budget_override(args, {"design": "fixture"})
    assert receipt["requested_tokens"] == 384
    assert calls == [(384, args.checkpoint, {"design": "fixture"},
                      delivery.RUNTIME, args.out)]


def test_t02_profile_builds_real_backend_policy_and_c2kv_keeps_original_factory(
        tmp_path, monkeypatch):
    config = with_racer_methods(base_config(), ("h2o",), ("t02",), 256)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    config.update(checkpoint=str(checkpoint), sglang_source=str(tmp_path / "engine"))
    config["c1"] = dict(config["c1"], detector="t02_risk",
                        embedding_model=str(embedding), embedding_device="cpu")
    previous = c1.ARM
    try:
        c1.select_arm("racer_h2o_t02_b256")
        delivery = c1.load_delivery()
        args = c1.delivery_args(config, "bfcl_base", tmp_path / "out", [], delivery)
        selected = copy.deepcopy(delivery.current.load_config())
        selected["checkpoint_selection"]["config_sha256"] = hashlib.sha256(
            (checkpoint / "config.json").read_bytes()).hexdigest()
        monkeypatch.setattr(delivery.current, "load_config", lambda: copy.deepcopy(selected))
        monkeypatch.setattr(delivery.evidence_sets, "load_config", lambda: copy.deepcopy(selected))
        monkeypatch.setattr(delivery, "bind_risk_artifact",
                            lambda artifact, _checkpoint: (artifact, {"status": "fixture"}))
        monkeypatch.setattr(delivery, "_history_budget_override", lambda _args: {
            "requested_tokens": 256, "kv_bytes_per_token": 1,
            "history_budget_bytes": 256, "workspace_budget_bytes": 256,
        })
        controller, profile = delivery.build_profile(args)
        assert controller["gp_experiments"]["set_selector"] == "risk"
        assert controller["racer_backend"] == resolve_racer_backend("h2o", "t02", 256)
        assert profile["racer_runtime_budget"]["history_budget_bytes"] == 256

        from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
        from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
        from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
        from benchmarks.memory_runtime.racer.allocator import PersistentHistoryAllocator
        from benchmarks.memory_runtime.racer.policies import BackendPolicy
        from benchmarks.memory_runtime.tests.test_candidate_allocation import (
            Tokenizer, packing, policy,
        )

        geometry = packing()
        geometry["ratios"] = [8]
        composed = build_event_native_controller(
            Tokenizer(), packing=geometry, policy=policy(256),
            view_mode=NATIVE_S0_MODE, model_context=100_000,
            s0_config=controller, benchmark="bfcl")
        assert isinstance(composed, BackendPolicy)
        assert composed.backend.policy == "t02"
        assert composed.inner.gp["set_selector"] == "risk"
        assert isinstance(composed.inner.base, PersistentHistoryAllocator)
        assert composed.inner.base.policy_config.history_budget_bytes == 256

        c2kv_controller = copy.deepcopy(controller)
        c2kv_controller["racer_backend"] = resolve_racer_backend("c2kv", "t02", 256)
        delegated = build_event_native_controller(
            Tokenizer(), packing=geometry, policy=policy(256),
            view_mode=NATIVE_S0_MODE, model_context=100_000,
            compression_policy=ALWAYS_COMPRESSION_POLICY,
            s0_config=c2kv_controller, benchmark="bfcl")
        assert not isinstance(delegated, BackendPolicy)
        assert not isinstance(delegated.base, PersistentHistoryAllocator)
        assert delegated.gp["set_selector"] == "risk"
    finally:
        c1.select_arm(previous)


def test_persistent_racer_summary_accepts_actual_receipts_and_exact_budget(tmp_path):
    delivery = c1.load_delivery()
    server = tmp_path / "server"
    server.mkdir()
    racer = resolve_racer_backend("h2o", "off", 256)
    identity = "racer:h2o:off:b256"
    receipt = {**racer, "identity": identity, "quality_validated": False}
    (server / "ready.json").write_text(json.dumps({
        "racer_backend": receipt,
        "runtime_policy_contract": {"effective_policy": {
            "kv_bytes_per_token": 2,
            "history_budget_bytes": 512,
            "workspace_budget_bytes": 512,
        }},
    }), encoding="utf-8")
    accounting = {
        "resident_prompt_tokens": 241,
        "active_history_tokens": 200,
        "native_evidence_tokens": 40,
        "history_and_evidence_tokens": 240,
    }
    usage = {"prompt_tokens": 241, "completion_tokens": 2, "total_tokens": 243}
    record = {
        "status": "completed", "session_id": "s", "decision_key": "d1",
        "pre_generation_budget_checks": [{"status": "passed"}],
        "generation_trace": [{
            "phase": "draft", "status": "completed", "usage": usage,
            "generation": {"token_ids": [1, 2], "stats": {
                "racer_backend": receipt,
                "racer_served_usage": usage,
                "racer_accounting": accounting,
                "kv_memory_report": {"history_kv_lifecycle": {
                    "session_id": "racer-s", "persistent_session_enabled": True,
                    "full_history_reprefill_performed": False,
                    "transaction": {"decision_id": "d1", "phase": "draft"},
                }},
                "generation_calls": 1,
            }},
        }],
    }
    (server / "steps.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    official = {"n": 1, "task_rows": [{
        "semantic_score": 0.0, "normal_termination": True, "protocol_legal": True,
    }]}
    telemetry = delivery.summarize_task("tau2", "0", tmp_path, official, 1.0)
    assert telemetry["racer_backend_identity"] == identity
    assert telemetry["racer_max_history_and_evidence_tokens"] == 240
    assert telemetry["racer_effective_budget_match"] is True
    assert all(delivery.functional_checks(
        "c2kv_only", "disabled", telemetry, racer_backend=racer)["required"].values())

    failed = dict(telemetry, racer_accounting_passed=False)
    assert not all(delivery.functional_checks(
        "c2kv_only", "disabled", failed, racer_backend=racer)["required"].values())
