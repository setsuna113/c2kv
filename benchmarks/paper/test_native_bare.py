"""The native bare arm must not silently inherit a C1/S0 controller."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1, c1_appworld, native_extra, runner


@pytest.fixture(params=[4, 8])
def native_config(tmp_path, request):
    original = c1.ARM
    arm = f"c2kv_native_r{request.param}"
    c1.select_arm(arm)
    config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    config = runner.with_native_ratios(config, [request.param])
    config["methods"] = [method for method in config["methods"] if method["method"] != "C2KV" or method["arm"] == arm]
    config.update(native_arm=arm, sglang_source=str(tmp_path / "sglang"))
    (tmp_path / "bfcl" / "bfcl_eval").mkdir(parents=True)
    config["bfcl_dir"] = str(tmp_path / "bfcl")
    yield config
    c1.select_arm(original)


def test_bare_native_dispatch_is_separate_from_historical_proxy(native_config, tmp_path):
    arm = get_arm(native_config["native_arm"])
    assert arm.native_controller == "bare_event_native"
    assert not arm.repair and not arm.recover
    cell = {"arm": arm.name, "benchmark": "bfcl_base"}
    for stage in ("closed_loop", "common_prefix"):
        command = runner.run_command(native_config, cell, tmp_path, tmp_path / "profile.json", stage)
        assert "benchmarks.paper.c1" in command
        assert command[command.index("--arm") + 1] == arm.name
    server = runner.server_command(native_config, tmp_path, arm.name)
    assert "--c2kv-shadow-feature-layer" not in server
    runner._guard_checkpoint_serving_layout(native_config, cell)
    assert get_arm("c2kv4").native_controller is None
    plan, _ = runner.prepare(native_config, tmp_path / "plan", tmp_path / "sglang")
    bare_cells = [row for row in plan if row["method"] == "C2KV"]
    assert {row["benchmark"] for row in bare_cells} == {
        "bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox", "tau2"}
    assert all(row["arm"] == arm.name and "benchmarks.paper.c1" in row["command"]
               for row in bare_cells)


def test_native_bare_commands_have_no_s0_detector_or_recovery(native_config, tmp_path):
    delivery = c1.load_delivery()
    args = c1.delivery_args(native_config, "bfcl_base", tmp_path, ["multi_turn_base_0"], delivery)
    assert args.method == "c2kv_native"
    controller_path = tmp_path / "controller.json"
    controller_path.write_text("{}")
    bfcl, _ = delivery.commands_for_task(args, "multi_turn_base_0", controller_path)
    appworld = c1_appworld.server_command(native_config, "3d9a636_1", tmp_path, delivery, controller_path)
    for command in (bfcl, appworld):
        assert command[command.index("--view-mode") + 1] == "ac_gist_static"
        assert command[command.index("--ratio") + 1] == str(c1.RATIO)
        assert command[command.index("--generation-backend") + 1] == "sglang"
        assert "--s0-config" not in command
        assert "--shadow-feature-config" not in command
    assert appworld[appworld.index("--benchmark") + 1] == "acon_appworld"


def test_bare_profile_and_acceptance_require_no_recovery(native_config, tmp_path, monkeypatch):
    delivery = c1.load_delivery()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = c1.delivery_args(dict(native_config, checkpoint=str(checkpoint)), "bfcl_base", tmp_path, [], delivery)
    selected = copy.deepcopy(delivery.current.load_config())
    selected["checkpoint_selection"]["config_sha256"] = delivery.hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(delivery.current, "load_config", lambda: selected)
    controller, profile = delivery.build_profile(args)
    assert controller == {}
    assert profile["arm"] == native_config["native_arm"]
    assert profile["ratio"] == c1.RATIO
    assert delivery._model_name(args) == native_config["native_arm"]
    assert profile["detector"] == "disabled"
    assert profile["recovery_enabled"] is False
    assert profile["s0_allocator_enabled"] is False
    assert profile["max_generations_per_decision"] == 1
    metrics = {"native_generate_requests": 2, "detector_calls": 0,
               "generation_calls": 2, "decision_count": 2, "recovery_count": 0,
               "native_packing_present": True, "gist_cache_hits": 0, "compression_ratio": 2}
    assert all(delivery.functional_checks("c2kv_native", "disabled", metrics)["required"].values())
    assert not all(delivery.functional_checks("c2kv_native", "disabled", dict(metrics, recovery_count=1))["required"].values())
    assert not all(delivery.functional_checks("c2kv_native", "disabled", dict(metrics, generation_calls=3))["required"].values())


def test_bare_manifest_rejects_s0_mislabelling(native_config, tmp_path):
    c1.load_delivery()
    import native_bare
    path = tmp_path / "ready.json"
    manifest = {"view_mode": "ac_gist_static", "ratio": c1.RATIO,
                "model_name": native_config["native_arm"],
                "route_contract": {"recovery_enabled": False, "max_generations_per_decision": 1}}
    path.write_text(json.dumps(manifest))
    native_bare.validate_manifest(path, c1.RATIO)
    for model in (None, "c2kv_native_r4" if c1.RATIO == 8 else "c2kv_native_r8"):
        manifest["model_name"] = model
        path.write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="declared method"):
            native_bare.validate_manifest(path, c1.RATIO)
    manifest["model_name"] = native_config["native_arm"]
    manifest["view_mode"] = "ac_native_s0_lexical_raw_reserve_failed_operation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="declared method"):
        native_bare.validate_manifest(path, c1.RATIO)


def test_bare_summary_does_not_claim_c1(native_config):
    result = c1.summarize_scores("bfcl_base", [])
    assert result["arm"] == native_config["native_arm"]
    assert result["method"] == "C2KV"
    assert result["ratio"] == c1.RATIO


@pytest.mark.parametrize("benchmark", ["tau2", "toolsandbox", "acebench_agent"])
def test_extra_native_ratio_reaches_server_and_ready_validation(native_config, tmp_path, benchmark):
    delivery = c1.load_delivery()
    command = native_extra.server_command(native_config, benchmark, "task_1", tmp_path,
                                          delivery, tmp_path / "controller.json")
    assert command[command.index("--ratio") + 1] == str(c1.RATIO)
    assert command[command.index("--model-name") + 1] == native_config["native_arm"]
    assert "--s0-config" not in command and "--shadow-feature-config" not in command
    namespace, source_profile = native_extra.BENCHMARKS[benchmark]
    manifest = {"schema": "a-event-native-server-v1", "status": "ready",
                "benchmark": namespace, "source_profile": source_profile,
                "allowed_task_ids": ["task_1"], "model_name": native_config["native_arm"],
                "view_mode": "ac_gist_static", "ratio": c1.RATIO,
                "generation_backend": "sglang",
                "route_contract": {"recovery_enabled": False, "max_generations_per_decision": 1}}
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(manifest))
    native_extra.validate_ready_manifest(native_config, benchmark, "task_1", path,
                                         tmp_path / "controller.json")
    manifest["ratio"] = 4 if c1.RATIO == 8 else 8
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="selected arm"):
        native_extra.validate_ready_manifest(native_config, benchmark, "task_1", path,
                                             tmp_path / "controller.json")


def test_native_ratio_overlay_preserves_defaults_and_does_not_duplicate():
    config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    original = copy.deepcopy(config)
    expanded = runner.with_native_ratios(config, [8, 8, 4])
    assert config == original
    assert expanded["methods"][:-1] == config["methods"]
    assert expanded["methods"][-1]["arm"] == "c2kv_native_r8"
    with pytest.raises(ValueError, match="ratios 4 and 8"):
        runner.with_native_ratios(config, [16])


def test_native_ratio_cannot_resume_another_ratio_output(native_config, tmp_path, monkeypatch):
    delivery = c1.load_delivery()
    monkeypatch.setattr(delivery, "build_profile", lambda args: ({}, {"method": "c2kv_native", "ratio": c1.RATIO}))
    monkeypatch.setattr(delivery, "preflight_sglang_backend", lambda args: {"checks": {"fixture": True}})
    c1.prepare_native(native_config, "bfcl_base", tmp_path / "cell", [], delivery)
    profile_path = tmp_path / "cell" / "native" / "profile.json"
    frozen = profile_path.read_bytes()
    other = "c2kv_native_r8" if c1.RATIO == 4 else "c2kv_native_r4"
    c1.select_arm(other)
    with pytest.raises(ValueError, match="separate cell output"):
        c1.prepare_native(dict(native_config, native_arm=other), "bfcl_base",
                          tmp_path / "cell", [], delivery)
    assert profile_path.read_bytes() == frozen


def test_appworld_wrong_native_model_stops_before_official_execution(native_config, tmp_path, monkeypatch):
    started = []
    stopped = []
    monkeypatch.setattr(c1_appworld, "server_command", lambda *args: ["--model-name", native_config["native_arm"]])
    monkeypatch.setattr(c1_appworld, "_delivery_runner", lambda *args: SimpleNamespace(
        _stop_server=lambda *args: stopped.append(True)))
    monkeypatch.setattr(c1_appworld.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(c1_appworld, "_run_official_harness", lambda *args: started.append(True))

    def ready(process, path, deadline):
        path.parent.mkdir(parents=True)
        manifest = {"base_url": "http://127.0.0.1:34100/v1",
                    "model_name": "c2kv_native_r4" if c1.RATIO == 8 else "c2kv_native_r8",
                    "view_mode": "ac_gist_static", "ratio": c1.RATIO,
                    "route_contract": {"recovery_enabled": False, "max_generations_per_decision": 1}}
        path.write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(c1_appworld, "_wait_ready", ready)
    with pytest.raises(RuntimeError, match="declared method"):
        c1_appworld.run_task(native_config, "task_1", tmp_path / "native",
                            c1.DELIVERY, tmp_path / "controller.json")
    assert not started
    assert stopped == [True]
