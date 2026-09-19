"""The native bare arm must not silently inherit a C1/S0 controller."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1, c1_appworld, runner


@pytest.fixture
def native_config(tmp_path):
    original = c1.ARM
    c1.select_arm("c2kv_native_r4")
    config = json.loads(runner.DEFAULT_CONFIG.read_text())
    config.update(native_arm="c2kv_native_r4", sglang_source=str(tmp_path / "sglang"))
    (tmp_path / "bfcl" / "bfcl_eval").mkdir(parents=True)
    config["bfcl_dir"] = str(tmp_path / "bfcl")
    yield config
    c1.select_arm(original)


def test_bare_native_dispatch_is_separate_from_historical_proxy(native_config, tmp_path):
    arm = get_arm("c2kv_native_r4")
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
        "bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox"}
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
        assert command[command.index("--ratio") + 1] == "4"
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
    manifest = {"view_mode": "ac_gist_static", "ratio": 4,
                "route_contract": {"recovery_enabled": False, "max_generations_per_decision": 1}}
    path.write_text(json.dumps(manifest))
    native_bare.validate_manifest(path)
    manifest["view_mode"] = "ac_native_s0_lexical_raw_reserve_failed_operation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="declared method"):
        native_bare.validate_manifest(path)


def test_bare_summary_does_not_claim_c1(native_config):
    result = c1.summarize_scores("bfcl_base", [])
    assert result["arm"] == "c2kv_native_r4"
    assert result["method"] == "C2KV"
    assert result["ratio"] == 4
