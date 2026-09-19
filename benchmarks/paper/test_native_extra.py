import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks.paper import native_extra
from benchmarks.paper.candidate_matrix import REPAIR_VARIANTS, VARIANT_TO_ARM


def _config(tmp_path):
    return {
        "checkpoint": str(tmp_path / "checkpoint"),
        "model": "c2kv-agent",
        "bench_python": sys.executable,
        "acebench_python": sys.executable,
        "toolsandbox_python": sys.executable,
        "acebench_dir": str(tmp_path / "acebench"),
        "toolsandbox_dir": str(tmp_path / "ToolSandbox"),
        "acebench_language": "en",
        "toolsandbox_suite": "full",
        "toolsandbox_scenarios": [],
        "upstream": "http://127.0.0.1:34000",
        "proxy_port": 34100,
        "c1": {"task_timeout": 120},
    }


def test_official_ace_tasks_and_subset(tmp_path):
    config = _config(tmp_path)
    root = Path(config["acebench_dir"])
    root.mkdir()
    (root / "category.py").write_text(
        "ACE_DATA_CATEGORY = {'agent': ['agent_multi_turn', 'agent_multi_step']}\n",
        encoding="utf-8",
    )
    data = root / "data_all" / "data_en"
    data.mkdir(parents=True)
    (data / "data_agent_multi_turn.json").write_text(
        '{"id":"agent_multi_turn_1"}\n', encoding="utf-8")
    (data / "data_agent_multi_step.json").write_text(
        '{"id":"agent_multi_step_2"}\n', encoding="utf-8")
    assert native_extra.selected_tasks(config, "acebench_agent") == [
        "agent_multi_turn_1", "agent_multi_step_2"]
    assert native_extra.selected_tasks(config, "acebench_agent", ["agent_multi_step_2"]) == [
        "agent_multi_step_2"]
    with pytest.raises(ValueError, match="official split"):
        native_extra.selected_tasks(config, "acebench_agent", ["unknown"])


def test_toolsandbox_uses_official_resolver_and_checks_configured_subset(tmp_path, monkeypatch):
    config = _config(tmp_path)
    Path(config["toolsandbox_dir"]).mkdir()
    observed = {}

    def resolve(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout=json.dumps(["wifi_off", "get_wifi"]))

    monkeypatch.setattr(native_extra, "run_owned", resolve)
    assert native_extra.selected_tasks(config, "toolsandbox") == ["wifi_off", "get_wifi"]
    assert "resolve_scenarios" in observed["command"][2]
    assert observed["kwargs"]["cwd"] == Path(config["toolsandbox_dir"])
    config["toolsandbox_scenarios"] = ["get_wifi"]
    assert native_extra.selected_tasks(config, "toolsandbox") == ["get_wifi"]
    with pytest.raises(ValueError, match="official split"):
        native_extra.selected_tasks(config, "toolsandbox", ["missing"])


@pytest.mark.parametrize("benchmark,namespace,profile", [
    ("acebench_agent", "acebench", "acebench-text-actions-v1"),
    ("toolsandbox", "toolsandbox", "openai-single-task-v1"),
])
def test_server_command_is_native_bare(tmp_path, benchmark, namespace, profile):
    config = _config(tmp_path)
    root = Path(__file__).resolve().parents[2] / "experiments" / "history_system"
    command = native_extra.server_command(
        config, benchmark, "task_1", tmp_path / "native", root,
        tmp_path / "controller.json")
    def value(flag):
        return command[command.index(flag) + 1]
    assert value("--view-mode") == "ac_gist_static"
    assert value("--ratio") == "4"
    assert value("--benchmark") == namespace
    assert value("--source-profile") == profile
    assert value("--model-name") == "c2kv_native_r4"
    assert "--s0-config" not in command
    assert "--shadow-feature-config" not in command
    if benchmark == "acebench_agent":
        assert value("--max-new-tokens") == "1000"


@pytest.mark.parametrize("benchmark", ["acebench_agent", "toolsandbox"])
def test_tool_on_c1_uses_recovery_controller_and_tool_flags(tmp_path, benchmark):
    config = _config(tmp_path)
    config.update(native_arm="c2kv_c1_t02_r8", tool_memory="t0:r8",
                  tool_checkpoint=str(tmp_path / "T0" / "checkpoint-500"),
                  tool_budget_tokens=512)
    config["c1"].update(detector="t02_risk", history_variant="H0",
                        recovery_rounds=1)
    root = Path(__file__).resolve().parents[2] / "experiments" / "history_system"
    controller = tmp_path / "controller.json"
    controller.write_text("{}", encoding="utf-8")
    command = native_extra.server_command(
        config, benchmark, "task_1", tmp_path / "native", root, controller)
    def value(flag):
        return command[command.index(flag) + 1]
    assert value("--view-mode") == "ac_native_s0_lexical_raw_reserve_failed_operation"
    assert value("--s0-config") == str(controller.resolve())
    assert value("--model-name") == "c1_t02_risk"
    assert value("--tool-memory") == "t0:r8"
    assert value("--tool-checkpoint") == str(Path(config["tool_checkpoint"]).resolve())
    assert value("--tool-budget-tokens") == "512"


@pytest.mark.parametrize("arm,detector,ratio,model,variant", [
    ("c2kv_native_r4", "disabled", 4, "c2kv_native_r4", None),
    ("c2kv_c1_t02_r8", "t02_risk", 8, "c1_t02_risk", None),
    ("c2kv_c1_t02_r4", "t02_risk", 4, "c1_t02_risk", None),
    ("c2kv_c1_t02_r8", "d3_hybrid", 8, "c1_d3_hybrid", None),
    ("c2kv_c1_t02_r4", "d3_hybrid", 4, "c1_d3_hybrid", None),
    *[(arm, "t02_risk", 8, f"c2kv_{variant}", variant)
      for variant, arm in VARIANT_TO_ARM.items()],
])
def test_ace_closed_loop_and_replay_command_use_actual_arm(
        tmp_path, arm, detector, ratio, model, variant):
    config = _config(tmp_path)
    config["native_arm"] = arm
    config["c1"]["detector"] = detector
    root = Path(__file__).resolve().parents[2] / "experiments" / "history_system"
    controller = tmp_path / "controller.json"
    controller.write_text("{}", encoding="utf-8")
    command = native_extra.server_command(
        config, "acebench_agent", "task_1", tmp_path / "native", root, controller)
    replay = native_extra.controller_command(
        config, "acebench_agent", "task_1", tmp_path / "native", root, controller)
    assert replay == command
    def value(flag):
        return command[command.index(flag) + 1]
    identity = native_extra.arm_identity(config)
    assert identity["method"] == ("c2kv_native" if arm == "c2kv_native_r4" else "proposed")
    assert identity["detector"] == ("disabled" if arm == "c2kv_native_r4" or variant in REPAIR_VARIANTS
                                    else "t02_risk" if variant else detector)
    assert identity["candidate_algorithm"] == variant
    assert value("--ratio") == str(ratio)
    assert value("--model-name") == model
    assert value("--benchmark") == "acebench"
    assert value("--source-profile") == "acebench-text-actions-v1"
    if arm == "c2kv_native_r4":
        assert value("--view-mode") == "ac_gist_static"
        assert "--s0-config" not in command
    else:
        assert value("--view-mode") == "ac_native_s0_lexical_raw_reserve_failed_operation"
        assert value("--s0-config") == str(controller.resolve())
        assert arm in value("--run-id")


@pytest.mark.parametrize("arm,detector,variant", [
    ("c2kv_native_r4", "disabled", None),
    ("c2kv_c1_t02_r4", "d3_hybrid", None),
    *[(arm, "disabled" if variant in REPAIR_VARIANTS else "t02_risk", variant)
      for variant, arm in VARIANT_TO_ARM.items()],
])
def test_ace_official_task_acceptance_uses_actual_arm_identity(
        tmp_path, monkeypatch, arm, detector, variant):
    config = _config(tmp_path)
    config["native_arm"] = arm
    config["c1"]["detector"] = "d3_hybrid"
    delivery = tmp_path / "delivery"
    (delivery / "runtime").mkdir(parents=True)
    seen = []

    class FakeRunner:
        def _stop_server(self, process, supervisor):
            final = supervisor.parent / "server" / "final.json"
            final.parent.mkdir()
            final.write_text(json.dumps({"status": "ok", "journal_summary": {
                "completed": 1, "failed": 0, "pending": 0}}), encoding="utf-8")

    class FakeDelivery:
        def summarize_task(self, *args):
            return {"decision_count": 1}

        def functional_checks(self, method, selected_detector, metrics, candidate):
            seen.append((method, selected_detector, candidate, metrics))
            return {"required": {"arm_identity": True}}

    monkeypatch.setattr(native_extra, "server_command",
                        lambda *args: [sys.executable, "--model-name", "test-model"])
    monkeypatch.setattr(native_extra.c1_appworld, "_delivery_path", lambda _: delivery)
    monkeypatch.setattr(native_extra.c1_appworld, "_delivery_runner", lambda _: FakeRunner())
    monkeypatch.setattr(native_extra.c1_appworld, "_delivery_run_c1", lambda *_: FakeDelivery())
    monkeypatch.setattr(native_extra.c1_appworld, "_wait_ready",
                        lambda *args: {"base_url": "http://127.0.0.1:34100/v1"})
    monkeypatch.setattr(native_extra, "_run_official",
                        lambda *args: {"n": 1, "semantic_score": 1.0})
    monkeypatch.setattr(native_extra, "validate_ready_manifest", lambda *args: None)
    monkeypatch.setattr(native_extra.subprocess, "Popen",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    native_extra.run_task(config, "acebench_agent", "task_1", tmp_path / "native",
                          delivery, tmp_path / "controller.json")
    assert seen == [("c2kv_native" if arm == "c2kv_native_r4" else "proposed",
                     detector, variant, {"decision_count": 1})]


@pytest.mark.parametrize("arm,detector,variant", [
    ("c2kv_native_r4", "disabled", None),
    ("c2kv_c1_t02_r8", "t02_risk", None),
    ("c2kv_c1_t02_r4", "d3_hybrid", None),
    *[(arm, "disabled" if variant in REPAIR_VARIANTS else "t02_risk", variant)
      for variant, arm in VARIANT_TO_ARM.items()],
])
def test_ready_manifest_binds_loaded_controller_and_candidate_variant(
        tmp_path, arm, detector, variant):
    config = _config(tmp_path)
    config["native_arm"] = arm
    config["c1"]["detector"] = detector
    identity = native_extra.arm_identity(config)
    controller = tmp_path / "controller.json"
    controller_config = (
        {"candidate_algorithm": {"variant": variant}} if variant in REPAIR_VARIANTS else
        {"candidate_algorithm": {"variant": variant, "risk_threshold": 0.5,
                                 "risk_artifact": {"model_kind": "c1_risk_logistic"}}} if variant else
        {"post_draft_recovery": {"gate": "prefill_linear_head"},
         "d3_hybrid_recovery": True} if detector == "d3_hybrid" else
        {"gp_experiments": {"set_selector": "risk", "selector_artifact": {
            "model_kind": "c1_risk_logistic"}}})
    controller.write_text(json.dumps(controller_config), encoding="utf-8")
    route = ({"recovery_enabled": False, "max_generations_per_decision": 1}
             if arm == "c2kv_native_r4" else
             {"recovery_enabled": True, "max_generations_per_decision": 2,
              "baseline_identity": f"{'c2kv-source-repair-v1' if variant in REPAIR_VARIANTS else 'c2kv-paper-candidates-v1'}:{variant}"}
             if variant else {})
    manifest = {
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": "acebench", "source_profile": "acebench-text-actions-v1",
        "allowed_task_ids": ["task_1"], "model_name": identity["model_name"],
        "view_mode": ("ac_gist_static" if arm == "c2kv_native_r4" else
                      "ac_native_s0_lexical_raw_reserve_failed_operation"),
        "ratio": identity["ratio"], "generation_backend": "sglang",
        "route_contract": route,
    }
    if arm != "c2kv_native_r4":
        manifest["s0_controller_contract"] = {
            "source": str(controller.resolve()), "config": controller_config,
            "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        }
    if variant:
        manifest["candidate_algorithm"] = {"variant": variant, "stable_call_ids": True}
    ready = tmp_path / "ready.json"
    ready.write_text(json.dumps(manifest), encoding="utf-8")
    native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
    if arm == "c2kv_c1_t02_r4":
        from benchmarks.toolmemory import parse_tool_memory_spec
        config["tool_memory"] = "t0:r8"
        config["tool_checkpoint"] = str(tmp_path / "tool-checkpoint")
        with pytest.raises(RuntimeError, match="tool memory contract"):
            native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
        manifest["tool_memory_contract"] = {
            "spec": parse_tool_memory_spec("t0:r8").as_dict(),
            "tool_budget_tokens": None,
            "checkpoint": {"checkpoint": config["tool_checkpoint"]},
        }
        ready.write_text(json.dumps(manifest), encoding="utf-8")
        native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
    manifest["ratio"] = 99
    ready.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected arm"):
        native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
    if arm != "c2kv_native_r4":
        manifest["ratio"] = identity["ratio"]
        manifest["s0_controller_contract"]["sha256"] = "0" * 64
        ready.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="different S0 controller"):
            native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
    if variant:
        manifest["s0_controller_contract"]["sha256"] = hashlib.sha256(controller.read_bytes()).hexdigest()
        manifest["candidate_algorithm"]["variant"] = "wrong_variant"
        ready.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="candidate controller identity"):
            native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)
    elif arm != "c2kv_native_r4":
        wrong = ({"gp_experiments": {"set_selector": "risk", "selector_artifact": {
            "model_kind": "c1_risk_logistic"}}} if detector == "d3_hybrid" else
                 {"post_draft_recovery": {"gate": "prefill_linear_head"},
                  "d3_hybrid_recovery": True})
        controller.write_text(json.dumps(wrong), encoding="utf-8")
        manifest["s0_controller_contract"] = {
            "source": str(controller.resolve()), "config": wrong,
            "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        }
        ready.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="detector controller identity"):
            native_extra.validate_ready_manifest(config, "acebench_agent", "task_1", ready, controller)


def test_ace_official_score_and_user_model_are_bound(tmp_path, monkeypatch):
    config = _config(tmp_path)
    task = "agent_multi_turn_1"
    output = tmp_path / "task"
    (output / "acebench").mkdir(parents=True)
    observed = {}

    def run_acebench(*args, **kwargs):
        observed.update(kwargs)
        observed["tool_context_env"] = __import__("os").environ.get("C2KV_TOOL_CONTEXT_ON")
        return {"n": 1, "semantic_score": 1.0,
                "selection": {"sources": [{"selected_ids": [task]}]}}

    from benchmarks.adapters import acebench_adapter
    monkeypatch.setattr(acebench_adapter, "run_acebench", run_acebench)
    official = native_extra._run_official(config, "acebench_agent", task,
                                          output, "http://127.0.0.1:34100/v1",
                                          "c2kv_native_r4")
    assert observed["model"] == "c2kv_native_r4"
    assert observed["user_model"] == "c2kv-agent"
    assert official["task_rows"][0]["semantic_score"] == 1.0
    assert observed["tool_context_env"] is None
    with mock.patch.object(acebench_adapter, "run_acebench", return_value={
        "n": 0, "semantic_score": 1.0, "selection": {"sources": []}}):
        with pytest.raises(RuntimeError, match="selection or scoring"):
            native_extra._run_official(config, "acebench_agent", task, output,
                                       "http://127.0.0.1:34100/v1", "c2kv_native_r4")


def test_ace_tool_annotation_hook_is_enabled_only_during_on_harness(tmp_path, monkeypatch):
    import os
    from benchmarks.adapters import acebench_adapter

    config = _config(tmp_path)
    config["tool_memory"] = "t0:r8"
    task = "agent_multi_turn_1"
    output = tmp_path / "task"
    (output / "acebench").mkdir(parents=True)
    observed = []

    def run_acebench(*args, **kwargs):
        observed.append(os.environ.get("C2KV_TOOL_CONTEXT_ON"))
        return {"n": 1, "semantic_score": 1.0,
                "selection": {"sources": [{"selected_ids": [task]}]}}

    monkeypatch.setattr(acebench_adapter, "run_acebench", run_acebench)
    previous = os.environ.get("C2KV_TOOL_CONTEXT_ON")
    native_extra._run_official(config, "acebench_agent", task, output,
                               "http://127.0.0.1:34100/v1", "c1_t02_risk")
    assert observed == ["1"]
    assert os.environ.get("C2KV_TOOL_CONTEXT_ON") == previous


def test_replay_requires_real_ace_receipts():
    payload = {"messages": [{"role": "system", "content": "system"},
                            {"role": "user", "content": "question"}],
               "temperature": 0.001, "top_p": 1, "max_tokens": 1000,
               "stream": False, "c2kv_measurement_session_id": "old"}
    with pytest.raises(ValueError, match="recorded execution receipts"):
        native_extra.replay_payload(payload, "agent_multi_turn_1", 0)
    payload["c2kv_ace_source"] = {"version": "acebench-text-actions-v1", "receipts": []}
    normalized = native_extra.replay_payload(payload, "agent_multi_turn_1", 0)
    assert normalized["messages"] == payload["messages"]
    assert normalized["temperature"] == 0.001
    assert normalized["max_tokens"] == 1000
    assert normalized["store"] is False
    assert "c2kv_measurement_session_id" not in normalized
    assert normalized["c2kv_eval_context"]["task_id"] == "agent_multi_turn_1"
