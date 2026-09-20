"""Tau2 is an orthogonal benchmark axis, not a new memory algorithm."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.paper import c1, native_extra, runner, tau2
from benchmarks.paper.candidate_matrix import with_candidate_methods


def config():
    return json.loads(runner.DEFAULT_CONFIG.read_text())


def test_tau2_extends_matrix_without_changing_existing_cells(tmp_path):
    current = config()
    historical = copy.deepcopy(current)
    historical["benchmarks"] = [b for b in historical["benchmarks"] if b["name"] != "tau2"]
    assert runner.extension_problem(historical, current, tmp_path, tmp_path) is None
    cells = [row for row in runner.cells(current) if row["benchmark"] == "tau2"]
    assert {row["arm"] for row in cells} == {
        method["arm"] for method in current["methods"] if "tau2" in method.get("benchmarks", [])}
    full = next(row for row in cells if row["arm"] == "full" and row["tool_context"] == "raw")
    command = runner.run_command(current, full, tmp_path, tmp_path / "profile.json")
    assert command[command.index("--task-set") + 1] == "airline"
    assert "--record-prefixes" in command
    assert "--acon-dir" not in command
    from benchmarks.run import build_parser
    parsed = build_parser().parse_args(command[2:])
    assert parsed.benchmark_dir == Path(current["tau2_dir"])
    for row in cells:
        server = runner.server_command(current, tmp_path, row["arm"], "tau2")
        assert server[server.index("--max-running-requests") + 1] == "2"


def test_tau2_native_closed_loop_and_replay_use_shared_controller(tmp_path):
    current = config()
    current["sglang_source"] = str(tmp_path)
    (tmp_path / "controller.json").write_text("{}")
    for arm in ("c2kv_native_r4", "c2kv_c1_t02_r8", "c2kv_c1_off_r8"):
        current["native_arm"] = arm
        command = native_extra.server_command(
            current, "tau2", "0", tmp_path, c1.DELIVERY, tmp_path / "controller.json")
        assert command[command.index("--benchmark") + 1] == "tau2"
        assert command[command.index("--source-profile") + 1] == "openai-single-task-v1"
        cell = {"arm": arm, "benchmark": "tau2"}
        for stage in ("closed_loop", "common_prefix"):
            launch = runner.run_command(current, cell, tmp_path, tmp_path / "profile.json", stage)
            assert "benchmarks.paper.c1" in launch
            assert ("--prefixes" in launch) == (stage == "common_prefix")


def test_tau2_delivery_args_bind_native_task_and_checkout_without_changing_bfcl(tmp_path):
    current = config()
    current["sglang_source"] = str(tmp_path)
    delivery = c1.load_delivery()
    tau2_args = c1.delivery_args(current, "tau2", tmp_path, ["0"], delivery)
    assert tau2_args.benchmark == "tau2"
    assert tau2_args.benchmark_dir == Path(current["tau2_dir"])
    assert tau2_args.tau2_dir == Path(current["tau2_dir"])
    assert tau2_args.tau2_python == current.get("tau2_python", current["bench_python"])
    assert tau2_args.tau2_task_id == ["0"]
    assert tau2_args.task_set == current.get("tau2_task_set", "airline")
    assert tau2_args.user_base_url == f"http://127.0.0.1:{current['server_port']}"
    assert delivery._identities(tau2_args) == ["0"]

    bfcl_args = c1.delivery_args(current, "bfcl_base", tmp_path, ["multi_turn_base_26"], delivery)
    assert bfcl_args.benchmark == "bfcl"
    assert bfcl_args.benchmark_dir == Path(current["bfcl_dir"])
    assert bfcl_args.task_id == ["multi_turn_base_26"]
    assert bfcl_args.tau2_dir is None and bfcl_args.tau2_task_id == []


def test_tau2_candidates_and_budget_axes_are_not_reimplemented():
    current = with_candidate_methods(config(), ("goal_rescue",), ("tau2",))
    assert any(row["benchmark"] == "tau2" and row["arm"] == "c2kv_goal_rescue_r8"
               for row in runner.cells(current))
    assert "tau2" in runner.BUDGET_TEXT_BENCHMARKS


def test_tau2_selection_and_official_scoring_use_same_adapter(tmp_path, monkeypatch):
    from benchmarks.adapters import tau2_adapter
    current = config()
    monkeypatch.setattr(tau2_adapter, "selected_task_ids", lambda *a, **kw: ["0", "1"])
    assert c1.selected_tasks(current, "tau2", ["1"]) == ["1"]
    with pytest.raises(ValueError, match="official split"):
        c1.selected_tasks(current, "tau2", ["unknown"])
    seen = {}
    def run(*args, **kwargs):
        seen.update(kwargs)
        Path(args[2]).mkdir(parents=True)
        return {"n": 1, "task_ids": ["1"], "semantic_score": 0.0,
                "task_rows": [{"termination": "max_steps"}]}
    monkeypatch.setattr(tau2_adapter, "run_tau2", run)
    official = native_extra._run_official(current, "tau2", "1", tmp_path, "http://agent", "native")
    assert official["task_id"] == "1"
    assert official["task_rows"][0]["semantic_score"] == 0.0
    assert official["task_rows"][0]["normal_termination"] is False
    assert seen["native"] is True and seen["user_model"] == current["model"]
    assert seen["task_ids"] == ["1"]


def test_paper_tau2_rejects_ambiguous_multi_trial_identity():
    current = config()
    current["tau2_num_trials"] = 2
    with pytest.raises(ValueError, match="one trial"):
        tau2.options(current)


def test_native_replay_rejects_missing_official_task(tmp_path, monkeypatch):
    from benchmarks.measurement.telemetry import canonical_sha256
    payload = {"messages": [{"role": "user", "content": "test"}],
               "c2kv_measurement_session_id": "0"}
    record = {"event_type": "recorded_prefix", "source_arm": "full",
              "conversation_id": "conversation", "replay_payload": payload,
              "canonical_sha256": canonical_sha256(payload)}
    prefixes = tmp_path / "prefixes.jsonl"
    prefixes.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(c1, "load_delivery", lambda: None)
    monkeypatch.setattr(c1, "selected_tasks", lambda *a: ["0", "1"])
    with pytest.raises(ValueError, match="exactly the selected"):
        c1.run_common_prefix(config(), "tau2", tmp_path, prefixes)
