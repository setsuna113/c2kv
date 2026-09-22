"""Offline rescore of held closed-loop cells (CPU only, no model or harness run)."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks import run as bench_run
from benchmarks.paper import rescore, runner
from benchmarks.paper.report import write_comparison

BENCH = Path(bench_run.__file__).resolve().parent
PORT = 43930
BUDGET = "hiagent_history_budget_exceeded"


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _session(key):
    return hashlib.sha256(json.dumps(["measurement_session", key], ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tau2_result(task, *, termination="agent_stop", reward=None):
    return {"task_id": task, "trial": 0, "termination_reason": termination,
            "messages": [{"role": "assistant", "content": "Done."}],
            "reward_info": {"reward": reward} if reward is not None else None}


def _root(tmp_path, benchmark, arm, extra_argv, write_cell):
    root = tmp_path / "root"
    cell_id = f"{benchmark}__{arm}"
    directory = root / "closed_loop" / cell_id
    cell = {"cell_id": cell_id, "benchmark": benchmark, "adapter": benchmark, "arm": arm,
            "method": "budget", "group": "budget", "category": "", "tool_context": "raw",
            "command": [sys.executable, str(BENCH / "run.py"), "--benchmark", benchmark,
                        "--arm", arm, "--upstream", "http://127.0.0.1:34000",
                        "--proxy-port", "34100", "--backend", "sglang", "--model", "c2kv-agent",
                        "--checkpoint", str(tmp_path / "checkpoint"),
                        "--out", str(directory), "--exact-out", "--run-name", cell_id,
                        "--num-workers", "1", *extra_argv],
            "replay_source": str(root / "closed_loop" / f"{benchmark}__full")}
    directory.mkdir(parents=True)
    (root / "config.resolved.json").write_text("{}", encoding="utf-8")
    (root / "commands.json").write_text(json.dumps([cell]), encoding="utf-8")
    (directory / "started.json").write_text(json.dumps({
        "stage": "closed_loop", "cell": cell, "config": {"proxy_port": PORT}}), encoding="utf-8")
    (directory / "checkpoint_profile.resolved.json").write_text(json.dumps({
        "profile_kind": "paper_deployment", "serving": {
            "doc_packing": "message", "max_doc_length": 512, "max_doc_num": 12,
            "query_projection": "base"}}), encoding="utf-8")
    (directory / "preflight.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (directory / "HOLD_cap_undeclared.txt").write_text("held by patrol\n", encoding="utf-8")
    write_cell(directory)
    return root, cell_id, directory


def _tau2_cell(directory):
    tau2_dir = directory.parents[2] / "tau2"
    (tau2_dir / "src" / "tau2").mkdir(parents=True)
    (directory / "tau2_protocol.json").write_text(json.dumps({
        "suite": "airline", "split": "base", "task_ids": ["11", "12"], "num_trials": 1,
        "source": str(tau2_dir), "python": "tau2-python", "native": False,
        "record_prefixes": None, "command": ["tau2", "run", "--save-to", "cell_ab12cd3"]}),
        encoding="utf-8")
    (directory / "official").mkdir()
    (directory / "official" / "results.json").write_text(json.dumps({"simulations": [
        _tau2_result("11", termination="infrastructure_error"), _tau2_result("12")]}),
        encoding="utf-8")
    _jsonl(directory / "measurement" / "harness_events.jsonl", [
        {"event_type": "episode_start", "episode_id": "11"},
        {"event_type": "decision", "episode_id": "11", "error": {"status_code": 422}},
        {"event_type": "episode_end", "episode_id": "11", "status": "error"},
        {"event_type": "episode_start", "episode_id": "12"},
        {"event_type": "decision", "episode_id": "12"},
        {"event_type": "episode_end", "episode_id": "12", "status": "ok"},
    ])
    _jsonl(directory / "logs" / f"proxy_hiagent_full_b192_{PORT}.jsonl", [
        {"conv_id": _session("12"), "status": "ok", "textarm": {"history_compressed": True}},
        {"conv_id": _session("11"), "status": BUDGET, "error_kind": BUDGET},
    ])


def _official_tau2(monkeypatch):
    adapter = bench_run.ADAPTERS["tau2"]
    calls = []

    def run_owned(command, **kwargs):
        calls.append(command)
        if "evaluate-trajs" in command:
            sims = Path(command[command.index("-o") + 1])
            (sims / "updated_results.json").write_text(json.dumps({"simulations": [
                _tau2_result("11", termination="infrastructure_error"),
                _tau2_result("12", reward=1.0)]}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setattr(adapter, "run_owned", run_owned)
    return calls


def _tau2_root(tmp_path):
    return _root(tmp_path, "tau2", "hiagent_full_b192",
                 ["--benchmark-dir", str(tmp_path / "tau2"), "--bench-python", "tau2-python",
                  "--task-set", "airline", "--tau2-num-trials", "1"], _tau2_cell)


def _digest(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def test_tau2_rescore_publishes_canonical_summary_and_leaves_run_artifacts(tmp_path, monkeypatch, capsys):
    root, cell_id, directory = _tau2_root(tmp_path)
    before = _digest(directory)
    calls = _official_tau2(monkeypatch)
    runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    printed = json.loads(capsys.readouterr().out)
    summary = json.loads((directory / "summary_hiagent_full_b192.json").read_text(encoding="utf-8"))
    assert printed == [{"cell_id": cell_id, "semantic_score": 0.5, "n": 2,
                        "task_failure_counts": {BUDGET: 1},
                        "result_scope": "preliminary, n=1; offline rescore of an existing run"}]
    assert summary["task_failures"] == {"11": BUDGET}
    assert [row["semantic_score"] for row in summary["task_rows"]] == [0.0, 1.0]
    # run.py's envelope, rebuilt from the cell's own receipts
    assert summary["arm"] == "hiagent_full_b192" and summary["benchmark"] == "tau2"
    assert summary["doc_packing"] == "message" and summary["preflight"] == {"ok": True}
    assert summary["textarm_summary"]["history_compressed_requests"] == 1
    assert summary["request_log"] == str(directory / "logs" / f"proxy_hiagent_full_b192_{PORT}.jsonl")
    provenance = summary["rescore"]
    assert provenance["score_provenance"] == "offline_rescore" and provenance["no_model_execution"]
    assert set(provenance["inputs"]) == {
        "started.json", "checkpoint_profile.resolved.json", "preflight.json",
        "tau2_protocol.json", "official/results.json", "measurement/harness_events.jsonl",
        f"logs/proxy_hiagent_full_b192_{PORT}.jsonl"}
    assert list(provenance["hold_evidence"]) == ["HOLD_cap_undeclared.txt"]
    receipt = json.loads((directory / "rescore.json").read_text(encoding="utf-8"))
    assert receipt["summary_sha256"] == hashlib.sha256(
        (directory / "summary_hiagent_full_b192.json").read_bytes()).hexdigest()
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    assert complete["score_provenance"] == "offline_rescore"
    # every pre-existing artifact is byte-identical; scoring wrote only rescore/
    after = _digest(directory)
    assert {path: after[path] for path in before} == before
    assert sorted(set(after) - set(before)) == [
        "complete.json", "rescore.json", "rescore/official/results.json",
        "rescore/official/updated_results.json", "rescore/sims/results.json",
        "rescore/sims/updated_results.json", "summary_hiagent_full_b192.json"]
    assert any("evaluate-trajs" in command for command in calls)


def test_rescore_refuses_complete_cells_and_never_overwrites(tmp_path, monkeypatch):
    root, cell_id, directory = _tau2_root(tmp_path)
    _official_tau2(monkeypatch)
    runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    published = _digest(directory)
    with pytest.raises(rescore.RescoreRefused, match="already complete"):
        runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    assert _digest(directory) == published


def test_rescore_refuses_inputs_that_change_while_scoring(tmp_path, monkeypatch):
    root, cell_id, directory = _tau2_root(tmp_path)
    _official_tau2(monkeypatch)
    adapter = bench_run.ADAPTERS["tau2"]
    original = adapter.rescore

    def racing_writer(ctx, workspace):
        summary = original(ctx, workspace)
        with (directory / "measurement" / "harness_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event_type": "late"}) + "\n")
        return summary

    monkeypatch.setattr(adapter, "rescore", racing_writer)
    with pytest.raises(rescore.RescoreRefused, match="inputs changed"):
        runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    assert not (directory / "summary_hiagent_full_b192.json").exists()
    assert not (directory / "complete.json").exists()
    with pytest.raises(rescore.RescoreRefused, match="previous rescore workspace"):
        runner.main(["rescore", "--output", str(root), "--cells", cell_id])


@pytest.mark.parametrize("change", ["plan", "infra", "log", "differs"])
def test_rescore_refuses_changed_or_unscorable_cells(tmp_path, monkeypatch, change):
    root, cell_id, directory = _tau2_root(tmp_path)
    _official_tau2(monkeypatch)
    if change == "plan":
        plan = json.loads((root / "commands.json").read_text())
        plan[0]["command"].append("--tau2-max-steps=5")
        (root / "commands.json").write_text(json.dumps(plan))
        match = "differs from the plan"
    elif change == "infra":
        (directory / "infra_failure.json").write_text("{}")
        match = "upstream infrastructure failure"
    elif change == "log":
        _jsonl(directory / "logs" / "proxy_hiagent_full_b192_1.jsonl", [])
        match = "exactly the run's request log"
    else:
        sims = tmp_path / "tau2" / "data" / "simulations" / "cell_ab12cd3"
        sims.mkdir(parents=True)
        (sims / "results.json").write_text("{}")
        match = "differs from the run's simulations"
    with pytest.raises((rescore.RescoreRefused, ValueError), match=match):
        runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    assert not (directory / "complete.json").exists()


@pytest.mark.parametrize("argv", [
    ["rescore"], ["rescore", "--cells", "x", "--stage", "common_prefix"],
    ["rescore", "--cells", "x", "--generation-timeout", "1800"],
    ["rescore", "--cells", "x", "--task-subset", "x=1"],
])
def test_rescore_cli_reads_only_the_frozen_root(tmp_path, argv):
    with pytest.raises(SystemExit):
        runner.main([*argv, "--output", str(tmp_path)])


def _toolsandbox_cell(directory):
    code = "acon_history_budget_exceeded"
    (directory / "toolsandbox_protocol.json").write_text(json.dumps({
        "suite": "subset", "scenarios": ["capped", "complete"]}), encoding="utf-8")
    (directory / "scenario_manifest.json").write_text(json.dumps({
        "scenario_ids": ["capped", "complete"], "expected": 2}), encoding="utf-8")
    (directory / "agent_run").mkdir()
    (directory / "agent_run" / "result_summary.json").write_text(json.dumps({
        "per_scenario_results": [
            {"name": "complete", "similarity": 0.8},
            {"name": "capped", "similarity": None,
             "exception_type": "UnprocessableEntityError",
             "traceback": f"Error code: 422 - {{'error': {{'code': '{code}'}}}}"}]}),
        encoding="utf-8")
    _jsonl(directory / "measurement" / "harness_events.jsonl", [
        {"event_type": "episode_start", "episode_id": "capped", "episode_instance_id": "i-1"}])
    _jsonl(directory / "logs" / f"proxy_acon_hist_ut_co_b128_{PORT}.jsonl", [
        {"conv_id": _session("i-1"), "status": code, "error_kind": code}])


def test_toolsandbox_rescore_matches_the_live_collector(tmp_path):
    root, cell_id, directory = _root(
        tmp_path, "toolsandbox", "acon_hist_ut_co_b128",
        ["--toolsandbox-dir", str(tmp_path / "ts"), "--bench-python", "ts-python",
         "--ts-parallel", "1", "--ts-scenarios", "capped,complete"], _toolsandbox_cell)
    receipts = rescore.rescore_cells(
        json.loads((root / "commands.json").read_text()), root, {cell_id})
    summary = json.loads((directory / "summary_acon_hist_ut_co_b128.json").read_text())
    live = bench_run.ADAPTERS["toolsandbox"].score_cli_run(directory, scenarios=["capped", "complete"])
    assert summary["semantic_score"] == live["semantic_score"] == 0.4
    assert summary["task_failures"] == live["task_failures"]
    assert summary["cost_join"] == bench_run.ADAPTERS["toolsandbox"].COST_JOIN
    assert receipts[0]["task_failure_counts"] == {
        "acon_history_budget_exceeded": 1, "hiagent_invalid_retrieval": 0}


def test_aggregate_and_comparison_mark_only_rescored_cells(tmp_path, monkeypatch):
    root, cell_id, directory = _tau2_root(tmp_path)
    _official_tau2(monkeypatch)
    runner.main(["rescore", "--output", str(root), "--cells", cell_id])
    for name in ("proxy_telemetry.jsonl", "server_telemetry.jsonl"):
        (directory / name).write_text("", encoding="utf-8")
    plan = json.loads((root / "commands.json").read_text())
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    _, coverage_path = runner.aggregate_results({"bench_python": "python"}, plan, root,
                                                ["closed_loop"], {cell_id})
    entry = json.loads(coverage_path.read_text())["cells"][0]
    assert entry["score_provenance"] == "offline_rescore"
    assert entry["rescore_receipt"] == str(directory / "rescore.json")
    (directory / "measurement_summary.json").write_text("{}", encoding="utf-8")
    rows = write_comparison(root, plan)
    assert [row["score_provenance"] for row in rows] == ["offline_rescore"]
    assert rows[0]["result_status"] == "preliminary, n=1"
    (directory / "rescore.json").unlink()
    assert "score_provenance" not in write_comparison(root, plan)[0]
