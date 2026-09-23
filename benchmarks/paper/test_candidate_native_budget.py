"""Candidate native history budgets cover every supported benchmark (c1v2rv).

CPU only. Budget variants of a candidate arm now follow its configured
benchmarks; other native arms keep the BFCL-only sweep; a task-bound capacity
failure stays a scored task-local method failure on the non-BFCL adapters.
"""
import json
import subprocess
import sys
import types

import pytest

from benchmarks.paper import c1 as paper_c1, native_extra, runner
from benchmarks.paper.candidate_matrix import (
    BFCL_BENCHMARKS, SUPPORTED_BENCHMARKS, native_budget_benchmarks, with_candidate_methods,
)

ARM = "c2kv_c1_v2_verified_r8"
EXTRA = ("tau2", "toolsandbox", "acebench_agent", "appworld")
BUDGETS = (128, 256, 512)


def config(benchmarks):
    base = dict(json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8")),
                history_kv_budget_tokens=256)
    return with_candidate_methods(base, ("c1_v2_verified",), benchmarks)


def with_budgets(cfg, arm=ARM):
    for budget in BUDGETS:
        cfg = runner.with_native_history_budget(cfg, arm, budget)
    return cfg


def test_budget_variants_cover_every_configured_candidate_benchmark(tmp_path):
    original = config(("bfcl_base", *EXTRA))
    output = tmp_path / "results"
    old, _ = runner.prepare(original, output, tmp_path / "engine")
    plan, profile = runner.prepare(with_budgets(original), output, tmp_path / "engine")
    by_id = {row["cell_id"]: row for row in plan}
    assert all(by_id[row["cell_id"]] == row for row in old)
    assert len(plan) == len(old) + len(BUDGETS) * (1 + len(EXTRA))
    for benchmark in ("bfcl_base", *EXTRA):
        for budget in BUDGETS:
            cell = by_id[f"{benchmark}__{ARM}_b{budget}"]
            assert cell["arm"] == ARM and cell["history_budget_tokens"] == budget
            argv = runner.run_command(with_budgets(original), cell,
                                      output / "closed_loop" / cell["cell_id"], profile)
            assert argv[argv.index("--benchmark") + 1] == benchmark
            assert argv[argv.index("--history-budget-tokens") + 1] == str(budget)


def test_bfcl_budget_cells_do_not_depend_on_the_new_scope(tmp_path):
    narrow = with_budgets(config(("bfcl_base",)))
    variants = [m for m in narrow["methods"] if m["arm"] == ARM and "history_budget_tokens" in m]
    assert [m["benchmarks"] for m in variants] == [["bfcl_base"]] * len(BUDGETS)
    narrow_plan, _ = runner.prepare(narrow, tmp_path / "narrow", tmp_path / "engine")
    wide_plan, _ = runner.prepare(with_budgets(config(("bfcl_base", *EXTRA))),
                                  tmp_path / "wide", tmp_path / "engine")
    wide = {row["cell_id"]: row for row in wide_plan}
    for row in narrow_plan:
        if row["cell_id"].startswith("bfcl_base__"):
            assert json.dumps(row, sort_keys=True).replace("narrow", "ROOT") == \
                json.dumps(wide[row["cell_id"]], sort_keys=True).replace("wide", "ROOT")


def test_other_native_arms_keep_the_bfcl_only_sweep(tmp_path):
    assert native_budget_benchmarks(ARM) == SUPPORTED_BENCHMARKS
    for arm in ("c2kv_c1_t02_r8", "c2kv_c1_off_r8", "c2kv_native_r8"):
        assert native_budget_benchmarks(arm) == BFCL_BENCHMARKS
    cfg = runner.with_native_history_budget(config(("bfcl_base",)), "c2kv_c1_t02_r8", 256)
    variant = cfg["methods"][-1]
    assert variant["arm"] == "c2kv_c1_t02_r8"
    assert set(variant["benchmarks"]) <= BFCL_BENCHMARKS
    variant["benchmarks"] = ["tau2"]
    with pytest.raises(ValueError, match="explicit BFCL scope"):
        runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")


@pytest.mark.parametrize("benchmark", sorted(SUPPORTED_BENCHMARKS))
def test_delivery_forwards_candidate_budget_on_every_benchmark(tmp_path, monkeypatch, benchmark):
    original = paper_c1.ARM
    try:
        paper_c1.select_arm(ARM)
        delivery = paper_c1.load_delivery()
        cfg = config(tuple(sorted(SUPPORTED_BENCHMARKS)))
        cfg.update(native_history_budget_tokens=256, sglang_source=str(tmp_path / "engine"))
        args = paper_c1.delivery_args(cfg, benchmark, tmp_path, [], delivery)
        assert args.history_budget_tokens == 256
        assert args.candidate_algorithm == "c1_v2_verified"
        seen = []
        fake = types.SimpleNamespace(resolve_override=lambda tokens, *rest: seen.append(tokens) or {})
        monkeypatch.setitem(sys.modules, "history_budget", fake)
        assert delivery._history_budget_override(args, {}) == {} and seen == [256]
        args.candidate_algorithm = None   # the C1 detector route keeps BFCL only
        if args.benchmark != "bfcl" and args.method != "c2kv_native":
            with pytest.raises(ValueError, match="BFCL only"):
                delivery._history_budget_override(args, {})
    finally:
        paper_c1.select_arm(original)


def ready_manifest(history=256 * 147456, workspace=256 * 147456, unit=147456):
    return {"runtime_policy_contract": {"effective_policy": {
        "kv_bytes_per_token": unit, "history_budget_bytes": history,
        "workspace_budget_bytes": workspace}}}


@pytest.mark.parametrize("manifest", [
    ready_manifest(history=768 * 147456), ready_manifest(workspace=768 * 147456),
    ready_manifest(unit=None), {"runtime_policy_contract": None}, {},
])
def test_budgeted_candidate_server_must_serve_its_derived_policy(manifest):
    native_extra.validate_native_budget_policy({}, manifest, ARM)   # unbudgeted: no check
    native_extra.validate_native_budget_policy({"native_history_budget_tokens": 256},
                                               ready_manifest(), ARM)
    with pytest.raises(RuntimeError, match="256-token history budget"):
        native_extra.validate_native_budget_policy({"native_history_budget_tokens": 256},
                                                   manifest, ARM)


def capacity_evidence(shard, benchmark):
    server = shard / "server"
    server.mkdir(parents=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready", "benchmark": benchmark,
        "allowed_task_ids": [shard.name]}))
    (server / "steps.jsonl").write_text(json.dumps({
        "schema": "a-acebench-event-step-v1" if benchmark == "acebench"
        else "a-event-native-exact-step-v1",
        "status": "failed", "session_id": f"{benchmark}/{shard.name}/attempt-0",
        "failure_kind": "method_failure", "failure_code": "c2kv_capacity_infeasible",
        "error": {"type": "CapacityInfeasible", "message": "B cannot fit one gist block"},
    }) + "\n")
    (server / "final.json").write_text(json.dumps({
        "status": "stopped", "journal_summary": {"completed": 3, "failed": 0, "pending": 0}}))


@pytest.mark.parametrize("benchmark,runtime,error", [
    ("tau2", "tau2", RuntimeError("tau2 task 5 ended with infrastructure_error")),
    ("acebench_agent", "acebench", subprocess.CalledProcessError(1, ["generate.py"])),
])
def test_capacity_failure_is_task_local_on_native_adapters(tmp_path, monkeypatch,
                                                           benchmark, runtime, error):
    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: object())
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_args: ["5", "6"])
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_args: (native, None, tmp_path / "controller.json"))

    def run_task(_config, _benchmark, task, _native, _delivery, _controller):
        calls.append(task)
        shard = native / "task_shards" / task
        if task == "6":
            shard.mkdir(parents=True)
            metrics = {"task_id": task, "official_score": 1.0, "normal_termination": True}
            return {"task_id": task, "status": "completed", "unified_metrics": metrics}, metrics
        capacity_evidence(shard, runtime)
        raise error

    monkeypatch.setattr(native_extra, "run_task", run_task)
    assert paper_c1.run_closed_loop({}, benchmark, cell) == native
    assert calls == ["5", "6"]
    failed = json.loads((native / "task_shards" / "5" / "paper_task_result.json").read_text())
    assert failed["status"] == "method_failure"
    assert failed["failure"]["kind"] == "capacity_infeasible"
    assert failed["unified_metrics"]["official_score"] == 0.0
    summary = json.loads((cell / f"summary_{paper_c1.ARM}.json").read_text())
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["method_failure_task_ids"] == ["5"]
