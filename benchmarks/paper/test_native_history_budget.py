"""Native sweeps preserve the historical release and route each explicit cap."""
import copy
import json
from unittest import mock

import pytest

from benchmarks.native_history_budget import NativeHistoryBudget, parse_native_history_budget
from benchmarks.paper import c1, runner
from benchmarks.paper.candidate_matrix import with_candidate_methods
from benchmarks.paper.report import write_comparison

ARM = "c2kv_goal_pending_r8"


def config():
    base = dict(json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8")), history_kv_budget_tokens=768)
    return with_candidate_methods(base, ("goal_pending",))


def test_eight_point_sweep_preserves_original_cells_and_commands(tmp_path):
    original = config()
    saved = copy.deepcopy(original)
    output = tmp_path / "results"
    old, profile = runner.prepare(original, output, tmp_path / "engine")
    expanded = original
    budgets = (256, 384, 512, 640, 768, 1024, 1536, 2048)
    for budget in budgets:
        expanded = runner.with_native_history_budget(expanded, ARM, budget)
    plan, _ = runner.prepare(expanded, output, tmp_path / "engine")
    assert original == saved
    by_id = {row["cell_id"]: row for row in plan}
    assert all(by_id[row["cell_id"]] == row for row in old)
    assert len(plan) == len(old) + len(budgets)
    for budget in budgets:
        cell = by_id[f"bfcl_base__{ARM}_b{budget}"]
        assert cell["arm"] == ARM and cell["ratio"] == 8
        assert cell["history_budget_tokens"] == budget
        for stage in ("closed_loop", "common_prefix"):
            argv = runner.run_command(expanded, cell, output / stage / cell["cell_id"], profile, stage)
            assert argv[argv.index("--history-budget-tokens") + 1] == str(budget)
            assert argv[argv.index("--arm") + 1] == ARM
    assert "history_budget_tokens" in (output / "matrix.csv").read_text()


def test_cli_repeat_and_selected_cell(tmp_path):
    args = ["--output", str(tmp_path / "results"), "--sglang-source", str(tmp_path / "engine"),
            "--candidate-arms", "goal_pending", "--native-history-budget", ARM + "=256",
            "--native-history-budget", ARM + "=2048"]
    runner.main(["prepare", "--history-kv-budget-tokens", "768", *args])
    selected = f"bfcl_base__{ARM}_b2048"
    with mock.patch.object(runner, "execute") as execute:
        runner.main(["run", "--history-kv-budget-tokens", "768", *args, "--stage", "closed_loop", "--cells", selected])
    assert execute.call_args.args[4:6] == (["closed_loop"], {selected})


@pytest.mark.parametrize("budget", [0, -1, True, None, "768", 1.5])
def test_invalid_cap_rejected_before_output(tmp_path, budget):
    cfg = config()
    cfg["methods"][-1]["history_budget_tokens"] = budget
    with pytest.raises(ValueError, match="positive integer"):
        runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")
    assert not (tmp_path / "results" / "config.resolved.json").exists()


def test_no_silent_budget_on_unsupported_arm_or_benchmark(tmp_path):
    for arm in ("full", "commitkv", "hiagent_full"):
        with pytest.raises(ValueError, match="does not support"):
            runner.with_native_history_budget(config(), arm, 768)
    cfg = runner.with_native_history_budget(config(), ARM, 768)
    with pytest.raises(ValueError, match="already exists"):
        runner.with_native_history_budget(cfg, ARM, 768)
    cfg["methods"][-1]["benchmarks"] = ["acebench_agent"]
    with pytest.raises(ValueError, match="explicit BFCL scope"):
        runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")


def test_delivery_forwards_budget_without_changing_algorithm(tmp_path):
    original = c1.ARM
    try:
        c1.select_arm(ARM)
        delivery = c1.load_delivery()
        cfg = config()
        cfg["sglang_source"] = str(tmp_path / "engine")
        legacy = c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
        assert legacy.history_budget_tokens is None
        cfg["native_history_budget_tokens"] = 2048
        args = c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
        assert args.history_budget_tokens == 2048
        assert args.ratio == legacy.ratio == 8
        assert args.candidate_algorithm == legacy.candidate_algorithm == "goal_pending"
        assert args.selector_threshold == legacy.selector_threshold
        with pytest.raises(ValueError, match="BFCL only"):
            c1.delivery_args(cfg, "acebench_agent", tmp_path, [], delivery)
    finally:
        c1.select_arm(original)


def test_changed_budget_cannot_resume_an_existing_cell(tmp_path):
    output = tmp_path / "results"
    cfg = runner.with_native_history_budget(config(), ARM, 768)
    runner.prepare(cfg, output, tmp_path / "engine")
    changed = runner.with_native_history_budget(config(), ARM, 2048)
    with pytest.raises(RuntimeError, match="cells removed"):
        runner.prepare(changed, output, tmp_path / "engine")


def test_comparison_separates_native_budgets(tmp_path):
    cfg = runner.with_native_history_budget(config(), ARM, 768)
    plan = [row for row in runner.cells(cfg) if row["arm"] == ARM]
    for cell in plan:
        directory = tmp_path / "closed_loop" / cell["cell_id"]
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text("{}")
        (directory / "measurement_summary.json").write_text("{}")
        (directory / f"summary_{ARM}.json").write_text('{"n": 200}')
    rows = write_comparison(tmp_path, plan)
    assert [r["history_budget_tokens"] for r in rows] == [None, 768]
    assert len({r["cell_id"] for r in rows}) == 2


def test_direct_delivery_cannot_reuse_unbudgeted_task_results(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "profile.json").write_text('{}')
    delivery = mock.Mock()
    delivery.build_profile.return_value = ({}, {"native_history_budget": {"target_tokens": 2048}})
    cfg = config()
    with mock.patch.object(c1, "delivery_args", return_value=object()):
        with pytest.raises(ValueError, match="separate cell output"):
            c1.prepare_native(cfg, "bfcl_base", tmp_path, [], delivery)
    delivery.preflight_sglang_backend.assert_not_called()
    assert (native / "profile.json").read_text() == '{}'


@pytest.mark.parametrize("value", ["", "goal_pending", ARM+"=0", ARM+"=-1", ARM+"=1.5"])
def test_malformed_budget_cli(value):
    with pytest.raises(ValueError):
        parse_native_history_budget(value)


def test_native_budget_parser():
    assert parse_native_history_budget(ARM + "=2048") == (ARM, 2048)
    assert NativeHistoryBudget(2048).cli_args() == ["--history-budget-tokens", "2048"]


@pytest.mark.parametrize("benchmark", ["bfcl_base", "bfcl_long_context", "appworld",
                                       "acebench_agent", "toolsandbox", "tau2"])
def test_bare_budget_reaches_all_native_adapters_without_s0(tmp_path, benchmark):
    original = c1.ARM
    try:
        c1.select_arm("c2kv_native_r8")
        cfg = config()
        cfg.update(native_arm=c1.ARM, native_history_budget_tokens=192,
                   sglang_source=str(tmp_path / "engine"))
        args = c1.delivery_args(cfg, benchmark, tmp_path, [], c1.load_delivery())
        assert args.history_budget_tokens == 192
        assert args.method == "c2kv_native" and args.ratio == 8
        assert args.racer_backend_config is None
    finally:
        c1.select_arm(original)


def test_explicit_native_generation_timeout_reaches_engine_command(tmp_path):
    original = c1.ARM
    try:
        c1.select_arm("c2kv_native_r8")
        cfg = config()
        cfg.update(native_arm=c1.ARM, generation_timeout=1234.5,
                   sglang_source=str(tmp_path / "engine"), bfcl_dir=str(tmp_path / "bfcl"))
        (tmp_path / "bfcl" / "bfcl_eval").mkdir(parents=True)
        delivery = c1.load_delivery()
        args = c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
        command, _ = delivery.commands_for_task(args, "multi_turn_base_0", tmp_path / "controller.json")
        assert command[command.index("--sglang-timeout-seconds") + 1] == "1234.5"
    finally:
        c1.select_arm(original)
