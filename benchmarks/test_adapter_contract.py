"""The adapter contract (adapters/base.py) and the PINNED external commands.

Two things this file protects:

1. every adapter module really implements NAME / add_arguments / run(ctx),
   and ``v1`` / ``RunContext.opt`` behave as the adapters assume;
2. **the argv handed to each external harness is byte-identical to what the
   pre-registry code sent.**  Those command lines are the experiment: a
   changed flag silently redefines every number produced with it, so each is
   asserted against a literal list rather than against the code that built
   it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402
from adapters import (  # noqa: E402
    acebench_adapter, acon_adapter, bfcl_adapter, base, tau2_adapter,
    toolsandbox_adapter,
)
from adapters.base import RunContext, v1  # noqa: E402

MODULES = [tau2_adapter, bfcl_adapter, toolsandbox_adapter, acon_adapter,
           acebench_adapter]


# ---- contract ---------------------------------------------------------------

def test_every_adapter_module_implements_the_contract():
    for module in MODULES:
        assert isinstance(module.NAME, str) and module.NAME
        assert callable(module.add_arguments)
        assert callable(module.run)
        names = getattr(module, "NAMES", (module.NAME,))
        assert all(name in run.ADAPTERS for name in names), module.NAME


def test_v1_appends_exactly_one_suffix():
    assert v1("http://127.0.0.1:34100") == "http://127.0.0.1:34100/v1"
    assert v1("http://127.0.0.1:34100/") == "http://127.0.0.1:34100/v1"
    assert v1("http://h:1///") == "http://h:1/v1"


def _ctx(**options):
    return RunContext(base_url="http://p", user_base_url="http://u",
                      out_dir=Path("out"), model="m", arm="c2kv",
                      options=options)


def test_run_context_defaults_and_opt():
    ctx = _ctx(split="", task_ids=[], max_iter=None, num_workers=0, full=False,
               tag="mytag")
    assert ctx.run_name == "c2kv_run" and ctx.request_log is None
    # argparse's empty forms fall back to the adapter default...
    assert ctx.opt("split", "test") == "test"
    assert ctx.opt("task_ids") is None
    assert ctx.opt("max_iter", 30) == 30
    assert ctx.opt("missing", "d") == "d"
    # ...but 0 and False are values, not emptiness
    assert ctx.opt("num_workers", 4) == 0
    assert ctx.opt("full", True) is False
    assert ctx.opt("tag", "fallback") == "mytag"


def test_add_arguments_registers_only_that_adapters_flags():
    import argparse

    owned = {
        tau2_adapter: {"--task-set"},
        bfcl_adapter: {"--categories"},
        toolsandbox_adapter: {"--full", "--ts-scenarios", "--ts-agent", "--ts-user"},
        acon_adapter: {"--acon-dir", "--split", "--tag", "--task-ids"},
        acebench_adapter: {"--acebench-dir", "--acebench-category",
                           "--acebench-language", "--user-model"},
    }
    for module, flags in owned.items():
        parser = argparse.ArgumentParser()
        module.add_arguments(parser)
        registered = {a.option_strings[0] for a in parser._actions if a.option_strings}
        assert registered - {"-h"} == flags, module.NAME


# ---- tau2: `tau2.cli run` / `evaluate-trajs` --------------------------------

def test_tau2_run_command_is_byte_identical():
    cmd = tau2_adapter.run_command(
        "http://127.0.0.1:34100", "http://127.0.0.1:35000", "airline",
        "c2kv-agent", 4, "c2kv_run_ab12", python="/py")
    assert cmd == [
        "/py", "-m", "tau2.cli", "run",
        "--domain", "airline",
        "--task-set-name", "airline",
        "--agent-llm", "openai/c2kv-agent",
        "--agent-llm-args",
        '{"api_base": "http://127.0.0.1:34100/v1", "api_key": "EMPTY", "temperature": 0.0}',
        "--user-llm", "openai/c2kv-agent",
        "--user-llm-args",
        '{"api_base": "http://127.0.0.1:35000/v1", "api_key": "EMPTY", "temperature": 0.0}',
        "--max-concurrency", "4",
        "--save-to", "c2kv_run_ab12",
        "--auto-resume",
    ]


def test_tau2_run_command_domain_and_num_tasks():
    cmd = tau2_adapter.run_command("http://p/", "http://u/", "telecom_small",
                                   "m", 2, "r", max_tasks=5, python="/py")
    # --domain is the task set's first underscore-separated token
    assert cmd[cmd.index("--domain") + 1] == "telecom"
    assert cmd[cmd.index("--task-set-name") + 1] == "telecom_small"
    assert cmd[-2:] == ["--num-tasks", "5"]
    assert cmd[-3] == "--auto-resume"


def test_tau2_evaluate_command_is_byte_identical(tmp_path):
    sims = tmp_path / "sims"
    cmd = tau2_adapter.evaluate_command(sims, python="/py")
    assert cmd == ["/py", "-m", "tau2.cli", "evaluate-trajs", "-o", str(sims),
                   str(sims / "results.json")]


def test_tau2_default_python_is_this_interpreter():
    assert tau2_adapter.run_command("http://p", "http://u", "airline", "m", 1,
                                    "r")[0] == sys.executable
    assert tau2_adapter.evaluate_command(Path("s"))[0] == sys.executable


# ---- bfcl: `bfcl generate` / `bfcl evaluate` --------------------------------

def test_bfcl_argv_is_byte_identical():
    assert bfcl_adapter.generate_argv("c2kv-full", "multi_turn_base") == [
        "generate", "--model", "c2kv-full", "--test-category", "multi_turn_base"]
    assert bfcl_adapter.evaluate_argv("c2kv-full", "multi_turn_base") == [
        "evaluate", "--model", "c2kv-full", "--test-category", "multi_turn_base"]


def test_bfcl_subset_argv_uses_run_ids_then_partial_eval():
    ids = ["multi_turn_base_0", "multi_turn_base_3"]
    # generate takes the BOOLEAN --run-ids (the ids go in
    # test_case_ids_to_generate.json); evaluate takes --partial-eval instead
    assert bfcl_adapter.generate_argv("c2kv-full", "memory", ids) == [
        "generate", "--model", "c2kv-full", "--test-category", "memory",
        "--run-ids"]
    assert bfcl_adapter.evaluate_argv("c2kv-full", "memory", ids) == [
        "evaluate", "--model", "c2kv-full", "--test-category", "memory",
        "--partial-eval"]


def test_bfcl_handler_key_dashes_the_arm():
    assert bfcl_adapter.handler_key("c2kv") == "c2kv-c2kv"
    assert bfcl_adapter.handler_key("c2kv_repair_tail") == "c2kv-c2kv-repair-tail"
    assert bfcl_adapter.handler_key("") == "c2kv-full"


# ---- toolsandbox: `tool_sandbox` -------------------------------------------

def test_toolsandbox_command_is_byte_identical(tmp_path):
    assert toolsandbox_adapter.cli_command(tmp_path) == [
        "tool_sandbox", "--user", "GPT_4_o_2024_05_13",
        "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path), "-t"]
    assert toolsandbox_adapter.cli_command(tmp_path, test_mode=False) == [
        "tool_sandbox", "--user", "GPT_4_o_2024_05_13",
        "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path)]


def test_toolsandbox_subset_command_overrides_test_mode(tmp_path):
    cmd = toolsandbox_adapter.cli_command(tmp_path, scenarios=["a", "b"],
                                          parallel="4")
    assert cmd == ["tool_sandbox", "--user", "GPT_4_o_2024_05_13",
                   "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path),
                   "-s", "a", "b", "-p", "4"]
    assert "-t" not in cmd  # a subset run is never also test mode


def test_toolsandbox_env_splits_agent_and_user():
    env = toolsandbox_adapter.harness_env("http://127.0.0.1:34100",
                                          "http://127.0.0.1:35000")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:34100/v1"
    assert env["TOOLSANDBOX_USER_BASE_URL"] == "http://127.0.0.1:35000/v1"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


# ---- acon: `run.py` / `run_all.py` / `appworld evaluate` --------------------

def test_acon_qa_command_is_byte_identical():
    assert acon_adapter.qa_command("/py", "c2kv-agent", "r_ab12", "test", 30) == [
        "/py", "run.py", "--split", "test", "--model_name", "c2kv-agent",
        "--tag", "r_ab12", "--max_iter", "30",
        "--data_folder", "data/nq_multi_8"]


def test_acon_appworld_command_is_byte_identical():
    assert acon_adapter.appworld_command("/py", "c2kv-agent", "r_ab12",
                                         "test_normal", 50) == [
        "/py", "run_all.py", "--split", "test_normal",
        "--model_name", "c2kv-agent", "--tag", "r_ab12",
        "--max_iter", "50", "--seed", "42"]


def test_acon_appworld_evaluate_command_is_byte_identical():
    assert acon_adapter.appworld_evaluate_command(
        "/venv/bin/appworld", "org/model", "r_ab12", "test_normal") == [
        "/venv/bin/appworld", "evaluate", "org_model_r_ab12", "test_normal"]


# ---- acebench: `generate.py` / `eval_main.py` ------------------------------

def test_acebench_commands_are_byte_identical(tmp_path):
    assert acebench_adapter.generate_command(
        "/py", tmp_path, "c2kv-agent", "agent", "en", 4, 40, "c2kv-agent",
        0.0, 1.0, 1200) == [
        "/py", str(tmp_path / "generate.py"),
        "--model", "c2kv-agent", "--category", "agent", "--language", "en",
        "--num-threads", "4", "--max-dialog-turns", "40",
        "--user-model", "c2kv-agent",
        "--temperature", "0.0", "--top-p", "1.0", "--max-tokens", "1200"]
    assert acebench_adapter.eval_command(
        "/py", tmp_path, "c2kv-agent", "agent", "en") == [
        "/py", str(tmp_path / "eval_main.py"), "--model", "c2kv-agent",
        "--category", "agent", "--language", "en"]


# ---- cost-join declarations -------------------------------------------------

@pytest.mark.parametrize("module", [tau2_adapter, bfcl_adapter,
                                    toolsandbox_adapter, acebench_adapter])
def test_unjoinable_adapters_declare_a_reason(module):
    assert module.COST_JOIN.startswith("not joinable: ")
    assert len(module.COST_JOIN) > len("not joinable: ")


def test_base_module_is_importable_as_a_package_member():
    assert base.RunContext is RunContext


def test_adapter_modules_run_standalone(tmp_path):
    """Each adapter documents a ``python benchmarks/adapters/<x>.py`` recipe;
    a RELATIVE ``from .base import ...`` would break every one of them."""
    import subprocess

    root = Path(__file__).resolve().parent
    for module in MODULES:
        path = root / "adapters" / f"{Path(module.__file__).name}"
        done = subprocess.run([sys.executable, str(path), "--help"],
                              capture_output=True, text=True, cwd=root.parent)
        assert done.returncode == 0, (path.name, done.stderr[-400:])
