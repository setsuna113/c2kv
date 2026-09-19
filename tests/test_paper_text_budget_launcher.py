"""The NPU launcher delegates the budget algorithm to the shared paper tree."""
from argparse import Namespace
import io
import json
from pathlib import Path
import urllib.error

import pytest

from generality import paper_text_budget as launcher
from controller_runtime.benchmarks.bfcl_completion import completion_kind


def _paper_root(root: Path) -> Path:
    for name in (
        "benchmarks/run.py", "benchmarks/arms.py", "benchmarks/proxy.py",
        "benchmarks/acon_budget.py", "benchmarks/hiagent_budget.py",
        "benchmarks/adapters/acebench_adapter.py", "benchmarks/acebench_cli.py",
        "benchmarks/paper/budget_server.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _args(tmp_path, benchmark="bfcl_base"):
    return Namespace(
        python="python", benchmark=benchmark, history_budget_tokens=768,
        upstream="http://127.0.0.1:36200", proxy_port=37400,
        out=tmp_path / "result", model="gen-c1000",
        checkpoint=tmp_path / "checkpoint", checkpoint_profile=None,
        run_ids=None,
    )


@pytest.mark.parametrize("benchmark,category", [
    ("bfcl_base", "multi_turn_base"),
    ("bfcl_long_context", "multi_turn_long_context"),
])
def test_command_selects_explicit_bfcl_budget_cell(tmp_path, benchmark, category):
    cmd = launcher.command(_args(tmp_path, benchmark))
    assert cmd[:3] == ["python", "-m", "benchmarks.run"]
    assert cmd[cmd.index("--arm") + 1] == "hiagent_full_b768"
    assert cmd[cmd.index("--categories") + 1] == category
    assert cmd[cmd.index("--upstream") + 1] == "http://127.0.0.1:36200"
    assert cmd[cmd.index("--capability-features") + 1] == "hiagent_trajectory_retrieval_v1"
    assert "sglang.launch_server" not in cmd
    assert "--device" not in cmd
    assert "--shared-engine" not in cmd


@pytest.mark.parametrize("method,arm,features", [
    ("acon", "acon_hist_ut_co_b768", "acebench_role_history_v1"),
    ("hiagent", "hiagent_full_b768", "hiagent_trajectory_retrieval_v1,acebench_role_history_v1"),
])
def test_ace_agent_budget_command_preserves_python_action_route(tmp_path, method, arm, features):
    args = _args(tmp_path, "acebench_agent")
    args.method = method
    args.acebench_dir = tmp_path / "acebench"
    args.acebench_language = "en"
    args.acebench_task_ids = "agent_multi_turn_1"
    args.bench_python = "ace-python"
    args.user_upstream = "http://127.0.0.1:36201"
    args.max_tasks = None
    cmd = launcher.command(args)
    assert cmd[cmd.index("--benchmark") + 1] == "acebench"
    assert cmd[cmd.index("--arm") + 1] == arm
    assert cmd[cmd.index("--acebench-category") + 1] == "agent"
    assert cmd[cmd.index("--acebench-task-ids") + 1] == "agent_multi_turn_1"
    assert cmd[cmd.index("--bench-python") + 1] == "ace-python"
    assert cmd[cmd.index("--user-upstream") + 1] == "http://127.0.0.1:36201"
    assert cmd[cmd.index("--capability-features") + 1] == features
    assert "--categories" not in cmd and "--run-ids" not in cmd
    assert "--tool-memory" not in cmd


def test_positive_budget_preflight_requires_server_tokenized_nonempty_history(monkeypatch):
    seen = {}

    class Opener:
        def open(self, request, timeout):
            seen["url"] = request.full_url
            seen["payload"] = json.loads(request.data)
            return io.BytesIO(json.dumps({
                "success": True, "server_tokenized": True,
                "history_tokens": 5, "prompt_tokens": 12,
                "history_start": 2, "history_end": 7,
            }).encode())

    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *args: Opener())
    receipt = launcher.validate_live_budget_server("http://127.0.0.1:36200", "gen-c1000")
    assert receipt["history_tokens"] == 5
    assert seen["url"].endswith("/v1/c2kv/chat_budget")
    assert seen["payload"]["c2kv_kv_memory_hint"]["paper_measurement"] == {
        "history_start_message_count": 0, "history_message_count": 2,
    }


def test_missing_or_legacy_budget_endpoint_refuses_run(monkeypatch, tmp_path):
    root = _paper_root(tmp_path / "paper")

    class MissingOpener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *args: MissingOpener())
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="chat_budget"):
        launcher.main([
            "--paper-root", str(root), "--benchmark", "bfcl_base",
            "--history-budget-tokens", "768", "--upstream", "http://127.0.0.1:36200",
            "--proxy-port", "37400", "--out", str(tmp_path / "result"),
            "--model", "gen-c1000", "--checkpoint", str(tmp_path / "checkpoint"),
            "--bfcl-dir", str(tmp_path / "bfcl"),
        ])


def test_dry_run_does_not_contact_server_or_launch_client(monkeypatch, tmp_path, capsys):
    root = _paper_root(tmp_path / "paper")
    monkeypatch.setattr(launcher, "validate_live_budget_server",
                        lambda *args: pytest.fail("dry run must not contact the engine"))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    launcher.main([
        "--paper-root", str(root), "--benchmark", "bfcl_long_context",
        "--history-budget-tokens", "768", "--upstream", "http://127.0.0.1:36200",
        "--proxy-port", "37400", "--out", str(tmp_path / "result"),
        "--model", "gen-c1000", "--checkpoint", str(tmp_path / "checkpoint"),
        "--bfcl-dir", str(tmp_path / "bfcl"), "--dry-run",
    ])
    plan = json.loads(capsys.readouterr().out)
    assert plan["cell_id"] == "bfcl_long_context__hiagent_full_b768"
    assert plan["live_budget_preflight"] == "skipped (dry-run)"


def test_ace_acon_dry_run_builds_shared_budget_entry_without_launch(monkeypatch, tmp_path, capsys):
    root = _paper_root(tmp_path / "paper")
    monkeypatch.setattr(launcher, "validate_live_budget_server",
                        lambda *args: pytest.fail("dry run must not contact the engine"))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    launcher.main([
        "--paper-root", str(root), "--benchmark", "acebench_agent", "--method", "acon",
        "--history-budget-tokens", "768", "--upstream", "http://127.0.0.1:36200",
        "--proxy-port", "37400", "--out", str(tmp_path / "result"),
        "--model", "gen-c1000", "--checkpoint", str(tmp_path / "checkpoint"),
        "--acebench-dir", str(tmp_path / "acebench"), "--dry-run",
    ])
    plan = json.loads(capsys.readouterr().out)
    assert plan["cell_id"] == "acebench_agent__acon_hist_ut_co_b768"
    assert plan["command"][plan["command"].index("--benchmark") + 1] == "acebench"
    assert plan["live_budget_preflight"] == "skipped (dry-run)"


def test_missing_shared_source_and_invalid_budget_fail_before_execution(tmp_path):
    with pytest.raises(ValueError, match="shared budget sources"):
        launcher.validate_paper_root(tmp_path)
    for budget in (0, -1, True, 768.0, "768", None):
        with pytest.raises(ValueError, match="positive integer"):
            launcher.arm_name(budget)


def test_bundled_completion_classifies_budget_rejection_as_terminal():
    row = {"result": [], "traceback": json.dumps({"error": {
        "code": "hiagent_history_budget_exceeded",
    }})}
    assert completion_kind(row) == "hiagent_history_budget_exceeded"
