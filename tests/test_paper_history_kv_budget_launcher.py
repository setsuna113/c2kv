"""The NPU entry point delegates all budget and adapter rules to paper code."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from generality import paper_history_kv_budget as launcher


def _paper_root(root: Path) -> Path:
    for name in ("benchmarks/paper/history_kv_client.py",
                 "benchmarks/paper/runner.py", "benchmarks/history_budget.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_wrapper_forwards_unmodified_cell_options_and_source_root(monkeypatch, tmp_path):
    root = _paper_root(tmp_path / "paper")
    seen = {}
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="nt", environ={}, pathsep=";"))

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    launcher.main(["--paper-root", str(root), "--python", "npu-python",
                   "--config", str(tmp_path / "npu-config.json"),
                   "--benchmark", "bfcl_base", "--history-kv-budget", "commitkv=768",
                   "--upstream", "http://127.0.0.1:36200", "--proxy-port", "37490",
                   "--out", str(tmp_path / "results"), "--dry-run"])
    assert seen["command"][:3] == ["npu-python", "-m", "benchmarks.paper.history_kv_client"]
    assert seen["command"][seen["command"].index("--history-kv-budget") + 1] == "commitkv=768"
    assert seen["command"][-1] == "--dry-run"
    assert seen["cwd"] == root.resolve()
    assert seen["env"]["PYTHONPATH"].split(launcher.os.pathsep)[0] == str(root.resolve())
    assert seen["check"] is True


def test_missing_shared_client_rejected_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(SystemExit):
        launcher.main(["--paper-root", str(tmp_path / "missing"),
                       "--config", "config.json", "--benchmark", "bfcl_base",
                       "--history-kv-budget", "agentkv=768", "--upstream", "http://localhost:36200",
                       "--out", str(tmp_path / "results")])


def test_posix_hands_process_ownership_to_shared_client(monkeypatch, tmp_path):
    root = _paper_root(tmp_path / "paper")
    seen = {}
    monkeypatch.setattr(launcher, "os", SimpleNamespace(
        name="posix", environ={}, pathsep=":",
        chdir=lambda path: seen.update(cwd=path),
        execvpe=lambda python, argv, env: seen.update(python=python, argv=argv, env=env)))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("must exec"))
    launcher.main(["--paper-root", str(root), "--python", "python3", "--help"])
    assert seen["cwd"] == root.resolve()
    assert seen["python"] == "python3"
    assert seen["argv"] == ["python3", "-m", "benchmarks.paper.history_kv_client", "--help"]
