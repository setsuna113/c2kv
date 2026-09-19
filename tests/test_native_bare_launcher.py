from argparse import Namespace
from pathlib import Path
import json

from generality import native_bare

from generality.native_bare import command


def test_native_client_targets_existing_engine_without_cuda_launcher(tmp_path):
    marker = tmp_path / "experiments/history_system/native_bare.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    args = Namespace(paper_root=tmp_path, python="python", config=tmp_path / "pod.json",
                     benchmark="bfcl_base", upstream="http://127.0.0.1:36203",
                     proxy_port=37490, out=tmp_path / "new-native-bare", stage="closed_loop",
                     task_ids="multi_turn_base_0", prefixes=None)
    argv = command(args)
    assert argv[argv.index("--arm") + 1] == "c2kv_native_r4"
    assert argv[argv.index("--upstream") + 1] == args.upstream
    assert "benchmarks.paper.c1" in argv
    assert "sglang.launch_server" not in argv
    assert "--device" not in argv
    assert "--tool-memory" not in argv


def test_native_client_tool_on_uses_shared_paper_source(tmp_path):
    marker = tmp_path / "experiments/history_system/native_bare.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    checkpoint = tmp_path / "T0" / "checkpoint-500"
    args = Namespace(paper_root=tmp_path, python="python", config=tmp_path / "pod.json",
                     benchmark="appworld", upstream="http://127.0.0.1:36203",
                     proxy_port=37490, out=tmp_path / "new-tool-on", stage="closed_loop",
                     task_ids="3d9a636_1", prefixes=None, arm="c2kv_c1_t02_r8",
                     tool_memory="t0:r8", tool_checkpoint=checkpoint,
                     tool_budget_tokens=512)
    argv = command(args)
    assert argv[argv.index("--arm") + 1] == "c2kv_c1_t02_r8"
    assert argv[argv.index("--tool-memory") + 1] == "t0:r8"
    assert argv[argv.index("--tool-checkpoint") + 1] == str(checkpoint.resolve())
    assert argv[argv.index("--tool-budget-tokens") + 1] == "512"
    assert argv[:3] == ["python", "-m", "benchmarks.paper.c1"]
    assert "sglang.launch_server" not in argv


def test_native_tool_dry_run_does_not_start_a_process(tmp_path, monkeypatch, capsys):
    marker = tmp_path / "experiments/history_system/native_bare.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    monkeypatch.setattr(native_bare.subprocess, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("dry-run started a process")))
    monkeypatch.setattr("sys.argv", ["native_bare.py", "--paper-root", str(tmp_path),
                                  "--config", str(tmp_path / "pod.json"),
                                  "--benchmark", "bfcl_base", "--upstream",
                                  "http://127.0.0.1:36203", "--proxy-port", "37490",
                                  "--out", str(tmp_path / "out"),
                                  "--tool-memory", "t0:r8", "--tool-checkpoint",
                                  str(tmp_path / "T0"), "--dry-run"])
    native_bare.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["engine_preflight"] == "skipped (dry-run)"
    assert receipt["command"][receipt["command"].index("--tool-memory") + 1] == "t0:r8"
