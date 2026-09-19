"""No-NPU checks for the shared-paper T0 reference-history launcher."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from generality import tool_context_cell as cell


def _args(tmp_path):
    return Namespace(
        python="python", benchmark="bfcl_base", arm="gen_h2o_k0",
        history_target_tokens=768,
        upstream="http://127.0.0.1:36203", user_upstream="",
        proxy_port=37490, out=tmp_path / "runs", model="gen-c1000",
        checkpoint=tmp_path / "C1000", tool_memory="t0:r8",
        tool_checkpoint=tmp_path / "T0", tool_budget_tokens=None,
        bench_python="",
    )


def test_reference_tool_cell_delegates_both_algorithms_to_paper(tmp_path):
    args = _args(tmp_path)
    argv = cell.command(args, ["--run-ids", "multi_turn_base_0"])
    assert argv[:3] == ["python", "-m", "benchmarks.run"]
    assert argv[argv.index("--benchmark") + 1] == "bfcl"
    assert argv[argv.index("--arm") + 1] == args.arm
    assert argv[argv.index("--history-kv-target-tokens") + 1] == "768"
    assert "--shared-engine" in argv
    assert argv[argv.index("--tool-memory") + 1] == "t0:r8"
    assert argv[argv.index("--tool-checkpoint") + 1] == str(args.tool_checkpoint.resolve())
    assert argv[argv.index("--out") + 1].endswith(
        "bfcl_base__gen_h2o_k0__history768__tools-t0_r8")
    assert argv[-4:] == ["--categories", "multi_turn_base",
                         "--run-ids", "multi_turn_base_0"]
    assert "sglang.launch_server" not in argv
    assert "--device" not in argv
    assert "--tool-budget-tokens" not in argv
    args.arm = "gen_snapkv_persistent_k0"
    args.tool_budget_tokens = 512
    snapkv = cell.command(args)
    assert snapkv[snapkv.index("--tool-budget-tokens") + 1] == "512"
    assert snapkv[snapkv.index("--out") + 1] != argv[argv.index("--out") + 1]
    args.tool_memory = "none"
    args.tool_checkpoint = None
    args.tool_budget_tokens = None
    raw = cell.command(args)
    assert "--tool-memory" not in raw
    assert "--tool-checkpoint" not in raw
    assert raw[raw.index("--history-kv-target-tokens") + 1] == "768"
    assert raw[raw.index("--out") + 1].endswith("__raw")


def test_reference_launcher_rejects_raw_tool_method_and_uses_unique_identity(tmp_path):
    args = _args(tmp_path)
    args.tool_memory = "h2o:r8"
    with pytest.raises(ValueError, match="T0 tool memory only"):
        cell.command(args)
    assert cell.cell_id("appworld", "gen_h2o_k0", 768, "t0:r8") != (
        cell.cell_id("appworld", "gen_h2o_k0", 768, "t0:r12"))
    assert cell.cell_id("bfcl_base", "gen_h2o_k0", 768, "none") != (
        cell.cell_id("bfcl_long_context", "gen_h2o_k0", 768, "none"))


def test_live_readiness_checks_actual_tool_projection_without_generation(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.tool_checkpoint.mkdir()
    (args.tool_checkpoint / "config.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(b"{}").hexdigest()
    info = {"c2kv_native_packed": {
        "enabled": True,
        "model_binding": {"model_path": str(args.checkpoint)},
        "tool_gist": {"enabled": True, "source": str(args.tool_checkpoint),
                      "config_sha256": digest, "extract_projection_set": "tool"},
    }}
    requested = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(info).encode()

    class Opener:
        def open(self, request, timeout):
            requested.append(request.full_url)
            return Response()

    monkeypatch.setattr(cell, "build_opener", lambda *args: Opener())
    receipt = cell.validate_live_engine(args.upstream, args.checkpoint,
                                        args.tool_checkpoint)
    assert receipt["tool_config_sha256"] == digest
    assert requested == [args.upstream + "/model_info"]
    info["c2kv_native_packed"]["tool_gist"]["config_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="T0 tool checkpoint binding differs"):
        cell.validate_live_engine(args.upstream, args.checkpoint, args.tool_checkpoint)


def test_dry_run_is_local_and_uses_shared_checkout(tmp_path, monkeypatch, capsys):
    for relative in ("benchmarks/run.py", "benchmarks/proxy.py",
                     "benchmarks/toolmemory.py", "benchmarks/arms.py"):
        path = tmp_path / "paper" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    monkeypatch.setattr(cell, "validate_live_engine",
                        lambda *args: pytest.fail("dry run contacted the engine"))
    args = _args(tmp_path)
    cell.main(["--paper-root", str(tmp_path / "paper"),
               "--benchmark", args.benchmark, "--arm", args.arm,
               "--history-target-tokens", str(args.history_target_tokens),
               "--upstream", args.upstream, "--proxy-port", str(args.proxy_port),
               "--out", str(args.out), "--model", args.model,
               "--checkpoint", str(args.checkpoint),
               "--tool-memory", args.tool_memory,
               "--tool-checkpoint", str(args.tool_checkpoint),
               "--dry-run", "--", "--run-ids", "multi_turn_base_0"])
    observed = json.loads(capsys.readouterr().out)
    assert observed["engine_preflight"] == "skipped (dry-run)"
    assert observed["command"][-4:] == ["--categories", "multi_turn_base",
                                        "--run-ids", "multi_turn_base_0"]


@pytest.mark.parametrize("benchmark,adapter,fixed", [
    ("bfcl_long_context", "bfcl", ["--categories", "multi_turn_long_context"]),
    ("acebench_agent", "acebench", ["--acebench-category", "agent"]),
    ("appworld", "acon_appworld", []),
    ("toolsandbox", "toolsandbox", []),
])
def test_all_official_npu_benchmarks_keep_distinct_logical_ids(
    tmp_path, benchmark, adapter, fixed,
):
    args = _args(tmp_path)
    args.benchmark = benchmark
    result = cell.command(args)
    assert result[result.index("--benchmark") + 1] == adapter
    if fixed:
        assert result[-len(fixed):] == fixed
    assert result[result.index("--out") + 1].endswith(
        f"{benchmark}__gen_h2o_k0__history768__tools-t0_r8")
