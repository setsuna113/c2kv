from pathlib import Path

import pytest

from generality.tool_definition_study import command


def test_npu_uses_shared_sglang_evaluator(tmp_path):
    entry = tmp_path / "benchmarks/tool_definition/cli.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")
    argv = command(tmp_path, "/npu/python", "evaluate", ["--manifest", "/data/manifest.json",
                                                        "--upstream", "http://localhost:30000"])
    assert argv == ["/npu/python", "-m", "benchmarks.tool_definition.cli", "evaluate",
                    "--manifest", "/data/manifest.json", "--upstream", "http://localhost:30000"]
    resumed = command(tmp_path, "/npu/python", "evaluate", [
        "--manifest", "/data/manifest.json", "--upstream", "http://localhost:30000",
        "--resume"])
    assert resumed == [*argv, "--resume"]
    assert "--device" not in command(tmp_path, "/npu/python", "prepare", [])
    with pytest.raises(ValueError, match="owns"):
        command(tmp_path, "/npu/python", "evaluate", ["--device=cuda"])
    with pytest.raises(ValueError, match="requires --upstream"):
        command(tmp_path, "/npu/python", "evaluate", ["--manifest", "/data/manifest.json"])


def test_npu_forwards_adaptive_selection_to_shared_preparation(tmp_path):
    entry = tmp_path / "benchmarks/tool_definition/cli.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")
    flags = ["--selector-policy", "last_user_adaptive_v1", "--k", "3"]
    assert command(tmp_path, "/npu/python", "prepare", flags)[-len(flags):] == flags
