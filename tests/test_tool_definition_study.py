from pathlib import Path

import pytest

from generality.tool_definition_study import command


def test_npu_uses_shared_evaluator_and_owns_device(tmp_path):
    entry = tmp_path / "benchmarks/tool_definition/cli.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")
    argv = command(tmp_path, "/npu/python", "evaluate", ["--manifest", "/data/manifest.json"])
    assert argv == ["/npu/python", "-m", "benchmarks.tool_definition.cli", "evaluate",
                    "--manifest", "/data/manifest.json", "--device", "npu:0"]
    assert "--device" not in command(tmp_path, "/npu/python", "prepare", [])
    with pytest.raises(ValueError, match="owns"):
        command(tmp_path, "/npu/python", "evaluate", ["--device=cuda"])
