import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from benchmarks.paper.artifact_io import atomic_json
from benchmarks.paper.runner import DEFAULT_CONFIG, prepare, resolve_history_kv_budgets


def test_failed_publication_keeps_previous_complete_artifact(tmp_path):
    target = tmp_path / "config.json"
    atomic_json(target, {"generation": 1})
    with mock.patch("benchmarks.paper.artifact_io.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            atomic_json(target, {"generation": 2})
    assert json.loads(target.read_text()) == {"generation": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]


def test_cell_start_claim_has_exactly_one_owner(tmp_path):
    target = tmp_path / "started.json"

    def claim(worker):
        try:
            atomic_json(target, {"worker": worker}, exclusive=True)
            return worker
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        owners = [value for value in pool.map(claim, range(16)) if value is not None]
    assert len(owners) == 1
    assert json.loads(target.read_text()) == {"worker": owners[0]}


@pytest.mark.skipif(os.name != "posix", reason="Production readers use POSIX rename semantics")
def test_concurrent_prepare_never_exposes_partial_json(tmp_path):
    config = dict(json.loads(DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    output = tmp_path / "results"
    source = tmp_path / "engine"
    prepare(config, output, source)
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    code = """
import json, sys
from pathlib import Path
from benchmarks.paper.runner import prepare
config = json.loads(Path(sys.argv[1]).read_text())
for _ in range(8):
    prepare(config, Path(sys.argv[2]), Path(sys.argv[3]))
"""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    processes = [subprocess.Popen(
        [sys.executable, "-c", code, str(config_path), str(output), str(source)],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) for _ in range(3)]
    try:
        while any(process.poll() is None for process in processes):
            for name in ("config.resolved.json", "commands.json", "deployment_profile.json"):
                json.loads((output / name).read_text())
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, (stdout, stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=20)
    assert json.loads((output / "config.resolved.json").read_text()) == dict(
        resolve_history_kv_budgets(config), sglang_source=str(source.resolve()))


def test_rapid_checkout_updates_preserve_every_previous_config(tmp_path):
    config = dict(json.loads(DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    output = tmp_path / "results"
    for index in range(3):
        prepare(config, output, tmp_path / f"engine-{index}")
    archived = [json.loads(path.read_text()) for path in output.glob("config.before_extension.*.json")]
    assert {row["sglang_source"] for row in archived} == {
        str((tmp_path / f"engine-{index}").resolve()) for index in range(2)}
