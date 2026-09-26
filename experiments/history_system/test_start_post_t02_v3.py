import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "tmp/start_post_t02_v3.py"
SPEC = importlib.util.spec_from_file_location("start_post_t02_v3", SCRIPT)
post = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(post)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _static_binding(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "root"
    training = base / "prepared_v7"
    relatives = ("source_files.json", "contract.json", "recovery/recovery_contract.json",
                 "recovery/source_plan.json")
    for relative in relatives:
        _write(training / relative, {"file": relative})
    binding = base / "post_t02_v3/prepared_v7.binding.json"
    _write(binding, {"schema": "post-t02-training-binding-v1",
                     "training_package": str(training.resolve()),
                     "sha256": {relative: hashlib.sha256((training / relative).read_bytes()).hexdigest()
                                for relative in relatives}})
    return base, binding


def test_v3_schedule_is_three_once_only_original_d20_lanes(tmp_path: Path):
    base, binding = _static_binding(tmp_path)
    value = post.verify_static_binding(base, binding)
    commands = post.build_commands(base, base / "post_t02_v3", binding, "/python")
    assert value["training_package"] == str((base / "prepared_v7").resolve())
    assert [lane for lane, _ in commands] == ["C1", "C4_turn", "C4_task"]
    assert all("prepared_v7" in " ".join(command)
               and "eval_trained_v3" in " ".join(command) for _, command in commands)
    assert all("prepared_v6" not in " ".join(command) for _, command in commands)
    assert commands[2][1][-2:] == ["--after-status",
        str(base / "eval_failed_repair_v1/lanes/C0/run/status.json")]


def test_v3_schedule_rejects_changed_frozen_v7_input(tmp_path: Path):
    base, binding = _static_binding(tmp_path)
    (base / "prepared_v7/contract.json").write_text("{}\n")
    with pytest.raises(ValueError, match="differs"):
        post.verify_static_binding(base, binding)
