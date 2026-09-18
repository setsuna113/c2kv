import copy
import json
from pathlib import Path
import tempfile

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper.c1 import ARM, replay_task_id, selected_tasks
from benchmarks.paper.runner import DEFAULT_CONFIG, prepare, server_command


def test_native_arm_requires_real_controller_and_keeps_bare_c2kv():
    assert get_arm(ARM).native_controller == "c1_t02"
    assert get_arm(ARM).ratio == 8
    assert get_arm("c2kv4").ratio == 4
    config = json.loads(DEFAULT_CONFIG.read_text())
    command = server_command(config, Path("sglang"), ARM)
    assert "--enable-return-hidden-states" in command
    assert "--c2kv-shadow-feature-layer" in command
    assert "--disable-cuda-graph" not in command
    assert "--enable-return-hidden-states" not in server_command(config, Path("sglang"), "full")


def test_append_final_arm_preserves_old_cells_and_completed_artifacts():
    config = json.loads(DEFAULT_CONFIG.read_text())
    previous = copy.deepcopy(config)
    previous["methods"] = previous["methods"][:-1]
    previous.pop("c1")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        old_plan, _ = prepare(previous, output, output / "sglang")
        completed = output / "closed_loop" / old_plan[0]["cell_id"] / "complete.json"
        completed.parent.mkdir(parents=True)
        completed.write_text('{"old_result": true}\n')
        new_plan, _ = prepare(config, output, output / "sglang")
        assert [row["cell_id"] for row in new_plan[:24]] == [row["cell_id"] for row in old_plan]
        assert completed.read_text() == '{"old_result": true}\n'
        assert json.loads((output / "config.before_c1_extension.json").read_text())["methods"] == previous["methods"]
        assert all("benchmarks.paper.c1" in row["command"] for row in new_plan[-3:])
        # An algorithm change to a previous arm cannot masquerade as extension.
        changed = copy.deepcopy(config)
        changed["methods"][3]["ratio"] = 8
        with pytest.raises(ValueError):
            prepare(changed, output, output / "sglang")


def test_task_selection_uses_the_official_file_and_rejects_foreign_ids():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "bfcl_eval" / "data"
        data.mkdir(parents=True)
        (data / "BFCL_v4_multi_turn_base.json").write_text(
            '{"id":"multi_turn_base_1"}\n{"id":"multi_turn_base_2"}\n')
        config = {"bfcl_dir": str(root)}
        assert selected_tasks(config, "bfcl_base") == ["multi_turn_base_1", "multi_turn_base_2"]
        assert selected_tasks(config, "bfcl_base", ["multi_turn_base_2"]) == ["multi_turn_base_2"]
        with pytest.raises(ValueError):
            selected_tasks(config, "bfcl_base", ["multi_turn_base_99"])


def test_replay_keeps_official_task_identity_for_native_evidence_ids():
    rows = [{"replay_payload": {"c2kv_measurement_session_id": "multi_turn_base_26"}}] * 2
    assert replay_task_id(rows, "hashed-proxy-conversation") == "multi_turn_base_26"
    assert replay_task_id([{"replay_payload": {}}], "synthetic").startswith("replay_")
