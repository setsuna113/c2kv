import json
from pathlib import Path

import pytest

from resource_policy import device_processes


def test_live_inventory_does_not_hide_occupied_devices():
    root = Path(__file__).resolve().parents[2]
    recorded = json.loads((root / "outputs/history_system_search/r001/resources.before_c2.json").read_text())
    processes = device_processes(recorded["npu_smi"]["stdout"])
    assert set(processes[7]) == {229399, 229942}
    assert {3540081, 3540678}.issubset(processes[4])
    assert all(processes[device] for device in (3, 5, 6))


def test_unknown_layout_is_not_idle():
    with pytest.raises(ValueError):
        device_processes("npu-smi failed")
