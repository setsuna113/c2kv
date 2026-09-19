from pathlib import Path
from unittest.mock import patch

import pytest

from controller_runtime.benchmarks.memory_runtime import event_native_appworld
from controller_runtime.benchmarks.memory_runtime import event_native_server
from generality import event_native_appworld as worker


def test_wrapper_launches_tracked_canonical_worker():
    root = Path(__file__).resolve().parents[1]
    expected = root / "generality" / "event_native_appworld.py"
    assert expected.is_file()

    with patch.object(event_native_appworld.runpy, "run_path") as run_path:
        event_native_appworld.main()

    run_path.assert_called_once_with(str(expected), run_name="__main__")


@pytest.mark.parametrize("sampling,accepted", [
    ({"mode": "greedy", "temperature": 0, "seed": 0}, True),
    (event_native_server._sampling_params_for_benchmark("acon_appworld"), True),
    ({"temperature": 0.0, "seed": 0, "top_p": 1.0}, True),
    ({"temperature": 0.001, "seed": 0}, False),
    ({"temperature": 0.0, "seed": 42}, False),
    ({"temperature": 0.0, "seed": 0, "presence_penalty": 0.5}, False),
    ({"temperature": False, "seed": 0}, False),
    ({"temperature": 0.0, "seed": False}, False),
])
def test_worker_accepts_only_the_actual_greedy_sampling_contract(sampling, accepted):
    ready = {
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": "acon_appworld", "decode_strategy": "incremental",
        "session_cache_policy": "external-sglang-content-addressed-chunks-v1",
        "route_contract": {"legacy_1088_equivalent": False},
        "allowed_task_ids": ["3d9a636_1"], "sampling": sampling,
    }
    health = dict(ready, schema="a-event-native-api-health-v1", terminal=False)
    if accepted:
        worker.validate_server_identity(ready, health)
    else:
        with pytest.raises(ValueError, match="sampling contract"):
            worker.validate_server_identity(ready, health)
