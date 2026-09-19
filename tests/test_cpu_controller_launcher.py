"""The CPU controller launcher sets thread limits before Python starts."""

import json
import shlex
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


@pytest.mark.parametrize("overrides,expected", [
    ({}, ("0", "8", "8", "8")),
    ({"OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "5",
      "MKL_NUM_THREADS": "7"}, ("0", "3", "5", "7")),
])
def test_launcher_sets_cpu_thread_limits_before_python_import(overrides, expected):
    code = "import json, os; print(json.dumps([os.environ.get(k) for k in " + repr(KEYS) + "]))"
    command = ["env"]
    for key in KEYS[1:]:
        command.extend(("-u", key))
    command.extend((
        "C2KV_CANN_SETUP=/dev/null", "C2KV_NNAL_SETUP=/dev/null",
        "TORCH_DEVICE_BACKEND_AUTOLOAD=1",
    ))
    command.extend(f"{key}={value}" for key, value in overrides.items())
    command.extend(("bash", "tools/launch_cpu_controller.sh", "python3", "-c", code))
    result = subprocess.run(
        ["bash", "-c", shlex.join(command)], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == list(expected)
