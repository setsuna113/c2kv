import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from benchmarks.paper.worker import claim, exclusive_lock, resolve_entry, wait_ports


def test_busy_port_is_preserved():
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen()
        port = owner.getsockname()[1]
        with pytest.raises(RuntimeError, match="Ports remain owned"):
            wait_ports([port], timeout=0)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass


def test_queue_overlay_keeps_config_and_root():
    assert resolve_entry("cell|config.json|results", Path("a"), Path("b")) == (
        "cell", Path("config.json"), Path("results"))
    with pytest.raises(ValueError):
        resolve_entry("../cell", Path("a"), Path("b"))


@pytest.mark.skipif(os.name != "posix", reason="Production locks use flock")
def test_claim_retains_durable_attempt_before_removing_queue(tmp_path):
    (tmp_path / "todo.closed_loop.txt").write_text("first\nsecond\n")
    receipt, row = claim(tmp_path, "closed_loop", {"pid": 42})
    assert json.loads(receipt.read_text())["entry"] == "first"
    assert (tmp_path / "todo.closed_loop.txt").read_text() == "second\n"
    assert row["status"] == "claimed"


@pytest.mark.skipif(os.name != "posix", reason="Production locks use flock")
def test_gpu_lock_rejects_second_owner(tmp_path):
    path = tmp_path / "gpu.lock"
    root = Path(__file__).resolve().parents[2]
    code = "from pathlib import Path; from benchmarks.paper.worker import exclusive_lock; import sys\nwith exclusive_lock(Path(sys.argv[1])): pass"
    with exclusive_lock(path):
        result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=root,
                                capture_output=True, text=True,
                                env=dict(os.environ, PYTHONPATH=str(root)))
    assert result.returncode != 0
    assert "Resource already owned" in result.stderr
    with exclusive_lock(path):
        pass
