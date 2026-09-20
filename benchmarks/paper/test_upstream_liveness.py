"""Regression coverage for a server dying before a dataset loop finishes."""

import errno
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest import mock

import pytest

from benchmarks.paper.process_lifecycle import run_owned
from benchmarks.paper.upstream_liveness import UpstreamLiveness, UpstreamUnavailable


def test_refusal_is_bounded_and_success_resets_streak():
    monitor = UpstreamLiveness("http://127.0.0.1:38000/v1")
    refused = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
    with mock.patch("socket.create_connection", side_effect=refused):
        monitor()
        monitor()
    with mock.patch("socket.create_connection"):
        monitor()
    with mock.patch("socket.create_connection", side_effect=refused):
        monitor()
        monitor()
        with pytest.raises(UpstreamUnavailable, match="3 consecutive"):
            monitor()


def test_busy_endpoint_timeout_is_not_server_death():
    monitor = UpstreamLiveness("http://127.0.0.1:38000")
    with mock.patch("socket.create_connection", side_effect=socket.timeout("busy")):
        for _ in range(5):
            monitor()
    assert monitor.refused == 0


def test_owned_process_exit_stops_immediately_without_http_request():
    server = mock.Mock()
    server.poll.return_value = 1
    monitor = UpstreamLiveness("http://127.0.0.1:38000", process=server)
    with mock.patch("socket.create_connection") as connect:
        with pytest.raises(UpstreamUnavailable, match="exited with code 1"):
            monitor()
    connect.assert_not_called()


def test_monitored_child_preserves_complete_stdout_and_timeout():
    checked = mock.Mock()
    result = run_owned([sys.executable, "-c", "import time; print('first', flush=True); time.sleep(.2); print('last')"],
                       capture_output=True, text=True, monitor=checked, poll_interval=.03)
    assert result.stdout == "first\nlast\n"
    assert checked.call_count > 2
    with pytest.raises(subprocess.TimeoutExpired):
        run_owned([sys.executable, "-c", "import time; time.sleep(60)"],
                  timeout=.1, monitor=mock.Mock(), poll_interval=.03)


@pytest.mark.skipif(os.name != "posix", reason="Owned process groups need POSIX")
def test_server_exit_preserves_first_result_and_stops_remaining_tasks(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    ready = tmp_path / "ready"
    server = subprocess.Popen([
        sys.executable, "-c",
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.write_text('ready'); "
        "time.sleep(.6)", str(ready),
    ])
    # This monitor represents the owned server, avoiding unrelated ports.
    def monitor():
        if server.poll() is not None:
            raise UpstreamUnavailable("owned server exited")

    worker = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "p.write_text('completed'); time.sleep(60); "
        "pathlib.Path(sys.argv[2]).write_text('must not run')"
    )
    try:
        with pytest.raises(UpstreamUnavailable):
            run_owned([sys.executable, "-c", worker, str(first), str(second)],
                      monitor=monitor, poll_interval=.03)
        assert first.read_text() == "completed"
        assert not second.exists()
    finally:
        server.wait(timeout=5)
