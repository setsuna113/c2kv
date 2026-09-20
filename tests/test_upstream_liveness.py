"""CPU regressions for stopping an owned worker after engine loss."""

import errno
import subprocess
from unittest.mock import Mock, patch

import pytest

from generality.process_lifecycle import wait_owned_worker
from generality.upstream_liveness import UpstreamLiveness, UpstreamUnavailable


def test_three_connection_refusals_stop_without_http_generation():
    monitor = UpstreamLiveness("http://127.0.0.1:30000", refused_limit=3)
    refused = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
    with patch("generality.upstream_liveness.socket.create_connection", side_effect=refused) as connect:
        monitor()
        monitor()
        with pytest.raises(UpstreamUnavailable, match="3 consecutive connections"):
            monitor()
    assert connect.call_count == 3
    connect.assert_called_with(("127.0.0.1", 30000), timeout=1.0)


def test_timeout_is_not_evidence_of_a_dead_busy_engine():
    monitor = UpstreamLiveness("http://127.0.0.1:30000", refused_limit=2)
    refused = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
    timed_out = TimeoutError(errno.ETIMEDOUT, "busy")
    with patch("generality.upstream_liveness.socket.create_connection", side_effect=[
        refused, timed_out, refused,
    ]):
        monitor()
        monitor()
        monitor()
    assert monitor.refused == 1


def test_finished_worker_wins_race_with_engine_shutdown():
    proc = Mock(args=["worker"], poll=Mock(return_value=0))
    monitor = Mock(side_effect=UpstreamUnavailable("engine exited"))
    assert wait_owned_worker(proc, monitor=monitor) == 0
    monitor.assert_not_called()
    proc.wait.assert_not_called()


def test_monitor_interrupts_still_running_worker():
    proc = Mock(args=["worker"], poll=Mock(return_value=None))
    proc.wait.side_effect = subprocess.TimeoutExpired(proc.args, 0.01)
    calls = [0]

    def monitor():
        calls[0] += 1
        if calls[0] == 3:
            raise UpstreamUnavailable("engine exited")

    with pytest.raises(UpstreamUnavailable, match="engine exited"):
        wait_owned_worker(proc, monitor=monitor, poll_interval=0.01)
    assert calls[0] == 3


def test_worker_exit_is_not_reclassified_by_post_exit_monitor():
    proc = Mock(args=["worker"], poll=Mock(return_value=None))
    proc.wait.return_value = 0
    monitor = Mock(side_effect=[None, UpstreamUnavailable("engine exited")])
    assert wait_owned_worker(proc, monitor=monitor) == 0
    assert monitor.call_count == 1
