from __future__ import annotations

import socket
import subprocess

import pytest

from benchmarks import run


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeProcess:
    def __init__(self, returncodes):
        self._returncodes = iter(returncodes)
        self._last_returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        try:
            self._last_returncode = next(self._returncodes)
        except StopIteration:
            pass
        return self._last_returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        return self._last_returncode


class _HealthResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def read(self):
        return b"ok"


def test_start_proxy_rejects_a_preowned_port_before_spawn(monkeypatch, tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])

        def unexpected_spawn(*args, **kwargs):
            del args, kwargs
            raise AssertionError("occupied port must fail before Popen")

        monkeypatch.setattr(run.subprocess, "Popen", unexpected_spawn)
        with pytest.raises(SystemExit, match=f"proxy port {port} is unavailable"):
            run.start_proxy("http://upstream", "full", port, tmp_path)


def test_stale_health_cannot_hide_a_child_bind_failure(monkeypatch, tmp_path):
    process = _FakeProcess([None, 98])
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: process)

    import urllib.request

    opener = type(
        "HealthyOldListener",
        (),
        {"open": lambda self, *args, **kwargs: _HealthResponse()},
    )()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(run.time, "sleep", lambda _: None)

    with pytest.raises(SystemExit, match="exited during startup with code 98"):
        run.start_proxy(
            "http://upstream", "full", _free_port(), tmp_path
        )

    assert process.waited is True
    assert process.terminated is False


def test_owned_process_shutdown_terminates_and_waits():
    process = _FakeProcess([None])

    run._terminate_owned_process(process)

    assert process.terminated is True
    assert process.waited is True
    assert process.killed is False


def test_startup_exception_reaps_the_spawned_process(monkeypatch, tmp_path):
    process = _FakeProcess([None, None])
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: process)

    import urllib.request

    opener = type(
        "BrokenHealthProbe",
        (),
        {"open": lambda self, *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("broken probe")
        )},
    )()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)

    with pytest.raises(RuntimeError, match="broken probe"):
        run.start_proxy(
            "http://upstream", "full", _free_port(), tmp_path
        )

    assert process.terminated is True
    assert process.waited is True
