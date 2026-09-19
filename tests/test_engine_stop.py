import json
import signal
from unittest.mock import patch

from generality import engine


def test_stop_refuses_unowned_or_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "ENGINE_LOG_DIR", tmp_path)
    (tmp_path / "test.launch.json").write_text(json.dumps({"pid": 123, "port": 36203}))
    with patch.object(engine, "_owned_engine_group", return_value=False), \
         patch.object(engine.os, "killpg", create=True) as kill:
        assert engine.stop("test")["reason"] == "engine_identity_mismatch"
        kill.assert_not_called()


def test_stop_signals_owned_group_including_children(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "ENGINE_LOG_DIR", tmp_path)
    (tmp_path / "test.launch.json").write_text(json.dumps({"pid": 123, "port": 36203}))
    with patch.object(engine, "_owned_engine_group", return_value=True), \
         patch.object(engine, "_group_alive", return_value=False), \
         patch.object(engine.os, "killpg", create=True) as kill:
        assert engine.stop("test")["signalled_group"] == 123
        kill.assert_called_once_with(123, signal.SIGTERM)


def test_engine_identity_checks_birth_and_exact_port():
    with patch.object(engine.os, "getsid", return_value=123, create=True), \
         patch.object(engine.os, "getpgid", return_value=123, create=True), \
         patch.object(engine, "_process_start_ticks", return_value=42), \
         patch.object(engine.Path, "read_bytes", return_value=b"python\0-m\0sglang.launch_server\0--port\0" + b"36203\0"):
        assert engine._owned_engine_group(123, 36203, 42)
        assert not engine._owned_engine_group(123, 36203, 41)
        assert not engine._owned_engine_group(123, 3620, 42)
