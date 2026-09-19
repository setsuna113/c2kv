"""A healthy old port must not be mistaken for a newly launched engine."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from generality import engine


class EngineReadyTests(unittest.TestCase):
    def test_exited_launch_is_checked_before_health(self):
        opener = Mock()
        with patch.object(engine, "_no_proxy_opener", return_value=opener), \
             patch.object(engine, "_pid_alive", return_value=False):
            self.assertEqual(
                engine.wait_ready(36205, timeout_s=1, pid=123),
                {"ready": False, "reason": "process_exit"},
            )
        opener.open.assert_not_called()

    def test_foreign_listener_is_not_ready(self):
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        opener = Mock()
        opener.open.return_value = response
        alive = iter((True, False))
        owns_listener = Mock(return_value=False)
        with patch.object(engine, "_no_proxy_opener", return_value=opener), \
             patch.object(engine, "_pid_alive", side_effect=lambda pid: next(alive)), \
             patch.object(engine, "_listener_in_session", owns_listener), \
             patch.object(engine.time, "sleep", return_value=None):
            self.assertEqual(
                engine.wait_ready(36205, timeout_s=1, pid=123),
                {"ready": False, "reason": "process_exit"},
            )
        self.assertEqual(opener.open.call_count, 1)
        owns_listener.assert_called_once_with(36205, 123)

    def test_owned_listener_is_ready(self):
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        opener = Mock()
        opener.open.return_value = response
        with patch.object(engine, "_no_proxy_opener", return_value=opener), \
             patch.object(engine, "_pid_alive", return_value=True), \
             patch.object(engine, "_listener_in_session", return_value=True):
            self.assertTrue(engine.wait_ready(36205, timeout_s=1, pid=123)["ready"])

    def test_listener_belongs_to_launched_session(self):
        ss = SimpleNamespace(
            returncode=0,
            stdout='LISTEN 0 2048 127.0.0.1:36205 0.0.0.0:* users:(("python",pid=4321,fd=58))',
        )
        with patch.object(engine.subprocess, "run", return_value=ss), \
             patch.object(engine.os, "getsid", return_value=123, create=True):
            self.assertTrue(engine._listener_in_session(36205, 123))
            self.assertFalse(engine._listener_in_session(36205, 999))


if __name__ == "__main__":
    unittest.main()
