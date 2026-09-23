"""Shared SGLang proxies keep episode state local to their own task."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm


class StartupReached(Exception):
    pass


@pytest.fixture
def startup_seam(monkeypatch):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "TOOL_MEMORY", None)
    monkeypatch.setattr(proxy, "ARM", None)
    monkeypatch.setattr(proxy, "BACKEND", None)
    monkeypatch.setattr(proxy, "SHARED_ENGINE", False)
    monkeypatch.setattr(
        proxy, "get_backend", lambda name, post: SimpleNamespace(name=name))
    monkeypatch.setattr(
        proxy, "ThreadingHTTPServer",
        lambda address, handler: (_ for _ in ()).throw(StartupReached()))


@pytest.mark.parametrize("arm", [
    "full", "acon_hist_ut_co_b256", "hiagent_full_b256", "hiagent_summary",
])
def test_shared_sglang_full_and_text_arms_reach_proxy_startup(startup_seam, arm):
    with pytest.raises(StartupReached):
        proxy.main(["--upstream", "http://127.0.0.1:1", "--backend", "sglang",
                    "--arm", arm, "--port", "1", "--shared-engine"])
    assert proxy.SHARED_ENGINE is True


@pytest.mark.parametrize("backend,arm", [
    ("hfserver", "full"),
    ("sglang", "c2kv4"),
    ("sglang", "cd_full"),
])
def test_shared_engine_rejects_other_backends_and_arms(startup_seam, backend, arm):
    with pytest.raises(ValueError, match="--shared-engine requires SGLang"):
        proxy.main(["--upstream", "http://127.0.0.1:1", "--backend", backend,
                    "--arm", arm, "--port", "1", "--shared-engine"])


def test_shared_text_episode_resets_local_state_without_flushing_other_work(monkeypatch):
    class Backend:
        supports_episode_reset = True

        def __init__(self):
            self.sessions = {"foreign-session"}
            self.closed = []

        def open_history_session(self, session_id):
            self.sessions.add(session_id)
            return session_id

        def close_history_session(self, session_id):
            assert session_id in self.sessions
            self.sessions.remove(session_id)
            self.closed.append(session_id)

        def flush_cache(self, timeout):
            pytest.fail("shared engine must not receive an engine-wide flush")

    backend = Backend()
    state = proxy.ProxyState()
    cache = proxy.ExtractCache()
    summary_cache = {}
    acon_state = {}
    method_states = {}
    monkeypatch.setattr(proxy, "BACKEND", backend)
    monkeypatch.setattr(proxy, "STATE", state)
    monkeypatch.setattr(proxy, "CACHE", cache)
    monkeypatch.setattr(proxy, "ARM", get_arm("acon_hist_ut_co"))
    monkeypatch.setattr(proxy, "SHARED_ENGINE", True)
    monkeypatch.setattr(proxy.textarms, "_SUMMARY_CACHE", summary_cache)
    monkeypatch.setattr(proxy.textarms, "_ACON_STATE", acon_state)
    monkeypatch.setattr(proxy.history_methods, "_STATES", method_states)

    proxy._activate_measurement_session("task-1")
    cache.get_or_put(("doc",), lambda: {"value": "task-1"})
    summary_cache["summary"] = "task-1"
    acon_state[("task-1", "acon_hist", "history")] = (1, "digest", "summary")
    method_states["conversation"] = object()
    owned_session = proxy._history_session_id("conversation")

    proxy._activate_measurement_session("task-1")
    assert cache.get_or_put(("doc",), lambda: pytest.fail("same task lost cache")) == {
        "value": "task-1"}
    assert summary_cache == {"summary": "task-1"}
    assert acon_state == {("task-1", "acon_hist", "history"): (1, "digest", "summary")}
    assert "conversation" in method_states
    assert backend.closed == []

    proxy._activate_measurement_session("task-2")
    assert cache.get_or_put(("doc",), lambda: {"value": "task-2"}) == {
        "value": "task-2"}
    assert summary_cache == {}
    assert acon_state == {}
    assert method_states == {}
    assert backend.closed == [owned_session]
    assert backend.sessions == {"foreign-session"}
    assert state.history_sessions == {}
