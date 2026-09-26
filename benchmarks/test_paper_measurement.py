from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.adapters import bfcl_adapter
from benchmarks.arms import get_arm
from benchmarks.backends.sglang import SglangBackend
from benchmarks.backends.base import BackendError
from benchmarks import proxy as proxy_mod
from benchmarks.measurement.aggregate import aggregate, distribution
from benchmarks.measurement.replay import replay_prefixes
from benchmarks.measurement.telemetry import (
    HarnessTelemetry, canonical_sha256, read_jsonl,
)


def test_harness_events_join_decision_to_action_and_episode(tmp_path):
    path = tmp_path / "harness.jsonl"
    telemetry = HarnessTelemetry(path, "bfcl")
    with telemetry.episode("case-1"):
        telemetry.record_decision(
            request_id="request-1", start_unix_ns=100, duration_ns=20,
            response={"ok": True},
        )
        telemetry.record_action(
            action="tool(x=1)", outcome="done", start_unix_ns=130,
            duration_ns=10,
        )
    rows = list(read_jsonl(path))
    action = next(row for row in rows if row["event_type"] == "tool_action")
    assert action["decision_request_id"] == "request-1"
    assert action["episode_id"] == "case-1"
    assert {row["event_type"] for row in rows} == {
        "episode_start", "decision", "tool_action", "episode_end"}


def test_harness_episode_accepts_unfrozen_clock_sources(tmp_path):
    unix_values = iter((1_000, 1_100, 1_200, 1_300))
    monotonic_values = iter((10, 35))
    path = tmp_path / "clocked.jsonl"
    telemetry = HarnessTelemetry(
        path, "appworld",
        unix_ns=lambda: next(unix_values),
        monotonic_ns=lambda: next(monotonic_values),
    )
    with telemetry.episode("task-1"):
        pass
    rows = list(read_jsonl(path))
    assert rows[-1]["duration_ns"] == 25
    assert rows[-1]["start_unix_ns"] == 1_000
    assert rows[-1]["end_unix_ns"] == 1_200


def test_aggregate_uses_server_jsonl_and_reports_missing_coverage():
    proxy = [{
        "event_type": "request", "request_id": "r1",
        "start_unix_ns": 100, "duration_ns": 50,
        "server_measurement": {},
    }, {
        "event_type": "request", "request_id": "r-empty",
        "start_unix_ns": 200, "duration_ns": 30,
        "server_measurement": {},
    }]
    harness = [{
        "event_type": "tool_action", "decision_request_id": "r1",
        "end_unix_ns": 180, "duration_ns": 10,
    }]
    server = [{
        "event_type": "server_request", "request_id": "r1",
        "phase": "c2kv_extract",
        "measurement": {
            "history_full_kv_tokens": 100,
            "history_active_kv_tokens": 25,
            "request_peak_resident_kv_tokens": 120,
            "request_peak_resident_kv_bytes": 480,
            "denominator_tokenization_duration_ns": None,
        },
    }, {
        "event_type": "server_request", "request_id": "r1",
        "phase": "generation",
        "measurement": {
            "history_full_kv_tokens": 100,
            "history_active_kv_tokens": 25,
            "request_peak_resident_kv_tokens": 80,
            "denominator_tokenization_duration_ns": 10,
        },
    }, {
        "event_type": "server_request", "request_id": "r1",
        "phase": "hiagent_retrieval_generation",
        "measurement": {
            "history_full_kv_tokens": 100,
            "history_active_kv_tokens": 25,
            "request_peak_resident_kv_tokens": 90,
            "denominator_tokenization_duration_ns": 5,
        },
    }]
    replay = [{
        "event_type": "prefix_replay",
        "duration_ns": 10,
        "source_paper_measurement": {"metrics": {
            "whole_full_kv_tokens": 200, "history_full_kv_tokens": 100}},
        "target_paper_measurement": {"metrics": {
            "whole_active_kv_tokens": 80, "history_active_kv_tokens": 25}},
    }]
    result = aggregate(proxy, harness, replay_rows=replay, server_rows=server)
    headline = result["latency_ms"]["complete_model_side_per_committed_action"]
    assert headline["observed_model_side_ns"] == 80
    assert headline["measurement_denominator_tokenization_ns"] == 15
    assert headline["total_model_side_ns"] == 65
    assert headline["mean_ms"] == pytest.approx(0.000065)
    assert result["latency_ms"]["action_associated_model_side_allocation"]["mean"] == pytest.approx(0.000035)
    assert result["latency_ms"]["measurement_denominator_tokenization"]["mean"] == pytest.approx(0.000015)
    assert result["latency_ms"]["zero_action_model_side"]["requests"] == 1
    assert result["latency_ms"]["decision_start_to_action_commit_wall"]["mean"] == pytest.approx(0.00008)
    assert result["token_ratios"]["history"]["ratio_of_sums"] == 0.25
    assert result["memory"]["request_peak_resident_kv_tokens"]["coverage"] == {
        "measured": 1, "requests": 2}
    assert result["memory"]["request_peak_resident_kv_tokens"]["mean"] == 120
    assert result["memory"]["resident_peak_chain"]["request_id"] == "r1"
    assert result["memory"]["resident_peak_chain"]["request_peak_cached_evictable_kv_bytes"] is None
    assert result["memory"]["torch_peak_reserved_bytes"]["mean"] is None
    assert result["common_prefix_token_ratios"]["whole"]["ratio_of_sums"] == 0.4
    assert result["common_prefix_token_ratios"]["history"]["ratio_of_sums"] == 0.25


def test_cached_evictable_follows_the_chain_resident_peak():
    from benchmarks.measurement.aggregate import _merge_server_measurement
    chain = {}
    # Auxiliary compression call: its own prompt is the chain's resident peak;
    # nothing was cached before it.
    _merge_server_measurement(chain, {
        "request_peak_resident_kv_bytes": 426_000_000,
        "request_peak_cached_evictable_kv_bytes": 0,
        "cached_evictable_kv_peak_bytes": 0,
    }, "aux_compression")
    # Generation after it: smaller own prompt, but the auxiliary request's KV
    # is still resident as evictable cache, so the chain peak moves here.
    _merge_server_measurement(chain, {
        "request_peak_resident_kv_bytes": 456_000_000,
        "request_peak_cached_evictable_kv_bytes": 426_000_000,
        "request_peak_c2kv_cached_evictable_kv_bytes": 50_000_000,
        "request_peak_c2kv_cache_accounting_available": True,
        "cached_evictable_kv_peak_bytes": 426_000_000,
    }, "generation")
    assert chain["request_peak_resident_kv_bytes"] == 456_000_000
    assert chain["request_peak_cached_evictable_kv_bytes"] == 426_000_000
    assert chain["request_peak_c2kv_cached_evictable_kv_bytes"] == 50_000_000
    assert chain["request_peak_c2kv_cache_accounting_available"] is True
    assert chain["cached_evictable_kv_peak_bytes"] == 426_000_000
    # A later, smaller request must not drag the at-peak line item with it.
    _merge_server_measurement(chain, {
        "request_peak_resident_kv_bytes": 60_000_000,
        "request_peak_cached_evictable_kv_bytes": 5_000_000,
        "request_peak_c2kv_cached_evictable_kv_bytes": 0,
        "request_peak_c2kv_cache_accounting_available": False,
        "cached_evictable_kv_peak_bytes": 5_000_000,
    }, "generation")
    assert chain["request_peak_resident_kv_bytes"] == 456_000_000
    assert chain["request_peak_cached_evictable_kv_bytes"] == 426_000_000
    assert chain["request_peak_c2kv_cached_evictable_kv_bytes"] == 50_000_000
    assert chain["request_peak_c2kv_cache_accounting_available"] is True
    assert chain["cached_evictable_kv_peak_bytes"] == 426_000_000


def test_distribution_has_declared_linear_percentiles():
    result = distribution([0, 10, 20, 30])
    assert result["n"] == 4
    assert result["mean"] == 15.0
    assert result["p50"] == 15.0
    assert result["p95"] == pytest.approx(28.5)
    assert result["p99"] == pytest.approx(29.7)
    assert result["max"] == 30


def test_sglang_does_not_alias_process_high_water_to_request_peak():
    backend = SglangBackend(lambda *args, **kwargs: {})
    normalized = backend.normalize_response({
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "metadata": {"sglang_runtime": {
            "kv_peak_resident_tokens": 999,
            "request_measurement": {"request_peak_resident_kv_tokens": 123},
        }},
    })
    assert normalized["cost"]["kv_peak_resident_tokens"] == 999
    assert normalized["cost"]["server_measurement"]["request_peak_resident_kv_tokens"] == 123


def test_sglang_reads_chat_measurement_from_runtime_contract():
    backend = SglangBackend(lambda *args, **kwargs: {})
    normalized = backend.normalize_response({
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "metadata": {"sglang_runtime": {"paper_measurement": {
            "outer_request_id": "outer-1",
            "metrics": {
                "request_peak_resident_kv_tokens": 321,
                "whole_full_kv_tokens": 100,
                "whole_active_kv_tokens": 25,
            },
        }}},
    })
    measurement = normalized["cost"]["server_measurement"]
    assert measurement["outer_request_id"] == "outer-1"
    assert measurement["request_peak_resident_kv_tokens"] == 321
    assert measurement["whole_full_kv_tokens"] == 100
    assert measurement["whole_active_kv_tokens"] == 25


def test_sglang_episode_reset_calls_close_then_flush_admin_endpoints():
    calls = []
    def post(path, payload, timeout):
        calls.append((path, payload, timeout))
        return None if path == "/close_session" else "Cache flushed.\nmore"
    backend = SglangBackend(post)
    backend.close_history_session("session-1")
    backend.flush_cache(timeout=10)
    assert calls == [
        ("/close_session", {"session_id": "session-1"}, 60),
        ("/flush_cache?timeout=10.0", {}, 15),
    ]


def test_timed_bfcl_executor_preserves_order_and_records_each_action(tmp_path, monkeypatch):
    package = types.ModuleType("bfcl_eval")
    model_handler = types.ModuleType("bfcl_eval.model_handler")
    base_handler = types.ModuleType("bfcl_eval.model_handler.base_handler")
    state = []

    def execute(actions, *args, **kwargs):
        if actions:
            state.extend(actions)
        return [f"out:{action}" for action in actions], {"state": list(state)}

    base_handler.execute_multi_turn_func_call = execute
    package.model_handler = model_handler
    model_handler.base_handler = base_handler
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.model_handler", model_handler)
    monkeypatch.setitem(sys.modules, "bfcl_eval.model_handler.base_handler", base_handler)

    path = tmp_path / "actions.jsonl"
    telemetry = HarnessTelemetry(path, "bfcl")
    bfcl_adapter._install_timed_executor(telemetry)
    with telemetry.episode("case"):
        telemetry.record_decision(
            request_id="r", start_unix_ns=1, duration_ns=1)
        results, instances = base_handler.execute_multi_turn_func_call(
            ["a()", "b()"], {}, [], "model", "case")
    assert results == ["out:a()", "out:b()"]
    assert instances == {"state": ["a()", "b()"]}
    actions = [row for row in read_jsonl(path) if row["event_type"] == "tool_action"]
    assert [row["action"] for row in actions] == ["a()", "b()"]
    assert all(row["decision_request_id"] == "r" for row in actions)


def test_replay_executes_canonical_prefixes_in_order(tmp_path, monkeypatch):
    prefixes = tmp_path / "prefixes.jsonl"
    output = tmp_path / "replay.jsonl"
    payloads = [{"messages": [{"role": "user", "content": str(i)}]}
                for i in range(2)]
    prefixes.write_text("".join(json.dumps({
        "event_type": "recorded_prefix", "source_arm": "full",
        "prefix_id": f"p{i}", "canonical_sha256": canonical_sha256(payload),
        "replay_payload": payload,
        "source_response": {"paper_measurement": {"metrics": {
            "whole_full_kv_tokens": 10 + i}}},
    }) + "\n" for i, payload in enumerate(payloads)), encoding="utf-8")

    seen = []

    class Response:
        status = 200
        def __init__(self, request):
            self.request = request
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            body = json.loads(self.request.data.decode())
            seen.append(body)
            return json.dumps({"choices": []}).encode()

    class Opener:
        def open(self, request, timeout):
            return Response(request)

    monkeypatch.setattr("benchmarks.measurement.replay._OPENER", Opener())
    result = replay_prefixes(
        prefixes, "http://localhost:1", output,
        source_run_id="full", target_run_id="h2o")
    assert result == {"prefixes": 2, "completed": 2, "failed": 0, "context_overflow": 0}
    assert seen == payloads
    rows = list(read_jsonl(output))
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["source_paper_measurement"]["metrics"]["whole_full_kv_tokens"] == 10


def test_replay_tolerates_only_context_overflow_failures(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    from benchmarks.measurement import replay as replay_mod
    prefixes = tmp_path / "prefixes.jsonl"
    output = tmp_path / "replay.jsonl"
    payloads = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(3)]
    prefixes.write_text("".join(json.dumps({
        "event_type": "recorded_prefix", "source_arm": "full",
        "prefix_id": f"p{i}", "canonical_sha256": canonical_sha256(payload),
        "replay_payload": payload, "source_response": {},
    }) + "\n" for i, payload in enumerate(payloads)), encoding="utf-8")
    overflow = json.dumps({"error": "upstream failed: upstream 400: {\"message\":\"The input (138202 tokens) "
                           "is longer than the model's context length (131072 tokens).\"}"}).encode()
    other = json.dumps({"error": "upstream failed: connection refused"}).encode()

    class Ok:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"choices": []}).encode()

    class Opener:
        def __init__(self, bodies): self.bodies = list(bodies)
        def open(self, request, timeout):
            body = self.bodies.pop(0)
            if body is None:
                return Ok()
            raise HTTPError(request.full_url, 502, "Bad Gateway", {}, __import__("io").BytesIO(body))

    monkeypatch.setattr("benchmarks.measurement.replay._OPENER", Opener([None, overflow, overflow]))
    result = replay_prefixes(prefixes, "http://localhost:1", output, source_run_id="full", target_run_id="x")
    assert result == {"prefixes": 3, "completed": 1, "failed": 2, "context_overflow": 2}
    # the CLI exits cleanly when every failure is a context overflow, and records the summary
    monkeypatch.setattr("benchmarks.measurement.replay._OPENER", Opener([None, overflow, overflow]))
    out2 = tmp_path / "replay2.jsonl"
    replay_mod.main(["--prefixes", str(prefixes), "--base-url", "http://localhost:1", "--output", str(out2),
                     "--source-run-id", "full", "--target-run-id", "x"])
    assert json.loads((tmp_path / "replay_summary.json").read_text())["context_overflow"] == 2
    # any other failure still fails the replay
    monkeypatch.setattr("benchmarks.measurement.replay._OPENER", Opener([None, overflow, other]))
    with pytest.raises(SystemExit, match="1 failed"):
        replay_mod.main(["--prefixes", str(prefixes), "--base-url", "http://localhost:1",
                         "--output", str(tmp_path / "replay3.jsonl"),
                         "--source-run-id", "full", "--target-run-id", "x"])


def test_bfcl_paper_measurement_rejects_concurrent_generator():
    with pytest.raises(ValueError, match="single-flight"):
        bfcl_adapter.generate_argv("model", "multi_turn_base", num_threads=2)


def test_episode_boundary_closes_session_flushes_and_clears_local_state(monkeypatch):
    class Backend:
        supports_episode_reset = True
        def __init__(self):
            self.closed = []
            self.flushes = 0
        def close_history_session(self, session_id):
            self.closed.append(session_id)
        def flush_cache(self, timeout):
            assert timeout == 10
            self.flushes += 1

    backend = Backend()
    state = proxy_mod.ProxyState()
    cache = proxy_mod.ExtractCache()
    cache.get_or_put(("x",), lambda: {"x": 1})
    resets = []
    monkeypatch.setattr(proxy_mod, "BACKEND", backend)
    monkeypatch.setattr(proxy_mod, "STATE", state)
    monkeypatch.setattr(proxy_mod, "CACHE", cache)
    monkeypatch.setattr(proxy_mod.textarms, "reset_state", lambda: resets.append(True))

    proxy_mod._activate_measurement_session("task-1")
    state.history_sessions["conv"] = "server-session-1"
    proxy_mod._activate_measurement_session("task-1")
    assert backend.flushes == 1
    proxy_mod._activate_measurement_session("task-2")
    assert backend.closed == ["server-session-1"]
    assert backend.flushes == 2
    assert state.history_sessions == {}
    assert resets == [True, True]


def test_shared_episode_closes_only_owned_session_without_global_flush(monkeypatch):
    class Backend:
        supports_episode_reset = True
        def __init__(self):
            self.opened = []
            self.closed = []
            self.flushes = 0
        def open_history_session(self, session_id):
            self.opened.append(session_id)
            return session_id
        def close_history_session(self, session_id):
            self.closed.append(session_id)
        def flush_cache(self, timeout):
            self.flushes += 1

    backend = Backend()
    state = proxy_mod.ProxyState()
    monkeypatch.setattr(proxy_mod, "BACKEND", backend)
    monkeypatch.setattr(proxy_mod, "STATE", state)
    monkeypatch.setattr(proxy_mod, "SHARED_ENGINE", True)
    proxy_mod._activate_measurement_session("probe")
    engine_id = proxy_mod._history_session_id("probe-conversation")
    assert engine_id != "probe" and engine_id.startswith("c2kv-bench-history-")
    proxy_mod._activate_measurement_session("next-task")
    assert backend.opened == backend.closed == [engine_id]
    assert backend.flushes == 0
    assert state.history_sessions == {}


def test_full_shared_engine_starts_and_resets_locally_without_flushing(monkeypatch):
    class Backend:
        name = "sglang"
        supports_episode_reset = True

        def __init__(self):
            self.flushes = 0

        def flush_cache(self, timeout):
            self.flushes += 1

    backend = Backend()
    monkeypatch.setattr(proxy_mod, "get_backend", lambda name, post: backend)
    monkeypatch.setattr(proxy_mod, "STATE", proxy_mod.ProxyState())
    monkeypatch.setattr(proxy_mod, "CACHE", proxy_mod.ExtractCache())

    class Server:
        def __init__(self, address, handler):
            pass

        def serve_forever(self):
            assert proxy_mod.ARM.name == "full" and proxy_mod.SHARED_ENGINE
            proxy_mod._activate_measurement_session("task-1")
            proxy_mod.CACHE.get_or_put(("local",), lambda: {"value": 1})
            proxy_mod._activate_measurement_session("task-2")
            assert proxy_mod.CACHE._cache == {}

        def server_close(self):
            pass

    monkeypatch.setattr(proxy_mod, "ThreadingHTTPServer", Server)
    proxy_mod.main(["--upstream", "http://127.0.0.1:1", "--backend", "sglang",
                    "--arm", "full", "--port", "1", "--shared-engine"])
    assert backend.flushes == 0


def test_proxy_close_endpoint_resolves_measurement_id_to_owned_engine_id(monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen

    class Backend:
        def __init__(self): self.closed = []
        def close_history_session(self, session_id): self.closed.append(session_id)

    backend = Backend()
    state = proxy_mod.ProxyState()
    state.active_measurement_session = "probe"
    state.history_sessions["hashed-probe"] = "c2kv-bench-history-owned"
    monkeypatch.setattr(proxy_mod, "BACKEND", backend)
    monkeypatch.setattr(proxy_mod, "STATE", state)
    server = ThreadingHTTPServer(("127.0.0.1", 0), proxy_mod.ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/close_measurement_session",
            data=json.dumps({"c2kv_measurement_session_id": "probe"}).encode(),
            headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            assert json.load(response) == {"closed_owned_sessions": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert backend.closed == ["c2kv-bench-history-owned"]
    assert state.history_sessions == {}


def test_proxy_sigterm_closes_last_owned_session_with_absolute_budget(monkeypatch):
    import signal

    class Backend:
        name = "sglang"
        def __init__(self): self.closed = []
        def close_history_session(self, session_id): self.closed.append(session_id)

    backend = Backend()
    state = proxy_mod.ProxyState()
    monkeypatch.setattr(proxy_mod, "BACKEND", backend)
    monkeypatch.setattr(proxy_mod, "STATE", state)
    monkeypatch.setattr(proxy_mod, "get_backend", lambda name, post: backend)

    class Server:
        def __init__(self, address, handler): self.closed = False
        def serve_forever(self):
            spec = proxy_mod.history_kv_spec(proxy_mod.ARM)
            assert (spec["target_tokens"], spec["retention_ratio"]) == (8, None)
            state.history_sessions["conversation"] = "engine-session-last"
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
        def server_close(self): self.closed = True

    monkeypatch.setattr(proxy_mod, "ThreadingHTTPServer", Server)
    with pytest.raises(SystemExit):
        proxy_mod.main([
            "--upstream", "http://127.0.0.1:1", "--backend", "sglang",
            "--arm", "gen_h2o_k0", "--port", "1",
            "--history-kv-target-tokens", "8", "--shared-engine"])
    assert backend.closed == ["engine-session-last"]


def test_sglang_close_uses_exact_engine_session_id_and_checks_response():
    calls = []
    backend = SglangBackend(lambda path, payload, timeout:
                            calls.append((path, payload, timeout)) or "")
    backend.close_history_session("c2kv-bench-history-owned")
    assert calls == [("/close_session",
                      {"session_id": "c2kv-bench-history-owned"}, 60)]
    backend = SglangBackend(lambda path, payload, timeout: {"error": "still open"})
    with pytest.raises(BackendError, match="unexpected close_session response"):
        backend.close_history_session("c2kv-bench-history-owned")


@pytest.mark.parametrize("arm_name", [
    "full", "c2kv4", "history_kv_h2o_r25_persistent",
    "history_kv_snapkv_r25_persistent",
])
def test_server_can_render_exact_full_denominator_for_unmodified_text(arm_name):
    assert proxy_mod._canonical_full_source(get_arm(arm_name)) is True


@pytest.mark.parametrize("arm_name", ["hiagent_full", "acon_hist_ut_co"])
def test_text_policy_requires_recorded_full_denominator(arm_name):
    assert proxy_mod._canonical_full_source(get_arm(arm_name)) is False


def test_paper_history_boundary_uses_explicit_empty_first_turn_contract():
    assert proxy_mod._paper_history_message_boundary(
        [{"role": "system"}, {"role": "user"}],
        {"current_start_out_index": 1},
    ) == (0, 0)


def test_paper_history_boundary_excludes_system_and_current_block():
    assert proxy_mod._paper_history_message_boundary(
        [
            {"role": "system"}, {"role": "user"},
            {"role": "assistant"}, {"role": "user"},
        ],
        {"current_start_out_index": 3},
    ) == (1, 3)
