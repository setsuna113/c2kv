"""Official tau2 selection, scoring, and agent-only identity seams."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.adapters import tau2_adapter as tau2
from benchmarks.measurement.telemetry import canonical_sha256


def _result(task_id: str, reward=None, *, termination="agent_stop"):
    return {"task_id": task_id, "trial": 0,
            "termination_reason": termination,
            "messages": [{"role": "assistant", "content": "Done."}],
            "reward_info": {"reward": reward} if reward is not None else None}


def test_selected_task_ids_uses_official_resolver_and_checkout(monkeypatch, tmp_path):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return types.SimpleNamespace(stdout='["11", "19"]\n')

    monkeypatch.setattr(tau2, "run_owned", fake_run)
    assert tau2.selected_task_ids(source, "/venv/python", task_set="airline",
                                  split="base", task_ids=["11", "19"]) == ["11", "19"]
    command, options = seen[0]
    assert command[:2] == ["/venv/python", "-c"]
    assert json.loads(command[3]) == ["airline", "base", ["11", "19"], None]
    assert options["cwd"] == source.resolve()
    assert str(source.resolve() / "src") in options["env"]["PYTHONPATH"]


@pytest.mark.parametrize("bad", ['["11", "11"]', '[]', '["12"]'])
def test_selected_task_ids_rejects_wrong_official_inventory(monkeypatch, tmp_path, bad):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    monkeypatch.setattr(tau2, "run_owned",
                        lambda *_args, **_kwargs: types.SimpleNamespace(stdout=bad))
    with pytest.raises(RuntimeError, match="task resolver"):
        tau2.selected_task_ids(source, "/venv/python", task_ids=["11"])


def test_run_tau2_scores_exact_official_task_and_routes_user_raw(monkeypatch, tmp_path):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    commands = []
    sims = source / "data" / "simulations" / "run_11"
    sims.mkdir(parents=True)

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if "-c" in command:
            return types.SimpleNamespace(
                stdout='["11"]' if "get_tasks" in command[2] else "[]")
        if "run" in command:
            (sims / "results.json").write_text(
                json.dumps({"simulations": [_result("11")]}), encoding="utf-8")
            events = out / "measurement" / "harness_events.jsonl"
            events.parent.mkdir(parents=True, exist_ok=True)
            events.write_text("\n".join(json.dumps(row) for row in (
                {"event_type": "episode_start", "episode_id": "11"},
                {"event_type": "decision", "episode_id": "11"},
                {"event_type": "episode_end", "episode_id": "11", "status": "ok"},
            )) + "\n", encoding="utf-8")
        elif "evaluate-trajs" in command:
            (sims / "updated_results.json").write_text(
                json.dumps({"simulations": [_result("11", 1.0)]}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tau2, "run_owned", fake_run)
    out = tmp_path / "out"
    summary = tau2.run_tau2(
        "http://agent", "http://raw", out, tau2_dir=source,
        python="/venv/python", run_name="run_11", task_ids=["11"],
        native=True, model="served")
    assert summary["n"] == 1
    assert summary["task_ids"] == ["11"]
    assert summary["protocol"]["user_simulator_transport_retries"] == 1
    assert summary["task_rows"][0]["semantic_score"] == 1.0
    assert (out / "official" / "updated_results.json").is_file()
    run, options = next((command, opts) for command, opts in commands if "run" in command)
    assert run[run.index("--task-ids") + 1] == "11"
    agent = json.loads(run[run.index("--agent-llm-args") + 1])
    user = json.loads(run[run.index("--user-llm-args") + 1])
    assert agent["api_base"] == "http://agent/v1"
    assert agent["max_tokens"] == 4096
    assert agent["num_retries"] == user["num_retries"] == 0
    assert user["api_base"] == "http://raw/v1"
    assert run[run.index("--max-retries") + 1] == "0"
    assert run[run.index("--hallucination-retries") + 1] == "0"
    assert options["env"]["C2KV_TAU2_NATIVE"] == "1"
    evaluation_env = next(opts["env"] for command, opts in commands
                          if "evaluate-trajs" in command)
    assert "C2KV_TAU2_TELEMETRY_PATH" not in evaluation_env
    assert "C2KV_TAU2_NATIVE" not in evaluation_env


def test_terminal_gate_rejects_missing_or_failed_task(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"simulations": [_result("11", 1.0)]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal-state mismatch"):
        tau2._terminal_results(path, ["11", "19"], 1)
    path.write_text(json.dumps({"simulations": [
        _result("11", 1.0, termination="infrastructure_error")]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="infrastructure_error"):
        tau2._terminal_results(path, ["11"], 1)


def test_full_replay_source_requires_exact_official_task_coverage(tmp_path):
    path = tmp_path / "prefixes.jsonl"
    payload = {"messages": [{"role": "user", "content": "Help"}],
               "c2kv_measurement_session_id": "11"}
    row = {"event_type": "recorded_prefix", "source_arm": "full",
           "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tau2._validate_recorded_prefixes(path, ["11"])
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        tau2._validate_recorded_prefixes(path, ["11", "19"])
    del payload["c2kv_measurement_session_id"]
    row["canonical_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="official task identity"):
        tau2._validate_recorded_prefixes(path, ["11"])


def test_harness_events_reject_restarted_or_failed_official_trial(tmp_path):
    path = tmp_path / "harness_events.jsonl"
    rows = [
        {"event_type": "episode_start", "episode_id": "11"},
        {"event_type": "decision", "episode_id": "11"},
        {"event_type": "episode_end", "episode_id": "11", "status": "ok"},
    ]
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    tau2._validate_harness_events(path, ["11"])
    with pytest.raises(RuntimeError, match="retry or missing episode"):
        tau2._validate_harness_events(path, ["11", "19"])
    path.write_text("\n".join(map(json.dumps, rows + rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="retry or missing episode"):
        tau2._validate_harness_events(path, ["11"])
    rows[-1]["status"] = "error"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ended in error"):
        tau2._validate_harness_events(path, ["11"])
    rows[-1]["status"] = "ok"
    rows[1]["error"] = "upstream request failed"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ended in error"):
        tau2._validate_harness_events(path, ["11"])


def test_typed_method_guard_keeps_completed_tau2_scores_and_failed_task(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    measurement = tmp_path / "measurement" / "harness_events.jsonl"
    for task, status in (("11", "ok"), ("19", "error")):
        append_jsonl(measurement, {"event_type": "episode_start", "episode_id": task})
        append_jsonl(measurement, {"event_type": "decision", "episode_id": task,
                                    "error": "APIError: budget" if task == "19" else None})
        append_jsonl(measurement, {"event_type": "episode_end", "episode_id": task,
                                    "status": status})
    append_jsonl(tmp_path / "logs" / "proxy_acon_34100.jsonl", {
        "conv_id": hashlib.sha256(b'["measurement_session","19"]').hexdigest(),
        "status": "acon_history_budget_exceeded"})
    rows = [_result("11", 1.0), _result("19", termination="infrastructure_error")]
    result = tmp_path / "results.json"
    result.write_text(json.dumps({"simulations": rows}), encoding="utf-8")
    parsed = tau2._terminal_results(result, ["11", "19"], 1,
                                    require_reward=False, inspect_failures=True)
    failures = tau2._declared_task_failures(tmp_path, parsed, 1)
    assert failures == {"19": "acon_history_budget_exceeded"}
    tau2._terminal_results(result, ["11", "19"], 1, task_failures=failures)
    tau2._validate_harness_events(measurement, ["11", "19"], task_failures=failures)
    scores = tau2.collect(result, tools=[], task_failures=failures)
    assert scores["n"] == 2 and scores["semantic_score"] == 0.5
    assert scores["task_rows"][1]["official_reward"] is None
    assert scores["task_rows"][1]["task_failure_kind"] == "acon_history_budget_exceeded"


def test_untyped_tau2_502_remains_incomplete(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "19", "error": "HTTP 502 connection refused"})
    append_jsonl(tmp_path / "logs" / "proxy_acon_34100.jsonl", {
        "conv_id": hashlib.sha256(b'["measurement_session","19"]').hexdigest(),
        "status": "upstream_error"})
    rows = [_result("19", termination="infrastructure_error")]
    assert tau2._declared_task_failures(tmp_path, rows, 1) == {}
    result = tmp_path / "results.json"
    result.write_text(json.dumps({"simulations": rows}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="infrastructure_error"):
        tau2._terminal_results(result, ["19"], 1)


@pytest.mark.parametrize("code", ["decision_cap_reached", "generation_cap_reached"])
def test_native_decision_cap_requires_structured_429_error(tmp_path, code):
    from benchmarks.measurement.telemetry import append_jsonl

    event = tmp_path / "measurement" / "harness_events.jsonl"
    append_jsonl(event, {"event_type": "decision", "episode_id": "19",
                         "error": {"exception_type": "APIStatusError", "message": "failed",
                                   "status_code": 429, "api_error_code": code}})
    rows = [_result("19", termination="infrastructure_error")]
    assert tau2._declared_task_failures(tmp_path, rows, 1) == {
        "19": code}
    assert tau2._declared_task_failures(tmp_path, rows, 2) == {}


@pytest.mark.parametrize("error", [
    "APIStatusError: decision_cap_reached",
    {"exception_type": "APIStatusError", "message": "decision_cap_reached",
     "status_code": 429, "api_error_code": None},
    {"exception_type": "APIStatusError", "message": "failed",
     "status_code": 500, "api_error_code": "decision_cap_reached"},
    {"exception_type": "APIStatusError", "message": "failed",
     "status_code": 429, "api_error_code": "untrusted_cap"},
])
def test_native_decision_cap_rejects_text_or_incomplete_client_error(tmp_path, error):
    from benchmarks.measurement.telemetry import append_jsonl

    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "19", "error": error})
    rows = [_result("19", termination="infrastructure_error")]
    assert tau2._declared_task_failures(tmp_path, rows, 1) == {}


@pytest.mark.parametrize("change", [None, "wrong_type", "wrong_status", "unknown_code",
                                     "wrong_task", "no_native_server"])
def test_native_agent_context_overflow_is_a_narrow_task_failure(tmp_path, change):
    from benchmarks.measurement.telemetry import append_jsonl

    error = {"exception_type": "ContextWindowExceededError",
             "message": "bounded fixture", "status_code": 400,
             "api_error_code": None}
    if change == "wrong_type":
        error["exception_type"] = "InternalServerError"
    elif change == "wrong_status":
        error["status_code"] = 500
    elif change == "unknown_code":
        error["api_error_code"] = "unknown_context_error"
    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "44", "error": error})
    server = tmp_path / "server"
    server.mkdir()
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": "tau2", "allowed_task_ids": ["other" if change == "wrong_task" else "44"],
        "run_id": "run-fixture",
    }), encoding="utf-8")
    rows = [_result("44", termination="infrastructure_error")]
    native_server = None if change == "no_native_server" else server
    expected = {} if change is not None else {"44": "context_overflow"}
    assert tau2._declared_task_failures(
        tmp_path, rows, 1, native_server_dir=native_server) == expected


def test_code_stripped_client_error_defers_to_task_bound_server_evidence(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_tau2_hook_status_alias", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    error = RuntimeError("The finite event-native decision cap is exhausted")
    error.status_code = 429
    error.code = "429"
    details = hook._exception_details(error)
    assert details == {
        "exception_type": "RuntimeError",
        "message": "The finite event-native decision cap is exhausted",
        "status_code": 429,
        "api_error_code": None,
    }
    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "19",
        "error": details})
    server = tmp_path / "server"
    server.mkdir()
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1",
        "status": "ready",
        "benchmark": "tau2",
        "allowed_task_ids": ["19"],
        "run_id": "run-fixture",
        "max_decisions": 96,
    }), encoding="utf-8")
    (server / "budget_rejections.jsonl").write_text(json.dumps({
        "schema": "a-event-native-budget-rejection-v1",
        "run_id": "run-fixture",
        "task_id": "19",
        "session_id": "tau2/19/attempt-0",
        "status_code": 429,
        "code": "decision_cap_reached",
        "max_decisions": 96,
        "decisions_reserved": 96,
    }) + "\n", encoding="utf-8")
    rows = [_result("19", termination="infrastructure_error")]
    assert tau2._declared_task_failures(
        tmp_path, rows, 1, native_server_dir=server) == {
            "19": "decision_cap_reached"}

    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "19", "error": {
            "exception_type": "APIStatusError", "message": "failed",
            "status_code": 500, "api_error_code": None}})
    assert tau2._declared_task_failures(
        tmp_path, rows, 1, native_server_dir=server) == {}

    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "19", "error": {
            "exception_type": "APIStatusError", "message": "failed",
            "status_code": 429, "api_error_code": "untrusted_cap"}})
    assert tau2._declared_task_failures(
        tmp_path, rows, 1, native_server_dir=server) == {}


def test_generation_cap_task_is_counted_without_masking_official_reward(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    events = tmp_path / "measurement" / "harness_events.jsonl"
    for task in ("5", "6"):
        append_jsonl(events, {"event_type": "episode_start", "episode_id": task})
        append_jsonl(events, {"event_type": "decision", "episode_id": task,
                             "error": ({"exception_type": "APIStatusError", "message": "failed",
                                        "status_code": 429,
                                        "api_error_code": "generation_cap_reached"}
                                       if task == "5" else None)})
        append_jsonl(events, {"event_type": "episode_end", "episode_id": task,
                             "status": "error" if task == "5" else "ok"})
    path = tmp_path / "results.json"
    rows = [_result("5", termination="infrastructure_error"), _result("6", 1.0)]
    path.write_text(json.dumps({"simulations": rows}), encoding="utf-8")
    failures = tau2._declared_task_failures(tmp_path, rows, 1)
    tau2._terminal_results(path, ["5", "6"], 1, task_failures=failures)
    tau2._validate_harness_events(events, ["5", "6"], task_failures=failures)
    collected = tau2.collect(path, tools=[], task_failures=failures)
    assert collected["n"] == 2
    assert [row["semantic_score"] for row in collected["task_rows"]] == [0.0, 1.0]
    assert collected["task_rows"][0]["official_reward"] is None
    assert collected["task_rows"][0]["task_failure_kind"] == "generation_cap_reached"


def test_legacy_untyped_generation_cap_is_not_reclassified(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    append_jsonl(tmp_path / "measurement" / "harness_events.jsonl", {
        "event_type": "decision", "episode_id": "5",
        "error": "RuntimeError: Finite generation-call cap exhausted before submission"})
    assert tau2._declared_task_failures(
        tmp_path, [_result("5", termination="infrastructure_error")], 1) == {}


def test_hook_extracts_only_bounded_structured_exception_status_and_code():
    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_tau2_hook_errors", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    response = types.SimpleNamespace(
        status_code=429,
        headers={"authorization": "must-not-be-recorded"},
        request={"body": "must-not-be-recorded"},
        json=lambda: {"error": {"code": "decision_cap_reached",
                                "message": "safe message"}})
    error = RuntimeError("outer failure")
    error.response = response
    error.code = "429"
    details = hook._exception_details(error)
    assert details == {"exception_type": "RuntimeError", "message": "outer failure",
                       "status_code": 429, "api_error_code": "decision_cap_reached"}
    assert "headers" not in details and "body" not in details and "request" not in details

    stripped = RuntimeError("The finite event-native decision cap is exhausted")
    stripped.status_code = 429
    assert hook._exception_details(stripped)["api_error_code"] is None

    unknown = RuntimeError("unknown")
    unknown.status_code = 429
    unknown.code = "untrusted_cap"
    assert hook._exception_details(unknown)["api_error_code"] == "untrusted_cap"

    cause = RuntimeError("inner")
    cause.status_code = 429
    cause.body = {"error": {"code": "generation_cap_reached"}}
    outer = RuntimeError("outer")
    outer.__cause__ = cause
    assert hook._exception_details(outer) == {
        "exception_type": "RuntimeError", "message": "outer",
        "status_code": 429, "api_error_code": "generation_cap_reached"}


def test_hook_records_code_stripped_litellm_error_as_structured_telemetry(
        monkeypatch, tmp_path):
    events = tmp_path / "events.jsonl"

    class LiteLLMError(RuntimeError):
        status_code = 429

    class Environment:
        def get_response(self, message):
            return message

    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.completion = lambda *_args, **_kwargs: None
    llm_agent = types.ModuleType("tau2.agent.llm_agent")
    user_simulator = types.ModuleType("tau2.user.user_simulator")
    user_simulator.generate = lambda *_args, **_kwargs: None
    calls = []

    def fail_generate(**_kwargs):
        calls.append(1)
        raise LiteLLMError("The finite event-native decision cap is exhausted")

    llm_agent.generate = fail_generate
    batch = types.ModuleType("tau2.runner.batch")

    def run_single_task(_config, _task, **_kwargs):
        llm_agent.generate(call_name="agent_response", model="openai/served")

    batch.run_single_task = run_single_task
    modules = {
        "tau2": types.ModuleType("tau2"),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": llm_agent,
        "tau2.environment": types.ModuleType("tau2.environment"),
        "tau2.environment.environment": types.ModuleType("tau2.environment.environment"),
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.batch": batch,
        "tau2.user": types.ModuleType("tau2.user"),
        "tau2.user.user_simulator": user_simulator,
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm_utils,
    }
    modules["tau2.environment.environment"].Environment = Environment
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("C2KV_TAU2_TELEMETRY_PATH", str(events))
    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_tau2_hook_failure", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.install()
    with pytest.raises(LiteLLMError):
        batch.run_single_task(None, types.SimpleNamespace(id="9"))
    assert len(calls) == 1
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    decision = next(row for row in rows if row["event_type"] == "decision")
    assert decision["error"] == {
        "exception_type": "LiteLLMError",
        "message": "The finite event-native decision cap is exhausted",
        "status_code": 429,
        "api_error_code": None,
    }


def _run_user_retry_fixture(monkeypatch, tmp_path, generate):
    events = tmp_path / "events.jsonl"

    class Environment:
        def get_response(self, message):
            return message

    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.completion = lambda *_args, **_kwargs: None
    llm_agent = types.ModuleType("tau2.agent.llm_agent")
    llm_agent.generate = lambda *_args, **_kwargs: None
    user_simulator = types.ModuleType("tau2.user.user_simulator")
    user_simulator.generate = generate
    batch = types.ModuleType("tau2.runner.batch")

    def run_single_task(_config, _task, **_kwargs):
        return user_simulator.generate(
            call_name="user_simulator_response", model="openai/user")

    batch.run_single_task = run_single_task
    modules = {
        "tau2": types.ModuleType("tau2"),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": llm_agent,
        "tau2.environment": types.ModuleType("tau2.environment"),
        "tau2.environment.environment": types.ModuleType("tau2.environment.environment"),
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.batch": batch,
        "tau2.user": types.ModuleType("tau2.user"),
        "tau2.user.user_simulator": user_simulator,
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm_utils,
    }
    modules["tau2.environment.environment"].Environment = Environment
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("C2KV_TAU2_TELEMETRY_PATH", str(events))
    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location(
        "fixture_tau2_hook_user_retry_" + tmp_path.name, hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.install()
    return batch, events


def _transport_error(*, status_code=500):
    protocol_type = type(
        "RemoteProtocolError", (RuntimeError,), {"__module__": "httpx"})
    connection_type = type("APIConnectionError", (RuntimeError,), {"__module__": "openai"})
    connection = connection_type("Connection error")
    connection.__cause__ = protocol_type("Server disconnected without a response")
    outer_type = type(
        "InternalServerError", (RuntimeError,), {"__module__": "litellm.exceptions"})
    outer = outer_type("litellm transport failure")
    outer.__cause__ = connection
    if status_code is not None:
        outer.status_code = status_code
    return outer


def test_user_simulator_retries_one_transient_transport_failure_and_logs_it(
        monkeypatch, tmp_path):
    calls = []

    def generate(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _transport_error()
        return "user response"

    batch, events = _run_user_retry_fixture(monkeypatch, tmp_path, generate)
    assert batch.run_single_task(None, types.SimpleNamespace(id="25")) == "user response"
    assert len(calls) == 2
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    retry = [row for row in rows if row["event_type"] == "user_simulator_retry"]
    assert len(retry) == 1
    assert retry[0]["episode_id"] == "25"
    assert retry[0]["retry_index"] == retry[0]["retry_limit"] == 1
    assert retry[0]["transport_cause"] == {
        "module": "openai", "exception_type": "APIConnectionError"}
    assert retry[0]["error"]["message"] == "litellm transport failure"
    assert rows[-1]["event_type"] == "episode_end" and rows[-1]["status"] == "ok"


def test_user_simulator_second_transport_failure_is_not_retried(
        monkeypatch, tmp_path):
    calls = []

    def generate(**_kwargs):
        calls.append(1)
        raise _transport_error()

    batch, events = _run_user_retry_fixture(monkeypatch, tmp_path, generate)
    with pytest.raises(RuntimeError, match="litellm transport failure"):
        batch.run_single_task(None, types.SimpleNamespace(id="25"))
    assert len(calls) == 2
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert sum(row["event_type"] == "user_simulator_retry" for row in rows) == 1
    assert rows[-1]["event_type"] == "episode_end"
    assert rows[-1]["status"] == "error"
    assert "litellm transport failure" in rows[-1]["error"]


def test_user_simulator_cancellation_with_transport_cause_is_not_retried(
        monkeypatch, tmp_path):
    calls = []
    cancellation = KeyboardInterrupt("cancelled")
    cancellation.__cause__ = _transport_error()

    def generate(**_kwargs):
        calls.append(1)
        raise cancellation

    batch, events = _run_user_retry_fixture(monkeypatch, tmp_path, generate)
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        batch.run_single_task(None, types.SimpleNamespace(id="25"))
    assert len(calls) == 1
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert all(row["event_type"] != "user_simulator_retry" for row in rows)


@pytest.mark.parametrize("error", [
    RuntimeError("model failure"),
    _transport_error(status_code=400),
    _transport_error(status_code=429),
])
def test_user_simulator_does_not_retry_nontransport_or_http_failures(
        monkeypatch, tmp_path, error):
    calls = []

    def generate(**_kwargs):
        calls.append(1)
        raise error

    batch, events = _run_user_retry_fixture(monkeypatch, tmp_path, generate)
    with pytest.raises(type(error)):
        batch.run_single_task(None, types.SimpleNamespace(id="25"))
    assert len(calls) == 1
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert all(row["event_type"] != "user_simulator_retry" for row in rows)


@pytest.mark.parametrize("native", [False, True])
def test_hook_adds_task_identity_only_to_proxy_agent_and_records_episode(
        monkeypatch, tmp_path, native):
    calls = []
    events = tmp_path / "events.jsonl"

    class Environment:
        def get_response(self, message):
            return types.SimpleNamespace(
                error=False, model_dump=lambda **_kwargs: {"content": "ok"})

    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.completion = lambda *_args, **kwargs: calls.append(kwargs) or types.SimpleNamespace(id="req-1")
    llm_agent = types.ModuleType("tau2.agent.llm_agent")
    llm_agent.generate = lambda **kwargs: llm_utils.completion(**kwargs) or None
    user_simulator = types.ModuleType("tau2.user.user_simulator")
    user_simulator.generate = lambda **kwargs: llm_utils.completion(**kwargs) or None
    batch = types.ModuleType("tau2.runner.batch")

    def run_single_task(_config, task, **_kwargs):
        llm_agent.generate(call_name="agent_response", model="openai/served")
        llm_utils.completion(model="openai/user")
        environment = Environment()
        message = types.SimpleNamespace(
            requestor="assistant", model_dump=lambda **_kwargs: {"name": "lookup"})
        environment.get_response(message)
        environment.get_response(types.SimpleNamespace(requestor="user"))
        return types.SimpleNamespace(task_id=task.id)

    batch.run_single_task = run_single_task
    modules = {
        "tau2": types.ModuleType("tau2"),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": llm_agent,
        "tau2.environment": types.ModuleType("tau2.environment"),
        "tau2.environment.environment": types.ModuleType("tau2.environment.environment"),
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.batch": batch,
        "tau2.user": types.ModuleType("tau2.user"),
        "tau2.user.user_simulator": user_simulator,
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm_utils,
    }
    modules["tau2.environment.environment"].Environment = Environment
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("C2KV_TAU2_TELEMETRY_PATH", str(events))
    monkeypatch.setenv("C2KV_TAU2_NATIVE", "1" if native else "0")
    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_tau2_hook", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.install()
    assert batch.run_single_task(None, types.SimpleNamespace(id="11")).task_id == "11"
    if native:
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == {"c2kv_measurement_session_id": "11"}
    assert "extra_body" not in calls[1]
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == [
        "episode_start", "decision", "tool_action", "episode_end"]
    assert all(row["episode_id"] == "11" for row in rows)
