"""A frozen --generation-timeout reaches the persistent proxy and the BFCL, ToolSandbox and tau2 agent clients.

CPU only: the runner's prepared command is parsed by run.py, run.py starts the
real proxy process, and a fake engine proves which deadline the proxy applies
to one persistent generation. The client legs check each agent deadline;
without a configured deadline ToolSandbox and tau2 keep their historical 600 s.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from benchmarks import run as bench_run
from benchmarks.model_identity import QWEN3_4B
from benchmarks.paper import runner

REFERENCE_ATTENTION_BFCL_LONG = (
    "bfcl_long_context__commitkv_r0p25", "bfcl_long_context__agentkv",
    "bfcl_long_context__commitkv_b768", "bfcl_long_context__agentkv_b768",
    "bfcl_long_context__history_kv_pyramidkv_persistent_r0p25",
)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """The same overlay prepared with and without an explicit deadline."""
    result = {}
    for name, extra in (("default", []), ("explicit", ["--generation-timeout", "1800"])):
        output = tmp_path_factory.mktemp(name) / "root"
        runner.main(["prepare", "--output", str(output), *extra,
                     "--history-kv-budget", "commitkv=768", "--history-kv-budget", "agentkv=768"])
        # Compare commands independently of where each root was prepared.
        text = (output / "commands.json").read_text(encoding="utf-8")
        text = text.replace(json.dumps(str(output))[1:-1], "ROOT")
        result[name] = {row["cell_id"]: row for row in json.loads(text)}
    return result


@pytest.mark.parametrize("cell_id", REFERENCE_ATTENTION_BFCL_LONG)
def test_prepared_reference_attention_commands_parse_to_the_frozen_deadline(plans, cell_id):
    explicit = plans["explicit"][cell_id]["command"]
    default = plans["default"][cell_id]["command"]
    assert bench_run.build_parser().parse_args(explicit[2:]).generation_timeout == 1800.0
    assert "--generation-timeout" not in default
    assert bench_run.build_parser().parse_args(default[2:]).generation_timeout == 600.0
    # the deadline is the only difference between the two prepared commands
    position = explicit.index("--generation-timeout")
    assert explicit[:position] + explicit[position + 2:] == default


def test_ordinary_cells_keep_their_commands(plans):
    for cell_id, row in plans["explicit"].items():
        if cell_id not in REFERENCE_ATTENTION_BFCL_LONG and "--generation-timeout" not in row["command"]:
            assert row["command"] == plans["default"][cell_id]["command"]


def test_bfcl_client_read_deadline_keeps_cleanup_headroom(plans, monkeypatch, tmp_path):
    bfcl_adapter = bench_run.ADAPTERS["bfcl"]
    args = bench_run.build_parser().parse_args(
        plans["explicit"]["bfcl_long_context__commitkv_b768"]["command"][2:])
    seen = {}
    monkeypatch.setattr(bfcl_adapter, "run_bfcl",
                        lambda *_args, **kwargs: seen.update(kwargs) or {"n": 0})
    monkeypatch.setattr(bfcl_adapter, "default_bfcl_dir", lambda: str(tmp_path))
    args.out = tmp_path / "cell"
    bfcl_adapter.run(bench_run.build_context(args, tmp_path / "proxy.jsonl"))
    assert seen["request_timeout"] == 1800.0
    timeout = bfcl_adapter.client_kwargs("http://proxy/v1", seen["request_timeout"])["timeout"]
    assert timeout.read == 1890.0 and timeout.connect == 8.0


def _toolsandbox_run_ts_kwargs(plans, plan, cell_id, monkeypatch, tmp_path):
    adapter = bench_run.ADAPTERS["toolsandbox"]
    args = bench_run.build_parser().parse_args(plans[plan][cell_id]["command"][2:])
    seen = {}
    monkeypatch.setattr(adapter, "run_ts", lambda *_args, **kwargs: seen.update(kwargs) or {"n": 0})
    args.out = tmp_path / "cell"
    adapter.run(bench_run.build_context(args, tmp_path / "proxy.jsonl"))
    return seen


@pytest.mark.parametrize("cell_id", ["toolsandbox__agentkv", "toolsandbox__agentkv_b768",
                                     "toolsandbox__history_kv_pyramidkv_persistent_r0p25"])
def test_toolsandbox_agent_client_outlives_the_frozen_deadline(plans, monkeypatch, tmp_path, cell_id):
    explicit = _toolsandbox_run_ts_kwargs(plans, "explicit", cell_id, monkeypatch, tmp_path)
    default = _toolsandbox_run_ts_kwargs(plans, "default", cell_id, monkeypatch, tmp_path)
    assert explicit.pop("agent_timeout") == 1890.0
    # without a frozen deadline run_ts receives exactly the historical call
    assert "agent_timeout" not in default and explicit == default


def test_toolsandbox_ordinary_cells_keep_the_historical_agent_call(plans, monkeypatch, tmp_path):
    for cell_id in ("toolsandbox__full", "toolsandbox__hiagent_full", "toolsandbox__agentfold"):
        assert "--generation-timeout" not in plans["explicit"][cell_id]["command"]
        assert "agent_timeout" not in _toolsandbox_run_ts_kwargs(
            plans, "explicit", cell_id, monkeypatch, tmp_path)


def _toolsandbox_cli_env(monkeypatch, tmp_path, **run_ts_kwargs):
    from types import SimpleNamespace

    adapter = bench_run.ADAPTERS["toolsandbox"]
    source, out, seen = tmp_path / "ToolSandbox", tmp_path / "out", {}
    source.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        (out / "scenario_manifest.json").write_text(
            json.dumps({"scenario_ids": ["one"], "expected": 1}), encoding="utf-8")
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter, "run_owned", fake_run)
    monkeypatch.setattr(adapter, "collect", lambda output: {"n": 1, "scenario_ids": ["one"]})
    adapter.run_ts("http://agent", out, scenarios=["one"], benchmark_dir=source, **run_ts_kwargs)
    return seen, json.loads((out / "toolsandbox_protocol.json").read_text(encoding="utf-8"))


def test_toolsandbox_run_ts_hands_the_agent_timeout_to_the_cli_only_when_set(monkeypatch, tmp_path):
    adapter = bench_run.ADAPTERS["toolsandbox"]
    # a stale outer value never reaches the default CLI
    monkeypatch.setenv(adapter.AGENT_TIMEOUT_ENV, "5.0")
    env, protocol = _toolsandbox_cli_env(monkeypatch, tmp_path / "default")
    assert adapter.AGENT_TIMEOUT_ENV not in env and "agent_timeout" not in protocol
    env, protocol = _toolsandbox_cli_env(monkeypatch, tmp_path / "set", agent_timeout=3690.0)
    assert env[adapter.AGENT_TIMEOUT_ENV] == "3690.0" and protocol["agent_timeout"] == 3690.0


def test_agent_client_timeout_rule():
    from adapters import generation_deadline as deadline

    assert bench_run.build_parser().get_default("generation_timeout") == deadline.DEFAULT_GENERATION_TIMEOUT
    for name in ("toolsandbox", "tau2"):
        assert bench_run.ADAPTERS[name].agent_client_timeout is deadline.agent_client_timeout
    assert deadline.agent_client_timeout(600.0) is None
    assert deadline.agent_client_timeout(3600) == 3690.0
    assert deadline.agent_client_timeout(120.0) == 210.0
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            deadline.agent_client_timeout(invalid)


def _tau2_run_tau2_kwargs(plans, plan, cell_id, monkeypatch, tmp_path):
    adapter = bench_run.ADAPTERS["tau2"]
    args = bench_run.build_parser().parse_args(plans[plan][cell_id]["command"][2:])
    seen = {}
    monkeypatch.setattr(adapter, "run_tau2", lambda *_args, **kwargs: seen.update(kwargs) or {"n": 0})
    args.out = tmp_path / "cell"
    adapter.run(bench_run.build_context(args, tmp_path / "proxy.jsonl"))
    return seen


@pytest.mark.parametrize("cell_id", ["tau2__agentkv", "tau2__agentkv_b768",
                                     "tau2__history_kv_pyramidkv_r25_persistent"])
def test_tau2_agent_litellm_timeout_outlives_the_frozen_deadline(plans, monkeypatch, tmp_path, cell_id):
    explicit = _tau2_run_tau2_kwargs(plans, "explicit", cell_id, monkeypatch, tmp_path)
    default = _tau2_run_tau2_kwargs(plans, "default", cell_id, monkeypatch, tmp_path)
    assert explicit.pop("agent_timeout") == 1890.0
    assert "agent_timeout" not in default and explicit == default


def test_tau2_ordinary_cells_keep_the_historical_agent_call(plans, monkeypatch, tmp_path):
    for cell_id in ("tau2__full", "tau2__hiagent_full", "tau2__agentfold"):
        assert "--generation-timeout" not in plans["explicit"][cell_id]["command"]
        assert "agent_timeout" not in _tau2_run_tau2_kwargs(
            plans, "explicit", cell_id, monkeypatch, tmp_path)


def _tau2_command_and_protocol(monkeypatch, tmp_path, **run_tau2_kwargs):
    tau2 = bench_run.ADAPTERS["tau2"]
    source, seen = tmp_path / "tau2", {}
    (source / "src" / "tau2").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tau2, "selected_task_ids", lambda *_args, **_kwargs: ["11"])
    monkeypatch.setattr(tau2, "run_owned", lambda command, **_kwargs: seen.setdefault("run", command))
    monkeypatch.setattr(tau2, "score_simulations", lambda *_args, **_kwargs: {"n": 1})
    tau2.run_tau2("http://agent", "http://raw", tmp_path / "out", tau2_dir=source,
                  python="/venv/python", run_name="run_11", task_ids=["11"], **run_tau2_kwargs)
    protocol = json.loads((tmp_path / "out" / "tau2_protocol.json").read_text(encoding="utf-8"))
    return seen["run"], protocol


def test_tau2_agent_timeout_changes_only_the_agent_llm_args(monkeypatch, tmp_path):
    default, default_protocol = _tau2_command_and_protocol(monkeypatch, tmp_path)
    agent = default.index("--agent-llm-args") + 1
    # the historical agent arguments, byte for byte
    assert default[agent] == ('{"api_base": "http://agent/v1", "api_key": "EMPTY", '
                              '"temperature": 0.0, "max_tokens": 4096, "num_retries": 0}')
    assert "agent_timeout" not in default_protocol
    configured, protocol = _tau2_command_and_protocol(
        monkeypatch, tmp_path, agent_timeout=3690.0)
    assert json.loads(configured[agent]) == {**json.loads(default[agent]), "timeout": 3690.0}
    # the user simulator's LiteLLM arguments and every other token are unchanged
    assert configured[:agent] + configured[agent + 1:] == default[:agent] + default[agent + 1:]
    assert protocol.pop("agent_timeout") == 3690.0
    assert protocol == {**default_protocol, "command": configured}


def test_toolsandbox_cli_roles_keep_600_s_unless_the_agent_timeout_is_set(monkeypatch):
    openai = pytest.importorskip("openai")
    bench_run.ADAPTERS["toolsandbox"]  # puts benchmarks/ on sys.path
    import toolsandbox_cli as cli

    url = "http://127.0.0.1:19876/v1"
    assert cli.agent_timeout_from_env({}) is None
    assert cli.role_client_kwargs(url, agent=True) == {
        "api_key": "EMPTY", "base_url": url, "timeout": 600.0, "max_retries": 0}
    assert cli.role_client_kwargs(url, agent=False) == {
        "api_key": "EMPTY", "base_url": url, "timeout": 600.0}
    timeout = cli.agent_timeout_from_env({cli.AGENT_TIMEOUT_ENV: "3690.0"})
    assert cli.role_client_kwargs(url, agent=False, agent_timeout=timeout)["timeout"] == 600.0
    agent = openai.OpenAI(**cli.role_client_kwargs(url, agent=True, agent_timeout=timeout))
    assert agent.timeout == 3690.0 and agent.max_retries == 0
    for invalid in ("0", "-5", "inf", "nan"):
        with pytest.raises(ValueError):
            cli.agent_timeout_from_env({cli.AGENT_TIMEOUT_ENV: invalid})


def test_toolsandbox_rapidapi_requests_keep_their_30_s_default(monkeypatch, tmp_path):
    from types import SimpleNamespace

    bench_run.ADAPTERS["toolsandbox"]
    import toolsandbox_cli as cli

    monkeypatch.setenv(cli.AGENT_TIMEOUT_ENV, "3690.0")
    observed = {}
    tools = SimpleNamespace(requests=SimpleNamespace(
        get=lambda *args, **kwargs: observed.update(kwargs) or SimpleNamespace(status_code=200)))
    cli.install_rapidapi_http_status(tools, tmp_path / "rapidapi_http_status.jsonl")
    tools.requests.get(url="https://example.rapidapi.com/x")
    assert observed["timeout"] == 30


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _SlowEngine(BaseHTTPRequestHandler):
    delay = 0.0
    aborts = []

    def log_message(self, *_args):
        return

    def _reply(self, value, status=200):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._reply({"model_type": "qwen3", "model_dimensions": QWEN3_4B,
                     "model_path": "/qwen3-4b-timeout-fixture"})

    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path.startswith("/flush_cache"):
            self._reply("Cache flushed. fake")
        elif self.path == "/open_session":
            self._reply(payload["session_id"])
        elif self.path == "/abort_request":
            self.__class__.aborts.append(payload)
            self._reply({"rid": payload["rid"], "session_id": payload["session_id"],
                         "request_status": "aborted", "session_status": "closed"})
        else:
            time.sleep(self.__class__.delay)
            self._reply({"id": "chat-1", "object": "chat.completion",
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": "done"}}],
                         "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                         "metadata": {"persistent_history_session": {
                             "continuation_mode": "exact_generated_prefix",
                             "generated_text": "done"}}})


@pytest.mark.parametrize(("deadline", "delay", "timed_out"), [(1.0, 3.0, True), (6.0, 1.0, False)])
def test_run_py_proxy_applies_the_deadline_to_a_persistent_generation(tmp_path, deadline, delay, timed_out):
    _SlowEngine.delay, _SlowEngine.aborts = delay, []
    engine = ThreadingHTTPServer(("127.0.0.1", _free_port()), _SlowEngine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    port = _free_port()
    process, log = bench_run.start_proxy(
        f"http://127.0.0.1:{engine.server_port}", "commitkv", port, tmp_path,
        benchmark="bfcl", doc_packing="message", generation_timeout=deadline)
    try:
        request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                           "c2kv_measurement_session_id": "task-1"}).encode("utf-8"))
        started = time.monotonic()
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        elapsed = time.monotonic() - started
    finally:
        bench_run._stop_process(process)
        engine.shutdown()
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if timed_out:
        assert status == 502 and deadline <= elapsed < delay
        assert rows[-1]["error"] == "upstream 0: timed out"
        assert len(_SlowEngine.aborts) == 1 and _SlowEngine.aborts[0]["close_session"] is True
    else:
        assert status == 200 and elapsed >= delay
        assert rows[-1]["status"] == "ok" and _SlowEngine.aborts == []
