"""Raw Full tool-source capture is replay provenance, never actor prompt text."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest

from benchmarks.adapters import acebench_adapter, acon_adapter
from benchmarks.measurement.replay import replay_prefixes, validate_tool_source_capture
from benchmarks.measurement.telemetry import canonical_sha256

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import source_annotations  # noqa: E402
import toolmemory  # noqa: E402
import proxy  # noqa: E402
from arms import get_arm  # noqa: E402
from backends.sglang import SglangBackend  # noqa: E402
from test_toolmemory import FakeTokenizer  # noqa: E402


def _patched_ace_source_function():
    """Execute the actual function added by the private official-harness patch."""
    patch = (HERE / "acebench_patches" / "0002-visible-tool-spans.patch").read_text(
        encoding="utf-8")
    first_hunk = patch.split("diff --git a/model_inference/apimodel_inference.py", 1)[0]
    source = "\n".join(line[1:] for line in first_hunk.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    namespace = {}
    exec(source, namespace)
    return namespace["tool_context_kwargs"]


def test_ace_raw_recording_captures_exact_visible_functions_and_t0_can_plan(monkeypatch):
    tool_context_kwargs = _patched_ace_source_function()
    functions = [{"name": "Wifi", "description": "Read connectivity"}]
    system = "Use ACEBench action syntax.\nAPIs:\n" + json.dumps(functions)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Check wifi"}]
    monkeypatch.setenv("C2KV_TOOL_CONTEXT_ON", "0")
    monkeypatch.delenv("C2KV_ACE_RECORD_SOURCE", raising=False)
    assert tool_context_kwargs(messages, functions) == {}
    monkeypatch.setenv("C2KV_ACE_RECORD_SOURCE", "1")
    spans = tool_context_kwargs(messages, functions)["extra_body"]["c2kv_tool_spans_v1"]
    assert len(spans) == 1
    assert system[spans[0]["start"]:spans[0]["end"]] == json.dumps(functions[0])
    payload = {"messages": messages, "c2kv_tool_spans_v1": spans}
    validate_tool_source_capture(payload, "acebench_agent")
    plan = toolmemory.plan_visible_tool_memory(
        payload, toolmemory.parse_tool_memory_spec("t0:r8"), FakeTokenizer())
    assert plan is not None and plan.info["n_visible_source_spans"] == 1
    assert plan.messages[0]["content"] != system
    assert plan.messages[0]["content"].startswith("Use ACEBench action syntax.")
    assert tool_context_kwargs(messages, [])["extra_body"]["c2kv_tool_spans_v1"] == []
    assert tool_context_kwargs([{"role": "system", "content": "not rendered"}], functions)[
        "extra_body"]["c2kv_tool_spans_v1"] is None
    monkeypatch.setenv("C2KV_TOOL_CONTEXT_ON", "1")
    with pytest.raises(ValueError, match="absent"):
        tool_context_kwargs([{"role": "system", "content": "not rendered"}], functions)


def test_ace_recording_env_installs_private_source_patch_without_tool_memory(monkeypatch):
    monkeypatch.setenv(acebench_adapter.TOOL_CONTEXT_ENV, "0")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    off = acebench_adapter.harness_env("http://agent", "", "m")
    on = acebench_adapter.harness_env("http://agent", "", "m", record_source=True)
    assert "PYTHONPATH" not in off
    assert str(HERE) in on["PYTHONPATH"].split(os.pathsep)
    assert on["C2KV_ACE_RECORD_SOURCE"] == "1"


def test_ace_raw_full_recording_applies_private_patch(monkeypatch, tmp_path):
    monkeypatch.setenv(acebench_adapter.TOOL_CONTEXT_ENV, "0")
    source = tmp_path / "official"
    source.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    applied, environments = [], []
    monkeypatch.setattr(acebench_adapter, "load_category_map",
                        lambda _root: {"agent": ["agent_multi_turn"]})
    monkeypatch.setattr(acebench_adapter, "prepare_workdir", lambda *_args, **_kwargs: work)
    monkeypatch.setattr(acebench_adapter, "prepare_tool_span_harness",
                        lambda _work, harness: applied.append(harness) or harness)
    monkeypatch.setattr(acebench_adapter, "run_owned",
                        lambda _command, **kwargs: environments.append(kwargs["env"]))
    monkeypatch.setattr(acebench_adapter, "check_terminal", lambda *_args: None)
    monkeypatch.setattr(acebench_adapter, "prepare_score_dir", lambda *_args: None)
    monkeypatch.setattr(acebench_adapter, "collect",
                        lambda *_args: {"categories": ["agent_multi_turn"]})
    acebench_adapter.run_acebench(
        "http://agent", "http://user", tmp_path, acebench_dir=source,
        record_prefixes=str(tmp_path / "prefix.jsonl"))
    assert applied == [source]
    assert len(environments) == 2
    assert all(env["C2KV_ACE_RECORD_SOURCE"] == "1" for env in environments)
    assert all(env["C2KV_TOOL_CONTEXT_ON"] == "0" for env in environments)


def test_appworld_recording_observes_executed_docs_and_marks_empty(monkeypatch, tmp_path):
    calls = []

    class FakeCompletions:
        def create(self, *_args, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(model_extra={"c2kv_proxy": {"request_id": "r"}})

    class FakeWorld:
        def execute(self, _code):
            return "{'api_name': 'Wifi', 'parameters': []}"

    class FakeEnv:
        def __init__(self):
            self.world = FakeWorld()
            self.config = types.SimpleNamespace(max_interactions=3)
            self.experiment_name = "fixture"
            self.num_interactions = 0

        def reset(self, seed=None, task_id=None, **_kwargs):
            return "ready"

        def close(self):
            return None

        def _clean_code(self, code):
            return code.strip()

    class FakeAgent:
        def forward(self, prompt):
            return prompt

    def module(name, **attrs):
        value = types.ModuleType(name)
        for key, item in attrs.items():
            setattr(value, key, item)
        return value

    modules = {
        "productive_agents": module("productive_agents"),
        "productive_agents.agents": module("productive_agents.agents"),
        "productive_agents.agents.unified_agent": module(
            "productive_agents.agents.unified_agent", UnifiedAgent=FakeAgent),
        "productive_agents.env": module("productive_agents.env"),
        "productive_agents.env.appworld": module("productive_agents.env.appworld"),
        "productive_agents.env.appworld.env": module(
            "productive_agents.env.appworld.env", AppWorldEnv=FakeEnv),
        "openai": module("openai"),
        "openai.resources": module("openai.resources"),
        "openai.resources.chat": module("openai.resources.chat"),
        "openai.resources.chat.completions": module(
            "openai.resources.chat.completions", Completions=FakeCompletions),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("C2KV_APPWORLD_TELEMETRY_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("C2KV_APPWORLD_RECORD_SOURCE", "1")
    monkeypatch.setenv("C2KV_TOOL_CONTEXT_ON", "0")
    hook_path = HERE / "appworld_instrumentation" / "c2kv_appworld_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_appworld_record_source", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.install()

    env = FakeEnv()
    env.reset(task_id="task-7")
    initial = [{"role": "system", "content": "Python actions only"},
               {"role": "user", "content": "Find an API"}]
    FakeCompletions().create(messages=initial)
    assert calls[-1]["extra_body"]["c2kv_tool_spans_v1"] == []
    code = "print(apis.api_docs.show_api_doc('Wifi'))"
    output = env._execute_code(code)
    visible = [*initial, {"role": "assistant", "content": code},
               {"role": "user", "content": "Output:\n" + output + "\nNext code:"}]
    FakeCompletions().create(messages=visible)
    spans = calls[-1]["extra_body"]["c2kv_tool_spans_v1"]
    assert len(spans) == 1
    assert spans[0]["source"] == "appworld.api_docs.show_api_doc"
    assert visible[spans[0]["message_index"]]["content"][spans[0]["start"]:spans[0]["end"]] == output
    payload = {"messages": visible, "c2kv_tool_spans_v1": spans}
    validate_tool_source_capture(payload, "appworld")
    plan = toolmemory.plan_visible_tool_memory(
        payload, toolmemory.parse_tool_memory_spec("t0:r8"), FakeTokenizer())
    assert plan is not None and plan.info["n_visible_source_spans"] == 1
    env.close()


def test_appworld_recording_env_is_separate_from_tool_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("C2KV_APPWORLD_RECORD_SOURCE", "1")
    args = ("http://agent", tmp_path / "events", tmp_path / "tasks")
    off = acon_adapter.appworld_runner_env(*args)
    on = acon_adapter.appworld_runner_env(*args, record_source=True)
    assert "C2KV_APPWORLD_RECORD_SOURCE" not in off
    assert on["C2KV_APPWORLD_RECORD_SOURCE"] == "1"


def test_appworld_adapter_forwards_raw_full_recording(monkeypatch, tmp_path):
    observed = {}

    def run_appworld(*_args, **kwargs):
        observed.update(kwargs)
        return {"n": 1}

    monkeypatch.setattr(acon_adapter, "run_appworld", run_appworld)
    options = {"benchmark": "acon_appworld", "record_prefixes": str(tmp_path / "prefix.jsonl")}
    ctx = types.SimpleNamespace(base_url="http://agent", out_dir=tmp_path,
                                model="m", run_name="run", request_log=None,
                                options=options,
                                opt=lambda key, default=None: options.get(key, default))
    assert acon_adapter.run(ctx) == {"n": 1}
    assert observed["record_prefixes"] == str(tmp_path / "prefix.jsonl")


def test_appworld_raw_full_prefix_preserves_model_wire_and_t0_source(monkeypatch, tmp_path):
    output = "{'api_name': 'Wifi', 'parameters': []}"
    code = "print(apis.api_docs.show_api_doc('Wifi'))"
    messages = [{"role": "system", "content": "Python actions only"},
                {"role": "user", "content": "Find an API"},
                {"role": "assistant", "content": code},
                {"role": "user", "content": "Output:\n" + output + "\nNext code:"}]
    spans = source_annotations.appworld_doc_spans(
        messages, [("appworld.api_docs.show_api_doc", output)])
    base = {"model": "m", "messages": messages,
            "c2kv_measurement_session_id": "task-7"}
    recorded = dict(base, c2kv_tool_spans_v1=spans)
    sent = []

    def post(_path, wire, _timeout, retries=2):
        sent.append(copy.deepcopy(wire))
        return {"choices": [{"message": {"role": "assistant", "content": "done"},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 12}}

    monkeypatch.setattr(proxy, "ARM", get_arm("full"))
    monkeypatch.setattr(proxy, "BENCHMARK", "acon_appworld")
    monkeypatch.setattr(proxy, "TOOL_MEMORY", None)
    monkeypatch.setattr(proxy, "QUERY_PROJECTION", None)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")
    monkeypatch.setattr(proxy, "TELEMETRY_LOG_PATH", "")
    monkeypatch.setattr(proxy.STATE, "recover", None)
    monkeypatch.setattr(proxy.STATE, "reference_log_path", "")
    monkeypatch.setattr(proxy, "_activate_measurement_session", lambda _session: None)
    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    monkeypatch.setattr(proxy, "_post_json", post)

    def drive(payload, prefix_path):
        monkeypatch.setattr(proxy, "PREFIX_LOG_PATH", str(prefix_path) if prefix_path else "")
        raw = json.dumps(payload).encode()
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        response = {}
        handler._send_json = lambda code, obj: response.update(code=code, obj=obj)
        handler.do_POST()
        assert response["code"] == 200, response

    prefix_path = tmp_path / "appworld_prefix.jsonl"
    drive(base, None)
    drive(recorded, prefix_path)
    assert sent[0] == sent[1]
    assert "c2kv_tool_spans_v1" not in sent[1]
    row = json.loads(prefix_path.read_text(encoding="utf-8"))
    assert row["replay_payload"]["c2kv_tool_spans_v1"] == spans
    plan = toolmemory.plan_visible_tool_memory(
        row["replay_payload"], toolmemory.parse_tool_memory_spec("t0:r8"), FakeTokenizer())
    assert plan is not None and plan.info["n_visible_source_spans"] == 1


def test_old_inline_prefix_replay_fails_before_network_and_explicit_empty_is_valid(
        monkeypatch, tmp_path):
    prefixes = tmp_path / "prefixes.jsonl"
    output = tmp_path / "replayed.jsonl"
    payload = {"messages": [{"role": "user", "content": "task"}]}

    def write_prefix(source):
        prefixes.write_text(json.dumps({
            "event_type": "recorded_prefix", "source_arm": "full",
            "canonical_sha256": canonical_sha256(source),
            "replay_payload": source,
        }) + "\n", encoding="utf-8")

    class Opener:
        def open(self, *_args, **_kwargs):
            pytest.fail("legacy source must fail before actor request")

    monkeypatch.setattr("benchmarks.measurement.replay._OPENER", Opener())
    write_prefix(payload)
    with pytest.raises(ValueError, match="captured c2kv_tool_spans_v1"):
        replay_prefixes(prefixes, "http://agent", output,
                        source_run_id="full", target_run_id="t0",
                        tool_source_benchmark="acebench_agent")
    assert not output.exists()
    empty = dict(payload, c2kv_tool_spans_v1=[])
    validate_tool_source_capture(empty, "acebench_agent")
    validate_tool_source_capture(payload, "bfcl_base")
    write_prefix(dict(payload, c2kv_tool_spans_v1=None))
    with pytest.raises(ValueError, match="captured c2kv_tool_spans_v1"):
        replay_prefixes(prefixes, "http://agent", output,
                        source_run_id="full", target_run_id="t0",
                        tool_source_benchmark="appworld")
