"""ACE Full-prefix recording observes the official action without changing the actor wire."""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import benchmarks
import pytest
from benchmarks import acebench_cli as hook


BENCHMARKS = Path(__file__).resolve().parent
RUNTIME = BENCHMARKS.parents[0] / "experiments" / "history_system" / "runtime"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
if str(RUNTIME / "python") not in sys.path:
    sys.path.insert(0, str(RUNTIME / "python"))
if str(RUNTIME / "benchmarks") not in benchmarks.__path__:
    benchmarks.__path__.append(str(RUNTIME / "benchmarks"))

from adapters import acebench_adapter  # noqa: E402
from arms import get_arm  # noqa: E402
from backends.sglang import SglangBackend  # noqa: E402
from benchmarks.memory_runtime.acebench_source import build_ace_event_store  # noqa: E402
from benchmarks.paper.native_extra import replay_payload  # noqa: E402
import proxy  # noqa: E402


def _official_scene(monkeypatch, tmp_path, *, recording):
    monkeypatch.delenv("C2KV_ACE_NATIVE", raising=False)
    if recording:
        monkeypatch.setenv("C2KV_ACE_RECORD_SOURCE", "1")
    else:
        monkeypatch.delenv("C2KV_ACE_RECORD_SOURCE", raising=False)
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent/v1")
    monkeypatch.setattr(hook.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    calls, observed = [], {"decode": 0, "execute": 0}

    def decode(message):
        observed["decode"] += 1
        assert message == "[Wifi()]"
        return ["Wifi()"]

    def execute(decoded):
        observed["execute"] += 1
        assert decoded == ["Wifi()"]
        return (["{}"], {})

    decoding, executing = hook.decode_wrapper(decode), hook.executor_wrapper(execute)

    @hook.execution_wrapper
    def official_execute(_executor, history):
        decoded = decoding(history[-1]["message"])
        values, _ = executing(decoded)
        return {"sender": "execution", "recipient": "agent",
                "message": [json.loads(value) for value in values]}, {}

    def create(_resource, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(c2kv_proxy={"request_id": f"decision-{len(calls)}"})

    request = hook.request_wrapper(create)
    resource = SimpleNamespace(_client=SimpleNamespace(base_url="http://agent/v1"))
    initial = [{"role": "system", "content": "system"},
               {"role": "user", "content": "wifi"}]
    history = [{"sender": "user", "message": "wifi"},
               {"sender": "agent", "message": "[Wifi()]"}]

    @hook.task_wrapper
    def episode(test_id):
        request(resource, messages=initial, model="m", temperature=0.001,
                top_p=1, max_tokens=1000)
        execution, _ = official_execute(object(), history)
        visible = [*initial, {"role": "assistant", "content": history[-1]["message"]},
                   {"role": "tool", "content": json.dumps(execution["message"]),
                    "tool_call_id": "acebench-execution-2"}]
        request(resource, messages=visible, model="m", temperature=0.001,
                top_p=1, max_tokens=1000)
        return visible

    @hook.inference_wrapper
    def inference(id):
        return episode(test_id=id.rsplit("_", 1)[-1])

    visible = inference("agent_multi_turn_1")
    return calls, visible, observed


def test_full_recording_observes_once_and_preserves_scene_request(tmp_path, monkeypatch):
    off, _, off_counts = _official_scene(monkeypatch, tmp_path / "off", recording=False)
    on, visible, on_counts = _official_scene(monkeypatch, tmp_path / "on", recording=True)
    assert off_counts == on_counts == {"decode": 1, "execute": 1}
    assert len(off) == len(on) == 2
    for raw, recorded in zip(off, on):
        assert (recorded["extra_body"]["c2kv_measurement_session_id"]
                == raw["extra_body"]["c2kv_measurement_session_id"]
                == "acebench:1:fixed")
        clean = copy.deepcopy(recorded)
        clean["extra_body"].pop("c2kv_ace_source")
        clean["extra_body"].pop("c2kv_ace_official_task_id")
        assert clean == raw
        assert recorded["extra_body"]["c2kv_ace_official_task_id"] == "agent_multi_turn_1"
    assert off[1]["messages"] == on[1]["messages"] == visible
    source = on[1]["extra_body"]["c2kv_ace_source"]
    assert source["version"] == "acebench-text-actions-v1"
    assert len(source["receipts"]) == 1
    assert source["receipts"][0]["decoded_calls"] == ["Wifi()"]
    store = build_ace_event_store("acebench/agent_multi_turn_1/attempt-0", visible, source)
    assert [(event.kind, event.complete) for event in store.events][-1] == ("tool_event", True)


def test_recorded_full_prefix_keeps_receipt_and_strips_model_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "ARM", get_arm("full"))
    monkeypatch.setattr(proxy, "BENCHMARK", "acebench")
    monkeypatch.setattr(proxy, "TOOL_MEMORY", None)
    monkeypatch.setattr(proxy, "QUERY_PROJECTION", None)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")
    monkeypatch.setattr(proxy, "TELEMETRY_LOG_PATH", "")
    monkeypatch.setattr(proxy.STATE, "recover", None)
    monkeypatch.setattr(proxy.STATE, "reference_log_path", "")
    monkeypatch.setattr(proxy, "_activate_measurement_session", lambda _session: None)
    sent = []

    def post(path, wire, timeout, retries=2):
        sent.append(copy.deepcopy(wire))
        return {"choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 12}}

    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    monkeypatch.setattr(proxy, "_post_json", post)
    off_calls, _, _ = _official_scene(monkeypatch, tmp_path / "off", recording=False)
    on_calls, visible, _ = _official_scene(monkeypatch, tmp_path / "on", recording=True)
    base = dict(off_calls[1], **off_calls[1]["extra_body"])
    base.pop("extra_body")
    recorded = dict(on_calls[1], **on_calls[1]["extra_body"])
    recorded.pop("extra_body")
    source = recorded["c2kv_ace_source"]

    def drive(payload, prefix):
        monkeypatch.setattr(proxy, "PREFIX_LOG_PATH", str(prefix) if prefix else "")
        raw = json.dumps(payload).encode()
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        response = {}
        handler._send_json = lambda code, obj: response.update(code=code, obj=obj)
        handler.do_POST()
        assert response["code"] == 200, response

    drive(base, None)
    drive(recorded, tmp_path / "prefixes.jsonl")
    assert len(sent) == 2 and sent[0] == sent[1]
    assert "c2kv_ace_source" not in sent[1]
    assert "c2kv_ace_official_task_id" not in sent[1]
    row = json.loads((tmp_path / "prefixes.jsonl").read_text(encoding="utf-8"))
    assert row["ace_official_task_id"] == "agent_multi_turn_1"
    assert row["replay_payload"]["c2kv_ace_source"] == source
    assert "c2kv_ace_official_task_id" not in row["replay_payload"]
    replay = replay_payload(row["replay_payload"], row["ace_official_task_id"], 0)
    assert replay["c2kv_ace_source"] == source
    assert replay["c2kv_eval_context"]["task_id"] == "agent_multi_turn_1"
    store = build_ace_event_store("acebench/agent_multi_turn_1/attempt-0",
                                  replay["messages"], replay["c2kv_ace_source"])
    assert replay["messages"] == visible
    assert [(event.kind, event.complete) for event in store.events][-1] == ("tool_event", True)


def test_recording_flag_is_run_scoped(monkeypatch):
    monkeypatch.setenv("C2KV_ACE_RECORD_SOURCE", "1")
    off = acebench_adapter.harness_env("http://agent", "", "m")
    on = acebench_adapter.harness_env("http://agent", "", "m", record_source=True)
    assert "C2KV_ACE_RECORD_SOURCE" not in off
    assert on["C2KV_ACE_RECORD_SOURCE"] == "1"


def test_adapter_passes_record_prefixes_to_official_run(monkeypatch, tmp_path):
    observed = {}

    def run_acebench(*_args, **kwargs):
        observed.update(kwargs)
        return {"categories": ["agent_multi_turn"]}

    monkeypatch.setattr(acebench_adapter, "run_acebench", run_acebench)
    ctx = SimpleNamespace(base_url="http://agent", user_base_url="http://user",
                          out_dir=tmp_path, model="m",
                          opt=lambda key, default=None: {
                              "record_prefixes": str(tmp_path / "prefix.jsonl")
                          }.get(key, default))
    acebench_adapter.run(ctx)
    assert observed["record_prefixes"] == str(tmp_path / "prefix.jsonl")


def test_recording_preserves_official_error_and_nonstandard_return(tmp_path, monkeypatch):
    monkeypatch.setenv("C2KV_ACE_RECORD_SOURCE", "1")
    monkeypatch.delenv("C2KV_ACE_NATIVE", raising=False)
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(tmp_path / "events.jsonl"))
    failure = ValueError("official decoder failed")
    calls = {"decode": 0}

    def decode(_text):
        calls["decode"] += 1
        raise failure

    wrapped_decode = hook.decode_wrapper(decode)

    @hook.execution_wrapper
    def failed_action(_executor, history):
        return wrapped_decode(history[-1]["message"])

    @hook.task_wrapper
    def failed_episode(test_id):
        hook._local.request = "decision"
        return failed_action(object(), [{"sender": "agent", "message": "bad"}])

    with pytest.raises(ValueError) as caught:
        failed_episode("1")
    assert caught.value is failure
    assert calls["decode"] == 1

    unusual = ("not an execution message", {"unchanged": True})

    @hook.execution_wrapper
    def unusual_action(_executor, _history):
        return unusual

    @hook.task_wrapper
    def unusual_episode(test_id):
        hook._local.request = "decision"
        value = unusual_action(object(), [{"sender": "user", "message": "wifi"},
                                          {"sender": "agent", "message": "[Wifi()]"}])
        return value, copy.deepcopy(hook._local.receipts)

    value, receipts = unusual_episode("1")
    assert value is unusual
    assert receipts[0]["decode_status"] == "invalid"
    source = {"version": "acebench-text-actions-v1", "receipts": receipts}
    visible = [{"role": "system", "content": "system"},
               {"role": "user", "content": "wifi"},
               {"role": "assistant", "content": "[Wifi()]"},
               {"role": "tool", "content": "{}"}]
    with pytest.raises(ValueError, match="ACEBench"):
        build_ace_event_store("acebench/agent_multi_turn_1/attempt-0", visible, source)
