"""Focused CPU contracts for the task170 native action-envelope control."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks import proxy, textarms
from benchmarks.arms import get_arm
from benchmarks.backends.event_native_hiagent import EventNativeHiAgentBackend
from benchmarks.memory_runtime import event_native_hiagent_bfcl as full_worker
from benchmarks.memory_runtime import event_native_hiagent_envelope_bfcl as control_worker


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "System contract."},
        {"role": "user", "content": "Find and update one record."},
        {
            "role": "assistant",
            "content": "Subgoal: locate the record",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "search",
            "content": "one synthetic result",
        },
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-2",
            "type": "function",
            "function": {"name": "inspect", "arguments": '{"id":1}'},
        }]},
        {"role": "tool", "tool_call_id": "call-2", "content": "synthetic value"},
    ]


def _sampling(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema": "a-native-hiagent-policy-sampling-v1",
        "temperature": 0.0,
        "seed": 0,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }), encoding="utf-8")
    return path


def _contract(tmp_path: Path) -> dict:
    sampling_path = _sampling(tmp_path / "sampling.json")
    return {
        "base_url": "http://127.0.0.1:26600",
        "server_manifest": {
            "run_id": "control-fixture",
            "model_name": "b500-fixture",
            "checkpoint_path": str(tmp_path / "checkpoint"),
        },
        "policy_sampling": {
            "path": str(sampling_path),
            "sampling": {
                "temperature": 0.0,
                "seed": 0,
                "max_completion_tokens": 4096,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
            },
        },
        "task_ids": ["multi_turn_long_context_170"],
        "category": "multi_turn_long_context",
        "proxy_port": 26700,
        "proxy_python": "python",
        "official_output": str(tmp_path / "official"),
    }


def test_transform_preserves_every_history_row_and_performs_zero_auxiliary_calls():
    source = _messages()
    source_snapshot = json.loads(json.dumps(source))
    transformed, stats = textarms.hiagent_envelope_only_transform(source)

    assert source == source_snapshot
    assert transformed[0]["content"] == (
        source[0]["content"].rstrip() + "\n" + textarms.HIAGENT_SUBGOAL_NOTE
    )
    assert transformed[1:] == source[1:]
    assert textarms.HIAGENT_RETRIEVAL_NOTE.strip() not in transformed[0]["content"]
    assert stats["n_compressor_calls"] == 0
    assert stats["n_summarized"] == 0
    assert stats["retrieved_subgoals"] == []
    assert stats["n_segments"] == 1


def test_proxy_control_keeps_native_tools_and_never_calls_compressor(monkeypatch):
    payload = {
        "model": "b500-fixture",
        "messages": _messages(),
        "tools": [{"type": "function", "function": {"name": "search"}}],
    }
    monkeypatch.setattr(
        proxy,
        "_textarm_compress",
        lambda *args, **kwargs: pytest.fail("envelope control called compressor"),
    )
    staged, stats = proxy._apply_text_arm(
        payload, get_arm("hiagent_envelope_only_native"), "fixture-conversation"
    )

    assert staged["messages"][1:] == payload["messages"][1:]
    assert staged["tools"] == payload["tools"]
    assert all(
        (tool.get("function") or {}).get("name") != textarms.HIAGENT_RETRIEVE_TOOL_NAME
        for tool in staged["tools"]
    )
    assert stats["n_compressor_calls"] == 0
    assert stats["compressor_usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "wall_sec": 0.0,
    }


def test_native_backend_and_bridge_accept_only_exact_control_shape(tmp_path):
    payload = {"messages": _messages(), "tools": []}
    backend = EventNativeHiAgentBackend(lambda *args: None)
    arm = get_arm("hiagent_envelope_only_native")
    assert backend.prepare_chat(payload, arm, None) == payload

    client = proxy._configure_native_hiagent_bridge(
        backend_name="event_native_hiagent",
        arm=arm,
        policy_sampling_path=str(_sampling(tmp_path / "sampling.json")),
        max_generation_attempts_per_task=96,
        no_upstream_retries=True,
        capture_request_views=True,
        request_log=str(tmp_path / "proxy.jsonl"),
        memory_runtime_config="",
        upstream="http://127.0.0.1:26600",
    )
    assert client.max_calls_per_task == 96


def test_control_worker_omits_retrieval_capability_and_preserves_full_worker(tmp_path):
    contract = _contract(tmp_path)
    control_argv = control_worker.build_run_argv(contract)
    assert control_argv[control_argv.index("--arm") + 1] == control_worker.ARM
    assert "--capability-features" not in control_argv
    assert control_argv[control_argv.index("--max-generation-attempts") + 1] == "96"
    assert control_argv[control_argv.index("--max-generation-attempts-per-task") + 1] == "96"
    assert control_argv[control_argv.index("--max-extraction-attempts") + 1] == "0"

    full_argv = full_worker.build_run_argv(contract)
    assert full_argv[full_argv.index("--arm") + 1] == "hiagent_full_native"
    assert full_argv[full_argv.index("--capability-features") + 1] == (
        "hiagent_trajectory_retrieval_v1"
    )
    assert full_worker.ARM == "hiagent_full_native"
    assert full_worker.WORKER_MODULE.endswith("event_native_hiagent_bfcl")
    with control_worker._control_identity():
        assert full_worker.WORKER_MODULE.endswith(
            "event_native_hiagent_envelope_bfcl"
        )
    assert full_worker.WORKER_MODULE.endswith("event_native_hiagent_bfcl")


def test_control_preflight_requires_same_native_transport_contract():
    from benchmarks.capabilities import preflight

    result = preflight(
        "bfcl",
        "hiagent_envelope_only_native",
        "event_native_hiagent",
        options={
            "native_hiagent_policy_sampling": "sampling.json",
            "max_generation_attempts_per_task": 96,
            "no_upstream_retries": True,
            "capture_request_views": True,
            "memory_runtime_config": "",
        },
    )
    native = next(
        item for item in result.requirements
        if item.code == "native_hiagent_explicit_contract"
    )
    assert native.satisfied is True
    variant = next(
        item for item in result.variants
        if item["name"] == "hiagent_action_envelope_control_v1"
    )
    assert variant["status"] == "control"
