"""Local end-to-end contract for an absolute PyramidKV budget arm.

This stays CPU-only: it exercises the same proxy assembly and SGLang request
shaping used by a live cell, then checks that the server's PyramidKV report is
flattened into the measured cost columns.  No model or remote server is
started by this test.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import proxy
from benchmarks.arms import get_arm, history_kv_spec
from benchmarks.backends.sglang import SglangBackend


def _messages():
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "current request"},
    ]


def test_pyramidkv_absolute_budget_reaches_chat_hint_and_cost_columns():
    arm = get_arm("gen_pyramidkv_k0")
    spec = dict(history_kv_spec(arm))
    spec["target_tokens"] = 768
    spec["retention_ratio"] = None
    arm = replace(arm, history_kv=spec)

    assembled, counts = proxy._assemble(_messages(), arm)
    context = proxy._history_kv_context(assembled, counts, arm)
    assert context is not None
    assert context["spec"]["method"] == "pyramidkv"
    assert context["spec"]["target_tokens"] == 768

    backend = SglangBackend(lambda path, payload, timeout, retries=2: {})
    prepared = backend.prepare_chat(
        {"model": "qwen3-4b", "messages": assembled},
        arm,
        None,
        context={"history_kv": context},
    )
    hint = prepared["c2kv_kv_memory_hint"]
    assert hint["history_kv_method"] == "pyramidkv"
    assert hint["history_kv_backend"] == "reference_attention"
    assert hint["history_kv_eviction"]["target_tokens"] == 768
    assert hint["active_history_kv_tokens"] == 768

    normalized = backend.normalize_response({
        "choices": [{"message": {"content": "ok", "tool_calls": None},
                     "finish_reason": "stop"}],
        "metadata": {"kv_memory_report": {
            "history_kv_method": "pyramidkv",
            "history_kv_backend": "reference_attention",
            "active_history_kv_tokens": 768,
            "reference_attention_backend": "torch_sdpa",
            "reference_history_token_slots": 221184,
            "reference_history_resident_bytes": 115015680,
            "full_equivalent_history_tokens": 2048,
            "history_kv_physical_eviction": {
                "success": True,
                "method": "pyramidkv",
                "kept_history_tokens": 768,
                "freed_physical_slots": 256,
                "freed_kv_bytes": 123,
            },
        }},
    })
    cost = normalized["cost"]
    assert cost["history_kv_method"] == "pyramidkv"
    assert cost["history_kv_kept_tokens"] == 768
    assert cost["history_kv_freed_bytes"] == 123


def test_reference_cost_keeps_measured_resident_storage():
    cost = SglangBackend._history_kv_cost({"metadata": {"kv_memory_report": {
        "history_kv_backend": "reference_attention",
        "reference_attention_backend": "torch_sdpa",
        "reference_history_token_slots": 321,
        "reference_history_resident_bytes": 12345,
    }}})
    assert cost["reference_history_token_slots"] == 321
    assert cost["reference_history_resident_bytes"] == 12345
    assert cost["reference_attention_backend"] == "torch_sdpa"


def test_event_roles_survive_tool_normalization(monkeypatch):
    monkeypatch.setattr(proxy, "BENCHMARK", "bfcl")
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "action"},
        {"role": "tool", "content": "observation", "tool_call_id": "a"},
    ]
    assembled, counts = proxy._assemble(messages, get_arm("commitkv"))
    assert assembled[-1]["role"] == "user"
    assert counts["history_kv_event_messages"][-1] == {
        "message_index": 3, "role": "tool", "phase": "tool"}
    assert counts["history_kv_event_messages"][0]["role"] == "system"


def test_reference_measurement_does_not_replace_external_history_with_zero_pool_slots():
    data = {"metadata": {"kv_memory_report": {
        "history_kv_backend": "reference_attention",
        "active_history_kv_tokens": 123,
        "full_equivalent_history_tokens": 500,
        "reference_history_resident_bytes": 9876,
        "history_kv_physical_eviction": {"kept_history_tokens": 0},
    }}}
    result = SglangBackend._server_measurement(data)
    assert result["history_active_kv_tokens"] == 123
    assert result["reference_history_resident_bytes"] == 9876
