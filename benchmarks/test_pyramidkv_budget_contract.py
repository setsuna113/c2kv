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
    assert hint["history_kv_eviction"]["target_tokens"] == 768
    assert hint["active_history_kv_tokens"] == 768

    normalized = backend.normalize_response({
        "choices": [{"message": {"content": "ok", "tool_calls": None},
                     "finish_reason": "stop"}],
        "metadata": {"kv_memory_report": {
            "history_kv_method": "pyramidkv",
            "history_kv_backend": "physical_eviction",
            "active_history_kv_tokens": 768,
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
