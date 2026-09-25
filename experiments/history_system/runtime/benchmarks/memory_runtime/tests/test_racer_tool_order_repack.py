"""A recovery repack renders the tool catalog exactly as the draft did.

box4 ToolSandbox B=256 (racer_v4 SnapKV/H2O/StreamingLLM c1_v2_verified): the
draft measured the canonical (key-sorted) catalog, while the recovery wrapper
kept the harness's key order. The repack then counted a longer common tool
prologue and the pre-generation check rejected the regeneration
(assembled_history_differs_from_controller_accounting, 262 > 256 tokens).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.packing import native_ids
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.racer.allocator import PersistentHistoryAllocator
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_racer_composition import config


def _inversions(value):
    if isinstance(value, dict):
        keys = list(value)
        return (sum(later < earlier for earlier, later in zip(keys, keys[1:]))
                + sum(_inversions(item) for item in value.values()))
    if isinstance(value, list):
        return sum(_inversions(item) for item in value)
    return 0


class OrderedTokenizer(Tokenizer):
    """The chat template renders the catalog in its given key order.

    One extra token per out-of-order key pair stands in for the real BPE
    merges that change with key order (6 tokens on the box4 prologue).
    """

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        ids = super().apply_chat_template(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return [0] * _inversions(tools or []) + ids


# serving_tools order: "type" before "function", as the single-task harness sends it.
HARNESS_TOOLS = [{"type": "function", "function": {
    "name": "search_contacts", "description": "Search the contact book.",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "strict": False}}]


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def test_repack_measures_the_draft_catalog_despite_harness_key_order():
    tokenizer = OrderedTokenizer()
    rows = messages()
    assert (len(native_ids(tokenizer, rows, tools=HARNESS_TOOLS))
            != len(native_ids(tokenizer, rows, tools=[_canonical(tool) for tool in HARNESS_TOOLS])))
    geometry = packing()
    geometry["ratios"] = [8]
    base = PersistentHistoryAllocator(tokenizer, packing=geometry, policy=policy(1024),
                                      backend_config=config("snapkv", budget=1024))
    control = CandidateRecoveryController(base, {"variant": "static_t02"}, risk_model=Risk(0.9))
    prepared = control.prepare({"session_id": "s", "decision_key": "d1", "messages": rows,
                                "tools": copy.deepcopy(HARNESS_TOOLS)}, ratio=8, max_new_tokens=32)
    assert history_budget_receipt(prepared.memory, prepared.metadata, base,
                                  ratio=8, phase="draft")["status"] == "passed"
    assert list(prepared._tools) == HARNESS_TOOLS  # the wrapper keeps the client's order
    candidate = prepared.memory.view.gist_event_ids[0]
    recovered, metadata, receipt = repack(control.base, prepared, candidate=candidate)
    assert receipt["status"] == "admitted"
    assert recovered.memory.common_tokens == prepared.memory.common_tokens
    assert recovered.memory.source_tools == prepared.memory.source_tools
    check = history_budget_receipt(recovered.memory, metadata, base, ratio=8, phase="regeneration")
    assert check["status"] == "passed", check["errors"]


def test_canonical_catalogs_are_measured_unchanged():
    tokenizer = OrderedTokenizer()
    geometry = packing()
    geometry["ratios"] = [8]
    base = PersistentHistoryAllocator(tokenizer, packing=geometry, policy=policy(1024),
                                      backend_config=config("h2o", budget=1024))
    canonical = [_canonical(tool) for tool in HARNESS_TOOLS]
    first = base.prepare({"session_id": "a", "decision_key": "d1", "messages": messages(),
                          "tools": copy.deepcopy(canonical)}, ratio=8, max_new_tokens=32)
    second = base.prepare({"session_id": "b", "decision_key": "d1", "messages": messages(),
                           "tools": copy.deepcopy(HARNESS_TOOLS)}, ratio=8, max_new_tokens=32)
    assert first.memory.source_tools == second.memory.source_tools == tuple(canonical)
    assert first.memory.common_tokens == second.memory.common_tokens
