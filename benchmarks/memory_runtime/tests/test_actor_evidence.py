"""Actor evidence reads are budgeted, bounded, and isolated from execution."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]
import proxy
from arms import get_arm
from memory_runtime.actor_evidence import ActorEvidenceRuntime
from memory_runtime.native_source_needs import TOOL_NAME
from memory_runtime.policy import PolicyInputError
from memory_runtime.tests.test_native_workspace import _setup
from memory_runtime.tests.test_source_needs_runtime import _config, _messages


def count(messages, tools):
    return len(json.dumps(messages, separators=(",", ":"))) + len(json.dumps(tools or [], separators=(",", ":")))


def prepare(monkeypatch, route="ac_actor_evidence_once", *, budget=20000, schema_cap=20000):
    _setup(monkeypatch, "ac_native_workspace")
    runtime = ActorEvidenceRuntime({**_config(route, budget), "actor_evidence_schema_token_cap": schema_cap}, count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("full" if route.startswith("raw_") else "c2kv4"))
    messages = _messages()
    # Avoid matching the old ls event in the lexical query so it is eligible.
    messages[5]["content"] = "Continue the current goal."
    out, counts, prepared = proxy._prepare_actor_evidence_input(messages,
        {"task_id": "needs-task", "attempt": 0, "user_turn": 1, "step": 1}, [])
    return runtime, out, counts, prepared


def request(ids, extra_calls=()):
    return {"tool_calls": [{"type": "function", "function": {"name": TOOL_NAME,
        "arguments": json.dumps({"needs": [{"kind": "prior_result", "source_ids": ids}]})}}, *extra_calls]}


def test_offer_preserves_application_tools_and_charges_actual_schema(monkeypatch):
    runtime, out, counts, prepared = prepare(monkeypatch)
    meta = counts["memory_runtime"]
    assert meta["actor_evidence"]["status"] == "offered"
    assert prepared.offered_tools[:-1] == prepared.application_tools == []
    delta = count([m for m in out if not m.get("c2kv_key_hash")], prepared.offered_tools) - count(
        [m for m in out if not m.get("c2kv_key_hash")], [])
    assert delta == meta["actor_evidence"]["schema_tokens"] > 0
    assert meta["active_history_bytes"] == meta["raw_history_tokens"] + meta["gist_tokens"] <= 20000
    assert meta["gist_tokens"] > 0
    assert "OLD-FILE-LIST" not in json.dumps(prepared.offered_tools)
    assert runtime.reconsider_actor(prepared, {"tool_calls": []}) is None


def test_raw_full_view_never_offers_redundant_evidence_tool(monkeypatch):
    runtime, out, counts, prepared = prepare(monkeypatch, "raw_actor_evidence_once")
    assert counts["memory_runtime"]["actor_evidence"]["status"] == "no_missing_raw_candidates"
    assert prepared.offered_tools == []
    assert "OLD-FILE-LIST" in json.dumps(out)
    assert runtime.reconsider_actor(prepared, {"tool_calls": []}) is None


def test_schema_cap_skip_retains_original_lexical_input(monkeypatch):
    runtime, out, counts, prepared = prepare(monkeypatch, schema_cap=1)
    assert counts["memory_runtime"]["actor_evidence"]["status"] == "offer_not_admitted"
    assert prepared.offered_tools == [] and not prepared.index
    assert counts["memory_runtime"]["actor_evidence"]["schema_tokens"] == 0


def test_internal_read_restores_source_without_committing_mixed_action(monkeypatch):
    runtime, out, counts, prepared = prepare(monkeypatch)
    ids = [entry["source_id"] for entry in prepared.index]
    assert "needs-task:m1" in ids
    app = {"type": "function", "function": {"name": "delete_file", "arguments": "{}"}}
    final, measured = runtime.reconsider_actor(prepared, request(["needs-task:m1"], [app]))
    assert "OLD-FILE-LIST" in json.dumps([row for row in final if not row.get("c2kv_key_hash")])
    assert "delete_file" not in json.dumps(final)
    meta = measured["memory_runtime"]
    assert meta["actor_evidence"]["mixed_application_calls_discarded"] == 1
    assert meta["actor_evidence"]["admitted_event_ids"] == ["needs-task:m1"]
    assert meta["actor_evidence"]["schema_tokens"] == 0
    assert meta["total_raw_prompt_tokens"] == count([m for m in final if not m.get("c2kv_key_hash")], [])
    assert meta["gist_tokens"] > 0 and meta["active_history_bytes"] <= 20000
    with pytest.raises(PolicyInputError, match="resolved"):
        runtime.reconsider_actor(prepared, request(["needs-task:m1"]))


@pytest.mark.parametrize("ids", [["future:m99"], ["needs-task:m1"] * 3])
def test_unoffered_or_over_budget_request_cannot_restore_history(monkeypatch, ids):
    runtime, out, counts, prepared = prepare(monkeypatch)
    with pytest.raises(PolicyInputError, match="Invalid internal"):
        runtime.reconsider_actor(prepared, request(ids))
