"""Source anchors through the legacy proxy's real history assembly."""
from __future__ import annotations

import sys
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy  # noqa: E402
import toolmemory  # noqa: E402
from arms import Arm  # noqa: E402
from test_toolmemory import memory  # noqa: E402


def _source_payload(messages, index, definition, source="appworld.api_docs.show_api_doc"):
    content = messages[index]["content"]
    start = content.index(definition)
    return {"messages": messages, toolmemory.TOOL_SPANS_FIELD: [
        {"message_index": index, "start": start, "end": start + len(definition),
         "source": source}]}


def _set_plan(monkeypatch, plan):
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)


def test_off_assemble_request_retains_exact_legacy_messages_and_counts(monkeypatch):
    _set_plan(monkeypatch, None)
    messages = [{"role": "user", "content": "Run Python code."}]
    arm = Arm(name="full", compress_history=False)
    assert proxy._assemble_request(messages, arm) == proxy._assemble(messages, arm)


def test_appworld_current_doc_anchor_accounts_for_default_system(tmp_path, monkeypatch):
    definition = "{'api_name': 'search', 'parameters': []}"
    messages = [{"role": "user", "content": "Use Python only.\n" + definition + "\nNow search."}]
    plan = memory(tmp_path).plan(_source_payload(messages, 0, definition))
    _set_plan(monkeypatch, plan)
    out, counts = proxy._assemble_request(plan.messages, Arm(name="full", compress_history=False))
    assert out[0] == {"role": "system", "content": proxy.DEFAULT_SYSTEM_PROMPT}
    assert out[1]["content"].startswith("Use Python only.")
    assert definition not in out[1]["content"]
    assert toolmemory.is_carrier(out[2])
    assert out[2][toolmemory.CARRIER_MARK]["anchor"]["message_index"] == 0
    assert counts["tool_memory_insert_at"] == 2
    assert "_tool_source_out_indices" not in counts


def test_history_doc_anchor_follows_its_retained_compressed_record(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "DOC_PACKING", "message")
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    definition = "{'api_name': 'search', 'parameters': []}"
    messages = [{"role": "system", "content": "Generate Python code."},
                {"role": "user", "content": "Document: " + definition},
                {"role": "assistant", "content": "print('observed')"},
                {"role": "user", "content": "Search now."}]
    plan = memory(tmp_path).plan(_source_payload(messages, 1, definition))
    _set_plan(monkeypatch, plan)
    out, counts = proxy._assemble_request(plan.messages, Arm(name="c2kv_r8", compress_history=True))
    history = counts["compressed_records"]
    source_doc = next(record for record in history if 1 in record["source_indices"])
    index = source_doc["out_index"]
    assert toolmemory.source_span_placeholder(plan.source_spans[0]) in out[index]["content"]
    assert toolmemory.is_carrier(out[index + 1])
    assert out[index + 1][toolmemory.CARRIER_MARK]["anchor"]["message_index"] == 1


def test_history_policy_cannot_silently_resurrect_dropped_tool_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    definition = "{'api_name': 'old', 'parameters': []}"
    messages = [{"role": "system", "content": "Generate Python code."},
                {"role": "user", "content": "Old doc: " + definition},
                {"role": "assistant", "content": "print('old')"},
                {"role": "user", "content": "A later turn."},
                {"role": "assistant", "content": "print('later')"},
                {"role": "user", "content": "Current query."}]
    plan = memory(tmp_path).plan(_source_payload(messages, 1, definition))
    _set_plan(monkeypatch, plan)
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_anchor_evicted"):
        proxy._assemble_request(plan.messages, Arm(name="c2kv_r8", compress_history=True))


def test_history_kv_carrier_events_align_with_engine_remap(tmp_path, monkeypatch):
    definition = "{'api_name': 'search', 'parameters': []}"
    messages = [{"role": "system", "content": "Generate Python code."},
                {"role": "assistant", "content": "print(apis.api_docs.show_api_doc())"},
                {"role": "user", "content": "Observed: " + definition},
                {"role": "user", "content": "Search now."}]
    plan = memory(tmp_path).plan(_source_payload(messages, 2, definition))
    _set_plan(monkeypatch, plan)
    arm = Arm(name="history_kv_h2o_r312", compress_history=False,
              history_kv={"method": "h2o", "retention_ratio": 0.312})
    out, counts = proxy._assemble_request(plan.messages, arm)
    events = counts["history_kv_event_messages"]
    assert len(events) == len(out)
    assert [event["message_index"] for event in events] == list(range(len(out)))
    carrier_indices = [index for index, message in enumerate(out) if toolmemory.is_carrier(message)]
    assert len(carrier_indices) == 1
    assert events[carrier_indices[0]]["phase"] == "others"

    engine_dir = HERE.parent.parent / "sglang-paper-tool-integration" / "python" / "sglang" / "srt" / "mem_cache"
    remap = runpy.run_path(str(engine_dir / "c2kv_composition.py"))["remap_message_metadata"]
    resolve = runpy.run_path(str(engine_dir / "history_kv_events.py"))["resolve_history_kv_event_token_spans"]
    hint = {"history_kv_event_messages": events}
    remap(hint, carrier_indices, len(out))
    kept = hint["history_kv_event_messages"]
    assert len(kept) == len(out) - len(carrier_indices)
    spans = resolve(total_tokens=10 * len(kept),
                    message_prefix_token_counts=list(range(0, 10 * len(kept) + 1, 10)),
                    event_messages=kept)
    assert [span["message_index"] for span in spans] == list(range(len(kept)))
