"""Check narration boundaries through the shared runtime and native renderer."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.raw_narration import TOOL_BLOCK, prune_older_narration
from memory_runtime.tests.test_source_needs_runtime import _messages, _prepare
from memory_runtime.tests.test_native_workspace import _count


def _source():
    source = _messages()
    for row in source:
        if row.get("tool_calls"):
            row["content"] = "I will inspect the files before making the next decision."
    return source


def test_candidate_preserves_calls_results_gist_and_protected_messages(monkeypatch):
    source = _source()
    original_source = copy.deepcopy(source)
    original, before = _prepare(monkeypatch, "ac_native_needs_lexical", source)
    candidate, after = _prepare(monkeypatch, "ac_native_needs_lexical_narration", source)
    base, meta = before["memory_runtime"], after["memory_runtime"]
    receipt = meta["raw_narration"]
    assert receipt["saved_raw_tokens"] > 0
    partial = set(receipt["partial_raw_tool_call_source_indices"])
    assert partial and partial <= set(base["source_coverage"]["gist_fully_represented_source_indices"])
    workspace = base["pre_generation_workspace"]
    changed_positions = {p for i,p in zip(workspace["restored_source_indices"], workspace["native_workspace_out_indices"]) if i in partial}
    for position, (old, new) in enumerate(zip(original, candidate)):
        if position in changed_positions:
            assert [m.group(0) for m in TOOL_BLOCK.finditer(old["content"])] == [m.group(0) for m in TOOL_BLOCK.finditer(new["content"])]
            assert {k:v for k,v in old.items() if k != "content"} == {k:v for k,v in new.items() if k != "content"}
        else:
            assert new == old
    assert 6 not in partial  # Latest complete action and its response are protected.
    assert meta["selected_source_indices"] == sorted(set(base["selected_source_indices"])-partial)
    assert meta["source_coverage"]["unrepresented_source_indices"] == base["source_coverage"]["unrepresented_source_indices"]
    assert meta["block_refs"] == base["block_refs"] and meta["gist_tokens"] == base["gist_tokens"]
    assert meta["active_history_bytes"] < base["active_history_bytes"]
    assert source == original_source and "raw_narration" not in base


def test_no_history_keeps_full_input(monkeypatch):
    source = [{"role":"user", "content":"Start."}]
    original, _ = _prepare(monkeypatch, "ac_native_needs_lexical", source)
    candidate, after = _prepare(monkeypatch, "ac_native_needs_lexical_narration", source)
    assert candidate == original
    assert after["memory_runtime"]["raw_narration"]["saved_raw_tokens"] == 0


def test_sources_without_complete_gist_are_kept(monkeypatch):
    source = _source()
    original, before = _prepare(monkeypatch, "ac_native_needs_lexical", source)
    before["memory_runtime"]["source_coverage"]["gist_fully_represented_source_indices"] = []
    candidate, after = prune_older_narration(source, original, before, [], _count)
    assert candidate == original
    assert after["memory_runtime"]["raw_narration"]["partial_raw_tool_call_source_indices"] == []


def test_parallel_calls_are_preserved_byte_for_byte(monkeypatch):
    source = _source()
    source[1]["tool_calls"].append({"id":"c4", "type":"function",
        "function":{"name":"stat", "arguments":json.dumps({"file":"a.txt"})}})
    source.insert(3, {"role":"tool", "tool_call_id":"c4", "content":'{"exists":true}'})
    original, _ = _prepare(monkeypatch, "ac_native_needs_lexical", source)
    candidate, after = _prepare(monkeypatch, "ac_native_needs_lexical_narration", source)
    assert 1 in after["memory_runtime"]["raw_narration"]["partial_raw_tool_call_source_indices"]
    calls = lambda rows: [m.group(0) for row in rows for m in TOOL_BLOCK.finditer(row.get("content", ""))]
    assert calls(candidate) == calls(original)
