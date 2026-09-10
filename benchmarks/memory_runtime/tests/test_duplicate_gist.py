"""Verify pruning preserves observable sources and the required active gist."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.duplicate_gist import prune_duplicate_gist
from memory_runtime.tests.test_source_needs_runtime import _messages, _prepare


def test_candidate_changes_only_gist_after_shared_allocation(monkeypatch):
    messages = _messages()
    original, before = _prepare(monkeypatch, "ac_native_needs_lexical", messages)
    candidate, after = _prepare(monkeypatch, "ac_native_needs_lexical_pruned", messages)
    base, result = before["memory_runtime"], after["memory_runtime"]
    assert [m for m in original if not m.get("c2kv_key_hash")] == [m for m in candidate if not m.get("c2kv_key_hash")]
    assert result["selected_source_indices"] == base["selected_source_indices"]
    assert result["source_coverage"]["unrepresented_source_indices"] == base["source_coverage"]["unrepresented_source_indices"]
    assert 0 < result["gist_tokens"] <= base["gist_tokens"]
    assert result["active_history_bytes"] <= base["active_history_bytes"]
    assert "gist_pruning" not in base


def test_candidate_without_history_stays_exact_full(monkeypatch):
    source = [{"role":"user", "content":"Start."}]
    original, _ = _prepare(monkeypatch, "ac_native_needs_lexical", source)
    candidate, counts = _prepare(monkeypatch, "ac_native_needs_lexical_pruned", source)
    assert candidate == original
    assert counts["memory_runtime"]["gist_tokens"] == 0
    assert counts["memory_runtime"]["gist_pruning"]["removed_gist_tokens"] == 0


def test_complete_raw_overlap_preserves_first_block(monkeypatch):
    from memory_runtime.tests.test_native_workspace import _count
    output, counts = _prepare(monkeypatch, "ac_native_needs_lexical", _messages())
    modified = copy.deepcopy(counts)
    meta = modified["memory_runtime"]
    # Assert the complete-overlap branch using the native factory's carriers.
    meta["selected_source_indices"] = list(meta["source_coverage"]["eligible_source_indices"])
    from memory_runtime.always_compress import coverage_accounting
    meta["source_coverage"] = coverage_accounting(
        eligible_sources=frozenset(meta["selected_source_indices"]),
        raw_sources=set(meta["selected_source_indices"]), retained_blocks=meta["block_refs"],
        packing_fragments=modified["history_packing_fragments"])
    pruned, result = prune_duplicate_gist(output, modified, [], _count)
    assert len(result["memory_runtime"]["block_refs"]) == 1
    assert result["memory_runtime"]["block_refs"][0] == meta["block_refs"][0]
    assert result["memory_runtime"]["gist_reservation"]["satisfied"]
    assert result["memory_runtime"]["gist_pruning"]["preserved_first_block_for_reservation"]
