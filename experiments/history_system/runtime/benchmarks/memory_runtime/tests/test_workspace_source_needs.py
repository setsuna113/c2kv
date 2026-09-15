"""Preserve observed workspace and source bounds in prediction diagnostics."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]
from memory_runtime.tests.test_source_needs import _store
from memory_runtime.workspace_source_needs import build_prediction_pair, verify_gist_layout


def fixture():
    store = _store()
    workspace = [{"role": "assistant", "content": "", "c2kv_key_hash": "gist-a"},
                 {"role": "user", "content": "Observed actor workspace"}]
    metadata = {"selected_source_indices": [], "block_refs": [{"key_hash": "gist-a", "gist_tokens": 20}]}
    def counter(messages, tools):
        return 50 + len(messages) * 10 + 30 * len(tools[0]["function"]["parameters"]["properties"]["needs"]["items"]["properties"]["source_ids"]["items"]["enum"]) if tools else 20
    return store, workspace, metadata, counter


def test_workspace_is_retained_and_only_actual_raw_visibility_excludes_sources():
    store, workspace, meta, count = fixture()
    original = copy.deepcopy(workspace)
    meta["selected_source_indices"] = list(store.event("needs-test:m1").source_indices)
    result = build_prediction_pair(store, [], workspace, meta, count)
    ids = [entry["source_id"] for entry in result["context"]["index"]]
    assert "needs-test:m1" not in ids
    assert result["messages"]["workspace"][:-1] == workspace == original
    assert result["gist_prompt_tokens"] == {"metadata": 0, "workspace": 20}
    assert "OLD-RAW-MARKER" not in json.dumps(result["messages"])


def test_empty_index_has_no_prediction_to_dispatch():
    store, workspace, meta, count = fixture()
    meta["selected_source_indices"] = [i for event in store.events for i in event.source_indices]
    result = build_prediction_pair(store, [], workspace, meta, count)
    assert result["status"] == "no_missing_raw_candidates"
    assert not result["context"]["index"]


def test_shared_fitting_counts_gist_and_never_truncates_workspace():
    store, workspace, meta, count = fixture()
    result = build_prediction_pair(store, [], workspace, meta, count,
        context_window=135, completion_token_cap=20)
    assert result["dropped_source_ids"]
    assert result["messages"]["workspace"][:-1] == workspace
    assert max(result["total_prompt_tokens"].values()) + 20 <= 135
    result = build_prediction_pair(store, [], workspace, meta, count,
        context_window=30, completion_token_cap=20)
    assert result["status"] == "input_exceeds_cap"


def test_gist_geometry_cannot_silently_use_different_carriers():
    store, workspace, meta, count = fixture()
    meta["block_refs"][0]["key_hash"] = "other-gist"
    with pytest.raises(ValueError, match="carriers"):
        build_prediction_pair(store, [], workspace, meta, count)


def test_backend_abbreviated_hash_layout_keeps_order_and_length_checks():
    blocks = [{"key_hash": "a" * 64, "gist_tokens": 17}, {"key_hash": "b" * 64, "gist_tokens": 23}]
    layout = [{"kind": "gist", "key_hash": "a" * 16, "gist_len": 17},
              {"kind": "gist", "key_hash": "b" * 16, "gist_len": 23}]
    assert verify_gist_layout(layout, blocks)["gist_tokens"] == 40
    for bad in (layout[::-1], layout[:1], [{**layout[0], "gist_len": 18}, layout[1]]):
        with pytest.raises(ValueError, match="layout"):
            verify_gist_layout(bad, blocks)
