"""Exercise subgoal packing through the actual source-needs renderer."""

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

import proxy
from arms import get_arm
from memory_runtime.extraction_telemetry import capture_extractions
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
from memory_runtime.tests.test_source_needs_runtime import _config


ROUTE = "ac_native_needs_lexical_raw_reserve_failed_operation"


def source():
    def call(identifier, content=None):
        return {"role": "assistant", "content": content, "tool_calls": [{
            "id": identifier, "type": "function", "function": {
                "name": "inspect", "arguments": "{}"}}]}
    return [
        {"role": "user", "content": "Inspect the archive and verify its contents."},
        call("a", "Subgoal: Inspect archive"),
        {"role": "tool", "tool_call_id": "a", "content": "archive located"},
        call("b"),
        {"role": "tool", "tool_call_id": "b", "content": "archive opened"},
        call("c", "Subgoal: Verify contents"),
        {"role": "tool", "tool_call_id": "c", "content": "CURRENT-RESULT"},
    ]


def test_real_renderer_groups_sources_and_leaves_s0_unchanged(monkeypatch):
    _setup(monkeypatch, "ac_native_workspace")
    messages = source()
    original = copy.deepcopy(messages)
    before, old_counts = proxy._assemble(messages, get_arm("c2kv4"))
    grouped, new_counts = proxy._assemble(
        messages, get_arm("c2kv4"), doc_packing="subgoal", subgoal_session_id="test")
    after, after_counts = proxy._assemble(messages, get_arm("c2kv4"))
    assert (before, old_counts) == (after, after_counts)
    assert messages == original and proxy.DOC_PACKING == "turn"
    assert new_counts["subgoal_organization"]["n_started"] == 2
    assert new_counts["subgoal_organization"]["n_active"] == 1
    assert new_counts["subgoal_organization"]["n_closed_by_next_declaration"] == 1
    assert grouped != before
    closed = [row for row in new_counts["compressed_records"]
              if row["subgoal_group"]["lifecycle_status"] == "closed_by_next_declaration"]
    assert any(row["subgoal_group"]["source_indices"] == [1, 2, 3, 4] for row in closed)
    old_sources = {i for row in old_counts["history_packing_fragments"] for i in row["source_indices"]}
    new_sources = {i for row in new_counts["history_packing_fragments"] for i in row["source_indices"]}
    assert old_sources == new_sources
    assert grouped[new_counts["current_start_out_index"]:] == before[old_counts["current_start_out_index"]:]


def test_subgoal_route_uses_shared_budget_and_native_suffix(monkeypatch):
    _setup(monkeypatch, "ac_native_workspace")
    config = {**_config(ROUTE, 3000), "history_organization": "subgoal-v1",
              "actor_prompt_protocol": "native-subgoal-note-v1",
              "latest_complete_tool_protection": "budgeted"}
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    output, counts = proxy._prepare_memory_input(
        source(), {"task_id": "test", "attempt": 0, "user_turn": 0, "step": 3}, [])
    meta = counts["memory_runtime"]
    assert meta["history_organization"] == "subgoal-v1"
    assert meta["subgoal_organization"]["n_started"] == 2
    assert meta["retained_subgoal_groups"]
    assert meta["active_history_bytes"] <= 3000 and meta["gist_tokens"] > 0
    assert any(message.get("content") == "CURRENT-RESULT" for message in output)
    assert "Subgoal:" in output[0]["content"]
    assert counts["doc_packing"] == "subgoal"


def test_pending_event_is_never_extracted_by_subgoal_renderer(monkeypatch):
    _setup(monkeypatch, "ac_native_workspace")
    messages = source()[:2] + [{"role": "user", "content": "Wait for the outstanding result."}]
    _, counts = proxy._assemble(messages, get_arm("c2kv4"), doc_packing="subgoal")
    sources = {i for row in counts["history_packing_fragments"] for i in row["source_indices"]}
    assert 1 not in sources
    assert counts["subgoal_organization"]["n_archivable"] == 0


def test_long_subgoal_fragments_keep_exact_part_provenance(monkeypatch):
    class Runtime:
        always_compress = True

    class Backend:
        @staticmethod
        def extract(content, role, ratio, tools=None):
            return {
                "key_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "gist_len": 1,
                "original_seq_len": len(content),
            }

    def call(identifier, content):
        return {"role": "assistant", "content": content, "tool_calls": [{
            "id": identifier, "type": "function", "function": {
                "name": "inspect", "arguments": "{}"}}]}

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "root request"},
        call("a", "Subgoal: inspect SRC2_" + "A" * 30),
        {"role": "tool", "tool_call_id": "a", "content": "SRC3_" + "B" * 30},
        call("b", "SRC4_" + "C" * 30),
        {"role": "tool", "tool_call_id": "b", "content": "SRC5_" + "D" * 30},
        call("c", "SRC6_" + "E" * 30),
        {"role": "tool", "tool_call_id": "c", "content": "SRC7_" + "F" * 30},
        {"role": "assistant", "content": "SRC8_" + "G" * 30},
        {"role": "user", "content": "current request"},
    ]
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", Runtime())
    monkeypatch.setattr(proxy, "BACKEND", Backend())
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 350)
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 20)

    with capture_extractions() as trace:
        _, counts = proxy._assemble(
            messages, get_arm("c2kv4"), doc_packing="subgoal",
            subgoal_session_id="long-provenance")

    records = [row for row in counts["compressed_records"]
               if (row.get("subgoal_group") or {}).get("subgoal_id")]
    full_group_sources = list(range(2, 9))
    assert len(records) > 1
    assert all(row["subgoal_group"]["source_indices"] == full_group_sources
               for row in records)
    assert all(row["source_indices"] != full_group_sources for row in records)
    assert any(len(row["source_indices"]) > 2 for row in records)

    for row in records:
        for source_index in full_group_sources:
            assert ((source_index in row["source_indices"])
                    == (f"SRC{source_index}_" in row["content"]))

    fragments = {row["fragment_id"]: row
                 for row in counts["history_packing_fragments"]}
    telemetry = trace.snapshot(
        block_refs=[], forwarded_requests=[], request_status="ok")["events"]
    telemetry_by_key = {}
    for event in telemetry:
        telemetry_by_key.setdefault(event["key_hash"], []).append(event)
    for row in records:
        fragment = fragments[row["packing_fragment_id"]]
        assert fragment["source_indices"] == row["source_indices"]
        assert any(event["source_indices"] == row["source_indices"]
                   for event in telemetry_by_key[row["record"]["key_hash"]])


def test_note_only_control_has_identical_common_prompt_and_turn_packing(monkeypatch):
    _setup(monkeypatch, "ac_native_workspace")
    config = {**_config(ROUTE, 3000), "actor_prompt_protocol": "native-subgoal-note-v1",
              "latest_complete_tool_protection": "budgeted"}
    results = []
    for organization in ("turn", "subgoal-v1"):
        runtime = SourceNeedsRuntime({**config, "history_organization": organization}, _count)
        monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
        results.append(proxy._prepare_memory_input(
            source(), {"task_id": "test", "attempt": 0, "user_turn": 0, "step": 3}, []))
    (turn_output, turn_counts), (grouped_output, grouped_counts) = results
    assert turn_output[0] == grouped_output[0]
    assert turn_counts["doc_packing"] == "turn"
    assert grouped_counts["doc_packing"] == "subgoal"
    assert turn_counts["memory_runtime"]["common_raw_prompt_tokens"] == grouped_counts["memory_runtime"]["common_raw_prompt_tokens"]
    assert "subgoal_organization" not in turn_counts["memory_runtime"]


def test_note_v2_configs_only_change_protocol_identity_and_count_in_common_prompt(monkeypatch):
    _setup(monkeypatch, "ac_native_workspace")
    pairs = [
        ("ac_subgoal_structure_bounded_latest.json",
         "ac_subgoal_structure_bounded_latest_note_v2.json", "subgoal"),
        ("ac_subgoal_note_turn_control_bounded_latest.json",
         "ac_subgoal_note_turn_control_bounded_latest_note_v2.json", "turn"),
    ]
    common_counts = []
    for old_name, new_name, expected_packing in pairs:
        config_dir = ROOT / "benchmarks" / "memory_runtime" / "configs"
        old = json.loads((config_dir / old_name).read_text(encoding="utf-8"))
        new = json.loads((config_dir / new_name).read_text(encoding="utf-8"))
        changed = {key for key in old.keys() | new.keys()
                   if old.get(key) != new.get(key)}
        assert changed == {"run_id", "actor_prompt_protocol"}
        assert new["actor_prompt_protocol"] == "native-subgoal-note-v2"

        runtime = SourceNeedsRuntime(new, _count)
        monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
        full, full_counts = proxy._assemble(source(), get_arm("full"))
        prompted_full = proxy._apply_actor_prompt_protocol(full)
        cutoff = full_counts["current_start_out_index"]
        expected_common = ([message for message in prompted_full[:cutoff]
                            if message.get("role") == "system"]
                           + prompted_full[cutoff:])

        output, counts = proxy._prepare_memory_input(
            source(), {"task_id": "test", "attempt": 0,
                       "user_turn": 0, "step": 3}, [])
        metadata = counts["memory_runtime"]
        raw_output = [message for message in output
                      if not message.get("c2kv_key_hash")]
        assert output[0]["content"].endswith(
            proxy.textarms.HIAGENT_SUBGOAL_NOTE_V2.strip())
        assert metadata["actor_prompt_protocol"] == "native-subgoal-note-v2"
        assert metadata["common_raw_prompt_tokens"] == _count(expected_common, [])
        assert metadata["total_raw_prompt_tokens"] == _count(raw_output, [])
        assert metadata["active_history_bytes"] == (
            metadata["total_raw_prompt_tokens"]
            - metadata["common_raw_prompt_tokens"]
            + metadata["gist_tokens"]
        ) * metadata["bytes_per_kv_token"]
        assert metadata["active_history_bytes"] <= metadata["history_budget_bytes"]
        assert counts["doc_packing"] == expected_packing
        common_counts.append(metadata["common_raw_prompt_tokens"])
    assert common_counts[0] == common_counts[1]
