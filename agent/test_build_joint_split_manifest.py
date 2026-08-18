# -*- coding: utf-8 -*-
"""CPU-only offline unit tests for agent/build_joint_split_manifest.py.

No real dataset and no network: sessions are synthetic spans mirroring the
agent-llm-traces parquet schema (``gen_ai.input.messages`` /
``gen_ai.output.messages`` / ``gen_ai.tool.definitions`` JSON strings plus
``start_time`` / ``span_id`` / ``status``), written to a tiny parquet file
with pyarrow.

Coverage:
a. identical (modulo case/whitespace) first-user instructions land in one
   task-proxy group and never straddle train/eval, across seeds;
b. --toolset_disjoint merges groups that share a toolset hash;
c. no-instruction sessions fall back to per-session singleton groups;
d. deterministic hash split (repeat runs identical; ratio 0/1 bounds);
e. straddle self-checks raise hard errors;
f. toolset-key semantics (order-insensitive, schema-sensitive);
g. manifest schema compatibility with the consumer access pattern in
   python/train/train_data_multiturn.py (_load_records_from_manifest).

Run from the repo root (local venv has pyarrow/pytest):
  pytest agent/test_build_joint_split_manifest.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make agent/ importable when pytest is invoked from the repo root.
_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

import build_joint_split_manifest as bjsm  # noqa: E402


_TOOLS_A = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files under one directory path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

_TOOLS_B = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run one SQL query.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]


def _tools_variant(name):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Run one namespaced action.",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            },
        },
    ]


def _span(span_id, start_time, input_messages, output_messages, tools=None):
    attributes = {
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }
    if tools is not None:
        attributes["gen_ai.tool.definitions"] = json.dumps(tools)
    return {
        "span_id": span_id,
        "start_time": start_time,
        "status": "ok",
        "attributes": attributes,
    }


def _session_spans(instruction, tools, n_spans=2):
    system = {"role": "system", "content": "You are an agent."}
    user1 = {"role": "user", "content": instruction}
    assistant1 = {"role": "assistant", "content": "Working on it."}
    user2 = {"role": "user", "content": "And then report back."}
    spans = [
        _span("span-1", "2026-01-01T00:00:01", [system, user1], [assistant1], tools=tools),
    ]
    if n_spans > 1:
        spans.append(
            _span(
                "span-2",
                "2026-01-01T00:00:02",
                [system, user1, assistant1, user2],
                [{"role": "assistant", "content": "Done."}],
            )
        )
    # Deliberately out of chronological order: the loader must sort spans.
    return list(reversed(spans))


_GROUP_A_INSTRUCTION = "List the files in /tmp please."
_GROUP_B_INSTRUCTION = "Create notes.txt for me."


def _synthetic_rows():
    rows = []
    # Group A: same instruction re-run as three sessions, two subsets, one with
    # case/whitespace variation.
    rows.append({"benchmark": "alpha-bench", "session_id": "sess-a1",
                 "spans": json.dumps(_session_spans(_GROUP_A_INSTRUCTION, _TOOLS_A))})
    rows.append({"benchmark": "beta-bench", "session_id": "sess-a2",
                 "spans": json.dumps(_session_spans(_GROUP_A_INSTRUCTION, _TOOLS_A, n_spans=1))})
    rows.append({"benchmark": "alpha-bench", "session_id": "sess-a3",
                 "spans": json.dumps(_session_spans("  list THE files\nin /tmp PLEASE. ", _TOOLS_A))})
    # Group B: different instruction, SAME toolset as group A.
    rows.append({"benchmark": "alpha-bench", "session_id": "sess-b1",
                 "spans": json.dumps(_session_spans(_GROUP_B_INSTRUCTION, _TOOLS_A))})
    # Groups C1..C6: distinct instructions, each with its OWN toolset (shared
    # toolsets would be merged by --toolset_disjoint).
    for index in range(6):
        rows.append({"benchmark": "gamma-bench", "session_id": f"sess-c{index}",
                     "spans": json.dumps(_session_spans(f"Handle database task number {index}.", _tools_variant(f"c_tool_{index}")))})
    # Group D: no user message anywhere -> per-session fallback group.
    no_user = [_span("span-1", "2026-01-01T00:00:01",
                     [{"role": "system", "content": "system only"}],
                     [{"role": "assistant", "content": "hi"}],
                     tools=_TOOLS_B)]
    rows.append({"benchmark": "gamma-bench", "session_id": "sess-d1", "spans": json.dumps(no_user)})
    # Group E: instruction but no tool definitions anywhere.
    rows.append({"benchmark": "delta-bench", "session_id": "sess-e1",
                 "spans": json.dumps(_session_spans("Run the integration suite.", None))})
    return rows


@pytest.fixture()
def synthetic_dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "agent-llm-traces"
    data_dir.mkdir()
    rows = _synthetic_rows()
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, data_dir / "shard.parquet")
    return str(data_dir)


def _args(dataset_path, **overrides):
    values = {
        "dataset_path": dataset_path,
        "out": None,
        "split_name": "taskproxy_disjoint",
        "seed": 42,
        "eval_ratio": 0.5,
        "toolset_disjoint": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _split_of(manifest, split_name, session_id):
    selected = manifest[split_name]
    if session_id in selected["train_session_ids"]:
        return "train"
    if session_id in selected["eval_session_ids"]:
        return "eval"
    raise AssertionError(f"{session_id} missing from the split")


# ---------------------------------------------------------------------------
# Grouping and split-safety tests.
# ---------------------------------------------------------------------------


def test_identical_instructions_share_one_group(synthetic_dataset):
    manifest = bjsm.build_manifest(_args(synthetic_dataset))
    metadata = manifest["metadata"]
    # 10 task-proxy groups: A (3 sessions), B, C0..C5, D, E.
    assert metadata["num_groups"] == 10
    assert metadata["num_sessions"] == 12
    assert metadata["collision_stats"] == {
        "groups_with_multiple_sessions": 1,
        "sessions_in_multi_session_groups": 3,
    }
    assert metadata["group_size_histogram"] == {"1": 9, "3-5": 1}
    groups = manifest["taskproxy_disjoint"]["session_groups"]
    assert groups["sess-a1"] == groups["sess-a2"] == groups["sess-a3"]
    assert groups["sess-b1"] != groups["sess-a1"]


def test_group_never_straddles_across_seeds(synthetic_dataset):
    for seed in (0, 1, 7, 42, 123, 999):
        manifest = bjsm.build_manifest(_args(synthetic_dataset, seed=seed, eval_ratio=0.3))
        selected = manifest["taskproxy_disjoint"]
        train = set(selected["train_session_ids"])
        eval_ = set(selected["eval_session_ids"])
        assert not (train & eval_)
        # Group A (3 re-runs of one instruction) is atomic.
        assert {"sess-a1", "sess-a2", "sess-a3"} <= train or {"sess-a1", "sess-a2", "sess-a3"} <= eval_
        # Every task-proxy group is entirely on one side.
        by_group = {}
        for session_id, group_id in selected["session_groups"].items():
            by_group.setdefault(group_id, set()).add(session_id)
        for members in by_group.values():
            assert members <= train or members <= eval_


def test_subset_counts_and_session_records(synthetic_dataset):
    manifest = bjsm.build_manifest(_args(synthetic_dataset))
    selected = manifest["taskproxy_disjoint"]
    counts = selected["subset_counts"]
    assert counts["alpha-bench"]["sessions"] == 3
    assert counts["alpha-bench"]["groups"] == 2
    assert counts["beta-bench"]["sessions"] == 1
    assert counts["gamma-bench"]["sessions"] == 7
    assert (
        counts["alpha-bench"]["train_sessions"] + counts["alpha-bench"]["eval_sessions"]
        == counts["alpha-bench"]["sessions"]
    )
    assert selected["session_subsets"]["sess-a2"] == "beta-bench"
    # Same toolset hash for the A/B sessions, different for C, empty for E.
    toolsets = selected["session_toolsets"]
    assert toolsets["sess-a1"] == toolsets["sess-b1"]
    assert toolsets["sess-c0"] != toolsets["sess-a1"]
    assert toolsets["sess-e1"] == ""


def test_toolset_disjoint_merges_groups_sharing_toolset(synthetic_dataset):
    manifest = bjsm.build_manifest(_args(synthetic_dataset, toolset_disjoint=True))
    metadata = manifest["metadata"]
    assert metadata["toolset_disjoint"] is True
    # Groups A and B share _TOOLS_A -> one super-group: 10 -> 9 split groups.
    assert metadata["num_split_groups"] == metadata["num_groups"] - 1
    selected = manifest["taskproxy_disjoint"]
    sides = {_split_of(manifest, "taskproxy_disjoint", sid) for sid in ("sess-a1", "sess-a2", "sess-a3", "sess-b1")}
    assert len(sides) == 1
    # No toolset straddles: every toolset hash is entirely on one side.
    train = set(selected["train_session_ids"])
    by_toolset = {}
    for session_id, toolset in selected["session_toolsets"].items():
        if toolset:
            by_toolset.setdefault(toolset, set()).add(session_id)
    for members in by_toolset.values():
        assert members <= train or members <= set(selected["eval_session_ids"])


def test_toolset_disjoint_merge_mapping_is_transitive():
    sessions = [
        {"session_id": "s1", "group_id": "g1", "toolset_key": "t1"},
        {"session_id": "s2", "group_id": "g2", "toolset_key": "t1"},
        {"session_id": "s3", "group_id": "g3", "toolset_key": "t2"},
        {"session_id": "s4", "group_id": "g4", "toolset_key": "t2"},
        {"session_id": "s5", "group_id": "g2b", "toolset_key": "t3"},
        {"session_id": "s6", "group_id": "g3b", "toolset_key": "t3"},
        {"session_id": "s7", "group_id": "g5", "toolset_key": None},
    ]
    # g2 and g3 do not share a toolset directly; chain them via a session pair.
    sessions.append({"session_id": "s8", "group_id": "g2", "toolset_key": "t2"})
    mapping = bjsm._merge_groups_by_toolset(sessions)
    assert mapping["g1"] == mapping["g2"] == mapping["g3"] == mapping["g4"]
    assert mapping["g2b"] == mapping["g3b"]
    assert mapping["g2b"] != mapping["g1"]
    # Tool-less groups never merge.
    assert mapping["g5"] == "g5"


def test_no_instruction_session_gets_singleton_fallback_group(synthetic_dataset):
    manifest = bjsm.build_manifest(_args(synthetic_dataset))
    metadata = manifest["metadata"]
    assert metadata["sessions_without_instruction"] == 1
    assert metadata["sessions_without_tools"] == 1
    groups = manifest["taskproxy_disjoint"]["session_groups"]
    assert groups["sess-d1"].startswith("__no_instruction__:sess-d1")
    assert list(groups.values()).count(groups["sess-d1"]) == 1


def test_split_is_deterministic_and_ratio_bounded(synthetic_dataset):
    first = bjsm.build_manifest(_args(synthetic_dataset, seed=42, eval_ratio=0.25))
    second = bjsm.build_manifest(_args(synthetic_dataset, seed=42, eval_ratio=0.25))
    assert first["taskproxy_disjoint"]["train_session_ids"] == second["taskproxy_disjoint"]["train_session_ids"]
    assert first["taskproxy_disjoint"]["eval_session_ids"] == second["taskproxy_disjoint"]["eval_session_ids"]

    all_train = bjsm.build_manifest(_args(synthetic_dataset, eval_ratio=0.0))
    assert all_train["taskproxy_disjoint"]["eval_session_ids"] == []
    all_eval = bjsm.build_manifest(_args(synthetic_dataset, eval_ratio=1.0))
    assert all_eval["taskproxy_disjoint"]["train_session_ids"] == []
    with pytest.raises(ValueError):
        bjsm.build_manifest(_args(synthetic_dataset, eval_ratio=1.5))


def test_is_eval_group_matches_split_for_group_style():
    # sha1(f"{seed}:{group_id}") / 2**160 below the ratio -> eval.
    assert bjsm._is_eval_group("anything", 42, 1.0) is True
    assert bjsm._is_eval_group("anything", 42, 0.0) is False
    assert bjsm._is_eval_group("g", 42, 0.5) == bjsm._is_eval_group("g", 42, 0.5)


# ---------------------------------------------------------------------------
# Self-check hard errors.
# ---------------------------------------------------------------------------


def test_straddle_self_check_raises():
    with pytest.raises(RuntimeError, match="straddle"):
        bjsm._assert_no_straddle({"g1": {"s1", "s2"}}, {"s1"}, {"s2"})
    with pytest.raises(RuntimeError, match="overlap"):
        bjsm._assert_no_straddle({"g1": {"s1"}}, {"s1"}, {"s1"})
    # Clean split passes.
    bjsm._assert_no_straddle({"g1": {"s1", "s2"}, "g2": {"s3"}}, {"s1", "s2"}, {"s3"})


def test_toolset_straddle_self_check_raises():
    sessions = [
        {"session_id": "s1", "toolset_key": "t1"},
        {"session_id": "s2", "toolset_key": "t1"},
    ]
    with pytest.raises(RuntimeError, match="Toolsets straddle"):
        bjsm._assert_no_toolset_straddle(sessions, {"s1"}, {"s2"})
    bjsm._assert_no_toolset_straddle(sessions, {"s1", "s2"}, set())


# ---------------------------------------------------------------------------
# Toolset key semantics.
# ---------------------------------------------------------------------------


def test_toolset_key_order_insensitive_schema_sensitive():
    shuffled = list(reversed(_TOOLS_A))
    assert bjsm._toolset_key(shuffled) == bjsm._toolset_key(_TOOLS_A)
    renamed = json.loads(json.dumps(_TOOLS_A))
    renamed[0]["function"]["parameters"]["properties"]["city"]["type"] = "integer"
    assert bjsm._toolset_key(renamed) != bjsm._toolset_key(_TOOLS_A)
    # Same semantics as the diagnose module: sha1 of sorted tool signatures.
    expected = bjsm._hash_json(
        sorted((bjsm._tool_signature(tool) for tool in _TOOLS_A), key=lambda item: item["name"])
    )
    assert bjsm._toolset_key(_TOOLS_A) == expected


def test_first_user_instruction_extraction():
    spans = _session_spans(_GROUP_A_INSTRUCTION, _TOOLS_A)
    assert bjsm._first_user_instruction(spans) == _GROUP_A_INSTRUCTION
    # Content given as a list of parts is flattened.
    parts_span = [
        _span("span-1", "2026-01-01T00:00:01",
              [{"role": "system", "content": "s"},
               {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}],
              [{"role": "assistant", "content": "ok"}])
    ]
    assert bjsm._first_user_instruction(parts_span) == "hello\nworld"
    assert bjsm._first_user_instruction([]) is None
    assert bjsm._normalize_text("  A  B\nC ") == "a b c"


# ---------------------------------------------------------------------------
# Manifest consumer compatibility (python/train/train_data_multiturn.py).
# ---------------------------------------------------------------------------


def test_manifest_consumer_access_pattern(synthetic_dataset, tmp_path):
    split_name = "taskproxy_disjoint"
    manifest = bjsm.build_manifest(_args(synthetic_dataset, split_name=split_name))
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Same json access pattern as train_data_multiturn._load_records_from_manifest
    # and _split_session_ids (lines ~748-807).
    loaded = json.loads(out.read_text(encoding="utf-8"))
    if "train_session_ids" in loaded and "eval_session_ids" in loaded:
        selected = loaded
    else:
        selected = loaded[split_name]
    train_ids = {str(item) for item in selected.get("train_session_ids", [])}
    eval_ids = {str(item) for item in selected.get("eval_session_ids", [])}

    expected = manifest[split_name]
    assert train_ids == set(expected["train_session_ids"])
    assert eval_ids == set(expected["eval_session_ids"])
    all_sessions = {f"sess-a{i}" for i in (1, 2, 3)} | {f"sess-c{i}" for i in range(6)} | {
        "sess-b1",
        "sess-d1",
        "sess-e1",
    }
    assert train_ids | eval_ids == all_sessions
