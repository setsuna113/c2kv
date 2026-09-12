"""CPU-only contracts for bounded H0/H1/H2/H3 preparation."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

import next_compression.history as history_module
from history_memory.dataset import iter_decisions
from next_compression.history import (
    A_VIEW_SOURCE_SHA256,
    A_VIEW_VENDOR_MANIFEST_SHA256,
    A_VIEW_VERSION,
    HistoryPreparationConfig,
    _prepare_a_view,
    prepare_history,
)


class FrozenTokenizer:
    """Character tokenizer with a prefix-separable native chat template."""

    vocab_size = 0x110000
    special_tokens_map = {}
    chat_template = "frozen-test-native-v1"
    eos_token_id = 0x10FFFE

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        **_kwargs,
    ):
        text = (
            "<tools>" + json.dumps(tools, ensure_ascii=False, sort_keys=True) + "</tools>"
            if tools
            else ""
        )
        for message in messages:
            text += (
                "<"
                + message["role"]
                + ">"
                + json.dumps(message, ensure_ascii=False, sort_keys=True)
                + "</end>"
                + chr(self.eos_token_id)
            )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(character) for character in text]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


TOKENIZER = FrozenTokenizer()


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _row(*, session_id: str = "history-1", split: str = "train") -> dict:
    return {
        "session_id": session_id,
        "source": "fixture:agent",
        "split": split,
        "task_id": f"task-{session_id}",
        "template_id": f"template-{session_id}",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit",
                    "description": "Commit a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            },
        ],
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Find alpha."},
            _call("call-1", "lookup", {"key": "alpha"}),
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps({"value": "alpha-17"}),
            },
            {"role": "assistant", "content": "I found alpha-17."},
            {
                "role": "user",
                "content": "Actually revise it: commit the exact alpha-17 value.",
            },
            _call("call-2", "commit", {"value": "alpha-17"}),
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": json.dumps({"success": True, "value": "alpha-17"}),
            },
            {"role": "assistant", "content": "Done."},
        ],
    }


def _balanced_row(*, session_id: str = "history-balanced") -> dict:
    row = _row(session_id=session_id)
    row["messages"].extend(
        [
            {"role": "user", "content": "Verify the committed value."},
            _call("call-3", "lookup", {"key": "alpha"}),
            {
                "role": "tool",
                "tool_call_id": "call-3",
                "content": json.dumps({"value": "alpha-17"}),
            },
            {"role": "assistant", "content": "Verified."},
        ]
    )
    return row


def _wide_config(**overrides) -> HistoryPreparationConfig:
    values = dict(
        max_encoder_tokens=100_000,
        max_system_tokens=100_000,
        max_workspace_tokens=100_000,
        max_sequence_tokens=100_000,
        a_max_new_tokens=32,
        history_budget_bytes=100_000,
        workspace_budget_bytes=100_000,
        kv_bytes_per_token=1,
        predictor_prompt_token_cap=100_000,
    )
    values.update(overrides)
    return HistoryPreparationConfig(**values)


def test_trio_is_paired_across_ratios_and_h3_keeps_positive_full_ce() -> None:
    records, audit = prepare_history([_balanced_row()], TOKENIZER, _wide_config())

    by_variant = {
        variant: {(record["decision_id"], record["ratio"]): record for record in rows}
        for variant, rows in records.items()
    }
    assert by_variant["H1"].keys() == by_variant["H2"].keys() == by_variant["H3"].keys()
    assert {ratio for _, ratio in by_variant["H1"]} == {8, 12}
    assert {row["metadata"]["decision_type"] for row in records["H1"]} == {
        "non_tool_response",
        "terminal_stop",
        "tool_call",
    }
    assert any(row["metadata"]["state_revision"] for row in records["H1"])
    assert any(row["metadata"]["post_tool"] for row in records["H1"])

    for key, h1 in by_variant["H1"].items():
        h2, h3 = by_variant["H2"][key], by_variant["H3"][key]
        assert h1["target_ids"] == h2["target_ids"] == h3["target_ids"]
        assert h2["memory"] == h3["memory"]
        assert h1["weight"] == h2["weight"] == h3["weight"] == 1.0
        assert len(h3["target_weights"]) == len(h3["target_ids"])
        assert all(weight > 0 for weight in h3["target_weights"])
        assert h3["metadata"]["target_weighting"]["eos_weighted_token_count"] == 1
    tool_h3 = next(
        row for row in records["H3"] if row["metadata"]["decision_type"] == "tool_call"
    )
    assert tool_h3["metadata"]["target_weighting"]["matched_argument_value_count"] == 1
    assert tool_h3["metadata"]["target_weighting"]["argument_weighted_token_count"] > 0
    assert audit["complete"] == 1
    assert audit["trio.decisions"] == 3
    assert audit["H1.records"] == audit["H2.records"] == audit["H3.records"] == 6


def test_a_view_keeps_cross_cutoff_tool_as_gist_and_raw() -> None:
    decision = tuple(iter_decisions(_row()))[-1]
    prepared = _prepare_a_view(decision, TOKENIZER, _wide_config())
    overlap = prepared.metadata["raw_gist_overlap_event_ids"]
    current_tool_event = next(
        event for event in decision.store.events if "call-2" in event.tool_call_ids
    )

    assert prepared.metadata["version"] == A_VIEW_VERSION
    assert current_tool_event.event_id in overlap
    assert current_tool_event.event_id in prepared.memory.view.gist_event_ids
    assert current_tool_event.event_id in prepared.memory.view.raw_event_ids
    assert prepared.metadata["unrepresented_eligible_source_count"] == 0


def test_joint_failure_emits_no_partial_h1_h2_h3_rows() -> None:
    config = _wide_config(variants=("H1", "H2", "H3"), max_target_tokens=1)
    records, audit = prepare_history([_balanced_row()], TOKENIZER, config)

    assert records == {"H1": [], "H2": [], "H3": []}
    assert sum(value for key, value in audit.items() if key.startswith("trio.skipped.")) > 0
    assert not any(key.endswith(".records") for key in audit)


def test_h0_selection_is_independent_and_uses_static_recent_tool_one() -> None:
    first = _row(session_id="old-selected")
    second = _balanced_row(session_id="not-old")
    first["history_variants"] = ["H0"]
    second["history_variants"] = ["H1", "H2", "H3"]
    records, _ = prepare_history([first, second], TOKENIZER, _wide_config())

    assert records["H0"]
    assert all(row["session_key"].endswith('"old-selected"]') for row in records["H0"])
    assert records["H1"]
    assert all(row["session_key"].endswith('"not-old"]') for row in records["H1"])
    assert all(row["metadata"]["view_policy"] == "static-recent-tool-one" for row in records["H0"])


def test_a_view_matches_current_sibling_controller_when_checkout_is_available(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    sibling = repo.parent / "c2kv-a-runtime"
    controller_file = sibling / "benchmarks" / "memory_runtime" / "event_native_s0_policy.py"
    if not controller_file.is_file():
        pytest.skip("current A runtime sibling checkout is unavailable")

    decision = tuple(iter_decisions(_row()))[-1]
    expected = _prepare_a_view(decision, TOKENIZER, _wide_config())
    payload = {
        "session_id": decision.store.session_id,
        "decision_key": decision.decision_id,
        "messages": [message.to_dict() for message in decision.store.messages],
        "tools": list(decision.tools),
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script = r'''
import json, sys
from pathlib import Path
root, payload_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(root / "benchmarks"), str(root / "python")]
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE, HISTORY_BUDGET_DEFINITION, POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, tokenize=True,
                            add_generation_prompt=False, **kwargs):
        text = ("<tools>" + json.dumps(tools, ensure_ascii=False, sort_keys=True) +
                "</tools>") if tools else ""
        for message in messages:
            text += ("<" + message["role"] + ">" +
                     json.dumps(message, ensure_ascii=False, sort_keys=True) +
                     "</end>" + chr(0x10FFFE))
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(character) for character in text]
packing = dict(ratios=[8,12], recent_tool_events=1, max_chunk_tokens=768,
    chunk_overlap=64, max_chunks=48, max_encoder_tokens=100000,
    max_system_tokens=100000, max_workspace_tokens=100000,
    max_target_tokens=4096, max_sequence_tokens=100000)
policy = dict(mode="persistent", history_budget_bytes=100000,
    workspace_budget_bytes=100000, lease_decisions=0, max_retrieved_events=2,
    kv_bytes_per_token=1, source_commit=POLICY_SOURCE_COMMIT,
    history_budget_definition=HISTORY_BUDGET_DEFINITION,
    workspace_budget_definition=WORKSPACE_BUDGET_DEFINITION,
    current_input_baseline=CURRENT_INPUT_BASELINE)
payload = json.loads(payload_path.read_text(encoding="utf-8"))
prepared = EventNativeS0Controller(Tokenizer(), packing=packing, policy=policy).prepare(
    payload, ratio=8, max_new_tokens=32)
memory = prepared.memory
print(json.dumps(dict(
    gist=list(memory.view.gist_event_ids), raw=list(memory.view.raw_event_ids),
    system=list(memory.system_input_ids), workspace=list(memory.workspace_input_ids),
    chunks=[dict(event_id=c.event_id, part_index=c.part_index,
                 source_indices=list(c.source_indices), token_start=c.source_token_start,
                 token_end=c.source_token_end, token_ids=list(c.token_ids))
            for c in memory.chunks])))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(sibling), str(payload_path)],
        cwd=sibling,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    actual = json.loads(completed.stdout)
    assert actual["gist"] == list(expected.memory.view.gist_event_ids)
    assert actual["raw"] == list(expected.memory.view.raw_event_ids)
    assert actual["system"] == list(expected.memory.system_input_ids)
    assert actual["workspace"] == list(expected.memory.workspace_input_ids)
    assert actual["chunks"] == [
        {
            "event_id": chunk.event_id,
            "part_index": chunk.part_index,
            "source_indices": list(chunk.source_indices),
            "token_start": chunk.source_token_start,
            "token_end": chunk.source_token_end,
            "token_ids": list(chunk.token_ids),
        }
        for chunk in expected.memory.chunks
    ]


def test_vendored_a_runtime_manifest_self_verifies() -> None:
    repo = Path(__file__).resolve().parents[2]
    manifest_path = (
        repo / "python" / "next_compression" / "vendor" / "a_runtime" /
        "SOURCE_MANIFEST.json"
    )
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == A_VIEW_VENDOR_MANIFEST_SHA256
    manifest = json.loads(manifest_bytes)
    controller = next(
        item
        for item in manifest["files"]
        if item["upstream_path"]
        == "benchmarks/memory_runtime/event_native_s0_policy.py"
    )
    assert controller["upstream_sha256"] == A_VIEW_SOURCE_SHA256
    for item in manifest["files"]:
        vendored = repo / item["vendored_path"]
        assert hashlib.sha256(vendored.read_bytes()).hexdigest() == item[
            "vendored_sha256"
        ]


def test_type_balancing_equalizes_effective_tool_and_stop_mass() -> None:
    tool_heavy = _balanced_row(session_id="balance")
    records, audit = prepare_history(
        [tool_heavy], TOKENIZER, _wide_config()
    )
    mass = defaultdict(float)
    seen = set()
    for row in records["H1"]:
        key = row["decision_id"]
        if key in seen or row["ratio"] != 8:
            continue
        seen.add(key)
        mass[row["metadata"]["decision_type"]] += row["weight"]
    assert len(set(mass.values())) == 1
    assert mass == {
        "non_tool_response": 1.0,
        "terminal_stop": 1.0,
        "tool_call": 1.0,
    }
    assert audit["trio.balance_dropped.type.tool_call"] == 1


def test_joint_pack_failure_refills_same_type_before_balancing(monkeypatch) -> None:
    original = history_module._attempt_trio
    attempted_tool_ids = []

    def fail_first_tool(item, tokenizer, config):
        if item.decision_type == "tool_call":
            attempted_tool_ids.append(item.decision.decision_id)
            if len(attempted_tool_ids) == 1:
                return item, None, None, None, "PackingBudgetError"
        return original(item, tokenizer, config)

    monkeypatch.setattr(history_module, "_attempt_trio", fail_first_tool)
    records, audit = prepare_history(
        [_balanced_row(session_id="packing-refill")],
        TOKENIZER,
        _wide_config(),
    )

    accepted = {
        kind: {
            row["decision_id"]
            for row in records["H1"]
            if row["metadata"]["decision_type"] == kind
        }
        for kind in ("tool_call", "non_tool_response", "terminal_stop")
    }
    assert {kind: len(ids) for kind, ids in accepted.items()} == {
        "tool_call": 1,
        "non_tool_response": 1,
        "terminal_stop": 1,
    }
    assert len(attempted_tool_ids) == 2
    assert accepted["tool_call"] == {attempted_tool_ids[1]}
    assert audit["trio.skipped.type.tool_call"] == 1
    assert audit["trio.balance_attempted.type.tool_call"] == 2
    assert audit["trio.balance_effective_quota_per_type"] == 1


def test_missing_action_type_drops_unbalanced_batch_with_audit() -> None:
    records, audit = prepare_history([_row()], TOKENIZER, _wide_config())

    assert records["H1"] == records["H2"] == records["H3"] == []
    assert audit["trio.balance_missing_type.non_tool_response"] == 1
    assert audit["trio.balance_batches_dropped_missing_type"] == 1


def test_explicit_ids_bypass_type_balancing_but_keep_joint_pack_gate() -> None:
    decision_id = tuple(iter_decisions(_row()))[-1].decision_id
    records, audit = prepare_history(
        [_row()],
        TOKENIZER,
        _wide_config(h1_decision_ids=frozenset({decision_id})),
    )

    assert len(records["H1"]) == len(records["H2"]) == len(records["H3"]) == 2
    assert {row["decision_id"] for row in records["H1"]} == {decision_id}
    assert audit["trio.balance_bypassed_explicit_ids"] == 1


def test_a_specific_history_byte_budget_does_not_filter_frozen_h0() -> None:
    decision_id = tuple(iter_decisions(_row()))[-1].decision_id
    config = _wide_config(
        variants=("H0",),
        history_budget_bytes=1,
        h0_decision_ids=frozenset({decision_id}),
    )
    records, audit = prepare_history([_row()], TOKENIZER, config)

    assert len(records["H0"]) == 2
    assert audit["H0.records"] == 2


def test_cpu_workers_preserve_deterministic_records_and_audit() -> None:
    rows = [
        _balanced_row(session_id="worker-a"),
        _balanced_row(session_id="worker-b"),
    ]
    serial_records, serial_audit = prepare_history(
        rows, TOKENIZER, _wide_config(workers=1)
    )
    parallel_records, parallel_audit = prepare_history(
        rows, TOKENIZER, _wide_config(workers=2)
    )

    assert parallel_records == serial_records
    assert {key: value for key, value in parallel_audit.items() if key != "workers"} == {
        key: value for key, value in serial_audit.items() if key != "workers"
    }
    assert parallel_audit["workers"] == 2
