"""Parity regression for the bundled event-native BFCL adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import terminal_check
from benchmarks.adapters import bfcl_adapter
from benchmarks.bfcl_completion import _complete_native_tool_calls


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_only_uses_valid_unique_rows_and_preserves_raw(tmp_path, monkeypatch):
    ids = ["multi_turn_base_0", "multi_turn_base_1"]
    result_path = (tmp_path / "result" / "c2kv-hf" / "multi_turn" /
                   "BFCL_v4_multi_turn_base_result.json")
    _write_jsonl(result_path, [
        {"id": ids[0], "result": ["old"]},
        {"id": ids[0], "result": ["latest"]},
        {"id": ids[1], "result": []},
        {"id": ids[1], "traceback": "failed"},
    ])
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {"multi_turn_base": ids})
    monkeypatch.setattr(bfcl_adapter, "summarize_audit", lambda path: {})

    calls = []

    def run_cli(argv):
        calls.append(argv)
        assert argv[0] == "evaluate"
        _write_jsonl(
            tmp_path / "score" / "c2kv-hf" / "multi_turn" /
            "BFCL_v4_multi_turn_base_score.json",
            [{"accuracy": 0.0, "correct_count": 0, "total_count": 2}],
        )

    monkeypatch.setattr(bfcl_adapter, "run_cli", run_cli)
    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", mode="evaluate", project_root=tmp_path)

    assert len(calls) == 1
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert rows == [
        {"id": ids[0], "result": ["latest"]},
        {"id": ids[1], "result": []},
    ]
    receipt = Path(summary["completion_ledger"]["receipt"])
    raw = next((receipt.parent / "raw").rglob("*.json"))
    assert len(raw.read_text().splitlines()) == 4
    assert summary["completion_ledger"]["duplicate_rows"] == 2
    assert summary["completion_ledger"]["invalid_rows"] == 1
    assert terminal_check.check_bfcl(
        2, ",".join(ids), handler="c2kv-hf", category="multi_turn_base",
        root=tmp_path,
    ) == 0


def test_terminal_method_and_context_failures_are_retained_for_official_zero(tmp_path):
    ids = ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_base_2", "multi_turn_base_3"]
    rows = [
        {"id": ids[0], "result": "error", "traceback": "The input (138237 tokens) is longer than the model's context length (131072 tokens)."},
        {"id": ids[1], "result": "error", "traceback": "HiAgent requested nonexistent completed subgoals: [1]"},
        {"id": ids[2], "result": "error", "traceback": "HiAgent requested an already revealed trajectory without advancing"},
        {"id": ids[3], "result": "error", "traceback": "upstream 502: Connection refused"},
    ]
    path = tmp_path / "result/c2kv-hf/multi_turn/BFCL_v4_multi_turn_base_result.json"
    _write_jsonl(path, rows)
    receipt = bfcl_adapter._canonicalize_completions(tmp_path, "c2kv-hf", {"multi_turn_base": ids})
    assert receipt["remaining"] == [ids[3]]
    assert set(receipt["terminal_failures"]) == set(ids[:3])
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows[:3]
    assert terminal_check.check_bfcl(3, ",".join(ids[:3]), handler="c2kv-hf", root=tmp_path) == 0
    raw = next((Path(receipt["receipt"]).parent / "raw").rglob("*.json"))
    assert [json.loads(line) for line in raw.read_text().splitlines()] == rows


def test_registered_fc_adapter_rejects_old_decode_error_but_keeps_model_text(tmp_path):
    ids = [f"multi_turn_base_{index}" for index in range(4)]
    valid_tool_text = '<tool_call>{"name":"lookup","arguments":{"city":"X"}}</tool_call>'
    old_error = {
        "id": ids[0], "result": [[valid_tool_text]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": valid_tool_text},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }
    model_text = {"id": ids[1], "result": [["<tool_call>bad</tool_call>"]]}
    native_final = {
        "id": ids[2], "result": [[[ {"lookup": "{}"}, "The answer is ready." ]]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": [{"lookup": "{}"}]},
            {"role": "tool", "content": "ok"},
        ], "step_1": [
            {"role": "assistant", "content": "The answer is ready."},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }
    malformed_with_error = {
        "id": ids[3], "result": [["<tool_call>bad</tool_call>"]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": "<tool_call>bad</tool_call>"},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }
    path = tmp_path / "result/c2kv-hf/multi_turn/BFCL_v4_multi_turn_base_result.json"
    _write_jsonl(path, [old_error, model_text, native_final, malformed_with_error])

    receipt = bfcl_adapter._canonicalize_completions(
        tmp_path, "c2kv-hf", {"multi_turn_base": ids})

    assert receipt["remaining"] == [ids[0]]
    assert receipt["invalid_rows"] == 1
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        model_text, native_final, malformed_with_error]
    raw = next((Path(receipt["receipt"]).parent / "raw").rglob("*.json"))
    assert [json.loads(line) for line in raw.read_text().splitlines()] == [
        old_error, model_text, native_final, malformed_with_error]


@pytest.mark.parametrize("text", [
    '<tool_call>{"name":"lookup","arguments":{"city":"X"}}</tool_call>',
    'No tool is needed.',
    '<tool_call>{"name":"lookup","arguments":{"city":"X"}',
    '<tool_call>{"name":"lookup","arguments":{"city":"X","city":"Y"}}</tool_call>',
    '<tool_call>{"name":"lookup","arguments":[]}</tool_call>',
])
def test_fc_guard_tool_block_evidence_matches_native_parser(text):
    from benchmarks.memory_runtime.event_native_draft import parse_native_draft

    assert _complete_native_tool_calls(text) == (
        parse_native_draft(text, call_id_prefix="audit").status == "tool_calls")
