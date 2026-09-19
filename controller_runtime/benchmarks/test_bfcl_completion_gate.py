"""Parity regression for the bundled event-native BFCL adapter."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks import terminal_check
from benchmarks.adapters import bfcl_adapter


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
