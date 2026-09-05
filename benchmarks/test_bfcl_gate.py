"""BFCL terminal gate + expected-count derivation (offline, no bfcl_eval)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import terminal_check  # noqa: E402
from adapters import bfcl_adapter  # noqa: E402


def _results(root: Path, handler: str, family: str, category: str, ids):
    path = root / "result" / handler / family / f"BFCL_v4_{category}_result.json"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in ids), encoding="utf-8")


def test_check_bfcl_finds_non_base_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_check, "GORILLA", tmp_path)
    _results(tmp_path, "c2kv-hybrid", "memory", "memory", ["memory_0", "memory_1", "memory_2"])
    assert terminal_check.check_bfcl(3, "", handler="c2kv-hybrid", category="memory") == 0
    assert terminal_check.check_bfcl(4, "", handler="c2kv-hybrid", category="memory") == 1
    # the default category still resolves the multi_turn family
    _results(tmp_path, "c2kv-hybrid", "multi_turn", "multi_turn_base", ["multi_turn_base_0"])
    assert terminal_check.check_bfcl(1, "", handler="c2kv-hybrid") == 0
    # a category never generated is "artifacts not found", not "complete"
    assert terminal_check.check_bfcl(200, "", handler="c2kv-hybrid",
                                     category="multi_turn_long_context") == 2


def test_check_bfcl_id_exact_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_check, "GORILLA", tmp_path)
    _results(tmp_path, "c2kv-full", "memory", "memory", ["memory_0", "memory_2"])
    assert terminal_check.check_bfcl(None, "memory_0,memory_2", handler="c2kv-full",
                                     category="memory") == 0
    assert terminal_check.check_bfcl(None, "memory_0,memory_1", handler="c2kv-full",
                                     category="memory") == 1


def test_expected_count_reads_category_data_file(tmp_path, capsys):
    data = tmp_path / "bfcl_eval" / "data"
    data.mkdir(parents=True)
    (data / "BFCL_v4_memory.json").write_text("{}\n{}\n\n{}\n", encoding="utf-8")
    assert bfcl_adapter.expected_count("memory", root=tmp_path) == 3
    assert bfcl_adapter.expected_count("web_search", root=tmp_path) == 200  # fallback
    assert "falls back" in capsys.readouterr().err
