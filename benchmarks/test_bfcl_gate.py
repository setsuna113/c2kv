"""BFCL terminal gate + expected-count derivation (offline, no bfcl_eval)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import terminal_check  # noqa: E402
from adapters import bfcl_adapter  # noqa: E402


def test_embedded_cli_returns_to_run_evaluation(monkeypatch):
    typer = pytest.importorskip("typer")
    cli = typer.Typer()
    completed = []

    @cli.command()
    def generate():
        completed.append("generate")

    @cli.command()
    def evaluate():
        completed.append("evaluate")

    monkeypatch.setitem(sys.modules, "bfcl_eval.__main__", SimpleNamespace(cli=cli))
    original_argv = list(sys.argv)
    bfcl_adapter.run_cli(["generate"])
    bfcl_adapter.run_cli(["evaluate"])
    assert completed == ["generate", "evaluate"]
    assert sys.argv == original_argv


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


def test_check_bfcl_explicit_root_ignores_stale_shared_results(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    isolated = tmp_path / "isolated"
    monkeypatch.setattr(terminal_check, "GORILLA", shared)
    _results(shared, "c2kv-full", "memory", "memory", ["memory_0"])
    _results(isolated, "c2kv-full", "memory", "memory", ["memory_1"])

    assert terminal_check.check_bfcl(
        None, "memory_1", handler="c2kv-full", category="memory",
        root=isolated,
    ) == 0
    assert terminal_check.check_bfcl(
        None, "memory_0", handler="c2kv-full", category="memory",
        root=isolated,
    ) == 1


def test_run_bfcl_sets_official_root_before_import_and_writes_ids_there(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    isolated = tmp_path / "isolated"
    shared.mkdir()
    monkeypatch.chdir(shared)
    monkeypatch.delenv("BFCL_PROJECT_ROOT", raising=False)
    seen = {"argv": []}

    def install(base_url, model, handler_name):
        seen["root_at_import"] = os.environ.get("BFCL_PROJECT_ROOT")

    def check(expected, run_ids, **kwargs):
        seen["check"] = (expected, run_ids, kwargs)
        return 0

    monkeypatch.setattr(bfcl_adapter, "install_handler", install)
    monkeypatch.setattr(bfcl_adapter, "expected_count", lambda category: 99)
    monkeypatch.setattr(bfcl_adapter, "run_cli", seen["argv"].append)
    monkeypatch.setattr(terminal_check, "check_bfcl", check)

    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", categories="memory", run_ids=["memory_7"],
        handler_name="c2kv-full", project_root=isolated,
    )

    assert seen["root_at_import"] == str(isolated.resolve())
    assert not (shared / "test_case_ids_to_generate.json").exists()
    assert json.loads((isolated / "test_case_ids_to_generate.json").read_text()) == {
        "memory": ["memory_7"]}
    assert seen["check"][2]["root"] == isolated.resolve()
    assert summary["bfcl_project_root"] == str(isolated.resolve())
    assert "BFCL_PROJECT_ROOT" not in os.environ


def test_expected_count_reads_category_data_file(tmp_path, capsys):
    data = tmp_path / "bfcl_eval" / "data"
    data.mkdir(parents=True)
    (data / "BFCL_v4_memory.json").write_text("{}\n{}\n\n{}\n", encoding="utf-8")
    assert bfcl_adapter.expected_count("memory", root=tmp_path) == 3
    assert bfcl_adapter.expected_count("web_search", root=tmp_path) == 200  # fallback
    assert "falls back" in capsys.readouterr().err
