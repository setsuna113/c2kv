"""BFCL terminal gate + expected-count derivation (offline, no bfcl_eval)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import terminal_check  # noqa: E402
from adapters import bfcl_adapter  # noqa: E402
from bfcl_completion import bfcl_row_is_terminal, completion_kind  # noqa: E402


@pytest.mark.parametrize("message,kind", [
    ("upstream 400: The input (138237 tokens) is longer than the model's context length (131072 tokens).", "context_overflow"),
    ("upstream 400: The input (138237 tokens) is longer than the model\\'s context length (131072 tokens).", "context_overflow"),
    ("HiAgent requested nonexistent completed subgoals: [1]", "hiagent_invalid_retrieval"),
    ("HiAgent requested an already revealed trajectory without advancing", "hiagent_invalid_retrieval"),
    ("upstream 502: Connection refused", "incomplete"),
    ("TimeoutError: upstream timed out", "incomplete"),
    ("AttributeError: 'NoneType' object has no attribute 'session_id'", "incomplete"),
    ("ValueError: an unknown method implementation bug", "incomplete"),
])
def test_completion_distinguishes_terminal_failures_from_infrastructure(message, kind):
    row = {"id": "multi_turn_base_0", "result": "error", "traceback": message}
    assert completion_kind(row) == kind
    assert completion_kind({"id": row["id"], "traceback": message}) == "incomplete"


def _legacy_fc_decode_row(task_id="multi_turn_base_0"):
    tool_text = '<tool_call>{"name":"lookup","arguments":{"city":"X"}}</tool_call>'
    return {
        "id": task_id,
        "result": [[tool_text]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": tool_text},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }


def test_fc_guard_rejects_only_legacy_handler_contract_error():
    legacy = _legacy_fc_decode_row()
    assert completion_kind(legacy) == "model_output"  # prompt-mode default
    assert completion_kind(legacy, fc_model=True) == "incomplete"
    assert not bfcl_row_is_terminal(legacy, fc_model=True)

    malformed_actor = {"id": legacy["id"], "result": legacy["result"]}
    assert completion_kind(malformed_actor, fc_model=True) == "model_output"
    assert completion_kind({"id": legacy["id"], "result": []},
                           fc_model=True) == "model_output"

    native_final = {
        "id": "multi_turn_base_164", "result": [[[ {"lookup": "{}"},
            "The requested answer is ready." ]]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": [{"lookup": "{}"}]},
            {"role": "handler_log", "model_response_decoded": ["lookup()"]},
            {"role": "tool", "content": "ok"},
        ], "step_1": [
            {"role": "assistant", "content": "The requested answer is ready."},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }
    assert completion_kind(native_final, fc_model=True) == "model_output"
    assert bfcl_row_is_terminal(native_final, fc_model=True)

    malformed_with_error = {
        "id": legacy["id"], "result": [["<tool_call>bad</tool_call>"]],
        "inference_log": [{"step_0": [
            {"role": "assistant", "content": "<tool_call>bad</tool_call>"},
            {"role": "handler_log", "error": "'str' object has no attribute 'items'"},
        ]}],
    }
    assert completion_kind(malformed_with_error, fc_model=True) == "model_output"


def test_fc_evaluate_refuses_old_handler_row_before_official_scorer(
        tmp_path, monkeypatch):
    task_id = "multi_turn_base_0"
    path = _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base",
                    [_legacy_fc_decode_row(task_id)])
    original = path.read_bytes()
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(bfcl_adapter, "official_category_ids",
                        lambda category: {"multi_turn_base": [task_id]})
    monkeypatch.setattr(bfcl_adapter, "run_cli",
                        lambda argv: pytest.fail("official scorer must not run"))

    with pytest.raises(RuntimeError, match="evaluation requires valid unique completions"):
        bfcl_adapter.run_bfcl("http://proxy/v1", mode="evaluate",
                              project_root=tmp_path)

    assert path.read_bytes() == original
    receipt = next((tmp_path / "measurement" / "bfcl_completion").rglob("ledger.json"))
    assert json.loads(receipt.read_text())["legacy_fc_decode_task_ids"] == [task_id]


def test_known_failures_reach_official_scorer_unchanged_without_refill(tmp_path, monkeypatch):
    ids = ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_base_2"]
    failures = [
        "The input (138237 tokens) is longer than the model's context length (131072 tokens).",
        "HiAgent requested nonexistent completed subgoals: [1]",
        "HiAgent requested an already revealed trajectory without advancing",
    ]
    rows = [{"id": task, "result": "Error", "traceback": failure}
            for task, failure in zip(ids, failures)]
    path = _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", rows)
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(bfcl_adapter, "official_category_ids", lambda category: {"multi_turn_base": ids})
    calls = []

    def official(argv):
        calls.append(argv[0])
        if argv[0] == "evaluate":
            assert [json.loads(line) for line in path.read_text().splitlines()] == rows
            _score(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base",
                   {"accuracy": 0.0, "correct_count": 0, "total_count": 3})

    monkeypatch.setattr(bfcl_adapter, "run_cli", official)
    summary = bfcl_adapter.run_bfcl("http://proxy/v1", project_root=tmp_path)
    assert calls == ["generate", "evaluate"]
    assert summary["n_total"] == summary["n_scored"] == 3
    assert summary["completion_ledger"]["remaining"] == []
    assert set(summary["completion_ledger"]["terminal_failures"]) == set(ids)
    assert summary["completion_ledger"]["refill_rounds_used"] == 0
    assert summary["semantic_score"] == 0


def test_shared_completion_classifier_matches_bundled_runtime():
    root = Path(__file__).resolve().parents[1]
    assert (root / "benchmarks/bfcl_completion.py").read_bytes() == (
        root / "experiments/history_system/runtime/benchmarks/bfcl_completion.py").read_bytes()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item if isinstance(item, dict) else {"id": item, "result": []}
            for item in ids]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _score(root: Path, handler: str, family: str, category: str, header):
    path = root / "score" / handler / family / f"BFCL_v4_{category}_score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return path


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
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {category: ["multi_turn_base_7"]})
    monkeypatch.setattr(bfcl_adapter, "run_cli", seen["argv"].append)
    monkeypatch.setattr(terminal_check, "check_bfcl", check)
    monkeypatch.setattr(
        bfcl_adapter, "collect_score_summary",
        lambda root, handler, counts: {
            "n": sum(counts.values()), "n_total": sum(counts.values()),
            "n_scored": sum(counts.values()), "correct_count": 1,
            "semantic_score": 1.0, "official_score_headers": [], "scored": True,
        })

    _results(isolated, "c2kv-full", "multi_turn", "multi_turn_base",
             ["multi_turn_base_7"])
    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", categories="multi_turn_base",
        run_ids=["multi_turn_base_7", "multi_turn_base_7"],
        handler_name="c2kv-full", project_root=isolated,
    )

    assert seen["root_at_import"] == str(isolated.resolve())
    assert not (shared / "test_case_ids_to_generate.json").exists()
    assert json.loads((isolated / "test_case_ids_to_generate.json").read_text()) == {
        "multi_turn_base": ["multi_turn_base_7"]}
    assert seen["check"][2]["root"] == isolated.resolve()
    assert summary["n"] == summary["n_scored"] == 1
    assert summary["semantic_score"] == 1.0
    assert summary["bfcl_project_root"] == str(isolated.resolve())
    assert "BFCL_PROJECT_ROOT" not in os.environ


def test_expected_count_uses_official_collection_mapping(monkeypatch):
    package = ModuleType("bfcl_eval")
    package.__path__ = []
    utils = ModuleType("bfcl_eval.utils")
    utils.parse_test_category_argument = lambda categories: ["part_a", "part_b"]
    utils.load_dataset_entry = lambda category, **kwargs: [
        {"id": f"{category}_{index}"}
        for index in range(2 if category == "part_a" else 3)
    ]
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", utils)

    assert bfcl_adapter.official_category_counts("collection") == {
        "part_a": 2, "part_b": 3}
    assert bfcl_adapter.expected_count("collection") == 5


def test_non_scoring_category_is_rejected(monkeypatch):
    package = ModuleType("bfcl_eval")
    package.__path__ = []
    utils = ModuleType("bfcl_eval.utils")
    utils.parse_test_category_argument = lambda categories: ["format_sensitivity"]
    utils.load_dataset_entry = lambda category, **kwargs: [{"id": "unused"}]
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", utils)

    with pytest.raises(ValueError, match="non-scoring"):
        bfcl_adapter.expected_count("format_sensitivity")


def test_collect_score_summary_reads_and_aggregates_official_headers(tmp_path):
    first = _score(tmp_path, "c2kv-full", "multi_turn", "part_a", {
        "accuracy": 0.5, "correct_count": 1, "total_count": 2})
    second = _score(tmp_path, "c2kv-full", "multi_turn", "part_b", {
        "accuracy": 1.0, "correct_count": 1, "total_count": 1})

    summary = bfcl_adapter.collect_score_summary(
        tmp_path, "c2kv-full", {"part_a": 2, "part_b": 1})

    assert summary["n"] == summary["n_scored"] == 3
    assert summary["correct_count"] == 2
    assert summary["semantic_score"] == pytest.approx(2 / 3)
    assert [row["path"] for row in summary["official_score_headers"]] == [
        str(first), str(second)]


def test_collect_score_summary_rejects_missing_or_wrong_denominator(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})

    _score(tmp_path, "c2kv-full", "multi_turn", "multi_turn_base", {
        "accuracy": 0.5, "correct_count": 1, "total_count": 2})
    with pytest.raises(RuntimeError, match="denominator mismatch"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})


def test_collect_score_summary_rejects_inconsistent_accuracy(tmp_path):
    _score(tmp_path, "c2kv-full", "multi_turn", "multi_turn_base", {
        "accuracy": 1.0, "correct_count": 0, "total_count": 1})
    with pytest.raises(RuntimeError, match="accuracy mismatch"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})


def test_generate_mode_does_not_claim_scored_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {"multi_turn_base": ["multi_turn_base_0"]})
    _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base",
             ["multi_turn_base_0"])
    argv = []
    monkeypatch.setattr(bfcl_adapter, "run_cli", argv.append)
    monkeypatch.setattr(terminal_check, "check_bfcl", lambda *args, **kwargs: 0)

    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", mode="generate", project_root=tmp_path)

    assert summary["n_generated"] == summary["n_total"] == 1
    assert summary["scored"] is False
    assert "n" not in summary and "n_scored" not in summary
    assert "semantic_score" not in summary
    assert argv == [bfcl_adapter.generate_argv("c2kv-hf", "multi_turn_base")]


def test_check_bfcl_requires_valid_requested_rows(tmp_path):
    _results(tmp_path, "c2kv-full", "memory", "memory", [
        {"id": "memory_0", "result": []},
        {"id": "memory_1", "traceback": "transport failed"},
        {"id": "memory_2"},
    ])
    assert terminal_check.check_bfcl(
        None, "memory_0,memory_1,memory_2", handler="c2kv-full",
        category="memory", root=tmp_path,
    ) == 1

    _results(tmp_path, "c2kv-full", "memory", "memory", [
        {"id": "memory_0", "result": []},
        {"id": "memory_foreign", "result": []},
    ])
    assert terminal_check.check_bfcl(
        None, "memory_0", handler="c2kv-full", category="memory",
        root=tmp_path,
    ) == 1


def test_evaluate_only_snapshots_then_canonicalizes_latest_valid(tmp_path, monkeypatch):
    ids = ["multi_turn_base_0", "multi_turn_base_1"]
    result_path = _results(
        tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", [
            {"id": ids[0], "result": ["old"]},
            {"id": ids[0], "result": ["new"]},
            {"id": ids[1], "result": []},
            {"id": ids[1], "traceback": "failed"},
        ])
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {"multi_turn_base": ids})

    calls = []

    def run_cli(argv):
        calls.append(argv)
        assert argv[0] == "evaluate"
        _score(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", {
            "accuracy": 0.0, "correct_count": 0, "total_count": 2})

    monkeypatch.setattr(bfcl_adapter, "run_cli", run_cli)
    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", mode="evaluate", project_root=tmp_path)

    assert len(calls) == 1
    canonical = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert canonical == [
        {"id": ids[0], "result": ["new"]},
        {"id": ids[1], "result": []},
    ]
    receipt = Path(summary["completion_ledger"]["round_receipts"][0])
    raw = next((receipt.parent / "raw").rglob("*.json"))
    assert len(raw.read_text().splitlines()) == 4
    assert summary["completion_ledger"]["duplicate_rows"] == 2
    assert summary["completion_ledger"]["invalid_rows"] == 1


def test_default_does_not_refill_invalid_completion(tmp_path, monkeypatch):
    ids = ["multi_turn_base_0", "multi_turn_base_1"]
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {"multi_turn_base": ids})
    calls = []

    def run_cli(argv):
        calls.append(argv)
        _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", [
            {"id": ids[0], "result": []},
            {"id": ids[1], "traceback": "failed"},
        ])

    monkeypatch.setattr(bfcl_adapter, "run_cli", run_cli)
    with pytest.raises(RuntimeError, match="lacks valid unique completions"):
        bfcl_adapter.run_bfcl(
            "http://proxy/v1", mode="both", project_root=tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == "generate"


def test_explicit_refill_preserves_prior_valid_and_restores_selection(
        tmp_path, monkeypatch):
    ids = ["multi_turn_base_0", "multi_turn_base_1"]
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_ids",
        lambda category: {"multi_turn_base": ids})
    calls = []

    def run_cli(argv):
        calls.append(argv)
        if argv[0] == "generate" and len([call for call in calls if call[0] == "generate"]) == 1:
            _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", [
                {"id": ids[0], "result": ["first"]},
                {"id": ids[0], "result": ["duplicate"]},
                {"id": ids[1], "traceback": "failed"},
            ])
        elif argv[0] == "generate":
            assert json.loads((tmp_path / "test_case_ids_to_generate.json").read_text()) == {
                "multi_turn_base": [ids[1]]}
            # Simulate an official generator that overwrites the active file.
            _results(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", [
                {"id": ids[1], "result": ["refilled"]},
            ])
        else:
            assert json.loads((tmp_path / "test_case_ids_to_generate.json").read_text()) == {
                "multi_turn_base": ids}
            _score(tmp_path, "c2kv-hf", "multi_turn", "multi_turn_base", {
                "accuracy": 0.0, "correct_count": 0, "total_count": 2})

    monkeypatch.setattr(bfcl_adapter, "run_cli", run_cli)
    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", mode="both", project_root=tmp_path,
        max_refill_rounds=1)

    assert [call[0] for call in calls] == ["generate", "generate", "evaluate"]
    assert summary["completion_ledger"]["valid_unique"] == 2
    assert summary["completion_ledger"]["remaining"] == []
    assert summary["completion_ledger"]["refill_rounds_used"] == 1
    assert len(summary["completion_ledger"]["round_receipts"]) == 2
    rows = [json.loads(line) for line in next(
        (tmp_path / "result" / "c2kv-hf").rglob("*.json")).read_text().splitlines()]
    assert [row["id"] for row in rows] == ids
