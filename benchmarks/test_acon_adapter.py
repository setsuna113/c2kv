"""Offline tests for adapters/acon_adapter.py (no subprocess, no network).

Covers: runner command lines and env, output-path derivation (must match
ACON's run.py / run_all.py rules verbatim), the QA and AppWorld collectors,
the terminal-state gate, and the loud failure on an unrecognised AppWorld
evaluation layout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import acon_adapter as A  # noqa: E402


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---- env / commands ---------------------------------------------------------

def test_runner_env_points_agent_at_proxy_v1(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://corp:3128")
    env = A.runner_env("http://127.0.0.1:34100/")
    assert env[A.BASE_URL_ENV] == "http://127.0.0.1:34100/v1"
    assert env[A.API_KEY_ENV] == "EMPTY"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_qa_command_uses_shipped_split_and_pins():
    cmd = A.qa_command("py", "c2kv-agent", "run_ab12", "test", 30, limit=5,
                       id_list_file=Path("/x/ids.txt"))
    assert cmd[:2] == ["py", "run.py"]
    assert cmd[cmd.index("--data_folder") + 1] == "data/nq_multi_8"
    assert cmd[cmd.index("--limit") + 1] == "5"
    assert cmd[cmd.index("--id_list_file") + 1] == str(Path("/x/ids.txt"))
    assert "--output_dir" not in cmd  # ignored upstream; path derived instead


def test_appworld_command_and_experiment_name():
    cmd = A.appworld_command("py", "org/model", "run_ab12", "test_normal", 50,
                             task_ids=["t1", "t2"])
    assert cmd[cmd.index("--split") + 1] == "test_normal"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[-3:] == ["--task_ids", "t1", "t2"]
    # run_all.py: model_name.replace("/", "_") + "_" + tag
    assert A.appworld_experiment("org/model", "run_ab12") == "org_model_run_ab12"


def test_output_paths_follow_runner_rules(tmp_path):
    # run.py: sanitised model/tag, dev -> test fold
    run_dir = A.qa_run_dir(tmp_path, "c2kv agent", "run/ab12", "dev")
    assert run_dir == (tmp_path / "experiments" / "smolagents" / "outputs"
                       / "c2kv-agent_run-ab12" / "test")
    # run_all.py output + appworld evaluate output (relative to runner cwd)
    assert A.appworld_run_dir(tmp_path, "m", "t", "test_normal") == (
        tmp_path / "experiments" / "appworld" / "outputs" / "m_t" / "test_normal")
    assert A.appworld_eval_path(tmp_path, "m", "t", "test_normal") == (
        tmp_path / "experiments" / "appworld" / "experiments" / "outputs" / "m_t"
        / "evaluations" / "test_normal.json")


def test_qa_expected_counts_shipped_file(tmp_path):
    data = tmp_path / "experiments" / "smolagents" / A.QA_DATA_FOLDER / "test.jsonl"
    _write_jsonl(data, [{"id": f"nq_multi8_test_{i}"} for i in range(7)])
    assert A.qa_expected(tmp_path, "test", None, None) == 7
    assert A.qa_expected(tmp_path, "dev", 3, None) == 3
    assert A.qa_expected(tmp_path, "test", 3, ["a", "b"]) == 2


# ---- QA collector -----------------------------------------------------------

def _qa_rows():
    return [
        {"id": "nq_multi8_test_1", "em": 0.5, "f1": 0.6, "iterations": 9, "success": True},
        {"id": "nq_multi8_test_2", "em": 0.0, "f1": 0.1, "iterations": 30, "success": False},
        {"id": "nq_multi8_test_3", "em": 1.0, "f1": 1.0, "iterations": 12, "success": True},
    ]


def test_collect_qa_rows_and_official_summary(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    (tmp_path / "summary.json").write_text(json.dumps({"avg_em": 0.5, "avg_f1": 0.5667, "total": 3}))
    summary = A.collect_qa(tmp_path, expected=3)
    assert summary["n"] == 3 and summary["n_clusters"] == 3
    assert summary["semantic_score"] == pytest.approx(0.5)
    assert summary["f1_mean"] == pytest.approx((0.6 + 0.1 + 1.0) / 3)
    assert summary["official_summary"]["total"] == 3
    assert summary["protocol_legal_rate"] is None  # code agent: no schema column


def test_collect_qa_terminal_gate(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    with pytest.raises(SystemExit, match="n_scored=3 < n_total=4"):
        A.collect_qa(tmp_path, expected=4)


def test_collect_qa_missing_predictions_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="predictions.jsonl"):
        A.collect_qa(tmp_path)


# ---- AppWorld collector -----------------------------------------------------

def test_appworld_per_task_recognised_shapes():
    by_dict = {"aggregate": {"tgc": 0.5}, "individual": {"t1": {"success": True}, "t2": {"success": False}}}
    assert A.appworld_per_task(by_dict) == {"t1": True, "t2": False}
    by_bool = {"tasks": {"t1": True, "t2": False}}
    assert A.appworld_per_task(by_bool) == {"t1": True, "t2": False}
    by_arrays = {"tasks": {"t1": {"passes": ["a"], "fails": []}, "t2": {"passes": [], "fails": ["x"]}}}
    assert A.appworld_per_task(by_arrays) == {"t1": True, "t2": False}
    by_list = {"results": [{"task_id": "t1", "passed": True}, {"task_id": "t2", "passed": False}]}
    assert A.appworld_per_task(by_list) == {"t1": True, "t2": False}


def test_appworld_per_task_unknown_layout_is_fatal():
    with pytest.raises(SystemExit, match="unrecognised appworld evaluation layout"):
        A.appworld_per_task({"tgc": 0.4, "sgc": 0.2})
    with pytest.raises(SystemExit):
        A.appworld_per_task({"individual": {"t1": {"score": 0.3}}})  # no boolean


def test_collect_appworld_joins_runner_results(tmp_path):
    eval_path = tmp_path / "evaluations" / "test_normal.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps({
        "tgc": 0.5, "sgc": 0.0,
        "individual": {"t1": {"success": True}, "t2": {"success": False}},
    }))
    run_dir = tmp_path / "outputs" / "m_t" / "test_normal"
    (run_dir / "task_t1").mkdir(parents=True)
    (run_dir / "task_t1" / "results.json").write_text(json.dumps(
        {"success": True, "iterations": 7, "termination_reason": "task_completed"}))
    (run_dir / "task_t2").mkdir(parents=True)
    # agent claimed success but the official scorer disagrees: semantic wins
    (run_dir / "task_t2" / "results.json").write_text(json.dumps(
        {"success": True, "iterations": 50, "termination_reason": "max_iterations"}))
    summary = A.collect_appworld(eval_path, run_dir, expected=2)
    assert summary["n"] == 2 and summary["semantic_score"] == pytest.approx(0.5)
    assert summary["official_aggregate"] == {"tgc": 0.5, "sgc": 0.0}


def test_collect_appworld_terminal_gate_and_missing_eval(tmp_path):
    eval_path = tmp_path / "test_normal.json"
    eval_path.write_text(json.dumps({"individual": {"t1": {"success": True}}}))
    with pytest.raises(SystemExit, match="n_scored=1 < n_total=168"):
        A.collect_appworld(eval_path, tmp_path / "none", expected=168)
    with pytest.raises(SystemExit, match="wrote no"):
        A.collect_appworld(tmp_path / "missing.json", tmp_path)


def test_run_dispatch_rejects_unknown_kind(tmp_path):
    with pytest.raises(SystemExit, match="unknown ACON benchmark kind"):
        A.run_kind("nope", "http://x", tmp_path)


# ---- cost join (the only adapter pair whose artefacts key the request log) ---

import proxy  # noqa: E402
import reqlog  # noqa: E402


def _session(task: str, turns: int = 2):
    """One ACON session as MemoryManager.dump_history writes it:
    [system, user, assistant, user, assistant, ...] (memory.py:112-168)."""
    session = [{"role": "system", "content": "SYSTEM PROMPT"},
               {"role": "user", "content": f"task {task}"}]
    for i in range(turns):
        session.append({"role": "assistant", "content": f"code {task} {i}"})
        session.append({"role": "user", "content": f"obs {task} {i}"})
    return session


def _dump_history(task_dir: Path, session):
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / A.HISTORY_FILE).write_text(json.dumps([session]), encoding="utf-8")


def _log_row(messages, **extra):
    row = {"status": "ok", "conv_id": proxy.conversation_id(messages)}
    row.update(extra)
    return row


def test_conversation_ids_are_the_two_the_proxy_sees(tmp_path):
    session = _session("t1")
    _dump_history(tmp_path, session)
    ids = A.conversation_ids(tmp_path / A.HISTORY_FILE)
    # first request: [system, user]; every later one: [system, user, assistant, ...]
    assert ids == [proxy.conversation_id(session[:2]),
                   proxy.conversation_id(session[:3])]
    # and the id really is stable once the assistant turn exists
    assert proxy.conversation_id(session[:5]) == ids[1]
    assert proxy.conversation_id(session) == ids[1]


def test_conversation_ids_missing_or_broken_history_is_empty(tmp_path):
    assert A.conversation_ids(tmp_path / "nope.json") == []
    (tmp_path / A.HISTORY_FILE).write_text("not json", encoding="utf-8")
    assert A.conversation_ids(tmp_path / A.HISTORY_FILE) == []
    (tmp_path / A.HISTORY_FILE).write_text('{"a": 1}', encoding="utf-8")
    assert A.conversation_ids(tmp_path / A.HISTORY_FILE) == []


def test_collect_qa_joins_cost_columns_from_the_request_log(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "predictions.jsonl", _qa_rows())
    sessions = {}
    for rec in _qa_rows():
        sessions[rec["id"]] = _session(rec["id"])
        _dump_history(A.qa_sample_dir(run_dir, rec["id"]), sessions[rec["id"]])
    first, second = "nq_multi8_test_1", "nq_multi8_test_2"
    log = tmp_path / "proxy.jsonl"
    rows = [
        _log_row(sessions[first][:2], wall_sec=1.0, gist_tokens=10,
                 original_tokens=100, n_docs=0, dropped_docs=0),
        _log_row(sessions[first][:3], wall_sec=2.0, gist_tokens=30,
                 original_tokens=300, n_docs=4, dropped_docs=1),
        _log_row(sessions[second][:2], wall_sec=0.5, gist_tokens=5,
                 original_tokens=50, n_docs=0, dropped_docs=0),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = A.collect_qa(run_dir, expected=3, request_log=log)
    # task 3 made no request -> no cost fields, so the means cover 2 tasks
    assert summary["wall_sec_mean"] == pytest.approx((3.0 + 0.5) / 2)
    assert summary["gist_tokens_mean"] == pytest.approx((40 + 5) / 2)
    assert summary["original_tokens_mean"] == pytest.approx((400 + 50) / 2)
    assert summary["cost_join"] == "joined: 2/3 tasks, 3/3 logged requests"
    # ...and the denominator of those means is a NUMBER in the summary, not
    # only prose: semantic_score covers 3 tasks, the cost means cover 2
    assert summary["n"] == 3 and summary["n_cost_joined"] == 2
    # the three joined fields metrics.aggregate does not mean
    assert summary["n_cost_requests"] == 3
    assert summary["n_docs_max"] == 4
    assert summary["dropped_docs_total"] == 1


def test_collect_qa_without_request_log_sets_no_cost_columns(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    summary = A.collect_qa(tmp_path, expected=3)
    assert summary["wall_sec_mean"] is None
    assert summary["gist_tokens_mean"] is None
    assert summary["cost_join"] == "not joinable: no request log for this run"
    assert summary["n_cost_joined"] == 0
    # nothing was measured: None, never a zero that reads as a measurement
    assert summary["n_cost_requests"] is None
    assert summary["n_docs_max"] is None
    assert summary["dropped_docs_total"] is None


def test_cost_join_returns_the_report_not_only_its_prose_line(tmp_path):
    """The summary needs the numeric denominator too, so cost_join hands back
    the whole reqlog report."""
    rows = [{"task_id": "t1"}, {"task_id": "t2"}]
    report = A.cost_join(rows, lambda tid: tmp_path / tid, None)
    assert report["n_rows"] == 2 and report["n_joined"] == 0
    assert (reqlog.cost_join_status(report)
            == "not joinable: no request log for this run")


def test_collect_qa_reports_an_unmatched_log_instead_of_a_number(tmp_path):
    """A history that does not describe what was sent must yield NOTHING."""
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "predictions.jsonl", _qa_rows())
    for rec in _qa_rows():
        _dump_history(A.qa_sample_dir(run_dir, rec["id"]), _session(rec["id"]))
    log = tmp_path / "proxy.jsonl"
    log.write_text(json.dumps({"status": "ok", "conv_id": "somethingelse",
                               "wall_sec": 9.0}) + "\n", encoding="utf-8")
    summary = A.collect_qa(run_dir, expected=3, request_log=log)
    assert summary["wall_sec_mean"] is None
    assert summary["cost_join"].startswith("not joinable: ")


def test_collect_appworld_joins_cost_columns(tmp_path):
    eval_path = tmp_path / "evaluations" / "test_normal.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps(
        {"individual": {"t1": {"success": True}, "t2": {"success": False}}}))
    run_dir = tmp_path / "outputs" / "m_t" / "test_normal"
    sessions = {t: _session(t) for t in ("t1", "t2")}
    for task, session in sessions.items():
        _dump_history(A.appworld_task_dir(run_dir, task), session)
    log = tmp_path / "proxy.jsonl"
    rows = [
        _log_row(sessions["t1"][:2], wall_sec=1.0, gist_tokens=8, n_docs=0),
        _log_row(sessions["t1"][:3], wall_sec=3.0, gist_tokens=24, n_docs=11,
                 dropped_docs=2),
        _log_row(sessions["t2"][:3], wall_sec=2.0, gist_tokens=16, n_docs=5),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = A.collect_appworld(eval_path, run_dir, expected=2, request_log=log)
    assert summary["wall_sec_mean"] == pytest.approx((4.0 + 2.0) / 2)
    assert summary["cost_join"] == "joined: 2/2 tasks, 3/3 logged requests"
    assert summary["n_cost_joined"] == 2 and summary["n_cost_requests"] == 3
    # t1's second request dropped 2 docs out of 11 — the fact that says the
    # task's own history was truncated by turn packing
    assert summary["n_docs_max"] == 11 and summary["dropped_docs_total"] == 2
