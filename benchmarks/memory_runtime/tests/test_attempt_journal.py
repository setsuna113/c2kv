"""Durability, pairing, concurrency, and privacy tests for attempt journals."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

from memory_runtime import attempt_journal as journal_module
from memory_runtime.attempt_journal import (
    AttemptJournal,
    attempt_journal_path,
    read_attempt_journal,
    summarize_attempt_journal,
)


def test_completed_and_failed_records_keep_only_allowlisted_context_and_usage(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    generation = journal.start(
        "generation", 1, "request-1",
        {
            "benchmark": "bfcl", "task_id": "multi_turn_base_183",
            "user_turn": 2, "step": 1, "attempt": 0,
            "messages": "PRIVATE_MESSAGE", "url": "PRIVATE_URL",
            "key": "PRIVATE_KEY",
        },
    )
    journal.finish(
        generation, "completed",
        usage={"prompt_tokens": 10, "completion_tokens": 2,
               "total_tokens": 12, "private_usage": "PRIVATE_USAGE"},
    )
    extraction = journal.start(
        "extraction", 1, "request-1",
        {"task_id": "multi_turn_base_183", "decision_id": "2:1"},
    )
    journal.finish(extraction, "failed")

    loaded = read_attempt_journal(path)
    assert loaded["truncated_tail"] is False
    assert len(loaded["records"]) == 4
    assert loaded["records"][0]["eval_context"] == {
        "benchmark": "bfcl", "task_id": "multi_turn_base_183",
        "user_turn": 2, "step": 1, "attempt": 0,
    }
    assert loaded["records"][1]["usage"] == {
        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    raw = path.read_text(encoding="utf-8")
    assert all(secret not in raw for secret in (
        "PRIVATE_MESSAGE", "PRIVATE_URL", "PRIVATE_KEY", "PRIVATE_USAGE"))

    summary = summarize_attempt_journal(path)
    assert (summary["started"], summary["finished"], summary["pending"]) == (2, 2, 0)
    assert (summary["completed"], summary["failed"]) == (1, 1)
    assert summary["finished_with_usage"] == 1
    assert summary["finished_usage_totals"] == {
        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert summary["by_kind"]["generation"]["completed"] == 1
    assert summary["by_kind"]["extraction"]["failed"] == 1


def test_finish_rejects_non_allowlisted_token_shapes(tmp_path):
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    handle = journal.start("generation", 1, "request", {})
    with pytest.raises(ValueError, match="nonnegative integer"):
        journal.finish(handle, "completed", usage={"prompt_tokens": -1})
    with pytest.raises(ValueError, match="nonnegative integer"):
        journal.finish(handle, "completed", usage={"prompt_tokens": True})
    assert summarize_attempt_journal(journal.path)["pending"] == 1


def test_concurrent_starts_have_unique_ids_and_complete_json_lines(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)

    def start(index):
        return journal.start(
            "generation", index, f"request-{index}", {"task_id": f"task-{index}"})

    with ThreadPoolExecutor(max_workers=16) as pool:
        handles = list(pool.map(start, range(1, 65)))
    assert len({handle.attempt_uid for handle in handles}) == 64
    assert path.read_bytes().endswith(b"\n")
    summary = summarize_attempt_journal(path)
    assert summary["started"] == 64
    assert summary["finished"] == 0
    assert summary["pending"] == 64


def test_terminated_child_leaves_fsynced_started_attempt_pending(tmp_path):
    path = tmp_path / "child-attempts.jsonl"
    code = "\n".join((
        "import sys, time",
        "from pathlib import Path",
        "from benchmarks.memory_runtime.attempt_journal import AttemptJournal",
        "journal = AttemptJournal(Path(sys.argv[1]))",
        "journal.start('generation', 1, 'child-request', {'task_id': 'task-1'})",
        "print('started-and-fsynced', flush=True)",
        "time.sleep(60)",
    ))
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "started-and-fsynced"
        proc.terminate()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    summary = summarize_attempt_journal(path)
    assert summary["started"] == 1
    assert summary["finished"] == 0
    assert summary["pending"] == 1
    assert summary["pending_token_accounting"] == "unknown"


def test_trailing_half_line_is_reported_and_never_completes_attempt(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    handle = journal.start("extraction", 1, "request", {"tool_step_index": 3})
    with path.open("ab") as stream:
        stream.write(json.dumps({
            "schema": "a-runtime-attempt-journal-v1",
            "event": "finished", "attempt_uid": handle.attempt_uid,
        }).encode("utf-8")[:30])
        stream.flush()
    loaded = read_attempt_journal(path)
    assert loaded["truncated_tail"] is True
    assert len(loaded["records"]) == 1
    summary = summarize_attempt_journal(path)
    assert summary["pending"] == 1
    assert summary["finished"] == 0
    assert summary["truncated_tail"] is True


def test_only_trailing_half_line_yields_no_fabricated_record(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"schema":"a-runtime-attempt-journal-v1","event":"started"')
    loaded = read_attempt_journal(path)
    assert loaded == {
        "schema": "a-runtime-attempt-journal-v1",
        "records": [],
        "truncated_tail": True,
    }
    summary = summarize_attempt_journal(path)
    assert summary["started"] == summary["finished"] == 0
    assert summary["truncated_tail"] is True


def test_missing_journal_is_not_interpreted_as_zero_attempts(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_attempt_journal(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        summarize_attempt_journal(tmp_path / "missing.jsonl")


def test_append_retries_short_os_write_before_fsync(monkeypatch, tmp_path):
    original_write = journal_module.os.write
    writes = []

    def short_write(descriptor, data):
        chunk = data[:max(1, len(data) // 2)]
        writes.append(len(chunk))
        return original_write(descriptor, chunk)

    monkeypatch.setattr(journal_module.os, "write", short_write)
    path = tmp_path / "attempts.jsonl"
    AttemptJournal(path).start("generation", 1, "request", {"task_id": "task"})
    assert len(writes) > 1
    assert len(read_attempt_journal(path)["records"]) == 1


def test_attempt_journal_path_avoids_proxy_log_glob(tmp_path):
    request_log = tmp_path / "proxy_full_34100.jsonl"
    assert attempt_journal_path(request_log) == tmp_path / "attempts_proxy_full_34100.jsonl"
    assert not attempt_journal_path(request_log).name.startswith("proxy_")
