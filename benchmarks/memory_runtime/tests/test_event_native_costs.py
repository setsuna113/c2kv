"""Task-level accounting contracts for event-native step records."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_costs import (
    main,
    read_event_native_steps,
    summarize_event_native_steps,
)


def _stats(
    *,
    materialized: int = 11,
    system_prefill: int = 7,
    target_input: int = 13,
    system_prefix: int = 20,
    gist_prefix: int = 30,
    final_cache: int = 120,
    scope_system_before: int = 0,
    scope_system_after: int = 20,
    scope_gist_before: int = 0,
    scope_gist_after: int = 30,
    allocator_peak=None,
    raw_prefill: int = 9,
    reused_raw: int = 4,
    reused_gists: int = 1,
    reused_encoder: int = 8,
    reused_system: int = 3,
) -> dict:
    return {
        "materialized_encoder_tokens": materialized,
        "system_prefill_tokens": system_prefill,
        "target_input_tokens": target_input,
        "recomputed_raw_tokens": target_input,
        "raw_prefill_input_tokens": raw_prefill,
        "session_reused_raw_tokens": reused_raw,
        "session_reused_gist_chunks": reused_gists,
        "session_reused_encoder_tokens": reused_encoder,
        "session_reused_system_tokens": reused_system,
        "system_prefix_kv_logical_bytes": system_prefix,
        "gist_prefix_kv_logical_bytes": gist_prefix,
        "resident_prefix_kv_bytes": system_prefix + gist_prefix,
        "resident_kv_logical_bytes_after_raw_prefill": final_cache - 10,
        "resident_kv_logical_bytes_final": final_cache,
        "scope_system_kv_logical_bytes_before": scope_system_before,
        "scope_system_kv_logical_bytes_after": scope_system_after,
        "scope_gist_kv_logical_bytes_before": scope_gist_before,
        "scope_gist_kv_logical_bytes_after": scope_gist_after,
        "torch_allocator_peak_allocated_bytes": allocator_peak,
    }


def _attempt(
    uid: str,
    *,
    phase: str = "draft",
    status: str = "completed",
    discarded: bool = False,
    prompt: int | None = 40,
    completion: int | None = 3,
    stats: dict | None = None,
    attempt_index: int = 1,
) -> dict:
    usage = None
    if prompt is not None and completion is not None:
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return {
        "phase": phase,
        "status": status,
        "discarded": discarded,
        "planned_resident_prompt_tokens": prompt,
        "usage": usage,
        "generation": None if stats is None else {"stats": stats},
        "attempt_uid": uid,
        "attempt_index": attempt_index,
    }


def _record(
    session: str,
    decision: str,
    attempts: list[dict],
    *,
    status: str = "ok",
    timing: dict | None = None,
    runtime: float | None = None,
    session_cache: dict | None = None,
) -> dict:
    value = {
        "schema": "a-event-native-exact-step-v1",
        "status": status,
        "session_id": session,
        "decision_key": decision,
        "generation_trace": attempts,
    }
    if timing is not None:
        value["controller_timing"] = timing
    if runtime is not None:
        value["decision_runtime_seconds"] = runtime
    if session_cache is not None:
        value["session_cache_after"] = session_cache
    return value


def _session_cache(
    session: str,
    *,
    cpu_bytes: int,
    device_logical: int,
    device_backing: int,
    transfer_in: int,
    transfer_out: int,
) -> dict:
    return {
        "policy": "persistent",
        "session_id": session,
        "generation": 1,
        "cpu_memo_present": cpu_bytes > 0,
        "cpu_memo_logical_bytes": cpu_bytes,
        "device_raw_snapshot_present": device_backing > 0,
        "device_raw_snapshot_logical_bytes": device_logical,
        "device_raw_snapshot_backing_bytes": device_backing,
        "device_raw_snapshot_prefix_tokens": 5,
        "device_raw_snapshot_raw_tokens": 7,
        "last_transfer_bytes_in": transfer_in,
        "last_transfer_bytes_out": transfer_out,
        "last_transfer_bytes_total": transfer_in + transfer_out,
    }


def _session(summary: dict, name: str = "session-a") -> dict:
    return next(item for item in summary["sessions"] if item["session_id"] == name)


def _journal_start(
    journal: AttemptJournal,
    session: str,
    decision: str,
    *,
    index: int = 1,
):
    return journal.start(
        "generation",
        index,
        json.dumps([session, decision], separators=(",", ":")),
        {"task_id": session, "decision_id": decision},
    )


def _cache_op(
    op_id: str,
    *,
    kind: str = "extract",
    status: str = "completed",
    requested: int | None = 8,
    completed: int | None = 5,
    transfer: int | None = 16,
    logical: int | None = 32,
    source_entry_id: str | None = None,
    result_entry_id: str | None = None,
    consumers: list[str] | None = None,
) -> dict:
    return {
        "op_id": op_id,
        "kind": kind,
        "status": status,
        "input_tokens_requested": requested,
        "input_tokens_completed": completed,
        "transfer_bytes": transfer,
        "logical_bytes": logical,
        "source_entry_id": source_entry_id,
        "result_entry_id": result_entry_id,
        "consumer_placement_ids": [] if consumers is None else consumers,
    }


def _cache_trace(
    uid: str,
    session: str,
    decision: str,
    *,
    phase: str = "draft",
    status: str = "completed",
    context_complete: bool = True,
    ops: list[dict] | None = None,
    entries: list[dict] | None = None,
    placements: list[dict] | None = None,
) -> dict:
    return {
        "schema": "event-native-cache-trace-v1",
        "attempt_uid": uid,
        "session_id": session,
        "decision_key": decision,
        "phase": phase,
        "context_complete": context_complete,
        "status": status,
        "ops": [] if ops is None else ops,
        "entries": [] if entries is None else entries,
        "placements": [] if placements is None else placements,
        "workspace_source_group": "raw-source-group-a",
        "commit_status": "committed" if status == "completed" else "failed",
    }


def _cache_placement(
    placement_id: str,
    *,
    access_op_id: str,
    accessed_entry_id: str | None = "entry-extract",
    origin_extraction_op_id: str | None = "op-extract",
) -> dict:
    return {
        "placement_id": placement_id,
        "event_id": "event-a",
        "part_index": 0,
        "source_indices": [0],
        "source_token_start": 0,
        "source_token_end": 5,
        "source_position_start": 0,
        "position_ids": [0, 1, 2, 3, 4],
        "access_op_id": access_op_id,
        "accessed_entry_id": accessed_entry_id,
        "origin_extraction_op_id": origin_extraction_op_id,
    }


def test_mixed_known_and_failed_attempts_keep_strict_totals_unknown() -> None:
    known = _record(
        "session-a",
        "d1",
        [_attempt("uid-known", stats=_stats())],
        timing={"prepare_seconds": 0.2, "reconsider_seconds": 0.1},
    )
    failed = _record(
        "session-a",
        "d2",
        [_attempt("uid-failed", status="failed", prompt=None, completion=None)],
        status="failed",
    )

    session = _session(summarize_event_native_steps([known, failed]))

    prompt = session["costs"]["openai_resident_usage"]["prompt_tokens"]
    assert prompt == {
        "strict_total": None,
        "known_total": 40,
        "known_calls": 1,
        "unknown_calls": 1,
    }
    target = session["costs"]["actual_model_work"]["target_input_tokens"]
    assert target["strict_total"] is None
    assert target["known_total"] == 13
    assert target["unknown_calls"] == 1
    allocator = session["costs"]["logical_kv_peaks"][
        "torch_allocator_peak_allocated_bytes"
    ]
    assert allocator["strict_peak"] is None
    assert allocator["known_peak"] is None
    assert allocator["unknown_calls"] == 2
    prepare = session["controller_timing"]["prepare_seconds"]
    assert prepare["strict_total"] is None
    assert prepare["known_total"] == pytest.approx(0.2)
    assert prepare["unknown_decisions"] == 1


def test_regeneration_charges_discarded_draft_and_takes_kv_maxima() -> None:
    draft_stats = _stats(
        materialized=11,
        system_prefill=7,
        target_input=13,
        system_prefix=20,
        gist_prefix=30,
        final_cache=140,
        scope_system_after=20,
        scope_gist_after=30,
    )
    regeneration_stats = _stats(
        materialized=5,
        system_prefill=0,
        target_input=9,
        system_prefix=20,
        gist_prefix=50,
        final_cache=130,
        scope_system_before=20,
        scope_system_after=20,
        scope_gist_before=30,
        scope_gist_after=50,
        allocator_peak=999,
    )
    record = _record(
        "session-a",
        "d1",
        [
            _attempt("uid-draft", discarded=True, stats=draft_stats),
            _attempt(
                "uid-regeneration",
                phase="regeneration",
                prompt=50,
                completion=2,
                stats=regeneration_stats,
                attempt_index=2,
            ),
        ],
    )

    session = _session(summarize_event_native_steps([record]))

    assert session["generation_attempts"] == 2
    assert session["discarded_attempts"] == 1
    assert session["costs"]["openai_resident_usage"]["prompt_tokens"][
        "strict_total"
    ] == 90
    assert session["costs"]["actual_model_work"]["materialized_encoder_tokens"][
        "strict_total"
    ] == 16
    assert session["costs"]["actual_model_work"]["raw_prefill_input_tokens"][
        "strict_total"
    ] == 18
    assert session["costs"]["session_reuse"]["session_reused_raw_tokens"][
        "strict_total"
    ] == 8
    peaks = session["costs"]["logical_kv_peaks"]
    assert peaks["resident_prefix_kv_logical_bytes"]["strict_peak"] == 70
    assert peaks["resident_total_kv_logical_bytes"]["strict_peak"] == 140
    assert peaks["resident_raw_kv_logical_bytes"]["strict_peak"] == 90
    assert peaks["scope_total_kv_logical_bytes"]["strict_peak"] == 70
    assert peaks["torch_allocator_peak_allocated_bytes"]["strict_peak"] is None
    assert peaks["torch_allocator_peak_allocated_bytes"]["known_peak"] == 999
    assert peaks["torch_allocator_peak_allocated_bytes"]["unknown_calls"] == 1
    assert [phase["phase"] for phase in session["phases"]] == [
        "draft",
        "regeneration",
    ]
    assert session["source_attempts"][0]["attempt_uid"] == "uid-draft"
    assert session["source_attempts"][1]["attempt_uid"] == "uid-regeneration"


def test_session_cache_peaks_transfers_and_runtime_are_decision_level() -> None:
    first = _record(
        "session-a",
        "d1",
        [_attempt("uid-1", stats=_stats())],
        timing={"prepare_seconds": 0.2, "reconsider_seconds": 0.1},
        runtime=0.5,
        session_cache=_session_cache(
            "session-a",
            cpu_bytes=100,
            device_logical=80,
            device_backing=120,
            transfer_in=10,
            transfer_out=4,
        ),
    )
    second = _record(
        "session-a",
        "d2",
        [_attempt("uid-2", stats=_stats())],
        timing={"prepare_seconds": 0.3, "reconsider_seconds": 0.2},
        runtime=0.8,
        session_cache=_session_cache(
            "session-a",
            cpu_bytes=90,
            device_logical=110,
            device_backing=150,
            transfer_in=7,
            transfer_out=3,
        ),
    )

    session = _session(summarize_event_native_steps([first, second]))

    runtime = session["decision_runtime"]["decision_runtime_seconds"]
    assert runtime["strict_total"] == pytest.approx(1.3)
    assert session["decision_runtime"]["includes_controller_timing"] is True
    assert session["controller_timing"]["prepare_seconds"]["strict_total"] == pytest.approx(0.5)
    cache = session["session_cache_after"]
    assert cache["successful_decisions"] == 2
    assert cache["host_device_byte_peaks"]["cpu_memo_logical_bytes"][
        "strict_peak"
    ] == 100
    assert cache["host_device_byte_peaks"]["device_raw_snapshot_logical_bytes"][
        "strict_peak"
    ] == 110
    assert cache["host_device_byte_peaks"]["device_raw_snapshot_backing_bytes"][
        "strict_peak"
    ] == 150
    assert cache["transfer_bytes"]["last_transfer_bytes_in"]["strict_total"] == 17
    assert cache["transfer_bytes"]["last_transfer_bytes_out"]["strict_total"] == 7
    assert cache["transfer_bytes"]["last_transfer_bytes_total"]["strict_total"] == 24


def test_failed_decision_is_excluded_from_successful_session_cache_accounting() -> None:
    successful = _record(
        "session-a",
        "d1",
        [_attempt("uid-1", stats=_stats())],
        session_cache=_session_cache(
            "session-a",
            cpu_bytes=100,
            device_logical=80,
            device_backing=120,
            transfer_in=10,
            transfer_out=4,
        ),
    )
    failed = _record(
        "session-a",
        "d2",
        [_attempt("uid-2", status="failed", prompt=None)],
        status="failed",
    )

    cache = _session(summarize_event_native_steps([successful, failed]))[
        "session_cache_after"
    ]
    assert cache["successful_decisions"] == 1
    assert cache["transfer_bytes"]["last_transfer_bytes_total"]["strict_total"] == 14


def test_missing_successful_cache_keeps_strict_transfer_and_peak_unknown() -> None:
    known = _record(
        "session-a",
        "d1",
        [_attempt("uid-1", stats=_stats())],
        session_cache=_session_cache(
            "session-a",
            cpu_bytes=100,
            device_logical=80,
            device_backing=120,
            transfer_in=10,
            transfer_out=4,
        ),
    )
    unknown = _record("session-a", "d2", [_attempt("uid-2", stats=_stats())])

    cache = _session(summarize_event_native_steps([known, unknown]))[
        "session_cache_after"
    ]
    transfer = cache["transfer_bytes"]["last_transfer_bytes_total"]
    assert transfer["strict_total"] is None
    assert transfer["known_total"] == 14
    assert transfer["unknown_calls"] == 1
    peak = cache["host_device_byte_peaks"]["cpu_memo_logical_bytes"]
    assert peak["strict_peak"] is None
    assert peak["known_peak"] == 100
    assert peak["unknown_calls"] == 1


def test_zero_generation_calls_have_known_zero_additive_cost() -> None:
    record = _record("session-a", "d1", [], status="failed", runtime=0.1)

    session = _session(summarize_event_native_steps(record))

    prompt = session["costs"]["openai_resident_usage"]["prompt_tokens"]
    assert prompt == {
        "strict_total": 0,
        "known_total": 0,
        "known_calls": 0,
        "unknown_calls": 0,
    }


def test_pending_journal_only_attempt_is_an_unrecorded_unknown_call(tmp_path) -> None:
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    handle = _journal_start(journal, "session-a", "d-cutoff")

    summary = summarize_event_native_steps([], attempt_journal=journal.path)

    assert summary["attempt_inventory"]["run_inventory_verified"] is True
    assert summary["attempt_inventory"]["journal_only_generation_attempts"] == 1
    session = _session(summary)
    assert session["decision_count"] == 1
    assert session["recorded_decisions"] == 0
    assert session["unrecorded_decisions"] == 1
    assert session["source_attempts"] == [
        {
            "source": "attempt_journal_only",
            "decision_key": "d-cutoff",
            "phase": "unrecorded",
            "attempt_uid": handle.attempt_uid,
            "attempt_index": 1,
            "status": "pending",
            "discarded": None,
        }
    ]
    prompt = session["costs"]["openai_resident_usage"]["prompt_tokens"]
    assert prompt["strict_total"] is None
    assert prompt["known_total"] is None
    assert prompt["unknown_calls"] == 1
    actual = session["costs"]["actual_model_work"]["target_input_tokens"]
    assert actual["strict_total"] is None
    assert actual["unknown_calls"] == 1
    runtime = session["decision_runtime"]["decision_runtime_seconds"]
    assert runtime["strict_total"] is None
    assert runtime["unknown_decisions"] == 1


def test_finished_journal_only_attempt_uses_usage_but_keeps_work_unknown(tmp_path) -> None:
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    handle = _journal_start(journal, "session-a", "d-cutoff")
    journal.finish(
        handle,
        "completed",
        usage={"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
    )

    summary = summarize_event_native_steps([], attempt_journal=journal.path)

    session = _session(summary)
    prompt = session["costs"]["openai_resident_usage"]["prompt_tokens"]
    assert prompt["strict_total"] == 40
    assert prompt["known_calls"] == 1
    work = session["costs"]["actual_model_work"]["materialized_encoder_tokens"]
    assert work["strict_total"] is None
    assert work["unknown_calls"] == 1
    assert session["phases"][0]["phase"] == "unrecorded"


def test_step_trace_and_journal_usage_conflict_is_rejected(tmp_path) -> None:
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    handle = _journal_start(journal, "session-a", "d1")
    journal.finish(
        handle,
        "completed",
        usage={"prompt_tokens": 41, "completion_tokens": 3, "total_tokens": 44},
    )
    record = _record(
        "session-a",
        "d1",
        [_attempt(handle.attempt_uid, stats=_stats())],
    )

    with pytest.raises(ValueError, match="disagrees with journal usage"):
        summarize_event_native_steps([record], attempt_journal=journal.path)


def test_partial_steps_and_journal_tails_are_explicitly_unverified(tmp_path) -> None:
    steps = tmp_path / "steps.jsonl"
    steps.write_bytes(b'{"session_id":"session-a"')
    loaded = read_event_native_steps(steps)
    assert loaded == {"records": [], "truncated_tail": True}

    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    _journal_start(journal, "session-a", "d-cutoff")
    with journal.path.open("ab") as handle:
        handle.write(b'{"schema":"a-runtime-attempt-journal-v1"')

    summary = summarize_event_native_steps(
        loaded["records"],
        attempt_journal=journal.path,
        steps_truncated_tail=loaded["truncated_tail"],
    )
    assert summary["steps_truncated_tail"] is True
    assert summary["attempt_journal_truncated_tail"] is True
    assert summary["attempt_inventory"]["run_inventory_verified"] is False
    assert summary["costs"]["framed_input_complete"] is False
    assert summary["costs"]["openai_resident_usage"]["prompt_tokens"][
        "strict_total"
    ] is None


def test_scripted_stats_do_not_become_zero_actual_costs() -> None:
    record = _record(
        "session-a",
        "d1",
        [
            _attempt(
                "uid-scripted",
                stats={"scripted_fixture": True, "model_forward_calls": 1},
            )
        ],
    )

    session = _session(summarize_event_native_steps(record))

    assert session["costs"]["openai_resident_usage"]["prompt_tokens"][
        "strict_total"
    ] == 40
    materialized = session["costs"]["actual_model_work"][
        "materialized_encoder_tokens"
    ]
    assert materialized["strict_total"] is None
    assert materialized["known_total"] is None
    assert materialized["unknown_calls"] == 1
    prefix = session["costs"]["logical_kv_peaks"][
        "resident_prefix_kv_logical_bytes"
    ]
    assert prefix["strict_peak"] is None
    assert prefix["known_peak"] is None


def test_duplicate_replay_is_free_but_conflicting_decision_is_rejected() -> None:
    record = _record(
        "session-a",
        "d1",
        [_attempt("uid-1", stats=_stats())],
        runtime=0.5,
        session_cache=_session_cache(
            "session-a",
            cpu_bytes=100,
            device_logical=80,
            device_backing=120,
            transfer_in=10,
            transfer_out=4,
        ),
    )
    summary = summarize_event_native_steps([record, copy.deepcopy(record)])

    assert summary["input_records"] == 2
    assert summary["unique_decisions"] == 1
    assert summary["duplicate_records_ignored"] == 1
    assert _session(summary)["generation_attempts"] == 1
    assert _session(summary)["costs"]["openai_resident_usage"]["prompt_tokens"][
        "strict_total"
    ] == 40
    assert _session(summary)["decision_runtime"]["decision_runtime_seconds"][
        "strict_total"
    ] == pytest.approx(0.5)
    assert _session(summary)["session_cache_after"]["transfer_bytes"][
        "last_transfer_bytes_total"
    ]["strict_total"] == 14

    conflict = copy.deepcopy(record)
    conflict["generation_trace"][0]["usage"]["prompt_tokens"] = 41
    conflict["generation_trace"][0]["usage"]["total_tokens"] = 44
    conflict["generation_trace"][0]["planned_resident_prompt_tokens"] = 41
    with pytest.raises(ValueError, match="conflicting duplicate decision"):
        summarize_event_native_steps([record, conflict])


def test_same_content_and_attempt_uid_in_different_sessions_are_both_charged() -> None:
    first = _record("session-a", "d1", [_attempt("shared-uid", stats=_stats())])
    second = copy.deepcopy(first)
    second["session_id"] = "session-b"

    summary = summarize_event_native_steps([first, second])

    assert summary["session_count"] == 2
    assert [item["session_id"] for item in summary["sessions"]] == [
        "session-a",
        "session-b",
    ]
    assert all(item["generation_attempts"] == 1 for item in summary["sessions"])
    assert all(
        item["costs"]["openai_resident_usage"]["prompt_tokens"]["strict_total"]
        == 40
        for item in summary["sessions"]
    )


def test_cli_reads_steps_jsonl_and_requires_fresh_output(tmp_path) -> None:
    steps = tmp_path / "steps.jsonl"
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    output = tmp_path / "costs.json"
    handle = _journal_start(journal, "session-a", "d1")
    journal.finish(
        handle,
        "completed",
        usage={"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
    )
    record = _record(
        "session-a", "d1", [_attempt(handle.attempt_uid, stats=_stats())]
    )
    steps.write_text(json.dumps(record) + "\n", encoding="utf-8")

    main(
        [
            "--steps",
            str(steps),
            "--attempt-journal",
            str(journal.path),
            "--output",
            str(output),
        ]
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema"] == "a-event-native-task-cost-summary-v1"
    assert persisted["unique_decisions"] == 1
    assert persisted["attempt_inventory"]["run_inventory_verified"] is True
    with pytest.raises(FileExistsError):
        main(
            [
                "--steps",
                str(steps),
                "--attempt-journal",
                str(journal.path),
                "--output",
                str(output),
            ]
        )


def test_missing_cache_trace_is_explicitly_unknown() -> None:
    record = _record("session-a", "d1", [_attempt("uid-1", stats=_stats())])

    provenance = summarize_event_native_steps(record)["cache_provenance"]

    assert provenance["trace_coverage"] == {
        "known_attempts": 0,
        "unknown_attempts": 1,
        "context_complete_attempts": 0,
        "context_incomplete_attempts": 0,
    }
    assert provenance["attempts"][0]["trace_coverage"] == "unknown"
    assert provenance["attempts"][0]["op_ids"] == []
    assert provenance["by_operation_kind"] == {}


def test_cache_trace_charges_one_extract_for_multiple_consumers() -> None:
    stats = _stats()
    stats["cache_trace"] = _cache_trace(
        "uid-1",
        "session-a",
        "d1",
        ops=[
            _cache_op(
                "op-extract",
                requested=9,
                completed=7,
                transfer=24,
                logical=48,
                result_entry_id="entry-extract",
                consumers=["placement-a", "placement-b"],
            ),
            _cache_op(
                "op-access",
                kind="access",
                requested=0,
                completed=0,
                transfer=0,
                logical=None,
                source_entry_id="entry-extract",
                consumers=["placement-a", "placement-b"],
            ),
        ],
        entries=[
            {
                "entry_id": "entry-extract",
                "producer_attempt_uid": "uid-1",
                "created_by_op_id": "op-extract",
                "origin_extraction_op_id": "op-extract",
                "parent_entry_id": None,
                "kind": "workspace",
            }
        ],
        placements=[
            _cache_placement("placement-a", access_op_id="op-access"),
            _cache_placement("placement-b", access_op_id="op-access"),
        ],
    )
    record = _record("session-a", "d1", [_attempt("uid-1", stats=stats)])

    provenance = summarize_event_native_steps(record)["cache_provenance"]

    extract = provenance["by_operation_kind"]["extract"]
    assert provenance["unique_operations"] == 2
    assert extract["unique_operations"] == 1
    assert extract["input_tokens_completed"]["known_total"] == 7
    assert extract["transfer_bytes"]["known_total"] == 24
    assert extract["logical_bytes"]["known_total"] == 48
    attempt = provenance["attempts"][0]
    assert attempt["op_ids"] == ["op-extract", "op-access"]
    assert attempt["placement_ids"] == ["placement-a", "placement-b"]
    assert attempt["workspace_source_group"] == "raw-source-group-a"


def test_cache_trace_preserves_json_safe_operation_entry_and_placement_extras() -> None:
    extract = _cache_op(
        "op-extract",
        result_entry_id="entry-extract",
        consumers=["placement-a"],
    )
    extract["implementation"] = {"tier": "cpu", "copied": False}
    access = _cache_op(
        "op-access",
        kind="access",
        requested=0,
        completed=0,
        transfer=0,
        logical=None,
        source_entry_id="entry-extract",
        consumers=["placement-a"],
    )
    entry = {
        "entry_id": "entry-extract",
        "producer_attempt_uid": "uid-1",
        "created_by_op_id": "op-extract",
        "origin_extraction_op_id": "op-extract",
        "parent_entry_id": None,
        "kind": "workspace",
        "layout": ["gist", 3],
    }
    placement = _cache_placement("placement-a", access_op_id="op-access")
    placement["resident"] = {"device": "npu", "bytes": 48}
    stats = _stats()
    stats["cache_trace"] = _cache_trace(
        "uid-1",
        "session-a",
        "d1",
        ops=[extract, access],
        entries=[entry],
        placements=[placement],
    )

    attempt = summarize_event_native_steps(
        _record("session-a", "d1", [_attempt("uid-1", stats=stats)])
    )["cache_provenance"]["attempts"][0]

    assert attempt["ops"][0]["implementation"] == {"tier": "cpu", "copied": False}
    assert attempt["entries"][0]["layout"] == ["gist", 3]
    assert attempt["placements"][0]["resident"] == {"device": "npu", "bytes": 48}


def test_failed_cache_trace_keeps_partial_operation_costs() -> None:
    failed = _attempt(
        "uid-failed",
        status="failed",
        prompt=None,
        completion=None,
    )
    failed["cache_trace"] = _cache_trace(
        "uid-failed",
        "session-a",
        "d1",
        status="failed",
        context_complete=False,
        ops=[
            _cache_op(
                "op-transfer",
                kind="transfer",
                status="failed",
                requested=9,
                completed=4,
                transfer=64,
                logical=None,
            )
        ],
    )
    record = _record("session-a", "d1", [failed], status="failed")

    provenance = summarize_event_native_steps(record)["cache_provenance"]

    transfer = provenance["by_operation_kind"]["transfer"]
    assert provenance["trace_coverage"]["known_attempts"] == 1
    assert provenance["trace_coverage"]["context_incomplete_attempts"] == 1
    assert transfer["completed_operations"] == 0
    assert transfer["input_tokens_completed"] == {
        "known_total": 4,
        "known_operations": 1,
        "unknown_operations": 0,
    }
    assert transfer["transfer_bytes"]["known_total"] == 64
    assert transfer["logical_bytes"]["known_total"] is None
    assert transfer["logical_bytes"]["unknown_operations"] == 1


def test_cache_trace_identity_mismatch_is_rejected() -> None:
    stats = _stats()
    stats["cache_trace"] = _cache_trace("wrong-uid", "session-a", "d1")
    record = _record("session-a", "d1", [_attempt("uid-1", stats=stats)])

    with pytest.raises(ValueError, match="attempt_uid disagrees"):
        summarize_event_native_steps(record)


def test_repeated_cache_operation_is_not_charged_twice() -> None:
    shared = _cache_op(
        "op-shared",
        requested=6,
        completed=5,
        transfer=10,
        logical=20,
    )
    first_stats = _stats()
    first_stats["cache_trace"] = _cache_trace(
        "uid-1", "session-a", "d1", ops=[shared]
    )
    second_stats = _stats()
    second_stats["cache_trace"] = _cache_trace(
        "uid-2", "session-a", "d2", ops=[copy.deepcopy(shared)]
    )
    summary = summarize_event_native_steps(
        [
            _record("session-a", "d1", [_attempt("uid-1", stats=first_stats)]),
            _record("session-a", "d2", [_attempt("uid-2", stats=second_stats)]),
        ]
    )

    extract = summary["cache_provenance"]["by_operation_kind"]["extract"]
    assert summary["cache_provenance"]["unique_operations"] == 1
    assert extract["input_tokens_completed"]["known_total"] == 5
    assert extract["transfer_bytes"]["known_total"] == 10
    assert extract["logical_bytes"]["known_total"] == 20


def test_cache_placement_must_reference_completed_local_access_op() -> None:
    stats = _stats()
    stats["cache_trace"] = _cache_trace(
        "uid-1",
        "session-a",
        "d1",
        ops=[_cache_op("op-access", kind="access", status="failed")],
        placements=[_cache_placement("placement-a", access_op_id="op-access")],
    )
    record = _record("session-a", "d1", [_attempt("uid-1", stats=stats)])

    with pytest.raises(ValueError, match="access op that did not complete"):
        summarize_event_native_steps(record)


def test_lifecycle_trace_is_deduplicated_without_generation_cost_charging() -> None:
    lifecycle = {
        "schema": "event-native-cache-lifecycle-v1",
        "lifecycle_id": "lifecycle-close-1",
        "associated_attempt_uid": "uid-1",
        "session_id": "session-a",
        "ops": [{"op_id": "op-close", "kind": "close", "status": "completed"}],
    }
    first_cache = _session_cache(
        "session-a",
        cpu_bytes=0,
        device_logical=0,
        device_backing=0,
        transfer_in=0,
        transfer_out=0,
    )
    first_cache["last_lifecycle_trace"] = lifecycle
    second_cache = _session_cache(
        "session-b",
        cpu_bytes=0,
        device_logical=0,
        device_backing=0,
        transfer_in=0,
        transfer_out=0,
    )
    second_cache["last_lifecycle_trace"] = copy.deepcopy(lifecycle)
    summary = summarize_event_native_steps(
        [
            _record("session-a", "d1", [], session_cache=first_cache),
            _record("session-b", "d2", [], session_cache=second_cache),
        ]
    )

    provenance = summary["cache_provenance"]
    assert provenance["unique_operations"] == 0
    assert provenance["by_operation_kind"] == {}
    lifecycle_records = provenance["lifecycle_traces"]
    assert lifecycle_records["records_with_trace"] == 2
    assert lifecycle_records["unique_lifecycle_traces"] == 1
    assert lifecycle_records["traces"] == [
        {
            "lifecycle_id": "lifecycle-close-1",
            "associated_attempt_uid": "uid-1",
            "session_id": "session-a",
            "ops": [
                {"op_id": "op-close", "kind": "close", "status": "completed"}
            ],
            "observing_session_id": "session-a",
            "observing_decision_key": "d1",
        }
    ]


def test_lifecycle_trace_is_read_from_after_close_snapshot() -> None:
    lifecycle = {
        "schema": "event-native-cache-lifecycle-v1",
        "lifecycle_id": "lifecycle-close-1",
        "associated_attempt_uid": "old-attempt",
        "session_id": "old-session",
        "ops": [{"op_id": "op-close", "kind": "release", "status": "completed"}],
    }
    after_close = _session_cache(
        "old-session",
        cpu_bytes=0,
        device_logical=0,
        device_backing=0,
        transfer_in=0,
        transfer_out=0,
    )
    after_close["last_lifecycle_trace"] = lifecycle
    record = _record("new-session", "d1", [])
    record["session_cache_after_close"] = after_close

    lifecycle_records = summarize_event_native_steps(record)["cache_provenance"][
        "lifecycle_traces"
    ]

    assert lifecycle_records["records_with_trace"] == 1
    assert lifecycle_records["unique_lifecycle_traces"] == 1
    assert lifecycle_records["traces"] == [
        {
            "lifecycle_id": "lifecycle-close-1",
            "associated_attempt_uid": "old-attempt",
            "session_id": "old-session",
            "ops": [
                {"op_id": "op-close", "kind": "release", "status": "completed"}
            ],
            "observing_session_id": "new-session",
            "observing_decision_key": "d1",
        }
    ]
