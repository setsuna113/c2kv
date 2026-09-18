from __future__ import annotations

import json

import pytest

from benchmarks.measurement.aggregate import aggregate
from benchmarks.measurement.c1 import convert_run
from benchmarks.measurement.telemetry import read_jsonl


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _trace(outer, attempt, phase, resident, *, extraction=None):
    telemetry = {
        "extractions": [] if extraction is None else [extraction],
        "generation": {
            "server_request_id": f"generation-{attempt}",
            "phase": "generation",
            "paper_measurement": {"metrics": {
                "request_peak_resident_kv_bytes": resident,
                "request_peak_cached_evictable_kv_bytes": resident // 4,
                "cached_evictable_kv_peak_bytes": resident // 3,
                "gist_generation_duration_ns": 0,
            }},
        },
    }
    return {
        "phase": phase,
        "status": "completed",
        "attempt_uid": attempt,
        "outer_request_id": outer,
        "duration_ns": 30,
        "controller": {
            "kv_bytes_per_token": 10,
            "same_prefix_full_reference": {
                "full_history_tokens": 10,
                "common_live_tokens": 20,
            },
            "compression_ratio": {"active_history_bytes": 40},
        },
        "generation": {"stats": {
            "native_request_id": f"generation-{attempt}",
            "native_telemetry": telemetry,
        }},
    }


def _record(outer, decision, traces, *, recovery=None, duration=90):
    return {
        "schema": "a-event-native-exact-step-v1",
        "status": "ok",
        "session_id": "bfcl/task/attempt-0",
        "decision_key": decision,
        "outer_request_id": outer,
        "decision_start_unix_ns": 10,
        "decision_end_unix_ns": 10 + duration,
        "decision_duration_ns": duration,
        "controller_timing": {
            "prepare_duration_ns": 11,
            "reconsider_duration_ns": 12,
        },
        "generation_trace": traces,
        "exact_recovery": recovery or {},
    }


def test_convert_run_joins_real_harness_and_deduplicates_native_requests(tmp_path):
    cell = tmp_path / "cell"
    native = cell / "native"
    task = native / "task_shards" / "task"
    outer_one = "c1-one"
    outer_two = "c1-two"
    extraction = {
        "server_request_id": "extract-one",
        "phase": "c1_gist_extraction",
        "cache_hit": False,
        "gist_generation_duration_ns": 7,
        "paper_measurement": {"metrics": {
            "gist_generation_duration_ns": 7,
            "extraction_duration_ns": 9,
            "cache_hit": False,
        }},
    }
    recovery = {
        "status": "recover",
        "selection": {
            "selector": "risk", "available": True, "score": 0.8,
        },
        "measurement_phases": [
            {"phase": "c1_detector", "duration_ns": 3},
            {"phase": "c1_retrieval_embedding_and_feasibility", "duration_ns": 5},
        ],
        "selection_model_calls": [{
            "capability": "embedding", "purpose": "document",
            "latency_seconds": 11e-9,
        }],
    }
    records = [
        _record(
            outer_one,
            "turn-0/step-0",
            [
                _trace(outer_one, "draft-one", "draft", 100, extraction=extraction),
                # The engine may repeat an already-observed extraction receipt
                # in the complete decision chain; the converter counts it once.
                _trace(outer_one, "regen-one", "regeneration", 120, extraction=extraction),
            ],
            recovery=recovery,
        ),
        _record(
            outer_two,
            "turn-0/step-1",
            [_trace(outer_two, "draft-two", "draft", 80)],
            duration=40,
        ),
    ]
    _write_jsonl(task / "server" / "steps.jsonl", records)
    harness = [
        {
            "event_type": "episode_start", "episode_id": "task",
            "episode_instance_id": "instance", "start_unix_ns": 1,
        },
        {
            "event_type": "decision", "episode_id": "task",
            "decision_request_id": outer_one, "start_unix_ns": 10,
            "end_unix_ns": 110, "duration_ns": 100,
        },
        {
            "event_type": "tool_action", "episode_id": "task",
            "decision_request_id": outer_one, "end_unix_ns": 140,
            "duration_ns": 20, "action": "tool()", "outcome": "ok",
        },
        {
            "event_type": "decision", "episode_id": "task",
            "decision_request_id": outer_two, "start_unix_ns": 200,
            "end_unix_ns": 250, "duration_ns": 50,
        },
        {
            "event_type": "episode_end", "episode_id": "task",
            "episode_instance_id": "instance", "duration_ns": 500,
        },
    ]
    _write_jsonl(
        task / "bfcl" / "bfcl" / "measurement" / "harness_events.jsonl",
        harness,
    )

    _write_jsonl(cell / "native_engine_telemetry.jsonl", [
        {
            "schema_version": 1,
            "outer_request_id": outer_one,
            "server_request_id": "extract-one",
            "phase": "c1_gist_extraction",
            "metrics": {
                "gist_generation_duration_ns": 7,
                "extraction_duration_ns": 9,
                "cache_hit": False,
            },
        },
        {
            "schema_version": 1,
            "outer_request_id": outer_one,
            "server_request_id": "generation-regen-one",
            "phase": "regeneration:generation",
            "kind": "generation",
            "metrics": {
                "request_peak_resident_kv_bytes": 120,
                "request_peak_cached_evictable_kv_bytes": 30,
                "cached_evictable_kv_peak_bytes": 40,
                "gist_generation_duration_ns": 0,
                "history_full_kv_tokens": 5,
                "whole_full_kv_tokens": 25,
                "whole_active_kv_tokens": 23,
                "generation_active_kv_tokens": 22,
                "canonical_full_source": True,
            },
        },
        {
            "schema_version": 1,
            "outer_request_id": "warmup-not-a-step",
            "server_request_id": "warmup-request",
            "phase": "generation",
            "kind": "generation",
            "metrics": {"request_peak_resident_kv_bytes": 999},
        },
    ])
    _write_jsonl(cell / "prefix_replay.jsonl", [{
        "schema": "c2kv.prefix_replay.v1",
        "event_type": "prefix_replay",
        "request_id": outer_one,
        "canonical_sha256": "preserved",
        "source_response": {"paper_measurement": {"metrics": {
            "whole_full_kv_tokens": 30,
        }}},
        "target_paper_measurement": {"metrics": {
            "whole_full_kv_tokens": 25,
            "whole_active_kv_tokens": 23,
        }},
    }])
    receipt = convert_run(
        native, cell, benchmark="bfcl", arm="c1", replay=True
    )
    proxy = list(read_jsonl(cell / "proxy_telemetry.jsonl"))
    requests = [row for row in proxy if row["event_type"] == "request"]
    phases = [row for row in proxy if row["event_type"] == "phase"]
    server = list(read_jsonl(cell / "server_telemetry.jsonl"))
    normalized_harness = list(read_jsonl(
        cell / "measurement" / "harness_events.jsonl"
    ))
    replay = list(read_jsonl(cell / "prefix_replay.jsonl"))

    assert receipt["coverage"] == {
        "harness_decisions": 2,
        "server_measurements": 4,
        "gist_generation_measurements": 3,
    }
    assert requests[0]["duration_ns"] == 100
    assert requests[0]["latency_source"] == "harness_client_request"
    assert requests[0]["c1_metrics"] == {
        "draft_generation_requests": 1,
        "regeneration_requests": 1,
        "generation_requests": 2,
        "detector_calls": 1,
        "detector_score_available": True,
        "recovery_committed": True,
        "gist_generation_duration_ns": 7,
        "gist_generation_measured_requests": 1,
    }
    # Both the request summary and additive server ledger count extract-one once.
    assert [row["server_request_id"] for row in server].count("extract-one") == 1
    assert sum(
        row.get("paper_measurement", {}).get("metrics", {}).get(
            "gist_generation_duration_ns", 0
        )
        for row in server
    ) == 7
    assert any(row["phase"] == "c1_detector" for row in phases)
    assert any(row["phase"] == "c1_embedding" for row in phases)
    assert len(normalized_harness) == len(harness)
    assert server[0]["native_engine_event"]["schema_version"] == 1
    assert receipt["native_engine_telemetry_sources"] == [
        str((cell / "native_engine_telemetry.jsonl").resolve())
    ]
    assert receipt["prefix_replay_enrichment"] == {
        "rows": 1, "source_added": 1, "target_added": 0,
        "target_refreshed": 1,
    }
    assert replay[0]["canonical_sha256"] == "preserved"
    assert replay[0]["source_paper_measurement"]["metrics"][
        "whole_full_kv_tokens"
    ] == 30
    assert replay[0]["target_paper_measurement"]["metrics"][
        "whole_active_kv_tokens"
    ] == 23
    regeneration = next(
        row for row in server
        if row["server_request_id"] == "generation-regen-one"
    )
    metrics = regeneration["paper_measurement"]["metrics"]
    assert metrics["history_full_kv_tokens"] == 10
    assert metrics["whole_full_kv_tokens"] == 30
    assert metrics["history_active_kv_tokens"] == 4
    assert metrics["canonical_full_source"] == (
        "controller.same_prefix_full_reference"
    )
    # Physical/active engine semantics are preserved.
    assert metrics["whole_active_kv_tokens"] == 23
    assert metrics["generation_active_kv_tokens"] == 22
    provenance = regeneration["denominator_provenance"]
    assert provenance["native_engine_reported"] == {
        "history_full_kv_tokens": 5,
        "whole_full_kv_tokens": 25,
        "canonical_full_source": True,
    }
    assert provenance["physical_engine_metrics_overridden"] is False
    assert receipt["native_engine_rows_ignored_without_step"] == 1

    summary = aggregate(proxy, normalized_harness, server_rows=server)
    headline = summary["latency_ms"]["complete_model_side_per_committed_action"]
    assert headline["total_model_side_ns"] == 150
    assert headline["committed_actions"] == 1
    assert summary["latency_ms"]["episode_wall"]["mean"] == pytest.approx(0.0005)
    assert summary["memory"]["request_peak_resident_kv_bytes"]["max"] == 120
    assert summary["token_ratios"]["history"]["ratio_of_sums"] == 0.4


def test_convert_run_does_not_fabricate_unavailable_engine_metrics(tmp_path):
    task = tmp_path / "native" / "task_shards" / "task"
    outer = "c1-missing"
    trace = _trace(outer, "draft", "draft", 10)
    trace["generation"]["stats"].pop("native_telemetry")
    trace["generation"]["stats"]["sglang_transport"] = {"rid": "draft"}
    _write_jsonl(
        task / "server" / "steps.jsonl",
        [_record(outer, "turn-0/step-0", [trace])],
    )
    receipt = convert_run(
        tmp_path / "native", tmp_path / "cell",
        benchmark="bfcl", arm="c1",
    )
    server = list(read_jsonl(tmp_path / "cell" / "server_telemetry.jsonl"))
    assert receipt["coverage"]["server_measurements"] == 1  # recorded Full denominator
    metrics = server[0]["paper_measurement"]
    assert "request_peak_resident_kv_bytes" not in metrics
    assert "gist_generation_duration_ns" not in metrics


def test_convert_run_rejects_conflicting_native_request_identity(tmp_path):
    task = tmp_path / "native" / "task_shards" / "task"
    outer = "c1-conflict"
    first = _trace(outer, "one", "draft", 10)
    second = _trace(outer, "two", "regeneration", 20)
    first["generation"]["stats"]["native_telemetry"]["generation"][
        "server_request_id"
    ] = "same"
    second["generation"]["stats"]["native_telemetry"]["generation"][
        "server_request_id"
    ] = "same"
    _write_jsonl(
        task / "server" / "steps.jsonl",
        [_record(outer, "turn-0/step-0", [first, second])],
    )
    with pytest.raises(ValueError, match="conflicting native server"):
        convert_run(
            tmp_path / "native", tmp_path / "cell",
            benchmark="bfcl", arm="c1",
        )
