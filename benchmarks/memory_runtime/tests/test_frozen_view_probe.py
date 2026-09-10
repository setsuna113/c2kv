from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

import pytest

from benchmarks.memory_runtime import frozen_view_probe as probe


CONFIG_PATH = Path(probe.__file__).with_name("configs") / "frozen_view_dev1.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_schedule_is_exactly_the_declared_24_cells():
    schedule = probe.validate_design(_config())

    assert len(schedule) == 24
    assert [(cell["phase"], cell["view"]) for cell in schedule[:6]] == [
        ("negative", "A"), ("negative", "C"),
        ("main", "A"), ("main", "B"), ("main", "D"), ("main", "C"),
    ]
    assert [(cell["phase"], cell["view"]) for cell in schedule[-6:]] == [
        ("negative", "C"), ("negative", "A"),
        ("main", "D"), ("main", "A"), ("main", "C"), ("main", "B"),
    ]
    assert len({cell["cell_id"] for cell in schedule}) == 24


def test_extraction_manifest_exactly_joins_recorded_carriers_and_response_ledger(tmp_path):
    source_root = tmp_path / "capacity"
    source_root.mkdir()
    ledger = tmp_path / "telemetry_probe_v1" / "transport.jsonl"
    ledger.parent.mkdir()
    key = "a" * 64
    ledger.write_text(json.dumps({
        "path": "/v1/c2kv/extract", "status": "completed",
        "response": {
            "key_hash": key, "gist_len": 5, "original_seq_len": 19,
            "success": True, "error": None,
        },
    }) + "\n", encoding="utf-8")
    request_tools = [{
        "type": "function", "function": {
            "name": "synthetic_tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    main_row = {
        "request_view": {
            "model": "synthetic", "messages": [{"role": "user", "content": "query"}],
            "tools": request_tools,
        },
        "memory_runtime": {"block_refs": [{
            "key_hash": key, "source_indices": [0, 1], "gist_tokens": 5,
        }]},
        "forwarded_request_views": [{"messages": [{
            "role": "user", "content": "synthetic history",
            "c2kv_key_hash": key, "c2kv_ratio": 4,
        }]}],
    }

    manifest = probe._build_extraction_manifest(source_root, main_row, _config())

    assert manifest["count"] == 1
    assert manifest["items"][0]["expected_response"] == {
        "key_hash": key, "gist_len": 5, "original_seq_len": 19,
    }
    assert manifest["items"][0]["source_indices"] == [0, 1]
    assert manifest["provenance"]["path"] == str(ledger.resolve())
    assert manifest["tools"] == []
    assert manifest["items"][0]["cache_key"]["tools_hash"] == probe.proxy._digest([])
    assert probe._base_payload(main_row, _config()["sampling"])["tools"] == request_tools


def test_extraction_manifest_rejects_unknown_original_length(tmp_path):
    source_root = tmp_path / "capacity"
    source_root.mkdir()
    ledger = tmp_path / "telemetry_probe_v1" / "transport.jsonl"
    ledger.parent.mkdir()
    key = "b" * 64
    ledger.write_text(json.dumps({
        "path": "/v1/c2kv/extract", "status": "completed",
        "response": {
            "key_hash": key, "gist_len": 5, "original_seq_len": None,
            "success": True,
        },
    }) + "\n", encoding="utf-8")
    main_row = {
        "request_view": {"tools": []},
        "memory_runtime": {"block_refs": [{
            "key_hash": key, "source_indices": [0], "gist_tokens": 5,
        }]},
        "forwarded_request_views": [{"messages": [{
            "role": "user", "content": "synthetic history",
            "c2kv_key_hash": key, "c2kv_ratio": 4,
        }]}],
    }

    with pytest.raises(ValueError, match="Invalid extraction ledger record"):
        probe._build_extraction_manifest(source_root, main_row, _config())


def test_prepared_view_validation_uses_nested_auxiliary_gate_geometry():
    evidence_packet = {"role": "system", "content": "synthetic evidence"}
    evidence = {
        "selected_event_ids": ["event-e"], "packet": evidence_packet,
        "selection_bytes": 147456,
    }
    full_payload = {
        "model": "synthetic", "messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "query"},
        ],
        "tools": [], "temperature": 0.001, "seed": 0, "max_tokens": 4096,
        "c2kv_use_gist_projection": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    shared_payload = copy.deepcopy(full_payload)
    shared_payload["messages"].insert(1, copy.deepcopy(evidence_packet))
    capacity_messages = [
        {"role": "user", "content": "history",
         "c2kv_key_hash": "a" * 64, "c2kv_ratio": 4},
        copy.deepcopy(evidence_packet),
        {"role": "user", "content": "query"},
    ]
    capacity_payload = copy.deepcopy(full_payload)
    capacity_payload["messages"] = copy.deepcopy(capacity_messages)
    no_gist_payload = copy.deepcopy(shared_payload)
    views = {
        "negative:A": {"payload": copy.deepcopy(full_payload)},
        "negative:C": {"payload": copy.deepcopy(full_payload)},
        "main:A": {
            "payload": full_payload,
            "server_expectations": {
                "raw_prompt_tokens": 2, "bytes_per_kv_token": 147456,
            },
        },
        "main:B": {
            "payload": shared_payload, "evidence": copy.deepcopy(evidence),
            "counts": {"compressed_records": []},
            "memory_runtime": {
                "evidence_out_index": 1, "budget_applies": False,
                "auxiliary_gate": {"full_raw_prompt_tokens": 2},
                "evidence_bytes": 147456, "gist_tokens": 0,
                "retrieved_event_ids": [], "retained_event_ids": [],
            },
            "server_expectations": {
                "raw_prompt_tokens": 3, "bytes_per_kv_token": 147456,
            },
        },
        "main:C": {
            "payload": capacity_payload, "evidence": copy.deepcopy(evidence),
            "counts": {"compressed_records": [{"record": {"gist_len": 1}}]},
            "memory_runtime": {
                "retrieved_event_ids": [], "retained_event_ids": [],
            },
        },
        "main:D": {
            "payload": no_gist_payload, "evidence": copy.deepcopy(evidence),
            "counts": {"compressed_records": []},
            "memory_runtime": {
                "raw_history_event_ids": ["event-r"],
                "selected_event_ids": ["event-e"],
                "active_history_bytes": 147456,
                "history_budget_bytes": 113246208,
                "gist_tokens": 0, "retrieved_event_ids": [], "retained_event_ids": [],
            },
        },
    }

    probe._validate_prepared_views(views, capacity_messages)

    broken = copy.deepcopy(views)
    broken["main:B"]["memory_runtime"]["auxiliary_gate"]["full_raw_prompt_tokens"] = 99
    with pytest.raises(ValueError, match="Full-shared Full/E geometry"):
        probe._validate_prepared_views(broken, capacity_messages)


def _prepared_artifact():
    config = _config()
    schedule = probe.validate_design(config)
    views = {}
    payloads = {
        "negative:A": [{"role": "user", "content": "negative"}],
        "negative:C": [{"role": "user", "content": "negative"}],
        "main:A": [{"role": "user", "content": "main"}],
        "main:B": [
            {"role": "system", "content": "evidence"},
            {"role": "user", "content": "main"},
        ],
        "main:C": [
            {"role": "user", "content": "history", "c2kv_key_hash": "a" * 64,
             "c2kv_ratio": 4},
            {"role": "system", "content": "evidence"},
            {"role": "user", "content": "main"},
        ],
        "main:D": [
            {"role": "system", "content": "evidence"},
            {"role": "user", "content": "main"},
        ],
    }
    for key, messages in payloads.items():
        payload = {
            "model": "synthetic", "messages": copy.deepcopy(messages), "tools": [],
            **config["sampling"], "c2kv_use_gist_projection": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        views[key] = {
            "payload": payload, "wire_digest": probe._digest(payload),
            "context": {"user_turn": 2, "step": 1 if key.startswith("negative") else 2},
            "server_expectations": {
                "raw_prompt_tokens": 7,
                "bytes_per_kv_token": config["bytes_per_kv_token"],
                "c2kv_query_proj_effective": "base",
                "c2kv_query_proj_decode_verified": True,
                "c2kv_tools_dump": "full",
                "c2kv_gist_seen": any(m.get("c2kv_key_hash") for m in messages),
            },
        }
    manifest = {
        "count": 1, "tools": [],
        "items": [{
            "source_indices": [0], "role": "user", "content": "history", "ratio": 4,
            "cache_key": {},
            "expected_response": {
                "key_hash": "a" * 64, "gist_len": 5, "original_seq_len": 19,
            },
        }],
        "provenance": {"path": "synthetic", "sha256": "c" * 64},
    }
    artifact = {
        "schema": probe.PREPARED_SCHEMA, "status": "prepared", "design": config,
        "source_bundle": {"schema": "synthetic"}, "checkpoint": "synthetic",
        "schedule": schedule, "views": views, "extraction_manifest": manifest,
    }
    artifact["freeze_digest"] = probe._digest({
        "design": config, "schedule": schedule, "views": views,
        "extraction_manifest": manifest,
    })
    return artifact


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class _Opener:
    def __init__(self, fail_generation=None):
        self.paths = []
        self.generation_count = 0
        self.fail_generation = fail_generation

    def open(self, request, timeout):
        assert 0 < timeout <= 900
        path = urlparse(request.full_url).path
        self.paths.append(path)
        payload = json.loads(request.data)
        if path == "/v1/c2kv/extract":
            assert payload["text"] == "history"
            return _Response({
                "key_hash": "a" * 64, "gist_len": 5, "original_seq_len": 19,
                "success": True,
            })
        assert path == "/v1/chat/completions"
        self.generation_count += 1
        if self.generation_count == self.fail_generation:
            raise URLError("synthetic transport failure")
        gist_seen = any(m.get("c2kv_key_hash") for m in payload["messages"])
        return _Response({
            "choices": [{
                "message": {"content": None, "tool_calls": [{
                    "id": "generated-id", "type": "function",
                    "function": {"name": "synthetic_tool", "arguments": "{}"},
                }]},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            "metadata": {"sglang_runtime": {
                "bytes_per_kv_token": 147456,
                "c2kv_query_proj_effective": "base",
                "c2kv_query_proj_decode_verified": True,
                "c2kv_tools_dump": "full", "c2kv_gist_seen": gist_seen,
            }},
        })


def _write_prepared(tmp_path, artifact):
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_execute_submits_one_materialization_then_exactly_24_frozen_cells(tmp_path):
    artifact = _prepared_artifact()
    prepared = _write_prepared(tmp_path, artifact)
    opener = _Opener()
    out = tmp_path / "live"

    receipt_path = probe.execute_prepared(
        artifact, prepared, "http://synthetic", out, opener=opener
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cells = [json.loads(line) for line in (out / probe.CELLS_NAME).read_text(
        encoding="utf-8").splitlines()]
    assert receipt["status"] == "completed"
    assert receipt["counts"] == {
        "materializations_completed": 1, "cells_completed": 24,
        "generation_attempts": 24, "extraction_attempts": 1,
    }
    assert opener.paths == ["/v1/c2kv/extract"] + ["/v1/chat/completions"] * 24
    assert len(cells) == 24
    assert [row["generation_attempt_index"] for row in cells] == list(range(1, 25))
    assert all(row["status"] == "completed" and row["response"] and row["normalized"]
               for row in cells)
    assert receipt["attempt_journal"]["by_kind"]["generation"]["completed"] == 24
    assert receipt["attempt_journal"]["by_kind"]["extraction"]["completed"] == 1


def test_execute_stops_after_first_transport_failure_without_refill(tmp_path):
    artifact = _prepared_artifact()
    prepared = _write_prepared(tmp_path, artifact)
    opener = _Opener(fail_generation=3)
    out = tmp_path / "failed-live"

    with pytest.raises(probe.LiveAttemptError, match="synthetic transport failure"):
        probe.execute_prepared(artifact, prepared, "http://synthetic", out, opener=opener)

    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    cells = [json.loads(line) for line in (out / probe.CELLS_NAME).read_text(
        encoding="utf-8").splitlines()]
    assert receipt["status"] == "failed"
    assert receipt["counts"]["generation_attempts"] == 3
    assert receipt["counts"]["cells_completed"] == 2
    assert len(cells) == 3
    assert cells[-1]["status"] == "transport_or_backend_failure"
    assert opener.paths.count("/v1/chat/completions") == 3


def test_transport_does_not_double_finish_when_completed_journal_write_fails(tmp_path):
    class Journal:
        def __init__(self):
            self.finishes = []

        def start(self, kind, attempt_index, request_id, eval_context):
            return object()

        def finish(self, handle, status, usage=None):
            self.finishes.append(status)
            if status == "completed":
                raise OSError("synthetic journal failure")

    journal = Journal()
    transport = probe._LiveTransport(
        "http://synthetic", probe.time.monotonic() + 900, journal,
        tmp_path / "transport.jsonl", 1, 1, opener=_Opener(),
    )
    payload = {"model": "synthetic", "messages": [], "tools": []}

    with pytest.raises(OSError, match="synthetic journal failure"):
        transport.post("/v1/chat/completions", payload, 900)

    assert journal.finishes == ["completed"]
    record = json.loads((tmp_path / "transport.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "journal_finish_failed"


def test_transport_checks_deadline_before_reserving_and_clears_stale_index(tmp_path):
    transport = probe._LiveTransport(
        "http://synthetic", probe.time.monotonic() - 1,
        probe.AttemptJournal(tmp_path / "attempts.jsonl"),
        tmp_path / "transport.jsonl", 1, 1, opener=_Opener(),
    )
    transport.last_attempt_index = 9

    with pytest.raises(probe.LiveWallTimeout):
        transport.post("/v1/chat/completions", {"messages": []}, 900)

    assert transport.last_attempt_index is None
    assert transport.generation.consumed == 0
    assert not (tmp_path / "attempts.jsonl").exists()


def test_terminal_receipt_survives_attempt_journal_summary_error(tmp_path, monkeypatch):
    artifact = _prepared_artifact()
    prepared = _write_prepared(tmp_path, artifact)
    out = tmp_path / "live-summary-error"
    monkeypatch.setattr(
        probe, "summarize_attempt_journal",
        lambda path: (_ for _ in ()).throw(ValueError("synthetic summary failure")),
    )

    receipt_path = probe.execute_prepared(
        artifact, prepared, "http://synthetic", out, opener=_Opener()
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["attempt_journal_error"] == {
        "error_type": "ValueError", "error": "synthetic summary failure",
    }


def test_linux_wall_timer_arms_for_remaining_process_time_and_restores_handler(monkeypatch):
    calls = []
    old_handler = object()
    monkeypatch.setattr(probe.signal, "SIGALRM", 14, raising=False)
    monkeypatch.setattr(probe.signal, "ITIMER_REAL", 0, raising=False)
    monkeypatch.setattr(probe.signal, "getsignal", lambda signum: old_handler)
    monkeypatch.setattr(
        probe.signal, "signal", lambda signum, handler: calls.append(("handler", handler))
    )
    monkeypatch.setattr(
        probe.signal, "setitimer",
        lambda which, seconds: calls.append(("timer", seconds)), raising=False,
    )
    timer = probe._AbsoluteWallTimer(probe.time.monotonic() + 10)

    timer.arm()
    timer.cancel()

    assert calls[0][0] == "handler"
    assert calls[1][0] == "timer" and 0 < calls[1][1] <= 10
    assert calls[-2] == ("timer", 0)
    assert calls[-1] == ("handler", old_handler)
