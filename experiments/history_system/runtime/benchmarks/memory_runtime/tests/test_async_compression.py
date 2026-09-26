"""CPU contracts for bounded native background history compression."""

from __future__ import annotations

from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from history_memory.sglang_generator import SGLangEventNativeGenerator


def _chunk(index):
    return EncoderChunk(
        event_id=f"event-{index}", part_index=0, source_indices=(index,),
        source_token_start=0, source_token_end=2,
        token_ids=(100 + index, 200 + index),
    )


def _generator(monkeypatch, *, async_enabled=True, cap=4, tool_cap=None):
    if async_enabled:
        monkeypatch.setenv("C2KV_NATIVE_ASYNC_COMPRESSION", "1")
    else:
        monkeypatch.delenv("C2KV_NATIVE_ASYNC_COMPRESSION", raising=False)
    monkeypatch.delenv("C2KV_NATIVE_CROSS_TURN_PREWARM", raising=False)
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path="/checkpoint",
        model_context=128, max_new_tokens=4, max_generation_calls=4,
        max_extraction_calls=cap, max_tool_extraction_calls=tool_cap,
        timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
        shadow_feature_config=SimpleNamespace(enabled=True, prefill_layer=1, memgen_layer=None),
    )
    generator._model_binding = {"model_path": "/checkpoint"}
    generator._kv_bytes_per_token = 1
    return generator


def _memory(selected):
    return PackedMemory(MemoryView((), ()), (11, 12), (13, 14),
                        (0,), tuple(selected))


def _native_response(payload, memory):
    selected = [*payload["encoder_chunks"],
                *(row for segment in payload.get("tool_gist_segments", ())
                  for row in segment["chunks"])]
    all_rows = [*selected, *payload["compression_chunks"]]
    returned = [
        {"chunk_id": row["chunk_id"], "handle": row["handle"],
         "cache_key": f"key-{index}", "cache_hit": False,
         "original_seq_len": len(row["token_ids"]),
         "gist_len": (len(row["token_ids"]) + 7) // 8}
        for index, row in enumerate(all_rows)
    ]
    expected = memory.costs(8)
    costs = {key: expected[key] for key in (
        "system_tokens", "raw_tokens", "presented_encoder_tokens",
        "gist_tokens", "resident_kv_tokens")}
    costs.update(
        system_prefix_kv_logical_bytes=expected["system_tokens"],
        gist_prefix_kv_logical_bytes=expected["gist_tokens"],
        raw_workspace_kv_logical_bytes=expected["raw_tokens"],
        resident_kv_logical_bytes=expected["resident_kv_tokens"],
    )
    return {
        "schema": "c2kv-native-packed-generation-response-v1",
        "rid": payload["rid"], "session_id": payload["session_id"],
        "generation_id": payload["generation_id"], "text": "answer",
        "output_ids": [42], "token_logprobs": [-0.1], "finish_reason": "stop",
        "encoder_chunks": returned[:len(selected)],
        "compression_chunks": returned[len(selected):], "costs": costs,
        "extraction": {
            "requested_chunks": len(all_rows), "unique_chunks": len(all_rows),
            "cache_hits": 0, "cache_misses": len(all_rows),
            "model_calls": len(all_rows),
            "max_extraction_calls": payload["max_extraction_calls"],
        },
        "shadow_features": {
            "schema": "event-native-shadow-features-v1",
            "prefill": {"status": "captured", "layer": 1,
                        "position": {"kind": "prompt_last", "logical_position": 5},
                        "readout": "decoder_layer_output", "hidden": [0.5]},
        },
    }


def _receipt(payload, *, status, calls=0, completed=None):
    count = len(payload["chunks"]) if payload["operation"] == "submit" else 2
    if completed is None:
        completed = calls
    result = {
        "schema": "c2kv-native-prewarm-response-v1",
        "owner_id": payload["owner_id"], "job_id": payload["job_id"],
        "session_id": payload["session_id"], "status": status,
        "submitted_chunks": count,
    }
    if status in {"completed", "cancelled", "failed"}:
        result.update(
            budget_known=True, completed_chunks=completed,
            cancelled_chunks=count - completed,
            cache_hits=completed - calls, model_calls=calls,
            extraction={"model_calls": calls, "history_model_calls": calls,
                        "tool_model_calls": 0},
        )
    return result


def test_generate_returns_while_extra_job_runs_and_preserves_selected_and_shadow(monkeypatch):
    selected, extra_a, extra_b = (_chunk(i) for i in range(3))
    memory = _memory((selected,))
    native_payloads = []
    prewarm_payloads = []
    async_generator = _generator(monkeypatch, cap=3)

    def prewarm(payload):
        prewarm_payloads.append(payload)
        return _receipt(payload, status="queued" if payload["operation"] == "submit" else "running")

    async_generator._prewarm_request = prewarm
    async_generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        native_payloads.append(payload) or _native_response(payload, memory), 200)
    with async_generator.decision_scope(session_id="s"):
        async_result = async_generator.generate(
            memory, ratio=8, max_new_tokens=2,
            compression_chunks=(selected, extra_a, extra_b))
    assert [item["operation"] for item in prewarm_payloads] == ["submit"]
    assert prewarm_payloads[0]["after_native_rid"] == native_payloads[0]["rid"]
    assert async_generator._prewarm_job is not None
    assert native_payloads[0]["compression_chunks"] == []
    assert native_payloads[0]["max_extraction_calls"] == 1
    assert async_result.stats["async_compression"]["poll_rpc_count"] == 0
    assert async_result.stats["async_compression"]["reserved_chunks"] == 2
    assert async_result.stats["async_compression"]["response_waited_for_extras"] is False
    assert async_generator.extraction_calls_reserved == 1
    assert (async_generator.extraction_calls_reserved
            + async_generator._background_reserved_chunks()) == 3
    assert async_generator._prewarm_job is not None

    legacy = _generator(monkeypatch, async_enabled=False)
    legacy_payloads = []
    legacy._post_native_generate = lambda payload, *_args, **_kwargs: (
        legacy_payloads.append(payload) or _native_response(payload, memory), 200)
    with legacy.decision_scope(session_id="s"):
        legacy_result = legacy.generate(
            memory, ratio=8, max_new_tokens=2,
            compression_chunks=(selected, extra_a, extra_b))
    assert native_payloads[0]["encoder_chunks"] == legacy_payloads[0]["encoder_chunks"]
    assert native_payloads[0]["system_input_ids"] == legacy_payloads[0]["system_input_ids"]
    assert native_payloads[0]["workspace_input_ids"] == legacy_payloads[0]["workspace_input_ids"]
    assert async_result.token_ids == legacy_result.token_ids == (42,)
    assert async_result.stats["shadow_features"] == legacy_result.stats["shadow_features"]


def test_poll_settles_once_and_capacity_never_overruns(monkeypatch):
    generator = _generator(monkeypatch, cap=3)
    selected, extra_a, extra_b = (_chunk(i) for i in range(3))
    requests = []

    def prewarm(payload):
        requests.append(payload)
        if payload["operation"] == "submit":
            return _receipt(payload, status="queued")
        result = _receipt(payload, status="completed", calls=2, completed=2)
        result["results"] = [
            {"handle": handle, "cache_hit": False}
            for handle in generator._prewarm_job["submitted_handles"]]
        return result

    generator._prewarm_request = prewarm
    offered = generator.offer_background_chunks(
        (selected, extra_a, extra_b), ratio=8, session_id="s",
        outer_request_id="outer", selected_chunks=(selected,))
    assert offered["submitted_chunks"] == 2
    assert requests[0]["max_extraction_calls"] == 2
    assert generator._background_reserved_chunks() == 2
    assert generator.reconcile_cross_turn_prewarm(operation="poll")["status"] == "completed"
    assert generator.extraction_calls_reserved == 2
    assert generator._background_reserved_chunks() == 0
    assert generator.reconcile_cross_turn_prewarm(operation="poll") is None
    assert generator.extraction_calls_reserved == 2
    assert generator.session_cache_info()["async_compression"]["settled_model_calls"] == 2


def test_running_offer_defers_and_close_cancels_with_exact_charge(monkeypatch):
    generator = _generator(monkeypatch, cap=3)
    chunks = (_chunk(1), _chunk(2))
    operations = []

    def prewarm(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "submit":
            return _receipt(payload, status="queued")
        if payload["operation"] == "poll":
            return _receipt(payload, status="running")
        result = _receipt(payload, status="cancelled", calls=1, completed=1)
        result["results"] = [{"handle": generator._prewarm_job["submitted_handles"][0],
                              "cache_hit": False}]
        return result

    generator._prewarm_request = prewarm
    generator.offer_background_chunks(chunks, ratio=8, session_id="s",
                                      outer_request_id="outer", selected_chunks=())
    deferred = generator.offer_background_chunks(chunks, ratio=8,
                                                 session_id="s", outer_request_id="outer",
                                                 selected_chunks=())
    assert deferred["status"] == "deferred"
    assert operations == ["submit", "poll"]
    generator.close_session()
    assert operations == ["submit", "poll", "cancel"]
    assert generator.extraction_calls_reserved == 1
    assert generator._prewarm_job is None


def test_repeated_nonblocking_polls_are_coalesced_without_delaying_cancel(monkeypatch):
    generator = _generator(monkeypatch, cap=3)
    now = [100.0]
    monkeypatch.setattr("history_memory.sglang_generator.time.perf_counter", lambda: now[0])
    operations = []

    def prewarm(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "submit":
            return _receipt(payload, status="queued")
        if payload["operation"] == "poll":
            return _receipt(payload, status="running")
        result = _receipt(payload, status="cancelled")
        result["results"] = []
        return result

    generator._prewarm_request = prewarm
    generator.offer_background_chunks((_chunk(0), _chunk(1)), ratio=8,
                                      session_id="s", outer_request_id="outer",
                                      selected_chunks=())
    assert generator.reconcile_cross_turn_prewarm(operation="poll")["status"] == "running"
    assert generator.reconcile_cross_turn_prewarm(operation="poll")["status"] == "running"
    assert operations == ["submit", "poll"]
    assert generator._background_reserved_chunks() == 2
    stats = generator.session_cache_info()["async_compression"]
    assert stats["poll_rpc_count"] == 1
    assert stats["coalesced_poll_count"] == 1
    now[0] += 0.051
    generator.reconcile_cross_turn_prewarm(operation="poll")
    assert operations == ["submit", "poll", "poll"]
    generator.close_session()
    assert operations[-1] == "cancel"
    assert generator._background_reserved_chunks() == 0


def test_confirmed_background_handles_are_not_resubmitted_in_same_session(monkeypatch):
    generator = _generator(monkeypatch, cap=4)
    chunks = (_chunk(0), _chunk(1))
    submits = []

    def prewarm(payload):
        if payload["operation"] == "submit":
            submits.append(payload)
            return _receipt(payload, status="queued")
        result = _receipt(payload, status="completed", calls=2, completed=2)
        result["results"] = [{"handle": handle, "cache_hit": False}
                             for handle in generator._prewarm_job["submitted_handles"]]
        return result

    generator._prewarm_request = prewarm
    generator.offer_background_chunks(chunks, ratio=8, session_id="s",
                                      outer_request_id="outer", selected_chunks=())
    generator.reconcile_cross_turn_prewarm(operation="poll")
    assert generator.session_cache_info()["materialized_history_handle_count"] == 2
    assert generator.offer_background_chunks(chunks, ratio=8, session_id="s",
                                             outer_request_id="outer", selected_chunks=()) is None
    assert len(submits) == 1
    with generator.decision_scope(session_id="s"):
        pass
    assert generator.offer_background_chunks(chunks, ratio=8, session_id="s",
                                             outer_request_id="outer", selected_chunks=()) is None
    assert len(submits) == 1


def test_confirmed_selected_handle_still_checks_engine_after_eviction(monkeypatch):
    generator = _generator(monkeypatch, cap=2)
    selected = _chunk(0)
    memory = _memory((selected,))
    generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        _native_response(payload, memory), 200)
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=2)
    assert generator.session_cache_info()["materialized_history_handle_count"] == 1
    with generator.decision_scope(session_id="s"):
        # The second native response is a miss even though this session knows the handle.
        result = generator.generate(memory, ratio=8, max_new_tokens=2)
    assert result.stats["extracted_chunks"] == 1
    assert generator.extraction_calls_reserved == 2


def test_materialized_handles_clear_on_failure_and_session_change(monkeypatch):
    generator = _generator(monkeypatch, cap=3)
    chunk = _chunk(0)
    memory = _memory((chunk,))
    generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        _native_response(payload, memory), 200)
    with generator.decision_scope(session_id="old"):
        generator.generate(memory, ratio=8, max_new_tokens=2)
    assert generator.session_cache_info()["materialized_history_handle_count"] == 1
    with generator.decision_scope(session_id="new"):
        assert generator.session_cache_info()["materialized_history_handle_count"] == 0
    with pytest.raises(RuntimeError, match="decision failed"):
        with generator.decision_scope(session_id="new"):
            generator.generate(memory, ratio=8, max_new_tokens=2)
            raise RuntimeError("decision failed")
    assert generator.session_cache_info()["materialized_history_handle_count"] == 0


def test_selected_tool_uses_separate_budget_when_admitting_history(monkeypatch):
    generator = _generator(monkeypatch, cap=1, tool_cap=1)
    generator._tool_projection_identity = "tool-projection"
    tool = replace(_chunk(0), projection_set="tool")
    history = _chunk(1)
    submitted = []
    generator._prewarm_request = lambda payload: (
        submitted.append(payload) or _receipt(payload, status="queued"))
    result = generator.offer_background_chunks(
        (history,), ratio=8, session_id="s", outer_request_id="outer",
        selected_chunks=(tool,))
    assert result["submitted_chunks"] == 1
    assert submitted[0]["max_extraction_calls"] == 1


def test_native_failure_cancels_and_settles_background_once(monkeypatch):
    generator = _generator(monkeypatch, cap=2)
    memory = _memory(())
    extras = (_chunk(0), _chunk(1))
    operations = []

    def prewarm(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "submit":
            return _receipt(payload, status="queued")
        result = _receipt(payload, status="cancelled", calls=1, completed=1)
        result["results"] = [{"handle": generator._prewarm_job["submitted_handles"][0],
                              "cache_hit": False}]
        return result

    generator._prewarm_request = prewarm
    generator._post_native_generate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("native failed"))
    with pytest.raises(RuntimeError, match="native failed"):
        with generator.decision_scope(session_id="s"):
            generator.generate(memory, ratio=8, max_new_tokens=2,
                               compression_chunks=extras)
    assert operations == ["submit", "cancel"]
    assert generator.extraction_calls_reserved == 1
    assert generator.reconcile_cross_turn_prewarm(operation="poll") is None


def test_new_session_cancels_old_job_before_scope(monkeypatch):
    generator = _generator(monkeypatch, cap=2)
    operations = []

    def prewarm(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "submit":
            return _receipt(payload, status="queued")
        result = _receipt(payload, status="cancelled")
        result["results"] = []
        return result

    generator._prewarm_request = prewarm
    generator.offer_background_chunks((_chunk(0), _chunk(1)), ratio=8,
                                      session_id="old", outer_request_id="outer",
                                      selected_chunks=())
    with generator.decision_scope(session_id="new"):
        assert generator._prewarm_job is None
    assert operations == ["submit", "cancel"]
    assert generator.extraction_calls_reserved == 0


def test_provider_offers_ready_history_during_native_http_without_spending_selected_cap(monkeypatch):
    generator = _generator(monkeypatch, cap=2)
    selected, known = _chunk(0), _chunk(1)
    memory = _memory((selected,))
    entered, release = threading.Event(), threading.Event()
    observed = []

    def native_post(payload, *_args, **_kwargs):
        observed.append(payload)
        entered.set()
        assert release.wait(2)
        return _native_response(payload, memory), 200

    def prewarm(payload):
        assert entered.is_set() and not release.is_set()
        observed.append(payload)
        release.set()
        return _receipt(payload, status="queued")

    generator._post_native_generate = native_post
    generator._prewarm_request = prewarm
    with generator.decision_scope(session_id="s"):
        result = generator.generate(
            memory, ratio=8, max_new_tokens=2,
            background_chunk_provider=lambda: (known,) if entered.is_set() else None,
        )
    assert observed[0]["max_extraction_calls"] == 1
    assert observed[1]["operation"] == "submit"
    assert observed[1]["after_native_rid"] == observed[0]["rid"]
    assert result.stats["async_compression"]["provider_offer"]["submitted_chunks"] == 1
    assert generator.extraction_calls_reserved == 1
    assert generator._background_reserved_chunks() == 1


def test_provider_interrupt_waits_for_foreground_receipt_before_cleanup(monkeypatch):
    generator = _generator(monkeypatch, cap=1)
    selected = _chunk(0)
    memory = _memory((selected,))
    entered, release = threading.Event(), threading.Event()

    def native_post(payload, *_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return _native_response(payload, memory), 200

    def provider():
        if not entered.is_set():
            return None
        release.set()
        raise KeyboardInterrupt("stop after foreground receipt")

    generator._post_native_generate = native_post
    with pytest.raises(KeyboardInterrupt, match="stop after foreground receipt"):
        with generator.decision_scope(session_id="s"):
            generator.generate(
                memory, ratio=8, max_new_tokens=2,
                background_chunk_provider=provider,
            )
    assert generator.extraction_calls_reserved == 1
    assert generator._prewarm_job is None


def test_async_admission_is_separate_from_legacy_prewarm(monkeypatch):
    generator = _generator(monkeypatch)
    generator._model_binding = None
    native = {"model_binding": {"model_path": "/checkpoint"},
              "kv_bytes_per_token": 1, "serving_features": {}}
    generator._read_json = lambda *args, **kwargs: (
        {"model_path": "/checkpoint", "c2kv_native_packed": native}, 200)
    with pytest.raises(Exception, match="nonblocking-history-v1 admission"):
        generator._ensure_model_info()
    native["serving_features"]["async_compression"] = "nonblocking-history-v1"
    generator._ensure_model_info()
