"""CPU contracts for whole-event budget admission of optional prewarm work."""

from __future__ import annotations

from dataclasses import replace
from threading import Event

import pytest
from benchmarks.memory_runtime.tests.test_async_compression import (
    _chunk,
    _generator,
    _memory,
    _native_response,
    _receipt,
)


def _fit_generator(monkeypatch, *, enabled=True, cap=4):
    if enabled:
        monkeypatch.setenv("C2KV_NATIVE_PREWARM_FIT_BUDGET", "1")
    else:
        monkeypatch.delenv("C2KV_NATIVE_PREWARM_FIT_BUDGET", raising=False)
    generator = _generator(monkeypatch, cap=cap)
    assert generator.background_fit_budget_enabled is enabled
    return generator


def _event_parts(*, second_ratio=None):
    first = replace(
        _chunk(0), event_id="shared-event", part_index=0,
        source_indices=(0,), source_token_start=0, source_token_end=9,
        token_ids=tuple(range(100, 109)),
    )
    second = replace(
        _chunk(1), event_id="shared-event", part_index=1,
        source_indices=(0,), source_token_start=9, source_token_end=18,
        token_ids=tuple(range(109, 118)), compression_ratio=second_ratio,
    )
    return first, second


def _capture_submissions(generator):
    submissions = []

    def prewarm(payload):
        if payload["operation"] == "submit":
            submissions.append(payload)
            return _receipt(payload, status="queued")
        handles = generator._prewarm_job["submitted_handles"]
        return {
            "schema": "c2kv-native-prewarm-response-v1",
            "owner_id": payload["owner_id"], "job_id": payload["job_id"],
            "session_id": payload["session_id"], "status": "completed",
            "submitted_chunks": len(handles), "budget_known": True,
            "completed_chunks": len(handles), "cancelled_chunks": 0,
            "cache_hits": 0, "model_calls": len(handles),
            "extraction": {"model_calls": len(handles),
                           "history_model_calls": len(handles), "tool_model_calls": 0},
            "results": [{"handle": handle, "cache_hit": False} for handle in handles],
        }

    generator._prewarm_request = prewarm
    return submissions


def test_selected_input_and_shadow_survive_rejected_optional_unit(monkeypatch):
    selected = _chunk(0)
    extra = replace(
        _chunk(1), source_token_end=16, token_ids=tuple(range(200, 216)))
    memory = _memory((selected,))
    observed = []

    for enabled in (False, True):
        generator = _fit_generator(monkeypatch, enabled=enabled, cap=2)
        submissions = _capture_submissions(generator)
        payloads = []

        def native_post(payload, *_args, _payloads=payloads, **_kwargs):
            _payloads.append(payload)
            return _native_response(payload, memory), 200

        generator._post_native_generate = native_post
        with generator.decision_scope(session_id="s"):
            result = generator.generate(
                memory, ratio=8, max_new_tokens=2,
                compression_chunks=(selected, extra),
                background_history_budget_tokens=1,
            )
        assert len(submissions) == (0 if enabled else 1)
        if submissions:
            assert len(submissions[0]["chunks"]) == 1
        observed.append((payloads[0], result))

    legacy_payload, legacy_result = observed[0]
    fitted_payload, fitted_result = observed[1]
    assert fitted_payload["encoder_chunks"] == legacy_payload["encoder_chunks"]
    assert fitted_payload["system_input_ids"] == legacy_payload["system_input_ids"]
    assert fitted_payload["workspace_input_ids"] == legacy_payload["workspace_input_ids"]
    assert fitted_payload["shadow_features"] == legacy_payload["shadow_features"]
    assert fitted_result.token_ids == legacy_result.token_ids
    assert fitted_result.stats["shadow_features"] == legacy_result.stats["shadow_features"]
    fit_stats = fitted_result.stats["async_compression"]["budget_fit_admission"]
    assert fit_stats["skipped_unit_checks"] == 1
    assert fit_stats["skipped_chunk_checks"] == 1


@pytest.mark.parametrize(
    "second_ratio, budget, admitted",
    [(None, 4, True), (None, 3, False), (16, 3, True), (16, 2, False)],
)
def test_whole_event_uses_sum_of_per_chunk_ceil_costs(
    monkeypatch, second_ratio, budget, admitted
):
    generator = _fit_generator(monkeypatch)
    parts = _event_parts(second_ratio=second_ratio)
    submissions = _capture_submissions(generator)
    result = generator.offer_background_chunks(
        parts, ratio=8, session_id="s", outer_request_id="outer",
        selected_chunks=(), history_budget_tokens=budget)

    if admitted:
        assert result["submitted_chunks"] == 2
        assert len(submissions) == 1
        assert [row["compression_ratio"] for row in submissions[0]["chunks"]] == [
            8, second_ratio or 8]
    else:
        assert result is None
        assert submissions == []
        stats = generator.session_cache_info()["async_compression"]["budget_fit_admission"]
        assert stats["skipped_unit_checks"] == 1
        assert stats["skipped_chunk_checks"] == 2


def test_cached_part_cannot_make_oversized_remainder_admissible(monkeypatch):
    generator = _fit_generator(monkeypatch)
    first, second = _event_parts()
    cached_handle = generator._chunk_payload(first, 8)["handle"]
    generator._record_materialized_history("s", (cached_handle,))
    submissions = _capture_submissions(generator)

    result = generator.offer_background_chunks(
        (first, second), ratio=8, session_id="s", outer_request_id="outer",
        selected_chunks=(), history_budget_tokens=3)

    assert result is None
    assert submissions == []
    assert generator._known_materialized_history("s") == {cached_handle}
    stats = generator.session_cache_info()["async_compression"]["budget_fit_admission"]
    assert stats["skipped_chunk_checks"] == 2


def test_flag_off_keeps_previous_optional_offer(monkeypatch):
    generator = _fit_generator(monkeypatch, enabled=False)
    submissions = _capture_submissions(generator)
    result = generator.offer_background_chunks(
        _event_parts(), ratio=8, session_id="s", outer_request_id="outer",
        selected_chunks=(), history_budget_tokens=1)

    assert result["submitted_chunks"] == 2
    assert len(submissions[0]["chunks"]) == 2
    assert "budget_fit_admission" not in generator.session_cache_info()["async_compression"]


def test_http_provider_receives_budget_and_filters_whole_unit(monkeypatch):
    generator = _fit_generator(monkeypatch)
    memory = _memory((_chunk(9),))
    release = Event()
    observed_budgets = []
    submissions = _capture_submissions(generator)
    offer = generator.offer_background_chunks

    def capture_offer(chunks, **kwargs):
        observed_budgets.append(kwargs["history_budget_tokens"])
        try:
            return offer(chunks, **kwargs)
        finally:
            release.set()

    def native_post(payload, *_args, **_kwargs):
        assert release.wait(2)
        return _native_response(payload, memory), 200

    generator.offer_background_chunks = capture_offer
    generator._post_native_generate = native_post
    with generator.decision_scope(session_id="s"):
        result = generator.generate(
            memory, ratio=8, max_new_tokens=2,
            background_chunk_provider=_event_parts,
            background_history_budget_tokens=3,
        )

    assert observed_budgets == [3]
    assert submissions == []
    assert result.stats["async_compression"]["budget_fit_admission"]["skipped_unit_checks"] == 1
