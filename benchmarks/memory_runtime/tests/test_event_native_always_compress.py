"""Tokenizer-only contracts for explicit event-native always-compress routes."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmarks.memory_runtime import event_native_server as server
from benchmarks.memory_runtime.always_compress import (
    ALWAYS_COMPRESSION_POLICY,
    CapacityInfeasible,
)
from benchmarks.memory_runtime.event_native_controls import (
    ALL_VIEW_MODES,
    ALWAYS_COMPRESS_VIEW_MODES,
    FINITE_VIEW_MODES,
    build_event_native_controller,
    describe_event_native_route,
)
from benchmarks.memory_runtime.event_native_eval_policy import (
    load_eval_policy,
    resolve_event_native_eval_policy,
)
from benchmarks.memory_runtime.event_native_exact_policy import (
    EventNativeExactController,
)
from benchmarks.memory_runtime.policy import BudgetExceeded
from benchmarks.memory_runtime.tests.test_event_native_exact import (
    MAX_NEW_TOKENS,
    PROTECTED_EVENT_ID,
    RATIO,
    TARGET_EVENT_ID,
    Tokenizer,
    _draft_calls,
    _messages,
    _payload,
    _settings,
)
from benchmarks.memory_runtime.tests.test_event_native_eval_policy import profile
from history_memory.events import EventStore


NEW_MODES = frozenset(
    {
        "ac_gist_static",
        "ac_protect",
        "ac_exact_once",
        "ac_exact_persistent",
        "ac_full_shared",
        "raw_exact_shared",
    }
)


def _controller(mode: str, packing: dict, policy: dict) -> EventNativeExactController:
    return EventNativeExactController(
        Tokenizer(),
        packing=copy.deepcopy(packing),
        policy=copy.deepcopy(policy),
        mode=mode,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
    )


def _wide_settings() -> tuple[dict, dict]:
    packing, policy = _settings(history_budget_bytes=10_000_000)
    probe = _controller("ac_gist_static", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    policy["history_budget_bytes"] = (
        probe.metadata["capacity_gate"]["full_history_bytes"] + 100_000
    )
    policy["workspace_budget_bytes"] = policy["history_budget_bytes"]
    return packing, policy


def test_new_routes_are_an_explicit_namespace_without_mutating_frozen_catalog() -> None:
    assert ALWAYS_COMPRESS_VIEW_MODES == NEW_MODES
    assert ALL_VIEW_MODES == FINITE_VIEW_MODES | NEW_MODES
    assert NEW_MODES.isdisjoint(FINITE_VIEW_MODES)

    for mode in sorted(NEW_MODES):
        with pytest.raises(ValueError, match="compression_policy"):
            describe_event_native_route(mode)
        route = describe_event_native_route(
            mode, compression_policy=ALWAYS_COMPRESSION_POLICY
        )
        assert route["view_mode"] == mode
        assert route["compression_policy"] == ALWAYS_COMPRESSION_POLICY
        assert route["implementation_profile"] == "event-native-always-compress-v1"
        assert route["training_static"] is False
        assert route["legacy_1088_equivalent"] is False

        packing, policy = _settings()
        controller = build_event_native_controller(
            Tokenizer(),
            packing=packing,
            policy=policy,
            view_mode=mode,
            compression_policy=ALWAYS_COMPRESSION_POLICY,
        )
        assert isinstance(controller, EventNativeExactController)

    with pytest.raises(ValueError, match="requires a new always-compress route"):
        describe_event_native_route(
            "static", compression_policy=ALWAYS_COMPRESSION_POLICY
        )

    legacy_only_candidate = "ac_acquire_for_next"
    assert legacy_only_candidate not in ALL_VIEW_MODES
    with pytest.raises(ValueError, match="view_mode"):
        describe_event_native_route(
            legacy_only_candidate,
            compression_policy=ALWAYS_COMPRESSION_POLICY,
        )
    packing, policy = _settings()
    with pytest.raises(ValueError, match="mode"):
        _controller(legacy_only_candidate, packing, policy)
    with pytest.raises(ValueError, match="view_mode"):
        resolve_event_native_eval_policy(
            profile(), view_mode=legacy_only_candidate
        )


def test_full_fits_budget_but_compressible_history_still_uses_real_gist() -> None:
    packing, policy = _wide_settings()
    always = _controller("ac_gist_static", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    legacy = EventNativeExactController(
        Tokenizer(),
        packing=copy.deepcopy(packing),
        policy=copy.deepcopy(policy),
        mode="capacity_protect",
    ).prepare(_payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)

    gate = always.metadata["capacity_gate"]
    assert gate["full_history_bytes"] <= gate["history_budget_bytes"]
    assert gate["activated"] is True
    assert gate["full_identity_bypass"] is False
    assert gate["eligible_event_ids"]
    assert always.memory.chunks
    assert always.memory.view.gist_event_ids
    assert always.memory.view.evidence_event_ids == ()
    assert always.metadata["selected_event_ids"] == []
    assert always.metadata["min_gist_reservation_met"] is True
    assert legacy.metadata["capacity_gate"]["full_identity_bypass"] is True
    assert legacy.memory.chunks == ()


def test_zero_compressible_history_is_the_only_raw_identity_exception() -> None:
    packing, policy = _wide_settings()
    payload = {
        "session_id": "zero-eligible/session",
        "decision_key": "d1",
        "messages": [
            {"role": "system", "content": "Use the current request."},
            {"role": "user", "content": "Begin."},
        ],
        "tools": [],
    }
    always = _controller("ac_gist_static", packing, policy).prepare(
        payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    full = build_event_native_controller(
        Tokenizer(),
        packing=packing,
        policy=policy,
        view_mode="full_original",
    ).prepare(payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)

    assert always.memory == full.memory
    assert always.memory.chunks == ()
    assert always.metadata["no_eligible_history"] is True
    assert always.metadata["capacity_gate"]["natural_zero_eligible_raw"] is True
    assert always.metadata["capacity_gate"]["full_identity_bypass"] is False
    assert always.metadata["source_coverage"]["eligible_source_indices"] == []


def test_min_gist_selection_keeps_prepacking_eligibility_and_classifies_drops() -> None:
    packing, policy = _settings(
        history_budget_bytes=10_000_000, max_chunk_tokens=256
    )
    packing["max_chunks"] = 1
    prepared = _controller("ac_gist_static", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    coverage = prepared.metadata["source_coverage"]

    assert TARGET_EVENT_ID in prepared.metadata["capacity_gate"]["eligible_event_ids"]
    assert TARGET_EVENT_ID in coverage["eligible_event_ids"]
    assert "before max_chunks" in prepared.metadata["capacity_gate"][
        "eligibility_stage"
    ]
    assert coverage["eligibility_stage"] == prepared.metadata["capacity_gate"][
        "eligibility_stage"
    ]
    assert TARGET_EVENT_ID not in prepared.memory.view.gist_event_ids
    assert TARGET_EVENT_ID in coverage["native_packing_omitted_event_ids"]
    assert coverage["native_packing_omitted_source_indices"]
    assert prepared.memory.view.gist_event_ids
    assert len(prepared.memory.chunks) == 1
    assert prepared.metadata["min_gist_reservation_met"] is True
    assert coverage["complete_history_coverage"] is False
    assert prepared.metadata["compression_ratio"]["includes_coverage_loss"] is True


def test_budget_loss_is_reported_and_too_small_a_budget_fails_explicitly() -> None:
    packing, policy = _settings(history_budget_bytes=5_000)
    prepared = _controller("ac_gist_static", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    coverage = prepared.metadata["source_coverage"]

    assert prepared.memory.view.gist_event_ids
    assert prepared.metadata["actual_history_bytes"] <= policy["history_budget_bytes"]
    assert coverage["budget_omitted_event_ids"]
    assert coverage["budget_omitted_source_indices"]
    assert coverage["unrepresented_source_indices"]
    assert coverage["complete_history_coverage"] is False
    assert prepared.metadata["compression_ratio"]["includes_coverage_loss"] is True

    infeasible_policy = copy.deepcopy(policy)
    infeasible_policy["history_budget_bytes"] = 1
    with pytest.raises(CapacityInfeasible, match="No complete eligible native gist block"):
        _controller("ac_gist_static", packing, infeasible_policy).prepare(
            _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )


def test_min_gist_skips_large_priority_blocks_to_fit_protected_evidence() -> None:
    packing, policy = _settings(
        history_budget_bytes=50_000, max_chunk_tokens=256
    )
    policy["workspace_budget_bytes"] = 45_000

    controller = _controller("ac_exact_once", packing, policy)
    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )

    assert prepared.metadata["protected_event_ids"] == [PROTECTED_EVENT_ID]
    assert prepared.metadata["min_gist_reservation_event_id"] == "exact/session:m6"
    assert [
        item["event_id"]
        for item in prepared.metadata["min_gist_reservation_skipped_events"]
    ] == [TARGET_EVENT_ID, "exact/session:m7"]
    assert prepared.metadata["admission_failures"] == []
    assert prepared.metadata["history_bytes"] <= policy["history_budget_bytes"]
    assert prepared.metadata["evidence_bytes"] <= policy["workspace_budget_bytes"]
    reconsidered = controller.reconsider(
        prepared,
        copy.deepcopy(_draft_calls()),
        draft_text="unsubmitted native draft",
    )
    assert reconsidered["regenerate"] is False
    assert reconsidered["decision"]["reason"] == "budget_exhausted"
    assert reconsidered["memory"] == prepared.memory
    assert (
        reconsidered["metadata"]["min_gist_reservation_event_id"]
        == prepared.metadata["min_gist_reservation_event_id"]
    )


def test_mandatory_selection_overflow_is_reported_as_capacity_infeasible(
    monkeypatch,
) -> None:
    packing, policy = _settings(history_budget_bytes=50_000)

    def fail_mandatory(*args, **kwargs):
        raise BudgetExceeded(
            required_event_ids=(PROTECTED_EVENT_ID,),
            required_cost=50_001,
            budget=50_000,
        )

    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_exact_policy."
        "ExactRecoveryMemory.prepare_decision",
        fail_mandatory,
    )
    with pytest.raises(CapacityInfeasible, match="Necessary event-native"):
        _controller("ac_protect", packing, policy).prepare(
            _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )


def test_once_and_persistent_share_first_view_and_preserve_reserved_gist_on_upgrade() -> None:
    packing, policy = _wide_settings()
    results = []
    for mode in ("ac_exact_once", "ac_exact_persistent"):
        controller = _controller(mode, packing, policy)
        first = controller.prepare(
            _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )
        upgraded = controller.reconsider(
            first,
            copy.deepcopy(_draft_calls()),
            draft_text="unsubmitted native draft",
        )
        assert upgraded["regenerate"] is True
        assert upgraded["metadata"]["decision_index"] == 1
        assert upgraded["metadata"]["actual_history_bytes"] <= policy[
            "history_budget_bytes"
        ]
        assert upgraded["metadata"]["min_gist_reservation_met"] is True
        first_reservation = first.metadata["min_gist_reservation_event_id"]
        reservation = upgraded["metadata"]["min_gist_reservation_event_id"]
        assert reservation == first_reservation
        assert reservation in upgraded["memory"].view.gist_event_ids
        assert TARGET_EVENT_ID in upgraded["memory"].view.raw_event_ids
        assert TARGET_EVENT_ID in upgraded["memory"].view.gist_event_ids
        assert TARGET_EVENT_ID in upgraded["metadata"]["source_coverage"][
            "raw_gist_overlap_event_ids"
        ]
        assert upgraded["memory"].costs(RATIO)["raw_gist_overlap_events"] == 1
        repeated = controller.reconsider(
            first,
            copy.deepcopy(_draft_calls()),
            draft_text="unsubmitted native draft",
        )
        assert repeated["memory"] == upgraded["memory"]
        assert repeated["metadata"] == upgraded["metadata"]
        results.append((first, upgraded))

    assert results[0][0].memory == results[1][0].memory
    assert results[0][1]["memory"] == results[1][1]["memory"]


def test_raw_exact_shared_refills_raw_budget_without_a_gist_reservation() -> None:
    packing, policy = _settings(history_budget_bytes=50_000)
    raw = _controller("raw_exact_shared", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    compressed = _controller("ac_protect", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )

    assert raw.memory.view.gist_event_ids == ()
    assert raw.metadata["min_gist_reservation_required"] is False
    assert raw.metadata["actual_history_bytes"] <= policy["history_budget_bytes"]
    assert raw.metadata["refilled_event_ids"]
    assert PROTECTED_EVENT_ID in raw.metadata["selected_event_ids"]
    assert PROTECTED_EVENT_ID in compressed.metadata["selected_event_ids"]
    assert compressed.memory.view.gist_event_ids


def test_current_request_is_common_input_and_recent_tool_history_is_charged() -> None:
    packing, policy = _wide_settings()
    packing["recent_tool_events"] = 1
    prepared = _controller("ac_gist_static", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    store = EventStore.from_messages("exact/session", _messages())
    recent_tool = store.event(PROTECTED_EVENT_ID)

    assert PROTECTED_EVENT_ID in prepared.memory.view.raw_event_ids
    assert set(recent_tool.source_indices) <= set(prepared.memory.raw_source_indices)
    assert len(_messages()) - 1 in prepared.memory.raw_source_indices
    assert prepared.metadata["actual_raw_history_tokens"] > 0
    assert prepared.metadata["same_prefix_full_reference"]["common_live_tokens"] > 0
    assert prepared.metadata["actual_history_bytes"] == (
        prepared.metadata["actual_gist_tokens"]
        + prepared.metadata["actual_raw_history_tokens"]
    ) * policy["kv_bytes_per_token"]
    assert prepared.metadata["compression_ratio"]["common_live_bytes"] == (
        prepared.metadata["same_prefix_full_reference"]["common_live_bytes"]
    )


def test_persistent_route_accepts_new_tool_results_without_aging_same_decision() -> None:
    packing, policy = _wide_settings()
    policy["lease_decisions"] = 3
    controller = _controller("ac_exact_persistent", packing, policy)
    first_payload = _payload()
    first = controller.prepare(
        first_payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    acquired = controller.reconsider(
        first,
        copy.deepcopy(_draft_calls()),
        draft_text="unsubmitted native draft",
    )
    assert acquired["regenerate"] is True
    assert controller.prepare(
        copy.deepcopy(first_payload), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    ) is first
    assert controller.reconsider(
        first,
        copy.deepcopy(_draft_calls()),
        draft_text="unsubmitted native draft",
    ) == acquired

    messages = _messages() + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "new-call",
                    "type": "function",
                    "function": {
                        "name": "use_record",
                        "arguments": '{"record_id":"item-17"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "new-call",
            "content": '{"status":"success"}',
        },
        {"role": "user", "content": "Report the new result."},
    ]
    second = controller.prepare(
        _payload(decision_key="d2", messages=messages),
        ratio=RATIO,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    store = EventStore.from_messages("exact/session", messages)
    new_event = next(
        event
        for event in store.events
        if event.kind == "tool_event" and "new-call" in event.tool_call_ids
    )
    represented_sources = set(second.memory.raw_source_indices) | {
        index for chunk in second.memory.chunks for index in chunk.source_indices
    }

    assert second.metadata["decision_index"] == 2
    assert TARGET_EVENT_ID in second.metadata["retained_event_ids"]
    assert set(new_event.source_indices) <= represented_sources
    assert second.metadata["min_gist_reservation_met"] is True
    assert second.metadata["actual_history_bytes"] <= policy["history_budget_bytes"]


def test_config_and_server_child_roundtrip_preserve_explicit_identity(tmp_path) -> None:
    config = Path(__file__).parents[1] / "configs" / "a_always_compress_v1.eval-policy.json"
    loaded = load_eval_policy(config)
    assert loaded["policy_id"] == "a-pre-b-always-compress-v1"
    assert loaded["policy"]["max_retrieved_events"] == 1
    for mode in sorted(NEW_MODES):
        runtime = resolve_event_native_eval_policy(
            profile(), view_mode=mode, policy_override=loaded
        )
        assert runtime["policy_id"] == loaded["policy_id"]
        assert runtime["runtime_recovery_cap"] == (
            1
            if mode
            in {
                "ac_exact_once",
                "ac_exact_persistent",
                "ac_full_shared",
                "raw_exact_shared",
            }
            else 0
        )

    from benchmarks.memory_runtime.event_native_method_contract import (
        current_method_contract,
    )

    frozen_legacy = copy.deepcopy(loaded)
    frozen_legacy["schema"] = "a-event-native-eval-policy-v2"
    frozen_legacy["method_contract"] = current_method_contract()
    with pytest.raises(ValueError, match="frozen legacy v2 method_contract"):
        resolve_event_native_eval_policy(
            profile(),
            view_mode="ac_exact_once",
            policy_override=frozen_legacy,
        )

    argv = [
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "always-compress-test",
        "--view-mode",
        "ac_exact_once",
        "--compression-policy",
        ALWAYS_COMPRESSION_POLICY,
        "--history-view-protocol",
        "fixed-budget-main",
        "--ratio",
        str(RATIO),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--task-ids",
        "synthetic-task",
        "--max-decisions",
        "1",
        "--max-generation-calls",
        "2",
        "--max-wall-seconds",
        "30",
        "--eval-policy",
        str(config),
    ]
    args = server.parser().parse_args(argv)
    child = server.parser().parse_args(server._child_command(args)[3:])

    assert child.serve_child is True
    assert child.view_mode == "ac_exact_once"
    assert child.compression_policy == ALWAYS_COMPRESSION_POLICY
    assert child.history_view_protocol == "fixed-budget-main"
    assert child.eval_policy.resolve() == config.resolve()

    missing_policy = server.parser().parse_args(
        [value for value in argv if value not in ("--compression-policy", ALWAYS_COMPRESSION_POLICY)]
    )
    with pytest.raises(ValueError, match="compression_policy"):
        server._serve(missing_policy)
    assert not missing_policy.out.exists()
