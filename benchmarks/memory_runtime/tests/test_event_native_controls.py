"""Pure-tokenizer contracts for finite event-native controller routing."""

from __future__ import annotations

import copy

import pytest

from benchmarks.memory_runtime import event_native_exact_policy as exact_policy_module
from benchmarks.memory_runtime.event_native_controls import (
    FINITE_VIEW_MODES,
    ONE_PASS_VIEW_MODES,
    EventNativeOnePassController,
    build_event_native_controller,
    describe_event_native_route,
)
from benchmarks.memory_runtime.event_native_exact_policy import (
    EventNativeExactController,
)
from benchmarks.memory_runtime.event_native_policy import EventNativeController
from benchmarks.memory_runtime.event_native_raw import build_raw_control
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_event_native_exact import (
    MAX_NEW_TOKENS,
    RATIO,
    _activated_settings,
    _draft_calls,
    _messages,
    _payload,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import (
    Tokenizer,
    contracts,
    recovery_sequence,
)
from history_memory.events import EventStore
from history_memory.packing import PackingBudgetError


def _factory(view_mode: str, *, packing=None, policy=None, model_context=None):
    default_packing, default_policy = contracts()
    return build_event_native_controller(
        Tokenizer(),
        packing=packing or default_packing,
        policy=policy or default_policy,
        view_mode=view_mode,
        model_context=model_context,
    )


def test_route_descriptions_keep_training_static_distinct_from_legacy() -> None:
    assert FINITE_VIEW_MODES == {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "full_original",
        "static",
    }
    assert ONE_PASS_VIEW_MODES == {
        "capacity_protect",
        "full_original",
        "static",
    }
    static = describe_event_native_route("static")
    assert static == {
        "view_mode": "static",
        "baseline_identity": "event-native-training-static",
        "recovery_enabled": False,
        "max_generations_per_decision": 1,
        "legacy_1088_equivalent": False,
    }
    assert describe_event_native_route("capacity_exact_once")[
        "baseline_identity"
    ] == "C2KV-recover-once"
    with pytest.raises(ValueError, match="view_mode"):
        describe_event_native_route("legacy")


@pytest.mark.parametrize(
    "view_mode",
    sorted(FINITE_VIEW_MODES - {"full_original", "static"}),
)
def test_factory_uses_exact_controller_for_budgeted_and_shared_routes(
    view_mode: str,
) -> None:
    assert isinstance(_factory(view_mode), EventNativeExactController)


@pytest.mark.parametrize("view_mode", ["full_original", "static"])
def test_factory_uses_one_pass_adapter_for_full_and_static(view_mode: str) -> None:
    assert isinstance(_factory(view_mode), EventNativeOnePassController)


def test_capacity_protect_is_exactly_once_first_draft_without_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packing, policy = _activated_settings(lease_decisions=3)
    protect = _factory("capacity_protect", packing=packing, policy=policy)
    once = _factory("capacity_exact_once", packing=packing, policy=policy)

    protected = protect.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    recovered = once.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    assert protected.memory == recovered.memory
    for key in (
        "protected_event_ids",
        "selected_event_ids",
        "retrieved_event_ids",
        "retained_event_ids",
        "expired_lease_event_ids",
        "revision_cancelled_event_ids",
    ):
        assert protected.metadata[key] == recovered.metadata[key]
    assert protected.metadata["retrieved_event_ids"] == []
    assert protected.metadata["retained_event_ids"] == []
    assert protected.metadata["pre_draft_retrieval"] is False

    def fail_detector(*args, **kwargs):
        raise AssertionError("capacity_protect must not invoke the detector")

    monkeypatch.setattr(exact_policy_module, "detect_exact_source_gap", fail_detector)
    result = protect.reconsider(
        protected, _draft_calls(), draft_text="unsubmitted native draft"
    )
    assert result["regenerate"] is False
    assert result["memory"] == protected.memory
    assert result["decision"] == {
        "version": exact_policy_module.EVENT_NATIVE_EXACT_VERSION,
        "status": "no_op",
        "reason": "recovery_disabled",
        "gap_type": None,
        "candidate_event_id": None,
        "bindings": [],
        "judges_action_correctness": False,
        "decision_index": 1,
        "upgrade_count": 0,
        "regeneration_allowed": False,
        "upgraded_event_id": None,
    }
    assert result["metadata"]["post_draft_exact_recovery_applied"] is False


def test_capacity_protect_never_retains_recovery_evidence_across_decisions() -> None:
    packing, policy = _activated_settings(lease_decisions=3)
    protect = _factory("capacity_protect", packing=packing, policy=policy)
    once = _factory("capacity_exact_once", packing=packing, policy=policy)
    first_protect = protect.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    first_once = once.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    protect.reconsider(first_protect, _draft_calls(), draft_text="draft")
    once.reconsider(first_once, [], draft_text="draft without exact binding")

    later_messages = _messages() + [
        {"role": "assistant", "content": "First decision completed."},
        {"role": "user", "content": "Continue."},
    ]
    later_payload = _payload(decision_key="d2", messages=later_messages)
    later_protect = protect.prepare(
        later_payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    later_once = once.prepare(
        later_payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    assert later_protect.memory == later_once.memory
    assert later_protect.metadata["retrieved_event_ids"] == []
    assert later_protect.metadata["retained_event_ids"] == []


def test_full_original_adapter_preserves_raw_builder_and_decision_clock() -> None:
    packing, policy = contracts()
    controller = _factory("full_original", packing=packing, policy=policy)
    requests = recovery_sequence()[:2]
    first = controller.prepare(requests[0], ratio=4, max_new_tokens=2)
    expected = build_raw_control(
        EventStore.from_messages(requests[0]["session_id"], requests[0]["messages"]),
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="full_original",
        max_new_tokens=2,
        tools=requests[0]["tools"],
    )
    assert first.memory == expected.memory
    assert first.metadata["decision_index"] == 1
    assert first.metadata["route"]["baseline_identity"] == "Full-original"

    second = controller.prepare(requests[1], ratio=4, max_new_tokens=2)
    assert second.metadata["decision_index"] == 2
    assert controller.prepare(requests[1], ratio=4, max_new_tokens=2) is second


def test_static_adapter_preserves_training_static_packing_with_recovery_disabled() -> None:
    packing, policy = contracts()
    request = recovery_sequence()[-1]
    expected = EventNativeController(
        Tokenizer(), packing=packing, policy=policy, view_mode="static"
    ).prepare(request, ratio=4, max_new_tokens=2)
    controller = _factory("static", packing=packing, policy=policy)
    prepared = controller.prepare(request, ratio=4, max_new_tokens=2)

    assert prepared.memory == expected.memory
    assert prepared.metadata["route"]["baseline_identity"] == (
        "event-native-training-static"
    )
    assert prepared.metadata["route"]["legacy_1088_equivalent"] is False
    result = controller.reconsider(
        prepared,
        [{"function": {"name": "anything", "arguments": "not json"}}],
        draft_text="draft",
        parse_error="malformed draft",
    )
    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "recovery_disabled"
    assert result["decision"]["judges_action_correctness"] is False


def test_one_pass_adapter_enforces_owner_staleness_and_idempotence() -> None:
    first_controller = _factory("static")
    second_controller = _factory("static")
    requests = recovery_sequence()[:2]
    first = first_controller.prepare(requests[0], ratio=4, max_new_tokens=2)

    with pytest.raises(PolicyInputError, match="another one-pass controller"):
        second_controller.reconsider(first, [], draft_text="draft")

    result = first_controller.reconsider(first, [], draft_text="draft")
    assert first_controller.reconsider(first, [], draft_text="draft") == result
    with pytest.raises(PolicyInputError, match="second different draft"):
        first_controller.reconsider(first, [], draft_text="different")

    next_prepared = first_controller.prepare(
        requests[1], ratio=4, max_new_tokens=2
    )
    assert next_prepared.metadata["decision_index"] == 2
    stale_controller = _factory("static")
    stale = stale_controller.prepare(requests[0], ratio=4, max_new_tokens=2)
    stale_controller.prepare(requests[1], ratio=4, max_new_tokens=2)
    with pytest.raises(PolicyInputError, match="stale"):
        stale_controller.reconsider(stale, [], draft_text="draft")


def test_one_pass_adapter_rejects_rewritten_prefix_tools_and_reused_key() -> None:
    requests = recovery_sequence()[:2]

    rewritten = _factory("full_original")
    rewritten.prepare(requests[1], ratio=4, max_new_tokens=2)
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        rewritten.prepare(requests[0], ratio=4, max_new_tokens=2)

    changed_tools = _factory("full_original")
    changed_tools.prepare(requests[0], ratio=4, max_new_tokens=2)
    tool_request = copy.deepcopy(requests[1])
    tool_request["tools"] = [{"type": "function", "function": {"name": "x"}}]
    with pytest.raises(PolicyInputError, match="Tools changed"):
        changed_tools.prepare(tool_request, ratio=4, max_new_tokens=2)

    reused = _factory("static")
    reused.prepare(requests[0], ratio=4, max_new_tokens=2)
    reused_key = copy.deepcopy(requests[1])
    reused_key["decision_key"] = requests[0]["decision_key"]
    with pytest.raises(PolicyInputError, match="reused with different input"):
        reused.prepare(reused_key, ratio=4, max_new_tokens=2)


def test_full_original_adapter_applies_model_context_as_a_hard_minimum() -> None:
    packing, policy = contracts()
    request = recovery_sequence()[-1]
    unconstrained = _factory(
        "full_original", packing=packing, policy=policy
    ).prepare(request, ratio=4, max_new_tokens=2)
    required = max(
        unconstrained.memory.workspace_position_start
        + len(unconstrained.memory.workspace_input_ids)
        + 2,
        unconstrained.memory.costs(4)["resident_kv_tokens"] + 2,
    )
    assert required > 1
    with pytest.raises(PackingBudgetError, match="sequence|context|positions"):
        _factory(
            "full_original",
            packing=packing,
            policy=policy,
            model_context=required - 1,
        ).prepare(request, ratio=4, max_new_tokens=2)


def test_factory_rejects_nonfinite_routes() -> None:
    with pytest.raises(ValueError, match="view_mode"):
        _factory("legacy")
    with pytest.raises(ValueError, match="view_mode"):
        _factory("policy")
