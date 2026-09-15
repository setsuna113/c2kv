"""Pure-tokenizer contracts for the event-native raw representation controls."""

import copy
import json

import pytest

from benchmarks.memory_runtime.event_native import (
    memory_from_dict,
    memory_to_dict,
    prepare_request_sequence,
)
from benchmarks.memory_runtime.event_native_policy import EventNativeController
from benchmarks.memory_runtime.event_native_raw import (
    RuntimeMemoryView,
    build_raw_control,
    render_raw_control_messages,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import (
    Tokenizer,
    contracts,
    recovery_sequence,
)
from history_memory.evidence import evidence_message
from history_memory.events import EventStore
from history_memory.packing import MemoryView, native_ids, raw_workspace_messages, select_view, visible_message


def runtime_contracts(*, packing_overrides=None, policy_overrides=None):
    packing, policy = contracts()
    # Exercise the production workspace cap while allowing tests to shrink a
    # particular boundary from measured fixture token counts.
    packing["max_workspace_tokens"] = 4096
    packing.update(packing_overrides or {})
    policy.update(policy_overrides or {})
    return packing, policy


def store_for(request):
    return EventStore.from_messages(request["session_id"], request["messages"])


def visible_indices(store, event_ids):
    selected = set(event_ids)
    return tuple(
        sorted(
            index
            for event in store.events
            if event.event_id in selected
            for index in event.source_indices
        )
    )


def visible_messages(store, event_ids):
    return tuple(visible_message(store.messages[index]) for index in visible_indices(store, event_ids))


def baseline_messages(store):
    indices = {
        index
        for event in store.events
        if event.kind == "instruction"
        for index in event.source_indices
    }
    users = [event for event in store.events if event.kind == "user"]
    if users:
        indices.add(max(users[-1].source_indices))
    if store.messages:
        indices.add(len(store.messages) - 1)
    return tuple(visible_message(store.messages[index]) for index in sorted(indices))


def token_count(tokenizer, messages, *, tools=()):
    return len(native_ids(tokenizer, messages, tools=tools, generation=True))


def history_bytes(store, tokenizer, rendered_messages, policy, *, tools=()):
    raw_tokens = token_count(tokenizer, rendered_messages, tools=tools)
    baseline_tokens = token_count(tokenizer, baseline_messages(store), tools=tools)
    return max(0, raw_tokens - baseline_tokens) * policy["kv_bytes_per_token"]


def profile(packing, policy):
    return {
        "packing_contract": packing,
        "policy_contract": policy,
        "declared_supported_ratios": list(packing["ratios"]),
        "model_geometry": {
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": packing["max_sequence_tokens"],
        },
    }


def test_no_gist_partitions_known_events_and_never_omits_mandatory_events():
    request = copy.deepcopy(recovery_sequence()[-1])
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "pending",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"later.txt"}'},
                    }
                ],
            },
            {"role": "user", "content": "Continue while that call is pending."},
        ]
    )
    packing, policy = runtime_contracts(packing_overrides={"recent_tool_events": 1})
    store = store_for(request)
    mandatory = select_view(store, recent_tool_events=1)
    mandatory_render = raw_workspace_messages(store, mandatory)
    policy["history_budget_bytes"] = history_bytes(store, Tokenizer(), mandatory_render, policy)

    prepared = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        max_new_tokens=2,
    )
    view = prepared.memory.view
    known = {event.event_id for event in store.events}

    assert isinstance(view, RuntimeMemoryView)
    assert prepared.memory.chunks == ()
    assert view.gist_event_ids == ()
    assert set(view.raw_event_ids) | set(view.omitted_event_ids) == known
    assert set(view.raw_event_ids).isdisjoint(view.omitted_event_ids)
    assert set(view.mandatory_raw_event_ids) == set(mandatory.raw_event_ids)
    assert set(view.mandatory_raw_event_ids) <= set(view.raw_event_ids)
    assert view.omitted_event_ids
    assert all(store.event(event_id).complete for event_id in view.omitted_event_ids)
    assert all(store.event(event_id).kind != "instruction" for event_id in view.omitted_event_ids)
    assert set(view.omitted_event_ids).isdisjoint(view.mandatory_raw_event_ids)


def test_no_gist_keeps_shared_evidence_once_and_out_of_the_native_raw_host():
    request = recovery_sequence()[-1]
    store = store_for(request)
    evidence_id = f"{request['session_id']}:m2"
    packing, policy = runtime_contracts()

    prepared = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        evidence_event_ids=(evidence_id,),
        max_new_tokens=2,
    )
    rendered = render_raw_control_messages(store, prepared.memory.view)

    assert prepared.memory.view.evidence_event_ids == (evidence_id,)
    assert evidence_id in prepared.memory.view.raw_event_ids
    assert sum(message == evidence_message(store, (evidence_id,)) for message in rendered) == 1
    assert all(message.get("tool_call_id") != "alpha" for message in rendered)
    assert all(
        call.get("id") != "alpha"
        for message in rendered
        for call in message.get("tool_calls", ())
    )


def test_no_gist_history_budget_admits_exact_boundary_and_denies_one_byte_below():
    request = recovery_sequence()[1]
    store = store_for(request)
    packing, policy = runtime_contracts(packing_overrides={"recent_tool_events": 1})
    first = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        max_new_tokens=2,
    )
    boundary = history_bytes(
        store,
        Tokenizer(),
        render_raw_control_messages(store, first.memory.view),
        policy,
    )
    assert boundary > 0

    exact_policy = {**policy, "history_budget_bytes": boundary}
    exact = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=exact_policy,
        mode="no_gist",
        max_new_tokens=2,
    )
    assert exact.memory == first.memory

    with pytest.raises(ValueError, match="history"):
        build_raw_control(
            store,
            Tokenizer(),
            packing=packing,
            policy={**policy, "history_budget_bytes": boundary - 1},
            mode="no_gist",
            max_new_tokens=2,
        )


def test_no_gist_refill_is_newest_first_but_skips_an_oversize_event():
    request = {
        "session_id": "raw/refill",
        "messages": [
            {"role": "system", "content": "Keep useful history."},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "newer small user"},
            {"role": "assistant", "content": "X" * 2000},
            {"role": "user", "content": "current"},
        ],
    }
    store = store_for(request)
    packing, policy = runtime_contracts()
    mandatory = select_view(store, recent_tool_events=0)
    small_id = "raw/refill:m3"
    oversize_id = "raw/refill:m4"
    selected_ids = tuple(
        event.event_id
        for event in store.events
        if event.event_id in set(mandatory.raw_event_ids) | {small_id}
    )
    selected_messages = visible_messages(store, selected_ids)
    policy["history_budget_bytes"] = history_bytes(store, Tokenizer(), selected_messages, policy)
    assert history_bytes(
        store,
        Tokenizer(),
        visible_messages(store, set(mandatory.raw_event_ids) | {oversize_id}),
        policy,
    ) > policy["history_budget_bytes"]

    prepared = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        max_new_tokens=2,
    )

    assert small_id in prepared.memory.view.raw_event_ids
    assert oversize_id in prepared.memory.view.omitted_event_ids


def test_no_gist_workspace_cost_uses_the_same_final_native_raw_host():
    request = recovery_sequence()[-1]
    store = store_for(request)
    evidence_id = f"{request['session_id']}:m2"
    packing, policy = runtime_contracts()
    first = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        evidence_event_ids=(evidence_id,),
        max_new_tokens=2,
    )
    view = first.memory.view
    host_event_ids = set(view.raw_event_ids) - set(view.evidence_event_ids)
    with_evidence = render_raw_control_messages(store, view)
    without_evidence = visible_messages(store, host_event_ids)
    evidence_bytes = (
        token_count(Tokenizer(), with_evidence) - token_count(Tokenizer(), without_evidence)
    ) * policy["kv_bytes_per_token"]
    assert evidence_bytes > 0
    assert first.metadata["evidence_bytes"] == evidence_bytes

    exact = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy={**policy, "workspace_budget_bytes": evidence_bytes},
        mode="no_gist",
        evidence_event_ids=(evidence_id,),
        max_new_tokens=2,
    )
    assert exact.memory == first.memory
    with pytest.raises(ValueError, match="workspace"):
        build_raw_control(
            store,
            Tokenizer(),
            packing=packing,
            policy={**policy, "workspace_budget_bytes": evidence_bytes - 1},
            mode="no_gist",
            evidence_event_ids=(evidence_id,),
            max_new_tokens=2,
        )


def test_no_gist_refill_may_exceed_training_workspace_after_mandatory_base_fits():
    request = recovery_sequence()[-1]
    store = store_for(request)
    packing, policy = runtime_contracts(policy_overrides={"history_budget_bytes": 0})
    mandatory = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="no_gist",
        max_new_tokens=2,
    )
    mandatory_limit = len(mandatory.memory.workspace_input_ids)
    assert mandatory_limit > 0

    refill_packing = {**packing, "max_workspace_tokens": mandatory_limit}
    refill = build_raw_control(
        store,
        Tokenizer(),
        packing=refill_packing,
        policy={**policy, "history_budget_bytes": 1_000_000},
        mode="no_gist",
        max_new_tokens=2,
    )
    assert len(refill.memory.workspace_input_ids) > mandatory_limit
    assert set(mandatory.memory.view.raw_event_ids) <= set(refill.memory.view.raw_event_ids)


def test_full_original_ignores_history_and_training_workspace_but_obeys_total_context():
    request = recovery_sequence()[-1]
    store = store_for(request)
    packing, policy = runtime_contracts(
        packing_overrides={"max_workspace_tokens": 1},
        policy_overrides={"history_budget_bytes": 0},
    )
    first = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="full_original",
        max_new_tokens=2,
    )
    resident = first.memory.costs(4)["resident_kv_tokens"]
    known = tuple(event.event_id for event in store.events)
    assert first.memory.view.raw_event_ids == known
    assert first.memory.view.omitted_event_ids == ()
    assert first.memory.chunks == ()
    assert len(first.memory.workspace_input_ids) > packing["max_workspace_tokens"]

    exact_packing = {**packing, "max_sequence_tokens": resident + 2}
    build_raw_control(
        store,
        Tokenizer(),
        packing=exact_packing,
        policy=policy,
        mode="full_original",
        max_new_tokens=2,
    )
    with pytest.raises(ValueError, match="sequence|context"):
        build_raw_control(
            store,
            Tokenizer(),
            packing={**packing, "max_sequence_tokens": resident + 1},
            policy=policy,
            mode="full_original",
            max_new_tokens=2,
        )


def test_full_shared_preserves_full_native_messages_and_charges_only_added_packet():
    request = recovery_sequence()[-1]
    store = store_for(request)
    evidence_id = f"{request['session_id']}:m2"
    packing, policy = runtime_contracts(policy_overrides={"history_budget_bytes": 0})
    original = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="full_original",
        max_new_tokens=2,
    )
    shared = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="full_shared",
        evidence_event_ids=(evidence_id,),
        max_new_tokens=2,
    )
    original_messages = render_raw_control_messages(store, original.memory.view)
    shared_messages = list(render_raw_control_messages(store, shared.memory.view))
    packet = evidence_message(store, (evidence_id,))
    assert packet is not None and shared_messages.count(packet) == 1
    shared_messages.remove(packet)

    assert tuple(shared_messages) == original_messages
    assert native_ids(Tokenizer(), shared_messages, generation=True) == (
        original.memory.system_input_ids + original.memory.workspace_input_ids
    )
    assert shared.memory.raw_source_indices == tuple(range(len(store.messages)))
    evidence_bytes = (
        token_count(Tokenizer(), render_raw_control_messages(store, shared.memory.view))
        - token_count(Tokenizer(), original_messages)
    ) * policy["kv_bytes_per_token"]
    assert evidence_bytes > 0
    assert shared.metadata["evidence_bytes"] == evidence_bytes

    exact = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy={**policy, "workspace_budget_bytes": evidence_bytes},
        mode="full_shared",
        evidence_event_ids=(evidence_id,),
        max_new_tokens=2,
    )
    assert exact.memory == shared.memory
    with pytest.raises(ValueError, match="workspace"):
        build_raw_control(
            store,
            Tokenizer(),
            packing=packing,
            policy={**policy, "workspace_budget_bytes": evidence_bytes - 1},
            mode="full_shared",
            evidence_event_ids=(evidence_id,),
            max_new_tokens=2,
        )


def test_raw_modes_reuse_the_policy_evidence_lifecycle_without_private_selection():
    packing, policy = runtime_contracts()
    payload = {"schema": "a-event-native-request-sequence-v1", "requests": recovery_sequence()}
    expected = prepare_request_sequence(
        profile(packing, policy),
        Tokenizer(),
        payload,
        view_mode="policy",
        ratio=4,
        max_new_tokens=2,
    )
    no_gist = prepare_request_sequence(
        profile(packing, policy),
        Tokenizer(),
        payload,
        view_mode="no_gist",
        ratio=4,
        max_new_tokens=2,
    )
    full_shared = prepare_request_sequence(
        profile(packing, policy),
        Tokenizer(),
        payload,
        view_mode="full_shared",
        ratio=4,
        max_new_tokens=2,
    )
    for policy_item, no_gist_item, shared_item in zip(expected, no_gist, full_shared):
        evidence = policy_item.memory.view.evidence_event_ids
        assert no_gist_item.memory.view.evidence_event_ids == evidence
        assert shared_item.memory.view.evidence_event_ids == evidence
        assert no_gist_item.metadata["shared_controller"] == policy_item.metadata
        assert shared_item.metadata["shared_controller"] == policy_item.metadata
        assert "selection" not in no_gist_item.metadata
        assert "selection" not in shared_item.metadata

    zero_policy = {**policy, "history_budget_bytes": 0}
    original = prepare_request_sequence(
        profile(packing, zero_policy),
        Tokenizer(),
        payload,
        view_mode="full_original",
        ratio=4,
        max_new_tokens=2,
    )
    assert len(original) == len(payload["requests"])


def test_packed_input_roundtrip_preserves_both_base_and_runtime_view_schemas():
    request = recovery_sequence()[-1]
    store = store_for(request)
    packing, policy = runtime_contracts()
    base = EventNativeController(
        Tokenizer(), packing=packing, policy=policy, view_mode="static"
    ).prepare(request, ratio=4, max_new_tokens=2).memory
    runtime = build_raw_control(
        store,
        Tokenizer(),
        packing=packing,
        policy={**policy, "history_budget_bytes": 0},
        mode="no_gist",
        max_new_tokens=2,
    ).memory
    assert runtime.view.omitted_event_ids

    base_payload = memory_to_dict(base)
    runtime_payload = memory_to_dict(runtime)
    assert base_payload["schema"] == "a-event-native-packed-input-v1"
    assert set(base_payload["view"]) == {
        "gist_event_ids",
        "raw_event_ids",
        "evidence_event_ids",
    }
    assert runtime_payload["schema"] == "a-event-native-packed-input-v2"
    assert set(runtime_payload["view"]) == {
        "gist_event_ids",
        "raw_event_ids",
        "evidence_event_ids",
        "omitted_event_ids",
        "mandatory_raw_event_ids",
        "raw_control_layout",
    }
    assert memory_from_dict(json.loads(json.dumps(base_payload))) == base
    assert memory_from_dict(json.loads(json.dumps(runtime_payload))) == runtime

    invalid_v1 = copy.deepcopy(base_payload)
    invalid_v1["view"]["omitted_event_ids"] = []
    with pytest.raises(ValueError, match="v1"):
        memory_from_dict(invalid_v1)


def test_sequence_no_gist_refill_respects_actual_model_context_before_admission():
    request = recovery_sequence()[-1]
    store = store_for(request)
    packing, policy = runtime_contracts(policy_overrides={"workspace_budget_bytes": 0})
    mandatory = build_raw_control(
        store, Tokenizer(), packing=packing,
        policy={**policy, "history_budget_bytes": 0}, mode="no_gist", max_new_tokens=2)
    actual_context = mandatory.memory.costs(4)["resident_kv_tokens"] + 2
    current_profile = profile(packing, policy)
    current_profile["model_geometry"]["max_position_embeddings"] = actual_context
    prepared = prepare_request_sequence(
        current_profile, Tokenizer(),
        {"schema": "a-event-native-request-sequence-v1", "requests": [request]},
        view_mode="no_gist", ratio=4, max_new_tokens=2)[0]
    assert prepared.memory == mandatory.memory
    assert prepared.memory.view.omitted_event_ids
    assert prepared.metadata["max_sequence_tokens"] == actual_context
    assert prepared.metadata["configured_max_sequence_tokens"] == packing["max_sequence_tokens"]
