"""Validate captured request-level invariants for exact recovery modes."""
from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXACT_MODES = frozenset({
    "capacity_exact_once",
    "capacity_exact_persistent",
    "full_exact_shared",
    "capacity_exact_no_gist",
})
_EXACT_STATUSES = frozenset({"no_op", "abstain", "gap"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exact request lacks {label}")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Exact request lacks {label}")
    return value


def _full_render(source: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # These imports are intentionally lazy: ordinary modes do not load the
    # proxy, its backend registry, or the training-dialect arm definitions.
    benchmark_root = str(Path(__file__).resolve().parent.parent)
    added = benchmark_root not in sys.path
    if added:
        sys.path.insert(0, benchmark_root)
    try:
        import proxy
        from arms import get_arm
    finally:
        if added:
            sys.path.remove(benchmark_root)

    rendered, _ = proxy._assemble(copy.deepcopy(source), get_arm("full"))
    return rendered


def _event_store(task_id: str, source: list[Mapping[str, Any]]):
    # adapter owns the repository's temporary python/history_memory import
    # bridge, so validation reuses the exact EventStore class used at runtime.
    EventStore, _, _ = _adapter_api()

    return EventStore.from_messages(task_id, source)


def _adapter_api():
    """Load adapter symbols in package and direct-file collector contexts."""
    benchmark_root = str(Path(__file__).resolve().parent.parent)
    added = benchmark_root not in sys.path
    if added:
        sys.path.insert(0, benchmark_root)
    try:
        from memory_runtime.adapter import EventStore, evidence_message, raw_source_cutoff
    finally:
        if added:
            sys.path.remove(benchmark_root)
    return EventStore, evidence_message, raw_source_cutoff


def _events(store, event_ids: Any, label: str, *, complete: bool = False):
    values = _list(event_ids, label)
    _require(
        all(isinstance(event_id, str) and event_id for event_id in values),
        f"{label} must contain nonempty event IDs",
    )
    _require(len(values) == len(set(values)), f"{label} contains duplicate event IDs")
    records = []
    for event_id in values:
        try:
            event = store.event(event_id)
        except KeyError as error:
            raise ValueError(f"{label} contains an event outside the captured source") from error
        if complete and not event.complete:
            raise ValueError(f"{label} contains an incomplete source event")
        records.append(event)
    return values, records


def _gate(metadata: Mapping[str, Any], mode: str, history_budget: Any) -> bool:
    full_shared = mode == "full_exact_shared"
    gate_name = "auxiliary_gate" if full_shared else "capacity_gate"
    activation_name = "auxiliary_activated" if full_shared else "compression_activated"
    gate = _mapping(metadata.get(gate_name), f"memory_runtime.{gate_name}")
    activated = gate.get(activation_name)
    full_history_bytes = gate.get("full_history_bytes")
    _require(type(activated) is bool, f"{gate_name}.{activation_name} must be boolean")
    _require(
        type(full_history_bytes) is int and full_history_bytes >= 0,
        f"{gate_name}.full_history_bytes must be a nonnegative integer",
    )
    _require(
        type(history_budget) is int and history_budget >= 0,
        "Exact runtime history budget must be a nonnegative integer",
    )
    _require(
        activated == (full_history_bytes > history_budget),
        f"{gate_name} activation disagrees with the captured Full history size",
    )
    return activated


def _validate_metadata_identity(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    task_id: str,
) -> int:
    expected = {
        "mode": config.get("mode"),
        "run_id": config.get("run_id"),
        "task_id": task_id,
        "history_budget_bytes": config.get("history_budget_bytes"),
        "workspace_budget_bytes": config.get("workspace_budget_bytes"),
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    _require(not mismatches, f"Exact runtime identity mismatch: {mismatches}")
    policy = _mapping(metadata.get("policy"), "memory_runtime.policy")
    _require(
        policy.get("pre_draft_retrieval") is False,
        "Exact policy must not retrieve from a model draft before generation",
    )
    decision_index = policy.get("decision_index")
    _require(
        type(decision_index) is int and decision_index >= 1,
        "Exact policy decision_index must be a positive integer",
    )
    return decision_index


def _validate_no_gist(metadata: Mapping[str, Any], messages: list[Any]) -> None:
    _require(metadata.get("gist_tokens") == 0, "NoGist/Full exact view has gist tokens")
    _require(metadata.get("block_refs") == [], "NoGist/Full exact view has gist blocks")
    _require(
        all(isinstance(message, Mapping) and "c2kv_key_hash" not in message
            for message in messages),
        "NoGist/Full exact wire contains a gist carrier",
    )


def _remove_evidence(
    messages: list[Any],
    metadata: Mapping[str, Any],
    store,
) -> list[Any]:
    selected_ids, _ = _events(
        store, metadata.get("selected_event_ids"), "selected_event_ids"
    )
    _, evidence_message, _ = _adapter_api()

    packet = evidence_message(store, selected_ids)
    index = metadata.get("evidence_out_index")
    if packet is None:
        _require(index is None, "Empty exact evidence has an output index")
        return list(messages)
    _require(
        type(index) is int and 0 <= index < len(messages),
        "Exact evidence_out_index is outside the forwarded wire",
    )
    _require(messages[index] == packet, "Forwarded exact evidence packet differs from EventStore")
    return list(messages[:index]) + list(messages[index + 1 :])


def _validate_full_generation(
    metadata: Mapping[str, Any],
    messages: list[Any],
    full: list[dict[str, Any]],
    store,
    activated: bool,
) -> None:
    _validate_no_gist(metadata, messages)
    _require(metadata.get("budget_applies") is False, "Full exact applied the history budget")
    if activated:
        expected_events = [event.event_id for event in store.events]
        _require(
            metadata.get("full_raw_visible_event_ids") == expected_events,
            "Full exact does not declare every captured source event visible",
        )
        without_evidence = _remove_evidence(messages, metadata, store)
        _require(without_evidence == full, "Full exact cropped or rewrote the Full renderer")
    else:
        _require(
            metadata.get("selected_source_indices") == list(range(len(store.messages))),
            "Below-B Full exact does not declare every source message visible",
        )
        _require(metadata.get("evidence_out_index") is None, "Below-B Full exact inserted E")
        _require(messages == full, "Below-B Full exact is not Full renderer identity")


def _validate_no_gist_generation(
    metadata: Mapping[str, Any],
    messages: list[Any],
    full: list[dict[str, Any]],
    source: list[Mapping[str, Any]],
    store,
    activated: bool,
) -> None:
    _validate_no_gist(metadata, messages)
    if not activated:
        _require(
            metadata.get("selected_source_indices") == list(range(len(source))),
            "Below-B NoGist does not declare every source message visible",
        )
        _require(metadata.get("evidence_out_index") is None, "Below-B NoGist inserted E")
        _require(messages == full, "Below-B NoGist is not Full renderer identity")
        return

    _, evidence_message, raw_source_cutoff = _adapter_api()

    source_cutoff = raw_source_cutoff(source)
    _require(
        metadata.get("source_cutoff") == source_cutoff,
        "NoGist source_cutoff differs from the runtime source boundary",
    )
    raw_ids, raw_events = _events(
        store, metadata.get("raw_history_event_ids"), "raw_history_event_ids", complete=True
    )
    evidence_ids, evidence_events = _events(
        store, metadata.get("selected_event_ids"), "selected_event_ids"
    )
    _require(
        not set(raw_ids).intersection(evidence_ids),
        "NoGist raw history duplicates an exact evidence event",
    )
    _require(
        all(event.kind != "instruction" and max(event.source_indices) < source_cutoff
            for event in raw_events),
        "NoGist raw history contains a common-suffix or instruction event",
    )

    raw_sources = {index for event in raw_events for index in event.source_indices}
    evidence_sources = {index for event in evidence_events for index in event.source_indices}
    _require(
        metadata.get("raw_history_source_indices") == sorted(raw_sources),
        "NoGist raw event IDs disagree with raw_history_source_indices",
    )

    shift = int(not any(message.get("role") == "system" for message in source))
    _require(
        len(full) == len(source) + shift,
        "Full renderer is not message-preserving for the captured source",
    )
    cutoff = source_cutoff + shift
    common_positions = {
        index
        for index, message in enumerate(full)
        if index >= cutoff or message.get("role") in {"system", "developer"}
    }
    common_sources = {index - shift for index in common_positions if index >= shift}
    _require(
        metadata.get("common_source_indices") == sorted(common_sources),
        "NoGist common_source_indices differ from the Full renderer boundary",
    )
    visible_sources = raw_sources | common_sources | evidence_sources
    _require(
        metadata.get("visible_source_indices") == sorted(visible_sources),
        "NoGist visibility metadata does not follow from raw/common/E sources",
    )

    positions = common_positions | {index + shift for index in raw_sources}
    before = [full[index] for index in sorted(positions) if index < cutoff]
    suffix = [full[index] for index in sorted(positions) if index >= cutoff]
    packet = evidence_message(store, evidence_ids)
    expected_index = len(before) if packet is not None else None
    expected = before + ([packet] if packet is not None else []) + suffix
    _require(
        metadata.get("evidence_out_index") == expected_index,
        "NoGist evidence_out_index differs from the reconstructed wire",
    )
    _require(messages == expected, "NoGist forwarded wire omits or rewrites a declared source")


def _validate_upgrade(trace: list[Mapping[str, Any]], store) -> str:
    first = _mapping(trace[0].get("memory_runtime"), "generation_trace[0].memory_runtime")
    final = _mapping(trace[1].get("memory_runtime"), "generation_trace[1].memory_runtime")
    first_ids, _ = _events(store, first.get("selected_event_ids"), "draft selected_event_ids")
    final_ids, _ = _events(store, final.get("selected_event_ids"), "regeneration selected_event_ids")
    first_set, final_set = set(first_ids), set(final_ids)
    _require(first_set <= final_set, "Exact upgrade dropped first-round evidence")
    added = final_set - first_set
    _require(len(added) == 1, "Exact upgrade must add exactly one source event")
    event_id = next(iter(added))
    event = store.event(event_id)
    _require(event.complete, "Exact upgrade added an incomplete source event")
    return event_id


def validate_exact_request(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate an exact-mode proxy capture without tokenizers or model calls.

    Ordinary runtime modes are deliberately ignored. Exact modes require the
    capture fields emitted by ``--capture-request-views`` and validate both the
    request-level recovery state machine and representation-specific wire.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Runtime config must be a mapping")
    mode = config.get("mode")
    if mode not in EXACT_MODES:
        return
    if not isinstance(row, Mapping):
        raise ValueError("Exact request row must be a mapping")

    request_view = _mapping(row.get("request_view"), "request_view")
    source = _list(request_view.get("messages"), "request_view.messages")
    _require(
        all(isinstance(message, Mapping) for message in source),
        "request_view.messages must contain message objects",
    )
    context = _mapping(row.get("eval_context"), "eval_context")
    task_id = context.get("task_id")
    _require(isinstance(task_id, str) and bool(task_id), "Exact request lacks task_id")
    # Official BFCL omits run_id; RuntimeAdapter resolves that omission from
    # its frozen config. An explicitly supplied different value is invalid.
    _require(
        context.get("run_id", config.get("run_id")) == config.get("run_id"),
        "Exact request context run_id differs from config",
    )

    trace = _list(row.get("generation_trace"), "generation_trace")
    forwarded = _list(row.get("forwarded_request_views"), "forwarded_request_views")
    _require(len(trace) in {1, 2}, "Exact request must contain one or two generations")
    _require(
        len(forwarded) == len(trace),
        "Exact generation trace and forwarded request captures have different lengths",
    )

    final_metadata = _mapping(row.get("memory_runtime"), "memory_runtime")
    decision_indices = [_validate_metadata_identity(final_metadata, config, task_id)]
    generation_metadata = []
    generation_messages = []
    for index, (generation, view) in enumerate(zip(trace, forwarded)):
        generation = _mapping(generation, f"generation_trace[{index}]")
        _require(
            generation.get("phase") == ("draft" if index == 0 else "regeneration"),
            f"Exact generation {index} has the wrong phase",
        )
        _require(generation.get("status") == "completed", "Exact generation did not complete")
        _require(
            generation.get("backend_verified") is True,
            "Exact generation lacks backend verification",
        )
        _require(
            generation.get("forwarded_request_index") == index,
            "Exact generation points to the wrong forwarded request capture",
        )
        metadata = _mapping(
            generation.get("memory_runtime"), f"generation_trace[{index}].memory_runtime"
        )
        decision_indices.append(_validate_metadata_identity(metadata, config, task_id))
        generation_metadata.append(metadata)
        captured = _mapping(view, f"forwarded_request_views[{index}]")
        generation_messages.append(
            _list(captured.get("messages"), f"forwarded_request_views[{index}].messages")
        )

    _require(
        len(set(decision_indices)) == 1,
        "Exact generations do not share one policy decision_index",
    )
    exact = _mapping(final_metadata.get("exact_recovery"), "memory_runtime.exact_recovery")
    _require(
        exact.get("version") in {"exact-source-gap-v1", "exact-source-gap-v2"},
        "Exact recovery version is invalid",
    )
    status = exact.get("status")
    _require(status in _EXACT_STATUSES, "Exact recovery status is invalid")
    _require(
        exact.get("decision_index") == decision_indices[0],
        "Exact recovery and policy decision_index differ",
    )
    _require(
        exact.get("judges_action_correctness") is False,
        "Exact recovery must not claim action-correctness judging",
    )
    upgrade_count = exact.get("upgrade_count")
    regeneration_allowed = exact.get("regeneration_allowed")
    _require(upgrade_count in {0, 1} and type(upgrade_count) is int,
             "Exact recovery upgrade_count must be 0 or 1")
    _require(type(regeneration_allowed) is bool,
             "Exact recovery regeneration_allowed must be boolean")

    gap = status == "gap"
    _require(upgrade_count == int(gap), "Exact status and upgrade_count disagree")
    _require(regeneration_allowed is gap, "Exact status and regeneration permission disagree")
    _require(len(trace) == 1 + int(gap), "Exact status and generation count disagree")
    expected_discarded = [gap] + ([False] if gap else [])
    _require(
        [generation.get("discarded") for generation in trace] == expected_discarded,
        "Exact discarded flags disagree with regeneration",
    )
    if gap:
        _require(
            generation_metadata[-1].get("exact_recovery") == exact,
            "Regeneration metadata differs from final exact recovery",
        )

    store = _event_store(task_id, source)
    activations = [
        _gate(metadata, mode, config.get("history_budget_bytes"))
        for metadata in generation_metadata
    ]
    _require(len(set(activations)) == 1, "Capacity gate changed within one exact decision")
    if gap:
        _require(activations[0], "A Full-visible below-B request reported a hidden-source gap")
        upgraded_event_id = _validate_upgrade(trace, store)
        _require(
            exact.get("upgraded_event_id") == upgraded_event_id,
            "Exact recovery upgraded_event_id differs from the added source event",
        )

    if mode == "full_exact_shared":
        _require(not gap, "Full exact reported a hidden-source gap")
        full = _full_render(source)
        for metadata, messages, activated in zip(
            generation_metadata, generation_messages, activations
        ):
            _validate_full_generation(metadata, messages, full, store, activated)
    elif mode == "capacity_exact_no_gist":
        full = _full_render(source)
        for metadata, messages, activated in zip(
            generation_metadata, generation_messages, activations
        ):
            _validate_no_gist_generation(
                metadata, messages, full, source, store, activated
            )
