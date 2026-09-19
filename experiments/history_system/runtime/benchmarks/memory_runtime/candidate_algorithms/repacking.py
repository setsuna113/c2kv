"""Measured replacement of complete events within the existing B0 budget."""
from __future__ import annotations

from ..recovery.admission import metadata_after_admission


def repack(base, prepared, *, candidate=None, derived_messages=None,
           goal_view=False):
    store = prepared._store
    view = prepared.memory.view
    mandatory = set(view.mandatory_raw_event_ids)
    protected = mandatory | set(prepared.metadata.get("protected_event_ids") or ())
    raw = list(view.raw_event_ids)
    gist = list(view.gist_event_ids)
    eligible = list(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    derived = list(prepared.metadata.get("derived_workspace_prefix_messages") or ())
    for message in derived_messages or ():
        if message not in derived:
            derived.append(message)
    derived = tuple(base._merge_protected_derived(tuple(derived)))
    if candidate is not None:
        event = store.event(candidate)
        if not event.complete or candidate not in eligible:
            raise ValueError("Recovery requires one eligible complete source event")
        raw = list(dict.fromkeys([*raw, candidate]))
        protected.add(candidate)
    # A complete exact copy replaces its compressed representation. Partial
    # derived packets are deliberately not treated as full source coverage.
    gist = [event_id for event_id in gist if event_id not in set(raw)]
    demoted, released = [], []
    if goal_view:
        # Rebuild optional memory around the current observed transaction;
        # unrelated old raw copies no longer take precedence over its record.
        users = [event for event in store.events if event.kind == "user"]
        boundary = max(users[-1].source_indices) if users else -1
        old = [event_id for event_id in raw if event_id not in protected
               and max(store.event(event_id).source_indices) < boundary]
        for event_id in old:
            raw.remove(event_id)
            if event_id in eligible:
                gist.append(event_id)
            demoted.append(event_id)

    def measure():
        result = base._try_measure(
            store, prepared._tools, raw, mandatory, gist, eligible,
            prepared.metadata["common_raw_prompt_tokens"],
            prepared.metadata["max_new_tokens"], derived_messages=derived)
        return result if result is not None and not result.reasons else None

    result = measure()
    optional = sorted((event_id for event_id in raw if event_id not in protected),
                      key=lambda event_id: min(store.event(event_id).source_indices))
    for event_id in optional:
        if result is not None:
            break
        raw.remove(event_id)
        if event_id in eligible and event_id not in gist:
            gist.append(event_id)
        demoted.append(event_id)
        result = measure()
    for event_id in sorted(gist, key=lambda item: min(store.event(item).source_indices)):
        if result is not None:
            break
        gist.remove(event_id)
        released.append(event_id)
        result = measure()
    receipt = {
        "policy": "candidate-budgeted-replacement-v1",
        "status": "admitted" if result is not None else "abstained",
        "candidate_event_id": candidate,
        "demoted_raw_event_ids": demoted, "released_gist_event_ids": released,
        "raw_gist_exclusive": True, "b0_rechecked_after_all_changes": True,
        "derived_workspace_messages": len(derived),
    }
    if result is None:
        return None, None, receipt
    metadata = metadata_after_admission(base, prepared, result, candidate, receipt)
    metadata["eligible_extraction"]["retained_encoder_unit_ids"] = list(
        dict.fromkeys(chunk.event_id for chunk in result.memory.chunks))
    metadata["derived_workspace_prefix_messages"] = list(derived)
    metadata["gist_reservation"] = {
        "policy": "raw-dominant-complete-source-v1", "reserved_event_id": None,
        "status": "replaced_by_exact_source_coverage",
    }
    metadata["min_gist_reservation_met"] = None
    return result, metadata, receipt
