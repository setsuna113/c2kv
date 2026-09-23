"""Measured replacement of complete events within the existing B0 budget."""
from __future__ import annotations
from contextlib import nullcontext

from ..recovery.admission import metadata_after_admission


def repack(base, prepared, *, candidate=None, derived_messages=None,
           goal_view=False):
    repack_sources = getattr(base, "repack_sources", None)
    if callable(repack_sources):
        return repack_sources(
            prepared, candidate=candidate, derived_messages=derived_messages,
            goal_view=goal_view)
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
    projected = {
        row["event_id"]
        for key in ("narrative_projections", "argument_projections")
        for row in prepared.metadata.get("capacity_fallback", {}).get(key, ())
    }
    gist = [
        event_id
        for event_id in gist
        if event_id not in set(raw) or event_id in projected
    ]
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
        context = getattr(base, "measurement_context", None)
        scope = context(prepared) if callable(context) else nullcontext()
        with scope:
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
        if event_id in projected and event_id in raw:
            # Measurement requires the original gist behind projected raw.
            continue
        gist.remove(event_id)
        released.append(event_id)
        result = measure()
    final_raw = set(result.memory.view.raw_event_ids) if result is not None else set(raw)
    final_gist = set(result.memory.view.gist_event_ids) if result is not None else set(gist)
    receipt = {
        "policy": "candidate-budgeted-replacement-v1",
        "status": "admitted" if result is not None else "abstained",
        "candidate_event_id": candidate,
        "demoted_raw_event_ids": demoted, "released_gist_event_ids": released,
        "raw_gist_exclusive": not bool(final_raw & final_gist),
        "b0_rechecked_after_all_changes": True,
        "derived_workspace_messages": len(derived),
    }
    if projected:
        receipt["projected_gist_event_ids"] = [
            event_id for event_id in gist if event_id in projected
        ]
    if result is None:
        return None, None, receipt
    metadata = metadata_after_admission(base, prepared, result, candidate, receipt)
    metadata["eligible_extraction"]["retained_encoder_unit_ids"] = list(
        dict.fromkeys(chunk.event_id for chunk in result.memory.chunks))
    metadata["derived_workspace_prefix_messages"] = list(derived)
    retained_projected_gist = [
        event_id
        for event_id in result.memory.view.gist_event_ids
        if event_id in projected
    ]
    if retained_projected_gist:
        metadata["gist_reservation"] = {
            "policy": "projected-raw-source-gist-backing-v1",
            "reserved_event_id": retained_projected_gist[0],
            "reserved_event_ids": retained_projected_gist,
            "status": "retained_for_projected_raw_source_coverage",
        }
        metadata["min_gist_reservation_met"] = True
    else:
        metadata["gist_reservation"] = {
            "policy": "raw-dominant-complete-source-v1",
            "reserved_event_id": None,
            "status": "replaced_by_exact_source_coverage",
        }
        metadata["min_gist_reservation_met"] = None
    if metadata.get("capacity_fallback", {}).get("argument_projections"):
        from .tool_event_rescue import refresh_rescue_metadata

        refresh_rescue_metadata(metadata, result.memory)
    return result, metadata, receipt
