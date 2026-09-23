"""Track C2KV residency while preserving RACER's initial and recovery policies."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from history_memory.events import EventStore
from history_memory.resident_state import ResidentSourceState, fragment_key

from ..event_native_policy import PolicyInputError


@dataclass
class _Decision:
    key: str
    prepared: object
    store: EventStore
    initial_policy_route: str
    views: list = field(default_factory=list)
    committed: object = None


class C2KVResidentPolicy:
    """Make old-source admissions explicit without replacing the S0 allocator.

    Initial RACER allocation can select archived evidence, as can its existing
    post-draft recovery policy. Only those selections may re-admit evicted
    sources. Unselected archived chunks are not sent for implicit extraction.
    The bare controller uses the same ResidentSourceState with no such policy.
    """

    def __init__(self, inner, backend):
        self.inner, self.backend = inner, backend
        self._resident = {}
        self._active = {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _receipt(self, session_id, memory, stage, initial_policy_route):
        resident = self._resident.get(session_id)
        admission = (resident.racer_admission_receipt(memory) if resident else {
            "newly_admitted_old_raw_source_indices": [],
            "newly_admitted_old_fragments": [],
        })
        return {
            "schema": "c2kv-resident-admission-v1",
            "stage": stage,
            "policy": self.backend.policy,
            "backend_allocation": self.backend.allocation,
            "initial_allocation": "configured_c2kv_policy",
            "initial_policy_route": initial_policy_route,
            "previous_source_message_count": resident.source_message_count if resident else 0,
            **admission,
        }

    def prepare(self, payload, *, ratio, max_new_tokens):
        session_id, key = payload["session_id"], payload["decision_key"]
        active = self._active.get(session_id)
        if active is not None and active.key != key and active.committed is None:
            raise PolicyInputError("Previous C2KV decision has no committed final memory")
        prepared = self.inner.prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if active is not None and active.key == key:
            if active.prepared is not prepared:
                raise PolicyInputError("C2KV prepared decision changed within a decision")
            return prepared
        store = EventStore.from_messages(session_id, payload["messages"],
                                         benchmark=getattr(self.inner, "benchmark", None))
        resident = self._resident.get(session_id)
        chunks = tuple(prepared.eligible_chunks)
        if resident is not None:
            selected = {fragment_key(chunk) for chunk in prepared.memory.chunks}
            allowed = {fragment_key(chunk) for chunk in resident.admitted_fragments(chunks)}
            # Raw sources explicitly selected by S0 may also supply their gist.
            raw = set(prepared.memory.raw_source_indices)
            chunks = tuple(chunk for chunk in chunks if
                           fragment_key(chunk) in allowed | selected or
                           all(index in raw for index in chunk.source_indices))
        prepared.eligible_chunks = chunks
        prepared.metadata = copy.deepcopy(prepared.metadata)
        planned = prepared.metadata.get("eligible_extraction")
        if isinstance(planned, dict):
            # Preserve the allocator's archive plan as provenance, while the
            # execution receipt describes only the chunks actually admitted.
            prepared.metadata["archive_extraction_plan"] = copy.deepcopy(planned)
            indices = sorted({index for chunk in chunks for index in chunk.source_indices})
            groups = _execution_groups(store, planned, chunks)
            archive_eligible_sources = set(planned.get("eligible_source_indices", ()))
            planned.update(
                source="resident memory + new input + explicit RACER selection",
                eligible_event_ids=list(dict.fromkeys(
                    event_id for group in groups for event_id in group)),
                encoding_event_groups=[list(group) for group in groups],
                eligible_encoder_unit_ids=list(dict.fromkeys(chunk.event_id for chunk in chunks)),
                eligible_source_indices=sorted(archive_eligible_sources & set(indices)),
                whole_event_encoded_source_indices=indices,
                eligible_chunk_count=len(chunks),
                eligible_presented_encoder_tokens=sum(len(chunk.token_ids) for chunk in chunks),
                eligible_unique_encoder_tokens=_unique_tokens(chunks),
                backend_execution_required=bool(chunks),
            )
        route = prepared.metadata.get("route")
        initial_policy_route = (
            route.get("baseline_identity") if isinstance(route, dict) else None
        )
        if not isinstance(initial_policy_route, str) or not initial_policy_route:
            initial_policy_route = type(self.inner).__name__
        prepared.metadata["c2kv_resident_state"] = self._receipt(
            session_id, prepared.memory, "initial_racer_allocation",
            initial_policy_route)
        self._active[session_id] = _Decision(
            key, prepared, store, initial_policy_route, [prepared.memory])
        return prepared

    def _decision(self, prepared):
        for session_id, active in self._active.items():
            if active.prepared is prepared:
                return session_id, active
        raise PolicyInputError("Foreign or stale C2KV prepared decision")

    def reconsider(self, prepared, *args, **kwargs):
        session_id, active = self._decision(prepared)
        result = self.inner.reconsider(prepared, *args, **kwargs)
        active.views.append(result["memory"])
        result["metadata"] = copy.deepcopy(result["metadata"])
        result["metadata"]["c2kv_resident_state"] = self._receipt(
            session_id, result["memory"], "post_draft_recovery",
            active.initial_policy_route)
        return result

    def commit_memory(self, prepared, final_memory):
        session_id, active = self._decision(prepared)
        if not any(final_memory is view for view in active.views):
            raise PolicyInputError("Final C2KV memory was not a prepared or recovered view")
        if active.committed is not None:
            if active.committed is not final_memory:
                raise PolicyInputError("C2KV decision already committed another memory")
            return {"status": "already_committed"}
        receipt = self._receipt(
            session_id, final_memory, "final_commit", active.initial_policy_route)
        self._resident[session_id] = ResidentSourceState.from_memory(active.store, final_memory)
        active.committed = final_memory
        return {"status": "committed", **receipt}


def _unique_tokens(chunks):
    positions = {}
    for chunk in chunks:
        positions.setdefault((chunk.event_id, chunk.source_indices), set()).update(
            range(chunk.source_token_start, chunk.source_token_end))
    return sum(map(len, positions.values()))


def _execution_groups(store, planned, chunks):
    archive_groups = planned.get("encoding_event_groups")
    if not isinstance(archive_groups, list):
        archive_groups = [[event.event_id] for event in store.events]
    active_sources = [set(chunk.source_indices) for chunk in chunks]
    groups = []
    for group in archive_groups:
        group_sources = {
            index for event_id in group
            for index in store.event(event_id).source_indices
        }
        if any(sources and sources <= group_sources for sources in active_sources):
            groups.append(tuple(group))
    return tuple(groups)
