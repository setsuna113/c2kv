"""Bounded, request-local capacity fallback for the frozen native S0 allocator."""

from __future__ import annotations

import contextlib
import copy
import hashlib
from contextvars import ContextVar
from typing import Any, Mapping

from history_memory.events import EventStore
from history_memory.packing import visible_message

from ..always_compress import ALWAYS_COMPRESSION_POLICY, CapacityInfeasible
from ..event_native_always import NATIVE_S0_MODE


POLICY_VERSION = "c2kv-s0-capacity-fallback-v1"
_MAX_PROJECTION_CANDIDATES = 12
_MAX_PROJECTION_ATTEMPTS = 2 * _MAX_PROJECTION_CANDIDATES - 1
_CONTEXT_ATTRIBUTE = "_capacity_fallback_measurement_context"


class CapacityFallbackAllocator:
    """Run the exact incumbent controller before bounded capacity fallbacks.

    The wrapped controller owns all session and reconsideration state.  A
    ContextVar changes measurement only while one fallback preparation or a
    repack of that prepared decision is active.
    """

    _OWN_ATTRIBUTES = frozenset({"_base", "_measurement_context"})

    def __init__(self, base: Any) -> None:
        if not callable(getattr(base, "prepare", None)) or not callable(
            getattr(base, "reconsider", None)
        ):
            raise TypeError("capacity fallback requires a native S0 controller")
        object.__setattr__(self, "_base", base)
        context = ContextVar(
            f"c2kv_capacity_fallback_{id(self)}", default=None
        )
        object.__setattr__(self, "_measurement_context", context)
        setattr(base, _CONTEXT_ATTRIBUTE, context.get)

    @property
    def base(self) -> Any:
        return self._base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._OWN_ATTRIBUTES or "_base" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ):
        initial_error = None
        try:
            # This call is deliberately unmodified.  Successful incumbent
            # decisions retain object identity, memory, metadata, and behavior.
            return self._base.prepare(
                payload, ratio=ratio, max_new_tokens=max_new_tokens
            )
        except CapacityInfeasible as error:
            initial_error = error
            if ratio != 8:
                raise

        assert initial_error is not None

        attempts: list[dict[str, Any]] = []
        prepared = self._attempt(
            payload,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            remove_duplicate_gist=False,
            projections=(),
            attempts=attempts,
            stage="requested_ratio_only",
        )
        if prepared is not None:
            return self._annotate(
                prepared,
                initial_error,
                attempts,
                stage="requested_ratio_only",
                remove_duplicate_gist=False,
                projections=(),
            )

        prepared = self._attempt(
            payload,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            remove_duplicate_gist=True,
            projections=(),
            attempts=attempts,
            stage="remove_redundant_gist",
        )
        if prepared is not None:
            return self._annotate(
                prepared,
                initial_error,
                attempts,
                stage="remove_redundant_gist",
                remove_duplicate_gist=True,
                projections=(),
            )

        measured_sets = [
            set(row["measured_raw_source_intersection"])
            for row in attempts
            if row.get("measured_raw_source_intersection")
        ]
        measured_raw = (
            set.intersection(*measured_sets) if measured_sets else set()
        )
        candidates = self._projection_candidates(
            payload, allowed_source_indices=measured_raw
        )
        for projections in _bounded_projection_sets(candidates):
            prepared = self._attempt(
                payload,
                ratio=ratio,
                max_new_tokens=max_new_tokens,
                remove_duplicate_gist=True,
                projections=projections,
                attempts=attempts,
                stage="omit_complete_tool_event_narrative",
            )
            if prepared is not None:
                active = tuple(
                    row for row in projections
                    if row["source_index"] in set(prepared.memory.raw_source_indices)
                )
                return self._annotate(
                    prepared,
                    initial_error,
                    attempts,
                    stage="omit_complete_tool_event_narrative",
                    remove_duplicate_gist=True,
                    projections=active,
                )

        return self._terminal_failure(
            payload, ratio=ratio, max_new_tokens=max_new_tokens,
            initial_error=initial_error, attempts=attempts,
            measured_raw=measured_raw, narrative_candidates=candidates,
        )

    def _terminal_failure(self, payload, *, ratio, max_new_tokens,
                          initial_error, attempts, measured_raw,
                          narrative_candidates):
        # The incumbent has no further fallback. Optional terminal policies
        # override this hook without changing any successful preparation.
        raise initial_error

    def reconsider(self, *args: Any, **kwargs: Any):
        return self._base.reconsider(*args, **kwargs)

    @contextlib.contextmanager
    def measurement_context(self, prepared):
        """Reapply one prepared fallback's renderer and ratio during repack."""

        receipt = getattr(prepared, "metadata", {}).get("capacity_fallback")
        if not isinstance(receipt, Mapping) or not receipt.get("applied"):
            yield
            return
        store = getattr(prepared, "_store", None)
        if not isinstance(store, EventStore):
            raise TypeError("fallback repack requires the original EventStore")
        source_indices = tuple(
            row["source_index"]
            for row in receipt.get("narrative_projections", ())
        )
        projections = self._projection_rows(store, source_indices)
        collector: dict[int, tuple[str, ...]] = {}
        value = self._context_value(
            ratio=int(receipt["requested_ratio"]),
            remove_duplicate_gist=bool(receipt["remove_duplicate_gist"]),
            projections=projections,
            collector=collector,
        )
        token = self._measurement_context.set(value)
        try:
            yield
        finally:
            self._measurement_context.reset(token)

    def _attempt(
        self,
        payload,
        *,
        ratio,
        max_new_tokens,
        remove_duplicate_gist,
        projections,
        attempts,
        stage,
    ):
        collector: dict[int, tuple[str, ...]] = {}
        value = self._context_value(
            ratio=ratio,
            remove_duplicate_gist=remove_duplicate_gist,
            projections=projections,
            collector=collector,
        )
        token = self._measurement_context.set(value)
        try:
            prepared = self._base.prepare(
                payload, ratio=ratio, max_new_tokens=max_new_tokens
            )
        except CapacityInfeasible as error:
            raw_sets = value["measured_raw_source_sets"]
            raw_intersection = (
                set.intersection(*(set(row) for row in raw_sets))
                if raw_sets
                else set()
            )
            attempts.append({
                "stage": stage,
                "status": "infeasible",
                "projected_source_indices": [
                    row["source_index"] for row in projections
                ],
                "failure_type": type(error).__name__,
                "measured_raw_source_intersection": sorted(raw_intersection),
            })
            return None
        finally:
            self._measurement_context.reset(token)
        removed = collector.get(id(prepared.memory), ())
        attempts.append({
            "stage": stage,
            "status": "selected",
            "projected_source_indices": [
                row["source_index"] for row in projections
            ],
            "removed_duplicate_gist_event_ids": list(removed),
        })
        prepared._capacity_fallback_removed_gist_event_ids = tuple(removed)
        return prepared

    def _context_value(
        self, *, ratio, remove_duplicate_gist, projections, collector
    ) -> dict[str, Any]:
        return {
            "budget_ratios": (ratio,),
            "remove_duplicate_gist": remove_duplicate_gist,
            "raw_message_overrides": {
                row["source_index"]: copy.deepcopy(row["projected_message"])
                for row in projections
            },
            "removed_duplicate_gist_by_memory_id": collector,
            "measured_raw_source_sets": [],
        }

    def _projection_candidates(
        self, payload, *, allowed_source_indices
    ) -> tuple[dict[str, Any], ...]:
        store = EventStore.from_messages(
            payload["session_id"], payload["messages"], benchmark=self.benchmark
        )
        from ..adapter import raw_source_cutoff

        cutoff = raw_source_cutoff(
            [message.to_dict() for message in store.messages]
        )
        rows = []
        for event in store.events:
            if (
                event.kind != "tool_event"
                or not event.complete
                or not event.tool_call_ids
            ):
                continue
            assistant_index = event.source_indices[0]
            if assistant_index >= cutoff:
                continue
            if assistant_index not in allowed_source_indices:
                continue
            source = store.messages[assistant_index].to_dict()
            content = source.get("content")
            if (
                source.get("role") != "assistant"
                or not isinstance(content, str)
                or not content
                or not source.get("tool_calls")
            ):
                continue
            projected = copy.deepcopy(source)
            projected["content"] = None
            before = self._base._count([visible_message(source)], ())
            after = self._base._count([visible_message(projected)], ())
            removed_tokens = before - after
            if removed_tokens <= 0:
                continue
            result_indices = tuple(event.source_indices[1:])
            result_ids = tuple(
                store.messages[index].to_dict().get("tool_call_id")
                for index in result_indices
            )
            if (
                len(result_indices) != len(event.tool_call_ids)
                or set(result_ids) != set(event.tool_call_ids)
            ):
                continue
            rows.append({
                "event_id": event.event_id,
                "source_index": assistant_index,
                "result_source_indices": result_indices,
                "call_ids": tuple(event.tool_call_ids),
                "projected_message": projected,
                "removed_content_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "removed_content_utf8_bytes": len(content.encode("utf-8")),
                "standalone_rendered_token_delta": removed_tokens,
            })
        # Keep the bounded pool most likely to recover capacity.  Search cost
        # below still prefers fewer omissions and then the least removed text.
        rows.sort(
            key=lambda row: (
                -row["standalone_rendered_token_delta"], row["source_index"]
            )
        )
        selected = rows[:_MAX_PROJECTION_CANDIDATES]
        selected.sort(key=lambda row: row["source_index"])
        return tuple(selected)

    @staticmethod
    def _projection_rows(
        store: EventStore, source_indices: tuple[int, ...]
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        for source_index in source_indices:
            source = store.messages[source_index].to_dict()
            projected = copy.deepcopy(source)
            projected["content"] = None
            rows.append({
                "source_index": source_index,
                "projected_message": projected,
            })
        return tuple(rows)

    def _annotate(
        self,
        prepared,
        initial_error,
        attempts,
        *,
        stage,
        remove_duplicate_gist,
        projections,
    ):
        removed_gist = tuple(
            getattr(prepared, "_capacity_fallback_removed_gist_event_ids", ())
        )
        if hasattr(prepared, "_capacity_fallback_removed_gist_event_ids"):
            del prepared._capacity_fallback_removed_gist_event_ids
        projection_receipts = [{
            "event_id": row["event_id"],
            "source_index": row["source_index"],
            "result_source_indices": list(row["result_source_indices"]),
            "call_ids": list(row["call_ids"]),
            "removed_content_sha256": row["removed_content_sha256"],
            "removed_content_utf8_bytes": row["removed_content_utf8_bytes"],
            "standalone_rendered_token_delta": row[
                "standalone_rendered_token_delta"
            ],
            "eligibility": (
                "historical complete native tool-call event with one result "
                "bound to every immutable call ID"
            ),
        } for row in projections]
        prepared.metadata["capacity_fallback"] = {
            "version": POLICY_VERSION,
            "applied": True,
            "stage": stage,
            "trigger": "initial_capacity_infeasible",
            "initial_failure_type": type(initial_error).__name__,
            "requested_ratio": prepared.metadata["requested_ratio"],
            "configured_ratios": list(self.packing.ratios),
            "measurement_ratios": [prepared.metadata["requested_ratio"]],
            "remove_duplicate_gist": remove_duplicate_gist,
            "removed_duplicate_gist_event_ids": list(removed_gist),
            "narrative_projections": projection_receipts,
            "projection_candidate_limit": _MAX_PROJECTION_CANDIDATES,
            "projection_attempt_limit": _MAX_PROJECTION_ATTEMPTS,
            "projection_search": (
                "all_singletons_by_ascending_removed_tokens_then_"
                "cumulative_largest_savings_prefixes"
            ),
            "projection_search_globally_optimal": False,
            "attempts": copy.deepcopy(attempts),
            "source_archive_unchanged": True,
            "encoder_gist_inputs_unchanged": True,
            "tool_calls_arguments_results_ids_order_unchanged": True,
            "current_user_text_unchanged": True,
            "field_or_token_truncation_used": False,
            "new_model_calls": 0,
            "common_raw_prompt_tokens": prepared.metadata[
                "common_raw_prompt_tokens"
            ],
            "rendered_raw_prompt_tokens": prepared.metadata["raw_prompt_tokens"],
            "actual_history_bytes": prepared.metadata["actual_history_bytes"],
        }
        return prepared


def _bounded_projection_sets(candidates):
    """Try minimal single omissions, then a capacity-effective bounded path."""

    if not candidates:
        return
    ascending = sorted(
        candidates,
        key=lambda row: (
            row["standalone_rendered_token_delta"], row["source_index"]
        ),
    )
    for row in ascending:
        yield (row,)

    descending = tuple(reversed(ascending))
    for size in range(2, len(descending) + 1):
        yield descending[:size]


def build_capacity_fallback_allocator(
    tokenizer,
    *,
    packing,
    policy,
    model_context,
    s0_config,
    benchmark,
    terminal_tool_rescue=False,
):
    """Build the actual configured S0 controller, then add the fallback."""

    from ..event_native_controls import build_event_native_controller

    base = build_event_native_controller(
        tokenizer,
        packing=packing,
        policy=policy,
        view_mode=NATIVE_S0_MODE,
        model_context=model_context,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=s0_config,
        benchmark=benchmark,
    )
    if terminal_tool_rescue:
        from .tool_event_rescue import ToolEventRescueAllocator

        return ToolEventRescueAllocator(base)
    return CapacityFallbackAllocator(base)


__all__ = [
    "POLICY_VERSION",
    "CapacityFallbackAllocator",
    "build_capacity_fallback_allocator",
]
