"""Split source protection only after the complete incumbent C1 allocator fails."""
from __future__ import annotations

from history_memory.source_packing import SourceMemoryView

from ..always_compress import CapacityInfeasible
from ..source_allocation import SOURCE_ALLOCATION_VERSION, SourceAllocatedS0Controller
from .capacity_fallback import build_capacity_fallback_allocator


POLICY_VERSION = "c2kv-capacity-gated-source-allocation-v1"


class CapacityGatedSourceAllocator:
    """Retain incumbent successes and share its lifecycle with source rescue."""

    _OWN_ATTRIBUTES = frozenset({"incumbent", "source_allocator"})

    def __init__(self, tokenizer, *, packing, policy, model_context=None,
                 s0_config=None, benchmark="bfcl"):
        arguments = dict(packing=packing, policy=policy, model_context=model_context,
                         s0_config=s0_config, benchmark=benchmark)
        object.__setattr__(self, "incumbent", build_capacity_fallback_allocator(
            tokenizer, **arguments, terminal_tool_rescue=True))
        object.__setattr__(self, "source_allocator", SourceAllocatedS0Controller(
            tokenizer, **arguments))
        # Both strategies use the same native S0 request-validation and decision
        # state layout. A rescued decision advances the same append-only session
        # and generation counter; the next decision still tries the incumbent.
        self.source_allocator._sessions = self.incumbent._sessions
        self.source_allocator._owner = self.incumbent._owner

    def __getattr__(self, name):
        return getattr(self.incumbent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRIBUTES:
            object.__setattr__(self, name, value)
        else:
            setattr(self.incumbent, name, value)
            setattr(self.source_allocator, name, value)

    def prepare(self, payload, *, ratio, max_new_tokens):
        try:
            # Includes every historical fallback and terminal argument rescue.
            # Return successful objects without changing their view or metadata.
            return self.incumbent.prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        except CapacityInfeasible:
            prepared = self.source_allocator.prepare(
                payload, ratio=ratio, max_new_tokens=max_new_tokens)
        prepared.metadata["source_allocation"]["phase"] = "after_incumbent_capacity_failure"
        prepared.metadata["source_allocation"]["legacy_fallback_invoked"] = True
        prepared.metadata["capacity_source_gate"] = {
            "version": POLICY_VERSION,
            "trigger": "incumbent_c1_capacity_infeasible",
            "incumbent_terminal_rescue_exhausted": True,
            "source_allocator_version": SOURCE_ALLOCATION_VERSION,
        }
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        allocator = self.source_allocator if isinstance(prepared.memory.view, SourceMemoryView) else self.incumbent
        return allocator.reconsider(prepared, draft_tool_calls,
                                    draft_text=draft_text, parse_error=parse_error)

    def repack_sources(self, prepared, *, candidate=None, derived_messages=None, goal_view=False):
        if isinstance(prepared.memory.view, SourceMemoryView):
            result, metadata, receipt = self.source_allocator.repack_sources(
                prepared, candidate=candidate, derived_messages=derived_messages, goal_view=goal_view)
            if metadata is not None:
                metadata["source_allocation"]["phase"] = "after_incumbent_capacity_failure"
                metadata["source_allocation"]["legacy_fallback_invoked"] = True
                metadata["capacity_source_gate"] = dict(prepared.metadata["capacity_source_gate"])
            return result, metadata, receipt
        from .repacking import repack

        return repack(self.incumbent, prepared, candidate=candidate,
                      derived_messages=derived_messages, goal_view=goal_view)
