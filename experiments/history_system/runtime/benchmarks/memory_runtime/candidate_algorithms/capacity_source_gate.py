"""Shared protection gate; smaller representations require incumbent infeasibility."""
from __future__ import annotations

from history_memory.source_packing import SourceMemoryView

from ..always_compress import CapacityInfeasible
from ..backend_capacity import capacity_scope, current_constraints
from ..initial_factory import instantiate_initial
from ..source_allocation import SOURCE_ALLOCATION_VERSION, SourceAllocatedS0Controller
from .capacity_fallback import build_capacity_fallback_allocator

POLICY_VERSION = "c2kv-capacity-gated-source-allocation-v1"


class SmallProtectionS0Controller(SourceAllocatedS0Controller):
    """The common small policy may release exact protection to native history."""

    allow_native_history_demotion = True


class CapacityGatedSourceAllocator:
    """One lifecycle owner, two explicit strategies using the same representation."""

    _OWN_ATTRIBUTES = frozenset({"incumbent", "source_allocator", "_sessions", "_owner"})
    capacity_scope = staticmethod(capacity_scope)

    def __init__(self, tokenizer, *, packing, policy, model_context=None,
                 s0_config=None, benchmark="bfcl", initial_allocator_factory=None):
        arguments = dict(packing=packing, policy=policy, model_context=model_context,
                         s0_config=s0_config, benchmark=benchmark,
                         initial_allocator_factory=initial_allocator_factory)
        object.__setattr__(self, "incumbent", build_capacity_fallback_allocator(
            tokenizer, **arguments, terminal_tool_rescue=True))
        object.__setattr__(self, "source_allocator", instantiate_initial(
            SmallProtectionS0Controller, tokenizer, **arguments))
        object.__setattr__(self, "_sessions", {})
        object.__setattr__(self, "_owner", object())
        for branch in (self.incumbent, self.source_allocator):
            branch._sessions = self._sessions
            branch._owner = self._owner

    @property
    def policy_branches(self):
        """Both strategies prepare the same request under this one owner.

        A benchmark adapter that binds its request contract to the configured
        S0 controller must bind it on every branch, not only the incumbent
        reached through ``base`` delegation.
        """
        return (self.incumbent, self.source_allocator)

    def __getattr__(self, name):
        return getattr(self.incumbent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRIBUTES:
            object.__setattr__(self, name, value)
        else:
            setattr(self.incumbent, name, value)
            setattr(self.source_allocator, name, value)

    def _selected(self, memory):
        return self.source_allocator if isinstance(memory.view, SourceMemoryView) else self.incumbent

    def _annotate(self, metadata, *, phase):
        gist_fallback = getattr(self.incumbent, "supports_gist_capacity_fallback", True)
        metadata["source_allocation"].update(
            phase="after_incumbent_capacity_failure", legacy_fallback_invoked=True)
        metadata["capacity_source_gate"] = {
            "version": POLICY_VERSION, "trigger": "incumbent_c1_capacity_infeasible",
            "incumbent_terminal_rescue_exhausted": phase == "initial" and gist_fallback,
            "source_allocator_version": SOURCE_ALLOCATION_VERSION, "phase": phase,
        }
        if not gist_fallback:
            metadata["capacity_source_gate"]["incumbent_gist_fallback"] = {
                "status": "inapplicable",
                "reason": "native_residual_pool_has_no_complete_source_gist",
            }

    def prepare(self, payload, *, ratio, max_new_tokens):
        current_constraints(payload["session_id"], payload["decision_key"], "draft")
        try:
            # Return the exact historical object on every successful old path.
            return self.incumbent.prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        except CapacityInfeasible:
            prepared = self.source_allocator.prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        self._annotate(prepared.metadata, phase="initial")
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        current_constraints(prepared._session_id, prepared._decision_key, "regeneration")
        return self._selected(prepared.memory).reconsider(
            prepared, draft_tool_calls, draft_text=draft_text, parse_error=parse_error)

    def _native_metadata(self, metadata, memory, *, stage):
        annotate = getattr(self._selected(memory), "_native_metadata", None)
        if callable(annotate):
            annotate(metadata, memory, stage=stage)

    def repack_sources(self, prepared, *, candidate=None, derived_messages=None, goal_view=False):
        from .repacking import repack

        current_constraints(prepared._store.session_id, prepared.metadata["decision_key"], "regeneration")
        source_selected = isinstance(prepared.memory.view, SourceMemoryView)
        if not source_selected:
            incumbent = repack(self.incumbent, prepared, candidate=candidate,
                               derived_messages=derived_messages, goal_view=goal_view)
            if incumbent[0] is not None:
                return incumbent
        result, metadata, receipt = self.source_allocator.repack_sources(
            prepared, candidate=candidate, derived_messages=derived_messages, goal_view=goal_view)
        if metadata is not None:
            if source_selected and candidate is None and not derived_messages and not goal_view:
                metadata["capacity_source_gate"] = dict(prepared.metadata["capacity_source_gate"])
            else:
                self._annotate(metadata, phase="recovery")
            self._native_metadata(metadata, result.memory, stage="recovery")
        return result, metadata, receipt
