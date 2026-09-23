"""Tool-region forwarding of the verified commit hooks.

Verified/Pending controllers record commit state on their own prepared decision.
The tool region wraps that decision in a frozen ``ToolPrepared``; forwarding the
wrapper instead of ``prepared.inner`` failed every Pending-Verified x tool-memory
decision on CUDA (box7 smoke, 2026-09-22) with FrozenInstanceError.
"""
from __future__ import annotations

from types import SimpleNamespace

from benchmarks.memory_runtime.event_native_tool import ToolPrepared, ToolRegionController


class RecordingInner:
    def commit_memory(self, prepared, final_memory):
        prepared.final_memory = final_memory
        return {"committed": True}

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        prepared.commit_accepted = parse_error is None
        prepared.commit_calls = list(candidate_calls)
        return {"accepted": prepared.commit_accepted, "draft_text": draft_text}

    def finalize_commit(self, prepared, candidate_calls):
        prepared.finalized = tuple(candidate_calls)
        return tuple(candidate_calls)


def wrapped(inner_prepared):
    return ToolPrepared(inner=inner_prepared, memory=None, metadata={}, plan=None,
                        eligible_chunks=None, ratio=8, max_new_tokens=16, source_payload={})


def controller():
    return ToolRegionController(RecordingInner(), tokenizer=None, spec=None,
                                model_context=1024, generator=None)


def test_commit_hooks_receive_the_inner_prepared_decision():
    inner = SimpleNamespace()
    region = controller()
    calls = [{"name": "ls", "arguments": {}}]

    verdict = region.validate_commit(wrapped(inner), calls, draft_text="draft")
    output = region.finalize_commit(wrapped(inner), calls)

    assert verdict == {"accepted": True, "draft_text": "draft"}
    assert inner.commit_calls == calls
    assert output == tuple(calls) and inner.finalized == tuple(calls)


def test_commit_hooks_pass_unwrapped_decisions_through():
    inner = SimpleNamespace()
    region = controller()

    region.validate_commit(inner, [], draft_text="", parse_error="bad json")

    assert inner.commit_accepted is False


def test_tool_region_commits_the_history_view_selected_by_final_generation():
    inner = SimpleNamespace()
    region = controller()
    first_history, second_history = object(), object()
    first_tool, second_tool = object(), object()
    prepared = wrapped(inner)
    prepared.history_views.extend(((first_tool, first_history),
                                   (second_tool, second_history)))

    assert region.commit_memory(prepared, second_tool) == {"committed": True}
    assert inner.final_memory is second_history
