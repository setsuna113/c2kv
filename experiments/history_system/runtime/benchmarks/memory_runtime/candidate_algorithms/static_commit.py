"""Render a proven duplicate-call removal without allowing action invention."""
from __future__ import annotations

from dataclasses import replace

from .goal_commit import corrected_draft


def render_filtered_commit(draft, calls, *, benchmark):
    remaining = iter(draft.tool_calls)
    for call in calls:
        if not any(old == call for old in remaining):
            raise ValueError("A ledger commit may only remove unchanged calls in order")
    if not calls:
        text = ("Finish conversation" if benchmark == "acebench" else
                "The requested action is already recorded as completed.")
        return replace(draft, text=text, content=text, tool_calls=(), status="text",
                       reason="verified_completed_action_suppressed", reasoning_content=None)
    # Reuse the strict field renderer with the already-validated subsequence.
    filtered = replace(draft, tool_calls=tuple(calls), content="", reasoning_content=None)
    return replace(corrected_draft(filtered, calls, benchmark=benchmark),
                   reason="verified_completed_actions_filtered")
