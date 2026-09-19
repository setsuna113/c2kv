"""Exact-budget HiAgent history for the tool-native paper comparison.

Subgoal IDs and retrieved trajectories come from the original, untrimmed
request. Completed subgoals use HiAgent's original summary prompt and decode
settings. The active trajectory starts raw; on overflow, FIFO may remove old
action/observation records while retaining the current subgoal text. This is
the budget adaptation of HiAgent's own oldest-record context trimming, not a
claim that the paper specified a token-bounded active trajectory.

The caller supplies the actor's exact history-token counter. Every synthetic
summary, retained user turn, retrieval trajectory, and feedback message is in
``history_indices``; current input at or after ``history_cutoff`` is outside it.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

if __package__:
    from . import textarms
else:
    import textarms


Message = Dict[str, Any]
Measure = Callable[[List[Message], List[int]], int]
Compress = Callable[[Dict[str, Any]], str]
_LOCK = threading.Lock()
_SUMMARY_CACHE: Dict[str, str] = {}
BUDGET_UNAVAILABLE_FEEDBACK = "budget_unavailable: continue without trajectory retrieval."


class BudgetExceeded(RuntimeError):
    """The required HiAgent history floor cannot fit the actor history cap."""

    kind = "hiagent_history_budget_exceeded"

    def __init__(self, reason: str, receipt: Dict[str, Any]):
        super().__init__(f"{self.kind}: {reason}")
        self.reason = reason
        self.receipt = dict(receipt)
        self.budget = self.receipt["budget"]


class RetrievalBudgetExceeded(BudgetExceeded):
    """A requested complete trajectory cannot fit without truncation."""

    kind = "hiagent_retrieval_budget_exceeded"


@dataclass(frozen=True)
class _Record:
    indices: Tuple[int, ...]
    segment_id: int
    pinned: bool = False


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _count(measure: Measure, messages: List[Message], indices: List[int]) -> int:
    value = measure(messages, indices)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("measure must return a nonnegative integer token count")
    return value


def _receipt(stats: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "policy": "hiagent", "variant": stats["variant"], "reason": reason,
        "n_compressor_calls": stats["n_compressor_calls"],
        "retrieved_subgoals": list(stats["retrieved_subgoals"]),
        "invalid_retrieval_subgoals": list(stats["invalid_retrieval_subgoals"]),
        "history_indices": list(stats["history_indices"]),
        "budget": dict(stats["budget"]),
    }


def _note(variant: str, environment_action_format: str) -> str:
    note = (textarms.HIAGENT_SUBGOAL_NOTE
            if environment_action_format == "native_tool_call"
            else textarms.HIAGENT_PYTHON_ACTION_NOTE)
    if variant == "full":
        retrieval = (textarms.HIAGENT_RETRIEVAL_NOTE
                     if environment_action_format == "native_tool_call"
                     else textarms.HIAGENT_PYTHON_RETRIEVAL_NOTE)
        note = note.rstrip() + "\n" + retrieval.strip() + "\n"
    return note


def _systems(raw: List[Message], note: str, default_system: str) -> List[Message]:
    systems = [dict(message) for message in raw if message.get("role") == "system"]
    if systems:
        systems[0]["content"] = textarms._content_of(systems[0]).rstrip() + "\n" + note
    else:
        systems = [{"role": "system", "content": default_system.rstrip() + "\n" + note.strip()}]
    return systems


def _segments(raw: List[Message]) -> Tuple[List[int], Dict[int, str], Dict[int, int]]:
    """Assign stable one-based IDs before any budget eviction."""
    segment_of = [0] * len(raw)
    subgoals: Dict[int, str] = {}
    marker_of: Dict[int, int] = {}
    active_id = 0
    for index, message in enumerate(raw):
        subgoal = textarms._subgoal_of(message)
        if subgoal is not None:
            active_id += 1
            subgoals[active_id] = subgoal
            marker_of[active_id] = index
        segment_of[index] = active_id
    return segment_of, subgoals, marker_of


def _records(raw: List[Message], cutoff: int, task_index: Optional[int],
             segment_of: List[int]) -> List[_Record]:
    """Group a native tool call with its contiguous matching results."""
    records: List[_Record] = []
    index = 0
    while index < cutoff:
        if raw[index].get("role") == "system" or index == task_index:
            index += 1
            continue
        message = raw[index]
        grouped = [index]
        pinned = False
        if message.get("role") == "assistant" and message.get("tool_calls"):
            lookahead = index + 1
            # OpenAI-style results form a contiguous run. Keep the whole run
            # with its call, including results without a tool_call_id.
            while lookahead < len(raw) and raw[lookahead].get("role") == "tool":
                if lookahead < cutoff:
                    grouped.append(lookahead)
                else:
                    pinned = True  # The current result must retain its call.
                lookahead += 1
            index = lookahead
        else:
            index += 1
        records.append(_Record(tuple(grouped), segment_of[grouped[0]], pinned))
    return records


def _feedback_message(feedback: Any) -> Optional[Message]:
    if feedback is None or feedback == "":
        return None
    if isinstance(feedback, dict) and feedback.get("role") == "user":
        return dict(feedback)
    body = feedback if isinstance(feedback, str) else json.dumps(
        feedback, ensure_ascii=False, sort_keys=True)
    return {"role": "user", "content": f"HiAgent retrieval feedback: {body}"}


def _summary(model: str, subgoal: str, trajectory: List[Message],
             compress: Compress, action_dialect, stats: Dict[str, Any]) -> str:
    rendered = "\n".join(textarms._render_line(message, action_dialect)
                         for message in trajectory)
    key = _digest((model, subgoal, rendered))
    with _LOCK:
        cached = _SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    prompt = textarms.HIAGENT_SUMMARY_USER_TEMPLATE.format(
        example=textarms.HIAGENT_EXAMPLE,
        formatted_trajectory=rendered, subgoal=subgoal)
    payload = textarms.compressor_payload(
        "hiagent", model, textarms.HIAGENT_SUMMARY_SYSTEM, prompt)
    stats["n_compressor_calls"] += 1
    result = compress(payload).strip()
    if not result:
        raise textarms.TextarmCompressorError(
            "hiagent budget compressor returned an empty summary")
    with _LOCK:
        _SUMMARY_CACHE.setdefault(key, result)
    return result


def transform(messages: List[Message], compress: Compress, action_dialect,
              conv: str, *, model: str, budget_tokens: int,
              history_cutoff: int, measure: Measure,
              preserve_task_packet: bool = False,
              retrieved_subgoals: Optional[List[int]] = None,
              retrieval_feedback: Any = None,
              environment_action_format: str = "native_tool_call",
              default_system: str = "You are a helpful assistant.",
              variant: str = "summary",
              ) -> Tuple[List[Message], Dict[str, Any]]:
    """Serialize HiAgent under an exact history cap, evicting oldest records.

    ``history_cutoff`` belongs to the original input. The first user message
    may be excluded as an immutable task packet. The current input is copied
    unchanged. Completed trajectories requested for retrieval are immutable;
    invalid IDs are reported without fabricating trajectories. A dropped
    active marker is replaced at its source position by its one-line subgoal
    text, allowing old active action/observation pairs to leave the budget.
    """
    del conv  # The original request itself is the authoritative episode archive.
    if variant not in ("summary", "full"):
        raise ValueError(f"unknown HiAgent variant {variant!r}")
    if environment_action_format not in ("native_tool_call", "python_content"):
        raise ValueError(f"unknown HiAgent environment action format {environment_action_format!r}")
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 1:
        raise ValueError("budget_tokens must be a positive integer")
    if (isinstance(history_cutoff, bool) or not isinstance(history_cutoff, int)
            or not 0 <= history_cutoff <= len(messages)):
        raise ValueError("history_cutoff must be an input message boundary")
    requested = set(retrieved_subgoals or [])
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in requested):
        raise ValueError("retrieved_subgoals must contain positive integer IDs")
    if requested and variant != "full":
        raise ValueError("trajectory retrieval requires variant='full'")

    raw = [dict(message) for message in messages]
    task_index = (next((i for i, message in enumerate(raw)
                        if message.get("role") == "user" and i < history_cutoff), None)
                  if preserve_task_packet else None)
    segment_of, subgoals, marker_of = _segments(raw)
    completed_ids = set(range(1, max(subgoals, default=0)))
    valid_requested = requested & completed_ids
    note = _note(variant, environment_action_format)
    systems = _systems(raw, note, default_system)
    records = _records(raw, history_cutoff, task_index, segment_of)
    feedback = _feedback_message(retrieval_feedback)
    active_id = max(subgoals, default=0)
    active_marker = marker_of.get(active_id)
    pinned_records = {
        index for index, record in enumerate(records)
        if record.pinned or record.segment_id in valid_requested
    }
    evictable = [index for index in range(len(records)) if index not in pinned_records]

    stats: Dict[str, Any] = {
        "policy": "hiagent", "variant": variant,
        "environment_action_format": environment_action_format,
        "n_compressor_calls": 0, "n_segments": len(subgoals),
        "n_summarized": 0,
        "degenerate": not subgoals and any(m.get("role") == "assistant" for m in raw),
        "retrieved_subgoals": sorted(valid_requested),
        "invalid_retrieval_subgoals": sorted(requested - completed_ids),
        "raw_chars": textarms._message_chars(raw),
        "history_indices": [],
        "budget": {"limit": budget_tokens, "history_before": 0,
                   "history_after": 0, "fixed_tokens": 0,
                   "reserved_feedback_tokens": 0,
                   "history_with_reserved_feedback": 0,
                   "reserve_passed": False,
                   "evicted_records": 0, "attempts": 0, "passed": False},
    }

    def view(retained: set[int], *, summarize: bool) -> Tuple[List[Message], List[int], int]:
        out = list(systems)
        if task_index is not None:
            out.append(dict(raw[task_index]))
        history_start = len(out)
        summarized = 0
        selected: Dict[int, List[Message]] = {}
        for record_index, record in enumerate(records):
            if record_index in retained:
                selected.setdefault(record.segment_id, []).extend(
                    dict(raw[index]) for index in record.indices)
        # Re-serialize every subgoal in original source order after each FIFO
        # removal, so a summary never describes an evicted record as visible.
        seen_segments: set[int] = set()
        for record_index, record in enumerate(records):
            segment_id = record.segment_id
            if segment_id in seen_segments:
                continue
            seen_segments.add(segment_id)
            trajectory = selected.get(segment_id, [])
            if segment_id == 0 or segment_id == active_id or segment_id in valid_requested:
                for later_index, later_record in enumerate(records):
                    if later_record.segment_id != segment_id:
                        continue
                    if (segment_id == active_id and active_marker is not None
                            and active_marker in later_record.indices
                            and later_index not in retained):
                        line = textarms._content_of(raw[active_marker]).split("\n", 1)[0]
                        out.append({"role": "assistant", "content": line})
                    if later_index in retained:
                        out.extend(dict(raw[index]) for index in later_record.indices)
                continue
            if not trajectory:
                continue
            out.extend(dict(message) for message in trajectory
                       if message.get("role") == "user")
            if summarize:
                body = _summary(model, subgoals[segment_id], trajectory,
                                compress, action_dialect, stats)
                out.append({"role": "user", "content":
                            f"Subgoal {segment_id}: {subgoals[segment_id]}\nSummary: {body}"})
                summarized += 1
        if feedback is not None:
            out.append(dict(feedback))
        history_indices = list(range(history_start, len(out)))
        out.extend(dict(raw[index]) for index in range(history_cutoff, len(raw))
                   if raw[index].get("role") != "system" and index != task_index)
        return out, history_indices, summarized

    def measured_candidate(candidate: List[Message], indices: List[int]) -> Tuple[int, int]:
        """Check both the sent view and the possible denial-feedback retry."""
        actual = _count(measure, candidate, indices)
        if variant != "full" or feedback is not None:
            return actual, actual
        reserved = list(candidate)
        insertion = indices[-1] + 1 if indices else len(systems) + int(task_index is not None)
        reserved.insert(insertion, _feedback_message(BUDGET_UNAVAILABLE_FEEDBACK))
        reserve_indices = list(range(indices[0], insertion + 1)) if indices else [insertion]
        return actual, _count(measure, reserved, reserve_indices)

    all_retained = set(range(len(records)))
    # Before is the complete unsummarized source history. The floor renderer
    # deliberately omits nonretrieved completed trajectories, so it cannot
    # represent this raw baseline.
    before_view = list(systems)
    if task_index is not None:
        before_view.append(dict(raw[task_index]))
    before_start = len(before_view)
    for record in records:
        before_view.extend(dict(raw[index]) for index in record.indices)
    if feedback is not None:
        before_view.append(dict(feedback))
    before_indices = list(range(before_start, len(before_view)))
    before_view.extend(dict(raw[index]) for index in range(history_cutoff, len(raw))
                       if raw[index].get("role") != "system" and index != task_index)
    stats["budget"]["history_before"] = _count(measure, before_view, before_indices)
    floor_view, floor_indices, _ = view(pinned_records, summarize=False)
    floor_tokens, floor_reserved = measured_candidate(floor_view, floor_indices)
    stats["budget"].update(
        fixed_tokens=floor_tokens, history_after=floor_tokens,
        reserved_feedback_tokens=max(0, floor_reserved - floor_tokens),
        history_with_reserved_feedback=floor_reserved,
        reserve_passed=floor_reserved <= budget_tokens)
    stats["history_indices"] = floor_indices
    if floor_tokens > budget_tokens or floor_reserved > budget_tokens:
        reason = "requested_full_trajectory_exceeds_budget" if valid_requested else "fixed_history_exceeds_budget"
        error = RetrievalBudgetExceeded if valid_requested else BudgetExceeded
        raise error(reason, _receipt(stats, reason))

    retained = set(all_retained)
    for removed in range(len(evictable) + 1):
        stats["budget"].update(evicted_records=removed, attempts=removed + 1)
        candidate, indices, n_summarized = view(retained, summarize=True)
        actual, reserved = measured_candidate(candidate, indices)
        stats["budget"].update(
            history_after=actual,
            reserved_feedback_tokens=max(0, reserved - actual),
            history_with_reserved_feedback=reserved,
            reserve_passed=reserved <= budget_tokens)
        stats["history_indices"] = indices
        if actual <= budget_tokens and reserved <= budget_tokens:
            stats["budget"]["passed"] = True
            stats["n_summarized"] = n_summarized
            stats["history_compressed"] = bool(n_summarized)
            stats["out_chars"] = textarms._message_chars(candidate)
            return candidate, stats
        if removed < len(evictable):
            retained.remove(evictable[removed])

    reason = "requested_full_trajectory_exceeds_budget" if valid_requested else "bounded_history_did_not_fit"
    error = RetrievalBudgetExceeded if valid_requested else BudgetExceeded
    raise error(reason, _receipt(stats, reason))


def reset_state() -> None:
    """Clear cached summaries when an evaluation episode is reset."""
    with _LOCK:
        _SUMMARY_CACHE.clear()
