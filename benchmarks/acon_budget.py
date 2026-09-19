"""Budget-bounded ACON history compression.

This is a separate arm from the original threshold-triggered ACON in
``textarms``. The caller supplies the exact actor-history token counter so
that every proposed output is checked on the surface the actor will see.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Tuple

if __package__:
    from . import textarms
else:
    import textarms


Message = Dict[str, Any]
Measure = Callable[[List[Message], List[int]], int]
Compress = Callable[[Dict[str, Any]], str]
_LOCK = threading.Lock()
_STATE: Dict[Tuple[Any, ...], Tuple[int, str, str]] = {}
_CACHE: Dict[str, str] = {}
_MAX_ATTEMPTS = 3


class BudgetExceeded(RuntimeError):
    """A required raw span or every bounded summary exceeds the history cap."""

    kind = "acon_history_budget_exceeded"

    def __init__(self, reason: str, receipt: Dict[str, Any]):
        super().__init__(f"{self.kind}: {reason}")
        self.reason = reason
        self.receipt = dict(receipt)
        self.budget = self.receipt["budget"]


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _measure(measure: Measure, messages: List[Message], indices: List[int]) -> int:
    result = measure(messages, indices)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError("measure must return a nonnegative integer token count")
    return result


def _summary_block(body: str) -> str:
    return f"\n<HISTORY_SUMMARY>\n{body}\n</HISTORY_SUMMARY>"


def _view(messages: List[Message], covered_until: int, summary: str,
          history_cutoff: int, task_index: int | None,
          preserve_task_packet: bool) -> Tuple[List[Message], List[int]]:
    """Build the actor view and classify by source, including synthetic summary."""
    out: List[Message] = []
    history: List[int] = []
    for message in messages:
        if message.get("role") == "system":
            out.append(dict(message))

    if preserve_task_packet and task_index is not None:
        out.append(dict(messages[task_index]))

    carrier = (task_index is not None and not preserve_task_packet
               and task_index < covered_until)
    if carrier:
        prompt = dict(messages[task_index])
        prompt["content"] = textarms._content_of(prompt) + _summary_block(summary)
        history.append(len(out))
        out.append(prompt)
    else:
        history.append(len(out))
        out.append({"role": "user", "content": _summary_block(summary).strip()})

    for source_index in range(covered_until, len(messages)):
        message = messages[source_index]
        if message.get("role") == "system" or source_index == task_index and preserve_task_packet:
            continue
        # A task instruction folded into the carrier is omitted from this
        # span; an unfolded instruction remains byte-for-byte raw.
        if source_index == task_index and carrier:
            continue
        if source_index < history_cutoff:
            history.append(len(out))
        out.append(dict(message))
    return out, history


def _receipt(stats: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "policy": "acon", "variant": "budget", "guideline": stats["guideline"],
        "reason": reason, "n_compressor_calls": stats["n_compressor_calls"],
        "budget": dict(stats["budget"]),
    }


def transform(messages: List[Message], compress: Compress, action_dialect,
              conv: str, *, model: str, budget_tokens: int,
              history_cutoff: int, measure: Measure,
              preserve_task_packet: bool = False, guideline: str = "ut_co",
              ) -> Tuple[List[Message], Dict[str, Any]]:
    """Fold completed history until the exact serialized history fits.

    ``history_cutoff`` is calculated on the original input by the proxy.
    ``measure`` must count only ``history_indices`` in the proposed actor
    messages, using the same serialization/tokenizer as the final guard.
    """
    if guideline not in textarms.ACON_HISTORY_PROMPTS:
        raise ValueError(f"unknown ACON guideline {guideline!r}")
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 1:
        raise ValueError("budget_tokens must be a positive integer")
    if (isinstance(history_cutoff, bool) or not isinstance(history_cutoff, int)
            or not 0 <= history_cutoff <= len(messages)):
        raise ValueError("history_cutoff must be an input message boundary")

    raw = [dict(message) for message in messages]
    task_index = next((i for i, message in enumerate(raw)
                       if message.get("role") == "user"), None)
    task_packet = task_index if preserve_task_packet else None
    raw_history = [i for i, message in enumerate(raw)
                   if i < history_cutoff and message.get("role") != "system"
                   and i != task_packet]
    before = _measure(measure, raw, raw_history)
    stats: Dict[str, Any] = {
        "policy": "acon", "variant": "budget", "guideline": guideline,
        "n_compressor_calls": 0, "history_compressed": False,
        "raw_chars": textarms._message_chars(raw),
        "out_chars": textarms._message_chars(raw),
        "history_indices": raw_history,
        "budget": {"limit": budget_tokens, "history_before": before,
                   "history_after": before, "attempts": 0,
                   "passed": before <= budget_tokens},
    }
    nonsystem = [i for i, message in enumerate(raw)
                 if message.get("role") != "system"]
    tail = set(nonsystem[-textarms.ACON_PRESERVE_LAST_K_MESSAGES:])
    foldable = [i for i in nonsystem if i < history_cutoff
                and i not in tail and i != task_packet]
    fold_end = max(foldable) + 1 if foldable else 0
    state_key = (conv, model, guideline, budget_tokens, preserve_task_packet)
    with _LOCK:
        state = _STATE.get(state_key)
    if (state is None or state[0] > fold_end
            or state[1] != _digest(raw[:state[0]])):
        covered_until, previous = 0, ""
    else:
        covered_until, previous = state[0], state[2]

    if previous:
        view, indices = _view(raw, covered_until, previous, history_cutoff,
                              task_index, preserve_task_packet)
        actual = _measure(measure, view, indices)
        stats["budget"]["history_after"] = actual
        if actual <= budget_tokens:
            stats.update(history_indices=indices,
                         out_chars=textarms._message_chars(view),
                         new_raw_messages_visible=sum(
                             i >= covered_until and i < history_cutoff
                             and raw[i].get("role") != "system"
                             and i != task_packet for i in range(len(raw))))
            stats["budget"]["passed"] = True
            return view, stats
    elif before <= budget_tokens:
        return raw, stats

    new_foldable = [i for i in foldable if i >= covered_until]
    if not new_foldable and not previous:
        stats["budget"]["passed"] = False
        raise BudgetExceeded("no foldable completed-history prefix", _receipt(stats, "no_foldable_prefix"))

    history_text = "\n".join(textarms._render_line(raw[i], action_dialect)
                             for i in new_foldable)
    task = textarms._task_text(raw)
    native_prompt = textarms.ACON_HISTORY_PROMPTS[guideline].format(
        task=task, prev_summary=previous or "(none)", history=history_text)
    empty_view, empty_indices = _view(raw, fold_end, "", history_cutoff,
                                       task_index, preserve_task_packet)
    fixed_tokens = _measure(measure, empty_view, empty_indices)
    stats["budget"]["fixed_tokens"] = fixed_tokens
    stats["budget"]["history_after"] = fixed_tokens
    if fixed_tokens >= budget_tokens:
        stats["budget"]["passed"] = False
        raise BudgetExceeded("required raw history and summary wrapper exceed budget",
                             _receipt(stats, "fixed_history_exceeds_budget"))

    output_cap = min(textarms.COMPRESSOR_DECODE["acon"]["max_tokens"],
                     budget_tokens - fixed_tokens)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        stats["budget"]["attempts"] = attempt
        prompt = (native_prompt + "\n\nBudget requirement: Return a complete "
                  f"HISTORY_SUMMARY using at most {output_cap} output tokens. "
                  "Keep the required facts and structure as concise as possible.")
        key = _digest(("acon-budget", model, guideline, budget_tokens,
                       output_cap, prompt))
        with _LOCK:
            summary = _CACHE.get(key)
        if summary is None:
            payload = textarms.compressor_payload("acon", model,
                                                   textarms.ACON_SYSTEM, prompt)
            payload["max_tokens"] = output_cap
            summary = compress(payload).strip()
            if not summary:
                raise textarms.TextarmCompressorError(
                    "acon budget compressor returned an empty summary")
            summary = textarms._history_summary_body(summary)
            if not summary:
                raise textarms.TextarmCompressorError(
                    "acon budget compressor returned an empty summary body")
            with _LOCK:
                _CACHE.setdefault(key, summary)
            stats["n_compressor_calls"] += 1
        candidate, indices = _view(raw, fold_end, summary, history_cutoff,
                                    task_index, preserve_task_packet)
        actual = _measure(measure, candidate, indices)
        stats["budget"]["history_after"] = actual
        if actual <= budget_tokens:
            with _LOCK:
                _STATE[state_key] = (fold_end, _digest(raw[:fold_end]), summary)
            stats.update(history_compressed=True, history_indices=indices,
                         out_chars=textarms._message_chars(candidate),
                         new_raw_messages_folded=len(new_foldable))
            stats["budget"]["passed"] = True
            return candidate, stats
        output_cap = max(1, min(output_cap - 1, output_cap // 2))

    stats["budget"]["passed"] = False
    raise BudgetExceeded("bounded compression did not fit",
                         _receipt(stats, "compression_attempts_exhausted"))


def reset_state() -> None:
    """Clear rolling summaries and independent compressor cache at episode reset."""
    with _LOCK:
        _STATE.clear()
        _CACHE.clear()
