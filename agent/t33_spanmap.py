# -*- coding: utf-8 -*-
"""Token-span map for ``<tool_call>`` emissions (survey item 4.0-3).

SPEC S2/S4 note: both repos have a text parser for the emitted action, but a
token-span mapping does not exist.  This module is that mapping, shared by

  * the in-process capture loop (server): maps ``<tool_call>`` char spans of
    the freshly generated continuation back to generated-token indices, so
    hidden states / logprobs can be sliced per region (name / arguments);
  * offline analysis (local): same parser over stored prediction text.

Pure stdlib on purpose — no torch, no tokenizer class.  The caller supplies a
``decode_fn`` (token-id list -> str); the capture loop passes
``lambda ids: tokenizer.decode(ids, skip_special_tokens=True)``.

Parse targets the battery dialect::

    Now let me ...
    Action:
    <tool_call>
    {"name":"mcp__environment__venmo__search_users","arguments":{"access_token":"eyJ..."}}
    </tool_call>

41/93 C->W predictions are censored at the 128-token cap and have NO closing
``</tool_call>``: the parser treats the region after ``<tool_call>`` up to the
end of text as the (unterminated) payload and reports ``closed=False``.  Rows
that fail strict JSON parsing are still given a lenient name/args span when a
regex can locate them — spans are recorded, never silently dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_ARGS_RE = re.compile(r'"arguments"\s*:\s*\{')


def _first_json_object_span(text: str, start: int) -> Optional[Tuple[int, int]]:
    """Char span of the first balanced JSON object at/after ``start``."""
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
    return None  # unterminated (cap-censored rows land here)


def parse_tool_call(text: str) -> Dict[str, Any]:
    """Parse the FIRST tool call in ``text``.

    Returns a dict with:
      has_tool_call   -- ``<tool_call>`` present
      closed          -- closing tag present after the payload
      payload_span    -- (cs, ce) of the JSON payload region (up to close tag
                         or end of text)
      json_span       -- (cs, ce) of the balanced JSON object, or None
      name            -- strict-parsed name, or None
      name_span       -- (cs, ce) of the name VALUE string content (lenient,
                         regex-based; present even when JSON is broken)
      args_span       -- (cs, ce) covering the arguments object (lenient)
      arguments       -- strict-parsed arguments dict, or None
      parse_ok        -- strict parse succeeded and produced a string name
      json_error      -- short reason when strict parse failed, else None
    """
    out: Dict[str, Any] = {
        "has_tool_call": False,
        "closed": False,
        "payload_span": None,
        "json_span": None,
        "name": None,
        "name_span": None,
        "args_span": None,
        "arguments": None,
        "parse_ok": False,
        "json_error": None,
    }
    open_at = text.find(TOOL_CALL_OPEN)
    if open_at < 0:
        return out
    out["has_tool_call"] = True
    payload_start = open_at + len(TOOL_CALL_OPEN)
    close_at = text.find(TOOL_CALL_CLOSE, payload_start)
    payload_end = close_at if close_at >= 0 else len(text)
    out["closed"] = close_at >= 0
    out["payload_span"] = (payload_start, payload_end)

    js = _first_json_object_span(text, payload_start)
    strict_end = js[1] if js else payload_end
    out["json_span"] = js
    payload = text[payload_start:strict_end]

    m = _NAME_RE.search(payload)
    if m:
        out["name_span"] = (payload_start + m.start(1), payload_start + m.end(1))
        if out["name"] is None:
            out["name"] = m.group(1)
    ma = _ARGS_RE.search(payload)
    if ma:
        # cover from the '{' to the end of the (possibly truncated) region
        out["args_span"] = (payload_start + ma.end() - 1, strict_end)

    if js is None:
        out["json_error"] = "unterminated_json"
        return out
    try:
        obj = json.loads(text[js[0]:js[1]])
    except json.JSONDecodeError as exc:
        out["json_error"] = f"json_decode:{exc.msg}"
        return out
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        out["json_error"] = "name_missing_or_not_str"
        return out
    out["name"] = obj["name"]
    out["arguments"] = obj.get("arguments")
    out["parse_ok"] = True
    return out


def token_char_offsets(decode_fn: Callable[[Sequence[int]], str],
                       generated_ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Exact per-token char offsets in the decoded continuation.

    Built from incremental prefix decodes — O(n^2) decode calls, n<=128 in the
    battery, so this costs milliseconds; it is exact for any tokenizer and
    avoids trusting ``return_offsets_mapping`` fast-path internals.
    """
    starts: List[int] = []
    ends: List[int] = []
    cursor = 0
    for i in range(len(generated_ids)):
        s = len(decode_fn(list(generated_ids[: i])))
        e = len(decode_fn(list(generated_ids[: i + 1])))
        starts.append(s)
        ends.append(e)
        cursor = e
    return list(zip(starts, ends))


def char_range_to_token(offsets: Sequence[Tuple[int, int]],
                        cs: int, ce: int) -> Optional[Tuple[int, int]]:
    """Map char span [cs, ce) to an inclusive token-index range.

    First token whose end > cs; last token whose start < ce.  Returns None
    when the span is empty or lies outside every token (e.g. zero-width
    decode artifacts); callers record None and count it, they do not drop the
    row.
    """
    first = None
    last = None
    for i, (s, e) in enumerate(offsets):
        if e > cs and s < ce:
            if first is None:
                first = i
            last = i
    if first is None:
        return None
    return (first, last)


def spans_from_generation(decode_fn: Callable[[Sequence[int]], str],
                          generated_ids: Sequence[int]) -> Dict[str, Any]:
    """Full span record for one generated continuation.

    Token indices are 0-based over ``generated_ids`` (the continuation only,
    NOT the prompt).  ``*_first/*_last`` are inclusive; None means the span
    could not be mapped — always counted, never silently dropped.
    """
    text = decode_fn(list(generated_ids))
    parsed = parse_tool_call(text)
    offsets = token_char_offsets(decode_fn, generated_ids)

    def to_tok(span):
        if span is None:
            return None, None
        rng = char_range_to_token(offsets, span[0], span[1])
        if rng is None:
            return None, None
        return rng[0], rng[1]

    name_first, name_last = to_tok(parsed.get("name_span"))
    args_first, args_last = to_tok(parsed.get("args_span"))
    payload_first, payload_last = to_tok(parsed.get("payload_span"))
    n = len(generated_ids)
    return {
        "n_generated": n,
        "text": text,
        "has_tool_call": parsed["has_tool_call"],
        "closed": parsed["closed"],
        "parse_ok": parsed["parse_ok"],
        "json_error": parsed["json_error"],
        "name": parsed["name"],
        "name_first": name_first,
        "name_last": name_last,
        "args_first": args_first,
        "args_last": args_last,
        "payload_first": payload_first,
        "payload_last": payload_last,
        # positional anchors independent of parsing
        "first_tok": 0,
        "last_tok": n - 1,
        "penult_tok": n - 2 if n >= 2 else None,
    }
