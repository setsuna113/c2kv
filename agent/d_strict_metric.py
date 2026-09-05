"""Post-hoc strict metric v2 (prereg v2.1): tool_name + exact arguments.

Design constraints (handoff §2.3, B14-B17):
- The tool NAME uses the harness parser verbatim (unclosed-block fallbacks
  included) so ``strict ⊆ tool_name_match`` BY CONSTRUCTION — the harness
  recovers names from 41/93 unclosed full-arm predictions that the old
  closed-tag-only parser silently dropped.
- Arguments parse from the FIRST successfully-parsing <tool_call> block,
  iterating <tool_call> openings paired with their nearest close (a plain
  findall lets a truncated first block swallow a well-formed second block).
- "arguments unparseable" and "arguments null" are SEPARATE states — the
  old shared None sentinel made null-vs-null a false positive.
- Files are read with explicit utf-8 (rows are written ensure_ascii=False).

Usage:
  python agent/d_strict_metric.py <arm1.jsonl> [<arm2.jsonl> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# VERBATIM copy of eval_agent_tool_definition_c2kv._extract_tool_name
# (that module imports torch at top level, unusable from torch-free test
# paths — keep byte-identical; the equivalence is asserted in
# metrology/test_strict_metric_v2.py against the frozen source shape).
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
NAME_KEY_RE = re.compile(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"')
NAME_IN_BLOCK_RE = re.compile(r"<tool_call>.*?([A-Za-z0-9_.:-]+).*?</tool_call>", re.S)


def _extract_tool_name(text: str) -> Optional[str]:
    if not text:
        return None
    blocks = TOOL_CALL_BLOCK_RE.findall(text)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            if name:
                return str(name)
    match = NAME_KEY_RE.search(text)
    if match:
        return match.group(1)
    match = NAME_IN_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    return None


# argument states — the two failure modes must never share a sentinel (B15)
ARGS_OK = "ok"              # arguments parsed as a JSON dict
ARGS_NULL = "null"          # explicit null literal
ARGS_UNPARSEABLE = "unparseable"  # string arguments whose json.loads fails
ARGS_ABSENT = "absent"      # parseable block carries no arguments key
ARGS_NO_BLOCK = "no_block"  # name recovered by fallback, no parseable block


def _first_json_dict_in(chunk: str) -> Optional[dict]:
    """First JSON object parsed from the longest `{...}` span in chunk
    (shrinking the end brace until json accepts)."""
    start = chunk.find("{")
    if start == -1:
        return None
    for end in range(len(chunk), start, -1):
        if chunk[end - 1] != "}":
            continue
        try:
            obj = json.loads(chunk[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _iter_tool_call_chunks(text: str):
    """Each <tool_call> opening paired with its NEAREST following close.

    B16: a plain findall lets a truncated first block's match span (and
    swallow) a well-formed second block; iterating openings recovers the
    second block as its own candidate.
    """
    for opener in re.finditer(r"<tool_call>", text or ""):
        close = text.find("</tool_call>", opener.end())
        if close == -1:
            continue
        yield text[opener.end():close]


def _parse_arguments(text: str) -> Dict[str, Any]:
    """Parse the first successfully-parsing block's arguments.

    Returns {"state": ..., "value": ...}. ``state`` distinguishes every
    failure mode; ``value`` is the canonical-JSON string when state == ok.
    """
    for chunk in _iter_tool_call_chunks(text):
        obj = _first_json_dict_in(chunk)
        if obj is None:
            continue
        if "arguments" not in obj and "parameters" not in obj:
            return {"state": ARGS_ABSENT, "value": None}
        args = obj.get("arguments", obj.get("parameters"))
        if args is None:
            return {"state": ARGS_NULL, "value": None}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return {"state": ARGS_UNPARSEABLE, "value": None}
        if isinstance(args, dict):
            return {"state": ARGS_OK, "value": _canonical_json(args)}
        return {"state": ARGS_UNPARSEABLE, "value": None}
    return {"state": ARGS_NO_BLOCK, "value": None}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """{'name': str, 'args_state': str, 'args_canonical': str|None} or None."""
    name = _extract_tool_name(text)
    if name is None:
        return None
    parsed = _parse_arguments(text)
    return {
        "name": name,
        "args_state": parsed["state"],
        "args_canonical": parsed["value"],
    }


def _args_payload(parsed: Dict[str, Any]):
    """Comparable payload; never equal across different states (B15 fix)."""
    if parsed["args_state"] == ARGS_OK:
        return ("value", parsed["args_canonical"])
    if parsed["args_state"] in (ARGS_NULL, ARGS_ABSENT):
        return (parsed["args_state"],)
    return None  # unparseable / no_block: arguments unverifiable — never a match


def strict_action_match(prediction: str, target: str) -> bool:
    """True iff same tool name AND exact canonical-JSON arguments.

    Unverifiable arguments (unparseable / no parseable block on either side)
    are NOT a match; explicit-null vs explicit-null and absent-vs-absent ARE.
    """
    pred = parse_tool_call(prediction)
    tgt = parse_tool_call(target)
    if pred is None or tgt is None:
        return False
    if pred["name"] != tgt["name"]:
        return False
    p, t = _args_payload(pred), _args_payload(tgt)
    if p is None or t is None:
        return False
    return p == t


def compute_arm(path: Path) -> Dict[str, Any]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("skipped")]
    n = len(rows)
    strict = sum(1 for r in rows if strict_action_match(r.get("prediction", ""), r.get("target", "")))
    tnm = sum(1 for r in rows if r.get("tool_name_match"))
    # nesting guarantee: strict must never exceed the harness name metric
    state_tally: Dict[str, int] = {}
    for r in rows:
        parsed = parse_tool_call(r.get("prediction", ""))
        state_tally[parsed["args_state"] if parsed else "no_name"] = (
            state_tally.get(parsed["args_state"] if parsed else "no_name", 0) + 1
        )
    return {
        "file": str(path),
        "n_scored": n,
        "strict_action_match": strict,
        "strict_rate": round(strict / n, 4) if n else None,
        "tool_name_match": tnm,
        "tool_name_rate": round(tnm / n, 4) if n else None,
        "strict_leq_tool_name": strict <= tnm,
        "pred_args_states": state_tally,
    }


def main(argv=None):
    if not argv:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print(__doc__)
        return 1
    results = [compute_arm(Path(p)) for p in argv]
    for r in results:
        print(f"{Path(r['file']).name}: n={r['n_scored']} strict={r['strict_action_match']}"
              f" ({r['strict_rate']}) tool_name={r['tool_name_match']} ({r['tool_name_rate']})"
              f" states={r['pred_args_states']}")
        if not r["strict_leq_tool_name"]:
            print(f"  !! NESTING VIOLATION: strict > tool_name — parser bug")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
