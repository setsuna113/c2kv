"""Post-hoc strict metric: tool_name + exact arguments match.

Reads a D-arm jsonl (rows with `prediction` and `target` full texts containing
<tool_call> blocks), computes:
  - strict_action_match: tool name AND canonical-JSON arguments equal
  - tool_name_match (diagnostic, already in rows)
  - protocol_legal (diagnostic)

Usage:
  python agent/d_strict_metric.py <arm1.jsonl> [<arm2.jsonl> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_call_payload(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first <tool_call> JSON block from text."""
    match = TOOL_CALL_RE.search(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("tool_name") or obj.get("function", {}).get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"name": name, "arguments": None}
    return {"name": name, "arguments": args}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strict_action_match(prediction: str, target: str) -> bool:
    """True iff both parse as tool calls with same name AND exact arguments."""
    pred = _parse_tool_call_payload(prediction)
    tgt = _parse_tool_call_payload(target)
    if pred is None or tgt is None:
        return False
    if pred["name"] != tgt["name"]:
        return False
    if pred["arguments"] is None or tgt["arguments"] is None:
        return pred["arguments"] == tgt["arguments"]
    return _canonical_json(pred["arguments"]) == _canonical_json(tgt["arguments"])


def compute_arm(path: Path) -> Dict[str, Any]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    rows = [r for r in rows if not r.get("skipped")]
    n = len(rows)
    strict = sum(1 for r in rows if strict_action_match(r.get("prediction",""), r.get("target","")))
    tnm = sum(1 for r in rows if r.get("tool_name_match"))
    return {
        "file": str(path),
        "n_scored": n,
        "strict_action_match": strict,
        "strict_rate": round(strict / n, 4) if n else None,
        "tool_name_match": tnm,
        "tool_name_rate": round(tnm / n, 4) if n else None,
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
              f" ({r['strict_rate']}) tool_name={r['tool_name_match']} ({r['tool_name_rate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
