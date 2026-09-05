# -*- coding: utf-8 -*-
"""Parameter-bearing denominator (survey item 4.0-4).

On the 93 C->W rows: how many carry at least one >=3-char argument leaf value,
counted once on the gold side (the TARGET action's arguments) and once on the
compressed arm's own emission (the PREDICTION action's arguments).  4.6's
grounding tier and 4.7's proposal-witness are only defined on this subset; if
it is smaller than what the MDE can resolve, that tier is not built.

Zero GPU — pure text over the frozen rows.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import load_jsonl  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402


def _leaf_values(arguments: Any) -> List[str]:
    """Stringify JSON leaves of an arguments object (dict values, list items)."""
    out: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif node is None:
            return
        else:
            out.append(str(node))

    walk(arguments)
    return out


def row_has_bearing_args(text: str, min_chars: int = 3, lenient: bool = True) -> bool:
    parsed = parse_tool_call(text or "")
    if parsed["arguments"] is not None:
        return any(len(v) >= min_chars for v in _leaf_values(parsed["arguments"]))
    if not lenient:
        return False
    # Lenient path: cap-censored emissions fail strict JSON parse
    # (unterminated_json) but their arguments region can still carry real
    # >=3-char leaf values — the strict-only reading misclassified 35 cap-hit
    # rows as non-bearing and shrank the denominator to 55/93
    if not parsed.get("args_span"):
        return False
    from t33_extract_features import arg_key_value_char_spans
    args_text = text[parsed["args_span"][0]:parsed["args_span"][1]]
    _keys, values = arg_key_value_char_spans(args_text)
    return any((ce - cs) >= min_chars for cs, ce in values)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2kv", required=True, help="battery_c2kv.jsonl (frozen r2)")
    parser.add_argument("--manifest", required=True, help="d_cw_manifest_r2.json")
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cw = set(manifest["cw_qids"])
    gold_side = emitted_side = both = 0
    strict_emitted = 0
    breakdown = {"strict_bearing": 0, "lenient_only": 0, "non_bearing": 0,
                 "non_bearing_censored": 0, "non_bearing_no_tool_call": 0}
    n = 0
    with io.open(args.c2kv, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["qid"] not in cw:
                continue
            n += 1
            g = row_has_bearing_args(row.get("target", ""), args.min_chars)
            e_len = row_has_bearing_args(row.get("prediction", ""), args.min_chars,
                                         lenient=True)
            e_str = row_has_bearing_args(row.get("prediction", ""), args.min_chars,
                                         lenient=False)
            strict_emitted += e_str
            if e_str:
                breakdown["strict_bearing"] += 1
            elif e_len:
                breakdown["lenient_only"] += 1
            else:
                breakdown["non_bearing"] += 1
                if int(row.get("generated_tokens") or 0) >= 128:
                    breakdown["non_bearing_censored"] += 1
                if "<tool_call>" not in (row.get("prediction") or ""):
                    breakdown["non_bearing_no_tool_call"] += 1
            gold_side += g
            emitted_side += e_len
            both += g and e_len

    assert n == len(cw), f"expected {len(cw)} C->W rows, saw {n}"
    stats = {
        "n_cw": n,
        "min_chars": args.min_chars,
        "gold_side_bearing": gold_side,
        "emitted_side_bearing": emitted_side,
        "emitted_side_bearing_strict_only": strict_emitted,
        "both_bearing": both,
        "gold_side_fraction": round(gold_side / n, 4),
        "emitted_side_fraction": round(emitted_side / n, 4),
        "non_strict_breakdown": breakdown,
        "note": ("4.6 grounding tier / 4.7 proposal-witness defined on "
                 "both_bearing rows; emitted side counts use the LENIENT "
                 "arg-span path (cap-censored rows still count when real "
                 ">=3-char values are present); compare against MDE 17-25pp "
                 "before building that tier."),
    }
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
