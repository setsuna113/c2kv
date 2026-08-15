"""R4 task D: build typed/random raw-KV anchor span tables for the frozen set.

For every qid in configs/r4_d_qids.json, rebuild the PR#1 history docs
(the SAME truncated per-doc ids the harness compresses) and detect control
token spans (typed arm) plus an equal-budget random span set (random arm):

typed (frozen definition, configs/r4_prereg.md):
  1. <|im_start|> / <|im_end|> occurrences (1-token spans);
  2. inside every <tool_call>...</tool_call> region: the tool-name field
     VALUE ("name"/"tool_name"/"function_name"), all parameter KEYS
     (recursive dict keys), all JSON structural chars { } [ ] : , ";
  3. tool-role messages whose content parses as JSON: same structural chars
     and keys.
Char spans are mapped to token indices via fast-tokenizer offset mapping,
asserted against the harness doc ids (per-doc fallback: token-subsequence
envelope + fully-structural-token scan, recorded as fallback_docs).

random (frozen): per qid, same span-count and same length MULTISET as typed,
positions uniform over NON-CONTROL tokens of the same doc when room allows
(else the qid-level non-control pool), seed 20260815 (per-qid subseed
sha256("20260815:"+qid)). Budget equality is by construction (identical
length multiset) and is reported globally.

CPU only. Usage (NPU server, repo root):
  python agent/r4_anchor_spans.py --out configs/r4_anchor_spans.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))

import eval_agent_history_c2kv as HH  # noqa: E402

logger = logging.getLogger("r4_anchor_spans")

SEED = 20260815
IM_START_ID = 151644
IM_END_ID = 151645
STRUCT_CHARS = set('{}[]:,"')
NAME_KEY_RE = re.compile(r'"(?:name|tool_name|function_name)"\s*:\s*"((?:[^"\\]|\\.)*)"')
GENERIC_KEY_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:')
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _pr1_args() -> Any:
    """argparse namespace reproducing the PR#1 (s4) invocation defaults."""
    argv = [
        "prog",
        "--model", "./checkpoints/qwen3-4b-agent-history-c2kv-npu",
        "--base_model", "./models/Qwen3-4B-Instruct-2507",
        "--tokenizer", "./models/Qwen3-4B-Instruct-2507",
        "--include_tools", "True",
        "--max_examples", "0",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def _doc_char_offsets(tokenizer: Any, doc_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    """(decoded text, per-token char offsets) with roundtrip assertion."""
    text = tokenizer.decode(doc_ids, skip_special_tokens=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == list(doc_ids), "retokenize roundtrip mismatch"
    return text, [tuple(o) for o in enc["offset_mapping"]]


def _char_spans_to_token_index(offsets: Sequence[Tuple[int, int]], char_spans: List[Tuple[int, int]]) -> List[int]:
    idx = set()
    for cs, ce in char_spans:
        for i, (ts, te) in enumerate(offsets):
            if ts < ce and te > cs:
                idx.add(i)
    return sorted(idx)


def _json_control_char_spans(region: str) -> List[Tuple[int, int]]:
    """Char spans of control content inside a JSON-ish region string."""
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[{}\[\]:,\"]", region):
        spans.append((m.start(), m.end()))
    name_m = NAME_KEY_RE.search(region)
    if name_m:
        spans.append((name_m.start(1), name_m.end(1)))
    for key_m in GENERIC_KEY_RE.finditer(region):
        spans.append((key_m.start(1), key_m.end(1)))
    return spans


def _typed_spans_for_doc(tokenizer: Any, message: Dict[str, Any], doc_ids: List[int]) -> Tuple[List[List[int]], bool]:
    """Return (spans as token [s, e) list, used_fallback)."""
    spans: List[List[int]] = [[i, i + 1] for i, t in enumerate(doc_ids) if t in (IM_START_ID, IM_END_ID)]
    try:
        text, offsets = _doc_char_offsets(tokenizer, doc_ids)
    except Exception:
        # Fallback: token-subsequence envelope detection + structural tokens.
        open_ids = tokenizer(TOOL_CALL_OPEN, add_special_tokens=False)["input_ids"]
        close_ids = tokenizer(TOOL_CALL_CLOSE, add_special_tokens=False)["input_ids"]

        def _find_all(hay: List[int], needle: List[int]) -> List[int]:
            return [i for i in range(len(hay) - len(needle) + 1) if hay[i : i + len(needle)] == needle]

        opens = _find_all(doc_ids, open_ids)
        closes = _find_all(doc_ids, close_ids)
        for o in opens:
            nxt = [c for c in closes if c >= o]
            e = (nxt[0] + len(close_ids)) if nxt else min(o + len(open_ids) + 64, len(doc_ids))
            spans.append([o, e])
        for i, t in enumerate(doc_ids):
            tok_text = tokenizer.decode([t], skip_special_tokens=False)
            if tok_text and all(c in STRUCT_CHARS or c.isspace() for c in tok_text):
                spans.append([i, i + 1])
        return _merge_spans(spans, len(doc_ids)), True

    role = message.get("role")
    cursor = 0
    while True:
        o = text.find(TOOL_CALL_OPEN, cursor)
        if o < 0:
            break
        c = text.find(TOOL_CALL_CLOSE, o)
        region_end = c if c >= 0 else len(text)
        region = text[o:region_end]
        char_spans = [(o + s, o + e) for s, e in _json_control_char_spans(region)]
        char_spans.append((o, o + len(TOOL_CALL_OPEN)))
        if c >= 0:
            char_spans.append((c, c + len(TOOL_CALL_CLOSE)))
        for i in _char_spans_to_token_index(offsets, char_spans):
            spans.append([i, i + 1])
        cursor = region_end + len(TOOL_CALL_CLOSE)
    if role == "tool":
        content = HH._agent_message_content_to_text(message.get("content"))
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            body_start = text.find(stripped[:32])
            if body_start >= 0:
                body_spans = _json_control_char_spans(text[body_start:])
                for i in _char_spans_to_token_index(
                    offsets, [(body_start + s, body_start + e) for s, e in body_spans]
                ):
                    spans.append([i, i + 1])
    return _merge_spans(spans, len(doc_ids)), False


def _merge_spans(spans: List[List[int]], doc_len: int) -> List[List[int]]:
    if not spans:
        return []
    spans = sorted((max(0, s), min(e, doc_len)) for s, e in spans if s < e)
    merged: List[List[int]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _random_spans_for_qid(
    qid: str, typed_by_doc: Dict[int, List[List[int]]], doc_lens: List[int]
) -> Dict[int, List[List[int]]]:
    rng = random.Random(int(hashlib.sha256(f"{SEED}:{qid}".encode()).hexdigest()[:16], 16))
    control = {
        d: {i for s, e in spans for i in range(s, e)} for d, spans in typed_by_doc.items()
    }
    result: Dict[int, List[List[int]]] = {}
    for d, spans in sorted(typed_by_doc.items()):
        doc_len = doc_lens[d]
        blocked = control.get(d, set())
        free = [i for i in range(doc_len) if i not in blocked]
        chosen: List[List[int]] = []
        for s, e in spans:
            length = e - s
            candidates = [st for st in free if st + length <= doc_len and all((st + k) not in blocked for k in range(length)) and all(not (st < ce and st + length > cs) for cs, ce in chosen)]
            if not candidates:
                continue
            st = rng.choice(candidates)
            chosen.append([st, st + length])
        if chosen:
            result[d] = _merge_spans(chosen, doc_len)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid_file", default="./configs/r4_d_qids.json")
    p.add_argument("--out", default="./configs/r4_anchor_spans.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    frozen = json.loads(Path(args.qid_file).read_text(encoding="utf-8"))
    qids: List[str] = frozen["qids"]
    hargs = _pr1_args()
    tokenizer = HH._load_tokenizer(hargs)
    examples, selection_skips = HH._load_examples(hargs, tokenizer)
    logger.info("source reproduced %d examples (selection_skips=%s)", len(examples), selection_skips)
    wanted = set(qids)
    by_qid = {}
    for ex in examples:
        if ex.qid in wanted and ex.qid not in by_qid:
            by_qid[ex.qid] = ex
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} frozen qids not reproduced: {missing[:5]}")

    per_qid: Dict[str, Any] = {}
    total_typed = 0
    total_random = 0
    total_fallback_docs = 0
    for n, qid in enumerate(qids):
        example = by_qid[qid]
        history = HH._history_messages(tokenizer, example, hargs)
        doc_ids = [HH._chat_template_ids(tokenizer, [m], max_length=hargs.max_doc_length) for m in history]
        doc_lens = [len(ids) for ids in doc_ids]
        typed_by_doc: Dict[int, List[List[int]]] = {}
        fallback_docs = 0
        for d, (message, ids) in enumerate(zip(history, doc_ids)):
            spans, used_fallback = _typed_spans_for_doc(tokenizer, message, ids)
            fallback_docs += int(used_fallback)
            if spans:
                typed_by_doc[d] = spans
        random_by_doc = _random_spans_for_qid(qid, typed_by_doc, doc_lens)
        typed_tokens = sum(e - s for spans in typed_by_doc.values() for s, e in spans)
        random_tokens = sum(e - s for spans in random_by_doc.values() for s, e in spans)
        total_typed += typed_tokens
        total_random += random_tokens
        total_fallback_docs += fallback_docs
        per_qid[qid] = {
            "session_id": example.qid.rsplit(":", 1)[0],
            "n_docs": len(doc_ids),
            "doc_lens": doc_lens,
            "typed_tokens": typed_tokens,
            "random_tokens": random_tokens,
            "fallback_docs": fallback_docs,
            "typed": {str(d): spans for d, spans in typed_by_doc.items()},
            "random": {str(d): spans for d, spans in random_by_doc.items()},
        }
        if (n + 1) % 50 == 0:
            logger.info("[%d/%d] typed=%d random=%d", n + 1, len(qids), total_typed, total_random)

    delta = abs(total_typed - total_random) / max(total_typed, 1)
    out = {
        "description": "R4 task D anchor spans. typed = control tokens (prereg definition); random = equal length-multiset spans at non-control positions (seed 20260815). Indices into truncated per-doc ids (max_doc_length=768).",
        "seed": SEED,
        "rule_version": "r4_anchor_v1",
        "qid_source": args.qid_file,
        "selection_skips": selection_skips,
        "budget": {
            "typed_tokens_total": total_typed,
            "random_tokens_total": total_random,
            "abs_delta_frac": round(delta, 6),
            "gate": "<= 0.02",
            "gate_passed": delta <= 0.02,
        },
        "fallback_docs_total": total_fallback_docs,
        "n_qids": len(qids),
        "per_qid": per_qid,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s: typed=%d random=%d delta=%.4f fallback_docs=%d",
        out_path, total_typed, total_random, delta, total_fallback_docs,
    )


if __name__ == "__main__":
    main()
