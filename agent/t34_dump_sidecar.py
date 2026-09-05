# -*- coding: utf-8 -*-
"""t34 §4.6 — server-side dump of the per-doc PLAINTEXT sidecar.

Nothing in this repo holds the text the model actually saw: the decoded grid
rows exist only inside the harness.  Every grounding feature in
``agent/triggers.py`` (SAAG AVEM/QSLO/VHR, SIEVE's UNRESOLVED rung, Tracy's
``grounding_visible``, the ``must_read_before_write`` identifier sets) reads
this dump and nothing else.

Decoding is done EXACTLY the way ``agent/d_witness_select.py`` decodes grid
rows — same harness args, same ``_build_history_chunks`` ->
``_chat_template_ids(max_length=max_doc_length)`` -> ``tokenizer.decode(...,
skip_special_tokens=False)`` chain — so the per-doc sha256 in the frozen
witness table matches byte for byte.  ``--check`` proves that instead of
asserting it.

FIXED SCHEMA (one JSON object per line, these keys and no others)::

    {qid, session_id, docs, query, tools, system_prompt,
     doc_lengths, dropped_docs, kept_history_tokens}

``docs``            decoded grid-row text per history block, in block order
``query``           decoded current-message block (the uncompressed query)
``tools``           the tool schema dicts as handed to the model
``doc_lengths``     token length of each grid row
``dropped_docs``    block indices removed by the tail window (indices into the
                    post-split history list, which is what the window selects
                    over) — the S8 column, free and available before generation
``kept_history_tokens``  sum of the kept grid rows' token lengths

``--with_dropped_text`` adds ONE extra key, ``dropped_doc_texts`` (the decoded
text of the dropped blocks).  It is off by default because it is outside the
fixed schema; it is the only way to compute the
``grounded_in_visible_history`` vs ``grounded_in_dropped_docs`` delta the
digest asks for, so turn it on deliberately and say so in the prereg.

INDEX SPACES (the trap this dump exists to make explicit)
---------------------------------------------------------
``docs`` holds ONLY THE KEPT blocks, in order: every entry is text the model
saw.  ``dropped_docs`` holds indices into the POST-SPLIT HISTORY list, which is
a DIFFERENT index space — it may never be used to slice or filter ``docs``
(``[d for i, d in enumerate(docs) if i not in dropped_docs]`` is a bug: it
silently drops kept blocks).  The bridge between the two spaces is the second
extension key, ``kept_history_indices``: ``kept_history_indices[i]`` is the
post-split index of ``docs[i]``, so ``dict(zip(kept_history_indices,
range(len(docs))))`` maps a history index to its ``docs`` slot and
``set(kept_history_indices) & set(dropped_docs) == set()`` by construction.  It
is always emitted when the harness's own kept list is co-indexed with ``docs``,
and OMITTED (with a loud summary counter) when it is not — the consumer side is
``triggers.sidecar_history_map``, which returns None on an omitted key rather
than guessing an alignment.

RUNBOOK
-------
[SERVER, c2kv env, CPU is enough — no GPU, no forward pass]::

    python agent/t34_dump_sidecar.py --arm c2kv --out results/t34/sidecar_c2kv.jsonl
    python agent/t34_dump_sidecar.py --arm full --out results/t34/sidecar_full.jsonl
    python agent/t34_dump_sidecar.py --arm c2kv --out results/t34/sidecar_c2kv.jsonl --check

The doc grid is IDENTICAL for both arms by construction (verified on the
frozen battery: 900/900 rows agree on ``doc_chunks`` and ``doc_tokens``); the
arms differ in what they do with the grid, not in what the grid is.  The two
files are still dumped separately so that ``triggers.py build-features --arm
full`` (the S0 twin) reads a file of its own and can never be handed the
compressed arm's text by accident.

[LOCAL] copy both files back, then run ``agent/triggers.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "per-doc plaintext sidecar",
        "paper": "n/a (infrastructure for 2512.06716 / 2607.18245 / 2607.07405 "
                 "/ 2608.02464 / Tracy's grounding)",
        "what": "The dump carries the decoded grid rows only; it does NOT carry "
                "the dropped blocks' text unless --with_dropped_text is passed.",
        "why": "The fixed schema in the task brief lists nine keys; the dropped "
               "text is an explicit opt-in extension, because whether the "
               "dropped-doc plaintext is retrievable post hoc is exactly the "
               "open question card 2607.18245 flags (open question 2).",
    },
    {
        "method": "dropped_docs index base",
        "paper": "n/a",
        "what": "``dropped_docs`` indexes the POST-SPLIT history list (the list "
                "the tail window selects over), not the raw dataset message "
                "list, because oversized messages are split before selection.",
        "why": "The window's own denominator is the post-split list; indexing "
               "the raw list would misalign with ``docs``/``doc_lengths``.",
    },
    {
        "method": "kept_history_indices bridge",
        "paper": "n/a",
        "what": "A second EXTENSION key (never part of SCHEMA_KEYS) gives the "
                "post-split history index of every ``docs[i]``, so a consumer "
                "can align the two index spaces instead of mistaking "
                "``dropped_docs`` for positions in ``docs``.  It is omitted, "
                "and counted in the run summary, whenever the replicated "
                "selection is not co-indexed with the harness's own doc grid.",
        "why": "``docs`` holds only the KEPT blocks, so filtering it by "
               "``dropped_docs`` silently deletes kept text; the bridge makes "
               "the alignment checkable rather than assumed, and validate_row "
               "refuses a bridge that disagrees with ``docs``/``dropped_docs``.",
    },
]

#: The fixed schema, in order.  ``validate_row`` refuses anything else.
SCHEMA_KEYS: Tuple[str, ...] = (
    "qid", "session_id", "docs", "query", "tools", "system_prompt",
    "doc_lengths", "dropped_docs", "kept_history_tokens",
)
EXTENSION_KEYS: Tuple[str, ...] = ("dropped_doc_texts", "kept_history_indices")


def kept_and_dropped_indices(n_messages: int, max_doc_num: int,
                             policy: str) -> Tuple[List[int], List[int]]:
    """Replicate ``train_data_multiturn._select_history`` at the index level.

    ``head``  -> keep ``[0 .. max_doc_num-1]``.
    ``tail``  -> keep ``[0] + the last (max_doc_num - 1)`` (the harness keeps
    the first message deliberately); ``max_doc_num <= 1`` keeps the last one.
    Nothing is dropped when the history already fits.
    """
    idx = list(range(int(n_messages)))
    if n_messages <= max_doc_num:
        return idx, []
    if policy == "head":
        kept = idx[:max_doc_num]
    elif policy == "tail":
        kept = idx[-max_doc_num:] if max_doc_num <= 1 else [idx[0]] + idx[-(max_doc_num - 1):]
    else:
        raise ValueError(f"Unsupported history selection policy: {policy}")
    kept_set = set(kept)
    return kept, [i for i in idx if i not in kept_set]


def sidecar_row(*, qid: str, session_id: str, docs: Sequence[str], query: str,
                tools: Sequence[Dict[str, Any]], system_prompt: str,
                doc_lengths: Sequence[int], dropped_docs: Sequence[int],
                kept_history_tokens: int,
                dropped_doc_texts: Optional[Sequence[str]] = None,
                kept_history_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Build one line of the sidecar in the fixed schema (plus the two
    extension keys).

    ``kept_history_indices[i]`` is the POST-SPLIT history index of ``docs[i]``
    (see the INDEX SPACES section in the module docstring).  ``validate_row``
    refuses it when it is not co-indexed with ``docs`` or when it intersects
    ``dropped_docs``: a wrong bridge is worse than none.
    """
    row: Dict[str, Any] = {
        "qid": qid,
        "session_id": session_id,
        "docs": list(docs),
        "query": query,
        "tools": list(tools),
        "system_prompt": system_prompt,
        "doc_lengths": [int(x) for x in doc_lengths],
        "dropped_docs": [int(x) for x in dropped_docs],
        "kept_history_tokens": int(kept_history_tokens),
    }
    if dropped_doc_texts is not None:
        row["dropped_doc_texts"] = list(dropped_doc_texts)
    if kept_history_indices is not None:
        row["kept_history_indices"] = [int(x) for x in kept_history_indices]
    validate_row(row)
    return row


def validate_row(row: Dict[str, Any]) -> None:
    """Refuse any key outside the fixed schema (+ declared extensions), and any
    row whose ``docs`` and ``doc_lengths`` disagree."""
    keys = set(row)
    allowed = set(SCHEMA_KEYS) | set(EXTENSION_KEYS)
    missing = set(SCHEMA_KEYS) - keys
    extra = keys - allowed
    if missing or extra:
        raise ValueError(f"sidecar row schema violation: missing={sorted(missing)} "
                         f"unexpected={sorted(extra)}")
    if len(row["docs"]) != len(row["doc_lengths"]):
        raise ValueError("docs and doc_lengths must have the same length")
    kept = row.get("kept_history_indices")
    if kept is not None:
        if len(kept) != len(row["docs"]):
            raise ValueError(
                f"kept_history_indices has {len(kept)} entries but docs has "
                f"{len(row['docs'])}: the two index spaces are not co-indexed")
        if set(kept) & set(row["dropped_docs"]):
            raise ValueError(
                "kept_history_indices intersects dropped_docs: a kept block "
                "cannot also be a dropped post-split index")
    dropped_texts = row.get("dropped_doc_texts")
    if dropped_texts is not None and len(dropped_texts) != len(row["dropped_docs"]):
        raise ValueError(
            f"dropped_doc_texts has {len(dropped_texts)} entries but "
            f"dropped_docs has {len(row['dropped_docs'])}")


def write_sidecar(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            validate_row(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("c2kv", "full"), default="c2kv",
                        help="names the output only; the doc grid is identical "
                             "across arms by construction")
    parser.add_argument("--root", default=".", help="worktree root")
    parser.add_argument("--out", default=None,
                        help="default results/t34/sidecar_<arm>.jsonl")
    parser.add_argument("--with_dropped_text", action="store_true",
                        help="also dump dropped_doc_texts (outside the fixed "
                             "schema). WITHOUT it triggers.py emits "
                             "grounding_dropped_delta = null on every step that "
                             "dropped a block and flags it as a missing input "
                             "(grounding_dropped_side_available = 0); the digest "
                             "asks for that delta, so turn this on deliberately "
                             "and record it in the prereg")
    parser.add_argument("--check", action="store_true",
                        help="re-read the dump and run "
                             "t34_common.check_docs_against_witness against the "
                             "frozen witness table's per-doc sha256")
    parser.add_argument("--limit", type=int, default=0, help="debug: first N qids only")
    # harness plumbing — same defaults as agent/d_witness_select.py
    parser.add_argument("--model", default="/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint")
    parser.add_argument("--base_model", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2")
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--history_selection", default="tail", choices=("tail", "head"))
    return parser.parse_args(list(argv) if argv is not None else None)


def _harness_args(args: argparse.Namespace):
    """Identical argv to agent/d_witness_select.py::_harness_args (frozen r2
    recipe) so the decoded grid rows are byte-identical to the witness table's."""
    import eval_agent_history_c2kv as HH  # noqa: PLC0415  (imports torch)

    argv = [
        "prog",
        "--model", args.model,
        "--base_model", args.base_model,
        "--tokenizer", args.tokenizer,
        "--dataset_path", args.dataset_path,
        "--split", "eval",
        "--include_tools", "True",
        "--require_tool_call", "False",
        "--max_examples", "0",
        "--max_samples_per_session", "0",
        "--eval_ratio", "0.1",
        "--split_seed", "42",
        "--split_manifest_name", "subset_disjoint",
        "--max_doc_length", str(args.max_doc_length),
        "--max_doc_num", str(args.max_doc_num),
        "--min_doc_num", "1",
        "--max_history_tokens", "12288",
        "--max_system_length", "4096",
        "--max_prompt_tokens", "1536",
        "--max_baseline_input_tokens", "16000",
        "--max_new_tokens", "128",
        "--history_selection", args.history_selection,
        "--system_attn_impl", "eager",
        "--gist_attn_impl", "eager",
        "--generate_attn_impl", "eager",
        "--device_type", "cpu",
        "--override_ratio", "8",
        "--hybrid_top_k", "3",
        "--hybrid_layout", "gist_first",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def _split_history(tokenizer, example, hargs) -> List[Dict[str, Any]]:
    """The post-split history list the tail window selects over.

    Mirrors ``train_data_multiturn._fit_reused_history`` up to (not including)
    ``_select_history``, so the dropped indices can be named.
    """
    import eval_agent_history_c2kv as HH  # noqa: PLC0415
    from train.train_data_multiturn import (  # noqa: PLC0415
        _message_token_length, _split_message_to_fit,
    )

    raw = [HH._normal_chat_message(m) for m in example.history_messages if m.get("content")]
    if getattr(hargs, "split_oversized_history_docs", True):
        out: List[Dict[str, Any]] = []
        for message in raw:
            out.extend(_split_message_to_fit(tokenizer, message, hargs.max_doc_length))
        return out
    return [m for m in raw
            if _message_token_length(tokenizer, m) <= hargs.max_doc_length]


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / f"results/t34/sidecar_{args.arm}.jsonl"

    from t34_common import FrozenAssets, check_docs_against_witness, load_decoded_docs  # noqa: PLC0415

    frame = FrozenAssets(root).load()
    want = list(frame.c2kv_by_qid.keys())
    if args.limit:
        want = want[: args.limit]

    import eval_agent_history_c2kv as HH  # noqa: PLC0415
    hargs = _harness_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    missing = [q for q in want if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} battery qids not in eval split: {missing[:3]}")

    rows: List[Dict[str, Any]] = []
    n_kept_mismatch = 0
    for i, qid in enumerate(want):
        example = by_qid[qid]
        context_input_ids, total_tokens, _n_docs, history, skip = HH._build_history_chunks(
            tokenizer, example, hargs)
        if context_input_ids is None:
            raise SystemExit(f"FATAL: qid {qid} skipped by harness: {skip}")
        doc_ids = [HH._chat_template_ids(tokenizer, [m], max_length=args.max_doc_length)
                   for m in history]
        docs = [tokenizer.decode(ids, skip_special_tokens=False) for ids in doc_ids]

        split = _split_history(tokenizer, example, hargs)
        kept_idx, dropped_idx = kept_and_dropped_indices(
            len(split), hargs.max_doc_num, hargs.history_selection)
        # the bridge between the two index spaces: kept_idx[i] is the post-split
        # index of docs[i].  Emitted ONLY when the replicated kept list is
        # co-indexed with the harness's own doc grid; otherwise the key is
        # omitted (counted in the summary) rather than emitted wrong.
        kept_for_row = list(kept_idx) if len(kept_idx) == len(docs) else None
        if kept_for_row is None:
            n_kept_mismatch += 1
        dropped_texts = None
        if args.with_dropped_text:
            dropped_texts = [
                tokenizer.decode(
                    HH._chat_template_ids(tokenizer, [split[j]], max_length=args.max_doc_length),
                    skip_special_tokens=False)
                for j in dropped_idx
            ]
        current = HH._current_messages(example)
        query = tokenizer.decode(
            HH._chat_template_ids(tokenizer, current), skip_special_tokens=False)

        rows.append(sidecar_row(
            qid=qid,
            session_id=frame.c2kv_by_qid[qid].get("session_id"),
            docs=docs,
            query=query,
            tools=list(example.tools or []),
            system_prompt=example.system_prompt,
            doc_lengths=[len(ids) for ids in doc_ids],
            dropped_docs=dropped_idx,
            kept_history_tokens=int(total_tokens),
            dropped_doc_texts=dropped_texts,
            kept_history_indices=kept_for_row,
        ))
        if (i + 1) % 100 == 0:
            logger.info("[%d/%d] dumped", i + 1, len(want))

    n = write_sidecar(out_path, rows)
    summary: Dict[str, Any] = {
        "out": str(out_path), "arm": args.arm, "n_rows": n,
        "n_with_dropped": sum(1 for r in rows if r["dropped_docs"]),
        "with_dropped_text": bool(args.with_dropped_text),
        "n_with_kept_history_indices": sum(1 for r in rows
                                           if "kept_history_indices" in r),
        "n_kept_history_index_mismatch": n_kept_mismatch,
        "kept_history_indices_note": (
            "kept_history_indices[i] is the POST-SPLIT history index of docs[i]; "
            "dropped_docs lives in that same post-split space and NEVER indexes "
            "docs. A non-zero n_kept_history_index_mismatch means the replicated "
            "selection disagreed with the harness's doc grid on that row and the "
            "bridge was omitted rather than guessed"),
    }
    if n_kept_mismatch:
        logger.warning("kept_history_indices omitted on %d/%d rows (replicated "
                       "selection disagreed with the harness doc grid)",
                       n_kept_mismatch, len(rows))
    if args.check:
        docs = load_decoded_docs(out_path)
        summary["check_docs_against_witness"] = check_docs_against_witness(docs, frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
