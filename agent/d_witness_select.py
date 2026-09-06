"""Witness-table builder (prereg v2.2 / v2.7) — CPU-only, no model forward.

Rebuilds each frozen qid's doc ids EXACTLY as the arms see them
(``HH._build_history_chunks`` + ``_chat_template_ids`` at
``max_doc_length``), decodes the grid rows (the text the model actually
saw — post-truncation, post-rendering, incl. the assistant tool_calls
dialect), runs the frozen witness-IDF selection, and freezes
``configs/bdf_pilot/d_witness_r2.json`` BEFORE any analysis reads a number.

Entry shape per qid (consumed by d_contract_driver / d2_short_erratum):
  n_docs, doc_lengths, doc_text_sha256, doc_text_chars,
  tool_name, arg_leaf_values, df, score, k_witness, k_median,
  target_doc_values  [[value, df], ...] — values occurring in doc k_witness,
  EXCLUDING the tool name (the D2 leak boundary, prereg v2.6), df-ascending.

Usage (c2kv env, CPU is enough):
  python agent/d_witness_select.py --output configs/bdf_pilot/d_witness_r2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import eval_agent_history_c2kv as HH
from d_strict_metric import TOOL_CALL_BLOCK_RE
from d_witness_core import occurs, select_k_star, target_values, witness_scores

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest_r2.json")
    parser.add_argument("--model", default="/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint")
    parser.add_argument("--base_model", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2")
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--output", default="./configs/bdf_pilot/d_witness_r2.json")
    return parser.parse_args(argv)


def _harness_args(args):
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
        "--history_selection", "tail",
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


def _parse_target(text: str):
    """(tool_name, args_object) from the target's first parseable block."""
    for block in TOOL_CALL_BLOCK_RE.findall(text or ""):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
            name = (
                obj.get("name") or obj.get("tool_name") or obj.get("function_name")
                or function.get("name")
            )
            args = obj.get("arguments", obj.get("parameters"))
            if args is None:
                args = function.get("arguments", {}) if function else {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return (str(name) if name else None), (args if isinstance(args, dict) else {})
    return None, {}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qids = list(manifest["cw_qids"])

    hargs = _harness_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} manifest qids not in eval split: {missing[:3]}")

    entries = {}
    n_none = 0
    for i, qid in enumerate(qids):
        example = by_qid[qid]
        context_input_ids, _, _, history, skip = HH._build_history_chunks(tokenizer, example, hargs)
        if context_input_ids is None:
            raise SystemExit(f"FATAL: qid {qid} skipped by harness: {skip}")
        doc_ids = [
            HH._chat_template_ids(tokenizer, [m], max_length=args.max_doc_length)
            for m in history
        ]
        # decoded grid rows — the exact text the compression forward sees
        texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in doc_ids]
        tool_name, target_args = _parse_target(example.answer)
        values = target_values(tool_name, target_args)
        df, score = witness_scores(texts, values)
        k_witness = select_k_star(texts, values)
        n_docs = len(doc_ids)
        k_median = (n_docs - 1) // 2

        # D2 erratum source: literal values occurring in the target doc,
        # tool name EXCLUDED (leak boundary), df ascending then first-seen
        target_doc_values = []
        if k_witness is not None:
            seen = []
            for v in values:
                if v == tool_name:
                    continue
                if occurs(v, texts[k_witness]) and v not in seen:
                    seen.append(v)
                    target_doc_values.append([v, df[v]])
            target_doc_values.sort(key=lambda item: (item[1], values.index(item[0])))

        if k_witness is None:
            n_none += 1
        entries[qid] = {
            "n_docs": n_docs,
            "doc_lengths": [len(ids) for ids in doc_ids],
            "doc_text_sha256": [
                hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
            ],
            "doc_text_chars": [len(t) for t in texts],
            "tool_name": tool_name,
            "arg_leaf_values": [v for v in values if v != tool_name],
            "df": df,
            "score": score,
            "k_witness": k_witness,
            "k_median": k_median,
            "target_doc_values": target_doc_values,
        }
        if (i + 1) % 20 == 0:
            logger.info("[%d/%d] processed", i + 1, len(qids))

    out = {
        "description": "Frozen witness table (prereg v2.2/v2.7): witness-IDF k* "
                       "selection over decoded grid rows. Freeze BEFORE any "
                       "analysis; amend the prereg, not this file.",
        "manifest": str(args.manifest),
        "algorithm": "d_witness_core.py (prereg v2 addendum, frozen verbatim)",
        "max_doc_length": args.max_doc_length,
        "max_doc_num": args.max_doc_num,
        "n_qids": len(qids),
        "n_k_none": n_none,
        "entries": entries,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    n_docs_hist = {}
    for e in entries.values():
        n_docs_hist[e["n_docs"]] = n_docs_hist.get(e["n_docs"], 0) + 1
    total_gens = sum(e["n_docs"] for e in entries.values())
    print(json.dumps({
        "output": str(out_path),
        "n_qids": len(entries),
        "n_k_none": n_none,
        "n_docs_hist": dict(sorted(n_docs_hist.items())),
        "sum_n_docs(sweep_generations)": total_gens,
        "witness_eq_median": sum(1 for e in entries.values()
                                 if e["k_witness"] is not None and e["k_witness"] == e["k_median"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
