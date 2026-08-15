"""R4 task B: failure taxonomy over all arms (per-checkpoint, never pooled).

Mutually exclusive cascade per row (tool-target rows only; non-tool-target
rows are reported separately as an overtrigger dimension, report-only):

  NO_CALL          predicate ("<tool_call>" in text) or ("Action:" in text) missed
  PROTOCOL_BROKEN  predicate hit but no parseable tool-call JSON
  WRONG_TOOL       parseable but name != target
                   (subtype: in_pool / out_of_pool hallucination, by word-
                   boundary match of the predicted name in the pool text)
  WRONG_ARGS       name correct, args mismatch (key-set exact or value match
                   < 1.0), when the reference args are parseable
  OTHER / CORRECT  name correct + args ok -> CORRECT; name correct but
                   reference or prediction args unparseable -> OTHER

Layers:
  paired76 : --rows_a (c2kv) + --rows_b (full) — discordant "excess failure"
             cell (c2kv wrong & full right) control-class share; pool text and
             chunk-boundary annotation from the 96-doc tooldef source.
  plain32  : --rows_a only (task-D plain arm) — control-class share of all
             failures; pool = example.tools; boundary dimension UNANNOTATABLE
             (compressed content is history, not the tool schema).

Boundary dimension (report-only, 76k layer): offline recompute of the
512-token chunk layout over doc_ids; whether the target tool schema block
crosses a chunk boundary; error rate per group.

Usage (NPU server, repo root):
  python agent/r4_error_taxonomy.py --layer paired76 --rows_a c2kv.jsonl \
      --rows_b full.jsonl --out results/r4/taxonomy_76k.json
  python agent/r4_error_taxonomy.py --layer plain32 --rows_a d_plain.jsonl \
      --out results/r4/taxonomy_32k.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))

logger = logging.getLogger("r4_error_taxonomy")

TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
CHUNK_76K = 512  # r3 T-E arm chunk length


def _has_call(text: str) -> bool:
    return ("<tool_call>" in (text or "")) or ("Action:" in (text or ""))


def _parse_call(text: str) -> Optional[Dict[str, Any]]:
    m = TOOL_CALL_JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _call_name(call: Dict[str, Any]) -> Optional[str]:
    for key in ("name", "tool_name", "function_name"):
        if isinstance(call.get(key), str):
            return call[key]
    fn = call.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def _call_args(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    args = call.get("arguments") or call.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else None


def _norm_value(v: Any) -> Any:
    return " ".join(v.split()) if isinstance(v, str) else v


def _classify(row: Dict[str, Any], pool_text: Optional[str]) -> Dict[str, Any]:
    text = row.get("prediction", row.get("text", ""))
    target_name = row.get("target_tool_name")
    if target_name is None:
        return {"category": "NON_TOOL_TARGET", "called": _has_call(text)}
    if not _has_call(text):
        return {"category": "NO_CALL"}
    call = _parse_call(text)
    if call is None or _call_name(call) is None:
        return {"category": "PROTOCOL_BROKEN"}
    pred_name = _call_name(call)
    if pred_name != target_name:
        in_pool = None
        if pool_text is not None:
            in_pool = bool(re.search(re.escape(pred_name) + r"\b", pool_text))
        return {"category": "WRONG_TOOL", "pred_name": pred_name, "in_pool": in_pool}
    pred_args = _call_args(call)
    tgt_call = _parse_call(row.get("target", ""))
    tgt_args = _call_args(tgt_call) if tgt_call else None
    if pred_args is None or tgt_args is None:
        return {"category": "OTHER", "note": "args unparseable"}
    if set(pred_args) != set(tgt_args):
        return {"category": "WRONG_ARGS", "note": "key-set mismatch"}
    match = sum(1 for k in tgt_args if _norm_value(pred_args.get(k)) == _norm_value(tgt_args[k]))
    if match < len(tgt_args):
        return {"category": "WRONG_ARGS", "value_match_rate": round(match / max(len(tgt_args), 1), 4)}
    return {"category": "CORRECT"}


def _load_rows(path: str) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            rows[row["qid"]] = row
    return rows


def _tooldef_pools() -> Dict[str, Any]:
    """qid -> (tool_definition text, doc_ids) under the r3 S1 config."""
    import eval_agent_tool_definition_c2kv as H
    from r3_bigpool_rerun import S1_DATA_KW
    from train_agent_tool_definition_c2kv import AgentLLMTracesSource, AgentToolDefinitionDataArgs

    tokenizer = H.AutoTokenizer.from_pretrained(
        "./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250",
        trust_remote_code=True, local_files_only=True, padding_side="right",
    )
    source = AgentLLMTracesSource(AgentToolDefinitionDataArgs(
        dataset_path="./datasets/agent-llm-traces",
        split_manifest_file="./configs/agent_tooldef_split_manifests.json",
        **S1_DATA_KW,
    ))
    pools: Dict[str, Any] = {}
    sess_cache: Dict[str, Any] = {}
    for example in source.iter_examples("eval"):
        sid = example.session_id
        if sid not in sess_cache:
            sess_cache[sid] = (example.tool_definition, H._tool_doc_ids(tokenizer, example.tool_definition))
        pools[example.qid] = sess_cache[sid]
    return pools


def _history_pools() -> Dict[str, str]:
    """qid -> concatenated tool names text (PR#1 history regime)."""
    import eval_agent_history_c2kv as HH
    from r4_anchor_spans import _pr1_args

    hargs = _pr1_args()
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    pools: Dict[str, str] = {}
    for ex in examples:
        names = []
        for tool in getattr(ex, "tools", None) or []:
            if isinstance(tool, dict):
                nm = tool.get("name") or (tool.get("function") or {}).get("name")
                if nm:
                    names.append(str(nm))
        pools[ex.qid] = "\n".join(names)
    return pools


def _boundary_groups_76k(pools: Dict[str, Any], qids: List[str], rows: Dict[str, Any]) -> Dict[str, Any]:
    """Whether the target tool schema block crosses a 512-token chunk boundary.

    Heuristic (documented): the schema block is approximated as the target
    tool NAME token position +/- 200 tokens; the name token position is
    computed by tokenizing the pool text prefix up to the name match
    (wrapper offset corrected against the assembled doc_ids length).
    """
    import eval_agent_tool_definition_c2kv as H

    tok = H.AutoTokenizer.from_pretrained(
        "./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250",
        trust_remote_code=True, local_files_only=True, padding_side="right",
    )
    groups: Dict[str, List[bool]] = {"crosses": [], "within": [], "UNANNOTATABLE": []}
    for q in qids:
        row = rows[q]
        target = row.get("target_tool_name")
        if target is None or q not in pools:
            groups["UNANNOTATABLE"].append(False)
            continue
        text, doc_ids = pools[q]
        m = re.search(re.escape(target) + r"\b", text)
        if not m:
            groups["UNANNOTATABLE"].append(False)
            continue
        raw_pos = len(tok(text[: m.start()], add_special_tokens=False)["input_ids"])
        wrapper = len(doc_ids) - len(tok(text, add_special_tokens=False)["input_ids"])
        name_pos = raw_pos + wrapper
        block_lo, block_hi = name_pos - 200, name_pos + 200
        crosses = any(block_lo < b < block_hi for b in range(CHUNK_76K, len(doc_ids), CHUNK_76K))
        failed = not bool(row.get("tool_name_match"))
        groups["crosses" if crosses else "within"].append(failed)
    return {
        "heuristic": "target name token position +/- 200 tokens; chunk=512 over doc_ids",
        **{
            k: {"n": len(v), "error_rate": round(sum(v) / len(v), 4) if v else None}
            for k, v in groups.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", choices=["paired76", "plain32"], required=True)
    p.add_argument("--rows_a", required=True, help="c2kv-side (paired76) or plain arm (plain32)")
    p.add_argument("--rows_b", help="full arm (paired76 only)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    rows_a = _load_rows(args.rows_a)
    if args.layer == "plain32":
        pools = _history_pools()
    else:
        pools_td = _tooldef_pools()
        pools = {q: v[0] for q, v in pools_td.items()}
    logger.info("pools rebuilt for %d qids", len(pools))

    cls_a: Dict[str, Any] = {}
    counts = Counter()
    for q, row in rows_a.items():
        c = _classify(row, pools.get(q))
        cls_a[q] = c
        counts[c["category"]] += 1
    tool_rows = [q for q in rows_a if cls_a[q]["category"] != "NON_TOOL_TARGET"]
    failures = [q for q in tool_rows if cls_a[q]["category"] != "CORRECT"]
    control = sum(counts[k] for k in ("PROTOCOL_BROKEN", "WRONG_TOOL", "WRONG_ARGS"))
    report: Dict[str, Any] = {
        "layer": args.layer,
        "rows_a": args.rows_a,
        "n_rows": len(rows_a),
        "n_tool_target": len(tool_rows),
        "n_failures": len(failures),
        "counts": dict(counts),
        "control_class_share_of_failures": round(control / len(failures), 4) if failures else None,
        "wrong_tool_subtypes": {
            "in_pool": sum(1 for q in rows_a if cls_a[q].get("category") == "WRONG_TOOL" and cls_a[q].get("in_pool")),
            "out_of_pool": sum(1 for q in rows_a if cls_a[q].get("category") == "WRONG_TOOL" and cls_a[q].get("in_pool") is False),
        },
        "overtrigger_non_tool_targets": {
            "n": sum(1 for q in rows_a if cls_a[q]["category"] == "NON_TOOL_TARGET"),
            "called": sum(1 for q in rows_a if cls_a[q]["category"] == "NON_TOOL_TARGET" and cls_a[q].get("called")),
        },
        "per_qid": cls_a,
    }

    if args.layer == "paired76":
        if not args.rows_b:
            raise SystemExit("paired76 needs --rows_b")
        rows_b = _load_rows(args.rows_b)
        discordant = [
            q for q in rows_a
            if q in rows_b and cls_a[q]["category"] != "CORRECT"
            and _classify(rows_b[q], pools.get(q))["category"] == "CORRECT"
        ]
        disc_counts = Counter(cls_a[q]["category"] for q in discordant)
        disc_control = sum(disc_counts[k] for k in ("PROTOCOL_BROKEN", "WRONG_TOOL", "WRONG_ARGS"))
        report["paired"] = {
            "rows_b": args.rows_b,
            "n_paired": sum(1 for q in rows_a if q in rows_b),
            "excess_failure_cell": len(discordant),
            "excess_cell_counts": dict(disc_counts),
            "excess_control_share": round(disc_control / len(discordant), 4) if discordant else None,
        }
        report["boundary_76k"] = _boundary_groups_76k(pools_td, list(rows_a), rows_a)
    else:
        report["boundary_32k"] = "UNANNOTATABLE (compressed content is history, not the tool schema)"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("counts=%s control_share=%s -> %s", dict(counts), report["control_class_share_of_failures"], out)


if __name__ == "__main__":
    main()
