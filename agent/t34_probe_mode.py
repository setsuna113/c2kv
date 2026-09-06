# -*- coding: utf-8 -*-
"""t34 §4.12 — battery probe-mode driver (SERVER: needs torch + the NPU model).

Re-runs the 161 frozen trigger rows through the ordinary battery harness with
ONLY the tail overridden: the compressed prefix (system prompt + gist KV) is
built exactly as in the frozen run and is never touched.  Two modes:

* ``--mode vista``      executes a probe plan from ``t34_selfreport plan-probes``
                        (P1/P2/P3/P4, one request per probe, greedy), optionally
                        with the ledger line inserted at the TAIL of the
                        compressed history (2606.30005 card pitfall (c): head
                        placement would re-prefill the whole prefix).
* ``--mode selfcheck``  appends the compressed arm's own emitted call plus the
                        SPEC §8.2 item 6 yes/no + 0-1 question after the
                        original query, reusing the same prefix, 32 new tokens.

WHERE THE TAIL GOES.  ``eval_agent_history_c2kv._generate_with_prefix`` (:1149)
starts with ``current_messages = prefix.get("current_messages") or
_current_messages(example)`` and templates ONLY that list after the cache
(:1157-1163), positioning it at ``system_length + history_length``.  So writing
``prefix["current_messages"]`` is exactly "after the compressed history, before
the current query" — the insertion point the VISTA card demands — and it leaves
``_build_c2kv_prefix`` (:1568) and its gist cache untouched.

RUNBOOK (all of this runs on the NPU server)
--------------------------------------------
1. python agent/t34_probe_mode.py --mode vista --arm c2kv \
       --plan results/t34/probe_plan_minus_ledger.jsonl \
       --model_path <fixed_joint ckpt> --base_model <base> \
       --tokenizer_path <tok> --dataset_path <agent-llm-traces> \
       --battery_full results/bdf_pilot/d_r2/battery_full.jsonl \
       --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \
       --manifest configs/bdf_pilot/d_cw_manifest_r2.json \
       --out results/t34/probes_vista_minus_ledger_c2kv.jsonl
2. (S0 twin, label-side control) same command with --arm full and its own --out,
   against a plan built from the FULL arm's own sidecar.  This pass is
   MANDATORY, not optional: P4's "was anything condensed" truth is constant on
   the compressed arm, so its detection half is only scorable pooled over both
   arms (``t34_selfreport score-p4-pooled``).
3. python agent/t34_probe_mode.py --mode selfcheck --arm c2kv \
       --p4-result results/t34/vista_minus_ledger.json ...        # gate
       --out results/t34/probes_selfcheck_c2kv.jsonl
   The selfcheck mode REFUSES to start without a passing VISTA P4 result unless
   --force is given (digest §4.12: run P4 first, then decide).

Output rows are whitelisted: qid / session_id / probe / condition / arm / query /
ledger / generation / parsed / generate_sec / prompt_tokens / generated_tokens.
No target, no gold, no scoring column is written, so the probe file can never
carry a label into a feature frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import t34_selfreport as SR  # noqa: E402  (torch-free)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("t34_probe")

try:  # pragma: no cover - exercised only on the server
    import torch  # noqa: F401

    from eval_agent_history_c2kv import (
        _build_c2kv_prefix,
        _build_full_or_truncate_prefix,
        _clear_device_cache,
        _current_messages,
        _generate_with_prefix,
        _is_oom_error,
        _load_examples,
        _load_tokenizer,
        _resolve_model_checkpoint,
    )
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device

    IMPORT_ERROR: Optional[BaseException] = None
except ImportError as error:
    IMPORT_ERROR = error

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "probe-mode row loop",
        "paper": "2606.30005 §14 / SPEC §8.2 item 6",
        "what": "The probes are executed by re-running the frozen battery rows "
                "with only ``prefix['current_messages']`` replaced; no harness "
                "file is modified and the compressed prefix builder is called "
                "unchanged.",
        "why": "Hard rule 1 (own files only) and the requirement that the "
               "compressed prefix stay byte-identical to the frozen run.",
    },
    {
        "method": "probe generation length",
        "paper": "2606.30005 §14 (greedy) / SPEC §8.2 item 6 (max 32)",
        "what": "Probes use greedy decoding with 64 new tokens; the selfcheck "
                "arm uses greedy with 32, matching the SPEC's contract.  The "
                "frozen battery's 128-token cap does not apply to probe answers.",
        "why": "A size estimate or a yes/no answer needs a few tokens; the cap "
               "of the ORIGINAL run is a property of the frozen rows, not of a "
               "probe request.  Reported so the cost column is reproducible.",
    },
]

#: Whitelist of columns the probe output may carry.
OUTPUT_COLUMNS = ("qid", "session_id", "arm", "mode", "probe", "condition",
                  "query", "ledger", "generation", "parsed", "generate_sec",
                  "prompt_tokens", "generated_tokens")


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Harness args for a probe run — mirrors ``t33_hidden_topup.build_args``
    so the prefix construction is identical to the frozen battery."""
    ns = argparse.Namespace(
        max_doc_length=768, max_doc_num=16, min_doc_num=1,
        max_history_tokens=12288, max_system_length=4096,
        max_prompt_tokens=1536, max_baseline_input_tokens=16000,
        history_selection="tail", truncate_selection="tail",
        split_oversized_history_docs=True,
        system_attn_impl=cli.attn_impl, gist_attn_impl=cli.attn_impl,
        generate_attn_impl=cli.attn_impl, override_ratio=cli.ratio,
        dataset_path=cli.dataset_path, split="eval",
        eval_ratio=0.1, split_seed=42,
        split_manifest_file=None, split_manifest_name="subset_disjoint",
        max_samples_per_session=4, max_source_examples=None,
        require_tool_call=False, max_input_chars=None, max_answer_chars=None,
        include_tools=True, prefix_history_doc_num=None, prefix_history_exact=False,
        selection_filter="c2kv", sample_seed=None, max_examples=0,
        tokenizer=cli.tokenizer_path, model=cli.model_path, base_model=None,
        mode="c2kv", dtype="bf16", baseline_model_class="auto", untrained_c2kv=False,
        max_new_tokens=cli.max_new_tokens, do_sample=False, temperature=None, top_p=None,
        t33_ctx=None,
    )
    if cli.arm == "full":
        ns.base_model = cli.base_model
        ns.model = cli.base_model
        ns.mode = "full"
    return ns


def load_plan(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def selfcheck_plan(qids: Sequence[str], predictions: Dict[str, str]) -> List[Dict[str, Any]]:
    """One selfcheck request per trigger row (SPEC §8.2 item 6)."""
    return [{"qid": q, "session_id": q.rsplit(":", 1)[0], "probe": "selfcheck",
             "condition": "prefix_reuse", "query": SR.SELFCHECK_QUESTION,
             "ledger": None, "prediction": predictions.get(q, "")}
            for q in qids]


def plan_key(row: Dict[str, Any]) -> str:
    return f"{row['qid']}|{row.get('probe')}|{row.get('condition')}"


def resume_done(path: str) -> set:
    """Keys already present in an output file (idempotent re-runs)."""
    done: set = set()
    p = Path(path)
    if not p.exists():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(plan_key(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def parse_for_probe(probe: str, generation: str,
                    n_blocks: int = 0) -> Dict[str, Any]:
    """Parse a probe generation with the matching parser (pure; unit-tested).

    P2 returns one value PER sampled block (2606.30005 §4 scores individual
    block size), so it needs the plan row's block count.
    """
    if probe == "P1":
        return {"value": SR.parse_integer_answer(generation)}
    if probe == "P2":
        return {"values": SR.parse_integer_list_answer(generation, int(n_blocks))}
    if probe == "P3":
        return {"choice": SR.parse_choice_answer(generation)}
    if probe == "P4":
        return SR.parse_loss_answer(generation)
    if probe == "selfcheck":
        return SR.parse_selfcheck(generation)
    return {}


def probe_row_record(plan_row: Dict[str, Any], arm: str, mode: str,
                     metrics: Dict[str, Any], generation: str) -> Dict[str, Any]:
    """Build one whitelisted output row (no target / gold / scoring column)."""
    rec = {
        "qid": plan_row["qid"],
        "session_id": plan_row.get("session_id") or plan_row["qid"].rsplit(":", 1)[0],
        "arm": arm,
        "mode": mode,
        "probe": plan_row.get("probe"),
        "condition": plan_row.get("condition"),
        "query": plan_row.get("query"),
        "ledger": plan_row.get("ledger"),
        "generation": generation,
        "parsed": parse_for_probe(
            plan_row.get("probe"), generation,
            n_blocks=len(plan_row.get("blocks") or [])),
        "generate_sec": metrics.get("generate_sec"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": metrics.get("generated_tokens"),
    }
    if "truth" in plan_row:           # carried through for the scorer, never a label
        rec["truth"] = plan_row["truth"]
    extra = sorted(set(rec) - set(OUTPUT_COLUMNS) - {"truth"})
    if extra:
        raise ValueError(f"probe output row carries non-whitelisted columns: {extra}")
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("RUNBOOK")[0].strip())
    ap.add_argument("--mode", required=True, choices=["vista", "selfcheck"])
    ap.add_argument("--arm", required=True, choices=["full", "c2kv"])
    ap.add_argument("--plan", default=None,
                    help="probe plan jsonl (required for --mode vista)")
    ap.add_argument("--p4-result", default=None,
                    help="VISTA P4 score json; gates --mode selfcheck")
    ap.add_argument("--force", action="store_true",
                    help="bypass the P4 gate (record it in the prereg)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--battery_full", required=True)
    ap.add_argument("--battery_c2kv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=int, default=8)
    ap.add_argument("--attn_impl", default="eager")
    ap.add_argument("--device_type", default="npu")
    ap.add_argument("--max_new_tokens", type=int, default=0,
                    help="0 = 32 for selfcheck, 64 for the VISTA probes")
    ap.add_argument("--ledger_role", default="system", choices=["system", "user"])
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args(argv)

    if args.mode == "selfcheck":
        SR.require_p4_gate(args.p4_result, force=args.force)
    if args.max_new_tokens <= 0:
        args.max_new_tokens = 32 if args.mode == "selfcheck" else 64
    if IMPORT_ERROR is not None:
        print(f"needs torch/transformers (server): {IMPORT_ERROR}", file=sys.stderr)
        return 2

    from t33_labels import build_label_frame, join_arms, load_jsonl

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pairs = join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv))
    frame = build_label_frame(pairs, manifest)
    subset = [r["qid"] for r in frame if r["label_cw"] in (0, 1)]
    if args.max_rows:
        subset = subset[: args.max_rows]

    if args.mode == "vista":
        if not args.plan:
            print("--mode vista needs --plan", file=sys.stderr)
            return 2
        plan = [r for r in load_plan(args.plan) if r["qid"] in set(subset)]
    else:
        preds = {c["qid"]: (c.get("prediction") or "") for _, c in pairs}
        plan = selfcheck_plan(subset, preds)

    eval_args = build_args(args)
    device = _setup_device(args.device_type)
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    model = _load_model(eval_args, tokenizer, device)

    wanted = {r["qid"] for r in plan}
    eval_args.qid_allowlist = set(wanted)   # start-up cost: only the frozen rows
    examples = {e.qid: e for e in _load_examples(eval_args, tokenizer)[0] if e.qid in wanted}
    done = resume_done(args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    logger.info("mode=%s arm=%s plan=%d loaded=%d resume=%d",
                args.mode, args.arm, len(plan), len(examples), len(done))

    with open(args.out, "a", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(plan):
            key = plan_key(row)
            if key in done or row["qid"] not in examples:
                continue
            example = examples[row["qid"]]
            try:
                if args.arm == "full":
                    prefix, skip = _build_full_or_truncate_prefix(
                        model, tokenizer, example, eval_args, "full")
                else:
                    prefix, skip = _build_c2kv_prefix(model, tokenizer, example, eval_args)
                if prefix is None:
                    logger.warning("skip %s: %s", row["qid"], skip)
                    continue
                # ---- the ONLY override: the tail after the compressed prefix
                if args.mode == "selfcheck":
                    prefix["current_messages"] = SR.selfcheck_current_messages(
                        _current_messages(example), row.get("prediction") or "")
                else:
                    prefix["current_messages"] = SR.current_messages_for_probe(
                        row["query"], ledger_line=row.get("ledger"),
                        role=args.ledger_role)
                prefix["target_override"] = ""   # probe answers have no target
                metrics = _generate_with_prefix(model, tokenizer, example, prefix,
                                                eval_args, args.arm)
            except RuntimeError as error:
                if _is_oom_error(error):
                    logger.warning("oom at %s, continuing", row["qid"])
                    _clear_device_cache(device)
                    continue
                raise
            rec = probe_row_record(row, args.arm, args.mode, metrics,
                                   metrics.get("prediction") or "")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            logger.info("[%d/%d] %s %s gen_sec=%s", i + 1, len(plan), row["qid"],
                        row.get("probe"), rec["generate_sec"])
            _clear_device_cache(device)
    logger.info("probe-mode done -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
