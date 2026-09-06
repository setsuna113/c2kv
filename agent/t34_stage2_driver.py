#!/usr/bin/env python
"""t34 stage-2 table driver (digest S4.8 / S4.12 cascade; SERVER: torch + the NPU model).

WHY.  The cascade's p > 0 rungs (``agent/t34_cascade.py sweep --expensive``) need,
for EVERY row of the 161-row trigger frame (93 C->W and 68 C->C), the outcome of
the same single-block escalation the D-line used: one raw_keepG splice -- the
block's pre-RoPE K/V from the compression sidecar, rotated onto its absolute
start and appended behind the FULL gist grid -- followed by the ordinary
regeneration.  The frozen ``results/bdf_pilot/d_r2/d_corr.jsonl`` covers the 93
positives only, so its availability IS the label
(``t34_cascade.expensive_coverage_audit`` refuses that).  This driver produces
the table for both classes from ONE code version.

BLOCK CHOICE IS GOLD-FREE BY CONSTRUCTION.  k = (n_docs - 1) // 2 (``k_median``,
the frozen ``d_sham_plan.k_star_for`` definition), the same position heuristic
the frozen d_corr arm used.  Nothing here reads the witness table, k_star, the
flip table or any scoring column; ``t34_cascade.expensive_block_choice_audit``
verifies the emitted ``d_corr_doc_index`` against ``k_median`` on every row it
can check.

ARM IDENTITY.  The splice is ``d1_arms.ksweep_prefix_for_k`` -- the exact
primitive of the D-line k-sweep (``agent/d_ksweep_driver.py``, arm
``raw_keepG_sweep``), so a stage-2 row here at k equals the sweep row at
``d_ksweep_k == k`` up to bf16 nondeterminism.  Rows are the harness generation
rows (same schema as ``d_ksweep_r2.jsonl``) plus ``d_arm="raw_keepG_kmedian"``,
``d_mode="d_raw_keepG"``, ``d_ksweep_k`` and ``d_corr_doc_index`` == k.

RUNBOOK
  NPU   python agent/t34_stage2_driver.py --root . --max_qids 3 \
          --output_file results/t34/smoke/stage2_smoke.jsonl        # smoke first
  NPU   python agent/t34_stage2_driver.py --root . \
          --output_file results/t34/stage2_161.jsonl --timing_report results/t34/stage2_timing.json
        (161 splice+generate; the k-sweep measured ~11.4 s each on a 910B3, so
         about 35-40 min on one chip.  --resume True (default) continues a
         partial file.)
  here  PYTHONIOENCODING=utf-8 python agent/t34_cascade.py sweep --root . --rungs 6 \
          --expensive results/t34/stage2_161.jsonl --out results/t34/cascade_sweep.json

Everything except :func:`run` is pure and unit-tested on a torch-free box.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "python", _HERE.parent / "python" / "inference", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)

MODE = "d_raw_keepG"
ARM = "raw_keepG_kmedian"

DEVIATIONS: List[Dict[str, str]] = [
    {"what": "stage-2 = raw_keepG at k_median for BOTH classes, re-run for the 93 "
             "C->W rows too instead of reusing the frozen d_corr.jsonl",
     "why": "one code version for both classes; the frozen file predates the "
            "runner's harness fixes and covers one class only (availability = label)"},
    {"what": "the sidecar-splice primitive (ksweep_prefix_for_k) rather than the "
             "harness's sequential raw prefill used by the original d_corr arm",
     "why": "the D-line k-sweep -- the flip table every t34 locator is read against -- "
            "used the sidecar primitive; the stage-2 table must be the same arm as "
            "the table it is compared with"},
]


def k_median(n_docs: int) -> int:
    """Frozen gold-free block choice: (n_docs - 1) // 2 (d_sham_plan.k_star_for)."""
    if n_docs <= 0:
        raise ValueError("k_median needs n_docs >= 1")
    return (n_docs - 1) // 2


def plan_qids(frame_rows: Sequence[Dict[str, Any]], classes: str = "both",
              qids: Optional[Sequence[str]] = None, max_qids: int = 0) -> List[str]:
    """Row order for the run: the trigger frame's own order, optionally one class
    (``cw`` / ``cc``) or an explicit qid list (must be inside the frame)."""
    by_qid = {r["qid"]: r for r in frame_rows}
    if classes == "cw":
        keep = [r["qid"] for r in frame_rows if r["label_cw"] == 1]
    elif classes == "cc":
        keep = [r["qid"] for r in frame_rows if r["label_cw"] == 0]
    elif classes == "both":
        keep = [r["qid"] for r in frame_rows]
    else:
        raise ValueError(f"classes must be both|cw|cc, got {classes!r}")
    if qids:
        unknown = [q for q in qids if q not in by_qid]
        if unknown:
            raise ValueError(f"--qids outside the trigger frame: {unknown[:3]}")
        wanted = set(qids)
        keep = [q for q in keep if q in wanted]
    if max_qids:
        keep = keep[:max_qids]
    return keep


def resume_done(path: Path) -> Set[Tuple[str, Optional[int]]]:
    """(qid, k) pairs already written by a previous run (skipped rows excluded)."""
    done: Set[Tuple[str, Optional[int]]] = set()
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("skipped"):
            done.add((row["qid"], row.get("d_corr_doc_index")))
    return done


def stamp_row(row: Dict[str, Any], k: int) -> Dict[str, Any]:
    row["d_arm"] = ARM
    row["d_mode"] = MODE
    row["d_ksweep_k"] = int(k)
    row["d_corr_doc_index"] = int(k)
    row["k_policy"] = "median"
    return row


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=".", help="worktree root (frozen assets)")
    p.add_argument("--model", default="/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint")
    p.add_argument("--base_model", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--tokenizer", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--dataset_path", default="/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2")
    p.add_argument("--device_type", default="npu")
    p.add_argument("--attn_impl", default="eager")
    p.add_argument("--ratio", type=int, default=8)
    p.add_argument("--max_doc_length", type=int, default=768)
    p.add_argument("--max_doc_num", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--output_file", required=True)
    p.add_argument("--timing_report", default="")
    p.add_argument("--classes", default="both", choices=("both", "cw", "cc"))
    p.add_argument("--qids", default="", help="comma-separated subset (inside the frame)")
    p.add_argument("--max_qids", type=int, default=0)
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> int:  # pragma: no cover - torch + NPU
    import d_ksweep_driver as KD  # imports torch at module level
    import eval_agent_history_c2kv as HH
    from d0_sidecar import SidecarStore
    from d1_arms import ksweep_prefix_for_k, prepare_d_contract_state
    from t34_common import FrozenAssets

    frame = FrozenAssets(Path(args.root)).load()
    rows = frame.trigger_subset()
    wanted = [q.strip() for q in args.qids.split(",") if q.strip()] if args.qids else None
    qids = plan_qids(rows, args.classes, wanted, args.max_qids)
    logger.info("stage-2 %s: %d rows (classes=%s)", ARM, len(qids), args.classes)

    hargs = KD._harness_args(args)
    hargs.qid_allowlist = set(qids)   # start-up cost: only these rows
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} qids not loaded: {missing[:3]}")

    device = HH._setup_device(args.device_type)
    hargs.mode = "c2kv"
    model = HH._load_model(hargs, tokenizer, device)
    store = SidecarStore(model)
    HH.D_CONTRACT_STORE = store
    HH.D_INTERVENE = {}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = resume_done(out_path) if args.resume else set()
    if done:
        logger.info("resume: %d rows already done", len(done))

    timing: List[Dict[str, Any]] = []
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for qi, qid in enumerate(qids):
            example = by_qid[qid]
            t_q = time.perf_counter()
            try:
                state, skip_reason = prepare_d_contract_state(model, tokenizer, example, hargs, store)
            except RuntimeError as e:
                if not HH._is_oom_error(e):
                    raise
                logger.warning("OOM in prepare qid=%s -> skipped", qid)
                handle.write(json.dumps(HH._oom_row(example, MODE, args.ratio), ensure_ascii=False) + "\n")
                HH._clear_device_cache(args.device_type)
                continue
            if state is None:
                row = HH._oom_row(example, MODE, args.ratio)
                row["skip_reason"] = skip_reason
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            n_docs = len(state["doc_ids"])
            k = k_median(n_docs)
            if (qid, k) in done:
                store.release(qid)
                continue
            t_prep = time.perf_counter() - t_q
            t_g = time.perf_counter()
            try:
                prefix = ksweep_prefix_for_k(model, state, store, k)
                original_mode = hargs.mode
                hargs.mode = MODE
                row = HH._generate_one(model, tokenizer, example, hargs, MODE, prefix_override=prefix)
                hargs.mode = original_mode
            except RuntimeError as e:
                if not HH._is_oom_error(e):
                    raise
                logger.warning("OOM qid=%s k=%d, skipped", qid, k)
                row = HH._oom_row(example, MODE, args.ratio)
                HH._clear_device_cache(args.device_type)
            stamp_row(row, k)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            store.release(qid)
            t_gen = time.perf_counter() - t_g
            timing.append({"qid": qid, "n_docs": n_docs, "k": k,
                           "prepare_sec": round(t_prep, 3), "splice_plus_generate_sec": round(t_gen, 3),
                           "wall_sec": round(time.perf_counter() - t_q, 3)})
            logger.info("[%d/%d] qid=%s n_docs=%d k=%d prepare=%.1fs splice+gen=%.1fs",
                        qi + 1, len(qids), qid, n_docs, k, t_prep, t_gen)
            HH._clear_device_cache(args.device_type)

    if args.timing_report:
        Path(args.timing_report).write_text(json.dumps({
            "arm": ARM, "mode": MODE, "n_rows": len(timing),
            "sum_wall_sec": round(sum(t["wall_sec"] for t in timing), 1),
            "rows": timing}, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "n_new_rows": len(timing), "arm": ARM}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
