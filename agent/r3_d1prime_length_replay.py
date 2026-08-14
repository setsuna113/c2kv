"""R3 T-C: offline replay of D1' condition lengths + length-confound regression.

The round-2 readout (outputs_lyc/r2_d1prime/d1prime_condition_readout.jsonl)
carries per-sample losses for the four conditions but NO condition-length
columns. The donor mapping is deterministic (batch_size=1, variant_seed=0,
sorted manifest qids — see readout.log line 2) and condition_text comes from
the same frozen source pipeline, so len_real / len_donor (raw condition token
count, capped at the 256-token window exactly like training) are recomputed
here on CPU by replaying eval_condition_variants' own helpers verbatim.

Regression: per-sample (loss_real - loss_shuffled) on (len_real - len_donor).
Reports slope, intercept (= length-adjusted point estimate at zero length
difference), R^2, the plain paired mean, and the fraction of the plain mean
attributed to the length term.

Pure CPU. Usage (NPU server, repo root):
  python agent/r3_d1prime_length_replay.py \
      --readout <outputs_lyc/r2_d1prime/d1prime_condition_readout.jsonl> \
      --out_prefix <out>/t_c_length
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

# Pin the same env gates BEFORE the source pipeline is built (the source only
# populates condition_text when the window env is > 0).
os.environ["C2KV_CONDITION_WINDOW_TOKENS"] = "256"
os.environ["C2KV_CONDITION_DROPOUT"] = "0"

import eval_condition_variants as V  # noqa: E402

logger = logging.getLogger("r3_d1prime_length_replay")

WINDOW = 256
BATCH_SIZE = 1  # readout.log: batch=1 window=256 variant_seed=0
VARIANT_SEED = 0


def _ols(xs: List[float], ys: List[float]) -> Dict[str, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = my - slope * mx if not math.isnan(slope) else float("nan")
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) if not math.isnan(slope) else ss_tot
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "mean_x": mx, "mean_y": my, "slope": slope, "intercept": intercept, "r2": r2, "ss_res": ss_res, "ss_tot": ss_tot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readout", required=True)
    p.add_argument("--val_manifest", default="./configs/d1prime_frozen_val.json")
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--tokenizer", default="./checkpoints/qwen3-4b-agent-history-c2kv-npu/checkpoint-2678")
    p.add_argument("--dataset_path", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    manifest = V._load_manifest(args.val_manifest)
    qids = sorted(manifest["qids"])
    dataset_path = args.dataset_path or manifest["created_from"]
    logger.info("Manifest: %d qids (dataset=%s)", len(qids), dataset_path)

    source = V._build_source(manifest.get("filters", {}), dataset_path)
    examples_by_qid: Dict[str, Any] = {example.qid: example for example in source}
    missing = [q for q in qids if q not in examples_by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} manifest qids not reproduced: {missing[:5]}")

    session_ids = [(q.rsplit(":", 1)[0] if ":" in q else q) for q in qids]
    shuffled_map, _ = V._donor_maps(qids, session_ids, BATCH_SIZE, VARIANT_SEED)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)

    def _cond_len(qid: str) -> int:
        text = (getattr(examples_by_qid[qid], "condition_text", "") or "").strip()
        if not text:
            return 0
        return min(len(tokenizer.encode(text, add_special_tokens=False)), WINDOW)

    # Readout join: per-sample real-shuffled paired diff (skipped rows dropped,
    # same as analyze_condition_readout.py).
    loss_by_qid: Dict[str, Dict[str, Any]] = {}
    with Path(args.readout).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            loss_by_qid[row["qid"]] = row

    records = []
    for i, qid in enumerate(qids):
        row = loss_by_qid.get(qid)
        if row is None or row.get("skipped"):
            continue
        if row.get("loss_real") is None or row.get("loss_shuffled") is None:
            continue
        donor_qid = qids[shuffled_map[i]]
        len_real = _cond_len(qid)
        len_donor = _cond_len(donor_qid)
        records.append(
            {
                "qid": qid,
                "session_id": row.get("session_id") or session_ids[i],
                "donor_qid": donor_qid,
                "len_real": len_real,
                "len_donor": len_donor,
                "len_diff": len_real - len_donor,
                "diff_rs": row["loss_real"] - row["loss_shuffled"],
            }
        )
    logger.info("Joined %d/%d readout rows with replayed lengths", len(records), len(loss_by_qid))

    xs = [r["len_diff"] for r in records]
    ys = [r["diff_rs"] for r in records]
    reg = _ols(xs, ys)
    plain_mean = reg["mean_y"]
    explained = (reg["slope"] * reg["mean_x"]) if not math.isnan(reg["slope"]) else float("nan")
    reg["plain_mean_diff_rs"] = plain_mean
    reg["length_term_at_mean_x"] = explained
    reg["adjusted_point_estimate"] = reg["intercept"]
    reg["fraction_explained_by_length"] = (explained / plain_mean) if plain_mean else float("nan")
    reg["window"] = WINDOW
    reg["batch_size"] = BATCH_SIZE
    reg["variant_seed"] = VARIANT_SEED
    reg["n_sessions"] = len({r["session_id"] for r in records})

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".lengths.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_prefix.with_suffix(".regression.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("OLS: slope=%.6f intercept=%.4f R2=%.4f plain_mean=%.4f frac_explained=%.3f",
                reg["slope"], reg["intercept"], reg["r2"], plain_mean, reg["fraction_explained_by_length"])
    logger.info("Wrote %s{.lengths.jsonl,.regression.json}", out_prefix)


if __name__ == "__main__":
    main()
