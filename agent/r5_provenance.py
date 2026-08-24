"""R5 S7：物证补齐（服务器 CPU，零 GPU）——prior 臂换池 provenance 提取 + 两种池序列化内容 sha256。

产出两件：
1. prior 臂换池 provenance：从 prior_prompts.jsonl（r4_prior_pool.py 产出，行内
   prior_swap.source_sessions / orig_doc_tokens / length_deviation / within_tol / seed）
   提取每 qid 的换池来源，剔除 input_ids 大字段，并 join r4_prior_76k.jsonl 的运行行
   核对 48 qid 全覆盖；标记 F7 的 4 个长度违规行（doc_tokens ≈ 25001 而非目标 ±5%）。
2. 池序列化内容 sha256：复用 r4_error_taxonomy._tooldef_pools() 重建 395 主集涉及的
   全部 session 池文本，按 (session_id, doc_token_len) 聚合，对**逐字序列化文本**
   （UTF-8 编码）算 sha256；输出两种池（75327/80171）的 hash、覆盖 session 数与
   qid 数；同时导出每 qid → 池 hash 的映射（供 manifest 引用）。

用法（NPU 服务器，仓库根目录）：
  python agent/r5_provenance.py \
    --prior_prompts ./outputs_lyc/r4_closure/prior_76k/prior_prompts.jsonl \
    --prior_rows <本地传入的 r4_prior_76k.jsonl 路径> \
    --rows outputs_lyc/r5_closeout/r4_rows/t_e_c2kv_r4.jsonl outputs_lyc/r5_closeout/r4_rows/f_ext_c2kv.jsonl \
    --out outputs_lyc/r5_closeout/provenance_s7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

logger = logging.getLogger("r5_provenance")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prior_provenance(prior_prompts: str, prior_rows: str) -> Dict[str, Any]:
    prompts = _load_jsonl(prior_prompts)
    run_rows = {r["qid"]: r for r in _load_jsonl(prior_rows)}
    per_qid: Dict[str, Any] = {}
    n_violation = 0
    for row in prompts:
        qid = row["qid"]
        swap = row.get("prior_swap") or {}
        doc_tokens = row.get("doc_tokens")
        # F7：4/48 行以 ~25001 token 而非 ~77k 运行（目标排除正则把通用词 finish
        # 匹配进几乎所有候选池导致静默欠填充）；判定 = 未落在预注册 ±5% 容差内。
        within = bool(swap.get("within_tol"))
        if not within:
            n_violation += 1
        per_qid[qid] = {
            "session_id": row.get("session_id"),
            "orig_doc_tokens": swap.get("orig_doc_tokens"),
            "sub_doc_tokens": doc_tokens,
            "length_deviation": swap.get("length_deviation"),
            "within_tol": within,
            "length_violation_F7": not within,
            "source_sessions": swap.get("source_sessions"),
            "target_tool_name": swap.get("target_tool_name"),
            "seed": swap.get("seed"),
            "ran_row_present": qid in run_rows,
            "run_n_tokens": (run_rows.get(qid) or {}).get("n_tokens"),
        }
    missing_run = [q for q in per_qid if q not in run_rows]
    return {
        "source_prompts": prior_prompts,
        "run_rows": prior_rows,
        "n_qids": len(per_qid),
        "n_run_rows": len(run_rows),
        "qids_missing_run_row": missing_run,
        "n_length_violations_F7": n_violation,
        "note": "F7：4/48 行（全 finish 目标）以 25,001 token 而非 ~77k 运行（−67.6%，预注册 ±5% 违规），"
                "因目标排除正则把通用词 finish 匹配进几乎所有候选池导致静默欠填充；"
                "剔除后地板 acc 0.1818（8/44）/ call 0.5909，性质为历史复制地板。",
        "per_qid": per_qid,
    }


def _pool_hashes(rows_paths: List[str]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for path in rows_paths:
        for r in _load_jsonl(path):
            rows.setdefault(r["qid"], r)
    logger.info("loaded %d rows for pool hashing", len(rows))

    import r4_error_taxonomy as R4

    pools_td = R4._tooldef_pools()  # qid -> (pool_text, doc_ids)
    logger.info("pools rebuilt for %d qids", len(pools_td))

    qid_pool: Dict[str, Any] = {}
    by_hash: Dict[str, Any] = {}
    for qid, row in rows.items():
        if qid not in pools_td:
            qid_pool[qid] = {"pool_sha256": None, "reason": "qid_not_in_pools"}
            continue
        text, doc_ids = pools_td[qid]
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = by_hash.setdefault(h, {"sha256": h, "doc_token_len": len(doc_ids), "sessions": set(), "qids": []})
        entry["sessions"].add(row.get("session_id"))
        entry["qids"].append(qid)
        qid_pool[qid] = {"pool_sha256": h, "doc_tokens_field": row.get("doc_tokens"), "doc_token_len": len(doc_ids)}

    pools_out = []
    for h, e in sorted(by_hash.items(), key=lambda kv: kv[1]["doc_token_len"]):
        pools_out.append({
            "sha256": h,
            "doc_token_len": e["doc_token_len"],
            "n_sessions": len(e["sessions"]),
            "sessions": sorted(e["sessions"]),
            "n_qids": len(e["qids"]),
        })
    return {
        "note": "池序列化内容 sha256（UTF-8 逐字）；F2：395 主集只有两种池序列化（75327 与 80171）。"
                "池归属按行内 doc_tokens 判定。",
        "n_distinct_pool_serializations": len(pools_out),
        "pools": pools_out,
        "qid_to_pool": qid_pool,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior_prompts", default="./outputs_lyc/r4_closure/prior_76k/prior_prompts.jsonl")
    p.add_argument("--prior_rows", required=True, help="r4_prior_76k.jsonl（repo results/r4/prior_76k/）")
    p.add_argument("--rows", nargs="+", required=True, help="c2kv 臂 395 行 jsonl（池 hash 范围）")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    report = {
        "task": "S7 provenance: prior-arm pool swap provenance + pool serialization content sha256",
        "produced_by": "agent/r5_provenance.py",
        "prior_arm": _prior_provenance(args.prior_prompts, args.prior_rows),
        "pool_serializations": _pool_hashes(args.rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("-> %s", out)


if __name__ == "__main__":
    main()
