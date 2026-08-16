# -*- coding: utf-8 -*-
"""S8 chunk B：M1–M3 统计分析（analyze_s8）。

读 chunk A 产出的 scored jsonl（metrology/bfcl_score.py 行 schema）与冻结样本
清单（configs/r5_metrology_sample.json），按 configs/r5_metrology_prereg.md §5/§6
口径计算三张测量表并输出全部数字到 --out 的 json，同时把三张表的 markdown 版
打印到 stdout（供报告直接粘贴）：

- M1：censoring 重分类率（cap128 失败 → cap1024 成功，主口径）+ 标签改变比例 +
  类别分解 + 首个分叉轮分布（multi_turn，需 --runs_dir 原始行）。
- M2：外壳-语义分裂率（split_row，主分母 = 冻结样本 360，缺失行记 missing_n）。
- M3：排名/税符号（基线 vs 修正口径）+ C2KV 描述性（INTERNAL-ONLY）+
  构成敏感性（方案 (a) 各层等权 / 方案 (b) 395 集边际映射）。

纯 stdlib、纯 CPU；不 import bfcl_eval、不 import 其他 metrology 模块。
判定字段一律是「辅助/描述字段」，不作为硬断言（prereg 已冻结口径）。

口径注（如实写入输出 json 与 md）：
- 各表 acc 分母：M1 标签改变 ÷ n_total（360）；M2 主分母 360；
  M3 基线/修正 acc 均以 360 冻结分母计、缺失行按 0 计（n_present 另列）。
- turn_index 一律 0 基（与 runner 行内一致）。
- 方案 (b) 的层→格映射为无自然对应下的固定约定（见 composition.mapping_note）。
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

CONDITIONS = ["base", "snapkv", "streamingllm"]
CAP_TIERS = ["default", "128", "1024"]
GROUP3 = {
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_long_context",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
    ],
    "parallel": ["parallel"],
    "parallel_multiple": ["parallel_multiple"],
}


# ══════════════════════════════════════════════════════════════════════════
# 基础 IO / 归一化 / 通用小工具
# ══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list:
    """读 jsonl；空行跳过；坏行直接报错退出（与 chunk A 一致）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON 行: {e}") from e
    return rows


def load_json(path: Path):
    """读 json 文件（sample 清单 / v1 分层文件）。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def runs_dir_digest(runs_dir: Path) -> tuple:
    """runs_dir 下全部 *.jsonl（按文件名排序、identity 文件排后）的组合摘要。

    返回 (combined_sha256, {filename: sha256})；目录为空/无 jsonl 返回 (None, {})。
    """
    files = sorted(
        [p for p in runs_dir.glob("*.jsonl")],
        key=lambda p: ("identity" in p.name, p.name),
    )
    h = hashlib.sha256()
    per_file = {}
    for p in files:
        s = sha256_of(p)
        per_file[p.name] = s
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    if not files:
        return None, {}
    return h.hexdigest(), per_file


def norm_cond(v) -> str:
    return str(v)


def norm_tier(v) -> str:
    return str(v)


def _sign(v):
    v = round(float(v), 12)
    if v > 0:
        return "+"
    if v < 0:
        return "-"
    return "0"


def _rank(accs: dict) -> list:
    """按 acc 降序排名；并列按条件名字典序（确定性）。"""
    return sorted(
        [c for c, a in accs.items() if a is not None],
        key=lambda c: (-accs[c], c),
    )


def _taxes(accs: dict) -> dict:
    """压缩税 = acc_base − acc_method；正 = 压缩掉点。返回 {method: {value, sign}}。"""
    if "base" not in accs or accs["base"] is None:
        return {}
    taxes = {}
    for m in accs:
        if m == "base" or accs[m] is None:
            continue
        value = round(accs["base"] - accs[m], 12)
        taxes[m] = {"value": value, "sign": _sign(value)}
    return taxes


def _fmt(v):
    """md 表格单元格格式化：浮点 4 位小数，None → '-'。"""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def frozen_map(sample: dict) -> dict:
    """id → {category, n_turns, gold_turn_tokens}。"""
    return {
        it["id"]: {
            "category": it["category"],
            "n_turns": it.get("n_turns"),
            "gold_turn_tokens": it.get("gold_turn_tokens", []),
        }
        for it in sample.get("items", [])
    }


def load_runs_rows(runs_dir) -> dict:
    """原始 runner 行：key = (id, condition, cap_tier) → 行。

    (id, condition, cap_tier) 重复时按文件名排序（identity 文件排后）取先者，
    重复数单独返回。返回 (rows, n_dups)。
    """
    rows = {}
    n_dups = 0
    files = sorted(
        [p for p in Path(runs_dir).glob("*.jsonl")],
        key=lambda p: ("identity" in p.name, p.name),
    )
    for p in files:
        for row in load_jsonl(p):
            key = (str(row["id"]), norm_cond(row.get("condition")),
                   norm_tier(row.get("cap_tier")))
            if key in rows:
                n_dups += 1
                continue
            rows[key] = row
    return rows, n_dups


def _scored_cells(scored: list, frozen: dict):
    """scored 行按 (condition, cap_tier) → {id: 行} 落格；返回 (cells, n_excluded,
    n_dups)。非冻结 id 行剔除（记数）；格内 (id, cond, tier) 重复取先者（记数）。"""
    cells = {}
    n_excluded = 0
    n_dups = 0
    for r in scored:
        if r["id"] not in frozen:
            n_excluded += 1
            continue
        key = (norm_cond(r.get("condition")), norm_tier(r.get("cap_tier")))
        cell = cells.setdefault(key, {})
        if r["id"] in cell:
            n_dups += 1
            continue
        cell[r["id"]] = r
    return cells, n_excluded, n_dups


# ══════════════════════════════════════════════════════════════════════════
# M1：censoring 重分类率 + 标签改变 + 类别分解 + 首个分叉轮
# ══════════════════════════════════════════════════════════════════════════

def _m1_row(cell128: dict, cell1024: dict, ids: list, n_total: int) -> dict:
    """一组 id 上的 M1 行（cell* = {id: scored 行}）。

    主口径：n_fail = cap128 下 native_valid=False 的样本数（含 cap1024 缺失者）；
    n_rec = 其中同 id 在 cap1024 下 native_valid=True 的样本数；重分类率 =
    n_rec/n_fail（n_fail=0 时 None）。标签改变：128↔1024 双双在场的 id 里
    native_valid 不一致数 ÷ n_total。
    """
    n_fail = 0
    fail_ids = []
    for i in ids:
        if i in cell128 and not cell128[i].get("native_valid"):
            n_fail += 1
            fail_ids.append(i)
    n_rec = sum(1 for i in fail_ids if i in cell1024 and cell1024[i].get("native_valid"))
    rate = (n_rec / n_fail) if n_fail else None
    both = [i for i in ids if i in cell128 and i in cell1024]
    n_change = sum(
        1 for i in both
        if bool(cell128[i].get("native_valid")) != bool(cell1024[i].get("native_valid"))
    )
    change_rate = (n_change / n_total) if n_total else None
    n_fail_missing_1024 = sum(1 for i in fail_ids if i not in cell1024)
    n_fail_invalid_1024 = sum(
        1 for i in fail_ids if i in cell1024 and not cell1024[i].get("native_valid")
    )
    return {
        "n_total": n_total,
        "n_fail_128": n_fail,
        "n_rec_1024": n_rec,
        "rate": rate,
        "rate_ge_10pct": bool(rate is not None and rate >= 0.10),
        "n_fail_missing_1024": n_fail_missing_1024,
        "n_fail_invalid_1024": n_fail_invalid_1024,
        "n_both_present": len(both),
        "n_missing_128": sum(1 for i in ids if i not in cell128),
        "n_missing_1024": sum(1 for i in ids if i not in cell1024),
        "n_label_change": n_change,
        "label_change_rate": change_rate,
        "fail_ids": fail_ids,
        "rec_ids": [i for i in fail_ids
                    if i in cell1024 and cell1024[i].get("native_valid")],
        "change_ids": [
            i for i in both
            if bool(cell128[i].get("native_valid")) != bool(cell1024[i].get("native_valid"))
        ],
    }


def _turn_steps_seq(row: dict) -> list:
    """行 → [[step parsed_text, ...], ...]（turn_index / step_index 升序）。"""
    seq = []
    for t in sorted(row.get("turns") or [], key=lambda x: x.get("turn_index", 0)):
        steps = sorted(t.get("steps") or [], key=lambda x: x.get("step_index", 0))
        seq.append([s.get("parsed_text") or "" for s in steps])
    return seq


def _first_diff_turn(a: list, b: list):
    """首个不同 turn_index（0 基）；完全一致返回 None；轮数不同分叉于较小轮数。"""
    if a == b:
        return None
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n


def _divergence(runs_rows: dict, frozen: dict) -> dict:
    """首个分叉轮（multi_turn 行，0 基 turn_index）。

    runs_rows 为 None → NOT-AVAILABLE；无 multi_turn 冻结样本 → NOT-APPLICABLE。
    """
    if runs_rows is None:
        return {"status": "NOT-AVAILABLE",
                "note": "--runs_dir 缺省，分叉轮统计不计算"}
    multi_ids = [i for i, it in frozen.items()
                 if it["category"].startswith("multi_turn")]
    if not multi_ids:
        return {"status": "NOT-APPLICABLE",
                "note": "无 multi_turn 冻结样本（单轮类记 NOT-APPLICABLE）"}
    per_condition = {}
    for cond in CONDITIONS:
        comparable = []
        for i in multi_ids:
            r128 = runs_rows.get((i, cond, "128"))
            r1024 = runs_rows.get((i, cond, "1024"))
            if r128 is None or r1024 is None:
                continue
            k = _first_diff_turn(_turn_steps_seq(r128), _turn_steps_seq(r1024))
            comparable.append((i, k))
        diverged = [k for _, k in comparable if k is not None]
        hist = {}
        for k in diverged:
            hist[k] = hist.get(k, 0) + 1
        per_condition[cond] = {
            "n_comparable": len(comparable),
            "n_identical": sum(1 for _, k in comparable if k is None),
            "n_diverged": len(diverged),
            "turn_index_min": min(diverged) if diverged else None,
            "turn_index_median": (float(statistics.median(diverged))
                                  if diverged else None),
            "turn_index_max": max(diverged) if diverged else None,
            "histogram": hist,
        }
    return {
        "status": "AVAILABLE",
        "note": "turn_index 为 0 基（与 runner 行内一致）；单轮类记 NOT-APPLICABLE",
        "per_condition": per_condition,
    }


def compute_m1(scored: list, sample: dict, runs_rows=None,
               input_sha: dict = None) -> dict:
    frozen = frozen_map(sample)
    n_total = int(sample.get("n_total") or len(frozen))
    cells, n_excluded, n_dups = _scored_cells(scored, frozen)
    all_ids = sorted(frozen.keys())

    main = {}
    for cond in CONDITIONS:
        main[cond] = _m1_row(
            cells.get((cond, "128"), {}), cells.get((cond, "1024"), {}),
            all_ids, n_total,
        )
    judgments = [
        main[c]["rate_ge_10pct"] for c in CONDITIONS if c != "base"
    ]
    judgment = "M1_SUPPORTED" if (judgments and all(judgments)) else "NOT-GENERALIZED"

    group3_ids = {}
    for gname, cats in GROUP3.items():
        group3_ids[gname] = sorted(
            i for i in all_ids if frozen[i]["category"] in cats
        )
    by_group3 = {}
    for cond in CONDITIONS:
        row = {}
        for gname, gids in group3_ids.items():
            row[gname] = _m1_row(
                cells.get((cond, "128"), {}), cells.get((cond, "1024"), {}),
                gids, len(gids),
            )
        by_group3[cond] = row
    by_category = {}
    for cond in CONDITIONS:
        row = {}
        for cat in sorted({it["category"] for it in frozen.values()}):
            gids = sorted(i for i in all_ids if frozen[i]["category"] == cat)
            row[cat] = _m1_row(
                cells.get((cond, "128"), {}), cells.get((cond, "1024"), {}),
                gids, len(gids),
            )
        by_category[cond] = row

    return {
        "n_total": n_total,
        "n_rows_scored_used": sum(len(c) for c in cells.values()),
        "n_rows_non_frozen_excluded": n_excluded,
        "n_duplicate_rows_kept_first": n_dups,
        "input_sha256": input_sha or {},
        "static_audit_ref":
            "configs/r5_metrology_prereg.md §2（BFCL 原生 OSS 路径默认 cap = "
            "min(4096, 剩余上下文)；本冻结样本 6 类金标每轮 P95 ≤ 192 token；"
            "cap=128 档正是要测量的预算 censoring 区间）",
        "main": main,
        "by_group3": by_group3,
        "by_category": by_category,
        "first_divergence_turn": _divergence(runs_rows, frozen),
        "judgment": judgment,
        "judgment_note": "M1 判定辅助字段（不作为硬断言）：两压缩条件重分类率均 "
                         "≥10% → M1_SUPPORTED，否则 NOT-GENERALIZED",
    }


# ══════════════════════════════════════════════════════════════════════════
# M2：外壳-语义分裂率（split_row；主分母 = 360 冻结样本）
# ══════════════════════════════════════════════════════════════════════════

def _closeout_arm(rows: list) -> dict:
    """S6 内部 scored 一行臂的 M2 统计（INTERNAL-ONLY，cap=256 固定）。

    行内 scoring.{protocol_valid, semantic_correct} 才计入；split =
    semantic_correct ∧ ¬protocol_valid。分母按任务书固定 89（n_rows 另列实际行数）。
    """
    n_rows = len(rows)
    used = [r for r in rows if isinstance(r.get("scoring"), dict)]
    split_n = sum(
        1 for r in used
        if bool(r["scoring"].get("semantic_correct"))
        and not bool(r["scoring"].get("protocol_valid"))
    )
    return {
        "n": 89,
        "n_rows": n_rows,
        "n_used": len(used),
        "split_n": split_n,
        "rate_split_89": split_n / 89.0,
    }


def compute_m2(scored: list, sample: dict, closeout_full=None,
               closeout_c2kv=None, input_sha: dict = None) -> dict:
    frozen = frozen_map(sample)
    n_total = int(sample.get("n_total") or len(frozen))
    cells, n_excluded, n_dups = _scored_cells(scored, frozen)

    table = {}
    max_nonbase_rate = 0.0
    for cond in CONDITIONS:
        table[cond] = {}
        for tier in CAP_TIERS:
            cell = cells.get((cond, tier), {})
            n_scored = len(cell)
            split_n = sum(1 for r in cell.values() if bool(r.get("split_row")))
            protocol_invalid_n = sum(
                1 for r in cell.values() if not bool(r.get("protocol_valid"))
            )
            split_ids = sorted(
                i for i, r in cell.items() if bool(r.get("split_row"))
            )
            rate_main = split_n / float(n_total) if n_total else None
            table[cond][tier] = {
                "n_total": n_total,
                "n_scored": n_scored,
                "missing_n": max(n_total - n_scored, 0),
                "split_n": split_n,
                "split_rate_main": rate_main,
                "split_rate_scored": (split_n / n_scored) if n_scored else None,
                "protocol_invalid_n": protocol_invalid_n,
                "split_rate_display_protocol": (
                    split_n / protocol_invalid_n
                ) if protocol_invalid_n else None,
                "split_ids": split_ids,
            }
            if cond != "base":
                max_nonbase_rate = max(max_nonbase_rate, rate_main or 0.0)

    judgment = (
        "M2_SUPPORTED" if max_nonbase_rate >= 0.05 else "NOT-SUPPORTED"
    )

    if closeout_full is not None and closeout_c2kv is not None:
        c2kv = {
            "status": "AVAILABLE",
            "cap": 256,
            "internal_only": True,
            "note": "S6 内部数据充当，cap=256 固定，不参与 M2 判定",
            "full": _closeout_arm(closeout_full),
            "c2kv": _closeout_arm(closeout_c2kv),
        }
    else:
        c2kv = {
            "status": "MISSING",
            "cap": 256,
            "internal_only": True,
            "reason": "--closeout_full / --closeout_c2kv 缺省",
        }

    return {
        "n_total": n_total,
        "input_sha256": input_sha or {},
        "cells": table,
        "max_nonbase_split_rate_main": max_nonbase_rate,
        "judgment": judgment,
        "judgment_note": "M2 判定字段：任一压缩条件（不含 base）任一 cap 的主分母"
                         "分裂率 ≥5% → M2_SUPPORTED，否则 NOT-SUPPORTED；"
                         "base 各 cap 作对照同表列出",
        "c2kv": c2kv,
    }


# ══════════════════════════════════════════════════════════════════════════
# M3：排名/税符号 + 构成敏感性 + C2KV 描述性
# ══════════════════════════════════════════════════════════════════════════

def nearest_rank_p95(lengths: list):
    """nearest-rank P95：升序后第 ceil(0.95*m) 个（1 基）。空列表 → None。"""
    m = len(lengths)
    if m == 0:
        return None
    return sorted(lengths)[int(math.ceil(0.95 * m)) - 1]


def _cap_c_by_category(sample: dict) -> dict:
    """每类别：{p95_per_turn, cap_c, tier}。cap_c = max(1024, 金标每轮 token 长度
    P95)（P95 用该类别冻结样本每轮长度的展开列表，nearest-rank）。"""
    lengths = {}
    for it in sample.get("items", []):
        lengths.setdefault(it["category"], []).extend(it.get("gold_turn_tokens", []))
    out = {}
    for cat, vals in lengths.items():
        p95 = nearest_rank_p95(vals)
        cap_c = max(1024, p95) if p95 is not None else None
        out[cat] = {
            "p95_per_turn": p95,
            "cap_c": cap_c,
            "tier": str(cap_c) if cap_c is not None else None,
            "n_lengths": len(vals),
        }
    return out


def _c2kv_m3(full_rows: list, c2kv_rows: list) -> dict:
    """C2KV 描述性（INTERNAL-ONLY）：89 配对行上两臂 acc（protocol 列 =
    primary_success；语义列 = semantic_correct），税 = full − c2kv，符号如实报。"""
    # closeout scored 行的主键是 qid（无 id 字段），兼容两者
    f = {str(r.get("qid") or r.get("id")): r for r in full_rows
         if isinstance(r.get("scoring"), dict)}
    c = {str(r.get("qid") or r.get("id")): r for r in c2kv_rows
         if isinstance(r.get("scoring"), dict)}
    pairs = sorted(set(f) & set(c))
    out = {
        "status": "AVAILABLE",
        "internal_only": True,
        "expected_n": 89,
        "n_pairs": len(pairs),
    }
    if pairs:
        fp = sum(bool(f[i]["scoring"].get("primary_success")) for i in pairs) / len(pairs)
        fs = sum(bool(f[i]["scoring"].get("semantic_correct")) for i in pairs) / len(pairs)
        cp = sum(bool(c[i]["scoring"].get("primary_success")) for i in pairs) / len(pairs)
        cs = sum(bool(c[i]["scoring"].get("semantic_correct")) for i in pairs) / len(pairs)
        tp = round(fp - cp, 12)
        ts = round(fs - cs, 12)
        out.update({
            "full": {"acc_protocol": fp, "acc_semantic": fs},
            "c2kv": {"acc_protocol": cp, "acc_semantic": cs},
            "tax_full_minus_c2kv": {
                "protocol": {"value": tp, "sign": _sign(tp)},
                "semantic": {"value": ts, "sign": _sign(ts)},
            },
        })
    return out


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v) if v in (0, 1) else None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n", ""):
            return False
    return None


def _to_pool(v):
    """pool 轴数值化：保留数值不做 bool 化（真实文件 pool_doc_tokens 为
    75327/80171 这类整数）。bool 一律返回 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\d+", v.strip()):
        return int(v.strip())
    return None


def parse_v1_cells(v1_obj: dict):
    """从 v1_stratified json 取 8 格 (clipped × pool × finish)。

    - list 分支（真实文件格式）：clipped → _to_bool；pool_doc_tokens → _to_pool
      （保留数值）；finish → _to_bool(finish_target / is_finish / finish 任一
      存在字段）；
    - dict 分支（兼容形态：嵌套 dict / 扁平字符串键 "a_b_c" / "aXbXc"）：
      同样只对第 0、2 位做 _to_bool，pool 位用 _to_pool；
    - 同键多格 n 求和去重；
    - 排序：clipped False 在前、pool 数值升序、finish False 在前。

    返回 [(clipped, pool, finish, n), ...]；无有效格返回 None（格数不足 8 的
    校验由 _composition 做，防止层→格映射越界）。
    """
    if not isinstance(v1_obj, dict):
        return None
    raw = v1_obj.get("cells_clipped_x_pool_x_finish")
    if raw is None:
        for k, v in v1_obj.items():
            if "clip" in str(k).lower() and isinstance(v, (dict, list)):
                raw = v
                break
    if raw is None:
        return None

    collected = []

    if isinstance(raw, list):
        for c in raw:
            if not isinstance(c, dict):
                continue
            clipped = _to_bool(c.get("clipped"))
            pool = _to_pool(c.get("pool_doc_tokens", c.get("pool")))
            finish = _to_bool(c.get("finish_target",
                                    c.get("is_finish", c.get("finish"))))
            n = c.get("n")
            if (clipped is None or pool is None or finish is None
                    or not isinstance(n, (int, float))):
                continue
            collected.append(((clipped, pool, finish), float(n)))
    else:  # dict：嵌套 3 层或扁平字符串键
        def walk(obj, prefix):
            if isinstance(obj, dict) and len(prefix) < 3:
                for k, v in obj.items():
                    walk(v, prefix + [k])
                return
            n = obj.get("n") if isinstance(obj, dict) else obj
            if not isinstance(n, (int, float)):
                return
            if len(prefix) == 3:
                collected.append((tuple(prefix), float(n)))
            elif len(prefix) == 1 and isinstance(prefix[0], str):
                parts = [p for p in re.split(r"[_x]", prefix[0].strip().lower())
                         if p]
                if len(parts) >= 3:
                    collected.append((tuple(parts), float(n)))
        walk(raw, [])

    agg = {}
    for key, n in collected:
        clipped = _to_bool(key[0])
        pool = _to_pool(key[1])
        finish = _to_bool(key[2])
        if clipped is None or pool is None or finish is None:
            continue
        cell = (clipped, pool, finish)
        agg[cell] = agg.get(cell, 0.0) + float(n)
    out = sorted([(k, v) for k, v in agg.items()], key=lambda x: x[0])
    return out or None


def _turn_bin(n_turns) -> str:
    if n_turns is None:
        return "NA"
    if n_turns <= 2:
        return "1-2"
    if n_turns <= 4:
        return "3-4"
    return "5+"


MAPPING_NOTE = (
    "固定约定（无自然对应下冻结）：层按 (category, 分箱) 字典序排序，格按 "
    "(clipped, pool_doc_tokens, finish_target) 排序（clipped False 在前、"
    "pool_doc_tokens 数值升序、finish_target False 在前），第 j 层（0 起）映射格 "
    "floor(j×8/S)（S=非空层数），w_j = 该格 395 集边际占比，归一化后使用。"
    "此映射为固定约定，报告必须如实注明。"
)


def _composition(scored_map: dict, frozen: dict, cap_c_by_cat: dict,
                 v1_obj) -> dict:
    strata = {}
    for i, it in frozen.items():
        key = (it["category"], _turn_bin(it.get("n_turns")))
        if key[1] == "NA":
            continue
        strata.setdefault(key, []).append(i)
    strata_keys = sorted(strata.keys())
    s = len(strata_keys)
    stratum_info = [
        {"category": c, "turn_bin": b, "n": len(strata[(c, b)])}
        for c, b in strata_keys
    ]

    def stratum_acc(ids, cond, metric):
        cat = frozen[ids[0]]["category"]
        if metric == "corrected":
            tier = cap_c_by_cat.get(cat, {}).get("tier")
            if tier is None:
                return None
            field = "semantic_correct"
        else:
            tier = "default"
            field = "native_valid"
        num = sum(
            1 for i in ids
            if (i, cond, tier) in scored_map
            and bool(scored_map[(i, cond, tier)].get(field))
        )
        return num / float(len(ids))

    def scheme(weights: dict) -> dict:
        out = {"weights": weights}
        for metric in ("corrected", "baseline"):
            accs = {}
            for cond in CONDITIONS:
                pairs = []
                for j, key in enumerate(strata_keys):
                    a = stratum_acc(strata[key], cond, metric)
                    if a is None:
                        continue
                    pairs.append((weights[_stratum_key_str(key)], a))
                if pairs:
                    accs[cond] = (
                        sum(w * a for w, a in pairs) /
                        sum(w for w, _ in pairs)
                    )
                else:
                    accs[cond] = None
            out[metric] = {
                "conditions": accs,
                "ranking": _rank(accs),
                "taxes": _taxes(accs),
            }
        return out

    # 方案 (a)：各非空层等权
    w_a = {_stratum_key_str(k): 1.0 / s for k in strata_keys} if s else {}
    scheme_a = {"weight_rule": "各非空层等权", **scheme(w_a)}

    # 方案 (b)：内部 395 集构成映射
    if v1_obj is not None:
        cells = parse_v1_cells(v1_obj)
    else:
        cells = None
    if cells is None or len(cells) < 8:
        reason = ("--v1_stratified 缺省" if v1_obj is None
                  else ("cells_clipped_x_pool_x_finish 未解析到有效格"
                        if cells is None
                        else (f"仅解析到 {len(cells)} 格，不足 8 格"
                              "（clipped×pool×finish 全组合），"
                              "不能做层→格映射")))
        scheme_b = {
            "status": "MISSING",
            "reason": reason,
            "MISSING_DETAIL": reason,
        }
    else:
        total = sum(n for _, n in cells)
        p = {cell: n / total for cell, n in cells}
        raw_w = {}
        for j, key in enumerate(strata_keys):
            mapped = cells[int(math.floor(j * 8.0 / s))][0]
            raw_w[_stratum_key_str(key)] = p[mapped]
        w_sum = sum(raw_w.values()) or 1.0
        w_b = {k: v / w_sum for k, v in raw_w.items()}
        scheme_b = {
            "weight_rule": "内部 395 集边际映射（固定约定，见 mapping_note）",
            "cells": [
                {"clipped": cl, "pool_doc_tokens": po, "finish_target": fi,
                 "n": n}
                for (cl, po, fi), n in cells
            ],
            "cells_total_n": total,
            **scheme(w_b),
        }

    return {
        "n_strata": s,
        "strata": stratum_info,
        "mapping_note": MAPPING_NOTE,
        "note": "构成敏感性为描述性，不入 M3 判定",
        "scheme_a": scheme_a,
        "scheme_b": scheme_b,
    }


def _stratum_key_str(key) -> str:
    return f"{key[0]}|{key[1]}"


def compute_m3(scored: list, sample: dict, closeout_full=None,
               closeout_c2kv=None, v1_obj=None, input_sha: dict = None) -> dict:
    frozen = frozen_map(sample)
    n_total = int(sample.get("n_total") or len(frozen))
    cap_c_by_cat = _cap_c_by_category(sample)

    scored_map = {}
    for r in scored:
        if r["id"] not in frozen:
            continue
        key = (r["id"], norm_cond(r.get("condition")), norm_tier(r.get("cap_tier")))
        scored_map.setdefault(key, r)

    baseline = {}
    for cond in CONDITIONS:
        num = 0
        n_present = 0
        for i in frozen:
            r = scored_map.get((i, cond, "default"))
            if r is None:
                continue
            n_present += 1
            num += int(bool(r.get("native_valid")))
        baseline[cond] = {
            "acc": (num / float(n_total)) if n_present else None,
            "n_present": n_present,
        }

    corrected = {}
    for cond in CONDITIONS:
        num = 0
        n_present = 0
        for i, it in frozen.items():
            tier = cap_c_by_cat.get(it["category"], {}).get("tier")
            if tier is None:
                continue
            r = scored_map.get((i, cond, tier))
            if r is None:
                continue
            n_present += 1
            num += int(bool(r.get("semantic_correct")))
        corrected[cond] = {
            "acc": (num / float(n_total)) if n_present else None,
            "n_present": n_present,
        }

    b_accs = {c: v["acc"] for c, v in baseline.items()}
    c_accs = {c: v["acc"] for c, v in corrected.items()}
    b_ranking = _rank(b_accs)
    c_ranking = _rank(c_accs)
    b_taxes = _taxes(b_accs)
    c_taxes = _taxes(c_accs)

    conds_b = set(b_ranking)
    conds_c = set(c_ranking)
    if conds_b and conds_b == conds_c:
        rank_swap = b_ranking != c_ranking
        flips = sorted(
            m for m in b_taxes if m in c_taxes
            and b_taxes[m]["sign"] != c_taxes[m]["sign"]
        )
        judgment = "M3_HAS_CONSEQUENCE" if (rank_swap or flips) else "M3_NO_CONSEQUENCE"
    else:
        rank_swap = None
        flips = []
        judgment = "M3_INDETERMINATE"

    if closeout_full is not None and closeout_c2kv is not None:
        c2kv = _c2kv_m3(closeout_full, closeout_c2kv)
    else:
        c2kv = {"status": "MISSING",
                "reason": "--closeout_full / --closeout_c2kv 缺省"}

    composition = _composition(scored_map, frozen, cap_c_by_cat, v1_obj)

    return {
        "n_total": n_total,
        "input_sha256": input_sha or {},
        "cap_c_by_category": cap_c_by_cat,
        "cap_c_note": "cap_c = max(1024, 该类别金标每轮 token 长度 P95)，"
                      "P95 用 nearest-rank（升序第 ceil(0.95*m) 个，1 基）",
        "baseline": {
            "metric": "default cap + native_valid",
            "denominator_note": "acc 以 360 冻结分母计，缺失行按 0 计；n_present 另列",
            "conditions": baseline,
            "ranking": b_ranking,
            "taxes": b_taxes,
        },
        "corrected": {
            "metric": "cap_c + semantic_correct",
            "denominator_note": "acc 以 360 冻结分母计，缺失行按 0 计；n_present 另列",
            "conditions": corrected,
            "ranking": c_ranking,
            "taxes": c_taxes,
        },
        "judgment": judgment,
        "judgment_note": "M3 判定字段：任一压缩方法排名对换（3 条件名次变化）或任一"
                         "方法税符号翻转（+/0/− 三态，0 与他态互转算翻转）→ "
                         "M3_HAS_CONSEQUENCE",
        "judgment_detail": {
            "rank_swap": rank_swap,
            "tax_sign_flips": flips,
            "baseline_ranking": b_ranking,
            "corrected_ranking": c_ranking,
        },
        "c2kv": c2kv,
        "composition": composition,
    }


# ══════════════════════════════════════════════════════════════════════════
# 汇总 / markdown 渲染 / CLI
# ══════════════════════════════════════════════════════════════════════════

def _render_m1(m1: dict) -> str:
    lines = ["## M1：censoring 重分类率（cap128 失败 → cap1024 成功；主口径）", ""]
    lines.append("| 条件 | n_fail(128) | n_rec(1024) | 重分类率 | ≥10% | 标签改变 n | 标签改变率(n/360) |")
    lines.append("|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        r = m1["main"][cond]
        lines.append(
            f"| {cond} | {r['n_fail_128']} | {r['n_rec_1024']} | {_fmt(r['rate'])} "
            f"| {r['rate_ge_10pct']} | {r['n_label_change']} | {_fmt(r['label_change_rate'])} |"
        )
    lines.append("")
    lines.append(f"M1 判定字段：{m1['judgment']}")
    lines.append("")
    lines.append("M1 类别分解（multi_turn 合并 / parallel / parallel_multiple）：")
    lines.append("")
    lines.append("| 组 | 条件 | n_fail | n_rec | 重分类率 |")
    lines.append("|---|---|---|---|---|")
    for cond in CONDITIONS:
        for gname in GROUP3:
            r = m1["by_group3"][cond][gname]
            lines.append(
                f"| {gname} | {cond} | {r['n_fail_128']} | {r['n_rec_1024']} | {_fmt(r['rate'])} |"
            )
    lines.append("")
    lines.append("M1 类别分解（6 类明细）：")
    lines.append("")
    lines.append("| 类别 | 条件 | n_fail | n_rec | 重分类率 |")
    lines.append("|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cat in sorted(m1["by_category"][cond]):
            r = m1["by_category"][cond][cat]
            lines.append(
                f"| {cat} | {cond} | {r['n_fail_128']} | {r['n_rec_1024']} | {_fmt(r['rate'])} |"
            )
    lines.append("")
    d = m1["first_divergence_turn"]
    lines.append(f"首个分叉轮（multi_turn，0 基 turn_index）：{d['status']}")
    if d["status"] == "AVAILABLE":
        lines.append("")
        lines.append("| 条件 | 可比对数 | 轨迹完全一致 | 分叉 n | min | 中位 | max |")
        lines.append("|---|---|---|---|---|---|---|")
        for cond in CONDITIONS:
            pc = d["per_condition"][cond]
            lines.append(
                f"| {cond} | {pc['n_comparable']} | {pc['n_identical']} | "
                f"{pc['n_diverged']} | {_fmt(pc['turn_index_min'])} | "
                f"{_fmt(pc['turn_index_median'])} | {_fmt(pc['turn_index_max'])} |"
            )
        lines.append("")
        lines.append("分叉轮直方图（详见输出 json M1.first_divergence_turn.histogram）")
    lines.append("")
    return "\n".join(lines)


def _render_m2(m2: dict) -> str:
    lines = ["## M2：外壳-语义分裂率（split_row；主分母 = 360 冻结样本）", ""]
    lines.append("| 条件 | cap | n_scored | missing_n | split_n | split_n/360 | split_n/n_scored | protocol_invalid_n | split_n/protocol_invalid |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for tier in CAP_TIERS:
            c = m2["cells"][cond][tier]
            lines.append(
                f"| {cond} | {tier} | {c['n_scored']} | {c['missing_n']} | "
                f"{c['split_n']} | {_fmt(c['split_rate_main'])} | "
                f"{_fmt(c['split_rate_scored'])} | {c['protocol_invalid_n']} | "
                f"{_fmt(c['split_rate_display_protocol'])} |"
            )
    lines.append("")
    lines.append(f"M2 判定字段：{m2['judgment']}（非 base 压缩条件下主分母分裂率最大 {_fmt(m2['max_nonbase_split_rate_main'])})")
    lines.append("")
    c2 = m2["c2kv"]
    if c2["status"] == "AVAILABLE":
        lines.append(f"C2KV 列（INTERNAL-ONLY，cap=256 固定，不参与 M2 判定）："
                     f"full 臂 split {c2['full']['split_n']}/89，"
                     f"c2kv 臂 split {c2['c2kv']['split_n']}/89")
    else:
        lines.append("C2KV 列：MISSING（--closeout_full / --closeout_c2kv 缺省）")
    lines.append("")
    return "\n".join(lines)


def _scheme_md(scheme: dict) -> str:
    lines = []
    for metric in ("corrected", "baseline"):
        lines.append(f"  {metric}（{'cap_c+semantic' if metric == 'corrected' else 'default+native'}）：")
        lines.append("  | 条件 | 加权 acc | 排名 | 税(值/符号) |")
        lines.append("  |---|---|---|---|")
        accs = scheme[metric]["conditions"]
        ranking = scheme[metric]["ranking"]
        taxes = scheme[metric]["taxes"]
        for cond in CONDITIONS:
            a = accs[cond]
            rank = (ranking.index(cond) + 1) if a is not None and cond in ranking else "-"
            tax = taxes.get(cond)
            tax_txt = f"{_fmt(tax['value'])}/{tax['sign']}" if tax else "-"
            lines.append(f"  | {cond} | {_fmt(a)} | {rank} | {tax_txt} |")
    return "\n".join(lines)


def _render_m3(m3: dict) -> str:
    lines = ["## M3：排名/税符号（税 = acc_base − acc_method，正 = 压缩掉点）", ""]
    lines.append("基线（default cap + native_valid，360 分母、缺失记 0）：")
    lines.append("")
    lines.append("| 条件 | acc | n_present | 排名 |")
    lines.append("|---|---|---|---|")
    b = m3["baseline"]
    for cond in CONDITIONS:
        acc = b["conditions"][cond]["acc"]
        rank = (b["ranking"].index(cond) + 1) if acc is not None and cond in b["ranking"] else "-"
        lines.append(
            f"| {cond} | {_fmt(acc)} | {b['conditions'][cond]['n_present']} | {rank} |"
        )
    lines.append("")
    lines.append("修正（cap_c + semantic_correct，360 分母、缺失记 0）：")
    lines.append("")
    lines.append("| 条件 | acc | n_present | 排名 |")
    lines.append("|---|---|---|---|")
    c = m3["corrected"]
    for cond in CONDITIONS:
        acc = c["conditions"][cond]["acc"]
        rank = (c["ranking"].index(cond) + 1) if acc is not None and cond in c["ranking"] else "-"
        lines.append(
            f"| {cond} | {_fmt(acc)} | {c['conditions'][cond]['n_present']} | {rank} |"
        )
    lines.append("")
    lines.append(f"cap_c 逐类别：{m3['cap_c_by_category']}")
    lines.append("")
    detail = m3["judgment_detail"]
    lines.append(
        f"M3 判定字段：{m3['judgment']}（排名对换 {detail['rank_swap']}，"
        f"税符号翻转 {detail['tax_sign_flips'] or '无'}）"
    )
    lines.append("")
    comp = m3["composition"]
    lines.append(f"构成敏感性（描述性，不入 M3；S={comp['n_strata']} 非空层）：")
    lines.append("")
    lines.append(f"方案 (a) {comp['scheme_a'].get('weight_rule', '各非空层等权')}：")
    lines.append(_scheme_md(comp["scheme_a"]))
    lines.append("")
    sb = comp["scheme_b"]
    if sb.get("status") == "MISSING":
        lines.append(f"方案 (b)：MISSING（{sb['reason']}）")
    else:
        lines.append(f"方案 (b) {sb.get('weight_rule', '395 集边际映射')}：")
        lines.append(_scheme_md(sb))
    lines.append("")
    lines.append(f"方案 (b) 映射注：{comp['mapping_note']}")
    lines.append("")
    c2 = m3["c2kv"]
    if c2["status"] == "AVAILABLE" and c2.get("n_pairs"):
        t = c2["tax_full_minus_c2kv"]
        lines.append(
            f"C2KV 描述性（INTERNAL-ONLY，{c2['n_pairs']}/89 配对行）："
            f"税(full−c2kv) protocol {_fmt(t['protocol']['value'])}（{t['protocol']['sign']}），"
            f"semantic {_fmt(t['semantic']['value'])}（{t['semantic']['sign']}）"
        )
    else:
        lines.append("C2KV 描述性：MISSING（--closeout_full / --closeout_c2kv 缺省或无可配对行）")
    lines.append("")
    return "\n".join(lines)


def render_md(result: dict) -> str:
    parts = [
        _render_m1(result["M1"]),
        _render_m2(result["M2"]),
        _render_m3(result["M3"]),
    ]
    return "\n".join(parts)


def analyze(args) -> dict:
    scored_path = Path(args.scored)
    sample_path = Path(args.sample)
    scored = load_jsonl(scored_path)
    sample = load_json(sample_path)

    input_sha = {
        "scored": sha256_of(scored_path),
        "sample": sha256_of(sample_path),
    }

    runs_rows = None
    if args.runs_dir:
        runs_rows, n_run_dups = load_runs_rows(args.runs_dir)
        digest, per_file = runs_dir_digest(Path(args.runs_dir))
        input_sha["runs_dir_combined"] = digest
    else:
        n_run_dups = 0
        input_sha["runs_dir_combined"] = None

    closeout_full = None
    closeout_c2kv = None
    if args.closeout_full:
        closeout_full = load_jsonl(Path(args.closeout_full))
        input_sha["closeout_full"] = sha256_of(Path(args.closeout_full))
    if args.closeout_c2kv:
        closeout_c2kv = load_jsonl(Path(args.closeout_c2kv))
        input_sha["closeout_c2kv"] = sha256_of(Path(args.closeout_c2kv))

    v1_obj = None
    if args.v1_stratified:
        v1_obj = load_json(Path(args.v1_stratified))
        input_sha["v1_stratified"] = sha256_of(Path(args.v1_stratified))

    m1 = compute_m1(scored, sample, runs_rows, input_sha)
    m2 = compute_m2(scored, sample, closeout_full, closeout_c2kv, input_sha)
    m3 = compute_m3(scored, sample, closeout_full, closeout_c2kv, v1_obj, input_sha)

    result = {
        "meta": {
            "task": "S8 chunk B M1–M3 统计分析",
            "inputs": {
                "scored": str(scored_path),
                "sample": str(sample_path),
                "runs_dir": str(args.runs_dir) if args.runs_dir else None,
                "closeout_full": str(args.closeout_full) if args.closeout_full else None,
                "closeout_c2kv": str(args.closeout_c2kv) if args.closeout_c2kv else None,
                "v1_stratified": str(args.v1_stratified) if args.v1_stratified else None,
            },
            "input_sha256": input_sha,
            "n_rows_scored": len(scored),
            "n_runs_duplicate_keys_kept_first": n_run_dups,
            "judgments": {
                "M1": m1["judgment"],
                "M2": m2["judgment"],
                "M3": m3["judgment"],
            },
        },
        "M1": m1,
        "M2": m2,
        "M3": m3,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[analyze] 输出 -> {out_path}", file=sys.stderr)

    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m metrology.analyze_s8",
        description="S8 chunk B：M1–M3 统计分析（scored jsonl + 冻结样本清单）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scored", required=True, help="chunk A 产出的 scored jsonl 路径")
    p.add_argument("--sample", required=True,
                   help="冻结样本清单（configs/r5_metrology_sample.json）")
    p.add_argument("--runs_dir", default=None,
                   help="可选：原始 runner 输出目录（仅用于首个分叉轮统计）")
    p.add_argument("--closeout_full", default=None,
                   help="可选：S6 内部 scored（full 臂，INTERNAL-ONLY）")
    p.add_argument("--closeout_c2kv", default=None,
                   help="可选：S6 内部 scored（c2kv 臂，INTERNAL-ONLY）")
    p.add_argument("--v1_stratified", default=None,
                   help="可选：results/r5/analysis/v1_stratified_strict.json（重加权方案 b）")
    p.add_argument("--out", required=True, help="全部数字的输出 json 路径")
    return p


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非必需能力，失败忽略
        pass
    args = build_parser().parse_args(argv)
    result = analyze(args)
    print(render_md(result))


if __name__ == "__main__":
    main()
