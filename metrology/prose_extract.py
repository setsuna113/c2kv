# -*- coding: utf-8 -*-
r"""S8 chunk A：规则式散文抽取器（冻结规则实现，configs/r5_metrology_prereg.md §5 逐字）。

冻结规则：
  ① 函数名词典 = 该样本 available functions 的函数名集合；生成文本中首个出现在
     词典内的函数名即语义函数名（按文本出现位置取最先）；
  ② 参数键值对：在函数名后 2000 字符窗口内以正则
     `"?(\w+)"?\s*[:=]\s*("([^"]*)"|\d+(\.\d+)?|true|false|null)` 抽取键值对，
     键集合 ∩ 金标参数键集合非空即记参数命中；
  ③ 语义判对 = 函数名命中 AND（金标无参数 OR 参数命中）。

本模块纯 Python，不 import bfcl_eval；公开函数只有 build_gold_param_keys 与
extract_semantic 两个（S8 chunk A 任务书冻结）。

实现口径（操作化决定 10）：
- 词典匹配用词边界（(?<!\w) / (?!\w)，逐个 re.escape），按文本出现位置取最先；
  同位置并列时取名字更长者（alternation 按名字长度降序排列）。
- 检索与窗口都以字符偏移计（Python str 索引）。窗口 = 函数名出现位置起 2000
  字符（含名字本身起算，即 text[name_pos : name_pos + 2000]）；键值对按「整体
  起始位置」判定是否在窗口内（起始 < name_pos + 2000 计入，起始 >= 则不计）。
    实现上在全文检索、按起始偏移过滤，与切片等价，且覆盖跨窗口右界的键值对。

v2（勘误修订）：extract_semantic_v2 为 30 例人工复核触发的新实现（金标函数名
词典 + 全覆盖），见函数 docstring；v1（extract_semantic）冻结保留原样，两套
实现互不影响。
"""

import re

# 冻结正则原文（prereg §5 逐字，不做任何修改）
_FROZEN_KV_RE = re.compile(
    r'"?(\w+)"?\s*[:=]\s*("([^"]*)"|\d+(\.\d+)?|true|false|null)'
)

# 冻结窗口长度（prereg §5：函数名后 2000 字符）
_WINDOW_CHARS = 2000


def build_gold_param_keys(gold_calls: list[dict]) -> set:
    """金标参数键并集。

    gold call 形如 {"func_name": {"param": [候选值...]}}（BFCL 单轮类
    possible_answer 的 ground_truth 元素格式）；返回全部参数键的并集。
    并集为空即「金标无参数」。
    """
    keys = set()
    for call in gold_calls:
        for params in call.values():
            if isinstance(params, dict):
                keys.update(params.keys())
    return keys


def extract_semantic(text: str, name_dict: set, gold_param_keys: set) -> dict:
    """按冻结规则抽取语义面。

    返回（键序固定）：
      {"name_hit": bool, "name": str|None, "name_pos": int|None,
       "param_keys": list[str], "param_hit": bool,
       "gold_no_params": bool, "correct": bool}
    """
    gold_no_params = len(gold_param_keys) == 0
    result = {
        "name_hit": False,
        "name": None,
        "name_pos": None,
        "param_keys": [],
        "param_hit": False,
        "gold_no_params": gold_no_params,
        "correct": False,
    }
    if not text or not name_dict:
        return result

    # 名字按长度降序排列：同位置并列时 alternation 优先取更长者（决策 10）
    names = sorted({n for n in name_dict if n}, key=len, reverse=True)
    if not names:
        return result
    name_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)"
    )
    match = name_pattern.search(text)
    if match is None:
        return result

    name_pos = match.start()
    result["name_hit"] = True
    result["name"] = match.group(0)
    result["name_pos"] = name_pos

    # 窗口 [name_pos, name_pos + 2000)：键值对整体起始于窗口内才计入。
    # 全文检索 + 起始偏移过滤（见模块 docstring 实现口径）。
    window_end = name_pos + _WINDOW_CHARS
    param_keys = []
    for kv in _FROZEN_KV_RE.finditer(text):
        if kv.start() < name_pos or kv.start() >= window_end:
            continue
        key = kv.group(1)
        if key not in param_keys:
            param_keys.append(key)
    result["param_keys"] = param_keys
    result["param_hit"] = any(k in gold_param_keys for k in param_keys)
    result["correct"] = gold_no_params or result["param_hit"]
    return result


# ══════════════════════════════════════════════════════════════════════════
# v2（勘误修订）：金标函数名词典 + 全覆盖。v1 冻结保留原样，本节为新增实现。
# ══════════════════════════════════════════════════════════════════════════

# v2 调用形态窗口：名字出现位置起 200 字符内出现调用分隔符即算「以调用形态出现」
#（窗口口径与 v1 的 2000 字符窗口一致：含名字本身起算）。
_V2_CALL_FORM_WINDOW = 200
_V2_CALL_FORM_CHARS = ("(", "[", "{", ":")


def _v2_last_segment(name: str) -> str:
    """函数名取最后一段 `.` 后文本（TradingBot.post_tweet → post_tweet）。"""
    return name.rsplit(".", 1)[-1]


def extract_semantic_v2(text: str, name_dict: set, gold_param_keys: set) -> dict:
    """v2 规则式散文抽取（勘误修订，erratum 后报告逐字引用本 docstring 规则）。

    规则：
      ① 函数名词典 = 金标函数名集合（missed_function 豁免减法由调用方在入参前
         完成）；
      ② 全覆盖：词典中每个函数名都要在文本中以调用形态出现——名字的最后一段
         `.` 后文本按 v1 同款词边界命中，且名字出现位置起 200 字符内有
         `(` / `[` / `{` / `:` 之一；任一缺失 → incorrect；
      ③ 参数规则与 v1 相同：首个命中名位置起 2000 字符窗口内正则抽取键值对，
         键集合 ∩ 金标参数键集合非空即命中；金标无参数豁免参数要求。
      correct = 全覆盖 AND（金标无参数 OR 参数命中）。

    返回（键序固定）：
      {"version": 2, "name_hit": bool, "name": str|None, "name_pos": int|None,
       "missing_names": list[str], "coverage_ok": bool,
       "param_keys": list[str], "param_hit": bool,
       "gold_no_params": bool, "correct": bool}

    实现口径（操作化决定）：
    - 词典名按（长度降序，名字升序）确定性排序；锚点（首个命中名）取全文最先
      出现位置，并列时取更长者（与 v1 决策 10 同口径）。
    - 同名最后一段的多个词典名按文本位置从左到右各消费一个出现（保守方向）。
    - 调用形态窗口与参数窗口均含名字本身起算（与 v1 窗口口径一致）。
    """
    gold_no_params = len(gold_param_keys) == 0
    result = {
        "version": 2,
        "name_hit": False,
        "name": None,
        "name_pos": None,
        "missing_names": [],
        "coverage_ok": False,
        "param_keys": [],
        "param_hit": False,
        "gold_no_params": gold_no_params,
        "correct": False,
    }

    names = sorted({n for n in name_dict if n}, key=lambda n: (-len(n), n))
    if not names:
        # 词典为空（全部豁免）：覆盖要求空真；参数规则无锚点。
        result["coverage_ok"] = True
        result["correct"] = gold_no_params
        return result
    if not text:
        result["missing_names"] = names
        return result

    # 锚点（首个命中名）：全部名字最后一段的词边界交替匹配，取全文最先位置。
    segments = sorted(
        {_v2_last_segment(n) for n in names if _v2_last_segment(n)},
        key=lambda s: (-len(s), s),
    )
    anchor = None
    if segments:
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(s) for s in segments) + r")(?!\w)"
        )
        anchor = pattern.search(text)
    if anchor is not None:
        anchor_segment = anchor.group(0)
        anchor_name = next(
            n for n in names if _v2_last_segment(n) == anchor_segment
        )
        result["name_hit"] = True
        result["name"] = anchor_name
        result["name_pos"] = anchor.start()

    # 全覆盖：每个词典名需要自己的调用形态出现（词边界命中 + 200 字符内有调用
    # 分隔符）。同名最后一段的多个词典名从左到右各消费一个出现位置。
    used_positions: set = set()
    for name in names:
        segment = _v2_last_segment(name)
        if not segment:
            result["missing_names"].append(name)
            continue
        seg_pattern = re.compile(r"(?<!\w)" + re.escape(segment) + r"(?!\w)")
        covered = False
        for m in seg_pattern.finditer(text):
            pos = m.start()
            if pos in used_positions:
                continue
            if any(
                c in text[pos:pos + _V2_CALL_FORM_WINDOW]
                for c in _V2_CALL_FORM_CHARS
            ):
                used_positions.add(pos)
                covered = True
                break
        if not covered:
            result["missing_names"].append(name)
    result["coverage_ok"] = not result["missing_names"]

    # 参数规则（与 v1 相同）：锚点位置起 2000 字符窗口。
    if result["name_hit"]:
        window_end = result["name_pos"] + _WINDOW_CHARS
        param_keys = []
        for kv in _FROZEN_KV_RE.finditer(text):
            if kv.start() < result["name_pos"] or kv.start() >= window_end:
                continue
            key = kv.group(1)
            if key not in param_keys:
                param_keys.append(key)
        result["param_keys"] = param_keys
        result["param_hit"] = any(k in gold_param_keys for k in param_keys)

    result["correct"] = result["coverage_ok"] and (
        gold_no_params or result["param_hit"]
    )
    return result
