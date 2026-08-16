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
