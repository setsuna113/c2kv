# -*- coding: utf-8 -*-
r"""S8 chunk A：BFCL v4 离线评分器（prose 抽取器 + 原生评分接线）。

对 metrology/bfcl_hf_runner 落盘的推理行（jsonl）做离线双列评分，为后续 chunk 的
M1/M2/M3 统计提供逐行标记：

- native_valid / native_error_type：基准原生严格评分。不复刻任何逐条逻辑——
  按 (condition, cap_tier, category) 分组，每组把行内 bfcl_result 组织成
  model_result 列表（与 prompt/possible_answer 位置 zip 对齐），直接调
  eval_runner.multi_turn_runner（multi_turn_* 四类）或 eval_runner.ast_file_runner
  （parallel / parallel_multiple）；runner 返回聚合 accuracy，错误明细写在
  score_dir 下的 score 文件（首行 header，其后每条错误明细含 id / error_type），
  逐行 native_valid = id 不在错误明细中。推理异常行（bfcl_result["result"] 为
  str）按原生语义自然落 valid=False（multi_turn 的 type!=list 分支 / ast 解码
  失败分支），不做特殊处理。
- protocol_valid / n_protocol_invalid_steps / step_protocol_valid：协议面
  （操作化决定 2/3）：multi_turn 逐 step 看行内存储的 decoded_calls（列表且非空
  响应）；单轮看行内存储的 decoded_ast（非异常字符串且函数调用格式）。
- prose：规则式散文抽取器 v2（metrology/prose_extract.py extract_semantic_v2，
  勘误修订：金标函数名词典 + 全覆盖，词典 = possible_answer gold calls 去重函数名
  减去 missed_function 豁免集）；prose_v1_frozen 为冻结 v1 结果原样保留
  （可用函数词典，extract_semantic）。
- semantic_correct = native_valid OR prose(v2).correct；split_row = semantic_correct
  AND (NOT protocol_valid)（操作化决定 5）；semantic_correct_v1 / split_row_v1 为
  v1 参照列（同式代入 prose_v1_frozen.correct）。
- censored：任一步 stop_reason == "length"（操作化决定 6）。
- runner_error：行内 error 非 null → 按操作化决定 7 全判 False（native 除外，
  native 仍走原生 runner 取 verdict）。
- summary json：除每格统计外注明 extractor version=2 与勘误（erratum）一句。

原生评分的 decode_execute / decode_ast / is_empty_execute_response /
is_function_calling_format_output / multi_turn_checker / ast_checker / 各 file
runner 一律复用 bfcl_eval 原实现（不重写）；handler 用
eval_runner.get_handler(MODEL_NAME) 从真实 MODEL_CONFIG_MAPPING 构造
（QwenHandler，is_fc_model=False）。本模块只做接线与逐行组装。纯 CPU，
不 import torch。

用法（仓库根）：
  python -m metrology.bfcl_score \
    --bfcl_pkg_path .foreman/ref/bfcl_pkg \
    --bfcl_data_dir .foreman/ref/bfcl_data \
    --runs_dir metrology/data/smoke_fixtures \
    --out <scored.jsonl> [--summary_out <summary.json>]
"""

from __future__ import annotations

import argparse
import ast as py_ast
import importlib
import importlib.abc
import json
import os
import re
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path

from metrology.bfcl_hf_runner import (
    _install_import_stubs,
    _inject_bfcl_syspath,
    _patch_load_file_utf8,
)
from metrology.prose_extract import (
    build_gold_param_keys,
    extract_semantic,
    extract_semantic_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# 与 runner 的 handler 注册名一致；model_name 字符串同时是 checker 实例缓存键
# 的组成部分（multi_turn_utils 的 globals 键），保持一致即可。
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# scored 行 prose 子对象的键序（prose_extract.extract_semantic 返回键序，
# prose_v1_frozen 用）
_PROSE_KEYS = [
    "name_hit", "name", "name_pos", "param_keys", "param_hit",
    "gold_no_params", "correct",
]

# scored 行 prose 子对象的键序（prose_extract.extract_semantic_v2 返回键序）
_PROSE_V2_KEYS = [
    "version", "name_hit", "name", "name_pos", "missing_names", "coverage_ok",
    "param_keys", "param_hit", "gold_no_params", "correct",
]

# runner error 行的 prose 全 False（操作化决定 7）
_PROSE_ALL_FALSE = {
    "name_hit": False, "name": None, "name_pos": None, "param_keys": [],
    "param_hit": False, "gold_no_params": False, "correct": False,
}

# runner error 行的 prose v2 全 False（操作化决定 7）
_PROSE_V2_ALL_FALSE = {
    "version": 2, "name_hit": False, "name": None, "name_pos": None,
    "missing_names": [], "coverage_ok": False, "param_keys": [],
    "param_hit": False, "gold_no_params": False, "correct": False,
}

# summary json 的勘误注明（extractor version=2 + erratum 一句）
_EXTRACTOR_ERRATUM = (
    "v2: gold-name dictionary + full coverage, erratum after 17/30 review"
)


# ══════════════════════════════════════════════════════════════════════════
# BFCL 环境准备（进程内只做一次；复用 runner 的 stub/syspath/utf-8 三件套 +
# 厂商 SDK 惰性 dummy 兜底，让真实 model_config 注册表可导入）
# ══════════════════════════════════════════════════════════════════════════

# model_config.py 顶层 import 全部厂商 handler，其中这些厂商 SDK（或其子模块）
# 本地未安装。真实包优先（PathFinder 先于兜底 finder），缺包时才提供 dummy。
# openai / requests / tenacity 同时在上游显式 stub 名单里：真实包缺席时顶层模块
# 由显式 stub 提供（其子模块 tenacity.stop / openai.types.responses 等由本 finder
# 兜底；先经 _fixup_runner_stubs 把手工 stub 设为包才能到达本 finder）。
_VENDOR_ROOTS = {
    "anthropic", "boto3", "bs4", "cohere", "datamodel_code_generator",
    "dotenv", "faiss", "google", "html2text", "mistralai", "numpy",
    "openai", "pandas", "qwen_agent", "rank_bm25", "requests",
    "sentence_transformers", "serpapi", "tenacity", "tqdm", "writerai",
}


class _VendorAny:
    """厂商 SDK 占位对象：可调用（如 load_dotenv 恒等）、可实例化（如 OpenAI
    式 client 构造）、任意属性访问返回新的占位对象。只用于满足 import 面；
    真被调用只发生在评分路径从不进入的分支（评分只走 decode_* 纯文本方法）。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _VendorAny()

    def __getattr__(self, name):
        return _VendorAny()


class _VendorModule(types.ModuleType):
    """dummy 厂商模块。__file__ / __cached__ / __doc__ 预置为标准值：torch 等
    库的 inspect 路径会遍历 sys.modules 触碰 __file__（inspect.getsourcefile →
    os.path.splitext），permissive __getattr__ 若返回占位对象会让其崩溃；预置
    字符串/None 后 dummy 模块在这些路径下行为与正常模块一致。"""

    def __init__(self, name: str):
        super().__init__(name)
        self.__file__ = f"<bfcl_score vendor dummy module '{name}'>"
        self.__cached__ = None
        self.__doc__ = None

    def __getattr__(self, name):
        return _VendorAny()


class _VendorLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _VendorModule(spec.name)
        mod.__path__ = []  # 支持 google.genai 等子模块路径
        return mod

    def exec_module(self, module):
        pass


class _VendorImportFinder(importlib.abc.MetaPathFinder):
    """对 _VENDOR_ROOTS 顶层名兜底：真实安装的包由 PathFinder 先行命中，缺包时
    本 finder 提供可导入的 dummy 包。find_spec 按包处理（is_package=True），
    子模块（google.genai 等）继续经本 finder 以 dummy 提供。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in _VENDOR_ROOTS:
            return None
        if fullname in sys.modules and sys.modules[fullname] is not None:
            return sys.modules[fullname].__spec__
        return importlib.machinery.ModuleSpec(
            fullname, _VendorLoader(), is_package=True
        )


def _install_vendor_import_stubs() -> None:
    """把 _VendorImportFinder 挂到 sys.meta_path 末尾（幂等：已挂则跳过）。"""
    for finder in sys.meta_path:
        if isinstance(finder, _VendorImportFinder):
            return
    sys.meta_path.append(_VendorImportFinder())
    print(f"[setup] 厂商 SDK 兜底 finder 已挂载（{len(_VENDOR_ROOTS)} 个顶层名，真实包优先）")


# runner（bfcl_hf_runner）_install_import_stubs 覆盖的 8 个显式 stub 目标名。
# runner 文件必须保持冻结（跑批 manifest 钉了 sha256），其手工 stub 的缺陷只能在
# scorer 侧 post-fixup 修复，不改 runner 源码。
_RUNNER_STUB_TARGETS = (
    "tree_sitter", "tree_sitter_java", "tree_sitter_javascript",
    "openai", "overrides", "tenacity", "filelock", "requests",
)


def _is_manual_stub(name: str) -> bool:
    """runner 的 _make_module 手工 stub 没有 __spec__（真实包 import 后必有），
    以此作为「手工 stub」特征。真实包（__spec__ 非 None）不动。"""
    mod = sys.modules.get(name)
    return mod is not None and getattr(mod, "__spec__", None) is None


def _reinstall_enhanced_tenacity() -> None:
    """runner 旧 tenacity stub 的 retry_if_* 返回 None，model_handler/utils.py 顶层
    retry_with_backoff 在类定义装饰期做 reduce(operator.or_, conditions)，None | None
    直接崩。此处用可 or 组合的空条件对象重新注册一个改进版 tenacity：
    retry 恒等装饰器 + retry_if_exception_type/message 返回可 or 对象 +
    wait_random_exponential 占位；其余字段行为不变。"""
    class _RetryCondition:
        def __or__(self, other):
            return self

        def __ror__(self, other):
            return self

    def _retry(*args, **kwargs):
        def _dec(fn):
            return fn
        return _dec

    def _retry_if_exception_type(*args, **kwargs):
        return _RetryCondition()

    def _retry_if_exception_message(*args, **kwargs):
        return _RetryCondition()

    def _noop(*args, **kwargs):
        return None

    mod = types.ModuleType("tenacity")
    mod.__dict__.update(
        {
            "retry": _retry,
            "wait_random_exponential": _noop,
            "retry_if_exception_message": _retry_if_exception_message,
            "retry_if_exception_type": _retry_if_exception_type,
        }
    )
    mod.__path__ = []
    sys.modules["tenacity"] = mod
    print("[setup] stub 修复: tenacity 已重注册为增强版（retry_if_* 可 or 组合）")


def _fixup_runner_stubs() -> None:
    """post-fixup：runner 的 8 个显式 stub（真实包缺席时）都是 _make_module 手工
    模块（__spec__ is None）。把它们设为包（__path__=[]）——否则
    `import tenacity.stop` / `import openai.types.responses` 之类子模块 import 会在
    meta_path 之前就被 "not a package" 挡住，_VendorImportFinder 接不到。
    tenacity 另外重注册增强版（见 _reinstall_enhanced_tenacity）；openai 等其他
    stub 无顶层求值问题（OpenAI 只在 __init__ 内构造），仅设 __path__。"""
    for name in _RUNNER_STUB_TARGETS:
        if not _is_manual_stub(name):
            continue
        mod = sys.modules[name]
        if not getattr(mod, "__path__", None):
            mod.__path__ = []
        print(f"[setup] stub 修复: {name} 设为包（__spec__=None 手工 stub，子模块走兜底 finder）")
    if _is_manual_stub("tenacity"):
        _reinstall_enhanced_tenacity()


_setup_state: dict = {"done": False, "bfcl_dir": None}


def _setup_bfcl(args) -> Path:
    """进程内只允许一次 bfcl setup（幂等）：安装依赖 stub → 厂商 SDK 兜底 finder
    → runner 手工 stub post-fixup → BFCL_PROJECT_ROOT 重定向（必须在 import
    bfcl_eval 之前）→ sys.path 注入 → import bfcl_eval → 数据目录重定向 →
    load_file utf-8 补丁。返回解析后的 bfcl_eval 包目录。第二次调用直接返回缓存的
    bfcl_dir。"""
    if _setup_state["done"]:
        return _setup_state["bfcl_dir"]

    stub_status = _install_import_stubs(
        force_stub=bool(os.getenv("METROLOGY_FORCE_STUBS", ""))
    )
    for name, st in sorted(stub_status.items()):
        print(f"[setup] import 面: {name:24s} -> {st}")
    _install_vendor_import_stubs()
    _fixup_runner_stubs()

    # eval_config.py 在 import 时按 BFCL_PROJECT_ROOT 建 result/score/.file_locks
    # 目录；重定向到临时目录，不得污染仓库（任务书要求 mkdtemp 前缀 bfcl_score_）。
    os.environ.setdefault("BFCL_PROJECT_ROOT", tempfile.mkdtemp(prefix="bfcl_score_"))

    bfcl_dir = _inject_bfcl_syspath(args.bfcl_pkg_path)

    import bfcl_eval  # noqa: F401  确认包可导入
    from bfcl_eval.constants import eval_config

    assert str(Path(bfcl_eval.__file__).resolve().parent) == str(bfcl_dir), (
        "sys.path 注入未生效：import 到的 bfcl_eval 不在 --bfcl_pkg_path 下"
    )

    data_dir = Path(args.bfcl_data_dir).resolve()
    eval_config.PROMPT_PATH = data_dir
    eval_config.MULTI_TURN_FUNC_DOC_PATH = data_dir / "multi_turn_func_doc"
    eval_config.POSSIBLE_ANSWER_PATH = data_dir / "possible_answer"
    print(f"[setup] BFCL 数据目录已重定向: {data_dir}")

    _patch_load_file_utf8()

    _setup_state.update(done=True, bfcl_dir=bfcl_dir)
    return bfcl_dir


# ══════════════════════════════════════════════════════════════════════════
# 数据加载与每样本上下文（词典 / 金标键）
# ══════════════════════════════════════════════════════════════════════════

def _load_data(categories: set) -> tuple[dict, dict]:
    """加载 prompt 与 possible_answer。返回 (prompt_by_id, answer_by_id)。

    与原生评分流水线同源：prompt 走 load_dataset_entry(cat, include_prereq=False,
    include_language_specific_hint=False)（eval_runner.py:683-685 同参）；答案走
    load_ground_truth_entry（eval_runner.py:698）。"""
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    prompt_by_id: dict = {}
    answer_by_id: dict = {}
    for cat in sorted(categories):
        entries = load_dataset_entry(
            cat, include_prereq=False, include_language_specific_hint=False
        )
        answers = load_ground_truth_entry(cat)
        prompt_by_id.update({e["id"]: e for e in entries})
        answer_by_id.update({a["id"]: a for a in answers})
        print(f"[data] {cat}: prompt {len(entries)} 条 / answer {len(answers)} 条")
    return prompt_by_id, answer_by_id


def _load_missed_functions(bfcl_data_dir: str, categories: set) -> dict:
    """missed_function 豁免集（勘误修订）：读 bfcl_data_dir 下 BFCL_v4_*.json
    原始 prompt 文件（不经过 load_dataset_entry，避开 populate 对
    missed_function 字段的原地改写）。

    missed_function = {轮次字符串: [函数名]}，拉平为集合；无该字段的样本
    （非 multi_turn 类）豁免集为空。返回 {id: 豁免函数名集合}。"""
    missed: dict = {}
    for cat in sorted(categories):
        p = Path(bfcl_data_dir) / f"BFCL_v4_{cat}.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mf = row.get("missed_function") or {}
                if not mf:
                    continue
                names: set = set()
                for values in mf.values():
                    if isinstance(values, list):
                        names.update(str(v) for v in values)
                if names:
                    missed[str(row["id"])] = names
    return missed


def _call_str_param_keys(call_str: str) -> set:
    r"""从 multi_turn 金标调用串（python 语法，如
    "estimate_distance(cityA='94016', cityB='83214')"）取最外层调用的关键字参数
    键集合。解析失败时退化用 `(\w+)\s*=` 兜底（金标串应为良构 python）。"""
    try:
        node = py_ast.parse(call_str, mode="eval").body
        if isinstance(node, py_ast.Call):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    except Exception:  # noqa: BLE001 金标串理论上恒可解析，兜底仅作防御
        pass
    return {m.group(1) for m in re.finditer(r"(\w+)\s*=", call_str)}


def _multi_turn_gold_param_keys(ground_truth: list) -> set:
    """multi_turn 金标参数键集合 = possible_answer 全部轮全部 gold call 的参数键
    并集（操作化决定 8）。并集为空即「金标无参数」。"""
    keys: set = set()
    for turn_calls in ground_truth:
        for call_str in turn_calls:
            keys.update(_call_str_param_keys(call_str))
    return keys


def _call_str_name(call_str: str) -> str | None:
    """multi_turn 金标调用串 → 最外层调用函数名（含类前缀完整路径，如
    "TradingBot.post_tweet"）。解析失败退化正则（金标串应为良构 python）。"""
    try:
        node = py_ast.parse(call_str, mode="eval").body
        if isinstance(node, py_ast.Call):
            return py_ast.unparse(node.func)
    except Exception:  # noqa: BLE001 金标串理论上恒可解析，兜底仅作防御
        pass
    m = re.match(r"\s*([\w.]+)\s*\(", call_str)
    return m.group(1) if m else None


def _gold_names(ground_truth: list, is_multi: bool) -> set:
    """金标函数名集合（v2 词典来源，与 v1 参数键同源 possible_answer）：
    multi_turn 各轮 gold 调用串去重函数名（含类前缀）；单轮类取 gold call 的
    键名。"""
    names: set = set()
    if is_multi:
        for turn_calls in ground_truth:
            for call_str in turn_calls:
                name = _call_str_name(str(call_str))
                if name:
                    names.add(name)
    else:
        for call in ground_truth:
            for fname in call.keys():
                if fname:
                    names.add(fname)
    return names


def _entry_context(prompt_entry: dict, answer_item: dict,
                   missed_names: set) -> dict:
    """每样本一次性算好：is_multi、v1 函数名词典（决策 9）、v2 金标函数名词典
    （勘误修订：possible_answer gold calls 去重函数名 − missed_function 豁免集）、
    金标参数键集合（决策 8）。

    multi_turn 词典：load_dataset_entry 内部已应用
    populate_test_cases_with_predefined_functions（utils.py:437，与推理期注入同一
    逻辑；miss_func 的 holdout 文档在 populate 中已移出 function 列表），故
    entry["function"] 即推理期注入的文档列表，取 name 集合。
    单轮类词典：prompt entry 顶层 function 字段的 name 集合。"""
    from bfcl_eval.utils import contain_multi_turn_interaction

    is_multi = contain_multi_turn_interaction(prompt_entry["id"])
    name_dict = {f["name"] for f in prompt_entry.get("function", []) if f.get("name")}
    if is_multi:
        gold_param_keys = _multi_turn_gold_param_keys(answer_item["ground_truth"])
    else:
        gold_param_keys = build_gold_param_keys(answer_item["ground_truth"])
    v2_name_dict = _gold_names(answer_item["ground_truth"], is_multi) - missed_names
    return {
        "is_multi": is_multi,
        "name_dict": name_dict,
        "v2_name_dict": v2_name_dict,
        "gold_param_keys": gold_param_keys,
        "ground_truth": answer_item["ground_truth"],
    }


# ══════════════════════════════════════════════════════════════════════════
# 原生评分（直接调 eval_runner 的 file runner；工具函数全部复用 bfcl_eval 原实现）
# ══════════════════════════════════════════════════════════════════════════

def _run_native_group(handler, group_rows: list[tuple[dict, str]],
                      prompt_by_id: dict, answer_by_id: dict) -> dict:
    """对 (condition, cap_tier, category) 一格的行调用原生 file runner。

    - model_result = [row["bfcl_result"] for rows of that cell]；prompt /
      possible_answer 列表按 model_result 的 id 逐行对齐（runner 按位置 zip，
      第 i 个位置对应同一行）。
    - multi_turn_* 调 multi_turn_runner，parallel / parallel_multiple 调
      ast_file_runner。
    - score_dir 用独立 tempfile 目录（save_eval_results 会写 score 文件，
      同类别不同格之间不得互相覆盖）；读返回值的 accuracy / total_count 数字。
    - 返回 (verdicts, accuracy)：verdicts = {id: (native_valid, error_type|None)}
      ——runner 只返回聚合 accuracy，错误明细在 score 文件里（首行 header 之后的
      每条含 id / error_type）；native_valid 逐行 = id 不在错误明细中。
    - bfcl_result["result"] 为 str 的推理异常行按原生语义自然落 valid=False
      （multi_turn 的 type!=list 分支 / ast 解码失败分支），无需特殊处理。
    """
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.eval_checker import eval_runner
    from bfcl_eval.utils import get_directory_structure_by_category, is_multi_turn

    model_result = [row["bfcl_result"] for row, _src in group_rows]
    prompt_list = [deepcopy(prompt_by_id[row["id"]]) for row, _src in group_rows]
    answer_list = [deepcopy(answer_by_id[row["id"]]) for row, _src in group_rows]
    category = group_rows[0][0]["category"]

    score_dir = Path(tempfile.mkdtemp(prefix="bfcl_score_dir_"))
    if is_multi_turn(category):
        accuracy, total_count = eval_runner.multi_turn_runner(
            handler, model_result, prompt_list, answer_list,
            MODEL_NAME, category, score_dir,
        )
    else:
        accuracy, total_count = eval_runner.ast_file_runner(
            handler, model_result, prompt_list, answer_list,
            category, MODEL_NAME, score_dir,
        )
    assert total_count == len(model_result), (
        f"runner 返回 total_count={total_count} 与输入 {len(model_result)} 行不一致"
    )

    score_file = (
        score_dir / MODEL_NAME / get_directory_structure_by_category(category)
        / f"{VERSION_PREFIX}_{category}_score.json"
    )
    with open(score_file, encoding="utf-8") as f:
        file_lines = [json.loads(line) for line in f if line.strip()]
    header, error_entries = file_lines[0], file_lines[1:]
    assert header.get("total_count") == len(model_result), (
        f"score 文件 header 与输入行数不一致: {header}"
    )

    verdicts: dict = {}
    for entry in error_entries:
        entry_id = entry["id"]
        error_type = entry.get("error_type")
        if error_type is None and isinstance(entry.get("error"), dict):
            error_type = entry["error"].get("error_type")
        verdicts[entry_id] = (False, error_type)
    for row, _src in group_rows:
        verdicts.setdefault(row["id"], (True, None))
    return verdicts, accuracy


# ══════════════════════════════════════════════════════════════════════════
# 协议面 / prose / 逐行组装
# ══════════════════════════════════════════════════════════════════════════

def _protocol(row: dict, is_multi: bool) -> tuple[bool, int, list]:
    """协议面（操作化决定 2/3）：
    multi_turn 逐 step：行内存储的 decoded_calls 是列表且经
    is_empty_execute_response 判定非空（异常类名字符串 → 无效）；
    单轮：存储的 decoded_ast 非异常字符串且 is_function_calling_format_output。"""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        is_empty_execute_response,
    )
    from bfcl_eval.utils import is_function_calling_format_output

    step_valid: list[bool] = []
    for turn in row["turns"]:
        for step in turn["steps"]:
            if is_multi:
                calls = step.get("decoded_calls")
                valid = isinstance(calls, list) and not is_empty_execute_response(calls)
            else:
                ast_out = step.get("decoded_ast")
                valid = isinstance(ast_out, list) and is_function_calling_format_output(
                    ast_out
                )
            step_valid.append(valid)
    n_invalid = sum(1 for v in step_valid if not v)
    return all(step_valid), n_invalid, step_valid


def _prose_text(row: dict, is_multi: bool) -> str:
    """散文文本（操作化决定 1）：multi_turn = 所有 step 的 parsed_text 按序
    "\n" 拼接；单轮 = 唯一 step 的 parsed_text。"""
    if is_multi:
        return "\n".join(
            s["parsed_text"] for t in row["turns"] for s in t["steps"]
        )
    return row["turns"][0]["steps"][0]["parsed_text"]


def _prose_pair(row: dict, is_multi: bool, ctx: dict) -> tuple[dict, dict]:
    """散文抽取双轨：返回 (v1 冻结结果, v2 结果)。

    v1 词典 = 可用函数集合（冻结参照）；v2 词典 = 金标函数名集合（豁免后）。
    两者共用同一文本与金标参数键集合。"""
    text = _prose_text(row, is_multi)
    v1 = extract_semantic(text, ctx["name_dict"], ctx["gold_param_keys"])
    v2 = extract_semantic_v2(text, ctx["v2_name_dict"], ctx["gold_param_keys"])
    return v1, v2


def _cleanup_multi_turn_instances(model_name_underline: str, entry_id: str):
    """multi_turn_checker 把类实例缓存在 multi_turn_utils 模块 globals（键 =
    f"{model_name}_{test_entry_id}_{class}_instance"，eval 运行加 "_eval" 后缀、
    金标运行加 "_ground_truth_eval"）。同一进程内同一样本的不同行（不同 cap /
    condition）必须清掉，否则状态跨行污染（与 runner _cleanup_multi_turn_instances
    同理，但覆盖 eval 期的两类实例）。"""
    mod = sys.modules.get("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    if mod is None:
        return
    for prefix in (
        f"{model_name_underline}_eval_{entry_id}_",
        f"{model_name_underline}_ground_truth_eval_{entry_id}_",
    ):
        for key in [k for k in mod.__dict__ if k.startswith(prefix)]:
            del mod.__dict__[key]


def _score_row(row: dict, prompt_by_id: dict, answer_by_id: dict,
               ctx_cache: dict, native_verdict: tuple,
               missed_by_id: dict) -> dict:
    """把一行推理结果评成一行 scored 记录（键序按任务书）。

    native_valid / native_error_type 取原生 file runner 的逐行 verdict（按
    (condition, cap_tier, category) 格预先算好）。runner_error 行（行内 error
    非 null）除 native 外按操作化决定 7 全判 False；native 仍走原生语义
    （bfcl_result["result"] 为 str → multi_turn type!=list 分支 / ast 解码失败，
    自然 valid=False）。"""
    entry_id = row["id"]
    native_valid, native_error_type = native_verdict
    is_runner_error = bool(row.get("error"))
    if is_runner_error:
        scored = {
            "id": entry_id,
            "category": row.get("category"),
            "condition": row.get("condition"),
            "cap_tier": row.get("cap_tier"),
            "n_turns": 0,
            "n_steps": 0,
            "native_valid": bool(native_valid),
            "native_error_type": native_error_type,
            "protocol_valid": False,
            "n_protocol_invalid_steps": 0,
            "step_protocol_valid": [],
            "prose": dict(_PROSE_V2_ALL_FALSE),
            "prose_v1_frozen": dict(_PROSE_ALL_FALSE),
            "semantic_correct": False,
            "semantic_correct_v1": False,
            "split_row": False,
            "split_row_v1": False,
            "censored": False,
            "runner_error": True,
        }
        return scored

    if entry_id not in ctx_cache:
        prompt_entry = prompt_by_id[entry_id]
        answer_item = answer_by_id[entry_id]
        ctx_cache[entry_id] = (
            prompt_entry,
            _entry_context(prompt_entry, answer_item,
                           missed_by_id.get(entry_id, set())),
        )
    prompt_entry, ctx = ctx_cache[entry_id]
    is_multi = ctx["is_multi"]

    n_turns = len(row["turns"])
    n_steps = sum(len(t["steps"]) for t in row["turns"])
    censored = any(
        s.get("stop_reason") == "length"
        for t in row["turns"] for s in t["steps"]
    )

    protocol_valid, n_protocol_invalid_steps, step_protocol_valid = _protocol(
        row, is_multi
    )
    prose_v1, prose_v2 = _prose_pair(row, is_multi, ctx)

    semantic_correct = bool(native_valid or prose_v2["correct"])
    semantic_correct_v1 = bool(native_valid or prose_v1["correct"])
    split_row = bool(semantic_correct and not protocol_valid)
    split_row_v1 = bool(semantic_correct_v1 and not protocol_valid)

    return {
        "id": entry_id,
        "category": row["category"],
        "condition": row["condition"],
        "cap_tier": row["cap_tier"],
        "n_turns": n_turns,
        "n_steps": n_steps,
        "native_valid": bool(native_valid),
        "native_error_type": native_error_type,
        "protocol_valid": bool(protocol_valid),
        "n_protocol_invalid_steps": int(n_protocol_invalid_steps),
        "step_protocol_valid": step_protocol_valid,
        "prose": {k: prose_v2[k] for k in _PROSE_V2_KEYS},
        "prose_v1_frozen": {k: prose_v1[k] for k in _PROSE_KEYS},
        "semantic_correct": semantic_correct,
        "semantic_correct_v1": semantic_correct_v1,
        "split_row": split_row,
        "split_row_v1": split_row_v1,
        "censored": censored,
        "runner_error": False,
    }


# ══════════════════════════════════════════════════════════════════════════
# 行收集 / 去重 / 汇总 / 主流程
# ══════════════════════════════════════════════════════════════════════════

def _collect_runs(runs_dir: str) -> list[tuple[dict, str]]:
    """--runs_dir 下所有 *.jsonl 都读；返回 [(行, 来源文件名), ...]。"""
    rows: list[tuple[dict, str]] = []
    files = sorted(Path(runs_dir).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"--runs_dir={runs_dir} 下没有 *.jsonl 文件")
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append((json.loads(line), p.name))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{p.name}:{line_no} 不是合法 JSON 行: {e}") from e
    return rows


def _check_duplicate_keys(rows: list[tuple[dict, str]]):
    """行按 (id, cap_tier, condition) 去重，重复即报错（任务书 CLI 规格）。"""
    seen: dict = {}
    duplicates: list = []
    for row, src in rows:
        key = (row["id"], str(row.get("cap_tier")), str(row.get("condition")))
        if key in seen:
            duplicates.append((key, seen[key], src))
        seen[key] = src
    if duplicates:
        lines = ["(id, cap_tier, condition) 重复，拒绝评分（任务书 CLI 规格）："]
        for key, first_src, dup_src in duplicates:
            lines.append(f"  {key}  首次出现于 {first_src}，重复于 {dup_src}")
        raise SystemExit("\n".join(lines))


def _build_summary(scored_rows: list[dict]) -> dict:
    """每 (condition, cap_tier) 一格：{n, native_valid_n, protocol_invalid_n,
    split_n, censored_n}。"""
    summary: dict = {}
    for row in scored_rows:
        cell = summary.setdefault(
            (row["condition"], row["cap_tier"]),
            {"n": 0, "native_valid_n": 0, "protocol_invalid_n": 0,
             "split_n": 0, "censored_n": 0},
        )
        cell["n"] += 1
        cell["native_valid_n"] += int(row["native_valid"])
        cell["protocol_invalid_n"] += int(not row["protocol_valid"])
        cell["split_n"] += int(row["split_row"])
        cell["censored_n"] += int(row["censored"])
    return summary


def _print_summary(summary: dict):
    print("[summary] (condition, cap_tier) 格：")
    for (condition, cap_tier), cell in sorted(summary.items()):
        print(
            f"  condition={condition:14s} cap_tier={cap_tier:8s} "
            f"n={cell['n']} native_valid_n={cell['native_valid_n']} "
            f"protocol_invalid_n={cell['protocol_invalid_n']} "
            f"split_n={cell['split_n']} censored_n={cell['censored_n']}"
        )


def _write_summary_out(summary: dict, summary_out: Path):
    nested: dict = {}
    for (condition, cap_tier), cell in summary.items():
        nested.setdefault(condition, {})[cap_tier] = cell
    payload = {
        "extractor_version": 2,
        "erratum": _EXTRACTOR_ERRATUM,
    }
    payload.update(nested)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(args):
    _setup_bfcl(args)
    from bfcl_eval.eval_checker.eval_runner import get_handler

    handler = get_handler(MODEL_NAME)
    print(f"[setup] handler: {type(handler).__name__}（MODEL_CONFIG_MAPPING 注册，"
          f"is_fc_model=False）")

    rows = _collect_runs(args.runs_dir)
    _check_duplicate_keys(rows)
    print(f"[score] 读入 {len(rows)} 行")

    categories = {row["category"] for row, _ in rows if row.get("category")}
    prompt_by_id, answer_by_id = _load_data(categories)
    missed_by_id = _load_missed_functions(args.bfcl_data_dir, categories)
    print(f"[data] missed_function 豁免集: {len(missed_by_id)} 个样本")
    missing_prompt = {r["id"] for r, _ in rows if r["id"] not in prompt_by_id}
    missing_answer = {r["id"] for r, _ in rows if r["id"] not in answer_by_id}
    if missing_prompt:
        raise SystemExit(f"prompt 数据缺失: {sorted(missing_prompt)}")
    if missing_answer:
        raise SystemExit(f"possible_answer 数据缺失: {sorted(missing_answer)}")

    # 按 (condition, cap_tier, category) 分组，逐组调原生 file runner。
    # 同一格内 (id, cap_tier, condition) 唯一（去重守卫），逐行 verdict 按 id 落位。
    groups: dict = {}
    for row, src in rows:
        key = (str(row.get("condition")), str(row.get("cap_tier")), row["category"])
        groups.setdefault(key, []).append((row, src))

    underline = handler.model_name_underline_replaced
    verdicts: dict = {}
    for key in sorted(groups.keys()):
        group_rows = groups[key]
        group_ids = [r["id"] for r, _ in group_rows]
        # 同一样本跨格（不同 cap / condition）重评：先清 multi_turn_utils 实例
        # 缓存，防跨格状态污染（金标实例也会被 checker 执行并缓存）。
        for entry_id in group_ids:
            _cleanup_multi_turn_instances(underline, entry_id)
        cell_verdicts, accuracy = _run_native_group(
            handler, group_rows, prompt_by_id, answer_by_id
        )
        for entry_id in group_ids:
            _cleanup_multi_turn_instances(underline, entry_id)
        verdicts[key] = cell_verdicts
        valid_n = sum(1 for v in cell_verdicts.values() if v[0])
        print(f"[native] 格 {key}: {len(group_rows)} 行, native_valid_n={valid_n}, "
              f"accuracy={accuracy:.4f}")

    ctx_cache: dict = {}
    scored_rows: list[dict] = []
    for row, src in rows:
        key = (str(row.get("condition")), str(row.get("cap_tier")), row["category"])
        scored = _score_row(row, prompt_by_id, answer_by_id, ctx_cache,
                            verdicts[key][row["id"]], missed_by_id)
        scored_rows.append(scored)
        print(
            f"[score] {src} {row['id']} condition={row.get('condition')} "
            f"cap_tier={row.get('cap_tier')} native_valid={scored['native_valid']} "
            f"native_error_type={scored['native_error_type']} "
            f"prose.correct={scored['prose']['correct']} "
            f"protocol_valid={scored['protocol_valid']} "
            f"split_row={scored['split_row']}"
        )

    # 输出顺序：按 (id, cap_tier, condition) 稳定排序
    scored_rows.sort(key=lambda x: (x["id"], x["cap_tier"], x["condition"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[score] 输出 {len(scored_rows)} 行 -> {out_path}")

    summary = _build_summary(scored_rows)
    _print_summary(summary)
    if args.summary_out:
        _write_summary_out(summary, Path(args.summary_out))
        print(f"[score] summary -> {args.summary_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m metrology.bfcl_score",
        description="BFCL v4 推理行离线评分器（原生评分 + 协议面 + 散文抽取 + 分裂标记）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bfcl_pkg_path", required=True,
        help="BFCL v4 快照中 bfcl_eval 包路径（或含 bfcl_eval/ 子包的目录）；只读",
    )
    p.add_argument(
        "--bfcl_data_dir", required=True,
        help="BFCL 数据目录（含 6 个 prompt 文件 + possible_answer/ + multi_turn_func_doc/）",
    )
    p.add_argument(
        "--runs_dir", required=True,
        help="推理结果目录；目录下所有 *.jsonl 都读",
    )
    p.add_argument("--out", required=True, help="scored 结果 jsonl 输出路径")
    p.add_argument("--summary_out", default=None, help="可选：summary JSON 输出路径")
    return p


def main(argv=None):
    # Windows 控制台默认 GBK：强制 UTF-8 输出，避免中文进度信息乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非必需能力，失败忽略
        pass
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
