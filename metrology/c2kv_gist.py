# -*- coding: utf-8 -*-
"""S9：BFCL runner 的 c2kv（gist 压缩）条件支持——doc 构建、前缀压缩与生成。

本模块承载 metrology/bfcl_hf_runner.py --condition c2kv 的全部 c2kv 专属逻辑，
runner 本体只做薄 hook（_query_prompting 分支 + 模型加载分支 + 行元数据）。
设计对齐 agent/eval_joint_next_action_c2kv.py（true-joint 评测驱动），关键映射：

══ BFCL 消息流 → c2kv 文档网格 ══

BFCL prompting 路径每步查询时 inference_data["message"] =
[system(含内联函数文档), ...历史轮(user/assistant/tool)..., 当前轮消息...]，
inference_data["function"] = 该类别的函数文档 dict 列表（同一批 dict 经
system_prompt_pre_processing_chat_model 内联进 system 文本）。

c2kv 条件把「文档类块」从原始 prompt 中移出、压缩成 gist KV 前缀：

- system：joint / tool_only 用「裸 system」（同一 BFCL 函数以 functions=[] 重算，
  移除内联工具文档，防泄漏；见 compute_bare_system_content）；history_only 保留
  BFCL 原 system（工具文档随之留在原文，这正是该臂「不压缩工具」的语义）。
- 工具文档：每个 function dict 一条 "Tool definition:\n" + 规范 <TOOL> 渲染
  （复用 train.train_data_joint._canonical_tool_doc，即训练侧 70% 主导格式；
  chat-template user 包裹，与训练/评测一致）。
- 历史文档：当前轮之前的历史按轮聚成 "Previous turn\n[User query]\n...
  [Assistant output]\n..." 文档（复用 train.train_data_multiturn.
  _agent_history_turn_docs 的原文逻辑；tool 角色渲染为 "[tool]\n..."）。
- 普通 prompt 后缀 = 当前轮消息（含轮内已生成的 assistant/tool 步骤），经
  handler._format_prompt 渲染（与 base 条件逐字节同源），末尾
  "<|im_start|>assistant\n"。

doc_mode（--c2kv_doc_mode）= 哪些文档类被 gist 压缩（未被压缩的类保留在原文，
与训练侧 tool_only/history_only「丢弃另一侧」的消融语义不同——BFCL 多轮任务
去掉工具或历史会破坏任务本身，故本评测臂只改「压缩/原文」不改「有无」）：

- joint：工具 + 历史都压缩；后缀 = 当前轮。
- tool_only：只压缩工具；后缀 = 历史（BFCL 原生消息）+ 当前轮（原始布局中
  历史与当前轮相邻，合并为一次后缀 prefill，position_ids 连续）。
- history_only：只压缩历史；工具留在原 system；后缀 = 当前轮。

══ 位置记账（与 joint 评测驱动逐值一致）══

gist KV 经 blend_gist_key_values 按各文档原始 token 长度累计重定位
（prefix_length=system_length 起算）；其后每个原文块（tool_only 的历史块、
prompt 后缀）以「原始未压缩布局」的 position_ids 续排（cursor 按原始 token 数
推进），attention_mask 按实际 cache 槽位（system + gist + 已 prefill 原文块）
置 1。生成用 use_gist=True（凡是发生过 gist 压缩的臂）——训练与既有 c2kv
评测的同一约定（prompt/answer 前向也走 gist 投影，见
agent/eval_joint_next_action_c2kv.py 模块 docstring 的 CAVEAT）。

══ 复用与出处 ══

- 经惰性 import 复用（python/ 与 python/inference 上 sys.path，首次 c2kv 调用时）：
  train.train_data_joint.TOOL_DOC_PREFIX / _canonical_tool_doc /
  _default_max_tool_chunks；train.train_data_multiturn._agent_history_turn_docs /
  _chat_template_ids / _fit_reused_history / _pad。
- 小函数复制并标注出处（agent/ 非包、import 链过重，不整体 import）：
  gist_compress_docs ≈ agent/eval_agent_tool_definition_c2kv.py:_build_tool_cache
  去掉 system 拼接（== eval_joint_next_action_c2kv.py:_compress_docs_to_cache）；
  prefill_block ≈ 同文件 _prefill_tokens_with_cache / _prefill_system；
  concat_prefix_caches ≈ 同文件 _build_tool_cache 的逐层 cat 与
  eval_joint_next_action_c2kv.py:_build_separate_prefix 的同形拼接；
  build_c2kv_config ≈ 同文件 _gist_compatible_config（训练注入口径见
  python/models/model_utils.py:151-201）。

预算分配镜像 _condition_doc_chunks 的 joint 分支：工具块先取，上限
min(max_tool_chunks, max_doc_num)（默认 2/3），历史块取余下槽位、超长切分、
尾偏选择；不实现 max_tool_definition_tokens 的 skip（BFCL 每类函数文档有界，
块上限已约束总量）。
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

C2KV_DOC_MODE_CHOICES = ["joint", "tool_only", "history_only"]
C2KV_DOC_MODES_WITH_TOOL_GIST = ("joint", "tool_only")
C2KV_DOC_MODES_WITH_HISTORY_GIST = ("joint", "history_only")

# 训练注入口径（python/models/model_utils.py:151-201 与
# agent/eval_agent_tool_definition_c2kv.py:_gist_compatible_config）
GIST_CONFIG_INJECTION = {
    "gist_type": "dynamic-interleave",
    "gist_param": "qkv",
    "gist_residual_type": "embed-mean",
    "gist_overlap": 64,
}

_TRAIN_IMPORTS: dict = {}


def _lazy_train_imports() -> dict:
    """惰性 import 训练侧 doc 构造函数（python/ 与 python/inference 上 sys.path）。

    首次调用约数秒（datasets/transformers 链）；非 c2kv 条件不触发。
    """
    if _TRAIN_IMPORTS:
        return _TRAIN_IMPORTS
    for rel in ("python", "python/inference"):
        p = str(REPO_ROOT / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    from train.train_data_joint import (  # noqa: E402
        TOOL_DOC_PREFIX,
        _canonical_tool_doc,
        _default_max_tool_chunks,
    )
    from train.train_data_multiturn import (  # noqa: E402
        _agent_history_turn_docs,
        _chat_template_ids,
        _fit_reused_history,
        _pad,
    )

    _TRAIN_IMPORTS.update(
        {
            "TOOL_DOC_PREFIX": TOOL_DOC_PREFIX,
            "_canonical_tool_doc": _canonical_tool_doc,
            "_default_max_tool_chunks": _default_max_tool_chunks,
            "_agent_history_turn_docs": _agent_history_turn_docs,
            "_chat_template_ids": _chat_template_ids,
            "_fit_reused_history": _fit_reused_history,
            "_pad": _pad,
        }
    )
    return _TRAIN_IMPORTS


# ══════════════════════════════════════════════════════════════════════════
# BFCL 消息切分与 doc 文本构建
# ══════════════════════════════════════════════════════════════════════════

def _is_tool_response_user(message: dict) -> bool:
    """Qwen chat template 里 tool 结果会被包成 <tool_response> user 块；
    BFCL message 列表中工具结果本为 role="tool"，此守卫与 qwen.py:121-127 的
    last_query_index 判定同义，防御其他 handler 风格。"""
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith("<tool_response>")
        and content.endswith("</tool_response>")
    )


def split_bfcl_messages(messages: list[dict]) -> tuple:
    """把 BFCL prompting 消息列表切成 (system_message|None, history, current)。

    current = 最后一条真实 user 消息及其后的全部消息（当前轮，含轮内
    assistant/tool 步骤）；history = system 之后、current 之前的所有消息。
    """
    system_message = None
    rest = list(messages)
    if rest and rest[0].get("role") == "system":
        system_message = rest[0]
        rest = rest[1:]
    last_user = None
    for idx, message in enumerate(rest):
        if message.get("role") == "user" and not _is_tool_response_user(message):
            last_user = idx
    if last_user is None:
        # 无真实 user（不应出现于 BFCL）：整体视为当前后缀，历史为空
        return system_message, [], rest
    return system_message, rest[:last_user], rest[last_user:]


def build_tool_doc_texts(function: list[dict]) -> list[str]:
    """每个 function dict 一条 "Tool definition:\n" + 规范 <TOOL> 渲染文档。

    渲染复用 train.train_data_joint._canonical_tool_doc（训练侧 70% 主导的
    canonical 表面；BFCL function dict 为扁平 {name,description,parameters}
    形态，_canonical_tool_doc 的扁平分支直接覆盖）。跳过空渲染。
    """
    imp = _lazy_train_imports()
    docs = []
    for func in function or []:
        if not isinstance(func, dict):
            continue
        rendered = imp["_canonical_tool_doc"](func, 600)
        if rendered and rendered.strip():
            docs.append(imp["TOOL_DOC_PREFIX"] + rendered)
    return docs


def build_history_doc_texts(history_messages: list[dict]) -> list[str]:
    """历史消息按轮聚成 "Previous turn\n[User query]\n...[Assistant output]\n..."
    文档（复用 train.train_data_multiturn._agent_history_turn_docs 原文逻辑：
    user 开启新一轮；assistant 追加输出；其他角色（tool）渲染 "[role]\n..."）。"""
    imp = _lazy_train_imports()
    docs = imp["_agent_history_turn_docs"](list(history_messages or []))
    return [str(doc.get("content") or "") for doc in docs if str(doc.get("content") or "").strip()]


def render_system_prefix_text(system_content: str) -> str:
    """与 qwen.py:115-116 的 system 渲染逐字符一致。"""
    return f"<|im_start|>system\n{system_content}<|im_end|>\n"


def build_c2kv_prompt_plan(
    messages: list[dict],
    function: list[dict],
    doc_mode: str,
    bare_system_content: str | None,
) -> dict:
    """按 doc_mode 组装一次查询的 c2kv 计划（纯文本/消息层，不 tokenize）。

    返回 {system_content, tool_doc_texts, history_doc_texts, suffix_messages,
    gist_tool, gist_history}。suffix_messages 交给 handler._format_prompt 渲染
    （与 base 条件同源）。joint/tool_only 需要 bare_system_content（裸 system，
    工具文档移入 gist）；history_only 用 BFCL 原 system（工具留原文）。
    """
    if doc_mode not in C2KV_DOC_MODE_CHOICES:
        raise ValueError(f"未知 c2kv doc_mode: {doc_mode!r}")
    system_message, history_messages, current_messages = split_bfcl_messages(messages)
    gist_tool = doc_mode in C2KV_DOC_MODES_WITH_TOOL_GIST
    gist_history = doc_mode in C2KV_DOC_MODES_WITH_HISTORY_GIST

    if gist_tool:
        if bare_system_content is None:
            raise RuntimeError(
                "c2kv doc_mode=joint/tool_only 需要裸 system（防工具文档泄漏），"
                "但 handler._c2kv_bare_system 未设置（_run_one_entry 应在 "
                "handler.inference 前 compute_bare_system_content 并挂载）"
            )
        system_content = bare_system_content
    else:
        system_content = system_message["content"] if system_message else ""

    suffix_messages = list(current_messages)
    if gist_tool and not gist_history:
        # tool_only：历史不被压缩，以 BFCL 原生消息形态留在普通 prompt
        suffix_messages = list(history_messages) + suffix_messages

    return {
        "system_content": system_content,
        "tool_doc_texts": build_tool_doc_texts(function) if gist_tool else [],
        "history_doc_texts": (
            build_history_doc_texts(history_messages) if gist_history else []
        ),
        "suffix_messages": suffix_messages,
        "gist_tool": gist_tool,
        "gist_history": gist_history,
        "has_system": system_message is not None or gist_tool,
    }


def chunk_doc_texts(
    tokenizer,
    tool_doc_texts: list[str],
    history_doc_texts: list[str],
    *,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_chunks: int | None,
) -> tuple[list[list[int]], list[list[int]]]:
    """doc 文本 → chat-template 包裹的 token 块（joint 预算分配）。

    镜像 agent/eval_joint_next_action_c2kv.py:_condition_doc_chunks 的 joint
    分支（:207-245）：工具文档逐条 _chat_template_ids 包裹后按 max_doc_length
    切片、上限 min(max_tool_chunks, max_doc_num)；历史文档取余下槽位，
    _fit_reused_history（超长切分 + 尾偏 tail 选择）后逐条包裹。
    不实现 max_tool_definition_tokens 的 skip（见模块 docstring）。
    """
    imp = _lazy_train_imports()
    if max_tool_chunks is None:
        max_tool_chunks = imp["_default_max_tool_chunks"](max_doc_num)

    tool_chunks: list[list[int]] = []
    tool_cap = min(max_tool_chunks, max_doc_num)
    for doc_text in tool_doc_texts:
        if not doc_text.strip():
            continue
        doc_ids = imp["_chat_template_ids"](
            tokenizer, [{"role": "user", "content": doc_text}]
        )
        for start in range(0, len(doc_ids), max_doc_length):
            tool_chunks.append(doc_ids[start : start + max_doc_length])
    tool_chunks = tool_chunks[:tool_cap]

    history_chunks: list[list[int]] = []
    history_budget = max_doc_num - len(tool_chunks)
    raw_history = [
        {"role": "user", "content": text}
        for text in history_doc_texts
        if text and text.strip()
    ]
    if history_budget > 0 and raw_history:
        fitted = imp["_fit_reused_history"](
            tokenizer,
            raw_history,
            max_doc_length=max_doc_length,
            max_doc_num=history_budget,
            policy="tail",
            split_oversized_history_docs=True,
        )
        history_chunks = [
            imp["_chat_template_ids"](tokenizer, [message], max_length=max_doc_length)
            for message in fitted
        ]
    return tool_chunks, history_chunks


# ══════════════════════════════════════════════════════════════════════════
# 裸 system（防工具文档泄漏）与 gist config 注入
# ══════════════════════════════════════════════════════════════════════════

def compute_bare_system_content(first_turn_message: list[dict], entry_id: str) -> str:
    """用 BFCL 自己的 system_prompt_pre_processing_chat_model 以 functions=[]
    重算 system 消息 = 移除内联工具文档后的「裸 system」。

    与 base_oss_handler.py:367-375 的注入完全同函数、同 entry_id（格式敏感
    配置逐字节一致），仅函数列表为空；entry 自带 system 内容时的 "\n\n" 追加
    行为也随之保留。在 handler.inference 之前调用（question[0] 未被原地改）。
    """
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    prepared = system_prompt_pre_processing_chat_model(
        deepcopy(first_turn_message), [], entry_id
    )
    assert prepared and prepared[0]["role"] == "system", (
        "functions=[] 时 BFCL 仍应生成 system 消息（formulate_system_prompt 恒产出）"
    )
    return prepared[0]["content"]


def build_c2kv_config(config_class, model_path: str, tokenizer):
    """未训练对照臂：为无 gist 字段的基座 config 注入训练口径的 gist 配置。

    复制自 agent/eval_agent_tool_definition_c2kv.py:_gist_compatible_config
    （:54-65）；训练侧等价注入见 python/models/model_utils.py:151-201
    （gist_token_id = tokenizer.eos_token_id）。
    """
    return config_class.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        pad_token_id=None,
        gist_token_id=tokenizer.eos_token_id,
        **GIST_CONFIG_INJECTION,
    )


def resolve_c2kv_config(config_class, weights_path: str, tokenizer, trained: bool):
    """trained=True（--c2kv_checkpoint）：config.json 已携 gist 字段，原样加载；
    trained=False（默认基座）：注入 gist 配置（gist 参数随机初始化、未训练）。"""
    if trained:
        return config_class.from_pretrained(
            weights_path, trust_remote_code=True, local_files_only=True
        )
    return build_c2kv_config(config_class, weights_path, tokenizer)


def load_c2kv_model_weights(args, tokenizer):
    """c2kv 条件的模型加载（CPU 就位；调用方随后 .to(device)，与 runner 一致）。

    权重路径 = --c2kv_checkpoint（训练 ckpt）或 --model（基座 + gist 注入，
    gist 参数未训练 = untrained 对照臂，镜像 joint 评测的 c2kv_untrained）。
    """
    import torch

    for rel in ("python", "python/inference"):
        p = str(REPO_ROOT / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import get_model_class  # noqa: E402

    weights_path = args.c2kv_checkpoint or args.model
    trained = bool(args.c2kv_checkpoint)
    config_class, model_class = get_model_class(weights_path, "qkv")
    config = resolve_c2kv_config(config_class, weights_path, tokenizer, trained)
    print(
        f"[model] c2kv 权重={weights_path}  trained={trained}  "
        f"gist_type={getattr(config, 'gist_type', None)}"
    )
    model = model_class.from_pretrained(
        weights_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    return model


# ══════════════════════════════════════════════════════════════════════════
# 行元数据
# ══════════════════════════════════════════════════════════════════════════

def c2kv_row_meta(args) -> dict:
    """行级 c2kv 元数据（rows["c2kv_meta"]），供 bfcl_score/分析期分层。

    模板对齐 snapkv 的 kv_budget 口径：condition 字符串保持 "c2kv"（评分按
    (condition, cap_tier) 分格不变），臂间差异全部落在此字段与逐步
    compression_meta；默认输出文件名含 doc_mode/ratio 防 resume 撞键。
    """
    return {
        "checkpoint": getattr(args, "c2kv_checkpoint", None),
        "trained": bool(getattr(args, "c2kv_checkpoint", None)),
        "ratio": int(getattr(args, "c2kv_ratio", 8)),
        "doc_mode": getattr(args, "c2kv_doc_mode", "joint"),
        "max_doc_length": int(getattr(args, "c2kv_max_doc_length", 1024)),
        "max_doc_num": int(getattr(args, "c2kv_max_doc_num", 24)),
    }


def _c2kv_compression_meta(
    *,
    doc_mode: str,
    ratio: int,
    system_length: int,
    doc_tokens: int,
    gist_tokens: int,
    tool_doc_chunks: int,
    history_doc_chunks: int,
    raw_history_tokens: int,
    suffix_tokens: int,
    degenerate: bool,
) -> dict:
    """逐步 compression_meta（snapkv 形状兼容键 + c2kv 专属键）。

    kept_tokens = 解码开始前 cache 实际槽位（system + gist + 原文历史 + 后缀）；
    budget/obs_window/n_sink 置 None 保持与 snapkv/streamingllm 相同的键集合。
    """
    kept = system_length + gist_tokens + raw_history_tokens + suffix_tokens
    return {
        "method": "c2kv",
        "budget": None,
        "kept_tokens": kept,
        "obs_window": None,
        "n_sink": None,
        "doc_mode": doc_mode,
        "ratio": ratio,
        "system_length": system_length,
        "doc_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "tool_doc_chunks": tool_doc_chunks,
        "history_doc_chunks": history_doc_chunks,
        "raw_history_tokens": raw_history_tokens,
        "suffix_tokens": suffix_tokens,
        "actual_compression_ratio": (
            round(doc_tokens / gist_tokens, 4) if gist_tokens else None
        ),
        "degenerate": degenerate,
    }


# ══════════════════════════════════════════════════════════════════════════
# gist 压缩、前缀拼接与生成（重手术路径，复制小函数并标注出处）
# ══════════════════════════════════════════════════════════════════════════

def _swap_attn_impl(model, attn_impl: str):
    """eval 驱动的 _attn_implementation 交换惯例（eager 进出）。"""
    original = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    return original


def _restore_attn_impl(model, original: str):
    model.model.config._attn_implementation = original


def gist_compress_docs(model, context_input_ids, prefix_length: int, ratio: int,
                       attn_impl: str = "eager"):
    """文档网格 → gist KV cache（不含 system 拼接）。

    复制自 agent/eval_agent_tool_definition_c2kv.py:_build_tool_cache（:289-337）
    去掉 system cache 拼接的部分（== agent/eval_joint_next_action_c2kv.py:
    _compress_docs_to_cache，:400-455；计时与 _sync_device 略）。逐文档 RoPE
    重定位由 blend_gist_key_values 内部按原始 token 长度从 prefix_length 累计。
    返回 (cache, doc_tokens, gist_tokens)。
    """
    import torch

    from models import blend_gist_key_values  # 惰性（models 链在加载时已就绪）

    device = model.device
    context_input_ids = context_input_ids.to(device)
    valid_mask = context_input_ids != -100
    doc_tokens = int(valid_mask.sum().item())
    input_ids = context_input_ids.clone()
    input_ids[~valid_mask] = model.model.gist_token_id

    original = _swap_attn_impl(model, attn_impl)
    gist_kwargs = {}
    if getattr(model.config, "gist_type", None) == "dynamic-interleave":
        gist_kwargs["ratio"] = ratio
    with torch.no_grad():
        outputs, gist_mask, pos_ids = model.model.generate_gist(
            input_ids=input_ids,
            attention_mask=valid_mask,
            **gist_kwargs,
        )
    _restore_attn_impl(model, original)

    with torch.no_grad():
        cache, _ = blend_gist_key_values(
            model.config,
            [outputs.past_key_values],
            [gist_mask],
            [pos_ids],
            model.model.rotary_emb,
            prefix_length,
        )
    gist_tokens = cache.get_seq_length()
    return cache, doc_tokens, gist_tokens


def concat_prefix_caches(caches: list):
    """逐层 cat 多个 cache 的 K/V（dim=-2）。

    复制自 agent/eval_agent_tool_definition_c2kv.py:_build_tool_cache（:330-332）
    的 system+gist 拼接与 eval_joint_next_action_c2kv.py:_build_separate_prefix
    （:539-549）的同形多层拼接，泛化为 N 段。返回第一段 cache（原地扩写）。
    """
    import torch

    prefix_cache = caches[0]
    for layer_index, prefix_layer in enumerate(prefix_cache.layers):
        prefix_layer.keys = torch.cat(
            [c.layers[layer_index].keys for c in caches], dim=-2
        )
        prefix_layer.values = torch.cat(
            [c.layers[layer_index].values for c in caches], dim=-2
        )
    return prefix_cache


def prefill_block(model, input_ids, past_key_values, use_gist: bool,
                  position_start: int, attn_impl: str = "eager"):
    """携 cache 前向一个原文块（system / 原文历史 / prompt 后缀）。

    复制自 agent/eval_agent_tool_definition_c2kv.py 的 _prefill_system（:230-245）
    与 _prefill_tokens_with_cache（:248-285）：attention_mask 按实际 cache 槽位
    置 1；position_ids 从 position_start 续排（调用方按原始未压缩布局给坐标，
    与 _prefill_tokens_with_cache 的 past_length 起算同义），logits_to_keep=1。
    use_gist 镜像既有 c2kv 评测的生成约定（发生过 gist 压缩则 prompt/answer
    前向也走 gist 投影）。返回 (cache, next_id)。
    """
    import torch

    original = _swap_attn_impl(model, attn_impl)
    cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    attention_mask = torch.ones(
        (input_ids.shape[0], cache_len + input_ids.shape[1]),
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = torch.arange(
        position_start,
        position_start + input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            use_gist=use_gist,
        )
    _restore_attn_impl(model, original)
    next_id = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    return outputs.past_key_values, next_id


def hf_generate_c2kv(handler, message: list[dict], function: list[dict],
                     max_tokens: int):
    """c2kv 条件的生成主路径（runner._query_prompting 的 c2kv 分支调用）。

    流程：build_c2kv_prompt_plan → chunk_doc_texts → system prefill（原文）→
    gist 压缩文档网格并拼接 → 后缀 prefill（tool_only 的后缀含原文历史段，
    与当前轮在原始布局中相邻，单次 prefill）→ greedy decode（显式原始布局
    position_ids，与 runner 压缩路径 decode 循环同构）。无可压缩文档时
    （如 history_only 的第 0 轮）退化为与 base 逐字节同路径的
    model.generate（meta degenerate=True）。
    返回 (generated_1d_long_tensor, compression_meta)。
    """
    import torch

    settings = handler._c2kv_settings
    doc_mode = settings["doc_mode"]
    ratio = int(settings["ratio"])
    max_doc_length = int(settings["max_doc_length"])
    max_doc_num = int(settings["max_doc_num"])
    max_tool_chunks = settings.get("max_tool_chunks")
    device = handler._hf_device
    model = handler._hf_model
    tokenizer = handler.tokenizer

    plan = build_c2kv_prompt_plan(
        message, function, doc_mode, handler._c2kv_bare_system
    )
    tool_chunks, history_chunks = chunk_doc_texts(
        tokenizer,
        plan["tool_doc_texts"],
        plan["history_doc_texts"],
        max_doc_length=max_doc_length,
        max_doc_num=max_doc_num,
        max_tool_chunks=max_tool_chunks,
    )
    chunks = [*tool_chunks, *history_chunks]

    suffix_text = handler._format_prompt(plan["suffix_messages"], function)
    suffix_ids = tokenizer(
        suffix_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    suffix_tokens = int(suffix_ids.shape[-1])

    # —— 退化：无可压缩文档 → 与 base 同路径（完整原文 generate）——
    if not chunks:
        full_text = handler._format_prompt(message, function)
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = dict(do_sample=False, max_new_tokens=max_tokens)
        if handler._eos_token_ids:
            gen_kwargs["eos_token_id"] = handler._eos_token_ids
        if handler._pad_token_id is not None:
            gen_kwargs["pad_token_id"] = handler._pad_token_id
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][prompt_len:]
        meta = _c2kv_compression_meta(
            doc_mode=doc_mode, ratio=ratio,
            system_length=0, doc_tokens=0, gist_tokens=0,
            tool_doc_chunks=0, history_doc_chunks=0,
            raw_history_tokens=0, suffix_tokens=prompt_len,
            degenerate=True,
        )
        return generated, meta

    eos_ids = set(handler._eos_token_ids or [])

    # —— 1) system 原文 prefill（无 gist，use_gist 缺省=False，同 _prefill_system）——
    system_text = (
        render_system_prefix_text(plan["system_content"]) if plan["has_system"] else ""
    )
    system_ids = tokenizer(
        system_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    system_length = int(system_ids.shape[-1])
    system_cache = None
    if system_length > 0:
        original = _swap_attn_impl(model, "eager")
        with torch.no_grad():
            sys_out = model(
                input_ids=system_ids,
                attention_mask=torch.ones_like(system_ids),
                use_cache=True,
                logits_to_keep=1,
            )
        _restore_attn_impl(model, original)
        system_cache = sys_out.past_key_values
        del sys_out

    # —— 2) 文档网格 gist 压缩 + 拼接（system 在前；无 system 时 gist 即前缀）——
    imp = _lazy_train_imports()
    grid = torch.tensor(
        [imp["_pad"](chunk, max_doc_length, -100) for chunk in chunks],
        dtype=torch.long,
    )
    gist_cache, doc_tokens, gist_tokens = gist_compress_docs(
        model, grid, prefix_length=system_length, ratio=ratio, attn_impl="eager"
    )
    cache = (
        concat_prefix_caches([system_cache, gist_cache])
        if system_cache is not None
        else gist_cache
    )

    # 原始未压缩布局的游标：system 原文 + 文档原始 token 数。
    # 其后所有原文块（后缀；tool_only 时后缀内已含历史段）以此续排 position_ids，
    # 与 eval_joint_next_action_c2kv.py:_generate_with_prefix 的
    # original_prefix_length = system_length + doc_length 逐值一致。
    cursor = system_length + doc_tokens

    # —— 3) 后缀 prefill（use_gist=True）→ 首 token ——
    # tool_only 的后缀 = 历史 + 当前轮（原始布局中二者相邻，单次 prefill 与分段
    # prefill 在因果注意力下逐值等价）；raw_history_tokens 仅为 meta 统计。
    raw_history_tokens = 0
    if plan["gist_tool"] and not plan["gist_history"]:
        _sys, history_messages, _cur = split_bfcl_messages(message)
        history_text = _remove_generation_prompt_tail(
            handler._format_prompt(history_messages, function)
        )
        if history_text:
            raw_history_tokens = int(tokenizer(
                history_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].shape[-1])
    cache, next_id = prefill_block(
        model, suffix_ids, cache, use_gist=True, attn_impl="eager",
        position_start=cursor,
    )
    pos = cursor + suffix_tokens

    # —— 4) greedy decode（与 runner 压缩路径 decode 循环同构）——
    gen_ids: list[int] = []
    past = cache
    with torch.no_grad():
        while len(gen_ids) < max_tokens:
            gen_ids.append(int(next_id[0, 0].item()))
            if next_id[0, 0].item() in eos_ids:
                break
            step_out = model(
                input_ids=next_id,
                attention_mask=None,
                position_ids=torch.tensor([[pos]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                use_gist=True,
            )
            past = step_out.past_key_values
            next_id = step_out.logits[:, -1].argmax(dim=-1, keepdim=True)
            pos += 1

    generated = torch.tensor(gen_ids, dtype=torch.long, device=device)
    meta = _c2kv_compression_meta(
        doc_mode=doc_mode, ratio=ratio,
        system_length=system_length, doc_tokens=doc_tokens,
        gist_tokens=gist_tokens,
        tool_doc_chunks=len(tool_chunks),
        history_doc_chunks=len(history_chunks),
        raw_history_tokens=raw_history_tokens,
        suffix_tokens=suffix_tokens,
        degenerate=False,
    )
    return generated, meta


def _remove_generation_prompt_tail(formatted_text: str) -> str:
    """_format_prompt 恒追加 "<|im_start|>assistant\n"（qwen.py:178）；
    统计/中间块渲染时去掉该尾缀。"""
    tail = "<|im_start|>assistant\n"
    if formatted_text.endswith(tail):
        return formatted_text[: -len(tail)]
    return formatted_text
