# -*- coding: utf-8 -*-
"""S9 纯 CPU 单测：metrology/bfcl_hf_runner.py 的 c2kv 条件 + metrology/c2kv_gist.py。

测试面（对应任务书 5 项）：
a. 条件参数解析：--condition c2kv 与 --c2kv_checkpoint/--c2kv_ratio/--c2kv_doc_mode
   （默认值、显式值、非法 doc_mode 拒绝）；
b. doc 构建：split_bfcl_messages 消息切分、工具/历史 doc 文本（复用训练侧
   _canonical_tool_doc / _agent_history_turn_docs 的精确格式断言）、
   build_c2kv_prompt_plan 三种 doc_mode 的 system/后缀/文档集组装、
   chunk_doc_texts 的 joint 预算分配（工具块上限、历史余量尾偏、超长切分）；
c. gist 配置注入：resolve_c2kv_config 未训练臂注入训练口径 gist 字段
   （mock config_class.from_pretrained，无重加载），训练臂原样加载；
d. 行元数据：c2kv_row_meta 字段；
e. 裸 system 防泄漏：compute_bare_system_content 对真实 .foreman/ref/bfcl_pkg
   重算（functions=[] 与 functions=[...] 对照、entry 自带 system 保留）；
f. 微型 gist Qwen3（合成权重，仓库自定义类）端到端：真实 handler 类 +
   _query_prompting 全路径，三种 doc_mode + 退化（无可压缩文档）+ use_gist
   传播与原始布局 position 记账断言。

运行：pytest metrology/test_bfcl_hf_runner_c2kv.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrology import bfcl_hf_runner as runner
from metrology import c2kv_gist

BFCL_PKG = _REPO_ROOT / ".foreman" / "ref" / "bfcl_pkg"


# ══════════════════════════════════════════════════════════════════════════
# 共享 fixture：canned BFCL 消息 / 函数文档 / stub tokenizer / 惰性依赖守卫
# ══════════════════════════════════════════════════════════════════════════

FUNC_1 = {
    "name": "math.add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "dict",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a"],
    },
}
FUNC_2 = {
    "name": "fs.ls",
    "description": "List files.",
    "parameters": {"type": "dict", "properties": {"path": {"type": "string"}}, "required": []},
}

# 两轮历史 + 当前轮（含轮内 assistant/tool 步骤）的 canned 消息流
MESSAGES = [
    {"role": "system", "content": "SYS WITH TOOLS"},
    {"role": "user", "content": "u one"},
    {"role": "assistant", "content": "a one"},
    {"role": "tool", "content": "t one"},
    {"role": "user", "content": "u two"},
    {"role": "assistant", "content": "a two"},
    {"role": "user", "content": "current query"},
    {"role": "assistant", "content": "cur a"},
    {"role": "tool", "content": "cur t"},
]

BARE_SYSTEM = "BARE SYSTEM (tools stripped)"


class StubTokenizer:
    """确定性词级 tokenizer（无外部文件）：满足 _chat_template_ids /
    runner tokenize / decode 的接口面；chat 渲染模仿 Qwen im_start 块。

    bos_token_id=2 且从不出现在渲染首位（词表 id 自 100 起编），与真实
    Qwen3（模板不 prepend bos、ids[0] != bos_token_id）行为一致，保留
    _chat_template_ids 的 max_length+1 口径原样。
    """

    bos_token_id = 2
    eos_token_id = 4000
    pad_token_id = 0
    model_max_length = 4096

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            nid = 100 + len(self._vocab)
            self._vocab[token] = nid
            self._inv[nid] = token
        return self._vocab[token]

    def _encode(self, text: str) -> list[int]:
        return [self._id(tok) for tok in text.split()]

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._encode(text)

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kw):
        import torch

        from transformers import BatchEncoding

        ids = self._encode(text)
        return BatchEncoding(
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        )

    def decode(self, ids, skip_special_tokens: bool = True, **kw) -> str:
        toks = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in (self.eos_token_id, self.pad_token_id):
                continue
            toks.append(self._inv.get(i, "?"))
        return " ".join(toks)

    def apply_chat_template(self, messages, tools=None, tokenize=True,
                            add_generation_prompt=False, enable_thinking=False,
                            max_length=None, truncation=False, **kw):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


def _train_imports_or_skip():
    try:
        return c2kv_gist._lazy_train_imports()
    except ImportError as e:  # pragma: no cover - 依赖缺失环境
        pytest.skip(f"train 依赖链不可用: {e}")


_BFCL_STATE = {"ready": False}


def _bfcl_ready_or_skip():
    """安装 runner 同款 import stub 并注入 .foreman/ref/bfcl_pkg（幂等）。

    与 bfcl_score._setup_bfcl 的顺序对齐：runner stub 之后立刻做
    _install_vendor_import_stubs + _fixup_runner_stubs（增强 tenacity 等），
    再 import 任何 bfcl_eval 模块——否则 bfcl_eval.model_handler.utils 会以
    朴素 tenacity stub 绑定 retry_if_*（装饰期 reduce(or_, None) 崩），
    污染同会话的 test_bfcl_score（该 fixup 幂等、只认 __spec__=None 手工 stub）。
    """
    if _BFCL_STATE["ready"]:
        return
    if not (BFCL_PKG / "bfcl_eval" / "__init__.py").exists():
        pytest.skip(f"bfcl_eval 包不在位: {BFCL_PKG}")
    from metrology import bfcl_score

    runner._install_import_stubs()
    bfcl_score._install_vendor_import_stubs()
    bfcl_score._fixup_runner_stubs()
    runner._inject_bfcl_syspath(str(BFCL_PKG))
    _BFCL_STATE["ready"] = True


# ══════════════════════════════════════════════════════════════════════════
# a. 条件参数解析
# ══════════════════════════════════════════════════════════════════════════

def test_condition_choices_include_c2kv():
    assert "c2kv" in runner.CONDITION_CHOICES


def test_c2kv_arg_defaults():
    args = runner.build_parser().parse_args(
        ["--bfcl_pkg_path", "x", "--condition", "c2kv", "--model", "m"]
    )
    assert args.condition == "c2kv"
    assert args.c2kv_checkpoint is None       # 缺省 = 基座 + gist 注入 + 未训练
    assert args.c2kv_ratio == 8
    assert args.c2kv_doc_mode == "joint"
    assert args.c2kv_max_doc_length == 1024
    assert args.c2kv_max_doc_num == 24


def test_c2kv_arg_explicit():
    args = runner.build_parser().parse_args(
        ["--bfcl_pkg_path", "x", "--condition", "c2kv", "--model", "m",
         "--c2kv_checkpoint", "/tmp/ck", "--c2kv_ratio", "16",
         "--c2kv_doc_mode", "history_only", "--c2kv_max_doc_num", "12"]
    )
    assert args.c2kv_checkpoint == "/tmp/ck"
    assert args.c2kv_ratio == 16
    assert args.c2kv_doc_mode == "history_only"
    assert args.c2kv_max_doc_num == 12


def test_c2kv_doc_mode_invalid_rejected():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--bfcl_pkg_path", "x", "--condition", "c2kv",
             "--c2kv_doc_mode", "bogus"]
        )


# ══════════════════════════════════════════════════════════════════════════
# b. doc 构建
# ══════════════════════════════════════════════════════════════════════════

def test_split_bfcl_messages_multi_turn():
    system, history, current = c2kv_gist.split_bfcl_messages(MESSAGES)
    assert system == {"role": "system", "content": "SYS WITH TOOLS"}
    # 最后一条真实 user = "current query"；其前（不含 system）全部进历史
    assert [m["content"] for m in history] == [
        "u one", "a one", "t one", "u two", "a two"
    ]
    assert [m["content"] for m in current] == ["current query", "cur a", "cur t"]


def test_split_bfcl_messages_single_turn():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "only q"},
    ]
    system, history, current = c2kv_gist.split_bfcl_messages(msgs)
    assert system["content"] == "s"
    assert history == []
    assert current == [{"role": "user", "content": "only q"}]


def test_split_bfcl_messages_tool_response_user_not_a_query():
    # <tool_response> 包裹的 user 块不是新一轮查询（与 qwen.py:121-127 同义）
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "real q"},
        {"role": "user", "content": "<tool_response>\nX\n</tool_response>"},
    ]
    _system, history, current = c2kv_gist.split_bfcl_messages(msgs)
    assert history == []
    assert current == msgs[1:]


def test_split_bfcl_messages_no_system():
    msgs = [{"role": "user", "content": "q"}]
    system, history, current = c2kv_gist.split_bfcl_messages(msgs)
    assert system is None and history == [] and current == msgs


def test_build_tool_doc_texts_exact_format():
    _train_imports_or_skip()
    docs = c2kv_gist.build_tool_doc_texts([FUNC_1, FUNC_2])
    assert len(docs) == 2
    assert docs[0].startswith("Tool definition:\n<TOOL>\n")
    assert "<NAME> math.add" in docs[0]
    assert '<PARAM name="a" type="integer" required="true">' in docs[0]
    assert docs[0].rstrip().endswith("</TOOL>")
    assert "<NAME> fs.ls" in docs[1]
    # 顺序保留（eval 不打乱）；空渲染过滤
    assert c2kv_gist.build_tool_doc_texts([]) == []


def test_build_history_doc_texts_exact_format():
    _train_imports_or_skip()
    _system, history, _current = c2kv_gist.split_bfcl_messages(MESSAGES)
    docs = c2kv_gist.build_history_doc_texts(history)
    assert docs == [
        "Previous turn\n[User query]\nu one\n[Assistant output]\na one\n\n[tool]\nt one",
        "Previous turn\n[User query]\nu two\n[Assistant output]\na two",
    ]


def _plan(doc_mode, messages=MESSAGES, function=(FUNC_1, FUNC_2), bare=BARE_SYSTEM):
    _train_imports_or_skip()
    return c2kv_gist.build_c2kv_prompt_plan(messages, list(function), doc_mode, bare)


def test_prompt_plan_joint():
    plan = _plan("joint")
    assert plan["gist_tool"] and plan["gist_history"]
    assert plan["system_content"] == BARE_SYSTEM          # 裸 system 防泄漏
    assert len(plan["tool_doc_texts"]) == 2               # 工具进 gist 文档
    assert len(plan["history_doc_texts"]) == 2            # 历史进 gist 文档
    # 后缀 = 当前轮（含轮内步骤）
    assert [m["content"] for m in plan["suffix_messages"]] == [
        "current query", "cur a", "cur t"
    ]


def test_prompt_plan_tool_only():
    plan = _plan("tool_only")
    assert plan["gist_tool"] and not plan["gist_history"]
    assert plan["system_content"] == BARE_SYSTEM          # 工具仍需移出 system
    assert len(plan["tool_doc_texts"]) == 2
    assert plan["history_doc_texts"] == []                # 历史不压缩
    # 历史以 BFCL 原生消息形态留进普通 prompt 后缀
    assert [m["content"] for m in plan["suffix_messages"]] == [
        "u one", "a one", "t one", "u two", "a two",
        "current query", "cur a", "cur t",
    ]


def test_prompt_plan_history_only():
    plan = _plan("history_only")
    assert not plan["gist_tool"] and plan["gist_history"]
    assert plan["system_content"] == "SYS WITH TOOLS"     # 工具留原 system
    assert plan["tool_doc_texts"] == []
    assert len(plan["history_doc_texts"]) == 2
    assert [m["content"] for m in plan["suffix_messages"]] == [
        "current query", "cur a", "cur t"
    ]


def test_prompt_plan_joint_requires_bare_system():
    _train_imports_or_skip()
    with pytest.raises(RuntimeError, match="裸 system"):
        c2kv_gist.build_c2kv_prompt_plan(MESSAGES, [FUNC_1], "joint", None)
    # history_only 不需要裸 system（工具不压缩）
    plan = c2kv_gist.build_c2kv_prompt_plan(MESSAGES, [FUNC_1], "history_only", None)
    assert plan["system_content"] == "SYS WITH TOOLS"


def test_prompt_plan_invalid_doc_mode():
    with pytest.raises(ValueError, match="doc_mode"):
        c2kv_gist.build_c2kv_prompt_plan(MESSAGES, [], "bogus", BARE_SYSTEM)


def test_chunk_doc_texts_budget_allocation():
    _train_imports_or_skip()
    tok = StubTokenizer()
    # 3 条短工具文档（包裹后各 1 块）；max_doc_num=4 / max_tool_chunks=2
    # → 工具块上限 2，历史得 2 槽；5 条历史尾偏保留 [h0, h4]
    tool_texts = [
        "Tool definition:\nalpha beta",
        "Tool definition:\ngamma delta",
        "Tool definition:\nepsilon zeta",
    ]
    history_texts = [f"Previous turn q{i} a{i}" for i in range(5)]
    tool_chunks, history_chunks = c2kv_gist.chunk_doc_texts(
        tok, tool_texts, history_texts,
        max_doc_length=16, max_doc_num=4, max_tool_chunks=2,
    )
    assert len(tool_chunks) == 2                       # 受 max_tool_chunks 上限
    assert all(len(c) <= 16 for c in tool_chunks)
    assert len(history_chunks) == 2                    # 历史槽位 = 4 - 2
    # 尾偏选择（_select_history tail：首条 + 最近 max_doc_num-1 条）
    joined = [" ".join(tok.decode(c).split()) for c in history_chunks]
    assert any("q0" in j for j in joined), joined      # 首条保留
    assert any("q4" in j for j in joined)              # 最近条保留
    assert not any("q2" in j for j in joined)          # 中间被尾偏挤出


def test_chunk_doc_texts_splits_oversized_tool_doc():
    _train_imports_or_skip()
    tok = StubTokenizer()
    long_doc = "Tool definition:\n" + " ".join(f"w{i}" for i in range(30))
    tool_chunks, history_chunks = c2kv_gist.chunk_doc_texts(
        tok, [long_doc], [], max_doc_length=8, max_doc_num=24, max_tool_chunks=None,
    )
    # 包裹后 ~33 token，按 max_doc_length=8 切片（joint 驱动 :218-222 口径）
    wrapped_len = len(tok.apply_chat_template(
        [{"role": "user", "content": long_doc}], tokenize=True
    ))
    expected = (wrapped_len + 7) // 8
    assert len(tool_chunks) == expected
    assert all(len(c) <= 8 for c in tool_chunks)
    assert [t for c in tool_chunks for t in c] == tok.apply_chat_template(
        [{"role": "user", "content": long_doc}], tokenize=True
    )  # 切片连续、不丢 token
    assert history_chunks == []


def test_chunk_doc_texts_tool_cap_default_two_thirds():
    _train_imports_or_skip()
    tok = StubTokenizer()
    tool_texts = [f"Tool definition:\nt{i} alpha beta" for i in range(10)]
    tool_chunks, history_chunks = c2kv_gist.chunk_doc_texts(
        tok, tool_texts, [], max_doc_length=32, max_doc_num=6, max_tool_chunks=None,
    )
    # 缺省 max_tool_chunks = max(1, 2*6//3) = 4（joint 驱动口径）
    assert len(tool_chunks) == 4
    assert history_chunks == []


# ══════════════════════════════════════════════════════════════════════════
# c. gist 配置注入（mock 重加载）
# ══════════════════════════════════════════════════════════════════════════

class _FakeConfigClass:
    last_kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.last_kwargs = dict(kwargs)
        cfg = types.SimpleNamespace(model_path=path, **kwargs)
        return cfg


class _FakeTokenizer:
    eos_token_id = 151645


def test_resolve_c2kv_config_untrained_injects_gist_fields():
    cfg = c2kv_gist.resolve_c2kv_config(
        _FakeConfigClass, "/fake/base", _FakeTokenizer(), trained=False
    )
    kw = _FakeConfigClass.last_kwargs
    assert kw["gist_type"] == "dynamic-interleave"
    assert kw["gist_param"] == "qkv"
    assert kw["gist_residual_type"] == "embed-mean"
    assert kw["gist_overlap"] == 64
    assert kw["gist_token_id"] == 151645          # = tokenizer.eos_token_id（训练口径）
    assert kw["pad_token_id"] is None
    assert kw["trust_remote_code"] and kw["local_files_only"]
    assert cfg.gist_type == "dynamic-interleave"


def test_resolve_c2kv_config_trained_loads_as_is():
    cfg = c2kv_gist.resolve_c2kv_config(
        _FakeConfigClass, "/fake/ckpt", _FakeTokenizer(), trained=True
    )
    kw = _FakeConfigClass.last_kwargs
    # 训练 ckpt 的 config.json 已携 gist 字段：不注入、不覆盖
    assert "gist_type" not in kw
    assert "gist_token_id" not in kw
    assert kw == {"trust_remote_code": True, "local_files_only": True}


# ══════════════════════════════════════════════════════════════════════════
# d. 行元数据
# ══════════════════════════════════════════════════════════════════════════

def test_c2kv_row_meta_fields():
    args = types.SimpleNamespace(
        c2kv_checkpoint="/ck/g_joint/checkpoint-100",
        c2kv_ratio=8, c2kv_doc_mode="joint",
        c2kv_max_doc_length=1024, c2kv_max_doc_num=24,
    )
    meta = c2kv_gist.c2kv_row_meta(args)
    assert meta == {
        "checkpoint": "/ck/g_joint/checkpoint-100",
        "trained": True,
        "ratio": 8,
        "doc_mode": "joint",
        "max_doc_length": 1024,
        "max_doc_num": 24,
    }
    # untrained 对照臂
    args.c2kv_checkpoint = None
    meta = c2kv_gist.c2kv_row_meta(args)
    assert meta["checkpoint"] is None and meta["trained"] is False


# ══════════════════════════════════════════════════════════════════════════
# e. 裸 system 防泄漏（真实 bfcl_pkg）
# ══════════════════════════════════════════════════════════════════════════

def test_compute_bare_system_content_strips_function_docs():
    _bfcl_ready_or_skip()
    from copy import deepcopy

    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    first_turn = [{"role": "user", "content": "hello"}]
    bare = c2kv_gist.compute_bare_system_content(first_turn, "multi_turn_base_0")
    with_funcs = system_prompt_pre_processing_chat_model(
        deepcopy(first_turn), [FUNC_1], "multi_turn_base_0"
    )
    assert "math.add" not in bare                 # 函数文档已移除
    assert "math.add" in with_funcs[0]["content"]  # 对照：同函数注入后存在
    assert bare != with_funcs[0]["content"]
    assert "function" in bare.lower()             # 调用格式指令仍保留


def test_compute_bare_system_content_preserves_entry_system():
    _bfcl_ready_or_skip()
    first_turn = [
        {"role": "system", "content": "ENTRY_SPECIFIC_RULE"},
        {"role": "user", "content": "hi"},
    ]
    bare = c2kv_gist.compute_bare_system_content(first_turn, "multi_turn_base_0")
    assert "ENTRY_SPECIFIC_RULE" in bare          # entry 自带 system 内容保留
    assert "math.add" not in bare


# ══════════════════════════════════════════════════════════════════════════
# f. 微型 gist Qwen3 端到端（真实 handler 类 + _query_prompting 全路径）
# ══════════════════════════════════════════════════════════════════════════

def _tiny_gist_qwen3():
    import torch

    for rel in ("python", "python/inference"):
        p = str(_REPO_ROOT / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=4096, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
        gist_type="dynamic-interleave", gist_param="qkv",
        gist_residual_type="embed-mean", gist_overlap=8,
        gist_token_id=StubTokenizer.eos_token_id,  # 训练口径 gist_token_id=eos
        pad_token_id=0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def _make_c2kv_handler(doc_mode, model=None):
    """真实 handler 类（bfcl stub 面）+ 微型 gist 模型 + stub tokenizer。"""
    _bfcl_ready_or_skip()
    _train_imports_or_skip()
    if model is None:
        model = _tiny_gist_qwen3()
    handler = runner._build_handler(
        handler_cls=runner._define_handler_class(),
        model_path="tiny_stub",
        tokenizer=StubTokenizer(),
        model=model,
        device="cpu",
        cap_tier="128",
        condition="c2kv",
        max_context_length=4096,
        kv_budget=None,
        c2kv_settings={
            "doc_mode": doc_mode,
            "ratio": 4,
            "max_doc_length": 32,
            "max_doc_num": 8,
            "max_tool_chunks": None,
        },
    )
    handler._c2kv_bare_system = BARE_SYSTEM
    return handler


def _run_query(handler, messages=MESSAGES, function=(FUNC_1, FUNC_2)):
    api_response, _elapsed = handler._query_prompting(
        {"function": list(function), "message": [dict(m) for m in messages]}
    )
    entry = handler._query_history[-1]
    return api_response, entry


def test_end_to_end_c2kv_all_doc_modes():
    for doc_mode in ("joint", "tool_only", "history_only"):
        handler = _make_c2kv_handler(doc_mode)
        api_response, entry = _run_query(handler)
        meta = entry["compression_meta"]
        assert meta["method"] == "c2kv"
        assert meta["doc_mode"] == doc_mode
        assert meta["ratio"] == 4
        assert meta["degenerate"] is False
        # 三种模式都有文档被压缩（joint: 2 类；tool_only: 工具；history_only: 历史）
        assert meta["doc_tokens"] > 0
        assert 0 < meta["gist_tokens"] < meta["doc_tokens"]
        assert meta["actual_compression_ratio"] == pytest.approx(
            round(meta["doc_tokens"] / meta["gist_tokens"], 4)
        )
        # kept_tokens 内部一致性：system + gist + 原文历史 + 后缀
        assert meta["kept_tokens"] == (
            meta["system_length"] + meta["gist_tokens"]
            + meta["raw_history_tokens"] + meta["suffix_tokens"]
        )
        if doc_mode == "tool_only":
            assert meta["tool_doc_chunks"] > 0 and meta["history_doc_chunks"] == 0
            assert meta["raw_history_tokens"] > 0      # 历史留原文
        elif doc_mode == "history_only":
            assert meta["history_doc_chunks"] > 0 and meta["tool_doc_chunks"] == 0
            assert meta["raw_history_tokens"] == 0
        else:
            assert meta["tool_doc_chunks"] > 0 and meta["history_doc_chunks"] > 0
            assert meta["raw_history_tokens"] == 0
        # 伪 api_response 与 base 同构
        assert api_response.usage.completion_tokens == entry["gen_tokens"]
        assert api_response.usage.prompt_tokens > 0
        assert entry["stop_reason"] in ("eos", "length", "other")
        assert isinstance(api_response.choices[0].text, str)


def test_end_to_end_c2kv_use_gist_and_positions():
    """use_gist 传播（system 原文=False / 后缀与 decode=True）与原始布局
    position 记账（后缀 position 起点 = system_length + doc_tokens）。"""
    handler = _make_c2kv_handler("joint")
    model = handler._hf_model
    calls = []
    orig_forward = model.forward

    def _spy(*args, **kwargs):
        calls.append(
            {
                "use_gist": kwargs.get("use_gist"),
                "position_ids": kwargs.get("position_ids"),
                "input_len": int(kwargs["input_ids"].shape[-1]),
            }
        )
        return orig_forward(*args, **kwargs)

    model.forward = _spy
    try:
        _api_response, entry = _run_query(handler)
    finally:
        model.forward = orig_forward
    meta = entry["compression_meta"]
    assert calls, "model.forward 未被调用"
    # 第 1 次 forward = system 原文 prefill（use_gist 缺省）
    assert calls[0]["use_gist"] in (None, False)
    # 其后（后缀 prefill + decode 步）全部 use_gist=True
    assert all(c["use_gist"] is True for c in calls[1:])
    # 后缀 prefill（第 2 次 forward）的 position 起点 = system + 文档原始 token 数
    suffix_call = calls[1]
    assert suffix_call["position_ids"] is not None
    assert int(suffix_call["position_ids"][0, 0]) == (
        meta["system_length"] + meta["doc_tokens"]
    )
    assert suffix_call["input_len"] == meta["suffix_tokens"]
    # decode 步 position 逐 +1 连续（原始布局延续）
    decode_positions = [
        int(c["position_ids"][0, 0]) for c in calls[2:] if c["position_ids"] is not None
    ]
    assert decode_positions == sorted(decode_positions)
    assert all(b == a + 1 for a, b in zip(decode_positions, decode_positions[1:]))
    assert decode_positions[0] == (
        meta["system_length"] + meta["doc_tokens"] + meta["suffix_tokens"]
    )


def test_end_to_end_c2kv_degenerate_no_docs():
    """history_only 第 0 轮（无历史）→ 无可压缩文档：退化为 base 同路径
    （model.generate），meta degenerate=True、gist 字段全 0。"""
    handler = _make_c2kv_handler("history_only")
    first_turn = [
        {"role": "system", "content": "SYS WITH TOOLS"},
        {"role": "user", "content": "first query"},
    ]
    _api_response, entry = _run_query(handler, messages=first_turn)
    meta = entry["compression_meta"]
    assert meta["method"] == "c2kv"
    assert meta["degenerate"] is True
    assert meta["doc_tokens"] == 0 and meta["gist_tokens"] == 0
    assert meta["tool_doc_chunks"] == 0 and meta["history_doc_chunks"] == 0
    assert meta["actual_compression_ratio"] is None
    assert entry["gen_tokens"] > 0


def test_handler_class_keeps_final_guard_and_hook():
    """_define_handler_class 的 @final 显式守卫仍成立；c2kv hook 已挂载。"""
    _bfcl_ready_or_skip()
    handler_cls = runner._define_handler_class()
    for m in runner._FINAL_METHODS:
        assert m not in handler_cls.__dict__
    assert "_hf_generate_c2kv" in handler_cls.__dict__
    assert "_query_prompting" in handler_cls.__dict__
