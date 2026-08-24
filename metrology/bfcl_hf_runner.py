# -*- coding: utf-8 -*-
"""S8.1：BFCL v4 本地 HF 推理 runner（metrology 包第一件）。

职责：在 NPU 服务器上用 HF transformers eager 路径（bf16、greedy、do_sample=False）跑
R5 冻结 360 样本清单（configs/r5_metrology_sample.json），逐行落盘完整原始生成文本与逐轮
元数据，供 S8 分析期（(i) censoring 重分类率、(ii) 外壳-语义分裂率）使用。
本 chunk 只实现 condition=base（无压缩对照）；压缩条件（SnapKV / H2O / StreamingLLM /
KVzip 择一，见 prereg §4）在后续 chunk 通过 _HFQwenPromptingHandler._hf_generate 内的
hook 点接入。

S8.2 起：condition=snapkv / streamingllm 已实现（训练无关的 KV 压缩）。接入方式 =
S8.1 预留的 hook 点（_HFQwenPromptingHandler._query_prompting 内）：完整 prompt 先
eager prefill（不物化全量注意力；冻结 360 样本 prompt 中位 ~6k、最长 9355 token，
(1,32,L,L) 概率 × 36 层在 64GB HBM 不可行），SnapKV 所需观察窗注意力由「末
obs_window 个 token 携 cache 二次前向（output_attentions=True）+ 切原 prompt 列」
获得（double-pass；与全量物化后切片数学等价，单测
test_micro_model_double_pass_attention_matches_full；随后把二次前向追加进 cache 的
obs 个位置裁回 [0, prompt_len)）；再按 metrology/kv_compress.py 的方法规则（官方
SnapKV / StreamingLLM 逻辑的等价重实现；对照与已知偏差见 kv_compress.py 模块
docstring）选出保留位置、对 past_key_values 做位置子集裁切，再进 greedy decode 循环
（显式 position_ids 延续原始绝对位置，与官方 SnapKV 语义一致）。
base 仍走 model.generate 原路径（零手术，输出与 S8.1 逐字节一致；--kv_budget 对 base
无效）。h2o / kvzip 仍为预留（未实现即报错）。预算：--kv_budget 或默认 prompt_len 的
50%，逐行记录 compression_meta。

S9 起：condition=c2kv 已实现（C2KV gist 压缩检查点的评测臂）。接入方式 = 同一
hook 点：_query_prompting 内 condition=="c2kv" 分支转调
metrology/c2kv_gist.hf_generate_c2kv——把工具文档（"Tool definition:\n" 块）
和/或历史轮（"Previous turn\n..." 块）按 --c2kv_doc_mode {joint,tool_only,
history_only} 压缩成 gist KV 前缀（ratio=--c2kv_ratio，默认 8），再用普通
prompt 后缀生成；doc 构建/位置记账逐值镜像 agent/eval_joint_next_action_c2kv.py
（详见 c2kv_gist.py 模块 docstring）。模型加载：--c2kv_checkpoint 给训练 ckpt
（config.json 已携 gist 字段）；缺省 = 基座 + gist 配置注入 + 未训练 gist 参数
（untrained 对照臂）。cap 口径不变（按完整未压缩 prompt token 数计算）。
逐行记录 rows["c2kv_meta"] 与逐步 compression_meta（method="c2kv" 等字段）。

══ 路径判断（prompting vs FC）：结论 = prompting ══

证据 1（官方注册表）：BFCL v4 constants/model_config.py:1672-1683 将
"Qwen/Qwen3-4B-Instruct-2507" 注册为 QwenHandler，is_fc_model=False（Prompt 模式，
model_config.py:1678），对应 SUPPORTED_MODELS.md 中 Qwen3-4B 的 Self-hosted Prompt 条目。
prompting 注册的原生 handler 即 model_handler/local_inference/qwen.py 的 QwenHandler。
证据 2（数据 schema）：本任务 6 类中 multi_turn_{base,long_context,miss_func,miss_param}
数据含 initial_config / involved_classes、无顶层 function 字段（函数文档由
utils.populate_test_cases_with_predefined_functions 从 data/multi_turn_func_doc 注入，
utils.py:772-801）；base_oss_handler.py:51-64 的 inference() 对
contain_multi_turn_interaction(id) 为真的条目分派到 base_handler.py:393 的
@final inference_multi_turn_prompting（prompting 多轮循环，内含
execute_multi_turn_func_call 真实执行仿真）。parallel / parallel_multiple 无多轮交互，
走 base_handler.py:721 的 @final inference_single_turn_prompting。
证据 3（FC 路径为何不适用）：FC 路径（QwenFCHandler + _compile_tools + OpenAI tool schema）
对应 is_fc_model=True 注册（model_config.py:1660-1671），与本任务选用的 prompting 注册
不符；且多轮数据没有 FC 路径所需的工具编译入口语义。故不复用 FC 路径。

══ 子类化策略：成功（未复制基类循环）══

子类 QwenHandler，仅 override 其 _query_prompting（原实现 base_oss_handler.py:317-364
走 vLLM/SGLang OpenAI-compatible server），替换为本地 HF generate：同一 _format_prompt、
同一 stop 行为（eos 停、无额外 stop string）、同一 max_tokens 语义（base_oss_handler.py:328-336
的 leftover 计算原样保留，128/1024 档直接覆盖）。基类 @final 循环与
execute_multi_turn_func_call 全部原样复用。
构造时不 override __init__（overrides 包的 EnforceOverrides 严格模式），而是在
QwenHandler 实例上挂载 _hf_model / tokenizer / cap 等属性（见 _build_handler）。

══ cap 口径（与 configs/r5_metrology_prereg.md §4 一致）══

- default 档：max_tokens 逐查询按 base_oss_handler.py:328-336 原式
  min(4096, max_context_length - input_token - 2) 计算，实际值逐行记录
  （steps[].actual_cap 与顶层 actual_cap_per_turn）。
- 128 / 1024 档：直接覆盖 max_tokens，其余逻辑不变。
- greedy：do_sample=False；行内 seed 记 null。

══ 依赖面（服务器零新 pip 安装目标）══

bfcl_eval import 链会拖入：openai（OSSHandler 构造 OpenAI client，构造不发网络请求）、
tree_sitter / tree_sitter_java / tree_sitter_javascript（仅 java/js 语法解析用；本 runner
6 类全为 python 格式 AST，不实际调用）、overrides / tenacity / filelock / requests。
本 runner 对上述模块做「先真实 import、失败则注册最小功能 stub」的条件隔离
（见 _install_import_stubs）；stub 只满足 import 与构造语义，真被调用即抛
NotImplementedError 以尽早暴露误用。真实执行后端（func_source_code 各模块）仅在
execute_multi_turn_func_call 时按需 import；6 类涉及的类（GorillaFileSystem / MathAPI /
MessageAPI / TwitterAPI / TicketAPI / TradingBot / TravelAPI / VehicleControlAPI）中仅
MathAPI 需要 mpmath（轻量纯 python，BFCL 自身依赖），dryrun 会自检。

══ 输出 jsonl 行 schema ══

{id, category, cap_tier, condition,
 turns: [{turn_index, steps_used, raw_text(该轮各 step 原文拼接),
          steps: [{step_index, raw_text(完整原文，含 <think> 片段), parsed_text
                   (BFCL _parse_query_response_prompting 剥离 reasoning 后的文本，与
                    BFCL 循环 decode 输入一致), gen_tokens,
                   stop_reason(eos/length/other), decoded_calls(执行串列表或异常类名),
                   decode_error_message, actual_cap, wall_sec,
                   (仅压缩条件附 compression_meta{method, budget, kept_tokens,
                    obs_window, n_sink}；c2kv 附 method="c2kv" 及 doc_mode/ratio/
                    doc_tokens/gist_tokens/tool_doc_chunks/history_doc_chunks/
                    actual_compression_ratio/degenerate 等扩展键)，
                   (仅单轮类附 decoded_ast / decode_ast_error)}]}],
 actual_cap_per_turn, model, max_context_length, seed: null, wall_sec,
 (仅 c2kv 附 c2kv_meta{checkpoint, trained, ratio, doc_mode, max_doc_length,
                      max_doc_num}),
 bfcl_result(与 BFCL 生成器 _llm_response_generation.py:214-218 同形，可直接
             handler.write() 落成 BFCL 原生结果文件), snapshot_manifest_sha256, error}
multi_turn 的 raw_text 完整保留原始文本（下游双列评分要用）；first_divergence_turn
由分析期计算，不在此写入。

用法：
  python -m metrology.bfcl_hf_runner --bfcl_pkg_path <bfcl_eval 包路径> --dryrun
  （服务器示例见文件尾 epilog / 交付说明）
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
import types
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_IDS_DEFAULT = REPO_ROOT / "configs" / "r5_metrology_sample.json"

# D 实验（KV repair）臂：condition 与 AppWorld 臂名 1:1（"c2kv_" + 臂名），
# 实现见 metrology/d_repair_arms.py（模块层 torch-free，可安全顶层 import）
from metrology.d_repair_arms import D_ARM_CONDITIONS  # noqa: E402

CONDITION_CHOICES = [
    "base", "snapkv", "h2o", "streamingllm", "kvzip", "c2kv",
    *D_ARM_CONDITIONS,
]
CAP_TIER_CHOICES = ["default", "128", "1024"]

# 压缩条件固定口径（与 metrology/kv_compress.py 默认一致；S8.2 任务书）
KV_OBS_WINDOW = 16   # SnapKV 观察窗（官方默认 32，本 sprint 口径 16，见对照表第 9 条）
KV_KERNEL = 7        # SnapKV maxpool 核（必须奇数）
KV_N_SINK = 4        # StreamingLLM sink 前缀长度

# 子类不得覆盖的基类 @final 方法（行号：base_handler.py / base_oss_handler.py）。
# 仅作显式守卫；本 runner 不复制这些方法，语义全部复用基类。
_FINAL_METHODS = (
    "inference_multi_turn_FC",            # base_handler.py:94
    "inference_multi_turn_prompting",     # base_handler.py:393
    "inference_single_turn_FC",           # base_handler.py:684
    "inference_single_turn_prompting",    # base_handler.py:721
    "write",                              # base_handler.py:768
    "spin_up_local_server",               # base_oss_handler.py:74
)


# ══════════════════════════════════════════════════════════════════════════
# 依赖 stub：条件 import 隔离（先真实、失败则 stub）
# ══════════════════════════════════════════════════════════════════════════

def _make_module(name: str, attrs: dict) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def _stub_tree_sitter(name: str) -> types.ModuleType:
    """tree_sitter stub：仅满足 java_parser/js_parser 模块顶层的 Language(...) 与
    Parser().set_language(...) 构造调用；真实 parse 一律报错。我们 6 类全为 python
    格式 AST（ast_parse 的 ReturnFormat.PYTHON 分支），不经过 java/js 解析。"""

    class _Language:
        def __init__(self, *args, **kwargs):
            pass

    class _Parser:
        def __init__(self, *args, **kwargs):
            pass

        def set_language(self, language):
            pass

        def parse(self, source_bytes):
            raise NotImplementedError(
                "[metrology stub] tree_sitter 仅为满足 bfcl_eval import 链而 stub；"
                "java/js 语法解析不用于本 runner 的 6 个类别"
            )

    return _make_module(name, {"Language": _Language, "Parser": _Parser})


def _stub_tree_sitter_grammar(name: str) -> types.ModuleType:
    return _make_module(name, {"language": lambda: object()})


def _stub_openai(name: str) -> types.ModuleType:
    """openai stub：OSSHandler.__init__ 只构造 OpenAI(base_url=..., api_key=...)，
    不发请求；我们 override 了 _query_prompting，client 不会被调用。"""

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    return _make_module(name, {"OpenAI": _OpenAI})


def _stub_overrides(name: str) -> types.ModuleType:
    """overrides stub：@final/@override 降级为恒等装饰器，EnforceOverrides 不再做
    严格检查。真实语义由本 runner 的 _FINAL_METHODS 显式守卫兜底。"""

    def _final(method):
        return method

    def _override(method):
        return method

    class _EnforceOverrides:
        pass

    return _make_module(
        name, {"final": _final, "override": _override, "EnforceOverrides": _EnforceOverrides}
    )


def _stub_tenacity(name: str) -> types.ModuleType:
    """tenacity stub：retry_with_backoff 仅在定义时引用装饰器名字，本 runner 不调用；
    stub 保证 model_handler/utils.py 的模块级 import 通过。"""

    def _retry(*args, **kwargs):
        def _dec(fn):
            return fn
        return _dec

    def _noop(*args, **kwargs):
        return None

    return _make_module(
        name,
        {
            "retry": _retry,
            "wait_random_exponential": _noop,
            "retry_if_exception_message": _noop,
            "retry_if_exception_type": _noop,
        },
    )


def _stub_filelock(name: str) -> types.ModuleType:
    """filelock stub：bfcl_eval.utils._get_file_lock 用 FileLock 做跨进程写锁；
    本 runner 单进程单写者，锁降级为 no-op（文件级 append+flush 已保证落盘）。"""

    class _FileLock:
        def __init__(self, lock_file, *args, **kwargs):
            self._lock_file = lock_file

        def acquire(self, *args, **kwargs):
            pass

        def release(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _make_module(name, {"FileLock": _FileLock})


def _stub_requests(name: str) -> types.ModuleType:
    """requests stub：仅 base_oss_handler 的服务器就绪轮询用到（spin_up_local_server），
    本 runner 不起服务器，不会调用。"""

    class _ConnectionError(ConnectionError):
        pass

    def _get(*args, **kwargs):
        raise NotImplementedError(
            "[metrology stub] requests 仅为满足 bfcl_eval import 链而 stub；"
            "本 runner 不启动 vLLM/SGLang server"
        )

    return _make_module(name, {"get": _get, "exceptions": types.SimpleNamespace(ConnectionError=_ConnectionError)})


_STUB_TARGETS = [
    # (模块名, stub 工厂, 说明)
    ("tree_sitter", _stub_tree_sitter, "java/js 语法解析器依赖；本 runner 不经过"),
    ("tree_sitter_java", _stub_tree_sitter_grammar, "同上"),
    ("tree_sitter_javascript", _stub_tree_sitter_grammar, "同上"),
    ("openai", _stub_openai, "OSSHandler 仅构造 client；_query_prompting 已被 override"),
    ("overrides", _stub_overrides, "@final/@override 装饰器；降级 + 显式守卫"),
    ("tenacity", _stub_tenacity, "retry_with_backoff 仅定义不调用"),
    ("filelock", _stub_filelock, "单进程写者，锁降级 no-op"),
    ("requests", _stub_requests, "仅 server 就绪轮询用；本 runner 不起服务器"),
]


def _install_import_stubs(force_stub: bool = False) -> dict:
    """对 _STUB_TARGETS 逐个尝试真实 import；失败（或 force_stub=True 强制自测）
    时把 stub 注册进 sys.modules。返回 {模块名: "real"|"stub"}。

    必须在本 runner import 任何 bfcl_eval 模块之前调用。
    """
    status = {}
    for name, factory, _note in _STUB_TARGETS:
        if not force_stub:
            try:
                importlib.import_module(name)
                status[name] = "real"
                continue
            except ImportError:
                pass
        sys.modules[name] = factory(name)
        status[name] = "stub"
    return status


# ══════════════════════════════════════════════════════════════════════════
# BFCL 相关工具（延迟 import，仅在 _setup_bfcl 之后可用）
# ══════════════════════════════════════════════════════════════════════════

def _inject_bfcl_syspath(bfcl_pkg_path: str) -> Path:
    """--bfcl_pkg_path 接受两种形态：bfcl_eval 包目录本身，或包含 bfcl_eval/ 子包的
    仓库目录。返回解析后的 bfcl_eval 包目录，并把其父目录插入 sys.path。"""
    p = Path(bfcl_pkg_path).resolve()
    if (p / "__init__.py").exists() and p.name == "bfcl_eval":
        sys.path.insert(0, str(p.parent))
        return p
    if (p / "bfcl_eval" / "__init__.py").exists():
        sys.path.insert(0, str(p))
        return p / "bfcl_eval"
    raise FileNotFoundError(
        f"--bfcl_pkg_path={bfcl_pkg_path} 下找不到 bfcl_eval 包（需含 bfcl_eval/__init__.py）"
    )


def _patch_load_file_utf8():
    """BFCL utils.load_file 内部 open 未指定 encoding（utils.py:359），在非 UTF-8
    默认编码平台（如 Windows GBK）会打不开数据文件。此处运行时等价替换（不改
    bfcl_eval 源文件）：仅把 open 补上 encoding='utf-8'，其余语义（filelock 锁、
    逐行 json.loads、sort_by_id）逐行保持一致。"""
    import locale

    if locale.getpreferredencoding(False).lower().replace("-", "") in ("utf8", "utf_8"):
        return

    import bfcl_eval.utils as _u

    def _load_file_utf8(file_path, sort_by_id: bool = False, use_lock: bool = True):
        result = []

        def _load_entries(input_path):
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    result.append(json.loads(line))

        if use_lock:
            with _u._get_file_lock(file_path):
                _load_entries(file_path)
        else:
            _load_entries(file_path)

        if sort_by_id:
            result.sort(key=_u.sort_key)
        return result

    _u.load_file = _load_file_utf8
    print("[setup] 已把 bfcl_eval.utils.load_file 运行时替换为 utf-8 版本（本地默认编码非 UTF-8）")


def _setup_bfcl(args, output_path: Path):
    """安装 stub → 重定向 BFCL_PROJECT_ROOT → sys.path 注入 → import bfcl_eval。

    返回 (bfcl, manifest_sha256, frozen_items)。
    BFCL_PROJECT_ROOT 必须在 import bfcl_eval 之前设置（eval_config.py 在 import 时
    建 result/score/.file_locks 目录），重定向到输出目录，避免污染快照与工作目录。
    """
    stub_status = _install_import_stubs(
        force_stub=bool(os.getenv("METROLOGY_FORCE_STUBS", ""))
    )
    for name, st in sorted(stub_status.items()):
        print(f"[setup] import 面: {name:24s} -> {st}")

    os.environ.setdefault("BFCL_PROJECT_ROOT", str(output_path.parent))

    bfcl_dir = _inject_bfcl_syspath(args.bfcl_pkg_path)

    import bfcl_eval  # noqa: F401  确认包可导入
    from bfcl_eval.constants import eval_config
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX

    assert str(Path(bfcl_eval.__file__).resolve().parent) == str(bfcl_dir), (
        "sys.path 注入未生效：import 到的 bfcl_eval 不在 --bfcl_pkg_path 下"
    )
    print(f"[setup] bfcl_eval @ {bfcl_dir}  version_prefix={VERSION_PREFIX}")

    if args.bfcl_data_dir:
        data_dir = Path(args.bfcl_data_dir).resolve()
        eval_config.PROMPT_PATH = data_dir
        eval_config.MULTI_TURN_FUNC_DOC_PATH = data_dir / "multi_turn_func_doc"
        eval_config.POSSIBLE_ANSWER_PATH = data_dir / "possible_answer"
        print(f"[setup] BFCL 数据目录已重定向: {data_dir}")

    _patch_load_file_utf8()

    manifest_sha = _build_manifest(args, bfcl_dir, Path(eval_config.PROMPT_PATH),
                                   output_path)
    frozen_items = _load_frozen_items(args.ids_file)
    return bfcl_dir, manifest_sha, frozen_items


def _load_frozen_items(ids_file: str) -> list[dict]:
    path = Path(ids_file)
    if not path.is_absolute():
        # 优先相对仓库根（configs/ 下的冻结清单），其次相对 cwd
        cand = REPO_ROOT / path
        path = cand if cand.exists() else path.resolve()
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    assert isinstance(spec.get("items"), list) and len(spec["items"]) > 0, (
        f"冻结清单 {path} 缺少 items 列表"
    )
    print(f"[setup] 冻结清单 {path}: {len(spec['items'])} 条, seed={spec.get('seed')}")
    return spec["items"]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(args, bfcl_dir: Path, data_dir: Path, output_path: Path) -> str:
    """快照不 vendor，用 sha256 manifest 记身份。

    覆盖：本 runner 实际 import/依赖的 bfcl_eval 源文件、BFCL 数据目录全部文件、
    冻结清单、runner 自身。manifest 整体再哈希一次，行内只写 manifest_sha256。
    """
    bfcl_files = [
        "model_handler/local_inference/base_oss_handler.py",
        "model_handler/local_inference/qwen.py",
        "model_handler/base_handler.py",
        "model_handler/utils.py",
        "model_handler/parser/java_parser.py",
        "model_handler/parser/js_parser.py",
        "model_handler/parser/json_parser.py",
        "model_handler/parser/xml_parser.py",
        "utils.py",
        "eval_checker/multi_turn_eval/multi_turn_utils.py",
        "constants/category_mapping.py",
        "constants/default_prompts.py",
        "constants/enums.py",
        "constants/type_mappings.py",
        "constants/eval_config.py",
        "constants/executable_backend_config.py",
    ]
    files = {}
    for rel in bfcl_files:
        p = bfcl_dir / rel
        files[f"bfcl_eval/{rel}"] = _hash_file(p)

    if data_dir.exists():
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                files[f"data/{p.relative_to(data_dir)}"] = _hash_file(p)

    ids_path = Path(args.ids_file)
    if not ids_path.is_absolute() and not ids_path.exists():
        ids_path = REPO_ROOT / ids_path
    files["ids_file"] = _hash_file(ids_path)
    files["runner"] = _hash_file(Path(__file__))

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bfcl_pkg_path": str(bfcl_dir),
        "bfcl_data_dir": str(data_dir),
        "files": files,
    }
    payload = json.dumps(
        sorted((k, v) for k, v in files.items()), sort_keys=True
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()

    manifest_path = Path(str(output_path)).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[setup] 快照 manifest: {manifest_path}")
    return manifest["manifest_sha256"]


# ══════════════════════════════════════════════════════════════════════════
# 本地 HF 生成的 Qwen prompting handler（子类化，只 override _query_prompting）
# ══════════════════════════════════════════════════════════════════════════

def _build_handler(handler_cls, model_path: str, tokenizer, model, device: str,
                   cap_tier: str, condition: str,
                   max_context_length: int | None,
                   kv_budget: int | None,
                   c2kv_settings: dict | None = None,
                   d_plan: dict | None = None):
    """构造 handler：直接调 handler_cls.__init__（即 QwenHandler.__init__，本 runner
    不 override __init__，规避 EnforceOverrides 严格模式），再把本地 HF 状态挂到实例上。

    BFCL 原流水线（_llm_response_generation.py:84-92）同样以 model_name 为
    registry_name 构造 handler；tokenizer / max_context_length 原由
    spin_up_local_server 加载（base_oss_handler.py:139-152），本 runner 直接注入。
    """
    handler = handler_cls(
        model_name=model_path,
        temperature=0.0,
        registry_name=model_path,
        is_fc_model=False,  # prompting 注册，见模块 docstring 路径判断
    )

    handler.tokenizer = tokenizer
    if max_context_length is None:
        cfg = getattr(model, "config", None)
        max_context_length = int(
            getattr(cfg, "max_position_embeddings", None)
            or getattr(tokenizer, "model_max_length", None)
        )
    handler.max_context_length = int(max_context_length)
    print(f"[model] max_context_length={handler.max_context_length}")

    eos = tokenizer.eos_token_id
    handler._eos_token_ids = ([eos] if isinstance(eos, int) else list(eos)) if eos else []
    handler._pad_token_id = getattr(tokenizer, "pad_token_id", None)
    handler._hf_model = model
    handler._hf_device = device
    handler._cap_tier = cap_tier
    handler._condition = condition
    handler._kv_budget = kv_budget        # None → 每查询 budget = prompt_len 的 50%
    handler._obs_window = KV_OBS_WINDOW   # 仅 snapkv
    handler._kv_kernel = KV_KERNEL        # 仅 snapkv
    handler._n_sink = KV_N_SINK           # 仅 streamingllm
    # 仅 c2kv：{doc_mode, ratio, max_doc_length, max_doc_num, max_tool_chunks}；
    # _c2kv_bare_system 由 _run_one_entry 逐样本挂载（裸 system，防工具文档泄漏）
    handler._c2kv_settings = c2kv_settings or {}
    handler._c2kv_bare_system = None
    # 仅 c2kv_d_*：--d_plan 全表；_d_plan_entry 由 _run_one_entry 逐样本挂载
    handler._d_plan = d_plan or {}
    handler._d_plan_entry = None
    handler._query_history = []  # 每次 _query_prompting 追加一条（调用序 == 基类循环步序）
    return handler


def _define_handler_class():
    """动态定义 QwenHandler 的子类（定义函数延迟到 _setup_bfcl 之后调用，确保
    bfcl_eval 的 import 面就绪后再解析基类；类定义本身无副作用）。"""
    """定义 QwenHandler 的子类。定义放在函数内，确保 bfcl_eval import 面就绪后再
    解析基类；类定义本身无副作用。

    说明：不用 overrides.override 装饰器——overrides 包（真实安装时）靠反汇编调用
    帧定位基类，对函数内定义的局部类会报 "No super class method found"。改用
    类体内的 __override__ = True 标记，效果等同装饰器：真实 overrides 安装时满足
    EnforceOverridesMeta 的检查（enforce.py:22-34），stub 时无副作用。"""
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

    class HFQwenPromptingHandler(QwenHandler):
        """子类化 BFCL QwenHandler：override _query_prompting 为本地 HF eager 生成。

        复用（不复制）的基类逻辑：
        - inference() 分派：base_oss_handler.py:51-64
        - 多轮循环 @final：base_handler.py:393-682（含 holdout 轮、空响应/解码失败
          break、MAXIMUM_STEP_LIMIT、execute_multi_turn_func_call 执行回注）
        - 单轮循环 @final：base_handler.py:721-754
        - prompt 格式化：qwen.py:19-179（_format_prompt，逐字符等价 chat template）
        - 响应解析：qwen.py:181-197（剥离 <think> reasoning）与 qwen.py:199-209
        - cap 计算：base_oss_handler.py:328-336（见 _query_prompting 注释）
        """

        def _query_prompting(self, inference_data: dict):
            """对齐 base_oss_handler.py:317-364 的语义，把 OpenAI-compatible server
            调用替换为本地 HF generate。返回 (伪 api_response, 耗时秒)。"""
            import torch

            function: list[dict] = inference_data["function"]
            message: list[dict] = inference_data["message"]

            formatted_prompt: str = self._format_prompt(message, function)
            inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

            # —— cap 计算：与 base_oss_handler.py:325-336 逐行一致 ——
            # 与基类一致用 tokenize 计数；fast tokenizer 若没有 tokenize 方法则兜底
            try:
                input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
            except (AttributeError, TypeError, NotImplementedError):
                input_token_count = len(
                    self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
                )
            if self.max_context_length < input_token_count + 2:
                leftover_tokens_count = 1000
            else:
                leftover_tokens_count = min(
                    4096,
                    self.max_context_length - input_token_count - 2,
                )
            if self._cap_tier == "default":
                max_tokens = leftover_tokens_count
            else:
                max_tokens = int(self._cap_tier)  # 128 / 1024 档直接覆盖
            # —— cap 计算结束 ——

            start_time = time.time()

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
                self._hf_device
            )
            prompt_len = int(inputs["input_ids"].shape[-1])

            gen_kwargs: dict = dict(
                do_sample=False,           # greedy（prereg §4）
                max_new_tokens=max_tokens,
            )
            if self._eos_token_ids:
                gen_kwargs["eos_token_id"] = self._eos_token_ids
            if self._pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self._pad_token_id

            # ══ S8.2：压缩条件接入点（S8.1 预留 hook 点）══
            # base：model.generate 原样（零手术，输出与 S8.1 逐字节一致；
            #       --kv_budget 对 base 无效）。
            # snapkv / streamingllm：prefill(output_attentions=True) →
            #   metrology.kv_compress 位置选择 + past_key_values 裁切 →
            #   greedy decode 循环（位置语义与官方 SnapKV 对齐，对照表见
            #   metrology/kv_compress.py 模块 docstring）。
            compression_meta = None
            with torch.no_grad():
                if self._condition == "base":
                    outputs = self._hf_model.generate(**inputs, **gen_kwargs)
                    generated = outputs[0][prompt_len:]
                elif self._condition == "c2kv":
                    # S9：c2kv gist 压缩（doc 构建与位置记账见
                    # metrology/c2kv_gist.py）；inputs 全量 prefill 不执行，
                    # prompt_len 仅用于伪 api_response 的 input_token 语义
                    generated, compression_meta = self._hf_generate_c2kv(
                        message, function, max_tokens
                    )
                elif self._condition in D_ARM_CONDITIONS:
                    # D 实验 KV repair 臂（metrology/d_repair_arms.py）：
                    # c2kv 前缀 + gist/后缀之间的 cache 手术
                    generated, compression_meta = self._hf_generate_c2kv_arm(
                        message, function, max_tokens
                    )
                else:
                    generated, compression_meta = self._hf_generate_compressed(
                        inputs, gen_kwargs, prompt_len
                    )

            gen_tokens = int(generated.shape[-1])

            eos_ids = set(self._eos_token_ids or [])
            if gen_tokens > 0 and int(generated[-1].item()) in eos_ids:
                stop_reason = "eos"
            elif gen_tokens >= max_tokens:
                stop_reason = "length"
            else:
                stop_reason = "other"

            # skip_special_tokens=True：与 vLLM 默认不输出 stop token 的行为对齐；
            # <think> 等片段仍完整保留在 raw_text 中，由下游按需剥离。
            raw_text = self.tokenizer.decode(generated, skip_special_tokens=True)

            elapsed = time.time() - start_time
            query_entry = {
                "raw_text": raw_text,
                "gen_tokens": gen_tokens,
                "stop_reason": stop_reason,
                "actual_cap": max_tokens,
                "wall_sec": round(elapsed, 4),
            }
            if compression_meta is not None:
                query_entry["compression_meta"] = compression_meta
            self._query_history.append(query_entry)

            # 伪 api_response：满足 qwen.py:181-197 的 _parse_query_response_prompting
            api_response = types.SimpleNamespace(
                choices=[types.SimpleNamespace(text=raw_text)],
                usage=types.SimpleNamespace(
                    prompt_tokens=prompt_len,
                    completion_tokens=gen_tokens,
                ),
            )
            return api_response, elapsed

        # 等价 @override 装饰器（见 _define_handler_class 说明）
        _query_prompting.__override__ = True

        def _hf_generate_compressed(self, inputs: dict, gen_kwargs: dict,
                                    prompt_len: int):
            """压缩条件的生成路径（仅 snapkv / streamingllm 进入；base 不调用）。

            prefill（不物化全量注意力）→ [snapkv] 观察窗二次前向取注意力 →
            metrology.kv_compress.compress_pkv 位置选择 + KV 裁切 → greedy decode 循环。

            - 预算：--kv_budget，或默认 prompt_len 的 50%（逐行记录）。
            - 位置语义：decode 每步显式 position_ids = prompt_len + step（原始绝对
              位置延续，与官方 SnapKV 的 kv_seq_len 记账一致；对照表第 4 条）；
              attention_mask=None（无 padding，mask 由 cache 长度推断）。
            - 注意力获取（double-pass）：全量 prefill 用 output_attentions=False
              （冻结 360 样本 prompt 中位 ~6k、最长 9355 token；每层 (1,32,L,L)
              fp32 softmax × 36 层物化在 64GB HBM 上不可行）。SnapKV 所需仅为
              观察窗行：prefill 后以末 obs_window 个 token 携 cache 二次前向
              （output_attentions=True，4D 因果加性 mask 逐行掩掉相对查询位置
              的「未来」prompt 键与重复列——裸二次前向的 cache 含全量 prompt，
              不加 mask 会让观察窗查询的 softmax 分母膨胀、topk 排名可能翻转），
              切出原 prompt 键列即得同一 eager softmax 的对应行（等价性单测
              test_micro_model_double_pass_attention_matches_full）；二次前向
              追加进 cache 的 obs 个位置随即裁回 [0, prompt_len)。
            返回 (generated, compression_meta)。
            """
            import torch

            from metrology.kv_compress import (
                apply_selection, compress_pkv, layer_kv_tensors,
            )

            device = self._hf_device
            budget = (
                int(self._kv_budget)
                if self._kv_budget is not None
                else max(1, prompt_len // 2)
            )
            input_ids = inputs["input_ids"]
            if int(input_ids.shape[0]) != 1:
                raise ValueError("压缩路径仅支持 batch=1（BFCL 单样本）")

            with torch.no_grad():
                prefill_out = self._hf_model(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=True,
                    output_attentions=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            pkv = prefill_out.past_key_values
            next_id = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
            del prefill_out

            n_layers = len(pkv)
            # 守卫：手术假定各层 cache 长度 == prompt_len（滑动窗口层布局不支持）
            for li in range(n_layers):
                k_li, _ = layer_kv_tensors(pkv, li)
                if int(k_li.shape[-2]) != prompt_len:
                    raise NotImplementedError(
                        f"层 {li} 的 cache 长度 {int(k_li.shape[-2])} != prompt_len "
                        f"{prompt_len}；滑动窗口 cache 布局不支持该手术路径"
                    )

            need_surgery = budget < prompt_len
            obs_ok = int(self._obs_window) < prompt_len
            if need_surgery and self._condition == "snapkv" and obs_ok:
                # 观察窗二次前向（double-pass，见 docstring）。必须带 4D 因果加性
                # mask：cache 内已含全量 prompt，裸二次前向会让观察窗查询看到
                # 相对自身位置的「未来」prompt 键，softmax 分母随之膨胀（列值被
                # 按行缩放，topk 排名可能翻转）。逐行掩掉 [p+1..L) 与全部重复列
                # [L..L+obs) 后，行 i 可见键恰为 [0..p]，与全量 prefill 的对应
                # 观察窗行逐值等价（单测
                # test_micro_model_double_pass_attention_matches_full 断言）。
                obs = int(self._obs_window)
                obs_ids = input_ids[:, -obs:]
                obs_pos = torch.arange(
                    prompt_len - obs, prompt_len, dtype=torch.long, device=device
                ).unsqueeze(0)
                dtype = next(self._hf_model.parameters()).dtype
                kv_len = prompt_len + obs
                obs_mask = torch.full(
                    (1, 1, obs, kv_len), torch.finfo(dtype).min,
                    dtype=dtype, device=device,
                )
                for i in range(obs):
                    p_abs = prompt_len - obs + i
                    obs_mask[0, 0, i, : p_abs + 1] = 0.0
                with torch.no_grad():
                    obs_out = self._hf_model(
                        input_ids=obs_ids,
                        attention_mask=obs_mask,
                        position_ids=obs_pos,
                        past_key_values=pkv,
                        use_cache=True,
                        output_attentions=True,
                        return_dict=True,
                    )
                # (1,H,obs,L+obs) → 仅保留原 prompt 键列 [0, L)
                attentions = [a[..., :prompt_len] for a in obs_out.attentions]
                del obs_out
                # 二次前向把 obs 个重复 token 追加进了 cache：裁回 [0, L)
                pkv = apply_selection(pkv, list(range(prompt_len)))
                for li in range(n_layers):
                    k_li, _ = layer_kv_tensors(pkv, li)
                    assert int(k_li.shape[-2]) == prompt_len, "cache 裁回失败"
                cfg = getattr(self._hf_model, "config", None)
                h_q = int(getattr(cfg, "num_attention_heads"))
                h_kv = int(getattr(cfg, "num_key_value_heads"))
                groups = max(1, h_q // h_kv)
                pkv, kept_tokens = compress_pkv(
                    pkv,
                    "snapkv",
                    budget,
                    attentions=attentions,
                    obs_window=self._obs_window,
                    kernel=self._kv_kernel,
                    n_sink=self._n_sink,
                    num_key_value_groups=groups,
                    prompt_len=prompt_len,
                )
                del attentions
            elif need_surgery and self._condition == "streamingllm":
                pkv, kept_tokens = compress_pkv(
                    pkv,
                    "streamingllm",
                    budget,
                    n_sink=self._n_sink,
                    prompt_len=prompt_len,
                )
            else:
                # budget >= prompt_len（恒等零手术），或 snapkv 观察窗覆盖全
                # prompt（obs_window >= prompt_len，无前缀可压缩，恒等）
                kept_tokens = prompt_len
            assert kept_tokens <= prompt_len, "压缩后保留数不应超过 prompt 长度"
            is_identity = kept_tokens == prompt_len
            assert is_identity == (
                (not need_surgery)
                or (self._condition == "snapkv" and not obs_ok)
            ), "恒等边界与 budget/obs_window 口径不一致"

            eos_ids = set(self._eos_token_ids or [])
            max_new_tokens = int(gen_kwargs["max_new_tokens"])

            gen_ids = []
            past = pkv
            pos = prompt_len
            with torch.no_grad():
                while len(gen_ids) < max_new_tokens:
                    gen_ids.append(int(next_id[0, 0].item()))
                    if next_id[0, 0].item() in eos_ids:
                        break
                    step_out = self._hf_model(
                        input_ids=next_id,
                        attention_mask=None,
                        position_ids=torch.tensor(
                            [[pos]], dtype=torch.long, device=device
                        ),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = step_out.past_key_values
                    next_id = step_out.logits[:, -1].argmax(dim=-1, keepdim=True)
                    pos += 1

            generated = torch.tensor(gen_ids, dtype=torch.long, device=device)
            meta = {
                "method": self._condition,
                "budget": budget,
                "kept_tokens": kept_tokens,
                "obs_window": (
                    self._obs_window if self._condition == "snapkv" else None
                ),
                "n_sink": self._n_sink if self._condition == "streamingllm" else None,
            }
            return generated, meta

        def _hf_generate_c2kv(self, message: list, function: list,
                              max_tokens: int):
            """c2kv 条件的薄 hook：全部逻辑在 metrology/c2kv_gist.py（惰性
            import；非 c2kv 条件不触发 train/models 依赖链）。"""
            from metrology.c2kv_gist import hf_generate_c2kv

            return hf_generate_c2kv(self, message, function, max_tokens)

        def _hf_generate_c2kv_arm(self, message: list, function: list,
                                  max_tokens: int):
            """c2kv_d_* 条件的薄 hook：全部逻辑在 metrology/d_repair_arms.py。
            臂名从 condition 推出；per-entry plan 在 _d_plan_entry（_run_one_entry
            逐样本挂载）。"""
            from metrology.d_repair_arms import hf_generate_c2kv_arm

            return hf_generate_c2kv_arm(self, message, function, max_tokens)

        def reset_query_history(self):
            self._query_history = []

    # 显式守卫：确认没有意外覆盖基类 @final 方法
    for _m in _FINAL_METHODS:
        assert _m not in HFQwenPromptingHandler.__dict__, f"不允许覆盖 @final 方法 {_m}"
    return HFQwenPromptingHandler


# ══════════════════════════════════════════════════════════════════════════
# 逐行结果组装与落盘
# ══════════════════════════════════════════════════════════════════════════

def _safe_decode_execute(handler, text: str):
    try:
        return handler.decode_execute(text, has_tool_call_tag=False), None, None
    except Exception as e:  # noqa: BLE001 记录异常类名即可，行内保留信息
        return None, type(e).__name__, str(e)[:300]


def _safe_decode_ast(handler, text: str):
    from bfcl_eval.constants.enums import ReturnFormat
    try:
        return (
            handler.decode_ast(text, ReturnFormat.PYTHON, has_tool_call_tag=False),
            None,
            None,
        )
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__, str(e)[:300]


def _run_one_entry(handler, entry: dict, cap_tier: str, condition: str,
                   model: str, manifest_sha: str, max_context_length: int) -> dict:
    """按 BFCL 基类循环跑一条样本，组装本 runner 的 jsonl 行。"""
    from bfcl_eval.utils import contain_multi_turn_interaction, extract_test_category_from_id

    entry_id = entry["id"]
    category = extract_test_category_from_id(entry_id)
    is_multi = contain_multi_turn_interaction(entry_id)

    handler.reset_query_history()
    entry_copy = deepcopy(entry)  # 循环会原地改 question[0]（system prompt 注入）
    t0 = time.time()
    try:
        if condition == "c2kv" or condition in D_ARM_CONDITIONS:
            # 裸 system（functions=[] 重算，防工具文档泄漏）；须在 handler.inference
            # 前计算（_pre_query_processing_prompting 会原地改 question[0]）
            from metrology.c2kv_gist import compute_bare_system_content

            handler._c2kv_bare_system = compute_bare_system_content(
                entry_copy["question"][0], entry_id
            )
        if condition in D_ARM_CONDITIONS:
            # D 臂 per-entry plan（无条目 = {}：corr 系臂不需要 payload，
            # sham 臂缺 sham_token_ids 时 fatal d_sham_plan_missing）
            handler._d_plan_entry = handler._d_plan.get(entry_id)
        result, metadata = handler.inference(
            entry_copy, include_input_log=False, exclude_state_log=True
        )
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[run] {entry_id}: 推理异常 {err}")
        return {
            "id": entry_id, "category": category,
            "cap_tier": cap_tier, "condition": condition,
            "turns": [], "actual_cap_per_turn": [],
            "model": model, "max_context_length": max_context_length,
            "seed": None, "wall_sec": round(time.time() - t0, 3),
            "bfcl_result": {"id": entry_id, "result": f"Error during inference: {err}"},
            "snapshot_manifest_sha256": manifest_sha,
            "error": err,
        }
    elapsed = time.time() - t0
    history = list(handler._query_history)

    turns: list[dict] = []
    cap_per_turn: list[list[int]] = []

    if is_multi:
        idx = 0
        for turn_idx, turn_responses in enumerate(result):
            steps: list[dict] = []
            caps: list[int] = []
            for step_idx, parsed_text in enumerate(turn_responses):
                q = history[idx]
                idx += 1
                calls, err_name, err_msg = _safe_decode_execute(handler, parsed_text)
                step = {
                    "step_index": step_idx,
                    "raw_text": q["raw_text"],
                    "parsed_text": parsed_text,
                    "gen_tokens": q["gen_tokens"],
                    "stop_reason": q["stop_reason"],
                    "decoded_calls": calls if calls is not None else err_name,
                    "decode_error_message": err_msg,
                    "actual_cap": q["actual_cap"],
                    "wall_sec": q["wall_sec"],
                }
                if q.get("compression_meta") is not None:
                    step["compression_meta"] = q["compression_meta"]
                steps.append(step)
                caps.append(q["actual_cap"])
            turns.append(
                {
                    "turn_index": turn_idx,
                    "steps_used": len(steps),
                    "raw_text": "\n".join(s["raw_text"] for s in steps),
                    "steps": steps,
                }
            )
            cap_per_turn.append(caps)
        assert idx == len(history), "query 历史与多轮响应步数不一致"
    else:
        parsed_text = result  # 单轮：result 即解析后的字符串
        q = history[0]
        assert len(history) == 1
        calls, err_name, err_msg = _safe_decode_execute(handler, parsed_text)
        ast_calls, ast_err_name, ast_err_msg = _safe_decode_ast(handler, parsed_text)
        step = {
            "step_index": 0,
            "raw_text": q["raw_text"],
            "parsed_text": parsed_text,
            "gen_tokens": q["gen_tokens"],
            "stop_reason": q["stop_reason"],
            "decoded_calls": calls if calls is not None else err_name,
            "decode_error_message": err_msg,
            "decoded_ast": ast_calls if ast_calls is not None else ast_err_name,
            "decode_ast_error": ast_err_msg,
            "actual_cap": q["actual_cap"],
            "wall_sec": q["wall_sec"],
        }
        if q.get("compression_meta") is not None:
            step["compression_meta"] = q["compression_meta"]
        steps = [step]
        turns = [
            {
                "turn_index": 0,
                "steps_used": 1,
                "raw_text": q["raw_text"],
                "steps": steps,
            }
        ]
        cap_per_turn = [[q["actual_cap"]]]

    return {
        "id": entry_id, "category": category,
        "cap_tier": cap_tier, "condition": condition,
        "turns": turns,
        "actual_cap_per_turn": cap_per_turn,
        "model": model, "max_context_length": max_context_length,
        "seed": None, "wall_sec": round(elapsed, 3),
        "bfcl_result": {"id": entry_id, "result": result, **metadata},
        "snapshot_manifest_sha256": manifest_sha,
        "error": None,
    }


def _cleanup_multi_turn_instances(model_name_underline_replaced: str, entry_id: str):
    """BFCL 的 execute_multi_turn_func_call 把类实例缓存在 multi_turn_utils 模块
    globals（键含 model_name+entry_id，multi_turn_utils.py:34-49）。同一进程内同一
    样本在不同 cap 档下重跑必须清掉实例，否则状态跨档污染。"""
    mod = sys.modules.get("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    if mod is None:
        return
    prefix = f"{model_name_underline_replaced}_{entry_id}_"
    for key in [k for k in mod.__dict__ if k.startswith(prefix)]:
        del mod.__dict__[key]


def _load_done_ids(output_path: Path, skip_errors: bool) -> set:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") is None:
                continue
            if not r.get("error") or skip_errors:
                done.add((r["id"], str(r.get("cap_tier")), str(r.get("condition"))))
    return done


def _append_row(output_path: Path, row: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ══════════════════════════════════════════════════════════════════════════
# 主流程：加载数据 / dryrun / 推理
# ══════════════════════════════════════════════════════════════════════════

def _load_bfcl_entries_by_id(frozen_items: list[dict]) -> dict:
    """按类别 load_dataset_entry（与 BFCL 生成流水线 _llm_response_generation.py:105
    完全同源），再按冻结 id 过滤。返回 {id: entry} 与缺失 id 列表。"""
    from bfcl_eval.utils import load_dataset_entry

    categories = sorted({it["category"] for it in frozen_items})
    id_to_entry: dict = {}
    for cat in categories:
        entries = load_dataset_entry(cat)
        cat_map = {e["id"]: e for e in entries}
        id_to_entry.update(cat_map)
        print(f"[data] {cat}: 全量 {len(entries)} 条")
    missing = [it["id"] for it in frozen_items if it["id"] not in id_to_entry]
    if missing:
        print(f"[data] 警告：冻结清单中 {len(missing)} 条 id 在 BFCL 数据中缺失（将记 data_missing 行）")
    return id_to_entry, missing


def _resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        try:
            import torch_npu  # noqa: F401  需要先 source ascend set_env.sh
            if torch.npu.is_available():
                return "npu:0"
        except ImportError:
            pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def _load_model(args):
    """加载 tokenizer + 模型（bf16 eager，greedy 由 generate 参数保证）。仅非 dryrun 调用。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[model] 加载 {args.model} (bf16, eager, device={args.device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
        # S9：仓库自定义 Qwen3 gist 类；--c2kv_checkpoint 训练 ckpt，缺省
        # 基座 + gist 配置注入（未训练 gist 参数 = untrained 对照臂）。
        # c2kv_d_* 臂在同一模型类上跑（修复手术发生在 cache 层）
        from metrology.c2kv_gist import load_c2kv_model_weights

        model = load_c2kv_model_weights(args, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    model.eval()
    device = _resolve_device(args.device)
    if device.startswith("npu"):
        import torch_npu  # noqa: F401  注册 npu 设备；需先 source ascend set_env.sh
    model.to(device)
    print(f"[model] 就绪，device={device}")
    return tokenizer, model, device


def run_dryrun(args, output_path: Path):
    """dryrun：不加载模型。对每个类别取首条冻结样本，跑通数据加载与 prompt 构建
    （_pre_query_processing_prompting → add_first_turn_message_prompting →
    _format_prompt），打印首条 prompt 前 500 字符与轮数；并对 python AST 解码与
    执行后端模块做冒烟检查。"""
    _bfcl_dir, manifest_sha, frozen_items = _setup_bfcl(args, output_path)
    id_to_entry, missing = _load_bfcl_entries_by_id(frozen_items)

    handler_cls = _define_handler_class()
    print(f"[dryrun] handler 类: {handler_cls.__module__}.{handler_cls.__name__}")
    handler = handler_cls(
        model_name="dryrun_model", temperature=0.0,
        registry_name="dryrun_model", is_fc_model=False,
    )
    # dryrun 不 tokenize、不生成：_format_prompt 纯字符串构建，不需要 tokenizer/model

    # 按类别分组，dryrun 默认每类取 1 条；--limit 解释为每类打印的样本数
    per_cat_limit = args.limit if args.limit is not None else 1
    picked: dict = {}
    for it in frozen_items:
        picked.setdefault(it["category"], []).append(it)
    per_category_list = [
        (cat, lst[:per_cat_limit]) for cat, lst in sorted(picked.items())
    ]

    print("=" * 100)
    for cat, items in per_category_list:
        for it in items:
            entry = id_to_entry.get(it["id"])
            if entry is None:
                print(f"[dryrun] {cat} / {it['id']}: 数据缺失，跳过 prompt 构建")
                continue
            entry_copy = deepcopy(entry)
            # 对齐 base_handler.py:465 与 484-487 的首轮准备序列
            inference_data = handler._pre_query_processing_prompting(entry_copy)
            inference_data = handler.add_first_turn_message_prompting(
                inference_data, entry_copy["question"][0]
            )
            formatted = handler._format_prompt(
                inference_data["message"], inference_data["function"]
            )
            print(
                f"[dryrun] {cat} / {it['id']}  n_turns={len(entry_copy['question'])}"
                f"  n_func_docs={len(entry_copy.get('function', []))}"
                f"  prompt_chars={len(formatted)}"
            )
            print("  首条 prompt 前 500 字符：")
            print("  " + formatted[:500].replace("\n", "\n  "))
        print("-" * 100)

    print("[dryrun] python AST 冒烟（不经 tree_sitter）：")
    smoke = "[spotify.play(artist='Taylor Swift', duration=20)]"
    try:
        print("  decode_execute:", handler.decode_execute(smoke, has_tool_call_tag=False))
    except Exception as e:  # noqa: BLE001
        print("  decode_execute 异常:", type(e).__name__, e)
    try:
        from bfcl_eval.constants.enums import ReturnFormat
        print("  decode_ast:", handler.decode_ast(smoke, ReturnFormat.PYTHON, has_tool_call_tag=False))
    except Exception as e:  # noqa: BLE001
        print("  decode_ast 异常:", type(e).__name__, e)

    print("[dryrun] 执行后端模块自检（仅 import，不执行）：")
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
    involved = set()
    for e in id_to_entry.values():
        involved.update(e.get("involved_classes", []))
    for cls in sorted(involved):
        try:
            importlib.import_module(CLASS_FILE_PATH_MAPPING[cls])
            print(f"  {cls}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"  {cls}: 缺失/异常 -> {type(e).__name__}: {e}")

    print("[dryrun] 汇总：")
    print(f"  冻结清单 {len(frozen_items)} 条；数据缺失 {len(missing)} 条")
    print(f"  快照 manifest_sha256={manifest_sha}")


def run_inference(args, output_path: Path):
    _bfcl_dir, manifest_sha, frozen_items = _setup_bfcl(args, output_path)
    id_to_entry, missing = _load_bfcl_entries_by_id(frozen_items)

    tokenizer, model, device = _load_model(args)
    handler_cls = _define_handler_class()
    c2kv_settings = None
    if args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
        c2kv_settings = {
            "doc_mode": args.c2kv_doc_mode,
            "ratio": args.c2kv_ratio,
            "max_doc_length": args.c2kv_max_doc_length,
            "max_doc_num": args.c2kv_max_doc_num,
            "max_tool_chunks": None,  # 缺省 = 2/3 max_doc_num（joint 驱动口径）
        }
    d_plan = None
    if args.condition in D_ARM_CONDITIONS and args.d_plan:
        from metrology.d_repair_arms import load_d_plan

        d_plan = load_d_plan(args.d_plan)
        print(f"[run] d_plan 载入 {len(d_plan)} 条（{args.d_plan}）")
    handler = _build_handler(
        handler_cls=handler_cls,
        model_path=args.model,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cap_tier=args.cap_tier,
        condition=args.condition,
        max_context_length=args.max_context_length,
        kv_budget=args.kv_budget,
        c2kv_settings=c2kv_settings,
        d_plan=d_plan,
    )
    handler_cls_ok = type(handler) is handler_cls
    print(f"[run] handler 实例类型 {type(handler).__name__}（子类化路径: {handler_cls_ok}）")

    done = _load_done_ids(output_path, skip_errors=args.skip_errors)
    to_run = [
        it for it in frozen_items
        if (it["id"], args.cap_tier, args.condition) not in done
    ]
    print(
        f"[run] 冻结 {len(frozen_items)} 条；已完成 {len(frozen_items) - len(to_run)} 条；"
        f"本次将跑 {len(to_run)} 条"
    )
    if args.limit is not None:
        to_run = to_run[: args.limit]
        print(f"[run] --limit={args.limit}，实际跑 {len(to_run)} 条")

    model_name_underline = handler.model_name_underline_replaced  # 与 multi_turn_utils 实例键一致

    for i, it in enumerate(to_run, 1):
        entry = id_to_entry.get(it["id"])
        if entry is None:
            row = {
                "id": it["id"], "category": it["category"],
                "cap_tier": args.cap_tier, "condition": args.condition,
                "turns": [], "actual_cap_per_turn": [],
                "model": args.model,
                "max_context_length": handler.max_context_length,
                "seed": None, "wall_sec": 0.0,
                "bfcl_result": {"id": it["id"], "result": "Error during inference: data_missing"},
                "snapshot_manifest_sha256": manifest_sha,
                "error": "data_missing",
            }
        else:
            row = _run_one_entry(
                handler, entry, args.cap_tier, args.condition,
                args.model, manifest_sha, handler.max_context_length,
            )
            _cleanup_multi_turn_instances(model_name_underline, it["id"])
        if args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
            # 行级 c2kv 元数据（覆盖成功/推理异常/data_missing 三种行；
            # 模板对齐 snapkv kv_budget 的逐行记录口径）；c2kv_d_* 臂附 d_arm
            from metrology.c2kv_gist import c2kv_row_meta

            row["c2kv_meta"] = c2kv_row_meta(args)
            if args.condition in D_ARM_CONDITIONS:
                row["c2kv_meta"]["d_arm"] = args.condition[len("c2kv_"):]
                row["c2kv_meta"]["d_plan"] = args.d_plan
        _append_row(output_path, row)
        print(
            f"[run] {i}/{len(to_run)} {it['id']}  "
            f"error={row['error']}  wall_sec={row['wall_sec']}"
        )

    print(f"[run] 完成。输出: {output_path}")


# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m metrology.bfcl_hf_runner",
        description="BFCL v4 冻结样本本地 HF 推理 runner（S8.2：base / snapkv / streamingllm；S9：c2kv）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bfcl_pkg_path", required=True,
        help="BFCL v4 快照中 bfcl_eval 包路径（或含 bfcl_eval/ 子包的目录）；只读，不修改",
    )
    p.add_argument(
        "--model", default=None,
        help="HF 模型路径（服务器: ~/c2kv/models/Qwen3-4B-Instruct-2507）；dryrun 不需要",
    )
    p.add_argument(
        "--ids_file", default=str(FROZEN_IDS_DEFAULT),
        help="冻结样本清单（默认 configs/r5_metrology_sample.json）",
    )
    p.add_argument(
        "--cap_tier", default="default", choices=CAP_TIER_CHOICES,
        help="cap 档：default=min(4096,剩余上下文)逐行记实际值 / 128 / 1024",
    )
    p.add_argument(
        "--condition", default="base", choices=CONDITION_CHOICES,
        help="压缩条件：base（无压缩对照，generate 原样）/ snapkv / streamingllm（S8.2 已实现，"
             "prefill 后 KV 手术）/ c2kv（S9：C2KV gist 压缩检查点评测臂，见 "
             "metrology/c2kv_gist.py）；h2o / kvzip 为后续 chunk 预留",
    )
    p.add_argument(
        "--kv_budget", type=int, default=None,
        help="压缩条件的 KV 预算（prefill 后保留 token 数）；默认=prompt_len 的 50%。"
             "仅 snapkv/streamingllm 生效（base/c2kv 忽略）；实际保留数逐行记入 compression_meta",
    )
    p.add_argument(
        "--c2kv_checkpoint", default=None,
        help="仅 c2kv：训练好的 c2kv 检查点目录（config.json 携 gist_type/"
             "gist_token_id 等字段）。缺省 None = 用 --model 基座 + gist 配置注入 +"
             " 未训练 gist 参数（untrained 对照臂）",
    )
    p.add_argument(
        "--c2kv_ratio", type=int, default=8,
        help="仅 c2kv：gist 压缩比（dynamic-interleave 的 ratio override，同 "
             "agent/eval_joint_next_action_c2kv.py 的 --override_ratio）",
    )
    p.add_argument(
        "--c2kv_doc_mode", default="joint",
        choices=["joint", "tool_only", "history_only"],
        help="仅 c2kv：哪些文档类被 gist 压缩——joint=工具文档+历史轮 / "
             "tool_only=仅工具文档（历史留原文）/ history_only=仅历史轮（工具留原 system）",
    )
    p.add_argument(
        "--c2kv_max_doc_length", type=int, default=1024,
        help="仅 c2kv：单文档块最大 token 数（joint 驱动 --max_doc_length 口径）",
    )
    p.add_argument(
        "--c2kv_max_doc_num", type=int, default=24,
        help="仅 c2kv：文档块总槽位（工具块上限缺省 2/3，历史块取余下槽位尾偏选择）",
    )
    p.add_argument(
        "--d_plan", default=None,
        help="仅 c2kv_d_* 臂：per-entry 干预计划 JSON "
             "{entry_id: {k_star, span_len, sham_token_ids}}。"
             "c2kv_d_sham_neutral 必需（sham token 来源）；其余臂可选"
             "（提供时做 k* 交叉校验，不符即 fatal 落 error 行）",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="调试参数：本次最多新跑的样本数（resume 过滤后）",
    )
    p.add_argument(
        "--dryrun", action="store_true",
        help="不加载模型：每类别取首条样本构建 prompt 并打印，验证 import 面与数据加载",
    )
    p.add_argument(
        "--output", default=None,
        help="结果 jsonl 路径（默认 metrology/outputs/bfcl_run_{condition}_{cap_tier}.jsonl）",
    )
    p.add_argument(
        "--device", default="auto",
        help="auto=npu>cuda>cpu；NPU 需先 source /usr/local/Ascend/ascend-toolkit/set_env.sh",
    )
    p.add_argument(
        "--max_context_length", type=int, default=None,
        help="覆盖模型 config 的 max_position_embeddings（默认从模型 config 读）",
    )
    p.add_argument(
        "--bfcl_data_dir", default=None,
        help="可选：把 BFCL 数据目录（PROMPT_PATH）重定向到该路径（如服务器上的数据拷贝）",
    )
    p.add_argument(
        "--skip_errors", action="store_true",
        help="resume 时把 error 行也视为已完成（默认 error 行会重跑）",
    )
    return p


def main(argv=None):
    # Windows 控制台默认 GBK：强制 UTF-8 输出，避免中文进度信息乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非必需能力，失败忽略
        pass
    args = build_parser().parse_args(argv)

    if args.condition in ("h2o", "kvzip"):
        raise SystemExit(
            f"condition={args.condition} 未实现：S8.2 实现 snapkv / streamingllm；"
            "h2o / kvzip 在后续 chunk 经 _query_prompting 的 hook 点接入"
        )
    if args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
        if args.c2kv_checkpoint is not None and not Path(args.c2kv_checkpoint).exists():
            raise SystemExit(f"--c2kv_checkpoint 不存在: {args.c2kv_checkpoint}")
        if args.c2kv_ratio < 1:
            raise SystemExit(f"--c2kv_ratio 必须 >= 1: {args.c2kv_ratio}")
    if args.condition in D_ARM_CONDITIONS:
        if args.condition == "c2kv_d_sham_neutral" and not args.d_plan:
            raise SystemExit(
                "condition=c2kv_d_sham_neutral 必须提供 --d_plan"
                "（sham_token_ids 来源；格式见 metrology/d_repair_arms.py docstring）"
            )
        if args.d_plan and not Path(args.d_plan).exists():
            raise SystemExit(f"--d_plan 不存在: {args.d_plan}")
    if not args.dryrun and not args.model:
        raise SystemExit("非 dryrun 模式必须提供 --model")

    if args.output:
        output_path = Path(args.output)
    elif args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
        # 臂名入文件名：防不同 doc_mode/ratio 的 resume 键 (id, cap_tier, condition) 撞车；
        # c2kv_d_* 的 condition 本身已含臂名（c2kv_d_corr 等）
        output_path = REPO_ROOT / "metrology" / "outputs" / (
            f"bfcl_run_{args.condition}-{args.c2kv_doc_mode}-r{args.c2kv_ratio}_{args.cap_tier}.jsonl"
        )
    else:
        output_path = REPO_ROOT / "metrology" / "outputs" / (
            f"bfcl_run_{args.condition}_{args.cap_tier}.jsonl"
        )
    kv_note = ""
    if args.condition in ("snapkv", "streamingllm", "h2o", "kvzip"):
        kv_note = f"  kv_budget={args.kv_budget if args.kv_budget is not None else '50%'}"
    elif args.condition == "c2kv" or args.condition in D_ARM_CONDITIONS:
        kv_note = (
            f"  c2kv_doc_mode={args.c2kv_doc_mode}  c2kv_ratio={args.c2kv_ratio}"
            f"  c2kv_checkpoint={args.c2kv_checkpoint or '(基座+注入,未训练)'}"
        )
        if args.condition in D_ARM_CONDITIONS:
            kv_note += f"  d_plan={args.d_plan or '(无)'}"
    print(f"[runner] condition={args.condition}  cap_tier={args.cap_tier}{kv_note}  "
          f"output={output_path}")

    if args.dryrun:
        run_dryrun(args, output_path)
    else:
        run_inference(args, output_path)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════
# epilog：服务器侧冒烟建议（S8.2 交付说明）
# ══════════════════════════════════════════════════════════════════════════
# 前置：source /usr/local/Ascend/ascend-toolkit/set_env.sh；用 ~/envs/c2kv/bin/python；
# 单测（纯 CPU、不占卡）：
#   ~/envs/c2kv/bin/python -m pytest metrology/test_kv_compress.py -v
# 冒烟（各 1 条样本、cap 128；NPU 空闲门限 ≥20G 时发车，窗口门控脚本参照
#   ~/s4_cover2.sh 用法；结果落 ~/c2kv/outputs_lyc/ 另命名避免与既有归档混淆）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   ~/envs/c2kv/bin/python -m metrology.bfcl_hf_runner \
#     --bfcl_pkg_path <bfcl_eval 包路径> --model ~/c2kv/models/Qwen3-4B-Instruct-2507 \
#     --cap_tier 128 --limit 1 --output ~/c2kv/outputs_lyc/smoke_s8_2_base.jsonl
#   （同命令分别以 --condition snapkv / --condition streamingllm [--kv_budget N] 重跑；
#     核对三行 compression_meta{method,budget,kept_tokens,obs_window,n_sink} 与
#     base 行无该字段；建议每条冒烟选 <2k token 的 multi_turn_base 样本，
#     先看 prefill 注意力物化（每层 (1,H,L,L) bf16）在本卡的峰值显存再放量）。
# S9 c2kv 冒烟（训练 ckpt；untrained 对照臂去掉 --c2kv_checkpoint 即可）：
#   ~/envs/c2kv/bin/python -m metrology.bfcl_hf_runner \
#     --bfcl_pkg_path <bfcl_eval 包路径> --model ~/c2kv/models/Qwen3-4B-Instruct-2507 \
#     --condition c2kv --c2kv_checkpoint ~/c2kv/outputs_lyc/g_joint/<ckpt> \
#     --c2kv_doc_mode joint --c2kv_ratio 8 \
#     --cap_tier 128 --limit 1 --output ~/c2kv/outputs_lyc/smoke_s9_c2kv.jsonl
#   （核对行内 c2kv_meta{checkpoint,trained,ratio,doc_mode,...} 与逐步
#     compression_meta{method="c2kv",gist_tokens,doc_tokens,...}）
