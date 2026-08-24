# -*- coding: utf-8 -*-
"""D 臂移植的纯 CPU 单测：metrology/d_repair_arms.py + runner 接线。

测试面：
a. 模块层 torch-free：本地无 torch 环境下 import 本模块与 runner 不炸
   （runner 顶层 import D_ARM_CONDITIONS 依赖这一性质）；
b. condition 注册：D_ARM_CONDITIONS 与 runner.CONDITION_CHOICES 同步、
   与 AppWorld 臂名 1:1（"c2kv_" + 臂名）；
c. plan 接口：load_d_plan 合法/非法 JSON 的校验（SystemExit）；
d. 网格几何：plan_k_star / chunk_offsets 的闭式值；
e. gist 计数闭式：gist_tokens_for_lengths 手算对照（dropped-gist 核算的上游）；
f. runner 校验：c2kv_d_sham_neutral 缺 --d_plan 拒绝、--d_plan 路径不存在拒绝、
   --d_plan 参数解析；
g.（torch 守卫）append_precomputed_span_cache 拼接/空 no-op 语义。

模型级验证（微型 gist Qwen3 端到端、d_sham_mech 与纯 c2kv 逐 token 一致性）
在 NPU/服务器侧进行——本地无 torch，见模块 docstring 与交付说明。

运行：pytest metrology/test_d_repair_arms.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrology import bfcl_hf_runner as runner
from metrology import d_repair_arms as dra


# ══════════════════════════════════════════════════════════════════════════
# b. condition 注册
# ══════════════════════════════════════════════════════════════════════════

def test_arm_conditions_one_to_one_with_appworld_modes():
    assert dra.D_ARM_MODES == (
        "d_corr", "d_corr_recompute", "d_corr_all", "d_sham_neutral", "d_sham_mech",
    )
    assert dra.D_ARM_CONDITIONS == [f"c2kv_{m}" for m in dra.D_ARM_MODES]


def test_runner_condition_choices_include_arms():
    for cond in dra.D_ARM_CONDITIONS:
        assert cond in runner.CONDITION_CHOICES
    # 既有条件不丢
    for cond in ("base", "snapkv", "streamingllm", "c2kv"):
        assert cond in runner.CONDITION_CHOICES


# ══════════════════════════════════════════════════════════════════════════
# c. plan 接口
# ══════════════════════════════════════════════════════════════════════════

def _write_plan(tmp_path: Path, payload) -> Path:
    p = tmp_path / "d_plan.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_d_plan_roundtrip(tmp_path):
    p = _write_plan(tmp_path, {
        "multi_turn_base_0": {"k_star": 2, "span_len": 40, "sham_token_ids": [1, 2, 3]},
        "multi_turn_base_1": {"k_star": 0},
    })
    plan = dra.load_d_plan(p)
    assert plan["multi_turn_base_0"]["k_star"] == 2
    assert plan["multi_turn_base_1"].get("sham_token_ids") is None


def test_load_d_plan_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        dra.load_d_plan(tmp_path / "nope.json")


@pytest.mark.parametrize("payload", [
    [1, 2, 3],                                            # 顶层非 object
    {"qid": "not-a-dict"},                                # 条目非 object
    {"qid": {"k_star": "2"}},                             # k_star 非 int
    {"qid": {"sham_token_ids": "abc"}},                   # sham 非列表
    {"qid": {"sham_token_ids": [1, "x"]}},                # sham 元素非 int
])
def test_load_d_plan_rejects_bad_shapes(tmp_path, payload):
    with pytest.raises(SystemExit):
        dra.load_d_plan(_write_plan(tmp_path, payload))


# ══════════════════════════════════════════════════════════════════════════
# d. 网格几何
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n,expected", [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (24, 11)])
def test_plan_k_star(n, expected):
    assert dra.plan_k_star(n) == expected


def test_chunk_offsets():
    chunks = [[0] * 10, [0] * 25, [0] * 7]
    assert dra.chunk_offsets(100, chunks) == [100, 110, 135]
    assert dra.chunk_offsets(0, chunks) == [0, 10, 35]


# ══════════════════════════════════════════════════════════════════════════
# e. gist 计数闭式（ratio=8, grid_width=1024 手算对照）
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("lengths,ratio,residual,width,expected", [
    ([16], 8, "embed-mean", 1024, 2),          # 整除：16/8
    ([17], 8, "embed-mean", 1024, 3),          # 17 → 24 → 3
    ([1], 8, "embed-mean", 1024, 1),           # 1 → 8 → 1
    ([17], 8, "none", 1024, 3),                # 无残差取整：ceil(17/8)
    ([1030], 8, "embed-mean", 1024, 128),      # 钳到网格宽：1024/8
    ([16, 17], 8, "embed-mean", 1024, 5),      # 逐文档累计
    ([0, 16], 8, "embed-mean", 1024, 2),       # 空文档跳过
])
def test_gist_tokens_for_lengths(lengths, ratio, residual, width, expected):
    assert dra.gist_tokens_for_lengths(lengths, ratio, residual, width) == expected


# ══════════════════════════════════════════════════════════════════════════════
# f. runner 校验与 CLI
# ══════════════════════════════════════════════════════════════════════════════

def test_cli_d_plan_parses():
    args = runner.build_parser().parse_args(
        ["--bfcl_pkg_path", "/x", "--condition", "c2kv_d_corr", "--d_plan", "p.json"]
    )
    assert args.d_plan == "p.json"
    assert args.condition == "c2kv_d_corr"


def test_sham_neutral_requires_d_plan():
    with pytest.raises(SystemExit, match="--d_plan"):
        runner.main([
            "--bfcl_pkg_path", "/x", "--dryrun",
            "--condition", "c2kv_d_sham_neutral",
        ])


def test_d_plan_path_must_exist():
    with pytest.raises(SystemExit, match="--d_plan 不存在"):
        runner.main([
            "--bfcl_pkg_path", "/x", "--dryrun",
            "--condition", "c2kv_d_corr", "--d_plan", "/nonexistent/plan.json",
        ])


def test_corr_arm_passes_validation_without_plan(tmp_path, capsys):
    """corr 臂无 plan 应通过 main() 校验段（随后 dryrun 才碰 bfcl 包——
    本测试只断言不提前 SystemExit；/nonexistent 的 bfcl_pkg_path 会让
    dryrun 抛异常，故用 pytest.raises 捕获任意非 SystemExit 异常即算过校验）。"""
    try:
        runner.main([
            "--bfcl_pkg_path", "/nonexistent", "--dryrun",
            "--condition", "c2kv_d_corr",
        ])
    except SystemExit as e:
        pytest.fail(f"校验段误拒 corr 臂（无 plan）：{e}")
    except Exception:  # noqa: BLE001 dryrun 阶段预期失败（无 bfcl 包）
        pass


# ══════════════════════════════════════════════════════════════════════════════
# g. torch 守卫：append_precomputed_span_cache 语义
# ══════════════════════════════════════════════════════════════════════════════

class _FakeLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


def test_append_precomputed_span_cache_concat_and_noop():
    torch = pytest.importorskip("torch")

    prefix = _FakeCache([
        _FakeLayer(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),
        _FakeLayer(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4)),
    ])
    span_kv = [
        (torch.full((1, 2, 2, 4), 7.0), torch.full((1, 2, 2, 4), 8.0)),
        (torch.full((1, 2, 2, 4), 9.0), torch.full((1, 2, 2, 4), 10.0)),
    ]
    out = dra.append_precomputed_span_cache(prefix, span_kv)
    assert out is prefix
    assert prefix.layers[0].keys.shape == (1, 2, 5, 4)
    assert torch.all(prefix.layers[0].keys[..., 3:, :] == 7.0)
    assert torch.all(prefix.layers[1].values[..., 3:, :] == 10.0)
    # 空 span = no-op（d_sham_mech 守卫臂依赖）
    before = prefix.layers[0].keys.clone()
    assert dra.append_precomputed_span_cache(prefix, []) is prefix
    assert torch.equal(prefix.layers[0].keys, before)
