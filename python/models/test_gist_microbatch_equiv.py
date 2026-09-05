"""Gist 分组 microbatch 与逐篇(mb=1)的数值等价性回归测试。

2026-08-29 审计 I4 实锤: ``C2KV_GIST_DOC_MICROBATCH>1`` 时 layer-0 embed-mean
残差在"组 max padded 网格"上切 chunk(旧 ``_apply_gist_residual_interleave``),
而 attention mask 按 per-sample 真实长度建——组内 L < 组 max 且 L%ratio!=0 的
文档, 其最后一个有效 gist row 的均值混入 [L, 组max) 区间的填充 embedding
(gist_token_id), 该 row 有效、进 past_key_values → 训练目标被污染。arm-1 因此
作废, 生产默认一度回退 mb=1(实测 27-31 s/it vs mb=16 的 4.1 s/it, ~7x)。

2026-08-30 修复: 残差改按每篇文档真实长度(generate_gist 从 2D attention mask
求和注入 ``gist_token_true_lens``)分块取均值; 组内全等长保留原向量化快路径
(逐位不变)。修复后分组前向在数学上等价逐篇——attention 本就是 per-row mask,
跨样本(乃至 bs=2 时跨训练样本的中间组)只剩纯 batching, 无信息串流。本文件
是该声明的硬门槛:

- 函数级: 混合长度组(覆盖 %8 余 0/1/3/7、组 max 自身、极短文档), float64
  逐位比较 per-doc 参考(每篇裁到自己长度, mb=1 语义) vs 修复后分组路径;
  另验证梯度(填充位梯度严格为 0, 内容位梯度与 per-doc 参考逐位一致)。
- 集成级: tiny Qwen3(CPU, float64, eager) 整层 ``generate_gist``: 逐篇 vs
  整组(batch=混合长度组) 的 gist K/V(含 layer-1, 即穿过了 layer-0 attention
  的完整堆叠)有效行对齐。

回归锚点(修复前旧路径在本 fixture 下的实测偏差, 详见 test_residual_per_doc...
内的打印; 旧代码未内联、不作断言):
  函数级 fixture(填充 embedding 幅值 ~1000, 本文件 DOC_LENS): 旧 padded-grid
  路径在各文档有效行上的 max abs 偏差实测 [0, 8.77e2, 6.26e2, 1.25e2, 0,
  6.26e2, 0]——L%8!=0 的短文档(17/19/23/3)最后一个有效 gist row 被填充以
  最高 7/8 权重混入; L%8==0 文档与组 max 文档偏差恰为 0(旧代码对它们本就
  正确)。集成级(tiny Qwen3, CPU float64 eager): 分组 vs 逐篇 gist K/V 实测
  逐位一致(max abs dev = 0.0), 故集成断言直接按逐位(rtol=0, atol=0)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from models.gist_utils import GistConfigMixin, get_apply_gist_residual_func  # noqa: E402

RATIO = 8
# 混合长度组: %8 余 0(24, 8)、1(17)、3(19, 3)、7(23, 31); 31 = 组 max 自身;
# 3 = 极短文档(< ratio, 只有 tail chunk)
DOC_LENS = [24, 17, 19, 23, 31, 3, 8]
GROUP_MAX = max(DOC_LENS)
HIDDEN = 16


def _residual_func():
    config = GistConfigMixin(gist_type=f"interleave-{RATIO}", gist_residual_type="embed-mean")
    return get_apply_gist_residual_func(config, layer_idx=0)


def _gist_nums(l: int) -> int:
    # 与 _build_interleave_mask_vectorized 相同: embed-mean 先把 seqlen 补齐到
    # ratio 的倍数(clamp 到组宽), gist 行数 = ceil(补齐后 / ratio)
    if l == 0:
        return 0
    padded = l if l % RATIO == 0 else min(GROUP_MAX, l + RATIO - l % RATIO)
    return (padded + RATIO - 1) // RATIO


def _fixture(dtype=torch.float64, grad=False):
    torch.manual_seed(0)
    b = len(DOC_LENS)
    tokens = torch.randn(b, GROUP_MAX, HIDDEN, dtype=dtype)
    # 填充槽放大三数量级: 让"混入填充"在旧路径下产生肉眼可见的偏差,
    # 同时等价性断言不依赖填充值(修复后路径根本不读它们)
    filler = 1000.0 + torch.randn(b, GROUP_MAX, HIDDEN, dtype=dtype)
    for d, l in enumerate(DOC_LENS):
        tokens[d, l:] = filler[d, l:]
    gist = torch.randn(b, (GROUP_MAX + RATIO - 1) // RATIO, HIDDEN, dtype=dtype)
    if grad:
        tokens.requires_grad_(True)
        gist.requires_grad_(True)
    true_lens = torch.tensor(DOC_LENS)
    return tokens, gist, true_lens


def _per_doc_reference(fn, tokens, gist, true_lens):
    """mb=1 语义: 每篇裁到自己长度单独过残差(即生产回退 mb=1 的行为)。"""
    refs = []
    for d, l in enumerate(true_lens.tolist()):
        gn = _gist_nums(l)
        ref = fn(tokens[d : d + 1, :l], gist[d : d + 1, :gn], ratio=RATIO,
                 gist_token_true_lens=true_lens[d : d + 1])
        refs.append(ref)
    return refs


def test_residual_per_doc_bitwise_matches_mb1():
    fn = _residual_func()
    tokens, gist, true_lens = _fixture()
    grouped = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=true_lens)
    refs = _per_doc_reference(fn, tokens, gist, true_lens)
    for d, l in enumerate(DOC_LENS):
        gn = _gist_nums(l)
        # 有效 gist 行: 与逐篇参考**逐位**一致
        torch.testing.assert_close(grouped[d, :gn], refs[d][0], rtol=0, atol=0)
        # 无效行(gist_mask 外, 不会被 attend): 有限即可
        assert torch.isfinite(grouped[d]).all()


def test_residual_old_padded_grid_deviation_anchor():
    """回归锚点: 修复前旧路径(gist_token_true_lens=None, 组 max padded 网格)在本 fixture
    下确实偏离逐篇参考——证明上面的等价性测试不是恒真。只断言"污染存在",
    不绑定旧代码的具体数值(量级见模块 docstring)。"""
    fn = _residual_func()
    tokens, gist, true_lens = _fixture()
    old = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=None)  # 修复前行为
    refs = _per_doc_reference(fn, tokens, gist, true_lens)
    devs = []
    for d, l in enumerate(DOC_LENS):
        gn = _gist_nums(l)
        devs.append((old[d, :gn] - refs[d][0]).abs().max().item())
    print(f"\n[anchor] old padded-grid max-abs deviation per doc (lens={DOC_LENS}): "
          f"{[f'{x:.3e}' for x in devs]}")
    polluted = [d for d, l in enumerate(DOC_LENS) if l != GROUP_MAX and l % RATIO != 0]
    clean = [d for d, l in enumerate(DOC_LENS) if l == GROUP_MAX or l % RATIO == 0]
    assert all(devs[d] > 1.0 for d in polluted)  # 填充(幅值~1000)必混入 -> 大偏差
    assert all(devs[d] == 0.0 for d in clean)  # %8==0 与组 max: 旧代码本就正确


def test_residual_per_doc_gradients_match_mb1():
    fn = _residual_func()
    tokens, gist, true_lens = _fixture(grad=True)
    grouped = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=true_lens)
    grouped.sum().backward()
    ref_tokens, ref_gist, ref_lens = _fixture(grad=True)
    refs = _per_doc_reference(fn, ref_tokens, ref_gist, ref_lens)
    for ref in refs:
        ref.sum().backward()
    for d, l in enumerate(DOC_LENS):
        # 填充位: 修复后路径完全不读 -> 梯度严格为 0(旧路径此处非 0, 即污染)
        assert tokens.grad[d, l:].abs().sum() == 0
        # 内容位: 与逐篇参考逐位一致
        torch.testing.assert_close(tokens.grad[d, :l], ref_tokens.grad[d, :l], rtol=0, atol=0)
        gn = _gist_nums(l)
        torch.testing.assert_close(gist.grad[d, :gn], ref_gist.grad[d, :gn], rtol=0, atol=0)


def test_residual_equal_length_group_fast_path_unchanged():
    """组内全等长: 向量化快路径与旧实现逐位不变(true_lens 有无都一样)。"""
    fn = _residual_func()
    torch.manual_seed(1)
    b, s = 4, 24
    tokens = torch.randn(b, s, HIDDEN, dtype=torch.float64)
    gist = torch.randn(b, s // RATIO, HIDDEN, dtype=torch.float64)
    base = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=None)
    with_lens = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=torch.full((b,), s))
    torch.testing.assert_close(base, with_lens, rtol=0, atol=0)
    # s % ratio != 0 的等长组同样走快路径
    tokens = torch.randn(b, s + 3, HIDDEN, dtype=torch.float64)
    gist = torch.randn(b, (s + 3 + RATIO - 1) // RATIO, HIDDEN, dtype=torch.float64)
    base = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=None)
    with_lens = fn(tokens, gist, ratio=RATIO, gist_token_true_lens=torch.full((b,), s + 3))
    torch.testing.assert_close(base, with_lens, rtol=0, atol=0)


def test_generate_gist_grouped_matches_mb1_tiny_qwen3():
    """集成级: tiny Qwen3(CPU, float64, eager) 整层 generate_gist。

    逐篇(每篇裁到自己长度, 即 _generate_gist_for_context_docs microbatch=1)
    vs 整组(microbatch=16 语义: 混合长度组一次前向)。比较每层 gist K/V 的
    有效行(layer-1 的 K/V 已穿过 layer-0 attention, 覆盖完整堆叠)、gist_mask
    与 position_ids。本机(CPU, float64, eager)实测逐位一致(max abs dev = 0),
    断言按逐位; 若换 BLAS/硬件出现末位 ulp 噪声, 可放宽到 rtol=1e-10。
    """
    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        pad_token_id=0,
        dtype=torch.float64,
        gist_type=f"interleave-{RATIO}",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_token_id=100,
    )
    config._attn_implementation = "eager"
    model = Qwen3Model(config).to(torch.float64).eval()

    torch.manual_seed(2)
    ids = torch.zeros(len(DOC_LENS), GROUP_MAX, dtype=torch.long)
    mask = torch.zeros(len(DOC_LENS), GROUP_MAX, dtype=torch.bool)
    for d, l in enumerate(DOC_LENS):
        ids[d, :l] = torch.randint(1, 99, (l,))
        mask[d, :l] = True

    with torch.no_grad():
        g_out, g_mask, g_pos = model.generate_gist(ids, mask)
        refs = []
        for d, l in enumerate(DOC_LENS):
            r_out, r_mask, r_pos = model.generate_gist(ids[d : d + 1, :l], mask[d : d + 1, :l])
            refs.append((r_out, r_mask, r_pos))

    max_dev = 0.0
    for d, l in enumerate(DOC_LENS):
        gn = _gist_nums(l)
        r_out, r_mask, r_pos = refs[d]
        assert int(g_mask[d].sum()) == gn == int(r_mask[0].sum())
        assert torch.equal(g_mask[d, :gn], r_mask[0])
        assert torch.equal(g_pos[d, :gn], r_pos[0])
        for layer in range(config.num_hidden_layers):
            for g_t, r_t in zip(g_out.past_key_values[layer], r_out.past_key_values[layer]):
                dev = (g_t[d, :, :gn] - r_t[0]).abs().max().item()
                max_dev = max(max_dev, dev)
                torch.testing.assert_close(g_t[d, :, :gn], r_t[0], rtol=0, atol=0)
    print(f"\n[integration] tiny Qwen3 generate_gist grouped vs mb=1 max abs dev: {max_dev:.3e}")
