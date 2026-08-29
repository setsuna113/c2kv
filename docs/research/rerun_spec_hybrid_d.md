# 重跑规格 — hybrid × D 组合实验（2026-08-29）

> 三个正确性缺陷被定位并修复后，哪些已产出的数字作废、哪些仍然有效、按什么命令重跑。
> 原规格不变：三个 benchmark 全跑，臂表照旧，sham 仍是原规格里那一个对照臂
> （不新增逐块 sham 扫描）。

## 1. 作废 / 保留

| 产出 | 状态 | 原因 |
|---|---|---|
| bench τ² **c2kv 0.10**（`h1_constrained_decoding.md`） | **作废** | proxy 把历史里每个 assistant 动作压成字面量 `""` |
| bench τ² **full 0.34 / cd_full 0.34**、协议 100% | **有效** | full/cd_full 不走压缩路径 |
| bench BFCL **full 2.5% / c2kv 2.5%** | full 有效，**c2kv 作废** | 同上；且 BFCL 存在地板效应（见 §4） |
| bench ToolSandbox **full 0.125** | **有效** | 同上 |
| battery D 线全部（`d_r2/`：none/sham/corr/corr_re/corr_all/full，n=93） | **有效** | 不经 proxy，也不经 `_build_hybrid_prefix` |
| K1 `corr@first 0.4086`、C3 crossover、A1/A2、B1 | **有效但无仓库内原始产物** | 报告指向服务器 `~/bench_results/`，见 §5 |
| 任何 hybrid 臂的数字 | **不存在** | 全仓库没有一行 `mode=hybrid` 的结果 |

## 2. 修了什么

**① proxy 删除了历史里的每一个 assistant 动作**（`benchmarks/proxy.py`）

OpenAI 的 assistant 工具调用轮 `content=None`、动作在 `tool_calls` 里。旧代码
`json.dumps(content or "")` 把它变成两个字符的 `""` 再送去 `/v1/c2kv/extract`。
被压缩的历史里，**agent 自己做过的每一步都被删掉了，不是被压缩**。
hf_server 对 *raw* 消息渲染训练方言（`Action:\n<tool_call>…`）、对压缩消息只用 gist，
所以缺陷精确地只落在压缩侧——`full` 无损，`c2kv` 灾难，`hybrid` 介于两者之间
（保原文的尾巴是好的，压缩的头部被清空）。**任何今天跑出来的 Block 1 对比表，
测的都是这个 bug，不是算法。**

修法：新增 `benchmarks/dialect.py`，proxy 和 hf_server 共用同一个渲染器，
压缩前先把 `tool_calls` 渲染进 `content`。

**② battery 的 hybrid 前缀时序是反的**（`agent/eval_agent_history_c2kv.py`）

`_build_hybrid_prefix` 的 `hybrid_full_after_c2kv` 默认 `False`，布局是
`S → R_最近k → G_更早的`——最近的原文排在更早的压缩块**前面**。

为什么默认是反的：`True` 那条分支用 `append_full_history` 调的是**工具定义 eval 的**
`_prefill_tokens_with_cache`，它的 attention mask 按 `past_length + input_length`
（逻辑位置）算，而 cache 里有 gist 时槽位远少于逻辑位置——`d_prereg.md` 的
"Suffix recompute" 一节明文警告过这一点。**所以正确的时序那条分支是坏的，
默认值是在绕开 bug，不是在表达算法。**

同一个文件里 `_build_raw_first15_hybrid_prefix`（L957）一直是对的：gist 在前、
原文尾巴在后、用 `_prefill_tokens_with_cache_maybe_gist(use_gist=bool(rest))`。
`use_gist` 不是小开关——它切换到 `gist_{q,k,v}_proj` 这一整套投影权重
（`modeling_qwen3.py:242-246`），训练的规则是"cache 里有 gist ⇒ 之后每次前向都用
gist 投影"。旧的 hybrid 尾巴用的是普通投影。

修法：`_build_hybrid_prefix` 对齐到那个参考实现。新增 `--hybrid_layout`
（默认 `chronological`），旧行为保留为 `legacy_tail_first` 仅供复现旧行。
每行结果都盖上 `hybrid_layout` / `hybrid_top_k_effective` / `hybrid_degenerate_to_full`。

**③ 带 `repair` 的臂会被静默当成普通压缩臂跑完**（`benchmarks/arms.py`）

docstring 写着"proxy 会拒绝"，但 `proxy._assemble` 从来不读 `Arm.repair`。
修法：`validate()` 现在直接 `NotImplementedError`。Block 2 的服务端 KV 原语
（保留原始 past_key_values + RoPE 归位后 concat）**仍未实现**——见 §6。

## 3. "算法是什么"——口径确定

| | 定义 | 依据 |
|---|---|---|
| **history** | 一条消息 = 一个 doc；超过 `max_doc_length`(768) 的消息被**切**成多个 doc，不合并 | `_fit_reused_history` → `_split_message_to_fit` |
| **doc 选择** | 超过 `max_doc_num`(16) 时，`tail` 策略保 **`[doc_0] + 最后 15 个`** | `_select_history` L1072-1074 |
| **hybrid 的 k** | **k 个 doc**（≈ k 条消息），不是 k 轮 | 一个 agent 轮 = assistant 动作 + tool 结果 = 2 条消息，所以 k=3 ≈ 1.5 轮 |
| **hybrid 布局** | `S → G_0..G_{T-k-1} → R_{T-k}..R_{T-1} → Q` | 见 ②；与 bench proxy 的时序一致 |
| **`offset:j`** | j 索引**全部** history doc（含被保为原文的尾巴），带越界检查 | `CORR_K_POLICY` 解析处 |

**为什么 doc 才是正确的单位**：gist 是**逐 doc** 产生的，压缩单位就是 doc。
"保留半轮原文"在这个栈里无法表达。计划里的"最近 k 轮"是宽松说法，
k=1/3/5 按 doc 读即可——bench 和 battery 因此是可比的。

**`doc_0` 是被选择策略特别保留的**（`[doc_0] + 最后15`）。这解释了为什么
`offset:0` 是个良定义且可能特殊的位置——不是巧合。

## 4. 重跑命令（原规格：三个 benchmark，全臂）

服务器上，模型服务已起（`~/bench_logs/launch_hf.sh`）。臂：
`full / c2kv / hybrid1 / hybrid / hybrid5`（k=1/3/5 已注册进 `arms.py`）。

```bash
# Block 1 — P 层对比表
for BM in tau2 bfcl toolsandbox; do
  PORT=34100
  for ARM in full c2kv hybrid1 hybrid hybrid5; do
    ~/envs/bench/bin/python benchmarks/run.py \
      --benchmark "$BM" --arm "$ARM" \
      --upstream http://127.0.0.1:34000 \
      --proxy-port $((PORT++)) \
      --run-name "c2kv_${BM}_${ARM}" \
      --out "results/bench/${BM}_${ARM}"
  done
done
```

`--run-name` 现在按臂区分：原来所有臂共用一个名字，**后一个臂会覆盖前一个臂的
轨迹**。每次运行现在还会写 `rows_<arm>.jsonl`（逐任务行），这是 oracle 连接的前提，
以前这些行建好就被丢掉了。

```bash
# Block 2 触发集 — 任务级 oracle（full 对、压缩臂错）
for ARM in c2kv hybrid; do
  python benchmarks/oracle_subset.py \
    --reference results/bench/tau2_full/rows_full.jsonl \
    --target    results/bench/tau2_${ARM}/rows_${ARM}.jsonl \
    --out       results/bench/tau2_oracle_${ARM}.txt
done
# 产出 .json 里带 n_paired / n_trigger / trigger_rate_L1，
# 以及 unpaired（只在一个臂里出现）和 unscored——两者都单列，不静默丢弃。
```

`--task-ids @<file>` 目前只有 BFCL 能用（它的 CLI 有 `--run-ids`）。
τ² 和 ToolSandbox 传了会**直接报错退出**，不会静默跑全集然后冒充 oracle 子集。

```bash
# Block 3.2 — battery：full / c2kv / hybrid k=3（约 3000 题，见 §5 功效）
MAX_EXAMPLES=3000 COMPARE_MODES=full,c2kv,hybrid \
HYBRID_TOP_K=3 HYBRID_LAYOUT=chronological \
  bash agent/eval_agent_history_c2kv_npu.sh

# 触发集：full 对、hybrid 错
python agent/extract_cw_triggers.py \
  --full_rows       results/battery/full.jsonl \
  --compressed_rows results/battery/hybrid_r8.jsonl \
  --base_hybrid_top_k 3 \
  --batch batch-TF-hybrid-k3 --s_metric tool_name_match \
  --chunk_policy fixed --ckpt_path "$G/fixed_joint" \
  --model_sha ... --eval_code_sha ... \
  --out_manifest configs/bdf_pilot/d_cw_manifest_hybrid_k3.json \
  --out_bundles   results/d/bundles_hybrid_k3.jsonl \
  --out_doc_table results/d/d_doc_ids_hybrid_k3.json

# 顺带：c2kv 对、hybrid 反而错（原规格要求的计数）
python agent/extract_cw_triggers.py \
  --full_rows results/battery/c2kv_r8.jsonl \
  --compressed_rows results/battery/hybrid_r8.jsonl \
  --base_hybrid_top_k 3 --batch batch-TF-hybrid-regression ...
```

```bash
# Block 3.3 — hybrid 基座上的修复臂
for ARM in hybrid_none sham corr full; do
  python agent/d_kv_intervene.py --arm "$ARM" \
    --base_hybrid_top_k 3 --hybrid_layout chronological \
    --corr_k_policy offset:0 \
    --manifest configs/bdf_pilot/d_cw_manifest_hybrid_k3.json \
    --output_file results/d_hybrid_k3/d_${ARM}.jsonl
done

# 逐块扫描：只扫被压缩的块。j >= n_docs-k 的块本来就是原文，
# 追加它自己的 KV 是重复而不是修复——harness 会以
# d_hybrid_k_star_in_raw_tail 跳过，不会算成一次失败的修复。
for J in $(seq 0 12); do
  python agent/d_kv_intervene.py --arm corr \
    --base_hybrid_top_k 3 --corr_k_policy "offset:${J}" \
    --manifest configs/bdf_pilot/d_cw_manifest_hybrid_k3.json \
    --output_file "results/d_hybrid_k3/scan/d_corr_off${J}.jsonl"
done
```

**新的护栏**：`--base_hybrid_top_k` 被写进 trigger manifest 的 `kv_recipe`，
`_assert_recipe_matches_run` 会拒绝用一个基座的触发集去修另一个基座。
**旧 manifest 缺这个字段一律按 0 处理**（它们确实都是纯 C2KV 基座冻结的），
所以拿旧 manifest 配 `--base_hybrid_top_k 3` 会直接 FATAL——这正是实际会发生的误用。

## 5. 跑之前必须知道的三件事

**① 逐块扫描的天花板是 max-over-B 统计量。** 从 `d_corr.jsonl` 数出来的每题块数
B：均值 9.98、中位数 11，**93 题里 41 题 B=16**。在"任意一块的救活率 = sham 率
0.0968"的纯零假设下，按实际 B 分布，"至少一块能救"的期望率是 **0.559**；
p=0.15 时 0.679。也就是说这个数天然会落在 55%–80%，**其中大部分不是信号**。
按原规格（单个 sham 臂，不做逐块 sham 扫描）跑，这个数只能当"oracle 定位器的
上界"报，**不能和单臂 sham 的 0.0968 相减，也不能读成"修复率"**——那个减法的零分布不对。
要把它变成可归因的量，需要的是逐块 sham 扫描；本次不做，所以这条限制必须写进表注。

**② 触发集会变小，功效随之下降。** 结构性事实：hybrid k=3 在 B≤3 的题上把**整段历史
都保成原文**（`history[-3:]` 在长度 ≤3 的列表上返回全部），即 hybrid ≡ full，
这些题**不可能进触发集**。当前 93 题里有 **23 题（24.7%）是 B≤3**。
所以 hybrid 的 C→W 率上界是 70/900 = 7.8%。以 n=93 / MDE 17-25pp 为锚按 1/√n 外推：

| 基座 N | hybrid C→W | 触发 n | MDE |
|---:|---:|---:|---|
| 900 | 0.078 | 70 | 20–29pp（勉强） |
| 900 | 0.050 | 45 | 24–36pp（**不够**） |
| 3000 | 0.078 | 234 | 11–16pp ✓ |
| 3000 | 0.050 | 150 | 13–20pp ✓ |

**主对比（corr@first − sham）在 3000 基座上站得住，在 900 上站不住。**
加性分析要的是 2×2 重合列联表，有效 n 是不一致对（约 n/3）：即使 3000 基座
MDE 也在 19–28pp——**加性这一问按当前设计只能给方向，不能给结论**。这条也要写进表注。

**③ BFCL 目前没有可测空间。** `h1_constrained_decoding.md`：full 2.5% / c2kv 2.5%，
报告自己写"瓶颈非压缩"。地板上跑 5 个臂会得到 5 个 2.5%。按原规格照跑，
但结果要按"该 checkpoint 在 BFCL 格式上不成立"来读，不是"压缩无损"。
ToolSandbox 只有 3 个场景，同样只能读方向。
τ² 是唯一有信号的（full 0.34 vs c2kv，n=50，臂间 MDE ≈ 18.6pp）——
**k=1/3/5 之间几个百分点的差，三个 benchmark 都分辨不出来。**

## 6. 仍然没做的

- **Block 2 的服务端 KV 修复原语**。`hf_server` 抽 gist 后就丢掉原始
  `past_key_values`（`_append_gist` 只存 blend 后的 gist 张量），所以 corr@first
  要追加的那份 raw KV 服务端根本不存在。需要：extract 时额外留每块的 per-layer
  K/V，追加时用已经 import 的 `rotate_k_cache_rope` 归位（即 battery
  `_append_span_cache` 的移植）。在此之前 `validate()` 会拒绝任何 repair 臂。
- **τ² / ToolSandbox 的 task-id 过滤**，oracle 子集重跑目前只有 BFCL 能做。
- **harm 臂**：`d_harm_manifest_r2.json`（188 qid，C→C ∪ W→C）已冻结但从未跑过，
  所以修复的**无条件**效应（对本来就对的题的伤害）一个数都没有。
  计划第三问"谁帕累托最优"在这个数出来之前无法回答。
- `hf_server` 的 `GIST_IMPL` 硬编码 `npu_fusion_attention` 且无 CLI 覆盖，
  而 D/battery 线跑 eager——两个栈现在还没法配成同一个 config 做逐 token 自检。
