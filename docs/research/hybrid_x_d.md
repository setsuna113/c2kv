# hybrid × D 组合实验报告

Status: IN PROGRESS (2026-08-29). Branch `task/hybrid-repair`. Checkpoint
`checkpoints_upstream/checkpoint-1088`（= upstream
`qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`；与 D 线历史模型
`fixed_joint` **config 相同、权重不同**——本实验全部数字在 1088 上重新基线化）。

三个问题（任务书）：

1. hybrid 把错误压低后，剩下的错还有多少能被 corr@first 救回？（加性还是冗余）
2. 救回来时是追加哪一块历史起的作用？"第一块"先验在 hybrid 基座上还成立吗？
   自动定位器的天花板多高？
3. 三方案（纯 c2kv+修复 / 纯 hybrid / hybrid+修复）在真实任务上的成功率与
   成本，谁帕累托最优？

## 0. 前置工作（本任务引入的基础设施变化）

- **hybrid 统一为 gist_first 原顺序布局**（docs/hybrid_spec.md）：battery 默认
  翻转（原 raw-first 重排为 legacy `--hybrid_layout raw_first`），并修复
  gist_first 下 raw-tail 预充不带 gist 投影的潜在 bug（use_gist 全局规则，
  与 hf_server/训练一致）。bench 栈本就是原顺序，无变化。
- **D 干预支持 hybrid 基座**：`d_kv_intervene --base hybrid --hybrid_top_k`，
  基座直接复用 `_build_hybrid_prefix`（单实现）；gist_first 保留原始绝对位置，
  追加机制与纯 c2kv 基座完全同构（offset:j 必须落在压缩块 j < T−k）。
- **bench 修复臂上线**：hf_server `_append_raw_block`（scratch raw pass 于原始
  logical offsets，目标块 span 不旋转 cat 到请求 cache 末尾）+ proxy
  `c2kv_repair`/`hybrid_repair` 臂 + `repair_oracle.py`（任务级 oracle：
  full✓∧base✗ 子集的 eligible/task/score）。
- **对拍自检**：`selfcheck_repair_vs_dharness.py`（同题同 ckpt，双基座，
  块/cache 形状/KV 张量/贪心 decode 四级对比）。结果：TODO
- **队列事故取证与修复**（2026-08-29 上午，~/bench_queue）：`10#` 十进制解析
  十六进制哈希 → PPORT 空 → 代理绑 hf 端口失败 + 空 cleanup grep 误杀全部代理
  → 当日全部 proxy 臂任务作废（full 臂三例幸存：ts_full 有效、tau2_full 有效、
  bfcl_full 被后续 hf_server 重启波及判废重排）。修复：`16#`、cleanup 守卫、
  代理健康检查、hf_server 代码戳重启、BFCL 结果归档、三个 benchmark 的
  id 子集限制（tau2 --task-ids / bfcl --run-ids / ts -s）。

## 1. P 层对比表（bench，真实任务面）

表：每臂 × {τ² airline (n=50), BFCL multi_turn_base (n=200), ToolSandbox
test 子集 (n=3)}。列为官方成功率 + 协议合法率 + 成本（proxy 记账：gist/
original tokens、wall、有效压缩比；TS 为 test 子集，只读大效应）。

| benchmark | 臂 | 成功率 | 协议合法率 | gist/orig tokens | 有效压缩比 | wall |
|---|---|---|---|---|---|---|
| τ² | full | TODO | | | | |
| τ² | c2kv 8× | TODO | | | | |
| τ² | hybrid k=1 | TODO | | | | |
| τ² | hybrid k=3 | TODO | | | | |
| τ² | hybrid k=5 | TODO | | | | |
| BFCL | (同上 5 臂) | TODO | | | | |
| TS | (同上 5 臂) | TODO | n/a | | | |

单种子；τ² n=50 / BFCL n=200 / TS n=3，只读大效应（几个百分点不作结论）。

## 2. 修复臂（bench，oracle 触发子集）

eligible = 任务级 full✓ ∧ base✗；修复臂 = corr@first（追加第一块压缩历史的
raw KV 于原始位置）。回答"修复在真实任务上能兑现多少"。

| benchmark | base | n_eligible | 修复后成功 | 救回率 | 修复成本（block tokens / prefill sec） |
|---|---|---|---|---|---|
| τ² | c2kv | TODO | | | |
| τ² | hybrid k3 | TODO | | | |
| BFCL | c2kv | TODO | | | |
| BFCL | hybrid k3 | TODO | | | |
| TS | c2kv | TODO | | | |
| TS | hybrid k3 | TODO | | | |

## 3. 机制表（battery，agent-llm-traces eval @1088，768/16，ratio 8）

- 触发器：C→W(c2kv) n=TODO；C→W(hybrid k3) n=TODO；c2kv✓∧hybrid✗（hybrid
  回退数）n=TODO。
- 修复率（corr@first − sham，同 ckpt 参照）：

| 基座 | sham | corr@first | 净修复 | full 上限 |
|---|---|---|---|---|
| c2kv（参照） | TODO | TODO | TODO | — |
| hybrid k3 | TODO | TODO | TODO | TODO |

- **加性**：hybrid 直接修好集 vs corr@first 修好集的重合度/并集覆盖 → TODO
- **位置分布**（逐块扫描 offset:j，j 遍历压缩块）：能救块直方图、
  P(j=0|有块能救)、至少一块能救比例（定位器天花板）→ TODO

## 4. 三问回答

1. **能不能叠加**：TODO（第 1+3 块合看：组合覆盖 vs 单独）
2. **修哪块**：TODO（第一块先验是否迁移到 hybrid 基座 + 天花板）
3. **谁划算**：TODO（bench 成本列 + K1 字节参照：corr@first ≈ 1/6.5 全量
   重算字节；C3 交叉点 ~4K token）

## 附：产物路径

- bench：~/bench_results/（task_*、bfcl_archive/<task>/）、
  ~/benchmarks/tau2/data/simulations/<task>/、~/bench_logs/proxy_task_*.jsonl
- battery：~/c2kv-bdf/results/hxd/battery_1088.parts/{full_r1,c2kv_r8,hybrid_r8}.jsonl
- D 臂/扫描：~/c2kv-bdf/results/hxd/d_*.jsonl（生成后回填）
- 分析：`agent/d_hybrid_repair_analysis.py --json-out results/hxd/combo_analysis.json`
