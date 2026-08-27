# H1 — `<tool_call>` 约束解码（XGrammar-2 迁移）

> 迁移手册条目：H1（P0，"立刻挂"）。报告出处：TL;DR(2)、KF5、Q5、迁移表 #2、Rec.1。
> 状态：**进行中** —— 数字在三个 benchmark 跑完后回填。

## 方法与源码

- 论文：XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs（arXiv:2601.04426）。
- 源码库：`mlc-ai/xgrammar`（服务器 `~/method_refs/xgrammar`，已通读）。
- 迁移方式：**不改训练、不动模型**。在 `benchmarks/hf_server.py` 的 decode 循环上挂
  `xgrammar.contrib.hf.LogitsProcessor`，grammar 用
  `xgr.get_model_structural_tag("qwen_3", tools=..., reasoning=False)` 编译
  （`GrammarCompiler.compile_structural_tag`，按工具池 sha256 缓存）。
  约束语义：`<tool_call>` 触发后强制
  `{"name": <池内枚举>, "arguments": <该工具 JSON schema>}`，块外文本自由。
  这正是 XGrammar-2 的 structural-tag 机制，非 CUDA 设备（NPU）每 token 走
  CPU bitmask（xgrammar 自带 fallback），其开销即报告里的"constraint tax/成本"轴。

## 臂设计（benchmarks/arms.py）

| 臂 | 历史 | 约束 | 回答的问题 |
|---|---|---|---|
| full | 原文 | 无 | 基线上界（已有） |
| c2kv | 8× gist | 无 | 压缩损失（已有） |
| cd_full | 原文 | 有 | 约束对基线协议合法率的抬升（= 严格修复上界的移动） |
| cd_c2kv | 8× gist | 有 | 约束 × 压缩叠加；协议列应≈cd_full，语义列看 constraint tax |

对照 `lenient_parse`：不改解码、只把评分端 JSON 解析放宽（琐碎语法错误的占比），
在分析阶段对同一批 full 输出重打分，不需要新跑。

## 评测面与指标

- τ²-bench airline（50 任务）：官方 reward（语义列）+ 全 assistant 轮的
  `benchmarks/metrics.py:protocol_columns_for_turn`（协议列：名字在池内 +
  参数过 schema + 无残破 `<tool_call>`）。
- BFCL v4 multi_turn_base（200 例）：官方 AST/执行检查器 + 同一协议列。
- ToolSandbox（test 子集）：官方 evaluation + 同一协议列。
- 成本列：`generate_sec`（hf_server 逐请求）对比 cd vs 非 cd（同臂），
  即每 token 约束开销；TTFT/p95 从 proxy request log 计算。
- 发射率监控（constraint tax，报告 Q5）：每任务平均 tool_calls 数
  （约束可能压低发射）。

## 预期与 kill 判据（来自迁移手册）

- 预期：cd_full 协议合法率 ≈100%（XGrammar-2 在 BFCL-v3 的硬保证）；
  full 臂的 48 条"名对协议错"类失败应被消灭。
- Kill：语义列（工具名正确/reward）因约束下降 >2pp（constraint tax 成立），
  或发射率显著下降 → 改用软版本（只在检测到非法时二次约束解码）。

## 结果（待回填）

| 臂 | τ² reward | τ² 协议合法率 | BFCL acc | BFCL 协议合法率 | TS acc | 发射率 | 约束开销/token |
|---|---|---|---|---|---|---|---|
| full | 0.33 (N=50) | 待测 | 待测 | 待测 | 待测 | 待测 | — |
| c2kv | 跑着 | 待测 | 待测 | 待测 | 待测 | 待测 | — |
| cd_full | | | | | | | |
| cd_c2kv | | | | | | | |

## 我的 evaluation 与 insight（跑完后写）
