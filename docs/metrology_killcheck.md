# 度量学 kill check memo（S4 / M0 预注册门）

对象：arXiv 2607.02577《Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation》（v1 提交 2026-06-30；作者 Jay Vaghasiya、Vishvesh Bhat（CoreThink AI）、Muhammad Ahmed Mohsin、Asad Aali（Stanford））。发布物：Tool-Veritas 基准与 Harness Lab 开源 harness。

代码获取情况：**可获取**（2026-08-03 公开；论文 v1 Availability 节仅称 "will release"，实际已放出）：
- 主仓 `github.com/CoreThink-AI/benchmarking-the-benchmarks`（README 直接引用 arXiv:2607.02577）
- 镜像仓 `github.com/CoreThink-AI/Tool-Veritas`、`github.com/CoreThink-AI/harness-lab-oss`

判定口径（预注册 §6）：开源 harness 代码中存在可运行实现（文件+行号）方计「已实现」；仅论文文字描述而代码缺失记 DESCRIBED-NOT-IMPLEMENTED，按未实现处理。

## (a) 生成预算充足性检查（按金标目标长度校验 cap）→ **NOT-COVERED（未实现）**

- 论文全文程序化检索 `max_tokens` / `max_new_tokens` / `truncat*` / `generation budget` / `token budget` / `gold length` 均零命中——论文从未讨论生成上限或截断伪影。
- 代码仅有静态 max_tokens 旋钮（默认 4096），无任何按金标输出长度校验/设置上限的逻辑：
  - `Tool-Veritas/benchmark/config.py:110` — `max_tokens: int = 4096`（本 memo 撰写时已复核原文该行）
  - `Tool-Veritas/benchmark/harness/llm_client.py:141,220,364`；`:277` 透传 `"max_tokens": self.max_tokens`
  - `Tool-Veritas/benchmark/runtime/benchmark_runner.py:189` — 原样透传
  - `harness-lab-oss/packages/evals/agent.py:23,29,52`；`clients/openrouter.py:33,50,56`
- 无截断检测：`agent.py:58-59` 仅以 `finish_reason == "stop" and not tool_calls` 判完成，从不检查 `finish_reason == "length"`；Tool-Veritas 全仓 `finish_reason` 仅出现于 mock 返回值（`llm_client.py:412`）。

## (b) 协议发射与语义正确分离的双列评分 → **部分实现，按口径计未实现**

已实现（可运行，已复核关键行）：
- 「确定性 gate vs 定性 judge」双轨分列：`Tool-Veritas/benchmark/gates/gating_engine.py:85-86` — `GateResult` 分列 `execution_rule_results`（max_tool_calls/required_tools/empty_tool_turn，评估逻辑 :149-163，:163 注明 hard failure 无修复窗）与 `state_check_results`（16 类语义状态检查，:184-200 含 repair window）。
- scorecard 分列落盘：`Tool-Veritas/evaluator/evaluator/integration/__init__.py:26-56`（`passed`/`gates_all_passed`/`eval_label` 分列输出），`:79` 合并 `passed = gates_all and eval_score.label == Label.SUCCESS`；`benchmark_runner.py:616-620` 落盘 `evaluation` 与 `gate_results` 分列。
- Harness Lab 对 τ²-Bench 官方 reward 组件（db/action/communication）透传暴露：`harness-lab-oss/apps/api/app/main.py:1962-1996`。

论文描述但代码缺失（DESCRIBED-NOT-IMPLEMENTED）：
- §3.6 式(4) 的 `(C_tool, C_task, C_outcome)` 三分解在全代码库无任何对应实现（`tool_invocation`/`outcome_verification` 零命中）。
- 加权双维评分只有 schema（`benchmark/models/test_case.py:311-318`），无评分代码消费该配置；`overall_score` 实为 0/1 二值。

**关键判定**：其「协议面」实为执行规则约束（调用次数/空调轮/必需工具），**不存在**把「输出是否满足格式/协议要求（schema 合规、闭合块可解析）」作为独立评分列的实现（`llm_client.py:19-27,42` 的内联 tool-call 格式恢复仅用于执行兜底，不产生协议分）。本任务书 (b) 所问的「协议发射 vs 语义正确」双列评分**按口径计未实现**。

## (c) KV/上下文压缩条件 → **NOT-COVERED**

- 论文全文 `kv cache` / `kv_cache` / `compress*` / `context compression` 零命中。
- 三仓代码检索零命中（"compress" 仅 GZip middleware 注释与 BFCL 数据中的 compress_file 工具名；"cache" 均为结果缓存）。

## M0 裁定

停机规则（预注册）：(a) 与 (b) 均已实现 → S8 = KILLED-BY-PRECEDENT。

实测：**(a) NOT-COVERED**；(b) 部分实现但非本任务书口径的双列评分、按口径计未实现；(c) NOT-COVERED。

**M0 不触发。S8 照常执行**，PR-B 相关工作节引用本 memo。我方 delta 确认：压缩条件 × 预算 censoring × 协议/语义双列 × 构成反转——四者在该最近邻工作中均不存在。

## 证据复核记录

- 2026-08-16：调研子代理完成全文+三仓检索；本人独立复核 `config.py:110` 与 `gating_engine.py:85-86` 两处关键行原文一致。
- 论文全文 https://arxiv.org/html/2607.02577v1 ；摘要页 https://arxiv.org/abs/2607.02577 。
