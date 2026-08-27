# 三 benchmark 基线 — checkpoint med_dsingle_joint/checkpoint-4186（g_joint 线）

> 2026-08-27/28，NPU ascend03。推理路径：`benchmarks/hf_server.py`（HF + npu_fusion_attention，
> 与实验 D 同一套 KV 原语），c2kv 臂经 `benchmarks/proxy.py`（8× gist，历史压缩、当前轮 raw）。
> τ² 的 user simulator 走 full 模式（不压缩），只压 agent 视角。

## τ²-bench airline（50 任务，max-steps 60，temperature 0）

| 臂 | reward | 95% CI（任务聚类 bootstrap） | 协议合法率 | 备注 |
|---|---|---|---|---|
| full | **0.34** | [0.22, 0.48] | **100%** | tool_calls 由 hf_server 的 `<tool_call>` 解析器产生 |
| c2kv (8×) | 跑着 | | | 有效压缩比 ≈7.5×（proxy 日志：11480→1529 tok） |

观察：
- full 臂协议合法率 100% —— **该 checkpoint 在 τ² airline 上没有协议层损失**。
  与 D 实验（agent-llm-traces 上 full 有 48 条"名对协议错"）不同：协议瓶颈是
  benchmark/工具池相关的。H1（约束解码）的收益空间要靠 BFCL 验证。
- 平均请求 wall ≈2.9s（含排队；单请求 generate_sec 在 hf_server 侧 1-3s，eager HF 路径）。

## BFCL v4 multi_turn_base（200 例）

| 臂 | 官方 acc | 协议合法率 | 状态 |
|---|---|---|---|
| full | 待测 | 待测 | 生成中 |
| c2kv | 待测 | 待测 | 排队 |

## ToolSandbox（test 子集）

| 臂 | 官方指标 | 状态 |
|---|---|---|
| full | 待测 | 排队（smoke 已通） |

## 成本口径

- 逐请求 `generate_sec`（hf_server）/ `wall_sec`（proxy，含 extract+组装）
  记录在 proxy request log；TTFT/p95 留给 C3 专用计时轮（无排队干扰）。
- 有效压缩比按 ResKV 口径 b=m+r：history raw tok / (gist tok + 保留 raw tok)。
