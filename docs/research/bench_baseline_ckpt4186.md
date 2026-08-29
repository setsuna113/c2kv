# 三 benchmark 基线 — 双 checkpoint 对照（ckpt-4186 vs 上游 ckpt-1088）

> 推理路径：`benchmarks/hf_server.py`（HF + npu_fusion_attention，与实验 D 同套 KV 原语），
> c2kv/cd_c2kv 臂经 `benchmarks/proxy.py`（8× gist；user simulator 恒 full 模式）。
> **口径注记**：4186 行为 256/2048 max-token 口径（review 重跑时点），上游行为对齐口径
> （4096/0/no-thinking）；两边 thinking 均关、温度均 0。渲染方言=训练方言
> （Action:/minified，见 README）。协议合法率经修复公式（False 计入分母）。
> 2026-08-28/29。上游 ckpt-1088 = zhuyuhan `qwen3-4b-agent-history-c2kv-toolcall-npu-v2`
> （已 cp 至 `~/checkpoints_upstream/`，dynamic-interleave/qkv/embed-mean）。

## τ²-bench airline（50 任务）

| 臂 | 4186 reward | 4186 协议率 | 上游1088 reward | 上游1088 协议率 |
|---|---:|---:|---:|---:|
| full | 0.26 [0.14,0.38] | **0.98** | **0.28** | 待收 |
| c2kv (8×) | 0.12 [0.04,0.22] | **0.776** | 重跑中(up2) | 待收 |
| cd_full | 0.26 [0.14,0.38] | 0.98 | 重跑中 | 待收 |
| cd_c2kv | 0.04 [0.0,0.1] | **1.00** | 重跑中 | 待收 |

关键事实：
1. **压缩的协议层代价**（被旧公式 bug 掩盖的核心发现）：c2kv 协议率 0.776 vs full 0.98
   （−20pp）——8× 压缩损害的很大一块在协议合法性，不只是语义。
2. **cd_c2kv 协议率 1.00**（XGrammar-2 硬保证兑现）但 reward 0.12→0.04：**约束解码在
   压缩臂上把协议拉满的同时压掉语义**（constraint tax −8pp，超迁移手册 2pp kill 线，
   → 建议软版本：仅检测到非法时二次约束）。
3. full≈cd_full（0.26≡0.26）：无压缩时约束解码零代价。

## BFCL v4 multi_turn_base（200 例）

| 臂 | 4186 acc | 上游1088 acc |
|---|---:|---:|
| full | 2.0% | 重跑中（曾完成，精度因日志抑制未落盘） |
| c2kv | 2.5% | 重跑中（同上） |
| cd_full | 2.0% | 重跑中（同上） |
| cd_c2kv | 未记录（重生成中） | 重跑中 |

两 checkpoint 全臂处于 1.5–2.5% 地板：**瓶颈是模型对 BFCL 文件操作多轮格式的把握
（叙述代替调用、`<|im_end|>` 泄漏），不是压缩**——臂间差异在此地板上无分辨力。
（已加分数持久化 `~/bench_results/bfcl_scores.log` 防再丢。）

## ToolSandbox（test 子集，3 场景；官方 similarity，minefield 均 0）

| 臂 | 4186 | 上游1088 |
|---|---:|---:|
| full | 0.125/0.70/0.134（均值 0.32） | **0.32**（0 aborts） |
| c2kv | 0.045 均值，**2/3 场景因选错工具被严格模式中止** | **0.0，3/3 中止** |

TS 上 c2kv 在两个 checkpoint 上都触发"在仅允许 end_conversation 时调用 send_message"
类硬失败——**压缩下的工具选择错误是 checkpoint 无关的系统性现象**。

## 成本口径

- proxy 请求日志（wall_sec/gist_tokens/original_tokens）按臂存于 `~/bench_logs/*.jsonl`；
  有效压缩比 τ² 会话 ≈7.5×；TTFT 未测（非流式）。
- 等效压缩率按 ResKV b=m+r 口径。

## 边界

- A1/A2/K1/C3/B1 等 D 线机制臂**不适用上游对照**：r2 触发集定义在 fixed_joint 上
  （C→W 转移是 checkpoint 相对的），机制结论按设计是 checkpoint 内部对照。
- 训练收官：med_dmulti_joint **checkpoint-6322** 已落盘（新一代最新）；本轮矩阵未含，
  是否补跑由后续决定（队列一行即可入队）。
