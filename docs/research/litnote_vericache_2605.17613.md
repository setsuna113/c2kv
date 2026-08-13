# 文献笔记：VeriCache（arXiv 2605.17613）

**VeriCache: Turning Lossy KV Cache into Lossless LLM Inference** — Yao, Shen, Du, Feng, Seo, Zhang, Huang, Huang, Lu, Jiang（U. Chicago / Tensormesh / Samsung / MSR），2026-05-17 v1，预印本。

## 方法一句话
VeriCache 不是新压缩器，而是 serving 框架：把有损压缩 KV 当作**自投机草稿器**（self-speculative drafter），每轮草稿对完整 KV（ offload 在 CPU DRAM/远端）并行验证、纠错，使输出与全量 KV greedy 解码逐 token 一致；系统贡献在跨资源流水（HBM 起草与 PCIe 取回重叠）、拉长验证周期（25–40 token/轮）与运行时调度。

## Q1：压缩算法族是否含学习式 gist？
**不含。** VeriCache 自身不压缩，通过统一接口接入七种现有方法（KVzip、KVzap、ExpectedAttention、SnapKV、KIVI、KVQuant、RotateKV）——全部是**选择/丢弃/量化类**，没有学习式 summary token。原文 §1/§6："any token-dropping or quantization method that conforms can serve as the drafter"。

## Q2：verify 信号是否被回用？
**在环内 recurrent 使用，但不回喂压缩器。** 验证每个草稿轮都触发（§4.3，约每 x+1 步一次），纠错结果直接成为输出并从最后接受位置续写；接受率还喂给调度器调节验证频率与 batch 组成（§1/§5）。但信号**不**用于调整压缩策略，也不缓存为模型输入——全量 KV 每次验证都重新取回。对我们的含义：verify 信号作为"压缩是否丢信息"的在线 readout 是现成范式，但"回喂压缩器"这条路在他们框架里没人占。

## Q3：是否测 agent 工具调用？
**测了，且结果直接支持我们的问题意识。** §3.1 报告 ComplexFuncBench 工具调用任务上 function call accuracy 在 KVzip 4× 压缩下掉到 10% 以下——与我们观测到的"压缩后调用率崩"互为印证。但注意其评测是单轮 function calling / 长上下文 agent trace（LMCache agentic trace、PISanitizer），**不是多轮对话历史压缩**场景。

## 置信度
ar5iv 全文读至 §8.2（相关工作和附录未读），机制性结论可靠。

## 对 C2KV 的相关性
- 学习式 gist 压缩 + verify 回用的组合在其接口层面是空位（他们的 drafter 接口假设压缩无参/启发式）。
- "压缩 KV 当草稿、全量 KV 兜底"是与"延迟压缩"正交的兜底路线；若 S4 证实行为触发可修复，VeriCache 式验证可作为 serving 层安全网。
