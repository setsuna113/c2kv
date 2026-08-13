# 文献笔记：Agent 失败实时检测与修复（arXiv 2608.02464）

**Real-Time Detection and Repair of LLM Agent Failures** — Sunny Dubey（独作），2026-08-03 v1，cs.AI/cs.LG/cs.SE。代码：github.com/sunnydubey1111/agent-trajectory-sentinel。

## 方法一句话
Agent episode 的运行时看门狗：每步只用可观测遥测（step 输出的 char-3-gram 哈希嵌入、token 不确定度聚合、动作元数据）打分，模型是在健康轨迹上训的 one-class 回声状态网络（ESN）集成 + 分通道 CUSUM 告警（拟合 1.7s、每步约 200µs）。2823 条真实 episode 上 5% 误报预算内检出率 0.71；再加确定性校验（按实际收到的工具结果重算题目给定的总数、确认必需调用都发生了、工具结果对 schema 校验）以 0 误报抓 60–96% 失败。检测闭环到修复：被标记的 run 回滚到检查点重跑，比 16% 的重采样对照多救回 45%。

## Q1：信号清单哪些已被占？
论文自己把占位情况说得很白（§2 "That position is occupied"）：AgentForesight（逐前缀 LLM 审计）、PrefixGuard（廉价在线前缀打分、结果监督）、Trajectory Guard（免标签嵌入异常检测）、运行时 guard、幻觉检测（自一致性、语义熵、隐状态探针）、修复侧 AgentTether。作者明确"不把单点的中途检测/可观测遥测/逐步廉价/免标签训练单独算作新意，新意在组合"。其信号清单：哈希嵌入遥测、token 不确定度、内容接地词法标记、确定性校验（数值接地/覆盖/schema 形状）、LLM judge 兜底。值得注意：消融里 token-surprisal 通道对 ESN 的贡献约 +0.000 AUROC——"部署要求的是结构化工具结果，不是 logprob 访问权"。**空位**：压缩语境的触发信号（用压缩/全量 KV 的 logp 差当失信息告警）不在其清单内。

## Q2：rollback 粒度？
**episode 内逐步检查点**："each flagged run is rolled back to its last fact-gathering step and re-run live"；与 AgentTether 的区别是他们的告警在 episode 中途、回滚到答案交付之前的检查点。不是 per-token/per-layer。

## Q3：有无压缩语境？
**没有。** 全文是在线失败检测+回滚修复，不涉及 KV/上下文压缩。对 C2KV 的相关性是结构性的：健康零假设的廉价 one-class 信号 + 确定性校验门控回滚决策 + 诚实的误报/重校准分析，这套方法学可以直接照搬到"压缩是否丢信息"的触发器设计上。

## 置信度
中等：摘要 + 全文 §§1–5.1 直接读过（HTML 在 §5.1 后截断）；§6–10 仅从摘要与贡献列表推断。

## 对 C2KV 的相关性
- S4 的 ΔlogP（压缩 vs 全量 KV 对强制前缀的 teacher-forced logp 差）本质上就是该文框架里"逐前缀打分"信号在压缩语境的实例化，且目前无人占用。
- 其"5% 误报预算下的检出率"报告方式适合直接借用作 ΔlogP 触发器的评测口径。
