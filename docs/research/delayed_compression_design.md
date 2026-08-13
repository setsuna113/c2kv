# 延迟压缩（Delayed Compression）设计文档

状态：设计稿，只写不跑。前置实验：S4 强制前缀诊断（task/s4-forced-prefix）。
相关文献：docs/research/litnote_delayed_compaction_2608.00902.md（延迟压实实证）、litnote_vericache_2605.17613.md、litnote_agent_sentinel_2608.02464.md。

## 0. 动机
C2KV（学习式 gist 压缩）把多轮历史压成 gist token 后，工具调用率崩约 7 倍（44%→6%），但仍发起调用的样本工具名准确率反升（85.2% vs 65.1%）——信息疑似还在压缩 KV 里，丢的是"发起调用"的行为触发。2608.00902 在 selection 类压实器上证明：把第 t 轮的压缩推迟 1 轮、用未来轮的真实 query 做 proxy，能一致找回大部分掉点。开放问题：**延迟与未来信号能否嫁接到学习式 gist 压缩器上**。该文自己预测 query 无关的压缩器从延迟中无收益（Limitations: learned compressors "complementary"）——这给出明确的零假设对照。

## 1. 四臂设计
| 臂 | 压缩时机 | 压缩器条件 | 预期 |
|---|---|---|---|
| A immediate | 第 t 轮完成即压 | 无（现状 C2KV） | 现状基线（调用率崩溃复现） |
| B delayed-unconditioned | 第 t 轮 raw 保留，t+1 轮完成后压 | 无 | 零假设：2608.00902 预测无收益 |
| C delayed-query-conditioned | 同 B 的时机 | 压缩时拼接 t+1 轮作条件窗口 | 主臂：未来信号通道 |
| D TE-AM baseline | 同 B 的时机 | SnapKV 式 TE + AM（非学习式） | 文献复现参照系 |

D 臂是 2608.00902 方法在我们数据/指标上的直接移植，用于校准"延迟收益在本设定下的量级天花板"；C 与 D 的差距回答"学习式 gist 是否值得保留"。

## 2. Matched KV budget 控制
四臂在同一决策点（第 t+2 轮生成前）必须占用相同的 KV token 预算 B：
- A：历史全部 4× 压缩，gist 数 = B。
- B/C：第 t 轮 raw（R_t tokens）+ 更早历史压缩。为保证预算相等：更早历史的 gist 配额 = B − R_t，压缩率相应调紧（实现上按 token 数动态选 ratio ∈ {4, 8, 16} 档，取不超过配额的最松档）。
- D：同一预算下选 retained token 数 = B。
预算不等的一切比较都不作数；每个样本落盘实际 KV token 数核对。

## 3. 条件化压缩器的最小实现（输入拼接，不改架构）
C2KV 的 gist 机制 = gist_token + gist_q/k/v 投影（gist_param="qkv"）。条件化不加任何参数：
- 压缩第 t 轮时，forward 输入 = [第 t 轮文本 token] + [未来窗口 token] + [gist token]。
- 未来窗口 = 同轨迹 t+1 轮的 user+assistant 文本，截断到 F=512 token，放在被压文本之后、gist token 之前；gist token 经注意力同时读历史与窗口。
- 窗口 token 只读：训练 loss 不落在窗口与 gist 之外的 token 上（沿用现有训练管线的 label mask 即可）；推理时窗口 K/V 压缩完即弃，不进入服务 cache。
- 与 2608.00902 的差异：他们用未来 query 向量做**选择/拟合目标**；我们把未来窗口作为**编码条件**，表示仍是学习式 gist。若 C>B，说明 gist 编码器确实能利用未来信号重写表示（他们框架做不到这一点）。

## 4. 训练数据构造
- 复用 agent-llm-traces 多轮轨迹与现有 c2kv 训练管线（train_agent_history_c2kv），同一 toolset_disjoint 划分。
- 对每个压缩目标轮 t：取同轨迹第 t+1 轮（user query + assistant response）作条件窗口；t 为末轮的样本归入 unconditioned 池（训 B 臂同一压缩器时窗口置空）。
- 一个压缩器权重同时服务 B/C 两臂（窗口有/无 = 条件 dropout 训练），排除"不同 checkpoint"混淆；条件窗口 dropout 率 0.5。

## 5. 评测 readout
- 主指标：tool_name accuracy（全样本口径）、call rate；逐样本配对。
- 行为触发诊断：直接复用 S4 的 forced-prefix + ΔlogP（logp_prefix_c2kv vs full）——Δ 缩小即触发抑制缓解。
- 触发信号 readout（若 S4 升级 1 成立）：ΔlogP 对漏调用样本的 AUROC，按 2608.02464 的"固定误报预算报检出率"口径报告。

## 6. Kill 条件
**conditioned（C）相对 unconditioned（B）无稳定收益即杀**：配对 tool_name accuracy 差的 95%CI 覆盖 0 且点估计小于实测 MDE → 未来信号通道关闭，学习式 gist 延迟压缩方向终止，退回 D 臂（selection 类延迟）或 VeriCache 式 serving 层兜底。B 相对 A 若意外有收益（零假设被证伪），先复查 budget 匹配实现再解释。
