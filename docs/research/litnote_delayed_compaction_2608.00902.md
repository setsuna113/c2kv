# 文献笔记：延迟压缩实证研究（arXiv 2608.00902）

**Practical Online KV Cache Compaction for LLM Agents: An Empirical Study** — Liu, Ji, An, Jain, Polatkan, Zhu, Chang（UC Santa Barbara / LinkedIn）。

## 方法一句话
系统研究在线 agent 回路中的 KV 压实（compaction）：把两类 query 驱动的压实器 TE（Token Eviction，SnapKV 式选择）与 AM（Attention Matching，额外拟合 attention bias 与重建 value）适配到逐轮压实，并比较 proxy-query 来源——当前轮 proxy（边界 token query、repeat-prefill 重建 query）vs **未来轮 proxy**：把第 t 轮的压实**延迟 k 轮**，用 agent 自己后续生成产生的 query 向量当 proxy。结论：立即压实常常掉点，延迟 1 轮"一致地找回大部分差距"；ratio 0.2 时延迟 TE 保住大部分精度、KV 降约 80%、吞吐最高 4.2×。

## Q1："延迟压缩"用什么 proxy 机制？
**不用学习式代理/打分器**：proxy 就是靠延迟收割的真实未来 query。§3.2："To compact T_t with a delay of k turns, the system keeps T_t in raw form while generating turns T_{t+1},…,T_{t+k}… we record the query vectors produced during the assistant generations… and use them as proxies when compacting T_t after T_{t+k} finishes"。消融还加入工具响应 prefill 的 query 作为额外来源；延迟窗口 k∈{1,3,5} 扫过（§4.3）。

## Q2：改表示还是只改时机？
**只改时机。** 表示本身完全不动：TE "is thus a pure selection method: after selecting S, the stored keys and values are unchanged"（§3.1）；AM 的 bias/value 拟合与 proxy 来源无关。延迟改变的是"何时压"与"用谁的 query 指导压"。

## Q3：可否拼接复用到学习式 gist 压缩器？
**部分可。** k 轮延迟窗口本身是压缩器无关的系统级调度选择——gist 压缩器同样可以让一轮保持 raw、k 轮后再压。但其**收益机制**（未来 query 作为选择/拟合目标）预设了 query 条件化的压实器；query 无关的 gist 编码器只延迟不改造将一无所获，必须改成"第二遍编码时能attend 未来窗口"的条件化形式。作者自己在 Limitations 里把 "learned compressors" 判为与其低开销在线场景 "complementary"。

## 置信度
arXiv HTML 全文主体（§1–6、Limitations、附录 A 部分）已读。

## 对 C2KV 的相关性
- 这是"延迟压缩"设计的直接前身，也是 TE-AM baseline 臂的出处。
- 其四臂结构（immediate / delayed-unconditioned / delayed-query-conditioned / TE-AM）里，delayed-unconditioned 臂是该文逻辑对 gist 压缩器的零预测对照：若我们也观测到无收益，说明收益确实来自未来信号通道而非时机本身。
- 差异点：他们从未在**多轮历史的学习式 gist 压缩**上测过延迟；也未测工具调用行为指标（call rate / tool-name accuracy）——我们的场景两者都占。
