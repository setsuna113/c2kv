# R4 预注册（prereg）

本文件先于一切 GPU 运行提交。PR 引用本文件的 commit hash。V1–V4 判据逐字引自 R4 任务书 v2 §9。

## 冻结参数（见数前写死）

- **V4 阈值 T**：PR#1（`gh pr view 1 --repo setsuna113/c2kv`）§2 结果表，全样本口径 tool_name acc：full 无强制臂（A）= 0.1571（66/420），c2kv 无强制臂（B）= 0.0247（11/445）。
  **T = 0.25 × (0.1571 − 0.0247) = 0.0331**。
  未选口径记录：工具目标子集口径（0.2845 / 0.0445 → T_alt = 0.0600）不采用——主指标口径与 V1 一致，取全样本。
- **任务 D 冻结集**：`configs/r4_d_qids.json`——PR#1 merged_{A,B,C,D}.jsonl 有效行 qid 并集，n=594（178 session），逐档案 sha256 绑定。子集规则（预估 >35 GPU·h 则取排序前 300）**未触发**：以 B 臂 latency 预估三臂合计 5.49 GPU·h（×1.25 裕量），故冻结全量并集。
- **任务 D regime 勘误（report-only）**：任务书称"32k 中池 regime"，但 PR#1 档案实测为 history 压缩 regime——doc_tokens≡kept_history_tokens，中位 2313、最大 9271（选择器上限 12288）。本包以档案实测为准，"32k"标签差异如实记录，不作并表结论。
- **任务 D 生成配置**：与 PR#1 一致——checkpoint-2678（qwen3-4b-agent-history-c2kv-npu）、ratio=4、greedy、max_new_tokens=128、enable_thinking=False、subset_disjoint eval split（eval_ratio=0.1, seed 42, max_samples_per_session=4, include_tools=True）、全 eager。
- **typed span 定义**（token 级，作用于 history 各 doc 的 chat 模板序列化 ids）：
  1. role/分隔标记：`<|im_start|>`、`<|im_end|>` 全部出现（各 1 token span）；
  2. 每个 `<tool_call>…</tool_call>` 区域内：工具名字段值（"name"/"tool_name"/"function_name" 的值）、全部参数键名、全部 JSON 结构符（`{` `}` `[` `]` `:` `,` `"`）；
  3. tool 角色返回消息内的 JSON 结构符同上。
  实现：先字符级结构解析，再用 fast tokenizer 的 offset mapping 映射回 token 下标并断言与 harness 生成的 doc ids 逐位一致。逐 qid span 清单落盘 `configs/r4_anchor_spans.json`（含 hash），跑前 commit。
- **random 臂**：逐 qid 与 typed 臂 span 数相同、长度多重集相同，位置从**非控制 token** 中均匀抽取（不重叠控制位，种子 seed=20260815 固定）。选非控制位为使 typed−random 对比只隔离"位置信息类型"单一变量；该选择在此写死。
- **预算对齐硬门**：两臂保留 raw token 总数偏差 ≤2%（按构造逐 qid 相等，全局必达）；逐臂物理 KV 总量（system raw + gist + anchor raw）入表，超差即判无效重跑。
- **anchor 注入语义（双覆盖）**：anchor span 的 raw KV 是**追加**——gist 摘要照常保留（不动 ratio-block 掩码算术），raw KV 按其原始绝对位置烤入 RoPE 后拼接进 cache。已知限定：retained raw KV 的上下文为 chunk 局部（压缩前向只见本 chunk），非 76k 全局——该限定写入 PR。
- **任务 A**：checkpoint-250 全池（不压缩）、HF eager + 分块 prefill（chunk=512，勘误见 `r4_erratum.md`）+ 单 token 交接解码，48 qid 冻结集（`configs/r3_s1_48_qids.json`）。OOM 预授权：chunk 减半并记录；连续两次仍 OOM 停下报告。
- **任务 E**：48 qid、checkpoint-250、任务 A 同路径；工具池替换为其他 session 的池（长度对齐原 doc_tokens ±5%、目标工具不在池内、替换池种子固定 seed=20260815）。report-only。
- **任务 F**：枚举宇宙 = r3 抽取配置（S1_DATA_KW：toolset_disjoint eval split、96-doc、max_doc_length=1024、require_tool_call、min_target_tokens=128），池长 = doc_tokens ≥ 70000，冻结 48 除外。枚举结果（或 NOT-AVAILABLE）见 `configs/r4_qids_ext.json`，其 commit 先于一切 GPU 运行。若执行：扩充集双臂 = c2kv@4（r3 T-E 配置：512×160）+ full（任务 A 路径）。
- **统计口径**：配对 = 按 qid；McNemar exact（b/c 格数必报）；session 聚类 bootstrap 20000 reps（seed 0）报 95%CI；session 数限制（76k 层 5 个 + 扩充、32k 层 178 个）必明写。不同 checkpoint 的数字不合并、不并排作结论。

## 判据原文（逐字）

**V1（同权重配对，主指标工具名正确）**：主分析集 = 48 + F 扩充（若 F 执行；原 48 子集单列为次要）。full>c2kv 且 McNemar p<0.05 → "质量代价存在"；c2kv 点估 ≥ full 且 p≥0.05 → "试点级支持（N/session 数限定）"，措辞禁"证明/显著优于"；其余 INCONCLUSIVE。调用率同表次要。

**V2（eager 路径连贯性，48 例）**：连贯 = 非空 且 重复 4-gram 占比<50%，另随机 8 例人工复核记录。≥46/48 → CONFIRMED-FULL-COVERAGE；40–45 → CONFIRMED-WITH-EXCEPTIONS（列 qid）；<40 → 归因范围修订。

**V3（失败构成）**：32k 大样本层：控制类（PROTOCOL_BROKEN+WRONG_TOOL+WRONG_ARGS）占 plain 臂全部失败 ≥60% → 支持"控制平面主导"，<40% → 不支持，40–60% → 中间；76k 配对层：超额失败格（c2kv 败而 full 成）控制类占 ≥50% → 支持。两层同向才判 SUPPORTED/NOT-SUPPORTED，否则 INCONCLUSIVE；某层 DEGRADED 须在裁定行标注。

**V4（锚点判别，主指标工具名正确，typed vs random 配对）**：阈值 T = 0.25 × (PR#1 全量无强制臂 acc − c2kv 无强制臂 acc)——由你从 PR#1 body 取数，在 prereg 中写死数值。**本包取值：T = 0.25 × (0.1571 − 0.0247) = 0.0331**。typed−random ≥ T 且 McNemar p<0.05 → SUPPORTED；typed−random 点估 ≤0 → NOT-SUPPORTED；其余 INCONCLUSIVE。前置有效性门：预算对齐 ≤2% 偏差，违者该臂重跑。typed vs plain、plain vs PR#1 记录值均 report-only。

**V5 无**：任务 E/F/C 及边界维度均 report-only，不设裁定行。
