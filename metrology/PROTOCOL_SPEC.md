# 修正协议规范书（S8 评测方法论产出）

本规范书给出在 KV/上下文压缩条件下评测 agent 工具调用时的四条修正协议要件。
每条附：规则原文、本仓库可运行实现出处、本 sprint（R5 S8）内的证据锚点。
预注册判据全文见 `configs/r5_metrology_prereg.md`（commit e12acf5）。

## 要件 1：预算充足性检查（金标长度键控）

**规则**：任何生成预算（max_new_tokens cap）在用于评测前，必须先通过金标输出长度
审计：取实际运行类别的金标每轮 token 长度分布（本仓库用 Qwen3 tokenizer，
nearest-rank P95），修正 cap 取 `cap_c = max(1024, 金标每轮长度 P95)`；低于金标
长度分位的 cap 档只可作 censoring 应力测试，不得作为结论性评分档位。每行产出
必须带 `gold_ge_cap` 旗标（金标长度 ≥ cap）与 `censored` 旗标（生成触顶）。

**实现**：`metrology/analyze_s8.py`（`cap_c_by_category` 计算）；本 sprint 六类别
金标每轮 P95 ∈ [60, 187]，故 cap_c 全类 = 1024（`metrology/data/s8_m1m2m3.json`
M3.cap_c_by_category @ cfa7e8b）。静态审计表冻结于
`configs/r5_metrology_prereg.md` §2。

**证据**：内部锚定案例①——本团队测试基建迭代中的自我修正案例：r4 层
max_new_tokens=128 下金标中位 130 token，211/395 行金标本身在 cap 内写不完，
约 94%（=(206−12)/206）的协议破坏标签系 cap 伪影
（`results/r5/analysis/` S3 修正表 @ PR-A `task/r5-closeout`）。

## 要件 2：双列评分（协议发射与语义正确分离）

**规则**：协议面（strict format 可解析性）与语义面（调用内容正确性）必须作为两列
分别评分、分别报告，**永不合并**为单一分数。协议面 = 基准原生严格格式评分；
语义面 = 基准 AST/可执行检查（有则用）+ 规则式散文抽取器。分裂行
（`split_row` = 语义对 ∧ 协议错）必须单列计数，分母为该条件×cap 全部冻结样本数。

**实现**：`metrology/bfcl_score.py`（`protocol_valid` / `semantic_correct` /
`split_row` 列）；散文抽取器 `metrology/prose_extract.py`（v2 规则，见
`docs` 级 erratum 记录与 `metrology/REPORT_S8.md` erratum 节；v1 冻结保留为
`prose_v1_frozen` 参照列）。抽取器规则须经 30 例分层人工复核门
（不一致 >3/30 须修抽取器并重抽复核）。

**证据**：本 sprint M2 测量（`metrology/data/s8_m1m2m3.json` M2 节 @ cfa7e8b）；
内部锚定案例②见 REPORT_S8.md §5。

## 要件 3：分层披露

**规则**：聚合数字必须伴随构成分层表同步披露；任何池/构成维度的层数 ≤2 时只分层
展示、明写不可做该维度的聚类推断。截断/触顶行（censored）在所有表中带占比列。
缺失行记 MISSING 并在同一冻结分母内计数，不得换样补齐。

**实现**：`metrology/analyze_s8.py`（类别×轮深分箱分层、censored 列、MISSING 纪律）；
内部 395 集 clipped×池×finish 三维分层表
（`results/r5/analysis/v1_stratified_strict.json` @ e1c1790，PR-A）。

**证据**：内部锚定案例③——构成混杂制造聚合符号翻转（聚合 p=0.0009 偏压缩臂 ↔
未截断层 p=9e-09 偏 full；`results/r4/analysis/paired_76k_main395.json` @ a2683be
与上述分层表）。

## 要件 4：对照去污染

**规则**：同一冻结样本清单（样本 id + seed）必须在所有条件 × 所有 cap 档下复用；
压缩方法的实现须先复现官方仓库一个小样例并把对照记录写入 PR，方可在
eager attention 上等价重实现；全部输入数据（运行 jsonl、评分输入、分析输入）以
sha256 清单入库，分析输出内嵌全部输入的 sha256。

**实现**：冻结清单 `configs/r5_metrology_sample.json`（seed=20260816，360 样本，
@ e12acf5）；压缩实现 `metrology/kv_compress.py`（SnapKV/StreamingLLM eager
重实现，官方逻辑小样例对照测试 `metrology/test_kv_compress.py` 19 项 @ cafac74）；
清单 `metrology/data/manifest_s8_runs.json` 与
`metrology/data/s8_m1m2m3.json` 内嵌 input_sha256（@ cfa7e8b）。

**证据**：内部锚定案例④——"先验地板"实为历史复制地板（剔除 4/48 长度违规行后
地板 acc 0.1818/call 0.5909，8 个"猜对"全为 3–9 轮重试延续中复制可见历史的同名
调用；PR-A S3/E 修正表 @ `task/r5-closeout`）。
