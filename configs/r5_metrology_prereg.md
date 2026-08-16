# R5 S8 度量学复现 sprint 预注册（configs/r5_metrology_prereg.md）

本文件先于一切 S8 实验运行冻结。判据 M0–M3 逐字引自任务书 §11；类别选择、抽样清单、seed、抽取器规则、判据以此为准，运行后不得修订（偏差只能以 erratum 形式记录）。

研究主张：KV/上下文压缩条件下的 agent 工具调用评测会由评测装置制造伪影。内部锚定素材表述口径：本团队测试基建迭代中的自我修正案例（数字以 PR-A 修正表为准引用，形如 `results/r5/... @ <hash>`）。

## 1. 底座与模型

- 底座：BFCL v4（github.com/ShishirPatil/gorilla，`berkeley-function-call-leaderboard/bfcl_eval`，快照 sha 见 manifest）。
- 模型：Qwen3-4B-Instruct-2507（本地权重，sha256 见 manifest），HF eager attention 路径，greedy（do_sample=False / temperature=0）。
- 执行环境：NPU 服务器；推理逐行落盘原始文本与逐 token id。
- tau2-bench 为可选第二底座；集成超 1.5 天记 DEGRADED-SKIPPED（本预注册即声明：若启动则另行 erratum 记录，不启动为默认）。

## 2. 类别选择规则与审计

规则（任务书 §10.1）：先做金标输出 token 长度静态审计（Qwen3 tokenizer；审计范围限实际运行的基准），取全部 multi-turn 子类，另加金标中位长度最高的 1–2 个类别凑足样本。

审计结果（`metrology/data/bfcl_goldlen_audit.json`，Qwen3-4B tokenizer，json.dumps 金标串）：

| 类别 | n_items | n_turns | 每轮中位 | 每轮 P95 | 每条总中位 |
|---|---|---|---|---|---|
| multi_turn_base | 200 | 734 | 24 | 72 | （见 json） |
| multi_turn_long_context | 200 | 734 | 24 | 78 | |
| multi_turn_miss_func | 200 | 934 | 16 | 64 | |
| multi_turn_miss_param | 200 | 934 | 16 | 64 | |
| parallel_multiple | 200 | 200 | 104 | 180 | |
| parallel | 200 | 200 | 89 | 192 | |

入选类别（6 个）：multi_turn_base、multi_turn_long_context、multi_turn_miss_func、multi_turn_miss_param（全部 multi-turn 子类）+ parallel_multiple、parallel（金标每轮中位最长的两个非 multi-turn 类，104/89 token）。

静态审计结论（(i) 的另一半）：BFCL 原生 OSS 路径默认 cap = min(4096, 剩余上下文)（base_oss_handler.py:328-336，全库唯一 cap 点、无 per-category 分支）；所选 6 类金标每轮 P95 ≤ 192 token，默认 cap 对金标长度充足；cap=128 档对金标 P95（multi_turn 64–78 / parallel* 180–192）部分不足——正是要测量的预算 censoring 区间。

## 3. 冻结样本

- 抽样：每入选类别按 id 字典序排序后以 seed=20260816 均匀随机抽 60 条，6 类共 360 条；清单文件 `configs/r5_metrology_sample.json`（含 id、类别、轮数、金标每轮 token 长度）随本预注册同 commit 提交。
- 同一冻结清单（样本 id + seed）在所有条件 × 所有 cap 下复用；某条件缺行记 MISSING，不得换样。
- 样本量核对：每条件 × 每 cap = 360 ≥ 300（满足任务书下限）。

## 4. 条件（压缩方法）与 cap 档

- 条件（3 个）：base（无压缩对照）、SnapKV、H2O 或 StreamingLLM 或 KVzip 择一（按 eager 等价重实现的集成难度择一；所选方法与官方小样例对照记录写入 PR-B）。
- 压缩实现：training-free，按官方仓库逻辑在 HF eager attention 上等价重实现（不依赖 flash-attn/CUDA kernel）；先复现官方仓库一个小样例并把对照记录写入 PR-B，再进入正式跑批。
- cap 三档（每轮 max_new_tokens）：基准默认（min(4096, 剩余上下文)，逐行记录实际值）、128、1024。
- C2KV 条件：用 S6 内部数据充当，参与测量 (ii)(iii)、不参与 (i)（其数据固定 cap=256，表内如实标注）；公共底座上的 C2KV 集成仅在自 S8 开工起 3 个自然日内能在 ≥10 条冻结样本上跑通时做，否则标 INTERNAL-ONLY；若 S6 未交付，C2KV 列记 MISSING 并说明原因。

## 5. 三类测量口径

### (i) censoring 重分类率（M1 裁定口径）
同一冻结样本、同一压缩条件、greedy 解码下，cap=128 时判失败或格式错的样本中，cap=1024 重跑后判成功的比例（「失败→成功」为主口径；任何标签改变比例同表另列）。cap 指每轮 max_new_tokens；multi-turn 中 cap 改变导致的轨迹分叉属测量对象本身，如实记录首个分叉轮，不作剔除。失败/成功的判定面 = 该类别基准原生评分（multi_turn 用 state+response 检查；parallel* 用 AST 检查）。

### (ii) 外壳-语义分裂率（M2 裁定口径）
分裂率 = 「语义面判对 且 协议面判错」行数 ÷ 该条件 × cap 全部冻结样本数（以协议判错行为分母的占比另列，仅展示；M2 以全样本分母裁定）。
- 协议面 = 基准原生严格格式评分：multi_turn 为逐步 decode_execute 可解析（eval_runner.py:220-236 语义）；单轮为 decode_ast 无异常且输出为函数调用格式（eval_runner.py:334-367 语义）。
- 语义面 = BFCL 的 AST/可执行检查（有则用）+ 规则式散文抽取器（规则冻结如下）。
- 散文抽取器规则（冻结）：① 函数名词典 = 该样本 initial_config/available functions 的函数名集合；生成文本中首个出现在词典内的函数名即语义函数名（按文本出现位置取最先）；② 参数键值对：在函数名后 2000 字符窗口内以正则 `"?(\w+)"?\s*[:=]\s*("([^"]*)"|\d+(\.\d+)?|true|false|null)` 抽取键值对，键集合 ∩ 金标参数键集合非空即记参数命中；③ 语义判对 = 函数名命中 AND（金标无参数 OR 参数命中）。此规则文本冻结后使用，运行中不得修订。
- 人工复核：30 例按分层随机（seed=20260816）从「语义对但协议错」判定行抽取；复核不一致 >3/30 须修抽取器并重抽复核（修订以 erratum 记录）。

### (iii) 构成敏感性（描述性，不入 M3）
直接标准化重加权：以（轮深分箱 × BFCL 类别）为层（轮深分箱：1–2 / 3–4 / 5+ 轮），权重取 (a) 各层等权与 (b) 内部 395 集构成（clipped×池×finish 边际分布，取自 results/r5/analysis/v1_stratified_strict.json @ PR-A commit）两组固定方案，报每方案下聚合压缩税点估与方法排名；内部锚定案例并列引用。

## 6. 判据（逐字，引任务书 §11）

- M0（S4 门）：2607.02577 按 §6 口径判「已实现」(a) AND (b) → S8=KILLED-BY-PRECEDENT。（已执行：M0 不触发，见 docs/metrology_killcheck.md。）
- M1：censoring 重分类率（「失败→成功」主口径）在 ≥1 个公共基准 × ≥2 个压缩方法上 ≥10% → 伪影泛化成立；<10% → NOT-GENERALIZED（内部案例仍成立但不外推）。
- M2：外壳-语义分裂率（全样本分母）在 ≥1 个压缩条件（不含 base；base 的分裂率作对照另列）下 ≥5% → 成立；否则 NOT-SUPPORTED。
- M3：基线 = 各基准默认 cap + 原生评分下的压缩方法排名与聚合压缩税符号；修正 = 金标长度键控 cap（取 max(1024, 金标长度 P95)）+ 双列评分的语义列；同一冻结样本集上任一压缩方法排名对换或税符号翻转 → 「修正有后果」；(iii) 重加权为描述性，不入 M3。三者全不成立 → S8 判 NEGATIVE，报告如实交付。

## 7. 必须对位的邻居（PR-B 相关工作节引用）

2607.02577（最近邻 form factor）；2605.07395（预算截断同类先例，routing 场景）；2608.01056（CompressAgent；若其环境已放出，讨论节备注可用性，不因此新增底座）；2608.01631（KV 压缩下推理忠实性，显式对位：其主张"指标不完整"，我方主张"装置制造伪影"）；辅助引用 2605.23950、2605.27922、2605.24660、2605.18857、2510.00231、2412.17483。引文流监控：开工与收尾各一次（检索引用 2608.01056 与 2608.01631 的新文及同关键词新文），命中即报。开工监控已完成（2026-08-16）：无新被引；同关键词命中 AGC-Bench 2607.01152v2 Appendix G（生成预算触顶检测+提高 cap 重跑 16 个数据集）作为方法学邻居列入相关工作。

## 8. 纪律

零训练；不产生新 checkpoint；不发布、不开源、不对外发帖；内部素材一律表述为"本团队测试基建迭代中的自我修正案例"；本 sprint 任何行文不得写成对本组既有结果或 C2KV 原论文评测的批评；凭据不出现在任何 commit/PR/日志/代码注释；所有数字带 N 与出处。
