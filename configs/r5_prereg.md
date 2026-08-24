# R5 预注册（prereg）

本文件先于一切 GPU 运行与一切 L1 重分析计算提交。PR-A 引用本文件的 commit hash。判据 W1/W2/W3、M0/M1/M2/M3 逐字引自 R5 任务书 §11；口径定义引自 §5。

## 一、口径定义（先于计算写死）

### 1.1 严格主口径（strict protocol-valid）

一行计「协议有效」当且仅当：
- 调用判定谓词命中（全局统一，不得改）：`("<tool_call>" in text) or ("Action:" in text)`；
- AND 从 `<tool_call>...</tool_call>` **闭合块**中解析出合法 JSON 且含 `"name"` 键（`arguments` 允许为空 dict `{}`）。

仅命中 `Action:` 而无闭合可解析 `<tool_call>` JSON 的行计**协议无效**（本数据集金标为 `<tool_call>` 格式）；此类行数 >0 时单列披露。

工具名判分复用 `agent/r3_aggregate_te.py` 与 harness `_extract_tool_name`（`agent/eval_agent_tool_definition_c2kv.py:105`）。

### 1.2 finish 语义评分（单列，不入主口径）

金标为 finish 调用的行：取金标 finish 调用的 `answer` 参数串 vs 生成文本，报 token-F1 与 ROUGE-L F1；判对线 = ROUGE-L F1 ≥ 0.5。附 10 例人工复核记录。

### 1.3 censored 标注

`generated_tokens ≥ 128` 的行标 `censored@128`（R5 新跑批次为 ≥256 标 `censored@256`），所有表带占比列。

### 1.4 分层维度

- **clipped 判定**：c2kv 臂行 `prompt_tokens == 1920`（交叉验证：ext 池 full 行 n_tokens 82102 = 11 系统 + 80171 池 + 1920 历史）。
- **池归属**：按行内 `doc_tokens` 判（75327 池 / 80171 池）。池仅 2 种序列化：分层展示，明写不可做池级推断；聚类推断只按 session。
- **finish 目标**：金标工具为 finish 的行。

### 1.5 taxonomy 修正口径

- 修 `agent/r4_error_taxonomy.py:82` 的 falsy bug：`arguments` 为 `{}` 时不得误判「args 不可解析」（判定条件改为显式 `is None` / 键缺失）。
- 触顶行（censored）单独归新类 `TRUNCATED`，不再入 PROTOCOL_BROKEN。
- 解析器与主评分器口径对齐（消除 62 行既被判 PROTOCOL_BROKEN 又被主评分器计工具名正确的矛盾）。
- 三层分报：32k（checkpoint-2678，594×3）/ 76k-48（checkpoint-250）/ 76k-395（checkpoint-250）；不同 checkpoint 不并表。

### 1.6 V2 连贯性判定规则（补交代码复现用）

连贯 = 生成文本非空 且 字符级重复 4-gram 占比 < 50%。预期复现 43/48 与 5 个例外 qid（a45d2c09567a_795c2422:0、a45d2c09567a_cfce19ea:8、b455f37f04c7_903ca285:0、0c890a5dde8c_012517c3:0、0c890a5dde8c_012517c3:6）；同规则跑 c2kv 臂 395 行作连贯性附表。

### 1.7 边界逐行 flag（report-only）

按 `agent/r4_error_taxonomy.py` 的边界重算逻辑（tokenizer + 池文本，CPU），导出每行 target-schema 是否跨 512-chunk 边界；跨界判定为 ±200 固定窗、与 schema 长度无关；存在池内位置混淆且未做 full 臂差分——**禁止任何因果表述**。产出 `results/r5/analysis/boundary_flags.json`。

## 二、判据原文（逐字）

**W1（S6 主判据）**：主终点＝协议有效任务成功（§5.1 严格口径解析成功且工具名正确）；主分析层＝未截断×非 finish；配对 exact McNemar + session 聚类 bootstrap（B≥10000）。**判据中 p 指 exact McNemar p；判 (a)/(c) 须同时满足 session 聚类 bootstrap 95%CI 不跨零，任一不满足则落 (b)。** (a) full>c2kv 且 p<0.05 → 「质量税确认：大池 4× 无质量税设计点关闭，不再追加大实验」；(b) p≥0.05 → 「未决（功效限定），按关闭处理、措辞留余地」，若同时 |点估差|≥0.10 加注「点估差量级大但未达显著，不据此翻转」；(c) c2kv>full 且 p<0.05 → 「意外结果，冻结结论待复核」。次要终点（同表，不改判）：截断层、finish 语义线、调用率（带 censoring 注记）、全部样本合并。

**W2（S5 门）**：并集口径翻转 ≤1/20 → NEGLIGIBLE；≥2/20 → SUSPECT+扩大重跑。（翻转行数＝调用谓词或工具名判分至少一项与 r4 原行不一致的 qid 数，并集口径。）

**W3（S3）**：描述性重分析，无裁定行；全部口径定义先于计算提交，每表注明与 r4 原表的差异来源（口径/修 bug/剔除）。

**M0（S4 门）**：2607.02577 按 §6 口径判「已实现」(a) AND (b) → S8=KILLED-BY-PRECEDENT。（§6 口径：开源 harness 代码中存在可运行实现方计已实现；仅论文文字描述而代码缺失记 DESCRIBED-NOT-IMPLEMENTED，按未实现处理；repo 确实无法获取则以论文正文为准并记 CODE-UNAVAILABLE。）

**M1**：censoring 重分类率（「失败→成功」主口径）在 ≥1 个公共基准 × ≥2 个压缩方法上 ≥10% → 伪影泛化成立；<10% → NOT-GENERALIZED（内部案例仍成立但不外推）。

**M2**：外壳-语义分裂率（全样本分母）在 ≥1 个**压缩**条件（不含 base；base 的分裂率作对照另列）下 ≥5% → 成立；否则 NOT-SUPPORTED。

**M3**：基线＝各基准默认 cap＋原生评分下的压缩方法排名与聚合压缩税符号；修正＝金标长度键控 cap（取 max(1024, 金标长度 P95)）＋双列评分的语义列；同一冻结样本集上任一压缩方法排名对换或税符号翻转 → 「修正有后果」；(iii) 重加权为描述性，不入 M3。三者全不成立 → S8 判 NEGATIVE，报告如实交付。

## 三、S6 冻结与运行参数（先于跑批写死）

- **冻结样本**：从 395 主集抽样，四格 = {未截断,截断}×{75327 池,80171 池}，每格配额 24。可用量：未截断×75327＝24、截断×75327＝17、未截断×80171＝123、截断×80171＝231——75327 两格不足或恰好配额，取全并如实记录（预期实际总样本 24+17+24+24＝89）。80171 两格内均匀随机抽样，seed=20260816，写入 `configs/r5_closeout_qids.json`；格内 finish 目标 ≤25%（每格 ≤6），超出时对 finish 行降采样补非 finish 行。
- **双臂全部新跑**（不复用 r4 行）：full=修复版 runner（off-by-one 已修）；c2kv=r3 T-E 配置（chunk 512×160、eager、checkpoint-250）；两臂统一 **max_new_tokens=256**，其余生成参数与 r4 逐项一致，写入 `configs/r5_run_config.json`；权重 sha256、runtime attention 实现入 manifest。
- **埋点 schema（冻结，S8 引用）**：每行必录——每生成步 chosen-token logprob 与该步 EOS logprob；停止原因（eos/length/其他）与停止位置；每处 `<tool_call>` 出现位置及该处 EOS-vs-续写 logprob 差；金标 token 长度与 `gold_ge_cap` 旗标；双列评分 `protocol_valid`（§1.1 严格口径）与 `semantic_correct`（finish 用语义线 §1.2、非 finish 用工具名）——**永不合并**；分层元数据（clipped/池/finish）；违约分解列：工具名 EM、参数键 schema 合法性、跨块 call-observation 引用可解析性与引用距离（数据无该结构时逐行记 NOT-APPLICABLE）；c2kv 臂块边界与位置偏移；seed 与逐行 raw 全部 commit 到 `results/r5/closeout/`。
- **OOM 预授权**：仅 full 臂 prefill chunk 减半一次（512→256）并记录；c2kv 臂 512×160 冻结配置不得变更，c2kv 臂 OOM 直接停下报告。

## 四、统计纪律（沿用 R4）

配对 = 按 qid；McNemar exact（b/c 格数必报）；session 聚类 bootstrap（S6 主判据 B≥10000；描述性表沿用 20000 reps, seed 0）报 95%CI；session 数必明写。不同 checkpoint/regime 的数字不合并、不并排作结论。所有数字带 N 与出处。
