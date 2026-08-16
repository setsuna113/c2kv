# R5-S8 评测度量学 sprint 报告（公共基准复现）

状态：COMPLETE。零训练、纯推理；全部数字带 N 与出处；本报告所有产物位于
`task/r5-metrology` 分支，内部引用指向 PR-A（`task/r5-closeout`，PR#8）对应表格与
commit hash。

- 预注册：`configs/r5_metrology_prereg.md` @ e12acf5（判据 M0–M3 逐字冻结、
  类别选择、360 冻结样本 seed=20260816、抽取器规则、散文复核门）
- M0 kill 检查：`docs/metrology_killcheck.md` @ d715de2
- 全部计算产物：`metrology/data/` @ cfa7e8b（评分、分析、复核包）
- 执行环境：NPU 服务器，HF eager attention 路径，Qwen3-4B-Instruct-2507，greedy

## 1. 裁定行（判据逐字引自预注册/任务书 §11）

**M0（kill 门）**：「2607.02577 按 §6 口径判『已实现』(a) AND (b) → S8=KILLED-BY-PRECEDENT」。
结果：(a) 生成预算充足性检查 NOT-COVERED；(b) 双列评分 DESCRIBED-NOT-IMPLEMENTED
（按口径计未实现）；(c) 压缩条件 NOT-COVERED → **M0 不触发，S8 照常执行**
（证据与行号见 kill memo @ d715de2）。

**M1**：「censoring 重分类率（『失败→成功』主口径）在 ≥1 个公共基准 × ≥2 个压缩方法上
≥10% → 伪影泛化成立；<10% → NOT-GENERALIZED」。
结果：SnapKV 0.0%、StreamingLLM 0.31%，均 <10% → **NOT-GENERALIZED**
（内部案例仍成立但不外推）。

**M2**：「外壳-语义分裂率（全样本分母）在 ≥1 个压缩条件（不含 base）下 ≥5% → 成立」。
结果：SnapKV 最高 33.3%（cap=128）、StreamingLLM 14.4%（全 cap），任一压缩条件
≥5% 成立 → **M2_SUPPORTED**（修正协议 v2 口径；v1 对照见 §5 erratum）。

**M3**：「同一冻结样本集上任一压缩方法排名对换或税符号翻转 → 修正有后果」。
结果：基线与修正排名均为 base > SnapKV > StreamingLLM，税符号无翻转 →
**M3_NO_CONSEQUENCE**。如实加注：SnapKV 聚合压缩税从 +0.1194 缩至 +0.0056
（≈0），StreamingLLM 税从 +0.3028 升至 +0.3861——修正有数值后果但未达预注册
裁定线；描述性重加权（iii，不入 M3）下两套权重方案均出现 SnapKV/base 排名
对换与 SnapKV 税符号翻转（§4.3）。

## 2. 实验设置

- 底座：BFCL v4（gorilla 仓库 berkeley-function-call-leaderboard，快照
  sha256 见 `metrology/data/manifest_s8_runs.json`）。类别选择按预注册规则：
  全部 multi_turn 子类（base/long_context/miss_func/miss_param）+
  金标每轮 P95 最高的 parallel（187）与 parallel_multiple（173）。
- 样本：360 冻结样本（`configs/r5_metrology_sample.json` @ e12acf5），
  同一清单在全部 条件×cap 下复用；缺行记 MISSING 不换样（实际 0 缺失，
  3240/3240 行齐全）。
- 条件：base（无压缩）、SnapKV、StreamingLLM（后两者为官方逻辑在 eager
  attention 上的等价重实现 `metrology/kv_compress.py`，官方小样例对照测试
  19 项 @ cafac74）。C2KV 条件由内部 S6 closeout 数据充当（cap=256 固定，
  仅参与 (ii)(iii) 描述，表内标注 INTERNAL-ONLY）；公共底座上的 C2KV 集成
  未在自 S8 开工起 3 个自然日内跑通 → 按预注册标 **INTERNAL-ONLY**。
- cap 三档：default（BFCL 原生 OSS 语义 = min(4096, 剩余上下文)）、128、1024；
  金标长度键控修正 cap_c = max(1024, 金标每轮 P95) = 1024（全六类，
  P95 ∈ [60,187]）。
- tau2-bench 第二底座：未在 1.5 天内完成集成 → **DEGRADED-SKIPPED**
  （M1/M2 的"≥1 个公共基准"要求由 BFCL 单底座满足）。

## 3. 静态审计（测量 i 的附属）

默认 cap = min(4096, 剩余上下文) 对所选六类金标每轮长度（P95 ≤ 187 token）
充裕；cap=128 低于部分类别金标分位，是刻意的 censoring 应力档（审计表冻结于
prereg §2）。

## 4. 三类测量结果（出处：`metrology/data/s8_m1m2m3.json` @ cfa7e8b；评分输入
`metrology/data/s8_scored.jsonl`；原始运行清单 `manifest_s8_runs.json`）

### 4.1 (i) censoring 重分类率（M1 主口径；同一样本 greedy，cap=128 判失败/格式错 → cap=1024 重跑判成功的比例）

| 条件 | cap128 失败 n | cap1024 恢复 n | 重分类率 | 任何标签改变率 |
|---|---|---|---|---|
| base | 215 | 2 | 0.93% | 0.56% |
| snapkv | 256 | 0 | 0.0% | 0.0% |
| streamingllm | 323 | 1 | 0.31% | 0.28% |

（每格 N=360 冻结样本。）本底座+本模型下失败由语义/解码因素主导而非预算
（censored@128 行占比仅 2.2%/3.1%/9.7%），与内部 76k 层"金标中位 130 > cap 128"
的构成不同——这是 M1 未泛化的直接读数，如实报告。

### 4.2 (ii) 外壳-语义分裂率（M2 主口径；分母 = 该条件×cap 全部冻结样本 360）

| 条件 | default | 128 | 1024 |
|---|---|---|---|
| base（对照） | 30.6% | 30.8% | 30.6% |
| snapkv | 32.2% | **33.3%** | 32.2% |
| streamingllm | 14.4% | 14.4% | 14.4% |

C2KV 条件（INTERNAL-ONLY，cap=256，S6 数据）：full 2/89 = 2.2%，
c2kv 2/89 = 2.2%（描述性，不参与 M2 判定）。

### 4.3 (iii) 构成敏感性（描述性，不入 M3）

以（类别 × 轮深分箱）14 个非空层直接标准化重加权：

| 方案 | 口径 | base | snapkv | streamingllm | 排名 | 税 snapkv / sllm |
|---|---|---|---|---|---|---|
| (a) 各层等权 | 基线 | 0.2942 | 0.2327 | 0.0554 | base>snapkv>sllm | +0.0616 / +0.2389 |
| (a) | 修正 | 0.5081 | **0.5322** | 0.1969 | **snapkv**>base>sllm | **−0.0241** / +0.3112 |
| (b) 内部 395 构成 | 基线 | 0.3597 | 0.2606 | 0.1096 | base>snapkv>sllm | +0.0991 / +0.2501 |
| (b) | 修正 | 0.5572 | **0.5682** | 0.2396 | **snapkv**>base>sllm | **−0.0111** / +0.3175 |

两套固定权重方案下，修正口径均使 SnapKV 与 base 排名对换、SnapKV 税符号翻转
（+ → −，量值 ≤0.024）。方案 (b) 的「层→格」映射为无自然对应下的固定约定
（analyze_s8.py MAPPING_NOTE，逐字见 s8_m1m2m3.json），不可做池级推断。
本小节为描述性结果，不入 M3 裁定，不作因果表述。

### 4.4 M3 主表（排名与税符号；acc 分母 = 360，缺失按 0 计，实际无缺失）

| 口径 | base | snapkv | streamingllm | 排名 | 税 snapkv / sllm |
|---|---|---|---|---|---|
| 基线（default cap + 原生评分） | 0.4083 | 0.2889 | 0.1056 | base>snapkv>sllm | +0.1194 / +0.3028 |
| 修正（cap_c=1024 + 语义列） | 0.6361 | 0.6306 | 0.2500 | base>snapkv>sllm | +0.0056 / +0.3861 |

无排名对换、无税符号翻转 → M3_NO_CONSEQUENCE（加注见 §1）。

## 5. erratum：散文抽取器 v1 → v2（预注册复核门触发的强制回路）

- v1（prereg 冻结规则：可用函数词典 + 首个命中 + 参数键沾边）产出的 30 例分层
  复核（seed=20260816，自 1765 分裂行抽样）判 **16/30 不一致**（初判 17/30，
  case 12 经全文复读更正为一致；`metrology/data/review_verdicts_notes.json` 含
  逐例理由与更正记录 @ 61e55a5/3d3879e），超过预注册 3/30 门限。
- 按预注册执行修订（erratum）：**v2 = 金标函数名词典（missed_function 豁免）
  + 全覆盖要求 + 参数规则不变**（规则逐字见 `metrology/prose_extract.py`
  `extract_semantic_v2` docstring @ ea85438）。v1 冻结保留为 `prose_v1_frozen`
  参照列。
- v2 重评分后分裂行 1765 → 839；同 seed 重抽 30 例复核判 **1/30 不一致**
  （case 16：金标 mean 未被调用，散文提及"mean"一词被 200 字符调用形态窗穿透
  误认覆盖——v2 已知残余假阳模式，如实记录；
  `metrology/data/review_verdicts_v2_notes.json` @ cfa7e8b），通过复核门。
- 本报告全部 M2/M3 数字以 v2 口径为准；v1 对照：M2 各格分裂率 53.6%–55.6%
  （方向不变，量值虚高约 1.7–3.8 倍）。
- v2 已知假阴性（保守方向，可接受）：3/13 人工一致例（v1 包 case 5/6/21，
  导航/冗余步缺失但任务已完成）v2 判 incorrect；此类行不进入分裂总体，不影响
  复核门。

## 6. 内部锚定案例（本团队测试基建迭代中的自我修正案例；数字以 PR-A 修正表为准）

1. **预算 censoring 误标失败类型**：76k c2kv 层 206/395 个 PROTOCOL_BROKEN 中约
   94%（=(206−12)/206；185 行直接触 128 上限）为生成预算伪影；金标中位 130
   token，211/395 金标在 cap 内写不完（`results/r5/analysis/` S3 修正表，
   PR-A @ `task/r5-closeout`；W2 门另判 r4 full/prior 臂 SUSPECT，
   `results/r5/analysis/offbyone_ab20.json` @ 662af25）。
2. **协议外壳 vs 语义正确的构念分裂**：内部 finish 目标上，full 臂 12/79 行
   （15.2%）严格主口径判错但 ROUGE-L F1 ≥ 0.5
   （`results/r5/analysis/finish_semantics.json` s8_anchor @ d3b345f）。
3. **构成混杂制造符号翻转**：395 主集聚合 p=0.0009 偏压缩臂，未截断（需读池）
   层 full .333 vs c2kv .082（p=9.3e-09）方向相反；session 聚类 CI
   [−0.2079,+0.0057] 跨零（`results/r4/analysis/paired_76k_main395.json` @
   a2683be；分层表 `results/r5/analysis/v1_stratified_strict.json` @ e1c1790）。
4. **"先验地板"实为历史复制地板**：prior 臂剔除 4/48 长度违规行后地板 acc
   0.1818/call 0.5909，8 个命中全为 3–9 轮重试延续中复制可见 1920-token 历史
   的同名调用（首轮 0/5）（PR-A S1/S3 修正记录 @ `task/r5-closeout`）。

W1（内部大池质量税终裁，背景）：matched closeout 89 配对、主层 n=47/22 session
full .3404 vs c2kv .1915，exact McNemar p=0.1435，聚类 CI [−0.0732,+0.3654]
跨零 → 预注册 (b) 未决（功效限定），按关闭处理、措辞留余地
（`results/r5/analysis/closeout_w1.json` @ f626ae5）。

## 7. 相关工作对位

- **2607.02577**（最近邻 form factor）：无压缩的工具调用评测效度审计 + 修正
  harness。我方 delta = 压缩 × 预算 censoring × 双列评分 × 构成反转；M0 判定其
  (a)(b) 未实现（kill memo @ d715de2）。
- **2605.07395**：预算截断作为评测伪影的同类先例（routing 场景）。
- **2608.01056**（Control Under Compression / CompressAgent）：text 层压缩
  benchmark，无伪影解剖；其环境放出情况在讨论中备注，不因此新增底座。
- **2608.01631**（KV 压缩下推理忠实性）：其主张"指标不完整"；我方主张"评测装置
  制造伪影"——本 sprint M2 为其提供公共基准上的装置级证据，M1 显示
  censoring 通道并不自动泛化，两主张需分通道对位。
- 辅助引用：2605.23950、2605.27922、2605.24660、2605.18857、2510.00231、
  2412.17483。

## 8. 引文流监控记录

- pass 1（开工时，2026-08）：检索引用 2608.01056 与 2608.01631 的新文及同关键词
  新文——无新被引；命中方法学邻居 AGC-Bench 2607.01152v2（Appendix G）。
- pass 2（收尾前，2026-08-16）：Semantic Scholar API 查两文被引 = 0/0；同关键词
  检索（tool-calling 评测效度/预算截断伪影/KV 压缩 agent 评测）无新同题文；
  新见 2607.17715（sidecar 式可学习压缩 token，方法邻居，非评测效度方向）。
- 结论：本 sprint 的测量空位（压缩 × censoring × 双列评分 × 构成反转）未被
  抢占。

## 9. 降级与事故日志

- tau2-bench 第二底座 DEGRADED-SKIPPED（§2）。
- C2KV 公共底座集成 INTERNAL-ONLY（§2）；其 (ii)(iii) 测量由 S6 内部数据充当，
  cap=256 固定已在表内标注。
- 抽取器 v1 → v2 erratum 全流程（§5），含复核者自身一处更正（case 12）。
- M3 分析器 c2kv 描述性配对键 bug（id/qid）在实跑中发现并修复
  （47a8ae2；n_pairs 1/89 → 89/89），修复后数字不变入主判定（描述性列）。
- 原始运行 jsonl 合计 108MB 不入 git（仓库体积纪律），以 sha256 清单
  （`metrology/data/manifest_s8_runs.json`）+ 冻结样本/seed/runner 可复现；
  衍生评分数据（2MB）已入库。

## 10. 数据清单（全部 @ cfa7e8b，除另注）

- `metrology/data/s8_scored.jsonl`（3240 行双列评分）+ `s8_scored_summary.json`
- `metrology/data/s8_m1m2m3.json`（全部判定数字 + 输入 sha256）
- `metrology/data/manifest_s8_runs.json`（9 格原始运行 sha256/行数/字节数 +
  快照 manifest sha256 ef5f252a…）
- `metrology/data/review_packet.json` / `review_verdicts.json` /
  `review_verdicts_notes.json`（v1 复核 @ 61e55a5/3d3879e）
- `metrology/data/review_packet_v2.json` / `review_verdicts_v2.json` /
  `review_verdicts_v2_notes.json`
- 代码：`metrology/bfcl_hf_runner.py`（f1daf42）、`kv_compress.py`（cafac74）、
  `bfcl_score.py` / `prose_extract.py` / `analyze_s8.py` / `review_sample.py`
  （ea85438），测试 75 项全绿。

## 11. 清场记录

跑批 driver PID 1813803 与 7 个 worker PID（1813841/1813842/1813844/1813846/
1813848/1813850/1813852，卡 0,1,3,4,5,6,7）于 2026-08-16 全部按精确 PID 复核
退出（kill -0 无存活）；退出后 npu-smi 复核无本任务残留进程与显存占用。
