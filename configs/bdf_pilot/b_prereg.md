# 实验 B · Stage-1 分块 policy pilot 预注册

> **状态**：预注册草案，跑第一臂之前定稿并冻结。定稿后本文件只允许追加「实际值填充」，
> 不允许修改判据、阈值或臂表——事后改判据 = post-hoc。
>
> **pilot 不判方向生死。** 本轮全部产出标注 `mechanism pilot, appworld-domain,
> n=1 checkpoint`；任何"某 policy 更差"的表述都受 §5 的 OOD 不对称规则约束。

上位文档：`24_实验设计手册_BCDFG_2026-08-21.md` 第 0 章与实验 B 章（B.4.2 / B.4.3 /
B.4.5 / B.4.8）。本文件是它在本 repo 的可执行落点，冲突处以 24 号为准。

---

## 1. 臂表与 flag 映射

内容集冻结：**四臂共享同一份已冻结的 history 文本流**（现役 P-turn 的
`_fit_reused_history` 输出 + provenance），其它 policy 只在这份文本流上重新划边界。
P-turn 臂因此字节不变（in-distribution 参照零回归），跨臂内容同一性由构造保证并由
`python/train/test_chunk_policy.py` 与 `agent/test_chunk_policy_traces_integration.py`
锁定。

| 臂 | 24 号编号 | flag | 说明 |
|---|---|---|---|
| P-fixed | P3 | `--chunk_policy fixed-1024` | 忽略轮边界，冻结内容顺序拼接后按 1024 token 切窗。**gist 申报的参照臂** |
| P-turn | P5 | `--chunk_policy agent-turn` | 现役默认（`_agent_history_turn_docs` + 超长切分），**in-distribution 参照** |
| P-struct | P6 | `--chunk_policy structural` | 每个原子块一个 doc；`(assistant with Action:/<tool_call>, 其后 tool/observation)` 为不可拆原子块 |
| P-delay | P5+L | `--chunk_policy agent-turn --delay_recent_turns 1` | 最近 1 轮不压、拼进 raw prompt（turn 粒度） |
| 参照 | P0 / P0′ | `--compare_modes full,truncate --chunk_policy agent-turn` | 一次共享跑；full 供 R_agent 与转移矩阵 |

臂→flag 的**唯一定义处**是 `agent/run_b_pilot_npu.sh` 的 `arm_flags()`，脚本启动时逐条 echo。

**P-struct 超长原子块处置（显式 supersede 24 号 B.4.3 P6 的字面「pair 内禁断」）**：单个
`(tool_call, observation)` 原子块 wrap 后超过 1024 token doc 预算时，「pair 内禁断」与
「装进单个 doc」不可同时满足；实现回落为 `_split_message_to_fit` 按语义单元在块内切分
（可能断进 observation 内部），逐行计入 `structural_fallback_docs`。配套计数语义：冻结期
已被切成多 part 的超长 turn shard 不重切、整体 pass-through（`structural_partial_docs`）；
不足两个语义单元的 doc 原样通过（`structural_passthrough_docs`）；真正按原子块重打包的
doc 计 `structural_repacked_docs`。内容字节流不变（内容集冻结不受影响）；判读时 P-struct
的描述表须带 fallback 行占比。

**P-fixed 有效窗口**：fixed-N 的实际切窗 = N − chat-template wrapper 开销 −
8 token 再编码 margin（`chunk_policy.FIXED_WINDOW_MARGIN`；BPE decode→re-encode 不保长），
即**内容守恒优先于名义窗宽**。wrap 期任何顶到 `max_doc_length` 上限且确有内容被截的 doc
逐行计入 `wrap_truncated_docs`（正常恒为 0，非 0 即 margin 不足、须查 re-encode 漂移）。

未列入本轮的 24 号臂：`fixed-256` / `fixed-512` 已实现（`--chunk_policy fixed-256|fixed-512`）
但不在 Stage-1 主扫内；`natural-paragraph` 不实现（24 号 B.2 依 arXiv 2410.13070 的负结论删臂）；
sink-decoy prefix 因子（+D）不在本轮。

**红线合规声明**：chunk policy 是**静态预注册配置**，全数据集统一应用，不存在任何 per-input
决策、评分或选择机制。代码与文档中不出现 router / gate / ratio selection /
adaptive compression ratio / verifier 作为我方设计命名；内部判据编号称「判据 N」。
本章产出定位为机制归因与 G8 数据管线配置证据，不进贡献列表（24 号 审查裁定 B-1）。

---

## 2. 冻结物与可追溯性

| 项 | 值 | 状态 |
|---|---|---|
| eval-200 qid manifest | `configs/bdf_pilot/b_eval200_qids.json` | **[待填]** 由 G-S2 嵌套冻结集的 eval-200 层导出，不得自行抽样（24 号 审查裁定 1-7/1-8） |
| manifest sha256 | `[待填: sha256]` | 冻结后填入；driver 侧以 `--qid_manifest` 引用，缺失 qid 计数进 run summary `qid_manifest_missing` |
| eval commit sha | `[待填: git rev-parse HEAD]` | 全臂必须同一 commit |
| checkpoint | `[待填: MODEL_PATH + checkpoint-*]` | small joint 臂 @32M（post-mask-fix 训练 / pre-capfix 数据 / post-capfix eval 代码） |
| base model | `[待填: BASE_MODEL]` | full/truncate 参照臂载体 |
| split manifest | `[待填: SPLIT_MANIFEST_FILE + name]` | |
| ratio | 8（`--ratios 8`） | 全臂同一 |
| decode | greedy（`--do_sample false`）；如启用采样则记 `temperature/top_p/gen_seed`，per-row 播种公式 `manual_seed((gen_seed*1_000_003) ^ crc32("{qid}:{mode}:{ratio}"))` | |

`qid_manifest_missing > 0` 即为**共同 qid 集不完整**，该轮不得进任何配对表。

---

## 3. 判据（阈值 + CI 规则，全部预注册）

### 判据 0 · RoPE 单测门（对应 24 号 门0）

`python/models/test_rope_reposition.py` 五条性质全过才准跑任何臂：零 delta 回归、
存-旋等价与 ±d 往返、位置记账 path1（`reconstruct_kwargs=None`）与 path2
（`reconstruct_kwargs={}` + eval 模式）、blend/concat 一致性。
任一失败 ⇒ **implementation-invalid**，修码前所有臂冻结。

### 判据 1 · gist 预算申报（对应 24 号 门1）

任一臂的 `avg_gist_tokens`（=`analyze_b_pilot._gist_declaration_table` 的
`mean_gist_tokens`）偏离参照臂 **P-fixed** 超过 **5%** ⇒ 该臂 **VOID**：其数字不进任何
排序，须实现 per-row gist 预算分配后重跑。**不 kill idea。**

P-delay 的未压缩近轮 KV 不是 gist 支出：它单列为 `mean_raw_recent_tokens` 成本列，
该臂**豁免** 5% 规则（偏差照报，标 `EXEMPT`）。

早期预警（不占 NPU）：`agent/test_chunk_policy_traces_integration.py` 在真 parquet +
真 tokenizer 上打印 Σceil(len/8)。**[待填: 本机 n=50 实测读数]**——首跑观测值见 §7。

### 判据 4 · 块长（对应 24 号 门4）

全部两两配对差的 95% CI **上界 < MDE** ⇒ 「块长在该 MDE 粒度下不可分」，可报负结果；
任一 CI 跨界 ⇒ **inconclusive**，只写 "not separable"。
**禁止以低于 MDE 的点估计写阴性结论。** 两种情形下资源决策均为 G8 维持 1024。

### 判据 5 · 结构臂筛选（对应 24 号 门5）

两条预指定 **primary contrast**：**P-struct vs P-fixed**、**P-struct vs P-turn**
（**P-turn vs P-fixed** 为第三预指定）。判定条件 = Δ ≥ MDE **且** 95% CI 下界 > 0。
"vs 最佳 fixed 臂"只作 exploratory 描述（赢家诅咒防线）。其余两两比较标 exploratory，
family 内 **Holm** 校正后才允许任何"胜出"表述。analyzer 自动打上 `primary`/`exploratory`
标签并只对 exploratory family 做 Holm。

### 判据 8 · delayed 因子（对应 24 号 门8）

P-delay 需在 **bytes-matched** 口径下 Δ ≥ MDE。realized bytes =
`(gist_tokens + raw_recent_tokens) × 147456 B`（Qwen3-4B，[per 22 号；算术自洽
36×8×128×2×2B]）。`raw_recent_tokens > 0.5 × 参照臂同 qid gist 预算` 的行在
bytes-matched 列**跳过并单列计数**（24 号 审查裁定 4-5）。只在 elastic 口径赢 = bytes-质量
曲线上的另一个工作点，**不得当作等预算胜利**。

---

## 4. 统计纪律

- 全部判读只在**共同 qid 配对**口径下进行（`analyze_b_pilot._common_qids`）。
- 主口径 S = `tool_name_match`；敏感性口径 exact_match / argument F1 单列，永不合并。
- 同报 **R_agent = P(S_arm=1 | S_full=1)** 与**绝对成功率**；只报条件量禁止。
- 配对转移 C→C / C→W / W→C / W→W 对 full 臂报；格子 n 现算，n < 54 时只报点估计 + CI，
  零排序断言。
- CI = session-cluster bootstrap（20000 reps, seed 0；item-resampling，**不含 seed 方差**，
  逐表标注）。McNemar 用精确二项，b/c 格子照报。
- 内容集：臂间 presented source tokens 差 > **2%** ⇒ 分析层 post-stratification（presented-token
  十分位分桶，桶内配对、桶间按参照臂份额加权），并在表格脚注声明。
- 配对 MDE：n=200 → 8.9pp（[interpolated from R2 power table, π_d=0.2 assumed]）；
  其它 n 按 1/√n 内插并标注。**细于 MDE 的排序主张 = 禁止。**
- 非配对比较（跨 checkpoint、跨池）本协议内**禁止出现**。

### 统一脚注（每张结果表逐字带，analyzer 强制输出）

```
<n>-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ <mde>pp; no claim below MDE is a ranking.
```

n=200 时即：

```
200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking.
```

### 四问判读卡（写进判读记录，不写进 kill 逻辑）

① headroom 存在吗（对 sham / 噪声地板）② 优于简单基线吗 ③ 成本合理吗 ④ 哪类失败最受益。

### 停止条件白名单（仅这五条）

implementation-invalid / 无 headroom / 被简单基线支配 / 成本不可接受 / 优先级。
**不得因「与某论文相似」停止。**

---

## 5. OOD 不对称判读规则

pilot 载体是用 agent-turn + 1024 切片训出来的，评其它 policy 对 extractor 是 OOD，
偏差方向**利好现役 policy**。因此：

- 非现役 policy **赢** 现役 policy ≥ MDE = 保守可信，进 Stage-3；
- 非现役 policy **输** = 不可归因（OOD confound），**不得写"该 policy 更差"**，
  只能写 "not separable under OOD screening"。

---

## 6. W&B tag 方案

真源 = W&B entity `liuyc1025-university-of-cambridge`；不在 W&B 的数字不进任何表。

每个 run 的 tag：

```
expB
arm=<P-fixed|P-turn|P-struct|P-delay>
policy=<fixed-1024|agent-turn|structural>
delay=<0|1>
eval_sha=<eval commit sha>
manifest_sha=<eval-200 manifest sha256 前 12 位>
ckpt=<checkpoint 目录名>
```

`analyze_b_pilot.py` 收尾打印 `wandb_tag_map arm=... chunk_policy=... delay_recent_turns=...`
逐臂映射行，直接抄进 run tag。

---

## 7. 执行与产物

```
NAME=b_pilot \
MODEL_PATH=... BASE_MODEL=... DATASET_PATH=... \
SPLIT_MANIFEST_FILE=... QID_MANIFEST=configs/bdf_pilot/b_eval200_qids.json \
bash agent/run_b_pilot_npu.sh
```

产物：`outputs/b_pilot/<arm>.jsonl` + `.summary.json`、`outputs/b_pilot/reference.jsonl`、
`outputs/b_pilot/b_pilot.analysis.{json,md}`。

判读记录（带时间戳，防 post-hoc）：**[待填]**。

### 首跑前的本机读数（真 traces + 真 Qwen3-4B tokenizer, n=50, ratio=8, joint 条件）

`agent/test_chunk_policy_traces_integration.py` 输出，**仅为早期预警，不是判据 1 的判定**
（判定在 NPU 实跑的 `avg_gist_tokens` 上）：

| 臂 | 平均 chunk 数 | content tok | wrapped tok | raw recent | Σceil(len/8) | vs P-fixed |
|---|---:|---:|---:|---:|---:|---:|
| P-fixed | 2.66 | 2169.3 | 2182.6 | 0.0 | 273.2 | 0.00% |
| P-turn | 5.46 | 2169.3 | 2192.3 | 0.0 | 276.4 | +1.15% |
| P-struct | 10.50 | 2169.3 | 2232.5 | 0.0 | 283.4 | +3.74% |
| P-delay | 4.28 | 2169.3 | 2192.3 | 459.8 | 218.3 | −20.09%（EXEMPT） |

读法：四臂 content token 完全相同（内容集冻结生效）；P-struct 在 ratio=8 下的 ceil 圆整 +
模板开销共计 +3.74%，**低于 5% 但没有多少余量**——若正式跑越过 5%，按判据 1 该臂 VOID 并
需 per-row 预算分配后重跑。presented（wrapped）token 最大偏差 2232.5 / 2182.6 = +2.29% >
2%，**post-stratification 预期会被触发**。P-delay 的 −20% gist 由 460 个 raw 近轮 token
换来，走判据 8 的 bytes-matched 口径，不参与判据 1。

（这一节的数字来自本机 CPU 上的 chunking 计算，非 NPU 评测结果；不含任何模型输出。）

---

## 8. 对存量代码的行为变更申报（集成阶段实施）

本轮为了让 B 的四臂能跑，改动了若干存量文件。**凡是会改变既有产物 schema 或既有
数值口径的，逐条申报在此**，供下游消费方（joint-eval summary 的读取脚本、W&B 面板、
以及任何比对历史 summary 的分析）核对。

### 8.1 `eval_joint_next_action_c2kv._summarize`：硬下标 → `.get` + 缺字段计数

**改了什么。** `_summarize` 原先用硬下标读四个 rate 字段
（`row["exact_match"]` / `row["tool_name_match"]` / `row["has_tool_call"]` /
`row["response_type_match"]`），并用 `row.get(field, 0.0)` 读各 avg 字段。现在四个 rate
一律走 `row.get(field)`，avg 走 `row.get(field) or 0.0`。

**为什么。** 这是集成阶段发现的既有 bug，不是本实验引入的：`--merge_only` 合并跨代码版本
的 shard 时（或某个 shard 写到一半），有的行根本没有这些键，`_summarize` 会直接抛
`KeyError` —— 长跑的最后一步整体失败，已算完的行全部作废。B 的流程正好会踩到：每臂一次
`--merge_only`，且不同臂/参照臂的行字段集不完全一致。

**下游要知道的（新键 + 新日志）。**

| 变化 | 位置 | 语义 |
|---|---|---|
| **新增键 `missing_metric_fields`** | 每个 `condition × mode × ratio` summary entry | `{字段名: 缺该键的 valid 行数}`；无缺失时为 `{}`。**不是空字典就说明该 entry 的 rate 被稀释过** |
| 新增 WARNING 日志 | `_summarize` | `... N/M valid rows are missing metric keys ... those rows were folded in as 0 and the affected rates are DEFLATED` |

**判读约束（写进判读记录）。** `missing_metric_fields` 非空的 entry，其 `exact_match` /
`tool_name_accuracy` / `tool_call_rate` / `response_type_accuracy` 是**被缺失行按 0 稀释过的
下界**，不是实测率。此类 entry **不得进任何配对表或排序**，须先查明缺字段来源（多半是混了
不同 commit 的 shard）并重跑。这与 §2 的 `qid_manifest_missing > 0` 同级处理。

**兼容性。** 只增键不改键：既有键的名字、类型、口径全部不变；缺失行为零的旧行为在
`missing_metric_fields == {}` 时与改前逐字节一致。读旧 summary 的脚本不受影响；读新 summary
的脚本若不认识该键会忽略它——因此**判读时必须人工/脚本显式检查这个键**，否则稀释会静默通过。

### 8.2 `build_history_chunks` 的 `content_tokens` 改为按需测量

`train_data_joint.build_history_chunks` 新增 keyword-only `need_content_tokens`
（默认 `False`）。为算 `content_tokens` 要把整段 history 再 encode 一遍（每例最多 ~24k
token），而这条路径同时跑在 trainer 的 dataset 构建里，会让训练侧 history 的 tokenization
成本大致翻倍。因此：

- **trainer**（`JointDataset.preprocess_example`）传 `False` —— 它从来只把这个数写进日志；
  此时 `content_tokens` / `policy_content_tokens` 为 **`None`（未测量）**，刻意区别于实测的 0。
- **eval driver**（`_condition_doc_chunks`）传 `True` —— 判据 1 的 gist 申报与 §4 的
  presented-token 检查都要用它。**每行输出的 `history_content_tokens` 语义与口径完全不变。**

分块结果（`kept` / `delayed` 的逐字节内容）与该开关无关，由
`python/train/test_chunk_policy.py::test_trainer_path_pays_no_extra_encode_for_content_tokens`
逐 policy 锁定。

### 8.3 `--do_sample true` 必须显式给 `--temperature`

`eval_joint_next_action_c2kv.parse_args` 新增校验：`--do_sample true` 且未给
`--temperature` 时直接 `parser.error` 退出。原因是 HF 在 `temperature=None` 时会静默回落到
checkpoint 自带的 `generation_config`，而 run summary 记的是 `"temperature": null` ——
**采样配置无法从产物复原**，违反 §2 的可追溯性要求。`agent/run_b_pilot_npu.sh` 相应给出
`TEMPERATURE=0.7` / `TOP_P=0.95` 默认值（与 `run_f_pilot_npu.sh` 同值），且**仅在
`DO_SAMPLE` 为真时**下发这两个 flag —— greedy 跑的 summary 仍如实记 `temperature: null`。
本轮 §2 约定的 decode 仍是 greedy，此项只影响将来启用采样的跑法。

---

## Changelog — 2026-08-22 pre-first-run amendments

首跑之前、定稿冻结之前的修订（审计 findings 处置）。**判据编号、全部阈值与臂表均未改动**；
以下为分析/记账机制补齐与文字对齐，逐条附一行理由：

1. **判据8 判定量落地**（analyzer 新增）：`analyze_b_pilot.py` 增算 bytes-matched 口径的
   delay 臂 vs 参照臂配对 contrast——只用通过 0.5× raw-bytes 守卫的行（守卫定义与
   `_delay_accounting` 完全一致），报 `n_used` / `n_excluded_budget_guard`，复用同一
   McNemar + session-cluster bootstrap，MDE 按缩减后 n 重算；写入
   `analysis.json` 的 `delay_bytes_matched` 与 md 的「Bytes-matched delayed-arm
   contrast (判据8)」节。理由：判据8 的判定量此前不在任何产物中，预注册判据无法机器判读。
2. **共同 qid 集完整性自动执行**（§2 硬规则的执行点）：`--merge_only` 合并 summary 透传
   `qid_manifest`（取首个非空，路径不一致打 WARNING）与 `qid_manifest_missing`
   （各 shard 求和）；`analyze_b_pilot.py` 新增 `--qid_manifest`，任一臂（含 full 参照）未
   覆盖 manifest 即在 analysis.md 顶部盖「INCOMPLETE COMMON-QID SET」横幅并写
   `qid_manifest_check.incomplete_common_qid_set` 布尔（表仍产出，仅描述性）；
   `run_b_pilot_npu.sh` 向 analyzer 下发 `QID_MANIFEST`。理由：§2 的「missing>0 不得进
   配对表」此前无任何自动执行点，且合并 summary 会丢该计数。
3. **§8.1 语义在 analyzer 侧执行**：缺 `tool_name_match` 键的行从所有配对表**剔除**
   （不再被 `bool(row.get(...))` 折叠为答错），逐臂计入 `missing_metric_rows` 并在 md 加
   WARNING 行。理由：analyzer 直读行文件绕过了 summary 的 `missing_metric_fields` 防线。
4. **post-stratification 增配 CI**（判据5 的混杂修正口径判读量）：`_poststratify` 对桶加权
   差做 session-cluster bootstrap（同 reps/seed 纪律；十分位划分与参照臂桶份额固定、桶内
   均值重采样、空桶权重在场桶间 renormalize），`weighted_diff_95ci_pp` 进 json 与 md。
   理由：§7 首跑预读已写明触发是预期主路径，而此前修正口径只有点估计、无 CI 可判。
5. **VOID 臂标注进 contrast 表**：涉及判据1-VOID 臂的 contrast 行在 json 加
   `void_involved: true`、md 行尾加 `[VOID]` 并附图例（数字保留，仅作诊断）。理由：
   「不进任何排序」此前只靠读表人交叉核对 gist 表。
6. **wrap 截断计数器**：eval 侧 kept/delayed 消息的 chat-template wrap 若顶到
   `max_doc_length` 上限且经无上限二次 encode 确认确有内容被截，逐行计入新字段
   `wrap_truncated_docs`（正常恒 0；候选为 0 时二次 encode 零成本）。理由：fixed 窗口的
   8-token margin 是启发值，re-encode 漂移超限此前静默丢内容、无任何可见信号。
7. **§1 文字对齐实现**：新增「P-struct 超长原子块处置」（显式 supersede 24 号 B.4.3 P6
   字面的「pair 内禁断」，并申报 `structural_fallback_docs` /
   `structural_partial_docs` / `structural_passthrough_docs` / `structural_repacked_docs`
   计数语义）与「P-fixed 有效窗口」两段。理由：单条消息 > 1024 token 时 P6 字面自相矛盾、
   prereg 此前对该情形沉默；fixed-1024 的名义窗宽与实现不符。
