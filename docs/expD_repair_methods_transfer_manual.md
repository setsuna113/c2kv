# 实验 D 后续：修复方法逐条迁移手册（对照 deep research 报告）

> **状态（2026-08-31 补记）**：本手册写于 2026-08-28，是 D 线批次 0–3 的原始计划书。
> 批次 1 已执行完毕，结果见 `29_本周实验汇报_2026-08-30.md` 与
> `30_下周素材_D1night与54首批_2026-08-31.md`；若干条已有裁决，**读本文时以周报为准**：
> - **K1**：定位是全部问题。witness 块（按答案线索选块）71/93 = 76.3%，oracle 87.1%，
>   乱修一块地板 25.0%；旧 median 策略 35.5%（witness==median 仅 19/90）。
>   本文里"k\*=median"的一切表述已作废。
> - **A2 corr_text / D2 文本 erratum**：**判死**（S=4/93，活性分母 9.5%）。修复需要块的 KV 内容。
> - **B1 放置**：布局是二阶量（各臂差 ≤2.2pp）；最佳臂 raw_erratum_tail（锚在历史末尾）80.6%。
> - **C3 成本交叉点**：~4K，4K 起打补丁严格更便宜（0.83×）。
> - **A1 re_only**：corr@first 已追平 corr_re（40.86%），corr_re 的优势来自 v1 选错了块。
> 仍然有效的部分：各方法的**文献映射与 arXiv 锚点**、**kill 判据**、批次 2/3 的设计。

> 用法：每条 = 方法 → 报告出处 → 文献（含报告标的核实状态）→ 报告给的证据 → 我们的实验设计（臂布局 / 对照 / 指标 / kill）→ 优先级。
> 臂记号沿用 D 线：`S` 系统提示 + 工具池，`Gi` 第 i 块 gist KV，`Ri` 第 i 块 raw KV，`Ti` 第 i 块原文重新 prefill，`R′/G′` 在修正后 prefix 上重算，`Q` 当前请求；示例仍取 5 块、k\*=2。
> 标【报告】的是报告原话或数字；标【推断】的是我据此做的设计判断，需要你自己核。
> 报告出处缩写：TL;DR、KF1–6（Key Findings）、Q1 表 a–k 行、Q2–Q8、迁移表 #1–6、Rec.1–6（Recommendations）、空白(1)–(6)（Caveats 里的空白清单）。

## 0. 总览表

| # | 方法 | 报告出处 | 主要文献 | 新增臂 | 训练 | 工程量 | 优先级 |
|---|---|---|---|---|---|---|---|
| A1 | errata 机制检验：下游污染 vs 下游变 raw | KF1 / Q1-a / 迁移表#6 | Models Take Notes 2606.17107 | re_only, corr_regist | 否 | 低 | P0 |
| A2 | 文本 errata（追加原文而非 raw KV） | Q1-i（报告"待找"） | 无先例 | corr_text | 否 | 低 | P0 |
| B1 | 放置方式 2×2：末尾追加 vs 原位；保 G 还是丢 G | Q3 末段 / Q7#3#4 / 迁移表#5 / 空白(2) | Leyline 2606.01065, KVLink 2502.16002, SSA 2604.20920 | corr_dropG, splice_keep, splice_rep | 否 | 低 | P0 |
| C1 | 下游重算改 gist（省显存）+ 等效压缩率算账 | 迁移表#1 / Q2 末条 | CacheBlend, ProphetKV, ResKV 会计 | corr_regist | 否 | 中 | P0 |
| C2 | 选择性重算 + 重算预算阶梯（省算力） | TL;DR(1) / KF2 / Q2 / Q3 / Q7#2#5 / 迁移表#1 / Rec.2 | CacheBlend 2405.16444, ProphetKV 2602.02579, EPIC/LegoLink, CacheClip 2510.10129 | corr_sel{dev,qaware,boundary,random}@r | 否 | 中高 | P1 |
| C3 | 成本 crossover 长度扫描 + p95 口径 | TL;DR 第二条 / Q2 / Rec.4 | ProphetKV, SIFT 2606.09441 | 无新臂 | 否 | 中 | P0 |
| D1 | 共享 offset 加性校正 | TL;DR(3) / KF3 / Q3 / 迁移表#3 / Rec.3 | AgentKVShift 2607.21604 | offset_kv, offset_act, offset_sham, offset+corr | 否 | 中 | P1 |
| D2 | gist 键的 attention logit bias | Q1-d | SelKV 2607.16213 | bias_b | 否 | 低 | P2 |
| D3 | 行为 steering（发射先验） | Q1-d | Cache Steering 2507.08799 | steer_emit | 否 | 低 | P2 |
| E1 | restore token + LoRA 学习补偿 | KF4 / Q1-e / 迁移表#4 / Rec.5 | RestoreKV 2608.01247, ResKV 2607.29591, LESS 2402.09398 | restore | 是 | 高 | P2 |
| F1 | 修复源的存储形态（冷层格式） | Q1-f / Q5 | KVReviver 2512.17917 | src_{text,rawcpu,q4,sketch} | 否 | 中 | P2 |
| G1 | full-KV 逐 token 验证 → 造"分歧点"标签（只做诊断） | Q1-g / Q6 | VeriCache 2605.17613 | 离线标注 | 否 | 中 | P1 |
| H1 | `<tool_call>` 约束解码 | TL;DR(2) / KF5 / Q5 / Q7#6 / 迁移表#2 / Rec.1 | XGrammar-2 2601.04426, constraint tax 2606.25605 | cd_*, lenient_parse | 否 | 低中 | P0 |
| H2 | 硬 token 侧通道（参数值恢复） | Q5 后半 / Q3 截断案例 / 空白(6) | 无先例 | scratch | 否 | 低 | P1 |
| I1 | 压缩感知 prompt / 自检 | Q1-i / Q4 | 无 | aware, selfcheck | 否 | 低 | P2 |
| I2 | 延迟压缩（cross-ref F 线，不新增 D 臂） | Q1-i / Q6 末条 | 2608.00902, TRACE 2608.06503 | — | 否 | — | — |
| J1 | gist+raw 混排布局补训 | KF6 / Q1-j / Q7#4 / Rec.5 | SSA 2604.20920, KVLink | mixlayout | 是 | 高 | P2 |
| J2 | 何时修 / 修哪的策略学习 | Q4 / Q6 / 空白(1) | 2608.00902（启发式） | policy | 分类器→bandit | 高 | P2 |
| K1 | 定位策略：k 怎么选 + 追加块数阶梯 | Q3 / KF6 / Q1-k / Rec.6 / 空白(3) | SSA, 2608.10502, CacheBlend | corr_k{median,last,attn,ref,oracle}, top-m | 否 | 低中 | P0 |
| K2 | 修复持久化、跨轮影响、误修伤害 | Q6 / Q8 | Leyline, 2608.10502, CommitKV 2608.07855, VeriCache | persist_{keep,regist,drop}, harm | 否 | 中 | P1 |
| T1 | 触发器 + 闭环 | Q4 / Rec.6 / 空白(1) | 2605.09502, 2608.02464, AgentTether 2607.06273, ERGO 2510.14077 | trig_*, closed-loop | 分类器 | 中 | P1 |
| V1 | 评测规范补齐 | Q8 / Q6 benchmark 段 / 空白(4)(5) | RestoreKV/ResKV 计数法, 2608.10502 指标, ReCache 2608.19662 | 指标列 | 否 | 低 | P0 |

---

## A. 追加式（我们的 corr 所在类）

### A1 errata 机制检验：下游污染 vs 下游变 raw

- **报告出处**：KF1；Q1 表 a 行；迁移表 #6。
- **文献**：Models Take Notes at Prefill: KV Cache Can Be Editable and Composable，2606.17107（报告：已打开原文）。
- **机制与证据【报告】**：prefill 时模型已把"字段条件化的结论"写进下游 aggregator token；字段自身 K/V 只驱动 <1% 决策，所以只改字段自身 KV 会被忽略（no-CoT），带 CoT 的 8B 上编辑字段可恢复决策（1.00，约 1% 计算）；追加一条 salient erratum 或重算受影响 suffix 都能廉价恢复。erratum 是 append-only、与 prefix caching 兼容（98.5% 命中），在线 vLLM 上 p90 TTFT 降 53–398×；edit+compose 与 full recompute 决策一致、延迟最多低 14.9×；可移植 note 拼接与 full recompute 的 logit cosine 0.90–0.999（12 个模型）。
- **对我们的含义**：【报告】corr 就是 erratum；corr_re − corr = +15pp 与"下游 note 已被污染、需要重刷"同向。【推断】但 corr_re 相对 corr 同时改了两件事：加了 R2，并把 G3 G4 换成 raw R3′ R4′（按实测 +305 MB 反推下游约 6 块 raw，比 corr 多了约 258 MB 的信息）。这 +15pp 里多少来自"重新条件化"、多少来自"下游变 raw"，现有臂分不开。
  - **【2026-08-31 裁决】已解决，但答案在别处**：corr@first 直接追平 corr_re（40.86% 相同，存储 1/6.5），所以 corr_re 的"优势"来自 v1 默认选错了块，不是下游表示。errata 叙事在"内容对了、布局是二阶量"的意义上成立。
- **实验设计**
  - 新臂 `re_only`：`S → G0 G1 G2 → R3′ R4′ → Q`。下游重算成 raw，但不加 R2。回答"重算本身贡献多少"。实现上就是 corr_re 去掉 append。
  - 新臂 `corr_regist`：`S → G0 G1 G2 → R2 → G3′ G4′ → Q`。下游按修正后 prefix 重新抽 gist（前提：extractor 支持带 prefix KV 抽 gist）。回答"重新条件化够不够，是否需要 raw 下游"。
  - 判读：corr_re ≫ re_only ≈ corr → 收益来自 R2 + 重条件化，errata 叙事成立；re_only ≈ corr_re → 收益主要来自下游变 raw，corr_re 只是"半个 full"；corr_regist ≈ corr_re → 显存能省回 gist 量级。
  - 顺带 CoT 消融【推断】：现有目标含 Thought；对比强制不出 Thought 直接出 `<tool_call>` 时 corr 的 rescue 是否下降。
  - 指标：D 线两列 + bytes + GPU-s；对照 sham。Kill：re_only 与 corr_re 的差在噪声内 → errata 叙事站不住，改写成"下游表示质量"叙事。
- **优先级** P0；零训练；改动小。

### A2 文本 errata：corr_text

- **报告出处**：Q1 表 i 行（"重注入 raw 文本"，代表工作"待找"）——报告没有找到先例。KF1 里 Models Take Notes 的 erratum 本身是文本 note。
- **【2026-08-31 裁决：判死】** corr_text @median 19.35% vs corr 25.81%；@first 27.96% vs 40.86%。D2 纯文本 erratum S=4/93（活性分母 4/42 = 9.5%）。**修复需要块的 KV 内容，不是一句写对值的便条。**
- **实验设计（历史留档）**
  - `corr_text`：`S → G0 G1 G2 G3 G4 → T2 → Q`，T2 = 第 2 块原文经正常 prefill（条件于 gist prefix），不做 KV 移植、不做 RoPE 搬位。变体：加一句模板"以下为历史第 k 轮原文，供参考："。
  - 对照 corr。请先确认 corr 里 R2 的来源：如果 R2 取自 full battery（带 R0 R1 真实上下文），那 corr 与 corr_text 的差 = "带真实上下文的 raw KV" vs "条件于 gist 的 raw 文本"；如果 R2 是独立编码的，差就只剩 KV 移植 vs 文本 prefill。
  - 【推断】corr_text 在训练分布内（Q 本来就是 gist 后的 raw 文本），线上只需存原文；corr 线上需要 raw KV 冷层（见 F1）。
  - Kill：corr_text ≤ sham + 噪声 → 文本层不可用；corr_text ≈ corr → 直接用文本，省掉 KV 移植整条工程线。
- **优先级** P0；零训练；半天。

---

## B. 替换 / 剪接式

### B1 放置方式 2×2

- **报告出处**：Q3"追加位置与顺序"；Q7 #3（块首 sink）、#4（OOD 布局）；迁移表 #5；空白(2)："末尾追加 vs 原位剪接在对话历史上的直接对比——没有"。
- **文献**：Leyline 2606.01065（已读：span→stub 原位剪接，下游 slot 闭式 δ-rotation 重锚，radix prefix 保留）；KVLink 2502.16002（已读：全局 RoPE 重编码 + 可训练 link token，QA +4%、TTFT ↓96%；独立编码+拼接的 train-test 布局不匹配可致 QA 相对下降 up to 35%）；SSA 2604.20920（核实：SG+SR 53.39 > SR-only 52.76，gist 与 raw 共存略优）；EPIC/LegoLink（二手：拼接后块首 token 变成 attention sink）。Kamera 2606.23581 是我们种子里的 RoPE 搬位精确性证据，报告未展开。
- **【2026-08-31 结果】布局是二阶量**：等存储 47.3MB 下 corr 25.81% / drop_g 24.73% / splice_keep 25.81%（逐位一致）/ splice_rep 23.66%。冻结 witness k\* 后六臂终表最佳为 **raw_erratum_tail 75/93 = 80.6%**（锚在历史末尾+推进位置账本），raw_SGSR（只保 k\* 的 gist、其余全丢）71/93 与 keepG 持平——**丢掉其他块 gist 零代价**。工程上选追加末尾（对 prefix cache 最友好、无重排成本）。
- **实验设计（历史留档）**

  | 臂 | 布局 | 变化 |
  |---|---|---|
  | corr | `S → G0 G1 G2 G3 G4 → R2 → Q` | 末尾追加，保 G2（现状） |
  | corr_dropG | `S → G0 G1 G3 G4 → R2 → Q` | 末尾追加，丢 G2 |
  | splice_keep | `S → G0 G1 G2 R2 G3 G4 → Q` | 原位插入，保 G2 |
  | splice_rep | `S → G0 G1 R2 G3 G4 → Q` | 原位替换（Leyline 式） |

  - 四臂都是一块 raw，bytes 只差一个 gist 块；R2 的 K 按目标位置做 RoPE 旋转（现成 reposition 原语，承重测试已过）。R2 来源固定为与 corr 相同。
  - 可选子臂：R2 前加 p≈4 个 blank token 吸 sink（22 号 B1 的 trick），对应 Q7 #3。
  - 与 A1 合并：splice_rep + G3′G4′ 重抽 = 最省的"原位修复"。
  - 指标：两列 + 重复退化率。Kill：四臂差在噪声内 → 放置无关，保留 append。
  - 功效：预期差只有几 pp，93 条的 MDE 17–25pp 不够 → 依赖批次 0 扩触发集。
- **优先级** P0；零训练；1–2 天。

---

## C. 选择性重算 / 成本

### C1 下游重算改 gist 与等效压缩率算账

- **报告出处**：迁移表 #1（把"重算全部下游 suffix"改便宜）；Q2 末条（ResKV 的 b = m + r 会计约定：补偿 KV 计入压缩率）。
- **算账【推断】**：n 块、每块 L token。full = nL；C2KV = nL/8；corr = nL/8 + L；corr_re ≈ (k+1)L/8 + L + (n−k−1)L。取 n=13、k=6（与 +305 MB ≈ 6.4 块 raw 一致）：corr_re ≈ 7.9L，对 full 13L 只剩 1.65×——corr_re 已经放弃了绝大部分压缩收益。corr_regist ≈ nL/8 + L，与 corr 同量级。
- **实验**：臂同 A1 的 corr_regist；从现在起每张成本表加一列"等效压缩率"（按 ResKV 约定把追加/驻留 raw 全部计入）。**这条仍然有效且强制执行。**

### C2 选择性重算 + 重算预算阶梯

- **报告出处**：TL;DR (1)；KF2；Q2（曲线、非单调）；Q3（粒度：长数字截断）；Q7 #2 #5；迁移表 #1；Rec.2；空白(3)。
- **文献**：CacheBlend 2405.16444（已读：只重算 5–18% 高偏差 token；选择 = 第 1 层全重算 + 第 2 层部分重算，比较 V 偏差取 top-k；TTFT ↓2.2–3.3×、吞吐 ↑2.8–5×，质量损失 ≤0.002 F1/Rouge-L；RULER 30–70% 重算区间反而更差——这条建议看原图）；ProphetKV 2602.02579（已读：query-aware 选择，20% 重算保 96–101%，比 CacheBlend/EPIC/KVShare 高 8.8–24.9% RULER、18.6–50.9% LongBench）；EPIC/LegoLink（二手：只重算每块开头固定数量 token）；CacheClip 2510.10129（22 号记录：20% 重算 → 85.2% NIAH）。
- **迁移的真实约束【推断】**：CacheBlend 假设"独立算好的 raw 块 KV 已在手"，只补 15% 偏差最大的 token。我们下游是 gist，线上没有下游 raw KV。所以这条只在两种前提下成立：(a) 有 raw KV 冷层（CPU/盘，见 F1）；(b) 把"选择性"施加在 corr_re 的 raw prefill 上——那就不省算力。所以 C2 实质是"冷层 raw + 部分重算"路线，不是纯算力优化，成本要连冷层搬运一起记。
  - **【2026-08-31 补记】** SidecarStore（正常压缩时用钩子顺手存一份每层 KV）已落地，前提 (a) 现在成立——打补丁前不再多跑一遍模型。
- **实验设计**
  - 臂 `corr_sel{sel}@r`：`S → G0 G1 G2 → R2 → R3ᵖ R4ᵖ → Q`，R3ᵖ = 独立 raw KV 中 r% 的 token 用修正后 prefix 重算、其余保留。sel ∈ {dev（前两层偏差，CacheBlend）, qaware（用 Q 的 attention 挑，ProphetKV）, boundary（每块前 p 个 token，EPIC）, random（对照）}；r ∈ {5, 10, 20, 50, 100}；r=100 即 corr_re。
  - span 补全规则（来自 CacheBlend 5663623→566362 案例）：被选 token 若落在字母数字串内，整串一起重算。
  - 指标：rescue–r 曲线、GPU-s（含选择开销 + 冷层搬运）、bytes。
  - 判读：dev/qaware 在 r=20 达到 corr_re 的 ≥90% 且显著高于 random → 迁移成立；曲线到 r=100 才起来 → gist 场景下偏差信号无效。若复现"中段更差"的非单调，记为与 CacheBlend 一致的独立证据。
- **优先级** P1；零训练；工程中高。

### C3 成本 crossover 长度扫描 + p95 口径

- **报告出处**：TL;DR 第二条；Q2 crossover 段；Rec.4。
- **文献**：ProphetKV 2602.02579（已核实：20% 重算在 8K/16K 上比 full re-prefill 快至多 5×；4K 上所有方法都没加速，固定系统开销占比大）；SIFT 2606.09441（加速随上下文增长，64K 达 1.71×；CacheBlend 退化 7K→64K 约 20%→56%）。报告确认：没有论文报 p95 修复延迟；异步/关键路径外修复有先例（CacheBlend 把重算与 KV 检索流水线化；2608.00902 利用轮间空闲延迟压缩）。
- **【2026-08-31 结果：已完成】** 60 个上下文 × 3 种长度取中位：2K 时打补丁与从头重算持平（1.00×），**4K 起打补丁严格更便宜（0.83×），交叉点 ~4K**，落在真实基准的会话长度区间（2–8K）内。端到端上打补丁的延迟代价 ≈ +1.1s/请求（+11%）。
- **优先级** P0（已完成）。

---

## D. 加性修正 / steering

### D1 共享 offset 校正（AgentKVShift 式 = 22 号 D-E1 Level-2）

- **报告出处**：TL;DR (3)；KF3；Q3"误差的单位"；Q1 表 d 行；迁移表 #3；Rec.3。
- **文献**：AgentKVShift 2607.21604（已读；注意报告写的日期 2026-05-15 与 ID 月份不符，引用前核）：training-free、probe-guided；per-memory 的 KV 复用残差 = 共享 memory-level offset + 小的 token-wise 扰动；用小探针集估计 offset，对整块所有 token 加权校正；只刷新 10–30% 即达近 full（CacheBlend 类要 45–55%）；2/4-bit 下仍保 >2× F1；prefill 2–3.5×（A100，3B–32B）。
- **它修的是什么【推断】**：复用残差 = "块在错误上下文里算出来"的误差，对应 P4 里独立切块的那一半损失（−6.57），不是 8× 压缩丢掉的信息（另一半 −6.15）。所以它是 corr 的互补，不是替代。
- **实验设计**
  - 残差定义（两种都做最好）：(i) KV 层：δ_K, δ_V = E[gist KV（带前序上下文抽取）− gist KV（独立抽取）]，按层、按 head、K/V 分开估（论文提 K/V 残差不对称）；(ii) 激活层（Cache Steering 式）：δ_ℓ = E[h_full − h_comp]，加在 Q 位置。
  - 估计集：held-out session（与评测 session 不重叠）；探针块数 {8, 32, 128} 看收敛。
  - 臂：`offset_kv`（加到所有 Gi）、`offset_act`、`offset_sham`（同范数随机向量，必须无效）、`offset+corr`（叠加）。
  - 必做分层：压缩率阶梯 4× / 8× / 16×（修复力在哪断）；按块位置与角色（早/晚、observation/assistant）分组估 offset，检验"共享"假设——这就是 22 号 D-d 空白的直接检验，不成立也是可发表的负结果。
  - 评测面：offset 是全局改动，要在 900 条全体上报两列，专门看 C→C 样本是否被打坏（harm）。
  - Kill：offset_kv ≤ offset_sham；或 8× 下增益消失。
  - **【2026-08-31 关联】** D3 codec 的离线 rate-distortion 结果给了一条互证：**K 保真度主导注意力输出、V 可回归**（vector_konly：K 保真 0.018、attn err 0.69 最佳；aatc 每通道标量量化 ~1.6bit/元素毁 V 被淘汰）。估 offset 时应优先保 K。
- **优先级** P1；零训练；2–3 天。

### D2 gist 键的 attention logit bias

- **报告出处**：Q1 表 d 行：SelKV 2607.16213（已读：用 prefill 注意力统计生成 logit bias 纠正合并失衡，零内存成本）。
- **迁移【推断】**：8× 后每块只剩 1/8 的 key，softmax 里 gist 键与 raw 键（S、Q、追加的 R2）的竞争可能失衡。给 gist 键的 attention logit 加常数 b（或乘温度），扫 b ∈ {−2, …, +2}。
- **实验**：`bias_b` 在 none 与 corr 上各扫一遍，报两列 + attention 质量分布。Kill：无单调效应。零内存、几乎零算力；半天。（selkv_bias / selkv_count 的 smoke 已在真机跑通。）
- **优先级** P2（便宜，顺手做）。

### D3 行为 steering（发射先验）

- **报告出处**：Q1 表 d 行：Cache Steering 2507.08799（已读）；Memory Inception 2605.06225（未核实）。22 号 D.2 已标"steering 打进 KV"的 claim 被占，只能当 baseline。
- **迁移【推断】**：早期 history 实验的崩点是"不动手了"（44%→6%）。取模型在"发出 tool call"与"纯文本回复"两类样本上的 KV/激活均值差作为方向向量，一次性加到 gist 块。它修行为先验，不修内容。
- **实验**：`steer_emit` 扫强度；报发射率、协议合法率、工具名正确率（看是不是只是"乱发"）。Kill：发射率升而工具名正确率不升。
- **优先级** P2。

---

## E. 学习式补偿

### E1 restore token + LoRA（RestoreKV 式）

- **报告出处**：KF4；Q1 表 e 行；Q2 成本会计；迁移表 #4；Rec.5；Caveats（基座是 eviction）。
- **文献**：RestoreKV 2608.01247（已读：Qwen3-4B；冻结压缩基座 + 8 个 restore token + LoRA 0.4% 参数 + 自蒸馏；5% 预算把 KVzip 的 RULER-4K 从 38.2 拉到 73.2；60 个预算配对赢 59；一次性构建开销 <0.5%）；ResKV 2607.29591（已读：固定预算拆成精确 main cache + 紧凑 residual cache，b = m + r；LongBench 32/32、RULER 63/64）；LESS 2402.09398（报告未核实原文）。22 号 A.2 / A.4 已有撞车判定与 kill。
- **实验设计**
  - 冻结 base 与 C2KV extractor；每块 gist 后接 r 个 restore token（或所有 gist 后接一个全局 restore 块），其 KV 由 LoRA 化的前向在 gist prefix（可选 + Q）条件下生成。
  - 监督：full-KV teacher 的 logit KL（自蒸馏）+ `<tool_call>` 内 token 加权 CE（工具名、参数 key/value）。
  - 预算会计：8× gist + r restore → 报等效压缩率。对照：C2KV@8 单独；C2KV@16 + restore（bytes 对齐到 8×）；RestoreKV 原方法在 Qwen3-4B + KVzip 上同预算复现。
  - 评测：全 battery 两列 + 触发集 rescue（restore 是全局方法，L1 也会变）。
  - Kill（22 号 A.4）：等 bytes 下 C2KV@16+restore ≤ C2KV@8；或触发集上 restore ≤ corr_re；或同预算打不过 RestoreKV。
  - **【2026-08-31 关联】** D11 KVSculpt（r 个自由 key 对块 raw teacher 优化，Adam + LSE mass 项 + GRKV 闭式 ridge V）已离线跑通，等 r 下胜 naive mean slots——这是 E1 的离线版前置。
- **优先级** P2；需训练窗口；与 J1 合并成一次训练。

---

## F. 重建式 / 修复源的存储形态

### F1 修复源从哪来（冷层格式）

- **报告出处**：Q1 表 f 行：KVReviver 2512.17917（已读摘要：sketch 反解 token，10% 预算达等价精度，仅在 2k 上下文验证）；Q5；22 号 C.1（CacheGen 编码可压 3.5–4.3×）、A.3 ④（4-bit 精确 KV）。
- **为什么绕不开【推断】**：C2、K1、T1 一旦线上化，修复需要 raw 信息，raw 从哪来就是成本的一部分。四种源：`src_text`（存原文，修复时 prefill）、`src_rawcpu`（独立编码的 raw KV 放 CPU，可用 CacheGen 式编码）、`src_q4`（4-bit 量化 raw KV，存 1/4）、`src_sketch`（KVReviver 式）。
- **【2026-08-31 部分裁决】** D3 离线 rate-distortion（真实 sidecar dump，30 blocks）：aatc 淘汰；幸存 kvtc（最省可用）、vector_konly（attn err 0.69 最佳）、raw_q4（DEFLATE 友好 −52%）。src_text 通道已被 D2 判死，所以冷层只能走 KV 形态。
- **优先级** P2。

---

## G. 验证—替换（只做诊断用）

### G1 VeriCache 式逐 token 验证 → 造"分歧点"标签

- **报告出处**：Q1 表 g 行；Q6 误差跨步累积段；22 号 D.2 #1（claim 被 VeriCache 完整占据，且要求 full KV 常驻）。
- **文献**：VeriCache 2605.17613（已读：压缩 KV 起草、CPU 常驻 full KV 单次前向验证，输出 bit-identical，4× 吞吐，7 种压缩器；只作用于 decode）。
- **用法【推断】**：不作为方法（与显存目标冲突、已被占），作为工具：离线用 full KV 对压缩流逐 token 验证，得到每条样本"第一个分歧 token"的位置，再对该位置做各 gist 块的 attention 归因 → 一批"错从哪开始、和哪块有关"的标签，供 T1 训触发器、K1 训定位器、K2 量化跨步传播。
- **【2026-08-31 关联】** D1 的 witness-block 规则（按答案线索的块级 IDF 打分）已经在没有 full-KV 验证的情况下拿到 76.3%，G1 的标签现在是"能否再往上够到 oracle 87.1%"的工具，不是前置。
- **优先级** P1。

---

## H. 解码层

### H1 `<tool_call>` 约束解码

- **报告出处**：TL;DR (2)；KF5；Q5；Q7 #6；迁移表 #2；Rec.1（"立刻挂"）。
- **文献**：XGrammar-2 2601.04426（已读：BFCL-v3 上 Llama-3.2-3B correct-call-rate 33.12→77.75%、schema-rate 40.70→100%；Llama-3.1-8B 59.48/66.95→80.93/100%；3B + 约束优于无约束 70B；一次性编译 20–50 ms，cached grammar 开销 <3%；边界：只保格式，BFCL-Live 仍有"schema 过、值错"）；constraint tax 2606.25605（已读：结构化约束会抑制部分开源模型发起 tool call）。
- **为什么现在更重要**：29 号周报的 BFCL 取证发现——full 臂也只有 3.0%（6/200），118 例解码失败；模型在压缩历史下约 **45% 的步骤直接输出散文而非工具调用**。这正是 H1 要解决的协议地板。官方 scorer 口径 6/82 = 7.3%（118 例被丢弃不进分母）。
- **实验设计**
  - 在所有现有臂上加解码约束：`cd_none, cd_sham, cd_corr, cd_corr_re, cd_full, cd_corr_all`。约束 = structural tag：`<tool_call>` 外自由文本，内部按当前工具池的 JSON schema（各工具 schema 的 union，name 字段限定在工具池内）；把 `</tool_call>` 与 EOS 纳入语法。
  - 对照 `lenient_parse`：不改解码，只把评分解析器换成宽松 JSON 修复。它回答"非法里多少是琐碎语法"，即约束解码真正多买到了什么。
  - 指标：两列全报 + 发射率（constraint tax 监控）+ 重复退化率 + 输出长度 + 每 token 解码开销。严格修复率的新上界 = cd_full。
  - 判读：cd_full 抬到 ≥90% 且 cd_corr_re 同步抬升 → 协议瓶颈解决；工具名正确率或 arg F1 下降、或发射率下降超噪声 → constraint tax 成立，改用软版本（只在检测到非法时二次约束解码）。
  - 工程：HF eager 下用 xgrammar / Outlines / llguidance 的 logits processor（CPU 侧 bitmask），NPU 上确认 mask 搬运开销；SGLang 栈已内置 xgrammar。
- **优先级** P0；零训练；1–2 天。**注：批次 1 里唯一尚未出数的 P0。**

### H2 硬 token 侧通道（参数值恢复）

- **报告出处**：Q5 后半（"没有一篇做 KV 层 verbatim 保真 + 参数 EM 收益"）；Q3 的长数字截断案例；空白(6)；07 号文档 D3。
- **实验**：`scratch`：正则 / NER 从各块原文抽 ID、路径、数字、实体、工具名，整理成紧凑文本块（结构化 scratchpad）接在 gist 后，`S → G0 G1 G2 G3 G4 → SCR → Q`；bytes 远小于一块 raw，且是分布内的 raw 文本。对照：sham（等长无关文本）、corr_text。指标重点：arg value F1、严格动作正确。Kill：arg F1 不升。
- **【2026-08-31 风险上调】** D2 判死（纯文本 erratum 只有 9.5% 活性）直接威胁 H2 的前提——如果"写对值的便条"救不回来，结构化 scratchpad 大概率同理。但两者不完全同构：D2 的 erratum 只有 ≈62 token 的 witness 字面值，H2 是全轨迹的硬 token 汇总。建议降到 P2，或先做一个 20 题的探针再决定。
- **优先级** P1 → **建议 P2**。

---

## I. 文本层

### I1 压缩感知 prompt / 自检

- **报告出处**：Q1 表 i 行（无代表工作）；Q4（self-consistency 触发候选）。
- **实验**：`aware`：system prompt 加一句"历史为压缩记忆，不确定时优先依据工具定义与最近轮次"；`selfcheck`：生成后让模型在同一压缩上下文里判断"该调用是否与历史一致"，作为触发信号之一进 T1。都在全 battery 上跑；预期小；作为 baseline 记录。
- **优先级** P2。

### I2 延迟压缩（cross-ref F 线，不新增 D 臂）

- **报告出处**：Q1 表 i 行、Q6 末条：Practical Online KV Cache Compaction for LLM Agents 2608.00902（已读：用未来 query 延迟压缩，恢复大部分差距；BrowseComp / WideSearch / AppWorld）；TRACE 2608.06503（文本压缩，AppWorld 上报 blocked actions / repeated exploration，作者自称 preliminary）。
- **对 D 的用处【推断】**：F 线 selector 缺信号，2608.00902 的"用下一步 query 决定压不压"是一个可部署信号；TRACE 的 blocked-action 指标可加进我们的语义列。不新增 D 臂。

---

## J. 训练 / 策略

### J1 gist+raw 混排布局补训

- **报告出处**：KF6；Q1 表 j 行；Q7 #4；Rec.5 后半。
- **文献**：SSA 2604.20920（核实：q·k_gist 打分选 top-k 块，选中块 gist + raw 同时进注意力；Qwen2-7B LongBench 8× 46.20 vs full 47.78，16× 45.39，32× 44.07；消融 SG+SR 53.39 > SR-only 52.76）；KVLink 2502.16002（link token；无训练时 OOD 布局掉 up to 35%）。
- **实验**：只训 gist 投影（base 冻结），训练样本随机采样布局：全 gist、`G…G R_k`（errata）、`G R G G`（原位）、`G G G R_last`（defer）。训后重跑 corr / splice / corr_text，以及 B 线 P-delay。Kill：rescue 不升且 battery 不降 → 布局不是问题。
- **【2026-08-31 降权】** B1 已实测布局是二阶量（各臂差 ≤2.2pp），J1 的预期收益随之下调。若还做，理由应改为"让 erratum_tail 布局进训练分布"而非"修 OOD 惩罚"。
- **优先级** P2 → **建议 P3**；与 E1 合并进同一训练窗口。

### J2 何时修 / 修哪的策略学习

- **报告出处**：Q4 闭环空白；Q6 RL 段；空白(1)。
- **实验【推断】**：先监督后 RL。特征 = T1 的触发信号 + K1 的定位信号 + 上下文长度；动作 = {不修, corr_k, corr_re_k, corr_text_k}；标签来自 full-vs-C2KV 配对（扩到 ~3000 qid）；奖励 = 严格正确 − λ·GPU-s。先训分类器报离线净覆盖，再考虑 bandit。
- **【2026-08-31 升权】** K1 的 witness 规则给了一个**无需学习**的强定位器（76.3%，oracle 87.1%），J2 的问题从"修哪"收窄成"何时修"——即 T1。定位那一半基本解决。
- **优先级** P2。

---

## K. 定位与持久化

### K1 修哪一块（k 的选择）+ 追加块数阶梯

- **报告出处**：Q3 定位信号段；KF6；Q1 表 k 行（2608.10502 依赖图定位）；Rec.6 后半；空白(3)。
- **文献**：SSA（用 gist key 与 query 打分选块——正是交接报告 §5.6"att-on-compressed ≈ att-on-full，压缩表示够路由"的文献版）；Dependency-Guided Rollback Repair 2608.10502（依赖图定位污染源后选择性重放，文本层）；CacheBlend（偏差信号）。
- **【2026-08-31 结果：本手册最重要的一条，已完成并改写整条 D 线】**

  | 修哪块 | 救回（n=93） |
  |---|---:|
  | **witness 块（按答案线索 IDF 打分，新规则）** | **71/93 = 76.3%** |
  | 正中间块 median（v1 默认） | 33/93 = 35.5% |
  | 最近块 last | 26.88% @median-caliber（与 median 无差别） |
  | 第一块 first | 40.86% |
  | 乱修一块（823 trials 地板） | 25.0% |
  | best-k 天花板（oracle 包络） | 81/93 = 87.1%（随机包络期望 78.9%） |

  R 触发门（prereg v2.8）**PASS**：单边二项 P(X≥71 | p=0.2503, n=93) = 4.7e-25。
  witness==median 仅 19/90 → **定量证实 v1 D 线大部分时间修错了块**，这是 v1 结果弱的候选解释。
  flip 集中度：42/81 翻转题恰好只有一个 k 翻转。
  混杂待拆：第一块比正中间块平均长 29%（413.4 vs 320.5 token），字节未控平。
  机制注记：文献的"保住开头"（attention sink，StreamingLLM 2309.17453；H2O；SnapKV）指的是**序列最前的 token**（约等于我们未压缩的 S 区），K1 的 first 是**第一块历史**——这是语义级首位效应（任务设定与最早观察由首块携带），层级不同，写作时不要混。
- **仍待做**：top-m 阶梯（m ∈ {1,2,4,all}）的预算—响应曲线；corr_all 已是 m=all 的点（44.1%，协议合法率掉到 46%），是"过量恢复有害"的现成数据点。
- **优先级** P0 → 主体完成，余 top-m 阶梯。

### K2 修复持久化、跨轮影响、误修伤害

- **报告出处**：Q6 前两条；Q8；22 号 D-a。
- **文献**：Leyline（(span, replacement) 原位剪接语义，最接近 commit）；2608.10502；CommitKV 2608.07855（报告判定：存在，但是按 commit 生命周期退休页的压缩方法，不是修复）；VeriCache。
- **实验设计**
  - 在触发样本所在 session 内继续 teacher-forced 往后跑 m 步（93 个触发来自 72 个 session）：`persist_keep`（保留 R_k）、`persist_regist`（用后把 R_k 重压成带上下文的 G_k′）、`persist_drop`（用完即弃）。报 t+1…t+m 的两列、recurrence、驻留 bytes 随步增长。runbook §4a 有完整跑法。
  - `harm` 臂：把 corr / offset / cd 施加在 C→C 样本上，报打坏率——闭环里误触发的代价就是它，T1 的净覆盖公式要用。
- **【2026-08-31 升权】** 端到端已经在跑任务级触发的修复臂（τ² 全压缩 9 触发救回 3、hybrid-3 轮 4 触发救回 3；ToolSandbox 1 个受损场景差距恢复 93%），harm 率是把这些数字变成线上可信数字的唯一缺口。
- **优先级** P1 → **建议 P0**。

---

## T. 触发器

### T1 线上判错信号 + 闭环

- **报告出处**：Q4 全部；Rec.6；空白(1)（"KV 层把触发器 + 定位器 + 修复器闭环并报端到端数字的工作——没有"）。
- **文献**：2605.09502（已读："The Signal Is Diagnostic, Not Causal"：探针 0.95 AUROC 读得出 CoT 错，首步 0.79，跨 Qwen / Llama / Phi 1.5B–72B；activation steering / probe-guided BoN / self-correction / activation patching 四种干预全部失败）；2608.02464（已读摘要：ESN+CUSUM 遥测，2823 条 agent 轨迹，5% 误报预算下检出 0.71、AUROC 0.872，每步微秒级，零重训迁移到 AFTraj-2K 0.745 / ATBench 0.779）；AgentTether 2607.06273、2608.10502（文本 / action 层闭环）；ERGO 2510.14077（熵触发 reset，未核实）。
- **实验设计**
  - 数据：扩到 ~3000 配对 qid；标签 C→W（正）vs 其余；按 session 分组 CV（14 号 R8 加固版探针协议：nested CV、pooling / 层 / 头选择限 inner fold、双 floor）。
  - 特征（只用压缩流）：解析失败；生成动作的 max / mean 熵；工具名 token 的 logit margin；attention 到 gist / S / Q 的质量比；隐状态线性探针；CUSUM 跨步统计；I1 的 selfcheck。
  - 报：5% 与 10% 误报下的召回、AUROC、每步开销；baseline = 只用解析失败。
  - 闭环：trig → corr@witness；净覆盖 = TPR·L1·L2 − FPR·harm（harm 由 K2 给）。
  - Kill：没有信号超过"只用解析失败"；或闭环净覆盖 < oracle 触发的一半。2605.09502 必须写进设计：触发器 AUROC 高不保证修得回来，触发与修复分别端到端报。
- **【2026-08-31 升权，现在是主要缺口】** 定位已解决（K1 witness 76.3%），修复通道已证实（80.6%），**闭环里唯一还用 oracle 的就是触发**。所有端到端数字（τ²/ToolSandbox 的修复臂、推导准确率 ≈0.046 / ≈0.093）都标着"按 oracle 触发的上限估计"。
- **优先级** P1 → **P0，本线第一优先**。

---

## V. 评测规范

### V1 指标与对照补齐

- **报告出处**：Q8；Q6 benchmark 段；空白(4)(5)。
- **现状【报告】**：修复论文多用"相对 full 的保留率"或"配对预算配置计数"（RestoreKV 59/60，ResKV 32/32 + 63/64）；2608.10502 用 recovery / recurrence / claim-invalidation F1；没有找到 sham / 等 bytes / 等 GPU-time 对照；基本不报配对 CI / MDE；触发式（只评 full 对 C2KV 错的子集）设计没有对应；多轮 agent KV 压缩论文几乎不报协议合法率——例外是 ReCache 2608.19662（Inv-F1 82.3% vs dense 82.4%）和 TRACE（blocked actions，非 KV 层）。AgentKVShift、CommitKV、IntentKV 2606.09916 都只报 task 分数。
- **采纳**：保留 sham、oracle 触发集、session 聚类 bootstrap；新增 harm rate、recurrence（K2）、等 GPU-time 对照、预算阶梯的配对计数报法；语义列加 Inv-F1；benchmark 扩到 BFCL v3 multi-turn + AppWorld 复现。
- **写作口径**：sham 与 oracle 触发集"未找到先例"是零命中结论，写"据我们检索"，不写"不存在"。
- **【2026-08-31 补记】** BFCL 双口径已确立：6/200 = 3.0%（生成分母）与 6/82 = 7.3%（官方 scorer 分母，118 例解码失败被丢弃）。同一事实两种分母，两个都报。
- **优先级** P0；零成本。

---

## 批次建议（原始，供对照）

- **批次 0（前提）**：配对 qid 从 900 扩到 ~3000（触发集 ~300，MDE 从 17–25pp 降到 10–14pp）；固定并写明 R_k 的来源口径；实现原位插入原语；接通 serving kernel 的长上下文路径；为评测集缓存独立编码 raw KV。→ **原位插入原语与 sidecar 捕获已完成；扩样本仍未做，是当前所有几-pp 结论的功效瓶颈。**
- **批次 1（零训练）**：H1、A1、A2、B1、K1、C3、H2。→ **除 H1 外全部完成**（见上方各条裁决）。
- **批次 2（诊断臂）**：D1（含 sham / 压缩率阶梯 / 分组）、C2 阶梯、K2（persist / harm）、D2 / D3 顺手、F1、G1 造标签、T1 离线特征评估。→ **当前批次**；建议顺序改为 T1 → K2(harm) → D1 → C2/F1。
- **批次 3（训练）**：J1 + E1 合并一次训练；T1 闭环上线评估；J2。

## 报告的六条空白 → 我们能占的位置

| 报告空白 | 对应臂 | 状态 |
|---|---|---|
| (1) KV 层触发器 + 定位器 + 修复器闭环端到端数字 | T1 + K1 + corr@witness | 定位与修复已就位，**触发仍是 oracle**——这就是 novelty 位，也是唯一缺口 |
| (2) 末尾追加 vs 原位剪接在对话历史上的直接对比 | B1 | **已占**（布局是二阶量；erratum_tail 最佳） |
| (3) 追加 raw 块数的预算—响应曲线 + 过量恢复有害 | K1 top-m 阶梯 | 部分（corr_all 是 m=all 的点，曲线未画） |
| (4) KV 修复论文的 sham / 等 bytes / 等 GPU-time 对照 | 已有 sham + V1 | 已占，写"据我们检索未见" |
| (5) 多轮 agent KV 压缩报协议合法率 / 发射率 | 双列评测本身 | 已占；BFCL 45% 输出散文是强证据 |
| (6) 硬 token verbatim 保真 + 参数 EM 收益 | H2 scratch | **风险上调**（D2 判死同源） |

## 引用前必须核实（报告自己标的 + 我看到的疑点）

- **未打开原文**：DeltaBox 2605.22781、LESS 2402.09398、Memory Inception 2605.06225、REFLECT 2606.09071、ERGO 2510.14077、2505.00212（二分归因）、2607.28495（KV 移植因果）、KVReviver 的多轮场景。
- **二手核实**：EPIC / LegoLink（经 ProphetKV / QCFuse 转述 + ICML slides）。
- **数字未逐格核**：ProphetKV Table 2 的 TTFT 秒数。
- **疑点**：AgentKVShift 日期 2026-05-15 与 ID 2607 不符；CacheBlend"中段重算更差"建议看原图；XGrammar-2 的 BFCL 数字、ReCache 2608.19662、CommitKV 2608.07855、IntentKV 2606.09916 都是新 ID。
- **报告判定**：REPAIRKV、"Cache You Later: Post-Compression KV Repair" 未找到，很可能不存在；CommitKV 存在但是压缩不是修复；C²KV 2607.17715 报告当作外部论文，应是楚恒的原论文；两个同名 TRACE（2608.06503 压缩 / 2606.00611 安全）。
- **基座警告**：除 SSA（learned gist）和 AgentKVShift（agent 结构化记忆）外，所有修复方法的基座是 eviction 或拼接复用，误差结构与 8× gist 不同——所有外推收益都要在触发集上重测后才能当预期写。
