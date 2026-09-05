# G 实验（true-joint 联合压缩训练）进度汇报 — 2026-08-20

分支：`task/g-joint-c2kv`（基于 `task/r5-metrology`）。算力：NPU 服务器（8×910B3）。

## 1. 实验要回答的问题

- **G-Q1**：以前是不是单纯训得太少？（token 预算制：small=0.0625P / medium=0.25P / large=1.0P，P=P_official 实测中）
- **G-Q2**：多数据集（20%QA + 50%traces + 25%Toucan + 5%Open-SWE）是否优于 traces 单源？
- **G-Q3**：联合训练本身有没有用？三臂对比（同一数据、同一 32M source-token 预算、同一冻结样本顺序、ratio=8、LR=5e-5）：
  - **J-separate**：tool / history 两个 extractor 分别训练，推理时拼接两套 KV（当前做法的公平基线）
  - **J-alternate**：一个 shared extractor，tool-only / history-only batch 交替
  - **J-joint**：一个 shared extractor，同一 forward 同时压缩 tool+history（真正的联合监督）

判读规则：J-joint > J-alternate 才说明"联合监督"有独立价值；J-alternate > J-separate 说明参数共享有正迁移。

## 2. 关键前置修复（已实机验证）

发现 transformers 5.8 的 mask 注册表缺 `npu_fusion_attention` → `create_causal_mask` 返回 None → teacher-forced 前向变成**双向注意力（标签泄漏）**，此前所有 NPU 训练 loss/eval_loss 均偏假（假 loss≈0.038，真实因果 NLL≈3.3–6.3）。已修复（commit `2d8313b`，`python/models/npu_attention.py` 注册 eager mask），并实测验证 fusion NLL 与 eager 位级一致。

**影响**：仓库里既有的 `checkpoint-2678`（history）与 `checkpoint-250`（tooldef）极可能同病，其历史 loss 口径不可信，建议后续审计重测。

## 3. 数据侧结论

- 训练池定为 **agent-llm-traces-v2**（10,056 session）：dedup 证实 v1 内容几乎全被 v2 包含（165 万条精确重复）。
- 切分：task-proxy 组级 disjoint（归一化首条 user instruction 的 sha1 分组），v2 共 2,680 组；训练顺序文件冻结 23,652 qid。
- v2 × BFCL 精确重复仅 14 条 → 对应 session 将从训练侧剔除（待做，不影响 small 阶段结论的相对比较）。
- traces-v2 数据卡未声明 license（已标记）；Toucan Apache-2.0、Open-SWE CC-BY-4.0 已就位，供 medium 的 D-multi 臂使用。
- **P_official 扫描仍在跑**（单进程 tokenizer 全扫，已 >40h）：落地前 small 预算用默认 32M（≈0.0625×512M 的假设值）。

## 4. Gate-1（LR 校准，8M source tokens，v1 池，joint 条件 @8×，128 例）

| LR | c2kv tool_name_acc | c2kv textF1 | full tool_name_acc（基线） |
|---|---|---|---|
| 5e-7 | 0.000 | 0.030 | 0.338 |
| 5e-6 | 0.000 | 0.052 | 0.312 |
| **5e-5** | **0.342** | **0.445** | 0.304 |

结论：5e-7/5e-6 在 8M 预算下 c2kv 完全没学起来；5e-5 追平 full 基线（textF1 反超）→ 选定 **5e-5**。注意 5e-5 是网格上沿，medium 阶段考虑补测 1e-4。

## 5. Small 四臂训练状态（各 32M source tokens，1 epoch，GRAD_ACCUM=4）

| 臂 | 步数 | 状态 | 训练时长 | final train_loss |
|---|---|---|---|---|
| joint | 731 | ✅ 完成 | 24.2h | 0.539 |
| sep_tool | 723 | ✅ 完成 | 24.7h | 0.885 |
| sep_hist | 684 | ✅ 完成 | 15.1h | 0.503 |
| alternate | 1406（交错双倍） | 🏃 58%（815/1406） | 预计 08-21 ~20:40（服务器时间）完成 | — |

公平性核查：三臂 `achieved_source_tokens` 均为 32,008,539、同一批 2,991 个训练样本、同一冻结顺序。

⚠️ **数据混合警示**：small 臂前 2,991 个样本全部来自 appworld 子集（`train_subset_counts={'appworld': 2991}`）——冻结顺序文件头部被 appworld 占满，small 预算实际只见到单一子集。对四臂**相对比较**公平（大家见到的数据完全相同），但对"联合训练在混合分布上的效果"这一外推需保留。medium 阶段应让顺序文件跨子集交错。

## 6. Gate-2 中间结果（sep_hist 臂，eval split @8×，128 例）

该臂只训 history 压缩，用于 J-separate 基线的一半：

| 条件 | num_valid | tool_name_acc | arg_value_f1 | textF1 |
|---|---|---|---|---|
| history_only（本行） | 115 | **0.470** | 0.097 | 0.409 |
| joint | 128 | 0.484 | 0.113 | 0.411 |
| tool_only | 122 | 0.107 | 0.045 | 0.260 |

共同 qid 子集（109 例）：history_only 0.4495 = joint 0.4495；tool_only 0.110。

解读：单训 history 的 extractor 在 history/joint 条件扎实（且明显优于 Gate-1 的 8M 预算 0.342，说明 32M 预算有效），tool_only 条件近乎失效——符合设计预期，它不是全能模型。

## 7. 正在运行的 Gate-2 评测电池（5 路并行，eval split @8×，128 例）

- joint 三条件、sep_tool 三条件
- J-separate 双 extractor 拼接评测 ×2（generator=tool / generator=history 两个变体互相校验）

预计 08-21 凌晨（服务器时间）出齐；alternate 训练完成后补其三条件评测，随后出**四臂完整对比表 + extend 决策**。

## 8. 下一步

1. Gate-2 判读（预计 08-21 晚）：joint vs alternate vs separate 三条件对比；若 joint≤alternate≤separate，则 medium 复验一次后仍如此即不 extend large。
2. medium（0.25P）：D-single vs D-multi 对照 + medium-large-pool vs medium-small-repeat 对照；多数据集 loader（Toucan/Open-SWE → joint source）待写。
3. large：G8 × 3 seeds + Gmulti（ratios {4,8,16}）；先比 Gmulti@8 vs G8@8。
4. 冻结评测：BFCL（runner 已就绪）+ ToolSandbox（评估集成成本）。
5. 清理中间 checkpoint → commit → PR。

原始产物（summary json / 日志 / manifest）在 NPU 服务器 `~/c2kv/outputs_lyc/g_joint/`，不入库。

---

# 2026-08-21 追记：capfix 接入 + Gate-2 一判

## 9. cap-fix（commit `9a1dffc`，刘言成 patch，已合入本分支）

发现并实现：旧 builder 的 budget 截断是 shuffle 后纯头部截断 → 饱和样本里目标工具的 schema 约有一半被丢掉；且单边 doc_mode（tool_only/history_only）独占全部 24 个槽位 → J-separate 基线实际呈现预算是 joint 的 **1.691×**（实测，`docs/g_joint/psrc_as_trained.json`）。修复后默认 per-side caps（tool≤16 / history 恒 8、不回收空槽）+ target-doc 保留截断，`LEGACY_MODE_CAPS=true` 可逐位复现旧行为；fixed regime 实测 `sep_combined_over_joint=0.994 ≈ 1.0`。真实 eval 数据 target 覆盖率：fixed **100%**（2015/2015），legacy joint 条件只剩 **74.5%**（25.5% 的目标 schema 被截掉）。

全套 pytest 服务器 204 绿（含 torch/metrology；2 处 bfcl 测试为环境符号链接问题已接，1 处 patch 自带测试少字段已修 `617b2600`）。

**口径警示**：名义预算（估算器）≠ 呈现预算。名义 32M 实际呈现 joint 12.9M。P_official 实测 **535.5M**（`docs/g_joint/official_tokens.json`，假设 512M 偏差 +4.6%）。medium 目标 0.25P=134M 呈现值：v2 训练池单 epoch 封顶 ~100M（23,652 qid × ~4.2k tok），**必须多数据集池或 ~1.3 epoch**。

## 10. Gate-2 一判：buggy 四臂 × fixed-eval（joint 条件，common-qid n=108，c2kv@8×，appworld-only，单 seed）

**严格全交集版（6 路 common qid n=128，joint 条件）**——这才是判读口径：

| 臂 | tool_acc | argF1 | textF1 |
|---|---|---|---|
| J-alternate（buggy） | **0.539** | 0.154 | 0.418 |
| J-joint（buggy） | 0.531 | 0.139 | 0.431 |
| J-separate（genTool / genHist） | 0.508 / 0.500 | 0.102 / 0.106 | 0.354 / 0.432 |
| J-sep_hist（半边） | 0.492 | 0.105 | 0.419 |
| J-sep_tool（半边） | 0.422 | 0.091 | 0.344 |

**判读（预注册规则）**：joint 0.531 vs alternate 0.539，Δ=−0.8pp，|Δ|<3pp → **平 → `inconclusive`**，等 fixed 四臂表终判（不许下"无用"结论）。严格单调触发式 joint≤alternate≤separate 不成立（separate 两变体均低于 joint/alternate）。
附带观察：separate 带着 1.691× 预算优势仍输 joint/alternate 约 3pp → "共享参数 > 双 extractor 拼接"（J-alternate > J-separate 一腿）有弱-中等迹象；history 半边（0.492）显著强于 tool 半边（0.422），下一动作预测的主要信号来自历史。

同 checkpoint buggy-eval vs fixed-eval（joint 条件）：joint 0.486→0.500，alternate 0.440→0.491，sep_tool 0.358→0.380，sep_hist 0.450→0.454——fixed eval 全面微升（alternate +5.1pp 最大），量化了 target 截断的伤害。

首个 fixed 训练臂预览（gate3_fixed_sep_hist，fixed-eval）：history_only 0.380 / joint 0.417 / tool_only 0.139——比 buggy sep_hist（fixed-eval 0.504/0.492/0.131）低，符合预期：fixed 后 sep 臂预算公平化（P_src 5.8M vs legacy 11.7M），单半边变弱是公平的代价，终判看 fixed 四臂整表。

## 11. 当前在跑（08-21 深夜）

- fixed 四臂重训（32M 名义，fixed 语义，同 order file）：sep_hist ✅、sep_tool ✅、joint 95%（~1h）、alternate 78%（~6h）
- 评测：gate2fix_separate ×2（重跑）、gate3_fixed_sep_tool、gate3_fixed_sep_hist ✅、其余 gate3 随训完随挂
- 决策点：08-23 中午冻结 `G8-small-v2`（joint-fixed + 评测通过 + psrc≈1.0 ✓已具备）；08-24 晚 fixed 四臂终表 → Gate-3 终判

---

# 2026-08-22 追记二：fixed 三臂先行表 + G8-small-v2 冻结

## 12. fixed 训练臂先行表（fixed ckpt × fixed-eval，三条件，common n=108，c2kv@8×）

| 臂 | tool_only | history_only | joint |
|---|---|---|---|
| fixed_joint | 0.343 | 0.407 | **0.500** |
| fixed_sep_tool | **0.481** | 0.333 | 0.454 |
| fixed_sep_hist | 0.139 | 0.380 | 0.417 |

对照 buggy（fixed-eval）：fixed_sep_tool 成为真正的 tool 专家（tool_only 0.352→**0.481**，+12.9pp，target 保留的直接效果）；fixed_sep_hist 弱化（0.504→0.380，预算公平化 11.7M→5.8M 的代价）；fixed_joint 与 buggy-joint 持平（joint 条件 0.500=0.500），三条件最均衡。

## 13. ✅ 冻结 `G8-small-v2` = `fixed_joint`（08-22 07:37 UTC+1，机制-pilot 口径）

预注册冻结条件全部满足：

- fixed_joint 评测通过：joint 条件 c2kv 0.500，与 buggy-joint 持平（无超噪声退化）；
- fixed regime `sep_combined_over_joint = 0.994 ≈ 1.0`（实测，`docs/g_joint/psrc_fixed_preview.json`）；
- 全套 pytest 204 绿；target 覆盖率 fixed 100%。

定位：B/C/D 机制 pilot 的**临时供货件**，非正式 G8（正式 = G8-medium，medium 阶段产出）。注意口径：32M 名义 ≈ 12.6M 呈现 P_src，appworld-only，单 seed。

## 14. 在跑（08-22 早晨）

- gate3_fixed_alternate（卡 5，fixed_alternate 已训完 ✅）→ fixed 四臂终表今天补齐（比预计提前一天）
- gate3_fixed_separate_genTool/genHist（卡 6/7，fixed ckpt 的 separate 行）
- lrcal3_1e-4（卡 0，8M LR 上沿复核，~6h）
- Gate-3 终判：四臂+separate 齐后按预注册规则给 extend 建议

---

# 2026-08-22 中午：Gate-3 终判（fixed 四臂，separate 行待补）

## 15. fixed 四臂严格共同 qid 表（n=128，joint 条件，c2kv@8×，regime=fixed 训练+fixed 评测，appworld-only，单 seed）

| 臂 | tool_acc | argF1 | textF1 |
|---|---|---|---|
| J-alternate | **0.578** | 0.124 | 0.415 |
| J-joint | 0.555 | 0.126 | **0.432** |
| J-sep_tool（半边） | 0.508 | 0.098 | 0.341 |
| J-sep_hist（半边） | 0.461 | 0.094 | 0.395 |
| J-separate（拼接，fixed） | 0.523（genTool）/ 0.484（genHist） | 0.099 / 0.098[^argf1] | 0.361 / 0.419 |

[^argf1]: argF1 列口径为 argument_value_f1（与其余四行一致）。本行初版误填 argument_name_f1（0.172/0.187）；同口径值见 docs/g_joint/gate3_fixed_separate_genTool.summary.json（argument_value_f1=0.09895859）与 gate3_fixed_separate_genHist.summary.json（0.09765625）。

三条件单项核查（无牺牲检查）：fixed_joint 三条件 0.343/0.407/0.500 均衡；fixed_sep_tool tool_only 0.481 最强 tool 专家；fixed_sep_hist 偏弱（公平预算代价）。

## 16. Gate-3 判读（预注册）

- joint − alternate = **−2.3pp**（buggy 表 −0.8pp），两个 regime 同向但都在 ±3pp 噪声地板内 → 严格单调式 joint≤alternate≤separate 的判定取决于 pending 的 separate 行；当前状态 **`inconclusive`（偏向"joint 无 small 尺度溢价"）**。
- 两个 regime 一致的方向性：alternate ≥ joint > separate —— "共享参数 > 双 extractor 拼接"成立迹象稳（separate 即使在 buggy 表带 1.691× 预算优势也输）；"同时拼接监督 > 交替监督"未显现。
- **按预注册走 medium 复验**：medium 加一个 joint-vs-alternate 验证臂（0.25P 预算下放大赛道差异）；**不直接 extend large**。G-Q3 在 small 尺度下：联合监督未见独立增益，参数共享有正迁移。
- small 尺度结论边界：appworld-only、单 seed、n=128、8× 单比例——medium 的多子集交错 + 多数据集池会改变数据分布，复验有实质信息量。

## 17. medium 计划确认（~08-25 启动）

- 预算口径：0.25 × P_official(535.5M) = **133.9M 呈现 P_src**。v2 单 epoch 封顶 ~100M 呈现 → D-single 需 ~1.3 epoch；D-multi 池（traces+Toucan+Open-SWE）天然覆盖。
- 臂：D-single、D-multi（alternate 训练方式——当前胜者，固定 8×）+ joint-vs-alternate 复验臂 + medium-small-repeat vs medium-large-pool 对照（G-Q1）。
- 前置：multi-dataset loader、跨子集交错 order、BFCL removal 接线（Phase 4，进行中）。
- LR：5e-5 为主，1e-4 复核在评（lrcal3_1e-4 训完，评测中）。



---

# 2026-08-22 晚：Gate-3 收口 + medium 预算重标定（降级，预注册时间公式）+ license 回退触发

## 18. Gate-3 终态（separate 行补齐，判读落定）

- separate fixed 行补齐（卡 1/2 重跑成功；卡 6/7 的 segfault/aicpu timeout 标 infra-flaky）：**genTool 0.523 / genHist 0.484**。
- 终序：alternate 0.578 > joint 0.555 > separate(genTool) 0.523 > sep_tool 0.508 > separate(genHist) 0.484 > sep_hist 0.461。
- 判读：严格式 joint≤alternate≤separate **不成立**（separate < joint）→ 非 dominated；joint − separate = **+3.1pp（4/128；未取整 0.5546875−0.5234375，初版"+3.2pp"是四舍五入后相减的口径误差）** 恰在噪声地板上沿，"共享参数 > 双 extractor 拼接"方向两个 regime 一致；"同时拼接监督 > 交替监督"未显现。**维持 inconclusive（偏"small 尺度无 joint 溢价"），medium 复验臂确认，不直接 extend large。**
- LR 复核收口：1e-4 只有 **0.266**（c2kv@8×，n=128，docs/g_joint/lrcal3_1e-4_eval.summary.json 背书），明显差于 5e-5 的 0.342（注：5e-5 对照为 Gate-1 产物，未入库，无 raw summary 背书）→ **medium 锁定 LR=5e-5**。

## 19. P_src 实测（fixed 四臂，measure_arm_psrc，docs/g_joint/psrc_fixed_arms.json）

- joint：P_src=12.62M，T_tgt=612K，22.7h → **0.556M 呈现/h/卡**。
- alternate：P_src=12.54M，T_tgt=1.21M（双侧监督 2×），~50h → **0.251M 呈现/h/卡**（alternate 每呈现 token 贵 2.2×）。
- **alternate_over_joint = 0.994**（fixed regime 公平性实测确认，capfix 达成设计目标）。

## 20. traces-v2 license：确认无声明 → 预注册回退触发

- HF cardData 实测（hf-mirror API，2026-08-22）：v2 无 license 字段、无 license tag；v1 = CDLA-permissive-2.0。按任务书：**medium 的 traces 家族改用 v1（1,781 sessions），Toucan 扩规模**。正式 G8 发布前仍需 v2 license 澄清。
- 影响：v1 池按 session 数估计 ~45M estimated tokens（planner 实测后更新），小于 D-single 的 80M 配额 → D-single 需 epochs≈2 补满（planner 报告实际值）；v1⊂v2 已证，split manifest（taskproxy_disjoint_v2）的 train/eval 划分对 v1 依然有效（eval 侧零泄漏由构造保证：只取 manifest train_ids ∩ v1）。

## 21. medium 预算降级（预注册时间公式，记录式降级，不静默超期）

- 实测吞吐下 0.25P（133.9M 呈现）：joint ≈ 10 天/卡、**alternate ≈ 22 天/卡** → 远超 08-29±1 交付窗口。
- **降级为每臂 100M estimated ≈ 39M 呈现 ≈ 0.073P**（estimator 口径，与 trainer `--max_source_tokens` 一致）：alternate ~6.5 天（08-23 早启动 → 08-29 晚完成），joint ~2.9 天。
- 臂（全部 fixed regime、8×、LR=5e-5、同一 order/清单机制，交错按 token-deficit 加权——禁止子集前置复现）：
  1. `med_dsingle_alt`：D-single（qa 20% + traces-v1 80%），alternate，epochs 按池子实测定（预计 2）
  2. `med_dsingle_joint`：**同一 order file**，joint —— Gate-3 复验对（P_src 匹配）
  3. `med_dmulti_alt`：D-multi（qa 20% + v1 50% + Toucan multi-turn 25% + Open-SWE resolved-only 5%），alternate，1 epoch
  4. `med_dmulti_repeat_alt`：D-multi 同配比、unique 池截 25M estimated、4 epochs —— G-Q1 对照（P_src 与臂 3 匹配，U_src ~4× 差）
- QA 家族经 JointExample 化走 joint grid（history_documents=QA 文档），非 legacy mdoc 类——记录为有意选择：统一 order/预算/审计基建，QA 仅占 20% 护栏份额。
- Open-SWE fixture 抽查发现 tools 列可能不含实际调用的工具（target_tool_doc_index=None 降级为普通截断）——服务器实跑前抽查真实 shards 确认普遍性，若普遍则该家族的 tool 侧贡献弱（仅 5% 份额，记录即可）。

## 22. Phase 4 基建状态（commit 7da08aa + 待提交交错修复）

- 已入库：multisource loaders（Toucan/Open-SWE/QA → JointExample，answer 面复用 `_render_agent_output_messages` 字节级一致）、trainer 接线（keep_qids 预过滤、train_source_counts）、mixture planner、dedup flatteners 扩展（trajectory/JSON-string 列）、34+21 项测试（本地 stub harness 全绿）。
- 在跑：新 dedup pass（train = v1+toucan+openswe+qa，eval = BFCL + v2-eval sessions，messages + raw 双通道）；服务器 pytest 待修复合入后跑。
- 下一步：dedup removal → planner 构建 4 臂 order/plan（含 realized 配比、时间公式 ETA）→ **08-23 早启动 medium 4 臂**（卡 1-4），卡 0 + 5-7 跑 eval/validation。

## 23. 两个补充实测（08-22 深夜）

- Open-SWE tools 列覆盖抽查（4 子集 × 首 shard 前 25 行，resolved=1 共 30 行）：**30/30 实际调用的工具全部在 tools 列内**——§21 末条的 fixture 缺口是我自己的截断 artifact，真实数据无此问题。
- 训练消费顺序核实：`GistMultiDocTrainer` 不 override sampler → HF 默认 RandomSampler 每 epoch 随机置换。因此 order 文件的机制是**成员资格 + 预算截断点**，small 臂的"appworld 前置"是成员资格偏斜（32M 前缀全是 appworld），非训练序列问题。medium planner 的配额抽样保证池子构成比例正确，token-deficit 交错保证审计窗口内局部配比可核查；RandomSampler 处理逐 epoch 序列。

## 24.（2026-08-22 晚–23）启动门修复入库 + 训测同源审计结论 + BDF pilot 并行上线

### 24.1 medium 启动门修复（外部审计 P0/P1/P2 全闭环）

- 8 commits（`adeab73..9c26028`）：per-side caps v2（tool-less/QA 例回收空槽 + gold 保留率计数；traces 例逐位不变，LEGACY 不动）；planner 分层池扫描（per-subset cap + 种子化文件序，QA/openswe 尾部子集不再零概率）；budget shrink 整体重归一化（qa 20% 护栏不再被 realized 配比冲破）；`measure_arm_psrc` 多源化；longmagpie qid 改 `qa:longmagpie:<shard>:<row>`（dedup removal 对齐生效）；alternate 臂 QA pass 不对称显式入 plan/manifest；ARM LAUNCH TABLE + 四臂 presented parity 2% 守护；provenance/ETA 单位/Toucan pin/removal-before-cap/2wiki 真实行。
- 验证：本地全量 180 绿（含 torch 套件）；**服务器全量 pytest 285/285 绿**（gjoint worktree，修复后 HEAD=9c260285）。
- §15/§18 引用修正已入库：J-separate argF1 同口径 0.099/0.098（原填 name_f1）；joint−separate=**+3.1pp（4/128）**；5e-5=0.342 注记"Gate-1 产物未入库"。

### 24.2 训测同源审计（dedup 深挖，全文见 `docs/g_joint/dedup_audit_summary.json`）

- **表面**：pass A removal=378,760（v1×v2eval exact 308,513 为大头）。
- **逐层证伪样板**：65% 是 12 字符 `(no content)`；其余为 system-reminder/Django 测试输出/τ² 转接语等 harness 模板（双侧 DF 高）；near-dup 样本全是差一位 call-id/returncode 的短模板（J≥0.98）；v1 消息按 span 全前缀重发导致重复计数。被判官逐条复核确认。
- **split 完整性**：`train_ids ∩ eval_ids = ∅`；任务指纹（task 定义消息 hash）跨 split **零碰撞**（774 eval → 277 distinct task，重复运行留在 eval 侧）。注：session_id 前缀是 run_id（每 run 29–87 sessions），不是 task id——初版 run-prefix 排除法作废。
- **真问题（任务级残留重叠）**：排除 **427 个 train 侧 session**（16 byte-identical 任务文本 / 231 单 eval session J≥0.95 近重复 / 188 低 J 含任务文本）→ `final_train_exclusion.json`（planner 消费 `removal_traces_final.json`）。
- **不惩罚**：跨任务共享的 appworld API 文档/browsecomp 语料/SWE harness 输出（488 exact + 121 near sessions）——benchmark 公共环境材料，eval 时本就可视，保留并记录在案。
- pass B（qa raw × v2eval raw + BFCL）：**removal=0**（新旧 qid 方案两遍一致）。
- **池影响**：v1 traces 池 1,364 → **937 sessions**（§20 的"由构造保证零泄漏"被本条细化取代：构造只保证 session 级，任务级排除以本条为准）。toucan×v2eval 39 + toucan×bfcl 1 + openswe×2 按原样保守移除。
- §21 预算注记：937 sessions 的 v1 池估计 ~31M estimated tokens（原假设 ~45M），d_single 的 shrink factor 会比预期更深；planner 正式跑后更新精确值。

### 24.3 BDF pilot 并行状态（branch task/bdf-pilot @ f84dc74，checkpoint 钉死 G8-small-v2=$G/fixed_joint，sha256 669502d3…）

- 部署门：522 passed / 2 failed（服务器同样缺 bfcl_eval 包——从 gjoint worktree 补 `.foreman/ref/bfcl_pkg` 后 49/49 绿，4 个 torch 文件全部 PASSED 非 SKIPPED）。
- **D**（卡 0）：电池 full/c2kv 各 200 例完成（history harness 768/16，默认导出切片——口径偏离见下）；trigger 提取冻结：n_base_paired=196，transitions C->C 22 / **C->W 20** / W->C 41 / W->W 113；sham plan 等字节 6467/6467；smoke/arms 进行中。
  - 偏离记录：D 电池最初误用 taskproxy manifest + require_tool_call=True，被 extractor 的 FATAL 守门拦下（其 `_harness_namespace` 复现 harness 默认参数）→ 改用默认导出切片（eval_ratio=0.1, seed=42）重跑，三件套口径自此一致。
- **B**（卡 1）：参照臂（full+truncate）+ P-fixed + P-turn 完成，P-struct/P-delay 收尾中；实测 joint harness ~16-17s/例/臂。
- **F**：B 完成后接（greedy_core GEN_SEED=0 → sampled GEN_SEED=20260822 → 合并报告）。
- eval200 临时清单：eval 侧 loader 前 200 example qid（200/200 appworld，含 Gate-3 的 128 条），已声明为临时。

### 24.4 边界状态

- **medium 四臂未启动**（卡 2-5 空置），等刘言成显式批准；planner 正式跑进行中（分层扫描 + 全 removal 接线），产出即入库并补 §25（realized 配比、shrink factor、臂表、ETA）。
- 卡 7 seed-2 复训同样待批准。

## 25.（2026-08-23 09:30）medium 方案 B 两臂已启动 + planner v4 全绿

### 25.1 决策落地（刘言成 08-23 指令）
- 只跑 **joint-mode 两臂**：`med_dsingle_joint`（卡2）、`med_dmulti_joint`（卡3）。alternate/repeat 臂本轮不跑（Q3 复验有意推迟）。
- 预算公式：`budget = floor(实测 v1 池 est / 0.8)` = floor(42,510,075/0.8) = **53,137,593**——两 recipe 的 traces 配额均不超池，shrink=1.000 不触发，parity 自然通过（guard 代码未动）。
- 池量出处：planner v4 实测 post-removal v1 池 42.51M est（scan 期移除 1,482 个 traces 例）；**removal 文件命中数恰为 427**（`removal_traces_final.json`，与审计 exclusion 一致）；toucan/openswe 侧 103 identifiers 命中 22 例；qa 0。
- 0.392 presented/estimated 系数标注：标定于 **v1-regime、traces-only 的 small 臂**（joint 12.62M 呈现/32.2M est、alternate 12.54M/27.7h；fixed per-side caps）——对 medium 的 QA/长文档家族该系数可能偏移，训练后 `measure_arm_psrc.py` 实测复核。

### 25.2 planner v4 产出（入库 docs/g_joint/medium_plan/）
| recipe | 例数 | total est | realized 配比 | shrink |
|---|---:|---:|---|---|
| d_single | 8,908 | 53.1M | qa 0.200 / traces 0.800 | 1.000 |
| d_multi | 13,041 | 53.1M | qa 0.200 / traces 0.500 / toucan 0.250 / openswe 0.050 | 1.000 |
- 臂表（full-pool parity，floor=41.66M/max=41.67M presented）：两臂 U=53.1M、epochs=2、MAX_SOURCE_TOKENS=U（53,137,647 / 53,149,845）。
- ETA（small 臂实测校准 22.7h/32M est）：**~75.4h/臂 → 08-26 晚**（比指令估计的 55h 保守；不阻 08-29）。
- 启动审计：两臂均 SEED=42、LR=5e-5、DOC_MODE=joint、git e3c8802+01e00ee（服务器 01e00ee）；命令/env 在 `$G/med_dsingle_joint.log` / `med_dmulti_joint.log` 头部。
- 诚实记录：planner 的截断 parity regime（`--presented_target_est`，本次未用）初版误假设 mid-epoch 截停——trainer 实际只在 epoch 循环前对冻结顺序一次性前缀截取（`_take_within_source_token_budget`），语义已修正为 per-epoch take × epochs 并有测试；本次两臂为全池 epochs=2，天然 parity，无需截断。

### 25.3 并行任务图（8 卡全占用，2026-08-23 09:30）
| 卡 | 任务 | ETA |
|---|---|---|
| 0 | D-r1 arms 收尾（corr_re/corr_all）→ paired analysis | ~1h |
| 1 | F pass1 greedy_core（104/200→）→ pass2 sampled | ~2h + ~3h |
| 2 | **med_dsingle_joint** | ~75h |
| 3 | **med_dmulti_joint** | ~75h |
| 4/5 | D-r2 扩展电池 n=900（full/c2kv，768/16 同口径） | ~5h |
| 6 | B OOM 缺行第 3 试（分配器调优） | ~15min |
| 7 | fixed_joint seed-2 复训（同 config 同 order，SEED=43） | ~23h + 电池 4.5h |
- D-r2 冻结件全部 `_r2` 后缀，不覆盖 r1；判据/prereg 未动。
