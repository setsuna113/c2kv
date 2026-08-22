# F 线预注册：speculative compaction timing-fork pilot（configs/bdf_pilot/f_prereg.md）

本文件在任何 F 线生成之前冻结。driver（`agent/f_timing_fork.py`）把本文件的 sha256 盖进每一行输出，
analyzer（`agent/analyze_f_fork.py`）把下面的判读卡与停止条件白名单原样写进报告。运行后不得修订；
偏差只能以 erratum 形式追加记录。

本 pilot 是**离线单决策**测量，不是系统实现，也不产出任何在线策略。

---

## 0. 术语与命名纪律

- **segment（fork 段）**：joint grid 的**最后一个 chunk**，即最后一个 history chunk。grid 的顺序是
  tool-schema chunks 在前、history chunks 按时间顺序在后（`train_data_joint.build_tool_chunks` +
  `_history_chunk_budget`，trainer 与 joint eval 共用），所以「最后一个 history chunk」就是「最后一个 chunk」。
- **branch A = `compress_now`**：已排定的 compaction 照常作用在 segment 上；这是本 pilot 的 **null policy**。
- **branch B = `defer`**：segment 在本次决策中保持 raw，更早的前缀两支共享同一份 gist 前缀。
- **deterministic check**：字段与函数一律 `deterministic_check_*`。全线代码与文档不给这个机械检查
  任何别名。
- 本文件与 F 线代码中**不使用** router / gate / ratio selection / adaptive compression ratio 作为我方设计命名；
  内部判据编号写作「判据 N」。

## 1. eligibility（E1–E4，冻结）

一个 example 进入 F 池，当且仅当四条全部满足：

- **E1**：继承 joint builder 的 skip（`_condition_doc_chunks` 返回 skip_reason 即出局，reason 原样记录）。
- **E2**：`len(history_chunks) >= 2`（skip_reason `history_chunks<2`）。
- **E3**：最后一个 history chunk 的 token 数 ∈ **[L_min = 64, max_doc_length = 1024]**
  （skip_reason `last_chunk_tokens<64` / `last_chunk_tokens>1024`）。
  L_min = 64 的理由：8× 压缩下至少留 8 个 gist 槽，避免 branch A 退化成「压了个寂寞」。
- **E4**：target 是 tool call（`require_tool_call=True`；skip_reason `target_has_tool_call=false`）。

**单 fork 规则**：每个合格 example **恰 1 个 fork**，位置固定为最后一个 history chunk 之后的边界。
不做多 fork、不做 fork 位置搜索。同一 session 内多行之间的相关性由 **session_id 聚类 bootstrap** 吸收，
不做行级独立性假设。

判定实现：`f_fork_common.fork_eligibility`；driver 每行写出 `fork_chunk_index` / `last_chunk_tokens` /
`history_chunk_count` / `tool_chunk_count`。

**`delay_recent_turns = 0` 是 F 的唯一支持配置**（F 不暴露这个 flag，driver 两支都按默认值读）。
理由：`delay_recent_turns > 0` 会把最近几轮从压缩 grid 里扣出来、以 raw 形式塞在当前 turn 前面
（branch A 照 joint 的 `_generate_with_prefix` 前置它们），而 fork 段本身就是 grid 的最后一个 chunk
——两者同时存在会让 raw 内容出现在 fork 点的**两侧**，本 pilot 没有预注册这种几何。
`_build_defer_prefix` 因此在 `raw_history_ids` 非空时抛 `SystemExit("implementation-invalid: ...")`，
而不是静默丢内容。

## 2. 位置不变式（implementation-invalid 触发器）

两支的 `original_prefix_length = system_length + doc_length` 必须**逐例相等**——`doc_length` 取
**原始 token 跨度**而非 cache 跨度，因此 branch B 的 raw segment 必须被放在
`system_length + Σ(shared chunks 的原始长度)` 上，`use_gist=True`。

driver 每例 assert；不相等即抛 `SystemExit("implementation-invalid: ...")`，整轮作废，不出数。
另设最便宜的装配自检 `--assert_greedy_repeat N`：重跑前 N 例的 branch-A greedy，要求文本逐字节一致，
不一致同样作废。

## 3. 臂定义

### Pass 1 `greedy_core`（必做，2 次生成/例）

| 臂 | 定义 | 生成 |
|---|---|---|
| **F0** | branch A greedy（compress-now，null policy） | A-greedy |
| **F2** | branch B greedy（defer） | B-greedy |
| **F3-greedy** | 用 deterministic checks 在 A/B 之间选（规则 R1） | 派生，0 额外生成 |
| **F4** | 种子硬币在 A/B 两条**已记录**输出间选 | 派生，0 额外生成 |
| **F5** | A/B 的 union（oracle） | 派生，0 额外生成 |

### Pass 2 `sampled`（3 次生成/例，前置 = sampling 开关落地）

T = 0.7，top_p = 0.95；seed 由 `(gen_seed, qid, branch, rollout)` 派生：
`torch.manual_seed((gen_seed * 1_000_003) ^ zlib.crc32(f"{qid}:{branch}:{rollout}".encode()))`。

| 臂 | 定义 | 生成 |
|---|---|---|
| **F1** | {A-s0, A-s1} 的 checks 选（**tie → rollout 0**） | A-s0, A-s1 |
| **F3-sampled** | {A-s0, B-s0} 的 checks 选（规则 R1） | + B-s0 |
| **F1-union / F3s-union** | 各自的 union，仅作 oracle 对照 | 派生 |

**F1 的前置条件**：`_generate_from_input_ids` 必须已带 `do_sample`/`temperature`/`top_p`（B/D 线负责）。
driver 用 `inspect.signature` 探测；缺失时 `--arm_set sampled` 直接报错退出。
**greedy_core 相永远不传这三个 keyword**，因此不依赖该开关。

### F3 tie 规则（R1 主 / R1b 敏感性）

设 a = branch A 通过 deterministic checks，b = branch B 通过：

| a | b | **R1（主）** | **R1b（敏感性）** |
|---|---|---|---|
| ✓ | ✗ | compress_now | compress_now |
| ✗ | ✓ | defer | defer |
| ✓ | ✓ | **compress_now** | **defer** |
| ✗ | ✗ | **compress_now** | **compress_now** |

R1 的理由：keep-compressed 是 null policy（「已排定的 compaction」），因此 F3−F0 的增益严格归因于
**check 驱动的 defer**；同时结构上防止 F3 靠「其实就是 F2」取胜。R1b 在分析期零成本算出，作为预声明敏感性，
两者同表并列，不事后择优。

### deterministic checks 定义（冻结）

一条输出通过 checks，当且仅当 `_parse_pred_call` 解析出的调用同时满足：
① `name` 是非空字符串；② `arguments` 能解析为 dict（JSON 字符串先解码一次）。
**不引用 gold**——这是一个上线时真能算的机械检查，不是评分。实现：`f_fork_common.deterministic_check_pass`。

## 4. 指标

- **主指标**：`tool_name_match`。
- **次指标**：严格 action key（`pred_action_key == gold_action_key`，即 name + 排序后 arguments 全等）、
  `argument_value_f1`。
- 派生臂的合成规则：**二值指标取 union（OR），连续指标取 max**。
- **Δ_oracle(timing) = F5 − max(在场单臂)**。
- **unconditional gap = F2 − F0** 单独一行报，**不并入 selective 部分**：它是「每次都 defer」的收益，
  与 check 驱动的选择无关。

## 5. 统计

- **配对**：同 qid 配对；只有两支（或该臂所需全部 rollout）齐备的 qid 进入该臂。
- **噪声地板**：F4 硬币重抽 200 个 seed 的成功率分布，报 95% 带。落在带内的差异不解读为排序。
- **CI**：session 聚类 bootstrap，**cluster = `session_id`，B = 2000，seed = 20260822**，percentile 法。
  预声明的对比：F3−F0、F3−F4、F3s−F1、F5−max(single)、F2−F0。单簇输入退化为点区间，如实标注。
- **每格 n 如实报**；不合格例逐条写 skip 行并在 analyzer 里按 reason 汇总。
- **MDE**：`MDE_pp = 100 * (z_0.975 + z_0.80) * sqrt(p_discordant / n_pairs)`，
  `p_discordant` 取主指标上两支的不一致率 `(compress_now_only + defer_only) / n`。
  **细于 MDE 的差异不解读为排序。**

## 6. 双台账

1. **rollout 台账**：每臂 `rollouts_generated` / `rollouts_kept` / `rollouts_per_decision_as_policy`
   （硬币作为**部署策略**只需 1 次生成，本 pilot 从两条已记录输出里选，二者分列）/ `is_oracle`。
2. **GPU-ms 台账**：每臂消耗的分支的 `system_prefill + tool_compress + full_prefill + blend + generate`
   秒和，prefill 分量分列；另出 **prefill-deduplicated** 一列（同一 (qid, branch) 的 prefill 只算一次），
   因为同分支多 rollout 在真实实现里共享一次 prefill。附 **success / GPU-sec**。

**branch B 的 decode 更贵**（raw 前缀更长）：这是**要报告的归因点**，离线**不做等化**，不额外买生成来抹平。

**factor-2 弹性**：本 pilot 的 GPU-ms 来自单卡 eager、逐例重建 prefix 的离线装置，与真实 serving 的
batching / paged KV / 融合算子相差可达 2 倍量级。因此成本结论只在 **factor-2 以内不做排序**；
超过 2× 的差距才当作成本信号。

## 7. 显存诚实条款（逐字，analyzer 必须原样输出）

> Inside the speculation window both branches are resident: the fork segment costs
> gist(x_T) + raw(x_T) = 1.125x raw(x_T) at ratio 8, so the window uses MORE memory than a
> full-only prefix, never less. Any saving materialises only after the commit. No claim of the
> form "compression frees memory, so we can afford more branches" is made.

对应字段：`resident_bytes_measured`（两分支物理和，本离线装置里两支各自持有一份共享前缀）与
`resident_bytes_logical_shared`（去重共享前缀后的诚实值 = shared + gist(x_T) + full(x_T)）。
每 token 字节数 `kv_bytes_per_token = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes`
（Qwen3-4B bf16 = 147456 B = 144 KiB）。

## 8. 四问判读卡（写进报告，不写进任何 kill 逻辑）

1. **headroom 存在吗**（对 sham / 噪声地板）→ `arm_table.delta_oracle_timing` 对 `noise_floor.band95`。
2. **优于简单基线吗** → `cis["F3g-F4"]`（硬币）与 `cis["F3g-F0"]`（null policy）。
3. **成本合理吗** → `cost_tables.rollout_ledger` / `gpu_ms_ledger` / `bytes_table`，按 §6 的 factor-2 弹性读。
4. **哪类失败最受益** → `four_cell_table` + `both_match_gold_block`（后者带 future-info caveat）。

`both_match_gold`（两支都已经打中 gold 的严格子集）用了 gold 信息，**不是任何在线策略可得**，
只用于说明「有多少决策无论怎么选都不会变」，从而稀释所有 selective 臂的 headroom。

## 9. claim 纪律

- **2608.00902**（引自 spec_F 的裁定，本会话未联网核验其题录 → `[UNVERIFIED: 2608.00902]`）
  **引为动机，不作为被我们重新发现的东西**。任何「我们发现了 X」的句式，只要 X 已在该文中，一律改写为
  「沿着 X 的动机，我们量化了 …」。
- **本 pilot 的贡献口径固定为**：在 agent next-action 决策上，**逐决策地量化 compaction timing 的
  oracle headroom**，并给出 check 驱动的选择相对硬币地板与 null policy 的位置。不是方法，不是系统。
- **oracle-union 固定句式**（analyzer 逐字输出，正文引用时不得改写）：
  > 立即压缩与延迟压缩任一成功的并集 ceiling，仅用于估计 draft-verify 理论空间，不构成选择机制
- 单 seed、单 checkpoint 的结果一律标 **preliminary, n=1**；每张结果表带脚注：
  > `<n>`-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1.
  > Training pool appworld-dominated. Paired MDE ≈ `<mde>`pp; no claim below MDE is a ranking.

## 10. 停止条件白名单（五条，穷尽）

1. implementation-invalid（位置不变式或 greedy repeat 自检失败）；
2. 无 headroom（Δ_oracle 落在 F4 硬币地板带内）；
3. 被简单基线支配；
4. 成本不可接受；
5. 优先级。

**不得因「与某论文相似」停止。** 报告里的任何数字都不接到 kill 逻辑上；判读由人做。

## 11. 冻结产物与可追溯性

driver 每行盖：`model_path` / `model_config_sha256`（checkpoint config.json 的 sha256；权重本体不逐次哈希，
成本不划算，路径与 config 已足以判断两次运行是否可比）/ `prereg_sha256`（本文件）/
`split_manifest_sha256` / `qid_manifest_sha256` / `git_short_sha` / `override_ratio` / `l_min` / `gen_seed`
（sampled 相另盖 `temperature` / `top_p` / 每行 `gen_seed_used`）。

resume 语义：只有**非 skipped** 行算作 done，key = `(qid, arm_pass, branch, rollout_index)`；
skip 行（eligibility 未过、OOM）在下次 resume 时重试。
**resume 粒度是「一个 pass」而不是「一条 rollout」**：位置不变式与双分支显存台账都需要同一 example 的两支
同时在手，因此只要某个 pass 有 rollout 未完成，整个 pass 重跑；重复行在分析期由
`index_rows_by_qid` 按 last-write-wins 折叠，不影响任何计数。
