# unified_recovery_stable52_npu67_20260903_002915 服务器调查报告(S0–S8)

调查日期:2026-09-04(晚)。访问方式:zhuyuhan@ascend03(:10005,只读,未改动服务器任何文件;临时分析脚本仅写入 /tmp/zh/)。所有原始输出已离线存档于本目录 `raw/`(索引见 §12)。

调查对象 run:`/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_npu67_20260903_002915`
- 14 个臂(不是 9):full / c2kv / rollback_d1,d2,d4 / replace_w1,w2,w4,all / recompute_w2 / append_w2 / append_w2_hint / hint_only / append_masked_w2
- 运行窗口:2026-09-03 00:29:16(manifest)→ 04:28:01(末臂 eval done);DEVICES=6,7;PORTS=34660,34670;每臂各自重启一代 server
- 模型:ckpt-1088(与 inv_1088 取证同源);ratio=4,checkpoint_interval=4,recent_full_units=0,temperature=0,MAX_COMPLETION_TOKENS=4096,REPAIR_WINDOW=1,C2KV_POOL_FRACTION=0.06,extract_source=auto,repair extract attn=prompt_flash
- 52 ids = `history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt`

---

## 0. TL;DR 判定表

| # | 问题 | 判定 | 关键证据 |
|---|---|---|---|
| S1.1 | append 损伤是 replay harness 混淆? | **否,Q1 解散** | v3 run(09-01,同 52 ids)sham_mech 8/52 与 c2kv 8/52 **逐列相同**,44 条失败记录**逐题 identical**;本 run hint_only 14/52 > c2kv 9/52 |
| 新 A | 注入失败静默 abort? | **无任何证据** | 5 个 server 日志三类 abort 串全 0;客户端 empty_response/decode_error/execution_error 全 0;errors=0;extract 全成 |
| 新 B | 各臂跑在不同代码版本? | **不成立(run 内部)** | 16 个脏文件 mtime 分三批(09-03 13:23 / 09-03 19:56-58 / 09-04 17:41-18:20),**全部晚于 run 结束 04:28**,无一落入 run 窗口 ⇒ 14 臂同一份代码 |
| S3.1/3.2 | M4 复制锚死? | **死** | append changed_action 121/226(53.5%)≈ replace 145/248(58.5%);逐字复制率 40.4% vs 35.0%,无量级差 |
| S3.5 | reference 对齐棘轮? | **对 append 不特有** | missing_reference:append 18/413(4.4%)**低于** c2kv 38/481(7.9%)与 hint_only 42/497(8.5%) |
| M2 位置碰撞 | append 的 raw KV 与 gist 同绝对位置? | **实测成立** | `repair_absolute_position_ranges` 起点 == 目标 gist 的 wrapper 起点(如 5548、5610),raw 长 = native span 长(≈4×gist_len) |
| M3 角色冲突 | 改了但改错? | 部分成立(下游表现) | append 臂独家 invalid_format 重生成(34+24 次,其余臂 0) |
| S8 | 需要新跑吗 | **主问题已决,不需要** | S8.1(APPEND_POSITION_FRAME=native)仍是把 M2 从"最强解释"升级为"充分解释"的唯一实验,值得但非必需 |

**机制链结论(一句话)**:append_w2 < c2kv 是真实模型面效应,根因是 **wrapper 坐标系下的绝对位置双占**:append 把 pre-RoPE raw KV 以目标 unit 的 wrapper 起点、native span 长度注入,与该 unit 自己的 gist 同起点双 key,且尾部(≈3/4 span)压到该 unit wrapper 区间的其余位置;这扰动了修复重生成的分布(仅 append 臂出现"文本带 `<tool_call>` 但解码不出动作"的 invalid_format),回合被截断(empty_turn 36/46),episode 提前结束(413 步 vs full 490)。replace/append_masked 的 raw 放 **native 绝对位置**、目标不放 gist(无碰撞),恢复到 0.6731。Tracy 的 "duplicate representation interference" 方向正确,但可以更精确:**毒不在"内容重复",在"绝对位置双占"(内容重复但位置错开的 append_masked == replace)**。

---

## 1. 主表与关键数字

### 1.1 19 列 quick 表(BFCL Acc / Turn Joint / step drift / trigger / recovery / steps / regen / calls-per-step)

| 臂 | Acc | 题数 | TurnJoint | StepExecDrift | StepStateDrift | Trigger | Recovery | Steps/Ep | Regen | Calls/Step |
|---|---|---|---|---|---|---|---|---|---|---|
| Full | 1.0000 | 52 | 1.0000 | 0 | 0 | - | - | 9.4231 | 0 | 1.0000 |
| C2KV | 0.1731 | 9 | 0.3932 | 0.3430 | 0.3493 | - | - | 9.2500 | 0 | 1.0000 |
| Rollback D1 | 0.5192 | 27 | 0.5680 | 0.2865 | 0.2200 | 0.4778 | 0.2481 | 10.4038 | 129 | 1.2384 |
| Rollback D2 | 0.9231 | 48 | 0.8301 | 0.1494 | 0.0575 | 0.4423 | 0.6783 | 10.0385 | 194 | 1.3736 |
| Rollback D4 | 1.0000 | 52 | 1.0000 | 0 | 0 | 0.4492 | 1.0000 | 9.4231 | 206 | 1.4837 |
| Replace W1 | 0.3846 | 20 | 0.5437 | 0.2727 | 0.2586 | 0.5440 | 0.2279 | 9.5192 | 290 | 1.6444 |
| Replace W2 | 0.6731 | 35 | 0.8010 | 0.0898 | 0.1065 | 0.5000 | 0.6290 | 9.2115 | 260 | 1.6200 |
| Replace W4 | 0.8462 | 44 | 0.8883 | 0.0475 | 0.0537 | 0.4722 | 0.7983 | 9.3077 | 249 | 1.5847 |
| Replace All | 0.9615 | 50 | 0.9515 | 0.0225 | 0.0000 | 0.4585 | 0.9138 | 9.3846 | 242 | 1.5635 |
| Recompute W2 | 0.7500 | 39 | 0.7961 | 0.0851 | 0.1120 | 0.5020 | 0.6080 | 9.2692 | 258 | 1.6058 |
| **Append W2** | **0.1154** | **6** | 0.4417 | 0.3293 | 0.3027 | 0.5885 | 0.0752 | **7.9423** | 287 | 1.8983 |
| Append W2+Hint | 0.1346 | 7 | 0.4515 | 0.3090 | 0.2588 | 0.5815 | 0.0909 | 7.6538 | 282 | 1.9523 |
| Hint Only | 0.2692 | 14 | 0.4515 | 0.3199 | 0.2837 | 0.5560 | 0.0863 | 9.5577 | 286 | 1.6137 |
| Append-Masked W2 | 0.6731 | 35 | 0.8010 | 0.0898 | 0.1065 | 0.5000 | 0.6290 | 9.2115 | 260 | 1.6200 |

注:大表(unified_recovery_comparison.md)的 executed_action_drift_rate/state_drift_rate 是 **episode 级**(append 0.9038/0.7115),minimal/quick 表是 **step 级**(0.3293/0.3027)。所有内存列(KV MiB/pool/compression)为 "-",`memory_report_coverage=0.0`。

### 1.2 append_masked_w2 ≡ replace_w2:两次真实独立运行,行为完全一致

两者所有 token 计数器**逐位相同**(chat_calls 739、extract 4571/4571、c2kv_extract 325111、repair_extract 2465052、history 359110/116824/294321、completion 38230、drift/trigger/recovery 全列同),但**耗时不同**(chat 1426.16s vs 1417.97s 等)⇒ 不是拷贝文件,是确定性重跑出同样结果。52 条 episode 的 `result` 文本逐一相同;46 条失败记录 error_type 逐一相同。adapter 源码注释明说这是设计意图:

```python
# Diagnostic parity mode for replace_w2: build the raw repair keys through
# the append plumbing, but expose them as a repair-only carrier after the
# remaining gist history so the active layout is G0...Gk + Rtarget,
# not Gtarget + Rtarget.
```

即 append_masked = "用 append 管道构建 raw key、作为 repair-only 载体放在 gist 历史末尾;目标 unit 不放 gist"。它和 replace 的差别只剩:raw 块物理位置(末尾 vs 原位)与消息顺序——两者都无位置双占,结果完全一致。**这把"内容重复是否致命"与"位置双占是否致命"干净地分开了:内容照样重复出现,只要位置不撞,0.1154 → 0.6731。**

### 1.3 三个 run 的重复性标尺(同 52 ids、同模型)

| 臂 | v3(09-01) | phaseB(09-02) | 本 run(09-03) |
|---|---|---|---|
| c2kv | 8 | 9 | 9 |
| append_w2 | 4 | 6 | 6 |
| replace_w2 | 33 | 35 | 35 |
| replace_w4 | 43 | - | 44 |
| replace_all | 49 | 50 | 50 |
| hint_only | 14 | - | 14 |
| rollback_d2 | 47 | - | 48 |

phaseB 与本 run 的共享臂**所有计数器逐位一致**(d_corr_w2:6/52、chat_calls 700、extract_calls 4419 —— 跨 server 重启的确定性复现);v3 差 ±1–2 是跨代码版本的噪声带。**append < c2kv 在三个 run 全部成立,效应稳定。**

---

## 2. S0 — 基线钉死

- **S0.1/S0.2**:两表 + 19 列 json 全文已存档(raw/s0_files.txt、raw/unified_recovery_comparison.json)。第 5、6 列(minimal 表)是 step 级 Step Exec Drift / Step State Drift;大表同名列是 episode 级。full 行 52/52、rollback_d4 52/52。
- **S0.3**:run_manifest.json 全文存档。要点在开头已列。launcher 里没有出现 REPAIR_TRIGGER 环境变量;adapter 默认 `repair_trigger=oracle` 时触发实际来自 **rule detector**(signal ≥ 5.0,detector_trigger_rate ≈ 0.44–0.59),`oracle` 谓词唯一活代码路径仍是 `== "always"`(adapter :2566-2567、:2997-2999),你之前的两条"已定死"结论在 run 代码里逐字复核成立。
- **S0.4**:**52 ids 是 "full 全对" 的 frozen 集合**(20260819 temp0 stability run 的 correct_ids),full=52/52 由构造成立 ⇒ 整表是 selected-on-full 子集上的条件比较。跨臂相对结论(append vs replace vs hint)不受影响;绝对分数不可外推到全集;对"可能引入新错误的臂"(append)在该子集上系统性不利——但 §3 的 sham 对照证明这不是 append 特有劣势的来源。

---

## 3. S1 — harness 混淆:解散

本 run **没有 sham_mech 臂**(manifest methods 14 个,无之)。替代证据两条:

1. **v3 run(09-01 00:27,同 52 ids、同 ckpt-1088、同栈)有 sham_mech**:
   - BFCL Acc 8/52 = c2kv 8/52;Turn Joint 0.3932 = c2kv;step drift 0.3389/0.3242 = c2kv;9.1346 steps/ep = c2kv;regenerated 274 步(全部丢弃)。
   - **per-id**:两个臂的 BFCL 败题记录(各 44 条)**逐题 identical**(id 集合与 error_type 全同)。
   - ⇒ "走同一抽取路径但丢弃不追加"对结果**零效应**,replay harness(触发、抽取、重生成、弃置)本身不产生任何损伤。
2. 本 run 内 **hint_only 14/52 > c2kv 9/52**:走完整 detector/regeneration 管道、只加文本 hint、不动 KV,不降反升。

**结论:Q1(append 有害是 harness artifact)解散;模型机制假设全部存活,进入 §5/§6 判别。**

---

## 4. S2 — 静默 abort:无任何证据

- **S2.1 服务端**:5 个 server 日志(顶层 2 + rollback 臂内 3)对 `C2KV injection failed` / `C2KV repair miss` / `cache miss while pinning` / `inject`(区分大小写逐串)**全部 0 命中**。
  - **重要限制**:顶层 `server_6_34660.log` / `server_7_34670.log` 每个**只含最后一代 server**(各仅 1 条 server_args、1 条 fired-up banner;时间戳 03:52:01/02,即 hint_only 与 append_masked_w2 那代)。launch 脚本对同一路径反复截断写,append_w2 那代(34660,03:10:42,pid 3770181)的 server 日志**已被覆盖,不可恢复**。⇒ 服务端 0-abort 证据覆盖:hint_only、append_masked_w2、rollback_d1/d2/d4;append_w2 需靠客户端证据(下条,同样干净)。
- **S2.2 客户端**:全部 14 臂 `errors=0`、`extract_success == extract_calls`(append_w2 4419/4419);drift_steps 里 `empty_response`、`decode_error`、`execution_error` **全 0**(所有臂)。BFCL checker 层的失败类型分布(§5.4)中 append 的 36/46 是 `empty_turn_model_response` = **回合被截断**(episode 步数耗尽后的空回合),不是请求被 abort——abort 会产生客户端可见错误,而 errors=0。
- **S2.3 池压**:客户端 `kv_runtime_report_missing == chat_calls`(全臂)⇒ memory_report_coverage=0.0,pool 数据客户端全缺(又是"未执行的默认值"模式,同 bench-serving-stack 里 selfcheck 0.00e+00 的教训)。server 侧存活代的 pool 行:峰值 74/26628 tokens、#c2kv-entry 5、usage 0.00 —— 离上限三个数量级。append 那代日志丢失,但 phaseB 与本 run 计数器逐位一致 ⇒ 若是池压/驱逐这类随机事件,不可能两次跑出逐位相同的计数器。
- **判定:新 A 死。append < c2kv 不需要任何"请求被静默吞掉"的机制。**

---

## 5. S3 — 机制判别

### 5.1 (S3.1) repair 改没改动作 —— M4(复制锚死)死

| 臂 | segments | changed_action | rate | success |
|---|---|---|---|---|
| replace_w1 | 250 | 128 | 51.2% | 31 |
| replace_w2 | 248 | 145 | 58.5% | 78 |
| replace_w4 | 252 | 151 | 59.9% | 95 |
| replace_all | 253 | 154 | 60.9% | 106 |
| recompute_w2 | 249 | 146 | 58.6% | 76 |
| **append_w2** | **226** | **121** | **53.5%** | **10** |
| append_w2_hint | 227 | 116 | 51.1% | 12 |
| hint_only | 250 | 93 | 37.2% | 12 |

append 下模型**过半触发步都改了动作**,改率与 replace 家族同级 ⇒ "模型压根没改主意、被重复件锚死"(M4)不成立。改了、但 success 只有 10/226 —— 损伤在"改的质量"上。

### 5.2 (S3.2) 逐字复制率 —— 无 M4 预测的量级差

repair_action == candidate_action(逐字):append_w2 **82/203(40.4%)** vs replace_w2 78/223(35.0%)、replace_all 55/209(26.3%)、hint_only 174/267(65.2%,无 raw KV 自然最"抄");repair_raw_text 全等:append 28/203(13.8%)vs replace 19/223(8.5%)。**append 仅略高,不构成 M4 的直接证据。**

### 5.3 (S3.3) diagnosis 全文已存档(raw/sa_schemas.txt)

append_w2:triggered 133,action w2c 10 / w2w 105 / **c2w 8** / c2c 10;state w2c 11 / w2w 88 / **c2w 0** / c2c 34。
replace_w2:triggered 124,action w2c 82 / w2w 35 / **c2w 0** / c2c 7;state c2w 1。
所有 8 个 append correct→wrong 样例(id/turn/segment/repair_target_indices/raw_tokens/physical_prefix 前/后/logical_position 前/后)完整在档。**每个样例 logical_position_before == after**(623→623、515→515、…),physical_prefix 每次恰好 +repair_raw_tokens —— 与 §7 的 else 分支账本一致。

### 5.4 失败类型直方图(BFCL error_type,仅败题落盘)

| 臂(败题数) | empty_turn | state_mismatch | exec_response_mismatch |
|---|---|---|---|
| c2kv(43) | 11 | 23 | 9 |
| replace_w2(17) | 8 | 8 | 1 |
| replace_all(2) | 1 | 0 | 1 |
| recompute_w2(13) | 5 | 8 | 0 |
| **append_w2(46)** | **36** | 8 | 2 |
| append_w2_hint(45) | 25 | 11 | 9 |
| hint_only(38) | 7 | 16 | 15 |
| append_masked_w2(17) | 8 | 8 | 1(= replace_w2) |

**append 的失败以回合截断为主(36/46=78%,c2kv 仅 26%)**,episode 平均 7.94 步 vs full 9.42 —— 整整少 ~1.5 步/题,66+ 步被烧掉。

### 5.5 invalid_format:仅 append 臂有的重生成病灶(新发现)

`repair_status` 直方图(触发步内):append_w2 **invalid_format 34**、append_w2_hint **24**,replace_w1/w2/w4/all、recompute、hint_only、append_masked **全部 0**。

语义(`history_step_common.py::decode_candidate`,源已核):`decoder.decode_execute(raw_text, has_tool_call_tag=False)` 返回空 action 且文本含 `<tool_call` 标签 → `invalid_format`,即**"模型明明输出了 tool_call 块,但 BFCL 解码器解不出任何动作"**。实例(raw/sg_analysis4.txt):

| id | candidate(被丢弃的投机步) | repair(注入 raw KV 后的重生成) |
|---|---|---|
| _57 | 反问用户要哪个 distance(无 tool call) | `{"name": "logarithm", "arguments": {"value": 630.0, ...}}`(幻觉参数,且解不出动作) |
| _100 | `{"name":"trading_login",...}`(紧凑 JSON,合法) | 同内容但 **JSON 带空格** `{"name": "trading_login", ...}` → 解码为空 |
| _110 | `{"name":"ticket_login",...}` 合法 | 同上,带空格变体 → 解码为空 |
| _111 | send_message(合理) | **换成** get_order_details(order_id=12345,幻觉) |

即:同一解码管道,candidate 全部正常解码;append 下的重生成出现"格式漂移(空格)/内容漂移(幻觉参数)"以致解码失败或动作错误——**这是上下文被污染后的分布扰动证据,append 特有**。交叉表:empty_turn 失败 × 有 invalid_format 的题 = **24 题**(36 个 empty_turn 失败里 24 题见过 invalid_format,另 12 题见过 empty_action 重生成);PASS 的 6 题里 3 题只有 empty_action、3 题无触发。

### 5.6 (S3.4) 配对显著性的方法学说明

BFCL score json 只落盘**败题**记录(pass 需由"未列出"反推),本轮我的 per-id McNemar 脚本因此没产出可用数字;替代强度证据:(a) v3 sham vs c2kv 用 44 条败题交集做到**逐题 identical**;(b) 三个 run 的重复(§1.3)把 append<c2kv 的噪声带定为 ±1–2 题,而差是 3–4 题且方向三次一致;(c) phaseB 与本 run 计数器逐位一致排除随机故障。若要正式 McNemar,从三 run 的败题集做配对即可(数据已齐,本地可算)。

### 5.7 (S3.5) reference 对齐棘轮 —— 对 append 不特有

`alignment_status` 只有 matched/missing_reference 两值(无 "beyond/exhausted" 细分):append 18/413(4.4%)**低于** c2kv 38/481(7.9%)、hint_only 42/497(8.5%),高于 replace 家族(1.7–0%)但量级只有 18 步,解释不了 105 w2w 里的大部分。**棘轮是普遍存在的表内偏差(所有压缩臂都有),不是 append 特有损伤的来源。**

---

## 6. S4 — 几何实测(M2 钉死)

### 6.1 实例:multi_turn_base_110,turn 1,step 0(append_w2,d_corr_w2,targets={1,2})

native 全史 3 个 unit;wrapper 比 native 长 47%(wrapper_native_length_ratio=1.472,因每 unit 单独套 chat 模板)。

| unit | native 帧区间 | wrapper 帧区间 | 物理内容 |
|---|---|---|---|
| 0(gist) | [5490,5534) 44 tok | [5490,5548) 58 | gist 15 |
| 1(gist+raw) | [5534,5575) 41 | [5548,5610) 62 | gist 16 **+ R1 raw 41 @ wrapper[5548,5589)** |
| 2(gist+raw) | [5575,5598) 23 | [5610,5649) 39 | gist 10 **+ R2 raw 23 @ wrapper[5610,5633)** |

- append:物理前缀 105(15+16+41+10+23)= gist 与 raw **全保留**;logical 108 前后不变。
- replace(同 id 同步):unit1/2 为 raw_replace,raw 放 **native 绝对区间 [5534,5575)/[5575,5598)**,目标不放 gist;物理前缀 79(15+41+23)。
- **要点 1(同起点双占)**:append 的 `repair_absolute_position_ranges` 起点与目标 gist 的 wrapper 起点**逐点相同**(5548、5610)。R1 长 41,gist 只有 16 ⇒ 至少 16 个 wrapper 绝对位置上同时坐着 gist key 和 raw key(gist token 的精确 per-token 位置未记录,position_frame_debug=false;按实现自然读法 gist 占 unit wrapper 区间的前 gist_len 个位置 —— 此步为**推断**,见 §11),R1 剩余 25 个位置延伸进该 unit wrapper 区间的其余部分(该处无其他 key,但把 unit 的"位置空间"撑乱)。
- **要点 2(帧分离是解药)**:replace/append_masked 的 raw 放 native 帧,gist 岛之间的空档区,与任何 gist 不共位。
- **要点 3(近端纯度不是毒)**:append 的最近 query 段是纯 gist(203/203 触发步 last_unit_mode=gist),replace 是 raw_replace(223/223)——append 近端是 gist 却崩、replace 近端是 raw 反而好 ⇒ 毒不在"靠近 query 的段是不是 raw",就在双占本身。

### 6.2 聚合(append_w2 全部 203 个触发步)

double-covered wrapper 位置共 **7579 个**(目标 unit 的 gist×raw 同起点重叠);hole=0(wrapper 区间连续)。replace_w2 的同口径计数因我的脚本把 native 帧长当作 wrapper 延伸会虚增,不引用;§6.1 的实例对照已足够。

### 6.3 (S4.3) wrapper/native 总比(eff/canonical,run 实测)

| 臂 | eff | canon | 比 | 组成 |
|---|---|---|---|---|
| full | 847 | 847 | 1.0000 | 纯 raw |
| c2kv | 49819 | 154822 | 0.3218 | 纯 gist |
| hint_only | 100905 | 314549 | 0.3208 | 纯 gist(+hint 文本不入 KV) |
| replace_w1 | 116521 | 322006 | 0.3619 | gist 96253 + repair 20268 |
| replace_w2 | 116824 | 294321 | 0.3969 | gist 82825 + repair 33999 |
| replace_w4 | 133377 | 295841 | 0.4508 | gist 75525 + repair 57852 |
| replace_all | 163195 | 293347 | 0.5564 | gist 60201 + repair 102994 |
| recompute_w2 | 118816(phys) | 298246 | 0.3395(eff/canon) | gist 83669 + repair 17136 + 下游 raw 18011 |
| **append_w2** | **98322** | **230498** | **0.4266** | **gist 75013 + repair 23309(两份都付)** |
| append_w2_hint | 148242 | 354103 | 0.4186 | gist 105134 + repair 43108 |
| append_masked_w2 | 116824 | 294321 | 0.3969 | = replace_w2 逐位 |

单步 wrapper_native_length_ratio 样例 1.472(wrapper 比 canonical 长,chat 模板膨胀)。

---

## 7. S5 — 脏树 diff(决定性 hunk 粘贴)

### 7.0 代码状态总表(关键!)

| 代码 | 状态 | mtime | 与 run 的关系 |
|---|---|---|---|
| kvoffload-sglang HEAD | `7de9e81051a3`(= manifest) | - | 你读的 committed 基线**有效** |
| c2kv_pool.py / c2kv_injection.py / ascend_backend.py | **干净(0 diff)** | - | = committed ⇒ 你已读的注入/池代码**就是 run 时代码** |
| scheduler.py / qwen3.py / io_struct.py / protocol.py / http_server.py / schedule_batch.py / forward_batch_info.py / model_runner.py 等 16 文件 | 脏,+1379/−60,+1 untracked | 09-03 13:23 批 3 个;09-03 19:56–58 批 5 个;09-04 17:41–18:20 批 8 个 | **全部晚于 run 结束(04:28)** ⇒ 现在的字节 ≠ run 时的字节;run 时是一个被覆盖的中间脏版 |
| bfcl(sweep/adapter 侧) | HEAD `1b6f3770`,manifest 是 `1efcc4ec`(是其祖先);脏 3 文件全是 *baselines* 侧 | 09-04 | **bfcl_history_kv_repair.py mtime 09-02 17:02、run_bfcl_kv_repair_sweep.sh 09-02 15:50、history_step_common.py 08-19 —— 早于 run 且未被后续触碰 ⇒ eval 侧代码 = 今天所读 = run 时代码** |

⇒ **run 时代码的可恢复部分:全部 bfcl/adapter 侧 + sglang 的 committed 基线 + 三个干净文件;不可恢复部分:scheduler.py / qwen3.py / io_struct.py / protocol.py 等的 run 时中间态(只能靠行为证据约束,见 §7.4)。**

### 7.1 (S5.2) 位置账本 hunk(当前脏版;`-` 为 committed)

```diff
@@ -3353,30 +3844,42 @@ class Scheduler(
         position_ids = self.c2kv_pool.get_position_ids(entry)
         position_start = int(position_ids[0].item())
         position_end = int(position_ids[-1].item()) + 1

         req.kv_committed_len = kv_start + repair_len
         req.kv_allocated_len = kv_start + repair_len
+        repair_kind = (
+            "recomputed"
+            if entry.repair_mode in {"d_corr_recompute", "d_corr_recompute_w2"}
+            else "repair"
+        )
+        self._add_c2kv_kv_memory_tokens(...)
         repair_advances_logical_position = entry.repair_mode in {
             "d_corr_recompute",
             "d_corr_recompute_w2",
             "d_corr_replace_w1",
             "d_corr_replace_w2",
             "d_corr_replace_w4",
             "d_corr_replace_all",
+            "append_masked_w2",
             "raw_all_replace",
             "raw_all_replace_direct",
-        }
+        } or str(entry.repair_mode or "").startswith("history_kv_")
         if repair_advances_logical_position:
             # Raw replacement repair KV is already RoPE-rotated at its original
             # Full-prompt absolute positions.  The compressed active prompt may
             # inject the raw span at a shorter physical KV offset, so following
             # query tokens must continue from the raw span's absolute end.
             req.c2kv_position_correction = position_end - req.kv_committed_len
         else:
             req.c2kv_position_correction -= repair_len
```

- committed(:3363)集合里**没有** append_masked_w2(是脏 diff 加的);d_corr_w2 / d_corr_w2_hint 两个版本都**不在集合** ⇒ append 家族走 else:`position_correction -= repair_len`,query 逻辑位置不动。
- **行为反推 run 时同样成立**:diagnosis 每例 logical_position_before==after(§5.3)⇒ run 时的 append 也走 else 分支。append_masked_w2 在 run 时是否已在集合里无法从字节证明,但其 raw 用 native 帧(§7.2,adapter 决定),advance 与否对它数值上无影响(结果=replace)。

### 7.2 append 的坐标系选择在 **adapter**(run 时代码,已钉死)

```python
# bfcl_history_kv_repair.py (mtime 09-02 17:02, = run 版)
append_coordinate_frame = (operation == "append" and self.c2kv_append_position_frame == "wrapper")
# self.c2kv_append_position_frame 默认 "wrapper" (env C2KV_APPEND_POSITION_FRAME, :3609)
...
if append_coordinate_frame:
    # The raw repair KV length is the native Full-context span length. The C2KV
    # wrapper unit may tokenize longer/shorter, so place raw KV in the wrapper
    # coordinate frame starting at the corresponding unit start, but keep the
    # position vector exactly token_len long.
    repair_position_ids = list(range(wrapper_starts[index], wrapper_starts[index] + span_len))
    raw_kv_position_mode = "pre_rope"
elif extract_source == "serving_cache" (with append+wrapper):
    raise RuntimeError("Append repair needs pre-RoPE raw K ...; serving_cache only
                        contains already-rotated native-frame K.")
```

replace / append_masked 的 operation ≠ "append" ⇒ 不进 wrapper 帧,`raw_kv_position_mode` 保持 `"rotated"`、position_offset = native `starts[index]`。**这就是 append 独享位置双占的代码源头,而且这段是 run 时代码(mtime 钉死)。**

### 7.3 (S5.3) qwen3 raw KV 捕获 hunk(当前脏版;mtime 09-03 19:57 = run 后)

```diff
@@ Qwen3ForCausalLM.generate_raw_repair_kv
+        repair_position_ids: Optional[List[int]] = None,
+        raw_kv_position_mode: str = "rotated", ...
-        """Run a correctness-first full prefill and capture raw RoPE'd KV.
+        """... In ``rotated`` mode the returned K already carries native
+        Full-prompt RoPE. In ``pre_rope`` mode the returned K is captured after
+        base QKV + QK norm but before RoPE; the caller supplies the position
+        IDs that will be used when it is injected."""
...
-            q, k, v = layer.self_attn.forward_prepare_native(...)
+            qkv, _ = layer.self_attn.qkv_proj(attn_input)
+            q, k_pre, v = qkv.split([...], dim=-1)
+            q, k_pre = apply_qk_norm(q=q, k=k_pre, ...)
+            q, k = layer.self_attn.rotary_emb(positions, q, k_pre)
+            repair_k = k_pre if raw_kv_position_mode == "pre_rope" else k
             raw_key_values.append(
-                    k[span_start:span_end].contiguous().clone(),
+                    repair_k[span_start:span_end].contiguous().clone(),
                     v[span_start:span_end].contiguous().clone(),
             )
             q = q.view(...); k_attn = k.view(...)   # 注意力继续用旋转后的 k
```

- **"未旋转 k 泄漏进抽取 attention"在当前代码不成立**:attention 用 `k`(已旋转),pre_rope 只影响**捕获值**;注入侧 `store_repair(..., already_rotated=(raw_kv_position_mode != "pre_rope"))`(scheduler hunk,raw/§7.5),pool 侧旋转逻辑在干净的 c2kv_pool.py(= committed,你已读)。
- committed qwen3(7de9e81)**没有** raw_kv_position_mode/pre_rope(grep 0 命中)⇒ run 时 qwen3 是脏版(被 19:57 的编辑覆盖)。**run 时 pre_rope 机制必然在**:adapter(run 版)在发该字段、4419 次 extract 全部 200、且 append 行为符合 wrapper 帧设计(logical 不动 + 双占几何)。当前版里的 `history_kv_method/h2o/snapkv/pyramidkv` 打分参数是 run 后新加的(服务于后续 eviction 实验,untracked history_kv_eviction.py 同批)。
- **S5.3 的最终裁决需要 S8.1**:如果给 append 换 native 帧(去掉双占、保留重复)分数恢复 ⇒ M2 充分;若仍崩 ⇒ 才轮到怀疑 run 时 qwen3 的 pre_rope 数值正确性(不可恢复字节)。

### 7.4 (S5.4) 协议面

committed protocol.py 已有 `c2kv_repair_key_hashes` / `c2kv_repair_only_key_hashes` / `c2kv_repair_token_start`;committed io_struct 已有 `repair_mode` / `already_rotated`。**committed 里没有** `c2kv_use_gist_projection`、`raw_kv_position_mode`、`repair_position_ids`、`history_kv_*` —— 但 adapter(run 版)在发前两者 ⇒ run 时 protocol.py/io_struct.py 是脏的。当前脏 diff 新增:protocol +`c2kv_use_gist_projection`(消息级) + 9 个 repair extract 字段;io_struct +`c2kv_kv_memory_hint`/`c2kv_use_gist_projection`(两个 Req 类) + `kv_runtime_stats`/`kv_memory_reports` 输出位 + 9 个 extract 字段。全文在 raw/sd_diffs.txt。

### 7.5 (S5.5) 期望为空检查

- `mem_cache/c2kv_pool.py`、`mem_cache/c2kv_injection.py`、`hardware_backend/npu/attention/ascend_backend.py`:**存在且 0 diff** ✔
- `managers/schedule_batch.py`(+70)、`model_executor/forward_batch_info.py`(+35):**非空** —— 但 mtime 09-04 17:50 / 09-03 19:56 均 run 后 ⇒ 现 diff ≠ run 时状态;内容方向是 history_kv eviction/位置修正相关。**"run 时这两个文件是否也脏"不可知**,这是全报告最诚实的一个开口(不过注入主路径在 scheduler + pool + ascend_backend,前两者一个被覆盖一个干净)。

---

## 8. S6 — 服务端运行时事实

- **S6.1 W^g**:server 日志 info 级**不打印任何权重加载行**(grep "Loaded weight/unexpected/missing/skipped/gist_*_proj" 全空)⇒ 无法直接验证。间接排除其作为 append 特有解释:replace_all 0.96 / replace_w4 0.85 证明 gist 权重功能正常;若 W^g 未加载,所有含 gist 臂(即全部压缩臂)会同崩,解释不了 0.11–0.96 的臂间差。H1 维持"不可直接验证但非 append 特有因素"。
- **S6.2**:attention_backend=**ascend**(prefill/decode 同);**disable_radix_cache=False(radix 开着)**;page_size=128;mem_fraction_static=0.55;chunked_prefill 8192;max_prefill_tokens 16384;c2kv pool: fraction 0.06 → **26628 tokens / max_entry 65536 / 147472 bytes/token / 3.66 GiB**,日志有 "[C2KV HYBRID BRIDGE]"/"C2KV round input prepared" 等分块注入路径行;hicache_ratio=2 只是参数默认(enable_hierarchical_cache=False);random_seed 每代不同(temp=0 下不影响)。
- **S6.3**:enable_c2kv=True, c2kv_gist_type='dynamic-interleave', c2kv_gist_param='qkv', ratio 由 ckpt(=4)决定;与 ckpt-1088 forensics 的训练侧一致。

---

## 9. S7 — 代码一致性(新 B 的最终裁定)

**时间线(run 窗口 00:29:16–04:28:01)**:

| 代(server) | 端口/卡 | 启动 | 跑的臂 |
|---|---|---|---|
| 1 | 34660/d6, 34670/d7 | 00:29:20 | full, c2kv |
| 2(臂内日志) | 34660/d6, 34670/d7 | 00:51:49 | rollback_d1, d2 |
| 3(臂内) | 34660/d6 | 01:19:55 | rollback_d4 |
| 4 | 34670/d7 | 01:19:55 | replace_w1 |
| 5 | 34660/d6, 34670/d7 | 01:58:08 | replace_w2, replace_w4 |
| 6 | 34660/d6, 34670/d7 | 02:35:15 | replace_all, recompute_w2 |
| 7 | 34660/d6, 34670/d7 | 03:10:42 | **append_w2(d_corr_w2, d6)**, append_w2_hint(d7) |
| 8 | 34660/d6, 34670/d7 | 03:51:44 | hint_only(d6), **append_masked_w2(d7)** |

**脏文件 mtime 三批:09-03 13:23(http_server/protocol/tokenizer_communicator)、09-03 19:56–19:58(qwen3/forward_batch_info/model_runner/utils/tp_worker)、09-04 17:41–18:20(scheduler/schedule_batch/serving_chat/io_struct/tokenizer_manager/scheduler_output_processor/detokenizer/multi_tokenizer + untracked history_kv_eviction.py)。全部晚于 04:28。**

⇒ **裁定:(a) run 窗口内没有任何一次代码写入(mtime 证据),14 臂(含九个修复臂)跑在同一份当时代码上,新 B 的"跨臂跨版本"对本 run 内部比较不成立;(b) 但 13:23/19:59 两批编辑把 run 时的 scheduler/qwen3/protocol/io_struct 覆盖了 ⇒ "现在的 diff ≠ 当时跑的代码"这个保留不仅成立而且加重——S5 里凡 mtime 晚于 04:28 的文件,其当前 hunk 只能当"后世版本"读,唯一可当 run 代码读的是 adapter 三件套(mtime 早于 run)与 sglang 侧干净文件。**

另一个独立发现:顶层 server_*.log 被每代 launch 截断重写 ⇒ append_w2 那代 server 日志永久丢失(§4)。

---

## 10. S8 — 需要新跑吗?

主问题(Q1、新 A、新 B、M4、棘轮)全部已被盘上证据裁决,**不需要新跑**。仍值得的:

- **S8.1 `C2KV_APPEND_POSITION_FRAME=native` 重跑 `d_corr_w2`(52 集)** —— 唯一能把 M2 从"最强解释"升级为"充分解释"的实验:保留 gist+raw 内容重复、去掉 wrapper 帧双占。预测:恢复到 ≈0.67(replace 水平)⇒ M2 充分;仍 ≈0.12 ⇒ M3/数值正确性(run 时 qwen3 pre_rote 不可恢复字节)上位。flag 已接好(adapter :3609 默认 wrapper)。
- **S8.2 `append_w1`(d_corr_w1)**:窗口=1,双占缩小到 1 个 unit,可测剂量-响应。
- **S8.3 `replace_w2_hint`**:不必新造——append_w2_hint 与 hint_only 已给出 hint 交互数据(hint 只在 append 侧小幅 +1,救不动双占)。

---

## 11. 保留与不确定性(诚实清单)

1. **run 时 scheduler.py/qwen3.py/protocol.py/io_struct.py 的确切字节不可恢复**(被 09-03 13:23 / 19:57 / 09-04 的编辑覆盖)。行为约束:extract 全成、append 的 logical 位置不动、双占几何与 adapter 发送的参数一致 ⇒ "append=wrapper 帧 pre_rope 双占"作为 run 时行为是数据钉死的,但**实现细节**(如 store_repair 调用形态)以当前 diff 为参考、标注为后世版本。
2. **gist token 的 per-token 绝对位置未记录**(position_frame_debug=false)。§6.1 的"16 个双 key 位置"按"gist 占 unit wrapper 区间前 gist_len 个位置"的读法;若实现是均分/其他映射,双占位置数不同,但"raw 与 gist 同起点、raw 长 4×gist"这两点不受影响(区间直接来自 run 数据)。
3. **McNemar 未正式做**(score 文件只落败题,方法见 §5.6);替代证据链已足够支撑结论,正式配对可离线补算。
4. **`repair_trigger=oracle` 的语义**:run 行为 = rule detector(阈值 5)触发,oracle 谓词只在 "always" 活——与你 grep 的结论一致,已复核 adapter run 版源码。
5. phaseB 与本 run 计数器逐位一致是**强**可复现证据,但它是"同代码同种子"的确定性;v3 与二者的 ±1–2 差异说明跨代码版本存在噪声带,本报告所有臂间比较都在同一 run 内(phaseB 作旁证)。
6. 顶层 server 日志只含最后一代 ⇒ append_w2 服务端 0-abort 是**推断**(由客户端 errors=0、extract 全成、跨 run 计数器一致、以及同管线末代 0-abort 共同支撑),非直接观测。

---

## 12. 工件索引(全部离线于本目录 raw/)

| 文件 | 内容 |
|---|---|
| s0_layout.txt | RUN_ROOT 目录结构、du、launcher 清单 |
| s0_files.txt | manifest 全文、14 个 launcher 全文、三张表 md、frozen ids、臂目录结构 |
| unified_recovery_comparison.json | 19 列 json 全量 |
| sa_schemas.txt | append_w2_diagnosis.json 全文、各臂 summary.json 头、metrics/details schema、runner.log |
| sb_serverlogs.txt | server 代数(1 banner/log)、server_args 全文 ×2、abort grep 矩阵、权重加载/c2kv config 行 |
| sc_git.txt | 两仓库 HEAD/status/diff --stat、16+3 脏文件 mtime、pool 峰值提取 |
| sd_diffs.txt + *.diff(16 个) | 全部定向 diff 全文(-U10),含期望为空检查、adapter mtime、bfcl 祖先检查 |
| se_adapter.txt / sf_adapter_core.txt | adapter:_arm_config 全文、raw KV 模式选择、append/replace/masked layout 构建、build_info 写入、repair_trigger、字段写入点 |
| sg_analysis1.txt | schema probe、(作废的 pass/mcnemar)、s31 changed、s43 比值 |
| sg_analysis2.txt | s32 复制率、s35 棘轮、s22 步级状态、s4 实例 layout + 聚合 |
| sg_analysis3.txt | error_type 直方图、v3 sham≡c2kv 逐题、v3↔本 run 一致性 |
| sg_analysis4.txt | empty_turn×invalid_format 交叉表 + 4 条 invalid 样例原文 |
| sh_check.txt | score 目录结构、phaseB 汇总 CSV + 报告、v3 目录/表 |
| si_tail.txt | per-entry score 文件路径、phaseB summary 全行 |
| sj/sk/sl/sm_*.txt | invalid_format 赋值语义(decode_candidate 源码)、repair_status 写入点 |
| analyze1-4.py / sg_run.sh | 全部分析脚本(可复算) |
