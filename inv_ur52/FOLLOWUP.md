# FOLLOWUP(第二轮):graph 状态、d42ce815f 与出表代码的关系、原始载荷落地

应第二轮审计的三组诉求做的服务器取证(2026-09-04 深夜~09-05 凌晨,zhuyuhan 账号只读)。原始输出:`raw/f1_graph_git.txt`、`raw/f2_commit.txt`、`raw/f3_commitdiffs.txt`、`raw/commitdiffs/commit_{1..7}.diff`;载荷:`payload/{run,v3}/`。

---

## ① 出表时的 graph 状态与错误日志

### Graph:开启且 decode 确实命中 replay(全部 5 个存活 server 代)

| server 代 | Capture banner | npu graph:True / False( decode 批日志行) |
|---|---|---|
| 34660 末代(hint_only,03:52) | begin 03:55:06 / end 03:55:13(6.26s) | 959 / 1929 |
| 34670 末代(append_masked_w2,03:52) | begin 03:55:08 / end 03:55:14(6.32s) | 955 / 1709 |
| rollback_d1 臂内(00:55) | begin/end 6.32s | 884 / 1786 |
| rollback_d2 臂内(00:55) | begin/end 6.30s | 922 / 1786 |
| rollback_d4 臂内(01:23) | begin/end 6.47s | 931 / 1783 |

server_args(每代都打印):`disable_cuda_graph=False`、`cuda_graph_max_bs=64`、`disable_piecewise_cuda_graph=True`、`attention_backend='ascend'`(decode/prefill 同)。**没有开启 `--debug-tensor-dump*`**(其会关 graph,不构成干扰)。

**关键代码事实(这一条是 run 钉死的,不受脏文件不可恢复影响)**:`cuda_graph_runner.py` 与 `npu_graph_runner.py` 在出表基线 `7de9e8105` 里就是干净的(grep c2kv/gist = **0 命中**),且在 `7de9e8105 → d42ce815f` 之间**零改动**(commit 文件清单里也没有它们)。即:**graph capture/replay 路径从 run 时到最新提交都不携带 gist projection 标志**。若 run 时的脏 qwen3 的 projection 分支在 eager 前向生效(见②的限制),"prefill 走 gist projection、decode replay 走 base QKV"的混合路径在出表时**成立**——这影响所有 c2kv 压缩臂,不能单独解释 append 差距,与你的判断一致。

### C2KV_USE_GIST_QUERY_PROJECTION

开关存在于 sglang 侧代码:`qwen3.py` 的 `forward_prepare_native` 里 `os.environ.get("C2KV_USE_GIST_QUERY_PROJECTION", "1") != "0"`(**默认开**;d42ce815f 版 :248 附近,输出原文在 f3)。sweep 脚本(`run_bfcl_kv_repair_sweep.sh`)与 launcher 日志里**没有任何对该 env 的设置或记录** ⇒ 按默认=开;但"出表时的进程环境里它是否被外部改过"**未记录**(launcher 全文与 manifest 均无)。客户端侧 `c2kv_use_gist_projection` 消息字段由 adapter(run 版,4 处)对每条 gist/repair 消息固定发 True。

### 错误日志(新增模式,全 0)

`C2KV injection failed` / `C2KV repair.*miss` / `cache miss while pinning` / `repair_inject_alloc_failed` / `repair_inject_context_overflow` / `Capture.*(begin|end)` 以外的 abort 类模式:在 5 个 server 日志里**全部 0 命中**。`log_requests=False`(无逐请求日志)。覆盖范围同前:两端口各只剩**最后一代**(hint_only / append_masked_w2)+ 3 个 rollback 臂内日志;**append_w2 那代(03:10,34660)日志被后续 launch 截断覆盖,不可恢复**——服务端 0-abort 对 append_w2 本代仍是推断而非观测。

### server 启动/重启时间线(8 代,run 窗口 09-03 00:29:16–04:28:01)

| 代 | 端口/卡 | server 启动 | 臂(runner start→done) |
|---|---|---|---|
| 1 | 34660/d6 | 00:29:20 | full 00:32:56→00:42:11 |
| 1 | 34670/d7 | 00:29:20 | c2kv 00:32:56→00:51:32 |
| 2(臂内日志) | 34660/d6 | 00:51:49 | rollback_d1 →01:19:20 |
| 2(臂内日志) | 34670/d7 | 00:51:49 | rollback_d2 →01:19:24 |
| 3(臂内日志) | 34660/d6 | 01:19:55 | rollback_d4 →01:47:22 |
| 4 | 34670/d7 | 01:19:55 | replace_w1 →01:57:49 |
| 5 | 34660/d6 + 34670/d7 | 01:58:08 | replace_w2 / replace_w4 →02:34 |
| 6 | 34660 + 34670 | 02:35:15 | replace_all / recompute_w2 →03:08/03:10 |
| 7 | 34660/d6 + 34670/d7 | 03:10:42 | **append_w2(d6)** / append_w2_hint(d7) →03:44/03:51 |
| 8 | 34660/d6 + 34670/d7 | 03:51:44 | hint_only / append_masked_w2 →04:27/04:28 |

---

## ② d42ce815f 与出表代码的关系

`d42ce815f`("fix append",20 文件 +2047/−63)提交于 **2026-09-05 00:32:03 +0800**——在我 09-04 深夜取证(HEAD 仍是 7de9e8105+16 脏文件)**之后**。当前工作区只剩 `session_aware_cache.py` 一个脏文件。

把它与我 09-04 19:40 存档的脏 diff(`raw/*.diff`)逐文件比对(归一化行尾后):

| 文件 | commit diff vs 我的存档 | 结论 |
|---|---|---|
| models/qwen3.py | **0 行差异**(593 行 diff 完全一致) | commit = 09-03 19:57 版本 |
| model_executor/forward_batch_info.py | **0 行差异** | commit = 09-04 17:50 版本 |
| managers/scheduler.py | 差 71 行,全部是 `persistent_session`/`session_controller` 新增(存档之后的 18:20→00:32 开发) | commit = 存档 + session 层 |
| cuda_graph_runner.py / npu_graph_runner.py / **c2kv_injection.py** / **c2kv_pool.py** | 7de9e81→d42ce815f **零改动** | 两版本相同 |

**裁定**:
1. `d42ce815f` ⊇ 我存档的"出表后最终脏状态" + 后续 session 功能。**不能用它推定出表(09-03 00:29–04:28)时这 16 个文件的内容**——它们的 mtime(09-03 13:23 / 19:56-58 / 09-04 17:41-18:20)全部晚于 run 结束,run 版本是被覆盖的前身,磁盘上无快照。此项按你的标准记为**无法追溯**。
2. 但以下文件的 run 版本**是钉死的**(run 时即干净 = committed 7de9e8105,且到 d42ce815f 未变):**c2kv_injection.py(注入侧二次旋转点 :299 就是 run 代码)、c2kv_pool.py、cuda_graph_runner.py、npu_graph_runner.py、ascend_backend.py**。
3. qwen3 的抽取顺序(`q,k = rotary_emb(...); repair_k = k_pre if pre_rope else k`,后 clone)是 **19:57 版本**;run 版本未知(只知 pre_rope 机制当时在运行:adapter 在发、4419 次 extract 全 200)。CUDA 的 in-place 缺陷是否在 NPU 上复现(以及 run 版本是否有同样顺序),只能靠**真实 shape 的逐层相位等式**检验——位置元数据(position_start)正确不能证明 K 相位正确,同意。
4. defect #1(graph 无 projection)对 run 的适用性:graph 开启 ✓(①)、graph runner 无标志 ✓(run 钉死)、qwen3 run 版本是否有 projection 分支 **不可直接证明**——若 run 版本没有该分支,则 projection 当时整体是 no-op(eager 也不用 gist QKV),混合路径问题不存在但"第一轮权重修复未生效"的问题更严重;若 run 版本有该分支(eager 生效),则混合路径成立。两种情况都需 graph/eager logits 对照来分辨。

---

## ③ 原始载荷(已落地 `payload/`,details 为 gzip 压缩)

| 文件 | 内容 |
|---|---|
| payload/run/{c2kv,append_w2,replace_w2,append_masked_w2}.details.jsonl.gz | 09-03 run 四臂 details.jsonl 原样(52 行/臂) |
| payload/run/*.score.json | BFCL_v4_multi_turn_base_score.json(**只落败题记录**,首行 aggregate) |
| payload/run/append_w2_diagnosis.json | diagnosis 全文 |
| payload/v3/{sham_mech,c2kv}.details.jsonl.gz + *.score.json | v3 run(09-01)两臂,供 sham parity 自核 |

details 每行字段:`result`(逐 turn 最终文本列表)、`drift_steps`(每步:`candidate_action/candidate_raw_text/candidate_status`、`repair_action/repair_raw_text/repair_status`、`repair_build_info`(含 history_layout 全部区间)、`reference_action`、`alignment_status`、`empty_response/decode_error/execution_error`、`state` 等)、`repair_segments`、`c2kv_drift_metrics`、`inference_log`。**adapter(run 版)不含 `finish_reason` 任何出现——客户端既不检查也不记录该字段**(你指出的缺口确认;`err=0` 不能排除 finish_reason=abort 路径)。

### 顺带两个新分析(脚本 `raw/analyze5.py`,输出 `raw/sg_analysis5.txt`)

1. **empty_turn 失败解剖**(区分 abort vs 截断):四臂全部 empty_turn 失败(c2kv 11 / append 36 / replace 8 / masked 8)都是 **"runner 的 result 在该 turn 有非空回复、checker 却报 response list 为空"**——没有任何一例是"turn 没跑到"或"回复为空字符串"。且 c2kv(无任何注入)也有 11 个同类 ⇒ 该错误类型与 checker/handoff 语义(该 turn 缺终结性非工具回复)成比例,不是逐请求 abort 的签名;masked 的 empty_turn 败题 id 集合与 replace 完全一致。
2. **v3 sham↔c2kv parity 升级到最强等级**:52/52 题 **result 全文逐一相同**、drift_steps 数逐一相同(此前只比了败题记录)。

---

## 与上一份报告的勘误

1. **撤回"内容重复不是毒、位置双占是毒"的强表述**:append_masked 同时改了三件事——删除 target gist、raw 用 native/rotated 帧(append 用 wrapper/pre-RoPE)、服务端 query cursor 接 native raw 末尾(append 保原逻辑 cursor)。它不是隔离实验;`C2KV_APPEND_POSITION_FRAME=native` 也因 cursor 与帧联动不是干净的"只去碰撞"。同意:干净实验应从同一份 append cache 出发,只删 target gist、保持其余 K/V 相位与 query position 不变。
2. **sham 的覆盖面修正**:sham 不经过 raw 注入,清不掉注入面(含 finish_reason=abort)的混淆——它只钉死触发/抽取/重生成/replay。逐 ID 比对上一轮已做、本轮升级为 52/52 全文相同(见上)。
3. 上一轮我实际**没有**主张 copy dominance 坐实(数据是 53.5% vs 58.5%,方向相反);"把被丢弃的动作放回 raw history"也不是我的报告内容。不冲突,仅澄清归属。
4. M2 的正确表述降级为:**"append 在 wrapper 帧与 target gist 同起点注入、双占绝对位置"是 run 数据实测的布局事实**;其因果权重未与 phase 正确性(pre_rope 双旋?)、graph 混合路径分离,排序实验前不写根因。

## 下一步(与你的计划一致)

1. graph/eager logits 对照:固定同一 cache + 同一 decode 输入,**不得**用 `--debug-tensor-dump-output-folder`(会自动关 graph);记录是否实际命中 graph。NPU 上 `npu graph:True/False` 日志行可作命中观测。
2. raw-K 相位等式(真实 extraction shape,逐层):`RoPE(pre_rope_K, native_positions) ≈ rotated_K` 且 `pre_rope_V == rotated_V`。注意 CUDA 的 in-place 缺陷在 NPU 的对应分支未定;小随机 tensor 可能走错 kernel 分支。
3. 通过后再做"同 cache 只删 target gist"的 snapshot 隔离。
