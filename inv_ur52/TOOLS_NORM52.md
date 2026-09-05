# tools 规范化后 52 集全量三臂评测(preliminary, n=1)

执行日期:2026-09-05 20:07–21:39(ascend03,dev6:34770,checkpoint-1088 只读,graph/BF16/prompt_flash/mem 0.55/pool 0.06 与原 run 同参)。独立目录 `$T = /tmp/c2kv-toolsnorm52.NIhbBb`(**按要求保留未删**);client = bfcl_eval+c2kv_eval 真实副本,sglang = python/ 真实副本(d42ce815f + 当时的 dirty 文件,无仪表化);来源记录 `PROVENANCE.md`。补丁:`c2kv_tools_norm_review.patch`(按你的规格重建——原附件未随消息到达;`_normalize_tool_for_sglang` 镜像服务端 `Tool.model_dump()`,丢 `response`、`strict` 用 `fn.get("strict", False)` 保留显式值),`git apply --check` 通过后应用,字节已归档。

## 启动前检查(全过)

- `__file__` 三项全部落 `$T`(bfcl_eval / c2kv drift / sglang);`BFCL_PROJECT_ROOT` 先于 import 设置。
- 52 ID 唯一齐全;全部 52 样本 tools 结构 = 服务端字段序 `description/name/parameters/strict`,零 `response` 泄漏。
- 52 样本模板 token 对拍:三探针 drop **1697/1150/1670 与你的 CPU 复算逐数相等**(如 _110: 5496→3799)。
- 运行插曲:首次启动 34762 端口被他人进程占用(server bind 失败,臂 rc=1),换 34770 后三臂全部干净跑完;期间原仓库路径上有他人(非本轮)的 sweep 进程与一个 hung 的 34770 启动器,均未触碰。

## 结果(唯一问题的回答)

| 臂 | 旧(09-03 run) | **新(tools 规范化)** | stayed | gained | lost |
|---|---:|---:|---:|---|---|
| C2KV | 9/52 | **9/52** | 9 | — | —(**逐 ID 完全相同**) |
| Append W2 | 6/52 | **35/52** | 5 | 30 题 | 1(_112) |
| Replace W2 | 35/52 | **37/52** | 34 | 3(_107、_110、_159) | 1(_39) |

**排序:Replace(37) ≥ Append(35) ≫ c2kv(9)——Append 从"低于不修"恢复到与 Replace 差 2 题。**

### 覆盖核查(--partial-eval 未掩盖)

三臂 `total_count=52`、details 恰 52 行、ID 集与 frozen 全等、无重复/缺失/extra、`errors=0`,无 inference error。

### 新 Append vs 新 Replace 逐 ID

- both_pass **35**;append_only **[]**;replace_only **[_112, _192]**;both_fail 15。
- **新 Append 的通过集是新 Replace 通过集的真子集**(差集恰为 _112、_192)。_112 同时是 Append 唯一 lost 题(旧 Append 通过、新失败)。

### 新旧 Append 转换

- gained(30):_100, _107, _110, _111, _115, _116, _118, _119, _12, _120, _121, _122, _126, _127, _128, _130, _132, _136, _142, _146, _151, _159, _161, _182, _191, _193, _33, _5, _57, _78
- lost(1):**_112**
- stayed(5):_101, _137, _139, _23, _50(旧 6 通过中的其余 5)

### c2kv 哨兵

新 c2kv 与旧 c2kv **逐 ID 完全一致**(9 题: _101, _112, _132, _137, _139, _182, _23, _50, _78)——运行条件无其他漂移。这也符合机制:服务端本就丢弃 `response`,补丁只改变客户端计算(preamble/抽取位置),不改变发给服务端的语义内容。

### 机制侧统计(新 Append)

repair_status 直方图:**decoded_action 133 / empty_action 86 / invalid_format 0**(旧 run:invalid_format 34+24 独占臂);segments 242、repair_success 71(旧 226/10)。缺括号类失败随帧对齐消失。

## 保留与边界

- 全部数字 **preliminary, n=1**(每臂单次)。Replace-all / rollback 等其他臂未重跑,不能与本表直接排成完整序。
- `$T` 保留(含 runs/、launcher/server 日志、preflight/summarize 脚本、补丁、PROVENANCE);本地归档 `inv_ur52/raw/toolsnorm52/`(三臂 score/details(gz)/metrics/summary/launcher logs + summary52.json + patch + 脚本)。
- 附注:_112(Append 唯一退步、Replace 通过)与 _192 是当前 Append-Replace 差距的全部内容,可作为下一步单点剖析对象;graph/重复表示干扰是否单独再测,按约定等整集结果再定。

## 工件索引

`raw/toolsnorm52/`:{c2kv,d_corr_w2,d_corr_replace_w2}.{score.json, details.jsonl.gz, metrics.jsonl, summary.json, launcher.log}、summary52.json、c2kv_tools_norm_review.patch、PROVENANCE.md、preflight52.py、summarize52.py;服务器端 `$T`(保留)。
