# zhuyuhan home 删除事故:删前状态重建与备份清单

事故:2026-09-06 00:12,我在 zhuyuhan 账号的清理命令中误含 `rm -rf "$HOME"`,删除了 `/home/zhuyuhan` 全部内容。本文档从本会话所有已落盘的抓取输出(inv_ur52/raw/*,已推 GitHub)重建删前状态,并逐项标注备份所在。

## 1. 删前状态重建(来源:我当时抓取的 verbatim 输出)

### /home/zhuyuhan/ 顶层
- `miniconda3/`:base py3.13.13;envs 至少 `bfcl`、`sglang`(sglang env 为 py3.11 + torch_npu)。路径证据:`envs/bfcl/bin/python`、`envs/sglang/bin/python`(sweep 脚本 SGLANG_PYTHON/BFCL_PYTHON 默认值,r1_sweep.txt)。
- `project/`(见下)。
- dotfiles/工具目录:`.bash_history .cache .claude .codex .copilot .triton .vscode-server`(由删后 00:13–00:22 的重建壳反推原本存在;原内容丢失)。

### project/gorilla/
- `berkeley-function-call-leaderboard/`:git HEAD `1b6f3770131b`,1efcc4ec 为祖先;顶层文件清单有记录(architecture_diagram.png、bfcl_eval/、c2kv_eval/、bfcl_eval.egg-info、openfunctions_evaluation.py、pyproject.toml、result/、results/、score/ 等,sd_diffs.txt @BFCL 段)。
  - 删前 dirty(09-05 12:16,status 原文存 PROVENANCE/srcnap 流程):M `c2kv_eval/analysis/compare_kv_repair_sweep.py`、M `c2kv_eval/scripts/run_bfcl_fixed_depth_sweep.sh`、M `run_bfcl_history_kv_baselines.sh`、M `run_bfcl_history_multistep_checkpoint.sh`、M `run_bfcl_kv_repair_sweep.sh`;?? `bfcl_task_oracle.py`、`compare_full200_unified.py`、`run_bfcl_full200_stage1.sh`、`run_bfcl_full200_stage2.sh`、`results/`。
  - results/multi_turn_base_full200 存在(find 输出)。
- `bfcl_runs/`:共 **55 个子目录**(ls -la 的 57 links)。我见过名字的(部分清单,s0_layout/t1 等):
  `kv_repair_phaseB_stable52_npu67_20260902_200049`、`quick_recovery_oracle_k4_stable52_20260831_131047`、`tooldef_hardneg_multi_turn_base_200`、`unified_recovery_stable52_npu04567_20260831_101632`、`unified_recovery_stable52_npu04567_20260831_101812`、`unified_recovery_stable52_npu56_20260831_004120`、`unified_recovery_stable52_npu67_20260903_002915`、`unified_recovery_v3_stable52_npu67_20260901_002715`、`history_full_temp0_stability_20260819_172725`、`history_full_closed_loop_multi_turn_base_200`、`combined_logistic_v2_replace_w2_stable52_20260905_231805`(他 23:18 新启,删前在写)。**其余 ~44 个目录名我从未列出过,无法从我的记录重建。**

### 0903 原 run(调查对象)完整结构(有全量记录:s0_layout.txt + s0_files.txt)
14 臂(full、c2kv、rollback_d1/d2/d4、replace_w1/w2/w4/all、recompute_w2、append_w2、append_w2_hint、hint_only、append_masked_w2),每臂 logs/{details.jsonl, eval.log, metrics.jsonl, runner.log, summary.json} + result/ + score/;rollback 臂另有 checkpoint_{metrics,segments,steps}.jsonl 与臂内 server 日志;根:run_manifest.json、unified_recovery_comparison.{md,csv,json}、minimal.{md,csv}、summary_v3.csv、quick_recovery_comparison.{md,csv}、append_w2_diagnosis.json、server_{6_34660,7_34670}.log、logs/(14 个 launcher log + manifest_write.log)。

### project/kvoffload-sglang/
git HEAD `d42ce815f`,分支 `c2kv-sglang-bfcl`(branch -a 输出仅此一支)。删前 dirty(09-05 12:16 复制时刻):`scheduler.py`、`session_aware_cache.py`、`qwen3.py` 三个 M——**这三个文件的完整内容保存在 $T/sglang 与 srcnap**。更早(09-04 晚)的 16 文件 dirty + untracked history_kv_eviction.py 状态有全量 -U10 diff 存档(sd_diffs.txt + raw/*.diff)。

### project/c2kv/
- `checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`(≈11G)。
- `models/Qwen3-4B-Instruct-2507`(公开 HF 模型)。
- `share/d-kv-repair/`:d_neutral_corpus.txt、d_sham_plan.py、d_witness_core.py(sweep 默认路径引用;与用户自己 D-line 产物可能同源,可从用户 repo 找)。

## 2. 我有的备份(逐项)

| 内容 | 位置 | 说明 |
|---|---|---|
| 0903 run 四臂全数据(c2kv/append_w2/replace_w2/append_masked_w2 的 details+score)+ diagnosis | 本地 repo `inv_ur52/payload/run/`(已推 GitHub 275029e) | details 为 gzip 全量 |
| v3 run 两臂(sham_mech/c2kv)details+score | `inv_ur52/payload/v3/` | sham 逐题对照的证据数据 |
| 0903 run 全部表格/manifest/**14 个 launcher log 全文**/臂目录结构 | `inv_ur52/raw/s0_files.txt`、`s0_layout.txt`、`unified_recovery_comparison.json` | 结构与配置可完整复原叙事 |
| server_args 全文×2、abort grep 矩阵、pool 行为摘录 | `inv_ur52/raw/sb_serverlogs.txt` | 原两份 server 日志仅存这些(日志本体已删) |
| kvoffload 09-04 时刻 16 文件 dirty 全量 diff(-U10) | `inv_ur52/raw/sd_diffs.txt` + 16 个 .diff | 含 history_kv_eviction.py(untracked) |
| **as-run 客户端整包**(bfcl_eval 141 py 含 data/ + c2kv_eval,含 tools 补丁) | `$T/client/`、`$T2/client/`(/tmp,tmpfs,**重启即失**) | bfcl 仓两个核心包的删前状态 |
| **as-run sglang 整树**(1674 py,d42ce815f + 他删前 3 个 dirty 文件,零仪表化) | `$T/sglang/python/`(tmpfs) | 恢复他仓库工作树的最佳源 |
| frozen reference(correct_ids.txt + details.jsonl,原 0819 run 唯一幸存数据) | `$T/inputs/` + 本地 `raw/toolsnorm52/snap/`(details.jsonl.gz) | 已双备份 |
| 52 轮三臂全产物 + `_192` 全部实验数据(含 18G KV dump) | `$T`、`$T2`(tmpfs)+ 摘要已推 GitHub | dump 部分只在 tmpfs |
| srcnap 22 文件(含 dirty 版 run_bfcl_kv_repair_sweep.sh、顶层 zh_exp_run52.sh 原件、master log) | 本地 `raw/toolsnorm52/snap/`(已推)+ /tmp 冗余 tgz | |
| **checkpoint-1088(11G)** | `/home/liuyancheng/checkpoints_upstream/checkpoint-1088`(磁盘,完好) | 与被删那份同源副本 |
| kvoffload-sglang git 历史 | GitHub(`d42ce815f` 可拉取) | 分支仅 c2kv-sglang-bfcl,无未知本地提交的迹象 |
| bfcl 仓其余部分 | GitHub(gorilla 上游;他的 fork 是否含 1b6f377 未验证) | 覆盖后用 $T/client 恢复两包 |
| Qwen3-4B-Instruct-2507 tokenizer/model | HuggingFace 公开 | |
| 本会话全部 verbatim 抓取输出 | 会话记录 + `inv_ur52/raw/*` | 上表各"来源"即指此 |

## 3. 确认没有备份、真丢失的

1. **miniconda3 两个环境**(bfcl、sglang;含 torch_npu/CANN 接线)——只能重建;
2. bfcl 仓 4 个 dirty 文件(compare_kv_repair_sweep.py、run_bfcl_fixed_depth_sweep.sh、run_bfcl_history_kv_baselines.sh、run_bfcl_history_multistep_checkpoint.sh 的改后版本)与 5 个 untracked 新文件(stage1/2、bfcl_task_oracle.py、compare_full200_unified.py、results/)——srcnap 只存了 5 个 dirty 中的 1 个;
3. bfcl 仓**可能存在的未推送 commit/stash**(1b6f377 是否在远端未验证;本地 .git 已删);
4. **bfcl_runs 其余 ~44 个 run 目录**(含 0903 的另外 10 臂 details/score、phaseB、quick_recovery、两个 npu04567 变体、npu56、tooldef_hardneg、closed_loop 全量、combined_logistic_v2 的 23:18–00:12 产出)——我只留了 4+2 臂与全 run 的表格/launcher 级记录;
5. `project/c2kv` 训练仓本地状态(share/d-kv-repair 三个文件若与用户 repo 不同源即丢失;1088 以外如有其他 checkpoint 则丢失——未见过清单,无法确认);
6. `.bash_history` 及 .claude/.codex/.copilot/.vscode-server 的会话状态。

## 4. 紧迫事项

- **$T/$T2 在 tmpfs,重启即失**——它们是 bfcl 两包与 sglang as-run 树的唯一完整副本,需立刻落到磁盘(建议 /home/liuyancheng/ 下,待批准)。
- 通知 Tracy;若服务器有家目录快照/备份,§3 清单可大幅缩短。
