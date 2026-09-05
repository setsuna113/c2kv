# srcsnap 回取清单(只读,来源 = $T=/tmp/c2kv-toolsnorm52.NIhbBb + 本轮实际顶层脚本)

取回时间:2026-09-05 22:22–22:30(服务器端 tar 打包后 scp,未修改 $T 任何文件)。
逐文件 md5 见 `../srcsnap_md5.txt`(与服务器端 `md5sum` 输出逐行核对一致;含两个顶层脚本的原件校验)。
目录保留服务器绝对路径结构(`tmp/c2kv-toolsnorm52.NIhbBb/...`),便于与 $T 及原仓库 diff。

## 客户端(实际运行版,含补丁)
- `tmp/.../client/c2kv_eval/adapters/bfcl_history_drift.py` — **patched**(_normalize_tool_for_sglang;md5 111f5bc0…)
- `tmp/.../client/c2kv_eval/adapters/bfcl_history_kv_repair.py` — copy 时的 dirty 版(md5 f4a3ef66…)
- `tmp/.../client/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh` — copy 时的 dirty 版(md5 76097f1c…)
- `zh_exp_run52.sh` — **本轮顶层启动脚本原件**(服务器 /tmp 幸存,非重建;md5 2d149d86…):三臂 for 循环 + 用户 env 块,`PORT=34770`(注意:文件内 DEVICE=6/PORT=34770 为 sed 替换后的最终执行版)

## 服务端(实际运行版 = copy 时 d42ce815f + 当时 dirty,无仪表化)
- `tmp/.../sglang/python/sglang/srt/models/qwen3.py`(md5 d7ce1796…)
- `tmp/.../sglang/python/sglang/srt/managers/scheduler.py`(md5 c07315ee…)
- `tmp/.../sglang/python/sglang/srt/mem_cache/session_aware_cache.py`(md5 7faec21f…)

## 日志
- 三臂 server 日志:`tmp/.../runs/{c2kv,d_corr_w2,d_corr_replace_w2}/server_6_34770.log`(每臂独立 server 代)
- 三臂 runner/eval:`tmp/.../runs/$arm/$arm/logs/{runner.log,eval.log}`
- 顶层 master:`zh_run52_master.log`(三臂 rc=0 时间线)
- (三臂 launcher.log、details/score/metrics/summary 已在上一 commit `raw/toolsnorm52/` 下,不重复)

## inputs 与记录
- `tmp/.../inputs/correct_ids.txt`(52 ID)、`inputs/details.jsonl`(frozen reference,已 gzip)
- `tmp/.../preflight52.py`(PASS 输出录于 TOOLS_NORM52.md:52/52 结构 + drop 1697/1150/1670)
- `tmp/.../PROVENANCE.md`、`tmp/.../c2kv_tools_norm_review.patch`(= 实际应用补丁字节)

## 说明
- 未跑任何模型;未用现原仓库文件替代任何内容(原仓库当时已再现新 dirty,见 PROVENANCE 记录的 copy 时刻状态)。
- 服务器上 `/tmp/toolsnorm52_srcsnap.tgz` 为本次打包产物(临时),$T 本体未动。
