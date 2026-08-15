# R4 勘误与零算力闭合清单

## E1. r3 探针 prefill chunk 值三方矛盾 → 裁定 512

三处记录：`configs/r3_run_config.json` 记 128、PR#6 §5 记 512、`agent/r3_chunked_prefill_probe.py` argparse 默认 2048。

**裁定证据（产物优先）**：`outputs_lyc/r3_discrimination/t_a/t_a_chunked_probe.jsonl` 逐例行记录 `"kernel": "eager, chunk=512"`（2/2 例）。探针脚本的 kernel 字段在运行时按实际传入值写入，是一手产物；`r3_run_config.json` 的 "chunk=128 after 2048/512 OOMed" 系事后整理时的误记（512 实为成功值，非 OOM 值）。PR#6 §5 与产物一致。

**R4 口径**：任务 A 全池 eager 分块 prefill 使用 **chunk=512**，OOM 预授权阶梯 512→256→（再 OOM 停报）。

## E2. r3 sglang 可信臂完整启动命令归档

服务器无 shell history（~/.bash_history 不存在）；启动命令以 `~/serve_sgl.sh` 为准，已与 `outputs_lyc/r3_discrimination/t_a/sglang_server.log` 的实际启动行核对一致。归档如下（不要求重启）：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /home/liuyancheng/c2kv-r3
ASCEND_RT_VISIBLE_DEVICES=${SGL_DEV:-1} \
/home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \
  --model-path /home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen3-4b --model-impl sglang \
  --device npu --attention-backend ascend \
  --dtype bfloat16 --mem-fraction-static 0.7 \
  --context-length 131072 \
  --host 127.0.0.1 --port 30000
```

服务端实际推算（log 实录）：`max_total_num_tokens=251904, chunked_prefill_size=8192, max_prefill_tokens=16384`。

## E3. chunk 计数 147 vs 148 → 已解决（无 147）

对 `t_e_c2kv_r4.jsonl` 全 48 行机器核验：`doc_chunks == ceil(doc_tokens/512)` 逐行成立，0 例不符；实测 doc_chunks 范围 **148–157**（doc_tokens 75327–80171）。PR#6 §5 的 "147–157" 下界系笔误——ceil(75327/512)=148，147 在任何一行都不存在。无未决项。

## E4. PR#1 "32k 中池" 标签与档案不符（report-only）

PR#1 四臂档案（merged_{A,B,C,D}.jsonl）实测为 history 压缩 regime：doc_tokens≡kept_history_tokens，中位 2313、最大 9271（选择器上限 max_history_tokens=12288）。任务书 "32k 中池 regime" 的标签与档案不符；R4 任务 D 一律以档案实测 regime 为准（详见 `r4_d_qids.json.regime`）。

## E5. r3 归档缺 source_sha256（本包补代码断言）

r3 的冻结 input_ids 归档（`t_a_prompts.jsonl` 及 48 例版）只记 n_tokens，未记 source_sha256；sha256 绑定存在于 `configs/r3_s1_48_qids.json`（source_sha256=dd31825b…，指向 `outputs_lyc/r2_bigpool/s1_full_48.jsonl`）。本轮由 `agent/r4_assert_inputs.py` 补逐 qid 机器断言（n_tokens + 源文件 sha256 + 与归档逐 token 一致）。
