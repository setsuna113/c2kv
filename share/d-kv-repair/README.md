# D-KV-Repair 共享包（restore-vs-sham KV 修复 · 信号层）

C2KV gist 压缩后 KV cache 的 in-place 修复实验核心。设计一句话：对"压缩压错"的触发点（C→W transition，压缩前对、压缩后错），用三个修复臂对两个对照臂，在同一失败集上测修复率与噪声地板。

本包刻意精简：只有臂实现 + eval 接入参考 + 两个数据无关工具。AppWorld 侧的 driver、冻结 manifest、行级结果、预注册全文是 harness/数据特定的，未包含（需要时可单独聊）。

## 臂定义表

| arm | harness mode | 注册 | 含义 |
|---|---|---|---|
| E-none | `c2kv` | yes | 未动的压缩前缀（下界） |
| E-sham | `d_sham_neutral` | yes | 等长中性 span 走同一注入路径（噪声地板） |
| E-corr | `d_corr` | yes | append-only erratum：doc k* 的 raw KV 追加到 full-grid gist 后（双重覆盖） |
| E-corr+re | `d_corr_recompute` | yes | docs 0..k* gist + 同一 raw slice + docs k*+1..T-1 在修正前缀上重算 |
| E-full | `full` | yes | 未压缩上界 |
| (诊断) | `d_corr_all` | **no** | 追加所有 doc 的 raw KV；只回答"append-only 通道是否活着" |
| (守卫) | `d_sham_mech` | **no** | 机械拆装重组；必须与 `c2kv` 逐 token 一致 |

两个关键设计点：

- **所有臂的 decode 位置完全一致**（会计字段里 `d_*_prefill_sec` 只是注入侧成本，不影响打分位置）。
- **single-variable claim**：`d_corr` 与 `d_corr_recompute` 的上游半边位级相同（网格行是压缩确定性的），两臂唯一变量是下游表示。

## 文件

| 文件 | 说明 |
|---|---|
| `kv_repair_arms.py` | 臂构造参考实现（292 行，从 `agent/eval_agent_history_c2kv.py` 原样提取 + 接口契约头） |
| `extract_cw_triggers.py` | C→W trigger 提取（纯 stdlib，零三方依赖） |
| `d_sham_plan.py` | sham/corr 修复计划生成（纯 stdlib） |
| `d_neutral_corpus.txt` | sham 臂中性语料（sha256 绑定进 plan） |

## 接 BFCL 的三步流程

**Step 1 — 两臂电池。** 用你们的 BFCL runner 在同一 qid 集上跑 `full` / `c2kv` 两臂，产出行级 jsonl（每行至少：qid、输出文本、correct 判定；行格式约定见 `extract_cw_triggers.py` 头部 docstring）。本仓库 `metrology/bfcl_hf_runner.py` 已有 c2kv 臂（含 doc_mode/ratio 参数与 untrained 对照），可作起点。

**Step 2 — 提触发点 + 生成修复计划。**

```bash
python extract_cw_triggers.py \
    --full_rows full.jsonl --compressed_rows c2kv.jsonl \
    --batch bfcl_r1 --s_metric tool_name_match \
    --out_manifest cw_manifest.json --out_doc_table d_doc_ids.json \
    --ckpt_path <ckpt> --model_sha <sha> --eval_code_sha <git_sha>

python d_sham_plan.py \
    --manifest cw_manifest.json --doc_table d_doc_ids.json \
    --corpus d_neutral_corpus.txt \
    --tokenizer <model_path> --out d_sham_plan.json
```

`--s_metric` 按 BFCL 的判定方式选择（可选值见 `--help`）；plan 自带 budget/neutrality 两道门，门不过会拒绝产出可用计划。

**Step 3 — 接臂。** 把 `kv_repair_arms.py` 的 `_build_d_intervene_prefix` 挂到 runner 的 prefix 构造分流处，运行前把 `d_sham_plan.json` 注入 `D_INTERVENE`。宿主侧需要提供的接口（现成 harness 里都有对应物）已列在该文件头部契约里。

**接入后先跑守卫臂**：`d_sham_mech` 的输出必须与 `c2kv` 基线臂逐 token 一致。不一致 = 手术管线有机械损伤，先修管线，再谈任何修复率数字。

## 权重获取

D 修复的是 c2kv gist 压缩后的 KV，需要 c2kv 训练权重（Qwen3-4B-Instruct-2507 基座）：

- 我们训好的 checkpoint 在 NPU 服务器（8×910B3），路径找言成要；
- 或按 C2KV 官方配方自训：120K 样本（3×40K）/ 1 epoch / global batch 32 / LR 5e-5 / warmup 0.06 / wd 0.1（论文 2607.17715 §Training Setup；注意 README 示例写 2 epochs 与论文矛盾，以论文为准）。

## 范围与分工

本包是 C 线"信号层"（哪里该修 / 怎么修 / 怎么设对照）。"成本层"（修复的延迟与显存账、downstream persistence）在另一条线，代码与预注册都在，需要可另聊。
