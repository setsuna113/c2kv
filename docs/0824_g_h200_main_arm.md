# G-H200 主 checkpoint 臂（task/g-h200-main）— 2026-08-24

**分支/基线**：`task/g-h200-main`，从 G 主线 `fork/task/g-joint-c2kv` tip `9aebbfe`（含 capfix `9a1dffc`）切出。主 worktree 的 `npu-fusion-attention`（D/B/F 线）不受影响。
**性质**：两份外部批评裁定的执行层落地——①训练从 8×Ascend 910B 移植到 2×H200；②G-medium 的数据配方与训练实例构造按"追求 BFCL 精度"修订。本文是 runbook + 记账基准；设计权威仍是 26 号手册 v3（§3-G）与 24 号 G 章，冲突处以手册为准。

## 1. 臂定义

| 项 | 值 |
|---|---|
| Base | Qwen3-4B-Instruct-2507 |
| 可训练参数 | gist sidecar only（`--only_train_gist True`，base 全冻结） |
| 初始化 | init gate 二选一：G8-small-v2 warm start vs 新鲜 gist init（见 §4） |
| 数据 | **60% Toucan + 30% τ² traces + 10% AppWorld**（按 estimated source tokens 配比；QA / OpenSWE 不进 recipe） |
| 实例构造 | 每 assistant decision point 一个实例；`max_samples_per_session=4` 按 early/middle/late = 1/1/2 分层；`REQUIRE_TOOL_CALL=False` + action-balanced（tool_call 目标占比 0.75）；tool-call target 放不下整条丢弃（`tool_call_target_truncated`），不再训练残缺 JSON；capfix 的 target-tool schema 保证不动 |
| Objective | 单一 next-action CE（context 全 -100，只在 assistant target 计 loss）；无 KL / distill / reconstruction / curriculum |
| 压缩 | `DOC_MODE=joint`，固定 8×（`C2KV_GIST_TRAIN_RATIOS=8`） |
| 优化 | LR 5e-5，cosine，weight decay 0.1，warmup_ratio 0.04，BF16 |
| Batch | per-device 1 × grad-accum 4 × 2 卡 = effective 8（microbatch 2 放得下则 2×2，eff-8 不变） |
| 硬件 | 2×H200（141GB），torchrun DDP / ZeRO-2（`configs/ds_config_h200.json`，无 CPU offload） |
| 预算 | 72 GPUh = 36h wall；目标 64M–96M **presented** source tokens（不以 epoch 为停止条件） |

## 2. 与 v3 手册的关系

- 本臂取代 d_multi（20% QA + 50% traces + 25% Toucan + 5% OpenSWE，24 号 :1170/:1354）成为 §3-G 问题②"多源是否改善泛化"的多源方；traces-heavy 对照由既有 G8-small-v2 自然承担，不再重复训练单源臂。
- G8 绝对阈值冻结判据不变；**预注册线数值须在 medium 判读前写入 prereg**（v3 §3-G 机制照行）。
- 记账纪律照 v3 §2.2：nominal/realized 双口径；`train_manifest_used.json` 新增 `action_type_counts` 与 `tool_call_target_truncated_skips` 两列入账。

## 3. Runbook（服务器侧）

```bash
# 0. 前置：CUDA 环境（torch+transformers+deepspeed，不装 torch_npu）；
#    落位 Qwen3-4B-Instruct-2507、Toucan SFT parquet、agent-llm-traces v2；
#    拷贝 G8-small-v2（NPU 服务器 ~/c2kv/outputs_lyc/g_joint/fixed_joint，sha256 669502d3… 校验）。

# 1. 前置扫描：确认 traces parquet 里 τ² 子集的真实命名（必要时用 --traces_subset_map 钉死映射）
python agent/build_joint_medium_plan.py \
    --traces_path ~/c2kv/datasets/agent-llm-traces \
    --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
    --removal_files outputs/cross_dataset_dedup.json \
    --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
    --out_dir outputs/joint_h200_plan --list_traces_subsets

# 2. 规划 60/30/10 order file（traces 0.4 × 内部 75/25 = 全局 30/10）
python agent/build_joint_medium_plan.py \
    --traces_path ~/c2kv/datasets/agent-llm-traces \
    --toucan_path ~/c2kv/datasets/toucan \
    --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
    --recipe g_h200_main=toucan:0.6,traces:0.4 \
    --split_traces_subsets \
    --subset_weights traces:tau2=0.75 --subset_weights traces:appworld=0.25 \
    --budget_estimated_tokens <N> --oversample_factor 1.25 \
    --removal_files outputs/cross_dataset_dedup.json \
    --order_seed 42 --out_dir outputs/joint_h200_plan \
    --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507

# 3. ρ 重测（presented/estimated，旧值 0.392 是 traces-only 小臂口径）：
#    用 agent/measure_arm_psrc.py 实测新 mixture 的 ρ_new；
#    MAX_SOURCE_TOKENS = presented_target / ρ_new（ρ=0.392 时 64M→≈163M，96M→≈245M estimated）。

# 4. Init gate：两张卡各跑一支 3–5M presented 短跑（同一 launcher，单卡 CUDA_VISIBLE_DEVICES=0）
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<G8-small-v2>  EXAMPLE_ORDER_FILE=outputs/joint_h200_plan/g_h200_main.order.json MAX_SOURCE_TOKENS=<small> bash agent/train_joint_next_action_c2kv_h200.sh
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=<Qwen3-4B-Instruct-2507> EXAMPLE_ORDER_FILE=同左 MAX_SOURCE_TOKENS=<small> bash agent/train_joint_next_action_c2kv_h200.sh
#    同一 dev manifest 评分，胜者 2 卡 RESUME_FROM_CHECKPOINT 续跑，败者弃。

# 5. 主训练：先 100–200 step 校准（sec/step、presented tokens/s、peak HBM、
#    tool_call_target_truncated 丢弃率）→ 回填 SAVE_STEPS 使每 ≈16M presented 存一档
#    （16/32/48/64/80/96M）→ 正式跑，目标 96M presented，保底 64M。
CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=<胜者 init> EXAMPLE_ORDER_FILE=同左 MAX_SOURCE_TOKENS=<折算值> SAVE_STEPS=<校准值> bash agent/train_joint_next_action_c2kv_h200.sh

# 6. Dev 评测选 checkpoint（不看 train loss）
python agent/build_bfcl_dev_manifest.py --bfcl <bfcl_pkg 或 multi_turn_base jsonl> --n 128 --seed 42 --out configs/bfcl_dev_v3_mt.json
CKPT=<档> BFCL_PKG_PATH=<pkg> BFCL_DATA_DIR=<dir> bash agent/eval_bfcl_dev_c2kv_h200.sh
```

时间分配：前 ~2h 校准（计入正式训练）→ 30–32h 主训练 → 2–4h dev 评测。C_general 出分后再决定是否做 BFCL-derived adaptation tail（条件项，不在本臂）。

## 4. 代码变更清单（本分支相对 9aebbfe）

| 文件 | 变更 |
|---|---|
| `python/train/train_data_joint.py` | decision point early/mid/late 分层（`_stratified_pick`）；`action_type` 标记 + action-balanced（`action_tool_call_frac=0.75`，仅 `require_tool_call=False` 时生效，True 旧行为逐位不变）；tool-call target 超预算整条丢弃；answer 抽取处剥 `<think>` |
| `python/train/train_data_joint_multisource.py` | Toucan/OpenSWE core 同步剥 `<think>` + 标 `action_type` |
| `agent/train_joint_next_action_c2kv.py` | `JointDataArgs.action_tool_call_frac`；manifest 增 `action_type_counts` / `tool_call_target_truncated_skips` |
| `agent/build_joint_medium_plan.py` | traces 拆 `appworld`/`tau2` substrata（`--split_traces_subsets` / `--traces_subset_map` / `--subset_weights traces:*`）；`--list_traces_subsets` dry-run；默认无 split 时旧 recipe 输出逐位不变 |
| `agent/train_joint_next_action_c2kv_h200.sh`（新） | 2×H200 launcher：CUDA + flex_attention + torchrun，ZeRO-2，LR 5e-5 / eff-8 / warmup 4% / fixed 8× / joint / REQUIRE_TOOL_CALL=False；头部注释写明 SAVE_STEPS 与 ρ 折算纪律、init-gate 用法 |
| `configs/ds_config_h200.json`（新） | ZeRO-2、无 offload、bf16 |
| `agent/build_bfcl_dev_manifest.py`（新） | BFCL V3 multi-turn dev manifest 生成（seed 42 / n 128 / sha256 冻结） |
| `agent/eval_bfcl_dev_c2kv_h200.sh`（新） | `metrology.bfcl_hf_runner --condition c2kv --ids_file <dev manifest> --device cuda` + `bfcl_score` 薄封装，输出落 `results/g_h200/` |
| `conftest.py`（新） | 主线上方 guarded test bootstrap 回填（本分支此前缺失，torch-free 机器上测试收集用；server venv 下 no-op） |

## 5. 验证状态

- **本机已验证**：`pytest` 119 passed（新增 23 用例：分层配额/确定性/回退、action balance 精确计数与回退、`require_tool_call=True` 回归逐位一致、target 截断丢弃/完整保留、planner 分类/75-25 配额/端到端 recipe/向后兼容）；`bash -n` 两个脚本；planner 与 manifest 生成器 CLI smoke；manifest 生成器对伪 jsonl 确定性复跑逐字节一致。
- **需服务器验证**（本机无 GPU/真数据）：launcher 端到端 smoke + 校准；`--list_traces_subsets` 确认真 τ² 子集命名；`--action_tool_call_frac` 在完整依赖下的 argparse 联通；dev 评测管线跑通一个 checkpoint。
- 唯一已知本机失败：`test_token_accounting.py::test_official_missing_mdoc_data_message`（需 torch，主基线上同样失败，与本分支无关）。
