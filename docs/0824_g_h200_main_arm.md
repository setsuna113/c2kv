# G-H200 主 checkpoint 臂（task/g-h200-main）— 2026-08-24

**分支/基线**：`task/g-h200-main`，从 G 主线 `fork/task/g-joint-c2kv` tip `9aebbfe`（含 capfix `9a1dffc`）切出。主 worktree 的 `npu-fusion-attention`（D/B/F 线）不受影响。
**性质**：两份外部批评裁定的执行层落地——①训练从 8×Ascend 910B 移植到 2×H200；②G-medium 的数据配方与训练实例构造按"追求 BFCL 精度"修订。本文是 runbook + 记账基准；设计权威仍是 26 号手册 v3（§3-G）与 24 号 G 章，冲突处以手册为准。

## 1. 臂定义

| 项 | 值 |
|---|---|
| Base | Qwen3-4B-Instruct-2507 |
| 可训练参数 | gist sidecar only（`--only_train_gist True`，base 全冻结） |
| 初始化 | **新鲜 gist init**（2026-08-25 裁定：G8-small-v2 在 NPU 服务器拿不到，init gate 取消，单臂直接跑） |
| 数据 | **60% Toucan + 30% τ² traces + 10% AppWorld**（按 estimated source tokens 配比；QA / OpenSWE 不进 recipe；traces 的 swebench/browsecompplus 兜底substrata 用 `traces:other=0` 显式排除） |
| 实例构造 | 每 assistant decision point 一个实例；`max_samples_per_session=4` 按 early/middle/late = 1/1/2 分层；`REQUIRE_TOOL_CALL=False` + action-balanced（tool_call 目标占比 0.75）；tool-call target 放不下整条丢弃（`tool_call_target_truncated`），不再训练残缺 JSON；capfix 的 target-tool schema 保证不动。**planner 扫描必须带 `--no-require_tool_call`（池子与 trainer 一致，否则 order file 滤掉全部非 tool-call 目标）** |
| Objective | 单一 next-action CE（context 全 -100，只在 assistant target 计 loss）；无 KL / distill / reconstruction / curriculum |
| 压缩 | `DOC_MODE=joint`，固定 8×（`C2KV_GIST_TRAIN_RATIOS=8`） |
| 优化 | LR 5e-5，cosine，weight decay 0.1，warmup_ratio 0.04，BF16 |
| Batch | per-device 1 × grad-accum 4 × 2 卡 = effective 8（microbatch 2 放得下则 2×2，eff-8 不变） |
| 硬件 | 2×H200（141GB），torchrun **plain DDP**（`USE_DEEPSPEED=0`；ZeRO-3 在 gist 双 pass 之外还有 per-rank generate_gist 调用计数漂移导致的 NCCL 计数错位挂死——2026-08-26 实锤，见 §5；ZeRO-2 双重 reduce 也不行。`configs/ds_config_h200.json` 留作后续修复后的备选） |
| 预算 | **144 GPUh = 72h wall**（2026-08-25 上调）；目标 96M–256M **presented** source tokens（不以 epoch 为停止条件；2026-08-26 实测 ρ=0.62、池子 26.6M presented/epoch，默认剂量 256M ≈10 epoch；更粗的 Toucan 大池备选 `g_h200_bigpool` 见 §3 注） |

## 2. 与 v3 手册的关系

- 本臂取代 d_multi（20% QA + 50% traces + 25% Toucan + 5% OpenSWE，24 号 :1170/:1354）成为 §3-G 问题②"多源是否改善泛化"的多源方；traces-heavy 对照由既有 G8-small-v2 自然承担，不再重复训练单源臂。
- G8 绝对阈值冻结判据不变；**预注册线数值须在 medium 判读前写入 prereg**（v3 §3-G 机制照行）。
- 记账纪律照 v3 §2.2：nominal/realized 双口径；`train_manifest_used.json` 新增 `action_type_counts` 与 `tool_call_target_truncated_skips` 两列入账。

## 3. Runbook（服务器侧）

**一条命令（2026-08-25 起，推荐）**：`bash <repo>/start_h200.sh` —— 无人值守状态机（recon→plan→calibrate→train→eval→select），幂等续跑，状态在 `outputs/g_h200_status/`；交互终端直接运行时自动 nohup 脱离会话（`FG=1` 强制前台；跟踪 `tail -f outputs/g_h200_status/logs/console.log`）。下面的手工步骤即该脚本各阶段的展开（排障时用）。

```bash
# 0. 前置（2026-08-25 已在 yancheng 集群侧完成落位, 详见 .foreman/ref/SOURCES.md）：
#    .venv（py3.12 + torch2.9.0+cu130 + transformers5.8 + deepspeed0.18.1, 离线可用）;
#    models/Qwen3-4B-Instruct-2507、datasets/agent-llm-traces（v1, CDLA）、
#    datasets/toucan（Agent-Ark/Toucan-1.5M 的 SFT/）、.foreman/ref/bfcl_{pkg,data}（gorilla 6ea5797）。
#    初始化 = 新鲜 gist init（G8-small-v2 拿不到, init gate 取消）。

#    注意必须带 --no-require_tool_call（与训练一致），否则池子缺全部非 tool-call 目标；
#    split 名与 builder 默认一致用 taskproxy_disjoint（planner/trainer 两侧同名）
python agent/build_joint_medium_plan.py \
    --traces_path ~/c2kv/datasets/agent-llm-traces \
    --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
    --split_manifest_name taskproxy_disjoint \
    --removal_files outputs/removal_traces_final.json \
    --no-require_tool_call \
    --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
    --out_dir outputs/joint_h200_plan --list_traces_subsets

# 2. 规划 60/30/10 order file（traces 0.4 × 内部 75/25 = 全局 30/10；
#    traces:other=0 显式排除 swebench/browsecompplus 兜底substrata）
python agent/build_joint_medium_plan.py \
    --traces_path ~/c2kv/datasets/agent-llm-traces \
    --toucan_path ~/c2kv/datasets/toucan \
    --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
    --split_manifest_name taskproxy_disjoint \
    --recipe g_h200_main=toucan:0.6,traces:0.4 \
    --split_traces_subsets \
    --subset_weights traces:tau2=0.75 --subset_weights traces:appworld=0.25 \
    --subset_weights traces:other=0 \
    --no-require_tool_call \
    --budget_estimated_tokens <N> --oversample_factor 1.25 \
    --removal_files outputs/removal_traces_final.json \
    --order_seed 42 --out_dir outputs/joint_h200_plan \
    --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507

# 3. ρ 重测（presented/estimated，旧值 0.392 是 traces-only 小臂口径）：
#    用 agent/measure_arm_psrc.py 实测新 mixture 的 ρ_new；
#    MAX_SOURCE_TOKENS = presented_target / ρ_new。

# 4.（已取消）Init gate：2026-08-25 裁定 G8-small-v2 拿不到，单臂新鲜 gist init 直接跑。

# 5. 主训练：先 100–200 step 校准（sec/step、presented tokens/s、peak HBM、
#    tool_call_target_truncated 丢弃率）→ 回填 SAVE_STEPS 使每 ≈16M presented 存一档
#    → 正式跑，目标 256M presented（≈10 epoch），保底 96M。
#    注：若要用 Toucan 大池（toucan:0.85/traces:0.15，traces 绝对量不变，池子
#    ≈70M presented/epoch），改用 g_h200_bigpool order file：
#    ORDER_FILE=outputs/joint_h200_plan/g_h200_bigpool.order.json \
#    G_H200_EXPECT_SHARES=toucan:0.85,traces:0.15 TARGET_PRESENTED_TOKENS=355000000 \
#      bash start_h200.sh
CUDA_VISIBLE_DEVICES=0,1 EXAMPLE_ORDER_FILE=outputs/joint_h200_plan/g_h200_main.order.json \
  MAX_SOURCE_TOKENS=<折算值> SAVE_STEPS=<校准值> bash agent/train_joint_next_action_c2kv_h200.sh

# 6. Dev 评测选 checkpoint（不看 train loss）
python agent/build_bfcl_dev_manifest.py <bfcl_data_dir 或 multi_turn_base jsonl> --n 128 --seed 42 --out configs/bfcl_dev_v3_mt.json
CKPT=<档> BFCL_PKG_PATH=<pkg> BFCL_DATA_DIR=<dir> bash agent/eval_bfcl_dev_c2kv_h200.sh
```

时间分配（144 GPUh = 72h wall）：前 ~1h 校准（计入正式训练）→ ≤62h 主训练 → ~6h dev 评测（milestone 隔档评，双卡 id 分片）→ ~3h buffer。C_general 出分后再决定是否做 BFCL-derived adaptation tail（条件项，不在本臂）。

## 4. 代码变更清单（本分支相对 9aebbfe）

| 文件 | 变更 |
|---|---|
| `python/train/train_data_joint.py` | decision point early/mid/late 分层（`_stratified_pick`）；`action_type` 标记 + action-balanced（`action_tool_call_frac=0.75`，仅 `require_tool_call=False` 时生效，True 旧行为逐位不变）；tool-call target 超预算整条丢弃；answer 抽取处剥 `<think>` |
| `python/train/train_data_joint_multisource.py` | Toucan/OpenSWE core 同步剥 `<think>` + 标 `action_type` |
| `agent/train_joint_next_action_c2kv.py` | `JointDataArgs.action_tool_call_frac`；manifest 增 `action_type_counts` / `tool_call_target_truncated_skips` |
| `agent/build_joint_medium_plan.py` | traces 拆 `appworld`/`tau2` substrata（`--split_traces_subsets` / `--traces_subset_map` / `--subset_weights traces:*`）；`--list_traces_subsets` dry-run；默认无 split 时旧 recipe 输出逐位不变 |
| `agent/build_joint_medium_plan.py`（2026-08-25 补丁） | `--require_tool_call`/`--action_tool_call_frac` 透传进扫描 knobs（修 order-file 池子与 trainer 不一致的致命缺口；默认 True 逐位不变）；`--subset_weights` 允许 0=整层跳过（`traces:other=0` 排除 swebench/browsecompplus），report 记 `skipped_zero_weight` |
| `agent/train_joint_next_action_c2kv_h200.sh`（新） | 2×H200 launcher：CUDA + flex_attention + torchrun，ZeRO-2，LR 5e-5 / eff-8 / warmup 4% / fixed 8× / joint / REQUIRE_TOOL_CALL=False；头部注释写明 SAVE_STEPS 与 ρ 折算纪律、init-gate 用法 |
| `configs/ds_config_h200.json`（新） | ZeRO-3、无 offload、bf16（2026-08-25 从 ZeRO-2 修正：gist 双 pass backward 在 ZeRO-2 下双重 reduce 报错；对齐 ds_config.json/ds_config_npu.json 的已验证 stage） |
| `agent/build_bfcl_dev_manifest.py`（新） | BFCL V3 multi-turn dev manifest 生成（seed 42 / n 128 / sha256 冻结） |
| `agent/eval_bfcl_dev_c2kv_h200.sh`（新） | `metrology.bfcl_hf_runner --condition c2kv --ids_file <dev manifest> --device cuda` + `bfcl_score` 薄封装，输出落 `results/g_h200/` |
| `conftest.py`（新） | 主线上方 guarded test bootstrap 回填（本分支此前缺失，torch-free 机器上测试收集用；server venv 下 no-op） |
| `python/train/trainer.py`（2026-08-25 补丁） | `_system_attn_impl` 的 flex→flash_attention_2 映射改为可配置（`C2KV_SYSTEM_ATTN_IMPL`，默认 `sdpa`）：flash-attn 预编译 wheel 要 glibc≥2.32，本镜像族 2.31 装不上；sdpa 数值等价（system 前缀是普通 causal attention）。gist 路径仍是 flex_attention（纯 torch） |
| `python/train/trainer.py`（2026-08-25 补丁②） | `label_tokens` 日志张量强制 float32（`new_tensor` 继承 labels 的 Long dtype，torch 2.9 下 `mean()` 直接报错——首个 logging step 必炸；修复后 4090 冒烟进入正常训练循环） |

## 5. 验证状态

- **本机已验证**：`pytest` 119 passed（新增 23 用例：分层配额/确定性/回退、action balance 精确计数与回退、`require_tool_call=True` 回归逐位一致、target 截断丢弃/完整保留、planner 分类/75-25 配额/端到端 recipe/向后兼容）；`bash -n` 两个脚本；planner 与 manifest 生成器 CLI smoke；manifest 生成器对伪 jsonl 确定性复跑逐字节一致。
- **需服务器验证**（本机无 GPU/真数据）：launcher 端到端 smoke + 校准；`--list_traces_subsets` 确认真 τ² 子集命名；`--action_tool_call_frac` 在完整依赖下的 argparse 联通；dev 评测管线跑通一个 checkpoint。
- **2026-08-25 集群侧验证**（yancheng 开发容器，RTX 4090 + cu128 代理 venv）：`--list_traces_subsets` 确认真子集命名（tau2_airline/retail/telecom 命中默认 map）；planner 真 tokenizer 全量跑出 60/40 realized（8,811 examples / 42.8M est，budget shrink 0.3565）；planner↔trainer 池一致性实测通过（`--no-require_tool_call` 下 order file 8,811 qid 全命中，无 unknown-qid 报错；`action_type_counts` = tool_call 3,292 / other 5,519——Toucan 全 decision-point 保留，traces 侧 0.75 目标占比经分层采样落在 91%，全局 tool_call 实例占比 37%，靠 loss 结构而非计数平衡，若要更高占比需给 Toucan 加 text-turn 子采样——留作后续 arm 旋钮）；4090 上 flex_attention 的 inductor kernel 需要 128KB smem 超过 sm_89 上限（101KB），H200(sm_90, 228KB)放得下；若 H200 侧仍触发，fallback `ATTN_IMPL=eager`（H200 141GB 上 eager 也能跑，只是慢）。
- **2026-08-25 端到端冒烟通过**：`start_h200.sh`（SMOKE=1 缩小配置，单 4090）全状态机 recon→plan→calibrate→train→eval→select 29 分钟 exit=0：校准 5 步落档→实测 ρ/回填 run_config→断点续跑到 26 步→milestone 瘦身（旧档 8.6G / 最新档 23G 完整可 resume）→BFCL dev 双分片评测出分→FINAL_SUMMARY 选档；幂等重跑全 skip、断档重跑只补未完阶段。生产侧（H200）默认全量配置 + flex_attention + ZeRO-3。
- **2026-08-26 H200 首跑事故与根因**：生产首挂在校准第 0 步 ~30 分钟后被 NCCL 看门狗 SIGABRT(`ALLGATHER_BASE 194M 元素超时`)。根因：ZeRO-3 逐执行 all-gather 参数，而 `process_context_input_ids` 会丢掉全空 doc 槽位、`_generate_gist_for_context_docs` 按有效文档数逐篇压缩——**两个 rank 的 microbatch 有效文档数不同 → generate_gist 调用次数不同 → 集合通信计数错位 → NCCL 超时挂死**。该 joint trainer 此前从未跑过多卡 ZeRO(NPU 各臂均为单卡）。对策：本臂改用 **plain DDP**(USE_DEEPSPEED=0,DDP 容忍 per-rank 变长计算图；4B 冻结 base + gist-only 梯队在 141GB 上绰绰有余）;`start_h200.sh` 加了停滞看门狗（日志 25 分钟无进展→杀掉重试）+ 降级阶梯（lvl1=plain DDP, lvl2=再降 sdpa)，生产状态目录已预置 `attn_fallback_level=1`（直接 DDP 起跑）。ZeRO-3 的正确修复（各 rank 以 all-reduce 对齐有效文档数、按最大值补齐哑迭代并丢弃其输出）留给作者线评估，本臂不依赖。
- 唯一已知本机失败：`test_token_accounting.py::test_official_missing_mdoc_data_message`（需 torch，主基线上同样失败，与本分支无关）——2026-08-25 全量 venv 下 314 passed（含 torch），未复现该失败。
