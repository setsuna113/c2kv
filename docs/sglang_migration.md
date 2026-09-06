# hf_server → SGLang c2kv fork 迁移记录

> **2026-09-02 更新**：serving 树改为 fork 分支 `task/c2kv-serve-align`
> （基于雨晗 `c2kv-sglang-bfcl` @ 7de9e8105，已含本文件后面描述的 4 文件补丁，
> `benchmarks/backends/sglang_patches/` 作废）。启动时加
> `--c2kv-query-proj gist`（训练一致的投影；`base` 只用于 A/B）。proxy 默认
> `--doc-packing turn`。语义与口径见 `docs/c2kv_semantics.md` 和 server 侧
> `c2kv/c2kv_serving_semantics.md`；上线前先跑
> `scripts/c2kv/smoke_c2kv_semantics.py --base-url http://127.0.0.1:PORT`。

Branch `task/bench-recover`。决策(刘言成 2026-08-30):O-1 选 (b) 直接接受
SGLang regime、不做 A/C 生成对拍闸门、bench 数字全部重基线;O-2 跟雨晗的
切法(整条消息不切);O-3 与雨晗重合不管。

## Phase −1 翻案验证(2026-08-31 04:13-04:16,PASS)

**结论:08-27 "SGLang 跑不了这个 NPU 栈"的定论被推翻。** ckpt-1088 在
c2kv-sglang-bfcl 线上完整服务,三项冒烟全过:

| 冒烟 | 结果 |
|---|---|
| `POST /v1/c2kv/extract` | `success=true, key_hash, gist_len=7, original_seq_len=55` — 与 hf_server 完全同形 |
| chat + tools(full 路径) | 正确 `tool_calls`(search_flights 参数全对),finish_reason=tool_calls |
| chat + `c2kv_key_hash`(压缩路径) | 正确读取 gist 内容;server 日志 c2kv pool 7/4437、#c2kv-entry=1、kv_committed_len=154 |

### 走通的三个必要条件(也是当年失败的原因链)

1. **正确的分支**:`origin/c2kv-sglang-bfcl`(雨晗线)。当年 08-27 打补丁的
   是 `c2kv-v0.5.10`(07-15 的老基线)——本地 checkout 停在
   `c2kv-v0.5.10 = dedffc723 + 我们的 7 个 compat 提交`,而 de-CUDA/NPU
   移植在另一条线上。两线共享基座 dedffc723。
2. **compat 移植**:新分支的 `qwen3.py:69` 硬 import
   `sgl_kernel_npu.norm.split_qkv_rmsnorm_rope`,依赖
   `triton.language.extra.cann`(本机 triton-ascend 3.2.0 没有)→ **模型
   文件被注册表静默忽略**("Ignore import error")→ 注册表里没有
   Qwen3ForCausalLM → 落到 TransformersForCausalLM 兜底 →
   `ValueError: No module or parameter named 'model.gist_embed_tokens'`
   → 子进程 sigquit 全体 Killed(第一手错误原文存
   `~/bench_logs/sgl_35000.log` 第一段)。修法 = 把我们 08-27 的
   `27f21a588`(optional split_qkv + native fallback)手工移植到新树
   (import guard + decode 分派点),注册表即通
   (`Qwen3ForCausalLM in registry: True`)。
3. **`--disable-cuda-graph`**:NPU 图捕获失败
   (`AclrtSynchronizeStreamWithTimeout 107027 — "Not allow to synchronize
   captured-stream"`,疑似与 native decode 回退在捕获期同步有关)。验证
   阶段关掉;性能损失后续量化,必要时再修。

### 部署形态

- 树:`~/sgl-22fbf3146/`(codeload tarball + 2 个补丁:hand-port 27f21a588
  + clamp debug)。git 对象暂时拉不到(github.com 主站不通,codeload 走
  squid 代理可用)——分支 checkout `~/kvoffload-sglang-c2kv` 已切到
  `c2kv-sglang-bfcl`(停在本地引用 4d08b7b92,等网络恢复 fetch 22fbf3146)。
- 启动:`~/bench_logs/sgl_deploy/launch_sgl1088.sh`(dev3 :35000,
  `--model-impl sglang --device npu --attention-backend ascend
  --tool-call-parser qwen25 --enable-c2kv --mem-fraction-static 0.30
  --disable-cuda-graph`,served-model-name=c2kv-agent;PYTHONPATH 前置
  tarball 树覆盖 editable 安装,不动 git checkout)。
- 补丁与脚本存档:`~/bench_logs/sgl_deploy/`。
- ckpt-1088 config 已核实 `gist_overlap=64 / gist_param=qkv /
  gist_residual_type=embed-mean / gist_type=dynamic-interleave` ✓。

### 语义决定(O-1b/O-2 已拍板,见上);已知运行期约束

- ratio-per-server(gist key=sha256(input_ids) 不含 ratio,二次 extract
  被静默忽略);
- 失败全是 HTTP 200:body `success=false` / `FINISH_ABORT` /
  `C2KV cache miss`(LRU 驱逐后)→ proxy 重试与 terminal_check 要显式识别;
- `metadata.sglang_runtime` 免费提供 kv_resident_tokens /
  kv_peak_resident_tokens / kv_pool_size(成本列数据源);
- `c2kv_repair_only_key_hashes` 不存在于本 fork(repair hashes 只在挂了
  c2kv_key_hash 的 message 上被读)→ "只 repair 不压缩"的臂表达不了
  (现有臂均压缩,无影响);
- Ascend 非 page 对齐 prefix 静默降级 native SDPA(`[C2KV HYBRID BRIDGE
  GUARD]`)→ 跑前 `C2KV_DEBUG_ASCEND_ATTN=1` 确认;
- 待验:`schedule_batch.py:2036` 用 `c2kv_position_correction > 0` 判 c2kv
  请求,repair 会把它往下减,减到 ≤0 走错分支。

## Phase 1-4

见 docs/research/hybrid_x_d.md §5 与本文件后续增补。

## 遗留物处置记录(2026-08-31)

- **`fix/c2kv-segment-offset-tools`(737974315)**:功能性已被 22fbf3146 收编
  ——目标树的 `serving_chat.py` 含 `_chat_template_tools` +
  `_c2kv_chat_template_input_ids(..., tools=tools)`(242/315-336 行),
  即本分支所修"tools 不进 segment 插入点"的成熟版。判定:过时,
  远程分支可删;删除因 GitHub 网络抖动暂缓(本地 `git push origin
  --delete fix/c2kv-segment-offset-tools` 待重试)。
- **z4_f3_bfcl_cd_c2kv_4186**:并行会话的任务,保持 `~/bench_queue/delayed/`,
  由其所有者决定(其 4186 ckpt + 旧 hf_server 栈在迁移后是否仍要跑)。
- **fork 同步**:github.com 主站从服务器不可达(codeload 走 squid 可用),
  git fetch 暂无法执行。部署以 tarball(22fbf3146)+
  `benchmarks/backends/sglang_patches/`(入库)为准,完全可复现;
  网络恢复后:checkout fetch → 在 c2kv-sglang-bfcl 上提交同内容三补丁 → push。
- **hf_server 退役**:README 已改(8bc3a30),`backends/hfserver.py` 保留为
  对照后端;`benchmarks/hf_server.py` 不再进评测路径。
