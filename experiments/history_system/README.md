# A: Prefill-guided event recovery

当前算法为 **D3 / C1000 / ratio8**，配置见 [current_algorithm.json](configs/current_algorithm.json)。
默认 controller、Prefill head、特征层及预算来自已选 D3 冻结包，直接随源码保存；不需要从 `tmp/` 或历史评测输出拼装算法。

## 从哪里读代码

| 职责 | 源码入口 |
| --- | --- |
| 默认运行入口和配置 | `current.py`、`configs/current_algorithm.json`、`runtime/configs/` |
| 一次决策：prepare → draft → 检测/恢复 → 最终提交 | `runtime/benchmarks/memory_runtime/event_native_step.py` |
| Recovery 协调及检测、选源、准入模块 | `runtime/benchmarks/memory_runtime/event_native_recovery.py`、`runtime/benchmarks/memory_runtime/recovery/` |
| 历史事件、raw/gist 表示与打包 | `runtime/python/history_memory/events.py`、`packing.py` |
| C0 基础工作区与 same-event bridge | `runtime/benchmarks/memory_runtime/event_native_s0_policy.py`、`same_event_bridge_only.py` |
| 实际生成前预算检查 | `runtime/benchmarks/memory_runtime/budget_guard.py` |
| C2KV 编码、增量推理、Prefill 特征与缓存 | `runtime/python/history_memory/inference.py`、`shadow_features.py`、`runtime.py` |
| 服务与 benchmark 适配 | `runtime/benchmarks/memory_runtime/event_native_server.py`、`multibench/` |
| 官方结果和成本汇总 | `collect_results.py`、`compression_metrics.py`、`reporting/` |

一个 decision 先生成 held draft，从实际 Prefill feature 计算风险；触发且额度允许时，按当前目标与 draft 检索一个完整历史 event，重新检查 B0 准入。准入成功才再生成一次并提交第二份动作；无触发、无来源、预算或额度不足时提交原 draft。不会递归恢复，也不读取 gold 或未来轨迹。

算法默认值以 JSON 为准：ratio8，history/workspace budget 各 113246208 bytes，每题最多 96 次 generation（含 draft 与 regeneration）和 1152 次 encoder 调用；恢复累计额度为 `ceil(t/5)`。这些是运行参数，不是性能结果。

## CPU 验证与服务入口

从本仓库根目录执行：

```powershell
$pyA = 'C:/Users/yl998/scoop/apps/miniforge3/current/python.exe'
& $pyA -B experiments/history_system/validate.py --candidate current_d3
& $pyA -B experiments/history_system/current.py preview --checkpoint C:/path/to/checkpoint-1000 --out outputs/manual_d3 --task-id multi_turn_base_0
```

`preview` 只打印命令。NPU 环境准备好后，将 `preview` 换为 `serve` 才会启动服务；checkpoint、Ascend 依赖和官方数据集仍需单独提供。该便捷入口使用 BFCL session 协议；其余 benchmark 使用 `multibench/` 适配与原冻结任务 manifest。

`freeze.py` 和 `multibench/freeze_suite.py` 默认绑定同一 D3 配置。`runner.py` 只执行显式冻结的 manifest。历史候选/队列记录保留，但本地 `advance.py --dispatch-next` 已禁用；没有待自动启动的新实验。

## 源码与历史产物

`runtime/` 是 A 当前唯一活跃算法源码。顶层 `python/`、`benchmarks/` 是共享模型与历史评测支持，D3 入口通过独立 `PYTHONPATH` 使用这里的 runtime。

已发布快照保留在 `releases/a_history_20260914/`，原始运行、失败和官方结果保留在 `outputs/`；这些是本地产物，不加入源码提交。旧运行继续使用各自冻结包，不把本次源码整理当成新的 benchmark 成绩。

后续未选用的 goal/task-packet、demand encoding、raw/gist 去重、预算可行选源和 source-record 原型已从当前 D3 实现撤出。对应既有评测记录仍用于历史追溯。交付结果是否完整见已发布 snapshot 的 `report.md`，不以源码验证代替模型评测。
