"""Publish one selected algorithm against external baselines from verified data."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from render_delivery import BENCHMARKS, LABELS, digest, percent, read, render

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/history_system_search/delivery_20260914"
ALLOWED = {"d3_prefill_event", "r002_d3_prefill_event_mixed20_v1"}


def main():
    source = BASE / "internal_report.sources.json"
    data = read(source)
    index = copy.deepcopy(data["current"])
    index["cells"] = [r for r in index["cells"] if r["method"] in ALLOWED]
    assert len(index["cells"]) == len(BENCHMARKS)
    index["extended_metrics"]["cells"] = [r for r in index["extended_metrics"]["cells"] if r["method"] in ALLOWED]
    index.pop("toolsandbox_divergence", None)
    index.pop("detector_calibration", None)
    index["algorithm"]["execution_note"] = "今天交付版固定为 Prefill-guided event recovery，使用 C1000 / ratio8 与已评测的同一控制器。后续内部迭代另行保存。"
    legacy = read(ROOT / "experiments/history_system/configs/delivery_20260914.baseline_sources.json")
    checkpoint = read(ROOT / "experiments/history_system/configs/checkpoint.selected.json")
    original = render(index, legacy, checkpoint)
    algorithm = original.split("## 新系统整题结果")[0]
    algorithm = "\n".join(line for line in algorithm.splitlines() if not line.startswith(("D5尝试", "预算可行选源")))
    lines = [algorithm.rstrip(), "", "## 与其他方法的质量比较", "",
        "所有结果均为 preliminary, n=1。百分比越高越好；ToolSandbox 为 mean similarity，其余为官方质量分数。",
        "",
        "本轮 ours 使用固定 BFCL 20题、tau2 6题、ToolSandbox 8题、ACEBench 8题、AppWorld 168题。legacy 使用 checkpoint-1088 和按 Full 表现筛选的旧子集，BFCL 为23题，其余旧表计划分母分别为6、8、8；旧 tau2 部分方法实际分母尚未查全。跨任务集结果作描述比较。压缩与成本采用下节注明的测量口径。",
        "", "| 方法 | 模型 | BFCL ↑ | tau2 ↑ | ToolSandbox ↑ | ACEBench ↑ | AppWorld ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    ours = {r["benchmark"]: r for r in index["cells"]}
    def score(r):
        return percent(r["official_score"]) if r.get("official_score") is not None else f"待补齐（已回收{r['n_scored']}/{r['n_planned']}）"
    lines.append("| **Ours: Prefill-guided event recovery** | **C1000 / ratio8** | " + " | ".join(score(ours[b]) for b in BENCHMARKS) + " |")
    names = {"full": "Full", "streamingllm_25pct": "StreamingLLM 25%", "h2o_25pct": "H2O 25%", "snapkv_25pct": "SnapKV 25%", "hiagent": "HiAgent", "acon": "ACON", "cacheblend": "CacheBlend"}
    for method, name in names.items():
        values = dict(legacy["latest_cross_method_table"]["scores"][method])
        if method == "full":
            values["acon_appworld"] = index["appworld_full_reference"]["official_score"]
        lines.append("| " + name + " | checkpoint-1088 | " + " | ".join(percent(values.get(b)) for b in BENCHMARKS) + " |")
    lines += ["", "Ours 的 tau2 已结束，未评分题保留在原分母中；AppWorld 仍在评测。Full 的 AppWorld 来自完整168题 native 参考，模型训练数据包含 AppWorld，且采样设置与 ours 不同。", ""]
    for note in index.get("reward_basis_notes", []):
        lines += [f"tau2 已评分的 {note['task_count']} 题官方 reward 均为1，评分采用 DB 和 COMMUNICATE；其中 communicate 检查项为空，未包含 nl_assertions。", ""]
    figure_root = ROOT / "reports/weekly/figures/history_20260914"
    import runpy
    runpy.run_path(str(Path(__file__).with_name("plot_selected_baseline_cost.py")), run_name="__main__")
    figure_data = read(figure_root / "baseline_compression_cost.data.json")
    index["formal_baseline_costs"] = figure_data
    lines += ["## 与正式 baseline 的压缩和成本", "",
        "BFCL，preliminary, n=1。baseline 是旧 F23 / checkpoint-1088，Ours 是本轮 mixed20 / C1000；每个压缩倍数都相对该方法自身轨迹的 Full-equivalent 上下文。整体逻辑 KV 与活动 history 分开计量，均非设备峰值显存。Ours 的 history 倍数包含历史淘汰，不能只凭这一列判断系统质量。", "",
        "![Compression versus formal baselines](figures/history_20260914/compression_vs_baselines.png)", "",
        "![Measured cost by timer scope](figures/history_20260914/cost_vs_baselines.png)", "",
        "旧 baseline 时间是包含请求内维护的 outer-request 累计墙钟；Ours 时间是包含再生成的模型推理累计时间。两者计时范围不同，成本图分开显示；目前尚缺同口径的全 pipeline 成本比较。", "",
        "| 方法 | Whole-context KV ↑ | History KV ↑ | 累计秒 ↓ | 计时范围 |",
        "| --- | ---: | ---: | ---: | --- |"]
    for row in figure_data["rows"]:
        history = "未测得" if row["history"] is None else f"{row['history']:.3f}×"
        lines.append(f"| {row['method']} | {row['whole_context']:.3f}× | {history} | {row['seconds']:.1f} | {row['timer']} |")
    lines.append("")
    metrics = original.split("## Detector 与运行成本", 1)[1].split("## AppWorld Full 参考", 1)[0]
    metrics = metrics.replace("，与前面的任务等权指标不同", "")
    lines += ["## 固定算法的压缩与运行指标", metrics.strip(), ""]
    out = ROOT / "reports/weekly/2026-09-14-history-system.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    index["selected_algorithm"] = read(BASE / "algorithm.selected.json")
    (BASE / "results.selected.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".metrics.json").write_text(json.dumps(index["extended_metrics"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".sources.json").write_text(json.dumps({"inputs": [digest(source)], "legacy": data["legacy"], "current": index, "generator": digest(Path(__file__))}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "selected_cells": len(index["cells"]), "internal_candidate_comparisons": 0}))


if __name__ == "__main__":
    main()
