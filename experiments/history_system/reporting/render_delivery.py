"""Render the meeting report from returned results and the audited legacy table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SYSTEM = HERE.parent
BENCHMARKS = ("bfcl", "tau2", "toolsandbox", "acebench", "acon_appworld")
LABELS = {"bfcl": "BFCL", "tau2": "tau2", "toolsandbox": "ToolSandbox",
          "acebench": "ACEBench", "acon_appworld": "AppWorld"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def method_name(value):
    return {
        "d3_prefill_event": "Prefill-guided event recovery",
        "r002_d3_prefill_event_mixed20_v1": "Prefill-guided event recovery",
        "d4_prefill_event_demand_extract": "Demand encoding + event recovery",
        "d5_persistent_goal_demand": "Persistent goal + demand encoding",
        "d9_budget_feasible_source": "Budget-feasible ranked source recovery",
        "d8_budgeted_task_packet_demand": "Budgeted task packet + demand encoding",
        "d7_raw_dominant_demand": "Raw-dominant deduplication + demand encoding",
    }.get(value, value)


def percent(value):
    if value is None:
        return "未返回"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("Quality scores must be verified fractions in [0, 1]")
    return f"{100 * value:.2f}%"


def checkpoint_label(value):
    if not isinstance(value, dict):
        return "未绑定"
    arm, step, ratio = value.get("selected_arm"), value.get("selected_step"), value.get("ratio")
    if arm is None or step is None:
        return "未绑定"
    label = f"Arm {arm} checkpoint-{step}"
    return label if ratio is None else f"{label}, ratio={ratio}"


def render(index, legacy, checkpoint):
    algorithm = index.get("algorithm", {})
    name = algorithm.get("name", "Risk-guided history recovery")
    rows = index.get("cells", [])
    lines = ["# C2KV 混合历史系统", "", "## 算法", "",
        "基础系统把历史事件编码为 C2KV gist，并在固定内存预算内保留部分原文和精确引用。"
        f"本次实现的 {name} 增加了动作提交前的风险预测与原文恢复。",
        "",
        algorithm.get("execution_note", "风险特征已进入真实模型运行；分类器拟合与新版整题评测仍在进行。"),
        "",
        "```mermaid", "flowchart TD",
        '  H["完整历史事件"] --> M["C2KV gist、保留原文与精确引用"]',
        '  M --> F["对当前输入 Prefill 并读取 hidden state"]',
        '  F --> D["生成动作草稿"]',
        '  D --> P["分类器预测短段完成风险"]',
        '  P -->|低风险或恢复额度耗尽| E["执行最终动作"]',
        '  P -->|高风险且有恢复额度| S["用最新 user 消息和草稿检索历史原文"]',
        '  S --> B["替换低优先级记忆并重新检查总预算"]',
        '  B -->|可装入| R["再生成一次动作"]',
        '  B -->|无来源或装不下| E',
        '  R --> E', '  E --> H', "```", "",
        "- 风险预测使用当前模型的 Prefill hidden state。首版标签判断一个动作批次及其后正常停止能否完成当前目标，由离线官方判读提供，"
        "在线预测只读取已发生的历史与当前草稿。",
        "- 恢复内容是当前压缩视图中缺少原文的完整历史事件，支持工具调用及返回结果，"
        "也支持代码与文本 observation。选择依据是最新 user 消息、草稿与历史内容的匹配。AppWorld 的环境 observation 也采用 user 角色，基础D3未将持久任务目标与 observation 分开用于检索。",
        "- 恢复时先撤出可由 gist 覆盖的低优先级原文，再释放必要的可选 gist；"
        "当前输入和受保护内容继续保留。总预算允许时才重新生成，工具只执行最终动作。",
        "- 恢复额度按已经发生的 decision 数计算；首次生成与再生成共用整题调用预算。",
        "",
        f"当前模型选择为 Arm {checkpoint['selected_arm']} checkpoint-{checkpoint['selected_step']}，"
        f"编码 ratio={checkpoint['ratio']}。整体压缩以实际保留的 gist、raw 和派生内容计量。",
        "",
        "D5尝试强制保留AppWorld首条user原文，但其中含长篇固定说明，真实运行可能在后续步骤超过B0。D8把官方模板中的身份信息和实际任务后缀逐字提取为task packet，记录来源区间与hash。先检查packet与最低必要上下文能否共存；可行时先保留packet，再分配其他可选记忆。所有packet token计入同一个B0，模板不匹配或最低配置仍放不下时不强制准入。恢复检索使用任务packet、最新observation和草稿。候选是否可用仍看下表整题结果。",
        "",
        "预算可行选源在任务packet版本上继续修改恢复分配：按相关性排序依次检查来源，首个完整事件装不下时尝试下一个，每次都从原工作区重新计算预算；选中第一个可行事件后再生成。逐个预算检查不调用模型，恢复次数与预算保持原值。该版本正在进行固定任务集评测，完整总分尚未返回。",
        "", "## 新系统整题结果", "", "所有本轮结果均为 preliminary, n=1。", "",
        "| Benchmark | 配置 | 任务集 | 官方成绩 | 已回收评分/计划题数 | Direction |",
        "| --- | --- | --- | ---: | ---: | --- |"]
    for benchmark in BENCHMARKS:
        selected = [row for row in rows if row["benchmark"] == benchmark]
        if not selected:
            lines.append(f"| {LABELS[benchmark]} | {name} | 待汇总 | 尚未返回 | 未返回 | higher |")
        for row in selected:
            if row.get("official_score") is not None and not row.get("sources"):
                raise ValueError("Each new score requires its official source paths")
            score = percent(row.get("official_score"))
            if row.get("status") == "infra_failed" and row.get("all_pieces_terminal"):
                score = f"无完整总分（已结束，{row['n_unscored']}题未评分）"
            elif row.get("status") == "infra_failed":
                score = "无完整总分（存在未评分失败）"
            elif row.get("status") != "completed":
                score += "（进行中）" if row.get("n_scored", 0) else "（尚未回收评分）"
            count = f"{row.get('n_scored', 0)}/{row['n_planned']}"
            lines.append("| " + " | ".join(map(cell, [LABELS[benchmark], method_name(row.get("method", name)),
                row["cohort"], score, count, "higher"])) + " |")
    for note in index.get("reward_basis_notes", []):
        lines += ["", f"tau2 原恢复版已评分的 {note['task_count']} 题采用 DB 和 COMMUNICATE reward。"
            "这些题的数据库保持匹配，communicate_info 检查项为空；"
            "read-action checks 的不匹配以及未运行的 nl_assertions 没有降低该官方分数。", ""]
    comparison = [row for row in index.get("comparison_cells", []) if row.get("benchmark") == "bfcl"]
    if comparison:
        lines += ["", "### BFCL same long10 legacy comparison", "",
                  "| 配置 | 模型 checkpoint | 任务集 | 官方成绩 | 已回收评分/计划题数 | Status |",
                  "| --- | --- | --- | ---: | ---: | --- |"]
        for row in comparison:
            if row.get("official_score") is not None and not row.get("sources"):
                raise ValueError("Each comparison score requires its official source paths")
            lines.append("| " + " | ".join(map(cell, [
                method_name(row.get("method", name)), checkpoint_label(row.get("checkpoint")), row["cohort"],
                percent(row.get("official_score")),
                f"{row.get('n_scored', 0)}/{row['n_planned']}", row.get("status", "pending")
            ])) + " |")
    compression = [row for row in rows if row.get("compression") is not None]
    if compression:
        lines += ["", "| Benchmark | 配置 | 任务等权历史压缩倍数 | 额外生成次数 | Direction |",
                  "| --- | --- | ---: | ---: | --- |"]
        for row in compression:
            metric = row["compression"]
            ratio = metric.get("full_bytes_over_resident_bytes")
            if ratio is not None and not metric.get("sources"):
                raise ValueError("Measured compression requires trace sources")
            lines.append("| " + " | ".join(map(cell, [LABELS[row["benchmark"]],
                method_name(row.get("method", name)), "未返回" if ratio is None else f"{ratio:.3f}×",
                metric.get("extra_generations", "未返回"), "higher / lower"])) + " |")
    if any((row.get("compression") or {}).get("full_bytes_over_resident_bytes") is not None
           and row["compression"]["full_bytes_over_resident_bytes"] < 1 for row in rows):
        lines += ["", "本表先计算各任务的历史压缩率，再对任务等权汇总；后面的 Aggregate history KV compression 则使用累计字节之比。两种汇总权重不同。低于一表示该口径下活动历史内存比原始历史更大。", ""]
    for row in compression:
        metric = row["compression"]
        coverage = metric.get("source_coverage") or {}
        fraction = coverage.get("fully_represented_source_occurrence_fraction")
        strict = metric.get("complete_coverage_history_reduction")
        if fraction is not None and strict is not None:
            lines += ["", f"{LABELS[row['benchmark']]} 的主压缩倍数包含历史选择和淘汰。"
                f"按步骤累计，有原文或完整 gist 块覆盖的来源消息占比为 {100*fraction:.2f}%；"
                f"只看没有来源遗漏的步骤，同一任务等权口径下的历史压缩倍数为 {strict:.3f}×。", ""]
    peers = index.get("peer_costs")
    if peers:
        lines += ["", "## 同一 long10 的历史成本对照", "", "preliminary, n=1。任务ID相同；旧同行使用B500，D3使用C1000，实际轨迹与步数不同。这里作整体系统的描述比较。Resident KV包含system、工作区及生成尾部；allocator峰值还包含权重与临时张量。", "",
            "| 方法 | Accuracy | 提交步骤 | 峰值 resident KV MiB | 峰值 allocated MiB | 累计推理秒 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in peers['cells']:
            m=row['metrics'] or {}
            kv=(m.get('peak_resident_total_kv_bytes') or {}).get('value')
            allocated=(m.get('peak_device_allocated_bytes') or {}).get('value')
            elapsed=(m.get('inference_cumulative_seconds') or {}).get('value')
            lines.append("| " + " | ".join([row['method'],percent(row['official_score']),str(m.get('committed_steps','未返回')),f"{kv/1048576:.2f}" if kv is not None else "未测得",f"{allocated/1048576:.2f}" if allocated is not None else "未测得",f"{elapsed:.2f}" if elapsed is not None else "未测得"])+" |")
        lines += ["", "HiAgent旧产物缺少同接口server trace，成本留空。当前D3并未在该切片取得低于旧Full的峰值resident KV；history压缩不能替代整体缓存峰值与质量的联合判断。", ""]
    calibration = index.get("detector_calibration")
    if calibration:
        m=calibration['overall_coverage']
        lines += ["", "## Detector 校准集短段诊断", "",
            f"固定Prefill head在原calibration集有 {m['known_labels']}/{m['rows']} 个已知短段标签。"
            "该集合同时用于确定触发阈值；这里报告校准诊断，不作为交付benchmark上的独立动作错误检测成绩。preliminary, n=1。", "",
            "| TP | FP | FN | TN | Precision | Recall | F1 | FPR | 标签覆盖率 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| " + " | ".join([str(m[k]) for k in ['tp','fp','fn','tn']]+[percent(m[k]) for k in ['precision','recall','f1','fpr','label_coverage']])+" |", ""]
    extended = index.get("extended_metrics")
    if extended:
        lines += ["", "## Detector 与运行成本", "", "preliminary, n=1。调用数只计生成调用，包含被丢弃的draft；编码调用单列。压缩使用最终提交步骤的字节总和之比，包含淘汰，与前面的任务等权指标不同。", "",
            "| Benchmark / 方法 | Trigger | 恢复准入/触发 | 再生成步骤 | Calls / committed step | Aggregate history KV | 峰值 history KV MiB | 累计推理秒 | Eval wall span 秒 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        def number(value): return "未测得" if value is None else f"{value:.3f}"
        for row in extended['cells']:
            m=row['metrics']
            lines.append("| " + " | ".join([LABELS[row['benchmark']]+" / "+method_name(row['method']),percent(m['detector_trigger_rate']['value']),percent(m.get('recovery_admission',{}).get('admission_rate')),str(m['regenerated_steps']) if m['committed_steps'] else "未返回",number(m['model_calls_per_committed_step']),number(m['aggregate_history_kv_compression']['ratio']),number(m['peak_active_history_kv_bytes']/1048576 if m['peak_active_history_kv_bytes'] is not None else None),number(m['inference_cumulative_seconds']['value']),number(m['evaluation_stage_wall_span_seconds'])])+" |")
        lines += ["", "恢复准入/触发表示触发后预算与额度允许恢复的比例，预算拒绝和额度耗尽的计数保存在metrics JSON。Precision、Recall、F1、FPR需要draft错误标签；Recovery Success需要恢复前后动作正确性；Reference-drift positive rate需要同前缀reference动作。当前尚未完成这些标签的关联，保留缺失，不用整题成败代替。完整pipeline墙钟也尚未贯通记录；这里单列评测阶段时间跨度。总context KV、完整来源覆盖压缩、allocator峰值与编码次数保存在配套metrics JSON中。", ""]
    if extended:
        failed = [r for r in extended['cells'] if r['metrics'].get('failed_server_decisions')]
        if failed:
            lines += ["", "## 已记录的 server 执行失败", "", "官方harness有时在server异常后仍输出0分，因此下表单列失败decision，保留原官方分数；这些不是正常完成任务的质量结果。", "", "| Benchmark / 方法 | 失败 decision | 异常类型及数量 |", "| --- | ---: | --- |"]
            for row in failed:
                m=row['metrics']
                lines.append("| "+LABELS[row['benchmark']]+" / "+method_name(row['method'])+" | "+str(m['failed_server_decisions'])+" | "+", ".join(f"{k}: {v}" for k,v in sorted(m['server_failure_types'].items()))+" |")
            lines.append("")
    if extended:
        bfcl=next((r for r in extended['cells'] if r['benchmark']=='bfcl'),None)
        part=(bfcl or {}).get('metrics',{}).get('peak_kv_decomposition')
        if part and all(isinstance(part.get(k),(int,float)) for k in ['active_history_bytes','common_live_bytes','decode_tail_growth_bytes']):
            lines += ["", f"BFCL本轮峰值步骤分解：活动history为 {part['active_history_bytes']/1048576:.2f} MiB，"
                f"system与当前输入为 {part['common_live_bytes']/1048576:.2f} MiB，"
                f"decode缓存增长为 {part['decode_tail_growth_bytes']/1048576:.2f} MiB。"
                "这些数值取自同一次generation；该峰值主要由当前输入和生成尾部决定。", ""]
    reference = index.get("appworld_full_reference")
    if reference:
        lines += ["", "## AppWorld Full 参考", "",
            f"Full native checkpoint-1088：{percent(reference['official_score'])}，"
            f"官方已评分 {reference['n_scored']}/{reference['n_planned']}，preliminary, n=1。"
            "此参考使用不同 checkpoint，1088 的训练数据包含 AppWorld，"
            "且 native harness 采样与本轮 greedy 不同；按描述性结果比较。", ""]
    historical = legacy["latest_cross_method_table"]
    lines += ["", "## 历史 baseline", "",
        "复用上次组会的固定子集结果，模型为 checkpoint-1088。"
        "这些子集按 Full 表现筛选；它们与上表逐项注明的本轮任务集、模型设置一起阅读。"
        "ToolSandbox 列为 mean similarity，其余列为各自官方质量指标。preliminary, n=1。",
        "",
        "| 方法 | BFCL | tau2 | ToolSandbox | ACEBench | AppWorld | Direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for method, scores in historical["scores"].items():
        lines.append("| " + " | ".join([cell(method),
            *(percent(scores.get(benchmark)) for benchmark in BENCHMARKS), "higher"]) + " |")
    lines += ["", "旧表未提供 AppWorld 完整 split 成绩。旧 tau2 表内部分方法的实际评分分母"
              "尚待原结果回溯，保留原报告数值。", ""]
    diagnostic = index.get("toolsandbox_divergence")
    if diagnostic:
        lines.extend(["", "## ToolSandbox 回答分歧", "",
            f"D7 的 {diagnostic['n_lower_score']} 道降分题，分差均集中在最后的回答 milestone（preliminary, n=1），前面的 milestone 分数未变。Wi-Fi 题完成同样的关闭操作，最后回复措辞不同；最旧消息题则选择了不同消息，不能把下降都解释成措辞相似度。两次运行的时间戳工具返回不同，整题分差也不等于单独去除重复 gist 的因果效果。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    legacy_path = SYSTEM / "configs/delivery_20260914.baseline_sources.json"
    checkpoint_path = SYSTEM / "configs/checkpoint.selected.json"
    index = read(args.results)
    legacy = read(legacy_path)
    audit_path = REPO / "outputs/history_system_search/delivery_20260914/tau2.reward_audit.json"
    audit_sources = []
    if audit_path.exists():
        audit = read(audit_path)
        matching = [row for row in index.get("cells", []) if row.get("benchmark") == "tau2"
                    and row.get("method") == audit.get("candidate_id")]
        if matching:
            known = {source["sha256"] for row in matching for source in row.get("sources", [])}
            if audit["source_stage"]["sha256"] not in known:
                raise ValueError("Reward audit does not match collected tau2 stage")
            for source in [audit["source_stage"], *audit["sources"]]:
                path = Path(source["path"])
                if digest(path)["sha256"] != source["sha256"] or path.stat().st_size != source["bytes"]:
                    raise ValueError("Reward audit source changed")
            tasks = audit["tasks"]
            if tasks and all(set(t["reward_basis"]) == {"DB", "COMMUNICATE"}
                and t["communicate_items"] == 0 and not t["nl_assertions_evaluated"]
                and t["reward_breakdown"] == {"DB": 1.0, "COMMUNICATE": 1.0}
                and t["matched_action_checks"] == 0 for t in tasks):
                index["reward_basis_notes"] = [{"task_count": len(tasks), "audit": audit}]
                audit_sources = [digest(audit_path)]
    reference_root = REPO / "outputs/history_system_search/delivery_20260914/appworld_full1088_reference_v1"
    if (reference_root / "returned/summary_full_native.json").exists():
        import sys
        sys.path.insert(0, str(SYSTEM / "multibench"))
        from full_reference import collect
        index["appworld_full_reference"] = collect(reference_root)
    peer_cost_path = REPO / "outputs/history_system_search/delivery_20260914/peer.costs.json"
    if peer_cost_path.exists():
        peer_costs = read(peer_cost_path)
        for row in peer_costs['cells']:
            for source in row['sources']:
                if digest(Path(source['path']))['sha256'] != source['sha256']:
                    raise ValueError("Peer cost source changed")
        index['peer_costs'] = peer_costs
    calibration_path = REPO / "outputs/history_system_search/delivery_20260914/detector.calibration_metrics.json"
    if calibration_path.exists():
        calibration = read(calibration_path)
        for source in calibration['sources']:
            if digest(Path(source['path']))['sha256'] != source['sha256']:
                raise ValueError("Detector calibration source changed")
        index['detector_calibration'] = calibration
    divergence_path = REPO / "outputs/history_system_search/delivery_20260914/d7.divergence_diagnostic.json"
    if divergence_path.exists():
        divergence = read(divergence_path)
        for source in divergence["sources"]:
            if digest(Path(source["path"]))["sha256"] != source["sha256"]:
                raise ValueError("ToolSandbox divergence source changed")
        index["toolsandbox_divergence"] = divergence
    from extended_metrics import collect as collect_extended
    index["extended_metrics"] = collect_extended(index)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(index["extended_metrics"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(index, legacy, read(checkpoint_path))
    sources = {"inputs": [digest(path) for path in (args.results, legacy_path, checkpoint_path)],
               "legacy": legacy["latest_cross_method_table"],
               "current": index, "reward_audits": audit_sources, "generator": digest(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    args.out.with_suffix(".sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    selection_root = REPO / "outputs/history_system_search/delivery_20260914"
    if (selection_root / "algorithm.selected.json").exists():
        import shutil
        shutil.copyfile(args.out, selection_root / "internal_report.md")
        shutil.copyfile(args.out.with_suffix(".sources.json"), selection_root / "internal_report.sources.json")
        from render_selected_delivery import main as render_selected
        render_selected()
    print(json.dumps({"report": str(args.out), "result_cells": len(index.get("cells", []))}))


if __name__ == "__main__":
    main()
