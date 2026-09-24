"""Render the ablation report tables (Markdown) from the analysis JSON files only.

usage: python -m racer_ablation.report_tables CLOSED_LOOP_JSON UTILITY_JSON [LIGHT_JSON] > tables.md
"""
from __future__ import annotations

import json
import sys

NAMES = {"M0": "M0 完整 RACER（含 argument correction）", "M1": "M1 RACER-core",
         "M2": "M2 相同初始分配，直接提交 draft", "M3": "M3 evidence-free self-revision",
         "M4": "M4 RACER-core，检索不用 draft"}


def pct(value):
    return f"{100 * value:.1f}"


def pp(value):
    return f"{100 * value:+.1f}"


def closed_loop(data):
    rows, paired = data["rows"], data["paired_vs_M1"]
    out = ["| 行 | 成功 / 200 | 成功率 % | 相对 M1 (pp) [95% CI] | 仅此行对 / 仅 M1 对 | 方法容量失败 | "
           "harness 失败 | regenerations/decision | actor generations/task | 平均 history KV (tokens) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for label in ("M1", "M2", "M3", "M4"):
        row = rows[label]
        if label == "M1":
            delta, split = "—", "—"
        else:
            p = paired[label]
            delta = f"{pp(p['difference'])} [{pp(p['ci95'][0])}, {pp(p['ci95'][1])}]"
            split = f"{p['other_only']} / {p['reference_only']}"
        out.append(f"| {NAMES[label]} | {row['correct']} | {pct(row['success_rate_full_denominator'])} | "
                   f"{delta} | {split} | {len(row['method_failures'])} | {len(row['harness_failures'])} | "
                   f"{row['regenerations_per_decision']:.3f} | {row['actor_generations_per_task']:.1f} | "
                   f"{row['mean_history_kv_tokens']:.1f} |")
    return "\n".join(out)


def appendix(data):
    rows, paired = data["rows"], data["paired_vs_M1"]
    m0, m1, p = rows["M0"], rows["M1"], paired["M0"]
    corr = data["correction_M0_vs_M1"]
    out = ["| 行 | 成功 / 200 | 相对 M1 (pp) [95% CI] | 仅 M0 对 / 仅 M1 对 | correction 改动的 decision | 涉及任务 |",
           "|---|---|---|---|---|---|",
           f"| {NAMES['M0']} | {m0['correct']} | {pp(p['difference'])} [{pp(p['ci95'][0])}, {pp(p['ci95'][1])}] | "
           f"{p['other_only']} / {p['reference_only']} | {corr['decisions_changed_by_correction']} | "
           f"{corr['tasks_with_changed_correction']} |",
           f"| {NAMES['M1']} | {m1['correct']} | — | — | 0 | 0 |", "",
           f"仅 M0 成功的任务中，发生过 correction 改动的：{len(corr['full_only_with_changed_correction'])} / "
           f"{len(corr['full_only_correct'])}；仅 M1 成功的任务中，M0 发生过改动的："
           f"{len(corr['core_only_with_changed_correction_in_full'])} / {len(corr['core_only_correct'])}。"]
    return "\n".join(out)


def funnel(data):
    keys = ("decisions", "triggered", "generation_limit", "no_feasible_candidate", "repack_feasible",
            "self_revision_context_limit", "regeneration_completed", "regenerated_action_committed",
            "parse_fallback", "correction_proposed", "correction_verified", "correction_changed",
            "failed_decisions")
    out = ["| 行 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for label in ("M0", "M1", "M2", "M3", "M4"):
        f = data["rows"][label]["funnel"]
        out.append(f"| {label} | " + " | ".join(str(f[key]) for key in keys) + " |")
    return "\n".join(out)


def cost(data):
    out = ["| 行 | prompt tokens/task | output tokens/task | self-revision 追加 tokens（总） | history KV 生成数 |",
           "|---|---|---|---|---|"]
    for label in ("M0", "M1", "M2", "M3", "M4"):
        row = data["rows"][label]
        out.append(f"| {label} | {row['prompt_tokens_per_task']:.0f} | {row['completion_tokens_per_task']:.0f} | "
                   f"{row['self_revision_added_tokens']} | {row['history_kv_generations']} |")
    return "\n".join(out)


def utility(data):
    cov = data["coverage"]
    lines = [f"目标任务 {cov['target_tasks']}，抽中 {cov['sampled']}，P 已运行 {cov['probe_run']}，"
             f"前缀复现 {cov['reproduced']}，配对可评分 N={cov['pair_scorable']}，可恢复池 |F|={cov['recoverable']}；"
             f"排除原因 {cov['reasons']}。"]
    if "curves" in data:
        n, curves = data["n"], data["curves"]
        lines.append("")
        lines.append("| k/N | 完整 detector U (pp) | 轻量 detector U (pp) | random 期望 U (pp) |")
        lines.append("|---|---|---|---|")
        size = len(curves["x"])
        for k in sorted({0, size // 4, size // 2, 3 * size // 4, size - 1}):
            light = f"{100 * curves['light'][k]:+.2f}" if "light" in curves else "待补"
            lines.append(f"| {curves['x'][k]:.3f} | {100 * curves['full'][k]:+.2f} | "
                         f"{light} | {100 * curves['random_expected'][k]:+.2f} |")
        lines.append("")
        lines.append(f"always（对 F 全部干预）端点：{100 * data['always_endpoint']:+.2f} pp；never 端点 0。")
        for scope, counts in data["transitions"].items():
            lines.append(f"- {scope}：" + "，".join(f"{key} {value}" for key, value in counts.items()))
        lines.append(f"- 补充 AUROC/AP（对 Y0 失败）：{data['supplement_auroc_ap_on_y0_failure']}")
        lines.append(f"- 一致性：{data['consistency']}")
    return "\n".join(lines)


def main(argv):
    closed = json.load(open(argv[0], encoding="utf-8"))
    util = json.load(open(argv[1], encoding="utf-8"))
    print("## 闭环消融（正文）\n")
    print(closed_loop(closed))
    print("\n## M0 vs M1（附录）\n")
    print(appendix(closed))
    print("\n## 干预漏斗\n")
    print(funnel(closed))
    print("\n## 计算与 token\n")
    print(cost(closed))
    print("\n## 冻结状态恢复效用\n")
    print(utility(util))
    if len(argv) > 2:
        light = json.load(open(argv[2], encoding="utf-8"))
        print(f"\n轻量 detector：C={light['fit']['selected_c']}，weights={light['weights']}，"
              f"intercept={light['intercept']:.4f}，训练中恒定特征 {light['fit']['feature_constant_in_training']}，"
              f"完整 artifact 复现 {light['pipeline_check']['reproduced']}")


if __name__ == "__main__":
    main(sys.argv[1:])
