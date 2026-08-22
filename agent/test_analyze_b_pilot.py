# -*- coding: utf-8 -*-
"""CPU-only, torch-free unit tests for agent/analyze_b_pilot.py.

Synthetic per-arm rows in the eval driver's jsonl shape; every statistic is
checked against a hand-computed value, not against a golden file.

Coverage:
a. row loading: skipped rows dropped, duplicate qid fatal, common-qid
   intersection in first-arm order;
b. McNemar b/c cells and the exact p on a hand-built discordance;
c. gist declaration: VOID fires exactly above 5%, a delay arm is exempt and
   its raw recent tokens land in their own column;
d. presented-token check: triggers exactly above 2%;
e. post-stratification: bucket weights are the reference arm's shares and the
   weighted diff is the hand-computed one;
f. R_agent = P(S_arm=1|S_full=1) plus the absolute rate, and the C→C/C→W/
   W→C/W→W transition counts;
g. delay accounting: bytes-matched skips exactly the rows above the 0.5x
   guard and counts them.

Run from the repo root (no torch needed):
  python -m pytest agent/test_analyze_b_pilot.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analyze_b_pilot import (  # noqa: E402
    _cluster_bootstrap,
    _common_qids,
    _decile_buckets,
    _delay_accounting,
    _footnote,
    _gist_declaration_table,
    _holm,
    _load_arm,
    _mcnemar_exact,
    _mde_pp,
    _paired_contrast,
    _poststratify,
    _presented_token_check,
    _r_agent,
    _transition_matrix,
    build_report,
    main,
)


def _row(qid, *, correct, gist=100, raw=0, presented=None, session=None):
    return {
        "qid": qid,
        "session_id": session or qid.rsplit(":", 1)[0],
        "subset": "appworld",
        "condition": "joint",
        "mode": "c2kv",
        "ratio": 8,
        "skipped": False,
        "tool_name_match": correct,
        "has_tool_call": True,
        "exact_match": correct,
        "gist_tokens": gist,
        "raw_recent_tokens": raw,
        "history_wrapped_tokens": gist * 8 if presented is None else presented,
        "history_content_tokens": 800,
        "chunk_policy": "agent-turn",
        "delay_recent_turns": 0,
    }


def _arm(flags, **kwargs):
    return {f"s{index // 2}:{index}": _row(f"s{index // 2}:{index}", correct=flag, **kwargs)
            for index, flag in enumerate(flags)}


# ---------------------------------------------------------------------------
# a. loading
# ---------------------------------------------------------------------------


def test_load_arm_drops_skipped_and_rejects_duplicates(tmp_path):
    path = tmp_path / "arm.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row("s0:0", correct=True),
                {"qid": "s0:1", "skipped": True, "skip_reason": "oom"},
                _row("s0:2", correct=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = _load_arm(str(path))
    assert sorted(rows) == ["s0:0", "s0:2"]

    path.write_text(
        json.dumps(_row("s0:0", correct=True)) + "\n" + json.dumps(_row("s0:0", correct=False)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="duplicate qid"):
        _load_arm(str(path))


def test_common_qids_uses_first_arm_order():
    arms = {
        "a": {"q2": {}, "q1": {}, "q3": {}},
        "b": {"q1": {}, "q3": {}},
    }
    assert _common_qids(arms) == ["q1", "q3"]


# ---------------------------------------------------------------------------
# b. McNemar / bootstrap / Holm / MDE
# ---------------------------------------------------------------------------


def test_mcnemar_cells_and_exact_p():
    # arm A right / B wrong on 3 qids, B right / A wrong on 0 -> b=3, c=0.
    qids = [f"s0:{i}" for i in range(4)]
    arm_a = {qid: _row(qid, correct=True) for qid in qids}
    arm_b = {qid: _row(qid, correct=index == 3) for index, qid in enumerate(qids)}
    block = _paired_contrast("A", arm_a, "B", arm_b, qids, reps=200, seed=0)
    assert block["b_a_wins"] == 3 and block["c_b_wins"] == 0
    assert block["acc_a"] == 1.0 and block["acc_b"] == 0.25
    assert block["diff_pp"] == pytest.approx(75.0)
    # Exact two-sided binomial with n=3, k=0: 2 * (1/8) = 0.25.
    assert block["mcnemar_exact_p"] == pytest.approx(0.25)
    assert _mcnemar_exact(0, 0) == 1.0


def test_cluster_bootstrap_is_deterministic_and_brackets_the_point():
    pairs = [(True, False)] * 5 + [(False, False)] * 5
    sessions = [f"s{index // 2}" for index in range(10)]
    first = _cluster_bootstrap(pairs, sessions, reps=500, seed=0)
    second = _cluster_bootstrap(pairs, sessions, reps=500, seed=0)
    assert first == second
    point, low, high = first
    assert point == pytest.approx(0.5)
    assert low <= point <= high


def test_holm_and_mde():
    adjusted = _holm({"x": 0.01, "y": 0.02, "z": 0.5})
    assert adjusted["x"] == pytest.approx(0.03)
    assert adjusted["y"] == pytest.approx(0.04)
    assert adjusted["z"] == pytest.approx(0.5)
    assert _mde_pp(200) == 8.9
    assert _mde_pp(50) == pytest.approx(17.8)
    assert "Paired MDE ≈ 8.9pp" in _footnote(200)
    assert _footnote(200).startswith("200-example teacher-forced")


# ---------------------------------------------------------------------------
# c. gist declaration
# ---------------------------------------------------------------------------


def test_gist_declaration_void_fires_exactly_above_five_percent():
    qids = ["s0:0", "s0:1"]
    arms = {
        "P-fixed": {qid: _row(qid, correct=True, gist=100) for qid in qids},
        "P-ok": {qid: _row(qid, correct=True, gist=105) for qid in qids},      # +5.0%
        "P-void": {qid: _row(qid, correct=True, gist=106) for qid in qids},    # +6.0%
    }
    table = _gist_declaration_table(arms, qids, "P-fixed")
    verdicts = {entry["arm"]: entry["verdict"] for entry in table["arms"]}
    assert verdicts["P-fixed"] == "OK"
    assert verdicts["P-ok"] == "OK"          # exactly 5% is NOT void (> 5% is)
    assert verdicts["P-void"] == "VOID"
    assert table["any_void"] is True


def test_delay_arm_is_exempt_and_reports_raw_recent_separately():
    qids = ["s0:0", "s0:1"]
    arms = {
        "P-fixed": {qid: _row(qid, correct=True, gist=100) for qid in qids},
        "P-delay": {qid: _row(qid, correct=True, gist=70, raw=240) for qid in qids},
    }
    table = _gist_declaration_table(arms, qids, "P-fixed")
    delay = next(entry for entry in table["arms"] if entry["arm"] == "P-delay")
    assert delay["delay_exempt"] is True
    assert delay["verdict"].startswith("EXEMPT")
    assert delay["mean_raw_recent_tokens"] == 240
    # -30% gist would otherwise be VOID; the exemption keeps it out of any_void.
    assert delay["deviation_vs_reference"] == pytest.approx(-0.30)
    assert table["any_void"] is False


# ---------------------------------------------------------------------------
# d. presented-token check
# ---------------------------------------------------------------------------


def test_presented_token_check_threshold():
    qids = ["s0:0", "s0:1"]
    arms = {
        "P-fixed": {qid: _row(qid, correct=True, presented=1000) for qid in qids},
        "P-turn": {qid: _row(qid, correct=True, presented=1020) for qid in qids},  # +2.0%
    }
    check = _presented_token_check(arms, qids, "P-fixed")
    assert check["max_abs_deviation"] == pytest.approx(0.02)
    assert check["poststratification_triggered"] is False

    arms["P-turn"] = {qid: _row(qid, correct=True, presented=1021) for qid in qids}
    check = _presented_token_check(arms, qids, "P-fixed")
    assert check["poststratification_triggered"] is True


# ---------------------------------------------------------------------------
# e. post-stratification
# ---------------------------------------------------------------------------


def test_decile_buckets_are_rank_based():
    values = {f"q{index}": index for index in range(10)}
    assignment = _decile_buckets(values, num_buckets=5)
    assert [assignment[f"q{index}"] for index in range(10)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_poststratify_weights_by_reference_bucket_share():
    # 4 qids, 2 buckets of 2.  Arm A wins both in the low bucket and loses one
    # in the high bucket: weighted diff = 0.5*1.0 + 0.5*(-0.5) = 0.25 -> 25pp.
    qids = ["s0:0", "s0:1", "s1:2", "s1:3"]
    presented = {"s0:0": 10, "s0:1": 20, "s1:2": 300, "s1:3": 400}
    reference = {qid: _row(qid, correct=True, presented=presented[qid]) for qid in qids}
    arm_a = {
        "s0:0": _row("s0:0", correct=True),
        "s0:1": _row("s0:1", correct=True),
        "s1:2": _row("s1:2", correct=True),
        "s1:3": _row("s1:3", correct=False),
    }
    arm_b = {
        "s0:0": _row("s0:0", correct=False),
        "s0:1": _row("s0:1", correct=False),
        "s1:2": _row("s1:2", correct=True),
        "s1:3": _row("s1:3", correct=True),
    }
    result = _poststratify(arm_a, arm_b, qids, reference, num_buckets=2)
    assert [bucket["n"] for bucket in result["buckets"]] == [2, 2]
    assert [bucket["weight"] for bucket in result["buckets"]] == [0.5, 0.5]
    assert [bucket["diff_pp"] for bucket in result["buckets"]] == [100.0, -50.0]
    assert result["weighted_diff_pp"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# f. R_agent + transitions
# ---------------------------------------------------------------------------


def test_r_agent_and_transition_matrix():
    qids = [f"s0:{index}" for index in range(4)]
    full = {
        "s0:0": _row("s0:0", correct=True),
        "s0:1": _row("s0:1", correct=True),
        "s0:2": _row("s0:2", correct=True),
        "s0:3": _row("s0:3", correct=False),
    }
    arm = {
        "s0:0": _row("s0:0", correct=True),
        "s0:1": _row("s0:1", correct=False),
        "s0:2": _row("s0:2", correct=True),
        "s0:3": _row("s0:3", correct=True),
    }
    block = _r_agent(arm, full, qids, reps=200, seed=0)
    assert block["n_full_correct"] == 3
    assert block["r_agent"] == pytest.approx(2 / 3, abs=1e-4)
    # The absolute rate must travel with the conditional (24号 0.2).
    assert block["absolute_accuracy"] == pytest.approx(0.75)
    assert _transition_matrix(arm, full, qids) == {
        "C->C": 2, "C->W": 1, "W->C": 1, "W->W": 0
    }


# ---------------------------------------------------------------------------
# g. delay accounting
# ---------------------------------------------------------------------------


def test_delay_accounting_bytes_matched_guard():
    qids = ["s0:0", "s0:1"]
    arms = {
        "P-fixed": {qid: _row(qid, correct=True, gist=100, raw=0) for qid in qids},
        "P-delay": {
            # raw 40 <= 0.5*100 -> kept; raw 60 > 0.5*100 -> skipped.
            "s0:0": _row("s0:0", correct=True, gist=80, raw=40),
            "s0:1": _row("s0:1", correct=True, gist=80, raw=60),
        },
    }
    accounting = _delay_accounting(arms, qids, "P-fixed", kv_bytes_per_token=10)
    delay = next(entry for entry in accounting["arms"] if entry["arm"] == "P-delay")
    assert delay["n_skipped_budget_guard"] == 1
    assert delay["n_matched"] == 1
    assert delay["mean_realized_bytes_matched"] == pytest.approx(120 * 10)
    assert delay["mean_realized_bytes_elastic"] == pytest.approx(((120) + (140)) / 2 * 10)
    fixed = next(entry for entry in accounting["arms"] if entry["arm"] == "P-fixed")
    assert fixed["n_skipped_budget_guard"] == 0


# ---------------------------------------------------------------------------
# End-to-end report + CLI
# ---------------------------------------------------------------------------


def _write_arm(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows.values()) + "\n", encoding="utf-8"
    )
    return str(path)


def _synthetic_arms():
    qids = [f"s{index // 4}:{index}" for index in range(16)]
    def make(pattern, **kwargs):
        return {
            qid: _row(qid, correct=pattern[index % len(pattern)], **kwargs)
            for index, qid in enumerate(qids)
        }
    # Presented tokens are pinned equal so the post-stratification trigger is
    # off by default; the trigger has its own test.
    return qids, {
        "P-fixed": make([True, False, True, False], gist=100, presented=1000),
        "P-turn": make([True, True, True, False], gist=101, presented=1000),
        "P-struct": make([True, True, False, False], gist=103, presented=1000),
        "P-delay": make([True, True, True, True], gist=70, raw=30, presented=1000),
    }


def test_poststratified_column_appears_only_when_triggered(tmp_path):
    from analyze_b_pilot import _markdown

    qids, arms = _synthetic_arms()
    report = build_report(arms, None, "P-fixed", reps=200, seed=0)
    assert report["presented_tokens"]["poststratification_triggered"] is False
    assert "post-strat Δpp" not in _markdown(report)
    assert all("poststratified" not in block for block in report["contrasts"])

    # Blow the presented-token budget on one arm: the column must appear and
    # every contrast must carry a weighted diff.
    for qid in qids:
        arms["P-struct"][qid]["history_wrapped_tokens"] = 2000
    report = build_report(arms, None, "P-fixed", reps=200, seed=0)
    assert report["presented_tokens"]["poststratification_triggered"] is True
    markdown = _markdown(report)
    assert "post-strat Δpp" in markdown
    for block in report["contrasts"]:
        assert "weighted_diff_pp" in block["poststratified"]


def test_build_report_families_and_footnote():
    qids, arms = _synthetic_arms()
    full = {qid: _row(qid, correct=True) for qid in qids}
    report = build_report(arms, full, "P-fixed", reps=200, seed=0)
    assert report["n_common_qids"] == 16
    families = {block["contrast"]: block["family"] for block in report["contrasts"]}
    assert families["P-struct vs P-fixed"] == "primary"
    assert families["P-struct vs P-turn"] == "primary"
    assert families["P-turn vs P-fixed"] == "primary"
    assert families["P-fixed vs P-delay"] == "exploratory"
    # Only the exploratory family is Holm-corrected.
    assert set(report["holm_exploratory"]) == {
        name for name, family in families.items() if family == "exploratory"
    }
    assert report["footnote"] == _footnote(16)
    assert "no claim below MDE is a ranking" in report["footnote"]
    assert set(report["r_agent"]) == set(arms)
    assert set(report["transitions"]) == set(arms)


def test_load_arm_mode_filter_splits_the_shared_reference_run(tmp_path):
    """One --compare_modes full,truncate file holds two rows per qid."""
    path = tmp_path / "ref.jsonl"
    rows = []
    for qid in ("s0:0", "s0:1"):
        full_row = _row(qid, correct=True)
        full_row["mode"] = "full"
        truncate_row = _row(qid, correct=False)
        truncate_row["mode"] = "truncate"
        rows.extend([full_row, truncate_row])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="duplicate qid"):
        _load_arm(str(path))
    only_full = _load_arm(str(path), "full")
    assert sorted(only_full) == ["s0:0", "s0:1"]
    assert all(row["tool_name_match"] for row in only_full.values())
    only_truncate = _load_arm(str(path), "truncate")
    assert not any(row["tool_name_match"] for row in only_truncate.values())


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    qids, arms = _synthetic_arms()
    full_rows = {}
    for qid in qids:
        row = _row(qid, correct=True)
        row["mode"] = "full"
        full_rows[qid] = row
    argv = ["--out_prefix", str(tmp_path / "b_pilot"), "--reps", "200"]
    for name, rows in arms.items():
        argv.extend(["--arm", f"{name}={_write_arm(tmp_path, name, rows)}"])
    argv.extend(["--full", _write_arm(tmp_path, "full", full_rows)])
    main(argv)

    report = json.loads((tmp_path / "b_pilot.analysis.json").read_text(encoding="utf-8"))
    assert report["reference_arm"] == "P-fixed"
    markdown = (tmp_path / "b_pilot.analysis.md").read_text(encoding="utf-8")
    # The mandated footnote appears under every table.
    assert markdown.count(report["footnote"]) >= 4
    assert "判据1" in markdown
    assert "pilot 不判方向生死" in markdown
    # No banned design vocabulary in the emitted report.
    for banned in ("router", "verifier", "adaptive compression ratio", "ratio selection"):
        assert banned not in markdown.lower()
    captured = capsys.readouterr()
    assert "n_common_qids" in captured.out


def test_missing_reference_arm_is_fatal():
    _, arms = _synthetic_arms()
    with pytest.raises(SystemExit, match="reference arm"):
        build_report(arms, None, "P-nope", reps=50)
