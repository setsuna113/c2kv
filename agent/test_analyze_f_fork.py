# -*- coding: utf-8 -*-
"""CPU-only, torch-free unit tests for agent/analyze_f_fork.py.

Synthetic branch rows are written to ``tmp_path`` and run through the real
CLI, so the test covers the whole path a pilot run takes: jsonl -> pairing ->
derived arms -> report -> ``.analysis.json`` + ``.analysis.md``.  Every number
asserted below is hand-computed from the fixture in
``_write_rows``' docstring — nothing is copied back out of the analyzer.

Coverage:
a. the analyzer imports with no torch installed;
b. report schema: every top-level block the F spec asks for is present;
c. hand-computed arm rates, Δ_oracle, the unconditional F2-F0 gap, four-cell
   counts, disagreement, both-match-gold;
d. the two cost ledgers sum to the recorded per-row seconds;
e. the R1 vs R1b tie-rule sensitivity block, including how many decisions flip;
f. the future-info caveat and the fixed oracle-union sentence appear verbatim,
   and the spec_shared footnote is rendered with the paired n and the MDE;
g. sampled arms (F1 / F3s) appear only when their rollouts are present.

Run from the repo root:
  python -m pytest agent/test_analyze_f_fork.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_f_fork as AF  # noqa: E402
from f_fork_common import BRANCH_COMPRESS_NOW, BRANCH_DEFER  # noqa: E402


KV_BYTES = 147456
A_PREFILL = {
    "system_prefill_sec": 0.01,
    "tool_compress_sec": 0.02,
    "full_prefill_sec": 0.0,
    "blend_sec": 0.005,
}
B_PREFILL = {
    "system_prefill_sec": 0.01,
    "tool_compress_sec": 0.02,
    "full_prefill_sec": 0.03,
    "blend_sec": 0.005,
}
A_PREFILL_SEC = sum(A_PREFILL.values())  # 0.035
B_PREFILL_SEC = sum(B_PREFILL.values())  # 0.065
A_DECODE_SEC = 0.1
B_DECODE_SEC = 0.15

GOLD = '{"arguments":{"city":"Paris"},"name":"get_weather"}'


def _branch_row(qid, session, arm_pass, branch, rollout, *, check, tool, pred):
    prefill = A_PREFILL if branch == BRANCH_COMPRESS_NOW else B_PREFILL
    decode = A_DECODE_SEC if branch == BRANCH_COMPRESS_NOW else B_DECODE_SEC
    cache_tokens = 900 if branch == BRANCH_COMPRESS_NOW else 1600
    peak_cache_tokens = cache_tokens + 200
    return {
        "qid": qid,
        "session_id": session,
        "subset": "appworld",
        "arm_pass": arm_pass,
        "branch": branch,
        "rollout_index": rollout,
        "skipped": False,
        "deterministic_check_pass": check,
        "pred_action_key": pred,
        "gold_action_key": GOLD,
        "action_key_match": pred == GOLD,
        "tool_name_match": tool,
        "argument_value_f1": 1.0 if tool else 0.0,
        "cache_tokens": cache_tokens,
        "peak_cache_tokens": peak_cache_tokens,
        "kv_bytes_per_token": KV_BYTES,
        "peak_bytes": peak_cache_tokens * KV_BYTES,
        "resident_bytes_measured": (1100 + 1800) * KV_BYTES,
        "resident_bytes_logical_shared": 1700 * KV_BYTES,
        "fork_segment_logical_ratio": 1.125,
        "generate_sec": decode,
        **prefill,
    }


# qid -> (session, A check, A tool, B check, B tool)
#
#   q0  s0  A pass/right   B pass/wrong   -> both pass : R1 keeps A, R1b keeps B
#   q1  s0  A fail/wrong   B pass/right   -> one passes: both rules take B
#   q2  s1  A pass/right   B fail/wrong   -> one passes: both rules take A
#   q3  s1  A fail/wrong   B fail/right   -> both fail : both rules keep A
#   q4  s2  A pass/right   B pass/right   -> both pass and both already gold
#
# F0 (always A) = q0,q2,q4 right          = 3/5 = 0.6
# F2 (always B) = q1,q3,q4 right          = 3/5 = 0.6
# F5 (union)    = every qid has one right = 5/5 = 1.0
# F3g R1  = A,B,A,A,A -> right,right,right,wrong,right = 4/5 = 0.8
# F3g R1b = B,B,A,A,B -> wrong,right,right,wrong,right = 3/5 = 0.6
GREEDY_PLAN = [
    ("q0", "s0", True, True, True, False),
    ("q1", "s0", False, False, True, True),
    ("q2", "s1", True, True, False, False),
    ("q3", "s1", False, False, False, True),
    ("q4", "s2", True, True, True, True),
]

# Sampled rollouts for two qids only, so F1 / F3s exist but on a smaller n.
#   q0: A_s0 fails, A_s1 passes/right, B_s0 passes/right
#       -> F1 takes A_s1 (right), F3s defers to B_s0 (right)
#   q2: A_s0 passes/right, A_s1 passes/wrong, B_s0 passes/wrong
#       -> F1 takes A_s0 (right), F3s both pass -> R1 keeps A_s0 (right)
SAMPLED_PLAN = [
    ("q0", "s0", (False, False), (True, True), (True, True)),
    ("q2", "s1", (True, True), (True, False), (True, False)),
]


def _write_rows(tmp_path, *, with_sampled=True, with_ineligible=True):
    rows = []
    for qid, session, a_check, a_tool, b_check, b_tool in GREEDY_PLAN:
        rows.append(
            _branch_row(
                qid, session, "greedy_core", BRANCH_COMPRESS_NOW, 0,
                check=a_check, tool=a_tool, pred=GOLD if a_tool else '{"name":"wrong_a"}',
            )
        )
        rows.append(
            _branch_row(
                qid, session, "greedy_core", BRANCH_DEFER, 0,
                check=b_check, tool=b_tool, pred=GOLD if b_tool else '{"name":"wrong_b"}',
            )
        )
    if with_sampled:
        for qid, session, a_s0, a_s1, b_s0 in SAMPLED_PLAN:
            for branch, rollout, (check, tool) in (
                (BRANCH_COMPRESS_NOW, 0, a_s0),
                (BRANCH_COMPRESS_NOW, 1, a_s1),
                (BRANCH_DEFER, 0, b_s0),
            ):
                rows.append(
                    _branch_row(
                        qid, session, "sampled", branch, rollout,
                        check=check, tool=tool,
                        pred=GOLD if tool else '{"name":"wrong_s"}',
                    )
                )
    if with_ineligible:
        rows.append({
            "qid": "q9",
            "session_id": "s3",
            "arm_pass": "greedy_core",
            "branch": "none",
            "rollout_index": 0,
            "skipped": True,
            "skip_reason": "last_chunk_tokens<64",
        })
        rows.append({
            "qid": "q8",
            "session_id": "s3",
            "arm_pass": "greedy_core",
            "branch": "none",
            "rollout_index": 0,
            "skipped": True,
            "skip_reason": "history_chunks<2",
        })
    path = tmp_path / "f_fork.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def report(tmp_path):
    path = _write_rows(tmp_path)
    return AF.build_report(AF.load_rows(path), coin_seed=0, bootstrap_b=200, noise_seeds=50)


# ---------------------------------------------------------------------------
# a-b. import + schema
# ---------------------------------------------------------------------------


def test_analyzer_is_torch_free_and_respects_the_naming_discipline():
    source = Path(AF.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    # "verifier" is banned as a name for our design; "draft-verify" survives
    # only inside the preregistered oracle-union sentence, which cites the
    # literature framing rather than naming a component of ours.
    assert "verifier" not in source.lower()


def test_report_carries_every_required_block(report):
    for key in (
        "n_paired_greedy",
        "n_paired_sampled",
        "n_sessions",
        "skips",
        "arm_table",
        "four_cell_table",
        "disagreement",
        "both_match_gold_block",
        "noise_floor",
        "cis",
        "cost_tables",
        "tie_rule_sensitivity",
        "mde",
        "footnote",
        "reading_card",
        "stopping_condition_whitelist",
        "memory_honesty_clause",
    ):
        assert key in report, key
    assert report["n_paired_greedy"] == 5
    assert report["n_paired_sampled"] == 2
    assert report["n_sessions"] == 3
    assert report["skips"]["num_skipped"] == 2
    assert report["skips"]["skip_reasons"] == {
        "history_chunks<2": 1,
        "last_chunk_tokens<64": 1,
    }


# ---------------------------------------------------------------------------
# c. hand-computed arm numbers
# ---------------------------------------------------------------------------


def test_arm_rates_match_the_hand_count(report):
    arms = report["arm_table"]["arms"]
    assert arms["F0"]["tool_name_match"] == 0.6
    assert arms["F2"]["tool_name_match"] == 0.6
    assert arms["F5"]["tool_name_match"] == 1.0
    assert arms["F3g"]["tool_name_match"] == 0.8
    assert arms["F3g_R1b"]["tool_name_match"] == 0.6
    assert arms["F0"]["n"] == 5


def test_delta_oracle_and_unconditional_gap(report):
    table = report["arm_table"]
    assert table["delta_oracle_timing"] == pytest.approx(0.4)
    assert table["best_single_arm"] in ("F0", "F2")
    # Deferring on EVERY decision is a wash here, and it is reported apart from
    # the selective story.
    assert table["unconditional_gap_F2_minus_F0"] == 0.0
    assert "not part of the selective" in table["unconditional_gap_note"]


def test_four_cell_disagreement_and_both_match_gold(report):
    cell = report["four_cell_table"]["tool_name_match"]
    assert cell["counts"] == {
        "both": 1,
        "compress_now_only": 2,
        "defer_only": 2,
        "neither": 0,
    }
    assert cell["n"] == 5
    disagree = report["disagreement"]
    assert disagree["disagree"] == 4 and disagree["n"] == 5
    both_gold = report["both_match_gold_block"]
    assert both_gold["count"] == 1
    assert both_gold["qids"] == ["q4"]
    assert both_gold["n_scored"] == 5


def test_mde_uses_the_observed_discordance(report):
    mde = report["mde"]
    assert mde["n_pairs"] == 5
    assert mde["discordant"] == 4
    assert mde["discordant_rate"] == 0.8
    assert mde["mde_pp"] == AF.paired_mde_pp(5, 0.8)
    assert "sqrt(p_discordant / n_pairs)" in mde["formula"]


def test_paired_mde_is_none_without_discordance():
    assert AF.paired_mde_pp(0, 0.5) is None
    assert AF.paired_mde_pp(10, 0.0) is None
    # Wider spread and smaller n both inflate the MDE.
    assert AF.paired_mde_pp(10, 0.8) > AF.paired_mde_pp(100, 0.8)


# ---------------------------------------------------------------------------
# d. cost ledgers
# ---------------------------------------------------------------------------


def test_rollout_ledger_counts_generated_and_kept(report):
    ledger = report["cost_tables"]["rollout_ledger"]
    assert ledger["F0"]["rollouts_generated"] == 5
    assert ledger["F0"]["rollouts_kept"] == 5
    assert ledger["F0"]["rollouts_per_decision_as_policy"] == 1
    assert ledger["F0"]["is_oracle"] is False
    assert ledger["F3g"]["rollouts_generated"] == 10
    assert ledger["F3g"]["rollouts_kept"] == 5
    assert ledger["F3g"]["rollouts_per_decision_as_policy"] == 2
    assert ledger["F5"]["is_oracle"] is True
    assert ledger["F5"]["rollouts_kept"] == 10
    # A coin is a one-rollout policy even though the pilot measured it from two.
    assert ledger["F4"]["rollouts_per_decision_as_policy"] == 1
    assert "already-recorded" in ledger["F4"]["note"]


def test_gpu_ms_ledger_sums_the_recorded_seconds(report):
    gpu = report["cost_tables"]["gpu_ms_ledger"]
    assert gpu["F0"]["gpu_ms_prefill"] == pytest.approx(1000 * 5 * A_PREFILL_SEC, abs=0.05)
    assert gpu["F0"]["gpu_ms_decode"] == pytest.approx(1000 * 5 * A_DECODE_SEC, abs=0.05)
    assert gpu["F2"]["gpu_ms_prefill"] == pytest.approx(1000 * 5 * B_PREFILL_SEC, abs=0.05)
    assert gpu["F3g"]["gpu_ms_total"] == pytest.approx(
        1000 * 5 * (A_PREFILL_SEC + B_PREFILL_SEC + A_DECODE_SEC + B_DECODE_SEC), abs=0.1
    )
    # 3 of 5 F0 decisions succeed over 5 * (0.035 + 0.1) GPU-seconds.
    assert gpu["F0"]["success_per_gpu_sec"] == pytest.approx(
        3 / (5 * (A_PREFILL_SEC + A_DECODE_SEC)), abs=1e-3
    )
    assert gpu["F0"]["components_ms"]["full_prefill_sec"] == 0.0
    assert gpu["F2"]["components_ms"]["full_prefill_sec"] == pytest.approx(150.0, abs=0.05)


def test_gpu_ms_ledger_dedups_shared_prefills_for_multi_rollout_arms(report):
    gpu = report["cost_tables"]["gpu_ms_ledger"]
    # F1 runs two rollouts on the SAME branch-A prefix: raw prefill counts it
    # twice, the dedup column once.
    assert gpu["F1"]["gpu_ms_prefill"] == pytest.approx(1000 * 2 * 2 * A_PREFILL_SEC, abs=0.05)
    assert gpu["F1"]["gpu_ms_prefill_dedup"] == pytest.approx(1000 * 2 * A_PREFILL_SEC, abs=0.05)
    # F3g spans two DIFFERENT branches, so nothing is deduplicated.
    assert gpu["F3g"]["gpu_ms_prefill_dedup"] == gpu["F3g"]["gpu_ms_prefill"]


def test_bytes_table_reports_measured_logical_and_the_honesty_clause(report):
    table = report["cost_tables"]["bytes_table"]
    assert table["per_branch"][BRANCH_COMPRESS_NOW]["avg_cache_tokens"] == 900.0
    assert table["per_branch"][BRANCH_DEFER]["avg_cache_tokens"] == 1600.0
    assert table["avg_resident_bytes_measured"] == (1100 + 1800) * KV_BYTES
    assert table["avg_resident_bytes_logical_shared"] == 1700 * KV_BYTES
    assert table["avg_fork_segment_logical_ratio"] == 1.125
    assert "1.125x raw(x_T)" in table["memory_honesty_clause"]
    assert "MORE memory" in table["memory_honesty_clause"]


# ---------------------------------------------------------------------------
# e. tie-rule sensitivity
# ---------------------------------------------------------------------------


def test_tie_rule_sensitivity_reports_both_rules_and_the_flip_count(report):
    block = report["tie_rule_sensitivity"]["pairs"]["F3g"]
    assert block["R1"] == 0.8
    assert block["R1b"] == 0.6
    assert block["delta_R1b_minus_R1"] == pytest.approx(-0.2)
    # Only the both-pass decisions can flip: q0 and q4.
    assert block["n_decisions_flipped"] == 2
    assert block["n"] == 5


# ---------------------------------------------------------------------------
# f. required verbatim text
# ---------------------------------------------------------------------------


def test_future_info_caveat_is_present_verbatim(report):
    assert report["both_match_gold_block"]["future_info_caveat"] == AF.FUTURE_INFO_CAVEAT
    assert "unavailable to any online policy" in AF.FUTURE_INFO_CAVEAT


def test_oracle_union_phrase_is_the_fixed_sentence(report):
    assert report["arm_table"]["oracle_union_phrase"] == AF.ORACLE_UNION_PHRASE
    assert "不构成选择机制" in AF.ORACLE_UNION_PHRASE
    assert "draft-verify" in AF.ORACLE_UNION_PHRASE


def test_footnote_matches_the_shared_spec_wording(report):
    footnote = report["footnote"]
    assert footnote.startswith("5-example teacher-forced next-action eval")
    assert "single seed, single checkpoint — preliminary, n=1" in footnote
    assert "Training pool appworld-dominated" in footnote
    assert "no claim below MDE is a ranking" in footnote
    assert f"{report['mde']['mde_pp']:g}pp" in footnote


def test_stopping_whitelist_has_five_entries_and_no_similarity_clause(report):
    whitelist = report["stopping_condition_whitelist"]
    assert len(whitelist) == 5
    assert not any("paper" in item.lower() for item in whitelist)
    note = report["stopping_condition_note"]
    assert "No threshold in this file is wired to any kill decision" in note
    assert "'resembles paper X' is not on the list" in note


# ---------------------------------------------------------------------------
# g. sampled arms + noise floor + CIs
# ---------------------------------------------------------------------------


def test_sampled_arms_appear_only_with_sampled_rollouts(tmp_path):
    greedy_only = AF.build_report(
        AF.load_rows(_write_rows(tmp_path, with_sampled=False)),
        coin_seed=0, bootstrap_b=100, noise_seeds=20,
    )
    assert "F1" not in greedy_only["arm_table"]["arms"]
    assert "F3s" not in greedy_only["arm_table"]["arms"]
    assert greedy_only["n_paired_sampled"] == 0


def test_sampled_arms_use_only_their_own_qids(report):
    arms = report["arm_table"]["arms"]
    assert arms["F1"]["n"] == 2
    assert arms["F3s"]["n"] == 2
    # q0: A_s1 right; q2: A_s0 right -> F1 = 2/2.
    assert arms["F1"]["tool_name_match"] == 1.0
    # q0 defers to a right B_s0; q2 both pass -> R1 keeps a right A_s0.
    assert arms["F3s"]["tool_name_match"] == 1.0


def test_noise_floor_band_brackets_the_coin(report):
    floor = report["noise_floor"]
    assert floor["n"] == 5
    assert floor["seeds"] == 50
    assert 0.0 <= floor["band95"][0] <= floor["mean"] <= floor["band95"][1] <= 1.0
    # Every qid has exactly one right branch, so any coin scores k/5.
    assert floor["min"] >= 0.0 and floor["max"] <= 1.0


def test_cis_cover_the_preregistered_contrasts_and_cluster_on_session(report):
    blocks = report["cis"]
    assert blocks["cluster"] == "session_id"
    for label in ("F3g-F0", "F3g-F4", "F2-F0"):
        assert label in blocks
        assert blocks[label]["n"] == 5
        assert blocks[label]["n_clusters"] == 3
    assert blocks["F3g-F0"]["point"] == pytest.approx(0.2)
    assert any(label.startswith("F5-") for label in blocks)
    assert blocks["F3s-F1"]["n"] == 2


# ---------------------------------------------------------------------------
# main(): files on disk + markdown rendering
# ---------------------------------------------------------------------------


def test_main_writes_json_and_markdown(tmp_path):
    path = _write_rows(tmp_path)
    AF.main([
        "--input_file", str(path),
        "--bootstrap_b", "100",
        "--noise_seeds", "20",
    ])
    json_path = tmp_path / "f_fork.analysis.json"
    md_path = tmp_path / "f_fork.analysis.md"
    assert json_path.exists() and md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_paired_greedy"] == 5
    assert payload["input_file"] == str(path)
    markdown = md_path.read_text(encoding="utf-8")
    assert "| F0 |" in markdown and "| F5 |" in markdown
    assert "Four-cell" in markdown
    assert AF.FUTURE_INFO_CAVEAT in markdown
    assert AF.ORACLE_UNION_PHRASE in markdown
    assert payload["footnote"] in markdown
    assert "Tie-rule sensitivity (R1 vs R1b)" in markdown


def test_every_results_section_carries_the_footnote(report):
    """A table copied out of this file must carry its own n / MDE caveat.

    Same convention as agent/analyze_b_pilot.py: the footnote goes under every
    results block, not only at the bottom of the document.
    """

    markdown = AF.render_markdown(report)
    footnote_line = f"_{report['footnote']}_"
    sections = [
        line for line in markdown.splitlines() if line.startswith("## ")
    ]
    results_sections = [
        "## Arm table",
        "## Four-cell (compress_now × defer)",
        "## Noise floor (F4 coin, reseeded)",
        "## Session-clustered CIs",
        "## Cost ledgers",
        "## Tie-rule sensitivity (R1 vs R1b)",
    ]
    assert results_sections == [s for s in sections if s != "## Reading card"]
    # One footnote per results section, plus the closing blockquote.
    assert markdown.count(footnote_line) == len(results_sections)
    # ... and each one sits inside its own section, not bunched at the end.
    body = markdown
    for section in results_sections:
        head, _, body = body.partition(section)
        del head
        block, _, _ = body.partition("\n## ")
        assert footnote_line in block, section
