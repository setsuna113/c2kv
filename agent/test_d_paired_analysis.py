# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/d_paired_analysis.py.

No torch, no rows from a real run: every statistic is checked against a
hand-computed value on a canned 2x2 or on canned prediction strings.

Coverage:
a. exact McNemar against closed-form binomial tail values;
b. session-cluster bootstrap reproducibility under a fixed seed;
c. two-level denominator arithmetic (both factors and the product);
d. the coherence triple on canned strings (legality, repeated 4-grams,
   length drift);
e. rescues that are not protocol-legal are excluded and counted separately;
f. per-token KV bytes from a config, against the 144 KiB cross-check;
g. the identity-check sentinel used by the smoke phase;
h. the markdown report: every table carries the footnote, and all four
   spec_D §5 tables are present.

Run from the repo root:
  python -m pytest agent/test_d_paired_analysis.py -v
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

import d_paired_analysis as A  # noqa: E402


LEGAL = '<tool_call>\n{"name": "%s", "arguments": {"a": 1}}\n</tool_call>'
TRUNCATED = '<tool_call>\n{"name": "%s", "arguments": {"a": 1}\n'
TARGET = LEGAL % "get_weather"


def _row(qid, text, *, generated_tokens=40, **extra):
    row = {
        "qid": qid,
        "session_id": qid.rsplit(":", 1)[0],
        "skipped": False,
        "prediction": text,
        "target": TARGET,
        "target_tool_name": "get_weather",
        "generated_tokens": generated_tokens,
        "has_tool_call": "<tool_call>" in text,
        "tool_name_match": '"name": "get_weather"' in text,
        "system_prefill_sec": 1.0,
        "tool_compress_sec": 2.0,
        "blend_sec": 0.5,
        "generate_sec": 4.0,
    }
    row.update(extra)
    return row


def _arm(rows):
    return {row["qid"]: row for row in rows}


def _manifest(qids, n_base_paired):
    return {
        "rule_version": "d_cw_v1",
        "batch": "batch-TF",
        "cw_qids": list(qids),
        "n_base_paired": n_base_paired,
    }


# --- a. McNemar -------------------------------------------------------------


@pytest.mark.parametrize(
    "b,c,expected",
    [
        (0, 0, 1.0),
        (1, 0, 1.0),        # 2 * C(1,0)/2^1
        (0, 2, 0.5),        # 2 * C(2,0)/2^2
        (3, 0, 0.25),       # 2 * C(3,0)/2^3
        (5, 0, 0.0625),     # 2 * 1/32
        (4, 1, 0.375),      # 2 * (C(5,0)+C(5,1))/32
        (10, 0, 2 / 1024),
        (3, 3, 1.0),        # symmetric cells cannot be significant
    ],
)
def test_mcnemar_exact_closed_form(b, c, expected):
    assert A.mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_is_symmetric():
    assert A.mcnemar_exact(7, 2) == A.mcnemar_exact(2, 7)


# --- b. bootstrap -----------------------------------------------------------


def test_estimators_come_from_the_shared_module():
    """No local copy: one estimator, shared with the B/F analyzers."""
    import paired_stats

    assert A.mcnemar_exact is paired_stats.mcnemar_exact
    assert A.cluster_bootstrap_diff is paired_stats.cluster_bootstrap_diff
    assert not hasattr(A, "cluster_bootstrap")


def test_cluster_bootstrap_is_seed_reproducible():
    # 12 sessions of 2 rows each; left wins in a third of them.
    pairs = []
    sessions = []
    for session in range(12):
        for row in range(2):
            index = 2 * session + row
            pairs.append((index % 3 == 0, index % 5 == 0))
            sessions.append(f"s{session}")
    first = A.cluster_bootstrap_diff(pairs, sessions, reps=500, seed=0)
    second = A.cluster_bootstrap_diff(pairs, sessions, reps=500, seed=0)
    third = A.cluster_bootstrap_diff(pairs, sessions, reps=500, seed=1)
    left_rate = sum(1 for left, _ in pairs if left) / len(pairs)
    right_rate = sum(1 for _, right in pairs if right) / len(pairs)
    assert first == second
    assert first[0] == pytest.approx(left_rate - right_rate)
    assert first[1] <= first[0] <= first[2]
    assert (first[1], first[2]) != (third[1], third[2])


def test_cluster_bootstrap_handles_empty_input():
    assert A.cluster_bootstrap_diff([], [], reps=10, seed=0) == (0.0, 0.0, 0.0)


# --- c. denominators --------------------------------------------------------


def test_two_level_denominator_arithmetic():
    qids = [f"s{i}:1" for i in range(10)]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    # Three of the ten triggers are repaired.
    corr = _arm(
        [
            _row(q, LEGAL % ("get_weather" if i < 3 else "wrong_tool"))
            for i, q in enumerate(qids)
        ]
    )
    report = A.analyze({"none": none, "corr": corr}, _manifest(qids, 100), reps=200)
    block = report["per_arm"]["corr"]["two_level_denominator"]
    assert block["L1_numerator_n_C2W"] == 10
    assert block["L1_denominator_n_base_paired"] == 100
    assert block["L1_trigger_rate"] == pytest.approx(0.10)
    assert block["L2_numerator_n_rescued"] == 3
    assert block["L2_denominator_n_C2W_scored"] == 10
    assert block["L2_rescue_rate_within_triggers"] == pytest.approx(0.30)
    assert block["product_rescued_over_base"] == pytest.approx(0.03)
    # L1 * L2 must equal the product that is also reported.
    assert block["L1_trigger_rate"] * block["L2_rescue_rate_within_triggers"] == pytest.approx(
        block["product_rescued_over_base"]
    )
    assert report["per_arm"]["none"]["two_level_denominator"]["L2_numerator_n_rescued"] == 0


def test_transition_matrix_is_labelled_as_trigger_set_only():
    qids = ["s0:1", "s0:2"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    corr = _arm([_row("s0:1", LEGAL % "get_weather"), _row("s0:2", LEGAL % "wrong_tool")])
    report = A.analyze({"none": none, "corr": corr}, _manifest(qids, 20), reps=100)
    matrix = report["transition_matrices"]["corr"]
    assert matrix["note"] == "transition on trigger set, not full set"
    assert matrix["cells"] == {"W->C": 1, "W->W": 1}
    assert report["verdict_scope"] == "mechanism only, no direction verdicts"
    assert "Paired MDE" in report["footnote"] and "preliminary, n=1" in report["footnote"]


def test_missing_rows_are_counted_not_imputed():
    qids = ["s0:1", "s0:2", "s0:3"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    corr = _arm([_row("s0:1", LEGAL % "get_weather")])
    report = A.analyze({"none": none, "corr": corr}, _manifest(qids, 30), reps=100)
    assert report["per_arm"]["corr"]["n_on_trigger_set"] == 1
    assert report["per_arm"]["corr"]["n_missing_from_trigger_set"] == 2
    assert report["per_arm"]["corr"]["two_level_denominator"]["L2_denominator_n_C2W_scored"] == 1


# --- d. coherence triple ----------------------------------------------------


def test_protocol_legality_on_canned_strings():
    assert A.protocol_legal(LEGAL % "get_weather")
    assert not A.protocol_legal(TRUNCATED % "get_weather")
    assert not A.protocol_legal("Action: just prose, no block at all")
    assert not A.protocol_legal('<tool_call>{"arguments": {}}</tool_call>')
    assert not A.protocol_legal("<tool_call>{name: get_weather}</tool_call>")
    assert A.protocol_legal('<tool_call>{"function": {"name": "x"}}</tool_call>')
    # Every emitted block must be legal, not just the first one.
    assert not A.protocol_legal((LEGAL % "a") + "\n" + '<tool_call>{"nope": 1}</tool_call>')


def test_repeat_4gram_rate_on_canned_strings():
    assert A.repeat_4gram_rate("") == 0.0
    assert A.repeat_4gram_rate("one two three") == 0.0
    assert A.repeat_4gram_rate("a b c d e f g h") == 0.0
    assert A.repeat_4gram_rate("a b c d a b c d") == pytest.approx(0.2)
    looping = " ".join(["x y z w"] * 10)
    assert A.repeat_4gram_rate(looping) > A.DEGENERATE_REPEAT_THRESHOLD


def test_coherence_block_reports_all_three():
    qids = ["s0:1", "s0:2"]
    none = _arm([_row(q, LEGAL % "wrong_tool", generated_tokens=100) for q in qids])
    arm = _arm(
        [
            _row("s0:1", LEGAL % "get_weather", generated_tokens=150),
            _row("s0:2", " ".join(["x y z w"] * 10), generated_tokens=50),
        ]
    )
    report = A.analyze({"none": none, "corr": arm}, _manifest(qids, 40), reps=100)
    coherence = report["per_arm"]["corr"]["coherence"]
    assert coherence["protocol_legal_rate"] == pytest.approx(0.5)
    assert coherence["degenerate_rate"] == pytest.approx(0.5)
    assert coherence["degenerate_threshold"] == 0.5
    # (150-100)/100 = +0.5 and (50-100)/100 = -0.5 -> mean 0.
    assert coherence["length_drift_vs_none_mean"] == pytest.approx(0.0)
    assert coherence["output_tokens_mean"] == pytest.approx(100.0)


# --- e. illegal rescues -----------------------------------------------------


def test_correct_but_illegal_flip_is_not_a_rescue():
    qids = ["s0:1", "s0:2"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    arm = _arm(
        [
            _row("s0:1", TRUNCATED % "get_weather"),   # right name, broken syntax
            _row("s0:2", LEGAL % "get_weather"),       # a real rescue
        ]
    )
    report = A.analyze({"none": none, "corr": arm}, _manifest(qids, 20), reps=100)
    stats = report["per_arm"]["corr"]
    assert stats["n_rescued"] == 1
    assert stats["n_correct_but_illegal"] == 1
    # The transition matrix uses raw correctness, so it still sees two flips.
    assert report["transition_matrices"]["corr"]["cells"] == {"W->C": 2}


def test_primary_contrast_is_corr_re_minus_sham():
    qids = [f"s0:{i}" for i in range(6)]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    sham = _arm(
        [_row(q, LEGAL % ("get_weather" if i < 1 else "wrong_tool")) for i, q in enumerate(qids)]
    )
    corr_re = _arm(
        [_row(q, LEGAL % ("get_weather" if i < 4 else "wrong_tool")) for i, q in enumerate(qids)]
    )
    report = A.analyze(
        {"none": none, "sham": sham, "corr_re": corr_re}, _manifest(qids, 60), reps=200
    )
    primary = report["primary_contrast"]
    assert primary["left_arm"] == "corr_re" and primary["right_arm"] == "sham"
    # Rates are rounded to 4 decimals in the report.
    assert primary["left_rate"] == pytest.approx(4 / 6, abs=5e-5)
    assert primary["right_rate"] == pytest.approx(1 / 6, abs=5e-5)
    assert primary["b_left_only"] == 3 and primary["c_right_only"] == 0
    assert primary["diff_point_pp"] == pytest.approx(50.0, abs=0.01)
    labels = {block["contrast"] for block in report["secondary_contrasts"]}
    assert "secondary: sham - none" in labels
    assert "secondary: corr_re - none" in labels


# --- f. cost axes -----------------------------------------------------------


def test_kv_bytes_per_token_matches_the_144kib_crosscheck():
    config = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2560,
        "torch_dtype": "bfloat16",
    }
    assert A.kv_bytes_per_token(config) == 144 * 1024
    assert A.kv_bytes_per_token(config) == A.REFERENCE_KV_BYTES_PER_TOKEN
    fp32 = dict(config, torch_dtype="float32")
    assert A.kv_bytes_per_token(fp32) == 2 * 144 * 1024


def test_kv_bytes_per_token_infers_head_dim():
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 32,
        "torch_dtype": "bfloat16",
    }
    assert A.kv_bytes_per_token(config) == 2 * 2 * 2 * 8 * 2


def test_gpu_seconds_include_the_full_arm_prefill():
    """full_prefill_sec is E-full's dominant cost (the whole-history prefill);
    the intervention arms carry it as 0.0, so only the full arm moves."""
    qids = ["s0:1"]
    none = _arm([_row("s0:1", LEGAL % "wrong_tool", full_prefill_sec=0.0)])
    full = _arm([_row("s0:1", LEGAL % "get_weather", full_prefill_sec=30.0)])
    report = A.analyze({"none": none, "full": full}, _manifest(qids, 10), reps=50)
    # 1.0 + 2.0 + 0.5 + 4.0 + 30.0
    assert report["per_arm"]["full"]["cost"]["gpu_sec_mean"] == pytest.approx(37.5)
    assert report["per_arm"]["none"]["cost"]["gpu_sec_mean"] == pytest.approx(7.5)
    pareto = {entry["arm"]: entry for entry in report["pareto"]}
    assert pareto["full"]["gpu_sec_mean"] == pytest.approx(37.5)
    # The markdown states the full-prefill term so a reader can audit the sum.
    assert "full prefill" in A.render_markdown(report)


def test_cost_axes_sum_the_declared_seconds_fields():
    qids = ["s0:1"]
    none = _arm([_row("s0:1", LEGAL % "wrong_tool")])
    arm = _arm(
        [
            _row(
                "s0:1",
                LEGAL % "get_weather",
                d_corr_span_tokens=120,
                d_recompute_tokens=300,
                d_corr_slice_prefill_sec=1.5,
                d_recompute_prefill_sec=2.5,
            )
        ]
    )
    report = A.analyze(
        {"none": none, "corr_re": arm}, _manifest(qids, 10), reps=50, bytes_per_token=1000
    )
    cost = report["per_arm"]["corr_re"]["cost"]
    assert cost["appended_tokens_mean"] == pytest.approx(420.0)
    assert cost["appended_kv_bytes_mean"] == pytest.approx(420_000.0)
    # 1.0 + 2.0 + 0.5 + 1.5 + 2.5 + 4.0
    assert cost["gpu_sec_mean"] == pytest.approx(11.5)
    assert report["kv_bytes_per_token_reference_check"]["matches"] is False


# --- g. identity sentinel ---------------------------------------------------


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def test_identity_check_passes_on_identical_arms(tmp_path):
    rows = [_row("s0:1", LEGAL % "get_weather"), _row("s0:2", LEGAL % "wrong_tool")]
    left = _write(tmp_path / "a.jsonl", rows)
    right = _write(tmp_path / "b.jsonl", rows)
    result = A.identity_check(left, right, ["prediction"])
    assert result["passed"] is True
    assert result["n_compared"] == 2
    assert result["n_mismatches"] == 0


def test_identity_check_fails_and_names_the_row(tmp_path):
    left = _write(tmp_path / "a.jsonl", [_row("s0:1", LEGAL % "get_weather")])
    right = _write(tmp_path / "b.jsonl", [_row("s0:1", LEGAL % "wrong_tool")])
    result = A.identity_check(left, right, ["prediction"])
    assert result["passed"] is False
    assert result["n_mismatches"] == 1
    assert result["mismatches"][0]["qid"] == "s0:1"
    assert result["mismatches"][0]["field"] == "prediction"


def test_identity_check_fails_on_empty_overlap(tmp_path):
    left = _write(tmp_path / "a.jsonl", [_row("s0:1", LEGAL % "get_weather")])
    right = _write(tmp_path / "b.jsonl", [_row("s9:9", LEGAL % "get_weather")])
    result = A.identity_check(left, right, ["prediction"])
    assert result["passed"] is False
    assert result["n_compared"] == 0
    assert result["n_only_left"] == 1 and result["n_only_right"] == 1


def test_cli_identity_check_exit_codes(tmp_path, capsys):
    rows = [_row("s0:1", LEGAL % "get_weather")]
    left = _write(tmp_path / "a.jsonl", rows)
    right = _write(tmp_path / "b.jsonl", rows)
    assert A.main(["--identity_check", left, right, "--identity_fields", "prediction"]) == 0
    other = _write(tmp_path / "c.jsonl", [_row("s0:1", LEGAL % "x")])
    assert A.main(["--identity_check", left, other]) == 1
    capsys.readouterr()


def test_cli_requires_arms_outside_identity_mode():
    with pytest.raises(SystemExit, match="--arm"):
        A.main(["--manifest", "x.json"])


# --- manifest sha binding ----------------------------------------------------


def test_rows_binding_tolerates_absent_sha_and_rejects_mismatch():
    good = {"s0:1": dict(_row("s0:1", LEGAL % "x"), bundle_manifest_sha256="a" * 64)}
    battery = {"s0:1": _row("s0:1", LEGAL % "x")}  # battery-reuse rows lack the field
    A.assert_rows_bind_to_manifest({"corr": good, "full": battery}, "a" * 64)
    stale = {"s0:1": dict(_row("s0:1", LEGAL % "x"), bundle_manifest_sha256="b" * 64)}
    with pytest.raises(SystemExit, match="different frozen trigger-set generation"):
        A.assert_rows_bind_to_manifest({"corr": stale}, "a" * 64)


def test_main_fatals_on_manifest_generation_mismatch(tmp_path):
    qids = ["s0:1"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(qids, 10)), encoding="utf-8")
    manifest_sha = A.sha256_text_file(manifest_path)
    ok_rows = [dict(_row("s0:1", LEGAL % "wrong_tool"), bundle_manifest_sha256=manifest_sha)]
    stale_rows = [dict(_row("s0:1", LEGAL % "get_weather"), bundle_manifest_sha256="f" * 64)]
    none_path = _write(tmp_path / "none.jsonl", ok_rows)
    stale_path = _write(tmp_path / "corr.jsonl", stale_rows)
    argv = [
        "--arm", f"none={none_path}",
        "--arm", f"corr={stale_path}",
        "--manifest", str(manifest_path),
        "--out_prefix", str(tmp_path / "out" / "d_paired"),
        "--reps", "50",
    ]
    with pytest.raises(SystemExit, match="FATAL"):
        A.main(argv)
    # Same rows re-stamped with the right sha analyze cleanly.
    fixed_path = _write(
        tmp_path / "corr_ok.jsonl",
        [dict(row, bundle_manifest_sha256=manifest_sha) for row in stale_rows],
    )
    argv[3] = f"corr={fixed_path}"
    assert A.main(argv) == 0


# --- no_downstream split ------------------------------------------------------


def test_no_downstream_split_reports_the_two_cells_apart():
    qids = ["s0:1", "s0:2", "s0:3"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    corr_re = _arm(
        [
            _row("s0:1", LEGAL % "get_weather"),   # T==1, rescued
            _row("s0:2", LEGAL % "get_weather"),   # T>1, rescued
            _row("s0:3", LEGAL % "wrong_tool"),    # T>1, not rescued
        ]
    )
    report = A.analyze(
        {"none": none, "corr_re": corr_re},
        _manifest(qids, 30),
        reps=50,
        no_downstream_qids={"s0:1"},
    )
    split = report["no_downstream_split"]
    assert split["available"] is True
    assert split["n_no_downstream"] == 1
    assert split["no_downstream_qids"] == ["s0:1"]
    assert split["per_arm"]["corr_re"] == {"n_scored": 1, "n_rescued": 1}
    markdown = A.render_markdown(report)
    assert "No-downstream split (T==1" in markdown
    assert "corr_re 1/1" in markdown


def test_no_downstream_split_marked_unavailable_without_a_source():
    qids = ["s0:1"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    report = A.analyze({"none": none}, _manifest(qids, 10), reps=50)
    assert report["no_downstream_split"]["available"] is False
    assert "unavailable" in A.render_markdown(report)


def test_load_no_downstream_qids_from_bundles_and_plan(tmp_path):
    assert A.load_no_downstream_qids(None, None) is None
    bundles = _write(
        tmp_path / "bundles.jsonl",
        [
            {"qid": "s0:1", "no_downstream": True},
            {"qid": "s0:2", "no_downstream": False},
            {"qid": "s0:3", "no_downstream": None},
        ],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"per_qid": {"s0:4": {"no_downstream": True}, "s0:5": {"no_downstream": False}}}),
        encoding="utf-8",
    )
    assert A.load_no_downstream_qids(bundles, None) == {"s0:1"}
    assert A.load_no_downstream_qids(None, str(plan_path)) == {"s0:4"}
    assert A.load_no_downstream_qids(bundles, str(plan_path)) == {"s0:1", "s0:4"}
    # Sources given but nothing is T==1: an EMPTY set, not None.
    empty = _write(tmp_path / "empty.jsonl", [{"qid": "s0:9", "no_downstream": False}])
    assert A.load_no_downstream_qids(empty, None) == set()


# --- harness divergence counter ----------------------------------------------


def test_harness_divergence_is_counted_not_just_warned():
    qids = ["s0:1", "s0:2"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    # The harness field lies on one row: re-score says correct, field says not.
    arm = _arm(
        [
            _row("s0:1", LEGAL % "get_weather", tool_name_match=False),
            _row("s0:2", LEGAL % "wrong_tool"),
        ]
    )
    report = A.analyze({"none": none, "corr": arm}, _manifest(qids, 20), reps=50)
    divergence = report["per_arm"]["corr"]["harness_divergence"]
    assert divergence["n_metric_disagreements"] == 1
    assert divergence["n_call_disagreements"] == 0
    assert report["per_arm"]["none"]["harness_divergence"]["n_metric_disagreements"] == 0
    assert report["n_harness_metric_disagreements"] == 1
    markdown = A.render_markdown(report)
    assert "Harness-score divergence:" in markdown
    assert "1 row(s)" in markdown and "corr 1" in markdown


# --- h. markdown report -----------------------------------------------------


def _five_arm_report(reps=100):
    qids = [f"s{i // 3}:{i % 3}" for i in range(9)]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    sham = _arm(
        [_row(q, LEGAL % ("get_weather" if i < 1 else "wrong_tool")) for i, q in enumerate(qids)]
    )
    corr = _arm(
        [
            _row(q, LEGAL % ("get_weather" if i < 3 else "wrong_tool"), d_corr_span_tokens=100)
            for i, q in enumerate(qids)
        ]
    )
    corr_re = _arm(
        [
            _row(
                q,
                LEGAL % ("get_weather" if i < 5 else "wrong_tool"),
                d_corr_span_tokens=100,
                d_recompute_tokens=200,
                d_recompute_prefill_sec=2.0,
            )
            for i, q in enumerate(qids)
        ]
    )
    full = _arm(
        [_row(q, LEGAL % ("get_weather" if i < 7 else "wrong_tool")) for i, q in enumerate(qids)]
    )
    return A.analyze(
        {"none": none, "sham": sham, "corr": corr, "corr_re": corr_re, "full": full},
        _manifest(qids, 90),
        reps=reps,
    )


def test_markdown_puts_the_footnote_under_every_table():
    report = _five_arm_report()
    markdown = A.render_markdown(report)
    # spec_D §5 asks for four result tables plus the Pareto data; each one is
    # a distinct H2 and each must be followed by the footnote.
    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    assert len(headings) == 5
    n_tables = markdown.count("\n|---")
    assert n_tables == len(headings)
    assert markdown.count(f"_{report['footnote']}_") == n_tables


def test_markdown_carries_the_required_labels():
    report = _five_arm_report()
    markdown = A.render_markdown(report)
    assert "transition on trigger set, not full set" in markdown
    assert "mechanism only, no direction verdicts" in markdown
    assert "MDE ≈ 17-25pp" in markdown
    # Two-level denominator: both factors and the product, never the product alone.
    assert "L1 = n_C2W/n_base" in markdown and "L2 = rescued/n_C2W" in markdown
    assert "product L1·L2" in markdown
    # Coherence triple.
    assert "protocol-legal rate" in markdown
    assert "repeat-4gram mean" in markdown
    assert "length drift vs none (mean)" in markdown
    # Primary contrast is stated as corr_re - sham.
    assert "primary: corr_re - sham" in markdown
    # Pareto axes.
    assert "appended KV bytes (mean)" in markdown and "GPU-sec (mean)" in markdown


def test_markdown_numbers_agree_with_the_json():
    report = _five_arm_report()
    markdown = A.render_markdown(report)
    # The denominator table is the 9-column one (the Pareto table repeats the
    # same arm/mode prefix with 5 columns).
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("| corr_re | `d_corr_recompute`")
    ]
    assert len(rows) == 2
    cells = next(row for row in rows if len(row) == 9)
    block = report["per_arm"]["corr_re"]["two_level_denominator"]
    assert cells[2] == str(report["per_arm"]["corr_re"]["n_on_trigger_set"])
    assert float(cells[4]) == pytest.approx(block["L1_trigger_rate"])
    assert float(cells[5]) == pytest.approx(block["L2_rescue_rate_within_triggers"])
    assert float(cells[7]) == pytest.approx(block["product_rescued_over_base"])


def test_markdown_renders_missing_values_as_dashes_not_zero():
    # An arm with no rows on the trigger set has None everywhere; a rendered
    # 0.0000 would read as a measured zero.
    qids = ["s0:1", "s0:2"]
    none = _arm([_row(q, LEGAL % "wrong_tool") for q in qids])
    report = A.analyze({"none": none, "corr": {}}, _manifest(qids, 20), reps=50)
    markdown = A.render_markdown(report)
    corr_row = [line for line in markdown.splitlines() if line.startswith("| corr | `d_corr`")]
    assert corr_row and "—" in corr_row[0]


def test_main_writes_both_json_and_markdown(tmp_path):
    qids = ["s0:1", "s0:2", "s0:3"]
    none_path = _write(tmp_path / "none.jsonl", [_row(q, LEGAL % "wrong_tool") for q in qids])
    corr_path = _write(
        tmp_path / "corr.jsonl",
        [_row(q, LEGAL % ("get_weather" if q.endswith("1") else "wrong_tool")) for q in qids],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(qids, 30)), encoding="utf-8")
    prefix = tmp_path / "out" / "d_paired"
    assert A.main([
        "--arm", f"none={none_path}",
        "--arm", f"corr={corr_path}",
        "--manifest", str(manifest_path),
        "--out_prefix", str(prefix),
        "--reps", "50",
    ]) == 0
    json_report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert markdown.startswith("# Task D pilot")
    assert json_report["footnote"] in markdown
    assert json_report["inputs"]["corr"] == corr_path
