# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/d_downstream_analysis.py.

No torch, no rows from a real run: statistics are checked against
hand-computed values on canned rows (the discipline of
test_d_paired_analysis.py / test_paired_stats.py).

Coverage (spec test numbers 15-21):
  15. (qid, offset) keying with last-GROUP-wins, skipped rows counted not
      loaded;
  16. paired contrast on a 4-qid fixture with hand-computed b/c and exact
      McNemar p; seed wired through and reproducible;
  17. every rendered table carries the footnote and an "exploratory" note;
  18. the out_prefix guard refuses frozen-round names/directories;
  19. the offset-0 identity sentinel passes on matching fixtures and exits 1
      on one perturbed prediction;
  20. a row bound to a foreign manifest sha is fatal;
  21. an arm-asymmetric skip raises the PAIR-BASE MISMATCH banner with the
      symmetric difference and marks the affected ΔS rows.

Run from the repo root:
  python -m pytest agent/test_d_downstream_analysis.py -v
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

import d_downstream_analysis as A  # noqa: E402
from extract_cw_triggers import sha256_text_file  # noqa: E402


LEGAL = '<tool_call>\n{"name": "%s", "arguments": {"a": 1}}\n</tool_call>'
TARGET_TOOL = "get_weather"


def _row(qid, offset, *, correct=False, skipped=False, skip_reason=None,
         terminal=False, offsets_available=None, **extra):
    if skipped:
        row = {
            "qid": qid,
            "session_id": qid.rsplit(":", 1)[0],
            "d_turn_offset": offset,
            "skipped": True,
            "skip_reason": skip_reason,
            "d_downstream_turns": 3,
        }
    else:
        name = TARGET_TOOL if correct else "wrong_tool"
        row = {
            "qid": qid,
            "session_id": qid.rsplit(":", 1)[0],
            "d_turn_offset": offset,
            "skipped": False,
            "prediction": LEGAL % name,
            "target": LEGAL % TARGET_TOOL,
            "target_tool_name": TARGET_TOOL,
            "tool_name_match": correct,
            "has_tool_call": True,
            "generated_tokens": 20,
            "generate_sec": 1.0,
            "d_ds_block_prefill_sec": 0.1,
            "d_ds_block_tokens": 50,
            "d_downstream_turns": 3,
        }
    if terminal:
        row["d_ds_terminal"] = True
        row["d_ds_offsets_available"] = 3 if offsets_available is None else offsets_available
    row.update(extra)
    return row


def _write_arm(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _three_arms(tmp_path, *, corr_re_oom=False, corr_re_offset0_oom=False):
    """Four single-span sessions; corr_re repairs s1/s2 at t*+1.

    With corr_re_oom=True, s1's corr_re group instead dies in an oom at
    offset 1 — the arm-asymmetric loss the pair-base check must surface.
    With corr_re_offset0_oom=True, s1's corr_re group is a single skipped
    offset-0 oom row (no terminal row — exactly what the driver leaves after
    a t* OOM in a K>0 run): the reconciliation must account for it, never
    render a false NO, and the banner must say oom@0, never 'scored'.
    """
    qids = [f"s{i}:0" for i in range(1, 5)]
    manifest = {
        "rule_version": "d_cw_v1",
        "batch": "batch-TF-test",
        "cw_qids": qids,
        "n_base_paired": 10,
    }

    def group(qid, arm):
        if arm == "corr_re" and corr_re_offset0_oom and qid == "s1:0":
            return [_row(qid, 0, skipped=True, skip_reason="oom")]
        rows = [_row(qid, 0, cache_tokens=40, gist_tokens=8)]
        if arm == "corr_re" and corr_re_oom and qid == "s1:0":
            rows.append(_row(qid, 1, skipped=True, skip_reason="oom",
                             terminal=True, offsets_available=1))
            return rows
        correct = arm == "corr_re" and qid in ("s1:0", "s2:0")
        rows.append(_row(qid, 1, correct=correct))
        rows.append(_row(qid, 2, skipped=True, skip_reason="d_ds_no_subsequent_turn"))
        rows.append(_row(qid, 3, skipped=True, skip_reason="d_ds_no_subsequent_turn",
                         terminal=True, offsets_available=1))
        return rows

    arms = {}
    for arm in ("none", "sham", "corr_re"):
        path = tmp_path / f"d_downstream_{arm}.jsonl"
        _write_arm(path, [row for qid in qids for row in group(qid, arm)])
        arms[arm] = A._load_downstream_arm(str(path))
    return arms, manifest


# --- 15. loader -------------------------------------------------------------


def test_load_downstream_arm_keys_and_dedup(tmp_path):
    path = tmp_path / "arm.jsonl"
    first = [
        _row("s1:0", 0),
        _row("s1:0", 1, correct=False),
        _row("s1:0", 2, skipped=True, skip_reason="oom"),
    ]
    retry = [
        _row("s1:0", 0),
        _row("s1:0", 1, correct=True),
        _row("s1:0", 2, skipped=True, skip_reason="d_ds_no_subsequent_turn"),
        _row("s1:0", 3, skipped=True, skip_reason="d_ds_no_subsequent_turn",
             terminal=True, offsets_available=1),
    ]
    _write_arm(path, first + retry)
    data = A._load_downstream_arm(str(path))
    assert set(data["rows"]) == {("s1:0", 0), ("s1:0", 1)}
    # last group wins: the retry's repaired offset-1 row, not the first pass
    assert data["rows"][("s1:0", 1)]["tool_name_match"] is True
    # the superseded group's oom never reaches the counters
    assert dict(data["skip_counts"]) == {"d_ds_no_subsequent_turn": 2}
    assert data["skip_reasons"][("s1:0", 2)] == "d_ds_no_subsequent_turn"
    assert data["offsets_available"] == {"s1:0": 1}
    assert data["n_undecodable"] == 0


def test_loader_skips_and_counts_undecodable_lines(tmp_path, caplog):
    """A crash mid-group-write leaves a partial JSON line that the driver's
    resume tolerates (it appends a clean retry group after it); the loader
    must read such a converged file — skip-and-count with a warning, never
    a JSONDecodeError that blocks smoke.ok / report generation."""
    import logging  # noqa: PLC0415

    path = tmp_path / "arm.jsonl"
    truncated = json.dumps(_row("s1:0", 1, correct=False))[:40]  # crash artifact
    lines = [
        json.dumps(_row("s1:0", 0)),
        truncated,
        json.dumps(_row("s1:0", 0)),
        json.dumps(_row("s1:0", 1, correct=True, terminal=True, offsets_available=1)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="d_downstream_analysis"):
        data = A._load_downstream_arm(str(path))
    assert data["n_undecodable"] == 1
    assert set(data["rows"]) == {("s1:0", 0), ("s1:0", 1)}
    assert data["rows"][("s1:0", 1)]["tool_name_match"] is True  # the retry group
    assert any("undecodable" in record.getMessage() for record in caplog.records)


# --- 16. hand-computed contrast ---------------------------------------------


def test_downstream_paired_contrast_hand_computed(tmp_path):
    arms, manifest = _three_arms(tmp_path)
    report = A.analyze(arms, manifest, reps=200, seed=0)
    primary = report["contrasts"][0]
    assert primary["contrast"].startswith("corr_re - none @ t*+1")
    assert "primary readout" in primary["contrast"]
    assert primary["n"] == 4
    # corr_re repairs s1 and s2; none repairs nothing: b=2, c=0.
    assert primary["b_left_only"] == 2
    assert primary["c_right_only"] == 0
    # exact McNemar: 2 * C(2,0) / 2^2
    assert primary["mcnemar_exact_p"] == pytest.approx(0.5)
    assert primary["diff_point_pp"] == pytest.approx(50.0)

    control = report["contrasts"][1]
    assert control["contrast"].startswith("sham - none @ t*+1")
    assert "nonspecific control" in control["contrast"]
    assert control["b_left_only"] == 0 and control["c_right_only"] == 0

    # seed wired through and reproducible
    assert report["bootstrap"]["seed"] == 0
    again = A.analyze(arms, manifest, reps=200, seed=0)
    assert again["contrasts"] == report["contrasts"]


# --- 16b. denominator reconciliation ------------------------------------------


def test_reconciliation_symmetric_fixture_all_yes(tmp_path):
    arms, manifest = _three_arms(tmp_path)
    report = A.analyze(arms, manifest, reps=100, seed=0)
    for arm_block in report["per_arm_offset"].values():
        for cell in arm_block["by_offset"].values():
            assert cell["reconciled"] is True
            assert cell["n_offset0_broken"] == 0


def test_reconciliation_offset0_oom_group_accounted(tmp_path):
    """An offset-0 oom leftover (driver writes the skip row, group is only
    retried on re-invocation) must reconcile — table 3's 'a NO means rows
    are unaccounted for' has to stay a true statement — and the pair-base
    banner must name the oom, never render the qid as 'scored'."""
    arms, manifest = _three_arms(tmp_path, corr_re_offset0_oom=True)
    report = A.analyze(arms, manifest, reps=100, seed=0)
    for arm_block in report["per_arm_offset"].values():
        for cell in arm_block["by_offset"].values():
            assert cell["reconciled"] is True
    corr = report["per_arm_offset"]["corr_re"]
    assert corr["n_offset0_broken"] == 1
    assert corr["n_terminal_groups"] == 3
    cell1 = corr["by_offset"][1]
    assert cell1["n_scored"] == 3
    assert cell1["n_broken_earlier"] == 1

    # the arm-asymmetric loss still raises the banner, with the true reason
    assert [entry["offset"] for entry in report["pair_base_mismatches"]] == [1]
    entry = report["pair_base_mismatches"][0]
    assert [item["qid"] for item in entry["symmetric_difference"]] == ["s1:0"]
    assert entry["symmetric_difference"][0]["skip_reasons"]["corr_re"] == "oom@0"
    assert entry["symmetric_difference"][0]["skip_reasons"]["sham"] is None

    markdown = A.render_markdown(report)
    assert "PAIR-BASE MISMATCH" in markdown
    assert "oom@0" in markdown
    assert "**NO**" not in markdown, "false unaccounted-rows alarms erode table 3"


# --- 17. footnote and exploratory note --------------------------------------


def test_tables_carry_footnote_and_exploratory_note(tmp_path):
    arms, manifest = _three_arms(tmp_path)
    report = A.analyze(
        arms, manifest, no_downstream_qids={"s1:0"}, reps=100, seed=0
    )
    markdown = A.render_markdown(report)
    foot_line = f"_{report['footnote']}_"
    assert markdown.count(foot_line) == 5, "every table carries the footnote"
    assert markdown.count("exploratory") >= 5, "every note is stamped exploratory"
    assert "corr_re − sham at t" in markdown  # registered primary contrast untouched
    assert "d-downstream" in markdown  # W&B ingestion note


# --- 18. out_prefix guard ---------------------------------------------------


def test_out_prefix_guard(tmp_path):
    with pytest.raises(SystemExit, match="frozen round"):
        A._assert_out_prefix_allowed(str(tmp_path / "d_r2"))
    with pytest.raises(SystemExit, match="frozen round"):
        A._assert_out_prefix_allowed(str(tmp_path / "d_r1"))
    with pytest.raises(SystemExit, match="frozen round"):
        A._assert_out_prefix_allowed(str(tmp_path / "d_r2" / "d_downstream_report"))
    A._assert_out_prefix_allowed(str(tmp_path / "d_downstream_report"))


# --- 19. offset-0 identity sentinel -----------------------------------------


def _identity_fixture(tmp_path, *, perturb=False):
    left = tmp_path / "d_downstream_none.jsonl"
    right = tmp_path / "battery_c2kv.jsonl"
    left_rows = []
    right_rows = []
    for index in (1, 2):
        qid = f"s{index}:0"
        left_rows.append(_row(qid, 0, cache_tokens=40 + index, gist_tokens=8))
        left_rows.append(_row(qid, 1, correct=True, terminal=True, offsets_available=1))
        right_rows.append({
            "qid": qid,
            "skipped": False,
            "prediction": LEGAL % "wrong_tool",
            "cache_tokens": 40 + index,
            "gist_tokens": 8,
        })
    if perturb:
        right_rows[0]["prediction"] = LEGAL % "another_tool"
    _write_arm(left, left_rows)
    _write_arm(right, right_rows)
    return left, right


def test_offset0_identity_sentinel(tmp_path, capsys):
    left, right = _identity_fixture(tmp_path)
    assert A.main(["--offset0_identity", str(left), str(right)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["n_compared"] == 2

    left, right = _identity_fixture(tmp_path, perturb=True)
    assert A.main(["--offset0_identity", str(left), str(right)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert result["mismatches"][0]["field"] == "prediction"


def test_offset0_identity_expect_n_closes_subset_blind_spot(tmp_path, capsys):
    """Without --expect_n the sentinel compares only qids present in LEFT, so
    silently lost t* rows still earn a pass; with it, coverage shortfall is
    a failure (full files pass the frozen trigger count, the smoke its 2)."""
    left, right = _identity_fixture(tmp_path)
    assert A.main(["--offset0_identity", str(left), str(right), "--expect_n", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True and result["expect_n"] == 2

    # drop one qid's group from LEFT: field-identity still holds on the
    # remainder, only --expect_n catches the loss
    kept = [
        line
        for line in Path(left).read_text(encoding="utf-8").splitlines()
        if json.loads(line)["qid"] != "s2:0"
    ]
    Path(left).write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert A.main(["--offset0_identity", str(left), str(right)]) == 0
    capsys.readouterr()
    assert A.main(["--offset0_identity", str(left), str(right), "--expect_n", "2"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert result["coverage_shortfall"] is True
    assert result["n_compared"] == 1 and result["expect_n"] == 2
    assert result["n_mismatches"] == 0


# --- 20. manifest binding ---------------------------------------------------


def test_manifest_binding_fatal(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"cw_qids": ["s1:0"], "n_base_paired": 10}), encoding="utf-8"
    )
    arm_path = tmp_path / "d_downstream_none.jsonl"
    _write_arm(arm_path, [
        _row("s1:0", 0, bundle_manifest_sha256="0" * 64),
        _row("s1:0", 1, correct=True, terminal=True, offsets_available=1,
             bundle_manifest_sha256="0" * 64),
    ])
    assert sha256_text_file(manifest_path) != "0" * 64
    with pytest.raises(SystemExit, match="different frozen"):
        A.main([
            "--arm", f"none={arm_path}",
            "--manifest", str(manifest_path),
            "--out_prefix", str(tmp_path / "d_downstream_report"),
        ])


# --- 21. pair-base mismatch banner ------------------------------------------


def test_pair_base_mismatch_banner(tmp_path):
    arms, manifest = _three_arms(tmp_path, corr_re_oom=True)
    report = A.analyze(arms, manifest, reps=100, seed=0)
    assert [entry["offset"] for entry in report["pair_base_mismatches"]] == [1]
    entry = report["pair_base_mismatches"][0]
    assert [item["qid"] for item in entry["symmetric_difference"]] == ["s1:0"]
    assert entry["symmetric_difference"][0]["skip_reasons"]["corr_re"] == "oom"
    assert entry["symmetric_difference"][0]["skip_reasons"]["sham"] is None

    markdown = A.render_markdown(report)
    assert "PAIR-BASE MISMATCH" in markdown
    assert "s1:0" in markdown
    # affected ΔS rows are marked
    offset1_rows = [
        line for line in markdown.splitlines() if "@ t*+1" in line and "|" in line
    ]
    assert offset1_rows and all("MISMATCH" in line for line in offset1_rows)

    # symmetric fixture: no banner, rows marked OK
    arms, manifest = _three_arms(tmp_path, corr_re_oom=False)
    report = A.analyze(arms, manifest, reps=100, seed=0)
    assert report["pair_base_mismatches"] == []
    assert "PAIR-BASE MISMATCH" not in A.render_markdown(report)
