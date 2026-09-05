# -*- coding: utf-8 -*-
"""Unit tests for agent/t33_labels.py (survey item 4.0-1).

Two things are pinned here:
  1. the label function + manifest cross-check (synthetic and, when the frozen
     r2 files are present, the real 900/93/68 census);
  2. the leakage guard — the historical specimens ``a_made_call`` and
     ``tool_name_match`` MUST be refused (the 0.0162-oriented accident is the
     live example of why this guard exists).

Run from the repo root:
  python -m pytest agent/test_t33_labels.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("agent",):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t33_labels import (  # noqa: E402
    KNOWN_LEAK_COLUMNS,
    build_label_frame,
    census,
    cw_label,
    guard_columns,
    join_arms,
    parse_fail_baseline,
)


def _row(qid, session, match, pred="x", target_has_call=True, gen=50):
    return {
        "qid": qid,
        "session_id": session,
        "tool_name_match": match,
        "prediction": pred,
        "target_has_tool_call": target_has_call,
        "generated_tokens": gen,
        "skipped": False,
    }


def _synthetic():
    full = [
        _row("s1:a", "s1", True),
        _row("s1:b", "s1", True),
        _row("s2:a", "s2", False),  # W->*
        _row("s2:b", "s2", False),  # W->*
    ]
    c2kv = [
        _row("s1:a", "s1", False),  # C->W
        _row("s1:b", "s1", True),   # C->C
        _row("s2:a", "s2", True),   # W->C
        _row("s2:b", "s2", False),  # W->W
    ]
    return full, c2kv


def _manifest():
    return {"cw_qids": ["s1:a"], "kv_recipe": {"max_new_tokens": 128},
            "transitions": {"C->C": 1, "C->W": 1}}


def test_cw_label_values():
    full, c2kv = _synthetic()
    labels = [cw_label(f, c) for f, c in zip(full, c2kv)]
    assert labels == [1, 0, None, None]


def test_label_frame_cross_checks_manifest():
    full, c2kv = _synthetic()
    frame = build_label_frame(join_arms(full, c2kv), _manifest())
    by_qid = {r["qid"]: r for r in frame}
    assert by_qid["s1:a"]["label_cw"] == 1
    assert by_qid["s1:b"]["label_cw"] == 0
    assert by_qid["s2:a"]["label_cw"] is None
    # three-valued deferral target (label side only)
    assert by_qid["s2:a"]["z_deferral"] == -1
    assert by_qid["s1:a"]["z_deferral"] == 1
    assert by_qid["s1:b"]["z_deferral"] == 0


def test_manifest_disagreement_raises():
    full, c2kv = _synthetic()
    bad = _manifest()
    bad["cw_qids"] = ["s1:b"]  # wrong qid
    with pytest.raises(ValueError, match="disagrees"):
        build_label_frame(join_arms(full, c2kv), bad)


def test_join_detects_qid_mismatch():
    full, c2kv = _synthetic()
    c2kv[0]["qid"] = "s1:zzz"
    with pytest.raises(ValueError, match="qid sets differ"):
        join_arms(full, c2kv)


def test_census_counts():
    full, c2kv = _synthetic()
    frame = build_label_frame(join_arms(full, c2kv), _manifest())
    stats = census(frame, _manifest())
    assert stats["n_paired"] == 4
    assert stats["n_cw"] == 1 and stats["n_cc"] == 1
    assert stats["n_sessions"] == 2
    assert stats["n_cw_sessions"] == 1


def test_censored_flag_uses_manifest_cap():
    full, c2kv = _synthetic()
    c2kv[0]["generated_tokens"] = 128
    frame = build_label_frame(join_arms(full, c2kv), _manifest())
    by_qid = {r["qid"]: r for r in frame}
    assert by_qid["s1:a"]["censored_at_cap"] is True
    assert by_qid["s1:b"]["censored_at_cap"] is False


# --- the leakage guard ----------------------------------------------------


@pytest.mark.parametrize("col", KNOWN_LEAK_COLUMNS)
def test_guard_catches_known_leak_columns(col):
    with pytest.raises(ValueError, match="refused"):
        guard_columns([col])


def test_guard_catches_full_arm_marker():
    with pytest.raises(ValueError, match="refused"):
        guard_columns(["gist_tokens", "full_prefill_sec"])
    with pytest.raises(ValueError, match="refused"):
        guard_columns(["logp_prefix_full"])


def test_guard_allows_neutral_and_compressed_arm_columns():
    guard_columns([
        "gist_tokens", "doc_chunks", "doc_tokens", "n_docs_kept", "dropped_docs",
        "actual_compression_ratio", "kept_history_tokens", "hybrid_top_k",
        "session_id", "qid", "prediction", "generated_tokens",
        "entropy_name_max", "gnll_smt", "flare_min_p_name", "margin_name_first",
        "ic_uniform", "surprise_min_k", "gzip_ratio_mean", "hoyer_k_mean",
    ])


def test_guard_message_names_the_column():
    with pytest.raises(ValueError, match="a_made_call"):
        guard_columns(["a_made_call"])


# --- parse-failure baseline ------------------------------------------------

def test_parse_fail_baseline_fires_on_unparseable():
    assert parse_fail_baseline("Action:\n<tool_call>\n" + '{"name":"mcp__x","argu', True) is True
    assert parse_fail_baseline("plain text, no call at all", True) is True


def test_parse_fail_baseline_silent_on_parseable():
    ok = 'Action:\n<tool_call>\n{"name":"mcp__x","arguments":{"a":"b"}}'
    assert parse_fail_baseline(ok, True) is False


def test_parse_fail_baseline_never_fires_when_no_call_expected():
    assert parse_fail_baseline("anything", False) is False


# --- real frozen battery (integration) -------------------------------------

_REAL = [
    _REPO_ROOT / "results" / "bdf_pilot" / "d_r2" / "battery_full.jsonl",
    _REPO_ROOT / "results" / "bdf_pilot" / "d_r2" / "battery_c2kv.jsonl",
    _REPO_ROOT / "configs" / "bdf_pilot" / "d_cw_manifest_r2.json",
]


@pytest.mark.skipif(not all(p.exists() for p in _REAL), reason="frozen r2 battery not checked out")
def test_real_battery_census_matches_prereg():
    from t33_labels import load_jsonl
    manifest = json.loads(_REAL[2].read_text(encoding="utf-8"))
    frame = build_label_frame(
        join_arms(load_jsonl(str(_REAL[0])), load_jsonl(str(_REAL[1]))), manifest)
    stats = census(frame, manifest)
    assert stats["n_paired"] == 900
    assert stats["n_sessions"] == 227
    assert stats["n_cw"] == 93 and stats["n_cw_sessions"] == 72
    assert stats["n_cc"] == 68 and stats["n_cc_sessions"] == 46
    assert stats["base_rate"] == pytest.approx(0.1033, abs=1e-4)
