# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/extract_cw_triggers.py.

No torch, no weights, no dataset: transitions are exercised with synthetic
rows, and the only heavyweight path (--bind_docs) is never touched.

Coverage:
a. local scorers and their cross-check against the harness fields;
b. all four transition cells from synthetic paired rows;
c. duplicate qid is fatal; unpaired qids are dropped, not guessed;
d. dialect handling: joint rows filtered by condition, history rows rejected
   when a condition is demanded;
e. manifest determinism (same inputs -> byte-identical manifest) and the
   bundle-sha binding;
f. torch-gated equivalence lock against the real harness _extract_tool_name;
g. skipif-gated integration over the 594 real r4 rows (schema/flow only —
   the numbers in those files are void and nothing here reads them).

Run from the repo root:
  python -m pytest agent/test_extract_cw_triggers.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import extract_cw_triggers as X  # noqa: E402


CALL = '<tool_call>\n{"name": "%s", "arguments": {"a": 1}}\n</tool_call>'
TARGET = CALL % "get_weather"


def _row(qid, tool, *, condition=None, harness_match=None, **extra):
    text = CALL % tool if tool else "I am not going to call anything."
    row = {
        "qid": qid,
        "session_id": qid.rsplit(":", 1)[0],
        "skipped": False,
        "prediction": text,
        "target": TARGET,
        "target_tool_name": "get_weather",
        "has_tool_call": bool(tool),
        "tool_name_match": (tool == "get_weather") if harness_match is None else harness_match,
        "doc_chunks": 4,
    }
    if condition is not None:
        row["condition"] = condition
    row.update(extra)
    return row


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _args(tmp_path, full_path, compressed_path, **overrides):
    argv = [
        "--full_rows", full_path,
        "--compressed_rows", compressed_path,
        "--batch", "batch-TF",
        "--s_metric", "tool_name_match",
        "--out_bundles", str(tmp_path / "bundles.jsonl"),
        "--out_manifest", str(tmp_path / "manifest.json"),
        "--out_doc_table", str(tmp_path / "d_doc_ids.json"),
        "--ckpt_path", "./checkpoints/fake",
        "--model_sha", "modelsha",
        "--eval_code_sha", "codesha",
        "--ratio", "8",
        "--chunk_policy", "pilot_v1",
        "--seed", "20260815",
        "--decode", "greedy",
    ]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    return X.parse_args(argv)


# --- a. scorers -------------------------------------------------------------


def test_extract_tool_name_variants():
    assert X.extract_tool_name(CALL % "search_files") == "search_files"
    assert X.extract_tool_name('{"tool_name": "abc"}') == "abc"
    assert X.extract_tool_name('{"function": {"name": "nested"}}') == "nested"
    # Broken JSON still yields the name via the field regex.
    assert X.extract_tool_name('<tool_call>{"name": "broken", </tool_call>') == "broken"
    assert X.extract_tool_name("") is None
    assert X.extract_tool_name("plain prose, no call at all") is None


def test_has_tool_call():
    assert X.has_tool_call(CALL % "x")
    assert X.has_tool_call("Action:\nsomething")
    assert not X.has_tool_call("just text")


def test_score_cross_checks_harness_fields(caplog):
    good = _row("s:1", "get_weather")
    assert X._score(good)["harness_metric_agrees"] is True
    lying = _row("s:2", "get_weather", harness_match=False)
    result = X._score(lying)
    assert result["correct"] is True
    assert result["harness_metric_agrees"] is False


def test_score_falls_back_to_target_text():
    row = _row("s:3", "get_weather")
    row.pop("target_tool_name")
    assert X._score(row)["target_tool_name"] == "get_weather"


# --- b/c. transitions and pairing -------------------------------------------


def test_all_four_transition_cells(tmp_path):
    full = [
        _row("s:1", "get_weather"),   # C -> C
        _row("s:2", "get_weather"),   # C -> W
        _row("s:3", "wrong_tool"),    # W -> C
        _row("s:4", "wrong_tool"),    # W -> W
    ]
    compressed = [
        _row("s:1", "get_weather"),
        _row("s:2", "wrong_tool"),
        _row("s:3", "get_weather"),
        _row("s:4", "other_tool"),
    ]
    args = _args(
        tmp_path,
        _write(tmp_path / "full.jsonl", full),
        _write(tmp_path / "comp.jsonl", compressed),
    )
    manifest = X.run(args)
    assert manifest["transitions"] == {"C->C": 1, "C->W": 1, "W->C": 1, "W->W": 1}
    assert manifest["n_base_paired"] == 4
    assert manifest["cw_qids"] == ["s:2"]
    bundles = [
        json.loads(line)
        for line in Path(args.out_bundles).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["transition"] == "C->W"
    assert bundle["trigger_source"] == "oracle"
    assert bundle["t_star"] is None
    assert bundle["turn"] == 4
    assert bundle["step_index_t"] == 2
    assert bundle["no_downstream"] is False
    assert bundle["doc_ids_sha256"] == "fingerprint_pending"
    assert bundle["kv_recipe"]["ratio"] == 8
    assert bundle["kv_recipe"]["chunk_policy"] == "pilot_v1"
    assert bundle["kv_recipe"]["eval_code_sha"] == "codesha"
    # The recipe must be enough to rebuild the prefix; no KV is stored.
    assert "cache" not in bundle and "kv" not in bundle


def test_no_downstream_flag_for_single_doc(tmp_path):
    full = [_row("s:1", "get_weather", history_doc_chunks=1)]
    compressed = [_row("s:1", "wrong_tool", history_doc_chunks=1)]
    args = _args(
        tmp_path,
        _write(tmp_path / "full.jsonl", full),
        _write(tmp_path / "comp.jsonl", compressed),
    )
    X.run(args)
    bundle = json.loads(Path(args.out_bundles).read_text(encoding="utf-8").strip())
    assert bundle["turn"] == 1
    assert bundle["no_downstream"] is True


def test_duplicate_qid_is_fatal(tmp_path):
    path = _write(tmp_path / "dup.jsonl", [_row("s:1", "get_weather"), _row("s:1", "wrong_tool")])
    with pytest.raises(SystemExit, match="duplicate qid"):
        X._load_rows_by_qid([path], None, "full")


def test_skipped_rows_are_dropped(tmp_path):
    rows = [_row("s:1", "get_weather"), dict(_row("s:2", "x"), skipped=True, skip_reason="oom")]
    loaded, stats = X._load_rows_by_qid([_write(tmp_path / "s.jsonl", rows)], None, "full")
    assert set(loaded) == {"s:1"}
    assert stats["n_skipped"] == 1


def test_unpaired_qids_are_excluded(tmp_path):
    full = [_row("s:1", "get_weather"), _row("s:9", "get_weather")]
    compressed = [_row("s:1", "wrong_tool")]
    args = _args(
        tmp_path,
        _write(tmp_path / "full.jsonl", full),
        _write(tmp_path / "comp.jsonl", compressed),
    )
    manifest = X.run(args)
    assert manifest["n_base_paired"] == 1
    assert manifest["cw_qids"] == ["s:1"]


# --- d. dialects ------------------------------------------------------------


def test_joint_dialect_condition_filter(tmp_path):
    rows = [
        _row("s:1", "get_weather", condition="joint"),
        _row("s:1", "wrong_tool", condition="tool_only"),
        _row("s:2", "get_weather", condition="joint"),
    ]
    path = _write(tmp_path / "joint.jsonl", rows)
    loaded, stats = X._load_rows_by_qid([path], "joint", "compressed")
    assert set(loaded) == {"s:1", "s:2"}
    assert stats["n_condition_filtered"] == 1
    assert stats["dialects"] == {"joint": 3}


def test_condition_on_history_dialect_is_fatal(tmp_path):
    path = _write(tmp_path / "hist.jsonl", [_row("s:1", "get_weather")])
    with pytest.raises(SystemExit, match="history-dialect"):
        X._load_rows_by_qid([path], "joint", "compressed")


def test_history_dialect_needs_no_condition(tmp_path):
    path = _write(tmp_path / "hist.jsonl", [_row("s:1", "get_weather")])
    loaded, stats = X._load_rows_by_qid([path], None, "full")
    assert set(loaded) == {"s:1"}
    assert stats["dialects"] == {"history": 1}


# --- e. manifest determinism ------------------------------------------------


def test_manifest_is_deterministic(tmp_path):
    full = [_row("s:1", "get_weather"), _row("s:2", "get_weather")]
    compressed = [_row("s:1", "wrong_tool"), _row("s:2", "get_weather")]
    full_path = _write(tmp_path / "full.jsonl", full)
    comp_path = _write(tmp_path / "comp.jsonl", compressed)
    first = X.run(_args(tmp_path, full_path, comp_path))
    first_text = Path(tmp_path / "manifest.json").read_text(encoding="utf-8")
    second = X.run(_args(tmp_path, full_path, comp_path))
    second_text = Path(tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert first == second
    assert first_text == second_text
    assert first["sources"]["full_rows"][0]["sha256"] == X._sha256_file(Path(full_path))
    assert first["bundles_sha256"] == X._sha256_file(Path(tmp_path / "bundles.jsonl"))
    assert first["doc_binding"] == "fingerprint_pending"


def test_manifest_sha_changes_when_a_source_changes(tmp_path):
    full_path = _write(tmp_path / "full.jsonl", [_row("s:1", "get_weather")])
    comp_path = _write(tmp_path / "comp.jsonl", [_row("s:1", "wrong_tool")])
    before = X.run(_args(tmp_path, full_path, comp_path))["sources"]["compressed_rows"][0]["sha256"]
    _write(tmp_path / "comp.jsonl", [_row("s:1", "other_tool")])
    after = X.run(_args(tmp_path, full_path, comp_path))["sources"]["compressed_rows"][0]["sha256"]
    assert before != after


# --- f. equivalence lock against the harness scorer -------------------------


def test_local_scorer_matches_harness(tmp_path):
    """The local regex copies must agree with the module they were copied from."""
    pytest.importorskip("torch")
    from eval_agent_tool_definition_c2kv import _extract_tool_name as harness_name
    from eval_agent_history_c2kv import _has_tool_call as harness_call

    samples = [
        "",
        "plain prose",
        TARGET,
        CALL % "a.b:c-d",
        '<tool_call>{"name": "broken", </tool_call>',
        '{"function": {"name": "nested"}}',
        '{"tool_name": "abc"}',
        "Action:\n" + (CALL % "two"),
        (CALL % "first") + "\n" + (CALL % "second"),
    ]
    for sample in samples:
        assert X.extract_tool_name(sample) == harness_name(sample), sample
        assert X.has_tool_call(sample) == harness_call(sample), sample


# --- g. integration over the real r4 rows -----------------------------------


def _audit_dir():
    override = os.environ.get("C2KV_AUDIT_DIR")
    candidates = [Path(override)] if override else []
    candidates += [
        _REPO_ROOT / ".audit_pr7_56179dd",
        Path(r"C:/Users/yl998/Documents/programming/c2kv/.audit_pr7_56179dd"),
    ]
    for candidate in candidates:
        if (candidate / "results" / "r4").is_dir():
            return candidate
    return None


def _r4_arm_files(arm):
    root = _audit_dir()
    if root is None:
        return []
    return sorted(str(p) for p in (root / "results" / "r4" / arm).glob("*.jsonl"))


@pytest.mark.skipif(_audit_dir() is None, reason="r4 audit rows not present on this machine")
def test_integration_on_real_r4_rows(tmp_path):
    """Schema/flow only. The numbers in these files are void by decision;
    nothing here asserts on any metric value."""
    plain = _r4_arm_files("d_plain")
    typed = _r4_arm_files("d_typed")
    if not plain or not typed:
        pytest.skip("d_plain / d_typed row files missing")
    argv = [
        "--full_rows", *plain,
        "--compressed_rows", *typed,
        "--batch", "r4-void",
        "--out_bundles", str(tmp_path / "bundles.jsonl"),
        "--out_manifest", str(tmp_path / "manifest.json"),
        "--out_doc_table", str(tmp_path / "d_doc_ids.json"),
        "--ckpt_path", "./checkpoints/void",
        "--model_sha", "void",
        "--eval_code_sha", "void",
        "--chunk_policy", "void",
    ]
    manifest = X.run(X.parse_args(argv))
    assert manifest["n_base_paired"] > 0
    assert sum(manifest["transitions"].values()) == manifest["n_base_paired"]
    assert manifest["n_cw"] == len(manifest["cw_qids"])
    assert len(set(manifest["cw_qids"])) == manifest["n_cw"]
    lines = [
        line for line in Path(tmp_path / "bundles.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == manifest["n_cw"]
    required = {
        "bundle_id", "qid", "turn", "step_index_t", "t_star", "trigger_source",
        "transition", "no_downstream", "doc_ids_sha256", "kv_recipe", "source",
        "target_known", "target_in_grid",
    }
    for line in lines:
        bundle = json.loads(line)
        assert required <= set(bundle), required - set(bundle)
        assert bundle["transition"] == "C->W"
