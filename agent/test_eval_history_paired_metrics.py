"""The paired block and the instrumentation counters of
agent/eval_agent_history_c2kv.py's mode summaries.

``max_baseline_input_tokens`` only ever skips the uncompressed arms, so an
appworld_dev cell can score c2kv on 700 decision points and ``full`` on the
102 with the shortest histories (43/102 tool targets vs 558/700).  Reading
those two unpaired numbers as "the cost of compression" compares two different
test sets; the paired block re-scores every mode on the rows no mode skipped.

The harness module imports torch, so the function under test is compiled out
of the source rather than imported -- these assertions must hold on any host.
"""
import argparse
import ast
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

SOURCE = Path(__file__).resolve().parent / "eval_agent_history_c2kv.py"
WRAPPER = Path(__file__).resolve().parent / "eval_history_dev_c2kv_h200.sh"


def _load(*names: str):
    """Compile the named module-level functions out of the harness source.

    The harness imports torch at module scope, so it cannot be imported on a
    CPU-only host; these assertions must hold anywhere.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted, "helper moved or was renamed"
    namespace: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple,
        "Counter": Counter, "argparse": argparse,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return [namespace[name] for name in names]


def _load_paired():
    return _load("_attach_paired_metrics")[0]


def _load_summarize():
    return _load("_summarize_rows", "_attach_paired_metrics")[0]


def _args(**overrides) -> argparse.Namespace:
    base = {"model": "ckpt", "base_model": "base", "dataset_path": "ds", "split": "eval"}
    base.update(overrides)
    return argparse.Namespace(**base)


def _instrumented_row(qid, mode="c2kv", **overrides):
    row = {
        "qid": qid,
        "mode": mode,
        "ratio": 8,
        "skipped": False,
        "tool_name_match": False,
        "target_has_tool_call": True,
        "target_tool_name": "book",
        "has_tool_call": False,
        "exact_match": False,
        "response_type_match": False,
        "text_token_f1": 0.0,
        "rouge_l_f1": 0.0,
        "generated_tokens": 8,
        "doc_tokens": 0,
        "gist_tokens": 0,
        "compressed_history_tokens": 0,
        "system_truncated": False,
        "prompt_truncated": False,
        "generation_capped": False,
        "uncompressed": False,
    }
    row.update(overrides)
    return row


def _wrapper_keep_keys() -> Tuple[str, ...]:
    """The KEEP tuple the wrapper copies into the normalized summary.json."""
    text = WRAPPER.read_text(encoding="utf-8")
    match = re.search(r"^KEEP = \((.*?)^\)$", text, re.S | re.M)
    assert match, "KEEP tuple moved or was renamed in the wrapper"
    return tuple(
        literal
        for literal in re.findall(r'^\s*"([^"]+)",', match.group(1), re.M)
    )


def _row(qid, mode, *, skipped=False, hit=False):
    return {
        "qid": qid,
        "mode": mode,
        "ratio": 8 if mode != "full" else 1,
        "skipped": skipped,
        "tool_name_match": hit,
        "target_has_tool_call": True,
        "target_tool_name": "book",
        "has_tool_call": hit,
        "exact_match": False,
        "response_type_match": True,
        "text_token_f1": 0.5 if hit else 0.0,
    }


def _summaries(rows):
    out = []
    for mode, ratio in sorted({(r["mode"], r["ratio"]) for r in rows}):
        group = [r for r in rows if r["mode"] == mode and r["ratio"] == ratio]
        valid = [r for r in group if not r["skipped"]]
        out.append({
            "mode": mode, "ratio": ratio,
            "num_valid": len(valid),
            "tool_name_accuracy": (
                sum(1 for r in valid if r["tool_name_match"]) / len(valid) if valid else 0.0
            ),
        })
    return out


def test_paired_block_scores_the_common_population():
    attach = _load_paired()
    # full is skipped on exactly the two examples the compressed arm gets right
    rows = [
        _row("a", "c2kv", hit=True), _row("b", "c2kv", hit=True),
        _row("c", "c2kv", hit=False), _row("d", "c2kv", hit=False),
        _row("a", "full", skipped=True), _row("b", "full", skipped=True),
        _row("c", "full", hit=True), _row("d", "full", hit=True),
    ]
    summaries = _summaries(rows)
    attach(summaries, rows)
    by_mode = {s["mode"]: s for s in summaries}
    # the misleading unpaired reading: c2kv 0.50 on n=4 vs full 1.00 on n=2
    assert (by_mode["c2kv"]["num_valid"], by_mode["full"]["num_valid"]) == (4, 2)
    assert by_mode["c2kv"]["tool_name_accuracy"] == pytest.approx(0.5)
    # paired: both arms on the same two rows -> c2kv 0.0, full 1.0
    assert by_mode["c2kv"]["paired"]["n"] == by_mode["full"]["paired"]["n"] == 2
    assert by_mode["c2kv"]["paired"]["tool_name_accuracy"] == pytest.approx(0.0)
    assert by_mode["full"]["paired"]["tool_name_accuracy"] == pytest.approx(1.0)
    assert by_mode["c2kv"]["paired"]["n_unpaired"] == 2
    assert by_mode["full"]["paired"]["n_unpaired"] == 0


def test_paired_block_matches_unpaired_when_no_mode_skips():
    attach = _load_paired()
    rows = [
        _row("a", "c2kv", hit=True), _row("b", "c2kv", hit=False),
        _row("a", "full", hit=True), _row("b", "full", hit=True),
    ]
    summaries = _summaries(rows)
    attach(summaries, rows)
    for summary in summaries:
        assert summary["paired"]["n"] == summary["num_valid"] == 2
        assert summary["paired"]["tool_name_accuracy"] == pytest.approx(
            summary["tool_name_accuracy"]
        )


def test_paired_block_is_skipped_without_row_identity():
    attach = _load_paired()
    rows = [_row("a", "c2kv", hit=True), _row("a", "full", hit=True)]
    for row in rows:
        row.pop("qid")
    summaries = _summaries(rows)
    attach(summaries, rows)
    assert all("paired" not in summary for summary in summaries)


def test_summary_counts_the_silent_divergences():
    summarize = _load_summarize()
    rows = [
        # Prefix right-truncated, current turn left-truncated, decode stopped at
        # --max_new_tokens: three separate reasons this row's score is bounded
        # by the harness rather than by the model.
        _instrumented_row(
            "a",
            system_truncated=True,
            prompt_truncated=True,
            generation_capped=True,
            doc_tokens=400,
            gist_tokens=40,
            compressed_history_tokens=40,
        ),
        # A "compressed" row that carries no gist block at all.
        _instrumented_row(
            "b",
            uncompressed=True,
            doc_tokens=100,
            gist_tokens=0,
            compressed_history_tokens=100,
        ),
        _instrumented_row("c", skipped=True, skip_reason="system_overflow"),
    ]
    summary = summarize(_args(), rows)[0]
    assert summary["num_valid"] == 2 and summary["num_skipped"] == 1
    assert summary["skip_reasons"] == {"system_overflow": 1}
    assert summary["num_system_truncated"] == 1
    assert summary["num_prompt_truncated"] == 1
    assert summary["num_generation_capped"] == 1
    assert summary["num_uncompressed_rows"] == 1
    assert summary["num_compressed_rows"] == 1
    # The all-rows ratio is dragged towards 1.0 by the uncompressed row;
    # realized_ratio_on_compressed reports only the rows that were compressed.
    assert summary["token_weighted_actual_compression_ratio"] == pytest.approx(500 / 140)
    assert summary["realized_ratio_on_compressed"] == pytest.approx(10.0)


def test_realized_ratio_on_compressed_is_doc_token_weighted():
    summarize = _load_summarize()
    rows = [
        _instrumented_row("a", doc_tokens=1000, gist_tokens=100, compressed_history_tokens=100),
        _instrumented_row("b", doc_tokens=100, gist_tokens=50, compressed_history_tokens=50),
    ]
    summary = summarize(_args(), rows)[0]
    # Weighted by document tokens (1100/150), not the mean of the per-row
    # ratios (10 and 2 -> 6.0).
    assert summary["realized_ratio_on_compressed"] == pytest.approx(1100 / 150)
    assert summary["num_uncompressed_rows"] == 0


def test_counters_are_zero_and_present_when_nothing_was_truncated():
    summarize = _load_summarize()
    rows = [_instrumented_row("a", doc_tokens=10, gist_tokens=5, compressed_history_tokens=5)]
    summary = summarize(_args(), rows)[0]
    for key in (
        "num_system_truncated",
        "num_prompt_truncated",
        "num_generation_capped",
        "num_uncompressed_rows",
    ):
        assert summary[key] == 0, key


def test_wrapper_keeps_every_instrumentation_counter():
    # start_h200.sh phase_select reads ONLY the normalized summary.json, so a
    # counter the wrapper does not copy is invisible where it matters.
    keep = _wrapper_keep_keys()
    for key in (
        "num_system_truncated",
        "num_prompt_truncated",
        "num_generation_capped",
        "num_uncompressed_rows",
        "num_compressed_rows",
        "realized_ratio_on_compressed",
    ):
        assert key in keep, f"{key} missing from the wrapper's KEEP tuple"


def test_every_kept_key_is_actually_emitted_by_the_summary():
    summarize = _load_summarize()
    rows = [
        _instrumented_row("a", doc_tokens=10, gist_tokens=5, compressed_history_tokens=5),
        _instrumented_row("a", mode="full", doc_tokens=10),
    ]
    summary = summarize(_args(), rows)[0]
    missing = [key for key in _wrapper_keep_keys() if key not in summary]
    assert not missing, f"wrapper copies keys the harness never emits: {missing}"
