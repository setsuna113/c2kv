"""The paired block of agent/eval_agent_history_c2kv.py's mode summaries.

``max_baseline_input_tokens`` only ever skips the uncompressed arms, so an
appworld_dev cell can score c2kv on 700 decision points and ``full`` on the
102 with the shortest histories (43/102 tool targets vs 558/700).  Reading
those two unpaired numbers as "the cost of compression" compares two different
test sets; the paired block re-scores every mode on the rows no mode skipped.

The harness module imports torch, so the function under test is compiled out
of the source rather than imported -- these assertions must hold on any host.
"""
import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

SOURCE = Path(__file__).resolve().parent / "eval_agent_history_c2kv.py"


def _load_paired():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"_attach_paired_metrics"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted, "paired helper moved or was renamed"
    namespace: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_attach_paired_metrics"]


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
