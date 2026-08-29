"""The hybrid prefix layout resolver.

`_build_hybrid_prefix` used to build `S -> raw recent tail -> older gists`
unconditionally (hybrid_full_after_c2kv defaulted to False), i.e. the model saw
the conversation out of order and no prefix-cache reuse was possible.  The
chronological branch existed but was unusable, because the tail was prefilled
through the tool-definition eval's `_prefill_tokens_with_cache`, whose
attention mask is sized from `past_length + input_length` (logical positions) --
wrong the moment the cache holds gists, which is exactly that branch.  See
configs/bdf_pilot/d_prereg.md "Suffix recompute".

These tests pin the resolver, which decides which layout a run gets.

torch-free: the function is extracted by AST so importing the harness (and
torch) is not required.

Run: python -m pytest agent/test_hybrid_layout.py -v
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolver():
    src = pathlib.Path(_HERE, "eval_agent_history_c2kv.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    layouts = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "HYBRID_LAYOUTS" for t in node.targets)
    )
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_hybrid_layout"
    )
    ns: dict = {"argparse": argparse, "SystemExit": SystemExit}
    exec(compile(ast.Module(body=[layouts, fn], type_ignores=[]), "<layout>", "exec"), ns)
    return ns["_hybrid_layout"], ns["HYBRID_LAYOUTS"]


def test_default_is_chronological():
    """Neither flag passed: the specified algorithm, not the legacy order."""
    resolve, _ = _resolver()
    assert resolve(argparse.Namespace()) == "chronological"
    assert resolve(argparse.Namespace(hybrid_layout=None,
                                      hybrid_full_after_c2kv=None)) == "chronological"


def test_explicit_layout_wins():
    resolve, layouts = _resolver()
    for layout in layouts:
        assert resolve(argparse.Namespace(hybrid_layout=layout)) == layout


def test_deprecated_bool_still_reproduces_archived_runs():
    """`--hybrid_full_after_c2kv False` is how every pre-fix row was produced."""
    resolve, _ = _resolver()
    args = argparse.Namespace(hybrid_layout=None, hybrid_full_after_c2kv=False)
    assert resolve(args) == "legacy_tail_first"
    args = argparse.Namespace(hybrid_layout=None, hybrid_full_after_c2kv=True)
    assert resolve(args) == "chronological"


def test_unknown_layout_fails_loudly():
    resolve, _ = _resolver()
    with pytest.raises(SystemExit) as exc:
        resolve(argparse.Namespace(hybrid_layout="tail_first"))
    assert "unknown --hybrid_layout" in str(exc.value)


def test_both_layouts_are_reachable():
    """A silent single-layout collapse would hide the A/B entirely."""
    _resolve, layouts = _resolver()
    assert set(layouts) == {"chronological", "legacy_tail_first"}


def test_hybrid_builder_uses_the_gist_aware_prefill():
    """The tail prefill must be the maybe_gist variant, per d_prereg.md.

    A regression here is invisible in results: the run still completes, the
    numbers still look like numbers, and only the mask is wrong.
    """
    src = pathlib.Path(_HERE, "eval_agent_history_c2kv.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_hybrid_prefix"
    )
    called = {
        node.func.id for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_prefill_tokens_with_cache_maybe_gist" in called
    assert "_prefill_tokens_with_cache" not in called


def test_tail_slice_is_clamped_to_history_length():
    """k > len(history) must not silently produce a full-arm row labelled hybrid."""
    src = pathlib.Path(_HERE, "eval_agent_history_c2kv.py").read_text(encoding="utf-8")
    fn_src = src[src.index("def _build_hybrid_prefix"):]
    fn_src = fn_src[: fn_src.index("\n@torch.inference_mode()", 10)]
    assert "min(int(full_count), len(history))" in fn_src
    assert "hybrid_degenerate_to_full" in fn_src
