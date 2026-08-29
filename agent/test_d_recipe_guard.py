"""The doc-grid / dialect guard that stops task-D intervening on the wrong context.

A C->W trigger is a statement about one specific context: this doc grid, built
at this budget, by this harness.  Rebuilding it under different geometry and
then patching it measures something else, and nothing downstream reveals the
swap -- the qids still resolve, the arms still run, the numbers still look like
numbers.  These tests keep the guard honest.

torch-free: only the argparse namespace and the pure check function are used.

Run: python -m pytest agent/test_d_recipe_guard.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(os.path.dirname(_HERE), "python")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _guard():
    """Import the checker without pulling the driver's torch-bearing deps."""
    import ast
    import pathlib

    src = pathlib.Path(_HERE, "d_kv_intervene.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_recipe_matches_run"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ns: dict = {"Dict": dict, "Any": object, "argparse": argparse}
    exec(compile(module, "<guard>", "exec"), ns)  # noqa: S102 - test-local extraction
    return ns["_assert_recipe_matches_run"]


def _args(**over):
    base = dict(max_doc_length=768, max_doc_num=16, ratio=8, arm="corr",
                base_hybrid_top_k=0)
    base.update(over)
    return argparse.Namespace(**base)


def _manifest(**over):
    recipe = {"max_doc_length": 768, "max_doc_num": 16, "ratio": 8}
    recipe.update(over.pop("recipe", {}))
    man = {"kv_recipe": recipe, "source_dialects": {"history": 40}}
    man.update(over)
    return man


def test_matching_recipe_passes():
    _guard()(_manifest(), _args())


def test_doc_length_mismatch_is_fatal():
    """768 (history convention) vs 1024 (joint convention) is the live hazard."""
    with pytest.raises(SystemExit) as exc:
        _guard()(_manifest(), _args(max_doc_length=1024))
    msg = str(exc.value)
    assert "doc-grid mismatch" in msg
    assert "768" in msg and "1024" in msg


def test_ratio_mismatch_is_fatal():
    with pytest.raises(SystemExit) as exc:
        _guard()(_manifest(), _args(ratio=4))
    assert "ratio mismatch" in str(exc.value)


def test_doc_num_mismatch_is_fatal():
    """768/16 vs the joint 1024/24 convention: the row budget selects the
    history tail, so k* would name a different document."""
    with pytest.raises(SystemExit) as exc:
        _guard()(_manifest(), _args(max_doc_num=24))
    msg = str(exc.value)
    assert "max_doc_num" in msg
    assert "16" in msg and "24" in msg


def test_recipe_without_doc_num_skips_that_check():
    """Manifests frozen before max_doc_num entered the recipe stay runnable."""
    man = _manifest()
    del man["kv_recipe"]["max_doc_num"]
    _guard()(man, _args(max_doc_num=24))


def test_joint_dialect_triggers_are_refused():
    """The extractor parses joint rows; the D harness cannot reproduce them."""
    man = _manifest(source_dialects={"history": 10, "joint": 3})
    with pytest.raises(SystemExit) as exc:
        _guard()(man, _args())
    msg = str(exc.value)
    assert "non-history rows" in msg
    assert "joint" in msg


def test_pure_history_dialect_passes():
    _guard()(_manifest(source_dialects={"history": 12}), _args())


def test_zero_count_foreign_dialect_is_not_fatal():
    """A dialect counter that reached zero is not evidence of contamination."""
    _guard()(_manifest(source_dialects={"history": 12, "joint": 0}), _args())


def test_absent_recipe_fields_skip_their_checks():
    """Manifests frozen before the recipe carried geometry stay runnable."""
    _guard()({"kv_recipe": {}, "source_dialects": {"history": 5}}, _args())
    _guard()({}, _args())


def test_full_arm_is_still_checked():
    """The full arm overrides ratio to 1 at call time but shares the frozen set."""
    with pytest.raises(SystemExit):
        _guard()(_manifest(), _args(arm="full", max_doc_length=1024))


def test_hybrid_base_mismatch_is_fatal():
    """A pure-C2KV trigger set must not be repaired on a hybrid base.

    Every manifest frozen before Block 3.1 predates the field, so its absence
    means base_hybrid_top_k=0 -- the guard has to catch the old-manifest case,
    which is the one that will actually happen.
    """
    with pytest.raises(SystemExit) as exc:
        _guard()(_manifest(), _args(base_hybrid_top_k=3))
    msg = str(exc.value)
    assert "base-layer mismatch" in msg
    assert "base_hybrid_top_k=0" in msg and "passes 3" in msg


def test_hybrid_base_match_passes():
    _guard()(_manifest(recipe={"base_hybrid_top_k": 3}), _args(base_hybrid_top_k=3))


def test_hybrid_manifest_run_on_c2kv_base_is_fatal():
    """The mirror image: a hybrid trigger set replayed on the pure base."""
    with pytest.raises(SystemExit) as exc:
        _guard()(_manifest(recipe={"base_hybrid_top_k": 3}), _args(base_hybrid_top_k=0))
    assert "base-layer mismatch" in str(exc.value)
