# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/build_joint_medium_plan.py.

No real dataset and no network: the planning core (``plan_recipes``) is pure —
pools are synthetic ``PoolEntry`` lists with known token estimates, so the
whitespace fake tokenizer is never even needed for the ratio math.  The order
files are validated with the trainer's own ``_apply_example_order_file``
against JointExample stubs carrying the pool qids.

Coverage:
a. ``parse_recipe``: parsing, declared order, share-sum validation;
b. ratio realization: per-family realized shares track the recipe within
   crossing-example granularity;
c. ``interleave_families``: declared-order round-robin, shorter families stop;
d. removal lists: bare-list and dedup-dict shapes, session-id AND qid entries,
   family-prefix-stripped matching (uuid / trajectory_id / qa row id);
e. repeat variants: qid uniqueness, ``recommended_epochs`` math;
f. order files pass ``_apply_example_order_file`` (unique, all loadable);
g. determinism: identical plans across runs; token-cache roundtrip + stamp
   invalidation; shortfall reporting on undersized pools.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest agent/test_build_joint_medium_plan.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make python/ and agent/ importable when pytest is invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_joint_medium_plan as bjmp  # noqa: E402
from build_joint_medium_plan import (  # noqa: E402
    PoolEntry,
    _removal_match_keys,
    apply_removals,
    interleave_families,
    load_removal_identifiers,
    parse_recipe,
    plan_recipes,
    write_outputs,
)
from train.train_data_joint import JointExample  # noqa: E402
from train_joint_next_action_c2kv import _apply_example_order_file  # noqa: E402


# ---------------------------------------------------------------------------
# parse_recipe.
# ---------------------------------------------------------------------------


def test_parse_recipe_ok_and_order_preserved():
    recipe = parse_recipe("d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05")
    assert recipe.name == "d_multi"
    assert recipe.shares == (("qa", 0.2), ("traces", 0.5), ("toucan", 0.25), ("openswe", 0.05))


def test_parse_recipe_validation():
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_recipe("bad=qa:0.2,traces:0.2")
    with pytest.raises(ValueError, match="unknown family"):
        parse_recipe("bad=mystery:1.0")
    with pytest.raises(ValueError, match="name=family"):
        parse_recipe("no-equals")
    with pytest.raises(ValueError, match="family:share"):
        parse_recipe("bad=nocolon")


# ---------------------------------------------------------------------------
# Ratio realization / interleave / determinism.
# ---------------------------------------------------------------------------


def _pool(family, count, tokens=100):
    # Uniform synthetic pool: qids carry the family prefix like the real
    # sources (traces qids are bare session:span).
    entries = []
    for index in range(count):
        if family == "traces":
            qid = f"sess-{index // 3}:{index}"
            session_id = f"sess-{index // 3}"
        else:
            qid = f"{family}:id-{index}:u0" if family == "toucan" else f"{family}:id-{index}"
            session_id = qid.rsplit(":", 1)[0] if family == "toucan" else qid
        entries.append(PoolEntry(qid=qid, session_id=session_id, estimated_tokens=tokens))
    return entries


def test_ratio_realization_within_tolerance():
    pools = {"qa": _pool("qa", 50), "traces": _pool("traces", 100)}
    recipes = [parse_recipe("r=qa:0.2,traces:0.8")]
    results = plan_recipes(pools, recipes, budget_estimated_tokens=5000, order_seed=42)
    plan = results["r"]["plan"]
    # Quotas: qa 1000 (10 entries), traces 4000 (40 entries) — exact with the
    # uniform 100-token entries; tolerance guards the crossing-example rule.
    assert plan["families"]["qa"]["examples"] == 10
    assert plan["families"]["traces"]["examples"] == 40
    assert abs(plan["families"]["qa"]["realized_share"] - 0.2) < 0.02
    assert abs(plan["families"]["traces"]["realized_share"] - 0.8) < 0.02
    assert plan["total_estimated_tokens"] == 5000
    assert plan["order_examples"] == 50
    # Sampled subsets are strict subsets of the pools.
    qa_qids = {entry.qid for entry in pools["qa"]}
    traces_qids = {entry.qid for entry in pools["traces"]}
    assert {qid for qid in results["r"]["order"] if qid in qa_qids} <= qa_qids
    assert {qid for qid in results["r"]["order"] if qid in traces_qids} <= traces_qids


def test_interleave_families_declared_order_round_robin():
    order = interleave_families([
        ("qa", ["q0", "q1"]),
        ("traces", ["t0", "t1", "t2"]),
        ("toucan", ["c0"]),
    ])
    assert order == ["q0", "t0", "c0", "q1", "t1", "t2"]


def test_interleave_inside_plan_respects_recipe_family_order():
    pools = {"qa": _pool("qa", 2), "traces": _pool("traces", 2)}
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    results = plan_recipes(pools, recipes, budget_estimated_tokens=400, order_seed=42)
    order = results["r"]["order"]
    qa_qids = {entry.qid for entry in pools["qa"]}
    # Declared order is qa first: rounds alternate qa, traces, qa, traces.
    assert order[0] in qa_qids
    assert order[1] not in qa_qids
    assert order[2] in qa_qids
    assert order[3] not in qa_qids


def test_plan_is_deterministic():
    pools = {"qa": _pool("qa", 30), "traces": _pool("traces", 60), "toucan": _pool("toucan", 20)}
    recipes = [parse_recipe("r=qa:0.2,traces:0.5,toucan:0.3")]
    kwargs = dict(budget_estimated_tokens=8000, order_seed=42, repeat_unique_tokens=2000)
    first = plan_recipes(pools, recipes, **kwargs)
    second = plan_recipes(pools, recipes, **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # A different order seed picks a different sample (overwhelmingly likely).
    third = plan_recipes(pools, recipes, **{**kwargs, "order_seed": 43})
    assert third["r"]["order"] != first["r"]["order"]


def test_missing_family_pool_is_a_hard_error():
    pools = {"qa": _pool("qa", 5)}
    recipes = [parse_recipe("r=qa:0.5,toucan:0.5")]
    with pytest.raises(ValueError, match="toucan"):
        plan_recipes(pools, recipes, budget_estimated_tokens=1000)


def test_shortfall_reported_when_pool_undersized():
    pools = {"qa": _pool("qa", 3), "traces": _pool("traces", 5)}  # 800 tokens total
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    results = plan_recipes(pools, recipes, budget_estimated_tokens=10000, order_seed=42)
    plan = results["r"]["plan"]
    assert plan["families"]["qa"]["shortfall_estimated_tokens"] == 5000 - 300
    assert plan["families"]["traces"]["shortfall_estimated_tokens"] == 5000 - 500
    # Short pools are taken in full.
    assert plan["families"]["qa"]["examples"] == 3
    assert plan["order_examples"] == 8


# ---------------------------------------------------------------------------
# Removal lists.
# ---------------------------------------------------------------------------


def test_removal_match_keys_cover_id_styles():
    keys = _removal_match_keys("toucan:uuid-1:u2", "toucan:uuid-1")
    assert {"toucan:uuid-1:u2", "toucan:uuid-1", "uuid-1:u2", "uuid-1"} <= keys
    keys = _removal_match_keys("openswe:traj-9:a4", "openswe:traj-9")
    assert {"openswe:traj-9:a4", "openswe:traj-9", "traj-9:a4", "traj-9"} <= keys
    keys = _removal_match_keys("qa:hotpotqa:row-7", "qa:hotpotqa:row-7")
    assert {"qa:hotpotqa:row-7", "hotpotqa:row-7", "row-7"} <= keys
    keys = _removal_match_keys("sess-1:0", "sess-1")
    assert keys == {"sess-1:0", "sess-1"}


def test_apply_removals_by_session_id_and_qid():
    pool = [
        PoolEntry("sess-a:0", "sess-a", 100),
        PoolEntry("sess-b:0", "sess-b", 100),
        PoolEntry("toucan:uuid-1:u2", "toucan:uuid-1", 100),
        PoolEntry("qa:hotpotqa:row-9", "qa:hotpotqa:row-9", 100),
    ]
    kept, removed = apply_removals(pool, frozenset({"sess-a", "uuid-1", "row-9"}))
    assert removed == 3
    assert [entry.qid for entry in kept] == ["sess-b:0"]
    # Full-qid entries work too (dedup unit_id style).
    kept, removed = apply_removals(pool, frozenset({"sess-b:0"}))
    assert removed == 1 and [entry.qid for entry in kept] == [
        "sess-a:0",
        "toucan:uuid-1:u2",
        "qa:hotpotqa:row-9",
    ]
    # Empty identifier set is a no-op.
    kept, removed = apply_removals(pool, frozenset())
    assert removed == 0 and kept == pool


def test_load_removal_identifiers_both_file_shapes(tmp_path):
    dedup_out = {
        "metadata": {},
        "removal_list": [
            {
                "unit_id": "traces:sess-a:0",
                "dataset": "traces",
                "record_id": "sess-a",
                "unit_index": 0,
                "unit_hash": "x" * 40,
                "match_type": "exact",
                "best_est_jaccard": 1.0,
                "matched_eval_unit": "bfcl:eval-1:0",
            }
        ],
    }
    dict_path = tmp_path / "dedup.json"
    dict_path.write_text(json.dumps(dedup_out), encoding="utf-8")
    list_path = tmp_path / "manual.json"
    list_path.write_text(json.dumps(["uuid-1", "qa:hotpotqa:row-9"]), encoding="utf-8")
    identifiers, per_file = load_removal_identifiers([str(dict_path), str(list_path)])
    assert {"sess-a", "traces:sess-a:0", "uuid-1", "qa:hotpotqa:row-9"} <= identifiers
    assert per_file[str(dict_path)] == 2  # record_id + unit_id
    assert per_file[str(list_path)] == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_removal_list": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="removal_list"):
        load_removal_identifiers([str(bad)])


def test_removals_applied_before_sampling_and_reported():
    pools = {"qa": _pool("qa", 10), "traces": _pool("traces", 10)}
    removal = frozenset({pools["traces"][0].session_id})  # drops its 3 entries
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    results = plan_recipes(
        pools, recipes, budget_estimated_tokens=1400, order_seed=42, removal_identifiers=removal
    )
    plan = results["r"]["plan"]
    assert plan["removals"]["by_family"]["traces"] == 3
    assert plan["removals"]["by_family"]["qa"] == 0
    # No remaining order qid belongs to the removed session.
    removed_session = pools["traces"][0].session_id
    assert not any(qid.startswith(removed_session + ":") for qid in results["r"]["order"])


# ---------------------------------------------------------------------------
# Repeat variants.
# ---------------------------------------------------------------------------


def test_repeat_variant_uniqueness_and_epochs_math():
    pools = {"qa": _pool("qa", 50), "traces": _pool("traces", 100)}
    recipes = [parse_recipe("r=qa:0.2,traces:0.8")]
    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=5000,
        order_seed=42,
        repeat_unique_tokens=2000,
    )
    repeat = results["r_repeat"]
    plan = repeat["plan"]
    # Per-family repeat quotas: qa 400 (4 entries), traces 1600 (16 entries).
    assert plan["families"]["qa"]["examples"] == 4
    assert plan["families"]["traces"]["examples"] == 16
    assert plan["unique_pool_estimated_tokens"] == 2000
    assert plan["recommended_epochs"] == 3  # ceil(5000 / 2000)
    assert plan["presented_estimated_tokens"] == 6000
    # Each qid appears exactly once; repetition is a train-time epoch count.
    assert len(repeat["order"]) == len(set(repeat["order"])) == 20
    assert plan["order_note"]
    # The base variant is still emitted alongside.
    assert results["r"]["plan"]["variant"] == "base"


def test_repeat_variant_short_pool_epochs_use_realized_total():
    pools = {"qa": _pool("qa", 2), "traces": _pool("traces", 3)}  # 500 tokens
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=5000,
        order_seed=42,
        repeat_unique_tokens=2000,
    )
    plan = results["r_repeat"]["plan"]
    assert plan["unique_pool_estimated_tokens"] == 500
    assert plan["recommended_epochs"] == 10  # ceil(5000 / 500)
    assert plan["families"]["qa"]["shortfall_estimated_tokens"] > 0


# ---------------------------------------------------------------------------
# Order files pass the trainer's validation; ETA fields; token cache.
# ---------------------------------------------------------------------------


def _joint_stub(qid):
    return JointExample(
        qid=qid,
        session_id=qid.rsplit(":", 1)[0],
        tool_documents=["tool doc " + qid],
        history_documents=["history doc " + qid],
        current_messages=[{"role": "user", "content": "question " + qid}],
        answer="answer " + qid,
    )


def test_order_files_pass_trainer_validation(tmp_path):
    pools = {"qa": _pool("qa", 12), "traces": _pool("traces", 20), "toucan": _pool("toucan", 8)}
    recipes = [parse_recipe("r=qa:0.2,traces:0.5,toucan:0.3")]
    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=3000,
        order_seed=42,
        repeat_unique_tokens=1000,
    )
    written = write_outputs(results, tmp_path)
    assert len(written) == 4  # base + repeat, order + plan
    all_examples = [_joint_stub(entry.qid) for pool in pools.values() for entry in pool]
    for name, result in results.items():
        order_path = tmp_path / f"{name}.order.json"
        on_disk = json.loads(order_path.read_text(encoding="utf-8"))
        assert isinstance(on_disk, list) and all(isinstance(qid, str) for qid in on_disk)
        ordered = _apply_example_order_file(all_examples, str(order_path))
        assert [example.qid for example in ordered] == result["order"]
        plan = json.loads((tmp_path / f"{name}.plan.json").read_text(encoding="utf-8"))
        assert plan["order_examples"] == len(on_disk)


def test_eta_fields_recorded_only_with_small_arm_hours():
    pools = {"traces": _pool("traces", 10)}
    recipes = [parse_recipe("r=traces:1.0")]
    with_hours = plan_recipes(
        pools, recipes, budget_estimated_tokens=500, order_seed=42, small_arm_hours=12.5
    )
    assert with_hours["r"]["plan"]["eta"] == {
        "small_arm_hours": 12.5,
        "tokens_per_unit": 32_000_000,
    }
    without = plan_recipes(pools, recipes, budget_estimated_tokens=500, order_seed=42)
    assert without["r"]["plan"]["eta"] is None
    line = bjmp._eta_line("r", 64_000_000, 10.0)
    assert line.startswith("r: 预计完成 = ")
    assert "presented≈64.0M" in line and "32M ≈ 10.0h" in line


def test_token_cache_roundtrip_and_stamp_invalidation(tmp_path):
    cache_path = tmp_path / "tokencache_qa.jsonl"
    cache = {("qa:hotpotqa:a", "stamp1"): 123, ("qa:hotpotqa:b", "stamp1"): 45}
    bjmp._write_token_cache(cache_path, cache)
    assert bjmp._load_token_cache(cache_path) == cache
    # A different stamp (tokenizer/seed change) misses old entries.
    loaded = bjmp._load_token_cache(cache_path)
    assert loaded.get(("qa:hotpotqa:a", "stamp2")) is None
    # Corrupt lines are skipped, not fatal.
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert bjmp._load_token_cache(cache_path)[("qa:hotpotqa:a", "stamp1")] == 123
