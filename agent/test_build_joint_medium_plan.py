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
c. ``interleave_families``: token-deficit weighted scheduling (declared
   order breaks ties; equal sizes degenerate to round-robin), sliding-window
   token shares track the global recipe shares (anti front-loading);
d. removal lists: bare-list and dedup-dict shapes, session-id AND qid entries,
   family-prefix-stripped matching (uuid / trajectory_id / qa row id);
e. repeat variants: qid uniqueness, ``recommended_epochs`` math;
   ``--epochs_override`` audit recording on base and repeat variants;
f. order files pass ``_apply_example_order_file`` (unique, all loadable);
g. determinism: identical plans across runs; token-cache roundtrip + stamp
   invalidation; shortfall reporting on undersized pools.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest agent/test_build_joint_medium_plan.py -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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


def test_interleave_families_equal_sizes_rotate_in_declared_order():
    # Equal sizes + equal shares degenerate to the old round-robin: the
    # deficit is always tied after each pair of steps, declared order breaks
    # ties, and the longer family's leftover tail trails in order.
    order = interleave_families(
        [
            ("qa", [("q0", 100), ("q1", 100)]),
            ("traces", [("t0", 100), ("t1", 100), ("t2", 100)]),
        ],
        (("qa", 0.5), ("traces", 0.5)),
    )
    assert order == ["q0", "t0", "q1", "t1", "t2"]


def test_interleave_families_spaces_out_large_examples():
    # 10x size spread at equal shares: after each 100-token traces example the
    # qa side must emit ~100 tokens (10 examples) before the next traces one —
    # a 1:1 round-robin would instead drain qa by position 50 and leave the
    # tail 100% traces (subset front-loading).
    order = interleave_families(
        [
            ("qa", [(f"q{i}", 10) for i in range(50)]),
            ("traces", [(f"t{i}", 100) for i in range(5)]),
        ],
        (("qa", 0.5), ("traces", 0.5)),
    )
    assert order[0] == "q0"  # first-step deficit tie -> declared order leads
    traces_positions = [index for index, qid in enumerate(order) if qid.startswith("t")]
    assert traces_positions == [1, 12, 23, 34, 45]
    assert len(order) == 55 and len(set(order)) == 55


def test_interleave_token_weighted_windows_track_global_shares():
    # The anti-front-loading acceptance test: three families at shares
    # 0.5/0.3/0.2 with a 15x example-size spread (55/340/23 tokens, sizes
    # chosen so quotas are filled by the crossing rule, not exact division).
    # Slicing the frozen order into 10 equal-token windows, each window's
    # per-family token share must track the realized global share to ±5pp.
    sizes = {"traces": 55, "toucan": 340, "qa": 23}
    pools = {
        family: [
            PoolEntry(qid=f"{family}:id-{i}", session_id=f"{family}:id-{i}", estimated_tokens=sizes[family])
            for i in range(count)
        ]
        for family, count in (("traces", 2000), ("toucan", 200), ("qa", 2000))
    }
    recipes = [parse_recipe("r=traces:0.5,toucan:0.3,qa:0.2")]
    results = plan_recipes(pools, recipes, budget_estimated_tokens=100000, order_seed=42)
    plan = results["r"]["plan"]
    order = results["r"]["order"]
    realized = {family: plan["families"][family]["realized_share"] for family in sizes}
    tokens_of = {entry.qid: entry.estimated_tokens for pool in pools.values() for entry in pool}
    family_of = {entry.qid: family for family, pool in pools.items() for entry in pool}

    total = sum(tokens_of[qid] for qid in order)
    target = total / 10
    windows = []
    current, current_tokens = {}, 0
    for qid in order:
        tokens = tokens_of[qid]
        if current_tokens > 0 and current_tokens + tokens > target and len(windows) < 9:
            windows.append((current, current_tokens))
            current, current_tokens = {}, 0
        family = family_of[qid]
        current[family] = current.get(family, 0) + tokens
        current_tokens += tokens
    windows.append((current, current_tokens))
    assert len(windows) == 10
    for index, (window, window_tokens) in enumerate(windows):
        for family, share in realized.items():
            local = window.get(family, 0) / window_tokens
            # Edge windows carry the scheduler's startup transient and the
            # token-count rounding remainder, so they are allowed 10pp; the
            # pre-registered anti-front-loading invariant is the ±5pp band on
            # the interior windows.
            tolerance = 0.05 if 0 < index < len(windows) - 1 else 0.10
            assert abs(local - share) <= tolerance, (
                f"window {index}: family {family} local share {local:.3f} vs global {share:.3f}"
            )


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


def test_budget_shrink_keeps_realized_shares_when_pool_undersized(caplog):
    # P1-3: an undersized family pool shrinks ALL family quotas together, so
    # the realized shares stay on recipe instead of skewing (the pre-fix
    # traces-v1 shortfall pushed d_single's qa realized share to ~31%).
    pools = {"qa": _pool("qa", 3), "traces": _pool("traces", 5)}  # 300 + 500 tokens
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    with caplog.at_level(logging.WARNING):
        results = plan_recipes(pools, recipes, budget_estimated_tokens=10000, order_seed=42)
    plan = results["r"]["plan"]
    assert plan["budget_shrink_factor"] == pytest.approx(0.06)  # 300 / 5000 binds
    assert plan["effective_budget_estimated_tokens"] == 600
    assert plan["families"]["qa"]["examples"] == 3
    assert plan["families"]["traces"]["examples"] == 3
    assert plan["families"]["qa"]["shortfall_estimated_tokens"] == 0
    assert abs(plan["families"]["qa"]["realized_share"] - 0.5) < 0.02
    assert abs(plan["families"]["traces"]["realized_share"] - 0.5) < 0.02
    assert any("BUDGET SHRINK" in record.message for record in caplog.records)


def test_budget_shrink_empty_pool_is_a_hard_error():
    pools = {"qa": _pool("qa", 3), "traces": []}
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    with pytest.raises(ValueError, match="empty"):
        plan_recipes(pools, recipes, budget_estimated_tokens=1000)


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
    # P1-5/6: longmagpie shard-local qids match extraction-unit ids exactly
    # (the extractor writes the full qid as the unit _id).
    keys = _removal_match_keys("qa:longmagpie:shard-0:17", "qa:longmagpie:shard-0:17")
    assert "qa:longmagpie:shard-0:17" in keys
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
    # P1-3: the repeat pool shrinks to the binding family (qa 200 of its 1000
    # quota -> factor 0.2), then epochs derive from the REALIZED unique total.
    assert plan["budget_shrink_factor"] == pytest.approx(0.2)
    assert plan["effective_repeat_unique_tokens"] == 400
    assert plan["unique_pool_estimated_tokens"] == 400
    assert plan["recommended_epochs"] == 13  # ceil(5000 / 400)
    assert plan["families"]["qa"]["shortfall_estimated_tokens"] == 0


# ---------------------------------------------------------------------------
# --epochs_override (pure audit record on both variants).
# ---------------------------------------------------------------------------


def test_parse_epochs_override_validation():
    assert bjmp.parse_epochs_override("d_multi=3") == ("d_multi", 3)
    for bad in ("d_multi=0", "d_multi=-1", "d_multi=1.5", "d_multi", "=2", "d_multi=x"):
        with pytest.raises(ValueError, match="epochs_override"):
            bjmp.parse_epochs_override(bad)
    with pytest.raises(ValueError, match="unknown recipe"):
        bjmp._parse_epochs_overrides(["nope=2"], {"d_single", "d_multi"})
    assert bjmp._parse_epochs_overrides(None, {"d_single"}) == {}
    assert bjmp._parse_epochs_overrides(["d_single=2", "d_multi=3"], {"d_single", "d_multi"}) == {
        "d_single": 2,
        "d_multi": 3,
    }


def test_epochs_override_scales_presented_tokens_on_both_variants():
    pools = {"qa": _pool("qa", 50), "traces": _pool("traces", 100)}
    recipes = [parse_recipe("r=qa:0.2,traces:0.8")]
    base_kwargs = dict(
        budget_estimated_tokens=5000,
        order_seed=42,
        repeat_unique_tokens=2000,
    )
    plain = plan_recipes(pools, recipes, **base_kwargs)
    # Without an override: base presents exactly the realized total; the key
    # is always present (None) for a stable plan schema.
    assert plain["r"]["plan"]["epochs_override"] is None
    assert plain["r"]["plan"]["presented_estimated_tokens"] == 5000
    assert plain["r_repeat"]["plan"]["epochs_override"] is None
    assert plain["r_repeat"]["plan"]["presented_estimated_tokens"] == 2000 * 3

    overridden = plan_recipes(pools, recipes, epochs_overrides={"r": 2}, **base_kwargs)
    base = overridden["r"]["plan"]
    assert base["epochs_override"] == 2
    assert base["presented_estimated_tokens"] == 5000 * 2
    repeat = overridden["r_repeat"]["plan"]
    # Repeat semantics preserved: recommended_epochs is still the computed
    # value; presented uses the override when given.
    assert repeat["recommended_epochs"] == 3
    assert repeat["epochs_override"] == 2
    assert repeat["presented_estimated_tokens"] == 2000 * 2
    # The override never touches the order itself.
    assert overridden["r"]["order"] == plain["r"]["order"]
    assert overridden["r_repeat"]["order"] == plain["r_repeat"]["order"]


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
        "small_arm_hours_alternate": 12.5,  # falls back to the joint calibration
        "tokens_per_unit": 32_000_000,
        "unit": "hours per 32M ESTIMATED source tokens (estimator 口径; presented is ~0.392x)",
    }
    dual = plan_recipes(
        pools, recipes, budget_estimated_tokens=500, order_seed=42,
        small_arm_hours=12.5, small_arm_hours_alternate=22.7,
    )
    assert dual["r"]["plan"]["eta"]["small_arm_hours_alternate"] == 22.7
    without = plan_recipes(pools, recipes, budget_estimated_tokens=500, order_seed=42)
    assert without["r"]["plan"]["eta"] is None
    line = bjmp._eta_line("r", 64_000_000, 10.0)
    assert line.startswith("r: 预计完成 = ")
    assert "estimated≈64.0M" in line and "32M estimated ≈ 10.0h" in line
    assert "presented" not in line  # estimator 口径 only, no mixed wording


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


# ---------------------------------------------------------------------------
# P0-2: stratified pool scanning (per-subset caps + seeded file order).
# ---------------------------------------------------------------------------


def _write_tiny_qa_tree(tmp_path):
    """Tiny qa corpus: hotpotqa jsonl + 2wiki parquet + longmagpie 2 shards.

    Word counts (whitespace estimator): hotpotqa row = 2 docs x 25 words = 50;
    2wiki row = 2 docs x ~14 words; longmagpie row = 1 doc x 40 words.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    hotpotqa_path = tmp_path / "hotpotqa.jsonl"
    with hotpotqa_path.open("w", encoding="utf-8") as handle:
        for index in range(30):
            handle.write(json.dumps({
                "_id": f"hp-{index}",
                "question": f"question {index}?",
                "answer": f"answer {index}",
                "documents": [
                    f"Document one of row {index} " + " ".join(f"a{index}-{j}" for j in range(20)),
                    f"Document two of row {index} " + " ".join(f"b{index}-{j}" for j in range(20)),
                ],
            }, ensure_ascii=False) + "\n")
    wiki2_dir = tmp_path / "2wiki"
    wiki2_dir.mkdir()
    # Real on-disk 2wiki: context is a JSON-ENCODED STRING column that parses
    # to [[title, [sentence, ...]], ...] (verified on the server parquet).
    pq.write_table(
        pa.table({
            "_id": [f"w2-{i}" for i in range(10)],
            "question": [f"question {i}?" for i in range(10)],
            "answer": [f"answer {i}" for i in range(10)],
            "context": [
                json.dumps([["TitleA", ["one two three four five six"]], ["TitleB", ["seven eight nine ten eleven twelve"]]])
                for _ in range(10)
            ],
        }),
        wiki2_dir / "train.parquet",
    )
    lm_dir = tmp_path / "longmagpie" / "data"
    lm_dir.mkdir(parents=True)
    for shard in ("shard-a", "shard-b"):
        pq.write_table(
            pa.table({
                "messages": [
                    [
                        {"role": "user", "content": " ".join(f"c{i}-{j}" for j in range(38)) + ".What is it?"},
                        {"role": "assistant", "content": "It is X."},
                    ]
                    for i in range(5)
                ]
            }),
            lm_dir / f"{shard}.parquet",
        )
    return hotpotqa_path, wiki2_dir, lm_dir.parent


def _scan_args(tmp_path, **overrides):
    hotpotqa_path, wiki2_dir, lm_root = _write_tiny_qa_tree(tmp_path)
    args = argparse.Namespace(
        traces_path=None,
        toucan_path=None,
        openswe_path=None,
        qa_hotpotqa_path=str(hotpotqa_path),
        qa_2wiki_path=str(wiki2_dir),
        qa_longmagpie_path=str(lm_root),
        split_manifest_file=None,
        split_manifest_name="subset_disjoint",
        split_seed=42,
        order_seed=42,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_stratified_scan_tail_subsets_represented_under_cap(tmp_path):
    args = _scan_args(tmp_path)
    cache: dict = {}
    # Cap below hotpotqa's own size: the pre-P0-2 prefix scan would have
    # filled the whole pool with hotpotqa and left 2wiki/longmagpie at zero.
    pool, report = bjmp.scan_family_pool(
        "qa", args, bjmp.WhitespaceTokenizer(), "stamp", cache, token_cap=600
    )
    by_subset = {}
    for entry in pool:
        by_subset.setdefault(entry.subset, 0)
        by_subset[entry.subset] += 1
    assert set(by_subset) == {"hotpotqa", "2wiki", "longmagpie"}
    assert set(report["subsets"]) == {"hotpotqa", "2wiki", "longmagpie"}
    # Sequential water-filling: hotpotqa cap 200 (equal split) -> 4 rows of 50
    # tokens; 2wiki cap 200 but exhausts at 140 (10 rows of 14); the 60-token
    # leftover lifts longmagpie's cap to 260 -> 7 rows of 40.
    assert report["subsets"]["hotpotqa"]["cap_estimated_tokens"] == 200
    assert report["subsets"]["hotpotqa"]["examples"] == 4
    assert report["subsets"]["2wiki"]["cap_estimated_tokens"] == 200
    assert report["subsets"]["2wiki"]["examples"] == 10
    assert report["subsets"]["2wiki"]["exhausted"] is True
    assert report["subsets"]["longmagpie"]["cap_estimated_tokens"] == 260
    assert report["subsets"]["longmagpie"]["examples"] == 7
    assert report["subsets"]["hotpotqa"]["exhausted"] is False
    # Plan-level attribution: take the whole pool (budget == pool total) and
    # every subset shows up with its counts and a positive share.
    results = plan_recipes(
        {"qa": pool}, [parse_recipe("r=qa:1.0")], budget_estimated_tokens=620, order_seed=42
    )
    sampled = results["r"]["plan"]["families"]["qa"]["subsets"]
    assert {name: sub["examples"] for name, sub in sampled.items()} == {
        "hotpotqa": 4,
        "2wiki": 10,
        "longmagpie": 7,
    }
    assert all(sub["share_within_family"] > 0 for sub in sampled.values())


def test_stratified_scan_subset_weights_and_validation(tmp_path):
    args = _scan_args(tmp_path)
    # qa:hotpotqa=3 (weights 3:1:1): hotpotqa cap 360 -> 8 rows (400 used);
    # 2wiki cap 100 -> 8 rows (112 used); longmagpie cap 88 -> 3 rows.
    pool, report = bjmp.scan_family_pool(
        "qa", args, bjmp.WhitespaceTokenizer(), "stamp", {},
        token_cap=600, subset_weights={"qa:hotpotqa": 3.0},
    )
    caps = {name: sub["cap_estimated_tokens"] for name, sub in report["subsets"].items()}
    assert caps == {"hotpotqa": 360, "2wiki": 100, "longmagpie": 88}
    assert report["subsets"]["hotpotqa"]["examples"] == 8
    with pytest.raises(ValueError, match="> 0"):
        bjmp.scan_family_pool(
            "qa", args, bjmp.WhitespaceTokenizer(), "stamp", {},
            token_cap=600, subset_weights={"qa:hotpotqa": 0.0},
        )
    for bad in ("hotpotqa=2", "qa:=2", "mystery:x=2", "qa:x=-1"):
        with pytest.raises(ValueError, match="subset_weights"):
            bjmp._parse_subset_weights([bad])
    assert bjmp._parse_subset_weights(["qa:hotpotqa=2", "openswe:cfg=0.5"]) == {
        "qa:hotpotqa": 2.0,
        "openswe:cfg": 0.5,
    }


def test_stratified_scan_subset_shortfall_redistributes_to_siblings(tmp_path):
    args = _scan_args(tmp_path)
    # Shrink longmagpie (the LAST subset) to a single 2-row shard: its unused
    # budget has no later siblings to flow to, so the family pool underfills
    # the cap (the family-level shortfall P1-3 renormalizes downstream).
    lm_dir = tmp_path / "longmagpie" / "data"
    import pyarrow as pa
    import pyarrow.parquet as pq

    (lm_dir / "shard-b.parquet").unlink()
    pq.write_table(
        pa.table({"messages": [[
            {"role": "user", "content": " ".join(["x"] * 38) + ".What is it?"},
            {"role": "assistant", "content": "It is X."},
        ]] * 2}),
        lm_dir / "shard-a.parquet",
    )
    pool, report = bjmp.scan_family_pool(
        "qa", args, bjmp.WhitespaceTokenizer(), "stamp", {}, token_cap=600
    )
    assert report["subsets"]["longmagpie"]["exhausted"] is True
    assert report["subsets"]["longmagpie"]["estimated_tokens"] == 76  # 2 rows x 38 words
    total = sum(sub["estimated_tokens"] for sub in report["subsets"].values())
    assert total == 200 + 140 + 76 < 600


def test_family_subsources_openswe_configs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for config in ("cfg_a", "cfg_b"):
        config_dir = tmp_path / "openswe" / "data" / config
        config_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "trajectory_id": [f"{config}-0"],
                "instance_id": [f"{config}-inst-0"],
                "resolved": [1],
                "trajectory": [[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "first", "tool_calls": [
                        {"type": "function", "function": {"name": "run", "arguments": "{}"}}
                    ]},
                    {"role": "tool", "content": "obs"},
                    {"role": "assistant", "content": "second", "tool_calls": [
                        {"type": "function", "function": {"name": "run", "arguments": "{}"}}
                    ]},
                ]],
                "tools": ['{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{}}}}'],
            }),
            config_dir / "shard-0.parquet",
        )
    args = _scan_args(tmp_path, openswe_path=str(tmp_path / "openswe"))
    subs = bjmp._family_subsources("openswe", args)
    assert [name for name, _ in subs] == ["cfg_a", "cfg_b"]
    # Each subset source streams only its own config's examples.
    qids = {name: [e.qid for e in source] for name, source in subs}
    assert qids["cfg_a"] == ["openswe:cfg_a-0:a4"]
    assert qids["cfg_b"] == ["openswe:cfg_b-0:a4"]
    # Per-config caps keep both configs in the pool under a tight family cap.
    pool, report = bjmp.scan_family_pool(
        "openswe", args, bjmp.WhitespaceTokenizer(), "stamp", {}, token_cap=60
    )
    assert set(report["subsets"]) == {"cfg_a", "cfg_b"}
    assert {entry.subset for entry in pool} == {"cfg_a", "cfg_b"}


# ---------------------------------------------------------------------------
# P1-7: alternate-pass asymmetry is explicit in the plan.
# P1-8: arm launch table + epochs x budget parity guard.
# ---------------------------------------------------------------------------


def test_alternate_pass_counts_record_qa_asymmetry():
    pools = {"qa": _pool("qa", 20), "traces": _pool("traces", 20)}
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    results = plan_recipes(
        pools, recipes, budget_estimated_tokens=1000, order_seed=42, repeat_unique_tokens=400
    )
    for key in ("r", "r_repeat"):
        counts = results[key]["plan"]["alternate_pass_counts"]
        assert counts["qa"] == {"tool_only": 0, "history_only": 1}
        assert counts["traces"] == {"tool_only": 1, "history_only": 1}
        assert "qa:doc_num<2" in results[key]["plan"]["alternate_pass_counts_note"]


def _medium_pools(tokens_per_family=20000):
    return {
        family: _pool(family, tokens_per_family // 100)
        for family in ("qa", "traces", "toucan", "openswe")
    }


def test_arm_launch_table_parity_ok_and_content():
    pools = _medium_pools()
    recipes = [
        parse_recipe("d_single=qa:0.2,traces:0.8"),
        parse_recipe("d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05"),
    ]
    # M=2000: repeat quotas (400/1000/500/100) are exact multiples of the
    # 100-token entries, so U=2000 and recommended epochs = 5 land the repeat
    # arm's presented total exactly on the base arms' 10000 x 0.392 = 3920.
    results = plan_recipes(
        pools, recipes, budget_estimated_tokens=10000, order_seed=42, repeat_unique_tokens=2000
    )
    table = results["d_single"]["plan"]["arm_launch_table"]
    assert table["parity_ok"] is True
    rows = {row["arm"]: row for row in table["arms"]}
    assert list(rows) == [
        "med_dsingle_alt",
        "med_dsingle_joint",
        "med_dmulti_alt",
        "med_dmulti_repeat_alt",
    ]
    assert [row["suggested_card"] for row in table["arms"]] == [2, 3, 4, 5]
    # The two d_single arms share one order file (the Gate-3 re-check pair).
    assert rows["med_dsingle_alt"]["order_file"] == "d_single.order.json"
    assert rows["med_dsingle_joint"]["order_file"] == "d_single.order.json"
    assert rows["med_dsingle_alt"]["doc_mode"] == "alternate"
    assert rows["med_dsingle_joint"]["doc_mode"] == "joint"
    # U=10000/e=1 for the base arms; U=2000/e=5 for the repeat variant.
    for arm in ("med_dsingle_alt", "med_dsingle_joint", "med_dmulti_alt"):
        assert rows[arm]["unique_est_tokens"] == 10000
        assert rows[arm]["effective_epochs"] == 1
    assert rows["med_dmulti_repeat_alt"]["unique_est_tokens"] == 2000
    assert rows["med_dmulti_repeat_alt"]["effective_epochs"] == 5
    for row in rows.values():
        assert row["presented_est_tokens"] == pytest.approx(3920.0)
        assert row["max_source_tokens"] == row["unique_est_tokens"]
    # Level-up suggestions with 2% headroom over the (equal) max: base arms 2,
    # repeat ceil(3920 x 1.02 / (2000 x 0.392)) = 6.
    assert rows["med_dsingle_alt"]["suggested_num_train_epochs"] == 2
    assert rows["med_dmulti_repeat_alt"]["suggested_num_train_epochs"] == 6
    assert table["skipped_arms"] == []
    # The same table object is embedded in every plan json.
    assert results["d_multi_repeat"]["plan"]["arm_launch_table"] is table
    lines = bjmp._arm_table_lines(table)
    assert any("ARM LAUNCH TABLE" in line for line in lines)
    assert any("med_dmulti_repeat_alt" in line for line in lines)


def test_arm_launch_table_guard_fires_on_imbalance():
    pools = _medium_pools()
    recipes = [
        parse_recipe("d_single=qa:0.2,traces:0.8"),
        parse_recipe("d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05"),
    ]
    # epochs_override d_multi=2 doubles that arm's presented total (7840 vs
    # the 3920 floor) -> the guard must fail loudly with suggestions.
    with pytest.raises(RuntimeError, match="ARM PARITY GUARD FAILED"):
        plan_recipes(
            pools, recipes, budget_estimated_tokens=10000, order_seed=42,
            repeat_unique_tokens=2000, epochs_overrides={"d_multi": 2},
        )
    # A repeat pool whose recommended epochs overshoot parity also fails:
    # M=3200 -> U=3300 (crossing granularity), recommended 4 -> 5174 presented.
    with pytest.raises(RuntimeError, match="ARM PARITY GUARD FAILED"):
        plan_recipes(
            pools, recipes, budget_estimated_tokens=10000, order_seed=42,
            repeat_unique_tokens=3200,
        )


def test_arm_launch_table_partial_runs_and_toy_recipes():
    # Only d_single planned: 2-arm table, the d_multi arms listed as skipped.
    pools = _medium_pools()
    recipes = [parse_recipe("d_single=qa:0.2,traces:0.8")]
    results = plan_recipes(pools, recipes, budget_estimated_tokens=10000, order_seed=42)
    table = results["d_single"]["plan"]["arm_launch_table"]
    assert [row["arm"] for row in table["arms"]] == ["med_dsingle_alt", "med_dsingle_joint"]
    assert table["parity_ok"] is True
    assert table["skipped_arms"] == ["med_dmulti_alt", "med_dmulti_repeat_alt"]
    # Toy recipe names (no fixed arm) -> no table, no guard.
    toy = plan_recipes({"qa": _pool("qa", 10)}, [parse_recipe("r=qa:1.0")], budget_estimated_tokens=500)
    assert "arm_launch_table" not in toy["r"]["plan"]


def test_arm_launch_table_truncation_parity_target():
    """presented_target_est: per-epoch take = ceil(P*/epochs) with
    MAX_SOURCE_TOKENS = take -> presented = take x epochs across arms; the
    guard requires take <= U per arm (no mid-epoch stop exists)."""
    pools = _medium_pools()
    recipes = [
        parse_recipe("d_single=qa:0.2,traces:0.8"),
        parse_recipe("d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05"),
    ]
    # target 9000 with default epochs (1/1/1, repeat recommended 5):
    # takes 9000/9000/9000/1800 all fit (U 10000/10000/10000/2000).
    results = plan_recipes(
        pools, recipes, budget_estimated_tokens=10000, order_seed=42,
        repeat_unique_tokens=2000, presented_target_est=9000,
    )
    table = results["d_single"]["plan"]["arm_launch_table"]
    assert table["parity_ok"] is True
    assert table["parity_mode"] == "truncation"
    assert table["presented_target_est_tokens"] == 9000
    rows = {row["arm"]: row for row in table["arms"]}
    for row in rows.values():
        assert row["presented_est_tokens"] == 9000.0
    assert rows["med_dsingle_alt"]["max_source_tokens"] == 9000
    assert rows["med_dmulti_repeat_alt"]["max_source_tokens"] == 1800
    # suggested epochs = smallest count whose take fits the pool.
    assert rows["med_dsingle_alt"]["suggested_num_train_epochs"] == 1
    assert rows["med_dmulti_repeat_alt"]["suggested_num_train_epochs"] == 5
    assert table["capacity_tight_arms"] == []


def test_arm_launch_table_truncation_capacity_guard_fires_and_rescues():
    pools = _medium_pools()
    recipes = [
        parse_recipe("d_single=qa:0.2,traces:0.8"),
        parse_recipe("d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05"),
    ]
    # target 12000 with defaults: base arms need take 12000 > U=10000 -> fail.
    with pytest.raises(RuntimeError, match="ARM CAPACITY GUARD FAILED"):
        plan_recipes(
            pools, recipes, budget_estimated_tokens=10000, order_seed=42,
            repeat_unique_tokens=2000, presented_target_est=12000,
        )
    # Base-key epochs alone cannot rescue the repeat variant (shared key
    # covers it: 2000-pool x 2 epochs -> take 6000 > 2000, still fails).
    with pytest.raises(RuntimeError, match="ARM CAPACITY GUARD FAILED"):
        plan_recipes(
            pools, recipes, budget_estimated_tokens=10000, order_seed=42,
            repeat_unique_tokens=2000, presented_target_est=12000,
            epochs_overrides={"d_single": 2, "d_multi": 2},
        )
    # Variant-specific key rescues: repeat at 6 epochs -> take 2000 <= U.
    results = plan_recipes(
        pools, recipes, budget_estimated_tokens=10000, order_seed=42,
        repeat_unique_tokens=2000, presented_target_est=12000,
        epochs_overrides={"d_single": 2, "d_multi": 2, "d_multi_repeat": 6},
    )
    table = results["d_single"]["plan"]["arm_launch_table"]
    assert table["parity_ok"] is True
    rows = {row["arm"]: row for row in table["arms"]}
    assert rows["med_dmulti_repeat_alt"]["effective_epochs"] == 6
    assert rows["med_dmulti_repeat_alt"]["max_source_tokens"] == 2000
    assert rows["med_dmulti_repeat_alt"]["presented_est_tokens"] == 12000.0
    assert rows["med_dsingle_alt"]["max_source_tokens"] == 6000


# ---------------------------------------------------------------------------
# P2: provenance block, order hash, removal-before-cap, epochs_override scope.
# ---------------------------------------------------------------------------


def test_plan_provenance_order_hash_and_override_scope():
    pools = {"qa": _pool("qa", 20), "traces": _pool("traces", 20)}
    recipes = [parse_recipe("r=qa:0.5,traces:0.5")]
    provenance = {
        "tokenizer": "whitespace-fake",
        "estimate_stamp": "stamp-x",
        "split_seed": 42,
        "oversample_factor": 1.25,
        "family_scan_caps_estimated_tokens": {"qa": 625, "traces": 625},
        "subset_weights": {},
        "repeat_unique_tokens": 400,
    }
    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=1000,
        order_seed=42,
        repeat_unique_tokens=400,
        epochs_overrides={"r": 2},
        provenance=provenance,
    )
    for key in ("r", "r_repeat"):
        plan = results[key]["plan"]
        assert plan["provenance"] == provenance
        assert plan["epochs_override_scope"] == "base_and_repeat_variants"
        assert plan["order_sha1"] == hashlib.sha1(
            json.dumps(results[key]["order"]).encode("utf-8")
        ).hexdigest()
    # The base and repeat orders differ, so their hashes must too.
    assert results["r"]["plan"]["order_sha1"] != results["r_repeat"]["plan"]["order_sha1"]
    # Provenance defaults to None for direct/test callers.
    plain = plan_recipes(pools, recipes, budget_estimated_tokens=1000)
    assert plain["r"]["plan"]["provenance"] is None


def test_removal_during_scan_preserves_cap_headroom(tmp_path):
    # P2: removal filtering happens DURING the scan, before cap accounting —
    # removed rows must not eat the oversample headroom.
    args = _scan_args(tmp_path)
    removal = frozenset(f"w2-{i}" for i in range(10))  # all bare 2wiki row ids
    pool, report = bjmp.scan_family_pool(
        "qa", args, bjmp.WhitespaceTokenizer(), "stamp", {},
        token_cap=600, removal_identifiers=removal,
    )
    assert report["subsets"]["2wiki"]["removed"] == 10
    assert report["subsets"]["2wiki"]["examples"] == 0
    assert report["removed_total"] == 10
    assert not any(entry.qid.startswith("qa:2wiki") for entry in pool)
    # Water-filling: hotpotqa used its 200; the empty 2wiki stratum leaves the
    # full remainder to longmagpie (cap 400 -> all 10 rows, 380 tokens).
    assert report["subsets"]["longmagpie"]["cap_estimated_tokens"] == 400
    assert report["subsets"]["longmagpie"]["examples"] == 10
    assert report["subsets"]["hotpotqa"]["examples"] == 4


def test_scan_stage_removals_reported_in_plan(tmp_path):
    args = _scan_args(tmp_path)
    removal = frozenset({"hp-1"})  # one hotpotqa row, by bare _id
    pool, report = bjmp.scan_family_pool(
        "qa", args, bjmp.WhitespaceTokenizer(), "stamp", {},
        token_cap=600, removal_identifiers=removal,
    )
    assert report["subsets"]["hotpotqa"]["removed"] == 1
    results = plan_recipes(
        {"qa": pool},
        [parse_recipe("r=qa:1.0")],
        budget_estimated_tokens=400,
        order_seed=42,
        removal_identifiers=removal,
        scan_removals={"qa": 1},
    )
    plan = results["r"]["plan"]
    assert plan["removals"]["by_family"] == {"qa": 1}
    assert plan["removals"]["applied_at"] == "scan"
    # The scan already filtered, so the plan-stage pass is a zero-hit safety
    # net (no residual field recorded).
    assert "plan_stage_residual" not in plan["removals"]
    assert not any(entry.qid == "qa:hotpotqa:hp-1" for entry in pool)
