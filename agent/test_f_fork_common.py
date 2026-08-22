# -*- coding: utf-8 -*-
"""CPU-only, torch-free unit tests for agent/f_fork_common.py.

The module under test imports nothing but stdlib, which is the whole point:
the F analyzer has to run on a laptop with no torch and no model weights.  A
top-level torch import creeping into f_fork_common would break that, and the
import smoke test below catches it.

Coverage:
a. ``kv_bytes_per_token`` against the Qwen3-4B bf16 hand calculation;
b. eligibility boundaries E2/E3/E4 (chunk count, 63/64/1024/1025 tokens, no
   tool call in the target) and the meta it reports;
c. ``deterministic_check_pass`` / ``action_key`` on well-formed and broken
   calls;
d. the F3 tie table: all four (A pass, B pass) combinations under R1 and R1b,
   plus the F1 rollout tie rule;
e. ``f4_coin`` determinism and balance on a synthetic pool;
f. ``derive_arms``: union = OR for binary and max for continuous, selection
   labels, arms omitted when their slots are absent, and the
   Δ_oracle = F5 − max(single arm) arithmetic on hand-counted rows;
g. ``four_cell`` / ``pairwise_disagreement`` / ``both_match_gold`` counts;
h. ``cluster_bootstrap_ci`` seed reproducibility and the single-cluster
   degenerate case;
i. ``load_done_keys``: skipped rows are NOT done and get retried on resume.

Run from the repo root:
  python -m pytest agent/test_f_fork_common.py -v
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

import f_fork_common as FC  # noqa: E402
from f_fork_common import (  # noqa: E402
    BRANCH_COMPRESS_NOW,
    BRANCH_DEFER,
    action_key,
    both_match_gold,
    cluster_bootstrap_ci,
    derive_arms,
    deterministic_check_pass,
    f3_select,
    f4_coin,
    fork_eligibility,
    four_cell,
    index_rows_by_qid,
    kv_bytes_per_token,
    load_done_keys,
    pairwise_disagreement,
    select_rollout_by_checks,
    slot_name,
)


# ---------------------------------------------------------------------------
# Row helpers.
# ---------------------------------------------------------------------------


def _row(
    qid,
    arm_pass,
    branch,
    rollout=0,
    *,
    check=True,
    pred="A",
    gold="A",
    tool=True,
    f1=1.0,
    session="s0",
    skipped=False,
    **extra,
):
    row = {
        "qid": qid,
        "session_id": session,
        "arm_pass": arm_pass,
        "branch": branch,
        "rollout_index": rollout,
        "deterministic_check_pass": check,
        "pred_action_key": pred,
        "gold_action_key": gold,
        "tool_name_match": tool,
        "action_key_match": pred is not None and pred == gold,
        "argument_value_f1": f1,
    }
    if skipped:
        row["skipped"] = True
    row.update(extra)
    return row


def _greedy_pair(qid, *, a_kwargs=None, b_kwargs=None, session="s0"):
    return [
        _row(qid, "greedy_core", BRANCH_COMPRESS_NOW, 0, session=session, **(a_kwargs or {})),
        _row(qid, "greedy_core", BRANCH_DEFER, 0, session=session, **(b_kwargs or {})),
    ]


# ---------------------------------------------------------------------------
# a. import smoke + kv arithmetic
# ---------------------------------------------------------------------------


def test_module_is_torch_free_and_respects_the_naming_discipline():
    # The module imported at collection time on a machine with no torch, which
    # is the real assertion; the source scan documents WHY it must stay that way.
    source = (Path(FC.__file__)).read_text(encoding="utf-8")
    assert "import torch" not in source
    # Banned as a name for our own design: the mechanical check is only ever
    # called deterministic_check_*.
    assert "verif" not in source.lower()
    assert "deterministic_check_pass" in source


def test_kv_bytes_per_token_qwen3_4b_bf16():
    # 36 layers x (K and V) x 8 KV heads x head_dim 128 x 2 bytes = 144 KiB.
    assert kv_bytes_per_token(36, 8, 128, 2) == 147456
    assert kv_bytes_per_token(36, 8, 128, 2) == 144 * 1024


def test_kv_bytes_per_token_rejects_nonpositive():
    with pytest.raises(ValueError, match="num_kv_heads"):
        kv_bytes_per_token(36, 0, 128, 2)


# ---------------------------------------------------------------------------
# b. eligibility (E2/E3/E4)
# ---------------------------------------------------------------------------


def test_eligibility_needs_two_history_chunks():
    ok, reason, meta = fork_eligibility([500], tool_chunk_count=4, target_has_tool_call=True)
    assert (ok, reason) == (False, "history_chunks<2")
    assert meta["history_chunk_count"] == 1
    assert meta["fork_chunk_index"] == 4  # 4 tool chunks + 1 history chunk - 1


def test_eligibility_no_history_chunks_at_all():
    ok, reason, meta = fork_eligibility([], tool_chunk_count=3, target_has_tool_call=True)
    assert (ok, reason) == (False, "history_chunks<2")
    assert meta["fork_chunk_index"] is None
    assert meta["last_chunk_tokens"] is None


@pytest.mark.parametrize(
    "last_len,expected_ok,expected_reason",
    [
        (63, False, "last_chunk_tokens<64"),
        (64, True, None),
        (1024, True, None),
        (1025, False, "last_chunk_tokens>1024"),
    ],
)
def test_eligibility_last_chunk_bounds(last_len, expected_ok, expected_reason):
    ok, reason, meta = fork_eligibility(
        [300, last_len], tool_chunk_count=5, target_has_tool_call=True
    )
    assert ok is expected_ok
    assert reason == expected_reason
    assert meta["last_chunk_tokens"] == last_len
    assert meta["shared_history_tokens"] == 300
    assert meta["fork_chunk_index"] == 6  # 5 tool chunks + 2 history chunks - 1
    assert meta["shared_chunk_count"] == 6


def test_eligibility_requires_tool_call_target():
    ok, reason, _meta = fork_eligibility(
        [300, 300], tool_chunk_count=2, target_has_tool_call=False
    )
    assert (ok, reason) == (False, "target_has_tool_call=false")


def test_eligibility_custom_l_min_shifts_the_boundary():
    assert fork_eligibility([300, 96], 2, True, l_min=128)[1] == "last_chunk_tokens<128"
    assert fork_eligibility([300, 96], 2, True, l_min=96)[0] is True


# ---------------------------------------------------------------------------
# c. deterministic checks + action keys
# ---------------------------------------------------------------------------


def test_deterministic_check_pass_accepts_dict_and_json_string_arguments():
    assert deterministic_check_pass({"name": "get_weather", "arguments": {"city": "Paris"}})
    assert deterministic_check_pass({"name": "get_weather", "arguments": '{"city": "Paris"}'})
    assert deterministic_check_pass({"name": "noop", "arguments": {}})


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {},
        {"name": "", "arguments": {}},
        {"name": "   ", "arguments": {}},
        {"name": "ok", "arguments": "not json at all"},
        {"name": "ok", "arguments": "[1, 2]"},
        {"name": "ok", "arguments": 17},
        {"name": 5, "arguments": {}},
    ],
)
def test_deterministic_check_pass_rejects_broken_calls(parsed):
    assert deterministic_check_pass(parsed) is False


def test_action_key_is_argument_order_insensitive():
    left = action_key({"name": "f", "arguments": {"a": 1, "b": 2}})
    right = action_key({"name": "f", "arguments": {"b": 2, "a": 1}})
    assert left == right
    assert action_key({"name": "f", "arguments": {"a": 2, "b": 1}}) != left
    assert action_key(None) is None
    assert action_key({"arguments": {}}) is None


# ---------------------------------------------------------------------------
# d. F3 tie table + F1 rollout tie rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_pass,b_pass,r1,r1b",
    [
        (True, False, BRANCH_COMPRESS_NOW, BRANCH_COMPRESS_NOW),
        (False, True, BRANCH_DEFER, BRANCH_DEFER),
        (True, True, BRANCH_COMPRESS_NOW, BRANCH_DEFER),
        (False, False, BRANCH_COMPRESS_NOW, BRANCH_COMPRESS_NOW),
    ],
)
def test_f3_select_truth_table(a_pass, b_pass, r1, r1b):
    row_a = {"deterministic_check_pass": a_pass}
    row_b = {"deterministic_check_pass": b_pass}
    assert f3_select(row_a, row_b, rule="R1") == r1
    assert f3_select(row_a, row_b, rule="R1b") == r1b


def test_f3_select_rejects_unknown_rule():
    with pytest.raises(ValueError, match="R1"):
        f3_select({"deterministic_check_pass": True}, {"deterministic_check_pass": True}, rule="R2")


def test_f3_select_falls_back_to_parsed_call_when_flag_absent():
    row_a = {"parsed_call": {"name": "", "arguments": {}}}
    row_b = {"parsed_call": {"name": "f", "arguments": {}}}
    assert f3_select(row_a, row_b) == BRANCH_DEFER


@pytest.mark.parametrize(
    "flags,expected",
    [((True, True), 0), ((True, False), 0), ((False, True), 1), ((False, False), 0)],
)
def test_select_rollout_by_checks_ties_to_rollout_zero(flags, expected):
    rows = [{"deterministic_check_pass": flag} for flag in flags]
    assert select_rollout_by_checks(rows) == expected


# ---------------------------------------------------------------------------
# e. F4 coin
# ---------------------------------------------------------------------------


def test_f4_coin_is_deterministic_and_seed_sensitive():
    assert f4_coin("s0:3", 0) == f4_coin("s0:3", 0)
    assert {f4_coin(f"q{i}", 0) for i in range(50)} == {BRANCH_COMPRESS_NOW, BRANCH_DEFER}
    flips = sum(1 for i in range(200) if f4_coin(f"q{i}", 0) != f4_coin(f"q{i}", 1))
    assert flips > 0


def test_f4_coin_is_balanced_on_a_synthetic_pool():
    qids = [f"session{i // 4}:{i % 4}" for i in range(1000)]
    heads = sum(1 for qid in qids if f4_coin(qid, 20260822) == BRANCH_COMPRESS_NOW)
    assert 0.42 < heads / len(qids) < 0.58


# ---------------------------------------------------------------------------
# f. derive_arms
# ---------------------------------------------------------------------------


def test_slot_name_maps_the_five_recorded_rollouts():
    assert slot_name(_row("q", "greedy_core", BRANCH_COMPRESS_NOW, 0)) == "A_greedy"
    assert slot_name(_row("q", "greedy_core", BRANCH_DEFER, 0)) == "B_greedy"
    assert slot_name(_row("q", "sampled", BRANCH_COMPRESS_NOW, 0)) == "A_s0"
    assert slot_name(_row("q", "sampled", BRANCH_COMPRESS_NOW, 1)) == "A_s1"
    assert slot_name(_row("q", "sampled", BRANCH_DEFER, 0)) == "B_s0"
    assert slot_name(_row("q", "greedy_core", BRANCH_DEFER, 1)) is None


def test_index_rows_by_qid_drops_skipped_and_unknown_slots():
    rows = [
        *_greedy_pair("q0"),
        _row("q1", "greedy_core", BRANCH_COMPRESS_NOW, 0, skipped=True),
        _row("q2", "greedy_core", "unknown_branch", 0),
    ]
    indexed = index_rows_by_qid(rows)
    assert set(indexed) == {"q0"}
    assert set(indexed["q0"]) == {"A_greedy", "B_greedy"}


def test_index_rows_by_qid_collapses_a_regenerated_pass_last_write_wins():
    # Resume regenerates a whole pass, so the jsonl can legitimately hold two
    # rows for the same slot; the later one is the live result.
    rows = [
        *_greedy_pair("q0", a_kwargs={"tool_name_match": False}),
        *_greedy_pair("q0", a_kwargs={"tool_name_match": True}),
    ]
    indexed = index_rows_by_qid(rows)
    assert len(indexed["q0"]) == 2
    assert indexed["q0"]["A_greedy"]["tool_name_match"] is True


def test_derive_arms_union_is_or_for_binary_and_max_for_continuous():
    rows = _greedy_pair(
        "q0",
        a_kwargs={"tool_name_match": False, "f1": 0.2, "pred": "X", "gold": "A"},
        b_kwargs={"tool_name_match": True, "f1": 0.6, "pred": "A", "gold": "A"},
    )
    derived = derive_arms(index_rows_by_qid(rows), seed=0)
    union = derived["arms"]["F5"]["q0"]
    assert union["tool_name_match"] is True
    assert union["action_key_match"] is True
    assert union["argument_value_f1"] == 0.6
    assert derived["selection"]["F5"]["q0"] == "union"


def test_derive_arms_omits_sampled_arms_on_a_greedy_only_run():
    rows = _greedy_pair("q0")
    derived = derive_arms(index_rows_by_qid(rows), seed=0)
    assert set(derived["arms"]) == {"F0", "F2", "F3g", "F3g_R1b", "F4", "F5"}


def test_derive_arms_includes_sampled_arms_when_slots_are_present():
    rows = [
        *_greedy_pair("q0"),
        _row("q0", "sampled", BRANCH_COMPRESS_NOW, 0, check=False, tool=False),
        _row("q0", "sampled", BRANCH_COMPRESS_NOW, 1, check=True, tool=True),
        _row("q0", "sampled", BRANCH_DEFER, 0, check=True, tool=True),
    ]
    derived = derive_arms(index_rows_by_qid(rows), seed=0)
    assert "F1" in derived["arms"] and "F3s" in derived["arms"]
    # A_s0 fails the checks, A_s1 passes -> F1 takes rollout 1.
    assert derived["selection"]["F1"]["q0"] == "A_s1"
    # A_s0 fails, B_s0 passes -> F3s defers.
    assert derived["selection"]["F3s"]["q0"] == "B_s0"


def test_derive_arms_delta_oracle_arithmetic_on_hand_counted_rows():
    # 4 qids: A right on q0 only; B right on q1 and q2; neither on q3.
    # F0 = 1/4, F2 = 2/4, F5 = 3/4 -> delta_oracle vs best single (F2) = 0.25.
    plan = {
        "q0": (True, False),
        "q1": (False, True),
        "q2": (False, True),
        "q3": (False, False),
    }
    rows = []
    for qid, (a_ok, b_ok) in plan.items():
        rows.extend(
            _greedy_pair(
                qid,
                a_kwargs={"tool_name_match": a_ok},
                b_kwargs={"tool_name_match": b_ok},
                session=qid[:2],
            )
        )
    derived = derive_arms(index_rows_by_qid(rows), seed=0)

    def _rate(arm):
        values = derived["arms"][arm]
        return sum(1 for item in values.values() if item["tool_name_match"]) / len(values)

    assert _rate("F0") == 0.25
    assert _rate("F2") == 0.5
    assert _rate("F5") == 0.75
    assert round(_rate("F5") - max(_rate("F0"), _rate("F2")), 6) == 0.25


def test_derive_arms_coin_selection_matches_f4_coin():
    rows = [*_greedy_pair("q0"), *_greedy_pair("q1"), *_greedy_pair("q2")]
    derived = derive_arms(index_rows_by_qid(rows), seed=7)
    for qid, chosen in derived["selection"]["F4"].items():
        expected = "A_greedy" if f4_coin(qid, 7) == BRANCH_COMPRESS_NOW else "B_greedy"
        assert chosen == expected


# ---------------------------------------------------------------------------
# g. descriptive blocks
# ---------------------------------------------------------------------------


def _four_cell_pool():
    plan = {
        "q0": (True, True),
        "q1": (True, False),
        "q2": (False, True),
        "q3": (False, False),
        "q4": (False, False),
    }
    rows = []
    for qid, (a_ok, b_ok) in plan.items():
        rows.extend(
            _greedy_pair(
                qid,
                a_kwargs={"tool_name_match": a_ok, "pred": "A" if a_ok else "X"},
                b_kwargs={"tool_name_match": b_ok, "pred": "A" if b_ok else "Y"},
            )
        )
    return index_rows_by_qid(rows)


def test_four_cell_counts():
    cell = four_cell(_four_cell_pool(), "tool_name_match")
    assert cell["counts"] == {
        "both": 1,
        "compress_now_only": 1,
        "defer_only": 1,
        "neither": 2,
    }
    assert cell["n"] == 5
    assert cell["rates"]["neither"] == 0.4
    assert cell["qids_by_cell"]["compress_now_only"] == ["q1"]


def test_pairwise_disagreement_counts_and_unparsed():
    rows = [
        *_greedy_pair("q0", a_kwargs={"pred": "A"}, b_kwargs={"pred": "A"}),
        *_greedy_pair("q1", a_kwargs={"pred": "A"}, b_kwargs={"pred": "B"}),
        *_greedy_pair("q2", a_kwargs={"pred": None}, b_kwargs={"pred": None}),
        *_greedy_pair("q3", a_kwargs={"pred": None}, b_kwargs={"pred": "B"}),
    ]
    block = pairwise_disagreement(index_rows_by_qid(rows))
    assert block["n"] == 4
    assert block["disagree"] == 2
    assert block["disagree_qids"] == ["q1", "q3"]
    assert block["both_unparsed"] == 1
    assert block["disagree_rate"] == 0.5


def test_both_match_gold_is_the_strict_subset():
    rows = [
        *_greedy_pair("q0", a_kwargs={"pred": "A"}, b_kwargs={"pred": "A"}),
        *_greedy_pair("q1", a_kwargs={"pred": "A"}, b_kwargs={"pred": "B"}),
        *_greedy_pair("q2", a_kwargs={"pred": "C"}, b_kwargs={"pred": "C"}),
        *_greedy_pair(
            "q3",
            a_kwargs={"pred": "A", "gold": None},
            b_kwargs={"pred": "A", "gold": None},
        ),
    ]
    block = both_match_gold(index_rows_by_qid(rows))
    # q3 has no gold key -> excluded from the denominator entirely.
    assert block["n_scored"] == 3
    assert block["count"] == 1
    assert block["qids"] == ["q0"]
    assert block["rate"] == round(1 / 3, 4)


# ---------------------------------------------------------------------------
# h. cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_ci_is_seed_reproducible():
    deltas = {"s0": [1.0, 0.0, 1.0], "s1": [0.0, 0.0], "s2": [1.0, 1.0, 0.0, 0.0]}
    left = cluster_bootstrap_ci(deltas, b=500, seed=1234)
    right = cluster_bootstrap_ci(deltas, b=500, seed=1234)
    assert left == right
    other = cluster_bootstrap_ci(deltas, b=500, seed=4321)
    assert other["point"] == left["point"]  # the point estimate is seed-free
    assert left["n"] == 9
    assert left["n_clusters"] == 3
    assert left["ci95"][0] <= left["point"] <= left["ci95"][1]
    assert left["degenerate_single_cluster"] is False


def test_cluster_bootstrap_ci_single_cluster_is_degenerate_and_flagged():
    block = cluster_bootstrap_ci({"s0": [1.0, 0.0, 1.0, 0.0]}, b=100, seed=7)
    assert block["degenerate_single_cluster"] is True
    assert block["ci95"] == [block["point"], block["point"]]


def test_cluster_bootstrap_ci_empty_input():
    block = cluster_bootstrap_ci({}, b=10, seed=0)
    assert block["point"] is None
    assert block["n"] == 0
    assert block["ci95"] == [None, None]


# ---------------------------------------------------------------------------
# i. resume bookkeeping
# ---------------------------------------------------------------------------


def test_load_done_keys_retries_skipped_rows(tmp_path):
    path = tmp_path / "f_fork.jsonl"
    rows = [
        _row("q0", "greedy_core", BRANCH_COMPRESS_NOW, 0),
        _row("q0", "greedy_core", BRANCH_DEFER, 0, skipped=True),
        _row("q1", "sampled", BRANCH_COMPRESS_NOW, 1),
        {"qid": "q2", "arm_pass": "greedy_core"},  # missing branch -> ignored
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write("\n")
        handle.write("{not json}\n")
    done = load_done_keys(path)
    assert done == {
        ("q0", "greedy_core", BRANCH_COMPRESS_NOW, 0),
        ("q1", "sampled", BRANCH_COMPRESS_NOW, 1),
    }
    # The skipped defer rollout is NOT done, so resume regenerates it.
    assert ("q0", "greedy_core", BRANCH_DEFER, 0) not in done


def test_load_done_keys_on_missing_file(tmp_path):
    assert load_done_keys(tmp_path / "nope.jsonl") == set()
