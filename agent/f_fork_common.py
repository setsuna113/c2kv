"""F-line (speculative compaction timing fork) pure logic — torch-free.

Shared by the driver (``agent/f_timing_fork.py``) and the analyzer
(``agent/analyze_f_fork.py``).  Only stdlib is imported here so the analyzer
never pulls torch: the driver writes ``pred_action_key`` / ``gold_action_key``
into every row (``eval_toolathlon_first_tool_c2kv._parse_pred_call`` composed
with :func:`action_key`), and every downstream derivation in this module works
off those plain strings plus the recorded metric fields.

Vocabulary (frozen in ``configs/bdf_pilot/f_prereg.md``):

- branch ``compress_now`` — the already-scheduled compaction runs on the last
  history chunk before the current turn is answered (the null policy);
- branch ``defer`` — that one chunk stays raw for this single decision while
  the older prefix stays the gist prefix both branches share;
- ``deterministic_check_*`` is the ONLY name the F line gives the mechanical
  output check, in fields, functions and prose alike; no synonym for it is
  introduced anywhere.

Nothing in this module implements a stopping rule or a kill decision: it
produces counts, derived arms and intervals, and the caller reads them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# --- branch / pass vocabulary ------------------------------------------------

BRANCH_COMPRESS_NOW = "compress_now"
BRANCH_DEFER = "defer"
BRANCHES = (BRANCH_COMPRESS_NOW, BRANCH_DEFER)

ARM_PASS_GREEDY = "greedy_core"
ARM_PASS_SAMPLED = "sampled"
ARM_PASSES = (ARM_PASS_GREEDY, ARM_PASS_SAMPLED)

# Row slot names: one recorded generation each.  A = compress_now, B = defer.
SLOT_BY_ROW_KEY: Dict[Tuple[str, str, int], str] = {
    (ARM_PASS_GREEDY, BRANCH_COMPRESS_NOW, 0): "A_greedy",
    (ARM_PASS_GREEDY, BRANCH_DEFER, 0): "B_greedy",
    (ARM_PASS_SAMPLED, BRANCH_COMPRESS_NOW, 0): "A_s0",
    (ARM_PASS_SAMPLED, BRANCH_COMPRESS_NOW, 1): "A_s1",
    (ARM_PASS_SAMPLED, BRANCH_DEFER, 0): "B_s0",
}

GREEDY_SLOTS = ("A_greedy", "B_greedy")
SAMPLED_SLOTS = ("A_s0", "A_s1", "B_s0")

# Arm -> (combination rule, slots it consumes).  Slot order is always
# (compress_now-side, defer-side) so "select" rules can name the winner.
ARM_PLAN: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "F0": ("single", ("A_greedy",)),
    "F2": ("single", ("B_greedy",)),
    "F3g": ("select_r1", GREEDY_SLOTS),
    "F3g_R1b": ("select_r1b", GREEDY_SLOTS),
    "F4": ("coin", GREEDY_SLOTS),
    "F5": ("union", GREEDY_SLOTS),
    "F1": ("select_rollout", ("A_s0", "A_s1")),
    "F1_union": ("union", ("A_s0", "A_s1")),
    "F3s": ("select_r1", ("A_s0", "B_s0")),
    "F3s_R1b": ("select_r1b", ("A_s0", "B_s0")),
    "F3s_union": ("union", ("A_s0", "B_s0")),
}

DEFAULT_METRICS = ("tool_name_match", "action_key_match", "argument_value_f1")

# Eligibility criteria names, mirroring configs/bdf_pilot/f_prereg.md E1-E4.
SKIP_HISTORY_CHUNKS = "history_chunks<{n}"
SKIP_LAST_CHUNK_SMALL = "last_chunk_tokens<{n}"
SKIP_LAST_CHUNK_LARGE = "last_chunk_tokens>{n}"
SKIP_NO_TOOL_CALL = "target_has_tool_call=false"

BOOTSTRAP_SEED = 20260822
BOOTSTRAP_B = 2000


# ---------------------------------------------------------------------------
# Memory arithmetic.
# ---------------------------------------------------------------------------


def kv_bytes_per_token(
    num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Bytes of KV cache one resident token costs across the whole stack.

    Both K and V are stored, hence the factor 2.  Qwen3-4B in bf16
    (36 layers, 8 KV heads, head_dim 128, 2 bytes) => 147456 B = 144 KiB.
    """

    for name, value in (
        ("num_layers", num_layers),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("dtype_bytes", dtype_bytes),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    return int(num_layers) * 2 * int(num_kv_heads) * int(head_dim) * int(dtype_bytes)


# ---------------------------------------------------------------------------
# Eligibility (prereg E1-E4).  E1 (builder skip) is inherited by the caller.
# ---------------------------------------------------------------------------


def fork_eligibility(
    history_chunk_lens: Sequence[int],
    tool_chunk_count: int,
    target_has_tool_call: bool,
    *,
    l_min: int = 64,
    max_doc_length: int = 1024,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Decide whether one example carries exactly one usable timing fork.

    E2 ``len(history_chunks) >= 2``; E3 the last history chunk holds
    ``[l_min, max_doc_length]`` tokens (``l_min`` keeps at least ``l_min/ratio``
    gist slots so branch A is not degenerate); E4 the target is a tool call.
    The fork is always the boundary AFTER the last history chunk, so an
    eligible example contributes exactly one paired decision.
    """

    lens = [int(item) for item in history_chunk_lens]
    last_chunk_tokens = lens[-1] if lens else None
    meta: Dict[str, Any] = {
        "tool_chunk_count": int(tool_chunk_count),
        "history_chunk_count": len(lens),
        "last_chunk_tokens": last_chunk_tokens,
        "fork_chunk_index": (int(tool_chunk_count) + len(lens) - 1) if lens else None,
        "shared_chunk_count": (int(tool_chunk_count) + len(lens) - 1) if lens else None,
        "shared_history_tokens": sum(lens[:-1]) if lens else 0,
        "l_min": int(l_min),
        "max_doc_length": int(max_doc_length),
        "target_has_tool_call": bool(target_has_tool_call),
    }
    if len(lens) < 2:
        return False, SKIP_HISTORY_CHUNKS.format(n=2), meta
    if last_chunk_tokens < l_min:
        return False, SKIP_LAST_CHUNK_SMALL.format(n=l_min), meta
    if last_chunk_tokens > max_doc_length:
        return False, SKIP_LAST_CHUNK_LARGE.format(n=max_doc_length), meta
    if not target_has_tool_call:
        return False, SKIP_NO_TOOL_CALL, meta
    return True, None, meta


# ---------------------------------------------------------------------------
# Deterministic checks + action keys.
# ---------------------------------------------------------------------------


def deterministic_check_pass(parsed_call: Optional[Mapping[str, Any]]) -> bool:
    """Mechanical, gold-free check on one parsed action.

    Passes when the call carries a non-empty ``name`` and an ``arguments``
    payload that resolves to a dict (a JSON string is decoded first).  This is
    everything a running system could check before the outcome is known — no
    reference to the gold action anywhere.
    """

    if not isinstance(parsed_call, Mapping):
        return False
    name = parsed_call.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    arguments = parsed_call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return False
    return isinstance(arguments, Mapping)


def action_key(parsed_call: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Canonical, order-insensitive string for one parsed action call.

    ``None`` when the text carried no parseable call.  Argument dicts are
    dumped with ``sort_keys=True`` so two calls that differ only in key order
    compare equal.
    """

    if not isinstance(parsed_call, Mapping):
        return None
    name = parsed_call.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = parsed_call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass
    if not isinstance(arguments, (Mapping, list)):
        arguments = {} if arguments is None else arguments
    try:
        payload = json.dumps(
            {"name": name.strip(), "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        payload = json.dumps(
            {"name": name.strip(), "arguments": str(arguments)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return payload


def _check_flag(row: Mapping[str, Any]) -> bool:
    value = row.get("deterministic_check_pass")
    if value is None:
        return deterministic_check_pass(row.get("parsed_call"))
    return bool(value)


def f3_select(
    row_a: Mapping[str, Any], row_b: Mapping[str, Any], rule: str = "R1"
) -> str:
    """F3 selection between the compress-now row and the defer row.

    R1 (preregistered primary): exactly one branch passes the deterministic
    checks -> keep it; BOTH pass -> keep ``compress_now``; BOTH fail -> keep
    ``compress_now``.  Keeping compressed on every tie makes the null policy
    ("the compaction that was already scheduled") the default, so F3-F0 is
    attributable to check-driven defers only, and F3 cannot win by silently
    collapsing into F2.  R1b is the preregistered sensitivity: both-pass keeps
    ``defer`` instead.  It costs nothing to compute at analysis time.
    """

    if rule not in ("R1", "R1b"):
        raise ValueError(f"Unsupported tie rule {rule!r}; choose R1 or R1b")
    a_pass = _check_flag(row_a)
    b_pass = _check_flag(row_b)
    if a_pass and not b_pass:
        return BRANCH_COMPRESS_NOW
    if b_pass and not a_pass:
        return BRANCH_DEFER
    if a_pass and b_pass:
        return BRANCH_DEFER if rule == "R1b" else BRANCH_COMPRESS_NOW
    return BRANCH_COMPRESS_NOW


def select_rollout_by_checks(rows: Sequence[Mapping[str, Any]]) -> int:
    """F1 selection among same-branch rollouts: first passing one, else 0.

    Both-pass and none-pass both resolve to rollout 0, which is the
    preregistered tie rule.
    """

    for index, row in enumerate(rows):
        if _check_flag(row):
            return index
    return 0


def f4_coin(qid: str, seed: int) -> str:
    """Seeded fair coin over the two branches — the F4 noise floor.

    Analysis-time only: it picks between two already-recorded generations, so
    it buys no extra rollouts.
    """

    digest = hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest()
    return BRANCH_DEFER if int(digest, 16) % 2 else BRANCH_COMPRESS_NOW


# ---------------------------------------------------------------------------
# Row indexing + arm derivation.
# ---------------------------------------------------------------------------


def slot_name(row: Mapping[str, Any]) -> Optional[str]:
    """Slot label of one recorded row, or None if it is not a fork rollout."""

    try:
        rollout_index = int(row.get("rollout_index") or 0)
    except (TypeError, ValueError):
        return None
    return SLOT_BY_ROW_KEY.get(
        (row.get("arm_pass"), row.get("branch"), rollout_index)
    )


def index_rows_by_qid(
    rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{qid: {slot: row}}`` over non-skipped fork rows (last write wins)."""

    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("skipped"):
            continue
        qid = row.get("qid")
        name = slot_name(row)
        if qid is None or name is None:
            continue
        out[qid][name] = dict(row)
    return dict(out)


def _union_value(left: Any, right: Any) -> Any:
    """Union of one metric across two branches: OR for binary, max otherwise."""

    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) or bool(right)
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def derive_arms(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    seed: int,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, Any]:
    """Derive every arm from the recorded rollouts — no extra generation.

    Returns ``{"arms": {arm: {qid: {metric: value}}}, "selection": {arm: {qid:
    slot}}, "slots": {arm: [...]}, "rules": {arm: rule}}``.  An arm appears
    only for the qids where every slot it consumes is present, and is omitted
    entirely when no qid supplies it (so a greedy-only run yields no F1/F3s).
    """

    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    selection: Dict[str, Dict[str, str]] = {}
    slots_used: Dict[str, List[str]] = {}
    rules_used: Dict[str, str] = {}
    for arm, (rule, slot_tuple) in ARM_PLAN.items():
        per_qid: Dict[str, Dict[str, Any]] = {}
        per_choice: Dict[str, str] = {}
        for qid, slot_rows in rows_by_qid.items():
            if any(slot not in slot_rows for slot in slot_tuple):
                continue
            if rule == "single":
                chosen = slot_tuple[0]
            elif rule == "union":
                left, right = (slot_rows[slot] for slot in slot_tuple)
                per_qid[qid] = {
                    metric: _union_value(left.get(metric), right.get(metric))
                    for metric in metrics
                }
                per_choice[qid] = "union"
                continue
            elif rule in ("select_r1", "select_r1b"):
                branch = f3_select(
                    slot_rows[slot_tuple[0]],
                    slot_rows[slot_tuple[1]],
                    rule="R1" if rule == "select_r1" else "R1b",
                )
                chosen = slot_tuple[0] if branch == BRANCH_COMPRESS_NOW else slot_tuple[1]
            elif rule == "coin":
                branch = f4_coin(qid, seed)
                chosen = slot_tuple[0] if branch == BRANCH_COMPRESS_NOW else slot_tuple[1]
            elif rule == "select_rollout":
                index = select_rollout_by_checks(
                    [slot_rows[slot] for slot in slot_tuple]
                )
                chosen = slot_tuple[index]
            else:  # pragma: no cover - ARM_PLAN is a closed table
                raise ValueError(f"Unknown arm rule {rule!r}")
            per_qid[qid] = {metric: slot_rows[chosen].get(metric) for metric in metrics}
            per_choice[qid] = chosen
        if per_qid:
            arms[arm] = per_qid
            selection[arm] = per_choice
            slots_used[arm] = list(slot_tuple)
            rules_used[arm] = rule
    return {
        "arms": arms,
        "selection": selection,
        "slots": slots_used,
        "rules": rules_used,
        "metrics": list(metrics),
        "coin_seed": seed,
    }


# ---------------------------------------------------------------------------
# Descriptive blocks over the branch pair.
# ---------------------------------------------------------------------------


def four_cell(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric: str,
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    """2x2 table of (compress_now succeeds) x (defer succeeds) on one metric."""

    counts = {"both": 0, "compress_now_only": 0, "defer_only": 0, "neither": 0}
    qids_by_cell: Dict[str, List[str]] = {key: [] for key in counts}
    for qid, slot_rows in sorted(rows_by_qid.items()):
        if any(slot not in slot_rows for slot in slots):
            continue
        a_ok = bool(slot_rows[slots[0]].get(metric))
        b_ok = bool(slot_rows[slots[1]].get(metric))
        cell = (
            "both" if a_ok and b_ok
            else "compress_now_only" if a_ok
            else "defer_only" if b_ok
            else "neither"
        )
        counts[cell] += 1
        qids_by_cell[cell].append(qid)
    total = sum(counts.values())
    return {
        "metric": metric,
        "slots": list(slots),
        "n": total,
        "counts": counts,
        "rates": {
            key: (round(value / total, 4) if total else 0.0)
            for key, value in counts.items()
        },
        "qids_by_cell": qids_by_cell,
    }


def pairwise_disagreement(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    """How often the two branches emit a different action at all.

    Compares ``pred_action_key``; two unparseable outputs count as agreeing
    and are also reported separately so an all-garbage run is visible.
    """

    disagree: List[str] = []
    both_unparsed = 0
    total = 0
    for qid, slot_rows in sorted(rows_by_qid.items()):
        if any(slot not in slot_rows for slot in slots):
            continue
        total += 1
        left = slot_rows[slots[0]].get("pred_action_key")
        right = slot_rows[slots[1]].get("pred_action_key")
        if left is None and right is None:
            both_unparsed += 1
            continue
        if left != right:
            disagree.append(qid)
    return {
        "slots": list(slots),
        "n": total,
        "disagree": len(disagree),
        "disagree_rate": round(len(disagree) / total, 4) if total else 0.0,
        "both_unparsed": both_unparsed,
        "disagree_qids": disagree,
    }


def both_match_gold(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    """Strict subset where BOTH branches already emit the gold action.

    On these decisions the timing choice cannot change the outcome, so they
    dilute every selective arm's headroom.  Uses gold information and is
    therefore descriptive only — never a selection signal.
    """

    matched: List[str] = []
    scored = 0
    for qid, slot_rows in sorted(rows_by_qid.items()):
        if any(slot not in slot_rows for slot in slots):
            continue
        gold = slot_rows[slots[0]].get("gold_action_key")
        if gold is None:
            continue
        scored += 1
        if all(slot_rows[slot].get("pred_action_key") == gold for slot in slots):
            matched.append(qid)
    return {
        "slots": list(slots),
        "n_scored": scored,
        "count": len(matched),
        "rate": round(len(matched) / scored, 4) if scored else 0.0,
        "qids": matched,
    }


# ---------------------------------------------------------------------------
# Session-cluster bootstrap.
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(
    deltas_by_cluster: Mapping[str, Sequence[float]],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Percentile CI of the mean paired delta, resampling whole clusters.

    ``deltas_by_cluster`` maps a cluster id (``session_id`` for the F pilot) to
    that cluster's per-qid paired deltas.  Clusters are resampled with
    replacement; a single-cluster input therefore yields a degenerate interval,
    which is flagged rather than hidden.
    """

    clusters = [
        [float(value) for value in values]
        for _, values in sorted(deltas_by_cluster.items())
        if values
    ]
    flat = [value for cluster in clusters for value in cluster]
    if not flat:
        return {
            "point": None,
            "ci95": [None, None],
            "b": b,
            "seed": seed,
            "n": 0,
            "n_clusters": 0,
            "degenerate_single_cluster": False,
        }
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(b):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        pooled = [value for cluster in sample for value in cluster]
        draws.append(sum(pooled) / len(pooled))
    draws.sort()
    lo = draws[int(0.025 * b)]
    hi = draws[min(b - 1, int(0.975 * b))]
    return {
        "point": round(point, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "b": b,
        "seed": seed,
        "n": len(flat),
        "n_clusters": len(clusters),
        "degenerate_single_cluster": len(clusters) < 2,
    }


# ---------------------------------------------------------------------------
# Resume bookkeeping.
# ---------------------------------------------------------------------------


def load_done_keys(path: Any) -> Set[Tuple[str, str, str, int]]:
    """Keys of already-written rollouts: ``(qid, arm_pass, branch, rollout)``.

    Only NON-skipped rows count as done — a skipped row (eligibility miss, OOM)
    is retried on the next resume, matching ``r4_anchor_rerun._load_done_qids``.
    """

    done: Set[Tuple[str, str, str, int]] = set()
    target = Path(path)
    if not target.exists():
        return done
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("skipped"):
                continue
            qid = row.get("qid")
            arm_pass = row.get("arm_pass")
            branch = row.get("branch")
            if qid is None or arm_pass is None or branch is None:
                continue
            try:
                rollout_index = int(row.get("rollout_index") or 0)
            except (TypeError, ValueError):
                continue
            done.add((qid, arm_pass, branch, rollout_index))
    return done
