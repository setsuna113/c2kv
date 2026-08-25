"""Build the frozen mixture/order plan for the G-medium true-joint arms.

Why this exists (pre-registered constraints, docs/0820_g_joint_progress.md):

- All arms of the medium experiment must train on the SAME frozen example
  order; only ``--doc_mode`` differs.  The per-recipe ``*.order.json`` files
  written here are the exact inputs the trainer's ``--example_order_file``
  validates (unique qids, all loadable from the configured sources).
- Mixture ratios are measured in ESTIMATED COMPRESSIBLE SOURCE TOKENS (tool
  documents + history documents), never row counts — the same estimator the
  trainer's ``--max_source_tokens`` path uses.
- Held-out eval (BFCL, traces-v2 eval sessions) must never leak into train:
  ``--removal_files`` consumes the removal lists produced by
  ``agent/dedup_cross_dataset.py`` and filters the pools before sampling.
- Determinism: sources are seeded, pool scans stop at deterministic token
  caps, quota sampling and interleaving derive from ``--order_seed``; same
  inputs -> same outputs (plan JSONs carry no wall-clock fields).
- Stratified pool scan (P0-2): each family is scanned subset-by-subset
  (``_family_subsources``: qa = hotpotqa/2wiki/longmagpie, openswe = one
  stratum per trajectory-config dir, toucan = single stratum, traces = single
  stratum unless ``--split_traces_subsets``/``--traces_subset_map`` splits it
  into appworld/tau2 substrata plus an ``other`` catch-all — see the
  g_h200_main example below) with per-subset token caps (default equal
  weights; ``--subset_weights family:subset=w`` overrides).  A subset that
  exhausts below its cap hands the remaining budget to later siblings
  (sequential water-filling), and within a subset the parquet/jsonl FILE
  order is shuffled with
  ``random.Random(f"{order_seed}:scan:{family}:{subset}")`` (row order inside
  a file is unchanged) — so a cap-truncated scan is a seeded-random file
  prefix of every subset rather than the alphabetical head of the first one.
  Plan JSONs record both the sampled examples' per-subset breakdown
  (``families.<family>.subsets``) and the scan itself (``pool_scan``).

Outputs per recipe (and per ``<recipe>_repeat`` variant when
``--repeat_unique_tokens`` is given):

- ``<out_dir>/<recipe>.order.json`` — bare JSON list of qids;
- ``<out_dir>/<recipe>.plan.json`` — audit trail: shares, per-family
  examples/estimated tokens/realized share + per-subset breakdown, removals
  applied, ``budget_shrink_factor`` (P1-3: a short family pool shrinks ALL
  family quotas proportionally, logged loudly), repeat-variant
  ``recommended_epochs``, optional ``epochs_override`` audit record, the
  explicit alternate-pass asymmetry (``alternate_pass_counts``, P1-7), the
  fixed-arm launch table with the epochs x budget parity guard
  (``arm_launch_table``, P1-8), seeds.

Token-estimate cache: ``<out_dir>/tokencache_<family>.jsonl``, one JSON object
per line ``{"qid", "stamp", "estimated_tokens"}``.  An entry is reused only
when qid AND stamp match; the stamp is sha1 over an estimator-version tag, the
tokenizer identity and ``--split_seed`` (tool-document rendering depends on
it), so changing tokenizer/seed starts a fresh cache automatically.
Source-content changes under UNCHANGED qids are not detected — delete the
``tokencache_*.jsonl`` files (or use a fresh ``--out_dir``) when the corpora
change.

The module itself imports no heavy dependencies: pyarrow is used lazily inside
the data sources, and the tokenizer (``transformers.AutoTokenizer``) is loaded
lazily from ``--tokenizer``.  Without ``--tokenizer`` a whitespace fake
tokenizer is used (word counts; a warning is logged).

Usage:
  python agent/build_joint_medium_plan.py \
      --traces_path ~/c2kv/datasets/agent-llm-traces \
      --toucan_path ~/c2kv/datasets/toucan \
      --openswe_path ~/c2kv/datasets/open-swe-traces \
      --qa_hotpotqa_path ~/c2kv/datasets/qa/hotpotqa_train_cleaned/hotpotqa_train.jsonl \
      --qa_2wiki_path ~/c2kv/datasets/qa/2wiki --qa_longmagpie_path ~/c2kv/datasets/qa/longmagpie_cleaned \
      --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
      --recipe d_single=qa:0.2,traces:0.8 \
      --recipe d_multi=qa:0.2,traces:0.5,toucan:0.25,openswe:0.05 \
      --budget_estimated_tokens 133900000 --oversample_factor 1.25 \
      --repeat_unique_tokens 32000000 \
      --removal_files outputs/cross_dataset_dedup.json \
      --order_seed 42 --out_dir outputs/joint_medium_plan \
      --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 --small_arm_hours 12.5

H200 main arm — g_h200_main = 60% toucan + 30% tau2 traces + 10% AppWorld
traces (qa/openswe excluded).  The traces family quota (0.4) is split 75/25
over the tau2/appworld substrata via --subset_weights (0.4 x 0.75 = 0.30,
0.4 x 0.25 = 0.10 overall); the pool composition follows the scan-cap weights
modulo exhaustion, exactly like the qa substrata.  The arm trains with
REQUIRE_TOOL_CALL=False, so the scan MUST run with --no-require_tool_call
(pool parity), and the swebench/browsecompplus catch-all is excluded with
traces:other=0 (a zero weight skips the stratum entirely).

Step 1 (前置扫描, dry-run — no planning; confirm the real benchmark/subset
strings on the server data and warm the traces token cache):

  python agent/build_joint_medium_plan.py \
      --traces_path ~/c2kv/datasets/agent-llm-traces \
      --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
      --split_manifest_name taskproxy_disjoint \
      --removal_files outputs/cross_dataset_dedup.json \
      --no-require_tool_call \
      --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
      --out_dir outputs/joint_h200_plan --list_traces_subsets

Step 2 (plan; default classification *appworld* -> appworld, airline/retail/
telecom -> tau2, anything else kept under its raw name in "other"):

  python agent/build_joint_medium_plan.py \
      --traces_path ~/c2kv/datasets/agent-llm-traces \
      --toucan_path ~/c2kv/datasets/toucan \
      --split_manifest_file outputs/agent_taskproxy_split_manifest.json \
      --recipe g_h200_main=toucan:0.6,traces:0.4 \
      --split_traces_subsets \
      --split_manifest_name taskproxy_disjoint \
      --subset_weights traces:tau2=0.75 --subset_weights traces:appworld=0.25 \
      --subset_weights traces:other=0 \
      --no-require_tool_call \
      --budget_estimated_tokens <N> --oversample_factor 1.25 \
      --removal_files outputs/cross_dataset_dedup.json \
      --order_seed 42 --out_dir outputs/joint_h200_plan \
      --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507

If step 1 shows tau2 values other than airline/retail/telecom, pin the table
explicitly (implies --split_traces_subsets; REPLACES the default map):
  --traces_subset_map appworld=appworld,tau2=airline:retail:telecom
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    # Allow running as `python agent/build_joint_medium_plan.py` from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from train.train_data_joint import JointExample

logger = logging.getLogger(__name__)


FAMILIES = ("traces", "toucan", "openswe", "qa")
# ETA calibration unit: the small-arm runs measured wall-clock hours per 32M
# ESTIMATED source tokens (the estimator's nominal unit — the SAME unit the
# budget/quota math uses; presented source tokens are ~0.392x that).  The
# calibration is supplied via --small_arm_hours (joint mode) /
# --small_arm_hours_alternate; mixing presented/estimated units here would
# overestimate runtimes ~2.5x.
_TOKENS_PER_ETA_UNIT = 32_000_000
_ESTIMATE_VERSION = "estimate-v1"


# ---------------------------------------------------------------------------
# Recipe parsing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeSpec:
    """One mixture recipe: per-family shares in declared order, summing to 1."""

    name: str
    shares: Tuple[Tuple[str, float], ...]


def parse_recipe(spec: str) -> RecipeSpec:
    """Parse ``name=family:share,family:share,...`` (shares sum to 1 ± 1e-6)."""
    name, sep, body = spec.partition("=")
    if not sep or not name.strip():
        raise ValueError(f"--recipe must be name=family:share,..., got: {spec!r}")
    shares: List[Tuple[str, float]] = []
    for part in body.split(","):
        family, sep2, value = part.partition(":")
        if not sep2 or not family.strip():
            raise ValueError(f"--recipe entries must be family:share, got {part!r} in {spec!r}")
        family = family.strip()
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r} in --recipe {spec!r} (choose from {FAMILIES})")
        share = float(value)
        if share <= 0.0:
            raise ValueError(f"--recipe shares must be positive, got {part!r} in {spec!r}")
        shares.append((family, share))
    if not shares:
        raise ValueError(f"--recipe has no families: {spec!r}")
    total = sum(share for _, share in shares)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--recipe {name.strip()} shares must sum to 1.0 (±1e-6), got {total}")
    return RecipeSpec(name=name.strip(), shares=tuple(shares))


def parse_epochs_override(spec: str) -> Tuple[str, int]:
    """Parse one ``--epochs_override name=n`` value (n an integer >= 1)."""
    name, sep, value = spec.partition("=")
    if not sep or not name.strip():
        raise ValueError(f"--epochs_override must be name=epochs, got: {spec!r}")
    try:
        epochs = int(value)
    except ValueError:
        raise ValueError(f"--epochs_override must be name=integer>=1, got: {spec!r}") from None
    if epochs < 1:
        raise ValueError(f"--epochs_override must be an integer >= 1, got: {spec!r}")
    return name.strip(), epochs


def _parse_epochs_overrides(specs: Optional[Sequence[str]], valid_names: Set[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for spec in specs or []:
        name, epochs = parse_epochs_override(spec)
        if name not in valid_names:
            raise ValueError(
                f"--epochs_override references unknown recipe {name!r} (recipes: {sorted(valid_names)})"
            )
        overrides[name] = epochs
    return overrides


# ---------------------------------------------------------------------------
# Pools and the source-token estimator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolEntry:
    """One scanned example reduced to what planning needs (memory bound)."""

    qid: str
    session_id: str
    estimated_tokens: int
    # Scan stratum inside the family ("hotpotqa"/"2wiki"/"longmagpie" for qa,
    # the trajectory-config dir for openswe, the family name otherwise) — the
    # unit the stratified pool scan splits caps over (P0-2).  "" for synthetic
    # pools in tests.
    subset: str = ""


class WhitespaceTokenizer:
    """Fallback token estimator: one token per whitespace-separated word.

    Used when ``--tokenizer`` is not given (and by the unit tests).  Ratios
    measured in words track ratios in BPE tokens well enough for pool
    scanning; the trainer's own token accounting is unaffected either way.
    """

    name_or_path = "whitespace-fake"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> List[str]:
        return str(text).split()


def _load_tokenizer(tokenizer_path: Optional[str]):
    if tokenizer_path is None:
        logger.warning(
            "--tokenizer not given: estimating source tokens with the whitespace "
            "fake tokenizer (word counts, not BPE tokens)"
        )
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer  # lazy: the planner is CPU-only otherwise

    return AutoTokenizer.from_pretrained(tokenizer_path)


def estimate_source_tokens(example: "JointExample", tokenizer) -> int:
    """Mirror of ``train_joint_next_action_c2kv._estimate_source_tokens``.

    Re-implemented (3 lines) because the trainer module imports torch/models
    at import time and this planner must stay CPU-only: pre-chunking estimate
    = tool documents + history documents.  Duck-typed on purpose so tests can
    feed lightweight stand-ins.
    """
    return sum(
        len(tokenizer.encode(doc))
        for doc in example.tool_documents + example.history_documents
    )


# ---------------------------------------------------------------------------
# Token-estimate cache (qid + config stamp -> estimate).
# ---------------------------------------------------------------------------


def _cache_stamp(tokenizer_tag: str, split_seed: int) -> str:
    raw = f"{_ESTIMATE_VERSION}|tokenizer={tokenizer_tag}|split_seed={split_seed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_token_cache(path: Path) -> Dict[Tuple[str, str], int]:
    cache: Dict[Tuple[str, str], int] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("qid") and row.get("stamp"):
                cache[(str(row["qid"]), str(row["stamp"]))] = int(row.get("estimated_tokens") or 0)
    return cache


def _write_token_cache(path: Path, cache: Dict[Tuple[str, str], int]) -> None:
    lines = [
        json.dumps({"qid": qid, "stamp": stamp, "estimated_tokens": tokens}, ensure_ascii=False)
        for (qid, stamp), tokens in sorted(cache.items())
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Removal lists (dedup output) and pool filtering.
# ---------------------------------------------------------------------------


def load_removal_identifiers(files: Sequence[str]) -> Tuple[FrozenSet[str], Dict[str, int]]:
    """Collect removal identifiers from dedup removal lists.

    Each file is either a bare JSON list of identifier strings, or the
    ``dedup_cross_dataset.py`` output dict: entries under ``removal_list``
    carry ``record_id`` (session-level id) and ``unit_id``; both are
    collected.  Returns the identifier set and per-file entry counts.
    """
    identifiers: Set[str] = set()
    per_file: Dict[str, int] = {}
    for file in files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        entries = payload.get("removal_list") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(
                f"--removal_files entry must be a JSON list of strings or a dict "
                f"with a removal_list key: {file}"
            )
        before = len(identifiers)
        for entry in entries:
            if isinstance(entry, str):
                identifiers.add(entry)
            elif isinstance(entry, dict):
                for key in ("record_id", "unit_id"):
                    value = entry.get(key)
                    if value:
                        identifiers.add(str(value))
        per_file[str(file)] = len(identifiers) - before
    return frozenset(identifiers), per_file


def _removal_match_keys(qid: str, session_id: str) -> Set[str]:
    """Identifiers that may name this example in a removal list.

    Entries may be full qids or session ids, with or without the family
    prefix: a Toucan removal carries the bare ``uuid`` (dedup ``record_id``),
    matching ``toucan:<uuid>`` sessions and ``toucan:<uuid>:u<i>`` qids; an
    Open-SWE removal carries the bare ``trajectory_id``.  QA extraction units
    (agent/extract_medium_dedup_units.py) carry the FULL qid as their ``_id``
    (``qa:<subset>:<row_id>``; longmagpie row ids are
    ``<shard_stem>:<row_in_shard>``), so removal matches pool qids exactly;
    the stripped forms below keep older removal lists working.  Traces
    qids/session ids are bare already.
    """
    keys = {qid, session_id}
    for value in (qid, session_id):
        family, _, rest = value.partition(":")
        if rest and family in ("toucan", "openswe"):
            keys.add(rest)  # "<uuid>:u<i>" / "<uuid>", "<trajectory_id>:a<k>" / "<trajectory_id>"
        elif rest and family == "qa":
            keys.add(rest)  # "<subset>:<row_id>"
            _, _, last = rest.partition(":")
            if last:
                keys.add(last)  # bare "<row_id>"
    return keys


def apply_removals(
    pool: Sequence[PoolEntry],
    removal_identifiers: FrozenSet[str],
) -> Tuple[List[PoolEntry], int]:
    """Drop pool entries named by the removal list (session-id OR qid match)."""
    if not removal_identifiers:
        return list(pool), 0
    kept = [
        entry
        for entry in pool
        if not (_removal_match_keys(entry.qid, entry.session_id) & removal_identifiers)
    ]
    return kept, len(pool) - len(kept)


# ---------------------------------------------------------------------------
# Quota sampling and token-deficit weighted interleave (the pure planning core).
# ---------------------------------------------------------------------------


def _sample_to_quota(
    pool: Sequence[PoolEntry],
    quota_tokens: float,
    rng: random.Random,
) -> Tuple[List[PoolEntry], int, int]:
    """Seeded random sample filling ``quota_tokens`` estimated tokens.

    The pool is shuffled with ``rng`` and taken greedily until the cumulative
    estimate reaches the quota (the crossing example is included, mirroring
    the trainer's ``_take_within_source_token_budget``).  A short pool is
    taken in full; the third return value is the shortfall in estimated
    tokens (0 when the quota was filled).
    """
    order = list(pool)
    rng.shuffle(order)
    taken: List[PoolEntry] = []
    total = 0
    for entry in order:
        if total >= quota_tokens:
            break
        taken.append(entry)
        total += entry.estimated_tokens
    shortfall = max(0, int(math.ceil(quota_tokens)) - total)
    return taken, total, shortfall


def interleave_families(
    family_examples: Sequence[Tuple[str, List[Tuple[str, int]]]],
    shares: Sequence[Tuple[str, float]],
) -> List[str]:
    """Token-deficit weighted interleave in the recipe's declared family order.

    Recipe ratios are defined on estimated source TOKENS, while per-family
    mean example sizes differ by orders of magnitude (Open-SWE actions are
    huge, QA docs small).  A 1:1 example round-robin would exhaust the
    small-share families in the head of the sequence — the subset
    front-loading artifact the pre-registration forbids (the small arms hit it
    with appworld).  Instead, each step emits the next example of the
    non-exhausted family with the largest token deficit
    ``share * global_emitted_tokens - family_emitted_tokens``; at the first
    step all deficits are 0, so the first non-exhausted family in declared
    order leads, and deficits ties break by declared order.  Within a family
    the ``_sample_to_quota`` order is kept unchanged.  Exhausted families drop
    out and the remaining shares renormalize automatically, so the local token
    mix of any sliding window approximates the global shares.
    """
    share_of = dict(shares)
    # [family, examples, share, cursor, emitted_tokens] — mutable per-family state.
    state = [
        [family, examples, share_of.get(family, 0.0), 0, 0]
        for family, examples in family_examples
    ]
    out: List[str] = []
    global_tokens = 0
    while True:
        best = None
        best_deficit = None
        for entry in state:
            _, examples, share, cursor, emitted = entry
            if cursor >= len(examples) or share <= 0:
                continue
            deficit = share * global_tokens - emitted
            if best is None or deficit > best_deficit:
                best = entry
                best_deficit = deficit
        if best is None:
            return out
        _, examples, _, cursor, emitted = best
        qid, tokens = examples[cursor]
        out.append(qid)
        best[3] = cursor + 1
        best[4] = emitted + tokens
        global_tokens += tokens


def _fill_recipe(
    recipe: RecipeSpec,
    pools: Dict[str, List[PoolEntry]],
    token_scale: int,
    order_seed: int,
    seed_suffix: str,
) -> Tuple[List[str], Dict[str, Any], int]:
    """Sample each recipe family to ``share * token_scale`` and interleave."""
    family_examples: List[Tuple[str, List[Tuple[str, int]]]] = []
    reports: Dict[str, Any] = {}
    total = 0
    for family, share in recipe.shares:
        quota = share * token_scale
        pool = pools[family]
        taken, realized, shortfall = _sample_to_quota(
            pool, quota, random.Random(f"{order_seed}:{recipe.name}:{family}:{seed_suffix}")
        )
        total += realized
        # Per-subset attribution of the SAMPLED examples (P0-2 audit): which
        # strata actually filled the family quota, and with what share.
        subset_breakdown: Dict[str, Dict[str, Any]] = {}
        for entry in taken:
            sub = subset_breakdown.setdefault(
                entry.subset or "unknown", {"examples": 0, "estimated_tokens": 0}
            )
            sub["examples"] += 1
            sub["estimated_tokens"] += entry.estimated_tokens
        for sub in subset_breakdown.values():
            sub["share_within_family"] = sub["estimated_tokens"] / realized if realized else 0.0
        reports[family] = {
            "share": share,
            "quota_estimated_tokens": int(math.ceil(quota)),
            "examples": len(taken),
            "estimated_tokens": realized,
            "subsets": subset_breakdown,
            "pool_examples": len(pool),
            "pool_estimated_tokens": sum(entry.estimated_tokens for entry in pool),
            "shortfall_estimated_tokens": shortfall,
            "realized_share": None,  # filled in below
        }
        family_examples.append((family, [(entry.qid, entry.estimated_tokens) for entry in taken]))
    for report in reports.values():
        report["realized_share"] = (report["estimated_tokens"] / total) if total else 0.0
    return interleave_families(family_examples, recipe.shares), reports, total


def _alternate_pass_counts(recipe: RecipeSpec) -> Dict[str, Dict[str, int]]:
    """Supervised passes per family under the alternate doc_mode (P1-7).

    QA examples carry no tool documents, so their tool_only pass renders zero
    documents and is dropped (``qa:doc_num<2``): QA families contribute ONE
    supervised pass per epoch where tool-bearing families contribute TWO.  The
    asymmetry is intended; it must be explicit in the plan, not hidden inside
    an inflated aggregate skip count.
    """
    return {
        family: {"tool_only": 0 if family == "qa" else 1, "history_only": 1}
        for family, _ in recipe.shares
    }


_ALTERNATE_PASS_COUNTS_NOTE = (
    "alternate arm renders every example twice (tool_only + history_only passes); "
    "qa examples have no tool side, so their tool_only pass is always skipped "
    "(qa:doc_num<2 in the trainer manifest's train_skip_counts_by_family) and qa "
    "contributes one supervised pass per epoch where tool-bearing families contribute two."
)


def plan_recipes(
    pools: Dict[str, List[PoolEntry]],
    recipes: Sequence[RecipeSpec],
    *,
    budget_estimated_tokens: int,
    order_seed: int = 42,
    repeat_unique_tokens: Optional[int] = None,
    removal_identifiers: FrozenSet[str] = frozenset(),
    removal_file_counts: Optional[Dict[str, int]] = None,
    small_arm_hours: Optional[float] = None,
    small_arm_hours_alternate: Optional[float] = None,
    epochs_overrides: Optional[Dict[str, int]] = None,
    scan_removals: Optional[Dict[str, int]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    presented_target_est: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fill per-recipe family quotas and interleave into frozen orders.

    Pure planning core: pools in, frozen orders + audit plans out — no IO, no
    tokenizer, no wall clock (deterministic for fixed inputs; the ETA line is
    printed by ``main`` only).  Returns ``{name: {"order": [...], "plan":
    {...}}}``, including ``<recipe>_repeat`` variants when
    ``repeat_unique_tokens`` is set.  ``epochs_overrides`` (per recipe name)
    is a pure audit record: written into the plan and scales the reported
    ``presented_estimated_tokens``, nothing else — and it applies to BOTH the
    base and the repeat variant of the named recipe (the P1-8 arm launch
    table consumes it for both).

    ``scan_removals``: per-family removal counts already applied DURING the
    pool scan (P2: filtering before the cap preserves the oversample
    headroom); the in-plan ``apply_removals`` pass stays as a zero-hit safety
    net.  ``provenance`` is copied verbatim into every plan json.
    """
    if budget_estimated_tokens <= 0:
        raise ValueError(f"--budget_estimated_tokens must be positive, got {budget_estimated_tokens}")
    removal_counts: Dict[str, int] = {}
    filtered_pools: Dict[str, List[PoolEntry]] = {}
    for family, pool in pools.items():
        kept, removed = apply_removals(pool, removal_identifiers)
        filtered_pools[family] = kept
        removal_counts[family] = removed
    removals_report = {
        "by_family": scan_removals if scan_removals is not None else removal_counts,
        "identifiers": len(removal_identifiers),
        "by_file": removal_file_counts or {},
        "applied_at": "scan" if scan_removals is not None else "plan",
    }
    plan_stage_residual = sum(removal_counts.values())
    if scan_removals is not None and plan_stage_residual:
        # Scan-stage filtering should leave nothing for the plan stage; a
        # residual means the pool and the scan disagree — surface it loudly.
        removals_report["plan_stage_residual"] = plan_stage_residual
        logger.warning("plan-stage removal residual after scan-stage filtering: %d", plan_stage_residual)
    # P1-3: a family whose (filtered) pool is smaller than its quota must
    # shrink EVERY family's quota proportionally — otherwise the realized
    # shares skew past the prereg guardrails (e.g. traces-v1 at ~45M against
    # an 80M quota left d_single's qa realized share at ~31%, breaking the
    # 20% ratio).  The shrink factor is per recipe and per variant (base and
    # repeat pools differ), recorded in the plan and logged loudly.
    available_tokens = {
        family: sum(entry.estimated_tokens for entry in pool)
        for family, pool in filtered_pools.items()
    }

    def _budget_shrink_factor(recipe: RecipeSpec, token_scale: float) -> float:
        factor = 1.0
        for family, share in recipe.shares:
            quota = share * token_scale
            if quota > 0:
                factor = min(factor, available_tokens.get(family, 0.0) / quota)
        return min(1.0, factor)
    eta_info = (
        {
            "small_arm_hours": small_arm_hours,
            "small_arm_hours_alternate": (
                small_arm_hours_alternate if small_arm_hours_alternate is not None else small_arm_hours
            ),
            "tokens_per_unit": _TOKENS_PER_ETA_UNIT,
            "unit": "hours per 32M ESTIMATED source tokens (estimator 口径; presented is ~0.392x)",
        }
        if small_arm_hours is not None
        else None
    )
    epochs_overrides = epochs_overrides or {}
    results: Dict[str, Dict[str, Any]] = {}
    for recipe in recipes:
        missing = [family for family, _ in recipe.shares if family not in filtered_pools]
        if missing:
            raise ValueError(
                f"recipe {recipe.name!r} needs families with no configured source path: {missing}"
            )
        epochs_override = epochs_overrides.get(recipe.name)
        base_shrink = _budget_shrink_factor(recipe, budget_estimated_tokens)
        if base_shrink <= 0.0:
            raise ValueError(
                f"recipe {recipe.name!r}: a family pool is empty after removals "
                f"(available estimated tokens: "
                f"{ {family: int(available_tokens.get(family, 0)) for family, _ in recipe.shares} })"
            )
        effective_budget = budget_estimated_tokens * base_shrink
        if base_shrink < 1.0:
            logger.warning(
                "BUDGET SHRINK recipe=%s variant=base: pools bind the budget at factor %.4f "
                "(%d -> %d estimated tokens); all family quotas scaled together",
                recipe.name,
                base_shrink,
                budget_estimated_tokens,
                int(math.ceil(effective_budget)),
            )
        order, family_reports, total_tokens = _fill_recipe(
            recipe, filtered_pools, effective_budget, order_seed, seed_suffix="sample"
        )
        if len(order) != len(set(order)):
            raise RuntimeError(f"recipe {recipe.name}: duplicate qids in the planned order")
        results[recipe.name] = {
            "order": order,
            "plan": {
                "recipe": recipe.name,
                "variant": "base",
                "shares": dict(recipe.shares),
                "budget_estimated_tokens": budget_estimated_tokens,
                "budget_shrink_factor": base_shrink,
                "effective_budget_estimated_tokens": int(math.ceil(effective_budget)),
                "families": family_reports,
                "total_estimated_tokens": total_tokens,
                "epochs_override": epochs_override,
                "epochs_override_scope": "base_and_repeat_variants",
                "presented_estimated_tokens": total_tokens * (epochs_override or 1),
                "order_examples": len(order),
                "order_sha1": hashlib.sha1(json.dumps(order).encode("utf-8")).hexdigest(),
                "interleave": "token_deficit",
                "alternate_pass_counts": _alternate_pass_counts(recipe),
                "alternate_pass_counts_note": _ALTERNATE_PASS_COUNTS_NOTE,
                "removals": removals_report,
                "provenance": provenance,
                "seeds": {"order_seed": order_seed},
                "eta": eta_info,
            },
        }
        if repeat_unique_tokens:
            # Repeat variant: each family's pool truncated to share * M
            # estimated tokens (seeded sample, shares preserved); the order
            # file contains each qid ONCE — repetition happens at train time
            # via ``recommended_epochs``, never via duplicate qids.
            repeat_shrink = _budget_shrink_factor(recipe, repeat_unique_tokens)
            effective_repeat = repeat_unique_tokens * repeat_shrink
            if repeat_shrink < 1.0:
                logger.warning(
                    "BUDGET SHRINK recipe=%s variant=repeat: pools bind the unique pool at "
                    "factor %.4f (%d -> %d estimated tokens); all family quotas scaled together",
                    recipe.name,
                    repeat_shrink,
                    repeat_unique_tokens,
                    int(math.ceil(effective_repeat)),
                )
            rep_order, rep_reports, rep_total = _fill_recipe(
                recipe, filtered_pools, effective_repeat, order_seed, seed_suffix="repeat"
            )
            if len(rep_order) != len(set(rep_order)):
                raise RuntimeError(f"recipe {recipe.name}_repeat: duplicate qids in the planned order")
            recommended_epochs = math.ceil(budget_estimated_tokens / rep_total) if rep_total > 0 else 0
            # Variant-specific override key "<recipe>_repeat" wins over the
            # shared recipe key (which still covers base AND repeat).
            repeat_epochs_override = epochs_overrides.get(f"{recipe.name}_repeat", epochs_override)
            effective_epochs = repeat_epochs_override if repeat_epochs_override is not None else recommended_epochs
            results[f"{recipe.name}_repeat"] = {
                "order": rep_order,
                "plan": {
                    "recipe": recipe.name,
                    "variant": "repeat",
                    "shares": dict(recipe.shares),
                    "repeat_unique_tokens": repeat_unique_tokens,
                    "budget_shrink_factor": repeat_shrink,
                    "effective_repeat_unique_tokens": int(math.ceil(effective_repeat)),
                    "budget_estimated_tokens": budget_estimated_tokens,
                    "families": rep_reports,
                    "unique_pool_estimated_tokens": rep_total,
                    "recommended_epochs": recommended_epochs,
                    "epochs_override": repeat_epochs_override,
                    "epochs_override_scope": "base_and_repeat_variants",
                    "epochs_override_repeat_key": f"{recipe.name}_repeat",
                    "presented_estimated_tokens": rep_total * effective_epochs,
                    "order_examples": len(rep_order),
                    "order_sha1": hashlib.sha1(json.dumps(rep_order).encode("utf-8")).hexdigest(),
                    "order_note": "each qid appears once; repetition is achieved at train time via epochs",
                    "interleave": "token_deficit",
                    "alternate_pass_counts": _alternate_pass_counts(recipe),
                    "alternate_pass_counts_note": _ALTERNATE_PASS_COUNTS_NOTE,
                    "removals": removals_report,
                    "provenance": provenance,
                    "seeds": {"order_seed": order_seed},
                    "eta": eta_info,
                },
            }
    # P1-8: the fixed medium arms' epochs x budget parity is computed and
    # guarded HERE (before any output is written), not by the operator.
    arm_table = build_arm_launch_table(results, presented_target_est=presented_target_est)
    if arm_table is not None:
        for result in results.values():
            result["plan"]["arm_launch_table"] = arm_table
    return results


# ---------------------------------------------------------------------------
# P1-8: medium arm launch table + epochs x budget parity guard.
# ---------------------------------------------------------------------------

# Small-arm measured presented/estimated coefficient (progress doc sec.19):
# presented source tokens ≈ 0.392 x estimated source tokens for BOTH the joint
# and the alternate doc mode under per-side caps (the alternate arm's two
# passes sum to about the joint arm's single grid), so one coefficient
# suffices for cross-arm parity arithmetic.
PRESENTED_PER_ESTIMATED = 0.392
PARITY_SLACK = 1.02  # arms must agree within 2% of presented tokens

# Fixed medium launch mapping: (arm name, plan key, doc_mode).  The two
# d_single arms intentionally share ONE order file (the Gate-3 re-check pair).
MEDIUM_ARM_TABLE = (
    ("med_dsingle_alt", "d_single", "alternate"),
    ("med_dsingle_joint", "d_single", "joint"),
    ("med_dmulti_alt", "d_multi", "alternate"),
    ("med_dmulti_repeat_alt", "d_multi_repeat", "alternate"),
)
MEDIUM_ARM_CARDS = (2, 3, 4, 5)  # suggested NPU card per arm, in table order


def build_arm_launch_table(
    results: Dict[str, Dict[str, Any]],
    presented_target_est: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Cross-arm epochs x budget parity table for the fixed medium arms.

    Per arm: unique pool estimated tokens U (the order file's realized total),
    the effective epochs (``--epochs_override`` else ``recommended_epochs``
    for the repeat variant else 1), and the planned presented tokens
    ``U x epochs x PRESENTED_PER_ESTIMATED``.

    Two parity regimes:

    * ``presented_target_est=None`` (default): full-pool epochs.  Every arm's
      ``U x epochs x 0.392`` must agree within PARITY_SLACK (max <= floor x
      1.02); a violation raises loudly with per-arm level-up suggestions.
      MAX_SOURCE_TOKENS = U (per-epoch budget = the full unique pool).
      NOTE: with realistic U spreads (e.g. 53M/85M/25M) integer epochs cannot
      reach 2% — use the target regime below in that case.

    * ``presented_target_est=P*`` (truncation parity — the pre-registered
      medium semantics, "same total training tokens" per arm): the trainer
      truncates the frozen order at P* estimated source tokens, so presented
      parity is EXACT by construction.  The guard instead asserts CAPACITY:
      every arm must reach P* with 2% headroom, ``U x epochs >= P* x 1.02``,
      with epochs from ``--epochs_override`` or the per-arm suggestion
      ``ceil(P* x 1.02 / U)``.  MAX_SOURCE_TOKENS = P*.

    Returns None when none of the fixed recipes was planned (e.g. unit tests
    with toy recipe names), so exploratory runs are unaffected.
    """
    rows: List[Dict[str, Any]] = []
    for position, (arm_name, plan_key, doc_mode) in enumerate(MEDIUM_ARM_TABLE):
        if plan_key not in results:
            continue
        plan = results[plan_key]["plan"]
        if plan["variant"] == "repeat":
            unique_est = plan["unique_pool_estimated_tokens"]
            planned_epochs = plan["recommended_epochs"]
        else:
            unique_est = plan["total_estimated_tokens"]
            planned_epochs = 1
        epochs = plan["epochs_override"] if plan["epochs_override"] is not None else planned_epochs
        presented = unique_est * epochs * PRESENTED_PER_ESTIMATED
        rows.append({
            "arm": arm_name,
            "plan_key": plan_key,
            "variant": plan["variant"],
            "doc_mode": doc_mode,
            "order_file": f"{plan_key}.order.json",
            "unique_est_tokens": unique_est,
            "effective_epochs": epochs,
            "max_source_tokens": unique_est,
            "presented_est_tokens": presented,
            "suggested_card": MEDIUM_ARM_CARDS[position],
        })
    if not rows:
        return None
    floor = min(row["presented_est_tokens"] for row in rows)
    top = max(row["presented_est_tokens"] for row in rows)
    if presented_target_est is not None:
        # Truncation parity, trainer-true semantics: the trainer prefix-takes
        # the frozen order up to MAX_SOURCE_TOKENS ONCE (before the epoch
        # loop, ``_take_within_source_token_budget``) and repeats that prefix
        # every epoch — there is NO mid-epoch stop.  Parity therefore means
        # per-epoch take x epochs, not "truncate the whole run at P*":
        #   per_epoch_take = ceil(P* / effective_epochs)   (must be <= U)
        #   presented_est  = per_epoch_take x effective_epochs
        #     (~P* up to crossing-example granularity at the take, < one
        #     example per epoch — far inside the 2% slack)
        target = presented_target_est
        for row in rows:
            epochs = row["effective_epochs"]
            take = math.ceil(target / epochs)
            row["max_source_tokens"] = take
            row["presented_est_tokens"] = float(take * epochs)
            # Suggested epochs: the SMALLEST count whose per-epoch take fits
            # the unique pool (take <= U).
            row["suggested_num_train_epochs"] = max(
                1, math.ceil(target / row["unique_est_tokens"] - 1e-9)
            )
        infeasible = [
            row for row in rows if row["max_source_tokens"] > row["unique_est_tokens"]
        ]
        if infeasible:
            rendered = "\n".join(
                f"  {row['arm']}: U={row['unique_est_tokens']:,} epochs={row['effective_epochs']} "
                f"needs per-epoch take {row['max_source_tokens']:,} > U "
                f"(suggested epochs={row['suggested_num_train_epochs']})"
                for row in infeasible
            )
            raise RuntimeError(
                "ARM CAPACITY GUARD FAILED: per-epoch take exceeds the unique pool "
                f"(presented target {target:,} estimated tokens). Re-run with "
                "--epochs_override per the suggestions ('<recipe>_repeat' keys the "
                "repeat variant) or lower --presented_target_est:\n" + rendered
            )
        tight = [
            row["arm"]
            for row in rows
            if row["unique_est_tokens"] * row["effective_epochs"] < target * PARITY_SLACK
        ]
        if tight:
            logger.warning(
                "ARM CAPACITY tight (<2%% headroom over target %d): %s — capacity "
                "== target, exact by take arithmetic but with zero estimator margin",
                target,
                tight,
            )
        realized_floor = min(row["presented_est_tokens"] for row in rows)
        realized_top = max(row["presented_est_tokens"] for row in rows)
        table = {
            "presented_per_estimated": PRESENTED_PER_ESTIMATED,
            "parity_slack": PARITY_SLACK,
            "parity_mode": "truncation",
            "presented_target_est_tokens": target,
            "parity_floor_presented_est": realized_floor,
            "parity_max_presented_est": realized_top,
            "parity_ok": True,
            "capacity_tight_arms": tight,
            "arms": rows,
            "skipped_arms": [
                arm_name for arm_name, plan_key, _ in MEDIUM_ARM_TABLE if plan_key not in results
            ],
            "notes": [
                "per-epoch take = ceil(target / epochs); MAX_SOURCE_TOKENS = take; the trainer "
                "prefix-takes the frozen order ONCE and repeats it every epoch, so presented "
                "parity across arms is take x epochs (~target within one example per epoch)",
                "med_dsingle_alt and med_dsingle_joint share d_single.order.json (Gate-3 re-check pair)",
                "capacity guard (hard): per-epoch take <= unique pool per arm; "
                "capacity_tight_arms lists arms with <2% headroom over the target",
            ],
        }
        return table
    for row in rows:
        row["suggested_num_train_epochs"] = max(
            1, math.ceil(top * PARITY_SLACK / (row["unique_est_tokens"] * PRESENTED_PER_ESTIMATED) - 1e-9)
        )
    parity_ok = top <= floor * PARITY_SLACK
    table = {
        "presented_per_estimated": PRESENTED_PER_ESTIMATED,
        "parity_slack": PARITY_SLACK,
        "parity_floor_presented_est": floor,
        "parity_max_presented_est": top,
        "parity_ok": parity_ok,
        "arms": rows,
        "skipped_arms": [
            arm_name for arm_name, plan_key, _ in MEDIUM_ARM_TABLE if plan_key not in results
        ],
        "notes": [
            "presented = unique_est x epochs x 0.392 (small-arm measured presented/estimated "
            "coefficient, identical for joint and alternate under per-side caps)",
            "med_dsingle_alt and med_dsingle_joint share d_single.order.json (Gate-3 re-check pair)",
            "suggested_num_train_epochs levels every arm up to the strongest arm's presented "
            "total with 2% headroom",
        ],
    }
    if not parity_ok:
        rendered = "\n".join(
            f"  {row['arm']}: U={row['unique_est_tokens']:,} epochs={row['effective_epochs']} "
            f"presented~{row['presented_est_tokens'] / 1e6:.2f}M "
            f"(suggested epochs={row['suggested_num_train_epochs']})"
            for row in rows
        )
        raise RuntimeError(
            "ARM PARITY GUARD FAILED: the fixed medium arms disagree on presented tokens by more "
            f"than {int((PARITY_SLACK - 1) * 100)}% (floor={floor / 1e6:.2f}M, max={top / 1e6:.2f}M). "
            "Adjust --epochs_override per the suggestions and re-run the planner:\n" + rendered
        )
    return table


# ---------------------------------------------------------------------------
# Source scanning (lazy heavy imports; stratified per subset, bounded by
# per-subset token caps).
# ---------------------------------------------------------------------------


# Traces substrata (OPT-IN split for the H200 main arm): the traces pool mixes
# AppWorld and tau2-bench sessions in one parquet pool; the trainer threads the
# parquet ``benchmark``/``subset`` column onto ``JointExample.subset``
# (train_data_joint.py), which is what the classification below consumes.
# Default table: anything containing "appworld" -> appworld; airline/retail/
# telecom (incl. tau2_*/tau_* prefixed variants, via substring match) -> tau2.
# ``--traces_subset_map`` REPLACES this table; unmatched values keep their RAW
# subset name as their own stratum (never silently dropped).
_TRACES_DEFAULT_SUBSET_MAP: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("appworld", ("appworld",)),
    ("tau2", ("airline", "retail", "telecom")),
)
# Catch-all scan stratum for unmapped subsets (reserved name).  It is scanned
# FIRST so an empty catch-all hands its cap to the declared strata through
# sequential water-filling (a trailing empty stratum would strand the budget).
_TRACES_OTHER_SUBSET = "other"


def _classify_traces_subset(raw_subset: str, subset_map=None) -> str:
    """Map a traces pool ``benchmark``/``subset`` value to its scan stratum.

    Case-insensitive SUBSTRING match against the map's patterns in declared
    order; ``subset_map=None`` applies ``_TRACES_DEFAULT_SUBSET_MAP``.
    Unmatched values return their raw name (""/None -> "unknown").
    """
    raw = str(raw_subset or "unknown")
    lowered = raw.lower()
    for stratum, patterns in (subset_map or _TRACES_DEFAULT_SUBSET_MAP):
        if any(pattern.lower() in lowered for pattern in patterns):
            return stratum
    return raw


def _parse_traces_subset_map(specs: Optional[Sequence[str]]):
    """Parse repeatable ``--traces_subset_map stratum=subset[:subset...][,...]``.

    Returns an ordered tuple of (stratum, patterns) or None when no override
    was given (the default table then applies).  "other" is reserved for the
    catch-all stratum.
    """
    if not specs:
        return None
    mapping: List[Tuple[str, Tuple[str, ...]]] = []
    seen: Set[str] = set()
    for spec in specs:
        for entry in spec.split(","):
            stratum, sep, value = entry.partition("=")
            if not sep or not stratum.strip() or not value.strip():
                raise ValueError(
                    f"--traces_subset_map must be stratum=subset[:subset...], got: {entry!r}"
                )
            stratum = stratum.strip()
            patterns = tuple(part.strip() for part in value.split(":") if part.strip())
            if not patterns:
                raise ValueError(f"--traces_subset_map stratum needs >= 1 subset, got: {entry!r}")
            if stratum == _TRACES_OTHER_SUBSET:
                raise ValueError(
                    f"--traces_subset_map stratum name {_TRACES_OTHER_SUBSET!r} is reserved "
                    "(it is the catch-all scan stratum)"
                )
            if stratum in seen:
                raise ValueError(f"--traces_subset_map stratum repeated: {stratum!r}")
            seen.add(stratum)
            mapping.append((stratum, patterns))
    return tuple(mapping)


def _traces_source_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Trainer-matching knobs for the traces source (shared by every mode)."""
    return dict(
        path=args.traces_path,
        split="train",
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=4,  # JointDataArgs default: MUST match the trainer run
        # MUST match the trainer run's REQUIRE_TOOL_CALL / ACTION_TOOL_CALL_FRAC
        # (G-H200 main arm: False / 0.75): the planner pool has to equal the
        # trainer pool, otherwise --example_order_file membership filtering
        # silently drops every non-tool-call target (or hard-errors on
        # missing qids when the per-session picks diverge).  getattr with the
        # JointDataArgs defaults keeps fabricated/test namespaces valid.
        require_tool_call=getattr(args, "require_tool_call", True),
        action_tool_call_frac=getattr(args, "action_tool_call_frac", 0.75),
    )


class _LazyTracesBase:
    """Construct ``AgentLLMTracesJointSource`` on FIRST iteration, not before.

    The traces source loads its whole pool eagerly at construction, and the
    ``--subset_weights`` fail-fast validation instantiates
    ``_family_subsources`` once ahead of the real scan — a lazy holder keeps
    that validation pass free and guarantees the pool loads at most once per
    scan, shared by all stratum views.
    """

    def __init__(self, source_kwargs: Dict[str, Any]) -> None:
        self._source_kwargs = source_kwargs
        self._source = None

    def __iter__(self):
        if self._source is None:
            from train.train_data_joint import AgentLLMTracesJointSource

            self._source = AgentLLMTracesJointSource(**self._source_kwargs)
        return iter(self._source)


class _TracesSubsetView:
    """Filtered view over a SHARED traces base source (split mode).

    Iterating a view is an in-memory filter on the trainer-threaded
    ``JointExample.subset`` (the parquet ``benchmark``/``subset`` column): it
    yields exactly the examples classifying to its stratum.  The catch-all
    ``other`` view yields the unmapped examples and tallies their raw subset
    values in ``unknown_subsets`` for the loud scan-report warning.
    ``subset_classifier`` is consumed by ``_scan_subset_prefix`` so pool
    entries are tagged with their classified stratum — for ``other`` that is
    the RAW subset value (unknown subsets surface as their own strata in the
    per-subset audit, never silently dropped).
    """

    def __init__(self, base: Any, subset_map, stratum: str) -> None:
        self._base = base
        self._stratum = stratum
        self._known = frozenset(name for name, _ in (subset_map or _TRACES_DEFAULT_SUBSET_MAP))
        self.subset_classifier: Callable[[str], str] = (
            lambda raw: _classify_traces_subset(raw, subset_map)
        )
        self.unknown_subsets: Counter = Counter()

    def __iter__(self):
        for example in self._base:
            classified = self.subset_classifier(getattr(example, "subset", "unknown"))
            if self._stratum == _TRACES_OTHER_SUBSET:
                if classified in self._known:
                    continue
                self.unknown_subsets[classified] += 1
            elif classified != self._stratum:
                continue
            yield example


def _traces_split_subsources(base: Any, subset_map) -> List[Tuple[str, Any]]:
    """One filtered view per stratum over ONE shared traces base source.

    Declared strata follow the map's order (default: appworld, tau2); the
    catch-all ``other`` goes FIRST so an empty one redistributes its cap to
    the declared strata (sequential water-filling only flows forward).
    """
    declared = [name for name, _ in (subset_map or _TRACES_DEFAULT_SUBSET_MAP)]
    return [
        (_TRACES_OTHER_SUBSET, _TracesSubsetView(base, subset_map, _TRACES_OTHER_SUBSET)),
        *[(name, _TracesSubsetView(base, subset_map, name)) for name in declared],
    ]


def _family_subsources(family: str, args: argparse.Namespace) -> List[Tuple[str, Any]]:
    """Instantiate the per-subset sources of one family (trainer-matching knobs).

    P0-2: scanning a family subset-by-subset (with per-subset caps) guarantees
    every subset reaches the pool — the old single-stream scan truncated at
    ``quota x oversample`` in source iteration order, so the tail subsets
    (longmagpie after hotpotqa/2wiki; alphabetically-late Open-SWE configs)
    had ZERO sampling probability whenever the cap hit early.

    Each subset source gets ``file_order_seed = f"{order_seed}:scan:{family}:
    {subset}"``: within a subset the parquet/jsonl FILE order is shuffled
    (row order inside a file is unchanged), so a cap-truncated subset pool is
    a seeded-random file prefix, not the alphabetical head.
    """
    if family == "traces":
        if not args.traces_path:
            raise ValueError("a recipe needs the traces family but --traces_path is not set")
        subset_map = _parse_traces_subset_map(getattr(args, "traces_subset_map", None))
        if not getattr(args, "split_traces_subsets", False) and subset_map is None:
            # Legacy single-stratum scan: identical knobs to pre-split runs, so
            # invocations without split options are bit-identical.
            from train.train_data_joint import AgentLLMTracesJointSource

            return [("traces", AgentLLMTracesJointSource(**_traces_source_kwargs(args)))]
        # Opt-in split (g_h200_main): appworld/tau2 (+ an "other" catch-all)
        # become independently weightable scan strata, addressable as
        # ``--subset_weights traces:appworld=w / traces:tau2=w / traces:other=w``.
        return _traces_split_subsources(_LazyTracesBase(_traces_source_kwargs(args)), subset_map)
    from train.train_data_joint_multisource import (
        OpenSWEJointSource,
        QADocsJointSource,
        ToucanJointSource,
    )

    common: Dict[str, Any] = dict(
        split="train",
        split_seed=args.split_seed,
        # MUST match the trainer run (see _traces_source_kwargs): with
        # require_tool_call=False the pools carry clarification/no-call/final
        # targets too, and the order file must list them.
        require_tool_call=getattr(args, "require_tool_call", True),
    )

    def _seed(subset: str) -> str:
        return f"{args.order_seed}:scan:{family}:{subset}"

    if family == "toucan":
        if not args.toucan_path:
            raise ValueError("a recipe needs the toucan family but --toucan_path is not set")
        return [("toucan", ToucanJointSource(args.toucan_path, file_order_seed=_seed("toucan"), **common))]
    if family == "openswe":
        if not args.openswe_path:
            raise ValueError("a recipe needs the openswe family but --openswe_path is not set")
        # One subset per trajectory-config dir (data/<config>/*.parquet); the
        # config dir name is both the stratum name and the examples' subset
        # suffix (``openswe:<config>``).  A layout without config subdirs
        # degenerates to a single subset over the whole root.
        root = Path(args.openswe_path)
        if root.is_file():
            return [("all", OpenSWEJointSource(str(root), file_order_seed=_seed("all"), **common))]
        data_root = root / "data" if (root / "data").is_dir() else root
        config_dirs = sorted(
            d for d in data_root.iterdir() if d.is_dir() and list(d.glob("*.parquet"))
        )
        if not config_dirs:
            return [("all", OpenSWEJointSource(str(root), file_order_seed=_seed("all"), **common))]
        return [
            (d.name, OpenSWEJointSource(str(d), file_order_seed=_seed(d.name), **common))
            for d in config_dirs
        ]
    if family == "qa":
        if not any([args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path]):
            raise ValueError("a recipe needs the qa family but no --qa_* path is set")
        subs: List[Tuple[str, Any]] = []
        for subset, path in (
            ("hotpotqa", args.qa_hotpotqa_path),
            ("2wiki", args.qa_2wiki_path),
            ("longmagpie", args.qa_longmagpie_path),
        ):
            if not path:
                continue
            kwargs: Dict[str, Any] = {
                "hotpotqa_path": None,
                "wiki2_path": None,
                "longmagpie_path": None,
                "file_order_seed": _seed(subset),
            }
            kwargs[{"hotpotqa": "hotpotqa_path", "2wiki": "wiki2_path", "longmagpie": "longmagpie_path"}[subset]] = path
            subs.append((subset, QADocsJointSource(**kwargs, **common)))
        return subs
    raise ValueError(f"unknown family: {family!r}")


def _scan_subset_prefix(
    source: Any,
    tokenizer,
    stamp: str,
    cache: Dict[Tuple[str, str], int],
    token_cap: Optional[float],
    seen: Set[str],
    family: str,
    subset: str,
    removal_identifiers: FrozenSet[str] = frozenset(),
    subset_classifier: Optional[Callable[[str], str]] = None,
) -> Tuple[List[PoolEntry], int, int, int]:
    """Stream one subset source into pool entries, stopping at ``token_cap``.

    The crossing example is included; ``token_cap=None`` scans the whole
    source.  Removal-listed examples are dropped BEFORE estimation and cap
    accounting (P2: filtering after the cap would let removed rows eat the
    oversample headroom and shrink the pool below ``quota x oversample``).
    Estimates are cached by (qid, stamp) and ``cache`` is updated in place.
    ``subset_classifier`` (traces split only): maps the example's own
    trainer-threaded ``subset`` field to its pool stratum instead of tagging
    every entry with the scan stratum name — the ``other`` catch-all thus
    records each unknown subset under its raw name.
    Returns (entries, total_estimated_tokens, cache_hits, removed_count).
    """
    entries: List[PoolEntry] = []
    total = 0
    cache_hits = 0
    removed = 0
    for example in source:
        if removal_identifiers and (
            _removal_match_keys(example.qid, example.session_id) & removal_identifiers
        ):
            removed += 1
            continue
        key = (example.qid, stamp)
        estimated = cache.get(key)
        if estimated is None:
            estimated = estimate_source_tokens(example, tokenizer)
            cache[key] = estimated
        else:
            cache_hits += 1
        if example.qid in seen:
            raise RuntimeError(f"duplicate qid from the {family} source: {example.qid}")
        seen.add(example.qid)
        entries.append(
            PoolEntry(
                qid=example.qid,
                session_id=example.session_id,
                estimated_tokens=estimated,
                subset=(
                    subset_classifier(str(getattr(example, "subset", "unknown") or "unknown"))
                    if subset_classifier is not None
                    else subset
                ),
            )
        )
        total += estimated
        if token_cap is not None and total >= token_cap:
            break
    return entries, total, cache_hits, removed


def scan_family_pool(
    family: str,
    args: argparse.Namespace,
    tokenizer,
    stamp: str,
    cache: Dict[Tuple[str, str], int],
    token_cap: Optional[float],
    subset_weights: Optional[Dict[str, float]] = None,
    removal_identifiers: FrozenSet[str] = frozenset(),
) -> Tuple[List[PoolEntry], Dict[str, Any]]:
    """Stratified scan of one family's pool (P0-2).

    The family cap is split over the subsets by weight (default: equal;
    ``--subset_weights family:subset=w`` overrides) and each subset is scanned
    up to its cap.  A subset that EXHAUSTS below its cap hands the unused
    budget to its not-yet-scanned siblings (sequential water-filling in
    declared subset order), so a small subset can never starve the family cap
    nor inflate its own share.  Removal-listed examples are filtered DURING
    the scan (before cap accounting; P2).  Returns the concatenated pool
    (subset scan order; quota sampling shuffles anyway) and a per-subset
    audit report including removal counts.
    """
    subsources = _family_subsources(family, args)
    weights: Dict[str, float] = {}
    skipped_zero: List[str] = []
    for name, _ in subsources:
        key = f"{family}:{name}"
        weight = (subset_weights or {}).get(key, 1.0)
        if weight < 0:
            raise ValueError(f"--subset_weights entry {key} must be >= 0, got {weight}")
        if weight == 0:
            # Explicit opt-out (e.g. traces:other=0 keeps swebench/browsecompplus
            # out of the g_h200_main pool): the stratum is not scanned at all
            # and takes no share of the family cap.
            skipped_zero.append(name)
            continue
        weights[name] = weight
    if skipped_zero:
        subsources = [(name, source) for name, source in subsources if name in weights]
        logger.warning(
            "family=%s: strata SKIPPED via zero --subset_weights: %s", family, skipped_zero
        )
        if not subsources and token_cap:
            raise ValueError(
                f"family={family}: every stratum was skipped by zero --subset_weights "
                f"but the family cap is {token_cap}"
            )
    pool: List[PoolEntry] = []
    seen: Set[str] = set()
    subsets_report: Dict[str, Any] = {}
    remaining_cap = token_cap
    remaining_weight = sum(weights.values())
    for name, source in subsources:
        sub_cap: Optional[float] = None
        if remaining_cap is not None:
            sub_cap = remaining_cap * weights[name] / remaining_weight if remaining_weight else 0.0
        entries, sub_total, cache_hits, removed = _scan_subset_prefix(
            source,
            tokenizer,
            stamp,
            cache,
            sub_cap,
            seen,
            family,
            name,
            removal_identifiers,
            subset_classifier=getattr(source, "subset_classifier", None),
        )
        pool.extend(entries)
        subsets_report[name] = {
            "examples": len(entries),
            "estimated_tokens": sub_total,
            "cap_estimated_tokens": int(math.ceil(sub_cap)) if sub_cap is not None else None,
            "weight": weights[name],
            "exhausted": sub_cap is None or sub_total < sub_cap,
            "cache_hits": cache_hits,
            "removed": removed,
        }
        unknown_subsets = getattr(source, "unknown_subsets", None)
        if unknown_subsets:
            # Traces split catch-all: unmapped subsets are KEPT (under their
            # raw names on the pool entries) — surface them loudly so the
            # operator can pin them with --traces_subset_map.
            subsets_report[name]["unknown_subsets"] = dict(unknown_subsets)
            logger.warning(
                "family=%s subset=%s: %d examples carry UNMAPPED raw subset names %s — kept "
                "under their raw names in the %r catch-all stratum; re-run with "
                "--traces_subset_map to weight them explicitly",
                family,
                name,
                sum(unknown_subsets.values()),
                sorted(unknown_subsets),
                name,
            )
        logger.info(
            "family=%s subset=%s: scanned %d examples (%d estimated tokens, cap=%s, "
            "cache_hits=%d, removed=%d)",
            family,
            name,
            len(entries),
            sub_total,
            int(sub_cap) if sub_cap is not None else "none",
            cache_hits,
            removed,
        )
        if remaining_cap is not None:
            remaining_cap -= sub_total
            remaining_weight -= weights[name]
            if remaining_cap <= 0:
                break
    report = {
        "subsets": subsets_report,
        "cap_estimated_tokens": int(token_cap) if token_cap is not None else None,
        "removed_total": sum(sub["removed"] for sub in subsets_report.values()),
        "skipped_zero_weight": skipped_zero,
    }
    return pool, report


def _parse_subset_weights(specs: Optional[Sequence[str]]) -> Dict[str, float]:
    """Parse repeatable ``--subset_weights family:subset=w`` (w >= 0; 0 skips the stratum)."""
    weights: Dict[str, float] = {}
    for spec in specs or []:
        key, sep, value = spec.partition("=")
        family, sep2, subset = key.partition(":")
        if not sep or not sep2 or not family.strip() or not subset.strip():
            raise ValueError(f"--subset_weights must be family:subset=weight, got: {spec!r}")
        if family.strip() not in FAMILIES:
            raise ValueError(f"--subset_weights unknown family in {spec!r} (choose from {FAMILIES})")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"--subset_weights weight must be >= 0 (0 skips the stratum), got: {spec!r}")
        weights[f"{family.strip()}:{subset.strip()}"] = weight
    return weights


def _list_traces_subsets(
    source: Any,
    tokenizer,
    stamp: str,
    cache: Dict[Tuple[str, str], int],
    removal_identifiers: FrozenSet[str] = frozenset(),
    subset_map=None,
) -> List[Dict[str, Any]]:
    """Inventory of the traces pool's raw ``benchmark``/``subset`` values.

    The ``--list_traces_subsets`` dry-run core: iterate the SAME source the
    planning scan uses (same knobs, same removal filtering, no cap — every
    record is counted) through the SAME token cache, so the pass also warms
    ``tokencache_traces.jsonl`` for the planning run.  Each row reports the
    raw subset value, its record/estimated-token counts, and the stratum it
    lands in under the active map.  Sorted by estimated tokens desc.
    """
    entries, _, _, _ = _scan_subset_prefix(
        source,
        tokenizer,
        stamp,
        cache,
        None,
        set(),
        "traces",
        "",
        removal_identifiers,
        subset_classifier=lambda raw: raw,
    )
    tally: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        row = tally.setdefault(entry.subset, {"examples": 0, "estimated_tokens": 0})
        row["examples"] += 1
        row["estimated_tokens"] += entry.estimated_tokens
    return [
        {
            "subset": name,
            "stratum": _classify_traces_subset(name, subset_map),
            "examples": row["examples"],
            "estimated_tokens": row["estimated_tokens"],
        }
        for name, row in sorted(
            tally.items(), key=lambda item: (-item[1]["estimated_tokens"], item[0])
        )
    ]


# ---------------------------------------------------------------------------
# Output writing + ETA.
# ---------------------------------------------------------------------------


def _arm_table_lines(table: Dict[str, Any]) -> List[str]:
    """Render the P1-8 arm launch table for stdout."""
    lines = [
        "ARM LAUNCH TABLE (P1-8 parity: presented = unique_est x epochs x "
        f"{PRESENTED_PER_ESTIMATED}; arms must agree within "
        f"{int((table['parity_slack'] - 1) * 100)}%)",
        "  arm                  plan_key       variant  doc_mode   unique_est_M  epochs  presented_M  suggested_epochs  max_source_tokens  card",
    ]
    for row in table["arms"]:
        lines.append(
            f"  {row['arm']:<20}  {row['plan_key']:<14} {row['variant']:<8} {row['doc_mode']:<10}"
            f" {row['unique_est_tokens'] / 1e6:>12.2f} {row['effective_epochs']:>6}"
            f" {row['presented_est_tokens'] / 1e6:>11.2f} {row['suggested_num_train_epochs']:>16}"
            f" {row['max_source_tokens']:>17} {row['suggested_card']:>5}"
        )
    if table["skipped_arms"]:
        lines.append(f"  (not planned in this run: {', '.join(table['skipped_arms'])})")
    lines.append(
        f"  parity: floor={table['parity_floor_presented_est'] / 1e6:.2f}M "
        f"max={table['parity_max_presented_est'] / 1e6:.2f}M ok={table['parity_ok']}"
    )
    return lines


def write_outputs(results: Dict[str, Dict[str, Any]], out_dir: Path) -> List[str]:
    """Write ``<recipe>.order.json`` (bare qid list) and ``<recipe>.plan.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, result in results.items():
        order_path = out_dir / f"{name}.order.json"
        order_path.write_text(
            json.dumps(result["order"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plan_path = out_dir / f"{name}.plan.json"
        plan_path.write_text(
            json.dumps(result["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.extend([str(order_path), str(plan_path)])
        logger.info("wrote %s and %s", order_path, plan_path)
    return written


def _eta_line(name: str, presented_tokens: int, small_arm_hours: float) -> str:
    finish = datetime.now() + timedelta(
        hours=(presented_tokens / _TOKENS_PER_ETA_UNIT) * small_arm_hours
    )
    # Both numbers are ESTIMATED source tokens (the same estimator the
    # trainer's --max_source_tokens uses); the small-arm calibration is
    # measured against that same nominal unit.  Say so explicitly.
    return (
        f"{name}: 预计完成 = {finish:%Y-%m-%d %H:%M} "
        f"(estimated≈{presented_tokens / 1e6:.1f}M tokens, estimator 口径; "
        f"32M estimated ≈ {small_arm_hours}h from the small arm)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen mixture/order plans (order files + audit plans) for the G-medium true-joint arms."
    )
    parser.add_argument("--traces_path", default=None)
    parser.add_argument("--toucan_path", default=None)
    parser.add_argument("--openswe_path", default=None)
    parser.add_argument("--qa_hotpotqa_path", default=None)
    parser.add_argument("--qa_2wiki_path", default=None)
    parser.add_argument("--qa_longmagpie_path", default=None)
    parser.add_argument("--split_manifest_file", default=None, help="passed through to the traces source")
    parser.add_argument("--split_manifest_name", default="subset_disjoint", help="passed through to the traces source")
    parser.add_argument("--split_seed", type=int, default=42, help="data-split/rendering seed for all sources (JointDataArgs default)")
    parser.add_argument(
        "--require_tool_call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pool-scan knob that MUST match the trainer run's REQUIRE_TOOL_CALL "
        "(default True keeps the legacy bit-identical pools). The G-H200 main arm trains "
        "with REQUIRE_TOOL_CALL=False: scan with --no-require_tool_call so "
        "clarification/no-call/final targets also reach the order file",
    )
    parser.add_argument(
        "--action_tool_call_frac",
        type=float,
        default=0.75,
        help="target tool-call share of the per-session stratified pick in the traces "
        "source when --no-require_tool_call (JointDataArgs default 0.75; ignored when "
        "--require_tool_call is on)",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        default=None,
        metavar="name=family:share,...",
        help=f"repeatable; families from {FAMILIES}; shares sum to 1.0 (required unless --list_traces_subsets)",
    )
    parser.add_argument("--budget_estimated_tokens", type=int, default=None, help="N: per-recipe total estimated source tokens (required unless --list_traces_subsets)")
    parser.add_argument("--oversample_factor", type=float, default=1.25, help="scan cap = quota x this factor")
    parser.add_argument("--repeat_unique_tokens", type=int, default=None, help="M: also emit <recipe>_repeat variants with ~M unique tokens per family pool")
    parser.add_argument("--epochs_override", action="append", default=None, metavar="name=n", help="repeatable audit record: recipe name -> train epochs (integer >= 1); scales presented_estimated_tokens in the plan jsons; applies to BOTH the base and the repeat variant of the named recipe")
    parser.add_argument("--presented_target_est", type=int, default=None, help="medium arm parity target: MAX_SOURCE_TOKENS per arm (estimated source tokens, trainer truncates the frozen order mid-epoch if needed); epochs then only need to supply target x 1.02 capacity per arm (the pre-registered 'same total training tokens' semantics). Default None keeps the full-pool-epochs parity guard")
    parser.add_argument("--removal_files", nargs="*", default=[], help="dedup removal lists (dicts with removal_list, or bare JSON string lists); applied DURING the pool scan (before cap accounting, P2)")
    parser.add_argument(
        "--subset_weights",
        action="append",
        default=None,
        metavar="family:subset=w",
        help="repeatable per-stratum scan-cap weight override (P0-2; default: equal weights "
        "across a family's subsets, e.g. qa splits its cap evenly over hotpotqa/2wiki/longmagpie; "
        "with --split_traces_subsets the traces strata are traces:appworld / traces:tau2 / "
        "traces:other). w=0 skips the stratum entirely (e.g. traces:other=0 keeps "
        "swebench/browsecompplus out of the g_h200_main pool)",
    )
    parser.add_argument(
        "--split_traces_subsets",
        action="store_true",
        help="scan the traces pool as independently-weightable substrata instead of one stratum: "
        "the parquet benchmark/subset column is classified *appworld* -> appworld, "
        "airline/retail/telecom (incl. tau2_*/tau_* variants) -> tau2, anything else keeps its "
        "raw name inside the 'other' catch-all stratum (scanned first, so an empty catch-all "
        "hands its cap to the declared strata). Default off: without split options the traces "
        "scan is the legacy single stratum, bit-identical to pre-split runs",
    )
    parser.add_argument(
        "--traces_subset_map",
        action="append",
        default=None,
        metavar="stratum=subset[:subset...][,...]",
        help="repeatable explicit classification table for the traces split (case-insensitive "
        "substring match in declared order); REPLACES the default table and implies "
        "--split_traces_subsets. Run --list_traces_subsets first to see the raw values, e.g. "
        "--traces_subset_map appworld=appworld,tau2=airline:retail:telecom",
    )
    parser.add_argument(
        "--list_traces_subsets",
        action="store_true",
        help="dry-run (no --recipe/--budget_estimated_tokens needed): scan the traces pool with "
        "the same source knobs and removal lists as planning, print the observed "
        "benchmark/subset values with record/estimated-token counts and their stratum under the "
        "active map, then exit; warms tokencache_traces.jsonl for the planning run",
    )
    parser.add_argument("--order_seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer", default=None, help="local HF tokenizer path; default: whitespace fake")
    parser.add_argument("--small_arm_hours", type=float, default=None, help="measured wall-clock hours per 32M ESTIMATED source tokens (estimator 口径 — the SAME unit the budget/quota math uses; presented tokens are ~0.392x) on the small JOINT-mode arm (enables the ETA line)")
    parser.add_argument("--small_arm_hours_alternate", type=float, default=None, help="same unit as --small_arm_hours but measured on an ALTERNATE-mode arm; default: reuse --small_arm_hours (the two modes have different throughputs — joint 0.556M vs alternate 0.453M presented/h on the small arms)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if args.list_traces_subsets:
        # Dry-run inventory (前置扫描): confirm the real benchmark/subset
        # strings before pinning a --traces_subset_map.  No planning.
        if not args.traces_path:
            raise ValueError("--list_traces_subsets requires --traces_path")
        tokenizer = _load_tokenizer(args.tokenizer)
        stamp = _cache_stamp(
            str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
            args.split_seed,
        )
        removal_identifiers, _ = load_removal_identifiers(args.removal_files)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "tokencache_traces.jsonl"
        cache = _load_token_cache(cache_path)
        cache_size = len(cache)
        from train.train_data_joint import AgentLLMTracesJointSource  # lazy heavy import

        subset_map = _parse_traces_subset_map(args.traces_subset_map)
        rows = _list_traces_subsets(
            AgentLLMTracesJointSource(**_traces_source_kwargs(args)),
            tokenizer,
            stamp,
            cache,
            removal_identifiers,
            subset_map,
        )
        if len(cache) != cache_size:
            _write_token_cache(cache_path, cache)
        print(
            "TRACES SUBSETS (train split, removal lists applied; stratum = classification "
            "under the active map):"
        )
        print(f"  {'subset':<24} {'stratum':<24} {'examples':>9} {'estimated_tokens':>17}")
        total_examples = 0
        total_tokens = 0
        for row in rows:
            total_examples += row["examples"]
            total_tokens += row["estimated_tokens"]
            print(
                f"  {row['subset']:<24} {row['stratum']:<24} "
                f"{row['examples']:>9} {row['estimated_tokens']:>17}"
            )
        print(f"  {'TOTAL':<24} {'':<24} {total_examples:>9} {total_tokens:>17}")
        return

    if not args.recipe:
        raise ValueError("--recipe is required (unless --list_traces_subsets)")
    if args.budget_estimated_tokens is None:
        raise ValueError("--budget_estimated_tokens is required (unless --list_traces_subsets)")
    recipes = [parse_recipe(spec) for spec in args.recipe]
    names = [recipe.name for recipe in recipes]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate --recipe names: {sorted(names)}")
    epochs_overrides = _parse_epochs_overrides(args.epochs_override, set(names))
    if not any([
        args.traces_path,
        args.toucan_path,
        args.openswe_path,
        args.qa_hotpotqa_path,
        args.qa_2wiki_path,
        args.qa_longmagpie_path,
    ]):
        raise ValueError("at least one source path is required")
    if args.oversample_factor < 1.0:
        raise ValueError(f"--oversample_factor must be >= 1.0, got {args.oversample_factor}")

    tokenizer = _load_tokenizer(args.tokenizer)
    stamp = _cache_stamp(
        str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
        args.split_seed,
    )
    removal_identifiers, removal_file_counts = load_removal_identifiers(args.removal_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scan each needed family once, up to the largest quota any recipe asks of
    # it (base budget and repeat pool both), times the oversample factor.
    caps: Dict[str, float] = {}
    for recipe in recipes:
        for family, share in recipe.shares:
            cap = share * args.budget_estimated_tokens
            if args.repeat_unique_tokens:
                cap = max(cap, share * args.repeat_unique_tokens)
            caps[family] = max(caps.get(family, 0.0), cap * args.oversample_factor)

    pools: Dict[str, List[PoolEntry]] = {}
    pool_scan_reports: Dict[str, Any] = {}
    subset_weights = _parse_subset_weights(args.subset_weights)
    if subset_weights:
        # Fail fast (before the expensive scans): every weight key must name a
        # family:subset stratum that some recipe actually scans.
        valid_keys = sorted(
            f"{family}:{name}"
            for family in sorted(caps)
            for name, _ in _family_subsources(family, args)
        )
        unknown = sorted(key for key in subset_weights if key not in set(valid_keys))
        if unknown:
            raise ValueError(
                f"--subset_weights entries matched no scanned family:subset: {unknown} "
                f"(scanned: {valid_keys})"
            )
    for family in sorted(caps):
        cache_path = out_dir / f"tokencache_{family}.jsonl"
        cache = _load_token_cache(cache_path)
        cache_size = len(cache)
        pool, scan_report = scan_family_pool(
            family,
            args,
            tokenizer,
            stamp,
            cache,
            caps[family],
            subset_weights=subset_weights,
            removal_identifiers=removal_identifiers,
        )
        pools[family] = pool
        pool_scan_reports[family] = scan_report
        if len(cache) != cache_size:
            _write_token_cache(cache_path, cache)

    tokenizer_tag = str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))
    provenance = {
        "tokenizer": tokenizer_tag,
        "estimate_stamp": stamp,
        "split_seed": args.split_seed,
        "oversample_factor": args.oversample_factor,
        "family_scan_caps_estimated_tokens": {
            family: int(math.ceil(cap)) for family, cap in sorted(caps.items())
        },
        "subset_weights": subset_weights,
        "repeat_unique_tokens": args.repeat_unique_tokens,
    }
    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=args.budget_estimated_tokens,
        order_seed=args.order_seed,
        repeat_unique_tokens=args.repeat_unique_tokens,
        removal_identifiers=removal_identifiers,
        removal_file_counts=removal_file_counts,
        small_arm_hours=args.small_arm_hours,
        small_arm_hours_alternate=args.small_arm_hours_alternate,
        epochs_overrides=epochs_overrides,
        scan_removals={
            family: report["removed_total"] for family, report in pool_scan_reports.items()
        },
        provenance=provenance,
        presented_target_est=args.presented_target_est,
    )
    # P0-2 audit: how each recipe family's pool was scanned (per-subset caps,
    # weights, exhaustion) — the same scan underlies every recipe/variant.
    for result in results.values():
        result["plan"]["pool_scan"] = {
            family: pool_scan_reports[family]
            for family in pool_scan_reports
            if family in result["plan"]["families"]
        }
    write_outputs(results, out_dir)

    arm_table = results[next(iter(results))]["plan"].get("arm_launch_table") if results else None
    if arm_table is not None:
        for line in _arm_table_lines(arm_table):
            print(line)
        if args.small_arm_hours is not None:
            # Per-mode calibrations: joint and alternate arms have different
            # throughputs, so a single hours number would misestimate one side.
            hours_by_mode = {
                "joint": args.small_arm_hours,
                "alternate": (
                    args.small_arm_hours_alternate
                    if args.small_arm_hours_alternate is not None
                    else args.small_arm_hours
                ),
            }
            for row in arm_table["arms"]:
                consumed = row["unique_est_tokens"] * row["effective_epochs"]
                hours = consumed / _TOKENS_PER_ETA_UNIT * hours_by_mode[row["doc_mode"]]
                finish = datetime.now() + timedelta(hours=hours)
                print(
                    f"  ETA {row['arm']}: ~{hours:.1f}h "
                    f"({row['doc_mode']}-mode calibration; {consumed / 1e6:.1f}M estimated) "
                    f"-> {finish:%Y-%m-%d %H:%M}"
                )

    for name, result in results.items():
        plan = result["plan"]
        for family, report in plan["families"].items():
            if report["shortfall_estimated_tokens"] > 0:
                logger.warning(
                    "SHORTFALL %s family=%s: pool fills only %d of %d quota estimated tokens "
                    "(family share %.3f under-realized)",
                    name,
                    family,
                    report["estimated_tokens"],
                    report["quota_estimated_tokens"],
                    report["share"],
                )
        presented = plan.get("presented_estimated_tokens", plan.get("total_estimated_tokens", 0))
        epochs_note = ""
        if plan.get("epochs_override") is not None:
            epochs_note = f", epochs_override={plan['epochs_override']}"
        elif plan["variant"] == "repeat":
            epochs_note = f", recommended_epochs={plan['recommended_epochs']}"
        print(
            f"{name}: {plan['order_examples']} examples, "
            f"estimated≈{presented / 1e6:.1f}M tokens (estimator 口径)" + epochs_note
        )
        if args.small_arm_hours is not None:
            print(_eta_line(name, presented, args.small_arm_hours))


if __name__ == "__main__":
    main()
