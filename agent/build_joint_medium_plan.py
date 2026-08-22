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

Outputs per recipe (and per ``<recipe>_repeat`` variant when
``--repeat_unique_tokens`` is given):

- ``<out_dir>/<recipe>.order.json`` — bare JSON list of qids;
- ``<out_dir>/<recipe>.plan.json`` — audit trail: shares, per-family
  examples/estimated tokens/realized share, removals applied, oversample
  shortfalls (pool too small to fill a quota — also logged as warnings),
  repeat-variant ``recommended_epochs``, seeds.

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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    # Allow running as `python agent/build_joint_medium_plan.py` from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from train.train_data_joint import JointExample

logger = logging.getLogger(__name__)


FAMILIES = ("traces", "toucan", "openswe", "qa")
# ETA calibration unit: the small-arm runs measured wall-clock per 32M
# presented (estimated) source tokens; supplied via --small_arm_hours.
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


# ---------------------------------------------------------------------------
# Pools and the source-token estimator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolEntry:
    """One scanned example reduced to what planning needs (memory bound)."""

    qid: str
    session_id: str
    estimated_tokens: int


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
    Open-SWE removal carries the bare ``trajectory_id``; a QA removal carries
    the bare ``_id`` / row index.  Traces qids/session ids are bare already.
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
# Quota sampling and round-robin interleave (the pure planning core).
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


def interleave_families(family_qids: Sequence[Tuple[str, List[str]]]) -> List[str]:
    """Round-robin in the recipe's declared family order.

    One example per family per round until all are placed; shorter families
    simply stop contributing rounds.
    """
    out: List[str] = []
    depth = max((len(qids) for _, qids in family_qids), default=0)
    for index in range(depth):
        for _, qids in family_qids:
            if index < len(qids):
                out.append(qids[index])
    return out


def _fill_recipe(
    recipe: RecipeSpec,
    pools: Dict[str, List[PoolEntry]],
    token_scale: int,
    order_seed: int,
    seed_suffix: str,
) -> Tuple[List[str], Dict[str, Any], int]:
    """Sample each recipe family to ``share * token_scale`` and interleave."""
    family_qids: List[Tuple[str, List[str]]] = []
    reports: Dict[str, Any] = {}
    total = 0
    for family, share in recipe.shares:
        quota = share * token_scale
        pool = pools[family]
        taken, realized, shortfall = _sample_to_quota(
            pool, quota, random.Random(f"{order_seed}:{recipe.name}:{family}:{seed_suffix}")
        )
        total += realized
        reports[family] = {
            "share": share,
            "quota_estimated_tokens": int(math.ceil(quota)),
            "examples": len(taken),
            "estimated_tokens": realized,
            "pool_examples": len(pool),
            "pool_estimated_tokens": sum(entry.estimated_tokens for entry in pool),
            "shortfall_estimated_tokens": shortfall,
            "realized_share": None,  # filled in below
        }
        family_qids.append((family, [entry.qid for entry in taken]))
    for report in reports.values():
        report["realized_share"] = (report["estimated_tokens"] / total) if total else 0.0
    return interleave_families(family_qids), reports, total


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
) -> Dict[str, Dict[str, Any]]:
    """Fill per-recipe family quotas and interleave into frozen orders.

    Pure planning core: pools in, frozen orders + audit plans out — no IO, no
    tokenizer, no wall clock (deterministic for fixed inputs; the ETA line is
    printed by ``main`` only).  Returns ``{name: {"order": [...], "plan":
    {...}}}``, including ``<recipe>_repeat`` variants when
    ``repeat_unique_tokens`` is set.
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
        "by_family": removal_counts,
        "identifiers": len(removal_identifiers),
        "by_file": removal_file_counts or {},
    }
    eta_info = (
        {"small_arm_hours": small_arm_hours, "tokens_per_unit": _TOKENS_PER_ETA_UNIT}
        if small_arm_hours is not None
        else None
    )
    results: Dict[str, Dict[str, Any]] = {}
    for recipe in recipes:
        missing = [family for family, _ in recipe.shares if family not in filtered_pools]
        if missing:
            raise ValueError(
                f"recipe {recipe.name!r} needs families with no configured source path: {missing}"
            )
        order, family_reports, total_tokens = _fill_recipe(
            recipe, filtered_pools, budget_estimated_tokens, order_seed, seed_suffix="sample"
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
                "families": family_reports,
                "total_estimated_tokens": total_tokens,
                "order_examples": len(order),
                "removals": removals_report,
                "seeds": {"order_seed": order_seed},
                "eta": eta_info,
            },
        }
        if repeat_unique_tokens:
            # Repeat variant: each family's pool truncated to share * M
            # estimated tokens (seeded sample, shares preserved); the order
            # file contains each qid ONCE — repetition happens at train time
            # via ``recommended_epochs``, never via duplicate qids.
            rep_order, rep_reports, rep_total = _fill_recipe(
                recipe, filtered_pools, repeat_unique_tokens, order_seed, seed_suffix="repeat"
            )
            if len(rep_order) != len(set(rep_order)):
                raise RuntimeError(f"recipe {recipe.name}_repeat: duplicate qids in the planned order")
            recommended_epochs = math.ceil(budget_estimated_tokens / rep_total) if rep_total > 0 else 0
            results[f"{recipe.name}_repeat"] = {
                "order": rep_order,
                "plan": {
                    "recipe": recipe.name,
                    "variant": "repeat",
                    "shares": dict(recipe.shares),
                    "repeat_unique_tokens": repeat_unique_tokens,
                    "budget_estimated_tokens": budget_estimated_tokens,
                    "families": rep_reports,
                    "unique_pool_estimated_tokens": rep_total,
                    "recommended_epochs": recommended_epochs,
                    "presented_estimated_tokens": rep_total * recommended_epochs,
                    "order_examples": len(rep_order),
                    "order_note": "each qid appears once; repetition is achieved at train time via epochs",
                    "removals": removals_report,
                    "seeds": {"order_seed": order_seed},
                    "eta": eta_info,
                },
            }
    return results


# ---------------------------------------------------------------------------
# Source scanning (lazy heavy imports; bounded by per-family token caps).
# ---------------------------------------------------------------------------


def _family_source(family: str, args: argparse.Namespace):
    """Instantiate the source for one family with trainer-matching knobs."""
    if family == "traces":
        if not args.traces_path:
            raise ValueError("a recipe needs the traces family but --traces_path is not set")
        from train.train_data_joint import AgentLLMTracesJointSource

        return AgentLLMTracesJointSource(
            path=args.traces_path,
            split="train",
            split_seed=args.split_seed,
            split_manifest_file=args.split_manifest_file,
            split_manifest_name=args.split_manifest_name,
            max_samples_per_session=4,  # JointDataArgs default: MUST match the trainer run
            require_tool_call=True,  # JointDataArgs default
        )
    from train.train_data_joint_multisource import (
        OpenSWEJointSource,
        QADocsJointSource,
        ToucanJointSource,
    )

    common: Dict[str, Any] = dict(
        split="train",
        split_seed=args.split_seed,
        require_tool_call=True,  # JointDataArgs default
    )
    if family == "toucan":
        if not args.toucan_path:
            raise ValueError("a recipe needs the toucan family but --toucan_path is not set")
        return ToucanJointSource(args.toucan_path, **common)
    if family == "openswe":
        if not args.openswe_path:
            raise ValueError("a recipe needs the openswe family but --openswe_path is not set")
        return OpenSWEJointSource(args.openswe_path, **common)
    if family == "qa":
        if not any([args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path]):
            raise ValueError("a recipe needs the qa family but no --qa_* path is set")
        return QADocsJointSource(
            hotpotqa_path=args.qa_hotpotqa_path,
            wiki2_path=args.qa_2wiki_path,
            longmagpie_path=args.qa_longmagpie_path,
            **common,
        )
    raise ValueError(f"unknown family: {family!r}")


def scan_family_pool(
    family: str,
    args: argparse.Namespace,
    tokenizer,
    stamp: str,
    cache: Dict[Tuple[str, str], int],
    token_cap: Optional[float],
) -> List[PoolEntry]:
    """Stream one family's source into a pool of estimated-token entries.

    Stops once the cumulative estimate reaches ``token_cap`` (the crossing
    example is included), so quota-filling scans stay bounded in memory/time;
    ``token_cap=None`` scans the whole source.  Estimates are cached by
    (qid, stamp) and the cache dict is updated in place.
    """
    source = _family_source(family, args)
    pool: List[PoolEntry] = []
    seen: Set[str] = set()
    total = 0
    cache_hits = 0
    for example in source:
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
        pool.append(
            PoolEntry(
                qid=example.qid,
                session_id=example.session_id,
                estimated_tokens=estimated,
            )
        )
        total += estimated
        if token_cap is not None and total >= token_cap:
            break
    logger.info(
        "family=%s: scanned %d examples (%d estimated tokens, cap=%s, cache_hits=%d)",
        family,
        len(pool),
        total,
        int(token_cap) if token_cap is not None else "none",
        cache_hits,
    )
    return pool


# ---------------------------------------------------------------------------
# Output writing + ETA.
# ---------------------------------------------------------------------------


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
    return (
        f"{name}: 预计完成 = {finish:%Y-%m-%d %H:%M} "
        f"(presented≈{presented_tokens / 1e6:.1f}M estimated tokens; "
        f"calibration 32M ≈ {small_arm_hours}h from the small arm)"
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
        "--recipe",
        action="append",
        required=True,
        metavar="name=family:share,...",
        help=f"repeatable; families from {FAMILIES}; shares sum to 1.0",
    )
    parser.add_argument("--budget_estimated_tokens", type=int, required=True, help="N: per-recipe total estimated source tokens")
    parser.add_argument("--oversample_factor", type=float, default=1.25, help="scan cap = quota x this factor")
    parser.add_argument("--repeat_unique_tokens", type=int, default=None, help="M: also emit <recipe>_repeat variants with ~M unique tokens per family pool")
    parser.add_argument("--removal_files", nargs="*", default=[], help="dedup removal lists (dicts with removal_list, or bare JSON string lists)")
    parser.add_argument("--order_seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer", default=None, help="local HF tokenizer path; default: whitespace fake")
    parser.add_argument("--small_arm_hours", type=float, default=None, help="measured wall-clock hours per 32M presented tokens on the small arm (enables the ETA line)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    recipes = [parse_recipe(spec) for spec in args.recipe]
    names = [recipe.name for recipe in recipes]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate --recipe names: {sorted(names)}")
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
    for family in sorted(caps):
        cache_path = out_dir / f"tokencache_{family}.jsonl"
        cache = _load_token_cache(cache_path)
        cache_size = len(cache)
        pools[family] = scan_family_pool(family, args, tokenizer, stamp, cache, caps[family])
        if len(cache) != cache_size:
            _write_token_cache(cache_path, cache)

    results = plan_recipes(
        pools,
        recipes,
        budget_estimated_tokens=args.budget_estimated_tokens,
        order_seed=args.order_seed,
        repeat_unique_tokens=args.repeat_unique_tokens,
        removal_identifiers=removal_identifiers,
        removal_file_counts=removal_file_counts,
        small_arm_hours=args.small_arm_hours,
    )
    write_outputs(results, out_dir)

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
        print(
            f"{name}: {plan['order_examples']} examples, "
            f"presented≈{presented / 1e6:.1f}M estimated tokens"
            + (f", recommended_epochs={plan['recommended_epochs']}" if plan["variant"] == "repeat" else "")
        )
        if args.small_arm_hours is not None:
            print(_eta_line(name, presented, args.small_arm_hours))


if __name__ == "__main__":
    main()
