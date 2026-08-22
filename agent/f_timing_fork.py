"""F pilot driver: one speculative-compaction timing fork per example.

Base harness is ``agent/eval_joint_next_action_c2kv.py`` (joint eval): the
joint grid is tool-schema chunks first, then history chunks in chronological
order, and the joint eval shares ``build_tool_chunks`` /
``_history_chunk_budget`` with the trainer, so both branches of the fork stay
inside the training distribution.  The fork point is the boundary AFTER the
LAST history chunk — the last chunk of the grid — which is exactly the
"shared old prefix, difference confined to the most recent segment" shape the
pilot is about.

Two branches per eligible example:

- ``compress_now`` (branch A) — the already-scheduled compaction runs on the
  last chunk too; this is the ordinary joint c2kv prefix
  (``_build_c2kv_prefix``), and it is the null policy of the pilot;
- ``defer`` (branch B) — the older chunks are gist-compressed exactly as in A
  and the LAST chunk stays raw for this one decision, appended with
  ``_prefill_tokens_with_cache_maybe_gist`` (the hybrid precedent,
  eval_agent_history_c2kv.py:770 / :915).

Both branches read the joint grid through the same builder knobs, and the
pilot runs with ``delay_recent_turns=0``: holding recent turns out of the grid
would put raw content on both sides of the fork point, so branch B refuses to
build rather than quietly dropping it.

Position invariant (implementation-invalid trigger): both branches place the
current turn at the same original position, i.e.
``system_length + doc_length`` must be identical.  The driver asserts it on
every example; a violation means the two branches are not comparable and the
run is aborted rather than reported.

Passes (compute-minimal, see configs/bdf_pilot/f_prereg.md):

- ``greedy_core`` — 2 generations/example (A-greedy, B-greedy).  F0/F2 are the
  single arms; F3-greedy, F4 and F5 are derived at analysis time from these two
  recorded outputs and buy no extra rollouts.
- ``sampled`` — 3 generations/example (A-s0, A-s1, B-s0) at T/top_p.  Requires
  the ``do_sample``/``temperature``/``top_p`` parameters on
  ``_generate_from_input_ids``; the driver probes for them with
  ``inspect.signature`` and refuses ``--arm_set sampled`` when they are
  missing.  The greedy path NEVER passes those keywords, so ``greedy_core``
  runs unchanged against the pre-sampling harness.

Memory honesty (verbatim in the prereg): inside the speculation window both
branches are resident, so the fork costs 1.125x the last segment rather than
saving anything; the saving only materialises after the commit.  No statement
of the form "compression frees memory, so we can afford more branches" is
made anywhere in this pilot.

Naming discipline: the mechanical output check is called
``deterministic_check_*`` in fields, functions and prose alike; no synonym for
it is introduced anywhere in the F line.

Usage (single card, resume-safe):
  python agent/f_timing_fork.py --model ./checkpoints/qwen3-4b-joint-c2kv-npu \\
      --base_model ./models/Qwen3-4B-Instruct-2507 \\
      --arm_set greedy_core --output_file ./outputs/f_pilot/f_fork.jsonl \\
      --prereg_file ./configs/bdf_pilot/f_prereg.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))
    sys.path.insert(0, str(_ROOT / "agent"))

from eval_joint_next_action_c2kv import (  # noqa: E402
    _build_c2kv_prefix,
    _condition_doc_chunks,
    _current_prompt_ids,
    _doc_grid,
    _load_examples,
    _prediction_metrics,
)
from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _generate_from_input_ids,
    _load_model,
    _prefill_system,
    _setup_device,
)
from eval_agent_history_c2kv import (  # noqa: E402
    _clear_device_cache,
    _has_tool_call,
    _is_oom_error,
    _load_tokenizer,
    _prefill_tokens_with_cache_maybe_gist,
    _resolve_model_checkpoint,
)
from eval_toolathlon_first_tool_c2kv import _parse_pred_call  # noqa: E402
from train.train_data_multiturn import _chat_template_ids  # noqa: E402

from f_fork_common import (  # noqa: E402
    ARM_PASS_GREEDY,
    ARM_PASS_SAMPLED,
    BRANCH_COMPRESS_NOW,
    BRANCH_DEFER,
    action_key,
    deterministic_check_pass,
    fork_eligibility,
    kv_bytes_per_token,
    load_done_keys,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger("f_timing_fork")

ARM_SETS = ("greedy_core", "sampled", "both")
SAMPLING_PARAMS = ("do_sample", "temperature", "top_p")

# (branch, rollout_index) plan per pass.
PASS_PLAN: Dict[str, Tuple[Tuple[str, int], ...]] = {
    ARM_PASS_GREEDY: ((BRANCH_COMPRESS_NOW, 0), (BRANCH_DEFER, 0)),
    ARM_PASS_SAMPLED: (
        (BRANCH_COMPRESS_NOW, 0),
        (BRANCH_COMPRESS_NOW, 1),
        (BRANCH_DEFER, 0),
    ),
}


# ---------------------------------------------------------------------------
# Provenance stamps.
# ---------------------------------------------------------------------------


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _model_config_sha256(model_path: Optional[str]) -> Optional[str]:
    """sha256 of the resolved checkpoint's config.json.

    Hashing multi-GB safetensors on every launch is not worth the wall clock;
    the config carries the architecture + gist fields that decide whether two
    runs are comparable, and ``model_path`` is stamped next to it.
    """

    if not model_path:
        return None
    return _sha256_file(str(Path(model_path) / "config.json"))


def _git_short_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _dtype_bytes(model: Any) -> int:
    dtype = getattr(model, "dtype", None) or getattr(model.config, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize:
        return int(itemsize)
    return 2


def _kv_bytes_per_token_for(model: Any) -> int:
    config = model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(
        getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    )
    head_dim = int(
        getattr(config, "head_dim", None)
        or (config.hidden_size // config.num_attention_heads)
    )
    return kv_bytes_per_token(num_layers, num_kv_heads, head_dim, _dtype_bytes(model))


# ---------------------------------------------------------------------------
# Eligibility (prereg E1-E4).
# ---------------------------------------------------------------------------


def _joint_chunks(tokenizer: Any, example: Any, args: argparse.Namespace):
    """Joint grid for one example, with the SAME knobs branch A's builder uses.

    ``chunk_policy`` / ``delay_recent_turns`` are read with ``getattr`` because
    F does not expose them today; forwarding them anyway keeps this helper
    byte-for-byte aligned with ``_build_c2kv_prefix`` (:501-502), so branch B
    can never end up chunking the history differently from branch A if the F
    line ever gains those flags.
    """

    return _condition_doc_chunks(
        tokenizer,
        example,
        "joint",
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
        per_side_caps=True,
        chunk_policy=getattr(args, "chunk_policy", "agent-turn"),
        delay_recent_turns=getattr(args, "delay_recent_turns", 0),
    )


def _check_eligibility(
    tokenizer: Any, example: Any, args: argparse.Namespace
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """E1 (builder skip) inherited, then E2-E4 via ``fork_eligibility``."""

    tool_chunks, history_chunks, skip_reason, _tool_meta = _joint_chunks(
        tokenizer, example, args
    )
    if skip_reason is not None:
        return False, skip_reason, {"builder_skip": skip_reason}
    return fork_eligibility(
        [len(chunk) for chunk in history_chunks],
        len(tool_chunks),
        _has_tool_call(example.answer),
        l_min=args.l_min,
        max_doc_length=args.max_doc_length,
    )


# ---------------------------------------------------------------------------
# Branch prefixes.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _build_compress_now_prefix(
    model: Any, tokenizer: Any, example: Any, args: argparse.Namespace
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Branch A: the ordinary joint c2kv prefix, every chunk compressed."""

    run_args = copy.copy(args)
    run_args.condition = "joint"
    run_args.legacy_mode_caps = False
    prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, run_args)
    if prefix is None:
        return None, skip_reason
    prefix.update({
        "branch": BRANCH_COMPRESS_NOW,
        "raw_segment_tokens": 0,
        "shared_gist_tokens": None,
    })
    return prefix, None


@torch.inference_mode()
def _build_defer_prefix(
    model: Any, tokenizer: Any, example: Any, args: argparse.Namespace
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Branch B: gist the older chunks, keep the LAST chunk raw for one turn.

    Step 3 places the raw segment at ``system_length + sum(original lengths of
    the shared chunks)`` — the ORIGINAL positions, not the compressed cache
    positions — so the current turn lands at the same absolute position as in
    branch A.  ``use_gist=True`` matches every other c2kv arm's answer-side
    convention (the hybrid builder does the same,
    eval_agent_history_c2kv.py:990-997).
    """

    tool_chunks, history_chunks, skip_reason, tool_meta = _joint_chunks(
        tokenizer, example, args
    )
    if skip_reason is not None:
        return None, skip_reason
    # A non-empty raw_history_ids means the builder held some recent turns OUT
    # of the compressed grid (--delay_recent_turns > 0 on the B line).  Branch
    # A rides them in front of the current turn; branch B cannot, because the
    # fork segment IS the last grid chunk and prepending delayed turns would
    # put raw content on BOTH sides of the fork point, which is not the
    # experiment this pilot preregistered.  Refuse loudly instead of silently
    # dropping the content.
    delayed_ids = tool_meta.get("raw_history_ids") or []
    if delayed_ids:
        raise SystemExit(
            "implementation-invalid: the defer branch has no defined semantics "
            f"when the builder delays recent turns ({len(delayed_ids)} raw "
            f"history tokens held out of the grid for qid={example.qid}). "
            "Run the F pilot with delay_recent_turns=0 (its only supported "
            "setting), or extend _build_defer_prefix before enabling the flag."
        )
    chunks = [*tool_chunks, *history_chunks]
    if len(chunks) < args.min_doc_num:
        return None, f"doc_num<{args.min_doc_num}"
    shared_chunks = chunks[:-1]
    fork_chunk = chunks[-1]
    if not shared_chunks:
        # E2 guarantees >= 2 history chunks, so this is unreachable in the
        # pilot; refuse loudly rather than silently degrade to a full prefill.
        return None, "no_shared_chunks"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    shared_grid = _doc_grid(shared_chunks, args.max_doc_length)
    (
        cache,
        shared_tokens,
        shared_gist_tokens,
        _shared_ratio,
        compress_sec,
        blend_sec,
    ) = _build_tool_cache(
        model,
        shared_grid,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.override_ratio,
    )

    fork_input_ids = torch.tensor([fork_chunk], dtype=torch.long, device=model.device)
    cache, raw_segment_tokens, full_prefill_sec = _prefill_tokens_with_cache_maybe_gist(
        model,
        fork_input_ids,
        past_key_values=cache,
        past_length=system_length + shared_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=True,
    )

    doc_tokens = shared_tokens + raw_segment_tokens
    compressed_tokens = shared_gist_tokens + raw_segment_tokens
    return {
        "cache": cache,
        "system_length": system_length,
        # doc_length is the ORIGINAL token span the prefix stands for, which is
        # what the position bookkeeping consumes -- identical to branch A.
        "doc_length": doc_tokens,
        "cache_length": cache.get_seq_length(),
        "use_gist": True,
        "doc_tokens": doc_tokens,
        "doc_chunks": len(chunks),
        "tool_doc_chunks": len(tool_chunks),
        "history_doc_chunks": len(history_chunks),
        "target_known": tool_meta.get("target_known"),
        "target_in_grid": tool_meta.get("target_in_grid"),
        "target_truncated_to_cap": tool_meta.get("target_truncated_to_cap"),
        "gist_tokens": shared_gist_tokens,
        "compressed_tokens": compressed_tokens,
        "actual_compression_ratio": (
            doc_tokens / compressed_tokens if compressed_tokens else 0.0
        ),
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": compress_sec,
        "full_prefill_sec": full_prefill_sec,
        "blend_sec": blend_sec,
        "branch": BRANCH_DEFER,
        # Always empty here -- the guard above refuses the only case that could
        # populate it.  Carried explicitly so both branch prefixes expose the
        # same key and _generate_branch needs no branch-specific special case.
        "raw_history_ids": [],
        "raw_recent_tokens": 0,
        "raw_segment_tokens": raw_segment_tokens,
        "shared_gist_tokens": shared_gist_tokens,
        "shared_original_tokens": shared_tokens,
    }, None


def _assert_position_invariant(
    compress_now_length: int, defer_length: int, qid: str
) -> int:
    """Both branches must place the current turn at the same original position.

    This is the implementation-invalid trigger of the F line: if it fires, the
    two branches are not comparable and no downstream number means anything.
    """

    if compress_now_length != defer_length:
        raise SystemExit(
            "implementation-invalid: branch position mismatch for qid="
            f"{qid}: compress_now original_prefix_length={compress_now_length} "
            f"!= defer={defer_length}"
        )
    return compress_now_length


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------


def _sampling_supported() -> bool:
    """Probe the shared generator for the B/D-line sampling parameters."""

    try:
        parameters = inspect.signature(_generate_from_input_ids).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return all(name in parameters for name in SAMPLING_PARAMS)


def _seed_generation(gen_seed: int, qid: str, branch: str, rollout_index: int) -> int:
    """Per-rollout seed, same primitive as the B/D per-row seeding contract."""

    key = f"{qid}:{branch}:{rollout_index}"
    seed = (int(gen_seed) * 1_000_003) ^ zlib.crc32(key.encode("utf-8"))
    torch.manual_seed(seed)
    return seed


@torch.inference_mode()
def _generate_branch(
    model: Any,
    tokenizer: Any,
    example: Any,
    prefix: Dict[str, Any],
    args: argparse.Namespace,
    *,
    branch: str,
    arm_pass: str,
    rollout_index: int,
) -> Dict[str, Any]:
    """One recorded rollout on one branch prefix (clone of joint's generator).

    Mirrors ``eval_joint_next_action_c2kv._generate_with_prefix`` (:750-808)
    and adds the fork bookkeeping.  The greedy pass passes NO sampling
    keywords at all, so it does not depend on the B/D-line signature change.
    """

    # Delayed history docs (raw_history_ids) stay uncompressed and ride in
    # front of the current turn in the plain prompt, exactly as in joint's
    # _generate_with_prefix (:847-856): position_ids already start at
    # system_length + doc_length and doc_length counts only the compressed
    # grid, so the raw turns occupy the positions right after it.  Branch B
    # refuses to build at all when this list is non-empty, so today it is
    # always [] on both branches; keeping the prepend here means a future
    # --delay_recent_turns on the F line cannot silently drop content.
    raw_ids = prefix.get("raw_history_ids") or []
    prompt_ids = _current_prompt_ids(tokenizer, example, args.max_prompt_tokens)
    prompt_input_ids = torch.tensor(
        [list(raw_ids) + prompt_ids], dtype=torch.long, device=model.device
    )
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = prefix["system_length"] + prefix["doc_length"]
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    sampling_kwargs: Dict[str, Any] = {}
    gen_seed_used: Optional[int] = None
    if arm_pass == ARM_PASS_SAMPLED:
        gen_seed_used = _seed_generation(
            args.gen_seed, example.qid, branch, rollout_index
        )
        sampling_kwargs = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
        **sampling_kwargs,
    )

    metrics = _prediction_metrics(tokenizer, example.answer, prediction)
    metrics["generated_tokens"] = generated_tokens
    ttft = (
        prefix["system_prefill_sec"]
        + prefix["tool_compress_sec"]
        + prefix["full_prefill_sec"]
        + prefix["blend_sec"]
    )
    parsed_pred = _parse_pred_call(prediction)
    pred_key = action_key(parsed_pred)
    gold_key = action_key(_parse_pred_call(example.answer))
    bytes_per_token = prefix.get("kv_bytes_per_token") or 0
    # Everything fed to the model this step is resident: the prefix cache, the
    # delayed raw turns, the current turn, and what was decoded.
    peak_cache_tokens = (
        prefix["cache_length"] + int(prompt_input_ids.shape[1]) + generated_tokens
    )
    metrics.update({
        "branch": branch,
        "arm_pass": arm_pass,
        "rollout_index": rollout_index,
        "gen_seed_used": gen_seed_used,
        "deterministic_check_pass": deterministic_check_pass(parsed_pred),
        "pred_action_key": pred_key,
        "gold_action_key": gold_key,
        "action_key_match": bool(gold_key is not None and pred_key == gold_key),
        "doc_tokens": prefix["doc_tokens"],
        "doc_chunks": prefix["doc_chunks"],
        "tool_doc_chunks": prefix["tool_doc_chunks"],
        "history_doc_chunks": prefix["history_doc_chunks"],
        "target_known": prefix.get("target_known"),
        "target_in_grid": prefix.get("target_in_grid"),
        "gist_tokens": prefix["gist_tokens"],
        "compressed_tokens": prefix["compressed_tokens"],
        "raw_segment_tokens": prefix.get("raw_segment_tokens"),
        "shared_gist_tokens": prefix.get("shared_gist_tokens"),
        "prompt_tokens": len(prompt_ids),
        "raw_recent_tokens": len(raw_ids),
        "cache_tokens": prefix["cache_length"],
        "peak_cache_tokens": peak_cache_tokens,
        "kv_bytes_per_token": bytes_per_token,
        "peak_bytes": peak_cache_tokens * bytes_per_token,
        "original_prefix_length": original_prefix_length,
        "actual_compression_ratio": round(prefix["actual_compression_ratio"], 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4),
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4),
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4),
        "blend_sec": round(prefix["blend_sec"], 4),
        "ttft_sec": round(ttft, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(ttft + generate_sec, 4),
    })
    return metrics


def _build_branch_prefix(
    model: Any, tokenizer: Any, example: Any, args: argparse.Namespace, branch: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if branch == BRANCH_COMPRESS_NOW:
        prefix, reason = _build_compress_now_prefix(model, tokenizer, example, args)
    elif branch == BRANCH_DEFER:
        prefix, reason = _build_defer_prefix(model, tokenizer, example, args)
    else:  # pragma: no cover - closed set
        raise ValueError(f"Unknown branch {branch!r}")
    if prefix is not None:
        prefix["kv_bytes_per_token"] = _kv_bytes_per_token_for(model)
    return prefix, reason


def _skip_row(
    example: Any, arm_pass: str, branch: str, rollout_index: int, reason: str
) -> Dict[str, Any]:
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "subset": example.subset,
        "arm_pass": arm_pass,
        "branch": branch,
        "rollout_index": rollout_index,
        "skipped": True,
        "skip_reason": reason,
    }


def _fork_rows_for_example(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    arm_pass: str,
    *,
    fork_meta: Dict[str, Any],
    device: str,
    assert_greedy_repeat: bool = False,
) -> List[Dict[str, Any]]:
    """All rollouts of one pass for one eligible example.

    Each rollout rebuilds its own branch prefix: ``model.generate`` mutates the
    cache it is handed, so reusing one prefix across two rollouts of the same
    branch would silently score the second rollout against a polluted cache.
    Rebuilding costs prefill time, which the ledger reports honestly (the
    analyzer also reports a prefill-deduplicated column).
    """

    rows: List[Dict[str, Any]] = []
    prefix_lengths: Dict[str, int] = {}
    branch_geometry: Dict[str, Dict[str, Any]] = {}
    repeat_reference: Optional[str] = None

    for branch, rollout_index in PASS_PLAN[arm_pass]:
        try:
            prefix, reason = _build_branch_prefix(
                model, tokenizer, example, args, branch
            )
            if prefix is None:
                row = _skip_row(
                    example, arm_pass, branch, rollout_index, reason or "no_prefix"
                )
            else:
                prefix_lengths[branch] = prefix["system_length"] + prefix["doc_length"]
                branch_geometry[branch] = {
                    "cache_length": prefix["cache_length"],
                    "gist_tokens": prefix["gist_tokens"],
                    "raw_segment_tokens": prefix.get("raw_segment_tokens") or 0,
                    "shared_gist_tokens": prefix.get("shared_gist_tokens"),
                    "kv_bytes_per_token": prefix["kv_bytes_per_token"],
                }
                row = _generate_branch(
                    model,
                    tokenizer,
                    example,
                    prefix,
                    args,
                    branch=branch,
                    arm_pass=arm_pass,
                    rollout_index=rollout_index,
                )
                if (
                    assert_greedy_repeat
                    and arm_pass == ARM_PASS_GREEDY
                    and branch == BRANCH_COMPRESS_NOW
                ):
                    repeat_reference = row["prediction"]
                del prefix
        except RuntimeError as error:
            if not _is_oom_error(error):
                raise
            logger.warning(
                "OOM: qid=%s pass=%s branch=%s rollout=%s -- row skipped, retried on resume",
                example.qid, arm_pass, branch, rollout_index,
            )
            row = _skip_row(example, arm_pass, branch, rollout_index, "oom")
        rows.append(row)
        _clear_device_cache(device)

    if repeat_reference is not None:
        _run_greedy_repeat_check(
            model, tokenizer, example, args, repeat_reference, device
        )

    # Both branches present: assert the position invariant, then fill the
    # two-branch residency ledger.
    if BRANCH_COMPRESS_NOW in prefix_lengths and BRANCH_DEFER in prefix_lengths:
        _assert_position_invariant(
            prefix_lengths[BRANCH_COMPRESS_NOW],
            prefix_lengths[BRANCH_DEFER],
            example.qid,
        )
        residency = _residency_bytes(branch_geometry, rows, fork_meta)
        for row in rows:
            if not row.get("skipped"):
                row.update(residency)
    for row in rows:
        row.setdefault("session_id", example.session_id)
        row.setdefault("subset", example.subset)
        row["qid"] = example.qid
        row.update({
            key: value
            for key, value in fork_meta.items()
            if key in ("fork_chunk_index", "last_chunk_tokens", "history_chunk_count", "tool_chunk_count")
        })
    return rows


def _residency_bytes(
    branch_geometry: Dict[str, Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    fork_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Measured vs logical-shared KV residency inside the speculation window.

    ``resident_bytes_measured`` is the naive physical sum of the two branch
    caches as this offline driver actually materialises them (each branch holds
    its own copy of the shared prefix).  ``resident_bytes_logical_shared`` is
    what an implementation that shares the old prefix would hold:
    shared_prefix + gist(x_T) + raw(x_T).  Against raw(x_T) alone that fork
    segment is 1.125x at ratio 8 -- the speculation window costs MORE memory,
    not less; the saving only lands after the commit.
    """

    bytes_per_token = branch_geometry[BRANCH_COMPRESS_NOW]["kv_bytes_per_token"]
    a_geometry = branch_geometry[BRANCH_COMPRESS_NOW]
    b_geometry = branch_geometry[BRANCH_DEFER]
    # One rollout per branch is resident at a time inside the window, so the
    # measured two-branch sum takes the WORST rollout of each branch, not the
    # sum over all recorded rollouts of the sampled pass.
    measured_tokens = sum(
        max(
            [
                row.get("peak_cache_tokens", 0)
                for row in rows
                if not row.get("skipped") and row.get("branch") == branch
            ]
            or [0]
        )
        for branch in (BRANCH_COMPRESS_NOW, BRANCH_DEFER)
    )

    shared_gist_tokens = b_geometry.get("shared_gist_tokens")
    raw_segment_tokens = b_geometry.get("raw_segment_tokens") or 0
    gist_fork_tokens: Optional[int] = None
    if shared_gist_tokens is not None:
        gist_fork_tokens = max(0, a_geometry["gist_tokens"] - shared_gist_tokens)
    shared_prefix_tokens = b_geometry["cache_length"] - raw_segment_tokens
    logical_tokens = (
        shared_prefix_tokens + (gist_fork_tokens or 0) + raw_segment_tokens
    )
    return {
        "resident_bytes_measured": measured_tokens * bytes_per_token,
        "resident_tokens_measured": measured_tokens,
        "resident_bytes_logical_shared": logical_tokens * bytes_per_token,
        "resident_tokens_logical_shared": logical_tokens,
        "fork_segment_gist_tokens": gist_fork_tokens,
        "fork_segment_raw_tokens": raw_segment_tokens,
        "fork_segment_logical_ratio": (
            round((gist_fork_tokens + raw_segment_tokens) / raw_segment_tokens, 4)
            if gist_fork_tokens is not None and raw_segment_tokens
            else None
        ),
        "shared_prefix_tokens": shared_prefix_tokens,
        "last_chunk_tokens": fork_meta.get("last_chunk_tokens"),
    }


def _run_greedy_repeat_check(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    reference: str,
    device: str,
) -> None:
    """Cheapest determinism / assembly self-check: rerun branch A greedily.

    A byte-for-byte mismatch means the prefix assembly or the decode is not
    reproducible, in which case every downstream paired number is meaningless
    -- abort instead of reporting.
    """

    prefix, reason = _build_branch_prefix(
        model, tokenizer, example, args, BRANCH_COMPRESS_NOW
    )
    if prefix is None:
        raise SystemExit(
            f"implementation-invalid: greedy repeat check could not rebuild qid={example.qid} ({reason})"
        )
    repeat = _generate_branch(
        model,
        tokenizer,
        example,
        prefix,
        args,
        branch=BRANCH_COMPRESS_NOW,
        arm_pass=ARM_PASS_GREEDY,
        rollout_index=0,
    )
    del prefix
    _clear_device_cache(device)
    if repeat["prediction"] != reference:
        raise SystemExit(
            "implementation-invalid: greedy branch-A output is not reproducible for "
            f"qid={example.qid}\n  first : {reference!r}\n  repeat: {repeat['prediction']!r}"
        )
    logger.info("greedy repeat check OK for qid=%s", example.qid)


# ---------------------------------------------------------------------------
# Eval loop.
# ---------------------------------------------------------------------------


def _passes_for(arm_set: str) -> List[str]:
    if arm_set == "greedy_core":
        return [ARM_PASS_GREEDY]
    if arm_set == "sampled":
        return [ARM_PASS_SAMPLED]
    return [ARM_PASS_GREEDY, ARM_PASS_SAMPLED]


def _load_and_select_examples(args: argparse.Namespace) -> List[Any]:
    """Load with the joint loader, then apply the frozen qid manifest if given.

    ``--max_examples`` is disabled for the load when a manifest is present: the
    manifest IS the frozen cap, and truncating first would make perfectly
    reproducible qids look "not reproduced by the loader".
    """

    load_args = args
    if args.qid_manifest:
        load_args = copy.copy(args)
        load_args.max_examples = 0
    return _select_examples(_load_examples(load_args), args)


def _select_examples(examples: Sequence[Any], args: argparse.Namespace) -> List[Any]:
    if not args.qid_manifest:
        return list(examples)
    manifest = json.loads(Path(args.qid_manifest).read_text(encoding="utf-8"))
    qids = manifest["qids"] if isinstance(manifest, dict) else list(manifest)
    by_qid: Dict[str, Any] = {}
    for example in examples:
        by_qid.setdefault(example.qid, example)
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} frozen qids not reproduced by the loader: {missing[:5]}"
        )
    return [by_qid[qid] for qid in qids]


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    passes = _passes_for(args.arm_set)
    if ARM_PASS_SAMPLED in passes and not _sampling_supported():
        raise SystemExit(
            "--arm_set requires the sampled pass, but "
            "eval_agent_tool_definition_c2kv._generate_from_input_ids does not accept "
            f"{SAMPLING_PARAMS}. Land the B/D-line sampling switch first, or run "
            "--arm_set greedy_core (which never passes those keywords)."
        )

    device = _setup_device(args.device_type)
    args.model = _resolve_model_checkpoint(args.model)
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = _load_and_select_examples(args)
    logger.info("Loaded %d joint %s examples", len(examples), args.split)

    stamps = {
        "model_path": args.model,
        "base_model": args.base_model,
        "model_config_sha256": _model_config_sha256(args.model),
        "prereg_sha256": _sha256_file(args.prereg_file),
        "split_manifest_sha256": _sha256_file(args.split_manifest_file),
        "qid_manifest_sha256": _sha256_file(args.qid_manifest),
        "git_short_sha": _git_short_sha(),
        "override_ratio": args.override_ratio,
        "l_min": args.l_min,
        "gen_seed": args.gen_seed,
        "temperature": args.temperature if ARM_PASS_SAMPLED in passes else None,
        "top_p": args.top_p if ARM_PASS_SAMPLED in passes else None,
    }
    logger.info("Provenance stamps: %s", json.dumps(stamps, ensure_ascii=False))

    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = False
    model_args.baseline_model_class = "custom"
    logger.info("Loading joint checkpoint %s", args.model)
    model = _load_model(model_args, tokenizer, device)
    attn_runtime = getattr(model.config, "_attn_implementation", None)
    logger.info("runtime attn impl=%s", attn_runtime)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(out_path) if args.resume else set()
    if done:
        logger.info("Resume: %d rollouts already recorded", len(done))

    counters = {"written": 0, "skipped": 0, "eligible": 0, "ineligible": 0}
    repeat_budget = max(0, int(args.assert_greedy_repeat))
    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as handle:

        def _emit(row: Dict[str, Any]) -> None:
            row.update(stamps)
            row["attn_impl_runtime"] = attn_runtime
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            counters["written"] += 1
            if row.get("skipped"):
                counters["skipped"] += 1

        for example in tqdm(examples, desc=f"f_fork/{args.arm_set}"):
            eligible, skip_reason, fork_meta = _check_eligibility(
                tokenizer, example, args
            )
            if not eligible:
                counters["ineligible"] += 1
                row = _skip_row(example, passes[0], "none", 0, skip_reason or "ineligible")
                row.update(fork_meta)
                _emit(row)
                continue
            counters["eligible"] += 1
            for arm_pass in passes:
                # Resume granularity is the whole PASS, not the single rollout:
                # the position invariant and the two-branch residency ledger
                # need both branches of the same example in hand.  A partially
                # written pass is therefore regenerated in full, and the
                # duplicate rows are collapsed last-write-wins by
                # f_fork_common.index_rows_by_qid.
                pending = [
                    (branch, rollout)
                    for branch, rollout in PASS_PLAN[arm_pass]
                    if (example.qid, arm_pass, branch, rollout) not in done
                ]
                if not pending:
                    continue
                start = time.perf_counter()
                rows = _fork_rows_for_example(
                    model,
                    tokenizer,
                    example,
                    args,
                    arm_pass,
                    fork_meta=fork_meta,
                    device=device,
                    assert_greedy_repeat=(
                        repeat_budget > 0 and arm_pass == ARM_PASS_GREEDY
                    ),
                )
                if repeat_budget > 0 and arm_pass == ARM_PASS_GREEDY:
                    repeat_budget -= 1
                wall = round(time.perf_counter() - start, 3)
                for row in rows:
                    row["wall_sec"] = wall
                    _emit(row)
            _clear_device_cache(device)

    summary = {
        "output_file": str(out_path),
        "arm_set": args.arm_set,
        "passes": passes,
        "num_examples": len(examples),
        "num_eligible": counters["eligible"],
        "num_ineligible": counters["ineligible"],
        "num_rows_written": counters["written"],
        "num_rows_skipped": counters["skipped"],
        "sampling_supported": _sampling_supported(),
        **stamps,
    }
    summary_path = out_path.with_suffix(".run.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %d rows -> %s", counters["written"], out_path)
    logger.info("Wrote run summary -> %s", summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F pilot: one speculative-compaction timing fork per example."
    )
    parser.add_argument("--model", required=True, help="Joint C2KV checkpoint (dir or checkpoint-* parent).")
    parser.add_argument("--base_model", help="Base model path (tokenizer fallback).")
    parser.add_argument("--tokenizer", help="Tokenizer path. Defaults to --base_model/--model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/f_pilot/f_timing_fork.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    # --- F-specific ---
    parser.add_argument("--arm_set", choices=list(ARM_SETS), default="greedy_core")
    parser.add_argument("--l_min", type=int, default=64, help="E3 lower bound on last-chunk tokens.")
    parser.add_argument("--gen_seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prereg_file", default="./configs/bdf_pilot/f_prereg.md")
    parser.add_argument("--qid_manifest", help="Frozen qid list (json list or {'qids': [...]}).")
    parser.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--assert_greedy_repeat",
        type=int,
        default=2,
        help="Rerun branch-A greedy for the first N eligible examples and require "
        "byte-identical text; a mismatch aborts the run as implementation-invalid.",
    )
    # --- loader / budget parameters, identical to the joint eval ---
    parser.add_argument("--max_examples", type=int, default=200, help="Maximum examples; <=0 means all.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max_input_chars", type=int)
    parser.add_argument("--max_answer_chars", type=int)
    parser.add_argument("--prefix_history_doc_num", type=int)
    parser.add_argument("--prefix_history_exact", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--canonical_format_prob", type=float, default=0.7)
    parser.add_argument("--minified_json_prob", type=float, default=0.2)
    parser.add_argument("--shuffle_tools", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--truncate_description_chars", type=int, default=600)
    parser.add_argument("--max_tools_per_sample", type=int, default=32)
    parser.add_argument("--same_namespace_negative_tools", type=int, default=8)
    parser.add_argument("--random_negative_tools", type=int, default=24)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=24)
    parser.add_argument("--max_tool_chunks", type=int, default=None)
    parser.add_argument("--min_doc_num", type=int, default=2)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--override_ratio", type=int, default=8)
    parser.add_argument("--history_selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--split_oversized_history_docs", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args(argv)
    if args.l_min <= 0:
        parser.error("--l_min must be positive")
    if args.l_min > args.max_doc_length:
        parser.error("--l_min must not exceed --max_doc_length (E3 would be empty)")
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
