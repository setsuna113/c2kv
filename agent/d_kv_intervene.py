"""Task D driver: one KV-intervention arm over the frozen C->W trigger set.

Arms (see configs/bdf_pilot/d_prereg.md for the frozen definitions):

  none       -> harness mode "c2kv"              (untouched compressed prefix)
  sham       -> harness mode "d_sham_neutral"    (equal-length neutral span)
  corr       -> harness mode "d_corr"            (append-only erratum at k*)
  corr_re    -> harness mode "d_corr_recompute"  (erratum + downstream rebuild)
  full       -> harness mode "full"              (uncompressed upper bound)
  corr_all   -> harness mode "d_corr_all"        (ceiling diagnostic, unregistered)
  sham_mech  -> harness mode "d_sham_mech"       (implementation-invalid guard)

Frozen state is bound by sha256 before anything runs: the bundle file against
the manifest that named it, the sham plan against the manifest it was built
from, and the neutral corpus against the plan.  Every emitted row carries the
manifest and plan shas, so a row can always be traced back to the exact
frozen inputs.  A frozen qid that the harness cannot reproduce is FATAL, not
a skip.

Per-sample incremental jsonl with resume (only NON-skipped rows count as
done, so skipped and OOM rows are retried on the next invocation), mirroring
agent/r4_anchor_rerun.py.

``--downstream_turns K`` (exploratory, prereg addendum 2026-08-23) continues
each trigger for up to K later decision points of the same session on the
LIVE intervened cache: the t* prompt and generated tokens are cropped, the
recorded inter-turn material is teacher-forced in at its true logical
positions, and t*+j is presented and scored exactly as the harness would.
K=0 is exactly the current behavior.  K>0 restricts to none/sham/corr_re and
refuses a full run without the downstream smoke marker.

Usage (NPU server, repo root):
  python agent/d_kv_intervene.py --arm corr_re \
      --output_file ./outputs/d_pilot/d_corr_re.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    for _sub in ("python", "agent", "python/inference"):
        _path = str(_ROOT / _sub)
        if _path not in sys.path:
            sys.path.insert(0, _path)

import eval_agent_history_c2kv as HH  # noqa: E402
from extract_cw_triggers import sha256_text_file as _sha256_file  # noqa: E402

logger = logging.getLogger("d_kv_intervene")

ARM_MODES = {
    "none": "c2kv",
    "sham": "d_sham_neutral",
    "corr": "d_corr",
    "corr_re": "d_corr_recompute",
    "full": "full",
    "corr_all": "d_corr_all",
    "sham_mech": "d_sham_mech",
    # Transfer-manual A1/A2 exploratory arms (2026-08-28)
    "re_only": "d_re_only",
    "corr_text": "d_corr_text",
    # Transfer-manual B1 placement 2x2 (2026-08-29)
    "drop_g": "d_drop_g",
    "splice_keep": "d_splice_keep",
    "splice_rep": "d_splice_rep",
}
# Hybrid-x-D combo (2026-08-29): arms that may run on --base hybrid.  The
# hybrid base (tail-k raw, gist_first layout, docs/hybrid_spec.md) preserves
# original offsets, so the append machinery is unchanged; the splice/recompute
# families pre-date it and stay pure-c2kv.  `full` is base-independent.
HYBRID_BASE_ARMS = {"none", "sham", "corr", "corr_all", "sham_mech", "full"}
PLAN_REQUIRED_ARMS = {"sham"}
PLAN_USING_ARMS = {"sham", "corr", "corr_re", "corr_all", "sham_mech"}
# d_re_only and d_corr_text read no plan payload and are not prereg arms.
DOWNSTREAM_ARMS = {"none", "sham", "corr_re"}
DOWNSTREAM_MAX_TURNS = 3
# Structural offset-0 fields the continuation rows re-carry so per-row cost
# sums keep working on downstream files.
DOWNSTREAM_CARRY_KEYS = (
    "doc_tokens",
    "gist_tokens",
    "d_corr_span_tokens",
    "d_sham_tokens",
    "d_recompute_tokens",
)


def _load_done_qids(path: Path, downstream_turns: int = 0) -> set:
    """Only NON-skipped rows count as done (skipped rows are retried).

    With ``downstream_turns > 0`` done-ness is decided on each qid's LAST
    group only (group = maximal contiguous run of rows starting at an
    offset-0 row): done iff that group has a non-skipped offset-0 row, a
    terminal row, no oom row, and was recorded under ``d_downstream_turns``
    >= the launch K.  Retries append a fresh complete group, so an earlier
    oom group can no longer block convergence.  A file recorded under a
    LARGER K is a wrong launch and fatal, never silently continued — in
    either direction: a K=0 launch pointed at a downstream file (rows carry
    ``d_downstream_turns``) is fatal too, since legacy scanning would parse
    continuation rows as done triggers and corrupt group boundaries.
    """
    if not path.exists():
        return set()
    if not downstream_turns:
        done = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "d_downstream_turns" in row:
                    raise SystemExit(
                        f"FATAL: {path} was produced by a --downstream_turns run "
                        "(rows carry d_downstream_turns); a K=0 launch must not "
                        "silently continue it. Point --output_file elsewhere or "
                        "relaunch with the recorded K."
                    )
                if "qid" in row and not row.get("skipped"):
                    done.add(row["qid"])
        return done
    groups: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "qid" not in row:
                continue
            recorded = int(row.get("d_downstream_turns") or 0)
            if recorded > downstream_turns:
                raise SystemExit(
                    f"FATAL: {path} holds rows recorded under --downstream_turns "
                    f"{recorded}, this launch passes {downstream_turns}. A file "
                    "produced under a larger K must not be silently continued "
                    "under a smaller one."
                )
            qid = row["qid"]
            if int(row.get("d_turn_offset") or 0) == 0:
                groups[qid] = {
                    "offset0_ok": not row.get("skipped"),
                    "terminal": False,
                    "oom": False,
                    "turns": recorded,
                }
            group = groups.get(qid)
            if group is None:
                continue
            if row.get("skip_reason") == "oom":
                group["oom"] = True
            if row.get("d_ds_terminal"):
                group["terminal"] = True
    return {
        qid
        for qid, group in groups.items()
        if group["offset0_ok"]
        and group["terminal"]
        and not group["oom"]
        and group["turns"] >= downstream_turns
    }


def _d_args(args: argparse.Namespace) -> Any:
    """History-harness namespace for the pilot configuration."""
    argv = [
        "prog",
        "--model", args.model,
        "--base_model", args.base_model or args.tokenizer,
        "--tokenizer", args.tokenizer,
        "--dataset_path", args.dataset_path,
        "--split", args.split,
        "--include_tools", args.include_tools,
        "--require_tool_call", args.require_tool_call,
        "--max_examples", str(args.max_examples),
        "--max_samples_per_session", str(args.max_samples_per_session),
        "--eval_ratio", str(args.eval_ratio),
        "--split_seed", str(args.split_seed),
        "--split_manifest_name", args.split_manifest_name,
        "--max_doc_length", str(args.max_doc_length),
        "--max_doc_num", str(args.max_doc_num),
        "--min_doc_num", str(args.min_doc_num),
        "--max_history_tokens", str(args.max_history_tokens),
        "--max_system_length", str(args.max_system_length),
        "--max_prompt_tokens", str(args.max_prompt_tokens),
        "--max_baseline_input_tokens", str(args.max_baseline_input_tokens),
        "--max_new_tokens", str(args.max_new_tokens),
        "--history_selection", args.history_selection,
        "--system_attn_impl", args.attn_impl,
        "--gist_attn_impl", args.attn_impl,
        "--generate_attn_impl", args.attn_impl,
        "--device_type", args.device_type,
        "--override_ratio", str(args.ratio),
        "--hybrid_top_k", str(args.hybrid_top_k),
        "--hybrid_layout", "gist_first",
    ]
    if args.split_manifest_file:
        argv += ["--split_manifest_file", args.split_manifest_file]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def _assert_recipe_matches_run(manifest: Dict[str, Any], args: argparse.Namespace) -> None:
    """Refuse to intervene on a grid other than the one that defined the trigger.

    A C->W trigger is a statement about one specific context: *this* doc grid,
    built at *this* budget by *this* harness, produced a wrong action where the
    full cache produced a right one.  Rebuilding the context under different
    geometry and then patching it measures something else entirely, and nothing
    downstream would reveal the swap -- the qids still resolve, the arms still
    run, the numbers still look like numbers.  The extractor records the grid it
    used precisely so this can be checked, so check it.

    Three ways the grid can silently differ:

    * ``max_doc_length`` -- the history harness convention is 768 while the
      joint harness uses 1024; a manifest frozen under one and a run launched
      under the other chunk the same history differently.
    * ``max_doc_num`` -- the row budget (768/16 vs the joint 1024/24); a
      different row count selects a different tail of the history, so k* would
      point at a different document.
    * harness dialect -- the extractor parses both joint-battery and
      history-harness rows, but D intervenes with the history harness only.
      Joint-dialect triggers describe contexts this driver cannot reproduce.
    """
    recipe = manifest.get("kv_recipe") or {}

    recorded_len = recipe.get("max_doc_length")
    if recorded_len is not None and int(recorded_len) != int(args.max_doc_length):
        raise SystemExit(
            "FATAL: doc-grid mismatch — the trigger manifest was frozen with "
            f"max_doc_length={recorded_len}, this run passes {args.max_doc_length}. "
            "The intervention would land on a different grid than the one whose "
            "failure defined the trigger. Re-run with the recorded value, or "
            "re-extract the triggers at the budget you intend to run."
        )

    recorded_num = recipe.get("max_doc_num")
    if recorded_num is not None and int(recorded_num) != int(args.max_doc_num):
        raise SystemExit(
            "FATAL: doc-grid mismatch — the trigger manifest was frozen with "
            f"max_doc_num={recorded_num}, this run passes {args.max_doc_num}. "
            "A different row budget selects a different history tail, so k* "
            "would no longer name the document whose failure defined the "
            "trigger. Re-run with the recorded value, or re-extract."
        )

    # args.ratio is the pilot's compression ratio; the `full` arm overrides it to
    # 1 at call time (it has no compression) but is still part of the same frozen
    # comparison, so the recorded ratio must match here too.
    recorded_ratio = recipe.get("ratio")
    if recorded_ratio is not None and int(recorded_ratio) != int(args.ratio):
        raise SystemExit(
            f"FATAL: ratio mismatch — manifest was frozen at ratio={recorded_ratio}, "
            f"this run passes --ratio {args.ratio}."
        )

    dialects = manifest.get("source_dialects") or {}
    foreign = {name: n for name, n in dialects.items() if name != "history" and n}
    if foreign:
        raise SystemExit(
            f"FATAL: trigger manifest carries non-history rows {foreign}. "
            "D intervenes with the history harness (eval_agent_history_c2kv); "
            "joint-battery contexts are built differently and cannot be "
            "reproduced here. Extract task-D triggers from history-dialect rows."
        )


def _bind_frozen_state(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and sha-verify manifest / bundles / sham plan."""
    manifest_path = Path(args.manifest)
    manifest_sha = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bundles_path = Path(args.bundles)
    bundles_sha = _sha256_file(bundles_path)
    expected = manifest.get("bundles_sha256")
    if expected and expected != bundles_sha:
        raise SystemExit(
            f"FATAL: bundle sha mismatch — manifest says {expected}, {bundles_path} is {bundles_sha}"
        )

    _assert_recipe_matches_run(manifest, args)

    plan: Optional[Dict[str, Any]] = None
    plan_sha: Optional[str] = None
    arm = args.arm
    if arm in PLAN_USING_ARMS:
        plan_path = Path(args.sham_plan)
        if not plan_path.exists():
            if arm in PLAN_REQUIRED_ARMS:
                raise SystemExit(f"FATAL: {arm} needs the sham plan but {plan_path} is absent")
            logger.warning("sham plan %s absent; %s runs without a k* cross-check", plan_path, arm)
        else:
            plan_sha = _sha256_file(plan_path)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_source = plan.get("qid_source_sha256")
            if plan_source and plan_source != manifest_sha:
                raise SystemExit(
                    f"FATAL: sham plan was frozen against manifest {plan_source}, "
                    f"this run uses {manifest_sha}"
                )
            corpus_path = plan.get("corpus_path")
            if corpus_path and Path(corpus_path).exists():
                corpus_sha = _sha256_file(Path(corpus_path))
                if corpus_sha != plan.get("corpus_sha256"):
                    raise SystemExit(
                        f"FATAL: neutral corpus sha mismatch — plan says {plan.get('corpus_sha256')}, "
                        f"{corpus_path} is {corpus_sha}"
                    )
            if arm in PLAN_REQUIRED_ARMS and not (
                plan.get("budget", {}).get("gate_passed") and plan.get("neutrality", {}).get("gate_passed")
            ):
                raise SystemExit("FATAL: sham plan gates did not pass; refusing to run the sham arm")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "bundles_sha256": bundles_sha,
        "plan": plan,
        "plan_sha256": plan_sha,
    }


def _intervene_table(plan: Optional[Dict[str, Any]], qids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per-qid payload for HH.D_INTERVENE."""
    if not plan:
        return {}
    table: Dict[str, Dict[str, Any]] = {}
    per_qid = plan.get("per_qid", {})
    for qid in qids:
        entry = per_qid.get(qid)
        if entry is None:
            continue
        table[qid] = {
            "k_star": int(entry["k_star"]),
            "span_len": int(entry["span_len"]),
            "sham_token_ids": [int(t) for t in entry.get("sham_token_ids", [])],
        }
    return table


def _normalized_conv(example: Any) -> List[Dict[str, Any]]:
    """Exactly _session_examples' normalization (train_data_multiturn.py):
    _normal_agent_message per raw message, drop None and role=="system"."""
    return [
        item
        for item in (
            HH._normal_agent_message(message) for message in (example.original_messages or [])
        )
        if item is not None and item.get("role") != "system"
    ]


def _continuation_block(
    prev: Any, nxt: Any
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Inter-turn material between two decision points of one session.

    The block is a pure function of the two frozen examples: prev's current
    user/observation messages, the gold assistant action(s) recorded at
    prev's span, and everything up to (excluding) nxt's last-user anchor —
    all taken from the recorded trace snapshots, never from model output.
    An empty block (no new user message) is legal, not a skip.
    """
    conv_p = _normalized_conv(prev)
    conv_n = _normalized_conv(nxt)
    lui_p = len(conv_p) - len(prev.current_messages)
    lui_n = len(conv_n) - len(nxt.current_messages)
    if conv_p[lui_p:] != prev.current_messages or conv_n[lui_n:] != nxt.current_messages:
        return None, "d_ds_conv_reconstruction_mismatch"
    if lui_n < lui_p or conv_n[: len(conv_p)] != conv_p:
        return None, "d_ds_prefix_mismatch"
    return conv_n[lui_p:lui_n], None


def _crop_cache(cache: Any, length: int) -> None:
    """Drop every cache entry past ``length`` (the prompt + generated tokens
    of the turn just scored).  Uses the cache's own crop when it ships one;
    the per-layer slice fallback covers the stub-cache test fixture.  The
    length assert is the tripwire against divergent length bookkeeping in a
    future transformers bump."""
    if callable(getattr(cache, "crop", None)):
        cache.crop(length)
    else:
        for layer in cache.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]
    assert cache.get_seq_length() == length, (
        f"cache crop left {cache.get_seq_length()} slots, expected {length}"
    )


def _downstream_rows(
    model: Any,
    tokenizer: Any,
    prefix: Dict[str, Any],
    trigger: Any,
    later: Sequence[Any],
    run_args: Any,
    args: argparse.Namespace,
    mode: str,
    row0: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Teacher-forced continuation of one trigger group (addendum 2026-08-23).

    Original-layout discipline: the physical cache stays short, the logical
    ledger stays raw.  Blocks enter only through
    _prefill_tokens_with_cache_maybe_gist at their true absolute logical
    positions, so logical - physical is a constant fixed at prefix build and
    rotate_k_cache_rope is never invoked on this path.
    """
    torch = HH.torch
    session_id = trigger.qid.rsplit(":", 1)[0] if ":" in trigger.qid else None

    def _skip_row(offset: int, reason: str, scored: Optional[str], start: float) -> Dict[str, Any]:
        return {
            "qid": trigger.qid,
            "session_id": session_id,
            "mode": mode,
            "ratio": run_args.override_ratio,
            "skipped": True,
            "skip_reason": reason,
            "d_turn_offset": offset,
            "d_ds_scored_qid": scored,
            "wall_sec": round(time.perf_counter() - start, 3),
        }

    pos_gap0 = (prefix["system_length"] + prefix["history_length"]) - prefix["cache_length"]
    expected_phys = prefix["cache_length"]  # build-time physical length
    # A stale target_override from any prefix builder must never reach the
    # continuation scoring: every t*+j row scores against later[j-1].answer
    # (_generate_with_prefix falls back to example.answer).
    prefix.pop("target_override", None)
    rows: List[Dict[str, Any]] = []
    prev = trigger
    with torch.inference_mode():
        for j in range(1, args.downstream_turns + 1):
            start = time.perf_counter()
            if j > len(later):
                # Span exhaustion is counted at EVERY unreached offset, so the
                # offset-2/3 denominators can be read off the rows.
                rows.append(_skip_row(j, "d_ds_no_subsequent_turn", None, start))
                continue
            nxt = later[j - 1]
            try:
                # Drop the prompt + generated tokens of turn j-1.
                _crop_cache(prefix["cache"], expected_phys)
                block, skip = _continuation_block(prev, nxt)
                if skip is not None:
                    rows.append(_skip_row(j, skip, nxt.qid, start))
                    break
                block_ids = HH._chat_template_ids(tokenizer, block) if block else []
                if block_ids:
                    # Templating a mid-conversation fragment must not inject a
                    # system header.  Deterministic per tokenizer, so a hard
                    # assert (implementation-invalid), never a counted skip.
                    first_msg_ids = HH._chat_template_ids(tokenizer, block[:1])
                    assert block_ids[: len(first_msg_ids)] == first_msg_ids, (
                        f"chat template injected a prologue ({trigger.qid} offset {j})"
                    )
                if (
                    expected_phys + len(block_ids) + args.max_prompt_tokens + args.max_new_tokens
                    > args.downstream_max_cache_tokens
                ):
                    rows.append(_skip_row(j, "d_ds_cache_over_budget", nxt.qid, start))
                    break
                block_sec = 0.0
                if block_ids:
                    block_input_ids = torch.tensor(
                        [block_ids], dtype=torch.long, device=model.device
                    )
                    cache, added, block_sec = HH._prefill_tokens_with_cache_maybe_gist(
                        model,
                        block_input_ids,
                        past_key_values=prefix["cache"],
                        past_length=prefix["system_length"] + prefix["history_length"],
                        attn_impl=run_args.generate_attn_impl,
                        use_gist=False,
                    )
                    prefix["cache"] = cache
                    prefix["history_length"] += added
                    expected_phys += added
                logical = prefix["system_length"] + prefix["history_length"]
                # Position invariant: logical - physical stays the build-time
                # constant along the whole continuation.
                assert logical - prefix["cache"].get_seq_length() == pos_gap0, (
                    f"position ledger drifted ({trigger.qid} offset {j}): "
                    f"{logical} - {prefix['cache'].get_seq_length()} != {pos_gap0}"
                )
                assert (
                    logical + args.max_prompt_tokens + args.max_new_tokens
                    < model.config.max_position_embeddings
                ), f"logical positions past max_position_embeddings ({trigger.qid} offset {j})"
                # Harness presentation of the next decision point.
                prefix["current_messages"] = HH._current_messages(nxt)
                metrics = HH._generate_with_prefix(model, tokenizer, nxt, prefix, run_args, mode)
                # HF generate never forwards the last sampled token, so the
                # cache gains prompt + generated - 1 entries; a copy-semantics
                # regression FATALs instead of silently measuring nothing.
                assert prefix["cache"].get_seq_length() == (
                    expected_phys + metrics["prompt_tokens"] + metrics["generated_tokens"] - 1
                ), (
                    f"generate cache-length tripwire ({trigger.qid} offset {j}): "
                    f"{prefix['cache'].get_seq_length()} != {expected_phys} + "
                    f"{metrics['prompt_tokens']} + {metrics['generated_tokens']} - 1"
                )
            except RuntimeError as error:
                if not HH._is_oom_error(error):
                    raise
                logger.warning(
                    "OOM: qid=%s downstream offset=%d — group retried on resume",
                    trigger.qid, j,
                )
                rows.append(_skip_row(j, "oom", nxt.qid, start))
                HH._clear_device_cache(args.device_type)
                break
            row = dict(metrics)
            row.update({
                "qid": trigger.qid,
                "session_id": session_id,
                "mode": mode,
                "ratio": run_args.override_ratio,
                "skipped": False,
                "d_turn_offset": j,
                "d_ds_scored_qid": nxt.qid,
                "d_ds_scored_span_index": int(nxt.qid.rsplit(":", 1)[1]),
                "d_ds_block_tokens": len(block_ids),
                "d_ds_block_messages": len(block),
                "d_ds_block_prefill_sec": round(block_sec, 4),
                "d_ds_cache_tokens": expected_phys,  # physical, pre-generate
                "d_ds_logical_tokens": logical,
                "d_ds_pos_gap": pos_gap0,
            })
            for key in DOWNSTREAM_CARRY_KEYS:
                if key in row0:
                    row[key] = row0[key]
            row["wall_sec"] = round(time.perf_counter() - start, 3)
            rows.append(row)
            prev = nxt
    return rows


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.base == "hybrid" and args.arm not in HYBRID_BASE_ARMS:
        raise SystemExit(
            f"FATAL: arm {args.arm!r} is not supported on --base hybrid "
            f"(supported: {sorted(HYBRID_BASE_ARMS)}). The splice/recompute "
            "families pre-date the hybrid base; re-extract triggers on the "
            "hybrid base before extending them."
        )
    if args.base == "c2kv" and args.hybrid_top_k != 3:
        raise SystemExit(
            "FATAL: --hybrid_top_k only applies to --base hybrid "
            "(the pure c2kv base compresses every history doc)."
        )
    frozen = _bind_frozen_state(args)
    code_sha: Optional[str] = None
    if args.downstream_turns:
        if args.downstream_turns < 0 or args.downstream_turns > DOWNSTREAM_MAX_TURNS:
            raise SystemExit(
                f"FATAL: --downstream_turns caps at {DOWNSTREAM_MAX_TURNS} and must "
                "not be negative (prereg addendum)"
            )
        if args.arm not in DOWNSTREAM_ARMS:
            raise SystemExit(
                "FATAL: downstream persistence runs none/sham/corr_re only "
                f"(prereg addendum); arm {args.arm!r} is refused"
            )
        if (
            not args.qids
            and not args.max_qids
            and os.environ.get("SKIP_DOWNSTREAM_SMOKE_CHECK") != "1"
        ):
            # The runner's smoke.ok gate (run_d_pilot_npu.sh), transplanted:
            # downstream launches are direct driver commands, so the driver
            # itself refuses a full run before the smoke sentinels pass.
            if not (args.downstream_smoke_ok and Path(args.downstream_smoke_ok).is_file()):
                raise SystemExit(
                    f"FATAL: {args.downstream_smoke_ok or '<--downstream_smoke_ok unset>'} "
                    "not found — the downstream smoke has not passed. Run the K=1 "
                    "smoke plus the three offset-0 identity sentinels first "
                    "(runbook 4a), point --downstream_smoke_ok at the marker they "
                    "write, or set SKIP_DOWNSTREAM_SMOKE_CHECK=1 to override "
                    "deliberately."
                )
        # A downstream row that cannot be traced to the code that produced it
        # defeats the prereg traceability design, so an unresolvable HEAD is
        # fatal.  Argument list, never a shell string.
        try:
            code_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"FATAL: cannot resolve git HEAD for d_code_sha: {error}")
        if args.max_samples_per_session != 0:
            logger.info(
                "downstream_turns=%d: forcing max_samples_per_session %d -> 0 so every "
                "harness-valid span of the eval sessions loads (the offset-0 identity "
                "sentinel certifies the trigger rows are unchanged)",
                args.downstream_turns, args.max_samples_per_session,
            )
            args.max_samples_per_session = 0
    manifest = frozen["manifest"]
    qids: List[str] = [str(q) for q in manifest.get("cw_qids", [])]
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in set(qids)]
        if unknown:
            raise SystemExit(f"FATAL: --qids not in the frozen trigger set: {unknown[:5]}")
        qids = wanted
    if args.max_qids:
        qids = qids[: args.max_qids]
    logger.info(
        "arm=%s mode=%s base=%s hybrid_top_k=%s qids=%d manifest_sha=%s… plan_sha=%s",
        args.arm, ARM_MODES[args.arm], args.base,
        args.hybrid_top_k if args.base == "hybrid" else "-",
        len(qids), frozen["manifest_sha256"][:16],
        (frozen["plan_sha256"] or "none")[:16],
    )

    hargs = _d_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, selection_skips = HH._load_examples(hargs, tokenizer)
    logger.info("source: %d examples, selection_skips=%s", len(examples), selection_skips)
    wanted_set = set(qids)
    by_qid: Dict[str, Any] = {}
    for example in examples:
        if example.qid in wanted_set and example.qid not in by_qid:
            by_qid[example.qid] = example
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} frozen qids not reproduced: {missing[:5]}")
    ordered = [by_qid[q] for q in qids]

    # Later decision points of each trigger session, in span-index order:
    # the harness's own examples from the same _load_examples call, so the
    # span filters and selection_filter have already been applied.
    session_spans: Dict[str, List[Tuple[int, Any]]] = {}
    if args.downstream_turns:
        for example in examples:
            session, _, span = example.qid.rpartition(":")
            session_spans.setdefault(session, []).append((int(span), example))
        for spans in session_spans.values():
            spans.sort(key=lambda item: item[0])

    HH.D_INTERVENE = _intervene_table(frozen["plan"], qids)
    HH.CORR_K_POLICY = args.corr_k_policy
    if args.corr_k_policy != "median":
        logger.info("K1 corr_k_policy=%s (plan k* pin bypassed)", args.corr_k_policy)
    if args.base == "hybrid":
        recorded_k = (manifest.get("kv_recipe") or {}).get("hybrid_top_k")
        if recorded_k is not None and int(recorded_k) != int(args.hybrid_top_k):
            raise SystemExit(
                f"FATAL: hybrid_top_k mismatch — the trigger manifest was frozen at "
                f"k={recorded_k}, this run passes {args.hybrid_top_k}. A different "
                "tail size changes which blocks are compressed, so k* would no "
                "longer name the block whose failure defined the trigger."
            )
    HH.D_HYBRID_TOP_K = args.hybrid_top_k if args.base == "hybrid" else None
    if args.arm in PLAN_REQUIRED_ARMS:
        without_payload = [q for q in qids if not HH.D_INTERVENE.get(q, {}).get("sham_token_ids")]
        if without_payload:
            raise SystemExit(
                f"FATAL: {len(without_payload)} qids have no sham payload: {without_payload[:5]}"
            )

    mode = ARM_MODES[args.arm]
    if args.base == "hybrid" and args.arm == "none":
        # The hybrid base IS the battery hybrid mode — same single builder
        # (_build_hybrid_prefix), so arm none on hybrid reproduces the plain
        # hybrid rows by construction (the combo self-check).
        mode = "hybrid"
    device = HH._setup_device(args.device_type)
    model_args = copy.copy(hargs)
    if args.arm == "full":
        model_args.mode = "full"
        model_args.model = HH._resolve_model_checkpoint(args.base_model or args.model)
    else:
        model_args.mode = "c2kv"
        model_args.model = HH._resolve_model_checkpoint(args.model)
    logger.info("Loading model %s (mode=%s, attn=%s)", model_args.model, model_args.mode, args.attn_impl)
    model = HH._load_model(model_args, tokenizer, device)
    attn_runtime = getattr(model.config, "_attn_implementation", None)
    logger.info("runtime attn impl=%s", attn_runtime)

    run_args = copy.copy(model_args)
    run_args.override_ratio = 1 if args.arm == "full" else args.ratio

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_qids(out_path, args.downstream_turns) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already done", len(done))

    remaining = len([e for e in ordered if e.qid not in done])
    n_written = 0
    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as handle:
        for example in ordered:
            if example.qid in done:
                continue
            start = time.perf_counter()
            prefix = None
            try:
                if args.downstream_turns:
                    row, prefix = HH._generate_one(
                        model, tokenizer, example, run_args, mode, return_state=True
                    )
                else:
                    row = HH._generate_one(model, tokenizer, example, run_args, mode)
            except RuntimeError as error:
                if not HH._is_oom_error(error):
                    raise
                logger.warning("OOM: arm=%s qid=%s — row skipped, retried on resume", args.arm, example.qid)
                row = HH._oom_row(example, mode, run_args.override_ratio)
                HH._clear_device_cache(args.device_type)
            wall = time.perf_counter() - start
            rows = [row]
            if args.downstream_turns:
                row["d_turn_offset"] = 0
                row["d_ds_scored_qid"] = example.qid
                if prefix is not None and not row.get("skipped"):
                    session, _, span = example.qid.rpartition(":")
                    later = [
                        ex for index, ex in session_spans.get(session, [])
                        if index > int(span)
                    ]
                    rows += _downstream_rows(
                        model, tokenizer, prefix, example, later, run_args, args, mode, row
                    )
                    rows[-1]["d_ds_terminal"] = True
                    rows[-1]["d_ds_offsets_available"] = min(args.downstream_turns, len(later))
            for item in rows:
                item["d_arm"] = args.arm
                item["d_mode"] = mode
                item["d_base"] = args.base
                item["d_hybrid_top_k"] = args.hybrid_top_k if args.base == "hybrid" else None
                item["bundle_manifest_sha256"] = frozen["manifest_sha256"]
                item["sham_plan_sha256"] = frozen["plan_sha256"]
                item["attn_impl_runtime"] = attn_runtime
                if args.downstream_turns:
                    item["d_downstream_turns"] = args.downstream_turns
                    item["d_code_sha"] = code_sha
            # Offset-0 wall covers the _generate_one call only; each
            # continuation row times its own segment.
            row["wall_sec"] = round(wall, 3)
            handle.write("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
            handle.flush()
            n_written += 1
            if row.get("skipped"):
                logger.info(
                    "[%d/%d] qid=%s SKIPPED (%s)", n_written, remaining, example.qid, row.get("skip_reason")
                )
            else:
                logger.info(
                    "[%d/%d] qid=%s tool_name_match=%s corr_tokens=%s sham_tokens=%s recompute=%s"
                    " cache_tokens=%s wall=%.1fs",
                    n_written, remaining, example.qid,
                    row.get("tool_name_match"), row.get("d_corr_span_tokens"),
                    row.get("d_sham_tokens"), row.get("d_recompute_tokens"),
                    row.get("cache_tokens"), row["wall_sec"],
                )
            if args.downstream_turns and len(rows) > 1:
                n_real = sum(1 for item in rows[1:] if not item.get("skipped"))
                logger.info(
                    "    downstream: %d scored, %d skipped (offsets 1..%d)",
                    n_real, len(rows) - 1 - n_real, args.downstream_turns,
                )
            HH._clear_device_cache(args.device_type)
    logger.info("Done. wrote %d groups -> %s", n_written, out_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_MODES), required=True)
    parser.add_argument(
        "--corr_k_policy",
        default="median",
        help="K1 erratum block selection: median (prereg default), last, or offset:<j>",
    )
    parser.add_argument(
        "--base", choices=["c2kv", "hybrid"], default="c2kv",
        help="Base prefix the D arms intervene on: pure c2kv (historical "
        "default) or hybrid tail-k raw (gist_first layout, docs/hybrid_spec.md). "
        "arm none on the hybrid base runs the battery hybrid mode itself.",
    )
    parser.add_argument(
        "--hybrid_top_k", type=int, default=3,
        help="Tail docs kept raw under --base hybrid.",
    )
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest.json")
    parser.add_argument("--bundles", default="./results/d/bundles_batch_tf.jsonl")
    parser.add_argument("--sham_plan", default="./configs/bdf_pilot/d_sham_plan.json")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="./checkpoints/qwen3-4b-agent-history-c2kv-npu")
    parser.add_argument("--base_model", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--downstream_turns", type=int, default=0,
        help="K continuation decision points after t* (0 = exactly current behavior; caps at 3).",
    )
    parser.add_argument(
        "--downstream_max_cache_tokens", type=int, default=28672,
        help="Physical KV slot budget for the continuation admission check (the single knob).",
    )
    parser.add_argument(
        "--downstream_smoke_ok", default="",
        help="Downstream smoke marker path; a full K>0 run refuses to start without it.",
    )
    parser.add_argument("--qids", default=None, help="Comma-separated subset of the frozen set (smoke).")
    parser.add_argument("--max_qids", type=int, default=0)
    parser.add_argument("--include_tools", default="True")
    parser.add_argument("--require_tool_call", default="False")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file", default=None)
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--min_doc_num", type=int, default=1)
    parser.add_argument("--max_history_tokens", type=int, default=12288)
    parser.add_argument("--max_system_length", type=int, default=4096)
    parser.add_argument("--max_prompt_tokens", type=int, default=1536)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--history_selection", default="tail")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
