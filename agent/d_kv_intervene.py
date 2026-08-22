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

Usage (NPU server, repo root):
  python agent/d_kv_intervene.py --arm corr_re \
      --output_file ./outputs/d_pilot/d_corr_re.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
}
PLAN_REQUIRED_ARMS = {"sham"}
PLAN_USING_ARMS = {"sham", "corr", "corr_re", "corr_all", "sham_mech"}


def _load_done_qids(path: Path) -> set:
    """Only NON-skipped rows count as done (skipped rows are retried)."""
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "qid" in row and not row.get("skipped"):
                    done.add(row["qid"])
    return done


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

    Two ways the grid can silently differ:

    * ``max_doc_length`` -- the history harness convention is 768 while the
      joint harness uses 1024; a manifest frozen under one and a run launched
      under the other chunk the same history differently.
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


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    frozen = _bind_frozen_state(args)
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
        "arm=%s mode=%s qids=%d manifest_sha=%s… plan_sha=%s",
        args.arm, ARM_MODES[args.arm], len(qids), frozen["manifest_sha256"][:16],
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

    HH.D_INTERVENE = _intervene_table(frozen["plan"], qids)
    if args.arm in PLAN_REQUIRED_ARMS:
        without_payload = [q for q in qids if not HH.D_INTERVENE.get(q, {}).get("sham_token_ids")]
        if without_payload:
            raise SystemExit(
                f"FATAL: {len(without_payload)} qids have no sham payload: {without_payload[:5]}"
            )

    mode = ARM_MODES[args.arm]
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
    done = _load_done_qids(out_path) if args.resume else set()
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
            try:
                row = HH._generate_one(model, tokenizer, example, run_args, mode)
            except RuntimeError as error:
                if not HH._is_oom_error(error):
                    raise
                logger.warning("OOM: arm=%s qid=%s — row skipped, retried on resume", args.arm, example.qid)
                row = HH._oom_row(example, mode, run_args.override_ratio)
                HH._clear_device_cache(args.device_type)
            row["d_arm"] = args.arm
            row["d_mode"] = mode
            row["bundle_manifest_sha256"] = frozen["manifest_sha256"]
            row["sham_plan_sha256"] = frozen["plan_sha256"]
            row["attn_impl_runtime"] = attn_runtime
            row["wall_sec"] = round(time.perf_counter() - start, 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
            HH._clear_device_cache(args.device_type)
    logger.info("Done. wrote %d rows -> %s", n_written, out_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_MODES), required=True)
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
