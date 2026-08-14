#!/usr/bin/env python
"""Clean teacher-forced forced-prefix logp recompute for the frozen S4 subset.

Round-1 S4 scored the forced action-prefix logp AFTER generation, and
model.generate appended prompt+generated KV to the prefix cache in place, so
all round-1 logp_prefix_* / delta_logp_prefix fields are void. This runner
recomputes them cleanly on the frozen qid subset (agent/extract_s4_frozen_qids.py):
NO generation function is ever called -- each sample is two prefix builds
(c2kv@RATIO and full-KV, both with the c2kv checkpoint model, mirroring round-1
arm C in agent/eval_agent_history_c2kv.py::_generate_one) plus two
teacher-forced scoring forwards via _prefix_continuation_logp.

Examples are loaded exactly as the S4 eval does: same data source, split
(subset_disjoint), filters and defaults (see _build_eval_args below, which
mirrors agent/eval_agent_history_s4_npu.sh -> eval parse_args defaults).

Append + flush per sample; --resume skips qids already done in --out. Per-sample
failures never abort the run: OOM (transient on the shared box) is recorded and
retried on the next launch; anything else is recorded as a permanent skip.

Requires torch/(torch_npu)/transformers -- run on the NPU server; on machines
without them it exits with a clear message instead of a stack trace.

Example:
  python agent/recompute_s4_logp.py \
    --qids_file configs/s4_frozen_qids.json \
    --out outputs/s4_logp_recompute.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Heavy imports are guarded so the script degrades gracefully on CPU-only
# machines (eval_agent_history_c2kv imports torch/transformers/tqdm at module
# top; train_data_multiturn pulls pandas/pyarrow for the parquet shards).
try:
    import torch  # noqa: F401  (imported for the early, clear failure only)

    from eval_agent_history_c2kv import (
        _build_c2kv_prefix,
        _build_full_or_truncate_prefix,
        _clear_device_cache,
        _current_messages,
        _force_action_prefix_ids,
        _is_oom_error,
        _load_examples,
        _load_tokenizer,
        _prefix_continuation_logp,
        _resolve_model_checkpoint,
    )
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device
    from train.train_data_multiturn import _chat_template_ids

    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as error:  # pragma: no cover - exercised only off-server
    _IMPORT_ERROR = error


def _git_commit() -> str:
    """Best-effort git commit hash of this repo; 'unknown' on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[1],
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _build_eval_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Minimal Namespace replicating the S4 eval configuration.

    Every field below is consumed by the prefix builders / scorers / example
    loader of agent/eval_agent_history_c2kv.py. Values are the eval script's
    parse_args() defaults (lines 3050-3217) unless the S4 runner
    agent/eval_agent_history_s4_npu.sh overrides them -- each is annotated.
    """
    return argparse.Namespace(
        # -- consumed by _build_c2kv_prefix / _build_full_or_truncate_prefix --
        max_doc_length=768,  # eval default
        max_doc_num=16,  # eval default
        min_doc_num=1,  # eval default
        max_history_tokens=12288,  # eval default
        max_system_length=4096,  # eval default
        max_prompt_tokens=1536,  # eval default (tail truncation of the current prompt)
        max_baseline_input_tokens=16000,  # eval default (full-KV total-length gate)
        history_selection="tail",  # eval default
        truncate_selection="tail",  # eval default; only used for mode="truncate", kept for completeness
        split_oversized_history_docs=True,  # eval default (read via getattr in _history_messages)
        system_attn_impl=cli.attn_impl,  # S4 runner: NPU_ATTN_IMPL, default "eager"
        gist_attn_impl=cli.attn_impl,  # S4 runner: NPU_ATTN_IMPL
        generate_attn_impl=cli.attn_impl,  # S4 runner: NPU_ATTN_IMPL; also the logp scoring attn impl
        override_ratio=cli.ratio,  # S4 runner: RATIO, default 4
        force_action_prefix=True,  # arms C/D semantics: score the forced 'Action:\n<tool_call>\n' prefix
        # -- consumed by _load_examples (mirrors eval_agent_history_s4_npu.sh) --
        dataset_path=cli.dataset_path,  # S4 runner: DATASET_PATH
        split="eval",  # S4 runner: SPLIT
        eval_ratio=0.1,  # S4 runner: EVAL_RATIO
        split_seed=42,  # S4 runner: SPLIT_SEED
        split_manifest_file=None,  # S4 runner: SPLIT_MANIFEST_FILE (unset)
        split_manifest_name="subset_disjoint",  # S4 runner: SPLIT_NAME
        max_samples_per_session=4,  # S4 runner: MAX_SAMPLES_PER_SESSION
        max_source_examples=None,  # eval default (unset)
        require_tool_call=False,  # eval default
        max_input_chars=None,  # eval default (unset)
        max_answer_chars=None,  # eval default (unset)
        include_tools=True,  # S4 runner: INCLUDE_TOOLS=True
        prefix_history_doc_num=None,  # eval default (unset)
        prefix_history_exact=False,  # eval default
        selection_filter="c2kv",  # eval default (same c2kv-feasibility filter as the eval)
        sample_seed=None,  # eval default (S4 runner leaves SAMPLE_SEED unset)
        max_examples=0,  # S4 runner: MAX_EXAMPLES=0 -> no cap; our --max_examples applies after qid filtering
        # -- consumed by _load_tokenizer / _load_model --
        tokenizer=cli.tokenizer_path,  # S4 runner: TOKENIZER_PATH=BASE_MODEL
        model=cli.model_path,  # S4 runner: MODEL_PATH (c2kv checkpoint, as in arms B/C)
        base_model=None,  # not needed: mode="c2kv" loads args.model; None also skips it in _load_tokenizer
        mode="c2kv",  # arm B/C model-loading path in _load_model
        dtype="bf16",  # eval default
        baseline_model_class="auto",  # eval default (only read for full/truncate modes)
        untrained_c2kv=False,  # eval default
    )


def _load_done_qids(out_path: str, qids: Set[str]) -> Set[str]:
    """Qids already done in --out.

    Done means: a clean row (both logps non-null), or a skipped row whose
    reason is not "oom" -- non-OOM failures are deterministic gates or
    permanent errors, so retrying them would just spin the window-gated loop.
    OOM rows are intentionally NOT done: they are retried once more free HBM
    is available.
    """
    done: Set[str] = set()
    path = Path(out_path)
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid not in qids or qid in done:
                continue
            clean = row.get("logp_prefix_c2kv") is not None and row.get("logp_prefix_full") is not None
            permanent_skip = bool(row.get("skipped")) and row.get("skip_reason") != "oom"
            if clean or permanent_skip:
                done.add(qid)
    return done


def _score_example(model: Any, tokenizer: Any, example: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Score one example: clean teacher-forced forced-prefix logp under c2kv and full KV.

    Never calls any generation function, so the prefix caches are never
    polluted in place (the round-1 bug). _prefix_continuation_logp deep-copies
    the cache before its scoring forward.
    """
    row: Dict[str, Any] = {
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "logp_prefix_c2kv": None,
        "logp_prefix_full": None,
        "delta_logp_prefix": None,
        "cache_tokens_c2kv": None,
        "cache_tokens_full": None,
        "doc_tokens": None,
        "skipped": False,
        "skip_reason": None,
    }
    try:
        prefix_c2kv, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
        if prefix_c2kv is None:
            # Deterministic gate (history_tokens>12288, history_docs>16, ...): permanent skip.
            row.update(skipped=True, skip_reason=skip_reason)
            return row
        row["cache_tokens_c2kv"] = prefix_c2kv.get("cache_length")
        row["doc_tokens"] = prefix_c2kv.get("doc_tokens")

        # Same prompt construction as the eval's forced-logp block
        # (_generate_one in agent/eval_agent_history_c2kv.py): current messages
        # with generation prompt, tail-truncated to max_prompt_tokens.
        prompt_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
        if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
            prompt_ids = prompt_ids[-args.max_prompt_tokens :]
        forced_ids = _force_action_prefix_ids(tokenizer, args)
        row["logp_prefix_c2kv"] = _prefix_continuation_logp(
            model, prefix_c2kv, prompt_ids, forced_ids, args.generate_attn_impl
        )

        prefix_full, full_skip = _build_full_or_truncate_prefix(model, tokenizer, example, args, "full")
        if prefix_full is None:
            # Same handling as round-1 arm C, which kept the row with
            # logp_prefix_full unset when the full build failed. We additionally
            # mark it skipped so the resume logic treats the deterministic gate
            # (e.g. baseline_input_tokens>16000) as done; the analysis only uses
            # rows with both logps non-null, and the c2kv-only value stays
            # available in the row.
            row.update(skipped=True, skip_reason=f"full_prefix:{full_skip}")
            return row
        row["cache_tokens_full"] = prefix_full.get("cache_length")
        row["logp_prefix_full"] = _prefix_continuation_logp(
            model, prefix_full, prompt_ids, forced_ids, args.generate_attn_impl
        )
        if row["logp_prefix_c2kv"] is not None and row["logp_prefix_full"] is not None:
            # Same delta convention as the eval: full minus c2kv.
            row["delta_logp_prefix"] = row["logp_prefix_full"] - row["logp_prefix_c2kv"]
    except RuntimeError as error:
        if _is_oom_error(error):
            # Transient on the shared box: retried by the window-gated runner.
            row.update(skipped=True, skip_reason="oom")
        else:
            row.update(skipped=True, skip_reason=f"error:{type(error).__name__}: {error}"[:500])
    except Exception as error:  # non-runtime failures (tokenization, data) are permanent
        row.update(skipped=True, skip_reason=f"error:{type(error).__name__}: {error}"[:500])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qids_file", default="./configs/s4_frozen_qids.json")
    parser.add_argument("--out", default="./outputs/s4_logp_recompute.jsonl")
    parser.add_argument("--resume", action="store_true", help="Skip qids already done in --out.")
    parser.add_argument("--max_examples", type=int, default=0, help="Cap scored examples after qid filtering (0 = no cap).")
    parser.add_argument("--model_path", default="./checkpoints/qwen3-4b-agent-history-c2kv-npu")
    parser.add_argument("--tokenizer_path", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--device", default="npu:0", help="Scoring device; the type prefix selects the backend.")
    parser.add_argument("--attn_impl", default="eager", help="Attention impl for system/gist/generate (S4 runner used eager).")
    parser.add_argument("--ratio", type=int, default=4, help="C2KV compression ratio (S4 arms B/C used 4).")
    cli = parser.parse_args()

    if _IMPORT_ERROR is not None:
        sys.exit(
            "recompute_s4_logp.py requires torch/(torch_npu)/transformers and the repo's "
            "python packages; run it on the NPU server (see agent/recompute_s4_logp_npu.sh). "
            f"Import failure: {_IMPORT_ERROR}"
        )

    with open(cli.qids_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    qids: Set[str] = set(payload["qids"])
    logger.info("Frozen subset: %d qids from %s", len(qids), cli.qids_file)

    eval_args = _build_eval_args(cli)
    device_type = cli.device.split(":", 1)[0]
    device = _setup_device(device_type)
    if device in {"npu", "cuda"} and ":" in cli.device:
        device = cli.device  # keep the explicit index, e.g. "npu:0"
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples, selection_skips = _load_examples(eval_args, tokenizer)
    logger.info("Loaded %d eval examples; selection_skips=%s", len(examples), selection_skips)
    examples = [example for example in examples if example.qid in qids]
    missing = len(qids) - len({example.qid for example in examples})
    if missing:
        logger.warning("%d/%d frozen qids are not in the loaded eval split", missing, len(qids))
    if cli.max_examples:
        examples = examples[: cli.max_examples]

    done_qids = _load_done_qids(cli.out, qids) if cli.resume else set()
    todo = [example for example in examples if example.qid not in done_qids]
    logger.info(
        "To score: %d/%d frozen qids (%d already done in %s)",
        len(todo),
        len(examples),
        len(done_qids),
        cli.out,
    )

    logger.info("Loading model %s on %s (git %s)", eval_args.model, device, _git_commit())
    model = _load_model(eval_args, tokenizer, device)

    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_clean = 0
    n_skipped = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, example in enumerate(todo, start=1):
            row = _score_example(model, tokenizer, example, eval_args)
            _clear_device_cache(device_type)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if row["logp_prefix_c2kv"] is not None and row["logp_prefix_full"] is not None:
                n_clean += 1
            if row["skipped"]:
                n_skipped += 1
                logger.warning("qid=%s skipped: %s", example.qid, row["skip_reason"])
            if index % 10 == 0:
                logger.info("progress: %d/%d scored (%d clean, %d skipped)", index, len(todo), n_clean, n_skipped)

    summary = {
        "git_commit": _git_commit(),
        "qids_file": cli.qids_file,
        "out": cli.out,
        "n_clean": n_clean,
        "n_skipped": n_skipped,
        "n_scored_this_run": len(todo),
        "clean_fraction": f"{n_clean}/{len(todo)}",
    }
    logger.info("done: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
