"""D1' step-2 four-condition first-signal readout.

Scores, on a frozen validation manifest (agent/build_frozen_val_manifest.py),
the teacher-forced TARGET-SPAN loss of ONE conditioned checkpoint under four
condition variants, paired per sample:

  empty-Q         no condition window (the condition_len=0 path; no
                  `condition_input_ids` kwarg is passed at all);
  real-Q          the sample's own next-turn user query, built EXACTLY like
                  training: CompressHistoryDataset.preprocess_example emits
                  `condition_input_ids` from `example.condition_text` via
                  tokenizer.encode(text, add_special_tokens=False)[:window],
                  -100 padded (python/train/train_data_multiturn.py). The eval
                  forces C2KV_CONDITION_DROPOUT=0 so the window is never dropped;
  shuffled-Q      real-Q windows permuted among samples WITHIN THE BATCH
                  (seeded; deranged so no sample keeps its own window). With
                  --batch_size 1 a within-batch permutation is degenerate, so
                  the donor falls back to a seeded pick of another manifest
                  sample (any session) — see _donor_maps;
  other-session-Q real-Q window of a sample from a DIFFERENT session (seeded,
                  deterministic per-qid mapping; falls back to any other
                  sample if the manifest has a single session).

Condition variants never change the number of stored/gist tokens (T3 parity in
tests/test_condition_interleave.py): gist slots depend only on the turn-t doc
tokens, and the shuffled/other-session arms only replace the window CONTENT
(the -100-padded width is identical for every arm).

Forward/label/position bookkeeping mirrors GistMultiDocTrainer.compute_loss
(python/train/trainer.py:261) exactly — per-sample system KV cache, context
trim, position_ids offset by past_length + real doc tokens (condition tokens
excluded, as upstream), label-driven logits_to_keep — except that labels never
enter the model (so no batch loss and no reconstruction path; identical logits
at eval) and the token-level CE is reduced PER SAMPLE over the answer/target
span only (labels -100 elsewhere). Teacher-forced only: NO generation anywhere.

The trainer samples the gist ratio from C2KV_GIST_TRAIN_RATIOS inside
process_context_input_ids; this script pins that env var to --ratio so the
eval is deterministic at the requested ratio.

Rows (jsonl, append + flush per row, --resume):
  {qid, session_id, loss_empty, loss_real, loss_shuffled, loss_other,
   n_target_tokens, skipped, skip_reason}

Heavy imports (torch/transformers/repo model+train modules) are guarded so the
script exits cleanly with a message on torch-less machines.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(REPO_ROOT / "python" / "inference"))
    sys.path.insert(0, str(REPO_ROOT / "agent"))

_HEAVY_IMPORT_ERROR = None
try:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from eval_agent_history_c2kv import (
        _is_oom_error,
        _load_tokenizer,
        _resolve_model_checkpoint,
    )
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device
    from train.train_data_multiturn import (
        AgentLLMTracesCompressHistorySource,
        CompressHistoryDataset,
        _pad,
    )
except ImportError as error:  # pragma: no cover - depends on the host env
    _HEAVY_IMPORT_ERROR = error

# Variant names -> row keys.
VARIANTS = ("empty", "real", "shuffled", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D1' four-condition teacher-forced target-span loss readout."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Conditioned checkpoint dir (or a run dir containing checkpoint-* subdirs; "
        "resolved like agent/eval_agent_history_c2kv.py does).",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --checkpoint.")
    parser.add_argument("--val_manifest", default="configs/d1prime_frozen_val.json")
    parser.add_argument("--out", required=True, help="Output jsonl path (append + flush per row).")
    parser.add_argument("--resume", action="store_true", help="Skip qids already present in --out.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Device spec, e.g. npu:0 / cuda:0 / cpu. The base type drives device_map.",
    )
    parser.add_argument("--attn_impl", default="eager", help="Attention impl for system + gist phases.")
    parser.add_argument("--ratio", type=int, default=4, help="Fixed gist compression ratio.")
    parser.add_argument("--max_examples", type=int, default=0, help="Cap manifest qids (0 = all).")
    parser.add_argument(
        "--condition_window_tokens",
        type=int,
        default=int(os.environ.get("C2KV_CONDITION_WINDOW_TOKENS", "256") or 256),
        help="D1' condition-window token budget; MUST match the conditioned training run.",
    )
    parser.add_argument(
        "--variant_seed",
        type=int,
        default=0,
        help="Seed for the shuffled/other-session donor mappings.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--dataset_path", default=None, help="Override manifest['created_from'].")
    # Training-side feature geometry (agent/train_agent_history_c2kv_npu.sh
    # defaults, NOT the generation-eval defaults): the readout measures the
    # trained condition, so features are built with the training geometry.
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--min_doc_num", type=int, default=1)
    parser.add_argument("--max_doc_num", type=int, default=12)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_system_length", type=int, default=4096)
    parser.add_argument("--history_selection", choices=["head", "tail"], default="tail")
    args = parser.parse_args()
    # Fields expected by the _load_model / _load_tokenizer idioms imported from
    # the round-1 eval scripts.
    args.model = args.checkpoint
    args.base_model = None
    args.mode = "c2kv"  # any non-("full"/"truncate") value takes the gist load path
    args.untrained_c2kv = False
    args.baseline_model_class = "gist"
    args.generate_attn_impl = args.attn_impl
    return args


def _pin_condition_env(args: argparse.Namespace) -> None:
    """Pin the D1' env gates BEFORE the source/features are built.

    - C2KV_CONDITION_WINDOW_TOKENS: the source only populates `condition_text`
      when this is > 0, and preprocess_example sizes the padded window with it.
    - C2KV_CONDITION_DROPOUT: forced to 0 — dropout is a training-time
      augmentation and must never fire during measurement.
    - C2KV_GIST_TRAIN_RATIOS: process_context_input_ids samples the gist ratio
      from this env var on the trainer path; pinning it to --ratio makes the
      eval deterministic.
    """
    os.environ["C2KV_CONDITION_WINDOW_TOKENS"] = str(args.condition_window_tokens)
    os.environ["C2KV_CONDITION_DROPOUT"] = "0"
    os.environ["C2KV_GIST_TRAIN_RATIOS"] = str(args.ratio)


def _load_manifest(path: str) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest.get("qids"), list) or not manifest["qids"]:
        raise ValueError(f"Manifest {path} has no qids")
    return manifest


def _build_source(filters: Dict[str, Any], dataset_path: str) -> Any:
    """Rebuild the SAME example pipeline the manifest was frozen from."""
    return AgentLLMTracesCompressHistorySource(
        dataset_path,
        split=filters.get("split", "eval"),
        eval_ratio=float(filters.get("eval_ratio", 0.1)),
        split_seed=int(filters.get("split_seed", 42)),
        split_manifest_file=filters.get("split_manifest_file"),
        split_manifest_name=filters.get("split_manifest_name", "subset_disjoint"),
        max_samples_per_session=filters.get("max_samples_per_session", 4),
        include_tools=bool(filters.get("include_tools", True)),
    )


def _build_features(example: Any, tokenizer: Any, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Build trainer-format features via the training dataset's own staticmethod.

    Returns None when the example is invalid under the training-time rules
    (short history, empty current/answer, truncation edge cases).
    """
    row = CompressHistoryDataset.preprocess_example(
        example,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_doc_length=args.max_doc_length,
        min_doc_num=args.min_doc_num,
        max_doc_num=args.max_doc_num,
        max_system_length=args.max_system_length,
        history_selection=args.history_selection,
        full_history_doc_num=0,
        split_oversized_history_docs=True,
    )
    if row is None:
        return None
    row.pop("dynamic", None)
    return row


def _condition_window_ids(tokenizer: Any, condition_text: str, window: int) -> List[int]:
    """Tokenize a condition window EXACTLY like training
    (python/train/train_data_multiturn.py preprocess_example): raw token ids
    without any chat-template wrapper, truncated to the window, -100 padded to
    the fixed width. Dropout is a training-time augmentation and is never
    applied here (C2KV_CONDITION_DROPOUT is pinned to 0 in _pin_condition_env).
    """
    condition_ids: List[int] = []
    if condition_text:
        condition_ids = tokenizer.encode(condition_text, add_special_tokens=False)[:window]
    return _pad(condition_ids, window, -100)


def _collate(rows: Sequence[Dict[str, Any]], device: str) -> Dict[str, Any]:
    """Stack the fixed-width feature fields (all widths are set by the geometry
    args, so no padding is needed — same shapes the trainer's collator yields)."""
    batch = {
        "system_input_ids": torch.tensor([row["system_input_ids"] for row in rows], dtype=torch.long),
        "context_input_ids": torch.tensor([row["context_input_ids"] for row in rows], dtype=torch.long),
        "input_ids": torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long),
        "attention_mask": torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long),
    }
    return {key: value.to(device) for key, value in batch.items()}


def _build_system_kv(model: Any, system_input_ids: Any, attn_impl: str) -> Tuple[Any, Any, int]:
    """Mirror of GistMultiDocTrainer._build_system_kv (python/train/trainer.py:219).

    Left-pads the real system tokens to the batch max real length so the padded
    past length is uniform, prefills once, and returns (cache, mask, L_sys).
    """
    device = next(model.parameters()).device
    system_input_ids = system_input_ids.to(device)
    real_mask = system_input_ids != -100
    real_lens = real_mask.sum(dim=1)
    batch_size = system_input_ids.shape[0]
    L_sys = int(real_lens.max().item())
    pad_id = model.model.config.pad_token_id
    if pad_id is None:
        pad_id = 0
    left_ids = system_input_ids.new_full((batch_size, L_sys), pad_id)
    system_mask = system_input_ids.new_zeros((batch_size, L_sys))
    for i in range(batch_size):
        n = int(real_lens[i].item())
        if n == 0:
            continue
        left_ids[i, L_sys - n :] = system_input_ids[i][real_mask[i]]
        system_mask[i, L_sys - n :] = 1
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    with torch.inference_mode():
        outputs = model(left_ids, attention_mask=system_mask, use_cache=True, logits_to_keep=1)
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, system_mask, L_sys


def _target_span_losses(
    model: Any,
    batch: Dict[str, Any],
    max_doc_length: int,
    attn_impl: str,
    condition_input_ids: Optional[Any] = None,
) -> Tuple[List[float], List[int]]:
    """One trainer-style forward; per-sample mean CE over the target span.

    Mirrors GistMultiDocTrainer.compute_loss (python/train/trainer.py:261)
    step by step: system KV -> context trim -> past/position bookkeeping ->
    label-driven logits_to_keep -> gist-attn forward. Unlike the trainer,
    labels are NOT passed to the model (identical logits at eval: labels only
    feed the loss and the train-time reconstruction path); the CE is computed
    here per sample with the standard causal shift.
    """
    device = batch["input_ids"].device
    batch_size, _doc_total_len = batch["context_input_ids"].shape
    context_masks = batch["context_input_ids"] != -100
    # Per-sample REAL doc token count (condition tokens are appended inside
    # process_context_input_ids and never counted here — mirrors the trainer).
    context_token_counts = context_masks.sum(dim=1)
    system_kv, system_mask, past_length = _build_system_kv(model, batch["system_input_ids"], attn_impl)

    # Trim batch-local doc padding before generate_gist (trainer.py:287-291).
    context_input_ids = batch["context_input_ids"].reshape((batch_size, -1, max_doc_length))
    doc_lengths = (context_input_ids != -100).sum(dim=2)
    max_doc_active_length = int(doc_lengths.max().item()) if doc_lengths.numel() else 0
    if 0 < max_doc_active_length < max_doc_length:
        context_input_ids = context_input_ids[:, :, :max_doc_active_length]

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # Trim batch-local label padding (trainer.py:295-300).
    active_lengths = attention_mask.sum(dim=1)
    max_active_length = int(active_lengths.max().item())
    if 0 < max_active_length < input_ids.shape[1]:
        input_ids = input_ids[:, :max_active_length]
        attention_mask = attention_mask[:, :max_active_length]
        labels = labels[:, :max_active_length]
    input_length = input_ids.shape[1]

    # Turn positions start after system KV + real doc tokens (trainer.py:301-306).
    position_ids = torch.arange(input_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
    for i, seqlen in enumerate(context_token_counts.tolist()):
        position_ids[i] += past_length + seqlen

    # Label-driven logits slicing (trainer.py:307-315).
    label_mask = labels != -100
    if not label_mask.any():
        raise ValueError("Batch has no supervised label tokens after preprocessing/truncation.")
    first_label_positions = label_mask.float().argmax(dim=1)
    first_label_position = int(first_label_positions[label_mask.any(dim=1)].min().item())
    logits_start = max(0, first_label_position - 1)
    kept_labels = labels
    logits_to_keep = 0
    if logits_start > 0:
        kept_labels = labels[:, logits_start:]
        logits_to_keep = input_length - logits_start

    inner_model = model.model if hasattr(model, "model") else model
    original_attn_impl = inner_model.config._attn_implementation
    inner_model.config._attn_implementation = attn_impl
    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": system_kv,
        "past_attention_mask": system_mask,
        "context_input_ids": context_input_ids,
        "logits_to_keep": logits_to_keep,
    }
    if condition_input_ids is not None:
        forward_kwargs["condition_input_ids"] = condition_input_ids.to(device)
    try:
        with torch.inference_mode():
            outputs = model(**forward_kwargs)
    finally:
        inner_model.config._attn_implementation = original_attn_impl

    # Per-sample target-span CE with the standard causal shift (the same shift
    # the model's loss_function applies), averaged over target tokens of EACH
    # sample — no batch normalization. Logits are upcast to float32 like
    # ForCausalLMLoss does.
    logits = outputs.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = kept_labels[:, 1:]
    token_nll = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        ignore_index=-100,
        reduction="none",
    )
    counts = (shift_labels != -100).sum(dim=1)
    losses = token_nll.sum(dim=1) / counts.clamp(min=1)
    return losses.detach().cpu().tolist(), counts.detach().cpu().tolist()


def _donor_maps(
    qids: Sequence[str],
    session_ids: Sequence[str],
    batch_size: int,
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Seeded deterministic donor index maps for the shuffled/other-session arms.

    shuffled: batches are the FIXED consecutive chunks of the sorted qid list
    (composition does not depend on --resume state, so mappings are stable
    across resumes). Within a batch, real windows are permuted with a
    derangement (no sample keeps its own window). batch_size == 1 falls back to
    a seeded pick of another manifest sample (any session).

    other:    seeded per-qid pick among samples of a DIFFERENT session; falls
    back to any other sample for a single-session manifest.
    """
    shuffled_map: Dict[int, int] = {}
    other_map: Dict[int, int] = {}
    n = len(qids)
    all_indices = list(range(n))
    for batch_index, start in enumerate(range(0, n, max(1, batch_size))):
        idxs = list(range(start, min(start + max(1, batch_size), n)))
        rng = random.Random(f"c2kv-d1prime-shuffled::{seed}::{batch_index}")
        if len(idxs) == 1:
            i = idxs[0]
            donor_rng = random.Random(f"c2kv-d1prime-shuffled1::{seed}::{qids[i]}")
            candidates = [j for j in all_indices if j != i]
            shuffled_map[i] = donor_rng.choice(candidates) if candidates else i
            continue
        donors = idxs[:]
        rng.shuffle(donors)
        attempts = 0
        while any(a == b for a, b in zip(donors, idxs)) and attempts < 1000:
            rng.shuffle(donors)
            attempts += 1
        if any(a == b for a, b in zip(donors, idxs)):
            # Rotation by one is always a derangement for len > 1.
            donors = idxs[1:] + idxs[:1]
        for i, donor in zip(idxs, donors):
            shuffled_map[i] = donor
    for i, qid in enumerate(qids):
        donor_rng = random.Random(f"c2kv-d1prime-other::{seed}::{qid}")
        candidates = [j for j in all_indices if session_ids[j] != session_ids[i] and j != i]
        if not candidates:
            candidates = [j for j in all_indices if j != i]
        other_map[i] = donor_rng.choice(candidates) if candidates else i
    return shuffled_map, other_map


def _load_done_qids(path: str) -> set:
    done = set()
    out_path = Path(path)
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "qid" in row:
                done.add(row["qid"])
    return done


def _skip_row(qid: str, session_id: Optional[str], reason: str) -> Dict[str, Any]:
    return {
        "qid": qid,
        "session_id": session_id,
        "loss_empty": None,
        "loss_real": None,
        "loss_shuffled": None,
        "loss_other": None,
        "n_target_tokens": 0,
        "skipped": True,
        "skip_reason": reason,
    }


def _write_rows(handle: Any, rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _clear_device_cache(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def evaluate(args: argparse.Namespace) -> Dict[str, int]:
    if _HEAVY_IMPORT_ERROR is not None:
        raise SystemExit(
            "eval_condition_variants.py needs torch/transformers and the repo "
            "model+train packages (run on the NPU box env); import failed with: "
            f"{type(_HEAVY_IMPORT_ERROR).__name__}: {_HEAVY_IMPORT_ERROR}"
        )
    _pin_condition_env(args)

    manifest = _load_manifest(args.val_manifest)
    qids = sorted(manifest["qids"])
    if args.max_examples and args.max_examples > 0:
        qids = qids[: args.max_examples]
    dataset_path = args.dataset_path or manifest["created_from"]
    logger.info("Manifest %s: %d qids to score (dataset=%s)", args.val_manifest, len(qids), dataset_path)

    # Resolve checkpoint + load tokenizer/model via the round-1 eval idioms.
    args.model = _resolve_model_checkpoint(args.checkpoint)
    args.checkpoint = args.model
    device_type, _, device_index = args.device.partition(":")
    device = _setup_device(device_type)
    if device_index:
        if device == "npu" and hasattr(torch, "npu"):
            torch.npu.set_device(int(device_index))
        elif device == "cuda":
            torch.cuda.set_device(int(device_index))
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = _load_model(args, tokenizer, device)
    logger.info("Loaded conditioned checkpoint %s on %s", args.model, device)

    # Rebuild the same example pipeline and index by qid.
    source = _build_source(manifest.get("filters", {}), dataset_path)
    examples_by_qid = {example.qid: example for example in source}
    missing = [qid for qid in qids if qid not in examples_by_qid]
    if missing:
        logger.warning("%d manifest qids not reproduced by the source pipeline (skipped)", len(missing))

    session_ids = [
        (qid.rsplit(":", 1)[0] if ":" in qid else qid) for qid in qids
    ]
    manifest_index = {qid: index for index, qid in enumerate(qids)}
    batch_size = max(1, args.batch_size)
    shuffled_map, other_map = _donor_maps(qids, session_ids, batch_size, args.variant_seed)

    done_qids = _load_done_qids(args.out) if args.resume else set()
    if done_qids:
        logger.info("Resuming: %d qids already present in %s", len(done_qids), args.out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    stats = {"scored": 0, "skipped": 0, "batches": 0}
    checked_window_construction = False
    with out_path.open(mode, encoding="utf-8") as handle:
        batches = [qids[start : start + batch_size] for start in range(0, len(qids), batch_size)]
        for batch_qids in tqdm(batches, desc="condition-variants"):
            pending = [qid for qid in batch_qids if qid not in done_qids]
            if not pending:
                continue
            rows_out: List[Dict[str, Any]] = []
            kept: List[Dict[str, Any]] = []
            for qid in pending:
                session_id = qid.rsplit(":", 1)[0] if ":" in qid else qid
                example = examples_by_qid.get(qid)
                if example is None:
                    rows_out.append(_skip_row(qid, session_id, "missing_example"))
                    continue
                if not (getattr(example, "condition_text", "") or "").strip():
                    rows_out.append(_skip_row(qid, session_id, "empty_condition_window"))
                    continue
                features = _build_features(example, tokenizer, args)
                if features is None:
                    rows_out.append(_skip_row(qid, session_id, "invalid_sample"))
                    continue
                if not checked_window_construction:
                    # One-time consistency check: the standalone window helper
                    # must reproduce the training feature's condition row
                    # byte-for-byte (dropout pinned to 0).
                    assert _condition_window_ids(
                        tokenizer, example.condition_text, args.condition_window_tokens
                    ) == list(features["condition_input_ids"]), (
                        "condition window construction drifted from preprocess_example"
                    )
                    checked_window_construction = True
                kept.append({"qid": qid, "session_id": session_id, "features": features})
            if kept:
                batch = _collate([item["features"] for item in kept], device)
                # Donor windows are tokenized straight from the donor examples'
                # condition_text (identical construction to the training
                # feature row) — no full donor feature build needed, so a donor
                # that is itself invalid under the training rules still works.
                condition_rows_by_variant: Dict[str, Any] = {
                    "real": torch.tensor(
                        [item["features"]["condition_input_ids"] for item in kept], dtype=torch.long
                    )
                }
                other_fallback_rows: List[bool] = []
                for variant in ("shuffled", "other"):
                    donor_map = shuffled_map if variant == "shuffled" else other_map
                    donor_windows = []
                    for item in kept:
                        index = manifest_index[item["qid"]]
                        donor = donor_map[index]
                        donor_example = examples_by_qid[qids[donor]]
                        donor_windows.append(
                            _condition_window_ids(
                                tokenizer,
                                donor_example.condition_text,
                                args.condition_window_tokens,
                            )
                        )
                        if variant == "other":
                            other_fallback_rows.append(session_ids[donor] == session_ids[index])
                    condition_rows_by_variant[variant] = torch.tensor(donor_windows, dtype=torch.long)
                try:
                    losses_by_variant: Dict[str, List[float]] = {}
                    counts: Optional[List[int]] = None
                    for variant in VARIANTS:
                        # empty-Q: no condition_input_ids kwarg at all (the
                        # condition_len=0 path); conditioned arms share the
                        # fixed -100-padded window width (T3 token parity).
                        variant_losses, variant_counts = _target_span_losses(
                            model,
                            batch,
                            args.max_doc_length,
                            args.attn_impl,
                            condition_input_ids=condition_rows_by_variant.get(variant),
                        )
                        losses_by_variant[variant] = variant_losses
                        if counts is None:
                            counts = variant_counts
                    for row_pos, item in enumerate(kept):
                        n_target = int(counts[row_pos]) if counts is not None else 0
                        if n_target <= 0:
                            rows_out.append(_skip_row(item["qid"], item["session_id"], "no_label_tokens"))
                            continue
                        row = {
                            "qid": item["qid"],
                            "session_id": item["session_id"],
                            "loss_empty": losses_by_variant["empty"][row_pos],
                            "loss_real": losses_by_variant["real"][row_pos],
                            "loss_shuffled": losses_by_variant["shuffled"][row_pos],
                            "loss_other": losses_by_variant["other"][row_pos],
                            "n_target_tokens": n_target,
                            "skipped": False,
                            "skip_reason": None,
                        }
                        if other_fallback_rows[row_pos]:
                            # Single-session manifest: this row's other-session
                            # donor fell back to any other sample.
                            row["other_session_fallback"] = True
                        rows_out.append(row)
                        stats["scored"] += 1
                except RuntimeError as error:
                    _clear_device_cache(device)
                    reason = "oom" if _is_oom_error(error) else f"error:{type(error).__name__}"
                    logger.exception("Batch failed (%s); marking %d rows skipped", reason, len(kept))
                    for item in kept:
                        rows_out.append(_skip_row(item["qid"], item["session_id"], reason))
                except Exception as error:  # noqa: BLE001 - a bad batch must not kill a resumable run
                    _clear_device_cache(device)
                    reason = f"error:{type(error).__name__}"
                    logger.exception("Batch failed (%s); marking %d rows skipped", reason, len(kept))
                    for item in kept:
                        rows_out.append(_skip_row(item["qid"], item["session_id"], reason))
                finally:
                    del batch
                    _clear_device_cache(device)
            stats["skipped"] += sum(1 for row in rows_out if row.get("skipped"))
            stats["batches"] += 1
            _write_rows(handle, rows_out)
    return stats


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    stats = evaluate(args)
    logger.info(
        "Done in %.1fs: scored=%d skipped=%d batches=%d -> %s",
        time.perf_counter() - start,
        stats["scored"],
        stats["skipped"],
        stats["batches"],
        args.out,
    )


if __name__ == "__main__":
    main()
