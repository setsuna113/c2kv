# -*- coding: utf-8 -*-
"""t34 U1b (b) -- Retrieval Head, Port 0 (arXiv 2404.15574), digest 4.5.

Port 0 is the prerequisite AND a result in its own right: run the paper's
Sec. 2 needle detection on RAW (uncompressed) contexts for the base FC model
and each C2KV checkpoint, and ask -- with the paper's own criterion, Pearson
r > 0.8 on the per-head score heatmaps -- whether C2KV training preserves the
retrieval-head set.  Ports 1 (locator restricted to the head subset) and 2
(sink_ratio / gist_ratio triggers) belong to unit U1a and consume the frozen
head-set json this module writes.

RUNBOOK

  1. NPU   python agent/t34_retrieval_head.py detect \\
             --model <path or hub id> --tag base \\
             --out results/t34/rethead/base.json
           repeat with --tag fixed_joint and --tag checkpoint-1088.
           (>=3 (q,k,x) tuples x 10 depths x the length grid; greedy decode;
            eager attention -- fused kernels return no probabilities.)
  2. here  PYTHONIOENCODING=utf-8 python agent/t34_retrieval_head.py compare \\
             --scores results/t34/rethead/base.json \\
             --scores results/t34/rethead/fixed_joint.json \\
             --scores results/t34/rethead/checkpoint-1088.json \\
             --heatmap-dir results/t34/rethead/heatmaps \\
             --out results/t34/rethead/compare.json
           Pairwise Pearson r of the per-head scores + the r > 0.8 verdict.
  3. here  PYTHONIOENCODING=utf-8 python agent/t34_retrieval_head.py head-set \\
             --scores results/t34/rethead/fixed_joint.json \\
             --out configs/t34/retrieval_heads_fixed_joint.json
           Frozen (layer, head) list, sha256 in the file, for U1a ports 1/2.

WIRING (step 1): decoding uses unit U1a's
``t34_attention.AttentionRowCapture`` and its ``argmax_tensor() -> [L, H, Q]``;
U1a is imported LAZILY inside :func:`run_needle_trial` so this module imports on
a torch-free box.  The capture runs in ``prefill_last_n`` mode with
``last_n = 1`` (``NEEDLE_QUERY_MODE``) because that is the mode whose rows cover
EVERY generated token; plain ``"decode"`` skips the prompt forward and its rows
start at generated token 1, which :func:`run_needle_trial` handles by shifting.  The
needle context is a plain raw prefill (no gist grid, no ``use_gist``): the
paper's criterion compares TOKEN IDENTITY at the most-attended position, which
is undefined over gist vectors -- only the head SET transfers to the
compressed arm (card, 'Mapping', row 3).

Everything except :func:`run_needle_trial` / :func:`detect_on_model` is pure
numpy and is unit-tested here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "needle detection grid (Sec. 2)",
        "paper": "2404.15574",
        "what": "20 context lengths uniformly sampled from 1K-50K become a "
                "5-point grid capped at the battery scale "
                "(max_doc_length 768 x max_doc_num 16 ~= 12K); 10 insertion "
                "depths and >=3 (q,k,x) tuples are kept exactly.",
        "why": "digest 4.5 Port 0; eager attention OOMs at 16K on our box "
               "(37.17 GiB), and a head set detected outside the battery's "
               "operating range would not describe the rows we score.",
    },
    {
        "method": "retrieval score |g_h & k| / |k| (Sec. 2)",
        "paper": "2404.15574",
        "what": "g_h and k are taken as SETS OF TOKEN IDS (the paper writes "
                "'the set of tokens h copy-pasted').  A positional variant "
                "(count_mode='positions') is kept because the paper does not "
                "disambiguate repeated needle tokens.",
        "why": "The ambiguity is the paper's; both readings are cheap, so we "
               "compute both rather than guess (card notes no results table "
               "exists for Sec. 2).",
    },
    {
        "method": "retrieval-head threshold 0.1",
        "paper": "2404.15574",
        "what": "Kept verbatim as RETRIEVAL_HEAD_THRESHOLD = 0.1 and reported "
                "as the paper's unvalidated constant; the head-set json also "
                "carries the full score map so any other cut is recomputable.",
        "why": "Card, 'Decision rule': the 0.1 has an interpretive "
               "justification only -- no sweep, no validation set, no "
               "sensitivity analysis.",
    },
    {
        "method": "haystack corpus",
        "paper": "2404.15574",
        "what": "The haystack x is ONE filler sentence repeated to the target "
                "length, not the paper's natural long documents (its NIAH "
                "setup uses real essay text).  The needle stays semantically "
                "irrelevant to the filler, which is the property Sec. 2 "
                "actually requires of x.",
        "why": "No licensed long-form corpus is pinned in this worktree and "
               "inventing one would put an unrecorded asset in the loop.  The "
               "cost is real and is NOT papered over: a periodic haystack is a "
               "degenerate attention environment, so the ABSOLUTE retrieval "
               "scores (and therefore the 0.1 cut) are not comparable with the "
               "paper's; only the cross-checkpoint Pearson r, which is a "
               "within-corpus comparison, is read.",
    },
    {
        "method": "retrieval score, positional variant",
        "paper": "2404.15574",
        "what": "count_mode='positions' is clipped at 1.0 "
                "(min(1, |events| / |k|)).",
        "why": "Without the clip a head that copies the same needle token more "
                "often than the needle is long would score above 1 and would "
                "not be comparable with the set-semantics primary; the paper "
                "defines a ratio of a subset to k, which is bounded by 1.",
    },
    {
        "method": "cross-checkpoint stability (Sec. 3.3)",
        "paper": "2404.15574",
        "what": "Pearson r of the per-head score heatmaps is computed base vs "
                "each C2KV checkpoint and pairwise between checkpoints; the "
                "verdict line r > 0.8 = 'head set preserved' is the paper's "
                "own base-vs-variant benchmark (cross-family r < 0.1 is the "
                "reverse anchor).",
        "why": "digest 4.5 Port 0 asks for exactly this read.  The residual "
               "risk it does NOT cover -- whether those heads do the same "
               "thing on GIST vectors -- is stated, not papered over.",
    },
    {
        "method": "downstream masking (Sec. 4)",
        "paper": "2404.15574",
        "what": "Not ported.  We read heads, never mask them.",
        "why": "The masking evidence is one model, no seeds, no error bars "
               "(card, Open question 6); and our endpoint is detection, not "
               "an intervention on the base circuit.",
    },
]

#: arXiv 2404.15574 Sec. 2: "retrieval score > 0.1 defines a retrieval head".
#: The card records that this constant has an interpretive justification only.
RETRIEVAL_HEAD_THRESHOLD = 0.1

#: Sec. 3.3's base-vs-variant benchmark; cross-family r < 0.1 is the anchor.
HEAD_SET_PRESERVED_R = 0.8
CROSS_FAMILY_R = 0.1

#: Battery operating range: max_doc_length 768 x max_doc_num 16.
BATTERY_MAX_CONTEXT = 768 * 16
DEFAULT_LENGTH_GRID: Tuple[int, ...] = (1024, 2048, 4096, 8192, 12288)
DEFAULT_DEPTHS: Tuple[float, ...] = tuple(round(0.1 * i, 3) for i in range(10))


# --------------------------------------------------------------------------
# the copy-paste rule (arXiv 2404.15574 Sec. 2)
# --------------------------------------------------------------------------

def copy_paste_tokens(
    argmax_positions: Sequence[int],
    generated_ids: Sequence[int],
    input_ids: Sequence[int],
    needle_ids: Sequence[int],
    needle_span: Tuple[int, int],
) -> List[int]:
    """Tokens one head copy-pasted, per arXiv 2404.15574 Sec. 2.

    A head scores a copy-paste on generated token ``w`` iff
      (1) ``w`` is a token of the needle ``k``, AND
      (2) ``x_j == w`` where ``j = argmax(a)`` and ``j`` lies inside the
          needle's position range ``i_q``.

    Returns the copied token ids in generation order (``g_h`` before
    de-duplication, so both count modes of :func:`retrieval_score` are
    derivable from it).
    """
    lo, hi = int(needle_span[0]), int(needle_span[1])
    needle_set = set(int(t) for t in needle_ids)
    x = list(input_ids)
    out: List[int] = []
    for t, w in enumerate(generated_ids):
        if t >= len(argmax_positions):
            break
        w = int(w)
        if w not in needle_set:                      # criterion (1)
            continue
        j = int(argmax_positions[t])
        if not (lo <= j < hi):                       # criterion (2), position
            continue
        if j < 0 or j >= len(x) or int(x[j]) != w:   # criterion (2), identity
            continue
        out.append(w)
    return out


def retrieval_score(copied: Sequence[int], needle_ids: Sequence[int],
                    *, count_mode: str = "set") -> float:
    """``retrieval score(h) = |g_h & k| / |k|`` (arXiv 2404.15574 Sec. 2).

    ``count_mode="set"`` (primary) reads ``g_h`` and ``k`` as sets of token
    ids, the paper's wording.  ``count_mode="positions"`` divides the number of
    copy-paste events by the needle length, the alternative reading the paper
    leaves open (see DEVIATIONS).
    """
    k = [int(t) for t in needle_ids]
    if not k:
        return float("nan")
    if count_mode == "set":
        return len(set(int(t) for t in copied) & set(k)) / len(set(k))
    if count_mode == "positions":
        return min(1.0, len(copied) / len(k))
    raise ValueError(f"unknown count_mode {count_mode!r}")


def trial_head_scores(
    argmax_tensor: np.ndarray,
    generated_ids: Sequence[int],
    input_ids: Sequence[int],
    needle_ids: Sequence[int],
    needle_span: Tuple[int, int],
    *,
    count_mode: str = "set",
) -> np.ndarray:
    """Per-head retrieval score for ONE needle trial (arXiv 2404.15574 Sec. 2).

    ``argmax_tensor`` is U1a's ``argmax_tensor() -> [L, H, T]`` collected over
    the ``T`` greedily decoded tokens (query_mode="decode").  Returns ``[L, H]``.
    """
    a = np.asarray(argmax_tensor)
    if a.ndim != 3:
        raise ValueError(f"argmax_tensor must be [L,H,T], got {a.shape}")
    n_layers, n_heads, n_steps = a.shape
    out = np.zeros((n_layers, n_heads), dtype=np.float64)
    gen = list(generated_ids)[:n_steps]
    for l in range(n_layers):
        for h in range(n_heads):
            copied = copy_paste_tokens(a[l, h, :].tolist(), gen, input_ids,
                                       needle_ids, needle_span)
            out[l, h] = retrieval_score(copied, needle_ids, count_mode=count_mode)
    return out


def aggregate_head_scores(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Average per-head scores over tuples x depths x lengths (Sec. 2:
    "average the per-test retrieval score per head")."""
    if not trials:
        raise ValueError("no trials")
    stack = np.stack([np.asarray(t, dtype=np.float64) for t in trials], axis=0)
    return stack.mean(axis=0)


def retrieval_head_mask(table: np.ndarray,
                        threshold: float = RETRIEVAL_HEAD_THRESHOLD) -> np.ndarray:
    """``score > 0.1`` (arXiv 2404.15574 Sec. 2; unvalidated constant)."""
    return np.asarray(table, dtype=np.float64) > float(threshold)


def head_list(table: np.ndarray,
              threshold: float = RETRIEVAL_HEAD_THRESHOLD,
              top_m: Optional[int] = None) -> List[Tuple[int, int]]:
    """(layer, head) pairs above the arXiv 2404.15574 Sec. 2 threshold, ranked
    by score (ties -> index).  This list is the only thing that transfers to
    the compressed arm: the SCORE is undefined over gist vectors."""
    t = np.asarray(table, dtype=np.float64)
    pairs = [(int(l), int(h), float(t[l, h]))
             for l in range(t.shape[0]) for h in range(t.shape[1])
             if t[l, h] > float(threshold)]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    if top_m is not None:
        pairs = pairs[: int(top_m)]
    return [(l, h) for l, h, _ in pairs]


def sparsity_report(table: np.ndarray,
                    threshold: float = RETRIEVAL_HEAD_THRESHOLD) -> Dict[str, Any]:
    """Our analogue of the paper's Sec. 3.1 sparsity prose (their numbers --
    3-6% above 0.1, 45-73% exactly zero -- are NOT imported, only the shape of
    the statistic)."""
    t = np.asarray(table, dtype=np.float64)
    n = t.size
    return {
        "n_heads_total": int(n),
        "n_above_threshold": int((t > threshold).sum()),
        "frac_above_threshold": float((t > threshold).mean()),
        "frac_exactly_zero": float((t == 0.0).mean()),
        "frac_in_0_to_threshold": float(((t > 0.0) & (t <= threshold)).mean()),
        "max": float(t.max()), "mean": float(t.mean()),
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------
# cross-checkpoint stability (arXiv 2404.15574 Sec. 3.3)
# --------------------------------------------------------------------------

def pearson_head_maps(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson r between two per-head score heatmaps (Sec. 3.3's statistic).

    None when either map is constant (r undefined) -- never 0.0, which would
    read as 'cross-family'."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"heatmap shapes differ: {a.shape} vs {b.shape}")
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_pearson(maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """All checkpoint pairs, with arXiv 2404.15574 Sec. 3.3's own verdict line
    attached to each (r > 0.8 preserved / r < 0.1 cross-family)."""
    names = sorted(maps)
    out: Dict[str, Any] = {"pairs": [], "criterion_preserved_r": HEAD_SET_PRESERVED_R,
                           "criterion_cross_family_r": CROSS_FAMILY_R}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = pearson_head_maps(maps[names[i]], maps[names[j]])
            out["pairs"].append({
                "a": names[i], "b": names[j], "pearson_r": r,
                "head_set_preserved": (None if r is None else bool(r > HEAD_SET_PRESERVED_R)),
                "overlap_top20": head_overlap_top(maps[names[i]], maps[names[j]], 20),
            })
    return out


def head_overlap_top(a: np.ndarray, b: np.ndarray, m: int) -> int:
    """Top-m head overlap -- a rank-space companion to the Pearson r of arXiv
    2404.15574 Sec. 3.3 (which is computed over the whole heatmap)."""
    def _top(t: np.ndarray) -> Set[Tuple[int, int]]:
        arr = np.asarray(t, dtype=np.float64)
        flat = [(int(l), int(h), float(arr[l, h]))
                for l in range(arr.shape[0]) for h in range(arr.shape[1])]
        flat.sort(key=lambda x: (-x[2], x[0], x[1]))
        return {(l, h) for l, h, _ in flat[:m]}
    return len(_top(a) & _top(b))


# --------------------------------------------------------------------------
# needle construction (arXiv 2404.15574 Sec. 2)
# --------------------------------------------------------------------------

#: >=3 (q, k, x-filler) tuples, needle semantically irrelevant to the haystack
#: so a correct answer must be COPIED, not recalled (Sec. 2).
NEEDLE_TUPLES: Tuple[Dict[str, str], ...] = (
    {"name": "magic_number",
     "question": "What is the magic maintenance number for vault 7?",
     "needle": "The magic maintenance number for vault 7 is 48-Q-13905.",
     "answer": "48-Q-13905"},
    {"name": "codeword",
     "question": "What is the codeword assigned to the courier?",
     "needle": "The codeword assigned to the courier is BRASS-LANTERN-914.",
     "answer": "BRASS-LANTERN-914"},
    {"name": "coordinate",
     "question": "What are the storage coordinates of crate DL-2?",
     "needle": "The storage coordinates of crate DL-2 are 51.7734 by -0.9218.",
     "answer": "51.7734 by -0.9218"},
)

FILLER_SENTENCE = (
    "The inventory clerk logged the routine transfer and filed the duplicate "
    "manifest in the archive cabinet before the shift handover. "
)


def build_needle_context(tokenizer: Any, tuple_spec: Dict[str, str],
                         length_tokens: int, depth: float) -> Dict[str, Any]:
    """Build ``x`` at ``length_tokens`` with the needle inserted at ``depth``
    (arXiv 2404.15574 Sec. 2: 10 insertion depths, needle semantically
    irrelevant to the haystack).

    Returns the token ids, the needle's absolute position span ``i_q`` inside
    the context, and the needle token ids ``k``.  Depth is the fraction of the
    filler that precedes the needle (the paper's 10 insertion depths)."""
    if length_tokens > BATTERY_MAX_CONTEXT:
        raise ValueError(
            f"length {length_tokens} exceeds the battery operating range "
            f"{BATTERY_MAX_CONTEXT} (see DEVIATIONS: eager attention OOMs)")
    filler_ids = tokenizer(FILLER_SENTENCE, add_special_tokens=False)["input_ids"]
    needle_ids = tokenizer(tuple_spec["needle"], add_special_tokens=False)["input_ids"]
    n_filler = max(0, int(length_tokens) - len(needle_ids))
    reps = (n_filler // max(1, len(filler_ids))) + 1
    hay = (filler_ids * reps)[:n_filler]
    cut = int(round(float(depth) * len(hay)))
    ids = hay[:cut] + needle_ids + hay[cut:]
    return {
        "input_ids": ids,
        "needle_span": (cut, cut + len(needle_ids)),
        "needle_ids": needle_ids,
        "question": tuple_spec["question"],
        "answer": tuple_spec["answer"],
        "tuple": tuple_spec["name"],
        "length_tokens": len(ids),
        "depth": float(depth),
    }


# --------------------------------------------------------------------------
# NPU-side detection (torch + U1a imported lazily)
# --------------------------------------------------------------------------

def _load_u1a():
    import t34_attention  # noqa: WPS433 -- deliberately lazy
    return t34_attention


#: Capture mode for the needle decode.  ``prefill_last_n`` with ``last_n = 1``
#: is the mode that covers EVERY generated token: U1a records the last row of
#: the prompt forward (which emits generated token 0) plus every 1-token decode
#: row.  Plain ``"decode"`` starts at generated token 1, so its rows have to be
#: aligned with an offset of 1 -- see :func:`run_needle_trial`.
NEEDLE_QUERY_MODE = "prefill_last_n"


def run_needle_trial(
    trial: Dict[str, Any],
    *,
    decode_fn,
    attention_module: Any = None,
    count_mode: str = "set",
    max_new_tokens: int = 32,
    query_mode: str = NEEDLE_QUERY_MODE,
) -> np.ndarray:
    """Greedy-decode one needle trial and score every head (arXiv 2404.15574
    Sec. 2 detection procedure).

    ``decode_fn(trial, capture, max_new_tokens)`` runs the harness forward with
    ``capture`` installed and returns the generated token ids; ``capture`` is
    U1a's ``AttentionRowCapture`` whose ``argmax_tensor()`` is ``[L, H, T]``.

    Row alignment (this is where a silent off-by-one would fake copy-paste
    events): with ``query_mode="prefill_last_n"`` row ``t`` emits generated
    token ``t``; with ``query_mode="decode"`` the prompt forward is not
    captured, so row ``t`` emits generated token ``t + 1`` and the generated
    sequence is shifted before scoring.
    """
    A = attention_module if attention_module is not None else _load_u1a()
    key_map = A.KeyClassMap(
        system_len=0,
        doc_spans=[trial["needle_span"]],
        raw_tail_span=None,
        query_span=(len(trial["input_ids"]), len(trial["input_ids"])),
        generated_start=len(trial["input_ids"]),
    )
    capture = A.AttentionRowCapture(key_map, query_mode=query_mode, last_n=1,
                                    layers=None)
    # install/remove is the decode driver's job (it owns the model); the driver
    # is expected to use the capture as a context manager or call
    # install(model) / remove() around its own generate loop.
    generated = list(decode_fn(trial, capture, max_new_tokens))
    argmax = np.asarray(capture.argmax_tensor())
    if query_mode == "decode":
        generated = generated[1:]
    n = min(argmax.shape[2], len(generated)) if argmax.ndim == 3 else 0
    argmax = argmax[:, :, :n]
    generated = generated[:n]
    return trial_head_scores(argmax, generated, trial["input_ids"],
                             trial["needle_ids"], trial["needle_span"],
                             count_mode=count_mode)


def make_greedy_decode_fn(model: Any, tokenizer: Any):  # pragma: no cover - NPU
    """Default ``decode_fn`` for :func:`detect_on_model`: greedy decoding of the
    answer with U1a's capture installed (arXiv 2404.15574 Sec. 2 decodes the
    answer greedily and reads the per-head attention of each generated token).

    WIRING: raw prefill, no gist grid, eager attention -- the copy-paste rule
    needs token identity at the most-attended position, which only exists over
    raw tokens.
    """
    import torch

    def decode_fn(trial: Dict[str, Any], capture: Any, max_new_tokens: int):
        prompt_ids = list(trial["input_ids"]) + tokenizer(
            "\nQuestion: " + trial["question"] + "\nAnswer:",
            add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        generated: List[int] = []
        capture.install(model)
        try:
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                generated.append(nxt)
                for _ in range(int(max_new_tokens) - 1):
                    step = torch.tensor([[nxt]], dtype=torch.long, device=model.device)
                    out = model(input_ids=step, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = int(out.logits[0, -1].argmax())
                    generated.append(nxt)
                    if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
                        break
        finally:
            try:
                capture.remove()
            except TypeError:
                capture.remove(model)
        return generated

    return decode_fn


def load_model_for_detection(model_path: str, *, device_type: str = "npu",
                             mode: str = "auto"):  # pragma: no cover - NPU only
    """Load a checkpoint with EAGER attention (fused kernels return no
    probabilities -- arXiv 2404.15574's signal is unreachable without them)
    through the harness loader (``t34_model_loader``): the gist checkpoints
    need the repo's model class (plain AutoModelForCausalLM drops every gist
    parameter) and the model must live on the NPU, not the CPU."""
    from t34_model_loader import load_model_and_tokenizer

    model, tok, resolved = load_model_and_tokenizer(
        model_path, device_type=device_type, attn_impl="eager", mode=mode)
    print(json.dumps({"loader": "harness", "mode": resolved, "device_type": device_type,
                      "model": model_path}))
    return model, tok


def detect_on_model(
    tokenizer: Any,
    *,
    decode_fn,
    tuples: Sequence[Dict[str, str]] = NEEDLE_TUPLES,
    lengths: Sequence[int] = DEFAULT_LENGTH_GRID,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    attention_module: Any = None,
    count_mode: str = "set",
    max_new_tokens: int = 32,
    query_mode: str = NEEDLE_QUERY_MODE,
) -> Dict[str, Any]:
    """arXiv 2404.15574 Sec. 2 detection, scaled to the battery range.

    >=3 (q, k, x) tuples x the length grid x 10 insertion depths, greedy
    decoding, per-head averaging.  ``decode_fn`` is the only model-dependent
    argument, so this loop is exercised in the tests with a fake.
    """
    trials: List[np.ndarray] = []
    index: List[Dict[str, Any]] = []
    for spec in tuples:
        for length in lengths:
            for depth in depths:
                trial = build_needle_context(tokenizer, spec, length, depth)
                score = run_needle_trial(trial, decode_fn=decode_fn,
                                         attention_module=attention_module,
                                         count_mode=count_mode,
                                         max_new_tokens=max_new_tokens,
                                         query_mode=query_mode)
                trials.append(score)
                index.append({"tuple": spec["name"], "length": length, "depth": depth})
    table = aggregate_head_scores(trials)
    return {
        "score_map": [[float(v) for v in row] for row in table],
        "n_trials": len(trials),
        "trial_index": index,
        "count_mode": count_mode,
        "sparsity": sparsity_report(table),
        "query_mode": query_mode,
        "length_grid": [int(x) for x in lengths],
        "depths": [float(d) for d in depths],
        "paper": "2404.15574 Sec. 2 (detection), Sec. 3.3 (stability)",
    }


# --------------------------------------------------------------------------
# heatmaps
# --------------------------------------------------------------------------

def write_heatmaps(maps: Dict[str, np.ndarray], out_dir: Path) -> List[str]:
    """Per-checkpoint retrieval-score heatmaps -- the figures arXiv 2404.15574
    Sec. 3.1/3.3 report sparsity and stability from.  Always the raw array
    (npy + json), plus a PNG when matplotlib is importable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, table in maps.items():
        arr = np.asarray(table, dtype=np.float64)
        npy = out_dir / f"{name}.npy"
        np.save(npy, arr)
        written.append(str(npy))
        js = out_dir / f"{name}.json"
        js.write_text(json.dumps([[float(v) for v in row] for row in arr]),
                      encoding="utf-8")
        written.append(str(js))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(arr, aspect="auto", origin="lower")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"retrieval score -- {name}")
        fig.colorbar(im, ax=ax)
        png = out_dir / f"{name}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(str(png))
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_score_map(path: Path) -> Tuple[str, np.ndarray]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    tag = obj.get("tag") or Path(path).stem
    return tag, np.asarray(obj["score_map"], dtype=np.float64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for the arXiv 2404.15574 port 0; see the RUNBOOK at the top of this
    module for the execution order and which stage runs where."""
    ap = argparse.ArgumentParser(
        description="Retrieval Head port 0: needle detection + head-set stability "
                    "(arXiv 2404.15574)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="[NPU] Sec. 2 needle detection on one checkpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="base | fixed_joint | checkpoint-1088")
    p.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTH_GRID))
    p.add_argument("--depths", type=int, default=len(DEFAULT_DEPTHS))
    p.add_argument("--count-mode", choices=("set", "positions"), default="set")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device_type", default="npu", help="npu (default) | cpu")
    p.add_argument("--mode", default="auto", choices=("auto", "full", "c2kv"),
                   help="harness model class: auto = c2kv for gist checkpoints, full otherwise")
    p.add_argument("--out", required=True)

    p = sub.add_parser("compare", help="pairwise Pearson r + heatmaps")
    p.add_argument("--scores", action="append", required=True)
    p.add_argument("--heatmap-dir", default=None)
    p.add_argument("--out", required=True)

    p = sub.add_parser("head-set", help="freeze the (layer, head) list for U1a")
    p.add_argument("--scores", required=True)
    p.add_argument("--threshold", type=float, default=RETRIEVAL_HEAD_THRESHOLD)
    p.add_argument("--top-m", type=int, default=None)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "detect":  # pragma: no cover - needs torch + a checkpoint
        model, tok = load_model_for_detection(args.model, device_type=args.device_type,
                                              mode=args.mode)
        lengths = [int(x) for x in args.lengths.split(",")]
        depths = tuple(round(i / args.depths, 4) for i in range(args.depths))
        table = detect_on_model(
            tok, decode_fn=make_greedy_decode_fn(model, tok),
            lengths=lengths, depths=depths, count_mode=args.count_mode,
            max_new_tokens=args.max_new_tokens)
        table["tag"] = args.tag
        table["model"] = args.model
        sha = C.freeze_json(Path(args.out), table)
        print(f"detect tag={args.tag} trials={table['n_trials']} "
              f"above_thresh={table['sparsity']['n_above_threshold']}"
              f"/{table['sparsity']['n_heads_total']} sha256={sha}")
        return 0

    if args.cmd == "compare":
        maps: Dict[str, np.ndarray] = {}
        for path in args.scores:
            tag, table = _load_score_map(Path(path))
            maps[tag] = table
        report = pairwise_pearson(maps)
        report["sparsity"] = {k: sparsity_report(v) for k, v in maps.items()}
        report["source_sha256"] = {Path(p).name: C.sha256_file(Path(p))
                                   for p in args.scores}
        report["residual_risk"] = (
            "Pearson r only checks the head set on RAW input.  Whether these "
            "heads do the same thing on GIST vectors has no cheap test "
            "(2404.15574 card, Open question 2; digest 4.5 Port 0 risk).")
        report["deviations"] = DEVIATIONS
        if args.heatmap_dir:
            report["heatmaps"] = write_heatmaps(maps, Path(args.heatmap_dir))
        sha = C.freeze_json(Path(args.out), report)
        for pair in report["pairs"]:
            print(f"{pair['a']} vs {pair['b']}: r={pair['pearson_r']} "
                  f"preserved={pair['head_set_preserved']} "
                  f"top20_overlap={pair['overlap_top20']}")
        print("report sha256=" + sha)
        return 0

    if args.cmd == "head-set":
        tag, table = _load_score_map(Path(args.scores))
        heads = head_list(table, args.threshold, args.top_m)
        obj = {
            "tag": tag,
            "heads": [[l, h] for l, h in heads],
            "threshold": float(args.threshold),
            "threshold_provenance": (
                "arXiv 2404.15574 Sec. 2; the paper gives an interpretive "
                "justification only -- no sweep, no validation set."),
            "top_m": args.top_m,
            "n_heads": len(heads),
            "sparsity": sparsity_report(table, args.threshold),
            "score_map": [[float(v) for v in row] for row in table],
            "source_sha256": C.sha256_file(Path(args.scores)),
            "consumed_by": "unit U1a t34_attention ports 1 (locator) and 2 (sink/gist ratio)",
        }
        sha = C.freeze_json(Path(args.out), obj)
        print(f"head-set n={len(heads)} of {table.size} heads "
              f"({obj['sparsity']['frac_above_threshold']:.4f}) sha256={sha}")
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
