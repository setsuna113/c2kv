# -*- coding: utf-8 -*-
"""t34 U9 - digest S4.10 "learned token / head": ALIEN, MemGen sparsity, CORA, Self-REF/[RESET].

Four migrations, one module, no torch at scoring time.

  (A) ALIEN (arXiv 2505.15443) - a d x |C| head on the penultimate-layer hidden
      state at the tool-name decode position, initialised from the ``lm_head``
      rows of each candidate tool name's FIRST token so that at init
      ``U_ALIEN == U_Entropy`` exactly, trained with
      ``L = BCE(U_ALIEN, e) + alpha * L_reg + beta * L2SP``.
  (B) MemGen memory trigger (arXiv 2509.24704) - only the reward-adaptive
      sparsity penalty is portable: a separate logistic head on the same
      hidden features with ``L = BCE(y, p) + lambda * mean(max(0, p - pbar))``,
      ``pbar`` = mean predicted fire probability on the C->C rows, recomputed
      each epoch.  Decision is ``p > tau``, never Bernoulli.
  (C) CORA (arXiv 2604.09155) - only the block splitter (whole ``session_id``
      blocks, second level by toolset) and the Conformal-Risk-Control
      calibrator ``tau_hat = sup{tau : (1/(n+1))(sum_i L(Z_i;tau) + 1) <= alpha}``
      with ``L(Z;tau) = harm(Z) * 1{s <= tau}``.  No Guardian is trained.
  (D) Self-REF (2410.13284) / ``[RESET]`` (2409.14586) - NO arm.  Only the
      label-structure utilities they motivate: three-valued supervision
      {C->W, W->W, correct} with alpha-downsampling applied to W->W, the
      first-divergence index for the ``[RESET]`` positive-prefix construction,
      and the enumeration of the third (never-tried) negative-pair class from
      the 120 W->C rows.  Every one of these is LABEL-side and is named
      ``*_label_*``.

Scoring contract (digest S4.0).  Evaluation frame = the 161-row trigger subset
(C->W 93 / C->C 68, 100 sessions); chance AP is that frame's prevalence 0.578,
never the 900-frame base rate 0.1033.  Baseline = parse-failure only.  Every CI
is a session-clustered bootstrap over the sessions present in the frame.  The
bare-entropy baseline (raw full-vocabulary entropy at the tool-name token) is
the FIRST row of every table this module prints - ALIEN's own ranking puts
plain entropy second of thirteen methods.

Feature-frame note.  ``alien_u_*_oof`` / ``memgen_p_oof`` are OUT-OF-FOLD
scores of heads that were trained on labels in other sessions' folds.  They are
model outputs, not raw observables; the inputs to the head are compressed-arm
hiddens only.  Orientations for every emitted column live in
``configs/t34/orientations_heads.json`` (+1 = higher is riskier).  ``pool_size_c``
and ``ctrl_generated_tokens`` / ``ctrl_cap_hit`` are nuisance/control columns,
declared with an orientation so the guard can see them, never a headline signal.

RUNBOOK (execution order; [NPU] runs on the Ascend server, [here] on this box)

  0. [NPU, torch] the t33 capture rerun already wrote
       <capture_dir>/<arm>/p0.steps.jsonl        (steps/entropy/top5/ic/anchors)
       <capture_dir>/<arm>/p0_XXXX.hid.npz       (anchor_hidden [L, A, H] fp16)
     Nothing new is needed there except the lm_head row dump:
       python agent/t34_heads.py dump-lm-head \
         --model <c2kv ckpt or base> --tokenizer <tok> \
         --steps <capture_dir>/c2kv/p0.steps.jsonl \
         --out results/t34/lm_head_rows.npz

  1. [here] build the head input frame (hiddens + pool + entropy scalars) and
     the feature jsonl:
       python agent/t34_heads.py build-inputs --root . \
         --capture-dir results/t33/capture --arm c2kv \
         --out results/t34/head_inputs_c2kv.npz \
         --features-out results/t34/features_heads.jsonl

  2. [here] ALIEN, both label arms + controls:
       python agent/t34_heads.py alien --root . \
         --inputs results/t34/head_inputs_c2kv.npz \
         --lm-head-rows results/t34/lm_head_rows.npz \
         [--inputs-full results/t34/head_inputs_full.npz] \
         --features-out results/t34/features_alien.jsonl \
         --out results/t34/alien_table.json

  3. [here] MemGen sparsity head (lambda swept in inner folds, lambda=0 ablation):
       python agent/t34_heads.py memgen --root . \
         --inputs results/t34/head_inputs_c2kv.npz \
         --features-out results/t34/features_memgen.jsonl \
         --out results/t34/memgen_table.json

  4. [here] CORA splitter + CRC on any scalar produced above:
       python agent/t34_heads.py cora --root . \
         --scores results/t34/features_heads.jsonl \
         --score-column alien_u_arm_wrongany_oof \
         --primary-endpoint executed_harm_rate \
         [--sidecar results/t34/sidecar_c2kv.jsonl] \
         --out results/t34/cora_table.json

  5. [here] Self-REF / [RESET] label structures (no training, no GPU):
       python agent/t34_heads.py labels --root . \
         --alpha 0.15 --out results/t34/heads_label_structures.json

WIRING.  Nothing in this module touches the bench face; CORA's slot is HOOK 2
(``benchmarks/proxy.py`` ``RecoverState.check``), and a deployed ``tau_hat``
from :func:`crc_threshold` would be compared there against whatever scalar the
proxy has - but this unit calibrates offline on the frozen battery only, so no
bench-face function is defined here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from t33_labels import guard_columns  # noqa: E402

# ---------------------------------------------------------------------------
# deviations from the papers (the prereg copies this list verbatim)
# ---------------------------------------------------------------------------

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S3.3",
        "what": "The paper duplicates an existing |C|-way CLASSIFIER head. Our model is "
                "generative, so C = the session's tool-name pool and theta_init = the "
                "lm_head rows of each candidate tool name's FIRST token id.",
        "why": "Card 2505.15443, 'Missing / needs redesign' (a): this is the faithful "
               "analogue and the one the card tells us to build; the full-vocabulary "
               "alternative makes log C meaningless.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S3.3",
        "what": "theta is ONE matrix over the global union of candidate first-token ids, "
                "masked per row to that row's session pool, instead of a separate head "
                "per label set.",
        "why": "227 sessions / 900 rows cannot train 227 heads. Masking keeps the init "
               "identity U_ALIEN == U_Entropy exact on every row.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S3.3",
        "what": "U_Entropy (the L_reg anchor) is the normalised restricted entropy under "
                "the INIT head applied to the penultimate hidden state (a logit-lens "
                "read), not the model's own final-layer decode distribution.",
        "why": "Qwen3's lm_head sits after a final RMSNorm on the LAST layer, so lm_head "
               "rows applied to a penultimate hidden do not reproduce the decode "
               "distribution. The paper's init identity is preserved by construction; "
               "the model's actual decode entropy is reported separately as the "
               "bare-entropy baseline row.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S3.1",
        "what": "h(x) is the PENULTIMATE CAPTURED layer at the tool-name decode position "
                "(digest S4.10), not the model's penultimate layer at the last input "
                "token: the t33 capture stores only a subset of layers, so "
                "build_head_inputs's default slot is penultimate-of-the-capture. The "
                "captured layer ids travel with the frame (HeadInputs.layers) and every "
                "stored slot is a grid axis chosen in the inner folds.",
        "why": "The model is generative and the capture is partial; naming the slot "
               "'penultimate' without this clause would overstate the match to S3.1.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S4.1 technical details",
        "what": "alpha grid extended with 0 ({0, 0.01, 0.1, 1.0}); beta and lr grids kept "
                "as published; 20 epochs kept.",
        "why": "Digest S4.10: without alpha=0 the output regulariser pins the score to "
               "entropy and the trained arm cannot be distinguished from the baseline.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S4.1 technical details",
        "what": "Optimiser (Adam, beta=(0.9,0.999), eps=1e-8) and batch size (32, not "
                "swept) are OURS; the L2SP gradient is applied at every minibatch step "
                "rather than once per epoch.",
        "why": "S4.1 fixes only the lr grid and the 20 epochs; it names no optimiser and "
               "no batch size, so these had to be chosen and must be declared.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S4.1",
        "what": "Grid selected in INNER folds of a session-grouped nested CV; 2000 "
                "session-clustered bootstrap reps.",
        "why": "The paper does not state which split its grid is scored on and uses 20 "
               "unclustered resamples over i.i.d. sentences; our rows are clustered by "
               "session (digest S4.0).",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S3.4",
        "what": "Two label arms: e_i = 'compressed arm wrong' (wrong-any, 712/900, the "
                "paper's single-arm label) and e_i = C->W (93 within the 161-row "
                "subset). Both are SCORED on the same 161-row frame against C->W.",
        "why": "Digest S4.10 / card: the paper's label is single-arm; the gap between the "
               "two arms is itself the result.",
    },
    {
        "method": "ALIEN head",
        "paper": "2505.15443 S4.6 tab:ablation",
        "what": "The 'Rand CLS BCE' ablation is kept as a null control and, as published, "
                "is trained with BCE ONLY (alpha = beta = 0, lr swept over the published "
                "grid); 'Rand linear BCE' is not implemented.",
        "why": "One random-init control is enough to answer 'does the lm_head init "
               "matter'; a single-output linear probe is D-E1 territory and is another "
               "unit's row.",
    },
    {
        "method": "MemGen trigger",
        "paper": "2509.24704 S4.2",
        "what": "Supervised BCE on the C->W label replaces the RL objective "
                "max_phi E[R(tau) - lambda * sum max(0, d - pbar)].",
        "why": "The battery is single-step teacher-forced: there is no trajectory and no "
               "reward. Card: 'we do not have an RL-trainable reward at usable n'.",
    },
    {
        "method": "MemGen trigger",
        "paper": "2509.24704 S4.2 (eq. p_j = sigma(T_trigger(H_{t,<j})))",
        "what": "The trigger input is ONE hidden vector (the tool-name decode position of "
                "the chosen layer slot), not the whole prefix H_{t,<j} in "
                "R^{(j-1) x d_model}; the head is a plain logistic map on it.",
        "why": "The battery generates once and stores anchor hiddens only, so no "
               "per-step prefix exists offline; the portable part of the card is the "
               "sparsity penalty, not the sequence encoder.",
    },
    {
        "method": "MemGen trigger",
        "paper": "2509.24704 S4.2",
        "what": "pbar = mean predicted fire probability over the C->C rows of the TRAIN "
                "fold, recomputed each epoch and treated as a constant in the backward "
                "pass (stop-gradient). The penalty is evaluated on the predicted "
                "probability p and averaged (digest S4.10's mean(max(0, p - pbar))), not "
                "on the sampled action d~ and not summed as in the paper's "
                "lambda * sum_{i,j} max(0, d~_{i,j} - pbar).",
        "why": "Digest S4.10 maps 'high-reward trajectories' to the 68 C->C rows. The "
               "paper never differentiates through its batch reference rate either.",
    },
    {
        "method": "MemGen trigger",
        "paper": "2509.24704 S4.2",
        "what": "Decision is fire = p > tau (tau at a fixed fire rate chosen on the "
                "TRAIN fold), not d ~ Bernoulli(p). No LoRA on q_proj/v_proj, no "
                "delimiter pre-gate, a detached head instead of an adapter.",
        "why": "Determinism gate (SPEC S5.5.9); the battery generates once with "
               "max_new_tokens=128 and the proxy is non-streaming, so there is no "
               "mid-generation slot for a separator gate; an adapter would move the "
               "backbone and invalidate every frozen asset.",
    },
    {
        "method": "MemGen trigger",
        "paper": "2509.24704 S4.2",
        "what": "lambda is swept in inner folds with lambda=0 as a mandatory ablation.",
        "why": "The paper never reports or sweeps lambda; digest S4.10 makes lambda=0 "
               "compulsory because lambda and the threshold are two knobs on one "
               "trade-off.",
    },
    {
        "method": "CORA",
        "paper": "2604.09155 S3.1, S14.3",
        "what": "The Guardian R_psi is NOT trained. Only the block splitter (S12.5) and "
                "the CRC calibrator (S3.4) are ported; they are applied to scalars "
                "produced elsewhere.",
        "why": "Guardian training is LoRA on two 9B VLMs with 1588/502 samples and an "
               "extra 9B forward per step - a third checkpoint and a cost-line breach.",
    },
    {
        "method": "CORA",
        "paper": "2604.09155 S12.5",
        "what": "Second-level grouping uses a toolset key recomputed here with the same "
                "semantics as build_joint_split_manifest._toolset_key (sha1 of sorted "
                "{name, parameter (name,type) pairs, required} signatures), taken from "
                "the sidecar's per-qid `tools`, not from the parquet spans; task "
                "templates and init seeds do not exist in our data.",
        "why": "APPROXIMATION, declared: build_joint_split_manifest imports pyarrow at "
               "module scope and reads the raw parquet; the sidecar carries the same "
               "tool dicts the model was given. A unit test asserts the two keys agree.",
    },
    {
        "method": "CORA",
        "paper": "2604.09155 S5.4, S11.1",
        "what": "The three-column table reports BOTH CORA's autonomy coverage (1 - fire "
                "rate) and our contract's C->W recall, and BOTH the CRC-controlled "
                "executed-harm rate (residual C->W over ALL rows) and the digest's "
                "residual C->W among non-fired rows. The primary endpoint is a required "
                "argument with no default.",
        "why": "Digest S4.10 risk clause: which of the three columns is primary must be "
               "pre-registered, and CORA's own CRC point has the HIGHEST executed harm "
               "of every threshold it sweeps.",
    },
    {
        "method": "Self-REF",
        "paper": "2410.13284 Alg.1",
        "what": "No arm, no checkpoint, no training. Three-valued supervision "
                "{C->W, W->W, correct} with subsample_alpha applied to the W->W class "
                "instead of to the whole UN class.",
        "why": "Digest S4.10: their UN class maps to C->W + W->W = 712/900 of which only "
               "93 are repairable; the false-positive budget would burn on the 619 W->W "
               "rows. Any fine-tune produces a third checkpoint and voids every frozen "
               "asset.",
    },
    {
        "method": "[RESET]",
        "paper": "2409.14586 S3",
        "what": "No arm, no SFT, no DPO. Only the positive-prefix supervision position "
                "(first divergence index between the compressed and full arms' emitted "
                "ids) and the enumeration of the three negative-pair classes.",
        "why": "Digest S4.10: their positive prefix is found by rejection sampling with "
                "Llama Guard 2; our first-divergence index is exact. The never-tried "
                "third negative class is supplied by the 120 W->C rows.",
    },
]

# ---------------------------------------------------------------------------
# emitted feature columns and their pre-declared risk orientations
# ---------------------------------------------------------------------------

ORIENTATIONS: Dict[str, int] = {
    # free decode-time scalars at the tool-name token (compressed arm only)
    "entropy_name_token_vocab": +1,      # ALIEN's bare-entropy baseline
    "span_entropy_mean": +1,             # digest S4.10 step 1: <tool_call>-span mean entropy
    "span_entropy_max": +1,              # ... span max entropy
    "span_seq_nll": +1,                  # ... span sequence NLL
    "margin_name_top1_top2": -1,         # larger margin = more confident = less risky
    # trained heads (out-of-fold scores)
    "alien_u_arm_cw_oof": +1,
    "alien_u_arm_wrongany_oof": +1,
    "memgen_p_oof": +1,
    # nuisance / pre-registered controls (never headline signals)
    "ctrl_generated_tokens": +1,
    "ctrl_cap_hit": +1,
    "pool_size_c": +1,
}

CORA_ENDPOINTS = (
    "coverage_recall_cw",
    "coverage_autonomous",
    "residual_cw_among_nonfired",
    "executed_harm_rate",
    "fire_rate",
    "false_reset_rate",
)

# ALIEN's published grids (2505.15443 S4.1 "Technical details"), alpha extended with 0.
ALIEN_ALPHA_GRID = (0.0, 0.01, 0.1, 1.0)
ALIEN_BETA_GRID = (0.01, 0.1, 1.0)
ALIEN_LR_GRID = (4e-4, 1e-4, 1e-5)
ALIEN_EPOCHS = 20

MEMGEN_LAMBDA_GRID = (0.0, 0.1, 1.0, 10.0)
MEMGEN_LR_GRID = (1e-3, 1e-4)
MEMGEN_EPOCHS = 20

_EPS = 1e-6


# ===========================================================================
# 0.  capture / sidecar loaders  (schemas fixed by the task brief)
# ===========================================================================

def load_capture_steps(path: Path) -> Dict[str, Dict[str, Any]]:
    """``<capture_dir>/<arm>/p0.steps.jsonl`` -> qid -> row.

    Row schema (t33_capture.T33CaptureContext.finish_row): ``steps`` (per token
    ``token_id`` / ``chosen_logprob`` / ``entropy_full`` / ``top5``), ``spans``
    (``name_first`` ... ), ``anchors`` ([[label, pos], ...]), ``ic``
    (``candidate_token_ids``, ``n_candidates``, ``anchors``), ``meta``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in C.load_jsonl(str(path)):
        qid = row.get("qid")
        if qid:
            out[str(qid)] = row
    return out


def load_anchor_hiddens(arm_dir: Path, *, anchor: str = "name_first") -> Dict[str, Dict[str, Any]]:
    """Read every ``*.hid.npz`` shard in ``arm_dir``; return qid -> anchor record.

    Shard keys are ``"<qid>::<field>"`` (t33_capture.flush_hidden_store) with
    fields ``anchor_hidden`` [L, A, H] fp16, ``anchor_labels``,
    ``anchor_positions``, ``anchor_valid``, ``layers``.  Later shards win, which
    matches the t33 top-up convention (the top-up shards are the good ones).
    """
    arm_dir = Path(arm_dir)
    per_qid: Dict[str, Dict[str, Any]] = {}
    for shard in sorted(arm_dir.glob("*.hid.npz")):
        with np.load(shard, allow_pickle=True) as data:
            for key in data.files:
                if "::" not in key:
                    continue
                qid, field_name = key.rsplit("::", 1)
                per_qid.setdefault(qid, {})[field_name] = data[key]
    out: Dict[str, Dict[str, Any]] = {}
    for qid, entry in per_qid.items():
        hidden = entry.get("anchor_hidden")
        labels = entry.get("anchor_labels")
        if hidden is None or labels is None:
            continue
        labels = [str(x) for x in np.atleast_1d(labels).tolist()]
        if anchor not in labels:
            continue
        a = labels.index(anchor)
        valid = entry.get("anchor_valid")
        ok = True
        if valid is not None:
            vv = np.atleast_1d(valid).tolist()
            ok = bool(vv[a]) if a < len(vv) else False
        arr = np.asarray(hidden)          # [L, A, H]
        if arr.ndim != 3 or a >= arr.shape[1]:
            continue
        layers = entry.get("layers")
        out[qid] = {
            "hidden": arr[:, a, :].astype(np.float32),   # [L, H]
            "valid": ok,
            "layers": (np.atleast_1d(layers).astype(int).tolist() if layers is not None
                       else list(range(arr.shape[0]))),
        }
    return out


def load_lm_head_rows(path: Path) -> Dict[int, np.ndarray]:
    """``{ids: [G], rows: [G, H]}`` npz written by ``dump-lm-head`` on the NPU."""
    with np.load(Path(path)) as data:
        ids = np.asarray(data["ids"]).astype(int).tolist()
        rows = np.asarray(data["rows"]).astype(np.float32)
    return {int(i): rows[k] for k, i in enumerate(ids)}


def load_sidecar_tools(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Sidecar ``results/t34/sidecar_<arm>.jsonl`` -> qid -> ``tools`` list."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in C.load_jsonl(str(path)):
        qid = row.get("qid")
        if qid is not None:
            out[str(qid)] = list(row.get("tools") or [])
    return out


# ===========================================================================
# 1.  head input frame
# ===========================================================================

@dataclass
class HeadInputs:
    """Per-qid inputs for both heads.  Compressed-arm observables only."""

    qids: List[str]
    sessions: List[str]
    hidden: np.ndarray                 # [N, L_sel, H] float32
    layers: List[int]                  # captured layer ids for the L_sel axis
    cand_ids: List[List[int]]          # session tool-pool first-token ids
    entropy_vocab: np.ndarray          # [N] full-vocabulary entropy at the name token
    margin_top1_top2: np.ndarray       # [N]
    generated_tokens: np.ndarray       # [N]
    cap_hit: np.ndarray                # [N] bool
    valid: np.ndarray                  # [N] bool (anchor present, |C| >= 2)
    global_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.global_ids:
            seen: Set[int] = set()
            for ids in self.cand_ids:
                seen.update(int(i) for i in ids)
            self.global_ids = sorted(seen)

    @property
    def n(self) -> int:
        return len(self.qids)

    def pool_mask(self) -> np.ndarray:
        """[N, G] boolean membership of each row's tool pool in the global set."""
        pos = {tid: k for k, tid in enumerate(self.global_ids)}
        mask = np.zeros((self.n, len(self.global_ids)), dtype=bool)
        for i, ids in enumerate(self.cand_ids):
            for tid in ids:
                k = pos.get(int(tid))
                if k is not None:
                    mask[i, k] = True
        return mask

    def pool_size(self) -> np.ndarray:
        return np.array([len(set(int(t) for t in ids)) for ids in self.cand_ids], dtype=int)

    def layer_view(self, layer_slot: int) -> np.ndarray:
        return np.ascontiguousarray(self.hidden[:, layer_slot, :].astype(np.float64))

    def subset(self, idx: Sequence[int]) -> "HeadInputs":
        idx = list(idx)
        return HeadInputs(
            qids=[self.qids[i] for i in idx],
            sessions=[self.sessions[i] for i in idx],
            hidden=self.hidden[idx],
            layers=list(self.layers),
            cand_ids=[self.cand_ids[i] for i in idx],
            entropy_vocab=self.entropy_vocab[idx],
            margin_top1_top2=self.margin_top1_top2[idx],
            generated_tokens=self.generated_tokens[idx],
            cap_hit=self.cap_hit[idx],
            valid=self.valid[idx],
            global_ids=list(self.global_ids),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in self.cand_ids]) \
            if self.cand_ids else np.zeros(0, dtype=np.int64)
        offs = np.cumsum([0] + [len(ids) for ids in self.cand_ids]).astype(np.int64)
        np.savez_compressed(
            path,
            qids=np.array(self.qids),
            sessions=np.array(self.sessions),
            hidden=self.hidden.astype(np.float16),
            layers=np.asarray(self.layers, dtype=np.int64),
            cand_flat=flat,
            cand_offsets=offs,
            entropy_vocab=self.entropy_vocab,
            margin_top1_top2=self.margin_top1_top2,
            generated_tokens=self.generated_tokens,
            cap_hit=self.cap_hit,
            valid=self.valid,
            global_ids=np.asarray(self.global_ids, dtype=np.int64),
        )

    @staticmethod
    def load(path: Path) -> "HeadInputs":
        with np.load(Path(path), allow_pickle=True) as d:
            offs = np.asarray(d["cand_offsets"]).astype(int)
            flat = np.asarray(d["cand_flat"]).astype(int)
            cand = [flat[offs[i]:offs[i + 1]].tolist() for i in range(len(offs) - 1)]
            return HeadInputs(
                qids=[str(x) for x in d["qids"].tolist()],
                sessions=[str(x) for x in d["sessions"].tolist()],
                hidden=np.asarray(d["hidden"]).astype(np.float32),
                layers=np.asarray(d["layers"]).astype(int).tolist(),
                cand_ids=cand,
                entropy_vocab=np.asarray(d["entropy_vocab"], dtype=float),
                margin_top1_top2=np.asarray(d["margin_top1_top2"], dtype=float),
                generated_tokens=np.asarray(d["generated_tokens"], dtype=float),
                cap_hit=np.asarray(d["cap_hit"]).astype(bool),
                valid=np.asarray(d["valid"]).astype(bool),
                global_ids=np.asarray(d["global_ids"]).astype(int).tolist(),
            )


def build_head_inputs(
    qids: Sequence[str],
    sessions: Sequence[str],
    capture: Dict[str, Dict[str, Any]],
    hiddens: Dict[str, Dict[str, Any]],
    *,
    cap_tokens: int,
    layer_slots: Optional[Sequence[int]] = None,
) -> HeadInputs:
    """Assemble :class:`HeadInputs` for ``qids`` from a capture + hidden dump.

    ``layer_slots`` indexes the captured layer axis; the default is the
    PENULTIMATE captured layer (2505.15443 S3.1 takes h(x) from the penultimate
    layer).  Rows without a ``name_first`` anchor, without ``ic``, or with a
    tool pool of fewer than 2 first-token ids are kept with ``valid=False`` and
    NaN scalars - never silently dropped, never sentinel-filled.
    """
    n_layers = None
    for entry in hiddens.values():
        n_layers = int(entry["hidden"].shape[0])
        break
    if n_layers is None:
        n_layers = 1
    if layer_slots is None:
        layer_slots = [max(0, n_layers - 2)]
    layer_slots = [int(s) for s in layer_slots]

    dim = None
    for entry in hiddens.values():
        dim = int(entry["hidden"].shape[1])
        break
    dim = dim or 1

    n = len(qids)
    hid = np.zeros((n, len(layer_slots), dim), dtype=np.float32)
    ent = np.full(n, np.nan)
    marg = np.full(n, np.nan)
    gen = np.full(n, np.nan)
    cap = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    cand: List[List[int]] = []

    for i, qid in enumerate(qids):
        row = capture.get(qid) or {}
        ic = row.get("ic") or {}
        ids = [int(x) for x in (ic.get("candidate_token_ids") or [])]
        cand.append(sorted(set(ids)))
        steps = row.get("steps") or []
        spans = row.get("spans") or {}
        n_gen = spans.get("n_generated") or len(steps) or 0
        gen[i] = float(n_gen) if n_gen else np.nan
        stop = row.get("stop_reason")
        # the MemGen control "finish_reason == length": the capture records
        # stop_reason; the frozen battery rows carry finish_reason=None, so the
        # cap comparison is the fallback.
        cap[i] = bool(stop == "length") if stop else bool(n_gen >= cap_tokens)
        pos = spans.get("name_first")
        if pos is not None and 0 <= int(pos) < len(steps):
            st = steps[int(pos)]
            e = st.get("entropy_full")
            if e is not None:
                ent[i] = float(e)
            top5 = st.get("top5") or []
            if len(top5) >= 2:
                marg[i] = float(top5[0][0]) - float(top5[1][0])
        h = hiddens.get(qid)
        if h is not None and h.get("valid") and len(cand[i]) >= 2:
            arr = np.asarray(h["hidden"])
            if arr.shape[0] > max(layer_slots):
                hid[i] = arr[layer_slots, :]
                valid[i] = True
    return HeadInputs(
        qids=list(qids), sessions=list(sessions), hidden=hid, layers=list(layer_slots),
        cand_ids=cand, entropy_vocab=ent, margin_top1_top2=marg,
        generated_tokens=gen, cap_hit=cap, valid=valid,
    )


def decode_span_scalars(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Digest S4.10 step 1 - the FIVE free decode-time scalars of a capture row.

    From the t33 capture record (``steps``: per-token ``entropy_full`` /
    ``chosen_logprob`` / ``top5``; ``spans``: token-index span map):
      * ``entropy_name_token_vocab``  full-vocabulary entropy at the tool-name first token
      * ``margin_name_top1_top2``     top1 - top2 logprob margin at that token
      * ``span_entropy_mean``         mean entropy over the <tool_call> payload span
      * ``span_entropy_max``          max entropy over that span
      * ``span_seq_nll``              sequence NLL (sum of -chosen_logprob) over that span
    Every value is None when its span is absent (no sentinel, no fallback to
    the whole output - the t33 audit's flare_min_p fallback bug).
    """
    steps = row.get("steps") or []
    spans = row.get("spans") or {}
    out: Dict[str, Optional[float]] = {
        "entropy_name_token_vocab": None, "margin_name_top1_top2": None,
        "span_entropy_mean": None, "span_entropy_max": None, "span_seq_nll": None,
    }
    nf = spans.get("name_first")
    if nf is not None and 0 <= int(nf) < len(steps):
        st = steps[int(nf)]
        if st.get("entropy_full") is not None:
            out["entropy_name_token_vocab"] = float(st["entropy_full"])
        top5 = st.get("top5") or []
        if len(top5) >= 2:
            out["margin_name_top1_top2"] = float(top5[0][0]) - float(top5[1][0])
    pf, pl = spans.get("payload_first"), spans.get("payload_last")
    if pf is not None and pl is not None and 0 <= int(pf) <= int(pl) < len(steps):
        seg = steps[int(pf):int(pl) + 1]
        ents = [float(s["entropy_full"]) for s in seg if s.get("entropy_full") is not None]
        lps = [float(s["chosen_logprob"]) for s in seg if s.get("chosen_logprob") is not None]
        if ents:
            out["span_entropy_mean"] = float(np.mean(ents))
            out["span_entropy_max"] = float(np.max(ents))
        if lps and len(lps) == len(seg):
            out["span_seq_nll"] = float(-np.sum(lps))
    return out


# ===========================================================================
# 2.  ALIEN head (2505.15443 S3.3-S3.4) - explicit forward/backward, numpy
# ===========================================================================

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """softmax over the True entries of ``mask`` (row-wise); 0 elsewhere."""
    any_row = mask.any(axis=1, keepdims=True)
    z = np.where(mask, logits, -np.inf)
    zmax = np.where(any_row, np.max(np.where(mask, z, -np.inf), axis=1, keepdims=True), 0.0)
    z = np.where(mask, z - zmax, -np.inf)
    e = np.where(mask, np.exp(z), 0.0)
    s = e.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return e / s


def normalised_entropy(p: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``U = -(1/log C) sum_c p log p`` (2505.15443 S3.3) and the raw entropy H.

    ``C`` is the row's own pool size.  Rows with ``C < 2`` get NaN (log C = 0 is
    undefined) - they are excluded from every fit and counted in the report.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    H = -plogp.sum(axis=1)
    c = mask.sum(axis=1).astype(float)
    logc = np.where(c >= 2, np.log(np.maximum(c, 2.0)), np.nan)
    return H / logc, H


@dataclass
class AlienHead:
    """``p(c|x) = softmax(W h + b)_c`` over the session's tool pool.

    Implements 2505.15443 S3.3 (head + normalised-entropy score) and S3.4
    (``L = L_BCE + alpha*L_reg + beta*L2SP``) with explicit gradients; the
    finite-difference check lives in ``test_t34_heads.py``.
    """

    theta: np.ndarray      # [G, d]
    bias: np.ndarray       # [G]
    theta_init: np.ndarray
    bias_init: np.ndarray

    @staticmethod
    def from_lm_head(global_ids: Sequence[int], rows: Dict[int, np.ndarray],
                     dim: int) -> "AlienHead":
        """theta_init = the lm_head row of each candidate tool name's FIRST token."""
        th = np.zeros((len(global_ids), dim), dtype=np.float64)
        for k, tid in enumerate(global_ids):
            vec = rows.get(int(tid))
            if vec is None:
                raise KeyError(f"lm_head row missing for candidate token id {tid}")
            if len(vec) != dim:
                raise ValueError(f"lm_head row dim {len(vec)} != hidden dim {dim}")
            th[k] = np.asarray(vec, dtype=np.float64)
        b = np.zeros(len(global_ids), dtype=np.float64)   # Qwen3 lm_head has no bias
        return AlienHead(theta=th.copy(), bias=b.copy(), theta_init=th.copy(), bias_init=b.copy())

    @staticmethod
    def random_init(global_ids: Sequence[int], dim: int, seed: int = 0) -> "AlienHead":
        """'Rand CLS BCE' ablation (2505.15443 Table tab:ablation)."""
        rng = np.random.default_rng(seed)
        th = rng.normal(0.0, 0.02, size=(len(global_ids), dim))
        b = np.zeros(len(global_ids))
        return AlienHead(theta=th.copy(), bias=b.copy(), theta_init=th.copy(), bias_init=b.copy())

    # -- forward ----------------------------------------------------------
    def logits(self, Hx: np.ndarray) -> np.ndarray:
        return Hx @ self.theta.T + self.bias[None, :]

    def score(self, Hx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """U_ALIEN(x) in [0, 1]."""
        p = masked_softmax(self.logits(Hx), mask)
        u, _ = normalised_entropy(p, mask)
        return u

    # -- loss + gradients -------------------------------------------------
    def loss_and_grads(
        self, Hx: np.ndarray, mask: np.ndarray, e: np.ndarray, u_ent: np.ndarray,
        *, alpha: float, beta: float,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
        n = len(e)
        z = self.logits(Hx)
        p = masked_softmax(z, mask)
        u, Hraw = normalised_entropy(p, mask)
        c = mask.sum(axis=1).astype(float)
        logc = np.log(np.maximum(c, 2.0))

        uc = np.clip(u, _EPS, 1.0 - _EPS)
        interior = (u > _EPS) & (u < 1.0 - _EPS)
        bce = -(e * np.log(uc) + (1.0 - e) * np.log(1.0 - uc))
        l_bce = float(bce.mean())
        reg = (u - u_ent) ** 2
        # L_reg = (1/N) sum (U_ALIEN - U_Entropy)^2 (2505.15443 S3.4).  A row with a
        # non-finite anchor contributes 0 to BOTH this sum and the gradient below, so
        # the loss and its gradient can never end up on different denominators
        # (nanmean would divide by the finite count while the gradient divides by n).
        l_reg = float(np.where(np.isfinite(reg), reg, 0.0).sum() / n) if n else 0.0
        dth = self.theta - self.theta_init
        db = self.bias - self.bias_init
        l2sp = float((dth * dth).sum() + (db * db).sum())
        total = l_bce + alpha * l_reg + beta * l2sp

        # dL/dU
        dU = (uc - e) / (uc * (1.0 - uc)) / n
        dU = np.where(interior, dU, 0.0)
        if alpha:
            reg_term = 2.0 * (u - u_ent) / n
            dU = dU + alpha * np.where(np.isfinite(reg_term), reg_term, 0.0)
        # dU/dz_j = -(1/logC) p_j (log p_j + H)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
        dUdz = -(p * (logp + Hraw[:, None])) / logc[:, None]
        dUdz = np.where(mask, dUdz, 0.0)
        dz = dU[:, None] * dUdz                                   # [N, G]
        g_theta = dz.T @ Hx + 2.0 * beta * dth
        g_bias = dz.sum(axis=0) + 2.0 * beta * db
        parts = {"bce": l_bce, "reg": l_reg, "l2sp": l2sp, "total": total}
        return total, g_theta, g_bias, parts

    # -- fit --------------------------------------------------------------
    def fit(
        self, Hx: np.ndarray, mask: np.ndarray, e: np.ndarray, u_ent: np.ndarray,
        *, alpha: float, beta: float, lr: float, epochs: int = ALIEN_EPOCHS,
        batch_size: int = 32, seed: int = 20260905,
    ) -> "AlienHead":
        """Adam, ``epochs`` passes.  2505.15443 S4.1 fixes lr/epochs but not the
        optimiser; Adam is our choice (declared in DEVIATIONS via the grid entry)."""
        rng = np.random.default_rng(seed)
        n = len(e)
        m_t = np.zeros_like(self.theta)
        v_t = np.zeros_like(self.theta)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)
        step = 0
        bs = max(1, min(int(batch_size), n))
        for _ in range(int(epochs)):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                _, gt, gb, _ = self.loss_and_grads(
                    Hx[idx], mask[idx], e[idx], u_ent[idx], alpha=alpha, beta=beta)
                step += 1
                for (par, grad, m, v) in ((self.theta, gt, m_t, v_t), (self.bias, gb, m_b, v_b)):
                    m *= 0.9
                    m += 0.1 * grad
                    v *= 0.999
                    v += 0.001 * grad * grad
                    mhat = m / (1 - 0.9 ** step)
                    vhat = v / (1 - 0.999 ** step)
                    par -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        return self


# ===========================================================================
# 3.  MemGen sparsity-penalty head (2509.24704 S4.2)
# ===========================================================================

@dataclass
class SparsityHead:
    """``p = sigmoid(w.h + b)``; ``L = BCE(y,p) + lambda * mean(max(0, p - pbar))``.

    ``pbar`` is the mean predicted fire probability on the C->C rows of the
    training fold (2509.24704's "high-reward trajectory" activation rate),
    recomputed each epoch and detached in the backward pass.
    """

    w: np.ndarray
    b: float

    @staticmethod
    def zeros(dim: int) -> "SparsityHead":
        return SparsityHead(w=np.zeros(dim, dtype=np.float64), b=0.0)

    def proba(self, Hx: np.ndarray) -> np.ndarray:
        z = Hx @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def loss_and_grads(
        self, Hx: np.ndarray, y: np.ndarray, *, lam: float, pbar: float,
    ) -> Tuple[float, np.ndarray, float, Dict[str, float]]:
        n = len(y)
        p = self.proba(Hx)
        pc = np.clip(p, _EPS, 1.0 - _EPS)
        l_bce = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
        excess = np.maximum(0.0, p - pbar)
        l_pen = float(excess.mean())
        total = l_bce + lam * l_pen
        dz = (p - y) / n
        if lam:
            dz = dz + lam * ((p > pbar).astype(float) * p * (1.0 - p)) / n
        g_w = Hx.T @ dz
        g_b = float(dz.sum())
        return total, g_w, g_b, {"bce": l_bce, "penalty": l_pen, "pbar": float(pbar),
                                 "total": total}

    def fit(
        self, Hx: np.ndarray, y: np.ndarray, cc_mask: np.ndarray, *,
        lam: float, lr: float, epochs: int = MEMGEN_EPOCHS, batch_size: int = 32,
        seed: int = 20260905,
    ) -> "SparsityHead":
        rng = np.random.default_rng(seed)
        n = len(y)
        m_w = np.zeros_like(self.w)
        v_w = np.zeros_like(self.w)
        m_b = v_b = 0.0
        step = 0
        bs = max(1, min(int(batch_size), n))
        for _ in range(int(epochs)):
            # pbar recomputed each epoch from the CURRENT head on the C->C rows
            pbar = float(self.proba(Hx[cc_mask]).mean()) if cc_mask.any() else 0.0
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                _, gw, gb, _ = self.loss_and_grads(Hx[idx], y[idx], lam=lam, pbar=pbar)
                step += 1
                m_w = 0.9 * m_w + 0.1 * gw
                v_w = 0.999 * v_w + 0.001 * gw * gw
                m_b = 0.9 * m_b + 0.1 * gb
                v_b = 0.999 * v_b + 0.001 * gb * gb
                self.w -= lr * (m_w / (1 - 0.9 ** step)) / (np.sqrt(v_w / (1 - 0.999 ** step)) + 1e-8)
                self.b -= lr * (m_b / (1 - 0.9 ** step)) / (math.sqrt(v_b / (1 - 0.999 ** step)) + 1e-8)
        return self


# ===========================================================================
# 4.  nested, session-grouped CV for an arbitrary head
# ===========================================================================

def nested_cv_head(
    fit_predict: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Dict[str, np.ndarray]],
    y_train: np.ndarray,
    groups: np.ndarray,
    grid: Sequence[Dict[str, Any]],
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 20260905,
    select_by: str = "auroc",
    fire_rates: Sequence[float] = (),
) -> Dict[str, Any]:
    """Same protocol as :func:`t34_common.nested_cv_logistic`, for any head.

    ``fit_predict(train_idx, test_idx, params) -> {"test": s_te, "train": s_tr}``.
    Hyper-parameters are chosen on INNER folds of the outer-train fold by
    ``select_by`` scored against ``y_train``; nothing is chosen on outer test.
    ``fire_rates`` thresholds are set with
    :func:`t34_common.fixed_rate_threshold` on the outer-TRAIN scores and
    applied to the outer-test scores.
    """
    y_train = np.asarray(y_train, dtype=float)
    groups = np.asarray(groups)
    n = len(y_train)
    oof = np.full(n, np.nan)
    fires = {q: np.zeros(n, dtype=bool) for q in fire_rates}
    chosen: List[Dict[str, Any]] = []
    metric = C.auroc if select_by == "auroc" else C.average_precision

    for outer_mask in C.grouped_folds(groups, outer_folds, seed):
        te = np.where(outer_mask)[0]
        tr = np.where(~outer_mask)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        ytr_bin = (y_train[tr] > 0.5).astype(int)
        if ytr_bin.sum() in (0, len(tr)):
            continue
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        gtr = groups[tr]
        for params in grid:
            inner = np.full(len(tr), np.nan)
            for inner_mask in C.grouped_folds(gtr, inner_folds, seed + 1):
                ite = np.where(inner_mask)[0]
                itr = np.where(~inner_mask)[0]
                if len(ite) == 0 or len(itr) == 0:
                    continue
                yb = (y_train[tr][itr] > 0.5).astype(int)
                if yb.sum() in (0, len(itr)):
                    continue
                out = fit_predict(tr[itr], tr[ite], params)
                inner[ite] = out["test"]
            ok = np.isfinite(inner)
            yb = ytr_bin[ok]
            if not ok.any() or yb.sum() in (0, int(ok.sum())):
                continue
            m = metric(inner[ok], yb)
            if m is not None and (best is None or m > best[0]):
                best = (float(m), dict(params))
        if best is None:
            continue
        out = fit_predict(tr, te, best[1])
        oof[te] = out["test"]
        chosen.append({"params": best[1], "inner_metric": best[0], "n_test": int(len(te))})
        s_tr = out.get("train")
        if s_tr is not None:
            for q in fire_rates:
                thr = C.fixed_rate_threshold(np.asarray(s_tr, dtype=float), q)
                if np.isfinite(thr):
                    fires[q][te] = np.asarray(out["test"], dtype=float) > thr
    ok = np.isfinite(oof)
    return {
        "oof_scores": oof,
        "scored_mask": ok,
        "n_scored": int(ok.sum()),
        "chosen": chosen,
        "fires": {q: fires[q] for q in fire_rates},
    }


# ===========================================================================
# 5.  metrics the papers add on top of the default three
# ===========================================================================

def risk_coverage(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: reject the highest-risk rows first (2505.15443 S4.1)."""
    s = np.asarray(scores, dtype=float)
    e = np.asarray(y, dtype=float)
    order = np.argsort(s, kind="mergesort")          # most confident first
    err = e[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Area under the risk-coverage curve (lower is better); None if empty."""
    if len(y) == 0:
        return None
    _, risk = risk_coverage(scores, y)
    return float(risk.mean())


def oracle_aurc(y: np.ndarray) -> Optional[float]:
    """The Oracle column of 2505.15443 Table tab:final_results_all_rejection_curve."""
    e = np.asarray(y, dtype=float)
    if len(e) == 0:
        return None
    return aurc(e, e)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Optional[float]:
    """Expected calibration error, equal-width bins (2505.15443 S4.1)."""
    p = np.asarray(probs, dtype=float)
    e = np.asarray(y, dtype=float)
    if len(p) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - e[m].mean())
    return float(total)


def score_row(
    label: str, scores: np.ndarray, y: np.ndarray, sessions: Sequence[str],
    *, probs: Optional[np.ndarray] = None, reps: int = 2000,
    n_fires: Sequence[int] = (),
) -> Dict[str, Any]:
    """One row of the output table on ONE row subset (n and n_pos reported)."""
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    ok = np.isfinite(s)
    s, yy = s[ok], yy[ok]
    sess = [sessions[i] for i in range(len(ok)) if ok[i]]
    cl = C.session_clusters(sess) if sess else np.zeros(0, dtype=int)
    ap = C.average_precision(s, yy)
    lo, hi, ncl = C.clustered_bootstrap(C.average_precision, s, yy, cl, reps=reps) \
        if len(s) else (None, None, 0)
    row: Dict[str, Any] = {
        "row": label,
        "n": int(len(s)),
        "n_pos": int(yy.sum()),
        "n_scored_of": int(len(ok)),
        "prevalence": C.prevalence(yy) if len(yy) else None,
        "auprc": ap,
        "auprc_ci95": [lo, hi],
        "n_clusters": int(ncl),
        "auroc": C.auroc(s, yy) if len(s) else None,
        "aurc": aurc(s, yy) if len(s) else None,
        "aurc_oracle": oracle_aurc(yy) if len(yy) else None,
        "ece": (ece(np.asarray(probs, dtype=float)[ok], yy) if probs is not None else None),
        "operating_points": [C.operating_point(s, yy, int(k)) for k in n_fires],
    }
    return row


# ===========================================================================
# 6.  CORA: block splitter + CRC calibrator (2604.09155 S12.5, S3.4)
# ===========================================================================

def _hash_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _schema_obj(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = (function.get("parameters") or tool.get("parameters")
              or tool.get("input_schema") or tool.get("schema") or {})
    return schema if isinstance(schema, dict) else {}


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(function.get("name") or tool.get("name") or "")


def _parameter_signature(schema: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out = []
    for name, value in properties.items():
        if isinstance(value, dict):
            typ = value.get("type") or value.get("anyOf") or value.get("oneOf") \
                or value.get("items") or "unknown"
        else:
            typ = "unknown"
        out.append((str(name), json.dumps(typ, ensure_ascii=False, sort_keys=True)))
    return tuple(sorted(out))


def toolset_key(tools: Sequence[Dict[str, Any]]) -> str:
    """sha1 of the sorted tool signatures.

    Same semantics as ``build_joint_split_manifest._toolset_key`` (re-implemented
    so this module does not pull in that file's module-scope ``pyarrow`` import);
    ``test_t34_heads.py`` asserts the two agree when pyarrow is available.
    """
    sigs = []
    for tool in tools:
        schema = _schema_obj(tool)
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        sigs.append({
            "name": _tool_name(tool),
            "parameters": _parameter_signature(schema),
            "required": tuple(sorted(str(item) for item in required)),
        })
    sigs = sorted(sigs, key=lambda item: item["name"])
    return _hash_json(sigs)


def cora_block_split(
    sessions: Sequence[str],
    *,
    toolset_keys: Optional[Dict[str, str]] = None,
    fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    seed: int = 20260905,
) -> Dict[str, Any]:
    """Blockwise split (2604.09155 S12.5): whole ``session_id`` blocks, never rows.

    Second level: sessions sharing a toolset key are union-found into one
    super-block, so a toolset assigned to ``cal`` contributes no row to
    ``train``/``test`` (their template+seed grouping; ours is by toolset - see
    DEVIATIONS).  Sessions without a toolset key never merge.
    """
    sessions = [str(s) for s in sessions]
    uniq = sorted(set(sessions))
    parent = {s: s for s in uniq}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if toolset_keys:
        by_key: Dict[str, List[str]] = {}
        for s in uniq:
            k = toolset_keys.get(s)
            if k:
                by_key.setdefault(k, []).append(s)
        for members in by_key.values():
            members = sorted(members)
            for other in members[1:]:
                union(members[0], other)

    blocks: Dict[str, List[str]] = {}
    for s in uniq:
        blocks.setdefault(find(s), []).append(s)
    counts = {s: 0 for s in uniq}
    for s in sessions:
        counts[s] += 1
    block_ids = sorted(blocks)
    rng = np.random.default_rng(seed)
    order = [block_ids[i] for i in rng.permutation(len(block_ids))]
    total = float(len(sessions))
    targets = [fractions[0] * total, (fractions[0] + fractions[1]) * total]
    assign: Dict[str, str] = {}
    running = 0.0
    for blk in order:
        members = blocks[blk]
        size = sum(counts[s] for s in members)
        split = "train" if running < targets[0] else ("cal" if running < targets[1] else "test")
        for s in members:
            assign[s] = split
        running += size
    out: Dict[str, Any] = {
        "unit": "session_id (72 sessions carry the 93 C->W rows; 100 carry the "
                "161-row trigger subset) - never qid",
        "assignment": assign,
        "blocks": {b: sorted(v) for b, v in blocks.items()},
        "n_sessions": len(uniq),
        "n_blocks": len(block_ids),
        "n_rows_by_split": {k: sum(counts[s] for s in uniq if assign[s] == k)
                            for k in ("train", "cal", "test")},
        # a toolset shared by every session merges the whole frame into ONE block:
        # blockwise splitting is then infeasible, not "0 % calibration rows".
        "max_block_row_share": (max(sum(counts[s] for s in v) for v in blocks.values())
                                / float(len(sessions))) if sessions else None,
    }
    out["degenerate"] = any(out["n_rows_by_split"][k] == 0 for k in ("train", "cal", "test"))
    return out


def crc_threshold(
    scores: np.ndarray,
    harm: np.ndarray,
    alpha: float,
    *,
    grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """``tau_hat = sup{tau : (1/(n+1))(sum_i L(Z_i;tau) + 1) <= alpha}``.

    2604.09155 S3.4 with ``L(Z;tau) = harm(Z) * 1{s <= tau}``: an action is
    executed (here: NOT repaired) when its risk score is at or below tau, so the
    controlled quantity is the executed-harm rate = residual C->W over ALL rows.
    ``L`` is monotone non-decreasing in tau, so the sup over the candidate grid
    is the largest feasible grid point.  Returns ``tau_hat=None`` when the budget
    is infeasible (``1/(n+1) > alpha``, i.e. ``n < 1/alpha - 1``).
    """
    s = np.asarray(scores, dtype=float)
    h = np.asarray(harm, dtype=float)
    n = len(s)
    if grid is None:
        # a FINITE sentinel strictly below every score means "execute nothing /
        # fire on everything"; -inf would serialise as invalid JSON.
        vals = sorted(set(s.tolist()))
        grid = ([float(vals[0]) - 1.0] + vals) if vals else [0.0]
    feasible: List[float] = []
    for tau in grid:
        loss = float((h * (s <= tau)).sum())
        if (loss + 1.0) / (n + 1.0) <= alpha:
            feasible.append(float(tau))
    tau_hat = max(feasible) if feasible else None
    return {
        "tau_hat": tau_hat,
        "alpha": float(alpha),
        "n_cal": int(n),
        "feasible": bool(feasible),
        "min_n_for_alpha": int(math.ceil(1.0 / alpha - 1.0)) if alpha > 0 else None,
        "empirical_loss_at_tau": (float((h * (s <= tau_hat)).sum() / n)
                                  if tau_hat is not None and n else None),
    }


def cora_columns(scores: np.ndarray, y: np.ndarray, tau: Optional[float]) -> Dict[str, Any]:
    """The ablation-table columns at one threshold; fire iff ``s > tau``."""
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    n = len(s)
    fire = (s > tau) if tau is not None else np.zeros(n, dtype=bool)
    n_fire = int(fire.sum())
    n_pos = int(yy.sum())
    n_neg = int((yy == 0).sum())
    residual = int(((~fire) & (yy == 1)).sum())
    n_nonfire = n - n_fire
    return {
        "tau": (None if tau is None else float(tau)),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fires": n_fire,
        "fire_rate": (n_fire / n) if n else None,
        "coverage_recall_cw": (int((fire & (yy == 1)).sum()) / n_pos) if n_pos else None,
        "coverage_autonomous": (n_nonfire / n) if n else None,
        "residual_cw": residual,
        "residual_cw_among_nonfired": (residual / n_nonfire) if n_nonfire else None,
        "executed_harm_rate": (residual / n) if n else None,
        "false_resets": int((fire & (yy == 0)).sum()),
        "false_reset_rate": (int((fire & (yy == 0)).sum()) / n_neg) if n_neg else None,
        "precision": (int((fire & (yy == 1)).sum()) / n_fire) if n_fire else None,
    }


def cora_ablation_table(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    primary_endpoint: str,
    alpha: float,
    cal_idx: Sequence[int],
    test_idx: Sequence[int],
    static_taus: Sequence[float] = (),
    parse_fail_fire: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """static-tau sweep vs the CRC ``tau_hat``, plus the parse-failure-only row.

    ``primary_endpoint`` is REQUIRED and must name one of :data:`CORA_ENDPOINTS`
    (2604.09155's own CRC point has the highest executed harm of every threshold
    it sweeps, so the endpoint has to be pre-registered by the caller).
    """
    if primary_endpoint not in CORA_ENDPOINTS:
        raise ValueError(f"primary_endpoint must be one of {CORA_ENDPOINTS}, got "
                         f"{primary_endpoint!r} - it has no default by design")
    if len(list(cal_idx)) == 0 or len(list(test_idx)) == 0:
        # a CRC certificate on n=0 calibration rows, or a table scored on 0 test rows,
        # would come back as a full column of nulls and read like "no signal".
        raise ValueError(
            "empty calibration or test fold (n_cal="
            f"{len(list(cal_idx))}, n_test={len(list(test_idx))}): blockwise splitting "
            "collapsed - usually every session shares one toolset key, so the "
            "second-level grouping merged the whole frame into a single block. "
            "Check cora_block_split(...)['max_block_row_share'] before reading a table.")
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    cal_idx = np.asarray(list(cal_idx), dtype=int)
    test_idx = np.asarray(list(test_idx), dtype=int)
    crc = crc_threshold(s[cal_idx], yy[cal_idx].astype(float), alpha)
    rows = [dict(cora_columns(s[test_idx], yy[test_idx], t), setting=f"static tau={t:g}")
            for t in static_taus]
    rows.append(dict(cora_columns(s[test_idx], yy[test_idx], crc["tau_hat"]),
                     setting=f"CRC tau_hat (alpha={alpha:g})"))
    if parse_fail_fire is not None:
        pf = np.asarray(parse_fail_fire, dtype=float)
        crc_pf = crc_threshold(pf[cal_idx], yy[cal_idx].astype(float), alpha)
        rows.append(dict(cora_columns(pf[test_idx], yy[test_idx], crc_pf["tau_hat"]),
                         setting=f"parse-failure only, CRC (alpha={alpha:g})"))
        rows.append(dict(cora_columns(pf[test_idx], yy[test_idx], 0.5),
                         setting="parse-failure only, fire=parse_fail"))
    return {
        "primary_endpoint": primary_endpoint,
        "alpha": float(alpha),
        "crc": crc,
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "rows": rows,
        "note": ("Quote all three columns together: in 2604.09155 S11.1 the CRC point "
                 "(89.95 / 2.42 / 10.05) has the HIGHEST executed harm of every static "
                 "threshold swept (0.56 at tau=0.5, 1.86 at tau=0.9)."),
    }


# ===========================================================================
# 7.  Self-REF / [RESET] label-structure utilities (LABEL side, no arm)
# ===========================================================================

def label_wrong_any(frame: "C.FrozenFrame") -> Dict[str, int]:
    """2505.15443's single-arm label ported: 1 = the compressed arm is wrong.

    LABEL side (reads ``tool_name_match``).  C->W u W->W = 712/900.
    """
    out: Dict[str, int] = {}
    for f, c in frame.pairs:
        out[f["qid"]] = 0 if c.get("tool_name_match") else 1
    return out


def selfref_label_three_valued(frame: "C.FrozenFrame") -> Dict[str, Any]:
    """Three-valued supervision {C->W, W->W, correct} (2410.13284 Alg.1, migrated).

    The paper's binary {UN, CN} maps to UN = C->W u W->W (712) which spends the
    whole false-positive budget on the 619 unrepairable W->W rows; the digest's
    correct shape splits UN.  LABEL side.
    """
    classes: Dict[str, str] = {}
    for f, c in frame.pairs:
        fm = bool(f.get("tool_name_match"))
        cm = bool(c.get("tool_name_match"))
        if cm:
            classes[f["qid"]] = "correct"          # C->C (68) u W->C (120)
        elif fm:
            classes[f["qid"]] = "cw"               # C->W (93)
        else:
            classes[f["qid"]] = "ww"               # W->W (619)
    counts = {k: sum(1 for v in classes.values() if v == k) for k in ("cw", "ww", "correct")}
    return {"classes": classes, "counts": counts, "n": len(classes)}


def selfref_label_alpha_downsample(
    classes: Dict[str, str], alpha: float, *, seed: int = 20260905,
) -> Dict[str, Any]:
    """``subsample_alpha`` (2410.13284 Alg.1) applied to the W->W class.

    The paper subsamples the whole UN set; we keep every C->W row (the only
    repairable positives) and subsample W->W only.  ``n_kept = floor(alpha * n_ww)``
    with a deterministic RNG; alpha=1 keeps all, alpha=0 keeps none.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1] (2410.13284 Alg.1)")
    ww = sorted(q for q, v in classes.items() if v == "ww")
    keep_n = int(math.floor(alpha * len(ww)))
    rng = np.random.default_rng(seed)
    pick = set(np.asarray(ww, dtype=object)[rng.permutation(len(ww))[:keep_n]].tolist()) \
        if ww and keep_n else set()
    kept = sorted([q for q, v in classes.items() if v in ("cw", "correct")] + sorted(pick))
    n_cw = sum(1 for v in classes.values() if v == "cw")
    n_ok = sum(1 for v in classes.values() if v == "correct")
    return {
        "alpha": float(alpha),
        "n_ww_total": len(ww),
        "n_ww_kept": keep_n,
        "n_cw_kept": n_cw,
        "n_correct_kept": n_ok,
        "n_kept": len(kept),
        "kept_qids": kept,
        "positive_fraction_after": ((n_cw + keep_n) / len(kept)) if kept else None,
    }


def reset_label_positive_prefixes(
    ids_c2kv: Dict[str, Sequence[int]],
    ids_full: Dict[str, Sequence[int]],
    qids: Sequence[str],
    *,
    cap_tokens: int = 128,
) -> Dict[str, Any]:
    """First divergence index for the ``[RESET]`` positive prefix (2409.14586 S3).

    The paper finds "when the generation went bad" by rejection-sampling random
    prefixes of ``y^-`` through Llama Guard 2; on paired arms the exact position
    is ``min{j : a_j != b_j}`` between the compressed and full arms' emitted ids.
    LABEL side (reads the full arm).  ``first_div`` is None when one id sequence
    is a prefix of the other; ``cap_hit`` is reported alongside and never merged
    into it (t34_common.first_divergence_index's contract).
    """
    out: Dict[str, Any] = {}
    for qid in qids:
        a = list(ids_c2kv.get(qid) or [])
        b = list(ids_full.get(qid) or [])
        j = C.first_divergence_index(a, b)
        out[qid] = {
            "first_div": j,
            "n_c2kv": len(a),
            "n_full": len(b),
            "cap_hit_c2kv": bool(len(a) >= cap_tokens),
            "cap_hit_full": bool(len(b) >= cap_tokens),
            "prefix_only": bool(j is None and len(a) != len(b)),
        }
    n_div = sum(1 for v in out.values() if v["first_div"] is not None)
    return {
        "entries": out,
        "n": len(out),
        "n_with_divergence": n_div,
        "n_no_divergence": len(out) - n_div,
    }


def reset_label_negative_pairs(frame: "C.FrozenFrame") -> Dict[str, Any]:
    """The three ``[RESET]`` preference classes mapped onto our census (2409.14586 S3).

    * positive pair ``prefix(y^-) + [RESET] + y^+ > y^-`` : the 93 C->W rows;
    * negative pair ``y^+ > prefix(y^+) + [RESET] + y^-`` : the 68 C->C rows;
    * the third class the paper says it NEVER tried,
      ``y^+ > prefix(y^+) + [RESET] + y'^+`` : the 120 W->C rows, where the
      compressed arm is right and the full arm is wrong, so firing is pure harm.
    """
    cw, cc, wc, ww = [], [], [], []
    for f, c in frame.pairs:
        fm, cm = bool(f.get("tool_name_match")), bool(c.get("tool_name_match"))
        (cw if (fm and not cm) else cc if (fm and cm) else wc if cm else ww).append(f["qid"])
    return {
        "positive_pairs_cw": sorted(cw),
        "negative_pairs_cc": sorted(cc),
        "untried_third_class_wc": sorted(wc),
        "excluded_ww": len(ww),
        "counts": {"cw": len(cw), "cc": len(cc), "wc": len(wc), "ww": len(ww)},
        "note": "No checkpoint is trained: 2409.14586 needs SFT+DPO and 2410.13284 needs "
                "LoRA on the backbone; either produces a third checkpoint and voids the "
                "frozen 900-row paired set.",
    }


# ===========================================================================
# 8.  drivers
# ===========================================================================

def _eval_frame(frame: "C.FrozenFrame") -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    sessions = [r["session_id"] for r in sub]
    y = np.array([int(r["label_cw"]) for r in sub], dtype=int)
    pf = np.array([1.0 if r["parse_fail_fire"] else 0.0 for r in sub], dtype=float)
    return qids, sessions, y, pf


def run_alien(
    inputs: HeadInputs,
    frame: "C.FrozenFrame",
    lm_rows: Dict[int, np.ndarray],
    *,
    inputs_full: Optional[HeadInputs] = None,
    alpha_grid: Sequence[float] = ALIEN_ALPHA_GRID,
    beta_grid: Sequence[float] = ALIEN_BETA_GRID,
    lr_grid: Sequence[float] = ALIEN_LR_GRID,
    epochs: int = ALIEN_EPOCHS,
    layer_slots: Optional[Sequence[int]] = None,
    reps: int = 2000,
    seed: int = 20260905,
) -> Dict[str, Any]:
    """The ALIEN table: bare entropy first, then the two label arms + controls."""
    eval_qids, eval_sessions, y_eval, pf = _eval_frame(frame)
    pos_by_qid = {q: i for i, q in enumerate(inputs.qids)}
    wrong_any = label_wrong_any(frame)

    mask_all = inputs.pool_mask()
    dim = inputs.hidden.shape[2]
    # every stored layer slot is a knob -> it goes through the INNER folds (rule 6);
    # the default store holds only the penultimate layer (2505.15443 S3.1).
    slots = [int(s) for s in (layer_slots if layer_slots else range(inputs.hidden.shape[1]))]
    grid = [{"alpha": a, "beta": b, "lr": lr, "layer": s}
            for a in alpha_grid for b in beta_grid for lr in lr_grid for s in slots]

    init_head = AlienHead.from_lm_head(inputs.global_ids, lm_rows, dim)
    views = {s: inputs.layer_view(s) for s in slots}
    u_ent_by_slot = {s: init_head.score(views[s], mask_all) for s in slots}

    def make_fit_predict(e_train: np.ndarray, rand_init: bool, rows: np.ndarray):
        """``rows`` maps nested_cv_head's SUBSET-local indices back to the
        global row index of every full-length array below.  Without this map a
        strict ``usable`` subset silently trains on the wrong rows and scatters
        each OOF score onto a different qid (regression test:
        ``test_run_alien_scores_the_right_rows_when_some_are_invalid``)."""
        rows = np.asarray(rows, dtype=int)

        def fit_predict(tr_idx, te_idx, params):
            s = int(params["layer"])
            Hs = views[s]
            g_tr = rows[np.asarray(tr_idx, dtype=int)]
            g_te = rows[np.asarray(te_idx, dtype=int)]
            head = (AlienHead.random_init(inputs.global_ids, dim, seed=seed)
                    if rand_init else AlienHead.from_lm_head(inputs.global_ids, lm_rows, dim))
            head.fit(Hs[g_tr], mask_all[g_tr], e_train[g_tr], u_ent_by_slot[s][g_tr],
                     alpha=params["alpha"], beta=params["beta"], lr=params["lr"],
                     epochs=epochs, seed=seed)
            return {"test": head.score(Hs[g_te], mask_all[g_te]),
                    "train": head.score(Hs[g_tr], mask_all[g_tr])}
        return fit_predict

    finite_u = np.all(np.stack([np.isfinite(u_ent_by_slot[s]) for s in slots]), axis=0)
    usable = inputs.valid & finite_u
    groups_all = np.asarray(inputs.sessions)

    arms: Dict[str, Dict[str, Any]] = {}
    for arm_name, label_map, row_filter in (
        ("arm_wrongany", wrong_any, usable),
        ("arm_cw", frame.label_by_qid, usable & np.array(
            [frame.label_by_qid.get(q) in (0, 1) for q in inputs.qids])),
    ):
        idx = np.where(row_filter)[0]
        if len(idx) == 0:
            arms[arm_name] = {"error": "no usable rows"}
            continue
        e_train = np.array([float(label_map.get(q) or 0) for q in inputs.qids], dtype=float)
        sub = idx
        res = nested_cv_head(
            make_fit_predict(e_train, False, sub),
            y_train=e_train[sub], groups=groups_all[sub],
            grid=grid, seed=seed,
        )
        oof = np.full(len(inputs.qids), np.nan)
        oof[sub] = res["oof_scores"]
        arms[arm_name] = {
            "chosen": res["chosen"],
            "n_train_rows": int(len(sub)),
            "n_train_pos": int((e_train[sub] > 0.5).sum()),
            "oof": oof,
        }

    # random-init control on the wrong-any arm (2505.15443 tab:ablation "Rand CLS BCE")
    e_rand = np.array([float(wrong_any.get(q) or 0) for q in inputs.qids], dtype=float)
    sub = np.where(usable)[0]
    # 2505.15443 tab:ablation "Rand CLS. BCE" is trained with BCE ONLY, so the
    # control grid pins alpha = beta = 0 and sweeps only the published lr grid.
    rand_grid = [{"alpha": 0.0, "beta": 0.0, "lr": lr, "layer": s}
                 for lr in lr_grid for s in slots]
    rand_res = nested_cv_head(
        make_fit_predict(e_rand, True, sub),
        y_train=e_rand[sub], groups=groups_all[sub],
        grid=rand_grid, seed=seed,
    )
    rand_oof = np.full(len(inputs.qids), np.nan)
    rand_oof[sub] = rand_res["oof_scores"]

    def on_eval(vec: np.ndarray) -> np.ndarray:
        return np.array([vec[pos_by_qid[q]] if q in pos_by_qid else np.nan for q in eval_qids])

    n_fires = [len(eval_qids) // 10, len(eval_qids) // 5]
    table: List[Dict[str, Any]] = []
    # ROW 1 (mandatory first row): raw full-vocabulary entropy at the tool-name token
    table.append(score_row("bare entropy @ tool-name token (2505.15443 baseline 'Entropy')",
                           on_eval(inputs.entropy_vocab), y_eval, eval_sessions,
                           reps=reps, n_fires=n_fires))
    table.append(score_row("U_Entropy: normalised restricted entropy under theta_init (lens)",
                           on_eval(u_ent_by_slot[slots[0]]), y_eval, eval_sessions,
                           probs=on_eval(u_ent_by_slot[slots[0]]), reps=reps, n_fires=n_fires))
    table.append(score_row("top1-top2 margin @ tool-name token (oriented -1)",
                           -on_eval(inputs.margin_top1_top2), y_eval, eval_sessions,
                           reps=reps, n_fires=n_fires))
    table.append(score_row("baseline: parse failure only", pf, y_eval, eval_sessions,
                           reps=reps, n_fires=n_fires))
    for arm_name in ("arm_wrongany", "arm_cw"):
        arm = arms.get(arm_name) or {}
        if "oof" not in arm:
            continue
        sc = on_eval(arm["oof"])
        table.append(score_row(f"ALIEN {arm_name}", sc, y_eval, eval_sessions,
                               probs=sc, reps=reps, n_fires=n_fires))
    table.append(score_row("control: Rand CLS BCE (2505.15443 tab:ablation)",
                           on_eval(rand_oof), y_eval, eval_sessions, reps=reps, n_fires=n_fires))
    table.append(score_row("control: generated_tokens (length)",
                           on_eval(inputs.generated_tokens), y_eval, eval_sessions,
                           reps=reps, n_fires=n_fires))

    def _paired(a: np.ndarray, b: np.ndarray, label: str) -> Optional[Dict[str, Any]]:
        """Paired session-clustered Delta-AUPRC on the rows where BOTH are finite."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            return None
        cl = C.session_clusters([eval_sessions[i] for i in np.where(ok)[0]])
        d, lo, hi = C.paired_delta_bootstrap(C.average_precision, a[ok], b[ok],
                                             y_eval[ok], cl, reps=reps)
        return {"contrast": label, "delta_auprc": d, "ci95": [lo, hi],
                "n": int(ok.sum()), "n_pos": int(y_eval[ok].sum()),
                "n_clusters": int(len(set(cl.tolist())))}

    # arm gap on the SAME rows
    gap = None
    if "oof" in (arms.get("arm_wrongany") or {}) and "oof" in (arms.get("arm_cw") or {}):
        g = _paired(on_eval(arms["arm_wrongany"]["oof"]), on_eval(arms["arm_cw"]["oof"]),
                    "ALIEN arm_wrongany - arm_cw")
        if g is not None:
            gap = {"delta_auprc_wrongany_minus_cw": g["delta_auprc"], "ci95": g["ci95"],
                   "n": g["n"], "n_pos": g["n_pos"]}

    # pre-registered length control (digest S4.0: the winner rule wants an increment
    # over the length control, not just a side-by-side row).
    length_increment = None
    if "oof" in (arms.get("arm_wrongany") or {}):
        length_increment = _paired(
            on_eval(arms["arm_wrongany"]["oof"]), on_eval(inputs.generated_tokens),
            "ALIEN arm_wrongany - generated_tokens (length control)")

    # S0 twin: the identical pipeline on the FULL arm's hiddens
    s0 = None
    if inputs_full is not None:
        s0_res = run_alien(inputs_full, frame, lm_rows, inputs_full=None,
                           alpha_grid=alpha_grid, beta_grid=beta_grid, lr_grid=lr_grid,
                           epochs=epochs, layer_slots=layer_slots, reps=reps, seed=seed)
        s0 = {"note": "S0 twin: same feature computed on the FULL arm (digest S4.0 winner rule)",
              "table": s0_res["table"]}
        s0_oof = (s0_res.get("oof_scores") or {}).get("arm_wrongany")
        if s0_oof is not None and "oof" in (arms.get("arm_wrongany") or {}):
            pos_full = {q: i for i, q in enumerate(inputs_full.qids)}
            s0_on_eval = np.array([
                float(np.asarray(s0_oof, dtype=float)[pos_full[q]]) if q in pos_full
                else np.nan for q in eval_qids])
            s0["delta_auprc_vs_s0"] = _paired(
                on_eval(arms["arm_wrongany"]["oof"]), s0_on_eval,
                "ALIEN arm_wrongany: compressed arm - S0 full arm (digest S4.0 winner "
                "rule: the clustered CI lower bound must exceed 0)")

    pool = inputs.pool_size()
    strata = {}
    for name, m in (("pool_C_eq_2", pool == 2), ("pool_C_gt_2", pool > 2)):
        sel = np.array([m[pos_by_qid[q]] if q in pos_by_qid else False for q in eval_qids])
        if sel.sum() >= 5 and 0 < y_eval[sel].sum() < sel.sum():
            strata[name] = score_row(f"ALIEN arm_wrongany | {name}",
                                     on_eval((arms.get("arm_wrongany") or {}).get(
                                         "oof", np.full(len(inputs.qids), np.nan)))[sel],
                                     y_eval[sel], [eval_sessions[i] for i in np.where(sel)[0]],
                                     reps=reps)
        else:
            strata[name] = {"row": name, "n": int(sel.sum()), "note": "too few rows to score"}
    cap = np.array([inputs.cap_hit[pos_by_qid[q]] if q in pos_by_qid else False
                    for q in eval_qids])
    cap_strata = {}
    for name, sel in (("censored_at_cap", cap), ("uncensored", ~cap)):
        if sel.sum() >= 5 and 0 < y_eval[sel].sum() < sel.sum():
            cap_strata[name] = score_row(
                f"ALIEN arm_wrongany | {name}",
                on_eval((arms.get("arm_wrongany") or {}).get(
                    "oof", np.full(len(inputs.qids), np.nan)))[sel],
                y_eval[sel], [eval_sessions[i] for i in np.where(sel)[0]], reps=reps)
        else:
            cap_strata[name] = {"row": name, "n": int(sel.sum()), "note": "too few rows to score"}

    return {
        "method": "ALIEN (2505.15443)",
        "frame": {"n": len(eval_qids), "n_pos": int(y_eval.sum()),
                  "prevalence_is_chance_ap": C.prevalence(y_eval),
                  "n_sessions": len(set(eval_sessions)),
                  # digest S4.10 asks for an Oracle column next to AURC; ours is the
                  # frozen repair ceiling raw_erratum_tail = 75/93 (never recomputed).
                  "oracle_repair_ceiling": C.LOCATE_BEST_ARM_HITS[0] / C.LOCATE_BEST_ARM_HITS[1],
                  "oracle_repair_ceiling_hits": list(C.LOCATE_BEST_ARM_HITS)},
        "table": table,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in arms.items()},
        "arm_gap": gap,
        "increment_over_length_control": length_increment,
        "s0_full_arm": s0,
        "pool_strata": strata,
        "cap_strata": cap_strata,
        "caveats": [
            "|C| varies by session, so the 1/log C normalisation makes U_ALIEN "
            "non-comparable ACROSS sessions (2505.15443 S3.3; card pitfall).",
            f"|C| == 2 rows: {int((pool == 2).sum())} of {len(pool)} - the entropy of a "
            "two-way softmax is symmetric in the two classes and cannot say WHICH tool "
            "was predicted (degenerate).",
            f"n rows without a usable name-token anchor or with |C| < 2: "
            f"{int((~usable).sum())} of {len(usable)}.",
            "Their smallest fine-tuning set is 3,394 rows; ours is 900 with 93 C->W. "
            "Expect the trained head to lose to raw entropy (2505.15443 S4.3).",
            "The ECE cell of the arm_wrongany row is NOT a calibration statement: that "
            "head is fitted to P(compressed arm wrong) and the ECE is taken against the "
            "C->W label, a different event. Only the arm_cw row's ECE is on-estimand; "
            "read the arm_wrongany ECE as a distribution-shape diagnostic only.",
        ],
        "oof_scores": {"arm_wrongany": (arms.get("arm_wrongany") or {}).get("oof"),
                       "arm_cw": (arms.get("arm_cw") or {}).get("oof"),
                       "rand_init": rand_oof},
    }


def run_memgen(
    inputs: HeadInputs,
    frame: "C.FrozenFrame",
    *,
    lam_grid: Sequence[float] = MEMGEN_LAMBDA_GRID,
    lr_grid: Sequence[float] = MEMGEN_LR_GRID,
    epochs: int = MEMGEN_EPOCHS,
    fire_rates: Sequence[float] = (0.1, 0.2),
    layer_slot: int = 0,
    reps: int = 2000,
    seed: int = 20260905,
) -> Dict[str, Any]:
    """The MemGen sparsity-penalty head, scored on the 161-row trigger subset."""
    if 0.0 not in tuple(lam_grid):
        raise ValueError("lambda=0 is a mandatory ablation (digest S4.10)")
    eval_qids, eval_sessions, y_eval, pf = _eval_frame(frame)
    pos = {q: i for i, q in enumerate(inputs.qids)}
    keep = [k for k, q in enumerate(eval_qids) if q in pos and inputs.valid[pos[q]]]
    idx = np.array([pos[eval_qids[k]] for k in keep], dtype=int)
    y = y_eval[keep]
    sessions = [eval_sessions[k] for k in keep]
    groups = np.asarray(sessions)
    Hs = inputs.layer_view(layer_slot)[idx]
    # standardise on the fit fold only -> do it inside fit_predict
    cc_mask_all = (y == 0)

    grid = [{"lam": l, "lr": lr} for l in lam_grid for lr in lr_grid]

    def fit_predict(tr, te, params):
        mu = Hs[tr].mean(axis=0)
        sd = Hs[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr = (Hs[tr] - mu) / sd
        Xte = (Hs[te] - mu) / sd
        head = SparsityHead.zeros(Hs.shape[1])
        head.fit(Xtr, y[tr].astype(float), cc_mask_all[tr], lam=params["lam"],
                 lr=params["lr"], epochs=epochs, seed=seed)
        return {"test": head.proba(Xte), "train": head.proba(Xtr)}

    res = nested_cv_head(fit_predict, y_train=y.astype(float), groups=groups,
                         grid=grid, seed=seed, fire_rates=fire_rates)
    oof = res["oof_scores"]

    # pre-registered controls: length / cap only, and the increment over them
    ctrl = np.column_stack([inputs.generated_tokens[idx], inputs.cap_hit[idx].astype(float)])
    ctrl_oof = np.full(len(y), np.nan)
    fin = np.isfinite(ctrl).all(axis=1)
    if fin.any():
        # nested_cv_logistic drops non-finite rows internally, so pass only the finite
        # ones and scatter the OOF scores back onto the full row index (never impute).
        ctrl_res = C.nested_cv_logistic(ctrl[fin], y[fin], groups[fin], seed=seed)
        pos_fin = np.where(fin)[0]
        ctrl_oof[pos_fin[ctrl_res["scored_mask"]]] = \
            ctrl_res["oof_scores"][ctrl_res["scored_mask"]]

    n_fires = [max(1, len(y) // 10), max(1, len(y) // 5)]
    table = [
        score_row("bare entropy @ tool-name token", inputs.entropy_vocab[idx], y, sessions,
                  reps=reps, n_fires=n_fires),
        score_row("baseline: parse failure only", pf[keep], y, sessions, reps=reps,
                  n_fires=n_fires),
        score_row("control: generated_tokens + cap_hit (logistic)", ctrl_oof, y, sessions,
                  reps=reps, n_fires=n_fires),
        score_row("MemGen sparsity head", oof, y, sessions, probs=oof, reps=reps,
                  n_fires=n_fires),
    ]
    ok = np.isfinite(oof) & np.isfinite(ctrl_oof)
    inc = None
    if ok.sum():
        cl = C.session_clusters([sessions[i] for i in np.where(ok)[0]])
        d, lo, hi = C.paired_delta_bootstrap(C.average_precision, oof[ok], ctrl_oof[ok],
                                             y[ok], cl, reps=reps)
        inc = {"delta_auprc_head_minus_length_control": d, "ci95": [lo, hi],
               "n": int(ok.sum()), "n_pos": int(y[ok].sum())}

    false_reset = {}
    for q, fires in res["fires"].items():
        n_neg = int((y == 0).sum())
        false_reset[str(q)] = {
            "fire_rate_target": q,
            "fires": int(fires.sum()),
            "coverage": int((fires & (y == 1)).sum()),
            "n_pos": int(y.sum()),
            "false_resets": int((fires & (y == 0)).sum()),
            "n_neg": n_neg,
            "false_reset_rate": (int((fires & (y == 0)).sum()) / n_neg) if n_neg else None,
            "per_step_false_fire_rate": C.per_step_false_fire_rate(fires, y),
        }
    return {
        "method": "MemGen memory trigger, sparsity penalty only (2509.24704 S4.2)",
        "frame": {"n": int(len(y)), "n_pos": int(y.sum()),
                  "prevalence_is_chance_ap": C.prevalence(y),
                  "n_sessions": len(set(sessions)),
                  "n_eval_rows_dropped_invalid": int(len(eval_qids) - len(keep))},
        "table": table,
        "chosen": res["chosen"],
        "false_reset_column": false_reset,
        "increment_over_length_control": inc,
        "oof_scores": oof,
        "caveats": [
            "2509.24704's learned trigger beats always-on by +0.87 to +1.96 pp with no "
            "error bars (tab:ablation-trigger): do NOT enter expecting a large effect.",
            "A hidden-state head will happily learn 'this generation is about to hit the "
            "cap'; the generated_tokens / cap_hit control row and the increment over it "
            "are the pre-registered guard.",
            "cap_hit is the capture's stop_reason == 'length' where recorded and "
            "generated_tokens >= max_new_tokens otherwise: the frozen battery rows carry "
            "finish_reason = None.",
        ],
    }


# ===========================================================================
# 9.  CLI
# ===========================================================================

def _f(vec: Optional[np.ndarray], i: int) -> Optional[float]:
    """One feature cell: None (json null) when undefined - never a sentinel."""
    if vec is None:
        return None
    v = float(np.asarray(vec, dtype=float)[i])
    return v if np.isfinite(v) else None


def _cmd_dump_lm_head(args: argparse.Namespace) -> int:
    """[NPU] dump the lm_head rows of every candidate tool-name first token."""
    import torch  # noqa: F401  (lazy: torch is absent on the analysis box)
    from transformers import AutoModelForCausalLM

    capture = load_capture_steps(Path(args.steps))
    ids: Set[int] = set()
    for row in capture.values():
        for tid in ((row.get("ic") or {}).get("candidate_token_ids") or []):
            ids.add(int(tid))
    ordered = sorted(ids)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    weight = model.get_output_embeddings().weight.detach().to("cpu").float().numpy()
    rows = np.stack([weight[i] for i in ordered]) if ordered else np.zeros((0, weight.shape[1]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.asarray(ordered, dtype=np.int64), rows=rows.astype(np.float32))
    print(f"wrote {out} ids={len(ordered)} dim={rows.shape[1] if rows.size else 0}")
    return 0


def _cmd_build_inputs(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    capture = load_capture_steps(Path(args.capture_dir) / args.arm / args.steps_name)
    hiddens = load_anchor_hiddens(Path(args.capture_dir) / args.arm, anchor=args.anchor)
    qids = [r["qid"] for r in frame.labels]
    sessions = [r["session_id"] for r in frame.labels]
    inputs = build_head_inputs(qids, sessions, capture, hiddens,
                               cap_tokens=frame.cap_tokens(),
                               layer_slots=(args.layers or None))
    inputs.save(Path(args.out))
    print(json.dumps({"out": str(args.out), "n": inputs.n,
                      "n_valid": int(inputs.valid.sum()),
                      "n_global_candidates": len(inputs.global_ids),
                      "layers": inputs.layers}, indent=1))
    if args.features_out:
        pool = inputs.pool_size()
        rows = []
        for i, qid in enumerate(inputs.qids):
            rows.append({
                "qid": qid,
                "session_id": inputs.sessions[i],
                "arm": args.arm,
                "entropy_name_token_vocab": (float(inputs.entropy_vocab[i])
                                             if np.isfinite(inputs.entropy_vocab[i]) else None),
                "margin_name_top1_top2": (float(inputs.margin_top1_top2[i])
                                          if np.isfinite(inputs.margin_top1_top2[i]) else None),
                "ctrl_generated_tokens": (float(inputs.generated_tokens[i])
                                          if np.isfinite(inputs.generated_tokens[i]) else None),
                "ctrl_cap_hit": float(inputs.cap_hit[i]),
                "pool_size_c": int(pool[i]) if pool[i] else None,
            })
            # digest S4.10 step 1: the span-level scalars (name-token ones above)
            sc = decode_span_scalars(capture.get(qid) or {})
            rows[-1]["span_entropy_mean"] = sc["span_entropy_mean"]
            rows[-1]["span_entropy_max"] = sc["span_entropy_max"]
            rows[-1]["span_seq_nll"] = sc["span_seq_nll"]
        n = C.write_features_jsonl(Path(args.features_out), rows, context="t34 heads features")
        print(f"features rows={n} -> {args.features_out}")
    return 0


def _cmd_alien(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    inputs = HeadInputs.load(Path(args.inputs))
    lm_rows = load_lm_head_rows(Path(args.lm_head_rows))
    full = HeadInputs.load(Path(args.inputs_full)) if args.inputs_full else None
    out = run_alien(inputs, frame, lm_rows, inputs_full=full, reps=args.reps, epochs=args.epochs)
    oof = out.pop("oof_scores", None) or {}
    if args.features_out:
        rows = []
        for i, qid in enumerate(inputs.qids):
            rows.append({
                "qid": qid, "session_id": inputs.sessions[i], "arm": "c2kv",
                "alien_u_arm_wrongany_oof": _f(oof.get("arm_wrongany"), i),
                "alien_u_arm_cw_oof": _f(oof.get("arm_cw"), i),
                "entropy_name_token_vocab": _f(inputs.entropy_vocab, i),
            })
        n = C.write_features_jsonl(Path(args.features_out), rows, context="t34 ALIEN features")
        print(f"features rows={n} -> {args.features_out}")
    sha = C.freeze_json(Path(args.out), out)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "rows": [r.get("row") for r in out["table"]]}, indent=1))
    return 0


def _cmd_memgen(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    inputs = HeadInputs.load(Path(args.inputs))
    out = run_memgen(inputs, frame, reps=args.reps, epochs=args.epochs)
    oof = np.asarray(out["oof_scores"], dtype=float)
    out["oof_scores"] = [None if not np.isfinite(v) else float(v) for v in oof]
    if args.features_out:
        eval_qids, _, _, _ = _eval_frame(frame)
        pos = {q: i for i, q in enumerate(inputs.qids)}
        keep = [q for q in eval_qids if q in pos and inputs.valid[pos[q]]]
        rows = [{"qid": q, "session_id": C.session_of(q), "arm": "c2kv",
                 "memgen_p_oof": _f(oof, i)} for i, q in enumerate(keep)]
        n = C.write_features_jsonl(Path(args.features_out), rows, context="t34 MemGen features")
        print(f"features rows={n} -> {args.features_out}")
    sha = C.freeze_json(Path(args.out), out)
    print(json.dumps({"out": str(args.out), "sha256": sha}, indent=1))
    return 0


def _cmd_cora(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    eval_qids, eval_sessions, y, pf = _eval_frame(frame)
    feats = {r["qid"]: r for r in C.load_jsonl(str(args.scores))}
    def _val(q: str) -> float:
        v = (feats.get(q) or {}).get(args.score_column)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan

    s = np.array([_val(q) for q in eval_qids], dtype=float)
    ok = np.isfinite(s)
    tools = load_sidecar_tools(Path(args.sidecar)) if args.sidecar else {}
    ts_keys: Dict[str, str] = {}
    ts_conflicts: List[str] = []
    for qid in sorted(tools):                     # sorted -> file order cannot change the split
        tl = tools[qid]
        if not tl:
            continue
        sess = C.session_of(qid)
        key = toolset_key(tl)
        if sess not in ts_keys:
            ts_keys[sess] = key
        elif ts_keys[sess] != key and sess not in ts_conflicts:
            # a session whose steps were given different tool sets: the first (lowest
            # qid) key wins, and the session is named so nobody reads the second-level
            # grouping as exact.
            ts_conflicts.append(sess)
    split = cora_block_split([eval_sessions[i] for i in np.where(ok)[0]],
                             toolset_keys=ts_keys or None, seed=args.seed)
    assign = split["assignment"]
    idx_ok = np.where(ok)[0]
    cal_idx = [i for i in idx_ok if assign.get(eval_sessions[i]) == "cal"]
    test_idx = [i for i in idx_ok if assign.get(eval_sessions[i]) == "test"]
    table = cora_ablation_table(
        s, y, primary_endpoint=args.primary_endpoint, alpha=args.alpha,
        cal_idx=cal_idx, test_idx=test_idx,
        static_taus=[float(t) for t in (args.static_taus or [])],
        parse_fail_fire=pf)
    table["split"] = {k: v for k, v in split.items() if k != "blocks"}
    table["score_column"] = args.score_column
    table["toolset_grouping"] = {
        "n_sessions_with_key": len(ts_keys),
        "sessions_with_conflicting_step_toolsets": sorted(ts_conflicts),
        "note": ("Second-level grouping is by toolset key (DEVIATIONS: CORA groups by task "
                 "template + init seed, which our data does not carry). Without --sidecar "
                 "every session is its own block and the split is the leaky session-random "
                 "one the digest warns about."),
    }
    table["n_scored"] = int(ok.sum())
    table["n_unscored"] = int((~ok).sum())
    sha = C.freeze_json(Path(args.out), table)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "tau_hat": table["crc"]["tau_hat"]}, indent=1))
    return 0


def _cmd_labels(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    three = selfref_label_three_valued(frame)
    down = selfref_label_alpha_downsample(three["classes"], args.alpha)
    down.pop("kept_qids", None)
    pairs = reset_label_negative_pairs(frame)
    out = {
        "selfref_three_valued": {"counts": three["counts"], "n": three["n"]},
        "selfref_alpha_downsample": down,
        "reset_negative_pairs": {"counts": pairs["counts"], "note": pairs["note"]},
        "wrong_any_denominator": {
            "n_wrong_any": sum(label_wrong_any(frame).values()),
            "n_total": len(frame.pairs),
        },
        "reset_positive_prefixes": (
            "run with --gen-ids-c2kv/--gen-ids-full (capture steps.jsonl generated_ids) "
            "to fill the first-divergence table"),
        "deviations": DEVIATIONS,
    }
    if args.gen_ids_c2kv and args.gen_ids_full:
        a = {q: r.get("generated_ids") or [] for q, r in
             load_capture_steps(Path(args.gen_ids_c2kv)).items()}
        b = {q: r.get("generated_ids") or [] for q, r in
             load_capture_steps(Path(args.gen_ids_full)).items()}
        pref = reset_label_positive_prefixes(a, b, frame.cw_qids(),
                                             cap_tokens=frame.cap_tokens())
        pref.pop("entries", None)
        out["reset_positive_prefixes"] = pref
    sha = C.freeze_json(Path(args.out), out)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "counts": three["counts"]}, indent=1))
    return 0


def _cmd_orientations(args: argparse.Namespace) -> int:
    guard_columns(sorted(ORIENTATIONS), context="t34 heads orientations")
    sha = C.freeze_json(Path(args.out), ORIENTATIONS)
    print(json.dumps({"out": str(args.out), "sha256": sha, "n": len(ORIENTATIONS)}, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t34_heads",
        description="U9 / digest S4.10: ALIEN head, MemGen sparsity penalty, CORA "
                    "block splitter + CRC, Self-REF/[RESET] label structures.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-lm-head", help="[NPU] dump lm_head rows for tool-name first tokens")
    d.add_argument("--model", required=True)
    d.add_argument("--steps", required=True, help="capture steps.jsonl (for candidate ids)")
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_dump_lm_head)

    b = sub.add_parser("build-inputs", help="[here] assemble the head input frame")
    b.add_argument("--root", default=".")
    b.add_argument("--capture-dir", required=True)
    b.add_argument("--arm", default="c2kv", choices=["c2kv", "full"])
    b.add_argument("--steps-name", default="p0.steps.jsonl")
    b.add_argument("--anchor", default="name_first")
    b.add_argument("--layers", type=int, nargs="*", default=None,
                   help="captured-layer slots (default: penultimate)")
    b.add_argument("--out", required=True)
    b.add_argument("--features-out", default=None)
    b.set_defaults(func=_cmd_build_inputs)

    a = sub.add_parser("alien", help="[here] ALIEN head, both label arms + controls")
    a.add_argument("--root", default=".")
    a.add_argument("--inputs", required=True)
    a.add_argument("--inputs-full", default=None, help="S0 twin on the full arm")
    a.add_argument("--lm-head-rows", required=True)
    a.add_argument("--epochs", type=int, default=ALIEN_EPOCHS)
    a.add_argument("--reps", type=int, default=2000)
    a.add_argument("--out", required=True)
    a.add_argument("--features-out", default=None, help="write the OOF head scores as features")
    a.set_defaults(func=_cmd_alien)

    m = sub.add_parser("memgen", help="[here] MemGen sparsity-penalty head")
    m.add_argument("--root", default=".")
    m.add_argument("--inputs", required=True)
    m.add_argument("--epochs", type=int, default=MEMGEN_EPOCHS)
    m.add_argument("--reps", type=int, default=2000)
    m.add_argument("--out", required=True)
    m.add_argument("--features-out", default=None, help="write the OOF head scores as features")
    m.set_defaults(func=_cmd_memgen)

    c = sub.add_parser("cora", help="[here] CORA block split + CRC calibration")
    c.add_argument("--root", default=".")
    c.add_argument("--scores", required=True, help="features jsonl holding the scalar")
    c.add_argument("--score-column", required=True)
    c.add_argument("--primary-endpoint", required=True, choices=list(CORA_ENDPOINTS),
                   help="pre-registered primary column; there is no default by design")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--static-taus", type=float, nargs="*", default=[])
    c.add_argument("--sidecar", default=None, help="sidecar_<arm>.jsonl for toolset grouping")
    c.add_argument("--seed", type=int, default=20260905)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_cora)

    l = sub.add_parser("labels", help="[here] Self-REF / [RESET] label structures (no arm)")
    l.add_argument("--root", default=".")
    l.add_argument("--alpha", type=float, default=0.15, help="W->W subsample proportion")
    l.add_argument("--gen-ids-c2kv", default=None)
    l.add_argument("--gen-ids-full", default=None)
    l.add_argument("--out", required=True)
    l.set_defaults(func=_cmd_labels)

    o = sub.add_parser("orientations", help="[here] write configs/t34/orientations_heads.json")
    o.add_argument("--out", default=str(_HERE.parent / "configs/t34/orientations_heads.json"))
    o.set_defaults(func=_cmd_orientations)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
