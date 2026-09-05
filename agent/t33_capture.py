# -*- coding: utf-8 -*-
"""t33 capture instrumentation (survey item 4.0-2): the ONE rerun pass.

Both battery arms rerun with ``--capture_out``; this module collects, per row:

  * per generated token: chosen/eos logprob, FULL-VOCAB entropy, top-k logprobs
    (4.2 FLARE / FC-UQ / KnowNo / Leyline, 4.3 all, ALIEN entropy features);
  * token-span map of the emitted ``<tool_call>`` (name/args token ranges);
  * per-layer hidden states at the last prompt token and at anchor positions
    of the continuation (4.4 probes; anchors = name span, args endpoints,
    first/last/penultimate);
  * restricted-unembed Internal-Consistency scalars over the session tool pool
    (4.4 IC; computed in-process where lm_head is available);
  * context side (best-effort, flag-guarded): last-history-token hidden per
    layer, per-chunk-boundary o_proj inputs (4.4 KWTS / joint probe context
    side), per-doc gist saturation stats (4.1);
  * per-row doc sidecar: doc texts' lexical features (gzip ratio, IDF
    surprise) computed in-process, doc token lengths, kept/dropped counts.

Failures in the optional context/gist parts are counted and never kill the
row: the decode-side capture is the load-bearing part.

Storage per part-shard:
  <capture_out>/<arm>/<part>.steps.jsonl   steps + spans + IC + meta
  <capture_out>/<arm>/<part>.hid.npz       fp16 hidden matrices, keyed qid
  <capture_out>/<arm>/<part>.docs.jsonl    doc sidecar (incl. gist stats)
"""

from __future__ import annotations

import gzip
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from t33_spanmap import spans_from_generation

try:  # torch is only needed on the eval (server) side; keep import errors loud
    import torch
except ImportError as _exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = _exc


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"t33_capture needs torch on the eval side: {_TORCH_IMPORT_ERROR}")


def _to_fp16_cpu(tensor: Any) -> np.ndarray:
    return tensor.detach().to(torch.float16).cpu().numpy()


class T33CaptureContext:
    """Per-process capture state; one instance per (arm, part-shard)."""

    def __init__(
        self,
        out_dir: str,
        *,
        part: str = "p0",
        topk: int = 5,
        ctx_layer_stride: int = 4,
        capture_context: bool = True,
        capture_gist_stats: bool = True,
        max_anchor_tokens: int = 24,
    ) -> None:
        _require_torch()
        self.out_dir = Path(out_dir)
        self.part = part
        self.topk = int(topk)
        self.ctx_layer_stride = int(ctx_layer_stride)
        self.capture_context = bool(capture_context)
        self.capture_gist_stats = bool(capture_gist_stats)
        self.max_anchor_tokens = int(max_anchor_tokens)

        self.arm = "unknown"
        self.errors: Dict[str, int] = {}
        self._opened = False

        # per-row state
        self._qid: Optional[str] = None
        self._row_meta: Dict[str, Any] = {}
        self._doc_info: Optional[Dict[str, Any]] = None
        self._gist_stats: Optional[Dict[str, Any]] = None
        self._ctx_capture: Dict[str, Any] = {}
        self._gen_buffers: Dict[int, List[Any]] = {}
        self._gen_hooks: List[Tuple[Any, Any]] = []
        self._cache_len: Optional[int] = None
        self._overhead_start: Optional[float] = None

        self._steps_fh = None
        self._docs_fh = None
        self._hid_store: Dict[str, Dict[str, np.ndarray]] = {}

    # ---------------- lifecycle ----------------

    def open(self) -> None:
        """Mark the context open; per-arm files are created lazily by set_arm
        (the arm is not known until the first row)."""
        self._opened = True
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _switch_part_files(self) -> None:
        self._close_files()
        arm_dir = self.out_dir / self.arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        self._steps_fh = (arm_dir / f"{self.part}.steps.jsonl").open("a", encoding="utf-8")
        self._docs_fh = (arm_dir / f"{self.part}.docs.jsonl").open("a", encoding="utf-8")

    def _close_files(self) -> None:
        for fh in (self._steps_fh, self._docs_fh):
            if fh is not None:
                fh.close()
        self._steps_fh = self._docs_fh = None

    def close(self) -> None:
        self._close_files()
        self.flush_hidden_store()

    def flush_hidden_store(self) -> None:
        if not self._hid_store:
            return
        arm_dir = self.out_dir / self.arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arm_dir / f"{self.part}.hid.npz", **{
            f"{qid}::{key}": arr for qid, entry in self._hid_store.items() for key, arr in entry.items()
        })
        self._hid_store = {}

    def set_arm(self, arm: str) -> None:
        if arm != self.arm:
            self.arm = arm
            if self._opened:
                self._switch_part_files()

    def begin_row(self, qid: str, *, mode: str, ratio: Any) -> None:
        self._qid = qid
        self._row_meta = {"qid": qid, "mode": mode, "ratio": ratio}
        self._doc_info = None
        self._gist_stats = None
        self._ctx_capture = {}
        self._gen_buffers = {}
        self._cache_len = None
        self._overhead_start = time.perf_counter()

    def note_cache_length(self, cache_length: int) -> None:
        """Must be called before install_generation_hooks: the step-0 forward
        input is [mock cache ids; prompt] and the hook needs the split point."""
        self._cache_len = int(cache_length)

    def _bump_error(self, key: str) -> None:
        self.errors[key] = self.errors.get(key, 0) + 1

    # ---------------- generation hooks ----------------

    def install_generation_hooks(self, model: Any) -> None:
        """Layer-output hooks around ONE model.generate call.

        First forward per layer (prompt prefill over [mock cache; prompt])
        stores the last-position vector and the mean over the real prompt
        region; every later forward stores the new token's vector.
        """
        self.remove_generation_hooks(model)
        ctx = self

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                try:
                    hidden = output[0] if isinstance(output, tuple) else output
                    buf = ctx._gen_buffers.setdefault(layer_idx, [])
                    if not buf:
                        # prefill forward: [1, cache_len + prompt_len, H]
                        prompt_start = ctx._cache_len or 0
                        buf.append({
                            "query_last": hidden[0, -1].detach(),
                            "query_mean": hidden[0, prompt_start:].to(torch.float32).mean(dim=0).detach(),
                        })
                    else:
                        buf.append({"token": hidden[0, -1].detach()})
                except Exception:
                    ctx._bump_error("gen_hook")
            return hook

        for i, layer in enumerate(model.model.layers):
            handle = layer.register_forward_hook(make_hook(i))
            self._gen_hooks.append((handle, i))

    def remove_generation_hooks(self, model: Any) -> None:
        for handle, _i in self._gen_hooks:
            handle.remove()
        self._gen_hooks = []

    # ---------------- generate output ----------------

    def handle_generate_output(
        self,
        model: Any,
        tokenizer: Any,
        *,
        input_ids: Any,
        sequences: Any,
        scores: Sequence[Any],
        eos_id: int,
        max_new_tokens: int,
        tool_pool_names: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        """Build the steps/spans/IC record; consumes hook buffers."""
        # input_ids includes the mock-cache prefix; the continuation is:
        gen_ids = sequences[0, input_ids.shape[1]:].tolist()
        n_gen = len(gen_ids)
        self._cache_len = self._cache_len if self._cache_len is not None else 0

        steps = []
        for step, score_t in enumerate(scores):
            with torch.no_grad():
                logprobs = torch.log_softmax(score_t.to(torch.float32), dim=-1)[0]
                token_id = gen_ids[step]
                chosen = float(logprobs[token_id].item())
                eos_lp = float(logprobs[eos_id].item())
                probs = logprobs.exp()
                entropy = float((-(probs * logprobs).sum()).item())
                top_lp, top_idx = logprobs.topk(self.topk)
                top5 = [[float(v.item()), int(i.item())] for v, i in zip(top_lp, top_idx)]
            steps.append({
                "step": step,
                "token_id": token_id,
                "chosen_logprob": chosen,
                "eos_logprob": eos_lp,
                "entropy_full": entropy,
                "top5": top5,
            })

        if n_gen >= max_new_tokens:
            stop_reason = "length"
        elif gen_ids and gen_ids[-1] == eos_id:
            stop_reason = "eos"
        else:
            stop_reason = "other"

        decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
        spans = spans_from_generation(decode_fn, gen_ids)

        anchors = self._select_anchors(spans)
        hid_entry = self._extract_anchor_hiddens(anchors)
        ic = self._compute_ic(model, tokenizer, hid_entry, spans, tool_pool_names)

        record = {
            **self._row_meta,
            "generated_ids": gen_ids,
            "stop_reason": stop_reason,
            "stop_pos": len(steps) - 1 if steps else None,
            "steps": steps,
            "spans": {k: v for k, v in spans.items() if k != "text"},
            "text": spans["text"],
            "anchors": anchors,
            "ic": ic,
            "_hid_entry": hid_entry,
        }
        return record

    def _select_anchors(self, spans: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Ordered, de-duplicated anchor positions (token idx into continuation)."""
        n = spans.get("n_generated", 0)
        wanted: List[Tuple[str, int]] = [("first", 0), ("last", n - 1 if n else None), ("penult", n - 2 if n >= 2 else None)]
        nf, nl = spans.get("name_first"), spans.get("name_last")
        if nf is not None and nl is not None and nl >= nf:
            wanted.append(("name_first", nf))
            wanted.append(("name_last", nl))
            mid = (nf + nl) // 2
            if mid not in (nf, nl):
                wanted.append(("name_mid", mid))
        for label, key in (("args_first", "args_first"), ("args_last", "args_last")):
            v = spans.get(key)
            if v is not None:
                wanted.append((label, v))
        seen_pos = {}
        for label, pos in wanted:
            if pos is None or pos < 0 or pos >= n:
                continue
            seen_pos.setdefault(int(pos), label)
        # name span tokens (for span-level probes) — cap total anchors
        if nf is not None and nl is not None:
            for pos in range(nf, min(nl + 1, nf + 10)):
                if pos not in seen_pos and len(seen_pos) < self.max_anchor_tokens:
                    seen_pos.setdefault(int(pos), f"name_span_{pos}")
        anchors = [(label, pos) for pos, label in sorted(seen_pos.items(), key=lambda kv: kv[0])]
        return anchors

    def _layer_count(self) -> int:
        return len(self._gen_buffers)

    def _extract_anchor_hiddens(self, anchors: List[Tuple[str, int]]) -> Dict[str, Any]:
        """hidden_for_token(j) = layer buffer entry j+1 ('token' records)."""
        out: Dict[str, Any] = {}
        try:
            layer_ids = sorted(self._gen_buffers.keys())
            n_layers = len(layer_ids)
            query_last = []
            query_mean = []
            anchor_mats = [[] for _ in anchors]  # [anchor][layer]
            for li in layer_ids:
                buf = self._gen_buffers[li]
                first = buf[0] if buf else {}
                query_last.append(_to_fp16_cpu(first.get("query_last", torch.zeros(0))))
                query_mean.append(_to_fp16_cpu(first.get("query_mean", torch.zeros(0))))
                # token j -> buf[j+1]
                token_vecs = [b.get("token") for b in buf[1:]]
                for a_idx, (_label, pos) in enumerate(anchors):
                    vec = token_vecs[pos] if pos < len(token_vecs) else None
                    anchor_mats[a_idx].append(_to_fp16_cpu(vec) if vec is not None else None)
            out["layers"] = layer_ids
            out["query_last"] = np.stack(query_last) if n_layers else None
            out["query_mean"] = np.stack(query_mean) if n_layers else None
            anchor_arr = []
            anchor_valid = []
            for a_idx, (label, pos) in enumerate(anchors):
                col = anchor_mats[a_idx]
                valid = [v is not None for v in col]
                if any(valid):
                    dim = next(v.shape[0] for v in col if v is not None)
                    filled = [v if v is not None else np.zeros(dim, dtype=np.float16) for v in col]
                    anchor_arr.append(np.stack(filled))
                    anchor_valid.append(bool(all(valid)))
                else:
                    dim = query_last[0].shape[0] if n_layers else 0
                    anchor_arr.append(np.zeros((n_layers, dim), dtype=np.float16))
                    anchor_valid.append(False)
            out["anchor_labels"] = [label for label, _pos in anchors]
            out["anchor_positions"] = [pos for _label, pos in anchors]
            out["anchor_hidden"] = (np.stack(anchor_arr, axis=1) if anchor_arr else None)  # [L, A, H]
            out["anchor_valid"] = anchor_valid
        except Exception:
            self._bump_error("anchor_extract")
        return out

    def _compute_ic(
        self,
        model: Any,
        tokenizer: Any,
        hid_entry: Dict[str, Any],
        spans: Dict[str, Any],
        tool_pool_names: Optional[Sequence[str]],
    ) -> Optional[Dict[str, Any]]:
        """Candidate-restricted unembed Internal Consistency (4.4).

        Candidates = FIRST token ids of the session's tool names.  At the
        name_first / name_last anchors, per layer: restricted softmax argmax;
        agreement with the final layer's restricted argmax at the same anchor.
        """
        if not tool_pool_names:
            return None
        try:
            labels = hid_entry.get("anchor_labels") or []
            anchor_hidden = hid_entry.get("anchor_hidden")
            if anchor_hidden is None or not labels:
                return None
            cand_ids = []
            for name in tool_pool_names:
                ids = tokenizer.encode(name, add_special_tokens=False)
                if ids:
                    cand_ids.append(ids[0])
            cand_ids = sorted(set(cand_ids))
            if len(cand_ids) < 2:
                return {"note": "pool collapsed to <2 first tokens", "n_candidates": len(cand_ids)}
            weight = model.get_output_embeddings().weight
            w = weight[torch.tensor(cand_ids, device=weight.device)].to(torch.float32)  # [C, H]
            n_layers = anchor_hidden.shape[0]

            def restricted(h_f16: np.ndarray):
                h = torch.from_numpy(h_f16.astype(np.float32)).to(w.device)
                logits = w @ h
                p = torch.softmax(logits, dim=0)
                top = torch.topk(p, 2)
                return int(top.indices[0].item()), float(top.values[0].item()), float(top.values[1].item())

            out_anchors: Dict[str, Dict[str, Any]] = {}
            for label in ("name_first", "name_last"):
                if label not in labels:
                    continue
                a = labels.index(label)
                col = anchor_hidden[:, a, :]  # [L, H]
                choices = []
                p1s, p2s = [], []
                for li in range(n_layers):
                    c, p1, p2 = restricted(col[li])
                    choices.append(c)
                    p1s.append(p1)
                    p2s.append(p2)
                final = choices[-1]
                agree = [1 if c == final else 0 for c in choices]
                n_layer = len(agree)
                out_anchors[label] = {
                    "choices": choices,
                    "agree": agree,
                    "ic_uniform": float(np.mean(agree)) if n_layer else None,
                    "ic_lastk": float(np.mean(agree[-12:])) if n_layer >= 12 else (float(np.mean(agree)) if n_layer else None),
                    "first_agree_layer": int(np.argmin([1 - a for a in agree])) if agree else None,
                    "margin_final": (p1s[-1] - p2s[-1]) if p1s else None,
                    "final_choice_index": final,
                }
            return {"n_candidates": len(cand_ids), "candidate_token_ids": cand_ids, "anchors": out_anchors}
        except Exception:
            self._bump_error("ic")
            return None

    # ---------------- context side ----------------

    @contextmanager
    def capture_plain_forward(
        self,
        model: Any,
        positions_by_row: Optional[Sequence[Sequence[int]]] = None,
        flat_positions: Optional[Sequence[int]] = None,
        *,
        want_oproj: bool = True,
        tag: str = "ctx",
    ):
        """Hooks around ONE context-building forward (history prefill or
        generate_gist).  ``flat_positions`` are absolute last-dim indices
        ([1, L] input); ``positions_by_row`` for grid forwards ([B, L]).
        Stores per selected layer: hidden at positions (stride-subsampled)
        plus o_proj inputs at the same positions."""
        if not self.capture_context:
            yield self
            return
        ctx = self
        stride = max(1, self.ctx_layer_stride)
        hooks: List[Any] = []
        collected: Dict[str, List[Any]] = {"hid": [], "oproj": []}
        meta: Dict[str, Any] = {"tag": tag, "hid_layers": [], "oproj_layers": []}

        def sel_positions(hidden_shape):
            # hidden: [B, L, H]
            if positions_by_row is not None:
                out = []
                for row, pos in enumerate(positions_by_row):
                    for p in pos:
                        if 0 <= p < hidden_shape[1]:
                            out.append((row, p))
                return out
            if flat_positions is not None:
                return [(0, p) for p in flat_positions if 0 <= p < hidden_shape[1]]
            return []

        def make_layer_hook(layer_idx: int):
            def hook(_m, _i, output):
                try:
                    if layer_idx % stride and layer_idx != len(model.model.layers) - 1:
                        return
                    hidden = output[0] if isinstance(output, tuple) else output
                    pos = sel_positions(hidden.shape)
                    if not pos:
                        return
                    rows = torch.tensor([r for r, _p in pos], device=hidden.device)
                    cols = torch.tensor([p for _r, p in pos], device=hidden.device)
                    collected["hid"].append(hidden[rows, cols].detach().to(torch.float16).cpu().numpy())
                    meta["hid_layers"].append(layer_idx)
                except Exception:
                    ctx._bump_error(f"{tag}_hid")
            return hook

        def make_oproj_hook(layer_idx: int):
            def hook(_m, args, _output):
                try:
                    if layer_idx % stride and layer_idx != len(model.model.layers) - 1:
                        return
                    inp = args[0] if isinstance(args, tuple) else args
                    pos = sel_positions(inp.shape)
                    if not pos:
                        return
                    rows = torch.tensor([r for r, _p in pos], device=inp.device)
                    cols = torch.tensor([p for _r, p in pos], device=inp.device)
                    collected["oproj"].append(inp[rows, cols].detach().to(torch.float16).cpu().numpy())
                    meta["oproj_layers"].append(layer_idx)
                except Exception:
                    ctx._bump_error(f"{tag}_oproj")
            return hook

        try:
            layers = model.model.layers
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(make_layer_hook(i)))
                if want_oproj:
                    hooks.append(layer.self_attn.o_proj.register_forward_hook(make_oproj_hook(i)))
            yield self
        finally:
            for h in hooks:
                h.remove()
            if collected["hid"]:
                self._ctx_capture[f"{tag}_hid"] = np.stack(collected["hid"])  # [Sel, K, H]
            if collected["oproj"]:
                self._ctx_capture[f"{tag}_oproj"] = np.stack(collected["oproj"])
            self._ctx_capture[f"{tag}_meta"] = meta

    def record_last_context_position(self, n_positions: int) -> None:
        """The flat input length of the context forward just captured — used
        to also keep the last position (joint-probe context side) even though
        it is generally part of the boundary set for the full arm."""
        self._ctx_capture["ctx_input_len"] = int(n_positions)

    # ---------------- docs / gist sidecar ----------------

    def record_history_docs(
        self,
        docs: Sequence[Dict[str, Any]],
        doc_token_lens: Optional[Sequence[int]],
        n_docs_original: Optional[int],
    ) -> None:
        """Doc sidecar + the in-process 4.1 lexical features (zero GPU)."""
        try:
            import gzip as _gzip

            from d_witness_core import leaves, occurs

            texts = [str(d.get("content") or "") for d in docs]
            n_kept = len(texts)
            gzip_lens = [len(_gzip.compress(t.encode("utf-8"))) for t in texts]
            chars = [len(t) for t in texts]

            # surprise(k): IDF-weighted overlap of doc k's own leaf vocabulary
            # against the concatenation of the OTHER docs (gold-free).
            per_doc_leaves = []
            for t in texts:
                try:
                    parsed = json.loads(t)
                except Exception:
                    parsed = None
                vals = []
                if isinstance(parsed, dict):
                    vals = [str(v) for v in leaves(parsed) if len(str(v)) >= 3]
                if not vals:
                    vals = [w for w in t.split() if len(w) >= 8][:64]
                per_doc_leaves.append(sorted(set(vals))[:128])
            df: Dict[str, int] = {}
            for vals in per_doc_leaves:
                for v in set(vals):
                    df[v] = df.get(v, 0) + 1
            n_docs = len(texts)
            surprise = []
            for k, vals in enumerate(per_doc_leaves):
                others = " ".join(texts[j] for j in range(n_docs) if j != k)
                score = 0.0
                hits = 0
                for v in vals:
                    idf = math.log(1.0 + n_docs / (1 + df.get(v, 0)))
                    if occurs(v, others):
                        score += idf
                        hits += 1
                surprise.append({"score": round(score, 4), "hit_rate": round(hits / len(vals), 4) if vals else None})

            self._doc_info = {
                "n_docs_kept": n_kept,
                "n_docs_original": n_docs_original,
                "dropped_docs": (n_docs_original - n_kept) if n_docs_original is not None else None,
                "doc_chars": chars,
                "doc_token_lens": list(doc_token_lens) if doc_token_lens is not None else None,
                "doc_gzip_lens": gzip_lens,
                "doc_gzip_ratios": [round(c / g, 4) if g else None for c, g in zip(chars, gzip_lens)],
                "surprise": surprise,
                "doc_text_sha256": [__import__("hashlib").sha256(t.encode("utf-8")).hexdigest() for t in texts],
            }
        except Exception:
            self._bump_error("doc_sidecar")
            self._doc_info = {"error": "doc_sidecar_failed"}

    def record_gist_stats(
        self,
        cache: Any,
        system_length: int,
        per_row_gist_counts: Optional[Sequence[int]],
        gist_stats_stride: int = 4,
    ) -> None:
        """Per-doc gist saturation stats (4.1): Hoyer, DCT spectral entropy,
        excess kurtosis, L2 norms — over the head_dim feature axis, pooled
        across positions within a doc, then min/mean/max/std across heads.

        Expensive stats run on stride-subsampled layers; norms on all layers.
        DCT via numpy rfft fallback when torch.fft is unavailable.
        """
        if not self.capture_gist_stats:
            return
        try:
            counts = list(per_row_gist_counts or [])
            if not counts or sum(counts) <= 0:
                self._gist_stats = {"error": "no_gist_counts"}
                return
            bounds = np.cumsum([0] + counts)
            total_gists = int(bounds[-1])
            per_layer: Dict[str, Any] = {}
            n_layers = len(cache.layers)
            for li, layer in enumerate(cache.layers):
                heavy = (li % gist_stats_stride == 0) or li == n_layers - 1
                stats_k = self._gist_tensor_stats(layer.keys, system_length, total_gists, bounds, heavy)
                stats_v = self._gist_tensor_stats(layer.values, system_length, total_gists, bounds, heavy)
                per_layer[str(li)] = {"k": stats_k, "v": stats_v}
            self._gist_stats = {
                "per_row_gist_counts": [int(c) for c in counts],
                "layers_reported_heavy_stride": gist_stats_stride,
                "per_layer": per_layer,
            }
        except Exception as exc:
            self._bump_error("gist_stats")
            self._gist_stats = {"error": f"gist_stats_failed: {exc!r}"[:400]}

    def _gist_tensor_stats(self, tensor: Any, system_length: int, total_gists: int, bounds: np.ndarray, heavy: bool) -> Dict[str, Any]:
        """tensor: [B, heads, seq, head_dim]; gist region [system_length : system_length+total_gists]."""
        t = tensor[0, :, system_length:system_length + total_gists, :].to(torch.float32)  # [heads, G, D]
        abs_t = t.abs()
        l1 = abs_t.sum(dim=-1)
        l2 = abs_t.pow(2).sum(dim=-1).sqrt()
        hoyer = ((math.sqrt(t.shape[-1]) * l2 - l1) / (math.sqrt(t.shape[-1]) - 1)).clamp_min(0)
        norms = l2
        out: Dict[str, Any] = {
            "norm_mean_over_pos": _pool_heads(norms.mean(dim=1)),
            "norm_max_over_pos": _pool_heads(norms.max(dim=1).values),
        }
        if heavy:
            # spectral entropy of |DCT-II(v)|^2 over the feature axis, per (head, pos)
            x = t.cpu().numpy()
            dct = _dct2_axis(x)  # same shape
            p = dct ** 2
            p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                spec_ent = -(p * np.log(p + 1e-12)).sum(axis=-1)
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1) + 1e-8
            kurt = ((x - mu) ** 4).mean(axis=-1) / (sd ** 4) - 3.0
            out["spectral_entropy_mean_over_pos"] = _pool_np(spec_ent.mean(axis=-1))
            out["kurtosis_mean_over_pos"] = _pool_np(kurt.mean(axis=-1))
            out["hoyer_mean_over_pos"] = _pool_np(hoyer.cpu().numpy().mean(axis=-1))
            # per-doc aggregation for the heavy stats
            per_doc_hoyer = []
            per_doc_ent = []
            h_np = hoyer.cpu().numpy()
            for d in range(len(bounds) - 1):
                s, e = int(bounds[d]) - 0, int(bounds[d + 1])
                if e <= s:
                    per_doc_hoyer.append(None)
                    per_doc_ent.append(None)
                    continue
                per_doc_hoyer.append(_pool_np(h_np[:, s:e].mean(axis=1)))
                per_doc_ent.append(_pool_np(spec_ent[:, s:e].mean(axis=1)))
            out["per_doc_hoyer"] = per_doc_hoyer
            out["per_doc_spectral_entropy"] = per_doc_ent
        return out

    # ---------------- finalize ----------------

    def finish_row(
        self,
        record: Optional[Dict[str, Any]],
        *,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Write steps/docs sidecars, stash hidden arrays; returns the compact
        row summary (safe to embed in the eval row)."""
        if self._qid is None:
            return None
        overhead = round(time.perf_counter() - self._overhead_start, 4) if self._overhead_start else None
        summary: Dict[str, Any] = {"qid": self._qid, "capture_overhead_sec": overhead}
        try:
            hid = None
            if record is not None:
                hid = record.pop("_hid_entry", None)
                record.setdefault("meta", {}).update(extra_meta or {})
                record["meta"]["capture_overhead_sec"] = overhead
                record["meta"]["capture_errors"] = dict(self.errors)
                self._steps_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary.update({
                    "stop_reason": record.get("stop_reason"),
                    "spans": {k: record["spans"].get(k) for k in
                              ("parse_ok", "closed", "has_tool_call", "name_first", "name_last",
                               "args_first", "args_last", "n_generated")},
                    "ic_uniform_name_last": ((record.get("ic") or {}).get("anchors", {}).get("name_last") or {}).get("ic_uniform"),
                })
            docs_entry = dict(self._doc_info or {})
            if self._gist_stats is not None:
                docs_entry["gist_stats"] = self._gist_stats
            docs_entry["qid"] = self._qid
            docs_entry["arm"] = self.arm
            docs_entry["meta"] = extra_meta or {}
            self._docs_fh.write(json.dumps(docs_entry, ensure_ascii=False) + "\n")
            summary["docs_recorded"] = docs_entry.get("n_docs_kept") is not None
            summary["gist_stats_recorded"] = self._gist_stats is not None
            if record is not None and hid:
                entry: Dict[str, np.ndarray] = {}
                if hid.get("query_last") is not None:
                    entry["query_last"] = hid["query_last"]
                    entry["query_mean"] = hid["query_mean"]
                    if hid.get("anchor_hidden") is not None:
                        entry["anchor_hidden"] = hid["anchor_hidden"]
                for key, arr in self._ctx_capture.items():
                    if isinstance(arr, np.ndarray):
                        entry[key] = arr
                if entry:
                    self._hid_store[self._qid] = entry
                    summary["hidden_stored"] = True
            self._maybe_flush()
        except Exception:
            self._bump_error("finish_row")
            summary["error"] = "finish_row_failed"
        self._qid = None
        return summary

    def _maybe_flush(self) -> None:
        if len(self._hid_store) >= 64:
            self.flush_hidden_store()
        if self._steps_fh is not None:
            self._steps_fh.flush()
        if self._docs_fh is not None:
            self._docs_fh.flush()


def _pool_heads(t: Any) -> Dict[str, float]:
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return {
        "min": float(arr.min()), "mean": float(arr.mean()),
        "max": float(arr.max()), "std": float(arr.std()),
    }


def _pool_np(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(arr.min()), "mean": float(arr.mean()),
        "max": float(arr.max()), "std": float(arr.std()),
    }


def _dct2_axis(x: np.ndarray) -> np.ndarray:
    """DCT-II along the last axis via the rfft trick (numpy only)."""
    n = x.shape[-1]
    v = np.concatenate([x[..., ::2], x[..., 1::][..., ::-1]], axis=-1)
    dct = np.real(np.fft.rfft(v, axis=-1))[..., :n]
    return dct
