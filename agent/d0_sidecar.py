"""D0 sidecar: transient raw KV/Q capture during C2KV compression.

Contract (D0, 2026-08-30): the repair payload P_k must be produced inside the
NORMAL compression forward — no extra forward, no re-reading history.  Inside
`Qwen3Attention.forward_with_gist` (modeling_qwen3.py) the raw per-token Q/K/V
of every doc token are already computed with the BASE projections and then
discarded at the return.  Raw and gist states are projected in SEPARATE calls
(q_proj/k_proj/v_proj on the raw-sliced hidden_states; gist_q/k/v_proj on the
gist slice), so a forward hook on the base projections fires exactly once per
layer with the raw tokens only.  Because the dynamic-interleave token-token
mask is plain causal and tokens never attend gists, those raw locals are
bit-identical to a standalone causal prefill of the doc.

Storage convention: K is stored PRE-RoPE (position-free, codec-friendly);
repair-time absolute positions are applied via `inference.abs_rope
.apply_abs_rope` (full per-token RoPE through the model's own rotary module)
on release — NOT `rotate_k_cache_rope`, which is only legal for already-RoPE'd
K.  Q is captured only when ``want_q=True`` (teacher arms: GRKV / RESA /
KVSculpt); per prereg v2.5 the bytes bill then includes it (GQA 4:1 ⇒ Q alone
is ~2x the k+v bill).

Timing: ``last_compress_with_capture_sec`` measures hook registration + the
WHOLE compression forward + unregistration, device-synced on both ends (the
contract's T_capture term).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import torch


def _sync_device(device) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


class SidecarStore:
    """Persistent sidecar: qid -> per-layer {"q","k","v"} -> per-doc tensors.

    Tensor layout per doc: (heads, doc_len, head_dim) — kv heads for k/v,
    query heads for q, matching forward_with_gist's post-transpose layout
    with the batch (grid-row) dimension collapsed into the per-doc list.
    """

    def __init__(self, model: Any, max_entries: int = 4096, want_q: bool = False):
        self.model = model
        self.entries: Dict[str, List[Dict[str, List[torch.Tensor]]]] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}
        self._max_entries = max_entries
        self.want_q = bool(want_q)
        self.last_compress_with_capture_sec = 0.0

    def capture(self, qid: str, model_call, doc_lengths: Sequence[int]):
        """Run ``model_call()`` (the generate_gist invocation) with capture.

        ``doc_lengths[i]`` = valid token count in grid row i (the grid is
        right-padded, valid region left-aligned).  Returns model_call()'s
        result unchanged.  Raises on: duplicate qid, store full (both BEFORE
        the forward), batch/grid-row mismatch, any layer's hook not firing
        exactly once.
        """
        if qid in self.entries:
            raise RuntimeError(f"sidecar qid {qid!r} already captured; release() it first")
        if len(self.entries) >= self._max_entries:
            raise RuntimeError("sidecar store full")
        inner_model = getattr(self.model, "model", self.model)
        layers = inner_model.layers
        n_layers = len(layers)
        per_layer: List[Dict[str, List[torch.Tensor]]] = [
            {"q": [], "k": [], "v": []} for _ in range(n_layers)
        ]
        doc_lengths = list(doc_lengths)
        fire_counts: Dict[tuple, int] = {}

        def make_hook(idx: int, which: str, n_heads: int):
            attn = layers[idx].self_attn

            def hook(module, inputs, output):
                fire_counts[(idx, which)] = fire_counts.get((idx, which), 0) + 1
                if fire_counts[(idx, which)] != 1:
                    raise RuntimeError(
                        f"sidecar hook fired {fire_counts[(idx, which)]}x on layer {idx}/{which}; "
                        "expected exactly once per forward"
                    )
                # output: (B, S, n_heads * head_dim), raw tokens only
                b, s, hd = output.shape
                if b != len(doc_lengths):
                    raise RuntimeError(
                        f"sidecar grid-row mismatch on layer {idx}/{which}: "
                        f"batch {b} != doc_lengths {len(doc_lengths)}"
                    )
                head_dim = hd // n_heads
                t = output.view(b, s, n_heads, head_dim).transpose(1, 2)
                if which == "q" and getattr(attn, "q_norm", None) is not None:
                    t = attn.q_norm(t)
                elif which == "k" and getattr(attn, "k_norm", None) is not None:
                    t = attn.k_norm(t)
                rows = []
                for r in range(b):
                    L = doc_lengths[r]
                    assert 0 <= L <= s, f"doc_lengths[{r}]={L} outside [0, {s}]"
                    if L == 0:
                        # grid filler row (all -100): no real doc — skip, so
                        # the stored per-doc list indexes REAL docs in order
                        continue
                    # move to CPU immediately: keeping the full grid's raw
                    # K/V/Q on NPU alongside the model (~60 GB) causes OOM;
                    # the sidecar only returns to NPU at repair time
                    rows.append(t[r, :, :L, :].detach().to("cpu", copy=False))
                per_layer[idx][which] = rows

            return hook

        handles = []
        for idx, layer in enumerate(layers):
            attn = layer.self_attn
            # Qwen3Attention stores head counts only on the config, not self
            cfg = inner_model.config
            if self.want_q:
                handles.append(attn.q_proj.register_forward_hook(
                    make_hook(idx, "q", cfg.num_attention_heads)))
            handles.append(attn.k_proj.register_forward_hook(
                make_hook(idx, "k", cfg.num_key_value_heads)))
            handles.append(attn.v_proj.register_forward_hook(
                make_hook(idx, "v", cfg.num_key_value_heads)))
        start = time.perf_counter()
        _sync_device(getattr(self.model, "device", "cpu"))
        try:
            result = model_call()
        finally:
            _sync_device(getattr(self.model, "device", "cpu"))
            self.last_compress_with_capture_sec = time.perf_counter() - start
            for h in handles:
                h.remove()
        expected = {(idx, w) for idx in range(n_layers) for w in (("q", "k", "v") if self.want_q else ("k", "v"))}
        missing = sorted(expected - set(fire_counts))
        if missing:
            raise RuntimeError(f"sidecar hook never fired on {missing[:5]}")
        self.entries[qid] = per_layer
        self.meta[qid] = {
            "doc_lengths": doc_lengths,
            "want_q": self.want_q,
            "compress_with_capture_sec": round(self.last_compress_with_capture_sec, 4),
        }
        return result

    def get(
        self,
        qid: str,
        doc: int,
        which: str,
        device=None,
        dtype=None,
    ) -> List[torch.Tensor]:
        """Per-layer tensor for one doc: List[layer] of (heads, L, D).

        Captures live on CPU; pass the splice target's ``device``/``dtype``
        to get the round-trip cast on release (B8: torch.cat of a CPU tensor
        with an NPU cache tensor raises).
        """
        tensors = [layer[which][doc] for layer in self.entries[qid]]
        if device is not None or dtype is not None:
            tensors = [t.to(device if device is not None else t.device,
                            dtype if dtype is not None else t.dtype) for t in tensors]
        return tensors

    def bytes_of(self, qid: str, docs: Sequence[int] | None = None) -> int:
        """Cold-storage bytes for the stored (or a subset of) docs.

        Counts whatever is resident: k+v always; q as well when the store was
        created with want_q=True (prereg v2.5 — teacher arms bill their Q).
        """
        per_layer = self.entries.get(qid)
        if per_layer is None:
            return 0
        n_docs = len(per_layer[0]["k"])
        docs = docs if docs is not None else range(n_docs)
        total = 0
        for layer in per_layer:
            for d in docs:
                for which in ("k", "v", "q"):
                    t = layer[which][d] if layer[which] else None
                    if t is not None:
                        total += t.numel() * t.element_size()
        return total

    def drop_docs(self, qid: str, keep: Sequence[int]):
        """Free all docs except ``keep`` (oracle_target_only semantics).

        Reindexes: after drop_docs(qid, [3]) the surviving doc is index 0.
        """
        if qid not in self.entries:
            return
        for layer in self.entries[qid]:
            for which in ("k", "v", "q"):
                if layer[which]:
                    layer[which] = [layer[which][d] for d in keep]
        self.meta[qid]["doc_lengths"] = [
            self.meta[qid]["doc_lengths"][d] for d in keep
        ]

    def release(self, qid: str) -> None:
        self.entries.pop(qid, None)
        self.meta.pop(qid, None)
