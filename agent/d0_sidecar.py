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
repair-time absolute positions are applied via `rotate_k_cache_rope` on
release.  Q is captured for teacher-based arms (GRKV / RESA / KVSculpt).

Timing: `last_capture_sec` measures the hook overhead added to the normal
compression forward (the contract's T_capture term).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import torch


class SidecarStore:
    """Persistent sidecar: qid -> per-layer {"q","k","v"} -> per-doc tensors.

    Tensor layout per doc: (heads, doc_len, head_dim) — kv heads for k/v,
    query heads for q, matching forward_with_gist's post-transpose layout
    with the batch (grid-row) dimension collapsed into the per-doc list.
    """

    def __init__(self, model: Any, max_entries: int = 4096):
        self.model = model
        self.entries: Dict[str, List[Dict[str, List[torch.Tensor]]]] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}
        self._max_entries = max_entries
        self.last_capture_sec = 0.0

    def capture(self, qid: str, model_call, doc_lengths: Sequence[int]):
        """Run ``model_call()`` (the generate_gist invocation) with capture.

        ``doc_lengths[i]`` = valid token count in grid row i (the grid is
        right-padded, valid region left-aligned).  Returns model_call()'s
        result unchanged; raises if any layer's hook never fired.
        """
        inner_model = getattr(self.model, "model", self.model)
        layers = inner_model.layers
        n_layers = len(layers)
        per_layer: List[Dict[str, List[torch.Tensor]]] = [
            {"q": [], "k": [], "v": []} for _ in range(n_layers)
        ]
        doc_lengths = list(doc_lengths)
        start = time.perf_counter()

        def make_hook(idx: int, which: str, n_heads: int):
            attn = layers[idx].self_attn

            def hook(module, inputs, output):
                # output: (B, S, n_heads * head_dim), raw tokens only
                b, s, hd = output.shape
                head_dim = hd // n_heads
                t = output.view(b, s, n_heads, head_dim).transpose(1, 2)
                if which == "q" and getattr(attn, "q_norm", None) is not None:
                    t = attn.q_norm(t)
                elif which == "k" and getattr(attn, "k_norm", None) is not None:
                    t = attn.k_norm(t)
                rows = []
                for r in range(b):
                    L = doc_lengths[r] if r < len(doc_lengths) else s
                    # move to CPU immediately: keeping the full grid's raw
                    # K/V/Q on NPU alongside the model (~60 GB) causes OOM;
                    # the sidecar only needs to return to NPU at repair time
                    rows.append(t[r, :, :L, :].detach().to("cpu", copy=False))
                per_layer[idx][which] = rows

            return hook

        handles = []
        for idx, layer in enumerate(layers):
            attn = layer.self_attn
            # Qwen3Attention stores head counts only on the config, not self
            cfg = inner_model.config
            handles.append(attn.q_proj.register_forward_hook(
                make_hook(idx, "q", cfg.num_attention_heads)))
            handles.append(attn.k_proj.register_forward_hook(
                make_hook(idx, "k", cfg.num_key_value_heads)))
            handles.append(attn.v_proj.register_forward_hook(
                make_hook(idx, "v", cfg.num_key_value_heads)))
        try:
            result = model_call()
        finally:
            for h in handles:
                h.remove()
        self.last_capture_sec = time.perf_counter() - start
        for idx in range(n_layers):
            if not per_layer[idx]["k"]:
                raise RuntimeError(f"sidecar hook never fired on layer {idx}")
        if len(self.entries) >= self._max_entries:
            raise RuntimeError("sidecar store full")
        self.entries[qid] = per_layer
        self.meta[qid] = {
            "doc_lengths": doc_lengths,
            "capture_sec": round(self.last_capture_sec, 4),
        }
        return result

    def get(self, qid: str, doc: int, which: str) -> List[torch.Tensor]:
        """Per-layer tensor for one doc: List[layer] of (heads, L, D)."""
        return [layer[which][doc] for layer in self.entries[qid]]

    def bytes_of(self, qid: str, docs: Optional[Sequence[int]] = None) -> int:
        """Cold-storage bytes for the stored (or a subset of) docs.

        Counts k+v (the repair payload); q excluded from the bytes bill
        unless a teacher arm explicitly needs it (then it bills itself).
        """
        per_layer = self.entries.get(qid)
        if per_layer is None:
            return 0
        n_docs = len(per_layer[0]["k"])
        docs = docs if docs is not None else range(n_docs)
        total = 0
        for layer in per_layer:
            for d in docs:
                total += layer["k"][d].numel() * layer["k"][d].element_size()
                total += layer["v"][d].numel() * layer["v"][d].element_size()
        return total

    def drop_docs(self, qid: str, keep: Sequence[int]):
        """Free all docs except ``keep`` (oracle_target_only semantics)."""
        if qid not in self.entries:
            return
        for layer in self.entries[qid]:
            for which in ("k", "v", "q"):
                layer[which] = [layer[which][d] for d in keep]
        self.meta[qid]["doc_lengths"] = [
            self.meta[qid]["doc_lengths"][d] for d in keep
        ]

    def release(self, qid: str) -> None:
        del self.entries[qid]
        self.meta.pop(qid, None)
