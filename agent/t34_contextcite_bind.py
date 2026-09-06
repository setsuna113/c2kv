#!/usr/bin/env python
"""t34 ContextCite server-side binding (digest S4.7, unit U5; SERVER: torch + the NPU model).

``agent/t34_contextcite.py run --score_module agent.t34_contextcite_bind:factory
--score_module_sham agent.t34_contextcite_bind:sham_factory`` loads these two
factories.  Each returns ``score_fn(qid, v) -> log p(reference action span |
ablate(C, v), Q)`` under the module's FLIPPED ablation semantics:

  v_k = 1  block k is RESTORED TO RAW K/V -- the D-line raw_keepG splice: the
           block's pre-RoPE K/V from the compression sidecar, rotated onto its
           absolute start and appended behind the FULL gist grid.  This is
           ``d1_arms.ksweep_prefix_for_k`` generalised to a SET of blocks
           appended in block order (the same primitive the flip table's k-sweep
           and ``t34_stage2_driver`` use).
  v_k = 0  block k stays gisted.
  v = 0    the plain c2kv prefix (the frozen arm's own context; the paper's
           "unablated" point under the flipped semantics).

FRESH PREFIX PER CALL.  ``d1_arms._merge_system_gist`` rebuilds the cache from
the per-qid immutable tensors on every call and every appended span is a new
tensor, so nothing scored can leak between ablations.  The per-qid state
(one compression forward with sidecar capture) is the only thing kept across
calls; its fingerprint is exposed as ``score_fn.state_fingerprint`` and
``t34_contextcite.run_attribution`` reads it before and after every call.

REFERENCE SPAN.  The compressed arm's OWN emitted continuation (t33 capture
ids first, else the battery prediction re-tokenised and roundtrip-checked --
``t34_extra_forward.emitted_ids_for_row``), teacher-forced through
``t33_svip_gamma._score_under_prefix`` (router convention).  The scored tokens
are the action payload span from ``t33_spanmap.spans_from_generation``
(``payload_first..payload_last``); when that span is unmapped the whole
continuation is scored and the qid is listed under ``span_fallback_qids`` in
the binding's ledger (``T34_CC_LEDGER``), so the two are never pooled silently.

SHAM (2409.00729 card pitfall 2).  ``sham_factory`` restores, for each k with
v_k = 1, an EQUAL-LENGTH NEUTRAL span instead of the raw block: len(doc_ids[k])
ids from the frozen neutral corpus ring (``d_sham_plan.corpus_offset(SEED,
f"{qid}:{k}")`` + ``ring_slice``), prefilled standalone
(``HH._prefill_ids_no_past``) and appended at the block's logical start
(``HH._append_span_cache``) -- the harness's ``d_sham_neutral`` construction,
per block.  Same call schema; ``placement="sham"``.

CONFIGURATION (the factories take no arguments): environment variables
  T34_CC_ROOT           worktree root (default .)
  T34_CC_MODEL / T34_CC_BASE / T34_CC_TOKENIZER / T34_CC_DATASET  (frozen r2 recipe defaults)
  T34_CC_DEVICE (npu)   T34_CC_ATTN (eager)   T34_CC_RATIO (8)
  T34_CC_CAPTURE_STEPS  t33 capture <dir>/c2kv/p0.steps.jsonl (preferred id source)
  T34_CC_CORPUS         configs/bdf_pilot/d_neutral_corpus.txt (sham only)
  T34_CC_SPAN           payload (default) | full
  T34_CC_LEDGER         where the binding writes its per-qid ledger jsonl

The raw and sham factories share one ``_Runtime`` (model, rows, sidecar store and the
per-qid state), so the pass loads the model once and compresses each qid once.
Everything except ``_Runtime`` / ``Binding`` is pure and unit-tested on a torch-free box.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "python", _HERE.parent / "python" / "inference", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)

SHAM_SEED = 20260815  # d_sham_plan.SEED (frozen)

DEVIATIONS: List[Dict[str, str]] = [
    {"what": "ablate(C, v) restores blocks with the sidecar raw_keepG splice (append "
             "behind the full gist grid) rather than physically removing sources",
     "why": "digest S4.7 flipped semantics: the deployable intervention is the "
            "D-line repair, and the flip table / stage-2 table use the same primitive"},
    {"what": "the reference response is the compressed arm's own frozen emission "
             "(payload span), not a fresh unablated generation",
     "why": "under the flipped semantics v = 0 IS the frozen arm's context; "
            "re-generating would score a different response than the one labelled"},
    {"what": "sham spans are drawn per (qid, block) from the neutral corpus ring "
             "instead of the frozen single-k* sham plan",
     "why": "the frozen plan holds one span per qid at k*; multi-block ablations "
            "need one equal-length span per restored block; the ring/offset rule is "
            "the plan's own (d_sham_plan.corpus_offset / ring_slice), seed kept"},
]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def blocks_from_v(v: Sequence[int]) -> List[int]:
    """Indices k with v_k = 1, ascending (append order)."""
    return [k for k, x in enumerate(v) if int(x) == 1]


def scored_slice(spans: Dict[str, Any], n_ids: int, mode: str = "payload"
                 ) -> Tuple[int, int, str]:
    """Inclusive (first, last) token range to sum log-probs over.

    ``payload``: the action payload span when mapped, else the whole
    continuation (flagged ``full_fallback``); ``full``: always the whole
    continuation.  ``n_ids`` must be >= 1."""
    if n_ids <= 0:
        raise ValueError("empty continuation")
    if mode == "full":
        return 0, n_ids - 1, "full"
    if mode != "payload":
        raise ValueError(f"unknown span mode {mode!r}")
    a, b = spans.get("payload_first"), spans.get("payload_last")
    if a is None or b is None or a > b or b >= n_ids:
        return 0, n_ids - 1, "full_fallback"
    return int(a), int(b), "payload"


def neutral_span_ids(corpus_ids: Sequence[int], qid: str, k: int, length: int,
                     seed: int = SHAM_SEED) -> List[int]:
    """Equal-length neutral ids for (qid, block k): the sham plan's ring rule
    keyed by ``f"{qid}:{k}"`` so every block of a qid gets its own offset."""
    from d_sham_plan import corpus_offset, ring_slice
    offset = corpus_offset(seed, f"{qid}:{k}", len(corpus_ids))
    return ring_slice(corpus_ids, offset, length)


def env_config(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    e = os.environ if env is None else env
    return {
        "root": e.get("T34_CC_ROOT", "."),
        "model": e.get("T34_CC_MODEL", "/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint"),
        "base_model": e.get("T34_CC_BASE", "/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507"),
        "tokenizer": e.get("T34_CC_TOKENIZER", "/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507"),
        "dataset_path": e.get("T34_CC_DATASET", "/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2"),
        "device_type": e.get("T34_CC_DEVICE", "npu"),
        "attn_impl": e.get("T34_CC_ATTN", "eager"),
        "ratio": int(e.get("T34_CC_RATIO", "8")),
        "max_doc_length": 768, "max_doc_num": 16, "max_new_tokens": 128,
        "capture_steps": e.get("T34_CC_CAPTURE_STEPS", ""),
        "corpus": e.get("T34_CC_CORPUS", "configs/bdf_pilot/d_neutral_corpus.txt"),
        "span": e.get("T34_CC_SPAN", "payload"),
        "ledger": e.get("T34_CC_LEDGER", "results/t34/contextcite_bind_ledger.jsonl"),
    }


# ---------------------------------------------------------------------------
# the binding (torch; server only)
# ---------------------------------------------------------------------------

class _Runtime:  # pragma: no cover - torch + NPU
    """Everything the raw and sham bindings SHARE: the loaded model, the
    frozen rows, the sidecar store, and the per-qid state (one compression
    forward with sidecar capture, prepared once per qid and reused by both
    placements -- ``run_attribution`` scores every design point through the
    raw scorer and then through the sham scorer on the SAME qid)."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        import argparse
        import torch  # noqa: F401
        import d_ksweep_driver as KD
        import eval_agent_history_c2kv as HH
        from d0_sidecar import SidecarStore
        from t34_common import FrozenAssets
        from t34_extra_forward import load_capture_steps

        self.cfg = cfg or env_config()
        self.HH = HH
        ns = argparse.Namespace(**{k: self.cfg[k] for k in (
            "model", "base_model", "tokenizer", "dataset_path", "device_type", "attn_impl",
            "ratio", "max_doc_length", "max_doc_num", "max_new_tokens")})
        self.hargs = KD._harness_args(ns)
        self.frame = FrozenAssets(Path(self.cfg["root"])).load()
        qids = [r["qid"] for r in self.frame.trigger_subset()]
        self.hargs.qid_allowlist = set(qids)
        self.tokenizer = HH._load_tokenizer(self.hargs)
        examples, _ = HH._load_examples(self.hargs, self.tokenizer)
        self.examples = {e.qid: e for e in examples}
        self.device = HH._setup_device(self.cfg["device_type"])
        self.hargs.mode = "c2kv"
        self.model = HH._load_model(self.hargs, self.tokenizer, self.device)
        self.store = SidecarStore(self.model)
        HH.D_CONTRACT_STORE = self.store
        HH.D_INTERVENE = {}
        cap = self.cfg.get("capture_steps")
        self.capture_index = load_capture_steps(Path(cap)) if cap and Path(cap).exists() else {}
        self._corpus_ids: Optional[List[int]] = None
        self.qid: Optional[str] = None
        self.state: Optional[Dict[str, Any]] = None
        self.prompt_ids: List[int] = []
        self.ids: List[int] = []
        self.slice: Tuple[int, int] = (0, 0)
        self.ledger_path = Path(self.cfg["ledger"])
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_ids(self) -> List[int]:
        if self._corpus_ids is None:
            text = Path(self.cfg["corpus"]).read_text(encoding="utf-8")
            self._corpus_ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if not self._corpus_ids:
                raise RuntimeError("empty neutral corpus")
        return self._corpus_ids

    def release(self) -> None:
        if self.qid is not None:
            self.store.release(self.qid)
        self.qid, self.state = None, None
        self.HH._clear_device_cache(self.cfg["device_type"])

    def ensure(self, qid: str) -> None:
        if self.qid == qid and self.state is not None:
            return
        self.release()
        from d1_arms import prepare_d_contract_state
        from t33_spanmap import spans_from_generation
        from t34_extra_forward import emitted_ids_for_row
        from train.train_data_multiturn import _chat_template_ids

        example = self.examples.get(qid)
        if example is None:
            raise KeyError(f"{qid}: not in the loaded eval rows")
        state, skip = prepare_d_contract_state(self.model, self.tokenizer, example, self.hargs, self.store)
        if state is None:
            raise RuntimeError(f"{qid}: harness skipped the row ({skip})")
        row_c = self.frame.c2kv_by_qid[qid]
        tok = self.tokenizer
        ids, src = emitted_ids_for_row(
            qid, self.capture_index, row_c.get("prediction") or "",
            lambda t: tok.encode(t, add_special_tokens=False),
            decode_fn=lambda i: tok.decode(i, skip_special_tokens=True))
        if not ids:
            raise RuntimeError(f"{qid}: no emitted ids to score")
        prompt_ids = _chat_template_ids(tok, self.HH._current_messages(example), add_generation_prompt=True)
        if self.hargs.max_prompt_tokens and len(prompt_ids) > self.hargs.max_prompt_tokens:
            prompt_ids = prompt_ids[-self.hargs.max_prompt_tokens:]
        spans = spans_from_generation(lambda i: tok.decode(i, skip_special_tokens=True), ids)
        a, b, kind = scored_slice(spans, len(ids), self.cfg["span"])
        self.qid, self.state = qid, state
        self.prompt_ids, self.ids, self.slice = list(prompt_ids), list(ids), (a, b)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"qid": qid, "ids_source": src, "n_ids": len(ids),
                                 "scored_slice": [a, b], "span_kind": kind,
                                 "n_docs": len(state["doc_ids"])}) + "\n")

    def fingerprint(self) -> Any:
        st = self.state
        if st is None:
            return None
        keys0 = st["gist_cache"].layers[0].keys
        return (self.qid, tuple(int(o) for o in st["offsets"]), int(st["total_gist_tokens"]),
                float(keys0.float().sum().item()), tuple(self.ids[:8]), self.slice)


class Binding:  # pragma: no cover - torch + NPU
    """Callable ``score_fn(qid, v)`` with a ``state_fingerprint()`` method;
    ``placement`` selects the raw_keepG restore or the equal-length sham."""

    def __init__(self, placement: str, runtime: _Runtime) -> None:
        if placement not in ("raw", "sham"):
            raise ValueError(placement)
        self.placement = placement
        self.rt = runtime

    def prepare(self, qid: str) -> None:
        """Build the per-qid state BEFORE the runner reads the first
        fingerprint: the state is prepared lazily otherwise, and a lazy first
        call would (correctly) trip the pollution sentinel with
        ``None -> <state>`` (seen on the 2026-09-06 NPU smoke)."""
        self.rt.ensure(qid)

    def state_fingerprint(self) -> Any:
        return self.rt.fingerprint()

    def __call__(self, qid: str, v: Sequence[int]) -> Optional[float]:
        import torch
        import t33_svip_gamma as SV
        from d1_arms import _cat_span_to_cache, _finish_prefix, _merge_system_gist, _sidecar_raw_span

        rt = self.rt
        rt.ensure(qid)
        st = rt.state
        assert st is not None
        d = len(st["doc_ids"])
        if len(v) != d:
            raise ValueError(f"{qid}: design d={len(v)} but the harness grid has {d} blocks")
        ks = blocks_from_v(v)
        model, HH = rt.model, rt.HH
        cache = _merge_system_gist(st, model.config)  # fresh cache from immutable tensors
        device, dtype = cache.layers[0].keys.device, cache.layers[0].keys.dtype
        span_tokens = 0
        for k in ks:
            anchor = int(st["offsets"][k])
            length = len(st["doc_ids"][k])
            if self.placement == "raw":
                span = _sidecar_raw_span(rt.store, qid, k, anchor, model.model.rotary_emb, device, dtype)
                cache = _cat_span_to_cache(cache, span)
            else:
                sham_ids = neutral_span_ids(rt.corpus_ids, qid, k, length)
                sham_t = torch.tensor([sham_ids], dtype=torch.long, device=model.device)
                sham_cache, _, _ = HH._prefill_ids_no_past(model, sham_t, rt.hargs.gist_attn_impl)
                cache = HH._append_span_cache(model, cache, sham_cache, anchor, list(range(length)))
                del sham_cache
            span_tokens += length
        prefix = _finish_prefix(
            st, cache, history_length=st["doc_tokens"], gist_tokens_final=st["total_gist_tokens"],
            span_tokens=span_tokens, dropped_gist_tokens=0,
            d_mode_info={"k_policy": "contextcite", "k_star": None, "blocks": ks,
                         "placement": self.placement, "injected": bool(ks)},
            t_load_sec=0.0)
        out = SV._score_under_prefix(model, rt.tokenizer, prefix, rt.prompt_ids, rt.ids,
                                     rt.cfg["attn_impl"])
        del prefix, cache
        HH._clear_device_cache(rt.cfg["device_type"])
        if out is None:
            return None
        a, b = rt.slice
        chosen = out["chosen_logprob"]
        return float(sum(chosen[a:b + 1]))


_RUNTIME: Dict[str, Any] = {}


def _runtime() -> _Runtime:  # pragma: no cover - server
    if "rt" not in _RUNTIME:
        _RUNTIME["rt"] = _Runtime()
    return _RUNTIME["rt"]


def factory():  # pragma: no cover - server
    """score_fn bound to the raw_keepG multi-block restore (shares the loaded
    model, rows and per-qid state with ``sham_factory``)."""
    return Binding("raw", _runtime())


def sham_factory():  # pragma: no cover - server
    """score_fn bound to the equal-length neutral-span restore (the sham floor)."""
    return Binding("sham", _runtime())
