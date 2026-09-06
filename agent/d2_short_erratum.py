"""D2: Short new erratum — v2 rewrite (2026-08-30).

Idea source: Models Take Notes at Prefill (arXiv:2606.17107).

Arm `short_erratum`: c2kv prefix + ONE forward of a compact correction
sentence composed ONLY of witness literal values (prereg v2.6 leak boundary):

> the erratum may contain only literal values that occurred in doc k*,
> NEVER the tool name, NEVER values absent from history.

The v1 file pasted the gold target ("The next action should be:
name(args)") into the prompt — answer leakage, not an erratum.  v2 draws
its values from the frozen witness table (target_doc_values: values found
in the decoded doc k*, tool name excluded), advances the position ledger
by the erratum tokens (v1 made erratum and prompt share absolute
positions), and device-syncs T_edit.

`short_erratum_kvbank` stays UNIMPLEMENTED until the D1 verdict (plan §2.5).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH

logger = logging.getLogger(__name__)

ERRATUM_TEMPLATE = "[correction] The following values from the earlier record are authoritative: {values}. Treat them as overriding any conflicting summary."


def compose_erratum_text(target_doc_values: List[List[Any]]) -> str:
    """Deterministic erratum from the witness table's literal values.

    ``target_doc_values`` is [[value, df], ...] (df-ascending, tool name
    already excluded, every value present in decoded doc k*).  All values
    are included — no hidden knobs; the token count is measured and
    reported per row.
    """
    rendered = "; ".join(str(v) for v, _df in target_doc_values)
    return ERRATUM_TEMPLATE.format(values=rendered)


@torch.inference_mode()
def build_short_erratum_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
    store: Optional[Any] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """c2kv prefix + one correction-sentence forward (counted as T_edit)."""
    from d1_arms import prepare_d_contract_state, _merge_system_gist

    if mode != "d_short_erratum":
        return None, f"d2_unknown_mode:{mode}"
    witness = HH.D_CONTRACT_WITNESS.get(example.qid)
    if witness is None:
        # the table is frozen before D2 runs; a missing entry is an
        # implementation error, not a scored outcome
        return None, "d2_no_witness_entry"

    state, skip_reason = prepare_d_contract_state(model, tokenizer, example, args, store)
    if state is None:
        return None, skip_reason
    if store is not None:
        store.release(example.qid)  # D2 never injects KV payloads
    prefix_cache = _merge_system_gist(state, model.config)
    system_length = state["system_length"]
    doc_tokens = state["doc_tokens"]
    k_star = witness.get("k_witness")

    d_mode_info: Dict[str, Any] = {
        "k_policy": "witness",
        "k_star": k_star,
        "sidecar_bytes_all": 0,
        "sidecar_bytes_target": 0,
        "sidecar_bytes_used": 0,
        "t_capture_sec": round(state["t_capture"], 4),
        "gist_span_of_target": None,
        "k_anchor": None,
        "erratum_values": [],
    }

    if k_star is None or not witness.get("target_doc_values"):
        # no literal witness (or none inside doc k*): nothing may be said
        # within the leak boundary -> explicit no-injection row
        d_mode_info.update(
            injected=False,
            note="k_star=None / no witness values: leak boundary forbids any erratum",
        )
        cache_length = prefix_cache.get_seq_length()
        return {
            "cache": prefix_cache,
            "system_length": system_length,
            "history_length": doc_tokens,
            "cache_length": cache_length,
            "doc_tokens": doc_tokens,
            "doc_chunks": state["doc_chunks"],
            "kept_history_tokens": doc_tokens,
            "gist_tokens": state["total_gist_tokens"],
            "actual_compression_ratio": float(doc_tokens / max(1, cache_length - system_length)),
            "system_prefill_sec": state["system_prefill_sec"],
            "full_prefill_sec": 0.0,
            "tool_compress_sec": state["t_capture"],
            "blend_sec": 0.0,
            "use_gist": True,
            "d_corr_doc_index": k_star,
            "d_corr_span_tokens": 0,
            "d_sham_tokens": 0,
            "d_recompute_tokens": 0,
            "d_recompute_docs": 0,
            "d_dropped_gist_tokens": 0,
            "d_corr_slice_prefill_sec": 0.0,
            "d_recompute_prefill_sec": 0.0,
            "d_contract_info": d_mode_info,
        }, None

    # --- compose the erratum inside the leak boundary (prereg v2.6) ---
    target_doc_values = witness["target_doc_values"]
    erratum_text = compose_erratum_text(target_doc_values)
    tool_name = witness.get("tool_name")
    assert tool_name is None or tool_name not in erratum_text, "leak boundary: tool name in erratum"

    erratum_ids = HH._chat_template_ids(
        tokenizer, [{"role": "user", "content": erratum_text}]
    )
    erratum_input = torch.tensor([erratum_ids], dtype=torch.long, device=model.device)
    erratum_tokens = len(erratum_ids)
    logical_start = system_length + doc_tokens  # tail: after the whole history
    past_length = prefix_cache.get_seq_length()

    # --- erratum forward = T_edit (device-synced on both ends) ---
    HH._sync_device(model.device)
    t_edit_start = time.perf_counter()
    attention_mask = torch.ones(
        1, past_length + erratum_input.shape[1], device=model.device, dtype=torch.long
    )
    position_ids = torch.arange(
        logical_start, logical_start + erratum_input.shape[1], device=model.device
    ).unsqueeze(0)
    model.model.config._attn_implementation = args.generate_attn_impl
    erratum_out = model(
        input_ids=erratum_input,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=prefix_cache,
        use_cache=True,
        use_gist=True,  # the cache holds gist KV -> gist projections apply
        logits_to_keep=1,
    )
    prefix_cache = erratum_out.past_key_values
    HH._sync_device(model.device)
    t_edit = time.perf_counter() - t_edit_start

    # ledger advance: the erratum occupies logical positions
    # [system+doc_tokens, system+doc_tokens+erratum_tokens); decode continues
    # after it — no position collision (v1 bug: shared absolute positions)
    history_length = doc_tokens + erratum_tokens
    cache_length = prefix_cache.get_seq_length()
    d_mode_info.update(
        injected=True,
        erratum_values=[v for v, _df in target_doc_values],
        erratum_text=erratum_text[:300],
        erratum_tokens=erratum_tokens,
        t_edit_sec=round(t_edit, 4),
        k_anchor=logical_start,
    )
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": history_length,
        "cache_length": cache_length,
        "doc_tokens": doc_tokens,
        "doc_chunks": state["doc_chunks"],
        "kept_history_tokens": doc_tokens,
        "gist_tokens": state["total_gist_tokens"],
        # resident footprint includes the erratum tokens (v1 ignored them)
        "actual_compression_ratio": float(doc_tokens / max(1, cache_length - system_length)),
        "system_prefill_sec": state["system_prefill_sec"],
        "full_prefill_sec": 0.0,
        "tool_compress_sec": state["t_capture"],
        "blend_sec": 0.0,
        "use_gist": True,
        "d_corr_doc_index": k_star,
        "d_corr_span_tokens": erratum_tokens,
        "d_sham_tokens": 0,
        "d_recompute_tokens": 0,
        "d_recompute_docs": 0,
        "d_dropped_gist_tokens": 0,
        "d_corr_slice_prefill_sec": round(t_edit, 4),
        "d_recompute_prefill_sec": 0.0,
        "d_contract_info": d_mode_info,
    }, None
