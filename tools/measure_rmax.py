"""Measure the concrete R_max (recovery allowance) on the real runtime.

R_max = byte upper bound of ONE tokens_1024 evidence packet: the rendered
evidence message (real renderer, real chat template, real tokenizer) with a
maximal <=1024-token unit, page-rounded, valued at the engine's
kv_bytes_per_token. Run on ascend03 with the controller runtime import path.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
C1_DELIVERY = GENERATION_ROOT / "src" / "c1_delivery"
RUNTIME = GENERATION_ROOT / "src" / "generality" / "controller_runtime"
CKPT = Path(
    "/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
    "arm-C/seed-42/checkpoint-1000"
)
EMB = Path("/home/liuyancheng/c2kv-evidence-sets-20260916/models/Qwen3-Embedding-0.6B")
RISK_ARTIFACT = C1_DELIVERY / "artifacts" / "c1_risk.t02_v1.json"
PAGE = 128
KV_BYTES_PER_TOKEN = 147456

sys.path.insert(0, str(C1_DELIVERY))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "python"))

import evidence_sets  # noqa: E402
from history_memory.events import EventStore  # noqa: E402
from benchmarks.memory_runtime.recovery.evidence_units import (  # noqa: E402
    build_catalog,
    render_units,
)


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=False)
    config, _ = evidence_sets.build_config(
        history="H0", selector="risk", selector_artifact=RISK_ARTIFACT,
        selector_threshold=0.5, embedding_model=str(EMB), embedding_device="cpu",
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    unit_rule, presentation, order = config["U"], config["P"], config["order"]

    long_payload = json.dumps(
        {"records": [{"id": i, "field_a": "value-" + "a" * 30, "field_b": ["t"] * 6}
                     for i in range(140)]}
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please look up the records."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "m1", "type": "function",
             "function": {"name": "lookup_records",
                          "arguments": json.dumps({"scope": "all"})}}]},
        {"role": "tool", "tool_call_id": "m1", "content": long_payload},
        {"role": "user", "content": "Summarize the records."},
    ]
    store = EventStore.from_messages("rmax-measure", messages)
    units = build_catalog(store, tokenizer, unit_rule)
    if not units:
        raise SystemExit("no evidence units produced")
    def unit_tokens(u):
        for attr in ("token_count", "tokens"):
            v = getattr(u, attr, None)
            if isinstance(v, int):
                return v
        return len(tokenizer.encode(u.text))
    biggest = max(units, key=unit_tokens)
    unit_tok = unit_tokens(biggest)

    from history_memory.packing import native_ids

    rendered = render_units([biggest], store, presentation, order)
    base = len(native_ids(tokenizer, messages, generation=True))
    with_packet = len(native_ids(tokenizer, [*messages, *rendered], generation=True))
    packet_tokens = with_packet - base

    resident = int(math.ceil(packet_tokens / PAGE) * PAGE)
    r_max = resident * KV_BYTES_PER_TOKEN
    result = {
        "schema": "c2kv-generality-rmax-measurement-v1",
        "unit_rule": unit_rule, "presentation": presentation, "order": order,
        "max_unit_token_count": unit_tok,
        "packet_tokens_in_context": packet_tokens,
        "packet_resident_tokens_after_page_rounding": resident,
        "page_size": PAGE, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "recovery_allowance_bytes": r_max,
        "tokenizer": str(CKPT),
        "renderer": "controller_runtime recovery.evidence_units.render_units",
    }
    out = GENERATION_ROOT / "config" / "rmax_measurement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
