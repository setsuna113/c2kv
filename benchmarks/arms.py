"""Arm registry for the modular benchmark layer.

An arm fully describes how conversation history reaches the model, so the
proxy and the metrics layer share one source of truth.  Semantics mirror
agent/api/eval_agent_history_sglang_api.py:

* full    — every message is sent as raw text.
* c2kv    — every history message is compressed via /v1/c2kv/extract and
            referenced by c2kv_key_hash; the current turn stays raw.
* hybrid  — top-k tail of history stays raw, the rest is compressed.

Repair arms (corr / corr_re / splice_keep / splice_rep / offset / ...) will
be added here once their server-side KV primitives exist in the SGLang fork;
each entry then declares which history block keeps/gets raw KV in addition
to the gist reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class Arm:
    name: str
    compress_history: bool
    ratio: int = 8
    hybrid_top_k: int = 0  # 0 = none raw; >0 = keep this many tail messages raw
    # Reserved for repair-arm extensions (block selection, raw-KV append,
    # recompute fraction, offset correction strength).  The proxy rejects
    # arms that use these until implemented.
    repair: Optional[Dict[str, object]] = None
    description: str = ""

    def validate(self) -> None:
        if self.compress_history and self.ratio < 2:
            raise ValueError(f"arm {self.name!r}: compression ratio must be >= 2")
        if not self.compress_history and (self.hybrid_top_k or self.repair):
            raise ValueError(f"arm {self.name!r}: hybrid/repair knobs need compression")


ARMS: Dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm(
            name="full",
            compress_history=False,
            description="raw text history, no compression (upper reference)",
        ),
        Arm(
            name="c2kv",
            compress_history=True,
            ratio=8,
            description="all history as gist KV at 8x, current turn raw",
        ),
        Arm(
            name="c2kv16",
            compress_history=True,
            ratio=16,
            description="all history as gist KV at 16x (compression ladder)",
        ),
        Arm(
            name="hybrid",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            description="top-3 tail messages raw, earlier history gist",
        ),
    )
}


def get_arm(name: str) -> Arm:
    try:
        return ARMS[name]
    except KeyError:
        known = ", ".join(sorted(ARMS))
        raise SystemExit(f"FATAL: unknown arm {name!r}; known arms: {known}")
