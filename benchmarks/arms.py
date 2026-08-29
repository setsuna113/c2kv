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
    constrain_tools: bool = False  # H1: xgrammar structural-tag decoding on <tool_call>
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
        # The docstring above used to claim the proxy rejected repair arms.  It
        # did not: proxy._assemble never reads Arm.repair, so a repair arm ran
        # as a plain c2kv/hybrid arm and its rows were labelled with the repair
        # arm's name.  Until the server-side KV primitives land, fail loudly
        # rather than emit mislabelled numbers.
        if self.repair:
            raise NotImplementedError(
                f"arm {self.name!r} declares repair={self.repair!r}, but no repair "
                "primitive exists in proxy.py/hf_server.py yet. Running it would "
                "silently produce plain-compression numbers under a repair label."
            )


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
        # hybrid_top_k counts MESSAGES, which is also what the battery counts
        # (a history "doc" there is one message, split if it exceeds
        # max_doc_length).  It is NOT turns: an agent turn is an assistant
        # tool-call message plus its tool result, so k messages ~ k/2 turns.
        # The compression unit is the doc, so the doc is the only unit in which
        # "keep the recent ones raw" is well defined -- k=1/3/5 here is the
        # plan's k=1/3/5 read in that unit.
        Arm(
            name="hybrid1",
            compress_history=True,
            ratio=8,
            hybrid_top_k=1,
            description="top-1 tail message raw, earlier history gist (~0.5 turn)",
        ),
        Arm(
            name="hybrid",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            description="top-3 tail messages raw, earlier history gist (~1.5 turns)",
        ),
        Arm(
            name="hybrid5",
            compress_history=True,
            ratio=8,
            hybrid_top_k=5,
            description="top-5 tail messages raw, earlier history gist (~2.5 turns)",
        ),
        Arm(
            name="cd_full",
            compress_history=False,
            constrain_tools=True,
            description="H1: full raw history + xgrammar structural-tag tool-call decoding",
        ),
        Arm(
            name="cd_c2kv",
            compress_history=True,
            ratio=8,
            constrain_tools=True,
            description="H1: 8x gist history + xgrammar structural-tag tool-call decoding",
        ),
    )
}


def get_arm(name: str) -> Arm:
    try:
        arm = ARMS[name]
    except KeyError:
        known = ", ".join(sorted(ARMS))
        raise SystemExit(f"FATAL: unknown arm {name!r}; known arms: {known}")
    arm.validate()
    return arm
