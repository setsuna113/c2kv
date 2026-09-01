"""Arm registry for the modular benchmark layer.

An arm fully describes how conversation history reaches the model, so the
proxy and the metrics layer share one source of truth.  Hybrid semantics are
defined once in docs/hybrid_spec.md (canonical gist_first layout):

* full    — every message is sent as raw text.
* c2kv    — every history message is compressed via /v1/c2kv/extract and
            referenced by c2kv_key_hash; the current turn stays raw.
* hybrid  — top-k tail of history stays raw, the rest is compressed
            (hybrid / hybrid_k1 / hybrid_k5 = k 3/1/5).
* *_repair — corr@first on the compressed prefix (docs/hybrid_spec.md
            "Repair interaction", docs/c2kv_semantics.md "Repair placement"):
            the raw KV of the policy-selected compressed doc, computed in its
            full context, is injected with an explicit placement:
              append_keep_ledger (default) = D-harness corr / keepG, the
                doc's gist stays, the span keeps its original RoPE phase;
              append_tail (*_repair_tail) = D-harness raw_erratum_tail, the
                span is re-rotated to the end of history and the ledger
                advances;
              in_place (*_repair_inplace) = D-harness replaceG, the span
                replaces the doc's gist.
* *_recover — step-level oracle recover (docs/hybrid_spec.md "Oracle
            recover"): the proxy diffs every generated action against a
            full-arm reference trajectory keyed by message fingerprint; at
            the first divergence it re-sends the identical payload assembled
            full-raw (the reference KV regime) and returns the regenerated
            step.  One repair per conversation; later mismatches are logged
            as re_diverged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


REPAIR_PLACEMENTS = ("append_keep_ledger", "append_tail", "in_place")


@dataclass(frozen=True)
class Arm:
    name: str
    compress_history: bool
    ratio: int = 8
    hybrid_top_k: int = 0  # 0 = none raw; >0 = keep this many tail messages raw
    constrain_tools: bool = False  # H1: xgrammar structural-tag decoding on <tool_call>
    # Repair-arm extension (docs/hybrid_spec.md "Repair interaction"):
    # {"policy": "first" | "offset:<j>" | "chunk:<i>"} — the proxy forwards
    # it as the request-level "c2kv_repair" field and hf_server appends the
    # raw KV of the selected COMPRESSED history block at its original
    # logical offset (offset:<j> indexes docs like the D harness
    # --corr_k_policy; chunk:<i> indexes the server's extract chunks).
    repair: Optional[Dict[str, object]] = None
    # Oracle-recover extension (docs/hybrid_spec.md "Oracle recover"):
    # {"once": True} — the proxy loads the full-arm reference trajectory
    # (--reference) and repairs the first divergence per conversation by
    # regenerating the step with full-raw history KV.  Mutually exclusive
    # with repair (both replace the same step's generation).
    recover: Optional[Dict[str, object]] = None
    description: str = ""

    def validate(self) -> None:
        if self.compress_history and self.ratio < 2:
            raise ValueError(f"arm {self.name!r}: compression ratio must be >= 2")
        if not self.compress_history and (self.hybrid_top_k or self.repair or self.recover):
            raise ValueError(f"arm {self.name!r}: hybrid/repair/recover knobs need compression")
        if self.repair and self.recover:
            raise ValueError(f"arm {self.name!r}: repair and recover are mutually exclusive")
        if self.repair:
            placement = str(self.repair.get("placement") or "append_keep_ledger")
            if placement not in REPAIR_PLACEMENTS:
                raise ValueError(
                    f"arm {self.name!r}: unknown repair placement {placement!r}; "
                    f"expected one of {REPAIR_PLACEMENTS}")


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
            description="hybrid k=3: top-3 tail messages raw, earlier history gist (docs/hybrid_spec.md)",
        ),
        Arm(
            name="hybrid_k1",
            compress_history=True,
            ratio=8,
            hybrid_top_k=1,
            description="hybrid k=1: top-1 tail message raw, earlier history gist",
        ),
        Arm(
            name="hybrid_k5",
            compress_history=True,
            ratio=8,
            hybrid_top_k=5,
            description="hybrid k=5: top-5 tail messages raw, earlier history gist",
        ),
        Arm(
            name="c2kv_repair",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "append_keep_ledger"},
            description="c2kv 8x + corr@first (D-harness corr/keepG): raw KV of doc 0 computed in context, appended, gist kept, original RoPE phase",
        ),
        Arm(
            name="c2kv_repair_tail",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "append_tail"},
            description="c2kv 8x + erratum_tail@first (D-harness raw_erratum_tail): raw KV of doc 0 re-rotated to the end of history, ledger advanced",
        ),
        Arm(
            name="c2kv_repair_inplace",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "in_place"},
            description="c2kv 8x + replaceG@first: raw KV of doc 0 replaces its gist in place",
        ),
        Arm(
            name="hybrid_repair",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            repair={"policy": "first", "placement": "append_keep_ledger"},
            description="hybrid k=3 + corr@first on the compressed prefix",
        ),
        Arm(
            name="hybrid_repair_tail",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            repair={"policy": "first", "placement": "append_tail"},
            description="hybrid k=3 + erratum_tail@first on the compressed prefix",
        ),
        Arm(
            name="c2kv_recover",
            compress_history=True,
            ratio=8,
            recover={"once": True},
            description="c2kv 8x + step-level oracle recover: first action divergence vs the full-arm reference is regenerated with full-raw history KV (once per conversation)",
        ),
        Arm(
            name="hybrid_recover",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            recover={"once": True},
            description="hybrid k=3 + step-level oracle recover (once per conversation)",
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
