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
* history_kv_<method>_r<per-mille> — the upstream history-KV eviction
            baselines (StreamingLLM / H2O / SnapKV / PyramidKV) ported from
            kvoffload-sglang ``c2kv_eval.adapters.bfcl_history_kv_baselines``.
            No gist compression: the completed history is prefilled once by
            ``/v1/c2kv/repair_extract`` with ``history_kv_method`` +
            ``history_kv_retention_ratio``, the server selects the surviving
            token slots and stores them as ONE repair entry, and the chat
            request carries that entry on a repair-only carrier message in
            place of the history text.  See README "History-KV eviction arms"
            for the deviations from the upstream client.
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
from typing import Any, Dict, Optional


REPAIR_PLACEMENTS = ("append_keep_ledger", "append_tail", "in_place")

# ---- history-KV eviction baselines (upstream parity) --------------------
# Names and defaults mirror kvoffload-sglang
# ``scripts/run_history_kv_baselines.sh`` (HISTORY_KV_* env defaults) and
# ``c2kv_eval.adapters.bfcl_history_kv_baselines.parse_args``.  The server
# accepts the two aliases below and normalizes them the same way
# (scheduler._build_history_kv_eviction_rounds, qwen3.generate_raw_repair_kv).
HISTORY_KV_METHODS = ("streamingllm", "h2o", "snapkv_persistent", "pyramidkv")
HISTORY_KV_METHOD_ALIASES = {"snapkv": "snapkv_persistent", "pyramid": "pyramidkv"}
# ``--runtime-history-kv-backend`` in the upstream runner.
HISTORY_KV_BACKENDS = ("repair_extract", "physical_eviction")
HISTORY_KV_POOLINGS = ("avgpool", "maxpool")
HISTORY_KV_DEFAULTS: Dict[str, Any] = {
    "backend": "repair_extract",
    "retention_ratio": None,      # --history-kv-retention-ratio (0.312)
    "target_tokens": None,        # absolute budget; overrides the ratio
    "recent_window": 64,          # --history-kv-recent-window
    "kernel_size": 5,             # --history-kv-kernel-size
    "pooling": "avgpool",         # --history-kv-pooling
    "h2o_recent_fraction": 0.5,   # --history-kv-h2o-recent-fraction
    "persistent_session": False,  # --persistent-history-kv-session
}


def history_kv_spec(arm: "Arm") -> Optional[Dict[str, Any]]:
    """Canonical, fully defaulted history-KV spec of ``arm`` (or None).

    Raises ValueError on an unusable spec, so an arm can never reach the wire
    with a method or a budget the server would reject.
    """
    config = getattr(arm, "history_kv", None)
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"arm {arm.name!r}: history_kv must be a dict")
    unknown = set(config) - set(HISTORY_KV_DEFAULTS) - {"method"}
    if unknown:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv keys {sorted(unknown)}")
    spec: Dict[str, Any] = dict(HISTORY_KV_DEFAULTS)
    spec.update(config)
    method = str(spec.get("method") or "").strip().lower()
    method = HISTORY_KV_METHOD_ALIASES.get(method, method)
    if method not in HISTORY_KV_METHODS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv method {spec.get('method')!r}; "
            f"expected one of {HISTORY_KV_METHODS} (aliases: "
            f"{sorted(HISTORY_KV_METHOD_ALIASES)})")
    spec["method"] = method
    backend = str(spec["backend"])
    if backend not in HISTORY_KV_BACKENDS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv backend {backend!r}; "
            f"expected one of {HISTORY_KV_BACKENDS}")
    if str(spec["pooling"]) not in HISTORY_KV_POOLINGS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv pooling {spec['pooling']!r}")
    ratio = spec["retention_ratio"]
    target = spec["target_tokens"]
    if ratio is None and target is None:
        raise ValueError(
            f"arm {arm.name!r}: history_kv needs retention_ratio or target_tokens")
    if ratio is not None and not 0.0 < float(ratio) <= 1.0:
        raise ValueError(
            f"arm {arm.name!r}: history_kv retention_ratio must be in (0, 1]")
    if target is not None and int(target) < 1:
        raise ValueError(f"arm {arm.name!r}: history_kv target_tokens must be >= 1")
    if int(spec["recent_window"]) < 1 or int(spec["kernel_size"]) < 1:
        raise ValueError(
            f"arm {arm.name!r}: history_kv recent_window/kernel_size must be >= 1")
    if not 0.0 <= float(spec["h2o_recent_fraction"]) <= 1.0:
        raise ValueError(
            f"arm {arm.name!r}: history_kv h2o_recent_fraction must be in [0, 1]")
    if backend == "physical_eviction" and target is None:
        # The physical path takes an ABSOLUTE budget: the scheduler reads
        # config["target_tokens"] only (scheduler.py
        # _select_history_kv_eviction_indices, mem_cache/history_kv_eviction.py
        # PhysicalHistoryKVEvictor.evict) and never derives it from a ratio,
        # and this proxy has no tokenizer with which to convert one.  See
        # README "History-KV eviction arms".
        raise ValueError(
            f"arm {arm.name!r}: history_kv backend 'physical_eviction' needs an "
            "absolute target_tokens (the server has no retention_ratio on that "
            "path and the proxy has no tokenizer)")
    if spec["persistent_session"] and backend != "physical_eviction":
        raise ValueError(
            f"arm {arm.name!r}: history_kv persistent_session requires backend "
            "'physical_eviction' (upstream run_history_kv_baselines.sh)")
    return spec


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
    # Text-level baseline arms (benchmarks/textarms.py): the proxy rewrites
    # the request history per the named paper's policy instead of any KV
    # compression; compressor calls go to the same served endpoint (same
    # model as policy — the papers' own protocol).  "hiagent" |
    # "acon_hist" | "acon_obs" (ACON evaluates history and observation
    # compression separately, audit ruling 6).
    text_policy: Optional[str] = None
    # History-KV eviction baselines (upstream
    # c2kv_eval.adapters.bfcl_history_kv_baselines): no gist compression at
    # all -- the completed history is compressed by TOKEN EVICTION inside the
    # server.  {"method": streamingllm|h2o|snapkv_persistent|pyramidkv,
    #  "retention_ratio": float | "target_tokens": int,
    #  "backend": "repair_extract" (default) | "physical_eviction",
    #  "recent_window", "kernel_size", "pooling", "h2o_recent_fraction",
    #  "persistent_session"}.  history_kv_spec() fills the defaults and
    # rejects anything the server would refuse.
    history_kv: Optional[Dict[str, object]] = None
    description: str = ""

    def validate(self) -> None:
        if self.history_kv:
            if self.compress_history or self.text_policy or self.repair or self.recover:
                raise ValueError(
                    f"arm {self.name!r}: history_kv is exclusive with gist "
                    "compression, text_policy, repair and recover")
            history_kv_spec(self)
        if self.compress_history and self.ratio < 2:
            raise ValueError(f"arm {self.name!r}: compression ratio must be >= 2")
        if not self.compress_history and (self.hybrid_top_k or self.repair or self.recover):
            raise ValueError(f"arm {self.name!r}: hybrid/repair/recover knobs need compression")
        if self.repair and self.recover:
            raise ValueError(f"arm {self.name!r}: repair and recover are mutually exclusive")
        if self.text_policy and self.compress_history:
            raise ValueError(f"arm {self.name!r}: text_policy and KV compression are exclusive")
        if self.text_policy and self.text_policy not in ("hiagent", "acon_hist", "acon_obs"):
            raise ValueError(f"arm {self.name!r}: unknown text_policy {self.text_policy!r}")
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
            name="hiagent",
            compress_history=False,
            text_policy="hiagent",
            description="HiAgent (2408.09559): subgoal-protocol note + completed subgoal segments "
                        "replaced by paper-§3.3 summaries (same model as compressor, decode per paper); "
                        "user turns survive; Trajectory Retrieval not ported; degenerate passthrough is "
                        "flagged in stats",
        ),
        Arm(
            name="acon_hist",
            compress_history=False,
            text_policy="acon_hist",
            description="acon-base (2510.00615), history face: prefix >4096 tok replaced by the rolling "
                        "structured summary embedded in the first user prompt's <HISTORY_SUMMARY> block, "
                        "last action/observation pair preserved. Label: 'ACON pipeline, base guideline, "
                        "guideline optimization not reproduced'",
        ),
        Arm(
            name="acon_obs",
            compress_history=False,
            text_policy="acon_obs",
            description="acon-base (2510.00615), observation face: tool observations >1024 tok refined "
                        "in place. Same base-guideline label as acon_hist (paper evaluates the two "
                        "faces separately)",
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
            description=(
                "c2kv 8x + replaceG@first: raw KV of doc 0 replaces its gist in place. "
                "REGIME BREAK vs the Sep-2 serve-align runs: policy 'first' on a "
                "single-doc turn leaves the request with zero gist segments, and the "
                "reconciled server (D7, c2kv_serving_semantics.md #1) serves such a "
                "request with GIST query projections where serve-align served base. "
                "Read c2kv_query_proj_effective + c2kv_gist_seen per row; the "
                "layout frame check is also unavailable on this arm "
                "(repair_frame.ok_reason = not_measured_in_place_first_doc). "
                "frame_delta IS measured on BFCL FC: the prologue is measured "
                "against the assembled list, which always carries proxy."
                "DEFAULT_SYSTEM_PROMPT when the client sends no system message."
            ),
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
        # ---- history-KV eviction baselines --------------------------
        # retention 0.312 = HISTORY_KV_RETENTION_RATIO in the upstream
        # run_history_kv_baselines.sh; the name suffix is that ratio in
        # per-mille (r312 = 0.312).  NOT RUN against a live server yet.
        Arm(
            name="history_kv_streamingllm_r312",
            compress_history=False,
            history_kv={"method": "streamingllm", "retention_ratio": 0.312},
            description="StreamingLLM history-KV eviction (recent-suffix keep) at "
                        "retention 0.312; server-side selection via "
                        "/v1/c2kv/repair_extract history_kv_method=streamingllm",
        ),
        Arm(
            name="history_kv_h2o_r312",
            compress_history=False,
            history_kv={"method": "h2o", "retention_ratio": 0.312},
            description="H2O heavy-hitter + recent history-KV eviction at retention "
                        "0.312 (h2o_recent_fraction 0.5, recent_window 64); the "
                        "attention scores come from the server's extraction prefill",
        ),
        Arm(
            name="history_kv_snapkv_r312",
            compress_history=False,
            history_kv={"method": "snapkv_persistent", "retention_ratio": 0.312},
            description="SnapKV-persistent history-KV eviction at retention 0.312 "
                        "(observation window 64, avgpool kernel 5); the compressed "
                        "entry is reused, not reselected per turn",
        ),
        Arm(
            name="history_kv_pyramidkv_r312",
            compress_history=False,
            history_kv={"method": "pyramidkv", "retention_ratio": 0.312},
            description="PyramidKV history-KV eviction at retention 0.312; the "
                        "per-layer funnel budget is realised as a layer-union keep "
                        "set on one shared page table (server-side globalisation)",
        ),
        # KNOWN CONFOUND on both cd_* arms: constrain_tools=True makes the
        # sglang backend rewrite request.tools with _inline_refs ($refs
        # inlined, loose types mapped, unsupported keywords stripped) so
        # xgrammar can compile them.  SGLang derives the grammar FROM
        # request.tools and the chat template renders the SAME field into the
        # prompt — there is no separate grammar channel — so a cd_* row
        # differs from its unconstrained twin by prompt AND grammar.  H1
        # therefore measures "structural-tag decoding + normalised tool
        # schemas", not decoding alone.  Not fixable proxy-side.
        Arm(
            name="cd_full",
            compress_history=False,
            constrain_tools=True,
            description=(
                "H1: full raw history + xgrammar structural-tag tool-call decoding. "
                "CONFOUND vs 'full': the tool schemas are also rewritten "
                "(_inline_refs) and SGLang renders that rewritten `tools` field "
                "into the prompt, so the prologue differs too."
            ),
        ),
        Arm(
            name="cd_c2kv",
            compress_history=True,
            ratio=8,
            constrain_tools=True,
            description=(
                "H1: 8x gist history + xgrammar structural-tag tool-call decoding. "
                "CONFOUND vs 'c2kv': same prologue rewrite as cd_full — the "
                "grammar and the rendered tool definitions come from one field."
            ),
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
