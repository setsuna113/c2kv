"""K / R_max / B resolution for the generality experiment.

R_max is the byte upper bound of ONE tokens_1024 evidence packet under the real
serialized wrapper and NPU page rounding (handoff section 3). The wrapper token
count is MEASURED on the NPU with the controller runtime's own renderer and
tokenizer (measure_wrapper_tokens); nothing here is hand-tuned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import design


def page_round_tokens(tokens: int, page_size: int = design.NPU_PAGE_SIZE) -> int:
    return int(math.ceil(tokens / page_size) * page_size)


def measure_wrapper_tokens(controller_root: str, out_json: Path) -> dict:
    """Render a maximal tokens_1024 evidence packet with the real runtime.

    Runs on the NPU inside the controller runtime import path; writes the
    observed wrapper token count (rendered message overhead around a
    UNIT_TOKEN_CAP-token unit) plus tokenizer identity.
    """
    raise NotImplementedError("executed remotely; see tools/measure_rmax.py")


def resolve(k_history_bytes: int, wrapper_tokens: int) -> dict:
    unit_tokens = design.UNIT_TOKEN_CAP + wrapper_tokens
    resident_tokens = page_round_tokens(unit_tokens)
    r_max = resident_tokens * design.KV_BYTES_PER_TOKEN
    return {
        "history_allowance_bytes": k_history_bytes,
        "unit_token_cap": design.UNIT_TOKEN_CAP,
        "wrapper_tokens": wrapper_tokens,
        "packet_resident_tokens_after_page_rounding": resident_tokens,
        "kv_bytes_per_token": design.KV_BYTES_PER_TOKEN,
        "recovery_allowance_bytes": r_max,
        "common_cap_bytes": k_history_bytes + r_max,
    }


def token_allowances(resolved: dict) -> dict:
    """Token-domain caps used by all history-KV adapters (target_tokens)."""
    per = design.KV_BYTES_PER_TOKEN
    return {
        "k_tokens": resolved["history_allowance_bytes"] // per,
        "r_tokens": resolved["recovery_allowance_bytes"] // per,
        "b_tokens": resolved["common_cap_bytes"] // per,
    }


def write_resolved(out: Path, wrapper_tokens: int) -> dict:
    resolved = {
        "schema": "c2kv-generality-budgets-v1",
        "r_max_rule": design.r_max_rule(),
        "wrapper_tokens_measurement": {
            "source": "tools/measure_rmax.py on ascend03 with controller runtime renderer",
            "wrapper_tokens": wrapper_tokens,
        },
        "working_points": {
            wid: resolve(wp["history_allowance_bytes"], wrapper_tokens)
            for wid, wp in design.WORKING_POINTS.items()
        },
    }
    out.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return resolved
