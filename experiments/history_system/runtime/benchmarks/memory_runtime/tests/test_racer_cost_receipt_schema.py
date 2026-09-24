"""RACER v2 generations must bind in the saved cost summary.

box8 results-v2-eqprobe bfcl_base__racer_v2_commitkv_bare_b256, multi_turn_base_0:
final.json failed with "generation_trace[0].generation.stats.racer_backend is invalid"
because the cost binding accepted only racer-backend-v1 receipts. The receipt, usage
and accounting below are copied from that task's turn-0/step-0 step record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_costs import _racer_served_usage

V2_RECEIPT = {
    "allocation": "backend_native_persistent", "backend": "commitkv",
    "backend_config": {"backend": "reference_attention", "method": "commitkv",
                       "persistent_session": True, "target_tokens": 256},
    "detector_calibration": "not_used", "history_budget_tokens": 256,
    "identity": "racer:v2:commitkv:bare:off:b256", "mode": "bare", "policy": "off",
    "quality_validated": False, "schema": "racer-backend-v2",
}
USAGE = {"completion_tokens": 20, "prompt_tokens": 4862, "total_tokens": 4882}
ACCOUNTING = {
    "active_history_tokens": 0, "canonical_delta_prefill_tokens": 4862,
    "checkpoint_tensor_bytes": 0, "full_history_reprefill_performed": False,
    "generated_tokens": 20, "history_and_evidence_tokens": 0, "native_evidence_tokens": 0,
    "normal_prompt_tokens": 4862, "resident_prompt_tokens": 4862,
    "source": "scheduler_resident_state",
}


def _trace(receipt):
    return {"usage": dict(USAGE), "planned_resident_prompt_tokens": 4862,
            "generation": {"token_ids": list(range(20)), "stats": {
                "racer_backend": receipt, "racer_served_usage": dict(USAGE),
                "racer_accounting": dict(ACCOUNTING)}}}


@pytest.mark.parametrize("schema", ["racer-backend-v1", "racer-backend-v2",
                                    "racer-backend-v3", "racer-backend-v4"])
def test_served_usage_binds_supported_receipts(schema):
    assert _racer_served_usage(_trace(dict(V2_RECEIPT, schema=schema)), label="trace") == USAGE


@pytest.mark.parametrize("receipt", [dict(V2_RECEIPT, schema="racer-backend-v999"),
                                     None, "racer-backend-v2"])
def test_served_usage_still_rejects_unknown_receipts(receipt):
    with pytest.raises(ValueError, match="racer_backend is invalid"):
        _racer_served_usage(_trace(receipt), label="trace")
