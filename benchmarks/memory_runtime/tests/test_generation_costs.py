"""Synthetic cost regressions for a discarded draft and a failed regeneration."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import reqlog
from memory_runtime.generation_costs import (
    request_generation_cost,
    request_generation_resources,
    summed_generation_resources,
)
from memory_runtime.collect_official import Collector


def _row():
    first = {"prompt_tokens": 100, "completion_tokens": 7}
    final = {"prompt_tokens": 160, "completion_tokens": 11}
    return {
        "status": "ok", "eval_context": {"task_id": "synthetic"},
        "wall_sec": 2.0, "gist_tokens": 0, "original_tokens": 0,
        "usage": final, "generation_attempts": 2, "generation_completed": 2,
        "generation_usage_total": {"prompt_tokens": 260, "completion_tokens": 18},
        "generation_trace": [
            {"phase": "draft", "status": "completed", "backend_verified": True,
             "discarded": True, "usage": first},
            {"phase": "regeneration", "status": "completed", "backend_verified": True,
             "discarded": False, "usage": final},
        ],
    }


def _resource_row():
    row = _row()
    row.update({
        "gist_tokens": 40,
        "original_tokens": 700,
        "kv_resident_tokens": 80,
        "total_gpu_kv_bytes": 800,
        "memory_runtime": {
            "controller_wall_sec": 0.3,
            "exact_recovery": {"controller_wall_sec": 0.4},
        },
        "extraction_telemetry": {
            "summary": {
                "lookups": 3,
                "client_cache_hits": 1,
                "producer_calls": 2,
                "producer_successes": 2,
                "producer_failures": 0,
                "lookup_wall_sec": 0.7,
                "producer_wall_sec": 0.5,
            },
        },
    })
    row["generation_trace"][0].update({
        "cost": {
            "kv_resident_tokens": 100,
            "kv_peak_resident_tokens": 110,
            "total_gpu_kv_bytes": 1000,
            "peak_total_gpu_kv_bytes": 1100,
        },
        "memory_runtime": {
            "active_history_bytes": 300,
            "evidence_bytes": 50,
            "gist_tokens": 30,
            "controller_wall_sec": 0.1,
            "compressed_assembly_wall_sec": 0.2,
        },
    })
    row["generation_trace"][1].update({
        "cost": {
            "kv_resident_tokens": 80,
            "kv_peak_resident_tokens": 90,
            "total_gpu_kv_bytes": 800,
            "peak_total_gpu_kv_bytes": 900,
        },
        "memory_runtime": {
            "active_history_bytes": 250,
            "evidence_bytes": 70,
            "gist_tokens": 40,
            "controller_wall_sec": 0.3,
            # This is a repeated snapshot and must not be summed.
            "compressed_assembly_wall_sec": 0.2,
        },
    })
    return row


def test_summaries_charge_both_generations_and_keep_final_usage_separate():
    row = _row()
    summary = reqlog.summarize([row])
    costs = summary["task_costs"]["synthetic"]
    assert costs["generation_attempts"] == 2
    assert costs["prompt_tokens"] == 260 and costs["completion_tokens"] == 18
    assert costs["known_prompt_tokens"] == 260
    metrics = Collector._request_metrics([row], "full")
    assert metrics["prompt_tokens"] == 260 and metrics["completion_tokens"] == 18
    assert row["usage"]["prompt_tokens"] == 160


def test_failed_second_call_preserves_known_first_cost_without_zero_imputation():
    row = _row()
    row.update(status="upstream_error", usage=None, generation_completed=1,
               generation_usage_total={"prompt_tokens": None, "completion_tokens": None})
    row["generation_trace"][1].update(status="failed", backend_verified=False, usage=None)
    costs = reqlog.summarize([row])["task_costs"]["synthetic"]
    assert costs["n_errors"] == 1 and costs["generation_attempts"] == 2
    assert costs["prompt_tokens"] is None and costs["completion_tokens"] is None
    assert costs["known_prompt_tokens"] == 100 and costs["known_completion_tokens"] == 7


@pytest.mark.parametrize("corrupt", ["total", "attempts", "discarded", "verified", "final"])
def test_corrupt_trace_cannot_be_reported_as_complete_cost(corrupt):
    row = _row()
    if corrupt == "total":
        row["generation_usage_total"]["prompt_tokens"] = 160
    elif corrupt == "attempts":
        row["generation_attempts"] = 1
    elif corrupt == "discarded":
        row["generation_trace"][0]["discarded"] = False
    elif corrupt == "verified":
        row["generation_trace"][0]["backend_verified"] = False
    else:
        row["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
    with pytest.raises(ValueError):
        request_generation_cost(row)


def test_legacy_summary_keeps_existing_single_response_shape():
    row = {"status": "ok", "eval_context": {"task_id": "legacy"},
           "usage": {"prompt_tokens": 40, "completion_tokens": 4}}
    costs = reqlog.summarize([row])["task_costs"]["legacy"]
    assert costs["prompt_tokens"] == 40 and costs["completion_tokens"] == 4
    assert "generation_attempts" not in costs
    assert request_generation_cost(copy.deepcopy(row))["scope"] == "legacy_final_response"
    summary = reqlog.summarize([row])
    assert "generation_resources" not in summary
    assert "original_tokens_scope" not in summary


def test_request_resources_span_attempts_without_double_counting_controller():
    resources = request_generation_resources(_resource_row())

    assert resources["resource_scope"] == "all_generation_attempts"
    assert resources["kv_resident_tokens_max"] == 100
    assert resources["kv_peak_resident_tokens_max"] == 110
    assert resources["total_gpu_kv_bytes_max"] == 1000
    assert resources["peak_total_gpu_kv_bytes_max"] == 1100
    assert resources["active_history_bytes_max"] == 300
    assert resources["evidence_bytes_max"] == 70
    assert resources["gist_tokens_sum"] == 70
    assert resources["gist_tokens_max"] == 40
    assert resources["controller_wall_seconds"] == pytest.approx(0.5)
    assert resources["compressed_assembly_wall_seconds"] == pytest.approx(0.2)
    assert resources["extraction_lookups"] == 3
    assert resources["extraction_client_cache_hits"] == 1
    assert resources["extraction_producer_calls"] == 2
    assert resources["extraction_producer_successes"] == 2
    assert resources["extraction_producer_failures"] == 0
    assert resources["extraction_lookup_wall_seconds"] == pytest.approx(0.7)
    assert resources["extraction_producer_wall_seconds"] == pytest.approx(0.5)
    assert "extraction_wall_seconds" not in resources
    assert "not request-exclusive" in resources["server_allocator_scope"]
    assert "must not be added" in resources["extraction_scope"]


def test_resource_missing_values_are_null_with_observed_lower_bounds():
    row = _resource_row()
    row["generation_trace"][1]["cost"].pop("kv_resident_tokens")
    for record in row["generation_trace"]:
        record["cost"].pop("total_gpu_kv_bytes")
    row["generation_trace"][0]["memory_runtime"]["gist_tokens"] = None
    row["memory_runtime"]["exact_recovery"].pop("controller_wall_sec")
    row["extraction_telemetry"]["summary"].pop("producer_wall_sec")

    resources = request_generation_resources(row)
    assert resources["kv_resident_tokens_max"] is None
    assert resources["kv_resident_tokens_max_known_lower_bound"] == 100
    assert resources["total_gpu_kv_bytes_max"] is None
    assert resources["total_gpu_kv_bytes_max_known_lower_bound"] is None
    assert resources["gist_tokens_sum"] is None
    assert resources["gist_tokens_sum_known_lower_bound"] == 40
    assert resources["controller_wall_seconds"] is None
    assert resources["controller_wall_seconds_known_lower_bound"] == pytest.approx(0.1)
    assert resources["extraction_producer_wall_seconds"] is None
    assert resources["extraction_producer_wall_seconds_known_lower_bound"] is None


def test_reqlog_uses_attempt_resources_and_marks_original_scope():
    row = _resource_row()
    summary = reqlog.summarize([row])

    assert summary["gist_tokens_total"] == 70
    assert summary["gist_tokens_peak"] == 40
    assert summary["kv_resident_p50"] == 100
    assert summary["server_total_gpu_kv_bytes_max"] == 1000
    assert summary["original_tokens_total"] == 700
    assert summary["logical_over_gist"] is None
    assert "final successful response views" in summary["original_tokens_scope"]
    assert summary["generation_resources"]["controller_wall_seconds"] == pytest.approx(0.5)
    task = summary["task_costs"]["synthetic"]
    assert task["generation_resources"]["gist_tokens_sum"] == 70


def test_summed_resources_use_sum_for_cost_and_max_for_peaks():
    first = _resource_row()
    second = _resource_row()
    second["generation_trace"][0]["cost"]["kv_resident_tokens"] = 120
    second["generation_trace"][0]["memory_runtime"]["gist_tokens"] = 20
    resources = summed_generation_resources([first, second])

    assert resources["kv_resident_tokens_max"] == 120
    assert resources["gist_tokens_max"] == 40
    assert resources["gist_tokens_sum"] == 130
    assert resources["controller_wall_seconds"] == pytest.approx(1.0)
    assert resources["extraction_lookup_wall_seconds"] == pytest.approx(1.4)
    assert resources["extraction_producer_wall_seconds"] == pytest.approx(1.0)
