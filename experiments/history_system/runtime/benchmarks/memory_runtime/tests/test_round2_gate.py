"""Round-two recovery-loop, detector composition, and calibration checks."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.recovery.calibration import calibrate_detector_thresholds
from benchmarks.memory_runtime.recovery.experiment_config import configure_controller
from benchmarks.memory_runtime.tests.test_event_native_recovery import (
    draft_call,
    margin_calibration,
    margin_trace,
    recovery_config,
    request,
)
from benchmarks.memory_runtime.tests.test_event_native_step import Generator
from benchmarks.memory_runtime.tests.test_gp_recovery import UnitTokenizer
from benchmarks.memory_runtime.tests.test_same_event_bridge_only import packing, policy, s0_config


def make_controller(**switches):
    config = s0_config()
    config["post_draft_recovery"] = recovery_config(
        "first_name_margin", margin_calibration=margin_calibration()
    )
    return build_event_native_controller(
        UnitTokenizer(),
        packing=packing(),
        policy=policy(),
        view_mode="ac_native_s0_lexical_raw_reserve_failed_operation",
        compression_policy="always-compress-v1",
        s0_config=configure_controller(
            config, {"D": "candidate_rule", **switches}
        ),
    )


def test_r4_requeries_each_new_draft_and_deduplicates_current_decision(
    tmp_path, monkeypatch
):
    controller = make_controller(U="field", R=4)
    payload = request()
    payload["messages"][-1:-1] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "d",
                "type": "function",
                "function": {"name": "ping", "arguments": '{"id":"other-d"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "d", "content": '{"ok":4}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "e",
                "type": "function",
                "function": {"name": "ping", "arguments": '{"id":"other-e"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "e", "content": '{"ok":5}'},
    ]
    from benchmarks.memory_runtime.recovery import selection

    observed_drafts = []
    real_select = selection.select_candidates

    def observe_select(*args, **kwargs):
        observed_drafts.append(kwargs["draft_text"])
        return real_select(*args, **kwargs)

    monkeypatch.setattr(selection, "select_candidates", observe_select)
    journal = tmp_path / "attempts.jsonl"
    drafts = [
        "lookup violet item-17",
        "ping other-a ok 1",
        "ping other-b ok 2",
        "ping other-c ok 3",
        "Final response.",
    ]
    runner = EventNativeDecisionRunner(
        controller,
        Generator(journal, drafts),
        UnitTokenizer(),
        ratio=4,
        max_new_tokens=32,
        max_generation_calls=96,
        journal=AttemptJournal(journal),
    )
    record = runner.run(payload)
    assert record["generation_attempts"] == 5
    assert len(record["recovery_rounds"]) == 4
    selected = [
        unit_id
        for round_receipt in record["recovery_rounds"]
        for unit_id in round_receipt["appended_unit_ids"]
    ]
    assert len(selected) == len(set(selected))
    assert [
        row["recovery_round"] for row in record["recovery_rounds"]
    ] == [1, 2, 3, 4]
    assert observed_drafts == drafts[:4]


def test_new_decision_can_recover_a_source_released_by_next_decision_policy():
    controller = make_controller(L="next_decision")
    first = controller.prepare(request("d1"), ratio=4, max_new_tokens=32)
    first_result = controller.reconsider(
        first, [draft_call()], draft_text="lookup violet item-17"
    )
    assert first_result["regenerate"]

    second = controller.prepare(request("d2"), ratio=4, max_new_tokens=32)
    assert second.metadata["gp_lifecycle"]["protected_unit_ids"] == []
    second_result = controller.reconsider(
        second, [draft_call()], draft_text="lookup violet item-17"
    )
    assert second_result["regenerate"]
    assert second_result["decision"]["selected_unit_ids"] == first_result["decision"][
        "selected_unit_ids"
    ]


def test_detector_threshold_overrides_only_the_runtime_threshold():
    controller = make_controller(
        D="detector",
        detector_threshold=0.95,
        detector_calibration_telemetry=True,
    )
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9)
    )
    result = controller.reconsider(
        prepared, [draft_call()], draft_text="lookup violet item-17"
    )
    assert result["regenerate"]
    gate = result["decision"]["gate"]
    assert gate["threshold"] == 0.95
    assert gate["base_detector_threshold"] == 0.5
    assert gate["threshold_source"] == "gp_experiments.detector_threshold"
    assert controller.config["margin_calibration"]["threshold"] == 0.5
    telemetry = result["decision"]["calibration_telemetry"]
    assert telemetry["detector_score"] == 0.9
    assert telemetry["candidate_feasible"] is True
    assert telemetry["model_calls"] == 0


def test_detector_telemetry_distinguishes_available_from_b0_feasible(monkeypatch):
    controller = make_controller(D="detector", detector_calibration_telemetry=True)
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9)
    )
    monkeypatch.setattr(controller._packer, "_try_measure", lambda *args, **kwargs: None)
    result = controller.reconsider(
        prepared, [draft_call()], draft_text="lookup violet item-17"
    )
    assert not result["regenerate"]
    assert result["decision"]["reason"] == "margin_above_threshold"
    assert result["decision"]["candidate_availability"]["candidate_available"] is True
    assert result["decision"]["candidate_feasibility"]["candidate_feasible"] is False
    telemetry = result["decision"]["calibration_telemetry"]
    assert telemetry["candidate_available"] is True
    assert telemetry["candidate_feasible"] is False
    assert telemetry["feasibility_stage"] == (
        "single_candidate_expansion_after_real_b0_admission"
    )


def test_default_detector_abstain_keeps_legacy_early_return(monkeypatch):
    controller = make_controller(D="detector")
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9)
    )

    def unexpected_catalog(*args, **kwargs):
        raise AssertionError("legacy detector abstain must not build a catalog")

    monkeypatch.setattr(
        "benchmarks.memory_runtime.recovery.evidence_units.build_catalog",
        unexpected_catalog,
    )
    result = controller.reconsider(
        prepared, [draft_call()], draft_text="lookup violet item-17"
    )
    assert not result["regenerate"]
    assert result["decision"]["reason"] == "margin_above_threshold"
    assert "candidate_feasibility" not in result["decision"]
    assert "calibration_telemetry" not in result["decision"]


def test_candidate_rule_can_opt_in_to_detector_calibration_without_changing_gate():
    controller = make_controller(detector_calibration_telemetry=True)
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9)
    )
    result = controller.reconsider(
        prepared, [draft_call()], draft_text="lookup violet item-17"
    )
    assert result["regenerate"]
    assert result["decision"]["gate"] == {
        "type": "candidate_rule",
        "triggered": True,
        "reason": "available_source_units",
    }
    assert result["decision"]["calibration_telemetry"]["detector_score"] == 0.9


def test_candidate_or_detector_records_both_conditions_and_action_equivalence():
    controller = make_controller(D="candidate_or_detector")
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.observe_draft_features(
        session_id="task-1", decision_key="d1", shadow_features=margin_trace(0.9)
    )
    result = controller.reconsider(
        prepared, [draft_call()], draft_text="lookup violet item-17"
    )
    assert result["regenerate"]
    gate = result["decision"]["gate"]
    assert gate["conditions"]["candidate_rule"]["triggered"] is True
    assert gate["conditions"]["detector"]["triggered"] is False
    assert gate["raw_or_triggered"] is True
    assert gate["append_action_equivalent_to_candidate_rule"] is True
    assert gate["detector_can_change_append_decision"] is False


def _observation(score, feasible=True, *, split="development", direction="at_or_above"):
    return {
        "split": split,
        "decision": {
            "calibration_telemetry": {
                "schema": "a-history-detector-feasibility-observation-v1",
                "detector_type": "prefill_linear_head",
                "detector_score": score,
                "detector_direction": direction,
                "detector_feature": "prefill.prompt_last.decoder_layer_output",
                "candidate_feasible": feasible,
                "feasibility_stage": (
                    "single_candidate_expansion_after_real_b0_admission"
                ),
            }
        },
    }


def test_offline_calibration_uses_feasible_dev_scores_and_quantifies_ties(tmp_path):
    rows = [
        _observation(0.9),
        _observation(0.8),
        _observation(0.8),
        _observation(0.2),
        _observation(0.1),
        _observation(1.0, feasible=False),
    ]
    source = tmp_path / "development-risk.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source.write_bytes(payload)

    artifact = calibrate_detector_thresholds(source)
    assert artifact["source"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert artifact["observations"] == {
        "all_development_states": 6,
        "feasible_candidate_states": 5,
        "infeasible_candidate_states_excluded": 1,
    }
    forty, sixty = artifact["calibrations"]
    assert forty["target_exposure"] == 0.4
    assert forty["target_count"] == 2
    assert forty["threshold"] == 0.8
    assert forty["actual_trigger_count_including_ties"] == 3
    assert forty["actual_exposure_including_ties"] == pytest.approx(0.6)
    assert forty["boundary"]["score_tie_count"] == 2
    assert sixty["target_exposure"] == 0.6
    assert sixty["threshold"] == pytest.approx(0.5)
    assert artifact["execution"] == {
        "model_calls": 0,
        "supervised_training_runs": 0,
        "uses_outcome_labels": False,
        "offline_only": True,
    }


@pytest.mark.parametrize(
    "rows,match",
    [
        ([_observation(0.4, split="test")], "non-development split"),
        ([_observation(None)], "real detector score"),
        (
            [
                {
                    "split": "development",
                    "decision": {
                        "calibration_telemetry": {
                            **_observation(0.4)["decision"]["calibration_telemetry"],
                            "candidate_feasible": None,
                        }
                    },
                }
            ],
            "candidate feasibility marker",
        ),
        (
            [
                {
                    "split": "development",
                    "decision": {
                        "calibration_telemetry": {
                            **_observation(0.4)["decision"]["calibration_telemetry"],
                            "feasibility_stage": "pre_admission_guess",
                        }
                    },
                }
            ],
            "explicit B0 admission feasibility",
        ),
    ],
)
def test_offline_calibration_refuses_test_or_invented_inputs(tmp_path, rows, match):
    source = tmp_path / "risk.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        calibrate_detector_thresholds(source)


@pytest.mark.parametrize("field,value,match", [
    ("detector_type", "first_name_margin", "mixes detector types"),
    ("detector_feature", "other_feature", "mixes detector features"),
])
def test_offline_calibration_refuses_mixed_detector_distributions(
    tmp_path, field, value, match
):
    first = _observation(0.8)
    second = _observation(0.2)
    second["decision"]["calibration_telemetry"][field] = value
    source = tmp_path / "mixed.json"
    source.write_text(json.dumps([first, second]), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        calibrate_detector_thresholds(source)
