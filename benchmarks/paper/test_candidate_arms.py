"""CPU-only checks for the explicitly selected BFCL candidate delivery."""

import copy
import hashlib
import json

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1, runner
from benchmarks.paper.candidate_matrix import VARIANT_TO_ARM, parse_candidate_arms, with_candidate_methods


def test_candidate_matrix_defaults_to_bfcl_base_and_explicitly_adds_acebench(tmp_path):
    original = json.loads(runner.DEFAULT_CONFIG.read_text())
    assert parse_candidate_arms("") == ()
    assert with_candidate_methods(original, ()) is original
    assert not any(row["arm"] in VARIANT_TO_ARM.values() for row in runner.cells(original))

    augmented = with_candidate_methods(original, parse_candidate_arms("all"))
    assert original["methods"] != augmented["methods"]
    assert len(runner.cells(augmented)) == len(runner.cells(original)) + 4
    for variant, arm in VARIANT_TO_ARM.items():
        rows = [row for row in runner.cells(augmented) if row["arm"] == arm]
        assert len(rows) == 1
        assert rows[0]["cell_id"] == f"bfcl_base__{arm}"
        assert (rows[0]["ratio"], rows[0].get("tool_context", "raw")) == (8, "raw")
        assert get_arm(arm).native_controller == "candidate_" + variant
    plan, _ = runner.prepare(augmented, tmp_path / "paper", tmp_path / "sglang")
    for row in plan:
        if row["arm"] in VARIANT_TO_ARM.values():
            assert "benchmarks.paper.c1" in row["command"]
            assert row["command"][row["command"].index("--arm") + 1] == row["arm"]

    ace = with_candidate_methods(original, parse_candidate_arms("all"),
                                 ("bfcl_base", "acebench_agent"))
    ace_rows = [row for row in runner.cells(ace) if row["arm"] in VARIANT_TO_ARM.values()]
    assert len(ace_rows) == 8
    assert {row["benchmark"] for row in ace_rows} == {"bfcl_base", "acebench_agent"}
    assert all(row["ratio"] == 8 for row in ace_rows)
    ace_plan, _ = runner.prepare(ace, tmp_path / "ace-paper", tmp_path / "sglang")
    for row in ace_plan:
        if row["arm"] in VARIANT_TO_ARM.values() and row["benchmark"] == "acebench_agent":
            assert "benchmarks.paper.c1" in row["command"]
            assert row["command"][row["command"].index("--arm") + 1] == row["arm"]
            assert row["cell_id"] == f"acebench_agent__{row['arm']}"
    with pytest.raises(ValueError, match="subset"):
        with_candidate_methods(original, ("static_t02",), ("bfcl_long_context",))


@pytest.mark.parametrize("variant,arm", VARIANT_TO_ARM.items())
def test_candidate_delivery_uses_ratio8_and_bound_artifact(tmp_path, monkeypatch, variant, arm):
    original_arm = c1.ARM
    try:
        c1.select_arm(arm)
        delivery = c1.load_delivery()
        config = json.loads(runner.DEFAULT_CONFIG.read_text())
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        config.update(checkpoint=str(checkpoint), sglang_source=str(tmp_path / "sglang"))
        args = c1.delivery_args(config, "bfcl_base", tmp_path / "out", [], delivery)
        assert args.candidate_algorithm == variant
        assert args.ratio == 8
        assert args.method == "proposed"

        artifact = tmp_path / "risk.json"
        artifact.write_text('{"artifact": "t02"}', encoding="utf-8")
        selected = {"ratio": 8, "checkpoint_selection": {
            "config_sha256": hashlib.sha256(b"{}").hexdigest()}}
        base = {"view_mode": "native_s0", "gp_experiments": {},
                "post_draft_recovery": {}, "d3_hybrid_recovery": {}}
        monkeypatch.setattr(delivery.current, "load_config", lambda: selected)
        monkeypatch.setattr(delivery.evidence_sets, "_base_controller", lambda: base)
        monkeypatch.setattr(delivery, "DEFAULT_RISK_ARTIFACT", artifact)
        monkeypatch.setattr(delivery, "DEFAULT_RISK_ARTIFACT_SHA256",
                            hashlib.sha256(artifact.read_bytes()).hexdigest())
        monkeypatch.setattr(delivery, "bind_risk_artifact",
                            lambda source, path: (dict(source, bound=True), {"checkpoint": str(path)}))
        controller, profile = delivery.build_profile(args)
        assert controller == {"view_mode": "native_s0", "candidate_algorithm": {
            "variant": variant, "risk_artifact": {"artifact": "t02", "bound": True},
            "risk_threshold": 0.5}}
        assert profile["candidate_algorithm"] == variant
        assert profile["ratio"] == 8
        assert profile["automatic_reruns"] == 0
        assert "candidate_algorithm" not in base

        bad_ratio = copy.copy(args)
        bad_ratio.ratio = 4
        with pytest.raises(ValueError, match="ratio 8"):
            delivery.build_profile(bad_ratio)
        bad_artifact = copy.copy(args)
        bad_artifact.selector_artifact = artifact
        with pytest.raises(ValueError, match="bundled T02"):
            delivery.build_profile(bad_artifact)
        bad_detector = copy.copy(args)
        bad_detector.detector = "d3_hybrid"
        with pytest.raises(ValueError, match="frozen T02"):
            delivery.build_profile(bad_detector)
        ace_args = c1.delivery_args(config, "acebench_agent", tmp_path / "out", [], delivery)
        assert ace_args.benchmark == "acebench"
        assert ace_args.candidate_algorithm == variant
        ace_controller, ace_profile = delivery.build_profile(ace_args)
        assert ace_controller["candidate_algorithm"]["variant"] == variant
        assert ace_profile["candidate_algorithm"] == variant
        with pytest.raises(ValueError, match="bfcl_base and acebench_agent"):
            c1.delivery_args(config, "bfcl_long_context", tmp_path / "out", [], delivery)
    finally:
        c1.select_arm(original_arm)


def test_candidate_acceptance_reads_durable_step_contract(tmp_path):
    delivery = c1.load_delivery()
    shard = tmp_path / "server"
    shard.mkdir()
    record = {
        "ratio": 8,
        "exact_recovery": {
            "version": "c2kv-paper-candidates-v1", "variant": "static_t02",
            "status": "keep", "gate": {"type": "risk", "score": 0.2, "triggered": False},
            "selection": {"selector": "risk", "score_semantics": "current_turn_failure_risk",
                          "available": True, "score": 0.2}},
        "pre_generation_budget_checks": [{"status": "passed"}],
        "generation_trace": [{
            "status": "completed", "phase": "draft",
            "controller": {"requested_ratio": 8,
                           "candidate_algorithm": {"stable_call_ids": True}},
            "generation": {"stats": {"backend": "sglang_c2kv_native_packed",
                                     "gist_tokens": 3, "workspace_tokens": 2}},
        }],
    }
    (shard / "steps.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    official = {"n_scored": 1, "n_generated": 1, "semantic_score": 1.0}
    telemetry = delivery.summarize_task("bfcl", "multi_turn_base_0", tmp_path, official, 1.0)
    required = delivery.functional_checks("proposed", "t02_risk", telemetry, "static_t02")["required"]
    assert all(required.values()), required
    assert telemetry["risk_detector_scores"] == 1
    assert telemetry["recovery_count"] == 0
    assert not all(delivery.functional_checks(
        "proposed", "t02_risk", dict(telemetry, candidate_budget_passed=False),
        "static_t02")["required"].values())
    assert not all(delivery.functional_checks(
        "proposed", "t02_risk", telemetry, "turn_c1")["required"].values())
