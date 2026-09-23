"""CPU-only checks for the explicitly selected BFCL candidate delivery."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.history_system.candidate_algorithms import (
    C1_V2_VARIANTS, C1_V2_VERSION, c1_v2_fields,
    INITIAL_VIEW_VARIANTS, INITIAL_VIEW_VERSION, initial_view_fields,
    STATIC_EXTENSION_VARIANTS, STATIC_EXTENSION_VERSION,
)
from benchmarks.arms import get_arm
from benchmarks.paper import c1, runner
from benchmarks.paper.candidate_matrix import (
    GOAL_VARIANTS, LEGACY_VARIANTS, REPAIR_VARIANTS, VERIFIED_VARIANTS,
    VARIANT_TO_ARM, parse_candidate_arms,
    with_candidate_methods,
)


def test_toolsandbox_candidate_prepare_preserves_named_method_and_scenarios(tmp_path):
    original = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    original["benchmarks"] = [row for row in original["benchmarks"]
                              if row["name"] == "toolsandbox"]
    original["methods"] = []
    config = with_candidate_methods(original, ("pending_verified",), ("toolsandbox",))
    plan, _ = runner.prepare(config, tmp_path / "paper", tmp_path / "engine")
    assert len(plan) == 1
    row = plan[0]
    assert row["cell_id"] == "toolsandbox__c2kv_pending_verified_r8"
    assert row["ratio"] == 8
    assert "benchmarks.paper.c1" in row["command"]
    assert row["command"][row["command"].index("--benchmark") + 1] == "toolsandbox"
    assert original["methods"] == []

    old_arm = c1.ARM
    try:
        c1.select_arm(row["arm"])
        config.pop("bfcl_dir")
        config["sglang_source"] = str(tmp_path / "engine")
        config["toolsandbox_python"] = "/venv-toolsandbox/bin/python"
        args = c1.delivery_args(config, "toolsandbox", tmp_path / "native",
                                ["wifi_off", "get_wifi"], c1.load_delivery())
        assert args.benchmark == "toolsandbox"
        assert args.ts_scenario == ["wifi_off", "get_wifi"]
        assert args.toolsandbox_python == "/venv-toolsandbox/bin/python"
        assert args.benchmark_dir == args.toolsandbox_dir == Path(config["toolsandbox_dir"])
        assert args.candidate_algorithm == "pending_verified"
    finally:
        c1.select_arm(old_arm)


def test_candidate_matrix_defaults_to_bfcl_base_and_explicitly_adds_acebench(tmp_path):
    original = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    assert parse_candidate_arms("") == ()
    assert with_candidate_methods(original, ()) is original
    assert not any(row["arm"] in VARIANT_TO_ARM.values() for row in runner.cells(original))

    assert parse_candidate_arms("all") == LEGACY_VARIANTS
    augmented = with_candidate_methods(original, parse_candidate_arms("all"))
    assert original["methods"] != augmented["methods"]
    assert len(runner.cells(augmented)) == len(runner.cells(original)) + len(LEGACY_VARIANTS)
    for variant in LEGACY_VARIANTS:
        arm = VARIANT_TO_ARM[variant]
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
    assert len(ace_rows) == 2 * len(LEGACY_VARIANTS)
    assert {row["benchmark"] for row in ace_rows} == {"bfcl_base", "acebench_agent"}
    assert all(row["ratio"] == 8 for row in ace_rows)
    ace_plan, _ = runner.prepare(ace, tmp_path / "ace-paper", tmp_path / "sglang")
    for row in ace_plan:
        if row["arm"] in VARIANT_TO_ARM.values() and row["benchmark"] == "acebench_agent":
            assert "benchmarks.paper.c1" in row["command"]
            assert row["command"][row["command"].index("--arm") + 1] == row["arm"]
            assert row["cell_id"] == f"acebench_agent__{row['arm']}"
    long_app = with_candidate_methods(original, ("goal_rescue", "goal_joint"),
                                      ("bfcl_long_context", "appworld"))
    for variant in ("goal_rescue", "goal_joint"):
        goal = [row for row in long_app["methods"]
                if row["arm"] == VARIANT_TO_ARM[variant]]
        assert goal and goal[0]["benchmarks"] == ["bfcl_long_context", "appworld"]
    with pytest.raises(ValueError, match="subset"):
        with_candidate_methods(original, ("static_t02",), ("unknown_benchmark",))


def test_c1_v2_requires_named_opt_in_and_dispatches_native_budget_sweeps(tmp_path):
    variant = "c1_v2_verified"
    arm = "c2kv_c1_v2_verified_r8"
    original = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    assert variant not in parse_candidate_arms("all")
    assert parse_candidate_arms(variant) == (variant,)
    assert c1_v2_fields(variant) == {
        "initial_view": {
            "policy": "s0_capacity_fallback",
            "version": "c2kv-s0-capacity-fallback-v1",
            "terminal_rescue": "c2kv-terminal-tool-arguments-gist-v1",
        },
        "recovery_backbone": "t02_complete_event",
        "completion_review": False,
        "proof_registry_version": "verified-binding-rules-v1",
    }
    with pytest.raises(ValueError, match="unknown C1 v2"):
        c1_v2_fields("static_verified_v2")

    config = with_candidate_methods(original, (variant,))
    assert get_arm(arm).native_controller == "candidate_c1_v2_verified"
    for budget in (256, 128, 64):
        config = runner.with_native_history_budget(config, arm, budget)
    plan, profile = runner.prepare(config, tmp_path / "paper", tmp_path / "sglang")
    rows = {row["cell_id"]: row for row in plan if row["arm"] == arm}
    assert set(rows) == {
        f"bfcl_base__{arm}",
        f"bfcl_base__{arm}_b256",
        f"bfcl_base__{arm}_b128",
        f"bfcl_base__{arm}_b64",
    }
    for budget in (256, 128, 64):
        row = rows[f"bfcl_base__{arm}_b{budget}"]
        argv = runner.run_command(
            config, row, tmp_path / row["cell_id"], profile, "closed_loop")
        assert argv[argv.index("--arm") + 1] == arm
        assert argv[argv.index("--history-budget-tokens") + 1] == str(budget)


@pytest.mark.parametrize("variants", [
    VERIFIED_VARIANTS, INITIAL_VIEW_VARIANTS, STATIC_EXTENSION_VARIANTS,
    C1_V2_VARIANTS,
])
def test_new_arms_require_named_opt_in_and_support_bfcl_appworld_ace(tmp_path, variants):
    original = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    assert not set(variants) & set(parse_candidate_arms("all"))
    assert len(parse_candidate_arms("all")) == 11
    selected = parse_candidate_arms(",".join(variants))
    assert selected == variants
    augmented = with_candidate_methods(
        original, selected,
        ("bfcl_base", "bfcl_long_context", "appworld", "acebench_agent"),
    )
    rows = [row for row in runner.cells(augmented)
            if row["arm"] in {VARIANT_TO_ARM[v] for v in variants}]
    assert len(rows) == 4 * len(variants)
    assert {row["benchmark"] for row in rows} == {
        "bfcl_base", "bfcl_long_context", "appworld", "acebench_agent"}
    assert all(row["ratio"] == 8 for row in rows)
    assert not any(row["arm"] in {VARIANT_TO_ARM[v] for v in variants}
                   for row in runner.cells(original))
    plan, _ = runner.prepare(augmented, tmp_path / "verified", tmp_path / "sglang")
    assert {row["cell_id"] for row in plan if row["arm"] in {
        VARIANT_TO_ARM[v] for v in variants}} == {
        f"{benchmark}__{VARIANT_TO_ARM[variant]}"
        for benchmark in ("bfcl_base", "bfcl_long_context", "appworld", "acebench_agent")
        for variant in variants}


@pytest.mark.parametrize("variant,arm", [
    (variant, arm) for variant, arm in VARIANT_TO_ARM.items()
    if variant not in REPAIR_VARIANTS
])
def test_candidate_delivery_uses_ratio8_and_bound_artifact(tmp_path, monkeypatch, variant, arm):
    original_arm = c1.ARM
    try:
        c1.select_arm(arm)
        delivery = c1.load_delivery()
        config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
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
        base = {"view_mode": "native_s0",
                "observed_entity_slot_policy":
                    "same-complete-event-reference-bridge-only-v1",
                "gp_experiments": {},
                "post_draft_recovery": {}, "d3_hybrid_recovery": {}}
        monkeypatch.setattr(delivery.current, "load_config", lambda: selected)
        monkeypatch.setattr(delivery.evidence_sets, "_base_controller", lambda: base)
        monkeypatch.setattr(delivery, "DEFAULT_RISK_ARTIFACT", artifact)
        monkeypatch.setattr(delivery, "DEFAULT_RISK_ARTIFACT_SHA256",
                            hashlib.sha256(artifact.read_bytes()).hexdigest())
        monkeypatch.setattr(delivery, "bind_risk_artifact",
                            lambda source, path: (dict(source, bound=True), {"checkpoint": str(path)}))
        controller, profile = delivery.build_profile(args)
        assert controller == {
            "view_mode": "native_s0",
            "observed_entity_slot_policy":
                "same-complete-event-reference-bridge-only-v1",
            "candidate_algorithm": {
            "variant": variant, "risk_artifact": {"artifact": "t02", "bound": True},
            "risk_threshold": 0.5,
            **({"proof_registry_version": "verified-binding-rules-v1"}
               if variant in VERIFIED_VARIANTS else {}),
            **initial_view_fields(variant),
            **(c1_v2_fields(variant) if variant in C1_V2_VARIANTS else {})}}
        assert profile["candidate_algorithm"] == variant
        assert profile["schema"] == ("c2kv-candidate-delivery-profile-v6"
                                     if variant in STATIC_EXTENSION_VARIANTS else
                                     "c2kv-candidate-delivery-profile-v7"
                                     if variant in C1_V2_VARIANTS else
                                     "c2kv-candidate-delivery-profile-v5"
                                     if variant in INITIAL_VIEW_VARIANTS else
                                     "c2kv-candidate-delivery-profile-v4"
                                     if variant in VERIFIED_VARIANTS else
                                     "c2kv-candidate-delivery-profile-v3"
                                     if variant in GOAL_VARIANTS else
                                     "c2kv-candidate-delivery-profile-v1")
        assert profile["selection_protocol"] == (STATIC_EXTENSION_VERSION
                                                  if variant in STATIC_EXTENSION_VARIANTS else
                                                  C1_V2_VERSION
                                                  if variant in C1_V2_VARIANTS else
                                                  INITIAL_VIEW_VERSION
                                                  if variant in INITIAL_VIEW_VARIANTS else
                                                  "c2kv-verified-binding-v1"
                                                  if variant in VERIFIED_VARIANTS else
                                                  "c2kv-goal-composition-v1"
                                                  if variant in GOAL_VARIANTS else
                                                  "candidate_algorithm_v1")
        candidate_fields = (c1_v2_fields(variant) if variant in C1_V2_VARIANTS
                            else initial_view_fields(variant))
        if variant in VERIFIED_VARIANTS or candidate_fields.get("proof_registry_version"):
            expected_proof = candidate_fields.get("proof_registry_version", "verified-binding-rules-v1")
            assert controller["candidate_algorithm"]["proof_registry_version"] == expected_proof
            assert profile["proof_registry_version"] == expected_proof
            without_proof = copy.deepcopy(controller)
            del without_proof["candidate_algorithm"]["proof_registry_version"]
            assert profile["controller_sha256"] != hashlib.sha256(
                json.dumps(without_proof, sort_keys=True).encode("utf-8")
            ).hexdigest()
        else:
            assert "proof_registry_version" not in controller["candidate_algorithm"]
            assert "proof_registry_version" not in profile
        if variant in INITIAL_VIEW_VARIANTS + STATIC_EXTENSION_VARIANTS + C1_V2_VARIANTS:
            for key, value in candidate_fields.items():
                assert profile[key] == value
                changed = copy.deepcopy(controller)
                del changed["candidate_algorithm"][key]
                assert profile["controller_sha256"] != hashlib.sha256(
                    json.dumps(changed, sort_keys=True).encode("utf-8")).hexdigest()
        else:
            assert "initial_view" not in profile
            assert "initial_view" not in controller["candidate_algorithm"]
        assert profile["selector_artifact_binding"]["checkpoint"] == str(checkpoint)
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
        long_args = c1.delivery_args(config, "bfcl_long_context", tmp_path / "out", [], delivery)
        assert long_args.benchmark == "bfcl"
        assert long_args.candidate_algorithm == variant
        app_args = c1.delivery_args(config, "appworld", tmp_path / "out", [], delivery)
        assert app_args.benchmark == "acon_appworld"
        app_controller, app_profile = delivery.build_profile(app_args)
        assert app_controller["candidate_algorithm"]["variant"] == variant
        assert app_profile["candidate_algorithm"] == variant
        ts_args = c1.delivery_args(config, "toolsandbox", tmp_path / "out", ["wifi_off"], delivery)
        assert ts_args.benchmark == "toolsandbox"
        assert ts_args.toolsandbox_dir == Path(config["toolsandbox_dir"])
        assert ts_args.ts_scenario == ["wifi_off"]
        ts_controller, ts_profile = delivery.build_profile(ts_args)
        assert ts_controller == controller
        assert ts_profile["candidate_algorithm"] == variant
    finally:
        c1.select_arm(original_arm)


@pytest.mark.parametrize("variant", sorted(REPAIR_VARIANTS))
def test_repair_delivery_has_no_t02_dependency(tmp_path, monkeypatch, variant):
    arm = VARIANT_TO_ARM[variant]
    original_arm = c1.ARM
    try:
        c1.select_arm(arm)
        delivery = c1.load_delivery()
        config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        config.update(checkpoint=str(checkpoint), sglang_source=str(tmp_path / "sglang"))
        args = c1.delivery_args(config, "bfcl_base", tmp_path / "out", [], delivery)
        selected = {"ratio": 8, "checkpoint_selection": {
            "config_sha256": hashlib.sha256(b"{}").hexdigest()}}
        monkeypatch.setattr(delivery.current, "load_config", lambda: selected)
        monkeypatch.setattr(delivery.evidence_sets, "_base_controller", lambda: {
            "view_mode": "native_s0", "gp_experiments": {},
            "post_draft_recovery": {}, "d3_hybrid_recovery": True})
        monkeypatch.setattr(delivery, "DEFAULT_RISK_ARTIFACT", tmp_path / "absent.json")
        monkeypatch.setattr(delivery, "bind_risk_artifact",
                            lambda *_: pytest.fail("T02 binding must not run"))
        controller, profile = delivery.build_profile(args)
        assert controller == {"view_mode": "native_s0",
                              "candidate_algorithm": {"variant": variant}}
        assert profile["schema"] == "c2kv-candidate-delivery-profile-v2"
        assert profile["selection_protocol"] == "c2kv-source-repair-v1"
        assert profile["ratio"] == 8
        assert profile["checkpoint_config_sha256"] == selected["checkpoint_selection"]["config_sha256"]
        assert not any(key.startswith("selector_") for key in profile)
        assert profile["automatic_reruns"] == 0
        assert c1.delivery_args(config, "acebench_agent", tmp_path / "out", [], delivery).benchmark == "acebench"
        bad_ratio = copy.copy(args)
        bad_ratio.ratio = 4
        with pytest.raises(ValueError, match="ratio 8"):
            delivery.build_profile(bad_ratio)
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


def test_repair_acceptance_does_not_require_t02_scores(tmp_path):
    delivery = c1.load_delivery()
    shard = tmp_path / "server"
    shard.mkdir()
    record = {
        "ratio": 8,
        "exact_recovery": {"version": "c2kv-source-repair-v1",
                           "variant": "request_contract", "status": "keep"},
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
    telemetry = delivery.summarize_task(
        "bfcl", "multi_turn_base_0", tmp_path,
        {"n_scored": 1, "n_generated": 1, "semantic_score": 1.0}, 1.0)
    required = delivery.functional_checks(
        "proposed", "t02_risk", telemetry, "request_contract")["required"]
    assert all(required.values()), required
    assert telemetry["risk_detector_scores"] == 0
    assert not all(delivery.functional_checks(
        "proposed", "t02_risk", dict(telemetry, risk_detector_scores=1),
        "request_contract")["required"].values())


@pytest.mark.parametrize("variant", GOAL_VARIANTS + VERIFIED_VARIANTS + INITIAL_VIEW_VARIANTS
                         + STATIC_EXTENSION_VARIANTS + C1_V2_VARIANTS)
def test_goal_acceptance_requires_frozen_risk_and_distinct_version(tmp_path, variant):
    delivery = c1.load_delivery()
    shard = tmp_path / "server"
    shard.mkdir()
    record = {
        "ratio": 8,
        "exact_recovery": {
            "version": (STATIC_EXTENSION_VERSION if variant in STATIC_EXTENSION_VARIANTS else
                        C1_V2_VERSION if variant in C1_V2_VARIANTS else
                        INITIAL_VIEW_VERSION if variant in INITIAL_VIEW_VARIANTS else
                        "c2kv-verified-binding-v1" if variant in VERIFIED_VARIANTS
                        else "c2kv-goal-composition-v1"), "variant": variant,
            "status": "keep",
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
    telemetry = delivery.summarize_task(
        "bfcl", "multi_turn_base_0", tmp_path,
        {"n_scored": 1, "n_generated": 1, "semantic_score": 1.0}, 1.0)
    required = delivery.functional_checks("proposed", "t02_risk", telemetry, variant)["required"]
    assert all(required.values()), required
    assert telemetry["risk_detector_scores"] == 1
    missing_risk = delivery.functional_checks(
        "proposed", "t02_risk", dict(telemetry, risk_detector_scores=0),
        variant)["required"]
    assert missing_risk["risk_scores"] is False


def test_repair_server_command_omits_shadow_feature_setup(tmp_path, monkeypatch):
    original_arm = c1.ARM
    try:
        c1.select_arm(VARIANT_TO_ARM["request_contract"])
        delivery = c1.load_delivery()
        config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
        config["checkpoint"] = str(tmp_path / "checkpoint")
        config["sglang_source"] = str(tmp_path / "sglang")
        args = c1.delivery_args(config, "bfcl_base", tmp_path / "out", [], delivery)
        controller_path = tmp_path / "controller.json"
        controller_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(delivery.current, "load_config", lambda: {
            "ratio": 8, "runtime": {"shadow_feature_config": "risk.json"}})
        seen = []
        monkeypatch.setattr(delivery.runner, "server_command",
                            lambda design, **_: seen.append(design) or ["--s0-config", "placeholder"])
        monkeypatch.setattr(delivery.runner, "worker_command", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(delivery, "_benchmark_dir", lambda _args: tmp_path)
        server, _ = delivery.commands_for_task(args, "multi_turn_base_0", controller_path)
        assert server == ["--s0-config", str(controller_path)]
        assert "shadow_feature_config" not in seen[0]["runtime"]
    finally:
        c1.select_arm(original_arm)


def test_goal_server_command_keeps_shadow_feature_setup(tmp_path, monkeypatch):
    original_arm = c1.ARM
    try:
        c1.select_arm(VARIANT_TO_ARM["goal_joint"])
        delivery = c1.load_delivery()
        config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
        config.update(checkpoint=str(tmp_path / "checkpoint"),
                      sglang_source=str(tmp_path / "sglang"))
        args = c1.delivery_args(config, "bfcl_base", tmp_path / "out", [], delivery)
        controller_path = tmp_path / "controller.json"
        controller_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(delivery.current, "load_config", lambda: {
            "ratio": 8, "runtime": {"shadow_feature_config": "risk.json"}})
        seen = []
        monkeypatch.setattr(delivery.runner, "server_command",
                            lambda design, **_: seen.append(design) or ["--s0-config", "placeholder"])
        monkeypatch.setattr(delivery.runner, "worker_command", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(delivery, "_benchmark_dir", lambda _args: tmp_path)
        delivery.commands_for_task(args, "multi_turn_base_0", controller_path)
        assert seen[0]["runtime"]["shadow_feature_config"] == "risk.json"
    finally:
        c1.select_arm(original_arm)
