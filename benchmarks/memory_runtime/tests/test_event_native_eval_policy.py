"""Pure-CPU contracts for explicit A event-native evaluation policies."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from benchmarks.memory_runtime.event_native_eval_policy import (
    EVAL_POLICY_FIELDS,
    EVAL_POLICY_SCHEMA,
    OVERRIDE_VIEW_MODES,
    RECOVERY_VIEW_MODES,
    RUNTIME_POLICY_SCHEMA,
    load_eval_policy,
    resolve_event_native_eval_policy,
)


def training_policy(**updates):
    policy = {
        "mode": "persistent",
        "history_budget_bytes": 2_147_483_648,
        "workspace_budget_bytes": 536_870_912,
        "lease_decisions": 3,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 147_456,
        "source_commit": "affe0e3bd29cce06beadd5a67b1e629f8ca77022",
        "history_budget_definition": (
            "gist plus charged raw after subtracting the fixed current-input baseline"
        ),
        "workspace_budget_definition": "incremental native evidence packet",
        "current_input_baseline": (
            "source system messages plus latest user source message plus last visible "
            "source message, deduplicated; same tools and generation prompt"
        ),
    }
    policy.update(updates)
    return policy


def profile(policy=None):
    return {
        "policy_contract": policy if policy is not None else training_policy(),
        "packing_contract": {"ratios": [4, 8], "recent_tool_events": 1},
        "model_geometry": {
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "profile_marker": {"unchanged": [1, 2, 3]},
    }


def eval_policy(**updates):
    policy = {
        "schema": EVAL_POLICY_SCHEMA,
        "policy_id": "a-shared-exact-dev8",
        "policy": {
            "history_budget_bytes": 113_246_208,
            "workspace_budget_bytes": 113_246_208,
            "lease_decisions": 3,
            "max_retrieved_events": 1,
        },
    }
    policy.update(updates)
    return policy


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_load_eval_policy_accepts_only_the_complete_external_schema(tmp_path):
    value = eval_policy()
    path = tmp_path / "eval-policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = load_eval_policy(path)

    assert loaded == value
    assert loaded is not value
    assert loaded["policy"] is not value["policy"]


@pytest.mark.parametrize(
    "value, error",
    [
        ([], TypeError),
        ({"schema": EVAL_POLICY_SCHEMA, "policy_id": "p"}, ValueError),
        ({**eval_policy(), "extra": 1}, ValueError),
        (eval_policy(schema="wrong"), ValueError),
        (eval_policy(policy_id=""), ValueError),
        (eval_policy(policy_id="   "), ValueError),
        (eval_policy(policy=[]), TypeError),
        (
            eval_policy(
                policy={
                    "history_budget_bytes": 1,
                    "workspace_budget_bytes": 1,
                    "lease_decisions": 1,
                }
            ),
            ValueError,
        ),
        (
            eval_policy(
                policy={**eval_policy()["policy"], "mode": "persistent"}
            ),
            ValueError,
        ),
        (
            eval_policy(
                policy={**eval_policy()["policy"], "history_budget_bytes": True}
            ),
            ValueError,
        ),
        (
            eval_policy(
                policy={**eval_policy()["policy"], "lease_decisions": -1}
            ),
            ValueError,
        ),
    ],
)
def test_load_eval_policy_rejects_malformed_shapes(tmp_path, value, error):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(error):
        load_eval_policy(path)


def test_load_eval_policy_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"a-event-native-eval-policy-v1",'
        '"policy_id":"first","policy_id":"second",'
        '"policy":{"history_budget_bytes":1,"workspace_budget_bytes":1,'
        '"lease_decisions":1,"max_retrieved_events":1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_eval_policy(path)


@pytest.mark.parametrize("view_mode", sorted(OVERRIDE_VIEW_MODES))
def test_every_declared_route_accepts_the_complete_override(view_mode):
    source_profile = profile()
    override = eval_policy()

    resolved = resolve_event_native_eval_policy(
        source_profile,
        view_mode=view_mode,
        policy_override=override,
        source_path="configs/a_eval.json",
    )

    assert set(resolved) == {
        "schema",
        "source",
        "policy_id",
        "policy_sha256",
        "eval_policy",
        "source_path",
        "effective_policy",
        "runtime_recovery_cap",
        "field_roles",
    }
    assert resolved["schema"] == RUNTIME_POLICY_SCHEMA
    assert resolved["source"] == "explicit_eval_policy"
    assert resolved["policy_id"] == override["policy_id"]
    assert resolved["policy_sha256"] == canonical_sha256(override)
    assert resolved["eval_policy"] == override
    assert resolved["source_path"] == "configs/a_eval.json"
    assert resolved["runtime_recovery_cap"] == (
        1 if view_mode in RECOVERY_VIEW_MODES else 0
    )
    assert {
        field: resolved["effective_policy"][field] for field in EVAL_POLICY_FIELDS
    } == override["policy"]
    preserved = set(training_policy()) - EVAL_POLICY_FIELDS
    assert {
        field: resolved["effective_policy"][field] for field in preserved
    } == {field: source_profile["policy_contract"][field] for field in preserved}


def test_two_training_policies_resolve_to_the_same_four_eval_values_without_other_mutation():
    left = profile(
        training_policy(
            history_budget_bytes=10,
            workspace_budget_bytes=20,
            lease_decisions=1,
            max_retrieved_events=0,
            kv_bytes_per_token=64,
        )
    )
    right = profile(
        training_policy(
            history_budget_bytes=30,
            workspace_budget_bytes=40,
            lease_decisions=9,
            max_retrieved_events=8,
            kv_bytes_per_token=128,
        )
    )
    left_before, right_before = copy.deepcopy(left), copy.deepcopy(right)
    override = eval_policy()

    left_result = resolve_event_native_eval_policy(
        left, view_mode="capacity_exact_once", policy_override=override
    )
    right_result = resolve_event_native_eval_policy(
        right, view_mode="capacity_exact_once", policy_override=override
    )

    assert {
        field: left_result["effective_policy"][field] for field in EVAL_POLICY_FIELDS
    } == override["policy"]
    assert {
        field: right_result["effective_policy"][field] for field in EVAL_POLICY_FIELDS
    } == override["policy"]
    assert left_result["effective_policy"]["kv_bytes_per_token"] == 64
    assert right_result["effective_policy"]["kv_bytes_per_token"] == 128
    assert left == left_before
    assert right == right_before
    assert left["packing_contract"] == left_before["packing_contract"]
    assert right["packing_contract"] == right_before["packing_contract"]
    assert left["model_geometry"] == left_before["model_geometry"]
    assert right["model_geometry"] == right_before["model_geometry"]


def test_override_does_not_alias_inputs_and_source_path_does_not_change_identity(tmp_path):
    source_profile = profile()
    override = eval_policy()
    profile_before, override_before = copy.deepcopy(source_profile), copy.deepcopy(override)

    first = resolve_event_native_eval_policy(
        source_profile,
        view_mode="full_original",
        policy_override=override,
        source_path=tmp_path / "first.json",
    )
    second = resolve_event_native_eval_policy(
        source_profile,
        view_mode="full_original",
        policy_override=override,
        source_path="elsewhere/second.json",
    )

    assert first["policy_sha256"] == second["policy_sha256"] == canonical_sha256(override)
    assert first["source_path"] == str(tmp_path / "first.json")
    assert second["source_path"] == "elsewhere/second.json"
    first["eval_policy"]["policy"]["lease_decisions"] = 99
    first["effective_policy"]["current_input_baseline"] = "changed"
    assert source_profile == profile_before
    assert override == override_before


@pytest.mark.parametrize("view_mode", [*sorted(OVERRIDE_VIEW_MODES), "static"])
def test_no_override_preserves_a_detached_checkpoint_training_policy(view_mode):
    source_profile = profile()
    before = copy.deepcopy(source_profile)

    resolved = resolve_event_native_eval_policy(source_profile, view_mode=view_mode)

    assert resolved == {
        "schema": RUNTIME_POLICY_SCHEMA,
        "source": "checkpoint_training_policy",
        "policy_id": None,
        "policy_sha256": None,
        "eval_policy": None,
        "source_path": None,
        "effective_policy": source_profile["policy_contract"],
        "runtime_recovery_cap": 1 if view_mode in RECOVERY_VIEW_MODES else 0,
        "field_roles": {
            "history_budget_bytes": (
                "unused"
                if view_mode == "full_original"
                else "auxiliary_gate_and_selection_only"
                if view_mode == "full_exact_shared"
                else "history_limit"
            ),
            "workspace_budget_bytes": (
                "unused" if view_mode == "full_original" else "evidence_limit"
            ),
            "lease_decisions": (
                "retained_evidence_lifetime"
                if view_mode
                in {
                    "capacity_exact_persistent",
                    "full_exact_shared",
                    "capacity_exact_no_gist",
                }
                else "unused"
            ),
            "max_retrieved_events": (
                "fixed_single_event_upgrade"
                if view_mode in RECOVERY_VIEW_MODES
                else "unused"
            ),
        },
    }
    assert resolved["effective_policy"] is not source_profile["policy_contract"]
    assert source_profile == before


def test_static_rejects_override_and_source_path_requires_override():
    with pytest.raises(ValueError, match="static"):
        resolve_event_native_eval_policy(
            profile(), view_mode="static", policy_override=eval_policy()
        )
    with pytest.raises(ValueError, match="source_path requires"):
        resolve_event_native_eval_policy(
            profile(), view_mode="capacity_protect", source_path="unused.json"
        )


def test_unknown_route_and_direct_partial_override_are_rejected():
    with pytest.raises(ValueError, match="view_mode"):
        resolve_event_native_eval_policy(profile(), view_mode="unknown")
    with pytest.raises(ValueError, match="exactly"):
        resolve_event_native_eval_policy(
            profile(),
            view_mode="capacity_protect",
            policy_override={
                "schema": EVAL_POLICY_SCHEMA,
                "policy_id": "partial",
                "policy": {"history_budget_bytes": 1},
            },
        )


@pytest.mark.parametrize("view_mode", sorted(RECOVERY_VIEW_MODES))
@pytest.mark.parametrize("configured_cap", [0, 2])
def test_recovery_route_override_requires_the_actual_single_event_cap(
    view_mode, configured_cap
):
    override = eval_policy()
    override["policy"]["max_retrieved_events"] = configured_cap

    with pytest.raises(ValueError, match="max_retrieved_events=1"):
        resolve_event_native_eval_policy(
            profile(), view_mode=view_mode, policy_override=override
        )


@pytest.mark.parametrize(
    "view_mode, runtime_cap, expected_roles",
    [
        (
            "capacity_protect",
            0,
            {
                "history_budget_bytes": "history_limit",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "unused",
                "max_retrieved_events": "unused",
            },
        ),
        (
            "capacity_exact_once",
            1,
            {
                "history_budget_bytes": "history_limit",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "unused",
                "max_retrieved_events": "fixed_single_event_upgrade",
            },
        ),
        (
            "capacity_exact_persistent",
            1,
            {
                "history_budget_bytes": "history_limit",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "retained_evidence_lifetime",
                "max_retrieved_events": "fixed_single_event_upgrade",
            },
        ),
        (
            "full_exact_shared",
            1,
            {
                "history_budget_bytes": "auxiliary_gate_and_selection_only",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "retained_evidence_lifetime",
                "max_retrieved_events": "fixed_single_event_upgrade",
            },
        ),
        (
            "capacity_exact_no_gist",
            1,
            {
                "history_budget_bytes": "history_limit",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "retained_evidence_lifetime",
                "max_retrieved_events": "fixed_single_event_upgrade",
            },
        ),
        (
            "full_original",
            0,
            {
                "history_budget_bytes": "unused",
                "workspace_budget_bytes": "unused",
                "lease_decisions": "unused",
                "max_retrieved_events": "unused",
            },
        ),
        (
            "static",
            0,
            {
                "history_budget_bytes": "history_limit",
                "workspace_budget_bytes": "evidence_limit",
                "lease_decisions": "unused",
                "max_retrieved_events": "unused",
            },
        ),
    ],
)
def test_all_routes_record_actual_runtime_cap_and_field_roles(
    view_mode, runtime_cap, expected_roles
):
    resolved = resolve_event_native_eval_policy(profile(), view_mode=view_mode)

    assert resolved["runtime_recovery_cap"] == runtime_cap
    assert resolved["field_roles"] == expected_roles


def test_bad_checkpoint_policy_cannot_be_masked_by_valid_override():
    malformed = training_policy()
    del malformed["mode"]

    with pytest.raises(ValueError, match="lacks required fields"):
        resolve_event_native_eval_policy(
            profile(malformed),
            view_mode="capacity_exact_persistent",
            policy_override=eval_policy(),
        )


def test_controller_parser_validates_training_then_effective_policy(monkeypatch):
    from benchmarks.memory_runtime.event_native_policy import EventNativeController

    original = EventNativeController._parse_policy
    observed = []

    def recording_parser(value):
        observed.append(copy.deepcopy(value))
        return original(value)

    monkeypatch.setattr(EventNativeController, "_parse_policy", recording_parser)
    source_profile = profile()
    override = eval_policy()

    resolved = resolve_event_native_eval_policy(
        source_profile,
        view_mode="capacity_exact_no_gist",
        policy_override=override,
    )

    assert observed == [source_profile["policy_contract"], resolved["effective_policy"]]
