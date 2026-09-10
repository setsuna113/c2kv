from __future__ import annotations

import copy
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/memory_runtime"))
import official_pilot


def test_designs_preserve_default_and_freeze_bounded_dev_designs():
    default = official_pilot.design_spec("first-dev4")
    assert default["task_ids"] == [f"multi_turn_base_{i}" for i in range(4)]
    assert default["variants"] == [
        ("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4")]
    assert default["command_extra"] == []

    lease = official_pilot.design_spec("lease-dev2")
    assert lease["task_ids"] == ["multi_turn_base_1", "multi_turn_base_30"]
    assert lease["variants"] == [
        ("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4"),
        ("recover_once", "c2kv4"), ("persistent", "c2kv4"),
        ("no_gist", "full")]
    assert lease["command_extra"] == [
        "--capture-request-views", "--bfcl-temperature", "0.001", "--bfcl-seed", "0"]

    pressure = official_pilot.design_spec("pressure-dev2")
    assert pressure["task_ids"] == ["multi_turn_base_1", "multi_turn_base_30"]
    assert pressure["variants"] == [
        ("full", "full"), ("raw_recency", "full"), ("protect", "c2kv4")]
    assert pressure["maximum_tasks"] == 6
    assert pressure["maximum_wall_seconds"] == 900
    assert pressure["command_extra"] == [
        "--capture-request-views", "--bfcl-temperature", "0.001", "--bfcl-seed", "0"]

    run_id = "a_pressure_dev2"
    raw = official_pilot.runtime_config_for(pressure, "raw_recency", run_id)
    protect = official_pilot.runtime_config_for(pressure, "protect", run_id)
    for variant, config in (("raw_recency", raw), ("protect", protect)):
        assert config["mode"] == variant
        assert config["run_id"] == f"{run_id}_{variant}"
        assert config["bytes_per_kv_token"] == 147456
        assert config["history_budget_bytes"] == 113246208
        assert config["workspace_budget_bytes"] == 113246208

    capacity = official_pilot.design_spec("capacity-dev2")
    assert capacity["task_ids"] == ["multi_turn_base_1", "multi_turn_base_30"]
    assert capacity["variants"] == [
        ("full", "full"), ("raw_recency", "full"),
        ("capacity_protect", "c2kv4")]
    assert capacity["maximum_tasks"] == 6
    assert capacity["maximum_wall_seconds"] == 900
    assert capacity["command_extra"] == [
        "--capture-request-views", "--bfcl-temperature", "0.001", "--bfcl-seed", "0"]
    compressed = official_pilot.runtime_config_for(
        capacity, "capacity_protect", "a_capacity_dev2")
    assert compressed["mode"] == "capacity_protect"
    assert compressed["run_id"] == "a_capacity_dev2_capacity_protect"
    assert compressed["history_budget_bytes"] == 113246208
    assert compressed["workspace_budget_bytes"] == 113246208


def test_reference_dev2_preserves_rank_order_and_fixed_arm_budgets(tmp_path):
    design = official_pilot.design_spec("reference-dev2")
    spec = design["reference_design"]
    assert design["task_ids"] == ["multi_turn_base_183", "multi_turn_base_180"]
    assert design["variants"] == [
        ("full", "full"), ("capacity_protect", "c2kv4"),
        ("capacity_exact_once", "c2kv4"),
        ("capacity_exact_no_gist", "full"),
    ]
    assert spec["maximum_tasks"] == 8
    assert spec["maximum_wall_seconds"] == 1200
    assert spec["maximum_generation_attempts_per_arm"] == 96
    assert spec["maximum_extraction_attempts_per_arm"] == 32
    assert design["reference_design_source"]["path"].endswith("reference_dev2.json")

    args = SimpleNamespace(
        bench_python="bench-python", proxy_python="proxy-python",
        upstream="http://upstream", checkpoint="checkpoint",
        out=tmp_path, design="reference-dev2")
    commands = [
        official_pilot.build_run_command(args, design, "a_ref", name, arm, 34000 + index)
        for index, (name, arm) in enumerate(design["variants"])
    ]
    for command in commands:
        assert command[command.index("--run-ids") + 1] == (
            "multi_turn_base_183,multi_turn_base_180")
        assert command[command.index("--max-generation-attempts") + 1] == "96"
        assert command[command.index("--max-extraction-attempts") + 1] == "32"
    for variant in ("capacity_protect", "capacity_exact_once", "capacity_exact_no_gist"):
        config = json.loads((tmp_path / f"{variant}.config.json").read_text())
        assert config["mode"] == variant
        assert config["history_budget_bytes"] == spec["history_budget_bytes"]
        assert config["workspace_budget_bytes"] == spec["workspace_budget_bytes"]


def _write_g_checkpoint_profile(tmp_path):
    checkpoint = tmp_path / "checkpoint-460"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "gist_param": "qkv",
        "gist_type": "slot",
        "gist_overlap": 0,
        "gist_residual_type": "none",
    }))
    profile_path = tmp_path / "g-profile.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1,
        "profile_kind": "legacy_artifacts",
        "model": {
            "gist_param": "qkv",
            "gist_type": "slot",
            "gist_overlap": 0,
            "gist_residual_type": "none",
        },
        "training": {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "doc_packing": "turn",
            "max_doc_length": 768,
            "max_doc_num": 16,
            "compression_ratios": [4, 8, 16],
        },
        "serving": {
            "compatible": True,
            "compatibility_reason": "test fixture",
            "query_projection": "gist",
            "doc_packing": "turn",
            "max_doc_length": 768,
            "max_doc_num": 16,
            "compression_ratios": [4, 8, 16],
        },
        "provenance": {"claim": "test fixture", "artifacts": []},
        "missing": ["exact_training_git_commit", "training_git_dirty"],
    }))
    return checkpoint, profile_path


def _write_1088_checkpoint_profile(tmp_path):
    checkpoint = tmp_path / "checkpoint-1088"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "gist_param": "qkv",
        "gist_type": "dynamic-interleave",
        "gist_overlap": 64,
        "gist_residual_type": "embed-mean",
    }))
    profile_path = tmp_path / "profile-1088.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1,
        "profile_kind": "reference",
        "reference_name": "checkpoint-1088",
        "model": {
            "gist_param": "qkv", "gist_type": "dynamic-interleave",
            "gist_overlap": 64, "gist_residual_type": "embed-mean",
        },
        "training": {
            "doc_mode": "history_only", "tools_in_system": True,
            "doc_packing": "turn", "max_doc_length": 512,
            "max_doc_num": 12, "compression_ratios": [4, 8, 16],
        },
        "serving": {
            "compatible": True, "compatibility_reason": "test fixture",
            "query_projection": "base", "doc_packing": "turn",
            "max_doc_length": 512, "max_doc_num": 12,
            "compression_ratios": [4, 8, 16],
        },
        "provenance": {"claim": "test fixture", "artifacts": []},
        "missing": ["exact_training_git_commit", "training_git_dirty"],
    }))
    return checkpoint, profile_path


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_frozen_p2_receipt_activates_only_conditional_p3_candidate(tmp_path):
    path = tmp_path / "revision_choice.json"
    receipt = {
        "schema": "a-pre-b-p2-revision-choice-v1",
        "status": official_pilot.pre_b.FROZEN_CANDIDATE_STATUS,
        "design": "pre-b-p2",
        "variant": official_pilot.pre_b.CANDIDATE_VARIANT,
        "method_id": "a-pre-b-acquire-for-next-v1",
        "algorithm": {"current": "keep draft", "next": "consume lease"},
        "evidence": {"analysis": "p2.json"},
    }
    path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    args = SimpleNamespace(
        design="pre-b-p3", p2_revision_receipt=path,
        expected_p2_revision_sha256=digest)

    candidate = official_pilot.resolve_pre_b_candidate(args)
    design = official_pilot.design_spec("pre-b-p3", candidate=candidate)

    assert candidate["method_id"] == "a-pre-b-acquire-for-next-v1"
    assert [variant for variant, _arm in design["variants"]] == [
        *official_pilot.pre_b.P3_VARIANTS,
        official_pilot.pre_b.CANDIDATE_VARIANT,
    ]
    assert design["pre_b_design"]["maximum_generation_attempts"] == 9 * 768


def test_pre_b_executes_rotated_task_shards_and_decrements_arm_caps(
        tmp_path, monkeypatch):
    base = official_pilot.design_spec("pre-b-p3")
    task_ids = list(base["task_ids"][:2])
    variants = [base["variants"][0], base["variants"][5]]
    base["task_ids"] = task_ids
    base["variants"] = variants
    spec = base["pre_b_design"]
    spec["task_ids"] = task_ids
    spec["variants"] = variants
    spec["maximum_tasks"] = 4
    spec["maximum_generation_attempts_per_arm"] = 192
    spec["maximum_generation_attempts"] = 384
    spec["maximum_extraction_attempts_by_arm"] = {
        "full": 0, "ac_exact_persistent": 10}
    spec["maximum_extraction_attempts"] = 10
    base["maximum_tasks"] = 4
    monkeypatch.setattr(
        official_pilot, "design_spec", lambda _name, **_kwargs: base)
    monkeypatch.setattr(
        official_pilot, "resolve_pre_b_execution_inputs",
        lambda *_args: {"bfcl_root": str(tmp_path), "fixture": True})
    monkeypatch.setattr(
        official_pilot, "admit_profile_upstream",
        lambda *_args, **_kwargs: {"schema": "admission", "status": "passed"})
    monkeypatch.setattr(
        official_pilot.pre_b, "execution_source_identity",
        lambda _root: {"schema": "source", "files": []})
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports",
        lambda count: list(range(34000, 34000 + count)))
    checkpoint, profile_path = _write_1088_checkpoint_profile(tmp_path)
    profile = official_pilot.resolve_pilot_checkpoint_profile(SimpleNamespace(
        checkpoint=str(checkpoint), checkpoint_profile=profile_path,
        expected_profile_fingerprint=None))
    monkeypatch.setattr(
        official_pilot, "resolve_pilot_checkpoint_profile", lambda _args: profile)
    out = tmp_path / "run"
    started = []

    class FakeProcess:
        pid = 777

        def __init__(self, argv, **_kwargs):
            task_id = _value_after(argv, "--run-ids")
            shard = Path(_value_after(argv, "--out"))
            variant = shard.name
            generation_cap = int(_value_after(argv, "--max-generation-attempts"))
            extraction_value = (_value_after(argv, "--max-extraction-attempts")
                                if "--max-extraction-attempts" in argv else None)
            started.append((task_id, variant, generation_cap,
                            int(extraction_value) if extraction_value else None))
            is_gist = variant == "ac_exact_persistent"
            attempts = 2 if is_gist and task_id == task_ids[0] else 1
            producer_calls = (3 if task_id == task_ids[0] else 2) if is_gist else 0
            row = {
                "status": "ok", "error_kind": None,
                "eval_context": {"task_id": task_id},
                "request_view": {}, "response_view": {},
                "forwarded_request_views": [{} for _ in range(attempts)],
                "generation_attempts": attempts,
                "n_native_tool_calls": 0, "native_tool_names": [],
                "generation_budget": {
                    "limit": generation_cap, "consumed_before": 0,
                    "consumed_after": attempts,
                    "attempt_indices": list(range(1, attempts + 1)),
                    "per_task_limit": 96, "task_id": task_id,
                    "task_consumed_before": 0,
                    "task_consumed_after": attempts,
                },
                "extraction_budget": ({
                    "limit": int(extraction_value), "consumed_before": 0,
                    "consumed_after": producer_calls,
                    "attempt_indices": list(range(1, producer_calls + 1)),
                } if is_gist else None),
                "extraction_telemetry": {
                    "events": [{
                        "producer_called": True,
                        "budget_attempt_index": index,
                        "client_cache_hit": False,
                    } for index in range(1, producer_calls + 1)],
                    "summary": {"producer_calls": producer_calls},
                },
            }
            logs = shard / "logs"
            logs.mkdir(parents=True)
            (logs / "proxy_test.jsonl").write_text(json.dumps(row) + "\n")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(official_pilot.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", str(checkpoint), "--checkpoint-profile", str(profile_path),
        "--proxy-python", "proxy-python", "--bench-python", "bench-python",
        "--protocol-root", str(tmp_path), "--out", str(out),
        "--design", "pre-b-p3", "--output-identity", "schedule-test",
    ])

    official_pilot.main()

    assert [(task, variant) for task, variant, _gen, _extract in started] == [
        (task_ids[0], "full"),
        (task_ids[0], "ac_exact_persistent"),
        (task_ids[1], "ac_exact_persistent"),
        (task_ids[1], "full"),
    ]
    assert [row[3] for row in started if row[1] == "ac_exact_persistent"] == [10, 7]
    manifest = json.loads((out / "pilot.json").read_text())
    assert len({item["proxy_port"] for item in manifest["commands"]}) == 4
    assert manifest["generation_attempts_by_arm"] == {
        "full": 2, "ac_exact_persistent": 3}
    assert manifest["extraction_attempts_by_arm"] == {
        "full": 0, "ac_exact_persistent": 5}


def test_pre_b_p4_resolves_incumbent_and_forwards_hard_caps(
        tmp_path):
    selection = {
        "status": official_pilot.pre_b.FROZEN_METHOD_STATUS,
        "selected_method": "ac_exact_persistent",
    }
    design = official_pilot.design_spec(
        "pre-b-p4b", method_selection=selection)
    assert [item[0] for item in design["variants"]] == [
        "full", "ac_exact_persistent", "raw_recency", "raw_exact_shared"]
    profile = {
        "profile_fingerprint": "f" * 64,
        "serving": {
            "query_projection": "gist", "doc_packing": "turn",
            "max_doc_length": 768, "max_doc_num": 16,
        },
    }
    args = SimpleNamespace(
        bench_python="bench-python", proxy_python="proxy-python",
        upstream="http://upstream", checkpoint="checkpoint-460",
        out=tmp_path, design="pre-b-p4b",
        max_extraction_attempts_per_arm=None)
    commands = {
        variant: official_pilot.build_run_command(
            args, design, "a_preb", variant, arm, 34000 + index,
            checkpoint_profile=profile,
            checkpoint_profile_path=tmp_path / "profile.json",
            write_config=False)
        for index, (variant, arm) in enumerate(design["variants"])
    }
    for command in commands.values():
        assert _value_after(command, "--max-generation-attempts") == "384"
        assert _value_after(
            command, "--max-generation-attempts-per-task") == "96"
    assert _value_after(
        commands["ac_exact_persistent"], "--max-extraction-attempts") == "6144"
    assert "--max-extraction-attempts" not in commands["full"]
    config = official_pilot.runtime_config_for(
        design, "ac_exact_persistent", "a_preb")
    assert config["mode"] == "ac_exact_persistent"
    assert config["compression_policy"] == "always-compress-v1"
    assert config["history_budget_bytes"] == 113246208


def test_pre_b_preview_emits_resolved_manifest_without_output_or_worker(
        tmp_path, monkeypatch, capsys):
    checkpoint, profile_path = _write_1088_checkpoint_profile(tmp_path)
    profile = official_pilot.resolve_pilot_checkpoint_profile(SimpleNamespace(
        checkpoint=str(checkpoint), checkpoint_profile=profile_path,
        expected_profile_fingerprint=None,
    ))
    bfcl_root = tmp_path / "bfcl"
    data_path = bfcl_root / "bfcl_eval" / "data" / "BFCL_v4_multi_turn_base.json"
    scorer_path = (bfcl_root / "bfcl_eval" / "eval_checker" /
                   "multi_turn_eval" / "multi_turn_checker.py")
    data_path.parent.mkdir(parents=True)
    scorer_path.parent.mkdir(parents=True)
    data_path.write_text('{"id":"fixture"}\n')
    scorer_path.write_text("# scorer fixture\n")
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    scorer_hash = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    original_stage = official_pilot.pre_b.load_stage("pre-b-p3")
    original_stage["data_contract"]["official_source_sha256"] = data_hash
    monkeypatch.setattr(
        official_pilot.pre_b, "load_stage",
        lambda _name, **_kwargs: copy.deepcopy(original_stage))
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports",
        lambda count: list(range(34000, 34000 + count)))
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("preview must not start a worker"))
    out = tmp_path / "preview-out"
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", str(checkpoint), "--checkpoint-profile", str(profile_path),
        "--expected-profile-fingerprint", profile["profile_fingerprint"],
        "--proxy-python", "proxy-python", "--bench-python", "bench-python",
        "--protocol-root", str(tmp_path / "unused-protocol"),
        "--out", str(out), "--design", "pre-b-p3", "--preview-only",
        "--output-identity", "p3-preview-test",
        "--bfcl-root", str(bfcl_root), "--bfcl-data-path", str(data_path),
        "--expected-data-sha256", data_hash,
        "--bfcl-scorer-path", str(scorer_path),
        "--expected-scorer-sha256", scorer_hash,
        "--expected-device", "npu:0", "--expected-dtype", "bfloat16",
    ])

    official_pilot.main()

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["status"] == "preview_only_no_requests"
    assert manifest["execution_state"] == "preview_only_no_requests"
    assert manifest["checkpoint_profile"]["upstream_admission"]["requests"] == 0
    assert manifest["execution_inputs"]["data"]["sha256"] == data_hash
    assert manifest["execution_inputs"]["scorer"]["sha256"] == scorer_hash
    assert manifest["variants"] == list(official_pilot.pre_b.P3_VARIANTS)
    assert manifest["generation_attempts_by_arm"] == {}
    assert manifest["extraction_attempts_by_arm"] == {}
    assert not out.exists()
    assert len(manifest["commands"]) == 64
    assert [item["variant"] for item in manifest["commands"][:8]] == list(
        official_pilot.pre_b.P3_VARIANTS)
    assert [item["variant"] for item in manifest["commands"][8:16]] == [
        *official_pilot.pre_b.P3_VARIANTS[1:],
        official_pilot.pre_b.P3_VARIANTS[0],
    ]
    assert all(item["task_id"] == manifest["task_ids"][0]
               for item in manifest["commands"][:8])
    assert all(item["task_id"] == manifest["task_ids"][1]
               for item in manifest["commands"][8:16])
    for item in manifest["commands"]:
        assert _value_after(item["argv"], "--max-generation-attempts") == "96"
        assert _value_after(
            item["argv"], "--max-generation-attempts-per-task") == "96"
        if item["variant"] in official_pilot.pre_b.GIST_VARIANTS:
            assert _value_after(
                item["argv"], "--max-extraction-attempts") == "9216"
        else:
            assert "--max-extraction-attempts" not in item["argv"]


def test_pre_b_p4_execution_rejects_provisional_method_before_output_or_worker(
        tmp_path, monkeypatch, capsys):
    out = tmp_path / "p4"
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"))
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", "checkpoint", "--proxy-python", "proxy-python",
        "--bench-python", "bench-python", "--protocol-root", str(tmp_path),
        "--out", str(out), "--design", "pre-b-p4a",
    ])
    with pytest.raises(SystemExit) as captured:
        official_pilot.main()
    assert captured.value.code == 2
    assert "requires --selected-method" in capsys.readouterr().err
    assert not out.exists()


def test_shared_exact_optional_g_profile_drives_all_seven_commands_without_runtime_changes(
        tmp_path):
    checkpoint, source_profile_path = _write_g_checkpoint_profile(tmp_path)
    profile = official_pilot.resolve_pilot_checkpoint_profile(SimpleNamespace(
        checkpoint=str(checkpoint), checkpoint_profile=source_profile_path,
        expected_profile_fingerprint=None,
    ))
    out = tmp_path / "pilot"
    out.mkdir()
    frozen_profile_path = out / "checkpoint_profile.resolved.json"
    official_pilot.write_resolved_profile(profile, frozen_profile_path)
    args = SimpleNamespace(
        bench_python="bench-python", proxy_python="proxy-python",
        upstream="http://upstream", checkpoint=str(checkpoint),
        out=out, design="shared-exact-dev8")
    design = official_pilot.design_spec("shared-exact-dev8")
    run_id = "a_shared_exact_dev8"
    expected_configs = {
        variant: official_pilot.runtime_config_for(design, variant, run_id)
        for variant, _arm in design["variants"] if variant != "full"
    }

    commands = [
        official_pilot.build_run_command(
            args, design, run_id, variant, arm, 34000 + index,
            checkpoint_profile=profile,
            checkpoint_profile_path=frozen_profile_path)
        for index, (variant, arm) in enumerate(design["variants"])
    ]

    assert len(commands) == 7
    for command in commands:
        assert "--reference-profile" not in command
        assert "--max-extraction-attempts" not in command
        assert _value_after(command, "--checkpoint-profile") == str(frozen_profile_path)
        assert _value_after(command, "--expected-profile-fingerprint") == profile[
            "profile_fingerprint"]
        assert _value_after(command, "--query-projection") == "gist"
        assert _value_after(command, "--doc-packing") == "turn"
        assert _value_after(command, "--max-doc-length") == "768"
        assert _value_after(command, "--max-doc-num") == "16"
    for variant, expected in expected_configs.items():
        assert json.loads((out / f"{variant}.config.json").read_text()) == expected
        assert expected["history_budget_bytes"] == 113246208
        assert expected["workspace_budget_bytes"] == 113246208
        assert expected["lease_decisions"] == 3
        if variant in {
                "full_exact_shared", "capacity_exact_once",
                "capacity_exact_persistent", "capacity_exact_no_gist"}:
            assert expected["max_retrieved_events"] == 1


def test_shared_exact_optional_extraction_cap_is_forwarded_to_all_seven_arms(
        tmp_path):
    args = SimpleNamespace(
        bench_python="bench-python", proxy_python="proxy-python",
        upstream="http://upstream", checkpoint="checkpoint",
        out=tmp_path, design="shared-exact-dev8",
        max_extraction_attempts_per_arm=6144)
    design = official_pilot.design_spec("shared-exact-dev8")

    commands = [
        official_pilot.build_run_command(
            args, design, "a_shared_cap", variant, arm, 34000 + index)
        for index, (variant, arm) in enumerate(design["variants"])
    ]

    assert len(commands) == 7
    assert all(_value_after(command, "--max-extraction-attempts") == "6144"
               for command in commands)


def test_default_command_preserves_checkpoint_1088_base_argv(tmp_path):
    args = SimpleNamespace(
        bench_python="bench-python", proxy_python="proxy-python",
        upstream="http://upstream", checkpoint="checkpoint",
        out=tmp_path, design="first-dev4")
    design = official_pilot.design_spec("first-dev4")
    command = official_pilot.build_run_command(
        args, design, "a_default", "full", "full", 34000)

    assert command == [
        "bench-python", str(official_pilot.ROOT / "benchmarks/run.py"),
        "--benchmark", "bfcl", "--arm", "full",
        "--upstream", "http://upstream", "--backend", "sglang",
        "--checkpoint", "checkpoint", "--reference-profile", "checkpoint-1088",
        "--query-projection", "base", "--model", "c2kv-agent",
        "--num-workers", "1", "--categories", "multi_turn_base",
        "--run-ids", "multi_turn_base_0,multi_turn_base_1,multi_turn_base_2,multi_turn_base_3",
        "--no-upstream-retries", "--proxy-python", "proxy-python",
        "--proxy-port", "34000", "--out", str(tmp_path / "full"),
        "--exact-out", "--run-name", "a_default_full",
    ]


def _pilot_argv(checkpoint, profile_path, fingerprint, protocol_root, out,
                design="shared-exact-dev8"):
    argv = [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", str(checkpoint), "--checkpoint-profile", str(profile_path),
        "--proxy-python", "proxy-python", "--bench-python", "bench-python",
        "--protocol-root", str(protocol_root), "--out", str(out),
        "--design", design,
    ]
    if fingerprint is not None:
        argv[5:5] = ["--expected-profile-fingerprint", fingerprint]
    return argv


@pytest.mark.parametrize(
    ("design", "cap", "message"),
    [
        ("first-dev4", "6144", "only valid for shared-exact-dev8"),
        ("shared-exact-dev8", "0", "must be a positive integer"),
    ],
)
def test_optional_extraction_cap_rejects_invalid_cli_before_output_or_worker(
        tmp_path, monkeypatch, capsys, design, cap, message):
    out = tmp_path / "pilot"
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"))
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", "checkpoint", "--proxy-python", "proxy-python",
        "--bench-python", "bench-python", "--protocol-root", str(tmp_path),
        "--out", str(out), "--design", design,
        "--max-extraction-attempts-per-arm", cap,
    ])

    with pytest.raises(SystemExit) as captured:
        official_pilot.main()

    assert captured.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


class _MetadataResponse:
    status = 200

    def __init__(self, value):
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.body[:limit]


class _MetadataOpener:
    def __init__(self, model_info, server_info, error=None):
        self.responses = {
            "/get_model_info": model_info,
            "/get_server_info": server_info,
        }
        self.error = error
        self.calls = []

    def open(self, request, timeout):
        endpoint = next(
            path for path in self.responses if request.full_url.endswith(path))
        self.calls.append((request.full_url, timeout))
        if self.error is not None:
            raise self.error
        return _MetadataResponse(self.responses[endpoint])


def _recorded_g_metadata(checkpoint_path):
    return ({
        "model_path": checkpoint_path,
        "tokenizer_path": checkpoint_path,
        "architectures": ["Qwen3ForCausalLM"],
    }, {
        "model_path": checkpoint_path,
        "tokenizer_path": checkpoint_path,
        "enable_c2kv": True,
        "c2kv_query_proj": "gist",
        "api_key": "PRIVATE_SERVER_VALUE",
        "admin_api_key": "PRIVATE_ADMIN_VALUE",
        "attention_backend": "ascend",
    })


def test_recorded_g_upstream_metadata_is_admitted_and_strictly_filtered():
    checkpoint_path = (
        "/home/user/c2kv-checkpoint-eval-20260908/checkpoints/"
        "g_hist_arm_c/checkpoint-1380"
    )
    model_info, server_info = _recorded_g_metadata(checkpoint_path)
    opener = _MetadataOpener(model_info, server_info)
    receipt = official_pilot.admit_profile_upstream(
        "http://127.0.0.1:35461",
        {"checkpoint": {"path": checkpoint_path},
         "serving": {"query_projection": "gist"}},
        opener=opener,
    )

    assert receipt["status"] == "passed"
    assert receipt["requests"] == 2
    assert receipt["retries"] == 0
    assert [url.rsplit("/", 1)[-1] for url, _timeout in opener.calls] == [
        "get_model_info", "get_server_info"]
    assert all(timeout == official_pilot.UPSTREAM_METADATA_TIMEOUT_SECONDS
               for _url, timeout in opener.calls)
    serialized = json.dumps(receipt)
    assert "api_key" not in serialized
    assert "PRIVATE_SERVER_VALUE" not in serialized
    assert "PRIVATE_ADMIN_VALUE" not in serialized
    assert set(receipt["observed"]["get_server_info"]) == {
        "http_status", "model_path", "tokenizer_path", "enable_c2kv",
        "c2kv_query_proj",
    }


def test_passed_upstream_admission_receipt_is_frozen_before_worker_start(
        tmp_path, monkeypatch):
    checkpoint, profile_path = _write_g_checkpoint_profile(tmp_path)
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    _write_protocol_gate(protocol_root)
    model_info, server_info = _recorded_g_metadata(str(checkpoint.resolve()))
    opener = _MetadataOpener(model_info, server_info)
    out = tmp_path / "pilot"
    monkeypatch.setattr(sys, "argv", _pilot_argv(
        checkpoint, profile_path, None, protocol_root, out, design="first-dev4"))
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports",
        lambda _count: (_ for _ in ()).throw(RuntimeError("stop after admission")))
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"))

    with pytest.raises(RuntimeError, match="stop after admission"):
        official_pilot.main(opener=opener)

    receipt = json.loads((out / "upstream_admission.json").read_text())
    assert receipt["status"] == "passed"
    assert "PRIVATE_SERVER_VALUE" not in json.dumps(receipt)
    assert (out / "checkpoint_profile.resolved.json").is_file()


@pytest.mark.parametrize(
    ("case", "message", "expected_gets"),
    [
        ("wrong_checkpoint", "does not match checkpoint profile", 2),
        ("wrong_query", "does not match checkpoint profile", 2),
        ("missing_metadata", "lacks required fields", 2),
        ("network_failure", "failed without retry", 1),
    ],
)
def test_upstream_admission_failures_start_zero_workers(
        tmp_path, monkeypatch, capsys, case, message, expected_gets):
    checkpoint, profile_path = _write_g_checkpoint_profile(tmp_path)
    profile = official_pilot.resolve_pilot_checkpoint_profile(SimpleNamespace(
        checkpoint=str(checkpoint), checkpoint_profile=profile_path,
        expected_profile_fingerprint=None,
    ))
    model_info, server_info = _recorded_g_metadata(str(checkpoint.resolve()))
    error = None
    if case == "wrong_checkpoint":
        model_info["model_path"] = str(tmp_path / "checkpoint-wrong")
    elif case == "wrong_query":
        server_info["c2kv_query_proj"] = "base"
    elif case == "missing_metadata":
        server_info.pop("tokenizer_path")
    else:
        error = official_pilot.urllib.error.URLError("synthetic offline")
    opener = _MetadataOpener(model_info, server_info, error=error)
    out = tmp_path / "pilot"
    workers = []
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: workers.append(True))
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports",
        lambda _count: pytest.fail("port/worker preparation must not start"))
    monkeypatch.setattr(sys, "argv", _pilot_argv(
        checkpoint, profile_path, profile["profile_fingerprint"],
        tmp_path / "missing-protocol", out))

    with pytest.raises(SystemExit) as captured:
        official_pilot.main(opener=opener)

    assert captured.value.code == 2
    assert message in capsys.readouterr().err
    assert workers == []
    assert len(opener.calls) == expected_gets
    assert not out.exists()


def test_changed_profile_fingerprint_fails_before_output_or_subprocess(
        tmp_path, monkeypatch, capsys):
    checkpoint, profile_path = _write_g_checkpoint_profile(tmp_path)
    out = tmp_path / "pilot"
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"))
    monkeypatch.setattr(sys, "argv", _pilot_argv(
        checkpoint, profile_path, "0" * 64, tmp_path / "missing-protocol", out))

    with pytest.raises(SystemExit) as error:
        official_pilot.main()

    assert error.value.code == 2
    assert "planned profile fingerprint" in capsys.readouterr().err
    assert not out.exists()


def test_changed_checkpoint_config_fails_before_output_or_subprocess(
        tmp_path, monkeypatch, capsys):
    checkpoint, source_profile_path = _write_g_checkpoint_profile(tmp_path)
    profile = official_pilot.resolve_pilot_checkpoint_profile(SimpleNamespace(
        checkpoint=str(checkpoint), checkpoint_profile=source_profile_path,
        expected_profile_fingerprint=None,
    ))
    frozen_profile_path = tmp_path / "checkpoint_profile.resolved.json"
    official_pilot.write_resolved_profile(profile, frozen_profile_path)
    (checkpoint / "config.json").write_text(json.dumps({
        "gist_param": "qkv", "gist_type": "changed",
        "gist_overlap": 0, "gist_residual_type": "none",
    }))
    out = tmp_path / "pilot"
    monkeypatch.setattr(
        official_pilot.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"))
    monkeypatch.setattr(sys, "argv", _pilot_argv(
        checkpoint, frozen_profile_path, profile["profile_fingerprint"],
        tmp_path / "missing-protocol", out))

    with pytest.raises(SystemExit) as error:
        official_pilot.main()

    assert error.value.code == 2
    assert "config.json changed" in capsys.readouterr().err
    assert not out.exists()


def write_gate(root, *, bad_cell=None):
    directory = root / "utilization_probe_v1"
    directory.mkdir()
    (directory / "receipt.json").write_text(json.dumps({
        "status": "completed", "chat_attempts": 24, "chat_completed": 24,
        "lease_gate_passed": True}))
    for index in range(24):
        row = {"cell_id": f"c{index}", "status": "completed", "counts": {
            "memory_runtime": {"raw_prompt_tokens_verified_by_backend": True,
                               "byte_geometry_verified_by_backend": True}}}
        if index == bad_cell:
            row["counts"]["memory_runtime"]["raw_prompt_tokens_verified_by_backend"] = False
        (directory / f"c{index}.json").write_text(json.dumps(row))
    return directory


def test_utilization_gate_reads_original_cells(tmp_path):
    directory = write_gate(tmp_path)
    assert official_pilot.require_utilization_gate(tmp_path) == directory / "receipt.json"
    (directory / "c23.json").unlink()
    with pytest.raises(SystemExit, match="exactly 24"):
        official_pilot.require_utilization_gate(tmp_path)


def test_utilization_gate_rejects_unverified_cell(tmp_path):
    write_gate(tmp_path, bad_cell=7)
    with pytest.raises(SystemExit, match="token/byte verification"):
        official_pilot.require_utilization_gate(tmp_path)


def source_bundle(hashes=None):
    hashes = hashes or {
        name: f"sha-{index}"
        for index, name in enumerate(official_pilot.RAW_RECENCY_METHOD_FILES)
    }
    return {
        "base_commit": "abc123",
        "source_tree_state": "uncommitted_snapshot",
        "source_files_sha256": hashes,
        "files": [*hashes, "executed_source.patch"],
    }


def test_raw_recency_cpu_gate_binds_exact_method_source_hashes(tmp_path):
    bundle = source_bundle()
    audit = {**official_pilot.RAW_RECENCY_CPU_AUDIT, "source_bundle": bundle}
    path = tmp_path / "raw_recency_cpu_audit.json"
    path.write_text(json.dumps(audit))
    assert official_pilot.require_raw_recency_cpu_audit(tmp_path, bundle) == path

    stale = json.loads(json.dumps(audit))
    stale["source_bundle"]["source_files_sha256"][
        official_pilot.RAW_RECENCY_METHOD_FILES[0]] = "stale"
    path.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="stale method sources"):
        official_pilot.require_raw_recency_cpu_audit(tmp_path, bundle)


def test_capacity_cpu_gate_binds_budget_and_all_method_source_hashes(tmp_path):
    hashes = {
        name: f"sha-{index}"
        for index, name in enumerate(official_pilot.CAPACITY_METHOD_FILES)
    }
    bundle = source_bundle(hashes)
    bundle["base_commit"] = official_pilot.CAPACITY_SOURCE_BASE_COMMIT
    audit = {**official_pilot.CAPACITY_CPU_AUDIT,
             "source_bundle": json.loads(json.dumps(bundle))}
    path = tmp_path / "capacity_cpu_audit.json"
    path.write_text(json.dumps(audit))
    assert official_pilot.require_capacity_cpu_audit(tmp_path, bundle) == path

    stale = json.loads(json.dumps(audit))
    stale["source_bundle"]["source_files_sha256"][
        official_pilot.CAPACITY_METHOD_FILES[-1]] = "stale"
    path.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="stale method sources"):
        official_pilot.require_capacity_cpu_audit(tmp_path, bundle)

    wrong_budget = json.loads(json.dumps(audit))
    wrong_budget["history_budget_bytes"] -= 1
    path.write_text(json.dumps(wrong_budget))
    with pytest.raises(SystemExit, match="contract failed"):
        official_pilot.require_capacity_cpu_audit(tmp_path, bundle)


def test_timeout_termination_escalates_to_owned_process_group(monkeypatch):
    signals = []
    monkeypatch.setattr(official_pilot.os, "killpg",
                        lambda pid, sig: signals.append((pid, sig)), raising=False)
    monkeypatch.setattr(official_pilot.signal, "SIGKILL", 9, raising=False)
    waits = iter([official_pilot.subprocess.TimeoutExpired("pilot", 10), 9])

    def wait(timeout=None):
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    proc = SimpleNamespace(pid=123, wait=wait)
    assert official_pilot.terminate_process_group(proc) == 9
    assert signals == [(123, official_pilot.signal.SIGTERM),
                       (123, official_pilot.signal.SIGKILL)]


def test_termination_kills_group_after_leader_term_exit_and_bounds_final_wait(
        monkeypatch):
    signals = []
    wait_timeouts = []
    monkeypatch.setattr(
        official_pilot.os, "killpg",
        lambda pid, sig: signals.append((pid, sig)), raising=False)
    monkeypatch.setattr(official_pilot.signal, "SIGKILL", 9, raising=False)
    waits = iter([23, official_pilot.subprocess.TimeoutExpired("pilot", 3)])

    def wait(timeout=None):
        wait_timeouts.append(timeout)
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    proc = SimpleNamespace(pid=456, wait=wait, returncode=23)
    assert official_pilot.terminate_process_group(
        proc, grace_seconds=2, kill_wait_seconds=3) == 23
    assert signals == [(456, official_pilot.signal.SIGTERM),
                       (456, official_pilot.signal.SIGKILL)]
    assert wait_timeouts == [2, 3]


def test_termination_returns_after_both_bounded_waits_expire(monkeypatch):
    signals = []
    wait_timeouts = []
    monkeypatch.setattr(
        official_pilot.os, "killpg",
        lambda pid, sig: signals.append((pid, sig)), raising=False)
    monkeypatch.setattr(official_pilot.signal, "SIGKILL", 9, raising=False)

    def wait(timeout=None):
        wait_timeouts.append(timeout)
        raise official_pilot.subprocess.TimeoutExpired("pilot", timeout)

    proc = SimpleNamespace(pid=789, wait=wait, returncode=None)
    assert official_pilot.terminate_process_group(
        proc, grace_seconds=4, kill_wait_seconds=6) is None
    assert signals == [(789, official_pilot.signal.SIGTERM),
                       (789, official_pilot.signal.SIGKILL)]
    assert wait_timeouts == [4, 6]


def _write_protocol_gate(root):
    first = root / "protocol_v1"
    remainder = root / "protocol_remaining_v1"
    first.mkdir()
    remainder.mkdir()
    rows = [{} for _ in range(8)]
    (first / "receipt.json").write_text(json.dumps({
        "request_receipts": rows, "requests_attempted": 8,
    }))
    (remainder / "receipt.json").write_text(json.dumps({
        "status": "completed", "request_receipts": rows,
        "requests_attempted": 8,
    }))
    (root / "protocol_cpu_audit.json").write_text(json.dumps({
        "status": "passed",
        "rows": [{"selected_view_unchanged": True, "raw_token_parity": True}],
    }))


@pytest.mark.parametrize(
    ("outcome", "clock_values", "expected_status", "expected_wall"),
    [
        ("completed", [0.0, 1.0, 2.0, 12.0], "completed", 12.0),
        ("timeout", [0.0, 1.0, 2.0, 13.0], "wall_budget_exhausted", 13.0),
        ("prestart_exhausted", [0.0, 11.0, 12.0],
         "wall_budget_exhausted", 12.0),
        ("runner_error", [0.0, 1.0, 2.0, 4.0],
         "stopped_on_runner_error", 4.0),
    ],
)
def test_terminal_paths_save_final_measured_wall(
        tmp_path, monkeypatch, outcome, clock_values, expected_status,
        expected_wall):
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    _write_protocol_gate(protocol_root)
    out = tmp_path / "pilot"
    design = {
        "task_ids": ["multi_turn_base_0"],
        "variants": [("full", "full")],
        "task_selection": "test fixture",
        "command_extra": [],
        "maximum_tasks": 1,
        "maximum_wall_seconds": 10,
    }
    monkeypatch.setattr(official_pilot, "design_spec", lambda _name: design)
    monkeypatch.setattr(official_pilot, "free_loopback_ports", lambda _count: [34000])
    ticks = iter(clock_values)
    monkeypatch.setattr(official_pilot.time, "monotonic", lambda: next(ticks))
    terminated = []
    monkeypatch.setattr(
        official_pilot, "terminate_process_group",
        lambda proc: terminated.append(proc.pid) or -15)

    class FakeProcess:
        pid = 123

        def __init__(self, *_args, **_kwargs):
            if outcome == "completed":
                logs = out / "full" / "logs"
                logs.mkdir(parents=True)
                (logs / "proxy_test.jsonl").write_text(json.dumps({
                    "eval_context": {"task_id": "multi_turn_base_0"},
                }) + "\n")

        def wait(self, timeout=None):
            if outcome == "timeout":
                raise official_pilot.subprocess.TimeoutExpired("pilot", timeout)
            return 7 if outcome == "runner_error" else 0

    monkeypatch.setattr(official_pilot.subprocess, "Popen", FakeProcess)

    class NoGetOpener:
        calls = 0

        def open(self, *_args, **_kwargs):
            self.calls += 1
            pytest.fail("default pilot must not request upstream metadata")

    no_get = NoGetOpener()
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", "checkpoint", "--proxy-python", "proxy-python",
        "--bench-python", "bench-python", "--protocol-root",
        str(protocol_root), "--out", str(out), "--design", "first-dev4",
    ])

    if outcome == "completed":
        official_pilot.main(opener=no_get)
    else:
        with pytest.raises(SystemExit):
            official_pilot.main(opener=no_get)

    receipt = json.loads((out / "pilot.json").read_text())
    assert receipt["status"] == expected_status
    assert receipt["wall_seconds"] == expected_wall
    assert receipt["wall_seconds_final"] is True
    assert receipt["wall_seconds_scope"] == official_pilot.WALL_SECONDS_SCOPE
    assert terminated == ([123] if outcome == "timeout" else [])
    assert no_get.calls == 0


def _three_arm_design():
    return {
        "task_ids": ["multi_turn_base_0"],
        "variants": [("full", "full"), ("second", "full"),
                     ("third", "full")],
        "task_selection": "test fixture",
        "command_extra": [],
        "maximum_tasks": 3,
        "maximum_wall_seconds": 60,
    }


def _configure_three_arm_main(tmp_path, monkeypatch):
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    _write_protocol_gate(protocol_root)
    out = tmp_path / "pilot"
    monkeypatch.setattr(
        official_pilot, "design_spec", lambda _name: _three_arm_design())
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports", lambda _count: [34000, 34001, 34002])
    monkeypatch.setattr(
        official_pilot, "build_run_command",
        lambda *_args, **_kwargs: [_args[3]])
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", "checkpoint", "--proxy-python", "proxy-python",
        "--bench-python", "bench-python", "--protocol-root",
        str(protocol_root), "--out", str(out), "--design", "first-dev4",
    ])
    return out


def test_interrupt_handlers_are_skipped_outside_main_thread(monkeypatch):
    worker = object()
    main = object()
    monkeypatch.setattr(official_pilot.threading, "current_thread", lambda: worker)
    monkeypatch.setattr(official_pilot.threading, "main_thread", lambda: main)
    monkeypatch.setattr(
        official_pilot.signal, "signal",
        lambda *_args: pytest.fail("non-main thread must not install handlers"))

    with official_pilot.pilot_interrupt_handlers():
        pass


def test_signal_cleans_active_group_restores_handlers_and_starts_no_next_arm(
        tmp_path, monkeypatch):
    out = _configure_three_arm_main(tmp_path, monkeypatch)
    previous = {
        official_pilot.signal.SIGINT: object(),
        official_pilot.signal.SIGTERM: object(),
    }
    active_handlers = {}
    signal_changes = []

    def install(signum, handler):
        signal_changes.append((signum, handler))
        active_handlers[signum] = handler

    monkeypatch.setattr(
        official_pilot.signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(official_pilot.signal, "signal", install)
    started_variants = []

    class FakeProcess:
        pid = 123

        def __init__(self, argv, **_kwargs):
            started_variants.append(argv[0])

        def wait(self, timeout=None):
            active_handlers[official_pilot.signal.SIGTERM](
                official_pilot.signal.SIGTERM, None)

    monkeypatch.setattr(official_pilot.subprocess, "Popen", FakeProcess)
    terminated = []
    monkeypatch.setattr(
        official_pilot, "terminate_process_group",
        lambda proc: terminated.append(proc.pid) or -9)

    with pytest.raises(SystemExit) as captured:
        official_pilot.main()

    assert captured.value.code == 128 + int(official_pilot.signal.SIGTERM)
    assert started_variants == ["full"]
    assert terminated == [123]
    assert active_handlers == previous
    assert signal_changes[-2:] == [
        (official_pilot.signal.SIGTERM, previous[official_pilot.signal.SIGTERM]),
        (official_pilot.signal.SIGINT, previous[official_pilot.signal.SIGINT]),
    ]
    manifest = json.loads((out / "pilot.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["wall_seconds_final"] is True
    assert manifest["interruption"]["signal"] == official_pilot.signal.SIGTERM
    assert manifest["interruption"]["cleanup_returncode"] == -9
    assert manifest["results"] == [{
        "variant": "full", "returncode": -9, "interrupted": True,
        "signal": official_pilot.signal.SIGTERM,
    }]


def test_controller_exception_cleans_active_group_and_starts_no_next_arm(
        tmp_path, monkeypatch):
    out = _configure_three_arm_main(tmp_path, monkeypatch)
    started_variants = []

    class FakeProcess:
        pid = 456

        def __init__(self, argv, **_kwargs):
            started_variants.append(argv[0])

        def wait(self, timeout=None):
            raise RuntimeError("synthetic wait failure")

    monkeypatch.setattr(official_pilot.subprocess, "Popen", FakeProcess)
    terminated = []
    monkeypatch.setattr(
        official_pilot, "terminate_process_group",
        lambda proc: terminated.append(proc.pid) or -9)

    with pytest.raises(RuntimeError, match="synthetic wait failure"):
        official_pilot.main()

    assert started_variants == ["full"]
    assert terminated == [456]
    manifest = json.loads((out / "pilot.json").read_text())
    assert manifest["status"] == "stopped_on_controller_error"
    assert manifest["wall_seconds_final"] is True
    assert manifest["controller_error"] == {
        "error_type": "RuntimeError",
        "error": "synthetic wait failure",
        "cleanup_returncode": -9,
    }
    assert manifest["results"] == [{
        "variant": "full", "returncode": -9, "controller_error": True,
    }]


def test_shared_exact_extraction_manifest_uses_semantic_caps_and_observed_ledgers(
        tmp_path, monkeypatch):
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    _write_protocol_gate(protocol_root)
    out = tmp_path / "pilot"
    monkeypatch.setattr(
        official_pilot, "load_source_bundle", lambda *_args: {"fixture": True})
    monkeypatch.setattr(
        official_pilot.shared_exact, "verify_current_bundle",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        official_pilot.shared_exact, "load_control_gate",
        lambda *_args, **_kwargs: {"status": "passed"})
    monkeypatch.setattr(
        official_pilot.shared_exact, "validate_generation_budget_rows",
        lambda rows, _limit: len(rows))
    design = official_pilot.design_spec("shared-exact-dev8")
    monkeypatch.setattr(
        official_pilot, "free_loopback_ports",
        lambda count: list(range(34000, 34000 + count)))
    started_variants = []

    class FakeProcess:
        pid = 789

        def __init__(self, argv, **_kwargs):
            variant = Path(argv[argv.index("--out") + 1]).name
            started_variants.append(variant)
            logs = out / variant / "logs"
            logs.mkdir(parents=True)
            rows = []
            is_zero = variant in official_pilot.SHARED_ZERO_EXTRACTION_VARIANTS
            for index, task_id in enumerate(design["task_ids"]):
                attempt_indices = [] if is_zero else [index + 1]
                consumed_before = 0 if is_zero else index
                consumed_after = consumed_before + len(attempt_indices)
                events = [] if is_zero else [{
                    "producer_called": True,
                    "budget_attempt_index": index + 1,
                }]
                rows.append({
                    "eval_context": {"task_id": task_id},
                    "status": "ok",
                    "error_kind": None,
                    "request_view": {},
                    "response_view": {},
                    "forwarded_request_views": [{}],
                    "generation_attempts": 1,
                    "n_native_tool_calls": 0,
                    "native_tool_names": [],
                    "extraction_budget": {
                        "limit": 6144,
                        "attempt_indices": attempt_indices,
                        "consumed_before": consumed_before,
                        "consumed_after": consumed_after,
                    },
                    "extraction_telemetry": {
                        "events": events,
                        "summary": {"producer_calls": len(events)},
                    },
                })
            (logs / "proxy_test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(official_pilot.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sys, "argv", [
        "official_pilot.py", "--upstream", "http://upstream",
        "--checkpoint", "checkpoint", "--proxy-python", "proxy-python",
        "--bench-python", "bench-python", "--protocol-root",
        str(protocol_root), "--out", str(out), "--design", "shared-exact-dev8",
        "--shared-exact-cpu-audit", str(tmp_path / "cpu.json"),
        "--shared-exact-live-root", str(tmp_path / "live"),
        "--max-extraction-attempts-per-arm", "6144",
    ])

    official_pilot.main()

    manifest = json.loads((out / "pilot.json").read_text())
    assert manifest["status"] == "completed"
    assert started_variants == [variant for variant, _arm in design["variants"]]
    assert manifest["maximum_extraction_attempts_per_arm"] == 6144
    assert manifest["maximum_extraction_attempts"] == 24576
    assert manifest["maximum_extraction_attempts_by_arm"] == {
        variant: (0 if variant in official_pilot.SHARED_ZERO_EXTRACTION_VARIANTS
                  else 6144)
        for variant, _arm in design["variants"]
    }
    assert manifest["extraction_attempts_by_arm"] == {
        variant: (0 if variant in official_pilot.SHARED_ZERO_EXTRACTION_VARIANTS
                  else len(design["task_ids"]))
        for variant, _arm in design["variants"]
    }
    assert manifest["budget_transfer_between_arms"] is False
    for command in manifest["commands"]:
        assert _value_after(
            command["argv"], "--max-extraction-attempts") == "6144"
