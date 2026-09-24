"""RACER mechanism ablations change only their named factor of C1 v2."""
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms import C1_V2_ABLATION_VARIANTS, C1_V2_VARIANTS
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import (
    SELF_REVISION_PROMPT, C1V2CoreController, C1V2NoDraftQueryController,
    C1V2ProbeController, C1V2SelfRevisionController, C1V2VerifiedController,
    c1_v2_fields, validate_c1_v2_config,
)
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_server import _candidate_ready_contract
from benchmarks.memory_runtime.racer.policies import InitialOnlyPolicy
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_static_extensions import proof_payload
from benchmarks.memory_runtime.tests.test_verified_binding import _call
from history_memory.packing import native_ids

CLASSES = {"c1_v2_verified": C1V2VerifiedController, "c1_v2_core": C1V2CoreController,
           "c1_v2_selfrev": C1V2SelfRevisionController,
           "c1_v2_nodraftq": C1V2NoDraftQueryController, "c1_v2_probe": C1V2ProbeController}
SESSION = "bfcl/multi_turn_long_context_7/attempt-0"


class FailRisk:
    def predict_risk(self, context):
        pytest.fail("the probe must not score a non-target decision")


def targets_fields(targets):
    return {"probe_targets": targets, "probe_targets_sha256": hashlib.sha256(json.dumps(
        dict(sorted(targets.items())), sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def config(variant, targets=None):
    return {"variant": variant, "risk_artifact": {"fixture": True}, "risk_threshold": 0.5,
            **c1_v2_fields(variant),
            **(targets_fields(targets or {"multi_turn_long_context_7": "d1"})
               if variant == "c1_v2_probe" else {})}


def factory(monkeypatch, variant, *, score=0.1, budget=4096, targets=None):
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
                        lambda artifact: Risk(score))
    runtime = Path(__file__).resolve().parents[3]
    frozen = json.loads((runtime / "configs/controller.json").read_text(encoding="utf-8"))
    frozen.pop("post_draft_recovery")
    return build_event_native_controller(
        Tokenizer(), packing={**packing(), "ratios": [8]}, policy=policy(budget),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**frozen, "candidate_algorithm": config(variant, targets)})


def history_rows():
    rows = [{"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Look up the order totals for keys 1 through 6."}]
    for number in range(1, 7):
        rows += tool_pair(number, result={"key": number, "total": 1000 + number,
                                          "note": "archived order record " * 20})
    rows.append({"role": "user", "content": "Now refund the order with key 2."})
    return rows


def decision(key="d1", rows=None):
    return {"session_id": SESSION, "decision_key": key, "messages": rows or history_rows(),
            "tools": []}


def triggered_core_result(monkeypatch, variant="c1_v2_core"):
    """A decision where RACER-core admits one archived event and regenerates."""
    c = factory(monkeypatch, variant, score=0.9, budget=600)
    prepared = c.prepare(decision(), ratio=8, max_new_tokens=32)
    calls = [_call("lookup", {"key": 2})]
    return c, prepared, calls, c.reconsider(prepared, calls, draft_text="lookup key 2")


def test_registries_and_ready_contract_cover_every_ablation():
    import importlib.util
    path = Path(__file__).resolve().parents[4] / "candidate_algorithms.py"
    spec = importlib.util.spec_from_file_location("experiment_candidate_registry", path)
    experiment_registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_registry)
    assert set(C1_V2_ABLATION_VARIANTS) < set(C1_V2_VARIANTS)
    assert experiment_registry.C1_V2_VARIANTS == C1_V2_VARIANTS
    assert experiment_registry.SELF_REVISION_PROMPT == SELF_REVISION_PROMPT
    for variant in C1_V2_VARIANTS:
        assert experiment_registry.c1_v2_fields(variant) == c1_v2_fields(variant)
        identity, baseline = _candidate_ready_contract(config(variant))
        assert baseline == "c2kv-c1-v2-verified-v1:" + variant
        assert identity["completion_review"] is False
    assert "ablation" not in c1_v2_fields("c1_v2_verified")


def test_probe_targets_are_bound_to_their_digest():
    bad = config("c1_v2_probe")
    bad["probe_targets"] = {"multi_turn_long_context_7": "d2"}
    with pytest.raises(ValueError, match="digest"):
        validate_c1_v2_config(bad)
    extra = config("c1_v2_core")
    extra.update(targets_fields({"x": "d1"}))
    with pytest.raises(ValueError, match="Only the C1 v2 probe"):
        validate_c1_v2_config(extra)


@pytest.mark.parametrize("variant", C1_V2_ABLATION_VARIANTS)
def test_every_ablation_keeps_the_c1_v2_initial_allocation(monkeypatch, variant):
    reference = factory(monkeypatch, "c1_v2_verified", budget=600)
    ablation = factory(monkeypatch, variant, budget=600)
    assert type(ablation) is CLASSES[variant]
    expected = reference.prepare(decision(), ratio=8, max_new_tokens=32)
    actual = ablation.prepare(decision(), ratio=8, max_new_tokens=32)
    assert actual.memory == expected.memory
    assert actual.eligible_chunks == expected.eligible_chunks


def test_core_keeps_parse_fallback_and_never_corrects_arguments(monkeypatch):
    full = factory(monkeypatch, "c1_v2_verified")
    core = factory(monkeypatch, "c1_v2_core")
    calls = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    outputs = []
    for c in (full, core):
        p = c.prepare(proof_payload(), ratio=8, max_new_tokens=32)
        result = c.reconsider(p, calls, draft_text="call")
        assert not result["regenerate"]
        assert c.validate_commit(p, calls, draft_text="call")["accepted"]
        outputs.append(c.finalize_commit(p, calls))
    (full_output, full_receipt), (core_output, core_receipt) = outputs
    assert full_receipt["changed"]
    assert list(core_output) == calls and core_receipt["changed"] is False
    assert core_receipt["status"] == "argument_correction_disabled_core"
    p = core.prepare(proof_payload() | {"decision_key": "d2"}, ratio=8, max_new_tokens=32)
    core.reconsider(p, calls, draft_text="call")
    verdict = core.validate_commit(p, calls, draft_text="call", parse_error="bad json")
    assert verdict == {"schema": "c2kv-c1-v2-verified-v1", "variant": "c1_v2_core",
                       "accepted": False, "fallback": "original", "reason": "malformed_selected_draft"}


def test_core_recovery_decision_matches_the_full_system(monkeypatch):
    _, _, _, full = triggered_core_result(monkeypatch, "c1_v2_verified")
    _, _, _, core = triggered_core_result(monkeypatch, "c1_v2_core")
    assert core["regenerate"] and core["memory"] == full["memory"]
    for key in ("reason", "gate", "source", "candidate_event_id", "candidate_trials"):
        assert core["decision"][key] == full["decision"][key]
    assert core["decision"]["verified_binding"]["status"] == "argument_correction_disabled_core"


def test_nodraftq_queries_request_and_observation_only(monkeypatch):
    seen = []
    import benchmarks.memory_runtime.candidate_algorithms.c1_v2 as module
    original = module.select_source_event

    def spy(prepared, calls, **kwargs):
        seen.append((list(calls), kwargs))
        return original(prepared, calls, **kwargs)

    monkeypatch.setattr(module, "select_source_event", spy)
    c, prepared, calls, result = triggered_core_result(monkeypatch, "c1_v2_nodraftq")
    assert seen == [([], {"draft_text": "", "include_latest_complete_observation": True,
                          "explicit_revision_abstain": False, "allow_empty_draft_query": True})]
    assert result["decision"]["source"]["draft_in_retrieval_query"] is False
    assert result["decision"]["gate"]["score"] == 0.9
    _, core_prepared, _, core = triggered_core_result(monkeypatch)
    assert (set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
            == set(core_prepared.metadata["eligible_extraction"]["eligible_event_ids"]))


def test_self_revision_keeps_initial_context_and_appends_prompt_then_draft(monkeypatch):
    _, _, _, core = triggered_core_result(monkeypatch)
    c, prepared, calls, result = triggered_core_result(monkeypatch, "c1_v2_selfrev")
    assert core["regenerate"] and result["regenerate"]
    memory = result["memory"]
    assert memory.view == prepared.memory.view and memory.chunks == prepared.memory.chunks
    assert memory.system_input_ids == prepared.memory.system_input_ids
    dummy = {"role": "user", "content": ""}
    plain = native_ids(c.tokenizer, [dummy])
    generation = native_ids(c.tokenizer, [dummy], generation=True)[len(plain):]
    review = native_ids(c.tokenizer, [dummy, {
        "role": "user", "content": SELF_REVISION_PROMPT + "\n\nlookup key 2"}])[len(plain):]
    workspace = tuple(prepared.memory.workspace_input_ids)
    assert tuple(memory.workspace_input_ids) == (
        workspace[:-len(generation)] + tuple(review) + tuple(generation))
    receipt = result["decision"]["self_revision"]
    assert receipt["evidence_provided_to_actor"] is False
    assert receipt["added_live_tokens"] == len(review)
    assert receipt["eligibility"]["candidate_event_id"] == core["decision"]["candidate_event_id"]
    assert "restored_event" not in result["decision"]
    assert result["decision"]["reason"] == "risk_triggered_evidence_free_self_revision"
    assert result["metadata"]["common_raw_prompt_tokens"] == (
        prepared.metadata["common_raw_prompt_tokens"] + len(review))
    check = history_budget_receipt(memory, result["metadata"], c, ratio=8, phase="regeneration")
    assert check["status"] == "passed"
    assert check["active_history_bytes"] == prepared.metadata["actual_history_bytes"]
    assert c._recovery_counts[SESSION] == 1


def test_self_revision_over_model_context_keeps_the_draft(monkeypatch):
    c = factory(monkeypatch, "c1_v2_selfrev", score=0.9, budget=600)
    prepared = c.prepare(decision(), ratio=8, max_new_tokens=32)
    c.model_context = (prepared.memory.workspace_position_start
                       + len(prepared.memory.workspace_input_ids) + 32)
    result = c.reconsider(prepared, [_call("lookup", {"key": 2})], draft_text="lookup key 2")
    assert not result["regenerate"] and result["memory"] == prepared.memory
    assert result["decision"]["reason"] == "self_revision_context_limit"
    assert result["decision"]["self_revision"]["context_check"]["reasons"] == ["model_logical_context"]
    assert c._recovery_counts[SESSION] == 0


def test_probe_skips_the_detector_off_target_and_forces_recovery_at_target(monkeypatch):
    targets = {"multi_turn_long_context_7": "d2"}
    c = factory(monkeypatch, "c1_v2_probe", score=0.1, budget=600, targets=targets)
    c.risk_model = FailRisk()
    off_target = c.prepare(decision("d1"), ratio=8, max_new_tokens=32)
    calls = [_call("lookup", {"key": 2})]
    result = c.reconsider(off_target, calls, draft_text="lookup key 2")
    assert not result["regenerate"] and result["memory"] == off_target.memory
    assert result["decision"]["reason"] == "probe_gate_closed"
    assert result["decision"]["selection"]["selector"] == "probe_closed"
    c.validate_commit(off_target, calls, draft_text="lookup key 2")
    output, receipt = c.finalize_commit(off_target, calls)
    assert list(output) == calls and receipt["changed"] is False

    c = factory(monkeypatch, "c1_v2_probe", score=0.1, budget=600, targets=targets)
    target = c.prepare(decision("d2"), ratio=8, max_new_tokens=32)
    forced = c.reconsider(target, calls, draft_text="lookup key 2")
    _, _, _, core = triggered_core_result(monkeypatch)
    assert forced["decision"]["gate"]["risk_triggered"] is False
    assert forced["decision"]["gate"]["score"] == 0.1
    assert forced["decision"]["gate"]["triggered"] is True
    assert forced["decision"]["probe"]["target"] is True
    assert forced["regenerate"] and forced["memory"] == core["memory"]


def test_probe_off_target_step_serves_the_same_response_as_recovery_off(tmp_path, monkeypatch):
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
    from benchmarks.memory_runtime.tests.test_static_extensions import DraftTokenizer, Generator

    text = '<tool_call>' + json.dumps({"name": "set_budget_limit", "arguments": {
        "access_token": "wrong", "budget_limit": 1500}}) + '</tool_call>'
    records = []
    for variant in ("c1_v2_probe", "c1_v2_verified"):
        c = factory(monkeypatch, variant, targets={"other_task": "d1"})
        if variant == "c1_v2_verified":
            c = InitialOnlyPolicy(c, config(variant))
        tokenizer = DraftTokenizer([text])
        runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8,
            max_new_tokens=32, max_generation_calls=2,
            journal=AttemptJournal(tmp_path / f"{variant}.jsonl"))
        records.append(runner.run(proof_payload()))
    probe, off = records
    assert probe["response"] == off["response"]
    assert probe["generation_completed"] == off["generation_completed"] == 1
    assert json.loads(probe["response"]["tool_calls"][0]["function"]["arguments"])["access_token"] == "wrong"
