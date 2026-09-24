"""CPU integration checks for shared-draft native tool supplementation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.event_native_tool import (
    ToolRegionController, parse_native_tool_spec, validate_tool_recovery_config,
)
from benchmarks.memory_runtime.tests.test_c1_v2_verified import controller, factory_controller
from benchmarks.memory_runtime.tests.test_candidate_allocation import messages
from benchmarks.memory_runtime.tests.test_event_native_tool import CharacterTokenizer, RecordingGenerator
from benchmarks.memory_runtime.tests.test_goal_composition import call
from benchmarks.memory_runtime.tests.test_verified_controller import Binding


def payload():
    return {
        "session_id": "tool-recovery", "decision_key": "turn-0/step-0",
        "messages": [
            {"role": "system", "content": "Use the visible APIs."},
            {"role": "user", "content": "Get the current fuel level."},
        ],
        "tools": [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {"option": {"type": "string"}},
                           "required": ["option"]},
        }} for name, description in (
            ("displayCarStatus", "Use option fuel for the fuel level, never fuelLevel."),
            ("lookupZipcode", "Look up a city zipcode before estimating distance."),
        )],
    }


def wrapped(*, mode="draft-full-raw", score=0.1, binding=None, model_context=50000):
    tokenizer = CharacterTokenizer()
    inner = controller(tokenizer=tokenizer, score=score, binding=binding)
    generator = RecordingGenerator()
    outer = ToolRegionController(
        inner, tokenizer, parse_native_tool_spec("t0:r8:uniform:schema"),
        model_context=model_context, generator=generator, tool_recovery=mode)
    return outer, inner, tokenizer


@pytest.mark.parametrize("spec,budget", [
    (None, None), ("t0:r8", None), ("t0:r8:hybrid3:schema", None),
    ("h2o:r8:uniform:schema", None), ("t0:r8:uniform:schema", 256),
])
def test_recovery_requires_the_declared_uniform_unbudgeted_contract(spec, budget):
    with pytest.raises(ValueError):
        validate_tool_recovery_config(parse_native_tool_spec(spec), "draft-full-raw", budget)


@pytest.mark.parametrize("calls,error,reason", [
    ([], None, "no_compressed_called_tool"),
    ([call("displayCarStatus")], "malformed", "malformed_draft"),
])
def test_coverage_gate_does_not_claim_semantic_failure(calls, error, reason):
    outer, _, _ = wrapped()
    p = outer.prepare(payload(), ratio=8, max_new_tokens=32)
    result = outer.reconsider(p, calls, draft_text="draft", parse_error=error)
    assert not result["regenerate"]
    receipt = result["decision"]["tool_recovery"]
    assert receipt["reason"] == reason and not receipt["triggered"]


@pytest.mark.parametrize("repeats_original", [False, True])
def test_tool_only_restoration_preserves_history_and_invalidates_old_proof(repeats_original):
    binding = Binding()
    outer, inner, tokenizer = wrapped(binding=binding)
    p = outer.prepare(payload(), ratio=8, max_new_tokens=32)
    calls = [call("displayCarStatus", {"option": "fuelLevel"})]
    result = outer.reconsider(p, calls, draft_text="draft")
    assert result["regenerate"]
    assert result["decision"]["history_recovery"]["reason"] == "risk_not_above_threshold"
    assert result["decision"]["tool_recovery"]["restored_tool_indices"] == [0, 1]
    initial = tokenizer.decode(p.memory.system_input_ids)
    restored = tokenizer.decode(result["memory"].system_input_ids)
    assert "never fuelLevel" not in initial and "never fuelLevel" in restored
    assert "before estimating distance" in restored
    assert result["memory"].view == p.inner.memory.view
    assert result["memory"].chunks == p.inner.memory.chunks
    assert result["memory"].workspace_input_ids == p.inner.memory.workspace_input_ids
    assert result["metadata"]["actual_history_bytes"] == p.inner.metadata["actual_history_bytes"]
    assert history_budget_receipt(result["memory"], result["metadata"], outer,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    assert inner._recovery_counts[p.source_payload["session_id"]] == 1
    changed_calls = calls if repeats_original else [call("lookupZipcode", {"option": "London"})]
    outer.validate_commit(p, changed_calls, draft_text="new draft")
    final, receipt = outer.finalize_commit(p, changed_calls)
    assert final == tuple(changed_calls)
    assert binding.applied == 0
    assert receipt["status"] == "tool_recovery_preserved"
    # Reading a previously reviewed decision must neither reapply proofs nor
    # spend another regeneration from the shared accounting.
    assert outer.reconsider(p, calls, draft_text="draft") == result
    assert inner._recovery_counts[p.source_payload["session_id"]] == 1
    with pytest.raises(ValueError, match="another held draft"):
        outer.reconsider(p, changed_calls, draft_text="different draft")


@pytest.mark.parametrize("restore_tools", [False, True])
def test_recovery_reuses_the_selected_history_and_counts_once(monkeypatch, restore_tools):
    outer, inner, _ = wrapped(score=0.9)
    data = payload()
    data["messages"] = messages()
    p = outer.prepare(data, ratio=8, max_new_tokens=32)
    candidate = p.inner.memory.view.gist_event_ids[0]
    monkeypatch.setattr(inner, "_select_source", lambda *args: (
        candidate, {"ranked_candidate_event_ids": [candidate]}))
    calls = [call("displayCarStatus")] if restore_tools else []
    selected = inner.reconsider(p.inner, calls, draft_text="draft")
    assert selected["regenerate"]
    assert candidate in selected["memory"].view.raw_event_ids
    assert candidate not in p.inner.memory.view.raw_event_ids
    result = outer.reconsider(p, calls, draft_text="draft")
    assert result["regenerate"]
    if restore_tools:
        assert result["decision"]["history_recovery"]["reason"] == selected["decision"]["reason"]
    else:
        assert result["decision"]["reason"] == selected["decision"]["reason"]
        assert not result["decision"]["tool_recovery"]["triggered"]
    assert result["memory"].view == selected["memory"].view
    assert result["memory"].workspace_input_ids == selected["memory"].workspace_input_ids
    assert inner._recovery_counts[p.source_payload["session_id"]] == 1
    committed = []
    monkeypatch.setattr(inner, "commit_memory", lambda prepared, memory: committed.append(memory), raising=False)
    outer.commit_memory(p, result["memory"])
    assert committed[0] == selected["memory"]
    assert all(chunk.projection_set != "tool" for chunk in committed[0].chunks)


def test_tool_context_rejection_preserves_an_admitted_history_recovery(monkeypatch):
    from benchmarks.memory_runtime.tool_recovery_packing import ToolRecoveryContextExceeded

    outer, inner, _ = wrapped(score=0.9)
    data = payload()
    data["messages"] = messages()
    p = outer.prepare(data, ratio=8, max_new_tokens=32)
    candidate = p.inner.memory.view.gist_event_ids[0]
    monkeypatch.setattr(inner, "_select_source", lambda *args: (
        candidate, {"ranked_candidate_event_ids": [candidate]}))

    def reject(*args, **kwargs):
        raise ToolRecoveryContextExceeded("fixture: raw tools exceed context")

    monkeypatch.setattr("benchmarks.memory_runtime.tool_recovery_packing.rebuild_tool_history_memory", reject)
    result = outer.reconsider(p, [call("displayCarStatus")], draft_text="draft")
    assert result["regenerate"]
    assert result["decision"]["tool_recovery"]["status"] == "model_context_exceeded"
    assert candidate in result["memory"].view.raw_event_ids
    assert inner._recovery_counts[p.source_payload["session_id"]] == 1
    assert history_budget_receipt(result["memory"], result["metadata"], outer,
                                  ratio=8, phase="regeneration")["status"] == "passed"


def test_exhausted_shared_budget_precedes_history_risk_and_recovery_count():
    outer, inner, _ = wrapped(score=0.9)
    p = outer.prepare(payload(), ratio=8, max_new_tokens=32)
    outer.observe_generation_budget(p, remaining_generation_calls=0)
    inner.risk_model.predict_risk = lambda *args: pytest.fail("Exhausted cap must precede history risk")
    result = outer.reconsider(p, [call("displayCarStatus")], draft_text="draft")
    assert not result["regenerate"]
    assert result["decision"]["reason"] == "shared_task_generation_limit"
    assert inner._recovery_counts.get(p.source_payload["session_id"], 0) == 0


def test_context_rejection_keeps_the_original_draft_and_verified_path(monkeypatch):
    from benchmarks.memory_runtime.tool_recovery_packing import ToolRecoveryContextExceeded

    def reject(*args, **kwargs):
        raise ToolRecoveryContextExceeded("fixture: raw documents exceed context")

    binding = Binding()
    outer, _, _ = wrapped(binding=binding)
    p = outer.prepare(payload(), ratio=8, max_new_tokens=32)
    monkeypatch.setattr("benchmarks.memory_runtime.tool_recovery_packing.rebuild_tool_history_memory", reject)
    calls = [call("displayCarStatus")]
    result = outer.reconsider(p, calls, draft_text="draft")
    assert not result["regenerate"]
    assert result["decision"]["tool_recovery"]["status"] == "model_context_exceeded"
    assert result["memory"] == p.memory
    outer.validate_commit(p, calls, draft_text="draft")
    _, receipt = outer.finalize_commit(p, calls)
    assert receipt["changed"] and binding.applied == 1


class TextGenerator(RecordingGenerator):
    def __init__(self, texts):
        super().__init__()
        self.texts = iter(texts)

    def generate(self, memory, **kwargs):
        self.memories.append(memory)
        self.requests.append(kwargs)
        tokens = tuple(map(ord, next(self.texts)))
        return SimpleNamespace(token_ids=tokens, finish_reason="stop",
                               token_logprobs=(0.0,) * len(tokens), stats={"eos_token_ids": ()})


@pytest.mark.parametrize("mode,first,cap,expected", [
    ("draft-full-raw", '<tool_call>{"name":"displayCarStatus","arguments":{"option":"fuelLevel"}}</tool_call>', 2, 2),
    ("draft-full-raw", "Done", 2, 1),
    ("always-full-raw", "Done", 2, 2),
    ("draft-full-raw", '<tool_call>{"name":"displayCarStatus","arguments":{"option":"fuelLevel"}}</tool_call>', 1, 1),
])
def test_real_c1_factory_runner_uses_one_shared_regeneration(tmp_path, monkeypatch, mode, first, cap, expected):
    tokenizer = CharacterTokenizer()
    inner = factory_controller(monkeypatch, tokenizer=tokenizer)
    corrected = '<tool_call>{"name":"lookupZipcode","arguments":{"option":"London"}}</tool_call>'
    generator = TextGenerator([first, corrected])
    outer = ToolRegionController(inner, tokenizer, parse_native_tool_spec("t0:r8:uniform:schema"),
                                 model_context=50000, generator=generator, tool_recovery=mode)
    runner = EventNativeDecisionRunner(outer, generator, tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=cap,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(payload())
    assert record["generation_completed"] == expected
    assert len(generator.memories) == expected
    assert all(row["status"] == "passed" for row in record["pre_generation_budget_checks"])
    if expected == 2:
        assert record["response"]["tool_calls"][0]["function"]["name"] == "lookupZipcode"
        assert record["commit_transform"]["status"] == "tool_recovery_preserved"
        assert not record["commit_transform"]["changed"]
        assert len(record["recovery_rounds"]) == 1
        assert all(chunk.projection_set != "tool" for chunk in generator.memories[1].chunks)
    if cap == 1:
        assert record["exact_recovery"]["reason"] == "shared_task_generation_limit"


@pytest.mark.parametrize("capacity_rejected", [False, True])
def test_runner_restores_original_proof_only_when_it_selects_original_draft(tmp_path, monkeypatch, capacity_rejected):
    binding = Binding()
    outer, _, tokenizer = wrapped(binding=binding)
    first = '<tool_call>{"name":"displayCarStatus","arguments":{"option":"fuelLevel"}}</tool_call>'
    generator = TextGenerator([first, '<tool_call>{broken</tool_call>'])
    outer.generator = generator
    runner = EventNativeDecisionRunner(outer, generator, tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    if capacity_rejected:
        monkeypatch.setattr(runner, "_regeneration_capacity", lambda memory: {"reason": "fixture_capacity"})
    record = runner.run(payload())
    assert record["generation_completed"] == (1 if capacity_rejected else 2)
    assert record["commit_restore"]["restored_original_draft_proof"]
    assert record["commit_restore"]["selected_generation_index"] == 0
    assert record["commit_transform"]["changed"]
    assert binding.applied == 1


def bfcl_payload():
    """BFCL shape: tools in the request field and no source system message."""
    data = payload()
    data["messages"] = data["messages"][1:]
    return data


def test_bfcl_shaped_source_without_system_restores_only_the_inserted_protocol():
    """box4 BFCL base: every task failed at decision 0 because the rebuild
    required a source system message that BFCL never sends."""
    outer, inner, tokenizer = wrapped()
    data = bfcl_payload()
    p = outer.prepare(data, ratio=8, max_new_tokens=32)
    assert p.plan.messages[0]["role"] == "system"
    assert len(p.plan.messages) == len(data["messages"]) + 1
    calls = [call("displayCarStatus", {"option": "fuelLevel"})]
    result = outer.reconsider(p, calls, draft_text="draft")
    assert result["regenerate"]
    receipt = result["decision"]["tool_recovery"]
    assert (receipt["reason"], receipt["status"]) == ("called_tool_document_compressed", "restored")
    initial = tokenizer.decode(p.memory.system_input_ids)
    restored = tokenizer.decode(result["memory"].system_input_ids)
    assert "never fuelLevel" not in initial and "never fuelLevel" in restored
    assert result["memory"].view == p.inner.memory.view
    assert result["memory"].workspace_input_ids == p.inner.memory.workspace_input_ids
    assert history_budget_receipt(result["memory"], result["metadata"], outer,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    assert inner._recovery_counts[p.source_payload["session_id"]] == 1


def test_bfcl_shaped_source_rejects_a_replacement_frame_without_the_protocol():
    from benchmarks.memory_runtime.tool_recovery_packing import rebuild_tool_history_memory

    outer, _, tokenizer = wrapped()
    data = bfcl_payload()
    p = outer.prepare(data, ratio=8, max_new_tokens=32)
    with pytest.raises(ValueError, match="source message indices"):
        rebuild_tool_history_memory(
            tokenizer, p.inner.memory, p.inner.metadata, p.source_payload,
            p.plan.messages, list(data["messages"]), ratio=8, max_new_tokens=32,
            model_context=50000)


def test_bfcl_shaped_runner_draft_and_tool_regeneration(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    inner = factory_controller(monkeypatch, tokenizer=tokenizer)
    first = '<tool_call>{"name":"displayCarStatus","arguments":{"option":"fuelLevel"}}</tool_call>'
    corrected = '<tool_call>{"name":"displayCarStatus","arguments":{"option":"fuel"}}</tool_call>'
    generator = TextGenerator([first, corrected])
    outer = ToolRegionController(inner, tokenizer, parse_native_tool_spec("t0:r8:uniform:schema"),
                                 model_context=50000, generator=generator, tool_recovery="draft-full-raw")
    runner = EventNativeDecisionRunner(outer, generator, tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(bfcl_payload())
    assert record["status"] == "ok"
    assert record["generation_completed"] == 2
    assert record["exact_recovery"]["tool_recovery"]["status"] == "restored"
    import json
    arguments = record["response"]["tool_calls"][0]["function"]["arguments"]
    assert (json.loads(arguments) if isinstance(arguments, str) else arguments) == {"option": "fuel"}
