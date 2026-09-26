"""Actual controller hold/commit semantics with a deterministic backend seam."""
import copy
import hashlib
import json
import pickle
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNTIME = Path(__file__).resolve().parent / "runtime"
sys.path.insert(0, str(RUNTIME / "python"))
sys.path.insert(0, str(RUNTIME))

from experiments.history_system.t02_runtime import T02Actor, _local_rng, _restore_local_rng
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.event_native_step import (
    EventNativeDecisionRunner,
    EventNativeStepError,
)
from benchmarks.memory_runtime.tests.test_event_native_recovery import request, margin_trace
from benchmarks.memory_runtime.tests.test_event_native_step import Generator, tool
from benchmarks.memory_runtime.tests.test_evidence_sets import LocalModels, config
from benchmarks.memory_runtime.tests.test_gp_recovery import make_controller
from history_memory.sglang_generator import SGLangExtractionBudgetExhausted


class SnapshotGenerator(Generator):
    def capture_exact_state(self):
        self.saved = copy.deepcopy((self.outputs, self.inputs))
        digest = hashlib.sha256(pickle.dumps(self.saved)).hexdigest()
        return {"snapshot_id": "test-backend", "component_digests":
                {k: digest for k in ("actor_kv", "actor_positions", "backend_stats", "rng")}}

    def restore_exact_state(self, snapshot):
        self.outputs, self.inputs = copy.deepcopy(self.saved)
        return copy.deepcopy(snapshot)

    def release_exact_state(self, snapshot):
        self.saved = None


def make_actor(tmp_path):
    controller = make_controller(**config(Q="archive_rrf", K=4,
        set_selector="candidate_rule", export_selection_state=True))
    controller.backends = LocalModels()
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    generator = SnapshotGenerator(journal.path, [tool("draft"), tool("regenerated")],
                                  stats={"shadow_features": margin_trace(0.1)})
    runner = EventNativeDecisionRunner(controller, generator, controller.tokenizer,
        ratio=4, max_new_tokens=32, max_generation_calls=96, journal=journal)
    return T02Actor(runner), generator


def tool_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_failed_submission_counts_cost_but_budget_rejection_does_not(tmp_path):
    actor, generator = make_actor(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("backend submission failed")

    generator.generate = fail
    with pytest.raises(RuntimeError, match="backend submission failed"):
        actor.hold(request())
    assert actor.total_generation_calls == actor.runner.generation_calls == 1
    actor.runner.max_generation_calls = 1
    with pytest.raises(RuntimeError, match="cap exhausted"):
        actor.hold(request())
    assert actor.total_generation_calls == 1
    actor.close()


def _extraction_budget_error():
    return SGLangExtractionBudgetExhausted(
        "C2KV_EXTRACTION_BUDGET_EXHAUSTED: denied miss",
        {
            "schema": "c2kv-native-extraction-failure-v1",
            "model_calls": 172,
        },
    )


def test_verified_extraction_budget_failure_terminates_source_as_capacity(tmp_path):
    actor, generator = make_actor(tmp_path)

    def fail(*args, **kwargs):
        raise _extraction_budget_error()

    generator.generate = fail
    with pytest.raises(CapacityInfeasible, match="EXTRACTION_BUDGET_EXHAUSTED"):
        actor.hold(request())
    assert actor.total_generation_calls == actor.runner.generation_calls == 1
    actor.close()


def test_verified_extraction_budget_failure_terminates_continuation_as_capacity(tmp_path):
    actor, generator = make_actor(tmp_path)

    def fail(*args, **kwargs):
        raise _extraction_budget_error()

    generator.generate = fail
    with pytest.raises(CapacityInfeasible, match="EXTRACTION_BUDGET_EXHAUSTED") as captured:
        actor.generate(request())
    step_error = captured.value.__cause__
    assert isinstance(step_error, EventNativeStepError)
    assert step_error.record["error"] == {
        "type": "SGLangExtractionBudgetExhausted",
        "message": "C2KV_EXTRACTION_BUDGET_EXHAUSTED: denied miss",
    }
    assert actor.total_generation_calls == actor.runner.generation_calls == 1
    actor.close()


def test_extraction_budget_failure_during_submit_is_not_recorded_complete(tmp_path):
    actor, generator = make_actor(tmp_path)
    state = actor.hold(request())
    action = next(
        item["candidate_ids"]
        for item in state["allowed_actions"]
        if len(item["candidate_ids"]) == 1
    )

    def fail(*args, **kwargs):
        raise _extraction_budget_error()

    generator.generate = fail
    with pytest.raises(CapacityInfeasible, match="EXTRACTION_BUDGET_EXHAUSTED"):
        actor.submit_held(action)
    assert actor.last_record is None
    assert actor.total_generation_calls == actor.runner.generation_calls == 2
    actor.close()


def test_snapshot_branches_keep_original_draft_and_restore_before_append(tmp_path):
    actor, generator = make_actor(tmp_path)
    state = actor.hold(request())
    assert len(state["candidates"]) >= 2
    assert actor.runner.controller._recovery_counts == {}
    snapshot = actor.capture()
    original_input = copy.deepcopy(generator.inputs)
    first = actor.submit_held([])
    assert first == state["held_draft_response"]
    assert generator.inputs == original_input
    actor.restore(snapshot)
    action = next(x["candidate_ids"] for x in state["allowed_actions"] if len(x["candidate_ids"]) == 1)
    second = actor.submit_held(action)
    assert second != first
    assert len(generator.inputs) == 2
    assert actor.last_record["exact_recovery"]["appended_unit_ids"] == action
    actor.restore(snapshot)
    third = actor.submit_held([])
    assert third == first
    assert generator.inputs == original_input
    assert actor.total_generation_calls == 2
    actor.close()


def test_restore_replays_append_only_bfcl_tool_reveal_in_same_session(tmp_path):
    actor, generator = make_actor(tmp_path)
    generator.outputs = [tool("draft"), tool("after-reveal"), tool("unused")]
    initial = request()
    initial["tools"] = [tool_schema(f"existing_{index}") for index in range(31)]

    first_state = actor.hold(initial)
    snapshot = actor.capture()
    first_response = actor.submit_held([])
    first_call = first_response["tool_calls"][0]
    extended = request("d2")
    extended["messages"] = [
        *initial["messages"],
        first_response,
        {
            "role": "tool",
            "tool_call_id": first_call["id"],
            "content": json.dumps({"ok": True}),
        },
        {"role": "user", "content": "The budget setter is now available."},
    ]
    extended["tools"] = [
        *copy.deepcopy(initial["tools"]),
        tool_schema("set_budget_limit"),
    ]
    assert (len(initial["tools"]), len(extended["tools"])) == (31, 32)

    second_state = actor.hold(extended)
    assert json.loads(actor.runner.controller._packer._sessions["task-1"].tools_json) == (
        extended["tools"]
    )
    actor.submit_held([])

    actor.restore(snapshot)
    assert json.loads(actor.runner.controller._packer._sessions["task-1"].tools_json) == (
        initial["tools"]
    )
    assert actor.submit_held([]) == first_state["held_draft_response"]
    replayed = actor.hold(extended)
    assert replayed["allowed_actions"] == second_state["allowed_actions"]
    assert json.loads(actor.runner.controller._packer._sessions["task-1"].tools_json) == (
        extended["tools"]
    )
    actor.submit_held([])
    actor.close()


def test_unlisted_action_is_not_executed(tmp_path):
    actor, generator = make_actor(tmp_path)
    actor.hold(request())
    with pytest.raises(ValueError, match="frozen legal"):
        actor.submit_held(["not-in-catalog"])
    assert len(generator.inputs) == 1
    actor.close()


def test_a2_proposal_keeps_c0_policy_and_does_not_generate_actor_tokens(tmp_path):
    actor, generator = make_actor(tmp_path)
    original = actor.hold(request())
    actor.runner.controller.backends.action = 2
    proposed = actor.propose_alternative()
    assert proposed["set_selector"] == "candidate_rule"
    assert actor.runner.controller.gp["set_selector"] == "candidate_rule"
    assert proposed["local_llm_selected_ids"] == proposed["allowed_actions"][2]["candidate_ids"]
    assert proposed["local_llm_proposal"]["selector"] == "local_llm"
    assert proposed["held_draft_response"] == original["held_draft_response"]
    assert len(generator.inputs) == 1
    assert actor.submit_held([]) == original["held_draft_response"]
    actor.close()


def test_restore_resets_shared_model_cache_lru_and_receipts_without_copying_weights(tmp_path):
    actor, _ = make_actor(tmp_path)
    service = actor.runner.controller.backends
    model = object()
    service._bundles = {"embedding": {"model": model}}
    service._embedding_cache = OrderedDict([("first", ([1.0], 1)), ("second", ([2.0], 1))])
    service._receipts = [{"capability": "embed", "input_count": 2}]
    actor.hold(request())
    snapshot = actor.capture()
    expected_cache = copy.deepcopy(service._embedding_cache)
    expected_receipts = copy.deepcopy(service._receipts)
    actor.submit_held([])
    service._embedding_cache.move_to_end("first")
    service._embedding_cache["branch-only"] = ([9.0], 1)
    service._receipts.append({"capability": "branch-only"})
    actor.restore(snapshot)
    assert actor.runner.controller.backends is service
    assert service._bundles["embedding"]["model"] is model
    assert service._embedding_cache == expected_cache
    assert service._receipts == expected_receipts
    actor.close()


class ScopedSnapshotGenerator(SnapshotGenerator):
    @contextmanager
    def decision_scope(self, *, session_id=None):
        self.sessions.append(session_id)
        self._active_decision_scope = SimpleNamespace(pending_stats=None)
        try:
            yield
            if self._active_decision_scope.pending_stats is not None:
                self._active_decision_scope.pending_stats["session_cache_commit_status"] = "committed"
        finally:
            self._active_decision_scope = None

    def generate(self, *args, **kwargs):
        scope = self._active_decision_scope
        if scope.pending_stats is not None:
            scope.pending_stats["session_cache_commit_status"] = "discarded_by_regeneration"
        result = super().generate(*args, **kwargs)
        result.stats["session_cache_commit_status"] = "pending"
        scope.pending_stats = result.stats
        return result

    def capture_exact_state(self):
        snapshot = super().capture_exact_state()
        self.scope_saved = copy.deepcopy(self._active_decision_scope.pending_stats)
        return snapshot

    def restore_exact_state(self, snapshot):
        restored = super().restore_exact_state(snapshot)
        self._active_decision_scope.pending_stats = copy.deepcopy(self.scope_saved)
        return restored


def test_restored_pending_stats_alias_held_result_and_cumulative_attempt_journal(tmp_path):
    from benchmarks.memory_runtime.attempt_journal import summarize_attempt_journal
    actor, old_generator = make_actor(tmp_path)
    actor.runner.generator = ScopedSnapshotGenerator(
        old_generator.journal_path, old_generator.outputs, stats=old_generator.stats)
    state = actor.hold(request())
    snapshot = actor.capture()
    actor.submit_held([])
    actor.restore(snapshot)
    actor.submit_held([])
    assert actor.last_record["generation_trace"][0]["generation"]["stats"]["session_cache_commit_status"] == "committed"
    action = next(row["candidate_ids"] for row in state["allowed_actions"] if len(row["candidate_ids"]) == 1)
    for _ in range(2):
        actor.restore(snapshot)
        actor.submit_held(action)
        trace = actor.last_record["generation_trace"]
        assert trace[0]["generation"]["stats"]["session_cache_commit_status"] == "discarded_by_regeneration"
        assert trace[1]["generation"]["stats"]["session_cache_commit_status"] == "committed"
    journal = summarize_attempt_journal(old_generator.journal_path)
    assert journal["completed"] == 3
    assert actor.total_generation_calls == 3
    assert actor.runner.generation_calls == 2
    actor.close()


def test_driver_rng_touches_only_configured_device(monkeypatch):
    class BytesState:
        def clone(self):
            return self

    class Backend:
        def __init__(self):
            self.calls = []

        def is_initialized(self):
            return True

        def get_rng_state(self, device):
            self.calls.append(("get", device))
            return BytesState()

        def set_rng_state(self, state, device):
            self.calls.append(("set", device))

        def get_rng_state_all(self):
            raise AssertionError("Do not enumerate unrelated devices")

        def set_rng_state_all(self, states):
            raise AssertionError("Do not restore unrelated devices")

    npu, cuda = Backend(), Backend()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        npu=npu, cuda=cuda, get_rng_state=BytesState, set_rng_state=lambda state: None))
    state = _local_rng(["npu:6"])
    _restore_local_rng(state)
    assert npu.calls == [("get", "npu:6"), ("set", "npu:6")]
    assert cuda.calls == []
