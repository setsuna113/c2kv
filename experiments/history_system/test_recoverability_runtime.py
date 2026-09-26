"""Exercise real packing and actor intervention with a CPU generator fixture."""
import copy

import pytest

from test_t02_runtime import make_actor, tool_schema
from benchmarks.memory_runtime.tests.test_event_native_recovery import request
from benchmarks.memory_runtime.tests.test_event_native_step import tool
from benchmarks.memory_runtime.recovery.evidence_units import build_catalog, _span_container_text
from recoverability_runtime import prepare_support, submit_support, submit_full_history, review_packet


def annotations(actor):
    prepared = actor._held["prepared"]
    unit = next(unit for unit in build_catalog(prepared._store, actor.runner.tokenizer, "tokens_1024")
                if unit.event_id in prepared.metadata["eligible_extraction"]["eligible_event_ids"]
                and not set(unit.source_indices) <= set(prepared.memory.raw_source_indices))
    span = unit.provenance[0]
    quote = _span_container_text(prepared._store, span)[span.char_start:span.char_end]
    return [{**span.to_dict(), "quote": quote, "reviewed": True,
             "support_reason": "fixture only: select source, do not inject this rationale"}]


def test_known_support_uses_frozen_append_executor_then_restores(tmp_path):
    actor, generator = make_actor(tmp_path)
    state = actor.hold(request())
    review = annotations(actor)
    selected = prepare_support(actor, review)
    assert selected["receipt"]["status"] == "admitted"
    snapshot = actor.capture()
    original = actor._held["prepared"].memory
    submit_support(actor, review, expected_candidate_ids=selected["candidate_ids"])
    assert len(generator.inputs) == 2
    assert actor.last_record["diagnostic"]["same_B0"] is True
    # The saved generation input is the actual serialized packed representation.
    prepared_input = actor.last_record["generation_trace"][-1]["prepared_input"]
    assert tuple(prepared_input["view"]["raw_event_ids"]) == original.view.raw_event_ids
    assert tuple(prepared_input["view"]["gist_event_ids"]) == original.view.gist_event_ids
    actor.restore(snapshot)
    assert actor.submit_held([]) == state["held_draft_response"]
    actor.close()


def test_full_history_continues_raw_without_changing_actor_or_limits(tmp_path):
    actor, generator = make_actor(tmp_path)
    state = actor.hold(request())
    snapshot = actor.capture()
    model_backend = actor.runner.generator
    cap, max_new = actor.runner.max_generation_calls, actor.runner.max_new_tokens
    submit_full_history(actor)
    assert actor.runner.generator is model_backend
    assert (actor.runner.max_generation_calls, actor.runner.max_new_tokens) == (cap, max_new)
    assert actor.last_record["diagnostic"]["same_B0"] is False
    trace = actor.last_record["generation_trace"][-1]
    assert trace["controller"]["full_source_coverage"] is True
    assert trace["controller"]["gist_tokens"] == 0
    assert trace["controller"]["view_mode"] == "full_original"
    assert actor.last_record["pre_generation_budget_checks"][-1]["status"] == "passed"
    generator.outputs.append(tool("continued"))
    actor.generate(request("next-decision"))
    assert actor.last_record["generation_trace"][0]["controller"]["view_mode"] == "full_original"
    actor.restore(snapshot)
    assert actor.runner.controller.set_protocol
    assert actor.submit_held([]) == state["held_draft_response"]
    actor.close()


def test_review_packet_contains_observed_archive_without_new_generation(tmp_path):
    actor, generator = make_actor(tmp_path)
    state = actor.hold(request())
    packet = review_packet(actor, state)
    assert packet["observed_archive_messages"] == [message.to_dict() for message in actor._held["prepared"]._store.messages]
    assert packet["support_annotations"] == [] and packet["prefix_validity"] is None
    assert len(generator.inputs) == 1 and packet["static_units"]
    actor.submit_held([])
    actor.close()


def test_full_reference_accepts_tool_reveal_and_rejects_tool_rewrite(tmp_path):
    actor, generator = make_actor(tmp_path)
    initial = request()
    initial["tools"] = [tool_schema("existing")]
    actor.hold(initial)
    submit_full_history(actor)
    previous_index = actor.last_record["generation_trace"][-1]["controller"]["decision_index"]
    following = copy.deepcopy(initial)
    following["decision_key"] = "turn-1/step-6"
    following["messages"].append({"role": "user", "content": "A new function is available."})
    following["tools"].append(tool_schema("set_budget_limit"))
    rewritten = copy.deepcopy(following)
    rewritten["tools"][0]["function"]["description"] = "changed existing tool"
    with pytest.raises(ValueError, match="append-only"):
        actor.runner.controller.prepare(rewritten, ratio=4, max_new_tokens=32)
    generator.outputs.append(tool("after-reveal"))
    actor.generate(following)
    assert actor.last_record["status"] == "ok"
    metadata = actor.last_record["generation_trace"][0]["controller"]
    assert metadata["decision_index"] == previous_index + 1
    assert metadata["full_source_coverage"] is True
    actor.close()


@pytest.mark.parametrize("branch", ["known_support", "full_history"])
@pytest.mark.parametrize("budget_exhausted", [False, True])
def test_failed_diagnostic_generation_counts_only_submitted_attempts(tmp_path, branch, budget_exhausted):
    actor, generator = make_actor(tmp_path)
    actor.hold(request())
    review = annotations(actor)
    selected = prepare_support(actor, review)
    generator.outputs = [RuntimeError("fixture backend failure")]
    if budget_exhausted:
        actor.runner.max_generation_calls = actor.runner.generation_calls
    with pytest.raises(RuntimeError):
        if branch == "known_support":
            submit_support(actor, review, expected_candidate_ids=selected["candidate_ids"])
        else:
            submit_full_history(actor)
    assert actor.total_generation_calls == (1 if budget_exhausted else 2)
    assert actor.last_record["status"] == "failed" and actor._held is None
    actor.close()
