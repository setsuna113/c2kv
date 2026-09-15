"""Check review evidence visibility and the non-executable verdict boundary."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.commit_review import build_review_input, parse_review, prepare_review, TOOL_NAME


def sample():
    messages = [
        {"role": "user", "content": "Inspect account."},
        {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "get", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": '{"secret_old_value": 91}'},
        {"role": "user", "content": "Set level to 2."},
        {"role": "assistant", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "get", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"level": "Medium"}'},
    ]
    metadata = {"task_id": "review-test", "selected_source_indices": [3, 4, 5]}
    return build_review_input(messages, metadata, [], {"content": "Already 2.", "tool_calls": None})


def response(context, **changes):
    value = {"verdict": "revise", "issue": "unsupported_completion",
             "evidence_source_ids": [context["current_goal"]["source_id"]], "reason": "The observation does not establish the requested value."}
    value.update(changes)
    return {"tool_calls": [{"function": {"name": TOOL_NAME, "arguments": json.dumps(value)}}]}


def test_only_complete_raw_visible_results_enter_review():
    context = sample()
    assert len(context["observed_tool_events"]) == 1
    assert "secret_old_value" not in json.dumps(context)
    assert context["observed_tool_events"][0]["results"][0]["content"] == '{"level": "Medium"}'


def test_accept_revise_and_unknown_are_distinct_valid_results():
    context = sample()
    assert parse_review(response(context), context).verdict == "revise"
    assert parse_review(response(context, verdict="accept", issue="none"), context).verdict == "accept"
    assert parse_review(response(context, verdict="insufficient_evidence", issue="insufficient_evidence"), context).verdict == "insufficient_evidence"


def test_unseen_sources_and_application_calls_are_not_verdicts():
    context = sample()
    assert parse_review(response(context, evidence_source_ids=["review-test:m1"]), context).status == "invalid_review"
    draft = response(context)
    draft["tool_calls"][0]["function"]["name"] = "set_level"
    assert parse_review(draft, context).status == "invalid_review"
    assert parse_review({"content": "accept"}, context).status == "invalid_review"


def test_caps_do_not_silently_drop_observations():
    context = sample()
    below = prepare_review(context, lambda messages, tools: 100, prompt_token_cap=99)
    equal = prepare_review(context, lambda messages, tools: 100, prompt_token_cap=100)
    assert below["status"] == "review_prompt_cap" and equal["status"] == "ready"
    assert below["messages"] == equal["messages"]
    assert parse_review(response(context, verdict="accept", issue="unsupported_completion"), context).status == "invalid_review"
