"""AgentFold's actor-directed range folding, adapted to benchmark tools.

Reference: Alibaba-NLP/DeepResearch/WebAgent/AgentFold/infer.py and
arXiv:2510.24699, section 3.3. The actor is the experiment's Qwen3-4B;
this implements the inference protocol, not the authors' trained checkpoint.
No summarizer, original-history replay, or hidden retry is used.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass, field


INSTRUCTION = """Manage your previous steps while choosing the next action.
When Previous Steps is not EMPTY, emit exactly one <compress> JSON block in
your response content before the action: {"compress_range": [start, end],
"compress_text": "replacement summary"}. End must be the latest step ID.
Start must be the start of an existing step or compressed block. Fold only
the latest step for granular condensation, or merge a suffix of existing
summaries and the latest step for deep consolidation. Preserve useful exact
identifiers, constraints and observations. Choose the range yourself.
Then perform the benchmark's normal action (native tool call or Python code)
or answer normally. The compress block is internal and is not a tool action.
Do not request or assume access to discarded original steps.
A final answer without a tool action does not require a compress block.
"""


@dataclass
class State:
    source: list = field(default_factory=list)
    systems: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    next_step: int = 0
    pending_action: dict | None = None
    completed_response: dict | None = None
    request_source: list | None = None


_STATES: dict[str, State] = {}


def reset_state():
    _STATES.clear()


def state_for(session):
    return _STATES.setdefault(session, State())


def _native_tool_call_ids(message):
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("AgentFold native tool_calls must be a list")  # noqa: TRY004
    identifiers = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("AgentFold native tool calls must be objects")  # noqa: TRY004
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("AgentFold requires every native tool call ID")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("AgentFold requires unique native tool call IDs")
    return identifiers


def prepare(payload, state, *, code_actions=False):
    messages = copy.deepcopy(payload.get("messages") or [])
    systems = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]
    if state.source and body[:len(state.source)] != state.source:
        raise ValueError("AgentFold requires append-only benchmark events")
    if state.systems and systems != state.systems:
        raise ValueError("AgentFold system prefix changed within an episode")
    if state.request_source == body and state.completed_response is not None:
        return None, {"replayed_response": True}
    delta = body[len(state.source):]
    # Validate the full append before committing any state. A malformed fold,
    # partial tool batch, or later stray message must leave the previous turn
    # retryable without a hidden partial advance.
    steps = copy.deepcopy(state.steps)
    goals = copy.deepcopy(state.goals)
    next_step = state.next_step
    pending_action = copy.deepcopy(state.pending_action)
    if pending_action is not None:
        # Only an actually returned action may become a completed interaction.
        if not delta or delta[0].get("role") != "assistant":
            raise ValueError("AgentFold expected the committed assistant action")
        expected = pending_action
        actual = delta[0]
        if (actual.get("content") or "") != (expected.get("content") or "") or (
                actual.get("tool_calls") or []) != (expected.get("tool_calls") or []):
            raise ValueError("AgentFold benchmark changed the committed action")
        feedback = []
        pos = 1
        if code_actions:
            if pos < len(delta) and delta[pos].get("role") == "user":
                feedback.append(delta[pos]); pos += 1
        else:
            while pos < len(delta) and delta[pos].get("role") == "tool":
                feedback.append(delta[pos]); pos += 1
            expected_ids = _native_tool_call_ids(expected)
            feedback_ids = [m.get("tool_call_id") for m in feedback]
            if any(not isinstance(value, str) or not value.strip()
                   for value in feedback_ids):
                raise ValueError("AgentFold requires every native tool result ID")
            if Counter(feedback_ids) != Counter(expected_ids):
                raise ValueError("AgentFold requires all results of the committed tool batch")
        is_action = bool(expected.get("tool_calls")) or code_actions
        if not feedback and is_action:
            raise ValueError("AgentFold requires executed action feedback")
        steps.append({"start": next_step, "end": next_step,
                      "content": json.dumps([expected, *feedback], ensure_ascii=False)})
        next_step += 1
        pending_action = None
        delta = delta[pos:]
    # New user goals remain exact; they are not confused with tool observations.
    for message in delta:
        if message.get("role") != "user":
            raise ValueError("AgentFold received an untracked action/observation")
        goals.append(message)
    previous = "\n\n".join(
        f"[Step {s['start']} to {s['end']}]\n{s['content']}" for s in steps) or "EMPTY"
    staged = copy.deepcopy(payload)
    staged["chat_template_kwargs"] = {**staged.get("chat_template_kwargs", {}),
                                       "enable_thinking": False}
    staged["messages"] = [*systems,
        {"role": "system", "content": INSTRUCTION},
        *goals,
        {"role": "user", "content": "### Previous Steps\n" + previous}]
    state.steps = steps
    state.goals = goals
    state.next_step = next_step
    state.pending_action = pending_action
    state.systems = systems
    state.source = body
    state.request_source = copy.deepcopy(body)
    state.completed_response = None
    return staged, {"method": "agentfold", "implementation": "actor_range_folding",
                    "policy_weights": "experiment_actor_not_author_checkpoint",
                    "step_count": next_step, "memory_blocks": len(steps)}


def finish(data, state, *, code_actions=False):
    result = copy.deepcopy(data)
    message = result["choices"][0]["message"]
    content = message.get("content") or ""
    blocks = list(re.finditer(r"<compress>\s*(.*?)\s*</compress>", content, re.DOTALL))
    next_steps = copy.deepcopy(state.steps)
    # The reference actor may finish with an answer instead of another folded
    # interaction. Native text without tool calls is the final-answer surface;
    # AppWorld content is executable code and does not imply completion.
    final_answer = (
        bool(content.strip()) and not code_actions and not message.get("tool_calls")
        and "<compress" not in content and "</compress" not in content
    )
    if state.steps and not (final_answer and not blocks):
        if len(blocks) != 1:
            raise ValueError("AgentFold actor must emit exactly one folding directive")
        directive = json.loads(blocks[0].group(1))
        if not isinstance(directive, dict):
            raise ValueError("AgentFold folding directive must be a JSON object")
        bounds = directive.get("compress_range")
        summary = directive.get("compress_text")
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(v) is not int for v in bounds) or
                not isinstance(summary, str) or not summary.strip()):
            raise ValueError("AgentFold malformed range or summary")
        start, end = bounds
        if start not in {s["start"] for s in state.steps} or end != state.steps[-1]["end"]:
            raise ValueError("AgentFold range must replace a whole suffix ending at the latest step")
        if code_actions:
            prefix = re.sub(r"<think>.*?</think>", "", content[:blocks[0].start()],
                            flags=re.DOTALL).strip()
            if prefix:
                raise ValueError("AgentFold folding directive must precede AppWorld code")
        next_steps = [s for s in state.steps if s["end"] < start] + [
            {"start": start, "end": end, "content": summary.strip()}]
    elif not state.steps and blocks:
        raise ValueError("AgentFold cannot fold an empty history")
    content = re.sub(
        r"<compress>.*?</compress>", "", content, flags=re.DOTALL
    ).strip()
    if code_actions:
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL
        ).strip()
        if message.get("tool_calls"):
            raise ValueError("AgentFold AppWorld actions must use response content")
    else:
        _native_tool_call_ids(message)
    if not content and not message.get("tool_calls"):
        raise ValueError("AgentFold actor returned no action or answer")
    # Keep native tool calls untouched and return only executable code for AppWorld.
    message["content"] = content or None
    state.steps = next_steps
    state.pending_action = copy.deepcopy(message)
    state.completed_response = copy.deepcopy(result)
    return result
