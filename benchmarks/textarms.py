"""Text-level baseline arms: HiAgent and ACON, ported faithfully per the
2026-09-03 audit rulings (paper wins over repo unless the paper's numbers
came from that repo code).

Both methods manage the agent's context at the TEXT level: the proxy
rewrites the request's history before it reaches the serving backend, and
the compression/summarization calls go to the SAME served ckpt-1088
endpoint (policy and compressor are the same model — the papers' own
protocol: HiAgent §4.1 uses gpt-4-turbo for both roles, ACON §4.2 uses
gpt-4.1 for both).  Cost columns must carry the compressor calls (one extra
LLM call per turn); the standing asymmetry (C2KV is trained for its
mechanism, these baselines are training-free here) is footnote material.

Sources:
* HiAgent  — arXiv 2408.09559.  The working-memory summarization prompt
  below is the PAPER's §3.3 prompt verbatim (subgoal-met judgment, one-line
  output constraint, {example} slot).  The repo's summarize.py
  (hiagent2024/hiagent @ cebdd8e) was AI-rebuilt in 2026-04 and post-dates
  the paper's numbers — audit ruling 1: not used.
  Deliberate deltas, recorded for the report:
  - "Trajectory Retrieval not ported" (ruling 2): the note's instruction 4
    advertising retrieve(k) is REMOVED; implementing it would only produce
    hallucinated tool calls in these benchmarks.
  - The repo's gripper/blocksworld summarization-off special case
    (cme_final.py:115-120) is NOT inherited (ruling 3: that is the paper's
    w/o-OS ablation, not the method).
  - User turns are NOT action-observation pairs: they survive verbatim in
    the output AND ride along in the summarizer input as "User:" lines
    (audit: a "$300 budget" constraint was silently dropped otherwise).
  - Compressor decode per paper §4.1: max_tokens~100, stop "\n\n",
    temperature 0, top_p 1; enable_thinking off (serving stack has no
    reasoning parser).
  - Degeneration is VISIBLE: when no assistant content ever declares a
    Subgoal (e.g. pure tool-call replies), the arm is a passthrough and
    stats["degenerate"] is True.
* ACON     — arXiv 2510.00615.  Thresholds per paper §8.3 (ruling 4):
  T_obs = 1024 tok (~4096 chars), history 4096 tok, preserve the last ONE
  action/observation PAIR = k=2 messages.  The compression prompts are the
  repo's base guidelines (microsoft/acon @ d63f9ae, context_opt jinja),
  verbatim; the §3.3-optimized guidelines (ACON-U) are printed in the
  paper's appendix and can be transcribed later — until then every row is
  labeled "acon-base: ACON pipeline, base guideline, guideline optimization
  not reproduced" (ruling 5 + critic correction; NOT "acon-prompting",
  which is a different ContextualizeWeb-derived baseline).
  History/observation are evaluated SEPARATELY by the paper (ruling 6):
  arms acon_hist and acon_obs.
  Summary is embedded into the first user prompt as a <HISTORY_SUMMARY>
  block (the original memory.py:481-498 shape), not a standalone message.

Token counts are estimated as chars/4 (no tokenizer in the proxy); the
estimate is recorded in stats.

Trigger note (2026-09-05 rewrite): faithful to the upstream — the
history compression fires when est(prev_summary + messages since the last
compression) exceeds T_hist (the first compression measures the raw
history alone, tool_calls included in the estimate); between compressions
the policy SEES the summary block + every new raw turn + the preserved
action/observation pair.  An earlier port compressed every turn past the
threshold and dropped the raw recent context — that was NOT conservative,
it removed ACON's recent-context signal.  Summaries are cached per content hash; an
EMPTY compressor result is a failure and is never cached.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- compressor plumbing -----------------------------------------------------

# decode parameters per policy (papers' originals + NPU serving reality:
# Qwen3 thinking must be off, the server has no reasoning parser)


COMPRESSOR_DECODE: Dict[str, Dict[str, Any]] = {
    "hiagent": {
        "max_tokens": 100, "stop": ["\n\n"],
        "temperature": 0.0, "top_p": 1.0, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "acon": {
        "max_tokens": 2048, "temperature": 0.0, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


class TextarmCompressorError(RuntimeError):
    """The summarizer/refiner call failed (transport, abort finish, or an
    empty completion).  Must surface as an infrastructure error, never as
    an empty summary that gets cached for the rest of the conversation."""


def compressor_payload(policy: str, model: str, system_text: str,
                       user_text: str) -> Dict[str, Any]:
    """Full chat request for one compressor call (decode params above)."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    }
    payload.update(COMPRESSOR_DECODE[policy])
    return payload


# compress(payload_dict) -> str : performs the POST, validates the finish
# reason and non-empty content, raises TextarmCompressorError otherwise.
Compress = Callable[[Dict[str, Any]], str]

_LOCK = threading.Lock()
_SUMMARY_CACHE: Dict[str, str] = {}
_ACON_STATE: Dict[str, Tuple[str, str]] = {}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _est_tokens(chars: int) -> int:
    return max(1, chars // 4)


def _content_of(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _message_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(
        len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or []))
        for m in messages
    )


def _render_line(message: Dict[str, Any], action_dialect) -> str:
    role = message.get("role") or "user"
    if role == "assistant":
        if message.get("tool_calls"):
            return "Action: " + action_dialect(message)
        return "Action: " + _content_of(message)
    if role == "tool":
        return "Observation: " + _content_of(message)
    if role == "user":
        return "User: " + _content_of(message)
    return f"{role}: " + _content_of(message)


def _summarize(cache_key: str, policy: str, model: str, system_text: str,
               user_text: str, compress: Compress,
               stats: Dict[str, Any]) -> str:
    """Cached summary with failure semantics: an empty result raises and is
    never stored."""
    with _LOCK:
        cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    summary = compress(compressor_payload(policy, model, system_text,
                                          user_text)).strip()
    if not summary:
        raise TextarmCompressorError(
            f"{policy} compressor returned an empty summary "
            f"(key={cache_key[:12]})")
    with _LOCK:
        summary = _SUMMARY_CACHE.setdefault(cache_key, summary)
    stats["n_compressor_calls"] = stats.get("n_compressor_calls", 0) + 1
    return summary


# ---- HiAgent -----------------------------------------------------------------

# == train_data.py:14 / proxy.DEFAULT_SYSTEM_PROMPT (kept in sync by
# test_hiagent_default_system_prefix; proxy passes it explicitly)
TRAINING_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

HIAGENT_SUBGOAL_NOTE = """
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal.
When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\".
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\".
Instructions:
1. You cannot output two subgoals consecutively.
2. Subgoal must be one line of text and does not print any newline characters.
3. Each subgoal must be followed by the execution of at least one valid action.
4. Actions in this environment are tool calls: emit the tool call that executes the action (the Subgoal line goes in the message text alongside the tool call).
"""

# Paper §3.3, verbatim (the repo's summarize.py is a 2026 rebuild and is
# NOT the prompt the paper's numbers were produced with — audit ruling 1).
HIAGENT_SUMMARY_USER_TEMPLATE = """You are an advanced AI system tasked with summarizing and analyzing a series of action-observation pairs (trajectories) and determining whether a specific subgoal has been met.

Your goal is to create a summary that captures all essential information, decisions, and outcomes from the given trajectories, and indicate whether the subgoal has been met based on the summarized observations.

If there are no valid actions taken, you need to analyze the reason.

### Instructions:

1. Provide a summarized observation related to the subgoal in a concise manner.

2. Determine whether the subgoal has been met.

3. Do not output anything except whether summary and subgoal are met. Your output should be only one line. Do not output things like '##Summary', '##Summary and Analysis'.

{example}

##Trajectory

{formatted_trajectory}

##Subgoal:

{subgoal}

###Output:"""

HIAGENT_SUMMARY_SYSTEM = "You are a helpful assistant."
# per-task in-context example from the paper's setup; none transcribed for
# these benchmarks — the slot stays for provenance
HIAGENT_EXAMPLE = ""


def _subgoal_of(message: Dict[str, Any]) -> Optional[str]:
    if (message.get("role") or "") != "assistant":
        return None
    text = _content_of(message).strip()
    if text.startswith("Subgoal:"):
        return text.split("\n", 1)[0][len("Subgoal:"):].strip()
    return None


def hiagent_transform(messages: List[Dict[str, Any]], compress: Compress,
                      action_dialect, model: str = "c2kv-agent",
                      default_system: str = TRAINING_DEFAULT_SYSTEM_PROMPT,
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Subgoal-protocol note into the system message; segment history by
    assistant 'Subgoal:' declarations; completed segments become their
    surviving user turns + one paper-prompt summary; the current segment
    stays raw.  Passthrough (degenerate=True) when no subgoal is ever
    declared — e.g. pure tool-call replies with null content."""
    stats: Dict[str, Any] = {"policy": "hiagent", "n_compressor_calls": 0}
    messages = [dict(m) for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = (_content_of(m).rstrip() + "\n" + HIAGENT_SUBGOAL_NOTE)
            break
    else:
        # audit 2026-09-05: inserting a system with ONLY the note made
        # _assemble skip the training default, so hiagent differed from
        # full by more than the note — insert default + note instead
        messages.insert(0, {"role": "system",
                            "content": (default_system.rstrip() + "\n"
                                        + HIAGENT_SUBGOAL_NOTE.strip())})

    segments: List[Dict[str, Any]] = [{"start": 0, "subgoal": None, "messages": []}]
    for m in messages:
        subgoal = _subgoal_of(m)
        if subgoal is not None and segments[-1]["messages"]:
            segments.append({"start": 0, "subgoal": subgoal, "messages": []})
        segments[-1]["messages"].append(m)
    pre = segments[0]["messages"] if segments[0]["subgoal"] is None else []
    segs = segments if segments[0]["subgoal"] is not None else segments[1:]
    completed, current = segs[:-1], segs[-1:]

    stats["n_segments"] = len(segs)
    stats["n_summarized"] = len(completed)
    stats["degenerate"] = bool(
        len(segs) == 0 and any(m.get("role") == "assistant" for m in messages))
    raw_chars = _message_chars(messages)

    out: List[Dict[str, Any]] = list(pre)
    for k, seg in enumerate(completed, start=1):
        # user turns are instructions, not trajectory: survive verbatim
        for m in seg["messages"]:
            if m.get("role") == "user":
                out.append(dict(m))
        trajectory = "\n".join(_render_line(m, action_dialect)
                               for m in seg["messages"])
        subgoal = seg["subgoal"] or f"(segment {k})"
        summary = _summarize(
            _sha(f"hiagent|{subgoal}|{trajectory}"),
            "hiagent", model, HIAGENT_SUMMARY_SYSTEM,
            HIAGENT_SUMMARY_USER_TEMPLATE.format(
                example=HIAGENT_EXAMPLE,
                formatted_trajectory=trajectory, subgoal=subgoal),
            compress, stats)
        out.append({"role": "user",
                    "content": f"Subgoal {k}: {subgoal}\nSummary: {summary}"})
    out.extend(m for seg in current for m in seg["messages"])

    out_chars = _message_chars(out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens(raw_chars)
    stats["out_est_tokens"] = _est_tokens(out_chars)
    return out, stats


# ---- ACON --------------------------------------------------------------------

ACON_SYSTEM = (
    "You are an agent tasked with extracting and refining a concise and "
    "optimized version of the context based on the user instruction and "
    "other provided information."
)

ACON_HISTORY_PROMPT = """You are maintaining a structured context-aware summary for a productivity agent. You will be given the user instruction for the agent, a list of interactions corresponding to actions taken by the agent, and the most recent previous summary if one exists. Produce the following:

### REASONING
Summarize key progress, decisions made, important observed outcomes, and rationale behind actions taken so far. Include how earlier steps influenced later ones and why certain data is retained in the summary.

### COMPLETED
List completed subtasks or successful outcomes, with brief results if applicable.

---

## [Information Source]

### USER INSTRUCTION

{task}

## [PREVIOUS SUMMARY] (if any)

{prev_summary}

## [HISTORY OF INTERACTIONS]

{history}

---

## PRIORITIZE

1. Keep all sections relevant and concise.
2. Use reusable structured formats when summarizing artifacts.
3. Ensure agent can resume task with no loss of information.
4. Include key info from errors or failed attempts to prevent repeated mistakes.
5. Preserve all essential artifacts and data needed to complete the task.

---

### [Output Format]

Do **not** include the input or any additional explanation. Only return the formatted summary."""

ACON_OBS_PROMPT = """Your task is to generate a "Reasoning" and a "Refined Observation" based on the inputs below.

In the "Reasoning", analyze the user instruction and history to identify what information from the current observation is necessary to complete the remaining steps.
Think about what parts can be summarized or transformed to reduce length, while ensuring that future actions can still be executed based on the refined observation alone.

In the "Refined Observation", include only the information that is minimal but sufficient for the next steps.

[Information source]
# User Instruction
{task}

# History of interactions
{history}

# Observation at the current time step
{observation}

[Output format]
# Reasoning
... your reasoning for what matters and how to optimize it ...
# Refined Observation
... reduced and actionable observation ..."""

# paper §8.3 (ruling 4): T_obs = 1024 tok; history = 4096 tok; chars = tok*4
ACON_OBS_THRESHOLD_CHARS = 1024 * 4
ACON_HISTORY_THRESHOLD_CHARS = 4096 * 4
# one preserved action/observation PAIR = 2 messages
ACON_PRESERVE_LAST_K_MESSAGES = 2
# the original feeds the full history to the obs compressor; we window it
# (documented approximation)
ACON_OBS_HISTORY_MESSAGES = 12
ACON_OBS_HISTORY_CHARS = 8000


def _task_text(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return _content_of(m)
    return "(no user instruction)"


def _acompress_obs(observation: str, task: str, history: str, compress,
                   cache_key: str, stats: Dict[str, Any], model: str) -> str:
    with _LOCK:
        refined = _SUMMARY_CACHE.get(cache_key)
    if refined is not None:
        return refined
    out = compress(compressor_payload(
        "acon", model, ACON_SYSTEM,
        ACON_OBS_PROMPT.format(task=task, history=history,
                               observation=observation)))
    marker = "# Refined Observation"
    if marker in out:
        out = out.split(marker, 1)[1].strip()
    refined = out.strip()
    if not refined:
        raise TextarmCompressorError(
            f"acon obs compressor returned empty (key={cache_key[:12]})")
    with _LOCK:
        refined = _SUMMARY_CACHE.setdefault(cache_key, refined)
    stats["n_compressor_calls"] = stats.get("n_compressor_calls", 0) + 1
    return refined


def acon_transform(messages: List[Dict[str, Any]], compress: Compress,
                   action_dialect, conv: str, mode: str = "both",
                   model: str = "c2kv-agent"
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """acon-base (ruling 5 label).  mode: 'obs' refines oversized tool
    observations in place; 'hist' replaces the covered prefix with the
    rolling structured summary embedded in the first user prompt
    (<HISTORY_SUMMARY> block, the original memory.py:481-498 shape);
    'both' applies obs first then hist, as the original pipeline does."""
    assert mode in ("obs", "hist", "both")
    stats: Dict[str, Any] = {"policy": "acon", "mode": mode,
                             "n_compressor_calls": 0, "n_obs_compressed": 0}
    messages = [dict(m) for m in messages]
    task = _task_text(messages)
    raw_chars = _message_chars(messages)

    if mode in ("obs", "both"):
        for idx, m in enumerate(messages):
            if (m.get("role") or "") != "tool":
                continue
            obs = _content_of(m)
            if len(obs) <= ACON_OBS_THRESHOLD_CHARS:
                continue
            window = messages[max(0, idx - ACON_OBS_HISTORY_MESSAGES):idx]
            history = ("\n".join(_render_line(h, action_dialect) for h in window)
                       )[:ACON_OBS_HISTORY_CHARS]
            key = _sha(f"acon-obs|{task}|{obs}")
            m["content"] = _acompress_obs(obs, task, history, compress, key,
                                          stats, model)
            stats["n_obs_compressed"] += 1

    out: List[Dict[str, Any]] = list(messages)
    stats["history_compressed"] = False
    if mode in ("hist", "both"):
        system_msgs = [m for m in out if m.get("role") == "system"]
        nonsystem = [m for m in out if m.get("role") != "system"]
        k = ACON_PRESERVE_LAST_K_MESSAGES
        prefix, tail = ((nonsystem[:-k], nonsystem[-k:])
                        if len(nonsystem) > k else ([], nonsystem))
        # token estimates INCLUDE tool_calls (audit 2026-09-05: counting
        # content only left assistant tool-call turns — where tau2/BFCL
        # keep every action — invisible to the trigger; the arm then never
        # fired and silently became a full arm)
        prefix_chars = _message_chars(prefix)
        stats["prefix_est_tokens"] = prefix_chars // 4

        # FAITHFUL upstream trigger (audit 2026-09-05): tiktoken(
        # prev_summary + messages since the last compression) > T_hist.
        # The FIRST compression measures the raw history alone.  Between
        # compressions the policy SEES the summary block + EVERY new raw
        # turn + the preserved pair — the raw recent context is never
        # dropped turn-by-turn (the old port did exactly that, wrongly).
        def _nonsystem_digest(until: int) -> str:
            return _sha(json.dumps(
                [{"role": m.get("role"), "content": _content_of(m)}
                 for m in nonsystem[:until]],
                ensure_ascii=False, sort_keys=True))

        with _LOCK:
            state = _ACON_STATE.get(conv)
        if (state is None or state[0] > len(prefix)
                or state[1] != _nonsystem_digest(state[0])):
            # fresh conversation, benchmark rewrote history, or a
            # conv-id collision (different content under the same id)
            covered_until, prev_summary = 0, ""
        else:
            covered_until, prev_summary = state[0], state[2]

        new_msgs = nonsystem[covered_until:len(prefix)]
        new_chars = _message_chars(new_msgs)
        trigger_chars = new_chars if not prev_summary else (
            len(prev_summary) + new_chars)

        if prefix and trigger_chars > ACON_HISTORY_THRESHOLD_CHARS:
            history_text = "\n".join(_render_line(m, action_dialect)
                                     for m in new_msgs)
            user_text = ACON_HISTORY_PROMPT.format(
                task=task, prev_summary=prev_summary or "(none)",
                history=history_text)
            # content-addressed cache key: the positional key
            # (conv|covered|len) let a DIFFERENT task's same-length
            # segment hit the wrong cached summary under a shared id
            summary = _summarize(
                _sha(f"acon-hist|{user_text}"),
                "acon", model, ACON_SYSTEM, user_text,
                compress, stats)
            with _LOCK:
                _ACON_STATE[conv] = (len(prefix), _nonsystem_digest(len(prefix)),
                                     summary)
            stats["new_raw_messages_folded"] = len(new_msgs)
            block = (f"\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>")
            # compressed view: keep system messages and the first user
            # prompt (the task instruction) carrying the block — the
            # memory.py:481-498 shape — plus the preserved tail; the
            # folded prefix messages are dropped (that IS the compression)
            placed = False
            rebuilt = list(system_msgs)
            for m in prefix:
                if (not placed and m.get("role") == "user"
                        and not m.get("tool_calls")):
                    rebuilt.append({"role": "user",
                                    "content": _content_of(m) + block})
                    placed = True
            for m in tail:
                rebuilt.append(dict(m))
            if not placed:  # no user instruction anywhere (should not happen)
                rebuilt.insert(len(system_msgs),
                               {"role": "user", "content": block.strip()})
            out = rebuilt
            stats["history_compressed"] = True
        elif prev_summary:
            # NOT triggered but a summary exists: the policy sees the
            # previous summary block + EVERY new raw turn since the last
            # compression + the preserved pair (upstream semantics — the
            # raw recent context is only folded at the next compression)
            block = (f"\n<HISTORY_SUMMARY>\n{prev_summary}\n</HISTORY_SUMMARY>")
            rebuilt = list(system_msgs)
            placed = False
            for m in nonsystem[:covered_until]:
                if (not placed and m.get("role") == "user"
                        and not m.get("tool_calls")):
                    rebuilt.append({"role": "user",
                                    "content": _content_of(m) + block})
                    placed = True
            if not placed:
                rebuilt.append({"role": "user", "content": block.strip()})
            rebuilt.extend(dict(m) for m in new_msgs)
            rebuilt.extend(dict(m) for m in tail)
            out = rebuilt
            stats["history_compressed"] = False
            stats["new_raw_messages_visible"] = len(new_msgs)
        # else: never compressed and under threshold — plain passthrough

    out_chars = _message_chars(out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens(raw_chars)
    stats["out_est_tokens"] = _est_tokens(out_chars)
    return out, stats


def reset_state() -> None:
    """Test isolation hook (also used by proxy startup)."""
    with _LOCK:
        _SUMMARY_CACHE.clear()
        _ACON_STATE.clear()
