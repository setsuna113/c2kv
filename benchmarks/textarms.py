"""Text-level baseline arms: HiAgent and ACON, ported from the official repos.

Both methods manage the agent's context at the TEXT level (no KV-cache
involvement): the proxy rewrites the request's history messages before the
request reaches the serving backend, and the compression/summarization
calls themselves go to the same upstream endpoint in full mode
("self-compression", the decision recorded in the consolidation plan).

Sources (SHAs pinned in benchmarks/ops/README.md baselines note):
* HiAgent  — hiagent2024/hiagent @ cebdd8e: agentboard/agents/cme_final.py
  (subgoal protocol + segment replacement) and summarize.py
  (TrajectorySummarizer prompt, verbatim below).
* ACON     — microsoft/acon @ d63f9ae: experiments/appworld/
  configs/context_opt/gpt-4.1-mini_{history,obs}.yaml (thresholds: history
  4096 tok, obs 256 tok, preserve_last_k_turns 1, rule reset) and
  prompts/context_opt/{prompt_history_v2,prompt_user,system_prompt}.jinja
  (verbatim below).

Deliberate deltas from the originals (recorded for the report):
1. HiAgent's action dialect is plain text ("Action: ..."); our benchmarks
   are OpenAI tool-call environments, so the subgoal note gains one
   sentence saying actions are tool calls, and its AgentBoard-specific
   "check valid actions" item is dropped.  The segmentation/summary
   mechanics are unchanged.
2. ACON's compressor model is the same served endpoint (self-compression)
   instead of gpt-4.1-mini; the guideline prompts are kept verbatim.
3. Token counts are estimated as chars/4 (no tokenizer in the proxy);
   thresholds are converted with the same factor and the estimate is
   recorded in the stats so the report can quote it.

Both transforms are deterministic given the message list plus a
content-hash-keyed summary cache, so re-sends of the same prefix never
re-summarize (the ExtractCache discipline of the KV arms, applied to text).
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- compressor callback contract -------------------------------------------
# compress(system: str, user: str) -> str  — a full-mode chat call to the
# same upstream endpoint; supplied by proxy.py (which owns _post_json).

Compress = Callable[[str, str], str]

_LOCK = threading.Lock()
_SUMMARY_CACHE: Dict[str, str] = {}
# ACON rolling state per conversation: conv id -> (covered prefix digest,
# prev_summary).  "reset" rule: when the summarized prefix no longer matches
# (new conversation, or history rewritten by the benchmark), start over.
_ACON_STATE: Dict[str, Tuple[str, str]] = {}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _content_of(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _render_segment_line(message: Dict[str, Any], action_dialect) -> str:
    """One 'Action:'/'Observation:' line for an assistant/tool message,
    reusing the proxy's training-dialect rendering for tool-call turns."""
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


# ---- HiAgent -----------------------------------------------------------------

HIAGENT_SUBGOAL_NOTE = """
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal.
When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\".
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\".
Instructions:
1. You cannot output two subgoals consecutively.
2. Subgoal must be one line of text and does not print any newline characters.
3. Each subgoal must be followed by the execution of at least one valid action.
4. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id_1, subgoal_id_2, ...)\" to obtain the detailed trajectory information.
5. Actions in this environment are tool calls: emit the tool call that executes the action (the Subgoal line goes in the message text alongside the tool call).
"""

HIAGENT_SUMMARY_SYSTEM = (
    "You are a helpful assistant that summarizes agent trajectories concisely."
)

HIAGENT_SUMMARY_USER = """Subgoal: {subgoal}
Trajectory:
{trajectory}

Please provide a concise summary (1-2 sentences) of what happened during this trajectory and the outcome. Focus on the key actions taken and the final result."""


def _subgoal_of(message: Dict[str, Any]) -> Optional[str]:
    """The subgoal text if this assistant message opens a new segment."""
    if (message.get("role") or "") != "assistant":
        return None
    text = _content_of(message).strip()
    if text.startswith("Subgoal:"):
        return text.split("\n", 1)[0][len("Subgoal:"):].strip()
    return None


def hiagent_transform(messages: List[Dict[str, Any]], compress: Compress,
                      action_dialect) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """HiAgent (cme_final.py make_prompt): inject the subgoal protocol note,
    segment history by assistant 'Subgoal:' declarations, replace every
    COMPLETED segment with subgoal line + trajectory summary (summarize.py),
    keep the final (current) segment raw."""
    stats: Dict[str, Any] = {"policy": "hiagent", "n_compressor_calls": 0}
    out: List[Dict[str, Any]] = []

    # 1. subgoal note into the first system message (or a new one)
    messages = [dict(m) for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = (_content_of(m).rstrip() + "\n" + HIAGENT_SUBGOAL_NOTE)
            break
    else:
        messages.insert(0, {"role": "system", "content": HIAGENT_SUBGOAL_NOTE.strip()})

    # 2. segment by Subgoal declarations
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
    raw_chars = sum(len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or [])) for m in messages)

    # 3. replace completed segments with subgoal + cached summary
    out = list(pre)
    for k, seg in enumerate(completed, start=1):
        trajectory = "\n".join(
            _render_segment_line(m, action_dialect) for m in seg["messages"]
            if (m.get("role") or "") in ("assistant", "tool")
        )
        subgoal = seg["subgoal"] or f"(segment {k})"
        key = _sha(f"hiagent|{subgoal}|{trajectory}")
        with _LOCK:
            summary = _SUMMARY_CACHE.get(key)
        if summary is None:
            summary = compress(
                HIAGENT_SUMMARY_SYSTEM,
                HIAGENT_SUMMARY_USER.format(subgoal=subgoal, trajectory=trajectory),
            ).strip()
            with _LOCK:
                summary = _SUMMARY_CACHE.setdefault(key, summary)
            stats["n_compressor_calls"] += 1
        out.append({
            "role": "user",
            "content": f"Subgoal {k}: {subgoal}\nSummary: {summary}",
        })

    out.extend(m for seg in current for m in seg["messages"])
    out_chars = sum(len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or [])) for m in out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens("x" * raw_chars)
    stats["out_est_tokens"] = _est_tokens("x" * out_chars)
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

# gpt-4.1-mini_obs.yaml / gpt-4.1-mini_history.yaml, chars = tokens * 4
ACON_OBS_THRESHOLD_CHARS = 256 * 4
ACON_HISTORY_THRESHOLD_CHARS = 4096 * 4
ACON_PRESERVE_LAST_K_TURNS = 1


def _task_text(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return _content_of(m)
    return "(no user instruction)"


def _compress_observation(observation: str, task: str, history: str,
                          compress: Compress, cache_key: str,
                          stats: Dict[str, Any]) -> str:
    with _LOCK:
        refined = _SUMMARY_CACHE.get(cache_key)
    if refined is None:
        out = compress(ACON_SYSTEM, ACON_OBS_PROMPT.format(
            task=task, history=history[:4000], observation=observation))
        # keep only the Refined Observation section when the model follows
        # the output format; the whole output is the fallback
        marker = "# Refined Observation"
        if marker in out:
            out = out.split(marker, 1)[1].strip()
        refined = out.strip()
        with _LOCK:
            refined = _SUMMARY_CACHE.setdefault(cache_key, refined)
        stats["n_compressor_calls"] += 1
    return refined


def acon_transform(messages: List[Dict[str, Any]], compress: Compress,
                   action_dialect, conv: str
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ACON (context_opt): compress oversized tool observations in place
    (>256 tok), then when the conversation exceeds 4096 tok replace the
    covered prefix with the structured rolling summary (REASONING/COMPLETED,
    prev_summary carried across turns, last 1 turn preserved)."""
    stats: Dict[str, Any] = {"policy": "acon", "n_compressor_calls": 0,
                             "n_obs_compressed": 0}
    messages = [dict(m) for m in messages]
    task = _task_text(messages)
    raw_chars = sum(len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or [])) for m in messages)

    # 1. observation compression (before history measurement, as in ACON)
    for idx, m in enumerate(messages):
        if (m.get("role") or "") != "tool":
            continue
        obs = _content_of(m)
        if len(obs) <= ACON_OBS_THRESHOLD_CHARS:
            continue
        history = "\n".join(
            _render_segment_line(h, action_dialect)
            for h in messages[max(0, idx - 6):idx]
        )
        key = _sha(f"acon-obs|{task}|{obs}")
        m["content"] = _compress_observation(obs, task, history, compress, key, stats)
        stats["n_obs_compressed"] += 1

    # 2. history compression (threshold on everything except the system
    #    message and the preserved tail turns)
    system_msgs = [m for m in messages if m.get("role") == "system"]
    nonsystem = [m for m in messages if m.get("role") != "system"]
    k = ACON_PRESERVE_LAST_K_TURNS
    prefix, tail = nonsystem[:-k] if len(nonsystem) > k else [], nonsystem
    prefix_chars = sum(len(_content_of(m)) for m in prefix)
    stats["prefix_est_tokens"] = prefix_chars // 4

    out: List[Dict[str, Any]] = list(system_msgs)
    if prefix and prefix_chars > ACON_HISTORY_THRESHOLD_CHARS:
        prefix_digest = _sha(json.dumps(
            [{"role": m.get("role"), "content": _content_of(m)} for m in prefix],
            ensure_ascii=False, sort_keys=True))
        with _LOCK:
            covered = _ACON_STATE.get(conv)
        # rolling summary: whatever was summarized last for this conversation
        # becomes prev_summary for the next (longer) prefix; the digest is
        # stored for diagnostics only — a new turn always changes the digest
        prev_summary = covered[1] if covered else "(none)"
        state_key = _sha(f"acon-hist|{conv}|{prefix_digest}")
        with _LOCK:
            summary = _SUMMARY_CACHE.get(state_key)
        if summary is None:
            history_text = "\n".join(
                _render_segment_line(m, action_dialect) for m in prefix)
            summary = compress(ACON_SYSTEM, ACON_HISTORY_PROMPT.format(
                task=task, prev_summary=prev_summary, history=history_text)).strip()
            with _LOCK:
                summary = _SUMMARY_CACHE.setdefault(state_key, summary)
            stats["n_compressor_calls"] += 1
        with _LOCK:
            _ACON_STATE[conv] = (prefix_digest, summary)
        out.append({"role": "user", "content": f"[CONTEXT SUMMARY]\n{summary}"})
        stats["history_compressed"] = True
    else:
        out.extend(prefix)
        stats["history_compressed"] = False
    out.extend(tail)

    out_chars = sum(len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or [])) for m in out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens("x" * raw_chars)
    stats["out_est_tokens"] = _est_tokens("x" * out_chars)
    return out, stats


def reset_state() -> None:
    """Clear caches/state (fresh proxy process does this implicitly)."""
    with _LOCK:
        _SUMMARY_CACHE.clear()
        _ACON_STATE.clear()
