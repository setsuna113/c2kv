"""True-joint C2KV training data from the medium-phase multi-source mixture.

``train_data_joint.AgentLLMTracesJointSource`` covers only agent-llm-traces.
The G-medium experiment (docs/0820_g_joint_progress.md) adds three sources and
mixes all four families by *compressible source tokens* (never row counts):

- ``traces``  — agent-llm-traces (existing source, untouched);
- ``toucan``  — Toucan SFT, ``subset_name == "multi-turn"`` rows only;
- ``openswe`` — NVIDIA Open-SWE-Traces, ``resolved == 1`` trajectories only;
- ``qa``      — official QA document corpora (HotpotQA / 2Wiki / LongMagpie):
  long-context document compression supervision with NO tool side.

Every source emits the same ``JointExample`` contract and reuses the traces
path's own rendering functions, so the answer surface
(``Action:\\n<tool_call>\\n{"name":...,"arguments":...}\\n</tool_call>``) and the
history document style (``Previous turn\\n[User query]...``) are
byte-compatible across families — the mixture varies WHICH examples are
compressed, never HOW.  Answer rendering goes through
``train_data_multiturn._render_agent_output_messages`` for both agentic
sources (no re-implementation): the normalized message shapes below are
exactly the ones the traces path already renders, so the emitted payloads keep
the same key order and minified JSON.

On-disk layouts (authoritative samples: ``testdata_gjoint/fixtures.json``,
real dumped rows):

- Toucan parquet (``SFT/*.parquet``): ``messages``/``tools`` are JSON-ENCODED
  STRING columns; ``tool_call`` message content is a python-dict literal whose
  ``arguments`` value is itself a JSON string (``ast.literal_eval`` then
  ``json.loads``).  The ``target_tools`` column is fuzzy (its names do not
  match schema names) and is ignored; the target tool comes from the first
  tool call of the assistant turn.  Conversations start with ``user`` (no
  system message) — ``system_prompt`` is the default.
- Open-SWE parquet (``data/*/*.parquet``): ``trajectory`` is a native list of
  role-tagged dicts; assistant ``tool_calls`` are standard OpenAI shape with
  ``arguments`` as a JSON string; ``tools`` is a list of JSON strings.
- 2Wiki ``context`` is ``[[title, [sentence, ...]], ...]`` (a native list, or
  a JSON string of one; an unparseable value falls back to one raw-text
  document instead of dropping the row).
- LongMagpie user content is one long document with the question appended as
  the trailing sentence(s) ending in ``?`` — see ``split_longmagpie_question``.
  Its qid is ``qa:longmagpie:<shard_stem>:<row_in_shard>`` (file-local row
  number, skipped rows counted): stable under shard-set changes (P1-6) and
  exactly equal to the dedup removal ids (P1-5).

Memory / determinism contract:

- Sources are LAZY: ``__iter__`` streams row -> example conversions and
  pyarrow is imported inside the IO methods only (the module itself imports
  cleanly without pyarrow), so the mixture planner
  (``agent/build_joint_medium_plan.py``) can stop a scan once a family quota
  is filled without materializing the whole dataset.  ``__len__``
  materializes and caches on first use.
- ``keep_qids`` (a frozen order file's qid list) prefilters examples during
  the stream so the trainer never materializes unlisted examples.  When it is
  set, ``max_samples_per_session`` sub-sampling is DISABLED: the order file
  fixes the exact example set and dropping a listed qid would hard-error the
  trainer's ``_apply_example_order_file``.
- These sources are train-split only (``split != "train"`` raises): held-out
  eval keeps coming from the traces source alone.
- Seeding mirrors the traces source: per-example tool rendering uses
  ``random.Random(f"{split_seed}:{session_id}:{span}:tools")`` with the qid
  suffix (``u<i>`` / ``a<k>``) as the span component, and per-conversation
  sub-sampling uses one ``random.Random(split_seed)`` advanced in iteration
  order.  Same inputs -> same outputs.

De-leak invariant: no produced ``system_prompt`` contains tool schemas (Toucan
and QA use the bare default prompt; Open-SWE uses the trajectory's own system
message, exactly what the traces path's ``_agent_system_prompt`` does — the
leak being prevented is schemas injected into the uncompressed prefix, which
no path here does), and history documents are always chronological.
"""

from __future__ import annotations

import ast
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

from .train_data import DEFAULT_SYSTEM_PROMPT
from .train_data_joint import (
    FAMILY_PREFIXES,
    JointExample,
    _first_tool_call_name,
    _render_tool_documents,
    _select_tools,
    _shuffled_system_tools,
    _strip_think_blocks,
    qid_source_family,
)
from .train_data_multiturn import (
    Message,
    _agent_history_turn_docs,
    _agent_system_prompt,
    _json_loads,
    _normal_agent_message,
    _render_agent_output_messages,
)

logger = logging.getLogger(__name__)


TOUCAN_SUBSET = "toucan:multi-turn"

# FAMILY_PREFIXES / qid_source_family are defined in train_data_joint (so the
# dataset builder can attribute skip counters per family without an import
# cycle) and re-exported here: existing import sites
# (agent/build_joint_medium_plan.py, agent/train_joint_next_action_c2kv.py)
# keep working.

__all__ = [
    "FAMILY_PREFIXES",
    "qid_source_family",
    "TOUCAN_SUBSET",
    "ToucanJointSource",
    "OpenSWEJointSource",
    "QADocsJointSource",
    "toucan_row_to_examples",
    "openswe_row_to_examples",
    "hotpotqa_row_to_example",
    "wiki2_row_to_example",
    "longmagpie_row_to_example",
    "split_longmagpie_question",
]


def _validate_split(split: str) -> None:
    if split != "train":
        raise ValueError(
            "Multi-source joint sources are train-split only (held-out eval comes "
            f"from the traces source alone); got split={split!r}"
        )


# ---------------------------------------------------------------------------
# Tool-call parsing (raw on-disk shapes -> the flat {"name", "arguments"}
# shape the traces path's ``_render_agent_tool_calls`` renders byte-identically
# to its own gen_ai output: ``arguments`` always a JSON value, minified).
# ---------------------------------------------------------------------------


def _parse_toucan_tool_call(content: Any) -> Optional[Dict[str, Any]]:
    """Parse one Toucan ``tool_call`` message content.

    On disk the content is a python-dict literal string such as
    ``"{'name': 'x-y', 'arguments': '{\\"a\\": 1}'}"`` — ``ast.literal_eval``
    for the outer literal, ``json.loads`` for the inner ``arguments`` string.
    An unparseable ``arguments`` string is kept verbatim (the renderer
    tolerates non-dict values), matching what the traces path does with a
    string ``arguments`` field.
    """
    parsed = content if isinstance(content, dict) else None
    if parsed is None:
        if not isinstance(content, str):
            return None
        try:
            parsed = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name") or parsed.get("tool_name") or parsed.get("function_name") or ""
    if not name:
        return None
    arguments = _json_loads(parsed.get("arguments"), parsed.get("arguments"))
    if arguments is None or arguments == "":
        arguments = {}
    return {"name": str(name), "arguments": arguments}


def _parse_openswe_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize Open-SWE assistant ``tool_calls`` to the flat shape.

    On disk: standard OpenAI ``[{"id", "type": "function", "function":
    {"name", "arguments": "<json string>"}}]``.  ``arguments`` is parsed so
    the rendered payload matches the traces path byte-for-byte.  Calls
    without a name are dropped.
    """
    calls = _json_loads(tool_calls, [])
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return []
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or ""
        if not name:
            continue
        raw_arguments = function.get("arguments")
        if raw_arguments is None:
            raw_arguments = call.get("arguments")
        arguments = _json_loads(raw_arguments, raw_arguments)
        if arguments is None or arguments == "":
            arguments = {}
        normalized.append({"name": str(name), "arguments": arguments})
    return normalized


# ---------------------------------------------------------------------------
# History normalization -> the traces path's exact rendering pipeline
# (``_normal_agent_message`` then ``_agent_history_turn_docs``), so history
# documents are byte-identical in style across all four mixture families:
# assistant tool calls render as ``Action:\n<tool_call>...`` blocks and tool
# results are folded into ``user`` messages (``_normal_agent_message`` maps
# role ``tool`` -> ``user``, like tool results in traces sessions).
# ---------------------------------------------------------------------------


def _toucan_history_messages(messages: Sequence[Any]) -> List[Message]:
    """Map Toucan roles onto the gen_ai message shapes the traces path renders.

    ``tool_call`` -> assistant message carrying normalized ``tool_calls``;
    ``tool_response`` -> ``tool`` role; everything else passes through with
    role/content only.  Blank assistant messages are skipped — they would
    otherwise leave an empty ``[Assistant output]`` section in the turn doc.
    """
    normalized: List[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "user"
        content = message.get("content")
        if role == "tool_call":
            call = _parse_toucan_tool_call(content)
            if call is not None:
                normalized.append({"role": "assistant", "tool_calls": [call]})
        elif role == "tool_response":
            normalized.append({"role": "tool", "content": content})
        elif role == "assistant" and not str(content or "").strip():
            continue
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def _openswe_history_messages(messages: Sequence[Any]) -> List[Message]:
    """Map Open-SWE trajectory messages onto the same gen_ai shapes.

    Assistant messages keep their text content and get normalized
    ``tool_calls``; ``reasoning_content``/``think`` are dropped — the traces
    path's ``_normal_agent_message`` does not render reasoning into history
    either (and neither does the multiturn path's
    ``_render_openswe_history_message``).  Thinking-only assistant messages
    (no content, no tool calls) are skipped: they would render as an empty
    ``[Assistant output]`` section.  ``tool`` results pass through as ``tool``
    role; ``system`` is dropped downstream.
    """
    normalized: List[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "user"
        if role == "assistant":
            content = message.get("content") or ""
            tool_calls = _parse_openswe_tool_calls(message.get("tool_calls"))
            if not str(content).strip() and not tool_calls:
                continue
            normalized.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
        else:
            normalized.append({"role": role, "content": message.get("content")})
    return normalized


def _history_documents(messages: Sequence[Message]) -> List[str]:
    """Render normalized history messages to chronological turn documents."""
    normalized = [
        item
        for item in (_normal_agent_message(message) for message in messages)
        if item is not None and item.get("role") != "system"
    ]
    return [str(doc.get("content") or "") for doc in _agent_history_turn_docs(normalized)]


def _render_example_tool_documents(
    tools: Sequence[Dict[str, Any]],
    target_tool: Optional[str],
    rng: random.Random,
    *,
    canonical_format_prob: float = 0.7,
    minified_json_prob: float = 0.2,
    shuffle_tools: bool = True,
    truncate_description_chars: int = 600,
    max_tools_per_sample: int = 32,
    same_namespace_negative_tools: int = 8,
    random_negative_tools: int = 24,
) -> Tuple[List[str], Optional[int], List[Dict[str, Any]]]:
    """Per-example target-inclusive tool pool + rendering (traces-source policy).

    Returns ``(docs, target_index, selected_tools)``; ``selected_tools`` is the
    post-selection pre-render tool dict list, carried on the JointExample so
    the ``tools_in_system`` arm can render the same pool RAW in the system
    prefix.
    """
    selected_tools = _select_tools(
        tools,
        target_tool,
        rng,
        max_tools_per_sample=max_tools_per_sample,
        same_namespace_negative_tools=same_namespace_negative_tools,
        random_negative_tools=random_negative_tools,
    )
    docs, target_index = _render_tool_documents(
        selected_tools,
        rng,
        canonical_format_prob=canonical_format_prob,
        minified_json_prob=minified_json_prob,
        shuffle_tools=shuffle_tools,
        truncate_description_chars=truncate_description_chars,
        target_tool=target_tool,
    )
    return docs, target_index, list(selected_tools)


# ---------------------------------------------------------------------------
# Conversion cores: parsed row dict -> examples.  Pure (no IO), so the unit
# tests drive them straight from testdata_gjoint/fixtures.json without
# pyarrow.
# ---------------------------------------------------------------------------


def toucan_row_to_examples(
    row: Dict[str, Any],
    *,
    split_seed: int = 42,
    require_tool_call: bool = True,
    canonical_format_prob: float = 0.7,
    minified_json_prob: float = 0.2,
    shuffle_tools: bool = True,
    truncate_description_chars: int = 600,
    max_tools_per_sample: int = 32,
    same_namespace_negative_tools: int = 8,
    random_negative_tools: int = 24,
) -> List[JointExample]:
    """Conversion core: one parsed Toucan row -> joint examples.

    Example points: every ``user`` message at index ``i > 0`` whose following
    assistant turn — the next ``assistant`` message plus the run of
    ``tool_call`` messages after it — contains >= 1 tool call.
    ``require_tool_call=False`` additionally keeps turns without a tool call
    (pure-text assistant replies).  ``i == 0`` never qualifies: the joint task
    needs non-empty history.
    """
    messages = _json_loads(row.get("messages"), [])
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list) or not messages:
        return []
    tools = [tool for tool in _json_loads(row.get("tools"), []) if isinstance(tool, dict)]
    if not tools:
        # The joint task needs both document types; skip tool-less rows like
        # the traces source skips tool-less sessions.
        return []
    uuid = str(row.get("uuid") or "").strip()
    if not uuid:
        return []
    session_id = f"toucan:{uuid}"
    examples: List[JointExample] = []
    for index, message in enumerate(messages):
        if index == 0 or not isinstance(message, dict) or message.get("role") != "user":
            continue
        current_content = str(message.get("content") or "").strip()
        if not current_content:
            continue
        follow = index + 1
        if follow >= len(messages) or not isinstance(messages[follow], dict):
            continue
        if messages[follow].get("role") != "assistant":
            continue
        turn = [messages[follow]]
        cursor = follow + 1
        while (
            cursor < len(messages)
            and isinstance(messages[cursor], dict)
            and messages[cursor].get("role") == "tool_call"
        ):
            turn.append(messages[cursor])
            cursor += 1
        output_messages: List[Message] = []
        for turn_message in turn:
            if turn_message.get("role") == "tool_call":
                call = _parse_toucan_tool_call(turn_message.get("content"))
                if call is not None:
                    output_messages.append({"role": "assistant", "tool_calls": [call]})
            else:
                output_messages.append({"role": "assistant", "content": turn_message.get("content")})
        answer, has_tool_call = _render_agent_output_messages(output_messages, None)
        if not answer or (require_tool_call and not has_tool_call):
            continue
        answer = _strip_think_blocks(answer)  # same residue policy as the traces source
        if not answer:
            continue
        history_documents = _history_documents(_toucan_history_messages(messages[:index]))
        if not history_documents:
            continue
        target_tool = _first_tool_call_name(output_messages)
        rng = random.Random(f"{split_seed}:{session_id}:u{index}:tools")
        tool_documents, target_doc_index, selected_tools = _render_example_tool_documents(
            tools,
            target_tool,
            rng,
            canonical_format_prob=canonical_format_prob,
            minified_json_prob=minified_json_prob,
            shuffle_tools=shuffle_tools,
            truncate_description_chars=truncate_description_chars,
            max_tools_per_sample=max_tools_per_sample,
            same_namespace_negative_tools=same_namespace_negative_tools,
            random_negative_tools=random_negative_tools,
        )
        examples.append(
            JointExample(
                qid=f"{session_id}:u{index}",
                session_id=session_id,
                tool_documents=list(tool_documents),
                history_documents=history_documents,
                current_messages=[{"role": "user", "content": current_content}],
                answer=answer,
                system_prompt=DEFAULT_SYSTEM_PROMPT,  # Toucan conversations carry no system message
                subset=TOUCAN_SUBSET,
                target_tool=target_tool,
                target_tool_doc_index=target_doc_index,
                action_type="tool_call" if has_tool_call else "other",
                # See the Open-SWE site below: shuffle the tools_in_system pool
                # so the gold tool is not always element 0 (2026-09-05).
                selected_tools=_shuffled_system_tools(
                    selected_tools, split_seed, session_id, f"u{index}"
                ),
            )
        )
    return examples


def openswe_row_to_examples(
    row: Dict[str, Any],
    *,
    subset: str,
    split_seed: int = 42,
    require_tool_call: bool = True,
    canonical_format_prob: float = 0.7,
    minified_json_prob: float = 0.2,
    shuffle_tools: bool = True,
    truncate_description_chars: int = 600,
    max_tools_per_sample: int = 32,
    same_namespace_negative_tools: int = 8,
    random_negative_tools: int = 24,
) -> List[JointExample]:
    """Conversion core: one parsed Open-SWE row -> joint examples.

    Only ``resolved == 1`` trajectories qualify (the medium arm trains on
    successful problem-solving traces).  Example points: every assistant
    message at trajectory index ``k`` with non-empty ``tool_calls`` and at
    least one earlier assistant message after the first user message (so the
    history is non-empty — the trajectory's first action never qualifies).
    The current prompt is the trajectory's FIRST user message; the target is
    action ``k`` rendered by ``_render_agent_output_messages`` (reasoning kept
    as ``Thought:`` blocks, like reasoning in traces span outputs).
    ``require_tool_call`` is a no-op guard here: candidate points already
    require tool calls.
    """
    if row.get("resolved") != 1:
        return []
    trajectory = _json_loads(row.get("trajectory"), [])
    if not isinstance(trajectory, list) or not trajectory:
        return []
    raw_tools = _json_loads(row.get("tools"), [])
    if not isinstance(raw_tools, list):
        raw_tools = [raw_tools]
    tools = []
    for item in raw_tools:
        tool = _json_loads(item, item)
        if isinstance(tool, dict):
            tools.append(tool)
    if not tools:
        return []
    session_tag = str(row.get("trajectory_id") or row.get("instance_id") or "").strip()
    if not session_tag:
        return []
    session_id = f"openswe:{session_tag}"
    system_prompt = _agent_system_prompt([m for m in trajectory if isinstance(m, dict)])
    first_user_index = next(
        (
            i
            for i, m in enumerate(trajectory)
            if isinstance(m, dict) and m.get("role") == "user"
        ),
        None,
    )
    if first_user_index is None:
        return []
    current_content = str(trajectory[first_user_index].get("content") or "").strip()
    if not current_content:
        return []
    current_messages = [{"role": "user", "content": current_content}]
    examples: List[JointExample] = []
    for k in range(first_user_index + 1, len(trajectory)):
        message = trajectory[k]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = _parse_openswe_tool_calls(message.get("tool_calls"))
        if not tool_calls:
            continue
        earlier = trajectory[first_user_index + 1 : k]
        if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in earlier):
            continue  # first action of the trajectory: history would be empty
        history_documents = _history_documents(_openswe_history_messages(earlier))
        if not history_documents:
            continue
        output_message = {
            "role": "assistant",
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": tool_calls,
        }
        answer, has_tool_call = _render_agent_output_messages([output_message], None)
        if not answer or (require_tool_call and not has_tool_call):
            continue
        answer = _strip_think_blocks(answer)  # same residue policy as the traces source
        if not answer:
            continue
        target_tool = _first_tool_call_name([output_message])
        rng = random.Random(f"{split_seed}:{session_id}:a{k}:tools")
        tool_documents, target_doc_index, selected_tools = _render_example_tool_documents(
            tools,
            target_tool,
            rng,
            canonical_format_prob=canonical_format_prob,
            minified_json_prob=minified_json_prob,
            shuffle_tools=shuffle_tools,
            truncate_description_chars=truncate_description_chars,
            max_tools_per_sample=max_tools_per_sample,
            same_namespace_negative_tools=same_namespace_negative_tools,
            random_negative_tools=random_negative_tools,
        )
        examples.append(
            JointExample(
                qid=f"{session_id}:a{k}",
                session_id=session_id,
                tool_documents=list(tool_documents),
                history_documents=history_documents,
                current_messages=current_messages,
                answer=answer,
                system_prompt=system_prompt,
                subset=subset,
                target_tool=target_tool,
                target_tool_doc_index=target_doc_index,
                action_type="tool_call" if has_tool_call else "other",
                # tools_in_system renders this list verbatim into the system
                # prefix; _select_tools puts the gold tool first, so shuffle
                # on the same dedicated RNG stream the traces path uses
                # (2026-09-05: the 2cea1d1 de-oracle fix covered traces only).
                selected_tools=_shuffled_system_tools(
                    selected_tools, split_seed, session_id, f"a{k}"
                ),
            )
        )
    return examples


# ---------------------------------------------------------------------------
# QA conversion cores (HotpotQA / 2Wiki / LongMagpie).  Every row maps to
# exactly one JointExample with ``tool_documents=[]``: the QA arm supplies
# long-context document compression supervision, no tool side.  ``session_id``
# is the qid itself (one example per row).
# ---------------------------------------------------------------------------


def _qa_joint_example(
    family: str,
    row_id: str,
    documents: Sequence[Any],
    question: Any,
    answer: Any,
    gold_history_doc_indices: Optional[Sequence[int]] = None,
) -> Optional[JointExample]:
    """Build the single QA JointExample for one row.

    ``gold_history_doc_indices`` (optional) indexes the ``documents`` argument
    as PASSED; empty/whitespace documents are dropped from the example, so the
    gold indices are remapped through that filter (a gold label on a dropped
    empty document vanishes rather than pointing at the wrong doc).
    """
    doc_texts = [str(doc) for doc in documents]
    kept = [index for index, text in enumerate(doc_texts) if text.strip()]
    history_documents = [doc_texts[index] for index in kept]
    question_text = str(question or "").strip()
    answer_text = str(answer or "").strip()
    row_id = str(row_id).strip()
    if not row_id or not history_documents or not question_text or not answer_text:
        return None
    remap = {old: new for new, old in enumerate(kept)}
    gold: Optional[List[int]] = None
    if gold_history_doc_indices:
        remapped = sorted({remap[index] for index in gold_history_doc_indices if index in remap})
        gold = remapped or None
    qid = f"qa:{family}:{row_id}"
    return JointExample(
        qid=qid,
        session_id=qid,
        tool_documents=[],
        history_documents=history_documents,
        current_messages=[{"role": "user", "content": question_text}],
        answer=answer_text,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        subset=f"qa:{family}",
        target_tool=None,
        target_tool_doc_index=None,
        gold_history_doc_indices=tuple(gold) if gold else None,
    )


def _supporting_fact_titles(supporting_facts: Any) -> List[str]:
    """Gold document titles from a supporting_facts payload.

    Handles the HotpotQA-cleaned dict shape ``{"title": [...], "sent_id":
    [...]}`` and the 2Wiki/raw list-of-pairs shape ``[[title, sent_id],
    ...]``; either may be a JSON-encoded string.  Unparseable payloads yield
    no titles (the example is kept, just unlabelled).
    """
    parsed = _json_loads(supporting_facts, supporting_facts)
    titles: List[str] = []
    if isinstance(parsed, dict):
        raw_titles = parsed.get("title")
        if isinstance(raw_titles, list):
            titles = [str(title) for title in raw_titles]
    elif isinstance(parsed, list):
        for pair in parsed:
            if isinstance(pair, (list, tuple)) and pair:
                titles.append(str(pair[0]))
    return [title for title in titles if title.strip()]


def _hotpotqa_gold_doc_indices(row: Dict[str, Any], documents: Sequence[Any]) -> List[int]:
    """Gold doc indices for a cleaned HotpotQA row (supporting_facts titles).

    Cleaned documents carry the title in a ``Document N (title: T)`` prefix;
    titles may themselves contain parentheses, so matching is by the exact
    ``(title: T)`` marker substring inside the document head rather than a
    regex parse of the prefix.
    """
    titles = _supporting_fact_titles(row.get("supporting_facts"))
    if not titles:
        return []
    indices: List[int] = []
    for index, doc in enumerate(documents):
        text = str(doc)
        for title in titles:
            marker = f"(title: {title})"
            if marker in text[: 40 + len(marker)]:
                indices.append(index)
                break
    return indices


def hotpotqa_row_to_example(row: Dict[str, Any], row_index: int = 0) -> Optional[JointExample]:
    """HotpotQA row: raw ``documents`` strings stay unprefixed history docs."""
    documents = _json_loads(row.get("documents"), [])
    if not isinstance(documents, list):
        documents = [documents]
    row_id = str(row.get("_id") or row.get("id") or row_index)
    gold = _hotpotqa_gold_doc_indices(row, documents)
    return _qa_joint_example(
        "hotpotqa", row_id, documents, row.get("question"), row.get("answer"),
        gold_history_doc_indices=gold,
    )


def wiki2_row_to_example(row: Dict[str, Any], row_index: int = 0) -> Optional[JointExample]:
    """2Wiki row: ``context`` ``[[title, [sentence, ...]], ...]`` -> one doc per entry."""
    context = _json_loads(row.get("context"), row.get("context"))
    gold_titles = set(_supporting_fact_titles(row.get("supporting_facts")))
    documents: List[str] = []
    doc_titles: List[Optional[str]] = []  # parallel to documents (None = untitled)
    if isinstance(context, list):
        for entry in context:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title, sentences = entry
                sentences = _json_loads(sentences, sentences)
                if isinstance(sentences, list):
                    documents.append(str(title) + "\n" + " ".join(str(s) for s in sentences))
                else:
                    documents.append(str(title) + "\n" + str(sentences))
                doc_titles.append(str(title))
            elif entry is not None:
                documents.append(str(entry))
                doc_titles.append(None)
    elif isinstance(context, str) and context.strip():
        # Serialized-but-unparseable context (e.g. truncated exports): keep
        # the raw text as one document rather than dropping the row.  Real
        # 2Wiki parquet yields the native list and never takes this branch.
        documents = [context]
        doc_titles = [None]
    gold = [index for index, title in enumerate(doc_titles) if title is not None and title in gold_titles]
    row_id = str(row.get("_id") or row.get("id") or row_index)
    return _qa_joint_example(
        "2wiki", row_id, documents, row.get("question"), row.get("answer"),
        gold_history_doc_indices=gold,
    )


# Trailing run of question sentences: segments without sentence-ending
# punctuation, each closed by ``?`` (plus optional closing quotes/brackets).
# The boundary after e.g. "...repaired.Can you summarize ...?" is found even
# without whitespace because "." is in the excluded class.
#
# Implementation note: the natural formulation ``((?:SEG)+)$`` backtracks
# catastrophically on long inputs whose tail does NOT tile cleanly (the outer
# ``+`` retries every split point over 100KB+ strings — this made the medium
# pool scan burn hours inside a single longmagpie shard).  A segment match is
# unambiguous at any position (``[^.!?]*`` cannot cross a ``?``), so the
# language is the same whether anchored-and-backtracked or tiled greedily:
# ``finditer`` produces THE unique segmentation in linear time, and the
# answer is the maximal contiguous trailing chain.
_LONGMAGPIE_SEGMENT_RE = re.compile(r"[^.!?]*\?[\"'”’)\]]*\s*")


def split_longmagpie_question(text: str) -> Optional[Tuple[str, str]]:
    """Split a LongMagpie user message into ``(context document, question)``.

    The question is the trailing run of sentences ending in ``?``; the
    remainder is the context.  Returns ``None`` when there is no such suffix —
    the caller then skips the row (and counts the skip).
    """
    body = (text or "").strip()
    if not body:
        return None
    end = len(body)
    start = end
    for match in reversed(list(_LONGMAGPIE_SEGMENT_RE.finditer(body))):
        if match.end() != start:
            break
        start = match.start()
    if start == end:
        return None
    question = body[start:end].strip()
    context = body[:start].rstrip()
    if not context or not question:
        return None
    return context, question


def longmagpie_row_to_example(
    row: Dict[str, Any], row_index: int = 0, shard: Optional[str] = None
) -> Optional[JointExample]:
    """LongMagpie row: the ``?``-suffixed trailing sentence(s) become the question.

    ``shard`` (parquet file stem) + ``row_index`` (row number WITHIN that
    shard, counting every row including ones the split rule skips) form the
    qid: ``qa:longmagpie:<shard>:<row_in_shard>``.  The previous global row
    index shifted every qid whenever the shard set/order changed (P1-6) and
    never matched the dedup removal ids (P1-5).  ``shard=None`` keeps the bare
    index for direct/ad-hoc calls; the source always passes the shard.
    """
    messages = _json_loads(row.get("messages"), [])
    if not isinstance(messages, list):
        return None
    user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    assistant = next((m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), None)
    if user is None or assistant is None:
        return None
    split = split_longmagpie_question(str(user.get("content") or ""))
    if split is None:
        return None
    context, question = split
    row_id = f"{shard}:{row_index}" if shard is not None else str(row_index)
    return _qa_joint_example("longmagpie", row_id, [context], question, assistant.get("content"))


# ---------------------------------------------------------------------------
# IO helpers (pyarrow imported lazily inside functions; the module imports
# cleanly without pyarrow installed).
# ---------------------------------------------------------------------------


def _find_parquet_files(path: Path, subdirs: Sequence[str] = ("data",)) -> List[Path]:
    """Sorted parquet file list under ``path`` (first matching layout wins)."""
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    roots = [path / subdir for subdir in subdirs] + [path]
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.parquet"))
            if files:
                return files
    return []


def _find_openswe_parquet_files(path: Path) -> List[Path]:
    """Open-SWE layout: ``data/<config>/*.parquet`` (mirrors the multiturn source)."""
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    root = path / "data"
    if not root.is_dir():
        root = path
    files = sorted(root.glob("*/*.parquet"))
    if not files:
        files = sorted(root.glob("*.parquet"))
    return files


def _iter_parquet_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Whole-file parquet read, one file at a time (lazy pyarrow).

    Mirrors ``dedup_cross_dataset._iter_records``: ``ParquetFile.iter_batches``
    raises ArrowNotImplementedError on nested columns (Open-SWE ``trajectory``,
    2Wiki ``context``), so the whole file is read at once.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except Exception:
        table = pq.ParquetFile(path).read()
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl files found under {path}")
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json_loads(line, None)
                if isinstance(row, dict):
                    yield row


def _apply_file_order_seed(files: Sequence[Path], file_order_seed: Optional[str]) -> List[Path]:
    """Seeded shuffle of the scan's FILE order (row order inside a file is
    unchanged); the default (None) keeps the sorted on-disk order.

    The mixture planner passes ``f"{order_seed}:scan:{family}:{subset}"`` so a
    cap-truncated pool scan is a seeded-random file prefix rather than the
    alphabetical head (P0-2).  qids must therefore never encode cross-file
    positions — longmagpie qids carry the shard stem for exactly this reason.
    """
    if file_order_seed is None:
        return list(files)
    files = list(files)
    random.Random(file_order_seed).shuffle(files)
    return files


class _StreamedJointSource:
    """Shared lazy-iteration surface for the multi-source joint sources.

    ``__iter__`` streams row -> JointExample conversions; ``__len__``
    materializes and caches (afterwards ``__iter__`` replays the cache).
    ``self.stats`` is a Counter reset at the start of every pass.
    """

    _cache: Optional[List[JointExample]]
    stats: Counter

    def _iter_records(self) -> Iterator[JointExample]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[JointExample]:
        if self._cache is not None:
            yield from self._cache
        else:
            yield from self._iter_records()

    def __len__(self) -> int:
        if self._cache is None:
            self._cache = list(self._iter_records())
        return len(self._cache)

    def _finalize_pass(self, emitted: int) -> None:
        logger.info(
            "%s: emitted %d examples (%s)",
            self.__class__.__name__,
            emitted,
            dict(self.stats),
        )

    def _postprocess(
        self,
        examples: List[JointExample],
        rng: random.Random,
    ) -> List[JointExample]:
        """keep_qids prefilter / per-conversation seeded subsampling.

        ``keep_qids`` wins over ``max_samples_per_session``: an order file
        fixes the exact example set, so subsampling must not drop listed qids.
        Subsampling mirrors the traces source: one rng shared across
        conversations in iteration order, ``rng.sample`` only when over cap.
        """
        if self.keep_qids is not None:
            kept = [example for example in examples if example.qid in self.keep_qids]
            self.stats["filtered_out_by_keep_qids"] += len(examples) - len(kept)
            return kept
        if self.max_samples_per_session and len(examples) > self.max_samples_per_session:
            self.stats["subsampled_conversations"] += 1
            return rng.sample(examples, self.max_samples_per_session)
        return examples


class ToucanJointSource(_StreamedJointSource):
    """True-joint source for the Toucan SFT ``multi-turn`` subset.

    Streams ``SFT/*.parquet`` with a pushed-down ``subset_name == "multi-turn"
    `` filter (pyarrow.dataset).  Conversion itself lives in
    ``toucan_row_to_examples``; this class is only IO + the common knobs.
    ``file_order_seed`` (planner pool scans only) seeded-shuffles the parquet
    file list before the dataset scan (P0-2 stratified pool scanning).
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        keep_qids: Optional[FrozenSet[str]] = None,
        max_records: Optional[int] = None,
        split_seed: int = 42,
        max_samples_per_session: Optional[int] = None,
        require_tool_call: bool = True,
        canonical_format_prob: float = 0.7,
        minified_json_prob: float = 0.2,
        shuffle_tools: bool = True,
        truncate_description_chars: int = 600,
        max_tools_per_sample: int = 32,
        same_namespace_negative_tools: int = 8,
        random_negative_tools: int = 24,
        file_order_seed: Optional[str] = None,
    ) -> None:
        _validate_split(split)
        self.path = Path(path)
        self.keep_qids = keep_qids
        self.max_records = max_records
        self.split_seed = split_seed
        self.max_samples_per_session = max_samples_per_session
        self.require_tool_call = require_tool_call
        self.canonical_format_prob = canonical_format_prob
        self.minified_json_prob = minified_json_prob
        self.shuffle_tools = shuffle_tools
        self.truncate_description_chars = truncate_description_chars
        self.max_tools_per_sample = max_tools_per_sample
        self.same_namespace_negative_tools = same_namespace_negative_tools
        self.random_negative_tools = random_negative_tools
        self.file_order_seed = file_order_seed
        self._cache = None

    def _iter_rows(self) -> Iterator[Dict[str, Any]]:
        import pyarrow.dataset as pds  # lazy: the module imports without pyarrow

        files = _find_parquet_files(self.path, subdirs=("SFT",))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.path} (or its SFT/ subdirectory)")
        files = _apply_file_order_seed(files, self.file_order_seed)
        dataset = pds.dataset([str(file) for file in files], format="parquet")
        wanted = [name for name in ("uuid", "subset_name", "tools", "messages") if name in dataset.schema.names]
        for batch in dataset.to_batches(
            filter=pds.field("subset_name") == "multi-turn",
            columns=wanted or None,
            batch_size=256,
            # Single-threaded: the (possibly file_order_seed-shuffled) file
            # list order must be the actual scan order, so a cap-truncated
            # planner scan is deterministic (P0-2/P2).
            use_threads=False,
        ):
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield row

    def _iter_records(self) -> Iterator[JointExample]:
        self.stats = Counter()
        rng = random.Random(self.split_seed)
        emitted = 0
        try:
            for row in self._iter_rows():
                self.stats["rows_scanned"] += 1
                examples = toucan_row_to_examples(
                    row,
                    split_seed=self.split_seed,
                    require_tool_call=self.require_tool_call,
                    canonical_format_prob=self.canonical_format_prob,
                    minified_json_prob=self.minified_json_prob,
                    shuffle_tools=self.shuffle_tools,
                    truncate_description_chars=self.truncate_description_chars,
                    max_tools_per_sample=self.max_tools_per_sample,
                    same_namespace_negative_tools=self.same_namespace_negative_tools,
                    random_negative_tools=self.random_negative_tools,
                )
                if not examples:
                    self.stats["rows_without_examples"] += 1
                    continue
                for example in self._postprocess(examples, rng):
                    if self.max_records is not None and emitted >= self.max_records:
                        self.stats["truncated_at_max_records"] += 1
                        return
                    yield example
                    emitted += 1
        finally:
            self._finalize_pass(emitted)


class OpenSWEJointSource(_StreamedJointSource):
    """True-joint source for NVIDIA Open-SWE-Traces (``resolved == 1`` only).

    Streams ``data/*/*.parquet``; ``subset`` is ``openswe:<subdir_name>`` (the
    trajectory-config directory, e.g. ``openswe:qwen35_sweagent``).
    Conversion itself lives in ``openswe_row_to_examples``.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        keep_qids: Optional[FrozenSet[str]] = None,
        max_records: Optional[int] = None,
        split_seed: int = 42,
        max_samples_per_session: Optional[int] = None,
        require_tool_call: bool = True,
        canonical_format_prob: float = 0.7,
        minified_json_prob: float = 0.2,
        shuffle_tools: bool = True,
        truncate_description_chars: int = 600,
        max_tools_per_sample: int = 32,
        same_namespace_negative_tools: int = 8,
        random_negative_tools: int = 24,
        file_order_seed: Optional[str] = None,
    ) -> None:
        _validate_split(split)
        self.path = Path(path)
        self.keep_qids = keep_qids
        self.max_records = max_records
        self.split_seed = split_seed
        self.max_samples_per_session = max_samples_per_session
        self.require_tool_call = require_tool_call
        self.canonical_format_prob = canonical_format_prob
        self.minified_json_prob = minified_json_prob
        self.shuffle_tools = shuffle_tools
        self.truncate_description_chars = truncate_description_chars
        self.max_tools_per_sample = max_tools_per_sample
        self.same_namespace_negative_tools = same_namespace_negative_tools
        self.random_negative_tools = random_negative_tools
        self.file_order_seed = file_order_seed
        self._cache = None

    def _iter_rows(self) -> Iterator[Tuple[Dict[str, Any], str]]:
        files = _find_openswe_parquet_files(self.path)
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.path} (or data/*/)")
        for file in _apply_file_order_seed(files, self.file_order_seed):
            subset = f"openswe:{file.parent.name}"
            for row in _iter_parquet_rows(file):
                yield row, subset

    def _iter_records(self) -> Iterator[JointExample]:
        self.stats = Counter()
        rng = random.Random(self.split_seed)
        emitted = 0
        try:
            for row, subset in self._iter_rows():
                self.stats["rows_scanned"] += 1
                if row.get("resolved") != 1:
                    self.stats["rows_unresolved"] += 1
                    continue
                examples = openswe_row_to_examples(
                    row,
                    subset=subset,
                    split_seed=self.split_seed,
                    require_tool_call=self.require_tool_call,
                    canonical_format_prob=self.canonical_format_prob,
                    minified_json_prob=self.minified_json_prob,
                    shuffle_tools=self.shuffle_tools,
                    truncate_description_chars=self.truncate_description_chars,
                    max_tools_per_sample=self.max_tools_per_sample,
                    same_namespace_negative_tools=self.same_namespace_negative_tools,
                    random_negative_tools=self.random_negative_tools,
                )
                if not examples:
                    self.stats["rows_without_examples"] += 1
                    continue
                for example in self._postprocess(examples, rng):
                    if self.max_records is not None and emitted >= self.max_records:
                        self.stats["truncated_at_max_records"] += 1
                        return
                    yield example
                    emitted += 1
        finally:
            self._finalize_pass(emitted)


class QADocsJointSource(_StreamedJointSource):
    """True-joint source for the official QA document corpora.

    Any subset path may be None to disable it.  Every qualifying row maps to
    exactly one example with ``tool_documents=[]`` (see the conversion cores).
    ``require_tool_call`` is accepted for a uniform cross-source call
    signature but intentionally NOT applied: QA examples have no tool calls by
    design (the knob filters the agentic sources only).
    ``max_samples_per_session`` is likewise a no-op here (one example per row,
    so a conversation never exceeds a positive cap).
    ``file_order_seed`` (planner pool scans only) shuffles the per-family FILE
    order; qids are file-position-independent (longmagpie qids embed the shard
    stem), so shuffling changes pool composition under a cap, never qid
    resolution.
    """

    def __init__(
        self,
        hotpotqa_path: Optional[str] = None,
        wiki2_path: Optional[str] = None,
        longmagpie_path: Optional[str] = None,
        split: str = "train",
        keep_qids: Optional[FrozenSet[str]] = None,
        max_records: Optional[int] = None,
        split_seed: int = 42,
        max_samples_per_session: Optional[int] = None,
        require_tool_call: bool = True,
        file_order_seed: Optional[str] = None,
    ) -> None:
        _validate_split(split)
        if not any([hotpotqa_path, wiki2_path, longmagpie_path]):
            raise ValueError(
                "QADocsJointSource needs at least one of "
                "hotpotqa_path / wiki2_path / longmagpie_path"
            )
        self.hotpotqa_path = hotpotqa_path
        self.wiki2_path = wiki2_path
        self.longmagpie_path = longmagpie_path
        self.keep_qids = keep_qids
        self.max_records = max_records
        self.split_seed = split_seed
        self.max_samples_per_session = max_samples_per_session
        self.require_tool_call = require_tool_call
        self.file_order_seed = file_order_seed
        self._cache = None

    def _iter_family(self, family: str) -> Iterator[JointExample]:
        if family == "longmagpie":
            # Per-file iteration: the qid embeds the shard stem and the row
            # number WITHIN the shard (P1-6), so adding/renaming/reordering
            # shards never shifts other rows' qids (the old global row index
            # did) and dedup removal ids match the qids exactly (P1-5).
            files = _find_parquet_files(Path(self.longmagpie_path))
            if not files:
                raise FileNotFoundError(f"No parquet files found under {self.longmagpie_path}")
            for file in _apply_file_order_seed(files, self.file_order_seed):
                for row_in_shard, row in enumerate(_iter_parquet_rows(file)):
                    self.stats["longmagpie_rows"] += 1
                    example = longmagpie_row_to_example(row, row_in_shard, shard=file.stem)
                    if example is None:
                        # The dominant cause is a user message without a
                        # trailing "?" question suffix (counted, per the split
                        # rule).  Skipped rows still consume row_in_shard.
                        self.stats["longmagpie_skipped"] += 1
                        continue
                    yield example
            return
        if family == "hotpotqa":
            rows = _iter_jsonl_rows(Path(self.hotpotqa_path))
            converter = hotpotqa_row_to_example
        else:
            files = _find_parquet_files(Path(self.wiki2_path), subdirs=())
            if not files:
                raise FileNotFoundError(f"No parquet files found under {self.wiki2_path}")
            files = _apply_file_order_seed(files, self.file_order_seed)
            rows = (row for file in files for row in _iter_parquet_rows(file))
            converter = wiki2_row_to_example
        for row_index, row in enumerate(rows):
            self.stats[f"{family}_rows"] += 1
            example = converter(row, row_index)
            if example is None:
                self.stats[f"{family}_skipped"] += 1
                continue
            yield example

    def _iter_records(self) -> Iterator[JointExample]:
        self.stats = Counter()
        rng = random.Random(self.split_seed)  # unused while max_samples_per_session is a no-op
        emitted = 0
        try:
            for family, enabled in (
                ("hotpotqa", self.hotpotqa_path),
                ("2wiki", self.wiki2_path),
                ("longmagpie", self.longmagpie_path),
            ):
                if not enabled:
                    continue
                for example in self._iter_family(family):
                    if self.max_records is not None and emitted >= self.max_records:
                        self.stats["truncated_at_max_records"] += 1
                        return
                    for kept in self._postprocess([example], rng):
                        yield kept
                        emitted += 1
        finally:
            self._finalize_pass(emitted)
