"""True-joint C2KV training data: tool schemas AND history as compressed documents.

The two existing training paths each leak one modality into the uncompressed
prefix:

- ``agent/train_agent_tool_definition_c2kv.py`` compresses only tool schemas;
  the conversation history stays in the ordinary prompt.
- ``python/train/train_data_multiturn.py`` (``CompressHistoryDataset``)
  compresses only history turns; the *complete* tool schemas stay in the
  system prefix (``_chat_template_ids(..., tools=example.tools ...)``).

``GistMultiDocTrainer`` / ``gist_utils.process_context_input_ids`` already
treat a context document as an opaque chat-template-wrapped string, so a true
joint dataset only needs this new data module:

- context documents = tool-schema chunks FIRST, then history-turn chunks in
  chronological order (each doc keeps its type prefix: ``"Tool definition:\n"``
  for tools, ``"Previous turn\n..."`` for history);
- ordinary prompt = current turn only (plus the raw history tail when the
  ``hybrid_tail_choices`` knob is set — see ``JointDataset``);
- target = next assistant action;
- system prefix = bare system prompt WITHOUT ``tools=`` (the de-leak), unless
  the ``tools_in_system`` knob is set — see ``JointDataset``.

Answer-format choice
--------------------
The answer is rendered with the history path's ``_render_agent_output_messages``
(one existing code path that handles reasoning/content/tool_calls/parts).  When
the span ends in tool calls this emits exactly the unified path's target
surface ``Action:\n<tool_call>\n{"name":...,"arguments":...}\n</tool_call>``
(same payload keys, same minified JSON), so next-action supervision is
consistent with ``train_unified_next_action_c2kv.py``; spans without tool calls
fall back to the assistant text (history-path behavior).  ``require_tool_call``
restricts to tool-call targets when set.  Inline ``<think>...</think>``
residue is stripped from the rendered answer (``_strip_think_blocks``), and a
tool-call answer that still does not fit the sequence budget after maximal
prompt truncation is dropped (``tool_call_target_truncated``) rather than
trained on as a partial action.

Document-budget allocation
--------------------------
``max_doc_num`` context slots per example.  Tool chunks are kept first, up to
``max_tool_chunks`` (default ``max(1, 2 * max_doc_num // 3)``); excess tool
chunks are dropped (``max_tool_definition_tokens`` additionally caps the total
tool token count, mirroring the tooldef path — over-cap examples are skipped).
History documents get the remaining slots (``max_doc_num - len(tool_chunks)``)
and are fitted with the history path's ``_fit_reused_history`` semantics:
oversized docs are split to ``max_doc_length`` and over-budget docs are
selected tail-biased (first doc + most recent) via ``_select_history``.
``doc_mode="tool_only"`` gives tools all ``max_doc_num`` slots and drops
history; ``doc_mode="history_only"`` does the opposite — both for the
J-alternate training arm and per-condition evals.

History chunking policy
-----------------------
``build_history_chunks`` is the single history-side entry point, shared with
``agent/eval_joint_next_action_c2kv.py``'s ``_condition_doc_chunks``.  With
its defaults (``chunk_policy="agent-turn"``, ``delay_recent_turns=0``) it
short-circuits to the ``_fit_reused_history`` call this module has always
made, bit for bit.  Other policies (``python/train/chunk_policy.py``) re-cut
the SAME frozen doc texts, and ``delay_recent_turns`` moves the last k turns
out of the context grid into the plain prompt — the ``full_history_doc_num``
semantics of ``train_data_multiturn.py:1210-1224``, ported here.  Because
``structural`` can emit more docs than the slot budget, a row whose total doc
count exceeds ``max_doc_num`` is skipped (``doc_num>N``) rather than silently
reshaping the fixed training grid.  ``content_tokens`` accounting is opt-in
(``need_content_tokens``): it costs a full extra encode of the history text,
so the trainer leaves it off (``None`` = not measured) and only the eval
driver, which needs it for the presented-token check, turns it on.

Per-side caps regime (``per_side_caps=True``, the default since the cap fix):
the tool side gets ``min(max_tool_chunks, max_doc_num)`` slots and the history
side a CONSTANT ``max_doc_num - min(max_tool_chunks, max_doc_num)`` in every
doc mode, so both presented budgets are identical across the J-arms (the G-Q3
fairness constraint); spare tool slots are NOT recycled.  Regime
``per_side_caps_v2_empty_tool_reclaim`` refines this for examples with NO
non-empty tool documents (the QA family): they reserve no tool slots at all
(``tool_cap=0``) and history gets the full ``max_doc_num`` grid.  Examples
WITH tool documents are bit-for-bit identical to v1; the legacy regime
(``per_side_caps=False``) is untouched byte-for-byte.  The regime string
recorded in manifests/summaries comes from ``cap_regime_name``.

Self-checks
-----------
``assert_no_leakage`` is NOT run per-sample in production; it is called from
tests and from ``python -m train.train_data_joint --self_test`` (optionally
with ``--tokenizer <local HF path>``; defaults to a built-in whitespace
tokenizer so the smoke test runs offline).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .chunk_policy import (
    FROZEN_JOIN,
    apply_policy,
    fit_history_with_provenance,
    parse_chunk_policy,
    split_delayed,
)
from .train_data import DEFAULT_SYSTEM_PROMPT
from .train_data_multiturn import (
    AgentLLMTracesCompressHistorySource,
    HistorySelection,
    Message,
    _agent_history_turn_docs,
    _agent_history_turn_units,
    _agent_message_parts,
    _agent_system_prompt,
    _chat_template_ids,
    _find_agent_jsonl_files,
    _find_agent_parquet_files,
    _fit_reused_history,
    _flatten_turn_units,
    _fit_reused_history_with_indices,
    _iter_agent_jsonl_rows,
    _iter_agent_rows,
    _json_loads,
    _message_token_length,
    _normal_agent_message,
    _normal_chat_message,
    _pad,
    _render_agent_output_messages,
    _select_history,
    _sort_agent_spans,
    _span_attributes,
    _split_message_to_fit,
    _tool_list_from_agent_value,
    _toolathlon_row_to_agent_session,
)

logger = logging.getLogger(__name__)


DocMode = Literal["joint", "tool_only", "history_only"]

TOOL_DOC_PREFIX = "Tool definition:\n"
HISTORY_DOC_PREFIX = "Previous turn"


@dataclass(frozen=True)
class JointExample:
    """One true-joint training example.

    ``tool_documents`` are per-tool-schema rendered texts (no prefix yet; the
    ``"Tool definition:\n"`` prefix is added when each document is wrapped in
    the chat template).  ``history_documents`` are per-turn ``"Previous turn
    ..."`` texts in chronological order, covering every turn BEFORE the current
    one.  ``current_messages`` is the current turn only and ``system_prompt``
    never contains tool schemas.
    """

    qid: str
    session_id: str
    tool_documents: List[str]
    history_documents: List[str]
    current_messages: List[Message]
    answer: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    subset: str = "unknown"
    # Which entry of ``tool_documents`` renders the target tool's schema
    # (post-shuffle), so budget truncation can keep it in the grid.  ``None``
    # when the target tool was not identified in the definitions.
    target_tool: Optional[str] = None
    target_tool_doc_index: Optional[int] = None
    # Role-preserving twin of ``history_documents``: one unit list per turn
    # (``_agent_history_turn_units``), same order and same length.  ``None``
    # on records built before the B-line chunking work; the ``structural``
    # policy degrades to a pass-through when it is missing.
    history_units: Optional[List[List[Dict[str, str]]]] = None
    # Indices into ``history_documents`` of the GOLD (supporting-fact)
    # documents, when the source corpus labels them (HotpotQA / 2Wiki QA
    # rows).  ``None`` when unlabelled.  Used only for the retention audit
    # counters — never for training decisions.
    gold_history_doc_indices: Optional[Tuple[int, ...]] = None
    # ``"tool_call"`` when the rendered answer carries a tool call (the
    # ``_render_agent_output_messages`` predicate at the extraction point),
    # else ``"other"`` (clarification / no-call / final response).  Drives
    # action-balanced per-session sampling and the manifest's
    # ``action_type_counts`` audit; never a training decision.
    action_type: str = "other"
    # The raw tool dicts chosen by ``_select_tools`` (post-selection,
    # pre-render), kept so a training arm can present the tool schemas RAW in
    # the system prefix (``tools_in_system``) exactly as every serving path
    # does, instead of through the gist grid.  ``None`` when the source has no
    # tool side (QA) or did not record the selection.
    selected_tools: Optional[List[Dict[str, Any]]] = None


# Mixture-family prefixes on qids (``toucan:`` / ``openswe:`` / ``qa:``);
# agent-llm-traces qids are bare ``session_id:span_index``.  Defined here (not
# in the multisource module) so JointDataset can attribute skip counters per
# family without an import cycle; re-exported by train_data_joint_multisource.
FAMILY_PREFIXES = ("toucan", "openswe", "qa")


def qid_source_family(qid: str) -> str:
    """Mixture family of an example qid (bare qids count as ``traces``)."""
    for prefix in FAMILY_PREFIXES:
        if qid.startswith(prefix + ":"):
            return prefix
    return "traces"


# Doc-budget regime names recorded in trainer manifests / eval summaries.
# ``per_side_caps`` (v1, the first fixed regime) and v2 differ ONLY for
# tool-less examples; both record legacy_mode_caps=False, so the boolean alone
# cannot tell them apart when merging shards — the string can.
CAP_REGIME_LEGACY = "legacy_mode_caps"
CAP_REGIME_V1 = "per_side_caps"
CAP_REGIME_V2_EMPTY_TOOL_RECLAIM = "per_side_caps_v2_empty_tool_reclaim"


def cap_regime_name(legacy_mode_caps: bool) -> str:
    """The regime THIS code produces: legacy, or v2 (empty-tool reclaim)."""
    return CAP_REGIME_LEGACY if legacy_mode_caps else CAP_REGIME_V2_EMPTY_TOOL_RECLAIM


def regime_from_record(legacy_mode_caps: Any, cap_regime: Any) -> str:
    """Normalize a manifest/summary's regime fields to a comparable string.

    Records written before the regime string existed carry only the
    ``legacy_mode_caps`` boolean; map those to v1 (they predate the v2
    reclaim).  Missing entirely -> "unknown".
    """
    if isinstance(cap_regime, str) and cap_regime:
        return cap_regime
    if legacy_mode_caps is None:
        return "unknown"
    return CAP_REGIME_LEGACY if bool(legacy_mode_caps) else CAP_REGIME_V1


def _has_tool_documents(example: JointExample) -> bool:
    """True when the example carries at least one non-empty tool document."""
    return any(str(doc).strip() for doc in example.tool_documents)


# ---------------------------------------------------------------------------
# Tool-document rendering variants.
#
# Copied from agent/train_unified_next_action_c2kv.py (``_render_tool_documents``
# and its helpers ``_namespace``/``_tool_name``/``_schema_obj``/
# ``_parameter_signature``/``_tool_description``/``_canonical_tool_doc``/
# ``_shuffle_json_keys``; ``_tool_name`` itself mirrors
# agent/eval_agent_tool_definition_hybrid_router.py).  The original module
# cannot be imported here: it sits under agent/ (not importable from
# python/train) and pulls in torch/models at import time.  The only adaptation
# is the signature: explicit scalar knobs instead of UnifiedNextActionDataArgs,
# and only the per-tool document list is returned (the joined string is unused
# here).  The 0.7/0.2/0.1 canonical/minified/shuffled variant policy is
# unchanged.
# ---------------------------------------------------------------------------


def _namespace(name: str) -> str:
    for sep in ("__", ".", "/", "-"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name.split("_", 1)[0] if "_" in name else name


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _schema_obj(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = function.get("parameters") or tool.get("parameters") or tool.get("input_schema") or tool.get("schema") or {}
    return schema if isinstance(schema, dict) else {}


def _parameter_signature(tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = _schema_obj(tool)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    # Real-world schemas ship ``"required": null``; ``.get`` with a default
    # does not cover an explicit null, so guard the type before iterating.
    required_raw = schema.get("required")
    required = set(str(item) for item in required_raw) if isinstance(required_raw, list) else set()
    rows = []
    for name, value in properties.items():
        value = value if isinstance(value, dict) else {}
        typ = value.get("type") or value.get("anyOf") or value.get("oneOf") or value.get("items") or "unknown"
        rows.append({
            "name": str(name),
            "type": json.dumps(typ, ensure_ascii=False, sort_keys=True) if not isinstance(typ, str) else typ,
            "required": str(name) in required,
        })
    return sorted(rows, key=lambda item: item["name"])


def _tool_description(tool: Dict[str, Any], limit: int) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    description = str(function.get("description") or tool.get("description") or "")
    return description[:limit].rstrip() if limit and len(description) > limit else description


def _canonical_tool_doc(tool: Dict[str, Any], description_limit: int) -> str:
    name = _tool_name(tool)
    blocks = [
        "<TOOL>",
        f"<NAMESPACE> {_namespace(name)}",
        f"<NAME> {name}",
    ]
    description = _tool_description(tool, description_limit)
    if description:
        blocks.append(f"<DESCRIPTION> {description}")
    blocks.append("<PARAMETERS>")
    for param in _parameter_signature(tool):
        blocks.append(
            f'<PARAM name="{param["name"]}" type="{param["type"]}" '
            f'required="{str(param["required"]).lower()}">'
        )
    blocks.extend(["</PARAMETERS>", "</TOOL>"])
    return "\n".join(blocks)


def _shuffle_json_keys(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: _shuffle_json_keys(item, rng) for key, item in items}
    if isinstance(value, list):
        return [_shuffle_json_keys(item, rng) for item in value]
    return value


def _shuffled_system_tools(
    selected_tools: Sequence[Dict[str, Any]],
    split_seed: int,
    session_id: Any,
    span_index: Any,
) -> List[Dict[str, Any]]:
    """Order the ``tools_in_system`` pool without a positional oracle.

    ``_select_tools`` builds its pool as ``target + same-namespace negatives +
    random negatives``, so the gold tool is always element 0 whenever the
    session ships more tools than ``max_tools_per_sample``.  The grid path
    never sees that order (``_render_tool_documents`` shuffles its own copy),
    but ``tools_in_system`` renders this list verbatim into the system prefix
    -- and "the answer is the first tool" is a shortcut that no serving path
    (BFCL, tau2, ToolSandbox all pass the caller's own tool order) reproduces.

    A dedicated RNG stream keyed by the example id keeps this deterministic
    and leaves the grid path's ``rng`` consumption untouched, so every
    ``doc_mode != history_only`` arm stays bit-identical.
    """
    tools = list(selected_tools)
    random.Random(f"{split_seed}:{session_id}:{span_index}:system_tool_order").shuffle(tools)
    return tools


def _render_tool_documents(
    tools: Sequence[Dict[str, Any]],
    rng: random.Random,
    *,
    canonical_format_prob: float = 0.7,
    minified_json_prob: float = 0.2,
    shuffle_tools: bool = True,
    truncate_description_chars: int = 600,
    target_tool: Optional[str] = None,
) -> tuple[List[str], Optional[int]]:
    """Render per-tool documents; also report the target tool's post-shuffle index.

    Returns ``(docs, target_index)`` where ``target_index`` is the position of
    the document rendering ``target_tool`` after the shuffle (``None`` when
    ``target_tool`` is not given or not present).  The index — not a text
    match — is what budget truncation uses to keep the target schema in the
    context grid, so it stays correct across all three render variants.
    """
    tools = list(tools)
    if shuffle_tools:
        rng.shuffle(tools)
    target_index: Optional[int] = None
    if target_tool:
        for index, tool in enumerate(tools):
            if _tool_name(tool) == target_tool:
                target_index = index
                break
    p = rng.random()
    if p < canonical_format_prob:
        docs = [_canonical_tool_doc(tool, truncate_description_chars) for tool in tools]
    elif p < canonical_format_prob + minified_json_prob:
        docs = [json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools]
    else:
        docs = [
            json.dumps(_shuffle_json_keys(tool, rng), ensure_ascii=False, separators=(",", ":"))
            for tool in tools
        ]
    return docs, target_index


def _first_tool_call_name(output_messages: Any) -> Optional[str]:
    """Name of the first tool call in a span's output messages.

    Handles the same shapes as ``train_data_multiturn._render_agent_tool_calls``:
    ``tool_calls``/``toolCalls``/``function_call`` keys or gen_ai ``parts``.
    """
    messages = _json_loads(output_messages, [])
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = (
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _agent_message_parts(message)
        )
        calls = _json_loads(calls, [])
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            if call.get("type") not in (None, "tool_call", "function_call") and "function" not in call:
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = (
                function.get("name")
                or call.get("name")
                or call.get("tool_name")
                or call.get("function_name")
                or ""
            )
            if name:
                return str(name)
    return None


def _select_tools(
    tools: Sequence[Dict[str, Any]],
    target_tool: Optional[str],
    rng: random.Random,
    *,
    max_tools_per_sample: int = 32,
    same_namespace_negative_tools: int = 8,
    random_negative_tools: int = 24,
) -> List[Dict[str, Any]]:
    """Target-inclusive bounded tool pool (unified path's policy, scalar knobs).

    Sessions like AppWorld ship far more tool schemas than fit the document
    budget, so each example compresses a bounded pool: the target tool plus
    same-namespace and random negatives, mirroring
    ``train_unified_next_action_c2kv._select_tools``.  If the target tool is
    not identified in the definitions, the leading tools in declared order
    are kept (deterministic) rather than a random subset.
    """
    if not max_tools_per_sample or len(tools) <= max_tools_per_sample:
        return list(tools)
    target = [tool for tool in tools if _tool_name(tool) == target_tool]
    if not target:
        return list(tools[:max_tools_per_sample])
    target_namespace = _namespace(target_tool or "")
    same_namespace = [
        tool for tool in tools
        if _tool_name(tool) != target_tool and _namespace(_tool_name(tool)) == target_namespace
    ]
    others = [
        tool for tool in tools
        if _tool_name(tool) != target_tool and _namespace(_tool_name(tool)) != target_namespace
    ]
    rng.shuffle(same_namespace)
    rng.shuffle(others)
    selected = target[:1]
    selected.extend(same_namespace[:same_namespace_negative_tools])
    remaining = max(0, max_tools_per_sample - len(selected))
    selected.extend(others[: min(random_negative_tools, remaining)])
    remaining = max(0, max_tools_per_sample - len(selected))
    if remaining:
        selected.extend((same_namespace[same_namespace_negative_tools:] + others[random_negative_tools:])[:remaining])
    return selected[:max_tools_per_sample]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(answer: str) -> str:
    """Remove inline ``<think>...</think>`` residue from a rendered answer.

    Raw traces can carry the model's reasoning inline in the assistant
    ``content`` string (Hermes-style ``<think>`` segments, docs/datastes.md)
    instead of a dedicated ``reasoning_content`` key, and no extraction
    helper strips them.  Reasoning is deliberately supervised only via the
    ``Thought:`` rendering of the dedicated keys, so inline segments are
    dropped here.  An unclosed trailing ``<think>`` (``max_answer_chars`` can
    cut a block in half) is stripped to the end as well.
    """
    stripped = _THINK_BLOCK_RE.sub("", answer)
    if "<think>" in stripped:
        stripped = stripped.split("<think>", 1)[0]
    return stripped.strip()


def _answer_has_tool_call(answer: str) -> bool:
    """The marker half of the ``_render_agent_output_messages`` tool-call
    predicate, applied to an already-rendered answer string."""
    marker_text = answer.lower()
    return any(
        marker in marker_text
        for marker in ("<tool_call>", "action:", "function_call", "tool call")
    )


def _stratified_pick(
    examples: Sequence[JointExample],
    k: int,
    rng: random.Random,
    action_tool_call_frac: float = 0.75,
) -> List[JointExample]:
    """Position-stratified, action-balanced down-sampling of one session.

    ``examples`` are in chronological decision-point order (the order
    ``_session_examples`` yields them).  The index range is split into
    early/middle/late thirds with quotas ``k // 3`` / ``k // 3`` / remainder
    (1/1/2 for the default ``k=4``): late decision points carry the longest
    histories and keep the largest share.  Within each bucket the seeded rng
    picks the members, preferring action types so the session's tool_call
    share approaches ``action_tool_call_frac``: per-session targets are
    ``round(k * frac)`` tool-call picks and the rest ``other``, consumed
    bucket by bucket; a bucket short on the preferred pool backfills from its
    other pool, and a bucket short overall is topped up after the pass in
    late -> middle -> early order.  Deterministic given the caller's rng —
    every DDP rank rebuilds the same example list independently.  Returns the
    picks in chronological order.
    """
    if len(examples) <= k:
        return list(examples)
    third = len(examples) // 3
    buckets = [examples[:third], examples[third : 2 * third], examples[2 * third :]]
    quotas = (k // 3, k // 3, k - 2 * (k // 3))
    tool_target = max(0, min(k, int(round(k * action_tool_call_frac))))
    other_target = k - tool_target
    picked: List[JointExample] = []
    for bucket, quota in zip(buckets, quotas):
        take = min(quota, len(bucket))
        tool_pool = [example for example in bucket if example.action_type == "tool_call"]
        other_pool = [example for example in bucket if example.action_type != "tool_call"]
        # Targets are clamped at 0: a bucket that had to backfill from the
        # non-preferred pool (``short`` below) can drive the other target
        # negative, and ``rng.sample(pool, -1)`` raises ValueError -- an ordinary
        # "text turns first, tool calls later" trajectory crashed dataset load
        # with the launcher default require_tool_call=False (2026-09-05 audit).
        n_tool = min(take, max(0, tool_target), len(tool_pool))
        n_other = min(take - n_tool, max(0, other_target), len(other_pool))
        short = take - n_tool - n_other
        if short:
            # Pool short on the preferred action: fill the bucket quota from
            # whichever pool still has members.
            extra_tool = min(short, len(tool_pool) - n_tool)
            n_tool += extra_tool
            n_other += min(short - extra_tool, len(other_pool) - n_other)
        chosen = rng.sample(tool_pool, n_tool) + rng.sample(other_pool, n_other)
        chosen_tool = sum(1 for example in chosen if example.action_type == "tool_call")
        tool_target = max(0, tool_target - chosen_tool)
        other_target = max(0, other_target - (len(chosen) - chosen_tool))
        picked.extend(chosen)
    if len(picked) < k:
        # Bucket short overall: backfill late -> middle -> early.
        chosen_ids = {id(example) for example in picked}
        for bucket in reversed(buckets):
            pool = [example for example in bucket if id(example) not in chosen_ids]
            extra = rng.sample(pool, min(k - len(picked), len(pool)))
            picked.extend(extra)
            chosen_ids.update(id(example) for example in extra)
            if len(picked) >= k:
                break
    position = {id(example): index for index, example in enumerate(examples)}
    return sorted(picked, key=lambda example: position[id(example)])


class AgentLLMTracesJointSource(AgentLLMTracesCompressHistorySource):
    """True-joint source for agent-llm-traces.

    Subclasses the history-path source so sessions/spans/tool definitions are
    read identically: same parquet/jsonl discovery, same span sorting by
    (start_time, span_id), same history/current split at the last user message,
    same split-manifest args.  Per-session ``max_samples_per_session``
    sub-sampling is the parent's seeded uniform pick when
    ``require_tool_call=True`` (bit-identical to the existing arms); with
    ``require_tool_call=False`` it is position-stratified and action-balanced
    (``_stratified_pick``, target tool-call share ``action_tool_call_frac``).
    The parsing addition is that tool definitions (first span of the session
    carrying ``gen_ai.tool.definitions``) are rendered into per-tool documents
    with the unified path's variant policy.  Rendering is PER EXAMPLE: the
    compressed pool is the target-inclusive bounded subset from
    ``_select_tools`` (default 32 tools), because full session toolsets
    (e.g. AppWorld) exceed any reasonable document budget; spans seen before
    the session's first tool definitions, and sessions without tool
    definitions at all, produce no examples.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        eval_ratio: float = 0.1,
        split_seed: int = 42,
        split_manifest_file: Optional[str] = None,
        split_manifest_name: str = "subset_disjoint",
        max_samples_per_session: Optional[int] = 4,
        max_records: Optional[int] = None,
        require_tool_call: bool = False,
        max_input_chars: Optional[int] = None,
        max_answer_chars: Optional[int] = None,
        prefix_history_doc_num: Optional[int] = None,
        prefix_history_exact: bool = False,
        canonical_format_prob: float = 0.7,
        minified_json_prob: float = 0.2,
        shuffle_tools: bool = True,
        truncate_description_chars: int = 600,
        max_tools_per_sample: int = 32,
        same_namespace_negative_tools: int = 8,
        random_negative_tools: int = 24,
        action_tool_call_frac: float = 0.75,
    ) -> None:
        # Set joint knobs BEFORE super().__init__(): the parent constructor
        # calls self._load_records(), which dispatches to the overridden
        # _session_examples below.
        self.canonical_format_prob = canonical_format_prob
        self.minified_json_prob = minified_json_prob
        self.shuffle_tools = shuffle_tools
        self.truncate_description_chars = truncate_description_chars
        self.max_tools_per_sample = max_tools_per_sample
        self.same_namespace_negative_tools = same_namespace_negative_tools
        self.random_negative_tools = random_negative_tools
        # Target tool-call share for the per-session down-sampling; consulted
        # only when require_tool_call=False (True keeps the legacy uniform
        # pick, so existing arms are bit-identical).
        self.action_tool_call_frac = action_tool_call_frac
        super().__init__(
            path=path,
            split=split,
            eval_ratio=eval_ratio,
            split_seed=split_seed,
            split_manifest_file=split_manifest_file,
            split_manifest_name=split_manifest_name,
            max_samples_per_session=max_samples_per_session,
            max_records=max_records,
            require_tool_call=require_tool_call,
            max_input_chars=max_input_chars,
            max_answer_chars=max_answer_chars,
            include_tools=True,
            prefix_history_doc_num=prefix_history_doc_num,
            prefix_history_exact=prefix_history_exact,
        )

    # The two loaders below are copies of the parent methods; the ONLY change
    # is that `subset` is threaded into `_session_examples` so JointExample can
    # record it (the parent drops subset when calling its own hook).

    def _load_records(self) -> List[JointExample]:
        data_files = _find_agent_parquet_files(self.path)
        jsonl_files: List[Path] = []
        if not data_files:
            jsonl_files = _find_agent_jsonl_files(self.path)
        if not data_files and not jsonl_files:
            raise FileNotFoundError(f"No parquet/jsonl files found under {self.path}")
        if self.split_manifest_file:
            return self._load_records_from_manifest(data_files or jsonl_files)
        sessions = []
        if data_files:
            for row_index, row in enumerate(_iter_agent_rows(data_files)):
                session_id = str(
                    row.get("session_id")
                    or row.get("trace_id")
                    or row.get("id")
                    or f"row-{row_index}"
                )
                subset = str(row.get("benchmark") or row.get("subset") or row.get("dataset") or row.get("task") or "unknown")
                spans = _sort_agent_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
                sessions.append({"session_id": session_id, "subset": subset, "spans": spans})
        else:
            for row_index, row in enumerate(_iter_agent_jsonl_rows(jsonl_files)):
                session = _toolathlon_row_to_agent_session(row, row_index)
                if session is not None:
                    sessions.append(session)

        train_ids, eval_ids = self._split_session_ids(sessions)
        keep_ids = train_ids if self.split == "train" else eval_ids
        rng = random.Random(self.split_seed + (0 if self.split == "train" else 1))
        records: List[JointExample] = []
        for session in sessions:
            if session["session_id"] not in keep_ids:
                continue
            examples = self._session_examples(session["session_id"], session["spans"], session["subset"])
            if self.max_samples_per_session and len(examples) > self.max_samples_per_session:
                if self.require_tool_call:
                    # Legacy uniform pick: bit-identical to the pre-change
                    # behavior the existing require_tool_call=True arms ran on.
                    examples = rng.sample(examples, self.max_samples_per_session)
                else:
                    examples = _stratified_pick(
                        examples,
                        self.max_samples_per_session,
                        rng,
                        action_tool_call_frac=self.action_tool_call_frac,
                    )
            records.extend(examples)
            if self.max_records is not None and len(records) >= self.max_records:
                return records[: self.max_records]
        return records

    def _load_records_from_manifest(self, data_files: Sequence[Path]) -> List[JointExample]:
        manifest = json.loads(Path(self.split_manifest_file).read_text(encoding="utf-8"))
        if "train_session_ids" in manifest and "eval_session_ids" in manifest:
            selected = manifest
        else:
            selected = manifest[self.split_manifest_name]
        keep_ids = {
            str(item)
            for item in selected.get(
                "train_session_ids" if self.split == "train" else "eval_session_ids",
                [],
            )
        }
        rng = random.Random(self.split_seed + (0 if self.split == "train" else 1))
        records: List[JointExample] = []
        row_iter = _iter_agent_rows(data_files) if data_files and data_files[0].suffix == ".parquet" else _iter_agent_jsonl_rows(data_files)
        for row_index, row in enumerate(row_iter):
            session_id = str(
                row.get("session_id")
                or row.get("trace_id")
                or row.get("id")
                or f"row-{row_index}"
            )
            subset = str(row.get("benchmark") or row.get("subset") or row.get("dataset") or row.get("task") or "unknown")
            if data_files and data_files[0].suffix == ".jsonl":
                session = _toolathlon_row_to_agent_session(row, row_index)
                if session is None:
                    continue
                session_id = session["session_id"]
                subset = session["subset"]
                spans = session["spans"]
            else:
                spans = _sort_agent_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
            if session_id not in keep_ids:
                continue
            examples = self._session_examples(session_id, spans, subset)
            if self.max_samples_per_session and len(examples) > self.max_samples_per_session:
                if self.require_tool_call:
                    # Legacy uniform pick: bit-identical to the pre-change
                    # behavior the existing require_tool_call=True arms ran on.
                    examples = rng.sample(examples, self.max_samples_per_session)
                else:
                    examples = _stratified_pick(
                        examples,
                        self.max_samples_per_session,
                        rng,
                        action_tool_call_frac=self.action_tool_call_frac,
                    )
            records.extend(examples)
            if self.max_records is not None and len(records) >= self.max_records:
                return records[: self.max_records]
        return records

    def _session_examples(
        self,
        session_id: str,
        spans: Sequence[Any],
        subset: str = "unknown",
    ) -> List[JointExample]:
        examples: List[JointExample] = []
        tools: List[Dict[str, Any]] = []
        for span_index, span in enumerate(spans):
            attributes = _span_attributes(span)
            if not tools:
                tools = _tool_list_from_agent_value(attributes.get("gen_ai.tool.definitions"))
            if not tools:
                # No tool definitions seen yet (or none in the session): the
                # joint task needs both document types, so skip these spans.
                continue
            raw_input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
            output_messages = attributes.get("gen_ai.output.messages")
            if not raw_input_messages or output_messages is None:
                continue
            if self.max_input_chars is not None and len(str(raw_input_messages)) > self.max_input_chars:
                continue
            system_prompt = _agent_system_prompt(_json_loads(raw_input_messages, []))
            messages = [
                item
                for item in (_normal_agent_message(message) for message in _json_loads(raw_input_messages, []))
                if item is not None and item.get("role") != "system"
            ]
            last_user_index = next(
                (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
                None,
            )
            if last_user_index is None:
                continue
            history_docs = _agent_history_turn_docs(messages[:last_user_index])
            # Same traversal, same turn boundaries: units[i] renders back to
            # history_docs[i]["content"] (locked by test_chunk_policy.py).
            history_units = _agent_history_turn_units(messages[:last_user_index])
            if self.prefix_history_doc_num is not None:
                if len(history_docs) < self.prefix_history_doc_num:
                    continue
                if self.prefix_history_exact and len(history_docs) != self.prefix_history_doc_num:
                    continue
                history_docs = history_docs[-self.prefix_history_doc_num :]
                history_units = history_units[-self.prefix_history_doc_num :]
            current_messages = messages[last_user_index:]
            answer, has_tool_call = _render_agent_output_messages(output_messages, self.max_answer_chars)
            if self.require_tool_call and not has_tool_call:
                continue
            # Inline <think> residue (reasoning embedded in the content
            # string, not a dedicated reasoning key) is training-surface
            # noise; strip it AFTER the require_tool_call filter so that
            # filter's predicate stays bit-identical.
            answer = _strip_think_blocks(answer)
            if not history_docs or not current_messages or not answer:
                continue
            # Per-example tool pool: target-inclusive bounded subset so large
            # session toolsets (e.g. AppWorld) stay within the doc budget.
            target_tool = _first_tool_call_name(output_messages)
            rng = random.Random(f"{self.split_seed}:{session_id}:{span_index}:tools")
            selected_tools = _select_tools(
                tools,
                target_tool,
                rng,
                max_tools_per_sample=self.max_tools_per_sample,
                same_namespace_negative_tools=self.same_namespace_negative_tools,
                random_negative_tools=self.random_negative_tools,
            )
            tool_documents, target_doc_index = _render_tool_documents(
                selected_tools,
                rng,
                canonical_format_prob=self.canonical_format_prob,
                minified_json_prob=self.minified_json_prob,
                shuffle_tools=self.shuffle_tools,
                truncate_description_chars=self.truncate_description_chars,
                target_tool=target_tool,
            )
            examples.append(
                JointExample(
                    qid=f"{session_id}:{span_index}",
                    session_id=session_id,
                    tool_documents=list(tool_documents),
                    history_documents=[str(doc.get("content") or "") for doc in history_docs],
                    current_messages=current_messages,
                    answer=answer,
                    system_prompt=system_prompt,
                    subset=subset,
                    target_tool=target_tool,
                    target_tool_doc_index=target_doc_index,
                    history_units=[list(units) for units in history_units],
                    action_type="tool_call" if has_tool_call else "other",
                    # ``_select_tools`` returns the target FIRST (it seeds the
                    # pool with ``target[:1]``).  ``tools_in_system`` renders
                    # this list verbatim into the system prefix, so leaving the
                    # order as returned would put the gold tool at position 0
                    # of every over-budget example -- a positional oracle no
                    # serving path ever offers (BFCL / tau2 / ToolSandbox ship
                    # the caller's own tool order).  Shuffle on a dedicated RNG
                    # stream so the grid path's ``rng`` consumption, and hence
                    # every doc_mode != history_only arm, stays bit-identical.
                    selected_tools=_shuffled_system_tools(
                        selected_tools, self.split_seed, session_id, span_index
                    ),
                )
            )
        return examples


def _default_max_tool_chunks(max_doc_num: int) -> int:
    return max(1, (2 * max_doc_num) // 3)


def _history_chunk_budget(
    doc_mode: str,
    max_doc_num: int,
    max_tool_chunks: int,
    num_tool_chunks: int,
    per_side_caps: bool,
    has_tool_documents: bool = True,
) -> int:
    """History-side slot budget for one example.

    ``per_side_caps=True``: a CONSTANT ``max_doc_num - min(max_tool_chunks,
    max_doc_num)`` in every doc mode, so the history-side presented budget is
    identical across ``joint``/``history_only`` (and across the J-arms that
    train them) — the G-Q3 fairness constraint.  Spare tool slots are NOT
    recycled into history — EXCEPT (v2 empty-tool reclaim) when the example
    has no non-empty tool documents at all (the QA family): then there is no
    tool side to stay fair against, ``tool_cap`` is 0, and history gets the
    full ``max_doc_num`` grid.  Tool-bearing examples are bit-for-bit
    identical to v1.

    ``per_side_caps=False`` (legacy, pre-fix behavior): ``history_only`` gets
    all ``max_doc_num`` slots and ``joint`` gets ``max_doc_num -
    num_tool_chunks`` (spare tool slots recycled).
    """
    if per_side_caps:
        if not has_tool_documents:
            return max_doc_num
        return max(0, max_doc_num - min(max_tool_chunks, max_doc_num))
    return max_doc_num if doc_mode == "history_only" else max(0, max_doc_num - num_tool_chunks)


def build_tool_chunks(
    tokenizer,
    example: JointExample,
    doc_mode: str,
    *,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_chunks: Optional[int],
    max_tool_definition_tokens: int,
    per_side_caps: bool = True,
) -> tuple[Optional[List[List[int]]], Optional[str], Dict[str, Any]]:
    """Chat-template, chunk and budget-truncate one example's tool documents.

    Shared by ``JointDataset.preprocess_example`` (training) and the eval
    driver's ``_condition_doc_chunks`` so the two sides cannot drift.

    Cap semantics — ``per_side_caps=True`` (default): the tool side gets
    ``min(max_tool_chunks, max_doc_num)`` slots in BOTH ``joint`` and
    ``tool_only`` modes, so the tool-side presented budget is identical
    across doc modes and J-arms.  ``per_side_caps=False`` reproduces the
    pre-fix behavior (``tool_only`` gets all ``max_doc_num`` slots).

    v2 empty-tool reclaim (``per_side_caps=True`` only): an example with NO
    non-empty tool documents (the QA family) gets ``tool_cap = 0`` — reserving
    dead tool slots would starve its history side (the P0-1 audit finding).
    Tool-bearing examples are bit-for-bit identical to v1, so the eval
    driver (``_condition_doc_chunks``), whose appworld examples always carry
    tool schemas, is unaffected by the change.

    Truncation keeps the target tool's schema: when
    ``example.target_tool_doc_index`` is known and the chunk list exceeds the
    cap, every chunk of the target document is selected first and the
    remaining slots fill with the other chunks in original order (relative
    order preserved, so no position shortcut is introduced).  The pre-fix
    behavior — a plain head-truncate after the render-time shuffle — dropped
    the target schema from roughly half the saturated joint examples.

    Returns ``(tool_chunks, skip_reason, meta)``.  ``meta`` reports
    ``target_known`` / ``target_in_grid`` (``None`` when the target document
    is unknown) / ``num_chunks_before_cap`` / ``tool_cap``.
    """
    if max_tool_chunks is None:
        max_tool_chunks = _default_max_tool_chunks(max_doc_num)
    meta: Dict[str, Any] = {
        "target_known": example.target_tool_doc_index is not None,
        "target_in_grid": None,
        "target_truncated_to_cap": False,
        "num_chunks_before_cap": 0,
        "tool_cap": 0,
    }
    if doc_mode == "history_only":
        return [], None, meta
    if per_side_caps:
        tool_cap = min(max_tool_chunks, max_doc_num)
        if not _has_tool_documents(example):
            # v2 empty-tool reclaim: no tool side -> reserve no tool slots.
            tool_cap = 0
    else:
        tool_cap = max_doc_num if doc_mode == "tool_only" else min(max_tool_chunks, max_doc_num)
    meta["tool_cap"] = tool_cap

    doc_id_groups: List[List[int]] = []
    doc_source_indices: List[int] = []
    for source_index, document in enumerate(example.tool_documents):
        if not document.strip():
            continue
        doc_id_groups.append(
            _chat_template_ids(
                tokenizer,
                [{"role": "user", "content": TOOL_DOC_PREFIX + document}],
            )
        )
        doc_source_indices.append(source_index)
    doc_tokens = sum(len(doc_ids) for doc_ids in doc_id_groups)
    if doc_tokens > max_tool_definition_tokens:
        return None, f"tool_definition_tokens>{max_tool_definition_tokens}", meta

    flat: List[tuple[int, List[int]]] = []
    for doc_ids, source_index in zip(doc_id_groups, doc_source_indices):
        for start in range(0, len(doc_ids), max_doc_length):
            flat.append((source_index, doc_ids[start : start + max_doc_length]))
    meta["num_chunks_before_cap"] = len(flat)

    target_index = example.target_tool_doc_index
    target_positions = (
        [position for position, (source_index, _) in enumerate(flat) if source_index == target_index]
        if target_index is not None
        else []
    )
    if len(flat) <= tool_cap:
        keep = set(range(len(flat)))
    elif target_index is None or not per_side_caps:
        # Legacy mode reproduces the pre-fix behavior bit-for-bit (plain
        # head-truncation, target retention included) so old runs can be
        # measured/diffed; target-unknown examples have nothing to retain.
        keep = set(range(tool_cap))
    else:
        keep = set(target_positions[:tool_cap])
        for position in range(len(flat)):
            if len(keep) >= tool_cap:
                break
            keep.add(position)
    if target_index is not None and tool_cap > 0:
        # target_in_grid=True means the target schema is FULLY present.  A
        # target document that alone chunks into more than tool_cap pieces is
        # retained up to the whole cap and flagged target_truncated_to_cap —
        # a data condition (schema larger than the entire tool budget), not a
        # retention failure; the constructor invariant must not fire on it.
        kept_target = [position for position in target_positions if position in keep]
        meta["target_in_grid"] = bool(target_positions) and len(kept_target) == len(target_positions)
        meta["target_truncated_to_cap"] = (
            bool(kept_target) and not meta["target_in_grid"] and len(kept_target) >= tool_cap
        )
    tool_chunks = [flat[position][1] for position in sorted(keep)]
    return tool_chunks, None, meta


# ---------------------------------------------------------------------------
# History-side chunking (shared by the trainer and the eval driver).
# ---------------------------------------------------------------------------


def _encode_fn(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode_fn(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)


def _history_pairs(example: JointExample) -> tuple[List[Message], List[List[Dict[str, str]]]]:
    """Non-empty history docs and their unit lists, filtered in lockstep.

    ``history_units`` must stay index-aligned with the raw history messages
    (``FrozenDoc.turn_index`` indexes both), so the empty-document filter has
    to drop the same positions from both lists.
    """

    units_source = example.history_units or []
    raw_history: List[Message] = []
    history_units: List[List[Dict[str, str]]] = []
    for index, text in enumerate(example.history_documents):
        if not text or not text.strip():
            continue
        raw_history.append({"role": "user", "content": text})
        history_units.append(list(units_source[index]) if index < len(units_source) else [])
    return raw_history, history_units


def build_history_chunks(
    tokenizer,
    example: JointExample,
    doc_mode: str,
    *,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_chunks: Optional[int],
    num_tool_chunks: int,
    per_side_caps: bool,
    history_selection: HistorySelection,
    split_oversized_history_docs: bool,
    chunk_policy: str = "agent-turn",
    delay_recent_turns: int = 0,
    has_tool_documents: Optional[bool] = None,
    need_content_tokens: bool = False,
) -> tuple[List[Message], List[Message], Dict[str, Any]]:
    """Chunk one example's history side under ``chunk_policy``.

    Shared by ``JointDataset.preprocess_example`` (training) and the eval
    driver's ``_condition_doc_chunks`` so the two sides cannot drift — the
    same constraint ``build_tool_chunks`` already satisfies for the tool side.

    Returns ``(kept_messages, delayed_messages, meta)``.  ``kept_messages`` go
    into the compressed context grid; ``delayed_messages`` are the last
    ``delay_recent_turns`` turns' docs, which the caller prepends to the plain
    prompt (the ``full_history_doc_num`` semantics of
    ``train_data_multiturn.py:1210-1224``, ported to the joint path).

    Fast path: ``chunk_policy == "agent-turn" and delay_recent_turns == 0``
    short-circuits to today's ``_fit_reused_history`` call, so the default
    pipeline is bit-identical to what it produced before this module existed.

    ``need_content_tokens`` is opt-in: measuring ``content_tokens`` costs one
    extra encode of the whole selected history (up to ~24k tokens per example),
    which would roughly DOUBLE the history-side tokenization cost of every
    dataset build.  The trainer only ever logged the number, so it passes
    False and gets ``content_tokens = policy_content_tokens = None`` ("not
    measured", deliberately not 0).  The eval driver needs it for the
    presented-token / gist-declaration checks and passes True.
    """

    if max_tool_chunks is None:
        max_tool_chunks = _default_max_tool_chunks(max_doc_num)
    _unmeasured = 0 if need_content_tokens else None
    meta: Dict[str, Any] = {
        "chunk_policy": chunk_policy,
        "delay_recent_turns": delay_recent_turns,
        "content_tokens": _unmeasured,
        "policy_content_tokens": _unmeasured,
        "history_chunk_count": 0,
        "structural_fallback_docs": 0,
        "structural_partial_docs": 0,
        "delayed_docs": 0,
        "history_docs_total": 0,
        "history_kept_source_indices": [],
    }
    kind, chunk_size = parse_chunk_policy(chunk_policy)
    if delay_recent_turns < 0:
        raise ValueError(f"delay_recent_turns must be non-negative, got {delay_recent_turns}")
    if kind == "fixed" and delay_recent_turns > 0:
        raise ValueError(
            f"chunk_policy={chunk_policy!r} destroys turn boundaries; "
            "--delay_recent_turns > 0 is only defined for agent-turn/structural"
        )
    if doc_mode == "tool_only":
        return [], [], meta

    raw_history, history_units = _history_pairs(example)
    meta["history_docs_total"] = len(raw_history)
    history_budget = _history_chunk_budget(
        doc_mode,
        max_doc_num,
        max_tool_chunks,
        num_tool_chunks,
        per_side_caps,
        has_tool_documents=(
            _has_tool_documents(example)
            if has_tool_documents is None
            else has_tool_documents
        ),
    )
    if history_budget <= 0 or not raw_history:
        return [], [], meta

    if kind == "agent-turn" and delay_recent_turns == 0:
        history, kept_source_indices = _fit_reused_history_with_indices(
            tokenizer,
            raw_history,
            max_doc_length=max_doc_length,
            max_doc_num=history_budget,
            policy=history_selection,
            split_oversized_history_docs=split_oversized_history_docs,
        )
        # Same measure as chunk_policy._content_tokens (same join, same texts),
        # so ``content_tokens`` is comparable between the fast path and every
        # other arm.  Skipped unless the caller asked for it: this is the hot
        # path (every trainer dataset build runs it) and the encode is a full
        # second pass over the history text.
        content_tokens = None
        if need_content_tokens:
            content_tokens = (
                len(
                    _encode_fn(
                        tokenizer,
                        FROZEN_JOIN.join(message["content"] for message in history),
                    )
                )
                if history
                else 0
            )
        meta.update(
            content_tokens=content_tokens,
            policy_content_tokens=content_tokens,
            history_chunk_count=len(history),
            history_kept_source_indices=sorted(set(kept_source_indices)),
        )
        return history, [], meta

    frozen_docs = fit_history_with_provenance(
        tokenizer,
        raw_history,
        max_doc_length=max_doc_length,
        max_doc_num=history_budget,
        policy=history_selection,
        split_oversized_history_docs=split_oversized_history_docs,
        split_fn=_split_message_to_fit,
        select_fn=_select_history,
        token_len_fn=_message_token_length,
    )
    policy_docs, policy_meta = apply_policy(
        tokenizer,
        frozen_docs,
        history_units,
        kind,
        chunk_size,
        max_doc_length,
        _encode_fn,
        _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
        need_content_tokens=need_content_tokens,
    )
    kept, delayed = split_delayed(
        policy_docs,
        frozen_docs,
        delay_recent_turns,
        doc_turn_indices=policy_meta["doc_turn_indices"],
    )
    meta.update(
        content_tokens=policy_meta["content_tokens"],
        policy_content_tokens=policy_meta["policy_content_tokens"],
        history_chunk_count=len(kept),
        structural_fallback_docs=policy_meta["structural_fallback_docs"],
        structural_partial_docs=policy_meta["structural_partial_docs"],
        structural_repacked_docs=policy_meta["structural_repacked_docs"],
        structural_passthrough_docs=policy_meta["structural_passthrough_docs"],
        fixed_window_tokens=policy_meta["fixed_window_tokens"],
        delayed_docs=len(delayed),
        history_kept_source_indices=sorted(
            {
                int(turn_index)
                for turn_index in policy_meta["doc_turn_indices"]
                if turn_index is not None
            }
        ),
    )
    return kept, delayed, meta


class JointDataset:
    """Convert JointExample records into GistMultiDocTrainer format.

    List-backed dataset mirroring ``AgentToolDefinitionDataset`` (same output
    keys/shapes) but without importing torch: HF's Trainer only needs
    ``__len__``/``__getitem__`` plus the collator.

    Output keys: ``system_input_ids`` (padded to ``max_system_length`` with
    -100), ``context_input_ids`` (flat ``max_doc_num * max_doc_length`` grid,
    -100 padded; tool chunks first, then history chunks in chronological
    order), ``input_ids``, ``labels`` (prompt masked -100, answer+EOS
    supervised), ``attention_mask``, ``dynamic``.

    Regime-first knobs (both default OFF; with them unset every feature is
    byte-identical to the pre-knob code path):

    - ``tools_in_system`` (``doc_mode="history_only"`` only): render the tool
      schemas RAW into the system prefix via the chat template's ``tools=``
      — the dialect every serving path actually runs — instead of putting them
      through the gist grid.  History then gets the full ``max_doc_num``
      grid, and an over-long prefix is skipped (``system_overflow``) rather
      than truncated.
    - ``hybrid_tail_choices``: per-example raw-tail depth k, drawn
      deterministically from the example qid.  The last k fitted history
      documents leave the grid and are prepended RAW to the prompt (the
      serving "hybrid" arm); at least ``min_doc_num`` documents stay
      compressed, and the tail is further shortened (oldest raw doc first)
      until the current turn and the answer still fit ``max_length``.
    """

    def __init__(
        self,
        examples: Sequence[JointExample],
        tokenizer,
        max_length: int = 1024,
        max_doc_length: int = 1024,
        min_doc_num: int = 2,
        max_doc_num: int = 10,
        max_system_length: int = 2048,
        history_selection: HistorySelection = "tail",
        doc_mode: DocMode = "joint",
        max_tool_chunks: Optional[int] = None,
        max_tool_definition_tokens: int = 32000,
        split_oversized_history_docs: bool = True,
        per_side_caps: bool = True,
        chunk_policy: str = "agent-turn",
        delay_recent_turns: int = 0,
        tools_in_system: bool = False,
        hybrid_tail_choices: Optional[Sequence[int]] = None,
    ) -> None:
        if doc_mode not in ("joint", "tool_only", "history_only"):
            raise ValueError(f"Unsupported doc_mode: {doc_mode!r}")
        if tools_in_system and doc_mode != "history_only":
            # Tools would be presented twice (raw system prefix AND gist grid).
            raise ValueError(
                f"tools_in_system=True requires doc_mode='history_only', got {doc_mode!r}"
            )
        tail_choices: List[int] = [int(value) for value in (hybrid_tail_choices or [])]
        if any(value < 0 for value in tail_choices):
            raise ValueError(f"hybrid_tail_choices must be non-negative: {tail_choices!r}")
        self.tools_in_system = tools_in_system
        self.hybrid_tail_choices = list(tail_choices)
        self.max_doc_length = max_doc_length
        self.min_doc_num = min_doc_num
        self.max_doc_num = max_doc_num
        self.max_system_length = max_system_length
        self.max_length = max_length
        self.doc_mode = doc_mode
        self.per_side_caps = per_side_caps
        self.chunk_policy = chunk_policy
        self.delay_recent_turns = delay_recent_turns
        self.data: List[Dict[str, Any]] = []
        skipped_by_reason: Counter[str] = Counter()
        skipped_by_family_reason: Counter[str] = Counter()
        target_known = 0
        target_in_grid = 0
        target_truncated = 0
        # QA retention audit (P0-1): source history docs / gold docs that
        # survived the grid budget, summed over emitted QA rows.
        qa_history_kept = 0
        qa_history_total = 0
        qa_gold_kept = 0
        qa_gold_total = 0
        qa_truncated_by_subset: Counter[str] = Counter()
        hybrid_tail_k_counts: Counter[int] = Counter()
        # Drawn k over ALL candidate examples (vs hybrid_tail_k_counts, which
        # counts EMITTED rows only): a stratum that is drawn but never emitted
        # is otherwise indistinguishable from one that was never drawn.
        hybrid_tail_k_drawn_counts: Counter[int] = Counter()
        # Rows presented through tools_in_system with no selected_tools (they
        # get a bare system prefix, i.e. NO tools at all).  Not a skip — QA
        # rows legitimately have none — but a silent mistrain risk worth a
        # counter in the manifest.
        tools_in_system_missing_tools = 0
        for example in examples:
            meta: Dict[str, Any] = {}
            hybrid_tail_k = (
                random.Random(f"{example.qid}:hybrid_tail").choice(tail_choices)
                if tail_choices
                else 0
            )
            hybrid_tail_k_drawn_counts[int(hybrid_tail_k)] += 1
            if tools_in_system and not example.selected_tools:
                tools_in_system_missing_tools += 1
            row, reason = self.preprocess_example(
                example,
                tokenizer=tokenizer,
                max_length=max_length,
                max_doc_length=max_doc_length,
                min_doc_num=min_doc_num,
                max_doc_num=max_doc_num,
                max_system_length=max_system_length,
                history_selection=history_selection,
                doc_mode=doc_mode,
                max_tool_chunks=max_tool_chunks,
                max_tool_definition_tokens=max_tool_definition_tokens,
                split_oversized_history_docs=split_oversized_history_docs,
                per_side_caps=per_side_caps,
                chunk_policy=chunk_policy,
                delay_recent_turns=delay_recent_turns,
                tools_in_system=tools_in_system,
                hybrid_tail_k=hybrid_tail_k,
                meta_out=meta,
            )
            if row is None:
                skipped_by_reason[reason] += 1
                skipped_by_family_reason[f"{qid_source_family(example.qid)}:{reason}"] += 1
                continue
            self.data.append(row)
            hybrid_tail_k_counts[int(meta.get("hybrid_tail_k") or 0)] += 1
            if doc_mode != "history_only" and meta.get("target_known") and meta.get("tool_cap", 0) > 0:
                target_known += 1
                if meta.get("target_in_grid"):
                    target_in_grid += 1
                elif meta.get("target_truncated_to_cap"):
                    target_truncated += 1
            if doc_mode != "tool_only" and qid_source_family(example.qid) == "qa":
                kept_sources = set(meta.get("history_kept_source_indices") or [])
                total_docs = int(meta.get("history_docs_total") or 0)
                qa_history_kept += len(kept_sources)
                qa_history_total += total_docs
                if len(kept_sources) < total_docs:
                    qa_truncated_by_subset[str(example.subset)] += 1
                gold = example.gold_history_doc_indices
                if gold:
                    qa_gold_total += len(gold)
                    qa_gold_kept += sum(1 for index in gold if index in kept_sources)
        self.target_stats = {
            "target_known": target_known,
            "target_in_grid": target_in_grid,
            "target_truncated_to_cap": target_truncated,
        }
        self.skipped_by_reason = dict(skipped_by_reason)
        # Per-family skip breakdown (P1-7): makes the alternate arm's
        # tool_only-pass QA skips (``qa:doc_num<2``) explicit instead of an
        # inflated aggregate skip count.
        self.skipped_by_family_reason = dict(skipped_by_family_reason)
        # Realized hybrid raw-tail depth over the EMITTED rows (always {} ->
        # {0: n} when the knob is off), for the manifest's per-pass audit.
        self.hybrid_tail_k_counts = dict(sorted(hybrid_tail_k_counts.items()))
        # Drawn k over all candidate examples (emitted + skipped).  A
        # drawn-vs-realized mismatch means a stratum is being lost to skips.
        self.hybrid_tail_k_drawn_counts = dict(sorted(hybrid_tail_k_drawn_counts.items()))
        self.tools_in_system_missing_tools = tools_in_system_missing_tools
        self.qa_retention_stats = {
            "qa_history_doc_retention": {"kept": qa_history_kept, "total": qa_history_total},
            "qa_gold_doc_retention": {"kept": qa_gold_kept, "total": qa_gold_total},
            "qa_history_truncated_examples_by_subset": dict(qa_truncated_by_subset),
        }
        # With per_side_caps the target-preserving truncation makes this an
        # invariant: a target-known row may be fully present or (when the
        # schema alone exceeds the whole tool budget) retained up to the cap —
        # but never silently ABSENT.  A violation means the retention logic
        # regressed.
        if per_side_caps and doc_mode != "history_only" and target_in_grid + target_truncated < target_known:
            raise AssertionError(
                f"target tool schema missing from the context grid for "
                f"{target_known - target_in_grid - target_truncated}/{target_known} examples "
                f"(doc_mode={doc_mode})"
            )
        logger.info(
            "Built %d joint samples (%s, per_side_caps=%s); skipped %d by reason=%s; "
            "target schema fully in grid for %d/%d target-known rows (%d truncated to the full cap); "
            "qa retention=%s",
            len(self.data),
            doc_mode,
            per_side_caps,
            sum(skipped_by_reason.values()),
            dict(skipped_by_reason),
            target_in_grid,
            target_known,
            target_truncated,
            self.qa_retention_stats,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]

    @staticmethod
    def preprocess_example(
        example: JointExample,
        tokenizer,
        max_length: int,
        max_doc_length: int,
        min_doc_num: int,
        max_doc_num: int,
        max_system_length: int,
        history_selection: HistorySelection = "tail",
        doc_mode: DocMode = "joint",
        max_tool_chunks: Optional[int] = None,
        max_tool_definition_tokens: int = 32000,
        split_oversized_history_docs: bool = True,
        per_side_caps: bool = True,
        chunk_policy: str = "agent-turn",
        delay_recent_turns: int = 0,
        tools_in_system: bool = False,
        hybrid_tail_k: int = 0,
        meta_out: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if doc_mode not in ("joint", "tool_only", "history_only"):
            raise ValueError(f"Unsupported doc_mode: {doc_mode!r}")
        if tools_in_system and doc_mode != "history_only":
            raise ValueError(
                f"tools_in_system=True requires doc_mode='history_only', got {doc_mode!r}"
            )
        if hybrid_tail_k < 0:
            raise ValueError(f"hybrid_tail_k must be non-negative, got {hybrid_tail_k}")
        if max_tool_chunks is None:
            max_tool_chunks = _default_max_tool_chunks(max_doc_num)

        # ---- tool chunks (first in the grid; shared with the eval driver) --
        tool_chunks, tool_skip_reason, tool_meta = build_tool_chunks(
            tokenizer,
            example,
            doc_mode,
            max_doc_length=max_doc_length,
            max_doc_num=max_doc_num,
            max_tool_chunks=max_tool_chunks,
            max_tool_definition_tokens=max_tool_definition_tokens,
            per_side_caps=per_side_caps,
        )
        if meta_out is not None:
            meta_out.update(tool_meta)
        if tool_skip_reason is not None:
            return None, tool_skip_reason

        # ---- history chunks (chronological) -------------------------------
        history, delayed_history, history_meta = build_history_chunks(
            tokenizer,
            example,
            doc_mode,
            max_doc_length=max_doc_length,
            max_doc_num=max_doc_num,
            max_tool_chunks=max_tool_chunks,
            num_tool_chunks=len(tool_chunks),
            per_side_caps=per_side_caps,
            history_selection=history_selection,
            split_oversized_history_docs=split_oversized_history_docs,
            chunk_policy=chunk_policy,
            delay_recent_turns=delay_recent_turns,
            has_tool_documents=False if tools_in_system else None,
            # Training never reads content_tokens (it was log-only bookkeeping),
            # and measuring it doubles the history-side tokenization cost of the
            # whole dataset build.  The eval driver asks for it explicitly.
            need_content_tokens=False,
        )
        if meta_out is not None:
            meta_out.update(history_meta)
        history_kept_source_indices = list(history_meta["history_kept_source_indices"])
        num_raw_history_docs = int(history_meta["history_docs_total"])

        current = [
            _normal_chat_message(message)
            for message in example.current_messages
            if message.get("content") or message.get("role") == "assistant"
        ]
        doc_count = len(tool_chunks) + len(history)
        # ---- hybrid raw tail: the last k fitted history docs stay RAW ------
        # Mirrors train_data_multiturn.preprocess_example's full_history_doc_num
        # split: only the compressed prefix enters the context grid, the tail
        # is chat-template rendered and PREPENDED to the ordinary prompt.  At
        # least ``min_doc_num`` documents stay compressed, and ``doc_count``
        # (the min_doc_num gate) is computed on the pre-split history.
        # The tail is additionally capped by the SEQUENCE budget: k raw docs of
        # up to ``max_doc_length`` tokens each can be several times
        # ``max_length``, and the left-truncation below would then leave no room
        # for the supervised answer, dropping every large-k row as
        # ``tool_call_target_truncated``.  An over-long tail must SHORTEN
        # (shedding the OLDEST raw docs back into the compressed grid, degrading
        # towards k=0), never drop the row: the hybrid arm has to train on the
        # same example set as the arm it is paired against.
        realized_tail_k = 0
        raw_tail_ids: List[int] = []
        if hybrid_tail_k > 0 and history:
            realized_tail_k = min(hybrid_tail_k, max(0, len(history) - min_doc_num))
            if realized_tail_k > 0:
                candidate_tail = list(history[len(history) - realized_tail_k :])
                tail_id_lists = [
                    _chat_template_ids(tokenizer, [message], max_length=max_doc_length)
                    for message in candidate_tail
                ]
                # Same reserve the truncation below enforces: current turn +
                # answer (+EOS) must still fit into ``max_length``.
                current_ids_probe = (
                    _chat_template_ids(tokenizer, current, add_generation_prompt=True)
                    if current
                    else []
                )
                answer_reserve = (
                    len(tokenizer.encode(example.answer, add_special_tokens=False)) + 1
                    if example.answer
                    else 0
                )
                fixed_budget = len(current_ids_probe) + answer_reserve
                while tail_id_lists and (
                    sum(len(ids) for ids in tail_id_lists) + fixed_budget > max_length
                ):
                    tail_id_lists.pop(0)
                    realized_tail_k -= 1
                if realized_tail_k > 0:
                    raw_tail_ids = [
                        token_id for ids in tail_id_lists for token_id in ids
                    ]
                    history = list(history[: len(history) - realized_tail_k])
        if meta_out is not None:
            meta_out.update(
                num_tool_chunks=len(tool_chunks),
                # Fitted history depth BEFORE the hybrid split (the history
                # depth of the example); the grid occupancy after the split is
                # num_compressed_history_docs.
                num_history_docs=len(history) + realized_tail_k,
                num_compressed_history_docs=len(history),
                hybrid_tail_k=realized_tail_k,
                # History retention audit (P0-1): how many of the example's
                # non-empty source history documents survived the budget (a
                # split doc counts once).  Used by JointDataset's
                # qa_history/qa_gold retention counters.
                history_docs_total=num_raw_history_docs,
                history_kept_source_indices=sorted(set(history_kept_source_indices)),
            )
        if doc_count < min_doc_num:
            return None, f"doc_num<{min_doc_num}"
        if doc_count > max_doc_num:
            # Only reachable off the default policy: ``structural`` turns one
            # frozen doc into one doc per atomic block, so the doc count is no
            # longer bounded by the slot budget.  The training grid is a fixed
            # ``max_doc_num * max_doc_length`` block, so the row is skipped
            # rather than silently reshaped.  (The eval driver has no fixed
            # grid and keeps every chunk.)
            return None, f"doc_num>{max_doc_num}"
        if not current:
            return None, "empty_current"
        if not example.answer:
            return None, "empty_answer"

        # ---- system prefix -------------------------------------------------
        # Default: bare system prompt, NO tools= (the de-leak).
        # ``tools_in_system``: the regime every serving path actually runs —
        # tool schemas RAW in the system prefix, only history compressed.  The
        # prefix is rendered UNTRUNCATED and an over-long one is SKIPPED: HF's
        # right-truncation would silently delete the tail tools and the closing
        # template tokens, producing a malformed prefix instead of a loud skip.
        if tools_in_system:
            system_ids = _chat_template_ids(
                tokenizer,
                [{"role": "system", "content": example.system_prompt}],
                tools=example.selected_tools or None,
                keep_bos=True,
                max_length=None,
            )
            if meta_out is not None:
                meta_out["system_tokens"] = len(system_ids)
            if len(system_ids) > max_system_length:
                return None, "system_overflow"
        else:
            system_ids = _chat_template_ids(
                tokenizer,
                [{"role": "system", "content": example.system_prompt}],
                keep_bos=True,
                max_length=max_system_length,
            )
            if meta_out is not None:
                meta_out["system_tokens"] = len(system_ids)
        system_input_ids = _pad(system_ids, max_system_length, -100)

        # ---- flat context grid: tool chunks, then history chunks ----------
        context_input_ids: List[int] = []
        for chunk in tool_chunks:
            context_input_ids.extend(_pad(chunk, max_doc_length, -100))
        for message in history:
            doc_ids = _chat_template_ids(
                tokenizer,
                [message],
                max_length=max_doc_length,
            )
            context_input_ids.extend(_pad(doc_ids, max_doc_length, -100))
        empty_docs = max_doc_num - len(tool_chunks) - len(history)
        context_input_ids.extend([-100] * (max_doc_length * empty_docs))

        # ---- ordinary prompt (current turn) + supervised answer -----------
        # Delayed docs (the last ``delay_recent_turns`` turns) are NOT
        # compressed: they are prepended raw to the prompt, exactly as
        # ``full_history_doc_num`` does on the history path
        # (train_data_multiturn.py:1210-1224).
        delayed_ids: List[int] = []
        for message in delayed_history:
            delayed_ids.extend(
                _chat_template_ids(tokenizer, [message], max_length=max_doc_length)
            )
        prompt_ids = _chat_template_ids(
            tokenizer,
            current,
            add_generation_prompt=True,
        )
        prompt_ids = delayed_ids + raw_tail_ids + prompt_ids
        answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
        if not answer_ids:
            return None, "empty_answer_ids"
        answer_ids.append(tokenizer.eos_token_id)
        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[-(max_length - 1):]
        answer_budget = max_length - len(prompt_ids)
        if _answer_has_tool_call(example.answer) and len(answer_ids) > answer_budget:
            # A truncated tool-call JSON is a broken supervision target: drop
            # the example (counted as tool_call_target_truncated) instead of
            # training on a partial action.  Non-tool-call answers keep the
            # plain budget truncation below.
            return None, "tool_call_target_truncated"
        answer_ids = answer_ids[:answer_budget]
        if not answer_ids:
            return None, "empty_answer_budget"
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        pad_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_length
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length

        return {
            "system_input_ids": system_input_ids,
            "context_input_ids": context_input_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "dynamic": 0,
        }, "ok"


# ---------------------------------------------------------------------------
# Leakage / order self-checks (tests and --self_test only, not per-sample).
# ---------------------------------------------------------------------------

# Documents shorter than this after whitespace normalization are skipped: a
# tiny string (e.g. a bare tool name) can legitimately recur inside the answer
# (tool calls repeat the target tool name), so a substring check on it would
# false-positive.  Checks use an 80-char normalized probe of each document so
# truncation of the leaked text does not hide a leak.
_LEAK_MIN_PROBE_CHARS = 24
_LEAK_PROBE_CHARS = 80


def _normalize_ws(text: Any) -> str:
    return " ".join(str(text).split())


def _leak_probe(text: str) -> str:
    normalized = _normalize_ws(text)
    if len(normalized) < _LEAK_MIN_PROBE_CHARS:
        return ""
    return normalized[:_LEAK_PROBE_CHARS]


def _decode_real_ids(tokenizer, ids: Sequence[int]) -> str:
    real = [int(item) for item in ids if int(item) >= 0]
    return tokenizer.decode(real, skip_special_tokens=True) if real else ""


def assert_no_leakage(
    example: JointExample,
    features: Dict[str, Any],
    tokenizer,
    *,
    tools_in_system: bool = False,
    hybrid_tail_k: int = 0,
) -> None:
    """Assert a preprocessed example keeps every document in the context grid.

    Checks (all on whitespace-normalized decodings):

    - no tool-document probe appears in ``system_input_ids`` or ``input_ids``;
    - no history-document probe appears in ``input_ids``;
    - history documents that survive into the context grid appear in
      chronological source order;
    - ``labels`` supervise exactly the answer (+EOS) and nothing else.

    Documents dropped by budget truncation legitimately absent from the grid
    are not required to appear there; doc modes that exclude a document type
    by design therefore pass as well.

    The regime knobs move documents on purpose, so the caller must declare
    them: with ``tools_in_system`` the tool schemas legitimately live in
    ``system_input_ids``, and with ``hybrid_tail_k > 0`` the last k history
    documents legitimately live in ``input_ids``.  Both checks are relaxed
    accordingly (the hybrid case relaxes the history check entirely — which of
    the fitted documents ended up in the tail is not recoverable from
    ``features``).
    """

    system_text = _normalize_ws(_decode_real_ids(tokenizer, features["system_input_ids"]))
    attention_mask = features.get("attention_mask") or [1] * len(features["input_ids"])
    prompt_answer_ids = [
        token_id for token_id, mask in zip(features["input_ids"], attention_mask) if mask
    ]
    prompt_answer_text = _normalize_ws(_decode_real_ids(tokenizer, prompt_answer_ids))
    context_text = _normalize_ws(_decode_real_ids(tokenizer, features["context_input_ids"]))

    tool_probes = [probe for probe in (_leak_probe(doc) for doc in example.tool_documents) if probe]
    history_probes = [probe for probe in (_leak_probe(doc) for doc in example.history_documents) if probe]

    for probe in tool_probes:
        if probe in system_text and not tools_in_system:
            raise AssertionError(f"tool document leaked into system_input_ids: {probe!r}")
        if probe in prompt_answer_text:
            raise AssertionError(f"tool document leaked into input_ids: {probe!r}")
    if hybrid_tail_k <= 0:
        for probe in history_probes:
            if probe in prompt_answer_text:
                raise AssertionError(f"history document leaked into input_ids: {probe!r}")

    if (tool_probes or history_probes) and not any(
        int(token_id) >= 0 for token_id in features["context_input_ids"]
    ):
        raise AssertionError("context_input_ids is empty although the example has documents")

    positions: List[int] = []
    for probe in history_probes:
        position = context_text.find(probe)
        if position < 0:
            continue  # dropped by truncation/selection: legitimate
        positions.append(position)
    if positions != sorted(positions):
        raise AssertionError("history documents are out of chronological order in context_input_ids")

    labels = features["labels"]
    input_ids = features["input_ids"]
    real_length = sum(1 for mask in attention_mask if mask)
    real_labels = labels[:real_length]
    real_ids = input_ids[:real_length]
    supervised = [index for index, value in enumerate(real_labels) if value != -100]
    if not supervised:
        raise AssertionError("labels supervise no token")
    first = supervised[0]
    if real_labels[first:] != real_ids[first:]:
        raise AssertionError("supervised label region is not a contiguous copy of input_ids")
    supervised_ids = real_ids[first:]
    expected = tokenizer.encode(example.answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(supervised_ids) > len(expected):
        raise AssertionError("labels supervise more than answer+EOS")
    if supervised_ids != expected[: len(supervised_ids)]:
        raise AssertionError("supervised tokens are not a prefix of answer+EOS")
    if len(supervised_ids) == len(expected) and supervised_ids[-1] != tokenizer.eos_token_id:
        raise AssertionError("full answer is supervised but EOS is missing")
    if any(value != -100 for value in labels[real_length:]):
        raise AssertionError("padded positions must stay masked (-100)")


def assert_target_tool_in_grid(example: JointExample, features: Dict[str, Any], tokenizer) -> None:
    """Assert the target tool's schema document survived into the context grid.

    Only meaningful for doc modes that include tool documents (``joint`` /
    ``tool_only``); callers must not run it for ``history_only``.  No-op when
    the target document is unknown (``target_tool_doc_index is None``) or too
    short to probe.
    """
    if example.target_tool_doc_index is None:
        return
    probe = _leak_probe(example.tool_documents[example.target_tool_doc_index])
    if not probe:
        return
    context_text = _normalize_ws(_decode_real_ids(tokenizer, features["context_input_ids"]))
    if probe not in context_text:
        raise AssertionError(
            f"target tool document missing from context_input_ids: {probe!r}"
        )


# ---------------------------------------------------------------------------
# --self_test CLI (offline smoke check; also reused by the unit tests).
# ---------------------------------------------------------------------------


class _WhitespaceSelfTestTokenizer:
    """Deterministic whitespace tokenizer for --self_test and unit tests.

    Implements only the interface the dataset actually uses:
    ``apply_chat_template`` (with ``tools``/``add_generation_prompt``/
    ``max_length``/``truncation``/``enable_thinking`` kwargs), ``encode``,
    ``decode`` and the ``bos_token_id``/``eos_token_id``/``pad_token_id``
    attributes.  Template markers are ``<|...|>``-style pseudo tokens, skipped
    on decode like real special tokens.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._inverse: Dict[int, str] = {}
        for token, token_id in (("<pad>", 0), ("<bos>", 1), ("<eos>", 2)):
            self._vocab[token] = token_id
            self._inverse[token_id] = token

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            token_id = len(self._vocab)
            self._vocab[token] = token_id
            self._inverse[token_id] = token
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self._token_id(token) for token in str(text).split()]
        return [self.bos_token_id] + ids if add_special_tokens else ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for item in ids:
            item = int(item)
            if item < 0:
                continue
            token = self._inverse.get(item)
            if token is None:
                continue
            if skip_special_tokens and (token.startswith("<|") or item <= self.eos_token_id):
                continue
            tokens.append(token)
        return " ".join(tokens)

    def apply_chat_template(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "system" and tools:
                # Deterministic, whitespace-separable marker so leakage /
                # tools_in_system assertions can locate the rendered schemas.
                content = (
                    content
                    + "\n# Tools\n<TOOLS> "
                    + json.dumps(list(tools), ensure_ascii=False)
                    + " </TOOLS>"
                )
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        ids = self.encode("\n".join(parts))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


def _self_test_examples() -> List[JointExample]:
    tool_documents = [
        "<TOOL>\n<NAMESPACE> weather\n<NAME> get_weather\n"
        "<DESCRIPTION> Fetch the current weather for one city.\n<PARAMETERS>\n"
        '<PARAM name="city" type="string" required="true">\n</PARAMETERS>\n</TOOL>',
        "<TOOL>\n<NAMESPACE> files\n<NAME> search_files\n"
        "<DESCRIPTION> Search files under one directory path.\n<PARAMETERS>\n"
        '<PARAM name="path" type="string" required="true">\n</PARAMETERS>\n</TOOL>',
    ]
    history_documents = [
        "Previous turn\n[User query]\nList the files in /tmp please.\n[Assistant output]\n"
        "Action:\n<tool_call>\n{\"name\":\"search_files\",\"arguments\":{\"path\":\"/tmp\"}}\n</tool_call>",
        "Previous turn\n[User query]\nfound a.txt and b.txt under /tmp",
    ]
    return [
        JointExample(
            qid="self-test:0",
            session_id="self-test",
            tool_documents=tool_documents,
            history_documents=history_documents,
            current_messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
            answer="Action:\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
            system_prompt="You are a careful data agent.",
            subset="self-test",
            target_tool="get_weather",
            target_tool_doc_index=0,
        )
    ]


def _truncation_stress_example(num_tools: int = 12, target_index: int = 10) -> JointExample:
    """Synthetic saturated example: many tool docs, target near the tail.

    Under the pre-fix head-truncation the target document is dropped whenever
    ``target_index >= tool_cap``; the target-preserving truncation must keep
    it regardless of position.
    """
    tool_documents = [
        f"<TOOL>\n<NAMESPACE> ns{index}\n<NAME> tool_{index}\n"
        f"<DESCRIPTION> Synthetic distractor tool number {index} used only by the truncation stress self-test.\n"
        f'<PARAMETERS>\n<PARAM name="arg{index}" type="string" required="true">\n</PARAMETERS>\n</TOOL>'
        for index in range(num_tools)
    ]
    return JointExample(
        qid="self-test:truncation",
        session_id="self-test",
        tool_documents=tool_documents,
        history_documents=[
            "Previous turn\n[User query]\nPlease continue the synthetic stress task from before.",
            "Previous turn\n[User query]\nSecond synthetic history turn with enough words to probe.",
        ],
        current_messages=[{"role": "user", "content": f"Call tool_{target_index} now please."}],
        answer=(
            "Action:\n<tool_call>\n"
            f"{{\"name\":\"tool_{target_index}\",\"arguments\":{{\"arg{target_index}\":\"x\"}}}}\n</tool_call>"
        ),
        system_prompt="You are a careful data agent.",
        subset="self-test",
        target_tool=f"tool_{target_index}",
        target_tool_doc_index=target_index,
    )


def _run_self_test(tokenizer) -> None:
    max_system_length = 512
    for example in _self_test_examples():
        features, reason = JointDataset.preprocess_example(
            example,
            tokenizer=tokenizer,
            max_length=512,
            max_doc_length=256,
            min_doc_num=2,
            max_doc_num=8,
            max_system_length=max_system_length,
        )
        if features is None:
            raise RuntimeError(f"self-test example was dropped: {reason}")
        assert_no_leakage(example, features, tokenizer)
        assert_target_tool_in_grid(example, features, tokenizer)

        # Negative control: inject the tool documents into the system prefix
        # (the leak mode of the history path) and require detection.
        leaked_system_ids = _chat_template_ids(
            tokenizer,
            [{
                "role": "system",
                "content": example.system_prompt + "\n" + "\n".join(example.tool_documents),
            }],
            keep_bos=True,
            max_length=max_system_length,
        )
        tampered = dict(features)
        tampered["system_input_ids"] = _pad(leaked_system_ids, max_system_length, -100)
        try:
            assert_no_leakage(example, tampered, tokenizer)
        except AssertionError:
            pass
        else:
            raise RuntimeError("negative control failed: injected tool text was not detected")

    # Truncation stress: a saturated tool pool with the target near the tail.
    # Positive control: per-side caps + target-preserving truncation keep the
    # target schema in the grid.  Negative control: the legacy head-truncation
    # drops it and assert_target_tool_in_grid must detect that.
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    stress_kwargs = dict(
        tokenizer=tokenizer,
        max_length=512,
        max_doc_length=256,
        min_doc_num=2,
        max_doc_num=8,
        max_system_length=max_system_length,
    )
    for doc_mode in ("joint", "tool_only"):
        meta: Dict[str, Any] = {}
        features, reason = JointDataset.preprocess_example(
            stress, doc_mode=doc_mode, per_side_caps=True, meta_out=meta, **stress_kwargs
        )
        if features is None:
            raise RuntimeError(f"truncation stress example was dropped ({doc_mode}): {reason}")
        assert_no_leakage(stress, features, tokenizer)
        assert_target_tool_in_grid(stress, features, tokenizer)
        if meta.get("target_in_grid") is not True:
            raise RuntimeError(f"truncation stress: meta reports target_in_grid={meta.get('target_in_grid')!r}")
    legacy_features, reason = JointDataset.preprocess_example(
        stress, doc_mode="joint", per_side_caps=False, **stress_kwargs
    )
    if legacy_features is None:
        raise RuntimeError(f"legacy truncation stress example was dropped: {reason}")
    try:
        assert_target_tool_in_grid(stress, legacy_features, tokenizer)
    except AssertionError:
        pass
    else:
        raise RuntimeError(
            "negative control failed: legacy head-truncation kept the tail target "
            "(or assert_target_tool_in_grid stopped detecting the drop)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="True-joint C2KV dataset builder self-test")
    parser.add_argument("--self_test", action="store_true", help="run leakage/order self-checks on synthetic examples")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="local HF tokenizer path; defaults to the built-in whitespace self-test tokenizer",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("nothing to do; pass --self_test")
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = _WhitespaceSelfTestTokenizer()
    _run_self_test(tokenizer)
    print("self_test OK")


if __name__ == "__main__":
    main()
