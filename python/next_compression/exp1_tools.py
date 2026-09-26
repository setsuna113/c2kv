"""Exp 1: tool-allocation layouts on recorded decisions (torch-free part).

Section 3.2 of the paper admits the top-k schemas ranked by a fixed lexical
ranker as original text and covers the remainder with the compressed state.
The T0/T1 training corpora know two layouts only: every schema compressed
(T0) or structural fields native (T1).  This module adds the inference-time
layouts that the experiment compares on one frozen decision manifest:

  full       every schema native, no compressed remainder       reference
  uniform    every schema as one T0 document                    no originals
  hybrid     lexical top-k native, remainder as T0 documents    ours
  random     seeded random top-k native, remainder as T0 docs   ranker control
  retrieval  lexical-ranked schemas native while the resident
             payload fits the matched hybrid allowance          no remainder
  t1         structural fields native, descriptive fields gist  field-level
  snapkv, snapkv_hybrid, h2o, h2o_hybrid
             the full native prefix with an eviction remainder  (exp1_evict)

Every layout is packed by the training packer so that protocol text,
workspace, and chunking stay byte-identical to T0/T1 training.  The eviction
layouts reuse the ``full`` record together with the per-tool token spans
stored in its metadata.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from history_memory.dataset import Decision
from history_memory.events import EventStore

from .common import PreparedDecision, make_record
from .selection import canonical_calls, parse_calls
from .tools import (
    ToolPackingError,
    ToolPreparationConfig,
    ToolVariantMaterial,
    _json_snapshot,
    pack_tool_memory,
    tool_variant_material,
)

EXP1_SCHEMA = "next-compression-exp1-tools-v1"
EXP1_PURPOSE = "exp1_tools"
EXP1_RATIOS = (8, 12)
RANKER = "lexical-name4-text1-last-user-v1"

GIST_LAYOUTS = ("full", "uniform", "hybrid", "random", "retrieval", "t1")
EVICTION_LAYOUTS = ("snapkv", "snapkv_hybrid", "h2o", "h2o_hybrid")
LAYOUTS = GIST_LAYOUTS + EVICTION_LAYOUTS
SELECTED_LAYOUTS = ("hybrid", "random", "retrieval")
DECISION_TYPES = ("tool_call", "non_tool_response", "terminal_stop")


# ---------------------------------------------------------------------------
# Fixed lexical ranker (ported from the July tool-definition driver)
# ---------------------------------------------------------------------------


def tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def tool_search_text(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    fields = [
        tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
    ]
    return " ".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        for item in fields
        if item
    )


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def query_text(messages: Sequence[Mapping[str, Any]], scope: str = "last_user") -> str:
    """The ranker reads the latest user message, or every message for ``all``."""
    if scope == "all":
        return "\n".join(message_text(message) for message in messages)
    if scope != "last_user":
        raise ValueError(f"Unknown ranker scope: {scope!r}")
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message)
    return message_text(messages[-1]) if messages else ""


def _tokens(text: str) -> list[str]:
    # Verbatim from the July driver: identifiers stay whole, so ``book_flight``
    # only matches a query that contains ``book_flight``; descriptions and
    # parameter text supply the word-level overlap.
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def lexical_rank(tools: Sequence[Mapping[str, Any]], query: str) -> tuple[int, ...]:
    """Catalog order by descending score; name overlap weighs four text overlaps."""
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return tuple(range(len(tools)))
    scored = []
    for index, tool in enumerate(tools):
        name_overlap = len(query_tokens & set(_tokens(tool_name(tool))))
        text_overlap = len(query_tokens & set(_tokens(tool_search_text(tool))))
        scored.append((-(4.0 * name_overlap + float(text_overlap)), index))
    scored.sort()
    return tuple(index for _, index in scored)


def random_rank(count: int, *, seed: int, decision_id: str) -> tuple[int, ...]:
    """A per-decision seeded permutation; the same seed reproduces the control."""
    order = list(range(count))
    random.Random(f"{seed}:{decision_id}").shuffle(order)
    return tuple(order)


def decision_messages(decision: Decision) -> tuple[dict[str, Any], ...]:
    return tuple(message.to_dict() for message in decision.store.messages)


def gold_tool_calls(decision: Decision) -> list[dict[str, Any]]:
    calls = decision.target.to_dict().get("tool_calls") or []
    return canonical_calls(list(calls))


def decision_type(decision: Decision, *, last_source_index: int) -> str:
    if gold_tool_calls(decision):
        return "tool_call"
    if decision.source_message_index == last_source_index:
        return "terminal_stop"
    return "non_tool_response"


# ---------------------------------------------------------------------------
# Layout material and packing
# ---------------------------------------------------------------------------


def _t0_documents(
    snapshots: Sequence[Mapping[str, Any]], indices: Iterable[int]
) -> tuple[dict[str, Any], ...]:
    # Byte-identical to the T0 document format in tools.tool_variant_material.
    return tuple(
        {"type": "tool_definition", "tool_index": index, "tool": snapshots[index]}
        for index in indices
    )


def _validate_native_indices(count: int, native_indices: Sequence[int]) -> tuple[int, ...]:
    indices = tuple(int(index) for index in native_indices)
    if len(set(indices)) != len(indices) or any(not 0 <= index < count for index in indices):
        raise ValueError("native_indices must be distinct catalog positions")
    return tuple(sorted(indices))


def layout_material(
    tools: Sequence[Mapping[str, Any]],
    layout: str,
    *,
    native_indices: Sequence[int] = (),
) -> ToolVariantMaterial:
    """Native schemas keep catalog order; the remainder keeps its tool_index."""
    if not tools:
        raise ToolPackingError("no_tool_definitions")
    if layout == "uniform":
        return tool_variant_material(tools, "T0")
    if layout == "t1":
        return tool_variant_material(tools, "T1")
    snapshots = tuple(_json_snapshot(tool) for tool in tools)
    if layout == "full":
        return ToolVariantMaterial(native_tools=snapshots, documents=())
    if layout not in SELECTED_LAYOUTS:
        raise ValueError(f"Unknown gist layout: {layout!r}")
    native = _validate_native_indices(len(snapshots), native_indices)
    native_tools = tuple(snapshots[index] for index in native)
    if layout == "retrieval":
        return ToolVariantMaterial(native_tools=native_tools, documents=())
    remainder = [index for index in range(len(snapshots)) if index not in native]
    return ToolVariantMaterial(
        native_tools=native_tools, documents=_t0_documents(snapshots, remainder)
    )


def layout_variant(layout: str) -> str:
    return "T1" if layout == "t1" else "T0"


def pack_layout(
    decision: Decision,
    tokenizer: Any,
    *,
    layout: str,
    config: ToolPreparationConfig,
    native_indices: Sequence[int] = (),
):
    material = layout_material(decision.tools, layout, native_indices=native_indices)
    return pack_tool_memory(
        decision.store,
        decision.tools,
        tokenizer,
        variant=layout_variant(layout),
        config=config,
        material=material,
    )


def retrieval_native_indices(
    decision: Decision,
    tokenizer: Any,
    *,
    ranked: Sequence[int],
    allowance_tokens: int,
    config: ToolPreparationConfig,
) -> tuple[int, ...]:
    """Greedily admit ranked schemas while the native prefix fits the allowance.

    A schema that does not fit is skipped and the scan continues, exactly as
    the July retrieval-only control did, so unused allowance can remain.
    """
    chosen: list[int] = []
    for index in ranked:
        candidate = sorted(chosen + [int(index)])
        try:
            memory = pack_layout(
                decision, tokenizer, layout="retrieval", config=config, native_indices=candidate
            )
        except ToolPackingError:
            continue
        if memory.costs(EXP1_RATIOS[0])["resident_kv_tokens"] <= allowance_tokens:
            chosen = candidate
    return tuple(chosen)


# ---------------------------------------------------------------------------
# Per-tool token spans inside the full native prefix
# ---------------------------------------------------------------------------


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _common_suffix(left: Sequence[int], right: Sequence[int], *, limit: int) -> int:
    index = 0
    while index < limit and left[-1 - index] == right[-1 - index]:
        index += 1
    return index


def native_tool_token_spans(
    decision: Decision,
    tokenizer: Any,
    *,
    config: ToolPreparationConfig,
) -> tuple[tuple[int, int], ...]:
    """Token span of every schema inside the ``full`` system prefix.

    The catalog region ends where the protocol tail begins (the longest common
    suffix of the full prefix and the prefix without any schema).  Tool ``i``
    owns the tokens that appear when it is appended to the first ``i`` tools,
    so the spans partition the region in catalog order.  Byte-pair merges at a
    schema boundary can move one token to a neighbour; every token of the
    region belongs to exactly one span regardless.
    """
    snapshots = tuple(_json_snapshot(tool) for tool in decision.tools)
    count = len(snapshots)
    if not count:
        raise ToolPackingError("no_tool_definitions")

    def render(upto: int) -> tuple[int, ...]:
        material = ToolVariantMaterial(native_tools=snapshots[:upto], documents=())
        return pack_tool_memory(
            decision.store,
            decision.tools,
            tokenizer,
            variant="T0",
            config=config,
            material=material,
        ).system_input_ids

    prefixes = [render(upto) for upto in range(count + 1)]
    full, empty = prefixes[-1], prefixes[0]
    head = _common_prefix(full, empty)
    # The shared prefix and the protocol tail may overlap inside ``empty`` (a
    # newline that opens both the first schema and the tail), so the suffix
    # scan is bounded by the string lengths only.
    tail = _common_suffix(full, empty, limit=min(len(full), len(empty)))
    region_end = len(full) - tail
    lengths = [len(prefixes[index + 1]) - len(prefixes[index]) for index in range(count)]
    if any(length <= 0 for length in lengths):
        raise ToolPackingError("tool_span_unresolved")
    region_start = region_end - sum(lengths)
    if region_start < 0 or region_start > head:
        raise ToolPackingError("tool_span_unresolved")
    spans: list[tuple[int, int]] = []
    cursor = region_start
    for length in lengths:
        spans.append((cursor, cursor + length))
        cursor += length
    return tuple(spans)


# ---------------------------------------------------------------------------
# Records for one decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutPlan:
    k_values: tuple[int, ...] = (1, 3, 5)
    seed: int = 42
    layouts: tuple[str, ...] = GIST_LAYOUTS
    ratios: tuple[int, ...] = EXP1_RATIOS
    ranker_scope: str = "last_user"

    def __post_init__(self) -> None:
        if not self.k_values or any(type(k) is not int or k <= 0 for k in self.k_values):
            raise ValueError("k_values must be positive integers")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must be distinct")
        unknown = [layout for layout in self.layouts if layout not in GIST_LAYOUTS]
        if unknown:
            raise ValueError(f"Unknown gist layouts: {unknown}")
        if tuple(self.ratios) != EXP1_RATIOS:
            raise ValueError("Exp 1 is frozen to ratios 8 and 12")


def _record(
    decision: Decision,
    memory,
    *,
    layout: str,
    k: int | None,
    ratio: int,
    target_ids: tuple[int, ...],
    session_key: str,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prepared = PreparedDecision(
        memory=memory,
        target_ids=target_ids,
        ratio=ratio,
        weight=1.0,
        decision_id=decision.decision_id,
    )
    costs = memory.costs(ratio)
    metadata = {
        **metadata,
        "layout": layout,
        "k": k,
        "variant": layout_variant(layout),
        "token_counts": {**costs, "target_tokens": len(target_ids)},
    }
    record = make_record(
        prepared, session_key=session_key, source=source, split="train", metadata=metadata
    )
    record["split"] = EXP1_PURPOSE
    record["layout"] = layout
    record["k"] = k
    return record


def build_layout_records(
    decision: Decision,
    tokenizer: Any,
    *,
    config: ToolPreparationConfig,
    plan: LayoutPlan,
    session_key: str,
    source: str,
    target_ids: tuple[int, ...],
    base_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Pack every planned gist layout of one decision at both ratios.

    ``full`` carries the per-tool token spans and ``uniform``/``hybrid`` carry
    the gist token counts that the eviction layouts match at evaluation time.
    """
    tools = decision.tools
    if not tools:
        raise ToolPackingError("no_tool_definitions")
    ranked = lexical_rank(tools, query_text(decision_messages(decision), plan.ranker_scope))
    gold = gold_tool_calls(decision)
    gold_names = [call["name"] for call in gold]
    names = [tool_name(tool) for tool in tools]

    def hit(native: Sequence[int]) -> bool | None:
        if not gold_names:
            return None
        native_names = {names[index] for index in native}
        return all(name in native_names for name in gold_names)

    def rank_of_gold() -> int | None:
        if not gold_names:
            return None
        positions = [
            position for position, index in enumerate(ranked, start=1)
            if names[index] == gold_names[0]
        ]
        return positions[0] if positions else None

    common = {
        **dict(base_metadata),
        "packing_profile": "next-compression-exp1-tools-v1",
        "ranker": RANKER,
        "ranker_scope": plan.ranker_scope,
        "lexical_rank": list(ranked),
        "gold_first_tool_rank": rank_of_gold(),
        "tool_count": len(tools),
    }
    records: list[dict[str, Any]] = []
    hybrids: dict[tuple[int, int], int] = {}

    def emit(layout: str, memory, *, k: int | None, extra: Mapping[str, Any]) -> None:
        for ratio in plan.ratios:
            record = _record(
                decision,
                memory,
                layout=layout,
                k=k,
                ratio=ratio,
                target_ids=target_ids,
                session_key=session_key,
                source=source,
                metadata={**common, **extra},
            )
            records.append(record)
            if layout == "hybrid":
                hybrids[(int(k), ratio)] = memory.costs(ratio)["resident_kv_tokens"]

    if "full" in plan.layouts:
        memory = pack_layout(decision, tokenizer, layout="full", config=config)
        spans = native_tool_token_spans(decision, tokenizer, config=config)
        emit("full", memory, k=None, extra={
            "native_tool_indices": list(range(len(tools))),
            "tool_token_spans": [list(span) for span in spans],
        })
    if "uniform" in plan.layouts:
        memory = pack_layout(decision, tokenizer, layout="uniform", config=config)
        emit("uniform", memory, k=None, extra={"native_tool_indices": []})
    if "t1" in plan.layouts:
        memory = pack_layout(decision, tokenizer, layout="t1", config=config)
        emit("t1", memory, k=None, extra={"native_tool_indices": list(range(len(tools)))})
    for k in plan.k_values:
        if k > len(tools):
            continue
        if "hybrid" in plan.layouts:
            native = tuple(sorted(ranked[:k]))
            memory = pack_layout(
                decision, tokenizer, layout="hybrid", config=config, native_indices=native
            )
            emit("hybrid", memory, k=k, extra={
                "native_tool_indices": list(native), "selection_hit": hit(native),
            })
        if "random" in plan.layouts:
            native = tuple(sorted(random_rank(len(tools), seed=plan.seed, decision_id=decision.decision_id)[:k]))
            memory = pack_layout(
                decision, tokenizer, layout="random", config=config, native_indices=native
            )
            emit("random", memory, k=k, extra={
                "native_tool_indices": list(native), "selection_hit": hit(native),
                "seed": plan.seed,
            })
    if "retrieval" in plan.layouts:
        if "hybrid" not in plan.layouts:
            raise ValueError("retrieval requires the hybrid layout for its allowance")
        for k in plan.k_values:
            for ratio in plan.ratios:
                allowance = hybrids.get((k, ratio))
                if allowance is None:
                    continue
                native = retrieval_native_indices(
                    decision, tokenizer, ranked=ranked, allowance_tokens=allowance, config=config
                )
                memory = pack_layout(
                    decision, tokenizer, layout="retrieval", config=config, native_indices=native
                )
                record = _record(
                    decision,
                    memory,
                    layout="retrieval",
                    k=k,
                    ratio=ratio,
                    target_ids=target_ids,
                    session_key=session_key,
                    source=source,
                    metadata={
                        **common,
                        "native_tool_indices": list(native),
                        "selection_hit": hit(native),
                        "allowance_tokens": allowance,
                        "allowance_source": f"hybrid.k{k}.ratio{ratio}",
                    },
                )
                records.append(record)
    return records


def validate_exp1_record(record: Mapping[str, Any]) -> None:
    from .common import validate_record

    if record.get("split") != EXP1_PURPOSE:
        raise ValueError("Exp 1 records carry the exp1_tools split")
    if record.get("layout") not in GIST_LAYOUTS:
        raise ValueError(f"Unknown Exp 1 layout: {record.get('layout')!r}")
    copy_ = dict(record)
    copy_["split"] = "train"
    copy_.pop("layout")
    copy_.pop("k")
    validate_record(copy_, ratios=EXP1_RATIOS)


# ---------------------------------------------------------------------------
# Scoring: per-decision outcomes and paired exact tests
# ---------------------------------------------------------------------------


def outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strict ordered call match, first-name match, and false-call flag."""
    metadata = row.get("metadata") or {}
    supplied = row.get("gold_tool_calls", metadata.get("gold_tool_calls"))
    gold = canonical_calls(supplied) if supplied is not None else parse_calls(row["target_text"])
    try:
        predicted = parse_calls(row["generated_text"])
        malformed = False
    except (ValueError, TypeError, KeyError):
        predicted, malformed = [], True
    if gold:
        return {
            "tool_decision": True,
            "call_correct": (not malformed) and predicted == gold,
            "name_correct": bool(predicted) and predicted[0]["name"] == gold[0]["name"],
            "false_call": None,
            "malformed": malformed,
        }
    return {
        "tool_decision": False,
        "call_correct": None,
        "name_correct": None,
        "false_call": malformed or bool(predicted),
        "malformed": malformed,
    }


def score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one layout at one ratio; denominators are explicit."""
    if not rows:
        raise ValueError("Cannot score an empty layout")
    outcomes: dict[str, dict[str, Any]] = {}
    kv = []
    for row in rows:
        if row["decision_id"] in outcomes:
            raise ValueError(f"Duplicate decision in layout scoring: {row['decision_id']}")
        outcomes[row["decision_id"]] = outcome(row)
        kv.append(int(row["resident_kv_tokens"]))
    tool = [item for item in outcomes.values() if item["tool_decision"]]
    non_tool = [item for item in outcomes.values() if not item["tool_decision"]]
    return {
        "records": len(rows),
        "tool_decisions": len(tool),
        "strict_ordered_call_correct": sum(item["call_correct"] for item in tool),
        "strict_ordered_call_accuracy": (
            sum(item["call_correct"] for item in tool) / len(tool) if tool else None
        ),
        "tool_name_correct": sum(item["name_correct"] for item in tool),
        "tool_name_accuracy": (
            sum(item["name_correct"] for item in tool) / len(tool) if tool else None
        ),
        "non_tool_decisions": len(non_tool),
        "false_tool_calls": sum(item["false_call"] for item in non_tool),
        "false_tool_call_rate": (
            sum(item["false_call"] for item in non_tool) / len(non_tool) if non_tool else None
        ),
        "malformed_outputs": sum(item["malformed"] for item in outcomes.values()),
        "resident_kv_tokens_mean": sum(kv) / len(kv),
        "resident_kv_tokens_total": sum(kv),
        "outcomes": outcomes,
        "scope": "Recorded-decision action proxy; no tool execution or official task success.",
    }


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    low = min(b, c)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_test(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, Any]:
    """Compare two layouts on the shared decisions where ``key`` is defined."""
    shared = sorted(
        decision_id
        for decision_id in left
        if decision_id in right
        and left[decision_id].get(key) is not None
        and right[decision_id].get(key) is not None
    )
    both = left_only = right_only = neither = 0
    for decision_id in shared:
        a, b = bool(left[decision_id][key]), bool(right[decision_id][key])
        if a and b:
            both += 1
        elif a:
            left_only += 1
        elif b:
            right_only += 1
        else:
            neither += 1
    return {
        "key": key,
        "n": len(shared),
        "left_correct": both + left_only,
        "right_correct": both + right_only,
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "difference": left_only - right_only,
        "p_exact_mcnemar": exact_mcnemar(left_only, right_only),
    }
