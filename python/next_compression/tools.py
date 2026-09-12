"""CPU preparation for matched tool-definition compression variants.

T0 encodes every complete tool definition as an independent document while
keeping the observable conversation prefix native. T1 keeps tool identity and
schema constraints in the native tools prefix and encodes only descriptive
fields as independent supplements. Both variants use the same immutable
decision target, ratios, and loss surface.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from history_memory.dataset import (
    Decision,
    event_store_session_id,
    iter_decisions,
    snapshot_and_validate_rows,
)
from history_memory.events import EventStore
from history_memory.packing import (
    EncoderChunk,
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    native_ids,
    pack_target,
    raw_workspace_messages,
)

from .common import PreparedDecision, make_record


TOOL_VARIANTS = ("T0", "T1")
TOOL_RATIOS = (8, 12)
TOOL_PACKING_PROFILE = "next-compression-tool-explicit-protocol-v2"
TOOL_PROTOCOL_HEAD = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "Tool definitions are available in compressed memory. Native tool schemas, "
    "when present, are listed within <tools></tools> XML tags:\n<tools>"
)
TOOL_PROTOCOL_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function name "
    "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


@dataclass(frozen=True)
class ToolPreparationConfig:
    """Auditable all-or-reject limits for CPU tokenization.

    A ``None`` limit is intentionally unbounded. No limit truncates a tool,
    native prefix, target, or chunk; the complete matched decision is skipped
    when either T0 or T1 exceeds a configured bound.
    """

    ratios: tuple[int, ...] = TOOL_RATIOS
    max_chunk_tokens: int = 768
    chunk_overlap: int = 64
    max_chunks: int | None = 48
    max_tool_tokens: int | None = 36_864
    max_raw_tokens: int | None = 4_096
    max_target_tokens: int | None = 4_096
    max_tools: int | None = None
    max_sequence_tokens: int = 16_384
    max_decisions_per_session: int = 64

    def __post_init__(self) -> None:
        if self.ratios != TOOL_RATIOS:
            raise ValueError("This tool checkpoint is fixed to mixed ratios 8 and 12")
        if (
            isinstance(self.max_chunk_tokens, bool)
            or not isinstance(self.max_chunk_tokens, int)
            or self.max_chunk_tokens <= 0
        ):
            raise ValueError("max_chunk_tokens must be a positive integer")
        if (
            isinstance(self.chunk_overlap, bool)
            or not isinstance(self.chunk_overlap, int)
            or self.chunk_overlap < 0
            or self.chunk_overlap >= self.max_chunk_tokens
        ):
            raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
        for name in (
            "max_chunks",
            "max_tool_tokens",
            "max_raw_tokens",
            "max_target_tokens",
            "max_tools",
            "max_sequence_tokens",
            "max_decisions_per_session",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_sequence_tokens is None or self.max_decisions_per_session is None:
            raise ValueError(
                "max_sequence_tokens and max_decisions_per_session must be bounded"
            )


@dataclass(frozen=True)
class ToolSupplement:
    """Fields removed from one T1 native tool, with typed paths for recovery."""

    tool_index: int
    tool_identity: Mapping[str, Any]
    fields: tuple[tuple[tuple[str | int, ...], Any], ...]

    def as_document(self) -> dict[str, Any]:
        return {
            "type": "tool_definition_supplement",
            "tool_index": self.tool_index,
            "tool_identity": copy.deepcopy(dict(self.tool_identity)),
            "fields": [
                {"path": list(path), "value": copy.deepcopy(value)}
                for path, value in self.fields
            ],
        }


@dataclass(frozen=True)
class ToolVariantMaterial:
    """Target-independent material used by preparation and online inference."""

    native_tools: tuple[dict[str, Any], ...] | None
    documents: tuple[dict[str, Any], ...]


class ToolPackingError(ValueError):
    """A named, auditable all-or-reject condition shared with serving."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


_DESCRIPTIVE_STRING_KEYS = frozenset({"description", "title", "$comment"})
_DESCRIPTIVE_SEQUENCE_KEYS = frozenset({"examples"})


def _json_snapshot(value: Any) -> Any:
    """Return a JSON-owned value and reject non-finite/non-JSON tool metadata."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _strip_descriptive_fields(
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> tuple[Any, tuple[tuple[tuple[str | int, ...], Any], ...]]:
    if isinstance(value, Mapping):
        native: dict[str, Any] = {}
        removed: list[tuple[tuple[str | int, ...], Any]] = []
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("Tool definition keys must be strings")
            child_path = path + (raw_key,)
            is_descriptive = (
                raw_key in _DESCRIPTIVE_STRING_KEYS and isinstance(child, str)
            ) or (
                raw_key in _DESCRIPTIVE_SEQUENCE_KEYS
                and isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
            )
            if is_descriptive:
                removed.append((child_path, _json_snapshot(child)))
                continue
            compact, nested = _strip_descriptive_fields(child, path=child_path)
            native[raw_key] = compact
            removed.extend(nested)
        return native, tuple(removed)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        native_items = []
        removed: list[tuple[tuple[str | int, ...], Any]] = []
        for index, child in enumerate(value):
            compact, nested = _strip_descriptive_fields(child, path=path + (index,))
            native_items.append(compact)
            removed.extend(nested)
        return native_items, tuple(removed)
    return _json_snapshot(value), ()


def split_tool_definitions(
    tools: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[ToolSupplement, ...]]:
    """Split tools without target-aware selection or schema-key assumptions.

    Only explicitly descriptive JSON Schema fields move to compressed
    supplements. Tool wrappers, tool types, names, parameter structure,
    required fields, enums, bounds, patterns, unions, and unknown extension
    keys remain native.
    """
    if isinstance(tools, (str, bytes, bytearray)) or not isinstance(tools, Sequence):
        raise TypeError("tools must be a sequence of mappings")
    native_tools: list[dict[str, Any]] = []
    supplements: list[ToolSupplement] = []
    for tool_index, raw_tool in enumerate(tools):
        if not isinstance(raw_tool, Mapping):
            raise ValueError(f"tools[{tool_index}] must be a mapping")
        compact, removed = _strip_descriptive_fields(raw_tool)
        if not isinstance(compact, dict):
            raise AssertionError("A tool mapping must remain a mapping")
        native_tools.append(compact)
        if removed:
            supplements.append(
                ToolSupplement(tool_index, _tool_identity(compact), removed)
            )
    return tuple(native_tools), tuple(supplements)


def _tool_identity(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a supplement to its native tool without target-side information."""
    identity: dict[str, Any] = {}
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        identity["type"] = tool_type
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        identity["name"] = function["name"]
    elif isinstance(tool.get("name"), str):
        identity["name"] = tool["name"]
    return identity


def _set_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    if not path:
        raise ValueError("A supplement field needs a nonempty path")
    parent = root
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(parent, list) or not 0 <= part < len(parent):
                raise ValueError("Supplement list path does not match native tool")
            parent = parent[part]
        else:
            if not isinstance(parent, dict) or part not in parent:
                raise ValueError("Supplement object path does not match native tool")
            parent = parent[part]
    leaf = path[-1]
    if isinstance(leaf, int):
        if not isinstance(parent, list) or not 0 <= leaf < len(parent):
            raise ValueError("Supplement list leaf does not match native tool")
        parent[leaf] = _json_snapshot(value)
    else:
        if not isinstance(parent, dict) or leaf in parent:
            raise ValueError("Supplement object leaf collides with native tool")
        parent[leaf] = _json_snapshot(value)


def restore_tool_definitions(
    native_tools: Sequence[Mapping[str, Any]],
    supplements: Sequence[ToolSupplement],
) -> tuple[dict[str, Any], ...]:
    """Reconstruct the lossless tool JSON represented by a T1 split."""
    restored = [_json_snapshot(tool) for tool in native_tools]
    seen: set[tuple[int, tuple[str | int, ...]]] = set()
    for supplement in supplements:
        if not 0 <= supplement.tool_index < len(restored):
            raise ValueError("Supplement tool_index is outside native tools")
        if dict(supplement.tool_identity) != _tool_identity(
            restored[supplement.tool_index]
        ):
            raise ValueError("Supplement identity differs from its native tool")
        for path, value in supplement.fields:
            key = (supplement.tool_index, path)
            if key in seen:
                raise ValueError("Duplicate supplement field path")
            seen.add(key)
            _set_path(restored[supplement.tool_index], path, value)
    return tuple(restored)


def tool_variant_material(
    tools: Sequence[Mapping[str, Any]], variant: str
) -> ToolVariantMaterial:
    """Build deterministic documents and the variant's native tools prefix."""
    if isinstance(tools, (str, bytes, bytearray)) or not isinstance(tools, Sequence):
        raise TypeError("tools must be a sequence of mappings")
    snapshots = []
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"tools[{tool_index}] must be a mapping")
        snapshots.append(_json_snapshot(tool))
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ToolPackingError("no_tool_definitions")
    if variant == "T0":
        documents = tuple(
            {"type": "tool_definition", "tool_index": index, "tool": tool}
            for index, tool in enumerate(snapshots)
        )
        return ToolVariantMaterial(native_tools=None, documents=documents)
    if variant != "T1":
        raise ValueError(f"Unknown tool variant: {variant!r}")
    native_tools, supplements = split_tool_definitions(snapshots)
    if restore_tool_definitions(native_tools, supplements) != snapshots:
        raise AssertionError("T1 native schemas and supplements must losslessly cover tools")
    documents = tuple(supplement.as_document() for supplement in supplements)
    if not documents:
        raise ToolPackingError("no_compressible_tool_information")
    return ToolVariantMaterial(native_tools=native_tools, documents=documents)


def _document_ids(tokenizer: Any, document: Mapping[str, Any]) -> tuple[int, ...]:
    envelope = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return native_ids(tokenizer, ({"role": "user", "content": envelope},))


def _document_chunks(
    tokenizer: Any,
    documents: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    config: ToolPreparationConfig,
) -> tuple[EncoderChunk, ...]:
    chunks: list[EncoderChunk] = []
    for document_index, document in enumerate(documents):
        ids = _document_ids(tokenizer, document)
        identity = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        event_id = f"{variant.lower()}-tool-{document_index}-{digest}"
        start = 0
        part_index = 0
        while start < len(ids):
            end = min(start + config.max_chunk_tokens, len(ids))
            chunks.append(
                EncoderChunk(
                    event_id=event_id,
                    part_index=part_index,
                    source_indices=(),
                    source_token_start=start,
                    source_token_end=end,
                    token_ids=ids[start:end],
                )
            )
            if end == len(ids):
                break
            start = end - config.chunk_overlap
            part_index += 1
    presented_tokens = sum(len(chunk.token_ids) for chunk in chunks)
    if config.max_tool_tokens is not None and presented_tokens > config.max_tool_tokens:
        raise ToolPackingError("tool_tokens_over_limit")
    if config.max_chunks is not None and len(chunks) > config.max_chunks:
        raise ToolPackingError("tool_chunks_over_limit")
    if not chunks:
        raise ToolPackingError("no_tool_document_tokens")
    return tuple(chunks)


def _explicit_tool_protocol(
    native_tools: Sequence[Mapping[str, Any]] | None,
) -> str:
    schemas = ""
    if native_tools:
        schemas = "".join(
            "\n"
            + json.dumps(
                tool,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            for tool in native_tools
        )
    return TOOL_PROTOCOL_HEAD + schemas + TOOL_PROTOCOL_TAIL


def _native_tool_memory(
    store: EventStore,
    view: MemoryView,
    tokenizer: Any,
    *,
    native_tools: Sequence[Mapping[str, Any]] | None,
    max_raw_tokens: int | None,
) -> PackedMemory:
    """Render one explicit raw protocol for both tool variants.

    Qwen's ``tools=`` branch couples the schema payload to the generic output
    protocol. Calling it only for T1 would remove the output protocol from T0.
    This explicit system block keeps that protocol byte-identical across the
    variants while allowing T0 to have no native schemas.
    """
    view.validate(store)
    raw_messages = list(raw_workspace_messages(store, view))
    if not raw_messages:
        raise ValueError("A tool decision requires an observable raw message")
    protocol = _explicit_tool_protocol(native_tools)
    if raw_messages[0]["role"] == "system":
        content = raw_messages[0].get("content")
        if not isinstance(content, str):
            raise ValueError("System instruction content must be text")
        raw_messages[0] = {**raw_messages[0], "content": content + "\n\n" + protocol}
    else:
        raw_messages.insert(0, {"role": "system", "content": protocol})

    full_ids = native_ids(tokenizer, raw_messages, generation=True)
    prefix_messages = []
    for message in raw_messages:
        if message["role"] != "system":
            break
        prefix_messages.append(message)
    dummy = {"role": "user", "content": ""}
    dummy_ids = native_ids(tokenizer, (dummy,))
    prefix_and_dummy = native_ids(tokenizer, (*prefix_messages, dummy))
    if not dummy_ids or prefix_and_dummy[-len(dummy_ids) :] != dummy_ids:
        raise ValueError("Native template does not support a separable tool prefix")
    prefix_ids = prefix_and_dummy[: -len(dummy_ids)]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Explicit tool prefix changed with workspace content")
    if max_raw_tokens is not None and len(full_ids) > max_raw_tokens:
        raise PackingBudgetError(
            f"Complete tool protocol and raw workspace need {len(full_ids)} tokens; "
            f"budget is {max_raw_tokens}"
        )
    raw_indices = tuple(
        sorted(
            {
                source_index
                for event_id in view.raw_event_ids
                for source_index in store.event(event_id).source_indices
            }
        )
    )
    return PackedMemory(
        view=view,
        system_input_ids=prefix_ids,
        workspace_input_ids=full_ids[len(prefix_ids) :],
        raw_source_indices=raw_indices,
        chunks=(),
    )


def pack_tool_memory(
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    variant: str,
    config: ToolPreparationConfig | None = None,
) -> PackedMemory:
    """Pack the target-independent T0/T1 prefix for training or inference."""
    config = config or ToolPreparationConfig()
    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if config.max_tools is not None and len(tools) > config.max_tools:
        raise ToolPackingError("tool_count_over_limit")
    material = tool_variant_material(tools, variant)
    all_raw = MemoryView(
        gist_event_ids=(),
        raw_event_ids=tuple(event.event_id for event in store.events),
        evidence_event_ids=(),
    )
    try:
        native = _native_tool_memory(
            store,
            all_raw,
            tokenizer,
            native_tools=material.native_tools,
            max_raw_tokens=config.max_raw_tokens,
        )
    except PackingBudgetError as exc:
        raise ToolPackingError("raw_tokens_over_limit") from exc
    chunks = _document_chunks(
        tokenizer,
        material.documents,
        variant=variant,
        config=config,
    )
    return PackedMemory(
        view=native.view,
        system_input_ids=native.system_input_ids,
        workspace_input_ids=native.workspace_input_ids,
        raw_source_indices=native.raw_source_indices,
        chunks=chunks,
        raw_layout_profile=native.raw_layout_profile,
    )


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(function["name"])
        if isinstance(tool.get("name"), str):
            names.add(tool["name"])
        tool_type = tool.get("type")
        if isinstance(tool_type, str) and tool_type != "function":
            names.add(tool_type)
    return frozenset(names)


def _called_tool_names(decision: Decision) -> tuple[str, ...]:
    target = decision.target_dict()
    return tuple(call["function"]["name"] for call in target.get("tool_calls") or ())


def _has_prior_history(decision: Decision) -> bool:
    user_events = [event for event in decision.store.events if event.kind == "user"]
    if not user_events:
        return False
    current_start = min(user_events[-1].source_indices)
    return any(
        event.kind != "instruction" and max(event.source_indices) < current_start
        for event in decision.store.events
    )


def _target_kind(decision: Decision) -> str:
    return "covered_tool_call" if _called_tool_names(decision) else "non_tool_response"


def _prepare_pair(
    decision: Decision,
    tokenizer: Any,
    config: ToolPreparationConfig,
) -> tuple[PackedMemory, PackedMemory, tuple[int, ...]]:
    tools = decision.tools
    if not tools:
        raise ToolPackingError("no_tool_definitions")
    if not any(event.kind == "user" for event in decision.store.events):
        raise ToolPackingError("no_current_user")
    called = _called_tool_names(decision)
    if called and not set(called) <= _tool_names(tools):
        raise ToolPackingError("target_tool_not_covered")
    try:
        target_ids = pack_target(
            tokenizer,
            decision.target,
            max_target_tokens=config.max_target_tokens,
        )
    except PackingBudgetError as exc:
        raise ToolPackingError("target_tokens_over_limit") from exc
    t0 = pack_tool_memory(
        decision.store, tools, tokenizer, variant="T0", config=config
    )
    t1 = pack_tool_memory(
        decision.store, tools, tokenizer, variant="T1", config=config
    )
    for variant, memory in (("T0", t0), ("T1", t1)):
        for ratio in config.ratios:
            sequence_tokens = memory.costs(ratio)["resident_kv_tokens"] + len(
                target_ids
            )
            if sequence_tokens > config.max_sequence_tokens:
                raise ToolPackingError(
                    f"sequence_tokens_over_limit.{variant}.ratio{ratio}"
                )
    return t0, t1, target_ids


def prepare_tool_decision(
    decision: Decision,
    tokenizer: Any,
    *,
    config: ToolPreparationConfig | None = None,
) -> dict[str, tuple[PreparedDecision, ...]]:
    """Prepare one complete matched decision at both fixed ratios."""
    config = config or ToolPreparationConfig()
    t0, t1, target_ids = _prepare_pair(decision, tokenizer, config)
    return {
        variant: tuple(
            PreparedDecision(
                memory=memory,
                target_ids=target_ids,
                ratio=ratio,
                weight=1.0,
                decision_id=decision.decision_id,
            )
            for ratio in config.ratios
        )
        for variant, memory in (("T0", t0), ("T1", t1))
    }


def iter_tool_records(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    config: ToolPreparationConfig | None = None,
    *,
    make_record_fn: Callable[..., dict[str, Any]] | None = None,
) -> tuple[Iterator[tuple[str, dict[str, Any]]], Counter[str]]:
    """Return matched T0/T1 records and a live audit filled during iteration.

    All rows are snapshotted and group-split validated before the iterator is
    returned. Consumers must exhaust the iterator before serializing ``audit``.
    Every accepted base decision yields T0 and T1 at ratios 8 and 12 with the
    same decision ID and exact target token sequence.
    """
    config = config or ToolPreparationConfig()
    if not isinstance(config, ToolPreparationConfig):
        raise TypeError("config must be ToolPreparationConfig")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must expose apply_chat_template")
    writer = make_record if make_record_fn is None else make_record_fn
    if not callable(writer):
        raise TypeError("make_record_fn must be callable")
    snapshots = snapshot_and_validate_rows(rows)
    audit: Counter[str] = Counter()
    audit["rows.total"] = len(snapshots)

    def generate() -> Iterator[tuple[str, dict[str, Any]]]:
        for row in snapshots:
            source = str(row["source"])
            if row["split"] != "train":
                audit["rows.skipped.non_train_split"] += 1
                audit[f"rows.skipped.non_train_split.source.{source}"] += 1
                continue
            audit["rows.train"] += 1
            last_source_index = len(row["messages"]) - 1
            session_key = event_store_session_id(source, str(row["session_id"]))
            candidate_count = sum(
                message.get("role") == "assistant"
                and (
                    message.get("content") not in (None, "")
                    or bool(message.get("tool_calls"))
                )
                for message in row["messages"]
            )
            audit["decisions.candidates"] += candidate_count
            audit[f"decisions.candidates.source.{source}"] += candidate_count
            capped = max(0, candidate_count - config.max_decisions_per_session)
            if capped:
                audit["sessions.decision_cap_reached"] += 1
                audit["decisions.skipped"] += capped
                audit["decisions.skipped.reason.per_session_limit"] += capped
                audit[f"decisions.skipped.source.{source}"] += capped
                audit[
                    f"decisions.skipped.source.{source}.reason.per_session_limit"
                ] += capped
            for decision_ordinal, decision in enumerate(iter_decisions(row)):
                if decision_ordinal >= config.max_decisions_per_session:
                    break
                audit["decisions.observed"] += 1
                target_kind = _target_kind(decision)
                audit[f"decisions.observed.kind.{target_kind}"] += 1
                audit[f"decisions.observed.source.{source}"] += 1
                audit[f"decisions.observed.source.{source}.kind.{target_kind}"] += 1
                if not _has_prior_history(decision):
                    audit["decisions.no_prior_history"] += 1
                    audit[f"decisions.no_prior_history.source.{source}"] += 1
                terminal_stop = (
                    target_kind == "non_tool_response"
                    and decision.source_message_index == last_source_index
                )
                if terminal_stop:
                    audit["decisions.source_grounded_terminal_stop"] += 1
                    audit[
                        f"decisions.source_grounded_terminal_stop.source.{source}"
                    ] += 1
                try:
                    prepared = prepare_tool_decision(
                        decision, tokenizer, config=config
                    )
                except ToolPackingError as exc:
                    audit["decisions.skipped"] += 1
                    audit[f"decisions.skipped.reason.{exc.reason}"] += 1
                    audit[f"decisions.skipped.source.{source}"] += 1
                    audit[
                        f"decisions.skipped.source.{source}.reason.{exc.reason}"
                    ] += 1
                    continue
                audit["decisions.accepted"] += 1
                audit[f"decisions.accepted.kind.{target_kind}"] += 1
                audit[f"decisions.accepted.source.{source}"] += 1
                audit[f"decisions.accepted.source.{source}.kind.{target_kind}"] += 1
                for variant in TOOL_VARIANTS:
                    for prepared_decision in prepared[variant]:
                        metadata = {
                            "packing_profile": TOOL_PACKING_PROFILE,
                            "variant": variant,
                            "target_kind": target_kind,
                            "source_grounded_terminal_stop": terminal_stop,
                            "task_id": decision.task_id,
                            "template_id": decision.template_id,
                            "decision_index": decision.decision_index,
                            "source_message_index": decision.source_message_index,
                            "limits": {
                                "max_chunk_tokens": config.max_chunk_tokens,
                                "chunk_overlap": config.chunk_overlap,
                                "max_chunks": config.max_chunks,
                                "max_tool_tokens": config.max_tool_tokens,
                                "max_raw_tokens": config.max_raw_tokens,
                            "max_target_tokens": config.max_target_tokens,
                            "max_tools": config.max_tools,
                            "max_sequence_tokens": config.max_sequence_tokens,
                            "max_decisions_per_session": config.max_decisions_per_session,
                            },
                            "token_counts": {
                                **prepared_decision.memory.costs(
                                    prepared_decision.ratio
                                ),
                                "target_tokens": len(prepared_decision.target_ids),
                                "sequence_tokens": prepared_decision.memory.costs(
                                    prepared_decision.ratio
                                )["resident_kv_tokens"]
                                + len(prepared_decision.target_ids),
                            },
                        }
                        record = writer(
                            prepared_decision,
                            session_key=session_key,
                            source=source,
                            split="train",
                            metadata=metadata,
                        )
                        audit["records.written"] += 1
                        audit[f"records.written.variant.{variant}"] += 1
                        audit[
                            f"records.written.variant.{variant}.ratio.{prepared_decision.ratio}"
                        ] += 1
                        audit[f"records.written.source.{source}"] += 1
                        yield variant, record

    return generate(), audit


def prepare_tools(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    config: ToolPreparationConfig | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Materialize the streaming API for tests and bounded CPU callers."""
    stream, audit = iter_tool_records(rows, tokenizer, config)
    records = {variant: [] for variant in TOOL_VARIANTS}
    for variant, record in stream:
        records[variant].append(record)
    return records, audit
