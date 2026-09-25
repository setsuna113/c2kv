"""Tool-definition KV memory (T0 encoder): an axis orthogonal to the history arm.

Every arm in arms.py describes how the *interaction history* reaches the
model; the tool catalog has always been rendered raw by the chat template.
This module adds the second context type of the paper (Section 3.2, "tool
allocation before generation") without touching any arm:

* the proxy is started with ``--tool-memory t0:r8[:hybrid3] --tool-checkpoint
  <T0 dir>``; without the flag nothing here runs and every request is
  byte-identical to before;
* per request, the catalog is split into ``native`` schemas (kept as text,
  ``hybrid`` top-k by the fixed lexical ranker; none for ``uniform``) and a
  compressed remainder.  The remainder is packed EXACTLY like T0 training
  (next_compression/tools.py): one document per tool definition,
  ``{"type":"tool_definition","tool_index":i,"tool":<schema>}`` rendered as a
  single user message, cut into <= ``chunk_tokens`` chunks with
  ``chunk_overlap`` carry-back, encoded by the server's *tool* projection set
  (``/v1/c2kv/extract`` with ``token_ids`` + ``projection_set="tool"``);
* the system message gets the explicit protocol block (native schemas
  inside ``<tools>``) and the request tells the server not to render
  ``tools`` (``c2kv_tools_in_prompt=false``); the gist entries ride as
  carrier messages right after the system prefix, so the server injects them
  at the system-prefix boundary with source-span positions, i.e. the layout
  ``system_input_ids -> tool chunks -> workspace`` of training.

Constants (protocol text, ranker, chunking) are ported verbatim from
``c2kv-next-training/python/next_compression/{tools,exp1_tools}.py`` and
pinned by tests; they are the training contract, not tunables.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from toolinterface import (InterfaceCopy, compact_tool, description_document,
                               interface_copy, json_string_value_spans, tool_prose)
except ModuleNotFoundError:
    # The history runtime imports this file by path under a synthetic module
    # name, without adding benchmarks/ to sys.path.
    _interface_path = Path(__file__).with_name("toolinterface.py")
    _interface_name = "c2kv_toolinterface_" + hashlib.sha256(
        str(_interface_path.resolve()).encode("utf-8")).hexdigest()[:12]
    _interface_spec = importlib.util.spec_from_file_location(_interface_name, _interface_path)
    if _interface_spec is None or _interface_spec.loader is None:
        raise RuntimeError(f"cannot load tool interface helper {_interface_path}")
    _interface_module = importlib.util.module_from_spec(_interface_spec)
    sys.modules[_interface_name] = _interface_module
    _interface_spec.loader.exec_module(_interface_module)
    InterfaceCopy = _interface_module.InterfaceCopy
    compact_tool = _interface_module.compact_tool
    description_document = _interface_module.description_document
    interface_copy = _interface_module.interface_copy
    json_string_value_spans = _interface_module.json_string_value_spans
    tool_prose = _interface_module.tool_prose

try:
    from toolselection import (ADAPTIVE_RELATIVE_THRESHOLD, DEFAULT_SELECTOR_POLICY,
                               SELECTOR_POLICIES,
                               last_user_query, lexical_rank as selection_lexical_rank,
                               selector_version, tool_selection)
except ModuleNotFoundError:
    # Keep path-based imports working for the copied history runtime.
    _selection_path = Path(__file__).with_name("toolselection.py")
    _selection_name = "c2kv_toolselection_" + hashlib.sha256(
        str(_selection_path.resolve()).encode("utf-8")).hexdigest()[:12]
    _selection_spec = importlib.util.spec_from_file_location(_selection_name, _selection_path)
    if _selection_spec is None or _selection_spec.loader is None:
        raise RuntimeError(f"cannot load tool selection helper {_selection_path}")
    _selection_module = importlib.util.module_from_spec(_selection_spec)
    sys.modules[_selection_name] = _selection_module
    _selection_spec.loader.exec_module(_selection_module)
    ADAPTIVE_RELATIVE_THRESHOLD = _selection_module.ADAPTIVE_RELATIVE_THRESHOLD
    DEFAULT_SELECTOR_POLICY = _selection_module.DEFAULT_SELECTOR_POLICY
    SELECTOR_POLICIES = _selection_module.SELECTOR_POLICIES
    last_user_query = _selection_module.last_user_query
    selection_lexical_rank = _selection_module.lexical_rank
    selector_version = _selection_module.selector_version
    tool_selection = _selection_module.tool_selection

TOOL_MEMORY_SCHEMA = "c2kv.tool_memory.v1"
RENDER_PROFILE = "next-compression-tool-explicit-protocol-v2"
RANKER = "lexical-name4-text1-last-user-v1"
ENCODERS = ("t0", "streamingllm", "h2o", "snapkv", "pyramidkv")
LAYOUTS = ("uniform", "hybrid")
SUPPORTED_RATIOS = (8, 12)
# Marker on the carrier messages the proxy inserts; every proxy-side message
# walker that classifies history must skip messages carrying it.
CARRIER_MARK = "c2kv_tool_memory"
INTERFACE_BLOCK_HEAD = "\n# Executable tool interfaces\n"
INTERFACE_RENDER_PROFILE = "tool-schema-split-v3"
DESCRIPTION_DOCUMENT_PROFILE = "tool-description-only-v1"
PERSISTENT_RENDER_PROFILE = "persistent-tool-schema-document-slots-v1"

# next_compression/tools.py TOOL_PROTOCOL_HEAD / TOOL_PROTOCOL_TAIL, verbatim.
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

# next_compression/tools.py ToolPreparationConfig defaults (the frozen T0
# training manifest binds the same values; the training-side selection run
# re-validates them against its manifest).
DEFAULT_CHUNK_TOKENS = 768
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_MAX_CHUNKS = 48
DEFAULT_MAX_TOOL_TOKENS = 36864


class ToolMemoryError(RuntimeError):
    """A tool-memory contract violation; the request must fail, not degrade."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"[{kind}] {detail}")
        self.kind = kind
        self.detail = detail


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolMemorySpec:
    ratio: int
    layout: str = "uniform"
    top_k: int = 0
    encoder: str = "t0"
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_chunks: int = DEFAULT_MAX_CHUNKS
    max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS
    interface_policy: str = "none"
    selector_policy: str = DEFAULT_SELECTOR_POLICY

    def validate(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"unknown tool-memory encoder {self.encoder!r}")
        if self.interface_policy not in ("none", "schema"):
            raise ValueError(f"unknown tool interface policy {self.interface_policy!r}")
        if self.selector_policy not in SELECTOR_POLICIES:
            raise ValueError(f"unknown tool selector policy {self.selector_policy!r}")
        if self.ratio not in SUPPORTED_RATIOS:
            raise ValueError(
                f"tool-memory ratio must be one of {SUPPORTED_RATIOS}, got {self.ratio}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown tool-memory layout {self.layout!r}")
        if self.layout == "hybrid" and self.top_k < 1:
            raise ValueError("hybrid tool memory needs top_k >= 1")
        if self.layout == "uniform" and self.top_k:
            raise ValueError("uniform tool memory keeps no native schemas")
        if self.chunk_tokens < 1 or not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("chunk_overlap must be in [0, chunk_tokens)")

    @property
    def name(self) -> str:
        base = f"{self.encoder}_r{self.ratio}"
        layout_name = base if self.layout == "uniform" else f"{base}_{self.layout}{self.top_k}"
        interface_name = (layout_name if self.interface_policy == "none"
                          else f"{layout_name}_schema")
        if self.selector_policy == DEFAULT_SELECTOR_POLICY:
            return interface_name
        return f"{interface_name}_selector_{self.selector_policy}"

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "schema": TOOL_MEMORY_SCHEMA, "name": self.name, "encoder": self.encoder,
            "ratio": self.ratio, "layout": self.layout, "top_k": self.top_k,
            "render_profile": RENDER_PROFILE, "ranker": RANKER,
            "chunk_tokens": self.chunk_tokens, "chunk_overlap": self.chunk_overlap,
            "max_chunks": self.max_chunks, "max_tool_tokens": self.max_tool_tokens,
        }
        if self.interface_policy != "none":
            result["interface_policy"] = self.interface_policy
            result["interface_render_profile"] = INTERFACE_RENDER_PROFILE
            result["description_document_profile"] = DESCRIPTION_DOCUMENT_PROFILE
        if self.selector_policy != DEFAULT_SELECTOR_POLICY:
            result["selector_policy"] = self.selector_policy
            result["selector_version"] = selector_version(self.selector_policy)
        return result


_SPEC_RE = re.compile(
    r"^(?P<encoder>t0|streamingllm|h2o|snapkv|pyramidkv):r(?P<ratio>\d+)"
    r"(?::(?P<layout>uniform|hybrid(?P<k>\d+)))?(?::(?P<interface>schema))?"
    r"(?::selector=(?P<selector>last_user_topk_v1|latest_event_topk_v1|last_user_adaptive_v1))?$")


def parse_tool_memory_spec(text: Optional[str]) -> Optional[ToolMemorySpec]:
    """Parse the codec/layout and an optional versioned selection policy."""
    value = (text or "").strip().lower()
    if value in ("", "none", "full", "raw"):
        return None
    match = _SPEC_RE.match(value)
    if not match:
        raise ValueError(
            f"unparseable tool-memory spec {text!r}; expected "
            "<encoder>:r<8|12>[:uniform|:hybrid<k>][:schema][:selector=<policy>]")
    layout = match.group("layout") or "uniform"
    top_k = int(match.group("k")) if match.group("k") else 0
    spec = ToolMemorySpec(
        ratio=int(match.group("ratio")), encoder=match.group("encoder"),
        layout="hybrid" if layout.startswith("hybrid") else "uniform", top_k=top_k,
        interface_policy=match.group("interface") or "none",
        selector_policy=match.group("selector") or DEFAULT_SELECTOR_POLICY)
    spec.validate()
    return spec


def validate_ready_tool_contract(
    manifest: Mapping[str, Any], spec_text: Optional[str],
    checkpoint: Optional[str | Path] = None,
    budget_tokens: Optional[int] = None,
    tool_recovery: str = "none",
) -> None:
    """Ensure a supervised native child actually loaded the selected tool arm."""
    spec = parse_tool_memory_spec(spec_text)
    if spec is None:
        if tool_recovery != "none":
            raise RuntimeError("Tool recovery requires an active tool memory arm")
        return
    loaded = manifest.get("tool_memory_contract")
    if (not isinstance(loaded, Mapping)
            or loaded.get("spec") != spec.as_dict()
            or loaded.get("tool_budget_tokens") != budget_tokens
            or loaded.get("tool_recovery", "none") != tool_recovery):
        raise RuntimeError("Native tool memory contract differs from the selected tool arm")
    if spec.encoder == "t0" and checkpoint is not None:
        loaded_checkpoint = loaded.get("checkpoint")
        if (not isinstance(loaded_checkpoint, Mapping)
                or Path(str(loaded_checkpoint.get("checkpoint", ""))).resolve()
                != Path(checkpoint).resolve()):
            raise RuntimeError("Native tool memory checkpoint differs from the selected tool arm")


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def load_tool_checkpoint_contract(checkpoint: Path, spec: ToolMemorySpec) -> Dict[str, Any]:
    """Read ``<checkpoint>/config.json`` and refuse anything but a T0 encoder
    that supports ``spec.ratio`` under the pinned render profile."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolMemoryError("tool_checkpoint", f"cannot read {config_path}: {error}") from error
    domain = config.get("history_memory_compression_domain")
    variant = config.get("history_memory_variant")
    profile = config.get("history_memory_render_profile")
    ratios = [int(r) for r in (config.get("history_memory_supported_ratios") or [])]
    if domain != "tool" or variant != "T0":
        raise ToolMemoryError(
            "tool_checkpoint",
            f"{checkpoint} is domain={domain!r} variant={variant!r}; tool memory needs a T0 encoder")
    if profile != RENDER_PROFILE:
        raise ToolMemoryError(
            "tool_checkpoint", f"{checkpoint} renders {profile!r}, proxy implements {RENDER_PROFILE!r}")
    if spec.ratio not in ratios:
        raise ToolMemoryError(
            "tool_checkpoint", f"{checkpoint} supports ratios {ratios}, spec asks {spec.ratio}")
    state: Dict[str, Any] = {}
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_file():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            state = {k: raw.get(k) for k in ("global_step", "parameter_version", "completed")}
        except (OSError, json.JSONDecodeError):
            state = {}
    return {
        "checkpoint": str(checkpoint),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "variant": variant, "compression_domain": domain, "render_profile": profile,
        "supported_ratios": ratios, "corpus_identity": config.get("history_memory_corpus_identity"),
        "trainer_state": state,
    }


# ---------------------------------------------------------------------------
# Fixed lexical ranker (next_compression/exp1_tools.py, verbatim port)
# ---------------------------------------------------------------------------


def tool_snapshot(value: Any) -> Any:
    """JSON-owned copy; rejects non-finite / non-JSON tool metadata (tools._json_snapshot)."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    return str(function.get("name") or tool.get("name") or tool.get("tool_name")
               or tool.get("function_name") or "")


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
        tool.get("text", ""),
    ]
    return " ".join(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                    for item in fields if item)


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def query_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The ranker reads the latest user message (scope ``last_user``)."""
    return last_user_query(messages)


TOOL_SPANS_FIELD = "c2kv_tool_spans_v1"
INLINE_TOOL_PLACEHOLDER = "[Tool definition {identity} available in compressed memory]"


@dataclass(frozen=True)
class VisibleToolSpan:
    """An exact, source-provided text interval already visible to the model."""

    message_index: int
    start: int
    end: int
    source: str
    text: str


def source_span_identity(span: VisibleToolSpan) -> str:
    """Stable across turns when an observed definition moves in history."""
    return hashlib.sha256((span.source + "\n" + span.text).encode("utf-8")).hexdigest()[:16]


def source_span_placeholder(span: VisibleToolSpan) -> str:
    return INLINE_TOOL_PLACEHOLDER.format(identity=source_span_identity(span))


def visible_tool_snapshot(span: VisibleToolSpan) -> Dict[str, Any]:
    """Use a visible JSON schema directly; preserve other docs as exact text."""
    try:
        value = json.loads(span.text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        return tool_snapshot(value)
    return {"source": span.source, "text": span.text}


def strip_request_annotations(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove proxy-private source annotations; preserve object identity if absent."""
    if TOOL_SPANS_FIELD not in payload:
        return payload
    return {key: value for key, value in payload.items() if key != TOOL_SPANS_FIELD}


def tool_visibility_status(payload: Mapping[str, Any]) -> str:
    """One receipt value even when the compressor has nothing visible to do."""
    if resolve_visible_tool_spans(payload):
        return "visible_source_spans"
    if payload.get("tools"):
        return "structured_tools"
    return "no_visible_definition"


def resolve_visible_tool_spans(payload: Mapping[str, Any]) -> Tuple[VisibleToolSpan, ...]:
    """Validate exact intervals supplied by the benchmark's source adapter.

    There is deliberately no prompt-wide regex or gold/catalog lookup here.
    An absent annotation means that no inline definition was observed.
    """
    annotations = payload.get(TOOL_SPANS_FIELD) or []
    if not isinstance(annotations, list):
        raise ToolMemoryError("tool_spans", f"{TOOL_SPANS_FIELD} must be a list")
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ToolMemoryError("tool_spans", "messages must be a list")
    spans: List[VisibleToolSpan] = []
    for item in annotations:
        if not isinstance(item, Mapping):
            raise ToolMemoryError("tool_spans", "each span must be an object")
        index, start, end, source = (item.get("message_index"), item.get("start"),
                                     item.get("end"), item.get("source"))
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (index, start, end)):
            raise ToolMemoryError("tool_spans", "message_index/start/end must be integers")
        if not isinstance(source, str) or not source:
            raise ToolMemoryError("tool_spans", "source must identify the official producer")
        if not 0 <= index < len(messages):
            raise ToolMemoryError("tool_spans", f"message_index {index} is outside messages")
        content = messages[index].get("content")
        if not isinstance(content, str) or not 0 <= start < end <= len(content):
            raise ToolMemoryError("tool_spans", f"invalid text interval {index}:{start}:{end}")
        spans.append(VisibleToolSpan(index, start, end, source, content[start:end]))
    spans.sort(key=lambda span: (span.message_index, span.start, span.end))
    for previous, current in zip(spans, spans[1:]):
        if previous.message_index == current.message_index and previous.end > current.start:
            raise ToolMemoryError("tool_spans", "source spans overlap")
    return tuple(spans)


def remove_visible_spans(messages: Sequence[Mapping[str, Any]],
                         spans: Sequence[VisibleToolSpan],
                         removed_indices: Sequence[int], *,
                         placeholder_indices: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """Replace compressed intervals only; retain surrounding text verbatim."""
    out = [dict(message) for message in messages]
    removed = set(removed_indices)
    placeholders = removed if placeholder_indices is None else set(placeholder_indices)
    for span_index in reversed(range(len(spans))):
        if span_index not in removed:
            continue
        span = spans[span_index]
        content = out[span.message_index]["content"]
        replacement = source_span_placeholder(span) if span_index in placeholders else ""
        out[span.message_index]["content"] = (
            content[:span.start] + replacement + content[span.end:])
    return out


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def lexical_rank(tools: Sequence[Mapping[str, Any]], query: str) -> Tuple[int, ...]:
    """Catalog order by descending score; name overlap weighs four text overlaps."""
    return selection_lexical_rank(tools, query)


# ---------------------------------------------------------------------------
# Layout, protocol, documents, chunks (next_compression/tools.py semantics)
# ---------------------------------------------------------------------------


def native_indices(tools: Sequence[Mapping[str, Any]], spec: ToolMemorySpec,
                   messages: Sequence[Mapping[str, Any]]) -> Tuple[int, ...]:
    return tuple(tool_selection(tools, spec, messages)["native_indices"])


def t0_documents(snapshots: Sequence[Any], indices: Sequence[int]) -> Tuple[Dict[str, Any], ...]:
    """Byte-identical to the T0 document format in tools.tool_variant_material."""
    return tuple({"type": "tool_definition", "tool_index": int(index), "tool": snapshots[index]}
                 for index in indices)


def description_documents(snapshots: Sequence[Any],
                          indices: Sequence[int]) -> Tuple[Dict[str, Any], ...]:
    """Versioned T0 inputs: prose and positional binding metadata only."""
    return tuple(document for index in indices
                 if (document := description_document(snapshots[index], int(index))) is not None)


def protocol_block(native_tools: Sequence[Any]) -> str:
    schemas = "".join("\n" + json.dumps(tool, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False)
                      for tool in native_tools)
    return TOOL_PROTOCOL_HEAD + schemas + TOOL_PROTOCOL_TAIL


def executable_interfaces(snapshots: Sequence[Any], spans: Sequence[VisibleToolSpan],
                          compressed: Sequence[int], native: Sequence[int], n_structured: int,
                          spec: ToolMemorySpec) -> Dict[int, InterfaceCopy]:
    """Render each compressed interface, and each moved source, exactly once."""
    if spec.interface_policy != "schema":
        return {}
    copies = {
        index: interface_copy(
            snapshots[index],
            source_text=spans[index - n_structured].text if index >= n_structured else None)
        for index in compressed
    }
    for index in native:
        if index >= n_structured:
            copies[index] = InterfaceCopy(spans[index - n_structured].text, True,
                                          "native_source_full")
    return dict(sorted(copies.items()))


def protocol_with_interfaces(native_tools: Sequence[Any],
                             copies: Mapping[int, InterfaceCopy], *,
                             structured: bool, label_indices: bool = False) -> Tuple[str, Tuple[Dict[str, Any], ...]]:
    """Build protocol text and character intervals for its protected copies."""
    base = protocol_block(native_tools) if structured else ""
    if not copies:
        return base, ()
    prefix = base[:-len(TOOL_PROTOCOL_TAIL)] if structured else ""
    suffix = TOOL_PROTOCOL_TAIL if structured else ""
    prefix += INTERFACE_BLOCK_HEAD
    intervals = []
    for index, copy in copies.items():
        prefix += "\n"
        if label_indices:
            prefix += f"[tool_index:{index}]\n"
        start = len(prefix)
        prefix += copy.text
        intervals.append({"catalog_index": index, "start": start, "end": len(prefix),
                          "text": copy.text, "fallback_full": copy.fallback,
                          "copy_reason": copy.reason})
    return prefix + suffix, tuple(intervals)


def has_protocol(messages: Sequence[Mapping[str, Any]]) -> bool:
    first = messages[0] if messages else {}
    return bool(first.get("role") == "system" and isinstance(first.get("content"), str)
                and TOOL_PROTOCOL_HEAD in first["content"])


def with_protocol_system(messages: Sequence[Mapping[str, Any]], protocol: str) -> List[Dict[str, Any]]:
    """tools._native_tool_memory: append the protocol to the leading system
    message, or insert a new system message when there is none."""
    out = [dict(m) for m in messages]
    if not protocol:
        return out
    if out and out[0].get("role") == "system":
        content = out[0].get("content")
        if not isinstance(content, str):
            raise ToolMemoryError("system_content", "system instruction content must be text")
        if protocol in content or (INTERFACE_BLOCK_HEAD not in protocol
                                   and TOOL_PROTOCOL_HEAD in content):
            return out  # idempotent
        out[0]["content"] = content + "\n\n" + protocol
    else:
        out.insert(0, {"role": "system", "content": protocol})
    return out


def locate_interface_spans(messages: Sequence[Mapping[str, Any]], protocol: str,
                           intervals: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    if not intervals:
        return ()
    content = messages[0].get("content") if messages else None
    start = content.rfind(protocol) if isinstance(content, str) else -1
    if start < 0:
        raise ToolMemoryError("interface_span", "protected interface protocol is missing")
    spans = tuple({**interval, "message_index": 0,
                   "start": start + int(interval["start"]),
                   "end": start + int(interval["end"])} for interval in intervals)
    for span in spans:
        if content[span["start"]:span["end"]] != span["text"]:
            raise ToolMemoryError("interface_span", "protected interface text does not match")
    return spans


def document_identity(document: Mapping[str, Any]) -> str:
    identity = json.dumps(document, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def document_envelope(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Chunk:
    event_id: str
    part_index: int
    source_token_start: int
    source_token_end: int
    token_ids: Tuple[int, ...]
    catalog_index: int = -1


def document_chunks(native_ids: Callable[[Sequence[Mapping[str, Any]]], Sequence[int]],
                    documents: Sequence[Mapping[str, Any]], spec: ToolMemorySpec) -> List[Chunk]:
    """tools._document_chunks: one user-message rendering per document, cut
    into <= chunk_tokens slices with chunk_overlap carry-back."""
    chunks: List[Chunk] = []
    for document_index, document in enumerate(documents):
        ids = tuple(int(t) for t in native_ids(({"role": "user", "content": document_envelope(document)},)))
        event_id = f"{spec.encoder}-tool-{document_index}-{document_identity(document)}"
        start = 0
        part_index = 0
        while start < len(ids):
            end = min(start + spec.chunk_tokens, len(ids))
            chunks.append(Chunk(event_id, part_index, start, end, ids[start:end],
                                int(document.get("tool_index", document_index))))
            if end == len(ids):
                break
            start = end - spec.chunk_overlap
            part_index += 1
    presented = sum(len(chunk.token_ids) for chunk in chunks)
    if presented > spec.max_tool_tokens:
        raise ToolMemoryError("tool_tokens_over_limit",
                              f"{presented} presented encoder tokens > {spec.max_tool_tokens}")
    if len(chunks) > spec.max_chunks:
        raise ToolMemoryError("tool_chunks_over_limit", f"{len(chunks)} chunks > {spec.max_chunks}")
    if not chunks:
        raise ToolMemoryError("no_tool_document_tokens", "compressed remainder has no tokens")
    return chunks


def expected_gist_len(token_count: int, ratio: int) -> int:
    return (token_count + ratio - 1) // ratio


@dataclass
class VisibleToolPlan:
    """Pure catalog selection before any model or server extraction."""

    spec: ToolMemorySpec
    messages: List[Dict[str, Any]]
    protocol: str
    chunks: List[Chunk]
    source_spans: Tuple[VisibleToolSpan, ...]
    compressed_source_indices: Tuple[int, ...]
    compressed_tool_indices: Tuple[int, ...]
    carrier_anchors: List[Dict[str, Any]]
    info: Dict[str, Any]
    raw_schema_spans: Tuple[Dict[str, Any], ...] = ()
    assembled_schema_spans: Optional[Tuple[Dict[str, Any], ...]] = None
    interface_spans: Tuple[Dict[str, Any], ...] = ()


def _isolated_content_tokens(tokenizer: Any, content: str) -> int:
    envelope = ({"role": "user", "content": ""},)
    rendered = ({"role": "user", "content": content},)
    return max(0, len(tokenizer.native_ids(rendered)) - len(tokenizer.native_ids(envelope)))


def enforce_tool_budget(plan: VisibleToolPlan | "ToolMemoryPlan",
                        budget_tokens: Optional[int]) -> None:
    """Cap total resident T0 tool slots; no cap means the frozen ratio policy."""
    if budget_tokens is None:
        return
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 1:
        raise ValueError("tool budget_tokens must be a positive integer")
    resident = plan.info.get("resident_tool_tokens")
    if resident is None:
        raise ToolMemoryError("tool_budget", "raw-KV budget needs post-assembly token selection")
    if int(resident) > budget_tokens:
        raise ToolMemoryError("tool_budget",
                              f"{resident} resident tool tokens exceed budget {budget_tokens}")


def plan_visible_tool_memory(payload: Mapping[str, Any], spec: ToolMemorySpec,
                             tokenizer: Any = None, *,
                             native_override: Optional[Sequence[int]] = None,
                             retrieval_only: bool = False) -> Optional[VisibleToolPlan]:
    """Plan one tool context from the request's *visible* definitions only.

    Structured ``tools`` and adapter-provided exact text spans share one
    catalog ranker and one global selection. T0 retains its training document
    envelope and chunking. Raw-KV plans preserve original messages so the
    final, query-bearing assembled prompt can be scored after history packing.
    """
    spec.validate()
    tools = payload.get("tools") or []
    if not isinstance(tools, list) or any(not isinstance(tool, Mapping) for tool in tools):
        raise ToolMemoryError("tools", "tools must be a list of objects")
    messages = list(payload.get("messages") or [])
    if any(not isinstance(message, Mapping) for message in messages):
        raise ToolMemoryError("messages", "messages must be a list of objects")
    spans = resolve_visible_tool_spans(payload)
    if not tools and not spans:
        return None

    snapshots = [tool_snapshot(tool) for tool in tools]
    for span in spans:
        snapshots.append(visible_tool_snapshot(span))
    selection = tool_selection(snapshots, spec, messages)
    score_selected = tuple(selection["native_indices"])
    requested_native = (score_selected if native_override is None
                        else tuple(sorted(int(index) for index in native_override)))
    native = requested_native
    opaque = []
    if spec.interface_policy == "schema":
        # Unknown producer formats have no safe interface/prose split. Keep
        # their original source intact and never send it through the encoder.
        for source_index, span in enumerate(spans):
            try:
                parsed = json.loads(span.text)
            except (TypeError, ValueError):
                parsed = None
            if compact_tool(parsed) is None:
                opaque.append(len(tools) + source_index)
        native = tuple(sorted(set(native) | set(opaque)))
    interface_forced = tuple(sorted(set(opaque) - set(requested_native)))
    if len(set(native)) != len(native) or any(index < 0 or index >= len(snapshots) for index in native):
        raise ValueError("native tool indices must be unique catalog indices")
    native_set = set(native)
    compressed = tuple(index for index in range(len(snapshots)) if index not in native_set)
    n_structured = len(tools)
    compressed_tools = tuple(index for index in compressed if index < n_structured)
    compressed_sources = tuple(index - n_structured for index in compressed if index >= n_structured)
    interfaces = executable_interfaces(snapshots, spans, compressed, native,
                                        n_structured, spec)
    documents = (description_documents(snapshots, compressed)
                 if spec.interface_policy == "schema" else t0_documents(snapshots, compressed))
    document_indices = {int(document["tool_index"]) for document in documents}
    anchored_sources = (tuple(index for index in compressed_sources
                              if n_structured + index in document_indices)
                        if spec.interface_policy == "schema" else compressed_sources)
    anchors = [{
        "catalog_index": n_structured + span_index,
        "message_index": spans[span_index].message_index,
        "start": spans[span_index].start,
        "end": spans[span_index].end,
        "source": spans[span_index].source,
        "identity": source_span_identity(spans[span_index]),
        "placeholder": source_span_placeholder(spans[span_index]),
    } for span_index in anchored_sources]
    if spec.encoder == "t0":
        if tokenizer is None:
            raise ValueError("T0 planning needs the checkpoint tokenizer")
        native_tools = [snapshots[index] for index in native if index < n_structured]
        protocol, interface_intervals = protocol_with_interfaces(
            native_tools, interfaces, structured=bool(n_structured),
            label_indices=spec.interface_policy == "schema")
        removed_sources = (tuple(range(len(spans))) if spec.interface_policy == "schema"
                           else compressed_sources)
        rewritten = remove_visible_spans(
            messages, spans, removed_sources,
            placeholder_indices=(anchored_sources if spec.interface_policy == "schema" else None))
        if protocol:
            rewritten = with_protocol_system(rewritten, protocol)
            # Inserting a leading system message shifts source anchors.
            if not messages or messages[0].get("role") != "system":
                for anchor in anchors:
                    anchor["rewritten_message_index"] = anchor["message_index"] + 1
        chunks = (document_chunks(tokenizer.native_ids, documents, spec)
                  if documents and not retrieval_only else [])
        system_only = [m for m in messages if m.get("role") == "system"][:1] or [
            {"role": "system", "content": ""}]
        protocol_prefix = [m for m in rewritten if m.get("role") == "system"][:1]
        protocol_tokens = (len(tokenizer.native_ids(protocol_prefix))
                           - len(tokenizer.native_ids(system_only))) if protocol else 0
        if interfaces:
            baseline_protocol = protocol_block(native_tools) if n_structured else ""
            baseline_messages = with_protocol_system(
                remove_visible_spans(
                    messages, spans, removed_sources,
                    placeholder_indices=(anchored_sources if spec.interface_policy == "schema" else None)),
                baseline_protocol)
            baseline_prefix = [m for m in baseline_messages if m.get("role") == "system"][:1] or [
                {"role": "system", "content": ""}]
            interface_copy_tokens = (len(tokenizer.native_ids(protocol_prefix))
                                     - len(tokenizer.native_ids(baseline_prefix)))
        else:
            interface_copy_tokens = 0
        interface_spans = locate_interface_spans(rewritten, protocol, interface_intervals)
        expected_gist_tokens = sum(expected_gist_len(len(chunk.token_ids), spec.ratio)
                                   for chunk in chunks)
        native_source_tokens = (0 if spec.interface_policy == "schema" else
                                sum(_isolated_content_tokens(tokenizer, spans[index].text)
                                    for index in range(len(spans)) if index not in compressed_sources))
        resident_tool_tokens = protocol_tokens + expected_gist_tokens + native_source_tokens
    else:
        protocol = ""
        rewritten = [dict(message) for message in messages]
        chunks = []
        protocol_tokens = expected_gist_tokens = native_source_tokens = resident_tool_tokens = None
        interface_copy_tokens = 0
        interface_spans = ()
    for anchor in anchors:
        anchor.setdefault("rewritten_message_index", anchor["message_index"])
    selector_info = ({
        "selector_policy": selection["policy"],
        "selector_version": selection["selector_version"],
        "selector_scores": list(selection["scores"]),
        "selector_rank": list(selection["rank"]),
        "selector_query_sha256": selection["query_sha256"],
        "selector_latest_io_present": selection["latest_io_present"],
        "score_selected_native_indices": list(score_selected),
        "interface_forced_native_indices": list(interface_forced),
        **({"native_override_indices": list(requested_native)}
           if native_override is not None else {}),
        **({"relative_threshold": ADAPTIVE_RELATIVE_THRESHOLD,
            "selection_count": len(score_selected)}
           if spec.selector_policy == "last_user_adaptive_v1" else {}),
    } if spec.selector_policy != DEFAULT_SELECTOR_POLICY else {})
    info = {
        "schema": TOOL_MEMORY_SCHEMA, "spec": spec.name, "encoder": spec.encoder,
        "ratio": spec.ratio, "layout": spec.layout,
        "top_k": (None if spec.selector_policy == "last_user_adaptive_v1" else spec.top_k),
        "interface_policy": spec.interface_policy,
        **selector_info,
        **({"interface_render_profile": INTERFACE_RENDER_PROFILE}
           if spec.interface_policy == "schema" else {}),
        **({"description_document_profile": DESCRIPTION_DOCUMENT_PROFILE}
           if spec.interface_policy == "schema" else {}),
        "ranker": RANKER, "render_profile": RENDER_PROFILE if spec.encoder == "t0" else "native-raw-kv",
        "representation": "t0_gist" if spec.encoder == "t0" else "selected_raw_kv",
        "action_protocol": "benchmark_original" if spans and not tools else "structured_tool_call",
        "structured_tools_in_prompt": spec.encoder != "t0" or not tools,
        "n_tools": len(snapshots), "n_structured_tools": n_structured,
        "n_visible_source_spans": len(spans), "n_native": len(native),
        "all_native": not compressed,
        "native_indices": list(native), "n_documents": len(documents) if spec.encoder == "t0" else 0,
        "n_chunks": len(chunks), "compressed_tool_indices": list(compressed_tools),
        "compressed_source_indices": list(compressed_sources),
        "source_status": "visible" if spans else ("structured_tools" if tools else "no_visible_definition"),
        "carrier_anchors": anchors,
        "protocol_prefix_tokens": protocol_tokens,
        "expected_gist_tokens": expected_gist_tokens,
        "native_source_tokens": native_source_tokens,
        "resident_tool_tokens": resident_tool_tokens,
        "interface_copy_tokens": interface_copy_tokens,
        "n_protected_interfaces": len(interfaces),
        "n_interface_fallbacks": len(opaque) + sum(
            copy.reason == "opaque_full" for copy in interfaces.values()),
        "interface_fallback_indices": sorted(set(opaque) | {
            index for index, copy in interfaces.items() if copy.reason == "opaque_full"}),
        "n_native_source_interface_copies": sum(
            copy.reason == "native_source_full" for copy in interfaces.values()),
        "native_source_interface_copy_indices": [index for index, copy in interfaces.items()
                                                 if copy.reason == "native_source_full"],
    }
    return VisibleToolPlan(spec, rewritten, protocol, chunks, spans,
                           compressed_sources, compressed_tools, anchors, info,
                           interface_spans=interface_spans)


def persistent_schema_tool_plan(payload: Mapping[str, Any], plan: VisibleToolPlan,
                                tokenizer: Any) -> VisibleToolPlan:
    """Place a frozen selection in a source-stable persistent document frame.

    Every executable interface stays in the same system protocol. Each prose
    document keeps its original T0 envelope and chunk IDs, whether its current
    view is full base KV or a tool-projection gist. Selection is never rerun.
    """
    if plan.spec.encoder != "t0" or plan.spec.interface_policy != "schema":
        return plan
    if plan.info.get("persistent_render_profile") == PERSISTENT_RENDER_PROFILE:
        return plan
    tools = list(payload.get("tools") or [])
    messages = list(payload.get("messages") or [])
    snapshots = [tool_snapshot(tool) for tool in tools] + [
        visible_tool_snapshot(span) for span in plan.source_spans]
    indices = tuple(range(len(snapshots)))
    copies = executable_interfaces(
        snapshots, plan.source_spans, indices, (), len(tools), plan.spec)
    protocol, intervals = protocol_with_interfaces(
        (), copies, structured=bool(tools), label_indices=True)
    rewritten = with_protocol_system(remove_visible_spans(
        messages, plan.source_spans, range(len(plan.source_spans)),
        placeholder_indices=()), protocol)
    documents = description_documents(snapshots, indices)
    chunks = document_chunks(tokenizer.native_ids, documents, plan.spec) if documents else []
    native = set(plan.info["native_indices"])
    native_tokens = sum(len(chunk.token_ids) for chunk in chunks if chunk.catalog_index in native)
    compressed = [chunk for chunk in chunks if chunk.catalog_index not in native]
    gist_tokens = sum(expected_gist_len(len(chunk.token_ids), plan.spec.ratio)
                      for chunk in compressed)
    source_system = [m for m in messages if m.get("role") == "system"][:1] or [
        {"role": "system", "content": ""}]
    protocol_system = [m for m in rewritten if m.get("role") == "system"][:1]
    protocol_tokens = (len(tokenizer.native_ids(protocol_system))
                       - len(tokenizer.native_ids(source_system))) if protocol else 0
    baseline = with_protocol_system(remove_visible_spans(
        messages, plan.source_spans, range(len(plan.source_spans)),
        placeholder_indices=()), protocol_block(()) if tools else "")
    baseline_system = [m for m in baseline if m.get("role") == "system"][:1] or [
        {"role": "system", "content": ""}]
    interface_tokens = (len(tokenizer.native_ids(protocol_system))
                        - len(tokenizer.native_ids(baseline_system))) if copies else 0
    prefix_count = 0
    while prefix_count < len(rewritten) and rewritten[prefix_count].get("role") == "system":
        prefix_count += 1
    prefix_ids = list(tokenizer.native_ids(rewritten[:prefix_count]))
    source_digest = hashlib.sha256(json.dumps({
        "profile": PERSISTENT_RENDER_PROFILE, "prefix_token_ids": prefix_ids,
        "documents": [list(chunk.token_ids) for chunk in chunks],
        "catalog": snapshots,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    info = {**plan.info,
        "render_profile": PERSISTENT_RENDER_PROFILE,
        "persistent_render_profile": PERSISTENT_RENDER_PROFILE,
        "representation": "source_stable_native_or_t0_document",
        "source_protocol_token_sha256": source_digest,
        "persistent_tool_slot_start": len(prefix_ids),
        "protocol_prefix_tokens": protocol_tokens,
        "interface_copy_tokens": interface_tokens,
        "native_source_tokens": native_tokens,
        "native_document_tokens": native_tokens,
        "expected_gist_tokens": gist_tokens,
        "presented_encoder_tokens": sum(len(chunk.token_ids) for chunk in compressed),
        "resident_tool_tokens": protocol_tokens + native_tokens + gist_tokens,
        "n_documents": len(documents), "n_chunks": len(chunks),
        "n_protected_interfaces": len(copies), "carrier_anchors": [],
        "n_native_source_interface_copies": 0,
        "native_source_interface_copy_indices": [],
    }
    return replace(plan, messages=rewritten, protocol=protocol, chunks=chunks,
                   carrier_anchors=[], info=info,
                   interface_spans=locate_interface_spans(rewritten, protocol, intervals))


# ---------------------------------------------------------------------------
# Tokenizer (transformers, local checkpoint files only)
# ---------------------------------------------------------------------------


class NativeTokenizer:
    """``history_memory.packing.native_ids`` semantics on the checkpoint tokenizer."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer  # heavy import, on demand
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(self.checkpoint), local_files_only=True)
        return self._tokenizer

    def native_ids(self, messages: Sequence[Mapping[str, Any]], *, tools: Any = None,
                   generation: bool = False) -> Tuple[int, ...]:
        if not messages:
            raise ValueError("the native text template needs at least one message")
        ids = self._load().apply_chat_template(
            list(messages), tools=tools, tokenize=True, add_generation_prompt=generation,
            enable_thinking=False, truncation=False)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return tuple(int(t) for t in ids)

    def sha256(self) -> str:
        path = self.checkpoint / "tokenizer.json"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


# ---------------------------------------------------------------------------
# Per-request plan and the proxy-facing service
# ---------------------------------------------------------------------------


@dataclass
class ToolMemoryPlan:
    spec: ToolMemorySpec
    messages: List[Dict[str, Any]]           # request messages with the protocol system
    protocol: str                            # the explicit protocol block (re-applied on re-assembly)
    chunks: List[Chunk]
    records: List[Dict[str, Any]]            # one extract record per chunk (key_hash, gist_len, ...)
    info: Dict[str, Any]
    carrier_anchors: List[Dict[str, Any]] = None
    source_spans: Tuple[VisibleToolSpan, ...] = ()
    interface_spans: Tuple[Dict[str, Any], ...] = ()

    def carriers(self) -> List[Dict[str, Any]]:
        anchors = {anchor["catalog_index"]: anchor for anchor in (self.carrier_anchors or [])}
        if self.info.get("persistent_render_profile") == PERSISTENT_RENDER_PROFILE:
            return [{
                "role": "user", "content": "", "c2kv_region": "tool",
                "c2kv_source_token_count": len(chunk.token_ids),
                "c2kv_use_gist_projection": False,
                **({"c2kv_repair_only_key_hashes": [record["key_hash"]],
                    "c2kv_repair_placement": "in_place",
                    "c2kv_source_token_end": record["source_start"] + len(chunk.token_ids)}
                   if record.get("native_document") else
                   {"c2kv_key_hash": record["key_hash"], "c2kv_ratio": self.spec.ratio}),
                CARRIER_MARK: {"event_id": chunk.event_id, "part_index": chunk.part_index,
                               "source_tokens": len(chunk.token_ids),
                               "gist_len": int(record.get("gist_len") or 0)},
            } for chunk, record in zip(self.chunks, self.records, strict=True)]
        return [{
            "role": "user", "content": "",
            "c2kv_key_hash": record["key_hash"],
            "c2kv_ratio": self.spec.ratio,
            "c2kv_region": "tool",
            "c2kv_source_token_count": len(chunk.token_ids),
            CARRIER_MARK: {"event_id": chunk.event_id, "part_index": chunk.part_index,
                           "source_tokens": len(chunk.token_ids), "gist_len": record["gist_len"],
                           "anchor": anchors.get(chunk.catalog_index)},
        } for chunk, record in zip(self.chunks, self.records)]


def is_carrier(message: Mapping[str, Any]) -> bool:
    return CARRIER_MARK in message


def strip_carrier_fields(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Carrier fields the server must not see (it only reads c2kv_key_hash)."""
    return {k: v for k, v in message.items() if k != CARRIER_MARK}


def _carrier_identity(carrier: Mapping[str, Any]) -> Tuple[Any, ...]:
    metadata = carrier.get(CARRIER_MARK) or {}
    anchor = metadata.get("anchor") or {}
    return (carrier.get("c2kv_key_hash"), metadata.get("event_id"),
            metadata.get("part_index"), anchor.get("identity"))


def insert_carriers(out_messages: List[Dict[str, Any]], counts: Dict[str, Any],
                    carriers: Sequence[Dict[str, Any]],
                    source_out_indices: Optional[Mapping[int, int]] = None,
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Insert static catalog carriers at prefix and source docs at their event.

    ``source_out_indices`` maps original source-message indices to assembled
    output indices. It is required for inline source spans because history
    assembly may merge, normalize, or insert messages. Existing carriers are
    deduplicated by content and source identity; later visible docs can append.
    """
    if not carriers:
        return out_messages, counts
    existing = {_carrier_identity(message) for message in out_messages if is_carrier(message)}
    pending = [carrier for carrier in carriers if _carrier_identity(carrier) not in existing]
    if not pending:
        return out_messages, counts
    prefix_at = 0
    while prefix_at < len(out_messages) and out_messages[prefix_at].get("role") == "system":
        prefix_at += 1
    placements: List[Tuple[int, Dict[str, Any]]] = []
    for carrier in pending:
        anchor = (carrier.get(CARRIER_MARK) or {}).get("anchor")
        if anchor:
            source_index = anchor["message_index"]
            if source_out_indices is None or source_index not in source_out_indices:
                raise ToolMemoryError("tool_anchor",
                                      f"no assembled message index for source message {source_index}")
            out_index = source_out_indices[source_index]
            if not isinstance(out_index, int) or not 0 <= out_index < len(out_messages):
                raise ToolMemoryError("tool_anchor", f"invalid assembled index {out_index}")
            insert_at = out_index + 1
        else:
            insert_at = prefix_at
        placements.append((insert_at, dict(carrier)))
    by_index: Dict[int, List[Dict[str, Any]]] = {}
    for insert_at, carrier in placements:
        by_index.setdefault(insert_at, []).append(carrier)
    out: List[Dict[str, Any]] = []
    for index in range(len(out_messages) + 1):
        out.extend(by_index.get(index, []))
        if index < len(out_messages):
            out.append(out_messages[index])
    def shifted(index: int) -> int:
        return index + sum(1 for insert_at, _ in placements if insert_at <= index)
    counts = dict(counts)
    for key in ("current_start_out_index", "task_packet_out_index"):
        value = counts.get(key)
        if isinstance(value, int):
            counts[key] = shifted(value)
    records = []
    for record in counts.get("compressed_records") or []:
        record = dict(record)
        idx = record.get("out_index")
        if isinstance(idx, int):
            record["out_index"] = shifted(idx)
        records.append(record)
    if records:
        counts["compressed_records"] = records
    original_events = counts.get("history_kv_event_messages")
    if original_events is not None:
        if len(original_events) != len(out_messages) or any(
            event.get("message_index") != index
            for index, event in enumerate(original_events)
        ):
            raise ToolMemoryError("tool_event_alignment",
                                  "history KV events must align with assembled messages")
        event_messages = []
        for index in range(len(out_messages) + 1):
            for carrier in by_index.get(index, []):
                event_messages.append({"message_index": len(event_messages),
                                       "role": carrier.get("role") or "user",
                                       "phase": "others"})
            if index < len(out_messages):
                event = dict(original_events[index])
                event["message_index"] = len(event_messages)
                event_messages.append(event)
        counts["history_kv_event_messages"] = event_messages
    counts["tool_memory_carriers"] = sum(1 for message in out if is_carrier(message))
    counts["tool_memory_insert_at"] = min(insert_at for insert_at, _ in placements)
    counts["tool_memory_insertions"] = [insert_at for insert_at, _ in placements]
    return out, counts


def prepare_raw_tool_plan(payload: Mapping[str, Any], plan: VisibleToolPlan,
                          tokenizer: Any, *, budget_tokens: Optional[int] = None,
                          target_resident_tokens: Optional[int] = None) -> VisibleToolPlan:
    """Prepare final raw-schema spans and reference budget without a checkpoint owner."""
    spec = plan.spec
    tools = list(payload.get("tools") or [])
    original = list(payload.get("messages") or [])
    snapshots = [tool_snapshot(tool) for tool in tools] + [
        visible_tool_snapshot(span) for span in plan.source_spans]
    native = set(plan.info["native_indices"])
    compressed = tuple(index for index in range(len(snapshots)) if index not in native)
    if spec.interface_policy == "schema":
        # The structured protocol is the sole full structured copy. Source
        # definitions move into the same system frame exactly once; their
        # prose values alone are eligible for raw-KV selection.
        interfaces = {
            len(tools) + index: InterfaceCopy(source.text, True,
                "native_source_full" if len(tools) + index in native else "raw_source_full")
            for index, source in enumerate(plan.source_spans)
        }
        rewritten_original = remove_visible_spans(
            original, plan.source_spans, tuple(range(len(plan.source_spans))),
            placeholder_indices=())
    else:
        interfaces = executable_interfaces(snapshots, plan.source_spans,
                                            compressed, tuple(sorted(native)), len(tools), spec)
        rewritten_original = original
    protocol, interface_intervals = protocol_with_interfaces(
        tools, interfaces, structured=bool(tools),
        label_indices=spec.interface_policy == "schema")
    plan.protocol = protocol
    plan.messages = with_protocol_system(rewritten_original, protocol)
    all_interface_spans = locate_interface_spans(
        plan.messages, protocol, interface_intervals)
    plan.interface_spans = (tuple(span for span in all_interface_spans
                                  if span["catalog_index"] in native)
                            if spec.interface_policy == "schema" else all_interface_spans)
    baseline = with_protocol_system(rewritten_original, protocol_block(tools) if tools else "")
    baseline_prefix = (baseline[:1] if baseline and baseline[0].get("role") == "system"
                       else [{"role": "system", "content": ""}])
    plan.info["interface_copy_tokens"] = (
        len(tokenizer.native_ids(plan.messages[:1]))
        - len(tokenizer.native_ids(baseline_prefix))) if interfaces else 0
    plan.info["protected_interface_spans"] = list(plan.interface_spans)
    spans = []
    full_structured_spans = []
    if protocol:
        content = plan.messages[0]["content"]
        start = content.rfind(protocol)
        if start < 0:
            raise ToolMemoryError("raw_protocol", "the tool protocol is missing")
        plan.info["tool_protocol_span"] = {"message_index": 0, "start": start,
                                            "end": start + len(protocol), "text": protocol}
    if tools:
        cursor = start + len(TOOL_PROTOCOL_HEAD)
        for index, tool in enumerate(tools):
            text = json.dumps(tool, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            cursor += 1  # protocol_block puts a newline before each schema.
            full_structured_spans.append({"schema_index": index, "message_index": 0,
                                          "start": cursor, "end": cursor + len(text),
                                          "text": text})
            if spec.interface_policy == "schema":
                paths = {tuple(item["path"]) for item in tool_prose(tool)}
                for value_span in json_string_value_spans(text, paths):
                    spans.append({"schema_index": index, "message_index": 0,
                                  "start": cursor + value_span["start"],
                                  "end": cursor + value_span["end"],
                                  "text": value_span["text"]})
            else:
                spans.append({"schema_index": index, "message_index": 0,
                              "start": cursor, "end": cursor + len(text), "text": text})
            cursor += len(text)
    if spec.interface_policy == "schema":
        for source_span in all_interface_spans:
            index = source_span["catalog_index"]
            if index in native:
                continue
            paths = {tuple(item["path"]) for item in tool_prose(snapshots[index])}
            for value_span in json_string_value_spans(source_span["text"], paths):
                spans.append({"schema_index": index, "message_index": 0,
                              "start": source_span["start"] + value_span["start"],
                              "end": source_span["start"] + value_span["end"],
                              "text": value_span["text"]})
    else:
        shift = int(bool(protocol) and (not original or original[0].get("role") != "system"))
        for index, source in enumerate(plan.source_spans):
            spans.append({"schema_index": len(tools) + index,
                          "message_index": source.message_index + shift,
                          "start": source.start, "end": source.end, "text": source.text})
    if spec.interface_policy == "schema" and tools and not spans:
        # The engine requires an exact schema span to issue an eviction receipt.
        # With no selectable prose, full structured schemas are protected and
        # the engine reports its supported all-protected no-op.
        spans.extend(full_structured_spans)
        plan.info["raw_noop_schema_indices"] = list(range(len(tools)))
    for span in spans:
        content = plan.messages[span["message_index"]].get("content")
        if not isinstance(content, str) or content[span["start"]:span["end"]] != span["text"]:
            raise ToolMemoryError("raw_schema_span", "raw schema no longer matches its source")
    plan.raw_schema_spans = tuple(spans)
    # Match the T0 packing calculation; neither path changes the original tool.
    reference = plan_visible_tool_memory(
        payload, replace(spec, encoder="t0"), tokenizer,
        native_override=plan.info["native_indices"])
    matched = (len(tokenizer.native_ids(reference.messages, generation=True))
               + int(reference.info["expected_gist_tokens"]))
    if target_resident_tokens is not None:
        matched = int(target_resident_tokens)
    if matched < 1:
        raise ValueError("target_resident_tokens must be positive")
    plan.info.update({"structured_tools_in_prompt": not bool(tools),
                      "render_profile": RENDER_PROFILE if tools else "benchmark_original",
                      "target_resident_tokens_per_layer": matched,
                      "matched_resident_tool_tokens": reference.info["resident_tool_tokens"],
                      "reference_tool_gist_tokens": reference.info["expected_gist_tokens"],
                      "reference_native_source_tokens": reference.info["native_source_tokens"],
                      "reference_protocol_prefix_tokens": reference.info["protocol_prefix_tokens"],
                      "budget_tokens": budget_tokens,
                      "selection_backend": "sglang_reference_attention",
                      "history_policy": "full"})
    return plan


class ToolMemory:
    """Proxy-side owner of the tool-memory axis (one instance per proxy process)."""

    def __init__(self, spec: ToolMemorySpec, checkpoint: Path,
                 extract_tokens: Callable[[Sequence[int], int, str], Dict[str, Any]],
                 tokenizer: Optional[NativeTokenizer] = None,
                 budget_tokens: Optional[int] = None):
        spec.validate()
        if budget_tokens is not None and (isinstance(budget_tokens, bool)
                                          or not isinstance(budget_tokens, int)
                                          or budget_tokens < 1):
            raise ValueError("tool budget_tokens must be a positive integer")
        self.spec = spec
        self.budget_tokens = budget_tokens
        self.checkpoint = Path(checkpoint)
        self.contract = (load_tool_checkpoint_contract(self.checkpoint, spec)
                         if spec.encoder == "t0" else {"checkpoint": str(self.checkpoint),
                                                        "config_sha256": ""})
        self.tokenizer = tokenizer or NativeTokenizer(self.checkpoint)
        self._extract_tokens = extract_tokens
        self._cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "applied": 0, "skipped_no_tools": 0,
                      "chunk_extracts": 0, "chunk_cache_hits": 0}

    # ---- extraction with a content-addressed memo ----
    def _extract(self, chunk: Chunk, *, force: bool = False) -> Dict[str, Any]:
        key = (chunk.event_id, chunk.part_index, self.spec.ratio)
        with self._lock:
            hit = not force and key in self._cache
            if hit:
                self.stats["chunk_cache_hits"] += 1
                return self._cache[key]
        record = self._extract_tokens(list(chunk.token_ids), self.spec.ratio, "tool")
        if int(record.get("original_seq_len") or 0) != len(chunk.token_ids):
            raise ToolMemoryError(
                "extract_length", f"server saw {record.get('original_seq_len')} tokens for a "
                f"{len(chunk.token_ids)}-token chunk {chunk.event_id}#{chunk.part_index}")
        expected = expected_gist_len(len(chunk.token_ids), self.spec.ratio)
        if int(record.get("gist_len") or 0) != expected:
            raise ToolMemoryError(
                "extract_gist_len", f"server returned gist_len {record.get('gist_len')} for a "
                f"{len(chunk.token_ids)}-token chunk at ratio {self.spec.ratio}; expected {expected}")
        with self._lock:
            self._cache[key] = record
            self.stats["chunk_extracts"] += 1
        return record

    def refresh(self, plan: ToolMemoryPlan) -> ToolMemoryPlan:
        """Cache-miss recovery: re-extract every chunk, bypassing the memo."""
        plan.records = [self._extract(chunk, force=True) for chunk in plan.chunks]
        return plan

    # ---- the per-request transform ----
    def plan(self, payload: Mapping[str, Any], *,
             native_override: Optional[Sequence[int]] = None,
             retrieval_only: bool = False,
             target_resident_tokens: Optional[int] = None) -> Optional[ToolMemoryPlan | VisibleToolPlan]:
        """Plan only definitions made visible by the current benchmark request."""
        self.stats["requests"] += 1
        visible = plan_visible_tool_memory(payload, self.spec, self.tokenizer,
                                           native_override=native_override,
                                           retrieval_only=retrieval_only)
        if visible is None:
            self.stats["skipped_no_tools"] += 1
            return None
        return self.materialize_visible_plan(
            payload, visible, retrieval_only=retrieval_only,
            target_resident_tokens=target_resident_tokens)

    def materialize_visible_plan(
        self,
        payload: Mapping[str, Any],
        visible: VisibleToolPlan,
        *,
        retrieval_only: bool = False,
        target_resident_tokens: Optional[int] = None,
    ) -> ToolMemoryPlan | VisibleToolPlan:
        """Attach server receipts to one already-selected visible plan.

        This seam lets a composed runtime freeze selection once per decision,
        then materialize the exact same plan without running the selector a
        second time during draft or regeneration.
        """
        if not isinstance(visible, VisibleToolPlan):
            raise TypeError("visible must be a VisibleToolPlan")
        if visible.spec != self.spec:
            raise ValueError("visible tool plan spec differs from this ToolMemory manager")
        if self.spec.encoder != "t0":
            if retrieval_only:
                raise ValueError("retrieval-only is a native-schema control, not a raw KV method")
            self._prepare_raw_plan(payload, visible, target_resident_tokens)
            self.stats["applied"] += 1
            return visible
        enforce_tool_budget(visible, self.budget_tokens)
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools") or []
        records = [self._extract(chunk) for chunk in visible.chunks]
        # Accounting for the paper's R_tool: the raw Qwen tool prologue vs the
        # protocol system prefix plus gist tokens.  Both measured on the same
        # tokenizer; the server's kv_resident_tokens remains the physical truth.
        system_only = [m for m in messages if m.get("role") == "system"][:1] or [
            {"role": "system", "content": ""}]
        raw_prologue = (len(self.tokenizer.native_ids(system_only, tools=[tool_snapshot(t) for t in tools]))
                        - len(self.tokenizer.native_ids(system_only))) if tools else 0
        raw_source = sum(_isolated_content_tokens(self.tokenizer, span.text)
                         for span in visible.source_spans)
        protocol_tokens = int(visible.info["protocol_prefix_tokens"])
        gist_tokens = sum(int(r["gist_len"]) for r in records)
        presented = sum(len(c.token_ids) for c in visible.chunks)
        snapshots = [tool_snapshot(tool) for tool in tools] + [
            visible_tool_snapshot(span) for span in visible.source_spans]
        info = {**visible.info,
            "native_names": [tool_name(snapshots[i]) for i in visible.info["native_indices"]],
            "presented_encoder_tokens": presented, "gist_tokens": gist_tokens,
            "raw_tool_prologue_tokens": raw_prologue,
            "raw_visible_source_tokens": raw_source,
            "protocol_prefix_tokens": protocol_tokens,
            "resident_tool_tokens": visible.info["resident_tool_tokens"],
            "budget_tokens": self.budget_tokens,
            "cache_hits": sum(1 for r in records if r.get("cache_hit")),
            "checkpoint": self.contract["checkpoint"],
            "checkpoint_config_sha256": self.contract["config_sha256"],
            "protected_interface_spans": list(visible.interface_spans),
        }
        self.stats["applied"] += 1
        return ToolMemoryPlan(self.spec, visible.messages, visible.protocol,
                              visible.chunks, records, info, visible.carrier_anchors,
                              visible.source_spans, visible.interface_spans)

    def _prepare_raw_plan(self, payload: Mapping[str, Any], plan: VisibleToolPlan,
                          target_resident_tokens: Optional[int]) -> None:
        prepare_raw_tool_plan(payload, plan, self.tokenizer,
                              budget_tokens=self.budget_tokens,
                              target_resident_tokens=target_resident_tokens)

    def prepare_full_history_request(self, payload: Mapping[str, Any], *,
                                     native_override: Optional[Sequence[int]] = None,
                                     retrieval_only: bool = False,
                                     target_resident_tokens: Optional[int] = None) -> tuple:
        """Shared request adapter for recorded decisions with uncompressed history."""
        plan = self.plan(payload, native_override=native_override,
                         retrieval_only=retrieval_only,
                         target_resident_tokens=target_resident_tokens)
        out = dict(payload)
        if plan is not None:
            out["messages"] = plan.messages
            if isinstance(plan, ToolMemoryPlan):
                anchors = {anchor["message_index"]: anchor["rewritten_message_index"]
                           for anchor in (plan.carrier_anchors or [])}
                out["messages"], _ = insert_carriers(
                    plan.messages, {}, plan.carriers(), source_out_indices=anchors or None)
                out["messages"] = [strip_carrier_fields(message) if is_carrier(message)
                                   else message for message in out["messages"]]
        out = self.stage_request(out, plan)
        out["chat_template_kwargs"] = {**dict(out.get("chat_template_kwargs") or {}),
                                       "enable_thinking": False}
        out["c2kv_use_gist_projection"] = False
        return out, plan

    def stage_request(self, payload: Mapping[str, Any],
                       plan: Optional[ToolMemoryPlan | VisibleToolPlan]) -> Dict[str, Any]:
        """Request-level fields for the upstream chat call (tools stay for the
        tool-call parser; the template must not render them)."""
        out = dict(strip_request_annotations(payload))
        if plan is not None and plan.info.get("structured_tools_in_prompt") is False:
            out["c2kv_tools_in_prompt"] = False
        if isinstance(plan, VisibleToolPlan):
            out["c2kv_tools_in_prompt"] = False
            # Offsets are validated again on the final assembled messages. Raw
            # schemas cannot silently migrate into a compressed history block.
            schema_spans = (plan.assembled_schema_spans if plan.assembled_schema_spans is not None
                            else plan.raw_schema_spans)
            for span in schema_spans:
                messages = out.get("messages") or []
                index = span["message_index"]
                content = messages[index].get("content") if index < len(messages) else None
                if not isinstance(content, str) or content[span["start"]:span["end"]] != span["text"]:
                    raise ToolMemoryError("raw_schema_assembly", "history assembly moved a raw tool schema")
            for span in plan.interface_spans:
                messages = out.get("messages") or []
                index = span["message_index"]
                content = messages[index].get("content") if index < len(messages) else None
                if not isinstance(content, str) or content[span["start"]:span["end"]] != span["text"]:
                    raise ToolMemoryError("interface_assembly", "history assembly moved a protected interface")
            if self.spec.interface_policy == "schema" and not schema_spans:
                # The history policy removed every old source definition. Its
                # protected interface is still raw in the system prefix.
                interface_tokens = int(plan.info.get("interface_copy_tokens") or 0)
                if self.budget_tokens is not None and interface_tokens > self.budget_tokens:
                    raise ToolMemoryError(
                        "tool_budget", "protected interfaces exceed the separate tool budget")
                plan.info["raw_tool_eviction"] = "skipped_no_retained_schema"
                out["c2kv_use_gist_projection"] = False
                return out
            hint = dict(out.get("c2kv_kv_memory_hint") or {})
            protected_schema_indices = (set(plan.info["native_indices"])
                                        | set(plan.info.get("raw_noop_schema_indices") or ()))
            spanned_schema_indices = {span["schema_index"] for span in schema_spans}
            hint["tool_kv_eviction"] = {
                "method": self.spec.encoder,
                "schema_spans": list(schema_spans),
                "protected_schema_indices": sorted(protected_schema_indices
                                                   & spanned_schema_indices),
                **({"protected_interface_spans": list(plan.interface_spans)}
                   if self.spec.interface_policy == "schema" else {}),
                "target_resident_tokens_per_layer": plan.info["target_resident_tokens_per_layer"],
                "recent_window": 64 if self.spec.encoder == "pyramidkv" else 16,
                "kernel_size": 5 if self.spec.encoder == "pyramidkv" else 7,
                "pooling": "maxpool" if self.spec.encoder == "snapkv" else "avgpool",
                "h2o_recent_fraction": 0.5,
            }
            if plan.info.get("joint_history_assembly"):
                hint["tool_kv_eviction"].update(
                    joint_history_assembly=True,
                    joint_tool_target_tokens_per_layer=plan.info["joint_tool_target_tokens_per_layer"])
            if self.budget_tokens is not None:
                hint["tool_kv_eviction"]["max_resident_tool_tokens"] = self.budget_tokens
            if "tool_protocol_span" in plan.info:
                hint["tool_kv_eviction"]["tool_protocol_span"] = plan.info["tool_protocol_span"]
            out["c2kv_kv_memory_hint"] = hint
            out["c2kv_use_gist_projection"] = False
        return out
