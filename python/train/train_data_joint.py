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
- ordinary prompt = current turn only;
- target = next assistant action;
- system prefix = bare system prompt WITHOUT ``tools=`` (the de-leak).

Answer-format choice
--------------------
The answer is rendered with the history path's ``_render_agent_output_messages``
(one existing code path that handles reasoning/content/tool_calls/parts).  When
the span ends in tool calls this emits exactly the unified path's target
surface ``Action:\n<tool_call>\n{"name":...,"arguments":...}\n</tool_call>``
(same payload keys, same minified JSON), so next-action supervision is
consistent with ``train_unified_next_action_c2kv.py``; spans without tool calls
fall back to the assistant text (history-path behavior).  ``require_tool_call``
restricts to tool-call targets when set.

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from .train_data import DEFAULT_SYSTEM_PROMPT
from .train_data_multiturn import (
    AgentLLMTracesCompressHistorySource,
    HistorySelection,
    Message,
    _agent_history_turn_docs,
    _agent_system_prompt,
    _chat_template_ids,
    _find_agent_jsonl_files,
    _find_agent_parquet_files,
    _fit_reused_history,
    _iter_agent_jsonl_rows,
    _iter_agent_rows,
    _json_loads,
    _normal_agent_message,
    _normal_chat_message,
    _pad,
    _render_agent_output_messages,
    _sort_agent_spans,
    _span_attributes,
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
    required = set(str(item) for item in schema.get("required", []) if isinstance(schema.get("required"), list))
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


def _render_tool_documents(
    tools: Sequence[Dict[str, Any]],
    rng: random.Random,
    *,
    canonical_format_prob: float = 0.7,
    minified_json_prob: float = 0.2,
    shuffle_tools: bool = True,
    truncate_description_chars: int = 600,
) -> List[str]:
    tools = list(tools)
    if shuffle_tools:
        rng.shuffle(tools)
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
    return docs


class AgentLLMTracesJointSource(AgentLLMTracesCompressHistorySource):
    """True-joint source for agent-llm-traces.

    Subclasses the history-path source so sessions/spans/tool definitions are
    read identically: same parquet/jsonl discovery, same span sorting by
    (start_time, span_id), same history/current split at the last user message,
    same ``max_samples_per_session`` sub-sampling, same split-manifest args.
    The only parsing addition is that tool definitions (first span of the
    session carrying ``gen_ai.tool.definitions``) are rendered into per-tool
    documents with the unified path's variant policy, once per session.
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
    ) -> None:
        # Set joint knobs BEFORE super().__init__(): the parent constructor
        # calls self._load_records(), which dispatches to the overridden
        # _session_examples below.
        self.canonical_format_prob = canonical_format_prob
        self.minified_json_prob = minified_json_prob
        self.shuffle_tools = shuffle_tools
        self.truncate_description_chars = truncate_description_chars
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
                examples = rng.sample(examples, self.max_samples_per_session)
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
                examples = rng.sample(examples, self.max_samples_per_session)
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
        tool_documents: List[str] = []
        for span_index, span in enumerate(spans):
            attributes = _span_attributes(span)
            if not tools:
                tools = _tool_list_from_agent_value(attributes.get("gen_ai.tool.definitions"))
                if tools:
                    # One rendering per session: every example of the session
                    # shares the same tool documents (deterministic seed).
                    rng = random.Random(f"{self.split_seed}:{session_id}:tool_documents")
                    tool_documents = _render_tool_documents(
                        tools,
                        rng,
                        canonical_format_prob=self.canonical_format_prob,
                        minified_json_prob=self.minified_json_prob,
                        shuffle_tools=self.shuffle_tools,
                        truncate_description_chars=self.truncate_description_chars,
                    )
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
            if self.prefix_history_doc_num is not None:
                if len(history_docs) < self.prefix_history_doc_num:
                    continue
                if self.prefix_history_exact and len(history_docs) != self.prefix_history_doc_num:
                    continue
                history_docs = history_docs[-self.prefix_history_doc_num :]
            current_messages = messages[last_user_index:]
            answer, has_tool_call = _render_agent_output_messages(output_messages, self.max_answer_chars)
            if self.require_tool_call and not has_tool_call:
                continue
            if not history_docs or not current_messages or not answer:
                continue
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
                )
            )
        return examples


def _default_max_tool_chunks(max_doc_num: int) -> int:
    return max(1, (2 * max_doc_num) // 3)


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
        max_tool_definition_tokens: int = 10000,
        split_oversized_history_docs: bool = True,
    ) -> None:
        if doc_mode not in ("joint", "tool_only", "history_only"):
            raise ValueError(f"Unsupported doc_mode: {doc_mode!r}")
        self.max_doc_length = max_doc_length
        self.min_doc_num = min_doc_num
        self.max_doc_num = max_doc_num
        self.max_system_length = max_system_length
        self.max_length = max_length
        self.doc_mode = doc_mode
        self.data: List[Dict[str, Any]] = []
        skipped_by_reason: Counter[str] = Counter()
        for example in examples:
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
            )
            if row is None:
                skipped_by_reason[reason] += 1
            else:
                self.data.append(row)
        logger.info(
            "Built %d joint samples (%s); skipped %d by reason=%s",
            len(self.data),
            doc_mode,
            sum(skipped_by_reason.values()),
            dict(skipped_by_reason),
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
        max_tool_definition_tokens: int = 10000,
        split_oversized_history_docs: bool = True,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if doc_mode not in ("joint", "tool_only", "history_only"):
            raise ValueError(f"Unsupported doc_mode: {doc_mode!r}")
        if max_tool_chunks is None:
            max_tool_chunks = _default_max_tool_chunks(max_doc_num)

        # ---- tool chunks (first in the grid) ------------------------------
        tool_chunks: List[List[int]] = []
        if doc_mode != "history_only":
            tool_cap = max_doc_num if doc_mode == "tool_only" else min(max_tool_chunks, max_doc_num)
            doc_id_groups = [
                _chat_template_ids(
                    tokenizer,
                    [{"role": "user", "content": TOOL_DOC_PREFIX + document}],
                )
                for document in example.tool_documents
                if document.strip()
            ]
            doc_tokens = sum(len(doc_ids) for doc_ids in doc_id_groups)
            if doc_tokens > max_tool_definition_tokens:
                return None, f"tool_definition_tokens>{max_tool_definition_tokens}"
            for doc_ids in doc_id_groups:
                tool_chunks.extend(
                    doc_ids[start : start + max_doc_length]
                    for start in range(0, len(doc_ids), max_doc_length)
                )
            tool_chunks = tool_chunks[:tool_cap]

        # ---- history chunks (remaining slots, chronological) --------------
        history: List[Message] = []
        if doc_mode != "tool_only":
            history_budget = max_doc_num if doc_mode == "history_only" else max_doc_num - len(tool_chunks)
            raw_history = [
                {"role": "user", "content": text}
                for text in example.history_documents
                if text and text.strip()
            ]
            if history_budget > 0 and raw_history:
                history = _fit_reused_history(
                    tokenizer,
                    raw_history,
                    max_doc_length=max_doc_length,
                    max_doc_num=history_budget,
                    policy=history_selection,
                    split_oversized_history_docs=split_oversized_history_docs,
                )

        current = [
            _normal_chat_message(message)
            for message in example.current_messages
            if message.get("content") or message.get("role") == "assistant"
        ]
        doc_count = len(tool_chunks) + len(history)
        if doc_count < min_doc_num:
            return None, f"doc_num<{min_doc_num}"
        if not current:
            return None, "empty_current"
        if not example.answer:
            return None, "empty_answer"

        # ---- bare system prefix: NO tools= (the de-leak) ------------------
        system_ids = _chat_template_ids(
            tokenizer,
            [{"role": "system", "content": example.system_prompt}],
            keep_bos=True,
            max_length=max_system_length,
        )
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
        prompt_ids = _chat_template_ids(
            tokenizer,
            current,
            add_generation_prompt=True,
        )
        answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
        if not answer_ids:
            return None, "empty_answer_ids"
        answer_ids.append(tokenizer.eos_token_id)
        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[-(max_length - 1):]
        answer_budget = max_length - len(prompt_ids)
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


def assert_no_leakage(example: JointExample, features: Dict[str, Any], tokenizer) -> None:
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
        if probe in system_text:
            raise AssertionError(f"tool document leaked into system_input_ids: {probe!r}")
        if probe in prompt_answer_text:
            raise AssertionError(f"tool document leaked into input_ids: {probe!r}")
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
                content = content + "\n# Tools\n" + json.dumps(list(tools), ensure_ascii=False)
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
        )
    ]


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
