#!/usr/bin/env python
"""S0 census for the big-tool-pool baseline (PR-E), pure CPU, read-only.

Counts how many usable spans survive the REAL harness filter chain on the
frozen toolset_disjoint EVAL split under the new packing rule
(MAX_TOOL_DEFINITION_TOKENS=97000, MAX_DOC_NUM=96, MAX_DOC_LENGTH=1024),
per subset (appworld, tau2_telecom, ...) and in total. This N feeds the
entry formula MDE(N) = z_0.975 * sqrt(psi / N) which decides
whether the appworld-scale big-pool experiment has enough N to enter.

Filter chain replicated (in the order the harness applies it):

  1. Split: eval_session_ids of --split (toolset_disjoint) from the frozen
     manifest, intersected with sessions present in the dataset.
     Source: train_agent_tool_definition_c2kv.py:634-687
     (AgentLLMTracesSource.split_session_ids / _split_session_ids_from_manifest),
     driven by eval_agent_tool_definition_c2kv.py:806-823.
  2. Span-level source filters inside _session_examples, in order:
     missing_tool_input_or_output (needs session tool_definition +
     gen_ai.input.messages + gen_ai.output.messages) -> empty_answer ->
     no_tool_call (require_tool_call) -> empty_prompt.
     Source: train_agent_tool_definition_c2kv.py:748-806.
     (max_tool_definition_chars / max_input_chars / max_target_chars filters
     at lines 761-769 are inactive: harness never sets them, defaults None.)
  3. min_target_tokens: **NO-OP in the eval selection chain.** The eval CLI
     accepts --min_target_tokens (eval_agent_tool_definition_c2kv.py:947,
     passed into AgentToolDefinitionDataArgs at line 820) but
     AgentLLMTracesSource never reads it; it is only used by the TRAINING
     preprocessing (train_agent_tool_definition_c2kv.py:984-993) as a
     packing budget. This census reports the stage as an identity step and
     additionally reports a clearly-labeled HYPOTHETICAL count (spans whose
     answer would tokenize below min_target_tokens) for information only.
  4. Per-session cap: max_samples_per_session=16 via rng.sample in
     iter_examples (train_agent_tool_definition_c2kv.py:702-703). The eval
     then applies the tool-chunk threshold per example
     (selection_filter="c2kv", eval_agent_tool_definition_c2kv.py:831-847).
     Because every example of a session shares the SAME session-level
     tool_definition (train_agent_tool_definition_c2kv.py:749-755), the
     threshold outcome is identical for all spans of a session, so cap and
     threshold commute and final N per session =
     chunk_ok(session) ? min(n_valid, cap) : 0.
  5. Tool-definition token threshold + doc packing, document_mode="full"
     (TOOL_DOCUMENT_EVAL_MODE=full in agent/eval_agent_tool_definition_bigpool_npu.sh):
     chat-template ids of {"role": "user", "content": "Tool definitions:\n" + tool_definition};
     skip if tokens > max_tool_definition_tokens; else if tokens >
     max_doc_length*max_doc_num and not truncate -> skip; else if
     ceil(tokens/max_doc_length) > max_doc_num -> skip.
     Source: eval_agent_tool_definition_c2kv.py:172-187 (_build_tool_chunks,
     full branch), mirrored by inspect_agent_llm_traces_bigpool_stats.py:58-77.

Round-1 reference numbers (for sanity-checking this census; different
thresholds, so eyeball only): 32k run with MAX_DOC_NUM=64,
MAX_SAMPLES_PER_SESSION=16, MIN_TARGET_TOKENS=128, toolset_disjoint eval:
  - appworld: 8 sessions / 29 spans at the 10k filter
  - tau2_telecom: 39 sessions / 156 spans
  - total usable at 32k: 576 spans
Manifest: configs/agent_tooldef_split_manifests.json toolset_disjoint has
268 eval session ids (950 train).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harness imports (preferred path: reuse the exact classes the eval uses).
# Mirrors the import block of inspect_agent_llm_traces_bigpool_stats.py:12-33.
# train_agent_tool_definition_c2kv imports torch/transformers at module level;
# on hosts without them we fall back to the verbatim stdlib copies below.
# ---------------------------------------------------------------------------
_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - exercised on the target server
    from train_agent_tool_definition_c2kv import (  # noqa: E402
        AgentLLMTracesSource,
        AgentToolDefinitionDataArgs,
        _canonical_tool_definition,
        _span_attributes,
    )
    from train.train_data_multiturn import _chat_template_ids as _hf_chat_template_ids  # noqa: E402

    _HAVE_HARNESS = True
except BaseException as exc:  # noqa: BLE001 - any import failure -> fallback
    _HAVE_HARNESS = False
    _IMPORT_ERROR = exc


# ===========================================================================
# Fallback: verbatim stdlib copies of the harness filter chain.
# Used only when `import train_agent_tool_definition_c2kv` fails (no torch).
# Each function cites its source in agent/train_agent_tool_definition_c2kv.py.
# ===========================================================================
if not _HAVE_HARNESS:

    def _json_loads(value: Any, default: Any) -> Any:
        # Copied from train_agent_tool_definition_c2kv.py:80-88.
        if value is None or value == "":
            return default
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _json_dumps(value: Any) -> str:
        # Copied from train_agent_tool_definition_c2kv.py:91-92.
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        # Copied from train_agent_tool_definition_c2kv.py:95-105.
        if not isinstance(obj, dict):
            return default
        if key in obj:
            return obj[key]
        current = obj
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def _first_value(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
        # Copied from train_agent_tool_definition_c2kv.py:108-113.
        for key in keys:
            value = _get_value(obj, key, None)
            if value is not None:
                return value
        return default

    def _canonical_tool_definition(value: Any) -> str:
        # Copied from train_agent_tool_definition_c2kv.py:116-120.
        parsed = _json_loads(value, value)
        if isinstance(parsed, str):
            return parsed.strip()
        return _json_dumps(parsed)

    def _as_tool_list(tool_definition: Any) -> List[Dict[str, Any]]:
        # Copied from train_agent_tool_definition_c2kv.py:123-134.
        parsed = _json_loads(tool_definition, [])
        if isinstance(parsed, dict):
            if isinstance(parsed.get("tools"), list):
                parsed = parsed["tools"]
            elif isinstance(parsed.get("functions"), list):
                parsed = parsed["functions"]
            else:
                parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _render_tool_calls(tool_calls: Any) -> Tuple[str, bool]:
        # Copied from train_agent_tool_definition_c2kv.py:233-262.
        tool_calls = _json_loads(tool_calls, [])
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, list):
            return "", False
        rendered = []
        for call in tool_calls:
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
            arguments = (
                function.get("arguments")
                or call.get("arguments")
                or call.get("args")
                or call.get("input")
                or {}
            )
            payload = {"name": name, "arguments": arguments}
            rendered.append("<tool_call>\n" + _json_dumps(payload) + "\n</tool_call>")
        return "\n".join(rendered), bool(rendered)

    def _message_parts(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Copied from train_agent_tool_definition_c2kv.py:265-272.
        parts = message.get("parts")
        parts = _json_loads(parts, parts)
        if isinstance(parts, dict):
            return [parts]
        if isinstance(parts, list):
            return [part for part in parts if isinstance(part, dict)]
        return []

    def _message_content_to_text(message: Dict[str, Any]) -> str:
        # Copied from train_agent_tool_definition_c2kv.py:275-299.
        content = message.get("content", "")
        if not content and _message_parts(message):
            content = _message_parts(message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in ("tool_call", "function_call"):
                        continue
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
                    else:
                        parts.append(_json_dumps(item))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return _json_dumps(content)

    def _normal_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Copied from train_agent_tool_definition_c2kv.py:302-322.
        if not isinstance(message, dict):
            return None
        role = message.get("role") or message.get("type") or "user"
        if role == "tool":
            role = "user"
        content_parts = []
        content = _message_content_to_text(message)
        if content:
            content_parts.append(content)
        tool_calls_text, _ = _render_tool_calls(
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _message_parts(message)
        )
        if tool_calls_text:
            content_parts.append("Action:\n" + tool_calls_text)
        if not content_parts and role != "assistant":
            return None
        return {"role": role, "content": "\n\n".join(content_parts)}

    def _render_output_messages(value: Any) -> Tuple[str, bool]:
        # Copied from train_agent_tool_definition_c2kv.py:325-372.
        messages = _json_loads(value, [])
        if isinstance(messages, dict):
            messages = [messages]
        if not isinstance(messages, list):
            messages = [{"role": "assistant", "content": str(messages)}]

        rendered_messages: List[str] = []
        has_tool_call = False
        for message in messages:
            if not isinstance(message, dict):
                if message:
                    rendered_messages.append(str(message))
                continue
            parts = []
            reasoning = (
                message.get("reasoning_content")
                or message.get("reasoning")
                or message.get("thought")
                or message.get("thinking")
                or message.get("cot")
                or ""
            )
            if reasoning:
                parts.append("Thought:\n" + str(reasoning).strip())
            content = _message_content_to_text(message).strip()
            if content:
                parts.append(content)
            tool_calls_text, found_tool_call = _render_tool_calls(
                message.get("tool_calls")
                or message.get("toolCalls")
                or message.get("function_call")
                or _message_parts(message)
            )
            if tool_calls_text:
                parts.append("Action:\n" + tool_calls_text)
            has_tool_call = has_tool_call or found_tool_call
            rendered = "\n\n".join(part for part in parts if part).strip()
            if rendered:
                rendered_messages.append(rendered)

        answer = "\n\n".join(rendered_messages).strip()
        marker_text = answer.lower()
        has_tool_call = has_tool_call or any(
            marker in marker_text
            for marker in ("<tool_call>", "action:", "function_call", "tool call")
        )
        return answer, has_tool_call

    def _iter_sessions(row: Dict[str, Any], row_index: int):
        # Copied from train_agent_tool_definition_c2kv.py:375-406.
        trace = _json_loads(_first_value(row, ["trace", "Trace"], row), row)
        sessions = _first_value(row, ["trace.sessions", "sessions"], None)
        if sessions is None and isinstance(trace, dict):
            sessions = _first_value(trace, ["trace.sessions", "sessions"], None)
        sessions = _json_loads(sessions, sessions)

        if sessions is None:
            spans = _first_value(row, ["spans", "trace.spans"], None)
            spans = _json_loads(spans, spans)
            if spans is not None:
                sessions = [{"session_id": _first_value(row, ["session_id", "id"], f"row-{row_index}"), "spans": spans}]

        if isinstance(sessions, dict):
            sessions = [sessions]
        if not isinstance(sessions, list):
            return

        for session_index, session in enumerate(sessions):
            if not isinstance(session, dict):
                continue
            spans = _json_loads(session.get("spans"), session.get("spans"))
            if not isinstance(spans, list):
                continue
            session_id = (
                session.get("session_id")
                or session.get("sessionId")
                or session.get("id")
                or _first_value(row, ["session_id", "trace_id", "id"], None)
                or f"row-{row_index}:session-{session_index}"
            )
            yield str(session_id), spans

    def _span_attributes(span: Any) -> Dict[str, Any]:
        # Copied from train_agent_tool_definition_c2kv.py:409-415.
        span = _json_loads(span, span)
        if not isinstance(span, dict):
            return {}
        attributes = span.get("attributes", span)
        attributes = _json_loads(attributes, attributes)
        return attributes if isinstance(attributes, dict) else {}

    def _sort_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Copied from train_agent_tool_definition_c2kv.py:625-632.
        return sorted(
            spans,
            key=lambda span: (
                span.get("start_time") or "",
                span.get("span_id") or "",
            ),
        )

    def _toolathlon_row_to_session(row: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
        # Copied from train_agent_tool_definition_c2kv.py:531-576.
        tools_payload = _json_loads(row.get("tool_calls"), {})
        tools = _as_tool_list(tools_payload)
        messages = _json_loads(row.get("messages"), [])
        if not tools or not isinstance(messages, list):
            return None
        config = _json_loads(row.get("config"), {})
        system_prompt = ""
        if isinstance(config, dict):
            system_prompts = config.get("system_prompts") if isinstance(config.get("system_prompts"), dict) else {}
            system_prompt = str(system_prompts.get("agent") or "").strip()
        system_message = {"role": "system", "content": system_prompt} if system_prompt else None
        spans = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            has_tool_call = bool(
                message.get("tool_calls")
                or message.get("toolCalls")
                or message.get("function_call")
                or _message_parts(message)
            )
            if not has_tool_call:
                continue
            input_messages = [item for item in messages[:message_index] if isinstance(item, dict)]
            if system_message and not any(item.get("role") == "system" for item in input_messages):
                input_messages = [system_message, *input_messages]
            if not any(item.get("role") == "user" for item in input_messages):
                continue
            spans.append({
                "start_time": f"{message_index:06d}",
                "span_id": f"toolathlon-{message_index}",
                "attributes": {
                    "gen_ai.tool.definitions": _json_dumps(tools),
                    "gen_ai.input.messages": _json_dumps(input_messages),
                    "gen_ai.output.messages": _json_dumps([message]),
                },
            })
        if not spans:
            return None
        session_id = str(row.get("request_id") or row.get("task_name") or f"toolathlon-row-{row_index}")
        return {
            "session_id": session_id,
            "subset": str(row.get("task_name") or row.get("modelname_run") or "toolathlon"),
            "spans": spans,
        }

    class FallbackSource:
        """Stdlib replica of AgentLLMTracesSource (loading + _session_examples).

        Mirrors train_agent_tool_definition_c2kv.py:418-806 for the args the
        harness actually exercises (tool_document_mode="full",
        max_*_chars=None). Only the census-relevant surface is provided:
        .sessions, .source_skips, .split_session_ids(), ._session_examples().
        """

        def __init__(self, args: argparse.Namespace) -> None:
            self.args = args
            self.path = Path(args.dataset_path)
            if not self.path.exists() and args.dataset_path.endswith("/datasets/agent-llm-traces"):
                # Fallback mirror of _resolve_dataset_path (lines 445-452).
                fallback = Path("./data/agent-llm-traces")
                if fallback.exists():
                    logger.warning("Using fallback dataset path %s", fallback)
                    self.path = fallback
            self.source_skips: Counter[str] = Counter()
            parquet_files = self._find_files("*.parquet")
            if parquet_files:
                logger.info("Loading %d parquet shards from %s", len(parquet_files), self.path)
                self.sessions = self._load_sessions(self._iter_parquet_rows(parquet_files))
            else:
                jsonl_files = self._find_files("*.jsonl")
                if not jsonl_files:
                    raise FileNotFoundError(f"No parquet/jsonl files found under dataset_path={self.path!s}")
                logger.info("Loading %d jsonl shards from %s", len(jsonl_files), self.path)
                sessions = []
                for row_index, row in enumerate(self._iter_jsonl_rows(jsonl_files)):
                    session = _toolathlon_row_to_session(row, row_index)
                    if session is None:
                        self.source_skips["unusable_jsonl_row"] += 1
                        continue
                    sessions.append(session)
                self.sessions = sessions
            if args.max_sessions is not None:
                self.sessions = self.sessions[: args.max_sessions]
            logger.info("Loaded %d sessions before train/eval split", len(self.sessions))

        def _find_files(self, pattern: str) -> List[str]:
            # Mirrors _find_parquet_files/_find_jsonl_files (lines 455-482).
            if self.path.is_file() and self.path.suffix == pattern.lstrip("*"):
                return [str(self.path)]
            files: List[Path] = []
            for root in (self.path / "data", self.path):
                if root.is_dir():
                    files.extend(sorted(root.glob(pattern)))
                    if not files:
                        files.extend(sorted(root.rglob(pattern)))
                if files:
                    break
            return [str(file) for file in files]

        @staticmethod
        def _iter_parquet_rows(data_files: Sequence[str]):
            # Mirrors _iter_parquet_rows (lines 504-517).
            import pyarrow.parquet as pq

            for data_file in data_files:
                table = pq.read_table(data_file)
                for row in table.to_pylist():
                    yield row

        @staticmethod
        def _iter_jsonl_rows(data_files: Sequence[str]):
            # Mirrors _iter_jsonl_rows (lines 519-528).
            for data_file in data_files:
                with Path(data_file).open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = _json_loads(line, None)
                        if isinstance(row, dict):
                            yield row

        def _load_sessions(self, raw) -> List[Dict[str, Any]]:
            # Mirrors _load_sessions (lines 578-622).
            sessions_by_id: Dict[str, List[Dict[str, Any]]] = {}
            subset_by_id: Dict[str, str] = {}
            for row_index, row in enumerate(raw):
                row_subset = str(
                    row.get("benchmark")
                    or row.get("subset")
                    or row.get("dataset")
                    or row.get("task")
                    or "unknown"
                )
                found_nested_session = False
                for session_id, spans in _iter_sessions(row, row_index):
                    found_nested_session = True
                    sessions_by_id.setdefault(session_id, []).extend(spans)
                    subset_by_id.setdefault(session_id, row_subset)
                if found_nested_session:
                    continue
                session_id = (
                    row.get("session_id")
                    or row.get("trace_id")
                    or row.get("TraceId")
                    or row.get("traceId")
                    or f"row-{row_index}"
                )
                sessions_by_id.setdefault(str(session_id), []).append(dict(row))
                subset_by_id.setdefault(str(session_id), row_subset)
            return [
                {
                    "session_id": session_id,
                    "subset": subset_by_id.get(session_id, "unknown"),
                    "spans": _sort_spans(spans),
                }
                for session_id, spans in sessions_by_id.items()
            ]

        def split_session_ids(self) -> Tuple[set, set]:
            # Mirrors split_session_ids + _split_session_ids_from_manifest
            # (lines 634-687). Census always passes a manifest; the manifest
            # branch is reproduced exactly, including the intersection with
            # available session ids and the train/eval overlap check.
            manifest_path = Path(self.args.split_manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "train_session_ids" in manifest and "eval_session_ids" in manifest:
                selected = manifest
                split_name = "root"
            else:
                split_name = self.args.split
                if split_name not in manifest:
                    raise KeyError(
                        f"Split {split_name!r} not found in {manifest_path}. "
                        f"Available splits: {sorted(manifest)}"
                    )
                selected = manifest[split_name]
            train_ids = set(str(item) for item in selected.get("train_session_ids", []))
            eval_ids = set(str(item) for item in selected.get("eval_session_ids", []))
            overlap = train_ids & eval_ids
            if overlap:
                raise RuntimeError(
                    f"Split manifest {manifest_path}::{split_name} has train/eval overlap: "
                    f"{sorted(overlap)[:5]}"
                )
            available_ids = {item["session_id"] for item in self.sessions}
            missing_train = train_ids - available_ids
            missing_eval = eval_ids - available_ids
            if missing_train or missing_eval:
                logger.warning(
                    "Split manifest %s::%s references missing sessions; missing_train=%d missing_eval=%d",
                    manifest_path,
                    split_name,
                    len(missing_train),
                    len(missing_eval),
                )
            train_ids &= available_ids
            eval_ids &= available_ids
            logger.info(
                "Using split manifest %s::%s with %d train sessions and %d eval sessions",
                manifest_path,
                split_name,
                len(train_ids),
                len(eval_ids),
            )
            return train_ids, eval_ids

        def _session_examples(
            self,
            session_id: str,
            spans: Sequence[Any],
            subset: str = "unknown",
        ) -> List[Dict[str, Any]]:
            # Mirrors _session_examples (lines 748-806) with
            # tool_document_mode="full" (so _build_tool_documents returns None,
            # lines 713-715) and max_tool_definition_chars / max_input_chars /
            # max_target_chars unset (harness never sets them, so the filters
            # at lines 761-769 are inactive). Returns dicts carrying the
            # fields the census reads (tool_definition, answer).
            tool_definition = ""
            candidates = []
            for span_index, span in enumerate(spans):
                attributes = _span_attributes(span)
                tool_value = attributes.get("gen_ai.tool.definitions")
                if tool_value and not tool_definition:
                    tool_definition = _canonical_tool_definition(tool_value)
                input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
                output_messages = attributes.get("gen_ai.output.messages")
                if not tool_definition or not input_messages or output_messages is None:
                    self.source_skips["missing_tool_input_or_output"] += 1
                    continue
                answer, has_tool_call = _render_output_messages(output_messages)
                if not answer:
                    self.source_skips["empty_answer"] += 1
                    continue
                if self.args.require_tool_call and not has_tool_call:
                    self.source_skips["no_tool_call"] += 1
                    continue
                normalized_messages = [
                    item
                    for item in (_normal_message(message) for message in _json_loads(input_messages, []))
                    if item is not None and item.get("role") != "system"
                ]
                if not normalized_messages:
                    self.source_skips["empty_prompt"] += 1
                    continue
                candidates.append(
                    {
                        "qid": f"{session_id}:{span_index}",
                        "session_id": session_id,
                        "tool_definition": tool_definition,
                        "answer": answer,
                        "has_tool_call": has_tool_call,
                        "subset": subset,
                    }
                )
            return candidates


# ---------------------------------------------------------------------------
# Tool-document tokenization, matching the harness entry points.
# ---------------------------------------------------------------------------
class ToolDocTokenCounter:
    """Token counts of the full-mode tool document, cached by content hash.

    hf backend: exact harness count = len(_chat_template_ids(tokenizer,
    [{"role": "user", "content": "Tool definitions:\n" + tool_definition}]))
    per eval_agent_tool_definition_c2kv.py:173-174 (and
    inspect_agent_llm_traces_bigpool_stats.py:118-123).
    estimate backend: ceil(chars / 4) of the same document text — CPU-only
    dry-run approximation, clearly NOT harness-exact.
    """

    def __init__(self, backend: str, tokenizer_path: str) -> None:
        self.backend = backend
        self.tokenizer = None
        self._chat_ids = None
        if backend == "hf":
            from transformers import AutoTokenizer  # lazy: needs transformers only here

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
                padding_side="right",
            )
            if _HAVE_HARNESS:
                self._chat_ids = _hf_chat_template_ids
        self._cache: Dict[str, int] = {}
        self._answer_cache: Dict[str, int] = {}

    def _apply_chat_template_ids(self, messages: List[Dict[str, Any]]) -> List[int]:
        if self._chat_ids is not None:
            return self._chat_ids(self.tokenizer, messages)
        # Replica of _chat_template_ids (python/train/train_data_multiturn.py:865-886)
        # for the call shape used here: tools=None, add_generation_prompt=False,
        # keep_bos=False, max_length=None.
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tools=None,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        if ids and ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]
        return ids

    def full_doc_tokens(self, tool_definition: str) -> int:
        key = hashlib.sha1(tool_definition.encode("utf-8")).hexdigest()
        if key not in self._cache:
            if self.backend == "estimate":
                # ESTIMATE ONLY: ceil(chars/4) of the document text.
                text = "Tool definitions:\n" + tool_definition
                self._cache[key] = int(math.ceil(len(text) / 4))
            else:
                tool_doc = {"role": "user", "content": "Tool definitions:\n" + tool_definition}
                self._cache[key] = len(self._apply_chat_template_ids([tool_doc]))
        return self._cache[key]

    def answer_tokens(self, answer: str) -> int:
        """For the HYPOTHETICAL min_target_tokens check only (not part of the
        real eval chain). hf: tokenizer.encode(answer, add_special_tokens=False)
        mirroring train_agent_tool_definition_c2kv.py:980; estimate: ceil(chars/4)."""
        key = hashlib.sha1(answer.encode("utf-8")).hexdigest()
        if key not in self._answer_cache:
            if self.backend == "estimate":
                self._answer_cache[key] = int(math.ceil(len(answer) / 4))
            else:
                self._answer_cache[key] = len(self.tokenizer.encode(answer, add_special_tokens=False))
        return self._answer_cache[key]


# ---------------------------------------------------------------------------
# Threshold / packing outcome, document_mode="full".
# Mirrors eval_agent_tool_definition_c2kv.py:172-187 (_build_tool_chunks full
# branch) via inspect_agent_llm_traces_bigpool_stats.py:58-77.
# ---------------------------------------------------------------------------
def _full_mode_chunk_outcome(
    doc_tokens: int,
    max_tool_definition_tokens: int,
    max_doc_length: int,
    max_doc_num: int,
    truncate_tool_definition: bool,
) -> Tuple[bool, str]:
    if doc_tokens > max_tool_definition_tokens:
        return False, f"tool_definition_tokens>{max_tool_definition_tokens}"
    max_context_tokens = max_doc_length * max_doc_num
    effective = doc_tokens
    if effective > max_context_tokens:
        if not truncate_tool_definition:
            return False, f"tool_definition_tokens>{max_context_tokens}"
        effective = max_context_tokens
    num_chunks = (effective + max_doc_length - 1) // max_doc_length
    if num_chunks > max_doc_num:
        return False, f"tool_definition_docs>{max_doc_num}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Entry formula.
# ---------------------------------------------------------------------------
def _mde_points(n: int, psi: float) -> Optional[float]:
    """MDE(N) = z_0.975 * sqrt(psi / N), expressed in points (x100)."""
    # Pre-registered calibration: psi-hat=0.5 <=> N>=400 boundary <=> 7 points
    # => 95% CI half-width form z_0.975 * sqrt(psi / N) (NOT the 80%-power form
    # (z_0.975 + z_0.80) * sqrt(psi / N), which gives 9.9 points at N=400).
    # Equivalently entry requires N >= psi * (1.96 / 0.07)^2 = 784 * psi.
    if n <= 0:
        return None
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.975)
    return round(100.0 * z * math.sqrt(psi / n), 3)


def _verdict(n: int, psi: float) -> Dict[str, Any]:
    mde = _mde_points(n, psi)
    if n < 24:
        label = "KILLED (N < 24)"
    elif mde is not None and mde <= 7.0:
        label = "ENTRY OK (MDE <= 7 points)"
    else:
        label = "GRAY ZONE (dataset inconclusive)"
    return {"N": n, "psi": psi, "MDE_points": mde, "verdict": label}


def _stat(values: Sequence[float]) -> Dict[str, Any]:
    # Same shape as inspect_agent_llm_traces_bigpool_stats.py:44-55.
    if not values:
        return {"min": 0, "avg": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": ordered[0],
        "avg": round(sum(ordered) / len(ordered), 4),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


# Round-1 bigpool reference numbers (see module docstring). Thresholds differ
# from this census (32k/64 docs vs 97k/96 docs), so these are sanity anchors
# for the funnel shape, not equality targets.
ROUND1_REFERENCE = {
    "config": "MAX_TOOL_DEFINITION_TOKENS=32000, MAX_DOC_NUM=64, MAX_DOC_LENGTH=1024, "
    "MAX_SAMPLES_PER_SESSION=16, MIN_TARGET_TOKENS=128, split=toolset_disjoint eval",
    "appworld_sessions": 8,
    "appworld_spans_at_10k_filter": 29,
    "tau2_telecom_sessions": 39,
    "tau2_telecom_spans": 156,
    "total_usable_spans_at_32k": 576,
    "manifest_toolset_disjoint_eval_session_ids": 268,
}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - tolerate any failure
        pass
    return None


def _new_bucket() -> Dict[str, Any]:
    return {
        "sessions_in_split": 0,
        "spans_raw": 0,
        "sessions_with_valid_spans": 0,
        "spans_after_require_tool_call": 0,  # = after ALL _session_examples filters
        "spans_after_min_target_tokens": 0,  # identity stage (no-op in eval chain)
        "hypothetical_min_target_drops": 0,  # NOT part of the eval chain; info only
        "sessions_after_tool_threshold": 0,
        "spans_after_tool_threshold": 0,
        "spans_after_session_cap": 0,  # final usable N
        "source_skip_reasons": Counter(),
        "threshold_skip_reasons": Counter(),
        "doc_tokens": [],
    }


def _ex_field(example: Any, name: str) -> Any:
    """Field access that works for both harness AgentToolDefinitionExample
    dataclasses (attribute) and fallback candidate dicts (subscript)."""
    value = getattr(example, name, None)
    if value is None and isinstance(example, dict):
        value = example.get(name)
    return value


def census(args: argparse.Namespace) -> Dict[str, Any]:
    if _HAVE_HARNESS:
        data_args = AgentToolDefinitionDataArgs(
            dataset_path=args.dataset_path,
            eval_ratio=args.eval_ratio,
            split_seed=args.split_seed,
            split_manifest_file=args.split_manifest,
            split_manifest_name=args.split,
            max_sessions=args.max_sessions,
            max_samples_per_session=args.max_samples_per_session,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
            require_tool_call=args.require_tool_call,
            min_target_tokens=args.min_target_tokens,
        )
        source = AgentLLMTracesSource(data_args)
        session_skips: Counter[str] = Counter()
        # Redirect per-call skip accounting so we can attribute source skips
        # to subsets; _session_examples only touches self.source_skips.
        source.source_skips = session_skips
    else:
        logger.warning(
            "Harness import unavailable (%r); using verbatim stdlib fallback copies.",
            _IMPORT_ERROR,
        )
        source = FallbackSource(args)

    counter = ToolDocTokenCounter(args.tokenizer_backend, args.tokenizer_path)

    _, eval_ids = source.split_session_ids()
    logger.info("Census over %d eval sessions of split %s", len(eval_ids), args.split)

    buckets: Dict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    overall = _new_bucket()

    for session in source.sessions:
        session_id = session["session_id"]
        if session_id not in eval_ids:
            continue
        subset = str(session.get("subset") or "unknown")
        spans = session["spans"]

        if _HAVE_HARNESS:
            session_skips.clear()
            examples = source._session_examples(session_id, spans, subset)
            skips = Counter(session_skips)
        else:
            source.source_skips = Counter()
            examples = source._session_examples(session_id, spans, subset)
            skips = source.source_skips

        n_valid = len(examples)
        # Mirror inspect_agent_llm_traces_bigpool_stats.py:256 (cap for counts
        # only; iter_examples rng.sample at train_...:702-703 picks WHICH spans
        # but the count is exactly min(n_valid, cap)).
        n_capped = min(n_valid, args.max_samples_per_session) if args.max_samples_per_session else n_valid

        # Session-level tool_definition, mirroring inspect script lines 257-263.
        tool_definition = _ex_field(examples[0], "tool_definition") if examples else ""
        if not tool_definition:
            for span in spans:
                tool_value = _span_attributes(span).get("gen_ai.tool.definitions")
                if tool_value:
                    tool_definition = _canonical_tool_definition(tool_value)
                    break

        # Hypothetical min_target_tokens filter (NOT in the real eval chain).
        hypothetical_drops = 0
        if args.min_target_tokens and args.min_target_tokens > 0:
            for example in examples:
                if counter.answer_tokens(_ex_field(example, "answer")) < args.min_target_tokens:
                    hypothetical_drops += 1

        if tool_definition:
            doc_tokens = counter.full_doc_tokens(tool_definition)
            survives, reason = _full_mode_chunk_outcome(
                doc_tokens,
                args.max_tool_definition_tokens,
                args.max_doc_length,
                args.max_doc_num,
                args.truncate_tool_definition,
            )
        else:
            # Cannot happen for sessions with valid spans (_session_examples
            # requires a tool_definition), recorded for completeness.
            doc_tokens = None
            survives, reason = False, "no_tool_definition"

        for bucket in (buckets[subset], overall):
            bucket["sessions_in_split"] += 1
            bucket["spans_raw"] += len(spans)
            bucket["source_skip_reasons"].update(skips)
            if n_valid > 0:
                bucket["sessions_with_valid_spans"] += 1
                bucket["spans_after_require_tool_call"] += n_valid
                # min_target_tokens stage: identity in the real eval chain.
                bucket["spans_after_min_target_tokens"] += n_valid
                bucket["hypothetical_min_target_drops"] += hypothetical_drops
            if doc_tokens is not None:
                bucket["doc_tokens"].append(doc_tokens)
            if n_valid > 0:
                if survives:
                    bucket["sessions_after_tool_threshold"] += 1
                    bucket["spans_after_tool_threshold"] += n_valid
                    bucket["spans_after_session_cap"] += n_capped
                else:
                    bucket["threshold_skip_reasons"][reason] += 1

    def _finalize(bucket: Dict[str, Any], subset: str) -> Dict[str, Any]:
        out = {
            "subset": subset,
            "sessions_in_split": bucket["sessions_in_split"],
            "spans_raw": bucket["spans_raw"],
            "sessions_with_valid_spans": bucket["sessions_with_valid_spans"],
            "spans_after_require_tool_call": bucket["spans_after_require_tool_call"],
            "spans_after_min_target_tokens": bucket["spans_after_min_target_tokens"],
            "hypothetical_min_target_drops": bucket["hypothetical_min_target_drops"],
            "sessions_after_tool_threshold": bucket["sessions_after_tool_threshold"],
            "spans_after_tool_threshold": bucket["spans_after_tool_threshold"],
            "spans_after_session_cap": bucket["spans_after_session_cap"],
            "source_skip_reasons": dict(bucket["source_skip_reasons"]),
            "threshold_skip_reasons": dict(bucket["threshold_skip_reasons"]),
            "tool_definition_doc_tokens": _stat(bucket["doc_tokens"]),
        }
        out["entry"] = _verdict(out["spans_after_session_cap"], args.psi)
        return out

    subset_rows = [_finalize(bucket, subset) for subset, bucket in sorted(buckets.items())]
    overall_row = _finalize(overall, "ALL")

    result = {
        "git_commit": _git_commit(),
        "dataset_path": args.dataset_path,
        "split_manifest": args.split_manifest,
        "split": args.split,
        "harness_import": _HAVE_HARNESS,
        "harness_import_error": None if _HAVE_HARNESS else repr(_IMPORT_ERROR),
        "tokenizer_backend": args.tokenizer_backend,
        "tokenizer_path": args.tokenizer_path if args.tokenizer_backend == "hf" else None,
        "token_count_mode": (
            "chat_template_ids (harness-exact)"
            if args.tokenizer_backend == "hf"
            else "ESTIMATE ceil(chars/4) -- NOT harness-exact, dry-run only"
        ),
        "params": {
            "max_tool_definition_tokens": args.max_tool_definition_tokens,
            "max_doc_num": args.max_doc_num,
            "max_doc_length": args.max_doc_length,
            "max_context_tokens": args.max_doc_length * args.max_doc_num,
            "truncate_tool_definition": args.truncate_tool_definition,
            "min_target_tokens": args.min_target_tokens,
            "max_samples_per_session": args.max_samples_per_session,
            "require_tool_call": args.require_tool_call,
            "psi": args.psi,
            "eval_ratio": args.eval_ratio,
            "split_seed": args.split_seed,
            "max_sessions": args.max_sessions,
            "tool_document_eval_mode": "full (fixed; harness TOOL_DOCUMENT_EVAL_MODE=full)",
        },
        "filter_chain_notes": [
            "split: manifest eval_session_ids intersected with dataset sessions "
            "(train_agent_tool_definition_c2kv.py:634-687).",
            "require_tool_call stage = all _session_examples span filters "
            "(train_agent_tool_definition_c2kv.py:748-806).",
            "min_target_tokens is NOT a filter in the eval selection chain "
            "(accepted at eval_agent_tool_definition_c2kv.py:820/947, unused by "
            "AgentLLMTracesSource; training-only at "
            "train_agent_tool_definition_c2kv.py:984-993). Stage reported as identity; "
            "hypothetical_min_target_drops is informational only.",
            "tool threshold/packing: document_mode=full, "
            "eval_agent_tool_definition_c2kv.py:172-187.",
            "per-session cap count = min(n_valid, cap); commutes with the "
            "threshold because tool_definition is session-level "
            "(train_agent_tool_definition_c2kv.py:749-755, 702-703).",
        ],
        "round1_reference": ROUND1_REFERENCE,
        "subsets": subset_rows,
        "overall": overall_row,
    }
    return result


def _md_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# S0 census: big-tool-pool usable spans (PR-E entry decision)")
    lines.append("")
    lines.append(f"- git commit: `{result['git_commit']}`")
    lines.append(f"- dataset: `{result['dataset_path']}`")
    lines.append(f"- split manifest: `{result['split_manifest']}` :: `{result['split']}` (eval ids)")
    lines.append(f"- harness import: `{result['harness_import']}` (fallback stdlib copies if False)")
    lines.append(f"- token count mode: **{result['token_count_mode']}**")
    params = result["params"]
    lines.append(
        "- packing: MAX_TOOL_DEFINITION_TOKENS={max_tool_definition_tokens}, "
        "MAX_DOC_NUM={max_doc_num}, MAX_DOC_LENGTH={max_doc_length} "
        "(context {max_context_tokens}), truncate={truncate_tool_definition}".format(**params)
    )
    lines.append(
        "- filters: require_tool_call={require_tool_call}, "
        "min_target_tokens={min_target_tokens} (no-op in eval chain), "
        "max_samples_per_session={max_samples_per_session}".format(**params)
    )
    lines.append("")

    rows = result["subsets"] + [result["overall"]]
    lines.append("## Filter funnel (toolset_disjoint EVAL split)")
    lines.append("")
    lines.append(
        "| subset | sessions | spans raw | after require_tool_call | after min_target_tokens* | "
        "sessions after tool threshold | spans after tool threshold | final usable N (after cap) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['subset']} | {row['sessions_in_split']} | {row['spans_raw']} | "
            f"{row['spans_after_require_tool_call']} | {row['spans_after_min_target_tokens']} | "
            f"{row['sessions_after_tool_threshold']} | {row['spans_after_tool_threshold']} | "
            f"{row['spans_after_session_cap']} |"
        )
    lines.append("")
    lines.append(
        "\\* min_target_tokens is accepted by the eval CLI but never applied as a selection "
        "filter (eval_agent_tool_definition_c2kv.py:820/947; training-only at "
        "train_agent_tool_definition_c2kv.py:984-993) — identity stage."
    )
    lines.append("")

    lines.append("## Skip reasons")
    lines.append("")
    lines.append("| subset | source span skips | tool-threshold skips | hypothetical min_target drops** |")
    lines.append("|---|---|---|---:|")
    for row in rows:
        lines.append(
            f"| {row['subset']} | {json.dumps(row['source_skip_reasons'])} | "
            f"{json.dumps(row['threshold_skip_reasons'])} | {row['hypothetical_min_target_drops']} |"
        )
    lines.append("")
    lines.append(
        "\\*\\* spans whose answer tokenizes below min_target_tokens; NOT part of the "
        "real eval chain, informational only."
    )
    lines.append("")

    lines.append("## Tool-definition doc tokens (full-mode chat-template document)")
    lines.append("")
    lines.append("| subset | p50 | p95 | max |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        stat = row["tool_definition_doc_tokens"]
        lines.append(f"| {row['subset']} | {stat['p50']} | {stat['p95']} | {stat['max']} |")
    lines.append("")

    lines.append("## Entry-formula verdict: MDE(N) = z_0.975 * sqrt(psi / N)")
    lines.append("")
    lines.append(f"psi = {params['psi']}; MDE expressed in points (x100).")
    lines.append("")
    lines.append("| subset | N (usable spans) | MDE (points) | verdict |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        entry = row["entry"]
        mde = "n/a" if entry["MDE_points"] is None else f"{entry['MDE_points']:.3f}"
        lines.append(f"| {row['subset']} | {entry['N']} | {mde} | {entry['verdict']} |")
    lines.append("")

    ref = result["round1_reference"]
    lines.append("## Round-1 reference (sanity anchor; different thresholds — eyeball only)")
    lines.append("")
    lines.append(f"- round-1 config: {ref['config']}")
    lines.append(f"- appworld: {ref['appworld_sessions']} sessions / {ref['appworld_spans_at_10k_filter']} spans at the 10k filter")
    lines.append(f"- tau2_telecom: {ref['tau2_telecom_sessions']} sessions / {ref['tau2_telecom_spans']} spans")
    lines.append(f"- total usable at 32k: {ref['total_usable_spans_at_32k']} spans")
    lines.append(
        f"- manifest toolset_disjoint eval ids: {ref['manifest_toolset_disjoint_eval_session_ids']}"
    )
    lines.append("")
    lines.append("## Filter-chain notes")
    lines.append("")
    for note in result["filter_chain_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S0 census (pure CPU, read-only): count usable spans surviving the real "
            "big-tool-pool harness filter chain on the toolset_disjoint EVAL split."
        )
    )
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--split_manifest", default="configs/agent_tooldef_split_manifests.json")
    parser.add_argument(
        "--split",
        default="toolset_disjoint",
        help="Manifest split name whose EVAL session ids are censused (harness runs --split eval).",
    )
    parser.add_argument("--max_tool_definition_tokens", type=int, default=97000)
    parser.add_argument("--max_doc_num", type=int, default=96)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_samples_per_session", type=int, default=16)
    parser.add_argument(
        "--require_tool_call",
        type=lambda x: str(x).lower() == "true",
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--truncate_tool_definition",
        type=lambda x: str(x).lower() == "true",
        nargs="?",
        const=False,
        default=False,
    )
    parser.add_argument("--tokenizer_path", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--tokenizer_backend",
        choices=["hf", "estimate"],
        default="hf",
        help="hf = harness-exact chat-template token counts (needs transformers + tokenizer). "
        "estimate = ceil(chars/4), CPU-only dry runs, NOT harness-exact.",
    )
    parser.add_argument("--psi", type=float, default=0.5)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--out_prefix", required=True, help="Writes <out_prefix>.json and <out_prefix>.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = census(args)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_md_report(result), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)

    overall = result["overall"]
    entry = overall["entry"]
    print(
        f"TOTAL usable spans N={entry['N']} "
        f"MDE={entry['MDE_points'] if entry['MDE_points'] is not None else 'n/a'} points "
        f"-> {entry['verdict']}"
    )
    for row in result["subsets"]:
        entry = row["entry"]
        print(
            f"  {row['subset']}: N={entry['N']} "
            f"MDE={entry['MDE_points'] if entry['MDE_points'] is not None else 'n/a'} -> {entry['verdict']}"
        )


if __name__ == "__main__":
    main()
