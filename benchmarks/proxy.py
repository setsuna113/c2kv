"""Arm-aware OpenAI-compatible reverse proxy (backend-abstracted).

All three benchmarks speak the OpenAI chat-completions protocol, so instead
of forking each benchmark we front the serving stack with this proxy:

    benchmark client -> proxy (arm assembly) -> backend (hf_server | sglang)

Per request the proxy decides, per the active arm, which *history* messages
are sent as raw text and which are replaced by a server-side gist reference
(``c2kv_key_hash``, produced by POST /v1/c2kv/extract).  The rule for what
counts as history: every message except the trailing block after the last
user/tool message; system messages are history by position but always kept
raw (never compressed).  The final user message and tool results of the
current turn stay raw.  This is the training rule
(train_data_multiturn._session_examples: everything before the last input
message is compressed).

How the compressed history is cut into docs is ``--doc-packing``
(docs/c2kv_semantics.md):

* ``turn`` (default) — the TRAINING format.  History is normalized like
  train_data_multiturn._normal_agent_message (tool->user, assistant
  tool_calls rendered as the Action dialect) and packed one doc per turn
  exactly like _agent_history_turn_docs ("Previous turn\n[User query]...\n
  [Assistant output]..."), split to <= --max-doc-length tokens and tail-
  selected to --max-doc-num docs with the doc-0 anchor (_fit_reused_history).
  Every doc is extracted as a user-role message.
* ``message`` — the pre-2026-09 bench format: one doc per message with its
  own role, no splitting, no cap.  Kept for reproducing older numbers.

Assistant tool_calls turns are rendered into the TRAINING dialect (content
+ "Action:" + minified <tool_call> JSON) on EVERY outgoing path —
compressed AND raw — so a backend without server-side normalization
(SGLang) sees the same surface the old hf_server normalized itself.

Oracle-recover arms (``recover`` in arms.py) implement the step-level
contract: during a full-arm run the proxy RECORDS a reference trajectory
(``--record-reference``); a recover arm then compares every generated
action against the reference entry with the same message fingerprint,
flags the first mismatch as ``divergence_step``, and ONCE per conversation
re-sends the identical payload assembled in full-raw mode — the
regenerated step replaces the divergent one.

Repair arms (``repair`` in arms.py): the target doc is selected by
benchmarks/repair_policy.py IN THE PROXY; the backend turns the plan into
its own protocol.  On sglang the raw KV is extracted with the FULL-CONTEXT
form of /v1/c2kv/repair_extract (messages + target_index + tools: the
server renders the prefix exactly like the chat request and captures the
target doc's KV inside it) and injected with an explicit
``c2kv_repair_placement`` (in_place / append_keep_ledger / append_tail,
see docs/c2kv_semantics.md "Repair placement").

History-KV eviction arms (``history_kv`` in arms.py) are the upstream
baselines (StreamingLLM / H2O / SnapKV / PyramidKV): nothing is gisted, and
the completed history is instead compressed by TOKEN EVICTION inside the
server.  The proxy owns the split (which assembled messages are completed
history, how they render into one span) and the per-conversation streaming
session; the backend owns the wire form.  See README "History-KV eviction
arms" for the two deviations from the upstream client and for the fact that
these arms have never been run against a live server.

Upstream failures are retried (2x, exponential backoff) and always leave a
request-log row with a failure kind — a benchmark entry must never vanish
without a trace.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from memory_runtime.extraction_telemetry import (
    ExtractionBudget, ExtractionBudgetExceeded, capture_extractions,
    current_extraction_trace, extraction_sources, positive_extraction_limit,
)
from memory_runtime.generation_budget import (
    GenerationBudget, GenerationBudgetExceeded, positive_generation_limit,
)
from memory_runtime.attempt_journal import AttemptJournal, attempt_journal_path

# this proxy is always a local sidecar talking to 127.0.0.1 upstreams; an
# ambient http_proxy env (login shells here carry one) must never intercept
# its upstream calls
_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))

import repair_policy
import textarms
from arms import Arm, get_arm, history_kv_spec, kv_reuse_spec  # type: ignore
from backends import BackendError, get_backend  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from d_witness_core import select_k_star, witness_scores

WITNESS_TOKENIZER = None
WITNESS_TOKENIZER_PATH = ""


def _witness_texts(records):
    """Decode the exact standalone grid rows used by /c2kv/extract."""
    global WITNESS_TOKENIZER
    if WITNESS_TOKENIZER is None:
        if not WITNESS_TOKENIZER_PATH:
            raise ValueError("gold witness requires --witness-tokenizer")
        from transformers import AutoTokenizer
        WITNESS_TOKENIZER = AutoTokenizer.from_pretrained(
            WITNESS_TOKENIZER_PATH, local_files_only=True)
    texts = []
    for item in records:
        rendered = WITNESS_TOKENIZER.apply_chat_template(
            [{"role": item["role"], "content": item["content"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        bos = WITNESS_TOKENIZER.bos_token
        if bos and rendered.startswith(bos):
            rendered = rendered[len(bos):]
        ids = WITNESS_TOKENIZER.encode(rendered, add_special_tokens=False)
        expected = int(item["record"]["original_seq_len"])
        if len(ids) != expected:
            raise ValueError(f"witness tokenization differs from extraction: {len(ids)} != {expected}")
        texts.append(WITNESS_TOKENIZER.decode(ids, skip_special_tokens=False))
    return texts


def plan_gold_repair(messages, arm, counts, oracle, tools, out_messages, force=False):
    """One frozen witness block for a benchmark-authorized failed turn.

    Gold controls only selection. It never becomes model-visible text.
    Raw KV is recomputed in context by repair_extract; this is not an
    offload-transfer latency measurement.
    """
    if not oracle:
        return None
    oracle_kind = oracle.get("kind")
    if not arm.gold_recovery or oracle_kind not in {
            "bfcl_gold_turn_v1", "bfcl_gold_turn_v2", "bfcl_gold_turn_v3"}:
        raise ValueError("privileged recovery payload requires a BFCL gold-recovery arm")
    if oracle.get("selector") != arm.gold_recovery:
        raise ValueError("oracle selector conflicts with the arm")
    records = counts.get("compressed_records") or []
    if not records:
        counts["gold_recovery"] = {"status": "no_compressed_history"}
        return None
    values = oracle.get("values")
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("witness values must be a list of strings")
    # Keep v1/v2 selections separate in a long-lived proxy: a v2 retry may
    # start at a different request step for the same task/turn.  v1 did not
    # carry an explicit numeric version, so retain its implicit version 1.
    oracle_version = oracle.get(
        "version", {
            "bfcl_gold_turn_v1": 1,
            "bfcl_gold_turn_v2": 2,
            "bfcl_gold_turn_v3": 3,
        }[oracle_kind])
    event = (str(oracle_kind), str(oracle_version),
             str(oracle.get("task_id")), int(oracle["turn"]))
    with STATE.lock:
        chosen = STATE.gold_choices.get(event)
    if chosen is None:
        texts = _witness_texts(records)
        df, scores = witness_scores(texts, values)
        witness = select_k_star(texts, values)
        index = witness
        if arm.gold_recovery == "random" and witness is not None:
            # Same token budget where possible; report absence of a match.
            budget = records[witness]["record"]["original_seq_len"]
            candidates = [i for i, r in enumerate(records)
                          if r["record"]["original_seq_len"] == budget]
            seed = int(_digest([event, "random-block-v1"]), 16)
            index = candidates[seed % len(candidates)]
        chosen = {
            "content_key": (_content_key(records[index]["role"], records[index]["content"])
                            if index is not None else None),
            "witness_index": witness, "selected_index_at_trigger": index,
            "scores": scores, "selector": arm.gold_recovery,
            "candidate_count": len(records),
        }
        with STATE.lock:
            chosen = STATE.gold_choices.setdefault(event, chosen)
    if chosen["content_key"] is None:
        counts["gold_recovery"] = {**chosen, "status": "no_literal_witness"}
        return None
    matches = [i for i, r in enumerate(records)
               if _content_key(r["role"], r["content"]) == chosen["content_key"]]
    if not matches:
        counts["gold_recovery"] = {**chosen, "status": "selected_doc_no_longer_resident"}
        return None
    index = matches[0]
    repair_arm = replace(arm, gold_recovery=None,
                         repair={"policy": f"offset:{index}", "placement": "append_keep_ledger"})
    cache_key = _digest([event, out_messages[:records[index]["out_index"] + 1], tools])
    with STATE.lock:
        plan = None if force else STATE.gold_plans.get(cache_key)
    t0 = time.perf_counter()
    cache_hit = plan is not None
    if plan is None:
        plan = plan_repair(messages, repair_arm, counts, tools=tools, out_messages=out_messages)
        with STATE.lock:
            STATE.gold_plans[cache_key] = plan
    # The raw prefix can be unchanged while later tool steps append history.
    # Reuse the KV entry, but resolve carrier insertion against this request.
    plan = dict(plan)
    plan["target_out_index"] = records[index]["out_index"]
    if "current_start_out_index" in counts:
        plan["current_start_out_index"] = counts["current_start_out_index"]
    plan["doc_index"] = index
    counts["gold_recovery"] = {
        **chosen, "status": "appended", "selected_index": index,
        "raw_kv_source": "full_context_recompute_from_text",
        "raw_kv_cache_hit": cache_hit,
        "recovery_extract_sec": (time.perf_counter() - t0) if not cache_hit else 0.0,
        "recovery_block_tokens": plan.get("repair_block_tokens"),
        "event_id": list(event),
        "oracle_kind": oracle_kind,
        "oracle_version": oracle_version,
    }
    return plan


class ExtractCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    def get_or_put(
        self, key: Tuple[str, str, int, str], producer, force: bool = False
    ) -> Dict[str, Any]:
        """Memoised extract.  ``force=True`` re-runs the producer and
        overwrites the memo: it is the cache-miss recovery path, where the
        memoised record describes a pool entry the SERVER has already evicted,
        so returning it would re-send a dead ``key_hash`` and miss again."""
        trace = current_extraction_trace()
        if trace is None:
            with self._lock:
                if force or key not in self._cache:
                    index = EXTRACTION_BUDGET.reserve()
                    handle = _start_attempt("extraction", index)
                    try:
                        self._cache[key] = producer()
                    except BaseException:
                        _finish_attempt(handle, "failed")
                        raise
                    else:
                        _finish_attempt(handle, "completed")
                return self._cache[key]
        started = time.perf_counter()
        with self._lock:
            cache_hit = not force and key in self._cache
            produced_sec = 0.0
            record = None
            error_type = None
            budget_attempt_index = None
            producer_called = False
            try:
                if not cache_hit:
                    budget_attempt_index = EXTRACTION_BUDGET.reserve()
                    handle = _start_attempt("extraction", budget_attempt_index)
                    producer_called = True
                    produced_start = time.perf_counter()
                    try:
                        self._cache[key] = producer()
                    except BaseException:
                        _finish_attempt(handle, "failed")
                        raise
                    else:
                        _finish_attempt(handle, "completed")
                    finally:
                        produced_sec = time.perf_counter() - produced_start
                record = self._cache[key]
                return record
            except Exception as error:
                error_type = type(error).__name__
                raise
            finally:
                trace.record_lookup(
                    key, cache_hit=cache_hit, force=force,
                    lookup_wall_sec=time.perf_counter() - started,
                    producer_wall_sec=produced_sec, result=record,
                    error_type=error_type, producer_called=producer_called,
                    budget_attempt_index=budget_attempt_index)


CACHE = ExtractCache()
ARM: Optional[Arm] = None
BACKEND = None  # set in main()
MEMORY_RUNTIME = None  # opt-in RuntimeAdapter, loaded in main()
MEMORY_RUNTIME_BYTES_PER_KV_TOKEN: Optional[int] = None
MEMORY_RUNTIME_FATAL_ERROR: Optional[str] = None
_memory_runtime_verify_lock = threading.Lock()
NO_UPSTREAM_RETRIES = False
CAPTURE_REQUEST_VIEWS = False
GENERATION_BUDGET = GenerationBudget(None)
EXTRACTION_BUDGET = ExtractionBudget(None)
ATTEMPT_JOURNAL = None
_ATTEMPT_REQUEST = ContextVar("proxy_attempt_request", default=None)
UPSTREAM = ""
REQUEST_LOG_PATH = ""


def _start_attempt(kind, index):
    if ATTEMPT_JOURNAL is None:
        return None
    context = _ATTEMPT_REQUEST.get() or {}
    return ATTEMPT_JOURNAL.start(
        kind, index, context.get("request_id", uuid.uuid4().hex),
        context.get("eval_context", {}))


def _finish_attempt(handle, status, usage=None):
    if handle is not None:
        ATTEMPT_JOURNAL.finish(handle, status, usage=usage)

SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
    "max_completion_tokens", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "stop", "tool_choice", "parallel_tool_calls",
    "response_format", "chat_template_kwargs",
)


def _sampling_fields(payload):
    """Record explicit wire fields without guessing server-side defaults."""
    return {key: payload[key] for key in SAMPLING_FIELDS if key in payload}


def _captured_request_view(payload):
    """Allowlist benchmark content only; never capture transport headers."""
    view = {key: payload[key] for key in ("model", "messages", "tools") if key in payload}
    view["sampling"] = _sampling_fields(payload)
    return json.loads(json.dumps(view, ensure_ascii=False))
_log_lock = threading.Lock()

# --doc-packing / --max-doc-length / --max-doc-num (see module docstring and
# docs/c2kv_semantics.md).  DEFAULTS = the checkpoint-1088 training values
# (HISTORY_MAX_DOC_LENGTH / HISTORY_MAX_DOC_NUM in
# agent/train_agent_history_c2kv_npu.sh: 512/12) — serving must match
# training; 768/16 (the old D-line harness caliber) is available by flag
# but shifts every compression arm off its trained regime.
DOC_PACKING = "turn"
QUERY_PROJECTION = None
MAX_DOC_LENGTH = 512
MAX_DOC_NUM = 12
DOC_PACKINGS = ("turn", "message")

_MEMORY_RUNTIME_C2KV_MODES = frozenset(
    {"legacy", "protect", "recover_once", "persistent", "capacity_protect",
     "capacity_exact_once", "capacity_exact_persistent"})
_MEMORY_RUNTIME_FULL_MODES = frozenset(
    {"no_gist", "full_shared", "raw_recency", "full_capacity_aux", "full_exact_shared",
     "capacity_exact_no_gist"})
_MEMORY_RUNTIME_FORBIDDEN_CONTROL_WORDS = (
    "oracle", "recover", "gold", "witness", "repair", "history_kv", "historykv")


class MemoryRuntimeError(RuntimeError):
    """A-runtime contract failure; the request must not be retried."""

    kind = "memory_runtime_error"

    def __init__(self, message, *, kind=None):
        super().__init__(message)
        if kind is not None:
            self.kind = kind


def _load_memory_runtime(config_path: str, tokenizer_path: str):
    """Load the runtime lazily so legacy proxy processes need no dependency."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    module_name, class_name = (
        ("phase4_policy", "Phase4RuntimeAdapter")
        if isinstance(config, dict) and "phase4_policy" in config
        else ("adapter", "RuntimeAdapter")
    )
    try:
        module = importlib.import_module(f"benchmarks.memory_runtime.{module_name}")
    except ModuleNotFoundError as error:
        # ``run.py`` launches this file by path.  Some Python environments put
        # only benchmarks/ (not the repository root) on sys.path, in which
        # case the same package is reachable as memory_runtime.adapter.
        if error.name not in {"benchmarks", "benchmarks.memory_runtime"}:
            raise
        module = importlib.import_module(f"memory_runtime.{module_name}")
    return getattr(module, class_name).from_config(config_path, tokenizer_path)


def _validate_memory_runtime_arm(runtime, arm: Arm) -> None:
    """A modes deliberately reuse only the existing plain c2kv4/full arms."""
    mode = getattr(runtime, "mode", None)
    if mode in _MEMORY_RUNTIME_C2KV_MODES:
        expected_arm = "c2kv4"
    elif mode in _MEMORY_RUNTIME_FULL_MODES:
        expected_arm = "full"
    else:
        raise MemoryRuntimeError(f"unsupported memory runtime mode: {mode!r}")

    conflicts = [
        name for name in (
            "hybrid_top_k", "constrain_tools", "repair", "recover",
            "text_policy", "history_kv", "kv_reuse", "gold_recovery",
            "native_messages",
        ) if getattr(arm, name, None)
    ]
    if arm.name != expected_arm or conflicts:
        detail = f"; conflicting arm fields: {', '.join(conflicts)}" if conflicts else ""
        raise MemoryRuntimeError(
            f"memory runtime mode {mode!r} requires plain arm {expected_arm!r}, "
            f"got {arm.name!r}{detail}")


def _memory_runtime_payload_conflicts(payload: Dict[str, Any]) -> List[str]:
    """Find client-supplied controls that would mix A with privileged paths."""
    conflicts: List[str] = []

    def inspect(mapping: Any, prefix: str) -> None:
        if not isinstance(mapping, dict):
            return
        for key in mapping:
            lowered = str(key).lower().replace("-", "_")
            if lowered == "c2kv_kv_memory_hint" or (
                    lowered.startswith("c2kv_")
                    and any(word in lowered for word in
                            _MEMORY_RUNTIME_FORBIDDEN_CONTROL_WORDS)):
                conflicts.append(f"{prefix}{key}")

    inspect(payload, "")
    for index, message in enumerate(payload.get("messages") or []):
        inspect(message, f"messages[{index}].")
    return sorted(set(conflicts))


def _memory_runtime_error_counts() -> Optional[Dict[str, Any]]:
    if MEMORY_RUNTIME is None:
        return None
    return {"memory_runtime": {
        "mode": getattr(MEMORY_RUNTIME, "mode", None),
        "route_mode": getattr(MEMORY_RUNTIME, "route_mode", None),
        "compression_policy": getattr(MEMORY_RUNTIME, "compression_policy", None),
        "byte_geometry_verified_by_backend": False,
    }}


def _check_memory_runtime_fatal() -> None:
    with _memory_runtime_verify_lock:
        error = MEMORY_RUNTIME_FATAL_ERROR
    if error:
        raise MemoryRuntimeError(error)


def _apply_memory_runtime(messages, assembled, counts, eval_context, tools, *, source_predictor=None):
    """Invoke the stateful adapter and pin the proxy-visible count contract."""
    if MEMORY_RUNTIME is None:
        return assembled, counts
    counts.setdefault("memory_runtime", {
        "mode": getattr(MEMORY_RUNTIME, "mode", None),
        "byte_geometry_verified_by_backend": False,
    })
    try:
        options = {}
        if getattr(MEMORY_RUNTIME, "source_needs_strategy", None) is not None:
            options["source_predictor"] = source_predictor
        if getattr(MEMORY_RUNTIME, "mode", None) == "raw_recency":
            # Select original source messages before the training-compatible
            # Full renderer inserts the system and normalizes tool actions.
            options["render_full"] = lambda source: _assemble(source, get_arm("full"))
        elif (getattr(MEMORY_RUNTIME, "mode", None) in {
                "capacity_protect", "capacity_exact_once", "capacity_exact_persistent"}
                or getattr(MEMORY_RUNTIME, "route_mode", None) == "ac_gist_static"):
            options["render_compressed"] = lambda source: _assemble(source, get_arm("c2kv4"))
        assembled, counts = MEMORY_RUNTIME.apply(
            messages, assembled, counts, eval_context, tools, **options)
    except MemoryRuntimeError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MemoryRuntimeError(str(error), kind=getattr(error, "kind", None)) from error
    return _pin_memory_runtime_output(assembled, counts)


def _pin_memory_runtime_output(assembled, counts):
    if not isinstance(assembled, list) or not isinstance(counts, dict):
        raise MemoryRuntimeError("RuntimeAdapter.apply must return (list, dict)")
    metadata = counts.get("memory_runtime")
    if not isinstance(metadata, dict):
        raise MemoryRuntimeError("RuntimeAdapter.apply omitted counts['memory_runtime']")
    mode = getattr(MEMORY_RUNTIME, "mode", None)
    if metadata.get("mode") not in (None, mode):
        raise MemoryRuntimeError("runtime metadata mode conflicts with the loaded adapter")
    metadata["mode"] = mode
    expected = metadata.get("bytes_per_kv_token")
    if type(expected) is not int or expected <= 0:
        raise MemoryRuntimeError(
            "runtime metadata needs a positive integer bytes_per_kv_token")
    # Only the backend observation below may set this true.
    metadata["byte_geometry_verified_by_backend"] = False
    return assembled, counts


def _commit_memory_runtime_final(prepared, normalized, counts):
    """Commit only the one normalized response that the proxy will return."""
    commit_final = getattr(MEMORY_RUNTIME, "commit_final", None)
    if not callable(commit_final):
        return
    metadata = counts.get("memory_runtime")
    if not isinstance(metadata, dict):
        raise MemoryRuntimeError("RuntimeAdapter omitted counts['memory_runtime']")
    try:
        metadata["phase4_final"] = commit_final(
            prepared, normalized.get("tool_calls"))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MemoryRuntimeError(str(error)) from error


def _prepare_memory_input(messages, eval_context, tools, *, source_predictor=None):
    """Check Full capacity before any optional compression/extraction work."""
    initial_arm = (get_arm("full") if getattr(MEMORY_RUNTIME, "always_compress", False)
                   or getattr(MEMORY_RUNTIME, "mode", None)
                   in {"capacity_protect", "capacity_exact_once", "capacity_exact_persistent"}
                   else ARM)
    assembled, counts = _assemble(messages, initial_arm)
    return _apply_memory_runtime(messages, assembled, counts, eval_context, tools,
                                 source_predictor=source_predictor)


def _prepare_exact_memory_input(messages, eval_context, tools):
    assembled, counts = _assemble(messages, get_arm("full"))
    try:
        assembled, counts, prepared = MEMORY_RUNTIME.prepare_exact(
            messages, assembled, counts, eval_context, tools,
            render_compressed=lambda source: _assemble(source, get_arm("c2kv4")))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MemoryRuntimeError(str(error), kind=getattr(error, "kind", None)) from error
    assembled, counts = _pin_memory_runtime_output(assembled, counts)
    return assembled, counts, prepared


def _generation_summary(records):
    """Account for every attempted generation, including an unknown failed call."""
    fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
    fields.update(key for record in records for key in (record.get("usage") or {}))
    totals = {}
    for key in sorted(fields):
        values = [(record.get("usage") or {}).get(key) for record in records]
        totals[key] = (sum(values) if values and all(
            type(value) in (int, float) for value in values) else None)
    return {
        "generation_attempts": len(records),
        "generation_completed": sum(record["status"] == "completed" for record in records),
        "generation_usage_total": totals,
        "generation_upstream_wall_sec": sum(record.get("wall_sec", 0) for record in records),
        "usage_scope": "standard usage and backend cost describe final generation; generation_usage_total includes all attempts; null means unknown",
    }


def _verify_memory_runtime_kv_bytes(
        counts: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    """Latch the first server KV geometry and fail closed on disagreement."""
    global MEMORY_RUNTIME_BYTES_PER_KV_TOKEN, MEMORY_RUNTIME_FATAL_ERROR
    if MEMORY_RUNTIME is None:
        return
    metadata = counts["memory_runtime"]
    expected = metadata["bytes_per_kv_token"]
    actual = (normalized.get("cost") or {}).get("bytes_per_kv_token")
    metadata["byte_geometry_verified_by_backend"] = False
    if actual is None:
        error = "backend omitted bytes_per_kv_token required for runtime budget verification"
    elif type(actual) is not int or actual <= 0:
        error = f"backend returned invalid bytes_per_kv_token: {actual!r}"
    else:
        with _memory_runtime_verify_lock:
            observed = MEMORY_RUNTIME_BYTES_PER_KV_TOKEN
            if observed is None:
                MEMORY_RUNTIME_BYTES_PER_KV_TOKEN = actual
                observed = actual
            if observed != actual:
                MEMORY_RUNTIME_FATAL_ERROR = (
                    "backend bytes_per_kv_token changed within the runtime: "
                    f"{observed} != {actual}")
            elif expected != actual:
                MEMORY_RUNTIME_FATAL_ERROR = (
                    "memory runtime bytes_per_kv_token disagrees with backend: "
                    f"{expected} != {actual}")
            error = MEMORY_RUNTIME_FATAL_ERROR
    if error:
        with _memory_runtime_verify_lock:
            MEMORY_RUNTIME_FATAL_ERROR = MEMORY_RUNTIME_FATAL_ERROR or error
            error = MEMORY_RUNTIME_FATAL_ERROR
        metadata["byte_geometry_verification"] = "missing" if actual is None else "mismatch"
        metadata["backend_bytes_per_kv_token"] = actual
        metadata["byte_geometry_error"] = error
        raise MemoryRuntimeError(error)
    metadata["backend_bytes_per_kv_token"] = actual
    metadata["byte_geometry_verification"] = "verified"
    metadata["byte_geometry_verified_by_backend"] = True
    if "c2kv_tools_dump_expected" in metadata:
        observed_dump = (normalized.get("cost") or {}).get("c2kv_tools_dump")
        metadata["backend_tools_dump"] = observed_dump
        if observed_dump != metadata["c2kv_tools_dump_expected"]:
            error = f"tool schema serialization disagrees with backend: {metadata['c2kv_tools_dump_expected']} != {observed_dump}"
            with _memory_runtime_verify_lock:
                MEMORY_RUNTIME_FATAL_ERROR = error
            raise MemoryRuntimeError(error)
    if "total_raw_prompt_tokens" in metadata:
        actual_raw = (normalized.get("usage") or {}).get("prompt_tokens")
        expected_raw = metadata["total_raw_prompt_tokens"]
        metadata["backend_raw_prompt_tokens"] = actual_raw
        metadata["raw_prompt_tokens_verified_by_backend"] = actual_raw == expected_raw
        if actual_raw != expected_raw:
            error = f"raw tokenizer count disagrees with backend: {expected_raw} != {actual_raw}"
            with _memory_runtime_verify_lock:
                MEMORY_RUNTIME_FATAL_ERROR = error
            raise MemoryRuntimeError(error)


class CacheMiss(RuntimeError):
    """SGLang c2kv pool eviction (400 C2KV cache miss) — recoverable by
    re-running /v1/c2kv/extract for the marked messages (the pool re-inserts
    the entry under the same content-derived hash) and retrying the chat."""

    def __init__(self, detail: str):
        super().__init__(f"c2kv cache miss: {detail[:500]}")


class UpstreamError(RuntimeError):
    """Non-200 transport failure after retries; carries the response body."""

    def __init__(self, status: int, body: str):
        super().__init__(f"upstream {status}: {body[:2000]}")
        self.status = status
        self.body = body


def _is_chat_path(path: str) -> bool:
    return path.endswith("/v1/chat/completions") or path.endswith(
        "/chat/completions"
    )


def _reserve_generation_attempt(path: str) -> Optional[int]:
    if _is_chat_path(path):
        context = _ATTEMPT_REQUEST.get() or {}
        task_id = (context.get("eval_context") or {}).get("task_id")
        return GENERATION_BUDGET.reserve(task_id)
    return None


def _post_json(path: str, payload: Dict[str, Any],
               timeout: int, retries: int = 2) -> Dict[str, Any]:
    """POST JSON to UPSTREAM, retrying 5xx/network failures with backoff.

    4xx (except 429) are deterministic client errors and are not retried.
    The final failure raises UpstreamError with the upstream body.  Note
    the SGLang stack reports many failures as HTTP 200 with error bodies —
    those are classified by the backend (BackendError), not here.
    """
    if (MEMORY_RUNTIME is not None or NO_UPSTREAM_RETRIES
            or GENERATION_BUDGET.enabled or EXTRACTION_BUDGET.enabled):
        # This function also backs gist extraction and repair extraction.
        # Freeze the complete transport attempt count, not only chat sends.
        retries = 0
    body = json.dumps(payload).encode("utf-8")
    last: Optional[UpstreamError] = None
    for attempt in range(retries + 1):
        req = urlrequest.Request(
            f"{UPSTREAM.rstrip('/')}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        index = _reserve_generation_attempt(path)
        # A durable start precedes transport. It does not prove that the
        # backend received the request if this process is killed next.
        handle = _start_attempt("generation", index) if index is not None else None
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except HTTPError as error:
            _finish_attempt(handle, "failed")
            text = ""
            try:
                text = error.read().decode("utf-8", "replace")
            except OSError:
                pass
            if error.code < 500 and error.code != 429:
                if error.code == 400 and "C2KV cache miss" in text:
                    # SGLang c2kv pool LRU eviction: the referenced gist is
                    # no longer resident (pool ~4437 tokens; long looping
                    # conversations evict their own early turns). Marked so
                    # the chat path can re-extract and retry instead of
                    # killing the task deterministically (~600s in, 5/5).
                    raise CacheMiss(text) from error
                raise UpstreamError(error.code, text) from error
            last = UpstreamError(error.code, text)
        except (URLError, OSError) as error:
            _finish_attempt(handle, "failed")
            last = UpstreamError(0, str(error))
        except BaseException:
            _finish_attempt(handle, "failed")
            raise
        else:
            # Journal write errors must propagate without a transport retry.
            _finish_attempt(handle, "completed", usage=(
                result.get("usage") if isinstance(result, dict) else None))
            return result
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
    assert last is not None
    raise last


def _content_key(role: str, content: str) -> str:
    return hashlib.sha256(f"{role}\x00{content}".encode("utf-8")).hexdigest()


def _extract(role: str, content: str, ratio: int, timeout: int = 600,
             tools: Optional[List[Dict[str, Any]]] = None,
             force: bool = False) -> Dict[str, Any]:
    # tools participate in the cache key: the same system text rendered
    # with different tool schemas yields different original_seq_len (the
    # Qwen template renders tools into the system block)
    tools_key = _digest(tools or [])
    key = (role, _content_key(role, content), ratio, tools_key)
    return CACHE.get_or_put(
        key, lambda: BACKEND.extract(content, role, ratio, tools=tools),
        force=force)


def _history_cutoff(messages: List[Dict[str, Any]]) -> int:
    """Index where the current (raw) block starts.

    Walk from the end: the trailing run that ends with the last user or tool
    message is current; everything before it is history.  System messages
    are classified as history here but are ALWAYS kept raw by _assemble
    (system prompts are never compressed).  A conversation whose last
    message is the user's is fully current except system/history — matching
    the teacher-forced harness (system + history gist, current prompt raw).
    """
    last_anchor = -1
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role")
        if role in ("user", "tool"):
            last_anchor = i
            break
    # The current block starts after the last assistant message that
    # precedes the final user/tool anchor.
    start = 0
    for i in range(last_anchor, -1, -1):
        if messages[i].get("role") == "assistant":
            start = i + 1
            break
    return start


def _render_action_dialect(message: Dict[str, Any]) -> str:
    """Assistant tool_calls -> the TRAINING dialect text (hf_server.chat's
    normalization, verbatim): content + "\\n\\n" + "Action:\\n" + the
    minified <tool_call> blocks.

    OpenAI-style assistant turns carry content=None with the actions in
    ``tool_calls``; extracting the bare content here used to send the literal
    string '""' to /v1/c2kv/extract — erasing every historical action from
    the compressed KV and drifting the logical ledger (original_seq_len~2
    instead of the real action length, shifting every later block's RoPE
    phases).  The same rendering must be used for the extract text and the
    compressed message content so a server-side re-extract reproduces the
    same gist."""
    blocks = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = function.get("arguments") or {}
        blocks.append(
            "<tool_call>\n"
            + json.dumps(
                {"name": function.get("name"), "arguments": arguments},
                ensure_ascii=False, separators=(",", ":"),
            )
            + "\n</tool_call>"
        )
    action = "Action:\n" + "\n".join(blocks)
    content = message.get("content") or ""
    return content + "\n\n" + action if content else action


def _normalize_history_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """train_data_multiturn._normal_agent_message, stdlib version.

    tool -> user; assistant tool_calls rendered as the Action dialect
    (_render_action_dialect == hf_server.chat == training renderer); a
    message that renders to nothing is dropped unless it is an assistant
    turn (training keeps empty assistant turns as empty outputs)."""
    role = message.get("role") or "user"
    if role == "tool":
        role = "user"
    content = message.get("content")
    if not isinstance(content, str):
        content = "" if content is None else json.dumps(content, ensure_ascii=False)
    if role == "assistant" and message.get("tool_calls"):
        content = _render_action_dialect(message)
    if not content and role != "assistant":
        return None
    return {"role": role, "content": content}


def _turn_docs(indexed_messages: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """train_data_multiturn._agent_history_turn_docs, stdlib version.

    One doc per turn: an input message (user query OR tool result, both are
    role user after normalization) opens a doc, every assistant output that
    follows joins it.  Rendered as
    "Previous turn\n[User query]\n...\n[Assistant output]\n..." and sent to
    the extractor as ONE user-role message.  ``source_indices`` records the
    original message indices behind each doc."""
    docs: List[Dict[str, Any]] = []
    current_user: Optional[str] = None
    outputs: List[str] = []
    sources: List[int] = []

    def flush() -> None:
        nonlocal current_user, outputs, sources
        if current_user is None and not outputs:
            return
        parts = ["Previous turn"]
        if current_user:
            parts.extend(["[User query]", current_user.strip()])
        if outputs:
            parts.extend([
                "[Assistant output]",
                "\n\n".join(item.strip() for item in outputs if item.strip()),
            ])
        docs.append({
            "role": "user",
            "content": "\n".join(parts).strip(),
            "source_indices": list(sources),
        })
        current_user = None
        outputs = []
        sources = []

    for index, message in indexed_messages:
        role = message.get("role", "user")
        content = str(message.get("content") or "").strip()
        if not content and role != "assistant":
            continue
        if role == "user":
            flush()
            current_user = content
        elif role == "assistant":
            outputs.append(content)
        else:
            outputs.append(f"[{role}]\n{content}")
        sources.append(index)
    flush()
    return docs


def _split_lines_keep(text: str) -> List[str]:
    """Line units with their newline kept (train_data_multiturn._semantic_units
    analogue): splitting only at line boundaries keeps the turn markers and
    tool-call blocks intact."""
    units = text.splitlines(keepends=True)
    return units or [text]


def _fit_doc(doc_text: str, ratio: int, extract_fn, max_doc_length: int,
             depth: int = 0) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract ``doc_text``; if its template length exceeds ``max_doc_length``
    split it (train_data_multiturn._split_message_to_fit: greedy line
    accumulation against a char budget, then hard halves) and extract the
    pieces.  Without a tokenizer the char budget is calibrated from the
    first extract's own chars/token; every piece is verified by its extract
    response, so the guarantee is exact, only the cut points are
    approximate."""
    record = extract_fn("user", doc_text, ratio)
    length = int(record.get("original_seq_len") or 0)
    if length <= max_doc_length or depth >= 6 or len(doc_text) < 8:
        return [(doc_text, record)]
    chars_per_token = max(1.0, len(doc_text) / max(1, length))
    budget = max(64, int(max_doc_length * chars_per_token * 0.9))
    pieces: List[str] = []
    current = ""
    for unit in _split_lines_keep(doc_text):
        if current and len(current) + len(unit) > budget:
            pieces.append(current)
            current = ""
        if len(unit) > budget:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(unit), budget):
                pieces.append(unit[start:start + budget])
            continue
        current += unit
    if current:
        pieces.append(current)
    if len(pieces) <= 1:  # cannot split further at line level: hard halves
        half = len(doc_text) // 2
        pieces = [doc_text[:half], doc_text[half:]]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for piece in pieces:
        if not piece.strip():
            continue
        out.extend(_fit_doc(piece, ratio, extract_fn, max_doc_length, depth + 1))
    return out


def _select_docs(docs: List[Any], max_doc_num: int) -> Tuple[List[Any], int]:
    """train_data_multiturn._select_history(policy="tail"): keep doc 0 (the
    session anchor) plus the last max_doc_num-1 docs; the rest are DROPPED,
    the model never sees them (same as training and the D-line harness)."""
    if max_doc_num <= 0 or len(docs) <= max_doc_num:
        return list(docs), 0
    if max_doc_num == 1:
        return list(docs[-1:]), len(docs) - 1
    kept = [docs[0]] + list(docs[-(max_doc_num - 1):])
    return kept, len(docs) - len(kept)


def _sorted_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_keys(v) for v in value]
    return value


def _canon_calls(calls: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Canonical tool calls: name + arguments JSON with recursively sorted keys.

    The benchmark clients re-serialize model arguments when echoing history,
    so key ORDER must not affect comparison or fingerprints."""
    canon = []
    for call in calls or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = function.get("arguments") or {}
        canon.append({"name": function.get("name"),
                      "arguments": _sorted_keys(arguments)})
    return canon


def _canonical_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Role + content + canonical tool_calls, dropping transport-only keys."""
    out = []
    for message in messages:
        role = message.get("role") or "user"
        content = message.get("content")
        content = content if isinstance(content, str) else json.dumps(
            content or "", ensure_ascii=False, sort_keys=True)
        item: Dict[str, Any] = {"role": role, "content": content}
        calls = message.get("tool_calls")
        if calls:
            item["tool_calls"] = _canon_calls(calls)
        out.append(item)
    return out


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def messages_fingerprint(messages: List[Dict[str, Any]]) -> str:
    """Stable id of the exact conversation state (pre-assembly, raw)."""
    return _digest(_canonical_messages(messages))


def conversation_id(messages: List[Dict[str, Any]]) -> str:
    """Stable per-conversation id: system head + first TWO non-system
    messages.  Two, not one: agent-first benchmarks (tau2 non-solo) open
    EVERY task with the same assistant greeting, so a one-message key
    collapsed all 50 tasks onto one id — cross-contaminating ACON rolling
    state under --max-concurrency and making RecoverState repair "once per
    BENCHMARK" instead of once per conversation (2026-09-04 audit).  The
    second message is the task instruction (distinct per task); the id
    shifts once when a conversation grows past its first message — a
    benign one-time state reset, no compression happens that early."""
    canon = _canonical_messages(messages)
    head = canon[0] if canon else {}
    nonsystem = [m for m in canon if m.get("role") != "system"][:2]
    return _digest([head] + nonsystem)


def action_canonical(message: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical form of a RESPONSE message: tool calls (sorted keys) + text."""
    return {
        "tool_calls": _canon_calls(message.get("tool_calls")),
        "text": (message.get("content") or "").strip(),
    }


# ---- oracle-recover decision layer (pure; unit-tested without HTTP) ----

class RecoverState:
    """Per-proxy recover bookkeeping, keyed by conversation id.

    ``reference`` maps a message fingerprint to the reference run's action
    at that state.  Divergence logic: before the first divergence both runs
    share identical raw message lists (greedy decoding, same inputs), so the
    fingerprint lookup hits; after a one-shot repair the regenerated action
    equals the reference action, so tracking continues and a later mismatch
    is a genuine re-divergence.
    """

    def __init__(self, reference: Dict[str, Dict[str, Any]]):
        self.reference = reference
        self.repaired: Set[str] = set()
        self.divergence_step: Dict[str, int] = {}
        self.re_diverged: Set[str] = set()
        self.tracking_lost: Set[str] = set()

    def check(self, conv: str, fingerprint: str, action: Dict[str, Any],
              turn: int) -> Dict[str, Any]:
        flags: Dict[str, Any] = {
            "match": None, "diverged_now": False,
            "divergence_step": None, "re_diverged": False, "tracking_lost": False,
        }
        if conv in self.re_diverged or conv in self.tracking_lost:
            return flags
        ref = self.reference.get(fingerprint)
        if ref is None:
            # unknown state: only meaningful after a repair (the repair made
            # the conversation leave the reference track) — else it is a
            # request class the reference run never saw
            if conv in self.repaired:
                self.tracking_lost.add(conv)
                flags["tracking_lost"] = True
            return flags
        if action == ref.get("action"):
            flags["match"] = True
            return flags
        flags["match"] = False
        if conv in self.repaired:
            self.re_diverged.add(conv)
            flags["re_diverged"] = True
            return flags
        self.divergence_step[conv] = int(ref.get("turn") or turn)
        flags["diverged_now"] = True
        flags["divergence_step"] = self.divergence_step[conv]
        return flags

    def should_recover(self, conv: str, flags: Dict[str, Any]) -> bool:
        """One whole-prefix repair per conversation (docs/hybrid_spec.md)."""
        return bool(flags.get("diverged_now")) and conv not in self.repaired


def load_reference(path: str) -> Dict[str, Dict[str, Any]]:
    reference: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("fp"):
                # later rows win (deterministic reruns of the same state)
                reference[str(row["fp"])] = row
    return reference


# full-mode pseudo-arm for the recover re-send: identical payload, history
# assembled raw — the reference KV regime of whichever backend is active.
FULL_ASSEMBLY = Arm(name="full_recover_assembly", compress_history=False)


# training dialect (python/train/train_data_multiturn.py): tool messages
# are rendered as bare user messages, and every sample carries a system
# prompt.  The raw path must match (audit: the mismatch is arm-invariant
# and a direct candidate for the tool-call-in-prose failures).
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _stringify_content(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _assemble(messages: List[Dict[str, Any]], arm: Arm, timeout: int = 600):
    """Return (out_messages, counts).

    ``counts`` carries the message-class breakdown (system/hybrid-tail/
    current kept raw vs compressed) plus the compressed-token ledger and
    the per-doc extract records (message index, role, record) in
    conversation order — the proxy-side ledger that repair planning and
    the logical-token cost column are computed from.  Raw token counts are
    NOT estimated here; physical numbers come from the backend's response.
    """
    if arm.native_messages:
        out = [dict(message) for message in messages]
        return out, {
            "system_raw": sum(m.get("role") == "system" for m in messages),
            "history_raw": 0, "current_raw": len(messages), "compressed": 0,
            "gist_tokens": 0, "original_tokens": 0, "n_gist_messages": 0,
            "compressed_records": [], "doc_packing": "native", "n_docs": 0,
            "dropped_docs": 0, "current_start_out_index": len(out),
            # Native/full requests do not identify a packed-history candidate
            # set, so these are intentionally N/A rather than false zeros.
            "history_packed_original_tokens": None,
            "history_dropped_original_tokens": None,
            "history_packed_candidate_doc_count": None,
            "history_retained_fraction": None,
        }
    # raw-path training dialect: tool -> bare user message (_normal_chat_message:
    # {"role": "user", "content": str}); a missing system prompt gets the
    # training default injected
    source_shift = int(not any(m.get("role") == "system" for m in messages))
    messages = [
        ({"role": "user", "content": _stringify_content(m)}
         if m.get("role") == "tool" else dict(m))
        for m in messages
    ]
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    cutoff = _history_cutoff(messages)
    out: List[Dict[str, Any]] = []
    gist_tokens = 0
    original_tokens = 0
    n_gist = 0
    message_counts = {"system_raw": 0, "history_raw": 0, "current_raw": 0,
                      "compressed": 0}
    compressed_records: List[Dict[str, Any]] = []
    track_fragments = bool(getattr(MEMORY_RUNTIME, "always_compress", False)) and arm.compress_history
    packing_fragments = []
    packing = DOC_PACKING if arm.compress_history else "message"
    dropped_docs = 0
    n_docs = 0
    # ``original_tokens`` remains the selected-history ledger for backwards
    # compatibility.  Turn packing can discard fitted candidates after
    # extraction, so keep an explicit pre-selection denominator alongside it.
    history_packed_original_tokens = 0 if arm.compress_history else None
    history_dropped_original_tokens = 0 if arm.compress_history else None
    history_packed_candidate_doc_count = 0 if arm.compress_history else None
    history_retained_fraction = None

    def _keep_raw(i: int, role: str) -> bool:
        return (
            not arm.compress_history
            or not (i < cutoff)
            or role == "system"
            or bool(arm.hybrid_top_k and i >= cutoff - arm.hybrid_top_k)
        )

    def _emit_raw(i: int, message: Dict[str, Any]) -> None:
        role = message.get("role") or "user"
        raw = dict(message)
        # training-dialect rendering applies to RAW assistant tool_calls
        # turns as well: backends without server-side normalization
        # (sglang) would otherwise feed the chat template's native
        # tool_calls branch, a surface the model was never trained on.
        if role == "assistant" and message.get("tool_calls"):
            raw["content"] = _render_action_dialect(message)
            raw.pop("tool_calls", None)
        out.append(raw)
        if role == "system":
            message_counts["system_raw"] += 1
        elif i < cutoff:
            message_counts["history_raw"] += 1
        else:
            message_counts["current_raw"] += 1

    def _emit_doc(doc_role: str, doc_text: str, record: Dict[str, Any],
                  source_indices: List[int], fragment_id=None) -> None:
        nonlocal gist_tokens, original_tokens, n_gist, n_docs
        gist_tokens += int(record.get("gist_len") or 0)
        original_tokens += int(record.get("original_seq_len") or 0)
        n_gist += 1
        n_docs += 1
        message_counts["compressed"] += 1
        compressed = {
            "role": doc_role,
            "content": doc_text,
            "c2kv_key_hash": record["key_hash"],
            # lets the server re-extract on cache miss (e.g. after a restart)
            "c2kv_ratio": arm.ratio,
        }
        compressed_records.append({
            "message_index": source_indices[0] if source_indices else -1,
            "source_indices": list(source_indices),
            "out_index": len(out),
            "role": doc_role, "content": doc_text, "record": record,
            **({"packing_fragment_id": fragment_id} if track_fragments else {}),
        })
        out.append(compressed)

    if packing == "turn":
        # TRAINING format: normalize, pack per turn, split to fit, tail-select.
        compressible = [
            (i, m) for i, m in enumerate(messages)
            if not _keep_raw(i, m.get("role") or "user")
        ]
        docs: List[Tuple[str, List[int]]] = []
        if compressible:
            normalized = []
            for i, m in compressible:
                item = _normalize_history_message(m)
                if item is not None:
                    normalized.append((i, item))
            for doc in _turn_docs(normalized):
                with extraction_sources([i - source_shift for i in doc["source_indices"]]):
                    for text, record in _fit_doc(
                        doc["content"], arm.ratio,
                        lambda role, text, ratio: _extract(role, text, ratio, timeout),
                        MAX_DOC_LENGTH,
                    ):
                        fragment_id = len(docs)
                        if track_fragments:
                            packing_fragments.append({
                                "fragment_id": fragment_id,
                                "source_indices": [i - source_shift for i in doc["source_indices"]],
                                "encoder_input_tokens": int(record.get("original_seq_len") or 0),
                            })
                        docs.append((text, record, doc["source_indices"], fragment_id))
            # Fit/extract precedes tail selection.  Count the already-returned
            # record lengths here; never re-tokenize or re-extract merely for
            # accounting.
            history_packed_candidate_doc_count = len(docs)
            history_packed_original_tokens = sum(
                int(record.get("original_seq_len") or 0)
                for _, record, _, _ in docs)
            docs, dropped_docs = _select_docs(docs, MAX_DOC_NUM)
            selected_original_tokens = sum(
                int(record.get("original_seq_len") or 0)
                for _, record, _, _ in docs)
            history_dropped_original_tokens = (
                history_packed_original_tokens - selected_original_tokens)
            if history_packed_original_tokens:
                history_retained_fraction = (
                    selected_original_tokens / history_packed_original_tokens)
        first_index = compressible[0][0] if compressible else None
        compressible_set = {i for i, _ in compressible}
        for i, message in enumerate(messages):
            if i == first_index:
                for text, record, sources, fragment_id in docs:
                    _emit_doc("user", text, record, sources, fragment_id)
                continue
            if i in compressible_set:
                continue
            _emit_raw(i, message)
    else:
        # LEGACY bench format: one doc per message with its own role.
        for i, message in enumerate(messages):
            role = message.get("role") or "user"
            if _keep_raw(i, role):
                _emit_raw(i, message)
                continue
            content = message.get("content")
            content = content if isinstance(content, str) else json.dumps(content or "")
            if role == "assistant" and message.get("tool_calls"):
                # see _render_action_dialect: never extract the bare (null)
                # content of a tool-call turn
                content = _render_action_dialect(message)
            with extraction_sources([i - source_shift]):
                record = _extract(role, content, arm.ratio, timeout)
            fragment_id = len(packing_fragments)
            if track_fragments:
                packing_fragments.append({
                    "fragment_id": fragment_id, "source_indices": [i - source_shift],
                    "encoder_input_tokens": int(record.get("original_seq_len") or 0),
                })
            _emit_doc(role, content, record, [i], fragment_id)
        # Message packing has no post-extract selection: its candidate and
        # retained sets are the same selected-history ledger.
        if arm.compress_history:
            history_packed_original_tokens = original_tokens
            history_dropped_original_tokens = 0
            history_packed_candidate_doc_count = n_docs
            if history_packed_original_tokens:
                history_retained_fraction = 1.0
    counts = dict(message_counts)
    counts["gist_tokens"] = gist_tokens
    counts["original_tokens"] = original_tokens
    counts["n_gist_messages"] = n_gist
    counts["compressed_records"] = compressed_records
    if track_fragments:
        counts["history_packing_fragments"] = packing_fragments
    counts["doc_packing"] = packing
    counts["n_docs"] = n_docs
    counts["dropped_docs"] = dropped_docs
    counts["history_packed_original_tokens"] = history_packed_original_tokens
    counts["history_dropped_original_tokens"] = history_dropped_original_tokens
    counts["history_packed_candidate_doc_count"] = history_packed_candidate_doc_count
    counts["history_retained_fraction"] = history_retained_fraction
    # index in `out` where the current (raw) block starts: repair-only
    # messages for append placements are inserted right before it
    counts["current_start_out_index"] = len(out) - message_counts["current_raw"]
    return out, counts


def _system_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        (m.get("content") or "") for m in messages if m.get("role") == "system")


def _strip_c2kv_fields(message: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in message.items() if not str(k).startswith("c2kv_")}


def _history_kv_context(out_messages: List[Dict[str, Any]],
                        counts: Dict[str, Any],
                        arm: Arm) -> Optional[Dict[str, Any]]:
    """Proxy-side history split for a history-KV eviction arm.

    The upstream client splits at the latest user query and compresses
    everything before it; here the split is the bench's own
    ``_history_cutoff`` (``counts["current_start_out_index"]``), so a
    history-KV row is comparable with every other arm's row on the same
    request.  System messages stay raw and OUTSIDE the compressed span (bench
    rule), which the upstream client does not do — see README.

    ``history_text`` is the SAME text the c2kv arm gists: the completed
    history normalized into the training dialect and packed into turn docs
    (``_normalize_history_message`` + ``_turn_docs``), joined into one span so
    a single budget covers the whole history exactly as upstream applies it.
    ``history_message_count`` is instead a count of LEADING assembled
    messages, which is what the physical-eviction path sends to the server
    (system message included: the server's own range resolution starts at
    message 0).
    """
    spec = history_kv_spec(arm)
    if spec is None:
        return None
    cutoff = int(counts.get("current_start_out_index") or 0)
    system_parts: List[str] = []
    history_indices: List[int] = []
    indexed: List[Tuple[int, Dict[str, Any]]] = []
    for index, message in enumerate(out_messages[:cutoff]):
        if (message.get("role") or "user") == "system":
            system_parts.append(str(message.get("content") or ""))
            continue
        item = _normalize_history_message(message)
        if item is None:
            continue
        history_indices.append(index)
        indexed.append((index, item))
    docs = _turn_docs(indexed)
    history_text = "\n\n".join(doc["content"] for doc in docs if doc["content"])
    return {
        "spec": spec,
        "method": spec["method"],
        "backend": spec["backend"],
        "system_text": "\n".join(part for part in system_parts if part),
        "history_text": history_text,
        "history_out_indices": history_indices,
        "history_message_count": cutoff,
        "current_start_out_index": cutoff,
        "n_history_messages": len(history_indices),
        "n_history_docs": len(docs),
    }


def _kv_reuse_context(out_messages: List[Dict[str, Any]],
                      counts: Dict[str, Any],
                      arm: Arm) -> Optional[Dict[str, Any]]:
    """Proxy-side history split for a KV-reuse (CacheBlend) arm.

    Same split and the same turn-doc packing as ``_history_kv_context`` (so
    a CacheBlend row is comparable with every other arm's row on the same
    request), but the docs are kept SEPARATE: each turn doc is one CacheBlend
    chunk, sent to the server as its own message of the multi-message
    ``repair_extract`` form (``backends/sglang.SglangBackend.kv_reuse_extract``).
    System messages stay raw and outside the reused span (bench rule); the
    artifact caches the system prompt as chunk 0 -- README "CacheBlend arms".
    """
    spec = kv_reuse_spec(arm)
    if spec is None:
        return None
    cutoff = int(counts.get("current_start_out_index") or 0)
    system_parts: List[str] = []
    history_indices: List[int] = []
    indexed: List[Tuple[int, Dict[str, Any]]] = []
    for index, message in enumerate(out_messages[:cutoff]):
        if (message.get("role") or "user") == "system":
            system_parts.append(str(message.get("content") or ""))
            continue
        item = _normalize_history_message(message)
        if item is None:
            continue
        history_indices.append(index)
        indexed.append((index, item))
    docs = _turn_docs(indexed)
    history_docs = [{"role": doc["role"], "content": doc["content"]}
                    for doc in docs if doc["content"]]
    return {
        "spec": spec,
        "method": spec["method"],
        "chunking": spec["chunking"],
        "system_text": "\n".join(part for part in system_parts if part),
        "history_docs": history_docs,
        "history_out_indices": history_indices,
        "current_start_out_index": cutoff,
        "n_history_messages": len(history_indices),
        "n_history_docs": len(history_docs),
    }


def _textarm_compress(payload: Dict[str, Any], meter=None) -> str:
    """One compressor call for a text-level baseline arm.  Validates the
    finish reason and non-empty content — HTTP-200-with-error-body and
    abort finishes are FAILURES (TextarmCompressorError), never empty
    summaries that would get cached for the rest of the conversation.
    "length" is a NORMAL finish: the HiAgent summarizer decodes with
    max_tokens~100 and works BY truncation.  ``meter`` (optional) receives
    the response usage block for cost accounting."""
    data = _post_json("/v1/chat/completions", payload, 600)
    try:
        choice = (data.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content") or ""
    except (AttributeError, IndexError):
        raise textarms.TextarmCompressorError(f"bad compressor response: {data!r:.400}")
    if meter is not None:
        meter(data.get("usage") or {})
    if finish not in ("stop", "length") or not str(content).strip():
        raise textarms.TextarmCompressorError(
            f"compressor call failed: finish_reason={finish!r} "
            f"content_len={len(str(content))}")
    return str(content)


def _apply_text_arm(payload: Dict[str, Any], arm, conv: str,
                    retrieve_subgoals: Optional[List[int]] = None
                    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rewrite history per the arm's text policy (textarms.py) BEFORE
    assembly; the arm is full-mode downstream.  Compressor calls go
    straight to the upstream endpoint (no arm semantics, no recursion
    through the chat handler).  Compressor tokens/wall-time accumulate
    into stats["compressor_usage"] (the fairness ruling: text baselines'
    extra LLM calls must appear in the cost columns)."""
    messages = payload.get("messages") or []
    model = payload.get("model") or "c2kv-agent"
    usage_acc = {"calls": 0, "prompt_tokens": 0,
                 "completion_tokens": 0, "wall_sec": 0.0}

    def _meter(u: Dict[str, Any]) -> None:
        usage_acc["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        usage_acc["completion_tokens"] += int(u.get("completion_tokens") or 0)

    def compress(pl: Dict[str, Any]) -> str:
        t0 = time.perf_counter()
        out = _textarm_compress(pl, meter=_meter)
        usage_acc["calls"] += 1
        usage_acc["wall_sec"] += time.perf_counter() - t0
        return out

    if arm.text_policy in ("hiagent", "hiagent_summary", "hiagent_full"):
        out, stats = textarms.hiagent_transform(
            messages, compress, _render_action_dialect, model=model,
            default_system=DEFAULT_SYSTEM_PROMPT,
            variant="full" if arm.text_policy == "hiagent_full" else "summary",
            retrieve_subgoals=retrieve_subgoals)
    else:
        parts = arm.text_policy.split("_", 2)
        if len(parts) < 2 or parts[0] != "acon" or parts[1] not in ("hist", "obs"):
            raise ValueError(f"unsupported text policy {arm.text_policy!r}")
        mode = parts[1]
        guideline = parts[2] if len(parts) > 2 else "base"
        out, stats = textarms.acon_transform(
            messages, compress, _render_action_dialect, conv,
            mode=mode, model=model, guideline=guideline)
    stats["compressor_usage"] = usage_acc
    staged = dict(payload)
    staged["messages"] = out
    if arm.text_policy == "hiagent_full":
        tools = list(payload.get("tools") or [])
        if any((tool.get("function") or {}).get("name") == textarms.HIAGENT_RETRIEVE_TOOL_NAME
               for tool in tools):
            raise ValueError("benchmark tool name collides with HiAgent's internal retrieval tool")
        staged["tools"] = tools + [textarms.hiagent_retrieval_tool()]
    return staged, stats


def _hiagent_retrieval_loop(original_payload, arm, conv, data, stats, send):
    """Consume context-retrieval calls internally, charging every extra call.

    Only a final environment action is returned to the official harness.
    The bounded loop fails explicitly if the model keeps retrieving without
    advancing; no meta-tool can leak into the benchmark's tool executor.
    """
    retrieved = set()
    retrieval_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(5):
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        ids = textarms.hiagent_retrieval_request(message)
        if ids is None:
            stats["retrieval_usage"] = retrieval_usage
            return data
        if attempt == 4:
            raise ValueError("HiAgent exceeded four internal trajectory retrieval rounds")
        if any((call.get("function") or {}).get("name") != textarms.HIAGENT_RETRIEVE_TOOL_NAME
               for call in message.get("tool_calls") or []):
            raise ValueError("HiAgent mixed internal retrieval and environment actions in one response")
        if set(ids) <= retrieved:
            raise ValueError("HiAgent requested an already revealed trajectory without advancing")
        retrieved.update(ids)
        usage = data.get("usage") or {}
        retrieval_usage["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            retrieval_usage[key] += int(usage.get(key) or 0)
        staged, updated = _apply_text_arm(original_payload, arm, conv, sorted(retrieved))
        if updated.get("invalid_retrieval_subgoals"):
            raise ValueError(f"HiAgent requested nonexistent completed subgoals: {updated['invalid_retrieval_subgoals']}")
        for key, value in (updated.get("compressor_usage") or {}).items():
            stats.setdefault("compressor_usage", {}).setdefault(key, 0)
            stats["compressor_usage"][key] += value
        stats["n_compressor_calls"] += int(updated.get("n_compressor_calls") or 0)
        stats.update({key: value for key, value in updated.items()
                      if key not in ("compressor_usage", "n_compressor_calls")})
        data = send(staged)


def plan_repair(messages: List[Dict[str, Any]], arm: Arm,
                counts: Dict[str, Any],
                tools: Optional[List[Dict[str, Any]]] = None,
                out_messages: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Resolve an arm's repair policy against the assembled request.

    The raw KV of the target doc is extracted with the FULL-CONTEXT form of
    /v1/c2kv/repair_extract: the server renders ``out_messages[:target+1]``
    (system messages + compressed docs as plain user messages, with the
    request's tools) exactly like a chat request and captures the target
    doc's K/V inside that context (docs/c2kv_semantics.md, "Raw KV").  The
    returned ``position_start`` is the doc's absolute position in that
    rendering; the server's gist ledger places the doc's gist at
    ``P + Σ original_seq_len(docs before)``, and the two must agree when the
    per-message rendering is additive (Qwen3 template).  The proxy records
    its own ledger expectation for the frame check in the request log.

    ``frame_delta`` is a MEASUREMENT of that additivity, not an identity, and
    it is not always computable: the prologue length ``P`` is only observable
    through a system-message extract.  That extract is taken from the
    ASSEMBLED list (``out_messages``), not from the caller's payload —
    ``_assemble`` injects ``DEFAULT_SYSTEM_PROMPT`` when the client sends no
    system message (BFCL FC sends none), and the server renders THAT, so
    measuring the caller's pre-assembly list left ``expected_offset = None``
    and the frame check unmeasured on every BFCL repair-arm row.  A list
    carrying more than one system message is still not measurable: the server
    renders them as separate blocks, so joining them would measure a prologue
    that is never rendered (``not_measured_multi_system``).
    ``frame_delta_status`` says which case a row is in — never read a null
    delta as a zero delta.
    """
    if not arm.repair or not getattr(BACKEND, "needs_repair_plan", False):
        return None
    policy = str((arm.repair or {}).get("policy") or "first")
    placement = str((arm.repair or {}).get("placement") or "append_keep_ledger")
    parsed = repair_policy.parse_policy(policy)
    records = counts.get("compressed_records") or []
    if not records:
        # no history compressed yet: a legitimate no-op (the FIRST request
        # of every repair-arm session has nothing to repair)
        return None
    doc_counts = [1] * len(records)  # whole docs (turn docs or messages)
    doc_index, _first_chunk, _span_len = repair_policy.span_selection(
        doc_counts, parsed["kind"], parsed["index"])
    target = records[doc_index]
    target_out_index = int(target.get("out_index", target["message_index"]))
    if out_messages is None:
        raise ValueError("plan_repair needs the assembled out_messages")
    context = [_strip_c2kv_fields(m) for m in out_messages[:target_out_index + 1]]
    span = BACKEND.repair_extract_messages(
        messages=context, target_index=target_out_index, tools=tools,
        source_doc_index=doc_index)
    # proxy-side ledger expectation (frame check): system block incl. tools
    # + Σ original_seq_len of the compressed docs before the target.  The
    # prologue is measured on the ASSEMBLED list, which is what the server
    # renders: _assemble injects DEFAULT_SYSTEM_PROMPT when the client sends
    # none, so a BFCL FC request DOES have a measurable prologue (measuring
    # the caller's list left it None and no arm ever ran the frame check).
    expected_offset: Optional[int] = None
    frame_delta_status: Optional[str] = None
    system_messages = [m for m in out_messages if m.get("role") == "system"]
    if len(system_messages) > 1:
        # the server renders each system message as its own block; a joined
        # extract would measure a prologue that is never rendered
        frame_delta_status = "not_measured_multi_system"
    else:
        system = _system_text(out_messages)
        if system:
            sys_record = _extract("system", system, arm.ratio, tools=tools)
            expected_offset = int(sys_record.get("original_seq_len") or 0)
            for record in records[:doc_index]:
                expected_offset += int(record["record"].get("original_seq_len") or 0)
    position_start = span.get("position_start")
    frame_delta = None
    # explicit status so a null delta is never read as a passing check
    if frame_delta_status is not None:
        pass  # already decided above (multi-system)
    elif expected_offset is None:
        frame_delta_status = "not_measured_no_system"
    elif position_start is None:
        frame_delta_status = "not_measured_no_position_start"
    else:
        frame_delta = int(position_start) - int(expected_offset)
        frame_delta_status = "measured"
    return {
        "policy": policy, "placement": placement,
        "message_index": target_out_index, "target_out_index": target_out_index,
        "doc_index": doc_index,
        "current_start_out_index": int(counts.get("current_start_out_index", len(out_messages))),
        "position_offset": position_start,
        "position_start": position_start, "position_end": span.get("position_end"),
        "expected_offset": expected_offset, "frame_delta": frame_delta,
        "frame_delta_status": frame_delta_status,
        "already_rotated": bool(span.get("already_rotated", False)),
        "repair_key_hash": span.get("key_hash"),
        "repair_block_tokens": span.get("token_len"),
    }


def _repair_frame_check(plan: Optional[Dict[str, Any]],
                        normalized: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compare the repair span's RoPE position with the server's gist ledger
    (metadata.sglang_runtime.c2kv_layout, docs/c2kv_semantics.md "Position
    frames").  For append placements the span must sit exactly where the
    target doc's gist sits (position_cursor of the doc_index-th gist); for
    in_place the target's gist is not injected, so the span must start
    where the previous gist ends.

    ``ok`` is None when the check is NOT COMPUTABLE — that is not a pass.
    ``ok_reason`` names the case; the one that matters in practice is
    in_place at doc_index 0 (policy "first" on a single-doc conversation, i.e.
    the c2kv_repair_inplace arm on BFCL), where the expectation is the end of
    the prologue and the gist ledger does not contain it.  Only THIS check is
    unavailable on such a row: plan["frame_delta"] is measured independently
    (against the assembled prologue) and is carried here as
    ``frame_delta_status``."""
    if not plan:
        return None
    layout = ((normalized.get("cost") or {}).get("c2kv_layout")) or []
    gists = [e for e in layout if e.get("kind") == "gist"]
    repairs = [e for e in layout if e.get("kind") == "repair"]
    result: Dict[str, Any] = {
        "placement": plan.get("placement"),
        "position_start": plan.get("position_start"),
        "n_gist_injections": len(gists),
        "n_repair_injections": len(repairs),
        "ok": None,
        "ok_reason": "not_measured_no_layout" if not layout else "not_measured",
        "frame_delta_status": plan.get("frame_delta_status"),
    }
    k = int(plan.get("doc_index", -1))
    expected = None
    if plan.get("placement") == "in_place":
        if k == 0:
            # first doc: the expectation is the end of the prologue, which the
            # gist ledger does not carry -- not derivable, and NOT a pass
            result["ok_reason"] = "not_measured_in_place_first_doc"
        elif 0 < k <= len(gists):
            prev = gists[k - 1]
            expected = int(prev["position_cursor"]) + int(prev["original_seq_len"])
        else:
            result["ok_reason"] = "not_measured_doc_index_outside_layout"
    elif 0 <= k < len(gists):
        expected = int(gists[k]["position_cursor"])
    else:
        result["ok_reason"] = "not_measured_doc_index_outside_layout"
    if expected is not None and plan.get("position_start") is not None:
        result["expected_from_layout"] = expected
        result["ok"] = int(plan["position_start"]) == expected
        result["ok_reason"] = "measured"
    elif expected is not None:
        result["ok_reason"] = "not_measured_no_position_start"
    if repairs:
        result["server_placement"] = repairs[0].get("placement")
        result["server_position_start"] = repairs[0].get("position_start")
    return result


class ProxyState:
    """Process-wide proxy state (backend, recover config, reference log,
    history-KV streaming sessions)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recover: Optional[RecoverState] = None
        self.reference_log_path: str = ""
        # conversation_id -> server streaming-session id (physical-eviction
        # history-KV arms only)
        self.history_sessions: Dict[str, str] = {}
        self.gold_choices = {}
        self.gold_plans = {}


STATE = ProxyState()


def _history_session_id(conv: str) -> str:
    """Streaming-session id for ``conv``, opened once on the server.

    The physical-eviction baselines only mean anything when the compacted KV
    survives across turns, which on this server needs a streaming session
    (``--enable-streaming-session``) whose id rides on every chat request of
    the conversation.  The upstream client holds that id per BFCL sample and
    closes it at the end; a stateless HTTP proxy has no end-of-conversation
    signal, so:

    * the id is keyed by ``proxy.conversation_id``, which by construction
      SHIFTS ONCE after a conversation grows past its first message (see
      conversation_id) — such a conversation opens two sessions, the second
      starting from an empty prefix;
    * sessions are never closed, so they live until the server restarts.

    Both limitations are documented in README "History-KV eviction arms".
    """
    with STATE.lock:
        session_id = STATE.history_sessions.get(conv)
        if session_id:
            return session_id
    session_id = f"c2kv-bench-history-{conv[:16]}-{uuid.uuid4().hex}"
    BACKEND.open_history_session(session_id)
    with STATE.lock:
        STATE.history_sessions.setdefault(conv, session_id)
        return STATE.history_sessions[conv]


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _is_chat(self) -> bool:
        return _is_chat_path(self.path)

    def do_POST(self):
        self.request_id = uuid.uuid4().hex
        token = _ATTEMPT_REQUEST.set({"request_id": self.request_id, "eval_context": {}})
        try:
            self._do_post_scoped()
        finally:
            _ATTEMPT_REQUEST.reset(token)

    def _do_post_scoped(self):
        with GENERATION_BUDGET.request_scope() as budget_record:
            self.generation_budget_record = budget_record
            self._generation_budget_final = None
            self._generation_budget_finalized = False
            with EXTRACTION_BUDGET.request_scope() as extraction_budget_record:
                self.extraction_budget_record = extraction_budget_record
                self._extraction_budget_final = None
                self._extraction_budget_finalized = False
                with capture_extractions(
                    enabled=MEMORY_RUNTIME is not None or CAPTURE_REQUEST_VIEWS
                ) as trace:
                    self.extraction_trace = trace
                    self.forwarded_gist_keys = []
                    self._do_post()

    def _predict_source_needs(self, parent_payload, messages, expected_prompt_tokens, *, tools=None):
        """Issue one budgeted auxiliary generation whose output is never executed."""
        staged = {key: parent_payload[key] for key in
                  ("model", "temperature", "seed", "top_p", "top_k") if key in parent_payload}
        staged.update(messages=messages, stream=False,
                      max_completion_tokens=MEMORY_RUNTIME.predictor_completion_token_cap,
                      c2kv_use_gist_projection=False)
        if tools is not None:
            staged["tools"] = tools
        out_payload = BACKEND.prepare_chat(staged, get_arm("full"), None)
        attempts_before = self.generation_budget_record.attempt_count
        lengths = (len(self.forwarded_sampling), len(self.forwarded_gist_keys),
                   len(self.forwarded_request_views))
        self.forwarded_sampling.append(_sampling_fields(out_payload))
        self.forwarded_gist_keys.append([])
        if CAPTURE_REQUEST_VIEWS:
            self.forwarded_request_views.append(_captured_request_view(out_payload))
        record = {"phase": "source_prediction", "status": "started", "discarded": False,
                  "action_candidate": False, "submitted_to_executor": False,
                  "backend_verified": False, "forwarded_request_index": lengths[0]}
        started = time.perf_counter()
        try:
            data = _post_json(self.path, out_payload, 600, retries=0)
            normalized = BACKEND.normalize_response(data)
            record.update(status="completed", usage=normalized.get("usage"),
                          cost=normalized.get("cost"), finish_reason=normalized.get("finish_reason"))
            verification = {"memory_runtime": {
                "bytes_per_kv_token": MEMORY_RUNTIME.bytes_per_kv_token,
                "total_raw_prompt_tokens": expected_prompt_tokens}}
            _verify_memory_runtime_kv_bytes(verification, normalized)
            record["backend_verified"] = True
            record["memory_runtime"] = verification["memory_runtime"]
            if CAPTURE_REQUEST_VIEWS:
                record["response_view"] = {key: normalized.get(key) for key in ("content", "tool_calls")}
            return normalized
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            if self.generation_budget_record.attempt_count > attempts_before:
                record["wall_sec"] = time.perf_counter() - started
                self.generation_records.append(record)
            else:
                del self.forwarded_sampling[lengths[0]:]
                del self.forwarded_gist_keys[lengths[1]:]
                del self.forwarded_request_views[lengths[2]:]

    def _do_post(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if not self._is_chat():
            self._passthrough(raw)
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return
        assert ARM is not None and BACKEND is not None
        runtime_payload_conflicts = (
            _memory_runtime_payload_conflicts(payload)
            if MEMORY_RUNTIME is not None else [])
        eval_context = payload.pop("c2kv_eval_context", None)
        oracle = payload.pop("c2kv_oracle", None)
        self.eval_context = eval_context if isinstance(eval_context, dict) else {}
        _ATTEMPT_REQUEST.get()["eval_context"] = self.eval_context
        request_budget = GENERATION_BUDGET.current_request()
        task_id = self.eval_context.get("task_id")
        if (GENERATION_BUDGET.per_task_limit is not None and request_budget is not None
                and isinstance(task_id, str) and task_id):
            # Bind before packing/admission, which may terminate the task
            # without issuing a generation request.
            request_budget.task_id = task_id
            request_budget.task_consumed_before = GENERATION_BUDGET.consumed_for_task(task_id)
        self.forwarded_sampling = []
        self.forwarded_request_views = []
        self.generation_records = None
        source_needs_active = getattr(MEMORY_RUNTIME, "source_needs_strategy", None) is not None
        if source_needs_active:
            self.generation_records = []
        start = time.perf_counter()
        messages = payload.get("messages") or []
        fingerprint = messages_fingerprint(messages)
        conv = conversation_id(messages)
        turn = len(messages)
        text_stats: Optional[Dict[str, Any]] = None
        original_payload = payload
        counts = _memory_runtime_error_counts()
        prepared_exact = None
        try:
            if MEMORY_RUNTIME is not None:
                _check_memory_runtime_fatal()
                _validate_memory_runtime_arm(MEMORY_RUNTIME, ARM)
                if runtime_payload_conflicts:
                    raise MemoryRuntimeError(
                        "memory runtime cannot be combined with client controls: "
                        + ", ".join(runtime_payload_conflicts))
            if getattr(ARM, "text_policy", None):
                payload, text_stats = _apply_text_arm(payload, ARM, conv)
                messages = payload["messages"]
            if getattr(MEMORY_RUNTIME, "supports_exact_recovery", False):
                self.generation_records = []
                messages_out, counts, prepared_exact = _prepare_exact_memory_input(
                    messages, self.eval_context, payload.get("tools"))
            else:
                messages_out, counts = _prepare_memory_input(
                    messages, self.eval_context, payload.get("tools"),
                    source_predictor=(lambda selected_messages, expected_tokens, **options:
                        self._predict_source_needs(payload, selected_messages, expected_tokens, **options))
                        if source_needs_active else None)
            if text_stats is not None:
                counts["textarm"] = text_stats
            repair_plan = plan_repair(messages, ARM, counts,
                                     tools=payload.get("tools"),
                                     out_messages=messages_out)
            if oracle:
                repair_plan = plan_gold_repair(
                    messages, ARM, counts, oracle, payload.get("tools"), messages_out)
            history_ctx = _history_kv_context(messages_out, counts, ARM)
            if history_ctx is not None:
                if history_ctx["spec"]["persistent_session"]:
                    history_ctx["session_id"] = _history_session_id(conv)
                counts["history_kv"] = {
                    k: history_ctx[k] for k in
                    ("method", "backend", "n_history_messages",
                     "n_history_docs", "history_message_count")}
            reuse_ctx = _kv_reuse_context(messages_out, counts, ARM)
            if reuse_ctx is not None:
                counts["kv_reuse"] = {
                    k: reuse_ctx[k] for k in
                    ("method", "chunking", "n_history_messages",
                     "n_history_docs")}
        except (RuntimeError, ValueError, URLError, OSError, UpstreamError,
                BackendError) as error:
            kind = getattr(error, "kind",
                           "textarm_error" if ARM.text_policy else "assemble_error")
            generation_budget = self._finalize_generation_budget()
            self._log_request(payload, None, counts, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn)
            response = {"error": f"c2kv assembly failed: {error}"}
            if generation_budget is not None:
                response["generation_budget"] = generation_budget
            extraction_budget = self._finalize_extraction_budget()
            if extraction_budget is not None:
                response["extraction_budget"] = extraction_budget
            self._send_json(502, response)
            return
        assemble_sec = time.perf_counter() - start

        def send_upstream(out_messages, plan):
            # prepare_chat must SEE the assembled messages: the sglang
            # backend attaches repair hashes to the (gist-marked) target
            # message — feeding it the raw payload made every rp-arm
            # request fail "repair target has no c2kv_key_hash"
            staged = dict(payload)
            staged["messages"] = out_messages
            if QUERY_PROJECTION is not None and BACKEND.name == "sglang":
                staged["c2kv_use_gist_projection"] = QUERY_PROJECTION == "gist"
            if getattr(BACKEND, "wants_request_context", False):
                # only backends that asked for it (base.Backend
                # .wants_request_context); hfserver keeps its 3-arg signature
                out_payload = BACKEND.prepare_chat(
                    staged, ARM, plan,
                    context={"conversation_id": conv, "history_kv": history_ctx,
                             "kv_reuse": reuse_ctx})
            else:
                out_payload = BACKEND.prepare_chat(staged, ARM, plan)
            self.forwarded_sampling.append(_sampling_fields(out_payload))
            self.forwarded_gist_keys.append([
                message["c2kv_key_hash"] for message in out_payload.get("messages", [])
                if message.get("c2kv_key_hash")])
            if CAPTURE_REQUEST_VIEWS:
                self.forwarded_request_views.append(_captured_request_view(out_payload))
            return _post_json(
                self.path, out_payload, 600,
                retries=(0 if MEMORY_RUNTIME is not None or NO_UPSTREAM_RETRIES
                         else 2)), out_payload

        def call_upstream(out_messages, plan):
            record = None
            if prepared_exact is not None:
                if len(self.generation_records) >= 2:
                    raise MemoryRuntimeError("Exact recovery allows at most two generation attempts")
            elif source_needs_active and any(record["phase"] == "action" for record in self.generation_records):
                raise MemoryRuntimeError("Source-needs allows one final action generation per decision")
            attempts_before = self.generation_budget_record.attempt_count
            forwarded_lengths = (
                len(self.forwarded_sampling),
                len(self.forwarded_gist_keys),
                len(self.forwarded_request_views),
            )
            upstream_started = time.perf_counter()

            def rollback_unreserved_forward():
                if self.generation_budget_record.attempt_count != attempts_before:
                    return
                sampling_len, gist_len, view_len = forwarded_lengths
                del self.forwarded_sampling[sampling_len:]
                del self.forwarded_gist_keys[gist_len:]
                del self.forwarded_request_views[view_len:]

            def record_reserved_attempt():
                nonlocal record
                if (self.generation_records is None or record is not None
                        or self.generation_budget_record.attempt_count == attempts_before):
                    return
                record = {
                    "phase": ("draft" if not self.generation_records else "regeneration")
                             if prepared_exact is not None else "action",
                    "status": "started", "discarded": False,
                    "backend_verified": False,
                    "forwarded_request_index": len(self.forwarded_sampling) - 1,
                    "memory_runtime": json.loads(json.dumps(counts["memory_runtime"])),
                }
                self.generation_records.append(record)

            try:
                data_, _ = send_upstream(out_messages, plan)
                record_reserved_attempt()
                normalized_ = BACKEND.normalize_response(data_)
                if record is not None:
                    record.update(status="completed", usage=normalized_.get("usage"),
                                  cost=normalized_.get("cost"),
                                  finish_reason=normalized_.get("finish_reason"))
                    if CAPTURE_REQUEST_VIEWS:
                        record["response_view"] = {key: normalized_.get(key)
                                                   for key in ("content", "tool_calls")}
                return data_, normalized_
            except BackendError as error:
                rollback_unreserved_forward()
                record_reserved_attempt()
                if record is not None:
                    record.update(status="failed", error_type=type(error).__name__)
                if getattr(error, "kind", "") == "cache_miss":
                    raise CacheMiss(error.detail) from error
                raise
            except Exception as error:
                rollback_unreserved_forward()
                record_reserved_attempt()
                if record is not None:
                    record.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                if record is not None:
                    record["wall_sec"] = time.perf_counter() - upstream_started

        try:
            try:
                data, normalized = call_upstream(messages_out, repair_plan)
                if ARM.text_policy == "hiagent_full":
                    def send_retrieved(staged):
                        nonlocal payload, messages_out
                        payload = staged
                        messages_out, _ = _assemble(staged["messages"], ARM)
                        return send_upstream(messages_out, None)[0]
                    data = _hiagent_retrieval_loop(
                        original_payload, ARM, conv, data, text_stats, send_retrieved)
                    normalized = BACKEND.normalize_response(data)
            except CacheMiss:
                if (MEMORY_RUNTIME is not None or NO_UPSTREAM_RETRIES
                        or GENERATION_BUDGET.enabled):
                    # The caller owns the attempt budget.  With A enabled, a
                    # transparent retry could also apply one lease decision
                    # twice and make state/cost provenance ambiguous.
                    raise
                # Pool-evicted gists and/or an evicted repair span.  Three
                # things are needed for the retry to be anything but a second
                # identical miss:
                #   1. force=True — the memoised record describes the entry
                #      the server has just dropped, so the plain _extract call
                #      that used to stand here issued NO http request at all
                #      (ExtractCache.get_or_put memoises forever and is never
                #      cleared) and the retry re-sent the same dead hash;
                #   2. refresh the hash actually carried by messages_out, in
                #      case the re-extract does not land on the same
                #      content-derived key_hash;
                #   3. re-plan the repair — BACKEND.repair_extract_messages is
                #      not memoised, but plan_repair was never re-run, so a
                #      miss on the repair key (the scheduler emits the same
                #      C2KV_CACHE_MISS prefix for c2kv_repair_*_key_hashes)
                #      survived the retry unchanged.
                for record in counts.get("compressed_records") or []:
                    fresh = _extract(record["role"], record["content"],
                                     ARM.ratio, force=True)
                    record["record"] = fresh
                    idx = record.get("out_index")
                    if (isinstance(idx, int) and 0 <= idx < len(messages_out)
                            and messages_out[idx].get("c2kv_key_hash")
                            and fresh.get("key_hash")):
                        messages_out[idx]["c2kv_key_hash"] = fresh["key_hash"]
                repair_plan = plan_repair(messages, ARM, counts,
                                          tools=payload.get("tools"),
                                          out_messages=messages_out)
                if oracle:
                    repair_plan = plan_gold_repair(
                        messages, ARM, counts, oracle, payload.get("tools"), messages_out, force=True)
                data, normalized = call_upstream(messages_out, repair_plan)
            _verify_memory_runtime_kv_bytes(counts, normalized)
            if source_needs_active:
                self.generation_records[-1]["backend_verified"] = True
                self.generation_records[-1]["memory_runtime"] = json.loads(json.dumps(counts["memory_runtime"]))
            if prepared_exact is not None:
                self.generation_records[-1]["backend_verified"] = True
                self.generation_records[-1]["memory_runtime"] = json.loads(
                    json.dumps(counts["memory_runtime"]))
                try:
                    reconsidered = MEMORY_RUNTIME.reconsider(
                        prepared_exact, normalized.get("tool_calls"))
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    raise MemoryRuntimeError(str(error), kind=getattr(error, "kind", None)) from error
                counts = reconsidered["counts"]
                if reconsidered["regenerate"]:
                    self.generation_records[-1]["discarded"] = True
                    messages_out, counts = _pin_memory_runtime_output(
                        reconsidered["messages"], counts)
                    data, normalized = call_upstream(messages_out, None)
                    _verify_memory_runtime_kv_bytes(counts, normalized)
                    self.generation_records[-1]["backend_verified"] = True
                    self.generation_records[-1]["memory_runtime"] = json.loads(
                        json.dumps(counts["memory_runtime"]))
        except (UpstreamError, BackendError, CacheMiss, RuntimeError, ValueError,
                URLError, OSError) as error:
            kind = getattr(error, "kind", "upstream_error")
            if isinstance(error, CacheMiss):
                kind = "cache_miss"
            generation_budget = self._finalize_generation_budget()
            self._log_request(payload, None, counts, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn, plan=self._slim_plan(repair_plan))
            response = {"error": f"upstream failed: {error}"}
            if generation_budget is not None:
                response["generation_budget"] = generation_budget
            extraction_budget = self._finalize_extraction_budget()
            if extraction_budget is not None:
                response["extraction_budget"] = extraction_budget
            self._send_json(502, response)
            return
        total_sec = time.perf_counter() - start

        # ---- oracle-recover (docs/hybrid_spec.md "Oracle recover") ----
        recover_flags: Dict[str, Any] = {}
        action = action_canonical({
            "content": normalized["content"],
            "tool_calls": normalized["tool_calls"],
        })
        if STATE.recover is not None:
            with STATE.lock:
                recover_flags = STATE.recover.check(conv, fingerprint, action, turn)
                recover_now = STATE.recover.should_recover(conv, recover_flags)
                if recover_now:
                    STATE.recover.repaired.add(conv)
            if recover_now:
                repair_t0 = time.perf_counter()
                try:
                    raw_out, _ = _assemble(messages, FULL_ASSEMBLY)
                    data_b, normalized_b = call_upstream(raw_out, None)
                except (UpstreamError, BackendError, RuntimeError, ValueError,
                        URLError, OSError) as error:
                    kind = getattr(error, "kind", "recover_error")
                    generation_budget = self._finalize_generation_budget()
                    self._log_request(payload, None, counts, status=kind,
                                      error=str(error), fingerprint=fingerprint,
                                      conv=conv, turn=turn, recover=recover_flags)
                    response = {"error": f"c2kv recover failed: {error}"}
                    if generation_budget is not None:
                        response["generation_budget"] = generation_budget
                    extraction_budget = self._finalize_extraction_budget()
                    if extraction_budget is not None:
                        response["extraction_budget"] = extraction_budget
                    self._send_json(502, response)
                    return
                ref = STATE.recover.reference.get(fingerprint)
                action_b = action_canonical({
                    "content": normalized_b["content"],
                    "tool_calls": normalized_b["tool_calls"],
                })
                recover_flags.update({
                    "repaired": True,
                    "repair_sec": round(time.perf_counter() - repair_t0, 4),
                    "repair_tokens": (normalized_b["usage"] or {}).get("completion_tokens"),
                    # did the full-regime regeneration reproduce the
                    # reference action verbatim?
                    "repair_fidelity": bool(ref and action_b == ref.get("action")),
                    "recovered_action_match": action_b == action,
                })
                data, normalized = data_b, normalized_b
                total_sec = time.perf_counter() - start

        if prepared_exact is not None:
            try:
                _commit_memory_runtime_final(prepared_exact, normalized, counts)
                total_sec = time.perf_counter() - start
            except (RuntimeError, ValueError, TypeError, KeyError) as error:
                kind = getattr(error, "kind", "memory_runtime_error")
                generation_budget = self._finalize_generation_budget()
                self._log_request(payload, None, counts, status=kind,
                                  error=str(error), fingerprint=fingerprint, conv=conv,
                                  turn=turn, plan=self._slim_plan(repair_plan))
                response = {"error": f"memory runtime finalization failed: {error}"}
                if generation_budget is not None:
                    response["generation_budget"] = generation_budget
                extraction_budget = self._finalize_extraction_budget()
                if extraction_budget is not None:
                    response["extraction_budget"] = extraction_budget
                self._send_json(502, response)
                return

        # ---- reference recording (full-arm run, --record-reference) ----
        if STATE.reference_log_path:
            final_action = action_canonical({
                "content": normalized["content"],
                "tool_calls": normalized["tool_calls"],
            })
            row = {
                "ts": time.time(), "arm": ARM.name, "conv_id": conv,
                "fp": fingerprint, "turn": turn, "action": final_action,
                "finish_reason": normalized["finish_reason"],
                "completion_tokens": (normalized["usage"] or {}).get("completion_tokens"),
            }
            with _log_lock:
                with open(STATE.reference_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Cost columns ride along on the response object.  TTFT is NOT
        # measured: this proxy is non-streaming (single buffered response).
        data.setdefault("c2kv_proxy", {})
        data["c2kv_proxy"].update(
            {
                "backend": BACKEND.name,
                "arm": ARM.name,
                "ratio": ARM.ratio,
                "gist_tokens": counts["gist_tokens"],
                "original_tokens": counts["original_tokens"],
                "history_packed_original_tokens": counts["history_packed_original_tokens"],
                "history_dropped_original_tokens": counts["history_dropped_original_tokens"],
                "history_packed_candidate_doc_count": counts["history_packed_candidate_doc_count"],
                "history_retained_fraction": counts["history_retained_fraction"],
                "n_gist_messages": counts["n_gist_messages"],
                "assemble_sec": round(assemble_sec, 4),
                "wall_sec": round(total_sec, 4),
            }
        )
        data["c2kv_proxy"].update(normalized["cost"])
        if self.generation_records is not None:
            data["c2kv_proxy"].update(_generation_summary(self.generation_records))
        generation_budget = self._finalize_generation_budget()
        if generation_budget is not None:
            data["c2kv_proxy"]["generation_budget"] = generation_budget
        extraction_budget = self._finalize_extraction_budget()
        if extraction_budget is not None:
            data["c2kv_proxy"]["extraction_budget"] = extraction_budget
        if counts.get("memory_runtime") is not None:
            data["c2kv_proxy"]["memory_runtime"] = counts["memory_runtime"]
        counts["repair_frame"] = _repair_frame_check(repair_plan, normalized)
        if counts.get("gold_recovery"):
            data["c2kv_proxy"]["gold_recovery"] = counts["gold_recovery"]
        unit_bytes = (normalized.get("cost") or {}).get("bytes_per_kv_token")
        if isinstance(unit_bytes, int) and unit_bytes > 0 and counts["gist_tokens"]:
            repair_tokens = int((repair_plan or {}).get("repair_block_tokens") or 0)
            counts["history_tensor_accounting"] = {
                "bytes_per_kv_token": unit_bytes,
                "full_equivalent_selected_history_bytes": counts["original_tokens"] * unit_bytes,
                "before_recovery_bytes": counts["gist_tokens"] * unit_bytes,
                "after_recovery_bytes": (counts["gist_tokens"] + repair_tokens) * unit_bytes,
                "recovery_bytes": repair_tokens * unit_bytes,
                "history_ratio_before": counts["original_tokens"] / counts["gist_tokens"],
                "history_ratio_after": counts["original_tokens"] / (counts["gist_tokens"] + repair_tokens),
                "scope": "history_KV_tensor_payload_only; excludes pool duplication, indexes and Python metadata",
            }
        counts["wall_sec"] = round(total_sec, 4)
        self._log_request(payload, normalized, counts, recover=recover_flags,
                          fingerprint=fingerprint, conv=conv, turn=turn,
                          plan=self._slim_plan(repair_plan))
        self._send_json(200, data)

    @staticmethod
    def _slim_plan(plan):
        if not plan:
            return None
        return {k: plan[k] for k in (
            "policy", "placement", "doc_index", "position_start", "position_end",
            "expected_offset", "frame_delta", "frame_delta_status",
            "repair_block_tokens", "already_rotated")
            if k in plan}

    def _finalize_generation_budget(self):
        record = getattr(self, "generation_budget_record", None)
        if record is None:
            return None
        if not getattr(self, "_generation_budget_finalized", False):
            self._generation_budget_final = record.metadata()
            self._generation_budget_finalized = True
        return getattr(self, "_generation_budget_final", None)

    def _finalize_extraction_budget(self):
        record = getattr(self, "extraction_budget_record", None)
        if record is None:
            return None
        if not getattr(self, "_extraction_budget_finalized", False):
            self._extraction_budget_final = record.metadata()
            self._extraction_budget_finalized = True
        return getattr(self, "_extraction_budget_final", None)

    def do_GET(self):
        try:
            with _OPENER.open(f"{UPSTREAM}{self.path}", timeout=60) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except OSError as error:
            self._send_json(502, {"error": str(error)})

    def _passthrough(self, raw: bytes):
        try:
            req = urlrequest.Request(
                f"{UPSTREAM}{self.path}",
                data=raw,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                method="POST",
            )
            with _OPENER.open(req, timeout=600) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except OSError as error:
            self._send_json(502, {"error": str(error)})

    def _send_json(self, code: int, obj: Dict[str, Any]):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_request(self, request, normalized, counts, recover=None, status="ok",
                     error=None, fingerprint=None, conv=None, turn=None, plan=None):
        if not REQUEST_LOG_PATH:
            return
        counts = counts or {}
        recover = recover or {}
        row: Dict[str, Any] = {
            "ts": time.time(),
            "backend": BACKEND.name if BACKEND else None,
            "arm": ARM.name if ARM else None,
            "status": status,
            "error_kind": None,
            "fp": fingerprint,
            "conv_id": conv,
            "turn": turn,
            "n_messages": len(request.get("messages") or []),
            "n_tools": len(request.get("tools") or []),
            "gist_tokens": counts.get("gist_tokens"),
            "original_tokens": counts.get("original_tokens"),
            "history_packed_original_tokens": counts.get("history_packed_original_tokens"),
            "history_dropped_original_tokens": counts.get("history_dropped_original_tokens"),
            "history_packed_candidate_doc_count": counts.get("history_packed_candidate_doc_count"),
            "history_retained_fraction": counts.get("history_retained_fraction"),
            "n_gist_messages": counts.get("n_gist_messages"),
            "wall_sec": counts.get("wall_sec"),
            "error": error,
            "usage": (normalized or {}).get("usage"),
            "finish_reason": (normalized or {}).get("finish_reason"),
            "n_native_tool_calls": len((normalized or {}).get("tool_calls") or []) if normalized else None,
            "native_tool_names": [
                (call.get("function") or {}).get("name")
                for call in ((normalized or {}).get("tool_calls") or [])
            ] if normalized else None,
        }
        row["eval_context"] = getattr(self, "eval_context", {})
        row["request_id"] = getattr(self, "request_id", None)
        generation_records = getattr(self, "generation_records", None)
        if generation_records is not None:
            row["generation_trace"] = generation_records
            row.update(_generation_summary(generation_records))
        generation_budget = self._finalize_generation_budget()
        if generation_budget is not None:
            row["generation_budget"] = generation_budget
        row["sampling_request"] = _sampling_fields(request)
        row["sampling_forwarded"] = getattr(self, "forwarded_sampling", [])
        row["sampling_scope"] = "explicit request fields; omitted server defaults are unknown"
        extraction_trace = getattr(self, "extraction_trace", None)
        if extraction_trace is not None:
            row["extraction_telemetry"] = extraction_trace.snapshot(
                block_refs=(counts.get("memory_runtime") or {}).get("block_refs", []),
                forwarded_requests=getattr(self, "forwarded_gist_keys", []),
                request_status=status)
        extraction_budget = self._finalize_extraction_budget()
        if extraction_budget is not None:
            row["extraction_budget"] = extraction_budget
        if CAPTURE_REQUEST_VIEWS:
            row["request_view"] = _captured_request_view(request)
            row["forwarded_request_views"] = getattr(self, "forwarded_request_views", [])
            row["response_view"] = ({key: normalized.get(key) for key in ("content", "tool_calls")}
                                    if normalized else None)
        if status != "ok":
            row["error_kind"] = status
        # raw-vs-compressed message-class breakdown
        row.update({f"raw_{k}": v for k, v in counts.items()
                    if k in ("system_raw", "history_raw", "current_raw", "compressed")})
        row.update({k: counts.get(k) for k in
                    ("doc_packing", "n_docs", "dropped_docs", "repair_frame",
                     "history_kv", "kv_reuse", "gold_recovery",
                     "history_tensor_accounting", "memory_runtime")
                    if k in counts})
        if counts.get("textarm") is not None:
            row["textarm"] = counts["textarm"]
        # backend cost block (hfserver: cache/logical/prompt/system_len;
        # sglang: kv_resident/kv_peak/kv_pool) + repair columns
        cost = (normalized or {}).get("cost") or {}
        row.update(cost)
        if plan:
            row.update({f"repair_{k}": v for k, v in plan.items()})
        if recover:
            row.update({k: v for k, v in recover.items()})
        with _log_lock:
            with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    global ARM, BACKEND, MEMORY_RUNTIME, UPSTREAM, REQUEST_LOG_PATH
    global DOC_PACKING, MAX_DOC_LENGTH, MAX_DOC_NUM, QUERY_PROJECTION
    global WITNESS_TOKENIZER_PATH
    global MEMORY_RUNTIME_BYTES_PER_KV_TOKEN, MEMORY_RUNTIME_FATAL_ERROR
    global NO_UPSTREAM_RETRIES, CAPTURE_REQUEST_VIEWS, GENERATION_BUDGET, EXTRACTION_BUDGET
    global ATTEMPT_JOURNAL
    textarms.reset_state()  # fresh caches/state per proxy process
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-tokenizer", default="")
    parser.add_argument("--memory-runtime-config", default="",
                        help="opt-in A runtime JSON config")
    parser.add_argument("--memory-tokenizer", default="",
                        help="local tokenizer used for A runtime budget accounting")
    parser.add_argument("--no-upstream-retries", action="store_true",
                        help="disable chat transport and CacheMiss retries")
    parser.add_argument("--capture-request-views", action="store_true",
                        help="record allowlisted benchmark messages/tools and forwarded views")
    parser.add_argument("--max-generation-attempts", type=positive_generation_limit,
                        help="process-wide hard cap on chat generation network calls")
    parser.add_argument("--max-generation-attempts-per-task", type=positive_generation_limit,
                        help="hard cap on actual draft and regeneration calls per explicit task_id")
    parser.add_argument("--max-extraction-attempts", type=positive_extraction_limit,
                        help="process-wide hard cap on client extraction producers")
    parser.add_argument("--upstream", required=True,
                        help="backend base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--backend", default="sglang",
                        choices=["hfserver", "sglang"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--record-reference", default="",
                        help="append a reference-trajectory row per request (full-arm run)")
    parser.add_argument("--reference", default="",
                        help="reference jsonl to diff against (recover arms)")
    parser.add_argument("--query-projection", choices=["base", "gist"],
                        help="checkpoint query projection; raw-KV baselines always use base")
    parser.add_argument("--doc-packing", default=DOC_PACKING, choices=DOC_PACKINGS,
                        help="how compressed history is cut into docs: 'turn' = "
                             "the training format (default), 'message' = one doc "
                             "per message (pre-2026-09 bench numbers)")
    parser.add_argument("--max-doc-length", type=int, default=MAX_DOC_LENGTH,
                        help="turn packing: split docs above this many template "
                             "tokens; supplied by the checkpoint profile")
    parser.add_argument("--max-doc-num", type=int, default=MAX_DOC_NUM,
                        help="turn packing: keep doc 0 + the last N-1 docs, drop "
                             "the rest; supplied by the checkpoint profile")
    args = parser.parse_args(argv)
    if bool(args.memory_runtime_config) != bool(args.memory_tokenizer):
        parser.error(
            "--memory-runtime-config and --memory-tokenizer must be supplied together")
    DOC_PACKING = args.doc_packing
    MAX_DOC_LENGTH = int(args.max_doc_length)
    MAX_DOC_NUM = int(args.max_doc_num)
    QUERY_PROJECTION = args.query_projection
    ARM = get_arm(args.arm)
    MEMORY_RUNTIME = None
    MEMORY_RUNTIME_BYTES_PER_KV_TOKEN = None
    MEMORY_RUNTIME_FATAL_ERROR = None
    NO_UPSTREAM_RETRIES = bool(args.no_upstream_retries)
    CAPTURE_REQUEST_VIEWS = bool(args.capture_request_views)
    GENERATION_BUDGET = GenerationBudget(
        args.max_generation_attempts, per_task_limit=args.max_generation_attempts_per_task)
    EXTRACTION_BUDGET = ExtractionBudget(args.max_extraction_attempts)
    if args.memory_runtime_config:
        MEMORY_RUNTIME = _load_memory_runtime(
            args.memory_runtime_config, args.memory_tokenizer)
        _validate_memory_runtime_arm(MEMORY_RUNTIME, ARM)
        if args.reference or args.witness_tokenizer:
            raise SystemExit(
                "FATAL: memory runtime cannot be combined with "
                "--reference or --witness-tokenizer")
    UPSTREAM = args.upstream.rstrip("/")
    REQUEST_LOG_PATH = args.request_log
    # A distinct prefix keeps the append-only journal out of the existing
    # proxy_*.jsonl request-log discovery used by the runner and collector.
    ATTEMPT_JOURNAL = (AttemptJournal(attempt_journal_path(REQUEST_LOG_PATH))
        if REQUEST_LOG_PATH else None)
    BACKEND = get_backend(args.backend, _post_json)
    WITNESS_TOKENIZER_PATH = args.witness_tokenizer
    STATE.reference_log_path = args.record_reference
    if args.reference:
        if not ARM.recover:
            raise SystemExit(f"FATAL: --reference needs a recover arm, got {ARM.name!r}")
        STATE.recover = RecoverState(load_reference(args.reference))
        print(f"loaded reference: {len(STATE.recover.reference)} states "
              f"from {args.reference}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    runtime_mode = getattr(MEMORY_RUNTIME, "mode", None)
    print(f"proxy backend={BACKEND.name} arm={ARM.name} memory_runtime={runtime_mode} "
          f"no_upstream_retries={NO_UPSTREAM_RETRIES} doc_packing={DOC_PACKING} "
          f"max_generation_attempts={GENERATION_BUDGET.limit} "
          f"max_extraction_attempts={EXTRACTION_BUDGET.limit} "
          f"max_doc_length={MAX_DOC_LENGTH} max_doc_num={MAX_DOC_NUM} listening on "
          f"{args.host}:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
