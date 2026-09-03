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

Upstream failures are retried (2x, exponential backoff) and always leave a
request-log row with a failure kind — a benchmark entry must never vanish
without a trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

# this proxy is always a local sidecar talking to 127.0.0.1 upstreams; an
# ambient http_proxy env (login shells here carry one) must never intercept
# its upstream calls
_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))

import repair_policy
import textarms
from arms import Arm, get_arm  # type: ignore
from backends import BackendError, get_backend  # type: ignore


class ExtractCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    def get_or_put(
        self, key: Tuple[str, str, int], producer
    ) -> Dict[str, Any]:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = producer()
            return self._cache[key]


CACHE = ExtractCache()
ARM: Optional[Arm] = None
BACKEND = None  # set in main()
UPSTREAM = ""
REQUEST_LOG_PATH = ""
_log_lock = threading.Lock()

# --doc-packing / --max-doc-length / --max-doc-num (see module docstring and
# docs/c2kv_semantics.md).  DEFAULTS = the checkpoint-1088 training values
# (HISTORY_MAX_DOC_LENGTH / HISTORY_MAX_DOC_NUM in
# agent/train_agent_history_c2kv_npu.sh: 512/12) — serving must match
# training; 768/16 (the old D-line harness caliber) is available by flag
# but shifts every compression arm off its trained regime.
DOC_PACKING = "turn"
MAX_DOC_LENGTH = 512
MAX_DOC_NUM = 12
DOC_PACKINGS = ("turn", "message")


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


def _post_json(path: str, payload: Dict[str, Any],
               timeout: int, retries: int = 2) -> Dict[str, Any]:
    """POST JSON to UPSTREAM, retrying 5xx/network failures with backoff.

    4xx (except 429) are deterministic client errors and are not retried.
    The final failure raises UpstreamError with the upstream body.  Note
    the SGLang stack reports many failures as HTTP 200 with error bodies —
    those are classified by the backend (BackendError), not here.
    """
    body = json.dumps(payload).encode("utf-8")
    last: Optional[UpstreamError] = None
    for attempt in range(retries + 1):
        req = urlrequest.Request(
            f"{UPSTREAM.rstrip('/')}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as error:
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
            last = UpstreamError(0, str(error))
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
    assert last is not None
    raise last


def _content_key(role: str, content: str) -> str:
    return hashlib.sha256(f"{role}\x00{content}".encode("utf-8")).hexdigest()


def _extract(role: str, content: str, ratio: int, timeout: int = 600,
             tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # tools participate in the cache key: the same system text rendered
    # with different tool schemas yields different original_seq_len (the
    # Qwen template renders tools into the system block)
    tools_key = _digest(tools or [])
    key = (role, _content_key(role, content), ratio, tools_key)
    return CACHE.get_or_put(
        key, lambda: BACKEND.extract(content, role, ratio, tools=tools))


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
    """Stable per-conversation id: system head + first non-system message."""
    canon = _canonical_messages(messages)
    head = canon[0] if canon else {}
    first = next((m for m in canon if m.get("role") != "system"), {})
    return _digest([head, first])


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
    # raw-path training dialect: tool -> bare user message (_normal_chat_message:
    # {"role": "user", "content": str}); a missing system prompt gets the
    # training default injected
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
    packing = DOC_PACKING if arm.compress_history else "message"
    dropped_docs = 0
    n_docs = 0

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
                  source_indices: List[int]) -> None:
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
                for text, record in _fit_doc(
                    doc["content"], arm.ratio,
                    lambda role, text, ratio: _extract(role, text, ratio, timeout),
                    MAX_DOC_LENGTH,
                ):
                    docs.append((text, record, doc["source_indices"]))
            docs, dropped_docs = _select_docs(docs, MAX_DOC_NUM)
        first_index = compressible[0][0] if compressible else None
        compressible_set = {i for i, _ in compressible}
        for i, message in enumerate(messages):
            if i == first_index:
                for text, record, sources in docs:
                    _emit_doc("user", text, record, sources)
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
            record = _extract(role, content, arm.ratio, timeout)
            _emit_doc(role, content, record, [i])
    counts = dict(message_counts)
    counts["gist_tokens"] = gist_tokens
    counts["original_tokens"] = original_tokens
    counts["n_gist_messages"] = n_gist
    counts["compressed_records"] = compressed_records
    counts["doc_packing"] = packing
    counts["n_docs"] = n_docs
    counts["dropped_docs"] = dropped_docs
    # index in `out` where the current (raw) block starts: repair-only
    # messages for append placements are inserted right before it
    counts["current_start_out_index"] = len(out) - message_counts["current_raw"]
    return out, counts


def _system_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        (m.get("content") or "") for m in messages if m.get("role") == "system")


def _strip_c2kv_fields(message: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in message.items() if not str(k).startswith("c2kv_")}


def _textarm_compress(payload: Dict[str, Any], meter=None) -> str:
    """One compressor call for a text-level baseline arm.  Validates the
    finish reason and non-empty content — HTTP-200-with-error-body and
    abort finishes are FAILURES (TextarmCompressorError), never empty
    summaries that would get cached for the rest of the conversation.
    "length" is a NORMAL finish: the HiAgent summarizer decodes with
    max_tokens~100 and works BY truncation.  ``meter`` (optional) receives
    the response usage block for cost accounting."""
    import textarms

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


def _apply_text_arm(payload: Dict[str, Any], arm, conv: str
                    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rewrite history per the arm's text policy (textarms.py) BEFORE
    assembly; the arm is full-mode downstream.  Compressor calls go
    straight to the upstream endpoint (no arm semantics, no recursion
    through the chat handler).  Compressor tokens/wall-time accumulate
    into stats["compressor_usage"] (the fairness ruling: text baselines'
    extra LLM calls must appear in the cost columns)."""
    import textarms

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

    if arm.text_policy == "hiagent":
        out, stats = textarms.hiagent_transform(
            messages, compress, _render_action_dialect, model=model)
    else:
        mode = "hist" if arm.text_policy == "acon_hist" else "obs"
        out, stats = textarms.acon_transform(
            messages, compress, _render_action_dialect, conv,
            mode=mode, model=model)
    stats["compressor_usage"] = usage_acc
    staged = dict(payload)
    staged["messages"] = out
    return staged, stats


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
    # + Σ original_seq_len of the compressed docs before the target.  Only
    # computable when a system message exists (the tool prologue length is
    # measured through it); BFCL FC sends none -> None, check skipped.
    expected_offset: Optional[int] = None
    system = _system_text(messages)
    if system:
        sys_record = _extract("system", system, arm.ratio, tools=tools)
        expected_offset = int(sys_record.get("original_seq_len") or 0)
        for record in records[:doc_index]:
            expected_offset += int(record["record"].get("original_seq_len") or 0)
    position_start = span.get("position_start")
    frame_delta = None
    if expected_offset is not None and position_start is not None:
        frame_delta = int(position_start) - int(expected_offset)
    return {
        "policy": policy, "placement": placement,
        "message_index": target_out_index, "target_out_index": target_out_index,
        "doc_index": doc_index,
        "current_start_out_index": int(counts.get("current_start_out_index", len(out_messages))),
        "position_offset": position_start,
        "position_start": position_start, "position_end": span.get("position_end"),
        "expected_offset": expected_offset, "frame_delta": frame_delta,
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
    where the previous gist ends.  ``ok`` is None when not computable."""
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
    }
    k = int(plan.get("doc_index", -1))
    expected = None
    if plan.get("placement") == "in_place":
        if k == 0 and gists:
            expected = None  # first doc: prologue end, not derivable from gists
        elif 0 < k <= len(gists):
            prev = gists[k - 1]
            expected = int(prev["position_cursor"]) + int(prev["original_seq_len"])
    elif 0 <= k < len(gists):
        expected = int(gists[k]["position_cursor"])
    if expected is not None and plan.get("position_start") is not None:
        result["expected_from_layout"] = expected
        result["ok"] = int(plan["position_start"]) == expected
    if repairs:
        result["server_placement"] = repairs[0].get("placement")
        result["server_position_start"] = repairs[0].get("position_start")
    return result


class ProxyState:
    """Process-wide proxy state (backend, recover config, reference log)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recover: Optional[RecoverState] = None
        self.reference_log_path: str = ""


STATE = ProxyState()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _is_chat(self) -> bool:
        return self.path.endswith("/v1/chat/completions") or self.path.endswith(
            "/chat/completions"
        )

    def do_POST(self):
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
        start = time.perf_counter()
        messages = payload.get("messages") or []
        fingerprint = messages_fingerprint(messages)
        conv = conversation_id(messages)
        turn = len(messages)
        text_stats: Optional[Dict[str, Any]] = None
        try:
            if getattr(ARM, "text_policy", None):
                payload, text_stats = _apply_text_arm(payload, ARM, conv)
                messages = payload["messages"]
            messages_out, counts = _assemble(messages, ARM)
            if text_stats is not None:
                counts["textarm"] = text_stats
            repair_plan = plan_repair(messages, ARM, counts,
                                     tools=payload.get("tools"),
                                     out_messages=messages_out)
        except (RuntimeError, ValueError, URLError, OSError, UpstreamError,
                BackendError) as error:
            kind = getattr(error, "kind",
                           "textarm_error" if text_stats is not None else "assemble_error")
            self._log_request(payload, None, None, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn)
            self._send_json(502, {"error": f"c2kv assembly failed: {error}"})
            return
        assemble_sec = time.perf_counter() - start

        def send_upstream(out_messages, plan):
            # prepare_chat must SEE the assembled messages: the sglang
            # backend attaches repair hashes to the (gist-marked) target
            # message — feeding it the raw payload made every rp-arm
            # request fail "repair target has no c2kv_key_hash"
            staged = dict(payload)
            staged["messages"] = out_messages
            out_payload = BACKEND.prepare_chat(staged, ARM, plan)
            return _post_json(self.path, out_payload, 600), out_payload

        def call_upstream(out_messages, plan):
            data_, _ = send_upstream(out_messages, plan)
            try:
                return data_, BACKEND.normalize_response(data_)
            except BackendError as error:
                if getattr(error, "kind", "") == "cache_miss":
                    raise CacheMiss(error.detail) from error
                raise

        try:
            try:
                data, normalized = call_upstream(messages_out, repair_plan)
            except CacheMiss:
                # pool-evicted gists: re-extract every compressed doc
                # (re-inserts entries under the same content hashes), then
                # retry the identical request once
                for record in counts.get("compressed_records") or []:
                    _extract(record["role"], record["content"], ARM.ratio)
                data, normalized = call_upstream(messages_out, repair_plan)
        except (UpstreamError, BackendError, CacheMiss) as error:
            kind = getattr(error, "kind", "upstream_error")
            if isinstance(error, CacheMiss):
                kind = "cache_miss"
            self._log_request(payload, None, counts, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn, plan=self._slim_plan(repair_plan))
            self._send_json(502, {"error": f"upstream failed: {error}"})
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
                    self._log_request(payload, None, counts, status=kind,
                                      error=str(error), fingerprint=fingerprint,
                                      conv=conv, turn=turn, recover=recover_flags)
                    self._send_json(502, {"error": f"c2kv recover failed: {error}"})
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
                "n_gist_messages": counts["n_gist_messages"],
                "assemble_sec": round(assemble_sec, 4),
                "wall_sec": round(total_sec, 4),
            }
        )
        data["c2kv_proxy"].update(normalized["cost"])
        counts["repair_frame"] = _repair_frame_check(repair_plan, normalized)
        self._send_json(200, data)
        counts["wall_sec"] = round(total_sec, 4)
        self._log_request(payload, normalized, counts, recover=recover_flags,
                          fingerprint=fingerprint, conv=conv, turn=turn,
                          plan=self._slim_plan(repair_plan))

    @staticmethod
    def _slim_plan(plan):
        if not plan:
            return None
        return {k: plan[k] for k in (
            "policy", "placement", "doc_index", "position_start", "position_end",
            "expected_offset", "frame_delta", "repair_block_tokens",
            "already_rotated")
            if k in plan}

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
            "n_gist_messages": counts.get("n_gist_messages"),
            "wall_sec": counts.get("wall_sec"),
            "error": error,
            "usage": (normalized or {}).get("usage"),
            "finish_reason": (normalized or {}).get("finish_reason"),
        }
        if status != "ok":
            row["error_kind"] = status
        # raw-vs-compressed message-class breakdown
        row.update({f"raw_{k}": v for k, v in counts.items()
                    if k in ("system_raw", "history_raw", "current_raw", "compressed")})
        row.update({k: counts.get(k) for k in
                    ("doc_packing", "n_docs", "dropped_docs", "repair_frame")
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
    global ARM, BACKEND, UPSTREAM, REQUEST_LOG_PATH
    global DOC_PACKING, MAX_DOC_LENGTH, MAX_DOC_NUM
    textarms.reset_state()  # fresh caches/state per proxy process
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--doc-packing", default=DOC_PACKING, choices=DOC_PACKINGS,
                        help="how compressed history is cut into docs: 'turn' = "
                             "the training format (default), 'message' = one doc "
                             "per message (pre-2026-09 bench numbers)")
    parser.add_argument("--max-doc-length", type=int, default=MAX_DOC_LENGTH,
                        help="turn packing: split docs above this many template "
                             "tokens (D-line caliber 768; ckpt-1088 trained at 512)")
    parser.add_argument("--max-doc-num", type=int, default=MAX_DOC_NUM,
                        help="turn packing: keep doc 0 + the last N-1 docs, drop "
                             "the rest (D-line caliber 16; ckpt-1088 trained at 12)")
    args = parser.parse_args(argv)
    DOC_PACKING = args.doc_packing
    MAX_DOC_LENGTH = int(args.max_doc_length)
    MAX_DOC_NUM = int(args.max_doc_num)
    ARM = get_arm(args.arm)
    UPSTREAM = args.upstream.rstrip("/")
    REQUEST_LOG_PATH = args.request_log
    BACKEND = get_backend(args.backend, _post_json)
    STATE.reference_log_path = args.record_reference
    if args.reference:
        if not ARM.recover:
            raise SystemExit(f"FATAL: --reference needs a recover arm, got {ARM.name!r}")
        STATE.recover = RecoverState(load_reference(args.reference))
        print(f"loaded reference: {len(STATE.recover.reference)} states "
              f"from {args.reference}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"proxy backend={BACKEND.name} arm={ARM.name} doc_packing={DOC_PACKING} "
          f"max_doc_length={MAX_DOC_LENGTH} max_doc_num={MAX_DOC_NUM} listening on "
          f"{args.host}:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
