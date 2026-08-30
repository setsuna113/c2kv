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
current turn stay raw.

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
its own protocol (hf_server: request-level c2kv_repair; sglang:
/v1/c2kv/repair_extract + message-level c2kv_repair_key_hashes with the
proxy-ledger position_offset).

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

import repair_policy
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
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as error:
            text = ""
            try:
                text = error.read().decode("utf-8", "replace")
            except OSError:
                pass
            if error.code < 500 and error.code != 429:
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


def _assemble(messages: List[Dict[str, Any]], arm: Arm, timeout: int = 600):
    """Return (out_messages, counts).

    ``counts`` carries the message-class breakdown (system/hybrid-tail/
    current kept raw vs compressed) plus the compressed-token ledger and
    the per-doc extract records (message index, role, record) in
    conversation order — the proxy-side ledger that repair planning and
    the logical-token cost column are computed from.  Raw token counts are
    NOT estimated here; physical numbers come from the backend's response.
    """
    cutoff = _history_cutoff(messages)
    out: List[Dict[str, Any]] = []
    gist_tokens = 0
    original_tokens = 0
    n_gist = 0
    message_counts = {"system_raw": 0, "history_raw": 0, "current_raw": 0,
                      "compressed": 0}
    compressed_records: List[Dict[str, Any]] = []
    for i, message in enumerate(messages):
        role = message.get("role") or "user"
        content = message.get("content")
        content = content if isinstance(content, str) else json.dumps(content or "")
        in_history = i < cutoff
        keep_raw = (
            not arm.compress_history
            or not in_history
            or role == "system"
            or (arm.hybrid_top_k and i >= cutoff - arm.hybrid_top_k)
        )
        if keep_raw:
            raw = dict(message)
            # training-dialect rendering applies to RAW assistant
            # tool_calls turns as well: backends without server-side
            # normalization (sglang) would otherwise feed the chat
            # template's native tool_calls branch — a surface the model
            # was never trained on.  Idempotent for hf_server (its chat()
            # normalization produced the identical text).
            if role == "assistant" and message.get("tool_calls"):
                raw["content"] = _render_action_dialect(message)
                raw.pop("tool_calls", None)
            out.append(raw)
            if role == "system":
                message_counts["system_raw"] += 1
            elif in_history:
                message_counts["history_raw"] += 1
            else:
                message_counts["current_raw"] += 1
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            # see _render_action_dialect: never extract the bare (null)
            # content of a tool-call turn
            content = _render_action_dialect(message)
        record = _extract(role, content, arm.ratio, timeout)
        gist_tokens += int(record.get("gist_len") or 0)
        original_tokens += int(record.get("original_seq_len") or 0)
        n_gist += 1
        message_counts["compressed"] += 1
        compressed = dict(message)
        compressed["content"] = content
        compressed.pop("tool_calls", None)
        compressed["c2kv_key_hash"] = record["key_hash"]
        # lets the server re-extract on cache miss (e.g. after a restart)
        compressed["c2kv_ratio"] = arm.ratio
        out.append(compressed)
        compressed_records.append({
            "message_index": i, "role": role, "content": content,
            "record": record,
        })
    counts = dict(message_counts)
    counts["gist_tokens"] = gist_tokens
    counts["original_tokens"] = original_tokens
    counts["n_gist_messages"] = n_gist
    counts["compressed_records"] = compressed_records
    return out, counts


def _system_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        (m.get("content") or "") for m in messages if m.get("role") == "system")


def plan_repair(messages: List[Dict[str, Any]], arm: Arm,
                counts: Dict[str, Any],
                tools: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Resolve an arm's repair policy against the proxy ledger.

    The target doc's position_offset (d_corr bakes the absolute RoPE phase
    at capture) = system tokens + Σ original_seq_len of every compressed
    message before the target.  Raw tail / current turns sit AFTER the
    compressed prefix and are never crossed.
    """
    if not arm.repair or not getattr(BACKEND, "needs_repair_plan", False):
        return None
    policy = str((arm.repair or {}).get("policy") or "first")
    parsed = repair_policy.parse_policy(policy)
    records = counts.get("compressed_records") or []
    if not records:
        # no history compressed yet: a legitimate no-op (hf_server's
        # `repair_policy is not None and compressed` guard — dropping it
        # made the FIRST request of every repair-arm session crash with an
        # uncaught ValueError and no log row)
        return None
    doc_counts = [1] * len(records)  # sglang: whole-message docs (O-2)
    doc_index, _first_chunk, _span_len = repair_policy.span_selection(
        doc_counts, parsed["kind"], parsed["index"])
    target = records[doc_index]
    system_len = 0
    system = _system_text(messages)
    if system:
        # one cached extract of the system block gives its template token
        # length (and doubles as the request-log system_len column source);
        # the resulting gist entry simply sits unused in the pool
        sys_record = _extract("system", system, arm.ratio, tools=tools)
        system_len = int(sys_record.get("original_seq_len") or 0)
    offset = system_len
    for record in records[:doc_index]:
        offset += int(record["record"].get("original_seq_len") or 0)
    span = BACKEND.repair_extract(
        text=target["content"], role=target["role"],
        span_start=0, span_end=None,
        position_offset=offset, source_doc_index=doc_index)
    return {
        "policy": policy, "message_index": target["message_index"],
        "doc_index": doc_index, "position_offset": offset,
        "repair_key_hash": span.get("key_hash"),
        "repair_block_tokens": span.get("token_len"),
    }


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
        try:
            messages_out, counts = _assemble(messages, ARM)
            repair_plan = plan_repair(messages, ARM, counts,
                                     tools=payload.get("tools"))
        except (RuntimeError, ValueError, URLError, OSError, UpstreamError,
                BackendError) as error:
            kind = getattr(error, "kind", "assemble_error")
            self._log_request(payload, None, None, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn)
            self._send_json(502, {"error": f"c2kv assembly failed: {error}"})
            return
        assemble_sec = time.perf_counter() - start

        def send_upstream(out_messages, plan):
            out_payload = BACKEND.prepare_chat(dict(payload), ARM, plan)
            out_payload["messages"] = out_messages
            return _post_json(self.path, out_payload, 600), out_payload

        try:
            data, _ = send_upstream(messages_out, repair_plan)
            normalized = BACKEND.normalize_response(data)
        except (UpstreamError, BackendError) as error:
            kind = getattr(error, "kind", "upstream_error")
            self._log_request(payload, None, counts, status=kind,
                              error=str(error), fingerprint=fingerprint, conv=conv,
                              turn=turn, repair=self._slim_plan(repair_plan))
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
                    data_b, _ = send_upstream(raw_out, None)
                    normalized_b = BACKEND.normalize_response(data_b)
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
            "policy", "doc_index", "position_offset", "repair_block_tokens")
            if k in plan}

    def do_GET(self):
        try:
            with urlrequest.urlopen(f"{UPSTREAM}{self.path}", timeout=60) as resp:
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
            with urlrequest.urlopen(req, timeout=600) as resp:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True,
                        help="backend base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--backend", default="hfserver",
                        choices=["hfserver", "sglang"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--record-reference", default="",
                        help="append a reference-trajectory row per request (full-arm run)")
    parser.add_argument("--reference", default="",
                        help="reference jsonl to diff against (recover arms)")
    args = parser.parse_args(argv)
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
    print(f"proxy backend={BACKEND.name} arm={ARM.name} listening on "
          f"{args.host}:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
