"""Arm-aware OpenAI-compatible reverse proxy.

All three benchmarks speak the OpenAI chat-completions protocol, so instead
of forking each benchmark we front the SGLang server with this proxy:

    benchmark client -> proxy (arm assembly) -> SGLang --enable-c2kv

Per request the proxy decides, per the active arm, which *history* messages
are sent as raw text and which are replaced by a server-side gist reference
(`c2kv_key_hash`, produced by POST /v1/c2kv/extract).  The rule for what
counts as history: every message except the trailing block after the last
user/tool message; system messages are history by position but always kept
raw (never compressed).  The final user message and tool results of the
current turn stay raw.

Extract results are cached by (role, sha256(content), ratio) so multi-turn
sessions do not re-compress the same prefix, which also keeps KV accounting
consistent with the teacher-forced harness (one gist per history block).

Timing: the proxy is NON-STREAMING (one buffered request/response per call).
It records per-request wall time and token accounting under the response
object's `c2kv_proxy` field plus a JSONL request log; TTFT is not measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import URLError

from arms import Arm, get_arm  # type: ignore


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
UPSTREAM = ""
REQUEST_LOG_PATH = ""
# Segment granularity handed to /v1/c2kv/extract: "message" (historical
# default) or "turn" (the granularity history_only arms are trained on).
DOC_PACKING = "message"
# Trainer max_doc_num tail cap; 0 = uncapped (historical default).
MAX_DOCS = 0
_log_lock = threading.Lock()


def _http_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{base_url.rstrip('/')}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _content_key(role: str, content: str) -> str:
    return hashlib.sha256(f"{role}\x00{content}".encode("utf-8")).hexdigest()


def _render_tool_calls(tool_calls: Any) -> str:
    """Render OpenAI ``tool_calls`` exactly as the trainer renders them.

    Byte-for-byte ``train.train_data_multiturn._render_agent_tool_calls``:
    one ``<tool_call>{"name":...,"arguments":...}</tool_call>`` block per
    call, compact JSON separators, ``ensure_ascii=False``.

    Without this the proxy hands ``/v1/c2kv/extract`` an assistant message
    whose ``content`` is ``None`` and whose action lives only in
    ``tool_calls`` -- and because ``serving_chat._compute_c2kv_segments``
    POPS every annotated message out of the request, that action is then
    absent from the prompt entirely.  The compressed history of the
    c2kv/hybrid arms would keep every tool result while losing every call the
    agent made to obtain it.
    """
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except json.JSONDecodeError:
            return ""
    if not tool_calls:
        return ""
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return ""
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
        rendered.append(
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": arguments},
                         ensure_ascii=False, separators=(",", ":"))
            + "\n</tool_call>"
        )
    return "\n".join(rendered)


def _message_doc_text(message: Dict[str, Any]) -> str:
    """Text the trainer would have compressed for this single message.

    ``content`` (stringified) plus, for assistant messages, the rendered
    ``tool_calls`` under an ``Action:`` header -- the same two parts
    ``train_data_multiturn._normal_agent_message`` concatenates before the
    turn-document builder ever sees the message.
    """
    content = message.get("content")
    if not isinstance(content, str):
        content = "" if content is None else json.dumps(content, ensure_ascii=False)
    parts = [content] if content else []
    calls = _render_tool_calls(
        message.get("tool_calls") or message.get("toolCalls") or message.get("function_call")
    )
    if calls:
        parts.append("Action:\n" + calls)
    return "\n\n".join(parts)


def _turn_documents(history: List[Dict[str, Any]]) -> List[str]:
    """Group history messages into the trainer's turn documents.

    Byte-for-byte the layout of
    ``train_data_multiturn._agent_history_turn_docs`` applied to
    ``_normal_agent_message`` output (which maps ``role="tool"`` to
    ``"user"``): a user message opens a turn, assistant output accumulates
    into it, any other role is appended as ``[role]\ncontent``.  Selected by
    ``--doc-packing turn``: this is the document granularity every
    ``doc_mode=history_only`` arm is trained on, whereas ``message`` packing
    (the default, kept for continuity with earlier matrices) compresses one
    document per raw message and therefore feeds the gist encoder a segment
    shape it never saw during training.
    """
    docs: List[str] = []
    current_user: Optional[str] = None
    outputs: List[str] = []

    def flush() -> None:
        nonlocal current_user, outputs
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
        docs.append("\n".join(parts).strip())
        current_user = None
        outputs = []

    for message in history:
        role = message.get("role") or "user"
        if role == "tool":
            role = "user"
        content = _message_doc_text(message).strip()
        if not content and role != "assistant":
            continue
        if role == "user":
            flush()
            current_user = content
        elif role == "assistant":
            outputs.append(content)
        else:
            outputs.append(f"[{role}]\n{content}")
    flush()
    return docs


def _extract(role: str, content: str, ratio: int, timeout: int) -> Dict[str, Any]:
    key = (role, _content_key(role, content), ratio)

    def produce() -> Dict[str, Any]:
        result = _http_json(
            UPSTREAM,
            "/v1/c2kv/extract",
            {
                "text": content,
                "compression_ratio": ratio,
                "role": role,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout,
        )
        if not result.get("success", True) or not result.get("key_hash"):
            raise RuntimeError(f"c2kv extract failed: {result.get('error') or result}")
        return result

    return CACHE.get_or_put(key, produce)


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


def _assemble(messages: List[Dict[str, Any]], arm: Arm, timeout: int):
    """Return (out_messages, gist_tokens, original_tokens, n_gist, n_dropped).

    ``DOC_PACKING`` selects the segment granularity handed to
    ``/v1/c2kv/extract``:

    * ``message`` -- one document per raw history message, keeping its own
      role.  Historical default; kept so earlier matrices stay reproducible.
    * ``turn``    -- the trainer's turn documents (``_turn_documents``), each
      extracted with ``role="user"``.  This is the packing every
      ``doc_mode=history_only`` checkpoint is trained on, so it is the only
      packing under which a serving number is a measurement of the checkpoint
      rather than of an unseen segment shape.

    ``MAX_DOCS`` mirrors the trainer's ``max_doc_num`` tail policy: when the
    history yields more documents than the grid the arm was trained with, the
    OLDEST documents are dropped (``0`` = no cap, the historical behaviour).
    Dropped documents are reported so a cell can never silently be scored on
    a truncated history.
    """
    cutoff = _history_cutoff(messages)
    gist_tokens = 0
    original_tokens = 0
    n_gist = 0

    if not arm.compress_history:
        return list(messages), 0, 0, 0, 0

    raw_tail = arm.hybrid_top_k or 0
    head: List[Dict[str, Any]] = []          # system prefix, always raw
    history: List[Dict[str, Any]] = []       # compressible block
    tail: List[Dict[str, Any]] = []          # raw hybrid tail + current turn
    for i, message in enumerate(messages):
        if (message.get("role") or "user") == "system":
            head.append(message)
        elif i >= cutoff or (raw_tail and i >= cutoff - raw_tail):
            tail.append(message)
        else:
            history.append(message)

    if DOC_PACKING == "turn":
        docs = [("user", text) for text in _turn_documents(history)]
    else:
        docs = []
        for message in history:
            text = _message_doc_text(message)
            if not text:
                continue
            docs.append((message.get("role") or "user", text))

    n_dropped = 0
    if MAX_DOCS and len(docs) > MAX_DOCS:
        n_dropped = len(docs) - MAX_DOCS
        docs = docs[-MAX_DOCS:]

    out: List[Dict[str, Any]] = list(head)
    for role, text in docs:
        record = _extract(role, text, arm.ratio, timeout)
        gist_tokens += int(record.get("gist_len") or 0)
        original_tokens += int(record.get("original_seq_len") or 0)
        n_gist += 1
        out.append({
            "role": role,
            "content": text,
            "c2kv_key_hash": record["key_hash"],
            # lets the server re-extract on cache miss (e.g. after a restart)
            "c2kv_ratio": arm.ratio,
        })
    out.extend(tail)
    return out, gist_tokens, original_tokens, n_gist, n_dropped


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
        assert ARM is not None
        if ARM.constrain_tools:
            self._send_json(
                400,
                {
                    "error": (
                        "constrain_tools is an hf_server-private field and is not supported "
                        "by the SGLang fork; cd_full/cd_c2kv are disabled in this matrix"
                    )
                },
            )
            return
        start = time.perf_counter()
        try:
            messages, gist_tokens, original_tokens, n_gist, n_dropped = _assemble(
                payload.get("messages") or [], ARM, 600
            )
        except (RuntimeError, URLError, OSError) as error:
            self._send_json(502, {"error": f"c2kv assembly failed: {error}"})
            return
        assemble_sec = time.perf_counter() - start
        out_payload = dict(payload)
        out_payload["messages"] = messages
        try:
            data = _http_json(UPSTREAM, self.path, out_payload, 600)
        except (RuntimeError, URLError, OSError) as error:
            self._send_json(502, {"error": f"upstream failed: {error}"})
            return
        total_sec = time.perf_counter() - start
        # Cost columns ride along on the response object.  TTFT is NOT
        # measured: this proxy is non-streaming (single buffered response).
        data.setdefault("c2kv_proxy", {})
        data["c2kv_proxy"].update(
            {
                "arm": ARM.name,
                "ratio": ARM.ratio,
                "gist_tokens": gist_tokens,
                "original_tokens": original_tokens,
                "n_gist_messages": n_gist,
                "doc_packing": DOC_PACKING,
                "dropped_docs": n_dropped,
                "assemble_sec": round(assemble_sec, 4),
                "wall_sec": round(total_sec, 4),
            }
        )
        self._send_json(200, data)
        self._log_request(
            payload, data, gist_tokens, original_tokens, n_gist, total_sec, n_dropped
        )

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

    def _log_request(self, request, response, gist_tokens, original_tokens, n_gist, total_sec, n_dropped=0):
        if not REQUEST_LOG_PATH:
            return
        row = {
            "ts": time.time(),
            "arm": ARM.name if ARM else None,
            "n_messages": len(request.get("messages") or []),
            "n_tools": len(request.get("tools") or []),
            "gist_tokens": gist_tokens,
            "original_tokens": original_tokens,
            "n_gist_messages": n_gist,
            "doc_packing": DOC_PACKING,
            "max_docs": MAX_DOCS,
            "dropped_docs": n_dropped,
            "wall_sec": round(total_sec, 4),
            "status": "ok",
            "usage": response.get("usage"),
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        }
        with _log_lock:
            with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    global ARM, UPSTREAM, REQUEST_LOG_PATH, DOC_PACKING, MAX_DOCS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="SGLang base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--doc-packing", choices=("message", "turn"), default="message",
        help=(
            "segment granularity for /v1/c2kv/extract. 'message' = one document"
            " per raw history message (historical default). 'turn' = the"
            " trainer's turn documents, i.e. the granularity every"
            " doc_mode=history_only checkpoint was trained on."
        ),
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help=(
            "cap on compressed history documents, mirroring the trainer's"
            " max_doc_num tail policy (oldest dropped). 0 = uncapped."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_docs < 0:
        parser.error("--max-docs must be >= 0")
    ARM = get_arm(args.arm)
    UPSTREAM = args.upstream.rstrip("/")
    REQUEST_LOG_PATH = args.request_log
    DOC_PACKING = args.doc_packing
    MAX_DOCS = args.max_docs
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(
        f"proxy arm={ARM.name} doc_packing={DOC_PACKING} max_docs={MAX_DOCS}"
        f" listening on {args.host}:{args.port} -> {UPSTREAM}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
