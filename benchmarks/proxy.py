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

The compressed block is cut into documents exactly the way the trainer cuts
them (``train_data_multiturn._fit_reused_history_with_indices``): turn
documents by default (``--doc-packing turn``), each split to at most
``--max-doc-length`` template tokens, and only then tail-selected to
``--max-docs`` documents keeping doc 0 plus the newest ones.  Split before
cap is the trainer's order, so the cap counts chunks and not turns.

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
# Segment granularity handed to /v1/c2kv/extract: "turn" (the granularity
# history_only arms are trained on) or "message" (one document per raw
# message, kept selectable for continuity with earlier matrices).
DOC_PACKING = "turn"
# Trainer max_doc_num tail cap; 0 = uncapped.
MAX_DOCS = 16
# Trainer max_doc_length: per-document template-token cap, enforced by
# splitting oversized turn documents; 0 = no split.
MAX_DOC_LENGTH = 768
# Stats shape used when assembly itself failed and nothing was compressed.
_EMPTY_STATS = {
    "gist_tokens": 0,
    "original_tokens": 0,
    "n_docs": 0,
    "n_split": 0,
    "dropped_docs": 0,
}
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
    one ``<tool_call>{"name":...,"arguments":{...}}</tool_call>`` block per
    call, compact JSON separators, ``ensure_ascii=False``, with the OpenAI
    JSON-string ``arguments`` parsed back into an object first (the trainer's
    traces already carry objects there).

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
        raw_arguments = (
            function.get("arguments")
            or call.get("arguments")
            or call.get("args")
            or call.get("input")
            or {}
        )
        # OpenAI carries ``function.arguments`` as a JSON *string*; the traces
        # the trainer reads carry it as an object.  Parse so both render to
        # the same bytes -- otherwise every historical action reaches the gist
        # encoder as an escaped string literal.
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                arguments = raw_arguments
        else:
            arguments = raw_arguments
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


def _split_lines_keep(text: str) -> List[str]:
    """Line units with their newline kept (``_semantic_units`` analogue):
    splitting only at line boundaries keeps the turn markers and tool-call
    blocks intact."""
    units = text.splitlines(keepends=True)
    return units or [text]


def _fit_doc(doc_text: str, ratio: int, extract_fn, max_doc_length: int,
             depth: int = 0) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract ``doc_text``; if its template length exceeds ``max_doc_length``
    split it (``train_data_multiturn._split_message_to_fit``: greedy line
    accumulation against a char budget, then hard halves) and extract the
    pieces.  The proxy has no tokenizer, so the char budget is calibrated
    from the first extract's own chars/token; every piece is verified by its
    own extract response, so the length guarantee is exact and only the cut
    points are approximate.  ``max_doc_length <= 0`` disables splitting.
    """
    record = extract_fn("user", doc_text, ratio)
    length = int(record.get("original_seq_len") or 0)
    if max_doc_length <= 0 or length <= max_doc_length or depth >= 6 or len(doc_text) < 8:
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
    """``train_data_multiturn._select_history(policy="tail")``: keep doc 0
    (the session anchor, which carries the task statement) plus the last
    ``max_doc_num - 1`` docs; the rest are DROPPED and the model never sees
    them, exactly as in training."""
    if max_doc_num <= 0 or len(docs) <= max_doc_num:
        return list(docs), 0
    if max_doc_num == 1:
        return list(docs[-1:]), len(docs) - 1
    kept = [docs[0]] + list(docs[-(max_doc_num - 1):])
    return kept, len(docs) - len(kept)


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
    """Return ``(out_messages, stats)``.

    ``stats`` carries the segmentation regime and its outcome:
    ``gist_tokens``, ``original_tokens``, ``n_docs`` (documents actually
    referenced), ``n_split`` (turn documents that exceeded
    ``MAX_DOC_LENGTH`` and were cut into pieces) and ``dropped_docs``.

    ``DOC_PACKING`` selects the segment granularity handed to
    ``/v1/c2kv/extract``:

    * ``turn``    -- the trainer's turn documents (``_turn_documents``), each
      extracted with ``role="user"``.  Default: this is the packing every
      ``doc_mode=history_only`` checkpoint is trained on, so it is the only
      packing under which a serving number is a measurement of the checkpoint
      rather than of an unseen segment shape.
    * ``message`` -- one document per raw history message, keeping its own
      role.  Kept selectable so earlier matrices stay reproducible.

    Turn documents are then fitted to ``MAX_DOC_LENGTH`` template tokens
    (``_fit_doc``) and only afterwards tail-selected to ``MAX_DOCS``
    (``_select_docs``) -- the trainer's split-then-select order, so the cap
    counts chunks rather than turns.  Dropped documents are reported so a
    cell can never silently be scored on a truncated history.
    """
    cutoff = _history_cutoff(messages)
    gist_tokens = 0
    original_tokens = 0
    n_split = 0

    if not arm.compress_history:
        return list(messages), {
            "gist_tokens": 0,
            "original_tokens": 0,
            "n_docs": 0,
            "n_split": 0,
            "dropped_docs": 0,
        }

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

    # (role, text, record-or-None); turn documents are extracted while being
    # fitted, message documents are extracted below.
    docs: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    if DOC_PACKING == "turn":
        def extract_fn(role: str, text: str, ratio: int) -> Dict[str, Any]:
            return _extract(role, text, ratio, timeout)

        for text in _turn_documents(history):
            pieces = _fit_doc(text, arm.ratio, extract_fn, MAX_DOC_LENGTH)
            if len(pieces) > 1:
                n_split += 1
            docs.extend(("user", piece, record) for piece, record in pieces)
    else:
        for message in history:
            text = _message_doc_text(message)
            if not text:
                continue
            docs.append((message.get("role") or "user", text, None))

    docs, n_dropped = _select_docs(docs, MAX_DOCS)

    out: List[Dict[str, Any]] = list(head)
    for role, text, record in docs:
        if record is None:
            record = _extract(role, text, arm.ratio, timeout)
        gist_tokens += int(record.get("gist_len") or 0)
        original_tokens += int(record.get("original_seq_len") or 0)
        out.append({
            "role": role,
            "content": text,
            "c2kv_key_hash": record["key_hash"],
            # Provenance only.  The server does not read c2kv_ratio (it does
            # not appear anywhere in the pinned fork): a pool miss returns
            # "C2KV_CACHE_MISS: ..." and aborts, it never re-extracts.
            "c2kv_ratio": arm.ratio,
        })
    out.extend(tail)
    return out, {
        "gist_tokens": gist_tokens,
        "original_tokens": original_tokens,
        "n_docs": len(docs),
        "n_split": n_split,
        "dropped_docs": n_dropped,
    }


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
            messages, stats = _assemble(payload.get("messages") or [], ARM, 600)
        except (RuntimeError, URLError, OSError) as error:
            # A failed cell must be visible in the request log: without this
            # row an aborted task is indistinguishable from a task the model
            # simply got wrong.
            self._send_json(502, {"error": f"c2kv assembly failed: {error}"})
            self._log_request(
                payload, None, _EMPTY_STATS, time.perf_counter() - start,
                status="assembly_failed", error=str(error),
            )
            return
        assemble_sec = time.perf_counter() - start
        out_payload = dict(payload)
        out_payload["messages"] = messages
        try:
            data = _http_json(UPSTREAM, self.path, out_payload, 600)
        except (RuntimeError, URLError, OSError) as error:
            self._send_json(502, {"error": f"upstream failed: {error}"})
            self._log_request(
                payload, None, stats, time.perf_counter() - start,
                status="upstream_failed", error=str(error),
            )
            return
        total_sec = time.perf_counter() - start
        # Cost columns ride along on the response object.  TTFT is NOT
        # measured: this proxy is non-streaming (single buffered response).
        data.setdefault("c2kv_proxy", {})
        data["c2kv_proxy"].update(
            {
                "arm": ARM.name,
                "ratio": ARM.ratio,
                "gist_tokens": stats["gist_tokens"],
                "original_tokens": stats["original_tokens"],
                "n_gist_messages": stats["n_docs"],
                "doc_packing": DOC_PACKING,
                "max_docs": MAX_DOCS,
                "max_doc_length": MAX_DOC_LENGTH,
                "n_docs": stats["n_docs"],
                "n_split": stats["n_split"],
                "dropped_docs": stats["dropped_docs"],
                "assemble_sec": round(assemble_sec, 4),
                "wall_sec": round(total_sec, 4),
            }
        )
        self._send_json(200, data)
        self._log_request(payload, data, stats, total_sec)

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

    def _log_request(self, request, response, stats, total_sec,
                     status="ok", error=None):
        if not REQUEST_LOG_PATH:
            return
        response = response or {}
        row = {
            "ts": time.time(),
            "arm": ARM.name if ARM else None,
            "n_messages": len(request.get("messages") or []),
            "n_tools": len(request.get("tools") or []),
            "gist_tokens": stats["gist_tokens"],
            "original_tokens": stats["original_tokens"],
            "n_gist_messages": stats["n_docs"],
            "doc_packing": DOC_PACKING,
            "max_docs": MAX_DOCS,
            "max_doc_length": MAX_DOC_LENGTH,
            "n_docs": stats["n_docs"],
            "n_split": stats["n_split"],
            "dropped_docs": stats["dropped_docs"],
            "wall_sec": round(total_sec, 4),
            "status": status,
            "error": error,
            "usage": response.get("usage"),
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        }
        with _log_lock:
            with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    global ARM, UPSTREAM, REQUEST_LOG_PATH, DOC_PACKING, MAX_DOCS, MAX_DOC_LENGTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="SGLang base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--doc-packing", choices=("message", "turn"), default=DOC_PACKING,
        help=(
            "segment granularity for /v1/c2kv/extract. 'turn' (default) = the"
            " trainer's turn documents, i.e. the granularity every"
            " doc_mode=history_only checkpoint was trained on. 'message' = one"
            " document per raw history message."
        ),
    )
    parser.add_argument(
        "--max-docs", type=int, default=MAX_DOCS,
        help=(
            "cap on compressed history documents, mirroring the trainer's"
            " max_doc_num tail policy (doc 0 plus the newest ones are kept)."
            " 0 = uncapped."
        ),
    )
    parser.add_argument(
        "--max-doc-length", type=int, default=MAX_DOC_LENGTH,
        help=(
            "per-document template-token cap, mirroring the trainer's"
            " max_doc_length: oversized turn documents are split on line"
            " boundaries before the --max-docs cap. 0 = no split."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_docs < 0:
        parser.error("--max-docs must be >= 0")
    if args.max_doc_length < 0:
        parser.error("--max-doc-length must be >= 0")
    ARM = get_arm(args.arm)
    UPSTREAM = args.upstream.rstrip("/")
    REQUEST_LOG_PATH = args.request_log
    DOC_PACKING = args.doc_packing
    MAX_DOCS = args.max_docs
    MAX_DOC_LENGTH = args.max_doc_length
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(
        f"proxy arm={ARM.name} doc_packing={DOC_PACKING} max_docs={MAX_DOCS}"
        f" max_doc_length={MAX_DOC_LENGTH}"
        f" listening on {args.host}:{args.port} -> {UPSTREAM}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
