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
    """Return (out_messages, gist_tokens, original_tokens, n_gist)."""
    cutoff = _history_cutoff(messages)
    out: List[Dict[str, Any]] = []
    gist_tokens = 0
    original_tokens = 0
    n_gist = 0
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
            out.append(message)
            continue
        record = _extract(role, content, arm.ratio, timeout)
        gist_tokens += int(record.get("gist_len") or 0)
        original_tokens += int(record.get("original_seq_len") or 0)
        n_gist += 1
        compressed = dict(message)
        compressed["content"] = content
        compressed["c2kv_key_hash"] = record["key_hash"]
        # lets the server re-extract on cache miss (e.g. after a restart)
        compressed["c2kv_ratio"] = arm.ratio
        out.append(compressed)
    return out, gist_tokens, original_tokens, n_gist


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
        start = time.perf_counter()
        try:
            messages, gist_tokens, original_tokens, n_gist = _assemble(
                payload.get("messages") or [], ARM, 600
            )
        except (RuntimeError, URLError, OSError) as error:
            self._send_json(502, {"error": f"c2kv assembly failed: {error}"})
            return
        assemble_sec = time.perf_counter() - start
        out_payload = dict(payload)
        out_payload["messages"] = messages
        if ARM.constrain_tools:
            out_payload["constrain_tools"] = True
        if ARM.repair:
            # repair arm (docs/hybrid_spec.md): hf_server resolves the policy
            # over the compressed chunks it assembles (it owns chunking and
            # the logical ledger)
            out_payload["c2kv_repair"] = dict(ARM.repair)
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
                "assemble_sec": round(assemble_sec, 4),
                "wall_sec": round(total_sec, 4),
            }
        )
        # surface the server-side repair cost columns when present
        upstream_c2kv = data.get("c2kv") or {}
        for key in ("repair_policy", "repair_block_tokens",
                    "repair_doc_index", "repair_prefill_sec"):
            if key in upstream_c2kv:
                data["c2kv_proxy"][key] = upstream_c2kv[key]
        self._send_json(200, data)
        self._log_request(payload, data, gist_tokens, original_tokens, n_gist, total_sec)

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

    def _log_request(self, request, response, gist_tokens, original_tokens, n_gist, total_sec):
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
            "wall_sec": round(total_sec, 4),
            "status": "ok",
            "usage": response.get("usage"),
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        }
        upstream_c2kv = response.get("c2kv") or {}
        for key in ("repair_policy", "repair_block_tokens",
                    "repair_doc_index", "repair_prefill_sec"):
            if key in upstream_c2kv:
                row[key] = upstream_c2kv[key]
        with _log_lock:
            with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    global ARM, UPSTREAM, REQUEST_LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="SGLang base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", default="")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    ARM = get_arm(args.arm)
    UPSTREAM = args.upstream.rstrip("/")
    REQUEST_LOG_PATH = args.request_log
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"proxy arm={ARM.name} listening on {args.host}:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
