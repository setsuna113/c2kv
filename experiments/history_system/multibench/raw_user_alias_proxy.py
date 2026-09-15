#!/usr/bin/env python3
"""Single-hop model-alias proxy for the frozen full-history user simulator.

The proxy deliberately performs one semantic request transformation: it
replaces the accepted public model alias with the upstream served model name.
All other request fields, including messages, tools, and sampling parameters,
are forwarded unchanged.  Upstream HTTP requests use a proxy-free opener and
are attempted exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit


SCHEMA = "history-system-raw-user-alias-v1"
_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))
_LOG_LOCK = threading.Lock()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _models_url(upstream: str) -> str:
    return upstream.rstrip("/") + "/models"


def _chat_url(upstream: str) -> str:
    return upstream.rstrip("/") + "/chat/completions"


def fetch_models(upstream: str, timeout: float) -> dict[str, Any]:
    req = urlrequest.Request(_models_url(upstream), method="GET")
    with _OPENER.open(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError("upstream /models did not return an OpenAI model list")
    return data


def has_model(models: dict[str, Any], model: str) -> bool:
    return any(isinstance(row, dict) and row.get("id") == model for row in models["data"])


class AliasServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *,
                 upstream: str, upstream_model: str, aliases: tuple[str, ...],
                 request_log: Path, timeout: float):
        super().__init__(address, handler)
        self.upstream = upstream
        self.upstream_model = upstream_model
        self.aliases = aliases
        self.accepted_models = frozenset((*aliases, upstream_model))
        self.request_log = request_log
        self.timeout = timeout


class AliasHandler(BaseHTTPRequestHandler):
    server: AliasServer
    protocol_version = "HTTP/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, status: int, value: Any) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        if path in ("/models", "/v1/models"):
            data = [
                {"id": model, "object": "model", "owned_by": "c2kv-local"}
                for model in (*self.server.aliases, self.server.upstream_model)
            ]
            self._send_json(HTTPStatus.OK, {"object": "list", "data": data})
            return
        if path == "/health":
            try:
                models = fetch_models(self.server.upstream, min(self.server.timeout, 10.0))
                healthy = has_model(models, self.server.upstream_model)
            except Exception:
                healthy = False
            self._send_json(
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "ok" if healthy else "upstream_unavailable", "schema": SCHEMA},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.rstrip("/") not in (
            "/chat/completions", "/v1/chat/completions"
        ):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0 or length > 64 * 1024 * 1024:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        raw = self.rfile.read(length)
        started = time.monotonic()
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            requested_model = payload.get("model")
            if requested_model not in self.server.accepted_models:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unsupported_model_alias"})
                return
            forwarded = dict(payload)
            forwarded["model"] = self.server.upstream_model
            forwarded_raw = _json_bytes(forwarded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json_request"})
            return

        req_headers = {"Content-Type": "application/json"}
        authorization = self.headers.get("Authorization")
        if authorization:
            req_headers["Authorization"] = authorization
        req = urlrequest.Request(
            _chat_url(self.server.upstream), data=forwarded_raw,
            headers=req_headers, method="POST",
        )
        status = HTTPStatus.BAD_GATEWAY
        response_bytes = 0
        response_hash = hashlib.sha256()
        try:
            # One call, with a proxy-free opener and no retry loop.
            response = _OPENER.open(req, timeout=self.server.timeout)
        except urlerror.HTTPError as exc:
            response = exc
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": type(exc).__name__})
            self._append_log(
                requested_model, raw, forwarded_raw, int(status), response_bytes,
                response_hash.hexdigest(), started,
            )
            return

        try:
            status = response.status
            self.send_response(status)
            content_type = response.headers.get("Content-Type")
            if content_type:
                self.send_header("Content-Type", content_type)
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                response_bytes += len(chunk)
                response_hash.update(chunk)
                self.wfile.write(chunk)
        finally:
            response.close()
            self._append_log(
                requested_model, raw, forwarded_raw, int(status), response_bytes,
                response_hash.hexdigest(), started,
            )

    def _append_log(self, requested_model: str, raw: bytes, forwarded_raw: bytes,
                    status: int, response_bytes: int, response_sha256: str,
                    started: float) -> None:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = {}
        row = {
            "schema": SCHEMA,
            "request_sha256": _sha256(raw),
            "forwarded_sha256": _sha256(forwarded_raw),
            "response_sha256": response_sha256,
            "requested_model": requested_model,
            "upstream_model": self.server.upstream_model,
            "message_count": len(parsed.get("messages", []))
            if isinstance(parsed.get("messages"), list) else None,
            "tool_count": len(parsed.get("tools", []))
            if isinstance(parsed.get("tools"), list) else None,
            "request_bytes": len(raw),
            "forwarded_bytes": len(forwarded_raw),
            "response_bytes": response_bytes,
            "status": status,
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
        }
        line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with _LOG_LOCK:
            with self.server.request_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True,
                        help="OpenAI base URL ending in /v1")
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--alias", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("FATAL: alias service must bind loopback only")
    aliases = tuple(dict.fromkeys(args.alias))
    if not aliases or any(not value for value in aliases):
        raise SystemExit("FATAL: at least one non-empty alias is required")
    args.out.mkdir(parents=True, exist_ok=False)
    request_log = args.out / "requests.jsonl"
    request_log.touch(exist_ok=False)
    models = fetch_models(args.upstream, 10.0)
    if not has_model(models, args.upstream_model):
        raise SystemExit(f"FATAL: upstream does not serve {args.upstream_model!r}")
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path.read_bytes())
    server = AliasServer(
        (args.host, args.port), AliasHandler, upstream=args.upstream,
        upstream_model=args.upstream_model, aliases=aliases,
        request_log=request_log, timeout=args.timeout,
    )
    ready = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "bind": f"http://{args.host}:{args.port}",
        "upstream": args.upstream,
        "upstream_model": args.upstream_model,
        "accepted_aliases": list(aliases),
        "payload_transform": {"only": "model", "value": args.upstream_model},
        "upstream_attempts_per_request": 1,
        "proxy_environment_used_for_upstream": False,
        "log_fields_only": [
            "hashes", "counts", "byte_counts", "model_names", "status", "latency_ms"
        ],
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "request_log": str(request_log),
    }
    _write_json(args.out / "ready.json", ready)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
