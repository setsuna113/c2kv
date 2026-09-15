"""Run four fixed-prefix requests through the real memory-runtime proxy.

This is a bounded development cost probe.  Every input is copied from the
completed capacity-dev2 request log; generated responses are recorded but never
fed into a later request, and no scorer is invoked.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import proxy
from arms import get_arm
from backends import get_backend
from memory_runtime.adapter import RuntimeAdapter


TASK_ID = "multi_turn_base_1"
BYTES_PER_KV_TOKEN = 147_456
HISTORY_BUDGET_BYTES = 113_246_208
WORKSPACE_BUDGET_BYTES = 113_246_208
MAXIMUM_WALL_SECONDS = 600
MAXIMUM_EXTRACT_HTTP_REQUESTS = 24
SAMPLING = {"temperature": 0.001, "seed": 0, "max_tokens": 512}
SOURCE_LOG = "capacity_protect/logs/proxy_c2kv4_38273.jsonl"
SOURCE_CELLS = (
    ("u2s1_within_budget", 2, 1, False, 10),
    ("u2s2_first_activation", 2, 2, True, 11),
    ("u2s2_idempotent_repeat", 2, 2, True, 11),
    ("u3s0_append", 3, 0, True, 12),
)
METHOD_FILES = (
    "benchmarks/arms.py",
    "benchmarks/backends/__init__.py",
    "benchmarks/backends/base.py",
    "benchmarks/backends/sglang.py",
    "benchmarks/memory_runtime/adapter.py",
    "benchmarks/memory_runtime/capacity.py",
    "benchmarks/memory_runtime/extraction_telemetry.py",
    "benchmarks/memory_runtime/policy.py",
    "benchmarks/memory_runtime/telemetry_probe.py",
    "benchmarks/memory_runtime/tokenization.py",
    "benchmarks/proxy.py",
    "python/history_memory/events.py",
    "python/history_memory/evidence.py",
    "python/history_memory/packing.py",
)
EXTRACT_RESPONSE_FIELDS = (
    "key_hash", "gist_len", "original_seq_len", "success", "error")


def save(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_method_bundle() -> dict[str, Any]:
    path = ROOT / "tmp/a_memory_runtime_20260907/source_bundle.json"
    bundle = _read_json(path)
    hashes = bundle.get("source_files_sha256")
    if (not isinstance(bundle.get("base_commit"), str)
            or bundle.get("source_tree_state") != "uncommitted_snapshot"
            or not isinstance(hashes, dict)):
        raise ValueError("Current source bundle lacks frozen snapshot provenance")
    mismatches = {}
    for name in METHOD_FILES:
        source = ROOT / name
        observed = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        expected = hashes.get(name)
        if observed != expected:
            mismatches[name] = {"bundle": expected, "current": observed}
    if mismatches:
        raise ValueError(f"Current method files differ from source bundle: {mismatches}")
    return {"path": str(path), **bundle, "verified_method_files": list(METHOD_FILES)}


def _load_captured_context_bundle(source_root: Path) -> dict[str, Any]:
    standalone = source_root / "source_bundle.json"
    if standalone.is_file():
        return {"path": str(standalone), **_read_json(standalone)}
    pilot_path = source_root / "pilot.json"
    pilot = _read_json(pilot_path)
    bundle = pilot.get("source_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("Source root has neither source_bundle.json nor pilot.json source_bundle")
    return {**copy.deepcopy(bundle), "loaded_from": f"{pilot_path}#source_bundle"}


def _load_source_rows(source_root: Path) -> list[dict[str, Any]]:
    path = source_root / SOURCE_LOG
    if not path.is_file():
        raise ValueError(f"Missing frozen capacity_protect request log: {path}")
    indexed = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            indexed.append((line_number, json.loads(line)))

    selected = []
    for label, user_turn, step, expected_gate, source_line in SOURCE_CELLS:
        matches = [item for item in indexed
                   if item[0] == source_line
                   and (item[1].get("eval_context") or {}).get("task_id") == TASK_ID
                   and (item[1].get("eval_context") or {}).get("user_turn") == user_turn
                   and (item[1].get("eval_context") or {}).get("step") == step
                   and (item[1].get("eval_context") or {}).get("attempt") == 0]
        if len(matches) != 1:
            raise ValueError(
                f"Frozen line {source_line} is not {TASK_ID} u{user_turn}s{step}")
        line_number, row = matches[0]
        gate = ((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
            "compression_activated")
        budget = ((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
            "history_budget_bytes")
        if row.get("status") != "ok" or gate is not expected_gate or budget != HISTORY_BUDGET_BYTES:
            raise ValueError(
                f"Frozen source gate mismatch for {label}: status={row.get('status')!r}, "
                f"gate={gate!r}, budget={budget!r}")
        if not isinstance(row.get("request_view"), dict):
            raise ValueError(f"Frozen source row lacks request_view: {label}")
        selected.append({
            "label": label,
            "user_turn": user_turn,
            "step": step,
            "expected_compression_activated": expected_gate,
            "source_path": SOURCE_LOG,
            "source_line": line_number,
            "row": row,
        })

    first_activated = next(
        ((row.get("eval_context") or {}).get("user_turn"),
         (row.get("eval_context") or {}).get("step"))
        for _, row in indexed
        if (row.get("eval_context") or {}).get("task_id") == TASK_ID
        and (row.get("eval_context") or {}).get("attempt") == 0
        and ((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
            "compression_activated") is True)
    if first_activated != (2, 2):
        raise ValueError(f"Frozen source first activation changed: {first_activated!r}")
    return selected


def _request_body(source: dict[str, Any], run_id: str) -> dict[str, Any]:
    row = source["row"]
    view = row["request_view"]
    body = {key: copy.deepcopy(view[key])
            for key in ("model", "messages", "tools") if key in view}
    if "model" not in body or not isinstance(body.get("messages"), list):
        raise ValueError(f"Invalid captured request view: {source['label']}")
    body.update(SAMPLING)
    context = copy.deepcopy(row.get("eval_context") or {})
    context.pop("run_id", None)
    context["run_id"] = run_id
    body["c2kv_eval_context"] = context
    return body


def _read_request_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit("Output exists; automatic rerun/resume is prohibited")
    args.out.mkdir(parents=True)
    rows_dir = args.out / "rows"
    rows_dir.mkdir()
    receipt_path = args.out / "receipt.json"
    request_log_path = args.out / "proxy_requests.jsonl"
    transport_path = args.out / "transport.jsonl"
    run_id = args.out.name
    started = time.monotonic()
    deadline = started + MAXIMUM_WALL_SECONDS
    receipt: dict[str, Any] = {
        "schema": "a-runtime-telemetry-probe-v1",
        "status": "preparing_freeze",
        "scope": "four fixed observable prefixes; development cost telemetry; preliminary, n=1",
        "run_id": run_id,
        "source_root": str(args.source_root.resolve()),
        "upstream": args.upstream,
        "checkpoint": args.checkpoint,
        "arm": "c2kv4",
        "runtime_mode": "capacity_protect",
        "sampling": dict(SAMPLING),
        "profile": {
            "query_projection": "base", "doc_packing": "turn",
            "max_doc_length": 512, "max_doc_num": 12,
            "bytes_per_kv_token": BYTES_PER_KV_TOKEN,
            "history_budget_bytes": HISTORY_BUDGET_BYTES,
            "workspace_budget_bytes": WORKSPACE_BUDGET_BYTES,
        },
        "execution": "single client, four sequential HTTP requests",
        "response_feedback": False,
        "scorer_calls": 0,
        "maximum_chat_requests": 4,
        "maximum_extract_http_requests": MAXIMUM_EXTRACT_HTTP_REQUESTS,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "transport_retries": 0,
        "cache_miss_retries": 0,
        "automatic_reruns": 0,
        "chat_attempts": 0,
        "chat_completed": 0,
        "extract_http_attempts": 0,
        "actual_transport_attempts_by_path": {},
        "request_log_path": str(request_log_path),
        "transport_log_path": str(transport_path),
        "rows": [],
    }
    save(receipt_path, receipt)

    server = None
    server_thread = None
    receipt_lock = threading.Lock()
    transport_lock = threading.Lock()
    handler_finished = threading.Semaphore(0)
    transport_counts: Counter[str] = Counter()
    delegate_original = proxy._post_json
    proxy_names = (
        "UPSTREAM", "QUERY_PROJECTION", "DOC_PACKING", "MAX_DOC_LENGTH",
        "MAX_DOC_NUM", "NO_UPSTREAM_RETRIES", "CAPTURE_REQUEST_VIEWS",
        "CACHE", "ARM", "BACKEND", "MEMORY_RUNTIME",
        "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", "MEMORY_RUNTIME_FATAL_ERROR",
        "REQUEST_LOG_PATH", "_post_json")
    proxy_before = {name: getattr(proxy, name) for name in proxy_names}

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Frozen 600-second wall budget exhausted")
        return value

    def save_receipt() -> None:
        with receipt_lock:
            receipt["actual_transport_attempts_by_path"] = dict(transport_counts)
            receipt["extract_http_attempts"] = transport_counts["/v1/c2kv/extract"]
            save(receipt_path, receipt)

    try:
        source_root = args.source_root.resolve()
        source_rows = _load_source_rows(source_root)
        method_bundle = _load_method_bundle()
        captured_bundle = _load_captured_context_bundle(source_root)
        row_paths = [rows_dir / f"{index:02d}_{source['label']}.json"
                     for index, source in enumerate(source_rows)]
        receipt.update(
            source_bundle=method_bundle,
            captured_context_source_bundle=captured_bundle,
            fixed_sources=[{
                key: source[key] for key in (
                    "label", "user_turn", "step",
                    "expected_compression_activated", "source_path", "source_line")}
                for source in source_rows],
            rows=[str(path) for path in row_paths],
        )

        config = _read_json(HERE / "configs/protect.json")
        config.update(
            mode="capacity_protect", run_id=run_id,
            bytes_per_kv_token=BYTES_PER_KV_TOKEN,
            history_budget_bytes=HISTORY_BUDGET_BYTES,
            workspace_budget_bytes=WORKSPACE_BUDGET_BYTES)
        config_path = args.out / "runtime_config.json"
        save(config_path, config)
        receipt["runtime_config_path"] = str(config_path)
        receipt["status"] = "frozen_before_first_live_request"
        save_receipt()

        runtime = RuntimeAdapter.from_config(str(config_path), args.checkpoint)

        # Save the true network delegate before replacing proxy._post_json.  The
        # backend captures tracked_post for extraction, while ProxyHandler's
        # send_upstream resolves proxy._post_json dynamically for chat.
        def tracked_post(path: str, payload: dict[str, Any], timeout: int,
                         retries: int = 0) -> dict[str, Any]:
            del retries
            network_timeout = min(float(timeout), remaining())
            limits = {
                "/v1/chat/completions": 4,
                "/v1/c2kv/extract": MAXIMUM_EXTRACT_HTTP_REQUESTS,
            }
            if path not in limits:
                receipt.setdefault("route_rejections", []).append(path)
                save_receipt()
                raise RuntimeError(f"POST route is outside the frozen probe: {path}")
            if transport_counts[path] >= limits[path]:
                receipt.setdefault("transport_cap_rejections", {})[path] = int(
                    receipt.get("transport_cap_rejections", {}).get(path, 0)) + 1
                save_receipt()
                raise RuntimeError(f"Transport cap reached before network call: {path}")
            transport_counts[path] += 1
            attempt = transport_counts[path]
            save_receipt()
            began = time.monotonic()
            record: dict[str, Any] = {
                "path": path, "path_attempt": attempt,
                "probe_row_path": receipt.get("current_row"), "status": "started"}
            try:
                result = delegate_original(
                    path, payload, timeout=network_timeout, retries=0)
                if path == "/v1/c2kv/extract":
                    record["response"] = {
                        key: copy.deepcopy(result.get(key))
                        for key in EXTRACT_RESPONSE_FIELDS}
                elif path.endswith("/chat/completions"):
                    record["response"] = {
                        "usage": copy.deepcopy(result.get("usage"))}
                record["status"] = "completed"
                return result
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                record["wall_seconds"] = time.monotonic() - began
                with transport_lock:
                    with transport_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        proxy.UPSTREAM = args.upstream.rstrip("/")
        proxy.QUERY_PROJECTION = "base"
        proxy.DOC_PACKING = "turn"
        proxy.MAX_DOC_LENGTH = 512
        proxy.MAX_DOC_NUM = 12
        proxy.NO_UPSTREAM_RETRIES = True
        proxy.CAPTURE_REQUEST_VIEWS = True
        proxy.CACHE = proxy.ExtractCache()
        proxy.ARM = get_arm("c2kv4")
        proxy.MEMORY_RUNTIME = runtime
        proxy.MEMORY_RUNTIME_BYTES_PER_KV_TOKEN = None
        proxy.MEMORY_RUNTIME_FATAL_ERROR = None
        proxy.REQUEST_LOG_PATH = str(request_log_path)
        proxy.BACKEND = get_backend("sglang", tracked_post)
        proxy._post_json = tracked_post

        class LoggedProxyHandler(proxy.ProxyHandler):
            def do_POST(self) -> None:
                try:
                    super().do_POST()
                finally:
                    handler_finished.release()

        server = ThreadingHTTPServer(("127.0.0.1", 0), LoggedProxyHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(
            target=server.serve_forever, name="telemetry-proxy", daemon=True)
        server_thread.start()
        local_url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        local_opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))

        receipt["status"] = "running"
        save_receipt()
        for index, (source, row_path) in enumerate(zip(source_rows, row_paths)):
            body = _request_body(source, run_id)
            row: dict[str, Any] = {
                "schema": "a-runtime-telemetry-probe-row-v1",
                "index": index,
                "label": source["label"],
                "source": {key: source[key] for key in (
                    "source_path", "source_line", "user_turn", "step",
                    "expected_compression_activated")},
                "eval_context": copy.deepcopy(body["c2kv_eval_context"]),
                "request_view": proxy._captured_request_view(body),
                "status": "frozen_before_request",
            }
            save(row_path, row)
            receipt["chat_attempts"] += 1
            receipt["current_row"] = str(row_path)
            save_receipt()
            began = time.monotonic()
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request = urlrequest.Request(
                local_url, data=encoded,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with local_opener.open(request, timeout=remaining()) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                response_body = error.read().decode("utf-8", errors="replace")
                try:
                    row["response"] = json.loads(response_body)
                except json.JSONDecodeError:
                    row["response"] = {"error": "non-JSON proxy error response"}
                row.update(status="failed", http_status=error.code,
                           wall_seconds=time.monotonic() - began)
                save(row_path, row)
                raise RuntimeError(f"Proxy request failed with HTTP {error.code}") from error
            finally:
                if not handler_finished.acquire(timeout=remaining()):
                    raise TimeoutError("Proxy handler did not finish within the wall budget")
            remaining()
            row.update(
                status="completed", http_status=response.status,
                wall_seconds=time.monotonic() - began, response=result)
            save(row_path, row)
            receipt["chat_completed"] += 1
            save_receipt()

        request_rows = _read_request_log(request_log_path)
        if len(request_rows) != 4 or any(row.get("status") != "ok" for row in request_rows):
            raise ValueError("Proxy request log does not contain four successful rows")
        gates = [((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
            "compression_activated") for row in request_rows]
        if gates != [False, True, True, True]:
            raise ValueError(f"Live capacity gates differ from frozen contexts: {gates!r}")
        first_repeat, second_repeat = request_rows[1], request_rows[2]
        repeat_fields = ("decision_id", "selected_event_ids", "protected_event_ids",
                         "retrieved_event_ids", "retained_event_ids", "policy")
        first_memory = first_repeat.get("memory_runtime") or {}
        second_memory = second_repeat.get("memory_runtime") or {}
        if any(first_memory.get(key) != second_memory.get(key) for key in repeat_fields):
            raise ValueError("Repeated u2s2 decision was not idempotent")
        telemetry = [row.get("extraction_telemetry") or {} for row in request_rows]
        producer_calls = sum(
            int((item.get("summary") or {}).get("producer_calls") or 0)
            for item in telemetry)
        extract_attempts = transport_counts["/v1/c2kv/extract"]
        if producer_calls != extract_attempts:
            raise ValueError(
                "Handler producer_calls disagree with tracked extraction HTTP attempts: "
                f"{producer_calls} != {extract_attempts}")
        if (receipt["chat_attempts"] != 4 or receipt["chat_completed"] != 4
                or transport_counts["/v1/chat/completions"] != 4):
            raise ValueError("Four-chat transport contract did not complete")
        receipt.update(
            status="completed",
            current_row=None,
            proxy_request_log_rows=len(request_rows),
            handler_extraction_summary={
                "lookups": sum(int((item.get("summary") or {}).get("lookups") or 0)
                               for item in telemetry),
                "client_cache_hits": sum(int((item.get("summary") or {}).get(
                    "client_cache_hits") or 0) for item in telemetry),
                "producer_calls": producer_calls,
                "producer_successes": sum(int((item.get("summary") or {}).get(
                    "producer_successes") or 0) for item in telemetry),
                "producer_failures": sum(int((item.get("summary") or {}).get(
                    "producer_failures") or 0) for item in telemetry),
            },
            transport_crosscheck="handler producer_calls == tracked /v1/c2kv/extract attempts",
        )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        for name, value in proxy_before.items():
            setattr(proxy, name, value)
        receipt["total_wall_seconds"] = time.monotonic() - started
        save_receipt()
        print(json.dumps({key: receipt.get(key) for key in (
            "status", "chat_attempts", "chat_completed", "extract_http_attempts")}))


if __name__ == "__main__":
    main()
