"""Finite profiled HTTP probes of exact-source controllers on real prefixes.

Model responses are recorded but never executed or fed back. These profiles
measure integration and observed trigger coverage, not task success or recovery
effectiveness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from backends import get_backend
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.telemetry_probe import (
    BYTES_PER_KV_TOKEN, HISTORY_BUDGET_BYTES, WORKSPACE_BUDGET_BYTES,
    SAMPLING, METHOD_FILES, _load_source_rows, _request_body, _read_request_log, save,
)

DEFAULT_PROFILE = "exact-recovery-v1"
BASE_METHOD_FILES = frozenset(METHOD_FILES) | {
    "benchmarks/memory_runtime/exact_gap.py",
    "benchmarks/memory_runtime/exact_policy.py",
    "benchmarks/memory_runtime/exact_probe.py",
}
PROFILES = {
    "exact-recovery-v1": {
        "modes": ("capacity_exact_once", "capacity_exact_persistent"),
        "source_indices": (0, 1),
        "arm": "c2kv4",
        "scope": (
            "real captured fixed prefixes; integration and trigger coverage only; "
            "preliminary, n=1"
        ),
        "method_files": BASE_METHOD_FILES,
        "config_files": {},
        "limits": {
            "maximum_proxy_requests": 4,
            "maximum_generation_attempts": 8,
            "maximum_extraction_attempts": 24,
            "maximum_wall_seconds": 600,
            "max_regenerations_per_decision": 1,
            "automatic_retries": 0,
            "automatic_reruns": 0,
        },
    },
    "exact-controls-v1": {
        "modes": ("full_exact_shared", "capacity_exact_no_gist"),
        "source_indices": (1,),
        "arm": "full",
        "scope": (
            "one predeclared above-capacity prefix per exact control; integration "
            "and trigger coverage only; preliminary, n=1"
        ),
        "method_files": BASE_METHOD_FILES | {
            "benchmarks/memory_runtime/exact_raw.py",
            "benchmarks/memory_runtime/configs/full_exact_shared.json",
            "benchmarks/memory_runtime/configs/capacity_exact_no_gist.json",
        },
        "config_files": {
            "full_exact_shared": "benchmarks/memory_runtime/configs/full_exact_shared.json",
            "capacity_exact_no_gist": (
                "benchmarks/memory_runtime/configs/capacity_exact_no_gist.json"
            ),
        },
        "limits": {
            "maximum_proxy_requests": 2,
            "maximum_generation_attempts": 4,
            "maximum_extraction_attempts": 0,
            "maximum_wall_seconds": 600,
            "max_regenerations_per_decision": 1,
            "automatic_retries": 0,
            "automatic_reruns": 0,
        },
    },
}


def _load_method_bundle(method_files):
    path = ROOT / "tmp/a_memory_runtime_20260907/source_bundle.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    hashes = bundle.get("source_files_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Frozen source bundle lacks source_files_sha256")
    observed = {}
    mismatches = {}
    for name in sorted(method_files):
        source = ROOT / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        observed[name] = digest
        if digest != hashes.get(name):
            mismatches[name] = {"bundle": hashes.get(name), "current": digest}
    if mismatches:
        raise ValueError(f"Method differs from frozen bundle: {mismatches}")
    manifest = {
        "source_bundle_path": str(path),
        "verified_files": sorted(method_files),
        "source_files_sha256": observed,
    }
    return bundle, manifest


def _runtime_config(profile, mode, run_id):
    config_path = profile["config_files"].get(mode)
    if config_path is None:
        return {
            "mode": mode,
            "run_id": run_id,
            "bytes_per_kv_token": BYTES_PER_KV_TOKEN,
            "history_budget_bytes": HISTORY_BUDGET_BYTES,
            "workspace_budget_bytes": WORKSPACE_BUDGET_BYTES,
            "lease_decisions": 3,
            "max_retrieved_events": 1,
        }
    config = json.loads((ROOT / config_path).read_text(encoding="utf-8"))
    expected = {
        "mode": mode,
        "bytes_per_kv_token": BYTES_PER_KV_TOKEN,
        "history_budget_bytes": HISTORY_BUDGET_BYTES,
        "workspace_budget_bytes": WORKSPACE_BUDGET_BYTES,
        "lease_decisions": 3,
        "max_retrieved_events": 1,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Frozen control config differs from probe contract: {config_path}")
    config["run_id"] = run_id
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE,
        help=f"bounded probe profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; automatic rerun/resume is prohibited")
    profile = PROFILES[args.profile]
    limits = dict(profile["limits"])
    bundle, method_manifest = _load_method_bundle(profile["method_files"])
    all_sources = _load_source_rows(args.source_root.resolve())
    sources = [all_sources[index] for index in profile["source_indices"]]
    modes = profile["modes"]
    cells = [(mode, source) for mode in modes for source in sources]
    if len(cells) != limits["maximum_proxy_requests"]:
        raise ValueError("Profile request count differs from its frozen limit")
    if limits["maximum_generation_attempts"] != len(cells) * (
        1 + limits["max_regenerations_per_decision"]
    ):
        raise ValueError(
            "Profile generation limit does not cover exactly one bounded regeneration"
        )
    if any(limits[key] != 0 for key in ("automatic_retries", "automatic_reruns")):
        raise ValueError("Exact probes prohibit automatic retries and reruns")
    if args.profile == "exact-controls-v1" and (
        len(sources) != 1 or sources[0]["source_line"] != 11
        or not sources[0]["expected_compression_activated"]
    ):
        raise ValueError("Control profile requires the predeclared capacity_protect line 11")
    args.out.mkdir(parents=True)
    run_id = args.out.name
    configs = {mode: _runtime_config(profile, mode, run_id) for mode in modes}
    selected_sources = [
        {key: source[key] for key in (
            "label", "source_path", "source_line", "user_turn", "step",
            "expected_compression_activated",
        )}
        for source in sources
    ]
    receipt = {
        "schema": "a-runtime-exact-probe-v2",
        "status": "frozen_before_first_live_request",
        "profile": args.profile,
        "scope": profile["scope"],
        "run_id": run_id,
        "source_bundle": bundle, "source_root": str(args.source_root.resolve()),
        "method_manifest": method_manifest,
        "upstream": args.upstream, "checkpoint": args.checkpoint,
        "arm": profile["arm"],
        "sampling": dict(SAMPLING),
        "renderer": {"query_projection": "base", "doc_packing": "turn",
                     "max_doc_length": 512, "max_doc_num": 12},
        "selected_sources": selected_sources,
        "configs": configs,
        "limits": limits,
        "response_feedback": False, "tool_executions": 0, "scorer_calls": 0,
        "counts": {
            "proxy_requests_started": 0,
            "proxy_requests_completed": 0,
            "transport_attempts_by_path": {},
        },
    }
    receipt_path = args.out / "receipt.json"
    save(receipt_path, receipt)
    for mode, config in configs.items():
        save(args.out / f"{mode}.json", config)
    started = time.monotonic()
    deadline = started + limits["maximum_wall_seconds"]
    transport = Counter()
    server = None
    server_thread = None
    handler_finished = threading.Semaphore(0)
    original_post = proxy._post_json

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Frozen wall budget exhausted")
        return value

    def tracked_post(path, payload, timeout, retries=0):
        route_limits = {
            "/v1/chat/completions": limits["maximum_generation_attempts"],
            "/v1/c2kv/extract": limits["maximum_extraction_attempts"],
        }
        if path not in route_limits or transport[path] >= route_limits[path]:
            raise RuntimeError(f"Frozen route/count cap rejected before network call: {path}")
        allowed_time = min(float(timeout), remaining())
        transport[path] += 1
        receipt["counts"]["transport_attempts_by_path"] = dict(transport)
        save(receipt_path, receipt)
        began = time.monotonic()
        record = {"path": path, "path_attempt": transport[path],
                  "cell": receipt.get("current_cell"), "status": "started"}
        try:
            result = original_post(path, payload, timeout=allowed_time, retries=0)
            record.update(status="completed", usage=result.get("usage"))
            return result
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - began
            with (args.out / "transport.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        prototype = RuntimeAdapter.from_config(str(args.out / f"{modes[0]}.json"), args.checkpoint)
        runtimes = {mode: RuntimeAdapter(config, prototype._token_counter)
                    for mode, config in configs.items()}
        proxy.UPSTREAM = args.upstream.rstrip("/")
        proxy.QUERY_PROJECTION, proxy.DOC_PACKING = "base", "turn"
        proxy.MAX_DOC_LENGTH, proxy.MAX_DOC_NUM = 512, 12
        proxy.NO_UPSTREAM_RETRIES = proxy.CAPTURE_REQUEST_VIEWS = True
        proxy.ARM = get_arm(profile["arm"])
        proxy.CACHE = proxy.ExtractCache()
        proxy.STATE = proxy.ProxyState()
        proxy.MEMORY_RUNTIME_BYTES_PER_KV_TOKEN = proxy.MEMORY_RUNTIME_FATAL_ERROR = None
        proxy.REQUEST_LOG_PATH = str(args.out / "proxy_requests.jsonl")
        proxy.BACKEND = get_backend("sglang", tracked_post)
        proxy._post_json = tracked_post

        class Handler(proxy.ProxyHandler):
            def do_POST(self):
                try:
                    super().do_POST()
                finally:
                    handler_finished.release()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
        receipt["status"] = "running"
        save(receipt_path, receipt)
        for mode, source in cells:
            proxy.MEMORY_RUNTIME = runtimes[mode]
            cell = f"{mode}_{source['label']}"
            receipt["current_cell"] = cell
            body = _request_body(source, run_id)
            row = {
                "cell": cell, "profile": args.profile, "mode": mode,
                "source": {key: source[key] for key in (
                    "label", "source_path", "source_line", "user_turn", "step")},
                "config": configs[mode], "limits": limits,
                "status": "frozen_before_request", "request": body,
            }
            row_path = args.out / f"{cell}.json"
            save(row_path, row)
            if (receipt["counts"]["proxy_requests_started"]
                    >= limits["maximum_proxy_requests"]):
                raise RuntimeError("Frozen proxy request cap rejected before network call")
            receipt["counts"]["proxy_requests_started"] += 1
            save(receipt_path, receipt)
            request = urlrequest.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with opener.open(request, timeout=remaining()) as response:
                    result = json.loads(response.read())
            except HTTPError as error:
                row.update(status="failed", response=json.loads(error.read()),
                           http_status=error.code)
                save(row_path, row)
                raise RuntimeError(f"Proxy failed with HTTP {error.code}") from error
            finally:
                if not handler_finished.acquire(timeout=remaining()):
                    raise TimeoutError("Proxy handler did not finish")
            row.update(status="completed", response=result)
            save(row_path, row)
            receipt["counts"]["proxy_requests_completed"] += 1
            metadata = result["c2kv_proxy"]
            runtime_metadata = metadata["memory_runtime"]
            if metadata.get("c2kv_query_proj_effective") != "base":
                raise ValueError("Unexpected effective query projection")
            gate_key, activated_key = (
                ("auxiliary_gate", "auxiliary_activated")
                if mode == "full_exact_shared"
                else ("capacity_gate", "compression_activated")
            )
            if (runtime_metadata[gate_key][activated_key]
                    != source["expected_compression_activated"]):
                raise ValueError("Capacity gate differs from frozen source context")
            if metadata["generation_attempts"] > 1 + limits["max_regenerations_per_decision"]:
                raise ValueError("Per-decision regeneration limit exceeded")
            if args.profile == "exact-controls-v1":
                if runtime_metadata["gist_tokens"] != 0:
                    raise ValueError("Exact control unexpectedly retained gist")
                expected_budget_applies = mode == "capacity_exact_no_gist"
                if runtime_metadata["budget_applies"] is not expected_budget_applies:
                    raise ValueError("Exact control budget_applies differs from its mode")
                if (runtime_metadata["evidence_bytes"]
                        > configs[mode]["workspace_budget_bytes"]):
                    raise ValueError("Exact control actual evidence exceeds W")
                if (mode == "capacity_exact_no_gist"
                        and runtime_metadata["active_history_bytes"]
                        > configs[mode]["history_budget_bytes"]):
                    raise ValueError("NoGist exact control exceeds B")
            save(receipt_path, receipt)
        rows = _read_request_log(Path(proxy.REQUEST_LOG_PATH))
        if (len(rows) != limits["maximum_proxy_requests"]
                or any(row["status"] != "ok" for row in rows)):
            raise ValueError("Profile requires every frozen proxy row to succeed")
        if sum(row["generation_attempts"] for row in rows) != transport["/v1/chat/completions"]:
            raise ValueError("Generation trace disagrees with actual transport count")
        if args.profile == "exact-controls-v1":
            if transport["/v1/c2kv/extract"] != 0 or any(
                row["extraction_telemetry"]["summary"]["lookups"] != 0
                for row in rows
            ):
                raise ValueError("Exact controls must perform zero extraction")
            for (mode, source), logged in zip(cells, rows):
                if mode != "full_exact_shared":
                    continue
                source_messages = source["row"]["request_view"]["messages"]
                expected_full, _ = proxy._assemble(source_messages, get_arm("full"))
                forwarded = logged["forwarded_request_views"]
                generations = logged["generation_trace"]
                if len(forwarded) != len(generations):
                    raise ValueError("Full control trace/view count mismatch")
                for view, generation in zip(forwarded, generations):
                    actual = view["messages"]
                    index = generation["memory_runtime"]["evidence_out_index"]
                    without_evidence = actual if index is None else (
                        actual[:index] + actual[index + 1:]
                    )
                    if without_evidence != expected_full:
                        raise ValueError("Full exact control did not preserve Full source")
        generation_records = [
            record for row in rows for record in row["generation_trace"]
        ]
        receipt["counts"].update(
            proxy_requests_completed=len(rows),
            generation_attempts=len(generation_records),
            extraction_attempts=transport["/v1/c2kv/extract"],
            detector_counts=dict(Counter(
                row["memory_runtime"]["exact_recovery"]["status"] for row in rows)),
            regenerations=sum(row["generation_attempts"] - 1 for row in rows),
            generation_usage_total=proxy._generation_summary(
                generation_records)["generation_usage_total"],
        )
        receipt.update(status="completed", current_cell=None)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["counts"]["transport_attempts_by_path"] = dict(transport)
        save(receipt_path, receipt)
        print(json.dumps({
            "status": receipt.get("status"), "profile": receipt.get("profile"),
            "counts": receipt.get("counts"),
        }))


if __name__ == "__main__":
    main()
