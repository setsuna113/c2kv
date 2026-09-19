"""Official AppWorld worker for controller-driven cells (acon_appworld).

Mirrors the event_native_bfcl worker contract: validate the frozen server
manifest, then drive exactly ONE task through the official ACON harness
(run_all.py + appworld evaluate) against the controller's OpenAI endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

RUNTIME_ROOT = Path(__file__).resolve().parents[2]   # controller_runtime/
SUPPORTED_SESSION_CACHE_POLICIES = {"external-sglang-content-addressed-chunks-v1"}


def _is_greedy_sampling(sampling):
    if not isinstance(sampling, dict) or set(sampling) - {
            "mode", "temperature", "seed", "top_p", "presence_penalty"}:
        return False
    if sampling.get("mode", "greedy") != "greedy":
        return False
    temperature = sampling.get("temperature")
    top_p = sampling.get("top_p")
    presence_penalty = sampling.get("presence_penalty")
    seed = sampling.get("seed")
    if (type(temperature) not in (int, float)
            or type(top_p) not in (int, float)
            or type(presence_penalty) not in (int, float)
            or type(seed) is not int):
        return False
    return (temperature == 0.0 and top_p == 1.0
            and presence_penalty == 0.5 and seed == 42)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_health(base_url):
    url = urlsplit(base_url)
    if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("base URL must be loopback")
    health_url = urlunsplit((url.scheme, url.netloc, "/health", "", ""))
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(health_url, method="GET"), timeout=5) as response:
        return json.load(response)


def validate_server_identity(ready, health):
    if ready.get("schema") != "a-event-native-server-v1" or ready.get("status") != "ready":
        raise ValueError("a ready event-native server manifest is required")
    if health.get("schema") != "a-event-native-api-health-v1":
        raise ValueError("endpoint health schema is not event-native")
    if ready.get("benchmark") != "acon_appworld":
        raise ValueError("the AppWorld worker requires the acon_appworld namespace")
    if ready.get("decode_strategy") not in {"incremental", "full_recompute"}:
        raise ValueError("server manifest lacks a supported decode strategy")
    if ready.get("session_cache_policy") not in SUPPORTED_SESSION_CACHE_POLICIES:
        raise ValueError("server manifest lacks the supported session cache policy")
    route = ready.get("route_contract")
    if not isinstance(route, dict) or route.get("legacy_1088_equivalent") is not False:
        raise ValueError("server manifest lacks an explicit event-native route identity")
    for key in ("run_id", "model_name", "view_mode", "decode_strategy",
                "session_cache_policy", "max_new_tokens", "max_decisions",
                "max_generation_calls", "route_contract", "runtime_policy_contract"):
        if health.get(key) != ready.get(key):
            raise ValueError(f"endpoint identity differs for {key}")
    if sorted(health.get("allowed_task_ids", [])) != sorted(ready.get("allowed_task_ids", [])):
        raise ValueError("endpoint frozen task IDs differ")
    if health.get("terminal") is not False:
        raise ValueError("endpoint is not available for a new run")
    if not _is_greedy_sampling(ready.get("sampling")):
        raise ValueError("server sampling contract is unsupported")


def run_appworld_task(base_url, acon_dir, appworld_root, bench_python, task_id,
                      out_dir, model, max_iter):
    from adapters import acon_adapter as adapter
    harness = out_dir / "harness"
    if harness.exists():
        raise ValueError(f"refusing existing harness output {harness}")
    old_root = os.environ.get("APPWORLD_ROOT")
    os.environ["APPWORLD_ROOT"] = str(Path(appworld_root).resolve())
    try:
        acon_src = Path(acon_dir) / "src"
        sys.path.insert(0, str(acon_src))
        summary = adapter.run_appworld(
            base_url, harness, acon_dir=Path(acon_dir), model=model,
            tag=f"generality_{task_id}", split="test_normal", max_iter=max_iter,
            task_ids=[task_id], python=bench_python)
    finally:
        if acon_src:
            sys.path.remove(str(acon_src))
        if old_root is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = old_root
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--acon-dir", type=Path, required=True)
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args(argv)

    ready = _read_json(args.server_manifest)
    started = time.monotonic()
    health = read_health(args.base_url)
    validate_server_identity(ready, health)
    args.out.mkdir(parents=True, exist_ok=False)
    score = None
    status = "failed"
    summary = None
    try:
        summary = run_appworld_task(
            args.base_url.rstrip("/").removesuffix("/v1") + "/v1"
            if not args.base_url.rstrip("/").endswith("/v1") else args.base_url.rstrip("/"),
            args.acon_dir, args.appworld_root, args.bench_python,
            args.task_id, args.out, ready["model_name"], args.max_iter)
        value = summary.get("semantic_score")
        if summary.get("n") == 1 and isinstance(value, (int, float)) and math.isfinite(value):
            score = value
            status = "completed"
        else:
            status = "score_invalid"
    except subprocess.TimeoutExpired:
        status = "wall_cap_reached"
    except Exception as error:
        status = f"failed:{type(error).__name__}"
        _save(args.out / "worker_error.json", {"error": f"{type(error).__name__}: {error}"})
    _save(args.out / "official_summary.json", {
        "schema": "a-event-native-appworld-run-v1", "status": status,
        "task_id": args.task_id, "n": 1 if score is not None else 0,
        "semantic_score": score,
        "summary": summary, "wall_s": time.monotonic() - started,
    })
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
