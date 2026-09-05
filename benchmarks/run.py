"""Unified runner: one benchmark x one arm -> unified rows + summary.

Thin dispatcher around the per-benchmark adapters; owns the proxy lifecycle
(spawn benchmarks/proxy.py in the requested arm, tear it down after).
The compression ratio comes from the arm registry (arms.py), NOT from a
--ratio flag.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def start_proxy(upstream: str, arm: str, port: int, log_dir: Path,
                record_reference: str = "", reference: str = "",
                backend: str = "sglang", doc_packing: str = "turn",
                max_doc_length: int = 512, max_doc_num: int = 12):
    log_path = log_dir / f"proxy_{arm}_{port}.jsonl"
    out_handle = open(log_dir / f"proxy_{arm}_{port}.out", "w")
    command = [
        sys.executable, str(HERE / "proxy.py"),
        "--upstream", upstream, "--arm", arm, "--backend", backend,
        "--port", str(port), "--request-log", str(log_path),
        "--doc-packing", doc_packing,
        "--max-doc-length", str(max_doc_length),
        "--max-doc-num", str(max_doc_num),
    ]
    if record_reference:
        command += ["--record-reference", record_reference]
    if reference:
        command += ["--reference", reference]
    proc = subprocess.Popen(
        command,
        stdout=out_handle,
        stderr=subprocess.STDOUT,
    )
    import urllib.request

    # never route the local health probe through an ambient http_proxy
    # (an inherited proxy env once made every run.py launch fail its own
    # gateway check)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    for _ in range(100):
        try:
            opener.open(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc, log_path
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"proxy did not come up on port {port}")


def run_benchmark(name: str, base_url: str, user_base_url: str, out_dir: Path,
                  model: str = "c2kv-agent", **kwargs) -> Dict[str, Any]:
    if name == "tau2":
        from adapters import tau2_adapter

        return tau2_adapter.run(
            kwargs.get("benchmark_dir"), base_url, user_base_url,
            out_dir, task_set=kwargs.get("task_set", "airline"),
            num_workers=kwargs.get("num_workers", 4),
            max_tasks=kwargs.get("max_tasks"),
            run_name=kwargs.get("run_name", "c2kv_run"),
            model=model,
        )
    if name == "bfcl":
        from adapters import bfcl_adapter

        # bfcl_eval resolves its data/result dirs from cwd; the adapter is
        # driven from the gorilla checkout ($BENCH_BFCL_DIR override).
        bfcl_dir = os.environ.get("BENCH_BFCL_DIR") or str(
            Path.home() / "benchmarks" / "gorilla"
            / "berkeley-function-call-leaderboard")
        prev_cwd = os.getcwd()
        os.chdir(bfcl_dir)
        try:
            # eval_runner.py:782 un-escapes the result dir with
            # replace("_", "/") — underscores in arm names would corrupt
            # the path, so the handler key uses dashes
            return bfcl_adapter.run(
                base_url,
                categories=kwargs.get("categories", "multi_turn_base"),
                model=model,
                handler_name=f"c2kv-{(kwargs.get('arm') or 'full').replace('_', '-')}",
            )
        finally:
            os.chdir(prev_cwd)
    if name == "toolsandbox":
        from adapters import toolsandbox_adapter

        return toolsandbox_adapter.run(
            base_url, out_dir, test_mode=not kwargs.get("full", False),
            agent=kwargs.get("ts_agent") or toolsandbox_adapter.AGENT,
            user=kwargs.get("ts_user") or toolsandbox_adapter.AGENT,
            # the user simulator must NOT ride the arm proxy: route it to
            # the raw upstream endpoint (tau2 already does the same split)
            user_base_url=user_base_url,
            scenarios=kwargs.get("ts_scenarios") or None,
        )
    raise SystemExit(f"unknown benchmark {name!r}")


def _git_short_sha() -> str:
    """Short commit of the running tree: run_name/out-dir suffix so a code
    change can never silently mix old and new trajectories via tau2's
    --auto-resume (which keeps every normally-terminated simulation)."""
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=10).stdout.strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["tau2", "bfcl", "toolsandbox"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--upstream", required=True,
                        help="backend base URL (REQUIRED: a default here once "
                             "silently aimed sglang runs at the hf_server port)")
    parser.add_argument("--user-upstream", default="",
                        help="base URL for user-simulator/judge traffic (defaults to --upstream; only the agent arm proxy compresses)")
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
    parser.add_argument("--record-reference", default="",
                        help="full-arm run: write the reference trajectory jsonl "
                             "the recover arms diff against")
    parser.add_argument("--reference", default="",
                        help="recover arm: reference trajectory jsonl (from a "
                             "--record-reference full-arm run)")
    parser.add_argument("--backend", default="sglang",
                        choices=["hfserver", "sglang"],
                        help="serving stack behind the proxy (sglang is the "
                             "eval path; hfserver survives as contrast only)")
    parser.add_argument("--model", default="c2kv-agent",
                        help="served model name at the endpoint (any "
                             "OpenAI-compatible endpoint serves any name; "
                             "tau2 agent/user LLMs and the BFCL handler "
                             "both use it; toolsandbox role keys are "
                             "separate, see --ts-agent)")
    parser.add_argument("--ts-scenarios", default="",
                        help="toolsandbox: comma-separated scenario names "
                             "for subset runs (-s); overrides --full")
    parser.add_argument("--ts-agent", default="",
                        help="toolsandbox: agent role key (default "
                             "GPT_4_o_2024_05_13 -> openai_api_agent)")
    parser.add_argument("--ts-user", default="",
                        help="toolsandbox: user-simulator role key (same default)")
    parser.add_argument("--doc-packing", default="turn", choices=["turn", "message"],
                        help="proxy doc packing: 'turn' = training format "
                             "(default), 'message' = pre-2026-09 per-message")
    parser.add_argument("--max-doc-length", type=int, default=512,
                        help="training regime = 512 (ckpt-1088); 768 was the "
                             "old D-harness caliber")
    parser.add_argument("--max-doc-num", type=int, default=12,
                        help="training regime = 12 (ckpt-1088)")
    args = parser.parse_args(argv)

    sha = _git_short_sha()
    if sha not in (args.run_name or ""):
        args.run_name = f"{args.run_name}_{sha}"
    if sha not in str(args.out):
        args.out = args.out.with_name(f"{args.out.name}_{sha}")

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    proxy_proc, request_log = start_proxy(
        args.upstream, args.arm, args.proxy_port, log_dir,
        record_reference=args.record_reference, reference=args.reference,
        backend=args.backend, doc_packing=args.doc_packing,
        max_doc_length=args.max_doc_length, max_doc_num=args.max_doc_num)
    try:
        # the BFCL handler expects an OpenAI base_url WITH /v1 (tau2 and
        # toolsandbox build their paths themselves)
        base_url = f"http://127.0.0.1:{args.proxy_port}" + (
            "/v1" if args.benchmark == "bfcl" else ""
        )
        user_base_url = args.user_upstream or args.upstream
        summary = run_benchmark(
            args.benchmark, base_url, user_base_url, args.out,
            model=args.model, arm=args.arm,
            task_set=args.task_set, categories=args.categories,
            num_workers=args.num_workers, max_tasks=args.max_tasks,
            run_name=args.run_name, full=args.full,
            ts_agent=args.ts_agent, ts_user=args.ts_user,
            ts_scenarios=[s.strip() for s in args.ts_scenarios.split(",")
                          if s.strip()],
        )
    finally:
        proxy_proc.terminate()
    summary["arm"] = args.arm
    summary["benchmark"] = args.benchmark
    summary["backend"] = args.backend
    summary["model"] = args.model
    if args.arm in ("hiagent", "acon_hist", "acon_obs"):
        # text-arm consumers: degeneration and compressor cost surfaced at
        # the RUN level (the per-request stats live in the request log)
        ta_rows = []
        try:
            with open(request_log, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line).get("textarm")
                        if isinstance(row, dict):
                            ta_rows.append(row)
        except OSError:
            pass
        degenerate_requests = sum(1 for t in ta_rows if t.get("degenerate"))
        compressed_requests = sum(
            1 for t in ta_rows if t.get("history_compressed"))
        summary["textarm_summary"] = {
            "textarm_requests": len(ta_rows),
            "degenerate_requests": degenerate_requests,
            "degenerate_arm": bool(
                ta_rows and degenerate_requests == len(ta_rows)),
            "history_compressed_requests": compressed_requests,
            "compressor_calls": sum(int(t.get("n_compressor_calls") or 0)
                                    for t in ta_rows),
            "compressor_prompt_tokens": sum(
                int((t.get("compressor_usage") or {}).get("prompt_tokens") or 0)
                for t in ta_rows),
            "compressor_completion_tokens": sum(
                int((t.get("compressor_usage") or {}).get("completion_tokens") or 0)
                for t in ta_rows),
            "compressor_wall_sec": round(sum(
                float((t.get("compressor_usage") or {}).get("wall_sec") or 0)
                for t in ta_rows), 1),
        }
        if (args.arm == "acon_hist" and ta_rows
                and compressed_requests == 0):
            print(f"WARNING: arm {args.arm!r} NEVER compressed history in "
                  f"{len(ta_rows)} requests — the trigger never fired; the "
                  f"row is effectively a full arm under the acon label")
        if any(t.get("degenerate") for t in ta_rows):
            print(f"WARNING: arm {args.arm!r} ran DEGENERATE (no Subgoal "
                  f"segments) on {sum(1 for t in ta_rows if t.get('degenerate'))}"
                  f"/{len(ta_rows)} requests — effectively a full arm")
    summary["doc_packing"] = args.doc_packing
    summary["max_doc_length"] = args.max_doc_length
    summary["max_doc_num"] = args.max_doc_num
    summary["request_log"] = str(request_log)
    if args.reference:
        summary["reference"] = args.reference
    if args.record_reference:
        summary["record_reference"] = args.record_reference
    (args.out / f"summary_{args.arm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
