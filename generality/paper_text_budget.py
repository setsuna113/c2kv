"""Run one budget-adapted paper text arm on a dedicated NPU budget server.

The shared paper checkout supplies the algorithm, proxy, and official adapter.
This launcher does not own the server process. The shared proxy flushes the
engine cache between episodes, so the upstream must be exclusively allocated
to this cell for its full duration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request


BFCL_CATEGORIES = {
    "bfcl_base": "multi_turn_base",
    "bfcl_long_context": "multi_turn_long_context",
}
BENCHMARKS = (*BFCL_CATEGORIES, "acebench_agent")
METHOD_ARMS = {"hiagent": "hiagent_full", "acon": "acon_hist_ut_co"}


def validate_paper_root(root: Path, method: str = "hiagent",
                        benchmark: str | None = None) -> Path:
    root = root.resolve()
    required = [
        "benchmarks/run.py",
        "benchmarks/arms.py",
        "benchmarks/proxy.py",
        "benchmarks/hiagent_budget.py",
        "benchmarks/paper/budget_server.py",
    ]
    if method == "acon":
        required.append("benchmarks/acon_budget.py")
    if benchmark == "acebench_agent":
        required.extend(("benchmarks/adapters/acebench_adapter.py",
                         "benchmarks/acebench_cli.py"))
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"--paper-root lacks shared budget sources: {', '.join(missing)}")
    return root


def arm_name(history_budget_tokens: int, method: str = "hiagent") -> str:
    if (isinstance(history_budget_tokens, bool) or
            not isinstance(history_budget_tokens, int) or
            history_budget_tokens < 1):
        raise ValueError("--history-budget-tokens must be a positive integer")
    if method not in METHOD_ARMS:
        raise ValueError(f"unknown budget method: {method!r}")
    return f"{METHOD_ARMS[method]}_b{history_budget_tokens}"


def command(args: argparse.Namespace) -> list[str]:
    method = getattr(args, "method", "hiagent")
    arm = arm_name(args.history_budget_tokens, method)
    benchmark = args.benchmark
    features = (["hiagent_trajectory_retrieval_v1"] if method == "hiagent" else [])
    if benchmark == "acebench_agent":
        features.append("acebench_role_history_v1")
    cmd = [
        args.python, "-m", "benchmarks.run",
        "--benchmark", "acebench" if benchmark == "acebench_agent" else "bfcl",
        *(["--categories", BFCL_CATEGORIES[benchmark]]
          if benchmark in BFCL_CATEGORIES else []),
        "--arm", arm, "--upstream", args.upstream.rstrip("/"),
        "--proxy-port", str(args.proxy_port), "--out", str(args.out.resolve()),
        "--exact-out", "--num-workers", "1", "--backend", "sglang",
        "--model", args.model, "--checkpoint", str(args.checkpoint.resolve()),
        "--run-name", f"{benchmark}__{arm}",
        "--telemetry-log", str(args.out.resolve() / "proxy_telemetry.jsonl"),
    ]
    if features:
        cmd += ["--capability-features", ",".join(features)]
    if args.checkpoint_profile:
        cmd += ["--checkpoint-profile", str(args.checkpoint_profile.resolve())]
    if benchmark == "acebench_agent":
        cmd += ["--acebench-category", "agent",
                "--acebench-dir", str(args.acebench_dir.resolve()),
                "--acebench-language", getattr(args, "acebench_language", "en"),
                "--bench-python", getattr(args, "bench_python", None) or args.python,
                "--user-upstream", getattr(args, "user_upstream", None) or args.upstream.rstrip("/")]
        if getattr(args, "acebench_task_ids", None):
            cmd += ["--acebench-task-ids", args.acebench_task_ids]
        if getattr(args, "max_tasks", None) is not None:
            cmd += ["--max-tasks", str(args.max_tasks)]
    else:
        if args.run_ids:
            cmd += ["--run-ids", args.run_ids]
    return cmd


def validate_live_budget_server(upstream: str, model: str) -> dict:
    """Require an actual served-chat tokenization response, without generation."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "budget preflight history"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "budget preflight current turn"},
        ],
        "max_tokens": 1,
        "c2kv_kv_memory_hint": {"paper_measurement": {
            "history_start_message_count": 0,
            "history_message_count": 2,
        }},
    }
    request = urllib.request.Request(
        upstream.rstrip("/") + "/v1/c2kv/chat_budget",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            receipt = json.load(response)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "The live upstream does not provide the required "
            "/v1/c2kv/chat_budget served-chat renderer; launch the shared "
            "benchmarks.paper.budget_server on an owned NPU engine"
        ) from error
    if (not isinstance(receipt, dict) or
            receipt.get("success") is not True or
            receipt.get("server_tokenized") is not True or
            not isinstance(receipt.get("history_tokens"), int) or
            isinstance(receipt["history_tokens"], bool) or
            receipt["history_tokens"] <= 0 or
            not isinstance(receipt.get("prompt_tokens"), int) or
            receipt["prompt_tokens"] < receipt["history_tokens"] or
            not isinstance(receipt.get("history_start"), int) or
            not isinstance(receipt.get("history_end"), int) or
            receipt["history_end"] - receipt["history_start"] != receipt["history_tokens"]):
        raise RuntimeError("The live chat budget endpoint returned an invalid tokenization receipt")
    return receipt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_ARMS), default="hiagent")
    parser.add_argument("--history-budget-tokens", type=int, required=True)
    parser.add_argument("--upstream", required=True,
                        help="exclusively allocated NPU budget_server base URL, without /v1")
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-profile", type=Path)
    parser.add_argument("--bfcl-dir", type=Path)
    parser.add_argument("--run-ids", help="comma-separated official BFCL case IDs")
    parser.add_argument("--acebench-dir", type=Path)
    parser.add_argument("--acebench-task-ids", help="comma-separated official ACEBench agent task IDs")
    parser.add_argument("--acebench-language", choices=("en", "zh"), default="en")
    parser.add_argument("--bench-python", help="ACEBench harness Python; defaults to --python")
    parser.add_argument("--user-upstream", help="raw endpoint for the ACEBench user simulator")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the client command without contacting the engine")
    args = parser.parse_args(argv)
    if args.history_budget_tokens < 1:
        parser.error("--history-budget-tokens must be positive")
    if args.benchmark == "acebench_agent":
        if not args.acebench_dir:
            parser.error("--acebench-dir is required for acebench_agent")
        if args.run_ids:
            parser.error("--run-ids is BFCL-only; use --acebench-task-ids")
    elif not args.bfcl_dir:
        parser.error("--bfcl-dir is required for BFCL")
    root = validate_paper_root(args.paper_root, args.method, args.benchmark)
    argv = command(args)
    if args.dry_run:
        print(json.dumps({"cell_id": f"{args.benchmark}__{arm_name(args.history_budget_tokens, args.method)}",
                          "command": argv, "live_budget_preflight": "skipped (dry-run)"}, indent=2))
        return
    validate_live_budget_server(args.upstream, args.model)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    if args.bfcl_dir:
        env["BENCH_BFCL_DIR"] = str(args.bfcl_dir.resolve())
    subprocess.run(argv, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
