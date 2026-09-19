"""Run one budget-adapted paper text arm on a dedicated NPU budget server.

The shared paper checkout supplies the algorithm, proxy, and BFCL adapter. This
launcher does not own the server process. The shared proxy flushes the engine
cache between BFCL episodes, so the upstream must be exclusively allocated to
this cell for its full duration.
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


def validate_paper_root(root: Path) -> Path:
    root = root.resolve()
    required = (
        "benchmarks/run.py",
        "benchmarks/arms.py",
        "benchmarks/proxy.py",
        "benchmarks/hiagent_budget.py",
        "benchmarks/paper/budget_server.py",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"--paper-root lacks shared budget sources: {', '.join(missing)}")
    return root


def arm_name(history_budget_tokens: int) -> str:
    if (isinstance(history_budget_tokens, bool) or
            not isinstance(history_budget_tokens, int) or
            history_budget_tokens < 1):
        raise ValueError("--history-budget-tokens must be a positive integer")
    return f"hiagent_full_b{history_budget_tokens}"


def command(args: argparse.Namespace) -> list[str]:
    arm = arm_name(args.history_budget_tokens)
    return [
        args.python, "-m", "benchmarks.run",
        "--benchmark", "bfcl", "--categories", BFCL_CATEGORIES[args.benchmark],
        "--arm", arm, "--upstream", args.upstream.rstrip("/"),
        "--proxy-port", str(args.proxy_port), "--out", str(args.out.resolve()),
        "--exact-out", "--num-workers", "1", "--backend", "sglang",
        "--model", args.model, "--checkpoint", str(args.checkpoint.resolve()),
        "--run-name", f"{args.benchmark}__{arm}",
        "--telemetry-log", str(args.out.resolve() / "proxy_telemetry.jsonl"),
        "--capability-features", "hiagent_trajectory_retrieval_v1",
        *(["--checkpoint-profile", str(args.checkpoint_profile.resolve())]
          if args.checkpoint_profile else []),
        *(["--run-ids", args.run_ids] if args.run_ids else []),
    ]


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
    parser.add_argument("--benchmark", choices=tuple(BFCL_CATEGORIES), required=True)
    parser.add_argument("--history-budget-tokens", type=int, required=True)
    parser.add_argument("--upstream", required=True,
                        help="exclusively allocated NPU budget_server base URL, without /v1")
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-profile", type=Path)
    parser.add_argument("--bfcl-dir", type=Path, required=True)
    parser.add_argument("--run-ids", help="comma-separated official BFCL case IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the client command without contacting the engine")
    args = parser.parse_args(argv)
    if args.history_budget_tokens < 1:
        parser.error("--history-budget-tokens must be positive")
    root = validate_paper_root(args.paper_root)
    argv = command(args)
    if args.dry_run:
        print(json.dumps({"cell_id": f"{args.benchmark}__{arm_name(args.history_budget_tokens)}",
                          "command": argv, "live_budget_preflight": "skipped (dry-run)"}, indent=2))
        return
    validate_live_budget_server(args.upstream, args.model)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["BENCH_BFCL_DIR"] = str(args.bfcl_dir.resolve())
    subprocess.run(argv, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
