"""NPU reference-history cell using the shared paper ON/OFF path.

The engine is already owned by the caller. This launcher invokes the paper
benchmark/proxy code directly; it contains no H2O, SnapKV or T0 algorithm copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener


CELL_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
BENCHMARKS = {
    "bfcl_base": ("bfcl", ("--categories", "multi_turn_base")),
    "bfcl_long_context": ("bfcl", ("--categories", "multi_turn_long_context")),
    "acebench_agent": ("acebench", ("--acebench-category", "agent")),
    "appworld": ("acon_appworld", ()),
    "toolsandbox": ("toolsandbox", ()),
}


def validate_paper_root(root: Path) -> Path:
    root = root.resolve()
    required = ("benchmarks/run.py", "benchmarks/proxy.py",
                "benchmarks/toolmemory.py", "benchmarks/arms.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("--paper-root lacks shared tool-memory sources: " + ", ".join(missing))
    return root


def cell_id(benchmark: str, arm: str, target_tokens: int,
            tool_memory: str) -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported official benchmark: {benchmark!r}")
    for name, value in (("benchmark", benchmark), ("arm", arm)):
        if not CELL_PART.fullmatch(value):
            raise ValueError(f"Invalid {name}: {value!r}")
    if not arm.startswith("gen_") or not arm.endswith(("_k0", "_k2", "_b0", "_b2")):
        raise ValueError("NPU reference-history cells require an absolute-budget gen_* arm")
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("--history-target-tokens must be the parent cell's positive K/B budget")
    if tool_memory not in {"", "none"} and not re.fullmatch(
            r"t0:r[0-9]+(?::hybrid[0-9]+)?", tool_memory):
        raise ValueError("This NPU launcher supports T0 tool memory only")
    label = ("raw" if tool_memory in {"", "none"}
             else "tools-" + tool_memory.replace(":", "_"))
    return f"{benchmark}__{arm}__history{target_tokens}__{label}"


def command(args: argparse.Namespace, benchmark_args=()) -> list[str]:
    identity = cell_id(args.benchmark, args.arm, args.history_target_tokens,
                       args.tool_memory)
    benchmark, fixed_args = BENCHMARKS[args.benchmark]
    output = args.out.resolve() / identity
    result = [
        args.python, "-m", "benchmarks.run",
        "--benchmark", benchmark, "--arm", args.arm,
        "--upstream", args.upstream.rstrip("/"),
        "--proxy-port", str(args.proxy_port),
        "--out", str(output), "--exact-out",
        "--num-workers", "1", "--backend", "sglang",
        "--model", args.model,
        "--checkpoint", str(args.checkpoint.resolve()),
        "--history-kv-target-tokens", str(args.history_target_tokens),
        "--shared-engine",
        "--run-name", identity,
        "--telemetry-log", str(output / "proxy_telemetry.jsonl"),
    ]
    if args.tool_memory not in {"", "none"}:
        if args.tool_checkpoint is None:
            raise ValueError("T0 tool memory requires --tool-checkpoint")
        result.extend(["--tool-memory", args.tool_memory,
                       "--tool-checkpoint", str(args.tool_checkpoint.resolve())])
    elif args.tool_checkpoint is not None or args.tool_budget_tokens is not None:
        raise ValueError("Tool checkpoint/budget requires T0 tool memory")
    if args.tool_budget_tokens is not None:
        if args.tool_budget_tokens <= 0:
            raise ValueError("--tool-budget-tokens must be positive")
        result.extend(["--tool-budget-tokens", str(args.tool_budget_tokens)])
    if args.user_upstream:
        result.extend(["--user-upstream", args.user_upstream.rstrip("/")])
    if args.bench_python:
        result.extend(["--bench-python", args.bench_python])
    result.extend(fixed_args)
    result.extend(benchmark_args)
    return result


def validate_live_engine(upstream: str, checkpoint: Path,
                         tool_checkpoint: Path | None = None) -> dict:
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(upstream.rstrip("/") + "/model_info", method="GET"),
                     timeout=15) as response:
        info = json.load(response)
    native = info.get("c2kv_native_packed") if isinstance(info, dict) else None
    binding = native.get("model_binding") if isinstance(native, dict) else None
    tool = native.get("tool_gist") if isinstance(native, dict) else None
    expected_hash = (hashlib.sha256((tool_checkpoint / "config.json").read_bytes()).hexdigest()
                     if tool_checkpoint is not None else None)
    problems = []
    if not isinstance(native, dict) or native.get("enabled") is not True:
        problems.append("native packed generation is unavailable")
    if not isinstance(binding, dict) or Path(binding.get("model_path", "")).resolve() != checkpoint.resolve():
        problems.append("history checkpoint binding differs")
    if tool_checkpoint is not None:
        if (not isinstance(tool, dict) or tool.get("enabled") is not True
                or tool.get("extract_projection_set") != "tool"):
            problems.append("T0 tool projection is unavailable")
        elif (Path(tool.get("source", "")).resolve() != tool_checkpoint.resolve()
              or tool.get("config_sha256") != expected_hash):
            problems.append("T0 tool checkpoint binding differs")
    if problems:
        raise RuntimeError("NPU engine readiness failed: " + "; ".join(problems))
    return {"history_checkpoint": str(checkpoint.resolve()),
            **({"tool_checkpoint": str(tool_checkpoint.resolve()),
                "tool_config_sha256": expected_hash}
               if tool_checkpoint is not None else {})}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--history-target-tokens", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--user-upstream", default="")
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tool-memory", default="none")
    parser.add_argument("--tool-checkpoint", type=Path)
    parser.add_argument("--tool-budget-tokens", type=int)
    parser.add_argument("--bench-python", default="")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    try:
        root = validate_paper_root(args.paper_root)
        cmd = command(args, forwarded)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"cell_id": cell_id(args.benchmark, args.arm,
                                               args.history_target_tokens, args.tool_memory),
                          "command": cmd, "engine_preflight": "skipped (dry-run)"}, indent=2))
        return
    validate_live_engine(args.upstream, args.checkpoint,
                         args.tool_checkpoint if args.tool_memory not in {"", "none"} else None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
