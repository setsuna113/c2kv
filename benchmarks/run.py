"""Unified runner: one benchmark x one arm -> unified summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms import get_arm  # noqa: E402


def start_proxy(
    upstream: str, arm: str, port: int, log_dir: Path,
    doc_packing: str = "turn", max_docs: int = 16, max_doc_length: int = 768,
):
    log_path = log_dir / f"proxy_{arm}_{port}.jsonl"
    proxy_out = log_dir / f"proxy_{arm}_{port}.out"
    with proxy_out.open("w", encoding="utf-8") as out_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "proxy.py"),
                "--upstream", upstream,
                "--arm", arm,
                "--port", str(port),
                "--request-log", str(log_path),
                "--doc-packing", doc_packing,
                "--max-docs", str(max_docs),
                "--max-doc-length", str(max_doc_length),
            ],
            stdout=out_handle,
            stderr=subprocess.STDOUT,
        )

    import urllib.request

    last_error: Optional[Exception] = None
    for _ in range(150):
        if proc.poll() is not None:
            raise SystemExit(
                f"FATAL: proxy exited before health check ({proc.returncode}); log={proxy_out}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).close()
            return proc, log_path
        except OSError as error:
            last_error = error
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"FATAL: proxy did not come up on port {port}: {last_error}; log={proxy_out}")


def run_benchmark(
    name: str,
    base_url: str,
    user_base_url: str,
    out_dir: Path,
    **kwargs,
) -> Dict[str, Any]:
    if name == "tau2":
        from adapters import tau2_adapter

        return tau2_adapter.run(
            kwargs.get("benchmark_dir"),
            base_url,
            user_base_url,
            out_dir,
            task_set=kwargs.get("task_set", "airline"),
            num_workers=kwargs.get("num_workers", 4),
            max_tasks=kwargs.get("max_tasks"),
            run_name=kwargs.get("run_name", "c2kv_run"),
        )
    if name == "bfcl":
        from adapters import bfcl_adapter

        return bfcl_adapter.run(
            base_url,
            categories=kwargs.get("categories", "multi_turn_base"),
            mode=kwargs.get("bfcl_mode", "both"),
            run_ids=kwargs.get("run_ids", ""),
            benchmark_dir=kwargs.get("benchmark_dir"),
            out_dir=out_dir,
            arm=kwargs.get("arm", "full"),
            served_model_name=kwargs.get("served_model_name"),
        )
    if name == "toolsandbox":
        from adapters import toolsandbox_adapter

        return toolsandbox_adapter.run(
            base_url,
            out_dir,
            test_mode=not kwargs.get("full", False),
            scenarios=kwargs.get("toolsandbox_scenarios"),
            num_workers=kwargs.get("num_workers", 4),
        )
    raise SystemExit(f"FATAL: unknown benchmark {name!r}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["tau2", "bfcl", "toolsandbox"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:34000")
    parser.add_argument(
        "--user-upstream",
        default="",
        help="defaults to --upstream; user-simulator traffic bypasses the arm proxy",
    )
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument(
        "--doc-packing", choices=["message", "turn"], default="turn",
        help=(
            "segment granularity for compressed history. 'turn' (default)"
            " matches the trainer's turn documents and is the only setting"
            " under which a doc_mode=history_only checkpoint is served in its"
            " own dialect."
        ),
    )
    parser.add_argument(
        "--max-docs", type=int, default=16,
        help=(
            "trainer max_doc_num tail cap on compressed docs, keeping doc 0"
            " plus the newest ones (0 = uncapped)"
        ),
    )
    parser.add_argument(
        "--max-doc-length", type=int, default=768,
        help=(
            "trainer max_doc_length: oversized turn documents are split on"
            " line boundaries before the --max-docs cap (0 = no split)"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="tau2 checkout or BFCL package root; adapters also honor env vars")
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--run-ids", default="", help="BFCL smoke subset, comma-separated")
    parser.add_argument("--bfcl-mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--toolsandbox-scenarios", nargs="*", default=None)
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
    args = parser.parse_args(argv)

    arm = get_arm(args.arm)
    if arm.constrain_tools:
        raise SystemExit(
            f"FATAL: arm {arm.name!r} requires hf_server constrain_tools and is disabled in this run"
        )
    if args.full and args.toolsandbox_scenarios:
        raise SystemExit("FATAL: --full cannot be combined with --toolsandbox-scenarios")

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    proxy_proc, request_log = start_proxy(
        args.upstream, args.arm, args.proxy_port, log_dir,
        doc_packing=args.doc_packing, max_docs=args.max_docs,
        max_doc_length=args.max_doc_length,
    )
    try:
        base_url = f"http://127.0.0.1:{args.proxy_port}"
        user_base_url = args.user_upstream or args.upstream
        summary = run_benchmark(
            args.benchmark,
            base_url,
            user_base_url,
            args.out,
            arm=args.arm,
            benchmark_dir=args.benchmark_dir,
            task_set=args.task_set,
            categories=args.categories,
            num_workers=args.num_workers,
            max_tasks=args.max_tasks,
            run_name=args.run_name,
            run_ids=args.run_ids,
            bfcl_mode=args.bfcl_mode,
            served_model_name=args.served_model_name,
            toolsandbox_scenarios=args.toolsandbox_scenarios,
            full=args.full,
        )
    finally:
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait()

    summary["arm"] = args.arm
    summary["benchmark"] = args.benchmark
    summary["request_log"] = str(request_log)
    summary["upstream"] = args.upstream
    summary_path = args.out / f"summary_{args.arm}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
