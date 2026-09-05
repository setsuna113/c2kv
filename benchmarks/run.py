"""Unified runner: one benchmark x one arm -> unified rows + summary.

Registry dispatch over the per-benchmark adapters (contract:
``adapters/base.py`` — NAME, add_arguments, run(ctx)); this module owns only
what is common to every benchmark: the proxy lifecycle (spawn
benchmarks/proxy.py in the requested arm, tear it down after), the git-sha
suffix on --run-name/--out, and the summary envelope.  The compression ratio
comes from the arm registry (arms.py), NOT from a --ratio flag.

Adding a benchmark = one adapter module + one ADAPTERS entry.  Server
scripts drive this file BY CLI ONLY: every flag below is the interface, so
a flag never changes name, default or meaning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapters import (  # noqa: E402
    acebench_adapter, acon_adapter, bfcl_adapter, tau2_adapter,
    toolsandbox_adapter,
)
from adapters.base import RunContext  # noqa: E402

# --benchmark value -> adapter module.  Two names share acon_adapter (the
# module dispatches on ctx.options["benchmark"]); add_arguments is called
# once per MODULE, so a shared flag would collide in argparse and must stay
# in the core block of main() instead.
ADAPTERS = {
    "tau2": tau2_adapter,
    "bfcl": bfcl_adapter,
    "toolsandbox": toolsandbox_adapter,
    "acon_appworld": acon_adapter,
    "acon_qa": acon_adapter,
    "acebench": acebench_adapter,
}
assert set(acon_adapter.NAMES) <= set(ADAPTERS)
assert all(module.NAME in ADAPTERS or name in getattr(module, "NAMES", ())
           for name, module in ADAPTERS.items())


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


def add_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags run.py itself owns: the proxy/arm/serving knobs, plus the few
    flags MORE THAN ONE adapter reads (argparse refuses a duplicate option
    string, so those cannot live in an adapter's add_arguments)."""
    parser.add_argument("--benchmark", required=True, choices=list(ADAPTERS))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--upstream", required=True,
                        help="backend base URL (REQUIRED: a default here once "
                             "silently aimed sglang runs at the hf_server port)")
    parser.add_argument("--user-upstream", default="",
                        help="base URL for user-simulator/judge traffic (defaults to --upstream; only the agent arm proxy compresses)")
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument("--out", type=Path, required=True)
    # shared by tau2 (--max-concurrency) and acebench (--num-threads)
    parser.add_argument("--num-workers", type=int, default=4)
    # shared by tau2 (--num-tasks) and acon_qa (--limit)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
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
    parser.add_argument("--doc-packing", default="turn", choices=["turn", "message"],
                        help="proxy doc packing: 'turn' = training format "
                             "(default), 'message' = pre-2026-09 per-message")
    parser.add_argument("--max-doc-length", type=int, default=512,
                        help="training regime = 512 (ckpt-1088); 768 was the "
                             "old D-harness caliber")
    parser.add_argument("--max-doc-num", type=int, default=12,
                        help="training regime = 12 (ckpt-1088)")
    # shared by acon_* (agent step cap) and acebench (--max-dialog-turns)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="acon_qa/acon_appworld: agent step cap (runner defaults "
                             "30/50); acebench: --max-dialog-turns (default 40)")
    # shared by acon_* and acebench
    parser.add_argument("--bench-python", default="",
                        help="python of the harness venv for acon_*/acebench "
                             "(default: this interpreter)")


def build_parser() -> argparse.ArgumentParser:
    """Core flags + every adapter's own flags (each module once)."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_core_arguments(parser)
    for module in dict.fromkeys(ADAPTERS.values()):
        module.add_arguments(parser)
    return parser


def build_context(args: argparse.Namespace, request_log: Path) -> RunContext:
    """The one object an adapter receives.  ``options`` is the whole parsed
    namespace, so an adapter reads exactly the flags it registered (plus the
    core ones) and run.py needs no per-benchmark knowledge."""
    options = dict(vars(args))
    options["benchmark"] = args.benchmark
    return RunContext(
        base_url=f"http://127.0.0.1:{args.proxy_port}",
        user_base_url=args.user_upstream or args.upstream,
        out_dir=args.out,
        model=args.model,
        arm=args.arm,
        run_name=args.run_name,
        request_log=Path(request_log) if request_log else None,
        options=options,
    )


def main(argv=None):
    parser = build_parser()
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
        # every adapter owns its own "/v1" (adapters/base.py:v1) and its own
        # cwd; run.py hands over the bare proxy URL and nothing else
        ctx = build_context(args, request_log)
        summary = ADAPTERS[args.benchmark].run(ctx)
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
        summary["textarm_summary"] = {
            "textarm_requests": len(ta_rows),
            "degenerate_requests": degenerate_requests,
            "degenerate_arm": bool(
                ta_rows and degenerate_requests == len(ta_rows)),
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
        if any(t.get("degenerate") for t in ta_rows):
            print(f"WARNING: arm {args.arm!r} ran DEGENERATE (no Subgoal "
                  f"segments) on {sum(1 for t in ta_rows if t.get('degenerate'))}"
                  f"/{len(ta_rows)} requests — effectively a full arm")
    # every arm: regime facts from the request log (doc drops, projection
    # mode mix, outcome mix, wall percentiles) — a compressed number whose
    # run dropped docs or mixed query_proj modes is not one regime
    import reqlog  # noqa: E402  (sibling module)

    summary["request_log_summary"] = reqlog.summarize_file(request_log)
    rl = summary["request_log_summary"]
    if rl.get("dropped_requests"):
        print(f"NOTE: {rl['dropped_requests']}/{rl['n_ok']} requests dropped history "
              f"docs (turn packing, max_doc_num={args.max_doc_num}); mean dropped "
              f"{rl['dropped_docs_mean']}")
    if rl.get("mixed_query_proj"):
        print(f"WARNING: request log mixes c2kv_query_proj modes {rl['c2kv_query_proj']} "
              "— not one serving regime")
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
