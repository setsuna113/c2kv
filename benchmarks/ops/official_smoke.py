"""Run one bounded official case per adapter against an isolated SGLang server.

These are plumbing gates, not estimates of benchmark quality. Each adapter
still uses its official scorer. Raw results, proxy logs and commands persist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

from validate_npu import stop_owned_group

RUNNER = Path(__file__).resolve().parents[1] / "run.py"
ENV_NAMES = {"bfcl": "bench", "tau2": "bench312", "toolsandbox": "benchts"}
CASES = {
    "bfcl": ["--categories", "multi_turn_base", "--run-ids", "multi_turn_base_1"],
    "tau2": ["--task-set", "airline", "--max-tasks", "1",
             "--tau2-num-trials", "1", "--tau2-max-steps", "12", "--tau2-timeout", "300"],
    "toolsandbox": ["--ts-scenarios",
                    "send_message_with_contact_content_cellular_off_multiple_user_turn"],
}


def run_case(args, name):
    python = args.env_root / ENV_NAMES[name] / "bin" / "python"
    cell = args.out / name
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(python), str(RUNNER), "--benchmark", name, "--arm", args.arm,
               "--upstream", args.upstream, "--user-upstream", args.upstream,
               "--model", args.model, "--proxy-port", str(port), "--backend", "sglang",
               "--num-workers", "1", "--out", str(cell),
               "--run-name", f"c2kv_integration_{name}_{uuid.uuid4().hex[:12]}",
               *CASES[name]]
    env = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
               TS_PARALLEL="1", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               OPENAI_API_KEY="EMPTY", OPENAI_API_KEY_USER="EMPTY")
    result = {"benchmark": name, "arm": args.arm, "command": command, "passed": False}
    (args.out / f"{name}.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (args.out / f"{name}.log").open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            result["returncode"] = proc.wait(timeout=args.timeout)
            if result["returncode"] != 0:
                result["error"] = "official adapter failed; inspect its log"
            else:
                summaries = list(args.out.glob(f"{name}*/summary_{args.arm}.json"))
                if len(summaries) != 1:
                    raise RuntimeError(f"expected one summary, found {len(summaries)}")
                summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                result["summary"] = str(summaries[0])
                result["n"] = (summary.get("n_scored") if name == "bfcl"
                               else summary.get("n"))
                requests = summary.get("request_log_summary") or {}
                result["n_ok_requests"] = requests.get("n_ok", 0)
                if (result["n"] != 1 or result["n_ok_requests"] < 1
                        or requests.get("n_error", 0)):
                    raise RuntimeError("one scored case with live proxy requests is required")
                if name == "bfcl":
                    score = (Path(summary["bfcl_project_root"]) / "score" /
                             f"c2kv-{args.arm.replace('_', '-')}" / "multi_turn" /
                             "BFCL_v4_multi_turn_base_score.json")
                    with score.open(encoding="utf-8") as handle:
                        official_score = json.loads(handle.readline())
                    if official_score.get("total_count") != 1:
                        raise RuntimeError("official BFCL scorer did not score exactly one case")
                    result["official_score"] = str(score)
                result["passed"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            stop_owned_group(proc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--arm", default="c2kv")
    parser.add_argument("--benchmarks", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--env-root", type=Path, default=Path("/home/liuyancheng/envs"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this gate uses the Linux benchmark environments")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for name in args.benchmarks:
        result = run_case(args, name)
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    raise SystemExit(0 if all(item["passed"] for item in results) else 1)


if __name__ == "__main__":
    main()
