"""ToolSandbox adapter (official-CLI driver).

The ToolSandbox Python API (Scenario/play/play_and_evaluate) does not expose
a Scenario.discover(); the supported entrypoint is the ``tool_sandbox`` CLI,
which runs scenarios against an OpenAI-compatible endpoint configured via
``OPENAI_BASE_URL`` and writes ``result_summary.json`` per run.  This
adapter drives that CLI and parses the summaries.

Verified metrics fields (per scenario): similarity / milestone_similarity /
minefield_similarity / turn_count.  There is no "main_acc".

Usage (benchts venv on the server):
    python benchmarks/adapters/toolsandbox_adapter.py \
        --base-url http://127.0.0.1:34002/v1 --out results/bench/ts_full
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.process_lifecycle import run_owned  # noqa: E402
from toolsandbox_cli import AGENT_TIMEOUT_ENV  # noqa: E402
from toolsandbox_suite import (  # noqa: E402
    THREE_DISTRACTION_TOOLS_129, load_named_suite, selected_scenarios,
)

from adapters.base import RunContext, v1  # noqa: E402
from adapters.text_budget_failures import (  # noqa: E402
    proxy_text_budget_failure_code, typed_text_budget_failure_code,
)

NAME = "toolsandbox"
TS_DIR = Path(os.environ.get("TS_DIR") or Path.home() / "benchmarks" / "ToolSandbox")
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys
_SERVER_INFO_OPENER = build_opener(ProxyHandler({}))
DEFAULT_GENERATION_TIMEOUT = 600.0  # run.py and proxy.py default deadline
CLEANUP_HEADROOM = 90.0  # as adapters.bfcl_adapter.client_kwargs


def agent_client_timeout(generation_timeout: float) -> "float | None":
    """Agent SDK timeout for a run whose proxy deadline is ``generation_timeout``.

    At the default deadline this is ``None``: toolsandbox_cli keeps its
    historical 600 s client timeout. A configured deadline (the paper runner
    passes one only to persistent history-KV cells) gets the BFCL client's
    headroom, so the proxy can return the engine's cleanup acknowledgement
    before the SDK gives up on the request.
    """
    timeout = float(generation_timeout)
    if not 0 < timeout < float("inf"):
        raise ValueError("generation_timeout must be finite and positive")
    if timeout == DEFAULT_GENERATION_TIMEOUT:
        return None
    return timeout + CLEANUP_HEADROOM


def _server_info(base_url: str) -> dict[str, Any] | None:
    """Read optional SGLang launch settings without going through host proxies."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        request = Request(root + "/server_info", headers={"Accept": "application/json"})
        with _SERVER_INFO_OPENER.open(request, timeout=3) as response:
            info = json.load(response)
    except (OSError, ValueError, UnicodeError):
        return None
    return info if isinstance(info, dict) else None


def require_sglang_tool_parser(base_url: str, user_base_url: str = "") -> None:
    """Reject an identifiable SGLang endpoint that cannot emit tool calls."""
    checked: set[str] = set()
    for role, endpoint in (("agent", base_url), ("user simulator", user_base_url)):
        if not endpoint:
            continue
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        if endpoint in checked:
            continue
        checked.add(endpoint)
        info = _server_info(endpoint)
        if not isinstance(info, dict) or not {
            "model_path", "tp_size", "tool_call_parser"
        }.issubset(info):
            continue
        if not info["tool_call_parser"]:
            raise RuntimeError(
                f"ToolSandbox preflight: SGLang {role} endpoint has tool_call_parser "
                "disabled; restart it with --tool-call-parser qwen25")


def add_arguments(parser) -> None:
    """ToolSandbox-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--toolsandbox-dir", type=Path, default=None,
                        help="ToolSandbox checkout (default $TS_DIR)")
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
    parser.add_argument("--ts-scenarios", default="",
                        help="toolsandbox: comma-separated scenario names "
                             "for subset runs (-s); overrides --full")
    parser.add_argument("--ts-suite", default="",
                        help=f"toolsandbox: frozen named suite ({THREE_DISTRACTION_TOOLS_129})")
    parser.add_argument("--ts-agent", default="",
                        help="toolsandbox: agent role key (default "
                             "GPT_4_o_2024_05_13 -> openai_api_agent)")
    parser.add_argument("--ts-user", default="",
                        help="toolsandbox: user-simulator role key (same default)")
    parser.add_argument("--ts-parallel", type=int, default=1, choices=[1],
                        help="toolsandbox: fixed to one process for attributable telemetry")


def cli_command(out_dir: Path, agent: str = AGENT, user: str = AGENT,
                test_mode: bool = True,
                scenarios: "list[str] | None" = None,
                parallel: "str | int" = 1) -> List[str]:
    """``tool_sandbox`` argv (PINNED).  ``-s names...`` is the subset form
    (the CLI also takes ``-p`` for parallelism, from $TS_PARALLEL); ``-t``
    is test mode, and a subset run overrides it."""
    if int(parallel) <= 0:
        raise ValueError("ToolSandbox parallel must be positive")
    cmd = ["tool_sandbox", "--user", user, "--agent", agent, "-o", str(out_dir)]
    if scenarios:
        cmd += ["-s"] + list(scenarios)
    elif test_mode:
        cmd.append("-t")
    cmd += ["-p", str(parallel)]
    return cmd


def split_scenarios(raw) -> "list[str] | None":
    """``--ts-scenarios a,b`` -> ["a", "b"]; empty -> None (full/test mode)."""
    if not raw:
        return None
    items = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in raw])
    return [s for s in items if s] or None


def _rapid_api_key_from_file(path: str) -> str:
    """Read one literal RAPID_API_KEY assignment without evaluating the file."""
    try:
        lines = Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise RuntimeError("ToolSandbox credential file could not be read") from None
    matches = []
    for line in lines:
        assignment = line.strip()
        if assignment.startswith("export "):
            assignment = assignment[len("export "):].lstrip()
        name, separator, value = assignment.partition("=")
        if separator and name.strip() == "RAPID_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            matches.append(value)
    if len(matches) != 1 or not matches[0]:
        raise ValueError("ToolSandbox credential file requires one nonempty RAPID_API_KEY")
    return matches[0]


def harness_env(base_url: str, user_base_url: str = "") -> Dict[str, str]:
    """``user_base_url`` (default: the raw upstream endpoint) routes the
    user simulator OUT of the arm proxy via TOOLSANDBOX_USER_BASE_URL —
    the patched openai_api_user role reads it.  Routing the simulator
    through the compression arm made every historical TS number an
    agent+user joint degradation (audit BLOCKER)."""
    env = os.environ.copy()
    # Polars reads this when the official ToolSandbox worker starts.
    env.setdefault("POLARS_MAX_THREADS", "4")
    credential_file = env.pop("TOOLSANDBOX_ENV_FILE", None)
    if not env.get("RAPID_API_KEY") and credential_file:
        env["RAPID_API_KEY"] = _rapid_api_key_from_file(credential_file)
    return {
        **env,
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_API_KEY_USER": "EMPTY",
        "OPENAI_BASE_URL": v1(base_url),
        # default: same endpoint the proxy itself fronts (full mode)
        "TOOLSANDBOX_USER_BASE_URL": v1(user_base_url) if user_base_url
        else os.environ.get("TOOLSANDBOX_USER_BASE_URL", v1(base_url)),
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }


def run(ctx: RunContext) -> Dict[str, Any]:
    """Drive the official ``tool_sandbox`` CLI against the arm proxy.

    No cost join: see ``COST_JOIN`` below.
    """
    explicit = split_scenarios(ctx.opt("ts_scenarios", ""))
    suite = ctx.opt("ts_suite", "")
    if suite == THREE_DISTRACTION_TOOLS_129 and explicit is not None:
        if explicit != selected_scenarios(suite):
            raise ValueError("ToolSandbox explicit scenarios differ from named suite")
    scenarios = selected_scenarios(suite, explicit)
    agent_timeout = agent_client_timeout(
        ctx.opt("generation_timeout", DEFAULT_GENERATION_TIMEOUT))
    summary = run_ts(
        ctx.base_url, ctx.out_dir,
        test_mode=not (ctx.options.get("full", False) or suite == "full"),
        agent=ctx.opt("ts_agent", AGENT), user=ctx.opt("ts_user", AGENT),
        # the user simulator must NOT ride the arm proxy: route it to the
        # raw upstream endpoint (tau2 already does the same split)
        user_base_url=ctx.user_base_url,
        scenarios=scenarios, suite=suite,
        benchmark_dir=ctx.opt("toolsandbox_dir"), python=ctx.opt("bench_python"),
        parallel=ctx.opt("ts_parallel", 1), model=ctx.model,
        **({} if agent_timeout is None else {"agent_timeout": agent_timeout}),
    )
    summary["cost_join"] = COST_JOIN
    return summary


COST_JOIN = ("joinable: toolsandbox_cli emits scenario/session/request/action "
             "ids without changing official execution or scoring")
RESCORE_INPUTS = ("toolsandbox_protocol.json", "scenario_manifest.json",
                  "agent_*/result_summary.json", "measurement/harness_events.jsonl",
                  "measurement/rapidapi_http_status.jsonl", "logs/proxy_*.jsonl")
RESCORE_PROCEDURE = "toolsandbox_adapter.score_cli_run on the saved official result summaries"


def rescore(ctx: RunContext, workspace: Path) -> Dict[str, Any]:
    """Offline rescore: run()'s post-CLI half on the saved official results."""
    out_dir = Path(ctx.out_dir).resolve()
    protocol = json.loads((out_dir / "toolsandbox_protocol.json").read_text(encoding="utf-8"))
    summary = score_cli_run(out_dir, scenarios=protocol["scenarios"])
    summary["cost_join"] = COST_JOIN
    return summary


def _proxy_scenario_rows(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Join each scenario to its last attributable proxy row."""
    from measurement.telemetry import read_jsonl

    events = out_dir / "measurement" / "harness_events.jsonl"
    if not events.is_file():
        return {}
    by_conversation = {}
    for row in read_jsonl(events):
        if row.get("event_type") != "episode_start":
            continue
        instance = row.get("episode_instance_id")
        scenario = row.get("episode_id")
        if not isinstance(instance, str) or not isinstance(scenario, str):
            continue
        key = hashlib.sha256(json.dumps(
            ["measurement_session", instance], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        by_conversation[key] = scenario
    latest = {}
    for path in sorted((out_dir / "logs").glob("proxy_*.jsonl")):
        for row in read_jsonl(path):
            scenario = by_conversation.get(row.get("conv_id"))
            if scenario:
                latest[scenario] = row
    return latest


def _invalid_retrieval_scenarios(out_dir: Path) -> set[str]:
    """Join typed proxy failures to official scenario names through the harness."""
    return {scenario for scenario, row in _proxy_scenario_rows(out_dir).items()
            if row.get("status") in {"textarm_error", "upstream_error"}
            and any(marker in str(row.get("error") or "") for marker in (
                "HiAgent requested nonexistent completed subgoals",
                "HiAgent requested an already revealed trajectory without advancing",
                "HiAgent exceeded four internal trajectory retrieval rounds",
                "HiAgent mixed internal retrieval and environment actions",
                "malformed hiagent_retrieve subgoal_ids",
            ))}


def _declared_budget_failure_scenarios(out_dir: Path) -> Dict[str, str]:
    """Return scenario-bound ACON/HiAgent budget declarations from proxy logs."""
    declared = {}
    for scenario, row in _proxy_scenario_rows(out_dir).items():
        code = proxy_text_budget_failure_code(row)
        if code is not None:
            declared[scenario] = code
    return declared


def reject_rapidapi_http_failures(out_dir: Path) -> None:
    path = Path(out_dir) / "measurement" / "rapidapi_http_status.jsonl"
    if not path.exists():
        return
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit("FATAL: ToolSandbox RapidAPI HTTP status log is invalid") from None
        status = row.get("status_code") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("event_type") != "rapidapi_http"
                or type(row.get("host")) is not str or type(status) is not int):
            raise SystemExit("FATAL: ToolSandbox RapidAPI HTTP status is unavailable")
        if status in (401, 403, 429) or status >= 500:
            failures.append((row["host"], status))
    if failures:
        raise SystemExit("FATAL: ToolSandbox RapidAPI infrastructure HTTP failure: "
                         + ", ".join(f"{host}={status}" for host, status in failures[:10]))


def run_ts(base_url: str, out_dir: Path, test_mode: bool = True,
           agent: str = AGENT, user: str = AGENT, expected: int = None,
           benchmark_dir: Path = None, user_base_url: str = "",
           scenarios: "list[str] | None" = None,
           suite: str = "",
           python: "str | None" = None, parallel: int = 1,
           model: str = "c2kv-agent", user_model: "str | None" = None,
           native_server_dir: "Path | None" = None,
           agent_timeout: "float | None" = None) -> Dict[str, Any]:
    """Run the CLI and collect ``result_summary.json``.

    ``agent_timeout`` (see ``agent_client_timeout``) replaces the agent SDK's
    600 s timeout; ``None`` keeps the historical CLI unchanged.
    """
    if parallel != 1:
        raise ValueError("instrumented ToolSandbox runs require parallel=1")
    if suite == THREE_DISTRACTION_TOOLS_129 and scenarios != selected_scenarios(suite):
        raise ValueError("ToolSandbox named suite IDs differ from frozen cohort")
    require_sglang_tool_parser(base_url, user_base_url)
    ts_dir = (Path(benchmark_dir) if benchmark_dir else TS_DIR).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = harness_env(base_url, user_base_url)
    env["C2KV_TOOLSANDBOX_MODEL"] = model
    env["C2KV_TOOLSANDBOX_USER_MODEL"] = user_model or model
    env["C2KV_TOOLSANDBOX_TELEMETRY"] = str(out_dir / "measurement" / "harness_events.jsonl")
    # The console script's directory is otherwise first on sys.path, and an
    # editable installation may silently import a different checkout.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ts_dir.resolve()), env.get("PYTHONPATH"))))
    env.pop(AGENT_TIMEOUT_ENV, None)
    if agent_timeout is not None:
        env[AGENT_TIMEOUT_ENV] = repr(float(agent_timeout))
    cmd = cli_command(out_dir, agent=agent, user=user, test_mode=test_mode,
                      scenarios=scenarios,
                      parallel=parallel)
    cmd[:1] = [python or sys.executable,
               str(Path(__file__).resolve().parents[1] / "toolsandbox_cli.py")]
    cohort = load_named_suite(suite) if suite == THREE_DISTRACTION_TOOLS_129 else None
    (out_dir / "toolsandbox_protocol.json").write_text(json.dumps({
        "suite": "subset" if scenarios else "test" if test_mode else "full",
        "named_suite": suite or None,
        "cohort": ({key: cohort[key] for key in
                    ("source_checkout_head", "scenario_count", "scenario_ids_sha256")}
                   if cohort is not None else None),
        "scenarios": scenarios, "parallel": int(parallel), "model": model,
        "user_model": user_model or model,
        "agent_role": agent, "user_role": user, "source": str(ts_dir),
        "command": cmd, "measurement": "runtime_scenario_request_action_v1",
        **({} if agent_timeout is None else {"agent_timeout": float(agent_timeout)}),
    }, indent=2) + "\n", encoding="utf-8")
    completed = run_owned(cmd, cwd=ts_dir, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"FATAL: tool_sandbox CLI exited {completed.returncode}")
    return score_cli_run(out_dir, scenarios=scenarios, expected=expected,
                         native_server_dir=native_server_dir)


def score_cli_run(out_dir: Path, *, scenarios: "list[str] | None", expected: int = None,
                  native_server_dir: "Path | None" = None) -> Dict[str, Any]:
    """Post-CLI half of run_ts: collect official summaries, check terminal state.

    The live run and an offline rescore share this function; it only reads
    ``out_dir``.
    """
    reject_rapidapi_http_failures(out_dir)
    summary = (collect(out_dir) if native_server_dir is None else
               collect(out_dir, native_server_dir=Path(native_server_dir)))
    summary["protocol"] = json.loads((out_dir / "toolsandbox_protocol.json").read_text(encoding="utf-8"))
    manifest_path = out_dir / "scenario_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("FATAL: ToolSandbox wrapper wrote no scenario_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_ids = [str(value) for value in manifest.get("scenario_ids") or []]
    expected_ids = set(resolved_ids)
    scored_list = summary.get("scenario_ids") or []
    scored_ids = set(scored_list)
    if len(expected_ids) != len(resolved_ids) or manifest.get("expected") != len(resolved_ids):
        raise SystemExit("FATAL: ToolSandbox resolver manifest has duplicate or missing IDs")
    if scenarios is not None and expected_ids != set(scenarios):
        raise SystemExit("FATAL: official ToolSandbox resolver differed from selected scenario IDs")
    if scored_ids != expected_ids or len(scored_list) != len(expected_ids):
        missing = sorted(expected_ids - scored_ids)
        extra = sorted(scored_ids - expected_ids)
        raise SystemExit(
            f"FATAL: ToolSandbox terminal-state mismatch: missing={missing[:10]} "
            f"extra={extra[:10]} scored={len(scored_list)} expected={len(expected_ids)}")
    summary["scenario_manifest"] = manifest
    # terminal-state check (acceptance 1): a scenario that never ran must
    # fail the run, not shrink the denominator
    n_scored = summary.get("n") if isinstance(summary, dict) else None
    if type(n_scored) is not int or n_scored != len(resolved_ids):
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: n_scored={n_scored} "
            f"n_total={len(resolved_ids)}")
    if expected is not None and n_scored is not None and n_scored != expected:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: n_scored={n_scored} != n_total={expected}")
    if expected is not None:
        print(f"TERMINAL-STATE ts: n_scored={n_scored} n_total={expected}")
    return summary


def collect(out_dir: Path, native_server_dir: "Path | None" = None) -> Dict[str, Any]:
    reject_rapidapi_http_failures(out_dir)
    summaries = sorted(out_dir.glob("agent_*/result_summary.json"))
    if not summaries:
        raise SystemExit(f"FATAL: no result_summary.json under {out_dir} — "
                         "the CLI produced nothing (check TS run logs)")
    from metrics import aggregate  # noqa: E402

    rows: List[Dict[str, Any]] = []
    crashed: List[str] = []
    invalid_retrieval_failures: List[str] = []
    capacity_failures: List[str] = []
    proxy_failures = _invalid_retrieval_scenarios(out_dir)
    proxy_budget_failures = _declared_budget_failure_scenarios(out_dir)
    budget_failures: Dict[str, List[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data.get("per_scenario_results") or []:
            if not isinstance(scenario, dict):
                raise SystemExit("FATAL: ToolSandbox official scenario result is not an object")
            scenario_id = scenario.get("name")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise SystemExit("FATAL: ToolSandbox official scenario result has no name")
            if scenario_id in seen_ids:
                raise SystemExit(f"FATAL: duplicate ToolSandbox official scenario: {scenario_id}")
            seen_ids.add(scenario_id)
            traceback = scenario.get("traceback")
            if traceback:
                # A request for nonexistent completed history is an actor
                # action, while an arbitrary 502 or runner crash is not.
                capacity_failure = None
                if native_server_dir is not None:
                    from native_budget_failure import native_capacity_failure
                    capacity_failure = native_capacity_failure(
                        Path(native_server_dir), scenario_id, "toolsandbox")
                traceback_budget_failure = typed_text_budget_failure_code(traceback)
                proxy_budget_failure = proxy_budget_failures.get(scenario_id)
                if (traceback_budget_failure is not None
                        and traceback_budget_failure == proxy_budget_failure
                        and scenario.get("exception_type") == "UnprocessableEntityError"):
                    budget_failures[traceback_budget_failure].append(scenario_id)
                    rows.append({"task_id": scenario_id, "semantic_score": 0.0,
                                 "official_similarity": None,
                                 "task_failure_kind": traceback_budget_failure,
                                 "protocol_legal": None})
                elif (capacity_failure == "c2kv_capacity_infeasible"
                        and scenario.get("exception_type") == "UnprocessableEntityError"):
                    capacity_failures.append(scenario_id)
                    rows.append({"task_id": scenario_id, "semantic_score": 0.0,
                                 "official_similarity": None,
                                 "task_failure_kind": capacity_failure,
                                 "protocol_legal": None})
                elif any(marker in str(traceback) for marker in (
                    "HiAgent requested nonexistent completed subgoals",
                    "HiAgent requested an already revealed trajectory without advancing",
                    "HiAgent exceeded four internal trajectory retrieval rounds",
                    "HiAgent mixed internal retrieval and environment actions",
                    "malformed hiagent_retrieve subgoal_ids",
                )) or (scenario_id in proxy_failures and "502" in str(traceback)):
                    invalid_retrieval_failures.append(scenario_id)
                    rows.append({"task_id": scenario_id, "semantic_score": 0.0,
                                 "official_similarity": None,
                                 "task_failure_kind": "hiagent_invalid_retrieval",
                                 "protocol_legal": None})
                else:
                    crashed.append(scenario_id)
                continue
            similarity = scenario.get("similarity")
            if (type(similarity) not in (int, float)
                    or not math.isfinite(float(similarity))):
                raise SystemExit(f"FATAL: ToolSandbox official similarity is unavailable: {scenario_id}")
            rows.append({
                "task_id": scenario_id,
                # official semantic column: dialogue similarity to the
                # reference (milestone-weighted); minefield = violations
                "semantic_score": similarity,
                "milestone_similarity": scenario.get("milestone_similarity"),
                "minefield_similarity": scenario.get("minefield_similarity"),
                "turn_count": scenario.get("turn_count"),
                "protocol_legal": None,  # TS has no tool-call legality metric
            })
    if crashed:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: {len(crashed)} scenario(s) "
            f"crashed (traceback in result_summary): {', '.join(crashed[:10])}")
    summary = aggregate(rows, cluster_key="task_id")
    summary["scenario_ids"] = sorted(str(row["task_id"]) for row in rows)
    summary["task_failures"] = {
        "hiagent_invalid_retrieval": sorted(invalid_retrieval_failures)}
    if capacity_failures:
        summary["task_failures"]["c2kv_capacity_infeasible"] = sorted(capacity_failures)
    for code, scenario_ids in sorted(budget_failures.items()):
        summary["task_failures"][code] = sorted(scenario_ids)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true",
                        help="run the full suite instead of test mode")
    parser.add_argument("--agent", default=AGENT,
                        help="agent role key (openai_api_agent config entry)")
    parser.add_argument("--user", default=AGENT,
                        help="user-simulator role key (openai_api_user config entry)")
    parser.add_argument("--ts-dir", type=Path, default=None,
                        help="ToolSandbox checkout (default $TS_DIR or ~/benchmarks/ToolSandbox)")
    args = parser.parse_args()
    summary = run_ts(args.base_url, args.out, test_mode=not args.full,
                     agent=args.agent, user=args.user, benchmark_dir=args.ts_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
