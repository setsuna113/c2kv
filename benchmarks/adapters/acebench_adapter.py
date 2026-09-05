"""ACEBench adapter (official ``generate.py`` / ``eval_main.py`` driver).

ACEBench/ACEBench evaluates tool use in three groups (normal / special /
agent).  This adapter drives the official scripts against the arm proxy and
parses their result and score files into unified rows.

Endpoints.  Upstream keys every client on the model NAME (``"gpt" in name``
-> GPT_* env vars, ...), so a served model name like ``c2kv-agent`` cannot
be routed at all.  The vendored ``benchmarks/acebench_patches/0001-endpoint-
env-and-model-registry.patch`` adds explicit overrides which this adapter
exports:

* ``ACEBENCH_AGENT_BASE_URL`` / ``_API_KEY`` — every client that IS the
  evaluated agent (single-turn inference, multi-turn agent, multi-step
  agent) -> the arm proxy;
* ``ACEBENCH_USER_BASE_URL`` / ``_API_KEY``  — the user simulator of the
  agent group -> the raw upstream (full mode; never the arm proxy);
* ``ACEBENCH_API_MODELS`` — registers the served model name in
  ``inference_map`` as an API model.

What the proxy sees — READ THIS before quoting an ACEBench arm number.  The
agent request is always exactly two messages: a system prompt and ONE user
message that embeds the whole ``user:/agent:/execution:`` transcript as
text (``multi_turn/APIModel_agent.py:respond``, ``multi_step`` likewise;
single-turn categories are system + question).  Under the training rule
nothing before the last user message exists, so every KV arm and every text
arm assembles zero history docs: an ACEBench column is a full-arm number
for every arm by construction.  It measures the served model's tool-calling
competence, not compression.  The request log's ``n_docs`` is the proof
(0 on every row).  The same shape makes ``proxy.conversation_id`` change on
every turn (it keys on the first two non-system messages and the single
user message grows), so any conversation-keyed arm state (recover, ACON
rolling) is meaningless here — another reason to run ``--arm full`` only.

Semantic column: the official checker (``eval_main.py``).  The score file
is one header row (accuracy / end_to_end_accuracy, process_accuracy,
correct_count, total_count) followed by one row per FAILED item; agent
categories key failures by index into the id-sorted result file, the other
categories by item id.  Rows here are per item (1/0); for
``normal_multi_turn_*`` the bootstrap cluster is the turn group (the id
without its item suffix), which is the unit the official accuracy is
computed over.

Working directory.  The scripts resolve ``./data_all``, ``./result_all`` and
``./score_all`` from cwd and ``generate.py`` resumes any id already present
in ``result_all``; each run therefore gets a private cwd under ``--out``
with ``data_all`` linked in, so one arm can never resume into another's
results.

Terminal-state check (acceptance 1): before scoring, every id of every
requested data file must have a result row, else the run FAILS.

Deviations from the official protocol (label them): the user simulator is
the same served model (upstream default ``gpt-4o``), ``--temperature 0``
(upstream default 0.7), English only (the paper table is zh+en combined).

Usage (server):
    python benchmarks/adapters/acebench_adapter.py \
        --base-url http://127.0.0.1:34100 --user-base-url http://127.0.0.1:35000 \
        --out results/bench/ace_c2kv --category agent
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate  # noqa: E402

from adapters.base import RunContext, v1  # noqa: E402

NAME = "acebench"
ACEBENCH_DIR = Path(os.environ.get("ACEBENCH_DIR") or Path.home() / "baselines" / "acebench")
AGENT_BASE_URL_ENV = "ACEBENCH_AGENT_BASE_URL"
AGENT_API_KEY_ENV = "ACEBENCH_AGENT_API_KEY"
USER_BASE_URL_ENV = "ACEBENCH_USER_BASE_URL"
USER_API_KEY_ENV = "ACEBENCH_USER_API_KEY"
MODELS_ENV = "ACEBENCH_API_MODELS"
DEFAULT_CATEGORY = "agent"
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_DIALOG_TURNS = 40
HEADER_KEYS = ("accuracy", "end_to_end_accuracy", "process_accuracy",
               "correct_count", "total_count")

# Why ACEBench gets no per-task cost columns — and would gain nothing from
# them.  The agent request is system + ONE user message carrying the whole
# transcript as text, so ``proxy.conversation_id`` (system head + first two
# non-system messages, proxy.py:434-447) changes on EVERY turn as that one
# message grows: there is no stable per-task conversation id to join on.
# The same shape is why every arm assembles zero docs here (module
# docstring), so the cost column would be a full-arm column anyway.
COST_JOIN = ("not joinable: the whole transcript rides in ONE growing user "
             "message, so the conversation id changes every turn")


def add_arguments(parser) -> None:
    """ACEBench-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--acebench-dir", type=Path, default=None,
                        help="ACEBench checkout (default $ACEBENCH_DIR or ~/baselines/acebench)")
    parser.add_argument("--acebench-category", default="agent",
                        help="ACE_DATA_CATEGORY key or one test name")
    parser.add_argument("--acebench-language", default="en", choices=["en", "zh"])
    parser.add_argument("--user-model", default="",
                        help="acebench: user-simulator model name at --user-upstream "
                             "(default: --model)")


def load_category_map(acebench_dir: Path) -> Dict[str, List[str]]:
    """``category.py:ACE_DATA_CATEGORY`` from the checkout (no package import:
    the checkout is not a package and the module name is generic)."""
    path = Path(acebench_dir) / "category.py"
    spec = importlib.util.spec_from_file_location("acebench_category", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: list(v) for k, v in module.ACE_DATA_CATEGORY.items()}


def expand_categories(category: str, category_map: Dict[str, List[str]]) -> List[str]:
    """eval_main.py rule: a category key expands to its test names, anything
    else is taken as a test name itself."""
    return list(category_map.get(category, [category]))


def harness_env(base_url: str, user_base_url: str, model: str) -> Dict[str, str]:
    """Agent clients -> arm proxy; user simulator -> raw upstream (full)."""
    return {
        **os.environ,
        AGENT_BASE_URL_ENV: v1(base_url),
        AGENT_API_KEY_ENV: "EMPTY",
        USER_BASE_URL_ENV: v1(user_base_url or base_url),
        USER_API_KEY_ENV: "EMPTY",
        MODELS_ENV: model,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }


def prepare_workdir(out_dir: Path, acebench_dir: Path) -> Path:
    work = Path(out_dir) / "acebench_work"
    work.mkdir(parents=True, exist_ok=True)
    data = work / "data_all"
    if not data.exists():
        source = Path(acebench_dir) / "data_all"
        try:
            os.symlink(source, data, target_is_directory=True)
        except OSError:
            shutil.copytree(source, data)
    return work


def generate_command(python: str, acebench_dir: Path, model: str, category: str,
                     language: str, num_threads: int, max_dialog_turns: int,
                     user_model: str, temperature: float, top_p: float,
                     max_tokens: int) -> List[str]:
    return [python, str(Path(acebench_dir) / "generate.py"),
            "--model", model, "--category", category, "--language", language,
            "--num-threads", str(num_threads),
            "--max-dialog-turns", str(max_dialog_turns),
            "--user-model", user_model,
            "--temperature", str(temperature), "--top-p", str(top_p),
            "--max-tokens", str(max_tokens)]


def eval_command(python: str, acebench_dir: Path, model: str, category: str,
                 language: str) -> List[str]:
    return [python, str(Path(acebench_dir) / "eval_main.py"),
            "--model", model, "--category", category, "--language", language]


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def data_path(work: Path, language: str, test: str) -> Path:
    return Path(work) / "data_all" / f"data_{language}" / f"data_{test}.json"


def result_path(work: Path, language: str, model: str, test: str) -> Path:
    return Path(work) / "result_all" / f"result_{language}" / model / f"data_{test}_result.json"


def score_path(work: Path, language: str, model: str, test: str) -> Path:
    return Path(work) / "score_all" / f"score_{language}" / model / f"data_{test}_score.json"


def check_terminal(work: Path, language: str, model: str, tests: List[str]) -> None:
    """Every id in every requested data file has a result row (the official
    evaluator would otherwise raise on the length mismatch, and a partial
    result file would shrink the denominator on a rerun)."""
    for test in tests:
        want = {str(r["id"]) for r in _jsonl(data_path(work, language, test))}
        results = result_path(work, language, model, test)
        got = {str(r["id"]) for r in _jsonl(results)} if results.exists() else set()
        missing = sorted(want - got)
        print(f"TERMINAL-STATE acebench/{test}: n_scored={len(want & got)} n_total={len(want)}")
        if missing:
            shown = ",".join(missing[:20])
            more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
            raise SystemExit(f"FATAL: acebench {test} has no result for: {shown}{more}")


def cluster_id(test: str, task_id: str) -> str:
    """Official unit for normal_multi_turn_* is the turn group (ids are
    ``<test>_<turn>_<item>``); everything else is scored per item."""
    if test.startswith("normal_multi_turn"):
        return task_id.rsplit("_", 1)[0]
    return task_id


def failed_task_ids(results: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Set[str]:
    """eval_main.py writes one row per failed item after the header.  Agent
    categories key it by INDEX into the id-sorted result file (``"id": i``);
    normal / special categories by the item id string."""
    failed: Set[str] = set()
    for row in failures:
        fid = row.get("id")
        if isinstance(fid, bool) or fid is None:
            continue
        if isinstance(fid, int):
            if 0 <= fid < len(results):
                failed.add(str(results[fid]["id"]))
        else:
            failed.add(str(fid))
    return failed


def collect(work: Path, language: str, model: str, tests: List[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_category: Dict[str, Dict[str, Any]] = {}
    for test in tests:
        results = _jsonl(result_path(work, language, model, test))
        score_file = score_path(work, language, model, test)
        if not score_file.exists():
            raise SystemExit(f"FATAL: eval_main.py wrote no {score_file}")
        score = _jsonl(score_file)
        if not score:
            raise SystemExit(f"FATAL: empty score file {score_file}")
        header, failures = score[0], score[1:]
        failed = failed_task_ids(results, failures)
        for rec in results:
            task_id = str(rec["id"])
            rows.append({
                "task_id": task_id,
                "cluster": cluster_id(test, task_id),
                "category": test,
                "semantic_score": 0.0 if task_id in failed else 1.0,
                "protocol_legal": None,  # tools are prompt text, not a schema
            })
        per_category[test] = {k: header[k] for k in HEADER_KEYS if k in header}
    summary = aggregate(rows, cluster_key="cluster")
    summary["per_category"] = per_category
    summary["categories"] = list(tests)
    summary["workdir"] = str(work)
    return summary


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: drive generate.py / eval_main.py against the arm proxy.

    The user simulator must NOT ride the arm proxy (same split as tau2 /
    toolsandbox).  NOTE the agent request is system + ONE user message, so
    no arm compresses anything here — see the module docstring.
    """
    summary = run_acebench(
        ctx.base_url, ctx.user_base_url, ctx.out_dir,
        acebench_dir=ctx.opt("acebench_dir"),
        category=ctx.opt("acebench_category", DEFAULT_CATEGORY),
        language=ctx.opt("acebench_language", DEFAULT_LANGUAGE),
        model=ctx.model, user_model=ctx.opt("user_model"),
        num_threads=ctx.opt("num_workers", 1),
        max_dialog_turns=ctx.opt("max_iter", DEFAULT_MAX_DIALOG_TURNS),
        python=ctx.opt("bench_python"),
    )
    summary["cost_join"] = COST_JOIN
    return summary


def run_acebench(base_url: str, user_base_url: str, out_dir: Path,
                 acebench_dir: Optional[Path] = None, category: str = DEFAULT_CATEGORY,
                 language: str = DEFAULT_LANGUAGE, model: str = "c2kv-agent",
                 user_model: Optional[str] = None, num_threads: int = 1,
                 max_dialog_turns: int = DEFAULT_MAX_DIALOG_TURNS,
                 temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 1200,
                 python: Optional[str] = None) -> Dict[str, Any]:
    acebench_dir = Path(acebench_dir) if acebench_dir else ACEBENCH_DIR
    python = python or sys.executable
    tests = expand_categories(category, load_category_map(acebench_dir))
    work = prepare_workdir(out_dir, acebench_dir)
    env = harness_env(base_url, user_base_url, model)
    subprocess.run(
        generate_command(python, acebench_dir, model, category, language, num_threads,
                         max_dialog_turns, user_model or model, temperature, top_p,
                         max_tokens),
        cwd=work, env=env, check=True)
    check_terminal(work, language, model, tests)
    subprocess.run(eval_command(python, acebench_dir, model, category, language),
                   cwd=work, env=env, check=True)
    summary = collect(work, language, model, tests)
    summary["user_model"] = user_model or model
    summary["language"] = language
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", default="",
                        help="user-simulator endpoint (raw upstream; default = --base-url)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acebench-dir", type=Path, default=None,
                        help="ACEBench checkout (default $ACEBENCH_DIR or ~/baselines/acebench)")
    parser.add_argument("--category", default=DEFAULT_CATEGORY,
                        help="ACE_DATA_CATEGORY key (agent | multi_turn | normal | "
                             "special | test_all | ...) or one test name")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=["en", "zh"])
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--user-model", default="", help="default = --model")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-dialog-turns", type=int,
                        default=DEFAULT_MAX_DIALOG_TURNS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--python", default=None,
                        help="python of the ACEBench venv (default: this interpreter)")
    args = parser.parse_args()
    summary = run_acebench(args.base_url, args.user_base_url, args.out,
                           acebench_dir=args.acebench_dir, category=args.category,
                           language=args.language, model=args.model,
                           user_model=args.user_model or None,
                           num_threads=args.num_threads,
                           max_dialog_turns=args.max_dialog_turns,
                           temperature=args.temperature, python=args.python)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
