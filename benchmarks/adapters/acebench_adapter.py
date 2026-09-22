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

Role history. With ``ACEBENCH_ROLE_HISTORY_V1=1``, the patch passes the
upstream scene's structured ``dialogue_history`` to the agent client and
emits one canonical message per entry: ``user`` -> user, ``agent`` ->
assistant, ``execution`` -> tool. API definitions remain in the system
message. It never tries to recover roles by splitting the legacy opaque
transcript. Without the flag, upstream's original two-message request is
unchanged. The adapter always enables the flag and advertises capability
``acebench_role_history_v1`` so non-full arms have compressible history.
The independently stateful user simulator still uses the raw endpoint and
is never routed through compression.

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
from collections import Counter
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.process_lifecycle import run_owned  # noqa: E402
from metrics import aggregate  # noqa: E402

from adapters.base import RunContext, v1  # noqa: E402
from adapters import acebench_task_failures  # noqa: E402

NAME = "acebench"
ACEBENCH_DIR = Path(os.environ.get("ACEBENCH_DIR") or Path.home() / "baselines" / "acebench")
AGENT_BASE_URL_ENV = "ACEBENCH_AGENT_BASE_URL"
AGENT_API_KEY_ENV = "ACEBENCH_AGENT_API_KEY"
USER_BASE_URL_ENV = "ACEBENCH_USER_BASE_URL"
USER_API_KEY_ENV = "ACEBENCH_USER_API_KEY"
MODELS_ENV = "ACEBENCH_API_MODELS"
ROLE_HISTORY_ENV = "ACEBENCH_ROLE_HISTORY_V1"
TOOL_CONTEXT_ENV = "C2KV_TOOL_CONTEXT_ON"
TOOL_SPANS_PATCH = Path(__file__).resolve().parents[1] / "acebench_patches" / "0002-visible-tool-spans.patch"
CAPABILITY_FEATURES = ("acebench_role_history_v1",)
DEFAULT_CATEGORY = "agent"
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_DIALOG_TURNS = 40
HEADER_KEYS = ("accuracy", "end_to_end_accuracy", "process_accuracy",
               "correct_count", "total_count")

# Agent runs use acebench_cli.py to bind official task, request and executor
# identities. Normal/special splits have no agent executor instrumentation.
COST_JOIN = "official task -> episode session -> proxy request -> executed action"


def add_arguments(parser) -> None:
    """ACEBench-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--acebench-dir", type=Path, default=None,
                        help="ACEBench checkout (default $ACEBENCH_DIR or ~/baselines/acebench)")
    parser.add_argument("--acebench-category", default="agent",
                        help="ACE_DATA_CATEGORY key or one test name")
    parser.add_argument("--acebench-language", default="en", choices=["en", "zh"])
    parser.add_argument("--acebench-task-ids", default="",
                        help="comma-separated official ACEBench ids; creates a private filtered data_all")
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


def harness_env(base_url: str, user_base_url: str, model: str,
                *, record_source: bool = False) -> Dict[str, str]:
    """Agent clients -> role history + arm proxy; simulator -> raw upstream."""
    env = {
        **os.environ,
        AGENT_BASE_URL_ENV: v1(base_url),
        AGENT_API_KEY_ENV: "EMPTY",
        USER_BASE_URL_ENV: v1(user_base_url or base_url),
        USER_API_KEY_ENV: "EMPTY",
        MODELS_ENV: model,
        ROLE_HISTORY_ENV: "1",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    if os.environ.get(TOOL_CONTEXT_ENV) == "1" or record_source:
        benchmark_modules = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(part for part in (
            benchmark_modules, os.environ.get("PYTHONPATH", "")) if part)
    if record_source:
        env["C2KV_ACE_RECORD_SOURCE"] = "1"
    else:
        env.pop("C2KV_ACE_RECORD_SOURCE", None)
    return env


def _task_id_list(value: str | None) -> List[str]:
    values = [part.strip() for part in (value or "").split(",") if part.strip()]
    if len(set(values)) != len(values):
        raise SystemExit("FATAL: --acebench-task-ids contains a duplicate id")
    return values


def prepare_workdir(out_dir: Path, acebench_dir: Path, *, language: str | None = None,
                    tests: Optional[List[str]] = None, task_ids: str = "",
                    max_tasks: Optional[int] = None) -> Path:
    """Create a private harness cwd, optionally with exact official rows only.

    ``generate.py`` resolves ``./data_all`` from its cwd and resumes result
    ids, so subset selection must be materialised here rather than passed as
    an unofficial scorer flag.  The upstream files stay read-only; the
    selection record makes a finite smoke denominator auditable.
    """
    work = Path(out_dir) / "acebench_work"
    work.mkdir(parents=True, exist_ok=True)
    data = work / "data_all"
    selected_ids = _task_id_list(task_ids)
    subset = bool(selected_ids or max_tasks is not None)
    if max_tasks is not None and max_tasks <= 0:
        raise SystemExit("FATAL: --max-tasks for ACEBench must be positive")
    if not subset:
        if data.exists():
            return work
        source = Path(acebench_dir) / "data_all"
        try:
            os.symlink(source, data, target_is_directory=True)
        except OSError:
            shutil.copytree(source, data)
        return work

    if not language or not tests:
        raise ValueError("subset preparation requires language and resolved test names")
    if data.exists():
        raise SystemExit(f"FATAL: refusing to mix ACEBench subset data in existing {data}")

    wanted = set(selected_ids)
    matched: Set[str] = set()
    remaining = max_tasks
    prepared = []
    for test in tests:
        source = Path(acebench_dir) / "data_all" / f"data_{language}" / f"data_{test}.json"
        rows = _jsonl(source)
        candidates = [row for row in rows if not wanted or str(row.get("id")) in wanted]
        if remaining is not None:
            candidates = candidates[:remaining]
            remaining -= len(candidates)
        if not candidates:
            # A finite category subset may spend its full budget in an earlier
            # test file; an explicit task id may similarly belong to a later
            # file.  The private category mapping below exposes only sources
            # that actually contributed rows.
            continue
        answer_source = source.parent / "possible_answer" / source.name
        answers = _jsonl(answer_source)
        answer_by_id = {str(row.get("id")): row for row in answers}
        answer_rows = [answer_by_id.get(str(row["id"])) for row in candidates]
        missing_answers = [str(row["id"]) for row, answer in zip(candidates, answer_rows)
                           if answer is None]
        if missing_answers:
            raise SystemExit(
                f"FATAL: ACEBench possible_answer has no rows for: {','.join(missing_answers)}")
        matched.update(str(row["id"]) for row in candidates)
        prepared.append((test, source, rows, candidates, answer_source, answer_rows))
    if not prepared:
        raise SystemExit("FATAL: ACEBench subset selected no official rows")
    missing = sorted(wanted - matched)
    if missing:
        raise SystemExit(f"FATAL: ACEBench task ids not found in requested category: {','.join(missing)}")

    target_dir = data / f"data_{language}"
    target_dir.mkdir(parents=True)
    answer_dir = target_dir / "possible_answer"
    answer_dir.mkdir()
    sources = []
    for test, source, rows, candidates, answer_source, answer_rows in prepared:
        target = target_dir / f"data_{test}.json"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in candidates), encoding="utf-8")
        answer_target = answer_dir / f"data_{test}.json"
        answer_target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                          for row in answer_rows), encoding="utf-8")
        sources.append({
            "test": test, "source_path": str(source), "source_rows": len(rows),
            "selected_rows": len(candidates),
            "selected_ids": [str(row["id"]) for row in candidates],
            "possible_answer_source_path": str(answer_source),
            "possible_answer_selected_rows": len(answer_rows),
        })
    (work / "selected_tasks.json").write_text(json.dumps({
        "schema_version": 1, "language": language, "requested_task_ids": selected_ids,
        "max_tasks": max_tasks, "sources": sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return work


def prepare_subset_harness(work: Path, acebench_dir: Path, category: str,
                           selected_tests: List[str]) -> Path:
    """Copy a patched, private ACEBench harness with a narrow category map.

    The upstream generator accepts only names defined in ``category.py`` and
    always resolves that module beside ``generate.py``.  A copied harness is
    therefore the smallest way to run one official test from a multi-test
    category without patching source data, the upstream scorer, or the shared
    checkout.  Its data files still resolve from the private workdir.
    """
    harness = work / "acebench_harness"
    if harness.exists():
        raise SystemExit(f"FATAL: refusing to reuse existing ACEBench subset harness {harness}")
    shutil.copytree(acebench_dir, harness, ignore=shutil.ignore_patterns(
        "data_all", "result_all", "score_all", "__pycache__", ".git"))
    (harness / "category.py").write_text(
        "# Private finite-smoke category; official generator/scorer are unchanged.\n"
        f"ACE_DATA_CATEGORY = {{{category!r}: {list(selected_tests)!r}}}\n",
        encoding="utf-8")
    return harness


def _tool_spans_installed(harness: Path) -> bool:
    source = Path(harness) / "model_inference"
    checks = (
        (source / "role_history.py", "def tool_context_kwargs"),
        (source / "apimodel_inference.py", "tool_context_kwargs(message, functions)"),
        (source / "multi_step" / "APIModel_agent.py", "tool_context_kwargs(message, self.functions)"),
        (source / "multi_turn" / "APIModel_agent.py", "tool_context_kwargs(message, self.functions)"),
    )
    return all(path.is_file() and marker in path.read_text(encoding="utf-8")
               for path, marker in checks)


def prepare_tool_span_harness(work: Path, source: Path) -> Path:
    """Apply the source-span patch to this run's private ACEBench code only."""
    work, source = Path(work).resolve(), Path(source).resolve()
    if source.is_relative_to(work):
        harness = source  # prepare_subset_harness already made a private copy
    else:
        harness = work / "acebench_tool_harness"
        if harness.exists():
            raise SystemExit(f"FATAL: refusing to reuse existing ACEBench tool harness {harness}")
        shutil.copytree(source, harness, ignore=shutil.ignore_patterns(
            "data_all", "result_all", "score_all", "__pycache__", ".git"))
    if _tool_spans_installed(harness):
        return harness
    if not TOOL_SPANS_PATCH.is_file():
        raise SystemExit(f"FATAL: missing ACEBench tool source patch {TOOL_SPANS_PATCH}")
    checked = subprocess.run(["git", "apply", "--check", str(TOOL_SPANS_PATCH)],
                             cwd=harness, capture_output=True, text=True)
    if checked.returncode:
        raise SystemExit("FATAL: ACEBench tool source patch does not apply to the "
                         f"private harness: {checked.stderr.strip()}")
    subprocess.run(["git", "apply", str(TOOL_SPANS_PATCH)], cwd=harness, check=True)
    if not _tool_spans_installed(harness):
        raise SystemExit("FATAL: ACEBench tool source patch lacks required request sites")
    return harness


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


def prepare_score_dir(work: Path, language: str, model: str) -> Path:
    """Create the parent path assumed by ACEBench's agent process scorer."""
    path = Path(work) / "score_all" / f"score_{language}" / model.replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def data_path(work: Path, language: str, test: str) -> Path:
    return Path(work) / "data_all" / f"data_{language}" / f"data_{test}.json"


def result_path(work: Path, language: str, model: str, test: str) -> Path:
    return Path(work) / "result_all" / f"result_{language}" / model / f"data_{test}_result.json"


def score_path(work: Path, language: str, model: str, test: str) -> Path:
    return Path(work) / "score_all" / f"score_{language}" / model / f"data_{test}_score.json"


def check_terminal(work: Path, language: str, model: str, tests: List[str],
                   declared: Optional[Set[str]] = None) -> None:
    """Result ids are an exact one-to-one match for the requested data ids.

    ``declared`` task failures (verified receipts) are the only ids that may
    lack a result row.
    """
    declared = set(declared or ())
    for test in tests:
        want_ids = [str(r["id"]) for r in _jsonl(data_path(work, language, test))]
        results = result_path(work, language, model, test)
        got_ids = [str(r["id"]) for r in _jsonl(results)] if results.exists() else []
        want_counts = Counter(want_ids)
        got_counts = Counter(got_ids)
        duplicate_data = sorted(key for key, count in want_counts.items() if count != 1)
        duplicate_results = sorted(key for key, count in got_counts.items() if count != 1)
        want, got = set(want_counts), set(got_counts)
        missing = sorted(want - got - declared)
        extra = sorted(got - want)
        print(f"TERMINAL-STATE acebench/{test}: n_scored={len(want & got)} n_total={len(want)}")
        problems = []
        for label, values in (
            ("duplicate data ids", duplicate_data),
            ("duplicate result ids", duplicate_results),
            ("missing result ids", missing),
            ("unexpected result ids", extra),
            ("declared task failures with results", sorted(declared & got)),
        ):
            if values:
                shown = ",".join(values[:20])
                more = f" (+{len(values) - 20} more)" if len(values) > 20 else ""
                problems.append(f"{label}: {shown}{more}")
        if problems:
            raise SystemExit(f"FATAL: acebench {test} terminal id mismatch; " + "; ".join(problems))


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


def collect(work: Path, language: str, model: str, tests: List[str],
            task_failures: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Unified rows from the official scorer; declared failures score 0."""
    task_failures = dict(task_failures or {})
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
        test_rows = []
        for rec in results:
            task_id = str(rec["id"])
            test_rows.append({
                "task_id": task_id,
                "cluster": cluster_id(test, task_id),
                "category": test,
                "semantic_score": 0.0 if task_id in failed else 1.0,
                "protocol_legal": None,  # tools are prompt text, not a schema
            })
        declared = [task_id for task_id in task_failures
                    if task_id.rsplit("_", 1)[0] == test]
        if declared:
            test_rows += [{"task_id": task_id, "cluster": cluster_id(test, task_id),
                           "category": test, "semantic_score": 0.0,
                           "protocol_legal": None,
                           "task_failure_kind": task_failures[task_id]}
                          for task_id in declared]
            # generate.py's result order (sort_json), now including the failures
            test_rows.sort(key=lambda row: int(row["task_id"].split("_")[-1]))
        rows += test_rows
        per_category[test] = {k: header[k] for k in HEADER_KEYS if k in header}
    if task_failures and (len({row["task_id"] for row in rows}) != len(rows)
                          or not set(task_failures) <= {row["task_id"] for row in rows}):
        raise SystemExit("FATAL: ACEBench declared task failures do not match the scored tests")
    summary = aggregate(rows, cluster_key="cluster")
    summary["per_category"] = per_category
    summary["categories"] = list(tests)
    summary["workdir"] = str(work)
    if task_failures:
        summary["n_official_scored"] = len(rows) - len(task_failures)
        summary["n_task_failures"] = len(task_failures)
        summary["task_failures"] = {
            code: sorted(task_id for task_id, kind in task_failures.items() if kind == code)
            for code in sorted(set(task_failures.values()))}
        summary["per_category_scope"] = (
            "official eval_main.py over the tasks without a declared task failure")
        summary["failure_score_policy"] = (
            "typed text-history budget failures are task-level method zeros; "
            "every other generator error still fails the run")
    return summary


def prepare_scoring_workdir(target: Path, work: Path, language: str, model: str,
                            tests: List[str], exclude: Set[str]) -> Path:
    """Private official-scorer cwd holding every requested row except ``exclude``.

    ``eval_main.py`` aligns data, possible answers and results by index, so a
    task without a result row must leave all three. Rows keep the data order.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    answers = target / "data_all" / f"data_{language}" / "possible_answer"
    answers.mkdir(parents=True)
    sources = []
    for test in tests:
        rows = _jsonl(data_path(work, language, test))
        answer_rows = _jsonl(data_path(work, language, test).parent / "possible_answer"
                             / f"data_{test}.json")
        if [str(row.get("id")) for row in answer_rows] != [str(row["id"]) for row in rows]:
            raise SystemExit(f"FATAL: ACEBench {test} possible answers are not aligned with data")
        results = {str(row["id"]): row for row in _jsonl(result_path(work, language, model, test))}
        kept = [index for index, row in enumerate(rows) if str(row["id"]) not in exclude]
        if not kept:
            raise SystemExit(f"FATAL: every ACEBench {test} task is a declared failure")
        for path, values in ((data_path(target, language, test), [rows[i] for i in kept]),
                             (answers / f"data_{test}.json", [answer_rows[i] for i in kept]),
                             (result_path(target, language, model, test),
                              [results[str(rows[i]["id"])] for i in kept])):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in values), encoding="utf-8")
        sources.append({"test": test, "source_rows": len(rows), "scored_rows": len(kept),
                        "excluded_ids": sorted(str(row["id"]) for row in rows
                                               if str(row["id"]) in exclude)})
    (target / "score_selection.json").write_text(json.dumps({
        "schema_version": 1, "generation_workdir": str(Path(work)), "sources": sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _declared_task_failures(out_dir: Path, work: Path, language: str, model: str,
                            tests: List[str]) -> Dict[str, str]:
    """Verified receipts, read only when the instrumented generator wrote any."""
    if not (Path(out_dir) / acebench_task_failures.RECEIPTS).is_file():
        return {}
    expected = {str(row["id"]) for test in tests for row in _jsonl(data_path(work, language, test))}
    produced = {str(row["id"]) for test in tests
                if result_path(work, language, model, test).exists()
                for row in _jsonl(result_path(work, language, model, test))}
    return acebench_task_failures.verified_task_failures(out_dir, expected, produced)


def score_generated_run(out_dir: Path, work: Path, harness: Path, *, python: str,
                        model: str, category: str, language: str, tests: List[str],
                        env: Dict[str, str], user_model: str,
                        score_work: Optional[Path] = None) -> Dict[str, Any]:
    """Post-generation half of run_acebench: terminal check, official scorer, rows.

    The live run and an offline rescore share this function. Verified declared
    task failures (see ``acebench_task_failures``), or an explicit
    ``score_work``, move the official scorer to a private cwd; otherwise it runs
    in the generation workdir exactly as before.
    """
    failures = _declared_task_failures(out_dir, work, language, model, tests)
    if failures:
        check_terminal(work, language, model, tests, declared=set(failures))
    else:
        check_terminal(work, language, model, tests)
    scoring = work
    if failures or score_work is not None:
        scoring = prepare_scoring_workdir(
            score_work if score_work is not None else Path(out_dir) / "acebench_score",
            work, language, model, tests, set(failures))
    prepare_score_dir(scoring, language, model)
    run_owned(eval_command(python, harness, model, category, language),
                   cwd=scoring, env=env, check=True)
    if failures:
        summary = collect(scoring, language, model, tests, task_failures=failures)
    else:
        summary = collect(scoring, language, model, tests)
    if scoring != work:
        summary["workdir"] = str(work)
        summary["score_workdir"] = str(scoring)
    summary["user_model"] = user_model
    summary["language"] = language
    selection = work / "selected_tasks.json"
    if selection.exists():
        summary["selection"] = json.loads(selection.read_text(encoding="utf-8"))
    summary["capability_features"] = list(CAPABILITY_FEATURES)
    summary["agent_history_protocol"] = "acebench_role_history_v1"
    summary["harness_telemetry"] = str(
        Path(out_dir).resolve() / "measurement" / "harness_events.jsonl")
    return summary


def _with_cost_join(summary: Dict[str, Any]) -> Dict[str, Any]:
    summary["cost_join"] = (COST_JOIN if all(c.startswith("agent_") for c in summary["categories"])
                            else "not joinable: non-agent split")
    return summary


def _declares_text_budget_failures(arm_name: Optional[str]) -> bool:
    """Only text-history budget arms can return the exact typed budget codes."""
    if arm_name is None:
        return False
    from arms import get_arm

    return get_arm(arm_name).text_history_budget_tokens is not None


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: drive generate.py / eval_main.py against the arm proxy.

    The user simulator stays on the raw endpoint (same split as tau2 /
    toolsandbox). The vendored patch supplies role-preserving agent history
    to the arm proxy; see the module docstring.
    """
    summary = run_acebench(
        ctx.base_url, ctx.user_base_url, ctx.out_dir,
        acebench_dir=ctx.opt("acebench_dir"),
        category=ctx.opt("acebench_category", DEFAULT_CATEGORY),
        language=ctx.opt("acebench_language", DEFAULT_LANGUAGE),
        model=ctx.model, user_model=ctx.opt("user_model"),
        num_threads=ctx.opt("num_workers", 1),
        max_dialog_turns=ctx.opt("max_iter", DEFAULT_MAX_DIALOG_TURNS),
        task_ids=ctx.opt("acebench_task_ids", ""), max_tasks=ctx.opt("max_tasks"),
        record_prefixes=ctx.opt("record_prefixes", ""),
        python=ctx.opt("bench_python"),
        declare_text_budget_failures=_declares_text_budget_failures(getattr(ctx, "arm", None)),
    )
    return _with_cost_join(summary)


RESCORE_INPUTS = ("acebench_work/selected_tasks.json", "acebench_work/data_all/**/*.json",
                  "acebench_work/result_all/**/*.json", "measurement/harness_events.jsonl",
                  acebench_task_failures.RECEIPTS.as_posix(), "logs/proxy_*.jsonl")
RESCORE_PROCEDURE = ("acebench_adapter.score_generated_run on the saved generation workdir; "
                     "official eval_main.py in a private scoring workdir")


def rescore(ctx: RunContext, workspace: Path) -> Dict[str, Any]:
    """Offline rescore: run()'s post-generation half on the saved workdir.

    The official scorer runs in ``workspace``; the cell's own generation
    workdir is only read. Tool-context and Full-recording cells used a
    private patched harness and are not supported.
    """
    if ctx.opt("tool_memory") or ctx.opt("record_prefixes"):
        raise ValueError("ACEBench rescore supports raw-tool, non-recording cells only")
    acebench_dir = Path(ctx.opt("acebench_dir") or ACEBENCH_DIR)
    category = ctx.opt("acebench_category", DEFAULT_CATEGORY)
    language = ctx.opt("acebench_language", DEFAULT_LANGUAGE)
    python = ctx.opt("bench_python") or sys.executable
    work = Path(ctx.out_dir) / "acebench_work"
    tests, harness = expand_categories(category, load_category_map(acebench_dir)), acebench_dir
    if (work / "selected_tasks.json").exists():
        selection = json.loads((work / "selected_tasks.json").read_text(encoding="utf-8"))
        tests = [str(source["test"]) for source in selection["sources"]]
        harness = work / "acebench_harness"
    if not (Path(harness) / "eval_main.py").is_file():
        raise FileNotFoundError(f"ACEBench scorer is missing: {Path(harness) / 'eval_main.py'}")
    env = harness_env(ctx.base_url, ctx.user_base_url, ctx.model)
    summary = score_generated_run(
        Path(ctx.out_dir), work, harness, python=python, model=ctx.model, category=category,
        language=language, tests=tests, env=env,
        user_model=ctx.opt("user_model") or ctx.model,
        score_work=Path(workspace) / "acebench_score")
    return _with_cost_join(summary)


def run_acebench(base_url: str, user_base_url: str, out_dir: Path,
                 acebench_dir: Optional[Path] = None, category: str = DEFAULT_CATEGORY,
                 language: str = DEFAULT_LANGUAGE, model: str = "c2kv-agent",
                 user_model: Optional[str] = None, num_threads: int = 1,
                 max_dialog_turns: int = DEFAULT_MAX_DIALOG_TURNS,
                 temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 1200, task_ids: str = "",
                 max_tasks: Optional[int] = None,
                 record_prefixes: str = "",
                 python: Optional[str] = None,
                 declare_text_budget_failures: bool = False) -> Dict[str, Any]:
    acebench_dir = Path(acebench_dir) if acebench_dir else ACEBENCH_DIR
    python = python or sys.executable
    tests = expand_categories(category, load_category_map(acebench_dir))
    effective_tests = tests
    work = prepare_workdir(out_dir, acebench_dir, language=language, tests=tests,
                           task_ids=task_ids, max_tasks=max_tasks)
    harness = acebench_dir
    if _task_id_list(task_ids) or max_tasks is not None:
        selection = json.loads((work / "selected_tasks.json").read_text(encoding="utf-8"))
        selected_tests = [str(source["test"]) for source in selection["sources"]]
        harness = prepare_subset_harness(work, acebench_dir, category, selected_tests)
        effective_tests = selected_tests
    if os.environ.get(TOOL_CONTEXT_ENV) == "1" or record_prefixes:
        harness = prepare_tool_span_harness(work, harness)
    env = harness_env(base_url, user_base_url, model,
                      record_source=bool(record_prefixes))
    telemetry_path = Path(out_dir).resolve() / "measurement" / "harness_events.jsonl"
    env["C2KV_ACEBENCH_TELEMETRY"] = str(telemetry_path)
    if declare_text_budget_failures:
        env[acebench_task_failures.ENV] = str(
            Path(out_dir).resolve() / acebench_task_failures.RECEIPTS)
    command = generate_command(python, harness, model, category, language, num_threads,
                               max_dialog_turns, user_model or model, temperature, top_p,
                               max_tokens)
    if all(test.startswith("agent_") for test in effective_tests):
        command = [python, str(Path(__file__).resolve().parents[1] / "acebench_cli.py"),
                   str(harness), *command[2:]]
    run_owned(
        command,
        cwd=work, env=env, check=True)
    return score_generated_run(
        Path(out_dir), work, harness, python=python, model=model, category=category,
        language=language, tests=effective_tests, env=env, user_model=user_model or model)


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
    parser.add_argument("--task-ids", default="", help="comma-separated official ids")
    parser.add_argument("--max-tasks", type=int, default=None)
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
                           temperature=args.temperature, task_ids=args.task_ids,
                           max_tasks=args.max_tasks, python=args.python)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
