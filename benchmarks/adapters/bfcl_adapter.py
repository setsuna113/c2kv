"""BFCL adapter (server-side verified version).

The working runner lives at ~/bench_results/bfcl_arm.py on the NPU server
(registers an OpenAI handler under model name ``c2kv-hf`` pointed at the
hf_server or the arm proxy, then drives the official bfcl CLI in-process).
This module mirrors it so the recipe is version-controlled; it must run in
the ``bench`` venv with cwd inside the gorilla checkout.

Usage:
    python benchmarks/adapters/bfcl_adapter.py \
        --base-url http://127.0.0.1:34000/v1 --categories multi_turn_base

ARGV CHANGE (2026-09-05), this standalone CLI only: it used to build its own
argv and append ``--run-ids`` to BOTH generate and evaluate; it now shares
``run_bfcl`` with ``run(ctx)``, so a subset run evaluates with
``--partial-eval`` (this vintage's evaluate has no ``--run-ids`` and scored
the whole category instead) and the terminal-state gate can exit non-zero.
Both entrypoints now keep the embedded Typer CLI alive through generation,
evaluation and terminal checks; run.py also isolates BFCL's result, score
and subset-id files under its output directory. A subset number from the OLD
standalone recipe is a full-category score — do not compare it with a new
one (README "BFCL standalone-CLI argv change").

Env fixes the bench venv needed: anthropic>=new, openai>=1.66, soundfile,
tree-sitter==0.21.3 + tree-sitter-java.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.base import RunContext, v1  # noqa: E402

NAME = "bfcl"
MODEL_NAME = "c2kv-hf"  # BFCL handler key / result-dir name (stable layout)
SERVED_MODEL = "c2kv-agent"  # default served model name at the endpoint


def add_arguments(parser) -> None:
    """BFCL-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--run-ids", default="",
                        help="bfcl: comma-separated official case ids for a subset run")


def default_bfcl_dir() -> str:
    """Checkout containing the installed BFCL package data."""
    return os.environ.get("BENCH_BFCL_DIR") or str(
        Path.home() / "benchmarks" / "gorilla"
        / "berkeley-function-call-leaderboard")


def handler_key(arm: str) -> str:
    """BFCL model key = result-dir name, one per arm so runs never overwrite
    each other.  eval_runner.py:782 un-escapes the result dir with
    ``replace("_", "/")`` — underscores in arm names would corrupt the path,
    so the key uses dashes."""
    return f"c2kv-{(arm or 'full').replace('_', '-')}"


def generate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None) -> List[str]:
    """``bfcl generate`` argv (PINNED; driven in-process by run_cli)."""
    argv = ["generate", "--model", handler_name, "--test-category", categories]
    if run_ids:
        argv.append("--run-ids")
    return argv


def evaluate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None) -> List[str]:
    """``bfcl evaluate`` argv.  A subset EVALUATE needs --partial-eval, NOT
    --run-ids (this vintage scores the full category otherwise)."""
    argv = ["evaluate", "--model", handler_name, "--test-category", categories]
    if run_ids:
        argv.append("--partial-eval")
    return argv


def install_handler(base_url: str, model: str = SERVED_MODEL,
                    handler_name: "str | None" = None) -> None:
    # NOTE: default resolved at CALL time — binding the default to
    # MODEL_NAME at def time made monkeypatched names register the
    # wrong key (val20 evaluate failure)
    handler_name = handler_name or MODEL_NAME
    import httpx
    from openai import OpenAI
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    class C2KVHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            return {
                "api_key": "EMPTY",
                "base_url": base_url,
                "timeout": httpx.Timeout(timeout=600.0, connect=8.0),
            }

        def _query_FC(self, inference_data: dict):
            kwargs = {
                "messages": inference_data["message"],
                "model": model,
                "temperature": self.temperature,
                "store": False,
                "max_completion_tokens": 4096,
            }
            if inference_data.get("tools"):
                kwargs["tools"] = inference_data["tools"]
            inference_data["inference_input_log"] = {
                "message": repr(inference_data["message"]),
                "tools": inference_data["tools"],
            }
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(**kwargs)
            return response, time.perf_counter() - t0

    MODEL_CONFIG_MAPPING[handler_name] = ModelConfig(
        model_name=model,
        display_name="c2kv-hf",
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def official_category_counts(category: str) -> Dict[str, int]:
    """Resolve a BFCL category/collection with the pinned official loader.

    A collection such as ``multi_turn`` is not a data filename. Memory and
    web-search concrete categories also share source files and are rewritten
    by the official loader, so counting ``BFCL_v4_<argument>.json`` is not a
    general denominator rule.
    """
    try:
        from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument
    except ImportError as error:
        raise RuntimeError(
            "BFCL expected-count resolution requires the pinned bfcl_eval package"
        ) from error

    concrete = parse_test_category_argument([category])
    if not concrete:
        raise ValueError(f"BFCL category resolves to no concrete categories: {category}")
    if "format_sensitivity" in concrete:
        raise ValueError(
            "BFCL format_sensitivity is non-scoring and cannot produce a scored summary"
        )

    counts: Dict[str, int] = {}
    for name in concrete:
        entries = load_dataset_entry(
            name, include_prereq=False, include_language_specific_hint=False)
        ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
        if not ids or any(item is None for item in ids):
            raise RuntimeError(f"BFCL category has no complete official id set: {name}")
        if len(ids) != len(set(map(str, ids))):
            raise RuntimeError(f"BFCL category has duplicate official ids: {name}")
        counts[name] = len(ids)
    return counts


def expected_count(category: str) -> int:
    """Number of scored tasks after official collection expansion."""
    return sum(official_category_counts(category).values())


def _score_header(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            header = next((json.loads(line) for line in handle if line.strip()), None)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid BFCL score file {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"BFCL score file has no JSON header: {path}")

    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if (not isinstance(total, int) or isinstance(total, bool) or total <= 0
            or not isinstance(correct, int) or isinstance(correct, bool)
            or correct < 0 or correct > total
            or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool)
            or not math.isfinite(float(accuracy))):
        raise RuntimeError(f"invalid BFCL score header in {path}: {header}")
    derived = correct / total
    if not math.isclose(float(accuracy), derived, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"BFCL score header accuracy mismatch in {path}: "
            f"accuracy={accuracy} correct_count={correct} total_count={total}")
    return header


def collect_score_summary(project_root: "Path | str", handler_name: str,
                          selected_counts: Mapping[str, int]) -> Dict[str, Any]:
    """Read and validate official per-category score headers without inference."""
    score_root = Path(project_root) / "score" / handler_name
    if not selected_counts:
        raise ValueError("BFCL selected category counts are empty")

    headers = []
    total = 0
    correct = 0
    for category, expected in selected_counts.items():
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"invalid selected count for BFCL/{category}: {expected!r}")
        filename = f"BFCL_v4_{category}_score.json"
        hits = sorted(score_root.rglob(filename)) if score_root.is_dir() else []
        if len(hits) != 1:
            raise RuntimeError(
                f"expected one BFCL score file for {category} under {score_root}, "
                f"found {len(hits)}")
        header = _score_header(hits[0])
        if header["total_count"] != expected:
            raise RuntimeError(
                f"BFCL scorer denominator mismatch for {category}: "
                f"total_count={header['total_count']} selected_expected={expected}")
        total += header["total_count"]
        correct += header["correct_count"]
        headers.append({
            "category": category,
            "path": str(hits[0]),
            "accuracy": float(header["accuracy"]),
            "correct_count": header["correct_count"],
            "total_count": header["total_count"],
        })

    selected_expected = sum(selected_counts.values())
    if total != selected_expected:
        raise RuntimeError(
            f"BFCL scorer coverage mismatch: n_scored={total} n={selected_expected}")
    return {
        "n": selected_expected,
        "n_total": selected_expected,
        "n_scored": total,
        "correct_count": correct,
        "semantic_score": correct / total,
        "official_score_headers": headers,
        "scored": True,
    }


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: run the official CLI from inside the gorilla checkout.

    BFCL reads ``BFCL_PROJECT_ROOT`` when its modules are first imported.
    Point it at this run's output directory before ``run_bfcl`` imports BFCL,
    while keeping cwd in the checkout so the installed package data resolves.
    The handler expects an OpenAI base_url WITH ``/v1``.

    No cost join: see ``COST_JOIN`` below.
    """
    project_root = ctx.out_dir.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.chdir(ctx.opt("bfcl_dir") or default_bfcl_dir())
    try:
        summary = run_bfcl(
            v1(ctx.base_url),
            categories=ctx.opt("categories", "multi_turn_base"),
            mode=ctx.opt("mode", "both"),
            run_ids=ctx.opt("run_ids"),
            model=ctx.model,
            handler_name=handler_key(ctx.arm),
            project_root=project_root,
        )
    finally:
        os.chdir(prev_cwd)
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root
    summary["cost_join"] = COST_JOIN
    return summary


# Why BFCL gets no per-task cost columns.  ``proxy.conversation_id`` shifts
# once per entry: request 1 carries only question[0] (the data file's turn-0
# user message), every later request also carries the FIRST ASSISTANT
# MESSAGE (proxy.py:434-447).  That assistant message is the raw OpenAI
# message object appended by _add_assistant_message_FC; the result file
# stores only the decoded ``model_responses`` (base_handler.py:243-253), and
# the verbatim wire payload is written only under ``--include-input-log``
# (base_handler.py:219-225) — a flag the pinned generate argv does not pass.
# Keying on the first id alone would attribute one request per entry, which
# is a wrong cost column rather than a missing one.
COST_JOIN = ("not joinable: the steady-state conversation id needs the first "
             "assistant message verbatim, which the BFCL result file does not "
             "store (see adapters/bfcl_adapter.py)")


def run_bfcl(base_url: str, categories: str = "multi_turn_base",
             mode: str = "both", run_ids: "list[str] | str | None" = None,
             model: str = SERVED_MODEL,
             handler_name: str = MODEL_NAME,
             project_root: "Path | str | None" = None) -> Dict[str, Any]:
    """Register the handler and drive the official generate/evaluate CLI
    in-process.

    ``handler_name`` is the BFCL model key (= result-dir name); ``run``
    passes ``handler_key(arm)`` so arms never overwrite each other's results.

    ``run_ids`` subsets the category: this BFCL vintage implements subsetting
    through <project-root>/test_case_ids_to_generate.json ({"<category>":
    [ids]}) + the boolean --run-ids flag, NOT a CLI value, so the ids are
    written to that file and the flag is passed (comma string or list).

    ``project_root`` is the isolated output root used by BFCL's official
    RESULT_PATH, SCORE_PATH and TEST_IDS_TO_GENERATE_PATH.  It must be set
    before the first official import.

    Terminal-state check (acceptance 1): every selected entry must have a
    result row and, after evaluation, an official score row. Generation-only
    summaries deliberately contain no ``n_scored`` or ``semantic_score``."""
    if mode not in ("generate", "evaluate", "both"):
        raise ValueError(f"invalid BFCL mode: {mode}")
    project_root = Path(
        project_root or os.environ.get("BFCL_PROJECT_ROOT") or Path.cwd()
    ).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    try:
        install_handler(base_url, model=model, handler_name=handler_name)
        category_counts = official_category_counts(categories)
        ids: Optional[List[str]] = None
        if run_ids:
            ids = ([i.strip() for i in run_ids.split(",") if i.strip()]
                   if isinstance(run_ids, str) else list(run_ids))
            if len(category_counts) != 1:
                raise ValueError(
                    "BFCL --run-ids requires one concrete category, not collection "
                    f"{categories!r} -> {sorted(category_counts)}")
            if not ids:
                raise ValueError("BFCL --run-ids resolved to an empty id list")
            # atomic write: concurrent runs racing on one file truncated ids
            id_file = project_root / "test_case_ids_to_generate.json"
            tmp = id_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({categories: ids}), encoding="utf-8")
            tmp.replace(id_file)
            selected_counts = {next(iter(category_counts)): len(ids)}
        else:
            selected_counts = category_counts
        expected = sum(selected_counts.values())
        if mode in ("generate", "both"):
            run_cli(generate_argv(handler_name, categories, ids))
        if mode in ("evaluate", "both"):
            run_cli(evaluate_argv(handler_name, categories, ids))
        import terminal_check  # noqa: E402  (sibling module, sys.path has parent)

        ids_str = ",".join(ids or [])
        for category, selected in selected_counts.items():
            code = terminal_check.check_bfcl(
                selected, ids_str, handler=handler_name,
                category=category, root=project_root)
            if code != 0:
                raise SystemExit(
                    f"FATAL: bfcl terminal-state check failed for {category} (rc={code})")

        summary: Dict[str, Any] = {
            "benchmark": "bfcl", "categories": categories, "mode": mode,
            "n_total": expected,
            "bfcl_project_root": str(project_root),
        }
        if mode == "generate":
            summary.update({"n_generated": expected, "scored": False})
        else:
            summary.update(collect_score_summary(
                project_root, handler_name, selected_counts))
            if mode == "both":
                summary["n_generated"] = expected
        return summary
    finally:
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root


def run_cli(argv):
    from bfcl_eval.__main__ import cli

    # Typer's standalone mode raises SystemExit(0) after generate, which
    # would skip evaluate, terminal checks and the unified run summary.
    cli(args=argv, prog_name="bfcl", standalone_mode=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:34000/v1")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--model", default=SERVED_MODEL,
                        help="served model name at the endpoint")
    parser.add_argument("--handler-name", default=MODEL_NAME,
                        help="BFCL model key / result-dir name")
    args = parser.parse_args(argv)
    # one code path: the CLI used to re-implement run() and had drifted
    # (evaluate got --run-ids instead of --partial-eval, no terminal gate)
    ids = [i.strip() for i in args.run_ids.split(",") if i.strip()] or None
    summary = run_bfcl(args.base_url, categories=args.categories, mode=args.mode,
                       run_ids=ids, model=args.model,
                       handler_name=args.handler_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
