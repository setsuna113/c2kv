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
``run.py --benchmark bfcl`` is unchanged.  A subset number from the OLD
standalone recipe is a full-category score — do not compare it with a new
one (README "BFCL standalone-CLI argv change").

Env fixes the bench venv needed: anthropic>=new, openai>=1.66, soundfile,
tree-sitter==0.21.3 + tree-sitter-java.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.base import RunContext, v1  # noqa: E402

NAME = "bfcl"
MODEL_NAME = "c2kv-hf"  # BFCL handler key / result-dir name (stable layout)
SERVED_MODEL = "c2kv-agent"  # default served model name at the endpoint


def add_arguments(parser) -> None:
    """BFCL-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--categories", default="multi_turn_base")


def default_bfcl_dir() -> str:
    """bfcl_eval resolves its data/result dirs from cwd; the adapter runs
    from the gorilla checkout ($BENCH_BFCL_DIR override)."""
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


def expected_count(category: str, root: "Path | None" = None,
                   fallback: int = 200) -> int:
    """Entries in the category's data file (bfcl_eval/data/BFCL_v4_<category>
    .json under the gorilla checkout = cwd).  The literal 200 was
    multi_turn_base only; memory has 155, web_search 100."""
    root = Path(root) if root else Path.cwd()
    data = root / "bfcl_eval" / "data" / f"BFCL_v4_{category}.json"
    if not data.exists():
        print(f"WARNING: {data} not found; expected count falls back to {fallback}",
              file=sys.stderr)
        return fallback
    return sum(1 for line in data.read_text(encoding="utf-8").splitlines()
               if line.strip())


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: run the official CLI from inside the gorilla checkout.

    bfcl_eval resolves data/ and result/ from the process cwd, so the chdir
    (and its restore) belongs here, not in run.py.  The handler expects an
    OpenAI base_url WITH ``/v1``.

    No cost join: see ``COST_JOIN`` below.
    """
    prev_cwd = os.getcwd()
    os.chdir(ctx.opt("bfcl_dir") or default_bfcl_dir())
    try:
        summary = run_bfcl(
            v1(ctx.base_url),
            categories=ctx.opt("categories", "multi_turn_base"),
            mode=ctx.opt("mode", "both"),
            run_ids=ctx.opt("run_ids"),
            model=ctx.model,
            handler_name=handler_key(ctx.arm),
        )
    finally:
        os.chdir(prev_cwd)
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
             handler_name: str = MODEL_NAME) -> Dict[str, Any]:
    """Register the handler and drive the official generate/evaluate CLI
    in-process.

    ``handler_name`` is the BFCL model key (= result-dir name); ``run``
    passes ``handler_key(arm)`` so arms never overwrite each other's results.

    ``run_ids`` subsets the category: this BFCL vintage implements subsetting
    through <gorilla-root>/test_case_ids_to_generate.json ({"<category>":
    [ids]}) + the boolean --run-ids flag, NOT a CLI value, so the ids are
    written to that file and the flag is passed (comma string or list).

    Terminal-state check (acceptance 1): every expected entry must have a
    result row — the run fails loudly instead of shrinking the denominator."""
    install_handler(base_url, model=model, handler_name=handler_name)
    expected = expected_count(categories)
    ids: Optional[List[str]] = None
    if run_ids:
        ids = ([i.strip() for i in run_ids.split(",") if i.strip()]
               if isinstance(run_ids, str) else list(run_ids))
        # atomic write: concurrent runs racing on one file truncated ids
        id_file = Path.cwd() / "test_case_ids_to_generate.json"
        tmp = id_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({categories: ids}), encoding="utf-8")
        tmp.replace(id_file)
        expected = len(ids)
    if mode in ("generate", "both"):
        run_cli(generate_argv(handler_name, categories, ids))
    if mode in ("evaluate", "both"):
        run_cli(evaluate_argv(handler_name, categories, ids))
    import terminal_check  # noqa: E402  (sibling module, sys.path has parent)

    ids_str = ",".join(run_ids) if isinstance(run_ids, list) else (run_ids or "")
    code = terminal_check.check_bfcl(expected, ids_str, handler=handler_name,
                                     category=categories)
    if code != 0:
        raise SystemExit(f"FATAL: bfcl terminal-state check failed (rc={code})")
    return {"benchmark": "bfcl", "categories": categories, "mode": mode,
            "n_total": expected, "n_scored": expected}


def run_cli(argv):
    from bfcl_eval.__main__ import cli

    sys.argv = ["bfcl"] + argv
    cli()


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
