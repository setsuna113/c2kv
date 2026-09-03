"""BFCL adapter (server-side verified version).

The working runner lives at ~/bench_results/bfcl_arm.py on the NPU server
(registers an OpenAI handler under model name ``c2kv-hf`` pointed at the
hf_server or the arm proxy, then drives the official bfcl CLI in-process).
This module mirrors it so the recipe is version-controlled; it must run in
the ``bench`` venv with cwd inside the gorilla checkout.

Usage:
    python benchmarks/adapters/bfcl_adapter.py \
        --base-url http://127.0.0.1:34000/v1 --categories multi_turn_base

Env fixes the bench venv needed: anthropic>=new, openai>=1.66, soundfile,
tree-sitter==0.21.3 + tree-sitter-java.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "c2kv-hf"  # BFCL handler key / result-dir name (stable layout)
SERVED_MODEL = "c2kv-agent"  # default served model name at the endpoint


def install_handler(base_url: str, model: str = SERVED_MODEL) -> None:
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

    MODEL_CONFIG_MAPPING[MODEL_NAME] = ModelConfig(
        model_name=model,
        display_name="c2kv-hf",
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def run(base_url: str, categories: str = "multi_turn_base",
        mode: str = "both", run_ids: str = "",
        model: str = SERVED_MODEL) -> Dict[str, Any]:
    """Programmatic entry for benchmarks/run.py: register the handler and
    drive the official generate/evaluate CLI in-process.

    Terminal-state check (acceptance 1): every expected entry must have a
    result row — the run fails loudly instead of shrinking the denominator."""
    install_handler(base_url, model=model)
    gen = ["generate", "--model", MODEL_NAME, "--test-category", categories]
    ev = ["evaluate", "--model", MODEL_NAME, "--test-category", categories]
    if run_ids:
        gen += ["--run-ids", run_ids]
        ev += ["--run-ids", run_ids]
    if mode in ("generate", "both"):
        run_cli(gen)
    if mode in ("evaluate", "both"):
        run_cli(ev)
    import terminal_check  # noqa: E402  (sibling module, sys.path has parent)

    expected = len([r for r in run_ids.split(",") if r.strip()]) or 200
    code = terminal_check.check_bfcl(expected, run_ids)
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
    args = parser.parse_args(argv)
    install_handler(args.base_url, model=args.model)
    gen = ["generate", "--model", MODEL_NAME, "--test-category", args.categories]
    ev = ["evaluate", "--model", MODEL_NAME, "--test-category", args.categories]
    if args.run_ids:
        gen += ["--run-ids", args.run_ids]
        ev += ["--run-ids", args.run_ids]
    if args.mode in ("generate", "both"):
        run_cli(gen)
    if args.mode in ("evaluate", "both"):
        run_cli(ev)


if __name__ == "__main__":
    main()
