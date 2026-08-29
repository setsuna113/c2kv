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

MODEL_NAME = "c2kv-hf"


def install_handler(base_url: str) -> None:
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
                "model": "c2kv-agent",
                "temperature": self.temperature,
                "store": False,
                "max_completion_tokens": 2048,
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
        model_name="c2kv-agent",
        display_name="c2kv-hf",
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def _score_roots() -> list:
    """Directories bfcl_eval may have written result/ and score/ into."""
    import os
    roots = []
    for var in ("BFCL_PROJECT_ROOT", "BFCL_RESULT_ROOT"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    try:
        import bfcl_eval
        roots.append(Path(bfcl_eval.__file__).resolve().parent)
    except Exception:
        pass
    roots.append(Path.cwd())
    return roots


def collect(categories: str) -> list:
    """Per-entry rows from bfcl_eval's own score output.

    Layout written by `bfcl evaluate` (v3/v4): JSONL score files at
    `score/<model>/BFCL_v*_<category>_score.json` whose FIRST line is the
    summary ({"accuracy", "correct_count", "total_count"}) and whose remaining
    lines are the failing entries, each carrying an "id".  The full entry set
    comes from the matching `result/<model>/..._result.json`.

    Raises rather than returning [] when nothing is found: an empty BFCL column
    that silently reads as "0 rows" is exactly how this adapter used to report
    no score at all.
    """
    model_dir = MODEL_NAME.replace("/", "_")
    wanted = [c.strip() for c in categories.split(",") if c.strip()]
    score_files, result_files = [], []
    for root in _score_roots():
        for cat in wanted:
            score_files += sorted((root / "score" / model_dir).glob(f"*{cat}*_score.json"))
            result_files += sorted((root / "result" / model_dir).glob(f"*{cat}*_result.json"))
        if score_files:
            break
    if not score_files:
        searched = ", ".join(str(r / "score" / model_dir) for r in _score_roots())
        raise SystemExit(
            f"FATAL: no BFCL score file for categories {categories!r} under: {searched}. "
            "Run `bfcl evaluate` first, or point BFCL_PROJECT_ROOT at its output root."
        )

    def _jsonl(path):
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    all_ids, failed = [], set()
    for path in result_files:
        all_ids += [str(r.get("id")) for r in _jsonl(path) if r.get("id") is not None]
    summaries = []
    for path in score_files:
        records = _jsonl(path)
        if not records:
            continue
        summaries.append(records[0])
        failed |= {str(r.get("id")) for r in records[1:] if r.get("id") is not None}
    if not all_ids:
        # No result file: fall back to the summary counts so the arm still
        # reports its official accuracy, with synthetic ids.
        total = sum(int(s.get("total_count") or 0) for s in summaries)
        correct = sum(int(s.get("correct_count") or 0) for s in summaries)
        if not total:
            raise SystemExit(
                f"FATAL: BFCL score files {[str(p) for p in score_files]} carry no "
                "total_count and no result file was found; refusing to report an "
                "empty accuracy."
            )
        return ([{"task_id": f"correct_{i}", "semantic_score": 1.0, "protocol_legal": None}
                 for i in range(correct)]
                + [{"task_id": f"incorrect_{i}", "semantic_score": 0.0, "protocol_legal": None}
                   for i in range(total - correct)])
    return [
        {"task_id": tid, "semantic_score": 0.0 if tid in failed else 1.0,
         "protocol_legal": None}
        for tid in all_ids
    ]


def run(base_url: str, categories: str = "multi_turn_base",
        mode: str = "both", run_ids: str = "") -> Dict[str, Any]:
    """Programmatic entry for benchmarks/run.py: register the handler and
    drive the official generate/evaluate CLI in-process."""
    install_handler(base_url)
    gen = ["generate", "--model", MODEL_NAME, "--test-category", categories]
    ev = ["evaluate", "--model", MODEL_NAME, "--test-category", categories]
    if run_ids:
        gen += ["--run-ids", run_ids]
        ev += ["--run-ids", run_ids]
    if mode in ("generate", "both"):
        run_cli(gen)
    if mode in ("evaluate", "both"):
        run_cli(ev)
    rows = collect(categories)
    if rows:
        import sys as _s; from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from metrics import aggregate  # noqa: E402
        summary = aggregate(rows, cluster_key="task_id")
    else:
        summary = {"n": 0}
    summary.update({"benchmark": "bfcl", "categories": categories, "mode": mode})
    summary["rows"] = rows
    return summary


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
    args = parser.parse_args(argv)
    install_handler(args.base_url)
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
