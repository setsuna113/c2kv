"""BFCL adapter using the official ``bfcl_eval`` CLI in-process."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BFCL_DIR = Path.cwd()
BFCL_PROJECT_DIRNAME = "bfcl_project"


def _benchmark_dir(benchmark_dir: Optional[Path] = None) -> Path:
    configured = benchmark_dir or Path(os.environ.get("BFCL_DIR", str(DEFAULT_BFCL_DIR)))
    resolved = configured.expanduser().resolve()
    if not (resolved / "pyproject.toml").is_file() or not (resolved / "bfcl_eval").is_dir():
        raise SystemExit(
            f"FATAL: BFCL_DIR is not the berkeley-function-call-leaderboard package root: {resolved}"
        )
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def _openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def install_handler(base_url: str, registry_name: str, served_model_name: str) -> None:
    """Register an isolated OpenAI-compatible handler for one benchmark arm."""
    import httpx
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    endpoint = _openai_base_url(base_url)

    class C2KVHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            return {
                "api_key": "EMPTY",
                "base_url": endpoint,
                "timeout": httpx.Timeout(timeout=600.0, connect=8.0),
            }

        def _query_FC(self, inference_data: dict):
            kwargs = {
                "messages": inference_data["message"],
                "model": self.model_name,
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
            started = time.perf_counter()
            response = self.client.chat.completions.create(**kwargs)
            return response, time.perf_counter() - started

    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=served_model_name,
        display_name=registry_name,
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def run_cli(argv: Sequence[str]) -> None:
    from bfcl_eval.__main__ import cli

    old_argv = sys.argv
    try:
        sys.argv = ["bfcl", *argv]
        cli()
    finally:
        sys.argv = old_argv


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"FATAL: expected BFCL JSONL output does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"FATAL: invalid BFCL JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise SystemExit(f"FATAL: non-object row in BFCL JSONL at {path}:{line_number}")
        rows.append(row)
    return rows


def _numeric_sum(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        nested = [_numeric_sum(item) for item in value]
        return sum(item for item in nested if item is not None) if nested else 0.0
    return float(value)


def _nested_last_sum(value: Any) -> Optional[float]:
    """Sum final cumulative values from BFCL's nested per-turn token counts."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return float(value)
    if not value:
        return 0.0
    if any(isinstance(item, (list, tuple)) for item in value):
        return sum(_nested_last_sum(item[-1]) for item in value)
    return _numeric_sum(value)


def _collect_official(
    project_root: Path,
    registry_name: str,
    category: str,
) -> Dict[str, Any]:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.utils import get_directory_structure_by_category
    from metrics import aggregate

    group = get_directory_structure_by_category(category)
    model_result = project_root / "result" / registry_name / group / f"{VERSION_PREFIX}_{category}_result.json"
    model_score = project_root / "score" / registry_name / group / f"{VERSION_PREFIX}_{category}_score.json"
    result_rows = _read_jsonl(model_result)
    score_rows = _read_jsonl(model_score)
    if not score_rows:
        raise SystemExit(f"FATAL: BFCL score output is empty: {model_score}")

    header = score_rows[0]
    invalid_ids = {row.get("id") for row in score_rows[1:] if row.get("id")}
    unified_rows = []
    for row in result_rows:
        task_id = row.get("id")
        if not task_id:
            raise SystemExit(f"FATAL: BFCL result row has no id in {model_result}")
        latency = row.get("latency")
        input_tokens = row.get("input_token_count")
        output_tokens = row.get("output_token_count")
        unified_rows.append(
            {
                "task_id": str(task_id),
                "semantic_score": float(task_id not in invalid_ids),
                "protocol_legal": None,
                "wall_sec": _numeric_sum(latency),
                "input_tokens": _nested_last_sum(input_tokens),
                "output_tokens": _numeric_sum(output_tokens),
            }
        )

    summary = aggregate(unified_rows, cluster_key="task_id")
    official_accuracy = float(header["accuracy"])
    if summary.get("n") != header.get("total_count"):
        raise SystemExit(
            f"FATAL: BFCL row count {summary.get('n')} != official total {header.get('total_count')}"
        )
    if summary.get("semantic_score") is None or abs(summary["semantic_score"] - official_accuracy) > 1e-12:
        raise SystemExit(
            "FATAL: aggregated BFCL accuracy does not match official score: "
            f"{summary.get('semantic_score')} != {official_accuracy}"
        )
    summary.update(
        {
            "benchmark": "bfcl",
            "categories": category,
            "bfcl_registry_model": registry_name,
            "bfcl_official": {
                "accuracy": official_accuracy,
                "correct_count": header.get("correct_count"),
                "total_count": header.get("total_count"),
                "result_path": str(model_result),
                "score_path": str(model_score),
            },
            "input_tokens_total": sum(row.get("input_tokens") or 0 for row in unified_rows),
            "output_tokens_total": sum(row.get("output_tokens") or 0 for row in unified_rows),
        }
    )
    return summary


def run(
    base_url: str,
    categories: str = "multi_turn_base",
    mode: str = "both",
    run_ids: str = "",
    benchmark_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    arm: str = "full",
    served_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and/or evaluate one isolated BFCL arm cell."""
    if mode not in {"generate", "evaluate", "both"}:
        raise SystemExit(f"FATAL: unknown BFCL mode {mode!r}")
    if out_dir is None:
        raise SystemExit("FATAL: BFCL adapter requires out_dir for isolated result/score storage")

    _benchmark_dir(benchmark_dir)
    project_root = out_dir.resolve() / BFCL_PROJECT_DIRNAME
    project_root.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)

    safe_arm = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in arm)
    registry_name = f"c2kv-bfcl-{safe_arm}"
    api_model = served_model_name or os.environ.get("SERVED_MODEL_NAME", "c2kv-agent")
    install_handler(base_url, registry_name, api_model)

    if run_ids:
        ids = [item.strip() for item in run_ids.split(",") if item.strip()]
        if not ids:
            raise SystemExit("FATAL: --run-ids contained no IDs")
        (project_root / "test_case_ids_to_generate.json").write_text(
            json.dumps({categories: ids}, indent=2) + "\n", encoding="utf-8"
        )

    if mode in {"generate", "both"}:
        # BFCL's Typer generate entrypoint terminates the Python process on
        # completion. Run it in an isolated child so this adapter can still
        # evaluate and write its unified summary afterwards.
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "bfcl_generate_cli.py"),
                "--base-url", base_url,
                "--categories", categories,
                "--benchmark-dir", str(_benchmark_dir(benchmark_dir)),
                "--out-dir", str(project_root),
                "--arm", safe_arm,
                "--served-model-name", api_model,
                *(["--run-ids"] if run_ids else []),
            ],
            check=True,
        )
    if mode in {"evaluate", "both"}:
        # The Typer application is not safely re-entrant when generate and
        # evaluate are both invoked in one process (the second call is a no-op).
        # Call the official evaluation entrypoint directly; handler registration
        # and MODEL_CONFIG_MAPPING remain in this process.
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

        evaluation_main(
            [registry_name],
            [categories],
            project_root / "result",
            project_root / "score",
            bool(run_ids),
        )
    if mode == "generate":
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        from bfcl_eval.utils import get_directory_structure_by_category

        group = get_directory_structure_by_category(categories)
        result_path = project_root / "result" / registry_name / group / f"{VERSION_PREFIX}_{categories}_result.json"
        rows = _read_jsonl(result_path)
        return {
            "benchmark": "bfcl",
            "categories": categories,
            "mode": mode,
            "n_generated": len(rows),
            "bfcl_registry_model": registry_name,
            "bfcl_result_path": str(result_path),
        }

    summary = _collect_official(project_root, registry_name, categories)
    summary["mode"] = mode
    summary["run_ids"] = run_ids
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="full")
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "c2kv-agent"))
    args = parser.parse_args(argv)
    summary = run(
        args.base_url,
        categories=args.categories,
        mode=args.mode,
        run_ids=args.run_ids,
        benchmark_dir=args.benchmark_dir,
        out_dir=args.out,
        arm=args.arm,
        served_model_name=args.served_model_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
