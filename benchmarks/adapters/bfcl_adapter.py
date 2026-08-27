"""BFCL adapter.

BFCL maps model names to handler classes through
``bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING``.  Instead of
patching the gorilla checkout we inject, at runtime, one entry whose handler
is an OpenAI client pointed at the arm proxy.  Prompt-style categories
(prompt / Python-ish) are irrelevant here; the interesting categories are
the multi-turn and AST families where the c2kv arms can actually change
protocol legality.

Usage on the server (bench venv):

    python benchmarks/adapters/bfcl_adapter.py \
        --base-url http://127.0.0.1:34100 --arm c2kv \
        --categories multi_turn --out results/bench/bfcl_c2kv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BFCL_DIR = Path.home() / "benchmarks" / "gorilla" / "berkeley-function-call-leaderboard"

HANDLER_TEMPLATE = '''
import os
from openai import OpenAI
import httpx
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from overrides import override


class C2KVProxyHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("C2KV_PROXY_KEY", "EMPTY"),
            base_url=os.getenv("C2KV_PROXY_BASE", "http://127.0.0.1:34100/v1"),
            timeout=httpx.Timeout(timeout=600.0, connect=8.0),
        )

    @override
    def _query_FC(self, inference_data: dict):
        kwargs = {
            "messages": inference_data["message"],
            "model": os.getenv("C2KV_PROXY_MODEL", "c2kv-agent"),
            "temperature": self.temperature,
            "store": False,
        }
        if inference_data.get("tools"):
            kwargs["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "tools": inference_data["tools"],
        }
        return self.client.chat.completions.create(**kwargs)
'''


def install_handler(base_url: str, model_name: str = "c2kv-agent"):
    """Import-time registration of the proxy handler inside bfcl_eval."""
    namespace: Dict[str, Any] = {}
    exec(HANDLER_TEMPLATE, namespace)
    handler_cls = namespace["C2KVProxyHandler"]
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    MODEL_CONFIG_MAPPING["c2kv-proxy"] = ModelConfig(
        model_name=model_name,
        display_name="c2kv-proxy",
        url="",
        org="c2kv",
        license="",
        model_handler=handler_cls,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    os.environ["C2KV_PROXY_BASE"] = base_url.rstrip("/") + "/v1"
    os.environ["C2KV_PROXY_MODEL"] = model_name


def run(base_url: str, categories: List[str], out_dir: Path, model_name: str = "c2kv-agent") -> Dict[str, Any]:
    install_handler(base_url, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(BFCL_DIR)
    from bfcl_eval._llm_response_generation import main as generation_main

    argv = [
        "gen", "--model", "c2kv-proxy",
        "--test-category", ",".join(categories),
        "--result-dir", str(out_dir),
    ]
    sys.argv = [sys.argv[0], *argv]
    generation_main()

    from bfcl_eval.eval_checker.eval_runner_constant import find_test_name_list
    from bfcl_eval.constants.config import BFCL_RESULT_PATH  # noqa: F401 (path layout check)

    # Official evaluation pass
    sys.argv = [sys.argv[0], "eval", "--model", "c2kv-proxy",
                "--test-category", ",".join(categories),
                "--result-dir", str(out_dir)]
    from bfcl_eval.eval_checker import eval_runner  # noqa: F401

    return collect(out_dir)


def collect(out_dir: Path) -> Dict[str, Any]:
    """Read BFCL score JSONs into unified rows."""
    from metrics import aggregate

    rows: List[Dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*.json")):
        if "score" not in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for category, models in (data.items() if isinstance(data, dict) else []):
            entry = models.get("c2kv-proxy") if isinstance(models, dict) else None
            if entry is None:
                continue
            rows.append(
                {
                    "task_id": f"{path.parent.name}/{category}",
                    "semantic_score": entry.get("accuracy",
                                                entry.get("valid", entry.get("result"))),
                    "protocol_legal": None,  # from per-test legality below
                }
            )
    return aggregate(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--categories", default="multi_turn")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="c2kv-agent")
    args = parser.parse_args()
    summary = run(args.base_url, args.categories.split(","), args.out, args.model_name)
    print(json.dumps(summary, indent=2))
