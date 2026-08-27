"""ToolSandbox adapter.

ToolSandbox runs scenarios through role clients; the OpenAI API agent
(``tool_sandbox.roles.openai_api_agent.OpenAIAPIAgent``) talks to any
OpenAI-compatible endpoint.  We point it at the arm proxy and keep the user
simulator / judge on a separate full-mode endpoint.

Usage on the server (bench venv):

    python benchmarks/adapters/toolsandbox_adapter.py \
        --base-url http://127.0.0.1:34100 --user-base-url http://127.0.0.1:34000 \
        --out results/bench/toolsandbox_c2kv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TS_DIR = Path.home() / "benchmarks" / "ToolSandbox"


def run(
    base_url: str,
    user_base_url: str,
    out_dir: Path,
    model_name: str = "c2kv-agent",
    max_scenarios: int = 0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_BASE_URL"] = base_url.rstrip("/") + "/v1"
    os.environ["C2KV_USER_BASE_URL"] = user_base_url.rstrip("/") + "/v1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.chdir(TS_DIR)

    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser
    from tool_sandbox.common.scenario import Scenario

    agent = OpenAIAPIAgent(
        model_name=model_name,
        base_url=base_url.rstrip("/") + "/v1",
    )
    user = OpenAIAPIUser(
        model_name=model_name,
        base_url=user_base_url.rstrip("/") + "/v1",
    )

    rows: List[Dict[str, Any]] = []
    scenarios = Scenario.discover() if hasattr(Scenario, "discover") else []
    for scenario in scenarios:
        if max_scenarios and len(rows) >= max_scenarios:
            break
        try:
            scenario.run(agent=agent, user=user)
        except Exception as error:  # noqa: BLE001 - record and continue
            rows.append({"task_id": scenario.name, "semantic_score": None,
                         "error": str(error)})
            continue
        metrics = scenario.calculate_metrics() if hasattr(scenario, "calculate_metrics") else {}
        rows.append(
            {
                "task_id": scenario.name,
                "semantic_score": metrics.get("main_acc"),
                "protocol_legal": None,
            }
        )

    from metrics import aggregate

    (out_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    summary = aggregate(rows, cluster_key="task_id")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="c2kv-agent")
    parser.add_argument("--max-scenarios", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.base_url, args.user_base_url, args.out, args.model_name, args.max_scenarios)
    print(json.dumps(summary, indent=2))
