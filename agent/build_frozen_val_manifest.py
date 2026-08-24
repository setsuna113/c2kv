"""Build the frozen validation manifest for the D1' four-condition readout.

Selects examples from the SAME AgentLLMTracesCompressHistorySource pipeline used
by the S4-style agent-history evals (see agent/eval_agent_history_c2kv.py
parse_args for the mirrored defaults: split=eval, split_manifest_name=
subset_disjoint, eval_ratio=0.1, split_seed=42, max_samples_per_session=4,
include_tools=True) and keeps only examples that carry a non-empty real D1'
condition window, i.e. the next turn after the compressed history is led by a
raw-role "user" message (never a tool result), so `condition_text` is non-empty.

Selection is deterministic: the first `--n` qualifying examples in pipeline
iteration order (mirroring _load_examples' first-max_examples selection in
agent/eval_agent_history_c2kv.py); `--seed` is the pipeline split_seed.

Output json: {"created_from", "filters": {...}, "n", "qids" (sorted),
"git_commit"}.

The condition window content does not depend on the token budget (the budget
only truncates token ids later, at feature-build time), so this script only
needs the source gate C2KV_CONDITION_WINDOW_TOKENS > 0 for condition_text to be
populated; it does not need a tokenizer.

Heavy third-party imports (datasets/transformers, pulled in by
train.train_data_multiturn) are guarded so the script exits cleanly with a
message on machines without them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT / "python"))

_HEAVY_IMPORT_ERROR = None
try:
    from train.train_data_multiturn import AgentLLMTracesCompressHistorySource
except ImportError as error:  # pragma: no cover - depends on the host env
    _HEAVY_IMPORT_ERROR = error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen D1' validation manifest (qid list) for the four-condition readout."
    )
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--out", default="configs/d1prime_frozen_val.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42, help="Pipeline split_seed (mirrors S4 defaults).")
    # S4-style filters, mirrored from agent/eval_agent_history_c2kv.py parse_args.
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_manifest_file", default=None)
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument(
        "--include_tools",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Mirror the S4 NPU eval default (INCLUDE_TOOLS=True).",
    )
    return parser.parse_args()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as error:  # noqa: BLE001 - manifest build should not die on git issues
        logger.warning("Could not resolve git commit: %s", error)
        return "unknown"


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    if _HEAVY_IMPORT_ERROR is not None:
        raise SystemExit(
            "build_frozen_val_manifest.py needs the training data dependencies "
            "(datasets/transformers and the repo `train` package); import failed with: "
            f"{type(_HEAVY_IMPORT_ERROR).__name__}: {_HEAVY_IMPORT_ERROR}"
        )
    # The source only populates `condition_text` when the D1' condition-window
    # gate is on. The gate value does not affect the window CONTENT (it only
    # truncates token ids at feature-build time), so any value > 0 is fine here;
    # keep a larger ambient value if one is already set.
    ambient_window = int(os.environ.get("C2KV_CONDITION_WINDOW_TOKENS", "0") or 0)
    condition_window_tokens = max(1, ambient_window)
    os.environ["C2KV_CONDITION_WINDOW_TOKENS"] = str(condition_window_tokens)

    source = AgentLLMTracesCompressHistorySource(
        args.dataset_path,
        split=args.split,
        eval_ratio=args.eval_ratio,
        split_seed=args.seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        include_tools=args.include_tools,
    )
    qids: List[str] = []
    seen_sessions = set()
    for example in source:
        # Non-empty real condition window: the next turn exists and is led by a
        # raw-role user message (tool-led turns stay unconditioned upstream).
        if not (example.condition_text or "").strip():
            continue
        qids.append(example.qid)
        seen_sessions.add(example.qid.rsplit(":", 1)[0] if ":" in example.qid else example.qid)
        if args.n and len(qids) >= args.n:
            break
    qids = sorted(qids)
    filters = {
        "split": args.split,
        "eval_ratio": args.eval_ratio,
        "split_seed": args.seed,
        "split_manifest_file": args.split_manifest_file,
        "split_manifest_name": args.split_manifest_name,
        "max_samples_per_session": args.max_samples_per_session,
        "include_tools": args.include_tools,
        "require_tool_call": False,
        "selection": "first_n_in_pipeline_order_with_non_empty_condition_text",
        "condition_window_tokens_gate": condition_window_tokens,
    }
    manifest = {
        "created_from": str(args.dataset_path),
        "filters": filters,
        "n": len(qids),
        "qids": qids,
        "git_commit": _git_commit(),
    }
    logger.info(
        "Selected %d examples across %d sessions (requested n=%d)",
        len(qids),
        len(seen_sessions),
        args.n,
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote manifest with n=%d qids to %s", manifest["n"], out_path)


if __name__ == "__main__":
    main()
