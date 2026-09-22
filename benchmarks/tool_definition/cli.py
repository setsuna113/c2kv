"""Prepare and evaluate the recorded-decision §4.4 tool-definition panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import prepare

SELECTOR_POLICIES = (
    "last_user_topk_v1", "last_user_adaptive_v1", "latest_event_topk_v1",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Pack paired layouts from recorded decisions")
    prep.add_argument("--input", type=Path, required=True,
                      help="JSONL rows: decision_id, messages, tools, gold_tool_calls")
    prep.add_argument("--checkpoint", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--k", type=int, default=3)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--ratios", default="8,12")
    prep.add_argument("--interface-policy", choices=("none", "schema"), default="none",
                      help="opt-in raw executable interfaces with prose-only compressed definitions")
    prep.add_argument("--selector-policy", choices=SELECTOR_POLICIES,
                      default="last_user_topk_v1",
                      help="freeze the hybrid native-schema selector")
    ev = commands.add_parser("evaluate", help="Evaluate next actions through one SGLang server")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--upstream", required=True, help="SGLang HTTP base URL")
    ev.add_argument("--model", help="SGLang model id; inferred when the server exposes one")
    ev.add_argument("--max-new-tokens", type=int, required=True)
    ev.add_argument("--methods", default="c2kv,streamingllm,h2o,snapkv,pyramidkv")
    ev.add_argument("--layouts", default="full,uniform,hybrid,random,retrieval")
    ev.add_argument("--limit", type=int,
                    help="First N sorted (decision_id, ratio) groups; a decision with two ratios occupies two groups")
    ev.add_argument("--resume", action="store_true",
                    help="Continue an output directory with the identical frozen run contract")
    ev.add_argument("--full-results", type=Path,
                    help="reuse validated c2kv/full anchors from a completed output directory")
    ev.add_argument("--interface-policy", choices=("none", "schema"), default="none",
                    help="must match the policy frozen by prepare")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.input, args.checkpoint, args.out,
                         k=args.k, seed=args.seed,
                         ratios=tuple(int(item) for item in args.ratios.split(",")),
                         interface_policy=args.interface_policy,
                         selector_policy=args.selector_policy)
    else:
        from .evaluate import evaluate
        result = evaluate(args.manifest, args.checkpoint, args.out,
                          upstream=args.upstream, model=args.model,
                          max_new_tokens=args.max_new_tokens,
                          methods=tuple(filter(None, args.methods.split(","))),
                          layouts=tuple(filter(None, args.layouts.split(","))),
                          limit=args.limit, resume=args.resume,
                          interface_policy=args.interface_policy,
                          full_results=args.full_results)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
