"""Prepare and evaluate the recorded-decision §4.4 tool-definition panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import prepare


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
    ev = commands.add_parser("evaluate", help="Evaluate next actions through one SGLang server")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--upstream", required=True, help="SGLang HTTP base URL")
    ev.add_argument("--model", help="SGLang model id; inferred when the server exposes one")
    ev.add_argument("--max-new-tokens", type=int, required=True)
    ev.add_argument("--methods", default="c2kv,streamingllm,h2o,snapkv,pyramidkv")
    ev.add_argument("--layouts", default="full,uniform,hybrid,random,retrieval")
    ev.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.input, args.checkpoint, args.out,
                         k=args.k, seed=args.seed,
                         ratios=tuple(int(item) for item in args.ratios.split(",")))
    else:
        from .evaluate import evaluate
        result = evaluate(args.manifest, args.checkpoint, args.out,
                          upstream=args.upstream, model=args.model,
                          max_new_tokens=args.max_new_tokens,
                          methods=tuple(filter(None, args.methods.split(","))),
                          layouts=tuple(filter(None, args.layouts.split(","))),
                          limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
