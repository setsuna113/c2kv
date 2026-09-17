#!/usr/bin/env python3
"""Expose a frozen next-compression checkpoint through the shared SGLang engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from next_compression.live import LiveNextCompressionService, make_http_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--training-manifest", action="append", required=True)
    parser.add_argument("--ratio", type=int, choices=(8, 12), required=True)
    parser.add_argument("--backend", choices=("sglang", "native"), default="sglang",
                        help="sglang is the evaluation backend; native is the local reference")
    parser.add_argument("--sglang-url", help="bare shared engine URL, e.g. http://127.0.0.1:34010")
    parser.add_argument("--device", default="cuda", choices=("cuda", "npu", "cpu"))
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-request-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=("compressed", "full"), default="compressed")
    parser.add_argument("--model", default="c2kv-next")
    parser.add_argument("--model-alias", action="append", default=[])
    args = parser.parse_args(argv)
    if args.backend == "sglang" and not args.sglang_url:
        parser.error("--sglang-url is required for the default SGLang backend")
    if args.backend == "native" and args.sglang_url:
        parser.error("--sglang-url cannot be combined with --backend native")
    if args.backend == "native" and args.device == "npu":
        parser.error("the native reference supports cpu/cuda; use SGLang for npu")
    if len(args.checkpoint) != 1 or len(args.training_manifest) != 1:
        parser.error(
            "exactly one --checkpoint and one --training-manifest are required; "
            "history/tool checkpoint composition is unsupported"
        )
    args.checkpoint = Path(args.checkpoint[0]).resolve()
    args.training_manifest = Path(args.training_manifest[0]).resolve()
    for field in ("max_new_tokens", "max_requests", "max_request_bytes"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    service = LiveNextCompressionService.from_checkpoint(
        args.checkpoint,
        args.training_manifest,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        sglang_url=args.sglang_url,
        journal_path=output_dir / "sglang_http.jsonl",
        ratio=args.ratio,
        mode=args.mode,
        model=args.model,
        model_aliases=args.model_alias,
        max_new_tokens=args.max_new_tokens,
        max_requests=args.max_requests,
        max_request_bytes=args.max_request_bytes,
        ledger_path=output_dir / "requests.jsonl",
    )
    health = service.health()
    (output_dir / "server.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    server = make_http_server(service, args.host, args.port)
    print(
        json.dumps(
            {
                "status": "serving",
                "url": f"http://{args.host}:{args.port}/v1",
                "health": health,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        service.generator.close_session()


if __name__ == "__main__":
    raise SystemExit(main())
