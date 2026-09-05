"""Minimal SGLang/C2KV smoke checks used before the benchmark matrix."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error


def _write(result: Dict[str, Any], output: Optional[Path]) -> None:
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def checkpoint_command(args: argparse.Namespace) -> None:
    from safetensors import safe_open

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    weights = args.checkpoint / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"FATAL: missing model.safetensors: {weights}")
    layers = int(config["num_hidden_layers"])
    expected = {"model.gist_embed_tokens.weight"}
    for layer in range(layers):
        for projection in ("q", "k", "v"):
            expected.add(f"model.layers.{layer}.self_attn.gist_{projection}_proj.weight")

    with safe_open(weights, framework="pt", device="cpu") as handle:
        present = set(handle.keys())
    missing = sorted(expected - present)
    result = {
        "gate": "checkpoint",
        "checkpoint": str(args.checkpoint.resolve()),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_hidden_layers": layers,
        "gist_residual_type": config.get("gist_residual_type"),
        "gist_type": config.get("gist_type"),
        "gist_param": config.get("gist_param"),
        "expected_gist_weight_count": len(expected),
        "missing_gist_weights": missing,
        "passed": (
            config.get("model_type") == "qwen3"
            and config.get("architectures") == ["Qwen3ForCausalLM"]
            and config.get("gist_residual_type") in {"none", "embed-mean", "mean"}
            and config.get("gist_type") == "dynamic-interleave"
            and config.get("gist_param") == "qkv"
            and not missing
        ),
    }
    _write(result, args.out)
    if not result["passed"]:
        raise SystemExit(2)


def service_command(args: argparse.Namespace) -> None:
    payload = {
        "text": "Airline policy smoke test. A passenger has one confirmed reservation and asks about seat upgrades.",
        "compression_ratio": 8,
        "role": "user",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = _post_json(args.base_url, "/v1/c2kv/extract", payload, 120)
    second = _post_json(args.base_url, "/v1/c2kv/extract", payload, 120)
    result = {
        "gate": "S1_extract",
        "first": first,
        "key_hash_stable": bool(first.get("key_hash") and first.get("key_hash") == second.get("key_hash")),
        "passed": bool(
            first.get("success")
            and second.get("success")
            and first.get("key_hash")
            and first.get("key_hash") == second.get("key_hash")
            and int(first.get("gist_len") or 0) > 0
            and int(first.get("original_seq_len") or 0) > 0
        ),
    }
    _write(result, args.out)
    if not result["passed"]:
        raise SystemExit(2)


def tools_command(args: argparse.Namespace) -> None:
    payload = {
        "model": args.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": "Call the get_weather tool for San Francisco with unit=celsius.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city", "unit"],
                    },
                },
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    response = _post_json(args.base_url, "/v1/chat/completions", payload, 300)
    choices = response.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    calls = message.get("tool_calls") or []
    result = {
        "gate": "S2_tools",
        "tool_call_count": len(calls),
        "first_tool_call": calls[0] if calls else None,
        "passed": bool(calls),
    }
    _write(result, args.out)
    if not result["passed"]:
        raise SystemExit(2)


def _wait_http(url: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"proxy exited with status {process.returncode}")
        try:
            urllib.request.urlopen(url, timeout=2).close()
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"proxy health timeout: {url}")


def _chat_payload(model: str) -> Dict[str, Any]:
    history = "Flight record: " + ("passenger segment and fare rule detail. " * 48)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise airline assistant."},
            {"role": "user", "content": history},
            {"role": "assistant", "content": "Recorded the flight details."},
            {"role": "user", "content": "Answer with exactly: SMOKE_OK"},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }


def _proxy_regime() -> Dict[str, Any]:
    """The segmentation regime the matrix runs with, read from proxy.py so a
    gate can never certify a regime the matrix does not use."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proxy  # noqa: E402

    return {
        "doc_packing": proxy.DOC_PACKING,
        "max_docs": proxy.MAX_DOCS,
        "max_doc_length": proxy.MAX_DOC_LENGTH,
    }


def _proxy_regime_flags(regime: Dict[str, Any]) -> List[str]:
    return [
        "--doc-packing", str(regime["doc_packing"]),
        "--max-docs", str(regime["max_docs"]),
        "--max-doc-length", str(regime["max_doc_length"]),
    ]


def proxy_command(args: argparse.Namespace) -> None:
    payload = _chat_payload(args.served_model_name)
    regime = _proxy_regime()
    direct = _post_json(args.base_url, "/v1/chat/completions", payload, 300)
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="c2kv-proxy-smoke-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        responses = {}
        for arm, port in (("full", args.full_port), ("c2kv", args.c2kv_port)):
            request_log = log_dir / f"proxy_{arm}_{port}.jsonl"
            output_log = log_dir / f"proxy_{arm}_{port}.out"
            with output_log.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("proxy.py")),
                        "--upstream", args.base_url,
                        "--arm", arm,
                        "--port", str(port),
                        "--request-log", str(request_log),
                        *_proxy_regime_flags(regime),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            processes.append(process)
            _wait_http(f"http://127.0.0.1:{port}/health", process)
            responses[arm] = _post_json(
                f"http://127.0.0.1:{port}", "/v1/chat/completions", payload, 600
            )

        def comparable(response: Dict[str, Any]) -> Dict[str, Any]:
            choice = (response.get("choices") or [{}])[0]
            return {
                "model": response.get("model"),
                "message": choice.get("message"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
            }

        full_normalized = comparable(responses["full"])
        direct_normalized = comparable(direct)
        metadata = (responses["c2kv"].get("c2kv_proxy") or {})
        original = int(metadata.get("original_tokens") or 0)
        gist = int(metadata.get("gist_tokens") or 0)
        ratio = original / gist if gist else None
        result = {
            "gate": "S3_proxy",
            "proxy_regime": regime,
            "full_matches_direct": full_normalized == direct_normalized,
            "c2kv_proxy": metadata,
            "effective_ratio": ratio,
            "request_logs": [str(log_dir / f"proxy_{arm}_{port}.jsonl")
                             for arm, port in (("full", args.full_port), ("c2kv", args.c2kv_port))],
            "passed": bool(
                full_normalized == direct_normalized
                and responses["c2kv"].get("choices")
                and ratio is not None
                and 6.0 <= ratio <= 10.0
            ),
        }
        _write(result, args.out)
        if not result["passed"]:
            raise SystemExit(2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


S6_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search flights between two cities on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reservation",
            "description": "Fetch a reservation by confirmation number.",
            "parameters": {
                "type": "object",
                "properties": {"confirmation": {"type": "string"}},
                "required": ["confirmation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_seat",
            "description": "Change the seat on a booked segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation": {"type": "string"},
                    "seat": {"type": "string"},
                },
                "required": ["confirmation", "seat"],
            },
        },
    },
]
S6_SYSTEM = "You are a precise airline assistant. Use the tools when needed."


def _s6_payload(model: str) -> Dict[str, Any]:
    """System + tools + a four-message history whose earlier turns compress."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": S6_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Find me a flight from SFO to LHR on 2026-10-02. "
                    + "Context: " + ("passenger segment and fare rule detail. " * 48)
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": (
                                '{"origin": "SFO", "destination": "LHR",'
                                ' "date": "2026-10-02"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": '{"flights": [{"number": "BA286", "fare": 812}]}',
            },
            {"role": "user", "content": "What is the fare of that flight?"},
        ],
        "tools": S6_TOOLS,
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _prologue_token_len(checkpoint: Path) -> int:
    """Token length of the rendered system+tools prologue, computed locally.

    This is the frame the server must place the first gist at: training's
    layout is [system+tools] -> [gists] -> [current turn]."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    messages = [{"role": "system", "content": S6_SYSTEM}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tools=S6_TOOLS, tokenize=True,
            add_generation_prompt=False, enable_thinking=False,
        )
    except TypeError:  # template without an enable_thinking knob
        ids = tokenizer.apply_chat_template(
            messages, tools=S6_TOOLS, tokenize=True, add_generation_prompt=False,
        )
    return len(ids)


def tools_proxy_command(args: argparse.Namespace) -> None:
    """S6: tools AND compressed history through the proxy in one request.

    S2 exercises tools without gists and S3 exercises gists without tools, so
    neither can see a gist inserted at a tool-free prefix length."""
    regime = _proxy_regime()
    payload = _s6_payload(args.served_model_name)
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="c2kv-s6-smoke-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / f"proxy_c2kv_{args.port}.jsonl"
    output_log = log_dir / f"proxy_c2kv_{args.port}.out"
    process = None
    try:
        with output_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("proxy.py")),
                    "--upstream", args.base_url,
                    "--arm", "c2kv",
                    "--port", str(args.port),
                    "--request-log", str(request_log),
                    *_proxy_regime_flags(regime),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        _wait_http(f"http://127.0.0.1:{args.port}/health", process)
        response = _post_json(
            f"http://127.0.0.1:{args.port}", "/v1/chat/completions", payload, 600
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    choices = response.get("choices") or []
    runtime = (response.get("metadata") or {}).get("sglang_runtime") or {}
    layout = runtime.get("c2kv_layout") or []
    gists = [entry for entry in layout if entry.get("kind") == "gist"]
    result: Dict[str, Any] = {
        "gate": "S6_tools_through_proxy",
        "proxy_regime": regime,
        "n_tools": len(S6_TOOLS),
        "n_messages": len(payload["messages"]),
        "choices_non_empty": bool(choices),
        "c2kv_proxy": response.get("c2kv_proxy") or {},
        "c2kv_query_proj": runtime.get("c2kv_query_proj"),
        "c2kv_layout_gists": gists,
        "request_log": str(request_log),
    }
    if not layout:
        # The layout block only exists from the serve-align pin onwards.
        result["layout_check"] = "layout metadata unavailable"
        result["passed"] = bool(choices)
    else:
        expected = _prologue_token_len(args.checkpoint)
        observed = int(gists[0]["position_cursor"]) if gists else None
        result["expected_prologue_tokens"] = expected
        result["first_gist_position_cursor"] = observed
        result["layout_check"] = "first gist position_cursor vs local system+tools prologue"
        result["passed"] = bool(choices) and observed == expected
    _write(result, args.out)
    if not result["passed"]:
        raise SystemExit(2)


def flex_command(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    # Qwen's byte tokenizer re-encodes this decimal string exactly one token
    # per character. Repeating prose would merge repeated byte tokens after
    # decode/encode and silently shrink the intended document length.
    token_stable_text = "1234567890"
    records: List[Dict[str, Any]] = []
    for target in (512, 2048, 4096, 8192):
        text = (token_stable_text * ((target + len(token_stable_text) - 1) // len(token_stable_text)))[:target]
        actual_source_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if actual_source_tokens != target:
            raise SystemExit(
                f"FATAL: flex fixture re-tokenization changed length: {actual_source_tokens} != {target}"
            )
        started = time.perf_counter()
        response = _post_json(
            args.base_url,
            "/v1/c2kv/extract",
            {
                "text": text,
                "compression_ratio": 8,
                "role": "user",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            900,
        )
        records.append(
            {
                "target_source_tokens": target,
                "actual_source_tokens": actual_source_tokens,
                "wall_sec": round(time.perf_counter() - started, 4),
                "response": response,
            }
        )

    suspicious = []
    if args.server_log and Path(args.server_log).is_file():
        pattern = re.compile(r"(?:\bnan\b|\bNaN\b|recompile_limit|torch\.compile.*limit)", re.I)
        suspicious = [line for line in Path(args.server_log).read_text(errors="replace").splitlines()
                      if pattern.search(line)]
    passed = all(
        item["response"].get("success")
        and int(item["response"].get("gist_len") or 0) > 0
        and int(item["response"].get("original_seq_len") or 0) >= item["target_source_tokens"]
        for item in records
    ) and not suspicious
    result = {
        "gate": "flex_attention_lengths",
        "records": records,
        "suspicious_server_log_lines": suspicious,
        "tensor_nan_check": "not_applicable_endpoint_does_not_return_tensors",
        "passed": passed,
    }
    _write(result, args.out)
    if not passed:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--out", type=Path, default=None)
    checkpoint.set_defaults(func=checkpoint_command)

    service = subparsers.add_parser("service")
    service.add_argument("--base-url", required=True)
    service.add_argument("--out", type=Path, default=None)
    service.set_defaults(func=service_command)

    tools = subparsers.add_parser("tools")
    tools.add_argument("--base-url", required=True)
    tools.add_argument("--served-model-name", required=True)
    tools.add_argument("--out", type=Path, default=None)
    tools.set_defaults(func=tools_command)

    proxy = subparsers.add_parser("proxy")
    proxy.add_argument("--base-url", required=True)
    proxy.add_argument("--served-model-name", required=True)
    proxy.add_argument("--full-port", type=int, default=34190)
    proxy.add_argument("--c2kv-port", type=int, default=34191)
    proxy.add_argument("--log-dir", type=Path, default=None)
    proxy.add_argument("--out", type=Path, default=None)
    proxy.set_defaults(func=proxy_command)

    tools_proxy = subparsers.add_parser("tools-proxy")
    tools_proxy.add_argument("--base-url", required=True)
    tools_proxy.add_argument("--served-model-name", required=True)
    tools_proxy.add_argument("--checkpoint", type=Path, required=True)
    tools_proxy.add_argument("--port", type=int, default=34192)
    tools_proxy.add_argument("--log-dir", type=Path, default=None)
    tools_proxy.add_argument("--out", type=Path, default=None)
    tools_proxy.set_defaults(func=tools_proxy_command)

    flex = subparsers.add_parser("flex")
    flex.add_argument("--base-url", required=True)
    flex.add_argument("--checkpoint", type=Path, required=True)
    flex.add_argument("--server-log", default=None)
    flex.add_argument("--out", type=Path, default=None)
    flex.set_defaults(func=flex_command)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
