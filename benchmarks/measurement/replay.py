"""Execute canonical Full prefixes against a live target arm proxy."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from urllib import request as urlrequest
from urllib.error import HTTPError

from .telemetry import append_jsonl, canonical_sha256, read_jsonl


_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))
EXACT_OUTPUT_ARMS = {"agentkv", "commitkv"}


def validate_teacher_forced_target(target_run_id: str) -> None:
    arm = target_run_id.rsplit("__", 1)[-1]
    if arm in EXACT_OUTPUT_ARMS:
        raise ValueError(
            f"{arm} requires its own unchanged prior assistant output; "
            "Full teacher-forced prefixes are incompatible with exact KV continuation"
        )


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def _paper_measurement(response: Any) -> Any:
    if not isinstance(response, dict):
        return None
    metadata = response.get("metadata")
    proxy = response.get("c2kv_proxy")
    candidates = [
        response.get("paper_measurement"),
        metadata.get("paper_measurement") if isinstance(metadata, dict) else None,
        proxy.get("server_measurement") if isinstance(proxy, dict) else None,
    ]
    return next((value for value in candidates if isinstance(value, dict)), None)



CONTEXT_OVERFLOW_MARKER = "is longer than the model's context length"


def is_context_overflow(raw_response):
    """A recorded Full prefix that does not fit the serving context under this arm.

    The same prefixes fail in the closed-loop run (the harness marks the case
    wrong), so a replay that fails only on them is complete: the failure is a
    property of the arm at that context length, not of the replay.
    """
    text = raw_response if isinstance(raw_response, str) else json.dumps(raw_response or "")
    return CONTEXT_OVERFLOW_MARKER in text

def replay_prefixes(
    prefixes: "str | Path", base_url: str, output: "str | Path",
    *, source_run_id: str, target_run_id: str, timeout: int = 600,
) -> Dict[str, int]:
    validate_teacher_forced_target(target_run_id)
    rows = [row for row in read_jsonl(prefixes)
            if row.get("event_type") == "recorded_prefix"]
    if not rows:
        raise ValueError(f"no recorded_prefix rows in {prefixes}")
    completed = 0
    failed = 0
    context_overflow = 0
    for sequence, row in enumerate(rows):
        if row.get("source_arm") != "full":
            raise ValueError(f"prefix {sequence} source_arm is not full")
        payload = row.get("replay_payload")
        if not isinstance(payload, dict):
            raise ValueError(f"prefix {sequence} has no replay_payload object")
        digest = canonical_sha256(payload)
        if digest != row.get("canonical_sha256"):
            raise ValueError(f"prefix {sequence} canonical digest mismatch")
        request_id = f"replay-{target_run_id}-{sequence:08d}-{uuid.uuid4().hex[:8]}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            chat_url(base_url), data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-C2KV-Measurement-Request-Id": request_id,
                "X-C2KV-Replay-Prefix-Id": str(row.get("prefix_id") or sequence),
            },
        )
        start_unix = time.time_ns()
        start_perf = time.perf_counter_ns()
        status = None
        raw_response = None
        error = None
        try:
            with _OPENER.open(req, timeout=timeout) as response:
                status = response.status
                raw_response = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            status = exc.code
            body = exc.read().decode("utf-8", "replace")
            try:
                raw_response = json.loads(body)
            except json.JSONDecodeError:
                raw_response = body
            error = f"HTTPError: {exc}"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter_ns() - start_perf
        append_jsonl(output, {
            "schema": "c2kv.prefix_replay.v1",
            "event_type": "prefix_replay",
            "source_run_id": source_run_id,
            "target_run_id": target_run_id,
            "sequence": sequence,
            "prefix_id": row.get("prefix_id"),
            "canonical_sha256": digest,
            "request_id": request_id,
            "start_unix_ns": start_unix,
            "end_unix_ns": start_unix + duration,
            "duration_ns": duration,
            "http_status": status,
            "request": payload,
            "response": raw_response,
            "source_paper_measurement": _paper_measurement(
                row.get("source_response")),
            "target_paper_measurement": _paper_measurement(raw_response),
            "error": error,
        })
        if error is not None or status != 200:
            failed += 1
            if is_context_overflow(raw_response):
                context_overflow += 1
        else:
            completed += 1
    return {"prefixes": len(rows), "completed": completed, "failed": failed,
            "context_overflow": context_overflow}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    summary = replay_prefixes(
        args.prefixes, args.base_url, args.output,
        source_run_id=args.source_run_id, target_run_id=args.target_run_id,
        timeout=args.timeout,
    )
    print(json.dumps(summary, sort_keys=True))
    Path(args.output).with_name("replay_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    unexplained = summary["failed"] - summary["context_overflow"]
    if unexplained:
        raise SystemExit(
            f"prefix replay completed with {unexplained} failed requests")


if __name__ == "__main__":
    main()
