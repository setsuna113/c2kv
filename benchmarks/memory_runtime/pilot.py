"""Finite protocol replay for the A memory-runtime smoke test.

This stage replays two frozen, observable BFCL request prefixes through eight
runtime variants.  It is deliberately not a benchmark scorer: the artifacts
prove request/response protocol parity and memory-runtime accounting before an
official full-task pilot is allowed to run.

The runner owns only the local proxy processes that it starts.  It never starts,
stops, or probes the upstream model server outside the normal chat request.
Every planned request is attempted at most once.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE.parent
FIXTURE_DIR = HERE / "tests" / "fixtures"

MODEL = "c2kv-agent"
TEMPERATURE = 0
SEED = 0
MAX_TOKENS = 512
MAX_CHAT_REQUESTS = 16
HTTP_TIMEOUT_SECONDS = 600.0
RUNTIME_VERSION = "a-event-runtime-v1"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    protocol_label: str
    arm: str
    config_name: str | None


@dataclass(frozen=True)
class ResolvedVariant:
    spec: VariantSpec
    config_path: Path | None
    config: Mapping[str, Any] | None


# ``full_native`` preserves the benchmark's native tool messages. ``full`` is
# the existing uncompressed training-dialect renderer.  A runtime config changes
# memory selection only; the underlying arm remains fixed by this table.
VARIANT_SPECS = (
    VariantSpec("full_native", "full_native", "full_native", None),
    VariantSpec("full", "full(training)", "full", None),
    VariantSpec("legacy", "legacy", "c2kv4", "legacy.json"),
    VariantSpec("protect", "protect", "c2kv4", "protect.json"),
    VariantSpec("recover_once", "recover_once", "c2kv4", "recover_once.json"),
    VariantSpec("persistent", "persistent", "c2kv4", "persistent.json"),
    VariantSpec("no_gist", "no_gist", "full", "no_gist.json"),
    VariantSpec("full_shared", "full_shared", "full", "full_shared.json"),
)


class PilotFailure(RuntimeError):
    """A fail-fast pilot error with a stable machine-readable kind."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PilotFailure("input_error", f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PilotFailure("contract_error", f"invalid JSON in {path}: {exc}") from exc


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PilotFailure("contract_error", f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotFailure("contract_error", f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise PilotFailure("contract_error", f"{field} must be {qualifier}")
    return value


def _validate_complete_events(messages: Sequence[Mapping[str, Any]], case_id: str) -> None:
    python_dir = str(HERE.parents[1] / "python")
    inserted = python_dir not in sys.path
    if inserted:
        sys.path.insert(0, python_dir)
    try:
        from history_memory.events import EventStore

        store = EventStore.from_messages(case_id, messages)
    except (TypeError, ValueError) as exc:
        raise PilotFailure(
            "contract_error", f"case {case_id} is not a valid observable event prefix: {exc}"
        ) from exc
    finally:
        if inserted:
            sys.path.remove(python_dir)
    incomplete = [event.event_id for event in store.events if not event.complete]
    if incomplete:
        raise PilotFailure(
            "contract_error", f"case {case_id} has incomplete events: {incomplete!r}"
        )


def load_corpus(fixture_dir: Path = FIXTURE_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and cross-check the frozen online-visible prefixes and provenance."""
    corpus_path = fixture_dir / "corpus.json"
    selection_path = fixture_dir / "source_selection.json"
    corpus = _read_json(corpus_path)
    selection = _read_json(selection_path)
    if not isinstance(corpus, dict) or corpus.get("schema") != "c2kv.memory_runtime.smoke_corpus.v1":
        raise PilotFailure("contract_error", f"unexpected corpus schema in {corpus_path}")
    if not isinstance(selection, dict) or selection.get("schema") != "c2kv.memory_runtime.source_selection.v1":
        raise PilotFailure("contract_error", f"unexpected source-selection schema in {selection_path}")

    cases = corpus.get("cases")
    selected = selection.get("selected")
    if not isinstance(cases, list) or len(cases) != 2:
        raise PilotFailure("contract_error", "protocol corpus must contain exactly two cases")
    if not isinstance(selected, list) or len(selected) != len(cases):
        raise PilotFailure("contract_error", "source selection must cover every protocol case")

    allowed_case_fields = {
        "case_id", "task_id", "source_path", "source_line", "messages", "tools"
    }
    seen_case_ids: set[str] = set()
    for index, (case, source) in enumerate(zip(cases, selected)):
        if not isinstance(case, dict) or set(case) != allowed_case_fields:
            raise PilotFailure(
                "contract_error",
                f"corpus case {index} must contain only online input and source fields",
            )
        if not isinstance(source, dict):
            raise PilotFailure("contract_error", f"source selection row {index} is not an object")
        case_id = _nonempty_string(case.get("case_id"), f"corpus case {index} case_id")
        if case_id in seen_case_ids:
            raise PilotFailure("contract_error", f"duplicate corpus case_id {case_id}")
        seen_case_ids.add(case_id)
        _nonempty_string(case.get("task_id"), f"case {case_id} task_id")
        _nonempty_string(case.get("source_path"), f"case {case_id} source_path")
        _integer(case.get("source_line"), f"case {case_id} source_line", positive=True)
        messages = case.get("messages")
        tools = case.get("tools")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) for message in messages
        ):
            raise PilotFailure("contract_error", f"case {case_id} has invalid messages")
        if not isinstance(tools, list) or not tools or any(
            not isinstance(tool, dict) for tool in tools
        ):
            raise PilotFailure("contract_error", f"case {case_id} has invalid tools")
        _validate_complete_events(messages, case_id)
        for field in ("case_id", "task_id", "source_line"):
            if source.get(field) != case.get(field):
                raise PilotFailure(
                    "contract_error",
                    f"case {case_id} disagrees with source selection on {field}",
                )
    return corpus, selection


def load_variants(config_dir: Path) -> tuple[list[ResolvedVariant], str]:
    """Resolve the frozen routing table and require one shared explicit run ID."""
    if not config_dir.is_dir():
        raise PilotFailure("input_error", f"config directory does not exist: {config_dir}")
    resolved: list[ResolvedVariant] = []
    run_ids: set[str] = set()
    for spec in VARIANT_SPECS:
        if spec.config_name is None:
            resolved.append(ResolvedVariant(spec, None, None))
            continue
        path = (config_dir / spec.config_name).resolve()
        config = _read_json(path)
        if not isinstance(config, dict):
            raise PilotFailure("contract_error", f"runtime config must be an object: {path}")
        if config.get("mode") != spec.name:
            raise PilotFailure(
                "contract_error",
                f"{path.name} mode must be {spec.name!r}, got {config.get('mode')!r}",
            )
        run_ids.add(_nonempty_string(config.get("run_id"), f"{path.name} run_id"))
        _integer(config.get("bytes_per_kv_token"), f"{path.name} bytes_per_kv_token", positive=True)
        for field in (
            "history_budget_bytes",
            "workspace_budget_bytes",
            "lease_decisions",
            "max_retrieved_events",
        ):
            _integer(config.get(field), f"{path.name} {field}")
        resolved.append(ResolvedVariant(spec, path, config))
    if len(run_ids) != 1:
        raise PilotFailure(
            "contract_error", f"all runtime configs must share one run_id, got {sorted(run_ids)!r}"
        )
    return resolved, next(iter(run_ids))


def _validate_upstream(upstream: str) -> str:
    parsed = urllib.parse.urlsplit(upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PilotFailure("input_error", "--upstream must be an absolute HTTP(S) base URL")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise PilotFailure(
            "input_error", "--upstream must be the server base URL without /v1, query, or fragment"
        )
    return upstream.rstrip("/")


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_proxy(
    upstream: str,
    variant: ResolvedVariant,
    port: int,
    log_dir: Path,
    proxy_python: str,
    tokenizer: str,
):
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))
    from run import start_proxy

    return start_proxy(
        upstream,
        variant.spec.arm,
        port,
        log_dir,
        backend="sglang",
        doc_packing="turn",
        max_doc_length=512,
        max_doc_num=12,
        query_projection="base",
        python_bin=proxy_python,
        memory_runtime_config=str(variant.config_path or ""),
        memory_tokenizer=tokenizer if variant.config_path else "",
        no_upstream_retries=True,
    )


def _stop_owned_proxy(proc: Any) -> dict[str, Any]:
    """Stop exactly the process object returned by this runner's launch call."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)
    return {"pid": proc.pid, "returncode": proc.returncode, "stopped_at": _utc_now()}


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _post_chat(endpoint: str, payload: Mapping[str, Any]) -> tuple[int, Any, str, float]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PilotFailure("upstream_error", f"chat request failed without retry: {exc}") from exc
    elapsed = time.perf_counter() - started
    raw_text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None
    return status, parsed, raw_text, elapsed


def _memory_metadata(response: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    proxy = response.get("c2kv_proxy")
    if not isinstance(proxy, dict):
        raise PilotFailure("contract_error", "response lacks c2kv_proxy metadata")
    direct = proxy.get("memory_runtime")
    counts = proxy.get("counts")
    nested = counts.get("memory_runtime") if isinstance(counts, dict) else None
    if direct is not None and nested is not None and direct != nested:
        raise PilotFailure("contract_error", "conflicting memory_runtime metadata paths")
    metadata = direct if direct is not None else nested
    path = (
        "c2kv_proxy.memory_runtime"
        if direct is not None
        else "c2kv_proxy.counts.memory_runtime" if nested is not None else None
    )
    if metadata is not None and not isinstance(metadata, dict):
        raise PilotFailure("contract_error", f"{path} must be an object")
    return metadata, path


def _validate_memory_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: ResolvedVariant,
    case: Mapping[str, Any],
    run_id: str,
) -> None:
    expected = {
        "version": RUNTIME_VERSION,
        "mode": variant.spec.name,
        "run_id": run_id,
        "task_id": case["task_id"],
        "attempt_id": 0,
        "decision_id": case["case_id"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise PilotFailure(
                "contract_error",
                f"{variant.spec.name}/{case['case_id']} metadata {field} "
                f"must be {value!r}, got {metadata.get(field)!r}",
            )
    for field in ("active_history_bytes", "evidence_bytes", "gist_tokens"):
        _integer(metadata.get(field), f"memory metadata {field}")
    _integer(metadata.get("bytes_per_kv_token"), "memory metadata bytes_per_kv_token", positive=True)
    if metadata.get("bytes_per_kv_token") != variant.config["bytes_per_kv_token"]:
        raise PilotFailure(
            "contract_error", "response bytes_per_kv_token differs from the selected config"
        )
    if metadata.get("byte_geometry_verified_by_backend") is not True:
        raise PilotFailure(
            "geometry_error",
            "memory runtime byte geometry was not verified by the backend",
        )
    budget_applies = variant.spec.name != "full_shared"
    if metadata.get("budget_applies") is not budget_applies:
        raise PilotFailure(
            "contract_error",
            f"{variant.spec.name} budget_applies must be {budget_applies}",
        )
    if (
        budget_applies
        and metadata["active_history_bytes"] > variant.config["history_budget_bytes"]
    ):
        raise PilotFailure(
            "budget_error",
            f"{variant.spec.name} active history exceeds its configured byte cap",
        )
    for field in (
        "evicted_gist_keys",
        "protected_event_ids",
        "retrieved_event_ids",
        "retained_event_ids",
    ):
        value = metadata.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise PilotFailure("contract_error", f"memory metadata {field} must be a string list")


def validate_response(
    response: Any,
    *,
    variant: ResolvedVariant,
    case: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise PilotFailure("contract_error", "chat response is not a JSON object")
    if response.get("error") is not None:
        raise PilotFailure("model_error", f"chat response contains error: {response['error']!r}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PilotFailure("contract_error", "chat response has no first choice")
    if not isinstance(choices[0].get("message"), dict):
        raise PilotFailure("contract_error", "chat response first choice has no message")
    metadata, metadata_path = _memory_metadata(response)
    if variant.config_path is None:
        if metadata is not None:
            raise PilotFailure(
                "contract_error", f"unconfigured variant {variant.spec.name} returned runtime metadata"
            )
    else:
        if metadata is None:
            raise PilotFailure(
                "contract_error", f"configured variant {variant.spec.name} lacks runtime metadata"
            )
        _validate_memory_metadata(
            metadata, variant=variant, case=case, run_id=run_id
        )
    usage = response.get("usage")
    return {
        "response_id": response.get("id"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage if isinstance(usage, dict) else None,
        "memory_runtime_path": metadata_path,
        "memory_runtime": metadata,
    }


def _request_payload(case: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": case["messages"],
        "tools": case["tools"],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "store": False,
        "stream": False,
        "c2kv_eval_context": {
            "run_id": run_id,
            "task_id": case["task_id"],
            "attempt_id": 0,
            "decision_id": case["case_id"],
        },
    }


def run_protocol(
    *,
    upstream: str,
    tokenizer: str,
    config_dir: Path,
    out_dir: Path,
    proxy_python: str,
    completed_receipt: Path | None = None,
) -> dict[str, Any]:
    upstream = _validate_upstream(upstream)
    _nonempty_string(tokenizer, "--tokenizer")
    _nonempty_string(proxy_python, "--proxy-python")
    corpus, selection = load_corpus()
    variants, run_id = load_variants(config_dir.resolve())
    cases = corpus["cases"]
    planned_requests = len(variants) * len(cases)
    if planned_requests > MAX_CHAT_REQUESTS:
        raise PilotFailure(
            "request_budget_error",
            f"planned {planned_requests} requests exceeds cap {MAX_CHAT_REQUESTS}",
        )
    if planned_requests != MAX_CHAT_REQUESTS:
        raise PilotFailure(
            "contract_error",
            f"protocol matrix must contain exactly {MAX_CHAT_REQUESTS} requests, got {planned_requests}",
        )

    prior = None
    already_completed = 0
    if completed_receipt is not None:
        prior = _read_json(completed_receipt)
        done = prior.get("request_receipts", [])
        # Continue only after startup failed between complete variants. A
        # failed or ambiguous model request is never resubmitted here.
        expected_pairs = [(v.spec.name, c["case_id"]) for v in variants for c in cases]
        observed_pairs = [(r["variant"], r["case_id"]) for r in done]
        old_configs = {v["name"]: v.get("runtime") for v in prior.get("variants", [])}
        new_configs = {v.spec.name: dict(v.config) if v.config is not None else None for v in variants}
        if (prior.get("status") != "failed" or prior.get("run_id") != run_id
                or prior.get("requests_attempted") != len(done)
                or prior.get("responses_received") != len(done)
                or not done or len(done) % len(cases)
                or observed_pairs != expected_pairs[:len(done)]
                or old_configs != new_configs
                or not str(prior.get("error", {}).get("message", "")).startswith("proxy did not come up")):
            raise PilotFailure("contract_error", "continuation requires complete leading variants and a startup-only failure")
        already_completed = len(done)
        variants = variants[already_completed // len(cases):]
        planned_requests -= already_completed
        if not planned_requests:
            raise PilotFailure("contract_error", "no unattempted protocol requests remain")

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise PilotFailure(
            "output_exists",
            f"refusing to repeat or overwrite an existing output directory: {out_dir}",
        )
    out_dir.mkdir(parents=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir()
    requests_path = out_dir / "requests.jsonl"
    responses_path = out_dir / "responses.jsonl"
    receipt_path = out_dir / "receipt.json"

    receipt: dict[str, Any] = {
        "schema": "c2kv.memory_runtime.protocol_receipt.v1",
        "stage": "protocol",
        "status": "running",
        "run_id": run_id,
        "started_at": _utc_now(),
        "upstream": upstream,
        "model": MODEL,
        "sampling": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "automatic_retries": 0,
        },
        "request_budget": MAX_CHAT_REQUESTS,
        "planned_requests": planned_requests,
        "completed_receipt": str(completed_receipt.resolve()) if completed_receipt else None,
        "previously_completed_requests": already_completed,
        "aggregate_planned_requests": planned_requests + already_completed,
        "requests_attempted": 0,
        "responses_received": 0,
        "fixture": {
            "corpus": str((FIXTURE_DIR / "corpus.json").resolve()),
            "corpus_schema": corpus["schema"],
            "source_selection": str((FIXTURE_DIR / "source_selection.json").resolve()),
            "source_selection_schema": selection["schema"],
            "case_ids": [case["case_id"] for case in cases],
        },
        "variants": [
            {
                "name": variant.spec.name,
                "protocol_label": variant.spec.protocol_label,
                "arm": variant.spec.arm,
                "runtime_config": str(variant.config_path) if variant.config_path else None,
                "runtime": dict(variant.config) if variant.config is not None else None,
            }
            for variant in variants
        ],
        "request_receipts": [],
        "owned_proxy_processes": [],
        "artifacts": {
            "requests": str(requests_path),
            "responses": str(responses_path),
            "proxy_logs": str(log_dir),
        },
    }
    _write_json_atomic(receipt_path, receipt)

    active_proxy = None
    try:
        for variant in variants:
            port = _free_loopback_port()
            active_proxy, request_log = _launch_proxy(
                upstream, variant, port, log_dir, proxy_python, tokenizer
            )
            proxy_record: dict[str, Any] = {
                "variant": variant.spec.name,
                "arm": variant.spec.arm,
                "pid": active_proxy.pid,
                "port": port,
                "request_log": str(Path(request_log).resolve()),
                "started_at": _utc_now(),
            }
            receipt["owned_proxy_processes"].append(proxy_record)
            _write_json_atomic(receipt_path, receipt)
            try:
                endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
                for case in cases:
                    if receipt["requests_attempted"] >= MAX_CHAT_REQUESTS:
                        raise PilotFailure(
                            "request_budget_error", "chat request cap reached before matrix completion"
                        )
                    request_index = receipt["requests_attempted"] + 1
                    payload = _request_payload(case, run_id)
                    _append_jsonl(
                        requests_path,
                        {
                            "request_index": request_index,
                            "variant": variant.spec.name,
                            "arm": variant.spec.arm,
                            "case_id": case["case_id"],
                            "task_id": case["task_id"],
                            "endpoint": endpoint,
                            "body": payload,
                        },
                    )
                    receipt["requests_attempted"] = request_index
                    _write_json_atomic(receipt_path, receipt)
                    try:
                        status, response, raw_text, elapsed = _post_chat(endpoint, payload)
                    except PilotFailure as exc:
                        _append_jsonl(
                            responses_path,
                            {
                                "request_index": request_index,
                                "variant": variant.spec.name,
                                "case_id": case["case_id"],
                                "status": "transport_error",
                                "error_kind": exc.kind,
                                "error": str(exc),
                            },
                        )
                        raise
                    response_row: dict[str, Any] = {
                        "request_index": request_index,
                        "variant": variant.spec.name,
                        "case_id": case["case_id"],
                        "http_status": status,
                        "wall_seconds": elapsed,
                    }
                    if response is None:
                        response_row["raw_body"] = raw_text
                    else:
                        response_row["body"] = response
                    _append_jsonl(responses_path, response_row)
                    receipt["responses_received"] += 1
                    if not 200 <= status < 300:
                        raise PilotFailure(
                            "upstream_or_model_error",
                            f"{variant.spec.name}/{case['case_id']} returned HTTP {status}",
                        )
                    validated = validate_response(
                        response, variant=variant, case=case, run_id=run_id
                    )
                    request_receipt = {
                        "request_index": request_index,
                        "variant": variant.spec.name,
                        "arm": variant.spec.arm,
                        "case_id": case["case_id"],
                        "task_id": case["task_id"],
                        "http_status": status,
                        "wall_seconds": elapsed,
                        **validated,
                    }
                    receipt["request_receipts"].append(request_receipt)
                    _write_json_atomic(receipt_path, receipt)
            finally:
                stopped = _stop_owned_proxy(active_proxy)
                proxy_record.update(stopped)
                active_proxy = None
                _write_json_atomic(receipt_path, receipt)

        if (
            receipt["requests_attempted"] != planned_requests
            or receipt["responses_received"] != planned_requests
            or len(receipt["request_receipts"]) != planned_requests
        ):
            raise PilotFailure("contract_error", "protocol matrix ended without complete receipts")
        receipt["status"] = "completed"
        receipt["completed_at"] = _utc_now()
        _write_json_atomic(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        if active_proxy is not None:
            # Normally the per-variant finally block owns this path.  This guard
            # covers failures between launch and entry into that block.
            try:
                receipt["owned_proxy_processes"][-1].update(_stop_owned_proxy(active_proxy))
            except Exception as stop_exc:
                receipt["proxy_stop_error"] = str(stop_exc)
        receipt["status"] = "failed"
        receipt["failed_at"] = _utc_now()
        receipt["error"] = {
            "kind": getattr(exc, "kind", "runner_error"),
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json_atomic(receipt_path, receipt)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["protocol"])
    parser.add_argument("--proxy-python", required=True)
    parser.add_argument("--completed-receipt", type=Path,
                        help="continue only unattempted variants after a proxy startup failure")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_protocol(
            upstream=args.upstream,
            tokenizer=args.tokenizer,
            config_dir=args.config_dir,
            out_dir=args.out,
            proxy_python=args.proxy_python,
            completed_receipt=args.completed_receipt,
        )
    except PilotFailure as exc:
        print(f"FATAL [{exc.kind}]: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "run_id": receipt["run_id"],
                "requests_attempted": receipt["requests_attempted"],
                "responses_received": receipt["responses_received"],
                "receipt": str((args.out.resolve() / "receipt.json")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
