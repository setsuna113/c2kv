"""Execute an immutable P1 manifest on the existing 1088 service, without retries."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from backends.sglang import SglangBackend
from memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal

PREFIX = "PREPARED_NO_GIST_VECTOR:"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate(artifact):
    if artifact.get("schema") != "a-pre-b-prefix-views-v1" or artifact.get("status") != "prepared_without_model_calls":
        raise ValueError("Wrong P1 preparation identity")
    if artifact.get("freeze_digest") != digest({k:v for k,v in artifact.items() if k != "freeze_digest"}):
        raise ValueError("Prepared P1 artifact changed")
    expected = dict(checkpoint_step=1088, query_projection="base", generation_limit=24,
                    extraction_limit=48, wall_seconds_limit=7200, per_cell_generations=1,
                    sampling=dict(temperature=0.001, seed=0, max_tokens=4096),
                    bytes_per_kv_token=147456, automatic_reruns=0, transport_retries=0,
                    cache_miss_retries=0, tool_execution=False, scorer_calls=0, response_feedback=False)
    for name, value in expected.items():
        if artifact.get(name) != value:
            raise ValueError("P1 contract changed: " + name)
    if not 1 <= len(artifact["schedule"]) <= 24 or not 0 <= len(artifact["extraction_manifest"]) <= 48:
        raise ValueError("P1 attempt manifest exceeds its cap")
    keys = {item["placeholder"] for item in artifact["extraction_manifest"]}
    if len(keys) != len(artifact["extraction_manifest"]):
        raise ValueError("P1 extraction manifest repeats a producer")
    for cell in artifact["schedule"]:
        view = artifact["views"][cell["cell_id"]]
        if digest(view["payload"]) != view["prepared_payload_digest"]:
            raise ValueError("P1 payload changed")
        for message in view["payload"]["messages"]:
            if message.get("c2kv_key_hash") and message["c2kv_key_hash"] not in keys:
                raise ValueError("P1 payload references an undeclared gist")


def execute(artifact, endpoint, output):
    validate(artifact)
    output = Path(output)
    if output.exists():
        raise ValueError("Never overwrite or resume a finite P1 run")
    output.mkdir(parents=True)
    save(output / "prepared.json", artifact)
    started = time.monotonic()
    deadline = started + artifact["wall_seconds_limit"]
    journal = AttemptJournal(output / "attempts.jsonl")
    counts = dict(generation=0, extraction=0)
    state = dict(schema="a-pre-b-prefix-live-v1", status="preflight", counts=counts,
                 endpoint=endpoint, owned_model_launches=0, automatic_reruns=0,
                 transport_retries=0, regeneration=0, tool_execution=0, scorer_calls=0,
                 artifact_digest=artifact["freeze_digest"], cells=[])
    opener = build_opener(ProxyHandler({}))
    backend = SglangBackend(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("No hidden transport")))

    def remaining():
        value = deadline - time.monotonic() - 15
        if value <= 0:
            raise TimeoutError("P1 absolute deadline exhausted")
        return value

    def post(kind, path, payload, request_id, context):
        remaining()
        limit = artifact["generation_limit" if kind == "generation" else "extraction_limit"]
        if counts[kind] >= limit:
            raise RuntimeError("P1 " + kind + " cap exhausted")
        counts[kind] += 1
        handle = journal.start(kind, counts[kind], request_id, context)
        save(output / "receipt.json", state)
        record = dict(kind=kind, path=path, request_id=request_id,
                      payload_digest=digest(payload), status="started")
        began = time.monotonic()
        try:
            request = Request(endpoint + path, data=json.dumps(payload).encode(),
                              headers={"Content-Type":"application/json"}, method="POST")
            with opener.open(request, timeout=min(300, remaining())) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError("Backend response is not an object")
            record.update(status="completed", response=value)
            journal.finish(handle, "completed", usage=value.get("usage") if kind == "generation" else None)
            return value
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            journal.finish(handle, "failed")
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - began
            append(output / "transport.jsonl", record)

    old_alarm = None
    if hasattr(signal, "SIGALRM"):
        def on_deadline(signum, frame):
            raise TimeoutError("P1 absolute model-work deadline exhausted")
        old_alarm = signal.signal(signal.SIGALRM, on_deadline)
        signal.setitimer(signal.ITIMER_REAL, remaining())
    try:
        with opener.open(endpoint + "/get_server_info", timeout=min(10, remaining())) as response:
            info = json.load(response)
        required = dict(model_path="/home/user/checkpoints_upstream/checkpoint-1088",
                        dtype="bfloat16", device="npu", context_length=16384, tp_size=1,
                        enable_c2kv=True, c2kv_query_proj="base", c2kv_tools_dump="full",
                        attention_backend="ascend", served_model_name="c2kv-agent")
        identity = {key:info.get(key) for key in required}
        if identity != required:
            raise ValueError("Existing 1088 server identity differs")
        save(output / "server_identity.json", identity)
        replacements = {}
        state["status"] = "materializing"
        for index, item in enumerate(artifact["extraction_manifest"], 1):
            response = post("extraction", "/v1/c2kv/extract", dict(
                text=item["content"], role=item["role"], compression_ratio=item["ratio"],
                chat_template_kwargs={"enable_thinking":False}), f"p1-extract-{index}",
                dict(benchmark="bfcl", run_id="pre-b-p1", task_id="fixed_prefix_materialization",
                     decision_id=str(index), attempt_id=0))
            if (response.get("success", True) is not True or
                    any(response.get(key) != item[key] for key in ("original_seq_len", "gist_len")) or
                    not isinstance(response.get("key_hash"), str) or not response["key_hash"]):
                raise ValueError("Actual extraction differs from tokenizer-only preparation")
            replacements[item["placeholder"]] = response["key_hash"]
            append(output / "materializations.jsonl", dict(placeholder=item["placeholder"], response=response))
        save(output / "gist_key_mapping.json", replacements)
        state["status"] = "generating"
        case_contexts = {case["case_id"]:case["eval_context"] for case in artifact["cases"]}
        for cell in artifact["schedule"]:
            view = artifact["views"][cell["cell_id"]]
            payload = copy.deepcopy(view["payload"])
            for message in payload["messages"]:
                if message.get("c2kv_key_hash"):
                    message["c2kv_key_hash"] = replacements[message["c2kv_key_hash"]]
            if PREFIX in json.dumps(payload):
                raise ValueError("Placeholder survived materialization")
            context = {**case_contexts[cell["case_id"]], "run_id":"pre-b-p1", "decision_id":cell["cell_id"]}
            save(output / (cell["cell_id"].replace(":", "_") + ".request.json"), payload)
            response = post("generation", "/v1/chat/completions", payload, cell["cell_id"], context)
            normalized = backend.normalize_response(response)
            usage = normalized.get("usage") or {}
            cost = normalized.get("cost") or {}
            if usage.get("prompt_tokens") != len(view["raw_input_ids"]):
                raise ValueError("Actual raw prompt count differs from the frozen view")
            if usage.get("completion_tokens", 4097) > 4096:
                raise ValueError("Generation output cap exceeded")
            if cost.get("bytes_per_kv_token") != artifact["bytes_per_kv_token"]:
                raise ValueError("Backend KV byte geometry differs")
            if view["gist_tokens"]:
                if cost.get("c2kv_query_proj_effective") != "base" or not cost.get("c2kv_gist_seen"):
                    raise ValueError("Backend did not verify the frozen gist/query route")
            row = dict(**cell, status="completed", usage=usage, cost=cost,
                       wire_digest=digest(payload), normalized=normalized,
                       sample_label="preliminary, n=1")
            append(output / "cells.jsonl", row)
            state["cells"].append({key:row[key] for key in ("cell_id", "case_id", "view", "status", "usage")})
            save(output / "receipt.json", state)
            print(json.dumps(dict(cell_id=cell["cell_id"], status="completed")), flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        state["wall_seconds"] = time.monotonic() - started
        if (output / "attempts.jsonl").exists():
            state["attempt_journal"] = summarize_attempt_journal(output / "attempts.jsonl")
        save(output / "receipt.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execute(json.loads(args.prepared.read_text(encoding="utf-8")), args.upstream.rstrip("/"), args.output)


if __name__ == "__main__":
    main()
