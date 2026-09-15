"""Finite development probes of evidence layout and continuous evidence leases.

All cells use captured observable prefixes. Generated answers never become later
inputs, so this runner does not measure closed-loop task success.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from backends import get_backend
from memory_runtime.adapter import RuntimeAdapter, EventStore
from memory_runtime.layout_views import build_layout_views


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_capture(path):
    capture = json.loads(path.read_text())
    if capture.get("schema") != "a-runtime-utilization-prefixes-v1":
        raise ValueError("Unexpected capture schema")
    if json.loads((path.parent / "receipt.json").read_text()).get("status") != "completed":
        raise ValueError("Source capture did not complete its contract")
    for prefix in capture["prefixes"]:
        source = (path.parent / prefix["source_path"]).resolve()
        source.relative_to(path.parent.resolve())
        row = json.loads(source.read_text().splitlines()[prefix["source_line"] - 1])
        view = row["request_view"]
        body = {k: view[k] for k in ("model", "messages", "tools") if k in view}
        body.update(view["sampling"])
        if body != prefix["request_body"] or row["eval_context"] != prefix["eval_context"]:
            raise ValueError("Prefix differs from its original captured request")
    return capture["prefixes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefixes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; automatic rerun/resume is prohibited")
    args.out.mkdir(parents=True)
    config = json.loads((HERE / "configs/protect.json").read_text())
    config["run_id"] = args.out.name
    receipt = {
        "schema": "a-runtime-utilization-probe-v1", "status": "frozen_before_preparation",
        "scope": "fixed observable prefixes; development, preliminary, n=1",
        "sampling": {"temperature": 0.001, "seed": 0, "max_tokens": 512},
        "maximum_chat_requests": 24, "maximum_wall_seconds": 600,
        "automatic_retries": 0, "automatic_reruns": 0, "config": config,
        "upstream": args.upstream, "checkpoint": args.checkpoint,
        "profile": {"query_projection": "base", "doc_packing": "turn", "max_doc_length": 512,
                    "max_doc_num": 12, "compression_ratio": 4},
        "layout_cells": "task1 t0:s1 and t1:s1 x six predeclared layouts",
        "lease_cells": "task30 t0:s2,s3,s4 x protect,recover_once,persistent,no_gist",
        "lease_gate": "skip all lease generations if once/persistent views never differ",
        "evidence_only": "same selected evidence without gist; diagnostic, not fair NoGist baseline",
        "continuation": "later prefixes come from the fixed capture, never generated answers",
        "chat_attempts": 0, "chat_completed": 0, "extraction_attempts": 0,
    }
    receipt_path = args.out / "receipt.json"
    bundle = HERE.parents[1] / "tmp/a_memory_runtime_20260907/source_bundle.json"
    if bundle.exists():
        receipt["source_commit"] = json.loads(bundle.read_text())["source_commit"]
    save(receipt_path, receipt)
    started = time.perf_counter()

    def deadline(_signum, _frame):
        raise TimeoutError("Frozen wall budget exhausted")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(receipt["maximum_wall_seconds"])
    raw_post = proxy._post_json

    def recorded_post(path, payload, timeout, retries=0):
        is_chat = path == "/v1/chat/completions"
        if not is_chat:
            receipt["extraction_attempts"] += 1
            save(receipt_path, receipt)
        at = time.perf_counter()
        record = {"path": path, "attempt": 1, "status": "started"}
        try:
            result = raw_post(path, payload, timeout=min(timeout, 120), retries=0)
            record.update(status="completed", response=result if not is_chat else None)
            return result
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - at
            with (args.out / "transport.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        proxy.UPSTREAM = args.upstream
        proxy.QUERY_PROJECTION = "base"
        proxy.DOC_PACKING = "turn"
        proxy.MAX_DOC_LENGTH, proxy.MAX_DOC_NUM = 512, 12
        proxy.NO_UPSTREAM_RETRIES = True
        proxy.CACHE = proxy.ExtractCache()
        proxy.BACKEND = get_backend("sglang", recorded_post)
        proxy.MEMORY_RUNTIME_BYTES_PER_KV_TOKEN = None
        proxy.MEMORY_RUNTIME_FATAL_ERROR = None
        config_path = args.out / "runtime_config.json"
        save(config_path, config)
        prototype = RuntimeAdapter.from_config(str(config_path), args.checkpoint)
        counter = prototype._token_counter
        proxy.MEMORY_RUNTIME = prototype

        def adapter(mode):
            return RuntimeAdapter({**config, "mode": mode}, counter)

        adapters = {mode: adapter(mode) for mode in (
            "full_shared", "legacy", "protect", "recover_once", "persistent", "no_gist")}
        cells = []

        def prepare(prefix, modes):
            body = prefix["request_body"]
            messages, tools = body["messages"], body.get("tools")
            context = {**prefix["eval_context"], "run_id": config["run_id"]}
            full, full_counts = proxy._assemble(messages, get_arm("full"))
            legacy, legacy_counts = proxy._assemble(messages, get_arm("c2kv4"), timeout=120)
            views = {}
            for mode in modes:
                raw, counts = (full, full_counts) if mode in {"full_shared", "no_gist"} else (legacy, legacy_counts)
                views[mode] = adapters[mode].apply(messages, raw, counts, context, tools)
            return body, views

        def add(prefix, group, label, messages, counts, notes):
            cells.append({"cell_id": f"{group}_{len(cells):02d}_{label}", "group": group,
                          "label": label, "eval_context": prefix["eval_context"],
                          "source": {k: prefix[k] for k in ("source_path", "source_line")},
                          "messages": messages, "counts": counts, "notes": notes,
                          "tools": prefix["request_body"].get("tools"), "status": "prepared"})

        prefixes = load_capture(args.prefixes)
        receipt["source_capture_verified_against_request_log"] = True
        save(args.out / "layout_source_prefixes.json", prefixes)
        if [(p["eval_context"]["task_id"], p["eval_context"]["user_turn"], p["eval_context"]["step"]) for p in prefixes] != [
            ("multi_turn_base_1", 0, 1), ("multi_turn_base_1", 1, 1)]:
            raise ValueError("Capture differs from the frozen two-prefix selection")
        for prefix in prefixes:
            body, views = prepare(prefix, ("full_shared", "legacy", "protect"))
            full, fc = views["full_shared"]
            legacy, lc = views["legacy"]
            protect, pc = views["protect"]
            if full != proxy._assemble(body["messages"], get_arm("full"))[0]:
                raise ValueError("Full-shared identity control changed the Full prompt")
            layouts = build_layout_views(
                store=EventStore.from_messages(prefix["eval_context"]["task_id"], body["messages"]),
                full_messages=full, full_counts=fc, legacy_messages=legacy, legacy_counts=lc,
                protect_messages=protect, protect_counts=pc, token_counter=counter, tools=body.get("tools"),
                bytes_per_kv_token=config["bytes_per_kv_token"], history_budget_bytes=config["history_budget_bytes"],
                workspace_budget_bytes=config["workspace_budget_bytes"])
            if len(layouts) != 6:
                raise ValueError("Layout matrix must have six cells per prefix")
            for view in layouts:
                add(prefix, "layout", view["label"], view["messages"], view["counts"], view["notes"])

        lease_source = HERE / "tests/fixtures/task31_continuous_lease_prefixes.json"
        lease_prefixes = json.loads(lease_source.read_text())["prefixes"]
        save(args.out / "lease_source_prefixes.json", lease_prefixes)
        lease_differences = []
        for prefix in lease_prefixes:
            _, views = prepare(prefix, ("protect", "recover_once", "persistent", "no_gist"))
            once, persistent = views["recover_once"], views["persistent"]
            lease_differences.append({"context": prefix["eval_context"],
                "selected_ids_differ": once[1]["memory_runtime"]["selected_event_ids"] != persistent[1]["memory_runtime"]["selected_event_ids"],
                "messages_differ": once[0] != persistent[0]})
            for mode, (messages, counts) in views.items():
                add(prefix, "lease", mode, messages, counts, "Fixed captured continuation; no generated action executed")
        lease_valid = any(row["selected_ids_differ"] and row["messages_differ"] for row in lease_differences)
        receipt.update(status="prepared_before_generation", lease_differences=lease_differences,
                       lease_gate_passed=lease_valid, prepared_cells=len(cells),
                       preparation_wall_seconds=time.perf_counter() - started)
        if len(cells) != receipt["maximum_chat_requests"]:
            raise ValueError("Prepared matrix differs from the frozen request cap")
        save(args.out / "prepared_views.json", cells)
        save(receipt_path, receipt)
        for cell in cells:
            if cell["group"] == "lease" and not lease_valid:
                cell["status"] = "skipped_degenerate_lease"
                continue
            arm = get_arm("c2kv4" if any(m.get("c2kv_key_hash") for m in cell["messages"]) else "full")
            payload = {"model": "c2kv-agent", "messages": cell["messages"], "tools": cell["tools"],
                       **receipt["sampling"], "c2kv_use_gist_projection": False}
            wire = proxy.BACKEND.prepare_chat(payload, arm, None)
            path = args.out / (cell["cell_id"] + ".json")
            result = {**cell, "forwarded": wire, "status": "attempted"}
            save(path, result)
            receipt["chat_attempts"] += 1
            receipt["current_cell"] = cell["cell_id"]
            save(receipt_path, receipt)
            at = time.perf_counter()
            response = recorded_post("/v1/chat/completions", wire, timeout=120)
            result.update(response=response, chat_wall_seconds=time.perf_counter() - at)
            save(path, result)
            normalized = proxy.BACKEND.normalize_response(response)
            proxy._verify_memory_runtime_kv_bytes(result["counts"], normalized)
            if normalized["cost"].get("c2kv_query_proj_effective") != "base":
                raise ValueError("Backend effective query projection differs from the frozen base profile")
            result.update(status="completed", normalized=normalized)
            save(path, result)
            receipt["chat_completed"] += 1
            save(receipt_path, receipt)
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        receipt["total_wall_seconds"] = time.perf_counter() - started
        save(receipt_path, receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "chat_attempts", "chat_completed", "extraction_attempts")}))


if __name__ == "__main__":
    main()
