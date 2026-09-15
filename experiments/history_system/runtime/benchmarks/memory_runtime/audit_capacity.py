"""CPU replay of capacity activation; no generation or extraction is allowed."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from backends.sglang import SglangBackend
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.capacity import measure_full_history


class ExpectedCompression(Exception):
    pass


def recorded_rows(root, variant):
    for path in sorted((root / variant / "logs").glob("proxy_*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                yield path, number, json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prototype = RuntimeAdapter.from_config(str(HERE / "configs/protect.json"), args.tokenizer)
    config = json.loads((HERE / "configs/protect.json").read_text())
    config["history_budget_bytes"] = config["workspace_budget_bytes"]
    config["mode"] = "capacity_protect"
    count = prototype._token_counter
    original_assemble = proxy._assemble

    def forbidden_network(*args, **kwargs):
        raise AssertionError("CPU capacity replay must never call generation or extraction")

    proxy._post_json = forbidden_network
    proxy._extract = forbidden_network
    backend = SglangBackend(forbidden_network)
    rows = []
    for path, number, request in recorded_rows(args.root, "full"):
        native = request["request_view"]
        source, tools = native["messages"], native.get("tools")
        full, full_counts = original_assemble(source, get_arm("full"))
        measurement = measure_full_history(full, full_counts, count, tools, config["bytes_per_kv_token"])
        if (full != request["forwarded_request_views"][0]["messages"]
                or measurement["total_raw_prompt_tokens"] != request["usage"]["prompt_tokens"]
                or request["bytes_per_kv_token"] != config["bytes_per_kv_token"]):
            raise ValueError("Full replay differs from the executed wire view or server tokens")
        expected_activation = measurement["active_history_bytes"] > config["history_budget_bytes"]
        calls = []

        def guarded_assemble(messages, arm, timeout=600):
            if arm.compress_history:
                calls.append(copy.deepcopy(messages))
                raise ExpectedCompression()
            return original_assemble(messages, arm, timeout)

        proxy.ARM = get_arm("c2kv4")
        proxy.MEMORY_RUNTIME = RuntimeAdapter(config, count)
        proxy._assemble = guarded_assemble
        bypass_identity = None
        try:
            out, counts = proxy._prepare_memory_input(source, request["eval_context"], tools)
        except ExpectedCompression:
            if not expected_activation or calls != [source]:
                raise ValueError("Capacity gate called compression for the wrong prefix")
        else:
            if expected_activation or calls or out != full:
                raise ValueError("Capacity gate changed an in-budget Full view or skipped required activation")
            payload = {"model": native.get("model"), "messages": out, "tools": tools,
                       **native.get("sampling", {}), "c2kv_use_gist_projection": False}
            bypass_identity = backend.prepare_chat(payload, get_arm("c2kv4"), None) == backend.prepare_chat(
                {**payload, "messages": full}, get_arm("full"), None)
            if not bypass_identity or counts["memory_runtime"]["capacity_gate"]["compression_activated"]:
                raise ValueError("In-budget backend payload differs from Full")
        finally:
            proxy._assemble = original_assemble

        raw_runtime = RuntimeAdapter({**config, "mode": "raw_recency"}, count)
        raw, raw_counts = raw_runtime.apply(source, full, full_counts, request["eval_context"], tools,
                                             render_full=lambda source: original_assemble(source, get_arm("full")))
        raw_meta = raw_counts["memory_runtime"]
        if ((raw == full) != (not expected_activation)
                or raw_meta["active_history_bytes"] > config["history_budget_bytes"]):
            raise ValueError("The paired raw-recency baseline failed its Full identity/budget contract")
        rows.append({
            "source_path": path.relative_to(args.root).as_posix(), "source_line": number,
            "context": request["eval_context"], "full": measurement,
            "compression_required": expected_activation, "compressed_renderer_calls": len(calls),
            "full_bypass_identity": bypass_identity, "raw_recency_within_budget": True,
        })

    replay = None
    for path, number, request in recorded_rows(args.root, "protect"):
        source = request["request_view"]["messages"]
        tools = request["request_view"].get("tools")
        full, full_counts = original_assemble(source, get_arm("full"))
        measurement = measure_full_history(full, full_counts, count, tools, config["bytes_per_kv_token"])
        old = request["memory_runtime"]
        # Select by capacity only. This CPU boundary test does not choose a
        # generation budget or filter tasks by their official outcome.
        if old["evicted_gist_keys"] or measurement["active_history_bytes"] <= old["active_history_bytes"]:
            continue
        recorded = request["forwarded_request_views"][0]["messages"]
        base = [copy.deepcopy(message) for index, message in enumerate(recorded)
                if index != old["evidence_out_index"]]
        blocks = {block["key_hash"]: block for block in old["block_refs"]}
        shifted = int(not any(message.get("role") == "system" for message in source))
        records = [{
            "out_index": index,
            "source_indices": [position + shifted for position in blocks[message["c2kv_key_hash"]]["source_indices"]],
            "record": {"key_hash": message["c2kv_key_hash"],
                       "gist_len": blocks[message["c2kv_key_hash"]]["gist_tokens"]},
        } for index, message in enumerate(base) if message.get("c2kv_key_hash")]
        saved_counts = dict(full_counts, current_start_out_index=len(base) - request["raw_current_raw"],
                            history_raw=0, compressed_records=records)
        boundary_config = {**config, "history_budget_bytes": measurement["active_history_bytes"] - 1}
        calls = []

        def recorded_compression(messages, arm, timeout=600):
            if arm.compress_history:
                calls.append(copy.deepcopy(messages))
                return copy.deepcopy(base), copy.deepcopy(saved_counts)
            return original_assemble(messages, arm, timeout)

        proxy.MEMORY_RUNTIME = RuntimeAdapter(boundary_config, count)
        proxy._assemble = recorded_compression
        try:
            out, counts = proxy._prepare_memory_input(source, request["eval_context"], tools)
        finally:
            proxy._assemble = original_assemble
        expected, _ = RuntimeAdapter({**boundary_config, "mode": "protect"}, count).apply(
            source, base, saved_counts, request["eval_context"], tools)
        if calls != [source] or out != expected or out != recorded:
            raise ValueError("Activated capacity protection differs from the recorded protection view")
        if counts["memory_runtime"]["total_raw_prompt_tokens"] != request["usage"]["prompt_tokens"]:
            raise ValueError("Activated replay does not match original server raw tokens")
        replay = {
            "source_path": path.relative_to(args.root).as_posix(), "source_line": number,
            "context": request["eval_context"], "full": measurement,
            "cpu_boundary_budget_bytes": boundary_config["history_budget_bytes"],
            "same_as_recorded_protect_wire": True, "same_as_current_protect": True,
            "compressed_renderer_calls": len(calls),
            "scope": "recorded gist carriers/lengths reused; no extraction or generation performed",
            "memory_runtime": counts["memory_runtime"],
        }
        break
    if not rows or replay is None:
        raise ValueError("Capacity replay lacks Full inputs or a recorded compression boundary case")
    result = {
        "schema": "a-runtime-capacity-cpu-audit-v1", "status": "passed",
        "additional_chat_requests": 0, "additional_extraction_requests": 0,
        "scope": "CPU gate and recorded-view replay; live compression after activation still requires the finite model pilot",
        "history_budget_bytes": config["history_budget_bytes"],
        "full_request_count": len(rows),
        "full_bypass_identity_count": sum(row["full_bypass_identity"] is True for row in rows),
        "above_budget_lazy_activation_count": sum(row["compression_required"] for row in rows),
        "raw_recency_checked_count": len(rows), "recorded_compression_replay_count": 1,
        "recorded_compression_replay": replay, "rows": rows,
        "source_bundle": json.loads((HERE.parents[1] / "tmp/a_memory_runtime_20260907/source_bundle.json").read_text()),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "recorded_compression_replay", "source_bundle"}}))


if __name__ == "__main__":
    main()
