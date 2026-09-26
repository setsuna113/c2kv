"""Prepare paired §4.4 layouts from an immutable recorded-decision JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks import toolmemory, toolselection
from .core import (DEFAULT_SELECTOR_POLICY, FULL_CONTROL_POLICY, canonical_calls,
                   full_tool_spans, pack_layout, retrieval_layout, runtime_modules,
                   sha256_file)

SCHEMA = "c2kv-paper-tool-definition-recorded-v1"
LAYOUTS = ("full", "uniform", "hybrid", "random", "retrieval")


def qualification_error(row: dict[str, Any]) -> str | None:
    """Reject calls the chat server cannot parse before freezing a cohort."""
    if "gold_tool_calls" not in row:
        return "missing_gold_tool_calls"
    try:
        canonical_calls(row["gold_tool_calls"])
    except (ValueError, TypeError, KeyError) as exc:
        return f"invalid_gold_tool_calls: {exc}"
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "invalid_messages: expected a list"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"invalid_messages: message {index} is not an object"
        if message.get("tool_calls") is not None:
            try:
                canonical_calls(message["tool_calls"])
            except (ValueError, TypeError, KeyError) as exc:
                return f"invalid_history_tool_calls: message {index}: {exc}"
    return None


def prepare(input_path: Path, checkpoint: Path, output: Path, *, k: int = 3,
            seed: int = 42, ratios: tuple[int, ...] = (8, 12),
            interface_policy: str = "none",
            selector_policy: str = DEFAULT_SELECTOR_POLICY) -> dict[str, Any]:
    input_path, checkpoint, output = input_path.resolve(), checkpoint.resolve(), output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output.exists():
        raise FileExistsError(output)
    if k < 1 or not ratios or any(ratio not in (8, 12) for ratio in ratios):
        raise ValueError("k must be positive and T0 ratios must be 8 or 12")
    toolmemory.ToolMemorySpec(
        ratio=ratios[0], layout="hybrid", top_k=k,
        interface_policy=interface_policy, selector_policy=selector_policy,
    ).validate()
    if not (checkpoint / "tokenizer.json").is_file():
        raise ValueError("T0 checkpoint needs tokenizer.json for a portable token identity")
    for ratio in ratios:
        toolmemory.load_tool_checkpoint_contract(checkpoint, toolmemory.ToolMemorySpec(ratio=ratio))
    model_files = sorted(checkpoint.glob("model*.safetensors"))
    if not model_files:
        raise ValueError("T0 checkpoint has no safetensors model weights")
    tokenizer = toolmemory.NativeTokenizer(checkpoint)._load()
    output.mkdir(parents=True)
    records_path = output / "records.jsonl"
    excluded_path = output / "excluded.jsonl"
    count, decisions, excluded = 0, 0, 0
    seen: set[str] = set()
    try:
        with (input_path.open("r", encoding="utf-8") as source,
              records_path.open("w", encoding="utf-8") as sink,
              excluded_path.open("w", encoding="utf-8") as excluded_sink):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Input row {line_number} is not an object")
                decision_id = row.get("decision_id")
                if decision_id in seen:
                    raise ValueError(f"Duplicate decision_id: {decision_id}")
                seen.add(decision_id)
                reason = qualification_error(row)
                if reason is not None:
                    excluded_sink.write(json.dumps({
                        "decision_id": decision_id, "input_line": line_number,
                        "reason": reason,
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    excluded += 1
                    continue
                _, _, _, _, native_ids = runtime_modules()
                base_prompt_tokens = len(native_ids(tokenizer, row["messages"], generation=True))
                decisions += 1
                for ratio in ratios:
                    full = pack_layout(row, tokenizer, layout="full", ratio=ratio, k=k,
                                       seed=seed, interface_policy=interface_policy,
                                       selector_policy=selector_policy)
                    full["tool_token_spans"] = [list(span) for span in full_tool_spans(row, tokenizer, ratio=ratio)]
                    uniform = pack_layout(row, tokenizer, layout="uniform", ratio=ratio, k=k,
                                          seed=seed, interface_policy=interface_policy,
                                          selector_policy=selector_policy)
                    hybrid = pack_layout(row, tokenizer, layout="hybrid", ratio=ratio, k=k,
                                         seed=seed, interface_policy=interface_policy,
                                         selector_policy=selector_policy)
                    random = pack_layout(row, tokenizer, layout="random", ratio=ratio, k=k,
                                         seed=seed, interface_policy=interface_policy,
                                         selector_policy=selector_policy)
                    retrieval = retrieval_layout(row, tokenizer, ratio=ratio,
                                                 allowance_tokens=hybrid["resident_kv_tokens"], k=k,
                                                 interface_policy=interface_policy,
                                                 selector_policy=selector_policy)
                    for record in (full, uniform, hybrid, random, retrieval):
                        record["schema"] = SCHEMA
                        record["input_line"] = line_number
                        record["base_prompt_tokens_without_tool_protocol"] = base_prompt_tokens
                        sink.write(json.dumps(record, ensure_ascii=False,
                                              separators=(",", ":"), allow_nan=False) + "\n")
                        count += 1
        manifest = {
            "schema": SCHEMA, "purpose": "recorded_next_action_tool_definition",
            "source": {"path": str(input_path), "sha256": sha256_file(input_path)},
            "checkpoint": {"path": str(checkpoint),
                           "config_sha256": sha256_file(checkpoint / "config.json"),
                           "model_files_sha256": {path.name: sha256_file(path) for path in model_files}},
            "tokenizer_files_sha256": {
                name: sha256_file(checkpoint / name)
                for name in ("tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json", "added_tokens.json",
                             "chat_template.jinja")
                if (checkpoint / name).is_file()
            },
            "selector": {"ranker": toolmemory.RANKER, "k": k, "seed": seed},
            "ratios": list(ratios), "layouts": list(LAYOUTS),
            "decisions": decisions,
            "qualification": {
                "excluded": excluded,
                "path": "excluded.jsonl",
                "sha256": sha256_file(excluded_path),
                "bytes": excluded_path.stat().st_size,
            },
            "records": {"path": "records.jsonl", "sha256": sha256_file(records_path),
                        "bytes": records_path.stat().st_size, "count": count},
        }
        if selector_policy == "last_user_adaptive_v1":
            manifest["selector"].pop("k")
            manifest["selector"].update({
                "policy": selector_policy,
                "selector_version": toolselection.selector_version(selector_policy),
                "selection_count": "adaptive",
                "relative_threshold": toolselection.ADAPTIVE_RELATIVE_THRESHOLD,
                "top_k_cap": None,
                "fixed_control_k": k,
            })
        elif selector_policy != DEFAULT_SELECTOR_POLICY:
            manifest["selector"].update({
                "policy": selector_policy,
                "selector_version": toolselection.selector_version(selector_policy),
            })
        if interface_policy != "none":
            manifest["interface_policy"] = interface_policy
            manifest["interface_render_profile"] = toolmemory.INTERFACE_RENDER_PROFILE
            manifest["full_control_policy"] = FULL_CONTROL_POLICY
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        if records_path.exists():
            records_path.unlink()
        if excluded_path.exists():
            excluded_path.unlink()
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
        raise
