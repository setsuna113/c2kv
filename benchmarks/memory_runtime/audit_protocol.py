"""Rebuild recorded protocol views on CPU and verify measured server tokens.

This does not rerun generation or extraction. Saved gist lengths and keys are
fixed inputs; a corrected policy must choose exactly the already executed view.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.tokenization import TOOL_SCHEMA_PROFILE, serving_tools
from history_memory.packing import native_ids, visible_message


def audit(root, tokenizer_path):
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def count(messages, tools):
        return len(native_ids(tokenizer, messages, tools=serving_tools(tools), generation=True))

    results = []
    states = {}
    for folder in ["protocol_v1", "protocol_remaining_v1"]:
        path = root / folder
        receipt = json.loads((path / "receipt.json").read_text())
        configs = {v["name"]: v["runtime"] for v in receipt["variants"]}
        requests = {row["request_index"]: row for row in map(json.loads, (path / "requests.jsonl").read_text().splitlines())}
        responses = {row["request_index"]: row for row in map(json.loads, (path / "responses.jsonl").read_text().splitlines())}
        for row in receipt["request_receipts"]:
            request = requests[row["request_index"]]["body"]
            body = responses[row["request_index"]]["body"]
            mode = row["variant"]
            source, tools = request["messages"], request["tools"]
            expected_tokens = body["usage"]["prompt_tokens"]
            full, full_counts = proxy._assemble(source, get_arm("full"))
            corrected = None
            old = row.get("memory_runtime")
            if mode == "full_native":
                actual_tokens = count([visible_message(m) for m in source], tools)
                unchanged = True
            elif mode == "full":
                actual_tokens = count(full, tools)
                unchanged = True
            else:
                adapter = states.setdefault(mode, RuntimeAdapter(configs[mode], count))
                if mode in {"no_gist", "full_shared"}:
                    assembled, counts = full, full_counts
                else:
                    cutoff = full_counts["current_start_out_index"]
                    systems = [m for m in full[:cutoff] if m.get("role") == "system"]
                    blocks = old["block_refs"]
                    carriers = [{"role": "user", "content": "recorded gist carrier", "c2kv_key_hash": b["key_hash"], "c2kv_ratio": 4} for b in blocks]
                    assembled = systems + carriers + full[cutoff:]
                    shifted = int(not any(m.get("role") == "system" for m in source))
                    counts = dict(full_counts)
                    counts.update(current_start_out_index=len(systems) + len(carriers),
                                  history_raw=0, compressed_records=[
                                      dict(out_index=len(systems) + i,
                                           source_indices=[j + shifted for j in b["source_indices"]],
                                           record=dict(key_hash=b["key_hash"], gist_len=b["gist_tokens"], original_seq_len=0))
                                      for i, b in enumerate(blocks)])
                out, corrected_counts = adapter.apply(source, assembled, counts, request["c2kv_eval_context"], tools)
                corrected = corrected_counts["memory_runtime"]
                actual_tokens = count([m for m in out if not m.get("c2kv_key_hash")], tools)
                old_keys = [b["key_hash"] for b in old["block_refs"]]
                new_keys = [b["key_hash"] for b in corrected["block_refs"]]
                unchanged = (old["selected_event_ids"] == corrected["selected_event_ids"] and old_keys == new_keys)
            results.append(dict(
                variant=mode, case_id=row["case_id"], response_id=row["response_id"],
                backend_prompt_tokens=expected_tokens, reconstructed_prompt_tokens=actual_tokens,
                raw_token_parity=actual_tokens == expected_tokens,
                selected_view_unchanged=unchanged, corrected_memory_runtime=corrected,
            ))
    passed = len(results) == 16 and all(r["raw_token_parity"] and r["selected_view_unchanged"] for r in results)
    return dict(schema="a-runtime-protocol-cpu-audit-v1", status="passed" if passed else "failed",
                tokenizer_path=tokenizer_path, tokenizer_class=type(tokenizer).__name__,
                transformers_version=transformers.__version__, tool_schema_profile=TOOL_SCHEMA_PROFILE,
                generated_requests=16, additional_model_requests=0, additional_gist_extractions=0,
                original_metadata_issue="len(BatchEncoding) counted fields instead of input_ids; original raw/evidence budget fields are invalid",
                interpretation="CPU replay validates the executed views and corrected byte budgets; it does not retroactively validate the original online enforcement",
                rows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.tokenizer)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
