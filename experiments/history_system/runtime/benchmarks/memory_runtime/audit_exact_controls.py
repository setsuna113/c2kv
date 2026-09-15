"""CPU/tokenizer replay of shared Full and NoGist on captured source prefixes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.audit_full_capacity_aux import _recorded_rows, _single_wire

MODES = ("full_exact_shared", "capacity_exact_no_gist")


def audit(source_root, tokenizer):
    configs = {mode: json.loads((HERE / f"configs/{mode}.json").read_text()) for mode in MODES}
    prototype = RuntimeAdapter.from_config(str(HERE / f"configs/{MODES[0]}.json"), tokenizer)
    counter = prototype._token_counter
    runtimes = {mode: RuntimeAdapter(config, counter) for mode, config in configs.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU controls audit cannot call extraction or generation")

    proxy._post_json = proxy._extract = forbidden
    rows = []
    for path, line, request in _recorded_rows(source_root):
        source, tools = request["request_view"]["messages"], request["request_view"].get("tools")
        full, full_counts = proxy._assemble(source, get_arm("full"))
        old = request["memory_runtime"]
        old_wire = _single_wire(request)
        observed_tokens = counter([m for m in old_wire if not m.get("c2kv_key_hash")], tools)
        if observed_tokens != request["usage"]["prompt_tokens"] or not old["raw_prompt_tokens_verified_by_backend"]:
            raise ValueError("Tokenizer differs from the recorded backend count")
        views = {}
        for mode, runtime in runtimes.items():
            context = {**request["eval_context"], "run_id": runtime.run_id}
            out, counts, prepared = runtime.prepare_exact(source, full, full_counts, context,
                tools, render_compressed=forbidden)
            metadata = counts["memory_runtime"]
            gate_key, activated_key = (("auxiliary_gate", "auxiliary_activated") if mode == MODES[0]
                                      else ("capacity_gate", "compression_activated"))
            gate = metadata[gate_key]
            activated = gate[activated_key]
            if activated != old["capacity_gate"]["compression_activated"]:
                raise ValueError("Shared gate differs from captured capacity transition")
            if counter(out, tools) != metadata["total_raw_prompt_tokens"]:
                raise ValueError("New rendered view disagrees with its actual tokenizer count")
            if counts["gist_tokens"] != 0 or counts["compressed_records"]:
                raise ValueError("Full/NoGist control contains gist")
            if not activated:
                if out != full or metadata["evidence_bytes"] != 0:
                    raise ValueError("Below-B control is not exact Full identity")
            else:
                index = metadata["evidence_out_index"]
                packet = out[index] if index is not None else None
                if packet != old_wire[old["evidence_out_index"]]:
                    raise ValueError("Shared initial protection packet differs from recorded source selection")
                if metadata["auxiliary_selection_bytes"] != old["evidence_bytes"]:
                    raise ValueError("Shared E reference cost differs from recorded selection")
                if metadata["evidence_bytes"] > runtime.config.workspace_budget_bytes:
                    raise ValueError("Actual E marginal exceeds W")
                if mode == MODES[0]:
                    if out[:index] + out[index+1:] != full or metadata["budget_applies"] is not False:
                        raise ValueError("Full-shared cropped Full or applied the history cap")
                else:
                    if metadata["active_history_bytes"] > runtime.config.history_budget_bytes:
                        raise ValueError("NoGist joint R/E view exceeds B")
                    if set(metadata["raw_history_event_ids"]) & set(metadata["selected_event_ids"]):
                        raise ValueError("Raw body duplicates an E event")
                    if set(metadata["visible_source_indices"]) != set(prepared.visible_source_indices):
                        raise ValueError("Detector visibility differs from actual NoGist sources")
            if mode == MODES[0] and set(prepared.visible_source_indices) != set(range(len(source))):
                raise ValueError("Full detector cannot see every original source")
            # No recorded draft is fed to a different representation's policy.
            # This preserves identical empty acquisition history for the paired
            # E check and makes no claim about natural triggers under new views.
            views[mode] = {
                "activated": activated, "memory_runtime": metadata,
                "forwarded_messages": out, "source_indices_visible_to_detector": sorted(prepared.visible_source_indices),
            }
        rows.append({"source_path": path.relative_to(source_root).as_posix(), "source_line": line,
                     "context": request["eval_context"], "views": views})
    counts = [sum(row["views"][mode]["activated"] for row in rows) for mode in MODES]
    if len(rows) != 24 or counts != [5, 5]:
        raise ValueError("Expected the frozen 24-prefix, 19/5 capacity split")
    return {
        "schema": "a-exact-controls-cpu-audit-v1", "status": "passed",
        "scope": "recorded prefixes, real tokenizer, paired empty acquisition state; no new model draft or task score",
        "source_bundle": json.loads((ROOT / "tmp/a_memory_runtime_20260907/source_bundle.json").read_text()),
        "source_root": str(source_root), "configs": configs, "prefixes": len(rows),
        "views": len(rows) * len(MODES), "below_B_identity_per_mode": 19,
        "above_B_shared_evidence_per_mode": 5,
        "generation_calls": 0, "extraction_calls": 0, "scorer_calls": 0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists")
    result = audit(args.source_root.resolve(), args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "prefixes", "views", "below_B_identity_per_mode", "above_B_shared_evidence_per_mode")}))


if __name__ == "__main__":
    main()
