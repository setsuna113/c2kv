"""Evaluate the T0 tool checkpoint on the July Agent-LLM-Traces tool-definition cohort.

The C2KV rows of the paper's tool-definition table are regenerated with T0's
own packing (explicit tool protocol in the system message, one document per
tool, 768/64 chunks) and its event-native generator. Examples, the lexical
top-k router, and every metric come from the July driver
(``eval_agent_tool_definition_reuse_baselines.py``), so the output rows merge
with that driver's SnapKV/EPIC rows through
``merge_agent_tool_definition_reuse_baselines_eval.py``.

One process evaluates one layout at one ratio.  ``--dry_run`` packs every
record on CPU without loading model weights.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT / "python", REPOSITORY_ROOT / "python" / "inference", REPOSITORY_ROOT / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_agent_tool_definition_hybrid_router import _query_text, _rank_tools, _tool_name  # noqa: E402
from eval_agent_tool_definition_reuse_baselines import (  # noqa: E402
    _as_tool_list,
    _augment_text_overlap_metrics,
    _extract_tool_name,
    _normalize_text,
    _select_examples,
    _summarize_rows,
)
from history_memory.dataset import Decision  # noqa: E402
from history_memory.events import EventStore, Message  # noqa: E402
from next_compression.exp1_eval import Exp1Record, run_gist_layouts  # noqa: E402
from next_compression.exp1_tools import (  # noqa: E402
    decision_messages,
    lexical_rank,
    pack_layout,
    query_text,
)
from next_compression.tools import ToolPackingError, ToolPreparationConfig  # noqa: E402

LAYOUTS = ("uniform", "hybrid", "full")
CODE_FILES = (
    "agent/eval_t0_traces_tooldef.py",
    "agent/eval_agent_tool_definition_reuse_baselines.py",
    "agent/eval_agent_tool_definition_hybrid_router.py",
    "agent/eval_agent_tool_definition_c2kv.py",
    "agent/train_agent_tool_definition_c2kv.py",
    "python/next_compression/exp1_tools.py",
    "python/next_compression/exp1_eval.py",
    "python/next_compression/tools.py",
    "python/next_compression/inference.py",
    "python/history_memory/inference.py",
    "python/history_memory/packing.py",
    "python/history_memory/runtime.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hashes() -> Dict[str, str]:
    # Line endings are normalized so Windows and server checkouts hash alike.
    return {
        name: _sha256_bytes((REPOSITORY_ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        for name in CODE_FILES
    }


def _load_keys(path: Path) -> List[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value["common_subset"]["common_sample_keys"]
    if not isinstance(value, list) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} does not hold a list of sample keys")
    # The July merge keys rows as "<session_id>\t<qid>"; qid is "<session_id>:<span>".
    qids = []
    for key in value:
        session_id, _, qid = key.rpartition("\t")
        if session_id and not qid.startswith(session_id + ":"):
            raise ValueError(f"Sample key has inconsistent session and qid: {key!r}")
        qids.append(qid)
    if len(set(qids)) != len(qids):
        raise ValueError(f"{path} contains duplicate sample keys")
    return qids


def _decision(example: Any, tools: Sequence[Dict[str, Any]]) -> Decision:
    # The prefix is exactly what the July driver presents: its fixed system
    # prompt followed by the normalized (tool -> user, calls as text) messages.
    messages = [{"role": "system", "content": example.system_prompt}, *example.input_messages]
    store = EventStore.from_messages(example.session_id, messages)
    return Decision(
        decision_id=example.qid,
        session_id=example.session_id,
        source="agent-llm-traces",
        split="eval",
        task_id=example.session_id,
        template_id="agent-llm-traces",
        decision_index=int(example.qid.rsplit(":", 1)[1]),
        source_message_index=len(messages),
        store=store,
        target=Message.from_dict({"role": "assistant", "content": example.answer}),
        tools_json=json.dumps(list(tools), ensure_ascii=False),
    )


def _prepare(args: argparse.Namespace, tokenizer: Any) -> tuple[List[Exp1Record], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    keys = _load_keys(Path(args.keys_file))
    examples, selection_skips = _select_examples(args, tokenizer)
    by_qid = {example.qid: example for example in examples}
    missing = [key for key in keys if key not in by_qid]
    if missing:
        raise ValueError(f"{len(missing)} cohort keys are not among the selected examples: {missing[:5]}")

    config = ToolPreparationConfig(max_raw_tokens=None)
    records: List[Exp1Record] = []
    context: Dict[str, Dict[str, Any]] = {}
    router_mismatch: List[str] = []
    pack_failures: Counter[str] = Counter()
    for qid in keys:
        example = by_qid[qid]
        tools = _as_tool_list(example.tool_definition)
        decision = _decision(example, tools)
        # Racer's selection is the July driver's lexical top-k; T0's port of
        # the same ranker must agree on every decision.
        ranked = _rank_tools(tools, _query_text(example.input_messages, "last_user"))
        ported = lexical_rank(tools, query_text(decision_messages(decision), "last_user"))
        native = tuple(sorted(ranked[: args.top_k]))
        if set(ported[: args.top_k]) != set(native):
            router_mismatch.append(qid)
        try:
            # The uniform prefix is the protocol without schemas; the full prefix
            # adds every schema. Their difference is the raw tool-KV footprint.
            empty = pack_layout(decision, tokenizer, layout="uniform", config=config)
            full = pack_layout(decision, tokenizer, layout="full", config=config)
            if args.layout == "uniform":
                memory = empty
            elif args.layout == "hybrid":
                memory = pack_layout(decision, tokenizer, layout="hybrid", config=config, native_indices=native)
            else:
                memory = full
        except ToolPackingError as error:
            pack_failures[str(error)] += 1
            context[qid] = {"example": example, "skip_reason": f"t0_packing:{error}"}
            continue
        protocol_tokens = len(empty.system_input_ids)
        target_ids = tuple(tokenizer.encode(example.answer.strip(), add_special_tokens=False))
        context[qid] = {
            "example": example,
            "tools": tools,
            "native_indices": list(native) if args.layout == "hybrid" else (list(range(len(tools))) if args.layout == "full" else []),
            "full_tool_tokens": len(full.system_input_ids) - protocol_tokens,
            "native_tool_tokens": len(memory.system_input_ids) - protocol_tokens,
            "top_tool_names": [_tool_name(tools[index]) for index in ranked[: args.top_k]],
        }
        records.append(
            Exp1Record(
                decision_id=qid,
                session_key=f"agent-llm-traces:{example.session_id}",
                source="agent-llm-traces",
                layout=args.layout,
                k=args.top_k if args.layout == "hybrid" else None,
                ratio=args.ratio,
                memory=memory,
                target_ids=target_ids,
                metadata={"native_tool_indices": context[qid]["native_indices"]},
            )
        )
    report = {
        "num_keys": len(keys),
        "num_selected_examples": len(examples),
        "selection_skips": selection_skips,
        "num_records": len(records),
        "pack_failures": dict(pack_failures),
        "router_mismatch": router_mismatch,
    }
    return records, context, report


def _row(record: Exp1Record, generated: Dict[str, Any], context: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    example = context["example"]
    costs = record.memory.costs(record.ratio)
    gist_tokens = int(costs["gist_tokens"])
    resident_tool_tokens = context["native_tool_tokens"] + gist_tokens
    prediction = generated["generated_text"]
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    tools = context["tools"]
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": f"t0_{args.layout}",
        "ratio": 1 if args.layout == "full" else record.ratio,
        "skipped": False,
        "render_profile": "next-compression-tool-explicit-protocol-v2",
        "num_tools": len(tools),
        "num_top_tools": len(context["native_indices"]),
        "num_rest_tools": len(tools) - len(context["native_indices"]),
        "top_tool_names": context["top_tool_names"] if args.layout == "hybrid" else [],
        "router_hit": (target_tool in set(context["top_tool_names"])) if (args.layout == "hybrid" and target_tool) else None,
        "doc_tokens": context["full_tool_tokens"],
        "top_doc_tokens": context["native_tool_tokens"],
        "rest_gist_tokens": gist_tokens,
        "gist_tokens": gist_tokens,
        "encoder_tokens": int(costs["presented_encoder_tokens"]),
        "actual_compression_ratio": round(context["full_tool_tokens"] / resident_tool_tokens, 4) if resident_tool_tokens else 0.0,
        "system_tokens": int(costs["system_tokens"]),
        "prompt_tokens": int(costs["raw_tokens"]),
        "target_tokens": len(record.target_ids),
        "generated_tokens": len(generated["generated_token_ids"]),
        "finish_reason": generated["finish_reason"],
        "generate_sec": generated.get("generate_sec", 0.0),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        profile: Dict[str, Any] = {"checkpoint": str(Path(args.checkpoint).resolve()), "weights_loaded": False}
        generator = None
    else:
        from next_compression.inference import load_next_checkpoint

        generator, tokenizer, profile = load_next_checkpoint(
            args.checkpoint, device=args.device, dtype=args.dtype, decode_strategy="incremental"
        )
    records, context, report = _prepare(args, tokenizer)

    rows: List[Dict[str, Any]] = []
    for qid, item in context.items():
        if "skip_reason" in item:
            example = item["example"]
            rows.append({"qid": qid, "session_id": example.session_id, "mode": f"t0_{args.layout}",
                         "ratio": 1 if args.layout == "full" else args.ratio,
                         "skipped": True, "skip_reason": item["skip_reason"]})
    if generator is None:
        for record in records:
            costs = record.memory.costs(record.ratio)
            item = context[record.decision_id]
            resident = item["native_tool_tokens"] + int(costs["gist_tokens"])
            rows.append({
                "qid": record.decision_id, "mode": f"t0_{args.layout}", "ratio": record.ratio, "dry_run": True,
                "doc_tokens": item["full_tool_tokens"], "top_doc_tokens": item["native_tool_tokens"],
                "gist_tokens": int(costs["gist_tokens"]), "chunks": len(record.memory.chunks),
                "system_tokens": int(costs["system_tokens"]), "prompt_tokens": int(costs["raw_tokens"]),
                "actual_compression_ratio": round(item["full_tool_tokens"] / resident, 4) if resident else 0.0,
            })
    else:
        start = time.perf_counter()
        generated = run_gist_layouts(
            generator, tokenizer, records, max_new_tokens=args.max_new_tokens,
            compute_uniform_ce=False, phase="tooldef_t0_traces",
        )
        elapsed = time.perf_counter() - start
        by_id = {row["decision_id"]: row for row in generated}
        for record in records:
            row = _row(record, by_id[record.decision_id], context[record.decision_id], args)
            rows.append(_augment_text_overlap_metrics(row))
        report["generation_wall_sec"] = round(elapsed, 2)

    summary_args = argparse.Namespace(
        model=str(Path(args.checkpoint).resolve()), base_model=None,
        dataset_path=args.dataset_path, split=args.split,
    )
    summary = {
        "schema": "tooldef-t0-traces-v1",
        "started_utc": started,
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "args": vars(args),
        "checkpoint_profile": {key: value for key, value in profile.items() if isinstance(value, (str, int, float, bool, list, type(None)))},
        "keys_file_sha256": _sha256_file(Path(args.keys_file)),
        "code_sha256_lf": _code_hashes(),
        "platform": {"python": sys.version.split()[0], "machine": platform.machine(), "node": platform.node()},
        "report": report,
        "results": [] if generator is None else _summarize_rows(summary_args, rows),
        "num_rows": len(rows),
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="T0 checkpoint directory.")
    parser.add_argument("--keys_file", required=True, help="JSON list of sample keys, or a July summary with common_subset.")
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument("--ratio", type=int, choices=(8, 12), default=8)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Pack records on CPU without model weights.")
    # Example selection: the July driver's arguments and defaults.
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--tool_document_eval_mode", choices=["full", "per_tool"], default="full")
    parser.add_argument("--min_num_tools", type=int, default=0)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="toolset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    args = parser.parse_args(argv)
    if args.layout == "full" and args.ratio != 8:
        parser.error("the full layout has no gist memory; run it once with the default ratio")
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps({"report": summary["report"], "results": summary["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
