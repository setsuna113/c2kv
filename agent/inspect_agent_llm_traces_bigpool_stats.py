from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer  # noqa: E402

from diagnose_agent_tool_definition_generalization import (  # noqa: E402
    _greedy_namespace_split,
    _namespace_set,
    _session_split,
    _split_groups,
    _toolset_key,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
    _as_tool_list,
    _canonical_tool_definition,
    _render_tool_definition,
    _span_attributes,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _stat(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "avg": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": ordered[0],
        "avg": round(sum(ordered) / len(ordered), 4),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _full_mode_outcome(
    doc_tokens: int,
    threshold: int,
    max_doc_length: int,
    max_doc_num: int,
    truncate: bool,
) -> tuple[bool, str]:
    """Mirror eval_agent_tool_definition_c2kv._build_tool_chunks document_mode="full"."""
    if doc_tokens > threshold:
        return False, f"tool_definition_tokens>{threshold}"
    max_context_tokens = max_doc_length * max_doc_num
    effective = doc_tokens
    if effective > max_context_tokens:
        if not truncate:
            return False, f"tool_definition_tokens>{max_context_tokens}"
        effective = max_context_tokens
    num_chunks = (effective + max_doc_length - 1) // max_doc_length
    if num_chunks > max_doc_num:
        return False, f"tool_definition_docs>{max_doc_num}"
    return True, "ok"


def _per_tool_mode_outcome(
    tool_doc_tokens: Sequence[int],
    threshold: int,
    max_doc_length: int,
    max_doc_num: int,
    truncate: bool,
) -> tuple[bool, str]:
    """Mirror eval_agent_tool_definition_c2kv._build_tool_chunks document_mode="per_tool"."""
    if not tool_doc_tokens:
        return False, "no_parseable_tools"
    doc_tokens = 0
    num_chunks = 0
    for count in tool_doc_tokens:
        doc_tokens += count
        if count > max_doc_length and not truncate:
            return False, f"tool_document_tokens>{max_doc_length}"
        num_chunks += (count + max_doc_length - 1) // max_doc_length
    if doc_tokens > threshold:
        return False, f"tool_definition_tokens>{threshold}"
    if num_chunks > max_doc_num and not truncate:
        return False, f"tool_definition_docs>{max_doc_num}"
    return True, "ok"


class ToolTokenCounter:
    """Token counts exactly as the eval computes them: Qwen chat-template ids of the
    same serialized tool document(s), cached by content hash."""

    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        self._full_cache: Dict[str, int] = {}
        self._per_tool_cache: Dict[str, int] = {}

    def full_doc_tokens(self, tool_definition: str) -> int:
        key = hashlib.sha1(tool_definition.encode("utf-8")).hexdigest()
        if key not in self._full_cache:
            tool_doc = {"role": "user", "content": "Tool definitions:\n" + tool_definition}
            self._full_cache[key] = len(_chat_template_ids(self.tokenizer, [tool_doc]))
        return self._full_cache[key]

    def per_tool_doc_tokens(self, tools: Sequence[Dict[str, Any]]) -> List[int]:
        counts = []
        for tool in tools:
            rendered = _render_tool_definition([tool])
            key = hashlib.sha1(rendered.encode("utf-8")).hexdigest()
            if key not in self._per_tool_cache:
                tool_doc = {"role": "user", "content": "Tool definition:\n" + rendered}
                self._per_tool_cache[key] = len(_chat_template_ids(self.tokenizer, [tool_doc]))
            counts.append(self._per_tool_cache[key])
        return counts


def _empty_threshold_bucket() -> Dict[str, Any]:
    return {
        "sessions_with_tools_surviving": 0,
        "sessions_with_examples_surviving": 0,
        "spans_valid_surviving": 0,
        "spans_capped_surviving": 0,
        "skip_reasons": Counter(),
    }


def _new_subset_bucket(thresholds: Sequence[int]) -> Dict[str, Any]:
    return {
        "sessions_total": 0,
        "sessions_with_tools": 0,
        "sessions_with_examples": 0,
        "spans_raw": 0,
        "spans_valid": 0,
        "spans_capped": 0,
        "tools_per_session": [],
        "doc_tokens": [],
        "full": {str(threshold): _empty_threshold_bucket() for threshold in thresholds},
        "per_tool": {str(threshold): _empty_threshold_bucket() for threshold in thresholds},
    }


def _record_session(bucket: Dict[str, Any], session: Dict[str, Any], thresholds: Sequence[int], args: argparse.Namespace) -> None:
    bucket["sessions_total"] += 1
    bucket["spans_raw"] += session["raw_spans"]
    has_definition = session["doc_tokens"] is not None
    has_tools = bool(session["tools"])
    if has_tools:
        bucket["sessions_with_tools"] += 1
        bucket["tools_per_session"].append(len(session["tools"]))
    if has_definition:
        bucket["doc_tokens"].append(session["doc_tokens"])
    if session["n_valid"] > 0:
        bucket["sessions_with_examples"] += 1
        bucket["spans_valid"] += session["n_valid"]
        bucket["spans_capped"] += session["n_capped"]
    for mode in ("full", "per_tool"):
        # full mode never parses the tool list (any non-empty definition counts);
        # per_tool mode requires parseable tools (else "no_parseable_tools").
        if mode == "full" and not has_definition:
            continue
        if mode == "per_tool" and not has_tools:
            continue
        for threshold in thresholds:
            if mode == "full":
                survives, reason = _full_mode_outcome(
                    session["doc_tokens"], threshold, args.max_doc_length, args.max_doc_num, args.truncate_tool_definition
                )
            else:
                survives, reason = _per_tool_mode_outcome(
                    session["per_tool_doc_tokens"], threshold, args.max_doc_length, args.max_doc_num, args.truncate_tool_definition
                )
            slot = bucket[mode][str(threshold)]
            if survives:
                slot["sessions_with_tools_surviving"] += 1
                if session["n_valid"] > 0:
                    slot["sessions_with_examples_surviving"] += 1
                    slot["spans_valid_surviving"] += session["n_valid"]
                    slot["spans_capped_surviving"] += session["n_capped"]
            else:
                slot["skip_reasons"][reason] += 1


def _finalize_bucket(bucket: Dict[str, Any], subset: str, thresholds: Sequence[int]) -> Dict[str, Any]:
    out = {
        "subset": subset,
        "sessions_total": bucket["sessions_total"],
        "sessions_with_tools": bucket["sessions_with_tools"],
        "sessions_with_examples": bucket["sessions_with_examples"],
        "spans_raw": bucket["spans_raw"],
        "spans_valid": bucket["spans_valid"],
        "spans_capped": bucket["spans_capped"],
        "tools_per_session": _stat(bucket["tools_per_session"]),
        "tool_definition_doc_tokens": _stat(bucket["doc_tokens"]),
    }
    for mode in ("full", "per_tool"):
        out[mode] = {}
        for threshold in thresholds:
            slot = bucket[mode][str(threshold)]
            out[mode][str(threshold)] = {
                "sessions_with_tools_surviving": slot["sessions_with_tools_surviving"],
                "sessions_with_examples_surviving": slot["sessions_with_examples_surviving"],
                "spans_valid_surviving": slot["spans_valid_surviving"],
                "spans_capped_surviving": slot["spans_capped_surviving"],
                "skip_reasons": dict(slot["skip_reasons"]),
            }
    return out


def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    thresholds = [int(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one integer")

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        max_sessions=args.max_sessions,
        max_samples_per_session=args.max_samples_per_session,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=max(thresholds),
        truncate_tool_definition=args.truncate_tool_definition,
        require_tool_call=args.require_tool_call,
    )
    source = AgentLLMTracesSource(data_args)
    counter = ToolTokenCounter(args.tokenizer)

    sessions: List[Dict[str, Any]] = []
    for session in source.sessions:
        session_id = session["session_id"]
        subset = str(session.get("subset") or "unknown")
        spans = session["spans"]
        candidates = source._session_examples(session_id, spans, subset)
        n_valid = len(candidates)
        n_capped = min(n_valid, args.max_samples_per_session) if args.max_samples_per_session else n_valid
        tool_definition = candidates[0].tool_definition if candidates else ""
        if not tool_definition:
            for span in spans:
                tool_value = _span_attributes(span).get("gen_ai.tool.definitions")
                if tool_value:
                    tool_definition = _canonical_tool_definition(tool_value)
                    break
        tools = _as_tool_list(tool_definition) if tool_definition else []
        sessions.append(
            {
                "session_id": session_id,
                "subset": subset,
                "raw_spans": len(spans),
                "n_valid": n_valid,
                "n_capped": n_capped,
                "tools": tools,
                "doc_tokens": counter.full_doc_tokens(tool_definition) if tool_definition else None,
                "per_tool_doc_tokens": counter.per_tool_doc_tokens(tools) if tools else [],
                "toolset_key": _toolset_key(tools) if tools else None,
                "namespaces": _namespace_set(tools) if tools else set(),
            }
        )
    logger.info(
        "Tokenized %d unique full tool docs and %d unique single-tool docs",
        len(counter._full_cache),
        len(counter._per_tool_cache),
    )

    subsets: Dict[str, Dict[str, Any]] = defaultdict(lambda: _new_subset_bucket(thresholds))
    for session in sessions:
        _record_session(subsets[session["subset"]], session, thresholds, args)
    overall = _new_subset_bucket(thresholds)
    for session in sessions:
        _record_session(overall, session, thresholds, args)

    # Splits, mirroring diagnose_agent_tool_definition_generalization._build_sessions:
    # the split universe is sessions with >=1 valid example AND parseable tools.
    universe = [session for session in sessions if session["n_valid"] > 0 and session["tools"]]
    universe_dicts = [
        {"session_id": session["session_id"], "namespaces": session["namespaces"]}
        for session in universe
    ]
    session_train, session_eval = _session_split(universe_dicts, args.eval_ratio, args.split_seed)
    toolset_groups: Dict[str, set] = defaultdict(set)
    for session in universe:
        toolset_groups[session["toolset_key"]].add(session["session_id"])
    toolset_train, toolset_eval = _split_groups(toolset_groups, args.eval_ratio, args.split_seed)
    namespace_train, namespace_eval = _greedy_namespace_split(universe_dicts, args.eval_ratio, args.split_seed)

    split_reports: Dict[str, Any] = {}
    for split_name, eval_ids in (
        ("session_disjoint", session_eval),
        ("toolset_disjoint", toolset_eval),
        ("namespace_disjoint_proxy", namespace_eval),
    ):
        split_subsets: Dict[str, Dict[str, Any]] = defaultdict(lambda: _new_subset_bucket(thresholds))
        split_overall = _new_subset_bucket(thresholds)
        for session in sessions:
            if session["session_id"] not in eval_ids:
                continue
            _record_session(split_subsets[session["subset"]], session, thresholds, args)
            _record_session(split_overall, session, thresholds, args)
        split_reports[split_name] = {
            "eval_sessions": len(eval_ids),
            "subsets": [
                _finalize_bucket(bucket, subset, thresholds)
                for subset, bucket in sorted(split_subsets.items())
            ],
            "overall": _finalize_bucket(split_overall, "ALL", thresholds),
        }

    result = {
        "dataset_path": args.dataset_path,
        "tokenizer": args.tokenizer,
        "token_count_mode": "chat_template_ids(eval serialization)",
        "params": {
            "thresholds": thresholds,
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
            "max_context_tokens": args.max_doc_length * args.max_doc_num,
            "truncate_tool_definition": args.truncate_tool_definition,
            "max_samples_per_session": args.max_samples_per_session,
            "eval_ratio": args.eval_ratio,
            "split_seed": args.split_seed,
            "require_tool_call": args.require_tool_call,
        },
        "num_sessions": len(sessions),
        "split_universe_sessions": len(universe),
        "num_toolset_groups": len(toolset_groups),
        "source_skips": dict(source.source_skips),
        "subsets": [
            _finalize_bucket(bucket, subset, thresholds)
            for subset, bucket in sorted(subsets.items())
        ],
        "overall": _finalize_bucket(overall, "ALL", thresholds),
        "splits": split_reports,
    }

    if args.manifest_output:
        manifests = {
            "session_disjoint": {
                "train_session_ids": sorted(session_train),
                "eval_session_ids": sorted(session_eval),
            },
            "toolset_disjoint": {
                "train_session_ids": sorted(toolset_train),
                "eval_session_ids": sorted(toolset_eval),
            },
            "namespace_disjoint_proxy": {
                "train_session_ids": sorted(namespace_train),
                "eval_session_ids": sorted(namespace_eval),
            },
        }
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote split manifests to %s", manifest_path)
        result["manifest_output"] = str(manifest_path)
    return result


def _print_subset_table(title: str, rows: List[Dict[str, Any]], thresholds: Sequence[int], mode: str) -> None:
    print(f"## {title} (document_mode={mode})")
    header = f"| subset | sess | w/tools | w/ex | spans valid | capped | tools avg | tools max | tok p50 | tok max |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for threshold in thresholds:
        header += f" sess@{threshold // 1000}k | spans@{threshold // 1000}k |"
        sep += "---:|---:|"
    print(header)
    print(sep)
    for row in rows:
        tools = row["tools_per_session"]
        tokens = row["tool_definition_doc_tokens"]
        line = (
            f"| {row['subset']} | {row['sessions_total']} | {row['sessions_with_tools']} | "
            f"{row['sessions_with_examples']} | {row['spans_valid']} | {row['spans_capped']} | "
            f"{tools['avg']} | {tools['max']} | {tokens['p50']} | {tokens['max']} |"
        )
        for threshold in thresholds:
            slot = row[mode][str(threshold)]
            line += f" {slot['sessions_with_examples_surviving']} | {slot['spans_capped_surviving']} |"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only big-tool-pool threshold stats for agent-llm-traces: per-benchmark "
            "session/span survival under max_tool_definition_tokens thresholds, plus "
            "toolset_disjoint split effects. Read-only against the dataset."
        )
    )
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--tokenizer", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", default="./outputs/agent_llm_traces_bigpool_stats.json")
    parser.add_argument("--thresholds", default="10000,32000,48000")
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=64)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max_sessions", type=int)
    parser.add_argument(
        "--manifest_output",
        help=(
            "Optional path to also write session_disjoint/toolset_disjoint/namespace_disjoint_proxy "
            "split manifests (same format as diagnose_agent_tool_definition_generalization.py)."
        ),
    )
    args = parser.parse_args()
    thresholds = [int(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    result = inspect(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote stats to %s", output_path)

    for mode in ("full", "per_tool"):
        _print_subset_table(f"All sessions (no split)", result["subsets"] + [result["overall"]], thresholds, mode)
        _print_subset_table(
            "toolset_disjoint EVAL split",
            result["splits"]["toolset_disjoint"]["subsets"] + [result["splits"]["toolset_disjoint"]["overall"]],
            thresholds,
            mode,
        )


if __name__ == "__main__":
    main()
