"""Build a task-proxy-grouped split manifest for agent-llm-traces.

agent-llm-traces parquet rows carry no underlying benchmark task id: the same
underlying task may be re-run by different models/harnesses as different
sessions. This script therefore groups sessions by a task proxy -- the sha1 of
the normalized first-user-message instruction of the session (first user
message of the first span's ``gen_ai.input.messages``; normalized by
lowercasing, collapsing whitespace, and stripping) -- and splits by GROUP,
never by session, with the deterministic hash assignment style of
``train_unified_next_action_c2kv.py:_split_for_group``.

Three levels are recorded per session:
  1. subset (the ``benchmark`` field);
  2. task-proxy group id (sha1 of the normalized first user instruction);
  3. toolset hash (same semantics as
     ``diagnose_agent_tool_definition_generalization._toolset_key``: sha1 of
     the sorted tool signatures {name, parameter (name, type) pairs,
     required}).

With ``--toolset_disjoint``, task-proxy groups that share a toolset hash are
merged into super-groups (union-find) before splitting, so no toolset
straddles train/eval. Sessions without tool definitions never merge.

Output schema is compatible with the manifest consumer in
``python/train/train_data_multiturn.py`` ({split_name: {train_session_ids,
eval_session_ids}}). A hard error is raised if any group straddles the split.

Usage:
  python agent/build_joint_split_manifest.py \
      --dataset_path ./datasets/agent-llm-traces \
      --out ./outputs/taskproxy_disjoint_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pyarrow.parquet as pq


def _find_parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    roots = [path / "data", path]
    files: List[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.parquet"))
            if not files:
                files = sorted(root.rglob("*.parquet"))
        if files:
            break
    return files


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _span_attributes(span: Any) -> Dict[str, Any]:
    span = _json_loads(span, span)
    if not isinstance(span, dict):
        return {}
    attributes = span.get("attributes", span)
    attributes = _json_loads(attributes, attributes)
    return attributes if isinstance(attributes, dict) else {}


def _sort_spans(spans: Any) -> List[Dict[str, Any]]:
    spans = _json_loads(spans, spans)
    if not isinstance(spans, list):
        return []
    return sorted(
        [span for span in spans if isinstance(span, dict)],
        key=lambda span: (
            span.get("start_time") or "",
            span.get("span_id") or "",
        ),
    )


def _as_tool_list(tool_definition: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(tool_definition, [])
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _schema_obj(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = (
        function.get("parameters")
        or tool.get("parameters")
        or tool.get("input_schema")
        or tool.get("schema")
        or {}
    )
    return schema if isinstance(schema, dict) else {}


def _parameter_signature(schema: Dict[str, Any]) -> tuple[tuple[str, str], ...]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out = []
    for name, value in properties.items():
        if isinstance(value, dict):
            typ = value.get("type") or value.get("anyOf") or value.get("oneOf") or value.get("items") or "unknown"
        else:
            typ = "unknown"
        out.append((str(name), json.dumps(typ, ensure_ascii=False, sort_keys=True)))
    return tuple(sorted(out))


def _tool_signature(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = _schema_obj(tool)
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    return {
        "name": _tool_name(tool),
        "parameters": _parameter_signature(schema),
        "required": tuple(sorted(str(item) for item in required)),
    }


def _hash_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _toolset_key(tools: Sequence[Dict[str, Any]]) -> str:
    signatures = sorted((_tool_signature(tool) for tool in tools), key=lambda item: item["name"])
    return _hash_json(signatures)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        # OpenTelemetry gen_ai shape: {"role": ..., "parts": [{"type": "text", "content": ...}]}
        content = message.get("parts")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _first_user_instruction(spans: Any) -> Optional[str]:
    ordered = _sort_spans(spans)
    for span in ordered:
        messages = _json_loads(_span_attributes(span).get("gen_ai.input.messages"), [])
        if isinstance(messages, dict):
            messages = [messages]
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, dict) and (message.get("role") or message.get("type")) == "user":
                text = _message_text(message)
                if text.strip():
                    return text
    return None


def _session_toolset_key(spans: Any) -> Optional[str]:
    for span in _sort_spans(spans):
        tools = _as_tool_list(_span_attributes(span).get("gen_ai.tool.definitions"))
        if tools:
            return _toolset_key(tools)
    return None


def _instruction_group_id(instruction: str) -> str:
    return hashlib.sha1(_normalize_text(instruction).encode("utf-8")).hexdigest()


def _is_eval_group(group_id: str, seed: int, eval_ratio: float) -> bool:
    value = int(hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).hexdigest(), 16) / (2**160)
    return value < eval_ratio


def _iter_rows(data_files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    wanted = ["benchmark", "subset", "dataset", "task", "session_id", "trace_id", "id", "spans"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        available = set(pf.schema_arrow.names)
        columns = [column for column in wanted if column in available]
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def _load_sessions(data_files: Sequence[Path]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    sessions: List[Dict[str, Any]] = []
    stats = {
        "duplicate_session_ids": 0,
        "sessions_without_instruction": 0,
        "sessions_without_tools": 0,
    }
    seen_session_ids: Set[str] = set()
    for row_index, row in enumerate(_iter_rows(data_files)):
        session_id = str(
            row.get("session_id")
            or row.get("trace_id")
            or row.get("id")
            or f"row-{row_index}"
        )
        if session_id in seen_session_ids:
            stats["duplicate_session_ids"] += 1
        seen_session_ids.add(session_id)
        subset = str(row.get("benchmark") or row.get("subset") or row.get("dataset") or row.get("task") or "unknown")
        instruction = _first_user_instruction(row.get("spans"))
        if instruction is None:
            stats["sessions_without_instruction"] += 1
            group_id = f"__no_instruction__:{session_id}"
        else:
            group_id = _instruction_group_id(instruction)
        toolset_key = _session_toolset_key(row.get("spans"))
        if toolset_key is None:
            stats["sessions_without_tools"] += 1
        sessions.append({
            "session_id": session_id,
            "subset": subset,
            "group_id": group_id,
            "toolset_key": toolset_key,
        })
    return sessions, stats


def _merge_groups_by_toolset(sessions: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Union-find merge of task-proxy groups that share a toolset hash.

    Returns {task-proxy group_id: super-group id}; groups whose sessions all
    lack tool definitions keep their own group id (they never merge).
    """
    parent: Dict[str, str] = {}

    def find(group: str) -> str:
        root = group
        while parent[root] != root:
            root = parent[root]
        while parent[group] != root:
            parent[group], group = root, parent[group]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    toolset_to_groups: Dict[str, Set[str]] = defaultdict(set)
    group_ids: Set[str] = set()
    for session in sessions:
        group_ids.add(session["group_id"])
        parent.setdefault(session["group_id"], session["group_id"])
        if session["toolset_key"]:
            toolset_to_groups[session["toolset_key"]].add(session["group_id"])
    for groups in toolset_to_groups.values():
        ordered = sorted(groups)
        for other in ordered[1:]:
            union(ordered[0], other)

    components: Dict[str, List[str]] = defaultdict(list)
    for group_id in group_ids:
        components[find(group_id)].append(group_id)
    super_id: Dict[str, str] = {}
    for members in components.values():
        canonical = min(members)
        for member in members:
            super_id[member] = canonical
    return super_id


def _size_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    if size <= 50:
        return "11-50"
    return "51+"


def _group_size_histogram(groups: Dict[str, Set[str]]) -> Dict[str, int]:
    histogram: Dict[str, int] = defaultdict(int)
    for members in groups.values():
        histogram[_size_bucket(len(members))] += 1
    ordered = ["1", "2", "3-5", "6-10", "11-50", "51+"]
    return {bucket: histogram[bucket] for bucket in ordered if histogram[bucket]}


def _assert_no_straddle(
    split_groups: Dict[str, Set[str]],
    train_ids: Set[str],
    eval_ids: Set[str],
) -> None:
    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(f"Train/eval session overlap detected: {sorted(overlap)[:5]}")
    straddling = [
        group_id
        for group_id, members in split_groups.items()
        if (members & train_ids) and (members & eval_ids)
    ]
    if straddling:
        raise RuntimeError(
            f"Split groups straddle train/eval (grouping is not split-safe): "
            f"{sorted(straddling)[:5]} ({len(straddling)} groups)"
        )


def _assert_no_toolset_straddle(
    sessions: Sequence[Dict[str, Any]],
    train_ids: Set[str],
    eval_ids: Set[str],
) -> None:
    sides: Dict[str, Set[str]] = defaultdict(set)
    for session in sessions:
        toolset_key = session["toolset_key"]
        if not toolset_key:
            continue
        if session["session_id"] in train_ids:
            sides[toolset_key].add("train")
        elif session["session_id"] in eval_ids:
            sides[toolset_key].add("eval")
    straddling = sorted(key for key, value in sides.items() if len(value) > 1)
    if straddling:
        raise RuntimeError(
            f"Toolsets straddle train/eval despite --toolset_disjoint: "
            f"{straddling[:5]} ({len(straddling)} toolsets)"
        )


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    if not 0.0 <= args.eval_ratio <= 1.0:
        raise ValueError(f"--eval_ratio must be in [0, 1], got {args.eval_ratio}")
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")

    sessions, load_stats = _load_sessions(data_files)
    if not sessions:
        raise RuntimeError(f"No sessions found under {args.dataset_path}")

    groups: Dict[str, Set[str]] = defaultdict(set)
    for session in sessions:
        groups[session["group_id"]].add(session["session_id"])

    if args.toolset_disjoint:
        super_id = _merge_groups_by_toolset(sessions)
    else:
        super_id = {group_id: group_id for group_id in groups}
    split_groups: Dict[str, Set[str]] = defaultdict(set)
    for session in sessions:
        split_groups[super_id[session["group_id"]]].add(session["session_id"])

    train_ids: Set[str] = set()
    eval_ids: Set[str] = set()
    for group_id in sorted(split_groups):
        target = eval_ids if _is_eval_group(group_id, args.seed, args.eval_ratio) else train_ids
        target.update(split_groups[group_id])

    _assert_no_straddle(split_groups, train_ids, eval_ids)
    if args.toolset_disjoint:
        _assert_no_toolset_straddle(sessions, train_ids, eval_ids)

    multi_session_groups = {key: members for key, members in groups.items() if len(members) > 1}
    subset_sessions: Dict[str, Set[str]] = defaultdict(set)
    subset_groups: Dict[str, Set[str]] = defaultdict(set)
    for session in sessions:
        subset_sessions[session["subset"]].add(session["session_id"])
        subset_groups[session["subset"]].add(session["group_id"])
    subset_counts = {
        subset: {
            "sessions": len(session_ids),
            "groups": len(subset_groups[subset]),
            "train_sessions": len(session_ids & train_ids),
            "eval_sessions": len(session_ids & eval_ids),
        }
        for subset, session_ids in sorted(subset_sessions.items())
    }

    split_name = args.split_name
    return {
        split_name: {
            "train_session_ids": sorted(train_ids),
            "eval_session_ids": sorted(eval_ids),
            "session_subsets": {session["session_id"]: session["subset"] for session in sessions},
            "session_groups": {session["session_id"]: session["group_id"] for session in sessions},
            "session_toolsets": {
                session["session_id"]: session["toolset_key"] or "" for session in sessions
            },
            "subset_counts": subset_counts,
        },
        "metadata": {
            "dataset_path": args.dataset_path,
            "num_parquet_files": len(data_files),
            "split_name": split_name,
            "seed": args.seed,
            "eval_ratio": args.eval_ratio,
            "toolset_disjoint": bool(args.toolset_disjoint),
            "num_sessions": len(sessions),
            "num_groups": len(groups),
            "num_split_groups": len(split_groups),
            "num_train_sessions": len(train_ids),
            "num_eval_sessions": len(eval_ids),
            "group_size_histogram": _group_size_histogram(groups),
            "collision_stats": {
                "groups_with_multiple_sessions": len(multi_session_groups),
                "sessions_in_multi_session_groups": sum(len(m) for m in multi_session_groups.values()),
            },
            "duplicate_session_ids": load_stats["duplicate_session_ids"],
            "sessions_without_instruction": load_stats["sessions_without_instruction"],
            "sessions_without_tools": load_stats["sessions_without_tools"],
            "subsets": sorted(subset_sessions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a task-proxy-grouped (instruction/toolset aware) split manifest for agent-llm-traces."
    )
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--out", default="./outputs/agent_taskproxy_split_manifest.json")
    parser.add_argument("--split_name", default="taskproxy_disjoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument(
        "--toolset_disjoint",
        action="store_true",
        help="Merge task-proxy groups sharing a toolset hash into super-groups before splitting.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = manifest["metadata"]
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print()
    print("| subset | groups | train sessions | eval sessions |")
    print("|---|---:|---:|---:|")
    for subset, item in manifest[args.split_name]["subset_counts"].items():
        print(f"| {subset} | {item['groups']} | {item['train_sessions']} | {item['eval_sessions']} |")


if __name__ == "__main__":
    main()
