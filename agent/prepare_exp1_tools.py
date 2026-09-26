#!/usr/bin/env python3
"""Build the frozen Exp 1 tool-allocation manifest on CPU.

Sessions are drawn round-robin from the normalized tool sources, skipped when
any of the six training corpora (H0-H3, T0, T1) contains them, and one
decision per session is admitted until the per-type quotas are filled.  Every
admitted decision is packed in every gist layout at ratios 8 and 12 with the
training packer, so the evaluator never re-tokenizes.  The eviction layouts
reuse the ``full`` records and their per-tool token spans.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from history_memory.dataset import iter_decisions
from history_memory.packing import PackingBudgetError, pack_target
from history_memory.preparation import tokenizer_identity
from next_compression.common import sha256_file
from next_compression.exp1_tools import (
    EXP1_PURPOSE,
    EXP1_RATIOS,
    EXP1_SCHEMA,
    GIST_LAYOUTS,
    LayoutPlan,
    build_layout_records,
    decision_type,
    gold_tool_calls,
    validate_exp1_record,
)
from next_compression.tools import ToolPackingError, ToolPreparationConfig


def _selection_module():
    """Reuse the checkpoint-selection helpers without duplicating them."""
    path = Path(__file__).with_name("prepare_next_selection.py")
    name = "_exp1_prepare_next_selection"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _positive_quota(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        key, _, number = item.partition("=")
        key = key.strip()
        if key not in ("tool_call", "non_tool_response", "terminal_stop"):
            raise argparse.ArgumentTypeError(f"Unknown decision type: {key!r}")
        if not number.strip().isdigit() or int(number) < 0:
            raise argparse.ArgumentTypeError(f"Quota must be a nonnegative integer: {item!r}")
        result[key] = int(number)
    if sum(result.values()) <= 0:
        raise argparse.ArgumentTypeError("Quotas must admit at least one decision")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not items or any(item <= 0 for item in items) or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("expected distinct positive integers")
    return items


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-manifest", required=True, help="Normalized tool source manifest")
    parser.add_argument("--training-dir", required=True, help="Parent of prepared H0..T1")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", nargs="+", help="Explicit tool source keys in read order")
    parser.add_argument("--quotas", type=_positive_quota, required=True,
                        help="e.g. tool_call=160,non_tool_response=40,terminal_stop=40")
    parser.add_argument("--k", type=_int_list, default=(1, 3, 5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layouts", nargs="+", default=list(GIST_LAYOUTS), choices=GIST_LAYOUTS)
    parser.add_argument("--max-sessions-scanned", type=int, default=20_000)
    parser.add_argument("--max-raw-tokens", type=int, default=None,
                        help="Native prefix budget; default unbounded so 'full' packs every catalog")
    parser.add_argument("--max-sequence-tokens", type=int, default=None,
                        help="Override the training sequence cap (default: training value)")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--verify-existing", action="store_true",
                        help="Re-hash and validate an already written output")
    args = parser.parse_args(argv)
    if args.sources is not None and len(args.sources) != len(set(args.sources)):
        parser.error("--sources must not contain duplicates")
    if args.max_sessions_scanned <= 0 or args.progress_every <= 0:
        parser.error("--max-sessions-scanned and --progress-every must be positive")
    if "hybrid" not in args.layouts and "retrieval" in args.layouts:
        parser.error("retrieval requires hybrid for its allowance")
    return args


class Exp1Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=False)
        self.records_path = self.path / "records.jsonl"
        self.handle = self.records_path.open("wb")
        self.count = 0
        self.cell_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.decision_ids: list[str] = []

    def write_decision(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            validate_exp1_record(record)
        for record in records:
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            self.handle.write((payload + "\n").encode("utf-8"))
            self.count += 1
            self.cell_counts[f"{record['layout']}.k{record['k']}.ratio{record['ratio']}"] += 1
            self.source_counts[str(record["source"])] += 1
        self.decision_ids.append(records[0]["decision_id"])

    def finish(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self.handle.close()
        manifest = {
            **manifest,
            "records": {
                "path": self.records_path.name,
                "sha256": sha256_file(self.records_path),
                "bytes": self.records_path.stat().st_size,
                "count": self.count,
            },
            "cell_counts": dict(sorted(self.cell_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "decision_ids": list(self.decision_ids),
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest


def _code_hashes() -> dict[str, str]:
    root = REPOSITORY_ROOT
    files = [
        Path(__file__),
        root / "agent" / "prepare_next_selection.py",
        root / "python" / "next_compression" / "exp1_tools.py",
        root / "python" / "next_compression" / "tools.py",
        root / "python" / "next_compression" / "common.py",
        root / "python" / "history_memory" / "packing.py",
        root / "python" / "history_memory" / "dataset.py",
    ]
    return {str(path.relative_to(root)).replace(os.sep, "/"): sha256_file(path) for path in files}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    selection = _selection_module()
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preparation output is not empty: {output}")

    tool_manifest_path = Path(args.tool_manifest).resolve()
    tool_manifest = selection._load_normalized_manifest(tool_manifest_path)
    training = selection.load_training_inventory(args.training_dir)
    specs = selection._source_specs(tool_manifest_path, tool_manifest, args.sources)
    raw_config = selection._training_preparation_config(training, ("T0", "T1"))
    training_config = selection._exact_dataclass_config(raw_config, ToolPreparationConfig)
    overrides: dict[str, Any] = {"max_raw_tokens": args.max_raw_tokens}
    if args.max_sequence_tokens is not None:
        overrides["max_sequence_tokens"] = args.max_sequence_tokens
    config = dataclasses.replace(training_config, **overrides)
    plan = LayoutPlan(k_values=tuple(args.k), seed=args.seed, layouts=tuple(args.layouts))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_info = tokenizer_identity(tokenizer)
    for variant in ("T0", "T1"):
        expected = training.manifests[variant].get("tokenizer", {}).get("sha256")
        if expected != tokenizer_info["sha256"]:
            raise ValueError(f"Tokenizer differs from the {variant} training corpus")

    quotas = dict(args.quotas)
    counts: Counter[str] = Counter()
    audit: Counter[str] = Counter()
    selected_sessions: list[str] = []
    writer = Exp1Writer(output)
    expected_total = sum(quotas.values())
    try:
        for source_key, row in selection._round_robin_rows(specs):
            if audit["sessions.scanned"] >= args.max_sessions_scanned:
                break
            audit["sessions.scanned"] += 1
            session_key = selection._row_identity(row)
            if session_key in training.all_sessions:
                audit["sessions.skipped_training_overlap"] += 1
                continue
            last_index = len(row["messages"]) - 1
            admitted = False
            for decision in iter_decisions(row):
                kind = decision_type(decision, last_source_index=last_index)
                if counts[kind] >= quotas.get(kind, 0):
                    audit[f"decisions.skipped.quota_full.{kind}"] += 1
                    continue
                if not decision.tools:
                    audit["decisions.skipped.no_tool_definitions"] += 1
                    continue
                try:
                    target_ids = pack_target(
                        tokenizer, decision.target, max_target_tokens=config.max_target_tokens
                    )
                except PackingBudgetError:
                    audit["decisions.skipped.target_tokens_over_limit"] += 1
                    continue
                base_metadata = {
                    "purpose": EXP1_PURPOSE,
                    "decision_type": kind,
                    "decision_index": decision.decision_index,
                    "source_message_index": decision.source_message_index,
                    "task_id": decision.task_id,
                    "template_id": decision.template_id,
                    "source_session_id": decision.session_id,
                    "gold_tool_calls": gold_tool_calls(decision),
                }
                try:
                    records = build_layout_records(
                        decision,
                        tokenizer,
                        config=config,
                        plan=plan,
                        session_key=session_key,
                        source=str(row["source"]),
                        target_ids=target_ids,
                        base_metadata=base_metadata,
                    )
                except (ToolPackingError, PackingBudgetError) as error:
                    reason = getattr(error, "reason", None) or type(error).__name__
                    audit[f"decisions.skipped.pack.{reason}"] += 1
                    continue
                over = [
                    record for record in records
                    if record["metadata"]["token_counts"]["resident_kv_tokens"] + len(target_ids)
                    > config.max_sequence_tokens
                ]
                if over:
                    audit["decisions.skipped.sequence_tokens_over_limit"] += 1
                    continue
                writer.write_decision(records)
                counts[kind] += 1
                selected_sessions.append(session_key)
                admitted = True
                audit["decisions.selected"] += 1
                audit[f"decisions.selected.type.{kind}"] += 1
                audit[f"decisions.selected.source.{row['source']}"] += 1
                if audit["decisions.selected"] % args.progress_every == 0:
                    print(json.dumps({
                        "event": "exp1_selection_progress",
                        "sessions_scanned": audit["sessions.scanned"],
                        "decisions": sum(counts.values()),
                        "type_counts": dict(counts),
                    }), flush=True)
                break  # one decision per session keeps sessions diverse
            if not admitted:
                audit["sessions.skipped_no_admissible_decision"] += 1
            if sum(counts.values()) == expected_total:
                break
    finally:
        writer.handle.close()
    if dict(counts) != {key: value for key, value in quotas.items() if value}:
        raise RuntimeError(
            "Exp 1 selection could not fill quotas within the bounded scan: "
            f"filled={dict(counts)}, required={quotas}, "
            f"max_sessions_scanned={args.max_sessions_scanned}"
        )
    selection.assert_session_disjoint(selected_sessions, training.sessions_by_variant)
    manifest = writer.finish({
        "schema": EXP1_SCHEMA,
        "purpose": EXP1_PURPOSE,
        "split": EXP1_PURPOSE,
        "ratios": list(EXP1_RATIOS),
        "layouts": list(plan.layouts),
        "eviction_layouts_from_full": ["snapkv", "snapkv_hybrid", "h2o", "h2o_hybrid"],
        "plan": {"k_values": list(plan.k_values), "seed": plan.seed,
                 "ranker_scope": plan.ranker_scope},
        "quotas": quotas,
        "type_counts": dict(counts),
        "tokenizer": tokenizer_info,
        "preparation": {
            "config": dataclasses.asdict(config),
            "training_config": dataclasses.asdict(training_config),
            "render_profile": training.manifests["T0"]["render_profile"],
            "packing_profile": "next-compression-exp1-tools-v1",
        },
        "source_files": {
            spec.key: {"path": str(spec.path), "sha256": spec.manifest_entry["sha256"],
                       "bytes": spec.manifest_entry["bytes"]}
            for spec in specs
        },
        "training_session_disjointness": {
            "status": "verified",
            "training_manifest_sha256": dict(training.manifest_sha256),
            "training_records_sha256": dict(training.records_sha256),
            "training_session_union_count": len(training.all_sessions),
            "selected_session_ids": selected_sessions,
        },
        "audit": dict(sorted(audit.items())),
        "provenance": {
            "argv": list(sys.argv),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "tool_manifest_sha256": sha256_file(tool_manifest_path),
            "code_sha256": _code_hashes(),
        },
    })
    print(json.dumps({"event": "exp1_manifest_written", "path": str(output / "manifest.json"),
                      "records": manifest["records"]}), flush=True)
    return manifest


def verify_existing(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != EXP1_SCHEMA:
        raise ValueError("Not an Exp 1 manifest")
    records = root / manifest["records"]["path"]
    if sha256_file(records) != manifest["records"]["sha256"]:
        raise ValueError("Exp 1 records hash differs from manifest")
    count = 0
    with records.open("r", encoding="utf-8") as handle:
        for line in handle:
            validate_exp1_record(json.loads(line))
            count += 1
    if count != manifest["records"]["count"]:
        raise ValueError("Exp 1 record count differs from manifest")
    digest = hashlib.sha256(records.read_bytes()).hexdigest()
    return {"status": "verified", "records": count, "records_sha256": digest}


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.verify_existing:
        print(json.dumps(verify_existing(args.output_dir)), flush=True)
        return
    prepare(args)


if __name__ == "__main__":
    main()
