#!/usr/bin/env python3
"""Prepare immutable 8/12 history corpora on CPU from frozen train sources."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from history_memory.preparation import tokenizer_identity
from next_compression.common import CorpusWriter, deserialize_memory, sha256_file
from next_compression.history import HistoryPreparationConfig, iter_history_records


def checked_file(root, name, info):
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError("Source must be directly inside its frozen directory")
    if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
        raise ValueError(f"Frozen source hash/size mismatch: {path.name}")
    return path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("Only frozen train sessions are accepted")
            yield row


def round_robin(iterables):
    active = list(map(iter, iterables))
    while active:
        following = []
        for iterator in active:
            try:
                yield next(iterator)
                following.append(iterator)
            except StopIteration:
                pass
        active = following


def atomic_records(iterator, variants):
    for decision_id, group in itertools.groupby(iterator, key=lambda item: item[1]["decision_id"]):
        records = list(group)
        expected = {(variant, ratio) for variant in variants for ratio in (8, 12)}
        actual = {(variant, record["ratio"]) for variant, record in records}
        if actual != expected or len(records) != len(expected):
            raise ValueError(f"Incomplete matched decision: {decision_id}")
        targets = {tuple(record["target_ids"]) for _, record in records}
        if len(targets) != 1:
            raise ValueError("Matched variants must supervise the same complete continuation")
        yield records


def init_worker(tokenizer_path, config):
    from transformers import AutoTokenizer
    global WORKER_TOKENIZER, WORKER_CONFIG
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    WORKER_CONFIG = replace(config, workers=1)


def prepare_batch(batch):
    iterator, audit = iter_history_records(batch, WORKER_TOKENIZER, WORKER_CONFIG)
    records = list(iterator)
    return records, dict(audit)


def packed_batches(rows, tokenizer, config, args):
    batches = iter(lambda: list(itertools.islice(rows, args.batch_sessions)), [])
    if args.workers == 1:
        for batch in batches:
            stream, audit = iter_history_records(batch, tokenizer, replace(config, workers=1))
            records = list(stream)
            yield records, audit
        return
    executor = ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                   initargs=(args.tokenizer, config))
    pending = deque()
    try:
        for batch in itertools.islice(batches, args.workers):
            pending.append(executor.submit(prepare_batch, batch))
        while pending:
            yield pending.popleft().result()
            batch = next(batches, None)
            if batch is not None:
                pending.append(executor.submit(prepare_batch, batch))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def prepare_group(rows, tokenizer, config, destination, source_files, args):
    writers = {variant: CorpusWriter(destination / variant, variant) for variant in config.variants}
    reference = config.variants[0]
    audit = Counter()
    emitted_ids = set()
    exhausted = False
    batches = packed_batches(iter(rows), tokenizer, config, args)
    for records, batch_audit in batches:
        for group in atomic_records(iter(records), config.variants):
            decision_id = group[0][1]["decision_id"]
            if decision_id in emitted_ids:
                raise ValueError("Repeated decision identity across source sessions")
            cost = sum(deserialize_memory(record["memory"]).costs(record["ratio"])["presented_encoder_tokens"]
                       for variant, record in group if variant == reference)
            current = writers[reference].counters["presented_encoder_tokens"]
            if current + cost > args.max_presented_tokens:
                audit["budget_first_nonfitting_decision"] += 1
                exhausted = True
                break
            for variant, record in group:
                writers[variant].write(record)
            emitted_ids.add(decision_id)
            if len(emitted_ids) >= args.max_decisions:
                audit["decision_limit_reached"] += 1
                exhausted = True
                break
        audit.update(batch_audit)
        audit["batches"] += 1
        print(json.dumps({"event": "history_batch", "group": reference, "decisions": len(emitted_ids),
                          "encoder_tokens": writers[reference].counters["presented_encoder_tokens"]}), flush=True)
        if exhausted:
            break
    batches.close()
    resolved = asdict(config)
    for key in ("h0_decision_ids", "h1_decision_ids"):
        ids = resolved.pop(key)
        resolved[key + "_count"] = None if ids is None else len(ids)
    audit["accepted_distinct_decisions"] = len(emitted_ids)
    if reference == "H0":
        audit["legacy_distinct_decisions_not_emitted"] = len(config.h0_decision_ids - emitted_ids)
    for variant, writer in writers.items():
        preparation = dict(config=resolved, batch_sessions=args.batch_sessions, cpu_processes=args.workers,
                           selection_profile="legacy-frozen-decision-ids" if variant == "H0" else "source-round-robin-batch-action-balance-v1",
                           budget_reference_variant=reference, max_presented_tokens=args.max_presented_tokens,
                           max_distinct_decisions=args.max_decisions,
                           render_profile="event-native-evidence-v1" if variant in {"H0", "H1"} else "a-event-native-s0-v1",
                           loss_profile="decision-mean-critical-token-weighted-ce-v1" if variant == "H3" else "decision-mean-complete-ce-v1")
        manifest = writer.finish(tokenizer=tokenizer_identity(tokenizer), preparation=preparation,
                                 source_files=source_files, audit=dict(audit))
        print(json.dumps({"event": "history_variant_complete", "variant": variant,
                          "records": manifest["records"], "counters": manifest["counters"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Normalized source directory with manifest.json")
    parser.add_argument("--legacy-prepared", required=True, help="Frozen prior B/C prepared directory")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", choices=("all", "reference", "trio"), default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-sessions", type=int, default=64)
    parser.add_argument("--max-presented-tokens", type=int, default=48000000)
    parser.add_argument("--max-decisions", type=int, default=100000)
    args = parser.parse_args(argv)
    if min(args.workers, args.batch_sessions, args.max_presented_tokens, args.max_decisions) <= 0:
        parser.error("Preparation limits must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    destination = Path(args.output_dir)
    sources, legacy = Path(args.sources), Path(args.legacy_prepared)
    if args.group in {"all", "reference"}:
        manifest_path = legacy / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tokenizer_identity(tokenizer)["sha256"] != manifest["tokenizer"]["sha256"]:
            raise ValueError("H0 must retain the prior tokenizer/template")
        verified = {name: checked_file(legacy, name, info) for name, info in manifest["file_integrity"].items()}
        legacy_costs = {}
        with verified["paired_decisions.jsonl"].open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                cost = row["arms"]["C"]["costs"]["presented_encoder_tokens"]
                if legacy_costs.setdefault(row["decision_id"], cost) != cost:
                    raise ValueError("Legacy reference encoder cost differs across ratios")
        ids = frozenset(legacy_costs)
        config = HistoryPreparationConfig(variants=("H0",), workers=args.workers, h0_decision_ids=ids)
        binding = dict(legacy_manifest_sha256=sha256_file(manifest_path), files=manifest["file_integrity"])
        binding["legacy_selected_ids_sha256"] = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
        reference_args = argparse.Namespace(**vars(args))
        reference_args.max_presented_tokens = 2 * sum(legacy_costs.values())
        binding["reference_pair_completion"] = "Every frozen decision receives both 8/12 ratios; prior unpaired final exposure is completed"
        prepare_group(read_rows(verified["sessions.jsonl"]), tokenizer, config, destination, binding, reference_args)
    if args.group in {"all", "trio"}:
        manifest_path = sources / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema"] != "next-compression-normalized-sources-v1":
            raise ValueError("Unexpected normalized source contract")
        paths = [checked_file(sources, info["path"], info) for _, info in sorted(manifest["sources"].items())]
        binding = dict(normalized_manifest_sha256=sha256_file(manifest_path), files=manifest["sources"])
        config = HistoryPreparationConfig(variants=("H1", "H2", "H3"), workers=args.workers)
        prepare_group(round_robin(read_rows(path) for path in paths), tokenizer, config, destination, binding, args)


if __name__ == "__main__":
    main()
