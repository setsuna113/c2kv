#!/usr/bin/env python3
"""Verify all six frozen corpora and their controlled-comparison contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from next_compression.common import VARIANTS, SerializedCorpus, deserialize_memory


_WORKER_TOKENIZER = None


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_history_type_balance(categories, batch_sessions):
    type_counts = [categories.get(kind, 0) for kind in ("tool_call", "non_tool_response", "terminal_stop")]
    if min(type_counts) <= 0 or max(type_counts) - min(type_counts) > 2 * batch_sessions:
        raise ValueError("History action types are unbalanced beyond the final partial batch")


def _contract_digest(entries):
    return digest(sorted(entries))


def _verify_variant(root, variant, tokenizer):
    corpus = SerializedCorpus(
        root / variant, tokenizer=tokenizer, expected_variant=variant
    )
    records, inputs = {}, {}
    ratios, categories, coverage = Counter(), Counter(), Counter()
    counters = Counter()
    with corpus.records_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["decision_id"], row["ratio"])
            if key in records:
                raise ValueError(f"Duplicate decision/ratio in {variant}")
            if row["weight"] != 1.0:
                raise ValueError("This round fixes decision weights to one")
            weights = row.get("target_weights")
            if variant == "H3":
                if weights is None:
                    raise ValueError("H3 requires explicit positive full-target weights")
                coverage["weighted_target_tokens"] += sum(weight > 1 for weight in weights)
            elif weights is not None and any(weight != 1 for weight in weights):
                raise ValueError("Only H3 may change token supervision weights")
            records[key] = digest(
                [row["target_ids"], row["session_key"], row["source"]]
            )
            inputs[key] = digest(row["memory"])
            ratios[row["ratio"]] += 1
            metadata = row.get("metadata", {})
            kind = metadata.get("decision_type")
            if kind is None:
                kind = (
                    "tool_call"
                    if metadata.get("target_kind") == "covered_tool_call"
                    else "terminal_stop"
                    if metadata.get("source_grounded_terminal_stop")
                    else "non_tool_response"
                )
            categories[kind] += 1
            coverage["terminal_stop"] += int(kind == "terminal_stop")
            for name in ("post_tool", "state_revision", "history_dependency"):
                if name in metadata:
                    coverage[name] += int(bool(metadata[name]))
            memory = deserialize_memory(row["memory"])
            costs = memory.costs(row["ratio"])
            for name in (
                "presented_encoder_tokens",
                "gist_tokens",
                "resident_kv_tokens",
            ):
                counters[name] += costs[name]
            counters["supervised_tokens"] += len(row["target_ids"])
            counters["gist_bearing_records"] += int(bool(memory.chunks))
    if ratios[8] != ratios[12] or any(
        (decision, 12 if ratio == 8 else 8) not in records
        for decision, ratio in records
    ):
        raise ValueError(f"Unpaired 8/12 examples in {variant}")
    if dict(counters) != corpus.manifest["counters"]:
        raise ValueError(f"Manifest token counters do not reproduce for {variant}")
    if variant in {"H1", "H2", "H3"}:
        check_history_type_balance(
            categories, corpus.manifest["preparation"]["batch_sessions"]
        )
    if variant == "H0":
        audit = corpus.manifest["audit"]
        expected = corpus.manifest["preparation"]["config"][
            "h0_decision_ids_count"
        ]
        if (
            len(records) != 2 * expected
            or audit["accepted_distinct_decisions"] != expected
            or audit["legacy_distinct_decisions_not_emitted"] != 0
        ):
            raise ValueError("H0 did not preserve every frozen legacy decision")
        selected_ids_hash = digest(sorted({key[0] for key in records}))
        if (
            selected_ids_hash
            != corpus.manifest["source_files"]["legacy_selected_ids_sha256"]
        ):
            raise ValueError("H0 identities differ from the frozen legacy selection")
    summary = dict(
        manifest_sha256=corpus.identity,
        records=len(corpus),
        distinct_decisions=len(corpus) // 2,
        ratios=dict(ratios),
        counters=dict(counters),
        decision_types=dict(categories),
        coverage=dict(coverage),
        source_counts=corpus.manifest["source_counts"],
        uncapped_two_epoch_steps_single_gpu_effective_batch32=2
        * math.ceil(len(corpus) / 32),
    )
    contract = {
        "ids": _contract_digest([[decision, ratio] for decision, ratio in records]),
        "targets": _contract_digest(
            [
                [decision, ratio, target_digest]
                for (decision, ratio), target_digest in records.items()
            ]
        ),
        "inputs": _contract_digest(
            [
                [decision, ratio, input_digest]
                for (decision, ratio), input_digest in inputs.items()
            ]
        ),
    }
    return summary, contract


def _assemble_result(results):
    summaries = {variant: results[variant][0] for variant in VARIANTS}
    contracts = {variant: results[variant][1] for variant in VARIANTS}
    for group in (("H1", "H2", "H3"), ("T0", "T1")):
        for variant in group[1:]:
            if (
                contracts[group[0]]["ids"] != contracts[variant]["ids"]
                or contracts[group[0]]["targets"]
                != contracts[variant]["targets"]
            ):
                raise ValueError(f"Matched decision/target/source contract broken: {group}")
    if contracts["H2"]["inputs"] != contracts["H3"]["inputs"]:
        raise ValueError("H2/H3 must differ only in target weighting")
    if not summaries["H3"]["coverage"]["weighted_target_tokens"]:
        raise ValueError("H3 has no actual weighted supervision")
    return dict(schema="next-compression-delivery-verification-v1", status="verified", variants=summaries,
                comparisons=dict(history="H1/H2/H3 matched decisions and complete targets; H2/H3 identical inputs",
                                 tool="T0/T1 matched decisions and complete targets"),
                h100_validation="not_run_no_host")


def verify(root, *, tokenizer=None):
    """Verify serially, preserving the tokenizer-object API used by tests."""
    root = Path(root)
    return _assemble_result(
        {
            variant: _verify_variant(root, variant, tokenizer)
            for variant in VARIANTS
        }
    )


def _initialize_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    if tokenizer_path is None:
        _WORKER_TOKENIZER = None
        return
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )


def _verify_variant_worker(item):
    root, variant = item
    return variant, _verify_variant(Path(root), variant, _WORKER_TOKENIZER)


def verify_parallel(root, *, tokenizer_path=None, workers=6):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(root).resolve()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(VARIANTS)),
        initializer=_initialize_worker,
        initargs=(tokenizer_path,),
    ) as executor:
        pairs = executor.map(
            _verify_variant_worker,
            ((str(root), variant) for variant in VARIANTS),
            chunksize=1,
        )
        results = dict(pairs)
    return _assemble_result(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1:
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, local_files_only=True
            )
        result = verify(Path(args.data_root), tokenizer=tokenizer)
    else:
        result = verify_parallel(
            Path(args.data_root),
            tokenizer_path=args.tokenizer,
            workers=args.workers,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
