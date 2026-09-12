#!/usr/bin/env python3
"""Verify all six frozen corpora and their controlled-comparison contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from next_compression.common import VARIANTS, SerializedCorpus, deserialize_memory


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(root, *, tokenizer=None):
    summaries, identities, memories = {}, {}, {}
    for variant in VARIANTS:
        corpus = SerializedCorpus(root / variant, tokenizer=tokenizer, expected_variant=variant)
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
                records[key] = digest([row["target_ids"], row["session_key"], row["source"]])
                inputs[key] = digest(row["memory"])
                ratios[row["ratio"]] += 1
                metadata = row.get("metadata", {})
                categories[metadata.get("decision_type", "unspecified")] += 1
                for name in ("post_tool", "state_revision", "history_dependency", "terminal_stop"):
                    coverage[name] += int(bool(metadata.get(name)))
                memory = deserialize_memory(row["memory"])
                costs = memory.costs(row["ratio"])
                for name in ("presented_encoder_tokens", "gist_tokens", "resident_kv_tokens"):
                    counters[name] += costs[name]
                counters["supervised_tokens"] += len(row["target_ids"])
                counters["gist_bearing_records"] += int(bool(memory.chunks))
        if ratios[8] != ratios[12] or any((decision, 12 if ratio == 8 else 8) not in records for decision, ratio in records):
            raise ValueError(f"Unpaired 8/12 examples in {variant}")
        if dict(counters) != corpus.manifest["counters"]:
            raise ValueError(f"Manifest token counters do not reproduce for {variant}")
        if variant == "H0":
            audit = corpus.manifest["audit"]
            expected = corpus.manifest["preparation"]["config"]["h0_decision_ids_count"]
            if (len(records) != 2 * expected or audit["accepted_distinct_decisions"] != expected
                    or audit["legacy_distinct_decisions_not_emitted"] != 0):
                raise ValueError("H0 did not preserve every frozen legacy decision")
            selected_ids_hash = digest(sorted({key[0] for key in records}))
            if selected_ids_hash != corpus.manifest["source_files"]["legacy_selected_ids_sha256"]:
                raise ValueError("H0 identities differ from the frozen legacy selection")
        identities[variant], memories[variant] = records, inputs
        summaries[variant] = dict(manifest_sha256=corpus.identity, records=len(corpus),
                                  distinct_decisions=len(corpus) // 2, ratios=dict(ratios),
                                  counters=dict(counters), decision_types=dict(categories), coverage=dict(coverage),
                                  source_counts=corpus.manifest["source_counts"],
                                  uncapped_two_epoch_steps_single_gpu_effective_batch32=2 * math.ceil(len(corpus) / 32))
    for group in (("H1", "H2", "H3"), ("T0", "T1")):
        for variant in group[1:]:
            if identities[group[0]] != identities[variant]:
                raise ValueError(f"Matched decision/target/source contract broken: {group}")
    if memories["H2"] != memories["H3"]:
        raise ValueError("H2/H3 must differ only in target weighting")
    if not summaries["H3"]["coverage"]["weighted_target_tokens"]:
        raise ValueError("H3 has no actual weighted supervision")
    return dict(schema="next-compression-delivery-verification-v1", status="verified", variants=summaries,
                comparisons=dict(history="H1/H2/H3 matched decisions and complete targets; H2/H3 identical inputs",
                                 tool="T0/T1 matched decisions and complete targets"),
                h100_validation="not_run_no_host")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = verify(Path(args.data_root), tokenizer=tokenizer)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
