"""Token accounting for C2KV training manifests: U_src / P_src / T_tgt.

Experiments are budgeted in TOKENS, not steps.  This module computes the
three budget metrics for any training manifest, and can also estimate
``P_official`` — the presented source tokens of the OFFICIAL two-stage
training recipe (mdoc mixture, then the open_swe agent stage) — so joint
runs can be budgeted against it.

Metrics
-------
- ``U_src`` (unique compressible source tokens): every document that feeds
  gist compression (joint mode: ``tool_documents + history_documents`` of
  each example; official mdoc stage: the sample's ``documents``; official
  agent stage: the history-message contents) is whitespace-normalized,
  sha1-hashed, and counted ONCE per unique hash with
  ``tokenizer.encode(text, add_special_tokens=False)``.  U_src is a
  pool-level metric: it includes documents later dropped by grid budget
  caps (``max_doc_num`` / ``max_tool_chunks`` / history selection) and by
  preprocessing filters, and it is epoch-invariant — repeating the pool for
  more epochs does not change it.
- ``P_src`` (presented non-padding source tokens): examples are run through
  the REAL preprocessing (``JointDataset.preprocess_example`` for joint
  manifests; ``MultiDocDataset._preprocess_mdoc_sample`` and
  ``CompressHistoryDataset.preprocess_example`` for the official stages)
  and ``context_input_ids != -100`` is summed per emitted row.  This counts
  the chat-template wrapper tokens around each compressed document as well,
  because those tokens are presented to (and compressed by) the gist path.
  ``--epochs`` multiplies P_src only (U_src stays epoch-invariant; T_tgt is
  reported per epoch).
- ``T_tgt`` (target tokens with loss): ``labels != -100`` summed per
  emitted row (answer + EOS).

Per-subset breakdowns dedup U_src WITHIN each subset; the totals dedup
globally, so ``total.U_src <= sum(per_subset.U_src)`` whenever a document
appears in more than one subset.

Official-mode approximations (what is counted, what is estimated)
-----------------------------------------------------------------
mdoc stage (scripts/train_qwen3-4b-mixed_mdoc.sh -> train.train_mdoc):

- Source paths are resolved with the OFFICIAL ``train_mdoc._candidate_roots``
  / ``train_mdoc._source_path`` under ``--data_root`` (the script's
  ``--train_data``); missing sources abort with an error listing every
  candidate path tried.  The three default sources resolve to cleaned
  datasets loaded with ``datasets.load_from_disk``, exactly like
  ``MultiDocDataset``.
- ``--mdoc_num_samples`` (default 32768 = 2**15, the per-source budget from
  the experiment plan) selects the FIRST N samples, mirroring
  ``MultiDocDataset.__init__``'s ``select(range(min(N, len)))``.  NOTE: the
  official script leaves ``--train_source_sizes`` unset, i.e.
  num_samples=None -> all samples AFTER the first 512 (the eval
  reservation).  Pass ``--mdoc_num_samples all`` to replicate that exactly.
- Each selected sample is preprocessed by calling
  ``MultiDocDataset._preprocess_mdoc_sample`` verbatim (max_length=1024,
  max_doc_length=1024, max_doc_num=10, max_system_length=256 — the
  MDocDataArgs defaults the official script does not override;
  hotpotqa/wikimqa get the identity extractor, i.e. the QA_QUERY_PROMPTS
  question prefix and the empty-document strip; longmagpie gets
  extract_docs=None), and rows with no supervised label are dropped exactly
  like MultiDocDataset's final filter.  The official
  ``.shuffle(seed=2948)`` is order-only and does not change token sums, so
  it is not reproduced.

agent stage (scripts/train_qwen3-4b-mixed_agent.sh -> train.train_compress_history):

- ``OpenSWETracesCompressHistorySource`` is constructed with the script's
  kwargs (resolved_only=True, recent_message_num=4,
  max_samples_per_trace=8), and the FIRST ``--agent_num_samples`` (default
  130000) expanded records are scanned, mirroring
  ``CompressHistoryDataset.__init__``'s
  ``select(range(min(num_samples, len)))``.
- Rows are produced by ``CompressHistoryDataset.preprocess_example`` with
  the script's args (max_doc_num=10, max_doc_length=1024,
  max_system_length=8192, max_length=16384, min_doc_num=2,
  history_selection="tail", full_history_doc_num=0,
  split_oversized_history_docs=True); records that the official map would
  mark ``dynamic == -1`` (invalid) are dropped here too.
- The official ``_expand_openswe_batch`` draws from the UNSEEDED global
  ``random`` module (``random.randint`` for the recent-message window,
  ``random.sample`` for the per-trace cap), so official trainer runs are
  themselves slightly non-deterministic.  This tool seeds
  ``random.seed(--split_seed)`` before expansion; use ``--num_proc 1`` for
  a fully deterministic scan (multi-proc map workers carry their own RNG
  state).
- The compressible source of the agent stage = history-message contents
  only; the system prompt + first user request stay in the uncompressed
  prefix and are NOT counted in U_src (they are not compressed).

CLI
---
joint mode (any AgentLLMTracesJointSource manifest)::

    python -m train.token_accounting joint \
        --dataset_path ~/c2kv/datasets/agent-llm-traces \
        --split_manifest_file configs/split_manifest.json \
        --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
        --max_doc_num 10 --max_doc_length 1024 --epochs 3 \
        --out joint_tokens.json

official mode (estimate P_official of the two-stage recipe)::

    python -m train.token_accounting official \
        --data_root /mnt/nas1/duchuheng/datasets \
        --agent_data /mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces--train \
        --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
        --num_proc 32 --out official_tokens.json

``--tokenizer fake`` selects the deterministic whitespace tokenizer shipped
with train_data_joint (offline smoke tests only; the numbers are
meaningless for budgeting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets

from .train_data import DEFAULT_SYSTEM_PROMPT, MultiDocDataset
from .train_data_joint import AgentLLMTracesJointSource, JointDataset, JointExample
from .train_data_multiturn import (
    CompressHistoryDataset,
    CompressHistoryExample,
    OpenSWETracesCompressHistorySource,
    _json_loads,
)

logger = logging.getLogger(__name__)


U_SRC_NOTE = (
    "U_src is pool-level and epoch-invariant: unique compressible documents (sha1 over "
    "whitespace-normalized raw text) counted once each with "
    "tokenizer.encode(text, add_special_tokens=False); documents later dropped by grid budget "
    "caps or preprocessing filters are still counted."
)
P_SRC_NOTE = (
    "P_src counts context_input_ids != -100 over emitted rows (real preprocessing, including the "
    "chat-template wrapper around each compressed document); --epochs multiplies P_src only."
)
T_TGT_NOTE = "T_tgt counts labels != -100 over emitted rows (answer + EOS), per epoch."


# ---------------------------------------------------------------------------
# Accumulator.
# ---------------------------------------------------------------------------


class TokenAccumulator:
    """Running U_src/P_src/T_tgt over a stream of source docs and emitted rows.

    U_src dedups by sha1 of the whitespace-normalized document text; the
    per-hash token counts are kept so accumulators can be merged (per-subset
    -> total, per-source -> stage -> grand total) with global dedup.
    """

    def __init__(self) -> None:
        self._doc_tokens: Dict[str, int] = {}
        self.p_src = 0
        self.t_tgt = 0
        self.samples = 0

    def add_source_documents(self, docs: Sequence[Any], tokenizer) -> None:
        for doc in docs:
            normalized = " ".join(str(doc).split())
            if not normalized:
                continue
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            if digest in self._doc_tokens:
                continue
            self._doc_tokens[digest] = len(tokenizer.encode(normalized, add_special_tokens=False))

    def add_row(self, row: Dict[str, Any]) -> None:
        self.samples += 1
        self.p_src += sum(1 for token_id in row["context_input_ids"] if int(token_id) != -100)
        self.t_tgt += sum(1 for token_id in row["labels"] if int(token_id) != -100)

    def merge(self, other: "TokenAccumulator") -> None:
        for digest, count in other._doc_tokens.items():
            self._doc_tokens.setdefault(digest, count)
        self.p_src += other.p_src
        self.t_tgt += other.t_tgt
        self.samples += other.samples

    def to_dict(self, epochs: int = 1) -> Dict[str, Any]:
        return {
            "U_src": sum(self._doc_tokens.values()),
            "P_src": self.p_src * epochs,
            "P_src_per_epoch": self.p_src,
            "T_tgt": self.t_tgt,
            "samples": self.samples,
            "unique_docs": len(self._doc_tokens),
        }


# ---------------------------------------------------------------------------
# Mode A: joint manifest scan.
# ---------------------------------------------------------------------------


def scan_joint_examples(
    examples: Sequence[JointExample],
    tokenizer,
    *,
    max_length: int = 1024,
    max_doc_length: int = 1024,
    min_doc_num: int = 2,
    max_doc_num: int = 10,
    max_system_length: int = 2048,
    history_selection: str = "tail",
    doc_mode: str = "joint",
    max_tool_chunks: Optional[int] = None,
    max_tool_definition_tokens: int = 32000,
    split_oversized_history_docs: bool = True,
    per_side_caps: bool = True,
    min_target_tokens: Optional[int] = None,
    epochs: int = 1,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute U_src/P_src/T_tgt for a joint manifest.

    Every example contributes its ``tool_documents + history_documents`` to
    U_src (pool-level, deduped per subset and globally); only examples that
    survive ``JointDataset.preprocess_example`` contribute rows to
    P_src/T_tgt.  Deterministic given the example order.
    """
    per_subset: Dict[str, TokenAccumulator] = {}
    total = TokenAccumulator()
    skipped: Counter[str] = Counter()
    scanned = 0
    for example in examples:
        if max_examples is not None and scanned >= max_examples:
            break
        scanned += 1
        subset = str(getattr(example, "subset", "unknown") or "unknown")
        accumulator = per_subset.setdefault(subset, TokenAccumulator())
        documents = list(example.tool_documents) + list(example.history_documents)
        accumulator.add_source_documents(documents, tokenizer)
        total.add_source_documents(documents, tokenizer)
        row, reason = JointDataset.preprocess_example(
            example,
            tokenizer=tokenizer,
            max_length=max_length,
            max_doc_length=max_doc_length,
            min_doc_num=min_doc_num,
            max_doc_num=max_doc_num,
            max_system_length=max_system_length,
            history_selection=history_selection,
            doc_mode=doc_mode,
            max_tool_chunks=max_tool_chunks,
            max_tool_definition_tokens=max_tool_definition_tokens,
            split_oversized_history_docs=split_oversized_history_docs,
            per_side_caps=per_side_caps,
        )
        if row is not None and min_target_tokens is not None:
            # Mirror MinTargetJointDataset: the trainer drops rows whose
            # supervised answer fell below the reserved floor, so an
            # "as-trained" P_src scan must drop them too.
            answer_token_count = len(tokenizer.encode(example.answer, add_special_tokens=False)) + 1
            reserved = min(answer_token_count, max(1, min_target_tokens))
            supervised = sum(1 for value in row["labels"] if value != -100)
            if supervised < reserved:
                row, reason = None, f"target_tokens<{reserved}"
        if row is None:
            skipped[reason] += 1
            continue
        accumulator.add_row(row)
        total.add_row(row)
    return {
        "examples_scanned": scanned,
        "epochs": epochs,
        "doc_mode": doc_mode,
        "total": total.to_dict(epochs),
        "per_subset": {subset: per_subset[subset].to_dict(epochs) for subset in sorted(per_subset)},
        "skipped_rows": dict(sorted(skipped.items())),
        "notes": [U_SRC_NOTE, P_SRC_NOTE, T_TGT_NOTE],
    }


# ---------------------------------------------------------------------------
# Mode B: official two-stage recipe scan.
# ---------------------------------------------------------------------------

OFFICIAL_MDOC_SOURCES = ("hotpotqa", "wikimqa", "longmagpie")

# MDocDataArgs defaults that scripts/train_qwen3-4b-mixed_mdoc.sh does not override.
MDOC_STAGE_KWARGS = {
    "max_length": 1024,
    "max_doc_length": 1024,
    "max_doc_num": 10,
    "max_system_length": 256,
}

# scripts/train_qwen3-4b-mixed_agent.sh + CompressHistoryDataArgs defaults.
AGENT_STAGE_KWARGS = {
    "max_length": 16384,
    "max_doc_length": 1024,
    "min_doc_num": 2,
    "max_doc_num": 10,
    "max_system_length": 8192,
}

OFFICIAL_ASSUMPTIONS = [
    "mdoc paths are resolved with train_mdoc._candidate_roots/_source_path under --data_root "
    "(cleaned layout) and loaded with datasets.load_from_disk, exactly like MultiDocDataset.",
    "mdoc num_samples selects the FIRST N samples per source (MultiDocDataset semantics); "
    "the default 32768 is the 2**15 per-source budget; the official script leaves "
    "--train_source_sizes unset (all samples after the first 512 eval-reserved ones) — pass "
    "--mdoc_num_samples all to replicate that exactly.",
    "mdoc rows are produced by MultiDocDataset._preprocess_mdoc_sample verbatim "
    "(max_length=1024, max_doc_length=1024, max_doc_num=10, max_system_length=256; "
    "hotpotqa/wikimqa use the identity extractor, incl. the QA_QUERY_PROMPTS question prefix and "
    "the empty-doc strip, longmagpie uses extract_docs=None) and dropped when no label is "
    "supervised, like MultiDocDataset's final filter; the official shuffle (seed 2948) is "
    "order-only and does not change token sums, so it is not reproduced.",
    "agent rows use OpenSWETracesCompressHistorySource (resolved_only=True, recent_message_num=4, "
    "max_samples_per_trace=8) + CompressHistoryDataset.preprocess_example (max_doc_num=10, "
    "max_doc_length=1024, max_system_length=8192, max_length=16384, min_doc_num=2, tail "
    "selection); the FIRST --agent_num_samples (default 130000) expanded records are scanned, "
    "like CompressHistoryDataset.__init__.",
    "agent expansion (_expand_openswe_batch) uses the UNSEEDED global random module in the "
    "official code; this scan seeds random.seed(--split_seed) so --num_proc 1 is deterministic; "
    "official trainer runs vary slightly for the same reason.",
    "U_src hashes whitespace-normalized raw document text (sha1) and counts "
    "tokenizer.encode(text, add_special_tokens=False) once per unique document, pool-level "
    "(documents dropped by grid budget caps or preprocessing filters are still counted); "
    "U_src is epoch-invariant. Agent-stage compressible source = history-message contents only; "
    "the system prompt + first user request stay uncompressed and are excluded.",
    "P_src counts context_input_ids != -100 per emitted row, INCLUDING the chat-template wrapper "
    "tokens around each compressed document; T_tgt counts labels != -100 (answer + EOS). "
    "The official recipe trains 1 epoch per stage, so P_official = stages.total.P_src.",
    U_SRC_NOTE,
    P_SRC_NOTE,
    T_TGT_NOTE,
]


def _resolve_mdoc_source_paths(
    sources: Sequence[str],
    data_root: str,
    train_data_cleaned: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve official mdoc source paths; raise an actionable error if missing."""
    from .train_mdoc import _candidate_roots, _source_path  # lazy: pulls in the model stack

    train_root, cleaned_root = _candidate_roots(str(data_root), train_data_cleaned)
    resolved: Dict[str, str] = {}
    errors: List[str] = []
    for source in sources:
        try:
            resolved[source] = _source_path(source, train_root, cleaned_root)
        except ValueError:
            raise  # unsupported source name: usage error, not missing data
        except FileNotFoundError as exc:
            errors.append(f"  {source}: {exc}")
    if errors:
        raise FileNotFoundError(
            f"official mdoc stage: training data not found under --data_root {str(data_root)!r} "
            f"for {len(errors)} source(s):\n" + "\n".join(errors) + "\n"
            "Fix: pass --data_root pointing at the mdoc dataset root (the --train_data of "
            "scripts/train_qwen3-4b-mixed_mdoc.sh, e.g. /mnt/nas1/duchuheng/datasets on the "
            "training server) laid out as python/train/train_mdoc.py:_source_path expects "
            "(<root>/hotpotqa_train_cleaned, <root>/wikimqa_train_cleaned, "
            "<root>/longmagpie_cleaned, or the same under <root>_cleaned/), or restrict "
            "--mdoc_sources to the available sources."
        )
    return resolved


def _check_agent_data(agent_data: str) -> None:
    """Replicate OpenSWETracesCompressHistorySource's parquet discovery, with a clear error."""
    path = Path(agent_data)
    if path.is_file():
        return
    search_root = path / "data" if (path / "data").is_dir() else path
    if sorted(search_root.glob("*/*.parquet")) or sorted(search_root.glob("*.parquet")):
        return
    raise FileNotFoundError(
        f"official agent stage: no parquet files found under --agent_data {str(path)!r} "
        f"(searched {search_root}/*/*.parquet and {search_root}/*.parquet). "
        "Fix: pass --agent_data pointing at the Open-SWE-Traces dataset dir (the --train_data "
        "of scripts/train_qwen3-4b-mixed_agent.sh, e.g. "
        "/mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces--train on the training server), "
        "or drop --agent_data to scan the mdoc stage only."
    )


def scan_official_mdoc_source(
    source: str,
    path: str,
    num_samples: Optional[int],
    tokenizer,
) -> Dict[str, Any]:
    """Stats-only scan of one official mdoc source (see module docstring)."""
    source = source.lower()
    if source not in OFFICIAL_MDOC_SOURCES:
        raise ValueError(f"Unsupported mdoc source for token accounting: {source!r}")
    data = datasets.load_from_disk(path)
    # Selection semantics of MultiDocDataset.__init__ (before its order-only shuffle).
    if num_samples is None:
        data = data.select(range(min(512, len(data)), len(data)))
    else:
        data = data.select(range(min(num_samples, len(data))))
    # Mirror MultiDocDataset's per-source extractor choice for the cleaned dirs.
    extract_docs = (lambda sample: sample) if source in ("hotpotqa", "wikimqa") else None
    accumulator = TokenAccumulator()
    dropped = 0
    for sample in data:
        accumulator.add_source_documents(sample.get("documents") or [], tokenizer)
        row = MultiDocDataset._preprocess_mdoc_sample(
            sample,
            tokenizer=tokenizer,
            extract_docs=extract_docs,
            preprocess_cache_version="mdoc_answer_preserve_v2",
            **MDOC_STAGE_KWARGS,
        )
        if not any(label != -100 for label in row["labels"]):
            dropped += 1
            continue
        accumulator.add_row(row)
    return {
        "accumulator": accumulator,
        "selected_samples": len(data),
        "dropped_no_supervision": dropped,
    }


def scan_official_agent(
    agent_data: str,
    num_samples: Optional[int],
    tokenizer,
    *,
    num_proc: int = 8,
    seed: int = 42,
    resolved_only: bool = True,
    recent_message_num: int = 4,
    max_samples_per_trace: Optional[int] = 8,
) -> Dict[str, Any]:
    """Stats-only scan of the official open_swe agent stage (see module docstring)."""
    _check_agent_data(agent_data)
    # The official _expand_openswe_batch draws from the unseeded global random
    # module; seed it so --num_proc 1 scans are reproducible (see docstring).
    random.seed(seed)
    source = OpenSWETracesCompressHistorySource(
        agent_data,
        resolved_only=resolved_only,
        languages=None,
        max_total_chars=None,
        max_answer_chars=None,
        recent_message_num=recent_message_num,
        num_proc=num_proc,
        max_samples_per_trace=max_samples_per_trace,
    )
    data = source.data
    if num_samples is not None:
        data = data.select(range(min(num_samples, len(data))))
    accumulator = TokenAccumulator()
    dropped = 0
    for record in data:
        history = _json_loads(record.get("history_messages"), [])
        accumulator.add_source_documents(
            [str(message.get("content") or "") for message in history],
            tokenizer,
        )
        example = CompressHistoryExample(
            qid=str(record.get("qid", "")),
            system_prompt=record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
            tools=_json_loads(record.get("tools"), []),
            history_messages=history,
            current_messages=_json_loads(record.get("current_messages"), []),
            answer=record.get("answer") or "",
        )
        row = CompressHistoryDataset.preprocess_example(
            example,
            tokenizer=tokenizer,
            history_selection="tail",
            full_history_doc_num=0,
            split_oversized_history_docs=True,
            **AGENT_STAGE_KWARGS,
        )
        if row is None:
            dropped += 1
            continue
        accumulator.add_row(row)
    return {
        "accumulator": accumulator,
        "expanded_records": len(source.data),
        "selected_records": len(data),
        "dropped_invalid": dropped,
    }


def run_official(args: argparse.Namespace, tokenizer) -> Dict[str, Any]:
    sources = [item.strip().lower() for item in args.mdoc_sources.split(",") if item.strip()]
    if not sources:
        raise ValueError("--mdoc_sources must not be empty")
    if args.agent_data:
        _check_agent_data(args.agent_data)  # fail fast before the heavy mdoc scan
    resolved = _resolve_mdoc_source_paths(sources, args.data_root, args.train_data_cleaned)
    num_samples = (
        None
        if str(args.mdoc_num_samples).lower() in ("all", "none", "-1")
        else int(args.mdoc_num_samples)
    )
    per_source: Dict[str, Any] = {}
    mdoc_total = TokenAccumulator()
    for source in sources:
        logger.info("Scanning official mdoc source %s from %s", source, resolved[source])
        result = scan_official_mdoc_source(source, resolved[source], num_samples, tokenizer)
        mdoc_total.merge(result["accumulator"])
        per_source[source] = {
            **result["accumulator"].to_dict(),
            "path": resolved[source],
            "selected_samples": result["selected_samples"],
            "dropped_no_supervision": result["dropped_no_supervision"],
        }
    agent_report = None
    agent_total = TokenAccumulator()
    if args.agent_data:
        logger.info("Scanning official agent stage (open_swe) from %s", args.agent_data)
        result = scan_official_agent(
            args.agent_data,
            args.agent_num_samples,
            tokenizer,
            num_proc=args.num_proc,
            seed=args.split_seed,
        )
        agent_total = result["accumulator"]
        agent_report = {
            "source": "open_swe",
            "path": str(args.agent_data),
            **agent_total.to_dict(),
            "expanded_records": result["expanded_records"],
            "selected_records": result["selected_records"],
            "dropped_invalid": result["dropped_invalid"],
        }
    grand_total = TokenAccumulator()
    grand_total.merge(mdoc_total)
    grand_total.merge(agent_total)
    return {
        "mode": "official",
        "tokenizer": args.tokenizer,
        "data_root": str(args.data_root),
        "stages": {
            "mdoc": {
                "num_samples_per_source": num_samples if num_samples is not None else "all",
                "per_source": per_source,
                "total": mdoc_total.to_dict(),
            },
            "agent": agent_report,
            "total": grand_total.to_dict(),
        },
        "assumptions": OFFICIAL_ASSUMPTIONS,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _load_tokenizer(spec: str):
    if spec.lower() in ("fake", "whitespace", "self-test", "self_test"):
        from .train_data_joint import _WhitespaceSelfTestTokenizer

        return _WhitespaceSelfTestTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec)


def run_joint(args: argparse.Namespace, tokenizer) -> Dict[str, Any]:
    source = AgentLLMTracesJointSource(
        path=args.dataset_path,
        split=args.split,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session or None,
        max_records=args.max_records,
        require_tool_call=args.require_tool_call,
        max_input_chars=args.max_input_chars,
        max_answer_chars=args.max_answer_chars,
        prefix_history_doc_num=args.prefix_history_doc_num,
        prefix_history_exact=args.prefix_history_exact,
        canonical_format_prob=args.canonical_format_prob,
        minified_json_prob=args.minified_json_prob,
        shuffle_tools=not args.no_shuffle_tools,
        truncate_description_chars=args.truncate_description_chars,
    )
    logger.info("Built %d joint examples from %s", len(source.records), args.dataset_path)
    report = scan_joint_examples(
        source.records,
        tokenizer,
        max_length=args.max_length,
        max_doc_length=args.max_doc_length,
        min_doc_num=args.min_doc_num,
        max_doc_num=args.max_doc_num,
        max_system_length=args.max_system_length,
        history_selection=args.history_selection,
        doc_mode=args.doc_mode,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        split_oversized_history_docs=not args.no_split_oversized_history_docs,
        per_side_caps=not args.legacy_mode_caps,
        epochs=args.epochs,
        max_examples=args.max_examples,
    )
    report.update({
        "mode": "joint",
        "dataset_path": args.dataset_path,
        "tokenizer": args.tokenizer,
    })
    return report


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="C2KV token-budget accounting (U_src / P_src / T_tgt); see module docstring.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    joint = subparsers.add_parser("joint", help="scan a true-joint manifest (AgentLLMTracesJointSource)")
    joint.add_argument("--dataset_path", required=True, help="agent-llm-traces parquet/jsonl dir")
    # Source args mirroring AgentLLMTracesJointSource.__init__.
    joint.add_argument("--split", default="train")
    joint.add_argument("--eval_ratio", type=float, default=0.1)
    joint.add_argument("--split_seed", type=int, default=42)
    joint.add_argument("--split_manifest_file", default=None)
    joint.add_argument("--split_manifest_name", default="subset_disjoint")
    joint.add_argument("--max_samples_per_session", type=int, default=4, help="0 disables")
    joint.add_argument("--max_records", type=int, default=None)
    joint.add_argument("--require_tool_call", action="store_true")
    joint.add_argument("--max_input_chars", type=int, default=None)
    joint.add_argument("--max_answer_chars", type=int, default=None)
    joint.add_argument("--prefix_history_doc_num", type=int, default=None)
    joint.add_argument("--prefix_history_exact", action="store_true")
    joint.add_argument("--canonical_format_prob", type=float, default=0.7)
    joint.add_argument("--minified_json_prob", type=float, default=0.2)
    joint.add_argument("--no_shuffle_tools", action="store_true")
    joint.add_argument("--truncate_description_chars", type=int, default=600)
    # Dataset args mirroring JointDataset.
    joint.add_argument("--max_length", type=int, default=1024)
    joint.add_argument("--max_doc_length", type=int, default=1024)
    joint.add_argument("--min_doc_num", type=int, default=2)
    joint.add_argument("--max_doc_num", type=int, default=10)
    joint.add_argument("--max_system_length", type=int, default=2048)
    joint.add_argument("--history_selection", default="tail", choices=["tail", "head"])
    joint.add_argument("--doc_mode", default="joint", choices=["joint", "tool_only", "history_only"])
    joint.add_argument("--max_tool_chunks", type=int, default=None)
    joint.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    joint.add_argument("--no_split_oversized_history_docs", action="store_true")
    joint.add_argument(
        "--legacy_mode_caps",
        action="store_true",
        help="Measure with the pre-fix doc budgets (as the pre-fix small arms trained).",
    )
    # Accounting args.
    joint.add_argument("--tokenizer", required=True, help="local HF tokenizer path ('fake' = offline smoke)")
    joint.add_argument("--max_examples", type=int, default=None)
    joint.add_argument("--epochs", type=int, default=1)
    joint.add_argument("--out", default=None, help="JSON output path (default: stdout)")

    official = subparsers.add_parser("official", help="estimate P_official of the official two-stage recipe")
    official.add_argument(
        "--data_root",
        required=True,
        help="mdoc dataset root (the --train_data of scripts/train_qwen3-4b-mixed_mdoc.sh)",
    )
    official.add_argument("--train_data_cleaned", default=None, help="optional cleaned root override")
    official.add_argument("--mdoc_sources", default="hotpotqa,wikimqa,longmagpie")
    official.add_argument(
        "--mdoc_num_samples",
        default="32768",
        help="per-source sample cap (default 2**15); 'all' replicates unset --train_source_sizes",
    )
    official.add_argument(
        "--agent_data",
        default=None,
        help="Open-SWE-Traces dir (the --train_data of scripts/train_qwen3-4b-mixed_agent.sh); "
        "omit to scan the mdoc stage only",
    )
    official.add_argument("--agent_num_samples", type=int, default=130000)
    official.add_argument("--num_proc", type=int, default=8, help="use 1 for a deterministic agent scan")
    official.add_argument("--split_seed", type=int, default=42)
    official.add_argument("--tokenizer", required=True, help="local HF tokenizer path ('fake' = offline smoke)")
    official.add_argument("--out", default=None, help="JSON output path (default: stdout)")

    args = parser.parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer)
    if args.mode == "joint":
        report = run_joint(args, tokenizer)
    else:
        report = run_official(args, tokenizer)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.out)
    else:
        print(text)
    return report


if __name__ == "__main__":
    main()
