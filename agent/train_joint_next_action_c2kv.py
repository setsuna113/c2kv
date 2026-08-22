from __future__ import annotations

import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from transformers import DataCollatorWithPadding, HfArgumentParser

from gist_args import ModelArgs, TrainingArgs
from models import format_numel_str, get_model_and_tokenizer
from train.trainer import GistMultiDocTrainer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.train_data_joint import (  # noqa: E402
    AgentLLMTracesJointSource,
    JointDataset,
    JointExample,
)
from train.train_data_joint_multisource import (  # noqa: E402
    OpenSWEJointSource,
    QADocsJointSource,
    ToucanJointSource,
    qid_source_family,
)
from train_agent_tool_definition_c2kv import _setup_device  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


DOC_MODES = ("joint", "tool_only", "history_only", "alternate")


@dataclass
class JointDataArgs:
    dataset_path: str = "./datasets/agent-llm-traces"
    eval_ratio: float = 0.1
    split_seed: int = 42
    split_manifest_file: Optional[str] = None
    split_manifest_name: str = "subset_disjoint"
    max_samples_per_session: int = 4
    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = None
    max_doc_length: int = 1024
    max_doc_num: int = 24
    max_tool_chunks: Optional[int] = None  # None -> JointDataset default: 2/3 of max_doc_num
    # Reproduce the pre-fix doc budgets (single-side modes got all max_doc_num
    # slots; plain head-truncation could drop the target tool schema).  Only
    # for diffing against the pre-fix small arms — new runs keep the default.
    legacy_mode_caps: bool = False
    max_length: int = 2048
    max_system_length: int = 512
    max_tool_definition_tokens: int = 32000
    min_target_tokens: int = 32
    require_tool_call: bool = True
    history_selection: str = "tail"
    doc_mode: str = "joint"  # joint | tool_only | history_only | alternate
    max_tools_per_sample: int = 32
    same_namespace_negative_tools: int = 8
    random_negative_tools: int = 24
    example_order_file: Optional[str] = None
    max_source_tokens: Optional[int] = None
    # G-medium multi-source mixture (train split only; eval stays traces-only).
    toucan_path: Optional[str] = None
    openswe_path: Optional[str] = None
    qa_hotpotqa_path: Optional[str] = None
    qa_2wiki_path: Optional[str] = None
    qa_longmagpie_path: Optional[str] = None
    multisource_max_records: Optional[int] = None  # smoke-test cap applied per extra source
    device_type: str = "auto"
    npu_attn_impl: str = "npu_fusion_attention"


class MinTargetJointDataset(JointDataset):
    """JointDataset plus the tool-definition path's min_target_tokens knob.

    ``AgentToolDefinitionDataset`` reserves ``min(len(answer), min_target_tokens)``
    tokens of answer budget when truncating the prompt.  ``JointDataset``
    tail-truncates an over-long prompt to ``max_length - 1`` instead, which
    can leave a single supervised token.  This wrapper drops exactly those
    rows — the ones whose answer was truncated below the reserved floor —
    with the same ``target_tokens<...`` skip reason.  Answers that fit in
    full are always kept, as in the tool-definition path.
    """

    def __init__(self, *args: Any, min_target_tokens: int = 32, **kwargs: Any) -> None:
        self._min_target_tokens = min_target_tokens
        super().__init__(*args, **kwargs)

    def preprocess_example(self, example: JointExample, tokenizer, **kwargs: Any):  # type: ignore[override]
        row, reason = JointDataset.preprocess_example(example, tokenizer=tokenizer, **kwargs)
        if row is None:
            return None, reason
        answer_token_count = len(tokenizer.encode(example.answer, add_special_tokens=False)) + 1  # + EOS
        reserved = min(answer_token_count, max(1, self._min_target_tokens))
        supervised = sum(1 for value in row["labels"] if value != -100)
        if supervised < reserved:
            return None, f"target_tokens<{reserved}"
        return row, "ok"


def _load_joint_examples(data_args: JointDataArgs, split: str) -> List[JointExample]:
    source = AgentLLMTracesJointSource(
        path=data_args.dataset_path,
        split=split,
        eval_ratio=data_args.eval_ratio,
        split_seed=data_args.split_seed,
        split_manifest_file=data_args.split_manifest_file,
        split_manifest_name=data_args.split_manifest_name,
        max_samples_per_session=data_args.max_samples_per_session,
        require_tool_call=data_args.require_tool_call,
        max_tools_per_sample=data_args.max_tools_per_sample,
        same_namespace_negative_tools=data_args.same_namespace_negative_tools,
        random_negative_tools=data_args.random_negative_tools,
    )
    return list(source)


def _load_multisource_examples(
    data_args: JointDataArgs,
    keep_qids: Optional[FrozenSet[str]] = None,
) -> List[JointExample]:
    """Load the G-medium extra sources (Toucan / Open-SWE / QA docs), train split.

    Disabled sources (path None) contribute nothing; with all paths None this
    is a no-op and the run behaves exactly as the single-source trainer.
    ``keep_qids`` (the ``--example_order_file`` qid set) prefilters during the
    source scans as a memory guard; the post-load ``_apply_example_order_file``
    call stays the authoritative filter/reorder.  ``max_samples_per_session``
    is deliberately NOT forwarded: the extra sources default to no subsampling
    and the mixture planner builds its pools with the same defaults, so the
    two sides cannot diverge (traces keeps its own subsampling above).
    """
    examples: List[JointExample] = []
    common: Dict[str, Any] = dict(
        split="train",
        keep_qids=keep_qids,
        max_records=data_args.multisource_max_records,
        split_seed=data_args.split_seed,
        require_tool_call=data_args.require_tool_call,
    )
    tool_knobs: Dict[str, Any] = dict(
        max_tools_per_sample=data_args.max_tools_per_sample,
        same_namespace_negative_tools=data_args.same_namespace_negative_tools,
        random_negative_tools=data_args.random_negative_tools,
    )
    if data_args.toucan_path:
        examples.extend(list(ToucanJointSource(data_args.toucan_path, **common, **tool_knobs)))
    if data_args.openswe_path:
        examples.extend(list(OpenSWEJointSource(data_args.openswe_path, **common, **tool_knobs)))
    if any([data_args.qa_hotpotqa_path, data_args.qa_2wiki_path, data_args.qa_longmagpie_path]):
        examples.extend(list(
            QADocsJointSource(
                hotpotqa_path=data_args.qa_hotpotqa_path,
                wiki2_path=data_args.qa_2wiki_path,
                longmagpie_path=data_args.qa_longmagpie_path,
                **common,
            )
        ))
    return examples


def _apply_example_order_file(
    examples: Sequence[JointExample],
    order_file: str,
) -> List[JointExample]:
    """Filter + reorder train examples to EXACTLY the qids in ``order_file``.

    All experiment arms must see identical sessions/targets/order; this file
    is the mechanism.  Hard error on unknown qids (listed but absent from the
    loaded examples), on duplicate qids (in the file or in the data), and on a
    malformed file.  Examples not listed are filtered out.
    """
    raw = json.loads(Path(order_file).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"--example_order_file must be a JSON list of qid strings: {order_file}")
    duplicates = sorted(qid for qid, count in Counter(raw).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"--example_order_file contains {len(duplicates)} duplicate qid(s), e.g. {duplicates[:5]}"
        )
    by_qid: Dict[str, JointExample] = {}
    for example in examples:
        if example.qid in by_qid:
            raise RuntimeError(f"duplicate qid in loaded train examples: {example.qid}")
        by_qid[example.qid] = example
    unknown = [qid for qid in raw if qid not in by_qid]
    if unknown:
        raise ValueError(
            f"--example_order_file lists {len(unknown)} qid(s) missing from the loaded "
            f"train examples, e.g. {unknown[:5]}"
        )
    logger.info(
        "example_order_file=%s: ordered %d qids, filtered out %d train examples",
        order_file,
        len(raw),
        len(examples) - len(raw),
    )
    return [by_qid[qid] for qid in raw]


def _estimate_source_tokens(example: JointExample, tokenizer) -> int:
    """Pre-chunking source-token estimate: tool documents + history documents."""
    return sum(
        len(tokenizer.encode(doc))
        for doc in example.tool_documents + example.history_documents
    )


def _take_within_source_token_budget(
    examples: Sequence[JointExample],
    tokenizer,
    max_source_tokens: int,
) -> Tuple[List[JointExample], int]:
    """Greedy prefix take until the cumulative source-token estimate reaches the budget.

    Applied AFTER the effective order is fixed (``--example_order_file``), so
    every experiment arm truncates the same sequence at the same point.  The
    example that crosses the budget is included.  Returns the kept prefix and
    the achieved estimate.  Precise token accounting happens elsewhere; this
    is the training-side knob.
    """
    kept: List[JointExample] = []
    total = 0
    for example in examples:
        if total >= max_source_tokens:
            break
        kept.append(example)
        total += _estimate_source_tokens(example, tokenizer)
    logger.info(
        "max_source_tokens=%d: kept %d/%d examples (achieved estimated source tokens=%d)",
        max_source_tokens,
        len(kept),
        len(examples),
        total,
    )
    return kept, total


def _interleave_rows(
    tool_rows: Sequence[Dict[str, Any]],
    history_rows: Sequence[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    """Deterministic zip-round-robin interleave of two list-backed datasets.

    Strict alternation (one row from each side per round; the longer side's
    leftover tail appended in order); a seeded coin flip decides once which
    side leads, so the shared extractor sees alternating batch types.  Plain
    lists in, plain list out — no torch dependency.
    """
    rng = random.Random(seed)
    first, second = (tool_rows, history_rows) if rng.random() < 0.5 else (history_rows, tool_rows)
    rows: List[Dict[str, Any]] = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            rows.append(first[index])
        if index < len(second):
            rows.append(second[index])
    return rows


def _dump_train_manifest(
    data_args: JointDataArgs,
    output_dir: str,
    train_examples: Sequence[JointExample],
    eval_examples: Sequence[JointExample],
    interleaved_train_len: Optional[int],
    achieved_source_tokens: Optional[int],
) -> str:
    path = Path(output_dir) / "train_manifest_used.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "doc_mode": data_args.doc_mode,
        "legacy_mode_caps": data_args.legacy_mode_caps,
        "split_seed": data_args.split_seed,
        "example_order_file": data_args.example_order_file,
        "max_source_tokens": data_args.max_source_tokens,
        "achieved_source_tokens": achieved_source_tokens,
        "num_train_examples": len(train_examples),
        "train_qids": [example.qid for example in train_examples],
        "train_subset_counts": dict(Counter(example.subset for example in train_examples)),
        # Mixture audit: qid-family counts (traces qids are bare
        # ``session_id:span_index`` and count as "traces").
        "train_source_counts": dict(
            Counter(qid_source_family(example.qid) for example in train_examples)
        ),
        "interleaved_train_len": interleaved_train_len,
        "num_eval_examples": len(eval_examples),
        "eval_qids": [example.qid for example in eval_examples],
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote effective train manifest to %s", path)
    return str(path)


def main() -> None:
    parser = HfArgumentParser([ModelArgs, TrainingArgs, JointDataArgs])
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    if data_args.doc_mode not in DOC_MODES:
        raise ValueError(f"Unsupported --doc_mode {data_args.doc_mode!r}; choose from {DOC_MODES}")
    device_type = _setup_device(model_args, data_args)

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils

        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        device=device_type,
        evaluation_mode=not training_args.do_train,
    )
    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_("gist" in name)

    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(
        "Trainable Model params: "
        f"{format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}"
    )

    # The joint source scans the full agent-llm-traces parquet directory and
    # can take several minutes.  Building it inside `main_process_first` makes
    # non-zero ranks wait at an HCCL barrier long enough to hit socket
    # timeouts on Ascend.  Every rank builds the same deterministic example
    # list instead (same split seed, same manifest/order file); this is more
    # I/O, but avoids distributed startup deadlocks.
    train_examples = _load_joint_examples(data_args, "train")
    eval_examples = _load_joint_examples(data_args, "eval")

    order_qids: Optional[FrozenSet[str]] = None
    if data_args.example_order_file:
        # Pre-read the frozen qid list as a scan-time prefilter for the extra
        # sources (a memory guard only); _apply_example_order_file below
        # re-reads the file and stays the authoritative filter/reorder.
        raw_order_qids = json.loads(Path(data_args.example_order_file).read_text(encoding="utf-8"))
        if isinstance(raw_order_qids, list):
            order_qids = frozenset(str(qid) for qid in raw_order_qids)
    train_examples.extend(_load_multisource_examples(data_args, keep_qids=order_qids))

    if data_args.example_order_file:
        train_examples = _apply_example_order_file(train_examples, data_args.example_order_file)
    achieved_source_tokens: Optional[int] = None
    if data_args.max_source_tokens is not None:
        train_examples, achieved_source_tokens = _take_within_source_token_budget(
            train_examples, tokenizer, data_args.max_source_tokens
        )
    if data_args.max_train_examples is not None:
        train_examples = train_examples[: data_args.max_train_examples]
    if data_args.max_eval_examples is not None:
        eval_examples = eval_examples[: data_args.max_eval_examples]
    logger.info(
        "Joint examples: train=%d eval=%d (train subsets=%s)",
        len(train_examples),
        len(eval_examples),
        dict(Counter(example.subset for example in train_examples)),
    )

    dataset_kwargs: Dict[str, Any] = dict(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_doc_length=data_args.max_doc_length,
        max_doc_num=data_args.max_doc_num,
        max_system_length=data_args.max_system_length,
        history_selection=data_args.history_selection,
        max_tool_chunks=data_args.max_tool_chunks,
        max_tool_definition_tokens=data_args.max_tool_definition_tokens,
        min_target_tokens=data_args.min_target_tokens,
        per_side_caps=not data_args.legacy_mode_caps,
    )
    interleaved_train_len: Optional[int] = None
    if data_args.doc_mode == "alternate":
        # Same examples rendered twice — tool documents only / history
        # documents only — then interleaved so the shared extractor sees
        # alternating batch types.  Eval stays joint.
        tool_dataset = MinTargetJointDataset(train_examples, doc_mode="tool_only", **dataset_kwargs)
        history_dataset = MinTargetJointDataset(train_examples, doc_mode="history_only", **dataset_kwargs)
        if len(tool_dataset) == 0 or len(history_dataset) == 0:
            logger.warning(
                "alternate doc_mode with an empty side: tool_only=%d history_only=%d",
                len(tool_dataset),
                len(history_dataset),
            )
        train_dataset: Any = _interleave_rows(tool_dataset.data, history_dataset.data, data_args.split_seed)
        interleaved_train_len = len(train_dataset)
        eval_dataset = MinTargetJointDataset(eval_examples, doc_mode="joint", **dataset_kwargs)
    else:
        train_dataset = MinTargetJointDataset(train_examples, doc_mode=data_args.doc_mode, **dataset_kwargs)
        eval_dataset = MinTargetJointDataset(eval_examples, doc_mode=data_args.doc_mode, **dataset_kwargs)

    if len(train_dataset) == 0:
        raise RuntimeError("No train samples remained after filtering")

    if training_args.do_train:
        _dump_train_manifest(
            data_args,
            training_args.output_dir,
            train_examples,
            eval_examples,
            interleaved_train_len,
            achieved_source_tokens,
        )

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        max_doc_length=data_args.max_doc_length,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        if len(eval_dataset) == 0:
            raise ValueError("Evaluation requested but eval dataset is empty")
        logger.info("Evaluation result: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
