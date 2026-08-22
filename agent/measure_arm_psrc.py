"""Measure each trained arm's ACTUAL presented source tokens (P_src).

The training-side budget knob (``--max_source_tokens``) selects examples by a
pre-chunking ESTIMATE (``_estimate_source_tokens``: raw docs, no chat
template, no caps); the G-Q3 prereg fairness constraint is stated in P_src —
presented non-padding ``context_input_ids`` tokens.  The progress report's
"achieved_source_tokens equal across arms" check only proves the arms share
one example pool (the estimator is doc-mode-independent), NOT that their
presented budgets match.  This driver closes that gap: it recomputes
U_src/P_src/T_tgt per arm from the arm's ``train_manifest_used.json``
(``doc_mode`` + ordered ``train_qids``) with the real preprocessing, then
reports the prereg comparisons directly:

- ``sep_combined_over_joint``: P_src(sep_tool) + P_src(sep_hist) vs
  P_src(joint) — the constraint says this should be ~1.0; the pre-fix slot
  uplift (single-side modes got all ``max_doc_num`` slots) pushes it toward
  2.0 on saturated samples.
- ``alternate_over_joint``: the alternate arm presents every sample twice
  (tool-only pass + history-only pass), so its P_src mirrors sep-combined.

Run it once with ``--legacy_mode_caps`` to measure the arms AS TRAINED
(pre-fix budgets), and once without to preview the fixed budgets.  Dataset /
source args must match the training launch (same manifest, same knobs).

G-medium mixture arms: pass the extra source paths exactly as the training
launch did — ``--toucan_path`` / ``--openswe_path`` / ``--qa_hotpotqa_path`` /
``--qa_2wiki_path`` / ``--qa_longmagpie_path`` (JointDataArgs mirrors).  The
extra sources load through the trainer's own ``_load_multisource_examples``
(same knobs, ``keep_qids`` prefilter to the union of the arms' train_qids),
so the measurement口径 is the training口径 and ``qa:``/``toucan:``/
``openswe:`` manifest qids resolve instead of dying as unknown (P1-4).

Example (NPU server):
  python agent/measure_arm_psrc.py \
    --dataset_path ~/c2kv/datasets/agent-llm-traces-v2 \
    --split_manifest_file ~/c2kv/outputs_lyc/g_joint/taskproxy_disjoint_v2.json \
    --tokenizer ~/c2kv/models/Qwen3-4B-Instruct-2507 \
    --require_tool_call --max_samples_per_session 4 \
    --max_doc_num 24 --max_length 2048 --max_system_length 512 \
    --arm joint=~/c2kv/outputs_lyc/g_joint/joint/train_manifest_used.json \
    --arm sep_tool=~/c2kv/outputs_lyc/g_joint/sep_tool/train_manifest_used.json \
    --arm sep_hist=~/c2kv/outputs_lyc/g_joint/sep_hist/train_manifest_used.json \
    --arm alternate=~/c2kv/outputs_lyc/g_joint/alternate/train_manifest_used.json \
    --legacy_mode_caps --out psrc_as_trained.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

if True:  # isort: keep the path bootstrap above the repo imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))

from train.token_accounting import scan_joint_examples  # noqa: E402
from train.train_data_joint import (  # noqa: E402
    CAP_REGIME_V1,
    CAP_REGIME_V2_EMPTY_TOOL_RECLAIM,
    AgentLLMTracesJointSource,
    JointExample,
    _has_tool_documents,
    cap_regime_name,
    regime_from_record,
)

logger = logging.getLogger(__name__)


def _load_examples(args: argparse.Namespace, needed_qids: Optional[FrozenSet[str]] = None) -> Dict[str, JointExample]:
    source = AgentLLMTracesJointSource(
        path=args.dataset_path,
        split="train",
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session or None,
        require_tool_call=args.require_tool_call,
        max_tools_per_sample=args.max_tools_per_sample,
        same_namespace_negative_tools=args.same_namespace_negative_tools,
        random_negative_tools=args.random_negative_tools,
    )
    by_qid: Dict[str, JointExample] = {}
    for example in source.records:
        if example.qid in by_qid:
            raise RuntimeError(f"duplicate qid in loaded train examples: {example.qid}")
        by_qid[example.qid] = example
    if any([args.toucan_path, args.openswe_path, args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path]):
        # G-medium mixture arms: load the extra sources through the TRAINER's
        # own multisource path (same knobs, same keep_qids prefilter) so the
        # measurement口径 is the training口径 — qa:/toucan:/openswe: qids in a
        # manifest used to die here with a RuntimeError (P1-4).  Lazy import:
        # the trainer module pulls in torch/models.
        from train_joint_next_action_c2kv import JointDataArgs, _load_multisource_examples

        data_args = JointDataArgs(
            dataset_path=args.dataset_path,
            split_seed=args.split_seed,
            split_manifest_file=args.split_manifest_file,
            split_manifest_name=args.split_manifest_name,
            require_tool_call=args.require_tool_call,
            max_tools_per_sample=args.max_tools_per_sample,
            same_namespace_negative_tools=args.same_namespace_negative_tools,
            random_negative_tools=args.random_negative_tools,
            toucan_path=args.toucan_path,
            openswe_path=args.openswe_path,
            qa_hotpotqa_path=args.qa_hotpotqa_path,
            qa_2wiki_path=args.qa_2wiki_path,
            qa_longmagpie_path=args.qa_longmagpie_path,
        )
        for example in _load_multisource_examples(data_args, keep_qids=needed_qids):
            if example.qid in by_qid:
                raise RuntimeError(f"duplicate qid across sources: {example.qid}")
            by_qid[example.qid] = example
        logger.info("multisource load: %d extra examples (keep_qids=%s)", len(by_qid), needed_qids is not None)
    return by_qid


def _arm_examples(
    args: argparse.Namespace, manifest_path: Path, by_qid: Dict[str, JointExample]
) -> tuple[str, List[JointExample], Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc_mode = str(manifest.get("doc_mode") or "")
    qids = [str(qid) for qid in manifest.get("train_qids") or []]
    if not doc_mode or not qids:
        raise RuntimeError(f"{manifest_path}: doc_mode/train_qids missing")
    # Cross-check the manifest against the CLI: a wrong split_seed silently
    # changes the per-example render rng (different shuffle/format variants),
    # and a cap-regime mismatch reports budgets the arm never trained with —
    # exactly the mismeasurements this tool exists to prevent.
    manifest_seed = manifest.get("split_seed")
    if manifest_seed is not None and int(manifest_seed) != args.split_seed:
        raise RuntimeError(
            f"{manifest_path}: manifest split_seed={manifest_seed} != --split_seed {args.split_seed}"
        )
    if "legacy_mode_caps" in manifest and bool(manifest["legacy_mode_caps"]) != args.legacy_mode_caps:
        raise RuntimeError(
            f"{manifest_path}: manifest legacy_mode_caps={manifest['legacy_mode_caps']} conflicts "
            f"with --legacy_mode_caps={args.legacy_mode_caps}; measure this arm in a separate run "
            f"with the matching flag"
        )
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise RuntimeError(
            f"{manifest_path}: {len(missing)} train_qids not found in the loaded source "
            f"(source args differ from the training launch?), e.g. {missing[:5]}"
        )
    examples = [by_qid[qid] for qid in qids]
    # Cap-regime cross-check, string-level (the boolean cannot tell the two
    # post-fix regimes apart): a v1 manifest (per_side_caps, no reclaim)
    # measured with v2 code (empty-tool reclaim) is bit-identical ONLY for
    # tool-bearing examples; a QA-bearing mismatch is a hard error.
    expected_regime = cap_regime_name(args.legacy_mode_caps)
    manifest_regime = regime_from_record(manifest.get("legacy_mode_caps"), manifest.get("cap_regime"))
    if manifest_regime not in ("unknown", expected_regime):
        if {manifest_regime, expected_regime} == {
            CAP_REGIME_V1,
            CAP_REGIME_V2_EMPTY_TOOL_RECLAIM,
        } and all(_has_tool_documents(example) for example in examples):
            logger.warning(
                "%s: manifest cap_regime=%s vs code regime=%s — identical for tool-bearing "
                "examples, and every example in this arm has tools; proceeding",
                manifest_path,
                manifest_regime,
                expected_regime,
            )
        else:
            raise RuntimeError(
                f"{manifest_path}: manifest cap_regime={manifest_regime!r} conflicts with the "
                f"code regime {expected_regime!r} (--legacy_mode_caps={args.legacy_mode_caps}); "
                f"the v1/v2 regimes differ for tool-less (QA) examples"
            )
    return doc_mode, examples, manifest


def _scan(args: argparse.Namespace, tokenizer, examples: Sequence[JointExample], doc_mode: str) -> Dict[str, Any]:
    return scan_joint_examples(
        examples,
        tokenizer,
        max_length=args.max_length,
        max_doc_length=args.max_doc_length,
        min_doc_num=args.min_doc_num,
        max_doc_num=args.max_doc_num,
        max_system_length=args.max_system_length,
        history_selection=args.history_selection,
        doc_mode=doc_mode,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        per_side_caps=not args.legacy_mode_caps,
        min_target_tokens=args.min_target_tokens,
    )


def _measure_arm(args: argparse.Namespace, tokenizer, doc_mode: str, examples: Sequence[JointExample]) -> Dict[str, Any]:
    if doc_mode == "alternate":
        # The alternate arm renders every example twice: a tool-only pass and
        # a history-only pass.  Its presented budget is the sum of both scans.
        tool_scan = _scan(args, tokenizer, examples, "tool_only")
        history_scan = _scan(args, tokenizer, examples, "history_only")
        return {
            "doc_mode": doc_mode,
            "num_examples": len(examples),
            "P_src": tool_scan["total"]["P_src"] + history_scan["total"]["P_src"],
            "T_tgt": tool_scan["total"]["T_tgt"] + history_scan["total"]["T_tgt"],
            "passes": {"tool_only": tool_scan, "history_only": history_scan},
        }
    scan = _scan(args, tokenizer, examples, doc_mode)
    return {
        "doc_mode": doc_mode,
        "num_examples": len(examples),
        "P_src": scan["total"]["P_src"],
        "T_tgt": scan["total"]["T_tgt"],
        "scan": scan,
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset_path", required=True)
    # G-medium extra mixture sources (mirrors JointDataArgs; all optional,
    # train split only).  When any is set, qa:/toucan:/openswe: qids in the
    # arm manifests resolve through the trainer's own multisource loader.
    parser.add_argument("--toucan_path", default=None)
    parser.add_argument("--openswe_path", default=None)
    parser.add_argument("--qa_hotpotqa_path", default=None)
    parser.add_argument("--qa_2wiki_path", default=None)
    parser.add_argument("--qa_longmagpie_path", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file", default=None)
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--require_tool_call", action="store_true")
    parser.add_argument("--max_tools_per_sample", type=int, default=32)
    parser.add_argument("--same_namespace_negative_tools", type=int, default=8)
    parser.add_argument("--random_negative_tools", type=int, default=24)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--min_doc_num", type=int, default=2)
    parser.add_argument("--max_doc_num", type=int, default=24)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--history_selection", default="tail", choices=["tail", "head"])
    parser.add_argument("--max_tool_chunks", type=int, default=None)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    parser.add_argument(
        "--min_target_tokens",
        type=int,
        default=32,
        help="Mirror MinTargetJointDataset's row drop (trainer default 32); <=0 disables.",
    )
    parser.add_argument(
        "--legacy_mode_caps",
        action="store_true",
        help="Measure with the pre-fix doc budgets, i.e. the arms AS TRAINED before the cap fix.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=train_manifest_used.json",
        help="Repeatable: arm name and its training manifest path.",
    )
    parser.add_argument("--tokenizer", required=True, help="local HF tokenizer path ('fake' = offline smoke)")
    parser.add_argument("--out", default=None, help="JSON output path (default: stdout)")
    args = parser.parse_args(argv)

    from train.token_accounting import _load_tokenizer

    tokenizer = _load_tokenizer(args.tokenizer)
    needed_qids: Optional[FrozenSet[str]] = None
    if any([args.toucan_path, args.openswe_path, args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path]):
        # Prefilter the extra-source scans to the union of arm qids (memory
        # guard); the per-arm manifest checks below stay authoritative.
        needed: set = set()
        for spec in args.arm:
            _, _, manifest_path = spec.partition("=")
            if not manifest_path:
                raise SystemExit(f"--arm expects NAME=path, got: {spec!r}")
            manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
            needed.update(str(qid) for qid in manifest.get("train_qids") or [])
        needed_qids = frozenset(needed)
    by_qid = _load_examples(args, needed_qids)

    if args.min_target_tokens is not None and args.min_target_tokens <= 0:
        args.min_target_tokens = None

    arms: Dict[str, Dict[str, Any]] = {}
    for spec in args.arm:
        name, _, manifest_path = spec.partition("=")
        if not manifest_path:
            raise SystemExit(f"--arm expects NAME=path, got: {spec!r}")
        doc_mode, examples, manifest = _arm_examples(args, Path(manifest_path).expanduser(), by_qid)
        arms[name] = _measure_arm(args, tokenizer, doc_mode, examples)
        arms[name]["manifest_legacy_mode_caps"] = manifest.get("legacy_mode_caps")

    report: Dict[str, Any] = {
        "legacy_mode_caps": args.legacy_mode_caps,
        "arms": {
            name: {key: value for key, value in arm.items() if key not in ("passes", "scan")}
            for name, arm in arms.items()
        },
        "detail": arms,
    }
    p_src = {name: arm["P_src"] for name, arm in arms.items()}
    if "joint" in p_src and p_src["joint"]:
        if "sep_tool" in p_src and "sep_hist" in p_src:
            report["sep_combined_over_joint"] = round(
                (p_src["sep_tool"] + p_src["sep_hist"]) / p_src["joint"], 4
            )
        if "alternate" in p_src:
            report["alternate_over_joint"] = round(p_src["alternate"] / p_src["joint"], 4)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    summary = {name: arm["P_src"] for name, arm in arms.items()}
    print("P_src per arm:", json.dumps(summary, sort_keys=True))
    for key in ("sep_combined_over_joint", "alternate_over_joint"):
        if key in report:
            print(f"{key}: {report[key]}")
    return report


if __name__ == "__main__":
    main()
