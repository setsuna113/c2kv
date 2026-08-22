"""C/D shared: freeze the C->W trigger set of a paired full/compressed battery.

Takes the two arms of one evaluation batch (an uncompressed reference arm and
a compressed arm evaluated on the same qids), re-scores both from their raw
text, classifies every paired qid into one of the four transition cells, and
writes

  * a bundle jsonl — one record per C->W qid holding everything needed to
    REBUILD the compressed prefix later (no KV is ever written to disk, only
    the recipe: doc-ids provenance, checkpoint, ratio, chunk policy, eval-code
    sha, seed, decode rule);
  * a frozen manifest — source shas, the full transition census, the sorted
    C->W qid list, the two-level denominator ``n_base_paired``, and the sha of
    the bundle file itself.

Scoring is redone here with local regexes rather than trusting the harness
fields, and any disagreement with the harness-written ``tool_name_match`` /
``has_tool_call`` is warned about (never silently corrected).  Both row
dialects are accepted: joint-battery rows carry a ``condition`` key and can be
filtered with --full_condition / --compressed_condition, history-harness rows
do not.

torch-free by construction — the only phase that touches the model stack is
--bind_docs, which imports the history harness inside the function.

Usage (repo root):
  python agent/extract_cw_triggers.py \
      --full_rows results/joint/full.jsonl \
      --compressed_rows results/joint/c2kv_r8.jsonl \
      --batch batch-TF --s_metric tool_name_match \
      --out_bundles results/d/bundles_batch_tf.jsonl \
      --out_manifest configs/bdf_pilot/d_cw_manifest.json \
      --ckpt_path ./checkpoints/qwen3-4b-joint-c2kv-npu \
      --model_sha <sha> --eval_code_sha <sha> --ratio 8 \
      --chunk_policy pilot_v1 --seed 20260815 --decode greedy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("extract_cw_triggers")

RULE_VERSION = "d_cw_v1"
S_METRICS = ("tool_name_match",)
TRANSITIONS = ("C->C", "C->W", "W->C", "W->W")
TRIGGER_SOURCES = ("oracle", "L1", "silent")  # enum kept for later phases
# What the harness stamps into rows of each arm (eval_agent_history_c2kv
# forces run_ratios=[1] for mode "full"); used to cross-check the CLI claims.
FULL_ROW_MODES = frozenset({"full"})
COMPRESSED_ROW_MODES = frozenset({"c2kv"})

# Local copies of the harness scorers (agent/eval_agent_tool_definition_c2kv.py
# ::_extract_tool_name and agent/eval_agent_history_c2kv.py::_has_tool_call).
# They are duplicated rather than imported because both modules pull torch at
# import time; test_extract_cw_triggers.py locks the equivalence wherever torch
# is available.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_NAME_FIELD_RE = re.compile(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"')
_TOOL_CALL_LOOSE_RE = re.compile(r"<tool_call>.*?([A-Za-z0-9_.:-]+).*?</tool_call>", re.S)


def extract_tool_name(text: str) -> Optional[str]:
    if not text:
        return None
    blocks = _TOOL_CALL_BLOCK_RE.findall(text)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            if name:
                return str(name)
    match = _NAME_FIELD_RE.search(text)
    if match:
        return match.group(1)
    match = _TOOL_CALL_LOOSE_RE.search(text)
    if match:
        return match.group(1)
    return None


def has_tool_call(text: str) -> bool:
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


def sha256_text_file(path: Path) -> str:
    """Newline-normalised sha256 of a text artifact.

    The single hashing convention for every frozen task-D artifact (row
    files, bundles, manifest, sham plan, neutral corpus).  Hashing the
    DECODED text rather than the raw bytes makes the digest identical
    whether the file was written on Windows (CRLF) or on the NPU server
    (LF) — the frozen-state assertions in agent/d_kv_intervene.py compare
    digests produced on different machines.
    """
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


_sha256_file = sha256_text_file


def _row_text(row: Dict[str, Any]) -> str:
    """r4_paired dialect tolerance: some arms write `text`, others `prediction`."""
    return row.get("text", row.get("prediction", "")) or ""


def _load_rows_by_qid(
    paths: Sequence[str],
    condition: Optional[str] = None,
    label: str = "rows",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """r4_paired semantics: skipped rows dropped, duplicate qid fatal.

    `condition` filters the joint-battery dialect; asking for a condition on
    history-dialect rows (no ``condition`` key) is an error rather than a
    silent pass-through.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    n_skipped = 0
    n_filtered = 0
    dialects: Counter = Counter()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                dialects["joint" if "condition" in row else "history"] += 1
                if row.get("skipped"):
                    n_skipped += 1
                    continue
                if condition is not None:
                    if "condition" not in row:
                        raise SystemExit(
                            f"FATAL: --{label}_condition={condition!r} given but {path} "
                            "carries history-dialect rows (no 'condition' key)"
                        )
                    if row.get("condition") != condition:
                        n_filtered += 1
                        continue
                qid = row.get("qid")
                if qid is None:
                    raise SystemExit(f"FATAL: row without qid in {path}")
                if qid in rows:
                    raise SystemExit(f"FATAL: duplicate qid {qid} in {label} ({path})")
                rows[qid] = row
    stats = {
        "files": list(paths),
        "condition": condition,
        "n_rows": len(rows),
        "n_skipped": n_skipped,
        "n_condition_filtered": n_filtered,
        "dialects": dict(dialects),
    }
    return rows, stats


def _assert_rows_match_claims(
    rows: Dict[str, Dict[str, Any]],
    label: str,
    expected_modes: frozenset,
    expected_ratio: int,
) -> None:
    """Anchor the frozen recipe in what the battery actually ran.

    The kv_recipe is typed by the operator, but the harness writes ``mode``
    and ``ratio`` into every row (and forces ratio 1 for mode "full").  A
    recipe frozen from a mistyped --ratio would later be ENFORCED by
    d_kv_intervene's guard — the guard's direction inverts — so a
    contradiction between the CLI claim and the rows is fatal here, before
    anything is frozen.  Rows without the fields (pre-recipe dialects) skip
    the check.
    """
    for qid, row in rows.items():
        mode = row.get("mode")
        if mode is not None and mode not in expected_modes:
            raise SystemExit(
                f"FATAL: {label} row {qid} carries mode={mode!r}, expected "
                f"{sorted(expected_modes)}. The rows handed to --{label}_rows are "
                "not the arm the recipe claims; fix the inputs, not the recipe."
            )
        ratio = row.get("ratio")
        if ratio is not None and int(ratio) != int(expected_ratio):
            raise SystemExit(
                f"FATAL: {label} row {qid} carries ratio={ratio}, but the recipe "
                f"would freeze ratio={expected_ratio}. A wrong frozen ratio is later "
                "enforced by the intervention driver's guard, so it must match the "
                "rows here."
            )


def _score(row: Dict[str, Any], s_metric: str = "tool_name_match") -> Dict[str, Any]:
    """Re-score one row locally and cross-check the harness fields."""
    if s_metric not in S_METRICS:
        raise ValueError(f"Unsupported s_metric {s_metric!r}; choose from {S_METRICS}")
    text = _row_text(row)
    pred_name = extract_tool_name(text)
    target_name = row.get("target_tool_name")
    if target_name is None:
        target_name = extract_tool_name(row.get("target", ""))
    correct = target_name is not None and pred_name == target_name
    call = has_tool_call(text)
    harness_correct = row.get(s_metric)
    harness_call = row.get("has_tool_call")
    metric_agrees = harness_correct is None or bool(harness_correct) == correct
    call_agrees = harness_call is None or bool(harness_call) == call
    if not metric_agrees:
        logger.warning(
            "qid=%s: harness %s=%s but re-scored %s", row.get("qid"), s_metric, harness_correct, correct
        )
    if not call_agrees:
        logger.warning(
            "qid=%s: harness has_tool_call=%s but re-scored %s", row.get("qid"), harness_call, call
        )
    return {
        "correct": correct,
        "has_tool_call": call,
        "pred_tool_name": pred_name,
        "target_tool_name": target_name,
        "harness_metric_agrees": metric_agrees,
        "harness_call_agrees": call_agrees,
    }


def _transition(full_correct: bool, compressed_correct: bool) -> str:
    return f"{'C' if full_correct else 'W'}->{'C' if compressed_correct else 'W'}"


def _target_args(row: Dict[str, Any]) -> Any:
    """Arguments of the target call: the row field when present, otherwise
    parsed out of the target text (same candidate order as extract_tool_name)."""
    if row.get("target_args") is not None:
        return row["target_args"]
    text = row.get("target", "") or ""
    blocks = _TOOL_CALL_BLOCK_RE.findall(text)
    for candidate in blocks or [text]:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            arguments = value.get("arguments", function.get("arguments"))
            if arguments is not None:
                return arguments
    return None


def _turn_count(row: Dict[str, Any]) -> Optional[int]:
    """History document count. Joint rows split tool/history chunks."""
    for key in ("history_doc_chunks", "doc_chunks"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def _step_index(qid: str, row: Dict[str, Any]) -> Optional[int]:
    tail = qid.rsplit(":", 1)[-1] if ":" in qid else ""
    if tail.isdigit():
        return int(tail)
    decision_step = row.get("decision_step")
    return int(decision_step) - 1 if decision_step is not None else None


def _bundle_row(
    qid: str,
    full_row: Dict[str, Any],
    compressed_row: Dict[str, Any],
    full_score: Dict[str, Any],
    compressed_score: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """One rebuildable trigger record. No KV on disk — only the recipe."""
    turn = _turn_count(compressed_row)
    target_args = _target_args(compressed_row)
    if target_args is None:
        target_args = _target_args(full_row)
    return {
        "bundle_id": f"{args.batch}:{qid}",
        "batch": args.batch,
        "qid": qid,
        "session_id": compressed_row.get("session_id") or (qid.rsplit(":", 1)[0] if ":" in qid else None),
        # doc 24 D.3.4 shared-schema name; "subset" is the harness's own field
        # and is kept alongside for row-level provenance.
        "benchmark": compressed_row.get("benchmark") or compressed_row.get("subset"),
        "subset": compressed_row.get("subset"),
        "turn": turn,
        "step_index_t": _step_index(qid, compressed_row),
        # t_star (the silent-divergence step) is not estimated in the pilot;
        # the trigger is the oracle transition itself.
        "t_star": None,
        "trigger_source": "oracle",  # enum: oracle | L1 | silent
        "transition": _transition(full_score["correct"], compressed_score["correct"]),
        "s_metric": args.s_metric,
        "full_correct": full_score["correct"],
        "compressed_correct": compressed_score["correct"],
        "full_has_tool_call": full_score["has_tool_call"],
        "compressed_has_tool_call": compressed_score["has_tool_call"],
        "target_tool_name": compressed_score["target_tool_name"] or full_score["target_tool_name"],
        "target_args": target_args,
        # Raw prediction texts, verbatim: line C replays the failure from the
        # bundle alone and must not need the battery row files for that.
        "full_output": _row_text(full_row),
        "compressed_output": _row_text(compressed_row),
        "full_pred_tool_name": full_score["pred_tool_name"],
        "compressed_pred_tool_name": compressed_score["pred_tool_name"],
        "harness_metric_agrees": full_score["harness_metric_agrees"] and compressed_score["harness_metric_agrees"],
        "target_known": compressed_row.get("target_known"),
        "target_in_grid": compressed_row.get("target_in_grid"),
        # T == 1 means nothing lives downstream of k*, so the +re arm cannot
        # differ from the plain corr arm for this qid.
        "no_downstream": (turn == 1) if turn is not None else None,
        "n_docs": None,
        "doc_lens": None,
        "doc_ids_sha256": "fingerprint_pending",
        # doc 24 D.3.4 shared-schema name for the per-qid doc-ids table both
        # assemblies rebuild from (same file kv_recipe.doc_ids_table points at).
        "docs_path": args.out_doc_table,
        "kv_recipe": {
            "ckpt_path": args.ckpt_path,
            "model_sha": args.model_sha,
            "eval_code_sha": args.eval_code_sha,
            "ratio": args.ratio,
            "chunk_policy": args.chunk_policy,
            "seed": args.seed,
            "decode": args.decode,
            "dataset_path": args.dataset_path,
            "tokenizer": args.tokenizer,
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
            "doc_ids_table": args.out_doc_table,
        },
        "source": {
            "full_rows": list(args.full_rows),
            "compressed_rows": list(args.compressed_rows),
            "full_condition": args.full_condition,
            "compressed_condition": args.compressed_condition,
        },
    }


def _harness_namespace(args: argparse.Namespace) -> Any:
    """argparse namespace for the history harness doc reconstruction."""
    import eval_agent_history_c2kv as HH  # noqa: PLC0415

    argv = [
        "prog",
        "--model", args.ckpt_path or args.tokenizer,
        "--base_model", args.base_model or args.tokenizer,
        "--tokenizer", args.tokenizer,
        "--dataset_path", args.dataset_path,
        "--include_tools", "True",
        "--max_examples", "0",
        "--max_doc_length", str(args.max_doc_length),
        "--max_doc_num", str(args.max_doc_num),
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def _bind_docs(bundles: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Lazy phase: rebuild the history docs and fingerprint them.

    Imports the history harness inside the function so the default (unbound)
    path stays torch-free.  Produces the per-qid doc-length side table the
    sham planner needs, and stamps each bundle with a doc-ids sha.
    """
    _ROOT = Path(__file__).resolve().parents[1]
    for sub in ("python", "agent", "python/inference"):
        path = str(_ROOT / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import eval_agent_history_c2kv as HH  # noqa: PLC0415

    hargs = _harness_namespace(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, selection_skips = HH._load_examples(hargs, tokenizer)
    wanted = {bundle["qid"] for bundle in bundles}
    by_qid: Dict[str, Any] = {}
    for example in examples:
        if example.qid in wanted and example.qid not in by_qid:
            by_qid[example.qid] = example
    missing = sorted(wanted - set(by_qid))
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} trigger qids not reproduced: {missing[:5]}")

    per_qid: Dict[str, Any] = {}
    for bundle in bundles:
        qid = bundle["qid"]
        example = by_qid[qid]
        history = HH._history_messages(tokenizer, example, hargs)
        doc_ids = [
            HH._chat_template_ids(tokenizer, [message], max_length=hargs.max_doc_length)
            for message in history
        ]
        doc_lens = [len(ids) for ids in doc_ids]
        digest = hashlib.sha256(
            json.dumps(doc_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        bundle["n_docs"] = len(doc_ids)
        bundle["doc_lens"] = doc_lens
        bundle["doc_ids_sha256"] = digest
        bundle["no_downstream"] = len(doc_ids) == 1
        per_qid[qid] = {
            "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
            "n_docs": len(doc_ids),
            "doc_lens": doc_lens,
            "doc_ids_sha256": digest,
        }
    table = {
        "description": "Per-qid history doc lengths and ids fingerprint for the task-D trigger set.",
        "rule_version": RULE_VERSION,
        "tokenizer": args.tokenizer,
        "dataset_path": args.dataset_path,
        "max_doc_length": args.max_doc_length,
        "selection_skips": selection_skips,
        "n_qids": len(per_qid),
        "per_qid": per_qid,
    }
    out_path = Path(args.out_doc_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote doc table %s (%d qids)", out_path, len(per_qid))
    return table


def _merged_dialects(*load_stats: Dict[str, Any]) -> Dict[str, int]:
    """Union the per-source dialect counters into one manifest-level view."""
    merged: Counter = Counter()
    for stats in load_stats:
        for name, count in (stats or {}).get("dialects", {}).items():
            merged[name] += int(count)
    return dict(merged)


def _freeze_manifest(
    args: argparse.Namespace,
    full_stats: Dict[str, Any],
    compressed_stats: Dict[str, Any],
    census: Counter,
    cw_qids: Sequence[str],
    n_base_paired: int,
    bundles_path: Path,
    bound: bool,
    divergence: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return {
        "description": (
            "Frozen C->W trigger manifest for the task-D pilot. n_base_paired is the "
            "level-1 denominator (qids scoreable in BOTH arms)."
        ),
        "rule_version": RULE_VERSION,
        "batch": args.batch,
        "s_metric": args.s_metric,
        "trigger_source": "oracle",
        "kv_recipe": {
            "ckpt_path": args.ckpt_path,
            "model_sha": args.model_sha,
            "eval_code_sha": args.eval_code_sha,
            "ratio": args.ratio,
            "chunk_policy": args.chunk_policy,
            "seed": args.seed,
            "decode": args.decode,
            # Doc-grid geometry.  The intervention driver rebuilds contexts from
            # this recipe; if it runs a different budget the intervention lands
            # on a grid that is not the one whose failure defined the trigger.
            # d_kv_intervene refuses to start on a mismatch.
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
        },
        # Which harness dialect the paired rows came from.  D intervenes with the
        # history harness, so joint-battery rows are not interchangeable here even
        # though the extractor can parse both.
        "source_dialects": _merged_dialects(full_stats, compressed_stats),
        "sources": {
            "full_rows": [
                {"path": path, "sha256": _sha256_file(Path(path))} for path in args.full_rows
            ],
            "compressed_rows": [
                {"path": path, "sha256": _sha256_file(Path(path))} for path in args.compressed_rows
            ],
            "full_condition": args.full_condition,
            "compressed_condition": args.compressed_condition,
            "full_load": full_stats,
            "compressed_load": compressed_stats,
        },
        "n_base_paired": n_base_paired,
        "transitions": {key: int(census.get(key, 0)) for key in TRANSITIONS},
        # Harness-vs-local rescore disagreements over ALL paired rows (both
        # arms), machine-readable per prereg §3 "warned about and counted".
        "harness_divergence": dict(divergence or {}),
        "n_cw": len(cw_qids),
        "cw_qids": list(cw_qids),
        "bundles_file": str(bundles_path.as_posix()),
        "bundles_sha256": _sha256_file(bundles_path),
        "doc_binding": "bound" if bound else "fingerprint_pending",
        "doc_ids_table": args.out_doc_table if bound else None,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_rows", nargs="+", required=True)
    parser.add_argument("--compressed_rows", nargs="+", required=True)
    parser.add_argument("--full_condition", default=None)
    parser.add_argument("--compressed_condition", default=None)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--s_metric", choices=list(S_METRICS), default="tool_name_match")
    parser.add_argument("--out_bundles", default="./results/d/bundles_batch_tf.jsonl")
    parser.add_argument("--out_manifest", default="./configs/bdf_pilot/d_cw_manifest.json")
    parser.add_argument("--out_doc_table", default="./results/d/d_doc_ids.json")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--model_sha", required=True)
    parser.add_argument("--eval_code_sha", required=True)
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--chunk_policy", required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--decode", default="greedy")
    parser.add_argument("--bind_docs", action="store_true")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--tokenizer", default="./models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    full, full_stats = _load_rows_by_qid(args.full_rows, args.full_condition, "full")
    compressed, compressed_stats = _load_rows_by_qid(
        args.compressed_rows, args.compressed_condition, "compressed"
    )
    _assert_rows_match_claims(full, "full", FULL_ROW_MODES, 1)
    _assert_rows_match_claims(compressed, "compressed", COMPRESSED_ROW_MODES, args.ratio)
    paired = sorted(set(full) & set(compressed))
    only_full = sorted(set(full) - set(compressed))
    only_compressed = sorted(set(compressed) - set(full))
    if only_full or only_compressed:
        logger.warning(
            "unpaired qids: %d full-only, %d compressed-only", len(only_full), len(only_compressed)
        )
    logger.info("paired n=%d (full=%d compressed=%d)", len(paired), len(full), len(compressed))

    census: Counter = Counter()
    bundles: List[Dict[str, Any]] = []
    # prereg §3: disagreements with the harness fields are warned about AND
    # counted — the count is frozen into the manifest, not left in the log.
    divergence = {"n_metric_disagreements": 0, "n_call_disagreements": 0}
    for qid in paired:
        full_score = _score(full[qid], args.s_metric)
        compressed_score = _score(compressed[qid], args.s_metric)
        for score in (full_score, compressed_score):
            if not score["harness_metric_agrees"]:
                divergence["n_metric_disagreements"] += 1
            if not score["harness_call_agrees"]:
                divergence["n_call_disagreements"] += 1
        transition = _transition(full_score["correct"], compressed_score["correct"])
        census[transition] += 1
        if transition != "C->W":
            continue
        bundles.append(
            _bundle_row(qid, full[qid], compressed[qid], full_score, compressed_score, args)
        )

    bound = False
    if args.bind_docs:
        _bind_docs(bundles, args)
        bound = True

    bundles_path = Path(args.out_bundles)
    bundles_path.parent.mkdir(parents=True, exist_ok=True)
    with bundles_path.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")
    logger.info("Wrote %d bundles -> %s", len(bundles), bundles_path)

    manifest = _freeze_manifest(
        args,
        full_stats,
        compressed_stats,
        census,
        [bundle["qid"] for bundle in bundles],
        len(paired),
        bundles_path,
        bound,
        divergence,
    )
    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s: n_base_paired=%d transitions=%s",
        manifest_path, manifest["n_base_paired"], manifest["transitions"],
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
