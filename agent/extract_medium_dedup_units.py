"""Extract dedup text-units for the G-medium cross-dataset dedup pass.

One clean, idempotent replacement for the one-off server scripts
(extract_medium_dedup.py + fix_extract_medium_dedup.py).  Writes up to four
jsonl intermediates under ``--out_dir`` (each section is skipped when its
input path is not given):

- ``v2eval_sessions.jsonl``   {"session_id", "spans": <json string>}
  agent-llm-traces-v2 EVAL sessions (per the split manifest), for the
  messages-unit dedup's eval side (traces flattener shape).
- ``v2eval_msgs_raw.jsonl``   {"_id": session_id, "text": content}
  the same sessions' message texts, flattened with
  ``dedup_cross_dataset._message_text`` (handles the gen_ai ``parts`` shape —
  the first extractor wrote zero rows because it only looked at ``content``).
- ``qa_docs_raw.jsonl``       {"_id": <FULL qid>, "text": doc_text}
  QA train-side raw units.  The ``_id`` IS the example qid the joint loader
  will produce (P1-5: removal lists then match the planner pool by exact qid
  equality — ``dedup_cross_dataset._record_id`` reads ``_id`` and the planner
  removal matcher takes the full qid):
    * hotpotqa: ``qa:hotpotqa:<_id>``  — one unit per history document;
    * 2wiki:    ``qa:2wiki:<_id>``     — one unit per flattened context entry
      (via ``wiki2_row_to_example``, so unit text is byte-identical to the
      training document);
    * longmagpie: ``qa:longmagpie:<shard_stem>:<row_in_shard>`` — one unit
      with the FULL user content per row; skipped rows (no trailing "?")
      still consume their ``row_in_shard``, exactly like the loader.
  Rows the QA converter rejects (missing question/answer/docs) produce no
  units for hotpotqa/2wiki — they are never trained on, so there is nothing
  to remove.
- ``openswe_resolved_msgs.jsonl``  {"trajectory_id", "messages": <json string>}
  Open-SWE resolved==1 trajectories flattened to user/assistant message texts
  (assistant tool-call argument strings are appended as text lines), for the
  messages-unit dedup's train side.

Determinism / idempotence: sorted file globs, fixed iteration order, no
wall-clock fields — same inputs produce byte-identical outputs, and re-running
simply overwrites the four files.

Usage:
  python agent/extract_medium_dedup_units.py \
      --traces_v2_dir ~/c2kv/datasets/agent-llm-traces-v2 \
      --split_manifest_file ~/c2kv/outputs_lyc/g_joint/taskproxy_disjoint_v2.json \
      --split_manifest_name taskproxy_disjoint \
      --openswe_dir ~/c2kv/datasets/open-swe-traces \
      --qa_hotpotqa_path ~/c2kv/datasets/qa/hotpotqa_train.jsonl \
      --qa_2wiki_path ~/c2kv/datasets/qa/xanhho_2WikiMultihopQA/train.parquet \
      --qa_longmagpie_path ~/c2kv/datasets/qa/longmagpie_raw \
      --out_dir ~/c2kv/outputs_lyc/g_joint/dedup_units
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

if __package__ in {None, ""}:
    # Allow running as `python agent/extract_medium_dedup_units.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dedup_cross_dataset import _json_loads, _message_text  # noqa: E402
from train.train_data_joint_multisource import (  # noqa: E402
    hotpotqa_row_to_example,
    wiki2_row_to_example,
)

logger = logging.getLogger(__name__)


def _iter_parquet_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Whole-file parquet read (nested columns break iter_batches)."""
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except Exception:
        table = pq.ParquetFile(path).read()
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def _sorted_parquet_files(path: Optional[str], patterns: tuple) -> List[str]:
    if not path:
        return []
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(path, pattern), recursive=True))
    return sorted(set(files))


def _load_eval_session_ids(manifest_file: str, manifest_name: str) -> set:
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    selected = (
        manifest
        if "eval_session_ids" in manifest
        else manifest[manifest_name]
    )
    return {str(item) for item in selected["eval_session_ids"]}


# ---------------------------------------------------------------------------
# Section 1+2: traces-v2 eval sessions (messages-unit shape + raw messages).
# ---------------------------------------------------------------------------


def extract_v2eval(
    traces_v2_dir: str,
    eval_ids: set,
    out_dir: Path,
) -> Dict[str, int]:
    sessions_path = out_dir / "v2eval_sessions.jsonl"
    raw_path = out_dir / "v2eval_msgs_raw.jsonl"
    n_sessions = n_raw = 0
    files = _sorted_parquet_files(traces_v2_dir, ("data/**/*.parquet", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {traces_v2_dir}")
    with sessions_path.open("w", encoding="utf-8") as fs, raw_path.open("w", encoding="utf-8") as fr:
        for file in files:
            for row in _iter_parquet_rows(Path(file)):
                session_id = str(
                    row.get("session_id") or row.get("trace_id") or row.get("id") or ""
                )
                if not session_id or session_id not in eval_ids:
                    continue
                spans = _json_loads(row.get("spans"), row.get("spans")) or []
                fs.write(
                    json.dumps(
                        {"session_id": session_id, "spans": json.dumps(spans, ensure_ascii=False)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_sessions += 1
                for span in spans:
                    attributes = _json_loads((span or {}).get("attributes"), {}) or {}
                    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
                        for message in _json_loads(attributes.get(key), []) or []:
                            text = _message_text(message)
                            if text.strip():
                                fr.write(
                                    json.dumps({"_id": session_id, "text": text}, ensure_ascii=False)
                                    + "\n"
                                )
                                n_raw += 1
    logger.info("v2eval: %d sessions, %d raw messages", n_sessions, n_raw)
    return {"v2eval_sessions": n_sessions, "v2eval_msgs_raw": n_raw}


# ---------------------------------------------------------------------------
# Section 3: QA raw document units (ids are the loader's full qids).
# ---------------------------------------------------------------------------


def extract_qa_docs(
    hotpotqa_path: Optional[str],
    wiki2_path: Optional[str],
    longmagpie_path: Optional[str],
    out_dir: Path,
) -> Dict[str, int]:
    qa_path = out_dir / "qa_docs_raw.jsonl"
    counts = {"hotpotqa": 0, "2wiki": 0, "longmagpie": 0}

    def _write_units(handle, qid: str, texts: List[str]) -> int:
        written = 0
        for text in texts:
            if isinstance(text, str) and text.strip():
                handle.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
                written += 1
        return written

    with qa_path.open("w", encoding="utf-8") as fq:
        if hotpotqa_path:
            with Path(hotpotqa_path).open("r", encoding="utf-8") as fh:
                for row_index, line in enumerate(fh):
                    if not line.strip():
                        continue
                    row = _json_loads(line, None)
                    if not isinstance(row, dict):
                        continue
                    example = hotpotqa_row_to_example(row, row_index)
                    if example is None:
                        continue
                    counts["hotpotqa"] += _write_units(fq, example.qid, example.history_documents)
        for file in _sorted_parquet_files(wiki2_path, ("*.parquet",)):
            for row_index, row in enumerate(_iter_parquet_rows(Path(file))):
                example = wiki2_row_to_example(row, row_index)
                if example is None:
                    continue
                counts["2wiki"] += _write_units(fq, example.qid, example.history_documents)
        for file in _sorted_parquet_files(longmagpie_path, ("data/*.parquet", "*.parquet")):
            shard = Path(file).stem
            for row_in_shard, row in enumerate(_iter_parquet_rows(Path(file))):
                # row_in_shard counts EVERY row (the loader's skip rule runs
                # later); the unit is the full user content (context+question).
                messages = _json_loads(row.get("messages"), []) or []
                user = next(
                    (m for m in messages if isinstance(m, dict) and m.get("role") == "user"),
                    messages[0] if messages and isinstance(messages[0], dict) else None,
                )
                text = (user or {}).get("content") or ""
                if isinstance(text, str) and text.strip():
                    fq.write(
                        json.dumps(
                            {"_id": f"qa:longmagpie:{shard}:{row_in_shard}", "text": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counts["longmagpie"] += 1
    logger.info("qa raw docs: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Section 4: Open-SWE resolved trajectories (messages-unit shape).
# ---------------------------------------------------------------------------


def extract_openswe(openswe_dir: str, out_dir: Path) -> Dict[str, int]:
    out_path = out_dir / "openswe_resolved_msgs.jsonl"
    n_out = 0
    files = _sorted_parquet_files(openswe_dir, ("data/*/*.parquet", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {openswe_dir}")
    with out_path.open("w", encoding="utf-8") as fo:
        for file in files:
            for row in _iter_parquet_rows(Path(file)):
                if row.get("resolved") != 1:
                    continue
                trajectory = _json_loads(row.get("trajectory"), []) or []
                kept: List[Dict[str, Any]] = []
                for message in trajectory:
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role")
                    if role == "user":
                        kept.append({"role": "user", "content": message.get("content")})
                    elif role == "assistant":
                        parts = [message.get("content") or ""]
                        for call in _json_loads(message.get("tool_calls"), []) or []:
                            function = (call or {}).get("function") or {}
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                parts.append(arguments)
                        kept.append(
                            {"role": "assistant", "content": "\n".join(p for p in parts if p)}
                        )
                trajectory_id = str(row.get("trajectory_id") or row.get("instance_id") or "")
                if trajectory_id and kept:
                    fo.write(
                        json.dumps(
                            {
                                "trajectory_id": trajectory_id,
                                "messages": json.dumps(kept, ensure_ascii=False),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_out += 1
    logger.info("openswe resolved trajectories: %d", n_out)
    return {"openswe_resolved": n_out}


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--traces_v2_dir", default=None, help="agent-llm-traces-v2 root (sections 1-2)")
    parser.add_argument("--split_manifest_file", default=None, help="split manifest with eval_session_ids (sections 1-2)")
    parser.add_argument("--split_manifest_name", default="taskproxy_disjoint", help="manifest block name when train/eval ids are not top-level")
    parser.add_argument("--openswe_dir", default=None, help="Open-SWE-Traces root (section 4)")
    parser.add_argument("--qa_hotpotqa_path", default=None, help="hotpotqa train jsonl (section 3)")
    parser.add_argument("--qa_2wiki_path", default=None, help="2wiki parquet file or dir (section 3)")
    parser.add_argument("--qa_longmagpie_path", default=None, help="longmagpie root (data/*.parquet) (section 3)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    if args.traces_v2_dir:
        if not args.split_manifest_file:
            raise ValueError("--traces_v2_dir needs --split_manifest_file")
        eval_ids = _load_eval_session_ids(args.split_manifest_file, args.split_manifest_name)
        logger.info("eval session ids: %d", len(eval_ids))
        summary.update(extract_v2eval(args.traces_v2_dir, eval_ids, out_dir))
    if any([args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path]):
        summary.update(
            extract_qa_docs(
                args.qa_hotpotqa_path, args.qa_2wiki_path, args.qa_longmagpie_path, out_dir
            )
        )
    if args.openswe_dir:
        summary.update(extract_openswe(args.openswe_dir, out_dir))
    if not summary:
        raise ValueError("nothing to extract: pass at least one input path")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
