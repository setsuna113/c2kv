"""Audit B training-population overlap without freezing a formal test split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit_b_training_overlap import audit_formal_corpus
from .audit_b_training_overlap_ace import audit_ace_formal_corpus


def read_task_ids(path: Path) -> tuple[list[str], dict]:
    """Accept the ID lists, development manifests and category maps used by pilots."""
    payload = path.read_bytes()
    value = json.loads(payload)
    if isinstance(value, list):
        ids = value
    elif isinstance(value, dict) and isinstance(value.get("ids"), list):
        ids = value["ids"]
        if "n_total" in value and value["n_total"] != len(ids):
            raise ValueError(f"{path}: declared n_total differs from task-ID count")
    elif isinstance(value, dict) and all(isinstance(items, list) for items in value.values()):
        ids = [task_id for items in value.values() for task_id in items]
    else:
        raise ValueError(f"{path}: expected a task-ID list, development manifest, or BFCL category map")
    if not all(isinstance(task_id, str) and task_id for task_id in ids):
        raise ValueError(f"{path}: task IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate task IDs")
    identity = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "task_count": len(ids)}
    if isinstance(value, dict) and "source_sha256" in value:
        identity["declared_question_source_sha256"] = value["source_sha256"]
    return ids, identity


def read_ace_task_ids(
    path: Path, *, benchmark: str, language: str, category: str
) -> tuple[list[str], dict]:
    """Read an ACE selection and bind any declared source metadata to the CLI."""
    ids, identity = read_task_ids(path)
    value = json.loads(path.read_bytes())
    if isinstance(value, dict):
        expected = {
            "benchmark": benchmark,
            "language": language,
            "category": category,
        }
        for field, wanted in expected.items():
            if field in value and value[field] != wanted:
                raise ValueError(
                    f"{path}: declared {field}={value[field]!r}; expected {wanted!r}"
                )
            if field in value:
                identity[f"declared_{field}"] = value[field]
        if "formal_split_frozen" in value:
            identity["declared_formal_split_frozen"] = value["formal_split_frozen"]
    return ids, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("bfcl", "acebench"), default="bfcl",
                        help="Benchmark input contract; default preserves the BFCL CLI")
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="Prepared B corpus directory containing manifest.json and its bound JSONL files")
    parser.add_argument("--bfcl-tasks", type=Path,
                        help="Official BFCL question JSONL; do not supply the possible_answer file")
    parser.add_argument("--acebench-tasks", type=Path,
                        help="Official ACEBench input JSONL; do not supply possible answers or results")
    parser.add_argument("--acebench-language",
                        help="ACEBench language; the exact adapter currently requires en")
    parser.add_argument("--acebench-category",
                        help="ACEBench category; the exact adapter currently requires agent_multi_step")
    parser.add_argument("--candidate-ids", type=Path, required=True,
                        help="Prospective task-ID JSON list, development manifest or BFCL category map")
    parser.add_argument("--dev-exclusion-ids", type=Path, action="append", required=True,
                        help="Development-exposure ID file; repeat to union several files")
    parser.add_argument("--checkpoint-dir", type=Path, action="append", default=[],
                        help="Returned event-native checkpoint metadata; repeat for a B/C pair")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON audit artifact; existing artifacts are not overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        if args.benchmark == "bfcl":
            if args.bfcl_tasks is None:
                raise ValueError("--bfcl-tasks is required when --benchmark=bfcl")
            if any(value is not None for value in (
                    args.acebench_tasks, args.acebench_language, args.acebench_category)):
                raise ValueError("ACEBench arguments cannot be used when --benchmark=bfcl")
            benchmark_tasks = args.bfcl_tasks
        else:
            if args.bfcl_tasks is not None:
                raise ValueError("--bfcl-tasks cannot be used when --benchmark=acebench")
            if args.acebench_tasks is None:
                raise ValueError("--acebench-tasks is required when --benchmark=acebench")
            if args.acebench_language is None:
                raise ValueError("--acebench-language is required when --benchmark=acebench")
            if args.acebench_category is None:
                raise ValueError("--acebench-category is required when --benchmark=acebench")
            benchmark_tasks = args.acebench_tasks
        if args.benchmark == "bfcl":
            task_id_reader = read_task_ids
        else:
            task_id_reader = lambda path: read_ace_task_ids(
                path,
                benchmark="acebench",
                language=args.acebench_language,
                category=args.acebench_category,
            )
        candidate_ids, candidate_source = task_id_reader(args.candidate_ids)
        dev_ids, dev_sources = set(), []
        for path in args.dev_exclusion_ids:
            ids, source = task_id_reader(path)
            dev_ids.update(ids)
            dev_sources.append(source)
        question_bytes = benchmark_tasks.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        question_sha = hashlib.sha256(question_bytes).hexdigest()
        for source in [candidate_source, *dev_sources]:
            if ("declared_question_source_sha256" in source
                    and source["declared_question_source_sha256"] != question_sha):
                benchmark_label = "BFCL" if args.benchmark == "bfcl" else "ACEBench"
                raise ValueError(
                    f"{source['path']}: declared {benchmark_label} question-source identity differs"
                )
        if args.benchmark == "bfcl":
            result = audit_formal_corpus(
                args.corpus_dir, bfcl_tasks_path=benchmark_tasks,
                candidate_ids=candidate_ids, dev_task_ids=sorted(dev_ids),
                checkpoint_dirs=args.checkpoint_dir,
            )
            selection_scope = "Prospective exact-overlap exclusions; no formal split is frozen"
        else:
            result = audit_ace_formal_corpus(
                args.corpus_dir,
                ace_tasks_path=benchmark_tasks,
                benchmark="acebench",
                language=args.acebench_language,
                category=args.acebench_category,
                candidate_ids=candidate_ids,
                dev_task_ids=sorted(dev_ids),
                checkpoint_dirs=args.checkpoint_dir,
            )
            selection_scope = (
                "Prospective ACEBench exact-overlap exclusions; no contamination conclusion "
                "and no formal split is frozen"
            )
        result["input_selection"] = {"candidate_ids": candidate_source, "dev_exclusions": dev_sources,
                                     "dev_union_task_count": len(dev_ids),
                                     "selection_scope": selection_scope}
        assert result["formal_split_frozen"] is False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation keeps an existing evidence artifact intact.
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"status": result["status"], "interpretation": result["interpretation"],
                      "eligible_candidates": len(result["eligible_candidate_ids"]),
                      "excluded_candidates": len(result["excluded_candidate_ids"]),
                      "formal_split_frozen": False, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
