"""Audit exact ACEBench input overlap against a prepared B/C corpus.

Only the pinned English ``agent_multi_step`` input surface is supported.  Its
``question`` is the initial user message presented directly to the evaluated
agent.  ``agent_multi_turn`` is deliberately rejected: that file's question is
an instruction for a separate user-simulator model, whose generated utterance
is not present in the official input file.

The audit reads neither possible answers nor generated results.  Empty exact
matches do not prove that the benchmark is uncontaminated or freeze a split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_b_training_overlap import (
    _audit_checkpoints,
    _file_identity,
    _normalize_task_id,
    _overlap_for_ids,
    _prompt_sha256,
    _read_jsonl,
    _require_mapping,
    _require_string,
    _training_inventory,
    _validate_corpus_rows,
    _validate_manifest,
    _validated_ids,
)


AUDIT_SCHEMA = "c2kv-b-training-overlap-ace-v1"
SUPPORTED_BENCHMARK = "acebench"
SUPPORTED_LANGUAGE = "en"
SUPPORTED_CATEGORY = "agent_multi_step"
EXPECTED_ROW_FIELDS = {
    "id",
    "question",
    "initial_config",
    "path",
    "function",
    "involved_classes",
}


def _validate_contract(*, benchmark: str, language: str, category: str) -> None:
    if benchmark != SUPPORTED_BENCHMARK:
        raise ValueError(f"ACE overlap audit requires benchmark={SUPPORTED_BENCHMARK!r}")
    if language != SUPPORTED_LANGUAGE:
        raise ValueError(
            f"ACE overlap audit supports only language={SUPPORTED_LANGUAGE!r}; got {language!r}"
        )
    if category == "agent_multi_turn":
        raise ValueError(
            "ACEBench agent_multi_turn is unsupported: source question is a user-simulator "
            "instruction, while the agent-visible user query is generated at runtime and is "
            "not available without reading results or running the user model"
        )
    if category != SUPPORTED_CATEGORY:
        raise ValueError(
            f"ACE overlap audit supports only category={SUPPORTED_CATEGORY!r}; got {category!r}"
        )


def _ace_prompt_units(
    path: Path,
    requested_ids: set[str],
    *,
    benchmark: str,
    language: str,
    category: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read the exact pinned multi-step input schema and hash direct user queries."""
    _validate_contract(benchmark=benchmark, language=language, category=category)
    units: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in requested_ids}
    canonical_to_id: dict[str, str] = {}
    seen_ids: set[str] = set()
    rows_seen = 0
    requested_rows = 0
    id_prefix = category + "_"

    for line_number, row in _read_jsonl(path):
        rows_seen += 1
        label = f"{path}:{line_number}"
        if set(row) != EXPECTED_ROW_FIELDS:
            missing = sorted(EXPECTED_ROW_FIELDS - set(row))
            extra = sorted(set(row) - EXPECTED_ROW_FIELDS)
            raise ValueError(
                f"{label} ACE input fields differ from the pinned schema; "
                f"missing={missing!r} extra={extra!r}"
            )

        task_id = _require_string(row.get("id"), f"{label}.id")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate ACEBench task ID: {task_id!r}")
        seen_ids.add(task_id)
        if not task_id.startswith(id_prefix):
            raise ValueError(f"{label}.id is outside ACEBench category {category!r}")
        canonical = _normalize_task_id(task_id)
        previous = canonical_to_id.setdefault(canonical, task_id)
        if previous != task_id:
            raise ValueError(
                f"ACE canonical task-ID collision: {previous!r} and {task_id!r}"
            )

        question = _require_string(row.get("question"), f"{label}.question")
        if not question.strip():
            raise ValueError(f"{label}.question must contain non-whitespace text")
        _require_mapping(row.get("initial_config"), f"{label}.initial_config")
        functions = row.get("function")
        if not isinstance(functions, list) or not functions:
            raise ValueError(f"{label}.function must be a nonempty list")
        for function_index, function in enumerate(functions):
            item = _require_mapping(function, f"{label}.function[{function_index}]")
            _require_string(item.get("name"), f"{label}.function[{function_index}].name")
        involved_classes = row.get("involved_classes")
        if (
            not isinstance(involved_classes, list)
            or not involved_classes
            or any(not isinstance(value, str) or not value for value in involved_classes)
        ):
            raise ValueError(f"{label}.involved_classes must be a nonempty string list")
        if not isinstance(row.get("path"), list):
            raise ValueError(f"{label}.path must be a list")

        if task_id not in requested_ids:
            continue
        requested_rows += 1
        units[task_id].append(
            {
                "task_id": task_id,
                "source_line": line_number,
                "field": "question",
                "surface": "agent_visible_initial_user_query",
                "prompt_sha256": _prompt_sha256(question),
            }
        )

    missing = sorted(requested_ids - {task_id for task_id, values in units.items() if values})
    if missing:
        raise ValueError(
            f"ACEBench task IDs are missing from {language}/{category}: {missing!r}"
        )
    return units, {
        **_file_identity(path),
        "benchmark": benchmark,
        "language": language,
        "category": category,
        "rows_seen": rows_seen,
        "requested_rows": requested_rows,
        "requested_task_ids": len(requested_ids),
        "row_fields": sorted(EXPECTED_ROW_FIELDS),
        "compared_text_field": "question",
        "compared_text_surface": "agent_visible_initial_user_query",
    }


def audit_ace_formal_corpus(
    corpus_dir: str | Path,
    *,
    ace_tasks_path: str | Path,
    benchmark: str,
    language: str,
    category: str,
    candidate_ids: Iterable[str],
    dev_task_ids: Iterable[str],
    checkpoint_dirs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Validate the prepared corpus and report bounded exact ACE input overlap."""
    _validate_contract(benchmark=benchmark, language=language, category=category)
    corpus_root = Path(corpus_dir)
    if not corpus_root.is_dir():
        raise ValueError(f"corpus_dir is not a directory: {corpus_root}")
    candidates = _validated_ids(candidate_ids, "candidate_ids")
    development = _validated_ids(dev_task_ids, "dev_task_ids")
    requested_ids = set(candidates) | set(development)
    if not requested_ids:
        raise ValueError("candidate_ids and dev_task_ids are both empty")

    manifest, corpus_identity, sessions_path, decisions_path = _validate_manifest(corpus_root)
    sessions, prefixes = _validate_corpus_rows(
        manifest, sessions_path, decisions_path
    )
    prompt_locations, task_id_locations, source_inventory, training_prefixes = (
        _training_inventory(sessions, prefixes)
    )
    ace_units, ace_source = _ace_prompt_units(
        Path(ace_tasks_path),
        requested_ids,
        benchmark=benchmark,
        language=language,
        category=category,
    )
    candidate_overlap = _overlap_for_ids(
        candidates, ace_units, prompt_locations, task_id_locations
    )
    development_overlap = _overlap_for_ids(
        development, ace_units, prompt_locations, task_id_locations
    )
    checkpoint_result = _audit_checkpoints(
        checkpoint_dirs, corpus_identity["manifest"]["sha256"]
    )

    dev_set = set(development)
    id_matched = set(candidate_overlap["canonical_exact_task_ids"]["matched_task_ids"])
    prompt_matched = set(
        candidate_overlap["normalized_exact_user_prompts"]["matched_task_ids"]
    )
    exclusions = (set(candidates) & dev_set) | id_matched | prompt_matched
    status = "checkpoint_bound" if checkpoint_result["status"] == "bound" else "corpus_only"
    exact_match_result = (
        "exact_match_found"
        if candidate_overlap["interpretation"] == "exact_match_found"
        else "no_exact_match"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": status,
        "formal_split_frozen": False,
        "interpretation": exact_match_result,
        "benchmark": {
            "name": benchmark,
            "language": language,
            "category": category,
        },
        "validation": {
            "status": "passed",
            "corpus_contract": "passed",
            "checkpoint_contract": (
                "passed" if checkpoint_result["status"] == "bound" else "not_provided"
            ),
            "ace_input_contract": "passed",
        },
        "normalization": {
            "user_prompt_exact": "lowercase; collapse regex \\s+ to one ASCII space; strip",
            "canonical_task_id_exact": (
                "Unicode NFKC; lowercase; replace each non-[0-9a-z] run with underscore; "
                "strip underscores; compare the complete normalized string without alias stripping"
            ),
            "near_or_semantic_overlap": "not_computed",
        },
        "scope": {
            "formal_readiness": "not_determined",
            "contamination_status": "not_determined",
            "near_or_semantic_overlap": "not_computed",
            "raw_prompts_or_tokens_stored": False,
            "population_interpretation": (
                "Prepared training population upper bound; checkpoint metadata binding "
                "is recorded separately; not an actual trained minibatch or exposure census"
            ),
            "ace_input_interpretation": {
                "question": (
                    "Compared as the direct initial user message presented to the evaluated "
                    "agent by the pinned English agent_multi_step runner"
                ),
                "id": "Compared only as a complete normalized task identity",
                "initial_config": (
                    "Validated as execution-simulator input; not compared with training user messages"
                ),
                "function": (
                    "Validated as agent API-documentation input; not compared with training user messages"
                ),
                "involved_classes": (
                    "Validated as simulator selection input; not compared with training user messages"
                ),
                "path": (
                    "Validated as a pinned-schema field; not compared with training user messages"
                ),
                "agent_multi_turn": (
                    "unsupported because source question is a user-simulator instruction, not an "
                    "agent-visible user query"
                ),
            },
            "claim_limit": (
                "This exact-overlap audit neither proves absence of contamination nor freezes a "
                "formal evaluation split"
            ),
        },
        "corpus": corpus_identity,
        "checkpoint_binding": checkpoint_result,
        "ace_source": ace_source,
        "training_prefixes": training_prefixes,
        "source_alias_diagnostic": {
            "scope": "bounded_observed_labels_only",
            "fields": ["source", "template_id", "source_metadata.benchmark"],
            "used_as_overlap_filter": False,
            "inventory": source_inventory,
        },
        "overlap": {
            "candidate": candidate_overlap,
            "development": development_overlap,
        },
        "excluded_candidate_ids": sorted(exclusions),
        "eligible_candidate_ids": sorted(set(candidates) - exclusions),
    }


__all__ = ["audit_ace_formal_corpus"]
