"""Exp 1 evaluation: every tool-allocation layout of one frozen manifest.

Gist layouts run through the event-native generator of the T0 checkpoint
(``t1`` through the T1 checkpoint).  Eviction layouts reuse the ``full``
records: the same frozen base weights prefill the full native prefix and
:mod:`exp1_evict` prunes the catalog remainder to the gist budget of the
matching T0 layout.  Every row records the resident prompt positions, and
layouts are compared with paired exact tests on the shared decisions.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from history_memory.runtime import PreparedDecision

from .common import deserialize_memory, sha256_file
from .exp1_tools import (
    EVICTION_LAYOUTS,
    EXP1_PURPOSE,
    EXP1_RATIOS,
    EXP1_SCHEMA,
    GIST_LAYOUTS,
    paired_test,
    score_rows,
    validate_exp1_record,
)
from .inference import _decode, load_next_checkpoint

EVAL_SCHEMA = "next-compression-exp1-eval-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PAIRED_COMPARISONS = (
    ("hybrid", "uniform"),
    ("hybrid", "retrieval"),
    ("hybrid", "random"),
    ("hybrid", "full"),
    ("hybrid", "snapkv_hybrid"),
    ("hybrid", "h2o_hybrid"),
    ("t1", "hybrid"),
    ("t1", "uniform"),
    ("uniform", "snapkv"),
    ("uniform", "h2o"),
)
PAIRED_KEYS = ("call_correct", "name_correct", "false_call")


@dataclass(frozen=True)
class EvictionSettings:
    obs_window: int = 16
    kernel: int = 7
    recent_fraction: float = 0.5
    chunk_size: int = 2048


@dataclass(frozen=True)
class Exp1Record:
    decision_id: str
    session_key: str
    source: str
    layout: str
    k: int | None
    ratio: int
    memory: Any
    target_ids: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def cell(self) -> str:
        return cell_name(self.layout, self.k, self.ratio)


@dataclass(frozen=True)
class Exp1Corpus:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    records: tuple[Exp1Record, ...]

    def select(self, layout: str, *, k: int | None = None, ratio: int | None = None):
        return [
            record
            for record in self.records
            if record.layout == layout
            and (k is None or record.k == k)
            and (ratio is None or record.ratio == ratio)
        ]


def cell_name(layout: str, k: int | None, ratio: int) -> str:
    return f"{layout}.k{k}.ratio{ratio}"


def load_exp1_corpus(root: str | Path, *, tokenizer: Any) -> Exp1Corpus:
    from history_memory.preparation import tokenizer_identity

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXP1_SCHEMA or manifest.get("purpose") != EXP1_PURPOSE:
        raise ValueError("Not an Exp 1 tool manifest")
    if manifest.get("tokenizer", {}).get("sha256") != tokenizer_identity(tokenizer)["sha256"]:
        raise ValueError("Exp 1 manifest tokenizer differs from checkpoint tokenizer")
    info = manifest["records"]
    records_path = (manifest_path.parent / str(info["path"])).resolve()
    if records_path.parent != manifest_path.parent or not records_path.is_file():
        raise ValueError("Exp 1 records must sit next to the manifest")
    if records_path.stat().st_size != info["bytes"] or sha256_file(records_path) != info["sha256"]:
        raise ValueError("Exp 1 records hash/size differs from manifest")
    vocab_size = len(tokenizer)
    records: list[Exp1Record] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            validate_exp1_record(raw)
            memory = deserialize_memory(raw["memory"])
            for tokens in (raw["target_ids"], memory.system_input_ids, memory.workspace_input_ids):
                if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
                    raise ValueError(f"Exp 1 row {line_number} has an out-of-vocabulary token")
            records.append(
                Exp1Record(
                    decision_id=raw["decision_id"],
                    session_key=raw["session_key"],
                    source=raw["source"],
                    layout=raw["layout"],
                    k=raw["k"],
                    ratio=int(raw["ratio"]),
                    memory=memory,
                    target_ids=tuple(raw["target_ids"]),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
    if len(records) != info["count"]:
        raise ValueError("Exp 1 record count differs from manifest")
    if not records:
        raise ValueError("Exp 1 corpus is empty")
    return Exp1Corpus(manifest_path, sha256_file(manifest_path), manifest, tuple(records))


def _eos_ids(tokenizer: Any) -> tuple[int, ...]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != getattr(tokenizer, "unk_token_id", None):
        ids.append(im_end)
    if not ids:
        raise ValueError("Tokenizer exposes no end-of-sequence token")
    return tuple(dict.fromkeys(ids))


def _row_base(record: Exp1Record, tokenizer: Any) -> dict[str, Any]:
    return {
        "decision_id": record.decision_id,
        "session_key": record.session_key,
        "source": record.source,
        "layout": record.layout,
        "k": record.k,
        "ratio": record.ratio,
        "cell": record.cell,
        "decision_type": record.metadata.get("decision_type"),
        "gold_tool_calls": record.metadata.get("gold_tool_calls", []),
        "target_token_ids": list(record.target_ids),
        "target_text": _decode(tokenizer, record.target_ids),
        "native_tool_indices": record.metadata.get("native_tool_indices"),
        "selection_hit": record.metadata.get("selection_hit"),
    }


def run_gist_layouts(
    generator: Any,
    tokenizer: Any,
    records: Sequence[Exp1Record],
    *,
    max_new_tokens: int,
    compute_uniform_ce: bool,
    phase: str,
) -> list[dict[str, Any]]:
    """Generate every record through the event-native generator, ratio-major."""
    import torch

    ordered = sorted(records, key=lambda record: (record.ratio, record.cell, record.decision_id))
    rows: list[dict[str, Any]] = []
    active_ratio = None
    for record in ordered:
        if active_ratio is not None and record.ratio != active_ratio:
            generator.close_session()
        active_ratio = record.ratio
        with generator.decision_scope(session_id=f"{record.cell}:{record.session_key}"):
            generation = generator.generate(
                record.memory,
                ratio=record.ratio,
                max_new_tokens=max_new_tokens,
                trace_context={
                    "session_id": record.session_key,
                    "decision_key": f"{record.cell}:{record.decision_id}",
                    "phase": phase,
                },
            )
        uniform_ce = None
        if compute_uniform_ce:
            decision = PreparedDecision(
                memory=record.memory,
                target_ids=record.target_ids,
                ratio=record.ratio,
                weight=1.0,
                decision_id=record.decision_id,
                target_weights=None,
            )
            # Match generation/training compute dtype: FP32 gist projections
            # under a BF16 load need autocast (see next_compression.inference).
            with torch.inference_mode(), generator._base_autocast():
                value = generator.runtime((decision,))["loss"].detach().float().item()
            if not math.isfinite(value):
                raise ValueError(f"Non-finite uniform CE for {record.cell}:{record.decision_id}")
            uniform_ce = value
        costs = record.memory.costs(record.ratio)
        rows.append(
            {
                **_row_base(record, tokenizer),
                "generated_token_ids": list(generation.token_ids),
                "generated_text": _decode(tokenizer, generation.token_ids),
                "finish_reason": generation.finish_reason,
                "uniform_ce": uniform_ce,
                "resident_kv_tokens": costs["resident_kv_tokens"],
                "gist_tokens": costs["gist_tokens"],
                "system_tokens": costs["system_tokens"],
                "raw_tokens": costs["raw_tokens"],
            }
        )
    generator.close_session()
    return rows


def _budget_source(corpus: Exp1Corpus, layout: str, *, k: int | None, ratio: int) -> dict[str, Exp1Record]:
    records = corpus.select(layout, k=k, ratio=ratio)
    if not records:
        raise ValueError(f"Eviction budgets need {cell_name(layout, k, ratio)} records")
    return {record.decision_id: record for record in records}


def run_eviction_layouts(
    model: Any,
    tokenizer: Any,
    corpus: Exp1Corpus,
    *,
    layouts: Sequence[str],
    ratios: Sequence[int],
    hybrid_k: int,
    max_new_tokens: int,
    settings: EvictionSettings,
) -> list[dict[str, Any]]:
    """Prefill the ``full`` prompt, evict the catalog remainder, generate."""
    import torch

    from .exp1_evict import evictable_mask, generate_with_eviction

    eos = _eos_ids(tokenizer)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        full_records = corpus.select("full", ratio=ratio)
        uniform = _budget_source(corpus, "uniform", k=None, ratio=ratio) if any(
            layout in ("snapkv", "h2o") for layout in layouts
        ) else {}
        hybrid = _budget_source(corpus, "hybrid", k=hybrid_k, ratio=ratio) if any(
            layout.endswith("_hybrid") for layout in layouts
        ) else {}
        for record in sorted(full_records, key=lambda item: item.decision_id):
            spans = [tuple(span) for span in record.metadata["tool_token_spans"]]
            region = (spans[0][0], spans[-1][1])
            prompt = tuple(record.memory.system_input_ids) + tuple(record.memory.workspace_input_ids)
            for layout in layouts:
                method = layout.split("_")[0]
                if layout.endswith("_hybrid"):
                    budget_record = hybrid[record.decision_id]
                    protected = list(budget_record.metadata["native_tool_indices"])
                    k = hybrid_k
                else:
                    budget_record = uniform[record.decision_id]
                    protected = []
                    k = None
                keep = int(budget_record.memory.costs(ratio)["gist_tokens"])
                mask = evictable_mask(region, spans, protected)
                with torch.inference_mode():
                    generation = generate_with_eviction(
                        model,
                        prompt,
                        region=region,
                        evictable=mask,
                        keep=keep,
                        method=method,
                        max_new_tokens=max_new_tokens,
                        eos_ids=eos,
                        obs_window=settings.obs_window,
                        kernel=settings.kernel,
                        recent_fraction=settings.recent_fraction,
                        chunk_size=settings.chunk_size,
                    )
                shadow = Exp1Record(
                    decision_id=record.decision_id,
                    session_key=record.session_key,
                    source=record.source,
                    layout=layout,
                    k=k,
                    ratio=ratio,
                    memory=record.memory,
                    target_ids=record.target_ids,
                    metadata={
                        **record.metadata,
                        "native_tool_indices": protected,
                        "selection_hit": budget_record.metadata.get("selection_hit"),
                    },
                )
                rows.append(
                    {
                        **_row_base(shadow, tokenizer),
                        "generated_token_ids": list(generation.token_ids),
                        "generated_text": _decode(tokenizer, generation.token_ids),
                        "finish_reason": generation.finish_reason,
                        "uniform_ce": None,
                        "resident_kv_tokens": generation.kept_tokens,
                        "gist_tokens": 0,
                        "system_tokens": len(record.memory.system_input_ids) - (
                            generation.evictable_tokens - generation.keep
                        ),
                        "raw_tokens": len(record.memory.workspace_input_ids),
                        "eviction": {
                            "method": method,
                            "budget_cell": budget_record.cell,
                            "keep": generation.keep,
                            "evictable_tokens": generation.evictable_tokens,
                            "prompt_len": generation.prompt_len,
                            "region": list(generation.region),
                            "prefill_chunks": generation.prefill_chunks,
                        },
                    }
                )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], *, hybrid_k: int) -> dict[str, Any]:
    """Per-cell metrics plus paired exact tests at the default k."""
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    metrics: dict[str, Any] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for cell, cell_rows in sorted(by_cell.items()):
        scored = score_rows(cell_rows)
        outcomes[cell] = scored.pop("outcomes")
        metrics[cell] = scored
    paired: list[dict[str, Any]] = []
    for ratio in EXP1_RATIOS:
        for left, right in PAIRED_COMPARISONS:
            left_cell = cell_name(left, hybrid_k if left in ("hybrid", "random", "retrieval", "snapkv_hybrid", "h2o_hybrid") else None, ratio)
            right_cell = cell_name(right, hybrid_k if right in ("hybrid", "random", "retrieval", "snapkv_hybrid", "h2o_hybrid") else None, ratio)
            if left_cell not in outcomes or right_cell not in outcomes:
                continue
            for key in PAIRED_KEYS:
                result = paired_test(outcomes[left_cell], outcomes[right_cell], key=key)
                if result["n"] == 0:
                    continue
                paired.append({"left": left_cell, "right": right_cell, "ratio": ratio, **result})
    return {"metrics": metrics, "paired": paired}


def _code_hashes() -> dict[str, str]:
    files = [
        REPOSITORY_ROOT / "python" / "next_compression" / "exp1_tools.py",
        REPOSITORY_ROOT / "python" / "next_compression" / "exp1_evict.py",
        REPOSITORY_ROOT / "python" / "next_compression" / "exp1_eval.py",
        REPOSITORY_ROOT / "python" / "next_compression" / "tools.py",
        REPOSITORY_ROOT / "python" / "next_compression" / "inference.py",
        REPOSITORY_ROOT / "python" / "next_compression" / "selection.py",
        REPOSITORY_ROOT / "python" / "history_memory" / "inference.py",
        REPOSITORY_ROOT / "python" / "history_memory" / "runtime.py",
        REPOSITORY_ROOT / "metrology" / "kv_compress.py",
    ]
    return {
        str(path.relative_to(REPOSITORY_ROOT)).replace(os.sep, "/"): sha256_file(path)
        for path in files
        if path.is_file()
    }


def _checkpoint_contract(checkpoint: Path, profile: Mapping[str, Any], *, hash_weights: bool) -> dict[str, Any]:
    contract = {
        "path": str(checkpoint),
        "variant": profile["variant"],
        "config_sha256": profile["config_sha256"],
        "corpus_identity": profile.get("corpus_identity"),
        "initialization_id": profile.get("initialization_id"),
    }
    trainer_state = checkpoint / "trainer_state.json"
    if trainer_state.is_file():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        contract["global_step"] = state.get("global_step")
        contract["parameter_version"] = state.get("parameter_version")
    if hash_weights:
        weights = sorted(checkpoint.glob("*.safetensors"))
        contract["weights_sha256"] = {path.name: sha256_file(path) for path in weights}
    return contract


def evaluate_exp1(
    *,
    manifest_root: str | Path,
    checkpoint_t0: str | Path,
    checkpoint_t1: str | Path | None = None,
    layouts: Sequence[str] = GIST_LAYOUTS + EVICTION_LAYOUTS,
    ratios: Sequence[int] = EXP1_RATIOS,
    hybrid_k: int = 3,
    max_new_tokens: int = 512,
    device: str = "cuda",
    dtype: str = "bfloat16",
    eviction: EvictionSettings = EvictionSettings(),
    compute_uniform_ce: bool = True,
    hash_weights: bool = False,
) -> dict[str, Any]:
    if tuple(ratios) != EXP1_RATIOS:
        raise ValueError("Exp 1 is frozen to ratios 8 and 12")
    unknown = [layout for layout in layouts if layout not in GIST_LAYOUTS + EVICTION_LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown layouts: {unknown}")
    if "t1" in layouts and checkpoint_t1 is None:
        raise ValueError("The t1 layout needs --checkpoint-t1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    import torch
    import transformers

    started = _dt.datetime.now(_dt.timezone.utc)
    checkpoint_t0 = Path(checkpoint_t0).resolve()
    generator, tokenizer, profile_t0 = load_next_checkpoint(
        checkpoint_t0, device=device, dtype=dtype, decode_strategy="incremental"
    )
    if profile_t0["variant"] != "T0":
        raise ValueError("--checkpoint-t0 must be a T0 checkpoint")
    corpus = load_exp1_corpus(manifest_root, tokenizer=tokenizer)

    rows: list[dict[str, Any]] = []
    gist_t0 = [record for record in corpus.records if record.layout in layouts and record.layout != "t1"]
    if gist_t0:
        rows.extend(
            run_gist_layouts(
                generator, tokenizer, gist_t0,
                max_new_tokens=max_new_tokens, compute_uniform_ce=compute_uniform_ce, phase="exp1_gist_t0",
            )
        )
    eviction_layouts = [layout for layout in layouts if layout in EVICTION_LAYOUTS]
    if eviction_layouts:
        rows.extend(
            run_eviction_layouts(
                generator.runtime.base_model, tokenizer, corpus,
                layouts=eviction_layouts, ratios=ratios, hybrid_k=hybrid_k,
                max_new_tokens=max_new_tokens, settings=eviction,
            )
        )
    contracts = {"t0": _checkpoint_contract(checkpoint_t0, profile_t0, hash_weights=hash_weights)}
    if "t1" in layouts:
        del generator
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        checkpoint_t1 = Path(checkpoint_t1).resolve()
        generator_t1, tokenizer_t1, profile_t1 = load_next_checkpoint(
            checkpoint_t1, device=device, dtype=dtype, decode_strategy="incremental"
        )
        if profile_t1["variant"] != "T1":
            raise ValueError("--checkpoint-t1 must be a T1 checkpoint")
        if tokenizer_t1.get_vocab() != tokenizer.get_vocab():
            raise ValueError("T1 tokenizer differs from T0 tokenizer")
        rows.extend(
            run_gist_layouts(
                generator_t1, tokenizer_t1, corpus.select("t1"),
                max_new_tokens=max_new_tokens, compute_uniform_ce=compute_uniform_ce, phase="exp1_gist_t1",
            )
        )
        contracts["t1"] = _checkpoint_contract(checkpoint_t1, profile_t1, hash_weights=hash_weights)

    summary = summarize(rows, hybrid_k=hybrid_k)
    return {
        "schema": EVAL_SCHEMA,
        "status": "completed",
        "reporting": "preliminary, n=1",
        "scope": "Recorded-decision action proxy on a session-disjoint manifest; no tool execution.",
        "contract": {
            "manifest_path": str(corpus.manifest_path),
            "manifest_sha256": corpus.manifest_sha256,
            "records_sha256": corpus.manifest["records"]["sha256"],
            "checkpoints": contracts,
            "code_sha256": _code_hashes(),
            "protocol": {
                "layouts": list(layouts),
                "ratios": list(ratios),
                "hybrid_k": hybrid_k,
                "max_new_tokens": max_new_tokens,
                "sampling": "greedy",
                "decode_strategy": "incremental",
                "eviction": asdict(eviction),
                "compute_uniform_ce": compute_uniform_ce,
            },
            "environment": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
                "device": device,
                "dtype": dtype,
            },
            "started_utc": started.isoformat(),
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "records_count": len(rows),
        "cells": sorted({row["cell"] for row in rows}),
        **summary,
        "rows": rows,
    }


def save_evaluation(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Evaluation output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
