"""Validated next-compression checkpoint loading and bounded corpus evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel, PreparedDecision

from .common import (
    SCHEMA,
    TRAINING_PROFILE,
    VARIANTS,
    deserialize_memory,
    sha256_file,
    validate_record,
)


EVAL_PURPOSE = "checkpoint_selection_dev"
EVAL_SCHEMA = "next-compression-checkpoint-eval-v1"
EXPECTED_RATIOS = (8, 12)
EXPECTED_RENDER_PROFILES = {
    "H0": "event-native-evidence-v1",
    "H1": "event-native-evidence-v1",
    "H2": "a-event-native-s0-v1",
    "H3": "a-event-native-s0-v1",
    "T0": "next-compression-tool-explicit-protocol-v2",
    "T1": "next-compression-tool-explicit-protocol-v2",
}
EXPECTED_LOSS_PROFILES = {
    variant: (
        "decision-mean-critical-token-weighted-ce-v1"
        if variant == "H3"
        else "decision-mean-complete-ce-v1"
    )
    for variant in VARIANTS
}


def _agent_module(module_name: str, filename: str):
    """Load a repository CLI module despite the ``python.agent`` name clash."""

    qualified_name = f"_next_compression_{module_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[2] / "agent" / filename
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load checkpoint utility: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _domain(variant: str) -> str:
    return "history" if variant.startswith("H") else "tool"


def _validate_next_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the six explicit next-compression model contracts."""

    # Reuse the exporter gate so direct and materialized checkpoints obey the
    # same initialization, loss, gist, and provenance rules.
    export_utility = _agent_module("export", "export_next_checkpoint.py")
    metadata = export_utility._validate_checkpoint_config(config)
    variant = metadata["variant"]
    expected_render = EXPECTED_RENDER_PROFILES[variant]
    if metadata["render_profile"] != expected_render:
        raise ValueError(
            f"Checkpoint render profile differs from {variant}: "
            f"{metadata['render_profile']!r} vs {expected_render!r}"
        )
    if tuple(metadata["ratios"]) != EXPECTED_RATIOS:
        raise ValueError("Checkpoint must support exactly ratios 8 and 12")
    if metadata["compression_domain"] != _domain(variant):
        raise ValueError("Checkpoint compression domain differs from its variant")
    return metadata


def _torch_dtype(name: str):
    import torch

    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(
            "dtype must be float32, bfloat16, or float16"
        ) from error


def _resolve_device(value: str):
    import torch

    if not isinstance(value, str) or not value:
        raise ValueError("device must be explicit and nonempty")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
    elif device.type != "cpu":
        raise ValueError("device must select cpu or cuda")
    return device


def load_next_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    decode_strategy: str = "incremental",
    max_extraction_calls: int | None = None,
) -> tuple[EventNativeGenerator, Any, dict[str, Any]]:
    """Load local model/tokenizer files while retaining saved gist FP32 exactly."""

    import torch
    from transformers import AutoTokenizer

    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint_path}")
    metadata = _validate_next_config(_read_json(config_path))
    compute_dtype = _torch_dtype(dtype)
    compute_device = _resolve_device(device)
    if (
        compute_device.type == "cuda"
        and compute_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("The selected CUDA device does not support bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, local_files_only=True
    )
    config = Qwen3Config.from_pretrained(checkpoint_path, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        dtype=compute_dtype,
        attn_implementation="eager",
    ).to(compute_device)
    # A global BF16/FP16 load casts the compressor tensors too. Restore their
    # saved FP32 values from safetensors before constructing the runtime.
    train_utility = _agent_module("train", "train_next_compression.py")
    train_utility._restore_warm_start_gist_precision(model, checkpoint_path)
    runtime = HistoryMemoryModel(model)
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.is_file():
        trainer_state = _read_json(trainer_state_path)
        parameter_version = trainer_state.get("parameter_version")
        if type(parameter_version) is int and parameter_version >= 0:
            runtime.restore_parameter_version(parameter_version)
    generator = EventNativeGenerator(
        runtime,
        decode_strategy=decode_strategy,
        max_extraction_calls=max_extraction_calls,
    )
    profile = {
        **metadata,
        "checkpoint": str(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "dtype": dtype,
        "device": str(compute_device),
        "decode_strategy": decode_strategy,
        "gist_parameter_dtype": "float32",
    }
    return generator, tokenizer, profile


@dataclass(frozen=True)
class EvaluationRecord:
    decision_id: str
    session_key: str
    ratio: int
    memory: Any
    target_ids: tuple[int, ...]
    metadata: dict[str, Any]
    gold_tool_calls: Any


@dataclass(frozen=True)
class EvaluationCorpus:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    records: tuple[EvaluationRecord, ...]


def _evaluation_manifest_path(data_root: str | Path, variant: str) -> Path:
    root = Path(data_root).resolve()
    candidates = (root / variant / "manifest.json", root / "manifest.json")
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            f"No evaluation manifest found for {variant} under {root}"
        )
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous evaluation manifests for {variant}: {found!r}"
        )
    return found[0]


def _validate_eval_manifest(
    manifest: Mapping[str, Any],
    *,
    variant: str,
    checkpoint_corpus_identity: str,
) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Evaluation manifest schema must be {SCHEMA!r}")
    if manifest.get("training_profile") != TRAINING_PROFILE:
        raise ValueError("Evaluation manifest has an incompatible training profile")
    if manifest.get("purpose") != EVAL_PURPOSE:
        raise ValueError(
            f"Evaluation manifest purpose must be {EVAL_PURPOSE!r}"
        )
    if manifest.get("variant") != variant:
        raise ValueError("Evaluation manifest variant differs from checkpoint")
    if manifest.get("compression_domain") != _domain(variant):
        raise ValueError("Evaluation compression domain differs from checkpoint")
    if manifest.get("render_profile") != EXPECTED_RENDER_PROFILES[variant]:
        raise ValueError("Evaluation render profile differs from checkpoint variant")
    if manifest.get("loss_profile") != EXPECTED_LOSS_PROFILES[variant]:
        raise ValueError("Evaluation loss profile differs from checkpoint variant")
    if manifest.get("ratios") != list(EXPECTED_RATIOS):
        raise ValueError("Evaluation manifest must declare ratios [8, 12]")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Evaluation manifest must declare provenance")
    disjointness = provenance.get("training_session_disjointness")
    if not isinstance(disjointness, Mapping):
        raise ValueError(
            "Evaluation manifest must declare training_session_disjointness"
        )
    if disjointness.get("status") != "verified":
        raise ValueError("Training-session disjointness must be verified")
    if disjointness.get("variants") != list(VARIANTS):
        raise ValueError("Disjointness provenance must cover all six variants")
    identities = disjointness.get("training_manifest_sha256")
    if not isinstance(identities, Mapping) or set(identities) != set(VARIANTS):
        raise ValueError(
            "Disjointness provenance must bind every variant training manifest"
        )
    if identities.get(variant) != checkpoint_corpus_identity:
        raise ValueError(
            "Checkpoint corpus identity differs from disjointness provenance"
        )


def load_evaluation_corpus(
    data_root: str | Path,
    *,
    tokenizer: Any,
    variant: str,
    checkpoint_corpus_identity: str,
) -> EvaluationCorpus:
    """Load a purpose-bound eval corpus without accepting train-split rows."""

    from history_memory.preparation import tokenizer_identity

    if variant not in VARIANTS:
        raise ValueError(f"Unknown next-compression variant: {variant!r}")
    manifest_path = _evaluation_manifest_path(data_root, variant)
    manifest = _read_json(manifest_path)
    _validate_eval_manifest(
        manifest,
        variant=variant,
        checkpoint_corpus_identity=checkpoint_corpus_identity,
    )
    tokenizer_contract = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer_contract, Mapping)
        or tokenizer_contract.get("sha256") != tokenizer_identity(tokenizer)["sha256"]
    ):
        raise ValueError("Evaluation tokenizer differs from checkpoint tokenizer")

    records_info = manifest.get("records")
    if not isinstance(records_info, Mapping):
        raise ValueError("Evaluation manifest must declare records")
    filename = records_info.get("path")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Evaluation records path must be a nonempty string")
    records_path = (manifest_path.parent / filename).resolve()
    if records_path.parent != manifest_path.parent.resolve() or not records_path.is_file():
        raise ValueError("Evaluation records must be inside the manifest directory")
    if (
        records_path.stat().st_size != records_info.get("bytes")
        or sha256_file(records_path) != records_info.get("sha256")
    ):
        raise ValueError("Evaluation records hash/size differs from manifest")

    vocab_size = len(tokenizer)
    rows: list[EvaluationRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Evaluation row {line_number} must be an object")
            if raw.get("split") != EVAL_PURPOSE:
                raise ValueError(
                    f"Evaluation row {line_number} split must be {EVAL_PURPOSE!r}"
                )
            # Reuse the complete training-record structural/token validation on
            # an in-memory copy. The source row and file remain unchanged.
            validation_copy = dict(raw)
            validation_copy["split"] = "train"
            validate_record(
                validation_copy,
                ratios=EXPECTED_RATIOS,
                vocab_size=vocab_size,
            )
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Evaluation row {line_number} metadata must be an object")
            gold_tool_calls = raw.get("gold_tool_calls")
            if gold_tool_calls is None:
                gold_tool_calls = metadata.get("gold_tool_calls")
            rows.append(
                EvaluationRecord(
                    decision_id=raw["decision_id"],
                    session_key=raw["session_key"],
                    ratio=raw["ratio"],
                    memory=deserialize_memory(raw["memory"]),
                    target_ids=tuple(raw["target_ids"]),
                    metadata=metadata,
                    gold_tool_calls=gold_tool_calls,
                )
            )
    if len(rows) != records_info.get("count"):
        raise ValueError("Evaluation record count differs from manifest")
    if not rows:
        raise ValueError("Evaluation corpus is empty")
    return EvaluationCorpus(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        records=tuple(rows),
    )


def _selected_records(
    corpus: EvaluationCorpus,
    *,
    ratios: Sequence[int],
    max_decisions_per_ratio: int,
) -> tuple[EvaluationRecord, ...]:
    if type(max_decisions_per_ratio) is not int or max_decisions_per_ratio <= 0:
        raise ValueError("max_decisions_per_ratio must be a positive integer")
    if tuple(ratios) != EXPECTED_RATIOS:
        raise ValueError("This evaluation protocol requires ratios 8 and 12")
    selected_by_ratio: dict[int, tuple[EvaluationRecord, ...]] = {}
    for ratio in ratios:
        candidates = tuple(record for record in corpus.records if record.ratio == ratio)
        selected = candidates[:max_decisions_per_ratio]
        if len(selected) != max_decisions_per_ratio:
            raise ValueError(
                f"Ratio {ratio} has {len(selected)} records; expected exactly "
                f"{max_decisions_per_ratio}"
            )
        ids = [record.decision_id for record in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Ratio {ratio} selection contains duplicate decision IDs")
        selected_by_ratio[ratio] = selected
    first_ids = [record.decision_id for record in selected_by_ratio[ratios[0]]]
    for ratio in ratios[1:]:
        if [record.decision_id for record in selected_by_ratio[ratio]] != first_ids:
            raise ValueError("Ratio selections must preserve the same ordered decision IDs")
    return tuple(
        record for ratio in ratios for record in selected_by_ratio[ratio]
    )


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    ratios: Sequence[int],
    max_new_tokens: int,
    max_decisions_per_ratio: int,
    device: str,
    dtype: str,
    compute_uniform_ce: bool = True,
) -> dict[str, Any]:
    """Greedily evaluate one checkpoint on a finite disjoint dev corpus."""

    import torch

    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    checkpoint_path = Path(checkpoint).resolve()
    config = _read_json(checkpoint_path / "config.json")
    metadata = _validate_next_config(config)
    # Load the tokenizer before selecting rows; the manifest binds it exactly.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    corpus = load_evaluation_corpus(
        data_root,
        tokenizer=tokenizer,
        variant=metadata["variant"],
        checkpoint_corpus_identity=metadata["corpus_identity"],
    )
    selected = _selected_records(
        corpus,
        ratios=ratios,
        max_decisions_per_ratio=max_decisions_per_ratio,
    )
    extraction_cap = sum(len(record.memory.chunks) for record in selected)
    generator, loaded_tokenizer, profile = load_next_checkpoint(
        checkpoint_path,
        device=device,
        dtype=dtype,
        decode_strategy="incremental",
        max_extraction_calls=max(1, extraction_cap),
    )
    if tokenizer.get_vocab() != loaded_tokenizer.get_vocab():
        raise RuntimeError("Checkpoint tokenizer changed between preflight and load")

    output_records = []
    active_ratio = None
    for record in selected:
        if active_ratio is not None and record.ratio != active_ratio:
            generator.close_session()
        active_ratio = record.ratio
        trace_context = {
            "session_id": record.session_key,
            "decision_key": record.decision_id,
            "phase": "checkpoint_selection_dev",
        }
        with generator.decision_scope(
            session_id=f"ratio{record.ratio}:{record.session_key}"
        ):
            generation = generator.generate(
                record.memory,
                ratio=record.ratio,
                max_new_tokens=max_new_tokens,
                trace_context=trace_context,
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
            with torch.inference_mode():
                value = generator.runtime((decision,))["loss"].detach().float().item()
            if not math.isfinite(value):
                raise ValueError(f"Non-finite uniform CE for {record.decision_id}")
            uniform_ce = value
        output_records.append(
            {
                "decision_id": record.decision_id,
                "ratio": record.ratio,
                "target_token_ids": list(record.target_ids),
                "target_text": _decode(loaded_tokenizer, record.target_ids),
                "generated_token_ids": list(generation.token_ids),
                "generated_text": _decode(loaded_tokenizer, generation.token_ids),
                "finish_reason": generation.finish_reason,
                "uniform_ce": uniform_ce,
                "metadata": record.metadata,
                "gold_tool_calls": record.gold_tool_calls,
            }
        )
    generator.close_session()
    return {
        "schema": EVAL_SCHEMA,
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "variant": profile["variant"],
        "config_sha256": profile["config_sha256"],
        "eval_manifest_sha256": corpus.manifest_sha256,
        "protocol": {
            "ratios": list(ratios),
            "max_new_tokens": max_new_tokens,
            "max_decisions_per_ratio": max_decisions_per_ratio,
            "decode_strategy": "incremental",
            "sampling": "greedy",
        },
        "records_count": len(output_records),
        "records": output_records,
    }


def save_evaluation(path: str | Path, value: Mapping[str, Any]) -> None:
    """Publish one completed result atomically without overwriting evidence."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".pending-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale pending output exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EVAL_PURPOSE",
    "EVAL_SCHEMA",
    "EXPECTED_RATIOS",
    "EXPECTED_RENDER_PROFILES",
    "EvaluationCorpus",
    "EvaluationRecord",
    "evaluate_checkpoint",
    "load_evaluation_corpus",
    "load_next_checkpoint",
    "save_evaluation",
]
