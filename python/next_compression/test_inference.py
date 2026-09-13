"""Tiny real-checkpoint tests for the next-compression inference bridge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from history_memory.preparation import tokenizer_identity
from next_compression.common import (
    PreparedDecision,
    SCHEMA,
    TRAINING_PROFILE,
    VARIANTS,
    make_record,
    sha256_file,
)
from next_compression.inference import (
    EVAL_PURPOSE,
    EXPECTED_RENDER_PROFILES,
    evaluate_checkpoint,
    load_evaluation_corpus,
    load_next_checkpoint,
)


def _load_train_entry():
    path = REPOSITORY_ROOT / "agent" / "train_next_compression.py"
    spec = importlib.util.spec_from_file_location("next_inference_test_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_train_entry()


def _load_eval_entry():
    path = REPOSITORY_ROOT / "agent" / "eval_next_checkpoint.py"
    spec = importlib.util.spec_from_file_location("next_inference_test_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory(domain: str, ratio: int = 8) -> PreparedDecision:
    event_id = f"{domain}-document"
    chunk = EncoderChunk(
        event_id=event_id,
        part_index=0,
        source_indices=(0,),
        source_token_start=0,
        source_token_end=24,
        token_ids=tuple(range(4, 28)),
    )
    memory = PackedMemory(
        view=MemoryView((event_id,), (f"{domain}-workspace",)),
        system_input_ids=(30, 31),
        workspace_input_ids=(32, 33),
        raw_source_indices=(1,),
        chunks=(chunk,),
    )
    return PreparedDecision(
        memory=memory,
        target_ids=(50, 2),
        ratio=ratio,
        decision_id="decision-1",
    )


def _training_identity(variant: str) -> str:
    return hashlib.sha256(f"training-manifest:{variant}".encode()).hexdigest()


def _tiny_checkpoint(
    root: Path,
    variant: str,
    *,
    render_profile: str | None = None,
) -> Path:
    args = train_entry.arguments(
        [
            "--cpu_smoke",
            "--variant",
            variant,
            "--output_dir",
            str(root / "unused-output"),
        ]
    )
    model, tokenizer = train_entry.build_model(args, torch.device("cpu"))
    train_entry.configure_training_metadata(
        model,
        args,
        corpus_identity=_training_identity(variant),
        render_profile=render_profile or EXPECTED_RENDER_PROFILES[variant],
        loss_profile=(
            "decision-mean-critical-token-weighted-ce-v1"
            if variant == "H3"
            else "decision-mean-complete-ce-v1"
        ),
    )
    checkpoint = root / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


@pytest.mark.parametrize("variant", ["H2", "T1"])
def test_saved_checkpoint_loads_fp32_gist_and_generates_packed_memory(
    tmp_path, variant
):
    checkpoint = _tiny_checkpoint(tmp_path, variant)
    generator, _, profile = load_next_checkpoint(
        checkpoint, device="cpu", dtype="float32"
    )
    saved = train_entry._checkpoint_gist_tensors(checkpoint)
    live = {
        name: parameter
        for name, parameter in generator.runtime.base_model.named_parameters()
        if train_entry._is_gist_parameter(name)
    }
    assert saved.keys() == live.keys()
    for name, parameter in live.items():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter.cpu(), saved[name], rtol=0, atol=0)

    decision = _memory(profile["compression_domain"])
    result = generator.generate(
        decision.memory,
        ratio=8,
        max_new_tokens=2,
        trace_context={
            "session_id": "session-1",
            "decision_key": decision.decision_id,
            "phase": EVAL_PURPOSE,
        },
    )
    assert 1 <= len(result.token_ids) <= 2
    assert result.finish_reason in {"eos", "length"}
    with pytest.raises(ValueError, match="ratio 4 is unsupported"):
        generator.generate(decision.memory, ratio=4, max_new_tokens=1)


def test_loader_rejects_variant_render_profile_mismatch_without_disk_patch(tmp_path):
    checkpoint = _tiny_checkpoint(
        tmp_path,
        "H2",
        render_profile="event-native-evidence-v1",
    )
    with pytest.raises(ValueError, match="render profile differs from H2"):
        load_next_checkpoint(checkpoint, device="cpu", dtype="float32")


def _write_eval_corpus(root: Path, checkpoint: Path, variant: str, *, purpose: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    variant_root = root / variant
    variant_root.mkdir(parents=True)
    rows = []
    for ratio in (8, 12):
        row = make_record(
            _memory("history" if variant.startswith("H") else "tool", ratio),
            session_key="session-1",
            source="tiny-eval",
            metadata={"gold_tool_calls": [{"name": "fixture", "arguments": {}}]},
        )
        row["split"] = EVAL_PURPOSE
        rows.append(row)
    records_path = variant_root / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    training_identities = {
        item: _training_identity(item) for item in VARIANTS
    }
    manifest = {
        "schema": SCHEMA,
        "training_profile": TRAINING_PROFILE,
        "purpose": purpose,
        "variant": variant,
        "compression_domain": "history" if variant.startswith("H") else "tool",
        "render_profile": EXPECTED_RENDER_PROFILES[variant],
        "loss_profile": "decision-mean-complete-ce-v1",
        "ratios": [8, 12],
        "tokenizer": tokenizer_identity(tokenizer),
        "records": {
            "path": records_path.name,
            "sha256": sha256_file(records_path),
            "bytes": records_path.stat().st_size,
            "count": len(rows),
        },
        "provenance": {
            "training_session_disjointness": {
                "status": "verified",
                "variants": list(VARIANTS),
                "training_manifest_sha256": training_identities,
            }
        },
    }
    (variant_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_bounded_eval_requires_dev_purpose_and_emits_fixed_protocol(tmp_path):
    checkpoint = _tiny_checkpoint(tmp_path / "model", "H0")
    rejected = tmp_path / "rejected"
    _write_eval_corpus(rejected, checkpoint, "H0", purpose="train")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    with pytest.raises(ValueError, match="purpose must be"):
        load_evaluation_corpus(
            rejected,
            tokenizer=tokenizer,
            variant="H0",
            checkpoint_corpus_identity=_training_identity("H0"),
        )

    accepted = tmp_path / "accepted"
    _write_eval_corpus(accepted, checkpoint, "H0", purpose=EVAL_PURPOSE)
    result = evaluate_checkpoint(
        checkpoint,
        accepted,
        ratios=(8, 12),
        max_new_tokens=2,
        max_decisions_per_ratio=1,
        device="cpu",
        dtype="float32",
    )
    assert result["status"] == "completed"
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["variant"] == "H0"
    assert result["protocol"] == {
        "ratios": [8, 12],
        "max_new_tokens": 2,
        "max_decisions_per_ratio": 1,
        "decode_strategy": "incremental",
        "sampling": "greedy",
    }
    assert result["records_count"] == len(result["records"]) == 2
    assert all(math.isfinite(record["uniform_ce"]) for record in result["records"])
    assert all(record["gold_tool_calls"] for record in result["records"])


def test_cli_rejects_existing_output_before_model_load(tmp_path, monkeypatch):
    eval_entry = _load_eval_entry()
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    called = False

    def unexpected_evaluation(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model evaluation must not start")

    monkeypatch.setattr(eval_entry, "evaluate_checkpoint", unexpected_evaluation)
    with pytest.raises(FileExistsError, match="already exists"):
        eval_entry.main(
            [
                "--checkpoint",
                str(tmp_path / "missing-checkpoint"),
                "--data-root",
                str(tmp_path / "missing-data"),
                "--output",
                str(output),
                "--ratios",
                "8,12",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-new-tokens",
                "2",
                "--max-decisions",
                "1",
            ]
        )
    assert called is False
