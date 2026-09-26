"""End-to-end CPU test of the Exp 1 evaluator on tiny T0/T1 checkpoints."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from history_memory.dataset import iter_decisions
from history_memory.packing import pack_target
from history_memory.preparation import tokenizer_identity
from next_compression.exp1_eval import (
    EVAL_SCHEMA,
    EvictionSettings,
    evaluate_exp1,
    load_exp1_corpus,
    save_evaluation,
)
from next_compression.exp1_tools import (
    EVICTION_LAYOUTS,
    EXP1_PURPOSE,
    EXP1_SCHEMA,
    GIST_LAYOUTS,
    LayoutPlan,
    build_layout_records,
    decision_type,
    gold_tool_calls,
)
from next_compression.inference import EXPECTED_RENDER_PROFILES
from next_compression.test_exp1_tools import _three_tools
from next_compression.test_tools import _row
from next_compression.tools import ToolPreparationConfig


def _load(name: str, filename: str):
    path = REPOSITORY_ROOT / "agent" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_entry = _load("exp1_eval_test_train", "train_next_compression.py")
prepare_entry = _load("exp1_eval_test_prepare", "prepare_exp1_tools.py")


def _tiny_checkpoint(root: Path, variant: str) -> Path:
    args = train_entry.arguments(
        ["--cpu_smoke", "--variant", variant, "--output_dir", str(root / "unused")]
    )
    model, tokenizer = train_entry.build_model(args, torch.device("cpu"))
    train_entry.configure_training_metadata(
        model,
        args,
        corpus_identity=hashlib.sha256(f"training-manifest:{variant}".encode()).hexdigest(),
        render_profile=EXPECTED_RENDER_PROFILES[variant],
        loss_profile="decision-mean-complete-ce-v1",
    )
    checkpoint = root / f"checkpoint-{variant}"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


def _manifest(root: Path, tokenizer) -> Path:
    config = ToolPreparationConfig(max_raw_tokens=None)
    rows = [_row(), _row(final_text=True)]
    for index, row in enumerate(rows):
        row["session_id"] = f"session-{index}"
        row["tools"] = _three_tools()
    writer = prepare_entry.Exp1Writer(root / "exp1")
    for row in rows:
        decisions = list(iter_decisions(row))
        decision = decisions[-1]
        target_ids = pack_target(tokenizer, decision.target)
        records = build_layout_records(
            decision,
            tokenizer,
            config=config,
            plan=LayoutPlan(k_values=(1,), seed=3),
            session_key=f"fixture:{row['session_id']}",
            source="fixture",
            target_ids=target_ids,
            base_metadata={
                "decision_type": decision_type(decision, last_source_index=len(row["messages"]) - 1),
                "gold_tool_calls": gold_tool_calls(decision),
            },
        )
        writer.write_decision(records)
    writer.finish({
        "schema": EXP1_SCHEMA,
        "purpose": EXP1_PURPOSE,
        "tokenizer": tokenizer_identity(tokenizer),
        "ratios": [8, 12],
    })
    return root / "exp1"


def test_evaluate_exp1_runs_every_layout_on_tiny_checkpoints(tmp_path):
    from transformers import AutoTokenizer

    checkpoint_t0 = _tiny_checkpoint(tmp_path, "T0")
    checkpoint_t1 = _tiny_checkpoint(tmp_path, "T1")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_t0, local_files_only=True)
    manifest_root = _manifest(tmp_path, tokenizer)

    corpus = load_exp1_corpus(manifest_root, tokenizer=tokenizer)
    assert {record.layout for record in corpus.records} == set(GIST_LAYOUTS)
    assert len(corpus.select("full")) == 4  # two decisions x two ratios

    result = evaluate_exp1(
        manifest_root=manifest_root,
        checkpoint_t0=checkpoint_t0,
        checkpoint_t1=checkpoint_t1,
        layouts=GIST_LAYOUTS + EVICTION_LAYOUTS,
        hybrid_k=1,
        max_new_tokens=2,
        device="cpu",
        dtype="float32",
        eviction=EvictionSettings(obs_window=2, kernel=3, chunk_size=7),
        compute_uniform_ce=True,
    )
    assert result["schema"] == EVAL_SCHEMA and result["status"] == "completed"
    expected_cells = set()
    for ratio in (8, 12):
        for layout in ("full", "uniform", "t1"):
            expected_cells.add(f"{layout}.kNone.ratio{ratio}")
        for layout in ("hybrid", "random", "retrieval", "snapkv_hybrid", "h2o_hybrid"):
            expected_cells.add(f"{layout}.k1.ratio{ratio}")
        for layout in ("snapkv", "h2o"):
            expected_cells.add(f"{layout}.kNone.ratio{ratio}")
    assert set(result["cells"]) == expected_cells
    assert result["records_count"] == 2 * len(expected_cells)
    for row in result["rows"]:
        assert 1 <= len(row["generated_token_ids"]) <= 2
        assert row["resident_kv_tokens"] > 0
        if row["layout"] in EVICTION_LAYOUTS:
            assert row["uniform_ce"] is None
            assert row["eviction"]["keep"] <= row["eviction"]["evictable_tokens"]
        else:
            assert row["uniform_ce"] is not None
    metrics = result["metrics"]
    for cell in expected_cells:
        assert metrics[cell]["records"] == 2
        assert metrics[cell]["tool_decisions"] == 1 and metrics[cell]["non_tool_decisions"] == 1
    # Eviction rows keep exactly the gist budget of the matching T0 layout.
    by_cell = {}
    for row in result["rows"]:
        by_cell.setdefault(row["cell"], {})[row["decision_id"]] = row
    for ratio in (8, 12):
        for decision_id, uniform in by_cell[f"uniform.kNone.ratio{ratio}"].items():
            for method in ("snapkv", "h2o"):
                evicted = by_cell[f"{method}.kNone.ratio{ratio}"][decision_id]
                assert evicted["eviction"]["keep"] == min(
                    uniform["gist_tokens"], evicted["eviction"]["evictable_tokens"]
                )
    assert result["paired"], "paired comparisons must be reported"
    contract = result["contract"]
    assert contract["checkpoints"]["t0"]["variant"] == "T0"
    assert contract["checkpoints"]["t1"]["variant"] == "T1"
    assert contract["manifest_sha256"] and contract["code_sha256"]
    output = tmp_path / "eval.json"
    save_evaluation(output, result)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        save_evaluation(output, result)
    with pytest.raises(ValueError, match="T0 checkpoint"):
        evaluate_exp1(
            manifest_root=manifest_root, checkpoint_t0=checkpoint_t1, layouts=("uniform",),
            max_new_tokens=1, device="cpu", dtype="float32",
        )
