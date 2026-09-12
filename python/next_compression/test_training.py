"""CPU contracts for next-compression runtime loss and the training entry."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history_memory.runtime import HistoryMemoryModel
from history_memory.training import seed_everything
from next_compression.common import CorpusWriter, SerializedCorpus, make_record

ENTRY_PATH = Path(__file__).resolve().parents[2] / "agent" / "train_next_compression.py"
SPEC = importlib.util.spec_from_file_location("next_compression_training_entry", ENTRY_PATH)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "agent" / "launch_next_compression.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "next_compression_launcher", LAUNCHER_PATH
)
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)


def _write_corpus(path: Path, variant: str = "H3") -> Path:
    from history_memory.preparation import tokenizer_identity

    tokenizer = entry.tiny_tokenizer()
    decisions = entry.smoke_corpus(variant)
    writer = CorpusWriter(path, variant)
    for index in range(len(decisions)):
        writer.write(
            make_record(
                decisions[index],
                session_key=decisions.group_ids[index],
                source="tiny-full-prepared-fixture",
                metadata={"fixture_index": index},
            )
        )
    writer.finish(
        tokenizer=tokenizer_identity(tokenizer),
        preparation={
            "render_profile": "tiny-full-prepared-render-v1",
            "loss_profile": (
                "decision-mean-critical-token-weighted-ce-v1"
                if variant == "H3"
                else "decision-mean-complete-ce-v1"
            ),
            "target_weighting": {
                "source": "prepared-fixture" if variant == "H3" else "none"
            },
        },
        source_files={"fixture": "generated-in-test"},
        audit={"purpose": "full-serialized-corpus-training-path"},
    )
    return path


def _write_base_receipt(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "C2KV_SOURCE_REVISION.json").write_text(
        json.dumps(
            {
                "repository": launcher.BASE_REPOSITORY,
                "revision": launcher.BASE_REVISION,
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(data: Path, output: Path, *, resume: Path | None = None, stop: int = -1):
    options = [
        "--cpu_smoke",
        "--data_path",
        str(data),
        "--variant",
        "H3",
        "--ratios",
        "8,12",
        "--output_dir",
        str(output),
        "--per_device_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--num_train_epochs",
        "2",
        "--max_steps",
        "2",
        "--save_steps",
        "100",
        "--stop_after_steps",
        str(stop),
    ]
    if resume is not None:
        options.extend(("--resume_from_checkpoint", str(resume)))
    return entry.main(options)


def test_h3_target_weights_are_complete_weighted_ce_and_change_gist_gradient(tmp_path):
    args = entry.arguments(
        ["--cpu_smoke", "--variant", "H3", "--output_dir", str(tmp_path / "unused")]
    )
    seed_everything(args.seed)
    model, _ = entry.build_model(args, torch.device("cpu"))
    weighted_runtime = HistoryMemoryModel(model)
    uniform_runtime = HistoryMemoryModel(copy.deepcopy(model))
    decision = entry.smoke_corpus("H3")[0]

    captured_logits = []
    handle = weighted_runtime.base_model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: captured_logits.append(output)
    )
    weighted = weighted_runtime([decision])
    handle.remove()
    token_losses = torch.nn.functional.cross_entropy(
        captured_logits[0].squeeze(0).float(),
        torch.tensor(decision.target_ids),
        reduction="none",
    )
    target_weights = torch.tensor(decision.target_weights)
    expected = (token_losses * target_weights).sum() / target_weights.sum()
    torch.testing.assert_close(weighted["loss"], expected)
    weighted["loss"].backward()

    uniform = uniform_runtime([replace(decision, target_weights=None)])
    uniform["loss"].backward()
    assert not torch.isclose(weighted["loss"], uniform["loss"], rtol=1e-7, atol=1e-7)
    weighted_gradients = {
        name: parameter.grad
        for name, parameter in weighted_runtime.named_parameters()
        if parameter.requires_grad
    }
    uniform_gradients = {
        name: parameter.grad
        for name, parameter in uniform_runtime.named_parameters()
        if parameter.requires_grad
    }
    assert any(
        not torch.allclose(weighted_gradients[name], uniform_gradients[name])
        for name in weighted_gradients
    )


def test_legacy_duck_record_without_target_weights_keeps_mean_ce(tmp_path):
    args = entry.arguments(
        ["--cpu_smoke", "--variant", "H0", "--output_dir", str(tmp_path / "unused")]
    )
    seed_everything(args.seed)
    model, _ = entry.build_model(args, torch.device("cpu"))
    runtime = HistoryMemoryModel(model)
    decision = entry.smoke_corpus("H0")[0]
    legacy = SimpleNamespace(
        memory=decision.memory,
        target_ids=decision.target_ids,
        ratio=decision.ratio,
        weight=decision.weight,
        decision_id=decision.decision_id,
    )
    torch.testing.assert_close(runtime([legacy])["loss"], runtime([decision])["loss"])


def test_full_serialized_corpus_fresh_resume_and_metadata(tmp_path):
    from safetensors.torch import load_file

    data = _write_corpus(tmp_path / "prepared")
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    expected_state = _run(data, continuous)
    stopped = _run(data, resumed, stop=1)
    assert stopped["global_step"] == 1 and not stopped["completed"]
    actual_state = _run(data, resumed, resume=resumed / "checkpoint-1")
    assert expected_state["completed"] and actual_state["completed"]

    expected = load_file(continuous / "checkpoint-2" / "model.safetensors")
    actual = load_file(resumed / "checkpoint-2" / "model.safetensors")
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    config = json.loads(
        (resumed / "checkpoint-2" / "config.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (resumed / "checkpoint-2" / "trainer_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["history_memory_training_profile"] == "next-compression-base-query-v1"
    assert config["history_memory_variant"] == "H3"
    assert config["history_memory_compression_domain"] == "history"
    assert config["history_memory_supported_ratios"] == [8, 12]
    assert config["history_memory_normal_query"] == "base"
    assert config["history_memory_corpus_identity"] == state["contract"]["dataset_sha256"]
    assert config["history_memory_corpus_identity"] == SerializedCorpus(data).identity
    assert state["training_profile"] == "next-compression-base-query-v1"
    assert state["contract"]["variant"] == "H3"
    assert state["contract"]["loss_profile"] == "decision-mean-critical-token-weighted-ce-v1"
    assert state["contract"]["render_profile"] == "tiny-full-prepared-render-v1"
    assert state["contract"]["initialization_id"] == config["history_memory_initialization_id"]


def test_explicit_warm_start_retains_saved_gist_weights(tmp_path):
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    data = _write_corpus(tmp_path / "prepared")
    source_output = tmp_path / "source"
    _run(data, source_output, stop=1)
    checkpoint = source_output / "checkpoint-1"
    saved = Qwen3ForCausalLM.from_pretrained(checkpoint, local_files_only=True)

    args = entry.arguments(
        [
            "--device",
            "cpu",
            "--data_path",
            str(data),
            "--variant",
            "H3",
            "--warm_start_checkpoint",
            str(checkpoint),
            "--initialization_id",
            "explicit-trained-gist-fixture-v1",
            "--output_dir",
            str(tmp_path / "warm"),
            "--no-gradient_checkpointing",
            "--wandb_mode",
            "disabled",
        ]
    )
    warmed, _ = entry.build_model(args, torch.device("cpu"))
    warmed_state = warmed.state_dict()
    assert warmed.model.embed_tokens.weight.dtype == torch.bfloat16
    for name, value in saved.state_dict().items():
        if "gist_" in name:
            assert warmed_state[name].dtype == torch.float32
            torch.testing.assert_close(warmed_state[name], value, rtol=0, atol=0)
    assert warmed.config.history_memory_initialization_kind == "warm-start"
    assert warmed.config.history_memory_initialization_id == "explicit-trained-gist-fixture-v1"


@pytest.mark.parametrize(
    "weights,match",
    [
        ((1.0,), "cover every"),
        ((1.0, 0.0), "strictly positive"),
        ((1.0, float("nan")), "strictly positive"),
    ],
)
def test_runtime_rejects_incomplete_or_nonpositive_target_weights(tmp_path, weights, match):
    args = entry.arguments(
        ["--cpu_smoke", "--variant", "H3", "--output_dir", str(tmp_path / "unused")]
    )
    seed_everything(args.seed)
    model, _ = entry.build_model(args, torch.device("cpu"))
    decision = replace(entry.smoke_corpus("H3")[0], target_weights=weights)
    with pytest.raises(ValueError, match=match):
        HistoryMemoryModel(model)([decision])


def test_launcher_uses_exactly_one_fresh_warm_or_resume_source(tmp_path):
    common = dict(
        python="python",
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        model="base-model",
        seed=42,
        epochs=2,
        save_steps=500,
        max_steps=1098,
        stop_after_steps=-1,
        per_device_batch_size=2,
        wandb_mode="offline",
        warm_start_history="history-warm",
        warm_start_tool="tool-warm",
    )
    history = launcher.build_command(
        SimpleNamespace(**common, resumes={}), "H0", ["0"]
    )
    assert "--warm_start_checkpoint" in history
    assert "history-warm" in history
    assert "--model_name_or_path" not in history
    assert "--resume_from_checkpoint" not in history

    resumed = launcher.build_command(
        SimpleNamespace(**common, resumes={"H0": "checkpoint-7"}), "H0", ["0"]
    )
    assert "--resume_from_checkpoint" in resumed
    assert "checkpoint-7" in resumed
    assert "--warm_start_checkpoint" not in resumed
    assert "--model_name_or_path" not in resumed

    fresh = launcher.build_command(
        SimpleNamespace(
            **(common | {"warm_start_history": None, "warm_start_tool": None}),
            resumes={},
        ),
        "T0",
        ["0", "1"],
    )
    assert "--model_name_or_path" in fresh and "base-model" in fresh
    assert "--warm_start_checkpoint" not in fresh
    assert "--resume_from_checkpoint" not in fresh
    assert fresh[:7] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(Path(launcher.__file__).resolve().with_name("train_next_compression.py")),
    ]

    batch_one = launcher.build_command(
        SimpleNamespace(
            **{**common, "per_device_batch_size": 1},
            resumes={},
        ),
        "T0",
        ["0", "1"],
    )
    assert batch_one[batch_one.index("--per_device_batch_size") + 1] == "1"
    assert batch_one[batch_one.index("--gradient_accumulation_steps") + 1] == "16"


def test_launcher_future_failure_receipt_retains_variant(tmp_path, monkeypatch):
    _write_corpus(tmp_path / "data" / "H0", "H0")
    model = _write_base_receipt(tmp_path / "base")
    monkeypatch.setattr(
        launcher,
        "check_devices",
        lambda devices: {device: {"name": "fixture"} for device in devices},
    )

    def fail_command(*_args, **_kwargs):
        raise RuntimeError("fixture launch failure")

    monkeypatch.setattr(launcher.subprocess, "run", fail_command)
    output = tmp_path / "output"
    result = launcher.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--model",
            str(model),
            "--output-root",
            str(output),
            "--devices",
            "0",
            "--variants",
            "H0",
            "--run",
        ]
    )
    assert result == 1
    receipt = json.loads(next(output.glob("queue-*.json")).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["outcomes"] == [
        {
            "variant": "H0",
            "devices": ["0"],
            "returncode": 1,
            "error": "fixture launch failure",
        }
    ]


def test_launcher_dry_run_validates_corpus_and_reports_uncapped_epochs(
    tmp_path, capsys
):
    _write_corpus(tmp_path / "data" / "H0", "H0")
    model = _write_base_receipt(tmp_path / "base")
    result = launcher.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--model",
            str(model),
            "--output-root",
            str(tmp_path / "output"),
            "--devices",
            "0",
            "--variants",
            "H0",
        ]
    )
    assert result == 0
    plan_text = capsys.readouterr().out.split("\n# H0\n", 1)[0]
    plan = json.loads(plan_text)
    assert plan["max_steps"] == -1
    assert plan["corpus_manifests"]["H0"]["records"] == 8
    assert plan["corpus_manifests"]["H0"]["uncapped_steps"] == 2
    assert plan["corpus_manifests"]["H0"]["planned_steps"] == 2
    command = plan["command_examples"]["H0"]
    assert command[command.index("--per_device_batch_size") + 1] == "2"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "16"
