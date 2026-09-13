"""Focused CPU contracts for live OpenAI-compatible checkpoint serving."""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from history_memory.preparation import tokenizer_identity
from next_compression.common import SCHEMA, TRAINING_PROFILE, sha256_file
from next_compression.history import HistoryPreparationConfig
from next_compression.live import (
    LiveNextCompressionService,
    LiveRequestError,
    TrainingBinding,
)
from next_compression.tools import ToolPreparationConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RENDER_PROFILES = {
    "H0": "event-native-evidence-v1",
    "H1": "event-native-evidence-v1",
    "H2": "a-event-native-s0-v1",
    "H3": "a-event-native-s0-v1",
    "T0": "next-compression-tool-explicit-protocol-v2",
    "T1": "next-compression-tool-explicit-protocol-v2",
}


class ByteChatTokenizer:
    """A reversible native template with prefix-separable byte tokens."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        truncation=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert truncation is False
        self.calls.append(json.loads(json.dumps({"messages": messages, "tools": tools})))
        parts = []
        if tools is not None:
            parts.append(
                "<tools>"
                + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                + "</tools>"
            )
        for message in messages:
            role = message["role"]
            value = {key: item for key, item in message.items() if key != "role"}
            parts.append(
                f"<{role}>"
                + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                + f"</{role}>"
            )
        if add_generation_prompt:
            parts.append("<assistant>")
        return [byte + 1 for byte in "".join(parts).encode("utf-8")]

    @staticmethod
    def decode(ids, **_kwargs) -> str:
        return bytes(int(token) - 1 for token in ids).decode("utf-8")


class FakeGenerator:
    def __init__(self, tokenizer: ByteChatTokenizer, output: str) -> None:
        self.tokenizer = tokenizer
        self.output = output
        self.memories = []
        self.closed = 0

    @contextmanager
    def decision_scope(self, *, session_id=None):
        assert session_id is None
        yield

    def generate(self, memory, **_kwargs):
        self.memories.append(memory)
        return SimpleNamespace(
            token_ids=tuple(byte + 1 for byte in self.output.encode("utf-8")),
            finish_reason="eos",
            stats={"target_forward_calls": 1},
        )

    def close_session(self) -> None:
        self.closed += 1


def _profile(variant: str) -> dict:
    return {
        "checkpoint": f"/checkpoint/{variant}",
        "config_sha256": "1" * 64,
        "corpus_identity": "2" * 64,
        "training_profile": TRAINING_PROFILE,
        "variant": variant,
        "compression_domain": "history" if variant.startswith("H") else "tool",
        "render_profile": EXPECTED_RENDER_PROFILES[variant],
        "loss_profile": "decision-mean-complete-ce-v1",
        "initialization_id": "tiny-fixture",
        "device": "cpu",
        "dtype": "float32",
    }


def _service(
    variant: str,
    output: str,
    *,
    mode: str = "compressed",
    aliases=(),
    config=None,
    max_new_tokens: int = 1024,
):
    tokenizer = ByteChatTokenizer()
    generator = FakeGenerator(tokenizer, output)
    if config is None:
        config = (
            HistoryPreparationConfig()
            if variant.startswith("H")
            else ToolPreparationConfig()
        )
    binding = TrainingBinding(Path("manifest.json"), "2" * 64, {}, config)
    service = LiveNextCompressionService(
        generator,
        tokenizer,
        _profile(variant),
        binding,
        ratio=8,
        mode=mode,
        model="c2kv-next",
        model_aliases=aliases,
        max_new_tokens=max_new_tokens,
        max_requests=20,
    )
    return service, generator, tokenizer


def _tool(*, descriptive: bool = True) -> dict:
    function = {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }
    if descriptive:
        function["description"] = "Look up an exact value."
    return {"type": "function", "function": function}


def _payload(*, tools=(), messages=None, model="c2kv-next", **extra) -> dict:
    return {
        "model": model,
        "messages": messages
        or [
            {"role": "system", "content": "Use the available context."},
            {"role": "user", "content": "Look up alpha."},
        ],
        "tools": list(tools),
        "temperature": 0,
        **extra,
    }


def test_native_tools_parse_multiple_calls_and_string_content():
    output = (
        "checking"
        '<tool_call>{"name":"lookup","arguments":{"key":"alpha"}}</tool_call>'
        '<tool_call>{"name":"lookup","arguments":"{\\"key\\":\\"beta\\"}"}</tool_call>'
    )
    service, generator, _ = _service("T0", output)
    response = service.complete(_payload(tools=[_tool()]))
    message = response["choices"][0]["message"]
    assert message["content"] == "checking"
    assert [call["function"]["name"] for call in message["tool_calls"]] == [
        "lookup",
        "lookup",
    ]
    assert json.loads(message["tool_calls"][1]["function"]["arguments"]) == {
        "key": "beta"
    }
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert response["x_c2kv"]["compression"]["chunk_count"] > 0
    assert generator.closed == 1


def test_invalid_native_tool_output_is_a_scored_200_response():
    text = '<tool_call>{"name":"lookup","arguments":'
    service, _, _ = _service("T0", text)
    response = service.complete(_payload(tools=[_tool()]))
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": text,
    }
    assert response["x_c2kv"]["native_parse_status"] == "invalid_tool_call"


def test_no_native_tools_stays_raw_and_never_parses_tool_markup():
    text = 'literal <tool_call>{"name":"quoted","arguments":{}}</tool_call>'
    service, _, _ = _service("T0", text)
    response = service.complete(_payload())
    assert response["choices"][0]["message"]["content"] == text
    assert "tool_calls" not in response["choices"][0]["message"]
    compression = response["x_c2kv"]["compression"]
    assert compression["reason"] == "no_native_tool_definitions"
    assert compression["budget_profile"] == "base-model-context-only"
    assert compression["coverage"]["tool_definition_coverage"] == 0.0
    assert response["x_c2kv"]["native_parse_status"] == "text_no_native_tools"


def test_t1_schema_without_descriptions_is_preserved_raw():
    service, generator, tokenizer = _service("T1", "plain text")
    response = service.complete(_payload(tools=[_tool(descriptive=False)]))
    compression = response["x_c2kv"]["compression"]
    assert compression["reason"] == "no_descriptive_tool_fields"
    assert compression["chunk_count"] == 0
    memory = generator.memories[-1]
    rendered = tokenizer.decode(memory.system_input_ids + memory.workspace_input_ids)
    assert '"name":"lookup"' in rendered
    assert compression["coverage"]["native_tool_definitions"] == 1
    assert compression["coverage"]["compressed_tool_definitions"] == 0


def test_appworld_controls_and_toolsandbox_alias_normalize_to_greedy():
    messages = [
        {"role": "system", "content": "Tools are described in this prompt."},
        {"role": "user", "content": "Open the task."},
        {"role": "assistant", "content": "I opened it."},
        {"role": "user", "content": "Continue."},
    ]
    service, _, _ = _service(
        "H1", "done", aliases=("gpt-4o-2024-05-13",)
    )
    response = service.complete(
        _payload(
            model="gpt-4o-2024-05-13",
            messages=messages,
            presence_penalty=0.5,
            seed=42,
            chat_template_kwargs={"enable_thinking": False},
        )
    )
    assert response["model"] == "gpt-4o-2024-05-13"
    controls = response["x_c2kv"]["generation_controls"]
    assert controls["client"]["presence_penalty"] == 0.5
    assert controls["client"]["seed"] == 42
    assert controls["normalized"]["presence_penalty"] == 0.0
    assert response["x_c2kv"]["model"] == "c2kv-next"
    assert service.health()["accepted_model_aliases"] == [
        "c2kv-next",
        "gpt-4o-2024-05-13",
    ]


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ace_orphan_execution_observation_is_preserved_as_opaque_history(variant):
    messages = [
        {"role": "system", "content": "Return ACE actions."},
        {"role": "user", "content": "Find the item."},
        {"role": "assistant", "content": "[Search(key='alpha')]"},
        {
            "role": "tool",
            "tool_call_id": "acebench-execution-1",
            "content": '[{"value":"alpha-17"}]',
        },
        {"role": "user", "content": "Use that value."},
    ]
    service, generator, tokenizer = _service(variant, "[Commit(value='alpha-17')]")
    response = service.complete(_payload(messages=messages))
    assert response["x_c2kv"]["source_profile"] == (
        "acebench-text-observation-opaque-v1"
    )
    memory = generator.memories[-1]
    rendered = tokenizer.decode(memory.system_input_ids + memory.workspace_input_ids)
    assert "acebench-execution-1" in rendered
    assert "alpha-17" in rendered


def test_full_mode_ignores_compressor_training_caps():
    config = ToolPreparationConfig(max_raw_tokens=1, max_target_tokens=1)
    service, _, _ = _service(
        "T0", "ok", mode="full", config=config, max_new_tokens=8
    )
    response = service.complete(_payload(max_tokens=8))
    assert response["x_c2kv"]["compression"]["reason"] == "mode_full"
    assert response["x_c2kv"]["compression"]["budget_profile"] == (
        "base-model-context-only"
    )


def test_unknown_alias_and_sampling_are_stable_client_errors():
    service, _, _ = _service("T0", "unused")
    with pytest.raises(LiveRequestError) as alias_error:
        service.complete(_payload(model="unknown"))
    assert alias_error.value.status == 400
    assert alias_error.value.code == "model_alias_mismatch"
    with pytest.raises(LiveRequestError) as sampling_error:
        service.complete(_payload(temperature=0.2))
    assert sampling_error.value.status == 400
    assert sampling_error.value.code == "greedy_only"


def _load_train_entry():
    path = REPOSITORY_ROOT / "agent" / "train_next_compression.py"
    spec = importlib.util.spec_from_file_location("live_test_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_real_checkpoint_serves_one_live_tool_request(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    train_entry = _load_train_entry()
    args = train_entry.arguments(
        ["--cpu_smoke", "--variant", "T0", "--output_dir", str(tmp_path / "unused")]
    )
    model, tokenizer = train_entry.build_model(args, torch.device("cpu"))
    config = ToolPreparationConfig()
    manifest = {
        "schema": SCHEMA,
        "training_profile": TRAINING_PROFILE,
        "variant": "T0",
        "compression_domain": "tool",
        "render_profile": EXPECTED_RENDER_PROFILES["T0"],
        "loss_profile": "decision-mean-complete-ce-v1",
        "ratios": [8, 12],
        "tokenizer": tokenizer_identity(tokenizer),
        "preparation": {"config": asdict(config)},
    }
    manifest_path = tmp_path / "training-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train_entry.configure_training_metadata(
        model,
        args,
        corpus_identity=sha256_file(manifest_path),
        render_profile=EXPECTED_RENDER_PROFILES["T0"],
        loss_profile="decision-mean-complete-ce-v1",
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    service = LiveNextCompressionService.from_checkpoint(
        checkpoint,
        manifest_path,
        device="cpu",
        dtype="float32",
        ratio=8,
        mode="compressed",
        model="c2kv-next",
        max_new_tokens=2,
        max_requests=1,
    )
    try:
        response = service.complete(_payload(tools=[_tool()]))
    finally:
        service.generator.close_session()
    assert response["object"] == "chat.completion"
    assert response["x_c2kv"]["variant"] == "T0"
    assert response["x_c2kv"]["compression"]["chunk_count"] > 0
    assert 0 <= response["usage"]["completion_tokens"] <= 2
