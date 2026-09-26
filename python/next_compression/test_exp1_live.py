from __future__ import annotations

from pathlib import Path

import pytest

from next_compression.live import LiveNextCompressionService, TrainingBinding
from next_compression.test_live import ByteChatTokenizer, FakeGenerator, _payload, _profile
from next_compression.tools import ToolPreparationConfig


def _tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }


def _hybrid_service(variant: str = "T0", *, top_k: int = 1, max_raw_tokens=None):
    tokenizer = ByteChatTokenizer()
    generator = FakeGenerator(tokenizer, "plain text")
    binding = TrainingBinding(Path("manifest.json"), "2" * 64, {}, ToolPreparationConfig())
    service = LiveNextCompressionService(
        generator,
        tokenizer,
        _profile(variant),
        binding,
        ratio=8,
        mode="compressed",
        model="c2kv-next",
        max_new_tokens=64,
        max_requests=5,
        tool_layout="hybrid",
        tool_top_k=top_k,
        max_raw_tokens=max_raw_tokens,
    )
    return service, generator, tokenizer


def test_hybrid_layout_keeps_the_ranked_schema_native_and_compresses_the_rest():
    service, generator, tokenizer = _hybrid_service(top_k=1)
    # The July ranker keeps identifiers whole ("book_flight" is one token), so
    # the description carries the query overlap here.
    tools = [_tool("lookup_weather", "Look up the weather."), _tool("book_flight", "Book a flight.")]
    messages = [
        {"role": "system", "content": "Use the available context."},
        {"role": "user", "content": "please book a flight"},
    ]
    response = service.complete(_payload(tools=tools, messages=messages))
    compression = response["x_c2kv"]["compression"]
    assert compression["tool_layout"]["layout"] == "hybrid"
    assert compression["tool_layout"]["k"] == 1
    assert compression["tool_layout"]["native_tool_indices"] == [1]
    assert compression["chunk_count"] == 1
    assert compression["coverage"]["compressed_tool_definitions"] == 1
    memory = generator.memories[-1]
    system = tokenizer.decode(memory.system_input_ids)
    assert "book_flight" in system and "lookup_weather" not in system
    encoded = "".join(tokenizer.decode(chunk.token_ids) for chunk in memory.chunks)
    assert "lookup_weather" in encoded
    health = service.health()
    assert health["tool_layout"] == "hybrid" and health["tool_top_k"] == 1


def test_hybrid_layout_with_k_covering_the_catalog_has_no_remainder():
    service, generator, _ = _hybrid_service(top_k=5)
    tools = [_tool("lookup_weather", "Weather."), _tool("book_flight", "Flights.")]
    response = service.complete(_payload(tools=tools))
    compression = response["x_c2kv"]["compression"]
    assert compression["tool_layout"]["native_tool_indices"] == [0, 1]
    assert compression["chunk_count"] == 0
    assert generator.memories[-1].chunks == ()


def test_hybrid_layout_requires_a_t0_checkpoint_and_valid_options():
    with pytest.raises(ValueError, match="T0 checkpoint"):
        _hybrid_service("T1")
    with pytest.raises(ValueError, match="tool_top_k"):
        _hybrid_service(top_k=0)


def test_max_raw_tokens_override_replaces_the_frozen_native_budget():
    service, _, _ = _hybrid_service(max_raw_tokens=123)
    assert service.binding.preparation_config.max_raw_tokens == 123
    assert service.health()["max_raw_tokens_override"] == 123
    with pytest.raises(ValueError, match="max_raw_tokens"):
        _hybrid_service(max_raw_tokens=0)
