"""CPU contracts for aligned and shifted 1024-token evidence windows."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from history_memory.events import EventStore  # noqa: E402
from memory_runtime.recovery.evidence_units import build_catalog  # noqa: E402


class OffsetTokenizer:
    @staticmethod
    def _pieces(text):
        return [match.span() for match in re.finditer(r"\s+|[\w]+|[^\w\s]", text)]

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(self._pieces(text))))

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        offsets = self._pieces(text)
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def one_message_store(session_id, content):
    return EventStore.from_messages(session_id, [{"role": "user", "content": content}])


def interval(unit):
    assert len(unit.provenance) == 1
    span = unit.provenance[0]
    return span.source_index, span.container_path, span.char_start, span.char_end


def test_aligned_windows_pack_complete_lines_and_preserve_exact_source_text():
    tokenizer = OffsetTokenizer()
    text = "".join(f"row-{index}\n" for index in range(900))
    units = build_catalog(
        one_message_store("aligned-lines", text), tokenizer, "tokens_1024_aligned"
    )

    assert len(units) > 1
    assert "".join(unit.text for unit in units) == text
    assert all(unit.token_count <= 1024 for unit in units)
    assert all(unit.text.endswith("\n") for unit in units)
    assert all(unit.metadata["alignment"] == "line" for unit in units)
    assert not any(unit.metadata["oversized_item_fallback"] for unit in units)


def test_aligned_windows_do_not_split_records_that_fit_individually():
    tokenizer = OffsetTokenizer()
    left = " ".join(f"left{index}" for index in range(350))
    right = " ".join(f"right{index}" for index in range(350))
    text = '[{"id":"left","body":"' + left + '"},{"id":"right","body":"' + right + '"}]'
    units = build_catalog(
        one_message_store("aligned-records", text), tokenizer, "tokens_1024_aligned"
    )

    left_units = [unit for unit in units if '"id":"left"' in unit.text]
    right_units = [unit for unit in units if '"id":"right"' in unit.text]
    assert len(left_units) == len(right_units) == 1
    assert left_units[0] is not right_units[0]
    assert all(unit.metadata["alignment"] == "record" for unit in units)
    assert not any(unit.metadata["oversized_item_fallback"] for unit in units)
    assert all(unit.token_count <= 1024 for unit in units)


def test_aligned_oversized_record_has_a_deterministic_bounded_fallback():
    tokenizer = OffsetTokenizer()
    body = " ".join(f"value{index}" for index in range(1400))
    text = '[{"id":"large","body":"' + body + '"}]'
    store = one_message_store("aligned-large", text)
    first = build_catalog(store, tokenizer, "tokens_1024_aligned")
    second = build_catalog(store, tokenizer, "tokens_1024_aligned")

    assert len(first) > 2
    assert [unit.unit_id for unit in first] == [unit.unit_id for unit in second]
    assert [unit.text for unit in first] == [unit.text for unit in second]
    assert all(unit.token_count <= 1024 for unit in first)
    assert any(unit.metadata["oversized_item_fallback"] for unit in first)
    assert all(unit.metadata["alignment"] == "record" for unit in first)


def test_shifted_windows_build_two_grids_without_duplicate_source_intervals():
    tokenizer = OffsetTokenizer()
    text = " ".join(f"token{index}" for index in range(2600))
    store = one_message_store("shifted", text)
    first = build_catalog(store, tokenizer, "tokens_1024_shifted")
    second = build_catalog(store, tokenizer, "tokens_1024_shifted")

    intervals = [interval(unit) for unit in first]
    assert {unit.metadata["window_offset_tokens"] for unit in first} == {0, 512}
    assert len(intervals) == len(set(intervals))
    assert all(unit.token_count <= 1024 for unit in first)
    assert [unit.unit_id for unit in first] == [unit.unit_id for unit in second]
    assert any(
        left.provenance[0].char_start < right.provenance[0].char_start < left.provenance[0].char_end
        for left in first
        for right in first
        if left.metadata["window_offset_tokens"] == 0
        and right.metadata["window_offset_tokens"] == 512
    )


def test_shifted_short_source_does_not_append_the_same_interval_twice():
    tokenizer = OffsetTokenizer()
    units = build_catalog(
        one_message_store("shifted-short", "short source"),
        tokenizer,
        "tokens_1024_shifted",
    )
    assert len(units) == 1
    assert units[0].metadata["window_offset_tokens"] == 0
