"""Torch-free corpus serialization shared by CPU preparation and H100 training."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from history_memory.packing import EncoderChunk, MemoryView, PackedMemory

SCHEMA = "next-compression-pretokenized-v1"
TRAINING_PROFILE = "next-compression-base-query-v1"
VARIANTS = ("H0", "H1", "H2", "H3", "T0", "T1")


@dataclass(frozen=True)
class PreparedDecision:
    memory: PackedMemory
    target_ids: tuple[int, ...]
    ratio: int
    weight: float = 1.0
    decision_id: str = ""
    target_weights: tuple[float, ...] | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_memory(memory: PackedMemory) -> dict[str, Any]:
    return asdict(memory)


def deserialize_memory(value: dict[str, Any]) -> PackedMemory:
    view = MemoryView(**{key: tuple(value["view"].get(key, ())) for key in
                       ("gist_event_ids", "raw_event_ids", "evidence_event_ids")})
    chunks = tuple(EncoderChunk(
        str(chunk["event_id"]), int(chunk["part_index"]), tuple(chunk["source_indices"]),
        int(chunk["source_token_start"]), int(chunk["source_token_end"]), tuple(chunk["token_ids"]),
    ) for chunk in value["chunks"])
    return PackedMemory(view, tuple(value["system_input_ids"]), tuple(value["workspace_input_ids"]),
                        tuple(value["raw_source_indices"]), chunks,
                        value.get("raw_layout_profile", "event-native-evidence-v1"))


def make_record(decision, *, session_key: str, source: str, split: str = "train",
                metadata: dict | None = None, target_weights=None) -> dict[str, Any]:
    weights = target_weights if target_weights is not None else getattr(decision, "target_weights", None)
    result = dict(decision_id=decision.decision_id, session_key=session_key, source=source, split=split,
                  ratio=decision.ratio, weight=decision.weight, target_ids=list(decision.target_ids),
                  memory=serialize_memory(decision.memory), metadata=metadata or {})
    if weights is not None:
        result["target_weights"] = list(weights)
    validate_record(result)
    return result


def validate_record(record: dict, *, ratios=(8, 12), vocab_size: int | None = None) -> None:
    if record.get("split") != "train":
        raise ValueError("Training artifacts must contain only train-split records")
    if record.get("ratio") not in ratios or not record.get("decision_id") or not record.get("session_key"):
        raise ValueError("Record needs a bound decision, session, and declared ratio")
    weight = record.get("weight", 1.0)
    if not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Decision weights must be finite and positive")
    target = record.get("target_ids", ())
    if not target:
        raise ValueError("An entire assistant continuation is required")
    weights = record.get("target_weights")
    if weights is not None and (len(weights) != len(target) or any(
            not isinstance(w, (float, int)) or not math.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("Every target token must retain finite positive supervision")
    memory = deserialize_memory(record["memory"])
    if not memory.workspace_input_ids:
        raise ValueError("A native generation prefix is required")
    token_arrays = [target, memory.system_input_ids, memory.workspace_input_ids]
    for chunk in memory.chunks:
        if not chunk.token_ids or chunk.source_token_start < 0 or chunk.source_token_end - chunk.source_token_start != len(chunk.token_ids):
            raise ValueError("Chunk source span must match its complete token payload")
        token_arrays.append(chunk.token_ids)
    for tokens in token_arrays:
        if any(type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in tokens):
            raise ValueError("Invalid token ID in pretokenized artifact")
    memory.gist_layout(record["ratio"])


class CorpusWriter:
    """Stream one immutable variant; publish its manifest only after completion."""

    def __init__(self, path: str | Path, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.path, self.variant = Path(path), variant
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise FileExistsError(f"Preparation output is not empty: {self.path}")
        self.pending = self.path / "records.jsonl.pending"
        self.handle = self.pending.open("wb")
        self.count = 0
        self.ratio_counts: dict[str, int] = {}
        self.source_counts: dict[str, int] = {}
        self.counters = {key: 0 for key in ("presented_encoder_tokens", "gist_tokens", "resident_kv_tokens", "supervised_tokens", "gist_bearing_records")}

    def write(self, record: dict) -> None:
        validate_record(record)
        self.handle.write((json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        self.count += 1
        ratio = str(record["ratio"])
        self.ratio_counts[ratio] = self.ratio_counts.get(ratio, 0) + 1
        source = record["source"]
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        memory = deserialize_memory(record["memory"])
        costs = memory.costs(record["ratio"])
        for key in ("presented_encoder_tokens", "gist_tokens", "resident_kv_tokens"):
            self.counters[key] += costs[key]
        self.counters["supervised_tokens"] += len(record["target_ids"])
        self.counters["gist_bearing_records"] += int(bool(memory.chunks))

    def finish(self, *, tokenizer: dict, preparation: dict, source_files: dict, audit: dict) -> dict:
        self.handle.close()
        if not self.count or set(self.ratio_counts) != {"8", "12"}:
            raise ValueError(f"Empty or incomplete ratio corpus: {self.variant}")
        records = self.path / "records.jsonl"
        self.pending.rename(records)
        manifest = dict(schema=SCHEMA, training_profile=TRAINING_PROFILE, variant=self.variant,
                        compression_domain="history" if self.variant.startswith("H") else "tool",
                        render_profile=preparation["render_profile"],
                        loss_profile=preparation.get("loss_profile", "decision-mean-complete-ce-v1"),
                        ratios=[8, 12], tokenizer=tokenizer, preparation=preparation, source_files=source_files,
                        records=dict(path=records.name, sha256=sha256_file(records), bytes=records.stat().st_size,
                                     count=self.count), ratio_counts=self.ratio_counts, source_counts=self.source_counts,
                        counters=self.counters, audit=audit)
        (self.path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifest


class SerializedCorpus:
    """Index JSONL offsets, keeping token payloads out of worker RAM until used."""

    def __init__(self, path: str | Path, tokenizer=None, expected_variant: str | None = None):
        path = Path(path)
        manifest_path = path / "manifest.json" if path.is_dir() else path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.identity = sha256_file(manifest_path)
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("training_profile") != TRAINING_PROFILE:
            raise ValueError("Unrecognized prepared training contract")
        if expected_variant is not None and self.manifest["variant"] != expected_variant:
            raise ValueError("Requested variant differs from prepared data")
        if self.manifest["ratios"] != [8, 12]:
            raise ValueError("This round is frozen to ratios 8 and 12")
        info = self.manifest["records"]
        self.records_path = manifest_path.parent / info["path"]
        if self.records_path.resolve().parent != manifest_path.parent.resolve():
            raise ValueError("Corpus records must be inside their manifest directory")
        if self.records_path.stat().st_size != info["bytes"] or sha256_file(self.records_path) != info["sha256"]:
            raise ValueError("Prepared records hash/size differs from the frozen manifest")
        vocab_size = None
        if tokenizer is not None:
            from history_memory.preparation import tokenizer_identity
            if tokenizer_identity(tokenizer)["sha256"] != self.manifest["tokenizer"]["sha256"]:
                raise ValueError("Training tokenizer differs from preparation")
            vocab_size = len(tokenizer)
        self.offsets, self.group_ids = [], []
        with self.records_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                validate_record(record, vocab_size=vocab_size)
                self.offsets.append(offset)
                self.group_ids.append(record["session_key"])
        if len(self.offsets) != info["count"]:
            raise ValueError("Prepared record count differs from manifest")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.records_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            record = json.loads(handle.readline())
        weights = record.get("target_weights")
        return PreparedDecision(deserialize_memory(record["memory"]), tuple(record["target_ids"]), record["ratio"],
                                float(record["weight"]), record["decision_id"],
                                tuple(weights) if weights is not None else None)
