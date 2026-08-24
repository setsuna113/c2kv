"""Cross-dataset text-unit dedup between training corpora and held-out eval sets.

Flattens one or more training corpora and held-out eval sets (BFCL,
ToolSandbox, ...) to TEXT UNITS, then flags exact and near duplicates across
the train/eval boundary so leaky training units can be dropped.

Input model: ``--train_inputs name=glob`` / ``--eval_inputs name=glob`` specs
where each glob yields jsonl/parquet files; records are flattened to units
with a global ``--unit`` strategy:

  * ``messages``: each message content of ``messages`` / ``conversations``
    (OpenAI or ShareGPT style), or of the ``gen_ai.input.messages`` /
    ``gen_ai.output.messages`` span attributes (agent-llm-traces parquet);
  * ``tools``: each tool schema of ``tools`` / ``functions`` /
    ``gen_ai.tool.definitions`` (direct or inside spans), serialized as
    canonical JSON;
  * ``raw``: the record's ``text``/``content``/``question``/``instruction``
    string field, else the whole record as canonical JSON.

Every unit gets {dataset, record_id, unit_id, unit_hash, text}; ``unit_hash``
is the sha1 of the normalized text (lowercase, whitespace collapsed,
stripped).

Dedup:
  * exact: shared ``unit_hash`` across the train/eval boundary;
  * near-dup: self-contained MinHash + banding LSH (char 5-gram shingles ->
    32-bit Rabin-style rolling hashes -> 128 universal-hash minhash values ->
    16 bands x 8 rows). No datasketch dependency. Candidate pairs are gated
    on the signature-estimated Jaccard (default >= 0.8). numpy is used for
    the shingle/signature inner loops when available (identical values from
    the pure-Python fallback).

Eval-side items are NEVER on the removal list: the removal list contains only
train-side units that exactly or nearly duplicate some eval unit.

Usage:
  python agent/dedup_cross_dataset.py \
      --train_inputs traces="./datasets/agent-llm-traces/data/*.parquet" \
      --bfcl_dir .foreman/ref/bfcl_data \
      --unit messages --out ./outputs/cross_dataset_dedup.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import random
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is an optional accelerator
    np = None


_MASK32 = (1 << 32) - 1
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _unit_hash(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if content is None:
        # OpenTelemetry gen_ai shape: {"role": ..., "parts": [{"type": "text", "content": ...}]}
        content = message.get("parts")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _span_attributes(span: Any) -> Dict[str, Any]:
    span = _json_loads(span, span)
    if not isinstance(span, dict):
        return {}
    attributes = span.get("attributes", span)
    attributes = _json_loads(attributes, attributes)
    return attributes if isinstance(attributes, dict) else {}


def _message_texts(value: Any) -> List[str]:
    messages = _json_loads(value, value)
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return []
    texts = []
    for message in messages:
        text = _message_text(message)
        if text.strip():
            texts.append(text)
    return texts


def _record_message_texts(record: Dict[str, Any]) -> List[str]:
    for key in ("messages", "conversations"):
        texts = _message_texts(record.get(key))
        if texts:
            return texts
    texts: List[str] = []
    spans = _json_loads(record.get("spans"), record.get("spans"))
    if isinstance(spans, list):
        for span in spans:
            attributes = _span_attributes(span)
            for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
                texts.extend(_message_texts(attributes.get(key)))
    return texts


def _as_tool_list(value: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(value, [])
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _record_tool_texts(record: Dict[str, Any]) -> List[str]:
    tools: List[Dict[str, Any]] = []
    for key in ("tools", "functions", "gen_ai.tool.definitions"):
        tools = _as_tool_list(record.get(key))
        if tools:
            break
    if not tools:
        spans = _json_loads(record.get("spans"), record.get("spans"))
        if isinstance(spans, list):
            for span in spans:
                tools = _as_tool_list(_span_attributes(span).get("gen_ai.tool.definitions"))
                if tools:
                    break
    return [json.dumps(tool, ensure_ascii=False, sort_keys=True) for tool in tools]


def _record_raw_text(record: Dict[str, Any]) -> List[str]:
    for key in ("text", "content", "question", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return [json.dumps(record, ensure_ascii=False, sort_keys=True)]


def _record_units(record: Dict[str, Any], unit: str) -> List[str]:
    if unit == "messages":
        return _record_message_texts(record)
    if unit == "tools":
        return _record_tool_texts(record)
    if unit == "raw":
        return _record_raw_text(record)
    raise ValueError(f"Unknown --unit strategy: {unit}")


def _record_id(record: Dict[str, Any], source_name: str, index: int) -> str:
    for key in ("id", "session_id", "trace_id", "qid", "uuid"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{source_name}-{index}"


def _parse_input_specs(specs: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Input spec must be name=glob, got: {spec!r}")
        name, pattern = spec.split("=", 1)
        name, pattern = name.strip(), pattern.strip()
        if not name or not pattern:
            raise ValueError(f"Input spec must be name=glob, got: {spec!r}")
        parsed.append((name, pattern))
    return parsed


def _iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        # Whole-file read: ParquetFile.iter_batches raises
        # ArrowNotImplementedError ("Nested data conversions not implemented
        # for chunked array outputs") on nested columns (e.g. traces-v2 spans).
        try:
            table = pq.read_table(path)
        except Exception:
            table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            if isinstance(row, dict):
                yield row
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = _json_loads(line, None)
            if isinstance(row, dict):
                yield row


def _load_units(specs: Optional[Sequence[str]], unit: str, side: str) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for dataset, pattern in _parse_input_specs(specs):
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"Input spec {dataset}={pattern!r} matched no files")
        for file_name in files:
            path = Path(file_name)
            for record_index, record in enumerate(_iter_records(path)):
                record_id = _record_id(record, f"{path.stem}", record_index)
                for unit_index, text in enumerate(_record_units(record, unit)):
                    units.append({
                        "dataset": dataset,
                        "record_id": record_id,
                        "unit_index": unit_index,
                        "unit_id": f"{dataset}:{record_id}:{unit_index}",
                        "unit_hash": _unit_hash(text),
                        "side": side,
                        "text": text,
                    })
    return units


def _bfcl_question_texts(record: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            text = _message_text(node)
            if text.strip():
                texts.append(text)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node.strip():
            texts.append(node)

    walk(record.get("question"))
    return texts


def _load_bfcl_units(bfcl_dir: str) -> List[Dict[str, Any]]:
    root = Path(bfcl_dir)
    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"--bfcl_dir {bfcl_dir} contains no *.json files")
    units: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = _json_loads(line, None)
                if not isinstance(record, dict):
                    continue
                record_id = str(record.get("id") or f"{path.stem}-{len(units)}")
                for unit_index, text in enumerate(_bfcl_question_texts(record)):
                    units.append({
                        "dataset": "bfcl",
                        "record_id": record_id,
                        "unit_index": unit_index,
                        "unit_id": f"bfcl:{record_id}:{unit_index}",
                        "unit_hash": _unit_hash(text),
                        "side": "eval",
                        "text": text,
                    })
    return units


# ---------------------------------------------------------------------------
# MinHash + banding LSH (self-contained, no datasketch).
# ---------------------------------------------------------------------------


def _hash32(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big")


_ROLLING_BASE = 257


def _rolling_hashes_python(data: bytes, shingle_size: int) -> Set[int]:
    """Rabin-style polynomial hashes of every ``shingle_size``-byte window."""
    top = _ROLLING_BASE ** (shingle_size - 1)
    value = 0
    for index in range(shingle_size):
        value = (value * _ROLLING_BASE + data[index]) & _MASK32
    values = {value}
    for index in range(1, len(data) - shingle_size + 1):
        value = (
            (value - data[index - 1] * top) * _ROLLING_BASE + data[index + shingle_size - 1]
        ) & _MASK32
        values.add(value)
    return values


def _rolling_hashes_numpy(data: bytes, shingle_size: int) -> Set[int]:
    """Same values as ``_rolling_hashes_python``, vectorized column-wise."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(arr, shingle_size)
    # Horner over the window columns; every intermediate stays < 2**40, so
    # uint64 never overflows regardless of shingle_size.
    values = np.zeros(windows.shape[0], dtype=np.uint64)
    for column in range(shingle_size):
        values = (values * _ROLLING_BASE + windows[:, column]) & np.uint64(_MASK32)
    return set(np.unique(values).tolist())


def _shingle_hashes(text: str, shingle_size: int, max_shingles: int) -> List[int]:
    """32-bit hashes of the char ``shingle_size``-grams of the normalized text.

    Shingles are hashed with a Rabin-style polynomial rolling hash over the
    UTF-8 bytes (identical values from the numpy and pure-Python paths);
    docs too short for one full window fall back to a blake2b of the text.
    Long docs are capped to the ``max_shingles`` smallest hash values
    (bottom-k): the selection function is identical for every unit, so
    near-duplicate docs keep highly overlapping shingle sets.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []
    data = normalized.encode("utf-8")
    if len(data) <= shingle_size:
        values = {_hash32(normalized)}
    elif np is not None:
        values = _rolling_hashes_numpy(data, shingle_size)
    else:
        values = _rolling_hashes_python(data, shingle_size)
    if len(values) > max_shingles:
        values = set(heapq.nsmallest(max_shingles, values))
    return sorted(values)


def _permutation_coefficients(num_perm: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    # Odd multipliers are bijective mod 2**32, which avoids degenerate perms.
    a = [rng.getrandbits(32) | 1 for _ in range(num_perm)]
    b = [rng.getrandbits(32) for _ in range(num_perm)]
    return a, b


def _empty_signature(num_perm: int) -> bytes:
    return b"\x00" * (4 * num_perm)


def _signature_python(shingle_hashes: Sequence[int], a: Sequence[int], b: Sequence[int]) -> bytes:
    if not shingle_hashes:
        return _empty_signature(len(a))
    sig = bytearray()
    for ai, bi in zip(a, b):
        best = _MASK32
        for h in shingle_hashes:
            value = (ai * h + bi) & _MASK32
            if value < best:
                best = value
        sig += struct.pack(">I", best)
    return bytes(sig)


def _signature_numpy(shingle_hashes: Sequence[int], a: Any, b: Any) -> bytes:
    if not shingle_hashes:
        return _empty_signature(int(a.shape[0]))
    hashes = np.asarray(shingle_hashes, dtype=np.uint64)
    values = (hashes[None, :] * a[:, None] + b[:, None]) & np.uint64(_MASK32)
    signature = values.min(axis=1).astype(">u4")
    return signature.tobytes()


class _MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        self.num_perm = num_perm
        a, b = _permutation_coefficients(num_perm, seed)
        self._a = a
        self._b = b
        if np is not None:
            self._a_np = np.asarray(a, dtype=np.uint64)
            self._b_np = np.asarray(b, dtype=np.uint64)
        else:
            self._a_np = self._b_np = None

    def signature(self, shingle_hashes: Sequence[int]) -> bytes:
        """128 minhash values packed as big-endian uint32 bytes."""
        if self._a_np is not None:
            return _signature_numpy(shingle_hashes, self._a_np, self._b_np)
        return _signature_python(shingle_hashes, self._a, self._b)


def _band_keys(signature: bytes, num_bands: int, rows_per_band: int) -> Iterator[Tuple[int, bytes]]:
    width = rows_per_band * 4
    for band in range(num_bands):
        yield band, signature[band * width : (band + 1) * width]


def _estimate_jaccard(signature_a: bytes, signature_b: bytes, num_perm: int) -> float:
    matches = sum(
        1
        for index in range(num_perm)
        if signature_a[4 * index : 4 * index + 4] == signature_b[4 * index : 4 * index + 4]
    )
    return matches / num_perm


# ---------------------------------------------------------------------------
# Dedup driver.
# ---------------------------------------------------------------------------


def _index_unique(units: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        entry = unique.setdefault(unit["unit_hash"], {"units": [], "datasets": set()})
        entry["units"].append(unit)
        entry["datasets"].add(unit["dataset"])
    return unique


def _representative_unit_id(entry: Dict[str, Any]) -> str:
    return min(unit["unit_id"] for unit in entry["units"])


def dedup(args: argparse.Namespace) -> Dict[str, Any]:
    if args.lsh_bands * args.lsh_rows != args.num_perm:
        raise ValueError(
            f"--lsh_bands * --lsh_rows must equal --num_perm "
            f"({args.lsh_bands} * {args.lsh_rows} != {args.num_perm})"
        )
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in (0, 1], got {args.threshold}")

    train_units = _load_units(args.train_inputs, args.unit, "train")
    eval_units = _load_units(args.eval_inputs, args.unit, "eval")
    if args.bfcl_dir:
        eval_units.extend(_load_bfcl_units(args.bfcl_dir))
    if not train_units:
        raise RuntimeError("No train-side units loaded; check --train_inputs")
    if not eval_units:
        raise RuntimeError("No eval-side units loaded; check --eval_inputs/--bfcl_dir")

    train_unique = _index_unique(train_units)
    eval_unique = _index_unique(eval_units)

    # Signatures are computed per unique normalized-text hash (exact-collapse),
    # then raw texts are dropped to bound memory.
    hasher = _MinHasher(args.num_perm, args.seed)
    signatures: Dict[str, bytes] = {}
    for unit_hash, entry in list(train_unique.items()) + list(eval_unique.items()):
        if unit_hash in signatures:
            continue
        signatures[unit_hash] = hasher.signature(
            _shingle_hashes(entry["units"][0]["text"], args.shingle_size, args.max_shingles)
        )
    for unit in train_units + eval_units:
        unit.pop("text", None)

    # Exact duplicates across the train/eval boundary.
    exact_dup_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    removal: Dict[str, Dict[str, Any]] = {}
    shared_hashes = sorted(set(train_unique) & set(eval_unique))
    for unit_hash in shared_hashes:
        train_entry = train_unique[unit_hash]
        eval_id = _representative_unit_id(eval_unique[unit_hash])
        for train_dataset in sorted(train_entry["datasets"]):
            for eval_dataset in sorted(eval_unique[unit_hash]["datasets"]):
                matched = [u for u in train_entry["units"] if u["dataset"] == train_dataset]
                exact_dup_counts[train_dataset][eval_dataset] += len(matched)
        for unit in train_entry["units"]:
            removal[unit["unit_id"]] = {
                "unit_id": unit["unit_id"],
                "dataset": unit["dataset"],
                "record_id": unit["record_id"],
                "unit_index": unit["unit_index"],
                "unit_hash": unit_hash,
                "match_type": "exact",
                "best_est_jaccard": 1.0,
                "matched_eval_unit": eval_id,
            }

    # Near-duplicates: banding LSH over the unique-hash signatures.
    buckets: List[Dict[bytes, List[Tuple[str, str]]]] = [defaultdict(list) for _ in range(args.lsh_bands)]
    sides: Dict[str, Set[str]] = defaultdict(set)
    for unit_hash in train_unique:
        sides[unit_hash].add("train")
    for unit_hash in eval_unique:
        sides[unit_hash].add("eval")
    all_entries: Dict[str, Dict[str, Any]] = {}
    for unit_hash, entry in train_unique.items():
        all_entries.setdefault(unit_hash, entry)
    for unit_hash, entry in eval_unique.items():
        all_entries.setdefault(unit_hash, entry)
    for unit_hash in all_entries:
        signature = signatures[unit_hash]
        for band, key in _band_keys(signature, args.lsh_bands, args.lsh_rows):
            buckets[band][key].append((unit_hash, "train" if "train" in sides[unit_hash] else "eval"))

    candidate_pairs: Set[Tuple[str, str]] = set()
    skipped_oversize_buckets = 0
    for band_buckets in buckets:
        for members in band_buckets.values():
            if len(members) < 2:
                continue
            train_hashes = sorted({h for h, side in members if side == "train"})
            eval_hashes = sorted({h for h, side in members if "eval" in sides[h]})
            if not train_hashes or not eval_hashes:
                continue
            if len(train_hashes) * len(eval_hashes) > args.max_bucket_pairs:
                skipped_oversize_buckets += 1
                continue
            for train_hash in train_hashes:
                for eval_hash in eval_hashes:
                    if train_hash != eval_hash:
                        candidate_pairs.add((train_hash, eval_hash))

    near_pairs: List[Dict[str, Any]] = []
    for train_hash, eval_hash in sorted(candidate_pairs):
        est = _estimate_jaccard(signatures[train_hash], signatures[eval_hash], args.num_perm)
        if est < args.threshold:
            continue
        train_entry = train_unique[train_hash]
        eval_entry = eval_unique[eval_hash]
        near_pairs.append({
            "train_unit": _representative_unit_id(train_entry),
            "eval_unit": _representative_unit_id(eval_entry),
            "est_jaccard": round(est, 4),
            "train_units": len(train_entry["units"]),
            "eval_units": len(eval_entry["units"]),
        })
        eval_id = _representative_unit_id(eval_entry)
        for unit in train_entry["units"]:
            existing = removal.get(unit["unit_id"])
            if existing is None:
                removal[unit["unit_id"]] = {
                    "unit_id": unit["unit_id"],
                    "dataset": unit["dataset"],
                    "record_id": unit["record_id"],
                    "unit_index": unit["unit_index"],
                    "unit_hash": train_hash,
                    "match_type": "near",
                    "best_est_jaccard": round(est, 4),
                    "matched_eval_unit": eval_id,
                }
            elif est > existing["best_est_jaccard"]:
                existing["best_est_jaccard"] = round(est, 4)
                existing["matched_eval_unit"] = eval_id

    near_pairs.sort(key=lambda item: (-item["est_jaccard"], item["train_unit"], item["eval_unit"]))
    total_near_dup_pairs = len(near_pairs)
    near_pairs = near_pairs[: args.max_pairs]

    removal_list = sorted(removal.values(), key=lambda item: item["unit_id"])
    train_unit_ids = {unit["unit_id"] for unit in train_units}
    leaked_eval = [item for item in removal_list if item["unit_id"] not in train_unit_ids]
    if leaked_eval:
        raise RuntimeError(f"Eval-side units leaked into the removal list: {leaked_eval[:5]}")

    train_unit_counts: Dict[str, int] = defaultdict(int)
    eval_unit_counts: Dict[str, int] = defaultdict(int)
    for unit in train_units:
        train_unit_counts[unit["dataset"]] += 1
    for unit in eval_units:
        eval_unit_counts[unit["dataset"]] += 1

    return {
        "metadata": {
            "unit": args.unit,
            "threshold": args.threshold,
            "num_perm": args.num_perm,
            "lsh_bands": args.lsh_bands,
            "lsh_rows": args.lsh_rows,
            "shingle_size": args.shingle_size,
            "max_shingles": args.max_shingles,
            "seed": args.seed,
            "signature_backend": "numpy" if hasher._a_np is not None else "python",
            "train_units": dict(sorted(train_unit_counts.items())),
            "eval_units": dict(sorted(eval_unit_counts.items())),
            "unique_train_hashes": len(train_unique),
            "unique_eval_hashes": len(eval_unique),
            "shared_exact_hashes": len(shared_hashes),
            "candidate_pairs": len(candidate_pairs),
            "skipped_oversize_buckets": skipped_oversize_buckets,
            "near_dup_pair_count": total_near_dup_pairs,
            "near_dup_pairs_reported": len(near_pairs),
            "removal_count": len(removal_list),
        },
        "exact_dup_counts": {
            train_dataset: dict(sorted(counts.items()))
            for train_dataset, counts in sorted(exact_dup_counts.items())
        },
        "near_dup_pairs": near_pairs,
        "removal_list": removal_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset exact + MinHash near-dup dedup of text units between train corpora and held-out eval sets."
    )
    parser.add_argument("--train_inputs", nargs="+", help="name=glob specs for training corpora (jsonl/parquet)")
    parser.add_argument("--eval_inputs", nargs="+", help="name=glob specs for held-out eval corpora (jsonl/parquet)")
    parser.add_argument("--bfcl_dir", help="Directory of BFCL *.json files; question texts become eval units")
    parser.add_argument("--unit", choices=["messages", "tools", "raw"], default="messages")
    parser.add_argument("--out", default="./outputs/cross_dataset_dedup.json")
    parser.add_argument("--threshold", type=float, default=0.8, help="Estimated Jaccard gate for near-dups")
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--lsh_bands", type=int, default=16)
    parser.add_argument("--lsh_rows", type=int, default=8)
    parser.add_argument("--shingle_size", type=int, default=5)
    parser.add_argument("--max_shingles", type=int, default=4096)
    parser.add_argument("--max_pairs", type=int, default=10000, help="Cap on reported near_dup_pairs")
    parser.add_argument("--max_bucket_pairs", type=int, default=20000, help="Skip LSH buckets whose train x eval cross product exceeds this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = dedup(args)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    print()
    print("| train dataset | eval dataset | exact dup train units |")
    print("|---|---|---:|")
    for train_dataset, counts in result["exact_dup_counts"].items():
        for eval_dataset, count in counts.items():
            print(f"| {train_dataset} | {eval_dataset} | {count} |")


if __name__ == "__main__":
    main()
