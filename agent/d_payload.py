"""Payload container: real bytes accounting for repair payloads (prereg v2.5).

Replaces ``len(pickle)`` accounting.  A payload is

    header (JSON: codec id, shapes, dtypes, bit widths, n_blocks context)
    + one flat uint8 bitstream (all packed arrays concatenated)
    + optional float side-arrays (scales/zero points/means) stored as raw
      little-endian bytes

``nbytes`` is the exact sum of those fields — no pickle overhead, no free
riders.  Session-shared artifacts (PCA basis / dictionaries / regression
weights) are NOT part of a block payload: they go into ``SharedArtifacts``
and are amortized over the session's block count, per prereg v2.5
("neither double-billed nor free").
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import torch


class Payload:
    """One block's serialized repair payload with honest byte accounting."""

    def __init__(self, codec_id: str):
        self.codec_id = codec_id
        self.header: Dict[str, Any] = {"codec": codec_id}
        self._arrays: List[Dict[str, Any]] = []  # insertion-ordered parts

    def add_packed(self, name: str, codes: torch.Tensor, bits: int) -> None:
        """A bit-packed unsigned integer array (the bitstream)."""
        from inference.bits import pack_bits, packed_bytes

        buf = pack_bits(codes, bits)
        self._arrays.append({
            "name": name, "kind": "packed", "shape": list(codes.shape),
            "bits": bits, "buf": buf,
        })
        self.header[name] = {"kind": "packed", "shape": list(codes.shape), "bits": bits}

    def add_floats(self, name: str, tensor: torch.Tensor, dtype: str = "float16") -> None:
        """A side array of floats (scales / zero points / means).

        dtype: "float16" (default — side arrays are scales, f16 is plenty)
        or "float32".
        """
        if dtype not in ("float16", "float32"):
            raise ValueError(f"unsupported side dtype {dtype}")
        t = tensor.detach().cpu().contiguous()
        cast = t.to(torch.float16) if dtype == "float16" else t.to(torch.float32)
        raw = cast.numpy().tobytes()
        self._arrays.append({
            "name": name, "kind": "floats", "shape": list(t.shape),
            "dtype": dtype, "raw": raw,
        })
        self.header[name] = {"kind": "floats", "shape": list(t.shape), "dtype": dtype}

    # ---- byte accounting -------------------------------------------------
    def part_bytes(self) -> Dict[str, int]:
        header_bytes = len(json.dumps(self.header, separators=(",", ":")).encode("utf-8"))
        parts = {"__header__": header_bytes}
        for arr in self._arrays:
            if arr["kind"] == "packed":
                parts[arr["name"]] = arr["buf"].numel()
            else:
                parts[arr["name"]] = len(arr["raw"])
        return parts

    @property
    def nbytes(self) -> int:
        """Exact disk bytes for THIS block's payload."""
        return sum(self.part_bytes().values())

    def serialize(self) -> bytes:
        blob = json.dumps(self.header, separators=(",", ":")).encode("utf-8")
        blob += b"|"
        for arr in self._arrays:
            if arr["kind"] == "packed":
                blob += arr["buf"].numpy().tobytes()
            else:
                blob += arr["raw"]
        return blob

    @staticmethod
    def deserialize(blob: bytes) -> "Payload":
        sep = blob.index(b"|")
        header = json.loads(blob[:sep].decode("utf-8"))
        payload = Payload(header["codec"])
        payload.header = header
        payload._blob_tail = blob[sep + 1:]
        payload._offset = 0
        return payload

    def _in_memory(self, name: str):
        for arr in self._arrays:
            if arr["name"] == name:
                return arr
        return None

    def read_packed(self, name: str) -> torch.Tensor:
        from inference.bits import unpack_bits

        meta = self.header[name]
        n = 1
        for d in meta["shape"]:
            n *= d
        n_bytes = (n * meta["bits"] + 7) // 8
        arr = self._in_memory(name)
        if arr is not None:
            buf = arr["buf"]
        else:
            buf = torch.frombuffer(
                bytearray(self._blob_tail[self._offset:self._offset + n_bytes]),
                dtype=torch.uint8)
            self._offset += n_bytes
        return unpack_bits(buf, meta["bits"], n).reshape(meta["shape"])

    def read_floats(self, name: str) -> torch.Tensor:
        import numpy as np

        meta = self.header[name]
        n = 1
        for d in meta["shape"]:
            n *= d
        itemsize = {"float16": 2, "float32": 4}[meta["dtype"]]
        arr = self._in_memory(name)
        if arr is not None:
            raw = arr["raw"]
        else:
            raw = self._blob_tail[self._offset:self._offset + n * itemsize]
            self._offset += n * itemsize
        dt = np.float16 if meta["dtype"] == "float16" else np.float32
        return torch.from_numpy(np.frombuffer(raw, dtype=dt).copy()).reshape(meta["shape"])


class SharedArtifacts:
    """Session-level artifacts (PCA basis, dictionary, regression W).

    Bytes are amortized over the number of blocks in the session, exactly
    as prereg v2.5 requires: neither billed per block nor given for free.
    """

    def __init__(self):
        self._parts: Dict[str, int] = {}

    def put_tensor(self, name: str, tensor: torch.Tensor, dtype: str = "float16") -> int:
        t = tensor.detach().cpu().to(torch.float16 if dtype == "float16" else torch.float32)
        nbytes = t.numel() * (2 if dtype == "float16" else 4)
        self._parts[name] = nbytes
        return nbytes

    def put_bytes(self, name: str, n: int) -> None:
        self._parts[name] = n

    @property
    def total_bytes(self) -> int:
        return sum(self._parts.values())

    def amortized_bytes(self, n_blocks: int) -> float:
        if n_blocks <= 0:
            return float(self.total_bytes)
        return self.total_bytes / n_blocks

    def breakdown(self) -> Dict[str, int]:
        return dict(self._parts)
