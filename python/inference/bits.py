"""True bit-packer: quantized ints -> packed uint8 bitstream and back.

Without this, every "4-bit"/"3-bit" codec stores int16 codes and the bytes
axis of the codec tournament is fiction (the v1 d3 defect).  Layout is
MSB-first within each byte, values zero-extended to ``bits`` bits; the byte
length is ceil(n * bits / 8) — deterministic and CPU-testable.

Implementation note: no advanced-indexing scatter (duplicate byte indices
with ``|='' silently clobber instead of accumulating) — bytes are built by
an (n_bytes, 8) @ weights product in int64.
"""
from __future__ import annotations

import torch

_BYTE_WEIGHTS = [128, 64, 32, 16, 8, 4, 2, 1]
_BIT_POSITIONS = [7, 6, 5, 4, 3, 2, 1, 0]


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int codes (any shape, values < 2**bits) into uint8."""
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    flat = codes.reshape(-1).to(torch.int64)
    if flat.numel() and int(flat.max()) >= (1 << bits):
        raise ValueError(f"code {int(flat.max())} does not fit in {bits} bits")
    # (n, bits) MSB-first bit matrix
    shifts = torch.arange(bits - 1, -1, -1, dtype=torch.int64)
    stream = (flat.unsqueeze(1) >> shifts) & 1          # (n, bits) int64
    stream = stream.reshape(-1)
    pad = (-stream.numel()) % 8
    if pad:
        stream = torch.cat([stream, torch.zeros(pad, dtype=torch.int64)])
    weights = torch.tensor(_BYTE_WEIGHTS, dtype=torch.int64)
    bytes_val = stream.reshape(-1, 8) @ weights          # (n_bytes,) int64
    return bytes_val.to(torch.uint8)


def unpack_bits(buf: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Inverse of pack_bits: n unsigned ints from the packed uint8 stream."""
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    shifts = torch.tensor(_BIT_POSITIONS, dtype=torch.int64)
    stream = (buf.to(torch.int64).unsqueeze(1) >> shifts) & 1   # (n_bytes, 8)
    stream = stream.reshape(-1)[: n * bits].reshape(n, bits)
    weights = (1 << torch.arange(bits - 1, -1, -1, dtype=torch.int64))
    return (stream * weights).sum(dim=-1)


def packed_bytes(n_values: int, bits: int) -> int:
    """Exact byte size of a packed stream (the honest bytes axis)."""
    return (n_values * bits + 7) // 8
