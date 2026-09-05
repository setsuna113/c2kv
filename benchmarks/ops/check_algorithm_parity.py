#!/usr/bin/env python3
"""Check C2KV paper-level tensor semantics against the official implementation.

This is a small, model-free fixture.  It source-loads only the relevant
definitions from an official C2KV checkout and the SGLang checkout under test,
so it does not import either package or allocate model weights.

Checks:
  * dynamic-interleave extraction masks and gist position IDs;
  * ``none`` / ``mean`` / ``embed-mean`` residuals;
  * pre-RoPE K storage followed by absolute-position rephasing, including the
    original-length cursor advance across two compressed segments;
  * base, full-gist, and mixed ``QkV`` query projection routing;
  * rejection of positions beyond the available RoPE table.

Example:
  python benchmarks/ops/check_algorithm_parity.py \
      --official-repo /tmp/C2KV \
      --sglang-repo /path/to/sglang-c2kv \
      --device npu
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


def _load_torch(device_name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the parity fixture") from exc

    if device_name in {"auto", "npu"}:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if device_name == "npu":
                raise RuntimeError("--device npu requires torch_npu")
        else:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return torch, torch.device("npu:0")
            if device_name == "npu":
                raise RuntimeError("--device npu requested, but torch.npu is unavailable")

    return torch, torch.device("cpu")


def _source_definitions(path: Path, names: set[str], namespace: dict[str, Any]):
    """Compile selected definitions from *path* without importing its package."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    missing = names - found
    if missing:
        raise AssertionError(f"missing definitions in {path}: {sorted(missing)}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, source, tree


def _nested_function(path: Path, function_name: str, namespace: dict[str, Any]):
    """Compile one method as a free function while retaining its ``self`` arg."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {function_name} in {path}, found {len(matches)}"
        )
    node = matches[0]
    node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name], source, tree


def _dense_create_block_mask(torch):
    def create_block_mask(mask_mod, *, B, H, Q_LEN, KV_LEN, device, **_kwargs):
        q_idx = torch.arange(Q_LEN, device=device, dtype=torch.long).view(-1, 1)
        kv_idx = torch.arange(KV_LEN, device=device, dtype=torch.long).view(1, -1)
        dense = mask_mod(0, 0, q_idx, kv_idx)
        return dense.to(torch.bool).view(1, 1, Q_LEN, KV_LEN)

    return create_block_mask


def _paper_mask(torch, seq_len: int, ratio: int, overlap: int, device):
    """Independent transcription of paper Section 3.2's four mask blocks."""
    gist_len = math.ceil(seq_len / ratio)
    total_len = seq_len + gist_len
    idx = torch.arange(total_len, device=device, dtype=torch.long)
    q_idx = idx.view(-1, 1)
    kv_idx = idx.view(1, -1)
    q_is_token = q_idx < seq_len
    kv_is_token = kv_idx < seq_len
    token_to_token = q_is_token & kv_is_token & (q_idx >= kv_idx)
    gist_index = q_idx - seq_len
    chunk_begin = gist_index * ratio - overlap
    chunk_end = (gist_index + 1) * ratio
    gist_to_token = (~q_is_token) & kv_is_token & (
        ((kv_idx >= chunk_begin) & (kv_idx < chunk_end)) | (kv_idx < ratio)
    )
    gist_to_gist = (~q_is_token) & (~kv_is_token) & (q_idx >= kv_idx)
    return (token_to_token | gist_to_token | gist_to_gist).view(
        1, 1, total_len, total_len
    )


def _assert_equal(torch, actual, expected, label: str):
    if not torch.equal(actual, expected):
        mismatch = int((actual != expected).sum().item())
        raise AssertionError(
            f"{label}: {mismatch} elements differ; "
            f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        )


def _assert_close(torch, actual, expected, label: str):
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as exc:
        raise AssertionError(f"{label}: {exc}") from exc


def _load_gist_modules(
    torch, official_path: Path, local_path: Path, sglang_path: Path
):
    common = {
        "torch": torch,
        "math": math,
        "dataclass": dataclass,
        "Any": Any,
        "Callable": Callable,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    official_names = {
        "GistConfigMixin",
        "rotate_half",
        "apply_rotary_pos_emb",
        "_build_interleave_mask_vectorized",
        "get_apply_gist_residual_func",
        "_concat_gist_key_values",
    }
    sglang_names = {
        "GistConfig",
        "get_prepare_gist_input_func",
        "_apply_gist_residual_interleave",
        "_apply_none",
        "get_apply_gist_residual_func",
    }
    official, official_source, official_tree = _source_definitions(
        official_path, official_names, dict(common)
    )
    local, local_source, local_tree = _source_definitions(
        local_path, official_names, dict(common)
    )
    sglang_common = dict(common)
    sglang_common["create_block_mask"] = _dense_create_block_mask(torch)
    sglang, sglang_source, sglang_tree = _source_definitions(
        sglang_path, sglang_names, sglang_common
    )
    return (
        official,
        local,
        sglang,
        (official_source, official_tree),
        (local_source, local_tree),
        (sglang_source, sglang_tree),
    )


def _check_masks(torch, device, official, local, sglang):
    cases = [
        (1, 1, 0),
        (3, 4, 0),
        (7, 4, 0),
        (9, 4, 2),
        (17, 8, 5),
        (19, 4, 64),
    ]
    for seq_len, ratio, overlap in cases:
        input_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        official_mask, official_gist, official_pos = official[
            "_build_interleave_mask_vectorized"
        ](
            input_ids,
            attention,
            ratio,
            0,
            "right",
            "none",
            overlap,
            torch.float32,
        )
        local_mask, local_gist, local_pos = local[
            "_build_interleave_mask_vectorized"
        ](
            input_ids,
            attention,
            ratio,
            0,
            "right",
            "none",
            overlap,
            torch.float32,
        )
        cfg = sglang["GistConfig"](gist_overlap=overlap)
        sglang_mask, sglang_gist, sglang_pos = sglang[
            "get_prepare_gist_input_func"
        ](cfg)(input_ids, attention, ratio=ratio)
        official_bool = torch.isfinite(official_mask)
        if not isinstance(sglang_mask, torch.Tensor):
            raise AssertionError(
                "SGLang mask fixture did not return a dense tensor; "
                "the CPU create_block_mask shim was not used"
            )
        oracle = _paper_mask(torch, seq_len, ratio, overlap, device)
        label = f"len={seq_len},ratio={ratio},overlap={overlap}"
        _assert_equal(torch, official_bool, oracle, f"official mask {label}")
        _assert_equal(torch, torch.isfinite(local_mask), oracle, f"local mask {label}")
        _assert_equal(torch, sglang_mask.bool(), oracle, f"SGLang mask {label}")
        _assert_equal(torch, local_gist, official_gist, f"local gist mask {label}")
        _assert_equal(torch, local_pos, official_pos, f"local position IDs {label}")
        _assert_equal(torch, sglang_gist, official_gist, f"gist mask {label}")
        _assert_equal(torch, sglang_pos, official_pos, f"position IDs {label}")
    return len(cases)


def _paper_residual(
    torch, tokens, gist, ratio: int, residual_type: str, layer_idx: int
):
    apply_mean = residual_type == "mean" or (
        residual_type == "embed-mean" and layer_idx == 0
    )
    if not apply_mean:
        return gist
    chunks = [
        tokens[:, begin : begin + ratio].mean(dim=1, keepdim=True)
        for begin in range(0, tokens.shape[1], ratio)
    ]
    return torch.cat(chunks, dim=1) + gist


def _check_residuals(torch, device, official, local, sglang):
    checked = 0
    for seq_len, ratio in ((1, 4), (7, 4), (8, 4), (9, 4), (17, 8)):
        hidden = 6
        tokens = torch.arange(
            seq_len * hidden, device=device, dtype=torch.float32
        ).view(1, seq_len, hidden) / 17.0
        gist_len = math.ceil(seq_len / ratio)
        gist = -torch.arange(
            gist_len * hidden, device=device, dtype=torch.float32
        ).view(1, gist_len, hidden) / 19.0
        for residual_type in ("none", "mean", "embed-mean"):
            for layer_idx in (0, 1):
                official_cfg = official["GistConfigMixin"](
                    gist_type="dynamic-interleave",
                    gist_residual_type=residual_type,
                )
                sglang_cfg = sglang["GistConfig"](
                    gist_type="dynamic-interleave",
                    gist_residual_type=residual_type,
                )
                oracle = _paper_residual(
                    torch, tokens, gist, ratio, residual_type, layer_idx
                )
                expected = official["get_apply_gist_residual_func"](
                    official_cfg, layer_idx
                )(tokens, gist, ratio=ratio)
                _assert_close(
                    torch,
                    expected,
                    oracle,
                    "official residual="
                    f"{residual_type},layer={layer_idx},len={seq_len},ratio={ratio}",
                )
                local_value = local["get_apply_gist_residual_func"](
                    local["GistConfigMixin"](
                        gist_type="dynamic-interleave",
                        gist_residual_type=residual_type,
                    ),
                    layer_idx,
                )(tokens, gist, ratio=ratio)
                _assert_close(
                    torch,
                    local_value,
                    oracle,
                    "local residual="
                    f"{residual_type},layer={layer_idx},len={seq_len},ratio={ratio}",
                )
                actual = sglang["get_apply_gist_residual_func"](
                    sglang_cfg, layer_idx
                )(tokens, gist, ratio=ratio)
                _assert_close(
                    torch,
                    actual,
                    oracle,
                    f"residual={residual_type},layer={layer_idx},len={seq_len},ratio={ratio}",
                )
                checked += 1
    return checked


def _rope_tables(torch, max_position: int, head_dim: int, device):
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    frequencies = torch.arange(
        head_dim // 2, device=device, dtype=torch.float32
    ) / 11.0
    phase = positions.view(-1, 1) * (0.07 + frequencies.view(1, -1))
    return torch.cos(phase), torch.sin(phase)


def _rotate_neox(torch, tensor, cos, sin):
    half = tensor.shape[-1] // 2
    first, second = tensor[..., :half], tensor[..., half:]
    cos = cos.view(cos.shape[0], *([1] * (tensor.ndim - 2)), half)
    sin = sin.view(sin.shape[0], *([1] * (tensor.ndim - 2)), half)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class _Entry:
    def __init__(self, positions, layers, original_seq_len):
        self.positions = positions
        self.layers = layers
        self.gist_len = int(positions.numel())
        self.original_seq_len = int(original_seq_len)


class _Pool:
    def __init__(self, entries, num_layers):
        self.entries = entries
        self.num_layers = num_layers

    @staticmethod
    def get_position_ids(entry):
        return entry.positions

    @staticmethod
    def get_layer_kv(entry, layer_idx):
        return entry.layers[layer_idx]


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class _TokenPool:
    def __init__(self):
        self.writes = []

    def set_kv_buffer(self, *, layer, loc, cache_k, cache_v):
        self.writes.append((layer.layer_id, loc.clone(), cache_k.clone(), cache_v.clone()))


def _check_rephase(torch, device, official, local, injection_path: Path):
    os.environ.pop("C2KV_DEBUG_INJECT_DUMP", None)
    head_dim, heads, layers = 8, 2, 2
    lengths = (7, 3)
    local_positions = (
        torch.tensor([2, 5, 6], device=device, dtype=torch.long),
        torch.tensor([1, 2], device=device, dtype=torch.long),
    )
    max_gists = 3
    gist_mask = torch.tensor(
        [[True, True, True], [True, True, False]], device=device
    )
    gist_positions = torch.tensor(
        [[2, 5, 6], [1, 2, 0]], device=device, dtype=torch.long
    )
    generator = torch.Generator(device="cpu").manual_seed(20260905)

    # Generate on CPU for an identical deterministic stream on CPU and NPU.
    official_layers = []
    for _ in range(layers):
        key = torch.randn(2, heads, max_gists, head_dim, generator=generator).to(device)
        value = torch.randn(2, heads, max_gists, head_dim, generator=generator).to(device)
        official_layers.append((key, value))

    max_position = 32
    cos_half, sin_half = _rope_tables(torch, max_position, head_dim, device)

    def official_rotary(_x, position_ids):
        cos = cos_half[position_ids]
        sin = sin_half[position_ids]
        return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)

    prefix = 3
    official_out = official["_concat_gist_key_values"](
        object(),
        tuple(official_layers),
        gist_mask,
        gist_positions.clone(),
        official_rotary,
        prefix,
        int(gist_mask.sum().item()),
    )
    local_out = local["_concat_gist_key_values"](
        object(),
        tuple((key.clone(), value.clone()) for key, value in official_layers),
        gist_mask,
        gist_positions.clone(),
        official_rotary,
        prefix,
        int(gist_mask.sum().item()),
    )
    for layer_idx in range(layers):
        _assert_close(
            torch,
            local_out[layer_idx][0],
            official_out[layer_idx][0],
            f"local rephased K layer {layer_idx}",
        )
        _assert_close(
            torch,
            local_out[layer_idx][1],
            official_out[layer_idx][1],
            f"local unchanged V layer {layer_idx}",
        )

    def apply_rotary_emb(tensor, cos, sin, is_neox_style):
        if not is_neox_style:
            raise AssertionError("fixture expects NeoX-style split-half RoPE")
        return _rotate_neox(torch, tensor, cos, sin)

    injection_ns = {
        "torch": torch,
        "os": os,
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "C2KVEntry": Any,
        "C2KVPool": Any,
        "apply_rotary_emb": apply_rotary_emb,
    }
    semantics, _source, _tree = _source_definitions(
        injection_path.with_name("c2kv_semantics.py"),
        {"validate_rope_position_range"},
        dict(injection_ns),
    )
    injection_ns["validate_rope_position_range"] = semantics[
        "validate_rope_position_range"
    ]
    injection, _source, _tree = _source_definitions(
        injection_path,
        {"_validate_rope_positions", "inject_c2kv_gist"},
        injection_ns,
    )
    inject = injection["inject_c2kv_gist"]

    entries = []
    for segment_idx, (positions, original_len) in enumerate(
        zip(local_positions, lengths)
    ):
        segment_layers = []
        valid = positions.numel()
        for key, value in official_layers:
            segment_layers.append(
                (
                    key[segment_idx, :, :valid].transpose(0, 1).contiguous(),
                    value[segment_idx, :, :valid].transpose(0, 1).contiguous(),
                )
            )
        entries.append(_Entry(positions, segment_layers, original_len))

    cos_sin_cache = torch.cat((cos_half, sin_half), dim=-1)
    pool = _Pool(entries, layers)
    token_pool = _TokenPool()
    attn_layers = [_Layer(i) for i in range(layers)]
    cursor = prefix
    absolute_positions = []
    for entry in entries:
        absolute_positions.append(cursor + entry.positions)
        inject(
            entry,
            pool,
            cursor,
            torch.arange(entry.gist_len, device=device),
            token_pool,
            attn_layers,
            cos_sin_cache,
            True,
        )
        cursor += entry.original_seq_len

    expected_positions = torch.tensor([5, 8, 9, 11, 12], device=device)
    _assert_equal(
        torch,
        torch.cat(absolute_positions),
        expected_positions,
        "absolute gist positions across segments",
    )
    if cursor != prefix + sum(lengths):
        raise AssertionError(
            "position cursor advanced by compressed length, not original length"
        )

    for layer_idx in range(layers):
        writes = [write for write in token_pool.writes if write[0] == layer_idx]
        actual_k = torch.cat([write[2] for write in writes], dim=0).transpose(0, 1)
        actual_v = torch.cat([write[3] for write in writes], dim=0).transpose(0, 1)
        _assert_close(
            torch, actual_k, official_out[layer_idx][0], f"rephased K layer {layer_idx}"
        )
        _assert_close(
            torch, actual_v, official_out[layer_idx][1], f"unchanged V layer {layer_idx}"
        )

    # Position aliasing is never a valid fallback.  The implementation must
    # reject a logical position outside its RoPE table instead of clamping it.
    overflow = _Entry(
        torch.tensor([max_position], device=device),
        [
            (
                torch.zeros(1, heads, head_dim, device=device),
                torch.zeros(1, heads, head_dim, device=device),
            )
            for _ in range(layers)
        ],
        max_position + 1,
    )
    try:
        inject(
            overflow,
            _Pool([overflow], layers),
            0,
            torch.zeros(1, device=device, dtype=torch.long),
            _TokenPool(),
            attn_layers,
            cos_sin_cache,
            True,
        )
    except (ValueError, RuntimeError, IndexError):
        pass
    else:
        raise AssertionError("out-of-range RoPE position was silently accepted")

    return {"segments": len(entries), "layers": layers}


class _Projection:
    def __init__(self, output):
        self.output = output

    def __call__(self, hidden_states):
        return self.output[: hidden_states.shape[0]].clone(), None


class _ForwardBatch:
    def __init__(self, mask):
        self.c2kv_use_gist_projection = mask


def _official_case_contract(source: str):
    required = [
        f"use_gist and '{part}' in self.gist_param" for part in "QKV"
    ]
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        raise AssertionError(
            "official query projection case contract changed: " + ", ".join(missing)
        )


def _sglang_mixed_case_wiring(source: str, tree: ast.AST):
    """Return whether initializer preserves uppercase query-part semantics."""
    initializers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        and "c2kv_query_proj_parts" in (ast.get_source_segment(source, node) or "")
    ]
    if not initializers:
        raise AssertionError("cannot find SGLang C2KV query projection initializer")
    initializer = initializers[0]
    name_values = {}
    parts_value = None
    for node in ast.walk(initializer):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                name_values[target.id] = node.value
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "c2kv_query_proj_parts"
                and not (
                    isinstance(node.value, ast.Call)
                    and not node.value.args
                )
            ):
                parts_value = node.value
    if parts_value is None:
        raise AssertionError("cannot find populated c2kv_query_proj_parts assignment")

    visited = set()

    def expressions(root):
        yield root
        for child in ast.walk(root):
            if isinstance(child, ast.Name) and child.id in name_values:
                if child.id not in visited:
                    visited.add(child.id)
                    yield from expressions(name_values[child.id])

    expanded = list(expressions(parts_value))
    uppercase_literal = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(char in node.value for char in "QKV")
        for expression in expanded
        for node in ast.walk(expression)
    )
    uppercase_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"upper", "isupper"}
        for expression in expanded
        for node in ast.walk(expression)
    )
    projection_block = ast.get_source_segment(source, parts_value) or ""
    return uppercase_literal or uppercase_call, projection_block


def _sglang_mixed_case_rejection(sglang_root: Path):
    marker = "C2KV_GIST_PARAM_CASE_UNSUPPORTED"
    definitions = []
    for path in (sglang_root / "python/sglang/srt").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if marker in source and "def validate_gist_param" in source:
            definitions.append(path)
    server_args = sglang_root / "python/sglang/srt/server_args.py"
    wired = "validate_gist_param(" in server_args.read_text(encoding="utf-8")
    if len(definitions) == 1 and wired:
        return definitions[0].relative_to(sglang_root).as_posix()
    return None


def _local_lowercase_query_semantics(source: str):
    lowered = "gist_param = self.gist_param.lower()" in source
    lowercase_tests = all(
        f"use_gist and '{part}' in gist_param" in source for part in "qkv"
    )
    if lowered and lowercase_tests:
        return "gist"
    if all(
        f"use_gist and '{part}' in self.gist_param" in source for part in "QKV"
    ):
        return "base"
    raise AssertionError("cannot classify local lowercase qkv query semantics")


def _check_projection_routing(
    torch,
    device,
    official_model_path: Path,
    local_model_path: Path,
    sglang_model_path: Path,
    sglang_root: Path,
):
    official_source = official_model_path.read_text(encoding="utf-8")
    _official_case_contract(official_source)
    local_semantics = _local_lowercase_query_semantics(
        local_model_path.read_text(encoding="utf-8")
    )

    method, sglang_source, sglang_tree = _nested_function(
        sglang_model_path,
        "_c2kv_project_qkv",
        {"torch": torch, "os": os},
    )
    mixed_supported, block = _sglang_mixed_case_wiring(sglang_source, sglang_tree)
    mixed_rejection = _sglang_mixed_case_rejection(sglang_root)
    if not mixed_supported and mixed_rejection is None:
        compact = " ".join(block.split())[:240]
        raise AssertionError(
            "SGLang neither preserves official mixed-case QkV query semantics "
            f"nor rejects it explicitly; inspected: {compact}"
        )

    rows = 3
    hidden = torch.zeros(rows, 5, device=device)
    base = torch.arange(rows * 4, device=device, dtype=torch.float32).view(rows, 4)
    gist = base + 1000.0

    class Attention:
        q_size = 2
        kv_size = 1
        qkv_proj = _Projection(base)
        gist_qkv_proj = _Projection(gist)

    attention = Attention()

    # Lowercase qkv in the official implementation means base projections for
    # every ordinary/query token.  The SGLang request mask must reproduce that.
    attention.c2kv_query_proj_parts = frozenset("qkv")
    actual_base = method(
        attention,
        hidden,
        _ForwardBatch(torch.zeros(rows, device=device, dtype=torch.bool)),
    )
    _assert_equal(torch, actual_base, base, "lowercase qkv/base query routing")

    actual_gist = method(
        attention,
        hidden,
        _ForwardBatch(torch.ones(rows, device=device, dtype=torch.bool)),
    )
    _assert_equal(torch, actual_gist, gist, "explicit full gist query routing")

    # Official QkV switches Q and V but leaves K on the base projection.
    attention.c2kv_query_proj_parts = frozenset("qv")
    actual_mixed = method(
        attention,
        hidden,
        _ForwardBatch(torch.ones(rows, device=device, dtype=torch.bool)),
    )
    expected_mixed = torch.cat((gist[:, :2], base[:, 2:3], gist[:, 3:]), dim=-1)
    _assert_equal(torch, actual_mixed, expected_mixed, "mixed QkV query routing")
    return {
        "official_lowercase_qkv": "base",
        "local_lowercase_qkv": local_semantics,
        "official_uppercase_QkV": "gist_q_base_k_gist_v",
        "sglang_uppercase_QkV": (
            "gist_q_base_k_gist_v"
            if mixed_supported
            else f"explicitly_rejected_by:{mixed_rejection}"
        ),
    }


def _paths(args):
    official_root = Path(args.official_repo).expanduser().resolve()
    local_root = Path(args.local_repo).expanduser().resolve()
    sglang_root = Path(args.sglang_repo).expanduser().resolve()
    paths = {
        "official_gist": official_root / "python/models/gist_utils.py",
        "official_qwen3": official_root / "python/models/qwen3/modeling_qwen3.py",
        "local_gist": local_root / "python/models/gist_utils.py",
        "local_qwen3": local_root / "python/models/qwen3/modeling_qwen3.py",
        "sglang_gist": sglang_root / "python/sglang/srt/mem_cache/gist_utils.py",
        "sglang_injection": sglang_root / "python/sglang/srt/mem_cache/c2kv_injection.py",
        "sglang_qwen3": sglang_root / "python/sglang/srt/models/qwen3.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required source files not found: " + ", ".join(missing))
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-repo",
        required=True,
        help="Path to a checkout of https://github.com/s7a9/C2KV",
    )
    parser.add_argument(
        "--local-repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to the local C2KV checkout (default: repository containing this script)",
    )
    parser.add_argument(
        "--sglang-repo",
        required=True,
        help="Path to the SGLang C2KV checkout under test",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "npu"),
        default="auto",
        help="Tensor device; auto selects an available NPU, otherwise CPU",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        torch, device = _load_torch(args.device)
        paths = _paths(args)
        official, local, sglang, _official_ast, _local_ast, _sglang_ast = (
            _load_gist_modules(
                torch,
                paths["official_gist"],
                paths["local_gist"],
                paths["sglang_gist"],
            )
        )
        check_functions = {
            "masks": lambda: _check_masks(
                torch, device, official, local, sglang
            ),
            "residuals": lambda: _check_residuals(
                torch, device, official, local, sglang
            ),
            "rephase": lambda: _check_rephase(
                torch, device, official, local, paths["sglang_injection"]
            ),
            "projection": lambda: _check_projection_routing(
                torch,
                device,
                paths["official_qwen3"],
                paths["local_qwen3"],
                paths["sglang_qwen3"],
                Path(args.sglang_repo).expanduser().resolve(),
            ),
        }
        checks = {}
        failures = {}
        for name, check in check_functions.items():
            try:
                checks[name] = check()
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
        result = {
            "status": "pass" if not failures else "fail",
            "device": str(device),
            "torch": torch.__version__,
            "checks": checks,
        }
        if failures:
            result["failures"] = failures
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not failures else 1
    except Exception as exc:
        print(
            json.dumps(
                {"status": "fail", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
