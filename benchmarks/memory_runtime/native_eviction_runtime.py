"""Full-system native history-boundary reselection and greedy decode.

This runtime keeps the native system/tools prefix and the S0 current-input
suffix in full.  Only source tokens before ``raw_source_cutoff`` that are not
instruction tokens consume the history budget.  The freshly-prefilled history
cache is reselected independently at every decision; this is not a persistent
SnapKV or H2O serving implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.packing import MemoryView, PackedMemory, native_ids, pack_memory, visible_message

from .adapter import raw_source_cutoff
from .event_native_draft import NATIVE_DRAFT_VERSION, decode_native_generation
from .native_eviction import SUPPORTED_METHODS


NATIVE_EVICTION_RUNTIME_VERSION = "a-native-history-boundary-runtime-v1"
_RECENT_WINDOW = 64
_SNAPKV_POOL_KERNEL = 5
_SCORE_CHUNK_SIZE = 2048


class NativeEvictionStepError(RuntimeError):
    """A terminal decision failure carrying the partial scalar trace."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class NativeBoundaryInput:
    """Exact token coordinates derived from one observable source prefix."""

    store: EventStore
    tools: tuple[dict[str, Any], ...]
    memory: PackedMemory
    full_ids: tuple[int, ...]
    source_cutoff: int
    common_source_indices: tuple[int, ...]
    common_baseline_tokens: int
    history_boundary: int
    history_indices: tuple[int, ...]
    protected_history_indices: tuple[int, ...]
    common_prefix_indices: tuple[int, ...]
    current_suffix_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    mandatory_history_indices: tuple[int, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "source_cutoff": self.source_cutoff,
            "common_source_indices": list(self.common_source_indices),
            "common_baseline_tokens": self.common_baseline_tokens,
            "history_boundary": self.history_boundary,
            "history_tokens": len(self.history_indices),
            "protected_history_indices": list(self.protected_history_indices),
            "common_prefix_ranges": _indices_to_ranges(self.common_prefix_indices),
            "current_suffix_range": [self.history_boundary, len(self.full_ids)],
            "current_suffix_tokens": len(self.current_suffix_indices),
            "query_indices": list(self.query_indices),
            "mandatory_history_indices": list(self.mandatory_history_indices),
            "full_prompt_tokens": len(self.full_ids),
            "full_prompt_ids_sha256": _ids_sha256(self.full_ids),
            "common_accounting_contract": (
                "instruction source indices union raw_source_cutoff suffix; "
                "latest user and current-crossing events remain charged when before cutoff"
            ),
        }


@dataclass
class _SessionState:
    messages: tuple[str, ...]
    tools_json: str
    decision_index: int


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))


def _ids_sha256(ids: Sequence[int]) -> str:
    encoded = json.dumps(list(ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _indices_to_ranges(indices: Sequence[int]) -> list[list[int]]:
    values = tuple(int(value) for value in indices)
    if not values:
        return []
    if values != tuple(sorted(set(values))):
        raise ValueError("indices must be unique and increasing")
    ranges: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append([start, previous + 1])
            start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges


def _synchronize_device(torch: Any, device: Any) -> None:
    """Make phase timings include queued accelerator work."""

    if getattr(device, "type", None) == "npu":
        torch.npu.synchronize(device)
    elif getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _message_prefixes(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    full_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    prefixes = []
    previous: tuple[int, ...] = ()
    for end in range(1, len(messages) + 1):
        prefix = native_ids(
            tokenizer,
            messages[:end],
            tools=tools or None,
            generation=False,
        )
        if full_ids[: len(prefix)] != prefix:
            # Qwen renders consecutive tool messages as one user block.  A
            # truncated tool run temporarily closes that block with
            # <|im_end|>, so only its exact token LCP is a boundary in the
            # complete request.  Derive that boundary from tokens and permit
            # this exception only inside a consecutive tool run.
            next_is_tool = end < len(messages) and messages[end].get("role") == "tool"
            if messages[end - 1].get("role") != "tool" or not next_is_tool:
                raise ValueError("native source prefix does not match the generation prompt")
            common = 0
            for prefix_token, full_token in zip(prefix, full_ids):
                if prefix_token != full_token:
                    break
                common += 1
            prefix = prefix[:common]
            if not prefix or full_ids[: len(prefix)] != prefix:
                raise ValueError("native grouped-tool prefix has no exact token boundary")
        if prefix[: len(previous)] != previous:
            raise ValueError("native chat template is not prefix-stable across source messages")
        if len(prefix) <= len(previous):
            raise ValueError("native source message has no increasing token boundary")
        prefixes.append(prefix)
        previous = prefix
    return tuple(prefixes)


def plan_native_boundary_input(
    tokenizer: Any,
    payload: Mapping[str, Any],
    *,
    recent_window: int = _RECENT_WINDOW,
) -> NativeBoundaryInput:
    """Render and classify exact native token coordinates without model work."""

    _positive_int("recent_window", recent_window)
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    allowed = {"session_id", "decision_key", "messages", "tools"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"targets and privileged request fields are forbidden: {unknown!r}")
    session_id = payload.get("session_id")
    decision_key = payload.get("decision_key")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("an explicit nonempty session_id is required")
    if not isinstance(decision_key, str) or not decision_key:
        raise ValueError("an explicit nonempty decision_key is required")
    raw_messages = payload.get("messages")
    if (
        not isinstance(raw_messages, Sequence)
        or isinstance(raw_messages, (str, bytes, bytearray))
        or not raw_messages
        or any(not isinstance(message, Mapping) for message in raw_messages)
    ):
        raise ValueError("messages must be a nonempty sequence of mappings")
    raw_tools = payload.get("tools", ())
    if raw_tools is None:
        raw_tools = ()
    if (
        not isinstance(raw_tools, Sequence)
        or isinstance(raw_tools, (str, bytes, bytearray))
        or any(not isinstance(tool, Mapping) for tool in raw_tools)
    ):
        raise ValueError("tools must be a sequence of mappings")
    tools = tuple(_json_snapshot(tool) for tool in raw_tools)
    store = EventStore.from_messages(session_id, raw_messages)
    messages = tuple(visible_message(message) for message in store.messages)

    all_event_ids = tuple(event.event_id for event in store.events)
    view = MemoryView(
        gist_event_ids=(),
        raw_event_ids=all_event_ids,
        evidence_event_ids=(),
    )
    memory = pack_memory(store, view, tokenizer, tools=tools or None)
    full_ids = memory.system_input_ids + memory.workspace_input_ids
    independently_rendered = native_ids(
        tokenizer,
        messages,
        tools=tools or None,
        generation=True,
    )
    if independently_rendered != full_ids:
        raise RuntimeError("full native prompt disagrees with the full-original packer")

    source = [message.to_dict() for message in store.messages]
    cutoff = raw_source_cutoff(source)
    common_source_indices = {
        index
        for event in store.events
        if event.kind == "instruction"
        for index in event.source_indices
    } | set(range(cutoff, len(store.messages)))
    common_messages = [messages[index] for index in sorted(common_source_indices)]
    common_baseline_tokens = (
        len(native_ids(tokenizer, common_messages, tools=tools or None, generation=True))
        if common_messages
        else 0
    )

    prefixes = _message_prefixes(tokenizer, messages, tools, full_ids)
    system_prefix_tokens = len(memory.system_input_ids)
    history_boundary = len(prefixes[cutoff - 1]) if cutoff else system_prefix_tokens
    if not system_prefix_tokens <= history_boundary < len(full_ids):
        raise ValueError("native history/current boundary is empty or outside the prompt")

    spans = []
    previous_end = 0
    for prefix in prefixes:
        spans.append(tuple(range(previous_end, len(prefix))))
        previous_end = len(prefix)

    common_prefix = set(range(system_prefix_tokens))
    for source_index in common_source_indices:
        if source_index < cutoff:
            common_prefix.update(spans[source_index])
    common_prefix = {value for value in common_prefix if value < history_boundary}
    history_indices = tuple(
        value for value in range(history_boundary) if value not in common_prefix
    )
    charged_history_tokens = len(full_ids) - common_baseline_tokens
    if charged_history_tokens != len(history_indices):
        raise ValueError(
            "native token-coordinate history disagrees with the frozen S0 common baseline: "
            f"coordinate={len(history_indices)}, accounting={charged_history_tokens}"
        )

    suffix_sources = set(range(cutoff, len(store.messages)))
    mandatory_ids = {
        event.event_id
        for event in store.events
        if (
            not event.complete
            or event.kind == "instruction"
            or bool(set(event.source_indices) & suffix_sources)
        )
    }
    users = [event for event in store.events if event.kind == "user"]
    if users:
        mandatory_ids.add(users[-1].event_id)
    protected_history = set()
    history_set = set(history_indices)
    for event in store.events:
        if event.event_id in mandatory_ids:
            for source_index in event.source_indices:
                if source_index < cutoff:
                    protected_history.update(history_set.intersection(spans[source_index]))

    query_indices = history_indices[-min(recent_window, len(history_indices)) :]
    mandatory_history = tuple(sorted(protected_history | set(query_indices)))
    current_suffix = tuple(range(history_boundary, len(full_ids)))
    return NativeBoundaryInput(
        store=store,
        tools=tools,
        memory=memory,
        full_ids=full_ids,
        source_cutoff=cutoff,
        common_source_indices=tuple(sorted(common_source_indices)),
        common_baseline_tokens=common_baseline_tokens,
        history_boundary=history_boundary,
        history_indices=history_indices,
        protected_history_indices=tuple(sorted(protected_history)),
        common_prefix_indices=tuple(sorted(common_prefix)),
        current_suffix_indices=current_suffix,
        query_indices=query_indices,
        mandatory_history_indices=mandatory_history,
    )


def select_boundary_history_indices(
    scores: Sequence[float],
    *,
    history_indices: Sequence[int],
    mandatory_indices: Sequence[int],
    history_budget_tokens: int,
    method: str,
) -> tuple[int, ...]:
    """Choose per-head original history coordinates under an exact budget."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)!r}")
    _positive_int("history_budget_tokens", history_budget_tokens)
    history = tuple(int(value) for value in history_indices)
    mandatory = tuple(int(value) for value in mandatory_indices)
    values = tuple(float(value) for value in scores)
    if history != tuple(sorted(set(history))):
        raise ValueError("history_indices must be unique and increasing")
    if mandatory != tuple(sorted(set(mandatory))) or not set(mandatory) <= set(history):
        raise ValueError("mandatory_indices must be an increasing history subset")
    if len(values) != len(history) or any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite and aligned with history_indices")
    if len(mandatory) > history_budget_tokens:
        raise ValueError(
            "protected current/recent history exceeds the native history budget"
        )
    keep = min(len(history), history_budget_tokens)
    if keep == len(history):
        return history
    ranked_scores: Sequence[float] = values
    if method == "snapkv_style":
        radius = _SNAPKV_POOL_KERNEL // 2
        ranked_scores = tuple(
            sum(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            / _SNAPKV_POOL_KERNEL
            for index in range(len(values))
        )
    mandatory_set = set(mandatory)
    candidates = [index for index, coordinate in enumerate(history) if coordinate not in mandatory_set]
    candidates.sort(
        key=lambda index: (ranked_scores[index], history[index]),
        reverse=True,
    )
    chosen = mandatory_set | {
        history[index] for index in candidates[: keep - len(mandatory)]
    }
    selected = tuple(coordinate for coordinate in history if coordinate in chosen)
    if len(selected) != keep:
        raise RuntimeError("history selector did not fill its exact token budget")
    return selected


def _attention_scores_for_positions(
    torch: Any,
    query: Any,
    keys: Any,
    *,
    query_positions: Any,
    scaling: float,
) -> Any:
    """Mean exact causal attention mass for explicit original query positions."""

    _, query_heads, query_tokens, head_dim = query.shape
    _, kv_heads, prefix_tokens, key_dim = keys.shape
    if head_dim != key_dim or query_heads % kv_heads:
        raise RuntimeError("query/KV head geometry is incompatible")
    if tuple(query_positions.shape) != (query_tokens,):
        raise ValueError("query position tensor is not aligned with captured queries")
    groups = query_heads // kv_heads
    grouped = query.reshape(1, kv_heads, groups, query_tokens, head_dim)
    log_denom = None
    for start in range(0, prefix_tokens, _SCORE_CHUNK_SIZE):
        end = min(start + _SCORE_CHUNK_SIZE, prefix_tokens)
        logits = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped, keys[:, :, start:end, :]
        ).float().mul_(float(scaling))
        key_positions = torch.arange(start, end, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        logits.masked_fill_(~allowed[None, None, None, :, :], -torch.inf)
        part = torch.logsumexp(logits, dim=-1)
        log_denom = part if log_denom is None else torch.logaddexp(log_denom, part)
    if log_denom is None or not bool(torch.isfinite(log_denom).all()):
        raise RuntimeError("history-query attention normalization is non-finite")
    scores = torch.empty(
        (kv_heads, prefix_tokens), dtype=torch.float32, device=query.device
    )
    for start in range(0, prefix_tokens, _SCORE_CHUNK_SIZE):
        end = min(start + _SCORE_CHUNK_SIZE, prefix_tokens)
        logits = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped, keys[:, :, start:end, :]
        ).float().mul_(float(scaling))
        key_positions = torch.arange(start, end, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        logits.masked_fill_(~allowed[None, None, None, :, :], -torch.inf)
        probabilities = torch.exp(logits - log_denom.unsqueeze(-1))
        scores[:, start:end] = probabilities.mean(dim=(0, 2, 3))
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("history-query attention scores are non-finite")
    return scores


def _gather_prefix_cache(
    torch: Any,
    DynamicCache: Any,
    cache: Any,
    retained_indices: Sequence[Sequence[Sequence[int]]],
    *,
    config: Any,
) -> Any:
    """Gather independent original token coordinates for every KV head."""

    if len(retained_indices) != len(cache.layers):
        raise ValueError("retained layer count does not match the cache")
    gathered = []
    expected_tokens = None
    for layer_index, (layer, head_rows) in enumerate(zip(cache.layers, retained_indices)):
        keys, values = layer.keys, layer.values
        if keys is None or values is None or keys.shape != values.shape:
            raise RuntimeError(f"cache layer {layer_index} is incomplete")
        if len(head_rows) != keys.shape[1]:
            raise ValueError("retained KV-head count does not match the cache")
        rows = [tuple(int(value) for value in row) for row in head_rows]
        if any(row != tuple(sorted(set(row))) for row in rows):
            raise ValueError("retained original indices must be unique and increasing")
        if any(value < 0 or value >= keys.shape[2] for row in rows for value in row):
            raise ValueError("retained original index is outside the prefetched boundary")
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError("every KV head must retain the same number of tokens")
        layer_tokens = next(iter(lengths))
        if expected_tokens is None:
            expected_tokens = layer_tokens
        elif expected_tokens != layer_tokens:
            raise ValueError("every layer must retain the same number of tokens")
        index = torch.tensor(rows, dtype=torch.long, device=keys.device)
        index = index.unsqueeze(0).unsqueeze(-1).expand(
            1, keys.shape[1], layer_tokens, keys.shape[3]
        )
        gathered.append((keys.gather(2, index), values.gather(2, index)))
    result = DynamicCache(gathered, config=config)
    if result.get_seq_length() != expected_tokens:
        raise RuntimeError("gathered cache has an unexpected physical length")
    return result


def _forward_chunks(
    generator: Any,
    torch: Any,
    model: Any,
    cache: Any,
    ids: Sequence[int],
    *,
    original_start: int,
    chunk_size: int,
    capture_positions: Sequence[int] = (),
) -> tuple[Any, Any, list[Any], int]:
    """Forward a contiguous source span and optionally capture exact raw Q rows."""

    from .native_eviction import _physical_mask

    device = model.model.embed_tokens.weight.device
    capture = tuple(int(value) for value in capture_positions)
    capture_set = set(capture)
    query_parts: list[list[Any]] = [[] for _ in model.model.layers]
    outputs = None
    calls = 0
    for local_start in range(0, len(ids), chunk_size):
        local_end = min(local_start + chunk_size, len(ids))
        absolute_start = original_start + local_start
        absolute_end = original_start + local_end
        local_capture = [
            value - absolute_start
            for value in capture
            if absolute_start <= value < absolute_end
        ]
        handles = []
        if local_capture:
            local_index = torch.tensor(local_capture, dtype=torch.long, device=device)
            for layer_index, layer in enumerate(model.model.layers):
                def save_query(
                    _module: Any,
                    _args: Any,
                    output: Any,
                    *,
                    index: int = layer_index,
                    positions: Any = local_index,
                ) -> None:
                    query_parts[index].append(output.index_select(1, positions).detach())
                handles.append(layer.self_attn.q_norm.register_forward_hook(save_query))
        try:
            token_tensor = torch.tensor(
                tuple(int(value) for value in ids[local_start:local_end]),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            position_ids = torch.arange(
                absolute_start, absolute_end, dtype=torch.long, device=device
            ).unsqueeze(0)
            physical_past = cache.get_seq_length()
            outputs = model(
                input_ids=token_tensor,
                attention_mask=_physical_mask(
                    generator, local_end - local_start, physical_past, device
                ),
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                use_gist=False,
                logits_to_keep=1,
            )
            cache = outputs.past_key_values
            if cache.get_seq_length() != physical_past + local_end - local_start:
                raise RuntimeError("prefill cache append produced an unexpected length")
            calls += 1
        finally:
            for handle in handles:
                handle.remove()
    if capture_set and any(not parts for parts in query_parts):
        raise RuntimeError("Qwen3 query capture missed a model layer")
    raw_queries = [torch.cat(parts, dim=1) for parts in query_parts] if capture_set else []
    if raw_queries and any(query.shape[1] != len(capture) for query in raw_queries):
        raise RuntimeError("captured query count differs from the exact source coordinates")
    return cache, outputs, raw_queries, calls


def _full_selection_layers(model: Any, history: Sequence[int]) -> list[dict[str, Any]]:
    layer_count = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    row = list(history)
    return [
        {
            "layer_index": layer_index,
            "kv_heads": [
                {"kv_head": head_index, "selected_history_indices": row}
                for head_index in range(kv_heads)
            ],
        }
        for layer_index in range(layer_count)
    ]


def _generate_boundary_evicted(
    generator: Any,
    boundary: NativeBoundaryInput,
    *,
    method: str,
    history_budget_tokens: int,
    max_new_tokens: int,
    prefill_chunk_size: int,
) -> tuple[Any, dict[str, Any]]:
    """Prefill history, reselect, prefill current, then greedy decode."""

    from history_memory.inference import EventNativeGenerationResult
    from .native_eviction import _rotated_queries, _validate_runtime

    torch, DynamicCache, model = _validate_runtime(generator)
    validator = getattr(generator, "_validate_model_contract", None)
    if callable(validator):
        validator(4)
    vocab_size = int(model.config.vocab_size)
    eos_ids, eos_source = generator._resolve_eos_ids(None, vocab_size)
    parameter_version = generator.runtime.parameter_version
    device = model.model.embed_tokens.weight.device
    cache = DynamicCache(config=model.config)
    requires_scores = (
        len(boundary.history_indices) > history_budget_tokens
        and history_budget_tokens > len(boundary.mandatory_history_indices)
    )
    capture_positions = boundary.query_indices if requires_scores else ()
    _synchronize_device(torch, device)
    started = time.perf_counter()
    history_prefill_started = time.perf_counter()
    with generator._temporary_eval(), torch.inference_mode(), generator._base_autocast():
        cache, _history_outputs, raw_queries, history_calls = _forward_chunks(
            generator,
            torch,
            model,
            cache,
            boundary.full_ids[: boundary.history_boundary],
            original_start=0,
            chunk_size=prefill_chunk_size,
            capture_positions=capture_positions,
        )
        if cache.get_seq_length() != boundary.history_boundary:
            raise RuntimeError("uncompressed history prefill did not cover the exact boundary")
        _synchronize_device(torch, device)
        history_prefill_seconds = time.perf_counter() - history_prefill_started
        full_boundary_kv_bytes = boundary.history_boundary * int(generator.kv_bytes_per_token())

        selection_started = time.perf_counter()
        if len(boundary.history_indices) <= history_budget_tokens:
            selected_rows = [
                [tuple(boundary.history_indices)] * int(layer.keys.shape[1])
                for layer in cache.layers
            ]
            score_status = "all_history_fits"
        elif history_budget_tokens == len(boundary.mandatory_history_indices):
            selected_rows = [
                [tuple(boundary.mandatory_history_indices)] * int(layer.keys.shape[1])
                for layer in cache.layers
            ]
            score_status = "mandatory_only"
        else:
            if not raw_queries:
                raise RuntimeError("history eviction requires captured recent queries")
            query_positions = torch.tensor(
                boundary.query_indices, dtype=torch.long, device=device
            )
            rotated = _rotated_queries(
                model, raw_queries, query_positions.unsqueeze(0)
            )
            history_tensor = torch.tensor(
                boundary.history_indices, dtype=torch.long, device=device
            )
            selected_rows = []
            for layer_index, (layer, query) in enumerate(zip(cache.layers, rotated)):
                scores = _attention_scores_for_positions(
                    torch,
                    query,
                    layer.keys,
                    query_positions=query_positions,
                    scaling=float(model.model.layers[layer_index].self_attn.scaling),
                )
                layer_rows = []
                for head_index in range(scores.shape[0]):
                    values = scores[head_index].index_select(0, history_tensor).detach().cpu().tolist()
                    layer_rows.append(
                        select_boundary_history_indices(
                            values,
                            history_indices=boundary.history_indices,
                            mandatory_indices=boundary.mandatory_history_indices,
                            history_budget_tokens=history_budget_tokens,
                            method=method,
                        )
                    )
                selected_rows.append(layer_rows)
                del scores
            score_status = (
                "snapkv_kernel5_average_recent_query_attention"
                if method == "snapkv_style"
                else "recent_query_attention_topk_not_cumulative_h2o"
            )

        retained = []
        common_prefix = set(boundary.common_prefix_indices)
        for rows in selected_rows:
            retained.append(
                [tuple(sorted(common_prefix | set(row))) for row in rows]
            )
        selected_cache = _gather_prefix_cache(
            torch, DynamicCache, cache, retained, config=model.config
        )
        selected_history_tokens = min(
            len(boundary.history_indices), history_budget_tokens
        )
        selected_boundary_tokens = len(boundary.common_prefix_indices) + selected_history_tokens
        if selected_cache.get_seq_length() != selected_boundary_tokens:
            raise RuntimeError("selected boundary cache violates the exact history budget")
        _synchronize_device(torch, device)
        selection_seconds = time.perf_counter() - selection_started
        cache = selected_cache

        current_ids = boundary.full_ids[boundary.history_boundary :]
        current_prefill_started = time.perf_counter()
        cache, outputs, _unused, current_calls = _forward_chunks(
            generator,
            torch,
            model,
            cache,
            current_ids,
            original_start=boundary.history_boundary,
            chunk_size=prefill_chunk_size,
        )
        if outputs is None:
            raise RuntimeError("native generation prompt has no current suffix")
        resident_prompt_tokens = selected_boundary_tokens + len(current_ids)
        if cache.get_seq_length() != resident_prompt_tokens:
            raise RuntimeError("selected prompt cache has an unexpected physical length")
        _synchronize_device(torch, device)
        current_prefill_seconds = time.perf_counter() - current_prefill_started
        prefill_seconds = history_prefill_seconds + current_prefill_seconds

        generated: list[int] = []
        token_logprobs: list[float] = []
        finish_reason = "length"
        decode_calls = 0
        decode_started = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        while len(generated) < max_new_tokens:
            if tuple(logits.shape) != (1, vocab_size):
                raise RuntimeError("Qwen3 returned unexpected final-position logits")
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            next_token = int(torch.argmax(log_probs).item())
            next_logprob = float(log_probs[next_token].item())
            if not math.isfinite(next_logprob):
                raise RuntimeError("greedy token has a non-finite log probability")
            generated.append(next_token)
            token_logprobs.append(next_logprob)
            if next_token in eos_ids:
                finish_reason = "stop"
                break
            if len(generated) == max_new_tokens:
                break
            cache, outputs, _unused, calls = _forward_chunks(
                generator,
                torch,
                model,
                cache,
                (generated[-1],),
                original_start=len(boundary.full_ids) + len(generated) - 1,
                chunk_size=1,
            )
            decode_calls += calls
            logits = outputs.logits[:, -1, :]
        _synchronize_device(torch, device)
        decode_seconds = time.perf_counter() - decode_started

    if generator.runtime.parameter_version != parameter_version:
        raise RuntimeError("generation changed the runtime parameter version")
    kv_bytes_per_token = int(generator.kv_bytes_per_token())
    expected_final_tokens = resident_prompt_tokens + max(0, len(generated) - 1)
    if cache.get_seq_length() != expected_final_tokens:
        raise RuntimeError("final cache contains an unconsumed generated token")
    layers = []
    for layer_index, rows in enumerate(selected_rows):
        layers.append(
            {
                "layer_index": layer_index,
                "kv_heads": [
                    {
                        "kv_head": head_index,
                        "selected_history_indices": list(row),
                    }
                    for head_index, row in enumerate(rows)
                ],
            }
        )
    stats = {
        "schema": NATIVE_EVICTION_RUNTIME_VERSION,
        "method": method,
        "kv_bytes_per_token": kv_bytes_per_token,
        "finish_reason": finish_reason,
        "eos_token_ids": sorted(eos_ids),
        "eos_source": eos_source,
        "target_execution_strategy": "incremental_original_rope_positions",
        "full_history_prefill_tokens": boundary.history_boundary,
        "full_history_prefill_kv_bytes": full_boundary_kv_bytes,
        "full_history_prefill_is_transient_not_resident_budget": True,
        "selected_history_tokens_per_head": selected_history_tokens,
        "selected_history_kv_bytes": selected_history_tokens * kv_bytes_per_token,
        "selected_boundary_resident_tokens": selected_boundary_tokens,
        "current_suffix_prefill_tokens": len(current_ids),
        "resident_prompt_tokens": resident_prompt_tokens,
        "resident_prompt_kv_bytes": resident_prompt_tokens * kv_bytes_per_token,
        "prefill_forward_calls": history_calls + current_calls,
        "prefill_input_tokens": len(boundary.full_ids),
        "selection_score_status": score_status,
        "selection_query_indices": list(boundary.query_indices),
        "selection_future_current_suffix_used": False,
        "decode_forward_calls": decode_calls,
        "decode_input_tokens": decode_calls,
        "actual_model_input_tokens": len(boundary.full_ids) + decode_calls,
        "completion_tokens": len(generated),
        "resident_kv_tokens_final": cache.get_seq_length(),
        "resident_kv_logical_bytes_final": cache.get_seq_length() * kv_bytes_per_token,
        "prefill_seconds": prefill_seconds,
        "history_prefill_seconds": history_prefill_seconds,
        "current_suffix_prefill_seconds": current_prefill_seconds,
        "selection_seconds": selection_seconds,
        "decode_seconds": decode_seconds,
        "phase_timings_device_synchronized": True,
        "elapsed_sec": time.perf_counter() - started,
    }
    result = EventNativeGenerationResult(
        token_ids=tuple(generated),
        finish_reason=finish_reason,
        token_logprobs=tuple(token_logprobs),
        stats=stats,
    )
    return result, {"layers": layers, "stats": copy.deepcopy(stats)}


class NativeEvictionRunner:
    """One-generation EventNativeAPI runner for native boundary reselection."""

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        method: str,
        history_budget_bytes: int = 113246208,
        max_new_tokens: int = 4096,
        max_generation_calls: int = 96,
        max_sequence_tokens: int = 40960,
        prefill_chunk_size: int = 256,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)!r}")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        for name, value in (
            ("history_budget_bytes", history_budget_bytes),
            ("max_new_tokens", max_new_tokens),
            ("max_generation_calls", max_generation_calls),
            ("max_sequence_tokens", max_sequence_tokens),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            _positive_int(name, value)
        kv_bytes = int(generator.kv_bytes_per_token())
        if kv_bytes <= 0:
            raise ValueError("generator must report positive KV bytes per token")
        if getattr(generator, "decode_strategy", None) != "incremental":
            raise ValueError("native boundary eviction requires incremental decode")
        if getattr(generator, "prefill_chunk_size", None) != prefill_chunk_size:
            raise ValueError(
                "runner prefill_chunk_size must equal the loaded generator setting"
            )
        if history_budget_bytes // kv_bytes <= 0:
            raise ValueError("history budget must retain at least one native KV token")
        self.generator = generator
        self.tokenizer = tokenizer
        self.method = method
        self.kv_bytes_per_token = kv_bytes
        self.history_budget_bytes = history_budget_bytes
        self.history_budget_tokens = history_budget_bytes // kv_bytes
        self.history_budget_remainder_bytes = history_budget_bytes % kv_bytes
        self.max_new_tokens = max_new_tokens
        self.max_generation_calls = max_generation_calls
        self.max_sequence_tokens = max_sequence_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.generation_calls = 0
        self._completed: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._sessions: dict[str, _SessionState] = {}
        self._terminal_error: dict[str, str] | None = None

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._terminal_error is not None:
            raise RuntimeError(
                "this runner stopped after a terminal failure; automatic retry is disabled"
            )
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("an explicit nonempty session_id is required")
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError("an explicit nonempty decision_key is required")
        signature = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        key = (session_id, decision_key)
        cached = self._completed.get(key)
        if cached is not None:
            if cached[0] != signature:
                self.close()
                raise ValueError("decision key reused with different visible input")
            return copy.deepcopy(cached[1])

        started = time.perf_counter()
        record: dict[str, Any] = {
            "schema": NATIVE_EVICTION_RUNTIME_VERSION,
            "status": "started",
            "session_id": session_id,
            "decision_key": decision_key,
            "method": self.method,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.history_budget_bytes,
            "history_budget_tokens": self.history_budget_tokens,
            "history_budget_remainder_bytes": self.history_budget_remainder_bytes,
            "generation_trace": [],
            "response": None,
            "exact_recovery": None,
            "scope": (
                "Fresh history-boundary reselection for one visible decision; no recovery, "
                "tool execution, scorer, or persistent eviction state."
            ),
        }
        try:
            boundary = plan_native_boundary_input(
                self.tokenizer, payload, recent_window=_RECENT_WINDOW
            )
            self._validate_monotone(boundary)
            model_context = min(
                self.max_sequence_tokens,
                int(self.generator._model_context_length()),
            )
            if len(boundary.full_ids) + self.max_new_tokens > model_context:
                raise ValueError(
                    "requested native prompt plus generation exceeds the logical model context"
                )
            if len(boundary.mandatory_history_indices) > self.history_budget_tokens:
                raise ValueError(
                    "protected current/recent history exceeds the native history budget"
                )
            if self.generation_calls >= self.max_generation_calls:
                raise RuntimeError("finite generation-call cap exhausted before submission")

            self.generator.close_session()
            self.generation_calls += 1
            decision_index = self._sessions[session_id].decision_index
            trace = {
                "phase": "generation",
                "status": "started",
                "discarded": False,
                "controller": boundary.metadata(),
                "usage": None,
                "generation": None,
                "native_draft": None,
            }
            record["generation_trace"].append(trace)
            if len(boundary.history_indices) <= self.history_budget_tokens:
                result = self.generator.generate(
                    boundary.memory,
                    ratio=4,
                    max_new_tokens=self.max_new_tokens,
                )
                model = self.generator.runtime.base_model
                selected_layers = _full_selection_layers(model, boundary.history_indices)
                resident_prompt_tokens = len(boundary.full_ids)
                system_prefill_calls = result.stats.get("system_prefill_calls")
                system_prefill_tokens = result.stats.get("system_prefill_tokens")
                workspace_prefill_calls = result.stats.get("raw_prefill_forward_calls")
                workspace_prefill_tokens = result.stats.get("raw_prefill_input_tokens")
                if not all(
                    type(value) is int and value >= 0
                    for value in (
                        system_prefill_calls,
                        system_prefill_tokens,
                        workspace_prefill_calls,
                        workspace_prefill_tokens,
                    )
                ):
                    raise RuntimeError("Full identity generator omitted exact prefill counters")
                prefill_calls = system_prefill_calls + workspace_prefill_calls
                prefill_tokens = system_prefill_tokens + workspace_prefill_tokens
                if prefill_tokens != len(boundary.full_ids):
                    raise RuntimeError("Full identity prefill counters do not cover the prompt once")
                decode_calls = result.stats.get("one_token_decode_forward_calls")
                decode_tokens = result.stats.get("one_token_decode_input_tokens")
                eviction = {
                    "layers": selected_layers,
                    "stats": {
                        "schema": NATIVE_EVICTION_RUNTIME_VERSION,
                        "method": self.method,
                        "kv_bytes_per_token": self.kv_bytes_per_token,
                        "selection_score_status": "all_history_fits_full_identity_path",
                        "selection_query_indices": list(boundary.query_indices),
                        "selection_future_current_suffix_used": False,
                        "full_history_prefill_tokens": boundary.history_boundary,
                        "full_history_prefill_kv_bytes": (
                            boundary.history_boundary * self.kv_bytes_per_token
                        ),
                        "full_history_prefill_is_transient_not_resident_budget": False,
                        "full_prompt_prefill_tokens": len(boundary.full_ids),
                        "full_prompt_prefill_kv_bytes": (
                            len(boundary.full_ids) * self.kv_bytes_per_token
                        ),
                        "selected_history_tokens_per_head": len(boundary.history_indices),
                        "selected_history_kv_bytes": (
                            len(boundary.history_indices) * self.kv_bytes_per_token
                        ),
                        "current_suffix_prefill_tokens": len(boundary.current_suffix_indices),
                        "resident_prompt_tokens": resident_prompt_tokens,
                        "resident_prompt_kv_bytes": (
                            resident_prompt_tokens * self.kv_bytes_per_token
                        ),
                        "system_prefill_forward_calls": system_prefill_calls,
                        "system_prefill_input_tokens": system_prefill_tokens,
                        "workspace_prefill_forward_calls": workspace_prefill_calls,
                        "workspace_prefill_input_tokens": workspace_prefill_tokens,
                        "prefill_forward_calls": prefill_calls,
                        "prefill_input_tokens": prefill_tokens,
                        "decode_forward_calls": decode_calls,
                        "decode_input_tokens": decode_tokens,
                        "actual_model_input_tokens": (
                            prefill_tokens + decode_tokens
                            if type(decode_tokens) is int
                            else None
                        ),
                        "completion_tokens": len(result.token_ids),
                        "resident_kv_tokens_final": result.stats.get(
                            "resident_kv_tokens_final"
                        ),
                        "resident_kv_logical_bytes_final": result.stats.get(
                            "resident_kv_logical_bytes_final"
                        ),
                        "full_identity_generator_path": True,
                    },
                }
            else:
                result, eviction = _generate_boundary_evicted(
                    self.generator,
                    boundary,
                    method=self.method,
                    history_budget_tokens=self.history_budget_tokens,
                    max_new_tokens=self.max_new_tokens,
                    prefill_chunk_size=self.prefill_chunk_size,
                )
                resident_prompt_tokens = eviction["stats"]["resident_prompt_tokens"]

            usage = {
                "prompt_tokens": resident_prompt_tokens,
                "completion_tokens": len(result.token_ids),
                "total_tokens": resident_prompt_tokens + len(result.token_ids),
            }
            draft = decode_native_generation(
                self.tokenizer,
                result,
                call_id_prefix=f"d{decision_index}_r0",
            )
            trace.update(
                status="completed",
                usage=usage,
                generation={
                    "token_ids": list(result.token_ids),
                    "finish_reason": result.finish_reason,
                    "token_logprobs": list(result.token_logprobs),
                    "stats": copy.deepcopy(result.stats),
                },
                native_draft={"version": NATIVE_DRAFT_VERSION, **asdict(draft)},
            )
            selected_history_tokens = min(
                len(boundary.history_indices), self.history_budget_tokens
            )
            ratio = {
                "n_history": (
                    len(boundary.history_indices) / selected_history_tokens
                    if boundary.history_indices and selected_history_tokens
                    else None
                ),
                "n_total": len(boundary.full_ids) / resident_prompt_tokens,
                "full_history_tokens": len(boundary.history_indices),
                "selected_history_tokens_per_head": selected_history_tokens,
                "common_baseline_tokens": boundary.common_baseline_tokens,
                "resident_prompt_tokens": resident_prompt_tokens,
            }
            recovery = {
                "version": NATIVE_EVICTION_RUNTIME_VERSION,
                "status": "no_op",
                "reason": "native_eviction_single_generation_recovery_disabled",
                "decision_index": decision_index,
                "regeneration_allowed": False,
                "upgrade_count": 0,
            }
            record.update(
                status="ok",
                response={
                    "role": "assistant",
                    "content": draft.content,
                    "tool_calls": list(draft.tool_calls),
                    "reasoning_content": draft.reasoning_content,
                    "native_parse_status": draft.status,
                    "native_parse_reason": draft.reason,
                    "finish_reason": result.finish_reason,
                },
                exact_recovery=recovery,
                post_draft_exact_recovery_applied=False,
                boundary=boundary.metadata(),
                eviction=eviction,
                ratio=ratio,
                generation_attempts=1,
                generation_completed=1,
                generation_usage_total=usage,
                generation_usage_known=copy.deepcopy(usage),
                usage_scope=(
                    "Prompt tokens are the selected resident prompt. The complete uncompressed "
                    "history-boundary prefill is separate transient model work in eviction.stats."
                ),
                decision_runtime_seconds=time.perf_counter() - started,
            )
            self._completed[key] = (signature, copy.deepcopy(record))
            return record
        except Exception as error:
            if record["generation_trace"]:
                record["generation_trace"][-1]["status"] = "failed"
            record.update(
                status="failed",
                response=None,
                error={"type": type(error).__name__, "message": str(error)},
                generation_attempts=len(record["generation_trace"]),
                generation_completed=sum(
                    item["status"] == "completed" for item in record["generation_trace"]
                ),
                generation_usage_total={
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                decision_runtime_seconds=time.perf_counter() - started,
            )
            self._terminal_error = copy.deepcopy(record["error"])
            self.close()
            raise NativeEvictionStepError(str(error), record) from error

    def _validate_monotone(self, boundary: NativeBoundaryInput) -> None:
        session_id = boundary.store.session_id
        message_json = tuple(message.json_text for message in boundary.store.messages)
        tools_json = json.dumps(
            boundary.tools,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        state = self._sessions.get(session_id)
        if state is None:
            self._sessions[session_id] = _SessionState(message_json, tools_json, 1)
            return
        if state.tools_json != tools_json:
            raise ValueError("tools changed within a native eviction session")
        if len(message_json) < len(state.messages) or message_json[: len(state.messages)] != state.messages:
            raise ValueError("visible messages must extend the previous exact source prefix")
        self._sessions[session_id] = _SessionState(
            message_json, tools_json, state.decision_index + 1
        )

    def close(self) -> None:
        self.generator.close_session()


__all__ = [
    "NATIVE_EVICTION_RUNTIME_VERSION",
    "NativeBoundaryInput",
    "NativeEvictionRunner",
    "NativeEvictionStepError",
    "plan_native_boundary_input",
    "select_boundary_history_indices",
]
