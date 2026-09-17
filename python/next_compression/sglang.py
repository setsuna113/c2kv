"""Validated SGLang transport for next-compression checkpoints.

This module reads checkpoint configuration and tokenizer files only. Model
weights stay owned by the shared C2KV SGLang process.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from history_memory.packing import PACKING_VERSION, RAW_LAYOUT_PROFILE, PackedMemory

from .common import sha256_file
from .inference import _read_json, _validate_next_config
from .vendor.sglang_transport import sglang_generator as _transport


MODEL_INFO_MAX_BYTES = 2 * 1024 * 1024
MODEL_INFO_TIMEOUT_SECONDS = 30.0
GENERATION_TIMEOUT_SECONDS = 600.0
CAPABILITY_SCHEMA = "c2kv-native-packed-capability-v1"
CAPABILITY_ENDPOINT = "/v1/c2kv/native_generate"
WEIGHT_VERSION_PREFIX = "next-compression:"
TRANSPORT_MANIFEST_SCHEMA = "next-compression-sglang-transport-vendor-v1"


class SGLangCheckpointBindingError(ValueError):
    """The shared engine does not serve the requested checkpoint contract."""


@dataclass
class _RequestState:
    session_id: str | None
    used: bool = False
    transport: _transport.SGLangEventNativeGenerator | None = None


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_dtype(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("dtype must be float32, bfloat16, or float16")
    aliases = {
        "float32": "float32",
        "fp32": "float32",
        "torch.float32": "float32",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "torch.bfloat16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "torch.float16": "float16",
    }
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise ValueError("dtype must be float32, bfloat16, or float16") from error


def _validate_device(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("device must be explicit and nonempty")
    if value not in {"cpu", "cuda", "npu"}:
        raise ValueError("device must be cpu, cuda, or npu")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SGLangCheckpointBindingError(f"{name} must be a JSON object")
    try:
        copied = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SGLangCheckpointBindingError(
            f"{name} must contain finite JSON data"
        ) from error
    if not isinstance(copied, dict):
        raise SGLangCheckpointBindingError(f"{name} must be a JSON object")
    return copied


def _read_endpoint_json(upstream: str, endpoint: str, label: str) -> dict[str, Any]:
    base_url = _transport._base_url(upstream)
    request = Request(
        base_url + endpoint,
        headers={"Accept": "application/json"},
        method="GET",
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=MODEL_INFO_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            body = response.read(MODEL_INFO_MAX_BYTES + 1)
    except HTTPError as error:
        raise SGLangCheckpointBindingError(
            f"SGLang {endpoint} returned HTTP {error.code}"
        ) from error
    except (OSError, URLError) as error:
        raise SGLangCheckpointBindingError(
            f"Cannot read SGLang {endpoint}: {error}"
        ) from error
    if status != 200:
        raise SGLangCheckpointBindingError(
            f"SGLang {endpoint} returned HTTP {status}"
        )
    if len(body) > MODEL_INFO_MAX_BYTES:
        raise SGLangCheckpointBindingError(
            f"SGLang {endpoint} response is too large"
        )
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SGLangCheckpointBindingError(
            f"SGLang {endpoint} did not return UTF-8 JSON"
        ) from error
    return _json_object(value, label)


def _read_model_info(upstream: str) -> dict[str, Any]:
    return _read_endpoint_json(upstream, "/model_info", "SGLang /model_info")


def _read_server_info(upstream: str) -> dict[str, Any]:
    return _read_endpoint_json(upstream, "/server_info", "SGLang /server_info")


def _expected_binding_fields(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "model_path": str(checkpoint_path),
        "tokenizer_path": str(checkpoint_path),
        "weight_version": WEIGHT_VERSION_PREFIX + config_sha256,
        "dtype": dtype,
        "kv_cache_dtype": dtype,
        "gist_parameter_dtype": "float32",
        "gist_compute_dtype": dtype,
        "gist_type": config.get("gist_type"),
        "gist_param": config.get("gist_param"),
        "gist_extra_embed_num": config.get("gist_extra_embed_num", 1),
        "gist_residual_type": config.get("gist_residual_type"),
        "gist_overlap": config.get("gist_overlap", 0),
        "pic_enabled": bool(config.get("pic_enabled", False)),
        "pic_param": config.get("pic_param", "qkv"),
        "query_projection": "base",
    }


def _validate_eos_token_ids(
    value: Any, *, source: str, vocab_size: int | None
) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError(f"{source} must be an integer or sequence of integers")
    values = (value,) if isinstance(value, int) else tuple(value or ())
    if not values or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        or (vocab_size is not None and item >= vocab_size)
        for item in values
    ):
        raise ValueError(f"{source} must declare valid eos_token_id values")
    return tuple(dict.fromkeys(values))


def _checkpoint_eos_token_ids(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[tuple[int, ...], str]:
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        vocab_size = None
    generation_path = checkpoint_path / "generation_config.json"
    if generation_path.is_file():
        generation_config = _read_json(generation_path)
        if generation_config.get("eos_token_id") is not None:
            source = "checkpoint_generation_config.eos_token_id"
            return (
                _validate_eos_token_ids(
                    generation_config["eos_token_id"],
                    source=source,
                    vocab_size=vocab_size,
                ),
                source,
            )
    if config.get("eos_token_id") is not None:
        source = "checkpoint_model_config.eos_token_id"
        return (
            _validate_eos_token_ids(
                config["eos_token_id"], source=source, vocab_size=vocab_size
            ),
            source,
        )
    source = "checkpoint_tokenizer.eos_token_id"
    return (
        _validate_eos_token_ids(
            tokenizer.eos_token_id, source=source, vocab_size=vocab_size
        ),
        source,
    )


def _validate_model_info(
    model_info: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    dtype: str,
) -> dict[str, Any]:
    info = _json_object(model_info, "SGLang /model_info")
    expected_binding = _expected_binding_fields(
        checkpoint_path, config, config_sha256, dtype
    )
    expected_weight_version = expected_binding["weight_version"]
    if info.get("model_path") != str(checkpoint_path):
        raise SGLangCheckpointBindingError(
            "SGLang /model_info model_path differs from the requested checkpoint"
        )
    if info.get("weight_version") != expected_weight_version:
        raise SGLangCheckpointBindingError(
            "SGLang /model_info weight_version differs from checkpoint config identity"
        )
    for field in ("model_type", "architectures"):
        if info.get(field) != config.get(field):
            raise SGLangCheckpointBindingError(
                f"SGLang /model_info {field} differs from checkpoint config"
            )

    native = _json_object(
        info.get("c2kv_native_packed"),
        "SGLang /model_info.c2kv_native_packed",
    )
    expected_capability = {
        "schema": CAPABILITY_SCHEMA,
        "enabled": True,
        "endpoint": CAPABILITY_ENDPOINT,
        "packing_version": PACKING_VERSION,
        "raw_layout_profile": RAW_LAYOUT_PROFILE,
        "parameter_version": expected_weight_version,
        "gist_parameter_dtype": "float32",
        "gist_compute_dtype": dtype,
        "base_query_enforced": True,
    }
    for field, expected in expected_capability.items():
        if native.get(field) != expected:
            raise SGLangCheckpointBindingError(
                f"SGLang native capability differs for {field}"
            )

    binding = _json_object(
        native.get("model_binding"),
        "SGLang /model_info.c2kv_native_packed.model_binding",
    )
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            raise SGLangCheckpointBindingError(
                f"SGLang native model_binding differs for {field}"
            )
    kv_bytes_per_token = native.get("kv_bytes_per_token")
    if type(kv_bytes_per_token) is not int or kv_bytes_per_token <= 0:
        raise SGLangCheckpointBindingError(
            "SGLang native kv_bytes_per_token must be a positive integer"
        )
    return {
        "backend": "sglang_c2kv_native_packed",
        "model_info_sha256": _canonical_sha256(info),
        "model_path": str(checkpoint_path),
        "weight_version": expected_weight_version,
        "model_binding": binding,
        "capability_schema": native["schema"],
        "endpoint": native["endpoint"],
        "packing_version": native["packing_version"],
        "raw_layout_profile": native["raw_layout_profile"],
        "base_query_enforced": native["base_query_enforced"],
        "gist_parameter_dtype": native["gist_parameter_dtype"],
        "gist_compute_dtype": native["gist_compute_dtype"],
        "kv_bytes_per_token": kv_bytes_per_token,
    }


def _validate_server_info(
    server_info: Mapping[str, Any],
    *,
    device: str,
    checkpoint_context_length: int,
) -> dict[str, Any]:
    info = _json_object(server_info, "SGLang /server_info")
    actual_device = info.get("device")
    if actual_device != device:
        raise SGLangCheckpointBindingError(
            "SGLang /server_info device differs from the requested device"
        )
    engine_context_length = info.get("context_length")
    if type(engine_context_length) is not int or engine_context_length <= 0:
        raise SGLangCheckpointBindingError(
            "SGLang /server_info context_length must be a positive integer"
        )
    return {
        "server_info_sha256": _canonical_sha256(info),
        "device": actual_device,
        "device_verification": "server_info.device",
        "execution": {
            field: info.get(field)
            for field in (
                "attention_backend",
                "disable_cuda_graph",
                "disable_piecewise_cuda_graph",
                "disable_overlap_schedule",
                "disable_radix_cache",
                "page_size",
                "max_running_requests",
            )
        },
        "checkpoint_context_length": checkpoint_context_length,
        "engine_context_length": engine_context_length,
        "effective_model_context": min(
            checkpoint_context_length, engine_context_length
        ),
    }


def _transport_source_identity() -> dict[str, Any]:
    manifest_path = (
        Path(__file__).resolve().parent
        / "vendor"
        / "sglang_transport"
        / "SOURCE_MANIFEST.json"
    )
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRANSPORT_MANIFEST_SCHEMA:
        raise RuntimeError("Vendored SGLang transport manifest schema mismatch")
    repository_root = Path(__file__).resolve().parents[2]
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Vendored SGLang transport manifest has no files")
    verified = []
    for record in files:
        if not isinstance(record, Mapping):
            raise RuntimeError("Vendored SGLang transport record must be an object")
        path = repository_root / str(record.get("vendored_path", ""))
        expected = record.get("vendored_sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(
                f"Vendored SGLang transport hash mismatch: {path}"
            )
        verified.append(
            {
                "upstream_path": record.get("upstream_path"),
                "upstream_sha256": record.get("upstream_sha256"),
                "vendored_path": record.get("vendored_path"),
                "vendored_sha256": expected,
            }
        )
    return {"schema": manifest["schema"], "files": verified}


class SGLangNextCompressionGenerator:
    """Request-isolated facade over the audited native-packed adapter."""

    decode_strategy = "incremental"

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        upstream: str,
        model_context: int,
        max_new_tokens: int,
        eos_token_ids: int | Sequence[int],
        eos_source: str,
        model_binding: Mapping[str, Any],
        kv_bytes_per_token: int,
        journal_path: str | Path | None,
        opener_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.upstream = _transport._base_url(upstream)
        self.model_context = _positive_int(model_context, "model_context")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.eos_token_ids = _transport._eos_ids(eos_token_ids)
        if not isinstance(eos_source, str) or not eos_source:
            raise ValueError("eos_source must be a nonempty string")
        self.eos_source = eos_source
        self.model_binding = _json_object(model_binding, "model_binding")
        self.kv_bytes_per_token = _positive_int(
            kv_bytes_per_token, "kv_bytes_per_token"
        )
        self._journal = (
            _transport._HTTPJournal(journal_path)
            if journal_path is not None
            else None
        )
        self._opener_factory = opener_factory
        self._request_state: contextvars.ContextVar[_RequestState | None] = (
            contextvars.ContextVar(
                f"next_compression_sglang_request_{id(self)}", default=None
            )
        )

    @contextmanager
    def decision_scope(self, *, session_id: str | None = None) -> Iterator[None]:
        if self._request_state.get() is not None:
            raise RuntimeError("decision_scope is not reentrant")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be None or a nonempty string")
        state = _RequestState(session_id=session_id)
        token = self._request_state.set(state)
        try:
            yield
        finally:
            try:
                if state.transport is not None:
                    state.transport.close_session()
            finally:
                self._request_state.reset(token)

    def generate(
        self,
        memory: PackedMemory,
        *,
        ratio: int,
        max_new_tokens: int,
        stop_strings: Sequence[str] = (),
        trace_context: Mapping[str, Any] | None = None,
    ) -> _transport.SGLangEventNativeGenerationResult:
        if not isinstance(stop_strings, Sequence) or isinstance(
            stop_strings, (str, bytes, bytearray)
        ):
            raise TypeError("stop_strings must be a sequence of strings")
        stops = tuple(stop_strings)
        if any(not isinstance(stop, str) or not stop for stop in stops):
            raise ValueError("stop_strings must contain nonempty strings")
        state = self._request_state.get()
        implicit = state is None
        if state is None:
            state = _RequestState(session_id=None)
        if state.used:
            raise RuntimeError("one decision_scope may submit only one generation")
        state.used = True

        extraction_budget = len(memory.chunks)
        sampling_params: dict[str, Any] = {
            "temperature": 0.0,
            "seed": 0,
            "no_stop_trim": True,
        }
        if stops:
            sampling_params["stop"] = list(stops)
        opener = self._opener_factory() if self._opener_factory is not None else None
        adapter = _transport.SGLangEventNativeGenerator(
            self.upstream,
            expected_model_path=self.checkpoint_path,
            model_context=self.model_context,
            max_new_tokens=self.max_new_tokens,
            max_generation_calls=1,
            max_extraction_calls=extraction_budget,
            timeout_seconds=GENERATION_TIMEOUT_SECONDS,
            eos_token_ids=self.eos_token_ids,
            eos_source=self.eos_source,
            encoding_scope="current",
            journal_path=None,
            sampling_params=sampling_params,
            opener=opener,
        )
        adapter._model_binding = copy.deepcopy(self.model_binding)
        adapter._kv_bytes_per_token = self.kv_bytes_per_token
        adapter._http_journal = self._journal
        state.transport = adapter
        try:
            with adapter.decision_scope(session_id=state.session_id):
                return adapter.generate(
                    memory,
                    ratio=ratio,
                    max_new_tokens=max_new_tokens,
                    trace_context=trace_context,
                )
        finally:
            if implicit:
                adapter.close_session()

    def close_session(self) -> None:
        state = self._request_state.get()
        if state is not None and state.transport is not None:
            state.transport.close_session()


def load_sglang_checkpoint(
    checkpoint: str | Path,
    *,
    upstream: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    max_requests: int,
    journal_path: str | Path | None = None,
) -> tuple[SGLangNextCompressionGenerator, Any, dict[str, Any]]:
    """Bind local config/tokenizer files to one already-loaded SGLang model."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_path}"
        )
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint is missing config.json: {checkpoint_path}"
        )
    config = _read_json(config_path)
    metadata = _validate_next_config(config)
    config_sha256 = sha256_file(config_path)
    requested_dtype = _normalize_dtype(dtype)
    requested_device = _validate_device(device)
    _positive_int(max_new_tokens, "max_new_tokens")
    _positive_int(max_requests, "max_requests")
    base_url = _transport._base_url(upstream)
    model_info = _read_model_info(base_url)
    serving_engine = _validate_model_info(
        model_info,
        checkpoint_path=checkpoint_path,
        config=config,
        config_sha256=config_sha256,
        dtype=requested_dtype,
    )
    checkpoint_context_length = config.get("max_position_embeddings")
    if type(checkpoint_context_length) is not int or checkpoint_context_length <= 0:
        raise ValueError(
            "Checkpoint config must declare positive max_position_embeddings"
        )
    serving_engine.update(
        _validate_server_info(
            _read_server_info(base_url),
            device=requested_device,
            checkpoint_context_length=checkpoint_context_length,
        )
    )
    serving_engine["upstream"] = base_url
    serving_engine["transport_source"] = _transport_source_identity()
    serving_engine["request_isolation"] = "one-native-adapter-per-request"
    serving_engine["extraction_budget"] = "selected-memory-chunk-count"
    serving_engine["generation_timeout_seconds"] = GENERATION_TIMEOUT_SECONDS
    serving_engine["request_budget_owner"] = "LiveNextCompressionService"
    serving_engine["frontend_max_requests"] = max_requests
    serving_engine["requested_device"] = requested_device

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, local_files_only=True
    )
    eos_token_ids, eos_source = _checkpoint_eos_token_ids(
        checkpoint_path, config, tokenizer
    )
    serving_engine["eos_token_ids"] = list(eos_token_ids)
    serving_engine["eos_source"] = eos_source
    model_context = serving_engine["effective_model_context"]
    generator = SGLangNextCompressionGenerator(
        checkpoint_path=checkpoint_path,
        upstream=base_url,
        model_context=model_context,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        eos_source=eos_source,
        model_binding=serving_engine["model_binding"],
        kv_bytes_per_token=serving_engine["kv_bytes_per_token"],
        journal_path=journal_path,
    )
    profile = {
        **metadata,
        "checkpoint": str(checkpoint_path),
        "config_sha256": config_sha256,
        "dtype": requested_dtype,
        "device": requested_device,
        "gist_parameter_dtype": serving_engine["gist_parameter_dtype"],
        "serving_engine": serving_engine,
    }
    return generator, tokenizer, profile


__all__ = [
    "SGLangCheckpointBindingError",
    "SGLangNextCompressionGenerator",
    "load_sglang_checkpoint",
]
