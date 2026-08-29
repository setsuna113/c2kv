"""OpenAI-compatible C2KV server on the HF/transformers path.

Motivation: the SGLang fork's c2kv serving path does not run on this NPU
software stack (torch 2.8 + current CANN); this server implements the same
protocol (`/v1/chat/completions` + `/v1/c2kv/extract` + `c2kv_key_hash`)
directly on the repo's HF primitives — the exact machinery that produced the
D-experiment results (`agent/eval_agent_history_c2kv.py`,
`python/inference/expr_c2kv.py`), so arm semantics are identical to the
teacher-forced line and repair arms can later reuse the same KV operations.

Layout rules (mirroring benchmarks/proxy.py + the history harness):
  * system message: raw prefill (position 0..), cached across requests
  * message with `c2kv_key_hash`: gist KV appended at the current cache end;
    logical positions advance by the ORIGINAL token count.  A message longer
    than MAX_DOC_LENGTH is split into chunks whose gists sit at sequential
    offsets inside the message (per-chunk rotation, harness-grid semantics).
  * plain message: sequential raw prefill with continuing position_ids.
  * generation: mock zero tokens occupy the cache slots; positions continue
    from the logical length.
  * use_gist projection rule (matches harness :1038/:1564 and training
    modeling_qwen3:660): once ANY gist KV is in the cache, every later
    forward — raw prefill included — runs with the gist projections; the
    switch is global per forward call.

Historical assistant tool_calls are re-rendered into the TRAINING dialect
(``content + "\\n\\n" + "Action:\\n" + <tool_call> blocks`` with minified
JSON, train_data_multiturn._normal_agent_message), not the chat template's
native tool_calls branch.

usage.prompt_tokens counts only the current-turn prompt segment (the mock
cache tokens are excluded); cache_tokens/logical_tokens carry the rest.

Single model instance behind a lock; run one request at a time
(Flask `threaded=False`).

Usage (c2kv env on the NPU server):
  python benchmarks/hf_server.py --model-path <ckpt> --host 127.0.0.1 --port 34000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flask import Flask, jsonify, request

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "inference"))
sys.path.insert(0, str(REPO / "agent"))

from models import get_model_class  # noqa: E402
from models.gist_utils import blend_gist_key_values  # noqa: E402
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
from inference.reuse_pipeline import tokenize_for_reuse  # noqa: E402
from inference.rope_reposition import rotate_k_cache_rope  # noqa: E402

app = Flask(__name__)

GIST_IMPL = "npu_fusion_attention"  # downgraded to eager automatically off-NPU
MAX_DOC_LENGTH = 768

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def _normalize_tool_schema(value: Any, defs: Dict[str, Any] = None,
                           seen: frozenset = frozenset()) -> Any:
    """Schema repair before grammar compilation.

    Benchmark tool schemas use loose types ("type": "dict"/"any"/...,
    {"type": "list"} without items) and $ref/$defs indirection, neither of
    which xgrammar's JSON-schema converter accepts.  Repair applies to the
    grammar input only, never to the tool definition the model sees:
    $refs are inlined (cycle-guarded), then types are mapped and
    unsupported keywords stripped.
    """
    if isinstance(value, dict):
        if "$ref" in value and defs is not None:
            name = str(value["$ref"]).split("/")[-1]
            if name in defs and name not in seen:
                return _normalize_tool_schema(
                    defs[name], defs, seen | {name}
                )
            return {"type": "object"}
        out = {}
        for key, item in value.items():
            if key in ("$schema", "$id", "$defs", "$ref"):
                continue
            if key == "type":
                if isinstance(item, str):
                    item = {
                        "dict": "object",
                        "any": "object",
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "tuple": "array",
                        "None": "null",
                    }.get(item, item)
                elif isinstance(item, list):
                    mapped = []
                    for t in item:
                        t = {
                            "dict": "object", "any": "object", "int": "integer",
                            "float": "number", "str": "string", "bool": "boolean",
                            "list": "array", "tuple": "array", "None": "null",
                        }.get(t, t)
                        if t not in mapped:
                            mapped.append(t)
                    if not mapped:
                        continue
                    item = mapped
            out[key] = _normalize_tool_schema(item, defs, seen)
        return out
    if isinstance(value, list):
        return [_normalize_tool_schema(v, defs, seen) for v in value]
    return value


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return _normalize_tool_schema(schema, defs)


def cfg_vocab_size(model) -> int:
    size = getattr(model.config, "vocab_size", None)
    if size is None:
        size = len(model.get_input_embeddings().weight)
    return int(size)


def _parse_tool_calls(raw_text: str):
    """Qwen-style <tool_call> blocks -> OpenAI tool_calls objects.

    Mirrors SGLang's qwen25 tool-call parser: matched blocks become
    tool_calls, surrounding text stays as content; unmatched blocks are left
    verbatim in content so protocol violations stay visible to the scorer.
    """
    calls = []
    spans = []
    for match in TOOL_CALL_RE.finditer(raw_text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            continue
        calls.append({
            "id": f"call_{len(calls)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
        spans.append(match.span())
    if not spans:
        return raw_text, []
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(raw_text[cursor:start])
        cursor = end
    pieces.append(raw_text[cursor:])
    content = "".join(pieces).strip()
    return content, calls


@dataclass
class GistEntry:
    """One compressed message, possibly split into <=MAX_DOC_LENGTH chunks.

    Chunk positions are 0-based within each chunk (generate_gist always
    builds from 0); the i-th chunk's gists logically belong at
    ``sum(original_len of chunks < i)`` inside the message, so _append_gist
    rotates each chunk by ``logical_start + its cumulative offset`` — the
    sequential-accumulation semantics of the harness grid
    (gist_utils._concat_gist_key_values), which the previous single-rotation
    version violated (all chunks collided on one position range).
    """

    chunks: List[Dict[str, Any]]  # per-chunk: keys, values, gist_len, original_len
    gist_len: int
    original_len: int
    role: str
    ratio: int


class C2KVServer:
    def __init__(self, model_path: str, device: str,
                 tokenizer_path: Optional[str] = None):
        self.device = device
        self.tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            tokenizer_path or model_path, trust_remote_code=True,
            local_files_only=True, padding_side="right",
        )
        config_class, model_class = get_model_class(model_path, "qkv")
        self.model = model_class.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
            device_map={"": device}, dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.eval()
        self._set_impl(GIST_IMPL)
        self.entries: Dict[str, GistEntry] = {}
        self.lock = threading.Lock()
        cfg = self.model.config
        self.dynamic_ratio = getattr(cfg, "gist_type", "") == "dynamic-interleave"
        self.rope_theta = getattr(cfg, "rope_theta", 1000000.0)
        self.rope_type = getattr(getattr(cfg, "rope_scaling", None), "rope_type", None)
        # H1 constrained decoding (xgrammar structural tag, qwen_3 dialect)
        self._xgr_tokenizer_info = None
        self._xgr_compiler = None
        self._grammar_cache: Dict[str, Any] = {}
        # L1 prefix cache: system+tools prefill output, reused verbatim by
        # every request with the same system prompt/tool pool (benchmarks
        # resend the same ~1.5k-token tool schema on every turn).  Stored
        # tensors are never mutated downstream — layer.keys rebinding via
        # torch.cat allocates fresh tensors — so sharing storages is safe.
        self._system_cache_store: Dict[str, Tuple[Any, int, float]] = {}

    def _prefill_system_cached(self, system_ids: List[int], tools):
        key = hashlib.sha256(
            json.dumps([system_ids, tools or []], ensure_ascii=False)
            .encode()
        ).hexdigest()
        hit = self._system_cache_store.get(key)
        if hit is not None:
            return self._clone_layers(hit[0]), hit[1], 0.0
        cache, added = self._prefill_append(None, system_ids, 0, use_gist=False)
        if len(self._system_cache_store) < 32:
            self._system_cache_store[key] = (cache, added, 0.0)
            return self._clone_layers(cache), added, 0.0
        return cache, added, 0.0

    @staticmethod
    def _clone_layers(cache):
        """Fresh layer objects sharing the same tensors.

        Callers rebind layer.keys/values via torch.cat (which allocates new
        tensors), so sharing storages is safe, but the layer objects
        themselves must not be shared or a cached state would be corrupted
        by the first request that extends it.
        """
        from transformers.cache_utils import DynamicCache

        clone = DynamicCache()
        for layer in cache.layers:
            clone.update(layer.keys, layer.values, layer_idx=len(clone.layers))
        return clone

    def _compiled_tool_grammar(self, tools: List[Dict[str, Any]]):
        """xgrammar structural tag for the tool pool (cached by tool set).

        get_model_structural_tag("qwen_3", ...) constrains
        <tool_call>{"name": ..., "arguments": <schema>}</tool_call> while
        leaving surrounding text free — the XGrammar-2 recipe (H1).
        """
        import xgrammar as xgr

        if self._xgr_compiler is None:
            self._xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=cfg_vocab_size(self.model)
            )
            self._xgr_compiler = xgr.GrammarCompiler(self._xgr_tokenizer_info)
        key = hashlib.sha256(
            json.dumps(tools, sort_keys=True).encode()
        ).hexdigest()
        if key not in self._grammar_cache:
            normalized = [
                {
                    "type": "function",
                    "function": _inline_refs(t.get("function") or t),
                }
                for t in tools
            ]
            tag = xgr.get_model_structural_tag(
                "qwen_3", tools=normalized, reasoning=False
            )
            self._grammar_cache[key] = self._xgr_compiler.compile_structural_tag(tag)
        return self._grammar_cache[key]

    def _set_impl(self, impl: str):
        self.model.config._attn_implementation = impl
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config._attn_implementation = impl

    # ---------------- extract ----------------
    def extract(self, text: str, role: str, ratio: int) -> Dict[str, Any]:
        tokenized = tokenize_for_reuse(
            self.tokenizer, [text], keep_bos=False, role=role
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        # repo's HF generate_gist expects (batch, seqlen)
        L = input_ids.shape[-1]
        chunks: List[Dict[str, Any]] = []
        for start in range(0, L, MAX_DOC_LENGTH):
            chunks.append(self._extract_grid(
                input_ids[..., start:start + MAX_DOC_LENGTH],
                attention_mask[..., start:start + MAX_DOC_LENGTH], ratio,
            ))
        gist_len = sum(c["gist_len"] for c in chunks)
        original_len = sum(c["original_len"] for c in chunks)
        key_hash = hashlib.sha256(
            f"{role}\x00{ratio}\x00{text}".encode("utf-8")
        ).hexdigest()
        self.entries[key_hash] = GistEntry(
            chunks=chunks, gist_len=gist_len,
            original_len=original_len, role=role, ratio=ratio,
        )
        return {
            "key_hash": key_hash, "gist_len": gist_len,
            "original_seq_len": original_len, "success": True, "error": None,
        }

    def _extract_grid(self, input_ids, attention_mask, ratio: int) -> Dict[str, Any]:
        original = self.model
        with torch.inference_mode():
            outputs, gist_mask, pos_ids = original.model.generate_gist(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **({"ratio": ratio} if self.dynamic_ratio else {}),
            )
            gist_len = int(gist_mask.shape[-1])
            pos_ids = pos_ids[:, -gist_len:]
            gist_cache, _ = blend_gist_key_values(
                original.config, [outputs.past_key_values], [gist_mask],
                [pos_ids], original.model.rotary_emb, prefix_length=0,
            )
        keys = [layer.keys for layer in gist_cache.layers]
        values = [layer.values for layer in gist_cache.layers]
        original_len = int(input_ids.shape[-1])
        return {"keys": keys, "values": values,
                "gist_len": gist_len, "original_len": original_len}

    # ---------------- chat ----------------
    def chat(self, messages: List[Dict[str, Any]], max_new_tokens: int,
             temperature: float, tools: Optional[List[Dict[str, Any]]] = None,
             constrain: bool = False,
             repair: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        # repair arm (docs/hybrid_spec.md "Repair interaction"): policy picks
        # the compressed history chunk whose raw KV gets appended at its
        # original logical offset.  "first" == D harness corr@first.
        repair_policy: Optional[str] = None
        repair_target = 0
        if repair:
            policy = str(repair.get("policy") or "first")
            if policy == "first":
                repair_target = 0
            elif policy.startswith("offset:"):
                try:
                    repair_target = int(policy.split(":", 1)[1])
                except ValueError:
                    return {"error": f"c2kv_repair: bad offset policy {policy!r}"}
            else:
                return {"error": f"c2kv_repair: unknown policy {policy!r}"}
            repair_policy = policy
        system_messages = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]
        if not body:
            return {"error": "no non-system messages"}
        system_text = "\n".join(m.get("content") or "" for m in system_messages)

        with self.lock, torch.inference_mode():
            # normalize OpenAI-style assistant tool_calls history into the
            # TRAINING dialect (train_data_multiturn._normal_agent_message):
            #   content + "\n\n" + "Action:\n" + "\n".join(<tool_call> blocks)
            # with minified JSON — the surface the model was trained on
            for message in body:
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    blocks = []
                    for call in message["tool_calls"]:
                        function = call.get("function") or {}
                        try:
                            arguments = json.loads(function.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            arguments = function.get("arguments") or {}
                        blocks.append(
                            "<tool_call>\n"
                            + json.dumps(
                                {"name": function.get("name"), "arguments": arguments},
                                ensure_ascii=False, separators=(",", ":"),
                            )
                            + "\n</tool_call>"
                        )
                    action = "Action:\n" + "\n".join(blocks)
                    content = message.get("content") or ""
                    message["content"] = (
                        content + "\n\n" + action if content else action
                    )
                    message.pop("tool_calls", None)
                content = message.get("content")
                if not isinstance(content, str):
                    message["content"] = (
                        json.dumps(content, ensure_ascii=False) if content else ""
                    )
            # 1. system prefill (tools render inside the system block, as in
            # the harness's _prefill_system); cached across requests
            cache = None
            logical = 0
            repair_system_ids: Optional[List[int]] = None
            if system_text or tools:
                system_ids = _chat_template_ids(
                    self.tokenizer,
                    [{"role": "system", "content": system_text}],
                    tools=tools, keep_bos=True,
                )
                repair_system_ids = system_ids
                cache, added, _ = self._prefill_system_cached(system_ids, tools)
                logical += added

            # 2. body[:-1]: gist refs and raw messages interleaved.
            # use_gist rule (harness :1038/:1564, training modeling_qwen3:660):
            # once ANY gist KV is in the cache, every later forward — raw
            # prefill included — runs with the gist projections.  The flag is
            # global per forward (modeling_qwen3:242-250), so a raw tail under
            # base projections after gist content would be a regime the
            # training pipeline never produces.
            cache_has_gist = False
            reextracted = 0
            # compressed history in conversation order, for the repair arm
            compressed: List[Tuple[Dict[str, Any], GistEntry]] = []
            for message in body[:-1]:
                key_hash = message.get("c2kv_key_hash")
                if key_hash:
                    entry = self.entries.get(key_hash)
                    if entry is None and message.get("content"):
                        # The server may have restarted since the proxy
                        # extracted this block (gist pool is in-memory);
                        # re-extract from the verbatim content instead of
                        # failing the request.
                        result = self.extract(
                            str(message.get("content")),
                            str(message.get("role") or "user"),
                            int(message.get("c2kv_ratio") or 8),
                        )
                        entry = self.entries.get(result["key_hash"]) or self.entries.get(key_hash)
                        reextracted += 1
                    if entry is None:
                        return {"error": f"C2KV cache miss: {key_hash}"}
                    cache = self._append_gist(cache, entry, logical)
                    logical += entry.original_len
                    cache_has_gist = True
                    compressed.append((message, entry))
                    continue
                ids = _chat_template_ids(self.tokenizer, [message])
                cache, added = self._prefill_append(
                    cache, ids, logical, use_gist=cache_has_gist
                )
                logical += added

            # 2.5 repair arm: append the raw KV of the policy-selected
            # compressed history block onto the assembled prefix (its gists
            # already advanced `logical`, so the span re-occupies its own
            # logical range at the physical cache end; decode positions are
            # untouched).  Short conversations with nothing compressed yet
            # are a legitimate no-op.
            repair_block_tokens = 0
            repair_prefill_sec = 0.0
            repair_doc_index: Optional[int] = None
            if repair_policy is not None and compressed:
                try:
                    cache, repair_block_tokens, repair_prefill_sec = (
                        self._append_raw_block(
                            cache, repair_system_ids, tools, compressed, repair_target
                        )
                    )
                    repair_doc_index = repair_target
                except ValueError as error:
                    return {"error": f"c2kv_repair: {error}"}

            # 3. final message + generation (harness _generate_with_prefix
            # shape: mock zeros occupy the cache slots, positions continue
            # from the logical length)
            last = body[-1]
            if last.get("c2kv_key_hash"):
                key_hash = last["c2kv_key_hash"]
                entry = self.entries.get(key_hash)
                if entry is None and last.get("content"):
                    self.extract(
                        str(last.get("content")),
                        str(last.get("role") or "user"),
                        int(last.get("c2kv_ratio") or 8),
                    )
                    entry = self.entries.get(key_hash)
                if entry is None:
                    return {"error": f"C2KV cache miss: {key_hash}"}
                cache = self._append_gist(cache, entry, logical)
                logical += entry.original_len
                cache_has_gist = True
                prompt_ids = _chat_template_ids(
                    self.tokenizer, [{"role": "user", "content": ""}],
                    add_generation_prompt=True,
                )
            else:
                prompt_ids = _chat_template_ids(
                    self.tokenizer, [last], add_generation_prompt=True
                )
            prompt_input_ids = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.device
            )
            cache_len = cache.get_seq_length() if cache is not None else 0
            mock = prompt_input_ids.new_zeros((1, cache_len))
            input_ids = torch.cat([mock, prompt_input_ids], dim=1)
            position_ids = torch.arange(
                logical, logical + prompt_input_ids.shape[1],
                dtype=torch.long, device=self.device,
            ).unsqueeze(0)
            self._set_impl(GIST_IMPL)
            # prefix-level debug snapshot for the cross-check harness
            # (benchmarks/selfcheck_repair_vs_dharness.py); harmless to
            # normal serving — plain attribute, read after chat() returns.
            self.last_debug = {
                "logical": logical,
                "cache_len": cache_len,
                "cache_has_gist": cache_has_gist,
                "repair_policy": repair_policy,
                "repair_block_tokens": repair_block_tokens,
                "repair_doc_index": repair_doc_index,
                "cache": cache,
            }
            logits_processor = None
            if constrain and tools:
                from xgrammar.contrib import hf as xgr_hf

                logits_processor = [
                    xgr_hf.LogitsProcessor(self._compiled_tool_grammar(tools))
                ]
            t_gen = time.perf_counter()
            gen_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=position_ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                pad_token_id=self.tokenizer.eos_token_id,
                use_gist=cache_has_gist,
            )
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
            outputs = self.model.generate(**gen_kwargs)
            generate_sec = time.perf_counter() - t_gen
            new_tokens = outputs[0][input_ids.shape[1]:]
            if hasattr(self, "last_debug"):
                self.last_debug["new_token_ids"] = [int(t) for t in new_tokens]
            raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            content, tool_calls = _parse_tool_calls(raw_text)
            text = content if content else self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
        return {
            "content": text,
            "tool_calls": tool_calls,
            "constrained": bool(constrain and tools),
            "reextracted": reextracted,
            "generated_tokens": int(new_tokens.shape[0]),
            "prompt_tokens": len(prompt_ids),
            "cache_tokens": cache_len,
            "logical_tokens": logical,
            "repair_policy": repair_policy,
            "repair_block_tokens": repair_block_tokens,
            "repair_doc_index": repair_doc_index,
            "repair_prefill_sec": round(repair_prefill_sec, 4),
            "generate_sec": round(generate_sec, 4),
            "wall_sec": round(time.perf_counter() - t0, 4),
        }


    def _prefill_append(self, cache, token_ids: List[int], logical_start: int,
                        use_gist: bool) -> Tuple[Any, int]:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        past_length = 0 if cache is None else cache.get_seq_length()
        attention_mask = torch.ones(
            1, past_length + input_ids.shape[1], device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.arange(
            logical_start, logical_start + input_ids.shape[1], device=self.device
        ).unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            use_gist=use_gist,
            logits_to_keep=1,
        )
        return outputs.past_key_values, input_ids.shape[1]

    def _append_gist(self, cache, entry: GistEntry, logical_start: int):
        """Append a message's gist chunks at sequential logical positions.

        Chunk i is rotated by ``logical_start + sum(original_len of chunks
        < i)`` — mirroring the harness grid's per-doc accumulation
        (gist_utils._concat_gist_key_values) instead of one shared offset,
        which used to collapse every chunk onto the same position range.
        """
        keys: List[List[torch.Tensor]] = []
        values: List[List[torch.Tensor]] = []
        offset = 0
        for chunk in entry.chunks:
            chunk_keys = []
            chunk_values = []
            for i, layer in enumerate(chunk["keys"]):
                # rotate_k_cache_rope expects (heads, seq, dim); stored keys
                # are (1, heads, seq, dim)
                k = rotate_k_cache_rope(
                    layer[0], logical_start + offset,
                    self.rope_theta, self.rope_type,
                ).unsqueeze(0)
                chunk_keys.append(k)
                chunk_values.append(chunk["values"][i])
            keys.append(chunk_keys)
            values.append(chunk_values)
            offset += int(chunk["original_len"])
        flat_keys = [torch.cat([ks[i] for ks in keys], dim=-2)
                     for i in range(len(keys[0]))]
        flat_values = [torch.cat([vs[i] for vs in values], dim=-2)
                       for i in range(len(values[0]))]
        if cache is None:
            from transformers.cache_utils import DynamicCache

            merged = DynamicCache()
            for k, v in zip(flat_keys, flat_values):
                merged.update(k, v, layer_idx=len(merged.layers))
            return merged
        for i, layer in enumerate(cache.layers):
            layer.keys = torch.cat([layer.keys, flat_keys[i]], dim=-2)
            layer.values = torch.cat([layer.values, flat_values[i]], dim=-2)
        return cache

    def _append_raw_block(self, cache, system_ids, tools,
                          compressed: List[Tuple[Dict[str, Any], "GistEntry"]],
                          target_chunk: int):
        """corr append — raw KV of a compressed history block (docs/hybrid_spec.md).

        Mirrors the D harness d_corr pass (eval_agent_history_c2kv.py
        _build_d_intervene_prefix): a scratch cache prefills the system prompt
        and then the raw text of every compressed message up to and including
        the target chunk, at the ORIGINAL logical offsets (use_gist=False —
        the scratch cache holds no gist).  The target chunk's span — already
        carrying its original RoPE phases — is concatenated onto the request
        cache end unrotated.  Chunk indices enumerate entry.chunks across
        ``compressed`` in conversation order.

        Returns (cache, block_tokens, prefill_sec); raises ValueError when
        target_chunk is out of range.
        """
        t0 = time.perf_counter()
        n_chunks = sum(len(entry.chunks) for _, entry in compressed)
        if not 0 <= target_chunk < n_chunks:
            raise ValueError(
                f"offset {target_chunk} out of range (0..{n_chunks - 1})"
            )
        with torch.inference_mode():
            if system_ids is not None:
                raw_cache, system_length, _ = self._prefill_system_cached(
                    system_ids, tools
                )
            else:
                raw_cache, system_length = None, 0
            logical = system_length
            block_tokens = 0
            remaining = target_chunk + 1  # chunks left to prefill
            for message, entry in compressed:
                if remaining <= 0:
                    break
                tokenized = tokenize_for_reuse(
                    self.tokenizer, [str(message.get("content") or "")],
                    keep_bos=False, role=str(message.get("role") or "user"),
                )
                ids = tokenized["input_ids"][0].tolist()
                pos = 0
                for chunk in entry.chunks:
                    if remaining <= 0:
                        break
                    chunk_len = int(chunk["original_len"])
                    chunk_ids = ids[pos: pos + chunk_len]
                    raw_cache, _ = self._prefill_append(
                        raw_cache, chunk_ids, logical, use_gist=False
                    )
                    logical += len(chunk_ids)
                    pos += chunk_len
                    block_tokens = len(chunk_ids)
                    remaining -= 1
            # the target chunk is the tail of the scratch cache; slice it out
            span_keys = [layer.keys[..., -block_tokens:, :].clone()
                         for layer in raw_cache.layers]
            span_values = [layer.values[..., -block_tokens:, :].clone()
                           for layer in raw_cache.layers]
            del raw_cache
            for i, layer in enumerate(cache.layers):
                layer.keys = torch.cat([layer.keys, span_keys[i]], dim=-2)
                layer.values = torch.cat([layer.values, span_values[i]], dim=-2)
            del span_keys, span_values
        return cache, block_tokens, time.perf_counter() - t0


SERVER: Optional[C2KVServer] = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/c2kv/extract", methods=["POST"])
def extract():
    data = request.get_json(force=True)
    try:
        result = SERVER.extract(
            str(data.get("text") or ""), str(data.get("role") or "user"),
            int(data.get("compression_ratio") or 8),
        )
        return jsonify(result)
    except Exception as error:  # noqa: BLE001
        import traceback

        return jsonify({"success": False,
                        "error": f"{error}\n{traceback.format_exc()}",
                        "key_hash": "", "gist_len": 0, "original_seq_len": 0})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    try:
        result = SERVER.chat(
            messages=data.get("messages") or [],
            # upstream alignment: 4096 max tokens when the client omits it
            # (tau2/ToolSandbox send no max_tokens), temperature 0 default
            max_new_tokens=int(data.get("max_completion_tokens")
                               or data.get("max_tokens") or 4096),
            temperature=float(data.get("temperature") or 0.0),
            tools=data.get("tools"),
            constrain=bool(data.get("constrain_tools")),
            repair=data.get("c2kv_repair"),
        )
    except Exception as error:  # noqa: BLE001
        import traceback

        return jsonify({"object": "error",
                        "message": f"{error}\n{traceback.format_exc()}"}), 500
    if "error" in result:
        return jsonify({"object": "error", "message": result["error"]}), 400
    message: Dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
    finish_reason = "stop"
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]
        finish_reason = "tool_calls"
    return jsonify({
        "id": "c2kv-hf",
        "object": "chat.completion",
        "model": data.get("model") or "c2kv-agent",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["generated_tokens"],
            "total_tokens": result["prompt_tokens"] + result["generated_tokens"],
        },
        "c2kv": {k: v for k, v in result.items()
                 if k not in ("content", "tool_calls")},
    })


def main(argv=None):
    global SERVER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default=None,
                        help="defaults to --model-path; pass the BASE model "
                        "dir when the checkpoint's tokenizer_config carries "
                        "extra_special_tokens as a list")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=34000)
    args = parser.parse_args(argv)
    SERVER = C2KVServer(args.model_path, "npu", tokenizer_path=args.tokenizer)
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
