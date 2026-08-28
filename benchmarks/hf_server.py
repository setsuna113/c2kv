"""OpenAI-compatible C2KV server on the HF/transformers path.

Motivation: the SGLang fork's c2kv serving path does not run on this NPU
software stack (torch 2.8 + current CANN); this server implements the same
protocol (`/v1/chat/completions` + `/v1/c2kv/extract` + `c2kv_key_hash`)
directly on the repo's HF primitives — the exact machinery that produced the
D-experiment results (`agent/eval_agent_history_c2kv.py`,
`python/inference/expr_c2kv.py`), so arm semantics are identical to the
teacher-forced line and repair arms can later reuse the same KV operations.

Layout rules (mirroring benchmarks/proxy.py + the history harness):
  * system message: raw prefill (position 0..)
  * message with `c2kv_key_hash`: gist KV layer-catted at the current cache
    end; logical positions advance by the ORIGINAL token count (RoPE for the
    gist keys was already applied at extract time with prefix_length=0 and is
    re-rotated here by the logical start — see rotate_k_cache_rope).
  * plain message: sequential raw prefill with continuing position_ids.
  * generation: mock zero tokens occupy the cache slots; positions continue
    from the logical length; `use_gist` is True iff the turn directly follows
    gist blocks (harness c2kv-mode behavior), else False.

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


def _normalize_tool_schema(value: Any) -> Any:
    """BFCL-style schema repair before grammar compilation.

    gorilla_file_system and friends declare "type": "dict" / "any" and bare
    {"type": "list"} without item schemas — not valid JSON Schema and
    rejected by xgrammar's converter.  Map to the closest valid form and
    strip unsupported keywords; repair is applied to the grammar input only,
    never to the tool definition the model sees.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in ("$schema", "$id", "$defs"):
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
            out[key] = _normalize_tool_schema(item)
        return out
    if isinstance(value, list):
        return [_normalize_tool_schema(v) for v in value]
    return value


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
    keys: List[torch.Tensor]   # per-layer (1, heads, gist_len, head_dim)
    values: List[torch.Tensor]
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
                    "function": _normalize_tool_schema(t.get("function") or t),
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
        if L > MAX_DOC_LENGTH:  # chunk into <=768-token docs, cat gists after
            chunks = []
            for start in range(0, L, MAX_DOC_LENGTH):
                chunks.append(self._extract_grid(
                    input_ids[..., start:start + MAX_DOC_LENGTH],
                    attention_mask[..., start:start + MAX_DOC_LENGTH], ratio,
                ))
            keys = [torch.cat([c["keys"][i] for c in chunks], dim=-2)
                    for i in range(len(chunks[0]["keys"]))]
            values = [torch.cat([c["values"][i] for c in chunks], dim=-2)
                      for i in range(len(chunks[0]["values"]))]
            gist_len = sum(c["gist_len"] for c in chunks)
            original_len = sum(c["original_len"] for c in chunks)
        else:
            out = self._extract_grid(input_ids, attention_mask, ratio)
            keys, values, gist_len, original_len = (
                out["keys"], out["values"], out["gist_len"], out["original_len"])
        key_hash = hashlib.sha256(
            f"{role}\x00{ratio}\x00{text}".encode("utf-8")
        ).hexdigest()
        self.entries[key_hash] = GistEntry(
            keys=keys, values=values, gist_len=gist_len,
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
             constrain: bool = False):
        t0 = time.perf_counter()
        system_messages = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]
        if not body:
            return {"error": "no non-system messages"}
        system_text = "\n".join(m.get("content") or "" for m in system_messages)

        with self.lock, torch.inference_mode():
            # normalize OpenAI-style assistant tool_calls history into the
            # Qwen <tool_call> text form the chat template understands
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
                                ensure_ascii=False,
                            )
                            + "\n</tool_call>"
                        )
                    message["content"] = (message.get("content") or "") + "\n".join(blocks)
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
            if system_text or tools:
                system_ids = _chat_template_ids(
                    self.tokenizer,
                    [{"role": "system", "content": system_text}],
                    tools=tools, keep_bos=True,
                )
                cache, added, _ = self._prefill_system_cached(system_ids, tools)
                logical += added

            # 2. body[:-1]: gist refs and raw messages interleaved
            follows_gist = False
            reextracted = 0
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
                    follows_gist = True
                    continue
                ids = _chat_template_ids(self.tokenizer, [message])
                cache, added = self._prefill_append(cache, ids, logical, use_gist=False)
                logical += added
                follows_gist = False

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
                prompt_ids = _chat_template_ids(
                    self.tokenizer, [{"role": "user", "content": ""}],
                    add_generation_prompt=True,
                )
                follows_gist = True
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
                use_gist=follows_gist,
            )
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
            outputs = self.model.generate(**gen_kwargs)
            generate_sec = time.perf_counter() - t_gen
            new_tokens = outputs[0][input_ids.shape[1]:]
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
        keys = []
        values = []
        for i, layer in enumerate(entry.keys):
            # rotate_k_cache_rope expects (heads, seq, dim); stored keys are
            # (1, heads, seq, dim)
            k = rotate_k_cache_rope(
                layer[0], logical_start, self.rope_theta, self.rope_type
            ).unsqueeze(0)
            keys.append(k)
            values.append(entry.values[i])
        if cache is None:
            from transformers.cache_utils import DynamicCache

            merged = DynamicCache()
            for k, v in zip(keys, values):
                merged.update(k, v, layer_idx=len(merged.layers))
            return merged
        for i, layer in enumerate(cache.layers):
            layer.keys = torch.cat([layer.keys, keys[i]], dim=-2)
            layer.values = torch.cat([layer.values, values[i]], dim=-2)
        return cache


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
            max_new_tokens=int(data.get("max_completion_tokens")
                               or data.get("max_tokens") or 256),
            temperature=float(data.get("temperature") or 0.0),
            tools=data.get("tools"),
            constrain=bool(data.get("constrain_tools")),
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
