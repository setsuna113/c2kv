import py_compile

BASE = "/tmp/zh_exp/python/sglang/srt"

def patch(path, subs):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in subs:
        n = src.count(old)
        assert n == cnt, f"{path}: anchor count {n} != {cnt} for: {old[:80]!r}"
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched", path)

EXP_BLOCK_QWEN3 = '''

# ==== EXP instrumentation (env-gated; /tmp copy only) ====
_EXP_HOOKS = os.environ.get("EXP_HOOKS") == "1"
_EXP_OUT = os.environ.get("EXP_OUT_DIR", "/tmp/zh_exp/out")
_EXP_PHASE_REC = {"seq": 0, "layers": [], "meta": {}}
_EXP_ARM = {"armed": False, "rid": None, "n": 0}

def _exp_reset_phase():
    _EXP_PHASE_REC["seq"] = 0
    _EXP_PHASE_REC["layers"] = []
    _EXP_PHASE_REC["meta"] = {}

def _exp_save_phase(extra_meta=None):
    import os as _os, torch as _torch, glob as _glob
    _os.makedirs(_EXP_OUT, exist_ok=True)
    n = len(_glob.glob(_os.path.join(_EXP_OUT, "phase_*.pt")))
    meta = dict(_EXP_PHASE_REC["meta"])
    if extra_meta:
        meta.update(extra_meta)
    _torch.save({"meta": meta, "layers": _EXP_PHASE_REC["layers"]},
                _os.path.join(_EXP_OUT, f"phase_{n:04d}.pt"))
    return n

def _exp_prep_log(name, positions, gist_state="n/a"):
    if not _EXP_HOOKS:
        return
    import os as _os, json as _json, time as _time
    _os.makedirs(_EXP_OUT, exist_ok=True)
    with open(_os.path.join(_EXP_OUT, "prep_log.jsonl"), "a") as f:
        f.write(_json.dumps({"ts": _time.time(), "prep": name,
                             "ntok": int(positions.numel()) if positions is not None else -1,
                             "gist": gist_state}) + "\\n")

def _exp_maybe_snapshot(positions, fb):
    if not _EXP_HOOKS or not _EXP_ARM.get("armed") or fb is None:
        return
    try:
        import os as _os, torch as _torch
        mode = str(getattr(fb, "forward_mode", ""))
        if "EXTEND" not in mode.upper():
            return
        _os.makedirs(_EXP_OUT, exist_ok=True)
        _ii = getattr(fb, "input_ids", None)
        _torch.save({
            "rid": _EXP_ARM.get("rid"),
            "positions": positions.detach().to(_torch.long).cpu(),
            "input_ids": _ii.detach().cpu() if _ii is not None else None,
            "extend_seq_lens_cpu": list(getattr(fb, "extend_seq_lens_cpu", []) or []),
            "extend_prefix_lens_cpu": list(getattr(fb, "extend_prefix_lens_cpu", []) or []),
        }, _os.path.join(_EXP_OUT, f"querypos_{_EXP_ARM['n']:04d}.pt"))
        _EXP_ARM["armed"] = False
        _EXP_ARM["n"] += 1
    except Exception:
        pass
# ==== end EXP instrumentation ====
'''

q = f"{BASE}/models/qwen3.py"
with open(q, encoding="utf-8") as f:
    qsrc = f.read()
anchor_log = "logger = logging.getLogger(__name__)"
assert qsrc.count(anchor_log) >= 1, "qwen3 logger anchor"
qsrc = qsrc.replace(anchor_log, anchor_log + "\n" + EXP_BLOCK_QWEN3, 1)

old_rot = """            q, k = layer.self_attn.rotary_emb(positions, q, k_pre)
            repair_k = k_pre if raw_kv_position_mode == "pre_rope" else k
            raw_key_values.append("""
new_rot = """            _exp_kb = k_pre.detach().clone() if _EXP_HOOKS else None
            q, k = layer.self_attn.rotary_emb(positions, q, k_pre)
            repair_k = k_pre if raw_kv_position_mode == "pre_rope" else k
            if _exp_kb is not None:
                if not _EXP_PHASE_REC["layers"]:
                    _EXP_PHASE_REC["meta"] = {
                        "rotary_type": type(layer.self_attn.rotary_emb).__name__,
                        "rotary_module": type(layer.self_attn.rotary_emb).__module__,
                        "dtype": str(k_pre.dtype),
                        "shape": tuple(k_pre.shape),
                        "positions_head": positions[:6].detach().tolist(),
                        "positions_dtype": str(positions.dtype),
                        "raw_kv_position_mode": raw_kv_position_mode,
                        "span": [int(span_start), int(span_end)],
                        "seq_len": int(seq_len),
                    }
                _EXP_PHASE_REC["layers"].append({
                    "layer": len(_EXP_PHASE_REC["layers"]),
                    "k_before_span": _exp_kb[span_start:span_end].float().cpu(),
                    "k_pre_now_span": k_pre.detach()[span_start:span_end].float().cpu(),
                    "inplace_max": (k_pre.detach() - _exp_kb).abs().max().item(),
                })
            raw_key_values.append("""
assert qsrc.count(old_rot) == 1, "rotary anchor"
qsrc = qsrc.replace(old_rot, new_rot)

old_tail = """        if repair_position_ids is None:"""
new_tail = """        if _EXP_HOOKS:
            try:
                _exp_save_phase({"repair_position_ids_head": (list(repair_position_ids[:6]) if repair_position_ids is not None else None)})
                _exp_reset_phase()
            except Exception:
                pass
        if repair_position_ids is None:"""
assert qsrc.count(old_tail) == 1, "tail anchor"
qsrc = qsrc.replace(old_tail, new_tail)

old_prep = """    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)"""
new_prep = """    def forward_prepare_native(self, positions, hidden_states):
        if _EXP_HOOKS:
            _fb = getattr(self, "_active_forward_batch", None)
            _gm = getattr(_fb, "c2kv_use_gist_projection", None)
            _st = "none" if _gm is None else ("anytrue" if bool(_gm.any().item()) else "allfalse")
            _exp_prep_log("native", positions, _st)
            _exp_maybe_snapshot(positions, _fb)
        qkv, _ = self.qkv_proj(hidden_states)"""
assert qsrc.count(old_prep) == 1, "prep anchor"
qsrc = qsrc.replace(old_prep, new_prep)

old_npu = """    def forward_prepare_npu(self, positions, hidden_states, forward_batch):"""
new_npu = """    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        if _EXP_HOOKS:
            _gm = getattr(forward_batch, "c2kv_use_gist_projection", None)
            _st = "none" if _gm is None else ("anytrue" if bool(_gm.any().item()) else "allfalse")
            _exp_prep_log("npu_fused", positions, _st)"""
assert qsrc.count(old_npu) == 1, "npu prep anchor"
qsrc = qsrc.replace(old_npu, new_npu)

with open(q, "w", encoding="utf-8") as f:
    f.write(qsrc)
print("patched", q)

s = f"{BASE}/managers/scheduler.py"
old_inj = """        position_ids = self.c2kv_pool.get_position_ids(entry)
        position_start = int(position_ids[0].item())
        position_end = int(position_ids[-1].item()) + 1

        req.kv_committed_len = kv_start + repair_len"""
new_inj = '''        position_ids = self.c2kv_pool.get_position_ids(entry)
        position_start = int(position_ids[0].item())
        position_end = int(position_ids[-1].item()) + 1

        if os.environ.get("EXP_HOOKS") == "1":
            try:
                import json as _json, time as _time
                _out = os.environ.get("EXP_OUT_DIR", "/tmp/zh_exp/out")
                os.makedirs(_out, exist_ok=True)
                _kv = model_runner.token_to_kv_pool_allocator.get_kvcache()
                _loc = loc.detach().to(torch.long).reshape(-1)

                def _gather_at(li, loc_idx):
                    b = _kv.get_key_buffer(li)
                    f = b.reshape(-1, b.shape[-2], b.shape[-1])
                    return f[loc_idx].float().cpu()

                _rec = {
                    "ts": _time.time(), "rid": req.rid,
                    "repair_mode": entry.repair_mode,
                    "key_hash": entry.key_hash[:16],
                    "token_len": int(entry.token_len),
                    "already_rotated": bool(entry.already_rotated),
                    "kv_start": int(kv_start),
                    "abs_pos": [int(x) for x in position_ids.tolist()],
                }
                with open(os.path.join(_out, "inject_log.jsonl"), "a") as f:
                    f.write(json.dumps(_rec) + "\\n")
                if entry.repair_mode == "d_corr_w2":
                    _nl = self.c2kv_pool.num_layers
                    _st = self.c2kv_pool.start_layer
                    _locall = model_runner.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : req.kv_committed_len
                    ].detach().to(torch.long).reshape(-1)

                    def _gather_pref_k(li):
                        b = _kv.get_key_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_locall].clone()

                    def _gather_pref_v(li):
                        b = _kv.get_value_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_locall].clone()

                    torch.save(
                        {
                            "rec": _rec,
                            "K": torch.stack([_gather_pref_k(_st + i) for i in range(_nl)]),
                            "V": torch.stack([_gather_pref_v(_st + i) for i in range(_nl)]),
                            "loc_prefix": _locall.cpu(),
                            "loc_repair": _loc.cpu(),
                            "repair_kv_len": int(entry.token_len),
                            "origin_input_ids": [int(x) for x in (getattr(req, "origin_input_ids", None) or [])],
                            "input_ids_now": [int(x) for x in (getattr(req, "input_ids", None) or [])],
                            "fill_ids_now": [int(x) for x in (getattr(req, "fill_ids", None) or [])],
                            "c2kv_position_correction": int(getattr(req, "c2kv_position_correction", 0) or 0),
                        },
                        os.path.join(
                            _out,
                            f"prefix_{_rec['key_hash']}_{int(_time.time() * 1000) % 100000}.pt",
                        ),
                    )
                    import sglang.srt.models.qwen3 as _q3
                    _q3._EXP_ARM.update({"armed": True, "rid": req.rid})
            except Exception:
                logger.warning("EXP inject dump failed", exc_info=True)

        req.kv_committed_len = kv_start + repair_len'''
patch(s, [(old_inj, new_inj, 1)])

c = f"{BASE}/entrypoints/openai/serving_chat.py"
old_resp = """        self._commit_persistent_history_session(adapted_request, ret)

        response = self._build_chat_response("""
new_resp = """        self._commit_persistent_history_session(adapted_request, ret)

        import os as _os, json as _json, time as _time
        if _os.environ.get("EXP_HOOKS") == "1":
            try:
                _out = _os.environ.get("EXP_OUT_DIR", "/tmp/zh_exp/out")
                _os.makedirs(_out, exist_ok=True)
                _r0 = ret[0] if isinstance(ret, list) and ret else {}
                _mi = _r0.get("meta_info") or {}
                with open(_os.path.join(_out, "gen_log.jsonl"), "a") as f:
                    f.write(_json.dumps({
                        "ts": _time.time(),
                        "model": str(getattr(request, "model", "")),
                        "finish_reason": _mi.get("finish_reason"),
                        "output_ids": list(_r0.get("output_ids") or []),
                        "text": str(_r0.get("text") or "")[:4000],
                        "prompt_tokens": (_mi.get("prompt_tokens")),
                    }) + "\\n")
            except Exception:
                pass

        response = self._build_chat_response("""
patch(c, [(old_resp, new_resp, 1)])

for p in [q, s, c]:
    py_compile.compile(p, doraise=True)
print("ALL PATCHED + COMPILED OK")
