import py_compile

def patch(path, subs):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in subs:
        n = src.count(old)
        assert n == cnt, f"{path}: anchor count {n} != {cnt} for {old[:70]!r}"
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched", path)

S = "/tmp/zh_exp/python/sglang/srt/managers/scheduler.py"
old = """                    def _gather_pref_v(li):
                        b = _kv.get_value_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_locall].clone()
"""
new = """                    def _gather_pref_v(li):
                        b = _kv.get_value_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_locall].clone()

                    def _gather_rep_k(li):
                        b = _kv.get_key_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_loc].float().cpu()

                    def _gather_rep_v(li):
                        b = _kv.get_value_buffer(li)
                        f = b.reshape(-1, b.shape[-2], b.shape[-1])
                        return f[_loc].float().cpu()
"""
patch(S, [(old, new, 1)])

old2 = """                            "K": torch.stack([_gather_pref_k(_st + i) for i in range(_nl)]),
                            "V": torch.stack([_gather_pref_v(_st + i) for i in range(_nl)]),"""
new2 = """                            "K": torch.stack([_gather_pref_k(_st + i) for i in range(_nl)]),
                            "V": torch.stack([_gather_pref_v(_st + i) for i in range(_nl)]),
                            "K_repair": torch.stack([_gather_rep_k(_st + i) for i in range(_nl)]),
                            "V_repair": torch.stack([_gather_rep_v(_st + i) for i in range(_nl)]),"""
patch(S, [(old2, new2, 1)])

Q = "/tmp/zh_exp/python/sglang/srt/models/qwen3.py"
old3 = """        mode = str(getattr(fb, "forward_mode", ""))
        if "EXTEND" not in mode.upper():
            return"""
new3 = """        mode = str(getattr(fb, "forward_mode", ""))
        if "EXTEND" not in mode.upper():
            return
        if positions is None or positions.numel() > 512:
            return"""
patch(Q, [(old3, new3, 1)])

py_compile.compile(S, doraise=True)
py_compile.compile(Q, doraise=True)
print("PATCH2 OK")
