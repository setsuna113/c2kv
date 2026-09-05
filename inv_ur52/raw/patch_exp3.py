import py_compile

def patch(path, subs):
    src = open(path, encoding="utf-8").read()
    for old, new, cnt in subs:
        n = src.count(old)
        assert n == cnt, f"{path}: {n} != {cnt} for {old[:70]!r}"
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)

Q = "/tmp/zh_exp/python/sglang/srt/models/qwen3.py"
old = """def _exp_maybe_snapshot(positions, fb):
    if not _EXP_HOOKS or not _EXP_ARM.get("armed") or fb is None:
        return
    try:
        import os as _os, torch as _torch
        mode = str(getattr(fb, "forward_mode", ""))
        if "EXTEND" not in mode.upper():
            return
        if positions is None or positions.numel() > 512:
            return"""
new = """def _exp_maybe_snapshot(positions, fb):
    if not _EXP_HOOKS or not _EXP_ARM.get("armed") or fb is None:
        return
    try:
        import os as _os, torch as _torch
        mode = str(getattr(fb, "forward_mode", ""))
        if "EXTEND" not in mode.upper():
            return
        _idx = _EXP_ARM.get("req_pool_idx")
        _fbidx = getattr(fb, "req_pool_indices", None)
        if _idx is not None and _fbidx is not None and len(_fbidx) > 0:
            if int(_fbidx[0]) != int(_idx):
                return
        if positions is None or positions.numel() > 16384:
            return"""
patch(Q, [(old, new, 1)])

S = "/tmp/zh_exp/python/sglang/srt/managers/scheduler.py"
old2 = """                    import sglang.srt.models.qwen3 as _q3
                    _q3._EXP_ARM.update({"armed": True, "rid": req.rid})"""
new2 = """                    import sglang.srt.models.qwen3 as _q3
                    _q3._EXP_ARM.update({
                        "armed": True,
                        "rid": req.rid,
                        "req_pool_idx": int(req.req_pool_idx),
                    })"""
patch(S, [(old2, new2, 1)])

py_compile.compile(Q, doraise=True)
py_compile.compile(S, doraise=True)
print("PATCH4 OK")
