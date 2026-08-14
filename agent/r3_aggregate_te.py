"""R3: aggregate T-E arms (trusted full via sglang + c2kv@4) with Wilson CIs.

Joins the raw-text generations (trusted full arm) with the frozen S1 targets
using the harness's own _extract_tool_name, and summarizes the c2kv rerun
jsonl. Prints a compact report used verbatim in the PR-F tables.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("R3_LYC", str(Path.home() / "c2kv")))
R3 = ROOT / "outputs_lyc/r3_discrimination"
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "agent"))
sys.path.insert(0, str(_REPO / "python"))

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from eval_agent_tool_definition_c2kv import _extract_tool_name  # noqa: E402


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(c / d - h, 4), round(c / d + h, 4))


def main():
    s1 = {}
    for line in (ROOT / "outputs_lyc/r2_bigpool/s1_full_48.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        s1[row["qid"]] = row

    gen_dir = R3 / "t_e/full_trusted"
    gens = [json.loads(l) for l in (gen_dir / "t_a_generations.jsonl").open(encoding="utf-8") if l.strip()]
    n = calls = acc = empty = 0
    for g in gens:
        qid_file = g["qid"].replace(":", "_")
        text = (gen_dir / f"gen_{qid_file}.txt").read_text(encoding="utf-8")
        if not text.strip():
            empty += 1
            continue
        n += 1
        tt = _extract_tool_name(s1[g["qid"]]["target"].strip())
        tc = _extract_tool_name(text)
        calls += bool("<tool_call>" in text or "Action:" in text)
        acc += tt is not None and tc == tt
    print("== trusted full arm (sglang ascend, base model, raw text) ==")
    print(f"N=48, non-empty={n}, empty={empty}")
    print(f"call_rate {calls}/48 -> {wilson(calls, 48)}")
    print(f"tool_name_acc {acc}/48 -> {wilson(acc, 48)}")

    c2 = [json.loads(l) for l in (R3 / "t_e/t_e_c2kv_r4.jsonl").open(encoding="utf-8") if l.strip()]
    valid = [r for r in c2 if not r.get("skipped")]
    nv = len(valid)
    c_calls = sum(1 for r in valid if r.get("has_tool_call"))
    c_acc = sum(1 for r in valid if r.get("tool_name_match"))
    print("== c2kv@4 arm (eager, 512x160 chunks) ==")
    print(f"rows={len(c2)} valid={nv} skipped={len(c2) - nv}")
    print(f"call_rate {c_calls}/{nv} -> {wilson(c_calls, nv)}")
    print(f"tool_name_acc {c_acc}/{nv} -> {wilson(c_acc, nv)}")
    ratios = [r.get("actual_compression_ratio") for r in valid if r.get("actual_compression_ratio")]
    if ratios:
        print(f"actual_compression_ratio avg {sum(ratios)/len(ratios):.3f} min {min(ratios)} max {max(ratios)}")
    gist = [r.get("gist_tokens") for r in valid if r.get("gist_tokens") is not None]
    if gist:
        print(f"gist_tokens avg {sum(gist)/len(gist):.0f} min {min(gist)} max {max(gist)}")

    tb = [json.loads(l) for l in (R3 / "t_b/t_b_full_32k.jsonl").open(encoding="utf-8") if l.strip()]
    vb = [r for r in tb if not r.get("skipped")]
    b_calls = sum(1 for r in vb if r.get("has_tool_call"))
    b_acc = sum(1 for r in vb if r.get("tool_name_match"))
    print("== T-B full arm, tool pool capped 32k, npu_fusion_attention ==")
    print(f"valid={len(vb)}  call_rate {b_calls}/{len(vb)} -> {wilson(b_calls, len(vb))}")
    print(f"tool_name_acc {b_acc}/{len(vb)} -> {wilson(b_acc, len(vb))}")


if __name__ == "__main__":
    main()
