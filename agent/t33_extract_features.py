# -*- coding: utf-8 -*-
"""t33 feature extraction (survey 4.1-4.4) from the capture artifacts.

Runs where the capture artifacts live (server preferred: the tokenizer is
available for SMT token offsets and the vocabulary size).  Pure numpy/stdlib.

Inputs : <capture_dir>/<arm>/p0.steps.jsonl   (steps + spans + IC + text)
         <capture_dir>/<arm>/p0.docs.jsonl    (doc sidecar incl. gist stats)
         --tokenizer (optional; improves the SMT mask, provides |V|)
Output : one features.jsonl row per qid — SCALAR features only.  Labels never
enter here; hidden-state probe inputs stay in p0.hid.npz for the fitter.

Column naming goes through the t33_labels guard at write time.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import guard_columns  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402

CAP_TOKENS = 128
ROLLING_W = 8
REPEAT_WINDOW = 16
FALLBACK_VOCAB = 151936


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with io.open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- SMT mask

def arg_key_value_char_spans(args_text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Char spans of JSON keys and leaf-value strings inside an arguments
    object text (best effort, truncation tolerant)."""
    keys: List[Tuple[int, int]] = []
    values: List[Tuple[int, int]] = []
    i = 0
    n = len(args_text)
    while i < n:
        c = args_text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if args_text[j] == "\\":
                    j += 2
                    continue
                if args_text[j] == '"':
                    break
                j += 1
            span = (i + 1, min(j, n))
            if _is_key_position(args_text, i):
                keys.append(span)
            else:
                values.append(span)
            i = j + 1
            continue
        i += 1
    return keys, values


def _is_key_position(text: str, quote_start: int) -> bool:
    """A quoted string is a KEY when the next non-space char after its close
    is ':' AND the previous non-space char is one of '{' or ','."""
    j = quote_start + 1
    while j < len(text) and text[j] != '"':
        if text[j] == "\\":
            j += 1
        j += 1
    k = j + 1
    while k < len(text) and text[k] in " \t\r\n":
        k += 1
    if k >= len(text) or text[k] != ":":
        return False
    p = quote_start - 1
    while p >= 0 and text[p] in " \t\r\n":
        p -= 1
    return p < 0 or text[p] in "{,"


def smt_token_set(text: str, spans: Dict[str, Any], offsets: Sequence[Tuple[int, int]]) -> Optional[set]:
    """FC-UQ SMT mask, mapped to token indices: the six grammar classes
    collapse to (opening token, name value tokens, arg key tokens, arg value
    tokens, separators) — separators excluded from the mask, per the paper."""
    parsed = parse_tool_call(text)
    toks: set = set()
    def add_span(cs: int, ce: int) -> None:
        for idx, (s, e) in enumerate(offsets):
            if e > cs and s < ce:
                toks.add(idx)
    if spans.get("name_first") is not None:
        toks.add(spans["name_first"])  # opening/decision token region start
    if parsed.get("name_span"):
        add_span(*parsed["name_span"])
    if parsed.get("args_span"):
        args_text = text[parsed["args_span"][0]:parsed["args_span"][1]]
        base = parsed["args_span"][0]
        keys, values = arg_key_value_char_spans(args_text)
        for cs, ce in values:
            add_span(base + cs, base + ce)
        # arg keys stay OUT of the semantic mask (they are schema echoes of
        # the tool definition); the paper's class (3) is excluded, class (4)
        # included — we follow the value-only reading.
    return toks if toks else None


# ---------------------------------------------------------------- features

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rolling_baseline(series: List[float], w: int) -> List[float]:
    out: List[float] = []
    for t in range(len(series)):
        lo = max(0, t - w)
        window = series[lo:t]
        base = sum(window) / len(window) if window else series[t]
        out.append(series[t] - base)
    return out


def _repeat_trigram_coverage(ids: Sequence[int], window: int) -> List[float]:
    out: List[float] = []
    for t in range(len(ids)):
        lo = max(0, t - window)
        local = [tuple(ids[i:i + 3]) for i in range(lo, max(lo, t - 2))]
        cur = [tuple(ids[i:i + 3]) for i in range(max(0, t - 2), t + 1)]
        if not local or not cur:
            out.append(0.0)
            continue
        seen = set(local)
        reps = sum(1 for g in cur if g in seen and cur.count(g) > 1)
        out.append(reps / len(cur))
    return out


def steps_features(rec: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = rec.get("steps") or []
    spans: Dict[str, Any] = rec.get("spans") or {}
    text: str = rec.get("text") or ""
    gen_ids: List[int] = rec.get("generated_ids") or []
    n = len(steps)
    f: Dict[str, Any] = {"n_generated": n, "censored": n >= CAP_TOKENS}
    if n == 0:
        return f
    lps = [s["chosen_logprob"] for s in steps]
    ents = [s["entropy_full"] for s in steps]

    nf, nl = spans.get("name_first"), spans.get("name_last")
    af, al = spans.get("args_first"), spans.get("args_last")
    pf, pl = spans.get("payload_first"), spans.get("payload_last")

    def rng(a, b):
        return range(a, b + 1) if (a is not None and b is not None and b >= a and b < n) else range(0)

    name_idx = list(rng(nf, nl))
    args_idx = list(rng(af, al))
    payload_idx = list(rng(pf, pl)) or list(range(n))

    f["n_name_tokens"] = len(name_idx) or None
    f["n_args_tokens"] = len(args_idx) or None

    # --- 4.2 FLARE / FC-UQ ---
    f["flare_min_p_all"] = math.exp(min(lps))
    f["flare_min_p_span"] = math.exp(min((lps[i] for i in payload_idx), default=min(lps)))
    f["flare_min_p_name"] = math.exp(min((lps[i] for i in name_idx), default=min(lps)))
    nll = [-x for x in lps]
    f["fc_max_nll_all"] = max(nll)
    f["fc_avg_nll_all"] = sum(nll) / n
    f["fc_gnll_all"] = sum(nll)
    f["name_region_nll"] = -sum(lps[i] for i in name_idx) if name_idx else None
    f["args_region_nll"] = -sum(lps[i] for i in args_idx) if args_idx else None

    smt: Optional[set] = None
    if tokenizer is not None and gen_ids:
        try:
            from t33_spanmap import token_char_offsets
            offs = token_char_offsets(lambda ids: tokenizer.decode(ids, skip_special_tokens=True), gen_ids)
            smt = smt_token_set(text, spans, offs)
        except Exception:
            smt = None
    if smt is None:
        smt = set(name_idx) | set(args_idx)
        f["smt_mask_source"] = "span_fallback"
    else:
        f["smt_mask_source"] = "tokenizer_offsets"
    smt_lps = [lps[i] for i in sorted(smt) if i < n]
    if smt_lps:
        f["fc_max_nll_smt"] = max(-x for x in smt_lps)
        f["fc_avg_nll_smt"] = sum(-x for x in smt_lps) / len(smt_lps)
        f["fc_gnll_smt"] = sum(-x for x in smt_lps)
        f["smt_token_frac"] = len(smt_lps) / n

    # --- 4.2 Leyline / KnowNo (name first token) ---
    vocab = FALLBACK_VOCAB
    if tokenizer is not None:
        try:
            vocab = len(tokenizer)
        except Exception:
            pass
    if nf is not None and nf < n:
        top5 = steps[nf].get("top5") or []
        if len(top5) >= 2:
            f["leyline_margin_name_first"] = top5[0][0] - top5[1][0]
        if top5:
            f["p1_name_first"] = math.exp(top5[0][0])
        pool = set(((rec.get("ic") or {}).get("candidate_token_ids")) or [])
        if pool:
            in_top = [(tid, math.exp(lp)) for lp, tid in top5 if tid in pool]
            mass = sum(p for _t, p in in_top)
            f["kono_pool_mass_top5"] = mass
            f["kono_none_mass"] = max(0.0, 1.0 - mass)
            f["kono_top_pool_prob"] = max((p for _t, p in in_top), default=0.0)
            f["kono_n_pool_in_top5"] = len(in_top)
            f["kono_emitted_in_pool"] = int(steps[nf]["token_id"] in pool)
    margins = []
    for s in steps:
        t5 = s.get("top5") or []
        if len(t5) >= 2:
            margins.append(t5[0][0] - t5[1][0])
    if margins:
        f["margin_min_all"] = min(margins)
        f["margin_mean_all"] = sum(margins) / len(margins)

    # --- 4.3 entropy family ---
    f["hbar_all"] = sum(ents) / n
    f["entropycache_max_all"] = max(ents)
    if rec.get("stop_reason") == "eos" and n > 1:
        f["entropycache_max_no_eos"] = max(ents[:-1])
        f["hbar_no_eos"] = sum(ents[:-1]) / (n - 1)
    if name_idx:
        f["hbar_name"] = sum(ents[i] for i in name_idx) / len(name_idx)
        f["entropy_name_max"] = max(ents[i] for i in name_idx)
    if args_idx:
        f["hbar_args"] = sum(ents[i] for i in args_idx) / len(args_idx)
        f["entropy_args_max"] = max(ents[i] for i in args_idx)
    if f.get("hbar_args") is not None and f.get("hbar_name") is not None:
        f["ergo_dh_region"] = f["hbar_args"] - f["hbar_name"]
    f["entropy_max_span"] = max((ents[i] for i in payload_idx), default=max(ents))

    # SVIP: no span averaging — point readouts at name/arg-value starts
    svip_points = {"name_first": nf, "args_first": af}
    for label, pos in svip_points.items():
        if pos is not None and pos < n:
            f[f"svip_sqrt_h_{label}"] = math.sqrt(max(0.0, ents[pos]))
    # arg-value first tokens: first tokens of each leaf value (approximation:
    # the first token after each arg key) — via the SMT value char spans
    if tokenizer is not None and gen_ids and spans.get("args_first") is not None:
        try:
            from t33_spanmap import token_char_offsets
            offs = token_char_offsets(lambda ids: tokenizer.decode(ids, skip_special_tokens=True), gen_ids)
            parsed = parse_tool_call(text)
            if parsed.get("args_span"):
                args_text = text[parsed["args_span"][0]:parsed["args_span"][1]]
                base = parsed["args_span"][0]
                _keys, values = arg_key_value_char_spans(args_text)
                sq = []
                for cs, ce in values:
                    for idx, (s, e) in enumerate(offs):
                        if e > base + cs and s < base + ce:
                            sq.append(math.sqrt(max(0.0, ents[idx])))
                            break
                if sq:
                    f["svip_sqrt_h_argvalue_max"] = max(sq)
        except Exception:
            pass

    # --- 4.3 CONF-KV composite ---
    log_v = math.log(vocab)
    cs_vals = []
    for s in steps:
        t5 = s.get("top5") or []
        p1 = math.exp(t5[0][0]) if t5 else math.exp(s["chosen_logprob"])
        m = (t5[0][0] - t5[1][0]) if len(t5) >= 2 else 0.0
        hhat = s["entropy_full"] / log_v if log_v > 0 else 0.0
        cs_vals.append(0.4 * (1 - hhat) + 0.3 * _sigmoid(m) + 0.3 * p1)
    f["confkv_c_min"] = min(cs_vals)
    f["confkv_c_mean"] = sum(cs_vals) / len(cs_vals)
    if nf is not None and nf < len(cs_vals):
        f["confkv_c_name"] = cs_vals[nf]

    # --- 4.3 e-CUSUM relative entropy + repeat channel ---
    u = _rolling_baseline(ents, ROLLING_W)
    hbar = f["hbar_all"]
    u_norm = [max(0.0, x) / (max(0.0, x) + hbar + 1e-9) for x in u]
    r = _repeat_trigram_coverage(gen_ids, REPEAT_WINDOW) if gen_ids else [0.0] * n
    a = [min(1.0, 0.7 * rv + 0.3 * un) for rv, un in zip(r, u_norm)]
    f["ecusum_u_max"] = max(u)
    f["ecusum_u_mean"] = sum(u) / n
    f["ecusum_a_max"] = max(a)
    f["ecusum_a_mean"] = sum(a) / n
    f["ecusum_a_seq"] = [round(x, 5) for x in a]

    # --- 4.4 IC / ALIEN scalars ---
    ic = rec.get("ic") or {}
    anchors_ic = ic.get("anchors") or {}
    for pos_label in ("name_first", "name_last"):
        node = anchors_ic.get(pos_label) or {}
        for key in ("ic_uniform", "ic_lastk", "first_agree_layer", "margin_final"):
            if node.get(key) is not None:
                f[f"ic_{key}_{pos_label}"] = node[key]

    return f


def docs_features(doc: Dict[str, Any]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    n_kept = doc.get("n_docs_kept")
    if n_kept is None:
        return f
    f["s8_n_docs_kept"] = n_kept
    f["s8_dropped_docs"] = doc.get("dropped_docs") or 0
    f["s8_kept_frac"] = (n_kept / doc["n_docs_original"]) if doc.get("n_docs_original") else 1.0
    f["s8_packing_sat"] = int(n_kept >= 16)
    lens = [x for x in (doc.get("doc_token_lens") or []) if x]
    if lens:
        f["s8_doc_tokens_sum"] = sum(lens)
        f["s8_n_ctx"] = sum(lens)
        f["boundary_max_doc_len"] = max(lens)
        f["boundary_mean_doc_len"] = sum(lens) / len(lens)
        mx = max(range(len(lens)), key=lambda i: lens[i])
        f["boundary_longest_doc_pos_frac"] = mx / max(1, len(lens) - 1)
    ratios = [x for x in (doc.get("doc_gzip_ratios") or []) if x]
    if ratios:
        f["gzip_ratio_mean"] = sum(ratios) / len(ratios)
        f["gzip_ratio_min"] = min(ratios)
        f["gzip_ratio_max"] = max(ratios)
    sur = [s.get("score") for s in (doc.get("surprise") or []) if s.get("score") is not None]
    if sur:
        f["surprise_max_k"] = max(sur)
        f["surprise_mean_k"] = sum(sur) / len(sur)
    hits = [s.get("hit_rate") for s in (doc.get("surprise") or []) if s.get("hit_rate") is not None]
    if hits:
        f["surprise_hit_rate_mean"] = sum(hits) / len(hits)
    f["rung0_dropped_any"] = int(bool(doc.get("dropped_docs")))
    gs = doc.get("gist_stats") or {}
    per_layer = gs.get("per_layer") or {}
    heavy_layers = [li for li in per_layer if "hoyer_mean_over_pos" in (per_layer[li].get("k") or {})]

    def collect(stat_name: str, kv: str, field: str) -> Optional[float]:
        vals = []
        for li in heavy_layers:
            node = per_layer[li].get(kv) or {}
            for pd in (node.get(stat_name) or []):
                if isinstance(pd, dict) and pd.get(field) is not None:
                    vals.append(pd[field])
        return (sum(vals) / len(vals)) if vals else None

    # per-doc stats exist only on heavy layers; norms on all layers
    def collect_pool(stat_name: str, kv: str, field: str) -> Optional[float]:
        vals = []
        for li, node_kv in per_layer.items():
            node = node_kv.get(kv) or {}
            pooled = node.get(stat_name)
            if isinstance(pooled, dict) and pooled.get(field) is not None:
                vals.append(pooled[field])
        return (sum(vals) / len(vals)) if vals else None

    for stat, tag in (("per_doc_hoyer", "sat_hoyer"),
                      ("per_doc_spectral_entropy", "sat_spec_ent")):
        v = collect(stat, "k", "mean")
        if v is not None:
            f[f"{tag}_k"] = v
        v = collect(stat, "v", "mean")
        if v is not None:
            f[f"{tag}_v"] = v
    for kv in ("k", "v"):
        v = collect_pool("norm_mean_over_pos", kv, "mean")
        if v is not None:
            f[f"sat_norm_{kv}"] = v
        v = collect_pool("kurtosis_mean_over_pos", kv, "mean")
        if v is not None:
            f[f"sat_kurt_{kv}"] = v
        v = collect_pool("spectral_entropy_mean_over_pos", kv, "mean")
        if v is not None:
            f[f"sat_specent_pool_{kv}"] = v
    counts = gs.get("per_row_gist_counts") or []
    if counts:
        kept_counts = [c for c in counts if c > 0]
        f["gist_gists_per_doc_mean"] = (sum(kept_counts) / len(kept_counts)) if kept_counts else None
        f["gist_gists_per_doc_max"] = max(kept_counts) if kept_counts else None
    return f


def text_surface_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist text-surface features (concealment gap s_text).  Field names
    never touch target/gold/_match/_f1/rouge — import-time guard applies."""
    text = rec.get("text") or ""
    spans = rec.get("spans") or {}
    parsed = parse_tool_call(text)
    f: Dict[str, Any] = {
        "text_n_chars": len(text),
        "text_has_tool_call": int(bool(spans.get("has_tool_call"))),
        "text_closed_tag": int(bool(spans.get("closed"))),
        "text_parse_ok": int(bool(spans.get("parse_ok"))),
        "text_name_len": (len(parsed["name"]) if parsed.get("name") else 0) or None,
        "text_payload_chars": (parsed["payload_span"][1] - parsed["payload_span"][0]) if parsed.get("payload_span") else None,
    }
    if text:
        digits = sum(c.isdigit() for c in text)
        braces = text.count("{") + text.count("}")
        f["text_digit_frac"] = digits / len(text)
        f["text_brace_count"] = braces
        toks = text.split()
        if toks:
            f["text_distinct_word_frac"] = len(set(toks)) / len(toks)
    return f


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture_dir", required=True)
    parser.add_argument("--arm", required=True, choices=["full", "c2kv"])
    parser.add_argument("--part", default="p0")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cap = Path(args.capture_dir) / args.arm
    steps_rows = load_jsonl(cap / f"{args.part}.steps.jsonl")
    docs_rows = {r.get("qid"): r for r in load_jsonl(cap / f"{args.part}.docs.jsonl")}

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] tokenizer unavailable ({exc}); SMT mask falls back to spans", file=sys.stderr)

    out_features: List[Dict[str, Any]] = []
    for rec in steps_rows:
        qid = rec["qid"]
        f: Dict[str, Any] = {"qid": qid, "arm": args.arm}
        f.update(steps_features(rec, tokenizer))
        f.update(docs_features(docs_rows.get(qid) or {}))
        f.update(text_surface_features(rec))
        f["stop_reason"] = rec.get("stop_reason")
        out_features.append(f)

    # leakage guard over every emitted column
    cols = set()
    for f in out_features:
        cols.update(f.keys())
    guard_columns(sorted(cols), context=f"features[{args.arm}]")

    with io.open(args.out, "w", encoding="utf-8") as fh:
        for f in out_features:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"[t33] wrote {len(out_features)} feature rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
