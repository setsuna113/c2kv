"""Task D (BDF pilot): freeze the E-sham neutral-span plan.

E-sham is the noise floor for the append-only erratum arms: an equal-length
span of information-free text pushed through the SAME injection path as
E-corr (standalone prefill, RoPE-rotated onto doc k*'s absolute start).  Byte
budget equality is by construction, not by search — the span length is taken
verbatim from ``len(doc k*)``, so ``abs_delta_frac`` is exactly zero and the
gate is ``== 0`` rather than r4's ``<= 0.02``.

Per C->W qid:
  k*        = (n_docs - 1) // 2          (median doc; both halves non-trivial)
  L         = doc_lens[k*]
  offset    = sha256("<seed>:<qid>") mod corpus_tokens
  sham ids  = L tokens taken from the corpus token ring starting at offset

Declared asymmetry (also recorded in configs/bdf_pilot/d_prereg.md): the
E-corr slice carries the full left context of the session, the E-sham slice
carries the context of a neutral essay.  That is an inherent property of
"text with no task information", not a fixable defect, and it is reported the
way r4 reported its chunk-local limitation.

CPU only, torch-free.  Usage (repo root):
  python agent/d_sham_plan.py \
      --doc_table results/d/d_doc_ids.json \
      --manifest configs/bdf_pilot/d_cw_manifest.json \
      --corpus configs/bdf_pilot/d_neutral_corpus.txt \
      --tokenizer ./models/Qwen3-4B-Instruct-2507 \
      --out configs/bdf_pilot/d_sham_plan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    for _sub in ("python", "agent"):
        _path = str(_ROOT / _sub)
        if _path not in sys.path:
            sys.path.insert(0, _path)

from extract_cw_triggers import sha256_text_file as _sha256_file  # noqa: E402

logger = logging.getLogger("d_sham_plan")

SEED = 20260815
RULE_VERSION = "d_sham_v1"

# Neutrality gate.  Structural characters are the same family r4 treated as
# control tokens; the word list covers harness / benchmark vocabulary that
# must never appear in a "no information" span.
FORBIDDEN_CHAR_RE = re.compile(r"[<>{}\[\]\"'`_:;0-9\\|/@#$%^&*=+~]")
FORBIDDEN_WORDS = (
    "tool", "tools", "json", "api", "function", "functions", "agent", "agents",
    "call", "calls", "called", "server", "servers", "user", "users", "assistant",
    "model", "models", "prompt", "dataset", "token", "tokens", "endpoint",
    "schema", "payload", "request", "response", "query", "parameter",
    "parameters", "argument", "arguments", "key", "keys", "value", "values",
    "name", "action", "environment", "directory", "file", "files", "http",
    "appworld", "toolathlon", "mcp",
)
FORBIDDEN_WORD_RE = re.compile(
    r"\b(?:" + "|".join(FORBIDDEN_WORDS) + r")\b", flags=re.IGNORECASE
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def k_star_for(n_docs: int) -> int:
    """Median document index. Frozen definition; do not tune."""
    return (n_docs - 1) // 2


def corpus_offset(seed: int, qid: str, corpus_tokens: int) -> int:
    """Per-qid start position on the corpus token ring."""
    if corpus_tokens <= 0:
        raise ValueError("empty corpus")
    digest = hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % corpus_tokens


def ring_slice(token_ids: Sequence[int], offset: int, length: int) -> List[int]:
    """Take `length` ids from the corpus ring; wraps as many times as needed."""
    total = len(token_ids)
    if total <= 0:
        raise ValueError("empty corpus")
    return [int(token_ids[(offset + i) % total]) for i in range(length)]


def neutrality_violations(text: str) -> List[str]:
    """Return the offending characters / words in a candidate sham span."""
    violations = sorted({f"char:{c}" for c in FORBIDDEN_CHAR_RE.findall(text)})
    violations.extend(sorted({f"word:{w.lower()}" for w in FORBIDDEN_WORD_RE.findall(text)}))
    return violations


def build_plan(
    per_qid_docs: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    corpus_ids: Sequence[int],
    decode: Callable[[List[int]], str],
    *,
    seed: int = SEED,
    header: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the frozen sham plan document.

    `per_qid_docs` is the doc-length side table written by
    extract_cw_triggers --bind_docs; `qids` is the frozen C->W list.
    """
    per_qid: Dict[str, Any] = {}
    missing: List[str] = []
    degenerate: List[str] = []
    violating: List[str] = []
    corr_total = 0
    sham_total = 0
    for qid in qids:
        entry = per_qid_docs.get(qid)
        if entry is None:
            missing.append(qid)
            continue
        doc_lens = [int(x) for x in entry["doc_lens"]]
        n_docs = int(entry.get("n_docs", len(doc_lens)))
        if n_docs != len(doc_lens):
            raise SystemExit(f"FATAL: qid {qid} n_docs={n_docs} but {len(doc_lens)} doc_lens")
        if n_docs <= 0:
            degenerate.append(qid)
            continue
        k = k_star_for(n_docs)
        span_len = doc_lens[k]
        if span_len <= 0:
            degenerate.append(qid)
            continue
        offset = corpus_offset(seed, qid, len(corpus_ids))
        sham_ids = ring_slice(corpus_ids, offset, span_len)
        sham_text = decode(sham_ids)
        violations = neutrality_violations(sham_text)
        if violations:
            violating.append(qid)
        corr_total += span_len
        sham_total += len(sham_ids)
        per_qid[qid] = {
            "session_id": entry.get("session_id") or qid.rsplit(":", 1)[0],
            "n_docs": n_docs,
            "doc_lens": doc_lens,
            "k_star": k,
            "span_len": span_len,
            "corpus_offset": offset,
            "sham_token_ids": sham_ids,
            "sham_text_sha256": _sha256_text(sham_text),
            "neutrality_violations": violations,
            # T == 1 leaves nothing downstream of k*, so E-corr+re degenerates
            # to E-corr for this qid; the analyzer reports the two cells apart.
            "no_downstream": n_docs == 1,
        }
    delta = abs(corr_total - sham_total) / max(corr_total, 1)
    plan: Dict[str, Any] = {
        "description": (
            "Task D E-sham plan. Equal-length neutral-corpus spans injected through the "
            "E-corr path at doc k* = (n_docs-1)//2. budget.typed_tokens_total keeps the "
            "r4 field name and holds the E-corr span budget (sum of len(doc k*))."
        ),
        "seed": seed,
        "rule_version": RULE_VERSION,
        "n_qids": len(per_qid),
        "budget": {
            "typed_tokens_total": corr_total,
            "sham_tokens_total": sham_total,
            "abs_delta_frac": round(delta, 6),
            "gate": "== 0",
            "gate_passed": corr_total == sham_total,
        },
        "neutrality": {
            "gate": "no structural characters and no harness vocabulary in the decoded span",
            "violating_qids": violating,
            "gate_passed": not violating,
        },
        "missing_qids": missing,
        "degenerate_qids": degenerate,
        "per_qid": per_qid,
    }
    if header:
        plan.update({k: v for k, v in header.items() if k not in plan})
    return plan


def _load_tokenizer(path: str) -> Any:
    """Lazy phase: transformers only, no torch."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _qids_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    qids = manifest.get("cw_qids")
    if qids is None:
        raise SystemExit("FATAL: manifest has no cw_qids list")
    return [str(q) for q in qids]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc_table", default="./results/d/d_doc_ids.json")
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest.json")
    parser.add_argument("--corpus", default="./configs/bdf_pilot/d_neutral_corpus.txt")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", default="./configs/bdf_pilot/d_sham_plan.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    doc_table_path = Path(args.doc_table)
    manifest_path = Path(args.manifest)
    corpus_path = Path(args.corpus)
    doc_table = json.loads(doc_table_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_text = corpus_path.read_text(encoding="utf-8")

    tokenizer = _load_tokenizer(args.tokenizer)
    corpus_ids = list(tokenizer(corpus_text, add_special_tokens=False)["input_ids"])
    logger.info("corpus: %d chars -> %d tokens", len(corpus_text), len(corpus_ids))

    qids = _qids_from_manifest(manifest)
    plan = build_plan(
        doc_table.get("per_qid", {}),
        qids,
        corpus_ids,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        seed=args.seed,
        header={
            "corpus_path": str(corpus_path.as_posix()),
            "corpus_sha256": _sha256_text(corpus_text),
            "corpus_tokens": len(corpus_ids),
            "tokenizer": args.tokenizer,
            "doc_table": str(doc_table_path.as_posix()),
            "doc_table_sha256": _sha256_file(doc_table_path),
            "qid_source": str(manifest_path.as_posix()),
            "qid_source_sha256": _sha256_file(manifest_path),
        },
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s: n_qids=%d corr=%d sham=%d budget_gate=%s neutrality_gate=%s",
        out_path,
        plan["n_qids"],
        plan["budget"]["typed_tokens_total"],
        plan["budget"]["sham_tokens_total"],
        plan["budget"]["gate_passed"],
        plan["neutrality"]["gate_passed"],
    )
    if plan["missing_qids"]:
        logger.warning("%d frozen qids absent from the doc table", len(plan["missing_qids"]))
    if not (plan["budget"]["gate_passed"] and plan["neutrality"]["gate_passed"]):
        logger.error("plan gates FAILED — do not run the sham arm with this plan")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
