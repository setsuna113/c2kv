"""R3 T-A: minimal raw-text client for a plain sglang server.

Sends the exact S1 full-arm input ids (from r3_extract_prompts.py) to
/generate and writes the RAW generated text to disk — no chat template, no
tool-call parser, no harness. Stdlib only (urllib), no dependencies.

Per qid artifacts in --out_dir:
  raw_<qid>.json   — full HTTP response (incl. meta_info)
  gen_<qid>.txt    — decoded generated text, complete
Summary rows appended to t_a_generations.jsonl:
  {qid, path, kernel, n_prompt_tokens, output_chars, head500, has_tool_call,
   finish_reason, error}

Usage:
  python agent/r3_sglang_rawtext.py --prompts_file <dir>/t_a_prompts.jsonl \
      --out_dir <dir> --base_url http://127.0.0.1:30000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("r3_sglang_rawtext")


def _post_generate(base_url: str, input_ids: list, max_new_tokens: int, timeout: int) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": True,
        },
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_url", default="http://127.0.0.1:30000")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--path_label", default="sglang-ascend")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "t_a_generations.jsonl"

    rows = [json.loads(l) for l in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        qid = row["qid"]
        rec = {
            "qid": qid,
            "path": args.path_label,
            "kernel": "sglang ascend backend (non npu_fusion_attention)",
            "n_prompt_tokens": row["n_tokens"],
        }
        start = time.perf_counter()
        try:
            resp = _post_generate(args.base_url, row["input_ids"], args.max_new_tokens, args.timeout)
            (out_dir / f"raw_{qid.replace(':', '_')}.json").write_text(
                json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            text = resp.get("text", "")
            meta = resp.get("meta_info", {}) or {}
            (out_dir / f"gen_{qid.replace(':', '_')}.txt").write_text(text, encoding="utf-8")
            rec.update(
                output_chars=len(text),
                head500=text[:500],
                has_tool_call=("<tool_call>" in text or "Action:" in text),
                finish_reason=str(meta.get("finish_reason")),
                completion_tokens=meta.get("completion_tokens"),
                wall_sec=round(time.perf_counter() - start, 2),
                error=None,
            )
            logger.info("qid=%s chars=%d finish=%s wall=%.1fs", qid, len(text), rec["finish_reason"], rec["wall_sec"])
        except Exception as exc:  # record and continue; verdict needs all 4 rows
            rec.update(error=f"{type(exc).__name__}: {exc}", wall_sec=round(time.perf_counter() - start, 2))
            logger.error("qid=%s FAILED: %s", qid, rec["error"])
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Done -> %s", summary_path)


if __name__ == "__main__":
    main()
