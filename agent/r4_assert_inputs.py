"""R4 A1: machine assertion of the frozen 48-qid inputs (zero-GPU).

R3 only hand-checked the frozen input_ids; this script is the code assertion:
  1. sha256 of the r2 source archive (outputs_lyc/r2_bigpool/s1_full_48.jsonl)
     == configs/r3_s1_48_qids.json.source_sha256;
  2. rebuild every qid's input_ids via the r3_extract_prompts assembly
     (system + tool doc + prompt, S1_DATA_KW regime) and assert
     n_tokens == per_qid[qid].input_tokens from the r3 config;
  3. assert the rebuilt ids are token-for-token identical to the archived
     frozen prompts (t_e/full_trusted/t_a_prompts.jsonl);
  4. per-qid input_ids sha256 digest for binding;
  5. weight-file sha256 manifest proving the same-weights claim:
     checkpoint-250 (full arm AND c2kv arm) + base model (T-A trusted arm).

Writes a JSON report; exits nonzero on the first failed assertion.

Usage (NPU server, repo root of c2kv-r4):
  python agent/r4_assert_inputs.py --out results/r4/input_assertions.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from r3_bigpool_rerun import S1_DATA_KW, _load_frozen_qids  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)

logger = logging.getLogger("r4_assert_inputs")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_ids(ids: List[int]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(ids).encode("ascii"))
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid_file", default="./configs/r3_s1_48_qids.json")
    p.add_argument("--source_archive", default="./outputs_lyc/r2_bigpool/s1_full_48.jsonl")
    p.add_argument("--frozen_prompts", default="./outputs_lyc/r3_discrimination/t_e/full_trusted/t_a_prompts.jsonl")
    p.add_argument("--tokenizer", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--checkpoint250", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--base_model", default="./models/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default="./results/r4/input_assertions.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = json.loads(Path(args.qid_file).read_text(encoding="utf-8"))
    qids: List[str] = cfg["qids"]
    per_qid_cfg = cfg["per_qid"]

    # 1. source archive sha256
    src_sha = _sha256_file(Path(args.source_archive))
    assert src_sha == cfg["source_sha256"], (
        f"source archive sha256 mismatch: {src_sha} != {cfg['source_sha256']}"
    )
    logger.info("[1] source archive sha256 OK (%s)", src_sha[:16])

    # 2+3. rebuild and compare against the archived frozen prompts
    archived: Dict[str, Any] = {}
    with Path(args.frozen_prompts).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            archived[row["qid"]] = row
    assert set(archived) == set(qids), "archived prompts do not cover the frozen 48"

    tokenizer = H.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)
    wanted = set(qids)
    by_qid: Dict[str, Any] = {}
    for example in source.iter_examples("eval"):
        if example.qid in wanted:
            by_qid[example.qid] = example
    missing = [q for q in qids if q not in by_qid]
    assert not missing, f"qids not reproduced: {missing}"

    per_qid_report: Dict[str, Any] = {}
    for qid in qids:
        example = by_qid[qid]
        system_ids = H._chat_template_ids(
            tokenizer, [{"role": "system", "content": example.system_prompt}],
            keep_bos=True, max_length=256,
        )
        doc_ids = H._tool_doc_ids(tokenizer, example.tool_definition)
        prompt_ids = H._chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
        if len(prompt_ids) > 1920:
            prompt_ids = prompt_ids[-1920:]
        input_ids = list(system_ids) + list(doc_ids) + list(prompt_ids)
        assert len(input_ids) == per_qid_cfg[qid]["input_tokens"], (
            f"{qid}: rebuilt n_tokens {len(input_ids)} != config {per_qid_cfg[qid]['input_tokens']}"
        )
        arc = archived[qid]
        assert len(input_ids) == arc["n_tokens"], (
            f"{qid}: rebuilt n_tokens {len(input_ids)} != archived {arc['n_tokens']}"
        )
        assert list(arc["input_ids"]) == list(input_ids), f"{qid}: input_ids differ from archive"
        per_qid_report[qid] = {
            "n_tokens": len(input_ids),
            "input_ids_sha256": _sha256_ids(input_ids),
            "match": True,
        }
    logger.info("[2+3] 48/48 rebuilt ids match config n_tokens AND archived ids")

    # 5. weight-file manifest
    weights: Dict[str, Any] = {}
    for label, root in (("checkpoint250", args.checkpoint250), ("base_model", args.base_model)):
        root_p = Path(root)
        files = sorted(
            p for p in root_p.iterdir()
            if p.suffix in {".safetensors", ".bin", ".json", ".jinja"} or p.name in {"merges.txt", "vocab.json"}
        )
        weights[label] = {
            "path": str(root_p.resolve()),
            "files": {p.name: _sha256_file(p) for p in files},
        }
        logger.info("[5] %s: %d files hashed", label, len(files))

    report = {
        "qid_file": args.qid_file,
        "source_archive": {"path": args.source_archive, "sha256": src_sha},
        "frozen_prompts_archive": args.frozen_prompts,
        "n_qids": len(qids),
        "all_assertions_passed": True,
        "per_qid": per_qid_report,
        "weights": weights,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("ALL ASSERTIONS PASSED -> %s", out)


if __name__ == "__main__":
    main()
