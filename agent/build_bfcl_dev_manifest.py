"""Build the frozen BFCL multi-turn dev manifest for G-H200 checkpoint selection.

Deterministically samples n entry ids from a BFCL multi-turn jsonl (default
category ``multi_turn_base``) and writes a manifest JSON consumed by
``metrology/bfcl_hf_runner.py --ids_file`` (each item needs at least
``id`` + ``category``; ``n_turns`` is recorded when the entry carries
``turns``). Field conventions mirror ``configs/r5_metrology_sample.json``
where sensible; ``gold_turn_tokens`` is intentionally omitted — it requires
the tokenizer gold-length audit, and the runner never reads it.

Determinism: ids are sorted before sampling with ``random.Random(seed)``, so
the selection is independent of source-file line order; the manifest-level
``ids_sha256`` is over the sorted sampled id list.

The sampling/IO core (load_entries / sample_ids / build_manifest) is pure
stdlib and unit-testable on a tiny fake jsonl — no BFCL package needed.

用法（仓库根）:
  python agent/build_bfcl_dev_manifest.py <bfcl数据目录或multi_turn jsonl> \
      [--category multi_turn_base] [--n 128] [--seed 42] \
      [--out configs/bfcl_dev_v3_mt.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

DEFAULT_CATEGORY = "multi_turn_base"
DEFAULT_OUT = "configs/bfcl_dev_v3_mt.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    """manifest 级哈希：仅覆盖排序后的 id 列表（与文件字节无关）。"""
    return hashlib.sha256(json.dumps(sorted(ids)).encode("utf-8")).hexdigest()


def load_entries(path: Path) -> list[dict]:
    """读 BFCL jsonl（每行一个 entry，必须含 id）；保持文件行序。"""
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "id" not in entry:
                raise ValueError(f"{path}:{lineno}: entry 缺少 id 字段")
            entries.append(entry)
    if not entries:
        raise ValueError(f"{path}: 没有任何 entry")
    return entries


def sample_ids(ids: list[str], n: int, seed: int) -> list[str]:
    """确定性抽样：先排序再 random.Random(seed).sample，返回排序后的 id。"""
    if n < 1:
        raise ValueError(f"--n 必须 >= 1: {n}")
    uniq = sorted(set(ids))
    if n > len(uniq):
        raise ValueError(f"--n={n} 超过可用 id 数 {len(uniq)}")
    return sorted(random.Random(seed).sample(uniq, n))


def build_manifest(source_path: Path, entries: list[dict], category: str,
                   n: int, seed: int) -> dict:
    ids = sample_ids([e["id"] for e in entries], n, seed)
    by_id = {e["id"]: e for e in entries}
    items = []
    for i in ids:
        item = {"id": i, "category": category}
        turns = by_id[i].get("turns")
        if isinstance(turns, list):
            item["n_turns"] = len(turns)
        items.append(item)
    return {
        "task": "G-H200 BFCL multi-turn dev manifest (checkpoint selection)",
        "seed": seed,
        "n_total": len(ids),
        "category": category,
        "categories": [category],
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ids": ids,
        "ids_sha256": ids_sha256(ids),
        "items": items,
    }


def resolve_source(source: str, category: str) -> Path:
    """source 可以是 jsonl 文件，也可以是数据目录（在目录及其 data/ 子目录下
    按 BFCL_v3_<cat>.json / BFCL_v4_<cat>.json / <cat>.json 顺序解析）。"""
    p = Path(source)
    if p.is_file():
        return p
    if not p.is_dir():
        raise SystemExit(f"source 不存在: {source}")
    candidates = []
    for base in (p, p / "data"):
        for name in (f"BFCL_v3_{category}.json", f"BFCL_v4_{category}.json",
                     f"{category}.json"):
            candidates.append(base / name)
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        f"在 {source}（及其 data/ 子目录）下未找到 {category} 的 jsonl；"
        "请直接给 multi-turn jsonl 文件路径"
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python agent/build_bfcl_dev_manifest.py",
        description="BFCL multi-turn dev 冻结清单生成器（G-H200 检查点选择用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", help="BFCL 数据目录 / bfcl 包路径 / multi-turn jsonl 文件")
    p.add_argument("--category", default=DEFAULT_CATEGORY,
                   help="BFCL 类别（决定目录解析文件名与 item.category）")
    p.add_argument("--n", type=int, default=128, help="抽样条数")
    p.add_argument("--seed", type=int, default=42, help="抽样种子")
    p.add_argument("--out", default=DEFAULT_OUT, help="清单输出路径")
    args = p.parse_args(argv)

    # Windows 控制台默认 GBK：强制 UTF-8 输出（同 metrology/bfcl_hf_runner.py）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非必需能力，失败忽略
        pass

    source_path = resolve_source(args.source, args.category)
    entries = load_entries(source_path)
    try:
        manifest = build_manifest(source_path, entries, args.category,
                                  args.n, args.seed)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"[manifest] source={source_path} 总条目={len(entries)}")
    print(f"[manifest] 抽样 {manifest['n_total']} 条 (seed={args.seed}) "
          f"ids_sha256={manifest['ids_sha256'][:16]}...")
    print(f"[manifest] 写出 -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
