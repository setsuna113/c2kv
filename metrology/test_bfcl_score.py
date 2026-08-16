# -*- coding: utf-8 -*-
"""bfcl_score 集成测试：跑完整 CLI（--runs_dir 用 metrology/data/smoke_fixtures）。

fixture 事实核对（任务书测试节的前提与实际数据不符，口径一律按冻结决策执行，
此处如实断言并随报告披露）：

1. smoke_identity_* 两文件行内 condition 与 smoke_snapkv/streamingllm 相同，
   (id, cap_tier, condition) 重复——按 CLI 规格「重复即报错」，先断言该报错路径
   （test_cli_dedup_guard_on_smoke_fixtures）。
2. 冒烟行并非「全解码成功」：base/snapkv 行各有 4 个 step 的 decoded_calls 为
   空列表（空响应步，按冻结决策 2 协议无效）；streamingllm 行 4 个 step 的
   decoded_calls 均为异常类名字符串（协议无效）。故「protocol_valid 全 True、
   n_protocol_invalid_steps 全 0」的前提不成立，测试断言相应放宽（操作化修订）：
   不要求 protocol_valid 全 True、也不与行内存储重算值精确比对，只断言
   step_protocol_valid 长度 == n_steps 且类型正确（以及输出内部的一致性）。
3. 完整评分在 3 个唯一键文件（base / snapkv / streamingllm）的临时拷贝上跑。

native_valid 的具体取值不断言（原生评分由 eval_runner file runner 给出，
结果待审核）；逐行打印 native_valid / native_error_type 供人工复核。
"""

import json
import shutil
from pathlib import Path

import pytest

from metrology import bfcl_score

REPO_ROOT = Path(__file__).resolve().parents[1]
BFCL_PKG = REPO_ROOT / ".foreman" / "ref" / "bfcl_pkg"
BFCL_DATA = REPO_ROOT / ".foreman" / "ref" / "bfcl_data"
SMOKE_DIR = REPO_ROOT / "metrology" / "data" / "smoke_fixtures"

ROW_KEYS = [
    "id", "category", "condition", "cap_tier", "n_turns", "n_steps",
    "native_valid", "native_error_type", "protocol_valid",
    "n_protocol_invalid_steps", "step_protocol_valid", "prose",
    "semantic_correct", "split_row", "censored", "runner_error",
]
PROSE_KEYS = [
    "name_hit", "name", "name_pos", "param_keys", "param_hit",
    "gold_no_params", "correct",
]
UNIQUE_FILES = [
    "smoke_base_128.jsonl",
    "smoke_snapkv_128.jsonl",
    "smoke_streamingllm_128.jsonl",
]


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().strip().split("\n") if line.strip()]


def _is_empty_call_list(calls: list) -> bool:
    # 冻结决策 2 的空响应判定（与 bfcl_eval is_empty_execute_response 同义）
    return len(calls) == 0 or (len(calls) == 1 and len(calls[0]) == 0)


def _cli_kwargs(runs_dir: Path, out: Path, summary_out: Path | None = None) -> list:
    args = [
        "--bfcl_pkg_path", str(BFCL_PKG),
        "--bfcl_data_dir", str(BFCL_DATA),
        "--runs_dir", str(runs_dir),
        "--out", str(out),
    ]
    if summary_out is not None:
        args += ["--summary_out", str(summary_out)]
    return args


def test_cli_dedup_guard_on_smoke_fixtures(tmp_path):
    # 5 个 fixture 文件里 identity_* 与同名条件文件的行键 (id, cap_tier, condition)
    # 重复：按 CLI 规格必须报错、不得静默去重、不得产生输出。
    out = tmp_path / "scored.jsonl"
    with pytest.raises(SystemExit) as exc:
        bfcl_score.main(_cli_kwargs(SMOKE_DIR, out))
    msg = str(exc.value)
    assert "multi_turn_base_61" in msg
    assert "snapkv" in msg and "streamingllm" in msg
    assert not out.exists()


def test_cli_scoring_on_unique_cells(tmp_path):
    # 3 个唯一键文件（base / snapkv / streamingllm）的完整 CLI 评分
    staged = tmp_path / "runs"
    staged.mkdir()
    for name in UNIQUE_FILES:
        shutil.copyfile(SMOKE_DIR / name, staged / name)

    out = tmp_path / "scored.jsonl"
    summary_out = tmp_path / "summary.json"
    bfcl_score.main(_cli_kwargs(staged, out, summary_out))

    scored = _read_rows(out)
    assert len(scored) == 3
    assert out.exists() and summary_out.exists()

    # 源行（按 (id, cap_tier, condition) 建索引）供 fixture 事实断言用
    src = {}
    for name in UNIQUE_FILES:
        row = _read_rows(staged / name)[0]
        src[(row["id"], row["cap_tier"], row["condition"])] = row

    for r in scored:
        key = (r["id"], r["cap_tier"], r["condition"])
        assert key in src
        assert list(r.keys()) == ROW_KEYS
        assert isinstance(r["id"], str) and isinstance(r["category"], str)
        assert isinstance(r["condition"], str) and isinstance(r["cap_tier"], str)
        assert isinstance(r["n_turns"], int) and r["n_turns"] == 4
        assert isinstance(r["n_steps"], int)
        assert isinstance(r["native_valid"], bool)
        assert r["native_error_type"] is None or isinstance(r["native_error_type"], str)
        assert isinstance(r["protocol_valid"], bool)
        assert isinstance(r["n_protocol_invalid_steps"], int)
        assert isinstance(r["step_protocol_valid"], list)
        assert all(isinstance(x, bool) for x in r["step_protocol_valid"])
        assert len(r["step_protocol_valid"]) == r["n_steps"]
        assert r["n_protocol_invalid_steps"] == sum(
            1 for x in r["step_protocol_valid"] if not x
        )
        prose = r["prose"]
        assert list(prose.keys()) == PROSE_KEYS
        assert isinstance(prose["name_hit"], bool)
        assert prose["name"] is None or isinstance(prose["name"], str)
        assert prose["name_pos"] is None or isinstance(prose["name_pos"], int)
        assert isinstance(prose["param_keys"], list)
        assert all(isinstance(k, str) for k in prose["param_keys"])
        assert isinstance(prose["param_hit"], bool)
        assert isinstance(prose["gold_no_params"], bool)
        assert isinstance(prose["correct"], bool)
        assert isinstance(r["semantic_correct"], bool)
        assert isinstance(r["split_row"], bool)
        assert isinstance(r["censored"], bool) and isinstance(r["runner_error"], bool)
        assert r["runner_error"] is False
        # 派生关系（决策 5）与输出内部一致性
        assert r["semantic_correct"] == (r["native_valid"] or prose["correct"])
        assert r["split_row"] == (r["semantic_correct"] and not r["protocol_valid"])
        assert r["protocol_valid"] == all(r["step_protocol_valid"])
        # 放宽断言（操作化修订）：不要求 protocol_valid 全 True、也不与行内存储
        # 重算值精确比对；只要求 step_protocol_valid 长度 == n_steps 且类型正确。
        assert len(r["step_protocol_valid"]) == r["n_steps"]
        assert all(isinstance(x, bool) for x in r["step_protocol_valid"])

    conditions = sorted(r["condition"] for r in scored)
    assert conditions == ["base", "snapkv", "streamingllm"]
    for r in scored:
        assert (r["id"], r["category"], r["cap_tier"]) == (
            "multi_turn_base_61", "multi_turn_base", "128",
        )

    # fixture 事实：base / snapkv 行 decoded_calls 全为列表（含 4 个空响应步）；
    # streamingllm 行 4 个 step 全为异常类名字符串 → 协议面全无效。
    for cond in ("base", "snapkv"):
        row = src[("multi_turn_base_61", "128", cond)]
        dcs = [s["decoded_calls"] for t in row["turns"] for s in t["steps"]]
        assert all(isinstance(x, list) for x in dcs)
        assert sum(1 for x in dcs if _is_empty_call_list(x)) == 4
    row = src[("multi_turn_base_61", "128", "streamingllm")]
    dcs = [s["decoded_calls"] for t in row["turns"] for s in t["steps"]]
    assert all(isinstance(x, str) for x in dcs) and len(dcs) == 4

    # summary_out：3 个 condition × cap_tier=128 各一格
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert set(summary.keys()) == {"base", "snapkv", "streamingllm"}
    for cond, cells in summary.items():
        assert set(cells.keys()) == {"128"}
        cell = cells["128"]
        assert cell["n"] == 1
        assert set(cell.keys()) == {
            "n", "native_valid_n", "protocol_invalid_n", "split_n", "censored_n",
        }

    # 打印评分摘要供人工审核（不断言 native_valid 取值）
    for r in scored:
        print(
            f"[scored] condition={r['condition']:14s} cap_tier={r['cap_tier']} "
            f"native_valid={r['native_valid']} "
            f"native_error_type={r['native_error_type']} "
            f"prose.name_hit={r['prose']['name_hit']} "
            f"prose.correct={r['prose']['correct']} "
            f"protocol_valid={r['protocol_valid']} "
            f"n_protocol_invalid_steps={r['n_protocol_invalid_steps']} "
            f"split_row={r['split_row']} censored={r['censored']}"
        )
