# -*- coding: utf-8 -*-
"""analyze_s8 / review_sample 单元测试（纯合成数据，纯 stdlib）。

覆盖（任务书测试节逐条）：
- M1：已知夹具重分类率（10 行 cap128 失败、3 行 cap1024 转成功 → 率 0.3）、
  标签改变另列、分叉轮（第 2 轮 parsed_text 不同 → 分叉轮 0 基 =1）。
- M2：主分母 = 360 口径（355 行缺 5 行 → missing_n=5）；C2KV 缺省 MISSING 路径。
- M3：排名翻转 / 税符号翻转夹具 → M3_HAS_CONSEQUENCE；不翻转不触发。
- 重加权：方案 (a)/(b) 权重和 = 1；方案 (b) 同输入两次运行一致。
- 抽样器：同 seed 两次输出完全一致；层覆盖正确；总体 <30 全取。
"""

import json

import pytest

from metrology import analyze_s8 as a8
from metrology import review_sample as rs


# ══════════════════════════════════════════════════════════════════════════
# 合成夹具构造器
# ══════════════════════════════════════════════════════════════════════════

def _scored_row(id_, category, condition, tier, native=False, semantic=False,
                split=False, protocol=True):
    return {
        "id": id_, "category": category, "condition": condition,
        "cap_tier": tier, "n_turns": 1, "n_steps": 1,
        "native_valid": native, "native_error_type": None,
        "protocol_valid": protocol, "n_protocol_invalid_steps": 0,
        "step_protocol_valid": [True],
        "prose": {"name_hit": True, "name": "f", "name_pos": 0,
                  "param_keys": [], "param_hit": False,
                  "gold_no_params": True, "correct": semantic},
        "semantic_correct": semantic, "split_row": split,
        "censored": False, "runner_error": False,
    }


def _sample(ids_cats_turns):
    """ids_cats_turns: {id: (category, n_turns, [每轮 token 长度...])}。"""
    items = [
        {"id": i, "category": c, "n_turns": t, "gold_turn_tokens": toks}
        for i, (c, t, toks) in sorted(ids_cats_turns.items())
    ]
    return {
        "task": "t", "seed": 20260816, "per_category": 60,
        "categories": sorted({c for _, (c, _, _) in ids_cats_turns.items()}),
        "n_total": len(items), "items": items,
    }


def _run_row(id_, cond, tier, turn_texts, category="multi_turn_base"):
    turns = [
        {"turn_index": ti, "steps": [
            {"step_index": si, "parsed_text": t}
            for si, t in enumerate(steps)
        ]}
        for ti, steps in enumerate(turn_texts)
    ]
    return {"id": id_, "category": category, "condition": cond,
            "cap_tier": tier, "turns": turns}


def _closeout_row(id_, protocol_valid, semantic_correct, primary_success):
    return {
        "id": id_,
        "scoring": {
            "protocol_valid": protocol_valid,
            "semantic_correct": semantic_correct,
            "primary_success": primary_success,
            "censored_at_cap": False,
        },
        "strata": {"clipped": False, "pool_doc_tokens": False, "is_finish": False},
    }


# ══════════════════════════════════════════════════════════════════════════
# M1
# ══════════════════════════════════════════════════════════════════════════

M1_IDS = [f"m1_{i:02d}" for i in range(10)]


def _m1_fixture(streamingllm_rec=0):
    """snapkv: 10 行 cap128 全失败，3 行 cap1024 转成功；base: 4 失败 1 转成功；
    streamingllm: 10 失败、streamingllm_rec 行转成功。"""
    scored = []
    for i in M1_IDS:
        scored.append(_scored_row(i, "multi_turn_base", "snapkv", "128"))
        scored.append(_scored_row(i, "multi_turn_base", "base", "128",
                                  native=(i not in {"m1_00", "m1_01",
                                                     "m1_02", "m1_03"})))
        scored.append(_scored_row(i, "multi_turn_base", "streamingllm", "128"))
    for i in M1_IDS[:3]:
        scored.append(_scored_row(i, "multi_turn_base", "snapkv", "1024",
                                  native=True))
    scored.append(_scored_row("m1_00", "multi_turn_base", "base", "1024",
                              native=True))
    for i in M1_IDS[:streamingllm_rec]:
        scored.append(_scored_row(i, "multi_turn_base", "streamingllm", "1024",
                                  native=True))
    return scored


def test_m1_rate_label_change_and_judgment():
    sample = _sample({i: ("multi_turn_base", 2, [10, 10]) for i in M1_IDS})
    m1 = a8.compute_m1(_m1_fixture(), sample)
    snap = m1["main"]["snapkv"]
    assert snap["n_fail_128"] == 10
    assert snap["n_rec_1024"] == 3
    assert snap["rate"] == pytest.approx(0.3)
    assert snap["n_label_change"] == 3
    assert snap["label_change_rate"] == pytest.approx(0.3)
    base = m1["main"]["base"]
    assert base["n_fail_128"] == 4
    assert base["n_rec_1024"] == 1
    assert base["rate"] == pytest.approx(0.25)
    # streamingllm 无转成功 → 两压缩条件不全 ≥10% → NOT-GENERALIZED
    assert m1["main"]["streamingllm"]["rate"] == 0.0
    assert m1["judgment"] == "NOT-GENERALIZED"
    # 类别分解（multi_turn 合并档）与主表一致
    assert m1["by_group3"]["snapkv"]["multi_turn"]["rate"] == pytest.approx(0.3)
    assert m1["by_category"]["snapkv"]["multi_turn_base"]["rate"] == pytest.approx(0.3)


def test_m1_supported_when_both_conditions_ge_10pct():
    sample = _sample({i: ("multi_turn_base", 2, [10, 10]) for i in M1_IDS})
    m1 = a8.compute_m1(_m1_fixture(streamingllm_rec=2), sample)
    assert m1["main"]["streamingllm"]["rate"] == pytest.approx(0.2)
    assert m1["judgment"] == "M1_SUPPORTED"


def test_m1_divergence_turn_second_turn_differs():
    ids = {"d0": ("multi_turn_base", 2, [10, 10]),
           "d1": ("multi_turn_base", 2, [10, 10])}
    sample = _sample(ids)
    runs = {
        ("d0", "snapkv", "128"): _run_row("d0", "snapkv", "128",
                                          [["x0"], ["x1a"]]),
        ("d0", "snapkv", "1024"): _run_row("d0", "snapkv", "1024",
                                           [["x0"], ["x1b"]]),
        ("d1", "snapkv", "128"): _run_row("d1", "snapkv", "128",
                                          [["y0"], ["y1"]]),
        ("d1", "snapkv", "1024"): _run_row("d1", "snapkv", "1024",
                                           [["y0"], ["y1"]]),
    }
    m1 = a8.compute_m1([], sample, runs_rows=runs)
    div = m1["first_divergence_turn"]
    assert div["status"] == "AVAILABLE"
    pc = div["per_condition"]["snapkv"]
    assert pc["n_comparable"] == 2
    assert pc["n_identical"] == 1
    assert pc["n_diverged"] == 1
    assert pc["histogram"] == {1: 1}
    assert pc["turn_index_min"] == 1
    assert pc["turn_index_median"] == 1.0
    assert pc["turn_index_max"] == 1
    # 其余条件无可比行
    assert div["per_condition"]["base"]["n_comparable"] == 0


def test_m1_divergence_not_available_without_runs_dir():
    sample = _sample({"d0": ("multi_turn_base", 2, [10, 10])})
    m1 = a8.compute_m1([], sample, runs_rows=None)
    assert m1["first_divergence_turn"]["status"] == "NOT-AVAILABLE"


def test_m1_divergence_not_applicable_for_single_turn_only():
    sample = _sample({"p0": ("parallel", 1, [50])})
    m1 = a8.compute_m1([], sample, runs_rows={})
    assert m1["first_divergence_turn"]["status"] == "NOT-APPLICABLE"


# ══════════════════════════════════════════════════════════════════════════
# M2
# ══════════════════════════════════════════════════════════════════════════

def test_m2_missing_rows_keep_360_denominator():
    ids = {f"i{i:03d}": ("parallel", 1, [50]) for i in range(360)}
    sample = _sample(ids)
    scored = []
    # base/128 只有 355 行 → missing_n=5，主分母仍 360
    for i in list(ids)[:355]:
        scored.append(_scored_row(i, "parallel", "base", "128", split=(int(i[1:]) % 10 == 0)))
    # snapkv/default 360 行，18 行分裂 → 18/360 = 0.05 → M2_SUPPORTED
    for i in ids:
        scored.append(_scored_row(i, "parallel", "snapkv", "default",
                                  semantic=True, split=(int(i[1:]) < 18),
                                  protocol=(int(i[1:]) >= 18)))
    m2 = a8.compute_m2(scored, sample)
    cell = m2["cells"]["base"]["128"]
    assert cell["n_scored"] == 355
    assert cell["missing_n"] == 5
    assert cell["split_rate_main"] == pytest.approx(cell["split_n"] / 360.0)
    snap_default = m2["cells"]["snapkv"]["default"]
    assert snap_default["split_n"] == 18
    assert snap_default["split_rate_main"] == pytest.approx(0.05)
    assert m2["judgment"] == "M2_SUPPORTED"


def test_m2_not_supported_below_5pct_and_c2kv_missing():
    ids = {f"i{i:03d}": ("parallel", 1, [50]) for i in range(360)}
    sample = _sample(ids)
    scored = []
    for i in ids:
        scored.append(_scored_row(i, "parallel", "snapkv", "default",
                                  split=(int(i[1:]) < 5)))
    m2 = a8.compute_m2(scored, sample)
    assert m2["judgment"] == "NOT-SUPPORTED"
    # C2KV 文件缺省 → MISSING 路径（任务书验收要求走过的路径）
    assert m2["c2kv"]["status"] == "MISSING"


def test_m2_c2kv_columns_from_closeouts():
    ids = {f"i{i:03d}": ("parallel", 1, [50]) for i in range(360)}
    sample = _sample(ids)
    full = [_closeout_row(f"i{i:03d}", protocol_valid=(i % 2 == 0),
                          semantic_correct=True, primary_success=True)
            for i in range(89)]
    c2kv = [_closeout_row(f"i{i:03d}", protocol_valid=True,
                          semantic_correct=(i % 3 == 0), primary_success=(i % 3 == 0))
            for i in range(89)]
    m2 = a8.compute_m2([], sample, closeout_full=full, closeout_c2kv=c2kv)
    assert m2["c2kv"]["status"] == "AVAILABLE"
    assert m2["c2kv"]["cap"] == 256
    assert m2["c2kv"]["full"]["n_rows"] == 89
    # full 臂：semantic 全 True、protocol 偶数行 True → split = 奇数行 → 44
    assert m2["c2kv"]["full"]["split_n"] == 44
    assert m2["c2kv"]["full"]["rate_split_89"] == pytest.approx(44 / 89.0)
    assert m2["c2kv"]["c2kv"]["split_n"] == 0


# ══════════════════════════════════════════════════════════════════════════
# M3
# ══════════════════════════════════════════════════════════════════════════

M3_IDS = [f"m3_{i:02d}" for i in range(10)]


def _m3_sample():
    return _sample({i: ("parallel", 1, [50]) for i in M3_IDS})


def _m3_scored(valid_native: dict, valid_semantic: dict):
    """valid_native/valid_semantic: {条件: 通过 id 集合}（default 用 native，
    1024 用 semantic）。"""
    scored = []
    for cond in ["base", "snapkv", "streamingllm"]:
        for i in M3_IDS:
            scored.append(_scored_row(i, "parallel", cond, "default",
                                      native=(i in valid_native[cond])))
            scored.append(_scored_row(i, "parallel", cond, "1024",
                                      semantic=(i in valid_semantic[cond])))
    return scored


def test_m3_rank_flip_and_tax_sign_flip_trigger_judgment():
    sample = _m3_sample()
    valid_native = {"base": set(M3_IDS[:9]), "snapkv": set(M3_IDS[:8]),
                    "streamingllm": set(M3_IDS[:7])}
    valid_semantic = {"base": set(M3_IDS[:9]), "snapkv": set(M3_IDS),
                      "streamingllm": set(M3_IDS[:7])}
    m3 = a8.compute_m3(_m3_scored(valid_native, valid_semantic), sample)
    # cap_c：金标每轮 P95=50 → max(1024, 50) = 1024
    assert m3["cap_c_by_category"]["parallel"]["cap_c"] == 1024
    assert m3["cap_c_by_category"]["parallel"]["tier"] == "1024"
    # 基线排名 base > snapkv > streamingllm；修正后 snapkv 第一 → 排名对换
    assert m3["baseline"]["ranking"] == ["base", "snapkv", "streamingllm"]
    assert m3["corrected"]["ranking"] == ["snapkv", "base", "streamingllm"]
    # 基线税 snapkv +0.1；修正税 −0.1 → 符号翻转
    assert m3["baseline"]["taxes"]["snapkv"]["sign"] == "+"
    assert m3["corrected"]["taxes"]["snapkv"]["sign"] == "-"
    assert m3["judgment"] == "M3_HAS_CONSEQUENCE"
    assert m3["judgment_detail"]["rank_swap"] is True
    assert "snapkv" in m3["judgment_detail"]["tax_sign_flips"]


def test_m3_no_flip_does_not_trigger():
    sample = _m3_sample()
    valid_native = {"base": set(M3_IDS[:9]), "snapkv": set(M3_IDS[:8]),
                    "streamingllm": set(M3_IDS[:7])}
    valid_semantic = {"base": set(M3_IDS), "snapkv": set(M3_IDS[:9]),
                      "streamingllm": set(M3_IDS[:8])}
    m3 = a8.compute_m3(_m3_scored(valid_native, valid_semantic), sample)
    assert m3["baseline"]["ranking"] == m3["corrected"]["ranking"]
    assert m3["baseline"]["taxes"]["snapkv"]["sign"] == "+"
    assert m3["corrected"]["taxes"]["snapkv"]["sign"] == "+"
    assert m3["baseline"]["taxes"]["streamingllm"]["sign"] == "+"
    assert m3["corrected"]["taxes"]["streamingllm"]["sign"] == "+"
    assert m3["judgment"] == "M3_NO_CONSEQUENCE"


def test_m3_c2kv_descriptive_paired_rows():
    sample = _m3_sample()
    full = [_closeout_row(f"m3_{i:02d}", protocol_valid=True,
                          semantic_correct=(i < 8), primary_success=(i < 9))
            for i in range(10)]
    c2kv = [_closeout_row(f"m3_{i:02d}", protocol_valid=True,
                          semantic_correct=(i < 6), primary_success=(i < 7))
            for i in range(10)]
    m3 = a8.compute_m3([], sample, closeout_full=full, closeout_c2kv=c2kv)
    c2 = m3["c2kv"]
    assert c2["status"] == "AVAILABLE"
    assert c2["n_pairs"] == 10
    assert c2["full"]["acc_protocol"] == pytest.approx(0.9)
    assert c2["c2kv"]["acc_semantic"] == pytest.approx(0.6)
    assert c2["tax_full_minus_c2kv"]["protocol"]["value"] == pytest.approx(0.2)
    assert c2["tax_full_minus_c2kv"]["protocol"]["sign"] == "+"


def test_m3_c2kv_missing_path():
    sample = _m3_sample()
    m3 = a8.compute_m3([], sample)
    assert m3["c2kv"]["status"] == "MISSING"


# ══════════════════════════════════════════════════════════════════════════
# 重加权（构成敏感性）
# ══════════════════════════════════════════════════════════════════════════

def _weight_sample():
    ids = {}
    for k in range(4):
        ids[f"a{k}"] = ("parallel", 1 + k, [50, 50])          # bin 1-2 / 3-4
    for k in range(4):
        ids[f"b{k}"] = ("multi_turn_base", 3 + k, [10, 10])   # bin 3-4 / 5+
    return _sample(ids)


def _v1_obj():
    """真实文件格式：8 个 dict 的列表，clipped/finish_target 为 bool，
    pool_doc_tokens 为整数（75327/80171），n 合计 395。"""
    cells = [
        {"clipped": False, "pool_doc_tokens": 75327, "finish_target": False,
         "n": 24},
        {"clipped": False, "pool_doc_tokens": 75327, "finish_target": True,
         "n": 0},
        {"clipped": False, "pool_doc_tokens": 80171, "finish_target": False,
         "n": 122},
        {"clipped": False, "pool_doc_tokens": 80171, "finish_target": True,
         "n": 1},
        {"clipped": True, "pool_doc_tokens": 75327, "finish_target": False,
         "n": 13},
        {"clipped": True, "pool_doc_tokens": 75327, "finish_target": True,
         "n": 4},
        {"clipped": True, "pool_doc_tokens": 80171, "finish_target": False,
         "n": 157},
        {"clipped": True, "pool_doc_tokens": 80171, "finish_target": True,
         "n": 74},
    ]
    return {"cells_clipped_x_pool_x_finish": cells}


def test_reweight_scheme_a_weights_sum_to_one():
    sample = _weight_sample()
    m3 = a8.compute_m3([], sample)
    comp = m3["composition"]
    wa = comp["scheme_a"]["weights"]
    assert comp["n_strata"] == 4
    assert sum(wa.values()) == pytest.approx(1.0)
    for w in wa.values():
        assert w == pytest.approx(0.25)


def test_reweight_scheme_b_weights_and_determinism():
    sample = _weight_sample()
    v1 = _v1_obj()
    m3a = a8.compute_m3([], sample, v1_obj=v1)
    m3b = a8.compute_m3([], sample, v1_obj=v1)
    wb = m3a["composition"]["scheme_b"]["weights"]
    assert sum(wb.values()) == pytest.approx(1.0)
    # 同输入两次运行完全一致（映射确定）
    assert wb == m3b["composition"]["scheme_b"]["weights"]
    # 4 层 → floor(j*8/4) = 0,2,4,6 → 排序后第 0/2/4/6 格
    cells = m3a["composition"]["scheme_b"]["cells"]
    assert len(cells) == 8
    assert m3a["composition"]["scheme_b"]["cells_total_n"] == pytest.approx(395)
    expect = {
        (False, 75327, False): 24 / 395.0,
        (False, 80171, False): 122 / 395.0,
        (True, 75327, False): 13 / 395.0,
        (True, 80171, False): 157 / 395.0,
    }
    norm = sum(expect.values())
    strata_keys = sorted(m3a["composition"]["strata"], key=lambda d: (d["category"], d["turn_bin"]))
    keys = [f"{d['category']}|{d['turn_bin']}" for d in strata_keys]
    assert keys == ["multi_turn_base|3-4", "multi_turn_base|5+",
                    "parallel|1-2", "parallel|3-4"]
    for key, cell in zip(keys, expect):
        assert wb[key] == pytest.approx(expect[cell] / norm)


def test_parse_v1_cells_real_format_8_cells_sorted_and_weights():
    """真实文件格式夹具：pool_doc_tokens 整数字段，断言解析出 8 格、排序首格为
    (False, 75327, False)、scheme_b 权重和 = 1、映射确定。"""
    v1 = _v1_obj()
    cells = a8.parse_v1_cells(v1)
    assert cells is not None
    assert len(cells) == 8
    assert cells[0][0] == (False, 75327, False)
    assert cells[-1][0] == (True, 80171, True)
    assert sum(n for _, n in cells) == pytest.approx(395)
    sample = _weight_sample()
    m3a = a8.compute_m3([], sample, v1_obj=v1)
    m3b = a8.compute_m3([], sample, v1_obj=v1)
    sb = m3a["composition"]["scheme_b"]
    assert "weights" in sb  # 成功路径无 status 键（MISSING 路径才有）
    assert sum(sb["weights"].values()) == pytest.approx(1.0)
    assert sb["weights"] == m3b["composition"]["scheme_b"]["weights"]
    # pool 轴保留数值：cells 输出里 pool_doc_tokens 为 int 而非 bool
    assert all(isinstance(c["pool_doc_tokens"], int) for c in sb["cells"])


def test_reweight_scheme_b_missing_when_fewer_than_8_cells():
    """不足 8 格 → scheme_b 落 MISSING 且 MISSING_DETAIL 写明原因，不抛 IndexError。"""
    v1 = {"cells_clipped_x_pool_x_finish": _v1_obj()["cells_clipped_x_pool_x_finish"][:7]}
    sample = _weight_sample()
    m3 = a8.compute_m3([], sample, v1_obj=v1)
    sb = m3["composition"]["scheme_b"]
    assert sb["status"] == "MISSING"
    assert "7" in sb["MISSING_DETAIL"]
    assert sb["MISSING_DETAIL"] == sb["reason"]


def test_reweight_scheme_b_missing_without_v1():
    sample = _weight_sample()
    m3 = a8.compute_m3([], sample)
    assert m3["composition"]["scheme_b"]["status"] == "MISSING"


# ══════════════════════════════════════════════════════════════════════════
# 抽样器
# ══════════════════════════════════════════════════════════════════════════

def _split_rows(stratum_sizes: dict):
    rows = []
    for (cond, tier), n in stratum_sizes.items():
        for k in range(n):
            rows.append(_scored_row(f"{cond}_{tier}_{k}", "parallel", cond, tier,
                                    semantic=True, split=True, protocol=False))
    return rows


def test_sampler_same_seed_identical_and_strata_covered():
    rows = _split_rows({("base", "128"): 20, ("snapkv", "128"): 12,
                        ("streamingllm", "128"): 8})
    p1 = rs.select_cases(rows)
    p2 = rs.select_cases(rows)
    assert json.dumps(p1) == json.dumps(p2)
    assert p1["n_selected"] == 30
    assert p1["population_n"] == 40
    # 层规模 20/12/8、n=30 → 配额恰为 15/9/6（覆盖正确）
    assert p1["allocation"] == {"base|128": 15, "snapkv|128": 9,
                                "streamingllm|128": 6}
    assert len(p1["cases"]) == 30
    assert len({c["case_no"] for c in p1["cases"]}) == 30
    assert {c["condition"] for c in p1["cases"]} == {"base", "snapkv",
                                                     "streamingllm"}


def test_sampler_population_below_30_takes_all():
    rows = _split_rows({("base", "128"): 3, ("snapkv", "128"): 2})
    p = rs.select_cases(rows)
    assert p["n_selected"] == 5
    assert p["population_n"] == 5
    assert p["selection_rule"].startswith("总体")
    assert {c["id"] for c in p["cases"]} == {
        "base_128_0", "base_128_1", "base_128_2", "snapkv_128_0", "snapkv_128_1"}


def test_largest_remainder_deterministic_tiebreak():
    sizes = {"A": 10, "B": 10, "C": 9}
    alloc = rs.allocate_largest_remainder(sizes, 5)
    assert alloc == {"A": 2, "B": 2, "C": 1}


def test_sampler_more_strata_than_n_takes_top_strata():
    rows = _split_rows({(f"c{k}", "128"): 1 for k in range(35)})
    p = rs.select_cases(rows)
    assert p["n_selected"] == 30
    assert p["n_strata"] == 35
    assert all(v == 1 for v in p["allocation"].values())
    assert len(p["allocation"]) == 30


def test_sampler_text_and_gold_rebuild_and_missing_paths():
    rows = []
    for k in range(3):
        rows.append(_scored_row(f"mt_{k}", "multi_turn_base", "base", "128",
                                semantic=True, split=True, protocol=False))
    runs = {
        ("mt_0", "base", "128"): _run_row("mt_0", "base", "128",
                                          [["a"], ["b"]]),
    }
    gold = {"mt_1": [[{"func": "f1", "param_keys": ["p"]}]]}
    p = rs.select_cases(rows, runs_rows=runs, gold_by_id=gold)
    by_id = {c["id"]: c for c in p["cases"]}
    assert by_id["mt_0"]["text"] == "a\nb"
    assert by_id["mt_1"]["text"] == "MISSING-TEXT"
    assert by_id["mt_1"]["gold_calls"] == [[{"func": "f1", "param_keys": ["p"]}]]
    assert by_id["mt_2"]["gold_calls"] == "MISSING-GOLD"


def test_sampler_compare_mode_counts_disagreements():
    rows = _split_rows({("base", "128"): 3, ("snapkv", "128"): 2})
    p = rs.select_cases(rows)
    verdicts = {1: "disagree", 2: "agree", 3: "disagree", 4: "agree", 5: "agree",
                99: "disagree"}
    r = rs.compare_verdicts(p, verdicts)
    assert r["n_disagree"] == 2
    assert r["n_agree"] == 3
    assert r["n_unmatched"] == 1
    assert [d["case_no"] for d in r["disagree_details"]] == [1, 3]


def test_compare_rejects_invalid_verdict():
    rows = _split_rows({("base", "128"): 3})
    p = rs.select_cases(rows)
    with pytest.raises(SystemExit):
        rs.compare_verdicts(p, {1: "maybe"})


def test_build_text_single_turn():
    row = _run_row("p0", "base", "128", [["only_text"]], category="parallel")
    assert rs.build_text(row) == "only_text"
