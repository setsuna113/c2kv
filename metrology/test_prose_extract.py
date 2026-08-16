# -*- coding: utf-8 -*-
"""prose_extract 单元测试（冻结规则 + 操作化决定 10 的边界语义）。

覆盖（任务书要求的全部用例）：
- 词典命中 / 未命中；多个词典名取位置最先者；
- 词边界：词典名 "add" 不得匹配文本里 "addition" 的子串，但独立 "add" 要命中；
- 参数窗口边界：键值对整体起始于 name_pos+1999 内计入、name_pos+2000 之外不计；
- 参数值四种形态：带引号串 / 整数 / 浮点 / true|false|null（及带引号的键）；
- 金标无参数时名字命中即 correct；
- 键交集为空 param_hit=False；
- 文本里出现非词典名不影响后续词典名命中；
- 同位置并列取名字更长者；函数名之前的键值对不计入；
- build_gold_param_keys 并集语义。
"""

from metrology.prose_extract import build_gold_param_keys, extract_semantic


def test_name_hit_with_param_hit():
    # 冻结正则的「带引号串」形态是双引号串（逐字：`"([^"]*)"`）
    text = 'estimate_distance(cityA="SF", cityB="Rivermist")'
    r = extract_semantic(text, {"estimate_distance"}, {"cityA", "cityB"})
    assert r["name_hit"] is True
    assert r["name"] == "estimate_distance"
    assert r["name_pos"] == 0
    assert r["param_keys"] == ["cityA", "cityB"]
    assert r["param_hit"] is True
    assert r["gold_no_params"] is False
    assert r["correct"] is True


def test_name_miss():
    r = extract_semantic("nothing here", {"add"}, {"x"})
    assert r["name_hit"] is False
    assert r["name"] is None
    assert r["name_pos"] is None
    assert r["param_keys"] == []
    assert r["param_hit"] is False
    assert r["gold_no_params"] is False
    assert r["correct"] is False


def test_name_miss_with_empty_gold_still_not_correct():
    # 金标无参数只能豁免参数命中，不能豁免函数名命中
    r = extract_semantic("nothing here", {"add"}, set())
    assert r["gold_no_params"] is True
    assert r["name_hit"] is False
    assert r["correct"] is False


def test_multiple_names_take_first_by_position():
    text = "bar(1) foo(2)"
    r = extract_semantic(text, {"foo", "bar"}, set())
    assert r["name"] == "bar"
    assert r["name_pos"] == 0
    assert r["correct"] is True  # 金标无参数


def test_word_boundary_no_substring_match():
    r = extract_semantic("addition(1) and more", {"add"}, set())
    assert r["name_hit"] is False
    assert r["name"] is None


def test_word_boundary_hits_standalone_name():
    text = "x addition(1) y add(2) z"
    r = extract_semantic(text, {"add"}, set())
    assert r["name_hit"] is True
    assert r["name"] == "add"
    assert r["name_pos"] == text.index("add(2)")


def test_window_boundary_pair_starting_at_1999_counted():
    # 函数名 "go" 在 0；键值对整体起始于 name_pos+1999（跨窗口右界）要计入
    # 填充用非词字符 "."，避免与函数名粘连成更长的词
    text = "go" + "." * (1999 - 2) + 'k="a"'
    assert text[1999:2004] == 'k="a"'
    r = extract_semantic(text, {"go"}, {"k"})
    assert r["name_pos"] == 0
    assert r["param_keys"] == ["k"]
    assert r["param_hit"] is True
    assert r["correct"] is True


def test_window_boundary_pair_starting_at_2000_not_counted():
    # 键值对整体起始于 name_pos+2000（窗口之外）不计入
    text = "go" + "." * (2000 - 2) + 'k="a"'
    assert text[2000:2005] == 'k="a"'
    r = extract_semantic(text, {"go"}, {"k"})
    assert r["name_pos"] == 0
    assert r["param_keys"] == []
    assert r["param_hit"] is False
    assert r["correct"] is False


def test_param_value_forms_and_quoted_key():
    # 带引号串 / 整数 / 浮点 / true / false / null，以及带引号的键
    text = 'go() a="hello" b=123 c=1.5 d=true e=false f=null "qk":2'
    r = extract_semantic(text, {"go"}, {"a", "b", "c", "d", "e", "f", "qk"})
    assert r["param_keys"] == ["a", "b", "c", "d", "e", "f", "qk"]
    assert r["param_hit"] is True
    assert r["correct"] is True


def test_gold_no_params_name_hit_is_correct():
    r = extract_semantic("go()", {"go"}, set())
    assert r["gold_no_params"] is True
    assert r["name_hit"] is True
    assert r["param_hit"] is False
    assert r["correct"] is True


def test_param_key_intersection_empty():
    text = 'go() k="v"'
    r = extract_semantic(text, {"go"}, {"z"})
    assert r["name_hit"] is True
    assert r["param_keys"] == ["k"]
    assert r["param_hit"] is False
    assert r["correct"] is False


def test_non_dict_name_does_not_block_later_dict_name():
    text = "multiply(2,3) then add(1,2)"
    r = extract_semantic(text, {"add"}, set())
    assert r["name_hit"] is True
    assert r["name"] == "add"
    assert r["name_pos"] == text.index("add(1,2)")


def test_same_position_takes_longer_name():
    # 同位置 "a" 与 "a.b" 并列：取更长者
    r = extract_semantic("a.b(1)", {"a", "a.b"}, set())
    assert r["name"] == "a.b"
    assert r["name_pos"] == 0


def test_pairs_before_name_not_counted():
    text = 'k="v" go()'
    r = extract_semantic(text, {"go"}, {"k"})
    assert r["name_hit"] is True
    assert r["name_pos"] == text.index("go")
    assert r["param_keys"] == []
    assert r["param_hit"] is False


def test_build_gold_param_keys_union():
    gold = [
        {"f1": {"x": [1], "y": [2]}},
        {"f2": {"y": [3], "z": [4]}},
    ]
    assert build_gold_param_keys(gold) == {"x", "y", "z"}


def test_build_gold_param_keys_empty():
    assert build_gold_param_keys([]) == set()
    assert build_gold_param_keys([{"f1": {}}]) == set()
