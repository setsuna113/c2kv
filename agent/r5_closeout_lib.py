"""R5 S6.1 共享库：严格协议解析 / 语义评分 / tool_call 埋点定位 / 违规分解。

纯 CPU（stdlib + 复用 r5_reanalysis / r5_finish_semantics，不 import torch），
供 S7 closeout 分析与 S8 引用。

==== SCHEMA 冻结说明（S8 将引用本文件的返回键集合，勿再改动键名/取值口径） ====

1. strict_protocol_parse(text) 返回键：
   {predicate_hit, action_only, parse_ok, name, arguments_keys, closed_block_found, protocol_valid}
   - 严格主口径与 agent/r5_reanalysis.py strict_protocol_valid 逐字一致：
     谓词命中 AND 存在闭合 <tool_call> 块 AND 块内 JSON 合法 AND 含 name 键
     （arguments 允许 {} 或缺失）。仅命中 Action: 且无 <tool_call> 标记 → action_only=True、
     protocol_valid=False。
   - arguments_keys：首个有效闭合块的 arguments 键列表（sorted）；arguments 缺失或 {}
     为 []；arguments 存在但不可解释为 dict（str 二次 json.loads 失败 / 非 dict 非 str 类型）
     为 None（无法判定）。protocol_valid 不含 arguments 合法性要求（口径同 r5_prereg §1.1）。

2. semantic_score(text, target_row) 返回键：
   {semantic_correct, line, token_f1, rouge_l_f1, name_em, answer_missing, answer_status, answer_detail}
   - line='finish_semantic'：finish 目标（target_row["target_tool_name"]=='finish'，
     口径同 agent/r5_reanalysis.py FINISH_TOOL_NAME）走金标 answer 参数串 vs 生成文本的
     token-F1 与 ROUGE-L F1（复用 agent/r5_finish_semantics.py），判对线 ROUGE-L F1>=0.5
     （未舍入浮点判定）。金标 answer 缺失 → answer_missing=True、semantic_correct=None。
   - line='tool_name_em'：非 finish 走工具名 EM（_extract_tool_name 对齐 harness，
     复用 R5R._extract_tool_name 逐字照抄版）；token_f1/rouge_l_f1 记 None。

3. tool_call_positions(generated_ids, tokenizer, steps=None) 返回键：
   每个出现位置 {step_index, chosen_logprob, eos_logprob, eos_minus_chosen}。
   - <tool_call> 在 Qwen3-4B-Instruct-2507 词表为单 token（id 151657，已验证
     tokenizer.convert_tokens_to_ids('<tool_call>')==151657）；本函数对未知词自动退回
     encode(add_special_tokens=False) 的多 token 序列匹配。
   - steps 为 harness capture 埋点 steps 列表（可选）：给定时按 step_index 附
     chosen/eos logprob 与 eos_minus_chosen（= eos_logprob - chosen_logprob）；不给定时
     三项记 None。

4. violation_decomposition(text, target_row, pool_names=None) 返回键：
   {name_em, args_keys_schema_valid, args_parse_ok, cross_block_ref, name_in_pool}
   - args_keys_schema_valid：生成块 arguments 键集合 ⊆ 金标 arguments 键集合且不含空键；
     金标 arguments 不可得（金标 target 无 name==target_tool_name 的闭合块 /
     arguments 不可解释为 dict）记 None；生成块不可解析记 None（args_parse_ok=False 披露）。
   - cross_block_ref 固定记 "NOT-APPLICABLE"：本数据集（agent-llm-traces）无跨块
     call-observation 引用结构（工具调用块之间不存在交叉引用语义），该分解维度不适用，
     逐行固定记此值以保持 schema 稳定。
   - name_in_pool：pool_names 给定时为生成 name ∈ pool_names（bool）；pool_names 未给或
     生成无 name 记 None。

本文件只定义纯函数；不做任何 git 操作、不写任何结果文件（自测打印除外）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import r5_reanalysis as R5R  # noqa: E402  复用严格口径解析器与工具名口径
import r5_finish_semantics as R5F  # noqa: E402  复用 finish 金标 answer 解析与 F1 实现

CROSS_BLOCK_REF_NOT_APPLICABLE = "NOT-APPLICABLE"

TOOL_CALL_MARKER = "<tool_call>"


def _arguments_keys_of(value: Dict[str, Any]) -> Optional[List[str]]:
    """取一个已解析工具调用 JSON dict 的 arguments 键列表（sorted）。

    arguments 缺失或为 {} → []；arguments 为字符串时二次 json.loads；
    arguments 不可解释为 dict（str 解析失败 / 非 dict 非 str 类型）→ None（无法判定）。
    """
    args = value.get("arguments")
    if args is None:
        return []
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if not isinstance(args, dict):
        return None
    return sorted(args.keys())


def strict_protocol_parse(text: Optional[str]) -> Dict[str, Any]:
    """严格主口径解析（与 agent/r5_reanalysis.py strict_protocol_valid 逐字一致）。

    口径：谓词命中（('<tool_call>' in text) or ('Action:' in text)）AND
    存在闭合 <tool_call>...</tool_call> 块 AND 块内 JSON 合法 AND 含 name 键
    （arguments 允许 {} 或缺失）。仅命中 Action: 且文本内无 <tool_call> 标记 →
    action_only=True、protocol_valid=False。

    返回::
      predicate_hit: 谓词是否命中
      action_only:   仅命中 Action: 且无 <tool_call> 标记
      parse_ok:      存在闭合块 JSON 合法且含 name 键（同 R5R 的 valid）
      name:          parse_ok 时首个有效闭合块的 name（str；name 为 null 时为 None）
      arguments_keys: parse_ok 时该块 arguments 键列表（见 _arguments_keys_of，可为 None）
      closed_block_found: 正则是否找到任意闭合 <tool_call> 块（JSON 未必合法）
      protocol_valid: == parse_ok（R5R valid 的同值字段）
    """
    t = text or ""
    predicate_hit = ("<tool_call>" in t) or ("Action:" in t)
    action_only = ("Action:" in t) and ("<tool_call>" not in t)
    closed_block_found = False
    parse_ok = False
    name: Optional[str] = None
    arguments_keys: Optional[List[str]] = None
    for block in R5R.TOOL_CALL_JSON_RE.findall(t):
        closed_block_found = True
        try:
            value = json.loads(block)
        except Exception:
            continue
        if isinstance(value, dict) and "name" in value:
            parse_ok = True
            raw = value.get("name")
            name = str(raw) if raw is not None else None
            arguments_keys = _arguments_keys_of(value)
            break
    return {
        "predicate_hit": predicate_hit,
        "action_only": action_only,
        "parse_ok": parse_ok,
        "name": name if parse_ok else None,
        "arguments_keys": arguments_keys if parse_ok else None,
        "closed_block_found": closed_block_found,
        "protocol_valid": parse_ok,
    }


def semantic_score(text: Optional[str], target_row: Dict[str, Any]) -> Dict[str, Any]:
    """行级语义评分：finish 目标走 finish_semantic 线，其余走 tool_name_em 线。

    finish 判定口径（对齐 agent/r5_reanalysis.py）：target_row['target_tool_name']=='finish'
    （FINISH_TOOL_NAME）。finish 线：金标 finish 调用的 arguments.answer 参数串 vs 生成文本，
    token-F1 与 ROUGE-L F1 复用 agent/r5_finish_semantics.py（tokenize/token_f1/rouge_l_f1），
    判对线 ROUGE-L F1 >= 0.5（未舍入浮点判定）。金标 answer 缺失（answer_missing）或
    金标解析失败（parse_error）→ semantic_correct=None，状态在 answer_status 披露。
    非 finish 线：_extract_tool_name(text) == target_tool_name 的 EM
    （对齐 harness 的 tool_name_match，含 target_tool_name is not None 门槛）；
    token_f1/rouge_l_f1 记 None。

    返回::
      semantic_correct: bool 或 None（answer 缺失/解析失败/不可判定）
      line: 'finish_semantic' | 'tool_name_em'
      token_f1 / rouge_l_f1: finish 线为 4 位小数浮点；tool_name_em 线为 None
      name_em: finish 线 None；tool_name_em 线 bool
      answer_missing: 金标 finish 调用无 answer 参数（仅 finish 线可为 True）
      answer_status / answer_detail: finish 线为金标解析状态（R5F.parse_gold_finish_answer），
        非 finish 线为 None
    """
    target = target_row.get("target") or ""
    target_tool = target_row.get("target_tool_name")
    if target_tool == R5R.FINISH_TOOL_NAME:
        parsed = R5F.parse_gold_finish_answer(target)
        gold = parsed["answer"] or ""
        gold_toks = R5F.tokenize(gold)
        pred_toks = R5F.tokenize(text or "")
        tf1 = R5F.token_f1(gold_toks, pred_toks)
        rl = R5F.rouge_l_f1(gold_toks, pred_toks)
        answer_missing = parsed["status"] == "answer_missing"
        if parsed["status"] in ("answer_missing", "parse_error"):
            semantic_correct = None
        else:
            semantic_correct = rl >= R5F.PASS_LINE
        return {
            "semantic_correct": semantic_correct,
            "line": "finish_semantic",
            "token_f1": round(tf1, 4),
            "rouge_l_f1": round(rl, 4),
            "name_em": None,
            "answer_missing": answer_missing,
            "answer_status": parsed["status"],
            "answer_detail": parsed["detail"],
        }
    pred_tool = R5R._extract_tool_name(text)
    name_em = target_tool is not None and target_tool == pred_tool
    return {
        "semantic_correct": name_em,
        "line": "tool_name_em",
        "token_f1": None,
        "rouge_l_f1": None,
        "name_em": name_em,
        "answer_missing": False,
        "answer_status": None,
        "answer_detail": None,
    }


def _find_subseq(ids: List[int], seq: List[int]) -> List[int]:
    """返回 seq 在 ids 中所有出现位置的起始下标（重叠亦计）。"""
    if not seq:
        return []
    positions: List[int] = []
    for i in range(len(ids) - len(seq) + 1):
        if ids[i : i + len(seq)] == seq:
            positions.append(i)
    return positions


def tool_call_positions(
    generated_ids: List[int],
    tokenizer: Any,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """在 generated_ids 中定位每处 <tool_call> 特殊 token，并可附 capture steps 的 logprob。

    tokenizer 词表检查：Qwen3-4B-Instruct-2507 中 <tool_call> 为单特殊 token
    （convert_tokens_to_ids('<tool_call>') == 151657，已打印验证）；若词表无此 token
    （返回 None），退回 encode('<tool_call>', add_special_tokens=False) 的多 token 序列做
    子序列匹配。

    返回每个出现位置::
      step_index:      出现位置在 generated_ids 中的起始步下标
      chosen_logprob:  steps 给定时 = steps[step_index]['chosen_logprob']（float），否则 None
      eos_logprob:     steps 给定时 = steps[step_index]['eos_logprob']（float），否则 None
      eos_minus_chosen: steps 给定时 = eos_logprob - chosen_logprob（4 位以上精度未舍入，
                        保留 6 位小数存储），否则 None
    """
    tid = tokenizer.convert_tokens_to_ids(TOOL_CALL_MARKER)
    if isinstance(tid, int):
        seq = [tid]
    else:
        seq = list(tokenizer.encode(TOOL_CALL_MARKER, add_special_tokens=False))
    positions = _find_subseq(list(generated_ids), seq)
    out: List[Dict[str, Any]] = []
    for pos in positions:
        rec: Dict[str, Any] = {"step_index": pos}
        chosen: Optional[float] = None
        eos: Optional[float] = None
        if steps is not None and 0 <= pos < len(steps):
            chosen = steps[pos].get("chosen_logprob")
            eos = steps[pos].get("eos_logprob")
        rec["chosen_logprob"] = chosen
        rec["eos_logprob"] = eos
        rec["eos_minus_chosen"] = (
            round(eos - chosen, 6) if chosen is not None and eos is not None else None
        )
        out.append(rec)
    return out


def _gold_arguments_keys(target: str, target_tool: Optional[str]) -> Optional[List[str]]:
    """金标 arguments 键列表：target 文本中首个 name==target_tool 的有效闭合块的
    arguments 键（sorted）；不可得（target_tool 为 None / 无匹配闭合块 / arguments
    不可解释为 dict）→ None（无法判定 schema 合法性）。"""
    if not target_tool:
        return None
    for block in R5R.TOOL_CALL_JSON_RE.findall(target or ""):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if not (isinstance(value, dict) and "name" in value):
            continue
        if str(value.get("name")) != target_tool:
            continue
        return _arguments_keys_of(value)
    return None


def violation_decomposition(
    text: Optional[str],
    target_row: Dict[str, Any],
    pool_names: Optional[Any] = None,
) -> Dict[str, Any]:
    """行级违规分解（closeout 用；纯 CPU）。

    返回::
      name_em: 工具名 EM（_extract_tool_name 对齐 harness，含 target_tool_name
               is not None 门槛）
      args_keys_schema_valid: 生成块 arguments 键集合 ⊆ 金标 arguments 键集合且不含
              空键（''）。金标 arguments 不可得（见 _gold_arguments_keys）或生成块
              不可解析（args_parse_ok=False）→ None（无法判定）。
      args_parse_ok: 生成文本存在闭合块 JSON 合法且含 name（strict_protocol_parse 的
              parse_ok 同值）
      cross_block_ref: 固定 "NOT-APPLICABLE"——本数据集（agent-llm-traces）无跨块
              call-observation 引用结构，工具调用块之间不存在交叉引用语义，该分解
              维度不适用，逐行固定记此值以保持 schema 稳定。
      name_in_pool: pool_names 给定时 = 生成 name ∈ pool_names（bool）；pool_names
              未给或生成无 name → None。
    """
    target = target_row.get("target") or ""
    target_tool = target_row.get("target_tool_name")
    pred_tool = R5R._extract_tool_name(text)
    name_em = target_tool is not None and target_tool == pred_tool
    parsed = strict_protocol_parse(text)
    gold_keys = _gold_arguments_keys(target, target_tool)
    gen_keys = parsed["arguments_keys"]
    if gen_keys is not None and gold_keys is not None:
        args_keys_schema_valid = ("" not in gen_keys) and all(k in gold_keys for k in gen_keys)
    else:
        args_keys_schema_valid = None
    if pool_names is not None and parsed["name"] is not None:
        name_in_pool: Optional[bool] = parsed["name"] in pool_names
    else:
        name_in_pool = None
    return {
        "name_em": name_em,
        "args_keys_schema_valid": args_keys_schema_valid,
        "args_parse_ok": parsed["parse_ok"],
        "cross_block_ref": CROSS_BLOCK_REF_NOT_APPLICABLE,
        "name_in_pool": name_in_pool,
    }


def _selftest() -> None:
    """纯 CPU 自测：4 个合成文本过 strict_protocol_parse、semantic_score 两线、
    tool_call_positions 与 violation_decomposition。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== r5_closeout_lib selftest（纯 CPU） ===")

    texts = {
        "合法闭合块": '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>',
        "未闭合块": '<tool_call>{"name": "get_weather"}',
        "空arguments": '<tool_call>{"name": "finish", "arguments": {}}</tool_call>',
        "Action: only": "Action: get_weather",
    }
    for label, t in texts.items():
        r = strict_protocol_parse(t)
        ref = R5R.strict_protocol_valid(t)
        assert r["protocol_valid"] == ref["valid"], (label, r, ref)
        assert r["name"] == ref["name"], (label, r, ref)
        print(f"strict_protocol_parse[{label}]:", json.dumps(r, ensure_ascii=False))

    assert texts["合法闭合块"] and strict_protocol_parse(texts["合法闭合块"])["arguments_keys"] == ["city"]
    assert strict_protocol_parse(texts["空arguments"])["arguments_keys"] == []
    assert strict_protocol_parse(texts["未闭合块"])["closed_block_found"] is False
    assert strict_protocol_parse(texts["Action: only"])["action_only"] is True
    assert strict_protocol_parse(texts["Action: only"])["protocol_valid"] is False

    finish_row = {
        "target_tool_name": "finish",
        "target": '<tool_call>{"name": "finish", "arguments": {"answer": "the weather in san francisco is sunny"}}</tool_call>',
    }
    finish_text = '<tool_call>{"name": "finish", "arguments": {"answer": "the weather in san francisco is sunny"}}</tool_call>'
    s = semantic_score(finish_text, finish_row)
    assert s["line"] == "finish_semantic" and s["semantic_correct"] is True, s
    assert 0.0 < s["rouge_l_f1"] < 1.0 and 0.0 < s["token_f1"] < 1.0, s
    print("semantic_score[finish 命中（整段生成文本 vs 金标 answer）]:", json.dumps(s, ensure_ascii=False))

    s_exact = semantic_score("the weather in san francisco is sunny", finish_row)
    assert s_exact["semantic_correct"] is True and s_exact["rouge_l_f1"] == 1.0, s_exact
    assert s_exact["token_f1"] == 1.0, s_exact
    print("semantic_score[finish 文本==answer 串（F1=1.0）]:", json.dumps(s_exact, ensure_ascii=False))

    finish_missing_row = {
        "target_tool_name": "finish",
        "target": '<tool_call>{"name": "finish", "arguments": {}}</tool_call>',
    }
    s2 = semantic_score(finish_text, finish_missing_row)
    assert s2["answer_missing"] is True and s2["semantic_correct"] is None, s2
    print("semantic_score[finish answer缺失]:", json.dumps(s2, ensure_ascii=False))

    tool_row = {
        "target_tool_name": "get_weather",
        "target": '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>',
    }
    s3 = semantic_score('<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>', tool_row)
    assert s3["line"] == "tool_name_em" and s3["name_em"] is True and s3["semantic_correct"] is True, s3
    s4 = semantic_score('<tool_call>{"name": "send_email", "arguments": {}}</tool_call>', tool_row)
    assert s4["line"] == "tool_name_em" and s4["name_em"] is False and s4["semantic_correct"] is False, s4
    print("semantic_score[tool_name_em 命中]:", json.dumps(s3, ensure_ascii=False))
    print("semantic_score[tool_name_em 不中]:", json.dumps(s4, ensure_ascii=False))

    class _FakeTokenizer:
        TOOL_CALL_ID = 151657

        def convert_tokens_to_ids(self, tok):
            return self.TOOL_CALL_ID if tok == TOOL_CALL_MARKER else None

        def encode(self, text, add_special_tokens=False):
            return [99, 99] if text == TOOL_CALL_MARKER else []

    fake_steps = [
        {"step": i, "token_id": tid, "chosen_logprob": -0.1 * i, "eos_logprob": -1.0 - 0.1 * i}
        for i, tid in enumerate([10, 151657, 20, 30, 151657, 40])
    ]
    pos = tool_call_positions([10, 151657, 20, 30, 151657, 40], _FakeTokenizer(), steps=fake_steps)
    assert [p["step_index"] for p in pos] == [1, 4], pos
    assert pos[0]["eos_minus_chosen"] == round(-1.1 - (-0.1), 6), pos
    pos_nosteps = tool_call_positions([10, 151657, 20], _FakeTokenizer())
    assert pos_nosteps[0]["chosen_logprob"] is None and pos_nosteps[0]["eos_minus_chosen"] is None
    print("tool_call_positions[单 token 151657]:", json.dumps(pos, ensure_ascii=False))

    vd_extra_key = violation_decomposition(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "SF", "foo": 1}}</tool_call>',
        tool_row,
    )
    assert vd_extra_key["name_em"] is True and vd_extra_key["args_keys_schema_valid"] is False, vd_extra_key
    vd_ok = violation_decomposition(
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>', tool_row,
        pool_names=["get_weather", "send_email"],
    )
    assert vd_ok["args_keys_schema_valid"] is True and vd_ok["name_in_pool"] is True, vd_ok
    assert vd_ok["cross_block_ref"] == "NOT-APPLICABLE"
    vd_nogold = violation_decomposition("Action: nothing", {"target_tool_name": None, "target": ""})
    assert vd_nogold["args_keys_schema_valid"] is None and vd_nogold["args_parse_ok"] is False, vd_nogold
    print("violation_decomposition[多余键]:", json.dumps(vd_extra_key, ensure_ascii=False))
    print("violation_decomposition[合法+池命中]:", json.dumps(vd_ok, ensure_ascii=False))
    print("=== selftest 全部断言通过 ===")


if __name__ == "__main__":
    _selftest()
