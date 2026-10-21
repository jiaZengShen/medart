"""
从模型 response 中提取答案并更新 result 字段。

用法：
    python fix_answers.py

流程：
    1. 读取 outputs/recognition_export_cot/Baichuan-M3-Plus.json
    2. 根据 claim_type 选择对应的 pattern 组：
       - medical → 匹配字母 A-I 的 pattern
       - 其他     → 匹配 Yes/No 的 pattern
    3. 取最后一条匹配结果作为最终答案
    4. 更新 result 字段，写回文件
"""

import json
import re
import os
import sys

# ── 文件路径 ──────────────────────────────────────────────────
JSON_PATH = r"outputs\recognition_export_cot\Baichuan-M3-Plus.json"
JSON_PATH_PROCESSED = r"outputs\recognition_export_cot\Baichuan-M3-Plus_processed.json"


# ── 字母类 pattern（medical 专用） ─────────────────────────────
# 捕获 A B C D E F G H I，不区分大小写

PATTERNS_LETTER = [
    # 1. 中文 "答案是" 系列
    # 匹配：答案是**B**   答案是 B   答案为：C
    re.compile(r'答案[是为:：\w]*?\*{0,2}\s*([A-HI-Ya-hi-y]+)\s*\*{0,2}'),

    # 2. 中文 "的诊断是" 系列
    # 匹配：的诊断是：**C**   的诊断是 D
    re.compile(r'的诊断[是:：\w]*?\*{0,2}\s*([A-HI-Ya-hi-y]+)\s*\*{0,2}'),

    # 3. 英文 "the answer is" / "answer is" 系列
    # 匹配：the answer is: E    answer is : B
    re.compile(r'(?:the\s+)?answer\s+is\s*:?\s*([A-Ha-h]+)', re.IGNORECASE),

    # 4. 方括号 [字母]
    # 匹配：[ C ]   [ B]
    re.compile(r'\[\s*([A-Ha-h])\s*\]', re.IGNORECASE),

    # 5. "结论：X" 等总结性提示
    # 匹配：结论：B   结论: C
    re.compile(r'结论[是为:：]?\s*\**\s*([A-Ha-h])\s*\**'),

    # 6. "answer is" + 粗体/括号 + 字母
    # 匹配：answer is: **A**    answer is: **(D)
    re.compile(r'(?:the\s+)?answer\s+is\s*:?\s*\**\s*\(?\s*\**\s*([A-Ha-h])\s*\**\s*\)?\s*\**', re.IGNORECASE),

    # 7. "答案是" + 粗体/括号 + 字母
    # 匹配：答案是：**(D)   答案是： **B**
    re.compile(r'答案[是为:：]*?\s*\**\s*\(?\s*\**\s*([A-HI-Ya-hi-y])\s*\**\s*\)?\s*\**'),

    # 8. "的诊断是" + 粗体/括号 + 字母
    # 匹配：的诊断是：**(A)   的诊断是 D
    re.compile(r'的诊断[是:：]*?\s*\**\s*\(?\s*\**\s*([A-HI-Ya-hi-y])\s*\**\s*\)?\s*\**'),
]


# ── Yes/No 类 pattern（非 medical 专用） ───────────────────────

PATTERNS_YESNO = [
    # ── **Yes** / **No**（纯粗体包裹） ─────────────────────
    re.compile(r'\b(Yes|No)\b', re.IGNORECASE),
    # re.compile(r'Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r'Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
    #
    # # ── 方括号 ───────────────────────────────────────────
    # re.compile(r'\[\s*(yes|no)\s*\]', re.IGNORECASE),
    #
    # # ── 最终答案是：Answer: ───────────────────────────────
    # re.compile(r'最终答案是：\s*Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r'最终答案是：\s*Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
    #
    # # ── the / final answer is: Answer: ────────────────────
    # re.compile(r',?\s*final answer is:\s*Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r',?\s*final answer is:\s*Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
    # re.compile(r',?\s*final answer is:\s*Answer:\s*(Yes|No)\s*\**', re.IGNORECASE),
    #
    # # ── the final answer is: Answer: ─────────────────────
    # re.compile(r',?\s*the final answer is:\s*Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r',?\s*the final answer is:\s*Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
    # re.compile(r',?\s*the final answer is:\s*Answer:\s*(Yes|No)\s*\**', re.IGNORECASE),
    #
    # # ── Therefore, the final answer is: Answer: ──────────
    # re.compile(r'Therefore,?\s*the final answer is:\s*Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r'Therefore,?\s*the final answer is:\s*Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
    # re.compile(r'Therefore,?\s*the final answer is:\s*Answer:\s*(Yes|No)\s*\**', re.IGNORECASE),
    #
    # # ── 最终答案是: Answer: ──────────────────────────────
    # re.compile(r'最终答案是:\s*Answer:\s*(Yes|No)', re.IGNORECASE),
    # re.compile(r'最终答案是:\s*Answer:\s*\**\s*(Yes|No)\s*\**', re.IGNORECASE),
]


def normalize_answer(raw: str ,claim_type) -> str | None:
    """将原始匹配结果转为标准格式，无效则返回 None。"""
    raw = raw.strip()
    upper = raw.upper()

    if upper in set("ABCDEFGHI") and claim_type == 'medical' :
        return upper
    if upper == "YES":
        return "Yes"
    if upper == "NO":
        return "No"
    return None


def extract_answer(response: str, claim_type: str) -> str | None:
    """
    根据 claim_type 选择 pattern 组，取最后一条有效匹配。

    参数
    ----------
    response : str  模型输出文本
    claim_type : str  取值 medical / recognition / parallel / measure

    返回
    -------
    str | None  标准化后的答案，未匹配到返回 None
    """
    patterns = PATTERNS_LETTER if claim_type == "medical" else PATTERNS_YESNO

    last_valid = None
    for pattern in patterns:
        for match in pattern.finditer(response):
            raw = match.group(1)
            answer = normalize_answer(raw,claim_type)
            if answer:
                last_valid = answer

    return last_valid


def main():
    json_path = JSON_PATH

    if not os.path.exists(json_path):
        print(f"[错误] 文件不存在: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    updated = 0
    still_unknown = 0

    for i, entry in enumerate(data):
        old_result = entry.get("result", "")
        response = entry.get("response", "")
        claim_type = entry.get("claim_type", "")

        id = entry.get("id","")
        if id == 'HC18/training_set/015_HC.png_measure_0' :
            print("ddddd")

        if not response:
            continue

        answer = extract_answer(response, claim_type)

        # 非 medical 的 Yes/No 转成 Python bool（JSON 序列化时变 true/false）
        if claim_type != "medical" and answer in ("Yes", "No"):
            answer = answer == "Yes"

        if answer is not None  and old_result == "unknown":
            entry["result"] = answer
            entry["old_result"] = old_result
            updated += 1
            print(f"  [{i+1}/{total}] {old_result} → {answer}")
        else:
            if old_result == "unknown":
                still_unknown += 1

    with open(JSON_PATH_PROCESSED, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 处理完成: {json_path}")
    print(f"   总条目: {total}")
    print(f"   已更新: {updated}")
    print(f"   仍为 unknown: {still_unknown}")


if __name__ == "__main__":
    main()
