# -*- coding: utf-8 -*-
"""
统计 recognition_export_cot 目录下 10 个 doubao json 文件中，
每个题目(id) 在 10 次分类中的错误次数。

判定规则(与 acc_evaluation.py get_acc 一致)：
    - 无 response 的样本: 只计入分母, 不判对(视为错误)。
    - 有 response: result 转大写, TRUE/1 -> '1', FALSE/0 -> '0';
      选择题答案(A~E)、'none'/'unknown' 等原样保留, 再与 label 字符串比较。
      字符串相等视为正确, 否则视为错误。

输出：
    1) question_error_counts.xlsx : Excel, 题目(id) --> 错误次数
    2) error_stats.json           : 统计信息(直方图: 错误0次/1次/2次...的题目个数)

跳过规则:
    not_valid_ids.json 中列出的 id 不参与统计(直接跳过)。

用法:
    python count_errors.py [json目录 或 直接指定多个json]
"""
import json
import os
import sys
import glob
from collections import Counter, defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
NOT_VALID_IDS = r"I:\icdInformations\3claim\med-ver\z_callvlama\input\not_valid_ids.json"


def judge_correct(item):
    """单条记录判定, 与 acc_evaluation.py 的 get_acc 完全一致:
    1) 无 response 的样本: 只计入分母, 永不判对(相当于错误)。
    2) 有 response: result 转大写, TRUE/1 -> '1', FALSE/0 -> '0',
       其余值(选择题答案 A~E、'none'/'unknown' 等)原样保留,
       再与 str(label) 字符串相等比较。
    """
    if not item.get("response"):
        return False
    label = str(item["label"]).strip()
    result_str = str(item.get("result", "")).strip().upper()
    if result_str in ("TRUE", "1"):
        result_str = "1"
    elif result_str in ("FALSE", "0"):
        result_str = "0"
    return label == result_str


def main():
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            json_dir = sys.argv[1]
            files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        else:
            files = sorted(sys.argv[1:])
    else:
        json_dir = DIR
        files = sorted(glob.glob(os.path.join(json_dir, "doubao-seed-2.0-lite_*.json")))
    if not files:
        print("未找到任何 json 文件")
        sys.exit(1)

    n_files = len(files)
    error_counts = defaultdict(int)   # id -> 错误次数
    total_counts = defaultdict(int)   # id -> 出现次数(应=文件数)
    meta = {}                         # id -> {label, claim, claim_type}

    # 加载不参与统计的 id 列表
    try:
        with open(NOT_VALID_IDS, "r", encoding="utf-8") as fh:
            skip_ids = {item.get("id") for item in json.load(fh)}
    except FileNotFoundError:
        print(f"[!] 未找到 not_valid_ids.json: {NOT_VALID_IDS}")
        skip_ids = set()
    skipped = set()                   # 统计被跳过的 id

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for rec in data:
            rid = rec["id"]
            if rid in skip_ids:
                skipped.add(rid)
                continue
            total_counts[rid] += 1
            if not judge_correct(rec):
                error_counts[rid] += 1
            if rid not in meta:
                meta[rid] = {
                    "label": rec.get("label"),
                    "claim": rec.get("claim"),
                    "claim_type": rec.get("claim_type"),
                }

    # 直方图: 错误次数 -> 题目个数
    hist = Counter(error_counts[rid] for rid in total_counts)  # 未出现的id默认0次错误
    n_questions = len(total_counts)
    total_errors = sum(error_counts.values())
    total_records = sum(total_counts.values())

    # ---- 生成 Excel: 题目 -> 错误次数 ----
    xlsx_path = os.path.join(DIR, "question_error_counts.xlsx")
    _write_xlsx(xlsx_path, total_counts, error_counts, meta)
    if xlsx_path:  # 未导出成功时下方 _write_xlsx 已打印提示
        print(f"[OK] Excel 已写出: {xlsx_path}")

    # ---- 生成统计 JSON ----
    stats = {
        "文件数": n_files,
        "文件列表": [os.path.basename(f) for f in files],
        "题目总数(唯一id数)": n_questions,
        "跳过的id数量(在not_valid_ids中)": len(skipped),
        "跳过列表": sorted(skipped),
        "总记录数(题目x文件数)": total_records,
        "推理次数": n_files,
        "总错误次数": total_errors,
        "总正确次数": total_records - total_errors,
        "总正确率": round((total_records - total_errors) / total_records, 4),
        "每题至少错1次的题目数": sum(1 for r in total_counts if error_counts[r] > 0),
        "每次推理均正确的题目数(0次错误)": hist.get(0, 0),
        "n次推理全部错误的题目数": sum(1 for r in total_counts if error_counts[r] == n_files),
        "错误次数分布": {
            f"错误{k}次": hist.get(k, 0) for k in range(n_files + 1)
        },
    }
    json_path = os.path.join(DIR, "error_stats.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(f"[OK] 统计 JSON 已写出: {json_path}")
    print()
    print("=== 错误次数分布 (题目个数) ===")
    for k in range(n_files + 1):
        c = hist.get(k, 0)
        bar = "#" * c if c else ""
        print(f"错 {k:>2} 次: {c:>5} 题  {bar}")


def _write_xlsx(xlsx_path, total_counts, error_counts, meta):
    rows = sorted(
        [
            (
                rid,
                error_counts[rid],
                total_counts[rid] - error_counts[rid],
                total_counts[rid],
                meta[rid]["label"],
                meta[rid]["claim_type"],
                meta[rid]["claim"],
            )
            for rid in total_counts
        ],
        key=lambda r: (-r[1], r[0]),
    )

    try:
        import openpyxl
    except ImportError:
        print("[!] 未安装 openpyxl, 改为写出 CSV 作为备选")
        import csv
        csv_path = os.path.join(DIR, "question_error_counts.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["题目(id)", "错误次数", "正确次数", "出现次数", "真实标签", "claim_type", "claim"])
            w.writerows(rows)
        print(f"[OK] CSV 已写出: {csv_path}")
        return None

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "题目-错误次数"
    ws.append(["题目(id)", "错误次数", "正确次数", "出现次数", "真实标签(label)", "claim_type", "claim"])
    for row in rows:
        ws.append(list(row))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_letter, width in zip("ABCDEFG", (46, 10, 10, 10, 12, 12, 80)):
        ws.column_dimensions[col_letter].width = width
    wb.save(xlsx_path)
    return xlsx_path


if __name__ == "__main__":
    main()