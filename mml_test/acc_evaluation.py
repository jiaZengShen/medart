import json
import os
import argparse

from tqdm import tqdm

# 加载需要跳过的无效 id 列表
NOT_VALID_IDS_PATH = r"I:\icdInformations\3claim\med-ver\z_callvlama\input\not_valid_ids.json"
if os.path.exists(NOT_VALID_IDS_PATH):
    with open(NOT_VALID_IDS_PATH, "r", encoding="utf-8") as f:
        NOT_VALID_IDS = set(item["id"] for item in json.load(f))
else:
    NOT_VALID_IDS = set()


def get_acc(examples):
    res = {
        "recognition": 0,
        "medical": 0,
        "measure": 0,
        "parallel": 0,
        "total": 0
    }
    numbers = {
        "recognition": 0,
        "medical": 0,
        "measure": 0,
        "parallel": 0,
        "total": 0
    }

    for item in examples:
        # 跳过无效 id
        if item.get("id") in NOT_VALID_IDS:
            continue

        claim_type = item['claim_type']
        label = item['label']
        numbers[claim_type] += 1
        numbers['total'] += 1
        if item.get("response"):
            result_str = str(item["result"]).strip().upper()
            if result_str == "TRUE" or result_str == "1":
                result_str = "1"
            elif result_str == "FALSE" or result_str == "0":
                result_str = "0"
            if label == result_str:
                res[claim_type] += 1
                res["total"] += 1
        else:
            model_name = item.get("model_name")
            id = item.get("id")
            #print("ddddd" + str(model_name) + "  " + str(id))
    for k, v in res.items():
        if numbers[k] > 0:
            res[k] /= numbers[k]
        else:
            res[k] = None
    print(f"{len(examples)}实际：{numbers['total']}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/recognition_export_cot")
    parser.add_argument('--eval_dir', type=str, default="processed_outputs")
    args = parser.parse_args()

    subdir = os.path.basename(args.output_dir)
    os.makedirs(os.path.join(args.eval_dir, subdir), exist_ok=True)

    for output_file in os.listdir(args.output_dir):
        if not output_file.endswith(".json"):
            continue
        if output_file.startswith("Baichuan-M3-Plus.json"):
            continue

        examples_path = os.path.join(args.output_dir, output_file)
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        eval_file = os.path.join(args.eval_dir, subdir, output_file)

        outputs = get_acc(examples)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()