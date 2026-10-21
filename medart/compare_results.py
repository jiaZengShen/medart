
import json
import csv
from pathlib import Path

def load_results(folder_path):
    json_path = Path(folder_path) / "single_agent_results.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    for sample in data['samples']:
        sample_id = sample['sample_id']
        results[sample_id] = {
            'correct': sample['correct'],
            'final_answer': sample['final_answer'],
            'true_label': sample['true_label'],
            'claim': sample['claim']
        }
    return results

def compare_and_generate_csv(folder1, folder2, folder3, output_csv):
    results1 = load_results(folder1)
    results2 = load_results(folder2)
    results3 = load_results(folder3)

    all_sample_ids = set(results1.keys()) | set(results2.keys())

    rows = []
    for sample_id in sorted(all_sample_ids):
        in_folder1 = sample_id in results1
        in_folder2 = sample_id in results2
        in_folder3 = sample_id in results3

        correct1 = results1[sample_id]['correct'] if in_folder1 else None
        correct2 = results2[sample_id]['correct'] if in_folder2 else None
        correct3 = results3[sample_id]['correct'] if in_folder3 else None

        # Determine comparison category across 3 folders
        presence_mask = (in_folder1, in_folder2, in_folder3)
        correct_mask = (correct1, correct2, correct3)

        if all(presence_mask):
            if all(correct_mask):
                comparison = "all_correct"
            elif not any(correct_mask):
                comparison = "all_incorrect"
            else:
                # Exactly which folders are correct
                parts = []
                if correct1: parts.append("f1")
                if correct2: parts.append("f2")
                if correct3: parts.append("f3")
                comparison = f"{'+'.join(parts)}_correct"
        else:
            # Not all folders have this sample
            present_parts = []
            for i, p in enumerate(presence_mask, 1):
                if p: present_parts.append(f"f{i}")
            comparison = f"only_in_{'+'.join(present_parts)}"

        # Pick claim and true_label from first available folder
        claim = ""
        if in_folder1:
            claim = results1[sample_id]['claim']
        elif in_folder2:
            claim = results2[sample_id]['claim']
        elif in_folder3:
            claim = results3[sample_id]['claim']

        true_label = ""
        if in_folder1:
            true_label = results1[sample_id]['true_label']
        elif in_folder2:
            true_label = results2[sample_id]['true_label']
        elif in_folder3:
            true_label = results3[sample_id]['true_label']

        row = {
            'sample_id': sample_id,
            'folder1_correct': correct1,
            'folder2_correct': correct2,
            'folder3_correct': correct3,
            'comparison': comparison,
            'true_label': true_label,
            'claim': claim
        }
        rows.append(row)

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['sample_id', 'folder1_correct', 'folder2_correct', 'folder3_correct',
                      'comparison', 'true_label', 'claim']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison complete! CSV saved to: {output_csv}")
    print("\nSummary statistics:")
    print(f"Total samples: {len(rows)}")

    categories = {}
    for row in rows:
        cat = row['comparison']
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    folder_names = [Path(folder1).name, Path(folder2).name, Path(folder3).name]
    for i in range(3):
        correct_key = f'folder{i+1}_correct'
        correct_count = sum(1 for r in rows if r[correct_key] is True)
        total = sum(1 for r in rows if r[correct_key] is not None)
        if total > 0:
            print(f"\n{folder_names[i]} accuracy: {correct_count}/{total} = {correct_count/total:.2%}")

if __name__ == "__main__":
    folder1 = "outputs_2026-07-25_11-12-58"
    folder2 = "outputs_2026-07-26_14-24-44"
    folder3 = "outputs_2026-08-01_21-28-48"
    output_csv = "comparison_results5.csv"

    compare_and_generate_csv(folder1, folder2, folder3, output_csv)
