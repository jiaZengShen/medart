"""
使用已训练的BioBERT模型进行预测
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_DIR = "./biobert_filter_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """加载已训练的模型"""
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, texts, batch_size=32):
    """批量预测"""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        batch_probs = probs.cpu().numpy()
        batch_preds = np.argmax(batch_probs, axis=1)

        all_preds.extend(batch_preds.tolist())
        all_probs.extend(batch_probs.max(axis=1).tolist())

        if (i + batch_size) % 500 == 0:
            print(f"已预测 {min(i + batch_size, len(texts))}/{len(texts)} 条...")

    return all_preds, all_probs

def softmax(logits):
    """将logits转换为概率"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

if __name__ == "__main__":
    print("=" * 50)
    print("BioBERT 模型预测")
    print("=" * 50)
    print(f"使用设备: {DEVICE}")

    print("\n加载模型...")
    model, tokenizer = load_model()
    print("模型加载完成!")

    train_csv = "./data/train.csv"
    val_csv = "./data/test.csv"

    for csv_path in [train_csv, val_csv]:
        source = "train" if "train" in csv_path else "val"
        print(f"\n预测 {source} 集...")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['claim'])
        df = df[df['label'].isin(['0', '1', 0, 1])]
        df['label'] = df['label'].astype(int)

        ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
        texts = df['claim'].tolist()
        labels = df['label'].tolist()

        print(f"共 {len(texts)} 条数据")

        preds, probs = predict(model, tokenizer, texts)

        result_df = pd.DataFrame({
            'id': ids,
            'claim': texts,
            'true_label': labels,
            'predicted_label': preds,
            'confidence': probs
        })

        output_path = f"{MODEL_DIR}/{source}_predictions.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"预测结果已保存到: {output_path}")

    print("\n全部预测完成!")
