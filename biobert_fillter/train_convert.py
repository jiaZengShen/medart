"""
BioBERT Fine-tuning for Statement Binary Classification
用于可猜测数据过滤器的训练

输入：声明文本（如"CT shows an obstructing mass in the left lung"）
输出：True/False 二分类
"""
import string

import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import os


# ==================== 1. 配置参数 ====================
class Config:
    # 模型路径
    model_name = "../../../model/biobert-base-cased-v1.2"  # BioBERT v1.1 (PubMed + PMC)
    
    # 数据路径
    #train_data_path = "./data/train.csv"  # 训练数据CSV
    train_data_path = "./data/test.csv"  # 训练数据CSV
    #val_data_path = "./data/test.csv"     # 验证数据CSV
    val_data_path = "./data/train.csv"     # 验证数据CSV
    
    # 训练参数
    output_dir = "./biobert_filter_model_convert"   # 模型保存路径
    max_seq_length = 256                     # 最大序列长度
    train_batch_size = 32                    # 训练批次大小
    eval_batch_size = 64                     # 评估批次大小
    learning_rate = 2e-5                     # 学习率
    num_epochs = 30                           # 训练轮数
    warmup_ratio = 0.1                       # 预热比例
    weight_decay = 0.01                      # 权重衰减
    
    # 过滤阈值（训练后的模型会用这个阈值判断）
    confidence_threshold = 0.9               # 置信度阈值


config = Config()


# ==================== 2. 数据加载与预处理 ====================
def load_and_prepare_data(config):
    """
    加载训练数据和验证数据
    格式：CSV文件，包含 claim 和 label 两列
    label: 支持 0/1 二分类以及 A/B/C/D/E/F/G 多分类
    """
    # 读取训练数据
    train_df = pd.read_csv(config.train_data_path)
    # 读取验证数据
    val_df = pd.read_csv(config.val_data_path)

    # 过滤空的 claim
    train_df = train_df.dropna(subset=['claim'])
    val_df = val_df.dropna(subset=['claim'])

    valid_labels = ['0', '1', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    train_df = train_df[train_df['label'].isin(valid_labels)]
    val_df = val_df[val_df['label'].isin(valid_labels)]

    # 统一为字符串
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)

    # 测试阶段：限制数据条数为10条
    # train_df = train_df.head(10)
    # val_df = val_df.head(10)

    # 确保列名正确
    assert 'claim' in train_df.columns, "训练CSV必须包含'claim'列"
    assert 'label' in train_df.columns, "训练CSV必须包含'label'列"
    assert 'claim' in val_df.columns, "验证CSV必须包含'claim'列"
    assert 'label' in val_df.columns, "验证CSV必须包含'label'列"

    # 构建 label 到 id 的映射（基于训练集与验证集中实际出现的标签）
    all_label_set = sorted(set(train_df['label'].tolist()) | set(val_df['label'].tolist()))
    label2id = {lab: i for i, lab in enumerate(all_label_set)}
    id2label = {i: lab for lab, i in label2id.items()}

    # 提取数据（包含id）
    train_ids = train_df['id'].tolist() if 'id' in train_df.columns else list(range(len(train_df)))
    train_texts = train_df['claim'].tolist()
    train_labels = [label2id[l] for l in train_df['label'].tolist()]
    val_ids = val_df['id'].tolist() if 'id' in val_df.columns else list(range(len(val_df)))
    val_texts = val_df['claim'].tolist()
    val_labels = [label2id[l] for l in val_df['label'].tolist()]

    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"标签映射: {label2id}")
    print(f"训练集标签分布: {train_df['label'].value_counts().to_dict()}")
    print(f"验证集标签分布: {val_df['label'].value_counts().to_dict()}")

    return train_texts, val_texts, train_labels, val_labels, train_ids, val_ids, label2id, id2label


def tokenize_function(tokenizer, texts, labels, max_length):
    """使用tokenizer对文本进行编码"""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None  # 返回list，稍后转dataset
    )
    
    # 构建dataset格式
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }


# ==================== 3. 定义评估指标 ====================
def compute_metrics(pred):
    """计算分类指标，包括精确率、召回率、F1、准确率（多分类用macro）"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# ==================== 4. 主训练流程 ====================
def main():
    print("=" * 50)
    print("BioBERT 微调训练")
    print("=" * 50)
    
    # 4.1 加载数据
    print("\n[1/5] 加载数据...")
    train_texts, val_texts, train_labels, val_labels, train_ids, val_ids, label2id, id2label = load_and_prepare_data(config)

    num_labels = len(label2id)
    print(f"类别数: {num_labels}")

    # 4.2 加载模型和分词器
    print("\n[2/5] 加载BioBERT模型和分词器...")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")
    
    # 4.3 数据Tokenization
    print("\n[3/5] 对训练和验证数据进行tokenization...")
    
    # 训练集
    train_encodings = tokenize_function(
        tokenizer, train_texts, train_labels, config.max_seq_length
    )
    # 验证集
    val_encodings = tokenize_function(
        tokenizer, val_texts, val_labels, config.max_seq_length
    )
    
    # 转换为HuggingFace Dataset格式
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    # 4.4 设置训练参数
    print("\n[4/5] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",           # 每个epoch结束后评估
        save_strategy="epoch",                 # 每个epoch结束后保存
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,           # 训练结束后加载最佳模型
        metric_for_best_model="f1",            # 用F1分数选择最佳模型
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        report_to="none",                      # 禁用wandb等日志工具
        save_total_limit=2,                    # 只保留最后2个checkpoint
        fp16=torch.cuda.is_available(),        # GPU可用时使用混合精度
    )
    
    # 4.5 开始训练
    print("\n[5/5] 开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 早停
    )
    
    # 执行训练
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"\n模型已保存到: {config.output_dir}")
    
    # 最终评估
    print("\n最终评估结果:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 获取验证集每个样本的预测结果
    print("\n验证集预测详情:")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    # 使用 softmax 将 logits 转换为概率
    exp_preds = np.exp(predictions.predictions - np.max(predictions.predictions, axis=-1, keepdims=True))
    probs = exp_preds / exp_preds.sum(axis=-1, keepdims=True)
    probs = probs.max(axis=-1)
    
    # 打印前20个样本的预测结果
    print(f"{'样本序号':<6} {'真实标签':<6} {'预测标签':<6} {'置信度':<10} {'claim'}")
    print("-" * 120)
    for i in range(min(20, len(val_texts))):
        true_lab = id2label[val_labels[i]]
        pred_lab = id2label[int(preds[i])]
        print(f"{i:<6} {true_lab:<6} {pred_lab:<6} {probs[i]:<10.4f} {val_texts[i][:50]}...")

    # 保存完整的预测结果到文件
    pred_df = pd.DataFrame({
        'id': val_ids,
        'claim': val_texts,
        'true_label': [id2label[l] for l in val_labels],
        'predicted_label': [id2label[int(p)] for p in preds],
        'confidence': probs.tolist()
    })
    pred_output_path = f"{config.output_dir}/val_predictions.csv"
    pred_df.to_csv(pred_output_path, index=False, encoding='utf-8')
    print(f"\n完整预测结果已保存到: {pred_output_path}")
    
    # 获取训练集每个样本的预测结果
    print("\n训练集预测详情:")
    train_predictions = trainer.predict(train_dataset)
    train_preds = train_predictions.predictions.argmax(-1)
    # 使用 softmax 将 logits 转换为概率
    train_exp_preds = np.exp(train_predictions.predictions - np.max(train_predictions.predictions, axis=-1, keepdims=True))
    train_probs = train_exp_preds / train_exp_preds.sum(axis=-1, keepdims=True)
    train_probs = train_probs.max(axis=-1)
    
    # 打印前20个样本的预测结果
    print(f"{'样本序号':<6} {'真实标签':<6} {'预测标签':<6} {'置信度':<10} {'claim'}")
    print("-" * 120)
    for i in range(min(20, len(train_texts))):
        true_lab = id2label[train_labels[i]]
        pred_lab = id2label[int(train_preds[i])]
        print(f"{i:<6} {true_lab:<6} {pred_lab:<6} {train_probs[i]:<10.4f} {train_texts[i][:50]}...")

    # 保存训练集预测结果到文件
    train_pred_df = pd.DataFrame({
        'id': train_ids,
        'claim': train_texts,
        'true_label': [id2label[l] for l in train_labels],
        'predicted_label': [id2label[int(p)] for p in train_preds],
        'confidence': train_probs.tolist()
    })
    train_pred_output_path = f"{config.output_dir}/train_predictions.csv"
    train_pred_df.to_csv(train_pred_output_path, index=False, encoding='utf-8')
    print(f"\n训练集完整预测结果已保存到: {train_pred_output_path}")
    
    return trainer, model, tokenizer


# ==================== 5. 模型加载与预测函数 ====================
def load_filter_model(model_dir: str):
    """加载训练好的模型用于后续过滤"""
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def predict_with_confidence(model, tokenizer, texts: list, device, batch_size=32):
    """
    批量预测并返回预测结果和置信度
    
    返回:
        predictions: 预测标签列表 (0或1)
        confidences: 置信度列表 (0-1之间的概率)
        probabilities: 完整的两类概率分布
    """
    model.eval()
    predictions = []
    confidences = []
    probabilities = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenization
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 移动到设备
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # 提取结果
        batch_probs = probs.cpu().numpy()
        batch_preds = np.argmax(batch_probs, axis=1)
        batch_conf = np.max(batch_probs, axis=1)
        
        predictions.extend(batch_preds.tolist())
        confidences.extend(batch_conf.tolist())
        probabilities.extend(batch_probs.tolist())
    
    return predictions, confidences, probabilities


def filter_guessable_samples(
    model, tokenizer, device, 
    texts: list, 
    ground_truth_labels: list = None,
    threshold: float = 0.9
):
    """
    过滤可猜测样本
    
    参数:
        texts: 声明文本列表
        ground_truth_labels: 真实标签列表（可选，用于评估）
        threshold: 置信度阈值
    
    返回:
        filtered_texts: 保留的文本
        filtered_labels: 保留的标签（如果有输入）
        filter_mask: 布尔掩码，True表示保留
        details: 详细过滤信息
    """
    # 获取预测结果
    preds, confs, probs = predict_with_confidence(model, tokenizer, texts, device)
    
    # 确定保留条件
    if ground_truth_labels is not None:
        # 有真实标签：预测正确且高置信度的剔除
        is_correct = (np.array(preds) == np.array(ground_truth_labels))
        is_high_conf = (np.array(confs) >= threshold)
        keep_mask = ~(is_correct & is_high_conf)  # 剔除条件：正确且高置信度
    else:
        # 无真实标签：只根据置信度判断
        keep_mask = np.array(confs) < threshold
    
    # 应用过滤
    filtered_texts = [texts[i] for i in range(len(texts)) if keep_mask[i]]
    
    if ground_truth_labels is not None:
        filtered_labels = [ground_truth_labels[i] for i in range(len(texts)) if keep_mask[i]]
    else:
        filtered_labels = None
    
    # 统计信息
    details = {
        'total': len(texts),
        'kept': int(keep_mask.sum()),
        'removed': int((~keep_mask).sum()),
        'removal_rate': (~keep_mask).mean(),
        'predictions': preds,
        'confidences': confs,
        'keep_mask': keep_mask.tolist()
    }
    
    return filtered_texts, filtered_labels, keep_mask, details


# ==================== 6. 示例：完整工作流程 ====================
def example_usage():
    """
    完整使用示例
    注意：需要先准备好训练数据才能运行
    """
    
    # Step 1: 训练模型（只需要运行一次）
    trainer, model, tokenizer = main()
    
    # Step 2: 加载已训练好的模型（后续过滤时使用）
    model, tokenizer, device = load_filter_model("../../../../model/biobert-base-cased-v1.2")
    
    # Step 3: 准备待过滤的测试样本
    test_texts = [
        "CT shows an obstructing mass in the left lung",
        "MRI demonstrates no abnormal enhancement",
        "X-ray reveals cardiomegaly",
        # ... 更多样本
    ]
    
    # Step 4: 如果有真实标签，可以一起传入
    test_labels = [1, 0, 1]  # 1=True, 0=False
    
    # Step 5: 执行过滤
    filtered_texts, filtered_labels, mask, details = filter_guessable_samples(
        model, tokenizer, device,
        texts=test_texts,
        ground_truth_labels=test_labels,
        threshold=0.9
    )
    
    print(f"过滤结果: {details['removed']}/{details['total']} 条样本被剔除")
    print(f"剔除比例: {details['removal_rate']:.2%}")
    print(f"保留样本数: {details['kept']}")
    
    return filtered_texts, filtered_labels, details


if __name__ == "__main__":
    # 运行训练（需要先准备好数据文件）
    # 如果需要训练，取消下面的注释
    main()
    
    # 示例：直接使用过滤功能（需要先有训练好的模型）
    # example_usage()
    
    print("\n" + "=" * 50)
    print("使用说明")
    print("=" * 50)
    print("""
    1. 准备训练数据:
       创建CSV文件 data/train.csv 和 data/test.csv，包含两列: 'claim' 和 'label'
       label: 1 表示 True（声明正确），0 表示 False（声明错误）
    
    2. 修改 Config 中的路径（如果需要）
    
    3. 运行 main() 开始训练
    
    4. 训练完成后，模型保存在 ./biobert_filter_model
    
    5. 使用 load_filter_model() 加载模型，使用 filter_guessable_samples() 过滤测试集
    """)