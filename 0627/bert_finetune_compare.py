import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import gc

# ====== HF缓存配置（参考ref/fine_tune.py） ======
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)
# =========================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import check_gpu, load_data
from utils import save_model, load_model, save_probs, load_probs, save_evaluation_results, evaluate_probs, plot_confusion_matrix, plot_roc_curve

# 配置
OUTPUT_DIR = '0627/results'
MODELS_DIR = '0627/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
N_SPLITS = 2  # 交叉验证折数
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=2024)
BERT_MODEL = 'bert-base-uncased'
REPORT_TO = ['tensorboard']  # 可改为['wandb']或[]

# 分批推理辅助函数
def predict_probs(model, tokenizer, texts, batch_size=32, max_length=256):
    model.eval()
    all_probs = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            for k in enc:
                enc[k] = enc[k].to(device)
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[:,1].cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)

# 数据加载
check_gpu()
print("\n加载数据...")
df_all_gpu = load_data()
df_all = df_all_gpu.to_pandas() if hasattr(df_all_gpu, 'to_pandas') else df_all_gpu
train_df = df_all[df_all['is_test']==0].copy()
test_df = df_all[df_all['is_test']==1].copy()

# 只用text和label
X = train_df['text'].tolist()
y = train_df['label'].values
test_texts = test_df['text'].tolist()

# Tokenizer
print(f"加载BERT模型: {BERT_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=True, cache_dir=HF_CACHE_DIR)
except OSError:
    print(f"Base model not found locally. Downloading from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, cache_dir=HF_CACHE_DIR)

def preprocess(texts, tokenizer, max_length=256):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

# 5折交叉验证微调BERT
fold_oof = np.zeros(len(X))
test_probs = np.zeros(len(test_texts))
fold_metrics = []
all_start = time.time()
for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
    fold_start = time.time()
    print(f"\n{'='*10} Fold {fold+1} / {CV.get_n_splits()} {'='*10}")
    # 显存清理与加速
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    gc.collect()
    fold_model_dir = os.path.join(MODELS_DIR, f'bert_fold{fold}')
    os.makedirs(fold_model_dir, exist_ok=True)
    # 划分数据
    X_train = [X[i] for i in train_idx]
    y_train = y[train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = y[val_idx]
    # Tokenize
    train_encodings = tokenizer(X_train, padding=True, truncation=True, max_length=256)
    val_encodings = tokenizer(X_val, padding=True, truncation=True, max_length=256)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # 强制long类型
            return item
        def __len__(self):
            return len(self.labels)
    train_dataset = SimpleDataset(train_encodings, y_train)
    val_dataset = SimpleDataset(val_encodings, y_val)
    # 检查模型是否已保存
    model_exist = os.path.exists(os.path.join(fold_model_dir, 'pytorch_model.bin'))
    if model_exist:
        print(f"检测到已保存模型，直接加载: {fold_model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(fold_model_dir)
        tokenizer_fold = AutoTokenizer.from_pretrained(fold_model_dir)
    else:
        print(f"未检测到已保存模型，开始训练...")
        # 在训练前添加环境配置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试
            torch.cuda.empty_cache()  # 清空GPU缓存
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL,
            num_labels=2,
            cache_dir=HF_CACHE_DIR,
            #torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=fold_model_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,  # 添加梯度裁剪
            warmup_steps=50,    # 引入学习率预热
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(fold_model_dir, 'logs'),
            logging_steps=50,
            load_best_model_at_end=True,
            save_total_limit=2, # Keep limited checkpoints
            metric_for_best_model="eval_loss",
            report_to=REPORT_TO,
            disable_tqdm=True,
            bf16=True if torch.cuda.is_available() else False,  # 优先使用bfloat16
            #fp16=True if not torch.cuda.is_available() else False,
            tf32=True if torch.cuda.is_available() else False, # 启用TF32加速
            gradient_checkpointing=True, # 启用检查点节省显存
            optim="adamw_torch",
            dataloader_pin_memory=True,  # 启用内存锁页
            dataloader_num_workers=8,    # 增加数据加载进程数 4->8
            eval_accumulation_steps=1,   # 减少评估时的内存占用
        )
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {"accuracy": acc, "f1": f1}
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        # 保存模型
        save_model(model, tokenizer, MODELS_DIR, f'bert_fold{fold}')
        tokenizer_fold = tokenizer
    # 分批推理
    val_probs = predict_probs(model, tokenizer_fold, X_val, batch_size=32)
    fold_oof[val_idx] = val_probs
    test_prob = predict_probs(model, tokenizer_fold, test_texts, batch_size=32)
    test_probs += test_prob / CV.get_n_splits()
    # 评估
    metrics = evaluate_probs(y_val, val_probs)
    fold_metrics.append(metrics)
    print(f"Fold{fold} acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")
    # 保存本折评估结果
    save_evaluation_results(metrics, fold_model_dir, prefix=f'bert_fold{fold}')
    # 显存彻底释放
    del model
    torch.cuda.empty_cache()
    gc.collect()
    fold_end = time.time()
    print(f"Fold{fold} 总用时: {fold_end - fold_start:.2f} 秒")
# 保存概率
save_probs(fold_oof, test_probs, OUTPUT_DIR, prefix="bert")
# 评估整体
overall_metrics = evaluate_probs(y, fold_oof)
save_evaluation_results(overall_metrics, OUTPUT_DIR, prefix="bert")
print(f"\nBERT微调整体评估: acc={overall_metrics['accuracy']:.4f}, f1={overall_metrics['f1']:.4f}, auc={overall_metrics['auc']:.4f}")
# 保存提交
submission_path = os.path.join(OUTPUT_DIR, 'submission_bert.txt')
with open(submission_path, 'w') as f:
    for label in (test_probs > 0.5).astype(int):
        f.write(f"{label}\n")
print(f"预测标签已保存到: {submission_path}")
# 混淆矩阵/ROC
plot_confusion_matrix(y, (fold_oof > 0.5).astype(int), 'BERT', os.path.join(OUTPUT_DIR, 'cm_bert.png'))
plot_roc_curve(y, fold_oof, 'BERT', os.path.join(OUTPUT_DIR, 'roc_bert.png'))
# 与0626结果对比
compare_path = os.path.join(OUTPUT_DIR, 'model_compare_with_0626.csv')
old_path = '../0626/results/model_compare_all.csv'
if os.path.exists(old_path):
    old_df = pd.read_csv(old_path, index_col=0)
    compare_df = old_df.copy()
    compare_df.loc['BERT'] = [overall_metrics['accuracy'], overall_metrics['f1'], overall_metrics['auc']]
    compare_df.to_csv(compare_path)
    print(f"与0626模型对比表已保存: {compare_path}")
    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(10, len(compare_df)*0.7), 6))
    x = np.arange(len(compare_df))
    width = 0.5
    plt.bar(x, compare_df['F1-Score'], label='F1-Score')
    plt.xticks(x, compare_df.index, rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('BERT与0626模型F1对比')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bert_vs_0626_f1.png'))
    plt.close()
    print('BERT与0626模型F1对比图已保存')
all_end = time.time()
print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")
print("\n如需可视化训练过程，请在终端运行: tensorboard --logdir 0627/models/ ，然后浏览器访问 http://localhost:6006") 