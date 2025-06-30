import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import gc
import pickle

# ====== NLTK路径配置 ======
nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..', 'nltk_data'))
import nltk
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
# ====== HF缓存配置 ======
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
from utils import (
    extract_aigc_features, save_model, load_model, save_probs, load_probs, save_evaluation_results,
    evaluate_probs, plot_confusion_matrix, plot_roc_curve, tta_augment_texts, generate_adversarial_examples
)

# 配置
OUTPUT_DIR = '0628/results'
MODELS_DIR = '0628/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
N_SPLITS = 5
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=2024)
MODEL_LIST = [
    ('bert-base-uncased', 'BERT'),
    ('microsoft/deberta-base', 'DeBERTa'),
    ('roberta-base', 'RoBERTa'),
]
REPORT_TO = ['tensorboard']

# 保存预测结果的函数
def save_predictions_to_txt(probs, output_path, threshold=0.5):
    """将预测概率保存为txt文件"""
    predictions = (probs >= threshold).astype(int)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"预测结果已保存: {output_path}")

def save_probabilities_to_txt(probs, output_path):
    """将预测概率保存为txt文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for prob in probs:
            f.write(f"{prob:.6f}\n")
    print(f"预测概率已保存: {output_path}")

# 1. 数据加载与特征提取
check_gpu()
print("\n加载数据...")
df_all_gpu = load_data()
df_all = df_all_gpu.to_pandas() if hasattr(df_all_gpu, 'to_pandas') else df_all_gpu
train_df = df_all[df_all['is_test']==0].copy()
test_df = df_all[df_all['is_test']==1].copy()
X = train_df['text'].tolist()
y = train_df['label'].values
test_texts = test_df['text'].tolist()
print("特征工程...")
feat_path = os.path.join(OUTPUT_DIR, 'features_all_feat.pkl')
feat_names_path = os.path.join(OUTPUT_DIR, 'features_feat_names.pkl')
if os.path.exists(feat_path) and os.path.exists(feat_names_path):
    print("检测到已保存特征，直接加载...")
    with open(feat_path, 'rb') as f:
        all_feat = pickle.load(f)
    with open(feat_names_path, 'rb') as f:
        feat_names = pickle.load(f)
else:
    t0 = time.time()
    all_feat, feat_names = extract_aigc_features(df_all)
    with open(feat_path, 'wb') as f:
        pickle.dump(all_feat, f)
    with open(feat_names_path, 'wb') as f:
        pickle.dump(feat_names, f)
    print(f"特征提取完成，用时 {time.time()-t0:.2f} 秒")
train_feat = all_feat.iloc[:len(train_df)].values
test_feat = all_feat.iloc[len(train_df):].values

all_start = time.time()
# 2. 多模型训练与推理（普通训练）
def load_tokenizer_model(model_ckpt, local_only=True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, local_files_only=local_only, cache_dir=HF_CACHE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2, local_files_only=local_only, cache_dir=HF_CACHE_DIR)
    except Exception as e:
        if local_only:
            print(f"未在本地找到{model_ckpt}，尝试从HuggingFace下载...")
            return load_tokenizer_model(model_ckpt, local_only=False)
        else:
            raise e
    return tokenizer, model

def is_model_saved(model_dir):
    config_exists = os.path.exists(os.path.join(model_dir, 'config.json'))
    bin_exists = os.path.exists(os.path.join(model_dir, 'pytorch_model.bin'))
    safetensors_exists = os.path.exists(os.path.join(model_dir, 'model.safetensors'))
    return config_exists and (bin_exists or safetensors_exists)

def is_probs_saved(output_dir, prefix):
    oof_path = os.path.join(output_dir, f"oof_{prefix}.npy")
    test_path = os.path.join(output_dir, f"test_{prefix}.npy")
    return os.path.exists(oof_path) and os.path.exists(test_path)

model_probs = {}
model_test_probs = {}
for model_ckpt, model_tag in MODEL_LIST:
    model_start = time.time()
    print(f"\n{'='*10} {model_tag} 训练与推理 {'='*10}")
    prefix = model_tag
    if is_probs_saved(OUTPUT_DIR, prefix):
        print(f"检测到已保存概率，直接加载: {prefix}")
        fold_oof, test_probs = load_probs(OUTPUT_DIR, prefix)
        model_probs[model_tag] = fold_oof
        model_test_probs[model_tag] = test_probs
        continue
    tokenizer, _ = load_tokenizer_model(model_ckpt, local_only=True)
    fold_oof = np.zeros(len(X))
    test_probs = np.zeros(len(test_texts))
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        fold_start = time.time()
        print(f"Fold {fold+1}/{N_SPLITS}")
        torch.cuda.empty_cache(); gc.collect()
        fold_model_dir = os.path.join(MODELS_DIR, f'{model_tag}_fold{fold}')
        os.makedirs(fold_model_dir, exist_ok=True)
        if is_model_saved(fold_model_dir):
            print(f"检测到已保存模型，直接加载: {fold_model_dir}")
            model = AutoModelForSequenceClassification.from_pretrained(fold_model_dir)
            fold_tokenizer = AutoTokenizer.from_pretrained(fold_model_dir)
        else:
            X_train = [X[i] for i in train_idx]
            y_train = y[train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = y[val_idx]
            train_encodings = tokenizer(X_train, padding=True, truncation=True, max_length=256)
            val_encodings = tokenizer(X_val, padding=True, truncation=True, max_length=256)
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, features, labels):
                    self.encodings = encodings
                    self.features = features
                    self.labels = labels
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                    return item
                def __len__(self):
                    return len(self.labels)
            train_dataset = SimpleDataset(train_encodings, train_feat[train_idx], y_train)
            val_dataset = SimpleDataset(val_encodings, train_feat[val_idx], y_val)
            _, model = load_tokenizer_model(model_ckpt, local_only=True)
            fold_tokenizer = tokenizer  # 使用外层的tokenizer
            training_args = TrainingArguments(
                output_dir=fold_model_dir,
                num_train_epochs=1,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                gradient_accumulation_steps=8,  # 从4增加到8，保持有效batch_size
                max_grad_norm=1.0,
                warmup_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir=os.path.join(fold_model_dir, 'logs'),
                logging_steps=50,
                load_best_model_at_end=True,
                save_total_limit=2,
                metric_for_best_model="eval_loss",
                report_to=REPORT_TO,
                disable_tqdm=True,
                #bf16=True if torch.cuda.is_available() else False,
                fp16=True if torch.cuda.is_available() else False,
                tf32=True if torch.cuda.is_available() else False,
                gradient_checkpointing=True,
                optim="adamw_torch",
                dataloader_pin_memory=True,
                dataloader_num_workers=16,
                eval_accumulation_steps=2,
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
            save_model(model, fold_tokenizer, MODELS_DIR, f'{model_tag}_fold{fold}')
            
            # 训练后立即清理内存
            del trainer, train_dataset, val_dataset, train_encodings, val_encodings
            del X_train, y_train, X_val, y_val
            torch.cuda.empty_cache()
            gc.collect()
        def predict_probs(model, tokenizer, texts, batch_size=32):
            model.eval()
            all_probs = []
            device = next(model.parameters()).device
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    for k in enc:
                        enc[k] = enc[k].to(device)
                    outputs = model(**enc)
                    probs = torch.softmax(outputs.logits, dim=-1)[:,1].cpu().numpy()
                    all_probs.append(probs)
                # 每个batch后清理
                del enc, outputs
                torch.cuda.empty_cache()
            return np.concatenate(all_probs)
        X_val = [X[i] for i in val_idx]
        val_probs = predict_probs(model, fold_tokenizer, X_val, batch_size=32)
        fold_oof[val_idx] = val_probs
        test_prob = predict_probs(model, fold_tokenizer, test_texts, batch_size=32)
        test_probs += test_prob / N_SPLITS
        
        # 每折结束后彻底清理内存
        del model, fold_tokenizer, val_probs, test_prob, X_val
        torch.cuda.empty_cache()
        gc.collect()
        
        fold_end = time.time()
        print(f"{model_tag} Fold{fold} 用时: {fold_end - fold_start:.2f} 秒")
    model_probs[model_tag] = fold_oof
    model_test_probs[model_tag] = test_probs
    save_probs(fold_oof, test_probs, OUTPUT_DIR, prefix=model_tag)
    metrics = evaluate_probs(y, fold_oof)
    save_evaluation_results(metrics, OUTPUT_DIR, prefix=model_tag)
    
    # 立即保存预测结果
    model_pred_path = os.path.join(OUTPUT_DIR, f'{model_tag.lower()}_predictions.txt')
    model_prob_path = os.path.join(OUTPUT_DIR, f'{model_tag.lower()}_probabilities.txt')
    save_predictions_to_txt(test_probs, model_pred_path)
    save_probabilities_to_txt(test_probs, model_prob_path)
    
    model_end = time.time()
    print(f"{model_tag} 全部用时: {model_end - model_start:.2f} 秒")

# ========== 对抗训练 ==========
adv_model_probs = {}
adv_model_test_probs = {}
for model_ckpt, model_tag in MODEL_LIST:
    model_start = time.time()
    print(f"\n{'='*10} {model_tag} 对抗训练 {'='*10}")
    prefix = f'Adv_{model_tag}'
    if is_probs_saved(OUTPUT_DIR, prefix):
        print(f"检测到已保存概率，直接加载: {prefix}")
        fold_oof, test_probs = load_probs(OUTPUT_DIR, prefix)
        adv_model_probs[model_tag] = fold_oof
        adv_model_test_probs[model_tag] = test_probs
        continue
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    fold_oof = np.zeros(len(X))
    test_probs = np.zeros(len(test_texts))
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        fold_start = time.time()
        print(f"[Adv] Fold {fold+1}/{N_SPLITS}")
        # 每折开始前清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        fold_model_dir = os.path.join(MODELS_DIR, f'Adv_{model_tag}_fold{fold}')
        os.makedirs(fold_model_dir, exist_ok=True)
        if is_model_saved(fold_model_dir):
            print(f"检测到已保存模型，直接加载: {fold_model_dir}")
            model = AutoModelForSequenceClassification.from_pretrained(fold_model_dir)
            fold_tokenizer = AutoTokenizer.from_pretrained(fold_model_dir)
        else:
            X_train = [X[i] for i in train_idx]
            y_train = y[train_idx]
            adv_X_train = generate_adversarial_examples(X_train, method='synonym', n_aug=1)
            adv_y_train = list(y_train) * 1
            X_train_all = X_train + adv_X_train
            y_train_all = list(y_train) + adv_y_train
            train_feat_all = np.concatenate([train_feat[train_idx], train_feat[train_idx]], axis=0)
            X_val = [X[i] for i in val_idx]
            y_val = y[val_idx]
            
            # 编码数据
            train_encodings = tokenizer(X_train_all, padding=True, truncation=True, max_length=256)
            val_encodings = tokenizer(X_val, padding=True, truncation=True, max_length=256)
            
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, features, labels):
                    self.encodings = encodings
                    self.features = features
                    self.labels = labels
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                    return item
                def __len__(self):
                    return len(self.labels)
            
            train_dataset = SimpleDataset(train_encodings, train_feat_all, y_train_all)
            val_dataset = SimpleDataset(val_encodings, train_feat[val_idx], y_val)
            
            model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
            fold_tokenizer = tokenizer  # 使用外层的tokenizer
            
            training_args = TrainingArguments(
                output_dir=fold_model_dir,
                num_train_epochs=1,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                gradient_accumulation_steps=8,  # 从4增加到8，保持有效batch_size
                max_grad_norm=1.0,
                warmup_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir=os.path.join(fold_model_dir, 'logs'),
                logging_steps=50,
                load_best_model_at_end=True,
                save_total_limit=2,
                metric_for_best_model="eval_loss",
                report_to=REPORT_TO,
                disable_tqdm=True,
                #bf16=True if torch.cuda.is_available() else False,
                #fp16=True if torch.cuda.is_available() else False,
                tf32=True if torch.cuda.is_available() else False,
                gradient_checkpointing=True,
                optim="adamw_torch",
                dataloader_pin_memory=True,
                dataloader_num_workers=16,
                eval_accumulation_steps=2,
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
            save_model(model, fold_tokenizer, MODELS_DIR, f'Adv_{model_tag}_fold{fold}')
            
            # 训练后立即清理内存
            del trainer, train_dataset, val_dataset, train_encodings, val_encodings
            del X_train, y_train, adv_X_train, adv_y_train, X_train_all, y_train_all, train_feat_all
            del X_val, y_val
            torch.cuda.empty_cache()
            gc.collect()
        
        def predict_probs(model, tokenizer, texts, batch_size=32):
            model.eval()
            all_probs = []
            device = next(model.parameters()).device
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    for k in enc:
                        enc[k] = enc[k].to(device)
                    outputs = model(**enc)
                    probs = torch.softmax(outputs.logits, dim=-1)[:,1].cpu().numpy()
                    all_probs.append(probs)
                # 每个batch后清理
                del enc, outputs
                torch.cuda.empty_cache()
            return np.concatenate(all_probs)
        
        X_val = [X[i] for i in val_idx]
        val_probs = predict_probs(model, fold_tokenizer, X_val, batch_size=32)
        fold_oof[val_idx] = val_probs
        test_prob = predict_probs(model, fold_tokenizer, test_texts, batch_size=32)
        test_probs += test_prob / N_SPLITS
        
        # 每折结束后彻底清理内存
        del model, fold_tokenizer, val_probs, test_prob, X_val
        torch.cuda.empty_cache()
        gc.collect()
        
        fold_end = time.time()
        print(f"Adv_{model_tag} Fold{fold} 用时: {fold_end - fold_start:.2f} 秒")
    
    adv_model_probs[model_tag] = fold_oof
    adv_model_test_probs[model_tag] = test_probs
    save_probs(fold_oof, test_probs, OUTPUT_DIR, prefix=f'Adv_{model_tag}')
    metrics = evaluate_probs(y, fold_oof)
    save_evaluation_results(metrics, OUTPUT_DIR, prefix=f'Adv_{model_tag}')
    
    # 立即保存预测结果
    model_pred_path = os.path.join(OUTPUT_DIR, f'adv_{model_tag.lower()}_predictions.txt')
    model_prob_path = os.path.join(OUTPUT_DIR, f'adv_{model_tag.lower()}_probabilities.txt')
    save_predictions_to_txt(test_probs, model_pred_path)
    save_probabilities_to_txt(test_probs, model_prob_path)
    
    model_end = time.time()
    print(f"Adv_{model_tag} 全部用时: {model_end - model_start:.2f} 秒")
# 对抗集成
adv_ensemble_start = time.time()
adv_ensemble_oof = np.mean([adv_model_probs[m] for m in adv_model_probs], axis=0)
adv_ensemble_test = np.mean([adv_model_test_probs[m] for m in adv_model_test_probs], axis=0)
save_probs(adv_ensemble_oof, adv_ensemble_test, OUTPUT_DIR, prefix='Adv_Ensemble')
metrics = evaluate_probs(y, adv_ensemble_oof)
save_evaluation_results(metrics, OUTPUT_DIR, prefix='Adv_Ensemble')

# 立即保存预测结果
adv_ensemble_pred_path = os.path.join(OUTPUT_DIR, 'adv_ensemble_predictions.txt')
adv_ensemble_prob_path = os.path.join(OUTPUT_DIR, 'adv_ensemble_probabilities.txt')
save_predictions_to_txt(adv_ensemble_test, adv_ensemble_pred_path)
save_probabilities_to_txt(adv_ensemble_test, adv_ensemble_prob_path)

print(f"\n对抗集成模型评估: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")
adv_ensemble_end = time.time()
print(f"对抗集成用时: {adv_ensemble_end - adv_ensemble_start:.2f} 秒")
# 3. 集成推理（概率平均）
ensemble_start = time.time()
ensemble_oof = np.mean([model_probs[m] for m in model_probs], axis=0)
ensemble_test = np.mean([model_test_probs[m] for m in model_test_probs], axis=0)
save_probs(ensemble_oof, ensemble_test, OUTPUT_DIR, prefix='Ensemble')
metrics = evaluate_probs(y, ensemble_oof)
save_evaluation_results(metrics, OUTPUT_DIR, prefix='Ensemble')

# 立即保存预测结果
ensemble_pred_path = os.path.join(OUTPUT_DIR, 'ensemble_predictions.txt')
ensemble_prob_path = os.path.join(OUTPUT_DIR, 'ensemble_probabilities.txt')
save_predictions_to_txt(ensemble_test, ensemble_pred_path)
save_probabilities_to_txt(ensemble_test, ensemble_prob_path)

print(f"\n集成模型评估: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")
ensemble_end = time.time()
print(f"普通集成用时: {ensemble_end - ensemble_start:.2f} 秒")
# 4. 与0626/0627对比
compare_path = os.path.join(OUTPUT_DIR, 'model_compare_with_0626_0627.csv')
old_path1 = '0626/results/model_compare_all.csv'
old_path2 = '0627/results/model_compare_with_0626.csv'
if os.path.exists(old_path1):
    old_df = pd.read_csv(old_path1, index_col=0)
else:
    old_df = None
if os.path.exists(old_path2):
    bert_df = pd.read_csv(old_path2, index_col=0)
else:
    bert_df = None
compare_rows = []

# 添加普通模型结果
for name in list(model_probs.keys()) + ['Ensemble']:
    new_metrics = evaluate_probs(y, model_probs[name] if name in model_probs else ensemble_oof)
    compare_rows.append({'Model': name, 'Accuracy': new_metrics['accuracy'], 'F1-Score': new_metrics['f1'], 'AUC': new_metrics['auc']})

# 添加对抗模型结果
for name in list(adv_model_probs.keys()):
    new_metrics = evaluate_probs(y, adv_model_probs[name])
    compare_rows.append({'Model': f'Adv_{name}', 'Accuracy': new_metrics['accuracy'], 'F1-Score': new_metrics['f1'], 'AUC': new_metrics['auc']})

# 添加对抗集成结果
adv_ensemble_metrics = evaluate_probs(y, adv_ensemble_oof)
compare_rows.append({'Model': 'Adv_Ensemble', 'Accuracy': adv_ensemble_metrics['accuracy'], 'F1-Score': adv_ensemble_metrics['f1'], 'AUC': adv_ensemble_metrics['auc']})

# 添加历史结果
if old_df is not None:
    for name in old_df.index:
        compare_rows.append({'Model': name, 'Accuracy': old_df.loc[name, 'Accuracy'], 'F1-Score': old_df.loc[name, 'F1-Score'], 'AUC': old_df.loc[name, 'AUC']})
if bert_df is not None:
    for name in bert_df.index:
        if name not in [r['Model'] for r in compare_rows]:
            compare_rows.append({'Model': name, 'Accuracy': bert_df.loc[name, 'Accuracy'], 'F1-Score': bert_df.loc[name, 'F1-Score'], 'AUC': bert_df.loc[name, 'AUC']})

compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(compare_path, index=False)
print(f"模型对比表已保存: {compare_path}")
# 5. 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(max(10, len(compare_df)*0.7), 6))
x = np.arange(len(compare_df))
plt.bar(x, compare_df['F1-Score'], label='F1-Score')
plt.xticks(x, compare_df['Model'], rotation=45, ha='right')
plt.ylabel('F1-Score')
plt.title('多模型集成与0626/0627对比')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_vs_0626_0627_f1.png'))
plt.close()
print('集成与0626/0627模型F1对比图已保存')
print("\n主流程完成。后续可补充对抗训练、TTA等增强实验。")
# 6. TTA集成实验
tta_start = time.time()
print("\n===== 测试时增强（TTA）集成实验 =====")
tta_model_probs = {}
tta_model_test_probs = {}
for model_ckpt, model_tag in MODEL_LIST:
    print(f"TTA: {model_tag}")
    prefix = f'TTA_{model_tag}'
    if is_probs_saved(OUTPUT_DIR, prefix):
        print(f"检测到已保存概率，直接加载: {prefix}")
        tta_oof, tta_test = load_probs(OUTPUT_DIR, prefix)
        tta_model_probs[model_tag] = tta_oof
        tta_model_test_probs[model_tag] = tta_test
        continue
    
    tta_oof = np.zeros(len(X))
    tta_test_fold_probs = []
    
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        print(f"TTA Fold {fold+1}/{N_SPLITS}")
        fold_model_dir = os.path.join(MODELS_DIR, f'{model_tag}_fold{fold}')
        if not is_model_saved(fold_model_dir):
            print(f"TTA跳过未训练模型: {fold_model_dir}")
            continue
            
        model = AutoModelForSequenceClassification.from_pretrained(fold_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(fold_model_dir)
        model.eval()
        device = next(model.parameters()).device
        
        # 优化：使用更大的batch_size进行TTA推理
        def tta_predict_batch(texts, batch_size=32):
            """批量TTA预测，提高速度"""
            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_aug_texts = []
                for text in batch_texts:
                    # 减少TTA增强次数从4次到2次，提高速度
                    aug_texts = tta_augment_texts(text, n_aug=2)
                    batch_aug_texts.extend(aug_texts)
                
                # 批量编码和推理
                enc = tokenizer(batch_aug_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    for k in enc:
                        enc[k] = enc[k].to(device)
                    outputs = model(**enc)
                    probs = torch.softmax(outputs.logits, dim=-1)[:,1].cpu().numpy()
                
                # 重新组织概率，每个原始文本对应2个增强版本
                for j in range(0, len(probs), 2):
                    if j+1 < len(probs):
                        avg_prob = np.mean(probs[j:j+2])
                    else:
                        avg_prob = probs[j]
                    all_probs.append(avg_prob)
                
                # 清理内存
                del enc, outputs
                torch.cuda.empty_cache()
            
            return np.array(all_probs)
        
        # OOF: 只对val_idx做TTA
        print(f"  OOF TTA预测...")
        val_texts = [X[i] for i in val_idx]
        val_tta_probs = tta_predict_batch(val_texts, batch_size=16)
        for i, idx in enumerate(val_idx):
            tta_oof[idx] = val_tta_probs[i]
        
        # TEST: 每折模型对全部test_texts做TTA
        print(f"  Test TTA预测...")
        test_tta_probs = tta_predict_batch(test_texts, batch_size=16)
        tta_test_fold_probs.append(test_tta_probs)
        
        # 清理内存
        del model, tokenizer, val_tta_probs, test_tta_probs
        torch.cuda.empty_cache()
        gc.collect()
    
    # TEST: 对所有折的TTA概率平均
    if tta_test_fold_probs:
        tta_test = np.mean(tta_test_fold_probs, axis=0)
    else:
        tta_test = np.zeros(len(test_texts))
    
    tta_model_probs[model_tag] = tta_oof
    tta_model_test_probs[model_tag] = tta_test
    save_probs(tta_oof, tta_test, OUTPUT_DIR, prefix=f'TTA_{model_tag}')
    metrics = evaluate_probs(y, tta_oof)
    save_evaluation_results(metrics, OUTPUT_DIR, prefix=f'TTA_{model_tag}')
    
    # 立即保存预测结果
    model_pred_path = os.path.join(OUTPUT_DIR, f'tta_{model_tag.lower()}_predictions.txt')
    model_prob_path = os.path.join(OUTPUT_DIR, f'tta_{model_tag.lower()}_probabilities.txt')
    save_predictions_to_txt(tta_test, model_pred_path)
    save_probabilities_to_txt(tta_test, model_prob_path)
    
    print(f"TTA {model_tag} 完成: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")

# TTA集成
tta_ensemble_oof = np.mean([tta_model_probs[m] for m in tta_model_probs], axis=0)
tta_ensemble_test = np.mean([tta_model_test_probs[m] for m in tta_model_test_probs], axis=0)
save_probs(tta_ensemble_oof, tta_ensemble_test, OUTPUT_DIR, prefix='TTA_Ensemble')
metrics = evaluate_probs(y, tta_ensemble_oof)
save_evaluation_results(metrics, OUTPUT_DIR, prefix='TTA_Ensemble')

# 立即保存预测结果
tta_ensemble_pred_path = os.path.join(OUTPUT_DIR, 'tta_ensemble_predictions.txt')
tta_ensemble_prob_path = os.path.join(OUTPUT_DIR, 'tta_ensemble_probabilities.txt')
save_predictions_to_txt(tta_ensemble_test, tta_ensemble_pred_path)
save_probabilities_to_txt(tta_ensemble_test, tta_ensemble_prob_path)

print(f"TTA集成模型评估: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")

# 将TTA集成结果添加到对比表
compare_rows.append({'Model': 'TTA_Ensemble', 'Accuracy': metrics['accuracy'], 'F1-Score': metrics['f1'], 'AUC': metrics['auc']})
compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(compare_path, index=False)
print(f"更新后的模型对比表已保存: {compare_path}")

tta_ensemble_end = time.time()
print(f"TTA集成用时: {tta_ensemble_end - tta_start:.2f} 秒")
all_end = time.time()
print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

# 生成最佳模型的预测结果（基于F1分数）
best_model_name = compare_df.loc[compare_df['F1-Score'].idxmax(), 'Model']
print(f"\n最佳模型（基于F1分数）: {best_model_name}")

if best_model_name == 'Ensemble':
    best_probs = ensemble_test
elif best_model_name == 'Adv_Ensemble':
    best_probs = adv_ensemble_test
elif best_model_name == 'TTA_Ensemble':
    best_probs = tta_ensemble_test
elif best_model_name.startswith('Adv_'):
    base_name = best_model_name[4:]  # 去掉'Adv_'前缀
    best_probs = adv_model_test_probs[base_name]
elif best_model_name.startswith('TTA_'):
    base_name = best_model_name[4:]  # 去掉'TTA_'前缀
    best_probs = tta_model_test_probs[base_name]
else:
    best_probs = model_test_probs[best_model_name]

best_pred_path = os.path.join(OUTPUT_DIR, 'best_model_predictions.txt')
best_prob_path = os.path.join(OUTPUT_DIR, 'best_model_probabilities.txt')
save_predictions_to_txt(best_probs, best_pred_path)
save_probabilities_to_txt(best_probs, best_prob_path)

print(f"\n所有预测结果文件已生成完成！")
print(f"预测结果保存在: {OUTPUT_DIR}")
print(f"最佳模型预测结果: {best_pred_path}")
print(f"最佳模型概率结果: {best_prob_path}") 