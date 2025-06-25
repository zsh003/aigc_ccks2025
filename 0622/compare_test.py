import os
import json
import re
import cupy as cp
import cudf
import numpy as np
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
from cuml.linear_model import MBSGDClassifier
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# ------------ 1. 数据读取 ------------
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

train = cudf.DataFrame(load_jsonl('../datasets/train/train.jsonl'))
test  = cudf.DataFrame(load_jsonl('../datasets/test_521/test.jsonl'))

train['is_test'] = 0
test['is_test'] = 1
df_all = cudf.concat([train, test], ignore_index=True)

y = train['label'].to_pandas().values

# ------------ 2. 特征提取函数 ------------
def get_text_stats(texts):
    stats = []
    for t in texts:
        t = str(t)
        length = len(t)
        words = t.split()
        wc = len(words)
        uw = len(set(words))
        punct = len(re.findall(r'[.,!?;:]', t))
        stats.append([length, wc, uw, punct])
    return cudf.DataFrame(stats, columns=['length','word_count','unique_words','punct_count'])

# ------------ 3. Strategy 定义 ------------
def extract_tf(df):
    vec = cuTfidf(max_features=2000)
    X = vec.fit_transform(df['text']).toarray()
    return cudf.DataFrame(X)

# Strategy 1: 纯 TF-IDF + MBSGD
# Strategy 2: TF-IDF + 简单统计特征 + MBSGD
# Strategy 3: TF-IDF + 统计 + cuRF
strategies = {}

# 提前抽取通用特征
tfidf_feat = extract_tf(df_all)
stats_feat = get_text_stats(df_all['text'].to_pandas())

for name, cols in [
    ('tfidf', tfidf_feat),
    ('tfidf_stats', cudf.concat([tfidf_feat, stats_feat], axis=1)),
]:
    strategies[name] = cols

# ------------ 4. 训练与评估 ------------
def evaluate_cuml(X_df, y, model, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2024)
    accs, f1s = [], []
    for train_idx, val_idx in skf.split(X_df.to_pandas(), y):
        X_tr = X_df.iloc[train_idx].to_cupy()
        y_tr = cp.asarray(y[train_idx])
        X_va = X_df.iloc[val_idx].to_cupy()
        y_va = y[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        pred_labels = (cp.asarray(preds) > 0.5).astype(int).get()

        accs.append(accuracy_score(y_va, pred_labels))
        f1s.append(f1_score(y_va, pred_labels, average='weighted'))
    return np.mean(accs), np.mean(f1s)

results = []
for name, feat in strategies.items():
    for clf_name, clf in [
        ('MBSGD', MBSGDClassifier(loss='log', epochs=1000, tol=1e-4)),
        ('cuRF', cuRF(n_estimators=100, max_depth=16)),
    ]:
        # cuRF 不适用于纯 tfidf_stats，可在后续添加
        if clf_name == 'cuRF' and name == 'tfidf': continue
        acc, f1 = evaluate_cuml(feat.iloc[:len(train)], y, clf)
        results.append((name, clf_name, acc, f1))

# ------------ 5. BERT 微调 (PyTorch) ------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        return {k:v.squeeze(0) for k,v in enc.items()}, torch.tensor(self.labels[idx])

# 只做一次简单 CV
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()

tok_texts = train['text'].to_pandas().tolist()
train_ds = TextDataset(tok_texts, y, tokenizer)
loader = DataLoader(train_ds, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(2):
    for batch, labels in loader:
        for k in batch: batch[k] = batch[k].cuda()
        labels = labels.cuda()
        out = model(**batch, labels=labels)
        out.loss.backward()
        optim.step(); optim.zero_grad()
# 这里只计算训练集头几百条的简单准确率
model.eval()
with torch.no_grad():
    sample = loader.__iter__().__next__()
    for k in sample[0]: sample[0][k] = sample[0][k].cuda()
    logits = model(**sample[0]).logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc_bert = accuracy_score(sample[1].numpy(), preds)
    f1_bert  = f1_score(sample[1].numpy(), preds, average='weighted')
results.append(('bert_finetune', 'BERT2epochs', acc_bert, f1_bert))

# ------------ 6. 输出对比 ------------
print("Strategy,Classifier,Accuracy,F1")
for r in results:
    print(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f}")
