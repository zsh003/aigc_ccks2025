import cudf
import cupy as cp
import numpy as np
import json
import re
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import gc

# 读取数据
with open('../datasets/train/train.jsonl', 'r', encoding='utf-8') as f:
    train = [json.loads(line) for line in f.readlines()]
    train = cudf.DataFrame(train)
with open('../datasets/test_521/test.jsonl', 'r', encoding='utf-8') as f:
    test = [json.loads(line) for line in f.readlines()]
    test = cudf.DataFrame(test)

train['is_test'] = 0
test['is_test'] = 1
df_all = cudf.concat([train, test], axis=0, ignore_index=True)

# 文本统计特征
def get_text_stats(text):
    text = str(text)
    length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    punct_count = len(re.findall(r'[.,!?;:]', text))
    digit_count = len(re.findall(r'\d', text))
    upper_count = sum(1 for c in text if c.isupper())
    return [length, word_count, avg_word_len, punct_count, digit_count, upper_count]

stats = df_all['text'].to_pandas().apply(get_text_stats)
stats = cudf.DataFrame(stats.tolist(), columns=['length', 'word_count', 'avg_word_len', 'punct_count', 'digit_count', 'upper_count'])

# TF-IDF特征（高维+N-gram）
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_df = tfidf.fit_transform(df_all['text'])
tfidf_feat = cudf.DataFrame(tfidf_df.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_df.shape[1])])

# 合并所有特征
all_feat = cudf.concat([tfidf_feat, stats.reset_index(drop=True)], axis=1)
all_feat['label'] = df_all['label']
all_feat['is_test'] = df_all['is_test']
all_feat = all_feat.fillna(0)

# 分离训练集和测试集
train_df = all_feat[all_feat['is_test']==0].reset_index(drop=True)
test_df = all_feat[all_feat['is_test']==1].reset_index(drop=True)
feats = [col for col in train_df.columns if col not in ['label','is_test']]

# Stacking模型
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
train_oof = cp.zeros(train_df.shape[0])
test_pred_rf = cp.zeros(test_df.shape[0])
test_pred_lr = cp.zeros(test_df.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df[feats].to_pandas(), train_df['label'].to_pandas())):
    X_tr = train_df.iloc[list(tr_idx)].reset_index(drop=True)[feats]
    y_tr = train_df.iloc[list(tr_idx)].reset_index(drop=True)['label']
    X_val = train_df.iloc[list(val_idx)].reset_index(drop=True)[feats]
    y_val = train_df.iloc[list(val_idx)].reset_index(drop=True)['label']
    # RandomForestClassifier (替代LGBM)
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=2024)
    rf_model.fit(X_tr, y_tr)
    val_pred_rf = rf_model.predict_proba(X_val).to_pandas().values[:,1]
    test_pred_rf += cp.asarray(rf_model.predict_proba(test_df[feats]).to_pandas().values[:,1]) / skf.n_splits
    # LogisticRegression
    lr_model = LogisticRegression(max_iter=1000, random_state=2024)
    lr_model.fit(X_tr, y_tr)
    val_pred_lr = lr_model.predict_proba(X_val).to_pandas().values[:,1]
    test_pred_lr += cp.asarray(lr_model.predict_proba(test_df[feats]).to_pandas().values[:,1]) / skf.n_splits
    # OOF融合
    train_oof[val_idx] = 0.6*val_pred_rf + 0.4*val_pred_lr
    val_f1 = f1_score(y_val.to_pandas(), (train_oof[val_idx].get()>0.5).astype(int), average='weighted')
    print(f'Fold {fold+1} F1: {val_f1:.5f}')
    gc.collect()

# 测试集融合
final_test_pred = 0.6*test_pred_rf + 0.4*test_pred_lr
labels = (final_test_pred.get() > 0.5).astype(int)

with open("0620/submit_gpu.txt", "w") as file:
    for label in labels:
        file.write(str(label) + "\n") 