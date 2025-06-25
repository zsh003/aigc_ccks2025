import os
import cudf
import cupy as cp
import numpy as np
import json
import re
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import MBSGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# ========== 特征工程 ==========
def get_text_stats(text):
    text = str(text)
    length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    punct_count = len(re.findall(r'[.,!?;:]', text))
    digit_count = len(re.findall(r'\d', text))
    upper_count = sum(1 for c in text if c.isupper())
    # AIGC相关特征
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count else 0
    repeat_rate = 1 - unique_word_ratio
    avg_sent_len = np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if s.strip()]) if text else 0
    punct_ratio = punct_count / length if length else 0
    digit_ratio = digit_count / length if length else 0
    upper_ratio = upper_count / length if length else 0
    special_char_count = len(re.findall(r'[@#$%^&*()_+=\[\]{}|;:<>~`]', text))
    special_char_ratio = special_char_count / length if length else 0
    return [length, word_count, avg_word_len, punct_count, digit_count, upper_count,
            unique_word_ratio, repeat_rate, avg_sent_len, punct_ratio, digit_ratio, upper_ratio, special_char_ratio]

stat_names = [
    'length', 'word_count', 'avg_word_len', 'punct_count', 'digit_count', 'upper_count',
    'unique_word_ratio', 'repeat_rate', 'avg_sent_len', 'punct_ratio', 'digit_ratio', 'upper_ratio', 'special_char_ratio'
]

# ========== 数据读取 ==========
# Ensure the paths are correct for your environment
with open('../datasets/train/train.jsonl', 'r', encoding='utf-8') as f:
    train = [json.loads(line) for line in f.readlines()]
    train = cudf.DataFrame(train)
with open('../datasets/test_521/test.jsonl', 'r', encoding='utf-8') as f:
    test = [json.loads(line) for line in f.readlines()]
    test = cudf.DataFrame(test)

train['is_test'] = 0
test['is_test'] = 1
df_all = cudf.concat([train, test], axis=0, ignore_index=True)

# 统计特征+AIGC特征
txt_stats = df_all['text'].to_pandas().apply(get_text_stats)
txt_stats = cudf.DataFrame(txt_stats.tolist(), columns=stat_names)

# TF-IDF特征
vectorizer = TfidfVectorizer(max_features=1500)
tfidf_df = vectorizer.fit_transform(df_all['text'])
tfidf_feat = cudf.DataFrame(tfidf_df.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_df.shape[1])])

# 合并特征
all_feat = cudf.concat([tfidf_feat, txt_stats.reset_index(drop=True)], axis=1)
all_feat['label'] = df_all['label']
all_feat['is_test'] = df_all['is_test']
all_feat = all_feat.fillna(0)

# 分离训练集和测试集
train_df = all_feat[all_feat['is_test']==0].reset_index(drop=True)
test_df = all_feat[all_feat['is_test']==1].reset_index(drop=True)
feats = [col for col in train_df.columns if col not in ['label','is_test']]

# ========== 模型训练与评估 ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
train_oof = cp.zeros(train_df.shape[0])
test_pred = cp.zeros(test_df.shape[0])
feature_importance = cp.zeros(len(feats))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df[feats].to_pandas(), train_df['label'].to_pandas())):
    # Convert training and validation data to CuPy arrays explicitly for model fitting and prediction
    X_tr_cp = train_df.iloc[list(tr_idx)].reset_index(drop=True)[feats].to_cupy()
    y_tr_cp = train_df.iloc[list(tr_idx)].reset_index(drop=True)['label'].to_cupy()
    X_val_cp = train_df.iloc[list(val_idx)].reset_index(drop=True)[feats].to_cupy()
    y_val_cp = train_df.iloc[list(val_idx)].reset_index(drop=True)['label'].to_cupy()
    
    model = MBSGDClassifier(loss="log", epochs=1500, tol=1e-5, n_iter_no_change=10) 
    
    model.fit(X_tr_cp, y_tr_cp) # Fit with CuPy arrays
    
    # Manually calculate decision scores using coef_ and intercept_
    # model.coef_ is typically (1, n_features) for binary classification,
    # so we transpose it to (n_features, 1) for matrix multiplication
    # model.intercept_ is typically (1,)
    val_decision_scores = X_val_cp @ model.coef_.T + model.intercept_
    
    # Apply sigmoid and flatten the result to a 1D array
    val_pred = cp.asarray(1 / (1 + cp.exp(-val_decision_scores.ravel())))

    train_oof[val_idx] = val_pred
    
    # Explicitly convert test_df[feats] (a cuDF DataFrame) to a CuPy array
    test_feats_cp = test_df[feats].to_cupy()
    # Manually calculate decision scores for test data
    test_decision_scores = test_feats_cp @ model.coef_.T + model.intercept_
    # Apply sigmoid and flatten the result
    test_pred += cp.asarray(1 / (1 + cp.exp(-test_decision_scores.ravel()))) / skf.n_splits
    
    # 累加特征重要性
    if hasattr(model, 'coef_'):
        feature_importance += cp.abs(model.coef_.ravel())
    
    # Convert CuPy arrays to NumPy for sklearn.metrics functions
    val_f1 = f1_score(y_val_cp.get(), (val_pred.get()>0.5).astype(int), average='weighted')
    val_acc = accuracy_score(y_val_cp.get(), (val_pred.get()>0.5).astype(int))
    print(f'Fold {fold+1} F1: {val_f1:.5f}, ACC: {val_acc:.5f}')

# ========== 评价与可视化 ==========
print("\n===== 交叉验证整体结果 =====")
oof_pred = (train_oof.get()>0.5).astype(int)
y_true = train_df['label'].to_pandas() # Still using original train_df label for overall evaluation
print(classification_report(y_true, oof_pred))

# 特征重要性排名
feature_importance = feature_importance.get() / skf.n_splits
feat_imp = sorted(zip(feats, feature_importance), key=lambda x: -x[1])
print("\n===== 特征重要性排名 =====")
for i, (f, imp) in enumerate(feat_imp[:30]):
    print(f"{i+1}. {f}: {imp:.4f}")

# 可视化
plt.figure(figsize=(10,6))
plt.bar([f for f, _ in feat_imp[:20]], [imp for _, imp in feat_imp[:20]])
plt.xticks(rotation=90)
plt.title('Top 20 Feature Importance')
plt.tight_layout()
os.makedirs('0621', exist_ok=True) # Ensure directory exists for saving plot
plt.savefig('0621/feature_importance.png')
plt.show()

# ========== 保存模型 ==========
os.makedirs('0621/model', exist_ok=True)
joblib.dump(model, '0621/model/sgd_aigc_model.pkl')
joblib.dump(vectorizer, '0621/model/tfidf_vectorizer.pkl')
print('模型和特征工程器已保存到0621/model/')

# ========== 预测并保存结果 ==========
labels = (test_pred.get() > 0.5).astype(int)
with open("0621/submit.txt", "w") as file:
    for label in labels:
        file.write(str(label) + "\n")
print('预测结果已保存到0621/submit.txt')
