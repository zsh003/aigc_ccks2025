import os
import json
import re
import cupy as cp
import cudf
import numpy as np
import pandas as pd
import joblib
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
from cuml.linear_model import MBSGDClassifier
from cuml.ensemble import RandomForestClassifier as cuRF
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import optuna
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt

# ========== 1. 数据读取 ==========
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]
train = pd.DataFrame(load_jsonl('../datasets/train/train.jsonl'))
test  = pd.DataFrame(load_jsonl('../datasets/test_521/test.jsonl'))
train['is_test'], test['is_test'] = 0, 1
df_all = pd.concat([train, test], ignore_index=True)
y_train = train['label'].values

# ========== 2. 特征提取 ==========
# 2.1 TF-IDF
word_vec = cuTfidf(max_features=3000, analyzer='word')
char_vec = cuTfidf(max_features=2000, analyzer='char_wb', ngram_range=(3,5))
X_word = word_vec.fit_transform(df_all['text']).toarray()
X_char = char_vec.fit_transform(df_all['text']).toarray()
# 2.2 统计特征
import textstat
stats = []
for t in df_all['text']:
    t = str(t)
    stats.append([
        len(t), len(t.split()),
        textstat.flesch_reading_ease(t), textstat.smog_index(t),
        len(set(t.split()))/max(len(t.split()),1),
        sum(1 for _ in re.finditer(r"(\b\w+\b).*(\b\w+\b)", t))
    ])
stats = np.array(stats)
# 2.3 困惑度
pt_tok = GPT2Tokenizer.from_pretrained('gpt2')
pt_mdl = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
def perplexity(text):
    enc = pt_tok(text, return_tensors='pt', truncation=True, max_length=512).to('cuda')
    with torch.no_grad(): outputs = pt_mdl(**enc, labels=enc['input_ids'])
    return float(torch.exp(outputs.loss))
perps = [perplexity(t) for t in df_all['text']]

# 合并特征
X = np.hstack([X_word, X_char, stats, np.array(perps)[:,None]])
X = cudf.DataFrame(X)
X_train = X.iloc[:len(train)].reset_index(drop=True)
X_test  = X.iloc[len(train):].reset_index(drop=True)

# ========== 3. 数据增强 & 重采样 ==========
sm = SMOTE(random_state=2024)
X_res, y_res = sm.fit_resample(X_train.to_pandas(), y_train)
X_res = cudf.DataFrame(X_res)

# ========== 4. 模型训练 & 评估 ==========
results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
models = {}
for tag, X_use, y_use in [('orig', X_train, y_train), ('smote', X_res, y_res)]:
    clfs = {
        'MBSGD': MBSGDClassifier(loss='log', epochs=1000, tol=1e-4),
        'cuRF': cuRF(n_estimators=200, max_depth=16),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
    }
    for name, clf in clfs.items():
        accs, f1s = [], []
        for tr, va in skf.split(X_use.to_pandas(), y_use):
            X_tr, X_va = X_use.iloc[tr], X_use.iloc[va]
            y_tr, y_va = y_use[tr], y_use[va]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            if hasattr(preds, 'to_array'): preds = preds.to_array()
            preds = (cp.asarray(preds)>0.5).astype(int).get() if isinstance(preds, cp.ndarray) else preds
            accs.append(accuracy_score(y_va, preds))
            f1s.append(f1_score(y_va, preds, average='weighted'))
        avg_acc, avg_f1 = np.mean(accs), np.mean(f1s)
        results.append((tag, name, avg_acc, avg_f1))
        models[f"{tag}_{name}"] = clf

# ========== 5. 集成 (Stacking) ==========
est = [('lgb', lgb.LGBMClassifier(n_estimators=100)),
       ('xgb', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'))]
stack = StackingClassifier(estimators=est, final_estimator=lgb.LGBMClassifier(), cv=3)
accs, f1s = [], []
for tr, va in skf.split(X_train.to_pandas(), y_train):
    stack.fit(X_train.iloc[tr].to_pandas(), y_train[tr])
    preds = stack.predict(X_train.iloc[va].to_pandas())
    accs.append(accuracy_score(y_train[va], preds))
    f1s.append(f1_score(y_train[va], preds, average='weighted'))
results.append(('stack', 'stacking', np.mean(accs), np.mean(f1s)))
models['stacking'] = stack

# ========== 6. 超参调优 (Optuna-LGB) ==========
def objective(trial):
    param = {'n_estimators': trial.suggest_int('n_estimators', 50,300),
             'max_depth': trial.suggest_int('max_depth',3,16)}
    mdl = lgb.LGBMClassifier(**param)
    accs=[]
    for tr,va in skf.split(X_train.to_pandas(), y_train):
        mdl.fit(X_train.iloc[tr].to_pandas(), y_train[tr])
        preds=mdl.predict(X_train.iloc[va].to_pandas())
        accs.append(accuracy_score(y_train[va], preds))
    return np.mean(accs)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
results.append(('optuna','lgb', study.best_value, None))
models['optuna_lgb'] = lgb.LGBMClassifier(**best_params).fit(X_train.to_pandas(), y_train)

# ========== 7. 可视化 & 结果保存 ==========
res_df = pd.DataFrame(results, columns=['data','model','accuracy','f1'])
res_df.to_csv('0622/results_summary.csv', index=False)


# 绘图：Accuracy & F1
plt.figure()
res_df.plot(x='model', y='accuracy', kind='bar')
plt.title('Accuracy Comparison')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')

plt.figure()
res_df.plot(x='model', y='f1', kind='bar')
plt.title('F1 Comparison')
plt.tight_layout()
plt.savefig('f1_comparison.png')

# 保存所有模型
os.makedirs('0622/models', exist_ok=True)
for name, mdl in models.items():
    joblib.dump(mdl, os.path.join('0622/models', f'{name}.pkl'))

print("All done. Results and models saved.")
