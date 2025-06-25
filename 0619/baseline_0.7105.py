import pandas as pd
import numpy as np
import json

# 训练集
with open('../datasets/train/train.jsonl', 'r', encoding='utf-8') as f:
    train = [json.loads(line) for line in f.readlines()]
    train = pd.DataFrame(train)
# 测试集
with open('../datasets/test_521/test.jsonl', 'r', encoding='utf-8') as f:
    test = [json.loads(line) for line in f.readlines()]
    test = pd.DataFrame(test)

# 合并
train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test],axis=0,ignore_index=True)
# Text 特征提取
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1500)  # 提取1500个TF-IDF特征
tfidf_df = tfidf.fit_transform(df_all['text'])
tfidf_feat = pd.DataFrame(tfidf_df.toarray(),columns=tfidf.get_feature_names_out())

# 数据集构造
tfidf_feat['label'] = df_all['label']
tfidf_feat['is_test'] = df_all['is_test']
tfidf_feat = tfidf_feat.fillna(0)

# 分离训练集，测试集
train_df = tfidf_feat[tfidf_feat['is_test']==0]
test_df = tfidf_feat[tfidf_feat['is_test']==1]

feats = [col for col in test_df if col not in ['id','label','is_test']]                                                                                                                                                                                      
print('feats_num:',len(feats))

import gc
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import SGDClassifier

# 交叉验证
def cross_validate_score(model, train_df, y, cv, test_df):
    val_scores = []
    test_preds = np.zeros((test_df.shape[0],))
    oof_preds = np.zeros((train_df.shape[0],)) # 训练集OOF预测

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, y)):

        X_train, y_train = train_df.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        X_val, y_val = train_df.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
        
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]  
        val_preds = np.where(val_probs>0.5, 1, 0)

        val_score = f1_score(val_preds, y_val, average='weighted')  
        print(f'Fold {fold+1}: score = {val_score:.5f}')
        
        val_scores.append(val_score)
        
        oof_preds[val_idx] = val_probs  

        test_preds += model.predict_proba(test_df)[:, 1] / cv.get_n_splits()  

    mean_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    print(f'Mean Validation score: {mean_val_score:.7f}')
    print(f'Std Validation score: {std_val_score:.7f}')

    del X_train, y_train, X_val, y_val, model
    
    gc.collect()

    return val_scores, test_preds, oof_preds

model = SGDClassifier(
    n_iter_no_change = 10,
    max_iter = 1500, 
    tol = 1e-5, 
    loss = "modified_huber", # Huber损失函数
    random_state = 2024
)
target_col = 'label'
cv = KFold(5, shuffle=True, random_state=2024)  # 5-KFold
val_scores, test_preds, oof_preds = cross_validate_score(model, train_df[feats], train_df[target_col], cv, test_df[feats])

# 保存结果
labels = np.where(test_preds > 0.5, 1, 0)
with open("submit.txt", "w") as file:
    for label in labels:
        file.write(str(label) + "\n")