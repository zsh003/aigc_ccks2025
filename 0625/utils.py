import os
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

def extract_tfidf_features(df_all, max_features=1500, ngram_range=(1,2), save_path=None):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    tfidf_df = cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import joblib
        joblib.dump(vectorizer, save_path)
    return tfidf_df, vectorizer

def train_and_eval_gpu_model(train_df, test_df, feats, y_col, model, model_name, results_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cp.zeros(train_df.shape[0])
    test_preds = cp.zeros(test_df.shape[0])
    X = train_df[feats]
    y = train_df[y_col]
    X_test = test_df[feats]
    for fold, (train_idx, val_idx) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
        X_train = X.iloc[train_idx].to_cupy()
        y_train = y.iloc[train_idx].to_cupy()
        X_val = X.iloc[val_idx].to_cupy()
        y_val = y.iloc[val_idx].to_cupy()
        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            pred_val = model.predict_proba(X_val)[:, 1]
            pred_val = (pred_val > 0.5).astype(cp.int32)
        else:
            pred_val = model.predict(X_val).astype(cp.int32)
        oof_preds[val_idx] = pred_val
        if hasattr(model, 'predict_proba'):
            test_pred = model.predict_proba(X_test.to_cupy())[:, 1]
        else:
            test_pred = model.predict(X_test.to_cupy())
        test_preds += test_pred / skf.n_splits
        y_val_np = cp.asnumpy(y_val)
        pred_val_np = cp.asnumpy(pred_val)
        acc = accuracy_score(y_val_np, pred_val_np)
        print(f"Fold {fold+1} Accuracy: {acc:.5f}")
    y_true_np = y.to_pandas().values
    oof_preds_np = cp.asnumpy(oof_preds)
    oof_acc = accuracy_score(y_true_np, oof_preds_np)
    oof_f1 = f1_score(y_true_np, oof_preds_np, average='weighted')
    print("\n--- Overall OOF Performance ---")
    print(f"Accuracy: {oof_acc:.5f}")
    print(f"F1 Score (Weighted): {oof_f1:.5f}")
    print(classification_report(y_true_np, oof_preds_np))
    # 保存结果
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'oof_{model_name}.npy'), oof_preds_np)
    np.save(os.path.join(results_dir, f'test_{model_name}.npy'), cp.asnumpy(test_preds))
    return oof_acc, oof_f1, test_preds

def plot_result_bar(result_dict, save_path):
    plt.figure(figsize=(8,5))
    names = list(result_dict.keys())
    values = list(result_dict.values())
    plt.bar(names, values)
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'可视化结果已保存到: {save_path}') 