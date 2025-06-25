import os
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

def extract_tfidf_features(df_all, max_features=1500, ngram_range=(1,1), save_path=None):
    """
    Extracts TF-IDF features, matching the baseline logic.
    Default ngram_range is now (1,1).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    tfidf_df = cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import joblib
        joblib.dump(vectorizer, save_path)
    return tfidf_df, vectorizer

def train_and_eval_baseline_gpu(train_df, test_df, feats, y_col, model):
    """
    GPU-accelerated training and evaluation, mimicking the 0619 baseline script.
    Uses KFold and F1-score.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=2024)
    test_preds = cp.zeros(test_df.shape[0], dtype=cp.float32)
    oof_preds = cp.zeros(train_df.shape[0], dtype=cp.float32)
    val_scores = []

    X = train_df[feats]
    y = train_df[y_col]
    X_test = test_df[feats]

    for fold, (train_idx, val_idx) in enumerate(cv.split(X.to_pandas(), y.to_pandas())):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_val, y_val = X.iloc[list(val_idx)], y.iloc[list(val_idx)]
        
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        
        val_preds = (val_probs > 0.5).astype(cp.int32)
        val_score = f1_score(cp.asnumpy(y_val), cp.asnumpy(val_preds), average='weighted')
        val_scores.append(val_score)
        print(f'Fold {fold+1} F1 score: {val_score:.5f}')
        
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds += cp.asarray(test_probs) / cv.get_n_splits()
        
    print(f'\nMean Validation F1 score: {np.mean(val_scores):.7f}')
    print(f'Std Validation F1 score: {np.std(val_scores):.7f}')

    return val_scores, test_preds, oof_preds
    
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

def plot_result_bar(result_dict, title, ylabel, save_path):
    """Plots a bar chart from a dictionary of results, with pretty x labels."""
    import matplotlib.pyplot as plt
    # 自动调整宽度，最小8，最多每个模型0.7宽度
    plt.figure(figsize=(max(8, len(result_dict)*0.7), 5))
    # 缩短标签
    names = [name if len(name) <= 12 else name[:10]+'…' for name in result_dict.keys()]
    values = list(result_dict.values())
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') 
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Visualization "{title}" saved to: {save_path}')

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plots and saves a confusion matrix."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion Matrix for {model_name} saved to: {save_path}')

def plot_roc_curves(y_true, oof_probs_dict, save_path):
    """Plots ROC curves for multiple models on the same axes."""
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 7))
    for model_name, oof_probs in oof_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, oof_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'ROC Curves saved to: {save_path}') 