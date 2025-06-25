import os
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import json
import re
import joblib
import matplotlib.pyplot as plt

# cuML imports
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.preprocessing import StandardScaler
from cuml.linear_model import MBSGDClassifier
from cuml.ensemble import RandomForestClassifier
from cuml.naive_bayes import MultinomialNB

# Scikit-learn for metrics and splitting
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# ========== 1. 数据读取与预处理 ==========
def load_data():
    """Loads train and test data into cuDF DataFrames."""
    print("Loading data...")
    # 使用你的数据集路径
    with open('../datasets/train/train.jsonl', 'r', encoding='utf-8') as f:
        train_list = [json.loads(line) for line in f.readlines()]
    with open('../datasets/test_521/test.jsonl', 'r', encoding='utf-8') as f:
        test_list = [json.loads(line) for line in f.readlines()]
    
    train_df = cudf.DataFrame(train_list)
    test_df = cudf.DataFrame(test_list)
    
    train_df['is_test'] = 0
    test_df['is_test'] = 1
    
    df_all = cudf.concat([train_df, test_df], axis=0, ignore_index=True)
    print("Data loaded successfully.")
    return df_all

# ========== 2. 特征工程策略 ==========

def get_text_stats(text_series_pd):
    """Generates statistical features from a pandas Series of text."""
    stats_list = []
    for text in text_series_pd:
        text = str(text)
        words = text.split()
        word_count = len(words)
        length = len(text)
        
        if word_count == 0:
            stats_list.append([0] * 9)
            continue
            
        avg_word_len = np.mean([len(w) for w in words])
        punct_count = len(re.findall(r'[.,!?;:]', text))
        unique_word_ratio = len(set(words)) / word_count
        avg_sent_len = np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if s.strip()]) if text else 0
        punct_ratio = punct_count / length
        # 你可以继续在这里添加更高级的特征，如可读性分数等
        
        stats_list.append([
            length, word_count, avg_word_len, punct_count, 
            unique_word_ratio, avg_sent_len, punct_ratio
        ])
        
    stat_names = [
        'length', 'word_count', 'avg_word_len', 'punct_count',
        'unique_word_ratio', 'avg_sent_len', 'punct_ratio'
    ]
    return cudf.DataFrame(stats_list, columns=stat_names)

def feature_strategy_tfidf_only(df_all):
    """Strategy 1: TF-IDF features only."""
    print("Running feature strategy: TF-IDF only")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    
    # Save vectorizer for inference
    joblib.dump(vectorizer, 'tfidf_only_vectorizer.pkl')
    
    return cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])

def feature_strategy_tfidf_and_scaled_stats(df_all):
    """Strategy 2: TF-IDF + Scaled Statistical features."""
    print("Running feature strategy: TF-IDF + Scaled Stats")
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    tfidf_feat_df = cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])
    
    # Stats
    stats_feat = get_text_stats(df_all['text'].to_pandas())
    
    # Scale ALL features together
    all_unscaled_features = cudf.concat([tfidf_feat_df, stats_feat.reset_index(drop=True)], axis=1)
    
    # IMPORTANT: Feature Scaling
    scaler = StandardScaler()
    all_scaled_features = scaler.fit_transform(all_unscaled_features.astype(cp.float32))
    
    # Save vectorizer and scaler for inference
    joblib.dump(vectorizer, 'tfidf_stats_vectorizer.pkl')
    joblib.dump(scaler, 'tfidf_stats_scaler.pkl')

    return cudf.DataFrame(all_scaled_features, columns=all_unscaled_features.columns)


# ========== 3. 模型训练与评估框架 ==========

def run_experiment(features_df, df_all, model, model_name):
    """Runs a cross-validation experiment for a given model and features."""
    print(f"\n===== Running Experiment for model: {model_name} =====")
    
    # Prepare data
    features_df['label'] = df_all['label']
    features_df['is_test'] = df_all['is_test']
    features_df = features_df.fillna(0)
    
    train_df = features_df[features_df['is_test']==0].reset_index(drop=True)
    test_df = features_df[features_df['is_test']==1].reset_index(drop=True)
    
    feats = [col for col in train_df.columns if col not in ['label', 'is_test']]
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cp.zeros(train_df.shape[0])
    test_preds = cp.zeros(test_df.shape[0])
    
    X = train_df[feats]
    y = train_df['label']
    X_test = test_df[feats]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.to_numpy())):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        pred_val = model.predict(X_val).astype(cp.int32)
        oof_preds[val_idx] = pred_val
        
        # Predict on test set
        test_preds += model.predict(X_test) / skf.n_splits

        acc = accuracy_score(y_val.get(), pred_val.get())
        print(f"Fold {fold+1} Accuracy: {acc:.5f}")

    # Overall OOF (Out-of-Fold) evaluation
    oof_acc = accuracy_score(y.get(), (oof_preds.get()))
    oof_f1 = f1_score(y.get(), oof_preds.get(), average='weighted')
    
    print("\n--- Overall OOF Performance ---")
    print(f"Accuracy: {oof_acc:.5f}")
    print(f"F1 Score (Weighted): {oof_f1:.5f}")
    print(classification_report(y.get(), oof_preds.get()))
    
    return oof_acc, oof_f1, test_preds.get()

# ========== 4. 主执行流程 ==========

if __name__ == '__main__':
    df_all = load_data()

    # Define models to test
    models = {
        "MBSGDClassifier": MBSGDClassifier(loss="log", penalty="l2", alpha=1e-4, epochs=1000, tol=1e-4, n_iter_no_change=10),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42),
        "NaiveBayes": MultinomialNB() # Only for non-negative features like TF-IDF
    }

    # Define feature strategies to test
    feature_strategies = {
        "TF-IDF Only": feature_strategy_tfidf_only,
        "TF-IDF + Scaled Stats": feature_strategy_tfidf_and_scaled_stats
    }
    
    results = []

    for f_name, f_func in feature_strategies.items():
        features_df = f_func(df_all)
        
        # Test each model with the current feature set
        for m_name, model in models.items():
            # Naive Bayes can't handle negative values from StandardScaler
            if f_name == "TF-IDF + Scaled Stats" and m_name == "NaiveBayes":
                print(f"\nSkipping {m_name} for scaled features.")
                continue

            acc, f1, test_predictions = run_experiment(features_df.copy(), df_all.copy(), model, m_name)
            
            results.append({
                "Feature Strategy": f_name,
                "Model": m_name,
                "CV Accuracy": acc,
                "CV F1-Score": f1
            })

            # Optionally, save submission file for this strategy
            # submission_labels = (test_predictions > 0.5).astype(int)
            # sub_df = pd.DataFrame({'label': submission_labels})
            # sub_df.to_csv(f'submission_{f_name}_{m_name}.csv', index=False, header=False)


    # Print final comparison table
    print("\n\n" + "="*50)
    print("            STRATEGY COMPARISON RESULTS")
    print("="*50)
    results_df = pd.DataFrame(results).sort_values(by="CV Accuracy", ascending=False)
    print(results_df.to_string(index=False))