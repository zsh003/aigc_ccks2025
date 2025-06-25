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

# ========== GPU设备检测 ==========
def check_gpu():
    """检查GPU设备并打印信息"""
    print("="*50)
    print("GPU设备检测")
    print("="*50)
    
    # 检查CUDA是否可用
    try:
        import cupy as cp
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA可用: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"CUDA设备数量: {cp.cuda.runtime.getDeviceCount()}")
            print(f"当前CUDA设备: {cp.cuda.runtime.getDevice()}")
            print(f"设备名称: {cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name']}")
            print(f"GPU内存: {cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['totalGlobalMem'] / 1024**3:.2f} GB")
        else:
            print("CUDA不可用，将使用CPU")
    except Exception as e:
        print(f"GPU检测出错: {e}")
    
    # 检查cuDF
    try:
        print(f"cuDF版本: {cudf.__version__}")
    except:
        print("cuDF版本信息获取失败")
    
    # 检查cuML
    try:
        import cuml
        print(f"cuML版本: {cuml.__version__}")
    except:
        print("cuML版本信息获取失败")
    
    print("="*50)

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

def get_text_stats_gpu(text_series):
    """Generates statistical features using GPU acceleration with robust error handling."""
    print("Computing text statistics on GPU with robust handling...")
    
    # 确保输入是字符串类型并处理缺失值
    text_series = text_series.fillna('').astype('str')
    
    # 基础统计特征
    length = text_series.str.len()
    
    # 计算单词数量 - 使用更稳定的方法
    word_count = cudf.Series([0] * len(text_series))
    for i in range(len(text_series)):
        try:
            text = str(text_series.iloc[i])
            if text.strip():
                word_count.iloc[i] = len(text.split())
        except:
            pass
    
    # 计算平均词长
    avg_word_len = cudf.Series([0.0] * len(text_series))
    for i in range(len(text_series)):
        try:
            text = str(text_series.iloc[i])
            words = text.split()
            if len(words) > 0:
                total_chars = sum(len(word) for word in words)
                avg_word_len.iloc[i] = total_chars / len(words)
        except:
            pass
    
    # 标点符号统计 - 使用字符计数而不是正则表达式
    punct_count = cudf.Series([0] * len(text_series))
    punct_chars = ['.', ',', '!', '?', ';', ':']
    for i in range(len(text_series)):
        try:
            text = str(text_series.iloc[i])
            for char in punct_chars:
                punct_count.iloc[i] += text.count(char)
        except:
            pass
    
    # 唯一词比例
    unique_word_ratio = cudf.Series([0.0] * len(text_series))
    for i in range(len(text_series)):
        try:
            text = str(text_series.iloc[i])
            words = text.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                unique_word_ratio.iloc[i] = unique_ratio
        except:
            pass
    
    # 句子长度统计 - 简化为单词数/句子数
    sentence_count = cudf.Series([1] * len(text_series))  # 默认至少1个句子
    for i in range(len(text_series)):
        try:
            text = str(text_series.iloc[i])
            # 计算句子结束符号的数量
            sentence_ends = text.count('.') + text.count('!') + text.count('?')
            sentence_count.iloc[i] = max(1, sentence_ends + 1)  # 至少1个句子
        except:
            pass
    
    avg_sent_len = word_count / sentence_count
    avg_sent_len = avg_sent_len.fillna(0)
    
    # 比例特征
    punct_ratio = punct_count / length
    punct_ratio = punct_ratio.fillna(0)
    
    # 合并所有特征
    stats_df = cudf.DataFrame({
        'length': length,
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'punct_count': punct_count,
        'unique_word_ratio': unique_word_ratio,
        'avg_sent_len': avg_sent_len,
        'punct_ratio': punct_ratio
    })
    
    return stats_df

def feature_strategy_tfidf_only(df_all):
    """Strategy 1: TF-IDF features only."""
    print("Running feature strategy: TF-IDF only")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    
    # Save vectorizer for inference
    joblib.dump(vectorizer, '0622/tfidf_only_vectorizer.pkl')
    
    return cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])

def feature_strategy_tfidf_and_scaled_stats(df_all):
    """Strategy 2: TF-IDF + Scaled Statistical features."""
    print("Running feature strategy: TF-IDF + Scaled Stats")
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    tfidf_feat_df = cudf.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])
    
    # Stats - 使用GPU加速版本
    stats_feat = get_text_stats_gpu(df_all['text'])
    
    # Scale ALL features together using GPU
    all_unscaled_features = cudf.concat([tfidf_feat_df, stats_feat.reset_index(drop=True)], axis=1)
    
    # IMPORTANT: Feature Scaling on GPU
    scaler = StandardScaler()
    all_scaled_features = scaler.fit_transform(all_unscaled_features.astype(cp.float32))
    
    # Save vectorizer and scaler for inference
    os.makedirs('0622', exist_ok=True)
    joblib.dump(vectorizer, '0622/tfidf_stats_vectorizer.pkl')
    joblib.dump(scaler, '0622/tfidf_stats_scaler.pkl')

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
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
        # 使用GPU数组进行训练
        X_train = X.iloc[train_idx].to_cupy()
        y_train = y.iloc[train_idx].to_cupy()
        X_val = X.iloc[val_idx].to_cupy()
        y_val = y.iloc[val_idx].to_cupy()
        
        # Fit model on GPU
        model.fit(X_train, y_train)
        
        # Predict on validation set
        if hasattr(model, 'predict_proba'):
            pred_val = model.predict_proba(X_val)[:, 1]
            pred_val = (pred_val > 0.5).astype(cp.int32)
        else:
            pred_val = model.predict(X_val).astype(cp.int32)
        
        oof_preds[val_idx] = pred_val
        
        # Predict on test set
        if hasattr(model, 'predict_proba'):
            test_pred = model.predict_proba(X_test.to_cupy())[:, 1]
        else:
            test_pred = model.predict(X_test.to_cupy())
        test_preds += test_pred / skf.n_splits

        # Convert to NumPy for sklearn metrics
        y_val_np = cp.asnumpy(y_val)
        pred_val_np = cp.asnumpy(pred_val)
        acc = accuracy_score(y_val_np, pred_val_np)
        print(f"Fold {fold+1} Accuracy: {acc:.5f}")

    # Overall OOF (Out-of-Fold) evaluation
    y_true_np = y.to_pandas().values
    oof_preds_np = cp.asnumpy(oof_preds)
    oof_acc = accuracy_score(y_true_np, oof_preds_np)
    oof_f1 = f1_score(y_true_np, oof_preds_np, average='weighted')
    
    print("\n--- Overall OOF Performance ---")
    print(f"Accuracy: {oof_acc:.5f}")
    print(f"F1 Score (Weighted): {oof_f1:.5f}")
    print(classification_report(y_true_np, oof_preds_np))
    
    return oof_acc, oof_f1, test_preds

# ========== 4. 主执行流程 ==========

if __name__ == '__main__':
    # 检查GPU设备
    check_gpu()
    
    df_all = load_data()

    # Define models to test - 使用GPU加速的模型
    models = {
        "MBSGDClassifier": MBSGDClassifier(loss="log", penalty="l2", alpha=1e-4, epochs=1000, tol=1e-4, n_iter_no_change=10),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42),
        "NaiveBayes": MultinomialNB() # Only for non-negative features like TF-IDF
    }

    # Define feature strategies to test
    feature_strategies = {
        "TF-IDF Only": feature_strategy_tfidf_only,
        #"TF-IDF + Scaled Stats": feature_strategy_tfidf_and_scaled_stats
    }
    
    results = []

    for f_name, f_func in feature_strategies.items():
        print(f"\n{'='*20} {f_name} {'='*20}")
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

            # Save submission file for this strategy
            if hasattr(test_predictions, 'get'):
                submission_labels = (test_predictions.get() > 0.5).astype(int)
            else:
                submission_labels = (test_predictions > 0.5).astype(int)
            
            os.makedirs('0622', exist_ok=True)
            with open(f'0622/submission_{f_name.replace(" ", "_")}_{m_name}.txt', 'w') as f:
                for label in submission_labels:
                    f.write(f"{label}\n")

    # Print final comparison table
    print("\n\n" + "="*50)
    print("            STRATEGY COMPARISON RESULTS")
    print("="*50)
    results_df = pd.DataFrame(results).sort_values(by="CV Accuracy", ascending=False)
    print(results_df.to_string(index=False))
    
    # Save results to file
    results_df.to_csv('0622/comparison_results.csv', index=False)
    print(f"\nResults saved to 0622/comparison_results.csv")