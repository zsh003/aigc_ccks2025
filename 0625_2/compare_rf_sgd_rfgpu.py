import os
import sys
import time
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from tqdm import tqdm
import joblib
import pickle

# 将项目根目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入自定义模块和模型
from base_utils import check_gpu, load_data
from utils import plot_confusion_matrix, plot_roc_curves, plot_result_bar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.linear_model import SGDClassifier
from cuml.ensemble import RandomForestClassifier as CumlRF

# 全局配置
OUTPUT_DIR = '0625_2'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
CV = KFold(n_splits=5, shuffle=True, random_state=2024)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_cpu_experiment(model, model_name, train_df, test_df):
    """在CPU上运行实验的通用函数，支持每fold模型保存和加载"""
    print("\n" + "="*20 + f" Running {model_name} " + "="*20)
    submission_path = os.path.join(RESULTS_DIR, f'submission_{model_name}.txt')

    y = train_df['label']
    X = train_df.drop(columns=['label'])
    X_test = test_df

    oof_probs = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])

    pbar = tqdm(enumerate(CV.split(X, y)), total=CV.get_n_splits(), desc=f"{model_name} Folds")
    for fold, (train_idx, val_idx) in pbar:
        fold_model_path = os.path.join(MODELS_DIR, f'{model_name}_fold{fold}.pkl')
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if os.path.exists(fold_model_path):
            fold_model = joblib.load(fold_model_path)
        else:
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_train, y_train)
            joblib.dump(fold_model, fold_model_path)
        fold_oof_probs = fold_model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = fold_oof_probs
        test_preds += fold_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        fold_f1 = f1_score(y_val, (fold_oof_probs > 0.5).astype(int), average='weighted')
        pbar.set_postfix({'F1': f'{fold_f1:.5f}'})
        print(f"Fold {fold+1} F1 score: {fold_f1:.5f}")
    # 保存预测结果txt
    submission_labels = (test_preds > 0.5).astype(int)
    with open(submission_path, "w") as f:
        for label in submission_labels:
            f.write(f"{label}\n")
    print(f"预测结果已保存: {submission_path}")
    return oof_probs, test_preds

def run_gpu_rf_experiment(train_df, test_df):
    """在GPU上运行RandomForest实验，支持每fold模型保存和加载"""
    print("\n" + "="*20 + " Running RandomForest (GPU) " + "="*20)
    model_name = 'RandomForest_GPU'
    submission_path = os.path.join(RESULTS_DIR, f'submission_{model_name}.txt')

    train_df_gpu = cudf.from_pandas(train_df)
    test_df_gpu = cudf.from_pandas(test_df)
    y_gpu = train_df_gpu['label']
    X_gpu = train_df_gpu.drop(columns=['label'])
    X_test_gpu = test_df_gpu

    oof_probs = cp.zeros(train_df_gpu.shape[0])
    test_preds = cp.zeros(test_df_gpu.shape[0])

    pbar = tqdm(enumerate(CV.split(X_gpu.to_pandas(), y_gpu.to_pandas())), total=CV.get_n_splits(), desc="RF_GPU Folds")
    for fold, (train_idx, val_idx) in pbar:
        fold_model_path = os.path.join(MODELS_DIR, f'{model_name}_fold{fold}.pkl')
        X_train, y_train = X_gpu.iloc[list(train_idx)], y_gpu.iloc[list(train_idx)]
        X_val, y_val = X_gpu.iloc[list(val_idx)], y_gpu.iloc[list(val_idx)]
        if os.path.exists(fold_model_path):
            with open(fold_model_path, 'rb') as f:
                fold_model = pickle.load(f)
        else:
            fold_model = CumlRF(n_estimators=100, random_state=2024)
            fold_model.fit(X_train, y_train)
            with open(fold_model_path, 'wb') as f:
                pickle.dump(fold_model, f)
        fold_oof_probs = fold_model.predict_proba(X_val)[1]
        oof_probs[list(val_idx)] = fold_oof_probs
        test_preds += fold_model.predict_proba(X_test_gpu)[1] / CV.get_n_splits()
        fold_f1 = f1_score(cp.asnumpy(y_val), cp.asnumpy(fold_oof_probs > 0.5).astype(int), average='weighted')
        pbar.set_postfix({'F1': f'{fold_f1:.5f}'})
        print(f"Fold {fold+1} F1 score: {fold_f1:.5f}")
    # 保存预测结果txt
    submission_labels = (cp.asnumpy(test_preds) > 0.5).astype(int)
    with open(submission_path, "w") as f:
        for label in submission_labels:
            f.write(f"{label}\n")
    print(f"预测结果已保存: {submission_path}")
    return cp.asnumpy(oof_probs), cp.asnumpy(test_preds)

def main():
    """主执行函数，用于对比模型"""
    check_gpu()
    # 1. 加载和预处理数据
    start_time = time.time()
    print("\nLoading data into Pandas DataFrame...")
    df_all_gpu = load_data()
    df_all = df_all_gpu.to_pandas()
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    # 2. 特征工程
    print("\nExtracting TF-IDF features with Scikit-learn...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=1500)
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    features_df = pd.DataFrame(tfidf_feat.toarray(), columns=vectorizer.get_feature_names_out())
    print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")
    train_df = features_df[df_all['is_test']==0].copy()
    train_df['label'] = df_all[df_all['is_test']==0]['label'].to_numpy()
    test_df = features_df[df_all['is_test']==1].copy()
    # 3. 运行实验
    models_to_run = {
        "SGD_CPU": (run_cpu_experiment, SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024)),
        "RandomForest_CPU": (run_cpu_experiment, SklearnRF(n_estimators=100, random_state=2024, n_jobs=-1)),
        "RandomForest_GPU": (run_gpu_rf_experiment, None) # GPU function handles its own model creation
    }
    models_oof_probs = {}
    model_times = {}
    for name, (experiment_func, model) in models_to_run.items():
        start_time = time.time()
        if model is not None:
            oof_probs, _ = experiment_func(model, name, train_df, test_df)
        else: # For GPU RF which creates model inside
            oof_probs, _ = experiment_func(train_df, test_df)
        model_times[name] = time.time() - start_time
        models_oof_probs[name] = oof_probs
    # 4. 评估和比较
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    y_true = train_df['label']
    results_summary = {}
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10} | {'Time (s)':<10}")
    print("-"*55)
    for name, probs in models_oof_probs.items():
        acc = accuracy_score(y_true, (probs > 0.5).astype(int))
        f1 = f1_score(y_true, (probs > 0.5).astype(int), average='weighted')
        exec_time = model_times[name]
        results_summary[name] = {'Accuracy': acc, 'F1-Score': f1, 'Time': exec_time}
        print(f"{name:<20} | {acc:<10.4f} | {f1:<10.4f} | {exec_time:<10.2f}")
    # 5. 可视化
    print("\nGenerating visualizations...")
    # 指标对比
    acc_dict = {name: data['Accuracy'] for name, data in results_summary.items()}
    plot_result_bar(acc_dict, 'Model Accuracy Comparison', 'Accuracy', os.path.join(RESULTS_DIR, 'accuracy_comparison.png'))
    time_dict = {name: data['Time'] for name, data in results_summary.items()}
    plot_result_bar(time_dict, 'Model Execution Time Comparison', 'Time (seconds)', os.path.join(RESULTS_DIR, 'time_comparison.png'))
    # 混淆矩阵
    for name, probs in models_oof_probs.items():
        preds = (probs > 0.5).astype(int)
        plot_confusion_matrix(y_true, preds, name, os.path.join(RESULTS_DIR, f'cm_{name}.png'))
    # ROC曲线
    plot_roc_curves(y_true, models_oof_probs, os.path.join(RESULTS_DIR, 'roc_curve_comparison.png'))
    print("\nComparison complete. Results saved in '0625_2/results/'.")

if __name__ == '__main__':
    main() 