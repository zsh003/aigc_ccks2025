import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import pickle
import scipy.special

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# 兼容性导入
try:
    import lightgbm as lgb
    has_lgb = True
except ImportError:
    has_lgb = False
try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
try:
    import catboost
    from catboost import CatBoostClassifier
    has_cat = True
except ImportError:
    has_cat = False
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CumlRF
    from cuml.linear_model import MBSGDClassifier as CumlSGD
    has_cuml = True
except ImportError:
    has_cuml = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import check_gpu, load_data
from utils import plot_confusion_matrix, plot_roc_curves, plot_result_bar

OUTPUT_DIR = '0625_2'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
CV = KFold(n_splits=5, shuffle=True, random_state=2024)

# 构建模型字典
models_to_run = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'SGDClassifier': SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=2024),
    'HistGBDT': HistGradientBoostingClassifier(max_iter=100, random_state=2024),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=2024),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    #'SVC': SVC(probability=True, random_state=2024),
    'MultinomialNB': MultinomialNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=2024),
    'QDA': QuadraticDiscriminantAnalysis(),
    'LDA': LinearDiscriminantAnalysis(),
    'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=2024),
    'Perceptron': Perceptron(max_iter=1000, random_state=2024),
    'RidgeClassifier': RidgeClassifier(),
    'BernoulliNB': BernoulliNB(),
    'ComplementNB': ComplementNB(),
    'CalibratedSGD': CalibratedClassifierCV(SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024)),
    'Dummy': DummyClassifier(strategy='most_frequent'),
    'GaussianNB': GaussianNB(),
    #'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=2024),
    'NearestCentroid': NearestCentroid(),
    'LabelPropagation': LabelPropagation(),
    'LabelSpreading': LabelSpreading(),
}
if has_lgb:
    models_to_run['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=2024, n_jobs=-1)
if has_xgb:
    models_to_run['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=2024, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if has_cat:
    models_to_run['CatBoost'] = CatBoostClassifier(iterations=100, random_state=2024, verbose=0)
if has_cuml:
    models_to_run['CumlRF'] = CumlRF(n_estimators=100, random_state=2024)
    models_to_run['CumlSGD'] = CumlSGD(loss="log", penalty="l2", alpha=1e-4, epochs=1000, tol=1e-4, n_iter_no_change=10)

# 集成模型单独处理
voting_estimators = [(k, v) for k, v in models_to_run.items() if k in ['LogisticRegression', 'RandomForest', 'SGDClassifier']]
stacking_estimators = [(k, v) for k, v in models_to_run.items() if k in ['LogisticRegression', 'RandomForest', 'SGDClassifier', 'MultinomialNB']]

def run_cv_experiment(model, model_name, train_df, test_df, is_gpu=False):
    """通用KFold交叉验证，支持CPU/GPU模型，保存每fold模型和预测txt"""
    submission_path = os.path.join(RESULTS_DIR, f'submission_{model_name}.txt')
    oof_probs = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    y = train_df['label']
    X = train_df.drop(columns=['label'])
    X_test = test_df
    pbar = tqdm(enumerate(CV.split(X, y)), total=CV.get_n_splits(), desc=f"{model_name} Folds")
    for fold, (train_idx, val_idx) in pbar:
        fold_model_path = os.path.join(MODELS_DIR, f'{model_name}_fold{fold}.pkl')
        if is_gpu:
            import cudf
            import cupy as cp
            # 转为 cudf
            X_train = cudf.from_pandas(X.iloc[list(train_idx)])
            y_train = cudf.from_pandas(y.iloc[list(train_idx)])
            X_val = cudf.from_pandas(X.iloc[list(val_idx)])
            y_val = cudf.from_pandas(y.iloc[list(val_idx)])
            X_test_gpu = cudf.from_pandas(X_test)
            if os.path.exists(fold_model_path):
                with open(fold_model_path, 'rb') as f:
                    fold_model = pickle.load(f)
            else:
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train, y_train)
                with open(fold_model_path, 'wb') as f:
                    pickle.dump(fold_model, f)
            # 预测
            if hasattr(fold_model, 'predict_proba'):
                val_pred = fold_model.predict_proba(X_val)[1].to_numpy()
                test_pred = fold_model.predict_proba(X_test_gpu)[1].to_numpy()
            elif hasattr(fold_model, 'decision_function'):
                val_pred = cp.asnumpy(fold_model.decision_function(X_val))
                test_pred = cp.asnumpy(fold_model.decision_function(X_test_gpu))
                val_pred = 1 / (1 + np.exp(-val_pred))
                test_pred = 1 / (1 + np.exp(-test_pred))
            else:
                val_pred = fold_model.predict(X_val).to_numpy()
                test_pred = fold_model.predict(X_test_gpu).to_numpy()
            oof_probs[list(val_idx)] = val_pred
            test_preds += test_pred / CV.get_n_splits()
            # 评估
            val_f1 = f1_score(y_val.to_numpy(), (val_pred > 0.5).astype(int), average='weighted')
            pbar.set_postfix({'F1': f'{val_f1:.5f}'})
        else:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            if os.path.exists(fold_model_path):
                fold_model = joblib.load(fold_model_path)
            else:
                # 集成模型直接用原对象，不用get_params()
                if isinstance(model, VotingClassifier) or isinstance(model, StackingClassifier):
                    fold_model = model
                else:
                    fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train, y_train)
                joblib.dump(fold_model, fold_model_path)
            fold_oof_probs = fold_model.predict_proba(X_val)[:, 1]
            oof_probs[np.array(val_idx)] = fold_oof_probs
            test_preds += fold_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
            fold_f1 = f1_score(y_val, (fold_oof_probs > 0.5).astype(int), average='weighted')
            pbar.set_postfix({'F1': f'{fold_f1:.5f}'})
    submission_labels = (test_preds > 0.5).astype(int)
    with open(submission_path, "w") as f:
        for label in submission_labels:
            f.write(f"{label}\n")
    return oof_probs, test_preds

def main():
    check_gpu()
    print("\nLoading data...")
    df_all_gpu = load_data()
    df_all = df_all_gpu.to_pandas() if hasattr(df_all_gpu, 'to_pandas') else df_all_gpu
    print("TF-IDF特征提取...")
    vectorizer = TfidfVectorizer(max_features=1500)
    tfidf_feat = vectorizer.fit_transform(df_all['text'])
    features_df = pd.DataFrame(tfidf_feat.toarray(), columns=vectorizer.get_feature_names_out())
    train_df = features_df[df_all['is_test']==0].copy()
    train_df['label'] = df_all[df_all['is_test']==0]['label'].to_numpy()
    test_df = features_df[df_all['is_test']==1].copy()
    y_true = train_df['label']
    results_summary = {}
    models_oof_probs = {}
    model_times = {}
    # 主模型循环
    for name, model in models_to_run.items():
        print(f"\n{'='*20} {name} {'='*20}")
        start_time = time.time()
        is_gpu = name.startswith('Cuml')
        try:
            oof_probs, _ = run_cv_experiment(model, name, train_df, test_df, is_gpu=is_gpu)
            exec_time = time.time() - start_time
            acc = accuracy_score(y_true, (oof_probs > 0.5).astype(int))
            f1 = f1_score(y_true, (oof_probs > 0.5).astype(int), average='weighted')
            try:
                auc = roc_auc_score(y_true, oof_probs)
            except:
                auc = np.nan
            results_summary[name] = {'Accuracy': acc, 'F1-Score': f1, 'AUC': auc, 'Time': exec_time}
            models_oof_probs[name] = oof_probs
            print(f"{name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, time={exec_time:.2f}s")
        except Exception as e:
            print(f"模型 {name} 运行失败: {e}")
    # 集成模型单独循环
    if len(voting_estimators) >= 2:
        print(f"\n{'='*20} Voting {'='*20}")
        try:
            voting_model = VotingClassifier(estimators=[(k, v) for k, v in voting_estimators], voting='soft', n_jobs=-1)
            oof_probs, _ = run_cv_experiment(voting_model, 'Voting', train_df, test_df, is_gpu=False)
            acc = accuracy_score(y_true, (oof_probs > 0.5).astype(int))
            f1 = f1_score(y_true, (oof_probs > 0.5).astype(int), average='weighted')
            try:
                auc = roc_auc_score(y_true, oof_probs)
            except:
                auc = np.nan
            results_summary['Voting'] = {'Accuracy': acc, 'F1-Score': f1, 'AUC': auc, 'Time': None}
            models_oof_probs['Voting'] = oof_probs
            print(f"Voting: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
        except Exception as e:
            print(f"模型 Voting 运行失败: {e}")
    if len(stacking_estimators) >= 2:
        print(f"\n{'='*20} Stacking {'='*20}")
        try:
            stacking_model = StackingClassifier(estimators=[(k, v) for k, v in stacking_estimators], final_estimator=LogisticRegression(), n_jobs=-1)
            oof_probs, _ = run_cv_experiment(stacking_model, 'Stacking', train_df, test_df, is_gpu=False)
            acc = accuracy_score(y_true, (oof_probs > 0.5).astype(int))
            f1 = f1_score(y_true, (oof_probs > 0.5).astype(int), average='weighted')
            try:
                auc = roc_auc_score(y_true, oof_probs)
            except:
                auc = np.nan
            results_summary['Stacking'] = {'Accuracy': acc, 'F1-Score': f1, 'AUC': auc, 'Time': None}
            models_oof_probs['Stacking'] = oof_probs
            print(f"Stacking: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
        except Exception as e:
            print(f"模型 Stacking 运行失败: {e}")
    # 保存结果表格
    results_df = pd.DataFrame(results_summary).T
    results_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_many_models.csv'))
    print("\n模型对比结果已保存: comparison_many_models.csv")
    # 可视化
    acc_dict = {name: data['Accuracy'] for name, data in results_summary.items()}
    plot_result_bar(acc_dict, 'Models Accurency Comparision', 'Accuracy', os.path.join(RESULTS_DIR, 'accuracy_many_models.png'))
    f1_dict = {name: data['F1-Score'] for name, data in results_summary.items()}
    plot_result_bar(f1_dict, 'Models F1-Score Comparision', 'F1-Score', os.path.join(RESULTS_DIR, 'f1_many_models.png'))
    # 混淆矩阵
    for name, probs in models_oof_probs.items():
        preds = (probs > 0.5).astype(int)
        plot_confusion_matrix(y_true, preds, name, os.path.join(RESULTS_DIR, f'cm_{name}.png'))
    # ROC曲线
    plot_roc_curves(y_true, models_oof_probs, os.path.join(RESULTS_DIR, 'roc_curve_many_models.png'))
    print("\n全部模型对比完成，结果已保存。")

if __name__ == '__main__':
    main() 