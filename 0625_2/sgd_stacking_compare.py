import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
import random
import itertools
import warnings
warnings.filterwarnings('ignore')


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
    import cuml
    from cuml.ensemble import RandomForestClassifier as CumlRF
    from cuml.linear_model import LogisticRegression as CumlLR
    has_cuml = True
except ImportError:
    has_cuml = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import check_gpu, load_data
from utils import plot_result_bar

OUTPUT_DIR = '0625_2'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
CV = KFold(n_splits=5, shuffle=True, random_state=2024)

# 构建模型池
candidate_models = {
    'SGDClassifier': SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=2024),
    'HistGBDT': HistGradientBoostingClassifier(max_iter=100, random_state=2024),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'MultinomialNB': MultinomialNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'BernoulliNB': BernoulliNB(),
    'ComplementNB': ComplementNB(),
    'NearestCentroid': NearestCentroid(),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=2024),
    'GaussianNB': GaussianNB(),
}
if has_lgb:
    candidate_models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=2024, n_jobs=-1)
if has_xgb:
    candidate_models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=2024, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if has_cat:
    candidate_models['CatBoost'] = CatBoostClassifier(iterations=100, random_state=2024, verbose=0)


def get_or_generate_oof_test_probs(model, model_name, X, y, X_test, model_dir=None, results_dir=None, use_cuml=False):
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, model_name)
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    oof_path = os.path.join(results_dir, f'oof_{model_name}.npy')
    test_path = os.path.join(results_dir, f'test_{model_name}.npy')
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof = np.load(oof_path)
        test_pred = np.load(test_path)
        print(f"{model_name} 概率已存在，直接加载。")
        return oof, test_pred
    oof = np.zeros(X.shape[0])
    test_pred = np.zeros(X_test.shape[0])
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        fold_model_path = os.path.join(model_dir, f'{model_name}_fold{fold}.pkl')
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        if os.path.exists(fold_model_path):
            model_ = joblib.load(fold_model_path)
        else:
            if use_cuml and has_cuml:
                # cuML模型
                if 'RF' in model_name:
                    model_ = CumlRF(n_estimators=100, random_state=2024)
                elif 'LR' in model_name:
                    model_ = CumlLR(max_iter=1000, random_state=2024)
                else:
                    raise ValueError('未知cuml模型类型')
            elif isinstance(model, StackingClassifier):
                model_ = StackingClassifier(
                    estimators=model.estimators,
                    final_estimator=model.final_estimator,
                    n_jobs=model.n_jobs,
                    passthrough=model.passthrough,
                    cv=model.cv,
                    stack_method=model.stack_method,
                    verbose=model.verbose
                )
            else:
                model_ = model.__class__(**model.get_params())
            model_.fit(X_train, y_train)
            joblib.dump(model_, fold_model_path)
        proba_val = model_.predict_proba(X_val)
        if hasattr(proba_val, 'get'):
            proba_val = proba_val.get()
        oof[val_idx] = proba_val[:, 1]

        proba_test = model_.predict_proba(X_test)
        if hasattr(proba_test, 'get'):
            proba_test = proba_test.get()
        test_pred += proba_test[:, 1] / CV.get_n_splits()
    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    return oof, test_pred

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
    y_true = train_df['label'].to_numpy()
    X = train_df.drop(columns=['label']).values
    X_test = test_df.values
    # 1. 单独SGDClassifier
    print(f"\n{'='*20} SGDClassifier 单独模型 {'='*20}")
    sgd = candidate_models['SGDClassifier']
    sgd_oof, sgd_test = get_or_generate_oof_test_probs(sgd, 'SGDClassifier', X, y_true, X_test)
    sgd_acc = accuracy_score(y_true, (sgd_oof > 0.5).astype(int))
    sgd_f1 = f1_score(y_true, (sgd_oof > 0.5).astype(int), average='weighted')
    try:
        sgd_auc = roc_auc_score(y_true, sgd_oof)
    except:
        sgd_auc = np.nan
    print(f"SGDClassifier: acc={sgd_acc:.4f}, f1={sgd_f1:.4f}, auc={sgd_auc:.4f}")
    with open(os.path.join(RESULTS_DIR, 'submission_SGDClassifier.txt'), 'w') as ftxt:
        for label in (sgd_test > 0.5).astype(int):
            ftxt.write(f"{label}\n")
    # 2. 遍历所有SGD+2模型组合做stacking（RF/LR双stacker+融合）
    all_results = []
    other_names = [k for k in candidate_models.keys() if k != 'SGDClassifier']
    stacking_sgdplus2_dir = os.path.join(MODELS_DIR, 'stacking_sgdplus2')
    stacking_sgdplus2_results = os.path.join(RESULTS_DIR, 'stacking_sgdplus2')
    os.makedirs(stacking_sgdplus2_dir, exist_ok=True)
    os.makedirs(stacking_sgdplus2_results, exist_ok=True)
    for m1, m2 in itertools.combinations(other_names, 2):
        combo_name = f'SGDStacking_{m1}_{m2}'
        estimators = [
            ('SGDClassifier', candidate_models['SGDClassifier']),
            (m1, candidate_models[m1]),
            (m2, candidate_models[m2])
        ]
        # RF stacker (cuML加速)
        if has_cuml:
            rf_name = f'{combo_name}_RF_cuml'
            print(f"\n{'='*20} {combo_name} RF_Stacker (cuML) {'='*20}")
            rf_oof, rf_test = get_or_generate_oof_test_probs(
                None, rf_name, X, y_true, X_test,
                model_dir=stacking_sgdplus2_dir,
                results_dir=stacking_sgdplus2_results,
                use_cuml=True
            )
        else:
            rf_stacker = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1), n_jobs=-1)
            rf_name = f'{combo_name}_RF'
            print(f"\n{'='*20} {combo_name} RF_Stacker {'='*20}")
            rf_oof, rf_test = get_or_generate_oof_test_probs(
                rf_stacker, rf_name, X, y_true, X_test,
                model_dir=stacking_sgdplus2_dir,
                results_dir=stacking_sgdplus2_results
            )
        rf_acc = accuracy_score(y_true, (rf_oof > 0.5).astype(int))
        rf_f1 = f1_score(y_true, (rf_oof > 0.5).astype(int), average='weighted')
        try:
            rf_auc = roc_auc_score(y_true, rf_oof)
        except:
            rf_auc = np.nan
        with open(os.path.join(stacking_sgdplus2_results, f'submission_{rf_name}.txt'), 'w') as ftxt:
            for label in (rf_test > 0.5).astype(int):
                ftxt.write(f"{label}\n")
        # LR stacker (cuML加速)
        if has_cuml:
            lr_name = f'{combo_name}_LR_cuml'
            print(f"\n{'='*20} {combo_name} LR_Stacker (cuML) {'='*20}")
            lr_oof, lr_test = get_or_generate_oof_test_probs(
                None, lr_name, X, y_true, X_test,
                model_dir=stacking_sgdplus2_dir,
                results_dir=stacking_sgdplus2_results,
                use_cuml=True
            )
        else:
            lr_stacker = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, solver='lbfgs'), n_jobs=-1)
            lr_name = f'{combo_name}_LR'
            print(f"\n{'='*20} {combo_name} LR_Stacker {'='*20}")
            lr_oof, lr_test = get_or_generate_oof_test_probs(
                lr_stacker, lr_name, X, y_true, X_test,
                model_dir=stacking_sgdplus2_dir,
                results_dir=stacking_sgdplus2_results
            )
        lr_acc = accuracy_score(y_true, (lr_oof > 0.5).astype(int))
        lr_f1 = f1_score(y_true, (lr_oof > 0.5).astype(int), average='weighted')
        try:
            lr_auc = roc_auc_score(y_true, lr_oof)
        except:
            lr_auc = np.nan
        with open(os.path.join(stacking_sgdplus2_results, f'submission_{lr_name}.txt'), 'w') as ftxt:
            for label in (lr_test > 0.5).astype(int):
                ftxt.write(f"{label}\n")
        # 融合
        w1, w2 = 0.5, 0.5
        fusion_oof = w1 * rf_oof + w2 * lr_oof
        fusion_test = w1 * rf_test + w2 * lr_test
        fusion_name = f'{combo_name}_Fusion'
        np.save(os.path.join(stacking_sgdplus2_results, f'oof_{fusion_name}.npy'), fusion_oof)
        np.save(os.path.join(stacking_sgdplus2_results, f'test_{fusion_name}.npy'), fusion_test)
        with open(os.path.join(stacking_sgdplus2_results, f'submission_{fusion_name}.txt'), 'w') as ftxt:
            for label in (fusion_test > 0.5).astype(int):
                ftxt.write(f"{label}\n")
        fusion_acc = accuracy_score(y_true, (fusion_oof > 0.5).astype(int))
        fusion_f1 = f1_score(y_true, (fusion_oof > 0.5).astype(int), average='weighted')
        try:
            fusion_auc = roc_auc_score(y_true, fusion_oof)
        except:
            fusion_auc = np.nan
        # 记录三种方案
        all_results.append({'Model': rf_name, 'Accuracy': rf_acc, 'F1-Score': rf_f1, 'AUC': rf_auc})
        all_results.append({'Model': lr_name, 'Accuracy': lr_acc, 'F1-Score': lr_f1, 'AUC': lr_auc})
        all_results.append({'Model': fusion_name, 'Accuracy': fusion_acc, 'F1-Score': fusion_f1, 'AUC': fusion_auc})
    # 保存所有对比结果
    results_df = pd.DataFrame([
        {'Model': 'SGDClassifier', 'Accuracy': sgd_acc, 'F1-Score': sgd_f1, 'AUC': sgd_auc}
    ] + all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'sgd_stacking_compare_all.csv'), index=False)
    print("\n全部SGD+2集成对比结果已保存。")

if __name__ == '__main__':
    main() 