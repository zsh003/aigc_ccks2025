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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import check_gpu, load_data
from utils import plot_result_bar

OUTPUT_DIR = '0625_2'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PROBS_DIR = os.path.join(RESULTS_DIR, 'probs')
STACKING_MODEL_DIR = os.path.join(MODELS_DIR, 'stacking')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROBS_DIR, exist_ok=True)
os.makedirs(STACKING_MODEL_DIR, exist_ok=True)
CV = KFold(n_splits=5, shuffle=True, random_state=2024)

# 构建基学习器池
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

# 1. 生成或加载一级模型oof/test概率
def get_or_generate_oof_probs(model, model_name, train_df, test_df, y_true):
    oof_path = os.path.join(PROBS_DIR, f'oof_{model_name}.npy')
    test_path = os.path.join(PROBS_DIR, f'test_{model_name}.npy')
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof_probs = np.load(oof_path)
        test_probs = np.load(test_path)
        print(f"{model_name} 概率已存在，直接加载。")
        return oof_probs, test_probs
    print(f"生成 {model_name} 的oof和test概率...")
    oof_probs = np.zeros(train_df.shape[0])
    test_probs = np.zeros(test_df.shape[0])
    X = train_df.drop(columns=['label'])
    y = train_df['label']
    X_test = test_df
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        fold_model_path = os.path.join(STACKING_MODEL_DIR, f'{model_name}_fold{fold}.pkl')
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if os.path.exists(fold_model_path):
            fold_model = joblib.load(fold_model_path)
        else:
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_train, y_train)
            joblib.dump(fold_model, fold_model_path)
        # 预测
        if hasattr(fold_model, 'predict_proba'):
            fold_oof_probs = fold_model.predict_proba(X_val)[:, 1]
            test_probs += fold_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        elif hasattr(fold_model, 'decision_function'):
            fold_oof_probs = fold_model.decision_function(X_val)
            fold_oof_probs = 1 / (1 + np.exp(-fold_oof_probs))
            test_probs += 1 / (1 + np.exp(-fold_model.decision_function(X_test))) / CV.get_n_splits()
        else:
            fold_oof_probs = fold_model.predict(X_val)
            test_probs += fold_model.predict(X_test) / CV.get_n_splits()
        oof_probs[np.array(val_idx)] = fold_oof_probs
    np.save(oof_path, oof_probs)
    np.save(test_path, test_probs)
    return oof_probs, test_probs

# 2. 二级stacking训练与评估
def stacking_train_and_eval(X_stack, y_true, X_stack_test, stacker, stacker_name):
    oof = np.zeros(X_stack.shape[0])
    test_pred = np.zeros(X_stack_test.shape[0])
    for fold, (train_idx, val_idx) in enumerate(CV.split(X_stack, y_true)):
        X_tr, y_tr = X_stack[train_idx], y_true.iloc[train_idx]
        X_val, y_val = X_stack[val_idx], y_true.iloc[val_idx]
        stacker.fit(X_tr, y_tr)
        oof[val_idx] = stacker.predict_proba(X_val)[:, 1]
        test_pred += stacker.predict_proba(X_stack_test)[:, 1] / CV.get_n_splits()
    acc = accuracy_score(y_true, (oof > 0.5).astype(int))
    f1 = f1_score(y_true, (oof > 0.5).astype(int), average='weighted')
    try:
        auc = roc_auc_score(y_true, oof)
    except:
        auc = np.nan
    return oof, test_pred, acc, f1, auc


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
    # 1. 一级模型概率特征
    oof_dict = {}
    test_dict = {}
    for name, model in candidate_models.items():
        oof_probs, test_probs = get_or_generate_oof_probs(model, name, train_df, test_df, y_true)
        oof_dict[name] = oof_probs
        test_dict[name] = test_probs
    # 2. stacking集成（全模型）
    X_stack = np.column_stack([oof_dict[name] for name in candidate_models.keys()])
    X_stack_test = np.column_stack([test_dict[name] for name in candidate_models.keys()])
    results = []
    for stacker, stacker_name in [(RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1), 'RF_Stacker'), (LogisticRegression(max_iter=1000, solver='lbfgs'), 'LR_Stacker')]:
        print(f"\n{'='*20} Stacking with {stacker_name} (All models) {'='*20}")
        oof, test_pred, acc, f1, auc = stacking_train_and_eval(X_stack, y_true, X_stack_test, stacker, stacker_name)
        np.save(os.path.join(PROBS_DIR, f'oof_{stacker_name}.npy'), oof)
        np.save(os.path.join(PROBS_DIR, f'test_{stacker_name}.npy'), test_pred)
        with open(os.path.join(RESULTS_DIR, f'submission_{stacker_name}.txt'), 'w') as ftxt:
            for label in (test_pred > 0.5).astype(int):
                ftxt.write(f"{label}\n")
        joblib.dump(stacker, os.path.join(STACKING_MODEL_DIR, f'{stacker_name}.pkl'))
        results.append({'Ablation': 'None', 'Stacker': stacker_name, 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})
    # 3. 消融实验
    for ablate_name in candidate_models.keys():
        ablate_names = [n for n in candidate_models.keys() if n != ablate_name]
        X_stack_ablate = np.column_stack([oof_dict[n] for n in ablate_names])
        X_stack_test_ablate = np.column_stack([test_dict[n] for n in ablate_names])
        for stacker, stacker_name in [(RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1), 'RF_Stacker'), (LogisticRegression(max_iter=1000, solver='lbfgs'), 'LR_Stacker')]:
            print(f"\n{'='*20} Stacking with {stacker_name} (Ablate {ablate_name}) {'='*20}")
            oof, test_pred, acc, f1, auc = stacking_train_and_eval(X_stack_ablate, y_true, X_stack_test_ablate, stacker, stacker_name)
            np.save(os.path.join(PROBS_DIR, f'oof_{stacker_name}_ablate_{ablate_name}.npy'), oof)
            np.save(os.path.join(PROBS_DIR, f'test_{stacker_name}_ablate_{ablate_name}.npy'), test_pred)
            with open(os.path.join(RESULTS_DIR, f'submission_{stacker_name}_ablate_{ablate_name}.txt'), 'w') as ftxt:
                for label in (test_pred > 0.5).astype(int):
                    ftxt.write(f"{label}\n")
            joblib.dump(stacker, os.path.join(STACKING_MODEL_DIR, f'{stacker_name}_ablate_{ablate_name}.pkl'))
            results.append({'Ablation': ablate_name, 'Stacker': stacker_name, 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'stacking_stacker_ablation_results.csv'), index=False)
    print("\n消融实验结果已保存: stacking_stacker_ablation_results.csv")
    # 可视化F1变化
    for stacker_name in ['RF_Stacker', 'LR_Stacker']:
        f1_dict = {row['Ablation']: row['F1-Score'] for row in results if row['Stacker']==stacker_name}
        plot_result_bar(f1_dict, f'Stacking Ablation F1-Score ({stacker_name})', 'F1-Score', os.path.join(RESULTS_DIR, f'stacking_stacker_ablation_f1_{stacker_name}.png'))
    print("\n全部消融实验完成，结果已保存。")
    # 融合RF和LR集成结果
    rf_pred = np.load(os.path.join(PROBS_DIR, 'test_RF_Stacker.npy'))
    lr_pred = np.load(os.path.join(PROBS_DIR, 'test_LR_Stacker.npy'))
    w1, w2 = 0.5, 0.5  # 可自定义权重
    weighted_pred = w1 * rf_pred + w2 * lr_pred
    np.save(os.path.join(PROBS_DIR, 'test_weighted.npy'), weighted_pred)
    with open(os.path.join(RESULTS_DIR, 'submission_weighted.txt'), 'w') as ftxt:
        for label in (weighted_pred > 0.5).astype(int):
            ftxt.write(f"{label}\n")
    print('\n融合预测结果已保存：submission_weighted.txt, test_weighted.npy')

    # 评估RF/LR/融合的oof效果
    rf_oof = np.load(os.path.join(PROBS_DIR, 'oof_RF_Stacker.npy'))
    lr_oof = np.load(os.path.join(PROBS_DIR, 'oof_LR_Stacker.npy'))
    weighted_oof = w1 * rf_oof + w2 * lr_oof
    def eval_oof(y_true, oof_pred):
        acc = accuracy_score(y_true, (oof_pred > 0.5).astype(int))
        f1 = f1_score(y_true, (oof_pred > 0.5).astype(int), average='weighted')
        try:
            auc = roc_auc_score(y_true, oof_pred)
        except:
            auc = np.nan
        return acc, f1, auc
    rf_acc, rf_f1, rf_auc = eval_oof(y_true, rf_oof)
    lr_acc, lr_f1, lr_auc = eval_oof(y_true, lr_oof)
    w_acc, w_f1, w_auc = eval_oof(y_true, weighted_oof)
    print(f"\nRF_Stacker OOF: acc={rf_acc:.4f}, f1={rf_f1:.4f}, auc={rf_auc:.4f}")
    print(f"LR_Stacker OOF: acc={lr_acc:.4f}, f1={lr_f1:.4f}, auc={lr_auc:.4f}")
    print(f"Weighted OOF:  acc={w_acc:.4f}, f1={w_f1:.4f}, auc={w_auc:.4f}")
    eval_df = pd.DataFrame([
        {'Model': 'RF_Stacker', 'Accuracy': rf_acc, 'F1-Score': rf_f1, 'AUC': rf_auc},
        {'Model': 'LR_Stacker', 'Accuracy': lr_acc, 'F1-Score': lr_f1, 'AUC': lr_auc},
        {'Model': 'Weighted',   'Accuracy': w_acc,  'F1-Score': w_f1,  'AUC': w_auc},
    ])
    eval_df.to_csv(os.path.join(RESULTS_DIR, 'stacking_stacker_ablation_eval.csv'), index=False)
    print(f"\n评估结果已保存: stacking_stacker_ablation_eval.csv")

if __name__ == '__main__':
    main() 