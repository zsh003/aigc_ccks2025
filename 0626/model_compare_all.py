import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

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
from utils import extract_aigc_features, plot_confusion_matrix, plot_roc_curves, plot_result_bar

OUTPUT_DIR = '0626'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PROBS_DIR = os.path.join(RESULTS_DIR, 'probs')
STACKING_MODEL_DIR = os.path.join(MODELS_DIR, 'stacking')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROBS_DIR, exist_ok=True)
os.makedirs(STACKING_MODEL_DIR, exist_ok=True)
CV = KFold(n_splits=5, shuffle=True, random_state=2024)

# 单模型池（与0625_2/compare_many_models.py一致）
single_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs'),
    #'SGDClassifier': SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=2024),
    'HistGBDT': HistGradientBoostingClassifier(max_iter=100, random_state=2024),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=2024),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=2024, n_jobs=-1),
    'MultinomialNB': MultinomialNB(),
    'DecisionTree': None, # 跳过
    'QDA': None,
    'LDA': LinearDiscriminantAnalysis(),
    'PassiveAggressive': None,
    'Perceptron': None,
    'RidgeClassifier': None,
    'BernoulliNB': BernoulliNB(),
    'ComplementNB': ComplementNB(),
    'CalibratedSGD': None,
    'GaussianNB': GaussianNB(),
    'NearestCentroid': NearestCentroid(),
}
if has_lgb:
    single_models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=2024, n_jobs=-1)
if has_xgb:
    single_models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=2024, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if has_cat:
    single_models['CatBoost'] = CatBoostClassifier(iterations=100, random_state=2024, verbose=0)

# 集成池（与0625_2/sgd_stacking_compare.py一致）
ensemble_models = {
    #'SGDClassifier': SGDClassifier(loss="modified_huber", max_iter=1500, tol=1e-5, n_iter_no_change=10, random_state=2024),
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
    ensemble_models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=2024, n_jobs=-1)
if has_xgb:
    ensemble_models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=2024, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if has_cat:
    ensemble_models['CatBoost'] = CatBoostClassifier(iterations=100, random_state=2024, verbose=0)

# 单模型CV

def run_cv_experiment(model, model_name, train_df, test_df):
    submission_path = os.path.join(RESULTS_DIR, f'submission_{model_name}.txt')
    oof_probs = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    y = train_df['label']
    X = train_df.drop(columns=['label'])
    X_test = test_df.drop(columns=['label']) if 'label' in test_df.columns else test_df
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
        oof_probs[np.array(val_idx)] = fold_oof_probs
        test_preds += fold_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        fold_f1 = f1_score(y_val, (fold_oof_probs > 0.5).astype(int), average='weighted')
        pbar.set_postfix({'F1': f'{fold_f1:.5f}'})
    submission_labels = (test_preds > 0.5).astype(int)
    with open(submission_path, "w") as f:
        for label in submission_labels:
            f.write(f"{label}\n")
    return oof_probs, test_preds

# stacking集成

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
    print("特征工程...")
    import pickle
    feat_path = os.path.join(RESULTS_DIR, 'features_all_feat.pkl')
    feat_names_path = os.path.join(RESULTS_DIR, 'features_feat_names.pkl')
    if os.path.exists(feat_path) and os.path.exists(feat_names_path):
        print("检测到已保存特征，直接加载...")
        with open(feat_path, 'rb') as f:
            all_feat = pickle.load(f)
        with open(feat_names_path, 'rb') as f:
            feat_names = pickle.load(f)
    else:
        t0 = time.time()
        all_feat, feat_names = extract_aigc_features(df_all)
        with open(feat_path, 'wb') as f:
            pickle.dump(all_feat, f)
        with open(feat_names_path, 'wb') as f:
            pickle.dump(feat_names, f)
        print(f"特征提取完成，用时 {time.time()-t0:.2f} 秒")
    all_feat['is_test'] = df_all['is_test'].values
    if 'label' in df_all:
        all_feat['label'] = df_all['label'].values
    train_df = all_feat[all_feat['is_test']==0].copy()
    test_df = all_feat[all_feat['is_test']==1].copy()
    y_true = train_df['label']
    # 1. 单模型对比（新特征）
    results_summary = {}
    models_oof_probs = {}
    print("\n单模型训练与评估...")
    feat_rank_dir = os.path.join(RESULTS_DIR, 'feats_rank')
    os.makedirs(feat_rank_dir, exist_ok=True)
    for name, model in tqdm(single_models.items(), desc='Single Models'):
        if model is None:
            continue
        print(f"\n{'='*20} {name} {'='*20}")
        start_time = time.time()
        try:
            oof_probs, _ = run_cv_experiment(model, name, train_df, test_df)
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
            # 特征重要性分析
            try:
                # 取第0折模型
                fold_model_path = os.path.join(MODELS_DIR, f'{name}_fold0.pkl')
                if os.path.exists(fold_model_path):
                    fold_model = joblib.load(fold_model_path)
                    if hasattr(fold_model, 'feature_importances_'):
                        importances = fold_model.feature_importances_
                    elif hasattr(fold_model, 'coef_'):
                        importances = np.abs(fold_model.coef_).ravel()
                    else:
                        importances = None
                    if importances is not None:
                        # 归一化（L2）
                        norm = np.linalg.norm(importances)
                        if norm > 0:
                            importances = importances / norm
                        # 排序并打印Top20
                        feat_imp = sorted(zip(feat_names, importances), key=lambda x: -abs(x[1]))
                        print(f"{name} 特征重要性Top20 (L2归一化):")
                        for i, (fname, imp) in enumerate(feat_imp[:20]):
                            print(f"{i+1:2d}. {fname:30s} {imp:.5f}")
                        # 可视化
                        import matplotlib.pyplot as plt
                        topn = 20
                        plt.figure(figsize=(8, max(6, topn*0.4)))
                        names = [x[0] for x in feat_imp[:topn]]
                        vals = [x[1] for x in feat_imp[:topn]]
                        plt.barh(range(topn), vals[::-1], color='skyblue')
                        plt.yticks(range(topn), names[::-1])
                        plt.xlabel('Importance (L2 normalized)')
                        plt.title(f'{name} Feature Importance (Top {topn})')
                        plt.tight_layout()
                        plt.savefig(os.path.join(feat_rank_dir, f'{name}_feat_importance.png'))
                        plt.close()
            except Exception as e:
                print(f"特征重要性分析失败: {e}")
        except Exception as e:
            print(f"模型 {name} 运行失败: {e}")
    # 综合全局特征重要性
    print("\n综合全模型特征重要性...")
    feat_importance_dict = {}
    for name, model in single_models.items():
        if model is None:
            continue
        try:
            fold_model_path = os.path.join(MODELS_DIR, f'{name}_fold0.pkl')
            if os.path.exists(fold_model_path):
                fold_model = joblib.load(fold_model_path)
                if hasattr(fold_model, 'feature_importances_'):
                    importances = fold_model.feature_importances_
                elif hasattr(fold_model, 'coef_'):
                    importances = np.abs(fold_model.coef_).ravel()
                else:
                    importances = None
                if importances is not None:
                    # 归一化（L2）
                    norm = np.linalg.norm(importances)
                    if norm > 0:
                        importances = importances / norm
                    for fname, imp in zip(feat_names, importances):
                        if fname not in feat_importance_dict:
                            feat_importance_dict[fname] = []
                        feat_importance_dict[fname].append(imp)
        except Exception as e:
            pass
    # 计算全局平均重要性
    global_feat_imp = {k: np.mean(v) for k, v in feat_importance_dict.items() if len(v)>0}
    global_feat_imp_sorted = sorted(global_feat_imp.items(), key=lambda x: -abs(x[1]))
    # 保存csv
    global_df = pd.DataFrame(global_feat_imp_sorted, columns=['Feature', 'Global_Importance'])
    global_df.to_csv(os.path.join(feat_rank_dir, 'global_feat_importance.csv'), index=False)
    print(f"全局特征重要性Top30:")
    for i, (fname, imp) in enumerate(global_feat_imp_sorted[:30]):
        print(f"{i+1:2d}. {fname:30s} {imp:.5f}")
    # 可视化
    import matplotlib.pyplot as plt
    topn = 30
    plt.figure(figsize=(9, max(7, topn*0.4)))
    names = [x[0] for x in global_feat_imp_sorted[:topn]]
    vals = [x[1] for x in global_feat_imp_sorted[:topn]]
    plt.barh(range(topn), vals[::-1], color='orange')
    plt.yticks(range(topn), names[::-1])
    plt.xlabel('Global Importance')
    plt.title(f'Global Feature Importance (Top {topn})')
    plt.tight_layout()
    plt.savefig(os.path.join(feat_rank_dir, 'global_feat_importance.png'))
    plt.close()
    # 2. stacking全模型集成（RF/LR/融合）（新特征）
    print(f"\n{'='*20} Stacking All Models {'='*20}")
    oof_dict = {}
    test_dict = {}
    for name, model in tqdm(ensemble_models.items(), desc='Ensemble Base Models'):
        t0 = time.time()
        oof_probs, test_probs = run_cv_experiment(model, name, train_df, test_df)
        print(f"{name} 集成基模型用时: {time.time()-t0:.2f} 秒")
        oof_dict[name] = oof_probs
        test_dict[name] = test_probs
    X_stack = np.column_stack([oof_dict[name] for name in ensemble_models.keys()])
    X_stack_test = np.column_stack([test_dict[name] for name in ensemble_models.keys()])
    stack_results = {}
    for stacker, stacker_name in [(RandomForestClassifier(n_estimators=100, random_state=2024, n_jobs=-1), 'RF_Stacker'), (LogisticRegression(max_iter=1000, solver='lbfgs'), 'LR_Stacker')]:
        print(f"\n{'='*20} Stacking with {stacker_name} (All models) {'='*20}")
        t0 = time.time()
        oof, test_pred, acc, f1, auc = stacking_train_and_eval(X_stack, y_true, X_stack_test, stacker, stacker_name)
        print(f"{stacker_name} 集成用时: {time.time()-t0:.2f} 秒")
        np.save(os.path.join(PROBS_DIR, f'oof_{stacker_name}.npy'), oof)
        np.save(os.path.join(PROBS_DIR, f'test_{stacker_name}.npy'), test_pred)
        with open(os.path.join(RESULTS_DIR, f'submission_{stacker_name}.txt'), 'w') as ftxt:
            for label in (test_pred > 0.5).astype(int):
                ftxt.write(f"{label}\n")
        joblib.dump(stacker, os.path.join(STACKING_MODEL_DIR, f'{stacker_name}.pkl'))
        stack_results[stacker_name] = {'Accuracy': acc, 'F1-Score': f1, 'AUC': auc}
    # 融合
    rf_pred = np.load(os.path.join(PROBS_DIR, 'test_RF_Stacker.npy'))
    lr_pred = np.load(os.path.join(PROBS_DIR, 'test_LR_Stacker.npy'))
    w1, w2 = 0.5, 0.5
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
    stack_results['Weighted'] = {'Accuracy': w_acc, 'F1-Score': w_f1, 'AUC': w_auc}
    # 汇总所有模型（新特征）
    all_results = results_summary.copy()
    all_results.update({'RF_Stacker': stack_results['RF_Stacker'], 'LR_Stacker': stack_results['LR_Stacker'], 'Weighted': stack_results['Weighted']})
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_compare_all.csv'))
    print("\n全部模型对比结果已保存: model_compare_all.csv")
    # 读取0625_2下的老特征结果
    old_single_path = '0625_2/results/comparison_many_models.csv'
    old_ensemble_path = '0625_2/results/stacking_stacker_ablation_eval.csv'
    if os.path.exists(old_single_path):
        old_single = pd.read_csv(old_single_path, index_col=0)
    else:
        old_single = None
    if os.path.exists(old_ensemble_path):
        old_ensemble = pd.read_csv(old_ensemble_path)
    else:
        old_ensemble = None
    # 合并新老特征对比表
    compare_rows = []
    for name in all_results:
        new_acc = all_results[name]['Accuracy'] if name in all_results else np.nan
        new_f1 = all_results[name]['F1-Score'] if name in all_results else np.nan
        new_auc = all_results[name]['AUC'] if name in all_results else np.nan
        old_acc = old_f1 = old_auc = np.nan
        if old_single is not None and name in old_single.index:
            old_acc = old_single.loc[name, 'Accuracy']
            old_f1 = old_single.loc[name, 'F1-Score']
            old_auc = old_single.loc[name, 'AUC']
        if old_ensemble is not None and 'Model' in old_ensemble.columns and name in old_ensemble['Model'].values:
            row = old_ensemble[old_ensemble['Model']==name].iloc[0]
            old_acc = row['Accuracy']
            old_f1 = row['F1-Score']
            old_auc = row['AUC']
        compare_rows.append({'Model': name, 'New_Accuracy': new_acc, 'Old_Accuracy': old_acc, 'New_F1': new_f1, 'Old_F1': old_f1, 'New_AUC': new_auc, 'Old_AUC': old_auc})
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(os.path.join(RESULTS_DIR, 'new_vs_old_feature_compare.csv'), index=False)
    print('\n新老特征对比表已保存: new_vs_old_feature_compare.csv')
    # 可视化新老特征对比
    for metric in ['Accuracy', 'F1', 'AUC']:
        new_dict = {row['Model']: row[f'New_{metric}'] for _, row in compare_df.iterrows()}
        old_dict = {row['Model']: row[f'Old_{metric}'] for _, row in compare_df.iterrows()}
        plot_result_bar(new_dict, f'New Feature {metric}', metric, os.path.join(RESULTS_DIR, f'new_feature_{metric.lower()}.png'))
        plot_result_bar(old_dict, f'Old Feature {metric}', metric, os.path.join(RESULTS_DIR, f'old_feature_{metric.lower()}.png'))
        # 新老对比条形图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(10, len(new_dict)*0.7), 6))
        x = np.arange(len(new_dict))
        width = 0.35
        plt.bar(x-width/2, list(new_dict.values()), width, label='New')
        plt.bar(x+width/2, list(old_dict.values()), width, label='Old')
        plt.xticks(x, list(new_dict.keys()), rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'New vs Old Feature {metric} Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'new_vs_old_{metric.lower()}_compare.png'))
        plt.close()
    print('\n全部新老特征对比可视化已保存。')
    # 其他可视化
    acc_dict = {name: data['Accuracy'] for name, data in all_results.items()}
    plot_result_bar(acc_dict, 'Models Accuracy Comparison', 'Accuracy', os.path.join(RESULTS_DIR, 'accuracy_all.png'))
    f1_dict = {name: data['F1-Score'] for name, data in all_results.items()}
    plot_result_bar(f1_dict, 'Models F1-Score Comparison', 'F1-Score', os.path.join(RESULTS_DIR, 'f1_all.png'))
    # 混淆矩阵
    for name, probs in models_oof_probs.items():
        preds = (probs > 0.5).astype(int)
        plot_confusion_matrix(y_true, preds, name, os.path.join(RESULTS_DIR, f'cm_{name}.png'))
    for name in ['RF_Stacker', 'LR_Stacker', 'Weighted']:
        if name == 'Weighted':
            preds = (weighted_oof > 0.5).astype(int)
            plot_confusion_matrix(y_true, preds, name, os.path.join(RESULTS_DIR, f'cm_{name}.png'))
        else:
            oof = np.load(os.path.join(PROBS_DIR, f'oof_{name}.npy'))
            preds = (oof > 0.5).astype(int)
            plot_confusion_matrix(y_true, preds, name, os.path.join(RESULTS_DIR, f'cm_{name}.png'))
    # ROC曲线
    all_oof_probs = models_oof_probs.copy()
    all_oof_probs['RF_Stacker'] = rf_oof
    all_oof_probs['LR_Stacker'] = lr_oof
    all_oof_probs['Weighted'] = weighted_oof
    plot_roc_curves(y_true, all_oof_probs, os.path.join(RESULTS_DIR, 'roc_curve_all.png'))
    print("\n全部模型对比完成，结果已保存。")

if __name__ == '__main__':
    main() 