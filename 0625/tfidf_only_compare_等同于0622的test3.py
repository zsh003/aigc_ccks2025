import os
from cuml.linear_model import MBSGDClassifier
from cuml.ensemble import RandomForestClassifier
from cuml.naive_bayes import MultinomialNB
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_utils import check_gpu, load_data, save_model
from utils import extract_tfidf_features, train_and_eval_gpu_model, plot_result_bar

if __name__ == '__main__':
    check_gpu()
    # 数据路径
    df_all = load_data()
    # 特征工程
    tfidf_df, vectorizer = extract_tfidf_features(df_all, max_features=1500, ngram_range=(1,2), save_path='0625/models/tfidf_only_vectorizer.pkl')
    features_df = tfidf_df.copy()
    features_df['label'] = df_all['label']
    features_df['is_test'] = df_all['is_test']
    features_df = features_df.fillna(0)
    train_df = features_df[features_df['is_test']==0].reset_index(drop=True)
    test_df = features_df[features_df['is_test']==1].reset_index(drop=True)
    feats = [col for col in train_df.columns if col not in ['label','is_test']]
    # 模型对比
    models = {
        "MBSGDClassifier": MBSGDClassifier(loss="log", penalty="l2", alpha=1e-4, epochs=1000, tol=1e-4, n_iter_no_change=10),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42),
        "NaiveBayes": MultinomialNB()
    }
    results = {}
    for m_name, model in models.items():
        print(f"\n{'='*20} {m_name} {'='*20}")
        acc, f1, test_preds = train_and_eval_gpu_model(train_df, test_df, feats, 'label', model, m_name, results_dir='0625/results')
        results[m_name] = acc
        # 保存模型
        save_model(model, f'0625/models/{m_name}.pkl')
        # 保存预测结果
        import numpy as np
        np.save(f'0625/results/test_pred_{m_name}.npy', test_preds.get() if hasattr(test_preds, 'get') else test_preds)
    # 可视化
    plot_result_bar(results, '0625/results/model_compare.png') 