import os
import sys
import numpy as np
from cuml.linear_model import MBSGDClassifier

# 将项目根目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_utils import check_gpu, load_data, save_model
from utils import extract_tfidf_features, train_and_eval_baseline_gpu

def main():
    """主执行函数"""
    # 检查GPU
    check_gpu()
    
    # 定义输出目录
    output_dir = '0625_2'
    model_dir = os.path.join(output_dir, 'models')
    
    # 加载数据
    print("Loading data...")
    df_all = load_data()
    
    # 特征工程: TF-IDF
    print("\nExtracting TF-IDF features...")
    features_df, vectorizer = extract_tfidf_features(df_all, max_features=1500)
    
    # 准备训练和测试数据
    features_df['label'] = df_all['label']
    features_df['is_test'] = df_all['is_test']
    features_df = features_df.fillna(0)
    
    train_df = features_df[features_df['is_test']==0].reset_index(drop=True)
    test_df = features_df[features_df['is_test']==1].reset_index(drop=True)
    
    feats = [col for col in train_df.columns if col not in ['label', 'is_test']]
    
    # 定义模型: MBSGDClassifier (cuML's SGD)
    # 参数根据 0619/baseline_0.71.py 进行映射
    print("\nInitializing MBSGDClassifier model...")
    model = MBSGDClassifier(
        loss="log",
        penalty="l2",
        epochs=1500,
        tol=1e-5,
        n_iter_no_change=10
    )
    
    # 训练和评估
    print("\nStarting training and cross-validation...")
    _, test_preds, _ = train_and_eval_baseline_gpu(
        train_df=train_df,
        test_df=test_df,
        feats=feats,
        y_col='label',
        model=model
    )
    
    # 保存模型
    print("\nSaving trained model...")
    model_path = os.path.join(model_dir, 'mbsgd_baseline_model.pkl')
    save_model(model, model_path)
    
    # 生成提交文件
    print("Generating submission file...")
    labels = (test_preds.get() > 0.5).astype(int)
    submission_path = os.path.join(output_dir, "submit_sgd_gpu.txt")
    with open(submission_path, "w") as file:
        for label in labels:
            file.write(str(label) + "\n")
    print(f"Submission file saved to: {submission_path}")

if __name__ == '__main__':
    main() 