import pandas as pd
from tqdm import tqdm
from utils import (
    split_train_test,
    normalize_column,
    inverse_normalize_column,
    evaluation,
    ensure_dir
)
from model import extract_features, train, predict
from visualization import plot_evaluation_metrics, plot_prediction_vs_actual, plot_feature_importance
import os


def process_station(station_id, capacity, problem, top_n_features=10):
    """
    处理单个电站的数据

    参数:
        station_id (str): 电站ID
        capacity (float): 电站容量
        problem (str): 问题名称，用于保存结果
        top_n_features (int): 要选择的最重要特征数量

    返回:
        tuple: (评估结果列表, 特征重要性DataFrame, 选定的特征列表)
    """
    # 读取数据
    station_data = pd.read_csv(f'../dataset/{station_id}_processed.csv')[[
        'date_time', 'power', 'nwp_globalirrad', 'nwp_directirrad'
    ]]

    # 归一化
    normalized_station_data, min_value, max_value = normalize_column(station_data, 'power')
    normalized_station_data, _, _ = normalize_column(normalized_station_data, 'nwp_globalirrad')
    normalized_station_data, _, _ = normalize_column(normalized_station_data, 'nwp_directirrad')

    # 数据集分割
    train_data, test_data_2, test_data_5, test_data_8, test_data_11, test_data_total = split_train_test(normalized_station_data)

    # 特征处理
    train_data = extract_features(train_data, is_training=True)
    test_data_2 = extract_features(test_data_2, is_training=False)
    test_data_5 = extract_features(test_data_5, is_training=False)
    test_data_8 = extract_features(test_data_8, is_training=False)
    test_data_11 = extract_features(test_data_11, is_training=False)
    test_data_total = extract_features(test_data_total, is_training=False)

    # 训练模型
    model_path = f'model/{problem}/{station_id}.pkl'
    ensure_dir(os.path.dirname(model_path))
    model, feature_importance, selected_features = train(train_data, model_path, top_n_features)

    # 预测
    result_path = f'results/{problem}'
    ensure_dir(result_path)
        
    test_data_2 = predict(model_path, test_data_2)
    test_data_2 = inverse_normalize_column(test_data_2, 'predicted_power', min_value, max_value)
    test_data_5 = predict(model_path, test_data_5)
    test_data_5 = inverse_normalize_column(test_data_5, 'predicted_power', min_value, max_value)
    test_data_8 = predict(model_path, test_data_8)
    test_data_8 = inverse_normalize_column(test_data_8, 'predicted_power', min_value, max_value)
    test_data_11 = predict(model_path, test_data_11)
    test_data_11 = inverse_normalize_column(test_data_11, 'predicted_power', min_value, max_value)
    test_data_total['predicted_power'] = pd.concat([
        test_data_2['predicted_power'],
        test_data_5['predicted_power'],
        test_data_8['predicted_power'],
        test_data_11['predicted_power']
    ])
    test_data_total[['date_time', 'predicted_power', 'power']].to_csv(f'{result_path}/result_{station_id}.csv', index=False)

    # 绘制预测结果对比图
    plot_prediction_vs_actual(
        test_data_total['predicted_power'],
        test_data_total['power'],
        station_id,
        f'{result_path}/prediction_vs_actual_{station_id}.png'
    )

    # 绘制特征重要性图
    plot_feature_importance(
        feature_importance,
        f'{result_path}/feature_importance_{station_id}.png'
    )

    # 评估
    evaluations = []
    for test_data, month in [
        (test_data_2, '2'),
        (test_data_5, '5'),
        (test_data_8, '8'),
        (test_data_11, '11'),
        (test_data_total, 'total')
    ]:
        if test_data is not None:
            eval_result = evaluation(
                predicted_list=test_data['predicted_power'].tolist(),
                actual_list=test_data['power'].tolist(),
                constant_capacity=capacity / 1e3,
                station_id=station_id,
                time_list=test_data['date_time'].tolist()
            )
            if eval_result:
                eval_result['month'] = month
                evaluations.append(eval_result)

    return evaluations, feature_importance, selected_features


def main():
    problem = 'problem_03_1_gpu'
    metadata = pd.read_csv('../dataset/metadata.csv')
    top_n_features = 10  # 选择最重要的10个特征

    all_evaluations = []
    all_feature_importances = []
    all_selected_features = []

    # 使用tqdm创建进度条
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="处理电站数据"):
        station_id = row['Station_ID']
        capacity = row['Capacity']
        
        # 处理单个电站
        evaluations, feature_importance, selected_features = process_station(
            station_id, capacity, problem, top_n_features
        )
        all_evaluations.extend(evaluations)
        all_feature_importances.append(feature_importance)
        all_selected_features.append({
            'station_id': station_id,
            'selected_features': selected_features
        })

        print(f"===={station_id} 已完成====")

    # 保存评估结果
    result_path = f'results/{problem}'
    ensure_dir(result_path)
    
    # 按月份分组保存评估结果
    evaluations_df = pd.DataFrame(all_evaluations)
    for month in ['2', '5', '8', '11', 'total']:
        month_evaluations = evaluations_df[evaluations_df['month'] == month]
        if not month_evaluations.empty:
            month_evaluations.to_csv(f'{result_path}/evaluations_{month}.csv', index=False)

    # 保存选定的特征
    selected_features_df = pd.DataFrame(all_selected_features)
    selected_features_df.to_csv(f'{result_path}/selected_features.csv', index=False)

    # 绘制总体评估指标图表
    if all_evaluations:
        plot_evaluation_metrics(
            evaluations_df[evaluations_df['month'] == 'total'],
            f'{result_path}/evaluation_metrics.png'
        )


if __name__ == "__main__":
    main()
