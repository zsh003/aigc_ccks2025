import joblib
import pandas as pd
import numpy as np
from cuml.ensemble import RandomForestRegressor as cuRFR
import cudf
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import os


class CuMLWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred_pd = self.model.predict(X)
        return y_pred_pd.to_numpy()

    def score(self, X, y):
        """
        计算模型在 (X, y) 上的 R² 分数
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def select_features(feature_importance_df, top_n=10):
    """
    根据特征重要性选择前N个最重要的特征

    参数:
        feature_importance_df (DataFrame): 包含特征重要性的数据框
        top_n (int): 要选择的特征数量

    返回:
        list: 选定的特征名称列表
    """
    # 按重要性排序并选择前N个特征
    selected_features = feature_importance_df.nlargest(top_n, 'importance')['feature'].tolist()
    return selected_features


def calculate_feature_importance(model, X_train_gpu, y_train_gpu, X_train):
    """
    计算并输出特征重要性

    参数:
        model: 训练好的模型
        X_train_gpu: GPU上的训练特征数据
        y_train_gpu: GPU上的训练目标数据
        X_train: CPU上的训练特征数据（用于获取特征名称）

    返回:
        DataFrame: 包含特征重要性排名的DataFrame
    """
    # 创建模型包装器
    model_wrapper = CuMLWrapper(model)
    
    # 将数据从GPU转换到CPU
    X_train_cpu = X_train_gpu.to_pandas()
    y_train_cpu = y_train_gpu.to_pandas()
    
    # 计算排列重要性
    result = permutation_importance(model_wrapper, X_train_cpu, y_train_cpu, n_repeats=10, random_state=42)
    importances = np.round(result.importances_mean, 6)
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })
    
    # 按重要性排序
    sorted_importance = feature_importance.sort_values('importance', ascending=False)
    
    return sorted_importance


def extract_features(station_data, is_training=True):
    """
    从光伏发电功率的时序数据中提取特征，避免数据泄漏

    参数:
        station_data (DataFrame): 包含 date_time 和 power 列的数据集
        is_training (bool): 是否为训练集，用于控制特征提取方式

    返回:
        DataFrame: 包含提取特征的数据集
    """
    # 确保 date_time 列是 datetime 类型
    station_data['date_time'] = pd.to_datetime(station_data['date_time'])
    
    # 按时间排序
    station_data = station_data.sort_values('date_time')
    
    # 添加时间特征
    #station_data['month'] = station_data['date_time'].dt.month
    #station_data['day'] = station_data['date_time'].dt.day
    station_data['hour'] = station_data['date_time'].dt.hour
    #station_data['minute'] = station_data['date_time'].dt.minute

    # 添加滞后特征
    # for lag in range(1, 5):
    #     station_data[f'lag_{lag}'] = station_data['power'].shift(lag)

    # 添加滑动窗口特征
    window_size = 4
    step_size = 2  # 步幅为2
    shifted_power = station_data['power'].shift(window_size - 1)
    for i in range(1, 5):
        station_data[f'rolling_change_rate_{i}'] = (shifted_power.rolling(window=window_size).max().shift(
            (i - 1) * step_size) - shifted_power.rolling(window=window_size).min().shift(
            (i - 1) * step_size)) / (window_size - 1)

    # 用0填充缺失值
    station_data.fillna(0, inplace=True)

    return station_data


def train(train_data, model_path, top_n_features=10):
    """
    使用cuML随机森林进行预测，并保存模型

    参数:
        train_data (DataFrame): 包含特征的训练数据集
        model_path (str): 保存模型的路径
        top_n_features (int): 要选择的最重要特征数量

    返回:
        tuple: (训练好的模型, 特征重要性DataFrame, 选定的特征列表)
    """
    train_data_copy = train_data.copy()

    # 分离特征和目标变量
    X_train = train_data_copy.drop(columns=['normalized_power', 'date_time', 'power'])
    y_train = train_data_copy['normalized_power']

    # 转换为cuDF格式
    X_train_gpu = cudf.DataFrame(X_train)
    y_train_gpu = cudf.Series(y_train)

    # 初始化cuML随机森林模型
    model = cuRFR(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_streams=1
    )

    # 训练模型
    model.fit(X_train_gpu, y_train_gpu)

    # 计算特征重要性
    feature_importance = calculate_feature_importance(model, X_train_gpu, y_train_gpu, X_train)
    print("\n特征重要性排名：")
    print(feature_importance)

    # 选择最重要的特征
    selected_features = select_features(feature_importance, top_n_features)
    print(f"\n选定的前{top_n_features}个特征：")
    print(selected_features)

    # 使用选定的特征重新训练模型
    X_train_selected = X_train[selected_features]
    X_train_gpu_selected = cudf.DataFrame(X_train_selected)
    
    model_selected = cuRFR(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_streams=1
    )
    
    model_selected.fit(X_train_gpu_selected, y_train_gpu)

    # 保存模型和选定的特征
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_info = {
        'model': model_selected,
        'selected_features': selected_features
    }
    joblib.dump(model_info, model_path)
    
    return model_selected, feature_importance, selected_features


def train_new(train_data, model_path, top_n_features=10):
    """
    使用cuML随机森林进行预测，并保存模型（新版本）

    参数:
        train_data (DataFrame): 包含特征的训练数据集
        model_path (str): 保存模型的路径
        top_n_features (int): 要选择的最重要特征数量

    返回:
        tuple: (训练好的模型, 特征重要性DataFrame, 选定的特征列表)
    """
    train_data_copy = train_data.copy()

    # 分离特征和目标变量
    X_train = train_data_copy.drop(columns=['normalized_power', 'date_time', 'power'])
    y_train = train_data_copy['normalized_power']

    # 转换为cuDF格式
    X_train_gpu = cudf.DataFrame(X_train)
    y_train_gpu = cudf.Series(y_train)

    # 初始化cuML随机森林模型
    model = cuRFR(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_streams=1,
        n_bins=128,
        max_features=1.0,
        bootstrap=True,
        max_leaves=-1,
        min_impurity_decrease=0.0,
        max_samples=1.0
    )

    # 训练模型
    model.fit(X_train_gpu, y_train_gpu)

    # 计算特征重要性
    feature_importance = calculate_feature_importance(model, X_train_gpu, y_train_gpu, X_train)
    print("\n特征重要性排名：")
    print(feature_importance)

    # 选择最重要的特征
    selected_features = select_features(feature_importance, top_n_features)
    print(f"\n选定的前{top_n_features}个特征：")
    print(selected_features)

    # 使用选定的特征重新训练模型
    X_train_selected = X_train[selected_features]
    X_train_gpu_selected = cudf.DataFrame(X_train_selected)
    
    model_selected = cuRFR(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_streams=1,
        n_bins=128,
        max_features=1.0,
        bootstrap=True,
        max_leaves=-1,
        min_impurity_decrease=0.0,
        max_samples=1.0
    )
    
    model_selected.fit(X_train_gpu_selected, y_train_gpu)

    # 保存模型和选定的特征
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_info = {
        'model': model_selected,
        'selected_features': selected_features
    }
    joblib.dump(model_info, model_path)
    
    return model_selected, feature_importance, selected_features


def predict(model_path, test_data):
    """
    使用加载的模型进行预测

    参数:
        model_path (str): 模型文件的路径
        test_data (DataFrame): 包含特征的测试数据集

    返回:
        DataFrame: 包含预测结果的数据集
    """
    # 加载模型和选定的特征
    model_info = joblib.load(model_path)
    model = model_info['model']
    selected_features = model_info['selected_features']

    # 分离特征和目标变量
    X_test = test_data.drop(columns=['normalized_power', 'date_time', 'power'], errors='ignore')
    y_test = test_data.get('normalized_power', None)

    if X_test.empty:
        return pd.DataFrame(columns=test_data.columns.tolist() + ['normalized_predicted_power'])

    # 只使用选定的特征
    X_test = X_test[selected_features]

    # 转换为cuDF格式
    X_test_gpu = cudf.DataFrame(X_test)

    # 进行预测
    y_pred = model.predict(X_test_gpu)

    # 将预测结果添加到原始数据框中
    test_data['normalized_predicted_power'] = y_pred.to_numpy()

    return test_data