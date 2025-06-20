import numpy as np
import pandas as pd
import os


def get_irrad_status(station_id):
    """
    读取指定station_id的CSV文件，生成一个字典，键是date_time，值表示nwp_globalirrad是否为非零

    参数:
        station_id (str): 气象站的ID，用于构造文件名

    返回:
        dict: 格式为 {date_time: bool} 的字典，nwp_globalirrad非零时为True（白天），否则为False（夜晚）
    """
    file_path = f'../dataset/{station_id}_processed.csv'
    try:
        df = pd.read_csv(file_path)
        if 'date_time' not in df.columns or 'nwp_globalirrad' not in df.columns:
            raise ValueError("CSV文件中缺少 'date_time' 或 'nwp_globalirrad' 列")
        return {
            row['date_time']: bool(row['nwp_globalirrad'] != 0.0)
            for _, row in df.iterrows()
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def evaluation(predicted_list, actual_list, constant_capacity, station_id, time_list):
    """
    计算预测功率与实际功率之间的多种评价指标（开机容量为常数），仅计算白天数据

    参数:
        predicted_list (list): 预测功率列表
        actual_list (list): 实际功率列表
        constant_capacity (float): 固定开机容量（如电站的额定容量或可用最大功率）
        station_id (str): 气象站的ID，用于获取白天/夜晚信息
        time_list (list): 时间戳列表，与predicted_list和actual_list一一对应

    返回:
        dict: 包含所有评价指标的字典（仅基于白天数据计算）
    """
    if len(actual_list) == 0:
        return None
    # 获取白天/夜晚信息字典
    not_night_dict = get_irrad_status(station_id)

    # 将输入列表转换为numpy数组，并筛选白天数据
    P_P = np.array(actual_list)
    P_M = np.array(predicted_list)
    C = constant_capacity

    # 筛选白天的数据
    day_mask = np.array([not_night_dict.get(str(time), False) for time in time_list])
    P_P_day = P_P[day_mask]
    P_M_day = P_M[day_mask]

    if len(P_P_day) == 0:
        raise ValueError("没有白天的数据可供计算")

    # 1. 均方根误差(E_rms)
    E_rms = np.sqrt(np.mean(((P_P_day - P_M_day) / C) ** 2))

    # 2. 平均绝对误差(E_mae)
    E_mae = np.mean(np.abs((P_P_day - P_M_day) / C))

    # 3. 平均误差(E_mean)
    E_mean = np.mean((P_P_day - P_M_day) / C)

    # 4. 相关系数(r)
    P_M_mean = np.mean(P_M_day)
    P_P_mean = np.mean(P_P_day)
    numerator = np.sum((P_M_day - P_M_mean) * (P_P_day - P_P_mean))
    denominator = np.sqrt(np.sum((P_M_day - P_M_mean) ** 2) * np.sum((P_P_day - P_P_mean) ** 2))
    r = numerator / denominator if denominator != 0 else 0

    # 5. 准确率(C_R)
    C_R = (1 - E_rms) * 100

    # 6. 合格率(Q_R)
    B_i = np.where(np.abs(P_P_day - P_M_day) / C < 0.25, 1, 0)
    Q_R = np.mean(B_i) * 100

    return {
        "station_id": station_id,
        "E_rms": E_rms,      # 均方根误差
        "E_mae": E_mae,      # 平均绝对误差
        "E_mean": E_mean,    # 平均误差
        "r": r,              # 相关系数
        "C_R": C_R,          # 准确率
        "Q_R": Q_R,          # 合格率
    }


def split_train_test(data):
    """
    根据 date_time 列将数据集划分为训练集和测试集。
    第 2、5、8、11 个月的最后七天数据作为测试集，其他数据作为训练集。
    每2、5、8、11 个月的最后七天数据单独返回，同时返回一个总的测试集。

    参数:
        data (DataFrame): 包含 date_time 列的数据集

    返回:
        tuple: (训练集, 测试集2月, 测试集5月, 测试集8月, 测试集11月, 总测试集)
    """
    data_split = data.copy()
    # 确保 date_time 列是 datetime 类型
    data_split['date_time'] = pd.to_datetime(data_split['date_time'])

    # 提取年份和月份
    data_split['year'] = data_split['date_time'].dt.year
    data_split['month'] = data_split['date_time'].dt.month

    # 找到每个月的最后七天
    data_split['day'] = data_split['date_time'].dt.day
    data_split['month_end'] = data_split['date_time'].dt.is_month_end

    # 筛选出第 2、5、8、11 个月的最后七天数据作为测试集
    test_data_2 = data_split[
        (data_split['month'] == 2) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 6)
        ].copy()

    test_data_5 = data_split[
        (data_split['month'] == 5) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 6)
        ].copy()

    test_data_8 = data_split[
        (data_split['month'] == 8) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 6)
        ].copy()

    test_data_11 = data_split[
        (data_split['month'] == 11) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 6)
        ].copy()

    # 如果没有筛选到数据，创建一个只有表头的空DataFrame
    if test_data_2.empty:
        test_data_2 = data_split.head(0).copy()
    if test_data_5.empty:
        test_data_5 = data_split.head(0).copy()
    if test_data_8.empty:
        test_data_8 = data_split.head(0).copy()
    if test_data_11.empty:
        test_data_11 = data_split.head(0).copy()

    # 其他数据作为训练集
    train_data = data_split[
        ~((data_split['month'].isin([2, 5, 8, 11])) &
          (data_split['day'] >= data_split['date_time'].dt.days_in_month - 6))
    ].copy()

    # 总测试集
    total_test_data = pd.concat([test_data_2, test_data_5, test_data_8, test_data_11])

    # 删除辅助列（使用赋值代替inplace）
    train_data = train_data.drop(columns=['year', 'month', 'day', 'month_end'])
    test_data_2 = test_data_2.drop(columns=['year', 'month', 'day', 'month_end'])
    test_data_5 = test_data_5.drop(columns=['year', 'month', 'day', 'month_end'])
    test_data_8 = test_data_8.drop(columns=['year', 'month', 'day', 'month_end'])
    test_data_11 = test_data_11.drop(columns=['year', 'month', 'day', 'month_end'])
    total_test_data = total_test_data.drop(columns=['year', 'month', 'day', 'month_end'])

    return train_data, test_data_2, test_data_5, test_data_8, test_data_11, total_test_data


def normalize_column(data, column_name):
    """
    对指定列进行归一化，使用 MAX-MIN 标度法

    参数:
        data (DataFrame): 包含需要归一化的列的数据集
        column_name (str): 需要归一化的列名

    返回:
        tuple: (归一化后的数据集, 最小值, 最大值)
    """
    normalized_data = data.copy()

    min_value = normalized_data[column_name].min()
    max_value = normalized_data[column_name].max()

    normalized_column_name = f"normalized_{column_name}"
    normalized_data[normalized_column_name] = (normalized_data[column_name] - min_value) / (max_value - min_value)

    return normalized_data, min_value, max_value


def inverse_normalize_column(normalized_data, column_name, min_value, max_value):
    """
    将归一化后的列逆归一化回原始值

    参数:
        data (DataFrame): 包含归一化列的数据集
        column_name (str): 原始列名
        min_value (float): 原始列的最小值
        max_value (float): 原始列的最大值

    返回:
        DataFrame: 包含逆归一化后的列的数据集
    """
    inverse_normalized_data = normalized_data.copy()

    normalized_column_name = f"normalized_{column_name}"
    inverse_normalized_data[column_name] = inverse_normalized_data[normalized_column_name] * (max_value - min_value) + min_value
    inverse_normalized_data.drop(columns=[normalized_column_name], inplace=True)

    return inverse_normalized_data


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建

    参数:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def split_train_test_new(data):
    """
    根据新的时间范围划分训练集和测试集
    测试集为8月和11月最后5天数据，其他数据作为训练集

    参数:
        data (DataFrame): 包含 date_time 列的数据集

    返回:
        tuple: (训练集, 测试集8月, 测试集11月, 总测试集)
    """
    data_split = data.copy()
    # 确保 date_time 列是 datetime 类型
    data_split['date_time'] = pd.to_datetime(data_split['date_time'])

    # 提取年份和月份
    data_split['year'] = data_split['date_time'].dt.year
    data_split['month'] = data_split['date_time'].dt.month
    data_split['day'] = data_split['date_time'].dt.day

    # 筛选出8月和11月的最后5天数据作为测试集
    test_data_8 = data_split[
        (data_split['month'] == 8) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 4)
    ].copy()

    test_data_11 = data_split[
        (data_split['month'] == 11) &
        (data_split['day'] >= data_split['date_time'].dt.days_in_month - 4)
    ].copy()

    # 如果没有筛选到数据，创建一个只有表头的空DataFrame
    if test_data_8.empty:
        test_data_8 = data_split.head(0).copy()
    if test_data_11.empty:
        test_data_11 = data_split.head(0).copy()

    # 其他数据作为训练集
    train_data = data_split[
        ~((data_split['month'].isin([8, 11])) &
          (data_split['day'] >= data_split['date_time'].dt.days_in_month - 4))
    ].copy()

    # 总测试集
    total_test_data = pd.concat([test_data_8, test_data_11])

    # 删除辅助列
    train_data = train_data.drop(columns=['year', 'month', 'day'])
    test_data_8 = test_data_8.drop(columns=['year', 'month', 'day'])
    test_data_11 = test_data_11.drop(columns=['year', 'month', 'day'])
    total_test_data = total_test_data.drop(columns=['year', 'month', 'day'])

    return train_data, test_data_8, test_data_11, total_test_data 