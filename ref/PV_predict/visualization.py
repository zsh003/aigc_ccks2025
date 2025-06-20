import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def setup_chinese_font():
    """设置中文字体支持"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def plot_evaluation_metrics(evaluations_df, save_path):
    """
    绘制评估指标的可视化图表

    参数:
        evaluations_df (DataFrame): 包含评估指标的数据框
        save_path (str): 保存图表的路径
    """
    setup_chinese_font()

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型评估指标分析', fontsize=16)

    # 1. 准确率(C_R)和合格率(Q_R)的箱线图
    sns.boxplot(data=evaluations_df[['C_R', 'Q_R']], ax=axes[0, 0])
    axes[0, 0].set_title('准确率和合格率分布')
    axes[0, 0].set_ylabel('百分比 (%)')

    # 2. 误差指标的热力图
    error_metrics = evaluations_df[['E_rms', 'E_mae', 'E_mean']]
    sns.heatmap(error_metrics.corr(), annot=True, cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title('误差指标相关性')

    # 3. 相关系数(r)的分布
    sns.histplot(data=evaluations_df, x='r', ax=axes[1, 0])
    axes[1, 0].set_title('相关系数分布')
    axes[1, 0].set_xlabel('相关系数')
    axes[1, 0].set_ylabel('频数')

    # 4. 各电站的准确率对比
    sns.barplot(data=evaluations_df, x='station_id', y='C_R', ax=axes[1, 1])
    axes[1, 1].set_title('各电站准确率对比')
    axes[1, 1].set_xlabel('电站ID')
    axes[1, 1].set_ylabel('准确率 (%)')
    plt.xticks(rotation=45)

    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path)
    plt.close()


def plot_prediction_vs_actual(predicted_data, actual_data, station_id, save_path):
    """
    绘制预测值与实际值的对比图

    参数:
        predicted_data (Series): 预测值
        actual_data (Series): 实际值
        station_id (str): 电站ID
        save_path (str): 保存图表的路径
    """
    setup_chinese_font()
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label='实际值', alpha=0.7)
    plt.plot(predicted_data, label='预测值', alpha=0.7)
    plt.title(f'电站 {station_id} 预测值与实际值对比')
    plt.xlabel('时间')
    plt.ylabel('功率 (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(feature_importance_df, save_path):
    """
    绘制特征重要性条形图

    参数:
        feature_importance_df (DataFrame): 包含特征重要性的数据框
        save_path (str): 保存图表的路径
    """
    setup_chinese_font()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('特征重要性排名')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 