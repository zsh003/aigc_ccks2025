import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader

def save_model(model, tokenizer, output_dir, model_name="bert_model"):
    """
    保存模型、tokenizer和训练配置
    
    Args:
        model: 训练好的模型
        tokenizer: 对应的tokenizer
        output_dir: 输出目录
        model_name: 模型名称
    """
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型和tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 保存为pickle格式
    model_pickle_path = os.path.join(save_dir, f"{model_name}.pkl")
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {save_dir}")
    return save_dir

def load_model(model_path, model_name="bert_model"):
    """
    加载保存的模型
    
    Args:
        model_path: 模型保存路径
        model_name: 模型名称
    
    Returns:
        model: 加载的模型
        tokenizer: 对应的tokenizer
    """
    # 加载pickle模型
    model_pickle_path = os.path.join(model_path, f"{model_name}.pkl")
    with open(model_pickle_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def plot_training_history(history, save_dir=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_dir: 保存图片的目录
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['eval_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('步数')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='训练准确率')
    plt.plot(history['eval_accuracy'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('步数')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()

def evaluate_model(model, tokenizer, test_dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        tokenizer: 对应的tokenizer
        test_dataset: 测试数据集
        device: 运行设备
    
    Returns:
        dict: 评估结果
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataset:
            # 将列表转换为张量
            input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(batch['token_type_ids']).unsqueeze(0).to(device)
            
            # 使用预处理后的输入
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.append(batch['labels'])
    
    # 计算评估指标
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()
    
    return report

def save_evaluation_results(results, save_dir):
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
    # 保存为JSON格式
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"评估结果已保存到: {results_path}")

def log_training_progress(trainer, save_dir):
    """
    记录训练进度
    
    Args:
        trainer: Trainer对象
        save_dir: 保存目录
    """
    # 获取训练历史
    history = trainer.state.log_history
    
    # 提取关键指标
    train_loss = [x.get('loss', None) for x in history if 'loss' in x]
    eval_loss = [x.get('eval_loss', None) for x in history if 'eval_loss' in x]
    train_accuracy = [x.get('accuracy', None) for x in history if 'accuracy' in x]
    eval_accuracy = [x.get('eval_accuracy', None) for x in history if 'eval_accuracy' in x]
    
    # 绘制训练历史
    plot_training_history({
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'train_accuracy': train_accuracy,
        'eval_accuracy': eval_accuracy
    }, save_dir)
    
    # 保存训练日志
    log_path = os.path.join(save_dir, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    
    print(f"训练日志已保存到: {log_path}")

def predict_sentiment(model, tokenizer, texts, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用模型预测文本情感
    
    Args:
        model: 训练好的模型
        tokenizer: 对应的tokenizer
        texts: 要预测的文本列表
        device: 运行设备
    
    Returns:
        list: 预测结果列表，每个元素包含文本和预测的情感
    """
    model.to(device)
    model.eval()
    
    results = []
    with torch.no_grad():
        for text in texts:
            # 对文本进行编码
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 获取预测结果
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 获取预测概率
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence = probs[0][predictions[0]].item()
            
            # 将预测结果转换为情感标签
            sentiment = "正面" if predictions[0].item() == 1 else "负面"
            
            results.append({
                "文本": text,
                "情感": sentiment,
                "置信度": f"{confidence:.2%}"
            })
    
    return results

def print_prediction_results(results):
    """
    打印预测结果
    
    Args:
        results: 预测结果列表
    """
    print("\n预测结果:")
    print("-" * 50)
    for result in results:
        print(f"文本: {result['文本']}")
        print(f"情感: {result['情感']}")
        print(f"置信度: {result['置信度']}")
        print("-" * 50) 