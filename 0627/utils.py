import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib


def save_model(model, tokenizer, output_dir, model_name="bert_model"):
    """
    保存transformers模型和tokenizer到指定目录
    """
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"模型和tokenizer已保存到: {model_dir}")
    return model_dir

def load_model(model_dir):
    """
    加载transformers模型和tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"模型和tokenizer已加载: {model_dir}")
    return model, tokenizer

def save_probs(oof_probs, test_probs, output_dir, prefix="bert"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"oof_{prefix}.npy"), oof_probs)
    np.save(os.path.join(output_dir, f"test_{prefix}.npy"), test_probs)
    print(f"概率文件已保存到: {output_dir}")

def load_probs(output_dir, prefix="bert"):
    oof_path = os.path.join(output_dir, f"oof_{prefix}.npy")
    test_path = os.path.join(output_dir, f"test_{prefix}.npy")
    oof_probs = np.load(oof_path) if os.path.exists(oof_path) else None
    test_probs = np.load(test_path) if os.path.exists(test_path) else None
    return oof_probs, test_probs

def save_evaluation_results(results, save_dir, prefix="bert"):
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"evaluation_{prefix}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"评估结果已保存到: {results_path}")

def evaluate_probs(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='weighted')
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = np.nan
    report = classification_report(y_true, preds, output_dict=True)
    return {"accuracy": acc, "f1": f1, "auc": auc, "report": report}

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion Matrix for {model_name} saved to: {save_path}')

def plot_roc_curve(y_true, probs, model_name, save_path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'ROC Curve for {model_name} saved to: {save_path}')

def predict(model, tokenizer, texts, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
            results.append({
                "text": text,
                "pred": pred,
                "confidence": confidence
            })
    return results

def print_prediction_results(results):
    print("\n预测结果:")
    print("-" * 50)
    for result in results:
        print(f"文本: {result['text']}")
        print(f"预测标签: {result['pred']}  置信度: {result['confidence']:.2%}")
        print("-" * 50) 