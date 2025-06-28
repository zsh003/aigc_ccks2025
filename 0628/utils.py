import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random

# 1. 传统特征提取（复用0626）
def extract_aigc_features(df, tfidf_max_features=1500):
    # 兼容pandas/cudf
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    import string, math
    from collections import Counter
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2))
    tfidf_feat = tfidf.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_feat.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])])
    def text_stats(text):
        text = str(text)
        tokens = word_tokenize(text)
        word_count = len(tokens)
        char_count = len(text)
        unique_words = set(tokens)
        type_token_ratio = len(unique_words) / (word_count + 1e-10)
        stopword_count = sum(1 for token in tokens if token.lower() in stop_words)
        stopword_ratio = stopword_count / (word_count + 1e-10)
        avg_word_length = sum(len(token) for token in tokens) / (word_count + 1e-10)
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        punctuation_ratio = punctuation_count / (char_count + 1e-10)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_sentence_length = word_count / sentence_count
        comma_freq = text.count(',') / (char_count + 1e-10)
        period_freq = text.count('.') / (char_count + 1e-10)
        question_freq = text.count('?') / (char_count + 1e-10)
        exclamation_freq = text.count('!') / (char_count + 1e-10)
        uppercase_ratio = sum(1 for char in text if char.isupper()) / (char_count + 1e-10)
        digit_ratio = sum(1 for char in text if char.isdigit()) / (char_count + 1e-10)
        repeat_word_ratio = 1 - len(unique_words) / (word_count + 1e-10) if word_count else 0
        try:
            import nltk
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)
            noun_count = sum(pos_counts.get(t, 0) for t in ['NN', 'NNS', 'NNP', 'NNPS'])
            verb_count = sum(pos_counts.get(t, 0) for t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
            adj_count = sum(pos_counts.get(t, 0) for t in ['JJ', 'JJR', 'JJS'])
            adv_count = sum(pos_counts.get(t, 0) for t in ['RB', 'RBR', 'RBS'])
        except:
            noun_count = verb_count = adj_count = adv_count = 0
        char_freq = Counter(text)
        total_chars = sum(char_freq.values())
        entropy = -sum((freq/total_chars) * math.log2(freq/total_chars) for freq in char_freq.values() if freq > 0) if total_chars else 0
        return [char_count, word_count, type_token_ratio, stopword_ratio, avg_word_length, punctuation_ratio, avg_sentence_length, comma_freq, period_freq, question_freq, exclamation_freq, uppercase_ratio, digit_ratio, repeat_word_ratio, noun_count, verb_count, adj_count, adv_count, entropy]
    stat_names = ['char_count','word_count','type_token_ratio','stopword_ratio','avg_word_length','punctuation_ratio','avg_sentence_length','comma_freq','period_freq','question_freq','exclamation_freq','uppercase_ratio','digit_ratio','repeat_word_ratio','noun_count','verb_count','adj_count','adv_count','entropy']
    stat_feat = df['text'].apply(text_stats)
    stat_df = pd.DataFrame(stat_feat.tolist(), columns=stat_names)
    all_feat = pd.concat([tfidf_df, stat_df], axis=1)
    return all_feat, list(all_feat.columns)

# 2. 对抗样本生成（简单同义词替换/噪声注入）
def synonym_augment(text):
    # 简单同义词替换（仅英文，随机替换1-2个词）
    try:
        import nlpaug.augmenter.word as naw
        aug = naw.SynonymAug(aug_src='wordnet', aug_max=2)
        return aug.augment(text)
    except:
        # 若无nlpaug则返回原文
        return text

def noise_augment(text):
    # 随机插入/删除/替换字符
    chars = list(text)
    if len(chars) < 5:
        return text
    idx = random.randint(0, len(chars)-1)
    op = random.choice(['insert','delete','replace'])
    if op=='insert':
        chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
    elif op=='delete' and len(chars)>1:
        chars.pop(idx)
    elif op=='replace':
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)

def generate_adversarial_examples(texts, method='synonym', n_aug=1):
    # texts: list[str]
    adv_texts = []
    for text in texts:
        for _ in range(n_aug):
            if method=='synonym':
                adv_texts.append(synonym_augment(text))
            elif method=='noise':
                adv_texts.append(noise_augment(text))
            else:
                adv_texts.append(text)
    return adv_texts

# 3. 测试时增强（TTA）
def tta_augment_texts(text, n_aug=5):
    # 返回原文+增强
    aug_texts = [text]
    for _ in range(n_aug):
        if random.random()<0.5:
            aug_texts.append(synonym_augment(text))
        else:
            aug_texts.append(noise_augment(text))
    return aug_texts

# 4. 持久化与评估

def save_model(model, tokenizer, output_dir, model_name="bert_model"):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"模型和tokenizer已保存到: {model_dir}")
    return model_dir

def load_model(model_dir):
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