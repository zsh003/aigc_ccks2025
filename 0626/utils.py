import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import string
import math
from collections import Counter
import nltk

def setup_nltk(nltk_data_path='../nltk_data'):
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)

def extract_aigc_features(df, tfidf_max_features=1500):
    setup_nltk()
    from nltk.tokenize import word_tokenize, sent_tokenize
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set(ENGLISH_STOP_WORDS)
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


def plot_result_bar(result_dict, title, ylabel, save_path):
    """Plots a bar chart from a dictionary of results, with pretty x labels."""
    import matplotlib.pyplot as plt
    # 自动调整宽度，最小8，最多每个模型0.7宽度
    plt.figure(figsize=(max(8, len(result_dict)*0.7), 5))
    # 缩短标签
    names = [name if len(name) <= 12 else name[:10]+'…' for name in result_dict.keys()]
    values = list(result_dict.values())
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') 
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Visualization "{title}" saved to: {save_path}')

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plots and saves a confusion matrix."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion Matrix for {model_name} saved to: {save_path}')

def plot_roc_curves(y_true, oof_probs_dict, save_path):
    """Plots ROC curves for multiple models on the same axes."""
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 7))
    for model_name, oof_probs in oof_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, oof_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'ROC Curves saved to: {save_path}') 