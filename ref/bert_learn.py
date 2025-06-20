import torch
#!python --version
# Python 3.11.12
print(f"CUDA available: {torch.cuda.is_available()}")  # 输出基础检测结果
print(f"CUDA device count: {torch.cuda.device_count()}")  # 显示可用设备数量
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")  # 显示设备名称
    print(f"CUDA version: {torch.version.cuda}")  # 显示CUDA版本
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

"""
CUDA available: True
CUDA device count: 1
Using CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA version: 12.6
"""

"""!pip list |grep torch
torch                     2.7.1+cu126
torchaudio                2.7.1+cu126
torchvision               0.22.1+cu126
"""
"""!pip list |grep transformer
transformers              4.52.4
"""

# Hugging face缓存配置
import os
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)

# 加载bert-uncased模型
from transformers import AutoTokenizer, BertModel
BASE_MODEL = 'bert-base-uncased'
print(f"Loading base model: {BASE_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, cache_dir=HF_CACHE_DIR)
except OSError:
    print(f"Base model not found locally. Downloading from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR)
        print("Download successfully!")
    except:
        print("Download error!")

model = BertModel.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR)

# 预处理输入
sentence="I love Paris"
tokens = tokenizer.tokenize(sentence)
print(tokens)
# ['i', 'love', 'paris']

tokens = ['[CLS]']+tokens+['[SEP]']
print(tokens)
# ['[CLS]', 'i', 'love', 'paris', '[SEP]']

tokens = tokens + ['[PAD]'] + ['[PAD]']
attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]
print(attention_mask)
# [1, 1, 1, 1, 1, 0, 0]

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# [101, 1045, 2293, 3000, 102, 0, 0]

token_ids=torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)
print(attention_mask)
# tensor([[1, 1, 1, 1, 1, 0, 0]])

hidden_rep, cls_head = model(token_ids, attention_mask=attention_mask, return_dict=False)
print(hidden_rep.shape)
print(cls_head.shape)
# torch.Size([1, 7, 768])
# torch.Size([1, 768])

tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, cache_dir=HF_CACHE_DIR)
model = BertModel.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR, output_hidden_states=True)

# 得到嵌入表示
last_hidden_state, pooler_output, hidden_states = model(token_ids, attention_mask=attention_mask, return_dict=False)
print(last_hidden_state.shape)
print(pooler_output.shape)
print(len(hidden_states))
print(hidden_states[0].shape)

# torch.Size([1, 7, 768])
# torch.Size([1, 768])
# 13
# torch.Size([1, 7, 768])

last_hidden_state[0][1][:5] #i
# tensor([ 0.2236,  0.6536, -0.2294, -0.4487, -0.0956], grad_fn=<SliceBackward0>)




# 微调
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import save_model, load_model, evaluate_model, save_evaluation_results, log_training_progress

TRAIN_DATA_FILE = r'../datasets/IMDB Dataset.csv'
df = pd.read_csv(TRAIN_DATA_FILE)
print(df.info())

# 标签编码
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
print("标签映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 将 Pandas DataFrame 转换为 Hugging Face Dataset 对象
train_set = Dataset.from_pandas(train_df)
val_set = Dataset.from_pandas(val_df)

def preprocess(data):
    # 使用与示例相同的预处理方式
    tokenized = tokenizer(
        data['review'],
        padding="max_length",
        truncation=True,
        max_length=512,  # 设置最大长度
        return_tensors="pt"  # 返回PyTorch张量
    )
    # 将 sentiment 映射为 labels
    tokenized["labels"] = data["sentiment"]
    return tokenized

train_set = train_set.map(preprocess, batched=True)
val_set = val_set.map(preprocess, batched=True)

# 移除不需要的列
columns_to_remove = ['__index_level_0__', 'review', 'sentiment'] if '__index_level_0__' in train_set.column_names else ['review', 'sentiment']
train_set = train_set.remove_columns(columns_to_remove)
val_set = val_set.remove_columns(columns_to_remove)



# 初始化分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,  # 二分类任务
    cache_dir=HF_CACHE_DIR
)

OUTPUT_DIR = "./result_bert_finetune"
LOGGING_DIR = "./logs_bert"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    optim="adamw_torch",
    save_steps=50,
    eval_strategy="steps",
    save_strategy="steps",             # 每个step保存模型
    load_best_model_at_end=True,       # 训练结束时加载最佳模型
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results =  metric.compute(predictions=predictions, references=labels)
    return results

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()
print("模型训练完成。")

# 保存模型和训练日志
save_dir = save_model(model, tokenizer, OUTPUT_DIR, "bert_sentiment")
log_training_progress(trainer, save_dir)

# 加载保存的模型进行评估
print("\n加载保存的模型...")
model, tokenizer = load_model(save_dir, "bert_sentiment")
print("模型加载完成。")

# 评估模型
print("\n开始评估模型...")
eval_results = evaluate_model(model, tokenizer, val_set)
save_evaluation_results(eval_results, save_dir)

print("\n评估结果摘要:")
print(f"准确率: {eval_results['accuracy']:.4f}")
print(f"宏平均F1分数: {eval_results['macro avg']['f1-score']:.4f}")
print(f"加权平均F1分数: {eval_results['weighted avg']['f1-score']:.4f}")

# 测试模型预测
print("\n开始测试模型预测...")
test_texts = [
    "I love Paris, it's a beautiful city with amazing architecture and culture.",
    "This movie was terrible, I couldn't watch it till the end.",
    "The food at this restaurant is absolutely delicious!",
    "I had a really bad experience with their customer service.",
    "The weather is perfect today, I'm having a great time!",
    "This product is not worth the money, very disappointed.",
    "I love Paris",  # 简单示例
    "I hate this movie",  # 简单示例
]

from utils import predict_sentiment, print_prediction_results

# 预测并打印结果
results = predict_sentiment(model, tokenizer, test_texts)
print_prediction_results(results)

# 保存预测结果
import json
prediction_results_path = os.path.join(save_dir, 'prediction_results.json')
with open(prediction_results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"\n预测结果已保存到: {prediction_results_path}")

