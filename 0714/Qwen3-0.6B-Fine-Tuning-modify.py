import os
import json
import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义路径
MODEL_NAME = "./Qwen3-0.6B"  # 修改为0.6B模型路径
TRAIN_PATH = "./datasets/train/train.jsonl"
TEST_PATH = "./datasets/test_521/test.jsonl"
OUTPUT_DIR = "./fine_tuned_model"
RESULT_PATH = "./submit.txt"

# 定义数据集类
class AIGCDetectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['text']
        
        # 构建提示模板
        prompt = f"判断以下文本是AI生成的还是人类撰写的？文本：{text}"
        
        if 'label' in item:
            label = item['label']
            answer = "AI生成" if label == 1 else "人类撰写"
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        else:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        encodings = self.tokenizer(full_prompt, 
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding="max_length",
                                  return_tensors="pt")
        
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # 对于训练数据，我们需要设置标签
        if 'label' in item:
            # 找到assistant回答的位置
            assistant_start = full_prompt.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            assistant_text = full_prompt[assistant_start:]
            
            # 对assistant部分进行编码
            assistant_encodings = self.tokenizer(assistant_text, add_special_tokens=False)
            assistant_ids = assistant_encodings.input_ids
            
            # 创建标签，设置-100为不计算损失的标记
            labels = torch.ones_like(input_ids) * -100
            
            # 找到input_ids中assistant回答的起始位置
            assistant_start_idx = len(encodings.input_ids[0]) - len(assistant_ids)
            
            # 设置assistant回答部分的标签
            for i, token_id in enumerate(assistant_ids):
                if assistant_start_idx + i < len(labels):
                    labels[assistant_start_idx + i] = token_id
                    
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

# 定义评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算准确率、精确率、召回率和F1值
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def predict_test_data(model, tokenizer, test_data):
    predictions = []
    device = model.device
    
    for _, row in test_data.iterrows():
        text = row['text']
        prompt = f"判断以下文本是AI生成的还是人类撰写的？文本：{text}"
        messages = [{"role": "user", "content": prompt}]
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # 移除不支持的生成参数，只保留必要的参数
            outputs = model.generate(
                **inputs,
                max_new_tokens=20
            )
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 检查输出中是否包含"AI生成"或"人类撰写"
        if "AI生成" in output_text:
            predictions.append(1)
        elif "人类撰写" in output_text:
            predictions.append(0)
        else:
            # 如果不确定，根据输出内容进行进一步分析
            if any(word in output_text.lower() for word in ["ai", "机器", "模型", "生成", "自动"]):
                predictions.append(1)
            else:
                predictions.append(0)
    
    return predictions

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Qwen3-0.6B模型微调与预测")
    parser.add_argument("--train", action="store_true", help="是否进行训练")
    parser.add_argument("--predict", action="store_true", help="是否进行预测")
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行全流程
    if not args.train and not args.predict:
        args.train = True
        args.predict = True
    
    # 加载测试数据（预测时需要）
    if args.predict:
        print("加载测试数据...")
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            test = [json.loads(line) for line in f.readlines()]
            test_df = pd.DataFrame(test)
    
    # 加载分词器（训练和预测都需要）
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if args.train:
        # 加载训练数据
        print("加载训练数据...")
        with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
            train = [json.loads(line) for line in f.readlines()]
            train_df = pd.DataFrame(train)
        
        # 创建训练数据集
        train_dataset = AIGCDetectionDataset(train_df, tokenizer)
        
        # 应用LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 增加rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 扩展目标模块
            bias="none",  # 不训练bias
            use_rslora=False,
        )
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=None
        )
        
        # 应用LoRA配置
        model = get_peft_model(model, peft_config)
        
    # 设置训练参数（针对RTX4060 8GB优化）
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,  # 减少batch_size避免显存不足
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # 增加梯度累积步数
            learning_rate=2e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,  # 添加梯度裁剪
            warmup_steps=50,    # 引入学习率预热
            save_strategy="steps",
            save_steps=500,     # 每500步保存一次
            save_total_limit=3, # 保留3个检查点
            logging_steps=100,
            remove_unused_columns=False,
            no_cuda=False,
            label_names=["labels"],
            # GPU加速设置
            fp16=True if torch.cuda.is_available() else False,
            tf32=True if torch.cuda.is_available() else False,
            gradient_checkpointing=False,  # 暂时禁用梯度检查点
            optim="adamw_torch",
            dataloader_pin_memory=False,  # 禁用pin_memory避免CUDA张量问题
            dataloader_num_workers=0,    # 禁用多进程数据加载
            eval_accumulation_steps=2,   # 减少评估时的内存占用
            report_to=[],  # 禁用wandb等报告工具
            disable_tqdm=False,  # 启用进度条
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([x['input_ids'] for x in data]),
                                       'attention_mask': torch.stack([x['attention_mask'] for x in data]),
                                       'labels': torch.stack([x['labels'] for x in data])}
        )
        
        # 开始训练
        print("开始训练模型...")
        trainer.train()
        
        # 保存最终模型
        print("保存模型...")
        trainer.save_model(OUTPUT_DIR)
    
    if args.predict:
        # 加载微调后的模型进行预测
        print("加载微调后的模型进行预测...")
        model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            torch_dtype=torch.float16,
            device_map="auto"  # 自动选择设备
        )
        
        # 预测测试集
        print("预测测试集...")
        predictions = predict_test_data(model, tokenizer, test_df)
        
        # 保存预测结果
        print("保存预测结果...")
        with open(RESULT_PATH, "w") as file:
            for label in predictions:
                file.write(str(label) + "\n")
        
        print(f"预测结果已保存至 {RESULT_PATH}")

if __name__ == "__main__":
    main()
