import os
import json
import pandas as pd
import numpy as np
import torch
import gc
import time
from torch.utils.data import Dataset, DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 设置环境变量解决CUDA多进程问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 添加内存管理相关的环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 定义路径
MODEL_NAME = "/mnt/e/Models/Qwen/Qwen3/Qwen3-0.6B"
TRAIN_PATH = "../datasets/train/train.jsonl"
TEST_PATH = "../datasets/test_521/test.jsonl"
OUTPUT_DIR = "0714/fine_tuned_model_0.6B_full-fv_v5"
RESULT_PATH = "0714/submit_0.6B_full-fv_v5.txt"
CHECKPOINT_DIR = "0714/checkpoints_0.6B_full-fv_v5"

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ====== HF缓存配置 ======
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)

# 添加内存清理函数
def cleanup_memory():
    """强制清理内存和显存"""
    print("执行内存清理...")
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 强制垃圾回收
    gc.collect()
    
    print("内存清理完成")

def safe_trainer_cleanup(trainer):
    """安全清理Trainer对象"""
    try:
        if hasattr(trainer, 'train_dataloader'):
            # 关闭数据加载器
            if hasattr(trainer.train_dataloader, 'dataset'):
                del trainer.train_dataloader.dataset
            if hasattr(trainer.train_dataloader, 'sampler'):
                del trainer.train_dataloader.sampler
            del trainer.train_dataloader
        
        # 清理模型
        if hasattr(trainer, 'model'):
            del trainer.model
        
        # 清理优化器
        if hasattr(trainer, 'optimizer'):
            del trainer.optimizer
        
        # 清理调度器
        if hasattr(trainer, 'lr_scheduler'):
            del trainer.lr_scheduler
        
        del trainer
        print("Trainer对象已安全清理")
    except Exception as e:
        print(f"清理Trainer时出错: {e}")

# 检查GPU
def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    else:
        print("GPU不可用，使用CPU")

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
            prompt_tokens = self.tokenizer(full_prompt[:assistant_start], add_special_tokens=False)
            assistant_start_idx = len(prompt_tokens.input_ids)
            
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

def predict_test_data(model, tokenizer, test_data, batch_size=16):
    """批量预测以提高速度"""
    predictions = []
    device = model.device
    model.eval()
    
    # 分批处理
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data.iloc[i:i+batch_size]
        batch_predictions = []
        
        for _, row in batch_data.iterrows():
            text = row['text']
            prompt = f"判断以下文本是AI生成的还是人类撰写的？文本：{text}"
            messages = [{"role": "user", "content": prompt}]
            
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text_input, return_tensors="pt")
            # 确保输入在正确的设备上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,  # 启用采样
                    temperature=0.7,  # 控制随机性
                    top_p=0.9,  #  nucleus sampling
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 修复：正确处理inputs字典
            input_length = inputs['input_ids'].shape[1]
            output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            # 调试信息（前几个样本）
            if i < 50:  # 只显示前50个样本的调试信息
                print(f"样本 {i+1}: 输入长度={input_length}, 输出长度={len(outputs[0])}")
                print(f"  输出文本: '{output_text}'")
            
            # 检查输出中是否包含"AI生成"或"人类撰写"
            if "AI生成" in output_text:
                batch_predictions.append(1)
            elif "人类撰写" in output_text:
                batch_predictions.append(0)
            else:
                # 如果不确定，根据输出内容进行进一步分析
                if any(word in output_text.lower() for word in ["ai", "机器", "模型", "生成", "自动"]):
                    batch_predictions.append(1)
                else:
                    batch_predictions.append(0)
            
            # 清理内存
            del inputs, outputs
            torch.cuda.empty_cache()
        
        predictions.extend(batch_predictions)
        
        # 显示进度
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(test_data):
            print(f"预测进度: {min(i + batch_size, len(test_data))}/{len(test_data)}")
    
    return predictions

def check_model_file_size(file_path):
    """检查模型文件大小，确保不是空文件"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    # 如果文件小于1KB，认为是空文件
    if file_size < 1024:
        print(f"警告: 文件 {file_path} 太小 ({file_size} bytes)，可能是空文件")
        return False
    
    return True

def is_model_saved(model_dir):
    """检查模型是否已保存"""
    if not os.path.exists(model_dir):
        return False
    
    # 检查pytorch_model.bin文件
    model_file = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.exists(model_file) and check_model_file_size(model_file):
        print(f"检测到已保存的全量模型: {model_dir}")
        return True
    else:
        print(f"未检测到全量模型: {model_dir}")
        return False

def main():
    all_start = time.time()
    
    # 检查GPU
    check_gpu()
    
    # 确保CUDA正确初始化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA初始化完成，当前设备: {torch.cuda.current_device()}")
    
    # 加载数据
    print("加载数据...")
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train = [json.loads(line) for line in f.readlines()]
        train_df = pd.DataFrame(train)
    
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
        test_df = pd.DataFrame(test)
    
    print(f"训练数据: {len(train_df)} 条")
    print(f"测试数据: {len(test_df)} 条")
    
    # 检查是否已有训练好的模型
    if is_model_saved(OUTPUT_DIR):
        print(f"检测到已保存的全量模型，直接加载进行预测: {OUTPUT_DIR}")
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None)
        # 添加dropout层
        model.lm_head.dropout = torch.nn.Dropout(p=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("✓ 全量模型加载成功")
        # 预测测试集
        print("预测测试集...")
        predictions = predict_test_data(model, tokenizer, test_df, batch_size=16)
        # 保存预测结果
        print("保存预测结果...")
        with open(RESULT_PATH, "w") as file:
            for label in predictions:
                file.write(str(label) + "\n")
        print(f"预测结果已保存至 {RESULT_PATH}")
        return
    
    # 加载模型和分词器
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 加载基础模型 - 全参数微调
    print("加载基础模型进行全参数微调...")
    use_bf16 = False
    use_fp16 = False
    try:
        # 检查BF16支持情况
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("使用BF16精度加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
            )
            use_bf16 = True
        else:
            print("使用FP16精度加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map=None
            )
            use_fp16 = True
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用trust_remote_code=True...")
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
                )
                use_bf16 = True
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
                )
                use_fp16 = True
            print("✓ 模型加载成功")
        except Exception as e2:
            print(f"第二次尝试失败: {e2}")
            print("尝试使用默认精度...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, device_map=None, trust_remote_code=True
            )
            print("✓ 模型加载成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("✓ 全量模型加载成功")
    
    # 创建训练数据集
    train_dataset = AIGCDetectionDataset(train_df, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,  # 降低学习率 
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,  # 增加预热步数
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        remove_unused_columns=False,
        no_cuda=False,
        label_names=["labels"],
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=True if torch.cuda.is_available() else False,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        eval_accumulation_steps=2,
        report_to=[],
        disable_tqdm=False,
        group_by_length=True,
        length_column_name="length",
        ignore_data_skip=False,
        dataloader_drop_last=True,
    )
    
    def collate_fn(data):
        batch = {
            'input_ids': torch.stack([x['input_ids'] for x in data]),
            'attention_mask': torch.stack([x['attention_mask'] for x in data]),
            'labels': torch.stack([x['labels'] for x in data])
        }
        return batch
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    print("\n=== 模型调试信息 ===")
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
    sample = train_dataset[0]
    print(f"样本键: {sample.keys()}")
    print(f"input_ids形状: {sample['input_ids'].shape}")
    print(f"labels形状: {sample['labels'].shape}")
    print(f"labels中非-100的数量: {(sample['labels'] != -100).sum()}")
    model.eval()
    with torch.no_grad():
        sample_input = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        outputs = model(**sample_input)
        print(f"模型输出logits形状: {outputs.logits.shape}")
    print("=== 调试信息结束 ===\n")
    
    print("开始训练模型...")
    print(f"训练参数: batch_size={training_args.per_device_train_batch_size}, "
          f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}, "
          f"effective_batch_size={training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    try:
        trainer.train()
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练被中断，正在清理资源...")
        cleanup_memory()
        trainer.save_model(os.path.join(CHECKPOINT_DIR, "interrupted"))
        print("检查点已保存，可以稍后继续训练")
        safe_trainer_cleanup(trainer)
        import multiprocessing
        try:
            multiprocessing.active_children()
            for p in multiprocessing.active_children():
                print(f"终止子进程: {p.pid}")
                p.terminate()
        except Exception as e:
            print(f"终止子进程时出错: {e}")
        return
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("正在清理资源...")
        cleanup_memory()
        trainer.save_model(os.path.join(CHECKPOINT_DIR, "error"))
        safe_trainer_cleanup(trainer)
        import multiprocessing
        try:
            multiprocessing.active_children()
            for p in multiprocessing.active_children():
                print(f"终止子进程: {p.pid}")
                p.terminate()
        except Exception as e:
            print(f"终止子进程时出错: {e}")
        return
    print("保存最终模型...")
    trainer.save_model(OUTPUT_DIR)
    print("清理训练资源...")
    safe_trainer_cleanup(trainer)
    torch.cuda.empty_cache()
    gc.collect()
    import multiprocessing
    try:
        multiprocessing.active_children()
        for p in multiprocessing.active_children():
            print(f"终止子进程: {p.pid}")
            p.terminate()
    except Exception as e:
        print(f"终止子进程时出错: {e}")
    print("使用已加载的模型进行预测...")
    model = model.to(device)
    print("预测测试集...")
    predictions = predict_test_data(model, tokenizer, test_df, batch_size=32)
    print("保存预测结果...")
    with open(RESULT_PATH, "w") as file:
        for label in predictions:
            file.write(str(label) + "\n")
    print(f"预测结果已保存至 {RESULT_PATH}")
    all_end = time.time()
    print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

if __name__ == "__main__":
    main() 