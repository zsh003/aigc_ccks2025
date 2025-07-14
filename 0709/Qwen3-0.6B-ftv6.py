import os
import json
import pandas as pd
import numpy as np
import torch
import gc
import time
import logging
from torch.utils.data import Dataset, DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from sklearn.model_selection import KFold

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
OUTPUT_DIR = "0709/fine_tuned_model_0.6B_v6"
RESULT_PATH = "0709/submit_0.6B_v6.txt"
CHECKPOINT_DIR = "0709/checkpoints_0.6B_v6"
LOG_PATH = "0709/training_log_v6.txt"

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info("执行内存清理...")
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 强制垃圾回收
    gc.collect()
    
    logger.info("内存清理完成")

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
        logger.info("Trainer对象已安全清理")
    except Exception as e:
        logger.error(f"清理Trainer时出错: {e}")

# 检查GPU
def check_gpu():
    if torch.cuda.is_available():
        logger.info(f"GPU可用: {torch.cuda.get_device_name()}")
        logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    else:
        logger.info("GPU不可用，使用CPU")

# 添加自定义回调类来记录训练参数
class TrainingLoggerCallback(TrainerCallback):
    """自定义回调类，用于记录训练过程中的详细参数"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """当有新的日志时调用"""
        if logs is not None:
            # 记录训练参数到日志文件
            log_message = "训练参数更新: "
            for key, value in logs.items():
                if isinstance(value, float):
                    log_message += f"{key}={value:.6f}, "
                else:
                    log_message += f"{key}={value}, "
            logger.info(log_message.rstrip(", "))
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """每步结束时调用"""
        if logs is not None and 'loss' in logs:
            logger.info(f"步骤 {state.global_step}: loss={logs['loss']:.6f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时调用"""
        if metrics is not None:
            logger.info("验证集评估结果:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")

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
            # 更精确地定位assistant回答的开始位置
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
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 修复：正确处理inputs字典
            input_length = inputs['input_ids'].shape[1]
            output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            # 调试信息（前几个样本）
            if i < 50:  # 只显示前50个样本的调试信息
                logger.info(f"样本 {i+1}: 输入长度={input_length}, 输出长度={len(outputs[0])}")
                logger.info(f"  输出文本: '{output_text}'")
            
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
            logger.info(f"预测进度: {min(i + batch_size, len(test_data))}/{len(test_data)}")
    
    return predictions

def check_model_file_size(file_path):
    """检查模型文件大小，确保不是空文件"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    # 如果文件小于1KB，认为是空文件
    if file_size < 1024:
        logger.warning(f"警告: 文件 {file_path} 太小 ({file_size} bytes)，可能是空文件")
        return False
    
    return True

def is_model_saved(model_dir):
    """检查模型是否已保存"""
    if not os.path.exists(model_dir):
        return False
    
    # 检查LoRA适配器文件
    adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
    adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))
    
    # 检查目录是否为空
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        if len(files) == 0:
            logger.warning(f"目录为空: {model_dir}")
            return False
    
    # 检查是否有有效的LoRA适配器文件
    if adapter_config_exists and adapter_model_exists:
        # 检查adapter_model.safetensors文件大小
        adapter_model_path = os.path.join(model_dir, 'adapter_model.safetensors')
        if check_model_file_size(adapter_model_path):
            logger.info(f"检测到有效LoRA适配器: {model_dir}")
            logger.info(f"  - adapter_config.json: {adapter_config_exists}")
            logger.info(f"  - adapter_model.safetensors: {adapter_model_exists}")
            return True
        else:
            logger.warning(f"LoRA适配器文件无效: {adapter_model_path}")
            return False
    else:
        logger.warning(f"目录存在但缺少LoRA适配器文件: {model_dir}")
        logger.warning(f"  - adapter_config.json: {adapter_config_exists}")
        logger.warning(f"  - adapter_model.safetensors: {adapter_model_exists}")
        return False

class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger_func(message)
    def flush(self):
        pass

def main():
    # 重定向所有标准输出和错误到日志
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)
    
    all_start = time.time()
    
    # 检查GPU
    check_gpu()
    
    # 确保CUDA正确初始化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA初始化完成，当前设备: {torch.cuda.current_device()}")
    
    # 加载数据
    logger.info("加载数据...")
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train = [json.loads(line) for line in f.readlines()]
        train_df = pd.DataFrame(train)
    
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
        test_df = pd.DataFrame(test)
    
    logger.info(f"训练数据: {len(train_df)} 条")
    logger.info(f"测试数据: {len(test_df)} 条")
    
    # 3折交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
        logger.info(f"\n===== Fold {fold+1}/3 =====")
        fold_output_dir = OUTPUT_DIR + f"_fold{fold+1}"
        fold_checkpoint_dir = CHECKPOINT_DIR + f"_fold{fold+1}"
        fold_result_path = RESULT_PATH.replace('.txt', f'_fold{fold+1}.txt')
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        # 检查是否已有训练好的模型
        if is_model_saved(fold_output_dir):
            logger.info(f"Fold {fold+1}: 检测到已保存的LoRA适配器，直接加载进行预测: {fold_output_dir}")
            # 验证LoRA适配器完整性
            try:
                logger.info(f"Fold {fold+1}: 加载基础模型: {MODEL_NAME}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map=None
                )
                logger.info(f"Fold {fold+1}: 加载LoRA适配器: {fold_output_dir}")
                model = PeftModel.from_pretrained(base_model, fold_output_dir)
                logger.info(f"Fold {fold+1}: ✓ LoRA适配器加载成功")
                del base_model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Fold {fold+1}: LoRA适配器验证失败: {e}")
                logger.info("将重新训练模型...")
                import shutil
                if os.path.exists(fold_output_dir):
                    shutil.rmtree(fold_output_dir)
                    logger.info(f"已删除不完整的适配器目录: {fold_output_dir}")
        
        # 划分本折的训练集和验证集
        train_fold_df = train_df.iloc[train_index].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_index].reset_index(drop=True)
        
        # 加载模型和分词器
        logger.info(f"Fold {fold+1}: 加载模型和分词器...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = AIGCDetectionDataset(train_fold_df, tokenizer)
        val_dataset = AIGCDetectionDataset(val_fold_df, tokenizer)
        logger.info(f"Fold {fold+1}: 加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=None
        )
        for param in model.parameters():
            param.requires_grad = False
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_rslora=False,
        )
        model = get_peft_model(model, peft_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.print_trainable_parameters()
        training_args = TrainingArguments(
            output_dir=fold_checkpoint_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_steps=50,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            logging_steps=50,
            remove_unused_columns=False,
            no_cuda=False,
            label_names=["labels"],
            fp16=True if torch.cuda.is_available() else False,
            tf32=True if torch.cuda.is_available() else False,
            optim="adamw_torch",
            dataloader_pin_memory=True,
            gradient_checkpointing=False,
            dataloader_num_workers=8,
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
        logger.info(f"Fold {fold+1}: 当前设备: {device}")
        logger.info(f"Fold {fold+1}: CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Fold {fold+1}: 当前CUDA设备: {torch.cuda.current_device()}")
            logger.info(f"Fold {fold+1}: GPU名称: {torch.cuda.get_device_name()}")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[TrainingLoggerCallback()],
        )
        logger.info(f"Fold {fold+1}: \n=== 模型调试信息 ===")
        logger.info(f"Fold {fold+1}: 模型设备: {next(model.parameters()).device}")
        logger.info(f"Fold {fold+1}: 可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        logger.info(f"Fold {fold+1}: 总参数数量: {sum(p.numel() for p in model.parameters())}")
        sample = train_dataset[0]
        logger.info(f"Fold {fold+1}: 样本键: {sample.keys()}")
        logger.info(f"Fold {fold+1}: input_ids形状: {sample['input_ids'].shape}")
        logger.info(f"Fold {fold+1}: labels形状: {sample['labels'].shape}")
        logger.info(f"Fold {fold+1}: labels中非-100的数量: {(sample['labels'] != -100).sum()}")
        model.eval()
        with torch.no_grad():
            sample_input = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            outputs = model(**sample_input)
            logger.info(f"Fold {fold+1}: 模型输出logits形状: {outputs.logits.shape}")
        logger.info(f"Fold {fold+1}: === 调试信息结束 ===\n")
        # 检查是否已保存模型，未保存则训练
        if not is_model_saved(fold_output_dir):
            logger.info(f"Fold {fold+1}: 开始训练模型...")
            logger.info(f"Fold {fold+1}: 训练参数: batch_size={training_args.per_device_train_batch_size}, "
                        f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}, "
                        f"effective_batch_size={training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
            try:
                trainer.train()
                logger.info(f"Fold {fold+1}: 训练完成！")
            except KeyboardInterrupt:
                logger.info(f"Fold {fold+1}: 训练被中断，正在清理资源...")
                cleanup_memory()
                trainer.save_model(os.path.join(fold_checkpoint_dir, "interrupted"))
                logger.info(f"Fold {fold+1}: 检查点已保存，可以稍后继续训练")
                safe_trainer_cleanup(trainer)
                import multiprocessing
                try:
                    multiprocessing.active_children()
                    for p in multiprocessing.active_children():
                        logger.info(f"终止子进程: {p.pid}")
                        p.terminate()
                except Exception as e:
                    logger.error(f"终止子进程时出错: {e}")
                return
            except Exception as e:
                logger.error(f"Fold {fold+1}: 训练过程中出现错误: {e}")
                logger.info(f"Fold {fold+1}: 正在清理资源...")
                cleanup_memory()
                trainer.save_model(os.path.join(fold_checkpoint_dir, "error"))
                safe_trainer_cleanup(trainer)
                import multiprocessing
                try:
                    multiprocessing.active_children()
                    for p in multiprocessing.active_children():
                        logger.info(f"终止子进程: {p.pid}")
                        p.terminate()
                except Exception as e:
                    logger.error(f"终止子进程时出错: {e}")
                return
            logger.info(f"Fold {fold+1}: 保存最终模型...")
            trainer.save_model(fold_output_dir)
            logger.info(f"Fold {fold+1}: 清理训练资源...")
            safe_trainer_cleanup(trainer)
            torch.cuda.empty_cache()
            gc.collect()
            import multiprocessing
            try:
                multiprocessing.active_children()
                for p in multiprocessing.active_children():
                    logger.info(f"终止子进程: {p.pid}")
                    p.terminate()
            except Exception as e:
                logger.error(f"终止子进程时出错: {e}")
        # 预测测试集
        logger.info(f"Fold {fold+1}: 使用已加载的模型进行预测...")
        model = model.to(device)
        logger.info(f"Fold {fold+1}: 预测测试集...")
        predictions = predict_test_data(model, tokenizer, test_df, batch_size=16)
        logger.info(f"Fold {fold+1}: 保存预测结果...")
        with open(fold_result_path, "w") as file:
            for label in predictions:
                file.write(str(label) + "\n")
        logger.info(f"Fold {fold+1}: 预测结果已保存至 {fold_result_path}")
    all_end = time.time()
    logger.info(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

    # 合并3折预测结果
    logger.info("开始合并3折预测结果...")
    fold_result_files = [RESULT_PATH.replace('.txt', f'_fold{i+1}.txt') for i in range(3)]
    fold_preds = []
    for file in fold_result_files:
        with open(file, 'r') as f:
            preds = [int(line.strip()) for line in f.readlines()]
            fold_preds.append(preds)
    # 转置，按样本聚合
    merged = []
    for preds_per_sample in zip(*fold_preds):
        # 多数投票
        if sum(preds_per_sample) >= 2:
            merged.append(1)
        else:
            merged.append(0)
    # 保存最终合并结果
    with open(RESULT_PATH, 'w') as f:
        for label in merged:
            f.write(str(label) + '\n')
    logger.info(f"已合并3折预测结果并保存至 {RESULT_PATH}")

if __name__ == "__main__":
    main()
