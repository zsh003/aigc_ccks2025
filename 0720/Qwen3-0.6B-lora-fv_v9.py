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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import batch_predict, predict_one_model, multiprocess_predict, BASE_MODEL_PATH

# 设置环境变量解决CUDA多进程问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 定义路径
MODEL_NAME = BASE_MODEL_PATH
TRAIN_PATH = "../datasets/train/train.jsonl"
TEST_PATH = "../datasets/test_717/test.jsonl"
OUTPUT_DIR = "0720/fine_tuned_model_0.6B_lora-fv_v9"
RESULT_PATH = "0720/submit_0.6B_lora-fv_v9.txt"
CHECKPOINT_DIR = "0720/checkpoints_0.6B_lora-fv_v9"
MULTI_RESULT_DIR = "0720/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MULTI_RESULT_DIR, exist_ok=True)


# ====== HF缓存配置 ======
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)

def cleanup_memory():
    print("执行内存清理...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("内存清理完成")

def safe_trainer_cleanup(trainer):
    try:
        if hasattr(trainer, 'train_dataloader'):
            if hasattr(trainer.train_dataloader, 'dataset'):
                del trainer.train_dataloader.dataset
            if hasattr(trainer.train_dataloader, 'sampler'):
                del trainer.train_dataloader.sampler
            del trainer.train_dataloader
        if hasattr(trainer, 'model'):
            del trainer.model
        if hasattr(trainer, 'optimizer'):
            del trainer.optimizer
        if hasattr(trainer, 'lr_scheduler'):
            del trainer.lr_scheduler
        del trainer
        print("Trainer对象已安全清理")
    except Exception as e:
        print(f"清理Trainer时出错: {e}")

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        # 检查是否支持BF16
        if torch.cuda.is_bf16_supported():
            print("✓ 支持BF16精度")
        else:
            print("✗ 不支持BF16精度，将使用FP16")
    else:
        print("GPU不可用，使用CPU")

class AIGCDetectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['text']
        prompt = f"判断以下文本是AI生成的还是人类撰写的？文本：{text}"
        if 'label' in item:
            label = item['label']
            answer = "AI生成" if label == 1 else "人类撰写"
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        else:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        encodings = self.tokenizer(full_prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        if 'label' in item:
            assistant_start = full_prompt.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            assistant_text = full_prompt[assistant_start:]
            assistant_encodings = self.tokenizer(assistant_text, add_special_tokens=False)
            assistant_ids = assistant_encodings.input_ids
            labels = torch.ones_like(input_ids) * -100
            prompt_tokens = self.tokenizer(full_prompt[:assistant_start], add_special_tokens=False)
            assistant_start_idx = len(prompt_tokens.input_ids)
            for i, token_id in enumerate(assistant_ids):
                if assistant_start_idx + i < len(labels):
                    labels[assistant_start_idx + i] = token_id
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(eval_pred):
    """修复numpy转换错误的评估函数"""
    predictions, labels = eval_pred
    
    # 确保数据类型正确
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # 将predictions转换为numpy数组
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    else:
        predictions = np.array(predictions)
    
    # 将labels转换为numpy数组
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    elif isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.array(labels)
    
    # 确保是2D数组
    if len(predictions.shape) == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    if len(labels.shape) == 2:
        labels = labels.reshape(-1)
    
    # 获取预测类别
    predictions = np.argmax(predictions, axis=1)
    
    # 过滤掉-100的标签（不计算损失的标记）
    valid_mask = labels != -100
    if valid_mask.sum() > 0:
        valid_predictions = predictions[valid_mask]
        valid_labels = labels[valid_mask]
        
        accuracy = accuracy_score(valid_labels, valid_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_predictions, average='binary')
    else:
        accuracy = precision = recall = f1 = 0.0
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def check_model_file_size(file_path):
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    if file_size < 1024:
        print(f"警告: 文件 {file_path} 太小 ({file_size} bytes)，可能是空文件")
        return False
    return True

def is_model_saved(model_dir):
    if not os.path.exists(model_dir):
        return False
    adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
    adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        if len(files) == 0:
            print(f"目录为空: {model_dir}")
            return False
    if adapter_config_exists and adapter_model_exists:
        adapter_model_path = os.path.join(model_dir, 'adapter_model.safetensors')
        if check_model_file_size(adapter_model_path):
            print(f"检测到有效LoRA适配器: {model_dir}")
            print(f"  - adapter_config.json: {adapter_config_exists}")
            print(f"  - adapter_model.safetensors: {adapter_model_exists}")
            return True
        else:
            print(f"LoRA适配器文件无效: {adapter_model_path}")
            return False
    else:
        print(f"目录存在但缺少LoRA适配器文件: {model_dir}")
        print(f"  - adapter_config.json: {adapter_config_exists}")
        print(f"  - adapter_model.safetensors: {adapter_model_exists}")
        return False

# 预测部分重构
MODELS = [
    ("0720-lora-v9", OUTPUT_DIR, "lora"),
]

def collate_fn(data):
    batch = {
        'input_ids': torch.stack([x['input_ids'] for x in data]),
        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
        'labels': torch.stack([x['labels'] for x in data])
    }
    return batch

def main():
    all_start = time.time()
    check_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA初始化完成，当前设备: {torch.cuda.current_device()}")
    print("加载数据...")
    
    # 验证数据文件路径
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"测试数据文件不存在: {TEST_PATH}")
    
    print(f"训练数据路径: {TRAIN_PATH}")
    print(f"测试数据路径: {TEST_PATH}")
    
    try:
        with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
        train = []
        for i, line in enumerate(train_lines):
            try:
                data = json.loads(line.strip())
                # 确保数据是字典格式
                if isinstance(data, dict):
                    train.append(data)
                else:
                    print(f"警告: 第{i+1}行数据不是字典格式，跳过")
            except json.JSONDecodeError as e:
                print(f"警告: 第{i+1}行JSON解析失败: {e}，跳过")
            except Exception as e:
                print(f"警告: 第{i+1}行数据处理失败: {e}，跳过")
        
        if not train:
            raise ValueError("没有成功加载任何训练数据")
        
        train_df = pd.DataFrame(train)
        print(f"成功加载训练数据: {len(train_df)} 条")
        
    except Exception as e:
        print(f"训练数据加载失败: {e}")
        raise
    
    try:
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            test_lines = f.readlines()
        test = []
        for i, line in enumerate(test_lines):
            try:
                data = json.loads(line.strip())
                # 确保数据是字典格式
                if isinstance(data, dict):
                    test.append(data)
                else:
                    print(f"警告: 第{i+1}行数据不是字典格式，跳过")
            except json.JSONDecodeError as e:
                print(f"警告: 第{i+1}行JSON解析失败: {e}，跳过")
            except Exception as e:
                print(f"警告: 第{i+1}行数据处理失败: {e}，跳过")
        
        if not test:
            raise ValueError("没有成功加载任何测试数据")
        
        test_df = pd.DataFrame(test)
        print(f"成功加载测试数据: {len(test_df)} 条")
        
    except Exception as e:
        print(f"测试数据加载失败: {e}")
        raise
    
    print(f"训练数据: {len(train_df)} 条")
    print(f"测试数据: {len(test_df)} 条")
    
    # 数据验证
    print("验证数据格式...")
    if 'text' not in train_df.columns:
        raise ValueError("训练数据缺少'text'列")
    if 'label' not in train_df.columns:
        raise ValueError("训练数据缺少'label'列")
    if 'text' not in test_df.columns:
        raise ValueError("测试数据缺少'text'列")
    
    # 确保label列是整数类型
    train_df['label'] = train_df['label'].astype(int)
    
    # 数据样本检查
    print("数据样本检查:")
    print(f"训练数据列: {list(train_df.columns)}")
    print(f"测试数据列: {list(test_df.columns)}")
    print(f"训练数据前3行:")
    for i in range(min(3, len(train_df))):
        print(f"  第{i+1}行: {dict(train_df.iloc[i])}")
    print(f"测试数据前3行:")
    for i in range(min(3, len(test_df))):
        print(f"  第{i+1}行: {dict(test_df.iloc[i])}")
    
    # 划分验证集（9:1）
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=42, shuffle=True)
    print(f"训练集: {len(train_df)} 条，验证集: {len(val_df)} 条")
    if is_model_saved(OUTPUT_DIR):
        print(f"检测到已保存的LoRA适配器，直接加载进行预测: {OUTPUT_DIR}")
        print("\n===== 多进程批量预测 =====")
        multiprocess_predict(
            models=MODELS,
            test_path=TEST_PATH,
            result_dir=MULTI_RESULT_DIR,
            batch_size=64,  # V100可以处理更大的batch
            base_model_path=BASE_MODEL_PATH
        )
        print("全部流程结束！")
        return
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("加载基础模型...")
    
    # V100优化：尝试使用BF16，如果不支持则使用FP16
    use_bf16 = False
    use_fp16 = False
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("使用BF16精度加载模型...")
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None)
            use_bf16 = True
        else:
            print("使用FP16精度加载模型...")
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None)
            use_fp16 = True
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用trust_remote_code=True...")
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
                use_bf16 = True
            else:
                base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None, trust_remote_code=True)
                use_fp16 = True
            print("✓ 模型加载成功")
        except Exception as e2:
            print(f"第二次尝试失败: {e2}")
            print("尝试使用默认精度...")
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=None, trust_remote_code=True)
            print("✓ 模型加载成功")
    
    # LoRA配置 - V100优化
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,  # 增加rank，V100有足够显存
        lora_alpha=64,  # 相应增加alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=False,
    )
    model = get_peft_model(base_model, peft_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("✓ LoRA模型加载成功")
    model.print_trainable_parameters()
    train_dataset = AIGCDetectionDataset(train_df, tokenizer)
    val_dataset = AIGCDetectionDataset(val_df, tokenizer)
    
    # V100优化的训练参数
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # V100可以处理更大的batch
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # 减少梯度累积，因为batch size增大了
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,  # 增加预热步数
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        no_cuda=False,
        label_names=["labels"],
        fp16=use_fp16,
        bf16=use_bf16,
        #tf32=True if torch.cuda.is_available() else False,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # V100显存充足，可以关闭
        dataloader_num_workers=4,  # 增加数据加载进程
        eval_accumulation_steps=16,  # 减少评估累积步数
        report_to=[],
        disable_tqdm=False,
        group_by_length=True,
        length_column_name="length",
        ignore_data_skip=False,
        dataloader_drop_last=True,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
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
    print("保存最终LoRA适配器...")
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
    print("\n===== 多进程批量预测 =====")
    multiprocess_predict(
        models=MODELS,
        test_path=TEST_PATH,
        result_dir=MULTI_RESULT_DIR,
        batch_size=64,  # V100可以处理更大的batch
        base_model_path=BASE_MODEL_PATH
    )
    print("全部流程结束！")
    all_end = time.time()
    print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main() 