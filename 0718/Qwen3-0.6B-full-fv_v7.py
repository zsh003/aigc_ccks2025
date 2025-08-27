import os
import json
import pandas as pd
import numpy as np
import GPUtil
import torch
import gc
import time
from torch.utils.data import Dataset, DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
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
OUTPUT_DIR = "0718/fine_tuned_model_0.6B_full-fv_v7"
RESULT_PATH = "0718/submit_0.6B_full-fv_v7.txt"
CHECKPOINT_DIR = "0718/checkpoints_0.6B_full-fv_v7"
MULTI_RESULT_DIR = "0718/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MULTI_RESULT_DIR, exist_ok=True)

# ====== HF缓存配置 ======
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
# os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR  # 移除以消除FutureWarning
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)

# 内存清理函数
def cleanup_memory():
    print("执行内存清理...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("内存清理完成")

def log_gpu_usage():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id}: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB")

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
    else:
        print("GPU不可用，使用CPU")

class MemoryCleanCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            log_gpu_usage()
        if state.global_step % 100 == 0:
            cleanup_memory()

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
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
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
    # 支持safetensors格式
    model_file = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        if file_size > 1024:
            print(f"检测到已保存的全量模型(safetensors): {model_dir}")
            return True
        else:
            print(f"警告: model.safetensors文件过小: {file_size} bytes")
            return False
    # 兼容pytorch_model.bin
    model_file_bin = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.exists(model_file_bin):
        file_size = os.path.getsize(model_file_bin)
        if file_size > 1024:
            print(f"检测到已保存的全量模型(bin): {model_dir}")
            return True
        else:
            print(f"警告: pytorch_model.bin文件过小: {file_size} bytes")
            return False
    print(f"未检测到全量模型: {model_dir}")
    return False

# 预测部分重构
MODELS = [
    ("0718-full-v7", OUTPUT_DIR, "full"),
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
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train = [json.loads(line) for line in f.readlines()]
        train_df = pd.DataFrame(train)
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
        test_df = pd.DataFrame(test)
    print(f"训练数据: {len(train_df)} 条")
    print(f"测试数据: {len(test_df)} 条")
    # 划分验证集（9:1）
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)
    val_df = val_df.iloc[:50]  # 只用前100条验证，防止评估OOM
    print(f"训练集: {len(train_df)} 条，验证集: {len(val_df)} 条 (仅前100条)")
    if is_model_saved(OUTPUT_DIR):
        print(f"检测到已保存的全量模型，直接加载进行预测: {OUTPUT_DIR}")
        # 预测部分
        print("\n===== 多进程批量预测 =====")
        multiprocess_predict(
            models=MODELS,
            test_path=TEST_PATH,
            result_dir=MULTI_RESULT_DIR,
            batch_size=32,
            base_model_path=BASE_MODEL_PATH
        )
        print("全部流程结束！")
        return
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("加载基础模型进行全参数微调...")
    use_bf16 = False
    use_fp16 = False
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("使用BF16精度加载模型...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None)
            use_bf16 = True
        else:
            print("使用FP16精度加载模型...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None)
            use_fp16 = True
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用trust_remote_code=True...")
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
                use_bf16 = True
            else:
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None, trust_remote_code=True)
                use_fp16 = True
            print("✓ 模型加载成功")
        except Exception as e2:
            print(f"第二次尝试失败: {e2}")
            print("尝试使用默认精度...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=None, trust_remote_code=True)
            print("✓ 模型加载成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("✓ 全量模型加载成功")
    model.config.use_cache = False  # 评估阶段禁用缓存
    torch.compile(model) # 编译优化
    train_dataset = AIGCDetectionDataset(train_df, tokenizer)
    val_dataset = AIGCDetectionDataset(val_df, tokenizer)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=True if torch.cuda.is_available() else False,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # 开启降低内存占用，但是训练速度降低
        dataloader_num_workers=0,
        #dataloader_prefetch_factor=2,
        eval_accumulation_steps=256,    # 你可以增大（如eval_accumulation_steps=16或更大），让验证集分批送入，减少单次显存/内存占用。 2->32->256->2800
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
        callbacks=[MemoryCleanCallback()],
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
    print("\n===== 多进程批量预测 =====")
    multiprocess_predict(
        models=MODELS,
        test_path=TEST_PATH,
        result_dir=MULTI_RESULT_DIR,
        batch_size=32,
        base_model_path=BASE_MODEL_PATH
    )
    print("全部流程结束！")
    all_end = time.time()
    print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main() 