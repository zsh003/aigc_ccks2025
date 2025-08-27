import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import gc
import time
from torch.utils.data import Dataset, DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils import batch_predict, predict_one_model, multiprocess_predict, \
    cleanup_memory, safe_trainer_cleanup, check_gpu, check_model_file_size, is_model_saved, collate_fn, \
    clear_all_resources, save_lora_model, run_multiprocess_predict, get_torch_precision

class Tee(object):
    def __init__(self, filename, mode="a", encoding="utf-8"):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.__stdout__
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

base_dir = "0720"
version = "v10"
ft_method = "lora"
model_para_scale = "0.6B"
os.makedirs(base_dir, exist_ok=True)
log_file = os.path.join(base_dir, "train_predict.log")
sys.stdout = Tee(log_file)
sys.stderr = sys.stdout
print(f"\n\n==== 日志开始 {datetime.now()} ====")

MODEL_NAME = f"../models/Qwen3-{model_para_scale}"
TRAIN_PATH = "../datasets/train/train.jsonl"
TEST_PATH = "../datasets/test_717/test.jsonl"
OUTPUT_DIR = f"{base_dir}/fine_tuned_model_{model_para_scale}_{ft_method}-fv_{version}"
RESULT_PATH = f"{base_dir}/submit_{model_para_scale}_{ft_method}-fv_{version}.txt"
CHECKPOINT_DIR = f"{base_dir}/checkpoints_{model_para_scale}_{ft_method}-fv_{version}"
MULTI_RESULT_DIR = f"{base_dir}/outputs"
MODELS = [
    (f"{base_dir}-{ft_method}-{version}", OUTPUT_DIR, ft_method),
]
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MULTI_RESULT_DIR, exist_ok=True)

HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".hf_cache", "hub")
os.environ["HF_HOME"] = os.path.dirname(HF_CACHE_DIR)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
DATASETS_CACHE = os.path.join(os.path.dirname(HF_CACHE_DIR), "datasets")
os.environ["DATASETS_CACHE"] = DATASETS_CACHE
os.makedirs(DATASETS_CACHE, exist_ok=True)

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
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=42, shuffle=True)
    print(f"训练集: {len(train_df)} 条，验证集: {len(val_df)} 条")
    if is_model_saved(OUTPUT_DIR):
        print(f"检测到已保存的LoRA适配器，直接加载进行预测: {OUTPUT_DIR}")
        print("\n===== 多进程批量预测 =====")
        # RTX5090 48G显卡，建议num_workers可设为8~16，根据实际情况调整
        multiprocess_predict(
            models=MODELS,
            test_path=TEST_PATH,
            result_dir=MULTI_RESULT_DIR,
            batch_size=64,
            base_model_path=BASE_MODEL_PATH,
            num_workers=8  # 可根据显卡资源调整，如16、32等
        )
        print("全部流程结束！")
        return
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("加载基础模型...")
    # 精度选择
    torch_dtype, use_bf16, use_fp16 = get_torch_precision()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map=None)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用trust_remote_code=True...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map=None, trust_remote_code=True)
            print("✓ 模型加载成功")
        except Exception as e2:
            print(f"第二次尝试失败: {e2}")
            print("尝试使用默认精度...")
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=None, trust_remote_code=True)
            print("✓ 模型加载成功")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
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
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,
        logging_steps=50,
        #save_strategy="steps",
        #save_steps=200,
        #save_total_limit=3,
        remove_unused_columns=False,
        no_cuda=False,
        label_names=["labels"],
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        eval_accumulation_steps=16,
        report_to=[],
        disable_tqdm=False,
        group_by_length=True,
        length_column_name="length",
        ignore_data_skip=False,
        dataloader_drop_last=True,
        # evaluation_strategy=None,
        # eval_steps=None,
        # load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        # 不传入eval_dataset和compute_metrics
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
        clear_all_resources(trainer, CHECKPOINT_DIR, "interrupted")
        return
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("正在清理资源...")
        clear_all_resources(trainer, CHECKPOINT_DIR, "error")
        return
    print("保存最终LoRA适配器...")
    save_lora_model(trainer, OUTPUT_DIR)
    print("清理训练资源...")
    clear_all_resources()
    run_multiprocess_predict(
        models=MODELS,
        test_path=TEST_PATH,
        result_dir=MULTI_RESULT_DIR,
        batch_size=64,
        base_model_path=MODEL_NAME,
        num_workers=8
    )
    print("全部流程结束！")
    all_end = time.time()
    print(f"\n全部流程总用时: {all_end - all_start:.2f} 秒")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main() 