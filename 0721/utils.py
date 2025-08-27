import os
import json
import pandas as pd
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import transformers
from multiprocessing import Process
import numpy as np
import multiprocessing

transformers.logging.set_verbosity_error()

def cleanup_memory():
    """清理CUDA显存和Python内存"""
    torch.cuda.empty_cache()
    gc.collect()

def safe_trainer_cleanup(trainer):
    """安全清理Trainer对象，释放内存"""
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
    """打印GPU信息"""
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        if torch.cuda.is_bf16_supported():
            print("✓ 支持BF16精度")
        else:
            print("✗ 不支持BF16精度，将使用FP16")
    else:
        print("GPU不可用，使用CPU")

def check_model_file_size(file_path):
    """检查模型文件大小，防止空文件"""
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    if file_size < 1024:
        print(f"警告: 文件 {file_path} 太小 ({file_size} bytes)，可能是空文件")
        return False
    return True

def is_model_saved(model_dir):
    """判断LoRA模型是否已保存且有效"""
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

def collate_fn(data):
    """自定义数据收集函数，组batch"""
    batch = {
        'input_ids': torch.stack([x['input_ids'] for x in data]),
        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
        'labels': torch.stack([x['labels'] for x in data])
    }
    return batch

def batch_predict(model, tokenizer, test_df, batch_size=64):
    model.eval()
    device = model.device
    predictions = []
    sample_idx = 0  # 增加样本计数器用于调试
    for i in tqdm(range(0, len(test_df), batch_size), desc="Batch预测进度"):
        batch = test_df.iloc[i:i+batch_size]
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"样本进度 {i+1}-{i+len(batch)}", leave=False):
            text = row['text']
            prompt = f"判断以下文本是AI生成的还是人类撰写的？文本：{text}"
            messages = [{"role": "user", "content": prompt}]
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_input, return_tensors="pt")
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
            input_length = inputs['input_ids'].shape[1]
            output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

            # 调试：打印前20条样本的输入和输出
            if sample_idx < 20:
                print(f"\n--- [调试信息] 样本 {sample_idx + 1} ---")
                # 替换换行符，防止日志格式错乱
                clean_text = text.replace('\n', ' ').replace('\r', '')
                clean_input = text_input.replace('\n', ' ').replace('\r', '')
                print(f"  - 原始文本: {clean_text[:200]}...")
                print(f"  - 输入提示词: {clean_input}")
                print(f"  - 模型生成内容: '{output_text}'")
                print("--- [调试信息结束] ---")

            if "AI生成" in output_text:
                predictions.append(1)
            elif "人类撰写" in output_text:
                predictions.append(0)
            else:
                if any(word in output_text.lower() for word in ["ai", "机器", "模型", "生成", "自动"]):
                    predictions.append(1)
                else:
                    predictions.append(0)
            del inputs, outputs
            torch.cuda.empty_cache()
            sample_idx += 1  # 更新计数器
    return predictions

def predict_one_model(model_name, model_path, mtype, test_path, result_file, batch_size, base_model_path):
    import torch, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import pandas as pd
    from tqdm import tqdm
    import transformers
    transformers.logging.set_verbosity_error()
    with open(test_path, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
    test_df = pd.DataFrame(test)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # V100优化：尝试使用BF16，如果不支持则使用FP16
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print(f"使用BF16精度加载模型: {model_name}")
            if mtype == "lora":
                base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
                model = PeftModel.from_pretrained(base_model, model_path)
                del base_model # 立即释放base_model内存
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            print(f"使用FP16精度加载模型: {model_name}")
            if mtype == "lora":
                base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
                model = PeftModel.from_pretrained(base_model, model_path)
                del base_model # 立即释放base_model内存
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    except Exception as e:
        print(f"模型加载失败，尝试使用trust_remote_code=True: {e}")
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                if mtype == "lora":
                    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
                    model = PeftModel.from_pretrained(base_model, model_path)
                    del base_model # 立即释放base_model内存
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            else:
                if mtype == "lora":
                    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, trust_remote_code=True)
                    model = PeftModel.from_pretrained(base_model, model_path)
                    del base_model # 立即释放base_model内存
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
        except Exception as e2:
            print(f"第二次尝试失败，使用默认精度: {e2}")
            if mtype == "lora":
                base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
                model = PeftModel.from_pretrained(base_model, model_path)
                del base_model # 立即释放base_model内存
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    model = model.to(device)
    preds = batch_predict(model, tokenizer, test_df, batch_size=batch_size)
    with open(result_file, "w") as f:
        for label in preds:
            f.write(str(label) + "\n")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"{model_name} 完成，AI生成={sum(preds)}, 人类撰写={len(preds)-sum(preds)}")
    return preds

def multiprocess_predict(models, test_path, result_dir, batch_size, base_model_path, num_workers):
    import time
    from multiprocessing import Process
    processes = []
    active_processes = []
    for idx, (name, model_path, mtype) in enumerate(models):
        result_file = os.path.join(result_dir, f"submit_{name}.txt")
        if os.path.exists(result_file):
            print(f"[跳过] {name} 已有预测结果: {result_file}")
            continue
        p = Process(target=predict_one_model, args=(name, model_path, mtype, test_path, result_file, batch_size, base_model_path))
        p.start()
        active_processes.append(p)
        # 控制最大并发进程数
        if len(active_processes) >= num_workers:
            for ap in active_processes:
                ap.join()
            active_processes = []
    # 等待剩余进程
    for ap in active_processes:
        ap.join()
    print("全部模型预测完成！")
    # 汇总结果
    results = {}
    for name, _, _ in models:
        result_file = os.path.join(result_dir, f"submit_{name}.txt")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                preds = [int(line.strip()) for line in f.readlines()]
            results[name] = preds
    print("\n=== 各模型预测分布 ===")
    for name, preds in results.items():
        print(f"{name}: AI生成={sum(preds)}, 人类撰写={len(preds)-sum(preds)}")
    if results:
        all_preds = np.array(list(results.values()))
        agree = np.all(all_preds == all_preds[0], axis=0)
        print(f"\n所有模型预测完全一致的样本数: {agree.sum()} / {len(all_preds[0])}")
    return results 

def clear_all_resources(trainer=None, checkpoint_dir=None, save_type=None):
    """
    统一资源清理函数：
    - 清理Trainer
    - 保存中断/错误检查点
    - 终止所有子进程
    - 清理显存
    save_type: None/"interrupted"/"error"
    """
    if trainer is not None and checkpoint_dir is not None and save_type is not None:
        try:
            trainer.save_model(os.path.join(checkpoint_dir, save_type))
            print(f"检查点已保存({save_type})，可以稍后继续训练")
        except Exception as e:
            print(f"保存检查点失败: {e}")
    if trainer is not None:
        safe_trainer_cleanup(trainer)
    try:
        multiprocessing.active_children()
        for p in multiprocessing.active_children():
            print(f"终止子进程: {p.pid}")
            p.terminate()
    except Exception as e:
        print(f"终止子进程时出错: {e}")
    cleanup_memory()


def save_lora_model(trainer, output_dir):
    """保存LoRA适配器"""
    try:
        trainer.save_model(output_dir)
        print(f"LoRA适配器已保存到: {output_dir}")
    except Exception as e:
        print(f"保存LoRA适配器失败: {e}")


def run_multiprocess_predict(models, test_path, result_dir, batch_size, base_model_path, num_workers=8):
    """
    多进程预测主控函数，带进度和异常处理
    """
    print("\n===== 多进程批量预测 =====")
    try:
        multiprocess_predict(
            models=models,
            test_path=test_path,
            result_dir=result_dir,
            batch_size=batch_size,
            base_model_path=base_model_path,
            num_workers=num_workers
        )
        print("全部流程结束！")
    except Exception as e:
        print(f"多进程预测异常: {e}")
        clear_all_resources() 

def get_torch_precision():
    """
    自动判断当前环境支持的精度，返回torch_dtype, use_bf16, use_fp16
    """
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("使用BF16精度")
        return torch.bfloat16, True, False
    else:
        print("使用FP16精度")
        return torch.float16, False, True 