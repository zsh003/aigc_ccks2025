import os
import json
import pandas as pd
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import logging
import transformers
from multiprocessing import Process
import numpy as np

# 屏蔽transformers无效参数警告
transformers.logging.set_verbosity_error()

# The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

# 基础模型路径
BASE_MODEL_PATH = "/mnt/e/Models/Qwen/Qwen3/Qwen3-0.6B"
# 新测试集路径
TEST_PATH = "../datasets/test_717/test.jsonl"
# 结果保存目录
RESULT_DIR = "0717/outputs"
os.makedirs(RESULT_DIR, exist_ok=True)

# 所有模型路径及类型（已根据各脚本实际保存路径整理）
MODELS = [
    # (模型名, 路径, 类型)
    ("0706-lora", "0706/fine_tuned_model", "lora"),
    ("0708-lora-v3", "0708/fine_tuned_model_0.6B_v3", "lora"),
    ("0708-lora-v5", "0708/fine_tuned_model_0.6B_v5", "lora"),
    ("0712-full", "0712/fine_tuned_model_0.6B_full-fv_v3", "full"),
    ("0713-full", "0713/fine_tuned_model_0.6B_full-fv_v4", "full"),
    ("0714-full", "0714/fine_tuned_model_0.6B_full-fv_v5", "full"),
]

# 加载测试集
def load_test_df():
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
    return pd.DataFrame(test)

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 推理函数（与训练脚本保持一致，支持LoRA和全量）
def predict(model, tokenizer, test_df, batch_size=32):
    model.eval()
    device = model.device
    predictions = []
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
    return predictions

# 单模型多进程预测函数
def predict_one_model(name, model_path, mtype, test_path, result_file, batch_size):
    import torch, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import pandas as pd
    from tqdm import tqdm
    import transformers
    transformers.logging.set_verbosity_error()
    # 重新加载测试集，避免多进程数据冲突
    with open(test_path, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f.readlines()]
    test_df = pd.DataFrame(test)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mtype == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device)
    preds = predict(model, tokenizer, test_df, batch_size=batch_size)
    with open(result_file, "w") as f:
        for label in preds:
            f.write(str(label) + "\n")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"{name} 完成，AI生成={sum(preds)}, 人类撰写={len(preds)-sum(preds)}")

# 主流程（多进程）
def main():
    test_df = load_test_df()
    batch_size = 32
    processes = []
    for name, model_path, mtype in MODELS:
        result_file = os.path.join(RESULT_DIR, f"submit_{name}.txt")
        if os.path.exists(result_file):
            print(f"[跳过] {name} 已有预测结果: {result_file}")
            continue
        p = Process(target=predict_one_model, args=(name, model_path, mtype, TEST_PATH, result_file, batch_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("全部模型预测完成！")
    # 对比分析
    results = {}
    for name, _, _ in MODELS:
        result_file = os.path.join(RESULT_DIR, f"submit_{name}.txt")
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

if __name__ == "__main__":
    main() 