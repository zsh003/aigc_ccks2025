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

transformers.logging.set_verbosity_error()

BASE_MODEL_PATH = "/mnt/e/Models/Qwen/Qwen3/Qwen3-0.6B"

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

def batch_predict(model, tokenizer, test_df, batch_size=32):
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
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05
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

def predict_one_model(model_name, model_path, mtype, test_path, result_file, batch_size=32, base_model_path=BASE_MODEL_PATH):
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
    if mtype == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
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

def multiprocess_predict(models, test_path, result_dir, batch_size=32, base_model_path=BASE_MODEL_PATH):
    processes = []
    for name, model_path, mtype in models:
        result_file = os.path.join(result_dir, f"submit_{name}.txt")
        if os.path.exists(result_file):
            print(f"[跳过] {name} 已有预测结果: {result_file}")
            continue
        p = Process(target=predict_one_model, args=(name, model_path, mtype, test_path, result_file, batch_size, base_model_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
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