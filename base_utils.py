import os
import cudf
import cupy as cp
import joblib
import json

def check_gpu():
    """检查GPU设备并打印信息"""
    print("="*50)
    print("GPU设备检测")
    print("="*50)
    try:
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA可用: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"CUDA设备数量: {cp.cuda.runtime.getDeviceCount()}")
            print(f"当前CUDA设备: {cp.cuda.runtime.getDevice()}")
            print(f"设备名称: {cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name']}")
            print(f"GPU内存: {cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['totalGlobalMem'] / 1024**3:.2f} GB")
        else:
            print("CUDA不可用，将使用CPU")
    except Exception as e:
        print(f"GPU检测出错: {e}")
    try:
        import cuml
        print(f"cuML版本: {cuml.__version__}")
    except:
        print("cuML版本信息获取失败")
    try:
        import cudf
        print(f"cuDF版本: {cudf.__version__}")
    except:
        print("cuDF版本信息获取失败")
    print("="*50)

def load_data():
    train_path = '../datasets/train/train.jsonl'
    test_path = '../datasets/test_521/test.jsonl'
    # cuDF加速
    with open(train_path, 'r', encoding='utf-8') as f:
        train_list = [json.loads(line) for line in f.readlines()]
    with open(test_path, 'r', encoding='utf-8') as f:
        test_list = [json.loads(line) for line in f.readlines()]
    train_df = cudf.DataFrame(train_list)
    test_df = cudf.DataFrame(test_list)
    train_df['is_test'] = 0
    test_df['is_test'] = 1
    df_all = cudf.concat([train_df, test_df], axis=0, ignore_index=True)
    return df_all

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"模型已保存到: {path}")

def load_model(path):
    model = joblib.load(path)
    print(f"模型已加载: {path}")
    return model 