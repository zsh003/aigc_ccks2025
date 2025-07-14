import os
import gc
import torch
import psutil
import subprocess
import time

def cleanup_memory():
    """强制清理内存和显存"""
    print("开始清理内存和显存...")
    
    # 1. 清理PyTorch缓存
    if torch.cuda.is_available():
        print("清理CUDA缓存...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 显示清理前后的显存使用情况
        print(f"清理前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"清理前显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 2. 强制垃圾回收
    print("执行垃圾回收...")
    gc.collect()
    
    # 3. 清理Python内存
    print("清理Python内存...")
    import sys
    sys.modules.clear()
    
    # 4. 显示内存使用情况
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前内存使用: {memory_info.rss / 1024**3:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"清理后显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"清理后显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def kill_python_processes():
    """终止所有Python进程（谨慎使用）"""
    print("警告：这将终止所有Python进程！")
    response = input("是否继续？(y/N): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    try:
        # 在Windows上终止Python进程
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        print("当前Python进程:")
        print(result.stdout)
        
        # 终止Python进程
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'])
        print("已终止所有Python进程")
    except Exception as e:
        print(f"终止进程时出错: {e}")

def check_memory_usage():
    """检查当前内存使用情况"""
    print("\n=== 内存使用情况 ===")
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"系统总内存: {memory.total / 1024**3:.2f} GB")
    print(f"系统已用内存: {memory.used / 1024**3:.2f} GB")
    print(f"系统可用内存: {memory.available / 1024**3:.2f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    
    # 当前进程内存
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前进程内存使用: {memory_info.rss / 1024**3:.2f} GB")
    
    # GPU显存
    if torch.cuda.is_available():
        print(f"\n=== GPU显存使用情况 ===")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"显存已用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"显存可用: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")

if __name__ == "__main__":
    print("内存清理工具")
    print("1. 检查内存使用情况")
    print("2. 清理内存和显存")
    print("3. 终止所有Python进程（危险操作）")
    print("4. 退出")
    
    while True:
        choice = input("\n请选择操作 (1-4): ")
        
        if choice == '1':
            check_memory_usage()
        elif choice == '2':
            cleanup_memory()
            check_memory_usage()
        elif choice == '3':
            kill_python_processes()
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入") 