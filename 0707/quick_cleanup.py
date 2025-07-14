#!/usr/bin/env python3
"""
快速内存清理脚本
用于立即释放显存和内存占用
"""

import os
import gc
import torch
import psutil
import subprocess
import signal
import sys

def force_cleanup():
    """强制清理内存和显存"""
    print("=== 开始强制清理内存和显存 ===")
    
    # 1. 清理PyTorch缓存
    if torch.cuda.is_available():
        print("清理CUDA缓存...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 显示清理前后的显存使用情况
        before_allocated = torch.cuda.memory_allocated() / 1024**3
        before_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"清理前显存使用: {before_allocated:.2f} GB")
        print(f"清理前显存缓存: {before_reserved:.2f} GB")
    
    # 2. 强制垃圾回收
    print("执行垃圾回收...")
    collected = gc.collect()
    print(f"回收了 {collected} 个对象")
    
    # 3. 清理Python内存
    print("清理Python内存...")
    import sys
    # 清理模块缓存
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('torch') or module_name.startswith('transformers'):
            del sys.modules[module_name]
    
    # 4. 再次清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        after_allocated = torch.cuda.memory_allocated() / 1024**3
        after_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"清理后显存使用: {after_allocated:.2f} GB")
        print(f"清理后显存缓存: {after_reserved:.2f} GB")
        
        if before_allocated > 0:
            freed_memory = before_allocated - after_allocated
            print(f"释放显存: {freed_memory:.2f} GB")
    
    # 5. 显示内存使用情况
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前进程内存使用: {memory_info.rss / 1024**3:.2f} GB")
    
    print("=== 清理完成 ===")

def check_memory_status():
    """检查内存状态"""
    print("\n=== 当前内存状态 ===")
    
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
        print(f"\n=== GPU显存状态 ===")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"显存已用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"显存可用: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")

def kill_python_workers():
    """终止Python工作进程"""
    print("查找并终止Python工作进程...")
    
    try:
        # 查找Python进程
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            python_processes = []
            
            for line in lines[3:]:  # 跳过标题行
                if line.strip() and 'python.exe' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        python_processes.append(pid)
            
            if python_processes:
                print(f"找到 {len(python_processes)} 个Python进程")
                for pid in python_processes:
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True, shell=True)
                        print(f"已终止进程 PID: {pid}")
                    except Exception as e:
                        print(f"终止进程 {pid} 时出错: {e}")
            else:
                print("未找到Python进程")
        else:
            print("无法获取进程列表")
            
    except Exception as e:
        print(f"终止进程时出错: {e}")

def main():
    print("快速内存清理工具")
    print("1. 检查内存状态")
    print("2. 强制清理内存")
    print("3. 终止Python工作进程")
    print("4. 执行完整清理")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请选择操作 (1-5): ").strip()
            
            if choice == '1':
                check_memory_status()
            elif choice == '2':
                force_cleanup()
                check_memory_status()
            elif choice == '3':
                kill_python_workers()
            elif choice == '4':
                print("执行完整清理...")
                kill_python_workers()
                force_cleanup()
                check_memory_status()
            elif choice == '5':
                print("退出程序")
                break
            else:
                print("无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"操作出错: {e}")

if __name__ == "__main__":
    main() 