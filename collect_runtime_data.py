#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np

def collect_runtime_data():
    """收集所有任务区域的运行时间数据"""
    runtime_data = {}
    
    # 方法和对应的结果目录
    method_dirs = {
        'CAPG-TCO': ['capg_tco_paths_diverse', 'capg_tco_beautiful'],
        'UGEOC': ['ugeoc_paths_diverse'],
        'Adaptive Boustrophedon': ['smart_lawnmower_paths_diverse']
    }
    
    for method, dirs in method_dirs.items():
        runtime_data[method] = {}
        
        for dir_name in dirs:
            if not os.path.exists(dir_name):
                continue
                
            # 遍历该方法下的所有任务区域
            task_dirs = glob.glob(os.path.join(dir_name, 'area_*'))
            for task_dir in task_dirs:
                task_id = os.path.basename(task_dir)
                
                # 尝试从不同文件中读取运行时间
                runtime = 0.0
                
                # 1. 检查runtime_info.npy
                runtime_file = os.path.join(task_dir, 'runtime_info.npy')
                if os.path.exists(runtime_file):
                    try:
                        runtime_info = np.load(runtime_file, allow_pickle=True).item()
                        if isinstance(runtime_info, dict):
                            runtime = runtime_info.get('total_runtime', 0.0)
                    except Exception as e:
                        print(f"Error reading {runtime_file}: {e}")
                
                # 2. 检查timing.json
                if runtime == 0.0:
                    timing_file = os.path.join(task_dir, 'timing.json')
                    if os.path.exists(timing_file):
                        try:
                            with open(timing_file, 'r') as f:
                                timing_data = json.load(f)
                                runtime = timing_data.get('total_runtime', 0.0)
                        except:
                            pass
                
                # 3. 检查performance_metrics.json
                if runtime == 0.0:
                    metrics_file = os.path.join(task_dir, 'performance_metrics.json')
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                                runtime = metrics_data.get('runtime', 0.0)
                        except:
                            pass
                
                # 4. 检查log.txt
                if runtime == 0.0:
                    log_file = os.path.join(task_dir, 'log.txt')
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                content = f.read()
                                # 查找运行时间记录
                                import re
                                time_matches = re.findall(r'Total runtime: (\d+\.\d+)', content)
                                if time_matches:
                                    runtime = float(time_matches[-1])
                                else:
                                    # 尝试其他格式
                                    time_matches = re.findall(r'Processing time: (\d+\.\d+)', content)
                                    if time_matches:
                                        runtime = float(time_matches[-1])
                        except:
                            pass
                
                # 5. 检查results.txt
                if runtime == 0.0:
                    results_file = os.path.join(task_dir, 'results.txt')
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                                time_matches = re.findall(r'Runtime: (\d+\.\d+)', content)
                                if time_matches:
                                    runtime = float(time_matches[-1])
                        except:
                            pass
                
                if runtime > 0.0:  # 只记录有效的运行时间
                    runtime_data[method][task_id] = runtime
                else:
                    print(f"Warning: No runtime found for {method} - {task_id}")
    
    return runtime_data

def save_runtime_data(runtime_data, output_file):
    """保存运行时间数据到JSON文件"""
    with open(output_file, 'w') as f:
        json.dump(runtime_data, f, indent=2)

def print_runtime_summary(runtime_data):
    """打印运行时间摘要"""
    print("\nRuntime Summary:")
    print("=" * 60)
    
    # 按地形复杂度分类
    complexity_categories = {
        'Minimal': [],
        'Low': [],
        'Medium': [],
        'High': []
    }
    
    for method in runtime_data:
        print(f"\n{method}:")
        times = runtime_data[method]
        if times:
            # 按复杂度分类
            for task_id, runtime in times.items():
                if 'minimal' in task_id.lower():
                    category = 'Minimal'
                elif 'low' in task_id.lower():
                    category = 'Low'
                elif 'medium' in task_id.lower():
                    category = 'Medium'
                elif 'high' in task_id.lower():
                    category = 'High'
                else:
                    continue
                
                complexity_categories[category].append({
                    'method': method,
                    'runtime': runtime,
                    'task_id': task_id
                })
            
            # 打印总体统计
            all_times = list(times.values())
            avg_time = np.mean(all_times)
            std_time = np.std(all_times)
            min_time = np.min(all_times)
            max_time = np.max(all_times)
            print(f"  Overall Statistics:")
            print(f"    Average: {avg_time:.1f}s")
            print(f"    Std Dev: {std_time:.1f}s")
            print(f"    Range: {min_time:.1f}s - {max_time:.1f}s")
            print(f"    Number of tasks: {len(all_times)}")
    
    # 打印每个复杂度类别的统计
    print("\nStatistics by Terrain Complexity:")
    print("=" * 60)
    for category in complexity_categories:
        if complexity_categories[category]:
            print(f"\n{category} Complexity:")
            for method in method_dirs.keys():
                method_times = [item['runtime'] for item in complexity_categories[category] 
                              if item['method'] == method]
                if method_times:
                    avg_time = np.mean(method_times)
                    std_time = np.std(method_times)
                    print(f"  {method}:")
                    print(f"    Average: {avg_time:.1f}s")
                    print(f"    Std Dev: {std_time:.1f}s")
                    print(f"    Tasks: {len(method_times)}")

def main():
    output_file = "runtime_data.json"
    
    print("Collecting runtime data...")
    runtime_data = collect_runtime_data()
    
    print("Saving runtime data...")
    save_runtime_data(runtime_data, output_file)
    
    print_runtime_summary(runtime_data)

if __name__ == "__main__":
    main() 