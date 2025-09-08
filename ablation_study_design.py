 #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验设计：CAPG-TCO方法的组件分析
目标：验证每个组件对整体性能的贡献
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import json

class AblationStudyDesign:
    """
    消融实验设计类
    
    实验变量：
    1. 高度量化策略 (Altitude Quantization)
    2. ATSP优化 vs 贪心连接
    3. 等高线选择策略
    4. 能量模型复杂度
    """
    
    def __init__(self):
        self.test_areas = ['area_minimal_01', 'area_low_02', 'area_medium_03', 'area_high_04']
        self.ablation_configs = {
            'full_method': {
                'altitude_quantization': True,
                'atsp_optimization': True,
                'smart_contour_selection': True,
                'full_energy_model': True,
                'description': 'Complete CAPG-TCO method'
            },
            'no_quantization': {
                'altitude_quantization': False,
                'atsp_optimization': True,
                'smart_contour_selection': True,
                'full_energy_model': True,
                'description': 'Without altitude quantization'
            },
            'greedy_connection': {
                'altitude_quantization': True,
                'atsp_optimization': False,
                'smart_contour_selection': True,
                'full_energy_model': True,
                'description': 'Greedy primitive connection instead of ATSP'
            },
            'simple_contour': {
                'altitude_quantization': True,
                'atsp_optimization': True,
                'smart_contour_selection': False,
                'full_energy_model': True,
                'description': 'Simple contour selection strategy'
            },
            'simple_energy': {
                'altitude_quantization': True,
                'atsp_optimization': True,
                'smart_contour_selection': True,
                'full_energy_model': False,
                'description': 'Simplified energy model (no acceleration terms)'
            }
        }
        
    def design_experiment_matrix(self) -> Dict:
        """设计完整的实验矩阵"""
        experiment_matrix = {}
        
        for area in self.test_areas:
            experiment_matrix[area] = {}
            for config_name, config in self.ablation_configs.items():
                experiment_matrix[area][config_name] = {
                    'parameters': config,
                    'metrics_to_collect': [
                        'total_energy_consumption',
                        'vertical_energy_component',
                        'horizontal_energy_component',
                        'coverage_percentage',
                        'path_length',
                        'number_of_primitives',
                        'computation_time',
                        'agl_standard_deviation'
                    ]
                }
        
        return experiment_matrix
    
    def generate_ablation_scripts(self):
        """生成消融实验脚本"""
        
        # 1. 高度量化开关实验
        self.create_quantization_ablation_script()
        
        # 2. ATSP vs 贪心连接实验
        self.create_connection_strategy_script()
        
        # 3. 等高线选择策略实验
        self.create_contour_selection_script()
        
        # 4. 能量模型复杂度实验
        self.create_energy_model_script()
        
    def create_quantization_ablation_script(self):
        """创建高度量化消融实验脚本"""
        script_content = '''#!/usr/bin/env python
"""
高度量化策略消融实验
比较开启/关闭高度量化对性能的影响
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from contour_path_generator_diverse import ContourPathGenerator
import numpy as np

def run_quantization_ablation():
    """运行高度量化消融实验"""
    
    test_areas = ['area_minimal_01', 'area_low_02', 'area_medium_03', 'area_high_04']
    results = {}
    
    for area in test_areas:
        print(f"Testing area: {area}")
        results[area] = {}
        
        # Load terrain data
        dem_file = f'task_areas_diverse/{area}_dem.npy'
        x_file = f'task_areas_diverse/{area}_x_meters.npy'
        y_file = f'task_areas_diverse/{area}_y_meters.npy'
        
        if not all(os.path.exists(f) for f in [dem_file, x_file, y_file]):
            print(f"Skipping {area}: missing data files")
            continue
            
        dem_data = np.load(dem_file)
        x_meters = np.load(x_file)
        y_meters = np.load(y_file)
        
        # Test with quantization (default)
        generator_with_quant = ContourPathGenerator(
            dem_data, x_meters, y_meters,
            target_agl=50.0,
            altitude_quantization=True,  # 开启量化
            quantization_tolerance=5.0
        )
        
        # Test without quantization
        generator_no_quant = ContourPathGenerator(
            dem_data, x_meters, y_meters,
            target_agl=50.0,
            altitude_quantization=False,  # 关闭量化
            quantization_tolerance=0.1   # 很小的容差，接近连续跟踪
        )
        
        # Generate primitives for both configurations
        primitives_with_quant = generator_with_quant.generate_all_primitives()
        primitives_no_quant = generator_no_quant.generate_all_primitives()
        
        # Collect metrics
        results[area]['with_quantization'] = {
            'num_primitives': len(primitives_with_quant),
            'total_length': sum(p['length'] for p in primitives_with_quant),
            'avg_agl_std': np.mean([p['agl_std'] for p in primitives_with_quant]),
            'generation_time': generator_with_quant.total_generation_time
        }
        
        results[area]['without_quantization'] = {
            'num_primitives': len(primitives_no_quant),
            'total_length': sum(p['length'] for p in primitives_no_quant),
            'avg_agl_std': np.mean([p['agl_std'] for p in primitives_no_quant]),
            'generation_time': generator_no_quant.total_generation_time
        }
    
    # Save results
    with open('ablation_quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Quantization ablation study completed!")
    return results

if __name__ == "__main__":
    run_quantization_ablation()
'''
        
        with open('ablation_quantization_study.py', 'w') as f:
            f.write(script_content)
    
    def create_connection_strategy_script(self):
        """创建连接策略消融实验脚本"""
        script_content = '''#!/usr/bin/env python
"""
连接策略消融实验：ATSP vs 贪心连接
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from primitive_tsp_planner_diverse import PrimitiveTSPPlanner
import numpy as np
import json

def greedy_connection(primitives):
    """贪心连接算法实现"""
    if not primitives:
        return [], 0
    
    unvisited = list(range(len(primitives)))
    path = [0]  # Start from first primitive
    unvisited.remove(0)
    total_cost = 0
    
    current = 0
    while unvisited:
        # Find nearest unvisited primitive
        min_cost = float('inf')
        next_primitive = None
        
        for candidate in unvisited:
            # Simple Euclidean distance cost
            cost = np.linalg.norm(
                np.array(primitives[current]['end_point']) - 
                np.array(primitives[candidate]['start_point'])
            )
            if cost < min_cost:
                min_cost = cost
                next_primitive = candidate
        
        path.append(next_primitive)
        unvisited.remove(next_primitive)
        total_cost += min_cost
        current = next_primitive
    
    return path, total_cost

def run_connection_strategy_ablation():
    """运行连接策略消融实验"""
    test_areas = ['area_minimal_01', 'area_low_02', 'area_medium_03', 'area_high_04']
    results = {}
    
    for area in test_areas:
        print(f"Testing area: {area}")
        results[area] = {}
        
        # Load primitives
        primitives_file = f'path_primitives_diverse/{area}_primitives.npy'
        if not os.path.exists(primitives_file):
            print(f"Skipping {area}: no primitives file")
            continue
            
        primitives_data = np.load(primitives_file, allow_pickle=True)
        primitives = primitives_data.tolist()
        
        # ATSP optimization
        planner_atsp = PrimitiveTSPPlanner(use_atsp=True)
        atsp_path, atsp_cost, atsp_time = planner_atsp.solve_tsp(primitives)
        
        # Greedy connection
        import time
        start_time = time.time()
        greedy_path, greedy_cost = greedy_connection(primitives)
        greedy_time = time.time() - start_time
        
        results[area] = {
            'atsp': {
                'path_length': len(atsp_path),
                'total_cost': atsp_cost,
                'computation_time': atsp_time
            },
            'greedy': {
                'path_length': len(greedy_path),
                'total_cost': greedy_cost,
                'computation_time': greedy_time
            },
            'improvement_ratio': greedy_cost / atsp_cost if atsp_cost > 0 else 1.0
        }
    
    # Save results
    with open('ablation_connection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Connection strategy ablation study completed!")
    return results

if __name__ == "__main__":
    run_connection_strategy_ablation()
'''
        
        with open('ablation_connection_study.py', 'w') as f:
            f.write(script_content)
    
    def create_analysis_script(self):
        """创建结果分析脚本"""
        analysis_script = '''#!/usr/bin/env python
"""
消融实验结果分析脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_ablation_results():
    """分析消融实验结果"""
    
    # Load all results
    results = {}
    
    try:
        with open('ablation_quantization_results.json', 'r') as f:
            results['quantization'] = json.load(f)
    except FileNotFoundError:
        print("Quantization results not found")
    
    try:
        with open('ablation_connection_results.json', 'r') as f:
            results['connection'] = json.load(f)
    except FileNotFoundError:
        print("Connection results not found")
    
    # Generate analysis plots
    if 'quantization' in results:
        plot_quantization_analysis(results['quantization'])
    
    if 'connection' in results:
        plot_connection_analysis(results['connection'])

def plot_quantization_analysis(data):
    """绘制高度量化分析图"""
    areas = list(data.keys())
    
    # Extract metrics
    agl_std_with = [data[area]['with_quantization']['avg_agl_std'] for area in areas]
    agl_std_without = [data[area]['without_quantization']['avg_agl_std'] for area in areas]
    
    x = np.arange(len(areas))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, agl_std_with, width, label='With Quantization', alpha=0.8)
    ax.bar(x + width/2, agl_std_without, width, label='Without Quantization', alpha=0.8)
    
    ax.set_xlabel('Test Areas')
    ax.set_ylabel('Average AGL Standard Deviation (m)')
    ax.set_title('Impact of Altitude Quantization on Flight Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_quantization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_connection_analysis(data):
    """绘制连接策略分析图"""
    areas = list(data.keys())
    improvement_ratios = [data[area]['improvement_ratio'] for area in areas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(areas, improvement_ratios, alpha=0.8, color='skyblue')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, improvement_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}x', ha='center', va='bottom')
    
    ax.set_xlabel('Test Areas')
    ax.set_ylabel('Cost Ratio (Greedy / ATSP)')
    ax.set_title('ATSP Optimization vs Greedy Connection')
    ax.set_xticklabels(areas, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_connection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_ablation_results()
'''
        
        with open('analyze_ablation_results.py', 'w') as f:
            f.write(analysis_script)

def main():
    """主函数：创建完整的消融实验框架"""
    study = AblationStudyDesign()
    
    # 设计实验矩阵
    experiment_matrix = study.design_experiment_matrix()
    
    # 保存实验设计
    with open('ablation_experiment_design.json', 'w') as f:
        json.dump(experiment_matrix, f, indent=2)
    
    # 生成实验脚本
    study.generate_ablation_scripts()
    study.create_analysis_script()
    
    print("消融实验设计完成!")
    print("生成的文件:")
    print("- ablation_experiment_design.json (实验设计矩阵)")
    print("- ablation_quantization_study.py (高度量化实验)")
    print("- ablation_connection_study.py (连接策略实验)")
    print("- analyze_ablation_results.py (结果分析)")
    
    print("\n实验运行步骤:")
    print("1. 运行 python ablation_quantization_study.py")
    print("2. 运行 python ablation_connection_study.py")
    print("3. 运行 python analyze_ablation_results.py")

if __name__ == "__main__":
    main()