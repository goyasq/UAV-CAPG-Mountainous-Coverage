#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

class UAVEnergyModel:
    """
    UAV能量消耗模型 - DJI Matrice 300 RTK
    
    基于以下文献的能量消耗模型：
    - Gong et al. (2022): 多旋翼UAV功率消耗建模和实验验证
    - 技术规格：DJI Matrice 300 RTK (6.3kg, TB60电池 274Wh, 55分钟飞行时间)
    - Liu et al. (2017): 多旋翼小型无人机系统功率消耗模型理论框架
    
    参数基于实际飞行测试数据和DJI Matrice 300 RTK技术规格标定。
    """
    
    def __init__(self, 
                 m=6.3,                # 无人机总质量 (kg) - DJI M300 RTK + 双TB60电池
                 g=9.8,                # 重力加速度 (m/s^2)
                 P_hover=300.0,        # 悬停功率基线 (W) - 基于Gong et al.实验数据和TB60规格分析
                 k_h=3.0,              # 水平速度系数 (W·s²/m²) - 基于多旋翼气动理论估算
                 P_asc=150.0,          # 上升功率系数 (W·s/m) - 基于Liu et al.理论框架调整
                 P_desc=80.0,          # 下降功率系数 (W·s/m) - 基于Liu et al.理论框架调整
                 v_xy=5.0,             # 水平飞行速度 (m/s) - 巡航最优速度
                 v_z_asc=3.0,          # 上升速度 (m/s)
                 v_z_desc=2.0,         # 下降速度 (m/s)
                 accel_time_xy=1.0,    # 水平加速时间 (s)
                 accel_time_z=0.5      # 垂直加速时间 (s)
                ):
        """
        初始化UAV能量模型参数
        
        参数说明：
        - P_hover: 300W基于Gong et al. (2022)对DJI M210的实验数据，
                   考虑M300 RTK更大重量调整得出
        - k_h: 水平速度系数基于多旋翼气动理论，参考相关文献估算
        - P_asc/P_desc: 基于Liu et al. (2017)的理论框架，
                        根据M300 RTK规格调整的垂直飞行功率系数
        """
        self.m = m
        self.g = g
        self.P_hover = P_hover
        self.k_h = k_h
        self.P_asc = P_asc
        self.P_desc = P_desc
        self.v_xy = v_xy
        self.v_z_asc = v_z_asc
        self.v_z_desc = v_z_desc
        self.accel_time_xy = accel_time_xy
        self.accel_time_z = accel_time_z
    
    def calculate_drag_v(self, velocity):
        """
        计算垂直空气阻力
        简化模型，假设阻力与速度平方成正比
        """
        drag_coefficient = 0.1  # 简化的阻力系数
        return drag_coefficient * velocity**2
    
    def calculate_P_d_xy(self, v_xy):
        """
        计算水平恒速飞行所需功率
        
        基于多旋翼功率模型：P = P_hover + k_h * v²
        
        参数:
            v_xy: 水平飞行速度 (m/s)
            
        返回:
            恒速水平飞行功率 (W)
        """
        return self.P_hover + self.k_h * v_xy**2
    
    def calculate_P_a_xy(self, v_xy):
        """
        计算水平加速/减速所需功率
        
        参数:
            v_xy: 水平飞行速度 (m/s)
            
        返回:
            水平加速功率 (W)
        """
        # 加速时功率增加约20%
        acceleration_factor = 1.2
        return acceleration_factor * self.calculate_P_d_xy(v_xy)
    
    def calculate_P_d_z_asc(self, v_z):
        """
        计算恒速上升所需功率
        
        基于Liu et al. (2017)理论框架：P = P_hover + P_asc * v_z
        
        参数:
            v_z: 上升速度 (m/s)
            
        返回:
            上升功率 (W)
        """
        return self.P_hover + self.P_asc * v_z
    
    def calculate_P_d_z_desc(self, v_z):
        """
        计算恒速下降所需功率
        
        基于Liu et al. (2017)理论框架：P = P_hover - P_desc * v_z
        下降时部分重力势能可以回收利用
        
        参数:
            v_z: 下降速度 (m/s) (应为正值)
            
        返回:
            下降功率 (W)
        """
        power = self.P_hover - self.P_desc * v_z
        # 确保功率不会低于基本维持功率（悬停功率的50%）
        return max(power, self.P_hover * 0.5)
    
    def calculate_P_a_z(self, v_z, is_ascending=True):
        """
        计算垂直加速/减速所需功率
        
        参数:
            v_z: 垂直速度 (m/s)
            is_ascending: 是否为上升状态
            
        返回:
            垂直加速功率 (W)
        """
        # 垂直加速时功率增加约15%
        acceleration_factor = 1.15
        if is_ascending:
            return acceleration_factor * self.calculate_P_d_z_asc(v_z)
        else:
            return acceleration_factor * self.calculate_P_d_z_desc(v_z)
    
    def calculate_K_xy(self, theta):
        """
        计算水平加速度的缩放因子K_xy
        
        参数:
            theta: 两段位移之间的夹角 (弧度)
            
        返回:
            缩放因子 K_xy
        """
        theta_abs = abs(theta)
        
        if theta_abs == 0:
            return 0.0
        elif 0 < theta_abs < math.pi/2:
            return 1.0 - math.sqrt(1.0 - (2.0 * theta_abs / math.pi)**2)
        else:  # theta_abs >= math.pi/2
            return 1.0
    
    def calculate_K_z(self, prev_z_velocity, curr_z_velocity):
        """
        计算垂直加速度的缩放因子K_z
        
        参数:
            prev_z_velocity: 前一段垂直速度 (m/s)
            curr_z_velocity: 当前段垂直速度 (m/s)
            
        返回:
            缩放因子 K_z
        """
        # 如果垂直速度状态发生变化，则K_z为1
        if (prev_z_velocity > 0 and curr_z_velocity <= 0) or \
           (prev_z_velocity < 0 and curr_z_velocity >= 0) or \
           (prev_z_velocity == 0 and curr_z_velocity != 0):
            return 1.0
        else:
            return 0.0
    
    def calculate_energy_between_points(self, point1, point2, prev_point=None):
        """
        计算两个3D点之间的能量消耗
        
        参数:
            point1: 起点坐标 [x, y, z]
            point2: 终点坐标 [x, y, z]
            prev_point: 前一点坐标 [x, y, z]，用于计算转向角和速度变化（可选）
            
        返回:
            总能量消耗 (焦耳)
        """
        # 计算水平能量
        E_d_xy, E_a_xy = self.calculate_horizontal_energy(point1, point2, prev_point)
        
        # 计算垂直能量
        E_d_z, E_a_z = self.calculate_vertical_energy(point1, point2, prev_point)
        
        # 返回总能量
        return E_d_xy + E_a_xy + E_d_z + E_a_z
    
    def calculate_horizontal_energy(self, point1, point2, prev_point=None):
        """
        计算两点间水平运动能量消耗
        
        参数:
            point1: 起点坐标 [x, y, z]
            point2: 终点坐标 [x, y, z]
            prev_point: 前一点坐标 [x, y, z]，用于计算转向角
            
        返回:
            水平位移能量 E_d_xy
            水平加速能量 E_a_xy
        """
        # 计算水平距离
        d_xy = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        # 计算水平位移能量
        P_d_xy = self.calculate_P_d_xy(self.v_xy)
        E_d_xy = P_d_xy * (d_xy / self.v_xy)
        
        # 计算转向角和水平加速能量
        E_a_xy = 0.0
        if prev_point is not None:
            # 计算向量v1和v2
            v1 = [point1[0] - prev_point[0], point1[1] - prev_point[1]]
            v2 = [point2[0] - point1[0], point2[1] - point1[1]]
            
            # 计算向量长度
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # 避免除以零
            if len_v1 > 0.001 and len_v2 > 0.001:
                # 计算向量点积
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                cos_theta = dot_product / (len_v1 * len_v2)
                # 防止数值误差导致cos_theta超出[-1,1]范围
                cos_theta = max(min(cos_theta, 1.0), -1.0)
                # 计算角度
                theta = math.acos(cos_theta)
                
                # 计算缩放因子
                K_xy = self.calculate_K_xy(theta)
                
                # 计算加速度能量
                P_a_xy = self.calculate_P_a_xy(self.v_xy)
                E_a_xy = K_xy * P_a_xy * self.accel_time_xy
        
        return E_d_xy, E_a_xy
    
    def calculate_vertical_energy(self, point1, point2, prev_point=None):
        """
        计算两点间垂直运动能量消耗
        
        参数:
            point1: 起点坐标 [x, y, z]
            point2: 终点坐标 [x, y, z]
            prev_point: 前一点坐标 [x, y, z]，用于计算垂直速度变化
            
        返回:
            垂直位移能量 E_d_z
            垂直加速能量 E_a_z
        """
        # 计算高度差
        delta_z = point2[2] - point1[2]
        
        # 初始化能量值
        E_d_z_asc = 0.0
        E_d_z_desc = 0.0
        E_a_z = 0.0
        
        # 计算当前段的垂直速度方向 (正值表示上升，负值表示下降，0表示水平)
        curr_z_velocity = 0
        if delta_z > 0.001:
            curr_z_velocity = self.v_z_asc
        elif delta_z < -0.001:
            curr_z_velocity = -self.v_z_desc
        
        # 计算垂直位移能量
        if delta_z > 0:  # 上升
            E_d_z_asc = self.calculate_P_d_z_asc(self.v_z_asc) * (delta_z / self.v_z_asc)
        elif delta_z < 0:  # 下降
            E_d_z_desc = self.calculate_P_d_z_desc(self.v_z_desc) * (abs(delta_z) / self.v_z_desc)
        
        # 计算垂直加速能量
        if prev_point is not None:
            # 计算前一段的垂直速度方向
            prev_delta_z = point1[2] - prev_point[2]
            prev_z_velocity = 0
            if prev_delta_z > 0.001:
                prev_z_velocity = self.v_z_asc
            elif prev_delta_z < -0.001:
                prev_z_velocity = -self.v_z_desc
            
            # 计算垂直加速度缩放因子
            K_z = self.calculate_K_z(prev_z_velocity, curr_z_velocity)
            
            # 根据当前垂直运动方向计算加速度能量
            if curr_z_velocity > 0:  # 上升
                P_a_z = self.calculate_P_a_z(self.v_z_asc, True)
                E_a_z = K_z * P_a_z * self.accel_time_z
            elif curr_z_velocity < 0:  # 下降
                P_a_z = self.calculate_P_a_z(self.v_z_desc, False)
                E_a_z = K_z * P_a_z * self.accel_time_z
        
        return E_d_z_asc + E_d_z_desc, E_a_z
    
    def calculate_path_energy(self, path):
        """
        计算整条路径的能量消耗
        
        参数:
            path: 路径点列表，每个点为 [x, y, z]
            
        返回:
            总能量消耗 E_total
            水平位移能量 E_d_xy
            水平加速能量 E_a_xy
            垂直位移能量 E_d_z
            垂直加速能量 E_a_z
        """
        # 初始化能量计数器
        E_d_xy_total = 0.0
        E_a_xy_total = 0.0
        E_d_z_total = 0.0
        E_a_z_total = 0.0
        
        # 遍历路径点
        for i in range(1, len(path)):
            prev_point = path[i-2] if i > 1 else None
            curr_point = path[i-1]
            next_point = path[i]
            
            # 计算水平能量
            E_d_xy, E_a_xy = self.calculate_horizontal_energy(curr_point, next_point, prev_point)
            E_d_xy_total += E_d_xy
            E_a_xy_total += E_a_xy
            
            # 计算垂直能量
            E_d_z, E_a_z = self.calculate_vertical_energy(curr_point, next_point, prev_point)
            E_d_z_total += E_d_z
            E_a_z_total += E_a_z
        
        # 计算总能量
        E_horizontal = E_d_xy_total + E_a_xy_total
        E_vertical = E_d_z_total + E_a_z_total
        E_total = E_horizontal + E_vertical
        
        return {
            'E_total': E_total,
            'E_horizontal': E_horizontal,
            'E_d_xy': E_d_xy_total,
            'E_a_xy': E_a_xy_total,
            'E_vertical': E_vertical,
            'E_d_z': E_d_z_total,
            'E_a_z': E_a_z_total
        }

# 测试代码
if __name__ == "__main__":
    # 创建能量模型实例
    model = UAVEnergyModel()
    
    # 创建一个测试路径
    test_path = np.array([
        [0, 0, 10],
        [10, 0, 10],
        [20, 10, 15],
        [30, 10, 5],
        [40, 0, 0]
    ])
    
    # 计算能量消耗
    energy = model.calculate_path_energy(test_path)
    
    print("能量消耗计算结果:")
    print(f"总能量: {energy['E_total']:.2f} J")
    print(f"水平能量: {energy['E_horizontal']:.2f} J (位移: {energy['E_d_xy']:.2f} J, 加速: {energy['E_a_xy']:.2f} J)")
    print(f"垂直能量: {energy['E_vertical']:.2f} J (位移: {energy['E_d_z']:.2f} J, 加速: {energy['E_a_z']:.2f} J)")
    
    # 测试两点间能量计算
    point1 = [0, 0, 10]
    point2 = [10, 10, 15]
    energy_between = model.calculate_energy_between_points(point1, point2)
    print(f"\n两点间能量消耗: {energy_between:.2f} J") 