"""
规划模块评估指标
包括轨迹平滑度、安全性、舒适性等指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    heading: float
    velocity: float
    acceleration: float
    curvature: float
    timestamp: float


@dataclass
class Trajectory:
    """轨迹"""
    points: List[TrajectoryPoint]
    
    def get_positions(self) -> np.ndarray:
        """获取位置数组"""
        return np.array([[p.x, p.y] for p in self.points])
    
    def get_velocities(self) -> np.ndarray:
        """获取速度数组"""
        return np.array([p.velocity for p in self.points])
    
    def get_accelerations(self) -> np.ndarray:
        """获取加速度数组"""
        return np.array([p.acceleration for p in self.points])
    
    def get_curvatures(self) -> np.ndarray:
        """获取曲率数组"""
        return np.array([p.curvature for p in self.points])
    
    def get_duration(self) -> float:
        """获取轨迹持续时间"""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp
    
    def get_length(self) -> float:
        """获取轨迹长度"""
        positions = self.get_positions()
        if len(positions) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(positions)):
            length += np.linalg.norm(positions[i] - positions[i-1])
        
        return length


@dataclass
class TrajectoryMetrics:
    """轨迹评估指标"""
    smoothness: float  # 平滑度
    safety_score: float  # 安全性评分
    comfort_score: float  # 舒适性评分
    efficiency: float  # 效率
    curvature_consistency: float  # 曲率一致性
    velocity_consistency: float  # 速度一致性
    total_length: float  # 总长度
    total_duration: float  # 总时长
    max_acceleration: float  # 最大加速度
    max_jerk: float  # 最大加加速度
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'smoothness': self.smoothness,
            'safety_score': self.safety_score,
            'comfort_score': self.comfort_score,
            'efficiency': self.efficiency,
            'curvature_consistency': self.curvature_consistency,
            'velocity_consistency': self.velocity_consistency,
            'total_length': self.total_length,
            'total_duration': self.total_duration,
            'max_acceleration': self.max_acceleration,
            'max_jerk': self.max_jerk,
        }


@dataclass
class Obstacle:
    """障碍物"""
    x: float
    y: float
    radius: float
    velocity_x: float
    velocity_y: float


class PlanningMetrics:
    """规划模块评估指标计算器"""
    
    def __init__(self):
        """初始化规划指标计算器"""
        pass
    
    def calculate_smoothness(
        self,
        trajectory: Trajectory,
        method: str = "curvature"
    ) -> float:
        """
        计算轨迹平滑度
        
        Args:
            trajectory: 轨迹
            method: 计算方法 ("curvature" 或 "jerk")
            
        Returns:
            平滑度评分（0-1，越高越平滑）
        """
        if len(trajectory.points) < 3:
            return 1.0
        
        if method == "curvature":
            # 基于曲率变化计算平滑度
            curvatures = trajectory.get_curvatures()
            curvature_changes = np.diff(curvatures)
            curvature_variance = np.var(curvature_changes)
            
            # 归一化到0-1
            smoothness = np.exp(-curvature_variance * 10)
            
        elif method == "jerk":
            # 基于加加速度计算平滑度
            accelerations = trajectory.get_accelerations()
            if len(accelerations) < 2:
                return 1.0
            
            jerks = np.diff(accelerations)
            jerk_variance = np.var(jerks)
            
            # 归一化到0-1
            smoothness = np.exp(-jerk_variance * 0.1)
        
        else:
            raise ValueError(f"Unknown smoothness method: {method}")
        
        return float(np.clip(smoothness, 0, 1))
    
    def calculate_safety_score(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle],
        safety_distance: float = 3.0
    ) -> float:
        """
        计算安全性评分
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表
            safety_distance: 安全距离
            
        Returns:
            安全性评分（0-1，越高越安全）
        """
        if not obstacles:
            return 1.0
        
        positions = trajectory.get_positions()
        min_distances = []
        
        for pos in positions:
            for obs in obstacles:
                dist = np.linalg.norm([pos[0] - obs.x, pos[1] - obs.y])
                min_distances.append(dist - obs.radius)
        
        if not min_distances:
            return 1.0
        
        min_distance = min(min_distances)
        
        # 计算安全性评分
        if min_distance >= safety_distance:
            return 1.0
        elif min_distance <= 0:
            return 0.0
        else:
            return min_distance / safety_distance
    
    def calculate_comfort_score(
        self,
        trajectory: Trajectory,
        max_comfort_acceleration: float = 2.0,
        max_comfort_jerk: float = 2.0
    ) -> float:
        """
        计算舒适性评分
        
        Args:
            trajectory: 轨迹
            max_comfort_acceleration: 最大舒适加速度
            max_comfort_jerk: 最大舒适加加速度
            
        Returns:
            舒适性评分（0-1，越高越舒适）
        """
        accelerations = trajectory.get_accelerations()
        
        if len(accelerations) == 0:
            return 1.0
        
        # 计算加速度评分
        accel_violations = np.abs(accelerations) > max_comfort_acceleration
        accel_score = 1.0 - np.mean(accel_violations)
        
        # 计算加加速度评分
        if len(accelerations) >= 2:
            jerks = np.diff(accelerations)
            jerk_violations = np.abs(jerks) > max_comfort_jerk
            jerk_score = 1.0 - np.mean(jerk_violations)
        else:
            jerk_score = 1.0
        
        # 综合评分
        comfort_score = 0.5 * accel_score + 0.5 * jerk_score
        
        return float(np.clip(comfort_score, 0, 1))
    
    def calculate_efficiency(
        self,
        trajectory: Trajectory,
        reference_trajectory: Optional[Trajectory] = None
    ) -> float:
        """
        计算轨迹效率
        
        Args:
            trajectory: 轨迹
            reference_trajectory: 参考轨迹（可选）
            
        Returns:
            效率评分（0-1，越高越高效）
        """
        if reference_trajectory is None:
            # 基于速度和加速度计算效率
            velocities = trajectory.get_velocities()
            if len(velocities) == 0:
                return 0.0
            
            avg_velocity = np.mean(velocities)
            max_velocity = 20.0  # 假设最大速度为20m/s
            
            efficiency = avg_velocity / max_velocity
        else:
            # 与参考轨迹比较
            traj_length = trajectory.get_length()
            ref_length = reference_trajectory.get_length()
            
            if ref_length == 0:
                return 0.0
            
            # 长度越接近参考轨迹越高效
            length_ratio = min(traj_length / ref_length, ref_length / traj_length)
            efficiency = length_ratio
        
        return float(np.clip(efficiency, 0, 1))
    
    def calculate_curvature_consistency(
        self,
        trajectory: Trajectory
    ) -> float:
        """
        计算曲率一致性
        
        Args:
            trajectory: 轨迹
            
        Returns:
            曲率一致性评分（0-1）
        """
        curvatures = trajectory.get_curvatures()
        
        if len(curvatures) < 2:
            return 1.0
        
        # 计算曲率变化的标准差
        curvature_changes = np.diff(curvatures)
        consistency = 1.0 / (1.0 + np.std(curvature_changes))
        
        return float(np.clip(consistency, 0, 1))
    
    def calculate_velocity_consistency(
        self,
        trajectory: Trajectory
    ) -> float:
        """
        计算速度一致性
        
        Args:
            trajectory: 轨迹
            
        Returns:
            速度一致性评分（0-1）
        """
        velocities = trajectory.get_velocities()
        
        if len(velocities) < 2:
            return 1.0
        
        # 计算速度变化的标准差
        velocity_changes = np.diff(velocities)
        consistency = 1.0 / (1.0 + np.std(velocity_changes))
        
        return float(np.clip(consistency, 0, 1))
    
    def calculate_max_jerk(
        self,
        trajectory: Trajectory
    ) -> float:
        """
        计算最大加加速度
        
        Args:
            trajectory: 轨迹
            
        Returns:
            最大加加速度
        """
        accelerations = trajectory.get_accelerations()
        
        if len(accelerations) < 2:
            return 0.0
        
        jerks = np.diff(accelerations)
        
        return float(np.max(np.abs(jerks)))
    
    def calculate_trajectory_metrics(
        self,
        trajectory: Trajectory,
        obstacles: Optional[List[Obstacle]] = None,
        reference_trajectory: Optional[Trajectory] = None
    ) -> TrajectoryMetrics:
        """
        计算轨迹的所有评估指标
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表（可选）
            reference_trajectory: 参考轨迹（可选）
            
        Returns:
            TrajectoryMetrics对象
        """
        if obstacles is None:
            obstacles = []
        
        smoothness = self.calculate_smoothness(trajectory)
        safety_score = self.calculate_safety_score(trajectory, obstacles)
        comfort_score = self.calculate_comfort_score(trajectory)
        efficiency = self.calculate_efficiency(trajectory, reference_trajectory)
        curvature_consistency = self.calculate_curvature_consistency(trajectory)
        velocity_consistency = self.calculate_velocity_consistency(trajectory)
        
        accelerations = trajectory.get_accelerations()
        max_acceleration = float(np.max(np.abs(accelerations))) if len(accelerations) > 0 else 0.0
        max_jerk = self.calculate_max_jerk(trajectory)
        
        return TrajectoryMetrics(
            smoothness=smoothness,
            safety_score=safety_score,
            comfort_score=comfort_score,
            efficiency=efficiency,
            curvature_consistency=curvature_consistency,
            velocity_consistency=velocity_consistency,
            total_length=trajectory.get_length(),
            total_duration=trajectory.get_duration(),
            max_acceleration=max_acceleration,
            max_jerk=max_jerk
        )
    
    def calculate_collision_probability(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle],
        time_horizon: float = 5.0
    ) -> float:
        """
        计算碰撞概率
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表
            time_horizon: 时间范围
            
        Returns:
            碰撞概率（0-1）
        """
        if not obstacles:
            return 0.0
        
        collision_count = 0
        total_checks = 0
        
        for point in trajectory.points:
            if point.timestamp > time_horizon:
                break
            
            for obs in obstacles:
                dist = np.linalg.norm([point.x - obs.x, point.y - obs.y])
                
                # 考虑障碍物半径和车辆半径
                collision_distance = obs.radius + 1.5  # 假设车辆半径1.5米
                
                if dist < collision_distance:
                    collision_count += 1
                
                total_checks += 1
        
        if total_checks == 0:
            return 0.0
        
        return collision_count / total_checks
    
    def compare_trajectories(
        self,
        trajectory1: Trajectory,
        trajectory2: Trajectory
    ) -> Dict[str, float]:
        """
        比较两条轨迹
        
        Args:
            trajectory1: 轨迹1
            trajectory2: 轨迹2
            
        Returns:
            比较结果字典
        """
        positions1 = trajectory1.get_positions()
        positions2 = trajectory2.get_positions()
        
        # 计算Hausdorff距离
        def hausdorff_distance(set1: np.ndarray, set2: np.ndarray) -> float:
            def directed_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
                distances = []
                for point in a:
                    dists = np.linalg.norm(b - point, axis=1)
                    distances.append(np.min(dists))
                return np.max(distances)
            
            return max(directed_hausdorff(set1, set2), directed_hausdorff(set2, set1))
        
        if len(positions1) > 0 and len(positions2) > 0:
            hausdorff_dist = hausdorff_distance(positions1, positions2)
        else:
            hausdorff_dist = float('inf')
        
        # 计算终点距离
        if len(positions1) > 0 and len(positions2) > 0:
            endpoint_dist = np.linalg.norm(positions1[-1] - positions2[-1])
        else:
            endpoint_dist = float('inf')
        
        # 计算长度差异
        length_diff = abs(trajectory1.get_length() - trajectory2.get_length())
        
        return {
            'hausdorff_distance': hausdorff_dist,
            'endpoint_distance': endpoint_dist,
            'length_difference': length_diff
        }
