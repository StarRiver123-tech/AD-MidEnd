"""
自动驾驶系统 - 轨迹优化器
实现轨迹优化功能
"""

from typing import Optional
import numpy as np

from ..common.data_types import Trajectory, TrajectoryPoint, PerceptionResult
from ..common.geometry import normalize_angle
from ..logs.logger import Logger


class TrajectoryOptimizer:
    """轨迹优化器"""
    
    def __init__(self, config: dict = None):
        """
        初始化轨迹优化器
        
        Args:
            config: 配置参数
        """
        self._config = config or {}
        self._logger = Logger("TrajectoryOptimizer")
        
        # 参数
        self._max_iterations = self._config.get('max_iterations', 100)
        self._convergence_threshold = self._config.get('convergence_threshold', 1e-3)
        
        # 约束参数
        self._max_longitudinal_acceleration = 3.0  # m/s^2
        self._max_lateral_acceleration = 2.0  # m/s^2
        self._max_curvature = 0.2  # 1/m
        
        self._logger.info("TrajectoryOptimizer initialized")
    
    def optimize(self, trajectory: Trajectory,
                perception_result: PerceptionResult) -> Trajectory:
        """
        优化轨迹
        
        Args:
            trajectory: 输入轨迹
            perception_result: 感知结果
        
        Returns:
            优化后的轨迹
        """
        if trajectory is None or not trajectory.points:
            return trajectory
        
        # 创建轨迹副本
        optimized = self._copy_trajectory(trajectory)
        
        # 1. 平滑轨迹
        self._smooth_trajectory(optimized)
        
        # 2. 应用运动学约束
        self._apply_kinematic_constraints(optimized)
        
        # 3. 避障优化
        self._obstacle_avoidance(optimized, perception_result)
        
        # 4. 重新计算轨迹属性
        self._recalculate_trajectory_properties(optimized)
        
        return optimized
    
    def _copy_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """复制轨迹"""
        from copy import deepcopy
        return deepcopy(trajectory)
    
    def _smooth_trajectory(self, trajectory: Trajectory) -> None:
        """平滑轨迹"""
        if len(trajectory.points) < 3:
            return
        
        # 提取位置
        positions = np.array([
            [p.pose.position.x, p.pose.position.y, p.pose.position.z]
            for p in trajectory.points
        ])
        
        # 使用移动平均平滑
        window_size = 5
        smoothed = positions.copy()
        
        for i in range(window_size // 2, len(positions) - window_size // 2):
            smoothed[i] = np.mean(positions[i - window_size // 2:i + window_size // 2 + 1], axis=0)
        
        # 更新轨迹点
        for i, point in enumerate(trajectory.points):
            point.pose.position.x = smoothed[i, 0]
            point.pose.position.y = smoothed[i, 1]
            point.pose.position.z = smoothed[i, 2]
    
    def _apply_kinematic_constraints(self, trajectory: Trajectory) -> None:
        """应用运动学约束"""
        if len(trajectory.points) < 2:
            return
        
        dt = trajectory.points[1].relative_time - trajectory.points[0].relative_time
        
        for i in range(1, len(trajectory.points)):
            point = trajectory.points[i]
            prev_point = trajectory.points[i - 1]
            
            # 限制纵向加速度
            if point.longitudinal_acceleration > self._max_longitudinal_acceleration:
                point.longitudinal_acceleration = self._max_longitudinal_acceleration
            elif point.longitudinal_acceleration < -self._max_longitudinal_acceleration:
                point.longitudinal_acceleration = -self._max_longitudinal_acceleration
            
            # 限制横向加速度
            if abs(point.lateral_acceleration) > self._max_lateral_acceleration:
                point.lateral_acceleration = np.sign(point.lateral_acceleration) * \
                                            self._max_lateral_acceleration
            
            # 限制曲率
            if abs(point.curvature) > self._max_curvature:
                point.curvature = np.sign(point.curvature) * self._max_curvature
            
            # 根据加速度限制更新速度
            dv = point.longitudinal_acceleration * dt
            point.longitudinal_velocity = prev_point.longitudinal_velocity + dv
            
            # 确保速度非负
            point.longitudinal_velocity = max(0, point.longitudinal_velocity)
    
    def _obstacle_avoidance(self, trajectory: Trajectory,
                           perception_result: PerceptionResult) -> None:
        """避障优化"""
        if not perception_result.obstacle_result:
            return
        
        for obstacle in perception_result.obstacle_result.obstacles:
            # 检查轨迹是否与障碍物冲突
            for point in trajectory.points:
                # 计算到障碍物的距离
                dx = point.pose.position.x - obstacle.bbox.center.position.x
                dy = point.pose.position.y - obstacle.bbox.center.position.y
                distance = np.sqrt(dx**2 + dy**2)
                
                # 安全距离
                safety_distance = 2.0
                
                # 如果距离过近，调整轨迹
                if distance < safety_distance:
                    # 计算避障方向
                    if dy > 0:
                        # 障碍物在左侧，向右避让
                        point.pose.position.y += (safety_distance - distance) * 0.5
                    else:
                        # 障碍物在右侧，向左避让
                        point.pose.position.y -= (safety_distance - distance) * 0.5
    
    def _recalculate_trajectory_properties(self, trajectory: Trajectory) -> None:
        """重新计算轨迹属性"""
        if len(trajectory.points) < 2:
            return
        
        # 计算总长度
        total_length = 0.0
        for i in range(1, len(trajectory.points)):
            prev = trajectory.points[i - 1]
            curr = trajectory.points[i]
            
            dx = curr.pose.position.x - prev.pose.position.x
            dy = curr.pose.position.y - prev.pose.position.y
            segment_length = np.sqrt(dx**2 + dy**2)
            total_length += segment_length
            
            # 更新航向角
            curr.theta = normalize_angle(np.arctan2(dy, dx))
            
            # 更新曲率
            if i > 1:
                prev_theta = trajectory.points[i - 1].theta
                dtheta = curr.theta - prev_theta
                if segment_length > 0:
                    curr.curvature = dtheta / segment_length
        
        trajectory.total_length = total_length
        
        # 计算总时间
        if trajectory.points:
            trajectory.total_time = trajectory.points[-1].relative_time
    
    def check_trajectory_feasibility(self, trajectory: Trajectory) -> bool:
        """检查轨迹可行性"""
        if trajectory is None or not trajectory.points:
            return False
        
        for point in trajectory.points:
            # 检查加速度约束
            if abs(point.longitudinal_acceleration) > self._max_longitudinal_acceleration:
                return False
            
            if abs(point.lateral_acceleration) > self._max_lateral_acceleration:
                return False
            
            # 检查曲率约束
            if abs(point.curvature) > self._max_curvature:
                return False
            
            # 检查速度非负
            if point.longitudinal_velocity < 0:
                return False
        
        return True
    
    def interpolate_trajectory(self, trajectory: Trajectory,
                              target_time: float) -> Optional[TrajectoryPoint]:
        """在轨迹中插值"""
        if not trajectory.points:
            return None
        
        # 找到目标时间前后的点
        for i in range(len(trajectory.points) - 1):
            curr = trajectory.points[i]
            next_point = trajectory.points[i + 1]
            
            if curr.relative_time <= target_time <= next_point.relative_time:
                # 线性插值
                t = (target_time - curr.relative_time) / \
                    (next_point.relative_time - curr.relative_time)
                
                interpolated = TrajectoryPoint()
                interpolated.relative_time = target_time
                
                # 位置插值
                interpolated.pose.position.x = curr.pose.position.x + \
                    t * (next_point.pose.position.x - curr.pose.position.x)
                interpolated.pose.position.y = curr.pose.position.y + \
                    t * (next_point.pose.position.y - curr.pose.position.y)
                interpolated.pose.position.z = curr.pose.position.z + \
                    t * (next_point.pose.position.z - curr.pose.position.z)
                
                # 速度插值
                interpolated.longitudinal_velocity = curr.longitudinal_velocity + \
                    t * (next_point.longitudinal_velocity - curr.longitudinal_velocity)
                
                return interpolated
        
        # 返回最近的点
        if target_time <= trajectory.points[0].relative_time:
            return trajectory.points[0]
        else:
            return trajectory.points[-1]
