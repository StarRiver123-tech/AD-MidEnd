"""
规划模块单元测试
测试轨迹生成、评估指标等功能
"""

import pytest
import numpy as np
from typing import List

from metrics.planning_metrics import (
    PlanningMetrics, Trajectory, TrajectoryPoint,
    TrajectoryMetrics, Obstacle
)


@pytest.mark.unit
@pytest.mark.planning
class TestTrajectoryGeneration:
    """轨迹生成单元测试"""
    
    def test_straight_trajectory(self):
        """测试直线轨迹"""
        points = []
        for i in range(50):
            t = i * 0.1
            point = TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            )
            points.append(point)
        
        trajectory = Trajectory(points=points)
        
        assert len(trajectory.points) == 50
        assert trajectory.get_length() > 0
        assert trajectory.get_duration() == 4.9  # 49 * 0.1
        
        # 检查直线特性
        curvatures = trajectory.get_curvatures()
        assert np.allclose(curvatures, 0)
    
    def test_curved_trajectory(self):
        """测试曲线轨迹"""
        points = []
        radius = 50.0
        velocity = 10.0
        
        for i in range(50):
            t = i * 0.1
            angle = velocity * t / radius
            
            point = TrajectoryPoint(
                x=radius * np.sin(angle),
                y=radius * (1 - np.cos(angle)),
                heading=angle,
                velocity=velocity,
                acceleration=0.0,
                curvature=1.0 / radius,
                timestamp=t
            )
            points.append(point)
        
        trajectory = Trajectory(points=points)
        
        assert len(trajectory.points) == 50
        assert trajectory.get_length() > 0
        
        # 检查曲率
        curvatures = trajectory.get_curvatures()
        assert np.all(curvatures > 0)
    
    def test_accelerating_trajectory(self):
        """测试加速轨迹"""
        points = []
        v0 = 5.0
        a = 2.0
        
        for i in range(50):
            t = i * 0.1
            v = v0 + a * t
            x = v0 * t + 0.5 * a * t ** 2
            
            point = TrajectoryPoint(
                x=x,
                y=0.0,
                heading=0.0,
                velocity=v,
                acceleration=a,
                curvature=0.0,
                timestamp=t
            )
            points.append(point)
        
        trajectory = Trajectory(points=points)
        
        # 检查速度递增
        velocities = trajectory.get_velocities()
        for i in range(1, len(velocities)):
            assert velocities[i] >= velocities[i-1]
    
    def test_decelerating_trajectory(self):
        """测试减速轨迹"""
        points = []
        v0 = 20.0
        a = -2.0
        
        for i in range(50):
            t = i * 0.1
            v = max(0, v0 + a * t)
            x = v0 * t + 0.5 * a * t ** 2
            
            point = TrajectoryPoint(
                x=x,
                y=0.0,
                heading=0.0,
                velocity=v,
                acceleration=a if v > 0 else 0,
                curvature=0.0,
                timestamp=t
            )
            points.append(point)
        
        trajectory = Trajectory(points=points)
        
        # 检查速度递减
        velocities = trajectory.get_velocities()
        for i in range(1, len(velocities)):
            if velocities[i-1] > 0:
                assert velocities[i] <= velocities[i-1]


@pytest.mark.unit
@pytest.mark.planning
class TestTrajectoryMetrics:
    """轨迹评估指标单元测试"""
    
    def test_smoothness_calculation(self, planning_metrics):
        """测试平滑度计算"""
        # 平滑轨迹
        smooth_points = []
        for i in range(50):
            t = i * 0.1
            smooth_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        smooth_trajectory = Trajectory(points=smooth_points)
        
        # 不平滑轨迹
        rough_points = []
        for i in range(50):
            t = i * 0.1
            rough_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=np.sin(t * 10) * 5,  # 剧烈变化的加速度
                curvature=np.sin(t * 10) * 0.1,  # 剧烈变化的曲率
                timestamp=t
            ))
        rough_trajectory = Trajectory(points=rough_points)
        
        smooth_smoothness = planning_metrics.calculate_smoothness(smooth_trajectory)
        rough_smoothness = planning_metrics.calculate_smoothness(rough_trajectory)
        
        assert smooth_smoothness > rough_smoothness
        assert 0 <= smooth_smoothness <= 1
        assert 0 <= rough_smoothness <= 1
    
    def test_safety_score_calculation(self, planning_metrics):
        """测试安全性评分计算"""
        # 安全轨迹（远离障碍物）
        safe_points = []
        for i in range(50):
            t = i * 0.1
            safe_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        safe_trajectory = Trajectory(points=safe_points)
        
        # 危险轨迹（靠近障碍物）
        dangerous_points = []
        for i in range(50):
            t = i * 0.1
            dangerous_points.append(TrajectoryPoint(
                x=20.0,  # 靠近障碍物
                y=2.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        dangerous_trajectory = Trajectory(points=dangerous_points)
        
        obstacles = [
            Obstacle(x=20, y=2, radius=1.5, velocity_x=0, velocity_y=0)
        ]
        
        safe_score = planning_metrics.calculate_safety_score(safe_trajectory, obstacles)
        dangerous_score = planning_metrics.calculate_safety_score(dangerous_trajectory, obstacles)
        
        assert safe_score > dangerous_score
        assert 0 <= safe_score <= 1
        assert 0 <= dangerous_score <= 1
    
    def test_comfort_score_calculation(self, planning_metrics):
        """测试舒适性评分计算"""
        # 舒适轨迹
        comfortable_points = []
        for i in range(50):
            t = i * 0.1
            comfortable_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.5,  # 小加速度
                curvature=0.0,
                timestamp=t
            ))
        comfortable_trajectory = Trajectory(points=comfortable_points)
        
        # 不舒适轨迹
        uncomfortable_points = []
        for i in range(50):
            t = i * 0.1
            uncomfortable_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=5.0,  # 大加速度
                curvature=0.0,
                timestamp=t
            ))
        uncomfortable_trajectory = Trajectory(points=uncomfortable_points)
        
        comfortable_score = planning_metrics.calculate_comfort_score(comfortable_trajectory)
        uncomfortable_score = planning_metrics.calculate_comfort_score(uncomfortable_trajectory)
        
        assert comfortable_score > uncomfortable_score
        assert 0 <= comfortable_score <= 1
        assert 0 <= uncomfortable_score <= 1
    
    def test_efficiency_calculation(self, planning_metrics):
        """测试效率计算"""
        # 高效轨迹（高速）
        efficient_points = []
        for i in range(50):
            t = i * 0.1
            efficient_points.append(TrajectoryPoint(
                x=t * 15,  # 高速
                y=0.0,
                heading=0.0,
                velocity=15.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        efficient_trajectory = Trajectory(points=efficient_points)
        
        # 低效轨迹（低速）
        inefficient_points = []
        for i in range(50):
            t = i * 0.1
            inefficient_points.append(TrajectoryPoint(
                x=t * 5,  # 低速
                y=0.0,
                heading=0.0,
                velocity=5.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        inefficient_trajectory = Trajectory(points=inefficient_points)
        
        efficient_score = planning_metrics.calculate_efficiency(efficient_trajectory)
        inefficient_score = planning_metrics.calculate_efficiency(inefficient_trajectory)
        
        assert efficient_score > inefficient_score
        assert 0 <= efficient_score <= 1
        assert 0 <= inefficient_score <= 1
    
    def test_curvature_consistency(self, planning_metrics):
        """测试曲率一致性"""
        # 一致曲率轨迹（圆弧）
        consistent_points = []
        radius = 50.0
        for i in range(50):
            t = i * 0.1
            consistent_points.append(TrajectoryPoint(
                x=radius * np.sin(t * 0.2),
                y=radius * (1 - np.cos(t * 0.2)),
                heading=t * 0.2,
                velocity=10.0,
                acceleration=0.0,
                curvature=1.0 / radius,  # 恒定曲率
                timestamp=t
            ))
        consistent_trajectory = Trajectory(points=consistent_points)
        
        # 不一致曲率轨迹
        inconsistent_points = []
        for i in range(50):
            t = i * 0.1
            inconsistent_points.append(TrajectoryPoint(
                x=t * 10,
                y=np.sin(t * 5) * 5,  # 正弦曲线
                heading=np.arctan(np.cos(t * 5) * 5),
                velocity=10.0,
                acceleration=0.0,
                curvature=np.abs(-np.sin(t * 5) * 25),  # 变化曲率
                timestamp=t
            ))
        inconsistent_trajectory = Trajectory(points=inconsistent_points)
        
        consistent_score = planning_metrics.calculate_curvature_consistency(consistent_trajectory)
        inconsistent_score = planning_metrics.calculate_curvature_consistency(inconsistent_trajectory)
        
        assert consistent_score > inconsistent_score
        assert 0 <= consistent_score <= 1
        assert 0 <= inconsistent_score <= 1
    
    def test_max_jerk_calculation(self, planning_metrics):
        """测试最大加加速度计算"""
        # 高加加速度轨迹
        high_jerk_points = []
        for i in range(50):
            t = i * 0.1
            high_jerk_points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=np.sin(t * 10) * 5,  # 快速变化的加速度
                curvature=0.0,
                timestamp=t
            ))
        high_jerk_trajectory = Trajectory(points=high_jerk_points)
        
        max_jerk = planning_metrics.calculate_max_jerk(high_jerk_trajectory)
        
        assert max_jerk > 0
    
    def test_full_trajectory_metrics(self, planning_metrics, sample_straight_trajectory, sample_obstacles):
        """测试完整轨迹指标计算"""
        metrics = planning_metrics.calculate_trajectory_metrics(
            sample_straight_trajectory,
            obstacles=sample_obstacles
        )
        
        assert isinstance(metrics, TrajectoryMetrics)
        assert 0 <= metrics.smoothness <= 1
        assert 0 <= metrics.safety_score <= 1
        assert 0 <= metrics.comfort_score <= 1
        assert 0 <= metrics.efficiency <= 1
        assert metrics.total_length > 0
        assert metrics.total_duration > 0
    
    def test_collision_probability(self, planning_metrics):
        """测试碰撞概率计算"""
        # 碰撞轨迹
        collision_points = []
        for i in range(50):
            t = i * 0.1
            collision_points.append(TrajectoryPoint(
                x=20.0,  # 穿过障碍物
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        collision_trajectory = Trajectory(points=collision_points)
        
        obstacles = [
            Obstacle(x=20, y=0, radius=2.0, velocity_x=0, velocity_y=0)
        ]
        
        collision_prob = planning_metrics.calculate_collision_probability(
            collision_trajectory, obstacles
        )
        
        assert collision_prob > 0  # 应该有碰撞
        assert collision_prob <= 1
    
    def test_trajectory_comparison(self, planning_metrics):
        """测试轨迹比较"""
        # 两条相似轨迹
        points1 = []
        points2 = []
        for i in range(50):
            t = i * 0.1
            points1.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
            points2.append(TrajectoryPoint(
                x=t * 10,
                y=0.5,  # 稍微偏移
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        
        trajectory1 = Trajectory(points=points1)
        trajectory2 = Trajectory(points=points2)
        
        comparison = planning_metrics.compare_trajectories(trajectory1, trajectory2)
        
        assert 'hausdorff_distance' in comparison
        assert 'endpoint_distance' in comparison
        assert 'length_difference' in comparison
        assert comparison['hausdorff_distance'] < 1.0  # 距离应该很小


@pytest.mark.unit
@pytest.mark.planning
class TestObstacleHandling:
    """障碍物处理单元测试"""
    
    def test_obstacle_creation(self):
        """测试障碍物创建"""
        obstacle = Obstacle(
            x=10.0,
            y=5.0,
            radius=1.5,
            velocity_x=2.0,
            velocity_y=0.0
        )
        
        assert obstacle.x == 10.0
        assert obstacle.y == 5.0
        assert obstacle.radius == 1.5
        assert obstacle.velocity_x == 2.0
        assert obstacle.velocity_y == 0.0
    
    def test_multiple_obstacles_safety(self, planning_metrics):
        """测试多障碍物安全性"""
        points = []
        for i in range(50):
            t = i * 0.1
            points.append(TrajectoryPoint(
                x=t * 10,
                y=0.0,
                heading=0.0,
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            ))
        trajectory = Trajectory(points=points)
        
        # 多个障碍物
        obstacles = [
            Obstacle(x=20, y=5, radius=2.0, velocity_x=0, velocity_y=0),
            Obstacle(x=40, y=-5, radius=2.0, velocity_x=0, velocity_y=0),
            Obstacle(x=30, y=0, radius=1.0, velocity_x=5, velocity_y=0),
        ]
        
        safety_score = planning_metrics.calculate_safety_score(trajectory, obstacles)
        
        assert 0 <= safety_score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
