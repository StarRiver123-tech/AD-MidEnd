"""
自动驾驶系统 - 轨迹生成器
实现轨迹生成和采样
"""

from typing import List, Optional
import numpy as np

from ..common.data_types import (
    Trajectory, TrajectoryPoint, PerceptionResult,
    Pose, Vector3D, Timestamp
)
from ..common.enums import BehaviorType
from ..common.geometry import normalize_angle
from ..logs.logger import Logger


class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, config: dict = None):
        """
        初始化轨迹生成器
        
        Args:
            config: 配置参数
        """
        self._config = config or {}
        self._logger = Logger("TrajectoryGenerator")
        
        # 参数
        self._num_trajectories = self._config.get('num_trajectories', 5)
        self._time_horizon = self._config.get('time_horizon', 8.0)  # 秒
        self._time_resolution = self._config.get('time_resolution', 0.1)  # 秒
        
        # 采样参数
        self._lateral_offsets = np.linspace(-2.0, 2.0, self._num_trajectories)
        
        self._logger.info(f"TrajectoryGenerator initialized: "
                         f"num_trajectories={self._num_trajectories}, "
                         f"time_horizon={self._time_horizon}s")
    
    def generate(self, perception_result: PerceptionResult,
                behavior_type: BehaviorType,
                current_speed: float) -> List[Trajectory]:
        """
        生成候选轨迹
        
        Args:
            perception_result: 感知结果
            behavior_type: 行为类型
            current_speed: 当前车速
        
        Returns:
            候选轨迹列表
        """
        trajectories = []
        
        # 根据行为类型生成轨迹
        if behavior_type == BehaviorType.CRUISE:
            trajectories = self._generate_cruise_trajectories(
                perception_result, current_speed
            )
        elif behavior_type == BehaviorType.FOLLOW:
            trajectories = self._generate_follow_trajectories(
                perception_result, current_speed
            )
        elif behavior_type == BehaviorType.LANE_CHANGE_LEFT:
            trajectories = self._generate_lane_change_trajectories(
                perception_result, current_speed, direction='left'
            )
        elif behavior_type == BehaviorType.LANE_CHANGE_RIGHT:
            trajectories = self._generate_lane_change_trajectories(
                perception_result, current_speed, direction='right'
            )
        elif behavior_type in [BehaviorType.STOP, BehaviorType.EMERGENCY_STOP]:
            trajectories = self._generate_stop_trajectories(
                perception_result, current_speed
            )
        else:
            # 默认生成巡航轨迹
            trajectories = self._generate_cruise_trajectories(
                perception_result, current_speed
            )
        
        # 评估轨迹代价
        for trajectory in trajectories:
            self._evaluate_trajectory(trajectory, perception_result)
        
        return trajectories
    
    def _generate_cruise_trajectories(self, perception_result: PerceptionResult,
                                     current_speed: float) -> List[Trajectory]:
        """生成巡航轨迹"""
        trajectories = []
        
        # 获取车道线信息
        lane_center_y = 0.0  # 默认车道中心
        if perception_result.lane_result and perception_result.lane_result.lane_lines:
            # 计算车道中心
            left_y = None
            right_y = None
            for line in perception_result.lane_result.lane_lines:
                if line.line_id == 0:  # 左车道线
                    left_y = line.coefficients[0]
                elif line.line_id == 1:  # 右车道线
                    right_y = line.coefficients[0]
            
            if left_y is not None and right_y is not None:
                lane_center_y = (left_y + right_y) / 2
        
        # 生成多条横向偏移的轨迹
        for i, offset in enumerate(self._lateral_offsets):
            trajectory = self._generate_straight_trajectory(
                target_y=lane_center_y + offset,
                current_speed=current_speed,
                target_speed=current_speed * 1.1  # 稍微加速
            )
            trajectory.trajectory_id = i
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_follow_trajectories(self, perception_result: PerceptionResult,
                                     current_speed: float) -> List[Trajectory]:
        """生成跟车轨迹"""
        trajectories = []
        
        # 找到前方车辆
        front_vehicle = None
        target_distance = 20.0
        
        if perception_result.obstacle_result:
            for obstacle in perception_result.obstacle_result.obstacles:
                if obstacle.bbox.center.position.x > 0 and \
                   abs(obstacle.bbox.center.position.y) < 2.0:
                    if front_vehicle is None or \
                       obstacle.bbox.center.position.x < front_vehicle.bbox.center.position.x:
                        front_vehicle = obstacle
        
        if front_vehicle:
            target_distance = front_vehicle.bbox.center.position.x
            target_speed = front_vehicle.velocity.x
        else:
            target_speed = current_speed
        
        # 生成减速轨迹
        for i, offset in enumerate(self._lateral_offsets):
            trajectory = self._generate_straight_trajectory(
                target_y=offset,
                current_speed=current_speed,
                target_speed=max(0, target_speed),
                end_distance=target_distance * 0.8
            )
            trajectory.trajectory_id = i
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_lane_change_trajectories(self, perception_result: PerceptionResult,
                                          current_speed: float,
                                          direction: str) -> List[Trajectory]:
        """生成变道轨迹"""
        trajectories = []
        
        # 变道目标偏移
        lane_width = 3.5
        target_offset = lane_width if direction == 'left' else -lane_width
        
        # 生成变道轨迹
        trajectory = self._generate_lateral_shift_trajectory(
            current_speed=current_speed,
            target_speed=current_speed,
            target_offset=target_offset,
            shift_time=3.0  # 变道时间
        )
        trajectory.trajectory_id = 0
        trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_stop_trajectories(self, perception_result: PerceptionResult,
                                   current_speed: float) -> List[Trajectory]:
        """生成停车轨迹"""
        trajectories = []
        
        # 找到停车位置
        stop_distance = 10.0
        
        if perception_result.obstacle_result:
            for obstacle in perception_result.obstacle_result.obstacles:
                if obstacle.bbox.center.position.x > 0:
                    stop_distance = min(stop_distance, obstacle.bbox.center.position.x - 2.0)
        
        # 生成减速到停止的轨迹
        trajectory = self._generate_deceleration_trajectory(
            current_speed=current_speed,
            stop_distance=max(0, stop_distance)
        )
        trajectory.trajectory_id = 0
        trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_straight_trajectory(self, target_y: float,
                                     current_speed: float,
                                     target_speed: float,
                                     end_distance: Optional[float] = None) -> Trajectory:
        """生成直线轨迹"""
        num_points = int(self._time_horizon / self._time_resolution) + 1
        
        points = []
        
        # 生成纵向位置
        if end_distance is None:
            end_distance = current_speed * self._time_horizon
        
        x_positions = np.linspace(0, end_distance, num_points)
        
        # 生成横向位置（平滑过渡到目标y）
        y_positions = np.linspace(0, target_y, num_points)
        
        # 生成速度曲线
        speeds = np.linspace(current_speed, target_speed, num_points)
        
        for i in range(num_points):
            point = TrajectoryPoint(
                timestamp=Timestamp.now(),
                relative_time=i * self._time_resolution,
                pose=Pose(position=Vector3D(x_positions[i], y_positions[i], 0)),
                longitudinal_velocity=speeds[i],
                lateral_velocity=0.0,
                longitudinal_acceleration=0.0,
                lateral_acceleration=0.0,
                curvature=0.0,
                theta=0.0
            )
            points.append(point)
        
        trajectory = Trajectory(
            trajectory_id=0,
            trajectory_type="normal",
            points=points,
            total_time=self._time_horizon,
            total_length=end_distance,
            cost=0.0,
            is_feasible=True
        )
        
        return trajectory
    
    def _generate_lateral_shift_trajectory(self, current_speed: float,
                                          target_speed: float,
                                          target_offset: float,
                                          shift_time: float) -> Trajectory:
        """生成横向偏移轨迹"""
        num_points = int(self._time_horizon / self._time_resolution) + 1
        
        points = []
        
        # 纵向距离
        end_distance = current_speed * self._time_horizon
        x_positions = np.linspace(0, end_distance, num_points)
        
        # 横向偏移（使用正弦曲线平滑过渡）
        shift_points = int(shift_time / self._time_resolution)
        y_positions = np.zeros(num_points)
        
        for i in range(min(shift_points, num_points)):
            t = i / shift_points
            y_positions[i] = target_offset * (0.5 - 0.5 * np.cos(np.pi * t))
        
        y_positions[shift_points:] = target_offset
        
        # 速度
        speeds = np.linspace(current_speed, target_speed, num_points)
        
        for i in range(num_points):
            point = TrajectoryPoint(
                timestamp=Timestamp.now(),
                relative_time=i * self._time_resolution,
                pose=Pose(position=Vector3D(x_positions[i], y_positions[i], 0)),
                longitudinal_velocity=speeds[i],
                lateral_velocity=0.0,
                longitudinal_acceleration=0.0,
                lateral_acceleration=0.0,
                curvature=0.0,
                theta=normalize_angle(np.arctan2(
                    y_positions[min(i+1, num_points-1)] - y_positions[i],
                    x_positions[min(i+1, num_points-1)] - x_positions[i]
                )) if i < num_points - 1 else 0.0
            )
            points.append(point)
        
        trajectory = Trajectory(
            trajectory_id=0,
            trajectory_type="lane_change",
            points=points,
            total_time=self._time_horizon,
            total_length=end_distance,
            cost=0.0,
            is_feasible=True
        )
        
        return trajectory
    
    def _generate_deceleration_trajectory(self, current_speed: float,
                                         stop_distance: float) -> Trajectory:
        """生成减速停车轨迹"""
        # 计算停车所需时间
        if current_speed > 0:
            deceleration = 2.0  # m/s^2
            stop_time = current_speed / deceleration
        else:
            stop_time = 0.0
        
        num_points = int(self._time_horizon / self._time_resolution) + 1
        stop_point = int(stop_time / self._time_resolution)
        
        points = []
        
        for i in range(num_points):
            t = i * self._time_resolution
            
            if i <= stop_point and stop_point > 0:
                # 减速阶段
                progress = i / stop_point
                x = stop_distance * (progress * 2 - progress ** 2)
                v = current_speed * (1 - progress)
                a = -deceleration
            else:
                # 停车阶段
                x = stop_distance
                v = 0.0
                a = 0.0
            
            point = TrajectoryPoint(
                timestamp=Timestamp.now(),
                relative_time=t,
                pose=Pose(position=Vector3D(x, 0, 0)),
                longitudinal_velocity=v,
                lateral_velocity=0.0,
                longitudinal_acceleration=a,
                lateral_acceleration=0.0,
                curvature=0.0,
                theta=0.0
            )
            points.append(point)
        
        trajectory = Trajectory(
            trajectory_id=0,
            trajectory_type="stop",
            points=points,
            total_time=self._time_horizon,
            total_length=stop_distance,
            cost=0.0,
            is_feasible=True
        )
        
        return trajectory
    
    def _evaluate_trajectory(self, trajectory: Trajectory,
                            perception_result: PerceptionResult) -> None:
        """评估轨迹代价"""
        # 安全性代价
        safety_cost = self._evaluate_safety(trajectory, perception_result)
        
        # 舒适性代价
        comfort_cost = self._evaluate_comfort(trajectory)
        
        # 效率代价
        efficiency_cost = self._evaluate_efficiency(trajectory)
        
        # 总代价
        trajectory.safety_cost = safety_cost
        trajectory.comfort_cost = comfort_cost
        trajectory.efficiency_cost = efficiency_cost
        trajectory.cost = safety_cost * 10.0 + comfort_cost * 2.0 + efficiency_cost
    
    def _evaluate_safety(self, trajectory: Trajectory,
                        perception_result: PerceptionResult) -> float:
        """评估安全性代价"""
        cost = 0.0
        
        # 检查与障碍物的碰撞
        if perception_result.obstacle_result:
            for obstacle in perception_result.obstacle_result.obstacles:
                for point in trajectory.points:
                    # 计算到障碍物的距离
                    dx = point.pose.position.x - obstacle.bbox.center.position.x
                    dy = point.pose.position.y - obstacle.bbox.center.position.y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # 距离越近，代价越高
                    if distance < 5.0:
                        cost += (5.0 - distance) / 5.0
        
        return cost
    
    def _evaluate_comfort(self, trajectory: Trajectory) -> float:
        """评估舒适性代价"""
        cost = 0.0
        
        # 计算加速度变化
        for i in range(1, len(trajectory.points)):
            prev = trajectory.points[i-1]
            curr = trajectory.points[i]
            
            # 纵向加速度变化
            long_jerk = abs(curr.longitudinal_acceleration - prev.longitudinal_acceleration)
            cost += long_jerk * 0.1
            
            # 横向加速度
            lat_accel = abs(curr.lateral_acceleration)
            cost += lat_accel * 0.1
        
        return cost
    
    def _evaluate_efficiency(self, trajectory: Trajectory) -> float:
        """评估效率代价"""
        # 平均速度越低，代价越高
        if trajectory.points:
            avg_speed = np.mean([p.longitudinal_velocity for p in trajectory.points])
            return max(0, 10.0 - avg_speed) * 0.1
        return 0.0
