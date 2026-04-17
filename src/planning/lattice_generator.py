"""
Lattice Trajectory Generator for Autonomous Driving
基于Lattice网格撒点的轨迹生成器
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class VehicleState:
    """车辆状态"""
    x: float  # 位置x
    y: float  # 位置y
    theta: float  # 航向角
    v: float  # 速度
    a: float = 0.0  # 加速度
    kappa: float = 0.0  # 曲率


@dataclass
class Trajectory:
    """轨迹类"""
    x: np.ndarray  # x坐标序列
    y: np.ndarray  # y坐标序列
    theta: np.ndarray  # 航向角序列
    v: np.ndarray  # 速度序列
    a: np.ndarray  # 加速度序列
    kappa: np.ndarray  # 曲率序列
    t: np.ndarray  # 时间序列
    cost: float = float('inf')  # 轨迹代价
    behavior_type: str = "unknown"  # 行为类型
    
    def __post_init__(self):
        if len(self.x) != len(self.y) or len(self.x) != len(self.t):
            raise ValueError("Trajectory arrays must have same length")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (
            np.array_equal(self.x, other.x) and
            np.array_equal(self.y, other.y) and
            np.array_equal(self.theta, other.theta) and
            np.array_equal(self.v, other.v) and
            np.array_equal(self.a, other.a) and
            np.array_equal(self.kappa, other.kappa) and
            np.array_equal(self.t, other.t) and
            self.cost == other.cost and
            self.behavior_type == other.behavior_type
        )
    
    def __hash__(self) -> int:
        return id(self)
    
    def get_point_at_time(self, time: float) -> Tuple[float, float, float, float]:
        """获取指定时间的轨迹点 (x, y, theta, v)"""
        idx = np.argmin(np.abs(self.t - time))
        return self.x[idx], self.y[idx], self.theta[idx], self.v[idx]
    
    def get_length(self) -> float:
        """计算轨迹长度"""
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        return np.sum(np.sqrt(dx**2 + dy**2))


@dataclass
class Obstacle:
    """障碍物"""
    x: float
    y: float
    vx: float
    vy: float
    category: str
    width: float = 2.0
    length: float = 4.5
    
    def get_position_at_time(self, t: float) -> Tuple[float, float]:
        """预测t时刻的障碍物位置"""
        return self.x + self.vx * t, self.y + self.vy * t
    
    def get_bounding_box(self, t: float = 0) -> Tuple[float, float, float, float]:
        """获取障碍物边界框 (min_x, max_x, min_y, max_y)"""
        px, py = self.get_position_at_time(t)
        half_w = self.width / 2
        half_l = self.length / 2
        return (px - half_l, px + half_l, py - half_w, py + half_w)


@dataclass
class LaneInfo:
    """车道信息"""
    x: np.ndarray
    y: np.ndarray
    width: float = 3.5
    
    def get_closest_point(self, x: float, y: float) -> Tuple[int, float, float]:
        """获取最近点索引和坐标"""
        dists = np.sqrt((self.x - x)**2 + (self.y - y)**2)
        idx = np.argmin(dists)
        return idx, self.x[idx], self.y[idx]
    
    def get_direction_at_index(self, idx: int) -> float:
        """获取指定索引处的车道方向"""
        if idx >= len(self.x) - 1:
            idx = len(self.x) - 2
        dx = self.x[idx + 1] - self.x[idx]
        dy = self.y[idx + 1] - self.y[idx]
        return math.atan2(dy, dx)


class LatticeGenerator:
    """
    Lattice轨迹生成器
    使用Frenet坐标系进行轨迹采样
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 采样参数
        self.num_t_samples = config.get('num_t_samples', 5)  # 时间采样数
        self.num_d_samples = config.get('num_d_samples', 7)  # 横向偏移采样数
        self.num_v_samples = config.get('num_v_samples', 5)  # 速度采样数
        
        # 采样范围
        self.t_min = config.get('t_min', 3.0)  # 最小预测时间
        self.t_max = config.get('t_max', 8.0)  # 最大预测时间
        self.d_min = config.get('d_min', -3.5)  # 最小横向偏移
        self.d_max = config.get('d_max', 3.5)   # 最大横向偏移
        self.v_min = config.get('v_min', 0.0)   # 最小速度
        self.v_max = config.get('v_max', 20.0)  # 最大速度 (m/s)
        
        # 车辆约束
        self.max_acceleration = config.get('max_acceleration', 3.0)  # 最大加速度
        self.max_deceleration = config.get('max_deceleration', -5.0)  # 最大减速度
        self.max_curvature = config.get('max_curvature', 0.2)  # 最大曲率
        self.max_jerk = config.get('max_jerk', 5.0)  # 最大jerk
        
        # 安全参数
        self.safe_distance = config.get('safe_distance', 2.0)  # 安全距离
        self.vehicle_width = config.get('vehicle_width', 2.0)
        self.vehicle_length = config.get('vehicle_length', 4.5)
    
    def generate_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle],
        target_speed: Optional[float] = None
    ) -> List[Trajectory]:
        """
        生成候选轨迹
        
        Args:
            current_state: 当前车辆状态
            lane_info: 车道信息
            obstacles: 障碍物列表
            target_speed: 目标速度 (可选)
            
        Returns:
            候选轨迹列表
        """
        trajectories = []
        
        # 采样参数
        t_samples = np.linspace(self.t_min, self.t_max, self.num_t_samples)
        d_samples = np.linspace(self.d_min, self.d_max, self.num_d_samples)
        
        if target_speed is None:
            target_speed = current_state.v
        v_samples = np.linspace(
            max(self.v_min, target_speed - 5),
            min(self.v_max, target_speed + 5),
            self.num_v_samples
        )
        
        # 对每个采样组合生成轨迹
        for t in t_samples:
            for d in d_samples:
                for v in v_samples:
                    traj = self._generate_single_trajectory(
                        current_state, lane_info, t, d, v
                    )
                    
                    if traj is not None and self._check_trajectory_validity(traj, obstacles):
                        trajectories.append(traj)
        
        return trajectories
    
    def _generate_single_trajectory(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        target_t: float,
        target_d: float,
        target_v: float
    ) -> Optional[Trajectory]:
        """
        生成单条轨迹
        
        使用五次多项式进行轨迹规划
        """
        try:
            # 时间步长
            dt = 0.1
            num_points = int(target_t / dt) + 1
            t_array = np.linspace(0, target_t, num_points)
            
            # 在Frenet坐标系下规划
            # 纵向: 四次多项式 (s)
            # 横向: 五次多项式 (d)
            
            # 当前Frenet坐标
            s0 = 0.0  # 纵向位移从0开始
            ds0 = current_state.v  # 纵向速度
            dds0 = current_state.a  # 纵向加速度
            
            d0 = 0.0  # 当前横向偏移 (假设在车道中心)
            dd0 = 0.0  # 横向速度
            ddd0 = 0.0  # 横向加速度
            
            # 目标状态
            s1 = ds0 * target_t + 0.5 * (target_v - ds0) * target_t  # 目标纵向位移
            ds1 = target_v  # 目标速度
            dds1 = 0.0  # 目标加速度为0
            
            d1 = target_d  # 目标横向偏移
            dd1 = 0.0  # 目标横向速度为0
            ddd1 = 0.0  # 目标横向加速度为0
            
            # 计算纵向多项式系数 (四次)
            s_coeff = self._quartic_polynomial(s0, ds0, dds0, ds1, dds1, target_t)
            
            # 计算横向多项式系数 (五次)
            d_coeff = self._quintic_polynomial(d0, dd0, ddd0, d1, dd1, ddd1, target_t)
            
            # 生成轨迹点
            s_traj = np.polyval(s_coeff[::-1], t_array)
            d_traj = np.polyval(d_coeff[::-1], t_array)
            
            # 计算速度和加速度
            ds_coeff = np.polyder(s_coeff[::-1])
            dds_coeff = np.polyder(ds_coeff)
            dd_coeff = np.polyder(d_coeff[::-1])
            ddd_coeff = np.polyder(dd_coeff)
            
            ds_traj = np.polyval(ds_coeff, t_array)
            dds_traj = np.polyval(dds_coeff, t_array)
            dd_traj = np.polyval(dd_coeff, t_array)
            ddd_traj = np.polyval(ddd_coeff, t_array)
            
            # 转换到笛卡尔坐标系
            x_traj = np.zeros(num_points)
            y_traj = np.zeros(num_points)
            theta_traj = np.zeros(num_points)
            v_traj = np.zeros(num_points)
            a_traj = np.zeros(num_points)
            kappa_traj = np.zeros(num_points)
            
            for i in range(num_points):
                # 找到参考车道上的对应点
                ref_idx = min(int(s_traj[i] / 0.5), len(lane_info.x) - 1)
                ref_idx = max(0, ref_idx)
                
                if ref_idx >= len(lane_info.x) - 1:
                    # 超出车道范围
                    return None
                
                # 参考点位置和方向
                ref_x = lane_info.x[ref_idx]
                ref_y = lane_info.y[ref_idx]
                ref_theta = lane_info.get_direction_at_index(ref_idx)
                
                # 转换到笛卡尔坐标
                x_traj[i] = ref_x + d_traj[i] * np.cos(ref_theta + np.pi/2)
                y_traj[i] = ref_y + d_traj[i] * np.sin(ref_theta + np.pi/2)
                
                # 计算航向角
                theta_traj[i] = ref_theta + np.arctan2(dd_traj[i], ds_traj[i])
                
                # 计算速度
                v_traj[i] = np.sqrt(ds_traj[i]**2 + dd_traj[i]**2)
                
                # 计算加速度
                a_traj[i] = dds_traj[i]
                
                # 计算曲率
                if i > 0:
                    dx = x_traj[i] - x_traj[i-1]
                    dy = y_traj[i] - y_traj[i-1]
                    dtheta = theta_traj[i] - theta_traj[i-1]
                    ds = np.sqrt(dx**2 + dy**2)
                    if ds > 0.001:
                        kappa_traj[i] = dtheta / ds
            
            kappa_traj[0] = kappa_traj[1] if num_points > 1 else 0.0
            
            return Trajectory(
                x=x_traj,
                y=y_traj,
                theta=theta_traj,
                v=v_traj,
                a=a_traj,
                kappa=kappa_traj,
                t=t_array
            )
            
        except Exception as e:
            return None
    
    def _quintic_polynomial(
        self,
        x0: float, dx0: float, ddx0: float,
        x1: float, dx1: float, ddx1: float,
        t: float
    ) -> np.ndarray:
        """
        计算五次多项式系数
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
            [0, 0, 2, 6*t, 12*t**2, 20*t**3]
        ])
        
        b = np.array([x0, dx0, ddx0, x1, dx1, ddx1])
        
        try:
            coeff = np.linalg.solve(A, b)
            return coeff
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            return coeff
    
    def _quartic_polynomial(
        self,
        x0: float, dx0: float, ddx0: float,
        dx1: float, ddx1: float,
        t: float
    ) -> np.ndarray:
        """
        计算四次多项式系数
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        """
        A = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 1, 2*t, 3*t**2, 4*t**3],
            [0, 0, 2, 6*t, 12*t**2]
        ])
        
        b = np.array([x0, dx0, ddx0, dx1, ddx1])
        
        try:
            coeff = np.linalg.solve(A, b)
            return coeff
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            return coeff
    
    def _check_trajectory_validity(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle]
    ) -> bool:
        """
        检查轨迹是否有效（满足运动学约束和安全约束）
        """
        # 检查加速度约束
        if np.any(trajectory.a > self.max_acceleration):
            return False
        if np.any(trajectory.a < self.max_deceleration):
            return False
        
        # 检查曲率约束
        if np.any(np.abs(trajectory.kappa) > self.max_curvature):
            return False
        
        # 检查速度约束
        if np.any(trajectory.v > self.v_max):
            return False
        if np.any(trajectory.v < 0):
            return False
        
        # 检查障碍物碰撞
        if not self._check_collision_free(trajectory, obstacles):
            return False
        
        return True
    
    def _check_collision_free(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle]
    ) -> bool:
        """
        检查轨迹是否与障碍物碰撞
        """
        for i, t in enumerate(trajectory.t):
            ego_x = trajectory.x[i]
            ego_y = trajectory.y[i]
            
            # 自车边界框
            ego_half_w = self.vehicle_width / 2
            ego_half_l = self.vehicle_length / 2
            
            for obs in obstacles:
                obs_x, obs_y = obs.get_position_at_time(t)
                obs_half_w = obs.width / 2
                obs_half_l = obs.length / 2
                
                # 简单的矩形碰撞检测
                dx = abs(ego_x - obs_x)
                dy = abs(ego_y - obs_y)
                
                min_dx = ego_half_l + obs_half_l + self.safe_distance
                min_dy = ego_half_w + obs_half_w + self.safe_distance
                
                if dx < min_dx and dy < min_dy:
                    return False
        
        return True
    
    def generate_behavior_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle],
        behavior_type: str
    ) -> List[Trajectory]:
        """
        根据行为类型生成特定轨迹
        
        Args:
            behavior_type: "keep_lane", "change_left", "change_right", "accelerate", "decelerate"
        """
        if behavior_type == "keep_lane":
            return self._generate_keep_lane_trajectories(
                current_state, lane_info, obstacles
            )
        elif behavior_type == "change_left":
            return self._generate_lane_change_trajectories(
                current_state, lane_info, obstacles, direction="left"
            )
        elif behavior_type == "change_right":
            return self._generate_lane_change_trajectories(
                current_state, lane_info, obstacles, direction="right"
            )
        elif behavior_type == "accelerate":
            return self._generate_accelerate_trajectories(
                current_state, lane_info, obstacles
            )
        elif behavior_type == "decelerate":
            return self._generate_decelerate_trajectories(
                current_state, lane_info, obstacles
            )
        else:
            return self.generate_trajectories(current_state, lane_info, obstacles)
    
    def _generate_keep_lane_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle]
    ) -> List[Trajectory]:
        """生成车道保持轨迹"""
        # 限制横向偏移在车道中心附近
        original_d_min = self.d_min
        original_d_max = self.d_max
        
        self.d_min = -0.5
        self.d_max = 0.5
        
        trajectories = self.generate_trajectories(current_state, lane_info, obstacles)
        
        # 恢复参数
        self.d_min = original_d_min
        self.d_max = original_d_max
        
        for traj in trajectories:
            traj.behavior_type = "keep_lane"
        
        return trajectories
    
    def _generate_lane_change_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle],
        direction: str
    ) -> List[Trajectory]:
        """生成换道轨迹"""
        lane_width = lane_info.width
        
        original_d_min = self.d_min
        original_d_max = self.d_max
        
        if direction == "left":
            self.d_min = lane_width * 0.8
            self.d_max = lane_width * 1.2
        else:  # right
            self.d_min = -lane_width * 1.2
            self.d_max = -lane_width * 0.8
        
        trajectories = self.generate_trajectories(current_state, lane_info, obstacles)
        
        self.d_min = original_d_min
        self.d_max = original_d_max
        
        for traj in trajectories:
            traj.behavior_type = f"change_{direction}"
        
        return trajectories
    
    def _generate_accelerate_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle]
    ) -> List[Trajectory]:
        """生成加速轨迹"""
        target_speed = min(current_state.v + 5, self.v_max)
        trajectories = self.generate_trajectories(
            current_state, lane_info, obstacles, target_speed
        )
        
        for traj in trajectories:
            traj.behavior_type = "accelerate"
        
        return trajectories
    
    def _generate_decelerate_trajectories(
        self,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle]
    ) -> List[Trajectory]:
        """生成减速轨迹"""
        target_speed = max(current_state.v - 5, 0)
        trajectories = self.generate_trajectories(
            current_state, lane_info, obstacles, target_speed
        )
        
        for traj in trajectories:
            traj.behavior_type = "decelerate"
        
        return trajectories
