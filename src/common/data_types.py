"""
自动驾驶系统 - 统一数据结构定义
定义所有模块间共享的数据类型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum, auto
import numpy as np
from datetime import datetime


@dataclass
class Timestamp:
    """高精度时间戳"""
    seconds: float = 0.0  # Unix时间戳（秒）
    nanoseconds: int = 0  # 纳秒部分
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """获取当前时间戳"""
        now = datetime.now().timestamp()
        sec = int(now)
        nsec = int((now - sec) * 1e9)
        return cls(seconds=sec, nanoseconds=nsec)
    
    def to_seconds(self) -> float:
        """转换为秒"""
        return self.seconds + self.nanoseconds / 1e9
    
    def __sub__(self, other: 'Timestamp') -> float:
        """计算时间差（秒）"""
        return self.to_seconds() - other.to_seconds()


@dataclass
class Vector3D:
    """三维向量"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class Quaternion:
    """四元数表示旋转"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def to_euler(self) -> Tuple[float, float, float]:
        """转换为欧拉角 (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw


@dataclass
class Pose:
    """位姿：位置+方向"""
    position: Vector3D = field(default_factory=Vector3D)
    orientation: Quaternion = field(default_factory=Quaternion)
    
    @property
    def x(self) -> float:
        return self.position.x
    
    @property
    def y(self) -> float:
        return self.position.y
    
    @property
    def z(self) -> float:
        return self.position.z


@dataclass
class BoundingBox3D:
    """3D边界框"""
    center: Pose = field(default_factory=Pose)
    size: Vector3D = field(default_factory=Vector3D)  # 长、宽、高
    velocity: Vector3D = field(default_factory=Vector3D)
    
    @property
    def length(self) -> float:
        return self.size.x
    
    @property
    def width(self) -> float:
        return self.size.y
    
    @property
    def height(self) -> float:
        return self.size.z


# ==================== 传感器数据类型 ====================

@dataclass
class ImageData:
    """摄像头图像数据"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    camera_id: str = ""
    image: Optional[np.ndarray] = None  # H x W x C
    width: int = 0
    height: int = 0
    channels: int = 3
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))  # 相机内参
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))  # 相机外参
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))  # 畸变系数
    
    def get_camera_matrix(self) -> np.ndarray:
        """获取相机投影矩阵"""
        return self.intrinsics @ self.extrinsics[:3, :]


@dataclass
class PointCloud:
    """LiDAR点云数据"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    lidar_id: str = ""
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # N x 4 (x, y, z, intensity)
    rings: Optional[np.ndarray] = None  # 每个点的线束编号
    timestamps: Optional[np.ndarray] = None  # 每个点的时间戳
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))  # LiDAR外参
    
    @property
    def num_points(self) -> int:
        return len(self.points)
    
    def get_xyz(self) -> np.ndarray:
        """获取XYZ坐标"""
        return self.points[:, :3]


@dataclass
class RadarTarget:
    """单个雷达目标"""
    range_distance: float = 0.0  # 距离
    azimuth: float = 0.0  # 方位角
    elevation: float = 0.0  # 俯仰角
    velocity: float = 0.0  # 径向速度
    rcs: float = 0.0  # 雷达散射截面
    snr: float = 0.0  # 信噪比


@dataclass
class RadarData:
    """雷达数据"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    radar_id: str = ""
    targets: List[RadarTarget] = field(default_factory=list)
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class UltrasonicData:
    """超声波传感器数据"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    sensor_id: str = ""
    distance: float = 0.0  # 检测距离
    confidence: float = 0.0  # 置信度
    is_valid: bool = False  # 是否有效
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class CANData:
    """CAN总线数据"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    can_id: int = 0
    can_channel: str = ""  # CAN通道
    data: bytes = field(default_factory=bytes)
    dlc: int = 8  # 数据长度
    is_extended: bool = False  # 是否为扩展帧
    
    # 车辆状态（解析后的数据）
    vehicle_speed: float = 0.0  # 车速 (m/s)
    steering_angle: float = 0.0  # 方向盘角度 (度)
    yaw_rate: float = 0.0  # 横摆角速度 (rad/s)
    longitudinal_accel: float = 0.0  # 纵向加速度 (m/s^2)
    lateral_accel: float = 0.0  # 横向加速度 (m/s^2)
    gear_position: int = 0  # 档位
    turn_signal: int = 0  # 转向灯状态


# ==================== 感知结果数据类型 ====================

@dataclass
class LaneLine:
    """单条车道线"""
    line_id: int = 0
    line_type: str = "unknown"  # solid, dashed, double, etc.
    color: str = "white"  # white, yellow, blue, etc.
    confidence: float = 0.0
    
    # 车道线点（车身坐标系）
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    
    # 多项式拟合系数 (c0 + c1*x + c2*x^2 + c3*x^3)
    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # 车道线起点和终点
    start_point: Vector3D = field(default_factory=Vector3D)
    end_point: Vector3D = field(default_factory=Vector3D)
    
    # 曲率
    curvature: float = 0.0
    curvature_rate: float = 0.0


@dataclass
class LaneDetectionResult:
    """车道线检测结果"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    lane_lines: List[LaneLine] = field(default_factory=list)
    
    # 车道信息
    ego_lane_id: int = -1  # 自车所在车道
    num_lanes: int = 0  # 检测到的车道数
    
    # 车道宽度
    left_lane_width: float = 0.0
    right_lane_width: float = 0.0
    
    # 车道偏离警告
    lane_departure_warning: bool = False
    departure_direction: str = ""  # left, right


@dataclass
class Obstacle:
    """障碍物/目标物"""
    obstacle_id: int = 0
    obstacle_type: str = "unknown"  # vehicle, pedestrian, cyclist, etc.
    obstacle_sub_type: str = ""  # 子类型
    
    # 3D边界框
    bbox: BoundingBox3D = field(default_factory=BoundingBox3D)
    
    # 跟踪信息
    tracking_id: int = -1
    tracking_age: int = 0
    confidence: float = 0.0
    
    # 运动信息
    velocity: Vector3D = field(default_factory=Vector3D)
    acceleration: Vector3D = field(default_factory=Vector3D)
    
    # 预测轨迹
    predicted_trajectory: Optional['Trajectory'] = None
    
    # 其他属性
    is_moving: bool = False
    is_occluded: bool = False
    is_static: bool = False
    
    # 行人姿态（仅当obstacle_type为pedestrian时有效）
    # 18个关键点，每个点包含(x, y, z, confidence)
    pedestrian_pose: Optional[np.ndarray] = None  # [18, 4]
    
    # 车辆灯态（仅当obstacle_type为vehicle时有效）
    # 包含：left_turn_signal, right_turn_signal, brake_light, headlight
    vehicle_light_status: Optional[Dict[str, str]] = None  # e.g., {'left_turn': 'off', 'brake': 'on'}


@dataclass
class ObstacleDetectionResult:
    """障碍物检测结果"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    obstacles: List[Obstacle] = field(default_factory=list)
    
    # 统计信息
    num_vehicles: int = 0
    num_pedestrians: int = 0
    num_cyclists: int = 0
    num_unknown: int = 0


@dataclass
class OccupancyCell:
    """占据栅格单元"""
    occupied: bool = False
    occupancy_prob: float = 0.0
    height: float = 0.0  # 占据高度
    semantic_class: int = 0  # 语义类别


@dataclass
class OccupancyGrid:
    """占据栅格"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    
    # 栅格参数
    resolution: float = 0.2  # 米/格
    width: int = 200  # 栅格宽度
    height: int = 200  # 栅格高度
    
    # 栅格原点（车辆坐标系）
    origin_x: float = -20.0  # 车辆后方20米
    origin_y: float = -20.0  # 车辆左侧20米
    
    # 占据数据 [H, W, C]
    data: np.ndarray = field(default_factory=lambda: np.zeros((200, 200, 1)))
    
    # 高度数据
    height_data: Optional[np.ndarray] = None
    
    # 语义数据
    semantic_data: Optional[np.ndarray] = None


@dataclass
class OccupancyResult:
    """占据网络输出结果"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    
    # 多尺度占据栅格
    occupancy_2d: Optional[OccupancyGrid] = None  # 2D鸟瞰图
    occupancy_3d: Optional[np.ndarray] = None  # 3D体素 [X, Y, Z, C]
    
    # 占据范围
    range_x: Tuple[float, float] = (-50.0, 50.0)  # 前后范围
    range_y: Tuple[float, float] = (-30.0, 30.0)  # 左右范围
    range_z: Tuple[float, float] = (-3.0, 3.0)  # 上下范围
    
    # 体素分辨率
    voxel_resolution: float = 0.5


@dataclass
class TrafficSign:
    """交通标识"""
    sign_id: int = 0
    sign_type: str = ""  # 标识类型：speed_limit, stop, yield, no_entry, etc.
    meaning: str = ""  # 标识含义描述
    
    # 位置信息（3D坐标）
    position: Vector3D = field(default_factory=Vector3D)
    
    # 边界框
    bbox: BoundingBox3D = field(default_factory=BoundingBox3D)
    
    # 置信度
    confidence: float = 0.0
    
    # 数值信息（如限速值）
    value: Optional[float] = None  # 例如：限速60km/h -> 60
    
    # 单位
    unit: str = ""  # 例如：km/h, m, etc.
    
    # 可见性
    is_visible: bool = True
    occlusion_level: float = 0.0  # 遮挡程度 0-1


@dataclass
class TrafficLight:
    """交通灯"""
    light_id: int = 0
    
    # 位置信息（3D坐标）
    position: Vector3D = field(default_factory=Vector3D)
    
    # 边界框
    bbox: BoundingBox3D = field(default_factory=BoundingBox3D)
    
    # 颜色状态
    state: str = "unknown"  # red, yellow, green, flashing_red, flashing_yellow, off
    
    # 指示方向
    direction: str = "unknown"  # straight, left, right, uturn, all
    
    # 置信度
    confidence: float = 0.0
    
    # 倒计时（如果有）
    countdown_seconds: Optional[float] = None
    
    # 是否可见
    is_visible: bool = True
    
    # 关联的车道
    associated_lane_ids: List[int] = field(default_factory=list)


@dataclass
class PerceptionResult:
    """感知模块综合输出"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    
    # 各子模块结果
    lane_result: Optional[LaneDetectionResult] = None
    obstacle_result: Optional[ObstacleDetectionResult] = None
    occupancy_result: Optional[OccupancyResult] = None
    
    # 交通标识和交通灯
    traffic_signs: List[TrafficSign] = field(default_factory=list)
    traffic_lights: List[TrafficLight] = field(default_factory=list)
    
    # 自车状态
    ego_pose: Pose = field(default_factory=Pose)
    ego_velocity: Vector3D = field(default_factory=Vector3D)
    
    # 处理时间
    processing_time_ms: float = 0.0


# ==================== 规划结果数据类型 ====================

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    relative_time: float = 0.0  # 相对于当前时间的偏移（秒）
    
    # 位置
    pose: Pose = field(default_factory=Pose)
    
    # 速度
    longitudinal_velocity: float = 0.0  # 纵向速度 (m/s)
    lateral_velocity: float = 0.0  # 横向速度 (m/s)
    
    # 加速度
    longitudinal_acceleration: float = 0.0  # 纵向加速度 (m/s^2)
    lateral_acceleration: float = 0.0  # 横向加速度 (m/s^2)
    
    # 曲率
    curvature: float = 0.0  # 曲率 (1/m)
    
    # 航向角
    theta: float = 0.0  # 相对于车身的航向角 (rad)


@dataclass
class Trajectory:
    """轨迹"""
    trajectory_id: int = 0
    trajectory_type: str = "normal"  # normal, emergency, fallback
    
    points: List[TrajectoryPoint] = field(default_factory=list)
    
    # 轨迹属性
    total_time: float = 0.0  # 总时长（秒）
    total_length: float = 0.0  # 总长度（米）
    
    # 代价
    cost: float = 0.0
    safety_cost: float = 0.0
    comfort_cost: float = 0.0
    efficiency_cost: float = 0.0
    
    # 可行性
    is_feasible: bool = True
    infeasible_reason: str = ""
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (
            self.trajectory_id == other.trajectory_id and
            self.trajectory_type == other.trajectory_type and
            self.points == other.points and
            self.total_time == other.total_time and
            self.total_length == other.total_length and
            self.cost == other.cost and
            self.safety_cost == other.safety_cost and
            self.comfort_cost == other.comfort_cost and
            self.efficiency_cost == other.efficiency_cost and
            self.is_feasible == other.is_feasible and
            self.infeasible_reason == other.infeasible_reason
        )
    
    def __hash__(self) -> int:
        return id(self)
    
    def get_point_at_time(self, t: float) -> Optional[TrajectoryPoint]:
        """获取指定时间的轨迹点"""
        if not self.points:
            return None
        for i, point in enumerate(self.points):
            if point.relative_time >= t:
                return self.points[max(0, i-1)]
        return self.points[-1]


@dataclass
class PlanningResult:
    """规划模块输出"""
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    
    # 行为决策
    behavior_type: str = "cruise"  # cruise, follow, overtake, stop, etc.
    behavior_explanation: str = ""  # 行为解释
    
    # 选中的轨迹
    selected_trajectory: Optional[Trajectory] = None
    
    # 候选轨迹
    candidate_trajectories: List[Trajectory] = field(default_factory=list)
    
    # 目标速度
    target_speed: float = 0.0
    
    # 目标车道
    target_lane_id: int = -1
    
    # 控制指令
    steering_angle: float = 0.0  # 方向盘角度
    throttle: float = 0.0  # 油门
    brake: float = 0.0  # 刹车
    
    # 处理时间
    processing_time_ms: float = 0.0
    
    # 是否有效
    is_valid: bool = True
    failure_reason: str = ""


# ==================== 配置数据类型 ====================

@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_id: str = ""
    sensor_type: str = ""
    enabled: bool = True
    
    # 硬件参数
    interface: str = ""  # 接口类型
    device_path: str = ""  # 设备路径
    ip_address: str = ""  # IP地址（网络设备）
    port: int = 0
    
    # 外参
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # 内参（相机）
    intrinsics: Optional[np.ndarray] = None
    distortion: Optional[np.ndarray] = None
    
    # 其他参数
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleConfig:
    """模块配置"""
    module_name: str = ""
    enabled: bool = True
    
    # 执行参数
    execution_frequency: float = 10.0  # Hz
    timeout_ms: float = 100.0
    
    # 算法参数
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # 输入输出配置
    input_topics: List[str] = field(default_factory=list)
    output_topics: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """系统配置"""
    system_name: str = "AutonomousDrivingSystem"
    version: str = "1.0.0"
    
    # 日志配置
    log_level: str = "INFO"
    log_path: str = "./logs"
    log_to_console: bool = True
    log_to_file: bool = True
    
    # 仿真配置
    simulation_mode: bool = False
    simulation_dataset: str = ""
    simulation_speed: float = 1.0
    
    # 传感器配置
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    
    # 模块配置
    modules: Dict[str, ModuleConfig] = field(default_factory=dict)
    
    # 其他参数
    parameters: Dict[str, Any] = field(default_factory=dict)
