"""
数据管理器模块

管理自动驾驶仿真数据，包括：
- 自车状态
- 障碍物信息
- 车道线信息
- 规划轨迹
- 传感器数据
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class BehaviorType(Enum):
    """行为决策类型"""
    KEEP_LANE = "保持车道"
    CHANGE_LEFT = "向左变道"
    CHANGE_RIGHT = "向右变道"
    FOLLOW = "跟随前车"
    STOP = "停车"
    EMERGENCY_STOP = "紧急停车"
    TURN_LEFT = "左转"
    TURN_RIGHT = "右转"
    UNKNOWN = "未知"


@dataclass
class EgoState:
    """自车状态"""
    x: float = 0.0  # 位置x (米)
    y: float = 0.0  # 位置y (米)
    z: float = 0.0  # 位置z (米)
    heading: float = 0.0  # 航向角 (弧度)
    velocity: float = 0.0  # 速度 (m/s)
    acceleration: float = 0.0  # 加速度 (m/s²)
    steering: float = 0.0  # 方向盘转角 (弧度)
    timestamp: float = 0.0  # 时间戳
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z, self.heading, 
                        self.velocity, self.acceleration, self.steering])


@dataclass
class Obstacle:
    """障碍物信息"""
    id: int
    obstacle_type: str  # 'vehicle', 'pedestrian', 'cyclist', 'unknown'
    x: float
    y: float
    z: float = 0.0
    length: float = 4.5
    width: float = 1.8
    height: float = 1.5
    heading: float = 0.0
    velocity: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0
    
    def get_corners(self) -> np.ndarray:
        """获取3D边界框角点 (8x3)"""
        l, w, h = self.length, self.width, self.height
        # 局部坐标系下的角点
        corners = np.array([
            [l/2, w/2, h/2], [l/2, -w/2, h/2], [-l/2, -w/2, h/2], [-l/2, w/2, h/2],
            [l/2, w/2, 0], [l/2, -w/2, 0], [-l/2, -w/2, 0], [-l/2, w/2, 0]
        ])
        
        # 旋转矩阵
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        rotation = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        # 变换到世界坐标系
        corners = corners @ rotation.T
        corners += np.array([self.x, self.y, self.z])
        return corners
    
    def get_bbox_2d(self) -> Tuple[float, float, float, float]:
        """获取2D边界框 (x_min, y_min, x_max, y_max)"""
        cos_h = np.abs(np.cos(self.heading))
        sin_h = np.abs(np.sin(self.heading))
        dx = self.length * cos_h + self.width * sin_h
        dy = self.length * sin_h + self.width * cos_h
        return (self.x - dx/2, self.y - dy/2, self.x + dx/2, self.y + dy/2)


@dataclass
class LaneLine:
    """车道线信息"""
    id: int
    line_type: str  # 'solid', 'dashed', 'double', 'curb'
    color: str  # 'white', 'yellow'
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    confidence: float = 1.0
    
    def __post_init__(self):
        if isinstance(self.points, list):
            self.points = np.array(self.points)


@dataclass
class Trajectory:
    """轨迹信息"""
    points: np.ndarray  # Nx3 或 Nx4 (x, y, z, [heading])
    velocities: Optional[np.ndarray] = None  # N
    timestamps: Optional[np.ndarray] = None  # N
    trajectory_type: str = 'candidate'  # 'candidate', 'selected', 'reference'
    cost: float = 0.0
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    def __post_init__(self):
        if isinstance(self.points, list):
            self.points = np.array(self.points)
        if self.velocities is not None and isinstance(self.velocities, list):
            self.velocities = np.array(self.velocities)
        if self.timestamps is not None and isinstance(self.timestamps, list):
            self.timestamps = np.array(self.timestamps)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (
            np.array_equal(self.points, other.points) and
            np.array_equal(self.velocities, other.velocities) and
            np.array_equal(self.timestamps, other.timestamps) and
            self.trajectory_type == other.trajectory_type and
            self.cost == other.cost and
            self.color == other.color
        )
    
    def __hash__(self) -> int:
        return id(self)


@dataclass
class PlanningResult:
    """规划结果"""
    selected_trajectory: Optional[Trajectory] = None
    candidate_trajectories: List[Trajectory] = field(default_factory=list)
    behavior: BehaviorType = BehaviorType.UNKNOWN
    behavior_description: str = ""
    target_speed: float = 0.0
    target_lane: int = 0
    timestamp: float = 0.0


@dataclass
class OccupancyGrid:
    """Occupancy网格"""
    data: np.ndarray  # HxW 二值或概率值
    resolution: float = 0.2  # 米/像素
    origin_x: float = -50.0  # 左下角x
    origin_y: float = -50.0  # 左下角y
    height: float = 100.0
    width: float = 100.0
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x = grid_x * self.resolution + self.origin_x
        y = grid_y * self.resolution + self.origin_y
        return x, y


@dataclass
class SensorData:
    """传感器数据"""
    # 相机图像
    camera_images: Dict[str, np.ndarray] = field(default_factory=dict)
    # LiDAR点云 (Nx4: x, y, z, intensity)
    lidar_points: Optional[np.ndarray] = None
    # 雷达数据
    radar_data: Optional[np.ndarray] = None
    timestamp: float = 0.0


class DataManager:
    """
    数据管理器
    
    管理一帧或多帧的自动驾驶数据
    """
    
    def __init__(self):
        self.frames: List[Dict] = []
        self.current_frame_idx: int = 0
        
        # 当前帧数据
        self.ego_state: Optional[EgoState] = None
        self.obstacles: List[Obstacle] = []
        self.lane_lines: List[LaneLine] = []
        self.planning_result: Optional[PlanningResult] = None
        self.occupancy_grid: Optional[OccupancyGrid] = None
        self.sensor_data: Optional[SensorData] = None
        
        # 扩展可视化数据
        self.annotated_image: Optional[np.ndarray] = None
        self.multi_camera_image: Optional[np.ndarray] = None
        self.trajectory_analysis_image: Optional[np.ndarray] = None
        
    def add_frame(self, frame_data: Dict):
        """添加一帧数据"""
        self.frames.append(frame_data)
        
    def load_frame(self, frame_idx: int) -> bool:
        """加载指定帧的数据"""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        
        self.current_frame_idx = frame_idx
        frame = self.frames[frame_idx]
        
        # 加载自车状态
        if 'ego_state' in frame:
            self.ego_state = frame['ego_state']
        
        # 加载障碍物
        if 'obstacles' in frame:
            self.obstacles = frame['obstacles']
        
        # 加载车道线
        if 'lane_lines' in frame:
            self.lane_lines = frame['lane_lines']
        
        # 加载规划结果
        if 'planning_result' in frame:
            self.planning_result = frame['planning_result']
        
        # 加载Occupancy网格
        if 'occupancy_grid' in frame:
            self.occupancy_grid = frame['occupancy_grid']
        
        # 加载传感器数据
        if 'sensor_data' in frame:
            self.sensor_data = frame['sensor_data']
        
        # 加载扩展可视化数据
        if 'annotated_image' in frame:
            self.annotated_image = frame['annotated_image']
        if 'multi_camera_image' in frame:
            self.multi_camera_image = frame['multi_camera_image']
        if 'trajectory_analysis_image' in frame:
            self.trajectory_analysis_image = frame['trajectory_analysis_image']
        
        return True
    
    def set_live_frame(self, frame_data: Dict):
        """直接设置当前帧数据（用于实时流）"""
        if 'ego_state' in frame_data:
            self.ego_state = frame_data['ego_state']
        if 'obstacles' in frame_data:
            self.obstacles = frame_data['obstacles']
        if 'lane_lines' in frame_data:
            self.lane_lines = frame_data['lane_lines']
        if 'planning_result' in frame_data:
            self.planning_result = frame_data['planning_result']
        if 'occupancy_grid' in frame_data:
            self.occupancy_grid = frame_data['occupancy_grid']
        if 'sensor_data' in frame_data:
            self.sensor_data = frame_data['sensor_data']
        if 'annotated_image' in frame_data:
            self.annotated_image = frame_data['annotated_image']
        if 'multi_camera_image' in frame_data:
            self.multi_camera_image = frame_data['multi_camera_image']
        if 'trajectory_analysis_image' in frame_data:
            self.trajectory_analysis_image = frame_data['trajectory_analysis_image']
    
    def next_frame(self) -> bool:
        """加载下一帧"""
        return self.load_frame(self.current_frame_idx + 1)
    
    def prev_frame(self) -> bool:
        """加载上一帧"""
        return self.load_frame(self.current_frame_idx - 1)
    
    def get_frame_count(self) -> int:
        """获取总帧数"""
        return len(self.frames)
    
    def clear(self):
        """清除所有数据"""
        self.frames.clear()
        self.current_frame_idx = 0
        self.ego_state = None
        self.obstacles = []
        self.lane_lines = []
        self.planning_result = None
        self.occupancy_grid = None
        self.sensor_data = None
    
    def generate_demo_data(self, num_frames: int = 100):
        """生成演示数据"""
        self.clear()
        
        for i in range(num_frames):
            t = i * 0.1  # 时间步长0.1s
            
            # 自车状态 - 沿S形路径行驶
            ego = EgoState(
                x=t * 5,
                y=np.sin(t * 0.5) * 5,
                z=0.0,
                heading=np.arctan2(np.cos(t * 0.5) * 2.5, 5),
                velocity=5.0 + np.sin(t) * 2,
                acceleration=0.0,
                steering=0.0,
                timestamp=t
            )
            
            # 障碍物
            obstacles = []
            # 前车
            obstacles.append(Obstacle(
                id=1,
                obstacle_type='vehicle',
                x=ego.x + 20,
                y=ego.y + 2,
                z=0,
                length=4.5,
                width=1.8,
                height=1.5,
                heading=ego.heading,
                velocity=4.0,
                confidence=0.95
            ))
            # 对向车辆
            obstacles.append(Obstacle(
                id=2,
                obstacle_type='vehicle',
                x=ego.x - 30,
                y=ego.y - 2,
                z=0,
                length=4.5,
                width=1.8,
                height=1.5,
                heading=ego.heading + np.pi,
                velocity=6.0,
                confidence=0.90
            ))
            # 行人
            if i > 30 and i < 70:
                obstacles.append(Obstacle(
                    id=3,
                    obstacle_type='pedestrian',
                    x=ego.x + 15 + (i - 30) * 0.1,
                    y=ego.y + 4,
                    z=0,
                    length=0.5,
                    width=0.5,
                    height=1.7,
                    heading=0,
                    velocity=1.0,
                    confidence=0.85
                ))
            
            # 车道线
            lane_lines = []
            # 左车道线
            left_points = []
            for j in range(-50, 100, 5):
                left_points.append([ego.x + j, ego.y + np.sin((ego.x + j) * 0.1) * 5 - 3.5, 0])
            lane_lines.append(LaneLine(
                id=1,
                line_type='solid',
                color='white',
                points=np.array(left_points),
                confidence=0.95
            ))
            # 右车道线
            right_points = []
            for j in range(-50, 100, 5):
                right_points.append([ego.x + j, ego.y + np.sin((ego.x + j) * 0.1) * 5 + 3.5, 0])
            lane_lines.append(LaneLine(
                id=2,
                line_type='solid',
                color='white',
                points=np.array(right_points),
                confidence=0.95
            ))
            # 中心线
            center_points = []
            for j in range(-50, 100, 5):
                center_points.append([ego.x + j, ego.y + np.sin((ego.x + j) * 0.1) * 5, 0])
            lane_lines.append(LaneLine(
                id=3,
                line_type='dashed',
                color='yellow',
                points=np.array(center_points),
                confidence=0.90
            ))
            
            # 规划轨迹
            candidate_trajs = []
            # 生成多条候选轨迹
            for offset in [-1.5, 0, 1.5]:
                traj_points = []
                for j in range(50):
                    s = j * 0.5
                    traj_points.append([
                        ego.x + s * np.cos(ego.heading),
                        ego.y + s * np.sin(ego.heading) + offset * (1 - np.exp(-s/10)),
                        0
                    ])
                traj = Trajectory(
                    points=np.array(traj_points),
                    trajectory_type='candidate',
                    cost=abs(offset) * 0.5,
                    color=(0.5, 0.5, 0.5) if offset != 0 else (0, 1, 0)
                )
                candidate_trajs.append(traj)
            
            selected_traj = candidate_trajs[1]  # 选择中间轨迹
            selected_traj.trajectory_type = 'selected'
            selected_traj.color = (0, 1, 0)
            
            planning_result = PlanningResult(
                selected_trajectory=selected_traj,
                candidate_trajectories=candidate_trajs,
                behavior=BehaviorType.KEEP_LANE if i < 40 else BehaviorType.FOLLOW,
                behavior_description="保持车道行驶" if i < 40 else "跟随前车",
                target_speed=5.0,
                target_lane=0,
                timestamp=t
            )
            
            # Occupancy网格
            grid_size = 200
            occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)
            for obs in obstacles:
                gx, gy = int((obs.x - ego.x + 50) / 0.5), int((obs.y - ego.y + 50) / 0.5)
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    cv = int(obs.length / 0.5)
                    occupancy[max(0, gy-cv):min(grid_size, gy+cv), 
                             max(0, gx-cv):min(grid_size, gx+cv)] = 1.0
            
            occupancy_grid = OccupancyGrid(
                data=occupancy,
                resolution=0.5,
                origin_x=ego.x - 50,
                origin_y=ego.y - 50,
                height=100.0,
                width=100.0
            )
            
            # 传感器数据
            sensor_data = SensorData(
                camera_images={},
                lidar_points=self._generate_lidar_points(obstacles, ego),
                timestamp=t
            )
            
            frame = {
                'ego_state': ego,
                'obstacles': obstacles,
                'lane_lines': lane_lines,
                'planning_result': planning_result,
                'occupancy_grid': occupancy_grid,
                'sensor_data': sensor_data
            }
            
            self.add_frame(frame)
        
        self.load_frame(0)
    
    def _generate_lidar_points(self, obstacles: List[Obstacle], 
                               ego: EgoState, num_points: int = 1000) -> np.ndarray:
        """生成模拟LiDAR点云"""
        points = []
        
        # 从障碍物生成点
        for obs in obstacles:
            n = int(num_points / len(obstacles)) if obstacles else 0
            for _ in range(n):
                # 在障碍物表面随机采样
                face = np.random.randint(0, 6)
                u = np.random.uniform(-0.5, 0.5)
                v = np.random.uniform(-0.5, 0.5)
                
                if face == 0:  # 前面
                    local = np.array([obs.length/2, u*obs.width, v*obs.height])
                elif face == 1:  # 后面
                    local = np.array([-obs.length/2, u*obs.width, v*obs.height])
                elif face == 2:  # 左面
                    local = np.array([u*obs.length, obs.width/2, v*obs.height])
                elif face == 3:  # 右面
                    local = np.array([u*obs.length, -obs.width/2, v*obs.height])
                elif face == 4:  # 上面
                    local = np.array([u*obs.length, v*obs.width, obs.height])
                else:  # 下面
                    local = np.array([u*obs.length, v*obs.width, 0])
                
                # 变换到世界坐标
                cos_h = np.cos(obs.heading)
                sin_h = np.sin(obs.heading)
                world = np.array([
                    local[0] * cos_h - local[1] * sin_h + obs.x,
                    local[0] * sin_h + local[1] * cos_h + obs.y,
                    local[2] + obs.z,
                    np.random.uniform(0.5, 1.0)  # intensity
                ])
                points.append(world)
        
        # 添加地面点
        for _ in range(200):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(5, 50)
            px = ego.x + dist * np.cos(angle)
            py = ego.y + dist * np.sin(angle)
            pz = 0
            points.append([px, py, pz, 0.3])
        
        return np.array(points) if points else np.zeros((0, 4))
