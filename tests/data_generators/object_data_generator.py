"""
目标数据生成器
生成模拟的车辆、行人、自行车等目标数据
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid


class ObjectType(Enum):
    """目标类型枚举"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    UNKNOWN = "unknown"


class ObjectStatus(Enum):
    """目标状态枚举"""
    MOVING = "moving"
    STATIONARY = "stationary"
    STOPPED = "stopped"
    PARKED = "parked"


@dataclass
class BoundingBox3D:
    """3D边界框"""
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    heading: float  # 朝向角（弧度）
    
    def get_corners(self) -> np.ndarray:
        """获取边界框的8个角点"""
        # 局部坐标系下的角点
        l, w, h = self.length / 2, self.width / 2, self.height / 2
        corners = np.array([
            [-l, -w, -h], [-l, -w, h], [-l, w, h], [-l, w, -h],
            [l, -w, -h], [l, -w, h], [l, w, h], [l, w, -h]
        ])
        
        # 旋转矩阵
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        rotation = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        # 旋转并平移
        corners = corners @ rotation.T
        corners += np.array([self.center_x, self.center_y, self.center_z])
        
        return corners


@dataclass
class TrackedObject:
    """跟踪目标"""
    id: str
    object_type: ObjectType
    status: ObjectStatus
    bbox: BoundingBox3D
    velocity: np.ndarray  # (3,) m/s
    acceleration: np.ndarray  # (3,) m/s^2
    timestamp: float
    confidence: float
    
    # 额外属性
    age: int = 0  # 跟踪帧数
    lost_count: int = 0  # 丢失帧数
    history: List[np.ndarray] = None  # 历史位置
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def predict_next_position(self, dt: float = 0.1) -> np.ndarray:
        """预测下一时刻位置"""
        current_pos = np.array([
            self.bbox.center_x,
            self.bbox.center_y,
            self.bbox.center_z
        ])
        return current_pos + self.velocity * dt + 0.5 * self.acceleration * dt ** 2


@dataclass
class ObjectDetection:
    """目标检测结果"""
    object_type: ObjectType
    bbox: BoundingBox3D
    confidence: float
    timestamp: float
    source_sensor: str


class ObjectDataGenerator:
    """目标数据生成器"""
    
    # 各类型的默认尺寸 (长, 宽, 高) 单位: 米
    DEFAULT_DIMENSIONS = {
        ObjectType.VEHICLE: (4.5, 1.8, 1.5),
        ObjectType.PEDESTRIAN: (0.5, 0.5, 1.7),
        ObjectType.CYCLIST: (1.8, 0.6, 1.5),
        ObjectType.MOTORCYCLE: (2.0, 0.8, 1.2),
        ObjectType.TRUCK: (10.0, 2.5, 3.5),
        ObjectType.BUS: (12.0, 2.5, 3.0),
        ObjectType.UNKNOWN: (2.0, 1.0, 1.5),
    }
    
    def __init__(self, seed: int = 42):
        """
        初始化目标数据生成器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.RandomState(seed)
        
    def generate_single_object(
        self,
        object_type: ObjectType = ObjectType.VEHICLE,
        position: Optional[Tuple[float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        timestamp: float = 0.0
    ) -> TrackedObject:
        """
        生成单个目标
        
        Args:
            object_type: 目标类型
            position: 位置 (x, y, z)，随机生成如果为None
            velocity: 速度 (vx, vy, vz)，随机生成如果为None
            timestamp: 时间戳
            
        Returns:
            TrackedObject对象
        """
        # 生成位置
        if position is None:
            x = self.rng.uniform(-50, 50)
            y = self.rng.uniform(-30, 30)
            z = self.rng.uniform(0, 2)
        else:
            x, y, z = position
        
        # 生成速度
        if velocity is None:
            vx = self.rng.uniform(-15, 15)
            vy = self.rng.uniform(-15, 15)
            vz = 0.0
        else:
            vx, vy, vz = velocity
        
        # 获取默认尺寸
        length, width, height = self.DEFAULT_DIMENSIONS[object_type]
        
        # 添加尺寸变化
        length *= self.rng.uniform(0.9, 1.1)
        width *= self.rng.uniform(0.9, 1.1)
        height *= self.rng.uniform(0.9, 1.1)
        
        # 计算朝向（根据速度方向）
        heading = np.arctan2(vy, vx)
        
        # 创建边界框
        bbox = BoundingBox3D(
            center_x=x,
            center_y=y,
            center_z=z + height / 2,
            length=length,
            width=width,
            height=height,
            heading=heading
        )
        
        # 确定状态
        speed = np.sqrt(vx**2 + vy**2)
        if speed < 0.5:
            status = ObjectStatus.STATIONARY
        else:
            status = ObjectStatus.MOVING
        
        return TrackedObject(
            id=str(uuid.uuid4())[:8],
            object_type=object_type,
            status=status,
            bbox=bbox,
            velocity=np.array([vx, vy, vz]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            timestamp=timestamp,
            confidence=self.rng.uniform(0.7, 0.99)
        )
    
    def generate_object_set(
        self,
        num_objects: int = 10,
        object_types: Optional[List[ObjectType]] = None,
        ego_position: Tuple[float, float, float] = (0, 0, 0),
        detection_range: float = 100.0,
        timestamp: float = 0.0
    ) -> List[TrackedObject]:
        """
        生成一组目标
        
        Args:
            num_objects: 目标数量
            object_types: 目标类型列表，随机选择如果为None
            ego_position: 自车位置
            detection_range: 检测范围
            timestamp: 时间戳
            
        Returns:
            TrackedObject列表
        """
        if object_types is None:
            object_types = [
                ObjectType.VEHICLE,
                ObjectType.PEDESTRIAN,
                ObjectType.CYCLIST,
                ObjectType.TRUCK
            ]
        
        objects = []
        
        for _ in range(num_objects):
            obj_type = self.rng.choice(object_types)
            
            # 在检测范围内随机生成位置
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = self.rng.uniform(5, detection_range)
            
            x = ego_position[0] + distance * np.cos(angle)
            y = ego_position[1] + distance * np.sin(angle)
            z = ego_position[2]
            
            obj = self.generate_single_object(
                object_type=obj_type,
                position=(x, y, z),
                timestamp=timestamp
            )
            objects.append(obj)
        
        return objects
    
    def generate_object_trajectory(
        self,
        object_type: ObjectType = ObjectType.VEHICLE,
        start_position: Tuple[float, float, float] = (0, 0, 0),
        duration: float = 10.0,
        sample_rate: float = 10.0,
        motion_pattern: str = "straight"
    ) -> List[TrackedObject]:
        """
        生成目标轨迹
        
        Args:
            object_type: 目标类型
            start_position: 起始位置
            duration: 持续时间（秒）
            sample_rate: 采样率（Hz）
            motion_pattern: 运动模式 ("straight", "turn", "accelerate", "decelerate")
            
        Returns:
            TrackedObject列表（按时间顺序）
        """
        num_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        
        trajectory = []
        obj_id = str(uuid.uuid4())[:8]
        
        # 初始状态
        x, y, z = start_position
        speed = 10.0  # m/s
        heading = 0.0  # rad
        
        for i in range(num_samples):
            t = i * dt
            
            # 根据运动模式更新状态
            if motion_pattern == "straight":
                # 直线运动
                pass
            elif motion_pattern == "turn":
                # 转弯
                heading += 0.02 * dt
            elif motion_pattern == "accelerate":
                # 加速
                speed += 0.5 * dt
            elif motion_pattern == "decelerate":
                # 减速
                speed -= 0.5 * dt
                speed = max(0, speed)
            
            # 更新位置
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            
            x += vx * dt
            y += vy * dt
            
            # 获取尺寸
            length, width, height = self.DEFAULT_DIMENSIONS[object_type]
            
            # 创建目标
            bbox = BoundingBox3D(
                center_x=x,
                center_y=y,
                center_z=z + height / 2,
                length=length,
                width=width,
                height=height,
                heading=heading
            )
            
            obj = TrackedObject(
                id=obj_id,
                object_type=object_type,
                status=ObjectStatus.MOVING if speed > 0.5 else ObjectStatus.STATIONARY,
                bbox=bbox,
                velocity=np.array([vx, vy, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                timestamp=t,
                confidence=0.95,
                age=i
            )
            
            trajectory.append(obj)
        
        return trajectory
    
    def generate_detection_result(
        self,
        ground_truth: TrackedObject,
        noise_level: float = 0.1,
        miss_rate: float = 0.05,
        false_positive_rate: float = 0.02
    ) -> Optional[ObjectDetection]:
        """
        根据真值生成检测结果（添加噪声）
        
        Args:
            ground_truth: 真值目标
            noise_level: 噪声水平
            miss_rate: 漏检率
            false_positive_rate: 虚警率
            
        Returns:
            ObjectDetection对象，可能为None（漏检）
        """
        # 模拟漏检
        if self.rng.random() < miss_rate:
            return None
        
        # 添加位置噪声
        x = ground_truth.bbox.center_x + self.rng.normal(0, noise_level)
        y = ground_truth.bbox.center_y + self.rng.normal(0, noise_level)
        z = ground_truth.bbox.center_z + self.rng.normal(0, noise_level * 0.5)
        
        # 添加尺寸噪声
        length = ground_truth.bbox.length * (1 + self.rng.normal(0, noise_level * 0.1))
        width = ground_truth.bbox.width * (1 + self.rng.normal(0, noise_level * 0.1))
        height = ground_truth.bbox.height * (1 + self.rng.normal(0, noise_level * 0.1))
        
        # 添加朝向噪声
        heading = ground_truth.bbox.heading + self.rng.normal(0, noise_level * 0.1)
        
        bbox = BoundingBox3D(
            center_x=x,
            center_y=y,
            center_z=z,
            length=length,
            width=width,
            height=height,
            heading=heading
        )
        
        # 置信度
        confidence = max(0.5, min(0.99, ground_truth.confidence + self.rng.normal(0, 0.05)))
        
        return ObjectDetection(
            object_type=ground_truth.object_type,
            bbox=bbox,
            confidence=confidence,
            timestamp=ground_truth.timestamp,
            source_sensor="simulated"
        )
    
    def generate_detection_results(
        self,
        ground_truth_objects: List[TrackedObject],
        noise_level: float = 0.1,
        miss_rate: float = 0.05,
        false_positive_rate: float = 0.02
    ) -> List[ObjectDetection]:
        """
        为多个真值目标生成检测结果
        
        Args:
            ground_truth_objects: 真值目标列表
            noise_level: 噪声水平
            miss_rate: 漏检率
            false_positive_rate: 虚警率
            
        Returns:
            ObjectDetection列表
        """
        detections = []
        
        # 生成检测结果
        for gt_obj in ground_truth_objects:
            det = self.generate_detection_result(gt_obj, noise_level, miss_rate, false_positive_rate)
            if det is not None:
                detections.append(det)
        
        # 添加虚警
        num_fp = int(len(ground_truth_objects) * false_positive_rate)
        for _ in range(num_fp):
            obj_type = self.rng.choice(list(ObjectType))
            fp_obj = self.generate_single_object(object_type=obj_type)
            fp_det = self.generate_detection_result(fp_obj, noise_level, 0, 0)
            if fp_det:
                fp_det.confidence = self.rng.uniform(0.5, 0.7)  # 虚警置信度较低
                detections.append(fp_det)
        
        return detections
