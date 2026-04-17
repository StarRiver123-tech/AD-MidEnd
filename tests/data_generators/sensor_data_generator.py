"""
传感器数据生成器
生成模拟的激光雷达、摄像头、毫米波雷达、IMU、GPS等传感器数据
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class SensorType(Enum):
    """传感器类型枚举"""
    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"


@dataclass
class PointCloud:
    """点云数据结构"""
    points: np.ndarray  # (N, 3) - x, y, z
    intensities: np.ndarray  # (N,) - 反射强度
    timestamps: np.ndarray  # (N,) - 时间戳
    
    def __post_init__(self):
        if len(self.points) != len(self.intensities):
            raise ValueError("points and intensities must have same length")


@dataclass
class ImageData:
    """图像数据结构"""
    data: np.ndarray  # (H, W, C) 图像数据
    timestamp: float
    camera_id: str
    intrinsics: np.ndarray  # (3, 3) 内参矩阵
    extrinsics: np.ndarray  # (4, 4) 外参矩阵


@dataclass
class RadarData:
    """毫米波雷达数据结构"""
    targets: List[Dict]  # 目标列表
    timestamp: float
    radar_id: str


@dataclass
class IMUData:
    """IMU数据结构"""
    acceleration: np.ndarray  # (3,) m/s^2
    angular_velocity: np.ndarray  # (3,) rad/s
    orientation: np.ndarray  # (4,) 四元数
    timestamp: float


@dataclass
class GPSData:
    """GPS数据结构"""
    latitude: float
    longitude: float
    altitude: float
    speed: float
    heading: float
    timestamp: float


class SensorDataGenerator:
    """传感器数据生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化传感器数据生成器
        
        Args:
            seed: 随机种子，保证可重复性
        """
        self.rng = np.random.RandomState(seed)
        self.base_timestamp = time.time()
        
    def generate_lidar_pointcloud(
        self,
        num_points: int = 10000,
        range_m: float = 100.0,
        add_noise: bool = True,
        timestamp: Optional[float] = None
    ) -> PointCloud:
        """
        生成激光雷达点云数据
        
        Args:
            num_points: 点的数量
            range_m: 探测范围（米）
            add_noise: 是否添加噪声
            timestamp: 时间戳（可选）
            
        Returns:
            PointCloud对象
        """
        # 生成球坐标系的随机点
        theta = self.rng.uniform(0, 2 * np.pi, num_points)  # 方位角
        phi = self.rng.uniform(0, np.pi, num_points)  # 俯仰角
        r = self.rng.uniform(0, range_m, num_points)  # 距离
        
        # 转换为笛卡尔坐标
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        points = np.stack([x, y, z], axis=1)
        
        # 添加噪声
        if add_noise:
            noise = self.rng.normal(0, 0.02, points.shape)
            points += noise
        
        # 生成反射强度
        intensities = self.rng.uniform(0, 255, num_points)
        
        # 生成时间戳
        if timestamp is None:
            timestamp = self.base_timestamp
        timestamps = np.full(num_points, timestamp)
        
        return PointCloud(
            points=points,
            intensities=intensities,
            timestamps=timestamps
        )
    
    def generate_camera_image(
        self,
        image_size: Tuple[int, int] = (1920, 1080),
        num_objects: int = 5,
        camera_id: str = "camera_front",
        timestamp: Optional[float] = None
    ) -> ImageData:
        """
        生成摄像头图像数据
        
        Args:
            image_size: 图像尺寸 (width, height)
            num_objects: 图像中模拟的目标数量
            camera_id: 摄像头ID
            timestamp: 时间戳（可选）
            
        Returns:
            ImageData对象
        """
        width, height = image_size
        
        # 生成基础图像（模拟道路场景）
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加天空背景（蓝色渐变）
        for i in range(height // 2):
            color = int(135 + (255 - 135) * (i / (height // 2)))
            image[i, :] = [color, color, 255]
        
        # 添加道路（灰色）
        road_start = height // 2
        image[road_start:, :] = [128, 128, 128]
        
        # 添加车道线（白色）
        lane_y = height // 2 + 50
        for i in range(0, width, 100):
            image[lane_y:lane_y+5, i:i+50] = [255, 255, 255]
        
        # 添加模拟目标（车辆）
        for _ in range(num_objects):
            x = self.rng.randint(100, width - 200)
            y = self.rng.randint(road_start + 20, height - 100)
            w = self.rng.randint(60, 120)
            h = self.rng.randint(40, 80)
            
            # 绘制车辆矩形
            color = [self.rng.randint(0, 255) for _ in range(3)]
            image[y:y+h, x:x+w] = color
        
        # 生成内参矩阵
        fx = width * 0.8
        fy = height * 0.8
        cx = width / 2
        cy = height / 2
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 生成外参矩阵（单位矩阵）
        extrinsics = np.eye(4)
        
        if timestamp is None:
            timestamp = self.base_timestamp
            
        return ImageData(
            data=image,
            timestamp=timestamp,
            camera_id=camera_id,
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    def generate_radar_data(
        self,
        num_targets: int = 10,
        max_range: float = 150.0,
        radar_id: str = "radar_front",
        timestamp: Optional[float] = None
    ) -> RadarData:
        """
        生成毫米波雷达数据
        
        Args:
            num_targets: 目标数量
            max_range: 最大探测距离
            radar_id: 雷达ID
            timestamp: 时间戳（可选）
            
        Returns:
            RadarData对象
        """
        targets = []
        
        for i in range(num_targets):
            target = {
                'id': i,
                'range': self.rng.uniform(5, max_range),
                'azimuth': self.rng.uniform(-np.pi/3, np.pi/3),
                'elevation': self.rng.uniform(-0.1, 0.1),
                'velocity': self.rng.uniform(-30, 30),
                'rcs': self.rng.uniform(0, 100),  # 雷达散射截面
                'snr': self.rng.uniform(10, 30),  # 信噪比
            }
            targets.append(target)
        
        if timestamp is None:
            timestamp = self.base_timestamp
            
        return RadarData(
            targets=targets,
            timestamp=timestamp,
            radar_id=radar_id
        )
    
    def generate_imu_data(
        self,
        duration: float = 1.0,
        sample_rate: float = 100.0,
        add_noise: bool = True
    ) -> List[IMUData]:
        """
        生成IMU数据序列
        
        Args:
            duration: 持续时间（秒）
            sample_rate: 采样率（Hz）
            add_noise: 是否添加噪声
            
        Returns:
            IMUData对象列表
        """
        num_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        
        imu_data_list = []
        
        for i in range(num_samples):
            t = i * dt
            
            # 基础加速度（重力 + 运动）
            accel = np.array([0.0, 0.0, 9.81])
            
            # 添加运动加速度
            accel[0] += 0.5 * np.sin(2 * np.pi * 0.5 * t)  # x方向加速度
            accel[1] += 0.3 * np.cos(2 * np.pi * 0.3 * t)  # y方向加速度
            
            # 角速度
            gyro = np.array([
                0.01 * np.sin(2 * np.pi * 0.2 * t),
                0.02 * np.cos(2 * np.pi * 0.3 * t),
                0.005 * np.sin(2 * np.pi * 0.1 * t)
            ])
            
            # 添加噪声
            if add_noise:
                accel += self.rng.normal(0, 0.01, 3)
                gyro += self.rng.normal(0, 0.001, 3)
            
            # 四元数（简化，假设小角度旋转）
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
            
            imu_data = IMUData(
                acceleration=accel,
                angular_velocity=gyro,
                orientation=orientation,
                timestamp=self.base_timestamp + t
            )
            imu_data_list.append(imu_data)
        
        return imu_data_list
    
    def generate_gps_data(
        self,
        duration: float = 1.0,
        sample_rate: float = 10.0,
        start_pos: Tuple[float, float] = (39.9042, 116.4074),  # 北京坐标
        add_noise: bool = True
    ) -> List[GPSData]:
        """
        生成GPS数据序列
        
        Args:
            duration: 持续时间（秒）
            sample_rate: 采样率（Hz）
            start_pos: 起始位置 (纬度, 经度)
            add_noise: 是否添加噪声
            
        Returns:
            GPSData对象列表
        """
        num_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        
        gps_data_list = []
        lat, lon = start_pos
        
        # 模拟车辆运动
        speed = 10.0  # m/s
        heading = 0.0  # 正北方向
        
        for i in range(num_samples):
            t = i * dt
            
            # 更新位置（简化模型）
            distance = speed * dt
            lat += distance * np.cos(heading) / 111320.0  # 纬度变化
            lon += distance * np.sin(heading) / (111320.0 * np.cos(np.radians(lat)))  # 经度变化
            
            # 添加噪声
            if add_noise:
                lat += self.rng.normal(0, 1e-6)
                lon += self.rng.normal(0, 1e-6)
            
            gps_data = GPSData(
                latitude=lat,
                longitude=lon,
                altitude=50.0 + self.rng.normal(0, 0.5),
                speed=speed + self.rng.normal(0, 0.5),
                heading=heading,
                timestamp=self.base_timestamp + t
            )
            gps_data_list.append(gps_data)
        
        return gps_data_list
    
    def generate_sensor_suite(
        self,
        timestamp: Optional[float] = None
    ) -> Dict[SensorType, any]:
        """
        生成完整的传感器套件数据
        
        Args:
            timestamp: 时间戳（可选）
            
        Returns:
            包含各种传感器数据的字典
        """
        if timestamp is None:
            timestamp = self.base_timestamp
            
        return {
            SensorType.LIDAR: self.generate_lidar_pointcloud(timestamp=timestamp),
            SensorType.CAMERA: self.generate_camera_image(timestamp=timestamp),
            SensorType.RADAR: self.generate_radar_data(timestamp=timestamp),
            SensorType.IMU: self.generate_imu_data(duration=0.1)[0],
            SensorType.GPS: self.generate_gps_data(duration=0.1)[0],
        }
