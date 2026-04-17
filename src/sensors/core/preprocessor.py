"""
数据预处理模块
实现传感器数据的预处理和格式转换
"""

import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..drivers.base_sensor import (
    SensorData, ImageData, PointCloudData, 
    RadarData, UltrasonicData, VehicleStateData
)


@dataclass
class ImagePreprocessConfig:
    """图像预处理配置"""
    target_size: Tuple[int, int] = (1920, 1080)
    normalize: bool = True
    mean: List[float] = None
    std: List[float] = None
    color_space: str = "rgb"
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]


@dataclass
class PointCloudPreprocessConfig:
    """点云预处理配置"""
    range_filter: Tuple[float, float] = (0.3, 200.0)
    voxel_size: float = 0.1
    max_points: int = 100000
    remove_ground: bool = False
    ground_height: float = -0.5


@dataclass
class RadarPreprocessConfig:
    """雷达预处理配置"""
    range_filter: Tuple[float, float] = (0.5, 250.0)
    velocity_filter: Tuple[float, float] = (-50.0, 50.0)
    rcs_filter: Tuple[float, float] = (-10.0, 50.0)


@dataclass
class UltrasonicPreprocessConfig:
    """超声波预处理配置"""
    median_filter_size: int = 3
    outlier_threshold: float = 0.5


class DataPreprocessor:
    """
    数据预处理器
    实现各类传感器数据的预处理
    """
    
    def __init__(self):
        self._image_config = ImagePreprocessConfig()
        self._pointcloud_config = PointCloudPreprocessConfig()
        self._radar_config = RadarPreprocessConfig()
        self._ultrasonic_config = UltrasonicPreprocessConfig()
        
        # 预处理器注册表
        self._preprocessors: Dict[type, Callable] = {
            ImageData: self.preprocess_image,
            PointCloudData: self.preprocess_pointcloud,
            RadarData: self.preprocess_radar,
            UltrasonicData: self.preprocess_ultrasonic,
            VehicleStateData: self.preprocess_vehicle_state,
        }
    
    def set_image_config(self, config: ImagePreprocessConfig) -> None:
        """设置图像预处理配置"""
        self._image_config = config
    
    def set_pointcloud_config(self, config: PointCloudPreprocessConfig) -> None:
        """设置点云预处理配置"""
        self._pointcloud_config = config
    
    def set_radar_config(self, config: RadarPreprocessConfig) -> None:
        """设置雷达预处理配置"""
        self._radar_config = config
    
    def set_ultrasonic_config(self, config: UltrasonicPreprocessConfig) -> None:
        """设置超声波预处理配置"""
        self._ultrasonic_config = config
    
    def preprocess(self, data: SensorData) -> SensorData:
        """
        预处理传感器数据
        Args:
            data: 原始传感器数据
        Returns:
            SensorData: 预处理后的数据
        """
        data_type = type(data)
        
        if data_type in self._preprocessors:
            return self._preprocessors[data_type](data)
        
        return data
    
    def preprocess_image(self, data: ImageData) -> ImageData:
        """
        预处理图像数据
        Args:
            data: 原始图像数据
        Returns:
            ImageData: 预处理后的图像数据
        """
        if not CV2_AVAILABLE or data.image is None or data.image.size == 0:
            return data
        
        image = data.image.copy()
        
        # 调整大小
        target_w, target_h = self._image_config.target_size
        if image.shape[1] != target_w or image.shape[0] != target_h:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 颜色空间转换
        if self._image_config.color_space == "rgb":
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self._image_config.color_space == "gray":
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)
        
        # 归一化
        if self._image_config.normalize:
            image = image.astype(np.float32) / 255.0
            
            # 标准化
            if len(image.shape) == 3 and image.shape[2] == 3:
                mean = np.array(self._image_config.mean).reshape(1, 1, 3)
                std = np.array(self._image_config.std).reshape(1, 1, 3)
                image = (image - mean) / std
        
        # 创建新的ImageData
        return ImageData(
            timestamp=data.timestamp,
            sensor_name=data.sensor_name,
            sensor_type=data.sensor_type,
            frame_id=data.frame_id,
            data=image,
            image=image,
            width=image.shape[1],
            height=image.shape[0],
            channels=1 if len(image.shape) == 2 else image.shape[2],
            encoding=self._image_config.color_space,
            metadata={**data.metadata, 'preprocessed': True}
        )
    
    def preprocess_pointcloud(self, data: PointCloudData) -> PointCloudData:
        """
        预处理点云数据
        Args:
            data: 原始点云数据
        Returns:
            PointCloudData: 预处理后的点云数据
        """
        if data.points is None or len(data.points) == 0:
            return data
        
        points = data.points.copy()
        intensities = data.intensities.copy() if data.intensities is not None and len(data.intensities) > 0 else None
        
        # 距离过滤
        r_min, r_max = self._pointcloud_config.range_filter
        distances = np.linalg.norm(points[:, :3], axis=1)
        range_mask = (distances >= r_min) & (distances <= r_max)
        points = points[range_mask]
        
        if intensities is not None and len(intensities) == len(range_mask):
            intensities = intensities[range_mask]
        
        # 地面点移除
        if self._pointcloud_config.remove_ground:
            ground_mask = points[:, 2] > self._pointcloud_config.ground_height
            points = points[ground_mask]
            if intensities is not None:
                intensities = intensities[ground_mask]
        
        # 点数限制
        max_points = self._pointcloud_config.max_points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            if intensities is not None:
                intensities = intensities[indices]
        
        return PointCloudData(
            timestamp=data.timestamp,
            sensor_name=data.sensor_name,
            sensor_type=data.sensor_type,
            frame_id=data.frame_id,
            data=points,
            points=points,
            intensities=intensities,
            num_points=len(points),
            metadata={**data.metadata, 'preprocessed': True}
        )
    
    def preprocess_radar(self, data: RadarData) -> RadarData:
        """
        预处理雷达数据
        Args:
            data: 原始雷达数据
        Returns:
            RadarData: 预处理后的雷达数据
        """
        if not data.targets:
            return data
        
        filtered_targets = []
        
        r_min, r_max = self._radar_config.range_filter
        v_min, v_max = self._radar_config.velocity_filter
        rcs_min, rcs_max = self._radar_config.rcs_filter
        
        for target in data.targets:
            # 距离过滤
            if not (r_min <= target.range <= r_max):
                continue
            
            # 速度过滤
            if not (v_min <= target.velocity <= v_max):
                continue
            
            # RCS过滤
            if not (rcs_min <= target.rcs <= rcs_max):
                continue
            
            filtered_targets.append(target)
        
        return RadarData(
            timestamp=data.timestamp,
            sensor_name=data.sensor_name,
            sensor_type=data.sensor_type,
            frame_id=data.frame_id,
            targets=filtered_targets,
            num_targets=len(filtered_targets),
            metadata={**data.metadata, 'preprocessed': True}
        )
    
    def preprocess_ultrasonic(self, data: UltrasonicData) -> UltrasonicData:
        """
        预处理超声波数据
        Args:
            data: 原始超声波数据
        Returns:
            UltrasonicData: 预处理后的超声波数据
        """
        # 超声波数据通常已经在驱动层进行了滤波
        # 这里可以添加额外的处理
        
        return UltrasonicData(
            timestamp=data.timestamp,
            sensor_name=data.sensor_name,
            sensor_type=data.sensor_type,
            frame_id=data.frame_id,
            distance=data.distance,
            confidence=data.confidence,
            temperature=data.temperature,
            metadata={**data.metadata, 'preprocessed': True}
        )
    
    def preprocess_vehicle_state(self, data: VehicleStateData) -> VehicleStateData:
        """
        预处理车辆状态数据
        Args:
            data: 原始车辆状态数据
        Returns:
            VehicleStateData: 预处理后的车辆状态数据
        """
        # 车辆状态数据通常不需要复杂预处理
        # 这里可以添加单位转换、平滑等操作
        
        return VehicleStateData(
            timestamp=data.timestamp,
            sensor_name=data.sensor_name,
            sensor_type=data.sensor_type,
            frame_id=data.frame_id,
            speed=data.speed,
            steering_angle=data.steering_angle,
            yaw_rate=data.yaw_rate,
            longitudinal_accel=data.longitudinal_accel,
            lateral_accel=data.lateral_accel,
            gear_position=data.gear_position,
            metadata={**data.metadata, 'preprocessed': True}
        )


class ImageTransformer:
    """
    图像变换器
    实现图像的几何变换和增强
    """
    
    def __init__(self):
        self._transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        添加变换函数
        Args:
            transform: 变换函数
        """
        self._transforms.append(transform)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        应用所有变换
        Args:
            image: 输入图像
        Returns:
            np.ndarray: 变换后的图像
        """
        result = image.copy()
        
        for transform in self._transforms:
            result = transform(result)
        
        return result
    
    @staticmethod
    def undistort(image: np.ndarray, 
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray) -> np.ndarray:
        """
        图像去畸变
        Args:
            image: 输入图像
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        Returns:
            np.ndarray: 去畸变后的图像
        """
        if not CV2_AVAILABLE:
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        
        return undistorted
    
    @staticmethod
    def crop(image: np.ndarray, 
             x: int, y: int, 
             w: int, h: int) -> np.ndarray:
        """
        裁剪图像
        Args:
            image: 输入图像
            x, y: 左上角坐标
            w, h: 宽高
        Returns:
            np.ndarray: 裁剪后的图像
        """
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """
        水平翻转
        Args:
            image: 输入图像
        Returns:
            np.ndarray: 翻转后的图像
        """
        if not CV2_AVAILABLE:
            return image
        return cv2.flip(image, 1)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整亮度
        Args:
            image: 输入图像
            factor: 亮度因子
        Returns:
            np.ndarray: 调整后的图像
        """
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整对比度
        Args:
            image: 输入图像
            factor: 对比度因子
        Returns:
            np.ndarray: 调整后的图像
        """
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


class PointCloudTransformer:
    """
    点云变换器
    实现点云的几何变换
    """
    
    @staticmethod
    def transform_points(points: np.ndarray, 
                        transform_matrix: np.ndarray) -> np.ndarray:
        """
        变换点云
        Args:
            points: Nx3或Nx4点云数组
            transform_matrix: 4x4变换矩阵
        Returns:
            np.ndarray: 变换后的点云
        """
        if points.shape[1] == 3:
            points_homo = np.hstack([points, np.ones((len(points), 1))])
        else:
            points_homo = points
        
        transformed = (transform_matrix @ points_homo.T).T
        
        if points.shape[1] == 3:
            return transformed[:, :3]
        return transformed
    
    @staticmethod
    def voxel_downsample(points: np.ndarray, 
                        voxel_size: float) -> np.ndarray:
        """
        体素降采样
        Args:
            points: Nx3点云数组
            voxel_size: 体素大小
        Returns:
            np.ndarray: 降采样后的点云
        """
        # 计算体素索引
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # 使用字典去重
        unique_voxels = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in unique_voxels:
                unique_voxels[key] = points[i]
        
        return np.array(list(unique_voxels.values()))
    
    @staticmethod
    def remove_outliers(points: np.ndarray, 
                       nb_neighbors: int = 20,
                       std_ratio: float = 2.0) -> np.ndarray:
        """
        移除离群点（简化实现）
        Args:
            points: Nx3点云数组
            nb_neighbors: 邻居点数
            std_ratio: 标准差比率
        Returns:
            np.ndarray: 过滤后的点云
        """
        if len(points) < nb_neighbors:
            return points
        
        # 计算每个点到其邻居的平均距离
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=nb_neighbors+1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # 基于统计阈值过滤
        mean = np.mean(mean_distances)
        std = np.std(mean_distances)
        threshold = mean + std_ratio * std
        
        mask = mean_distances < threshold
        return points[mask]
    
    @staticmethod
    def crop_region(points: np.ndarray,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   z_range: Tuple[float, float]) -> np.ndarray:
        """
        裁剪区域
        Args:
            points: Nx3点云数组
            x_range, y_range, z_range: 各轴范围
        Returns:
            np.ndarray: 裁剪后的点云
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask]


class DataNormalizer:
    """
    数据归一化器
    实现数据的归一化和标准化
    """
    
    @staticmethod
    def min_max_normalize(data: np.ndarray, 
                         axis: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        最小-最大归一化
        Args:
            data: 输入数据
            axis: 归一化轴
        Returns:
            Tuple[np.ndarray, Dict]: 归一化后的数据和参数
        """
        if axis is None:
            min_val = np.min(data)
            max_val = np.max(data)
        else:
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
        
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        
        normalized = (data - min_val) / range_val
        
        params = {'min': min_val, 'max': max_val, 'method': 'min_max'}
        
        return normalized, params
    
    @staticmethod
    def z_score_normalize(data: np.ndarray,
                         axis: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Z-score标准化
        Args:
            data: 输入数据
            axis: 标准化轴
        Returns:
            Tuple[np.ndarray, Dict]: 标准化后的数据和参数
        """
        if axis is None:
            mean = np.mean(data)
            std = np.std(data)
        else:
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
        
        std = np.where(std == 0, 1, std)
        
        normalized = (data - mean) / std
        
        params = {'mean': mean, 'std': std, 'method': 'z_score'}
        
        return normalized, params
    
    @staticmethod
    def denormalize(normalized: np.ndarray, params: Dict) -> np.ndarray:
        """
        反归一化
        Args:
            normalized: 归一化后的数据
            params: 归一化参数
        Returns:
            np.ndarray: 原始数据
        """
        method = params.get('method')
        
        if method == 'min_max':
            return normalized * (params['max'] - params['min']) + params['min']
        elif method == 'z_score':
            return normalized * params['std'] + params['mean']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
