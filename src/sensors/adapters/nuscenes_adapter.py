"""
nuScenes数据集适配器
支持nuScenes数据集作为仿真输入
"""

import os
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..drivers.base_sensor import (
    SensorData, ImageData, PointCloudData, 
    RadarData, RadarTarget, SensorType
)


@dataclass
class NuScenesConfig:
    """nuScenes配置"""
    data_root: str
    version: str = "v1.0-trainval"
    verbose: bool = False


class NuScenesAdapter:
    """
    nuScenes数据集适配器
    将nuScenes数据转换为传感器数据格式
    """
    
    # nuScenes传感器到本系统的映射
    SENSOR_MAPPING = {
        # 摄像头
        "CAM_FRONT": "front_long",
        "CAM_FRONT_LEFT": "left_front",
        "CAM_FRONT_RIGHT": "right_front",
        "CAM_BACK": "rear",
        "CAM_BACK_LEFT": "left_rear",
        "CAM_BACK_RIGHT": "right_rear",
        # LiDAR
        "LIDAR_TOP": "main_lidar",
        # 雷达
        "RADAR_FRONT": "front_radar",
        "RADAR_FRONT_LEFT": "left_front_radar",
        "RADAR_FRONT_RIGHT": "right_front_radar",
        "RADAR_BACK_LEFT": "left_rear_radar",
        "RADAR_BACK_RIGHT": "right_rear_radar",
    }
    
    def __init__(self, config: NuScenesConfig):
        self.config = config
        self._nusc = None
        self._initialized = False
        
        # 场景和样本数据
        self._current_scene = None
        self._current_sample = None
        self._sample_tokens: List[str] = []
        self._current_index = 0
        
        # 传感器数据缓存
        self._sensor_data_cache: Dict[str, Any] = {}
        
    def initialize(self) -> bool:
        """
        初始化nuScenes适配器
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 尝试导入nuScenes库
            from nuscenes.nuscenes import NuScenes
            
            # 检查数据路径
            if not os.path.exists(self.config.data_root):
                print(f"nuScenes data root not found: {self.config.data_root}")
                return False
            
            # 初始化NuScenes
            self._nusc = NuScenes(
                version=self.config.version,
                dataroot=self.config.data_root,
                verbose=self.config.verbose
            )
            
            self._initialized = True
            print(f"NuScenes adapter initialized with {len(self._nusc.scene)} scenes")
            return True
            
        except ImportError:
            print("nuscenes-devkit not installed, running in simulation mode")
            self._initialized = False
            return False
        except Exception as e:
            print(f"NuScenes initialization error: {e}")
            self._initialized = False
            return False
    
    def load_scene(self, scene_idx: int = 0) -> bool:
        """
        加载场景
        Args:
            scene_idx: 场景索引
        Returns:
            bool: 加载是否成功
        """
        if not self._initialized or self._nusc is None:
            return False
        
        try:
            self._current_scene = self._nusc.scene[scene_idx]
            self._sample_tokens = []
            
            # 收集所有样本token
            sample_token = self._current_scene['first_sample_token']
            while sample_token != "":
                self._sample_tokens.append(sample_token)
                sample = self._nusc.get('sample', sample_token)
                sample_token = sample['next']
            
            self._current_index = 0
            
            print(f"Loaded scene '{self._current_scene['name']}' with {len(self._sample_tokens)} samples")
            return True
            
        except Exception as e:
            print(f"Load scene error: {e}")
            return False
    
    def next_sample(self) -> Optional[Dict[str, SensorData]]:
        """
        获取下一个样本的数据
        Returns:
            Dict[str, SensorData]: 传感器数据字典
        """
        if not self._initialized or self._nusc is None:
            return self._generate_simulation_data()
        
        if self._current_index >= len(self._sample_tokens):
            print("Reached end of scene")
            return None
        
        sample_token = self._sample_tokens[self._current_index]
        self._current_sample = self._nusc.get('sample', sample_token)
        
        data = {}
        
        # 加载各传感器数据
        for sensor_name, sample_data_token in self._current_sample['data'].items():
            sensor_data = self._load_sensor_data(sensor_name, sample_data_token)
            if sensor_data is not None:
                mapped_name = self.SENSOR_MAPPING.get(sensor_name, sensor_name)
                data[mapped_name] = sensor_data
        
        self._current_index += 1
        
        return data
    
    def _load_sensor_data(self, sensor_name: str, 
                         sample_data_token: str) -> Optional[SensorData]:
        """
        加载传感器数据
        Args:
            sensor_name: nuScenes传感器名称
            sample_data_token: 样本数据token
        Returns:
            SensorData: 传感器数据
        """
        try:
            sample_data = self._nusc.get('sample_data', sample_data_token)
            
            # 根据传感器类型加载数据
            if 'CAM' in sensor_name:
                return self._load_camera_data(sensor_name, sample_data)
            elif 'LIDAR' in sensor_name:
                return self._load_lidar_data(sensor_name, sample_data)
            elif 'RADAR' in sensor_name:
                return self._load_radar_data(sensor_name, sample_data)
            
            return None
            
        except Exception as e:
            print(f"Load sensor data error: {e}")
            return None
    
    def _load_camera_data(self, sensor_name: str, 
                         sample_data: Dict) -> Optional[ImageData]:
        """
        加载摄像头数据
        Args:
            sensor_name: 传感器名称
            sample_data: 样本数据
        Returns:
            ImageData: 图像数据
        """
        try:
            import cv2
            
            # 加载图像文件
            file_path = os.path.join(self.config.data_root, sample_data['filename'])
            
            if not os.path.exists(file_path):
                print(f"Image file not found: {file_path}")
                return None
            
            image = cv2.imread(file_path)
            if image is None:
                return None
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mapped_name = self.SENSOR_MAPPING.get(sensor_name, sensor_name)
            
            return ImageData(
                timestamp=sample_data['timestamp'] / 1e6,  # 转换为秒
                sensor_name=mapped_name,
                sensor_type=SensorType.CAMERA,
                data=image,
                image=image,
                width=image.shape[1],
                height=image.shape[0],
                channels=image.shape[2],
                encoding="rgb8",
                metadata={
                    'filename': sample_data['filename'],
                    'ego_pose_token': sample_data['ego_pose_token'],
                    'calibrated_sensor_token': sample_data['calibrated_sensor_token']
                }
            )
            
        except Exception as e:
            print(f"Load camera data error: {e}")
            return None
    
    def _load_lidar_data(self, sensor_name: str,
                        sample_data: Dict) -> Optional[PointCloudData]:
        """
        加载LiDAR数据
        Args:
            sensor_name: 传感器名称
            sample_data: 样本数据
        Returns:
            PointCloudData: 点云数据
        """
        try:
            from nuscenes.utils.data_classes import LidarPointCloud
            
            # 加载点云文件
            file_path = os.path.join(self.config.data_root, sample_data['filename'])
            
            if not os.path.exists(file_path):
                print(f"LiDAR file not found: {file_path}")
                return None
            
            # 加载点云
            pc = LidarPointCloud.from_file(file_path)
            points = pc.points.T  # 转置为Nx4
            
            # 分离坐标和强度
            xyz = points[:, :3]
            intensity = points[:, 3] if points.shape[1] > 3 else np.ones(len(xyz))
            
            mapped_name = self.SENSOR_MAPPING.get(sensor_name, sensor_name)
            
            return PointCloudData(
                timestamp=sample_data['timestamp'] / 1e6,
                sensor_name=mapped_name,
                sensor_type=SensorType.LIDAR,
                data=xyz,
                points=xyz,
                intensities=intensity,
                num_points=len(xyz),
                metadata={
                    'filename': sample_data['filename'],
                    'ego_pose_token': sample_data['ego_pose_token'],
                    'calibrated_sensor_token': sample_data['calibrated_sensor_token']
                }
            )
            
        except Exception as e:
            print(f"Load LiDAR data error: {e}")
            return None
    
    def _load_radar_data(self, sensor_name: str,
                        sample_data: Dict) -> Optional[RadarData]:
        """
        加载雷达数据
        Args:
            sensor_name: 传感器名称
            sample_data: 样本数据
        Returns:
            RadarData: 雷达数据
        """
        try:
            from nuscenes.utils.data_classes import RadarPointCloud
            
            # 加载雷达数据文件
            file_path = os.path.join(self.config.data_root, sample_data['filename'])
            
            if not os.path.exists(file_path):
                print(f"Radar file not found: {file_path}")
                return None
            
            # 加载雷达点云
            pc = RadarPointCloud.from_file(file_path)
            points = pc.points.T
            
            # 解析雷达目标
            targets = []
            for i, point in enumerate(points):
                # nuScenes雷达数据格式: [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms]
                target = RadarTarget(
                    id=int(point[4]) if len(point) > 4 else i,
                    range=np.sqrt(point[0]**2 + point[1]**2),
                    azimuth=np.arctan2(point[1], point[0]),
                    elevation=np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2)),
                    velocity=np.sqrt(point[6]**2 + point[7]**2) if len(point) > 7 else 0.0,
                    rcs=point[5] if len(point) > 5 else 0.0,
                    snr=20.0  # 默认值
                )
                targets.append(target)
            
            mapped_name = self.SENSOR_MAPPING.get(sensor_name, sensor_name)
            
            return RadarData(
                timestamp=sample_data['timestamp'] / 1e6,
                sensor_name=mapped_name,
                sensor_type=SensorType.RADAR,
                targets=targets,
                num_targets=len(targets),
                metadata={
                    'filename': sample_data['filename'],
                    'ego_pose_token': sample_data['ego_pose_token'],
                    'calibrated_sensor_token': sample_data['calibrated_sensor_token']
                }
            )
            
        except Exception as e:
            print(f"Load radar data error: {e}")
            return None
    
    def _generate_simulation_data(self) -> Dict[str, SensorData]:
        """
        生成仿真数据（当nuScenes不可用时）
        Returns:
            Dict[str, SensorData]: 仿真数据
        """
        data = {}
        
        # 生成摄像头仿真数据
        for cam_name in ["front_long", "left_front", "right_front", "rear"]:
            image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            data[cam_name] = ImageData(
                timestamp=time.time(),
                sensor_name=cam_name,
                sensor_type=SensorType.CAMERA,
                data=image,
                image=image,
                width=1920,
                height=1080,
                channels=3,
                encoding="rgb8",
                metadata={'simulation': True}
            )
        
        # 生成LiDAR仿真数据
        num_points = 10000
        points = np.random.randn(num_points, 3) * 20
        data["main_lidar"] = PointCloudData(
            timestamp=time.time(),
            sensor_name="main_lidar",
            sensor_type=SensorType.LIDAR,
            data=points,
            points=points,
            intensities=np.random.rand(num_points),
            num_points=num_points,
            metadata={'simulation': True}
        )
        
        # 生成雷达仿真数据
        targets = []
        for i in range(10):
            target = RadarTarget(
                id=i,
                range=np.random.uniform(5, 100),
                azimuth=np.random.uniform(-0.5, 0.5),
                elevation=0.0,
                velocity=np.random.uniform(-20, 20),
                rcs=np.random.uniform(-5, 20),
                snr=np.random.uniform(10, 30)
            )
            targets.append(target)
        
        data["front_radar"] = RadarData(
            timestamp=time.time(),
            sensor_name="front_radar",
            sensor_type=SensorType.RADAR,
            targets=targets,
            num_targets=len(targets),
            metadata={'simulation': True}
        )
        
        return data
    
    def get_sensor_calibration(self, sensor_name: str) -> Optional[Dict]:
        """
        获取传感器标定信息
        Args:
            sensor_name: 传感器名称
        Returns:
            Dict: 标定信息
        """
        if not self._initialized or self._nusc is None:
            return None
        
        try:
            # 找到对应的nuScenes传感器名称
            nusc_name = None
            for n_name, mapped_name in self.SENSOR_MAPPING.items():
                if mapped_name == sensor_name:
                    nusc_name = n_name
                    break
            
            if nusc_name is None:
                return None
            
            # 获取当前样本的传感器标定
            if self._current_sample is None:
                return None
            
            sample_data_token = self._current_sample['data'].get(nusc_name)
            if sample_data_token is None:
                return None
            
            sample_data = self._nusc.get('sample_data', sample_data_token)
            calibrated_sensor = self._nusc.get(
                'calibrated_sensor', 
                sample_data['calibrated_sensor_token']
            )
            
            return {
                'translation': calibrated_sensor['translation'],
                'rotation': calibrated_sensor['rotation'],
                'camera_intrinsic': calibrated_sensor.get('camera_intrinsic')
            }
            
        except Exception as e:
            print(f"Get calibration error: {e}")
            return None
    
    def reset(self) -> None:
        """重置到场景开头"""
        self._current_index = 0
        self._current_sample = None
    
    def get_progress(self) -> float:
        """
        获取当前进度
        Returns:
            float: 进度(0-1)
        """
        if len(self._sample_tokens) == 0:
            return 0.0
        return self._current_index / len(self._sample_tokens)
    
    def is_initialized(self) -> bool:
        """
        检查是否已初始化
        Returns:
            bool: 是否已初始化
        """
        return self._initialized


class NuScenesPlayer:
    """
    nuScenes数据播放器
    支持按时间顺序播放nuScenes数据
    """
    
    def __init__(self, adapter: NuScenesAdapter):
        self._adapter = adapter
        self._playing = False
        self._playback_thread: Optional[threading.Thread] = None
        self._data_callbacks: List[Callable[[Dict[str, SensorData]], None]] = []
        self._playback_speed = 1.0
        
    def play(self, loop: bool = False) -> None:
        """
        开始播放
        Args:
            loop: 是否循环播放
        """
        if self._playing:
            return
        
        self._playing = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(loop,),
            daemon=True
        )
        self._playback_thread.start()
    
    def pause(self) -> None:
        """暂停播放"""
        self._playing = False
    
    def stop(self) -> None:
        """停止播放"""
        self._playing = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
    
    def set_playback_speed(self, speed: float) -> None:
        """
        设置播放速度
        Args:
            speed: 播放速度倍率
        """
        self._playback_speed = speed
    
    def register_callback(self, callback: Callable[[Dict[str, SensorData]], None]) -> None:
        """
        注册数据回调
        Args:
            callback: 回调函数
        """
        self._data_callbacks.append(callback)
    
    def _playback_loop(self, loop: bool) -> None:
        """播放循环"""
        while self._playing:
            data = self._adapter.next_sample()
            
            if data is None:
                if loop:
                    self._adapter.reset()
                    continue
                else:
                    break
            
            # 通知回调
            for callback in self._data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Playback callback error: {e}")
            
            # 控制播放速度
            time.sleep(0.1 / self._playback_speed)
