"""
传感器管理器模块
统一管理所有传感器的初始化和运行
"""

import os
import yaml
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path

from ..drivers.base_sensor import BaseSensor, SensorConfig, SensorData, SensorType, SensorState
from ..drivers.camera_driver import CameraDriver, CameraConfig, CameraArray, CAMERA_PRESETS
from ..drivers.lidar_driver import LiDARDriver, LiDARConfig, LiDARArray, LIDAR_PRESETS
from ..drivers.radar_driver import RadarDriver, RadarConfig, RadarArray, RADAR_PRESETS
from ..drivers.ultrasonic_driver import UltrasonicDriver, UltrasonicConfig, UltrasonicArray, ULTRASONIC_PRESETS
from ..drivers.can_driver import CANDriver, CANConfig, VehicleStateManager, CAN_PRESETS
from .sync_manager import SyncManager, SyncConfig, create_default_sync_groups
from .data_publisher import DataPublisher, get_global_publisher
from .preprocessor import DataPreprocessor


class SensorManager:
    """
    传感器管理器
    统一管理所有传感器的生命周期
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化传感器管理器
        Args:
            config_path: 配置文件路径
        """
        # 配置
        self._config = self._load_config(config_path)
        
        # 传感器阵列
        self._camera_array = CameraArray()
        self._lidar_array = LiDARArray()
        self._radar_array = RadarArray()
        self._ultrasonic_array = UltrasonicArray()
        self._vehicle_manager = VehicleStateManager()
        
        # 同步管理器
        self._sync_manager = SyncManager(self._get_sync_config())
        
        # 数据发布器
        self._publisher = get_global_publisher()
        
        # 预处理器
        self._preprocessor = DataPreprocessor()
        
        # 运行状态
        self._running = False
        self._lock = threading.RLock()
        
        # 统计信息
        self._start_time = 0.0
        
    def _load_config(self, config_path: str) -> Dict:
        """
        加载配置文件
        Args:
            config_path: 配置文件路径
        Returns:
            Dict: 配置字典
        """
        if config_path is None:
            # 使用默认配置
            config_path = Path(__file__).parent.parent / "config" / "sensor_config.yaml"
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Loaded sensor config from {config_path}")
                return config
            except Exception as e:
                print(f"Failed to load config: {e}")
        
        # 返回默认配置
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'cameras': {},
            'lidars': {},
            'radars': {},
            'ultrasonics': {},
            'can_bus': {},
            'sync': {
                'mode': 'software',
                'master_clock': 'lidar',
                'time_tolerance_ms': 10.0,
                'sync_period_ms': 100.0
            }
        }
    
    def _get_sync_config(self) -> SyncConfig:
        """获取同步配置"""
        sync_config = self._config.get('sync', {})
        return SyncConfig(
            mode=sync_config.get('mode', 'software'),
            master_clock=sync_config.get('master_clock', 'lidar'),
            time_tolerance_ms=sync_config.get('time_tolerance_ms', 10.0),
            sync_period_ms=sync_config.get('sync_period_ms', 100.0)
        )
    
    def initialize_all(self) -> bool:
        """
        初始化所有传感器
        Returns:
            bool: 初始化是否成功
        """
        print("Initializing all sensors...")
        
        # 初始化摄像头
        self._init_cameras()
        
        # 初始化LiDAR
        self._init_lidars()
        
        # 初始化雷达
        self._init_radars()
        
        # 初始化超声波
        self._init_ultrasonics()
        
        # 初始化CAN总线
        self._init_can_bus()
        
        # 初始化同步组
        self._init_sync_groups()
        
        print("All sensors initialized")
        return True
    
    def _init_cameras(self) -> None:
        """初始化摄像头"""
        camera_configs = self._config.get('cameras', {})
        
        for name, cfg in camera_configs.items():
            if not cfg.get('enabled', True):
                continue
            
            config = CameraConfig(
                name=name,
                resolution=cfg.get('resolution', [1920, 1080]),
                fps=cfg.get('fps', 30),
                fov=cfg.get('fov', 90.0),
                channel=cfg.get('channel', 0),
                interface=cfg.get('interface', 'gmsl'),
                lens_type=cfg.get('lens_type', 'normal'),
                position=cfg.get('position', [0.0, 0.0, 0.0]),
                orientation=cfg.get('orientation', [0.0, 0.0, 0.0]),
                enabled=True
            )
            
            if self._camera_array.add_camera(config):
                print(f"  Camera '{name}' initialized")
            else:
                print(f"  Failed to initialize camera '{name}'")
    
    def _init_lidars(self) -> None:
        """初始化LiDAR"""
        lidar_configs = self._config.get('lidars', {})
        
        for name, cfg in lidar_configs.items():
            if not cfg.get('enabled', True):
                continue
            
            v_fov = cfg.get('vertical_fov', [-25.0, 15.0])
            
            config = LiDARConfig(
                name=name,
                model=cfg.get('model', 'pandar64'),
                channels=cfg.get('channels', 64),
                range_max=cfg.get('range', 200.0),
                frequency=cfg.get('frequency', 10.0),
                points_per_second=cfg.get('points_per_second', 1152000),
                horizontal_fov=cfg.get('horizontal_fov', 360.0),
                vertical_fov=(v_fov[0], v_fov[1]) if isinstance(v_fov, list) else v_fov,
                ip=cfg.get('ip', '192.168.1.201'),
                port=cfg.get('port', 2368),
                data_port=cfg.get('data_port', 2369),
                position=cfg.get('position', [0.0, 0.0, 2.0]),
                orientation=cfg.get('orientation', [0.0, 0.0, 0.0]),
                enabled=True
            )
            
            if self._lidar_array.add_lidar(config):
                print(f"  LiDAR '{name}' initialized")
            else:
                print(f"  Failed to initialize LiDAR '{name}'")
    
    def _init_radars(self) -> None:
        """初始化雷达"""
        radar_configs = self._config.get('radars', {})
        
        for name, cfg in radar_configs.items():
            if not cfg.get('enabled', True):
                continue
            
            config = RadarConfig(
                name=name,
                model=cfg.get('model', 'continental_ars430'),
                range_max=cfg.get('range_max', 250.0),
                azimuth_fov=cfg.get('azimuth_fov', 60.0),
                elevation_fov=cfg.get('elevation_fov', 15.0),
                can_channel=cfg.get('can_channel', 0),
                can_id=int(cfg.get('can_id', '0x200'), 16) if isinstance(cfg.get('can_id'), str) else cfg.get('can_id', 0x200),
                position=cfg.get('position', [0.0, 0.0, 0.0]),
                orientation=cfg.get('orientation', [0.0, 0.0, 0.0]),
                enabled=True
            )
            
            if self._radar_array.add_radar(config):
                print(f"  Radar '{name}' initialized")
            else:
                print(f"  Failed to initialize radar '{name}'")
    
    def _init_ultrasonics(self) -> None:
        """初始化超声波传感器"""
        ultrasonic_configs = self._config.get('ultrasonics', {})
        
        for name, cfg in ultrasonic_configs.items():
            if not cfg.get('enabled', True):
                continue
            
            r_range = cfg.get('range', [0.15, 5.0])
            
            config = UltrasonicConfig(
                name=name,
                range_min=r_range[0] if isinstance(r_range, list) else 0.15,
                range_max=r_range[1] if isinstance(r_range, list) else 5.0,
                spi_channel=cfg.get('spi_channel', 0),
                spi_bus=cfg.get('spi_bus', 0),
                position=cfg.get('position', [0.0, 0.0, 0.0]),
                orientation=cfg.get('orientation', [0.0, 0.0, 0.0]),
                enabled=True
            )
            
            if self._ultrasonic_array.add_ultrasonic(config):
                print(f"  Ultrasonic '{name}' initialized")
            else:
                print(f"  Failed to initialize ultrasonic '{name}'")
    
    def _init_can_bus(self) -> None:
        """初始化CAN总线"""
        can_configs = self._config.get('can_bus', {})
        
        for name, cfg in can_configs.items():
            if not cfg.get('enabled', True):
                continue
            
            config = CANConfig(
                name=name,
                channel=cfg.get('channel', 'can0'),
                interface=cfg.get('interface', 'socketcan'),
                bitrate=cfg.get('bitrate', 500000),
                enabled=True
            )
            
            if self._vehicle_manager.add_can_channel(config):
                print(f"  CAN channel '{name}' initialized")
            else:
                print(f"  Failed to initialize CAN channel '{name}'")
    
    def _init_sync_groups(self) -> None:
        """初始化同步组"""
        sync_groups = self._config.get('sync', {}).get('sync_groups', {})
        
        if not sync_groups:
            sync_groups = create_default_sync_groups()
        
        for group_name, sensor_list in sync_groups.items():
            for sensor_name in sensor_list:
                self._sync_manager.register_sensor(sensor_name, group_name)
        
        print(f"  Registered {len(sync_groups)} sync groups")
    
    def start_all(self) -> None:
        """启动所有传感器"""
        with self._lock:
            if self._running:
                return
            
            print("Starting all sensors...")
            
            # 启动摄像头
            self._camera_array.start_all()
            
            # 启动LiDAR
            self._lidar_array.start_all()
            
            # 启动雷达
            self._radar_array.start_all()
            
            # 启动超声波
            self._ultrasonic_array.start_all()
            
            # 启动CAN
            self._vehicle_manager.start_all()
            
            # 启动同步管理器
            self._sync_manager.start()
            
            self._running = True
            self._start_time = time.time()
            
            print("All sensors started")
    
    def stop_all(self) -> None:
        """停止所有传感器"""
        with self._lock:
            if not self._running:
                return
            
            print("Stopping all sensors...")
            
            # 停止同步管理器
            self._sync_manager.stop()
            
            # 停止摄像头
            self._camera_array.stop_all()
            
            # 停止LiDAR
            self._lidar_array.stop_all()
            
            # 停止雷达
            self._radar_array.stop_all()
            
            # 停止超声波
            self._ultrasonic_array.stop_all()
            
            # 停止CAN
            self._vehicle_manager.stop_all()
            
            self._running = False
            
            print("All sensors stopped")
    
    def release_all(self) -> None:
        """释放所有传感器资源"""
        self.stop_all()
        
        print("Releasing all sensor resources...")
        
        # 释放摄像头
        self._camera_array.release_all()
        
        # 释放LiDAR
        self._lidar_array.release_all()
        
        # 释放雷达
        self._radar_array.release_all()
        
        # 释放超声波
        self._ultrasonic_array.release_all()
        
        # 释放CAN
        self._vehicle_manager.release_all()
        
        print("All sensor resources released")
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """
        获取传感器状态
        Returns:
            Dict: 状态信息
        """
        return {
            'running': self._running,
            'cameras': self._camera_array.get_camera_states(),
            'lidars': {name: lidar.get_state().value 
                      for name, lidar in self._lidar_array.lidars.items()},
            'radars': {name: radar.get_state().value 
                      for name, radar in self._radar_array.radars.items()},
            'ultrasonics': {name: us.get_state().value 
                           for name, us in self._ultrasonic_array.ultrasonics.items()},
            'uptime': time.time() - self._start_time if self._running else 0.0
        }
    
    def get_all_data(self) -> Dict[str, SensorData]:
        """
        获取所有传感器的最新数据
        Returns:
            Dict[str, SensorData]: 数据字典
        """
        data = {}
        
        # 摄像头数据
        for name, camera in self._camera_array.get_all_cameras().items():
            cam_data = camera.get_latest_data()
            if cam_data is not None:
                data[name] = cam_data
        
        # LiDAR数据
        lidar_data = self._lidar_array.capture_all()
        data.update(lidar_data)
        
        # 雷达数据
        radar_data = self._radar_array.capture_all()
        data.update(radar_data)
        
        # 超声波数据
        us_data = self._ultrasonic_array.capture_all()
        data.update(us_data)
        
        return data
    
    def get_camera_array(self) -> CameraArray:
        """获取摄像头阵列"""
        return self._camera_array
    
    def get_lidar_array(self) -> LiDARArray:
        """获取LiDAR阵列"""
        return self._lidar_array
    
    def get_radar_array(self) -> RadarArray:
        """获取雷达阵列"""
        return self._radar_array
    
    def get_ultrasonic_array(self) -> UltrasonicArray:
        """获取超声波阵列"""
        return self._ultrasonic_array
    
    def get_vehicle_manager(self) -> VehicleStateManager:
        """获取车辆状态管理器"""
        return self._vehicle_manager
    
    def get_sync_manager(self) -> SyncManager:
        """获取同步管理器"""
        return self._sync_manager
    
    def get_publisher(self) -> DataPublisher:
        """获取数据发布器"""
        return self._publisher
    
    def get_preprocessor(self) -> DataPreprocessor:
        """获取预处理器"""
        return self._preprocessor
    
    def register_data_callback(self, sensor_type: SensorType, 
                              callback: Callable[[SensorData], None]) -> None:
        """
        注册数据回调
        Args:
            sensor_type: 传感器类型
            callback: 回调函数
        """
        # 根据类型注册到相应的传感器
        if sensor_type == SensorType.CAMERA:
            for camera in self._camera_array.get_all_cameras().values():
                camera.register_callback(callback)
        elif sensor_type == SensorType.LIDAR:
            for lidar in self._lidar_array.lidars.values():
                lidar.register_callback(callback)
        elif sensor_type == SensorType.RADAR:
            for radar in self._radar_array.radars.values():
                radar.register_callback(callback)
        elif sensor_type == SensorType.ULTRASONIC:
            for us in self._ultrasonic_array.ultrasonics.values():
                us.register_callback(callback)


def create_sensor_manager(config_path: str = None) -> SensorManager:
    """
    创建传感器管理器
    Args:
        config_path: 配置文件路径
    Returns:
        SensorManager: 传感器管理器实例
    """
    return SensorManager(config_path)
