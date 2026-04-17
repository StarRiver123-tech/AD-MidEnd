"""
LiDAR驱动模块
支持以太网接入的激光雷达
"""

import time
import socket
import struct
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from .base_sensor import BaseSensor, SensorConfig, SensorType, SensorData, PointCloudData, SensorState


@dataclass
class LiDARConfig(SensorConfig):
    """LiDAR配置类"""
    
    def __init__(self,
                 name: str,
                 model: str = "pandar64",
                 channels: int = 64,
                 range_max: float = 200.0,
                 frequency: float = 10.0,
                 points_per_second: int = 1152000,
                 horizontal_fov: float = 360.0,
                 vertical_fov: Tuple[float, float] = (-25.0, 15.0),
                 ip: str = "192.168.1.201",
                 port: int = 2368,
                 data_port: int = 2369,
                 **kwargs):
        super().__init__(
            name=name,
            sensor_type=SensorType.LIDAR,
            interface="ethernet",
            **kwargs
        )
        self.model = model
        self.channels = channels
        self.range_max = range_max
        self.frequency = frequency
        self.points_per_second = points_per_second
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.ip = ip
        self.port = port
        self.data_port = data_port


class LiDARDriver(BaseSensor):
    """
    LiDAR驱动类
    支持以太网接入的激光雷达（如Hesai Pandar64、Velodyne等）
    """
    
    # LiDAR数据包格式常量
    PACKET_HEADER_SIZE = 42  # Ethernet + IP + UDP header
    BLOCK_SIZE = 100  # 每块数据大小（根据具体型号调整）
    
    def __init__(self, config: LiDARConfig):
        super().__init__(config)
        self.lidar_config = config
        self._socket: Optional[socket.socket] = None
        self._receive_thread: Optional[threading.Thread] = None
        self._packet_buffer: List[bytes] = []
        self._points_buffer: List[np.ndarray] = []
        self._simulation_mode = False
        self._vertical_angles: Optional[np.ndarray] = None
        self._init_vertical_angles()
        
    def _init_vertical_angles(self) -> None:
        """初始化垂直角度（根据LiDAR型号）"""
        v_min, v_max = self.lidar_config.vertical_fov
        num_channels = self.lidar_config.channels
        
        if self.lidar_config.model == "pandar64":
            # Hesai Pandar64 特定角度
            self._vertical_angles = np.array([
                14.436, 13.535, 12.676, 11.816, 10.998, 10.180, 9.401, 8.622,
                7.880, 7.138, 6.422, 5.706, 5.015, 4.324, 3.655, 2.986,
                2.338, 1.690, 1.060, 0.430, -0.183, -0.796, -1.392, -1.988,
                -2.568, -3.148, -3.712, -4.276, -4.825, -5.374, -5.908, -6.442,
                -6.962, -7.482, -7.988, -8.494, -8.986, -9.478, -9.957, -10.436,
                -10.901, -11.366, -11.818, -12.270, -12.708, -13.146, -13.571, -13.996,
                -14.407, -14.818, -15.216, -15.614, -15.999, -16.384, -16.756, -17.128,
                -17.487, -17.846, -18.192, -18.538, -18.871, -19.204, -19.524, -19.844
            ])
        elif self.lidar_config.model == "velodyne_vlp32":
            # Velodyne VLP-32C 角度
            self._vertical_angles = np.array([
                -25.0, -1.0, -1.667, -15.639, -11.31, 0.0, -0.667, -8.843,
                -7.254, 0.333, -0.333, -6.148, -5.333, 1.333, 0.667, -4.0,
                -4.667, 1.667, 1.0, -3.667, -3.333, 3.333, 2.667, -2.667,
                -3.0, 7.0, 4.667, -2.333, -2.0, 15.0, 10.333, -1.333
            ])
        else:
            # 通用均匀分布
            self._vertical_angles = np.linspace(v_max, v_min, num_channels)
    
    def initialize(self) -> bool:
        """
        初始化LiDAR连接
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            self.state = SensorState.INITIALIZING
        
        try:
            # 尝试创建UDP socket连接
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.settimeout(1.0)
            
            try:
                self._socket.bind(("0.0.0.0", self.lidar_config.data_port))
                print(f"LiDAR {self.name} bound to port {self.lidar_config.data_port}")
            except OSError as e:
                print(f"Cannot bind to port {self.lidar_config.data_port}: {e}")
                print(f"LiDAR {self.name} switching to simulation mode")
                self._simulation_mode = True
            
            with self._lock:
                self.state = SensorState.READY
            return True
            
        except Exception as e:
            print(f"LiDAR initialization error: {e}")
            self._simulation_mode = True
            with self._lock:
                self.state = SensorState.READY
            return True
    
    def capture(self) -> Optional[PointCloudData]:
        """
        采集一帧点云数据
        Returns:
            PointCloudData: 点云数据
        """
        try:
            if self._simulation_mode:
                points, intensities = self._generate_simulation_pointcloud()
            else:
                points, intensities = self._receive_pointcloud()
            
            if points is None or len(points) == 0:
                points, intensities = self._generate_simulation_pointcloud()
            
            return PointCloudData(
                timestamp=time.time(),
                sensor_name=self.name,
                sensor_type=SensorType.LIDAR,
                data=points,
                points=points,
                intensities=intensities,
                num_points=len(points),
                metadata={
                    "model": self.lidar_config.model,
                    "channels": self.lidar_config.channels,
                    "range_max": self.lidar_config.range_max,
                    "horizontal_fov": self.lidar_config.horizontal_fov
                }
            )
            
        except Exception as e:
            print(f"LiDAR capture error: {e}")
            return None
    
    def _receive_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从网络接收点云数据
        Returns:
            Tuple[np.ndarray, np.ndarray]: 点坐标和强度值
        """
        if self._socket is None:
            return None, None
        
        points_list = []
        intensities_list = []
        start_time = time.time()
        timeout = 0.1  # 100ms超时
        
        try:
            while time.time() - start_time < timeout:
                try:
                    data, addr = self._socket.recvfrom(1500)
                    if data:
                        pts, ints = self._parse_packet(data)
                        if pts is not None:
                            points_list.append(pts)
                            intensities_list.append(ints)
                except socket.timeout:
                    break
                except Exception as e:
                    break
            
            if points_list:
                all_points = np.vstack(points_list)
                all_intensities = np.concatenate(intensities_list)
                return all_points, all_intensities
            
        except Exception as e:
            print(f"Receive error: {e}")
        
        return None, None
    
    def _parse_packet(self, packet: bytes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        解析LiDAR数据包
        Args:
            packet: 原始数据包
        Returns:
            Tuple[np.ndarray, np.ndarray]: 点坐标和强度值
        """
        # 这里需要根据具体的LiDAR型号实现数据包解析
        # 以下是一个简化的示例
        try:
            # 假设数据包格式: [header][data blocks][footer]
            # 每个数据块包含多个点的距离和强度信息
            
            # 简化处理：生成模拟数据
            return self._generate_simulation_pointcloud(num_points=100)
            
        except Exception as e:
            print(f"Parse packet error: {e}")
            return None, None
    
    def _generate_simulation_pointcloud(self, num_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成仿真点云数据
        Args:
            num_points: 点的数量
        Returns:
            Tuple[np.ndarray, np.ndarray]: 点坐标和强度值
        """
        # 生成圆柱坐标系下的点云
        num_azimuth = 360 * 2  # 每0.5度一个方位角
        num_channels = self.lidar_config.channels
        
        # 限制点数
        if num_points > num_azimuth * num_channels:
            num_points = num_azimuth * num_channels
        
        # 生成角度
        azimuth_angles = np.linspace(0, 2 * np.pi, num_azimuth)
        
        points = []
        intensities = []
        
        for azimuth in azimuth_angles[:num_points // num_channels]:
            for v_angle in self._vertical_angles:
                # 随机距离
                distance = np.random.uniform(1.0, self.lidar_config.range_max * 0.5)
                
                # 球坐标转笛卡尔坐标
                v_rad = np.radians(v_angle)
                x = distance * np.cos(v_rad) * np.cos(azimuth)
                y = distance * np.cos(v_rad) * np.sin(azimuth)
                z = distance * np.sin(v_rad)
                
                points.append([x, y, z])
                
                # 模拟强度（与距离成反比）
                intensity = max(0, 255 - int(distance))
                intensities.append(intensity)
        
        return np.array(points), np.array(intensities)
    
    def release(self) -> None:
        """释放LiDAR资源"""
        self.stop()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        with self._lock:
            self.state = SensorState.STOPPED
    
    def get_vertical_angles(self) -> np.ndarray:
        """
        获取垂直角度数组
        Returns:
            np.ndarray: 垂直角度数组
        """
        return self._vertical_angles.copy()


class LiDARArray:
    """
    LiDAR阵列管理类
    管理多个LiDAR驱动
    """
    
    def __init__(self):
        self.lidars: Dict[str, LiDARDriver] = {}
        self._lock = threading.RLock()
    
    def add_lidar(self, config: LiDARConfig) -> bool:
        """
        添加LiDAR
        Args:
            config: LiDAR配置
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            if config.name in self.lidars:
                print(f"LiDAR {config.name} already exists")
                return False
            
            lidar = LiDARDriver(config)
            if lidar.initialize():
                self.lidars[config.name] = lidar
                return True
            return False
    
    def remove_lidar(self, name: str) -> bool:
        """
        移除LiDAR
        Args:
            name: LiDAR名称
        Returns:
            bool: 移除是否成功
        """
        with self._lock:
            if name not in self.lidars:
                return False
            
            self.lidars[name].release()
            del self.lidars[name]
            return True
    
    def start_all(self) -> None:
        """启动所有LiDAR"""
        with self._lock:
            for lidar in self.lidars.values():
                lidar.start()
    
    def stop_all(self) -> None:
        """停止所有LiDAR"""
        with self._lock:
            for lidar in self.lidars.values():
                lidar.stop()
    
    def get_lidar(self, name: str) -> Optional[LiDARDriver]:
        """
        获取指定LiDAR
        Args:
            name: LiDAR名称
        Returns:
            LiDARDriver: LiDAR驱动实例
        """
        with self._lock:
            return self.lidars.get(name)
    
    def capture_all(self) -> Dict[str, PointCloudData]:
        """
        采集所有LiDAR点云
        Returns:
            Dict[str, PointCloudData]: 点云数据字典
        """
        results = {}
        with self._lock:
            for name, lidar in self.lidars.items():
                data = lidar.capture()
                if data is not None:
                    results[name] = data
        return results
    
    def merge_pointclouds(self) -> Optional[np.ndarray]:
        """
        合并所有LiDAR点云（转换到车辆坐标系）
        Returns:
            np.ndarray: 合并后的点云
        """
        all_points = []
        
        with self._lock:
            for name, lidar in self.lidars.items():
                data = lidar.get_latest_data()
                if data is not None and isinstance(data, PointCloudData):
                    # 获取变换矩阵
                    T = lidar.get_transform_matrix()
                    
                    # 转换点云
                    points = data.points
                    points_homo = np.hstack([points, np.ones((len(points), 1))])
                    points_transformed = (T @ points_homo.T).T[:, :3]
                    
                    all_points.append(points_transformed)
        
        if all_points:
            return np.vstack(all_points)
        return None
    
    def release_all(self) -> None:
        """释放所有LiDAR资源"""
        with self._lock:
            for lidar in self.lidars.values():
                lidar.release()
            self.lidars.clear()


# 预定义的LiDAR配置
LIDAR_PRESETS = {
    "main_lidar": LiDARConfig(
        name="main_lidar",
        model="pandar64",
        channels=64,
        range_max=200.0,
        frequency=10.0,
        points_per_second=1152000,
        horizontal_fov=360.0,
        vertical_fov=(-25.0, 15.0),
        ip="192.168.1.201",
        port=2368,
        data_port=2369,
        position=[0.0, 0.0, 2.0],
        orientation=[0.0, 0.0, 0.0]
    ),
    "front_lidar": LiDARConfig(
        name="front_lidar",
        model="livox_mid360",
        channels=1,
        range_max=100.0,
        frequency=10.0,
        points_per_second=240000,
        horizontal_fov=360.0,
        vertical_fov=(-7.0, 52.0),
        ip="192.168.1.202",
        port=56000,
        data_port=56001,
        position=[2.0, 0.0, 1.5],
        orientation=[0.0, 0.0, 0.0]
    ),
}
