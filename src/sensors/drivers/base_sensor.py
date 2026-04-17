"""
基础传感器接口模块
定义所有传感器的抽象基类和通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import time
import threading
import numpy as np
from collections import deque


class SensorState(Enum):
    """传感器状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class SensorType(Enum):
    """传感器类型枚举"""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GNSS = "gnss"
    CAN = "can"


@dataclass
class SensorConfig:
    """传感器配置数据类"""
    name: str
    sensor_type: SensorType
    enabled: bool = True
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    interface: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorData:
    """传感器数据基类"""
    timestamp: float
    sensor_name: str
    sensor_type: SensorType
    frame_id: int = 0
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ImageData(SensorData):
    """图像数据类"""
    image: np.ndarray = field(default_factory=lambda: np.array([]))
    width: int = 0
    height: int = 0
    channels: int = 0
    encoding: str = "rgb8"
    
    def __post_init__(self):
        super().__post_init__()
        if self.data is not None and isinstance(self.data, np.ndarray):
            self.image = self.data
            if self.image.size > 0:
                self.height, self.width = self.image.shape[:2]
                self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]


@dataclass
class PointCloudData(SensorData):
    """点云数据类"""
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    intensities: np.ndarray = field(default_factory=lambda: np.array([]))
    num_points: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.data is not None and isinstance(self.data, np.ndarray):
            self.points = self.data
            self.num_points = len(self.points)


@dataclass
class RadarTarget:
    """雷达目标数据类"""
    id: int
    range: float
    azimuth: float
    elevation: float
    velocity: float
    rcs: float  # 雷达散射截面
    snr: float  # 信噪比
    

@dataclass
class RadarData(SensorData):
    """雷达数据类"""
    targets: List[RadarTarget] = field(default_factory=list)
    num_targets: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.num_targets = len(self.targets)


@dataclass
class UltrasonicData(SensorData):
    """超声波数据类"""
    distance: float = 0.0
    confidence: float = 0.0
    temperature: float = 25.0


@dataclass
class VehicleStateData(SensorData):
    """车辆状态数据类"""
    speed: float = 0.0
    steering_angle: float = 0.0
    yaw_rate: float = 0.0
    longitudinal_accel: float = 0.0
    lateral_accel: float = 0.0
    gear_position: int = 0


class BaseSensor(ABC):
    """
    传感器基类
    所有具体传感器驱动都需要继承此类
    """
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.name = config.name
        self.sensor_type = config.sensor_type
        self.state = SensorState.UNINITIALIZED
        self._callbacks: List[Callable[[SensorData], None]] = []
        self._lock = threading.RLock()
        self._frame_id = 0
        self._latest_data: Optional[SensorData] = None
        self._data_buffer: deque = deque(maxlen=100)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._error_count = 0
        self._max_errors = 10
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化传感器
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def capture(self) -> Optional[SensorData]:
        """
        采集一帧数据
        Returns:
            SensorData: 采集到的数据，失败返回None
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """释放传感器资源"""
        pass
    
    def start(self) -> bool:
        """
        启动传感器数据采集
        Returns:
            bool: 启动是否成功
        """
        with self._lock:
            if self.state == SensorState.RUNNING:
                return True
                
            if self.state == SensorState.UNINITIALIZED:
                if not self.initialize():
                    self.state = SensorState.ERROR
                    return False
            
            self._running = True
            self.state = SensorState.RUNNING
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            return True
    
    def stop(self) -> None:
        """停止传感器数据采集"""
        with self._lock:
            self._running = False
            self.state = SensorState.STOPPED
            
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
    
    def pause(self) -> None:
        """暂停数据采集"""
        with self._lock:
            if self.state == SensorState.RUNNING:
                self.state = SensorState.PAUSED
    
    def resume(self) -> None:
        """恢复数据采集"""
        with self._lock:
            if self.state == SensorState.PAUSED:
                self.state = SensorState.RUNNING
    
    def register_callback(self, callback: Callable[[SensorData], None]) -> None:
        """
        注册数据回调函数
        Args:
            callback: 回调函数，接收SensorData参数
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[SensorData], None]) -> None:
        """
        注销数据回调函数
        Args:
            callback: 要注销的回调函数
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def get_latest_data(self) -> Optional[SensorData]:
        """
        获取最新一帧数据
        Returns:
            SensorData: 最新数据
        """
        with self._lock:
            return self._latest_data
    
    def get_data_buffer(self) -> List[SensorData]:
        """
        获取数据缓冲区
        Returns:
            List[SensorData]: 数据列表
        """
        with self._lock:
            return list(self._data_buffer)
    
    def get_state(self) -> SensorState:
        """
        获取传感器状态
        Returns:
            SensorState: 当前状态
        """
        with self._lock:
            return self.state
    
    def is_running(self) -> bool:
        """
        检查传感器是否正在运行
        Returns:
            bool: 是否运行中
        """
        with self._lock:
            return self.state == SensorState.RUNNING and self._running
    
    def _capture_loop(self) -> None:
        """数据采集循环线程"""
        while self._running:
            try:
                with self._lock:
                    if self.state == SensorState.PAUSED:
                        time.sleep(0.001)
                        continue
                
                data = self.capture()
                
                if data is not None:
                    self._frame_id += 1
                    data.frame_id = self._frame_id
                    
                    with self._lock:
                        self._latest_data = data
                        self._data_buffer.append(data)
                        self._error_count = 0
                    
                    # 调用回调函数
                    for callback in self._callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Callback error for {self.name}: {e}")
                
            except Exception as e:
                self._error_count += 1
                print(f"Capture error in {self.name}: {e}")
                
                if self._error_count >= self._max_errors:
                    with self._lock:
                        self.state = SensorState.ERROR
                    print(f"Sensor {self.name} entered error state")
                    break
                
                time.sleep(0.01)
    
    def _notify_callbacks(self, data: SensorData) -> None:
        """通知所有回调函数"""
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        获取传感器到车辆坐标系的变换矩阵
        Returns:
            np.ndarray: 4x4变换矩阵
        """
        pos = self.config.position
        ori = np.radians(self.config.orientation)
        
        # 旋转矩阵 (ZYX欧拉角)
        cx, sx = np.cos(ori[0]), np.sin(ori[0])
        cy, sy = np.cos(ori[1]), np.sin(ori[1])
        cz, sz = np.cos(ori[2]), np.sin(ori[2])
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        return T
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.sensor_type.value}, state={self.state.value})"
