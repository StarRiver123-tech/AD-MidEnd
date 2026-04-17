"""
超声波传感器驱动模块
支持SPI总线接入的超声波传感器
"""

import time
import struct
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from .base_sensor import BaseSensor, SensorConfig, SensorType, SensorData, UltrasonicData, SensorState


@dataclass
class UltrasonicConfig(SensorConfig):
    """超声波传感器配置类"""
    
    def __init__(self,
                 name: str,
                 range_min: float = 0.15,
                 range_max: float = 5.0,
                 spi_channel: int = 0,
                 spi_bus: int = 0,
                 **kwargs):
        super().__init__(
            name=name,
            sensor_type=SensorType.ULTRASONIC,
            interface="spi",
            **kwargs
        )
        self.range_min = range_min
        self.range_max = range_max
        self.spi_channel = spi_channel
        self.spi_bus = spi_bus


class UltrasonicDriver(BaseSensor):
    """
    超声波传感器驱动类
    支持SPI总线接入的超声波传感器
    """
    
    # 声速 (m/s) at 20°C
    SOUND_SPEED = 343.0
    
    def __init__(self, config: UltrasonicConfig):
        super().__init__(config)
        self.ultrasonic_config = config
        self._spi_interface: Optional[Any] = None
        self._distance_history: List[float] = []
        self._history_size = 5
        self._simulation_mode = False
        
    def initialize(self) -> bool:
        """
        初始化超声波传感器
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            self.state = SensorState.INITIALIZING
        
        try:
            # 尝试初始化SPI接口
            try:
                import spidev
                self._spi_interface = spidev.SpiDev()
                self._spi_interface.open(
                    self.ultrasonic_config.spi_bus,
                    self.ultrasonic_config.spi_channel
                )
                self._spi_interface.max_speed_hz = 1000000  # 1MHz
                self._spi_interface.mode = 0
                print(f"Ultrasonic {self.name} connected to SPI bus {self.ultrasonic_config.spi_bus}, "
                      f"channel {self.ultrasonic_config.spi_channel}")
            except Exception as e:
                print(f"Cannot connect to SPI: {e}")
                print(f"Ultrasonic {self.name} switching to simulation mode")
                self._simulation_mode = True
            
            with self._lock:
                self.state = SensorState.READY
            return True
            
        except Exception as e:
            print(f"Ultrasonic initialization error: {e}")
            self._simulation_mode = True
            with self._lock:
                self.state = SensorState.READY
            return True
    
    def capture(self) -> Optional[UltrasonicData]:
        """
        采集超声波数据
        Returns:
            UltrasonicData: 超声波数据
        """
        try:
            if self._simulation_mode:
                distance = self._generate_simulation_distance()
            else:
                distance = self._read_spi_data()
            
            # 应用滤波
            filtered_distance = self._apply_filter(distance)
            
            # 计算置信度
            confidence = self._calculate_confidence(filtered_distance)
            
            return UltrasonicData(
                timestamp=time.time(),
                sensor_name=self.name,
                sensor_type=SensorType.ULTRASONIC,
                distance=filtered_distance,
                confidence=confidence,
                temperature=25.0,
                metadata={
                    "range_min": self.ultrasonic_config.range_min,
                    "range_max": self.ultrasonic_config.range_max,
                    "spi_channel": self.ultrasonic_config.spi_channel
                }
            )
            
        except Exception as e:
            print(f"Ultrasonic capture error: {e}")
            return None
    
    def _read_spi_data(self) -> float:
        """
        从SPI读取数据
        Returns:
            float: 测量距离(m)
        """
        if self._spi_interface is None:
            return self._generate_simulation_distance()
        
        try:
            # 发送触发命令
            trigger_cmd = [0x01, 0x00, 0x00]
            self._spi_interface.xfer2(trigger_cmd)
            
            # 等待测量完成
            time.sleep(0.01)
            
            # 读取结果
            read_cmd = [0x00, 0x00, 0x00, 0x00]
            response = self._spi_interface.xfer2(read_cmd)
            
            # 解析响应（根据具体传感器协议）
            # 假设响应格式: [status, distance_high, distance_low, checksum]
            if len(response) >= 4:
                distance_raw = (response[1] << 8) | response[2]
                # 转换为米
                distance = distance_raw / 1000.0
                
                # 范围限制
                distance = max(self.ultrasonic_config.range_min,
                             min(distance, self.ultrasonic_config.range_max))
                
                return distance
            
            return self._generate_simulation_distance()
            
        except Exception as e:
            print(f"SPI read error: {e}")
            return self._generate_simulation_distance()
    
    def _generate_simulation_distance(self) -> float:
        """
        生成仿真距离数据
        Returns:
            float: 仿真距离
        """
        # 生成在有效范围内的随机距离
        distance = np.random.uniform(
            self.ultrasonic_config.range_min,
            self.ultrasonic_config.range_max * 0.8
        )
        
        # 偶尔添加异常值
        if np.random.random() < 0.05:
            distance = np.random.choice([
                self.ultrasonic_config.range_min - 0.1,  # 过近
                self.ultrasonic_config.range_max + 0.1,  # 过远
                0.0  # 无回波
            ])
        
        return distance
    
    def _apply_filter(self, distance: float) -> float:
        """
        应用中值滤波
        Args:
            distance: 原始距离
        Returns:
            float: 滤波后的距离
        """
        # 添加到历史记录
        self._distance_history.append(distance)
        
        # 保持历史记录大小
        if len(self._distance_history) > self._history_size:
            self._distance_history.pop(0)
        
        # 中值滤波
        if len(self._distance_history) >= 3:
            return float(np.median(self._distance_history))
        
        return distance
    
    def _calculate_confidence(self, distance: float) -> float:
        """
        计算测量置信度
        Args:
            distance: 测量距离
        Returns:
            float: 置信度 (0-1)
        """
        # 在有效范围内置信度高
        if distance < self.ultrasonic_config.range_min:
            return 0.0
        elif distance > self.ultrasonic_config.range_max:
            return 0.0
        elif distance < self.ultrasonic_config.range_min * 2:
            # 近处可能有多径效应
            return 0.7
        elif distance > self.ultrasonic_config.range_max * 0.9:
            # 远处信噪比降低
            return 0.8
        else:
            return 1.0
    
    def release(self) -> None:
        """释放超声波传感器资源"""
        self.stop()
        if self._spi_interface is not None:
            self._spi_interface.close()
            self._spi_interface = None
        with self._lock:
            self.state = SensorState.STOPPED
    
    def is_obstacle_detected(self, threshold: float = 1.0) -> bool:
        """
        检测是否有障碍物
        Args:
            threshold: 距离阈值(m)
        Returns:
            bool: 是否检测到障碍物
        """
        data = self.get_latest_data()
        if data is not None and isinstance(data, UltrasonicData):
            return (data.distance < threshold and 
                   data.distance >= self.ultrasonic_config.range_min and
                   data.confidence > 0.5)
        return False


class UltrasonicArray:
    """
    超声波传感器阵列管理类
    管理12个超声波传感器
    """
    
    def __init__(self):
        self.ultrasonics: Dict[str, UltrasonicDriver] = {}
        self._lock = threading.RLock()
    
    def add_ultrasonic(self, config: UltrasonicConfig) -> bool:
        """
        添加超声波传感器
        Args:
            config: 超声波配置
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            if config.name in self.ultrasonics:
                print(f"Ultrasonic {config.name} already exists")
                return False
            
            ultrasonic = UltrasonicDriver(config)
            if ultrasonic.initialize():
                self.ultrasonics[config.name] = ultrasonic
                return True
            return False
    
    def remove_ultrasonic(self, name: str) -> bool:
        """
        移除超声波传感器
        Args:
            name: 传感器名称
        Returns:
            bool: 移除是否成功
        """
        with self._lock:
            if name not in self.ultrasonics:
                return False
            
            self.ultrasonics[name].release()
            del self.ultrasonics[name]
            return True
    
    def start_all(self) -> None:
        """启动所有超声波传感器"""
        with self._lock:
            for ultrasonic in self.ultrasonics.values():
                ultrasonic.start()
    
    def stop_all(self) -> None:
        """停止所有超声波传感器"""
        with self._lock:
            for ultrasonic in self.ultrasonics.values():
                ultrasonic.stop()
    
    def get_ultrasonic(self, name: str) -> Optional[UltrasonicDriver]:
        """
        获取指定超声波传感器
        Args:
            name: 传感器名称
        Returns:
            UltrasonicDriver: 超声波驱动实例
        """
        with self._lock:
            return self.ultrasonics.get(name)
    
    def capture_all(self) -> Dict[str, UltrasonicData]:
        """
        采集所有超声波数据
        Returns:
            Dict[str, UltrasonicData]: 超声波数据字典
        """
        results = {}
        with self._lock:
            for name, ultrasonic in self.ultrasonics.items():
                data = ultrasonic.capture()
                if data is not None:
                    results[name] = data
        return results
    
    def get_all_distances(self) -> Dict[str, float]:
        """
        获取所有传感器距离
        Returns:
            Dict[str, float]: 传感器名称到距离的映射
        """
        distances = {}
        
        with self._lock:
            for name, ultrasonic in self.ultrasonics.items():
                data = ultrasonic.get_latest_data()
                if data is not None and isinstance(data, UltrasonicData):
                    distances[name] = data.distance
        
        return distances
    
    def get_obstacle_map(self, threshold: float = 1.0) -> Dict[str, bool]:
        """
        获取障碍物检测图
        Args:
            threshold: 距离阈值
        Returns:
            Dict[str, bool]: 传感器名称到检测结果的映射
        """
        obstacle_map = {}
        
        with self._lock:
            for name, ultrasonic in self.ultrasonics.items():
                obstacle_map[name] = ultrasonic.is_obstacle_detected(threshold)
        
        return obstacle_map
    
    def get_front_obstacles(self, threshold: float = 1.0) -> List[str]:
        """
        获取前方障碍物
        Args:
            threshold: 距离阈值
        Returns:
            List[str]: 检测到障碍物的传感器名称列表
        """
        obstacles = []
        front_sensors = ["front_left_1", "front_left_2", "front_right_1", "front_right_2"]
        
        with self._lock:
            for name in front_sensors:
                if name in self.ultrasonics:
                    if self.ultrasonics[name].is_obstacle_detected(threshold):
                        obstacles.append(name)
        
        return obstacles
    
    def get_rear_obstacles(self, threshold: float = 1.0) -> List[str]:
        """
        获取后方障碍物
        Args:
            threshold: 距离阈值
        Returns:
            List[str]: 检测到障碍物的传感器名称列表
        """
        obstacles = []
        rear_sensors = ["rear_left_1", "rear_left_2", "rear_right_1", "rear_right_2"]
        
        with self._lock:
            for name in rear_sensors:
                if name in self.ultrasonics:
                    if self.ultrasonics[name].is_obstacle_detected(threshold):
                        obstacles.append(name)
        
        return obstacles
    
    def get_min_distance(self) -> Tuple[Optional[str], float]:
        """
        获取最小距离及其传感器
        Returns:
            Tuple[str, float]: (传感器名称, 最小距离)
        """
        min_distance = float('inf')
        min_sensor = None
        
        with self._lock:
            for name, ultrasonic in self.ultrasonics.items():
                data = ultrasonic.get_latest_data()
                if data is not None and isinstance(data, UltrasonicData):
                    if data.distance < min_distance and data.confidence > 0.5:
                        min_distance = data.distance
                        min_sensor = name
        
        return min_sensor, min_distance if min_distance != float('inf') else 0.0
    
    def release_all(self) -> None:
        """释放所有超声波传感器资源"""
        with self._lock:
            for ultrasonic in self.ultrasonics.values():
                ultrasonic.release()
            self.ultrasonics.clear()


# 预定义的超声波配置（12个传感器）
ULTRASONIC_PRESETS = {
    # 前部传感器
    "front_left_1": UltrasonicConfig(
        name="front_left_1",
        range_min=0.15,
        range_max=5.0,
        spi_channel=0,
        position=[3.8, 0.8, 0.3],
        orientation=[0.0, 0.0, 20.0]
    ),
    "front_left_2": UltrasonicConfig(
        name="front_left_2",
        range_min=0.15,
        range_max=5.0,
        spi_channel=1,
        position=[3.8, 0.4, 0.3],
        orientation=[0.0, 0.0, 10.0]
    ),
    "front_right_1": UltrasonicConfig(
        name="front_right_1",
        range_min=0.15,
        range_max=5.0,
        spi_channel=2,
        position=[3.8, -0.4, 0.3],
        orientation=[0.0, 0.0, -10.0]
    ),
    "front_right_2": UltrasonicConfig(
        name="front_right_2",
        range_min=0.15,
        range_max=5.0,
        spi_channel=3,
        position=[3.8, -0.8, 0.3],
        orientation=[0.0, 0.0, -20.0]
    ),
    # 后部传感器
    "rear_left_1": UltrasonicConfig(
        name="rear_left_1",
        range_min=0.15,
        range_max=5.0,
        spi_channel=4,
        position=[-3.8, 0.8, 0.3],
        orientation=[0.0, 0.0, 160.0]
    ),
    "rear_left_2": UltrasonicConfig(
        name="rear_left_2",
        range_min=0.15,
        range_max=5.0,
        spi_channel=5,
        position=[-3.8, 0.4, 0.3],
        orientation=[0.0, 0.0, 170.0]
    ),
    "rear_right_1": UltrasonicConfig(
        name="rear_right_1",
        range_min=0.15,
        range_max=5.0,
        spi_channel=6,
        position=[-3.8, -0.4, 0.3],
        orientation=[0.0, 0.0, -170.0]
    ),
    "rear_right_2": UltrasonicConfig(
        name="rear_right_2",
        range_min=0.15,
        range_max=5.0,
        spi_channel=7,
        position=[-3.8, -0.8, 0.3],
        orientation=[0.0, 0.0, -160.0]
    ),
    # 侧部传感器
    "left_front": UltrasonicConfig(
        name="left_front",
        range_min=0.15,
        range_max=5.0,
        spi_channel=8,
        position=[2.0, 1.0, 0.3],
        orientation=[0.0, 0.0, 90.0]
    ),
    "left_rear": UltrasonicConfig(
        name="left_rear",
        range_min=0.15,
        range_max=5.0,
        spi_channel=9,
        position=[-2.0, 1.0, 0.3],
        orientation=[0.0, 0.0, 90.0]
    ),
    "right_front": UltrasonicConfig(
        name="right_front",
        range_min=0.15,
        range_max=5.0,
        spi_channel=10,
        position=[2.0, -1.0, 0.3],
        orientation=[0.0, 0.0, -90.0]
    ),
    "right_rear": UltrasonicConfig(
        name="right_rear",
        range_min=0.15,
        range_max=5.0,
        spi_channel=11,
        position=[-2.0, -1.0, 0.3],
        orientation=[0.0, 0.0, -90.0]
    ),
}
