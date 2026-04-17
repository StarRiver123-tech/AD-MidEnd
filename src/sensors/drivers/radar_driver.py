"""
雷达驱动模块
支持CAN总线接入的毫米波雷达
"""

import time
import struct
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from .base_sensor import BaseSensor, SensorConfig, SensorType, SensorData, RadarData, RadarTarget, SensorState


@dataclass
class RadarConfig(SensorConfig):
    """雷达配置类"""
    
    def __init__(self,
                 name: str,
                 model: str = "continental_ars430",
                 range_max: float = 250.0,
                 azimuth_fov: float = 60.0,
                 elevation_fov: float = 15.0,
                 can_channel: int = 0,
                 can_id: int = 0x200,
                 **kwargs):
        super().__init__(
            name=name,
            sensor_type=SensorType.RADAR,
            interface="can",
            **kwargs
        )
        self.model = model
        self.range_max = range_max
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        self.can_channel = can_channel
        self.can_id = can_id


class RadarDriver(BaseSensor):
    """
    雷达驱动类
    支持CAN总线接入的毫米波雷达（如Continental ARS430等）
    """
    
    def __init__(self, config: RadarConfig):
        super().__init__(config)
        self.radar_config = config
        self._can_interface: Optional[Any] = None
        self._targets: List[RadarTarget] = []
        self._target_id_counter = 0
        self._simulation_mode = False
        self._last_update_time = 0.0
        
    def initialize(self) -> bool:
        """
        初始化雷达连接
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            self.state = SensorState.INITIALIZING
        
        try:
            # 尝试初始化CAN接口
            try:
                import can
                # 尝试创建CAN总线接口
                self._can_interface = can.interface.Bus(
                    channel=f"can{self.radar_config.can_channel}",
                    bustype='socketcan'
                )
                print(f"Radar {self.name} connected to CAN channel {self.radar_config.can_channel}")
            except Exception as e:
                print(f"Cannot connect to CAN bus: {e}")
                print(f"Radar {self.name} switching to simulation mode")
                self._simulation_mode = True
            
            with self._lock:
                self.state = SensorState.READY
            return True
            
        except Exception as e:
            print(f"Radar initialization error: {e}")
            self._simulation_mode = True
            with self._lock:
                self.state = SensorState.READY
            return True
    
    def capture(self) -> Optional[RadarData]:
        """
        采集一帧雷达数据
        Returns:
            RadarData: 雷达数据
        """
        try:
            if self._simulation_mode:
                targets = self._generate_simulation_targets()
            else:
                targets = self._receive_can_data()
            
            self._targets = targets
            self._last_update_time = time.time()
            
            return RadarData(
                timestamp=time.time(),
                sensor_name=self.name,
                sensor_type=SensorType.RADAR,
                targets=targets,
                num_targets=len(targets),
                metadata={
                    "model": self.radar_config.model,
                    "range_max": self.radar_config.range_max,
                    "azimuth_fov": self.radar_config.azimuth_fov,
                    "elevation_fov": self.radar_config.elevation_fov,
                    "can_id": self.radar_config.can_id
                }
            )
            
        except Exception as e:
            print(f"Radar capture error: {e}")
            return None
    
    def _receive_can_data(self) -> List[RadarTarget]:
        """
        从CAN总线接收雷达数据
        Returns:
            List[RadarTarget]: 雷达目标列表
        """
        if self._can_interface is None:
            return []
        
        targets = []
        start_time = time.time()
        timeout = 0.05  # 50ms超时
        
        try:
            while time.time() - start_time < timeout:
                msg = self._can_interface.recv(timeout=0.01)
                if msg is None:
                    break
                
                # 检查CAN ID是否匹配
                if msg.arbitration_id == self.radar_config.can_id:
                    target = self._parse_can_message(msg.data)
                    if target is not None:
                        targets.append(target)
                        
        except Exception as e:
            print(f"CAN receive error: {e}")
        
        return targets
    
    def _parse_can_message(self, data: bytes) -> Optional[RadarTarget]:
        """
        解析CAN消息
        Args:
            data: CAN消息数据
        Returns:
            RadarTarget: 雷达目标
        """
        # 这里需要根据具体的雷达型号实现CAN消息解析
        # 以下是一个简化的Continental ARS430示例
        
        try:
            if len(data) < 8:
                return None
            
            # 假设数据格式（需要根据实际协议调整）:
            # bytes 0-1: 距离 (0.01 m/bit)
            # bytes 2-3: 方位角 (0.1 deg/bit, offset -204.8)
            # bytes 4-5: 速度 (0.01 m/s/bit, offset -163.84)
            # byte 6: RCS (1 dBsm/bit, offset -64)
            # byte 7: SNR (0.5 dB/bit)
            
            range_val = struct.unpack('>H', data[0:2])[0] * 0.01
            azimuth = struct.unpack('>h', data[2:4])[0] * 0.1 - 204.8
            velocity = struct.unpack('>h', data[4:6])[0] * 0.01 - 163.84
            rcs = data[6] - 64
            snr = data[7] * 0.5
            
            self._target_id_counter += 1
            
            return RadarTarget(
                id=self._target_id_counter,
                range=range_val,
                azimuth=azimuth,
                elevation=0.0,  # 单平面雷达无俯仰角
                velocity=velocity,
                rcs=rcs,
                snr=snr
            )
            
        except Exception as e:
            print(f"Parse CAN message error: {e}")
            return None
    
    def _generate_simulation_targets(self, num_targets: int = 5) -> List[RadarTarget]:
        """
        生成仿真雷达目标
        Args:
            num_targets: 目标数量
        Returns:
            List[RadarTarget]: 雷达目标列表
        """
        targets = []
        
        for i in range(num_targets):
            # 随机生成目标参数
            distance = np.random.uniform(5.0, self.radar_config.range_max * 0.8)
            azimuth = np.random.uniform(
                -self.radar_config.azimuth_fov / 2,
                self.radar_config.azimuth_fov / 2
            )
            elevation = np.random.uniform(
                -self.radar_config.elevation_fov / 2,
                self.radar_config.elevation_fov / 2
            )
            velocity = np.random.uniform(-30.0, 30.0)
            rcs = np.random.uniform(-10.0, 30.0)
            snr = np.random.uniform(10.0, 40.0)
            
            target = RadarTarget(
                id=i,
                range=distance,
                azimuth=azimuth,
                elevation=elevation,
                velocity=velocity,
                rcs=rcs,
                snr=snr
            )
            targets.append(target)
        
        return targets
    
    def release(self) -> None:
        """释放雷达资源"""
        self.stop()
        if self._can_interface is not None:
            self._can_interface.shutdown()
            self._can_interface = None
        with self._lock:
            self.state = SensorState.STOPPED
    
    def get_targets_in_cartesian(self) -> np.ndarray:
        """
        获取笛卡尔坐标系下的目标位置
        Returns:
            np.ndarray: Nx3数组 [x, y, z]
        """
        positions = []
        
        for target in self._targets:
            # 极坐标转笛卡尔坐标
            azimuth_rad = np.radians(target.azimuth)
            elevation_rad = np.radians(target.elevation)
            
            x = target.range * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = target.range * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = target.range * np.sin(elevation_rad)
            
            positions.append([x, y, z])
        
        return np.array(positions) if positions else np.array([])


class RadarArray:
    """
    雷达阵列管理类
    管理多个雷达驱动
    """
    
    def __init__(self):
        self.radars: Dict[str, RadarDriver] = {}
        self._lock = threading.RLock()
    
    def add_radar(self, config: RadarConfig) -> bool:
        """
        添加雷达
        Args:
            config: 雷达配置
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            if config.name in self.radars:
                print(f"Radar {config.name} already exists")
                return False
            
            radar = RadarDriver(config)
            if radar.initialize():
                self.radars[config.name] = radar
                return True
            return False
    
    def remove_radar(self, name: str) -> bool:
        """
        移除雷达
        Args:
            name: 雷达名称
        Returns:
            bool: 移除是否成功
        """
        with self._lock:
            if name not in self.radars:
                return False
            
            self.radars[name].release()
            del self.radars[name]
            return True
    
    def start_all(self) -> None:
        """启动所有雷达"""
        with self._lock:
            for radar in self.radars.values():
                radar.start()
    
    def stop_all(self) -> None:
        """停止所有雷达"""
        with self._lock:
            for radar in self.radars.values():
                radar.stop()
    
    def get_radar(self, name: str) -> Optional[RadarDriver]:
        """
        获取指定雷达
        Args:
            name: 雷达名称
        Returns:
            RadarDriver: 雷达驱动实例
        """
        with self._lock:
            return self.radars.get(name)
    
    def capture_all(self) -> Dict[str, RadarData]:
        """
        采集所有雷达数据
        Returns:
            Dict[str, RadarData]: 雷达数据字典
        """
        results = {}
        with self._lock:
            for name, radar in self.radars.items():
                data = radar.capture()
                if data is not None:
                    results[name] = data
        return results
    
    def merge_targets(self) -> List[Tuple[str, RadarTarget]]:
        """
        合并所有雷达目标
        Returns:
            List[Tuple[str, RadarTarget]]: (雷达名称, 目标)列表
        """
        all_targets = []
        
        with self._lock:
            for name, radar in self.radars.items():
                data = radar.get_latest_data()
                if data is not None and isinstance(data, RadarData):
                    for target in data.targets:
                        all_targets.append((name, target))
        
        return all_targets
    
    def get_all_targets_cartesian(self) -> Dict[str, np.ndarray]:
        """
        获取所有雷达的笛卡尔坐标目标
        Returns:
            Dict[str, np.ndarray]: 雷达名称到目标位置的映射
        """
        results = {}
        
        with self._lock:
            for name, radar in self.radars.items():
                targets = radar.get_targets_in_cartesian()
                if len(targets) > 0:
                    results[name] = targets
        
        return results
    
    def release_all(self) -> None:
        """释放所有雷达资源"""
        with self._lock:
            for radar in self.radars.values():
                radar.release()
            self.radars.clear()


# 预定义的雷达配置
RADAR_PRESETS = {
    "front_radar": RadarConfig(
        name="front_radar",
        model="continental_ars430",
        range_max=250.0,
        azimuth_fov=60.0,
        elevation_fov=15.0,
        can_channel=0,
        can_id=0x200,
        position=[3.5, 0.0, 0.5],
        orientation=[0.0, 0.0, 0.0]
    ),
    "left_front_radar": RadarConfig(
        name="left_front_radar",
        model="continental_ars430",
        range_max=100.0,
        azimuth_fov=90.0,
        elevation_fov=15.0,
        can_channel=0,
        can_id=0x210,
        position=[1.5, 1.0, 0.5],
        orientation=[0.0, 0.0, 45.0]
    ),
    "right_front_radar": RadarConfig(
        name="right_front_radar",
        model="continental_ars430",
        range_max=100.0,
        azimuth_fov=90.0,
        elevation_fov=15.0,
        can_channel=0,
        can_id=0x220,
        position=[1.5, -1.0, 0.5],
        orientation=[0.0, 0.0, -45.0]
    ),
    "left_rear_radar": RadarConfig(
        name="left_rear_radar",
        model="continental_ars430",
        range_max=80.0,
        azimuth_fov=150.0,
        elevation_fov=15.0,
        can_channel=1,
        can_id=0x230,
        position=[-1.5, 1.0, 0.5],
        orientation=[0.0, 0.0, 135.0]
    ),
    "right_rear_radar": RadarConfig(
        name="right_rear_radar",
        model="continental_ars430",
        range_max=80.0,
        azimuth_fov=150.0,
        elevation_fov=15.0,
        can_channel=1,
        can_id=0x240,
        position=[-1.5, -1.0, 0.5],
        orientation=[0.0, 0.0, -135.0]
    ),
}
