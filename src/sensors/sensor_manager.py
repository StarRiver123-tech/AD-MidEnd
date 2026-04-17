"""
自动驾驶系统 - 传感器管理器
管理所有传感器的初始化和运行
"""

from typing import Dict, List, Optional, Type
import threading

from .sensor_base import SensorBase, SensorState
from .camera_sensor import CameraSensor
from .lidar_sensor import LidarSensor
from .radar_sensor import RadarSensor
from .ultrasonic_sensor import UltrasonicSensor
from .can_interface import CANInterface
from ..common.data_types import SensorConfig
from ..config.config_manager import ConfigManager
from ..communication.message_bus import MessageBus
from ..logs.logger import Logger


class SensorManager:
    """传感器管理器"""
    
    # 传感器类型映射
    SENSOR_TYPES: Dict[str, Type[SensorBase]] = {
        'camera': CameraSensor,
        'lidar': LidarSensor,
        'radar': RadarSensor,
        'ultrasonic': UltrasonicSensor
    }
    
    def __init__(self, message_bus: Optional[MessageBus] = None):
        """
        初始化传感器管理器
        
        Args:
            message_bus: 消息总线实例
        """
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger("SensorManager")
        
        # 传感器实例
        self._sensors: Dict[str, SensorBase] = {}
        self._can_interface: Optional[CANInterface] = None
        
        # 状态
        self._running = False
        self._initialized = False
        
        # 锁
        self._lock = threading.RLock()
    
    def initialize(self, config_manager: Optional[ConfigManager] = None) -> bool:
        """
        初始化所有传感器
        
        Args:
            config_manager: 配置管理器
        
        Returns:
            是否初始化成功
        """
        if self._initialized:
            self._logger.warning("SensorManager already initialized")
            return True
        
        if config_manager is None:
            config_manager = ConfigManager()
        
        # 获取传感器配置
        sensor_configs = config_manager.get_all_sensor_configs()
        
        if not sensor_configs:
            self._logger.warning("No sensor configurations found")
            return False
        
        self._logger.info(f"Initializing {len(sensor_configs)} sensors...")
        
        # 初始化每个传感器
        for sensor_id, config in sensor_configs.items():
            if not config.enabled:
                self._logger.info(f"Sensor {sensor_id} is disabled, skipping")
                continue
            
            if self._initialize_sensor(sensor_id, config):
                self._logger.info(f"Sensor {sensor_id} initialized successfully")
            else:
                self._logger.error(f"Failed to initialize sensor {sensor_id}")
        
        # 初始化CAN接口
        can_config = sensor_configs.get('can_vehicle')
        if can_config and can_config.enabled:
            self._can_interface = CANInterface(can_config, self._message_bus)
            if self._can_interface.initialize():
                self._logger.info("CAN interface initialized successfully")
            else:
                self._logger.error("Failed to initialize CAN interface")
                self._can_interface = None
        
        self._initialized = True
        self._logger.info("SensorManager initialization completed")
        return True
    
    def _initialize_sensor(self, sensor_id: str, config: SensorConfig) -> bool:
        """初始化单个传感器"""
        sensor_type = config.sensor_type.lower()
        
        if sensor_type not in self.SENSOR_TYPES:
            self._logger.error(f"Unknown sensor type: {sensor_type}")
            return False
        
        # 创建传感器实例
        sensor_class = self.SENSOR_TYPES[sensor_type]
        sensor = sensor_class(sensor_id, config, message_bus=self._message_bus)
        
        # 初始化传感器
        if sensor.initialize():
            with self._lock:
                self._sensors[sensor_id] = sensor
            return True
        else:
            return False
    
    def start(self) -> bool:
        """启动所有传感器"""
        if not self._initialized:
            self._logger.error("SensorManager not initialized")
            return False
        
        if self._running:
            self._logger.warning("SensorManager already running")
            return True
        
        self._logger.info("Starting all sensors...")
        
        # 启动传感器
        with self._lock:
            for sensor_id, sensor in self._sensors.items():
                if sensor.state == SensorState.READY:
                    if sensor.start():
                        self._logger.debug(f"Sensor {sensor_id} started")
                    else:
                        self._logger.error(f"Failed to start sensor {sensor_id}")
        
        # 启动CAN接口
        if self._can_interface:
            self._can_interface.start()
        
        self._running = True
        self._logger.info("All sensors started")
        return True
    
    def stop(self) -> None:
        """停止所有传感器"""
        if not self._running:
            return
        
        self._logger.info("Stopping all sensors...")
        
        # 停止传感器
        with self._lock:
            for sensor_id, sensor in self._sensors.items():
                sensor.stop()
                self._logger.debug(f"Sensor {sensor_id} stopped")
        
        # 停止CAN接口
        if self._can_interface:
            self._can_interface.stop()
        
        self._running = False
        self._logger.info("All sensors stopped")
    
    def get_sensor(self, sensor_id: str) -> Optional[SensorBase]:
        """获取传感器实例"""
        with self._lock:
            return self._sensors.get(sensor_id)
    
    def get_all_sensors(self) -> Dict[str, SensorBase]:
        """获取所有传感器"""
        with self._lock:
            return self._sensors.copy()
    
    def get_sensors_by_type(self, sensor_type: str) -> List[SensorBase]:
        """获取指定类型的传感器"""
        with self._lock:
            return [
                sensor for sensor in self._sensors.values()
                if sensor.sensor_type.lower() == sensor_type.lower()
            ]
    
    def get_can_interface(self) -> Optional[CANInterface]:
        """获取CAN接口"""
        return self._can_interface
    
    def get_sensor_status(self) -> Dict:
        """获取所有传感器状态"""
        with self._lock:
            return {
                sensor_id: {
                    'type': sensor.sensor_type,
                    'state': sensor.state.name,
                    'frame_count': sensor.frame_count,
                    'actual_fps': sensor.actual_fps
                }
                for sensor_id, sensor in self._sensors.items()
            }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            total_frames = sum(s.frame_count for s in self._sensors.values())
            
            return {
                'num_sensors': len(self._sensors),
                'running': self._running,
                'initialized': self._initialized,
                'total_frames': total_frames,
                'sensor_stats': {
                    sensor_id: sensor.get_stats()
                    for sensor_id, sensor in self._sensors.items()
                }
            }
    
    def reset(self) -> None:
        """重置传感器管理器"""
        self.stop()
        
        with self._lock:
            self._sensors.clear()
        
        self._can_interface = None
        self._initialized = False
        
        self._logger.info("SensorManager reset")
