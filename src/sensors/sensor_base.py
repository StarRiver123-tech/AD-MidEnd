"""
自动驾驶系统 - 传感器基类
定义所有传感器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from enum import Enum, auto
import threading
import time
import numpy as np

from ..common.data_types import Timestamp, SensorConfig
from ..communication.message_bus import MessageBus, Publisher
from ..logs.logger import Logger


class SensorState(Enum):
    """传感器状态"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPED = auto()


class SensorBase(ABC):
    """传感器基类"""
    
    def __init__(self, sensor_id: str, config: SensorConfig,
                 message_bus: Optional[MessageBus] = None):
        """
        初始化传感器
        
        Args:
            sensor_id: 传感器唯一标识
            config: 传感器配置
            message_bus: 消息总线实例
        """
        self._sensor_id = sensor_id
        self._config = config
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger(f"Sensor-{sensor_id}")
        
        # 状态
        self._state = SensorState.UNINITIALIZED
        self._state_lock = threading.Lock()
        
        # 发布者
        self._publisher: Optional[Publisher] = None
        
        # 数据采集线程
        self._acquisition_thread: Optional[threading.Thread] = None
        self._running = False
        self._acquisition_rate = 10.0  # Hz
        
        # 统计
        self._frame_count = 0
        self._error_count = 0
        self._last_frame_time = 0.0
        self._actual_fps = 0.0
        
        # 回调
        self._data_callback: Optional[Callable] = None
    
    @property
    def sensor_id(self) -> str:
        return self._sensor_id
    
    @property
    def sensor_type(self) -> str:
        return self._config.sensor_type
    
    @property
    def state(self) -> SensorState:
        with self._state_lock:
            return self._state
    
    @state.setter
    def state(self, value: SensorState) -> None:
        with self._state_lock:
            old_state = self._state
            self._state = value
            self._logger.debug(f"State changed: {old_state.name} -> {value.name}")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def actual_fps(self) -> float:
        return self._actual_fps
    
    def set_data_callback(self, callback: Callable[[Any], None]) -> None:
        """设置数据回调函数"""
        self._data_callback = callback
    
    def initialize(self) -> bool:
        """
        初始化传感器
        
        Returns:
            是否初始化成功
        """
        if self.state != SensorState.UNINITIALIZED:
            self._logger.warning(f"Cannot initialize in state {self.state.name}")
            return False
        
        self.state = SensorState.INITIALIZING
        
        try:
            # 创建发布者
            topic = self._get_topic()
            self._publisher = Publisher(self._sensor_id, topic, self._message_bus)
            
            # 硬件初始化
            if not self._initialize_hardware():
                self.state = SensorState.ERROR
                return False
            
            self.state = SensorState.READY
            self._logger.info(f"Sensor {self._sensor_id} initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            self.state = SensorState.ERROR
            return False
    
    @abstractmethod
    def _initialize_hardware(self) -> bool:
        """初始化硬件（子类实现）"""
        pass
    
    def start(self) -> bool:
        """启动数据采集"""
        if self.state not in [SensorState.READY, SensorState.STOPPED]:
            self._logger.warning(f"Cannot start in state {self.state.name}")
            return False
        
        self._running = True
        self.state = SensorState.RUNNING
        
        # 启动采集线程
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop,
            name=f"Sensor-{self._sensor_id}",
            daemon=True
        )
        self._acquisition_thread.start()
        
        self._logger.info(f"Sensor {self._sensor_id} started")
        return True
    
    def stop(self) -> None:
        """停止数据采集"""
        self._running = False
        
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=2.0)
        
        self._stop_hardware()
        self.state = SensorState.STOPPED
        
        self._logger.info(f"Sensor {self._sensor_id} stopped")
    
    @abstractmethod
    def _stop_hardware(self) -> None:
        """停止硬件（子类实现）"""
        pass
    
    def _acquisition_loop(self) -> None:
        """数据采集循环"""
        period = 1.0 / self._acquisition_rate
        
        while self._running:
            start_time = time.time()
            
            try:
                # 采集数据
                data = self._acquire_data()
                
                if data is not None:
                    # 更新时间戳
                    current_time = time.time()
                    self._actual_fps = 1.0 / (current_time - self._last_frame_time) \
                        if self._last_frame_time > 0 else self._acquisition_rate
                    self._last_frame_time = current_time
                    
                    # 发布数据
                    self._publish_data(data)
                    
                    # 调用回调
                    if self._data_callback:
                        self._data_callback(data)
                    
                    self._frame_count += 1
                
            except Exception as e:
                self._error_count += 1
                self._logger.error(f"Acquisition error: {e}")
            
            # 控制采集频率
            elapsed = time.time() - start_time
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    @abstractmethod
    def _acquire_data(self) -> Any:
        """采集数据（子类实现）"""
        pass
    
    def _publish_data(self, data: Any) -> None:
        """发布数据"""
        if self._publisher:
            self._publisher.publish(data)
    
    @abstractmethod
    def _get_topic(self) -> str:
        """获取发布主题（子类实现）"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'sensor_id': self._sensor_id,
            'sensor_type': self.sensor_type,
            'state': self.state.name,
            'frame_count': self._frame_count,
            'error_count': self._error_count,
            'actual_fps': self._actual_fps,
            'target_fps': self._acquisition_rate
        }
    
    def get_extrinsics(self) -> np.ndarray:
        """获取外参"""
        return self._config.extrinsics.copy()
    
    def transform_to_vehicle_frame(self, points: np.ndarray) -> np.ndarray:
        """将点转换到车辆坐标系"""
        from ..common.geometry import transform_points
        return transform_points(points, self._config.extrinsics)
