"""
自动驾驶系统 - 超声波传感器
实现超声波距离检测
"""

import numpy as np
from typing import Optional
import spidev  # SPI接口

from .sensor_base import SensorBase
from ..common.data_types import UltrasonicData, SensorConfig, Timestamp


class UltrasonicSensor(SensorBase):
    """超声波传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig, **kwargs):
        super().__init__(sensor_id, config, **kwargs)
        
        # 超声波参数
        self._max_range = config.parameters.get('max_range', 5.0)  # 最大检测距离
        self._min_range = config.parameters.get('min_range', 0.2)  # 最小检测距离
        self._acquisition_rate = 10.0  # 10Hz
        
        # SPI参数
        self._spi_channel = config.parameters.get('spi_channel', 0)
        
        # SPI接口
        self._spi: Optional[spidev.SpiDev] = None
        
        # 仿真模式
        self._simulation_mode = False
    
    def _initialize_hardware(self) -> bool:
        """初始化超声波传感器硬件"""
        try:
            # 初始化SPI接口
            self._spi = spidev.SpiDev()
            self._spi.open(0, self._spi_channel)
            self._spi.max_speed_hz = 1000000
            self._spi.mode = 0
            
            self._logger.info(f"Ultrasonic {self._sensor_id} initialized on SPI channel {self._spi_channel}")
            return True
            
        except Exception as e:
            self._logger.warning(f"Ultrasonic SPI initialization failed: {e}, using simulation mode")
            self._simulation_mode = True
            return True
    
    def _stop_hardware(self) -> None:
        """停止超声波传感器"""
        if self._spi:
            self._spi.close()
            self._spi = None
    
    def _acquire_data(self) -> Optional[UltrasonicData]:
        """采集超声波数据"""
        if self._simulation_mode:
            return self._acquire_simulation_data()
        
        try:
            # 通过SPI读取数据
            # 发送触发命令
            self._spi.xfer2([0x01])
            
            # 读取距离数据
            response = self._spi.readbytes(2)
            
            # 解析距离（根据具体传感器协议）
            distance = (response[0] << 8 | response[1]) * 0.01  # 转换为米
            
            # 读取置信度
            confidence = self._spi.readbytes(1)[0] / 255.0
            
            # 检查有效性
            is_valid = self._min_range <= distance <= self._max_range
            
            ultrasonic_data = UltrasonicData(
                timestamp=Timestamp.now(),
                sensor_id=self._sensor_id,
                distance=distance,
                confidence=confidence,
                is_valid=is_valid,
                extrinsics=self._config.extrinsics
            )
            
            return ultrasonic_data
            
        except Exception as e:
            self._logger.error(f"Ultrasonic read error: {e}")
            return None
    
    def _acquire_simulation_data(self) -> UltrasonicData:
        """采集仿真超声波数据"""
        # 生成随机距离
        distance = np.random.uniform(self._min_range, self._max_range)
        confidence = np.random.uniform(0.7, 1.0)
        is_valid = True
        
        ultrasonic_data = UltrasonicData(
            timestamp=Timestamp.now(),
            sensor_id=self._sensor_id,
            distance=distance,
            confidence=confidence,
            is_valid=is_valid,
            extrinsics=self._config.extrinsics
        )
        
        return ultrasonic_data
    
    def _get_topic(self) -> str:
        """获取发布主题"""
        return "sensor/ultrasonic"
    
    def get_detection_cone(self, ultrasonic_data: UltrasonicData) -> dict:
        """获取检测锥形区域"""
        # 超声波传感器的检测锥
        cone_angle = 30.0  # 锥形角度（度）
        
        return {
            'origin': [0, 0, 0],  # 传感器位置
            'direction': [1, 0, 0],  # 检测方向
            'angle': cone_angle,
            'max_range': self._max_range,
            'detected_range': ultrasonic_data.distance if ultrasonic_data.is_valid else None
        }
