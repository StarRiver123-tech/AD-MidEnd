"""
自动驾驶系统 - 雷达传感器
实现雷达目标数据采集
"""

import numpy as np
from typing import Optional, List
import can  # python-can库

from .sensor_base import SensorBase
from ..common.data_types import RadarData, RadarTarget, SensorConfig, Timestamp


class RadarSensor(SensorBase):
    """雷达传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig, **kwargs):
        super().__init__(sensor_id, config, **kwargs)
        
        # 雷达参数
        self._max_range = config.parameters.get('max_range', 150.0)
        self._fov = config.parameters.get('fov', 30.0)  # 视场角（度）
        self._acquisition_rate = 20.0  # 雷达通常20Hz
        
        # CAN参数
        self._can_channel = config.interface
        self._can_id = config.parameters.get('can_id', 0x200)
        
        # CAN总线接口
        self._can_bus: Optional[can.Bus] = None
        
        # 仿真模式
        self._simulation_mode = False
    
    def _initialize_hardware(self) -> bool:
        """初始化雷达硬件"""
        try:
            # 尝试连接CAN总线
            self._can_bus = can.Bus(
                channel=self._can_channel,
                bustype='socketcan',
                bitrate=500000
            )
            
            self._logger.info(f"Radar {self._sensor_id} initialized on {self._can_channel}")
            return True
            
        except Exception as e:
            self._logger.warning(f"Radar CAN initialization failed: {e}, using simulation mode")
            self._simulation_mode = True
            return True
    
    def _stop_hardware(self) -> None:
        """停止雷达"""
        if self._can_bus:
            self._can_bus.shutdown()
            self._can_bus = None
    
    def _acquire_data(self) -> Optional[RadarData]:
        """采集雷达数据"""
        if self._simulation_mode:
            return self._acquire_simulation_data()
        
        # 从CAN总线接收数据
        targets = []
        timeout = 0.05  # 50ms超时
        
        try:
            # 接收CAN消息
            start_time = Timestamp.now().to_seconds()
            while Timestamp.now().to_seconds() - start_time < timeout:
                msg = self._can_bus.recv(timeout=0.01)
                
                if msg is None:
                    break
                
                # 解析雷达数据（根据具体雷达协议）
                if msg.arbitration_id == self._can_id:
                    target = self._parse_can_message(msg)
                    if target:
                        targets.append(target)
        
        except Exception as e:
            self._logger.error(f"CAN receive error: {e}")
        
        if not targets:
            return None
        
        radar_data = RadarData(
            timestamp=Timestamp.now(),
            radar_id=self._sensor_id,
            targets=targets,
            extrinsics=self._config.extrinsics
        )
        
        return radar_data
    
    def _parse_can_message(self, msg: can.Message) -> Optional[RadarTarget]:
        """解析CAN消息（简化实现）"""
        # 这里应该根据具体的雷达协议解析
        # 不同雷达厂商有不同的数据格式
        
        try:
            data = msg.data
            
            # 简化的解析示例
            # 实际应根据雷达协议文档实现
            target = RadarTarget()
            
            # 假设数据格式：
            # bytes 0-1: 距离 (0.01m/bit)
            # bytes 2-3: 方位角 (0.01度/bit)
            # bytes 4-5: 径向速度 (0.01m/s/bit)
            # bytes 6: RCS
            # bytes 7: 信噪比
            
            if len(data) >= 8:
                target.range_distance = (data[0] | (data[1] << 8)) * 0.01
                target.azimuth = (data[2] | (data[3] << 8)) * 0.01 - 180.0
                target.velocity = (data[4] | (data[5] << 8)) * 0.01 - 100.0
                target.rcs = data[6] - 100  # dBm
                target.snr = data[7]
            
            return target
            
        except Exception as e:
            self._logger.debug(f"CAN message parsing error: {e}")
            return None
    
    def _acquire_simulation_data(self) -> RadarData:
        """采集仿真雷达数据"""
        # 生成随机目标
        num_targets = np.random.randint(0, 10)
        targets = []
        
        for _ in range(num_targets):
            target = RadarTarget(
                range_distance=np.random.uniform(5, self._max_range),
                azimuth=np.random.uniform(-self._fov/2, self._fov/2),
                elevation=np.random.uniform(-5, 5),
                velocity=np.random.uniform(-30, 30),
                rcs=np.random.uniform(-10, 30),
                snr=np.random.uniform(10, 50)
            )
            targets.append(target)
        
        radar_data = RadarData(
            timestamp=Timestamp.now(),
            radar_id=self._sensor_id,
            targets=targets,
            extrinsics=self._config.extrinsics
        )
        
        return radar_data
    
    def _get_topic(self) -> str:
        """获取发布主题"""
        return "sensor/radar"
    
    def targets_to_cartesian(self, radar_data: RadarData) -> np.ndarray:
        """将极坐标目标转换为笛卡尔坐标"""
        points = []
        
        for target in radar_data.targets:
            # 极坐标转笛卡尔
            x = target.range_distance * np.cos(np.radians(target.elevation)) * \
                np.cos(np.radians(target.azimuth))
            y = target.range_distance * np.cos(np.radians(target.elevation)) * \
                np.sin(np.radians(target.azimuth))
            z = target.range_distance * np.sin(np.radians(target.elevation))
            
            points.append([x, y, z, target.velocity, target.rcs])
        
        return np.array(points) if points else np.zeros((0, 5))
    
    def filter_targets(self, radar_data: RadarData,
                      range_limit: Optional[tuple] = None,
                      rcs_limit: Optional[tuple] = None,
                      snr_limit: Optional[tuple] = None) -> RadarData:
        """过滤雷达目标"""
        filtered_targets = []
        
        for target in radar_data.targets:
            # 距离过滤
            if range_limit and not (range_limit[0] <= target.range_distance <= range_limit[1]):
                continue
            
            # RCS过滤
            if rcs_limit and not (rcs_limit[0] <= target.rcs <= rcs_limit[1]):
                continue
            
            # SNR过滤
            if snr_limit and not (snr_limit[0] <= target.snr <= snr_limit[1]):
                continue
            
            filtered_targets.append(target)
        
        return RadarData(
            timestamp=radar_data.timestamp,
            radar_id=radar_data.radar_id,
            targets=filtered_targets,
            extrinsics=radar_data.extrinsics
        )
