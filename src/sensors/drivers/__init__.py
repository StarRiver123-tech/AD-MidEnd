"""
传感器驱动模块
"""

from .base_sensor import (
    BaseSensor,
    SensorConfig,
    SensorData,
    SensorState,
    SensorType,
    ImageData,
    PointCloudData,
    RadarData,
    RadarTarget,
    UltrasonicData,
    VehicleStateData,
)

from .camera_driver import (
    CameraDriver,
    CameraConfig,
    CameraArray,
    CAMERA_PRESETS,
)

from .lidar_driver import (
    LiDARDriver,
    LiDARConfig,
    LiDARArray,
    LIDAR_PRESETS,
)

from .radar_driver import (
    RadarDriver,
    RadarConfig,
    RadarArray,
    RADAR_PRESETS,
)

from .ultrasonic_driver import (
    UltrasonicDriver,
    UltrasonicConfig,
    UltrasonicArray,
    ULTRASONIC_PRESETS,
)

from .can_driver import (
    CANDriver,
    CANConfig,
    CANSignal,
    VehicleStateManager,
    CAN_PRESETS,
    VEHICLE_SIGNALS,
)

__all__ = [
    # 基础类
    'BaseSensor',
    'SensorConfig',
    'SensorData',
    'SensorState',
    'SensorType',
    'ImageData',
    'PointCloudData',
    'RadarData',
    'RadarTarget',
    'UltrasonicData',
    'VehicleStateData',
    # 摄像头
    'CameraDriver',
    'CameraConfig',
    'CameraArray',
    'CAMERA_PRESETS',
    # LiDAR
    'LiDARDriver',
    'LiDARConfig',
    'LiDARArray',
    'LIDAR_PRESETS',
    # 雷达
    'RadarDriver',
    'RadarConfig',
    'RadarArray',
    'RADAR_PRESETS',
    # 超声波
    'UltrasonicDriver',
    'UltrasonicConfig',
    'UltrasonicArray',
    'ULTRASONIC_PRESETS',
    # CAN
    'CANDriver',
    'CANConfig',
    'CANSignal',
    'VehicleStateManager',
    'CAN_PRESETS',
    'VEHICLE_SIGNALS',
]
