"""
传感器数据接入模块
==================

提供自动驾驶系统中各类传感器的驱动接口、数据同步、预处理功能。
支持真实硬件和nuScenes数据集仿真输入。

主要组件:
- SensorManager: 传感器管理器
- SyncManager: 数据同步管理器
- DataPublisher: 数据发布/订阅机制
- 各类传感器驱动: Camera, LiDAR, Radar, Ultrasonic, CAN
"""

from .core.sensor_manager import SensorManager
from .core.sync_manager import SyncManager
from .core.data_publisher import DataPublisher
from .core.preprocessor import DataPreprocessor
from .adapters.nuscenes_adapter import NuScenesAdapter

__version__ = "1.0.0"
__all__ = [
    "SensorManager",
    "SyncManager", 
    "DataPublisher",
    "DataPreprocessor",
    "NuScenesAdapter",
]
