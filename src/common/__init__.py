"""
自动驾驶系统 - 通用模块
包含统一的数据结构、枚举类型和工具函数
"""

from .data_types import *
from .enums import *
from .geometry import *

__all__ = [
    # 数据类型
    'Timestamp', 'Vector3D', 'Quaternion', 'Pose', 'BoundingBox3D',
    'ImageData', 'PointCloud', 'RadarData', 'UltrasonicData', 'CANData',
    'LaneLine', 'LaneDetectionResult', 'Obstacle', 'ObstacleDetectionResult',
    'OccupancyGrid', 'OccupancyResult', 'PerceptionResult',
    'TrajectoryPoint', 'Trajectory', 'BehaviorType', 'PlanningResult',
    'SensorConfig', 'ModuleConfig', 'SystemConfig',
    # 枚举类型
    'SensorType', 'LaneType', 'ObstacleType', 'BehaviorType',
    'ModuleState', 'LogLevel', 'CoordinateType',
    # 几何工具
    'transform_point', 'transform_pose', 'calculate_distance',
    'calculate_iou_3d', 'interpolate_trajectory'
]
