"""
自动驾驶系统 - 枚举类型定义
"""

from enum import Enum, auto


class SensorType(Enum):
    """传感器类型"""
    CAMERA = auto()
    LIDAR = auto()
    RADAR = auto()
    ULTRASONIC = auto()
    GNSS = auto()
    IMU = auto()
    CAN = auto()


class CameraType(Enum):
    """摄像头类型"""
    FRONT_LONG = auto()  # 前视长焦
    FRONT_WIDE = auto()  # 前视广角
    REAR = auto()  # 后视
    SIDE_LEFT = auto()  # 左侧
    SIDE_RIGHT = auto()  # 右侧
    FISHEYE_FRONT = auto()  # 前鱼眼
    FISHEYE_REAR = auto()  # 后鱼眼
    FISHEYE_LEFT = auto()  # 左鱼眼
    FISHEYE_RIGHT = auto()  # 右鱼眼
    SURROUND = auto()  # 环视


class LaneType(Enum):
    """车道线类型"""
    UNKNOWN = auto()
    SOLID_WHITE = auto()
    SOLID_YELLOW = auto()
    DASHED_WHITE = auto()
    DASHED_YELLOW = auto()
    DOUBLE_SOLID = auto()
    DOUBLE_DASHED = auto()
    SOLID_DASHED = auto()
    DASHED_SOLID = auto()
    ROAD_EDGE = auto()
    CURB = auto()


class ObstacleType(Enum):
    """障碍物类型"""
    UNKNOWN = auto()
    VEHICLE = auto()
    VEHICLE_CAR = auto()
    VEHICLE_TRUCK = auto()
    VEHICLE_BUS = auto()
    VEHICLE_BIKE = auto()
    VEHICLE_MOTORCYCLE = auto()
    PEDESTRIAN = auto()
    PEDESTRIAN_ADULT = auto()
    PEDESTRIAN_CHILD = auto()
    CYCLIST = auto()
    TRAFFIC_CONE = auto()
    TRAFFIC_SIGN = auto()
    TRAFFIC_LIGHT = auto()
    BARRIER = auto()
    ANIMAL = auto()


class BehaviorType(Enum):
    """行为类型"""
    CRUISE = auto()  # 巡航
    FOLLOW = auto()  # 跟车
    OVERTAKE = auto()  # 超车
    LANE_CHANGE_LEFT = auto()  # 向左变道
    LANE_CHANGE_RIGHT = auto()  # 向右变道
    TURN_LEFT = auto()  # 左转
    TURN_RIGHT = auto()  # 右转
    U_TURN = auto()  # 掉头
    STOP = auto()  # 停车
    EMERGENCY_STOP = auto()  # 紧急停车
    YIELD = auto()  # 让行
    PULL_OVER = auto()  # 靠边停车


class ModuleState(Enum):
    """模块状态"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class LogLevel(Enum):
    """日志级别"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class CoordinateType(Enum):
    """坐标系类型"""
    VEHICLE = auto()  # 车辆坐标系
    CAMERA = auto()  # 相机坐标系
    LIDAR = auto()  # LiDAR坐标系
    WORLD = auto()  # 世界坐标系
    ENU = auto()  # 东北天坐标系


class TrajectoryType(Enum):
    """轨迹类型"""
    NORMAL = auto()
    EMERGENCY = auto()
    FALLBACK = auto()
    LANE_FOLLOW = auto()
    LANE_CHANGE = auto()
    OVERTAKE = auto()


class TrafficLightState(Enum):
    """交通灯状态"""
    UNKNOWN = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    FLASHING_RED = auto()
    FLASHING_YELLOW = auto()
    OFF = auto()


class RoadType(Enum):
    """道路类型"""
    UNKNOWN = auto()
    HIGHWAY = auto()
    CITY_ROAD = auto()
    RURAL_ROAD = auto()
    PARKING_LOT = auto()
    INTERSECTION = auto()
    ROUNDABOUT = auto()
    TUNNEL = auto()
    BRIDGE = auto()


class WeatherType(Enum):
    """天气类型"""
    UNKNOWN = auto()
    SUNNY = auto()
    CLOUDY = auto()
    RAINY = auto()
    SNOWY = auto()
    FOGGY = auto()
    NIGHT = auto()
