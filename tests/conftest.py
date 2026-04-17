"""
Pytest配置文件
提供测试fixtures
"""

import pytest
import numpy as np
from typing import List, Dict

# 导入数据生成器
from data_generators.sensor_data_generator import (
    SensorDataGenerator, SensorType, PointCloud, ImageData, 
    RadarData, IMUData, GPSData
)
from data_generators.object_data_generator import (
    ObjectDataGenerator, ObjectType, ObjectStatus, 
    TrackedObject, ObjectDetection
)
from data_generators.lane_data_generator import (
    LaneDataGenerator, LaneType, LanePosition,
    LaneLine, LanePoint, LaneDetectionResult
)

# 导入评估指标
from metrics.perception_metrics import PerceptionMetrics, DetectionResult
from metrics.planning_metrics import PlanningMetrics, Trajectory, TrajectoryPoint


# ==================== Fixtures for Data Generators ====================

@pytest.fixture(scope="session")
def sensor_generator():
    """传感器数据生成器fixture"""
    return SensorDataGenerator(seed=42)


@pytest.fixture(scope="session")
def object_generator():
    """目标数据生成器fixture"""
    return ObjectDataGenerator(seed=42)


@pytest.fixture(scope="session")
def lane_generator():
    """车道线数据生成器fixture"""
    return LaneDataGenerator(seed=42)


# ==================== Fixtures for Metrics ====================

@pytest.fixture(scope="session")
def perception_metrics():
    """感知指标计算器fixture"""
    return PerceptionMetrics(iou_threshold=0.5)


@pytest.fixture(scope="session")
def planning_metrics():
    """规划指标计算器fixture"""
    return PlanningMetrics()


# ==================== Fixtures for Sensor Data ====================

@pytest.fixture
def sample_lidar_pointcloud(sensor_generator):
    """示例激光雷达点云数据"""
    return sensor_generator.generate_lidar_pointcloud(num_points=1000)


@pytest.fixture
def sample_camera_image(sensor_generator):
    """示例摄像头图像数据"""
    return sensor_generator.generate_camera_image()


@pytest.fixture
def sample_radar_data(sensor_generator):
    """示例雷达数据"""
    return sensor_generator.generate_radar_data(num_targets=5)


@pytest.fixture
def sample_imu_data(sensor_generator):
    """示例IMU数据"""
    return sensor_generator.generate_imu_data(duration=1.0)[0]


@pytest.fixture
def sample_gps_data(sensor_generator):
    """示例GPS数据"""
    return sensor_generator.generate_gps_data(duration=1.0)[0]


@pytest.fixture
def sample_sensor_suite(sensor_generator):
    """示例传感器套件数据"""
    return sensor_generator.generate_sensor_suite()


# ==================== Fixtures for Object Data ====================

@pytest.fixture
def sample_vehicle(object_generator):
    """示例车辆目标"""
    return object_generator.generate_single_object(
        object_type=ObjectType.VEHICLE,
        position=(10, 5, 0),
        velocity=(5, 0, 0)
    )


@pytest.fixture
def sample_pedestrian(object_generator):
    """示例行人目标"""
    return object_generator.generate_single_object(
        object_type=ObjectType.PEDESTRIAN,
        position=(15, 3, 0),
        velocity=(1, 0.5, 0)
    )


@pytest.fixture
def sample_object_set(object_generator):
    """示例目标集合"""
    return object_generator.generate_object_set(num_objects=10)


@pytest.fixture
def sample_object_trajectory(object_generator):
    """示例目标轨迹"""
    return object_generator.generate_object_trajectory(
        object_type=ObjectType.VEHICLE,
        duration=5.0,
        motion_pattern="straight"
    )


# ==================== Fixtures for Lane Data ====================

@pytest.fixture
def sample_straight_lane(lane_generator):
    """示例直线车道线"""
    return lane_generator.generate_straight_lane(
        lane_type=LaneType.DASHED_WHITE,
        position=LanePosition.CENTER
    )


@pytest.fixture
def sample_curved_lane(lane_generator):
    """示例曲线车道线"""
    return lane_generator.generate_curved_lane(
        curvature=0.01,
        lane_type=LaneType.SOLID_WHITE,
        position=LanePosition.LEFT
    )


@pytest.fixture
def sample_lane_segment(lane_generator):
    """示例车道段"""
    return lane_generator.generate_lane_segment(
        num_lanes=3,
        curvature=0.0
    )


# ==================== Fixtures for Detection Results ====================

@pytest.fixture
def sample_detection_results(object_generator):
    """示例检测结果"""
    ground_truth = object_generator.generate_object_set(num_objects=5)
    detections = object_generator.generate_detection_results(ground_truth)
    return ground_truth, detections


# ==================== Fixtures for Trajectory ====================

@pytest.fixture
def sample_straight_trajectory():
    """示例直线轨迹"""
    points = []
    for i in range(50):
        t = i * 0.1
        point = TrajectoryPoint(
            x=t * 10,
            y=0.0,
            heading=0.0,
            velocity=10.0,
            acceleration=0.0,
            curvature=0.0,
            timestamp=t
        )
        points.append(point)
    return Trajectory(points=points)


@pytest.fixture
def sample_curved_trajectory():
    """示例曲线轨迹"""
    points = []
    radius = 50.0
    velocity = 10.0
    
    for i in range(50):
        t = i * 0.1
        angle = velocity * t / radius
        
        point = TrajectoryPoint(
            x=radius * np.sin(angle),
            y=radius * (1 - np.cos(angle)),
            heading=angle,
            velocity=velocity,
            acceleration=0.0,
            curvature=1.0 / radius,
            timestamp=t
        )
        points.append(point)
    
    return Trajectory(points=points)


@pytest.fixture
def sample_obstacles():
    """示例障碍物"""
    from metrics.planning_metrics import Obstacle
    return [
        Obstacle(x=20, y=2, radius=1.5, velocity_x=-5, velocity_y=0),
        Obstacle(x=40, y=-2, radius=2.0, velocity_x=0, velocity_y=3),
    ]


# ==================== Fixtures for Test Configuration ====================

@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        'sensor_range': 100.0,
        'detection_range': 80.0,
        'safety_distance': 3.0,
        'max_comfort_acceleration': 2.0,
        'max_comfort_jerk': 2.0,
        'iou_threshold': 0.5,
        'confidence_threshold': 0.5,
    }


# ==================== Custom Markers ====================

def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "simulation: 仿真测试")
    config.addinivalue_line("markers", "regression: 回归测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "perception: 感知模块测试")
    config.addinivalue_line("markers", "planning: 规划模块测试")
    config.addinivalue_line("markers", "sensor: 传感器测试")
