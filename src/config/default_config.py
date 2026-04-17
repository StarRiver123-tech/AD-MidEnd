"""
自动驾驶系统 - 默认配置
提供系统默认配置
"""

import numpy as np
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "system_name": "AutonomousDrivingSystem",
        "version": "1.0.0",
        
        # 日志配置
        "log_level": "INFO",
        "log_path": "./logs",
        "log_to_console": True,
        "log_to_file": True,
        
        # 仿真配置
        "simulation_mode": False,
        "simulation_dataset": "nuscenes",
        "simulation_speed": 1.0,
        
        # 传感器配置
        "sensors": {
            # 前视摄像头
            "camera_front_long": {
                "sensor_type": "camera",
                "camera_type": "front_long",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video0",
                "resolution": [1920, 1080],
                "fps": 30,
                "intrinsics": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {
                    "exposure": "auto",
                    "gain": "auto"
                }
            },
            "camera_front_wide": {
                "sensor_type": "camera",
                "camera_type": "front_wide",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video1",
                "resolution": [1920, 1080],
                "fps": 30,
                "intrinsics": [
                    [800.0, 0.0, 960.0],
                    [0.0, 800.0, 540.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.1, 0.01, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            # 后视摄像头
            "camera_rear": {
                "sensor_type": "camera",
                "camera_type": "rear",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video2",
                "resolution": [1920, 1080],
                "fps": 30,
                "intrinsics": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [-1.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            # 周视摄像头
            "camera_left": {
                "sensor_type": "camera",
                "camera_type": "side_left",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video3",
                "resolution": [1920, 1080],
                "fps": 30,
                "intrinsics": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]
                ],
                "extrinsics": [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 1.0],
                    [-1.0, 0.0, 0.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "camera_right": {
                "sensor_type": "camera",
                "camera_type": "side_right",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video4",
                "resolution": [1920, 1080],
                "fps": 30,
                "intrinsics": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]
                ],
                "extrinsics": [
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -1.0],
                    [1.0, 0.0, 0.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            # 环视鱼眼摄像头
            "camera_fisheye_front": {
                "sensor_type": "camera",
                "camera_type": "fisheye_front",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video5",
                "resolution": [1280, 720],
                "fps": 30,
                "intrinsics": [
                    [400.0, 0.0, 640.0],
                    [0.0, 400.0, 360.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.3, 0.1, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "camera_fisheye_rear": {
                "sensor_type": "camera",
                "camera_type": "fisheye_rear",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video6",
                "resolution": [1280, 720],
                "fps": 30,
                "intrinsics": [
                    [400.0, 0.0, 640.0],
                    [0.0, 400.0, 360.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.3, 0.1, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [-1.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "camera_fisheye_left": {
                "sensor_type": "camera",
                "camera_type": "fisheye_left",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video7",
                "resolution": [1280, 720],
                "fps": 30,
                "intrinsics": [
                    [400.0, 0.0, 640.0],
                    [0.0, 400.0, 360.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.3, 0.1, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 1.0],
                    [-1.0, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "camera_fisheye_right": {
                "sensor_type": "camera",
                "camera_type": "fisheye_right",
                "enabled": True,
                "interface": "gmsl",
                "device_path": "/dev/video8",
                "resolution": [1280, 720],
                "fps": 30,
                "intrinsics": [
                    [400.0, 0.0, 640.0],
                    [0.0, 400.0, 360.0],
                    [0.0, 0.0, 1.0]
                ],
                "distortion": [0.3, 0.1, 0.0, 0.0, 0.0],
                "extrinsics": [
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -1.0],
                    [1.0, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            # LiDAR
            "lidar_main": {
                "sensor_type": "lidar",
                "enabled": True,
                "interface": "ethernet",
                "ip_address": "192.168.1.201",
                "port": 2368,
                "extrinsics": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.8],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {
                    "channels": 128,
                    "range": 200.0,
                    "frequency": 10.0
                }
            },
            # 雷达
            "radar_front": {
                "sensor_type": "radar",
                "enabled": True,
                "interface": "can",
                "can_channel": "can0",
                "can_id": 0x200,
                "extrinsics": [
                    [1.0, 0.0, 0.0, 3.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {
                    "max_range": 150.0,
                    "fov": 30.0
                }
            },
            "radar_rear": {
                "sensor_type": "radar",
                "enabled": True,
                "interface": "can",
                "can_channel": "can0",
                "can_id": 0x201,
                "extrinsics": [
                    [-1.0, 0.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "radar_front_left": {
                "sensor_type": "radar",
                "enabled": True,
                "interface": "can",
                "can_channel": "can1",
                "can_id": 0x202,
                "extrinsics": [
                    [0.707, 0.707, 0.0, 2.5],
                    [-0.707, 0.707, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "radar_front_right": {
                "sensor_type": "radar",
                "enabled": True,
                "interface": "can",
                "can_channel": "can1",
                "can_id": 0x203,
                "extrinsics": [
                    [0.707, -0.707, 0.0, 2.5],
                    [0.707, 0.707, 0.0, -1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            "radar_rear_left": {
                "sensor_type": "radar",
                "enabled": True,
                "interface": "can",
                "can_channel": "can1",
                "can_id": 0x204,
                "extrinsics": [
                    [-0.707, 0.707, 0.0, -0.5],
                    [-0.707, -0.707, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {}
            },
            # 超声波
            "ultrasonic_front_center": {
                "sensor_type": "ultrasonic",
                "enabled": True,
                "interface": "spi",
                "spi_channel": 0,
                "extrinsics": [
                    [1.0, 0.0, 0.0, 3.8],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "parameters": {
                    "max_range": 5.0
                }
            },
            # CAN总线
            "can_vehicle": {
                "sensor_type": "can",
                "enabled": True,
                "interface": "can",
                "can_channel": "can0",
                "bitrate": 500000,
                "parameters": {}
            }
        },
        
        # 模块配置
        "modules": {
            "sensor_manager": {
                "enabled": True,
                "execution_frequency": 100.0,
                "timeout_ms": 10.0,
                "input_topics": [],
                "output_topics": [
                    "sensor/camera/front",
                    "sensor/camera/rear",
                    "sensor/lidar",
                    "sensor/radar",
                    "sensor/ultrasonic",
                    "sensor/can/vehicle"
                ],
                "algorithm_params": {}
            },
            "perception": {
                "enabled": True,
                "execution_frequency": 10.0,
                "timeout_ms": 100.0,
                "input_topics": [
                    "sensor/camera/front",
                    "sensor/camera/rear",
                    "sensor/lidar",
                    "sensor/radar"
                ],
                "output_topics": [
                    "perception/lane",
                    "perception/obstacle",
                    "perception/occupancy",
                    "perception/fusion"
                ],
                "algorithm_params": {
                    "lane_detection": {
                        "confidence_threshold": 0.5,
                        "max_detection_distance": 100.0
                    },
                    "obstacle_detection": {
                        "confidence_threshold": 0.6,
                        "max_detection_distance": 150.0,
                        "min_detection_distance": 0.5
                    },
                    "occupancy": {
                        "resolution": 0.2,
                        "range_x": [-50, 50],
                        "range_y": [-30, 30],
                        "range_z": [-3, 3]
                    }
                }
            },
            "planning": {
                "enabled": True,
                "execution_frequency": 10.0,
                "timeout_ms": 100.0,
                "input_topics": [
                    "perception/lane",
                    "perception/obstacle",
                    "perception/occupancy",
                    "sensor/can/vehicle"
                ],
                "output_topics": [
                    "planning/trajectory",
                    "planning/behavior"
                ],
                "algorithm_params": {
                    "trajectory_generation": {
                        "num_trajectories": 5,
                        "time_horizon": 8.0,
                        "time_resolution": 0.1
                    },
                    "trajectory_optimization": {
                        "max_iterations": 100,
                        "convergence_threshold": 1e-3
                    },
                    "behavior_planning": {
                        "reaction_time": 1.0,
                        "comfort_deceleration": 2.0,
                        "emergency_deceleration": 6.0
                    }
                }
            },
            "control": {
                "enabled": True,
                "execution_frequency": 100.0,
                "timeout_ms": 10.0,
                "input_topics": [
                    "planning/trajectory",
                    "sensor/can/vehicle"
                ],
                "output_topics": [
                    "control/command"
                ],
                "algorithm_params": {
                    "lateral_control": {
                        "kp": 1.0,
                        "ki": 0.1,
                        "kd": 0.01
                    },
                    "longitudinal_control": {
                        "kp": 2.0,
                        "ki": 0.5,
                        "kd": 0.1
                    }
                }
            },
            "visualization": {
                "enabled": True,
                "execution_frequency": 30.0,
                "timeout_ms": 33.0,
                "input_topics": [
                    "sensor/camera/front",
                    "perception/lane",
                    "perception/obstacle",
                    "planning/trajectory"
                ],
                "output_topics": [],
                "algorithm_params": {
                    "window_size": [1920, 1080],
                    "show_lidar_points": True,
                    "show_trajectory": True
                }
            }
        },
        
        # 其他参数
        "parameters": {
            "vehicle_params": {
                "wheelbase": 2.8,
                "track_width": 1.8,
                "max_steering_angle": 35.0,
                "max_acceleration": 3.0,
                "max_deceleration": 6.0,
                "max_speed": 120.0
            },
            "safety_params": {
                "min_follow_distance": 3.0,
                "emergency_stop_distance": 1.0,
                "pedestrian_safety_distance": 2.0
            }
        }
    }
