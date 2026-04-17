"""
传感器测试模块
提供传感器数据接入模块的测试接口
"""

import time
import sys
from typing import Dict, Any
import numpy as np

from .core.sensor_manager import SensorManager, create_sensor_manager
from .core.sync_manager import SyncManager, SyncConfig, SyncedFrame
from .core.data_publisher import DataPublisher, get_global_publisher, DataPriority
from .core.preprocessor import DataPreprocessor, ImagePreprocessConfig, PointCloudPreprocessConfig
from .drivers.base_sensor import SensorData, SensorType, ImageData, PointCloudData, RadarData, UltrasonicData
from .adapters.nuscenes_adapter import NuScenesAdapter, NuScenesConfig


def test_camera_driver():
    """测试摄像头驱动"""
    print("\n" + "="*60)
    print("Testing Camera Driver")
    print("="*60)
    
    from .drivers.camera_driver import CameraDriver, CameraConfig, CAMERA_PRESETS
    
    # 使用预设配置
    config = CAMERA_PRESETS['front_long']
    
    # 创建驱动
    camera = CameraDriver(config)
    
    # 初始化
    if not camera.initialize():
        print("Failed to initialize camera")
        return False
    
    print(f"Camera initialized: {camera.name}")
    print(f"  Resolution: {camera.camera_config.resolution}")
    print(f"  FPS: {camera.camera_config.fps}")
    print(f"  FOV: {camera.camera_config.fov}")
    
    # 采集测试
    print("\nCapturing frames...")
    camera.start()
    
    for i in range(3):
        time.sleep(0.1)
        data = camera.get_latest_data()
        if data is not None and isinstance(data, ImageData):
            print(f"  Frame {i+1}: {data.width}x{data.height}, "
                  f"channels={data.channels}, timestamp={data.timestamp:.3f}")
    
    camera.stop()
    camera.release()
    
    print("Camera test passed!")
    return True


def test_lidar_driver():
    """测试LiDAR驱动"""
    print("\n" + "="*60)
    print("Testing LiDAR Driver")
    print("="*60)
    
    from .drivers.lidar_driver import LiDARDriver, LiDARConfig, LIDAR_PRESETS
    
    config = LIDAR_PRESETS['main_lidar']
    
    lidar = LiDARDriver(config)
    
    if not lidar.initialize():
        print("Failed to initialize LiDAR")
        return False
    
    print(f"LiDAR initialized: {lidar.name}")
    print(f"  Model: {lidar.lidar_config.model}")
    print(f"  Channels: {lidar.lidar_config.channels}")
    print(f"  Range: {lidar.lidar_config.range_max}m")
    
    # 采集测试
    print("\nCapturing point clouds...")
    lidar.start()
    
    for i in range(3):
        time.sleep(0.1)
        data = lidar.get_latest_data()
        if data is not None and isinstance(data, PointCloudData):
            print(f"  Point cloud {i+1}: {data.num_points} points, "
                  f"timestamp={data.timestamp:.3f}")
    
    lidar.stop()
    lidar.release()
    
    print("LiDAR test passed!")
    return True


def test_radar_driver():
    """测试雷达驱动"""
    print("\n" + "="*60)
    print("Testing Radar Driver")
    print("="*60)
    
    from .drivers.radar_driver import RadarDriver, RadarConfig, RADAR_PRESETS
    
    config = RADAR_PRESETS['front_radar']
    
    radar = RadarDriver(config)
    
    if not radar.initialize():
        print("Failed to initialize radar")
        return False
    
    print(f"Radar initialized: {radar.name}")
    print(f"  Model: {radar.radar_config.model}")
    print(f"  Range: {radar.radar_config.range_max}m")
    print(f"  CAN ID: 0x{radar.radar_config.can_id:03X}")
    
    # 采集测试
    print("\nCapturing radar data...")
    radar.start()
    
    for i in range(3):
        time.sleep(0.1)
        data = radar.get_latest_data()
        if data is not None and isinstance(data, RadarData):
            print(f"  Radar frame {i+1}: {data.num_targets} targets, "
                  f"timestamp={data.timestamp:.3f}")
            if data.targets:
                target = data.targets[0]
                print(f"    Target 0: range={target.range:.2f}m, "
                      f"azimuth={target.azimuth:.2f}deg, "
                      f"velocity={target.velocity:.2f}m/s")
    
    radar.stop()
    radar.release()
    
    print("Radar test passed!")
    return True


def test_ultrasonic_driver():
    """测试超声波驱动"""
    print("\n" + "="*60)
    print("Testing Ultrasonic Driver")
    print("="*60)
    
    from .drivers.ultrasonic_driver import UltrasonicDriver, UltrasonicConfig, ULTRASONIC_PRESETS
    
    config = ULTRASONIC_PRESETS['front_left_1']
    
    ultrasonic = UltrasonicDriver(config)
    
    if not ultrasonic.initialize():
        print("Failed to initialize ultrasonic")
        return False
    
    print(f"Ultrasonic initialized: {ultrasonic.name}")
    print(f"  Range: {ultrasonic.ultrasonic_config.range_min}m - "
          f"{ultrasonic.ultrasonic_config.range_max}m")
    print(f"  SPI Channel: {ultrasonic.ultrasonic_config.spi_channel}")
    
    # 采集测试
    print("\nCapturing ultrasonic data...")
    ultrasonic.start()
    
    for i in range(5):
        time.sleep(0.05)
        data = ultrasonic.capture()
        if data is not None and isinstance(data, UltrasonicData):
            print(f"  Measurement {i+1}: distance={data.distance:.2f}m, "
                  f"confidence={data.confidence:.2f}")
    
    ultrasonic.stop()
    ultrasonic.release()
    
    print("Ultrasonic test passed!")
    return True


def test_sync_manager():
    """测试同步管理器"""
    print("\n" + "="*60)
    print("Testing Sync Manager")
    print("="*60)
    
    sync_config = SyncConfig(
        mode="software",
        master_clock="lidar",
        time_tolerance_ms=50.0,
        sync_period_ms=100.0
    )
    
    sync_manager = SyncManager(sync_config)
    
    # 注册传感器
    sensors = ["camera_front", "lidar_main", "radar_front"]
    for sensor in sensors:
        sync_manager.register_sensor(sensor, "perception_group")
    
    print(f"Registered {len(sensors)} sensors to sync group")
    
    # 添加模拟数据
    from .drivers.base_sensor import SensorType
    
    for i, sensor_name in enumerate(sensors):
        data = SensorData(
            timestamp=time.time() + i * 0.01,
            sensor_name=sensor_name,
            sensor_type=SensorType.CAMERA
        )
        sync_manager.add_data(sensor_name, data)
    
    # 注册同步回调
    synced_frames = []
    
    def on_sync(frame: SyncedFrame):
        synced_frames.append(frame)
        print(f"  Synced frame {frame.frame_id}: {len(frame.data)} sensors, "
              f"complete={frame.is_complete}")
    
    sync_manager.register_sync_callback(on_sync)
    
    # 启动同步
    sync_manager.start()
    time.sleep(0.3)
    sync_manager.stop()
    
    # 检查统计
    stats = sync_manager.get_sync_statistics()
    print(f"\nSync statistics:")
    print(f"  Frame count: {stats['frame_count']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    
    print("Sync Manager test passed!")
    return True


def test_data_publisher():
    """测试数据发布器"""
    print("\n" + "="*60)
    print("Testing Data Publisher")
    print("="*60)
    
    publisher = get_global_publisher()
    
    # 订阅测试
    received_messages = []
    
    def callback(data: SensorData):
        received_messages.append(data)
        print(f"  Received: {data.sensor_name} at {data.timestamp:.3f}")
    
    subscriber = publisher.subscribe("test/topic", callback)
    
    # 发布测试
    print("\nPublishing messages...")
    
    from .drivers.base_sensor import SensorType
    
    for i in range(3):
        data = SensorData(
            timestamp=time.time(),
            sensor_name=f"sensor_{i}",
            sensor_type=SensorType.CAMERA
        )
        count = publisher.publish("test/topic", data)
        print(f"  Published message {i+1}, delivered to {count} subscribers")
        time.sleep(0.05)
    
    time.sleep(0.1)
    
    # 取消订阅
    publisher.unsubscribe(subscriber)
    
    print(f"\nReceived {len(received_messages)} messages")
    
    print("Data Publisher test passed!")
    return True


def test_preprocessor():
    """测试预处理器"""
    print("\n" + "="*60)
    print("Testing Data Preprocessor")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    # 测试图像预处理
    print("\nTesting image preprocessing...")
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    image_data = ImageData(
        timestamp=time.time(),
        sensor_name="test_camera",
        sensor_type=SensorType.CAMERA,
        data=image,
        image=image,
        width=1920,
        height=1080,
        channels=3,
        encoding="rgb8"
    )
    
    config = ImagePreprocessConfig(
        target_size=(960, 540),
        normalize=True
    )
    preprocessor.set_image_config(config)
    
    processed = preprocessor.preprocess_image(image_data)
    print(f"  Original: {image_data.width}x{image_data.height}")
    print(f"  Processed: {processed.width}x{processed.height}")
    print(f"  Image dtype: {processed.image.dtype}")
    
    # 测试点云预处理
    print("\nTesting point cloud preprocessing...")
    num_points = 10000
    points = np.random.randn(num_points, 3) * 50
    
    pc_data = PointCloudData(
        timestamp=time.time(),
        sensor_name="test_lidar",
        sensor_type=SensorType.LIDAR,
        data=points,
        points=points,
        num_points=num_points
    )
    
    pc_config = PointCloudPreprocessConfig(
        range_filter=(1.0, 100.0),
        max_points=5000
    )
    preprocessor.set_pointcloud_config(pc_config)
    
    processed_pc = preprocessor.preprocess_pointcloud(pc_data)
    print(f"  Original: {pc_data.num_points} points")
    print(f"  Processed: {processed_pc.num_points} points")
    
    print("Preprocessor test passed!")
    return True


def test_nuscenes_adapter():
    """测试nuScenes适配器"""
    print("\n" + "="*60)
    print("Testing nuScenes Adapter")
    print("="*60)
    
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-trainval"
    )
    
    adapter = NuScenesAdapter(config)
    
    if not adapter.initialize():
        print("nuScenes not available, using simulation mode")
    
    # 获取数据
    print("\nLoading sample data...")
    data = adapter.next_sample()
    
    if data:
        print(f"Loaded {len(data)} sensor data:")
        for name, sensor_data in data.items():
            if isinstance(sensor_data, ImageData):
                print(f"  {name}: Image {sensor_data.width}x{sensor_data.height}")
            elif isinstance(sensor_data, PointCloudData):
                print(f"  {name}: PointCloud {sensor_data.num_points} points")
            elif isinstance(sensor_data, RadarData):
                print(f"  {name}: Radar {sensor_data.num_targets} targets")
    
    print("nuScenes Adapter test passed!")
    return True


def test_sensor_manager():
    """测试传感器管理器"""
    print("\n" + "="*60)
    print("Testing Sensor Manager")
    print("="*60)
    
    # 创建管理器
    manager = create_sensor_manager()
    
    # 初始化所有传感器
    print("\nInitializing all sensors...")
    manager.initialize_all()
    
    # 获取状态
    status = manager.get_sensor_status()
    print(f"\nSensor status:")
    print(f"  Running: {status['running']}")
    print(f"  Cameras: {len(status['cameras'])}")
    print(f"  LiDARs: {len(status['lidars'])}")
    print(f"  Radars: {len(status['radars'])}")
    print(f"  Ultrasonics: {len(status['ultrasonics'])}")
    
    # 启动传感器
    print("\nStarting all sensors...")
    manager.start_all()
    
    # 运行一段时间
    print("\nRunning for 0.5 seconds...")
    time.sleep(0.5)
    
    # 获取数据
    data = manager.get_all_data()
    print(f"\nCollected {len(data)} sensor data")
    
    # 停止传感器
    print("\nStopping all sensors...")
    manager.stop_all()
    
    # 释放资源
    manager.release_all()
    
    print("Sensor Manager test passed!")
    return True


def test_sensors():
    """
    运行所有传感器测试
    
    Returns:
        Dict[str, bool]: 测试结果
    """
    print("\n" + "="*60)
    print("SENSOR MODULE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # 运行各项测试
    tests = [
        ("Camera Driver", test_camera_driver),
        ("LiDAR Driver", test_lidar_driver),
        ("Radar Driver", test_radar_driver),
        ("Ultrasonic Driver", test_ultrasonic_driver),
        ("Sync Manager", test_sync_manager),
        ("Data Publisher", test_data_publisher),
        ("Preprocessor", test_preprocessor),
        ("nuScenes Adapter", test_nuscenes_adapter),
        ("Sensor Manager", test_sensor_manager),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n{test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 打印测试结果汇总
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    # 运行测试
    results = test_sensors()
    
    # 退出码
    sys.exit(0 if all(results.values()) else 1)
