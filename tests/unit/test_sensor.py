"""
传感器接入单元测试
测试各类传感器数据生成和处理
"""

import pytest
import numpy as np

from data_generators.sensor_data_generator import (
    SensorDataGenerator, SensorType, PointCloud, ImageData,
    RadarData, IMUData, GPSData
)


@pytest.mark.unit
@pytest.mark.sensor
class TestLidarSensor:
    """激光雷达传感器单元测试"""
    
    def test_pointcloud_generation(self, sensor_generator):
        """测试点云数据生成"""
        pointcloud = sensor_generator.generate_lidar_pointcloud(
            num_points=1000,
            range_m=100.0
        )
        
        assert isinstance(pointcloud, PointCloud)
        assert len(pointcloud.points) == 1000
        assert len(pointcloud.intensities) == 1000
        assert len(pointcloud.timestamps) == 1000
        
        # 检查点的范围
        distances = np.linalg.norm(pointcloud.points, axis=1)
        assert np.all(distances <= 100.0)
    
    def test_pointcloud_dimensions(self, sensor_generator):
        """测试点云维度"""
        pointcloud = sensor_generator.generate_lidar_pointcloud(num_points=100)
        
        assert pointcloud.points.shape == (100, 3)
        assert pointcloud.intensities.shape == (100,)
        assert pointcloud.timestamps.shape == (100,)
    
    def test_pointcloud_noise(self, sensor_generator):
        """测试点云噪声"""
        # 无噪声
        pointcloud_no_noise = sensor_generator.generate_lidar_pointcloud(
            num_points=100,
            add_noise=False
        )
        
        # 有噪声
        pointcloud_with_noise = sensor_generator.generate_lidar_pointcloud(
            num_points=100,
            add_noise=True
        )
        
        # 两次生成的点应该不同
        assert not np.allclose(
            pointcloud_no_noise.points,
            pointcloud_with_noise.points
        )
    
    def test_pointcloud_timestamp(self, sensor_generator):
        """测试点云时间戳"""
        timestamp = 1234567890.0
        pointcloud = sensor_generator.generate_lidar_pointcloud(
            num_points=100,
            timestamp=timestamp
        )
        
        assert np.all(pointcloud.timestamps == timestamp)


@pytest.mark.unit
@pytest.mark.sensor
class TestCameraSensor:
    """摄像头传感器单元测试"""
    
    def test_image_generation(self, sensor_generator):
        """测试图像数据生成"""
        image = sensor_generator.generate_camera_image(
            image_size=(1920, 1080),
            num_objects=5
        )
        
        assert isinstance(image, ImageData)
        assert image.data.shape == (1080, 1920, 3)
        assert image.camera_id == "camera_front"
        assert image.timestamp > 0
    
    def test_image_intrinsics(self, sensor_generator):
        """测试相机内参"""
        image = sensor_generator.generate_camera_image()
        
        assert image.intrinsics.shape == (3, 3)
        assert image.intrinsics[0, 0] > 0  # fx
        assert image.intrinsics[1, 1] > 0  # fy
        assert image.intrinsics[0, 2] > 0  # cx
        assert image.intrinsics[1, 2] > 0  # cy
    
    def test_image_extrinsics(self, sensor_generator):
        """测试相机外参"""
        image = sensor_generator.generate_camera_image()
        
        assert image.extrinsics.shape == (4, 4)
        assert np.allclose(image.extrinsics, np.eye(4))
    
    def test_different_image_sizes(self, sensor_generator):
        """测试不同图像尺寸"""
        sizes = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        
        for width, height in sizes:
            image = sensor_generator.generate_camera_image(image_size=(width, height))
            assert image.data.shape == (height, width, 3)


@pytest.mark.unit
@pytest.mark.sensor
class TestRadarSensor:
    """毫米波雷达传感器单元测试"""
    
    def test_radar_data_generation(self, sensor_generator):
        """测试雷达数据生成"""
        radar_data = sensor_generator.generate_radar_data(
            num_targets=10,
            max_range=150.0
        )
        
        assert isinstance(radar_data, RadarData)
        assert len(radar_data.targets) == 10
        assert radar_data.radar_id == "radar_front"
        assert radar_data.timestamp > 0
    
    def test_radar_target_properties(self, sensor_generator):
        """测试雷达目标属性"""
        radar_data = sensor_generator.generate_radar_data(num_targets=5)
        
        for target in radar_data.targets:
            assert 'id' in target
            assert 'range' in target
            assert 'azimuth' in target
            assert 'velocity' in target
            assert 'rcs' in target
            assert 'snr' in target
            
            # 检查范围
            assert 0 <= target['range'] <= 150.0
            # 检查方位角
            assert -np.pi/3 <= target['azimuth'] <= np.pi/3
            # 检查速度
            assert -30 <= target['velocity'] <= 30
    
    def test_radar_range_filtering(self, sensor_generator):
        """测试雷达距离过滤"""
        radar_data = sensor_generator.generate_radar_data(
            num_targets=20,
            max_range=50.0
        )
        
        for target in radar_data.targets:
            assert target['range'] <= 50.0


@pytest.mark.unit
@pytest.mark.sensor
class TestIMUSensor:
    """IMU传感器单元测试"""
    
    def test_imu_data_generation(self, sensor_generator):
        """测试IMU数据生成"""
        imu_data_list = sensor_generator.generate_imu_data(
            duration=1.0,
            sample_rate=100.0
        )
        
        assert len(imu_data_list) == 100
        
        for imu_data in imu_data_list:
            assert isinstance(imu_data, IMUData)
            assert imu_data.acceleration.shape == (3,)
            assert imu_data.angular_velocity.shape == (3,)
            assert imu_data.orientation.shape == (4,)
    
    def test_imu_gravity(self, sensor_generator):
        """测试IMU重力加速度"""
        imu_data_list = sensor_generator.generate_imu_data(duration=0.1)
        
        for imu_data in imu_data_list:
            # z方向应该有重力加速度
            assert abs(imu_data.acceleration[2] - 9.81) < 1.0
    
    def test_imu_timestamps(self, sensor_generator):
        """测试IMU时间戳连续性"""
        imu_data_list = sensor_generator.generate_imu_data(
            duration=1.0,
            sample_rate=10.0
        )
        
        timestamps = [imu.timestamp for imu in imu_data_list]
        
        # 检查时间戳递增
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]
            assert abs(timestamps[i] - timestamps[i-1] - 0.1) < 0.001
    
    def test_imu_noise(self, sensor_generator):
        """测试IMU噪声"""
        # 无噪声
        imu_no_noise = sensor_generator.generate_imu_data(
            duration=0.1,
            add_noise=False
        )
        
        # 有噪声
        imu_with_noise = sensor_generator.generate_imu_data(
            duration=0.1,
            add_noise=True
        )
        
        # 检查噪声是否添加
        accel_no_noise = imu_no_noise[0].acceleration
        accel_with_noise = imu_with_noise[0].acceleration
        
        assert not np.allclose(accel_no_noise, accel_with_noise)


@pytest.mark.unit
@pytest.mark.sensor
class TestGPSSensor:
    """GPS传感器单元测试"""
    
    def test_gps_data_generation(self, sensor_generator):
        """测试GPS数据生成"""
        gps_data_list = sensor_generator.generate_gps_data(
            duration=1.0,
            sample_rate=10.0
        )
        
        assert len(gps_data_list) == 10
        
        for gps_data in gps_data_list:
            assert isinstance(gps_data, GPSData)
            assert -90 <= gps_data.latitude <= 90
            assert -180 <= gps_data.longitude <= 180
            assert gps_data.speed >= 0
            assert 0 <= gps_data.heading < 360
    
    def test_gps_position_change(self, sensor_generator):
        """测试GPS位置变化"""
        gps_data_list = sensor_generator.generate_gps_data(
            duration=5.0,
            sample_rate=10.0
        )
        
        # 检查位置是否变化
        start_pos = (gps_data_list[0].latitude, gps_data_list[0].longitude)
        end_pos = (gps_data_list[-1].latitude, gps_data_list[-1].longitude)
        
        assert start_pos != end_pos
    
    def test_gps_speed_consistency(self, sensor_generator):
        """测试GPS速度一致性"""
        gps_data_list = sensor_generator.generate_gps_data(duration=1.0)
        
        speeds = [gps.speed for gps in gps_data_list]
        
        # 速度应该在合理范围内
        assert all(0 <= s < 50 for s in speeds)
    
    def test_gps_start_position(self, sensor_generator):
        """测试GPS起始位置"""
        start_pos = (39.9042, 116.4074)  # 北京
        
        gps_data_list = sensor_generator.generate_gps_data(
            duration=0.1,
            start_pos=start_pos
        )
        
        # 检查起始位置
        assert abs(gps_data_list[0].latitude - start_pos[0]) < 0.0001
        assert abs(gps_data_list[0].longitude - start_pos[1]) < 0.0001


@pytest.mark.unit
@pytest.mark.sensor
class TestSensorSuite:
    """传感器套件单元测试"""
    
    def test_sensor_suite_generation(self, sensor_generator):
        """测试传感器套件生成"""
        sensor_suite = sensor_generator.generate_sensor_suite()
        
        assert SensorType.LIDAR in sensor_suite
        assert SensorType.CAMERA in sensor_suite
        assert SensorType.RADAR in sensor_suite
        assert SensorType.IMU in sensor_suite
        assert SensorType.GPS in sensor_suite
        
        assert isinstance(sensor_suite[SensorType.LIDAR], PointCloud)
        assert isinstance(sensor_suite[SensorType.CAMERA], ImageData)
        assert isinstance(sensor_suite[SensorType.RADAR], RadarData)
        assert isinstance(sensor_suite[SensorType.IMU], IMUData)
        assert isinstance(sensor_suite[SensorType.GPS], GPSData)
    
    def test_sensor_suite_timestamp_sync(self, sensor_generator):
        """测试传感器套件时间戳同步"""
        timestamp = 1234567890.0
        sensor_suite = sensor_generator.generate_sensor_suite(timestamp=timestamp)
        
        # 检查各传感器时间戳
        assert sensor_suite[SensorType.LIDAR].timestamps[0] == timestamp
        assert sensor_suite[SensorType.CAMERA].timestamp == timestamp
        assert sensor_suite[SensorType.RADAR].timestamp == timestamp
        assert sensor_suite[SensorType.IMU].timestamp == timestamp
        assert sensor_suite[SensorType.GPS].timestamp == timestamp


@pytest.mark.unit
@pytest.mark.sensor
class TestSensorDataValidation:
    """传感器数据验证单元测试"""
    
    def test_pointcloud_validation(self, sensor_generator):
        """测试点云数据验证"""
        pointcloud = sensor_generator.generate_lidar_pointcloud(num_points=100)
        
        # 检查NaN和Inf
        assert not np.any(np.isnan(pointcloud.points))
        assert not np.any(np.isinf(pointcloud.points))
        assert not np.any(np.isnan(pointcloud.intensities))
        assert not np.any(np.isinf(pointcloud.intensities))
    
    def test_image_data_validation(self, sensor_generator):
        """测试图像数据验证"""
        image = sensor_generator.generate_camera_image()
        
        # 检查像素值范围
        assert np.all(image.data >= 0)
        assert np.all(image.data <= 255)
        
        # 检查数据类型
        assert image.data.dtype == np.uint8
    
    def test_imu_data_validation(self, sensor_generator):
        """测试IMU数据验证"""
        imu_data = sensor_generator.generate_imu_data(duration=0.1)[0]
        
        # 检查NaN和Inf
        assert not np.any(np.isnan(imu_data.acceleration))
        assert not np.any(np.isinf(imu_data.acceleration))
        assert not np.any(np.isnan(imu_data.angular_velocity))
        assert not np.any(np.isinf(imu_data.angular_velocity))
    
    def test_gps_data_validation(self, sensor_generator):
        """测试GPS数据验证"""
        gps_data = sensor_generator.generate_gps_data(duration=0.1)[0]
        
        # 检查数值有效性
        assert not np.isnan(gps_data.latitude)
        assert not np.isnan(gps_data.longitude)
        assert not np.isnan(gps_data.altitude)
        assert not np.isnan(gps_data.speed)
        assert not np.isnan(gps_data.heading)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
