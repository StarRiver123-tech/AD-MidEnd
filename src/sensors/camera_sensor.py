"""
自动驾驶系统 - 摄像头传感器
实现摄像头数据采集
"""

import numpy as np
from typing import Optional
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .sensor_base import SensorBase
from ..common.data_types import ImageData, SensorConfig, Timestamp


class CameraSensor(SensorBase):
    """摄像头传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig, **kwargs):
        super().__init__(sensor_id, config, **kwargs)
        
        # 摄像头参数
        self._resolution = config.parameters.get('resolution', [1920, 1080])
        self._fps = config.parameters.get('fps', 30)
        self._acquisition_rate = self._fps
        
        # 内参
        self._intrinsics = config.intrinsics
        self._distortion = config.distortion
        
        # 视频捕获对象
        self._cap: Optional[any] = None
        
        # 仿真模式
        self._simulation_mode = False
        self._simulation_images = []
        self._simulation_index = 0
    
    def _initialize_hardware(self) -> bool:
        """初始化摄像头硬件"""
        if not CV2_AVAILABLE:
            self._logger.warning("OpenCV not available, using simulation mode")
            self._simulation_mode = True
            return True
        
        device_path = self._config.device_path
        
        try:
            self._cap = cv2.VideoCapture(device_path)
            
            if not self._cap.isOpened():
                self._logger.warning(f"Cannot open camera {device_path}, using simulation mode")
                self._simulation_mode = True
                return True
            
            # 设置分辨率
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            
            self._logger.info(f"Camera {self._sensor_id} initialized: "
                            f"{self._resolution[0]}x{self._resolution[1]}@{self._fps}fps")
            return True
            
        except Exception as e:
            self._logger.error(f"Camera initialization failed: {e}")
            self._simulation_mode = True
            return True
    
    def _stop_hardware(self) -> None:
        """停止摄像头"""
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def _acquire_data(self) -> Optional[ImageData]:
        """采集图像数据"""
        if self._simulation_mode:
            return self._acquire_simulation_data()
        
        if not self._cap:
            return None
        
        ret, frame = self._cap.read()
        
        if not ret:
            self._logger.warning("Failed to capture frame")
            return None
        
        # 创建ImageData
        image_data = ImageData(
            timestamp=Timestamp.now(),
            camera_id=self._sensor_id,
            image=frame,
            width=frame.shape[1],
            height=frame.shape[0],
            channels=frame.shape[2] if len(frame.shape) > 2 else 1,
            intrinsics=self._intrinsics if self._intrinsics is not None else np.eye(3),
            extrinsics=self._config.extrinsics,
            distortion=self._distortion if self._distortion is not None else np.zeros(5)
        )
        
        return image_data
    
    def _acquire_simulation_data(self) -> Optional[ImageData]:
        """采集仿真数据"""
        # 生成随机图像用于仿真
        if CV2_AVAILABLE:
            # 生成彩色噪声图像
            frame = np.random.randint(0, 255, 
                                     (self._resolution[1], self._resolution[0], 3), 
                                     dtype=np.uint8)
        else:
            # 使用numpy数组
            frame = np.random.rand(self._resolution[1], self._resolution[0], 3)
        
        image_data = ImageData(
            timestamp=Timestamp.now(),
            camera_id=self._sensor_id,
            image=frame,
            width=self._resolution[0],
            height=self._resolution[1],
            channels=3,
            intrinsics=self._intrinsics if self._intrinsics is not None else np.eye(3),
            extrinsics=self._config.extrinsics,
            distortion=self._distortion if self._distortion is not None else np.zeros(5)
        )
        
        return image_data
    
    def _get_topic(self) -> str:
        """获取发布主题"""
        camera_type = self._config.parameters.get('camera_type', 'unknown')
        if 'front' in camera_type:
            return "sensor/camera/front"
        elif 'rear' in camera_type:
            return "sensor/camera/rear"
        else:
            return f"sensor/camera/{camera_type}"
    
    def set_simulation_images(self, images: list) -> None:
        """设置仿真图像序列"""
        self._simulation_images = images
        self._simulation_index = 0
    
    def get_intrinsics(self) -> np.ndarray:
        """获取相机内参"""
        return self._intrinsics.copy() if self._intrinsics is not None else np.eye(3)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """去畸变图像"""
        if not CV2_AVAILABLE or self._intrinsics is None or self._distortion is None:
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self._intrinsics, self._distortion, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, self._intrinsics, self._distortion, None, new_camera_matrix
        )
        
        return undistorted
