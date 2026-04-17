"""
摄像头驱动模块
支持真实摄像头硬件和仿真模式
"""

import time
import threading
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available, camera simulation mode only")

from .base_sensor import BaseSensor, SensorConfig, SensorType, SensorData, ImageData, SensorState


class CameraConfig(SensorConfig):
    """摄像头配置类"""
    
    def __init__(self, 
                 name: str,
                 resolution: Tuple[int, int] = (1920, 1080),
                 fps: int = 30,
                 fov: float = 90.0,
                 channel: int = 0,
                 interface: str = "gmsl",
                 lens_type: str = "normal",
                 **kwargs):
        super().__init__(
            name=name,
            sensor_type=SensorType.CAMERA,
            interface=interface,
            **kwargs
        )
        self.resolution = resolution
        self.fps = fps
        self.fov = fov
        self.channel = channel
        self.lens_type = lens_type
        self.frame_interval = 1.0 / fps


class CameraDriver(BaseSensor):
    """
    摄像头驱动类
    支持GMSL/USB摄像头和仿真模式
    """
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.camera_config = config
        self._cap: Optional[Any] = None
        self._last_capture_time = 0.0
        self._simulation_mode = False
        self._simulation_image: Optional[np.ndarray] = None
        
    def initialize(self) -> bool:
        """
        初始化摄像头
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            self.state = SensorState.INITIALIZING
            
        try:
            if not CV2_AVAILABLE:
                # 无cv2时进入仿真模式
                self._simulation_mode = True
                self._init_simulation()
                self.state = SensorState.READY
                return True
            
            # 尝试打开真实摄像头
            if self.camera_config.interface == "usb":
                self._cap = cv2.VideoCapture(self.camera_config.channel)
            elif self.camera_config.interface == "gmsl":
                # GMSL摄像头通常通过V4L2接口访问
                self._cap = cv2.VideoCapture(self.camera_config.channel)
            else:
                # 其他接口使用仿真模式
                self._simulation_mode = True
                self._init_simulation()
                self.state = SensorState.READY
                return True
            
            if not self._cap.isOpened():
                print(f"Failed to open camera {self.name}, switching to simulation mode")
                self._simulation_mode = True
                self._init_simulation()
            else:
                # 设置摄像头参数
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
                self._cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
                
            with self._lock:
                self.state = SensorState.READY
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self._simulation_mode = True
            self._init_simulation()
            with self._lock:
                self.state = SensorState.READY
            return True
    
    def _init_simulation(self) -> None:
        """初始化仿真模式"""
        width, height = self.camera_config.resolution
        # 创建彩色测试图像
        self._simulation_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        print(f"Camera {self.name} initialized in simulation mode")
    
    def capture(self) -> Optional[ImageData]:
        """
        采集一帧图像
        Returns:
            ImageData: 图像数据
        """
        # 帧率控制
        current_time = time.time()
        elapsed = current_time - self._last_capture_time
        if elapsed < self.camera_config.frame_interval:
            time.sleep(self.camera_config.frame_interval - elapsed)
        
        try:
            if self._simulation_mode:
                image = self._generate_simulation_image()
            else:
                ret, image = self._cap.read()
                if not ret or image is None:
                    # 读取失败，使用仿真图像
                    image = self._generate_simulation_image()
            
            self._last_capture_time = time.time()
            
            return ImageData(
                timestamp=time.time(),
                sensor_name=self.name,
                sensor_type=SensorType.CAMERA,
                data=image,
                encoding="rgb8",
                metadata={
                    "fov": self.camera_config.fov,
                    "lens_type": self.camera_config.lens_type,
                    "resolution": self.camera_config.resolution,
                    "fps": self.camera_config.fps
                }
            )
            
        except Exception as e:
            print(f"Camera capture error: {e}")
            return None
    
    def _generate_simulation_image(self) -> np.ndarray:
        """生成仿真图像"""
        width, height = self.camera_config.resolution
        
        # 创建带有时间戳和相机名称的测试图像
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加渐变背景
        for i in range(height):
            image[i, :] = [int(255 * i / height), 100, int(255 * (1 - i / height))]
        
        # 添加网格线
        grid_size = 100
        for i in range(0, width, grid_size):
            image[:, i:i+2] = [255, 255, 255]
        for i in range(0, height, grid_size):
            image[i:i+2, :] = [255, 255, 255]
        
        if CV2_AVAILABLE:
            # 添加文字信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{self.name} | Frame: {self._frame_id}"
            cv2.putText(image, text, (50, 50), font, 1, (255, 255, 255), 2)
            
            timestamp_text = f"Time: {time.time():.3f}"
            cv2.putText(image, timestamp_text, (50, 100), font, 1, (255, 255, 255), 2)
        
        return image
    
    def release(self) -> None:
        """释放摄像头资源"""
        self.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self.state = SensorState.STOPPED
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        获取相机内参矩阵
        Returns:
            np.ndarray: 3x3相机内参矩阵
        """
        width, height = self.camera_config.resolution
        fov_rad = np.radians(self.camera_config.fov)
        
        # 计算焦距
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # 假设像素是正方形
        
        cx = width / 2
        cy = height / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """
        获取畸变系数
        Returns:
            np.ndarray: 畸变系数 [k1, k2, p1, p2, k3]
        """
        if self.camera_config.lens_type == "fisheye":
            # 鱼眼镜头畸变系数
            return np.array([0.1, 0.01, 0.0, 0.0, 0.0])
        else:
            # 普通镜头畸变系数
            return np.array([0.01, 0.001, 0.0, 0.0, 0.0])


class CameraArray:
    """
    摄像头阵列管理类
    管理多个摄像头驱动
    """
    
    def __init__(self):
        self.cameras: Dict[str, CameraDriver] = {}
        self._lock = threading.RLock()
        
    def add_camera(self, config: CameraConfig) -> bool:
        """
        添加摄像头
        Args:
            config: 摄像头配置
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            if config.name in self.cameras:
                print(f"Camera {config.name} already exists")
                return False
            
            camera = CameraDriver(config)
            if camera.initialize():
                self.cameras[config.name] = camera
                return True
            return False
    
    def remove_camera(self, name: str) -> bool:
        """
        移除摄像头
        Args:
            name: 摄像头名称
        Returns:
            bool: 移除是否成功
        """
        with self._lock:
            if name not in self.cameras:
                return False
            
            self.cameras[name].release()
            del self.cameras[name]
            return True
    
    def start_all(self) -> None:
        """启动所有摄像头"""
        with self._lock:
            for camera in self.cameras.values():
                camera.start()
    
    def stop_all(self) -> None:
        """停止所有摄像头"""
        with self._lock:
            for camera in self.cameras.values():
                camera.stop()
    
    def get_camera(self, name: str) -> Optional[CameraDriver]:
        """
        获取指定摄像头
        Args:
            name: 摄像头名称
        Returns:
            CameraDriver: 摄像头驱动实例
        """
        with self._lock:
            return self.cameras.get(name)
    
    def get_all_cameras(self) -> Dict[str, CameraDriver]:
        """
        获取所有摄像头
        Returns:
            Dict[str, CameraDriver]: 摄像头字典
        """
        with self._lock:
            return dict(self.cameras)
    
    def capture_all(self) -> Dict[str, ImageData]:
        """
        采集所有摄像头图像
        Returns:
            Dict[str, ImageData]: 图像数据字典
        """
        results = {}
        with self._lock:
            for name, camera in self.cameras.items():
                data = camera.capture()
                if data is not None:
                    results[name] = data
        return results
    
    def get_camera_states(self) -> Dict[str, str]:
        """
        获取所有摄像头状态
        Returns:
            Dict[str, str]: 状态字典
        """
        with self._lock:
            return {name: cam.get_state().value for name, cam in self.cameras.items()}
    
    def release_all(self) -> None:
        """释放所有摄像头资源"""
        with self._lock:
            for camera in self.cameras.values():
                camera.release()
            self.cameras.clear()


# 预定义的摄像头配置
CAMERA_PRESETS = {
    "front_long": CameraConfig(
        name="front_long",
        resolution=(3840, 2160),
        fps=30,
        fov=30,
        channel=0,
        interface="gmsl",
        position=[2.0, 0.0, 1.2],
        orientation=[0.0, 0.0, 0.0]
    ),
    "front_wide": CameraConfig(
        name="front_wide",
        resolution=(3840, 2160),
        fps=30,
        fov=120,
        channel=1,
        interface="gmsl",
        position=[2.0, 0.0, 1.1],
        orientation=[0.0, 0.0, 0.0]
    ),
    "rear": CameraConfig(
        name="rear",
        resolution=(1920, 1536),
        fps=30,
        fov=100,
        channel=2,
        interface="gmsl",
        position=[-2.0, 0.0, 1.2],
        orientation=[0.0, 0.0, 180.0]
    ),
    "left_front": CameraConfig(
        name="left_front",
        resolution=(1920, 1536),
        fps=30,
        fov=100,
        channel=3,
        interface="gmsl",
        position=[1.5, 1.0, 1.0],
        orientation=[0.0, 0.0, 45.0]
    ),
    "right_front": CameraConfig(
        name="right_front",
        resolution=(1920, 1536),
        fps=30,
        fov=100,
        channel=4,
        interface="gmsl",
        position=[1.5, -1.0, 1.0],
        orientation=[0.0, 0.0, -45.0]
    ),
    "left_rear": CameraConfig(
        name="left_rear",
        resolution=(1920, 1536),
        fps=30,
        fov=100,
        channel=5,
        interface="gmsl",
        position=[-1.5, 1.0, 1.0],
        orientation=[0.0, 0.0, 135.0]
    ),
    "right_rear": CameraConfig(
        name="right_rear",
        resolution=(1920, 1536),
        fps=30,
        fov=100,
        channel=6,
        interface="gmsl",
        position=[-1.5, -1.0, 1.0],
        orientation=[0.0, 0.0, -135.0]
    ),
    "front_fisheye": CameraConfig(
        name="front_fisheye",
        resolution=(1920, 1536),
        fps=30,
        fov=190,
        channel=7,
        interface="gmsl",
        lens_type="fisheye",
        position=[2.0, 0.0, 0.8],
        orientation=[0.0, -30.0, 0.0]
    ),
    "rear_fisheye": CameraConfig(
        name="rear_fisheye",
        resolution=(1920, 1536),
        fps=30,
        fov=190,
        channel=8,
        interface="gmsl",
        lens_type="fisheye",
        position=[-2.0, 0.0, 0.8],
        orientation=[0.0, -30.0, 180.0]
    ),
    "left_fisheye": CameraConfig(
        name="left_fisheye",
        resolution=(1920, 1536),
        fps=30,
        fov=190,
        channel=9,
        interface="gmsl",
        lens_type="fisheye",
        position=[0.0, 1.0, 0.8],
        orientation=[0.0, -30.0, 90.0]
    ),
    "right_fisheye": CameraConfig(
        name="right_fisheye",
        resolution=(1920, 1536),
        fps=30,
        fov=190,
        channel=10,
        interface="gmsl",
        lens_type="fisheye",
        position=[0.0, -1.0, 0.8],
        orientation=[0.0, -30.0, -90.0]
    ),
}
