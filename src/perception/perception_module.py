"""
自动驾驶系统 - 感知模块
整合车道线检测、障碍物检测和Occupancy网络
"""

from typing import Optional, Dict, Any
import threading
import time
import numpy as np

from .lane_detector import LaneDetector
from .obstacle_detector import ObstacleDetector
from .occupancy_network import OccupancyNetwork
from ..common.data_types import (
    PerceptionResult, ImageData, PointCloud, RadarData,
    CANData, Timestamp, Vector3D, Pose
)
from ..common.enums import ModuleState
from ..communication.message_bus import MessageBus, Message
from ..communication import Publisher, Subscriber
from ..config.config_manager import ConfigManager
from ..logs.logger import Logger


class PerceptionModule:
    """感知模块"""
    
    def __init__(self, message_bus: Optional[MessageBus] = None):
        """
        初始化感知模块
        
        Args:
            message_bus: 消息总线实例
        """
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger("PerceptionModule")
        
        # 状态
        self._state = ModuleState.UNINITIALIZED
        self._enabled = True
        
        # 子模块
        self._lane_detector: Optional[LaneDetector] = None
        self._obstacle_detector: Optional[ObstacleDetector] = None
        self._occupancy_network: Optional[OccupancyNetwork] = None
        
        # 发布者
        self._lane_publisher = Publisher("perception", "perception/lane", self._message_bus)
        self._obstacle_publisher = Publisher("perception", "perception/obstacle", self._message_bus)
        self._occupancy_publisher = Publisher("perception", "perception/occupancy", self._message_bus)
        self._fusion_publisher = Publisher("perception", "perception/fusion", self._message_bus)
        
        # 订阅者
        self._subscriber = Subscriber("perception", self._message_bus)
        
        # 输入数据缓存
        self._latest_camera_data: Optional[ImageData] = None
        self._latest_lidar_data: Optional[PointCloud] = None
        self._latest_radar_data: Optional[RadarData] = None
        self._latest_can_data: Optional[CANData] = None
        
        # 数据锁
        self._data_lock = threading.Lock()
        
        # 处理线程
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._processing_frequency = 10.0  # Hz
        
        # 配置
        self._config: Optional[Dict[str, Any]] = None
        
        # 统计
        self._processing_count = 0
        self._processing_time_ms = 0.0
    
    def initialize(self, config_manager: Optional[ConfigManager] = None) -> bool:
        """初始化感知模块"""
        if self._state != ModuleState.UNINITIALIZED:
            self._logger.warning(f"Cannot initialize in state {self._state.name}")
            return False
        
        self._state = ModuleState.INITIALIZING
        
        try:
            if config_manager is None:
                config_manager = ConfigManager()
            
            module_config = config_manager.get_module_config('perception')
            if module_config:
                self._enabled = module_config.enabled
                self._processing_frequency = module_config.execution_frequency
                self._config = module_config.algorithm_params
            
            # 初始化子模块
            self._lane_detector = LaneDetector(self._config.get('lane_detection', {}))
            self._obstacle_detector = ObstacleDetector(self._config.get('obstacle_detection', {}))
            self._occupancy_network = OccupancyNetwork(self._config.get('occupancy', {}))
            
            # 设置订阅
            self._setup_subscriptions()
            
            self._state = ModuleState.READY
            self._logger.info("PerceptionModule initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            self._state = ModuleState.ERROR
            return False
    
    def _setup_subscriptions(self) -> None:
        """设置数据订阅"""
        # 订阅摄像头数据
        self._subscriber.subscribe(
            topic="sensor/camera/front",
            callback=self._on_camera_data
        )
        
        # 订阅LiDAR数据
        self._subscriber.subscribe(
            topic="sensor/lidar",
            callback=self._on_lidar_data
        )
        
        # 订阅雷达数据
        self._subscriber.subscribe(
            topic="sensor/radar",
            callback=self._on_radar_data
        )
        
        # 订阅CAN数据
        self._subscriber.subscribe(
            topic="sensor/can/vehicle",
            callback=self._on_can_data
        )
    
    def _on_camera_data(self, message: Message) -> None:
        """处理摄像头数据"""
        with self._data_lock:
            self._latest_camera_data = message.data
    
    def _on_lidar_data(self, message: Message) -> None:
        """处理LiDAR数据"""
        with self._data_lock:
            self._latest_lidar_data = message.data
    
    def _on_radar_data(self, message: Message) -> None:
        """处理雷达数据"""
        with self._data_lock:
            self._latest_radar_data = message.data
    
    def _on_can_data(self, message: Message) -> None:
        """处理CAN数据"""
        with self._data_lock:
            self._latest_can_data = message.data
    
    def start(self) -> bool:
        """启动感知模块"""
        if self._state not in [ModuleState.READY, ModuleState.SHUTDOWN]:
            self._logger.warning(f"Cannot start in state {self._state.name}")
            return False
        
        if not self._enabled:
            self._logger.info("PerceptionModule is disabled")
            return False
        
        self._running = True
        self._state = ModuleState.RUNNING
        
        # 启动处理线程
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="PerceptionProcessing",
            daemon=True
        )
        self._processing_thread.start()
        
        self._logger.info("PerceptionModule started")
        return True
    
    def stop(self) -> None:
        """停止感知模块"""
        self._running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        
        self._subscriber.unsubscribe_all()
        self._state = ModuleState.SHUTDOWN
        
        self._logger.info("PerceptionModule stopped")
    
    def _processing_loop(self) -> None:
        """处理循环"""
        period = 1.0 / self._processing_frequency
        
        while self._running:
            start_time = time.time()
            
            try:
                # 执行感知处理
                self._process_frame()
                
            except Exception as e:
                self._logger.error(f"Processing error: {e}")
            
            # 控制处理频率
            elapsed = time.time() - start_time
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame(self) -> None:
        """处理一帧数据"""
        start_time = time.time()
        
        # 获取数据
        with self._data_lock:
            camera_data = self._latest_camera_data
            lidar_data = self._latest_lidar_data
            radar_data = self._latest_radar_data
            can_data = self._latest_can_data
        
        # 检查是否有足够的数据
        if camera_data is None and lidar_data is None:
            return
        
        # 获取自车状态
        ego_pose = Pose()
        ego_velocity = Vector3D()
        if can_data:
            ego_velocity.x = can_data.vehicle_speed
        
        # 执行车道线检测
        lane_result = None
        if camera_data and self._lane_detector:
            lane_result = self._lane_detector.detect(camera_data)
            if lane_result:
                self._lane_publisher.publish(lane_result)
        
        # 执行障碍物检测
        obstacle_result = None
        if self._obstacle_detector:
            obstacle_result = self._obstacle_detector.detect(
                camera_data=camera_data,
                lidar_data=lidar_data,
                radar_data=radar_data
            )
            if obstacle_result:
                self._obstacle_publisher.publish(obstacle_result)
        
        # 执行Occupancy检测
        occupancy_result = None
        if lidar_data and self._occupancy_network:
            occupancy_result = self._occupancy_network.predict(lidar_data)
            if occupancy_result:
                self._occupancy_publisher.publish(occupancy_result)
        
        # 创建综合感知结果
        processing_time = (time.time() - start_time) * 1000  # ms
        
        perception_result = PerceptionResult(
            timestamp=Timestamp.now(),
            lane_result=lane_result,
            obstacle_result=obstacle_result,
            occupancy_result=occupancy_result,
            ego_pose=ego_pose,
            ego_velocity=ego_velocity,
            processing_time_ms=processing_time
        )
        
        # 发布融合结果
        self._fusion_publisher.publish(perception_result)
        
        # 更新统计
        self._processing_count += 1
        self._processing_time_ms = processing_time
    
    def get_latest_result(self) -> Optional[PerceptionResult]:
        """获取最新的感知结果"""
        # 这里应该缓存最新的结果
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.name,
            'enabled': self._enabled,
            'processing_count': self._processing_count,
            'processing_time_ms': self._processing_time_ms,
            'frequency': self._processing_frequency
        }
