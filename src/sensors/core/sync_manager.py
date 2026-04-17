"""
数据同步管理模块
实现多传感器数据的时间戳对齐和同步
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from ..drivers.base_sensor import SensorData, SensorType


@dataclass
class SyncConfig:
    """同步配置"""
    mode: str = "software"  # "software" | "hardware" | "hybrid"
    master_clock: str = "lidar"  # 主时钟源
    time_tolerance_ms: float = 10.0  # 时间容差(ms)
    sync_period_ms: float = 100.0  # 同步周期(ms)
    max_wait_time_ms: float = 50.0  # 最大等待时间(ms)


@dataclass
class SyncedFrame:
    """同步帧数据"""
    timestamp: float
    frame_id: int
    data: Dict[str, SensorData] = field(default_factory=dict)
    missing_sensors: List[str] = field(default_factory=list)
    is_complete: bool = False


class SyncManager:
    """
    数据同步管理器
    实现多传感器数据的时间戳对齐
    """
    
    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        
        # 传感器数据缓冲区
        self._data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 同步组配置
        self._sync_groups: Dict[str, List[str]] = {}
        
        # 同步回调函数
        self._sync_callbacks: List[Callable[[SyncedFrame], None]] = []
        
        # 同步线程
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._frame_count = 0
        self._sync_success_count = 0
        self._sync_fail_count = 0
        
    def register_sensor(self, sensor_name: str, group: str = "default") -> None:
        """
        注册传感器到同步管理器
        Args:
            sensor_name: 传感器名称
            group: 同步组名称
        """
        with self._lock:
            if group not in self._sync_groups:
                self._sync_groups[group] = []
            
            if sensor_name not in self._sync_groups[group]:
                self._sync_groups[group].append(sensor_name)
                print(f"Registered sensor '{sensor_name}' to sync group '{group}'")
    
    def unregister_sensor(self, sensor_name: str, group: str = None) -> None:
        """
        注销传感器
        Args:
            sensor_name: 传感器名称
            group: 同步组名称，None表示所有组
        """
        with self._lock:
            if group is None:
                for g in self._sync_groups:
                    if sensor_name in self._sync_groups[g]:
                        self._sync_groups[g].remove(sensor_name)
            else:
                if group in self._sync_groups and sensor_name in self._sync_groups[group]:
                    self._sync_groups[group].remove(sensor_name)
    
    def add_data(self, sensor_name: str, data: SensorData) -> None:
        """
        添加传感器数据到缓冲区
        Args:
            sensor_name: 传感器名称
            data: 传感器数据
        """
        with self._lock:
            self._data_buffers[sensor_name].append(data)
    
    def register_sync_callback(self, callback: Callable[[SyncedFrame], None]) -> None:
        """
        注册同步回调函数
        Args:
            callback: 回调函数，接收SyncedFrame参数
        """
        with self._lock:
            if callback not in self._sync_callbacks:
                self._sync_callbacks.append(callback)
    
    def start(self) -> None:
        """启动同步管理器"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
            print("SyncManager started")
    
    def stop(self) -> None:
        """停止同步管理器"""
        with self._lock:
            self._running = False
        
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        
        print("SyncManager stopped")
    
    def _sync_loop(self) -> None:
        """同步循环线程"""
        sync_period = self.config.sync_period_ms / 1000.0
        
        while self._running:
            try:
                # 对每个同步组执行同步
                for group_name, sensor_list in self._sync_groups.items():
                    synced_frame = self._sync_group(group_name, sensor_list)
                    
                    if synced_frame is not None:
                        # 通知回调
                        for callback in self._sync_callbacks:
                            try:
                                callback(synced_frame)
                            except Exception as e:
                                print(f"Sync callback error: {e}")
                
                time.sleep(sync_period)
                
            except Exception as e:
                print(f"Sync loop error: {e}")
                time.sleep(0.01)
    
    def _sync_group(self, group_name: str, sensor_list: List[str]) -> Optional[SyncedFrame]:
        """
        同步一个传感器组
        Args:
            group_name: 组名称
            sensor_list: 传感器列表
        Returns:
            SyncedFrame: 同步帧
        """
        with self._lock:
            # 获取参考时间戳（主时钟）
            reference_time = self._get_reference_timestamp(sensor_list)
            
            if reference_time is None:
                return None
            
            # 收集同步数据
            synced_data = {}
            missing_sensors = []
            
            for sensor_name in sensor_list:
                data = self._find_closest_data(sensor_name, reference_time)
                
                if data is not None:
                    synced_data[sensor_name] = data
                else:
                    missing_sensors.append(sensor_name)
            
            # 判断是否同步成功
            is_complete = len(missing_sensors) == 0
            
            if is_complete:
                self._sync_success_count += 1
            else:
                self._sync_fail_count += 1
            
            self._frame_count += 1
            
            return SyncedFrame(
                timestamp=reference_time,
                frame_id=self._frame_count,
                data=synced_data,
                missing_sensors=missing_sensors,
                is_complete=is_complete
            )
    
    def _get_reference_timestamp(self, sensor_list: List[str]) -> Optional[float]:
        """
        获取参考时间戳
        Args:
            sensor_list: 传感器列表
        Returns:
            float: 参考时间戳
        """
        # 优先使用主时钟源
        if self.config.master_clock in sensor_list:
            buffer = self._data_buffers.get(self.config.master_clock)
            if buffer and len(buffer) > 0:
                return buffer[-1].timestamp
        
        # 使用最新的时间戳
        latest_time = None
        
        for sensor_name in sensor_list:
            buffer = self._data_buffers.get(sensor_name)
            if buffer and len(buffer) > 0:
                timestamp = buffer[-1].timestamp
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
        
        return latest_time
    
    def _find_closest_data(self, sensor_name: str, target_time: float) -> Optional[SensorData]:
        """
        找到最接近目标时间的数据
        Args:
            sensor_name: 传感器名称
            target_time: 目标时间戳
        Returns:
            SensorData: 最接近的数据
        """
        buffer = self._data_buffers.get(sensor_name)
        
        if not buffer or len(buffer) == 0:
            return None
        
        tolerance = self.config.time_tolerance_ms / 1000.0
        
        # 找到最接近的数据
        closest_data = None
        min_diff = float('inf')
        
        for data in buffer:
            diff = abs(data.timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_data = data
        
        # 检查是否在容差范围内
        if min_diff <= tolerance:
            return closest_data
        
        return None
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """
        获取同步统计信息
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            total = self._sync_success_count + self._sync_fail_count
            success_rate = self._sync_success_count / total if total > 0 else 0.0
            
            return {
                'frame_count': self._frame_count,
                'sync_success': self._sync_success_count,
                'sync_fail': self._sync_fail_count,
                'success_rate': success_rate,
                'buffer_sizes': {name: len(buffer) for name, buffer in self._data_buffers.items()}
            }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._frame_count = 0
            self._sync_success_count = 0
            self._sync_fail_count = 0


class TimeSynchronizer:
    """
    时间同步器
    实现传感器时间戳的统一和校准
    """
    
    def __init__(self):
        self._time_offsets: Dict[str, float] = {}
        self._reference_time: Optional[float] = None
        self._lock = threading.RLock()
    
    def set_reference_time(self, timestamp: float) -> None:
        """
        设置参考时间
        Args:
            timestamp: 参考时间戳
        """
        with self._lock:
            self._reference_time = timestamp
    
    def calibrate_sensor(self, sensor_name: str, sensor_time: float, reference_time: float) -> None:
        """
        校准传感器时间偏移
        Args:
            sensor_name: 传感器名称
            sensor_time: 传感器时间戳
            reference_time: 参考时间戳
        """
        with self._lock:
            offset = reference_time - sensor_time
            self._time_offsets[sensor_name] = offset
    
    def synchronize_timestamp(self, sensor_name: str, timestamp: float) -> float:
        """
        同步时间戳
        Args:
            sensor_name: 传感器名称
            timestamp: 原始时间戳
        Returns:
            float: 同步后的时间戳
        """
        with self._lock:
            offset = self._time_offsets.get(sensor_name, 0.0)
            return timestamp + offset
    
    def get_time_offset(self, sensor_name: str) -> float:
        """
        获取时间偏移
        Args:
            sensor_name: 传感器名称
        Returns:
            float: 时间偏移
        """
        with self._lock:
            return self._time_offsets.get(sensor_name, 0.0)


class DataAligner:
    """
    数据对齐器
    实现不同频率传感器数据的对齐
    """
    
    def __init__(self):
        self._sensor_frequencies: Dict[str, float] = {}
        self._interpolation_methods: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def set_sensor_frequency(self, sensor_name: str, frequency: float) -> None:
        """
        设置传感器频率
        Args:
            sensor_name: 传感器名称
            frequency: 频率(Hz)
        """
        with self._lock:
            self._sensor_frequencies[sensor_name] = frequency
    
    def interpolate_data(self, 
                        data_list: List[SensorData], 
                        target_time: float,
                        method: str = "linear") -> Optional[SensorData]:
        """
        插值数据到目标时间
        Args:
            data_list: 数据列表
            target_time: 目标时间戳
            method: 插值方法
        Returns:
            SensorData: 插值后的数据
        """
        if len(data_list) < 2:
            return data_list[0] if data_list else None
        
        # 找到目标时间两侧的数据
        before = None
        after = None
        
        for data in data_list:
            if data.timestamp <= target_time:
                before = data
            if data.timestamp >= target_time and after is None:
                after = data
                break
        
        if before is None:
            return after
        if after is None:
            return before
        if before == after:
            return before
        
        # 线性插值
        if method == "linear":
            t1 = before.timestamp
            t2 = after.timestamp
            
            if t2 == t1:
                return before
            
            ratio = (target_time - t1) / (t2 - t1)
            
            # 创建插值数据
            # 注意：这里需要根据具体数据类型实现插值
            return before  # 简化处理
        
        return before
    
    def align_to_frequency(self, 
                          data_dict: Dict[str, List[SensorData]], 
                          target_frequency: float) -> List[Dict[str, SensorData]]:
        """
        对齐数据到目标频率
        Args:
            data_dict: 传感器数据字典
            target_frequency: 目标频率
        Returns:
            List[Dict]: 对齐后的数据序列
        """
        period = 1.0 / target_frequency
        
        # 找到时间范围
        min_time = float('inf')
        max_time = 0.0
        
        for data_list in data_dict.values():
            for data in data_list:
                min_time = min(min_time, data.timestamp)
                max_time = max(max_time, data.timestamp)
        
        # 生成对齐的数据帧
        aligned_frames = []
        current_time = min_time
        
        while current_time <= max_time:
            frame = {}
            
            for sensor_name, data_list in data_dict.items():
                interpolated = self.interpolate_data(data_list, current_time)
                if interpolated is not None:
                    frame[sensor_name] = interpolated
            
            if frame:
                aligned_frames.append(frame)
            
            current_time += period
        
        return aligned_frames


def create_default_sync_groups() -> Dict[str, List[str]]:
    """
    创建默认同步组
    Returns:
        Dict: 同步组配置
    """
    return {
        "perception_group": [
            "front_long",
            "front_wide", 
            "main_lidar",
            "front_radar"
        ],
        "surround_group": [
            "left_front",
            "right_front",
            "left_rear",
            "right_rear",
            "left_front_radar",
            "right_front_radar",
            "left_rear_radar",
            "right_rear_radar"
        ],
        "parking_group": [
            "front_fisheye",
            "rear_fisheye",
            "left_fisheye",
            "right_fisheye",
            "ultrasonics"
        ],
        "vehicle_state": [
            "can0"
        ]
    }
