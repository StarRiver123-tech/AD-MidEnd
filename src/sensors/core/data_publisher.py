"""
数据发布/订阅模块
实现传感器数据的发布和订阅机制
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..drivers.base_sensor import SensorData, SensorType


class DataPriority(Enum):
    """数据优先级"""
    CRITICAL = 0    # 紧急数据
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级


@dataclass
class DataMessage:
    """数据消息"""
    topic: str
    data: SensorData
    priority: DataPriority = DataPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSubscriber:
    """
    数据订阅者
    """
    
    def __init__(self, 
                 topic: str,
                 callback: Callable[[SensorData], None],
                 priority: DataPriority = DataPriority.NORMAL):
        self.topic = topic
        self.callback = callback
        self.priority = priority
        self._message_queue: queue.Queue = queue.Queue(maxsize=100)
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        
    def start(self) -> None:
        """启动订阅者处理线程"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processing_thread = threading.Thread(
                target=self._process_loop, 
                daemon=True
            )
            self._processing_thread.start()
    
    def stop(self) -> None:
        """停止订阅者处理线程"""
        with self._lock:
            self._running = False
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
    
    def enqueue(self, message: DataMessage) -> bool:
        """
        将消息加入队列
        Args:
            message: 数据消息
        Returns:
            bool: 是否成功加入
        """
        try:
            self._message_queue.put_nowait(message)
            return True
        except queue.Full:
            return False
    
    def _process_loop(self) -> None:
        """消息处理循环"""
        while self._running:
            try:
                message = self._message_queue.get(timeout=0.1)
                
                # 调用回调函数
                try:
                    self.callback(message.data)
                except Exception as e:
                    print(f"Subscriber callback error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Process loop error: {e}")


class DataPublisher:
    """
    数据发布器
    实现传感器数据的发布和订阅管理
    """
    
    def __init__(self):
        # 订阅者字典: topic -> list of subscribers
        self._subscribers: Dict[str, List[DataSubscriber]] = {}
        
        # 主题统计
        self._topic_stats: Dict[str, Dict] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # 发布统计
        self._publish_count = 0
        
    def subscribe(self, 
                  topic: str, 
                  callback: Callable[[SensorData], None],
                  priority: DataPriority = DataPriority.NORMAL) -> DataSubscriber:
        """
        订阅主题
        Args:
            topic: 主题名称
            callback: 回调函数
            priority: 优先级
        Returns:
            DataSubscriber: 订阅者对象
        """
        with self._lock:
            subscriber = DataSubscriber(topic, callback, priority)
            
            if topic not in self._subscribers:
                self._subscribers[topic] = []
                self._topic_stats[topic] = {
                    'publish_count': 0,
                    'subscriber_count': 0
                }
            
            self._subscribers[topic].append(subscriber)
            self._topic_stats[topic]['subscriber_count'] += 1
            
            # 启动订阅者
            subscriber.start()
            
            print(f"Subscribed to topic '{topic}'")
            return subscriber
    
    def unsubscribe(self, subscriber: DataSubscriber) -> bool:
        """
        取消订阅
        Args:
            subscriber: 订阅者对象
        Returns:
            bool: 是否成功取消
        """
        with self._lock:
            topic = subscriber.topic
            
            if topic not in self._subscribers:
                return False
            
            if subscriber in self._subscribers[topic]:
                subscriber.stop()
                self._subscribers[topic].remove(subscriber)
                self._topic_stats[topic]['subscriber_count'] -= 1
                
                print(f"Unsubscribed from topic '{topic}'")
                return True
            
            return False
    
    def publish(self, 
                topic: str, 
                data: SensorData,
                priority: DataPriority = DataPriority.NORMAL,
                metadata: Dict[str, Any] = None) -> int:
        """
        发布数据
        Args:
            topic: 主题名称
            data: 传感器数据
            priority: 优先级
            metadata: 元数据
        Returns:
            int: 成功发送的订阅者数量
        """
        with self._lock:
            if topic not in self._subscribers:
                return 0
            
            message = DataMessage(
                topic=topic,
                data=data,
                priority=priority,
                metadata=metadata or {}
            )
            
            # 更新统计
            self._publish_count += 1
            self._topic_stats[topic]['publish_count'] += 1
            
            # 发送给所有订阅者
            success_count = 0
            for subscriber in self._subscribers[topic]:
                if subscriber.enqueue(message):
                    success_count += 1
            
            return success_count
    
    def publish_sensor_data(self, data: SensorData, 
                           priority: DataPriority = DataPriority.NORMAL) -> int:
        """
        发布传感器数据（自动根据类型选择主题）
        Args:
            data: 传感器数据
            priority: 优先级
        Returns:
            int: 成功发送的订阅者数量
        """
        # 根据传感器类型选择主题
        topic_map = {
            SensorType.CAMERA: f"camera/{data.sensor_name}",
            SensorType.LIDAR: f"lidar/{data.sensor_name}",
            SensorType.RADAR: f"radar/{data.sensor_name}",
            SensorType.ULTRASONIC: f"ultrasonic/{data.sensor_name}",
            SensorType.IMU: f"imu/{data.sensor_name}",
            SensorType.GNSS: f"gnss/{data.sensor_name}",
            SensorType.CAN: f"vehicle/{data.sensor_name}",
        }
        
        topic = topic_map.get(data.sensor_type, f"sensor/{data.sensor_name}")
        
        return self.publish(topic, data, priority)
    
    def get_topics(self) -> List[str]:
        """
        获取所有主题
        Returns:
            List[str]: 主题列表
        """
        with self._lock:
            return list(self._subscribers.keys())
    
    def get_topic_stats(self, topic: str = None) -> Dict:
        """
        获取主题统计信息
        Args:
            topic: 主题名称，None表示所有主题
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            if topic is not None:
                return self._topic_stats.get(topic, {}).copy()
            
            return {
                'total_publish': self._publish_count,
                'topics': self._topic_stats.copy()
            }
    
    def has_subscribers(self, topic: str) -> bool:
        """
        检查主题是否有订阅者
        Args:
            topic: 主题名称
        Returns:
            bool: 是否有订阅者
        """
        with self._lock:
            return topic in self._subscribers and len(self._subscribers[topic]) > 0


class DataRouter:
    """
    数据路由器
    实现数据的智能路由和分发
    """
    
    def __init__(self, publisher: DataPublisher):
        self._publisher = publisher
        self._routing_rules: Dict[str, List[str]] = {}
        self._filters: Dict[str, Callable[[SensorData], bool]] = {}
        self._lock = threading.RLock()
    
    def add_routing_rule(self, source_topic: str, 
                        target_topics: List[str]) -> None:
        """
        添加路由规则
        Args:
            source_topic: 源主题
            target_topics: 目标主题列表
        """
        with self._lock:
            self._routing_rules[source_topic] = target_topics
    
    def add_filter(self, topic: str, 
                   filter_func: Callable[[SensorData], bool]) -> None:
        """
        添加数据过滤器
        Args:
            topic: 主题名称
            filter_func: 过滤函数
        """
        with self._lock:
            self._filters[topic] = filter_func
    
    def route(self, source_topic: str, data: SensorData) -> int:
        """
        路由数据
        Args:
            source_topic: 源主题
            data: 传感器数据
        Returns:
            int: 成功路由的数量
        """
        with self._lock:
            if source_topic not in self._routing_rules:
                return 0
            
            target_topics = self._routing_rules[source_topic]
            success_count = 0
            
            for target_topic in target_topics:
                # 应用过滤器
                if target_topic in self._filters:
                    if not self._filters[target_topic](data):
                        continue
                
                # 发布到目标主题
                if self._publisher.publish(target_topic, data) > 0:
                    success_count += 1
            
            return success_count


class DataBuffer:
    """
    数据缓冲区
    实现传感器数据的缓冲管理
    """
    
    def __init__(self, max_size: int = 100):
        self._buffer: List[SensorData] = []
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def push(self, data: SensorData) -> None:
        """
        添加数据到缓冲区
        Args:
            data: 传感器数据
        """
        with self._lock:
            self._buffer.append(data)
            
            # 保持缓冲区大小
            if len(self._buffer) > self._max_size:
                self._buffer.pop(0)
    
    def get_latest(self, n: int = 1) -> List[SensorData]:
        """
        获取最新的n条数据
        Args:
            n: 数量
        Returns:
            List[SensorData]: 数据列表
        """
        with self._lock:
            return self._buffer[-n:] if n <= len(self._buffer) else self._buffer.copy()
    
    def get_by_time_range(self, 
                         start_time: float, 
                         end_time: float) -> List[SensorData]:
        """
        获取时间范围内的数据
        Args:
            start_time: 开始时间
            end_time: 结束时间
        Returns:
            List[SensorData]: 数据列表
        """
        with self._lock:
            return [
                data for data in self._buffer
                if start_time <= data.timestamp <= end_time
            ]
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
    
    def size(self) -> int:
        """
        获取缓冲区大小
        Returns:
            int: 大小
        """
        with self._lock:
            return len(self._buffer)


class MultiTopicSubscriber:
    """
    多主题订阅者
    支持同时订阅多个主题
    """
    
    def __init__(self, publisher: DataPublisher):
        self._publisher = publisher
        self._subscribers: Dict[str, DataSubscriber] = {}
        self._data_buffer: Dict[str, DataBuffer] = {}
        self._lock = threading.RLock()
    
    def subscribe_topics(self, 
                        topics: List[str],
                        callback: Callable[[str, SensorData], None],
                        buffer_size: int = 10) -> None:
        """
        订阅多个主题
        Args:
            topics: 主题列表
            callback: 回调函数，接收(topic, data)参数
            buffer_size: 缓冲区大小
        """
        with self._lock:
            for topic in topics:
                # 创建缓冲区
                self._data_buffer[topic] = DataBuffer(max_size=buffer_size)
                
                # 创建包装回调
                def make_callback(t):
                    def wrapper(data):
                        self._data_buffer[t].push(data)
                        callback(t, data)
                    return wrapper
                
                # 订阅
                subscriber = self._publisher.subscribe(topic, make_callback(topic))
                self._subscribers[topic] = subscriber
    
    def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        with self._lock:
            for subscriber in self._subscribers.values():
                self._publisher.unsubscribe(subscriber)
            
            self._subscribers.clear()
            self._data_buffer.clear()
    
    def get_buffered_data(self, topic: str) -> List[SensorData]:
        """
        获取缓冲区的数据
        Args:
            topic: 主题名称
        Returns:
            List[SensorData]: 数据列表
        """
        with self._lock:
            buffer = self._data_buffer.get(topic)
            if buffer:
                return buffer.get_latest(buffer.size())
            return []


# 全局发布器实例
_global_publisher: Optional[DataPublisher] = None
_global_lock = threading.Lock()


def get_global_publisher() -> DataPublisher:
    """
    获取全局发布器实例
    Returns:
        DataPublisher: 全局发布器
    """
    global _global_publisher
    
    with _global_lock:
        if _global_publisher is None:
            _global_publisher = DataPublisher()
        
        return _global_publisher
