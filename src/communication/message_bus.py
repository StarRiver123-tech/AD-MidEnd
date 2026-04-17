"""
自动驾驶系统 - 消息总线
实现模块间通信的核心机制
"""

import threading
import queue
import time
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import copy

from ..common.data_types import Timestamp
from ..logs.logger import Logger


class Topic(Enum):
    """预定义的主题"""
    # 传感器数据
    CAMERA_FRONT = "sensor/camera/front"
    CAMERA_REAR = "sensor/camera/rear"
    CAMERA_SURROUND = "sensor/camera/surround"
    LIDAR = "sensor/lidar"
    RADAR = "sensor/radar"
    ULTRASONIC = "sensor/ultrasonic"
    CAN_VEHICLE = "sensor/can/vehicle"
    
    # 感知结果
    PERCEPTION_LANE = "perception/lane"
    PERCEPTION_OBSTACLE = "perception/obstacle"
    PERCEPTION_OCCUPANCY = "perception/occupancy"
    PERCEPTION_FUSION = "perception/fusion"
    PERCEPTION_ALL = "perception/all"
    
    # 规划结果
    PLANNING_TRAJECTORY = "planning/trajectory"
    PLANNING_BEHAVIOR = "planning/behavior"
    PLANNING_ALL = "planning/all"
    
    # 控制指令
    CONTROL_COMMAND = "control/command"
    
    # 系统状态
    SYSTEM_STATE = "system/state"
    SYSTEM_LOG = "system/log"


@dataclass
class Message:
    """消息数据结构"""
    topic: str
    data: Any
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    sequence_num: int = 0
    publisher_id: str = ""
    priority: int = 0  # 优先级，数字越小优先级越高
    
    def copy(self) -> 'Message':
        """创建消息的深拷贝"""
        return Message(
            topic=self.topic,
            data=copy.deepcopy(self.data),
            timestamp=self.timestamp,
            sequence_num=self.sequence_num,
            publisher_id=self.publisher_id,
            priority=self.priority
        )


class MessageBus:
    """
    消息总线 - 单例模式
    提供发布-订阅模式的模块间通信
    """
    _instance: Optional['MessageBus'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'MessageBus':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._logger = Logger("MessageBus")
        
        # 订阅者字典: topic -> {subscriber_id: callback}
        self._subscribers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        
        # 消息队列
        self._message_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # 历史消息缓存（用于延迟订阅）
        self._history: Dict[str, List[Message]] = defaultdict(list)
        self._max_history_size = 10
        
        # 运行状态
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._message_count = 0
        self._dropped_count = 0
        self._sequence_counter: Dict[str, int] = defaultdict(int)
        
        # 同步锁
        self._lock = threading.RLock()
    
    def start(self) -> bool:
        """启动消息总线"""
        with self._lock:
            if self._running:
                return True
            
            self._running = True
            self._dispatcher_thread = threading.Thread(
                target=self._dispatch_loop,
                name="MessageDispatcher",
                daemon=True
            )
            self._dispatcher_thread.start()
            self._logger.info("MessageBus started")
            return True
    
    def stop(self) -> None:
        """停止消息总线"""
        with self._lock:
            self._running = False
        
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=2.0)
        
        self._logger.info("MessageBus stopped")
    
    def subscribe(self, topic: str, callback: Callable[[Message], None], 
                  subscriber_id: str, receive_history: bool = False) -> bool:
        """
        订阅主题
        
        Args:
            topic: 主题名称
            callback: 消息回调函数
            subscriber_id: 订阅者唯一标识
            receive_history: 是否接收历史消息
        
        Returns:
            是否订阅成功
        """
        with self._lock:
            if subscriber_id in self._subscribers[topic]:
                self._logger.warning(
                    f"Subscriber {subscriber_id} already subscribed to {topic}"
                )
                return False
            
            self._subscribers[topic][subscriber_id] = callback
            self._logger.debug(f"Subscriber {subscriber_id} subscribed to {topic}")
            
            # 发送历史消息
            if receive_history and topic in self._history:
                for msg in self._history[topic]:
                    try:
                        callback(msg)
                    except Exception as e:
                        self._logger.error(f"Error sending history: {e}")
            
            return True
    
    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """取消订阅"""
        with self._lock:
            if topic in self._subscribers and subscriber_id in self._subscribers[topic]:
                del self._subscribers[topic][subscriber_id]
                self._logger.debug(f"Subscriber {subscriber_id} unsubscribed from {topic}")
                return True
            return False
    
    def unsubscribe_all(self, subscriber_id: str) -> None:
        """取消所有订阅"""
        with self._lock:
            for topic in list(self._subscribers.keys()):
                if subscriber_id in self._subscribers[topic]:
                    del self._subscribers[topic][subscriber_id]
    
    def publish(self, message: Message) -> bool:
        """
        发布消息
        
        Args:
            message: 要发布的消息
        
        Returns:
            是否发布成功
        """
        if not self._running:
            self._logger.warning("MessageBus not running, cannot publish")
            return False
        
        # 设置序列号
        with self._lock:
            self._sequence_counter[message.topic] += 1
            message.sequence_num = self._sequence_counter[message.topic]
            self._message_count += 1
        
        # 添加到消息队列（优先级队列）
        try:
            # 使用优先级作为队列键值的一部分
            priority_key = (message.priority, message.timestamp.to_seconds())
            self._message_queue.put((priority_key, message), block=False)
            
            # 保存到历史
            self._add_to_history(message)
            
            return True
        except queue.Full:
            self._dropped_count += 1
            self._logger.warning(f"Message queue full, dropped message to {message.topic}")
            return False
    
    def publish_sync(self, message: Message) -> bool:
        """
        同步发布消息（直接调用回调，不经过队列）
        
        Args:
            message: 要发布的消息
        
        Returns:
            是否发布成功
        """
        with self._lock:
            self._sequence_counter[message.topic] += 1
            message.sequence_num = self._sequence_counter[message.topic]
            
            subscribers = dict(self._subscribers.get(message.topic, {}))
        
        # 直接调用回调
        success = True
        for subscriber_id, callback in subscribers.items():
            try:
                callback(message)
            except Exception as e:
                self._logger.error(f"Error in subscriber {subscriber_id}: {e}")
                success = False
        
        return success
    
    def _dispatch_loop(self) -> None:
        """消息分发循环"""
        while self._running:
            try:
                # 从队列获取消息（带超时）
                _, message = self._message_queue.get(timeout=0.1)
                self._dispatch_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"Error in dispatch loop: {e}")
    
    def _dispatch_message(self, message: Message) -> None:
        """分发消息到所有订阅者"""
        with self._lock:
            subscribers = dict(self._subscribers.get(message.topic, {}))
        
        for subscriber_id, callback in subscribers.items():
            try:
                callback(message)
            except Exception as e:
                self._logger.error(
                    f"Error dispatching to {subscriber_id} on {message.topic}: {e}"
                )
    
    def _add_to_history(self, message: Message) -> None:
        """添加消息到历史缓存"""
        with self._lock:
            self._history[message.topic].append(message.copy())
            # 限制历史大小
            if len(self._history[message.topic]) > self._max_history_size:
                self._history[message.topic].pop(0)
    
    def get_subscriber_count(self, topic: str) -> int:
        """获取主题的订阅者数量"""
        with self._lock:
            return len(self._subscribers.get(topic, {}))
    
    def get_topics(self) -> List[str]:
        """获取所有有订阅者的主题"""
        with self._lock:
            return list(self._subscribers.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'message_count': self._message_count,
                'dropped_count': self._dropped_count,
                'queue_size': self._message_queue.qsize(),
                'topics': self.get_topics(),
                'subscriber_counts': {
                    topic: len(subs) 
                    for topic, subs in self._subscribers.items()
                }
            }
    
    def clear_history(self, topic: Optional[str] = None) -> None:
        """清除历史消息"""
        with self._lock:
            if topic:
                self._history[topic].clear()
            else:
                self._history.clear()


# 便捷函数
def get_message_bus() -> MessageBus:
    """获取消息总线实例"""
    return MessageBus()
