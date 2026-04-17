"""
自动驾驶系统 - 订阅者类
封装消息订阅功能
"""

from typing import Callable, Optional, List, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque
import threading

from .message_bus import MessageBus, Message
from ..logs.logger import Logger


class CallbackType(Enum):
    """回调类型"""
    SYNC = "sync"  # 同步回调
    ASYNC = "async"  # 异步回调


@dataclass
class ReceivedMessage:
    """接收到的消息包装"""
    message: Message
    receive_time: float


class Subscriber:
    """消息订阅者"""
    
    def __init__(self, subscriber_id: str, 
                 message_bus: Optional[MessageBus] = None,
                 callback_type: CallbackType = CallbackType.ASYNC,
                 max_queue_size: int = 100):
        """
        初始化订阅者
        
        Args:
            subscriber_id: 订阅者唯一标识
            message_bus: 消息总线实例（可选）
            callback_type: 回调类型
            max_queue_size: 消息队列最大大小
        """
        self._subscriber_id = subscriber_id
        self._message_bus = message_bus or MessageBus()
        self._callback_type = callback_type
        self._logger = Logger(f"Subscriber-{subscriber_id}")
        
        # 订阅的主题
        self._subscribed_topics: List[str] = []
        
        # 消息队列（用于异步处理）
        self._message_queue: deque = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()
        
        # 自定义回调
        self._custom_callbacks: dict[str, Callable[[Message], None]] = {}
        
        # 统计
        self._received_count = 0
        self._dropped_count = 0
    
    @property
    def subscriber_id(self) -> str:
        return self._subscriber_id
    
    def subscribe(self, topic: str, 
                  callback: Optional[Callable[[Message], None]] = None,
                  receive_history: bool = False) -> bool:
        """
        订阅主题
        
        Args:
            topic: 主题名称
            callback: 自定义回调函数（可选）
            receive_history: 是否接收历史消息
        
        Returns:
            是否订阅成功
        """
        if callback:
            self._custom_callbacks[topic] = callback
        
        # 使用内部回调包装
        success = self._message_bus.subscribe(
            topic=topic,
            callback=self._on_message,
            subscriber_id=self._subscriber_id,
            receive_history=receive_history
        )
        
        if success:
            self._subscribed_topics.append(topic)
            self._logger.debug(f"Subscribed to {topic}")
        
        return success
    
    def unsubscribe(self, topic: str) -> bool:
        """取消订阅"""
        success = self._message_bus.unsubscribe(topic, self._subscriber_id)
        if success and topic in self._subscribed_topics:
            self._subscribed_topics.remove(topic)
            if topic in self._custom_callbacks:
                del self._custom_callbacks[topic]
        return success
    
    def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        self._message_bus.unsubscribe_all(self._subscriber_id)
        self._subscribed_topics.clear()
        self._custom_callbacks.clear()
    
    def _on_message(self, message: Message) -> None:
        """内部消息回调"""
        import time
        
        self._received_count += 1
        
        if self._callback_type == CallbackType.SYNC:
            # 同步处理
            self._process_message(message)
        else:
            # 异步处理：加入队列
            with self._queue_lock:
                if len(self._message_queue) >= self._message_queue.maxlen:
                    self._dropped_count += 1
                else:
                    self._message_queue.append(
                        ReceivedMessage(message, time.time())
                    )
    
    def _process_message(self, message: Message) -> None:
        """处理消息"""
        # 调用自定义回调
        if message.topic in self._custom_callbacks:
            try:
                self._custom_callbacks[message.topic](message)
            except Exception as e:
                self._logger.error(f"Error in callback for {message.topic}: {e}")
    
    def process_queue(self, max_messages: int = -1) -> int:
        """
        处理消息队列中的消息
        
        Args:
            max_messages: 最大处理消息数，-1表示处理所有
        
        Returns:
            处理的消息数
        """
        processed = 0
        
        with self._queue_lock:
            queue_size = len(self._message_queue)
            if max_messages < 0:
                max_messages = queue_size
            
            for _ in range(min(max_messages, queue_size)):
                received = self._message_queue.popleft()
                self._process_message(received.message)
                processed += 1
        
        return processed
    
    def has_messages(self) -> bool:
        """检查是否有待处理的消息"""
        with self._queue_lock:
            return len(self._message_queue) > 0
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        with self._queue_lock:
            return len(self._message_queue)
    
    def get_subscribed_topics(self) -> List[str]:
        """获取已订阅的主题"""
        return self._subscribed_topics.copy()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'subscriber_id': self._subscriber_id,
            'subscribed_topics': self._subscribed_topics,
            'received_count': self._received_count,
            'dropped_count': self._dropped_count,
            'queue_size': self.get_queue_size()
        }


class FilteredSubscriber(Subscriber):
    """带过滤功能的订阅者"""
    
    def __init__(self, subscriber_id: str, 
                 message_bus: Optional[MessageBus] = None,
                 callback_type: CallbackType = CallbackType.ASYNC,
                 max_queue_size: int = 100):
        super().__init__(subscriber_id, message_bus, callback_type, max_queue_size)
        
        # 过滤器
        self._filters: dict[str, Callable[[Message], bool]] = {}
    
    def subscribe_with_filter(self, topic: str, 
                              callback: Callable[[Message], None],
                              filter_fn: Callable[[Message], bool],
                              receive_history: bool = False) -> bool:
        """
        带过滤器的订阅
        
        Args:
            topic: 主题
            callback: 回调函数
            filter_fn: 过滤函数，返回True表示接受消息
            receive_history: 是否接收历史
        """
        self._filters[topic] = filter_fn
        return self.subscribe(topic, callback, receive_history)
    
    def _on_message(self, message: Message) -> None:
        """带过滤的消息处理"""
        # 应用过滤器
        if message.topic in self._filters:
            if not self._filters[message.topic](message):
                return  # 消息被过滤
        
        super()._on_message(message)
