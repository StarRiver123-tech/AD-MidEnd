"""
自动驾驶系统 - 发布者类
封装消息发布功能
"""

from typing import Any, Optional
from .message_bus import MessageBus, Message
from ..common.data_types import Timestamp
from ..logs.logger import Logger


class Publisher:
    """消息发布者"""
    
    def __init__(self, publisher_id: str, topic: str, 
                 message_bus: Optional[MessageBus] = None):
        """
        初始化发布者
        
        Args:
            publisher_id: 发布者唯一标识
            topic: 发布的主题
            message_bus: 消息总线实例（可选，默认使用全局实例）
        """
        self._publisher_id = publisher_id
        self._topic = topic
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger(f"Publisher-{publisher_id}")
        
        self._sequence_num = 0
        self._published_count = 0
    
    @property
    def publisher_id(self) -> str:
        return self._publisher_id
    
    @property
    def topic(self) -> str:
        return self._topic
    
    def publish(self, data: Any, priority: int = 0, sync: bool = False) -> bool:
        """
        发布消息
        
        Args:
            data: 消息数据
            priority: 消息优先级（数字越小优先级越高）
            sync: 是否同步发布
        
        Returns:
            是否发布成功
        """
        self._sequence_num += 1
        
        message = Message(
            topic=self._topic,
            data=data,
            timestamp=Timestamp.now(),
            sequence_num=self._sequence_num,
            publisher_id=self._publisher_id,
            priority=priority
        )
        
        if sync:
            success = self._message_bus.publish_sync(message)
        else:
            success = self._message_bus.publish(message)
        
        if success:
            self._published_count += 1
        
        return success
    
    def get_stats(self) -> dict:
        """获取发布统计"""
        return {
            'publisher_id': self._publisher_id,
            'topic': self._topic,
            'published_count': self._published_count,
            'sequence_num': self._sequence_num
        }
