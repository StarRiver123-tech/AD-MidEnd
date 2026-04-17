"""
自动驾驶系统 - 通信模块
提供模块间通信的消息总线和发布订阅机制
"""

from .message_bus import MessageBus, Message, Topic
from .subscriber import Subscriber, CallbackType

try:
    from .publisher import Publisher
except ImportError:
    Publisher = None

__all__ = [
    'MessageBus',
    'Message',
    'Topic',
    'Subscriber',
    'CallbackType',
    'Publisher'
]
