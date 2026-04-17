"""
自动驾驶系统 - 源代码包
=======================

包含所有自动驾驶系统模块的实现

模块:
    - config: 配置管理
    - communication: 消息总线和通信
    - common: 通用数据类型和工具
    - dataset: 数据集适配器
    - logs: 日志系统
    - perception: 感知模块
    - planning: 规划模块
    - sensors: 传感器模块
    - visualization: 可视化模块
"""

__version__ = "1.0.0"
__author__ = "Autonomous Driving Team"

# 导出主要类
from .config.config_manager import ConfigManager
from .communication.message_bus import MessageBus
from .logs.logger import Logger

__all__ = [
    'ConfigManager',
    'MessageBus',
    'Logger',
]
