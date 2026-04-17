"""
自动驾驶系统 - 配置管理模块
提供配置加载、解析和管理功能
"""

from .config_manager import ConfigManager, get_config_manager
from .default_config import get_default_config

__all__ = ['ConfigManager', 'get_config_manager', 'get_default_config']
