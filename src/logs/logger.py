"""
自动驾驶系统 - 日志系统
提供统一的日志记录和管理功能
"""

import logging
import logging.handlers
import os
import sys
import threading
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path


class LogLevel(Enum):
    """日志级别"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # 保存原始级别名称
        original_levelname = record.levelname
        
        # 添加颜色
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        
        # 恢复原始级别名称
        record.levelname = original_levelname
        
        return result


class LoggerManager:
    """日志管理器 - 单例模式"""
    
    _instance: Optional['LoggerManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'LoggerManager':
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
        
        # 配置
        self._log_level = LogLevel.INFO
        self._log_path = "./logs"
        self._log_to_console = True
        self._log_to_file = True
        self._max_file_size = 10 * 1024 * 1024  # 10MB
        self._backup_count = 5
        
        # 日志器缓存
        self._loggers: Dict[str, logging.Logger] = {}
        
        # 初始化根日志器
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """设置根日志器"""
        self._root_logger = logging.getLogger("AutonomousDriving")
        self._root_logger.setLevel(self._log_level.value)
        self._root_logger.handlers = []  # 清除现有处理器
        
        # 控制台处理器
        if self._log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._log_level.value)
            console_formatter = ColoredFormatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self._root_logger.addHandler(console_handler)
        
        # 文件处理器
        if self._log_to_file:
            self._setup_file_handler()
    
    def _setup_file_handler(self) -> None:
        """设置文件处理器"""
        # 创建日志目录
        log_dir = Path(self._log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"autonomous_driving_{timestamp}.log"
        
        # 创建轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._max_file_size,
            backupCount=self._backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self._log_level.value)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self._root_logger.addHandler(file_handler)
    
    def configure(self, 
                  log_level: Optional[LogLevel] = None,
                  log_path: Optional[str] = None,
                  log_to_console: Optional[bool] = None,
                  log_to_file: Optional[bool] = None) -> None:
        """配置日志系统"""
        if log_level is not None:
            self._log_level = log_level
        if log_path is not None:
            self._log_path = log_path
        if log_to_console is not None:
            self._log_to_console = log_to_console
        if log_to_file is not None:
            self._log_to_file = log_to_file
        
        # 重新设置
        self._setup_root_logger()
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取命名日志器"""
        if name not in self._loggers:
            logger = logging.getLogger(f"AutonomousDriving.{name}")
            self._loggers[name] = logger
        return self._loggers[name]


class Logger:
    """日志器包装类"""
    
    def __init__(self, name: str):
        """
        初始化日志器
        
        Args:
            name: 日志器名称（通常使用模块或类名）
        """
        self._name = name
        self._logger = LoggerManager().get_logger(name)
    
    @classmethod
    def set_global_level(cls, level: LogLevel) -> None:
        """设置全局日志级别"""
        LoggerManager().configure(log_level=level)
    
    @property
    def name(self) -> str:
        return self._name
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误日志"""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误日志"""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """记录异常日志（包含堆栈跟踪）"""
        self._logger.exception(message, **kwargs)
    
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """记录指定级别的日志"""
        self._logger.log(level.value, message, **kwargs)


# 便捷函数
def get_logger(name: str) -> Logger:
    """获取日志器实例"""
    return Logger(name)


def configure_logging(config: Dict[str, Any]) -> None:
    """使用配置字典配置日志系统"""
    manager = LoggerManager()
    
    log_level = config.get('log_level', 'INFO')
    if isinstance(log_level, str):
        log_level = LogLevel[log_level.upper()]
    
    manager.configure(
        log_level=log_level,
        log_path=config.get('log_path', './logs'),
        log_to_console=config.get('log_to_console', True),
        log_to_file=config.get('log_to_file', True)
    )
