"""
自动驾驶系统 - 配置管理器
提供配置的加载、解析和管理功能
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

from ..common.data_types import SystemConfig, ModuleConfig, SensorConfig
from ..logs.logger import Logger


class NumpyEncoder(json.JSONEncoder):
    """处理numpy数组的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class ConfigManager:
    """配置管理器 - 单例模式"""
    
    _instance: Optional['ConfigManager'] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._logger = Logger("ConfigManager")
        
        # 配置存储
        self._config: Dict[str, Any] = {}
        self._system_config: Optional[SystemConfig] = None
        
        # 配置文件路径
        self._config_file: Optional[str] = None
    
    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载配置
        
        Args:
            file_path: 配置文件路径（支持.json和.yaml格式）
        
        Returns:
            是否加载成功
        """
        path = Path(file_path)
        
        if not path.exists():
            self._logger.error(f"Config file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.json']:
                    self._config = json.load(f)
                elif path.suffix in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                else:
                    self._logger.error(f"Unsupported config format: {path.suffix}")
                    return False
            
            self._config_file = file_path
            self._parse_system_config()
            self._logger.info(f"Config loaded from {file_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load config: {e}")
            return False
    
    def load_from_dict(self, config: Dict[str, Any]) -> bool:
        """从字典加载配置"""
        self._config = config.copy()
        self._parse_system_config()
        self._logger.info("Config loaded from dictionary")
        return True
    
    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            file_path: 目标文件路径，默认使用加载时的路径
        """
        save_path = file_path or self._config_file
        
        if not save_path:
            self._logger.error("No file path specified for saving config")
            return False
        
        path = Path(save_path)
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if path.suffix in ['.json']:
                    json.dump(self._config, f, indent=2, cls=NumpyEncoder)
                elif path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False)
            
            self._logger.info(f"Config saved to {save_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save config: {e}")
            return False
    
    def _parse_system_config(self) -> None:
        """解析系统配置"""
        if not self._config:
            return
        
        # 创建SystemConfig对象
        self._system_config = SystemConfig(
            system_name=self._config.get('system_name', 'AutonomousDrivingSystem'),
            version=self._config.get('version', '1.0.0'),
            log_level=self._config.get('log_level', 'INFO'),
            log_path=self._config.get('log_path', './logs'),
            log_to_console=self._config.get('log_to_console', True),
            log_to_file=self._config.get('log_to_file', True),
            simulation_mode=self._config.get('simulation_mode', False),
            simulation_dataset=self._config.get('simulation_dataset', ''),
            simulation_speed=self._config.get('simulation_speed', 1.0),
            sensors=self._parse_sensor_configs(),
            modules=self._parse_module_configs(),
            parameters=self._config.get('parameters', {})
        )
    
    def _parse_sensor_configs(self) -> Dict[str, SensorConfig]:
        """解析传感器配置"""
        sensors = {}
        sensor_configs = self._config.get('sensors', {})
        
        for sensor_id, config in sensor_configs.items():
            # 解析外参
            extrinsics = np.array(config.get('extrinsics', np.eye(4).tolist()))
            
            # 解析内参（相机）
            intrinsics = None
            if 'intrinsics' in config:
                intrinsics = np.array(config['intrinsics'])
            
            distortion = None
            if 'distortion' in config:
                distortion = np.array(config['distortion'])
            
            sensors[sensor_id] = SensorConfig(
                sensor_id=sensor_id,
                sensor_type=config.get('sensor_type', ''),
                enabled=config.get('enabled', True),
                interface=config.get('interface', ''),
                device_path=config.get('device_path', ''),
                ip_address=config.get('ip_address', ''),
                port=config.get('port', 0),
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                distortion=distortion,
                parameters=config.get('parameters', {})
            )
        
        return sensors
    
    def _parse_module_configs(self) -> Dict[str, ModuleConfig]:
        """解析模块配置"""
        modules = {}
        module_configs = self._config.get('modules', {})
        
        for module_name, config in module_configs.items():
            modules[module_name] = ModuleConfig(
                module_name=module_name,
                enabled=config.get('enabled', True),
                execution_frequency=config.get('execution_frequency', 10.0),
                timeout_ms=config.get('timeout_ms', 100.0),
                algorithm_params=config.get('algorithm_params', {}),
                input_topics=config.get('input_topics', []),
                output_topics=config.get('output_topics', [])
            )
        
        return modules
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # 更新SystemConfig
        self._parse_system_config()
    
    def get_system_config(self) -> Optional[SystemConfig]:
        """获取系统配置对象"""
        return self._system_config
    
    def get_sensor_config(self, sensor_id: str) -> Optional[SensorConfig]:
        """获取传感器配置"""
        if self._system_config:
            return self._system_config.sensors.get(sensor_id)
        return None
    
    def get_module_config(self, module_name: str) -> Optional[ModuleConfig]:
        """获取模块配置"""
        if self._system_config:
            return self._system_config.modules.get(module_name)
        return None
    
    def get_all_sensor_configs(self) -> Dict[str, SensorConfig]:
        """获取所有传感器配置"""
        if self._system_config:
            return self._system_config.sensors
        return {}
    
    def get_all_module_configs(self) -> Dict[str, ModuleConfig]:
        """获取所有模块配置"""
        if self._system_config:
            return self._system_config.modules
        return {}
    
    def update_sensor_config(self, sensor_id: str, config: SensorConfig) -> None:
        """更新传感器配置"""
        if 'sensors' not in self._config:
            self._config['sensors'] = {}
        
        self._config['sensors'][sensor_id] = {
            'sensor_type': config.sensor_type,
            'enabled': config.enabled,
            'interface': config.interface,
            'device_path': config.device_path,
            'ip_address': config.ip_address,
            'port': config.port,
            'extrinsics': config.extrinsics.tolist(),
            'intrinsics': config.intrinsics.tolist() if config.intrinsics is not None else None,
            'distortion': config.distortion.tolist() if config.distortion is not None else None,
            'parameters': config.parameters
        }
        
        self._parse_system_config()
    
    def update_module_config(self, module_name: str, config: ModuleConfig) -> None:
        """更新模块配置"""
        if 'modules' not in self._config:
            self._config['modules'] = {}
        
        self._config['modules'][module_name] = {
            'enabled': config.enabled,
            'execution_frequency': config.execution_frequency,
            'timeout_ms': config.timeout_ms,
            'algorithm_params': config.algorithm_params,
            'input_topics': config.input_topics,
            'output_topics': config.output_topics
        }
        
        self._parse_system_config()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self._config.copy()
    
    def validate(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        
        # 验证传感器配置
        sensors = self.get_all_sensor_configs()
        if not sensors:
            errors.append("No sensors configured")
        
        for sensor_id, config in sensors.items():
            if not config.sensor_type:
                errors.append(f"Sensor {sensor_id}: missing sensor_type")
        
        # 验证模块配置
        modules = self.get_all_module_configs()
        if not modules:
            errors.append("No modules configured")
        
        return errors


# 便捷函数
def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager()
