"""
CAN总线驱动模块
支持整车CAN数据接入
"""

import time
import struct
import threading
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .base_sensor import BaseSensor, SensorConfig, SensorType, SensorData, VehicleStateData, SensorState


@dataclass
class CANSignal:
    """CAN信号定义"""
    name: str
    can_id: int
    start_bit: int
    length: int
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    signed: bool = False


@dataclass
class CANConfig(SensorConfig):
    """CAN总线配置类"""
    
    def __init__(self,
                 name: str,
                 channel: str = "can0",
                 interface: str = "socketcan",
                 bitrate: int = 500000,
                 signals: Dict[str, CANSignal] = None,
                 **kwargs):
        super().__init__(
            name=name,
            sensor_type=SensorType.CAN,
            interface=interface,
            **kwargs
        )
        self.channel = channel
        self.bitrate = bitrate
        self.signals = signals or {}


class CANDriver(BaseSensor):
    """
    CAN总线驱动类
    支持整车CAN数据接入
    """
    
    def __init__(self, config: CANConfig):
        super().__init__(config)
        self.can_config = config
        self._can_bus: Optional[Any] = None
        self._signal_values: Dict[str, float] = {}
        self._signal_callbacks: Dict[str, List[Callable[[float], None]]] = {}
        self._message_buffer: deque = deque(maxlen=1000)
        self._receive_thread: Optional[threading.Thread] = None
        self._running_receive = False
        self._simulation_mode = False
        
    def initialize(self) -> bool:
        """
        初始化CAN总线
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            self.state = SensorState.INITIALIZING
        
        try:
            # 尝试初始化CAN接口
            try:
                import can
                self._can_bus = can.interface.Bus(
                    channel=self.can_config.channel,
                    bustype=self.can_config.interface,
                    bitrate=self.can_config.bitrate
                )
                print(f"CAN bus {self.name} initialized on channel {self.can_config.channel}")
                
                # 启动接收线程
                self._running_receive = True
                self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self._receive_thread.start()
                
            except Exception as e:
                print(f"Cannot initialize CAN bus: {e}")
                print(f"CAN {self.name} switching to simulation mode")
                self._simulation_mode = True
            
            with self._lock:
                self.state = SensorState.READY
            return True
            
        except Exception as e:
            print(f"CAN initialization error: {e}")
            self._simulation_mode = True
            with self._lock:
                self.state = SensorState.READY
            return True
    
    def _receive_loop(self) -> None:
        """CAN接收循环线程"""
        while self._running_receive:
            try:
                if self._can_bus is not None:
                    msg = self._can_bus.recv(timeout=0.01)
                    if msg is not None:
                        self._process_message(msg)
            except Exception as e:
                if not self._simulation_mode:
                    print(f"CAN receive error: {e}")
    
    def _process_message(self, msg) -> None:
        """
        处理CAN消息
        Args:
            msg: CAN消息
        """
        # 存储消息
        self._message_buffer.append({
            'timestamp': time.time(),
            'arbitration_id': msg.arbitration_id,
            'data': msg.data,
            'dlc': msg.dlc
        })
        
        # 解析信号
        for signal_name, signal in self.can_config.signals.items():
            if signal.can_id == msg.arbitration_id:
                value = self._parse_signal(msg.data, signal)
                self._signal_values[signal_name] = value
                
                # 调用信号回调
                if signal_name in self._signal_callbacks:
                    for callback in self._signal_callbacks[signal_name]:
                        try:
                            callback(value)
                        except Exception as e:
                            print(f"Signal callback error: {e}")
    
    def _parse_signal(self, data: bytes, signal: CANSignal) -> float:
        """
        解析CAN信号
        Args:
            data: CAN消息数据
            signal: 信号定义
        Returns:
            float: 信号值
        """
        # 计算字节位置和位偏移
        byte_pos = signal.start_bit // 8
        bit_offset = signal.start_bit % 8
        
        # 读取原始值
        raw_value = 0
        bits_read = 0
        
        while bits_read < signal.length:
            if byte_pos >= len(data):
                break
            
            # 计算当前字节可读取的位数
            bits_in_byte = min(8 - bit_offset, signal.length - bits_read)
            
            # 读取位
            mask = ((1 << bits_in_byte) - 1) << bit_offset
            byte_value = (data[byte_pos] & mask) >> bit_offset
            
            # 添加到结果
            raw_value |= byte_value << bits_read
            
            bits_read += bits_in_byte
            byte_pos += 1
            bit_offset = 0
        
        # 处理有符号数
        if signal.signed:
            if raw_value & (1 << (signal.length - 1)):
                raw_value -= (1 << signal.length)
        
        # 应用比例和偏移
        value = raw_value * signal.scale + signal.offset
        
        return value
    
    def capture(self) -> Optional[VehicleStateData]:
        """
        采集车辆状态数据
        Returns:
            VehicleStateData: 车辆状态数据
        """
        try:
            if self._simulation_mode:
                self._generate_simulation_signals()
            
            return VehicleStateData(
                timestamp=time.time(),
                sensor_name=self.name,
                sensor_type=SensorType.CAN,
                speed=self._signal_values.get('vehicle_speed', 0.0),
                steering_angle=self._signal_values.get('steering_angle', 0.0),
                yaw_rate=self._signal_values.get('yaw_rate', 0.0),
                longitudinal_accel=self._signal_values.get('longitudinal_accel', 0.0),
                lateral_accel=self._signal_values.get('lateral_accel', 0.0),
                gear_position=int(self._signal_values.get('gear_position', 0)),
                metadata={
                    "channel": self.can_config.channel,
                    "bitrate": self.can_config.bitrate,
                    "all_signals": self._signal_values.copy()
                }
            )
            
        except Exception as e:
            print(f"CAN capture error: {e}")
            return None
    
    def _generate_simulation_signals(self) -> None:
        """生成仿真信号"""
        # 生成随时间变化的车辆状态
        t = time.time()
        
        self._signal_values['vehicle_speed'] = 10.0 + 5.0 * np.sin(t * 0.5)
        self._signal_values['steering_angle'] = 5.0 * np.sin(t * 0.3)
        self._signal_values['yaw_rate'] = 2.0 * np.sin(t * 0.4)
        self._signal_values['longitudinal_accel'] = 0.5 * np.sin(t * 0.6)
        self._signal_values['lateral_accel'] = 0.3 * np.sin(t * 0.5)
        self._signal_values['gear_position'] = 3  # D档
    
    def get_signal_value(self, signal_name: str) -> Optional[float]:
        """
        获取信号值
        Args:
            signal_name: 信号名称
        Returns:
            float: 信号值
        """
        return self._signal_values.get(signal_name)
    
    def register_signal_callback(self, signal_name: str, callback: Callable[[float], None]) -> None:
        """
        注册信号回调
        Args:
            signal_name: 信号名称
            callback: 回调函数
        """
        with self._lock:
            if signal_name not in self._signal_callbacks:
                self._signal_callbacks[signal_name] = []
            self._signal_callbacks[signal_name].append(callback)
    
    def send_message(self, can_id: int, data: bytes) -> bool:
        """
        发送CAN消息
        Args:
            can_id: CAN ID
            data: 消息数据
        Returns:
            bool: 发送是否成功
        """
        if self._can_bus is None or self._simulation_mode:
            return False
        
        try:
            import can
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            )
            self._can_bus.send(msg)
            return True
        except Exception as e:
            print(f"CAN send error: {e}")
            return False
    
    def release(self) -> None:
        """释放CAN总线资源"""
        self.stop()
        self._running_receive = False
        
        if self._receive_thread is not None:
            self._receive_thread.join(timeout=1.0)
        
        if self._can_bus is not None:
            self._can_bus.shutdown()
            self._can_bus = None
        
        with self._lock:
            self.state = SensorState.STOPPED


class VehicleStateManager:
    """
    车辆状态管理器
    管理整车CAN数据
    """
    
    def __init__(self):
        self.can_drivers: Dict[str, CANDriver] = {}
        self._vehicle_state: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._state_callbacks: List[Callable[[Dict[str, float]], None]] = []
        
    def add_can_channel(self, config: CANConfig) -> bool:
        """
        添加CAN通道
        Args:
            config: CAN配置
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            if config.name in self.can_drivers:
                print(f"CAN channel {config.name} already exists")
                return False
            
            driver = CANDriver(config)
            if driver.initialize():
                self.can_drivers[config.name] = driver
                # 注册状态更新回调
                driver.register_callback(self._on_data_received)
                return True
            return False
    
    def _on_data_received(self, data: SensorData) -> None:
        """数据接收回调"""
        if isinstance(data, VehicleStateData):
            with self._lock:
                self._vehicle_state.update({
                    'speed': data.speed,
                    'steering_angle': data.steering_angle,
                    'yaw_rate': data.yaw_rate,
                    'longitudinal_accel': data.longitudinal_accel,
                    'lateral_accel': data.lateral_accel,
                    'gear_position': data.gear_position,
                    'timestamp': data.timestamp
                })
            
            # 通知状态更新
            for callback in self._state_callbacks:
                try:
                    callback(self._vehicle_state)
                except Exception as e:
                    print(f"State callback error: {e}")
    
    def register_state_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        """
        注册状态更新回调
        Args:
            callback: 回调函数
        """
        with self._lock:
            self._state_callbacks.append(callback)
    
    def get_vehicle_state(self) -> Dict[str, float]:
        """
        获取车辆状态
        Returns:
            Dict[str, float]: 车辆状态字典
        """
        with self._lock:
            return self._vehicle_state.copy()
    
    def get_speed(self) -> float:
        """获取车速"""
        with self._lock:
            return self._vehicle_state.get('speed', 0.0)
    
    def get_steering_angle(self) -> float:
        """获取转向角"""
        with self._lock:
            return self._vehicle_state.get('steering_angle', 0.0)
    
    def get_yaw_rate(self) -> float:
        """获取横摆角速度"""
        with self._lock:
            return self._vehicle_state.get('yaw_rate', 0.0)
    
    def start_all(self) -> None:
        """启动所有CAN通道"""
        with self._lock:
            for driver in self.can_drivers.values():
                driver.start()
    
    def stop_all(self) -> None:
        """停止所有CAN通道"""
        with self._lock:
            for driver in self.can_drivers.values():
                driver.stop()
    
    def release_all(self) -> None:
        """释放所有CAN资源"""
        with self._lock:
            for driver in self.can_drivers.values():
                driver.release()
            self.can_drivers.clear()


# 预定义的CAN信号配置
VEHICLE_SIGNALS = {
    'vehicle_speed': CANSignal(
        name='vehicle_speed',
        can_id=0x130,
        start_bit=0,
        length=16,
        scale=0.01,
        offset=0,
        unit='m/s'
    ),
    'steering_angle': CANSignal(
        name='steering_angle',
        can_id=0x131,
        start_bit=0,
        length=16,
        scale=0.1,
        offset=-540.0,
        unit='deg'
    ),
    'yaw_rate': CANSignal(
        name='yaw_rate',
        can_id=0x132,
        start_bit=0,
        length=16,
        scale=0.01,
        offset=-327.68,
        unit='deg/s'
    ),
    'longitudinal_accel': CANSignal(
        name='longitudinal_accel',
        can_id=0x133,
        start_bit=0,
        length=16,
        scale=0.01,
        offset=-320.0,
        unit='m/s^2'
    ),
    'lateral_accel': CANSignal(
        name='lateral_accel',
        can_id=0x134,
        start_bit=0,
        length=16,
        scale=0.01,
        offset=-320.0,
        unit='m/s^2'
    ),
    'gear_position': CANSignal(
        name='gear_position',
        can_id=0x135,
        start_bit=0,
        length=3,
        scale=1,
        offset=0,
        unit='enum'
    ),
}

# 预定义的CAN配置
CAN_PRESETS = {
    "can0": CANConfig(
        name="can0",
        channel="can0",
        interface="socketcan",
        bitrate=500000,
        signals=VEHICLE_SIGNALS
    ),
    "can1": CANConfig(
        name="can1",
        channel="can1",
        interface="socketcan",
        bitrate=500000,
        signals={}  # 雷达CAN信号
    ),
}
