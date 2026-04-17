"""
自动驾驶系统 - CAN总线接口
实现车辆CAN总线数据读取
"""

import numpy as np
from typing import Optional, Dict, Callable
import can  # python-can库
import threading

from ..common.data_types import CANData, SensorConfig, Timestamp
from ..communication.message_bus import MessageBus, Publisher
from ..logs.logger import Logger


class CANInterface:
    """CAN总线接口"""
    
    # 标准CAN ID定义
    CAN_ID_VEHICLE_SPEED = 0x100
    CAN_ID_STEERING_ANGLE = 0x101
    CAN_ID_YAW_RATE = 0x102
    CAN_ID_ACCEL = 0x103
    CAN_ID_GEAR = 0x104
    CAN_ID_TURN_SIGNAL = 0x105
    
    def __init__(self, config: SensorConfig, message_bus: Optional[MessageBus] = None):
        """
        初始化CAN接口
        
        Args:
            config: 传感器配置
            message_bus: 消息总线实例
        """
        self._config = config
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger("CANInterface")
        
        # CAN总线
        self._can_bus: Optional[can.Bus] = None
        self._can_channel = config.parameters.get('can_channel', 'can0')
        self._bitrate = config.parameters.get('bitrate', 500000)
        
        # 发布者
        self._publisher = Publisher("can_interface", "sensor/can/vehicle", self._message_bus)
        
        # 车辆状态
        self._vehicle_state = {
            'speed': 0.0,
            'steering_angle': 0.0,
            'yaw_rate': 0.0,
            'longitudinal_accel': 0.0,
            'lateral_accel': 0.0,
            'gear_position': 0,
            'turn_signal': 0
        }
        
        # 运行状态
        self._running = False
        self._receive_thread: Optional[threading.Thread] = None
        
        # 仿真模式
        self._simulation_mode = False
        
        # 回调函数
        self._callbacks: Dict[int, Callable] = {}
    
    def initialize(self) -> bool:
        """初始化CAN接口"""
        try:
            # 尝试连接CAN总线
            self._can_bus = can.Bus(
                channel=self._can_channel,
                bustype='socketcan',
                bitrate=self._bitrate
            )
            
            self._logger.info(f"CAN interface initialized on {self._can_channel}")
            return True
            
        except Exception as e:
            self._logger.warning(f"CAN initialization failed: {e}, using simulation mode")
            self._simulation_mode = True
            return True
    
    def start(self) -> bool:
        """启动CAN接收"""
        self._running = True
        
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name="CANReceive",
            daemon=True
        )
        self._receive_thread.start()
        
        self._logger.info("CAN interface started")
        return True
    
    def stop(self) -> None:
        """停止CAN接收"""
        self._running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        
        if self._can_bus:
            self._can_bus.shutdown()
            self._can_bus = None
        
        self._logger.info("CAN interface stopped")
    
    def _receive_loop(self) -> None:
        """CAN接收循环"""
        while self._running:
            try:
                if self._simulation_mode:
                    self._generate_simulation_data()
                    import time
                    time.sleep(0.01)  # 100Hz仿真
                else:
                    # 接收CAN消息
                    msg = self._can_bus.recv(timeout=0.01)
                    
                    if msg:
                        self._process_can_message(msg)
            
            except Exception as e:
                self._logger.error(f"CAN receive error: {e}")
    
    def _process_can_message(self, msg: can.Message) -> None:
        """处理CAN消息"""
        can_id = msg.arbitration_id
        data = msg.data
        
        # 解析车辆状态
        if can_id == self.CAN_ID_VEHICLE_SPEED:
            self._vehicle_state['speed'] = self._parse_speed(data)
        elif can_id == self.CAN_ID_STEERING_ANGLE:
            self._vehicle_state['steering_angle'] = self._parse_steering(data)
        elif can_id == self.CAN_ID_YAW_RATE:
            self._vehicle_state['yaw_rate'] = self._parse_yaw_rate(data)
        elif can_id == self.CAN_ID_ACCEL:
            self._vehicle_state['longitudinal_accel'], \
            self._vehicle_state['lateral_accel'] = self._parse_acceleration(data)
        elif can_id == self.CAN_ID_GEAR:
            self._vehicle_state['gear_position'] = self._parse_gear(data)
        elif can_id == self.CAN_ID_TURN_SIGNAL:
            self._vehicle_state['turn_signal'] = self._parse_turn_signal(data)
        
        # 调用注册的回调
        if can_id in self._callbacks:
            self._callbacks[can_id](msg)
        
        # 发布车辆状态
        self._publish_vehicle_state()
    
    def _parse_speed(self, data: bytes) -> float:
        """解析车速"""
        # 假设：0.01 km/h/bit, 偏移-100 km/h
        speed_kmh = (data[0] | (data[1] << 8)) * 0.01 - 100.0
        return speed_kmh / 3.6  # 转换为 m/s
    
    def _parse_steering(self, data: bytes) -> float:
        """解析方向盘角度"""
        # 假设：0.1 deg/bit, 偏移-1000 deg
        return (data[0] | (data[1] << 8)) * 0.1 - 1000.0
    
    def _parse_yaw_rate(self, data: bytes) -> float:
        """解析横摆角速度"""
        # 假设：0.01 deg/s/bit, 偏移-100 deg/s
        return np.radians((data[0] | (data[1] << 8)) * 0.01 - 100.0)
    
    def _parse_acceleration(self, data: bytes) -> tuple:
        """解析加速度"""
        # 假设：0.01 m/s^2/bit, 偏移-100 m/s^2
        long_accel = (data[0] | (data[1] << 8)) * 0.01 - 100.0
        lat_accel = (data[2] | (data[3] << 8)) * 0.01 - 100.0
        return long_accel, lat_accel
    
    def _parse_gear(self, data: bytes) -> int:
        """解析档位"""
        return data[0]
    
    def _parse_turn_signal(self, data: bytes) -> int:
        """解析转向灯"""
        return data[0]
    
    def _generate_simulation_data(self) -> None:
        """生成仿真车辆数据"""
        # 模拟车辆运动
        import time
        t = time.time()
        
        self._vehicle_state['speed'] = 10.0 + 2.0 * np.sin(t * 0.5)  # m/s
        self._vehicle_state['steering_angle'] = 5.0 * np.sin(t * 0.3)  # deg
        self._vehicle_state['yaw_rate'] = 0.05 * np.sin(t * 0.3)  # rad/s
        self._vehicle_state['longitudinal_accel'] = 0.5 * np.cos(t * 0.5)  # m/s^2
        self._vehicle_state['lateral_accel'] = 0.2 * np.sin(t * 0.3)  # m/s^2
        self._vehicle_state['gear_position'] = 4  # D档
        self._vehicle_state['turn_signal'] = 0
        
        # 发布车辆状态
        self._publish_vehicle_state()
    
    def _publish_vehicle_state(self) -> None:
        """发布车辆状态"""
        can_data = CANData(
            timestamp=Timestamp.now(),
            can_id=0,
            can_channel=self._can_channel,
            vehicle_speed=self._vehicle_state['speed'],
            steering_angle=self._vehicle_state['steering_angle'],
            yaw_rate=self._vehicle_state['yaw_rate'],
            longitudinal_accel=self._vehicle_state['longitudinal_accel'],
            lateral_accel=self._vehicle_state['lateral_accel'],
            gear_position=self._vehicle_state['gear_position'],
            turn_signal=self._vehicle_state['turn_signal']
        )
        
        self._publisher.publish(can_data)
    
    def register_callback(self, can_id: int, callback: Callable[[can.Message], None]) -> None:
        """注册CAN消息回调"""
        self._callbacks[can_id] = callback
    
    def get_vehicle_state(self) -> Dict:
        """获取车辆状态"""
        return self._vehicle_state.copy()
    
    def send_message(self, can_id: int, data: bytes) -> bool:
        """发送CAN消息"""
        if self._simulation_mode or not self._can_bus:
            return False
        
        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            )
            self._can_bus.send(msg)
            return True
        except Exception as e:
            self._logger.error(f"CAN send error: {e}")
            return False
