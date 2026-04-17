"""
自动驾驶系统 - 规划模块
整合行为规划、轨迹生成和轨迹优化
"""

from typing import Optional, Dict, Any, List
import threading
import time
import numpy as np

from .behavior_planner import BehaviorPlanner
from .trajectory_generator import TrajectoryGenerator
from .trajectory_optimizer import TrajectoryOptimizer
from ..common.data_types import (
    PerceptionResult, CANData, PlanningResult, Trajectory,
    Timestamp, Vector3D
)
from ..common.enums import ModuleState, BehaviorType
from ..communication.message_bus import MessageBus, Message
from ..communication import Publisher, Subscriber
from ..config.config_manager import ConfigManager
from ..logs.logger import Logger


class PlanningModule:
    """规划模块"""
    
    def __init__(self, message_bus: Optional[MessageBus] = None):
        """
        初始化规划模块
        
        Args:
            message_bus: 消息总线实例
        """
        self._message_bus = message_bus or MessageBus()
        self._logger = Logger("PlanningModule")
        
        # 状态
        self._state = ModuleState.UNINITIALIZED
        self._enabled = True
        
        # 子模块
        self._behavior_planner: Optional[BehaviorPlanner] = None
        self._trajectory_generator: Optional[TrajectoryGenerator] = None
        self._trajectory_optimizer: Optional[TrajectoryOptimizer] = None
        
        # 发布者
        self._trajectory_publisher = Publisher("planning", "planning/trajectory", self._message_bus)
        self._behavior_publisher = Publisher("planning", "planning/behavior", self._message_bus)
        self._planning_publisher = Publisher("planning", "planning/all", self._message_bus)
        
        # 订阅者
        self._subscriber = Subscriber("planning", self._message_bus)
        
        # 输入数据缓存
        self._latest_perception_result: Optional[PerceptionResult] = None
        self._latest_can_data: Optional[CANData] = None
        
        # 数据锁
        self._data_lock = threading.Lock()
        
        # 处理线程
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._processing_frequency = 10.0  # Hz
        
        # 配置
        self._config: Optional[Dict[str, Any]] = None
        
        # 当前规划结果
        self._current_planning_result: Optional[PlanningResult] = None
        
        # 统计
        self._processing_count = 0
        self._processing_time_ms = 0.0
    
    def initialize(self, config_manager: Optional[ConfigManager] = None) -> bool:
        """初始化规划模块"""
        if self._state != ModuleState.UNINITIALIZED:
            self._logger.warning(f"Cannot initialize in state {self._state.name}")
            return False
        
        self._state = ModuleState.INITIALIZING
        
        try:
            if config_manager is None:
                config_manager = ConfigManager()
            
            module_config = config_manager.get_module_config('planning')
            if module_config:
                self._enabled = module_config.enabled
                self._processing_frequency = module_config.execution_frequency
                self._config = module_config.algorithm_params
            
            # 初始化子模块
            self._behavior_planner = BehaviorPlanner(self._config.get('behavior_planning', {}))
            self._trajectory_generator = TrajectoryGenerator(self._config.get('trajectory_generation', {}))
            self._trajectory_optimizer = TrajectoryOptimizer(self._config.get('trajectory_optimization', {}))
            
            # 设置订阅
            self._setup_subscriptions()
            
            self._state = ModuleState.READY
            self._logger.info("PlanningModule initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            self._state = ModuleState.ERROR
            return False
    
    def _setup_subscriptions(self) -> None:
        """设置数据订阅"""
        # 订阅感知结果
        self._subscriber.subscribe(
            topic="perception/fusion",
            callback=self._on_perception_result
        )
        
        # 订阅CAN数据
        self._subscriber.subscribe(
            topic="sensor/can/vehicle",
            callback=self._on_can_data
        )
    
    def _on_perception_result(self, message: Message) -> None:
        """处理感知结果"""
        with self._data_lock:
            self._latest_perception_result = message.data
    
    def _on_can_data(self, message: Message) -> None:
        """处理CAN数据"""
        with self._data_lock:
            self._latest_can_data = message.data
    
    def start(self) -> bool:
        """启动规划模块"""
        if self._state not in [ModuleState.READY, ModuleState.SHUTDOWN]:
            self._logger.warning(f"Cannot start in state {self._state.name}")
            return False
        
        if not self._enabled:
            self._logger.info("PlanningModule is disabled")
            return False
        
        self._running = True
        self._state = ModuleState.RUNNING
        
        # 启动处理线程
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="PlanningProcessing",
            daemon=True
        )
        self._processing_thread.start()
        
        self._logger.info("PlanningModule started")
        return True
    
    def stop(self) -> None:
        """停止规划模块"""
        self._running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        
        self._subscriber.unsubscribe_all()
        self._state = ModuleState.SHUTDOWN
        
        self._logger.info("PlanningModule stopped")
    
    def _processing_loop(self) -> None:
        """处理循环"""
        period = 1.0 / self._processing_frequency
        
        while self._running:
            start_time = time.time()
            
            try:
                # 执行规划
                self._plan()
                
            except Exception as e:
                self._logger.error(f"Planning error: {e}")
            
            # 控制处理频率
            elapsed = time.time() - start_time
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _plan(self) -> None:
        """执行规划"""
        start_time = time.time()
        
        # 获取数据
        with self._data_lock:
            perception_result = self._latest_perception_result
            can_data = self._latest_can_data
        
        # 检查是否有足够的数据
        if perception_result is None:
            return
        
        # 获取当前车速
        current_speed = can_data.vehicle_speed if can_data else 0.0
        
        # 1. 行为规划
        behavior_type, behavior_explanation = self._behavior_planner.plan(
            perception_result=perception_result,
            current_speed=current_speed
        )
        
        # 发布行为决策
        self._behavior_publisher.publish({
            'behavior_type': behavior_type.name,
            'explanation': behavior_explanation,
            'timestamp': Timestamp.now()
        })
        
        # 2. 轨迹生成
        candidate_trajectories = self._trajectory_generator.generate(
            perception_result=perception_result,
            behavior_type=behavior_type,
            current_speed=current_speed
        )
        
        if not candidate_trajectories:
            self._logger.warning("No candidate trajectories generated")
            return
        
        # 3. 轨迹评估和选择
        selected_trajectory = self._select_best_trajectory(
            candidate_trajectories,
            perception_result
        )
        
        # 4. 轨迹优化
        if selected_trajectory:
            optimized_trajectory = self._trajectory_optimizer.optimize(
                selected_trajectory,
                perception_result
            )
        else:
            optimized_trajectory = selected_trajectory
        
        # 计算目标速度
        target_speed = self._calculate_target_speed(
            optimized_trajectory,
            perception_result,
            current_speed
        )
        
        # 计算控制指令
        steering, throttle, brake = self._calculate_control_commands(
            optimized_trajectory,
            can_data
        )
        
        # 创建规划结果
        processing_time = (time.time() - start_time) * 1000  # ms
        
        planning_result = PlanningResult(
            timestamp=Timestamp.now(),
            behavior_type=behavior_type.name.lower(),
            behavior_explanation=behavior_explanation,
            selected_trajectory=optimized_trajectory,
            candidate_trajectories=candidate_trajectories,
            target_speed=target_speed,
            steering_angle=steering,
            throttle=throttle,
            brake=brake,
            processing_time_ms=processing_time,
            is_valid=True
        )
        
        # 发布规划结果
        self._trajectory_publisher.publish(optimized_trajectory)
        self._planning_publisher.publish(planning_result)
        
        # 更新当前结果
        self._current_planning_result = planning_result
        
        # 更新统计
        self._processing_count += 1
        self._processing_time_ms = processing_time
    
    def _select_best_trajectory(self, trajectories: List[Trajectory],
                                perception_result: PerceptionResult) -> Optional[Trajectory]:
        """选择最佳轨迹"""
        if not trajectories:
            return None
        
        # 过滤不可行轨迹
        feasible_trajectories = [t for t in trajectories if t.is_feasible]
        
        if not feasible_trajectories:
            # 如果没有可行轨迹，选择代价最小的
            feasible_trajectories = trajectories
        
        # 选择代价最小的轨迹
        best_trajectory = min(feasible_trajectories, key=lambda t: t.cost)
        
        return best_trajectory
    
    def _calculate_target_speed(self, trajectory: Optional[Trajectory],
                               perception_result: PerceptionResult,
                               current_speed: float) -> float:
        """计算目标速度"""
        if trajectory is None or not trajectory.points:
            return 0.0
        
        # 获取轨迹中的最大速度
        max_speed = max(p.longitudinal_velocity for p in trajectory.points)
        
        # 考虑障碍物
        if perception_result.obstacle_result:
            for obstacle in perception_result.obstacle_result.obstacles:
                # 如果前方有障碍物，降低速度
                if obstacle.bbox.center.position.x > 0 and \
                   abs(obstacle.bbox.center.position.y) < 2.0:
                    distance = obstacle.bbox.center.position.x
                    if distance < 20.0:
                        max_speed = min(max_speed, distance * 0.5)
        
        return max_speed
    
    def _calculate_control_commands(self, trajectory: Optional[Trajectory],
                                   can_data: Optional[CANData]) -> tuple:
        """计算控制指令"""
        if trajectory is None or not trajectory.points:
            return 0.0, 0.0, 1.0  # 停车
        
        # 简化的控制计算
        # 实际应该使用更复杂的控制算法
        
        # 获取轨迹的第一个点
        first_point = trajectory.points[0]
        
        # 计算方向盘角度
        steering = np.degrees(first_point.theta) * 2.0  # 简化的映射
        steering = np.clip(steering, -35, 35)
        
        # 计算油门/刹车
        target_speed = first_point.longitudinal_velocity
        current_speed = can_data.vehicle_speed if can_data else 0.0
        
        speed_error = target_speed - current_speed
        
        if speed_error > 0:
            throttle = min(speed_error * 0.5, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(speed_error) * 0.5, 1.0)
        
        return steering, throttle, brake
    
    def get_current_planning_result(self) -> Optional[PlanningResult]:
        """获取当前规划结果"""
        return self._current_planning_result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.name,
            'enabled': self._enabled,
            'processing_count': self._processing_count,
            'processing_time_ms': self._processing_time_ms,
            'frequency': self._processing_frequency
        }
