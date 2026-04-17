"""
自动驾驶系统 - 行为规划器
实现行为决策和解释
"""

from typing import Tuple, Optional
import numpy as np

from ..common.data_types import PerceptionResult, Obstacle
from ..common.enums import BehaviorType
from ..common.geometry import calculate_distance_2d
from ..logs.logger import Logger


class BehaviorPlanner:
    """行为规划器"""
    
    def __init__(self, config: dict = None):
        """
        初始化行为规划器
        
        Args:
            config: 配置参数
        """
        self._config = config or {}
        self._logger = Logger("BehaviorPlanner")
        
        # 参数
        self._reaction_time = self._config.get('reaction_time', 1.0)  # 反应时间
        self._comfort_deceleration = self._config.get('comfort_deceleration', 2.0)  # 舒适减速度
        self._emergency_deceleration = self._config.get('emergency_deceleration', 6.0)  # 紧急减速度
        
        # 当前行为
        self._current_behavior = BehaviorType.CRUISE
        self._behavior_start_time = 0.0
        
        self._logger.info("BehaviorPlanner initialized")
    
    def plan(self, perception_result: PerceptionResult, 
             current_speed: float) -> Tuple[BehaviorType, str]:
        """
        规划行为
        
        Args:
            perception_result: 感知结果
            current_speed: 当前车速 (m/s)
        
        Returns:
            (行为类型, 行为解释)
        """
        # 1. 检查紧急情况
        emergency_behavior = self._check_emergency(perception_result, current_speed)
        if emergency_behavior:
            return emergency_behavior, "Emergency situation detected"
        
        # 2. 分析前方交通状况
        front_obstacle = self._find_front_obstacle(perception_result)
        
        if front_obstacle:
            distance = front_obstacle.bbox.center.position.x
            relative_speed = current_speed - front_obstacle.velocity.x
            
            # 计算安全距离
            safe_distance = self._calculate_safe_distance(current_speed, relative_speed)
            
            if distance < safe_distance * 0.5:
                # 距离过近，需要停车
                return BehaviorType.STOP, f"Obstacle too close: {distance:.1f}m"
            
            elif distance < safe_distance:
                # 需要减速跟车
                return BehaviorType.FOLLOW, f"Following vehicle at {distance:.1f}m"
        
        # 3. 检查车道线
        if perception_result.lane_result:
            lane_result = perception_result.lane_result
            
            # 检查车道偏离
            if lane_result.lane_departure_warning:
                direction = lane_result.departure_direction
                if direction == "left":
                    return BehaviorType.LANE_CHANGE_RIGHT, "Correcting lane departure to left"
                else:
                    return BehaviorType.LANE_CHANGE_LEFT, "Correcting lane departure to right"
        
        # 4. 默认行为：巡航
        return BehaviorType.CRUISE, "Cruising at target speed"
    
    def _check_emergency(self, perception_result: PerceptionResult,
                        current_speed: float) -> Optional[BehaviorType]:
        """检查紧急情况"""
        if not perception_result.obstacle_result:
            return None
        
        # 检查前方是否有紧急情况
        for obstacle in perception_result.obstacle_result.obstacles:
            # 只考虑前方的障碍物
            if obstacle.bbox.center.position.x <= 0:
                continue
            
            # 计算距离
            distance = obstacle.bbox.center.position.x
            
            # 计算紧急制动距离
            emergency_stop_distance = self._calculate_emergency_stop_distance(current_speed)
            
            # 如果距离小于紧急制动距离，触发紧急停车
            if distance < emergency_stop_distance:
                self._logger.warning(f"Emergency stop! Obstacle at {distance:.1f}m")
                return BehaviorType.EMERGENCY_STOP
            
            # 如果距离很近且障碍物静止
            if distance < 5.0 and obstacle.is_static:
                return BehaviorType.STOP
        
        return None
    
    def _find_front_obstacle(self, perception_result: PerceptionResult) -> Optional[Obstacle]:
        """找到前方最近的障碍物"""
        if not perception_result.obstacle_result:
            return None
        
        front_obstacles = []
        
        for obstacle in perception_result.obstacle_result.obstacles:
            # 只考虑前方的障碍物
            if obstacle.bbox.center.position.x > 0:
                # 只考虑在车道内的障碍物
                if abs(obstacle.bbox.center.position.y) < 2.0:
                    front_obstacles.append(obstacle)
        
        if not front_obstacles:
            return None
        
        # 返回最近的障碍物
        return min(front_obstacles, key=lambda o: o.bbox.center.position.x)
    
    def _calculate_safe_distance(self, current_speed: float, relative_speed: float) -> float:
        """计算安全距离"""
        # 使用2秒规则
        time_gap = 2.0
        
        # 基础安全距离
        safe_distance = current_speed * time_gap
        
        # 考虑相对速度
        if relative_speed > 0:
            # 前车较慢，需要更多距离
            braking_distance = (relative_speed ** 2) / (2 * self._comfort_deceleration)
            safe_distance += braking_distance
        
        # 最小安全距离
        safe_distance = max(safe_distance, 3.0)
        
        return safe_distance
    
    def _calculate_emergency_stop_distance(self, current_speed: float) -> float:
        """计算紧急制动距离"""
        # 反应距离
        reaction_distance = current_speed * self._reaction_time
        
        # 制动距离
        braking_distance = (current_speed ** 2) / (2 * self._emergency_deceleration)
        
        # 安全余量
        safety_margin = 2.0
        
        return reaction_distance + braking_distance + safety_margin
    
    def explain_behavior(self, behavior_type: BehaviorType, 
                        perception_result: PerceptionResult) -> str:
        """
        解释行为决策
        
        Args:
            behavior_type: 行为类型
            perception_result: 感知结果
        
        Returns:
            行为解释文本
        """
        explanations = {
            BehaviorType.CRUISE: "保持当前车道巡航",
            BehaviorType.FOLLOW: "跟随前方车辆",
            BehaviorType.OVERTAKE: "执行超车动作",
            BehaviorType.LANE_CHANGE_LEFT: "向左变道",
            BehaviorType.LANE_CHANGE_RIGHT: "向右变道",
            BehaviorType.TURN_LEFT: "左转",
            BehaviorType.TURN_RIGHT: "右转",
            BehaviorType.U_TURN: "掉头",
            BehaviorType.STOP: "停车等待",
            BehaviorType.EMERGENCY_STOP: "紧急停车",
            BehaviorType.YIELD: "让行",
            BehaviorType.PULL_OVER: "靠边停车"
        }
        
        base_explanation = explanations.get(behavior_type, "未知行为")
        
        # 添加更多上下文信息
        if perception_result.obstacle_result:
            num_obstacles = len(perception_result.obstacle_result.obstacles)
            if num_obstacles > 0:
                base_explanation += f"（检测到{num_obstacles}个障碍物）"
        
        return base_explanation
