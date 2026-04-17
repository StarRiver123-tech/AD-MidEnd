"""
Behavior Explainer and Scoring Module
行为解释和评分模块
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

from lattice_generator import Trajectory, VehicleState, Obstacle, LaneInfo


@dataclass
class BehaviorExplanation:
    """行为解释"""
    behavior_type: str  # 行为类型
    description: str    # 自然语言描述
    safety_score: float  # 安全性评分
    comfort_score: float  # 舒适性评分
    efficiency_score: float  # 效率评分
    legality_score: float  # 合法性评分
    total_score: float  # 综合评分
    risk_factors: List[str]  # 风险因素
    recommendations: List[str]  # 建议


class BehaviorExplainer:
    """
    行为解释器
    对生成的轨迹进行行为解析和评估
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 评分权重
        self.safety_weight = config.get('safety_weight', 0.4)
        self.comfort_weight = config.get('comfort_weight', 0.2)
        self.efficiency_weight = config.get('efficiency_weight', 0.2)
        self.legality_weight = config.get('legality_weight', 0.2)
        
        # 评分阈值
        self.min_safety_score = config.get('min_safety_score', 0.6)
        self.min_comfort_score = config.get('min_comfort_score', 0.5)
        
        # 参考参数
        self.target_speed = config.get('target_speed', 15.0)  # 目标速度 m/s
        self.speed_limit = config.get('speed_limit', 20.0)  # 限速
    
    def explain_and_score(
        self,
        trajectory: Trajectory,
        current_state: VehicleState,
        lane_info: LaneInfo,
        obstacles: List[Obstacle],
        traffic_signs: Optional[List[Dict]] = None
    ) -> BehaviorExplanation:
        """
        对轨迹进行解释和评分
        
        Args:
            trajectory: 待评估轨迹
            current_state: 当前车辆状态
            lane_info: 车道信息
            obstacles: 障碍物列表
            traffic_signs: 交通标志列表
            
        Returns:
            行为解释和评分
        """
        # 确定行为类型
        behavior_type = self._classify_behavior(
            trajectory, current_state, lane_info
        )
        
        # 生成描述
        description = self._generate_description(
            trajectory, behavior_type, current_state
        )
        
        # 计算各项评分
        safety_score = self._evaluate_safety(
            trajectory, obstacles, lane_info
        )
        
        comfort_score = self._evaluate_comfort(trajectory)
        
        efficiency_score = self._evaluate_efficiency(
            trajectory, lane_info
        )
        
        legality_score = self._evaluate_legality(
            trajectory, lane_info, traffic_signs
        )
        
        # 计算综合评分
        total_score = (
            self.safety_weight * safety_score +
            self.comfort_weight * comfort_score +
            self.efficiency_weight * efficiency_score +
            self.legality_weight * legality_score
        )
        
        # 识别风险因素
        risk_factors = self._identify_risk_factors(
            trajectory, obstacles, lane_info, traffic_signs,
            safety_score, comfort_score, efficiency_score, legality_score
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(
            risk_factors, behavior_type
        )
        
        explanation = BehaviorExplanation(
            behavior_type=behavior_type,
            description=description,
            safety_score=safety_score,
            comfort_score=comfort_score,
            efficiency_score=efficiency_score,
            legality_score=legality_score,
            total_score=total_score,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
        
        # 将评分存储到轨迹中
        trajectory.cost = 1.0 - total_score
        
        return explanation
    
    def _classify_behavior(
        self,
        trajectory: Trajectory,
        current_state: VehicleState,
        lane_info: LaneInfo
    ) -> str:
        """
        分类行为类型
        """
        # 使用轨迹自带的行为类型
        if trajectory.behavior_type != "unknown":
            return trajectory.behavior_type
        
        # 分析轨迹特征
        start_x, start_y = trajectory.x[0], trajectory.y[0]
        end_x, end_y = trajectory.x[-1], trajectory.y[-1]
        
        # 计算横向位移
        _, ref_start_x, ref_start_y = lane_info.get_closest_point(start_x, start_y)
        _, ref_end_x, ref_end_y = lane_info.get_closest_point(end_x, end_y)
        
        # 计算车道方向
        lane_theta = lane_info.get_direction_at_index(0)
        
        # 计算横向偏移变化
        start_lat = -(start_x - ref_start_x) * np.sin(lane_theta) + \
                    (start_y - ref_start_y) * np.cos(lane_theta)
        end_lat = -(end_x - ref_end_x) * np.sin(lane_theta) + \
                  (end_y - ref_end_y) * np.cos(lane_theta)
        
        lateral_change = end_lat - start_lat
        lane_width = lane_info.width
        
        # 分析速度变化
        v_change = trajectory.v[-1] - trajectory.v[0]
        
        # 分类
        if abs(lateral_change) > lane_width * 0.5:
            if lateral_change > 0:
                return "change_left"
            else:
                return "change_right"
        elif v_change > 2.0:
            return "accelerate"
        elif v_change < -2.0:
            return "decelerate"
        else:
            return "keep_lane"
    
    def _generate_description(
        self,
        trajectory: Trajectory,
        behavior_type: str,
        current_state: VehicleState
    ) -> str:
        """
        生成自然语言描述
        """
        avg_speed = np.mean(trajectory.v)
        max_acc = np.max(np.abs(trajectory.a))
        duration = trajectory.t[-1]
        length = trajectory.get_length()
        
        descriptions = {
            "keep_lane": f"保持车道行驶，平均速度{avg_speed:.1f}m/s，"
                        f"持续时间{duration:.1f}s，行驶距离{length:.1f}m",
            "change_left": f"向左换道，平均速度{avg_speed:.1f}m/s，"
                          f"持续时间{duration:.1f}s",
            "change_right": f"向右换道，平均速度{avg_speed:.1f}m/s，"
                           f"持续时间{duration:.1f}s",
            "accelerate": f"加速行驶，从{current_state.v:.1f}m/s加速到"
                         f"{trajectory.v[-1]:.1f}m/s，持续时间{duration:.1f}s",
            "decelerate": f"减速行驶，从{current_state.v:.1f}m/s减速到"
                         f"{trajectory.v[-1]:.1f}m/s，持续时间{duration:.1f}s",
            "unknown": f"行驶轨迹，平均速度{avg_speed:.1f}m/s，"
                      f"持续时间{duration:.1f}s"
        }
        
        return descriptions.get(behavior_type, descriptions["unknown"])
    
    def _evaluate_safety(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle],
        lane_info: LaneInfo
    ) -> float:
        """
        评估安全性
        """
        scores = []
        
        # 1. 障碍物距离评分
        min_obs_distance = float('inf')
        for i, t in enumerate(trajectory.t):
            ego_x, ego_y = trajectory.x[i], trajectory.y[i]
            for obs in obstacles:
                obs_x, obs_y = obs.get_position_at_time(t)
                dist = np.sqrt((ego_x - obs_x)**2 + (ego_y - obs_y)**2)
                min_obs_distance = min(min_obs_distance, dist)
        
        if min_obs_distance == float('inf'):
            obs_score = 1.0
        else:
            # 距离越近，评分越低
            safe_dist = 5.0
            obs_score = min(1.0, min_obs_distance / safe_dist)
        scores.append(obs_score)
        
        # 2. 曲率安全性
        max_kappa = np.max(np.abs(trajectory.kappa))
        kappa_limit = 0.15  # 安全曲率限制
        kappa_score = max(0.0, 1.0 - max_kappa / kappa_limit)
        scores.append(kappa_score)
        
        # 3. 加速度安全性
        max_acc = np.max(np.abs(trajectory.a))
        acc_limit = 3.0  # m/s^2
        acc_score = max(0.0, 1.0 - max_acc / acc_limit)
        scores.append(acc_score)
        
        # 4. 车道偏离评分
        lane_deviation_score = self._evaluate_lane_deviation(
            trajectory, lane_info
        )
        scores.append(lane_deviation_score)
        
        return np.mean(scores)
    
    def _evaluate_lane_deviation(
        self,
        trajectory: Trajectory,
        lane_info: LaneInfo
    ) -> float:
        """
        评估车道偏离程度
        """
        deviations = []
        lane_width = lane_info.width
        
        for i in range(len(trajectory.x)):
            x, y = trajectory.x[i], trajectory.y[i]
            _, ref_x, ref_y = lane_info.get_closest_point(x, y)
            
            dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
            
            # 偏离车道中心超过半个车道宽度则扣分
            if dist > lane_width / 2:
                deviation = max(0.0, 1.0 - (dist - lane_width/2) / lane_width)
            else:
                deviation = 1.0
            
            deviations.append(deviation)
        
        return np.mean(deviations)
    
    def _evaluate_comfort(self, trajectory: Trajectory) -> float:
        """
        评估舒适性
        """
        scores = []
        
        # 1. 加速度平滑度
        acc_jerk = np.diff(trajectory.a) / np.diff(trajectory.t)
        max_jerk = np.max(np.abs(acc_jerk)) if len(acc_jerk) > 0 else 0
        jerk_limit = 3.0  # m/s^3
        jerk_score = max(0.0, 1.0 - max_jerk / jerk_limit)
        scores.append(jerk_score)
        
        # 2. 曲率变化平滑度
        kappa_change = np.diff(trajectory.kappa)
        max_kappa_change = np.max(np.abs(kappa_change)) if len(kappa_change) > 0 else 0
        kappa_smooth_score = max(0.0, 1.0 - max_kappa_change / 0.1)
        scores.append(kappa_smooth_score)
        
        # 3. 速度平滑度
        vel_change = np.diff(trajectory.v)
        max_vel_change = np.max(np.abs(vel_change)) if len(vel_change) > 0 else 0
        vel_smooth_score = max(0.0, 1.0 - max_vel_change / 2.0)
        scores.append(vel_smooth_score)
        
        # 4. 横向加速度
        lateral_acc = trajectory.v**2 * np.abs(trajectory.kappa)
        max_lat_acc = np.max(lateral_acc)
        lat_acc_limit = 2.0  # m/s^2
        lat_acc_score = max(0.0, 1.0 - max_lat_acc / lat_acc_limit)
        scores.append(lat_acc_score)
        
        return np.mean(scores)
    
    def _evaluate_efficiency(
        self,
        trajectory: Trajectory,
        lane_info: LaneInfo
    ) -> float:
        """
        评估效率
        """
        scores = []
        
        # 1. 平均速度接近目标速度的程度
        avg_speed = np.mean(trajectory.v)
        speed_diff = abs(avg_speed - self.target_speed)
        speed_score = max(0.0, 1.0 - speed_diff / self.target_speed)
        scores.append(speed_score)
        
        # 2. 轨迹长度效率
        start_x, start_y = trajectory.x[0], trajectory.y[0]
        end_x, end_y = trajectory.x[-1], trajectory.y[-1]
        
        # 计算直线距离
        straight_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        actual_length = trajectory.get_length()
        
        if actual_length > 0:
            length_efficiency = straight_dist / actual_length
            scores.append(length_efficiency)
        
        # 3. 时间效率
        duration = trajectory.t[-1]
        expected_duration = actual_length / max(self.target_speed, 0.1)
        time_efficiency = min(1.0, expected_duration / duration)
        scores.append(time_efficiency)
        
        return np.mean(scores)
    
    def _evaluate_legality(
        self,
        trajectory: Trajectory,
        lane_info: LaneInfo,
        traffic_signs: Optional[List[Dict]]
    ) -> float:
        """
        评估合法性
        """
        scores = []
        
        # 1. 速度限制检查
        max_speed = np.max(trajectory.v)
        if max_speed > self.speed_limit:
            speed_legal_score = max(0.0, 1.0 - (max_speed - self.speed_limit) / 5.0)
        else:
            speed_legal_score = 1.0
        scores.append(speed_legal_score)
        
        # 2. 车道边界检查
        lane_boundary_score = self._check_lane_boundary(trajectory, lane_info)
        scores.append(lane_boundary_score)
        
        # 3. 交通标志遵守
        if traffic_signs:
            traffic_sign_score = self._check_traffic_signs(
                trajectory, traffic_signs
            )
            scores.append(traffic_sign_score)
        
        return np.mean(scores)
    
    def _check_lane_boundary(
        self,
        trajectory: Trajectory,
        lane_info: LaneInfo
    ) -> float:
        """
        检查是否超出车道边界
        """
        violations = 0
        total_points = len(trajectory.x)
        lane_width = lane_info.width
        
        for i in range(total_points):
            x, y = trajectory.x[i], trajectory.y[i]
            _, ref_x, ref_y = lane_info.get_closest_point(x, y)
            
            dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
            
            # 超出车道边界
            if dist > lane_width:
                violations += 1
        
        return 1.0 - violations / total_points if total_points > 0 else 1.0
    
    def _check_traffic_signs(
        self,
        trajectory: Trajectory,
        traffic_signs: List[Dict]
    ) -> float:
        """
        检查是否遵守交通标志
        """
        # 简化实现，实际应该根据具体标志类型进行检查
        return 1.0
    
    def _identify_risk_factors(
        self,
        trajectory: Trajectory,
        obstacles: List[Obstacle],
        lane_info: LaneInfo,
        traffic_signs: Optional[List[Dict]],
        safety_score: float,
        comfort_score: float,
        efficiency_score: float,
        legality_score: float
    ) -> List[str]:
        """
        识别风险因素
        """
        risks = []
        
        # 安全性风险
        if safety_score < 0.6:
            # 检查障碍物距离
            for i, t in enumerate(trajectory.t):
                ego_x, ego_y = trajectory.x[i], trajectory.y[i]
                for obs in obstacles:
                    obs_x, obs_y = obs.get_position_at_time(t)
                    dist = np.sqrt((ego_x - obs_x)**2 + (ego_y - obs_y)**2)
                    if dist < 3.0:
                        risks.append(f"近距离接近障碍物，距离{dist:.1f}m")
                        break
            
            # 检查曲率
            max_kappa = np.max(np.abs(trajectory.kappa))
            if max_kappa > 0.15:
                risks.append(f"曲率过大，最大曲率{max_kappa:.3f}")
            
            # 检查加速度
            max_acc = np.max(np.abs(trajectory.a))
            if max_acc > 3.0:
                risks.append(f"加速度过大，最大加速度{max_acc:.1f}m/s²")
        
        # 舒适性风险
        if comfort_score < 0.5:
            acc_jerk = np.diff(trajectory.a) / np.diff(trajectory.t)
            max_jerk = np.max(np.abs(acc_jerk)) if len(acc_jerk) > 0 else 0
            if max_jerk > 3.0:
                risks.append(f"加加速度过大，最大jerk{max_jerk:.1f}m/s³")
        
        # 效率风险
        if efficiency_score < 0.5:
            avg_speed = np.mean(trajectory.v)
            if avg_speed < self.target_speed * 0.5:
                risks.append(f"速度过低，平均速度{avg_speed:.1f}m/s")
        
        # 合法性风险
        if legality_score < 0.6:
            max_speed = np.max(trajectory.v)
            if max_speed > self.speed_limit:
                risks.append(f"超速行驶，最高速度{max_speed:.1f}m/s")
        
        return risks if risks else ["无明显风险"]
    
    def _generate_recommendations(
        self,
        risk_factors: List[str],
        behavior_type: str
    ) -> List[str]:
        """
        生成建议
        """
        recommendations = []
        
        for risk in risk_factors:
            if "障碍物" in risk:
                recommendations.append("建议增加与障碍物的安全距离")
            elif "曲率" in risk:
                recommendations.append("建议降低转向速率，使轨迹更平滑")
            elif "加速度" in risk:
                recommendations.append("建议平缓加速/减速")
            elif "速度过低" in risk:
                recommendations.append("建议适当提高行驶速度")
            elif "超速" in risk:
                recommendations.append("建议降低车速以符合限速要求")
        
        # 根据行为类型添加建议
        if behavior_type == "change_left" or behavior_type == "change_right":
            recommendations.append("换道时请确认盲区安全")
        elif behavior_type == "accelerate":
            recommendations.append("加速时请注意前方路况")
        elif behavior_type == "decelerate":
            recommendations.append("减速时请注意后方车辆")
        
        return recommendations if recommendations else ["保持当前行驶状态"]
    
    def rank_trajectories(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation]
    ) -> List[Tuple[Trajectory, BehaviorExplanation]]:
        """
        根据评分对轨迹进行排序
        
        Returns:
            按评分降序排列的(轨迹, 解释)列表
        """
        scored_trajectories = list(zip(trajectories, explanations))
        scored_trajectories.sort(key=lambda x: x[1].total_score, reverse=True)
        return scored_trajectories
