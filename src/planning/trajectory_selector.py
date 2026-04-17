"""
Trajectory Selection Module
轨迹选择模块
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import math

from lattice_generator import Trajectory, VehicleState, Obstacle, LaneInfo
from behavior_explainer import BehaviorExplanation


@dataclass
class SelectionCriteria:
    """选择标准"""
    min_safety_score: float = 0.6
    min_comfort_score: float = 0.5
    min_efficiency_score: float = 0.4
    min_legality_score: float = 0.7
    prefer_behavior: Optional[str] = None  # 偏好行为类型
    prefer_speed_range: Optional[Tuple[float, float]] = None  # 偏好速度范围


class TrajectorySelector:
    """
    轨迹选择器
    根据轨迹和行为解释及评分选择最优轨迹
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 选择参数
        self.num_candidates = config.get('num_candidates', 5)  # 候选轨迹数量
        self.use_preference_model = config.get('use_preference_model', False)
        
        # 损失函数权重
        self.safety_weight = config.get('safety_weight', 0.4)
        self.comfort_weight = config.get('comfort_weight', 0.2)
        self.efficiency_weight = config.get('efficiency_weight', 0.2)
        self.smoothness_weight = config.get('smoothness_weight', 0.1)
        self.consistency_weight = config.get('consistency_weight', 0.1)
        
        # 偏好模型参数 (用于模型训练)
        self.preference_weights = np.array([
            self.safety_weight,
            self.comfort_weight,
            self.efficiency_weight,
            self.smoothness_weight,
            self.consistency_weight
        ])
    
    def select_trajectory(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation],
        current_state: VehicleState,
        previous_trajectory: Optional[Trajectory] = None,
        criteria: Optional[SelectionCriteria] = None
    ) -> Tuple[Trajectory, BehaviorExplanation]:
        """
        选择最优轨迹
        
        Args:
            trajectories: 候选轨迹列表
            explanations: 对应的行为解释列表
            current_state: 当前车辆状态
            previous_trajectory: 上一时刻选择的轨迹 (用于一致性检查)
            criteria: 选择标准
            
        Returns:
            选择的轨迹和行为解释
        """
        if criteria is None:
            criteria = SelectionCriteria()
        
        # 第一步：过滤不满足基本要求的轨迹
        filtered = self._filter_trajectories(
            trajectories, explanations, criteria
        )
        
        if not filtered:
            # 如果没有满足条件的轨迹，选择评分最高的
            best_idx = np.argmax([e.total_score for e in explanations])
            return trajectories[best_idx], explanations[best_idx]
        
        filtered_trajs, filtered_expls = zip(*filtered)
        filtered_trajs = list(filtered_trajs)
        filtered_expls = list(filtered_expls)
        
        # 第二步：计算综合损失
        losses = self._compute_losses(
            filtered_trajs, filtered_expls, current_state, previous_trajectory
        )
        
        # 第三步：选择损失最小的轨迹
        best_idx = np.argmin(losses)
        
        return filtered_trajs[best_idx], filtered_expls[best_idx]
    
    def select_multiple_trajectories(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation],
        current_state: VehicleState,
        previous_trajectory: Optional[Trajectory] = None,
        criteria: Optional[SelectionCriteria] = None,
        num_selections: int = 3
    ) -> List[Tuple[Trajectory, BehaviorExplanation]]:
        """
        选择多条候选轨迹
        
        用于提供备选方案或集成规划
        """
        if criteria is None:
            criteria = SelectionCriteria()
        
        # 过滤
        filtered = self._filter_trajectories(
            trajectories, explanations, criteria
        )
        
        if not filtered:
            # 如果没有满足条件的轨迹，返回评分最高的几个
            scored = list(zip(trajectories, explanations))
            scored.sort(key=lambda x: x[1].total_score, reverse=True)
            return scored[:num_selections]
        
        filtered_trajs, filtered_expls = zip(*filtered)
        filtered_trajs = list(filtered_trajs)
        filtered_expls = list(filtered_expls)
        
        # 计算损失
        losses = self._compute_losses(
            filtered_trajs, filtered_expls, current_state, previous_trajectory
        )
        
        # 选择损失最小的num_selections条轨迹
        sorted_indices = np.argsort(losses)
        
        results = []
        for idx in sorted_indices[:num_selections]:
            results.append((filtered_trajs[idx], filtered_expls[idx]))
        
        return results
    
    def _filter_trajectories(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation],
        criteria: SelectionCriteria
    ) -> List[Tuple[Trajectory, BehaviorExplanation]]:
        """
        根据选择标准过滤轨迹
        """
        filtered = []
        
        for traj, expl in zip(trajectories, explanations):
            # 检查各项评分
            if expl.safety_score < criteria.min_safety_score:
                continue
            if expl.comfort_score < criteria.min_comfort_score:
                continue
            if expl.efficiency_score < criteria.min_efficiency_score:
                continue
            if expl.legality_score < criteria.min_legality_score:
                continue
            
            # 检查偏好行为类型
            if criteria.prefer_behavior is not None:
                if expl.behavior_type != criteria.prefer_behavior:
                    continue
            
            # 检查偏好速度范围
            if criteria.prefer_speed_range is not None:
                avg_speed = np.mean(traj.v)
                min_speed, max_speed = criteria.prefer_speed_range
                if not (min_speed <= avg_speed <= max_speed):
                    continue
            
            filtered.append((traj, expl))
        
        return filtered
    
    def _compute_losses(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation],
        current_state: VehicleState,
        previous_trajectory: Optional[Trajectory]
    ) -> np.ndarray:
        """
        计算每条轨迹的损失
        
        损失越小越好
        """
        losses = np.zeros(len(trajectories))
        
        for i, (traj, expl) in enumerate(zip(trajectories, explanations)):
            # 1. 安全性损失 (越高越好，所以取反)
            safety_loss = 1.0 - expl.safety_score
            
            # 2. 舒适性损失
            comfort_loss = 1.0 - expl.comfort_score
            
            # 3. 效率损失
            efficiency_loss = 1.0 - expl.efficiency_score
            
            # 4. 平滑度损失
            smoothness_loss = self._compute_smoothness_loss(traj)
            
            # 5. 一致性损失 (与上一时刻轨迹的连续性)
            consistency_loss = self._compute_consistency_loss(
                traj, previous_trajectory
            )
            
            # 加权求和
            total_loss = (
                self.safety_weight * safety_loss +
                self.comfort_weight * comfort_loss +
                self.efficiency_weight * efficiency_loss +
                self.smoothness_weight * smoothness_loss +
                self.consistency_weight * consistency_loss
            )
            
            losses[i] = total_loss
        
        return losses
    
    def _compute_smoothness_loss(self, trajectory: Trajectory) -> float:
        """
        计算平滑度损失
        """
        # 加速度变化率
        if len(trajectory.a) > 1:
            acc_jerk = np.diff(trajectory.a) / np.diff(trajectory.t)
            jerk_loss = np.mean(acc_jerk**2)
        else:
            jerk_loss = 0.0
        
        # 曲率变化
        if len(trajectory.kappa) > 1:
            kappa_change = np.diff(trajectory.kappa)
            kappa_loss = np.mean(kappa_change**2)
        else:
            kappa_loss = 0.0
        
        # 速度变化
        if len(trajectory.v) > 1:
            vel_change = np.diff(trajectory.v)
            vel_loss = np.mean(vel_change**2)
        else:
            vel_loss = 0.0
        
        # 归一化并求和
        smoothness_loss = (jerk_loss / 10.0 + kappa_loss * 10.0 + vel_loss / 5.0) / 3.0
        
        return min(1.0, smoothness_loss)
    
    def _compute_consistency_loss(
        self,
        trajectory: Trajectory,
        previous_trajectory: Optional[Trajectory]
    ) -> float:
        """
        计算与上一时刻轨迹的一致性损失
        """
        if previous_trajectory is None:
            return 0.0
        
        # 计算起始点的一致性
        start_diff_x = trajectory.x[0] - previous_trajectory.x[0]
        start_diff_y = trajectory.y[0] - previous_trajectory.y[0]
        start_diff_theta = trajectory.theta[0] - previous_trajectory.theta[0]
        start_diff_v = trajectory.v[0] - previous_trajectory.v[0]
        
        # 计算轨迹形状的一致性 (使用部分重叠区域)
        overlap_length = min(len(trajectory.x), len(previous_trajectory.x))
        if overlap_length > 1:
            x_diff = trajectory.x[:overlap_length] - previous_trajectory.x[:overlap_length]
            y_diff = trajectory.y[:overlap_length] - previous_trajectory.y[:overlap_length]
            shape_loss = np.mean(x_diff**2 + y_diff**2)
        else:
            shape_loss = 0.0
        
        # 综合一致性损失
        consistency_loss = (
            abs(start_diff_x) / 10.0 +
            abs(start_diff_y) / 10.0 +
            abs(start_diff_theta) / np.pi +
            abs(start_diff_v) / 20.0 +
            shape_loss / 100.0
        ) / 5.0
        
        return min(1.0, consistency_loss)
    
    def update_preference_weights(
        self,
        selected_trajectory: Trajectory,
        selected_explanation: BehaviorExplanation,
        user_feedback: float
    ):
        """
        根据用户反馈更新偏好权重
        
        Args:
            user_feedback: 用户满意度 (0-1)
        """
        if not self.use_preference_model:
            return
        
        # 简单的在线学习更新权重
        learning_rate = 0.01
        
        # 根据用户反馈调整权重
        if user_feedback > 0.7:
            # 用户满意，增强当前特征
            self.preference_weights[0] += learning_rate * selected_explanation.safety_score
            self.preference_weights[1] += learning_rate * selected_explanation.comfort_score
            self.preference_weights[2] += learning_rate * selected_explanation.efficiency_score
        elif user_feedback < 0.3:
            # 用户不满意，降低当前特征
            self.preference_weights[0] -= learning_rate * (1 - selected_explanation.safety_score)
            self.preference_weights[1] -= learning_rate * (1 - selected_explanation.comfort_score)
            self.preference_weights[2] -= learning_rate * (1 - selected_explanation.efficiency_score)
        
        # 归一化权重
        self.preference_weights = np.clip(self.preference_weights, 0.01, 1.0)
        self.preference_weights /= np.sum(self.preference_weights)
        
        # 更新配置
        self.safety_weight = self.preference_weights[0]
        self.comfort_weight = self.preference_weights[1]
        self.efficiency_weight = self.preference_weights[2]
    
    def get_selection_statistics(
        self,
        trajectories: List[Trajectory],
        explanations: List[BehaviorExplanation]
    ) -> Dict:
        """
        获取选择统计信息
        """
        stats = {
            'total_trajectories': len(trajectories),
            'behavior_distribution': {},
            'score_statistics': {
                'safety': {
                    'mean': np.mean([e.safety_score for e in explanations]),
                    'std': np.std([e.safety_score for e in explanations]),
                    'min': np.min([e.safety_score for e in explanations]),
                    'max': np.max([e.safety_score for e in explanations])
                },
                'comfort': {
                    'mean': np.mean([e.comfort_score for e in explanations]),
                    'std': np.std([e.comfort_score for e in explanations]),
                    'min': np.min([e.comfort_score for e in explanations]),
                    'max': np.max([e.comfort_score for e in explanations])
                },
                'efficiency': {
                    'mean': np.mean([e.efficiency_score for e in explanations]),
                    'std': np.std([e.efficiency_score for e in explanations]),
                    'min': np.min([e.efficiency_score for e in explanations]),
                    'max': np.max([e.efficiency_score for e in explanations])
                },
                'total': {
                    'mean': np.mean([e.total_score for e in explanations]),
                    'std': np.std([e.total_score for e in explanations]),
                    'min': np.min([e.total_score for e in explanations]),
                    'max': np.max([e.total_score for e in explanations])
                }
            }
        }
        
        # 统计行为类型分布
        for expl in explanations:
            behavior = expl.behavior_type
            if behavior not in stats['behavior_distribution']:
                stats['behavior_distribution'][behavior] = 0
            stats['behavior_distribution'][behavior] += 1
        
        return stats
