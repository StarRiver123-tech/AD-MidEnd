"""
轨迹可视化模块

提供规划轨迹的可视化功能，包括：
- 候选轨迹显示
- 选中轨迹高亮
- 轨迹成本可视化
- 行为决策显示
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .data_manager import Trajectory, PlanningResult, EgoState, BehaviorType
from .font_utils import draw_chinese_texts


@dataclass
class TrajectoryConfig:
    """轨迹可视化配置"""
    # 候选轨迹颜色
    candidate_color: Tuple[int, int, int] = (128, 128, 128)
    candidate_thickness: int = 1
    
    # 选中轨迹颜色
    selected_color: Tuple[int, int, int] = (0, 255, 0)
    selected_thickness: int = 3
    
    # 参考轨迹颜色
    reference_color: Tuple[int, int, int] = (255, 255, 0)
    reference_thickness: int = 2
    
    # 轨迹点标记
    show_points: bool = True
    point_radius: int = 3
    point_interval: int = 5  # 每隔几个点画一个点
    
    # 速度颜色映射
    velocity_colormap: bool = True
    velocity_min: float = 0.0
    velocity_max: float = 15.0
    
    # 行为决策显示
    behavior_font_scale: float = 0.7
    behavior_color: Tuple[int, int, int] = (255, 255, 255)
    behavior_bg_color: Tuple[int, int, int] = (0, 100, 200)


class TrajectoryVisualizer:
    """
    轨迹可视化器
    
    在BEV视图上渲染规划轨迹
    """
    
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()
    
    def velocity_to_color(self, velocity: float) -> Tuple[int, int, int]:
        """
        将速度映射到颜色
        
        低速: 红色 -> 高速: 绿色
        """
        # 归一化
        t = (velocity - self.config.velocity_min) / \
            (self.config.velocity_max - self.config.velocity_min)
        t = max(0.0, min(1.0, t))
        
        # 红到绿的渐变
        r = int((1 - t) * 255)
        g = int(t * 255)
        b = 0
        
        return (b, g, r)  # BGR格式
    
    def draw_trajectory_on_bev(self, canvas: np.ndarray, 
                                trajectory: Trajectory,
                                ego: EgoState,
                                world_to_screen_func,
                                is_selected: bool = False):
        """
        在BEV画布上绘制单条轨迹
        
        Args:
            canvas: BEV画布
            trajectory: 轨迹数据
            ego: 自车状态
            world_to_screen_func: 世界坐标到屏幕坐标的转换函数
            is_selected: 是否为选中轨迹
        """
        if len(trajectory.points) < 2:
            return
        
        # 确定颜色和线宽
        if is_selected:
            color = self.config.selected_color
            thickness = self.config.selected_thickness
        elif trajectory.trajectory_type == 'reference':
            color = self.config.reference_color
            thickness = self.config.reference_thickness
        else:
            color = trajectory.color if hasattr(trajectory, 'color') else self.config.candidate_color
            # 转换为BGR
            if isinstance(color, tuple) and max(color) <= 1.0:
                color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            thickness = self.config.candidate_thickness
        
        # 转换为屏幕坐标
        screen_points = []
        for point in trajectory.points:
            screen_points.append(world_to_screen_func(point[0], point[1]))
        
        # 绘制轨迹线
        if self.config.velocity_colormap and trajectory.velocities is not None:
            # 使用速度颜色映射
            for i in range(len(screen_points) - 1):
                vel = trajectory.velocities[i] if i < len(trajectory.velocities) else 0
                segment_color = self.velocity_to_color(vel)
                cv2.line(canvas, screen_points[i], screen_points[i+1], 
                        segment_color, thickness)
        else:
            # 使用统一颜色
            for i in range(len(screen_points) - 1):
                cv2.line(canvas, screen_points[i], screen_points[i+1], 
                        color, thickness)
        
        # 绘制轨迹点
        if self.config.show_points:
            for i in range(0, len(screen_points), self.config.point_interval):
                if self.config.velocity_colormap and trajectory.velocities is not None:
                    vel = trajectory.velocities[i] if i < len(trajectory.velocities) else 0
                    point_color = self.velocity_to_color(vel)
                else:
                    point_color = color
                
                cv2.circle(canvas, screen_points[i], self.config.point_radius, 
                          point_color, -1)
        
        # 如果是选中轨迹，绘制终点标记
        if is_selected and len(screen_points) > 0:
            end_point = screen_points[-1]
            cv2.circle(canvas, end_point, 8, (0, 255, 255), 2)
            cv2.circle(canvas, end_point, 4, (0, 255, 255), -1)
    
    def draw_planning_result(self, canvas: np.ndarray,
                            planning_result: PlanningResult,
                            ego: EgoState,
                            world_to_screen_func):
        """
        绘制完整的规划结果
        
        Args:
            canvas: BEV画布
            planning_result: 规划结果
            ego: 自车状态
            world_to_screen_func: 世界坐标到屏幕坐标的转换函数
        """
        # 绘制候选轨迹
        for traj in planning_result.candidate_trajectories:
            if traj is not planning_result.selected_trajectory:
                self.draw_trajectory_on_bev(canvas, traj, ego, 
                                           world_to_screen_func, False)
        
        # 绘制选中轨迹（最后绘制，显示在最上层）
        if planning_result.selected_trajectory is not None:
            self.draw_trajectory_on_bev(canvas, planning_result.selected_trajectory,
                                       ego, world_to_screen_func, True)
    
    def draw_behavior_info(self, canvas: np.ndarray, 
                          planning_result: PlanningResult,
                          position: Tuple[int, int] = (10, 150)):
        """
        绘制行为决策信息
        
        Args:
            canvas: 画布
            planning_result: 规划结果
            position: 显示位置
        """
        if planning_result is None:
            return
        
        x, y = position
        line_height = 25
        padding = 10
        
        # 准备文本
        texts = [
            f"行为决策: {planning_result.behavior.value}",
            f"决策描述: {planning_result.behavior_description}",
            f"目标速度: {planning_result.target_speed:.2f} m/s",
            f"目标车道: {planning_result.target_lane}",
        ]
        
        # 如果有选中轨迹，添加轨迹信息
        if planning_result.selected_trajectory is not None:
            traj = planning_result.selected_trajectory
            texts.append(f"轨迹点数: {len(traj.points)}")
            if traj.cost > 0:
                texts.append(f"轨迹成本: {traj.cost:.3f}")
        
        # 计算面板大小
        max_text_width = max([len(t) for t in texts]) * 10
        panel_width = max_text_width + 2 * padding
        panel_height = len(texts) * line_height + 2 * padding
        
        # 绘制背景
        cv2.rectangle(canvas,
                     (x, y),
                     (x + panel_width, y + panel_height),
                     self.config.behavior_bg_color, -1)
        cv2.rectangle(canvas,
                     (x, y),
                     (x + panel_width, y + panel_height),
                     (255, 255, 255), 1)
        
        # 绘制文本（使用PIL支持中文）
        font_size = max(12, int(self.config.behavior_font_scale * 26))
        baseline_y = y + padding + line_height - 5
        text_top_y = baseline_y - int(font_size * 0.85)
        draw_chinese_texts(
            canvas, texts,
            (x + padding, text_top_y),
            self.config.behavior_color,
            font_size, line_height
        )
    
    def draw_trajectory_comparison(self, canvas: np.ndarray,
                                    trajectories: List[Trajectory],
                                    ego: EgoState,
                                    world_to_screen_func,
                                    selected_idx: int = -1):
        """
        绘制多条轨迹对比
        
        用于显示候选轨迹的对比视图
        """
        # 颜色列表
        colors = [
            (255, 0, 0),    # 蓝
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 红
            (255, 255, 0),  # 青
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
        ]
        
        for idx, traj in enumerate(trajectories):
            if len(traj.points) < 2:
                continue
            
            # 选择颜色
            if idx == selected_idx:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = colors[idx % len(colors)]
                thickness = 1
            
            # 转换为屏幕坐标
            screen_points = []
            for point in traj.points:
                screen_points.append(world_to_screen_func(point[0], point[1]))
            
            # 绘制轨迹
            for i in range(len(screen_points) - 1):
                cv2.line(canvas, screen_points[i], screen_points[i+1],
                        color, thickness)
            
            # 绘制起点标记
            if len(screen_points) > 0:
                label = f"#{idx}"
                if traj.cost > 0:
                    label += f" c={traj.cost:.2f}"
                cv2.putText(canvas, label, 
                           (screen_points[0][0] + 5, screen_points[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def create_trajectory_legend(self, width: int = 200, 
                                  height: int = 150) -> np.ndarray:
        """
        创建轨迹图例
        
        Returns:
            图例图像
        """
        legend = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        items = [
            ("选中轨迹", self.config.selected_color, 3),
            ("候选轨迹", self.config.candidate_color, 1),
            ("参考轨迹", self.config.reference_color, 2),
        ]
        
        y = 30
        for name, color, thickness in items:
            # 绘制示例线
            cv2.line(legend, (10, y), (50, y), color, thickness)
            # 绘制文字
            draw_chinese_texts(legend, [name], (60, y - 10),
                              (255, 255, 255), 14, 20)
            y += 35
        
        # 添加速度颜色条
        draw_chinese_texts(legend, ["速度:"], (10, y - 10),
                          (255, 255, 255), 14, 20)
        y += 20
        
        # 绘制渐变色条
        for i in range(width - 20):
            vel = self.config.velocity_min + (self.config.velocity_max - self.config.velocity_min) * i / (width - 20)
            color = self.velocity_to_color(vel)
            cv2.line(legend, (10 + i, y), (10 + i, y + 15), color, 1)
        
        # 标注
        cv2.putText(legend, f"{self.config.velocity_min:.0f}", (5, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(legend, f"{self.config.velocity_max:.0f}", (width - 35, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return legend


def create_trajectory_figure(trajectories: List[Trajectory],
                             selected_idx: int = -1,
                             figsize: Tuple[int, int] = (10, 8)) -> np.ndarray:
    """
    使用matplotlib创建轨迹对比图
    
    Args:
        trajectories: 轨迹列表
        selected_idx: 选中轨迹的索引
        figsize: 图像大小
    
    Returns:
        渲染后的图像
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Trajectory Analysis', fontsize=14)
    
    # XY轨迹图
    ax_xy = axes[0, 0]
    for idx, traj in enumerate(trajectories):
        if len(traj.points) < 2:
            continue
        
        x = traj.points[:, 0]
        y = traj.points[:, 1]
        
        if idx == selected_idx:
            ax_xy.plot(x, y, 'g-', linewidth=3, label=f'Selected #{idx}')
        else:
            ax_xy.plot(x, y, '--', alpha=0.5, label=f'Candidate #{idx}')
    
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Trajectory XY')
    ax_xy.legend()
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    # 速度曲线
    ax_vel = axes[0, 1]
    for idx, traj in enumerate(trajectories):
        if traj.velocities is None or len(traj.velocities) < 2:
            continue
        
        s = np.arange(len(traj.velocities))
        
        if idx == selected_idx:
            ax_vel.plot(s, traj.velocities, 'g-', linewidth=2, label=f'Selected #{idx}')
        else:
            ax_vel.plot(s, traj.velocities, '--', alpha=0.5, label=f'Candidate #{idx}')
    
    ax_vel.set_xlabel('Point Index')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Velocity Profile')
    ax_vel.legend()
    ax_vel.grid(True)
    
    # 曲率
    ax_curv = axes[1, 0]
    for idx, traj in enumerate(trajectories):
        if len(traj.points) < 3:
            continue
        
        # 计算曲率
        x = traj.points[:, 0]
        y = traj.points[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)
        
        s = np.arange(len(curvature))
        
        if idx == selected_idx:
            ax_curv.plot(s, curvature, 'g-', linewidth=2, label=f'Selected #{idx}')
        else:
            ax_curv.plot(s, curvature, '--', alpha=0.5, label=f'Candidate #{idx}')
    
    ax_curv.set_xlabel('Point Index')
    ax_curv.set_ylabel('Curvature (1/m)')
    ax_curv.set_title('Curvature Profile')
    ax_curv.legend()
    ax_curv.grid(True)
    
    # 成本对比
    ax_cost = axes[1, 1]
    costs = [t.cost for t in trajectories]
    indices = list(range(len(trajectories)))
    colors = ['green' if i == selected_idx else 'gray' for i in indices]
    ax_cost.bar(indices, costs, color=colors)
    ax_cost.set_xlabel('Trajectory Index')
    ax_cost.set_ylabel('Cost')
    ax_cost.set_title('Trajectory Cost Comparison')
    
    plt.tight_layout()
    
    # 转换为numpy数组
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    plt.close(fig)
    
    # 转换为BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img
