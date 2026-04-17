"""
Planning Visualization Module
规划可视化模块

用于可视化轨迹规划结果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from typing import List, Optional

from lattice_generator import Trajectory, VehicleState, Obstacle, LaneInfo
from behavior_explainer import BehaviorExplanation
from planning_module import PlanningModule, PlanningInput, PlanningOutput


def visualize_trajectory(
    trajectory: Trajectory,
    obstacles: List[Obstacle],
    lane_info: LaneInfo,
    explanation: Optional[BehaviorExplanation] = None,
    title: str = "Trajectory Planning",
    save_path: Optional[str] = None
):
    """
    可视化单条轨迹
    
    Args:
        trajectory: 轨迹
        obstacles: 障碍物列表
        lane_info: 车道信息
        explanation: 行为解释
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 轨迹俯视图
    ax1 = axes[0, 0]
    
    # 绘制车道
    lane_x = lane_info.x
    lane_y = lane_info.y
    lane_width = lane_info.width
    
    # 车道边界
    ax1.plot(lane_x, lane_y - lane_width/2, 'k--', linewidth=1, alpha=0.5, label='Lane boundary')
    ax1.plot(lane_x, lane_y + lane_width/2, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(lane_x, lane_y, 'k-', linewidth=1, alpha=0.3, label='Lane center')
    
    # 绘制障碍物
    for obs in obstacles:
        rect = Rectangle(
            (obs.x - obs.length/2, obs.y - obs.width/2),
            obs.length, obs.width,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
        )
        ax1.add_patch(rect)
        # 绘制速度向量
        ax1.arrow(obs.x, obs.y, obs.vx*2, obs.vy*2, 
                 head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.7)
    
    # 绘制轨迹
    points = np.array([trajectory.x, trajectory.y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 使用颜色映射表示速度
    norm = plt.Normalize(trajectory.v.min(), trajectory.v.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(trajectory.v)
    lc.set_linewidth(3)
    ax1.add_collection(lc)
    
    # 起点和终点
    ax1.scatter(trajectory.x[0], trajectory.y[0], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax1.scatter(trajectory.x[-1], trajectory.y[-1], c='red', s=100, marker='s', 
               label='End', zorder=5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 添加颜色条
    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label('Velocity (m/s)')
    
    # 2. 速度曲线
    ax2 = axes[0, 1]
    ax2.plot(trajectory.t, trajectory.v, 'b-', linewidth=2, label='Velocity')
    ax2.fill_between(trajectory.t, trajectory.v, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 加速度曲线
    ax3 = axes[1, 0]
    ax3.plot(trajectory.t, trajectory.a, 'r-', linewidth=2, label='Acceleration')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.fill_between(trajectory.t, trajectory.a, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Acceleration Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 曲率曲线
    ax4 = axes[1, 1]
    ax4.plot(trajectory.t, trajectory.kappa, 'g-', linewidth=2, label='Curvature')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.fill_between(trajectory.t, trajectory.kappa, alpha=0.3, color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Curvature (1/m)')
    ax4.set_title('Curvature Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 添加评分信息
    if explanation:
        info_text = (
            f"Behavior: {explanation.behavior_type}\n"
            f"Safety: {explanation.safety_score:.3f}\n"
            f"Comfort: {explanation.comfort_score:.3f}\n"
            f"Efficiency: {explanation.efficiency_score:.3f}\n"
            f"Legality: {explanation.legality_score:.3f}\n"
            f"Total: {explanation.total_score:.3f}"
        )
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_multiple_trajectories(
    trajectories: List[Trajectory],
    obstacles: List[Obstacle],
    lane_info: LaneInfo,
    selected_idx: int = 0,
    title: str = "Multiple Trajectories",
    save_path: Optional[str] = None
):
    """
    可视化多条候选轨迹
    
    Args:
        trajectories: 轨迹列表
        obstacles: 障碍物列表
        lane_info: 车道信息
        selected_idx: 选中的轨迹索引
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 绘制车道
    lane_x = lane_info.x
    lane_y = lane_info.y
    lane_width = lane_info.width
    
    ax.plot(lane_x, lane_y - lane_width/2, 'k--', linewidth=1, alpha=0.5)
    ax.plot(lane_x, lane_y + lane_width/2, 'k--', linewidth=1, alpha=0.5)
    ax.plot(lane_x, lane_y, 'k-', linewidth=1, alpha=0.3)
    
    # 绘制障碍物
    for obs in obstacles:
        rect = Rectangle(
            (obs.x - obs.length/2, obs.y - obs.width/2),
            obs.length, obs.width,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
        )
        ax.add_patch(rect)
    
    # 绘制所有轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        if i == selected_idx:
            # 选中的轨迹用粗线
            ax.plot(traj.x, traj.y, '-', linewidth=3, color='green', 
                   label=f'Selected (cost={traj.cost:.3f})', zorder=10)
            ax.scatter(traj.x[0], traj.y[0], c='green', s=100, marker='o', zorder=10)
            ax.scatter(traj.x[-1], traj.y[-1], c='green', s=100, marker='s', zorder=10)
        else:
            ax.plot(traj.x, traj.y, '--', linewidth=1.5, color=colors[i], 
                   alpha=0.6, label=f'Candidate {i+1} (cost={traj.cost:.3f})')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_planning_output(
    output: PlanningOutput,
    obstacles: List[Obstacle],
    lane_info: LaneInfo,
    save_path: Optional[str] = None
):
    """
    可视化规划输出
    
    Args:
        output: 规划输出
        obstacles: 障碍物列表
        lane_info: 车道信息
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Planning Output Visualization', fontsize=16, fontweight='bold')
    
    trajectory = output.trajectory
    explanation = output.explanation
    
    # 1. 主轨迹视图
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # 车道
    lane_x = lane_info.x
    lane_y = lane_info.y
    lane_width = lane_info.width
    
    ax_main.plot(lane_x, lane_y - lane_width/2, 'k--', linewidth=1, alpha=0.5)
    ax_main.plot(lane_x, lane_y + lane_width/2, 'k--', linewidth=1, alpha=0.5)
    ax_main.plot(lane_x, lane_y, 'k-', linewidth=1, alpha=0.3)
    
    # 障碍物
    for obs in obstacles:
        rect = Rectangle(
            (obs.x - obs.length/2, obs.y - obs.width/2),
            obs.length, obs.width,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
        )
        ax_main.add_patch(rect)
    
    # 主轨迹
    points = np.array([trajectory.x, trajectory.y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(trajectory.v.min(), trajectory.v.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(trajectory.v)
    lc.set_linewidth(4)
    ax_main.add_collection(lc)
    
    # 备选轨迹
    for alt_traj in output.alternative_trajectories:
        ax_main.plot(alt_traj.x, alt_traj.y, '--', linewidth=1.5, 
                    alpha=0.4, color='gray')
    
    ax_main.scatter(trajectory.x[0], trajectory.y[0], c='green', s=150, 
                   marker='o', label='Start', zorder=5, edgecolors='black')
    ax_main.scatter(trajectory.x[-1], trajectory.y[-1], c='red', s=150, 
                   marker='s', label='End', zorder=5, edgecolors='black')
    
    ax_main.set_xlabel('X (m)')
    ax_main.set_ylabel('Y (m)')
    ax_main.set_title('Trajectory with Alternatives')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    ax_main.axis('equal')
    
    cbar = plt.colorbar(lc, ax=ax_main)
    cbar.set_label('Velocity (m/s)')
    
    # 2. 速度曲线
    ax_v = fig.add_subplot(gs[0, 2])
    ax_v.plot(trajectory.t, trajectory.v, 'b-', linewidth=2)
    ax_v.fill_between(trajectory.t, trajectory.v, alpha=0.3)
    ax_v.set_xlabel('Time (s)')
    ax_v.set_ylabel('Velocity (m/s)')
    ax_v.set_title('Velocity')
    ax_v.grid(True, alpha=0.3)
    
    # 3. 加速度曲线
    ax_a = fig.add_subplot(gs[1, 2])
    ax_a.plot(trajectory.t, trajectory.a, 'r-', linewidth=2)
    ax_a.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_a.fill_between(trajectory.t, trajectory.a, alpha=0.3, color='red')
    ax_a.set_xlabel('Time (s)')
    ax_a.set_ylabel('Acceleration (m/s²)')
    ax_a.set_title('Acceleration')
    ax_a.grid(True, alpha=0.3)
    
    # 4. 曲率曲线
    ax_k = fig.add_subplot(gs[2, 0])
    ax_k.plot(trajectory.t, trajectory.kappa, 'g-', linewidth=2)
    ax_k.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_k.fill_between(trajectory.t, trajectory.kappa, alpha=0.3, color='green')
    ax_k.set_xlabel('Time (s)')
    ax_k.set_ylabel('Curvature (1/m)')
    ax_k.set_title('Curvature')
    ax_k.grid(True, alpha=0.3)
    
    # 5. 航向角曲线
    ax_theta = fig.add_subplot(gs[2, 1])
    ax_theta.plot(trajectory.t, np.degrees(trajectory.theta), 'm-', linewidth=2)
    ax_theta.set_xlabel('Time (s)')
    ax_theta.set_ylabel('Heading (deg)')
    ax_theta.set_title('Heading Angle')
    ax_theta.grid(True, alpha=0.3)
    
    # 6. 评分信息
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis('off')
    
    info_text = f"""
    Planning Information
    ====================
    
    Behavior Type: {explanation.behavior_type}
    
    Scores:
    --------
    Safety:     {explanation.safety_score:.3f}
    Comfort:    {explanation.comfort_score:.3f}
    Efficiency: {explanation.efficiency_score:.3f}
    Legality:   {explanation.legality_score:.3f}
    Total:      {explanation.total_score:.3f}
    
    Planning Metrics:
    ----------------
    Planning Time: {output.planning_time*1000:.1f} ms
    Candidates: {output.num_candidates}
    Trajectory Length: {trajectory.get_length():.1f} m
    Avg Speed: {np.mean(trajectory.v):.1f} m/s
    
    Risk Factors:
    -------------
    """
    for risk in explanation.risk_factors[:3]:
        info_text += f"  • {risk}\n"
    
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def create_demo_visualization():
    """
    创建演示可视化
    """
    print("Creating demo visualization...")
    
    # 创建规划模块
    planner = PlanningModule()
    
    # 创建测试场景
    ego_state = VehicleState(
        x=0.0, y=0.0, theta=0.0, v=10.0, a=0.0
    )
    
    # 障碍物
    objects = [
        {'x': 25.0, 'y': 0.0, 'vx': 8.0, 'vy': 0.0, 
         'category': 'vehicle', 'width': 2.0, 'length': 4.5},
        {'x': 45.0, 'y': 3.5, 'vx': 10.0, 'vy': 0.0, 
         'category': 'vehicle', 'width': 2.0, 'length': 4.5},
    ]
    
    # 车道
    lane_x = np.linspace(0, 100, 100)
    lane_y = np.zeros(100)
    lanes = [{'x': lane_x.tolist(), 'y': lane_y.tolist(), 'width': 3.5}]
    
    # 规划输入
    planning_input = PlanningInput(
        ego_state=ego_state,
        objects=objects,
        occupancy=[],
        lanes=lanes,
        traffic_signs=[],
        timestamp=0.0
    )
    
    # 执行规划
    output = planner.plan(planning_input)
    
    # 预处理障碍物
    obstacles = []
    for obj in objects:
        obstacles.append(Obstacle(
            x=obj['x'], y=obj['y'],
            vx=obj['vx'], vy=obj['vy'],
            category=obj['category'],
            width=obj['width'],
            length=obj['length']
        ))
    
    lane_info = LaneInfo(x=lane_x, y=lane_y, width=3.5)
    
    # 可视化
    visualize_planning_output(
        output, obstacles, lane_info,
        save_path='/mnt/okcomputer/output/autonomous_driving/src/planning/demo_output.png'
    )
    
    print("Demo visualization completed!")


if __name__ == "__main__":
    create_demo_visualization()
