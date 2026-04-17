"""
BEV (Bird's Eye View) 可视化模块

提供俯视视角的可视化功能，包括：
- 自车显示
- 车道线显示
- 障碍物显示（3D边界框投影）
- Occupancy网格显示
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .data_manager import EgoState, Obstacle, LaneLine, OccupancyGrid
from .font_utils import draw_chinese_texts


@dataclass
class BEVConfig:
    """BEV可视化配置"""
    # 画布大小
    width: int = 1200
    height: int = 800
    
    # 显示范围 (米)
    view_range_x: float = 100.0  # 前后范围
    view_range_y: float = 60.0   # 左右范围
    
    # 颜色配置 (BGR格式)
    background_color: Tuple[int, int, int] = (30, 30, 30)
    ego_color: Tuple[int, int, int] = (0, 255, 0)
    obstacle_colors: Dict[str, Tuple[int, int, int]] = None
    lane_color_solid: Tuple[int, int, int] = (255, 255, 255)
    lane_color_dashed: Tuple[int, int, int] = (0, 200, 255)
    lane_color_curb: Tuple[int, int, int] = (128, 128, 128)
    occupancy_color: Tuple[int, int, int] = (100, 100, 100)
    grid_color: Tuple[int, int, int] = (50, 50, 50)
    
    def __post_init__(self):
        if self.obstacle_colors is None:
            self.obstacle_colors = {
                'vehicle': (0, 100, 255),
                'pedestrian': (255, 0, 255),
                'cyclist': (255, 255, 0),
                'unknown': (128, 128, 128)
            }


class BEVVisualizer:
    """
    BEV可视化器
    
    将自动驾驶数据渲染为俯视视角图像
    """
    
    def __init__(self, config: Optional[BEVConfig] = None):
        self.config = config or BEVConfig()
        self.canvas: Optional[np.ndarray] = None
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        
        # 视图变换参数
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        
    def _init_canvas(self):
        """初始化画布"""
        self.canvas = np.ones((self.config.height, self.config.width, 3), 
                              dtype=np.uint8) * np.array(self.config.background_color, dtype=np.uint8)
        
        # 计算缩放比例
        self.scale_x = self.config.width / (self.config.view_range_x * 2)
        self.scale_y = self.config.height / (self.config.view_range_y * 2)
        
        # 偏移量 (自车位于画面下方1/4处)
        self.offset_x = self.config.width / 2
        self.offset_y = self.config.height * 0.75
    
    def world_to_screen(self, x: float, y: float, ego: EgoState) -> Tuple[int, int]:
        """
        将世界坐标转换为屏幕坐标
        
        Args:
            x, y: 世界坐标
            ego: 自车状态（用于相对坐标转换）
        
        Returns:
            屏幕坐标 (u, v)
        """
        # 转换到自车坐标系
        dx = x - ego.x
        dy = y - ego.y
        
        # 旋转到自车视角
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        local_x = dx * cos_h - dy * sin_h
        local_y = dx * sin_h + dy * cos_h
        
        # 应用缩放和平移
        screen_x = int(self.offset_x + local_x * self.scale_x * self.zoom + self.pan_x)
        screen_y = int(self.offset_y - local_y * self.scale_y * self.zoom + self.pan_y)
        
        return screen_x, screen_y
    
    def screen_to_world(self, u: int, v: int, ego: EgoState) -> Tuple[float, float]:
        """
        将屏幕坐标转换为世界坐标
        """
        # 逆变换
        local_x = (u - self.offset_x - self.pan_x) / (self.scale_x * self.zoom)
        local_y = (self.offset_y + self.pan_y - v) / (self.scale_y * self.zoom)
        
        # 旋转到世界坐标系
        cos_h = np.cos(ego.heading)
        sin_h = np.sin(ego.heading)
        
        world_x = local_x * cos_h - local_y * sin_h + ego.x
        world_y = local_x * sin_h + local_y * cos_h + ego.y
        
        return world_x, world_y
    
    def draw_grid(self, ego: EgoState, grid_size: float = 10.0):
        """绘制背景网格"""
        if self.canvas is None:
            return
        
        # 计算网格范围
        x_min = ego.x - self.config.view_range_x
        x_max = ego.x + self.config.view_range_x
        y_min = ego.y - self.config.view_range_y
        y_max = ego.y + self.config.view_range_y
        
        # 绘制垂直线
        x_start = np.floor(x_min / grid_size) * grid_size
        for x in np.arange(x_start, x_max, grid_size):
            p1 = self.world_to_screen(x, y_min, ego)
            p2 = self.world_to_screen(x, y_max, ego)
            cv2.line(self.canvas, p1, p2, self.config.grid_color, 1)
        
        # 绘制水平线
        y_start = np.floor(y_min / grid_size) * grid_size
        for y in np.arange(y_start, y_max, grid_size):
            p1 = self.world_to_screen(x_min, y, ego)
            p2 = self.world_to_screen(x_max, y, ego)
            cv2.line(self.canvas, p1, p2, self.config.grid_color, 1)
        
        # 绘制坐标轴
        origin = self.world_to_screen(ego.x, ego.y, ego)
        x_axis = self.world_to_screen(ego.x + 5, ego.y, ego)
        y_axis = self.world_to_screen(ego.x, ego.y + 5, ego)
        cv2.arrowedLine(self.canvas, origin, x_axis, (0, 0, 255), 2)
        cv2.arrowedLine(self.canvas, origin, y_axis, (0, 255, 0), 2)
    
    def draw_ego_vehicle(self, ego: EgoState, length: float = 4.5, width: float = 1.8):
        """绘制自车"""
        if self.canvas is None:
            return
        
        # 计算自车四个角点
        l, w = length / 2, width / 2
        corners = np.array([
            [l, w], [l, -w], [-l, -w], [-l, w]
        ])
        
        # 转换为屏幕坐标
        screen_corners = []
        for corner in corners:
            world_x = ego.x + corner[0] * np.cos(ego.heading) - corner[1] * np.sin(ego.heading)
            world_y = ego.y + corner[0] * np.sin(ego.heading) + corner[1] * np.cos(ego.heading)
            screen_corners.append(self.world_to_screen(world_x, world_y, ego))
        
        # 绘制车身
        pts = np.array(screen_corners, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.canvas, [pts], self.config.ego_color)
        cv2.polylines(self.canvas, [pts], True, (0, 200, 0), 2)
        
        # 绘制车头方向
        front_x = ego.x + l * 1.5 * np.cos(ego.heading)
        front_y = ego.y + l * 1.5 * np.sin(ego.heading)
        front_screen = self.world_to_screen(front_x, front_y, ego)
        center_screen = self.world_to_screen(ego.x, ego.y, ego)
        cv2.line(self.canvas, center_screen, front_screen, (0, 0, 255), 2)
        
        # 绘制速度向量
        if ego.velocity > 0.1:
            vel_x = ego.x + ego.velocity * 2 * np.cos(ego.heading)
            vel_y = ego.y + ego.velocity * 2 * np.sin(ego.heading)
            vel_screen = self.world_to_screen(vel_x, vel_y, ego)
            cv2.arrowedLine(self.canvas, center_screen, vel_screen, (255, 255, 0), 2)
    
    def draw_obstacles(self, obstacles: List[Obstacle], ego: EgoState):
        """绘制障碍物"""
        if self.canvas is None:
            return
        
        for obs in obstacles:
            color = self.config.obstacle_colors.get(obs.obstacle_type, (128, 128, 128))
            
            # 获取3D边界框角点
            corners = obs.get_corners()
            
            # 投影到BEV (只取底面的4个点)
            bev_corners = corners[4:, :2]  # 底面4个点的x, y
            
            # 转换为屏幕坐标
            screen_corners = []
            for corner in bev_corners:
                screen_corners.append(self.world_to_screen(corner[0], corner[1], ego))
            
            # 绘制边界框
            pts = np.array(screen_corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.canvas, [pts], True, color, 2)
            
            # 填充半透明
            overlay = self.canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, self.canvas, 0.7, 0, self.canvas)
            
            # 绘制中心点和ID
            center_screen = self.world_to_screen(obs.x, obs.y, ego)
            cv2.circle(self.canvas, center_screen, 3, color, -1)
            
            # 绘制ID和速度
            info_text = f"ID:{obs.id} {obs.velocity:.1f}m/s"
            cv2.putText(self.canvas, info_text, 
                       (center_screen[0] + 10, center_screen[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制速度向量
            if obs.velocity > 0.1:
                vel_x = obs.x + obs.velocity * 2 * np.cos(obs.heading)
                vel_y = obs.y + obs.velocity * 2 * np.sin(obs.heading)
                vel_screen = self.world_to_screen(vel_x, vel_y, ego)
                cv2.arrowedLine(self.canvas, center_screen, vel_screen, (255, 255, 255), 1)
    
    def draw_lane_lines(self, lane_lines: List[LaneLine], ego: EgoState):
        """绘制车道线"""
        if self.canvas is None:
            return
        
        for lane in lane_lines:
            if len(lane.points) < 2:
                continue
            
            # 选择颜色
            if lane.line_type == 'solid':
                color = self.config.lane_color_solid
            elif lane.line_type == 'dashed':
                color = self.config.lane_color_dashed
            elif lane.line_type == 'curb':
                color = self.config.lane_color_curb
            else:
                color = (200, 200, 200)
            
            # 转换为屏幕坐标
            screen_points = []
            for point in lane.points:
                screen_points.append(self.world_to_screen(point[0], point[1], ego))
            
            # 绘制线段
            if lane.line_type == 'dashed':
                # 虚线效果
                for i in range(0, len(screen_points) - 1, 2):
                    if i + 1 < len(screen_points):
                        cv2.line(self.canvas, screen_points[i], screen_points[i+1], color, 2)
            else:
                # 实线
                for i in range(len(screen_points) - 1):
                    cv2.line(self.canvas, screen_points[i], screen_points[i+1], color, 2)
    
    def draw_occupancy_grid(self, grid: OccupancyGrid, ego: EgoState, 
                           alpha: float = 0.3):
        """绘制Occupancy网格"""
        if self.canvas is None or grid is None:
            return
        
        # 创建网格覆盖层
        overlay = self.canvas.copy()
        
        h, w = grid.data.shape
        for i in range(h):
            for j in range(w):
                if grid.data[i, j] > 0.1:
                    # 网格坐标转世界坐标
                    world_x, world_y = grid.grid_to_world(j, i)
                    
                    # 世界坐标转屏幕坐标
                    screen_x, screen_y = self.world_to_screen(world_x, world_y, ego)
                    
                    # 计算网格大小
                    next_x, next_y = grid.grid_to_world(j + 1, i + 1)
                    next_screen_x, next_screen_y = self.world_to_screen(next_x, next_y, ego)
                    
                    grid_width = abs(next_screen_x - screen_x)
                    grid_height = abs(next_screen_y - screen_y)
                    
                    # 绘制网格单元
                    intensity = int(grid.data[i, j] * 255)
                    color = (intensity, intensity // 2, 0)
                    cv2.rectangle(overlay, 
                                (screen_x - grid_width//2, screen_y - grid_height//2),
                                (screen_x + grid_width//2, screen_y + grid_height//2),
                                color, -1)
        
        # 混合覆盖层
        cv2.addWeighted(overlay, alpha, self.canvas, 1 - alpha, 0, self.canvas)
    
    def draw_trajectory(self, points: np.ndarray, ego: EgoState, 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2, style: str = 'solid'):
        """
        绘制轨迹
        
        Args:
            points: Nx2或Nx3的点数组
            ego: 自车状态
            color: 颜色
            thickness: 线宽
            style: 'solid' 或 'dashed'
        """
        if self.canvas is None or len(points) < 2:
            return
        
        # 转换为屏幕坐标
        screen_points = []
        for point in points:
            screen_points.append(self.world_to_screen(point[0], point[1], ego))
        
        # 绘制
        if style == 'dashed':
            for i in range(0, len(screen_points) - 1, 2):
                if i + 1 < len(screen_points):
                    cv2.line(self.canvas, screen_points[i], screen_points[i+1], 
                            color, thickness)
        else:
            for i in range(len(screen_points) - 1):
                cv2.line(self.canvas, screen_points[i], screen_points[i+1], 
                        color, thickness)
    
    def draw_info_panel(self, ego: EgoState, info_text: List[str]):
        """绘制信息面板"""
        if self.canvas is None:
            return
        
        # 面板位置和大小
        panel_x = 10
        panel_y = 10
        line_height = 20
        padding = 10
        
        # 计算面板大小
        max_width = 300
        panel_height = len(info_text) * line_height + 2 * padding
        
        # 绘制背景
        cv2.rectangle(self.canvas, 
                     (panel_x, panel_y), 
                     (panel_x + max_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(self.canvas, 
                     (panel_x, panel_y), 
                     (panel_x + max_width, panel_y + panel_height),
                     (255, 255, 255), 1)
        
        # 绘制文字（使用PIL支持中文和特殊符号）
        font_size = 14
        baseline_y = panel_y + padding + line_height - 5
        text_top_y = baseline_y - int(font_size * 0.85)
        draw_chinese_texts(
            self.canvas, info_text,
            (panel_x + padding, text_top_y),
            (255, 255, 255),
            font_size, line_height
        )
    
    def render(self, ego: EgoState, 
               obstacles: Optional[List[Obstacle]] = None,
               lane_lines: Optional[List[LaneLine]] = None,
               occupancy_grid: Optional[OccupancyGrid] = None,
               show_grid: bool = True,
               show_info: bool = True) -> np.ndarray:
        """
        渲染BEV视图
        
        Args:
            ego: 自车状态
            obstacles: 障碍物列表
            lane_lines: 车道线列表
            occupancy_grid: Occupancy网格
            show_grid: 是否显示背景网格
            show_info: 是否显示信息面板
        
        Returns:
            渲染后的图像
        """
        self._init_canvas()
        
        # 绘制背景网格
        if show_grid:
            self.draw_grid(ego)
        
        # 绘制Occupancy网格
        if occupancy_grid is not None:
            self.draw_occupancy_grid(occupancy_grid, ego)
        
        # 绘制车道线
        if lane_lines is not None:
            self.draw_lane_lines(lane_lines, ego)
        
        # 绘制障碍物
        if obstacles is not None:
            self.draw_obstacles(obstacles, ego)
        
        # 绘制自车
        self.draw_ego_vehicle(ego)
        
        # 绘制信息面板
        if show_info:
            info_text = [
                f"Time: {ego.timestamp:.2f}s",
                f"Position: ({ego.x:.2f}, {ego.y:.2f})",
                f"Heading: {np.degrees(ego.heading):.2f}°",
                f"Velocity: {ego.velocity:.2f} m/s",
                f"Acceleration: {ego.acceleration:.2f} m/s²",
            ]
            self.draw_info_panel(ego, info_text)
        
        return self.canvas.copy()
    
    def set_zoom(self, zoom: float):
        """设置缩放级别"""
        self.zoom = max(0.1, min(5.0, zoom))
    
    def set_pan(self, pan_x: float, pan_y: float):
        """设置平移偏移"""
        self.pan_x = pan_x
        self.pan_y = pan_y
    
    def reset_view(self):
        """重置视图"""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
