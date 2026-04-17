"""
传感器数据可视化模块

提供传感器数据的可视化功能，包括：
- 相机图像显示（多视角）
- LiDAR点云显示（3D和BEV）
- 雷达数据显示
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .data_manager import SensorData, Obstacle, EgoState


@dataclass
class SensorConfig:
    """传感器可视化配置"""
    # 点云显示
    point_size: int = 2
    point_color_mode: str = 'height'  # 'height', 'intensity', 'distance', 'class'
    
    # 高度颜色映射范围
    height_min: float = -2.0
    height_max: float = 2.0
    
    # 强度颜色映射范围
    intensity_min: float = 0.0
    intensity_max: float = 1.0
    
    # 距离颜色映射范围
    distance_max: float = 100.0
    
    # BEV点云
    bev_resolution: float = 0.1  # 米/像素
    bev_size: int = 800  # 像素
    bev_range: float = 50.0  # 显示范围（米）
    
    # 3D点云投影
    fov_horizontal: float = 120.0  # 水平视场角
    fov_vertical: float = 30.0     # 垂直视场角


class SensorVisualizer:
    """
    传感器数据可视化器
    """
    
    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or SensorConfig()
    
    def height_to_color(self, height: float) -> Tuple[int, int, int]:
        """
        将高度映射到颜色
        
        低: 蓝色 -> 高: 红色
        """
        t = (height - self.config.height_min) / \
            (self.config.height_max - self.config.height_min)
        t = max(0.0, min(1.0, t))
        
        # 使用HSV色彩空间
        hue = int((1 - t) * 120)  # 120(绿) -> 0(红)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    
    def intensity_to_color(self, intensity: float) -> Tuple[int, int, int]:
        """将强度映射到颜色"""
        t = (intensity - self.config.intensity_min) / \
            (self.config.intensity_max - self.config.intensity_min)
        t = max(0.0, min(1.0, t))
        gray = int(t * 255)
        return (gray, gray, gray)
    
    def distance_to_color(self, distance: float) -> Tuple[int, int, int]:
        """将距离映射到颜色"""
        t = distance / self.config.distance_max
        t = max(0.0, min(1.0, t))
        
        # 近: 红色 -> 远: 蓝色
        r = int((1 - t) * 255)
        b = int(t * 255)
        return (b, 0, r)
    
    def render_lidar_bev(self, lidar_points: np.ndarray, 
                         ego: EgoState,
                         obstacles: Optional[List[Obstacle]] = None,
                         canvas_size: Tuple[int, int] = (600, 600)) -> np.ndarray:
        """
        渲染LiDAR点云的BEV视图
        
        Args:
            lidar_points: Nx4数组 (x, y, z, intensity)
            ego: 自车状态
            obstacles: 障碍物列表（用于着色）
            canvas_size: 画布大小
        
        Returns:
            BEV图像
        """
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        if lidar_points is None or len(lidar_points) == 0:
            return canvas
        
        # 计算缩放
        scale = canvas_size[0] / (self.config.bev_range * 2)
        center_x = canvas_size[0] // 2
        center_y = canvas_size[1] // 2
        
        # 绘制网格
        grid_spacing = 10  # 米
        grid_pixels = int(grid_spacing * scale)
        for i in range(0, canvas_size[0], grid_pixels):
            cv2.line(canvas, (i, 0), (i, canvas_size[1]), (30, 30, 30), 1)
        for i in range(0, canvas_size[1], grid_pixels):
            cv2.line(canvas, (0, i), (canvas_size[0], i), (30, 30, 30), 1)
        
        # 转换点云到自车坐标系
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        local_points = []
        for point in lidar_points:
            dx = point[0] - ego.x
            dy = point[1] - ego.y
            
            local_x = dx * cos_h - dy * sin_h
            local_y = dx * sin_h + dy * cos_h
            
            # 过滤范围外的点
            if abs(local_x) < self.config.bev_range and abs(local_y) < self.config.bev_range:
                local_points.append([local_x, local_y, point[2], point[3]])
        
        # 绘制点云
        for point in local_points:
            x, y, z, intensity = point
            
            # 计算屏幕坐标
            screen_x = int(center_x + x * scale)
            screen_y = int(center_y - y * scale)
            
            # 确定颜色
            if self.config.point_color_mode == 'height':
                color = self.height_to_color(z)
            elif self.config.point_color_mode == 'intensity':
                color = self.intensity_to_color(intensity)
            elif self.config.point_color_mode == 'distance':
                dist = np.sqrt(x**2 + y**2)
                color = self.distance_to_color(dist)
            else:
                color = (255, 255, 255)
            
            # 绘制点
            cv2.circle(canvas, (screen_x, screen_y), 
                      self.config.point_size, color, -1)
        
        # 绘制自车位置
        cv2.circle(canvas, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.line(canvas, (center_x, center_y), 
                (center_x, center_y - 20), (0, 255, 0), 2)
        
        # 添加图例
        legend_y = 20
        cv2.putText(canvas, "LiDAR BEV", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(canvas, f"Points: {len(lidar_points)}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return canvas
    
    def render_lidar_3d_projection(self, lidar_points: np.ndarray,
                                    image_size: Tuple[int, int] = (800, 600),
                                    elevation_angle: float = 25.0,
                                    azimuth_angle: float = 45.0) -> np.ndarray:
        """
        渲染LiDAR点云的3D投影视图（等轴测正交投影）
        
        Args:
            lidar_points: Nx4数组
            image_size: 图像大小
            elevation_angle: 仰角（度）
            azimuth_angle: 方位角（度）
        
        Returns:
            3D投影图像
        """
        canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        if lidar_points is None or len(lidar_points) == 0:
            return canvas
        
        # 3D旋转参数
        elev_rad = np.radians(elevation_angle)
        azim_rad = np.radians(azimuth_angle)
        
        cos_e, sin_e = np.cos(elev_rad), np.sin(elev_rad)
        cos_a, sin_a = np.cos(azim_rad), np.sin(azim_rad)
        
        # 先计算所有旋转后的点以确定缩放比例
        rotated = []
        for point in lidar_points:
            x, y, z = point[0], point[1], point[2]
            intensity = point[3] if len(point) > 3 else 0.0
            
            # 绕Z轴旋转（方位角）
            x1 = x * cos_a - y * sin_a
            y1 = x * sin_a + y * cos_a
            z1 = z
            
            # 绕X轴旋转（仰角）
            x2 = x1
            y2 = y1 * cos_e - z1 * sin_e
            z2 = y1 * sin_e + z1 * cos_e
            
            rotated.append((x2, y2, z2, z, intensity))
        
        if not rotated:
            return canvas
        
        xs = [r[0] for r in rotated]
        ys = [r[1] for r in rotated]
        
        # 计算合适的缩放比例，留 10% 边距
        margin = 0.9
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
        
        scale = min(image_size[0] / x_range, image_size[1] / y_range) * margin
        scale = max(0.5, scale)
        
        center_x = image_size[0] / 2
        center_y = image_size[1] / 2
        offset_x = center_x - (sum(xs) / len(xs)) * scale
        offset_y = center_y - (sum(ys) / len(ys)) * scale
        
        # 按深度（z2）从远到近排序，实现基本的深度遮挡
        rotated.sort(key=lambda r: r[2], reverse=True)
        
        # 绘制点
        for x2, y2, z2, z_orig, intensity in rotated:
            screen_x = int(offset_x + x2 * scale)
            screen_y = int(offset_y - y2 * scale)
            
            if 0 <= screen_x < image_size[0] and 0 <= screen_y < image_size[1]:
                if self.config.point_color_mode == 'height':
                    color = self.height_to_color(z_orig)
                elif self.config.point_color_mode == 'intensity':
                    color = self.intensity_to_color(intensity)
                else:
                    dist = np.sqrt(x2**2 + y2**2)
                    color = self.distance_to_color(dist)
                
                cv2.circle(canvas, (screen_x, screen_y), max(1, self.config.point_size), color, -1)
        
        # 添加坐标轴
        axis_length = min(80, int(min(image_size) * 0.12))
        origin = (int(center_x), int(center_y))
        # X轴 (红色)
        cv2.line(canvas, origin, (origin[0] + int(axis_length * cos_a), origin[1] - int(axis_length * sin_a)), (0, 0, 255), 2)
        # Y轴 (绿色)
        cv2.line(canvas, origin, (origin[0] - int(axis_length * sin_a), origin[1] - int(axis_length * cos_a)), (0, 255, 0), 2)
        # Z轴 (蓝色，向上)
        cv2.line(canvas, origin, (origin[0], origin[1] - int(axis_length * 0.7)), (255, 0, 0), 2)
        
        # 标题
        cv2.putText(canvas, f"LiDAR 3D View | Points: {len(lidar_points)}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return canvas
    
    def render_camera_view(self, image: np.ndarray, 
                          detections: Optional[List[Dict]] = None,
                          title: str = "Camera") -> np.ndarray:
        """
        渲染相机图像（带检测框）
        
        Args:
            image: 输入图像
            detections: 检测结果列表 [{'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}]
            title: 标题
        
        Returns:
            渲染后的图像
        """
        if image is None:
            # 创建空白图像
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "No Image", (250, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return canvas
        
        # 复制图像
        canvas = image.copy()
        
        # 绘制检测框
        if detections is not None:
            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                class_name = det.get('class', 'unknown')
                confidence = det.get('confidence', 0)
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 选择颜色
                color_map = {
                    'vehicle': (0, 100, 255),
                    'pedestrian': (255, 0, 255),
                    'cyclist': (255, 255, 0),
                    'traffic_sign': (0, 255, 0),
                }
                color = color_map.get(class_name, (200, 200, 200))
                
                # 绘制边界框
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(canvas, (x1, y1 - label_size[1] - 5),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(canvas, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加标题
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, title, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return canvas
    
    def render_multi_camera(self, images: Dict[str, np.ndarray],
                           layout: str = '2x2') -> np.ndarray:
        """
        渲染多相机视图
        
        Args:
            images: 相机图像字典 {'front': img, 'left': img, ...}
            layout: 布局 '2x2', '2x3', '1x4', 'vertical'
        
        Returns:
            组合后的图像
        """
        if not images:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 标准化图像大小
        target_size = (320, 240)
        standardized = []
        titles = []
        
        for name, img in images.items():
            if img is not None:
                resized = cv2.resize(img, target_size)
                titled = self.render_camera_view(resized, title=name)
                standardized.append(titled)
                titles.append(name)
        
        if not standardized:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 组合图像
        if layout == '2x2' and len(standardized) >= 4:
            top = np.hstack(standardized[0:2])
            bottom = np.hstack(standardized[2:4])
            combined = np.vstack([top, bottom])
        elif layout == '2x3' and len(standardized) >= 6:
            top = np.hstack(standardized[0:3])
            bottom = np.hstack(standardized[3:6])
            combined = np.vstack([top, bottom])
        elif layout == '1x4' and len(standardized) >= 4:
            combined = np.hstack(standardized[0:4])
        elif layout == 'vertical':
            combined = np.vstack(standardized)
        else:
            # 默认水平排列
            combined = np.hstack(standardized)
        
        return combined
    
    def create_sensor_fusion_view(self, camera_image: np.ndarray,
                                   lidar_points: np.ndarray,
                                   projection_matrix: np.ndarray,
                                   ego: EgoState) -> np.ndarray:
        """
        创建相机-LiDAR融合视图
        
        将LiDAR点云投影到相机图像上
        
        Args:
            camera_image: 相机图像
            lidar_points: LiDAR点云
            projection_matrix: 3x4投影矩阵
            ego: 自车状态
        
        Returns:
            融合图像
        """
        if camera_image is None or lidar_points is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        canvas = camera_image.copy()
        h, w = canvas.shape[:2]
        
        # 投影点云
        for point in lidar_points:
            x, y, z, intensity = point
            
            # 转换到相机坐标系（简化处理）
            # 实际应用中需要完整的坐标变换
            point_cam = np.array([x - ego.x, y - ego.y, z, 1])
            
            # 投影
            projected = projection_matrix @ point_cam
            if projected[2] > 0:
                u = int(projected[0] / projected[2])
                v = int(projected[1] / projected[2])
                
                if 0 <= u < w and 0 <= v < h:
                    # 根据高度着色
                    color = self.height_to_color(z)
                    cv2.circle(canvas, (u, v), 3, color, -1)
        
        return canvas
    
    def create_point_cloud_histogram(self, lidar_points: np.ndarray,
                                     bins: int = 50) -> np.ndarray:
        """
        创建点云高度直方图
        
        Args:
            lidar_points: LiDAR点云
            bins: 直方图bin数量
        
        Returns:
            直方图图像
        """
        if lidar_points is None or len(lidar_points) == 0:
            return np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 提取高度值
        heights = lidar_points[:, 2]
        
        # 创建直方图
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(heights, bins=bins, color='skyblue', edgecolor='black')
        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Count')
        ax.set_title('Point Cloud Height Distribution')
        ax.grid(True, alpha=0.3)
        
        # 转换为图像
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        
        # 转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img


def create_range_image(lidar_points: np.ndarray,
                       range_bins: int = 100,
                       azimuth_bins: int = 360) -> np.ndarray:
    """
    创建距离图像（Range Image）
    
    Args:
        lidar_points: Nx4点云数组
        range_bins: 距离bin数量
        azimuth_bins: 方位角bin数量
    
    Returns:
        距离图像
    """
    if lidar_points is None or len(lidar_points) == 0:
        return np.zeros((range_bins, azimuth_bins), dtype=np.uint8)
    
    # 计算距离和方位角
    x = lidar_points[:, 0]
    y = lidar_points[:, 1]
    z = lidar_points[:, 2]
    
    ranges = np.sqrt(x**2 + y**2)
    azimuths = np.arctan2(y, x)  # -pi to pi
    
    # 归一化
    max_range = np.percentile(ranges, 99)
    range_normalized = ranges / max_range * (range_bins - 1)
    azimuth_normalized = (azimuths + np.pi) / (2 * np.pi) * (azimuth_bins - 1)
    
    # 创建图像
    range_image = np.zeros((range_bins, azimuth_bins), dtype=np.float32)
    
    for i in range(len(lidar_points)):
        r_idx = int(range_normalized[i])
        a_idx = int(azimuth_normalized[i])
        if 0 <= r_idx < range_bins and 0 <= a_idx < azimuth_bins:
            range_image[r_idx, a_idx] = max(range_image[r_idx, a_idx], z[i] + 2)
    
    # 归一化到0-255
    range_image = (range_image / range_image.max() * 255).astype(np.uint8)
    
    # 应用颜色映射
    range_image_color = cv2.applyColorMap(range_image, cv2.COLORMAP_JET)
    
    return range_image_color
