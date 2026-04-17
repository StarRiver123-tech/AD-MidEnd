"""
自动驾驶系统 - 障碍物检测器
实现障碍物检测和跟踪功能
"""

from typing import List, Optional, Dict
import numpy as np

from ..common.data_types import (
    ImageData, PointCloud, RadarData,
    Obstacle, ObstacleDetectionResult, BoundingBox3D, Pose, Vector3D, Timestamp
)
from ..common.geometry import transform_points, calculate_distance_2d
from ..logs.logger import Logger


class ObstacleDetector:
    """障碍物检测器"""
    
    def __init__(self, config: dict = None):
        """
        初始化障碍物检测器
        
        Args:
            config: 配置参数
        """
        self._config = config or {}
        self._logger = Logger("ObstacleDetector")
        
        # 参数
        self._confidence_threshold = self._config.get('confidence_threshold', 0.6)
        self._max_detection_distance = self._config.get('max_detection_distance', 150.0)
        self._min_detection_distance = self._config.get('min_detection_distance', 0.5)
        
        # 跟踪器
        self._next_tracking_id = 0
        self._tracked_obstacles: Dict[int, Obstacle] = {}
        
        # 模型（实际应该加载深度学习模型）
        self._model = None
        
        self._logger.info("ObstacleDetector initialized")
    
    def detect(self, 
               camera_data: Optional[ImageData] = None,
               lidar_data: Optional[PointCloud] = None,
               radar_data: Optional[RadarData] = None) -> Optional[ObstacleDetectionResult]:
        """
        检测障碍物
        
        Args:
            camera_data: 摄像头数据
            lidar_data: LiDAR数据
            radar_data: 雷达数据
        
        Returns:
            障碍物检测结果
        """
        obstacles = []
        
        # 基于LiDAR检测障碍物
        if lidar_data is not None:
            lidar_obstacles = self._detect_from_lidar(lidar_data)
            obstacles.extend(lidar_obstacles)
        
        # 基于雷达检测障碍物
        if radar_data is not None:
            radar_obstacles = self._detect_from_radar(radar_data)
            obstacles.extend(radar_obstacles)
        
        # 基于摄像头检测障碍物（可选）
        if camera_data is not None:
            camera_obstacles = self._detect_from_camera(camera_data)
            obstacles.extend(camera_obstacles)
        
        # 融合检测结果
        fused_obstacles = self._fuse_detections(obstacles)
        
        # 跟踪障碍物
        tracked_obstacles = self._track_obstacles(fused_obstacles)
        
        # 统计
        num_vehicles = sum(1 for o in tracked_obstacles if 'vehicle' in o.obstacle_type)
        num_pedestrians = sum(1 for o in tracked_obstacles if o.obstacle_type == 'pedestrian')
        num_cyclists = sum(1 for o in tracked_obstacles if o.obstacle_type == 'cyclist')
        num_unknown = len(tracked_obstacles) - num_vehicles - num_pedestrians - num_cyclists
        
        result = ObstacleDetectionResult(
            timestamp=Timestamp.now(),
            obstacles=tracked_obstacles,
            num_vehicles=num_vehicles,
            num_pedestrians=num_pedestrians,
            num_cyclists=num_cyclists,
            num_unknown=num_unknown
        )
        
        return result
    
    def _detect_from_lidar(self, lidar_data: PointCloud) -> List[Obstacle]:
        """基于LiDAR点云检测障碍物"""
        obstacles = []
        
        # 获取点云
        points = lidar_data.get_xyz()
        
        if len(points) == 0:
            return obstacles
        
        # 简单的聚类算法（实际应该使用更复杂的算法如DBSCAN）
        # 这里使用简化的网格聚类
        clusters = self._cluster_points(points)
        
        for cluster_id, cluster_points in clusters.items():
            if len(cluster_points) < 10:  # 过滤小簇
                continue
            
            # 计算边界框
            bbox = self._compute_bounding_box(cluster_points)
            
            # 估计障碍物类型
            obstacle_type = self._classify_obstacle(cluster_points, bbox)
            
            obstacle = Obstacle(
                obstacle_id=cluster_id,
                obstacle_type=obstacle_type,
                bbox=bbox,
                confidence=min(1.0, len(cluster_points) / 100),
                is_moving=False,  # 需要通过多帧判断
                is_occluded=False
            )
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def _cluster_points(self, points: np.ndarray, 
                       grid_size: float = 0.5) -> Dict[int, np.ndarray]:
        """简单的网格聚类"""
        clusters = {}
        
        # 创建网格
        grid_indices = np.floor(points[:, :2] / grid_size).astype(int)
        unique_indices, inverse = np.unique(grid_indices, axis=0, return_inverse=True)
        
        # 将点分配到簇
        for i, idx in enumerate(inverse):
            if idx not in clusters:
                clusters[idx] = []
            clusters[idx].append(points[i])
        
        # 转换为numpy数组
        clusters = {k: np.array(v) for k, v in clusters.items()}
        
        return clusters
    
    def _compute_bounding_box(self, points: np.ndarray) -> BoundingBox3D:
        """计算点云的3D边界框"""
        # 计算中心点
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        center_z = np.mean(points[:, 2])
        
        # 计算尺寸
        size_x = np.max(points[:, 0]) - np.min(points[:, 0])
        size_y = np.max(points[:, 1]) - np.min(points[:, 1])
        size_z = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # 限制最小尺寸
        size_x = max(size_x, 0.5)
        size_y = max(size_y, 0.5)
        size_z = max(size_z, 0.5)
        
        bbox = BoundingBox3D(
            center=Pose(
                position=Vector3D(center_x, center_y, center_z),
                orientation=None
            ),
            size=Vector3D(size_x, size_y, size_z),
            velocity=Vector3D(0, 0, 0)
        )
        
        return bbox
    
    def _classify_obstacle(self, points: np.ndarray, bbox: BoundingBox3D) -> str:
        """分类障碍物类型"""
        length = bbox.size.x
        width = bbox.size.y
        height = bbox.size.z
        
        # 基于尺寸分类
        if height < 0.8:
            return 'unknown'
        elif height < 1.5:
            if length < 2.0 and width < 1.0:
                return 'pedestrian'
            else:
                return 'vehicle_bike'
        elif height < 2.5:
            if length > 3.0:
                return 'vehicle_car'
            elif length > 1.5:
                return 'cyclist'
            else:
                return 'pedestrian'
        else:
            if length > 5.0:
                return 'vehicle_truck'
            elif length > 3.0:
                return 'vehicle_bus'
            else:
                return 'vehicle_car'
    
    def _detect_from_radar(self, radar_data: RadarData) -> List[Obstacle]:
        """基于雷达数据检测障碍物"""
        obstacles = []
        
        for i, target in enumerate(radar_data.targets):
            # 极坐标转笛卡尔
            x = target.range_distance * np.cos(np.radians(target.azimuth))
            y = target.range_distance * np.sin(np.radians(target.azimuth))
            z = 0.5  # 假设高度
            
            # 估计尺寸（基于RCS）
            size = self._estimate_size_from_rcs(target.rcs)
            
            bbox = BoundingBox3D(
                center=Pose(position=Vector3D(x, y, z)),
                size=Vector3D(size[0], size[1], size[2]),
                velocity=Vector3D(target.velocity, 0, 0)
            )
            
            obstacle = Obstacle(
                obstacle_id=i,
                obstacle_type='vehicle',  # 雷达通常检测车辆
                bbox=bbox,
                confidence=min(1.0, target.snr / 50),
                velocity=Vector3D(target.velocity, 0, 0),
                is_moving=abs(target.velocity) > 0.5
            )
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def _estimate_size_from_rcs(self, rcs: float) -> tuple:
        """基于RCS估计目标尺寸"""
        # 简化的估计
        if rcs > 20:  # 大目标
            return (5.0, 2.0, 1.8)
        elif rcs > 10:  # 中等目标
            return (4.5, 1.8, 1.6)
        else:  # 小目标
            return (4.0, 1.7, 1.5)
    
    def _detect_from_camera(self, camera_data: ImageData) -> List[Obstacle]:
        """基于摄像头检测障碍物"""
        # 这里应该调用深度学习模型
        # 简化实现：返回空列表
        return []
    
    def _fuse_detections(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        """融合多传感器检测结果"""
        if len(obstacles) <= 1:
            return obstacles
        
        # 基于位置进行融合
        fused = []
        used = set()
        
        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
            
            # 寻找匹配的障碍物
            matches = [obs1]
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                if j in used:
                    continue
                
                # 计算距离
                dist = calculate_distance_2d(
                    obs1.bbox.center.position,
                    obs2.bbox.center.position
                )
                
                # 如果距离小于阈值，认为是同一目标
                if dist < 2.0:
                    matches.append(obs2)
                    used.add(j)
            
            # 融合匹配的障碍物
            if len(matches) > 1:
                fused_obs = self._merge_obstacles(matches)
                fused.append(fused_obs)
            else:
                fused.append(obs1)
            
            used.add(i)
        
        return fused
    
    def _merge_obstacles(self, obstacles: List[Obstacle]) -> Obstacle:
        """合并多个障碍物检测结果"""
        # 选择置信度最高的
        best = max(obstacles, key=lambda o: o.confidence)
        
        # 计算平均位置
        avg_x = np.mean([o.bbox.center.position.x for o in obstacles])
        avg_y = np.mean([o.bbox.center.position.y for o in obstacles])
        avg_z = np.mean([o.bbox.center.position.z for o in obstacles])
        
        # 更新边界框
        merged = Obstacle(
            obstacle_id=best.obstacle_id,
            obstacle_type=best.obstacle_type,
            bbox=BoundingBox3D(
                center=Pose(position=Vector3D(avg_x, avg_y, avg_z)),
                size=best.bbox.size,
                velocity=best.bbox.velocity
            ),
            confidence=min(1.0, best.confidence + 0.1),
            velocity=best.velocity,
            is_moving=best.is_moving
        )
        
        return merged
    
    def _track_obstacles(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        """跟踪障碍物"""
        # 简化的跟踪：基于最近邻匹配
        tracked = []
        
        for obstacle in obstacles:
            # 寻找最佳匹配
            best_match_id = None
            best_match_dist = float('inf')
            
            for track_id, tracked_obs in self._tracked_obstacles.items():
                dist = calculate_distance_2d(
                    obstacle.bbox.center.position,
                    tracked_obs.bbox.center.position
                )
                
                if dist < 3.0 and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = track_id
            
            if best_match_id is not None:
                # 更新跟踪
                obstacle.tracking_id = best_match_id
                obstacle.tracking_age = self._tracked_obstacles[best_match_id].tracking_age + 1
                self._tracked_obstacles[best_match_id] = obstacle
            else:
                # 新目标
                obstacle.tracking_id = self._next_tracking_id
                obstacle.tracking_age = 1
                self._tracked_obstacles[self._next_tracking_id] = obstacle
                self._next_tracking_id += 1
            
            tracked.append(obstacle)
        
        return tracked
