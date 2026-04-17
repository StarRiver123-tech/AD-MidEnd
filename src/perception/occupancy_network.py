"""
自动驾驶系统 - Occupancy占据网络
实现3D空间占据预测
"""

from typing import Optional, Tuple
import numpy as np

from ..common.data_types import (
    PointCloud, OccupancyResult, OccupancyGrid, OccupancyCell, Timestamp
)
from ..logs.logger import Logger


class OccupancyNetwork:
    """Occupancy占据网络"""
    
    def __init__(self, config: dict = None):
        """
        初始化Occupancy网络
        
        Args:
            config: 配置参数
        """
        self._config = config or {}
        self._logger = Logger("OccupancyNetwork")
        
        # 参数
        self._resolution = self._config.get('resolution', 0.2)  # 米/格
        self._range_x = self._config.get('range_x', [-50, 50])
        self._range_y = self._config.get('range_y', [-30, 30])
        self._range_z = self._config.get('range_z', [-3, 3])
        
        # 体素参数
        self._voxel_resolution = self._config.get('voxel_resolution', 0.5)
        
        # 模型（实际应该加载深度学习模型）
        self._model = None
        
        # 栅格尺寸
        self._grid_width = int((self._range_x[1] - self._range_x[0]) / self._resolution)
        self._grid_height = int((self._range_y[1] - self._range_y[0]) / self._resolution)
        self._grid_depth = int((self._range_z[1] - self._range_z[0]) / self._voxel_resolution)
        
        self._logger.info(f"OccupancyNetwork initialized: "
                         f"grid_size={self._grid_width}x{self._grid_height}, "
                         f"resolution={self._resolution}m")
    
    def predict(self, lidar_data: PointCloud) -> Optional[OccupancyResult]:
        """
        预测Occupancy
        
        Args:
            lidar_data: LiDAR点云数据
        
        Returns:
            Occupancy预测结果
        """
        # 这里应该调用深度学习模型
        # 简化实现：基于点云生成Occupancy栅格
        
        # 生成2D Occupancy栅格
        occupancy_2d = self._generate_2d_occupancy(lidar_data)
        
        # 生成3D Occupancy体素
        occupancy_3d = self._generate_3d_occupancy(lidar_data)
        
        result = OccupancyResult(
            timestamp=Timestamp.now(),
            occupancy_2d=occupancy_2d,
            occupancy_3d=occupancy_3d,
            range_x=tuple(self._range_x),
            range_y=tuple(self._range_y),
            range_z=tuple(self._range_z),
            voxel_resolution=self._voxel_resolution
        )
        
        return result
    
    def _generate_2d_occupancy(self, lidar_data: PointCloud) -> OccupancyGrid:
        """生成2D Occupancy栅格"""
        points = lidar_data.get_xyz()
        
        # 创建栅格
        grid = np.zeros((self._grid_height, self._grid_width), dtype=np.float32)
        height_grid = np.zeros((self._grid_height, self._grid_width), dtype=np.float32)
        semantic_grid = np.zeros((self._grid_height, self._grid_width), dtype=np.int32)
        
        # 将点云投影到栅格
        for point in points:
            x, y, z = point
            
            # 检查范围
            if not (self._range_x[0] <= x <= self._range_x[1] and
                    self._range_y[0] <= y <= self._range_y[1]):
                continue
            
            # 计算栅格坐标
            grid_x = int((x - self._range_x[0]) / self._resolution)
            grid_y = int((y - self._range_y[0]) / self._resolution)
            
            # 检查边界
            if 0 <= grid_x < self._grid_width and 0 <= grid_y < self._grid_height:
                grid[grid_y, grid_x] = 1.0
                
                # 更新高度
                if z > height_grid[grid_y, grid_x]:
                    height_grid[grid_y, grid_x] = z
                
                # 简单的语义分类
                semantic_grid[grid_y, grid_x] = self._classify_semantic(z)
        
        # 膨胀栅格（模拟占据的不确定性）
        grid = self._dilate_grid(grid, iterations=2)
        
        occupancy_grid = OccupancyGrid(
            timestamp=Timestamp.now(),
            resolution=self._resolution,
            width=self._grid_width,
            height=self._grid_height,
            origin_x=self._range_x[0],
            origin_y=self._range_y[0],
            data=grid.reshape(self._grid_height, self._grid_width, 1),
            height_data=height_grid,
            semantic_data=semantic_grid
        )
        
        return occupancy_grid
    
    def _generate_3d_occupancy(self, lidar_data: PointCloud) -> np.ndarray:
        """生成3D Occupancy体素"""
        points = lidar_data.get_xyz()
        
        # 创建3D体素栅格
        voxel_x = int((self._range_x[1] - self._range_x[0]) / self._voxel_resolution)
        voxel_y = int((self._range_y[1] - self._range_y[0]) / self._voxel_resolution)
        voxel_z = int((self._range_z[1] - self._range_z[0]) / self._voxel_resolution)
        
        occupancy_3d = np.zeros((voxel_x, voxel_y, voxel_z), dtype=np.float32)
        
        # 将点云投影到体素
        for point in points:
            x, y, z = point
            
            # 检查范围
            if not (self._range_x[0] <= x <= self._range_x[1] and
                    self._range_y[0] <= y <= self._range_y[1] and
                    self._range_z[0] <= z <= self._range_z[1]):
                continue
            
            # 计算体素坐标
            voxel_i = int((x - self._range_x[0]) / self._voxel_resolution)
            voxel_j = int((y - self._range_y[0]) / self._voxel_resolution)
            voxel_k = int((z - self._range_z[0]) / self._voxel_resolution)
            
            # 检查边界
            if (0 <= voxel_i < voxel_x and 
                0 <= voxel_j < voxel_y and 
                0 <= voxel_k < voxel_z):
                occupancy_3d[voxel_i, voxel_j, voxel_k] = 1.0
        
        return occupancy_3d
    
    def _dilate_grid(self, grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """膨胀栅格"""
        from scipy import ndimage
        
        dilated = grid.copy()
        for _ in range(iterations):
            dilated = ndimage.binary_dilation(dilated).astype(np.float32)
        
        # 添加概率衰减
        for _ in range(iterations):
            dilated = ndimage.gaussian_filter(dilated, sigma=1.0)
        
        return np.clip(dilated, 0, 1)
    
    def _classify_semantic(self, height: float) -> int:
        """简单的语义分类"""
        if height < 0.2:
            return 1  # 地面
        elif height < 0.5:
            return 2  # 低矮障碍物
        elif height < 1.5:
            return 3  # 中等高度
        elif height < 2.5:
            return 4  # 车辆
        else:
            return 5  # 高大物体
    
    def get_occupancy_at_position(self, occupancy_result: OccupancyResult,
                                  x: float, y: float, z: Optional[float] = None) -> float:
        """获取指定位置的占据概率"""
        if z is None:
            # 2D查询
            if occupancy_result.occupancy_2d is None:
                return 0.0
            
            grid = occupancy_result.occupancy_2d
            
            # 检查范围
            if not (grid.origin_x <= x <= grid.origin_x + grid.width * grid.resolution and
                    grid.origin_y <= y <= grid.origin_y + grid.height * grid.resolution):
                return 0.0
            
            # 计算栅格坐标
            grid_x = int((x - grid.origin_x) / grid.resolution)
            grid_y = int((y - grid.origin_y) / grid.resolution)
            
            # 检查边界
            if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
                return grid.data[grid_y, grid_x, 0]
            
            return 0.0
        else:
            # 3D查询
            if occupancy_result.occupancy_3d is None:
                return 0.0
            
            # 计算体素坐标
            voxel_x = int((x - occupancy_result.range_x[0]) / occupancy_result.voxel_resolution)
            voxel_y = int((y - occupancy_result.range_y[0]) / occupancy_result.voxel_resolution)
            voxel_z = int((z - occupancy_result.range_z[0]) / occupancy_result.voxel_resolution)
            
            # 检查边界
            shape = occupancy_result.occupancy_3d.shape
            if (0 <= voxel_x < shape[0] and 
                0 <= voxel_y < shape[1] and 
                0 <= voxel_z < shape[2]):
                return occupancy_result.occupancy_3d[voxel_x, voxel_y, voxel_z]
            
            return 0.0
    
    def check_collision(self, occupancy_result: OccupancyResult,
                       positions: np.ndarray, radius: float = 0.5) -> np.ndarray:
        """检查位置是否与占据栅格碰撞"""
        collisions = np.zeros(len(positions), dtype=bool)
        
        for i, pos in enumerate(positions):
            x, y, z = pos
            
            # 检查中心点
            if self.get_occupancy_at_position(occupancy_result, x, y, z) > 0.5:
                collisions[i] = True
                continue
            
            # 检查周围区域
            for dx in [-radius, 0, radius]:
                for dy in [-radius, 0, radius]:
                    if self.get_occupancy_at_position(occupancy_result, x + dx, y + dy, z) > 0.5:
                        collisions[i] = True
                        break
                if collisions[i]:
                    break
        
        return collisions
