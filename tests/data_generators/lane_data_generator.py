"""
车道线数据生成器
生成模拟的车道线数据
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class LaneType(Enum):
    """车道线类型"""
    SOLID_WHITE = "solid_white"
    DASHED_WHITE = "dashed_white"
    SOLID_YELLOW = "solid_yellow"
    DOUBLE_YELLOW = "double_yellow"
    BROKEN_YELLOW = "broken_yellow"
    UNKNOWN = "unknown"


class LanePosition(Enum):
    """车道线位置"""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


@dataclass
class LanePoint:
    """车道线上的点"""
    x: float
    y: float
    z: float
    curvature: float  # 曲率
    heading: float  # 朝向角
    confidence: float


@dataclass
class LaneLine:
    """车道线"""
    id: str
    lane_type: LaneType
    position: LanePosition
    points: List[LanePoint]
    width: float
    confidence: float
    visible: bool
    
    def get_length(self) -> float:
        """计算车道线长度"""
        if len(self.points) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(self.points)):
            p1 = self.points[i-1]
            p2 = self.points[i]
            length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
        
        return length
    
    def get_average_curvature(self) -> float:
        """计算平均曲率"""
        if not self.points:
            return 0.0
        curvatures = [p.curvature for p in self.points]
        return np.mean(curvatures)


@dataclass
class LaneInfo:
    """车道信息"""
    lane_id: str
    left_boundary: Optional[LaneLine]
    right_boundary: Optional[LaneLine]
    center_line: Optional[LaneLine]
    lane_width: float
    is_valid: bool


@dataclass
class LaneDetectionResult:
    """车道线检测结果"""
    timestamp: float
    lane_lines: List[LaneLine]
    ego_lane: Optional[LaneInfo]
    confidence: float


class LaneDataGenerator:
    """车道线数据生成器"""
    
    # 默认车道线宽度
    DEFAULT_LANE_WIDTH = 3.5  # 米
    
    # 默认车道线颜色宽度
    LINE_WIDTHS = {
        LaneType.SOLID_WHITE: 0.15,
        LaneType.DASHED_WHITE: 0.15,
        LaneType.SOLID_YELLOW: 0.15,
        LaneType.DOUBLE_YELLOW: 0.30,
        LaneType.BROKEN_YELLOW: 0.15,
        LaneType.UNKNOWN: 0.15,
    }
    
    def __init__(self, seed: int = 42):
        """
        初始化车道线数据生成器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.RandomState(seed)
        self.lane_width = self.DEFAULT_LANE_WIDTH
    
    def generate_straight_lane(
        self,
        start_y: float = 0.0,
        length: float = 100.0,
        num_points: int = 100,
        lane_type: LaneType = LaneType.DASHED_WHITE,
        position: LanePosition = LanePosition.CENTER,
        timestamp: float = 0.0
    ) -> LaneLine:
        """
        生成直线车道线
        
        Args:
            start_y: 起始y坐标
            length: 车道线长度
            num_points: 点的数量
            lane_type: 车道线类型
            position: 车道线位置
            timestamp: 时间戳
            
        Returns:
            LaneLine对象
        """
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = 0.0
            y = start_y + length * t
            z = 0.0
            
            point = LanePoint(
                x=x,
                y=y,
                z=z,
                curvature=0.0,
                heading=0.0,
                confidence=self.rng.uniform(0.9, 0.99)
            )
            points.append(point)
        
        return LaneLine(
            id=f"lane_{position.value}_{int(timestamp*1000)}",
            lane_type=lane_type,
            position=position,
            points=points,
            width=self.LINE_WIDTHS[lane_type],
            confidence=self.rng.uniform(0.9, 0.99),
            visible=True
        )
    
    def generate_curved_lane(
        self,
        start_y: float = 0.0,
        length: float = 100.0,
        curvature: float = 0.01,
        num_points: int = 100,
        lane_type: LaneType = LaneType.DASHED_WHITE,
        position: LanePosition = LanePosition.CENTER,
        timestamp: float = 0.0
    ) -> LaneLine:
        """
        生成曲线车道线
        
        Args:
            start_y: 起始y坐标
            length: 车道线长度
            curvature: 曲率（1/R）
            num_points: 点的数量
            lane_type: 车道线类型
            position: 车道线位置
            timestamp: 时间戳
            
        Returns:
            LaneLine对象
        """
        points = []
        
        # 使用圆弧模型
        radius = 1.0 / curvature if curvature != 0 else float('inf')
        
        for i in range(num_points):
            t = i / (num_points - 1)
            arc_length = length * t
            angle = arc_length / radius
            
            x = radius * (1 - np.cos(angle))
            y = start_y + radius * np.sin(angle)
            z = 0.0
            
            point = LanePoint(
                x=x,
                y=y,
                z=z,
                curvature=curvature,
                heading=angle,
                confidence=self.rng.uniform(0.85, 0.98)
            )
            points.append(point)
        
        return LaneLine(
            id=f"lane_{position.value}_{int(timestamp*1000)}",
            lane_type=lane_type,
            position=position,
            points=points,
            width=self.LINE_WIDTHS[lane_type],
            confidence=self.rng.uniform(0.85, 0.98),
            visible=True
        )
    
    def generate_sine_lane(
        self,
        start_y: float = 0.0,
        length: float = 100.0,
        amplitude: float = 2.0,
        frequency: float = 0.1,
        num_points: int = 100,
        lane_type: LaneType = LaneType.DASHED_WHITE,
        position: LanePosition = LanePosition.CENTER,
        timestamp: float = 0.0
    ) -> LaneLine:
        """
        生成正弦曲线车道线
        
        Args:
            start_y: 起始y坐标
            length: 车道线长度
            amplitude: 振幅
            frequency: 频率
            num_points: 点的数量
            lane_type: 车道线类型
            position: 车道线位置
            timestamp: 时间戳
            
        Returns:
            LaneLine对象
        """
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            y = start_y + length * t
            x = amplitude * np.sin(2 * np.pi * frequency * t * length)
            z = 0.0
            
            # 计算曲率（二阶导数）
            dx_dt = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t * length)
            d2x_dt2 = -amplitude * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * t * length)
            curvature = abs(d2x_dt2) / (1 + dx_dt**2)**1.5
            
            # 计算朝向
            heading = np.arctan2(dx_dt, 1.0)
            
            point = LanePoint(
                x=x,
                y=y,
                z=z,
                curvature=curvature,
                heading=heading,
                confidence=self.rng.uniform(0.8, 0.95)
            )
            points.append(point)
        
        return LaneLine(
            id=f"lane_{position.value}_{int(timestamp*1000)}",
            lane_type=lane_type,
            position=position,
            points=points,
            width=self.LINE_WIDTHS[lane_type],
            confidence=self.rng.uniform(0.8, 0.95),
            visible=True
        )
    
    def generate_lane_segment(
        self,
        ego_position: Tuple[float, float, float] = (0, 0, 0),
        num_lanes: int = 3,
        lane_width: float = 3.5,
        length: float = 100.0,
        curvature: float = 0.0,
        timestamp: float = 0.0
    ) -> LaneDetectionResult:
        """
        生成车道段（包含多条车道线）
        
        Args:
            ego_position: 自车位置
            num_lanes: 车道数量
            lane_width: 车道宽度
            length: 车道线长度
            curvature: 曲率
            timestamp: 时间戳
            
        Returns:
            LaneDetectionResult对象
        """
        lane_lines = []
        
        # 生成中心车道线
        if curvature == 0:
            center_line = self.generate_straight_lane(
                start_y=ego_position[1],
                length=length,
                lane_type=LaneType.DASHED_WHITE,
                position=LanePosition.CENTER,
                timestamp=timestamp
            )
        else:
            center_line = self.generate_curved_lane(
                start_y=ego_position[1],
                length=length,
                curvature=curvature,
                lane_type=LaneType.DASHED_WHITE,
                position=LanePosition.CENTER,
                timestamp=timestamp
            )
        
        # 根据车道数量生成边界
        for i in range(num_lanes + 1):
            offset = (i - num_lanes / 2) * lane_width
            
            # 确定车道线类型
            if i == 0:
                lane_type = LaneType.SOLID_WHITE
                position = LanePosition.LEFT
            elif i == num_lanes:
                lane_type = LaneType.SOLID_WHITE
                position = LanePosition.RIGHT
            else:
                lane_type = LaneType.DASHED_WHITE
                position = LanePosition.CENTER
            
            # 生成车道线
            if curvature == 0:
                lane = self.generate_straight_lane(
                    start_y=ego_position[1],
                    length=length,
                    lane_type=lane_type,
                    position=position,
                    timestamp=timestamp
                )
            else:
                lane = self.generate_curved_lane(
                    start_y=ego_position[1],
                    length=length,
                    curvature=curvature,
                    lane_type=lane_type,
                    position=position,
                    timestamp=timestamp
                )
            
            # 平移车道线
            for point in lane.points:
                point.x += offset
            
            lane_lines.append(lane)
        
        # 生成自车所在车道信息
        ego_lane = LaneInfo(
            lane_id="ego_lane",
            left_boundary=lane_lines[num_lanes // 2],
            right_boundary=lane_lines[num_lanes // 2 + 1],
            center_line=center_line,
            lane_width=lane_width,
            is_valid=True
        )
        
        return LaneDetectionResult(
            timestamp=timestamp,
            lane_lines=lane_lines,
            ego_lane=ego_lane,
            confidence=self.rng.uniform(0.9, 0.98)
        )
    
    def generate_lane_with_noise(
        self,
        ground_truth: LaneLine,
        position_noise: float = 0.1,
        point_dropout_rate: float = 0.05,
        timestamp: float = 0.0
    ) -> LaneLine:
        """
        为车道线添加噪声（模拟检测结果）
        
        Args:
            ground_truth: 真值车道线
            position_noise: 位置噪声
            point_dropout_rate: 点丢失率
            timestamp: 时间戳
            
        Returns:
            带噪声的LaneLine对象
        """
        noisy_points = []
        
        for point in ground_truth.points:
            # 模拟点丢失
            if self.rng.random() < point_dropout_rate:
                continue
            
            # 添加位置噪声
            noisy_point = LanePoint(
                x=point.x + self.rng.normal(0, position_noise),
                y=point.y + self.rng.normal(0, position_noise),
                z=point.z + self.rng.normal(0, position_noise * 0.5),
                curvature=point.curvature + self.rng.normal(0, 0.001),
                heading=point.heading + self.rng.normal(0, 0.01),
                confidence=max(0.5, point.confidence + self.rng.normal(0, 0.05))
            )
            noisy_points.append(noisy_point)
        
        return LaneLine(
            id=f"detected_{ground_truth.id}",
            lane_type=ground_truth.lane_type,
            position=ground_truth.position,
            points=noisy_points,
            width=ground_truth.width,
            confidence=max(0.5, ground_truth.confidence + self.rng.normal(0, 0.05)),
            visible=len(noisy_points) > 5  # 如果点数太少则标记为不可见
        )
    
    def generate_occluded_lane(
        self,
        ground_truth: LaneLine,
        occlusion_start: float = 0.3,
        occlusion_end: float = 0.7,
        timestamp: float = 0.0
    ) -> LaneLine:
        """
        生成被遮挡的车道线
        
        Args:
            ground_truth: 真值车道线
            occlusion_start: 遮挡开始位置（0-1）
            occlusion_end: 遮挡结束位置（0-1）
            timestamp: 时间戳
            
        Returns:
            被遮挡的LaneLine对象
        """
        n = len(ground_truth.points)
        start_idx = int(n * occlusion_start)
        end_idx = int(n * occlusion_end)
        
        visible_points = []
        for i, point in enumerate(ground_truth.points):
            if i < start_idx or i > end_idx:
                visible_points.append(point)
        
        return LaneLine(
            id=f"occluded_{ground_truth.id}",
            lane_type=ground_truth.lane_type,
            position=ground_truth.position,
            points=visible_points,
            width=ground_truth.width,
            confidence=ground_truth.confidence * 0.7,
            visible=len(visible_points) > n * 0.3
        )
