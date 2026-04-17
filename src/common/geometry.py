"""
自动驾驶系统 - 几何工具函数
提供坐标变换、距离计算等几何运算
"""

import numpy as np
from typing import Tuple, List, Optional
from .data_types import Vector3D, Pose, Quaternion, BoundingBox3D, TrajectoryPoint


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """从欧拉角创建旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_x = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])
    
    R_y = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])
    
    R_z = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x


def rotation_matrix_from_quaternion(q: Quaternion) -> np.ndarray:
    """从四元数创建旋转矩阵"""
    x, y, z, w = q.x, q.y, q.z, q.w
    
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def quaternion_from_rotation_matrix(R: np.ndarray) -> Quaternion:
    """从旋转矩阵创建四元数"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return Quaternion(x, y, z, w)


def transform_matrix_from_pose(pose: Pose) -> np.ndarray:
    """从位姿创建4x4变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix_from_quaternion(pose.orientation)
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def pose_from_transform_matrix(T: np.ndarray) -> Pose:
    """从4x4变换矩阵创建位姿"""
    position = Vector3D(T[0, 3], T[1, 3], T[2, 3])
    orientation = quaternion_from_rotation_matrix(T[:3, :3])
    return Pose(position, orientation)


def transform_point(point: Vector3D, transform: np.ndarray) -> Vector3D:
    """使用4x4变换矩阵变换点"""
    p = np.array([point.x, point.y, point.z, 1.0])
    p_transformed = transform @ p
    return Vector3D(p_transformed[0], p_transformed[1], p_transformed[2])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """批量变换点 Nx3 -> Nx3"""
    N = points.shape[0]
    ones = np.ones((N, 1))
    points_h = np.hstack([points, ones])  # Nx4
    points_transformed = (transform @ points_h.T).T  # Nx4
    return points_transformed[:, :3]  # Nx3


def transform_pose(pose: Pose, transform: np.ndarray) -> Pose:
    """变换位姿"""
    T = transform_matrix_from_pose(pose)
    T_new = transform @ T
    return pose_from_transform_matrix(T_new)


def calculate_distance(p1: Vector3D, p2: Vector3D) -> float:
    """计算两点间欧氏距离"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def calculate_distance_2d(p1: Vector3D, p2: Vector3D) -> float:
    """计算两点间2D欧氏距离（忽略Z）"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calculate_bounding_box_corners(bbox: BoundingBox3D) -> np.ndarray:
    """计算3D边界框的8个角点"""
    l, w, h = bbox.size.x, bbox.size.y, bbox.size.z
    
    # 局部坐标系中的角点
    corners_local = np.array([
        [l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
        [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
    ])
    
    # 变换到世界坐标系
    T = transform_matrix_from_pose(bbox.center)
    corners_world = transform_points(corners_local, T)
    
    return corners_world


def calculate_iou_3d(bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
    """计算两个3D边界框的IoU（简化版，使用2D投影）"""
    # 获取边界框在XY平面的投影
    def get_bbox_2d(bbox: BoundingBox3D):
        cx, cy = bbox.center.position.x, bbox.center.position.y
        l, w = bbox.size.x, bbox.size.y
        yaw = bbox.center.orientation.to_euler()[2]
        
        # 计算旋转后的角点
        corners = np.array([
            [-l/2, -w/2], [-l/2, w/2], [l/2, w/2], [l/2, -w/2]
        ])
        
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        
        corners_rotated = (R @ corners.T).T + np.array([cx, cy])
        return corners_rotated
    
    # 使用shapely计算IoU（如果可用）
    try:
        from shapely.geometry import Polygon
        poly1 = Polygon(get_bbox_2d(bbox1))
        poly2 = Polygon(get_bbox_2d(bbox2))
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        return intersection / union if union > 0 else 0.0
    except ImportError:
        # 简化计算：使用中心点距离
        dist = calculate_distance_2d(bbox1.center.position, bbox2.center.position)
        max_dist = max(bbox1.size.x, bbox1.size.y, bbox2.size.x, bbox2.size.y)
        return max(0.0, 1.0 - dist / max_dist)


def interpolate_trajectory(trajectory: List[TrajectoryPoint], 
                          target_time: float) -> Optional[TrajectoryPoint]:
    """在轨迹中插值获取指定时间的点"""
    if not trajectory:
        return None
    
    # 找到目标时间前后的点
    prev_point = None
    next_point = None
    
    for point in trajectory:
        if point.relative_time <= target_time:
            prev_point = point
        if point.relative_time >= target_time and next_point is None:
            next_point = point
            break
    
    if prev_point is None:
        return trajectory[0]
    if next_point is None:
        return trajectory[-1]
    if prev_point == next_point:
        return prev_point
    
    # 线性插值
    t1, t2 = prev_point.relative_time, next_point.relative_time
    if t2 - t1 < 1e-6:
        return prev_point
    
    alpha = (target_time - t1) / (t2 - t1)
    
    interpolated = TrajectoryPoint()
    interpolated.relative_time = target_time
    
    # 位置插值
    interpolated.pose.position.x = prev_point.pose.position.x + \
        alpha * (next_point.pose.position.x - prev_point.pose.position.x)
    interpolated.pose.position.y = prev_point.pose.position.y + \
        alpha * (next_point.pose.position.y - prev_point.pose.position.y)
    interpolated.pose.position.z = prev_point.pose.position.z + \
        alpha * (next_point.pose.position.z - prev_point.pose.position.z)
    
    # 速度插值
    interpolated.longitudinal_velocity = prev_point.longitudinal_velocity + \
        alpha * (next_point.longitudinal_velocity - prev_point.longitudinal_velocity)
    
    return interpolated


def calculate_curvature_from_three_points(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> float:
    """通过三点计算曲率"""
    # 使用公式: k = 4*A / (|a|*|b|*|c|)
    # 其中A是三角形面积，a,b,c是边长
    
    a = calculate_distance_2d(p2, p3)
    b = calculate_distance_2d(p1, p3)
    c = calculate_distance_2d(p1, p2)
    
    # 使用叉积计算面积
    area = 0.5 * abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    
    if a * b * c < 1e-6:
        return 0.0
    
    curvature = 4 * area / (a * b * c)
    return curvature


def world_to_vehicle(point: Vector3D, ego_pose: Pose) -> Vector3D:
    """将世界坐标系中的点转换到车辆坐标系"""
    T_world_to_vehicle = np.linalg.inv(transform_matrix_from_pose(ego_pose))
    return transform_point(point, T_world_to_vehicle)


def vehicle_to_world(point: Vector3D, ego_pose: Pose) -> Vector3D:
    """将车辆坐标系中的点转换到世界坐标系"""
    T_vehicle_to_world = transform_matrix_from_pose(ego_pose)
    return transform_point(point, T_vehicle_to_world)


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def calculate_heading_from_points(start: Vector3D, end: Vector3D) -> float:
    """计算从起点到终点的航向角"""
    dx = end.x - start.x
    dy = end.y - start.y
    return np.arctan2(dy, dx)
