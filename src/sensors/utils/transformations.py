"""
坐标变换工具模块
提供传感器坐标系转换相关的函数
"""

import numpy as np
from typing import Tuple, List


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角转换为旋转矩阵（ZYX顺序）
    
    Args:
        roll: 横滚角（弧度）
        pitch: 俯仰角（弧度）
        yaw: 偏航角（弧度）
    
    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    
    R_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    
    R_z = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    
    # ZYX顺序：R = Rz * Ry * Rx
    R = R_z @ R_y @ R_x
    
    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    将旋转矩阵转换为欧拉角（ZYX顺序）
    
    Args:
        R: 3x3旋转矩阵
    
    Returns:
        Tuple[float, float, float]: (roll, pitch, yaw) 弧度
    """
    # 检查万向节锁
    if np.abs(R[2, 0]) >= 1.0:
        # 万向节锁情况
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    
    return roll, pitch, yaw


def compose_transform(translation: np.ndarray, 
                     rotation: np.ndarray) -> np.ndarray:
    """
    组合平移和旋转为4x4变换矩阵
    
    Args:
        translation: 3维平移向量 [x, y, z]
        rotation: 3x3旋转矩阵
    
    Returns:
        np.ndarray: 4x4变换矩阵
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """
    计算变换矩阵的逆
    
    Args:
        T: 4x4变换矩阵
    
    Returns:
        np.ndarray: 4x4逆变换矩阵
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    return T_inv


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    使用变换矩阵变换点云
    
    Args:
        points: Nx3或Nx4的点云数组
        T: 4x4变换矩阵
    
    Returns:
        np.ndarray: 变换后的点云
    """
    if points.shape[1] == 3:
        # 转换为齐次坐标
        points_homo = np.hstack([points, np.ones((len(points), 1))])
    else:
        points_homo = points
    
    # 应用变换
    points_transformed = (T @ points_homo.T).T
    
    if points.shape[1] == 3:
        return points_transformed[:, :3]
    
    return points_transformed


def transform_pose(pose: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    变换位姿（位置和方向）
    
    Args:
        pose: 6维位姿向量 [x, y, z, roll, pitch, yaw]
        T: 4x4变换矩阵
    
    Returns:
        np.ndarray: 变换后的位姿
    """
    # 提取位置和方向
    position = pose[:3]
    orientation = pose[3:]
    
    # 变换位置
    position_homo = np.append(position, 1.0)
    position_transformed = (T @ position_homo)[:3]
    
    # 变换方向（旋转矩阵复合）
    R_original = euler_to_rotation_matrix(*orientation)
    R_transform = T[:3, :3]
    R_result = R_transform @ R_original
    
    orientation_transformed = np.array(rotation_matrix_to_euler(R_result))
    
    return np.concatenate([position_transformed, orientation_transformed])


def interpolate_transforms(T1: np.ndarray, T2: np.ndarray, 
                          alpha: float) -> np.ndarray:
    """
    在两个变换矩阵之间进行插值
    
    Args:
        T1: 起始变换矩阵
        T2: 结束变换矩阵
        alpha: 插值系数 (0-1)
    
    Returns:
        np.ndarray: 插值后的变换矩阵
    """
    # 插值平移
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # 球面线性插值旋转（简化使用线性插值）
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    
    # 将旋转矩阵转换为四元数
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    
    # 四元数球面线性插值
    q_interp = slerp_quaternions(q1, q2, alpha)
    
    # 转换回旋转矩阵
    R_interp = quaternion_to_rotation_matrix(q_interp)
    
    return compose_transform(t_interp, R_interp)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为四元数 [w, x, y, z]
    
    Args:
        R: 3x3旋转矩阵
    
    Returns:
        np.ndarray: 四元数 [w, x, y, z]
    """
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
    
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数 [w, x, y, z]
    
    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def slerp_quaternions(q1: np.ndarray, q2: np.ndarray, 
                     alpha: float) -> np.ndarray:
    """
    四元数球面线性插值
    
    Args:
        q1: 起始四元数
        q2: 结束四元数
        alpha: 插值系数
    
    Returns:
        np.ndarray: 插值后的四元数
    """
    # 归一化
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算点积
    dot = np.dot(q1, q2)
    
    # 如果点积为负，反转一个四元数
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q1 + alpha * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 计算插值角度
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return (s1 * q1) + (s2 * q2)


def get_sensor_extrinsics(position: List[float], 
                         orientation: List[float]) -> np.ndarray:
    """
    从位置和方向获取传感器外参矩阵
    
    Args:
        position: [x, y, z] 位置（米）
        orientation: [roll, pitch, yaw] 方向（度）
    
    Returns:
        np.ndarray: 4x4外参矩阵（传感器到车辆坐标系）
    """
    # 转换为弧度
    roll, pitch, yaw = np.radians(orientation)
    
    # 创建旋转矩阵
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # 创建变换矩阵
    T = compose_transform(np.array(position), R)
    
    return T


def create_camera_intrinsics(fx: float, fy: float, 
                             cx: float, cy: float) -> np.ndarray:
    """
    创建相机内参矩阵
    
    Args:
        fx: x方向焦距（像素）
        fy: y方向焦距（像素）
        cx: 主点x坐标
        cy: 主点y坐标
    
    Returns:
        np.ndarray: 3x3内参矩阵
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def project_points_to_image(points: np.ndarray, 
                            K: np.ndarray, 
                            T_cam: np.ndarray,
                            image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D点投影到图像平面
    
    Args:
        points: Nx3的3D点数组
        K: 3x3相机内参矩阵
        T_cam: 4x4相机外参矩阵
        image_size: (width, height) 图像尺寸
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (2D投影点, 有效掩码)
    """
    # 变换点到相机坐标系
    points_cam = transform_points(points, inverse_transform(T_cam))
    
    # 只保留相机前方的点
    valid_mask = points_cam[:, 2] > 0
    
    # 投影到图像平面
    points_2d = np.zeros((len(points), 2))
    
    if np.any(valid_mask):
        valid_points = points_cam[valid_mask]
        
        # 透视投影
        z = valid_points[:, 2:3]
        uv = (K @ (valid_points / z).T).T
        
        points_2d[valid_mask] = uv[:, :2]
        
        # 检查是否在图像范围内
        width, height = image_size
        in_image = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        )
        valid_mask = valid_mask & in_image
    
    return points_2d, valid_mask
