"""
可视化工具模块
提供传感器数据可视化功能
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def visualize_pointcloud(points: np.ndarray, 
                        intensities: Optional[np.ndarray] = None,
                        title: str = "Point Cloud") -> Dict[str, Any]:
    """
    创建点云可视化信息
    
    Args:
        points: Nx3点云数组
        intensities: 强度值数组
        title: 标题
    
    Returns:
        Dict: 可视化信息字典
    """
    if len(points) == 0:
        return {
            'title': title,
            'num_points': 0,
            'info': 'Empty point cloud'
        }
    
    # 计算统计信息
    info = {
        'title': title,
        'num_points': len(points),
        'bounds': {
            'x': [float(np.min(points[:, 0])), float(np.max(points[:, 0]))],
            'y': [float(np.min(points[:, 1])), float(np.max(points[:, 1]))],
            'z': [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
        },
        'center': [
            float(np.mean(points[:, 0])),
            float(np.mean(points[:, 1])),
            float(np.mean(points[:, 2]))
        ],
    }
    
    # 距离分布
    distances = np.linalg.norm(points[:, :3], axis=1)
    info['distance_stats'] = {
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances))
    }
    
    # 高度分布
    info['height_stats'] = {
        'min': float(np.min(points[:, 2])),
        'max': float(np.max(points[:, 2])),
        'mean': float(np.mean(points[:, 2]))
    }
    
    # 强度信息
    if intensities is not None and len(intensities) > 0:
        info['intensity_stats'] = {
            'min': float(np.min(intensities)),
            'max': float(np.max(intensities)),
            'mean': float(np.mean(intensities))
        }
    
    return info


def visualize_radar_targets(targets: List[Any], 
                           title: str = "Radar Targets") -> Dict[str, Any]:
    """
    创建雷达目标可视化信息
    
    Args:
        targets: 雷达目标列表
        title: 标题
    
    Returns:
        Dict: 可视化信息字典
    """
    if not targets:
        return {
            'title': title,
            'num_targets': 0,
            'info': 'No targets detected'
        }
    
    # 提取目标属性
    ranges = [t.range for t in targets]
    azimuths = [np.degrees(t.azimuth) for t in targets]
    velocities = [t.velocity for t in targets]
    rcs_values = [t.rcs for t in targets]
    
    info = {
        'title': title,
        'num_targets': len(targets),
        'targets': []
    }
    
    # 每个目标的信息
    for i, target in enumerate(targets):
        target_info = {
            'id': target.id,
            'range': float(target.range),
            'azimuth': float(np.degrees(target.azimuth)),
            'elevation': float(np.degrees(target.elevation)) if hasattr(target, 'elevation') else 0.0,
            'velocity': float(target.velocity),
            'rcs': float(target.rcs),
            'snr': float(target.snr) if hasattr(target, 'snr') else 0.0
        }
        info['targets'].append(target_info)
    
    # 统计信息
    info['range_stats'] = {
        'min': float(np.min(ranges)),
        'max': float(np.max(ranges)),
        'mean': float(np.mean(ranges))
    }
    
    info['velocity_stats'] = {
        'min': float(np.min(velocities)),
        'max': float(np.max(velocities)),
        'mean': float(np.mean(velocities))
    }
    
    # 按距离排序的目标
    sorted_targets = sorted(info['targets'], key=lambda x: x['range'])
    info['closest_target'] = sorted_targets[0] if sorted_targets else None
    
    return info


def create_sensor_status_display(sensor_status: Dict[str, Any]) -> str:
    """
    创建传感器状态显示字符串
    
    Args:
        sensor_status: 传感器状态字典
    
    Returns:
        str: 格式化状态字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SENSOR STATUS")
    lines.append("=" * 60)
    
    # 运行状态
    running = sensor_status.get('running', False)
    uptime = sensor_status.get('uptime', 0.0)
    lines.append(f"System Running: {'YES' if running else 'NO'}")
    lines.append(f"Uptime: {uptime:.2f}s")
    lines.append("")
    
    # 摄像头状态
    cameras = sensor_status.get('cameras', {})
    lines.append(f"Cameras ({len(cameras)}):")
    for name, state in cameras.items():
        status_icon = "✓" if state == "running" else "✗"
        lines.append(f"  {status_icon} {name}: {state}")
    lines.append("")
    
    # LiDAR状态
    lidars = sensor_status.get('lidars', {})
    lines.append(f"LiDARs ({len(lidars)}):")
    for name, state in lidars.items():
        status_icon = "✓" if state == "running" else "✗"
        lines.append(f"  {status_icon} {name}: {state}")
    lines.append("")
    
    # 雷达状态
    radars = sensor_status.get('radars', {})
    lines.append(f"Radars ({len(radars)}):")
    for name, state in radars.items():
        status_icon = "✓" if state == "running" else "✗"
        lines.append(f"  {status_icon} {name}: {state}")
    lines.append("")
    
    # 超声波状态
    ultrasonics = sensor_status.get('ultrasonics', {})
    lines.append(f"Ultrasonics ({len(ultrasonics)}):")
    for name, state in ultrasonics.items():
        status_icon = "✓" if state == "running" else "✗"
        lines.append(f"  {status_icon} {name}: {state}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def create_data_summary(data_dict: Dict[str, Any]) -> str:
    """
    创建数据汇总字符串
    
    Args:
        data_dict: 传感器数据字典
    
    Returns:
        str: 格式化汇总字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total sensors: {len(data_dict)}")
    lines.append("")
    
    for name, data in data_dict.items():
        lines.append(f"{name}:")
        
        # 根据数据类型显示不同信息
        if hasattr(data, 'sensor_type'):
            lines.append(f"  Type: {data.sensor_type.value}")
        
        if hasattr(data, 'timestamp'):
            lines.append(f"  Timestamp: {data.timestamp:.3f}")
        
        if hasattr(data, 'frame_id'):
            lines.append(f"  Frame ID: {data.frame_id}")
        
        # 图像数据
        if hasattr(data, 'width') and hasattr(data, 'height'):
            lines.append(f"  Resolution: {data.width}x{data.height}")
        
        # 点云数据
        if hasattr(data, 'num_points'):
            lines.append(f"  Points: {data.num_points}")
        
        # 雷达数据
        if hasattr(data, 'num_targets'):
            lines.append(f"  Targets: {data.num_targets}")
        
        # 超声波数据
        if hasattr(data, 'distance'):
            lines.append(f"  Distance: {data.distance:.2f}m")
        
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_transform_matrix(T: np.ndarray, 
                           name: str = "Transform") -> str:
    """
    格式化变换矩阵显示
    
    Args:
        T: 4x4变换矩阵
        name: 矩阵名称
    
    Returns:
        str: 格式化字符串
    """
    lines = []
    lines.append(f"{name} Matrix:")
    lines.append("-" * 40)
    
    for row in T:
        row_str = "  [" + ", ".join([f"{x:8.4f}" for x in row]) + "]"
        lines.append(row_str)
    
    # 提取位置和欧拉角
    position = T[:3, 3]
    R = T[:3, :3]
    
    from .transformations import rotation_matrix_to_euler
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    
    lines.append("")
    lines.append(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
    lines.append(f"  Rotation: roll={np.degrees(roll):.2f}°, "
                f"pitch={np.degrees(pitch):.2f}°, "
                f"yaw={np.degrees(yaw):.2f}°")
    
    return "\n".join(lines)


def create_sync_statistics_display(stats: Dict[str, Any]) -> str:
    """
    创建同步统计信息显示
    
    Args:
        stats: 同步统计字典
    
    Returns:
        str: 格式化字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SYNC STATISTICS")
    lines.append("=" * 60)
    
    lines.append(f"Total Frames: {stats.get('frame_count', 0)}")
    lines.append(f"Sync Success: {stats.get('sync_success', 0)}")
    lines.append(f"Sync Fail: {stats.get('sync_fail', 0)}")
    lines.append(f"Success Rate: {stats.get('success_rate', 0.0):.2%}")
    lines.append("")
    
    buffer_sizes = stats.get('buffer_sizes', {})
    if buffer_sizes:
        lines.append("Buffer Sizes:")
        for name, size in buffer_sizes.items():
            lines.append(f"  {name}: {size}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def create_obstacle_map_display(ultrasonic_array, 
                                threshold: float = 1.0) -> str:
    """
    创建障碍物地图显示
    
    Args:
        ultrasonic_array: 超声波阵列
        threshold: 距离阈值
    
    Returns:
        str: 格式化字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("OBSTACLE MAP")
    lines.append("=" * 60)
    lines.append(f"Threshold: {threshold}m")
    lines.append("")
    
    # 获取障碍物信息
    front_obstacles = ultrasonic_array.get_front_obstacles(threshold)
    rear_obstacles = ultrasonic_array.get_rear_obstacles(threshold)
    min_sensor, min_dist = ultrasonic_array.get_min_distance()
    
    # 前部障碍物
    lines.append("Front Obstacles:")
    if front_obstacles:
        for obs in front_obstacles:
            us = ultrasonic_array.get_ultrasonic(obs)
            if us:
                data = us.get_latest_data()
                if data:
                    lines.append(f"  ⚠ {obs}: {data.distance:.2f}m")
    else:
        lines.append("  ✓ Clear")
    
    lines.append("")
    
    # 后部障碍物
    lines.append("Rear Obstacles:")
    if rear_obstacles:
        for obs in rear_obstacles:
            us = ultrasonic_array.get_ultrasonic(obs)
            if us:
                data = us.get_latest_data()
                if data:
                    lines.append(f"  ⚠ {obs}: {data.distance:.2f}m")
    else:
        lines.append("  ✓ Clear")
    
    lines.append("")
    
    # 最小距离
    if min_sensor:
        lines.append(f"Closest Obstacle: {min_sensor} at {min_dist:.2f}m")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
