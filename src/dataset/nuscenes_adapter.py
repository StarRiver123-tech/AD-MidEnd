"""
nuScenes Dataset Adapter Module
================================

This module provides a complete adapter for the nuScenes dataset,
supporting camera images, LiDAR point clouds, radar data, and annotations.

Features:
- Data loading for multiple sensor modalities
- Format conversion to unified internal representation
- Coordinate system transformation (ego, global, sensor)
- Data preprocessing and augmentation
- Training and evaluation interfaces

Author: Autonomous Driving Team
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
from copy import deepcopy

# Optional imports with fallback
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("Warning: nuScenes devkit not available. Using mock implementation.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# Data Structures and Enums
# =============================================================================

class SensorType(Enum):
    """Sensor types in nuScenes."""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"


class CoordinateSystem(Enum):
    """Coordinate systems for transformations."""
    GLOBAL = "global"
    EGO = "ego"
    SENSOR = "sensor"


@dataclass
class Vector3D:
    """3D vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class Quaternion3D:
    """Quaternion representation for rotations."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion3D':
        return cls(w=arr[0], x=arr[1], y=arr[2], z=arr[3])


@dataclass
class BoundingBox3D:
    """3D bounding box representation."""
    center: Vector3D = field(default_factory=Vector3D)
    size: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    rotation: Quaternion3D = field(default_factory=Quaternion3D)
    velocity: Vector3D = field(default_factory=Vector3D)
    
    # Additional metadata
    category: str = ""
    instance_token: str = ""
    sample_token: str = ""
    confidence: float = 1.0
    
    def get_corners(self) -> np.ndarray:
        """Get 8 corners of the bounding box."""
        # Box dimensions
        l, w, h = self.size.x, self.size.y, self.size.z
        
        # 3D bounding box corners in box coordinate system
        corners = np.array([
            [l/2, w/2, h/2], [l/2, w/2, -h/2],
            [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ]).T
        
        # Rotate and translate
        rotation_matrix = self.rotation.to_rotation_matrix()
        corners = np.dot(rotation_matrix, corners)
        center_arr = self.center.to_array()
        corners = corners + center_arr.reshape(3, 1)
        
        return corners


@dataclass
class CameraImage:
    """Camera image data structure."""
    data: np.ndarray
    camera_name: str
    timestamp: int
    sensor2ego_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    ego2global_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    intrinsics: Optional[np.ndarray] = None
    distortion: Optional[np.ndarray] = None
    
    # Image metadata
    width: int = 0
    height: int = 0
    file_path: str = ""


@dataclass
class PointCloud:
    """Point cloud data structure (LiDAR or Radar)."""
    points: np.ndarray  # Shape: (N, C) where C >= 3 (x, y, z, ...)
    sensor_type: SensorType
    timestamp: int
    sensor2ego_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    ego2global_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # Point cloud metadata
    num_points: int = 0
    file_path: str = ""
    
    # For LiDAR: intensity, ring index, etc.
    # For Radar: velocity, rcs, etc.
    point_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.points is not None:
            self.num_points = len(self.points)


@dataclass
class ObjectAnnotation:
    """Object annotation with 3D bounding box."""
    bbox_3d: BoundingBox3D
    category: str
    instance_token: str
    sample_token: str
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    visibility: int = 0  # 0-4 visibility level
    num_lidar_points: int = 0
    num_radar_points: int = 0


@dataclass
class LaneAnnotation:
    """Lane annotation data structure."""
    token: str
    lane_type: str  # 'lane', 'lane_divider', 'road_divider', etc.
    geometry: np.ndarray  # Shape: (N, 3) for 3D points
    
    # Lane attributes
    from_edge_line: bool = False
    to_edge_line: bool = False
    left_lane_divider: bool = False
    right_lane_divider: bool = False
    
    # Connectivity
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class SampleData:
    """Complete sample data containing all sensor data and annotations."""
    token: str
    timestamp: int
    scene_token: str
    
    # Sensor data
    camera_images: Dict[str, CameraImage] = field(default_factory=dict)
    lidar_data: Optional[PointCloud] = None
    radar_data: Dict[str, PointCloud] = field(default_factory=dict)
    
    # Annotations
    object_annotations: List[ObjectAnnotation] = field(default_factory=list)
    lane_annotations: List[LaneAnnotation] = field(default_factory=list)
    
    # Ego vehicle information
    ego_pose: Dict[str, Any] = field(default_factory=dict)
    ego2global_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # Metadata
    prev_token: str = ""
    next_token: str = ""


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class NuScenesConfig:
    """Configuration for nuScenes dataset."""
    
    # Dataset paths
    data_root: str = "/data/nuscenes"
    version: str = "v1.0-trainval"
    
    # Sensor configuration
    use_camera: bool = True
    use_lidar: bool = True
    use_radar: bool = True
    
    # Camera settings
    camera_names: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ])
    image_size: Tuple[int, int] = (900, 1600)  # (H, W)
    normalize_image: bool = True
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # LiDAR settings
    lidar_name: str = "LIDAR_TOP"
    lidar_points_range: List[float] = field(default_factory=lambda: [-50, -50, -5, 50, 50, 3])
    max_lidar_points: int = 40000
    
    # Radar settings
    radar_names: List[str] = field(default_factory=lambda: [
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
    ])
    
    # Data augmentation
    enable_augmentation: bool = False
    flip_probability: float = 0.5
    rotation_range: float = 0.0  # degrees
    scale_range: Tuple[float, float] = (0.95, 1.05)
    
    # Coordinate system
    target_coordinate_system: CoordinateSystem = CoordinateSystem.EGO
    
    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NuScenesConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data_root': self.data_root,
            'version': self.version,
            'use_camera': self.use_camera,
            'use_lidar': self.use_lidar,
            'use_radar': self.use_radar,
            'camera_names': self.camera_names,
            'image_size': list(self.image_size),
            'normalize_image': self.normalize_image,
            'image_mean': self.image_mean,
            'image_std': self.image_std,
            'lidar_name': self.lidar_name,
            'lidar_points_range': self.lidar_points_range,
            'max_lidar_points': self.max_lidar_points,
            'radar_names': self.radar_names,
            'enable_augmentation': self.enable_augmentation,
            'flip_probability': self.flip_probability,
            'rotation_range': self.rotation_range,
            'scale_range': list(self.scale_range),
            'target_coordinate_system': self.target_coordinate_system.value,
            'cache_enabled': self.cache_enabled,
            'cache_dir': self.cache_dir,
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# =============================================================================
# Data Preprocessing Classes
# =============================================================================

class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self, config: NuScenesConfig):
        self.config = config
        self.target_size = config.image_size
        self.normalize = config.normalize_image
        self.mean = np.array(config.image_mean).reshape(1, 1, 3)
        self.std = np.array(config.image_std).reshape(1, 1, 3)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: resize and normalize."""
        # Resize image
        if image.shape[:2] != self.target_size:
            if CV2_AVAILABLE:
                image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            elif PIL_AVAILABLE:
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((self.target_size[1], self.target_size[0]))
                image = np.array(pil_img)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization if enabled
        if self.normalize:
            image = (image - self.mean) / self.std
        
        return image
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Reverse normalization for visualization."""
        if self.normalize:
            image = image * self.std + self.mean
        return np.clip(image * 255, 0, 255).astype(np.uint8)


class PointCloudPreprocessor:
    """Point cloud preprocessing utilities."""
    
    def __init__(self, config: NuScenesConfig):
        self.config = config
        self.points_range = config.lidar_points_range
    
    def filter_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points by range."""
        x_range = (self.points_range[0], self.points_range[3])
        y_range = (self.points_range[1], self.points_range[4])
        z_range = (self.points_range[2], self.points_range[5])
        
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        )
        return points[mask]
    
    def voxelize(self, points: np.ndarray, voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.2)) -> np.ndarray:
        """Simple voxelization of point cloud."""
        # Calculate voxel indices
        voxel_indices = np.floor(points[:, :3] / np.array(voxel_size)).astype(np.int32)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        
        # Average points in each voxel
        voxel_points = np.zeros((len(unique_voxels), points.shape[1]))
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxel_points[i] = np.mean(points[mask], axis=0)
        
        return voxel_points
    
    def preprocess(self, point_cloud: PointCloud) -> PointCloud:
        """Preprocess point cloud."""
        points = point_cloud.points.copy()
        
        # Filter by range
        points = self.filter_range(points)
        
        # Limit number of points
        if len(points) > self.config.max_lidar_points:
            indices = np.random.choice(len(points), self.config.max_lidar_points, replace=False)
            points = points[indices]
        
        # Create new point cloud with processed points
        processed_pc = deepcopy(point_cloud)
        processed_pc.points = points
        processed_pc.num_points = len(points)
        
        return processed_pc


# =============================================================================
# Data Augmentation Classes
# =============================================================================

class DataAugmentor:
    """Data augmentation for training."""
    
    def __init__(self, config: NuScenesConfig):
        self.config = config
        self.flip_prob = config.flip_probability
        self.rotation_range = np.radians(config.rotation_range)
        self.scale_range = config.scale_range
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image."""
        if not self.config.enable_augmentation:
            return image
        
        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            image = np.fliplr(image).copy()
        
        return image
    
    def augment_point_cloud(self, point_cloud: PointCloud, 
                          annotations: List[ObjectAnnotation]) -> Tuple[PointCloud, List[ObjectAnnotation]]:
        """Apply augmentation to point cloud and annotations."""
        if not self.config.enable_augmentation:
            return point_cloud, annotations
        
        points = point_cloud.points.copy()
        augmented_annotations = deepcopy(annotations)
        
        # Random rotation around z-axis
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
            
            # Rotate annotations
            for ann in augmented_annotations:
                center = ann.bbox_3d.center.to_array()
                center = np.dot(rotation_matrix, center)
                ann.bbox_3d.center = Vector3D.from_array(center)
                
                # Update rotation
                angle_quat = Quaternion3D()
                angle_quat.w = np.cos(angle / 2)
                angle_quat.z = np.sin(angle / 2)
                # Combine rotations (simplified)
        
        # Random scaling
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        points[:, :3] *= scale
        
        for ann in augmented_annotations:
            ann.bbox_3d.center.x *= scale
            ann.bbox_3d.center.y *= scale
            ann.bbox_3d.center.z *= scale
            ann.bbox_3d.size.x *= scale
            ann.bbox_3d.size.y *= scale
            ann.bbox_3d.size.z *= scale
        
        # Create augmented point cloud
        augmented_pc = deepcopy(point_cloud)
        augmented_pc.points = points
        
        return augmented_pc, augmented_annotations


# =============================================================================
# Coordinate Transformation Utilities
# =============================================================================

class CoordinateTransformer:
    """Handle coordinate system transformations."""
    
    @staticmethod
    def get_transform_matrix(translation: np.ndarray, 
                            rotation: np.ndarray) -> np.ndarray:
        """Create 4x4 transformation matrix from translation and rotation."""
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    
    @staticmethod
    def transform_points(points: np.ndarray, 
                        transform: np.ndarray) -> np.ndarray:
        """Transform points using 4x4 transformation matrix."""
        # Add homogeneous coordinate
        points_homo = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
        points_transformed = np.dot(transform, points_homo.T).T
        return points_transformed[:, :3]
    
    @staticmethod
    def sensor_to_ego(points: np.ndarray, 
                     sensor2ego_transform: np.ndarray) -> np.ndarray:
        """Transform points from sensor to ego vehicle coordinate system."""
        return CoordinateTransformer.transform_points(points, sensor2ego_transform)
    
    @staticmethod
    def ego_to_global(points: np.ndarray, 
                     ego2global_transform: np.ndarray) -> np.ndarray:
        """Transform points from ego to global coordinate system."""
        return CoordinateTransformer.transform_points(points, ego2global_transform)
    
    @staticmethod
    def global_to_go(points: np.ndarray, 
                    ego2global_transform: np.ndarray) -> np.ndarray:
        """Transform points from global to ego vehicle coordinate system."""
        inv_transform = np.linalg.inv(ego2global_transform)
        return CoordinateTransformer.transform_points(points, inv_transform)
    
    @staticmethod
    def transform_bbox(bbox: BoundingBox3D, 
                      transform: np.ndarray) -> BoundingBox3D:
        """Transform 3D bounding box."""
        # Transform center
        center_homo = np.array([bbox.center.x, bbox.center.y, bbox.center.z, 1.0])
        center_transformed = np.dot(transform, center_homo)
        
        # Create new bbox
        new_bbox = deepcopy(bbox)
        new_bbox.center = Vector3D(
            x=center_transformed[0],
            y=center_transformed[1],
            z=center_transformed[2]
        )
        
        # Transform rotation (simplified - should use quaternion multiplication)
        rotation_matrix = transform[:3, :3]
        original_rotation = bbox.rotation.to_rotation_matrix()
        new_rotation = np.dot(rotation_matrix, original_rotation)
        
        # Convert back to quaternion (simplified)
        # In practice, use proper quaternion conversion
        
        return new_bbox


# =============================================================================
# Main nuScenes Adapter Class
# =============================================================================

class NuScenesAdapter:
    """
    Main adapter class for nuScenes dataset.
    
    Provides unified interface for loading and processing nuScenes data.
    """
    
    # nuScenes category mapping
    CATEGORY_MAPPING = {
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.bus': 'bus',
        'vehicle.bicycle': 'bicycle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone',
        'static_object.bicycle_rack': 'bicycle_rack',
    }
    
    def __init__(self, config: Optional[NuScenesConfig] = None):
        """
        Initialize nuScenes adapter.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or NuScenesConfig()
        self.nusc = None
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.pc_preprocessor = PointCloudPreprocessor(self.config)
        self.augmentor = DataAugmentor(self.config)
        self.transformer = CoordinateTransformer()
        
        # Cache
        self._cache = {}
        self._map_cache = {}
        
        # Load dataset if available
        if NUSCENES_AVAILABLE:
            self._load_nuscenes()
        
        print(f"NuScenesAdapter initialized with config:")
        print(f"  - Data root: {self.config.data_root}")
        print(f"  - Version: {self.config.version}")
        print(f"  - Use camera: {self.config.use_camera}")
        print(f"  - Use LiDAR: {self.config.use_lidar}")
        print(f"  - Use radar: {self.config.use_radar}")
    
    def _load_nuscenes(self):
        """Load nuScenes dataset."""
        try:
            self.nusc = NuScenes(
                version=self.config.version,
                dataroot=self.config.data_root,
                verbose=True
            )
            print(f"Loaded nuScenes with {len(self.nusc.sample)} samples")
        except Exception as e:
            print(f"Failed to load nuScenes: {e}")
            self.nusc = None
    
    def _load_image(self, sample_data_token: str) -> CameraImage:
        """Load camera image from sample data token."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return self._create_mock_image()
        
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        # Load image
        file_path = os.path.join(self.config.data_root, sd_record['filename'])
        
        if CV2_AVAILABLE:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif PIL_AVAILABLE:
            image = np.array(Image.open(file_path))
        else:
            raise RuntimeError("Neither cv2 nor PIL available for image loading")
        
        # Get transforms
        sensor2ego = self.transformer.get_transform_matrix(
            np.array(cs_record['translation']),
            Quaternion(cs_record['rotation']).rotation_matrix
        )
        
        ego2global = self.transformer.get_transform_matrix(
            np.array(ep_record['translation']),
            Quaternion(ep_record['rotation']).rotation_matrix
        )
        
        # Get intrinsics
        intrinsics = np.array(cs_record['camera_intrinsic'])
        
        return CameraImage(
            data=image,
            camera_name=sd_record['channel'],
            timestamp=sd_record['timestamp'],
            sensor2ego_transform=sensor2ego,
            ego2global_transform=ego2global,
            intrinsics=intrinsics,
            width=image.shape[1],
            height=image.shape[0],
            file_path=file_path
        )
    
    def _load_lidar(self, sample_data_token: str) -> PointCloud:
        """Load LiDAR point cloud from sample data token."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return self._create_mock_lidar()
        
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        # Load point cloud
        file_path = os.path.join(self.config.data_root, sd_record['filename'])
        pc = LidarPointCloud.from_file(file_path)
        
        # Get transforms
        sensor2ego = self.transformer.get_transform_matrix(
            np.array(cs_record['translation']),
            Quaternion(cs_record['rotation']).rotation_matrix
        )
        
        ego2global = self.transformer.get_transform_matrix(
            np.array(ep_record['translation']),
            Quaternion(ep_record['rotation']).rotation_matrix
        )
        
        # Points are (x, y, z, intensity)
        points = pc.points.T
        
        return PointCloud(
            points=points,
            sensor_type=SensorType.LIDAR,
            timestamp=sd_record['timestamp'],
            sensor2ego_transform=sensor2ego,
            ego2global_transform=ego2global,
            file_path=file_path,
            point_attributes={
                'intensity': points[:, 3] if points.shape[1] > 3 else np.ones(len(points))
            }
        )
    
    def _load_radar(self, sample_data_token: str) -> PointCloud:
        """Load radar point cloud from sample data token."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return self._create_mock_radar()
        
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        # Load point cloud
        file_path = os.path.join(self.config.data_root, sd_record['filename'])
        pc = RadarPointCloud.from_file(file_path)
        
        # Get transforms
        sensor2ego = self.transformer.get_transform_matrix(
            np.array(cs_record['translation']),
            Quaternion(cs_record['rotation']).rotation_matrix
        )
        
        ego2global = self.transformer.get_transform_matrix(
            np.array(ep_record['translation']),
            Quaternion(ep_record['rotation']).rotation_matrix
        )
        
        # Points contain: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, 
        # is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms
        points = pc.points.T
        
        return PointCloud(
            points=points[:, :3],  # Use x, y, z only for main points
            sensor_type=SensorType.RADAR,
            timestamp=sd_record['timestamp'],
            sensor2ego_transform=sensor2ego,
            ego2global_transform=ego2global,
            file_path=file_path,
            point_attributes={
                'velocity_x': points[:, 6] if points.shape[1] > 6 else np.zeros(len(points)),
                'velocity_y': points[:, 7] if points.shape[1] > 7 else np.zeros(len(points)),
                'rcs': points[:, 5] if points.shape[1] > 5 else np.zeros(len(points)),
            }
        )
    
    def _load_annotations(self, sample_token: str) -> List[ObjectAnnotation]:
        """Load object annotations for a sample."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return self._create_mock_annotations()
        
        sample = self.nusc.get('sample', sample_token)
        annotations = []
        
        for ann_token in sample['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            
            # Get category
            category_name = ann_record['category_name']
            mapped_category = self.CATEGORY_MAPPING.get(category_name, 'unknown')
            
            # Create 3D bounding box
            # nuScenes size format is [width, length, height]
            # BoundingBox3D expects [length, width, height]
            bbox = BoundingBox3D(
                center=Vector3D(
                    x=ann_record['translation'][0],
                    y=ann_record['translation'][1],
                    z=ann_record['translation'][2]
                ),
                size=Vector3D(
                    x=ann_record['size'][1],
                    y=ann_record['size'][0],
                    z=ann_record['size'][2]
                ),
                rotation=Quaternion3D.from_array(
                    Quaternion(ann_record['rotation']).elements
                ),
                velocity=Vector3D(
                    x=ann_record.get('velocity', [0.0, 0.0])[0],
                    y=ann_record.get('velocity', [0.0, 0.0])[1],
                    z=0.0
                ),
                category=mapped_category,
                instance_token=ann_record['instance_token'],
                sample_token=sample_token
            )
            
            # Get attributes
            attributes = {}
            for attr_token in ann_record['attribute_tokens']:
                attr = self.nusc.get('attribute', attr_token)
                attributes[attr['name']] = True
            
            annotation = ObjectAnnotation(
                bbox_3d=bbox,
                category=mapped_category,
                instance_token=ann_record['instance_token'],
                sample_token=sample_token,
                attributes=attributes,
                visibility=ann_record['visibility_token'],
                num_lidar_points=ann_record['num_lidar_pts'],
                num_radar_points=ann_record['num_radar_pts']
            )
            
            annotations.append(annotation)
        
        return annotations
    
    def _load_lanes(self, sample_token: str) -> List[LaneAnnotation]:
        """Load lane annotations (from map data)."""
        lanes = []
        
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return lanes
        
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']
        
        # Get ego pose for this sample (use lidar pose as reference)
        lidar_token = sample['data'].get(self.config.lidar_name, '')
        if not lidar_token:
            return lanes
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        ego_x, ego_y = ego_pose['translation'][:2]
        
        # ------------------------------------------------------------------
        # 1) Try vector map (expansion JSON) first
        # ------------------------------------------------------------------
        try:
            if location not in self._map_cache:
                map_path = os.path.join(self.config.data_root, 'maps', 'expansion')
                if os.path.isdir(map_path):
                    self._map_cache[location] = NuScenesMap(
                        dataroot=self.config.data_root,
                        map_name=location
                    )
            if location in self._map_cache:
                nusc_map = self._map_cache[location]
                radius = 60.0
                line_layers = ['lane_divider', 'road_divider']
                polygon_layers = ['stop_line', 'ped_crossing']
                area_layers = ['drivable_area']
                records = nusc_map.get_records_in_radius(
                    ego_x, ego_y, radius,
                    line_layers + polygon_layers + area_layers + ['lane']
                )
                
                for layer_name in line_layers:
                    for token in records.get(layer_name, []):
                        line = nusc_map.extract_line(token)
                        if line is None:
                            continue
                        coords = np.array(line.coords)
                        coords_3d = np.zeros((len(coords), 3))
                        coords_3d[:, :2] = coords
                        lanes.append(LaneAnnotation(
                            token=token, lane_type=layer_name, geometry=coords_3d
                        ))
                
                for layer_name in polygon_layers + area_layers:
                    for token in records.get(layer_name, []):
                        polygon = nusc_map.extract_polygon(token)
                        if polygon is None:
                            continue
                        coords = np.array(polygon.exterior.coords)
                        coords_3d = np.zeros((len(coords), 3))
                        coords_3d[:, :2] = coords
                        lanes.append(LaneAnnotation(
                            token=token, lane_type=layer_name, geometry=coords_3d
                        ))
                
                lane_tokens = records.get('lane', [])
                if lane_tokens:
                    discretized = nusc_map.discretize_lanes(lane_tokens, resolution_meters=1.0)
                    for token, points in discretized.items():
                        if not points:
                            continue
                        coords = np.array(points)
                        lanes.append(LaneAnnotation(
                            token=token, lane_type='lane_centerline', geometry=coords
                        ))
                
                if lanes:
                    return lanes
        except Exception:
            pass
        
        # ------------------------------------------------------------------
        # 2) Fallback: extract drivable-area boundaries from bitmap mask (PNG)
        # ------------------------------------------------------------------
        try:
            import cv2
            map_record = next((m for m in self.nusc.map if log['token'] in m['log_tokens']), None)
            if map_record is None:
                return lanes
            
            mask = map_record['mask']
            mask_img = mask.mask()  # uint8 0/255
            H, W = mask_img.shape
            T = mask.transform_matrix
            sx, tx = T[0, 0], T[0, 3]
            sy, ty = T[1, 1], T[1, 3]
            
            radius = 60.0
            x_min, x_max = ego_x - radius, ego_x + radius
            y_min, y_max = ego_y - radius, ego_y + radius
            
            px_min = int(np.floor(sx * x_min + tx))
            px_max = int(np.ceil(sx * x_max + tx))
            py_min = int(np.floor(sy * y_min + ty))
            py_max = int(np.ceil(sy * y_max + ty))
            
            if px_min > px_max:
                px_min, px_max = px_max, px_min
            if py_min > py_max:
                py_min, py_max = py_max, py_min
            
            px_min = max(0, px_min)
            px_max = min(W, px_max)
            py_min = max(0, py_min)
            py_max = min(H, py_max)
            
            if px_min >= px_max or py_min >= py_max:
                return lanes
            
            patch = mask_img[py_min:py_max, px_min:px_max]
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                cnt = cnt.squeeze(1)
                if len(cnt) < 2:
                    continue
                # Downsample to avoid too many points
                step = max(1, len(cnt) // 100)
                cnt = cnt[::step]
                
                global_pts = np.zeros((len(cnt), 3))
                global_pts[:, 0] = (cnt[:, 0] + px_min - tx) / sx
                global_pts[:, 1] = (cnt[:, 1] + py_min - ty) / sy
                global_pts[:, 2] = 0.0
                
                lanes.append(LaneAnnotation(
                    token=f'mask_contour_{len(lanes)}',
                    lane_type='drivable_area',
                    geometry=global_pts
                ))
        except Exception:
            pass
        
        return lanes
    
    def _create_mock_image(self) -> CameraImage:
        """Create mock image for testing without nuScenes."""
        return CameraImage(
            data=np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8),
            camera_name="CAM_FRONT",
            timestamp=0,
            width=1600,
            height=900
        )
    
    def _create_mock_lidar(self) -> PointCloud:
        """Create mock LiDAR point cloud for testing."""
        num_points = 1000
        points = np.random.randn(num_points, 4) * 10
        points[:, 3] = np.random.rand(num_points)  # intensity
        return PointCloud(
            points=points,
            sensor_type=SensorType.LIDAR,
            timestamp=0,
            num_points=num_points
        )
    
    def _create_mock_radar(self) -> PointCloud:
        """Create mock radar point cloud for testing."""
        num_points = 100
        points = np.random.randn(num_points, 3) * 20
        return PointCloud(
            points=points,
            sensor_type=SensorType.RADAR,
            timestamp=0,
            num_points=num_points
        )
    
    def _create_mock_annotations(self) -> List[ObjectAnnotation]:
        """Create mock annotations for testing."""
        annotations = []
        for i in range(5):
            bbox = BoundingBox3D(
                center=Vector3D(x=np.random.randn()*10, y=np.random.randn()*10, z=0.5),
                size=Vector3D(x=4.0, y=1.8, z=1.5),
                category='car'
            )
            ann = ObjectAnnotation(
                bbox_3d=bbox,
                category='car',
                instance_token=f'instance_{i}',
                sample_token='sample_mock'
            )
            annotations.append(ann)
        return annotations
    
    def load_sample(self, sample_token: str, 
                   load_annotations: bool = True,
                   apply_augmentation: bool = False) -> SampleData:
        """
        Load complete sample data.
        
        Args:
            sample_token: nuScenes sample token
            load_annotations: Whether to load annotations
            apply_augmentation: Whether to apply data augmentation
        
        Returns:
            SampleData object containing all sensor data and annotations
        """
        # Check cache
        cache_key = f"{sample_token}_{load_annotations}_{apply_augmentation}_v1"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        if NUSCENES_AVAILABLE and self.nusc is not None:
            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            
            # Get ego pose
            lidar_token = sample['data'].get(self.config.lidar_name, '')
            if lidar_token:
                lidar_sd = self.nusc.get('sample_data', lidar_token)
                ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
                ego2global = self.transformer.get_transform_matrix(
                    np.array(ego_pose['translation']),
                    Quaternion(ego_pose['rotation']).rotation_matrix
                )
            else:
                ego2global = np.eye(4)
                ego_pose = {}
        else:
            sample = {'token': sample_token, 'timestamp': 0, 'scene_token': '',
                     'data': {}, 'anns': [], 'prev': '', 'next': ''}
            ego_pose = {}
            ego2global = np.eye(4)
        
        # Initialize sample data
        sample_data = SampleData(
            token=sample_token,
            timestamp=sample.get('timestamp', 0),
            scene_token=sample.get('scene_token', ''),
            ego_pose=ego_pose,
            ego2global_transform=ego2global,
            prev_token=sample.get('prev', ''),
            next_token=sample.get('next', '')
        )
        
        # Load camera images
        if self.config.use_camera:
            for cam_name in self.config.camera_names:
                if NUSCENES_AVAILABLE and self.nusc is not None:
                    cam_token = sample['data'].get(cam_name)
                    if cam_token:
                        image = self._load_image(cam_token)
                        image.data = self.image_preprocessor.preprocess(image.data)
                        sample_data.camera_images[cam_name] = image
                else:
                    sample_data.camera_images[cam_name] = self._create_mock_image()
        
        # Load LiDAR
        if self.config.use_lidar:
            if NUSCENES_AVAILABLE and self.nusc is not None:
                lidar_token = sample['data'].get(self.config.lidar_name)
                if lidar_token:
                    lidar_pc = self._load_lidar(lidar_token)
                    lidar_pc = self.pc_preprocessor.preprocess(lidar_pc)
                    sample_data.lidar_data = lidar_pc
            else:
                sample_data.lidar_data = self._create_mock_lidar()
        
        # Load radar
        if self.config.use_radar:
            for radar_name in self.config.radar_names:
                if NUSCENES_AVAILABLE and self.nusc is not None:
                    radar_token = sample['data'].get(radar_name)
                    if radar_token:
                        radar_pc = self._load_radar(radar_token)
                        sample_data.radar_data[radar_name] = radar_pc
                else:
                    sample_data.radar_data[radar_name] = self._create_mock_radar()
        
        # Load annotations
        if load_annotations:
            annotations = self._load_annotations(sample_token)
            
            # Apply augmentation if requested
            if apply_augmentation and sample_data.lidar_data is not None:
                sample_data.lidar_data, annotations = self.augmentor.augment_point_cloud(
                    sample_data.lidar_data, annotations
                )
            
            sample_data.object_annotations = annotations
            
            # Load lane annotations
            sample_data.lane_annotations = self._load_lanes(sample_token)
        
        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = sample_data
        
        return sample_data
    
    def get_sample_tokens(self, split: str = 'train') -> List[str]:
        """
        Get list of sample tokens for a given split.
        
        Args:
            split: Dataset split ('train', 'val', 'test', 'mini_train', 'mini_val')
        
        Returns:
            List of sample tokens
        """
        if not NUSCENES_AVAILABLE or self.nusc is None:
            # Return mock tokens for testing
            return [f'sample_{i}' for i in range(100)]
        
        # Get scene tokens for the split
        scene_tokens = self._get_scene_tokens_for_split(split)
        
        # Get all samples from these scenes
        sample_tokens = []
        for scene_token in scene_tokens:
            scene = self.nusc.get('scene', scene_token)
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
        
        return sample_tokens
    
    def _get_scene_tokens_for_split(self, split: str) -> List[str]:
        """Get scene tokens for a given split."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return []
        
        # Load split information
        split_file = os.path.join(
            self.config.data_root, 
            f'{split}.txt' if not split.startswith('mini') else f'{split.replace("mini_", "")}.txt'
        )
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                scene_names = [line.strip() for line in f]
            
            scene_tokens = []
            for scene in self.nusc.scene:
                if scene['name'] in scene_names:
                    scene_tokens.append(scene['token'])
            return scene_tokens
        
        # Default: return all scenes
        return [scene['token'] for scene in self.nusc.scene]
    
    def get_all_scenes(self) -> List[Dict[str, Any]]:
        """Get all scenes from the dataset."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return []
        return list(self.nusc.scene)
    
    def get_scene_sample_tokens(self, scene_token: str) -> List[str]:
        """Get all sample tokens for a given scene."""
        if not NUSCENES_AVAILABLE or self.nusc is None:
            return []
        
        scene = self.nusc.get('scene', scene_token)
        sample_tokens = []
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample_tokens.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
        
        return sample_tokens
    
    def load_sample_data(self, sample_token: str) -> Dict[str, Any]:
        """
        Load sample data and convert to internal dictionary format
        compatible with main.py expectations.
        """
        sample_data = self.load_sample(sample_token, load_annotations=True)
        
        result = {
            'token': sample_data.token,
            'timestamp': sample_data.timestamp,
            'scene_token': sample_data.scene_token,
            'ego_pose': sample_data.ego_pose,
            'ego2global': sample_data.ego2global_transform.tolist(),
        }
        
        # Camera data
        if sample_data.camera_images:
            result['cameras'] = {}
            for cam_name, cam_data in sample_data.camera_images.items():
                result['cameras'][cam_name] = {
                    'image': cam_data.data,
                    'intrinsics': cam_data.intrinsics.tolist() if cam_data.intrinsics is not None else None,
                    'sensor2ego': cam_data.sensor2ego_transform.tolist(),
                    'ego2global': cam_data.ego2global_transform.tolist(),
                }
        
        # LiDAR data
        if sample_data.lidar_data is not None:
            result['lidar'] = {
                'points': sample_data.lidar_data.points,
                'sensor2ego': sample_data.lidar_data.sensor2ego_transform.tolist(),
                'ego2global': sample_data.lidar_data.ego2global_transform.tolist(),
                'attributes': sample_data.lidar_data.point_attributes,
            }
        
        # Radar data
        if sample_data.radar_data:
            result['radars'] = {}
            for radar_name, radar_data in sample_data.radar_data.items():
                result['radars'][radar_name] = {
                    'points': radar_data.points,
                    'sensor2ego': radar_data.sensor2ego_transform.tolist(),
                    'ego2global': radar_data.ego2global_transform.tolist(),
                    'attributes': radar_data.point_attributes,
                }
        
        # Annotations
        if sample_data.object_annotations:
            result['annotations'] = []
            for ann in sample_data.object_annotations:
                ann_dict = {
                    'category': ann.category,
                    'instance_token': ann.instance_token,
                    'bbox_3d': {
                        'center': [ann.bbox_3d.center.x, ann.bbox_3d.center.y, ann.bbox_3d.center.z],
                        'size': [ann.bbox_3d.size.x, ann.bbox_3d.size.y, ann.bbox_3d.size.z],
                        'rotation': ann.bbox_3d.rotation.to_array().tolist(),
                        'velocity': [ann.bbox_3d.velocity.x, ann.bbox_3d.velocity.y, ann.bbox_3d.velocity.z],
                    },
                    'corners': ann.bbox_3d.get_corners().tolist(),
                    'attributes': ann.attributes,
                    'visibility': ann.visibility,
                    'num_lidar_points': ann.num_lidar_points,
                    'num_radar_points': ann.num_radar_points,
                }
                result['annotations'].append(ann_dict)
        
        # Lane annotations
        if sample_data.lane_annotations:
            result['lane_annotations'] = []
            for lane in sample_data.lane_annotations:
                result['lane_annotations'].append({
                    'token': lane.token,
                    'lane_type': lane.lane_type,
                    'geometry': lane.geometry.tolist(),
                })
        
        return result
    
    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return list(set(self.CATEGORY_MAPPING.values()))
    
    def get_num_categories(self) -> int:
        """Get number of categories."""
        return len(self.get_category_names())
    
    def convert_to_internal_format(self, sample_data: SampleData) -> Dict[str, Any]:
        """
        Convert SampleData to internal format dictionary.
        
        Args:
            sample_data: SampleData object
        
        Returns:
            Dictionary in internal format
        """
        result = {
            'token': sample_data.token,
            'timestamp': sample_data.timestamp,
            'scene_token': sample_data.scene_token,
            'ego2global': sample_data.ego2global_transform.tolist(),
        }
        
        # Camera data
        if sample_data.camera_images:
            result['camera_images'] = {}
            for cam_name, cam_data in sample_data.camera_images.items():
                result['camera_images'][cam_name] = {
                    'data': cam_data.data,
                    'intrinsics': cam_data.intrinsics.tolist() if cam_data.intrinsics is not None else None,
                    'sensor2ego': cam_data.sensor2ego_transform.tolist(),
                    'ego2global': cam_data.ego2global_transform.tolist(),
                }
        
        # LiDAR data
        if sample_data.lidar_data is not None:
            result['lidar'] = {
                'points': sample_data.lidar_data.points,
                'sensor2ego': sample_data.lidar_data.sensor2ego_transform.tolist(),
                'ego2global': sample_data.lidar_data.ego2global_transform.tolist(),
                'attributes': sample_data.lidar_data.point_attributes,
            }
        
        # Radar data
        if sample_data.radar_data:
            result['radar'] = {}
            for radar_name, radar_data in sample_data.radar_data.items():
                result['radar'][radar_name] = {
                    'points': radar_data.points,
                    'sensor2ego': radar_data.sensor2ego_transform.tolist(),
                    'ego2global': radar_data.ego2global_transform.tolist(),
                    'attributes': radar_data.point_attributes,
                }
        
        # Annotations
        if sample_data.object_annotations:
            result['annotations'] = []
            for ann in sample_data.object_annotations:
                ann_dict = {
                    'category': ann.category,
                    'instance_token': ann.instance_token,
                    'bbox_3d': {
                        'center': [ann.bbox_3d.center.x, ann.bbox_3d.center.y, ann.bbox_3d.center.z],
                        'size': [ann.bbox_3d.size.x, ann.bbox_3d.size.y, ann.bbox_3d.size.z],
                        'rotation': ann.bbox_3d.rotation.to_array().tolist(),
                        'velocity': [ann.bbox_3d.velocity.x, ann.bbox_3d.velocity.y, ann.bbox_3d.velocity.z],
                    },
                    'corners': ann.bbox_3d.get_corners().tolist(),
                    'attributes': ann.attributes,
                    'visibility': ann.visibility,
                    'num_lidar_points': ann.num_lidar_points,
                    'num_radar_points': ann.num_radar_points,
                }
                result['annotations'].append(ann_dict)
        
        return result


# =============================================================================
# PyTorch Dataset Interface
# =============================================================================

class NuScenesDataset:
    """
    PyTorch-compatible dataset interface for nuScenes.
    
    Usage:
        dataset = NuScenesDataset(config, split='train')
        for batch in dataset:
            # Process batch
    """
    
    def __init__(self, config: NuScenesConfig, split: str = 'train',
                 transform=None, target_transform=None):
        """
        Initialize dataset.
        
        Args:
            config: NuScenes configuration
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform for input data
            target_transform: Optional transform for targets
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Initialize adapter
        self.adapter = NuScenesAdapter(config)
        
        # Get sample tokens
        self.sample_tokens = self.adapter.get_sample_tokens(split)
        
        print(f"NuScenesDataset [{split}]: {len(self.sample_tokens)} samples")
    
    def __len__(self) -> int:
        return len(self.sample_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing sample data
        """
        sample_token = self.sample_tokens[idx]
        
        # Load sample with augmentation for training
        apply_augmentation = (self.split == 'train' and self.config.enable_augmentation)
        sample_data = self.adapter.load_sample(
            sample_token, 
            load_annotations=True,
            apply_augmentation=apply_augmentation
        )
        
        # Convert to internal format
        data = self.adapter.convert_to_internal_format(sample_data)
        
        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get a batch of samples."""
        return [self.__getitem__(i) for i in indices]


# =============================================================================
# Data Loader Utilities
# =============================================================================

def collate_nuscenes_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching nuScenes data.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Batched data dictionary
    """
    # This is a simplified collate function
    # In practice, you would handle variable-sized data properly
    
    collated = {
        'tokens': [b['token'] for b in batch],
        'timestamps': [b['timestamp'] for b in batch],
    }
    
    # Collate camera images
    if 'camera_images' in batch[0]:
        collated['camera_images'] = {}
        for cam_name in batch[0]['camera_images'].keys():
            collated['camera_images'][cam_name] = np.stack([
                b['camera_images'][cam_name]['data'] for b in batch
            ])
    
    # Collate LiDAR (variable size - need special handling)
    if 'lidar' in batch[0]:
        collated['lidar_points'] = [b['lidar']['points'] for b in batch]
    
    # Collate annotations (variable size)
    if 'annotations' in batch[0]:
        collated['annotations'] = [b['annotations'] for b in batch]
    
    return collated


class NuScenesDataLoader:
    """
    Data loader for nuScenes dataset.
    
    Usage:
        loader = NuScenesDataLoader(config, split='train', batch_size=4)
        for batch in loader:
            # Process batch
    """
    
    def __init__(self, config: NuScenesConfig, split: str = 'train',
                 batch_size: int = 1, shuffle: bool = True,
                 num_workers: int = 0, collate_fn=None):
        """
        Initialize data loader.
        
        Args:
            config: NuScenes configuration
            split: Dataset split
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            collate_fn: Custom collate function
        """
        self.config = config
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or collate_nuscenes_batch
        
        # Create dataset
        self.dataset = NuScenesDataset(config, split)
        
        # Create indices
        self.indices = list(range(len(self.dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Dict[str, Any]:
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Load batch
        batch = self.dataset.get_batch(batch_indices)
        
        # Collate
        return self.collate_fn(batch)
    
    def reset(self):
        """Reset the data loader."""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# Evaluation Interface
# =============================================================================

class NuScenesEvaluator:
    """
    Evaluation interface for nuScenes dataset.
    
    Supports evaluation metrics:
    - 3D object detection (mAP, NDS)
    - Tracking (AMOTA, AMOTP)
    - Prediction (minADE, minFDE)
    """
    
    def __init__(self, config: NuScenesConfig, split: str = 'val'):
        """
        Initialize evaluator.
        
        Args:
            config: NuScenes configuration
            split: Evaluation split
        """
        self.config = config
        self.split = split
        self.adapter = NuScenesAdapter(config)
        
        # Get ground truth data
        self.sample_tokens = self.adapter.get_sample_tokens(split)
        
        print(f"NuScenesEvaluator [{split}]: {len(self.sample_tokens)} samples")
    
    def evaluate_detection(self, predictions: List[Dict[str, Any]], 
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate 3D object detection.
        
        Args:
            predictions: List of prediction dictionaries
            iou_threshold: IoU threshold for matching
        
        Returns:
            Dictionary of evaluation metrics
        """
        # This is a simplified evaluation
        # In practice, use nuScenes evaluation API
        
        metrics = {
            'mAP': 0.0,
            'mATE': 0.0,  # Average Translation Error
            'mASE': 0.0,  # Average Scale Error
            'mAOE': 0.0,  # Average Orientation Error
            'mAVE': 0.0,  # Average Velocity Error
            'mAAE': 0.0,  # Average Attribute Error
            'NDS': 0.0,   # nuScenes Detection Score
        }
        
        # TODO: Implement full evaluation logic
        
        return metrics
    
    def evaluate_tracking(self, tracking_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate multi-object tracking.
        
        Args:
            tracking_results: List of tracking results
        
        Returns:
            Dictionary of tracking metrics
        """
        metrics = {
            'AMOTA': 0.0,  # Average Multi-Object Tracking Accuracy
            'AMOTP': 0.0,  # Average Multi-Object Tracking Precision
            'MOTA': 0.0,
            'MOTP': 0.0,
            'IDF1': 0.0,
        }
        
        # TODO: Implement tracking evaluation
        
        return metrics
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format predictions for nuScenes evaluation.
        
        Args:
            predictions: Raw predictions
        
        Returns:
            Formatted predictions
        """
        formatted = {
            'results': {},
            'meta': {
                'use_camera': self.config.use_camera,
                'use_lidar': self.config.use_lidar,
                'use_radar': self.config.use_radar,
            }
        }
        
        for pred in predictions:
            sample_token = pred['token']
            formatted['results'][sample_token] = []
            
            for det in pred.get('detections', []):
                formatted['results'][sample_token].append({
                    'sample_token': sample_token,
                    'translation': det['center'],
                    'size': det['size'],
                    'rotation': det['rotation'],
                    'velocity': det.get('velocity', [0, 0]),
                    'detection_name': det['category'],
                    'detection_score': det.get('score', 1.0),
                    'attribute_name': det.get('attribute', ''),
                })
        
        return formatted


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_config() -> NuScenesConfig:
    """Create default nuScenes configuration."""
    return NuScenesConfig()


def load_config(config_path: str) -> NuScenesConfig:
    """Load configuration from file."""
    return NuScenesConfig.from_yaml(config_path)


def save_config(config: NuScenesConfig, config_path: str):
    """Save configuration to file."""
    config.to_yaml(config_path)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("nuScenes Dataset Adapter - Example Usage")
    print("=" * 60)
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
        use_camera=True,
        use_lidar=True,
        use_radar=True,
    )
    
    # Initialize adapter
    adapter = NuScenesAdapter(config)
    
    # Get sample tokens
    sample_tokens = adapter.get_sample_tokens('mini_train')
    print(f"\nFound {len(sample_tokens)} samples")
    
    # Load a sample
    if sample_tokens:
        sample_token = sample_tokens[0]
        print(f"\nLoading sample: {sample_token}")
        
        sample_data = adapter.load_sample(sample_token)
        
        print(f"  Timestamp: {sample_data.timestamp}")
        print(f"  Cameras: {list(sample_data.camera_images.keys())}")
        print(f"  LiDAR points: {sample_data.lidar_data.num_points if sample_data.lidar_data else 0}")
        print(f"  Radar sensors: {list(sample_data.radar_data.keys())}")
        print(f"  Objects: {len(sample_data.object_annotations)}")
        
        # Convert to internal format
        internal_format = adapter.convert_to_internal_format(sample_data)
        print(f"\nInternal format keys: {list(internal_format.keys())}")
    
    # Create dataset
    print("\n" + "=" * 60)
    print("Creating Dataset")
    print("=" * 60)
    
    dataset = NuScenesDataset(config, split='mini_train')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
    
    # Create data loader
    print("\n" + "=" * 60)
    print("Creating DataLoader")
    print("=" * 60)
    
    loader = NuScenesDataLoader(config, split='mini_train', batch_size=2)
    print(f"Number of batches: {len(loader)}")
    
    # Iterate through one batch
    for batch in loader:
        print(f"Batch tokens: {batch['tokens']}")
        if 'camera_images' in batch:
            for cam_name, images in batch['camera_images'].items():
                print(f"  {cam_name} shape: {images.shape}")
        break
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
