# nuScenes Dataset Adapter

This module provides a complete adapter for the nuScenes autonomous driving dataset, supporting multiple sensor modalities and annotation types.

## Features

### 1. Data Loading
- **Camera Images**: Load and preprocess images from 6 cameras (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)
- **LiDAR Point Clouds**: Load and filter LiDAR point clouds from LIDAR_TOP
- **Radar Data**: Load radar point clouds from 5 radar sensors
- **Object Annotations**: Load 3D bounding box annotations with categories, attributes, and visibility
- **Lane Annotations**: Load lane and map data

### 2. Data Format Conversion
- Convert nuScenes format to unified internal representation
- Support multiple coordinate systems (sensor, ego vehicle, global)
- Unified timestamp handling

### 3. Data Preprocessing
- **Image Preprocessing**: Resize, normalize (ImageNet statistics)
- **Point Cloud Preprocessing**: Range filtering, voxelization, point limit
- **Coordinate Transformations**: Transform between sensor, ego, and global frames

### 4. Data Augmentation
- Random horizontal flip
- Random rotation (around z-axis)
- Random scaling
- Applied to both images and point clouds with synchronized annotations

### 5. Training & Evaluation Interfaces
- PyTorch-compatible Dataset interface
- Batch data loading with custom collate functions
- Evaluation metrics (mAP, NDS, etc.)

## Installation

```bash
# Install nuScenes devkit (optional but recommended)
pip install nuscenes-devkit

# Install other dependencies
pip install numpy pyyaml pillow opencv-python
```

## Quick Start

```python
from dataset.nuscenes_adapter import (
    NuScenesAdapter,
    NuScenesDataset,
    NuScenesDataLoader,
    NuScenesConfig,
)

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

# Load a sample
sample_tokens = adapter.get_sample_tokens('mini_train')
sample_data = adapter.load_sample(sample_tokens[0])

# Access data
print(f"Cameras: {list(sample_data.camera_images.keys())}")
print(f"LiDAR points: {sample_data.lidar_data.num_points}")
print(f"Objects: {len(sample_data.object_annotations)}")
```

## Configuration

See `config/dataset_config.yaml` for all available options:

```yaml
# Dataset paths
data_root: "/data/nuscenes"
version: "v1.0-trainval"

# Sensor configuration
use_camera: true
use_lidar: true
use_radar: true

# Camera settings
camera_names:
  - "CAM_FRONT"
  - "CAM_FRONT_LEFT"
  - "CAM_FRONT_RIGHT"
  - "CAM_BACK"
  - "CAM_BACK_LEFT"
  - "CAM_BACK_RIGHT"
image_size: [900, 1600]
normalize_image: true

# LiDAR settings
lidar_points_range: [-50, -50, -5, 50, 50, 3]
max_lidar_points: 40000

# Data augmentation
enable_augmentation: true
flip_probability: 0.5
rotation_range: 10.0
scale_range: [0.95, 1.05]
```

## Usage Examples

### 1. Basic Data Loading
```python
adapter = NuScenesAdapter(config)
sample_tokens = adapter.get_sample_tokens('mini_train')
sample_data = adapter.load_sample(sample_tokens[0])
```

### 2. Dataset Interface
```python
dataset = NuScenesDataset(config, split='train')
sample = dataset[0]
```

### 3. DataLoader for Training
```python
loader = NuScenesDataLoader(config, split='train', batch_size=4)
for batch in loader:
    # Process batch
    pass
```

### 4. Custom Preprocessing
```python
from dataset.nuscenes_adapter import ImagePreprocessor, PointCloudPreprocessor

image_preprocessor = ImagePreprocessor(config)
processed_image = image_preprocessor.preprocess(image)

pc_preprocessor = PointCloudPreprocessor(config)
processed_pc = pc_preprocessor.preprocess(point_cloud)
```

### 5. Data Augmentation
```python
from dataset.nuscenes_adapter import DataAugmentor

augmentor = DataAugmentor(config)
augmented_pc, augmented_anns = augmentor.augment_point_cloud(
    point_cloud, annotations
)
```

### 6. Coordinate Transformations
```python
from dataset.nuscenes_adapter import CoordinateTransformer

transformer = CoordinateTransformer()
points_ego = transformer.sensor_to_ego(points, sensor2ego_transform)
points_global = transformer.ego_to_global(points_ego, ego2global_transform)
```

### 7. Evaluation
```python
from dataset.nuscenes_adapter import NuScenesEvaluator

evaluator = NuScenesEvaluator(config, split='val')
metrics = evaluator.evaluate_detection(predictions)
```

## Data Structures

### SampleData
Complete sample containing all sensor data and annotations:
```python
@dataclass
class SampleData:
    token: str
    timestamp: int
    scene_token: str
    camera_images: Dict[str, CameraImage]
    lidar_data: PointCloud
    radar_data: Dict[str, PointCloud]
    object_annotations: List[ObjectAnnotation]
    lane_annotations: List[LaneAnnotation]
    ego_pose: Dict[str, Any]
    ego2global_transform: np.ndarray
```

### BoundingBox3D
3D bounding box representation:
```python
@dataclass
class BoundingBox3D:
    center: Vector3D
    size: Vector3D
    rotation: Quaternion3D
    velocity: Vector3D
    category: str
```

### PointCloud
Point cloud data structure:
```python
@dataclass
class PointCloud:
    points: np.ndarray  # (N, C) where C >= 3
    sensor_type: SensorType
    timestamp: int
    sensor2ego_transform: np.ndarray
    ego2global_transform: np.ndarray
    point_attributes: Dict[str, np.ndarray]
```

## Category Mapping

nuScenes categories are mapped to simplified names:

| nuScenes Category | Mapped Name |
|------------------|-------------|
| vehicle.car | car |
| vehicle.truck | truck |
| vehicle.bus | bus |
| vehicle.bicycle | bicycle |
| vehicle.motorcycle | motorcycle |
| human.pedestrian.adult | pedestrian |
| human.pedestrian.child | pedestrian |
| movable_object.barrier | barrier |
| movable_object.trafficcone | traffic_cone |

## Testing Without nuScenes

The module includes mock implementations for testing without the actual nuScenes dataset:

```python
# Will use mock data if nuScenes is not available
config = NuScenesConfig()
adapter = NuScenesAdapter(config)  # Works without nuScenes installed
```

## API Reference

### NuScenesAdapter
Main adapter class for loading nuScenes data.

**Methods:**
- `load_sample(sample_token, load_annotations=True, apply_augmentation=False)`: Load a complete sample
- `get_sample_tokens(split)`: Get list of sample tokens for a split
- `convert_to_internal_format(sample_data)`: Convert to internal format

### NuScenesDataset
PyTorch-compatible dataset interface.

**Methods:**
- `__getitem__(idx)`: Get sample by index
- `__len__()`: Get dataset size

### NuScenesDataLoader
Batch data loading.

**Methods:**
- `__iter__()`: Iterate through batches
- `__len__()`: Get number of batches
- `reset()`: Reset the data loader

## License

This module is part of the autonomous driving project.
