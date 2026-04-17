"""
nuScenes Adapter Usage Examples
==============================

This file provides comprehensive examples of using the nuScenes adapter module.

Examples covered:
1. Basic data loading
2. Configuration setup
3. Training data loading
4. Evaluation data loading
5. Custom preprocessing
6. Data augmentation
7. Visualization
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset.nuscenes_adapter import (
    NuScenesAdapter,
    NuScenesDataset,
    NuScenesDataLoader,
    NuScenesEvaluator,
    NuScenesConfig,
    CoordinateSystem,
    SensorType,
    create_default_config,
    load_config,
    save_config,
)


def example_1_basic_loading():
    """
    Example 1: Basic Data Loading
    =============================
    
    Demonstrates how to:
    - Create a configuration
    - Initialize the adapter
    - Load a single sample
    - Access sensor data and annotations
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Data Loading")
    print("=" * 70)
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",  # Use mini version for quick testing
        use_camera=True,
        use_lidar=True,
        use_radar=True,
    )
    
    # Initialize adapter
    adapter = NuScenesAdapter(config)
    
    # Get sample tokens
    sample_tokens = adapter.get_sample_tokens('mini_train')
    print(f"Found {len(sample_tokens)} samples in mini_train split")
    
    if not sample_tokens:
        print("No samples found. Make sure nuScenes data is available.")
        return
    
    # Load first sample
    sample_token = sample_tokens[0]
    print(f"\nLoading sample: {sample_token}")
    
    sample_data = adapter.load_sample(sample_token)
    
    # Print sample information
    print(f"\nSample Information:")
    print(f"  Token: {sample_data.token}")
    print(f"  Timestamp: {sample_data.timestamp}")
    print(f"  Scene Token: {sample_data.scene_token}")
    
    # Camera data
    print(f"\nCamera Data:")
    for cam_name, cam_data in sample_data.camera_images.items():
        print(f"  {cam_name}:")
        print(f"    Shape: {cam_data.data.shape}")
        print(f"    Timestamp: {cam_data.timestamp}")
        print(f"    Intrinsics available: {cam_data.intrinsics is not None}")
    
    # LiDAR data
    if sample_data.lidar_data:
        print(f"\nLiDAR Data:")
        print(f"  Points: {sample_data.lidar_data.num_points}")
        print(f"  Point attributes: {list(sample_data.lidar_data.point_attributes.keys())}")
    
    # Radar data
    print(f"\nRadar Data:")
    for radar_name, radar_data in sample_data.radar_data.items():
        print(f"  {radar_name}: {radar_data.num_points} points")
    
    # Annotations
    print(f"\nObject Annotations: {len(sample_data.object_annotations)}")
    for i, ann in enumerate(sample_data.object_annotations[:5]):  # Show first 5
        print(f"  {i+1}. {ann.category}: "
              f"center=({ann.bbox_3d.center.x:.2f}, {ann.bbox_3d.center.y:.2f}, {ann.bbox_3d.center.z:.2f}), "
              f"size=({ann.bbox_3d.size.x:.2f}, {ann.bbox_3d.size.y:.2f}, {ann.bbox_3d.size.z:.2f})")
    
    return adapter, sample_data


def example_2_configuration():
    """
    Example 2: Configuration Management
    ===================================
    
    Demonstrates how to:
    - Create default configuration
    - Load configuration from YAML
    - Save configuration to YAML
    - Modify configuration programmatically
    """
    print("\n" + "=" * 70)
    print("Example 2: Configuration Management")
    print("=" * 70)
    
    # Create default configuration
    config = create_default_config()
    print("Default configuration created:")
    print(f"  Data root: {config.data_root}")
    print(f"  Version: {config.version}")
    print(f"  Use camera: {config.use_camera}")
    print(f"  Use LiDAR: {config.use_lidar}")
    print(f"  Use radar: {config.use_radar}")
    
    # Modify configuration
    config.data_root = "/custom/path/to/nuscenes"
    config.version = "v1.0-trainval"
    config.image_size = (600, 1067)  # Smaller image size
    config.max_lidar_points = 20000
    
    print(f"\nModified configuration:")
    print(f"  Data root: {config.data_root}")
    print(f"  Image size: {config.image_size}")
    print(f"  Max LiDAR points: {config.max_lidar_points}")
    
    # Save configuration
    config_path = "/tmp/nuscenes_config.yaml"
    save_config(config, config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Load configuration
    loaded_config = load_config(config_path)
    print(f"\nLoaded configuration:")
    print(f"  Data root: {loaded_config.data_root}")
    print(f"  Version: {loaded_config.version}")
    
    return config


def example_3_dataset_interface():
    """
    Example 3: Dataset Interface
    ============================
    
    Demonstrates how to:
    - Create a dataset
    - Access samples by index
    - Iterate through dataset
    """
    print("\n" + "=" * 70)
    print("Example 3: Dataset Interface")
    print("=" * 70)
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
        use_camera=True,
        use_lidar=True,
    )
    
    # Create dataset
    dataset = NuScenesDataset(config, split='mini_train')
    print(f"Dataset size: {len(dataset)}")
    
    # Access sample by index
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample keys: {list(sample.keys())}")
        
        # Show camera images
        if 'camera_images' in sample:
            print(f"\nCamera images:")
            for cam_name, cam_data in sample['camera_images'].items():
                print(f"  {cam_name} shape: {cam_data['data'].shape}")
        
        # Show LiDAR data
        if 'lidar' in sample:
            print(f"\nLiDAR points shape: {sample['lidar']['points'].shape}")
        
        # Show annotations
        if 'annotations' in sample:
            print(f"\nNumber of annotations: {len(sample['annotations'])}")
    
    # Iterate through dataset (first 3 samples)
    print(f"\nIterating through first 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: token={sample['token']}, "
              f"annotations={len(sample.get('annotations', []))}")
    
    return dataset


def example_4_dataloader():
    """
    Example 4: DataLoader for Training
    ==================================
    
    Demonstrates how to:
    - Create a data loader
    - Iterate through batches
    - Handle batched data
    """
    print("\n" + "=" * 70)
    print("Example 4: DataLoader for Training")
    print("=" * 70)
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
        use_camera=True,
        use_lidar=True,
        image_size=(450, 800),  # Smaller for faster loading
        max_lidar_points=10000,
    )
    
    # Create data loader
    loader = NuScenesDataLoader(
        config,
        split='mini_train',
        batch_size=2,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"Number of batches: {len(loader)}")
    
    # Iterate through batches
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Tokens: {batch['tokens']}")
        print(f"  Timestamps: {batch['timestamps']}")
        
        # Camera images
        if 'camera_images' in batch:
            for cam_name, images in batch['camera_images'].items():
                print(f"  {cam_name} batch shape: {images.shape}")
        
        # LiDAR (variable size)
        if 'lidar_points' in batch:
            print(f"  LiDAR points per sample: {[p.shape for p in batch['lidar_points']]}")
        
        # Annotations (variable size)
        if 'annotations' in batch:
            print(f"  Annotations per sample: {[len(a) for a in batch['annotations']]}")
        
        # Only show first 2 batches
        if batch_idx >= 1:
            break
    
    return loader


def example_5_preprocessing():
    """
    Example 5: Custom Preprocessing
    ===============================
    
    Demonstrates how to:
    - Apply custom image preprocessing
    - Apply custom point cloud preprocessing
    - Filter point clouds by range
    """
    print("\n" + "=" * 70)
    print("Example 5: Custom Preprocessing")
    print("=" * 70)
    
    from dataset.nuscenes_adapter import (
        ImagePreprocessor,
        PointCloudPreprocessor,
        PointCloud,
        SensorType
    )
    
    # Create configuration
    config = NuScenesConfig(
        image_size=(600, 1067),
        normalize_image=True,
        lidar_points_range=[-30, -30, -3, 30, 30, 2],
        max_lidar_points=15000,
    )
    
    # Image preprocessing
    print("\nImage Preprocessing:")
    image_preprocessor = ImagePreprocessor(config)
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"  Original image shape: {sample_image.shape}")
    
    processed_image = image_preprocessor.preprocess(sample_image)
    print(f"  Processed image shape: {processed_image.shape}")
    print(f"  Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # Denormalize for visualization
    denormalized = image_preprocessor.denormalize(processed_image)
    print(f"  Denormalized image shape: {denormalized.shape}")
    
    # Point cloud preprocessing
    print("\nPoint Cloud Preprocessing:")
    pc_preprocessor = PointCloudPreprocessor(config)
    
    # Create sample point cloud
    num_points = 50000
    points = np.random.randn(num_points, 4) * 40
    points[:, 3] = np.random.rand(num_points)  # intensity
    
    point_cloud = PointCloud(
        points=points,
        sensor_type=SensorType.LIDAR,
        timestamp=0,
        num_points=num_points
    )
    
    print(f"  Original points: {point_cloud.num_points}")
    
    # Filter by range
    filtered_points = pc_preprocessor.filter_range(points)
    print(f"  After range filtering: {len(filtered_points)}")
    
    # Full preprocessing
    processed_pc = pc_preprocessor.preprocess(point_cloud)
    print(f"  After full preprocessing: {processed_pc.num_points}")
    
    # Voxelization
    voxelized = pc_preprocessor.voxelize(processed_pc.points, voxel_size=(0.5, 0.5, 0.5))
    print(f"  After voxelization: {len(voxelized)}")


def example_6_augmentation():
    """
    Example 6: Data Augmentation
    ============================
    
    Demonstrates how to:
    - Enable data augmentation
    - Apply augmentation to images
    - Apply augmentation to point clouds and annotations
    """
    print("\n" + "=" * 70)
    print("Example 6: Data Augmentation")
    print("=" * 70)
    
    from dataset.nuscenes_adapter import (
        DataAugmentor,
        BoundingBox3D,
        Vector3D,
        Quaternion3D,
        ObjectAnnotation,
        PointCloud,
        SensorType
    )
    
    # Create configuration with augmentation enabled
    config = NuScenesConfig(
        enable_augmentation=True,
        flip_probability=0.5,
        rotation_range=10.0,  # degrees
        scale_range=(0.9, 1.1),
    )
    
    augmentor = DataAugmentor(config)
    
    # Image augmentation
    print("\nImage Augmentation:")
    sample_image = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
    print(f"  Original image shape: {sample_image.shape}")
    
    augmented_image = augmentor.augment_image(sample_image)
    print(f"  Augmented image shape: {augmented_image.shape}")
    
    # Point cloud and annotation augmentation
    print("\nPoint Cloud & Annotation Augmentation:")
    
    # Create sample point cloud
    points = np.random.randn(10000, 4) * 20
    point_cloud = PointCloud(
        points=points,
        sensor_type=SensorType.LIDAR,
        timestamp=0,
        num_points=10000
    )
    
    # Create sample annotations
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
            sample_token='sample_test'
        )
        annotations.append(ann)
    
    print(f"  Original points: {point_cloud.num_points}")
    print(f"  Original annotations: {len(annotations)}")
    print(f"  Original bbox centers: "
          f"[({annotations[0].bbox_3d.center.x:.2f}, {annotations[0].bbox_3d.center.y:.2f}), ...]")
    
    # Apply augmentation
    augmented_pc, augmented_anns = augmentor.augment_point_cloud(point_cloud, annotations)
    
    print(f"\n  Augmented points: {augmented_pc.num_points}")
    print(f"  Augmented annotations: {len(augmented_anns)}")
    print(f"  Augmented bbox centers: "
          f"[({augmented_anns[0].bbox_3d.center.x:.2f}, {augmented_anns[0].bbox_3d.center.y:.2f}), ...]")


def example_7_evaluation():
    """
    Example 7: Evaluation Interface
    ===============================
    
    Demonstrates how to:
    - Create an evaluator
    - Format predictions
    - Evaluate detection results
    """
    print("\n" + "=" * 70)
    print("Example 7: Evaluation Interface")
    print("=" * 70)
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
    )
    
    # Create evaluator
    evaluator = NuScenesEvaluator(config, split='mini_val')
    print(f"Evaluator created for {len(evaluator.sample_tokens)} samples")
    
    # Create mock predictions
    print("\nCreating mock predictions...")
    predictions = []
    for sample_token in evaluator.sample_tokens[:5]:  # First 5 samples
        pred = {
            'token': sample_token,
            'detections': [
                {
                    'category': 'car',
                    'center': [10.0, 5.0, 0.5],
                    'size': [4.0, 1.8, 1.5],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'velocity': [2.0, 0.0],
                    'score': 0.9,
                },
                {
                    'category': 'pedestrian',
                    'center': [5.0, 3.0, 0.8],
                    'size': [0.8, 0.8, 1.7],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'velocity': [0.5, 0.0],
                    'score': 0.85,
                }
            ]
        }
        predictions.append(pred)
    
    # Format predictions
    formatted = evaluator.format_predictions(predictions)
    print(f"\nFormatted predictions:")
    print(f"  Number of samples: {len(formatted['results'])}")
    for token, dets in list(formatted['results'].items())[:2]:
        print(f"  {token}: {len(dets)} detections")
    
    # Evaluate (mock)
    print("\nEvaluation metrics (mock):")
    metrics = evaluator.evaluate_detection(predictions)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def example_8_custom_transform():
    """
    Example 8: Custom Transform Functions
    =====================================
    
    Demonstrates how to:
    - Create custom transform functions
    - Apply transforms to dataset
    """
    print("\n" + "=" * 70)
    print("Example 8: Custom Transform Functions")
    print("=" * 70)
    
    # Define custom transform
    def custom_transform(data):
        """Custom transform that adds a 'processed' flag and normalizes LiDAR."""
        data['processed'] = True
        
        # Normalize LiDAR points
        if 'lidar' in data and data['lidar'] is not None:
            points = data['lidar']['points']
            # Center points
            mean = points[:, :3].mean(axis=0)
            points[:, :3] -= mean
            data['lidar']['points_normalized'] = points
            data['lidar']['mean'] = mean.tolist()
        
        return data
    
    # Create configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
        use_lidar=True,
    )
    
    # Create dataset with custom transform
    dataset = NuScenesDataset(
        config,
        split='mini_train',
        transform=custom_transform
    )
    
    print(f"Dataset with custom transform created: {len(dataset)} samples")
    
    # Load a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys after transform: {list(sample.keys())}")
        print(f"  processed flag: {sample.get('processed', False)}")
        
        if 'lidar' in sample and 'points_normalized' in sample['lidar']:
            print(f"  Original points shape: {sample['lidar']['points'].shape}")
            print(f"  Normalized points shape: {sample['lidar']['points_normalized'].shape}")
            print(f"  Mean: {sample['lidar']['mean']}")


def example_9_coordinate_transforms():
    """
    Example 9: Coordinate System Transformations
    ============================================
    
    Demonstrates how to:
    - Transform between coordinate systems
    - Transform points
    - Transform bounding boxes
    """
    print("\n" + "=" * 70)
    print("Example 9: Coordinate System Transformations")
    print("=" * 70)
    
    from dataset.nuscenes_adapter import CoordinateTransformer, BoundingBox3D, Vector3D, Quaternion3D
    
    transformer = CoordinateTransformer()
    
    # Create sample points in sensor coordinate system
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    print(f"Original points (sensor frame):\n{points}")
    
    # Create transformation matrix (sensor to ego)
    # Sensor is 1.5m forward and 0.5m up from ego center
    translation = np.array([1.5, 0.0, 0.5])
    rotation = np.eye(3)  # No rotation for simplicity
    sensor2ego = transformer.get_transform_matrix(translation, rotation)
    print(f"\nSensor-to-ego transformation:\n{sensor2ego}")
    
    # Transform points
    points_ego = transformer.sensor_to_ego(points, sensor2ego)
    print(f"\nPoints in ego frame:\n{points_ego}")
    
    # Create ego-to-global transformation
    ego_translation = np.array([100.0, 50.0, 0.0])
    # 45 degree rotation around z-axis
    angle = np.pi / 4
    ego_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    ego2global = transformer.get_transform_matrix(ego_translation, ego_rotation)
    print(f"\nEgo-to-global transformation:\n{ego2global}")
    
    # Transform to global
    points_global = transformer.ego_to_global(points_ego, ego2global)
    print(f"\nPoints in global frame:\n{points_global}")
    
    # Transform back to ego
    points_back_to_ego = transformer.global_to_go(points_global, ego2global)
    print(f"\nPoints back in ego frame:\n{points_back_to_ego}")
    
    # Transform bounding box
    print("\nBounding Box Transformation:")
    bbox = BoundingBox3D(
        center=Vector3D(x=5.0, y=2.0, z=0.5),
        size=Vector3D(x=4.0, y=1.8, z=1.5),
        rotation=Quaternion3D(w=1.0, x=0.0, y=0.0, z=0.0),
        category='car'
    )
    print(f"  Original center: ({bbox.center.x}, {bbox.center.y}, {bbox.center.z})")
    
    transformed_bbox = transformer.transform_bbox(bbox, ego2global)
    print(f"  Transformed center: ({transformed_bbox.center.x:.2f}, "
          f"{transformed_bbox.center.y:.2f}, {transformed_bbox.center.z:.2f})")


def example_10_batch_processing():
    """
    Example 10: Batch Processing Pipeline
    =====================================
    
    Demonstrates a complete training pipeline with batch processing.
    """
    print("\n" + "=" * 70)
    print("Example 10: Batch Processing Pipeline")
    print("=" * 70)
    
    # Configuration
    config = NuScenesConfig(
        data_root="/data/nuscenes",
        version="v1.0-mini",
        use_camera=True,
        use_lidar=True,
        use_radar=False,
        image_size=(450, 800),
        max_lidar_points=8000,
        enable_augmentation=True,
    )
    
    print("Configuration:")
    print(f"  Batch size: 4")
    print(f"  Image size: {config.image_size}")
    print(f"  Max LiDAR points: {config.max_lidar_points}")
    print(f"  Augmentation: {config.enable_augmentation}")
    
    # Create data loader
    loader = NuScenesDataLoader(
        config,
        split='mini_train',
        batch_size=4,
        shuffle=True,
    )
    
    print(f"\nStarting training loop simulation...")
    print(f"Total batches: {len(loader)}")
    
    # Simulate training loop
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}/{len(loader)}:")
        
        # Process batch
        batch_size = len(batch['tokens'])
        print(f"  Batch size: {batch_size}")
        
        # Camera data
        if 'camera_images' in batch:
            cam_data = batch['camera_images']['CAM_FRONT']
            print(f"  CAM_FRONT batch shape: {cam_data.shape}")
            print(f"  CAM_FRONT dtype: {cam_data.dtype}")
            print(f"  CAM_FRONT range: [{cam_data.min():.3f}, {cam_data.max():.3f}]")
        
        # LiDAR data
        if 'lidar_points' in batch:
            total_points = sum(p.shape[0] for p in batch['lidar_points'])
            print(f"  Total LiDAR points in batch: {total_points}")
            print(f"  Points per sample: {[p.shape[0] for p in batch['lidar_points']]}")
        
        # Annotations
        if 'annotations' in batch:
            total_anns = sum(len(a) for a in batch['annotations'])
            print(f"  Total annotations in batch: {total_anns}")
        
        # Simulate forward pass
        # In real training: output = model(batch)
        # loss = criterion(output, targets)
        # loss.backward()
        # optimizer.step()
        
        # Stop after 2 batches for demo
        if batch_idx >= 1:
            print("\n... (stopping after 2 batches for demo)")
            break
    
    print("\nTraining loop simulation completed!")


# =============================================================================
# Run All Examples
# =============================================================================

def run_all_examples():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("# nuScenes Adapter - Complete Usage Examples")
    print("#" * 70)
    
    examples = [
        ("Basic Data Loading", example_1_basic_loading),
        ("Configuration Management", example_2_configuration),
        ("Dataset Interface", example_3_dataset_interface),
        ("DataLoader for Training", example_4_dataloader),
        ("Custom Preprocessing", example_5_preprocessing),
        ("Data Augmentation", example_6_augmentation),
        ("Evaluation Interface", example_7_evaluation),
        ("Custom Transform Functions", example_8_custom_transform),
        ("Coordinate Transformations", example_9_coordinate_transforms),
        ("Batch Processing Pipeline", example_10_batch_processing),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#" * 70)
    print("# All examples completed!")
    print("#" * 70)


if __name__ == '__main__':
    # Run all examples
    run_all_examples()
