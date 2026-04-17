"""
Lane Detection Module Usage Examples
车道线检测模块使用示例
"""

import torch
import numpy as np
from lane_detection import (
    LanePerceptionModule,
    LaneDetectionOutput,
    LaneInstance,
    LanePoint,
    test_lane_detection
)


def example_1_basic_usage():
    """示例1: 基本使用方式"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # 1. Initialize the module
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    module = LanePerceptionModule(config_path)
    print("✓ Module initialized")
    
    # 2. Prepare input data
    batch_size = 1
    num_lanes = 5
    num_points = 20
    
    # Create dummy lane vectors [B, N, P, 3]
    lane_vectors = torch.randn(batch_size, num_lanes, num_points, 3)
    lane_types = torch.randint(0, 5, (batch_size, num_lanes))
    bev_image = torch.randn(batch_size, 3, 512, 512)
    
    print(f"✓ Input prepared: {num_lanes} lanes, {num_points} points each")
    
    # 3. Run detection
    output = module.detect(
        lane_vectors=lane_vectors,
        lane_types=lane_types,
        bev_image=bev_image,
        timestamp=123.456
    )
    
    print(f"✓ Detection completed")
    print(f"  - Detected {len(output.lane_instances)} lane instances")
    print(f"  - Adjacency matrix shape: {output.adjacency_matrix.shape}")
    print(f"  - Fork points: {len(output.fork_points)}")
    print(f"  - Merge points: {len(output.merge_points)}")
    
    return output


def example_2_output_processing():
    """示例2: 处理输出结果"""
    print("\n" + "=" * 60)
    print("Example 2: Processing Output")
    print("=" * 60)
    
    # Run detection
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    module = LanePerceptionModule(config_path)
    
    lane_vectors = torch.randn(1, 4, 20, 3)
    output = module.detect(lane_vectors=lane_vectors, timestamp=0.0)
    
    # Process lane instances
    print("\nLane Instances:")
    for lane in output.lane_instances:
        print(f"  Lane {lane.lane_id}:")
        print(f"    - Type: {lane.lane_type}")
        print(f"    - Color: {lane.color}")
        print(f"    - Points: {len(lane.points)}")
        print(f"    - First point: ({lane.points[0].x:.2f}, {lane.points[0].y:.2f}, {lane.points[0].z:.2f})")
        print(f"    - Last point: ({lane.points[-1].x:.2f}, {lane.points[-1].y:.2f}, {lane.points[-1].z:.2f})")
    
    # Process adjacency matrix
    print("\nAdjacency Matrix:")
    adj_matrix = output.adjacency_matrix
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0.5:
                print(f"  Lane {i} -> Lane {j}: {adj_matrix[i, j]:.3f}")
    
    # Process topology types
    print("\nTopology Types:")
    for topo in output.topology_types:
        print(f"  - {topo.name}: {topo.confidence:.3f}")
    
    # Process fork/merge points
    print("\nFork Points:")
    for fp in output.fork_points:
        print(f"  - ({fp.x:.2f}, {fp.y:.2f}, {fp.z:.2f})")
    
    print("\nMerge Points:")
    for mp in output.merge_points:
        print(f"  - ({mp.x:.2f}, {mp.y:.2f}, {mp.z:.2f})")
    
    # Convert to dictionary
    output_dict = output.to_dict()
    print(f"\n✓ Output converted to dictionary with {len(output_dict)} keys")
    
    return output


def example_3_batch_processing():
    """示例3: 批量处理"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    module = LanePerceptionModule(config_path)
    
    # Process multiple frames
    num_frames = 3
    results = []
    
    for i in range(num_frames):
        lane_vectors = torch.randn(1, 5, 20, 3)
        output = module.detect(
            lane_vectors=lane_vectors,
            timestamp=float(i) * 0.1
        )
        results.append(output)
        print(f"  Frame {i}: {len(output.lane_instances)} lanes detected")
    
    print(f"\n✓ Processed {num_frames} frames")
    return results


def example_4_network_components():
    """示例4: 使用网络组件"""
    print("\n" + "=" * 60)
    print("Example 4: Using Network Components")
    print("=" * 60)
    
    from lane_detection import (
        VectorLaneEncoder,
        PointPredictor,
        TopologyTypePredictor,
        ForkMergePointPredictor,
        SplineCoefficientPredictor
    )
    
    config = {
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'num_points_per_lane': 20,
        'num_topology_types': 7,
        'spline_degree': 3,
        'num_control_points': 10
    }
    
    # Create components
    encoder = VectorLaneEncoder(config)
    point_pred = PointPredictor(config)
    topology_pred = TopologyTypePredictor(config)
    fork_merge_pred = ForkMergePointPredictor(config)
    spline_pred = SplineCoefficientPredictor(config)
    
    print("✓ All components created")
    
    # Test forward pass
    batch_size, num_lanes, num_points = 2, 5, 20
    lane_vectors = torch.randn(batch_size, num_lanes, num_points, 3)
    
    # Encoder
    lane_features = encoder(lane_vectors)
    print(f"  VectorLaneEncoder: {lane_vectors.shape} -> {lane_features.shape}")
    
    # Point predictor
    level1, level2 = point_pred(lane_features)
    print(f"  PointPredictor: {lane_features.shape} -> Level1: {level1.shape}, Level2: {level2.shape}")
    
    # Topology predictor
    topology_logits = topology_pred(lane_features)
    print(f"  TopologyTypePredictor: {lane_features.shape} -> {topology_logits.shape}")
    
    # Fork/Merge predictor
    fork_preds, merge_preds = fork_merge_pred(lane_features)
    print(f"  ForkMergePointPredictor: {lane_features.shape} -> fork: {fork_preds.shape}, merge: {merge_preds.shape}")
    
    # Spline predictor
    spline_coeffs = spline_pred(lane_features)
    print(f"  SplineCoefficientPredictor: {lane_features.shape} -> {spline_coeffs.shape}")


def example_5_save_load():
    """示例5: 保存和加载模型"""
    print("\n" + "=" * 60)
    print("Example 5: Save and Load Model")
    print("=" * 60)
    
    import tempfile
    import os
    
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    module = LanePerceptionModule(config_path)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
    
    module.save_checkpoint(checkpoint_path)
    
    # Load checkpoint
    module2 = LanePerceptionModule(config_path)
    module2.load_pretrained(checkpoint_path)
    
    # Clean up
    os.unlink(checkpoint_path)
    
    print("✓ Model saved and loaded successfully")


def example_6_nuscenes_format():
    """示例6: nuScenes格式输出"""
    print("\n" + "=" * 60)
    print("Example 6: nuScenes Format Output")
    print("=" * 60)
    
    from lane_detection import NuScenesLaneDataset
    
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    module = LanePerceptionModule(config_path)
    dataset = NuScenesLaneDataset(module.config)
    
    # Load sample
    sample = dataset.load_sample("sample_token_001")
    print(f"✓ Loaded sample with {sample['lane_vectors'].shape[1]} lanes")
    
    # Run detection
    output = module.detect(
        lane_vectors=sample['lane_vectors'],
        lane_types=sample['lane_types'],
        timestamp=sample['timestamp']
    )
    
    # Convert to nuScenes format
    nuscenes_output = dataset.format_output_to_nuscenes(output)
    print(f"✓ Converted to nuScenes format")
    print(f"  - Lane instances: {len(nuscenes_output['lane_instances'])}")
    
    return nuscenes_output


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Lane Detection Module - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_output_processing()
    example_3_batch_processing()
    example_4_network_components()
    example_5_save_load()
    example_6_nuscenes_format()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
