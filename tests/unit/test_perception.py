"""
感知模块单元测试
测试目标检测、跟踪、车道线检测等功能
"""

import pytest
import numpy as np
from typing import List

from data_generators.object_data_generator import (
    ObjectDataGenerator, ObjectType, ObjectStatus,
    TrackedObject, ObjectDetection
)
from data_generators.lane_data_generator import (
    LaneDataGenerator, LaneType, LanePosition,
    LaneLine, LanePoint
)
from metrics.perception_metrics import (
    PerceptionMetrics, DetectionResult, DetectionMetrics
)


@pytest.mark.unit
@pytest.mark.perception
class TestObjectDetection:
    """目标检测单元测试"""
    
    def test_object_generation(self, object_generator):
        """测试目标生成"""
        obj = object_generator.generate_single_object(
            object_type=ObjectType.VEHICLE,
            position=(10, 5, 0),
            velocity=(5, 0, 0)
        )
        
        assert obj is not None
        assert obj.object_type == ObjectType.VEHICLE
        assert obj.bbox.center_x == 10
        assert obj.bbox.center_y == 5
        assert obj.confidence > 0
        assert obj.confidence <= 1.0
    
    def test_object_set_generation(self, object_generator):
        """测试目标集合生成"""
        objects = object_generator.generate_object_set(num_objects=10)
        
        assert len(objects) == 10
        
        for obj in objects:
            assert obj.id is not None
            assert obj.object_type in ObjectType
            assert obj.confidence > 0
    
    def test_object_trajectory_generation(self, object_generator):
        """测试目标轨迹生成"""
        trajectory = object_generator.generate_object_trajectory(
            object_type=ObjectType.VEHICLE,
            duration=5.0,
            sample_rate=10.0
        )
        
        assert len(trajectory) == 50  # 5秒 * 10Hz
        
        # 检查轨迹连续性
        for i in range(1, len(trajectory)):
            prev_pos = np.array([
                trajectory[i-1].bbox.center_x,
                trajectory[i-1].bbox.center_y
            ])
            curr_pos = np.array([
                trajectory[i].bbox.center_x,
                trajectory[i].bbox.center_y
            ])
            
            # 检查位置变化是否合理
            displacement = np.linalg.norm(curr_pos - prev_pos)
            assert displacement < 5.0  # 0.1秒内移动不超过5米
    
    def test_detection_result_generation(self, object_generator):
        """测试检测结果生成"""
        ground_truth = object_generator.generate_single_object(
            object_type=ObjectType.VEHICLE
        )
        
        detection = object_generator.generate_detection_result(
            ground_truth,
            noise_level=0.1,
            miss_rate=0.0,
            false_positive_rate=0.0
        )
        
        assert detection is not None
        assert detection.object_type == ground_truth.object_type
        assert detection.confidence > 0
    
    def test_detection_with_miss(self, object_generator):
        """测试带漏检的检测"""
        ground_truth = object_generator.generate_single_object()
        
        # 高漏检率
        detections = []
        for _ in range(100):
            det = object_generator.generate_detection_result(
                ground_truth, miss_rate=0.5
            )
            if det is not None:
                detections.append(det)
        
        # 漏检率应该在0.5左右
        miss_rate_actual = 1 - len(detections) / 100
        assert 0.3 < miss_rate_actual < 0.7


@pytest.mark.unit
@pytest.mark.perception
class TestObjectTracking:
    """目标跟踪单元测试"""
    
    def test_trajectory_continuity(self, object_generator):
        """测试轨迹连续性"""
        trajectory = object_generator.generate_object_trajectory(
            motion_pattern="straight"
        )
        
        # 检查ID一致性
        ids = [obj.id for obj in trajectory]
        assert len(set(ids)) == 1
        
        # 检查年龄递增
        ages = [obj.age for obj in trajectory]
        assert ages == list(range(len(trajectory)))
    
    def test_motion_patterns(self, object_generator):
        """测试不同运动模式"""
        patterns = ["straight", "turn", "accelerate", "decelerate"]
        
        for pattern in patterns:
            trajectory = object_generator.generate_object_trajectory(
                motion_pattern=pattern
            )
            assert len(trajectory) > 0
            
            # 计算总位移
            start_pos = np.array([
                trajectory[0].bbox.center_x,
                trajectory[0].bbox.center_y
            ])
            end_pos = np.array([
                trajectory[-1].bbox.center_x,
                trajectory[-1].bbox.center_y
            ])
            displacement = np.linalg.norm(end_pos - start_pos)
            assert displacement > 0


@pytest.mark.unit
@pytest.mark.perception
class TestLaneDetection:
    """车道线检测单元测试"""
    
    def test_straight_lane_generation(self, lane_generator):
        """测试直线车道线生成"""
        lane = lane_generator.generate_straight_lane(
            length=100.0,
            num_points=100
        )
        
        assert len(lane.points) == 100
        assert lane.lane_type == LaneType.DASHED_WHITE
        
        # 检查直线特性
        for point in lane.points:
            assert point.x == 0  # 直线在x=0
            assert point.curvature == 0  # 直线曲率为0
    
    def test_curved_lane_generation(self, lane_generator):
        """测试曲线车道线生成"""
        curvature = 0.01
        lane = lane_generator.generate_curved_lane(
            length=100.0,
            curvature=curvature
        )
        
        assert len(lane.points) > 0
        
        # 检查曲率
        avg_curvature = lane.get_average_curvature()
        assert abs(avg_curvature - curvature) < 0.001
    
    def test_sine_lane_generation(self, lane_generator):
        """测试正弦曲线车道线生成"""
        lane = lane_generator.generate_sine_lane(
            length=100.0,
            amplitude=2.0,
            frequency=0.1
        )
        
        assert len(lane.points) > 0
        
        # 检查是否有弯曲
        curvatures = [p.curvature for p in lane.points]
        assert max(curvatures) > 0
    
    def test_lane_segment_generation(self, lane_generator):
        """测试车道段生成"""
        result = lane_generator.generate_lane_segment(
            num_lanes=3,
            lane_width=3.5
        )
        
        assert len(result.lane_lines) == 4  # 3车道有4条边界线
        assert result.ego_lane is not None
        assert result.ego_lane.lane_width == 3.5
    
    def test_lane_with_noise(self, lane_generator):
        """测试带噪声的车道线"""
        ground_truth = lane_generator.generate_straight_lane()
        noisy_lane = lane_generator.generate_lane_with_noise(
            ground_truth,
            position_noise=0.1
        )
        
        assert len(noisy_lane.points) <= len(ground_truth.points)
        
        # 检查噪声是否添加
        if len(noisy_lane.points) > 0:
            assert noisy_lane.points[0].x != 0  # 应该偏离x=0
    
    def test_occluded_lane(self, lane_generator):
        """测试被遮挡的车道线"""
        ground_truth = lane_generator.generate_straight_lane(num_points=100)
        occluded_lane = lane_generator.generate_occluded_lane(
            ground_truth,
            occlusion_start=0.3,
            occlusion_end=0.7
        )
        
        # 应该丢失约40%的点
        assert len(occluded_lane.points) < len(ground_truth.points)
        assert len(occluded_lane.points) > len(ground_truth.points) * 0.5


@pytest.mark.unit
@pytest.mark.perception
class TestPerceptionMetrics:
    """感知评估指标单元测试"""
    
    def test_iou_calculation(self, perception_metrics):
        """测试IoU计算"""
        bbox1 = np.array([0, 0, 0, 4, 2, 1.5, 0])
        bbox2 = np.array([1, 0, 0, 4, 2, 1.5, 0])
        
        iou = perception_metrics.calculate_iou_bev(bbox1, bbox2)
        
        assert 0 <= iou <= 1
        assert iou > 0  # 有重叠
    
    def test_perfect_match(self, perception_metrics, object_generator):
        """测试完美匹配"""
        ground_truth = object_generator.generate_object_set(num_objects=5)
        
        # 无噪声的检测
        detections = object_generator.generate_detection_results(
            ground_truth,
            noise_level=0.0,
            miss_rate=0.0,
            false_positive_rate=0.0
        )
        
        # 转换为DetectionResult格式
        gt_results = [
            DetectionResult(
                object_type=gt.object_type,
                bbox=np.array([
                    gt.bbox.center_x, gt.bbox.center_y, gt.bbox.center_z,
                    gt.bbox.length, gt.bbox.width, gt.bbox.height, gt.bbox.heading
                ]),
                confidence=gt.confidence
            )
            for gt in ground_truth
        ]
        
        det_results = [
            DetectionResult(
                object_type=det.object_type,
                bbox=np.array([
                    det.bbox.center_x, det.bbox.center_y, det.bbox.center_z,
                    det.bbox.length, det.bbox.width, det.bbox.height, det.bbox.heading
                ]),
                confidence=det.confidence
            )
            for det in detections
        ]
        
        metrics = perception_metrics.calculate_detection_metrics(gt_results, det_results)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
    
    def test_detection_metrics_with_noise(self, perception_metrics, object_generator):
        """测试带噪声的检测指标"""
        ground_truth = object_generator.generate_object_set(num_objects=10)
        detections = object_generator.generate_detection_results(
            ground_truth,
            noise_level=0.2,
            miss_rate=0.1,
            false_positive_rate=0.1
        )
        
        # 转换为DetectionResult格式
        gt_results = [
            DetectionResult(
                object_type=gt.object_type,
                bbox=np.array([
                    gt.bbox.center_x, gt.bbox.center_y, gt.bbox.center_z,
                    gt.bbox.length, gt.bbox.width, gt.bbox.height, gt.bbox.heading
                ]),
                confidence=gt.confidence
            )
            for gt in ground_truth
        ]
        
        det_results = [
            DetectionResult(
                object_type=det.object_type,
                bbox=np.array([
                    det.bbox.center_x, det.bbox.center_y, det.bbox.center_z,
                    det.bbox.length, det.bbox.width, det.bbox.height, det.bbox.heading
                ]),
                confidence=det.confidence
            )
            for det in detections
        ]
        
        metrics = perception_metrics.calculate_detection_metrics(gt_results, det_results)
        
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.true_positives >= 0
        assert metrics.false_positives >= 0
        assert metrics.false_negatives >= 0
    
    def test_ap_calculation(self, perception_metrics, object_generator):
        """测试AP计算"""
        ground_truth = object_generator.generate_object_set(num_objects=5)
        detections = object_generator.generate_detection_results(ground_truth)
        
        # 转换为DetectionResult格式
        gt_results = [
            DetectionResult(
                object_type=gt.object_type,
                bbox=np.array([
                    gt.bbox.center_x, gt.bbox.center_y, gt.bbox.center_z,
                    gt.bbox.length, gt.bbox.width, gt.bbox.height, gt.bbox.heading
                ]),
                confidence=gt.confidence
            )
            for gt in ground_truth
        ]
        
        det_results = [
            DetectionResult(
                object_type=det.object_type,
                bbox=np.array([
                    det.bbox.center_x, det.bbox.center_y, det.bbox.center_z,
                    det.bbox.length, det.bbox.width, det.bbox.height, det.bbox.heading
                ]),
                confidence=det.confidence
            )
            for det in detections
        ]
        
        ap = perception_metrics.calculate_ap(gt_results, det_results)
        
        assert 0 <= ap <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
