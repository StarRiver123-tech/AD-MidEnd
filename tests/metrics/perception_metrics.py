"""
感知模块评估指标
包括准确率、召回率、F1分数、IoU等指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    """目标类型"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """检测结果"""
    object_type: ObjectType
    bbox: np.ndarray  # (7,) - x, y, z, l, w, h, heading
    confidence: float
    

@dataclass
class DetectionMetrics:
    """检测指标"""
    precision: float  # 准确率
    recall: float  # 召回率
    f1_score: float  # F1分数
    average_iou: float  # 平均IoU
    ap_50: float  # AP@0.5
    ap_70: float  # AP@0.7
    true_positives: int
    false_positives: int
    false_negatives: int
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'average_iou': self.average_iou,
            'ap_50': self.ap_50,
            'ap_70': self.ap_70,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
        }


class PerceptionMetrics:
    """感知模块评估指标计算器"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        初始化感知指标计算器
        
        Args:
            iou_threshold: IoU阈值，用于判断匹配
        """
        self.iou_threshold = iou_threshold
    
    def calculate_iou_3d(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray
    ) -> float:
        """
        计算两个3D边界框的IoU
        
        Args:
            bbox1: 边界框1 (x, y, z, l, w, h, heading)
            bbox2: 边界框2 (x, y, z, l, w, h, heading)
            
        Returns:
            IoU值（0-1）
        """
        # 简化的IoU计算（基于中心距离和尺寸）
        center1 = bbox1[:3]
        center2 = bbox2[:3]
        size1 = bbox1[3:6]
        size2 = bbox2[3:6]
        
        # 中心距离
        center_dist = np.linalg.norm(center1 - center2)
        
        # 尺寸差异
        size_diff = np.abs(size1 - size2)
        
        # 简化的IoU估计
        max_dist = np.linalg.norm(size1) + np.linalg.norm(size2)
        if max_dist == 0:
            return 1.0
        
        # 基于中心距离和尺寸相似度的综合评分
        center_score = max(0, 1 - center_dist / (max_dist * 0.5))
        size_score = 1 - np.mean(size_diff) / (np.mean(size1) + 1e-6)
        size_score = max(0, size_score)
        
        return center_score * size_score
    
    def calculate_iou_bev(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray
    ) -> float:
        """
        计算鸟瞰图(BEV)上的IoU
        
        Args:
            bbox1: 边界框1 (x, y, z, l, w, h, heading)
            bbox2: 边界框2 (x, y, z, l, w, h, heading)
            
        Returns:
            BEV IoU值
        """
        # 提取BEV参数
        x1, y1, l1, w1, heading1 = bbox1[0], bbox1[1], bbox1[3], bbox1[4], bbox1[6]
        x2, y2, l2, w2, heading2 = bbox2[0], bbox2[1], bbox2[3], bbox2[4], bbox2[6]
        
        # 简化的BEV IoU（使用中心距离和尺寸）
        center_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        avg_size = (np.sqrt(l1**2 + w1**2) + np.sqrt(l2**2 + w2**2)) / 2
        
        if avg_size == 0:
            return 1.0 if center_dist < 0.1 else 0.0
        
        iou = max(0, 1 - center_dist / avg_size)
        return iou
    
    def match_detections(
        self,
        ground_truths: List[DetectionResult],
        predictions: List[DetectionResult],
        iou_threshold: Optional[float] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匹配检测结果与真值
        
        Args:
            ground_truths: 真值列表
            predictions: 预测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            (匹配对列表, 未匹配的真值索引, 未匹配的预测索引)
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        matches = []
        unmatched_gts = list(range(len(ground_truths)))
        unmatched_preds = list(range(len(predictions)))
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(ground_truths), len(predictions)))
        for i, gt in enumerate(ground_truths):
            for j, pred in enumerate(predictions):
                if gt.object_type == pred.object_type:
                    iou_matrix[i, j] = self.calculate_iou_bev(gt.bbox, pred.bbox)
        
        # 贪婪匹配
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            gt_idx, pred_idx = max_idx
            
            matches.append((gt_idx, pred_idx))
            unmatched_gts.remove(gt_idx)
            unmatched_preds.remove(pred_idx)
            
            # 移除已匹配的行和列
            iou_matrix[gt_idx, :] = -1
            iou_matrix[:, pred_idx] = -1
        
        return matches, unmatched_gts, unmatched_preds
    
    def calculate_detection_metrics(
        self,
        ground_truths: List[DetectionResult],
        predictions: List[DetectionResult],
        confidence_threshold: float = 0.5
    ) -> DetectionMetrics:
        """
        计算检测指标
        
        Args:
            ground_truths: 真值列表
            predictions: 预测结果列表
            confidence_threshold: 置信度阈值
            
        Returns:
            DetectionMetrics对象
        """
        # 过滤低置信度预测
        filtered_predictions = [p for p in predictions if p.confidence >= confidence_threshold]
        
        # 匹配检测
        matches, unmatched_gts, unmatched_preds = self.match_detections(
            ground_truths, filtered_predictions
        )
        
        # 计算TP, FP, FN
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        
        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算F1分数
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算平均IoU
        if matches:
            ious = []
            for gt_idx, pred_idx in matches:
                iou = self.calculate_iou_bev(
                    ground_truths[gt_idx].bbox,
                    filtered_predictions[pred_idx].bbox
                )
                ious.append(iou)
            average_iou = np.mean(ious)
        else:
            average_iou = 0.0
        
        # 计算AP@0.5和AP@0.7 (使用简化计算避免递归)
        ap_50 = self._calculate_ap_simple(ground_truths, predictions, iou_threshold=0.5)
        ap_70 = self._calculate_ap_simple(ground_truths, predictions, iou_threshold=0.7)
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            average_iou=average_iou,
            ap_50=ap_50,
            ap_70=ap_70,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )
    
    def calculate_ap(
        self,
        ground_truths: List[DetectionResult],
        predictions: List[DetectionResult],
        iou_threshold: float = 0.5
    ) -> float:
        """
        计算Average Precision (AP)
        
        Args:
            ground_truths: 真值列表
            predictions: 预测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            AP值
        """
        if not predictions:
            return 0.0
        
        # 按置信度排序
        sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # 计算每个置信度阈值下的精确率和召回率
        precisions = []
        recalls = []
        
        for i in range(len(sorted_preds)):
            threshold = sorted_preds[i].confidence
            filtered_preds = [p for p in sorted_preds if p.confidence >= threshold]
            
            metrics = self.calculate_detection_metrics(ground_truths, filtered_preds)
            precisions.append(metrics.precision)
            recalls.append(metrics.recall)
        
        # 使用11点插值计算AP
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            # 找到召回率大于等于r的最大精确率
            max_prec = 0.0
            for prec, rec in zip(precisions, recalls):
                if rec >= r:
                    max_prec = max(max_prec, prec)
            ap += max_prec / 11.0
        
        return ap
    
    def calculate_mAP(
        self,
        ground_truths_by_type: Dict[ObjectType, List[DetectionResult]],
        predictions_by_type: Dict[ObjectType, List[DetectionResult]],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算mAP（所有类别的平均AP）
        
        Args:
            ground_truths_by_type: 按类型分组的真值
            predictions_by_type: 按类型分组的预测
            iou_threshold: IoU阈值
            
        Returns:
            包含各类别AP和mAP的字典
        """
        results = {}
        aps = []
        
        for obj_type in ObjectType:
            gts = ground_truths_by_type.get(obj_type, [])
            preds = predictions_by_type.get(obj_type, [])
            
            ap = self.calculate_ap(gts, preds, iou_threshold)
            results[f'ap_{obj_type.value}'] = ap
            aps.append(ap)
        
        results['mAP'] = np.mean(aps) if aps else 0.0
        
        return results
    
    def calculate_tracking_metrics(
        self,
        ground_truth_trajectories: Dict[str, List[np.ndarray]],
        predicted_trajectories: Dict[str, List[np.ndarray]],
        distance_threshold: float = 2.0
    ) -> Dict[str, float]:
        """
        计算跟踪指标（MOTA, MOTP）
        
        Args:
            ground_truth_trajectories: 真值轨迹 {id: [positions]}
            predicted_trajectories: 预测轨迹 {id: [positions]}
            distance_threshold: 距离阈值（米）
            
        Returns:
            跟踪指标字典
        """
        total_matches = 0
        total_misses = 0
        total_false_positives = 0
        total_distance = 0.0
        
        # 获取所有时间步
        max_len = max(
            max(len(traj) for traj in ground_truth_trajectories.values()) if ground_truth_trajectories else 0,
            max(len(traj) for traj in predicted_trajectories.values()) if predicted_trajectories else 0
        )
        
        for t in range(max_len):
            # 获取当前时间步的真值和预测
            gt_positions = {
                id: traj[t] for id, traj in ground_truth_trajectories.items()
                if t < len(traj)
            }
            pred_positions = {
                id: traj[t] for id, traj in predicted_trajectories.items()
                if t < len(traj)
            }
            
            # 匹配
            matched_gts = set()
            matched_preds = set()
            
            for gt_id, gt_pos in gt_positions.items():
                for pred_id, pred_pos in pred_positions.items():
                    if pred_id in matched_preds:
                        continue
                    
                    dist = np.linalg.norm(gt_pos - pred_pos)
                    if dist < distance_threshold:
                        total_matches += 1
                        total_distance += dist
                        matched_gts.add(gt_id)
                        matched_preds.add(pred_id)
                        break
            
            total_misses += len(gt_positions) - len(matched_gts)
            total_false_positives += len(pred_positions) - len(matched_preds)
        
        # 计算MOTA和MOTP
        total_objects = sum(len(traj) for traj in ground_truth_trajectories.values())
        
        mota = 1 - (total_misses + total_false_positives) / total_objects if total_objects > 0 else 0.0
        motp = total_distance / total_matches if total_matches > 0 else 0.0
        
        return {
            'MOTA': mota,
            'MOTP': motp,
            'matches': total_matches,
            'misses': total_misses,
            'false_positives': total_false_positives
        }
