"""
Lane Detection Module for Autonomous Driving
基于Transformer的车道线检测网络

Features:
- Vector Lane Encoding with Self-Attention and Cross-Attention
- Point Prediction (Level1/Level2)
- Topology Type Prediction
- Fork/Merge Point Prediction
- Spline Coefficient Prediction
- nuScenes Dataset Support

Author: Autonomous Driving Perception Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import yaml
from pathlib import Path


# ==================== Data Structures ====================

@dataclass
class LanePoint:
    """车道线点数据结构"""
    x: float
    y: float
    z: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'LanePoint':
        return cls(x=tensor[0].item(), y=tensor[1].item(), z=tensor[2].item())


@dataclass
class LaneInstance:
    """车道线实例数据结构"""
    lane_id: int
    points: List[LanePoint] = field(default_factory=list)
    lane_type: str = "unknown"  # "solid", "dashed", "double", etc.
    color: str = "white"  # "white", "yellow", etc.
    confidence: float = 1.0
    spline_coeffs: Optional[torch.Tensor] = None
    
    def get_centerline(self) -> torch.Tensor:
        """获取车道线中心线点"""
        if not self.points:
            return torch.empty(0, 3)
        return torch.stack([p.to_tensor() for p in self.points])


@dataclass
class TopologyType:
    """道路拓扑类型"""
    LANE_FOLLOW = 0
    LEFT_TURN = 1
    RIGHT_TURN = 2
    STRAIGHT = 3
    U_TURN = 4
    MERGE = 5
    FORK = 6
    
    type_id: int = 0
    confidence: float = 1.0
    
    @property
    def name(self) -> str:
        names = {
            0: "LANE_FOLLOW",
            1: "LEFT_TURN",
            2: "RIGHT_TURN",
            3: "STRAIGHT",
            4: "U_TURN",
            5: "MERGE",
            6: "FORK"
        }
        return names.get(self.type_id, "UNKNOWN")


@dataclass
class LaneDetectionOutput:
    """车道线检测输出数据结构"""
    lane_instances: List[LaneInstance]
    adjacency_matrix: torch.Tensor  # [num_lanes, num_lanes]
    topology_types: List[TopologyType]
    fork_points: List[LanePoint]
    merge_points: List[LanePoint]
    timestamp: float
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "lane_instances": [
                {
                    "lane_id": li.lane_id,
                    "points": [(p.x, p.y, p.z) for p in li.points],
                    "lane_type": li.lane_type,
                    "color": li.color,
                    "confidence": li.confidence
                }
                for li in self.lane_instances
            ],
            "adjacency_matrix": self.adjacency_matrix.cpu().numpy().tolist(),
            "topology_types": [tt.name for tt in self.topology_types],
            "fork_points": [(p.x, p.y, p.z) for p in self.fork_points],
            "merge_points": [(p.x, p.y, p.z) for p in self.merge_points],
            "timestamp": self.timestamp
        }


# ==================== Transformer Components ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Cross-attention (if context provided)
        if context is not None:
            cross_attn_out = self.cross_attn(x, context, context)
            x = self.norm2(x + cross_attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        
        return x


class VectorLaneEncoder(nn.Module):
    """
    Vector Lane Encoding模块
    将车道线向量编码为特征表示
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        d_model = config.get('d_model', 256)
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_encoder_layers', 6)
        d_ff = config.get('d_ff', 1024)
        dropout = config.get('dropout', 0.1)
        
        # Input embedding
        self.point_embed = nn.Linear(3, d_model)  # (x, y, z) -> d_model
        self.type_embed = nn.Embedding(10, d_model)  # Lane type embedding
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
        
    def forward(self, lane_vectors: torch.Tensor, 
                lane_types: Optional[torch.Tensor] = None,
                bev_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            lane_vectors: [B, N, P, 3] - batch, num_lanes, points_per_lane, xyz
            lane_types: [B, N] - lane type indices
            bev_features: [B, C, H, W] - BEV feature map for cross-attention
        
        Returns:
            lane_features: [B, N, d_model] - encoded lane features
        """
        B, N, P, _ = lane_vectors.shape
        
        # Flatten lanes and points
        x = lane_vectors.view(B * N, P, 3)  # [B*N, P, 3]
        x = self.point_embed(x)  # [B*N, P, d_model]
        
        # Add type embedding
        if lane_types is not None:
            type_emb = self.type_embed(lane_types.view(B * N))  # [B*N, d_model]
            x = x + type_emb.unsqueeze(1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Prepare BEV features for cross-attention
        context = None
        if bev_features is not None:
            # Flatten spatial dimensions
            BEV_C, BEV_H, BEV_W = bev_features.shape[1:]
            context = bev_features.view(B, BEV_C, -1).permute(0, 2, 1)  # [B, H*W, C]
            context = context.repeat_interleave(N, dim=0)  # [B*N, H*W, C]
            # Project to d_model
            context = nn.Linear(BEV_C, self.d_model).to(context.device)(context)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            # Alternate between self-attention and cross-attention
            if i % 2 == 0 or context is None:
                x = layer(x, None)  # Self-attention only
            else:
                x = layer(x, context)  # With cross-attention
        
        # Pool over points
        x = x.mean(dim=1)  # [B*N, d_model]
        x = x.view(B, N, self.d_model)  # [B, N, d_model]
        
        return x


# ==================== Prediction Heads ====================

class PointPredictor(nn.Module):
    """
    Point Predictor - 预测车道线上的点
    Level1: 粗略位置预测
    Level2: 精细位置优化
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        d_model = config.get('d_model', 256)
        num_points = config.get('num_points_per_lane', 20)
        
        # Level 1: Coarse prediction
        self.level1_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_points * 3)
        )
        
        # Level 2: Fine refinement
        self.level2_mlp = nn.Sequential(
            nn.Linear(d_model + num_points * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_points * 3)
        )
        
        self.num_points = num_points
        
    def forward(self, lane_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lane_features: [B, N, d_model]
        
        Returns:
            level1_points: [B, N, P, 3] - coarse predictions
            level2_points: [B, N, P, 3] - refined predictions
        """
        B, N, D = lane_features.shape
        
        # Level 1 prediction
        level1_flat = self.level1_mlp(lane_features)  # [B, N, P*3]
        level1_points = level1_flat.view(B, N, self.num_points, 3)
        
        # Level 2 refinement
        level2_input = torch.cat([lane_features, level1_flat], dim=-1)
        level2_residual = self.level2_mlp(level2_input)
        level2_residual = level2_residual.view(B, N, self.num_points, 3)
        
        # Residual connection
        level2_points = level1_points + level2_residual
        
        return level1_points, level2_points


class TopologyTypePredictor(nn.Module):
    """
    Topology Type Predictor - 预测车道线之间的拓扑关系类型
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        d_model = config.get('d_model', 256)
        num_topology_types = config.get('num_topology_types', 7)
        
        # Topology classification head
        self.topology_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(d_model, num_topology_types)
        )
        
        self.num_topology_types = num_topology_types
        
    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_features: [B, N, d_model]
        
        Returns:
            topology_logits: [B, N, N, num_topology_types]
        """
        B, N, D = lane_features.shape
        
        # Create pairwise features
        lane_i = lane_features.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        lane_j = lane_features.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        
        # Concatenate pairwise
        pair_features = torch.cat([lane_i, lane_j], dim=-1)  # [B, N, N, 2D]
        
        # Predict topology type for each pair
        topology_logits = self.topology_mlp(pair_features)  # [B, N, N, num_types]
        
        return topology_logits


class ForkMergePointPredictor(nn.Module):
    """
    Fork/Merge Point Predictor - 预测车道线的分叉和合并点
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        d_model = config.get('d_model', 256)
        
        # Fork point prediction
        self.fork_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # (x, y, z, confidence)
        )
        
        # Merge point prediction
        self.merge_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # (x, y, z, confidence)
        )
        
    def forward(self, lane_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lane_features: [B, N, d_model]
        
        Returns:
            fork_preds: [B, N, 4] - (x, y, z, confidence)
            merge_preds: [B, N, 4] - (x, y, z, confidence)
        """
        fork_preds = self.fork_head(lane_features)
        merge_preds = self.merge_head(lane_features)
        
        return fork_preds, merge_preds


class SplineCoefficientPredictor(nn.Module):
    """
    Spline Coefficient Predictor - 预测B样条曲线系数
    用于平滑的车道线表示
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        d_model = config.get('d_model', 256)
        spline_degree = config.get('spline_degree', 3)
        num_control_points = config.get('num_control_points', 10)
        
        # Each control point has (x, y, z) coordinates
        self.num_coeffs = num_control_points * 3
        
        self.spline_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_coeffs)
        )
        
        self.spline_degree = spline_degree
        self.num_control_points = num_control_points
        
    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_features: [B, N, d_model]
        
        Returns:
            spline_coeffs: [B, N, num_control_points, 3]
        """
        B, N, D = lane_features.shape
        
        coeffs = self.spline_mlp(lane_features)  # [B, N, num_coeffs]
        coeffs = coeffs.view(B, N, self.num_control_points, 3)
        
        return coeffs
    
    def evaluate_spline(self, coeffs: torch.Tensor, 
                       num_samples: int = 50) -> torch.Tensor:
        """
        评估B样条曲线
        
        Args:
            coeffs: [B, N, num_control_points, 3]
            num_samples: number of points to sample
        
        Returns:
            points: [B, N, num_samples, 3]
        """
        B, N, K, _ = coeffs.shape
        
        # Simple B-spline evaluation (using de Boor algorithm would be more accurate)
        # Here we use a simplified version with uniform parameterization
        t = torch.linspace(0, 1, num_samples, device=coeffs.device)
        
        # For simplicity, use polynomial interpolation
        points = []
        for i in range(num_samples):
            ti = t[i]
            # Simple polynomial evaluation
            point = torch.zeros(B, N, 3, device=coeffs.device)
            for k in range(K):
                weight = self._basis_function(ti, k, K)
                point += weight * coeffs[:, :, k, :]
            points.append(point)
        
        return torch.stack(points, dim=2)  # [B, N, num_samples, 3]
    
    def _basis_function(self, t: float, k: int, K: int) -> float:
        """简化的B样条基函数"""
        # Uniform B-spline basis (simplified)
        step = 1.0 / (K - 1)
        center = k * step
        width = step * 2
        
        if abs(t - center) < width / 2:
            return 1.0 - abs(t - center) / (width / 2)
        return 0.0


# ==================== Main Network ====================

class LaneDetectionNetwork(nn.Module):
    """
    完整的车道线检测网络
    
    Architecture:
    1. Vector Lane Encoding (Self-Attention → Cross-Attention → Self-Attention)
    2. Point Predictor (Level1/Level2)
    3. Topology Type Predictor
    4. Fork/Merge Point Predictor
    5. Spline Coefficient Predictor
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Vector Lane Encoder
        self.lane_encoder = VectorLaneEncoder(config)
        
        # Prediction heads
        self.point_predictor = PointPredictor(config)
        self.topology_predictor = TopologyTypePredictor(config)
        self.fork_merge_predictor = ForkMergePointPredictor(config)
        self.spline_predictor = SplineCoefficientPredictor(config)
        
        # BEV feature encoder (if input is image)
        if config.get('use_bev_encoder', True):
            self.bev_encoder = self._build_bev_encoder(config)
        else:
            self.bev_encoder = None
        
    def _build_bev_encoder(self, config: Dict) -> nn.Module:
        """构建BEV特征编码器"""
        in_channels = config.get('bev_in_channels', 3)
        out_channels = config.get('d_model', 256)
        
        return nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels, 64, 3, 2, 1),  # /2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # /4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # /8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 3, 2, 1),  # /16
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, 
                lane_vectors: torch.Tensor,
                lane_types: Optional[torch.Tensor] = None,
                bev_image: Optional[torch.Tensor] = None,
                dense_world_tensor: Optional[torch.Tensor] = None) -> Dict:
        """
        前向传播
        
        Args:
            lane_vectors: [B, N, P, 3] - initial lane point estimates
            lane_types: [B, N] - lane type indices
            bev_image: [B, 3, H, W] - BEV camera image
            dense_world_tensor: [B, C, H, W] - dense world representation
        
        Returns:
            outputs: Dict containing all predictions
        """
        # Encode BEV features if available
        bev_features = None
        if bev_image is not None and self.bev_encoder is not None:
            bev_features = self.bev_encoder(bev_image)
        elif dense_world_tensor is not None:
            bev_features = dense_world_tensor
        
        # Vector Lane Encoding
        lane_features = self.lane_encoder(lane_vectors, lane_types, bev_features)
        
        # Point Prediction
        level1_points, level2_points = self.point_predictor(lane_features)
        
        # Topology Type Prediction
        topology_logits = self.topology_predictor(lane_features)
        
        # Fork/Merge Point Prediction
        fork_preds, merge_preds = self.fork_merge_predictor(lane_features)
        
        # Spline Coefficient Prediction
        spline_coeffs = self.spline_predictor(lane_features)
        
        return {
            'lane_features': lane_features,
            'level1_points': level1_points,
            'level2_points': level2_points,
            'topology_logits': topology_logits,
            'fork_predictions': fork_preds,
            'merge_predictions': merge_preds,
            'spline_coefficients': spline_coeffs
        }


# ==================== nuScenes Dataset Support ====================

class NuScenesLaneDataset:
    """
    nuScenes数据集车道线数据加载器
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = config.get('nuscenes_root', './data/nuscenes')
        self.version = config.get('version', 'v1.0-trainval')
        
    def load_sample(self, sample_token: str) -> Dict:
        """加载单个样本"""
        # This is a placeholder - actual implementation would load from nuScenes
        # For now, return dummy data for testing
        
        num_lanes = np.random.randint(3, 8)
        num_points = self.config.get('num_points_per_lane', 20)
        
        # Generate dummy lane data
        lane_vectors = torch.randn(1, num_lanes, num_points, 3)
        lane_types = torch.randint(0, 5, (1, num_lanes))
        
        return {
            'lane_vectors': lane_vectors,
            'lane_types': lane_types,
            'timestamp': 0.0,
            'sample_token': sample_token
        }
    
    def format_output_to_nuscenes(self, 
                                   detection_output: LaneDetectionOutput) -> Dict:
        """将检测结果格式化为nuScenes格式"""
        # Convert to nuScenes prediction format
        nuscenes_format = {
            'lane_instances': [],
            'topology': []
        }
        
        for lane in detection_output.lane_instances:
            nuscenes_format['lane_instances'].append({
                'token': f"lane_{lane.lane_id}",
                'points': [(p.x, p.y, p.z) for p in lane.points],
                'type': lane.lane_type,
                'color': lane.color
            })
        
        return nuscenes_format


# ==================== Post-Processing ====================

class LanePostProcessor:
    """车道线检测结果后处理"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_topology_types = config.get('num_topology_types', 7)
        self.topology_threshold = config.get('topology_threshold', 0.5)
        self.fork_merge_threshold = config.get('fork_merge_threshold', 0.3)
        
    def process(self, network_output: Dict, timestamp: float = 0.0) -> LaneDetectionOutput:
        """
        处理网络输出，生成最终的检测结果
        """
        level2_points = network_output['level2_points']  # [B, N, P, 3]
        topology_logits = network_output['topology_logits']  # [B, N, N, T]
        fork_preds = network_output['fork_predictions']  # [B, N, 4]
        merge_preds = network_output['merge_predictions']  # [B, N, 4]
        spline_coeffs = network_output['spline_coefficients']  # [B, N, K, 3]
        
        B, N, P, _ = level2_points.shape
        
        # Build lane instances
        lane_instances = []
        for i in range(N):
            points = [
                LanePoint(
                    x=level2_points[0, i, j, 0].item(),
                    y=level2_points[0, i, j, 1].item(),
                    z=level2_points[0, i, j, 2].item()
                )
                for j in range(P)
            ]
            
            lane = LaneInstance(
                lane_id=i,
                points=points,
                spline_coeffs=spline_coeffs[0, i]
            )
            lane_instances.append(lane)
        
        # Build adjacency matrix from topology predictions
        topology_probs = F.softmax(topology_logits, dim=-1)  # [B, N, N, T]
        
        # Adjacency: lanes are connected if they have any meaningful topology
        adjacency_matrix = torch.zeros(N, N)
        topology_types = []
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Get most likely topology type
                    max_prob, max_type = topology_probs[0, i, j].max(dim=-1)
                    if max_prob.item() > self.topology_threshold:
                        adjacency_matrix[i, j] = max_prob.item()
                        topology_types.append(TopologyType(
                            type_id=max_type.item(),
                            confidence=max_prob.item()
                        ))
        
        # Extract fork/merge points
        fork_points = []
        merge_points = []
        
        for i in range(N):
            # Fork point
            if fork_preds[0, i, 3].sigmoid().item() > self.fork_merge_threshold:
                fork_points.append(LanePoint(
                    x=fork_preds[0, i, 0].item(),
                    y=fork_preds[0, i, 1].item(),
                    z=fork_preds[0, i, 2].item()
                ))
            
            # Merge point
            if merge_preds[0, i, 3].sigmoid().item() > self.fork_merge_threshold:
                merge_points.append(LanePoint(
                    x=merge_preds[0, i, 0].item(),
                    y=merge_preds[0, i, 1].item(),
                    z=merge_preds[0, i, 2].item()
                ))
        
        return LaneDetectionOutput(
            lane_instances=lane_instances,
            adjacency_matrix=adjacency_matrix,
            topology_types=topology_types,
            fork_points=fork_points,
            merge_points=merge_points,
            timestamp=timestamp
        )


# ==================== Main Interface ====================

class LanePerceptionModule:
    """
    车道线感知模块主接口
    """
    def __init__(self, config_path: Optional[str] = None):
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                # Flatten config: use model section as main config
                self.config = self._flatten_config(full_config)
        else:
            self.config = self._default_config()
        
        # Build network
        self.network = LaneDetectionNetwork(self.config)
        self.post_processor = LanePostProcessor(self.config)
        self.dataset = NuScenesLaneDataset(self.config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def _flatten_config(self, full_config: Dict) -> Dict:
        """将嵌套配置展平为单层配置"""
        flat_config = {}
        
        # Extract model configuration
        if 'model' in full_config:
            flat_config.update(full_config['model'])
        
        # Extract bev_encoder configuration
        if 'bev_encoder' in full_config:
            for key, value in full_config['bev_encoder'].items():
                flat_config[f'bev_{key}'] = value
        
        # Extract post_processing configuration
        if 'post_processing' in full_config:
            for key, value in full_config['post_processing'].items():
                flat_config[key] = value
        
        # Extract dataset configuration
        if 'dataset' in full_config:
            flat_config['nuscenes_root'] = full_config['dataset'].get('root', './data/nuscenes')
            flat_config['version'] = full_config['dataset'].get('version', 'v1.0-trainval')
        
        # Extract inference configuration
        if 'inference' in full_config:
            flat_config['max_lanes'] = full_config['inference'].get('max_lanes', 20)
        
        return flat_config
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'd_model': 256,
            'num_heads': 8,
            'num_encoder_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'num_points_per_lane': 20,
            'num_topology_types': 7,
            'spline_degree': 3,
            'num_control_points': 10,
            'use_bev_encoder': True,
            'bev_in_channels': 3,
            'topology_threshold': 0.5,
            'fork_merge_threshold': 0.3,
            'nuscenes_root': './data/nuscenes',
            'version': 'v1.0-trainval'
        }
    
    def detect(self, 
               lane_vectors: torch.Tensor,
               lane_types: Optional[torch.Tensor] = None,
               bev_image: Optional[torch.Tensor] = None,
               dense_world_tensor: Optional[torch.Tensor] = None,
               timestamp: float = 0.0) -> LaneDetectionOutput:
        """
        执行车道线检测
        
        Args:
            lane_vectors: [B, N, P, 3] - initial lane point estimates
            lane_types: [B, N] - lane type indices
            bev_image: [B, 3, H, W] - BEV camera image
            dense_world_tensor: [B, C, H, W] - dense world representation
            timestamp: timestamp for the detection
        
        Returns:
            LaneDetectionOutput with all detection results
        """
        # Move to device
        lane_vectors = lane_vectors.to(self.device)
        if lane_types is not None:
            lane_types = lane_types.to(self.device)
        if bev_image is not None:
            bev_image = bev_image.to(self.device)
        if dense_world_tensor is not None:
            dense_world_tensor = dense_world_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            network_output = self.network(
                lane_vectors=lane_vectors,
                lane_types=lane_types,
                bev_image=bev_image,
                dense_world_tensor=dense_world_tensor
            )
        
        # Post-process
        detection_output = self.post_processor.process(network_output, timestamp)
        
        return detection_output
    
    def load_pretrained(self, checkpoint_path: str):
        """加载预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        print(f"Loaded pretrained model from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str):
        """保存模型检查点"""
        torch.save({
            'network': self.network.state_dict(),
            'config': self.config
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


# ==================== Test Interface ====================

def test_lane_detection(config_path: Optional[str] = None) -> Dict:
    """
    车道线检测模块测试接口
    
    测试内容包括：
    1. 模块初始化
    2. 前向传播
    3. 输出格式验证
    4. 各预测头功能验证
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        test_results: 测试结果字典
    """
    print("=" * 60)
    print("Lane Detection Module Test")
    print("=" * 60)
    
    test_results = {
        'success': True,
        'tests': {},
        'errors': []
    }
    
    try:
        # Test 1: Module initialization
        print("\n[1/6] Testing module initialization...")
        module = LanePerceptionModule(config_path)
        test_results['tests']['initialization'] = 'PASSED'
        print("  ✓ Module initialized successfully")
        
        # Test 2: Create dummy input
        print("\n[2/6] Creating test input data...")
        batch_size = 1
        num_lanes = 5
        num_points = module.config.get('num_points_per_lane', 20)
        
        lane_vectors = torch.randn(batch_size, num_lanes, num_points, 3)
        lane_types = torch.randint(0, 5, (batch_size, num_lanes))
        bev_image = torch.randn(batch_size, 3, 512, 512)
        
        test_results['tests']['input_creation'] = 'PASSED'
        print(f"  ✓ Created input: {num_lanes} lanes, {num_points} points each")
        
        # Test 3: Forward pass
        print("\n[3/6] Testing forward pass...")
        output = module.detect(
            lane_vectors=lane_vectors,
            lane_types=lane_types,
            bev_image=bev_image,
            timestamp=123.456
        )
        test_results['tests']['forward_pass'] = 'PASSED'
        print("  ✓ Forward pass completed")
        
        # Test 4: Output validation
        print("\n[4/6] Validating output format...")
        assert len(output.lane_instances) == num_lanes, "Lane instance count mismatch"
        assert output.adjacency_matrix.shape == (num_lanes, num_lanes), "Adjacency matrix shape mismatch"
        assert isinstance(output.timestamp, float), "Timestamp type mismatch"
        
        test_results['tests']['output_validation'] = 'PASSED'
        print(f"  ✓ Output validated: {len(output.lane_instances)} lane instances")
        print(f"  ✓ Adjacency matrix shape: {output.adjacency_matrix.shape}")
        print(f"  ✓ Topology types: {len(output.topology_types)}")
        print(f"  ✓ Fork points: {len(output.fork_points)}")
        print(f"  ✓ Merge points: {len(output.merge_points)}")
        
        # Test 5: Network components
        print("\n[5/6] Testing network components...")
        network = module.network
        
        # Test VectorLaneEncoder
        lane_features = network.lane_encoder(lane_vectors, lane_types)
        assert lane_features.shape == (batch_size, num_lanes, module.config['d_model'])
        print(f"  ✓ VectorLaneEncoder: output shape {lane_features.shape}")
        
        # Test PointPredictor
        level1, level2 = network.point_predictor(lane_features)
        assert level1.shape == (batch_size, num_lanes, num_points, 3)
        assert level2.shape == (batch_size, num_lanes, num_points, 3)
        print(f"  ✓ PointPredictor: Level1 {level1.shape}, Level2 {level2.shape}")
        
        # Test TopologyTypePredictor
        topology_logits = network.topology_predictor(lane_features)
        assert topology_logits.shape == (batch_size, num_lanes, num_lanes, module.config['num_topology_types'])
        print(f"  ✓ TopologyTypePredictor: output shape {topology_logits.shape}")
        
        # Test ForkMergePointPredictor
        fork_preds, merge_preds = network.fork_merge_predictor(lane_features)
        assert fork_preds.shape == (batch_size, num_lanes, 4)
        assert merge_preds.shape == (batch_size, num_lanes, 4)
        print(f"  ✓ ForkMergePointPredictor: fork {fork_preds.shape}, merge {merge_preds.shape}")
        
        # Test SplineCoefficientPredictor
        spline_coeffs = network.spline_predictor(lane_features)
        num_control_points = module.config['num_control_points']
        assert spline_coeffs.shape == (batch_size, num_lanes, num_control_points, 3)
        print(f"  ✓ SplineCoefficientPredictor: output shape {spline_coeffs.shape}")
        
        test_results['tests']['network_components'] = 'PASSED'
        
        # Test 6: Output conversion
        print("\n[6/6] Testing output conversion...")
        output_dict = output.to_dict()
        assert 'lane_instances' in output_dict
        assert 'adjacency_matrix' in output_dict
        assert 'topology_types' in output_dict
        assert 'fork_points' in output_dict
        assert 'merge_points' in output_dict
        assert 'timestamp' in output_dict
        
        test_results['tests']['output_conversion'] = 'PASSED'
        print("  ✓ Output conversion successful")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for test_name, result in test_results['tests'].items():
            print(f"  {test_name}: {result}")
        print("=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        test_results['success'] = False
        test_results['errors'].append(str(e))
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return test_results


# ==================== Entry Point ====================

if __name__ == "__main__":
    # Run tests when executed directly
    config_path = "/mnt/okcomputer/output/autonomous_driving/config/lane_config.yaml"
    results = test_lane_detection(config_path)
    
    if results['success']:
        print("\nLane detection module is ready to use!")
    else:
        print("\nLane detection module has errors. Please check the output above.")
