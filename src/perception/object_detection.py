"""
障碍物感知模块 - 基于BEV的3D目标检测网络
Object Detection Module for Autonomous Driving

支持功能:
- Temporal Alignment (Self-Attention) - 时序对齐
- Occupancy Net - 占据网络
- Detection Head - 检测头
- ROI Temporal Alignment - ROI时序对齐
- 输出: 目标位置、类别、速度、加速度、行人姿态、形状网格、未来轨迹

支持数据集: nuScenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np
import math


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ObjectDetectionConfig:
    """障碍物检测配置类"""
    # BEV特征维度
    bev_channels: int = 256
    bev_height: int = 200
    bev_width: int = 200
    
    # 时序配置
    num_frames: int = 8  # 历史帧数
    num_future_frames: int = 6  # 未来预测帧数
    
    # Transformer配置
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # 检测头配置
    num_classes: int = 10  # nuScenes类别数
    num_queries: int = 300  # 查询数量
    
    # 3D检测配置
    num_sizes: int = 10  # 尺寸聚类数
    num_rotations: int = 12  # 旋转角度离散数
    
    # 行人姿态配置
    num_pose_keypoints: int = 18  # 关键点数量
    
    # 形状网格配置
    mesh_resolution: int = 32  # 网格分辨率
    
    # 轨迹预测配置
    trajectory_points: int = 12  # 轨迹点数
    
    # 类别定义 (nuScenes)
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
            ]


# =============================================================================
# 基础模块
# =============================================================================

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))


class FeedForward(nn.Module):
    """前馈网络模块"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.norm(x + self.dropout2(x2))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attn(x, mask)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# =============================================================================
# Temporal Alignment Module
# =============================================================================

class TemporalAlignment(nn.Module):
    """
    时序对齐模块 (Self-Attention)
    对历史BEV特征进行时序对齐和融合
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_frames = config.num_frames
        
        # 特征投影
        self.bev_proj = nn.Sequential(
            nn.Conv2d(config.bev_channels, config.d_model, 1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU()
        )
        
        # 时序位置编码
        self.temporal_pe = nn.Parameter(torch.randn(1, config.num_frames, config.d_model))
        
        # 自注意力编码器
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.bev_channels)
        )
        
    def forward(self, bev_features: torch.Tensor, ego_motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_features: [B, T, C, H, W] 历史BEV特征
            ego_motion: [B, T, 4, 4] 自车运动矩阵
        Returns:
            aligned_features: [B, C, H, W] 对齐后的特征
        """
        B, T, C, H, W = bev_features.shape
        
        # 将BEV特征投影到d_model维度
        bev_flat = bev_features.view(B * T, C, H, W)
        bev_proj = self.bev_proj(bev_flat)  # [B*T, d_model, H, W]
        
        # 空间池化得到时序特征
        bev_pooled = F.adaptive_avg_pool2d(bev_proj, 1).view(B, T, self.d_model)
        
        # 添加时序位置编码
        bev_pooled = bev_pooled + self.temporal_pe[:, :T, :]
        
        # 自注意力编码
        encoded = self.encoder(bev_pooled)  # [B, T, d_model]
        
        # 时序聚合 (使用注意力权重加权)
        attn_weights = F.softmax(torch.matmul(encoded, encoded.transpose(1, 2)), dim=-1)
        aggregated = torch.matmul(attn_weights, encoded)  # [B, T, d_model]
        
        # 输出投影
        output = self.output_proj(aggregated.mean(dim=1))  # [B, bev_channels]
        
        # 恢复到BEV空间
        output = output.view(B, self.config.bev_channels, 1, 1).expand(B, self.config.bev_channels, H, W)
        
        return output


# =============================================================================
# Occupancy Net Module
# =============================================================================

class OccupancyNet(nn.Module):
    """
    占据网络模块
    预测3D空间的占据情况
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # 深度维度 (BEV高度方向)
        self.depth_dim = 32
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(config.bev_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # 3D占据预测头
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.depth_dim, 1),
        )
        
        # 占据概率预测
        self.occupancy_prob = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 语义分割头
        self.semantic_head = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, config.num_classes, 1)
        )
        
    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_features: [B, C, H, W] BEV特征
        Returns:
            occupancy: 占据预测结果
        """
        B, C, H, W = bev_features.shape
        
        # 编码特征
        encoded = self.encoder(bev_features)  # [B, 128, H, W]
        
        # 预测深度方向的占据
        depth_features = self.occupancy_head(encoded)  # [B, depth_dim, H, W]
        
        # 构建3D体积 [B, 1, D, H, W]
        volume = depth_features.unsqueeze(1)
        
        # 占据概率
        occupancy_prob = self.occupancy_prob(volume)  # [B, 1, D, H, W]
        
        # 语义类别
        semantic_logits = self.semantic_head(volume)  # [B, num_classes, D, H, W]
        
        return {
            'occupancy_prob': occupancy_prob.squeeze(1),  # [B, D, H, W]
            'semantic_logits': semantic_logits,  # [B, num_classes, D, H, W]
            'volume_features': volume  # [B, 1, D, H, W]
        }


# =============================================================================
# Detection Head Module
# =============================================================================

class DetectionHead(nn.Module):
    """
    3D检测头模块
    预测目标的位置、尺寸、方向、速度等
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        
        # 特征金字塔网络
        self.fpn = self._build_fpn(config.bev_channels)
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, config.num_classes, 1)
        )
        
        # 中心点热图头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # 3D位置回归头 (x, y, z)
        self.pos_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)
        )
        
        # 尺寸回归头 (w, l, h)
        self.size_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)
        )
        
        # 方向回归头 (sin, cos)
        self.rot_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )
        
        # 速度回归头 (vx, vy)
        self.vel_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )
        
        # 加速度回归头 (ax, ay)
        self.acc_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )
        
        # 属性预测头 (车辆灯态等)
        self.attr_head = nn.Sequential(
            nn.Conv2d(config.bev_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 8, 1)  # 各种属性
        )
        
    def _build_fpn(self, in_channels: int) -> nn.Module:
        """构建特征金字塔网络"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        
    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_features: [B, C, H, W] BEV特征
        Returns:
            检测结果字典
        """
        # FPN特征
        fpn_features = self.fpn(bev_features)
        
        # 各类预测
        heatmap = self.heatmap_head(fpn_features)  # [B, 1, H, W]
        cls_logits = self.cls_head(fpn_features)  # [B, num_classes, H, W]
        pos_offset = self.pos_head(fpn_features)  # [B, 3, H, W]
        size = self.size_head(fpn_features)  # [B, 3, H, W]
        rotation = self.rot_head(fpn_features)  # [B, 2, H, W]
        velocity = self.vel_head(fpn_features)  # [B, 2, H, W]
        acceleration = self.acc_head(fpn_features)  # [B, 2, H, W]
        attributes = self.attr_head(fpn_features)  # [B, 8, H, W]
        
        return {
            'heatmap': heatmap,
            'cls_logits': cls_logits,
            'pos_offset': pos_offset,
            'size': size,
            'rotation': rotation,
            'velocity': velocity,
            'acceleration': acceleration,
            'attributes': attributes,
            'features': fpn_features
        }


# =============================================================================
# Dense World Tensor Module
# =============================================================================

class DenseWorldTensor(nn.Module):
    """
    稠密世界张量模块
    融合动态信息，构建世界表示
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        
        # 检测特征编码
        self.detection_encoder = nn.Sequential(
            nn.Conv2d(config.bev_channels + 3 + 3 + 2 + 2 + 8, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, config.d_model, 3, padding=1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU(),
        )
        
        # 自车运动编码
        self.ego_motion_encoder = nn.Sequential(
            nn.Linear(16, 128),  # 4x4矩阵展平
            nn.ReLU(),
            nn.Linear(128, config.d_model),
        )
        
        # 轨迹编码
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(config.num_future_frames * 3, 256),  # (x, y, yaw) per frame
            nn.ReLU(),
            nn.Linear(256, config.d_model),
        )
        
        # 交通灯状态编码
        self.traffic_light_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 4种状态
            nn.ReLU(),
            nn.Linear(64, config.d_model),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(config.d_model * 4, config.d_model, 1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU(),
        )
        
        # 动态信息编码器
        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(config.d_model, config.d_model, 3, padding=1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU(),
            nn.Conv2d(config.d_model, config.d_model, 3, padding=1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU(),
        )
        
    def forward(self, 
                detection_features: torch.Tensor,
                detection_output: Dict[str, torch.Tensor],
                ego_motion: torch.Tensor,
                candidate_trajectory: torch.Tensor,
                traffic_light_status: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detection_features: [B, C, H, W] 检测特征
            detection_output: 检测头输出
            ego_motion: [B, 4, 4] 自车运动
            candidate_trajectory: [B, T, 3] 候选轨迹
            traffic_light_status: [B, 4] 交通灯状态
        Returns:
            world_tensor: [B, d_model, H, W] 世界张量
        """
        B, C, H, W = detection_features.shape
        
        # 编码检测输出
        det_cat = torch.cat([
            detection_features,
            detection_output['pos_offset'],
            detection_output['size'],
            detection_output['velocity'],
            detection_output['acceleration'],
            detection_output['attributes']
        ], dim=1)
        det_encoded = self.detection_encoder(det_cat)  # [B, d_model, H, W]
        
        # 编码自车运动
        ego_encoded = self.ego_motion_encoder(ego_motion.view(B, -1))  # [B, d_model]
        ego_encoded = ego_encoded.view(B, self.config.d_model, 1, 1).expand(B, self.config.d_model, H, W)
        
        # 编码轨迹
        traj_encoded = self.trajectory_encoder(candidate_trajectory.view(B, -1))  # [B, d_model]
        traj_encoded = traj_encoded.view(B, self.config.d_model, 1, 1).expand(B, self.config.d_model, H, W)
        
        # 编码交通灯
        tl_encoded = self.traffic_light_encoder(traffic_light_status)  # [B, d_model]
        tl_encoded = tl_encoded.view(B, self.config.d_model, 1, 1).expand(B, self.config.d_model, H, W)
        
        # 融合所有信息
        fused = torch.cat([det_encoded, ego_encoded, traj_encoded, tl_encoded], dim=1)
        world_tensor = self.fusion(fused)
        
        # 动态信息编码
        world_tensor = self.dynamic_encoder(world_tensor)
        
        return world_tensor


# =============================================================================
# ROI Temporal Alignment Module
# =============================================================================

class ROITemporalAlignment(nn.Module):
    """
    ROI时序对齐模块 (Self-Attention)
    对检测到的目标进行精细化的时序特征融合
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # ROI特征提取
        self.roi_extractor = nn.Sequential(
            nn.Conv2d(config.d_model, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 时序自注意力
        self.temporal_attn = TransformerEncoder(
            num_layers=4,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # 行人姿态预测头
        self.pose_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_pose_keypoints * 3)  # (x, y, confidence) per keypoint
        )
        
        # 形状网格预测头
        self.mesh_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * config.mesh_resolution ** 3)  # 3D网格顶点
        )
        
        # 未来轨迹预测头
        self.future_traj_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.num_future_frames * config.trajectory_points * 3)  # (x, y, z) per point
        )
        
    def forward(self, world_tensor: torch.Tensor, rois: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            world_tensor: [B, d_model, H, W] 世界张量
            rois: [B, N, 4] 目标区域 (x1, y1, x2, y2)
        Returns:
            精细化预测结果
        """
        B, C, H, W = world_tensor.shape
        N = rois.shape[1]
        
        # 提取ROI特征
        roi_features = []
        for b in range(B):
            batch_rois = rois[b]  # [N, 4]
            # 将ROI坐标归一化到[-1, 1]
            grid = self._roi_to_grid(batch_rois, H, W)  # [N, 7, 7, 2]
            # 双线性插值采样
            sampled = F.grid_sample(world_tensor[b:b+1].expand(N, -1, -1, -1), 
                                   grid, align_corners=True)  # [N, C, 7, 7]
            roi_features.append(sampled)
        
        roi_features = torch.stack(roi_features, dim=0)  # [B, N, C, 7, 7]
        roi_features = roi_features.view(B * N, C, 7, 7)
        
        # 池化到特征向量
        roi_pooled = self.roi_extractor(roi_features).view(B, N, 256)
        
        # 时序自注意力 (这里假设每个ROI有历史信息)
        roi_aligned = self.temporal_attn(roi_pooled)
        
        # 预测各种输出
        B, N, _ = roi_aligned.shape
        roi_flat = roi_aligned.view(B * N, -1)
        
        # 行人姿态
        pose = self.pose_head(roi_flat).view(B, N, self.config.num_pose_keypoints, 3)
        
        # 形状网格
        mesh = self.mesh_head(roi_flat).view(B, N, 3, self.config.mesh_resolution, 
                                              self.config.mesh_resolution, self.config.mesh_resolution)
        
        # 未来轨迹
        future_traj = self.future_traj_head(roi_flat).view(
            B, N, self.config.num_future_frames, self.config.trajectory_points, 3
        )
        
        return {
            'roi_features': roi_aligned,
            'pedestrian_pose': pose,
            'shape_mesh': mesh,
            'future_trajectory': future_traj
        }
    
    def _roi_to_grid(self, rois: torch.Tensor, H: int, W: int, grid_size: int = 7) -> torch.Tensor:
        """将ROI转换为采样网格"""
        N = rois.shape[0]
        device = rois.device
        
        # 创建归一化网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, grid_size, device=device),
            torch.linspace(-1, 1, grid_size, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)
        
        # 根据ROI缩放网格
        x1, y1, x2, y2 = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3]
        
        # 归一化到[-1, 1]
        x1_norm = 2 * x1 / W - 1
        x2_norm = 2 * x2 / W - 1
        y1_norm = 2 * y1 / H - 1
        y2_norm = 2 * y2 / H - 1
        
        # 缩放网格
        grid_x_scaled = x1_norm.view(N, 1, 1) + (grid[..., 0] + 1) / 2 * (x2_norm - x1_norm).view(N, 1, 1)
        grid_y_scaled = y1_norm.view(N, 1, 1) + (grid[..., 1] + 1) / 2 * (y2_norm - y1_norm).view(N, 1, 1)
        
        grid_scaled = torch.stack([grid_x_scaled, grid_y_scaled], dim=-1)
        return grid_scaled


# =============================================================================
# 完整的障碍物检测网络
# =============================================================================

class ObjectDetectionNet(nn.Module):
    """
    完整的3D障碍物检测网络
    基于BEV的时序检测网络
    """
    def __init__(self, config: Optional[ObjectDetectionConfig] = None):
        super().__init__()
        self.config = config or ObjectDetectionConfig()
        
        # 时序对齐模块
        self.temporal_alignment = TemporalAlignment(self.config)
        
        # 占据网络
        self.occupancy_net = OccupancyNet(self.config)
        
        # 检测头
        self.detection_head = DetectionHead(self.config)
        
        # 稠密世界张量
        self.world_tensor = DenseWorldTensor(self.config)
        
        # ROI时序对齐
        self.roi_alignment = ROITemporalAlignment(self.config)
        
    def forward(self, 
                bev_layers: torch.Tensor,
                ego_motion: torch.Tensor,
                candidate_trajectory: torch.Tensor,
                traffic_light_status: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            bev_layers: [B, T, C, H, W] 历史BEV特征
            ego_motion: [B, T, 4, 4] 自车运动矩阵
            candidate_trajectory: [B, T_future, 3] 候选轨迹
            traffic_light_status: [B, 4] 交通灯状态
            
        Returns:
            完整的检测结果
        """
        # 1. 时序对齐
        aligned_bev = self.temporal_alignment(bev_layers, ego_motion)
        
        # 2. 占据网络
        occupancy_output = self.occupancy_net(aligned_bev)
        
        # 3. 检测头
        detection_output = self.detection_head(aligned_bev)
        
        # 4. 稠密世界张量
        world_tensor = self.world_tensor(
            detection_output['features'],
            detection_output,
            ego_motion[:, -1],  # 使用最新帧的自车运动
            candidate_trajectory,
            traffic_light_status
        )
        
        # 5. 从热图提取ROI
        rois = self._extract_rois_from_heatmap(detection_output['heatmap'])
        
        # 6. ROI时序对齐
        roi_output = self.roi_alignment(world_tensor, rois)
        
        # 组合所有输出
        output = {
            # 占据网络输出
            'occupancy_prob': occupancy_output['occupancy_prob'],
            'occupancy_semantic': occupancy_output['semantic_logits'],
            
            # 检测头输出
            'heatmap': detection_output['heatmap'],
            'cls_logits': detection_output['cls_logits'],
            'pos_offset': detection_output['pos_offset'],
            'size': detection_output['size'],
            'rotation': detection_output['rotation'],
            'velocity': detection_output['velocity'],
            'acceleration': detection_output['acceleration'],
            'attributes': detection_output['attributes'],
            
            # ROI精细化输出
            'pedestrian_pose': roi_output['pedestrian_pose'],
            'shape_mesh': roi_output['shape_mesh'],
            'future_trajectory': roi_output['future_trajectory'],
            
            # 中间特征
            'world_tensor': world_tensor,
            'rois': rois
        }
        
        return output
    
    def _extract_rois_from_heatmap(self, heatmap: torch.Tensor, topk: int = 100) -> torch.Tensor:
        """从热图提取ROI区域"""
        B, _, H, W = heatmap.shape
        device = heatmap.device
        
        # 使用NMS提取峰值
        heatmap_flat = heatmap.view(B, -1)
        scores, indices = torch.topk(heatmap_flat, topk, dim=1)
        
        # 转换回2D坐标
        ys = (indices // W).float()
        xs = (indices % W).float()
        
        # 构建ROI (x1, y1, x2, y2)
        roi_size = 7  # ROI大小
        x1 = torch.clamp(xs - roi_size / 2, 0, W - 1)
        y1 = torch.clamp(ys - roi_size / 2, 0, H - 1)
        x2 = torch.clamp(xs + roi_size / 2, 0, W - 1)
        y2 = torch.clamp(ys + roi_size / 2, 0, H - 1)
        
        rois = torch.stack([x1, y1, x2, y2], dim=-1)
        return rois


# =============================================================================
# nuScenes数据集支持
# =============================================================================

class nuScenesObject:
    """nuScenes目标格式"""
    def __init__(self):
        self.token: str = ""  # 唯一标识
        self.sample_token: str = ""  # 样本标识
        self.translation: np.ndarray = np.zeros(3)  # (x, y, z)
        self.size: np.ndarray = np.zeros(3)  # (w, l, h)
        self.rotation: np.ndarray = np.zeros(4)  # 四元数
        self.velocity: np.ndarray = np.zeros(2)  # (vx, vy)
        self.acceleration: np.ndarray = np.zeros(2)  # (ax, ay)
        self.category: str = ""
        self.attributes: Dict[str, bool] = {}
        self.pose_keypoints: Optional[np.ndarray] = None  # 行人姿态
        self.future_trajectory: Optional[np.ndarray] = None  # 未来轨迹
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'token': self.token,
            'sample_token': self.sample_token,
            'translation': self.translation.tolist(),
            'size': self.size.tolist(),
            'rotation': self.rotation.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'category': self.category,
            'attributes': self.attributes,
            'pose_keypoints': self.pose_keypoints.tolist() if self.pose_keypoints is not None else None,
            'future_trajectory': self.future_trajectory.tolist() if self.future_trajectory is not None else None
        }


class nuScenesConverter:
    """nuScenes格式转换器"""
    
    # nuScenes类别映射
    CLASS_MAPPING = {
        0: 'car',
        1: 'truck',
        2: 'bus',
        3: 'trailer',
        4: 'construction_vehicle',
        5: 'pedestrian',
        6: 'motorcycle',
        7: 'bicycle',
        8: 'traffic_cone',
        9: 'barrier'
    }
    
    # 属性映射
    ATTR_MAPPING = {
        'vehicle_light': ['off', 'on', 'blinking'],
        'pedestrian_pose': ['standing', 'walking', 'sitting', 'lying'],
        'cycle_rider': [False, True]
    }
    
    @staticmethod
    def network_output_to_nuscenes(network_output: Dict[str, torch.Tensor], 
                                    config: ObjectDetectionConfig) -> List[nuScenesObject]:
        """将网络输出转换为nuScenes格式"""
        batch_size = network_output['heatmap'].shape[0]
        results = []
        
        for b in range(batch_size):
            objects = nuScenesConverter._process_single_frame(
                {k: v[b] for k, v in network_output.items()},
                config
            )
            results.append(objects)
        
        return results
    
    @staticmethod
    def _process_single_frame(output: Dict[str, torch.Tensor], 
                               config: ObjectDetectionConfig) -> List[nuScenesObject]:
        """处理单帧输出"""
        objects = []
        
        # 从热图提取目标
        heatmap = output['heatmap'].squeeze(0)  # [H, W]
        H, W = heatmap.shape
        
        # 找到峰值位置
        peaks = (heatmap > 0.3).nonzero(as_tuple=False)  # 阈值0.3
        
        for peak in peaks:
            y, x = peak[0].item(), peak[1].item()
            
            obj = nuScenesObject()
            
            # 提取位置
            pos_offset = output['pos_offset'][:, y, x].cpu().numpy()
            obj.translation = np.array([
                (x / W - 0.5) * 100 + pos_offset[0],  # 假设BEV范围100m
                (y / H - 0.5) * 100 + pos_offset[1],
                pos_offset[2]
            ])
            
            # 提取尺寸
            obj.size = output['size'][:, y, x].cpu().numpy()
            
            # 提取旋转
            rot = output['rotation'][:, y, x].cpu().numpy()
            angle = np.arctan2(rot[0], rot[1])
            obj.rotation = nuScenesConverter._angle_to_quaternion(angle)
            
            # 提取速度
            obj.velocity = output['velocity'][:, y, x].cpu().numpy()
            
            # 提取加速度
            obj.acceleration = output['acceleration'][:, y, x].cpu().numpy()
            
            # 提取类别
            cls_logits = output['cls_logits'][:, y, x]
            cls_id = cls_logits.argmax().item()
            obj.category = nuScenesConverter.CLASS_MAPPING.get(cls_id, 'unknown')
            
            # 提取属性
            attrs = output['attributes'][:, y, x].cpu().numpy()
            obj.attributes = {
                'vehicle_light': nuScenesConverter.ATTR_MAPPING['vehicle_light'][int(attrs[0]) % 3],
                'moving': bool(attrs[1] > 0.5),
                'parked': bool(attrs[2] > 0.5)
            }
            
            # 提取行人姿态 (如果是行人)
            if obj.category == 'pedestrian' and 'pedestrian_pose' in output:
                obj.pedestrian_pose = output['pedestrian_pose'].cpu().numpy()
            
            # 提取未来轨迹
            if 'future_trajectory' in output:
                obj.future_trajectory = output['future_trajectory'][0].cpu().numpy()
            
            objects.append(obj)
        
        return objects
    
    @staticmethod
    def _angle_to_quaternion(angle: float) -> np.ndarray:
        """将角度转换为四元数 (绕z轴旋转)"""
        return np.array([
            np.cos(angle / 2),
            0,
            0,
            np.sin(angle / 2)
        ])


# =============================================================================
# 测试接口
# =============================================================================

def test_object_detection():
    """
    障碍物检测模块测试接口
    测试完整的检测流程
    """
    print("=" * 60)
    print("障碍物感知模块测试")
    print("=" * 60)
    
    # 创建配置
    config = ObjectDetectionConfig()
    
    # 创建网络
    print("\n[1] 创建ObjectDetectionNet网络...")
    model = ObjectDetectionNet(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    总参数量: {total_params:,}")
    print(f"    可训练参数量: {trainable_params:,}")
    
    # 创建测试输入
    print("\n[2] 创建测试输入...")
    batch_size = 2
    bev_layers = torch.randn(batch_size, config.num_frames, config.bev_channels, 
                              config.bev_height, config.bev_width)
    ego_motion = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, config.num_frames, -1, -1)
    candidate_trajectory = torch.randn(batch_size, config.num_future_frames, 3)
    traffic_light_status = torch.randn(batch_size, 4)
    
    print(f"    BEV Layers: {bev_layers.shape}")
    print(f"    Ego Motion: {ego_motion.shape}")
    print(f"    Candidate Trajectory: {candidate_trajectory.shape}")
    print(f"    Traffic Light Status: {traffic_light_status.shape}")
    
    # 前向传播
    print("\n[3] 执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(bev_layers, ego_motion, candidate_trajectory, traffic_light_status)
    
    # 检查输出
    print("\n[4] 检查输出...")
    expected_outputs = [
        'occupancy_prob',
        'occupancy_semantic',
        'heatmap',
        'cls_logits',
        'pos_offset',
        'size',
        'rotation',
        'velocity',
        'acceleration',
        'attributes',
        'pedestrian_pose',
        'shape_mesh',
        'future_trajectory',
        'world_tensor',
        'rois'
    ]
    
    for key in expected_outputs:
        if key in output:
            print(f"    {key}: {output[key].shape}")
        else:
            print(f"    {key}: MISSING!")
    
    # 测试nuScenes格式转换
    print("\n[5] 测试nuScenes格式转换...")
    nuscenes_objects = nuScenesConverter.network_output_to_nuscenes(output, config)
    print(f"    检测到 {len(nuscenes_objects[0])} 个目标")
    
    if len(nuscenes_objects[0]) > 0:
        obj = nuscenes_objects[0][0]
        print(f"    示例目标:")
        print(f"      - 类别: {obj.category}")
        print(f"      - 位置: {obj.translation}")
        print(f"      - 尺寸: {obj.size}")
        print(f"      - 速度: {obj.velocity}")
        print(f"      - 加速度: {obj.acceleration}")
    
    # 测试各个子模块
    print("\n[6] 测试子模块...")
    
    # Temporal Alignment
    print("    测试 Temporal Alignment...")
    temporal_aligned = model.temporal_alignment(bev_layers, ego_motion)
    print(f"      输出形状: {temporal_aligned.shape}")
    
    # Occupancy Net
    print("    测试 Occupancy Net...")
    occupancy_out = model.occupancy_net(temporal_aligned)
    print(f"      occupancy_prob: {occupancy_out['occupancy_prob'].shape}")
    print(f"      semantic_logits: {occupancy_out['semantic_logits'].shape}")
    
    # Detection Head
    print("    测试 Detection Head...")
    detection_out = model.detection_head(temporal_aligned)
    print(f"      heatmap: {detection_out['heatmap'].shape}")
    print(f"      velocity: {detection_out['velocity'].shape}")
    
    # Dense World Tensor
    print("    测试 Dense World Tensor...")
    world_tensor = model.world_tensor(
        detection_out['features'],
        detection_out,
        ego_motion[:, -1],
        candidate_trajectory,
        traffic_light_status
    )
    print(f"      输出形状: {world_tensor.shape}")
    
    # ROI Temporal Alignment
    print("    测试 ROI Temporal Alignment...")
    rois = model._extract_rois_from_heatmap(detection_out['heatmap'])
    roi_out = model.roi_alignment(world_tensor, rois)
    print(f"      pedestrian_pose: {roi_out['pedestrian_pose'].shape}")
    print(f"      shape_mesh: {roi_out['shape_mesh'].shape}")
    print(f"      future_trajectory: {roi_out['future_trajectory'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！所有模块运行正常。")
    print("=" * 60)
    
    return model, output


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # 运行测试
    model, output = test_object_detection()
