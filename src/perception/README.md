# Lane Detection Module

基于Transformer的车道线检测网络实现

## 功能特性

- **Vector Lane Encoding**: 使用Self-Attention和Cross-Attention编码车道线特征
- **Point Prediction**: 两级点预测器(Level1粗预测 + Level2精细优化)
- **Topology Type Prediction**: 预测车道线之间的拓扑关系类型
- **Fork/Merge Point Prediction**: 检测车道线的分叉和合并点
- **Spline Coefficient Prediction**: B样条曲线系数预测，实现平滑车道线表示
- **nuScenes数据集支持**: 兼容nuScenes数据格式

## 算法流程

```
输入: Dense World Tensor (Static Info)
    ↓
Vector Lane Encoding
    ├── Self-Attention
    ├── Cross-Attention (with BEV features)
    └── Self-Attention
    ↓
预测头:
    ├── Point Predictor (Level1/Level2)
    ├── Topology Type Predictor
    ├── Fork Point Predictor
    ├── Merge Point Predictor
    └── Spline Coefficient Predictor
    ↓
输出: Lane Instances, Adjacency Matrix, Topology Types, Fork/Merge Points, Spline Coeffs
```

## 文件结构

```
autonomous_driving/
├── src/perception/
│   ├── lane_detection.py          # 主模块实现
│   ├── lane_detection_example.py  # 使用示例
│   └── README.md                   # 本文档
├── config/
│   └── lane_config.yaml           # 配置文件
```

## 快速开始

### 1. 基本使用

```python
from lane_detection import LanePerceptionModule
import torch

# 初始化模块
config_path = "config/lane_config.yaml"
module = LanePerceptionModule(config_path)

# 准备输入数据
lane_vectors = torch.randn(1, 5, 20, 3)  # [B, N, P, 3]
lane_types = torch.randint(0, 5, (1, 5))  # [B, N]
bev_image = torch.randn(1, 3, 512, 512)   # [B, 3, H, W]

# 执行检测
output = module.detect(
    lane_vectors=lane_vectors,
    lane_types=lane_types,
    bev_image=bev_image,
    timestamp=123.456
)

# 处理结果
print(f"Detected {len(output.lane_instances)} lane instances")
print(f"Adjacency matrix: {output.adjacency_matrix.shape}")
```

### 2. 运行测试

```python
from lane_detection import test_lane_detection

# 运行完整测试
results = test_lane_detection("config/lane_config.yaml")
```

### 3. 运行示例

```bash
cd src/perception
python lane_detection_example.py
```

## 输出格式

### LaneInstance
```python
@dataclass
class LaneInstance:
    lane_id: int              # 车道线ID
    points: List[LanePoint]   # 车道线点链 [(x, y, z), ...]
    lane_type: str            # 类型: "solid", "dashed", "double", etc.
    color: str                # 颜色: "white", "yellow", etc.
    confidence: float         # 置信度
    spline_coeffs: Tensor     # B样条系数
```

### LaneDetectionOutput
```python
@dataclass
class LaneDetectionOutput:
    lane_instances: List[LaneInstance]  # 车道线实例列表
    adjacency_matrix: Tensor            # 邻接矩阵 [N, N]
    topology_types: List[TopologyType]  # 拓扑类型列表
    fork_points: List[LanePoint]        # 分叉点列表
    merge_points: List[LanePoint]       # 合并点列表
    timestamp: float                    # 时间戳
```

## 拓扑类型

| ID | 类型 | 描述 |
|----|------|------|
| 0 | LANE_FOLLOW | 车道跟随 |
| 1 | LEFT_TURN | 左转 |
| 2 | RIGHT_TURN | 右转 |
| 3 | STRAIGHT | 直行 |
| 4 | U_TURN | 掉头 |
| 5 | MERGE | 车道合并 |
| 6 | FORK | 车道分叉 |

## 配置参数

### 模型架构
- `d_model`: 特征维度 (默认: 256)
- `num_heads`: 注意力头数 (默认: 8)
- `num_encoder_layers`: 编码器层数 (默认: 6)
- `num_points_per_lane`: 每条车道线的点数 (默认: 20)
- `num_control_points`: B样条控制点数 (默认: 10)

### 后处理
- `topology_threshold`: 拓扑关系置信度阈值 (默认: 0.5)
- `fork_merge_threshold`: 分叉/合并点置信度阈值 (默认: 0.3)

## 网络组件

### VectorLaneEncoder
将车道线向量编码为特征表示
- 输入: `[B, N, P, 3]` - 车道线点坐标
- 输出: `[B, N, d_model]` - 车道线特征

### PointPredictor
两级点预测器
- Level1: 粗略位置预测
- Level2: 精细位置优化
- 输出: `[B, N, P, 3]`

### TopologyTypePredictor
预测车道线之间的拓扑关系
- 输出: `[B, N, N, num_topology_types]`

### ForkMergePointPredictor
检测分叉和合并点
- 输出: `[B, N, 4]` - (x, y, z, confidence)

### SplineCoefficientPredictor
预测B样条曲线系数
- 输出: `[B, N, num_control_points, 3]`

## 独立组件使用

```python
from lane_detection import (
    VectorLaneEncoder,
    PointPredictor,
    TopologyTypePredictor
)

config = {'d_model': 256, 'num_heads': 8, ...}

# 单独使用编码器
encoder = VectorLaneEncoder(config)
lane_features = encoder(lane_vectors)

# 单独使用预测器
point_pred = PointPredictor(config)
level1, level2 = point_pred(lane_features)
```

## 依赖项

- PyTorch >= 1.9.0
- NumPy
- PyYAML

## 许可证

MIT License
