# Autonomous Driving Planning Module

自动驾驶规划模块 - 基于Lattice网格撒点的轨迹规划系统

## 功能概述

本规划模块实现了完整的自动驾驶轨迹规划流程，包括：

1. **Behavior & Trajectory Generation** - 行为轨迹生成
   - 基于Lattice网格撒点生成候选轨迹
   - 支持多种行为类型：车道保持、换道、加速、减速
   - 使用五次多项式进行平滑轨迹规划

2. **Behavior Explanation** - 行为解释
   - 对生成的轨迹进行行为解析
   - 多维度评分：安全性、舒适性、效率、合法性
   - 风险因素识别和建议生成

3. **Behavior & Trajectory Selection** - 轨迹选择
   - 基于加权损失函数的轨迹选择
   - 支持用户偏好模型
   - 考虑轨迹一致性

4. **Trajectory Optimization** - 轨迹优化
   - 样条平滑处理
   - 安全距离调整
   - 运动学约束优化
   - 速度曲线优化

## 项目结构

```
planning/
├── __init__.py                 # 模块初始化
├── planning_module.py          # 主规划模块
├── lattice_generator.py        # Lattice轨迹生成器
├── behavior_explainer.py       # 行为解释器
├── trajectory_selector.py      # 轨迹选择器
├── trajectory_optimizer.py     # 轨迹优化器
├── visualize_planning.py       # 可视化工具
├── config/
│   └── planning_config.yaml    # 配置文件
└── README.md                   # 说明文档
```

## 安装依赖

```bash
pip install numpy scipy matplotlib pyyaml
```

## 快速开始

### 1. 基本使用

```python
from planning import PlanningModule, PlanningInput
from lattice_generator import VehicleState
import numpy as np

# 创建规划模块
planner = PlanningModule(config_path='config/planning_config.yaml')

# 设置自车状态
ego_state = VehicleState(
    x=0.0,      # 位置x
    y=0.0,      # 位置y
    theta=0.0,  # 航向角
    v=10.0,     # 速度
    a=0.0       # 加速度
)

# 设置障碍物
objects = [
    {
        'x': 30.0, 'y': 0.0,
        'vx': 8.0, 'vy': 0.0,
        'category': 'vehicle',
        'width': 2.0, 'length': 4.5
    }
]

# 设置车道
lane_x = np.linspace(0, 100, 100)
lane_y = np.zeros(100)
lanes = [{
    'x': lane_x.tolist(),
    'y': lane_y.tolist(),
    'width': 3.5
}]

# 创建规划输入
planning_input = PlanningInput(
    ego_state=ego_state,
    objects=objects,
    occupancy=[],
    lanes=lanes,
    traffic_signs=[],
    timestamp=0.0
)

# 执行规划
output = planner.plan(planning_input)

# 获取结果
print(f"Selected behavior: {output.selected_behavior}")
print(f"Safety score: {output.explanation.safety_score}")
print(f"Trajectory length: {output.trajectory.get_length()} m")
```

### 2. 指定行为偏好

```python
# 指定行为偏好进行规划
output = planner.plan(planning_input, behavior_preference="change_left")
```

支持的行为类型：
- `keep_lane` - 车道保持
- `change_left` - 向左换道
- `change_right` - 向右换道
- `accelerate` - 加速
- `decelerate` - 减速

### 3. 格式化输出

```python
from planning import format_output_for_control

# 转换为控制模块格式
control_output = format_output_for_control(output)

# 输出格式: (x, y, time, velocity)
print(control_output['trajectory']['x'])       # x坐标序列
print(control_output['trajectory']['y'])       # y坐标序列
print(control_output['trajectory']['time'])    # 时间序列
print(control_output['trajectory']['velocity']) # 速度序列
```

## nuScenes数据集支持

```python
from planning import convert_nuscenes_to_planning_input

# 转换nuScenes数据
nuscenes_data = {...}  # nuScenes原始数据
ego_pose = {...}       # 自车姿态

planning_input = convert_nuscenes_to_planning_input(
    nuscenes_data, ego_pose, timestamp=0.0
)

output = planner.plan(planning_input)
```

## 可视化

```python
from visualize_planning import visualize_planning_output
from lattice_generator import Obstacle, LaneInfo

# 准备数据
obstacles = [Obstacle(x=30, y=0, vx=8, vy=0, category='vehicle')]
lane_info = LaneInfo(x=lane_x, y=lane_y, width=3.5)

# 可视化
visualize_planning_output(output, obstacles, lane_info, 
                         save_path='output.png')
```

## 配置说明

配置文件 `config/planning_config.yaml` 包含以下主要参数：

### Lattice采样参数
- `num_t_samples`: 时间采样数
- `num_d_samples`: 横向偏移采样数
- `num_v_samples`: 速度采样数
- `t_min/t_max`: 预测时间范围
- `d_min/d_max`: 横向偏移范围
- `v_min/v_max`: 速度范围

### 运动学约束
- `max_acceleration`: 最大加速度 (m/s²)
- `max_deceleration`: 最大减速度 (m/s²)
- `max_curvature`: 最大曲率 (1/m)
- `max_jerk`: 最大加加速度 (m/s³)

### 评分权重
- `safety_weight`: 安全性权重
- `comfort_weight`: 舒适性权重
- `efficiency_weight`: 效率权重
- `legality_weight`: 合法性权重

## 测试

```bash
# 运行测试
python planning_module.py

# 创建可视化演示
python visualize_planning.py
```

## 算法流程

```
输入: 自车状态、障碍物、车道信息、交通标志
  ↓
1. Behavior & Trajectory Generation
   - Lattice网格撒点
   - 生成候选轨迹
   - 初步筛选
  ↓
2. Behavior Explanation
   - 行为分类
   - 多维度评分
   - 风险识别
  ↓
3. Behavior & Trajectory Selection
   - 损失函数计算
   - 轨迹排序
   - 最优选择
  ↓
4. Trajectory Optimization
   - 平滑处理
   - 安全调整
   - 运动学约束优化
  ↓
输出: 最优轨迹 (x, y, time, velocity)
```

## 性能指标

- 规划频率: 10 Hz
- 规划时域: 3-8 秒
- 候选轨迹数: 20-50 条
- 平均规划时间: < 100 ms

## 车辆运动学模型

使用自行车模型(Bicycle Model)进行运动学约束：

```
ẋ = v * cos(θ)
ẏ = v * sin(θ)
θ̇ = v * tan(δ) / L
v̇ = a
```

其中：
- (x, y): 车辆位置
- θ: 航向角
- v: 速度
- δ: 转向角
- L: 轴距
- a: 加速度

## 轨迹表示

轨迹使用以下参数表示：
- `x`: x坐标序列
- `y`: y坐标序列
- `theta`: 航向角序列
- `v`: 速度序列
- `a`: 加速度序列
- `kappa`: 曲率序列
- `t`: 时间序列

## 输出格式

控制模块输入格式：
```python
{
    'trajectory': {
        'x': [...],          # x坐标
        'y': [...],          # y坐标
        'time': [...],       # 时间
        'velocity': [...],   # 速度
        'acceleration': [...],
        'heading': [...],
        'curvature': [...]
    },
    'behavior': {
        'type': 'keep_lane',
        'safety_score': 0.85,
        'comfort_score': 0.78,
        'efficiency_score': 0.82,
        'total_score': 0.82
    },
    'planning_info': {
        'planning_time': 0.05,
        'num_candidates': 35
    }
}
```

## 许可证

MIT License
