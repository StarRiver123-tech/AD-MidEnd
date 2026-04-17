# 自动驾驶系统 (Autonomous Driving System)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个模块化的自动驾驶系统，支持仿真模式、nuScenes数据集模式和实车模式。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    自动驾驶系统 (ADS)                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   感知模块    │  │   规划模块    │  │   控制模块    │          │
│  │  Perception  │→│   Planning   │→│    Control   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↑                                    ↓                  │
│  ┌──────────────┐                  ┌──────────────┐            │
│  │   传感器层    │                  │   执行机构    │            │
│  │   Sensors    │                  │   Actuators  │            │
│  └──────────────┘                  └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   消息总线    │  │   配置管理    │  │   日志系统    │          │
│  │ Message Bus  │  │    Config    │  │    Logger    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 功能特性

### 感知模块 (Perception)
- **车道线检测**: 基于深度学习和传统图像处理的车道线检测
- **障碍物检测**: 多类别目标检测 (车辆、行人、骑行者等)
- **Occupancy网络**: 3D空间占据预测
- **多传感器融合**: 摄像头、LiDAR、雷达数据融合

### 规划模块 (Planning)
- **行为规划**: 决策制定和行为选择
- **轨迹生成**: Lattice-based轨迹采样
- **轨迹优化**: 基于约束的轨迹优化
- **轨迹选择**: 多目标评估和最优轨迹选择

### 传感器模块 (Sensors)
- **摄像头**: 多路摄像头支持 (前视、后视、环视)
- **LiDAR**: 点云数据采集和处理
- **雷达**: 毫米波雷达数据
- **超声波**: 近距离障碍物检测
- **CAN总线**: 车辆状态读取

### 可视化模块 (Visualization)
- **BEV可视化**: 鸟瞰图显示
- **轨迹可视化**: 规划轨迹显示
- **传感器可视化**: 传感器数据实时显示
- **交互式界面**: PyQt5图形界面

## 安装

### 环境要求
- Python 3.8+
- CUDA 11.0+ (GPU加速可选)
- 8GB+ RAM

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/your-repo/autonomous-driving.git
cd autonomous-driving
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 安装可选依赖 (nuScenes数据集支持)
```bash
pip install nuscenes-devkit pyquaternion
```

## 快速开始

### 1. 系统检查
```bash
python run_system.py --skip-checks
```

### 2. 运行仿真模式
```bash
# 快速启动
python run_system.py

# 或使用主程序
python main.py --mode simulation
```

### 3. 运行nuScenes数据集模式
```bash
# 需要下载nuScenes数据集
python run_system.py --mode nuscenes --data-root /path/to/nuscenes
```

## 使用指南

### 命令行参数

#### 主程序 (main.py)
```bash
python main.py [选项]

选项:
  --config, -c        配置文件路径 (默认: config/system_config.yaml)
  --mode, -m          运行模式: simulation | nuscenes | real_vehicle
  --data-root, -d     nuScenes数据集路径
  --log-level         日志级别: DEBUG | INFO | WARNING | ERROR
  --no-viz            禁用可视化
  --module-test       运行模块测试
```

#### 启动脚本 (run_system.py)
```bash
python run_system.py [选项]

选项:
  --mode, -m          运行模式
  --test-module       运行指定模块测试
  --benchmark         运行性能测试
  --run-tests         运行测试套件
  --generate-report   生成系统报告
```

### 运行模式

#### 仿真模式 (Simulation)
```bash
# 基本用法
python main.py --mode simulation

# 自定义配置
python main.py --mode simulation --config config/my_config.yaml

# 禁用可视化
python main.py --mode simulation --no-viz
```

#### nuScenes数据集模式
```bash
# 指定数据集路径
python main.py --mode nuscenes --data-root /data/nuscenes

# 使用配置文件中的路径
python main.py --mode nuscenes
```

#### 实车模式 (Real Vehicle)
```bash
# ⚠️ 警告: 此模式会控制真实车辆!
python main.py --mode real_vehicle
```

### 模块测试

#### 测试单个模块
```bash
# 测试感知模块
python test_module.py --module perception

# 测试规划模块
python test_module.py --module planning

# 测试传感器模块
python test_module.py --module sensor

# 测试可视化模块
python test_module.py --module visualization
```

#### 测试所有模块
```bash
python test_module.py --module all

# 保存测试结果
python test_module.py --module all --output test_results.json
```

#### 使用启动脚本测试
```bash
python run_system.py --test-module perception
```

### 性能测试

```bash
# 测试所有模块性能
python run_system.py --benchmark --duration 60

# 测试指定模块
python run_system.py --benchmark --modules perception planning --duration 120
```

### 可视化工具

```bash
# 仅运行可视化
python run_system.py --visualization-only

# 或通过可视化模块
python src/visualization/run_visualization.py
```

## 配置文件

### 系统配置 (config/system_config.yaml)

```yaml
system:
  name: "Autonomous Driving System"
  version: "1.0.0"
  mode: "simulation"  # simulation | nuscenes | real_vehicle
  log_level: "INFO"
  main_loop_frequency: 50  # Hz

modules:
  perception:
    enabled: true
    execution_frequency: 10  # Hz
    algorithm_params:
      lane_detection:
        enabled: true
        confidence_threshold: 0.5
      obstacle_detection:
        enabled: true
        confidence_threshold: 0.6
      occupancy:
        enabled: true
        grid_resolution: 0.2

  planning:
    enabled: true
    execution_frequency: 10  # Hz
    algorithm_params:
      behavior_planning:
        enabled: true
        planning_horizon: 100.0
      trajectory_generation:
        enabled: true
        num_trajectories: 50
      trajectory_optimization:
        enabled: true
        max_iterations: 100

sensors:
  cameras:
    front_camera:
      enabled: true
      resolution: [1920, 1080]
      fps: 30
  lidars:
    main_lidar:
      enabled: true
      channels: 64
      range: 200.0
```

## 数据流

```
传感器数据 → 消息总线 → 感知模块 → 消息总线 → 规划模块 → 消息总线 → 控制模块
    ↓                                                            ↓
    └──────────────────→ 可视化模块 ←────────────────────────────┘
```

### 消息主题

| 主题 | 描述 | 发布者 |
|------|------|--------|
| sensor/camera/front | 前视摄像头数据 | 传感器模块 |
| sensor/lidar | LiDAR点云数据 | 传感器模块 |
| sensor/radar | 雷达数据 | 传感器模块 |
| sensor/can/vehicle | 车辆状态数据 | 传感器模块 |
| perception/lane | 车道线检测结果 | 感知模块 |
| perception/obstacle | 障碍物检测结果 | 感知模块 |
| perception/occupancy | Occupancy网格 | 感知模块 |
| perception/fusion | 融合感知结果 | 感知模块 |
| planning/trajectory | 规划轨迹 | 规划模块 |
| planning/behavior | 行为决策 | 规划模块 |
| planning/all | 完整规划结果 | 规划模块 |

## 模块说明

### 感知模块

#### 车道线检测
- 文件: `src/perception/lane_detector.py`
- 功能: 检测车道线、计算车道曲率、估计车道宽度

#### 障碍物检测
- 文件: `src/perception/obstacle_detector.py`
- 功能: 多类别目标检测、跟踪、速度估计

#### Occupancy网络
- 文件: `src/perception/occupancy_network.py`
- 功能: 3D空间占据预测、可通行区域检测

### 规划模块

#### 行为规划
- 文件: `src/planning/behavior_planner.py`
- 功能: 决策制定、行为解释

#### 轨迹生成
- 文件: `src/planning/trajectory_generator.py`
- 功能: Lattice采样、候选轨迹生成

#### 轨迹优化
- 文件: `src/planning/trajectory_optimizer.py`
- 功能: 约束优化、平滑处理

### 传感器模块

#### 传感器管理器
- 文件: `src/sensors/core/sensor_manager.py`
- 功能: 统一管理所有传感器

#### 同步管理器
- 文件: `src/sensors/core/sync_manager.py`
- 功能: 多传感器时间同步

## API参考

### 系统类

```python
from main import AutonomousDrivingSystem

# 创建系统实例
system = AutonomousDrivingSystem()

# 初始化系统
system.initialize(
    config_path='config/system_config.yaml',
    mode='simulation'
)

# 启动系统
system.start()

# 运行主循环
system.run()

# 停止系统
system.stop()

# 获取系统状态
status = system.get_status()
```

### 感知模块

```python
from src.perception.perception_module import PerceptionModule

# 创建感知模块
perception = PerceptionModule(message_bus)

# 初始化
perception.initialize(config_manager)

# 启动
perception.start()

# 停止
perception.stop()

# 获取统计信息
stats = perception.get_stats()
```

### 规划模块

```python
from src.planning.planning_module import PlanningModule

# 创建规划模块
planning = PlanningModule(message_bus)

# 初始化
planning.initialize(config_manager)

# 启动
planning.start()

# 停止
planning.stop()

# 获取当前规划结果
result = planning.get_current_planning_result()
```

## 测试

### 运行测试套件
```bash
# 使用pytest
pytest tests/ -v

# 使用启动脚本
python run_system.py --run-tests

# 生成覆盖率报告
python run_system.py --run-tests --coverage
```

### 测试结构
```
tests/
├── unit/                 # 单元测试
│   ├── test_perception.py
│   ├── test_planning.py
│   └── test_sensor.py
├── integration/          # 集成测试
├── simulation/           # 仿真测试
├── metrics/              # 性能指标
└── run_tests.py          # 测试运行器
```

## 故障排除

### 常见问题

#### 1. 导入错误
```
ModuleNotFoundError: No module named 'xxx'
```
**解决方案**: 安装缺失的依赖
```bash
pip install -r requirements.txt
```

#### 2. 配置文件未找到
```
Config file not found: config/system_config.yaml
```
**解决方案**: 使用默认配置或指定正确路径
```bash
python main.py --config /path/to/config.yaml
```

#### 3. nuScenes数据集未找到
```
nuScenes data root not found: data/nuscenes
```
**解决方案**: 下载数据集并指定正确路径
```bash
python main.py --mode nuscenes --data-root /path/to/nuscenes
```

#### 4. 可视化无法启动
```
Visualization not available (PyQt5 not installed)
```
**解决方案**: 安装PyQt5
```bash
pip install PyQt5
```

### 调试模式

```bash
# 启用DEBUG日志
python main.py --log-level DEBUG

# 详细模块测试
python test_module.py --module perception --verbose
```

## 性能优化

### 1. GPU加速
```yaml
modules:
  perception:
    algorithm_params:
      use_gpu: true
      gpu_id: 0
```

### 2. 调整处理频率
```yaml
modules:
  perception:
    execution_frequency: 10  # 降低频率以减少CPU占用
  planning:
    execution_frequency: 10
```

### 3. 禁用不需要的模块
```yaml
modules:
  visualization:
    enabled: false  # 禁用可视化以提高性能
```

## 贡献指南

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目主页: https://github.com/your-repo/autonomous-driving
- 问题反馈: https://github.com/your-repo/autonomous-driving/issues
- 邮箱: your-email@example.com

## 致谢

- nuScenes数据集: https://www.nuscenes.org/
- PyQt5: https://www.riverbankcomputing.com/software/pyqt/
- OpenCV: https://opencv.org/

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 感知模块: 车道线检测、障碍物检测、Occupancy网络
- 规划模块: 行为规划、轨迹生成、轨迹优化
- 传感器模块: 摄像头、LiDAR、雷达、超声波、CAN总线
- 可视化模块: BEV显示、轨迹显示、传感器数据显示
- 支持仿真模式、nuScenes数据集模式、实车模式
