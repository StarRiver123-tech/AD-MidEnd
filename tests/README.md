# 自动驾驶模块测试框架

基于pytest的自动驾驶模块测试框架，支持单元测试、模块级仿真测试和回归测试。

## 功能特性

### 1. 单元测试框架
- **感知模块测试**: 目标检测、跟踪、车道线检测
- **规划模块测试**: 轨迹生成、评估指标
- **传感器接入测试**: 激光雷达、摄像头、毫米波雷达、IMU、GPS

### 2. 模块级仿真测试
- 单模块独立测试
- 模拟输入数据生成
- 性能评估指标

### 3. 测试数据生成
- 模拟传感器数据生成
- 模拟目标数据生成
- 模拟车道线数据生成

### 4. 评估指标
- **感知模块**: 准确率、召回率、F1分数、IoU、AP、mAP
- **规划模块**: 轨迹平滑度、安全性、舒适性、效率

### 5. 回归测试
- 批量测试支持
- 测试报告生成（JSON、HTML格式）
- 基线比较

## 目录结构

```
tests/
├── conftest.py                    # pytest配置文件和fixtures
├── pytest.ini                     # pytest配置
├── run_tests.py                   # 测试运行脚本
├── requirements.txt               # 依赖包
├── README.md                      # 本文档
│
├── data_generators/               # 测试数据生成工具
│   ├── __init__.py
│   ├── sensor_data_generator.py   # 传感器数据生成
│   ├── object_data_generator.py   # 目标数据生成
│   └── lane_data_generator.py     # 车道线数据生成
│
├── metrics/                       # 评估指标模块
│   ├── __init__.py
│   ├── perception_metrics.py      # 感知指标
│   └── planning_metrics.py        # 规划指标
│
├── unit/                          # 单元测试
│   ├── __init__.py
│   ├── test_perception.py         # 感知模块测试
│   ├── test_planning.py           # 规划模块测试
│   └── test_sensor.py             # 传感器测试
│
├── simulation/                    # 仿真测试
│   ├── __init__.py
│   ├── simulation_framework.py    # 仿真框架
│   └── test_simulation.py         # 仿真测试用例
│
└── regression/                    # 回归测试
    ├── __init__.py
    ├── regression_framework.py    # 回归测试框架
    └── test_regression.py         # 回归测试用例
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行所有测试

```bash
python run_tests.py --all
```

### 运行单元测试

```bash
python run_tests.py --unit
# 或
pytest -m unit
```

### 运行仿真测试

```bash
python run_tests.py --simulation
# 或
pytest -m simulation
```

### 运行回归测试

```bash
python run_tests.py --regression
# 或
pytest -m regression
```

### 运行特定模块测试

```bash
# 感知模块
python run_tests.py --perception

# 规划模块
python run_tests.py --planning

# 传感器
python run_tests.py --sensor
```

### 运行特定测试文件

```bash
python run_tests.py --test unit/test_perception.py
```

### 排除慢速测试

```bash
python run_tests.py --all --no-slow
```

### 不生成报告

```bash
python run_tests.py --all --no-report
```

## 测试标记

| 标记 | 说明 |
|------|------|
| `unit` | 单元测试 |
| `integration` | 集成测试 |
| `simulation` | 仿真测试 |
| `regression` | 回归测试 |
| `slow` | 慢速测试 |
| `perception` | 感知模块测试 |
| `planning` | 规划模块测试 |
| `sensor` | 传感器测试 |
| `performance` | 性能测试 |

## 数据生成器使用示例

### 传感器数据生成

```python
from data_generators.sensor_data_generator import SensorDataGenerator

generator = SensorDataGenerator(seed=42)

# 生成激光雷达点云
pointcloud = generator.generate_lidar_pointcloud(num_points=1000)

# 生成摄像头图像
image = generator.generate_camera_image(image_size=(1920, 1080))

# 生成雷达数据
radar_data = generator.generate_radar_data(num_targets=10)

# 生成IMU数据
imu_data = generator.generate_imu_data(duration=1.0)

# 生成GPS数据
gps_data = generator.generate_gps_data(duration=1.0)

# 生成完整传感器套件
sensor_suite = generator.generate_sensor_suite()
```

### 目标数据生成

```python
from data_generators.object_data_generator import ObjectDataGenerator, ObjectType

generator = ObjectDataGenerator(seed=42)

# 生成单个目标
vehicle = generator.generate_single_object(
    object_type=ObjectType.VEHICLE,
    position=(10, 5, 0),
    velocity=(5, 0, 0)
)

# 生成目标集合
objects = generator.generate_object_set(num_objects=10)

# 生成目标轨迹
trajectory = generator.generate_object_trajectory(
    object_type=ObjectType.VEHICLE,
    duration=10.0,
    motion_pattern="straight"
)

# 生成检测结果（带噪声）
detections = generator.generate_detection_results(
    ground_truth_objects,
    noise_level=0.1,
    miss_rate=0.05
)
```

### 车道线数据生成

```python
from data_generators.lane_data_generator import LaneDataGenerator, LaneType, LanePosition

generator = LaneDataGenerator(seed=42)

# 生成直线车道线
straight_lane = generator.generate_straight_lane(
    length=100.0,
    lane_type=LaneType.DASHED_WHITE
)

# 生成曲线车道线
curved_lane = generator.generate_curved_lane(
    length=100.0,
    curvature=0.01
)

# 生成车道段
lane_segment = generator.generate_lane_segment(
    num_lanes=3,
    lane_width=3.5
)
```

## 评估指标使用示例

### 感知指标

```python
from metrics.perception_metrics import PerceptionMetrics, DetectionResult

metrics = PerceptionMetrics(iou_threshold=0.5)

# 计算IoU
iou = metrics.calculate_iou_bev(bbox1, bbox2)

# 计算检测指标
detection_metrics = metrics.calculate_detection_metrics(
    ground_truths, predictions
)

# 计算AP
ap = metrics.calculate_ap(ground_truths, predictions)

# 计算mAP
map_results = metrics.calculate_mAP(
    ground_truths_by_type, predictions_by_type
)
```

### 规划指标

```python
from metrics.planning_metrics import PlanningMetrics, Trajectory

metrics = PlanningMetrics()

# 计算轨迹平滑度
smoothness = metrics.calculate_smoothness(trajectory)

# 计算安全性评分
safety = metrics.calculate_safety_score(trajectory, obstacles)

# 计算舒适性评分
comfort = metrics.calculate_comfort_score(trajectory)

# 计算完整指标
trajectory_metrics = metrics.calculate_trajectory_metrics(
    trajectory, obstacles
)
```

## 仿真框架使用示例

```python
from simulation.simulation_framework import (
    SimulationConfig, PerceptionModuleSimulator
)

# 创建配置
config = SimulationConfig(
    duration=10.0,
    time_step=0.1
)

# 创建仿真器
simulator = PerceptionModuleSimulator(config)

# 定义输入生成函数
def input_generator(t: float):
    return {
        'num_objects': 5,
        'noise_level': 0.1
    }

# 运行仿真
result = simulator.run(input_generator)

# 获取指标
metrics = simulator.get_detection_metrics()
```

## 回归测试使用示例

```python
from regression.regression_framework import RegressionTestRunner

# 创建运行器
runner = RegressionTestRunner(output_dir="test_reports")

# 定义测试函数
def test_function():
    assert some_condition

# 运行批量测试
runner.run_batch_tests({
    'test1': test_function,
    'test2': another_test
}, iterations=3)

# 生成报告
report_files = runner.generate_report(
    report_name="regression_report",
    formats=["json", "html"]
)

# 与基线比较
comparison = runner.compare_with_baseline("baseline.json")

# 打印摘要
runner.print_summary()
```

## 测试报告

测试报告默认输出到 `test_reports/` 目录，包含：

- **JSON格式**: 详细的测试结果数据
- **HTML格式**: 可视化的测试报告

报告内容包括：
- 测试摘要（总数、通过、失败、通过率）
- 各测试套件详细结果
- 测试耗时统计
- 失败测试的详细信息

## 配置说明

### pytest.ini 配置

```ini
[pytest]
markers =
    unit: 单元测试
    integration: 集成测试
    simulation: 仿真测试
    regression: 回归测试
    slow: 慢速测试

addopts = 
    -v
    --tb=short
    --strict-markers
    --color=yes
```

### conftest.py Fixtures

框架提供了丰富的fixtures：

- `sensor_generator`: 传感器数据生成器
- `object_generator`: 目标数据生成器
- `lane_generator`: 车道线数据生成器
- `perception_metrics`: 感知指标计算器
- `planning_metrics`: 规划指标计算器
- `sample_*`: 各种示例数据

## 扩展开发

### 添加新的数据生成器

1. 在 `data_generators/` 目录下创建新文件
2. 继承基类或创建新的生成器类
3. 在 `__init__.py` 中导出

### 添加新的评估指标

1. 在 `metrics/` 目录下创建新文件
2. 实现指标计算类
3. 在 `__init__.py` 中导出

### 添加新的测试用例

1. 在相应目录下创建 `test_*.py` 文件
2. 使用pytest标记分类测试
3. 使用fixtures获取测试数据

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

MIT License
