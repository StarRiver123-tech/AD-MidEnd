# 自动驾驶仿真可视化工具

基于PyQt5的图形化仿真显示界面，用于自动驾驶系统的数据可视化和分析。

## 功能特性

### 1. BEV视角可视化
- **自车显示**: 绿色矩形表示自车，带方向指示
- **车道线显示**: 支持实线、虚线、路沿等不同类型
- **障碍物显示**: 3D边界框投影，支持车辆、行人、骑行者
- **Occupancy网格**: 半透明叠加显示

### 2. 规划结果显示
- **候选轨迹**: 多条候选轨迹对比显示
- **选中轨迹**: 绿色高亮显示最终选择轨迹
- **行为决策**: 文字面板显示当前决策
- **轨迹分析**: 速度、曲率、成本对比图表

### 3. 传感器数据显示
- **LiDAR BEV**: 俯视点云图
- **LiDAR 3D**: 3D投影视图
- **点云统计**: 点数统计信息

### 4. 用户交互
- **播放控制**: 播放/暂停/步进/重置
- **视图控制**: 缩放/平移/重置
- **图层选择**: 独立控制各图层显示
- **截图保存**: 保存当前视图

## 安装

### 安装依赖
```bash
pip install -r requirements.txt
```

### 依赖包
- PyQt5 >= 5.15.0
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- matplotlib >= 3.3.0

## 使用方法

### 启动可视化工具
```bash
# 使用演示数据（默认）
python run_visualization.py

# 指定帧率
python run_visualization.py --fps 20

# 指定窗口大小
python run_visualization.py --width 1920 --height 1080

# 全屏模式
python run_visualization.py --fullscreen

# 查看帮助
python run_visualization.py --help
```

### 快捷键
| 快捷键 | 功能 |
|--------|------|
| Space | 播放/暂停 |
| ← | 上一帧 |
| → | 下一帧 |
| Ctrl + S | 保存截图 |
| Ctrl + O | 打开数据文件 |
| Ctrl + Q | 退出 |

## 模块结构

```
visualization/
├── __init__.py              # 模块初始化
├── data_manager.py          # 数据管理器
├── bev_visualizer.py        # BEV可视化
├── trajectory_visualizer.py # 轨迹可视化
├── sensor_visualizer.py     # 传感器可视化
├── visualizer.py            # 主GUI界面
├── run_visualization.py     # 启动脚本
├── requirements.txt         # 依赖包
└── README.md               # 说明文档
```

## API使用示例

### 基本使用
```python
from visualization import AutonomousDrivingVisualizer
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = AutonomousDrivingVisualizer()
window.show()
window.load_demo_data()
sys.exit(app.exec_())
```

### 自定义数据
```python
from visualization.data_manager import DataManager, EgoState, Obstacle

# 创建数据管理器
dm = DataManager()

# 添加自车状态
ego = EgoState(x=0, y=0, heading=0, velocity=5.0)

# 添加障碍物
obstacle = Obstacle(
    id=1,
    obstacle_type='vehicle',
    x=20, y=0,
    length=4.5, width=1.8, height=1.5,
    heading=0, velocity=4.0
)

# 添加到帧
dm.add_frame({
    'ego_state': ego,
    'obstacles': [obstacle],
    'lane_lines': [],
    'planning_result': None,
    'occupancy_grid': None,
    'sensor_data': None
})
```

### BEV可视化
```python
from visualization.bev_visualizer import BEVVisualizer, BEVConfig

# 创建配置
config = BEVConfig(
    width=1200,
    height=800,
    view_range_x=100.0,
    view_range_y=60.0
)

# 创建可视化器
bev_vis = BEVVisualizer(config)

# 渲染
image = bev_vis.render(
    ego=ego_state,
    obstacles=obstacles,
    lane_lines=lane_lines,
    occupancy_grid=occupancy,
    show_grid=True,
    show_info=True
)
```

### 轨迹可视化
```python
from visualization.trajectory_visualizer import TrajectoryVisualizer
from visualization.data_manager import Trajectory

# 创建轨迹
trajectory = Trajectory(
    points=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    velocities=np.array([5.0, 5.5, 6.0]),
    trajectory_type='selected',
    color=(0, 1, 0)
)

# 绘制轨迹
traj_vis = TrajectoryVisualizer()
traj_vis.draw_trajectory_on_bev(
    canvas=image,
    trajectory=trajectory,
    ego=ego_state,
    world_to_screen_func=world_to_screen,
    is_selected=True
)
```

### 传感器可视化
```python
from visualization.sensor_visualizer import SensorVisualizer

# 创建可视化器
sensor_vis = SensorVisualizer()

# 渲染LiDAR BEV
lidar_bev = sensor_vis.render_lidar_bev(
    lidar_points=points,
    ego=ego_state,
    obstacles=obstacles
)

# 渲染3D投影
lidar_3d = sensor_vis.render_lidar_3d_projection(
    lidar_points=points,
    elevation_angle=20.0,
    azimuth_angle=45.0
)
```

## 配置选项

### BEV配置 (BEVConfig)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| width | int | 1200 | 画布宽度 |
| height | int | 800 | 画布高度 |
| view_range_x | float | 100.0 | 前后显示范围(米) |
| view_range_y | float | 60.0 | 左右显示范围(米) |
| background_color | tuple | (30,30,30) | 背景颜色 |
| ego_color | tuple | (0,255,0) | 自车颜色 |

### 轨迹配置 (TrajectoryConfig)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| candidate_color | tuple | (128,128,128) | 候选轨迹颜色 |
| selected_color | tuple | (0,255,0) | 选中轨迹颜色 |
| velocity_colormap | bool | True | 使用速度颜色映射 |
| velocity_min/max | float | 0/15 | 速度映射范围 |

### 传感器配置 (SensorConfig)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| point_size | int | 2 | 点大小 |
| point_color_mode | str | 'height' | 颜色模式 |
| bev_resolution | float | 0.1 | BEV分辨率 |
| bev_range | float | 50.0 | BEV显示范围 |

## 演示数据

运行工具时会自动生成演示数据，包括：
- 自车沿S形路径行驶
- 前车和对向车辆
- 行人横穿场景
- 车道线（左右车道线+中心线）
- 多条候选轨迹和选中轨迹
- 模拟LiDAR点云

## 扩展开发

### 添加新的可视化图层
1. 在 `bev_visualizer.py` 中添加绘制方法
2. 在 `visualizer.py` 中添加显示选项
3. 在 `render_frame()` 中调用绘制方法

### 添加新的数据类型
1. 在 `data_manager.py` 中定义数据类
2. 在 `DataManager` 中添加加载方法
3. 在可视化器中添加渲染方法

## 许可证

MIT License
