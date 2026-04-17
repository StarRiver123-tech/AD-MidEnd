#!/usr/bin/env python3
"""
自动驾驶系统 - 主入口程序
==========================

这是自动驾驶系统的主入口，负责：
1. 系统初始化和配置加载
2. 模块生命周期管理
3. 主循环执行
4. 异常处理和系统关闭

使用方法:
    python main.py --config config/system_config.yaml --mode simulation
    python main.py --mode nuscenes --data-root /path/to/nuscenes
    python main.py --module-test perception

作者: Autonomous Driving Team
版本: 1.0.0
"""

import os
import sys
import time
import math
import signal
import argparse
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入系统模块
from src.config.config_manager import ConfigManager
from src.communication.message_bus import MessageBus, Message
from src.communication.publisher import Publisher
from src.logs.logger import Logger, LogLevel

# 导入核心模块
from src.sensors.core.sensor_manager import SensorManager
from src.perception.perception_module import PerceptionModule
from src.planning.planning_module import PlanningModule

# 导入数据集适配器
try:
    from src.dataset.nuscenes_adapter import NuScenesAdapter, NuScenesConfig
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False

# 导入可视化模块
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    from src.visualization.visualizer import AutonomousDrivingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class AutonomousDrivingSystem:
    """
    自动驾驶系统主类
    管理整个系统的生命周期和数据流
    """
    
    def __init__(self):
        """初始化自动驾驶系统"""
        self._logger = Logger("AutonomousDrivingSystem")
        
        # 系统状态
        self._running = False
        self._initialized = False
        self._shutdown_requested = False
        
        # 配置
        self._config_path: Optional[str] = None
        self._config_manager: Optional[ConfigManager] = None
        self._system_config: Optional[Dict[str, Any]] = None
        
        # 消息总线
        self._message_bus: Optional[MessageBus] = None
        
        # 核心模块
        self._sensor_manager: Optional[SensorManager] = None
        self._perception_module: Optional[PerceptionModule] = None
        self._planning_module: Optional[PlanningModule] = None
        
        # 数据集适配器
        self._nuscenes_adapter: Optional[Any] = None
        
        # 可视化
        self._visualizer: Optional[Any] = None
        self._qt_app: Optional[Any] = None
        
        # 运行模式
        self._mode: str = "simulation"  # simulation | nuscenes | real_vehicle
        
        # nuScenes 回放状态（用于可视化模式）
        self._nuscenes_scenes: List[Dict[str, Any]] = []
        self._nuscenes_scene_idx: int = 0
        self._nuscenes_sample_tokens: List[str] = []
        self._nuscenes_sample_idx: int = 0
        self._nuscenes_timer: Optional[Any] = None
        
        # 统计信息
        self._start_time: float = 0.0
        self._loop_count: int = 0
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理系统信号"""
        self._logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True
        self.stop()
    
    def initialize(self, config_path: Optional[str] = None,
                   mode: str = "simulation",
                   data_root: Optional[str] = None) -> bool:
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
            mode: 运行模式 (simulation | nuscenes | real_vehicle)
            data_root: 数据集根目录 (nuScenes模式)
        
        Returns:
            bool: 初始化是否成功
        """
        self._logger.info("=" * 60)
        self._logger.info("Initializing Autonomous Driving System")
        self._logger.info("=" * 60)
        
        self._mode = mode
        self._config_path = config_path
        
        try:
            # 1. 加载配置
            if not self._load_config(config_path):
                self._logger.error("Failed to load configuration")
                return False
            
            # 2. 初始化消息总线
            self._message_bus = MessageBus()
            self._logger.info("Message bus initialized")
            
            # 3. 根据模式初始化
            if mode == "nuscenes":
                if not self._init_nuscenes_mode(data_root):
                    return False
            elif mode == "real_vehicle":
                if not self._init_real_vehicle_mode():
                    return False
            else:  # simulation mode
                if not self._init_simulation_mode():
                    return False
            
            # 4. 初始化核心模块
            if not self._init_core_modules():
                return False
            
            # 5. 初始化可视化 (如果启用)
            self._init_visualization()
            
            self._initialized = True
            self._logger.info("=" * 60)
            self._logger.info("System initialization completed successfully")
            self._logger.info("=" * 60)
            return True
            
        except Exception as e:
            self._logger.error(f"System initialization failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
    
    def _load_config(self, config_path: Optional[str]) -> bool:
        """加载系统配置"""
        self._config_manager = ConfigManager()
        
        if config_path and os.path.exists(config_path):
            if not self._config_manager.load_from_file(config_path):
                self._logger.warning("Failed to load config file, using defaults")
        else:
            # 使用默认配置
            default_config = PROJECT_ROOT / "config" / "system_config.yaml"
            if default_config.exists():
                self._config_manager.load_from_file(str(default_config))
            else:
                self._logger.warning("No config file found, using defaults")
        
        self._system_config = self._config_manager.get_system_config()
        self._logger.info("Configuration loaded successfully")
        return True
    
    def _init_simulation_mode(self) -> bool:
        """初始化仿真模式"""
        self._logger.info("Initializing simulation mode...")
        
        # 初始化传感器管理器 (仿真模式使用模拟数据)
        self._sensor_manager = SensorManager()
        
        # 配置仿真传感器
        sensor_config = self._config_manager.get('sensors', {})
        
        # 这里可以添加仿真特定的初始化
        self._logger.info("Simulation mode initialized")
        return True
    
    def _init_nuscenes_mode(self, data_root: Optional[str]) -> bool:
        """初始化nuScenes数据集模式"""
        self._logger.info("Initializing nuScenes dataset mode...")
        
        if not NUSCENES_AVAILABLE:
            self._logger.error("nuScenes adapter not available. "
                             "Please install nuscenes-devkit")
            return False
        
        # 获取数据集路径
        if data_root is None:
            nuscenes_config = self._config_manager.get('nuscenes', {})
            data_root = nuscenes_config.get('data_root', 'data/nuscenes')
        
        if not os.path.exists(data_root):
            self._logger.error(f"nuScenes data root not found: {data_root}")
            return False
        
        # 初始化nuScenes适配器
        try:
            version = self._config_manager.get('nuscenes.version', 'v1.0-mini')
            nuscenes_config = NuScenesConfig(
                data_root=data_root,
                version=version
            )
            self._nuscenes_adapter = NuScenesAdapter(config=nuscenes_config)
            
            # 初始化nuScenes数据发布者
            self._nuscenes_camera_pub = Publisher("nuscenes_adapter", "sensor/camera", self._message_bus)
            self._nuscenes_lidar_pub = Publisher("nuscenes_adapter", "sensor/lidar", self._message_bus)
            self._nuscenes_radar_pub = Publisher("nuscenes_adapter", "sensor/radar", self._message_bus)
            self._nuscenes_can_pub = Publisher("nuscenes_adapter", "sensor/can/vehicle", self._message_bus)
            
            # 启动消息总线
            self._message_bus.start()
            
            self._logger.info(f"nuScenes adapter initialized: {data_root}")
        except Exception as e:
            self._logger.error(f"Failed to initialize nuScenes adapter: {e}")
            return False
        
        return True
    
    def _init_real_vehicle_mode(self) -> bool:
        """初始化实车模式"""
        self._logger.info("Initializing real vehicle mode...")
        
        # 初始化传感器管理器
        sensor_config_path = PROJECT_ROOT / "config" / "sensor_config.yaml"
        self._sensor_manager = SensorManager(str(sensor_config_path) if sensor_config_path.exists() else None)
        
        # 初始化所有传感器
        if not self._sensor_manager.initialize_all():
            self._logger.error("Failed to initialize sensors")
            return False
        
        self._logger.info("Real vehicle mode initialized")
        return True
    
    def _init_core_modules(self) -> bool:
        """初始化核心处理模块"""
        self._logger.info("Initializing core modules...")
        
        modules_config = self._config_manager.get('modules', {})
        
        # 1. 初始化感知模块
        if modules_config.get('perception', {}).get('enabled', True):
            self._perception_module = PerceptionModule(self._message_bus)
            if not self._perception_module.initialize(self._config_manager):
                self._logger.error("Failed to initialize perception module")
                return False
            self._logger.info("Perception module initialized")
        
        # 2. 初始化规划模块
        if modules_config.get('planning', {}).get('enabled', True):
            self._planning_module = PlanningModule(self._message_bus)
            if not self._planning_module.initialize(self._config_manager):
                self._logger.error("Failed to initialize planning module")
                return False
            self._logger.info("Planning module initialized")
        
        return True
    
    def _init_visualization(self) -> bool:
        """初始化可视化模块"""
        if not VISUALIZATION_AVAILABLE:
            self._logger.warning("Visualization not available (PyQt5 not installed)")
            return False
        
        modules_config = self._config_manager.get('modules', {})
        if not modules_config.get('visualization', {}).get('enabled', True):
            self._logger.info("Visualization disabled in config")
            return False
        
        try:
            self._qt_app = QApplication(sys.argv)
            self._visualizer = AutonomousDrivingVisualizer()
            self._logger.info("Visualization initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize visualization: {e}")
            return False
    
    def start(self) -> bool:
        """启动系统"""
        if not self._initialized:
            self._logger.error("System not initialized")
            return False
        
        if self._running:
            self._logger.warning("System already running")
            return True
        
        self._logger.info("=" * 60)
        self._logger.info("Starting Autonomous Driving System")
        self._logger.info("=" * 60)
        
        try:
            # 启动传感器 (实车模式)
            if self._mode == "real_vehicle" and self._sensor_manager:
                self._sensor_manager.start_all()
            
            # 启动核心模块
            if self._perception_module:
                self._perception_module.start()
            
            if self._planning_module:
                self._planning_module.start()
            
            self._running = True
            self._start_time = time.time()
            
            self._logger.info("System started successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start system: {e}")
            return False
    
    def run(self) -> None:
        """运行主循环"""
        if not self._running:
            self._logger.error("System not started")
            return
        
        self._logger.info("Entering main loop...")
        
        # 获取主循环频率
        frequency = self._config_manager.get('system.main_loop_frequency', 50)
        period = 1.0 / frequency
        
        try:
            if self._mode == "nuscenes" and self._nuscenes_adapter:
                self._run_nuscenes_loop(period)
            elif self._visualizer:
                self._run_with_visualization()
            else:
                self._run_simple_loop(period)
                
        except Exception as e:
            self._logger.error(f"Error in main loop: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
        finally:
            self.stop()
    
    def _run_simple_loop(self, period: float) -> None:
        """简单主循环 (无可视化)"""
        self._logger.info(f"Running main loop at {1.0/period:.1f} Hz")
        
        while self._running and not self._shutdown_requested:
            loop_start = time.time()
            
            # 执行主循环任务
            self._main_loop_iteration()
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._logger.info("Main loop exited")
    
    def _run_with_visualization(self) -> None:
        """带可视化的主循环"""
        self._logger.info("Running with visualization")
        
        # 显示可视化窗口
        self._visualizer.show()
        
        # 使用Qt的事件循环
        self._qt_app.exec_()
    
    def _run_nuscenes_loop(self, period: float) -> None:
        """nuScenes数据集处理循环"""
        self._logger.info("Running nuScenes dataset processing loop")
        
        # 获取所有场景
        self._nuscenes_scenes = self._nuscenes_adapter.get_all_scenes()
        self._nuscenes_scene_idx = 0
        self._nuscenes_sample_idx = 0
        self._logger.info(f"Found {len(self._nuscenes_scenes)} scenes")
        
        use_viz = self._visualizer is not None
        if use_viz:
            self._logger.info("Showing visualization window for nuScenes mode")
            self._visualizer.show()
            self._visualizer.raise_()
            self._visualizer.activateWindow()
            self._visualizer.timer.stop()
            self._visualizer.data_manager.clear()
            
            # 使用 QTimer 在 Qt 事件循环中驱动 nuScenes 回放
            # 限制最小间隔为 100ms，确保可视化能跟上
            interval_ms = max(int(period * 1000), 100)
            self._nuscenes_timer = QTimer()
            self._nuscenes_timer.timeout.connect(self._process_nuscenes_frame)
            self._nuscenes_timer.start(interval_ms)
            self._qt_app.exec_()
        else:
            # 无可视化：使用阻塞循环
            for scene_idx, scene in enumerate(self._nuscenes_scenes):
                if self._shutdown_requested:
                    break
                self._logger.info(f"Processing scene {scene_idx + 1}/{len(self._nuscenes_scenes)}: {scene['name']}")
                sample_tokens = self._nuscenes_adapter.get_scene_sample_tokens(scene['token'])
                for sample_idx, sample_token in enumerate(sample_tokens):
                    if self._shutdown_requested:
                        break
                    sample_data = self._nuscenes_adapter.load_sample_data(sample_token)
                    self._publish_nuscenes_data(sample_data)
                    time.sleep(period)
                    if (sample_idx + 1) % 10 == 0:
                        progress = (sample_idx + 1) / len(sample_tokens) * 100
                        self._logger.info(f"  Progress: {progress:.1f}% ({sample_idx + 1}/{len(sample_tokens)})")
                self._logger.info(f"Scene {scene['name']} completed")
    
    def _process_nuscenes_frame(self):
        """处理 nuScenes 的下一帧（由 QTimer 调用）"""
        if self._shutdown_requested:
            if self._nuscenes_timer:
                self._nuscenes_timer.stop()
            if self._qt_app:
                self._qt_app.quit()
            return
        
        if self._nuscenes_scene_idx >= len(self._nuscenes_scenes):
            # 所有场景播放完毕：停止 timer 但保持窗口打开
            if self._nuscenes_timer:
                self._nuscenes_timer.stop()
            self._logger.info("All scenes completed. Close the visualization window to exit.")
            if self._visualizer:
                self._visualizer.setWindowTitle("nuScenes Replay - Completed")
            return
        
        scene = self._nuscenes_scenes[self._nuscenes_scene_idx]
        
        # 如果是新场景，获取样本列表
        if self._nuscenes_sample_idx == 0:
            self._nuscenes_sample_tokens = self._nuscenes_adapter.get_scene_sample_tokens(scene['token'])
            self._logger.info(
                f"Processing scene {self._nuscenes_scene_idx + 1}/{len(self._nuscenes_scenes)}: {scene['name']}"
            )
        
        if self._nuscenes_sample_idx < len(self._nuscenes_sample_tokens):
            sample_token = self._nuscenes_sample_tokens[self._nuscenes_sample_idx]
            sample_data = self._nuscenes_adapter.load_sample_data(sample_token)
            
            # 发布到消息总线
            self._publish_nuscenes_data(sample_data)
            
            # 更新可视化
            if self._visualizer:
                frame_data = self._convert_nuscenes_to_frame(sample_data)
                
                # 注入规划模块生成的真实规划结果（如果有）
                if self._planning_module:
                    pr = self._planning_module.get_current_planning_result()
                    if pr is not None:
                        viz_pr = self._convert_planning_result_for_viz(pr)
                        if viz_pr is not None:
                            num_candidates = len(viz_pr.candidate_trajectories)
                            self._logger.debug(
                                f"Injecting planning result: behavior={viz_pr.behavior.value}, "
                                f"candidates={num_candidates}, selected={'yes' if viz_pr.selected_trajectory else 'no'}"
                            )
                            frame_data['planning_result'] = viz_pr
                
                self._visualizer.data_manager.set_live_frame(frame_data)
                self._visualizer.render_live_frame()
            
            self._nuscenes_sample_idx += 1
            
            # 打印进度
            if self._nuscenes_sample_idx % 10 == 0:
                progress = self._nuscenes_sample_idx / len(self._nuscenes_sample_tokens) * 100
                self._logger.info(
                    f"  Progress: {progress:.1f}% ({self._nuscenes_sample_idx}/{len(self._nuscenes_sample_tokens)})"
                )
        else:
            # 场景结束
            self._logger.info(f"Scene {scene['name']} completed")
            self._nuscenes_scene_idx += 1
            self._nuscenes_sample_idx = 0
    
    def _publish_nuscenes_data(self, sample_data: Dict[str, Any]) -> None:
        """发布nuScenes数据到消息总线"""
        # 发布摄像头数据（同时发布到通用topic和具体camera topic）
        if 'cameras' in sample_data:
            for cam_name, cam_data in sample_data['cameras'].items():
                self._nuscenes_camera_pub.publish({
                    'camera_name': cam_name,
                    'data': cam_data
                })
                # 将 front camera 发布到 perception 期望的 topic
                if cam_name == 'CAM_FRONT':
                    self._message_bus.publish_sync(
                        Message(
                            topic='sensor/camera/front',
                            data=cam_data,
                            publisher_id='nuscenes_adapter'
                        )
                    )
        
        # 发布LiDAR数据
        if 'lidar' in sample_data:
            self._nuscenes_lidar_pub.publish(sample_data['lidar'])
        
        # 发布雷达数据
        if 'radars' in sample_data:
            for radar_name, radar_data in sample_data['radars'].items():
                self._nuscenes_radar_pub.publish({
                    'radar_name': radar_name,
                    'data': radar_data
                })
        
        # 发布车辆状态
        if 'ego_pose' in sample_data:
            self._nuscenes_can_pub.publish(sample_data['ego_pose'])
    
    def _convert_trajectory_for_viz(self, traj) -> Optional[Any]:
        """将 common.data_types.Trajectory 转换为可视化模块的 Trajectory"""
        if traj is None or not getattr(traj, 'points', None):
            return None
        from src.visualization.data_manager import Trajectory as VizTrajectory
        pts = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in traj.points])
        vels = np.array([p.longitudinal_velocity for p in traj.points]) if traj.points else None
        times = np.array([p.relative_time for p in traj.points]) if traj.points else None
        return VizTrajectory(points=pts, velocities=vels, timestamps=times, cost=getattr(traj, 'cost', 0.0))
    
    def _convert_planning_result_for_viz(self, pr) -> Optional[Any]:
        """将 common.data_types.PlanningResult 转换为可视化模块的 PlanningResult"""
        if pr is None:
            return None
        from src.visualization.data_manager import PlanningResult as VizPlanningResult, BehaviorType
        behavior_map = {
            'cruise': BehaviorType.KEEP_LANE,
            'follow': BehaviorType.FOLLOW,
            'stop': BehaviorType.STOP,
            'emergency_stop': BehaviorType.EMERGENCY_STOP,
            'overtake': BehaviorType.CHANGE_LEFT,
            'change_left': BehaviorType.CHANGE_LEFT,
            'change_right': BehaviorType.CHANGE_RIGHT,
            'turn_left': BehaviorType.TURN_LEFT,
            'turn_right': BehaviorType.TURN_RIGHT,
        }
        behavior = behavior_map.get(getattr(pr, 'behavior_type', '').lower(), BehaviorType.UNKNOWN)
        
        # 先转换候选轨迹；如果 selected_trajectory 与某个候选是同一对象引用，
        # 则让 selected 直接指向转换后的候选对象，保证 visualizer 中 index/in 判断正确。
        candidates = []
        selected = None
        for ct in pr.candidate_trajectories:
            vt = self._convert_trajectory_for_viz(ct)
            if vt:
                vt.trajectory_type = 'candidate'
                vt.color = (0.5, 0.5, 0.5)
                candidates.append(vt)
                if pr.selected_trajectory is not None and ct is pr.selected_trajectory:
                    selected = vt
        
        # 如果 selected_trajectory 不在候选列表中，单独转换
        if selected is None and pr.selected_trajectory is not None:
            selected = self._convert_trajectory_for_viz(pr.selected_trajectory)
        
        if selected:
            selected.trajectory_type = 'selected'
            selected.color = (0.0, 1.0, 0.0)
        
        ts = 0.0
        if hasattr(pr.timestamp, 'to_seconds'):
            ts = pr.timestamp.to_seconds()
        elif isinstance(pr.timestamp, (int, float)):
            ts = float(pr.timestamp)
        
        return VizPlanningResult(
            selected_trajectory=selected,
            candidate_trajectories=candidates,
            behavior=behavior,
            behavior_description=getattr(pr, 'behavior_explanation', getattr(pr, 'behavior_description', '')),
            target_speed=getattr(pr, 'target_speed', 0.0),
            target_lane=getattr(pr, 'target_lane_id', 0),
            timestamp=ts
        )
    
    def _convert_nuscenes_to_frame(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """将nuScenes样本数据转换为可视化帧格式"""
        import numpy as np
        import cv2
        from src.visualization.data_manager import (
            EgoState, Obstacle, LaneLine, SensorData, PlanningResult, Trajectory, BehaviorType
        )
        from src.communication.message_bus import Message
        
        frame: Dict[str, Any] = {}
        
        # 自车状态
        ego_pose = sample_data.get('ego_pose', {})
        if ego_pose:
            trans = ego_pose.get('translation', [0.0, 0.0, 0.0])
            rot = ego_pose.get('rotation', [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
            w, x, y, z = rot
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            frame['ego_state'] = EgoState(
                x=trans[0],
                y=trans[1],
                z=trans[2],
                heading=yaw,
                velocity=0.0,
                timestamp=sample_data.get('timestamp', 0) / 1e6
            )
        
        # 障碍物
        obstacles: List[Obstacle] = []
        for i, ann in enumerate(sample_data.get('annotations', [])):
            bbox = ann.get('bbox_3d', {})
            center = bbox.get('center', [0.0, 0.0, 0.0])
            size = bbox.get('size', [1.0, 1.0, 1.0])
            rot = bbox.get('rotation', [1.0, 0.0, 0.0, 0.0])
            w, x, y, z = rot
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            vel = bbox.get('velocity', [0.0, 0.0, 0.0])
            velocity = math.sqrt(vel[0]**2 + vel[1]**2)
            
            obstacles.append(Obstacle(
                id=i,
                obstacle_type=ann.get('category', 'unknown'),
                x=center[0],
                y=center[1],
                z=center[2],
                # nuScenes size format is [width, length, height]
                length=size[1],
                width=size[0],
                height=size[2],
                heading=yaw,
                velocity=velocity,
                vx=vel[0],
                vy=vel[1],
            ))
        frame['obstacles'] = obstacles
        
        # 车道线
        lane_lines: List[LaneLine] = []
        lane_colors = {
            'lane_divider': 'white',
            'road_divider': 'yellow',
            'stop_line': 'red',
            'ped_crossing': 'white',
            'drivable_area': 'white',
            'lane_centerline': 'yellow',
        }
        lane_types = {
            'lane_divider': 'dashed',
            'road_divider': 'double',
            'stop_line': 'solid',
            'ped_crossing': 'dashed',
            'drivable_area': 'solid',
            'lane_centerline': 'dashed',
        }
        for i, lane_ann in enumerate(sample_data.get('lane_annotations', [])):
            geom = lane_ann.get('geometry', [])
            if len(geom) < 2:
                continue
            pts = np.array(geom)
            lane_lines.append(LaneLine(
                id=i,
                line_type=lane_types.get(lane_ann.get('lane_type', ''), 'solid'),
                color=lane_colors.get(lane_ann.get('lane_type', ''), 'white'),
                points=pts,
                confidence=1.0
            ))
        frame['lane_lines'] = lane_lines
        
        # 传感器数据
        sensor_data = SensorData(timestamp=sample_data.get('timestamp', 0) / 1e6)
        if 'lidar' in sample_data:
            sensor_data.lidar_points = sample_data['lidar']['points']
        if 'cameras' in sample_data:
            for cam_name, cam_data in sample_data['cameras'].items():
                if 'image' in cam_data and cam_data['image'] is not None:
                    img = cam_data['image']
                    # 反归一化（nuscenes adapter 的 preprocessor 可能将图像归一化到 float）
                    if img.dtype != np.uint8:
                        img = self._nuscenes_adapter.image_preprocessor.denormalize(img)
                    # nuscenes adapter 加载的是 RGB，但 OpenCV 可视化流程基于 BGR
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    sensor_data.camera_images[cam_name] = img
        frame['sensor_data'] = sensor_data
        
        # 规划结果（占位，使用简单的空结果）
        frame['planning_result'] = PlanningResult(
            behavior=BehaviorType.UNKNOWN,
            behavior_description="nuScenes replay",
            timestamp=sample_data.get('timestamp', 0) / 1e6
        )
        
        # 生成 front camera 的 3D 标注投影图像
        front_cam = sample_data.get('cameras', {}).get('CAM_FRONT')
        if front_cam is not None and 'image' in front_cam:
            front_img = front_cam['image']
            if front_img.dtype != np.uint8:
                front_img = self._nuscenes_adapter.image_preprocessor.denormalize(front_img)
            # 转为 BGR 供 OpenCV 使用
            if len(front_img.shape) == 3 and front_img.shape[2] == 3:
                front_img = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
            
            annotated_img = self._project_annotations_to_image(
                front_img.copy(),
                sample_data.get('annotations', []),
                front_cam.get('ego2global'),
                front_cam.get('sensor2ego'),
                front_cam.get('intrinsics')
            )
            
            # 投影车道线到前视相机
            if sample_data.get('lane_annotations'):
                annotated_img = self._project_lanes_to_image(
                    annotated_img,
                    sample_data.get('lane_annotations', []),
                    front_cam.get('ego2global'),
                    front_cam.get('sensor2ego'),
                    front_cam.get('intrinsics')
                )
            
            # 如果有 LiDAR 数据，将点云投影到 front camera 上
            if 'lidar' in sample_data:
                annotated_img = self._project_lidar_to_image(
                    annotated_img,
                    sample_data['lidar']['points'],
                    sample_data['lidar'].get('ego2global'),
                    sample_data['lidar'].get('sensor2ego'),
                    front_cam.get('ego2global'),
                    front_cam.get('sensor2ego'),
                    front_cam.get('intrinsics')
                )
            
            frame['annotated_image'] = annotated_img
        
        return frame
    
    def _project_annotations_to_image(self, image: np.ndarray,
                                       annotations: List[Dict[str, Any]],
                                       ego2global_list: Optional[List[List[float]]],
                                       sensor2ego_list: Optional[List[List[float]]],
                                       intrinsics_list: Optional[List[List[float]]]) -> np.ndarray:
        """将3D标注框投影到相机图像上"""
        import numpy as np
        import cv2
        
        if ego2global_list is None or sensor2ego_list is None or intrinsics_list is None:
            return image
        
        ego2global = np.array(ego2global_list)
        sensor2ego = np.array(sensor2ego_list)
        intrinsics = np.array(intrinsics_list)
        
        canvas = image.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        
        # 变换矩阵
        ego2global_inv = np.linalg.inv(ego2global)
        sensor2ego_inv = np.linalg.inv(sensor2ego)
        
        color_map = {
            'car': (0, 255, 0),
            'truck': (0, 200, 0),
            'bus': (0, 150, 255),
            'pedestrian': (255, 0, 255),
            'bicycle': (255, 255, 0),
            'motorcycle': (255, 128, 0),
            'barrier': (128, 128, 128),
            'traffic_cone': (0, 255, 255),
            'bicycle_rack': (128, 0, 128),
            'unknown': (200, 200, 200),
        }
        
        # 3D bbox 的边连接关系 (注意 get_corners 返回的是 (3, 8))
        # corners 顺序: 0(+l,+w,+h), 1(+l,+w,-h), 2(+l,-w,+h), 3(+l,-w,-h),
        #              4(-l,+w,+h), 5(-l,+w,-h), 6(-l,-w,+h), 7(-l,-w,-h)
        edges = [
            (0, 2), (2, 6), (6, 4), (4, 0),  # 顶部 (z = +h/2)
            (1, 3), (3, 7), (7, 5), (5, 1),  # 底部 (z = -h/2)
            (0, 1), (2, 3), (6, 7), (4, 5),  # 立柱
        ]
        
        h, w = canvas.shape[:2]
        num_drawn = 0
        
        for ann in annotations:
            corners_arr = np.array(ann.get('corners', []))  # shape could be (3, 8)
            if corners_arr.size == 0:
                continue
            
            # get_corners 返回的是 (3, 8)，需要转置为 (8, 3)
            if corners_arr.shape[0] == 3 and corners_arr.shape[1] == 8:
                corners = corners_arr.T  # now (8, 3)
            elif corners_arr.shape[0] == 8 and corners_arr.shape[1] == 3:
                corners = corners_arr
            else:
                continue
            
            category = ann.get('category', 'unknown')
            color = color_map.get(category, (200, 200, 200))
            
            # 投影角点
            projected = []
            for corner in corners:
                p_global = np.array([corner[0], corner[1], corner[2], 1.0])
                p_ego = ego2global_inv @ p_global
                p_cam = sensor2ego_inv @ p_ego
                
                if p_cam[2] <= 0.1:
                    projected.append(None)
                    continue
                
                u = intrinsics[0, 0] * p_cam[0] / p_cam[2] + intrinsics[0, 2]
                v = intrinsics[1, 1] * p_cam[1] / p_cam[2] + intrinsics[1, 2]
                
                u, v = int(u), int(v)
                projected.append((u, v))
            
            # 绘制边（只绘制在图像内或附近的边）
            valid_edges = 0
            for e in edges:
                p1 = projected[e[0]]
                p2 = projected[e[1]]
                if p1 is None or p2 is None:
                    continue
                # 至少有一个点在图像内才绘制
                in_img1 = (0 <= p1[0] < w and 0 <= p1[1] < h)
                in_img2 = (0 <= p2[0] < w and 0 <= p2[1] < h)
                if in_img1 or in_img2:
                    cv2.line(canvas, p1, p2, color, 2)
                    valid_edges += 1
            
            # 绘制类别标签（找最前面的角点）
            if valid_edges > 0:
                valid_pts = [p for p in projected if p is not None]
                if valid_pts:
                    # 使用 top-left 角点作为标签位置
                    min_x = min(p[0] for p in valid_pts)
                    min_y = min(p[1] for p in valid_pts)
                    label = category
                    # 画一个半透明的背景条
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(canvas, (min_x, max(0, min_y - th - 8)),
                                  (min_x + tw + 8, min_y), color, -1)
                    cv2.putText(canvas, label, (min_x + 4, max(th + 2, min_y - 4)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    num_drawn += 1
        
        # 添加标题和计数
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
        title = f"CAM_FRONT - 3D GT Annotations | Objects: {num_drawn}"
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return canvas
    
    def _project_lanes_to_image(self, image: np.ndarray,
                                 lane_annotations: List[Dict[str, Any]],
                                 ego2global_list: Optional[List[List[float]]],
                                 sensor2ego_list: Optional[List[List[float]]],
                                 intrinsics_list: Optional[List[List[float]]]) -> np.ndarray:
        """将车道线真值投影到相机图像上"""
        import numpy as np
        import cv2
        
        if ego2global_list is None or sensor2ego_list is None or intrinsics_list is None:
            return image
        
        ego2global = np.array(ego2global_list)
        sensor2ego = np.array(sensor2ego_list)
        intrinsics = np.array(intrinsics_list)
        
        canvas = image.copy()
        h, w = canvas.shape[:2]
        
        ego2global_inv = np.linalg.inv(ego2global)
        sensor2ego_inv = np.linalg.inv(sensor2ego)
        
        lane_colors = {
            'lane_divider': (255, 255, 255),
            'road_divider': (0, 255, 255),
            'stop_line': (0, 0, 255),
            'ped_crossing': (255, 0, 255),
            'drivable_area': (0, 255, 255),
            'lane_centerline': (0, 165, 255),
        }
        lane_thickness = {
            'lane_divider': 2,
            'road_divider': 2,
            'stop_line': 3,
            'ped_crossing': 2,
            'drivable_area': 3,
            'lane_centerline': 2,
        }
        polygon_types = {'drivable_area', 'stop_line', 'ped_crossing'}
        
        for lane in lane_annotations:
            geom = np.array(lane.get('geometry', []))
            if len(geom) < 2:
                continue
            
            lane_type = lane.get('lane_type', 'lane_divider')
            color = lane_colors.get(lane_type, (200, 200, 200))
            thickness = lane_thickness.get(lane_type, 2)
            is_polygon = lane_type in polygon_types
            
            # 投影线段点
            projected = []
            for pt in geom:
                p_global = np.array([pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0, 1.0])
                p_ego = ego2global_inv @ p_global
                p_cam = sensor2ego_inv @ p_ego
                
                if p_cam[2] <= 0.1:
                    projected.append(None)
                    continue
                
                u = intrinsics[0, 0] * p_cam[0] / p_cam[2] + intrinsics[0, 2]
                v = intrinsics[1, 1] * p_cam[1] / p_cam[2] + intrinsics[1, 2]
                projected.append((int(u), int(v)))
            
            # 绘制线段（只保留图像内或跨越图像边界的有效段）
            num_pts = len(projected)
            for i in range(num_pts - 1):
                p1 = projected[i]
                p2 = projected[i + 1]
                if p1 is None or p2 is None:
                    continue
                in_img1 = (0 <= p1[0] < w and 0 <= p1[1] < h)
                in_img2 = (0 <= p2[0] < w and 0 <= p2[1] < h)
                if in_img1 or in_img2:
                    cv2.line(canvas, p1, p2, color, thickness)
            
            # 闭合多边形（绘制最后一条边）
            if is_polygon and num_pts >= 3:
                p1 = projected[-1]
                p2 = projected[0]
                if p1 is not None and p2 is not None:
                    in_img1 = (0 <= p1[0] < w and 0 <= p1[1] < h)
                    in_img2 = (0 <= p2[0] < w and 0 <= p2[1] < h)
                    if in_img1 or in_img2:
                        cv2.line(canvas, p1, p2, color, thickness)
        
        return canvas
    
    def _project_lidar_to_image(self, image: np.ndarray,
                                 lidar_points: np.ndarray,
                                 lidar_ego2global_list: Optional[List[List[float]]],
                                 lidar_sensor2ego_list: Optional[List[List[float]]],
                                 cam_ego2global_list: Optional[List[List[float]]],
                                 cam_sensor2ego_list: Optional[List[List[float]]],
                                 intrinsics_list: Optional[List[List[float]]]) -> np.ndarray:
        """将LiDAR点云投影到相机图像上（使用向量化运算）"""
        import numpy as np
        import cv2
        
        if (lidar_ego2global_list is None or lidar_sensor2ego_list is None or
            cam_ego2global_list is None or cam_sensor2ego_list is None or intrinsics_list is None):
            return image
        
        lidar_ego2global = np.array(lidar_ego2global_list)
        lidar_sensor2ego = np.array(lidar_sensor2ego_list)
        cam_ego2global = np.array(cam_ego2global_list)
        cam_sensor2ego = np.array(cam_sensor2ego_list)
        intrinsics = np.array(intrinsics_list)
        
        canvas = image.copy()
        h, w = canvas.shape[:2]
        
        # 变换矩阵: lidar -> global -> ego -> cam
        lidar2global = lidar_ego2global @ np.linalg.inv(lidar_sensor2ego)
        global2cam = np.linalg.inv(cam_sensor2ego) @ np.linalg.inv(cam_ego2global)
        lidar2cam = global2cam @ lidar2global
        
        # 采样点云（太多点会很慢）
        pts = lidar_points[:, :3]
        if len(pts) > 20000:
            step = len(pts) // 20000 + 1
            pts = pts[::step]
        
        # 向量化投影
        ones = np.ones((len(pts), 1), dtype=pts.dtype)
        homo = np.hstack([pts, ones])  # Nx4
        p_cam = (lidar2cam @ homo.T).T  # Nx4
        
        # 过滤相机后方的点
        valid = p_cam[:, 2] > 0.5
        p_cam = p_cam[valid]
        if len(p_cam) == 0:
            return canvas
        
        u = intrinsics[0, 0] * p_cam[:, 0] / p_cam[:, 2] + intrinsics[0, 2]
        v = intrinsics[1, 1] * p_cam[:, 1] / p_cam[:, 2] + intrinsics[1, 2]
        depth = p_cam[:, 2]
        
        # 过滤图像范围外的点（保留稍微超出边界的以便绘制）
        in_view = (u >= -10) & (u < w + 10) & (v >= -10) & (v < h + 10)
        u, v, depth = u[in_view], v[in_view], depth[in_view]
        
        # 按深度排序（从远到近绘制）
        order = np.argsort(depth)[::-1]
        u, v, depth = u[order].astype(np.int32), v[order].astype(np.int32), depth[order]
        
        # 向量化着色
        colors = np.empty((len(depth), 3), dtype=np.uint8)
        colors[depth < 10] = [0, 0, 255]
        colors[(depth >= 10) & (depth < 30)] = [0, 255, 255]
        colors[(depth >= 30) & (depth < 60)] = [0, 255, 0]
        colors[depth >= 60] = [255, 255, 0]
        
        # 绘制点
        for i in range(len(u)):
            if 0 <= u[i] < w and 0 <= v[i] < h:
                cv2.circle(canvas, (u[i], v[i]), 1, tuple(int(c) for c in colors[i]), -1)
        
        return canvas
    
    def _main_loop_iteration(self) -> None:
        """主循环单次迭代"""
        self._loop_count += 1
        
        # 这里可以添加系统监控、状态检查等任务
        
        # 定期打印统计信息
        if self._loop_count % 500 == 0:  # 每500次循环
            self._print_stats()
    
    def _print_stats(self) -> None:
        """打印系统统计信息"""
        elapsed = time.time() - self._start_time
        loop_rate = self._loop_count / elapsed if elapsed > 0 else 0
        
        self._logger.info("-" * 40)
        self._logger.info(f"System Statistics:")
        self._logger.info(f"  Runtime: {elapsed:.1f} seconds")
        self._logger.info(f"  Loop count: {self._loop_count}")
        self._logger.info(f"  Loop rate: {loop_rate:.1f} Hz")
        
        # 模块统计
        if self._perception_module:
            stats = self._perception_module.get_stats()
            self._logger.info(f"  Perception: {stats['state']}, "
                            f"processed {stats['processing_count']} frames")
        
        if self._planning_module:
            stats = self._planning_module.get_stats()
            self._logger.info(f"  Planning: {stats['state']}, "
                            f"processed {stats['processing_count']} frames")
        
        self._logger.info("-" * 40)
    
    def stop(self) -> None:
        """停止系统"""
        if not self._running:
            return
        
        self._logger.info("=" * 60)
        self._logger.info("Stopping Autonomous Driving System")
        self._logger.info("=" * 60)
        
        self._running = False
        
        # 停止核心模块
        if self._planning_module:
            self._planning_module.stop()
            self._logger.info("Planning module stopped")
        
        if self._perception_module:
            self._perception_module.stop()
            self._logger.info("Perception module stopped")
        
        # 停止传感器
        if self._sensor_manager:
            self._sensor_manager.stop_all()
            self._sensor_manager.release_all()
            self._logger.info("Sensor manager stopped")
        
        # 打印最终统计
        self._print_final_stats()
        
        self._logger.info("System stopped")
    
    def _print_final_stats(self) -> None:
        """打印最终统计信息"""
        elapsed = time.time() - self._start_time
        
        self._logger.info("=" * 60)
        self._logger.info("Final Statistics:")
        self._logger.info(f"  Total runtime: {elapsed:.1f} seconds")
        self._logger.info(f"  Total loops: {self._loop_count}")
        self._logger.info(f"  Average loop rate: {self._loop_count / elapsed:.1f} Hz")
        self._logger.info("=" * 60)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'initialized': self._initialized,
            'running': self._running,
            'mode': self._mode,
            'runtime': time.time() - self._start_time if self._running else 0,
            'loop_count': self._loop_count,
            'perception': self._perception_module.get_stats() if self._perception_module else None,
            'planning': self._planning_module.get_stats() if self._planning_module else None,
        }


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Autonomous Driving System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行仿真模式
  python main.py --mode simulation
  
  # 运行nuScenes数据集模式
  python main.py --mode nuscenes --data-root /path/to/nuscenes
  
  # 使用自定义配置
  python main.py --config config/my_config.yaml --mode simulation
  
  # 运行模块测试
  python main.py --module-test perception
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/system_config.yaml',
        help='Path to configuration file (default: config/system_config.yaml)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['simulation', 'nuscenes', 'real_vehicle'],
        default='simulation',
        help='Run mode (default: simulation)'
    )
    
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        help='Path to nuScenes dataset root directory'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--module-test',
        type=str,
        choices=['perception', 'planning', 'sensor', 'visualization'],
        help='Run specific module test'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    Logger.set_global_level(LogLevel[args.log_level])
    
    # 创建系统实例
    system = AutonomousDrivingSystem()
    
    # 模块测试模式
    if args.module_test:
        from test_module import run_module_test
        return run_module_test(args.module_test)
    
    # 初始化系统
    if not system.initialize(
        config_path=args.config,
        mode=args.mode,
        data_root=args.data_root
    ):
        print("System initialization failed!")
        return 1
    
    # 启动系统
    if not system.start():
        print("System start failed!")
        return 1
    
    # 运行主循环
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
