"""
自动驾驶可视化主界面

基于PyQt5的图形化仿真显示界面
"""

import sys
import os
import numpy as np
import cv2
from typing import Optional, Dict, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QSplitter, QTabWidget,
    QFrame, QScrollArea, QToolBar, QStatusBar, QApplication,
    QMenuBar, QMenu, QAction, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QIcon

# 导入可视化模块
from .data_manager import DataManager, EgoState, Obstacle, LaneLine, OccupancyGrid
from .bev_visualizer import BEVVisualizer, BEVConfig
from .trajectory_visualizer import TrajectoryVisualizer, TrajectoryConfig
from .sensor_visualizer import SensorVisualizer, SensorConfig


class VisualizationWorker(QThread):
    """可视化渲染工作线程"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            if self.visualizer.is_playing:
                self.visualizer.update_frame()
            self.msleep(int(1000 / self.visualizer.fps))
    
    def stop(self):
        self.running = False


class AutonomousDrivingVisualizer(QMainWindow):
    """
    自动驾驶可视化主窗口
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化数据管理器
        self.data_manager = DataManager()
        
        # 初始化可视化器
        self.bev_visualizer = BEVVisualizer(BEVConfig())
        self.trajectory_visualizer = TrajectoryVisualizer(TrajectoryConfig())
        self.sensor_visualizer = SensorVisualizer(SensorConfig())
        
        # 播放控制
        self.is_playing = False
        self.fps = 10
        self.current_frame = 0
        self.total_frames = 0
        
        # 显示选项
        self.show_grid = True
        self.show_ego = True
        self.show_obstacles = True
        self.show_lane_lines = True
        self.show_occupancy = True
        self.show_trajectories = True
        self.show_behavior = True
        self.show_sensor = True
        
        # 初始化UI
        self.init_ui()
        
        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 生成演示数据
        self.load_demo_data()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('自动驾驶仿真可视化工具')
        self.setGeometry(100, 100, 1600, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 中间：主显示区域
        display_area = self.create_display_area()
        splitter.addWidget(display_area)
        
        # 右侧：传感器数据
        sensor_panel = self.create_sensor_panel()
        splitter.addWidget(sensor_panel)
        
        # 设置分割器比例
        splitter.setSizes([250, 900, 350])
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_tool_bar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')
        
        # 快捷键
        self.create_shortcuts()
    
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        layout = QVBoxLayout()
        
        # 播放控制
        playback_group = QGroupBox("播放控制")
        playback_layout = QGridLayout()
        
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn, 0, 0)
        
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self.pause_playback)
        self.pause_btn.setEnabled(False)
        playback_layout.addWidget(self.pause_btn, 0, 1)
        
        self.step_back_btn = QPushButton("⏮ 上一帧")
        self.step_back_btn.clicked.connect(self.step_backward)
        playback_layout.addWidget(self.step_back_btn, 1, 0)
        
        self.step_fwd_btn = QPushButton("⏭ 下一帧")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        playback_layout.addWidget(self.step_fwd_btn, 1, 1)
        
        self.reset_btn = QPushButton("⏹ 重置")
        self.reset_btn.clicked.connect(self.reset_playback)
        playback_layout.addWidget(self.reset_btn, 2, 0, 1, 2)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # 帧控制
        frame_group = QGroupBox("帧控制")
        frame_layout = QGridLayout()
        
        frame_layout.addWidget(QLabel("帧率:"), 0, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.valueChanged.connect(self.set_fps)
        frame_layout.addWidget(self.fps_spin, 0, 1)
        
        frame_layout.addWidget(QLabel("当前帧:"), 1, 0)
        self.frame_label = QLabel("0 / 0")
        frame_layout.addWidget(self.frame_label, 1, 1)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        frame_layout.addWidget(self.frame_slider, 2, 0, 1, 2)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.grid_cb = QCheckBox("显示网格")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(lambda: self.toggle_layer('grid'))
        display_layout.addWidget(self.grid_cb)
        
        self.ego_cb = QCheckBox("显示自车")
        self.ego_cb.setChecked(True)
        self.ego_cb.stateChanged.connect(lambda: self.toggle_layer('ego'))
        display_layout.addWidget(self.ego_cb)
        
        self.obstacles_cb = QCheckBox("显示障碍物")
        self.obstacles_cb.setChecked(True)
        self.obstacles_cb.stateChanged.connect(lambda: self.toggle_layer('obstacles'))
        display_layout.addWidget(self.obstacles_cb)
        
        self.lanes_cb = QCheckBox("显示车道线")
        self.lanes_cb.setChecked(True)
        self.lanes_cb.stateChanged.connect(lambda: self.toggle_layer('lane_lines'))
        display_layout.addWidget(self.lanes_cb)
        
        self.occupancy_cb = QCheckBox("显示Occupancy")
        self.occupancy_cb.setChecked(True)
        self.occupancy_cb.stateChanged.connect(lambda: self.toggle_layer('occupancy'))
        display_layout.addWidget(self.occupancy_cb)
        
        self.trajectories_cb = QCheckBox("显示轨迹")
        self.trajectories_cb.setChecked(True)
        self.trajectories_cb.stateChanged.connect(lambda: self.toggle_layer('trajectories'))
        display_layout.addWidget(self.trajectories_cb)
        
        self.behavior_cb = QCheckBox("显示行为决策")
        self.behavior_cb.setChecked(True)
        self.behavior_cb.stateChanged.connect(lambda: self.toggle_layer('behavior'))
        display_layout.addWidget(self.behavior_cb)
        
        self.sensor_cb = QCheckBox("显示传感器数据")
        self.sensor_cb.setChecked(True)
        self.sensor_cb.stateChanged.connect(lambda: self.toggle_layer('sensor'))
        display_layout.addWidget(self.sensor_cb)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 视图控制
        view_group = QGroupBox("视图控制")
        view_layout = QGridLayout()
        
        view_layout.addWidget(QLabel("缩放:"), 0, 0)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.1, 5.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.valueChanged.connect(self.set_zoom)
        view_layout.addWidget(self.zoom_spin, 0, 1)
        
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        view_layout.addWidget(self.reset_view_btn, 1, 0, 1, 2)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_display_area(self) -> QWidget:
        """创建主显示区域"""
        area = QWidget()
        layout = QVBoxLayout()
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # BEV视图标签页
        bev_tab = QWidget()
        bev_layout = QVBoxLayout()
        self.bev_label = QLabel()
        self.bev_label.setAlignment(Qt.AlignCenter)
        self.bev_label.setMinimumSize(800, 600)
        self.bev_label.setStyleSheet("background-color: #1a1a1a;")
        bev_layout.addWidget(self.bev_label)
        bev_tab.setLayout(bev_layout)
        self.tab_widget.addTab(bev_tab, "BEV视图")
        
        # 传感器数据标签页（内部子标签）
        sensor_tab = QWidget()
        sensor_layout = QVBoxLayout()
        self.sensor_tab_widget = QTabWidget()
        
        # 多相机视图
        multi_cam_tab = QWidget()
        multi_cam_layout = QVBoxLayout()
        self.multi_cam_label = QLabel()
        self.multi_cam_label.setAlignment(Qt.AlignCenter)
        self.multi_cam_label.setMinimumSize(800, 600)
        self.multi_cam_label.setStyleSheet("background-color: #0a0a0a;")
        multi_cam_layout.addWidget(self.multi_cam_label)
        multi_cam_tab.setLayout(multi_cam_layout)
        self.sensor_tab_widget.addTab(multi_cam_tab, "多相机")
        
        # 相机标注视图
        annotated_tab = QWidget()
        annotated_layout = QVBoxLayout()
        self.annotated_label = QLabel()
        self.annotated_label.setAlignment(Qt.AlignCenter)
        self.annotated_label.setMinimumSize(800, 600)
        self.annotated_label.setStyleSheet("background-color: #0a0a0a;")
        annotated_layout.addWidget(self.annotated_label)
        annotated_tab.setLayout(annotated_layout)
        self.sensor_tab_widget.addTab(annotated_tab, "相机标注")
        
        # LiDAR BEV大图
        lidar_bev_big_tab = QWidget()
        lidar_bev_big_layout = QVBoxLayout()
        self.lidar_bev_big_label = QLabel()
        self.lidar_bev_big_label.setAlignment(Qt.AlignCenter)
        self.lidar_bev_big_label.setMinimumSize(800, 600)
        self.lidar_bev_big_label.setStyleSheet("background-color: #0a0a0a;")
        lidar_bev_big_layout.addWidget(self.lidar_bev_big_label)
        lidar_bev_big_tab.setLayout(lidar_bev_big_layout)
        self.sensor_tab_widget.addTab(lidar_bev_big_tab, "LiDAR BEV")
        
        # LiDAR 3D大图
        lidar_3d_big_tab = QWidget()
        lidar_3d_big_layout = QVBoxLayout()
        self.lidar_3d_big_label = QLabel()
        self.lidar_3d_big_label.setAlignment(Qt.AlignCenter)
        self.lidar_3d_big_label.setMinimumSize(800, 600)
        self.lidar_3d_big_label.setStyleSheet("background-color: #0a0a0a;")
        lidar_3d_big_layout.addWidget(self.lidar_3d_big_label)
        lidar_3d_big_tab.setLayout(lidar_3d_big_layout)
        self.sensor_tab_widget.addTab(lidar_3d_big_tab, "LiDAR 3D")
        
        sensor_layout.addWidget(self.sensor_tab_widget)
        sensor_tab.setLayout(sensor_layout)
        self.tab_widget.addTab(sensor_tab, "传感器数据")
        
        # 轨迹分析标签页
        traj_tab = QWidget()
        traj_layout = QVBoxLayout()
        self.traj_label = QLabel()
        self.traj_label.setAlignment(Qt.AlignCenter)
        self.traj_label.setMinimumSize(800, 600)
        self.traj_label.setStyleSheet("background-color: #1a1a1a;")
        traj_layout.addWidget(self.traj_label)
        traj_tab.setLayout(traj_layout)
        self.tab_widget.addTab(traj_tab, "轨迹分析")
        
        # 感知BEV标签页
        perception_bev_tab = QWidget()
        perception_bev_layout = QVBoxLayout()
        self.perception_bev_label = QLabel()
        self.perception_bev_label.setAlignment(Qt.AlignCenter)
        self.perception_bev_label.setMinimumSize(800, 600)
        self.perception_bev_label.setStyleSheet("background-color: #0a0a0a;")
        perception_bev_layout.addWidget(self.perception_bev_label)
        perception_bev_tab.setLayout(perception_bev_layout)
        self.tab_widget.addTab(perception_bev_tab, "感知BEV")
        
        layout.addWidget(self.tab_widget)
        area.setLayout(layout)
        return area
    
    def create_sensor_panel(self) -> QWidget:
        """创建传感器数据显示面板"""
        panel = QGroupBox("传感器数据")
        layout = QVBoxLayout()
        
        # LiDAR BEV视图
        lidar_group = QGroupBox("LiDAR BEV")
        lidar_layout = QVBoxLayout()
        self.lidar_bev_label = QLabel()
        self.lidar_bev_label.setAlignment(Qt.AlignCenter)
        self.lidar_bev_label.setMinimumSize(300, 300)
        self.lidar_bev_label.setStyleSheet("background-color: #0a0a0a;")
        lidar_layout.addWidget(self.lidar_bev_label)
        lidar_group.setLayout(lidar_layout)
        layout.addWidget(lidar_group)
        
        # LiDAR 3D视图
        lidar_3d_group = QGroupBox("LiDAR 3D")
        lidar_3d_layout = QVBoxLayout()
        self.lidar_3d_label = QLabel()
        self.lidar_3d_label.setAlignment(Qt.AlignCenter)
        self.lidar_3d_label.setMinimumSize(300, 200)
        self.lidar_3d_label.setStyleSheet("background-color: #0a0a0a;")
        lidar_3d_layout.addWidget(self.lidar_3d_label)
        lidar_3d_group.setLayout(lidar_3d_layout)
        layout.addWidget(lidar_3d_group)
        
        # 点云统计
        stats_group = QGroupBox("点云统计")
        stats_layout = QVBoxLayout()
        self.point_stats_label = QLabel("点数: 0")
        stats_layout.addWidget(self.point_stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        panel.setLayout(layout)
        return panel
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        open_action = QAction('打开数据文件', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_data_file)
        file_menu.addAction(open_action)
        
        save_action = QAction('保存截图', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        reset_view_action = QAction('重置视图', self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        # 播放菜单
        playback_menu = menubar.addMenu('播放')
        
        play_action = QAction('播放/暂停', self)
        play_action.setShortcut('Space')
        play_action.triggered.connect(self.toggle_playback)
        playback_menu.addAction(play_action)
        
        step_fwd_action = QAction('下一帧', self)
        step_fwd_action.setShortcut('Right')
        step_fwd_action.triggered.connect(self.step_forward)
        playback_menu.addAction(step_fwd_action)
        
        step_back_action = QAction('上一帧', self)
        step_back_action.setShortcut('Left')
        step_back_action.triggered.connect(self.step_backward)
        playback_menu.addAction(step_back_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        toolbar.addAction('▶', self.toggle_playback)
        toolbar.addAction('⏸', self.pause_playback)
        toolbar.addAction('⏮', self.step_backward)
        toolbar.addAction('⏭', self.step_forward)
        toolbar.addSeparator()
        toolbar.addAction('📷', self.save_screenshot)
    
    def create_shortcuts(self):
        """创建快捷键"""
        QShortcut(QKeySequence('Space'), self, self.toggle_playback)
        QShortcut(QKeySequence('Left'), self, self.step_backward)
        QShortcut(QKeySequence('Right'), self, self.step_forward)
        QShortcut(QKeySequence('Ctrl+S'), self, self.save_screenshot)
    
    def load_demo_data(self):
        """加载演示数据"""
        self.data_manager.generate_demo_data(num_frames=100)
        self.total_frames = self.data_manager.get_frame_count()
        self.frame_slider.setRange(0, self.total_frames - 1)
        self.update_frame_display()
    
    def update_frame(self):
        """更新当前帧"""
        if self.is_playing:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            self.frame_slider.setValue(self.current_frame)
        
        self.render_frame()
    
    def _do_render(self):
        """实际渲染逻辑"""
        ego = self.data_manager.ego_state
        obstacles = self.data_manager.obstacles if self.show_obstacles else []
        lane_lines = self.data_manager.lane_lines if self.show_lane_lines else []
        occupancy = self.data_manager.occupancy_grid if self.show_occupancy else None
        planning = self.data_manager.planning_result
        sensor_data = self.data_manager.sensor_data
        
        if ego is None:
            return
        
        # 渲染BEV视图
        bev_image = self.bev_visualizer.render(
            ego=ego,
            obstacles=obstacles if self.show_obstacles else None,
            lane_lines=lane_lines if self.show_lane_lines else None,
            occupancy_grid=occupancy if self.show_occupancy else None,
            show_grid=self.show_grid,
            show_info=True
        )
        
        # 叠加轨迹
        if self.show_trajectories and planning is not None:
            self.trajectory_visualizer.draw_planning_result(
                bev_image, planning, ego, 
                lambda x, y: self.bev_visualizer.world_to_screen(x, y, ego)
            )
        
        # 叠加行为决策信息
        if self.show_behavior and planning is not None:
            self.trajectory_visualizer.draw_behavior_info(
                bev_image, planning, (10, 150)
            )
        
        # 显示BEV图像
        self.display_image(self.bev_label, bev_image)
        
        # 渲染轨迹分析图
        if planning is not None:
            if len(planning.candidate_trajectories) > 0:
                try:
                    from .trajectory_visualizer import create_trajectory_figure
                    traj_image = create_trajectory_figure(
                        planning.candidate_trajectories,
                        planning.candidate_trajectories.index(planning.selected_trajectory)
                        if planning.selected_trajectory in planning.candidate_trajectories else -1
                    )
                    self.display_image(self.traj_label, traj_image)
                except Exception as e:
                    self._logger.error(f"Trajectory figure rendering failed: {e}")
                    import traceback
                    self._logger.error(traceback.format_exc())
                    fallback = np.zeros((600, 800, 3), dtype=np.uint8)
                    cv2.putText(fallback, "Trajectory Render Error", (50, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    self.display_image(self.traj_label, fallback)
            else:
                # 没有候选轨迹时，显示决策信息面板
                info_canvas = np.ones((600, 800, 3), dtype=np.uint8) * 40
                self.trajectory_visualizer.draw_behavior_info(
                    info_canvas, planning, (20, 20)
                )
                # 额外添加英文提示，防止中文字体缺失时完全空白
                cv2.putText(info_canvas, "No candidate trajectories", (20, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                self.display_image(self.traj_label, info_canvas)
        else:
            # planning 为 None 时显示提示
            none_canvas = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(none_canvas, "No Planning Data", (50, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
            self.display_image(self.traj_label, none_canvas)
        
        # 渲染传感器数据
        if self.show_sensor and sensor_data is not None:
            # 多相机视图
            if sensor_data.camera_images:
                layout = '2x3' if len(sensor_data.camera_images) >= 6 else '2x2'
                multi_cam_img = self.sensor_visualizer.render_multi_camera(
                    sensor_data.camera_images, layout=layout
                )
                self.display_image(self.multi_cam_label, multi_cam_img)
            
            # 相机标注视图
            if self.data_manager.annotated_image is not None:
                self.display_image(self.annotated_label, self.data_manager.annotated_image)
            
            # LiDAR BEV
            if sensor_data.lidar_points is not None:
                lidar_bev = self.sensor_visualizer.render_lidar_bev(
                    sensor_data.lidar_points, ego, obstacles,
                    canvas_size=(800, 800)
                )
                self.display_image(self.lidar_bev_label, lidar_bev)
                self.display_image(self.lidar_bev_big_label, lidar_bev)
                
                # LiDAR 3D
                lidar_3d = self.sensor_visualizer.render_lidar_3d_projection(
                    sensor_data.lidar_points, image_size=(800, 600)
                )
                self.display_image(self.lidar_3d_label, lidar_3d)
                self.display_image(self.lidar_3d_big_label, lidar_3d)
                
                # 更新统计
                self.point_stats_label.setText(f"点数: {len(sensor_data.lidar_points)}")
        
        # 渲染感知BEV
        perception_bev = self.bev_visualizer.render(
            ego=ego,
            obstacles=obstacles if self.show_obstacles else None,
            lane_lines=lane_lines if self.show_lane_lines else None,
            occupancy_grid=occupancy if self.show_occupancy else None,
            show_grid=True,
            show_info=False
        )
        self.display_image(self.perception_bev_label, perception_bev)
        
        # 更新状态栏
        total_frames = max(1, self.total_frames)
        self.status_bar.showMessage(
            f'帧: {self.current_frame + 1}/{total_frames} | '
            f'时间: {ego.timestamp:.2f}s | '
            f'速度: {ego.velocity:.2f}m/s'
        )
    
    def render_frame(self):
        """渲染当前帧（从预加载的frames中加载）"""
        # 加载数据
        if not self.data_manager.load_frame(self.current_frame):
            return
        self._do_render()
    
    def render_live_frame(self):
        """渲染当前实时帧（直接使用data_manager中的当前数据）"""
        self._do_render()
    
    def display_image(self, label: QLabel, image: np.ndarray):
        """在标签上显示图像"""
        if image is None:
            return
        
        # 确保图像是BGR格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为QImage
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放以适应标签
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
    
    def update_frame_display(self):
        """更新帧显示"""
        self.frame_label.setText(f"{self.current_frame + 1} / {self.total_frames}")
        self.frame_slider.setValue(self.current_frame)
    
    # 播放控制槽函数
    def toggle_playback(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """开始播放"""
        self.is_playing = True
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.timer.start(int(1000 / self.fps))
    
    def pause_playback(self):
        """暂停播放"""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.timer.stop()
    
    def step_forward(self):
        """前进一帧"""
        self.pause_playback()
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame_display()
            self.render_frame()
    
    def step_backward(self):
        """后退一帧"""
        self.pause_playback()
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame_display()
            self.render_frame()
    
    def reset_playback(self):
        """重置播放"""
        self.pause_playback()
        self.current_frame = 0
        self.update_frame_display()
        self.render_frame()
    
    def on_slider_changed(self, value: int):
        """滑块值改变"""
        self.pause_playback()
        self.current_frame = value
        self.update_frame_display()
        self.render_frame()
    
    def set_fps(self, fps: int):
        """设置帧率"""
        self.fps = fps
        if self.is_playing:
            self.timer.setInterval(int(1000 / self.fps))
    
    def toggle_layer(self, layer: str):
        """切换图层显示"""
        if layer == 'grid':
            self.show_grid = self.grid_cb.isChecked()
        elif layer == 'ego':
            self.show_ego = self.ego_cb.isChecked()
        elif layer == 'obstacles':
            self.show_obstacles = self.obstacles_cb.isChecked()
        elif layer == 'lane_lines':
            self.show_lane_lines = self.lanes_cb.isChecked()
        elif layer == 'occupancy':
            self.show_occupancy = self.occupancy_cb.isChecked()
        elif layer == 'trajectories':
            self.show_trajectories = self.trajectories_cb.isChecked()
        elif layer == 'behavior':
            self.show_behavior = self.behavior_cb.isChecked()
        elif layer == 'sensor':
            self.show_sensor = self.sensor_cb.isChecked()
        
        self.render_frame()
    
    def set_zoom(self, zoom: float):
        """设置缩放"""
        self.bev_visualizer.set_zoom(zoom)
        self.render_frame()
    
    def reset_view(self):
        """重置视图"""
        self.zoom_spin.setValue(1.0)
        self.bev_visualizer.reset_view()
        self.render_frame()
    
    def open_data_file(self):
        """打开数据文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "Data Files (*.json *.pkl *.npz);;All Files (*)"
        )
        if filename:
            # TODO: 实现数据加载
            QMessageBox.information(self, "提示", "数据加载功能待实现")
    
    def save_screenshot(self):
        """保存截图"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存截图", 
            f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "Images (*.png *.jpg *.bmp)"
        )
        if filename:
            # 获取当前BEV图像
            ego = self.data_manager.ego_state
            obstacles = self.data_manager.obstacles
            lane_lines = self.data_manager.lane_lines
            occupancy = self.data_manager.occupancy_grid
            
            bev_image = self.bev_visualizer.render(
                ego=ego,
                obstacles=obstacles,
                lane_lines=lane_lines,
                occupancy_grid=occupancy,
                show_grid=self.show_grid,
                show_info=True
            )
            
            cv2.imwrite(filename, bev_image)
            self.status_bar.showMessage(f'截图已保存: {filename}')
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
            "自动驾驶仿真可视化工具 v1.0\n\n"
            "功能：\n"
            "- BEV视角可视化\n"
            "- 轨迹规划显示\n"
            "- 传感器数据显示\n"
            "- 实时播放控制\n\n"
            "快捷键：\n"
            "Space - 播放/暂停\n"
            "←/→ - 上一帧/下一帧\n"
            "Ctrl+S - 保存截图"
        )
    
    def closeEvent(self, event):
        """关闭事件"""
        self.timer.stop()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = AutonomousDrivingVisualizer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
