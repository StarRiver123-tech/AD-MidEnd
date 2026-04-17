"""
模块级仿真测试框架
支持单模块独立测试、模拟输入数据、性能评估
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class SimulationStatus(Enum):
    """仿真状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """仿真配置"""
    duration: float = 10.0  # 仿真持续时间（秒）
    time_step: float = 0.1  # 时间步长（秒）
    real_time_factor: float = 1.0  # 实时因子（1.0为实时）
    log_interval: float = 1.0  # 日志记录间隔（秒）
    enable_visualization: bool = False  # 是否启用可视化
    random_seed: int = 42  # 随机种子


@dataclass
class SimulationMetrics:
    """仿真指标"""
    execution_time: float = 0.0  # 执行时间
    total_steps: int = 0  # 总步数
    average_step_time: float = 0.0  # 平均步长时间
    max_step_time: float = 0.0  # 最大步长时间
    min_step_time: float = float('inf')  # 最小步长时间
    memory_usage: float = 0.0  # 内存使用（MB）
    
    def to_dict(self) -> Dict:
        return {
            'execution_time': self.execution_time,
            'total_steps': self.total_steps,
            'average_step_time': self.average_step_time,
            'max_step_time': self.max_step_time,
            'min_step_time': self.min_step_time if self.min_step_time != float('inf') else 0.0,
            'memory_usage': self.memory_usage,
        }


@dataclass
class SimulationResult:
    """仿真结果"""
    status: SimulationStatus
    metrics: SimulationMetrics
    data: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'metrics': self.metrics.to_dict(),
            'data': self.data,
            'logs': self.logs,
            'errors': self.errors,
        }


class ModuleSimulator:
    """模块仿真器基类"""
    
    def __init__(self, name: str, config: SimulationConfig):
        """
        初始化模块仿真器
        
        Args:
            name: 模块名称
            config: 仿真配置
        """
        self.name = name
        self.config = config
        self.status = SimulationStatus.IDLE
        self.current_time = 0.0
        self.step_count = 0
        self.metrics = SimulationMetrics()
        self.data_log: List[Dict] = []
        self.error_log: List[str] = []
        
    def initialize(self) -> bool:
        """
        初始化模块
        
        Returns:
            是否初始化成功
        """
        self.status = SimulationStatus.IDLE
        return True
    
    def step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单步仿真
        
        Args:
            input_data: 输入数据
            
        Returns:
            输出数据
        """
        raise NotImplementedError("Subclasses must implement step method")
    
    def run(self, input_generator: Callable[[float], Dict[str, Any]]) -> SimulationResult:
        """
        运行仿真
        
        Args:
            input_generator: 输入数据生成函数
            
        Returns:
            仿真结果
        """
        self.status = SimulationStatus.RUNNING
        start_time = time.time()
        step_times = []
        
        try:
            while self.current_time < self.config.duration:
                step_start = time.time()
                
                # 生成输入数据
                input_data = input_generator(self.current_time)
                
                # 执行单步
                output_data = self.step(input_data)
                
                # 记录数据
                self.data_log.append({
                    'time': self.current_time,
                    'input': input_data,
                    'output': output_data
                })
                
                # 更新时间和计数
                self.current_time += self.config.time_step
                self.step_count += 1
                
                # 记录步长时间
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # 实时仿真延迟
                if self.config.real_time_factor > 0:
                    expected_step_time = self.config.time_step / self.config.real_time_factor
                    if step_time < expected_step_time:
                        time.sleep(expected_step_time - step_time)
            
            self.status = SimulationStatus.COMPLETED
            
        except Exception as e:
            self.status = SimulationStatus.ERROR
            self.error_log.append(str(e))
        
        # 计算指标
        end_time = time.time()
        self.metrics.execution_time = end_time - start_time
        self.metrics.total_steps = self.step_count
        
        if step_times:
            self.metrics.average_step_time = np.mean(step_times)
            self.metrics.max_step_time = np.max(step_times)
            self.metrics.min_step_time = np.min(step_times)
        
        return SimulationResult(
            status=self.status,
            metrics=self.metrics,
            data={'log': self.data_log},
            logs=[f"Step {i}" for i in range(self.step_count)],
            errors=self.error_log
        )
    
    def reset(self):
        """重置仿真器"""
        self.status = SimulationStatus.IDLE
        self.current_time = 0.0
        self.step_count = 0
        self.metrics = SimulationMetrics()
        self.data_log = []
        self.error_log = []


class PerceptionModuleSimulator(ModuleSimulator):
    """感知模块仿真器"""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("PerceptionModule", config)
        from data_generators.object_data_generator import ObjectDataGenerator
        from metrics.perception_metrics import PerceptionMetrics
        
        self.object_generator = ObjectDataGenerator(seed=config.random_seed)
        self.metrics_calculator = PerceptionMetrics()
        self.ground_truth_objects = []
        self.detected_objects = []
    
    def initialize(self) -> bool:
        """初始化感知模块"""
        super().initialize()
        self.ground_truth_objects = []
        self.detected_objects = []
        return True
    
    def step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行感知模块单步
        
        Args:
            input_data: 包含传感器数据的字典
            
        Returns:
            检测结果
        """
        # 模拟感知处理延迟
        time.sleep(0.001)
        
        # 生成真值目标
        num_objects = input_data.get('num_objects', 5)
        self.ground_truth_objects = self.object_generator.generate_object_set(
            num_objects=num_objects
        )
        
        # 生成检测结果（带噪声）
        noise_level = input_data.get('noise_level', 0.1)
        miss_rate = input_data.get('miss_rate', 0.05)
        
        self.detected_objects = self.object_generator.generate_detection_results(
            self.ground_truth_objects,
            noise_level=noise_level,
            miss_rate=miss_rate
        )
        
        return {
            'num_detections': len(self.detected_objects),
            'detections': self.detected_objects,
            'processing_time': 0.001
        }
    
    def get_detection_metrics(self) -> Dict:
        """获取检测指标"""
        from metrics.perception_metrics import DetectionResult
        
        gt_results = [
            DetectionResult(
                object_type=gt.object_type,
                bbox=np.array([
                    gt.bbox.center_x, gt.bbox.center_y, gt.bbox.center_z,
                    gt.bbox.length, gt.bbox.width, gt.bbox.height, gt.bbox.heading
                ]),
                confidence=gt.confidence
            )
            for gt in self.ground_truth_objects
        ]
        
        det_results = [
            DetectionResult(
                object_type=det.object_type,
                bbox=np.array([
                    det.bbox.center_x, det.bbox.center_y, det.bbox.center_z,
                    det.bbox.length, det.bbox.width, det.bbox.height, det.bbox.heading
                ]),
                confidence=det.confidence
            )
            for det in self.detected_objects
        ]
        
        return self.metrics_calculator.calculate_detection_metrics(
            gt_results, det_results
        ).to_dict()


class PlanningModuleSimulator(ModuleSimulator):
    """规划模块仿真器"""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("PlanningModule", config)
        from metrics.planning_metrics import PlanningMetrics
        
        self.metrics_calculator = PlanningMetrics()
        self.current_trajectory = None
        self.obstacles = []
    
    def initialize(self) -> bool:
        """初始化规划模块"""
        super().initialize()
        self.current_trajectory = None
        self.obstacles = []
        return True
    
    def step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规划模块单步
        
        Args:
            input_data: 包含障碍物和目标位置的字典
            
        Returns:
            规划轨迹
        """
        # 模拟规划处理延迟
        time.sleep(0.002)
        
        # 获取障碍物
        self.obstacles = input_data.get('obstacles', [])
        
        # 生成规划轨迹
        from metrics.planning_metrics import Trajectory, TrajectoryPoint
        
        points = []
        target_x = input_data.get('target_x', 100.0)
        target_y = input_data.get('target_y', 0.0)
        
        num_points = 50
        for i in range(num_points):
            t = i * self.config.time_step
            ratio = i / num_points
            
            point = TrajectoryPoint(
                x=target_x * ratio,
                y=target_y * ratio,
                heading=np.arctan2(target_y, target_x),
                velocity=10.0,
                acceleration=0.0,
                curvature=0.0,
                timestamp=t
            )
            points.append(point)
        
        self.current_trajectory = Trajectory(points=points)
        
        return {
            'trajectory': self.current_trajectory,
            'num_points': len(points),
            'processing_time': 0.002
        }
    
    def get_planning_metrics(self) -> Dict:
        """获取规划指标"""
        if self.current_trajectory is None:
            return {}
        
        metrics = self.metrics_calculator.calculate_trajectory_metrics(
            self.current_trajectory,
            obstacles=self.obstacles
        )
        
        return metrics.to_dict()


class SensorModuleSimulator(ModuleSimulator):
    """传感器模块仿真器"""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("SensorModule", config)
        from data_generators.sensor_data_generator import SensorDataGenerator
        
        self.sensor_generator = SensorDataGenerator(seed=config.random_seed)
        self.sensor_data = {}
    
    def initialize(self) -> bool:
        """初始化传感器模块"""
        super().initialize()
        self.sensor_data = {}
        return True
    
    def step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行传感器模块单步
        
        Args:
            input_data: 包含传感器配置的字典
            
        Returns:
            传感器数据
        """
        # 模拟传感器采集延迟
        time.sleep(0.0005)
        
        timestamp = self.current_time
        
        # 生成传感器数据
        self.sensor_data = {
            'lidar': self.sensor_generator.generate_lidar_pointcloud(
                num_points=input_data.get('lidar_points', 1000),
                timestamp=timestamp
            ),
            'camera': self.sensor_generator.generate_camera_image(
                timestamp=timestamp
            ),
            'radar': self.sensor_generator.generate_radar_data(
                num_targets=input_data.get('radar_targets', 10),
                timestamp=timestamp
            ),
            'imu': self.sensor_generator.generate_imu_data(
                duration=self.config.time_step
            )[0],
            'gps': self.sensor_generator.generate_gps_data(
                duration=self.config.time_step
            )[0],
        }
        
        return {
            'sensor_data': self.sensor_data,
            'timestamp': timestamp,
            'processing_time': 0.0005
        }


class SimulationRunner:
    """仿真运行器"""
    
    def __init__(self):
        self.simulators: Dict[str, ModuleSimulator] = {}
        self.results: Dict[str, SimulationResult] = {}
    
    def register_simulator(self, simulator: ModuleSimulator):
        """注册仿真器"""
        self.simulators[simulator.name] = simulator
    
    def run_simulation(
        self,
        simulator_name: str,
        input_generator: Callable[[float], Dict[str, Any]]
    ) -> SimulationResult:
        """
        运行单个仿真
        
        Args:
            simulator_name: 仿真器名称
            input_generator: 输入数据生成函数
            
        Returns:
            仿真结果
        """
        if simulator_name not in self.simulators:
            raise ValueError(f"Simulator {simulator_name} not found")
        
        simulator = self.simulators[simulator_name]
        simulator.reset()
        
        if not simulator.initialize():
            return SimulationResult(
                status=SimulationStatus.ERROR,
                metrics=SimulationMetrics(),
                errors=["Initialization failed"]
            )
        
        result = simulator.run(input_generator)
        self.results[simulator_name] = result
        
        return result
    
    def run_all_simulations(
        self,
        input_generators: Dict[str, Callable[[float], Dict[str, Any]]]
    ) -> Dict[str, SimulationResult]:
        """
        运行所有仿真
        
        Args:
            input_generators: 各仿真器的输入数据生成函数
            
        Returns:
            所有仿真结果
        """
        for name, generator in input_generators.items():
            self.run_simulation(name, generator)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """获取仿真摘要"""
        summary = {
            'total_simulations': len(self.results),
            'completed': 0,
            'errors': 0,
            'total_execution_time': 0.0,
            'results': {}
        }
        
        for name, result in self.results.items():
            if result.status == SimulationStatus.COMPLETED:
                summary['completed'] += 1
            elif result.status == SimulationStatus.ERROR:
                summary['errors'] += 1
            
            summary['total_execution_time'] += result.metrics.execution_time
            summary['results'][name] = {
                'status': result.status.value,
                'execution_time': result.metrics.execution_time,
                'total_steps': result.metrics.total_steps
            }
        
        return summary
    
    def export_results(self, filepath: str):
        """导出仿真结果到文件"""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
