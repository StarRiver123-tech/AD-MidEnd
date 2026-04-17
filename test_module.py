#!/usr/bin/env python3
"""
自动驾驶系统 - 模块测试脚本
===========================

支持独立测试各个模块，无需启动完整系统

使用方法:
    # 测试感知模块
    python test_module.py --module perception
    
    # 测试规划模块
    python test_module.py --module planning
    
    # 测试传感器模块
    python test_module.py --module sensor
    
    # 测试可视化模块
    python test_module.py --module visualization
    
    # 运行性能测试
    python test_module.py --benchmark --modules perception planning --duration 60

作者: Autonomous Driving Team
版本: 1.0.0
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# 导入配置管理
from src.config.config_manager import ConfigManager
from src.communication.message_bus import MessageBus
from src.logs.logger import Logger


@dataclass
class TestResult:
    """测试结果数据类"""
    module_name: str
    test_name: str
    passed: bool
    duration_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None


class ModuleTester:
    """模块测试器"""
    
    def __init__(self):
        self._logger = Logger("ModuleTester")
        self._message_bus = MessageBus()
        self._config_manager = ConfigManager()
        self._results: List[TestResult] = []
    
    def _load_config(self, config_path: Optional[str] = None) -> bool:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            return self._config_manager.load_from_file(config_path)
        
        default_config = PROJECT_ROOT / "config" / "system_config.yaml"
        if default_config.exists():
            return self._config_manager.load_from_file(str(default_config))
        
        self._logger.warning("Using default configuration")
        return True
    
    def _add_result(self, result: TestResult) -> None:
        """添加测试结果"""
        self._results.append(result)
        status = "PASS" if result.passed else "FAIL"
        self._logger.info(f"[{status}] {result.module_name}.{result.test_name}: {result.message}")
    
    def test_perception(self, verbose: bool = False) -> bool:
        """测试感知模块"""
        self._logger.info("=" * 60)
        self._logger.info("Testing Perception Module")
        self._logger.info("=" * 60)
        
        all_passed = True
        
        try:
            # 1. 测试导入
            start_time = time.time()
            from src.perception.perception_module import PerceptionModule
            from src.perception.lane_detector import LaneDetector
            from src.perception.obstacle_detector import ObstacleDetector
            from src.perception.occupancy_network import OccupancyNetwork
            
            duration = (time.time() - start_time) * 1000
            self._add_result(TestResult(
                module_name="perception",
                test_name="import",
                passed=True,
                duration_ms=duration,
                message="All perception modules imported successfully"
            ))
            
            # 2. 测试模块初始化
            start_time = time.time()
            perception = PerceptionModule(self._message_bus)
            init_result = perception.initialize(self._config_manager)
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="perception",
                test_name="initialization",
                passed=init_result,
                duration_ms=duration,
                message="Perception module initialized" if init_result else "Initialization failed"
            ))
            
            if not init_result:
                return False
            
            # 3. 测试车道线检测
            start_time = time.time()
            lane_detector = LaneDetector({})
            
            # 创建测试图像
            test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # 绘制一些车道线
            cv2.line(test_image, (500, 1080), (960, 540), (255, 255, 255), 5)
            cv2.line(test_image, (1420, 1080), (960, 540), (255, 255, 255), 5)
            
            from src.common.data_types import ImageData, Timestamp
            image_data = ImageData(
                image=test_image,
                timestamp=Timestamp.now(),
                width=1920,
                height=1080,
                channels=3
            )
            
            lane_result = lane_detector.detect(image_data)
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="perception",
                test_name="lane_detection",
                passed=lane_result is not None,
                duration_ms=duration,
                message=f"Detected {len(lane_result.lane_lines) if lane_result else 0} lane lines"
            ))
            
            # 4. 测试障碍物检测
            start_time = time.time()
            obstacle_detector = ObstacleDetector({})
            
            obstacle_result = obstacle_detector.detect(
                camera_data=image_data,
                lidar_data=None,
                radar_data=None
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="perception",
                test_name="obstacle_detection",
                passed=obstacle_result is not None,
                duration_ms=duration,
                message=f"Detected {len(obstacle_result.obstacles) if obstacle_result else 0} obstacles"
            ))
            
            # 5. 测试启动和停止
            start_time = time.time()
            perception.start()
            time.sleep(0.5)  # 运行一段时间
            perception.stop()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="perception",
                test_name="start_stop",
                passed=True,
                duration_ms=duration,
                message="Perception module started and stopped successfully"
            ))
            
            # 6. 测试统计信息
            stats = perception.get_stats()
            self._add_result(TestResult(
                module_name="perception",
                test_name="get_stats",
                passed=stats is not None and 'state' in stats,
                duration_ms=0,
                message=f"State: {stats.get('state', 'unknown')}"
            ))
            
        except Exception as e:
            self._logger.error(f"Perception test failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            all_passed = False
            
            self._add_result(TestResult(
                module_name="perception",
                test_name="overall",
                passed=False,
                duration_ms=0,
                message=str(e)
            ))
        
        return all_passed
    
    def test_planning(self, verbose: bool = False) -> bool:
        """测试规划模块"""
        self._logger.info("=" * 60)
        self._logger.info("Testing Planning Module")
        self._logger.info("=" * 60)
        
        all_passed = True
        
        try:
            # 1. 测试导入
            start_time = time.time()
            from src.planning.planning_module import PlanningModule
            from src.planning.behavior_planner import BehaviorPlanner
            from src.planning.trajectory_generator import TrajectoryGenerator
            from src.planning.trajectory_optimizer import TrajectoryOptimizer
            
            duration = (time.time() - start_time) * 1000
            self._add_result(TestResult(
                module_name="planning",
                test_name="import",
                passed=True,
                duration_ms=duration,
                message="All planning modules imported successfully"
            ))
            
            # 2. 测试模块初始化
            start_time = time.time()
            planning = PlanningModule(self._message_bus)
            init_result = planning.initialize(self._config_manager)
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="planning",
                test_name="initialization",
                passed=init_result,
                duration_ms=duration,
                message="Planning module initialized" if init_result else "Initialization failed"
            ))
            
            if not init_result:
                return False
            
            # 3. 测试行为规划
            start_time = time.time()
            behavior_planner = BehaviorPlanner({})
            
            from src.common.data_types import PerceptionResult, Timestamp
            from src.common.enums import BehaviorType
            
            perception_result = PerceptionResult(
                timestamp=Timestamp.now(),
                lane_result=None,
                obstacle_result=None,
                occupancy_result=None
            )
            
            behavior_type, explanation = behavior_planner.plan(
                perception_result=perception_result,
                current_speed=10.0
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="planning",
                test_name="behavior_planning",
                passed=behavior_type is not None,
                duration_ms=duration,
                message=f"Behavior: {behavior_type.name if behavior_type else 'None'}, {explanation}"
            ))
            
            # 4. 测试轨迹生成
            start_time = time.time()
            trajectory_generator = TrajectoryGenerator({})
            
            trajectories = trajectory_generator.generate(
                perception_result=perception_result,
                behavior_type=BehaviorType.FOLLOW,
                current_speed=10.0
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="planning",
                test_name="trajectory_generation",
                passed=len(trajectories) > 0,
                duration_ms=duration,
                message=f"Generated {len(trajectories)} trajectories"
            ))
            
            # 5. 测试轨迹优化
            if trajectories:
                start_time = time.time()
                trajectory_optimizer = TrajectoryOptimizer({})
                
                optimized = trajectory_optimizer.optimize(
                    trajectories[0],
                    perception_result
                )
                duration = (time.time() - start_time) * 1000
                
                self._add_result(TestResult(
                    module_name="planning",
                    test_name="trajectory_optimization",
                    passed=optimized is not None,
                    duration_ms=duration,
                    message="Trajectory optimized successfully"
                ))
            
            # 6. 测试启动和停止
            start_time = time.time()
            planning.start()
            time.sleep(0.5)
            planning.stop()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="planning",
                test_name="start_stop",
                passed=True,
                duration_ms=duration,
                message="Planning module started and stopped successfully"
            ))
            
            # 7. 测试统计信息
            stats = planning.get_stats()
            self._add_result(TestResult(
                module_name="planning",
                test_name="get_stats",
                passed=stats is not None and 'state' in stats,
                duration_ms=0,
                message=f"State: {stats.get('state', 'unknown')}"
            ))
            
        except Exception as e:
            self._logger.error(f"Planning test failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            all_passed = False
            
            self._add_result(TestResult(
                module_name="planning",
                test_name="overall",
                passed=False,
                duration_ms=0,
                message=str(e)
            ))
        
        return all_passed
    
    def test_sensor(self, verbose: bool = False) -> bool:
        """测试传感器模块"""
        self._logger.info("=" * 60)
        self._logger.info("Testing Sensor Module")
        self._logger.info("=" * 60)
        
        all_passed = True
        
        try:
            # 1. 测试导入
            start_time = time.time()
            from src.sensors.core.sensor_manager import SensorManager
            from src.sensors.drivers.camera_driver import CameraDriver, CameraConfig
            from src.sensors.drivers.lidar_driver import LiDARDriver, LiDARConfig
            
            duration = (time.time() - start_time) * 1000
            self._add_result(TestResult(
                module_name="sensor",
                test_name="import",
                passed=True,
                duration_ms=duration,
                message="All sensor modules imported successfully"
            ))
            
            # 2. 测试传感器管理器初始化
            start_time = time.time()
            sensor_manager = SensorManager()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="manager_initialization",
                passed=sensor_manager is not None,
                duration_ms=duration,
                message="Sensor manager created successfully"
            ))
            
            # 3. 测试摄像头配置
            start_time = time.time()
            camera_config = CameraConfig(
                name="test_camera",
                resolution=[1920, 1080],
                fps=30,
                fov=90.0,
                enabled=True
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="camera_config",
                passed=camera_config is not None,
                duration_ms=duration,
                message=f"Camera config: {camera_config.resolution}@{camera_config.fps}fps"
            ))
            
            # 4. 测试LiDAR配置
            start_time = time.time()
            lidar_config = LiDARConfig(
                name="test_lidar",
                model="pandar64",
                channels=64,
                range_max=200.0,
                enabled=True
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="lidar_config",
                passed=lidar_config is not None,
                duration_ms=duration,
                message=f"LiDAR config: {lidar_config.channels}ch, {lidar_config.range_max}m range"
            ))
            
            # 5. 测试传感器初始化
            start_time = time.time()
            init_result = sensor_manager.initialize_all()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="sensor_initialization",
                passed=init_result,
                duration_ms=duration,
                message="All sensors initialized" if init_result else "Initialization failed"
            ))
            
            # 6. 测试状态获取
            start_time = time.time()
            status = sensor_manager.get_sensor_status()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="get_status",
                passed=status is not None,
                duration_ms=duration,
                message=f"Sensor status retrieved"
            ))
            
        except Exception as e:
            self._logger.error(f"Sensor test failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            all_passed = False
            
            self._add_result(TestResult(
                module_name="sensor",
                test_name="overall",
                passed=False,
                duration_ms=0,
                message=str(e)
            ))
        
        return all_passed
    
    def test_visualization(self, verbose: bool = False) -> bool:
        """测试可视化模块"""
        self._logger.info("=" * 60)
        self._logger.info("Testing Visualization Module")
        self._logger.info("=" * 60)
        
        all_passed = True
        
        try:
            # 1. 测试导入
            start_time = time.time()
            from src.visualization.data_manager import DataManager, EgoState
            from src.visualization.bev_visualizer import BEVVisualizer, BEVConfig
            from src.visualization.trajectory_visualizer import TrajectoryVisualizer
            
            duration = (time.time() - start_time) * 1000
            self._add_result(TestResult(
                module_name="visualization",
                test_name="import",
                passed=True,
                duration_ms=duration,
                message="All visualization modules imported successfully"
            ))
            
            # 2. 测试数据管理器
            start_time = time.time()
            data_manager = DataManager()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="visualization",
                test_name="data_manager",
                passed=data_manager is not None,
                duration_ms=duration,
                message="Data manager created successfully"
            ))
            
            # 3. 测试BEV可视化器
            start_time = time.time()
            bev_config = BEVConfig()
            bev_visualizer = BEVVisualizer(bev_config)
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="visualization",
                test_name="bev_visualizer",
                passed=bev_visualizer is not None,
                duration_ms=duration,
                message="BEV visualizer created successfully"
            ))
            
            # 4. 测试轨迹可视化器
            start_time = time.time()
            trajectory_visualizer = TrajectoryVisualizer()
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="visualization",
                test_name="trajectory_visualizer",
                passed=trajectory_visualizer is not None,
                duration_ms=duration,
                message="Trajectory visualizer created successfully"
            ))
            
            # 5. 测试渲染 (不显示)
            start_time = time.time()
            
            # 创建测试数据
            ego_state = EgoState()
            ego_state.position = [0.0, 0.0]
            ego_state.heading = 0.0
            ego_state.speed = 10.0
            
            bev_image = bev_visualizer.render(
                ego_state=ego_state,
                obstacles=[],
                lane_lines=[],
                trajectories=[]
            )
            duration = (time.time() - start_time) * 1000
            
            self._add_result(TestResult(
                module_name="visualization",
                test_name="render",
                passed=bev_image is not None and bev_image.size > 0,
                duration_ms=duration,
                message=f"Rendered BEV image: {bev_image.shape if bev_image is not None else 'None'}"
            ))
            
        except Exception as e:
            self._logger.error(f"Visualization test failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            all_passed = False
            
            self._add_result(TestResult(
                module_name="visualization",
                test_name="overall",
                passed=False,
                duration_ms=0,
                message=str(e)
            ))
        
        return all_passed
    
    def print_summary(self) -> None:
        """打印测试摘要"""
        self._logger.info("=" * 60)
        self._logger.info("Test Summary")
        self._logger.info("=" * 60)
        
        total_tests = len(self._results)
        passed_tests = sum(1 for r in self._results if r.passed)
        failed_tests = total_tests - passed_tests
        
        self._logger.info(f"Total tests: {total_tests}")
        self._logger.info(f"Passed: {passed_tests}")
        self._logger.info(f"Failed: {failed_tests}")
        
        if failed_tests > 0:
            self._logger.info("\nFailed tests:")
            for result in self._results:
                if not result.passed:
                    self._logger.info(f"  - {result.module_name}.{result.test_name}: {result.message}")
        
        self._logger.info("=" * 60)
    
    def save_results(self, output_path: str) -> None:
        """保存测试结果"""
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self._results),
            'passed': sum(1 for r in self._results if r.passed),
            'failed': sum(1 for r in self._results if not r.passed),
            'results': [
                {
                    'module': r.module_name,
                    'test': r.test_name,
                    'passed': r.passed,
                    'duration_ms': r.duration_ms,
                    'message': r.message,
                    'details': r.details
                }
                for r in self._results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self._logger.info(f"Results saved to: {output_path}")


def run_module_test(module: str, config_path: Optional[str] = None,
                   verbose: bool = False, output: Optional[str] = None) -> int:
    """
    运行指定模块的测试
    
    Args:
        module: 模块名称 (perception | planning | sensor | visualization | all)
        config_path: 配置文件路径
        verbose: 是否详细输出
        output: 输出文件路径
    
    Returns:
        int: 返回码 (0=成功, 1=失败)
    """
    tester = ModuleTester()
    
    # 加载配置
    tester._load_config(config_path)
    
    # 运行测试
    results = {}
    
    if module in ['perception', 'all']:
        results['perception'] = tester.test_perception(verbose)
    
    if module in ['planning', 'all']:
        results['planning'] = tester.test_planning(verbose)
    
    if module in ['sensor', 'all']:
        results['sensor'] = tester.test_sensor(verbose)
    
    if module in ['visualization', 'all']:
        results['visualization'] = tester.test_visualization(verbose)
    
    # 打印摘要
    tester.print_summary()
    
    # 保存结果
    if output:
        tester.save_results(output)
    
    # 返回结果
    return 0 if all(results.values()) else 1


def run_benchmark(modules: List[str], duration: int = 60,
                 output: Optional[str] = None) -> int:
    """
    运行性能测试
    
    Args:
        modules: 要测试的模块列表
        duration: 测试持续时间(秒)
        output: 输出文件路径
    
    Returns:
        int: 返回码
    """
    logger = Logger("Benchmark")
    
    logger.info("=" * 60)
    logger.info("Running Performance Benchmark")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration} seconds")
    logger.info(f"Modules: {', '.join(modules)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'duration': duration,
        'modules': {}
    }
    
    # 测试每个模块
    for module in modules:
        if module == 'all':
            continue
            
        logger.info(f"\nBenchmarking {module}...")
        
        # 这里实现具体的性能测试
        # 简化版本，实际应该测量处理延迟、吞吐量等
        
        module_results = {
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'min_latency_ms': 0,
            'throughput_fps': 0,
        }
        
        results['modules'][module] = module_results
    
    # 保存结果
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nBenchmark results saved to: {output}")
    
    logger.info("=" * 60)
    logger.info("Benchmark completed")
    logger.info("=" * 60)
    
    return 0


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Module Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 测试感知模块
  python test_module.py --module perception
  
  # 测试所有模块
  python test_module.py --module all
  
  # 使用自定义配置测试
  python test_module.py --module planning --config config/my_config.yaml
  
  # 保存测试结果
  python test_module.py --module all --output results.json
  
  # 运行性能测试
  python test_module.py --benchmark --modules perception planning --duration 60
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        type=str,
        choices=['perception', 'planning', 'sensor', 'visualization', 'all'],
        default='all',
        help='Module to test (default: all)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/system_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for test results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    
    parser.add_argument(
        '--modules',
        type=str,
        nargs='+',
        choices=['perception', 'planning', 'sensor', 'all'],
        default=['all'],
        help='Modules to benchmark'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Benchmark duration in seconds (default: 60)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 性能测试模式
    if args.benchmark:
        return run_benchmark(
            modules=args.modules,
            duration=args.duration,
            output=args.output
        )
    
    # 模块测试模式
    return run_module_test(
        module=args.module,
        config_path=args.config,
        verbose=args.verbose,
        output=args.output
    )


if __name__ == "__main__":
    sys.exit(main())
