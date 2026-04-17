"""
模块级仿真测试
测试感知、规划、传感器模块的仿真
"""

import pytest
import numpy as np
from typing import Dict, Any

from simulation.simulation_framework import (
    SimulationConfig, SimulationStatus,
    PerceptionModuleSimulator, PlanningModuleSimulator,
    SensorModuleSimulator, SimulationRunner
)


@pytest.mark.simulation
class TestPerceptionSimulation:
    """感知模块仿真测试"""
    
    def test_perception_initialization(self):
        """测试感知模块初始化"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        
        assert simulator.initialize() is True
        assert simulator.status == SimulationStatus.IDLE
    
    def test_perception_single_step(self):
        """测试感知模块单步执行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        simulator.initialize()
        
        input_data = {
            'num_objects': 5,
            'noise_level': 0.1,
            'miss_rate': 0.05
        }
        
        output = simulator.step(input_data)
        
        assert 'num_detections' in output
        assert 'detections' in output
        assert output['num_detections'] >= 0
    
    def test_perception_simulation_run(self):
        """测试感知模块完整仿真运行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'num_objects': np.random.randint(3, 8),
                'noise_level': 0.1,
                'miss_rate': 0.05
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.total_steps == 10
        assert result.metrics.execution_time > 0
    
    def test_perception_metrics_calculation(self):
        """测试感知指标计算"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'num_objects': 5,
                'noise_level': 0.1,
                'miss_rate': 0.0
            }
        
        simulator.run(input_generator)
        metrics = simulator.get_detection_metrics()
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_perception_with_different_noise_levels(self):
        """测试不同噪声水平的感知仿真"""
        noise_levels = [0.0, 0.1, 0.2, 0.5]
        
        for noise_level in noise_levels:
            config = SimulationConfig(duration=0.5, time_step=0.1)
            simulator = PerceptionModuleSimulator(config)
            
            def input_generator(t: float) -> Dict[str, Any]:
                return {
                    'num_objects': 5,
                    'noise_level': noise_level,
                    'miss_rate': 0.0
                }
            
            result = simulator.run(input_generator)
            assert result.status == SimulationStatus.COMPLETED


@pytest.mark.simulation
class TestPlanningSimulation:
    """规划模块仿真测试"""
    
    def test_planning_initialization(self):
        """测试规划模块初始化"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PlanningModuleSimulator(config)
        
        assert simulator.initialize() is True
        assert simulator.status == SimulationStatus.IDLE
    
    def test_planning_single_step(self):
        """测试规划模块单步执行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PlanningModuleSimulator(config)
        simulator.initialize()
        
        input_data = {
            'target_x': 100.0,
            'target_y': 0.0,
            'obstacles': []
        }
        
        output = simulator.step(input_data)
        
        assert 'trajectory' in output
        assert 'num_points' in output
        assert output['num_points'] > 0
    
    def test_planning_simulation_run(self):
        """测试规划模块完整仿真运行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PlanningModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'target_x': 50.0 + t * 10,
                'target_y': np.sin(t) * 5,
                'obstacles': []
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.total_steps == 10
    
    def test_planning_metrics_calculation(self):
        """测试规划指标计算"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PlanningModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'target_x': 100.0,
                'target_y': 0.0,
                'obstacles': []
            }
        
        simulator.run(input_generator)
        metrics = simulator.get_planning_metrics()
        
        assert 'smoothness' in metrics
        assert 'safety_score' in metrics
        assert 'comfort_score' in metrics
        assert 0 <= metrics['smoothness'] <= 1
        assert 0 <= metrics['safety_score'] <= 1
    
    def test_planning_with_obstacles(self):
        """测试带障碍物的规划仿真"""
        from metrics.planning_metrics import Obstacle
        
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = PlanningModuleSimulator(config)
        
        obstacles = [
            Obstacle(x=30, y=2, radius=1.5, velocity_x=0, velocity_y=0),
            Obstacle(x=60, y=-2, radius=2.0, velocity_x=0, velocity_y=0),
        ]
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'target_x': 100.0,
                'target_y': 0.0,
                'obstacles': obstacles
            }
        
        result = simulator.run(input_generator)
        assert result.status == SimulationStatus.COMPLETED
        
        metrics = simulator.get_planning_metrics()
        assert metrics['safety_score'] < 1.0  # 有障碍物时安全性应该降低


@pytest.mark.simulation
class TestSensorSimulation:
    """传感器模块仿真测试"""
    
    def test_sensor_initialization(self):
        """测试传感器模块初始化"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = SensorModuleSimulator(config)
        
        assert simulator.initialize() is True
        assert simulator.status == SimulationStatus.IDLE
    
    def test_sensor_single_step(self):
        """测试传感器模块单步执行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = SensorModuleSimulator(config)
        simulator.initialize()
        
        input_data = {
            'lidar_points': 1000,
            'radar_targets': 10
        }
        
        output = simulator.step(input_data)
        
        assert 'sensor_data' in output
        assert 'timestamp' in output
        assert 'lidar' in output['sensor_data']
        assert 'camera' in output['sensor_data']
        assert 'radar' in output['sensor_data']
    
    def test_sensor_simulation_run(self):
        """测试传感器模块完整仿真运行"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = SensorModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'lidar_points': 1000,
                'radar_targets': 10
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.total_steps == 10
    
    def test_sensor_data_consistency(self):
        """测试传感器数据一致性"""
        config = SimulationConfig(duration=1.0, time_step=0.1)
        simulator = SensorModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'lidar_points': 100,
                'radar_targets': 5
            }
        
        result = simulator.run(input_generator)
        
        # 检查所有时间戳是否递增
        timestamps = []
        for log_entry in result.data['log']:
            timestamps.append(log_entry['output']['timestamp'])
        
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]


@pytest.mark.simulation
class TestSimulationRunner:
    """仿真运行器测试"""
    
    def test_runner_registration(self):
        """测试仿真器注册"""
        runner = SimulationRunner()
        
        config = SimulationConfig(duration=0.5, time_step=0.1)
        perception_sim = PerceptionModuleSimulator(config)
        
        runner.register_simulator(perception_sim)
        
        assert 'PerceptionModule' in runner.simulators
    
    def test_runner_single_simulation(self):
        """测试运行单个仿真"""
        runner = SimulationRunner()
        
        config = SimulationConfig(duration=0.5, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        runner.register_simulator(simulator)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {'num_objects': 5}
        
        result = runner.run_simulation('PerceptionModule', input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert 'PerceptionModule' in runner.results
    
    def test_runner_multiple_simulations(self):
        """测试运行多个仿真"""
        runner = SimulationRunner()
        
        config = SimulationConfig(duration=0.5, time_step=0.1)
        
        perception_sim = PerceptionModuleSimulator(config)
        planning_sim = PlanningModuleSimulator(config)
        sensor_sim = SensorModuleSimulator(config)
        
        runner.register_simulator(perception_sim)
        runner.register_simulator(planning_sim)
        runner.register_simulator(sensor_sim)
        
        input_generators = {
            'PerceptionModule': lambda t: {'num_objects': 5},
            'PlanningModule': lambda t: {'target_x': 100.0, 'obstacles': []},
            'SensorModule': lambda t: {'lidar_points': 100}
        }
        
        results = runner.run_all_simulations(input_generators)
        
        assert len(results) == 3
        for result in results.values():
            assert result.status == SimulationStatus.COMPLETED
    
    def test_runner_summary(self):
        """测试仿真摘要"""
        runner = SimulationRunner()
        
        config = SimulationConfig(duration=0.5, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        runner.register_simulator(simulator)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {'num_objects': 5}
        
        runner.run_simulation('PerceptionModule', input_generator)
        
        summary = runner.get_summary()
        
        assert summary['total_simulations'] == 1
        assert summary['completed'] == 1
        assert 'results' in summary
    
    def test_runner_export(self, tmp_path):
        """测试仿真结果导出"""
        runner = SimulationRunner()
        
        config = SimulationConfig(duration=0.5, time_step=0.1)
        simulator = PerceptionModuleSimulator(config)
        runner.register_simulator(simulator)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {'num_objects': 5}
        
        runner.run_simulation('PerceptionModule', input_generator)
        
        export_path = tmp_path / "simulation_results.json"
        runner.export_results(str(export_path))
        
        assert export_path.exists()


@pytest.mark.simulation
@pytest.mark.slow
class TestSimulationPerformance:
    """仿真性能测试"""
    
    def test_perception_performance(self):
        """测试感知模块性能"""
        config = SimulationConfig(duration=5.0, time_step=0.05)
        simulator = PerceptionModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'num_objects': 20,
                'noise_level': 0.1,
                'miss_rate': 0.05
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.total_steps == 100
        
        # 检查性能指标
        assert result.metrics.average_step_time < 0.01  # 每步应该小于10ms
    
    def test_planning_performance(self):
        """测试规划模块性能"""
        config = SimulationConfig(duration=5.0, time_step=0.05)
        simulator = PlanningModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'target_x': 100.0,
                'target_y': 0.0,
                'obstacles': []
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.average_step_time < 0.02  # 每步应该小于20ms
    
    def test_sensor_performance(self):
        """测试传感器模块性能"""
        config = SimulationConfig(duration=5.0, time_step=0.05)
        simulator = SensorModuleSimulator(config)
        
        def input_generator(t: float) -> Dict[str, Any]:
            return {
                'lidar_points': 5000,
                'radar_targets': 20
            }
        
        result = simulator.run(input_generator)
        
        assert result.status == SimulationStatus.COMPLETED
        assert result.metrics.average_step_time < 0.005  # 每步应该小于5ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
