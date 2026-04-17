#!/usr/bin/env python3
"""
自动驾驶模块测试运行脚本
支持运行单元测试、仿真测试、回归测试
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regression.regression_framework import (
    RegressionTestRunner, TestSuiteResult, TestResult, TestCase, TestStatus
)


class TestRunner:
    """测试运行器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        """
        初始化测试运行器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self.regression_runner = RegressionTestRunner(output_dir=output_dir)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def run_unit_tests(
        self,
        verbose: bool = True,
        markers: Optional[List[str]] = None
    ) -> int:
        """
        运行单元测试
        
        Args:
            verbose: 是否显示详细信息
            markers: 要运行的测试标记
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行单元测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "unit"])
        
        if markers:
            args.extend(["-m", " and ".join(markers)])
        
        args.extend(["unit/"])
        
        return pytest.main(args)
    
    def run_simulation_tests(
        self,
        verbose: bool = True,
        exclude_slow: bool = False
    ) -> int:
        """
        运行仿真测试
        
        Args:
            verbose: 是否显示详细信息
            exclude_slow: 是否排除慢速测试
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行仿真测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        
        if exclude_slow:
            args.extend(["-m", "simulation and not slow"])
        else:
            args.extend(["-m", "simulation"])
        
        args.extend(["simulation/"])
        
        return pytest.main(args)
    
    def run_regression_tests(
        self,
        verbose: bool = True,
        generate_report: bool = True
    ) -> int:
        """
        运行回归测试
        
        Args:
            verbose: 是否显示详细信息
            generate_report: 是否生成报告
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行回归测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "regression"])
        args.extend(["regression/"])
        
        exit_code = pytest.main(args)
        
        if generate_report:
            self._generate_regression_report()
        
        return exit_code
    
    def run_all_tests(
        self,
        verbose: bool = True,
        exclude_slow: bool = False,
        generate_report: bool = True
    ) -> int:
        """
        运行所有测试
        
        Args:
            verbose: 是否显示详细信息
            exclude_slow: 是否排除慢速测试
            generate_report: 是否生成报告
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行所有测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        
        if exclude_slow:
            args.extend(["-m", "not slow"])
        
        args.extend(["unit/", "simulation/", "regression/"])
        
        exit_code = pytest.main(args)
        
        if generate_report:
            self._generate_regression_report()
        
        return exit_code
    
    def run_perception_tests(self, verbose: bool = True) -> int:
        """
        运行感知模块测试
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行感知模块测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "perception"])
        
        return pytest.main(args)
    
    def run_planning_tests(self, verbose: bool = True) -> int:
        """
        运行规划模块测试
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行规划模块测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "planning"])
        
        return pytest.main(args)
    
    def run_sensor_tests(self, verbose: bool = True) -> int:
        """
        运行传感器测试
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行传感器测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "sensor"])
        
        return pytest.main(args)
    
    def run_performance_tests(self, verbose: bool = True) -> int:
        """
        运行性能测试
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print("运行性能测试")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.extend(["-m", "performance or slow"])
        
        return pytest.main(args)
    
    def run_specific_test(self, test_path: str, verbose: bool = True) -> int:
        """
        运行特定测试
        
        Args:
            test_path: 测试文件或目录路径
            verbose: 是否显示详细信息
            
        Returns:
            退出码
        """
        print("\n" + "="*60)
        print(f"运行测试: {test_path}")
        print("="*60)
        
        args = ["-v"] if verbose else []
        args.append(test_path)
        
        return pytest.main(args)
    
    def _generate_regression_report(self):
        """生成回归测试报告"""
        print("\n" + "="*60)
        print("生成测试报告")
        print("="*60)
        
        report_name = datetime.now().strftime("regression_%Y%m%d_%H%M%S")
        report_files = self.regression_runner.generate_report(
            report_name=report_name,
            formats=["json", "html"]
        )
        
        print(f"报告已生成:")
        for format_type, filepath in report_files.items():
            print(f"  {format_type.upper()}: {filepath}")
    
    def print_help(self):
        """打印帮助信息"""
        help_text = """
自动驾驶模块测试框架

用法: python run_tests.py [选项]

选项:
  --unit              运行单元测试
  --simulation        运行仿真测试
  --regression        运行回归测试
  --all               运行所有测试
  --perception        运行感知模块测试
  --planning          运行规划模块测试
  --sensor            运行传感器测试
  --performance       运行性能测试
  --test PATH         运行特定测试文件或目录
  --no-slow           排除慢速测试
  --no-report         不生成报告
  --quiet             减少输出信息
  --help              显示帮助信息

示例:
  python run_tests.py --unit                    # 运行单元测试
  python run_tests.py --all --no-slow           # 运行所有测试（排除慢速）
  python run_tests.py --perception              # 运行感知模块测试
  python run_tests.py --test unit/test_sensor.py # 运行特定测试文件
        """
        print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动驾驶模块测试框架",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 测试类型选项
    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="运行单元测试"
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="运行仿真测试"
    )
    parser.add_argument(
        "--regression", "-r",
        action="store_true",
        help="运行回归测试"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有测试"
    )
    parser.add_argument(
        "--perception", "-p",
        action="store_true",
        help="运行感知模块测试"
    )
    parser.add_argument(
        "--planning",
        action="store_true",
        help="运行规划模块测试"
    )
    parser.add_argument(
        "--sensor",
        action="store_true",
        help="运行传感器测试"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="运行性能测试"
    )
    parser.add_argument(
        "--test", "-t",
        type=str,
        help="运行特定测试文件或目录"
    )
    
    # 其他选项
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="排除慢速测试"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成报告"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="减少输出信息"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_reports",
        help="报告输出目录"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认运行单元测试
    if not any([
        args.unit, args.simulation, args.regression, args.all,
        args.perception, args.planning, args.sensor, args.performance,
        args.test
    ]):
        args.unit = True
    
    # 创建测试运行器
    runner = TestRunner(output_dir=args.output_dir)
    
    verbose = not args.quiet
    generate_report = not args.no_report
    
    start_time = time.time()
    exit_codes = []
    
    # 运行指定的测试
    if args.test:
        exit_codes.append(runner.run_specific_test(args.test, verbose))
    
    if args.unit or args.all:
        exit_codes.append(runner.run_unit_tests(verbose))
    
    if args.simulation or args.all:
        exit_codes.append(runner.run_simulation_tests(
            verbose, exclude_slow=args.no_slow
        ))
    
    if args.regression or args.all:
        exit_codes.append(runner.run_regression_tests(
            verbose, generate_report=generate_report
        ))
    
    if args.perception:
        exit_codes.append(runner.run_perception_tests(verbose))
    
    if args.planning:
        exit_codes.append(runner.run_planning_tests(verbose))
    
    if args.sensor:
        exit_codes.append(runner.run_sensor_tests(verbose))
    
    if args.performance:
        exit_codes.append(runner.run_performance_tests(verbose))
    
    # 打印总结
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"总耗时: {elapsed_time:.2f}秒")
    
    # 返回退出码
    if any(code != 0 for code in exit_codes):
        print("部分测试失败")
        return 1
    else:
        print("所有测试通过")
        return 0


if __name__ == "__main__":
    sys.exit(main())
