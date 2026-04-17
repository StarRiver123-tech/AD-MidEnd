#!/usr/bin/env python3
"""
自动驾驶系统 - 启动脚本
=======================

提供多种运行模式的便捷启动方式

使用方法:
    # 快速启动仿真模式
    python run_system.py
    
    # 启动nuScenes数据集模式
    python run_system.py --mode nuscenes --data-root /path/to/nuscenes
    
    # 启动实车模式
    python run_system.py --mode real_vehicle
    
    # 仅运行感知模块
    python run_system.py --module perception
    
    # 运行性能测试
    python run_system.py --benchmark

作者: Autonomous Driving Team
版本: 1.0.0
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")


def print_info(text: str) -> None:
    """打印信息"""
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {text}")


def print_success(text: str) -> None:
    """打印成功信息"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {text}")


def print_warning(text: str) -> None:
    """打印警告信息"""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {text}")


def print_error(text: str) -> None:
    """打印错误信息"""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {text}")


def check_dependencies() -> bool:
    """检查系统依赖"""
    print_header("Checking Dependencies")
    
    dependencies = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'yaml': 'pyyaml',
        'scipy': 'scipy',
    }
    
    optional_dependencies = {
        'PyQt5': 'PyQt5',
        'nuscenes': 'nuscenes-devkit',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
    }
    
    all_ok = True
    
    # 检查必需依赖
    print_info("Checking required dependencies...")
    for module, package in dependencies.items():
        try:
            __import__(module)
            print_success(f"  {package}")
        except ImportError:
            print_error(f"  {package} - NOT FOUND")
            all_ok = False
    
    # 检查可选依赖
    print_info("Checking optional dependencies...")
    for module, package in optional_dependencies.items():
        try:
            __import__(module)
            print_success(f"  {package}")
        except ImportError:
            print_warning(f"  {package} - NOT FOUND (optional)")
    
    if not all_ok:
        print_error("\nSome required dependencies are missing!")
        print_info("Install with: pip install -r requirements.txt")
        return False
    
    print_success("All required dependencies are installed!")
    return True


def check_system_config() -> bool:
    """检查系统配置"""
    print_header("Checking System Configuration")
    
    config_path = PROJECT_ROOT / "config" / "system_config.yaml"
    
    if not config_path.exists():
        print_warning(f"Configuration file not found: {config_path}")
        print_info("Using default configuration")
        return True
    
    print_success(f"Configuration file found: {config_path}")
    
    # 检查关键目录
    directories = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "records",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data",
    ]
    
    print_info("Checking directories...")
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print_info(f"  Created: {directory}")
        else:
            print_success(f"  Exists: {directory}")
    
    return True


def run_simulation_mode(args: argparse.Namespace) -> int:
    """运行仿真模式"""
    print_header("Running Simulation Mode")
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--mode", "simulation",
        "--config", args.config,
        "--log-level", args.log_level,
    ]
    
    if args.no_viz:
        cmd.append("--no-viz")
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def run_nuscenes_mode(args: argparse.Namespace) -> int:
    """运行nuScenes数据集模式"""
    print_header("Running nuScenes Dataset Mode")
    
    # 检查数据集路径
    if not args.data_root:
        # 尝试从配置文件读取
        import yaml
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            args.data_root = config.get('nuscenes', {}).get('data_root')
    
    if not args.data_root:
        print_error("nuScenes data root not specified!")
        print_info("Use --data-root to specify the dataset path")
        return 1
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print_error(f"nuScenes data root not found: {data_root}")
        return 1
    
    print_success(f"Using nuScenes dataset: {data_root}")
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--mode", "nuscenes",
        "--config", args.config,
        "--data-root", str(data_root),
        "--log-level", args.log_level,
    ]
    
    if args.no_viz:
        cmd.append("--no-viz")
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def run_real_vehicle_mode(args: argparse.Namespace) -> int:
    """运行实车模式"""
    print_header("Running Real Vehicle Mode")
    
    print_warning("WARNING: This mode will interact with real vehicle hardware!")
    print_warning("Make sure all safety precautions are in place.")
    
    confirm = input("\nDo you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print_info("Cancelled")
        return 0
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--mode", "real_vehicle",
        "--config", args.config,
        "--log-level", args.log_level,
    ]
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def run_module_test(module: str, args: argparse.Namespace) -> int:
    """运行模块测试"""
    print_header(f"Running {module.upper()} Module Test")
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "test_module.py"),
        "--module", module,
        "--config", args.config,
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """运行性能测试"""
    print_header("Running Performance Benchmark")
    
    from test_module import run_benchmark as benchmark_main
    
    return benchmark_main(
        modules=args.modules,
        duration=args.duration,
        output=args.output
    )


def run_visualization_only(args: argparse.Namespace) -> int:
    """仅运行可视化"""
    print_header("Running Visualization Only")
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "visualization" / "run_visualization.py"),
    ]
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def run_tests(args: argparse.Namespace) -> int:
    """运行测试套件"""
    print_header("Running Test Suite")
    
    cmd = [
        sys.executable,
        "-m", "pytest",
        str(PROJECT_ROOT / "tests"),
        "-v",
    ]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html"])
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0


def generate_report(args: argparse.Namespace) -> int:
    """生成系统报告"""
    print_header("Generating System Report")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
        },
        'project_structure': {},
        'modules': {},
    }
    
    # 扫描项目结构
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            report['project_structure'][item.name] = {
                'type': 'directory',
                'files': len(list(item.rglob('*.py')))
            }
    
    # 检查模块状态
    modules = ['perception', 'planning', 'sensors', 'visualization']
    for module in modules:
        module_path = PROJECT_ROOT / 'src' / module
        if module_path.exists():
            report['modules'][module] = {
                'status': 'available',
                'files': len(list(module_path.rglob('*.py')))
            }
        else:
            report['modules'][module] = {
                'status': 'not_found'
            }
    
    # 保存报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / 'logs' / f'system_report_{datetime.now():%Y%m%d_%H%M%S}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Report saved to: {output_path}")
    
    # 打印摘要
    print_info("\nReport Summary:")
    print(f"  Total modules: {len(report['modules'])}")
    print(f"  Available modules: {sum(1 for m in report['modules'].values() if m['status'] == 'available')}")
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Autonomous Driving System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 快速启动仿真模式
  python run_system.py
  
  # 启动nuScenes数据集模式
  python run_system.py --mode nuscenes --data-root /path/to/nuscenes
  
  # 启动实车模式
  python run_system.py --mode real_vehicle
  
  # 运行感知模块测试
  python run_system.py --test-module perception
  
  # 运行性能测试
  python run_system.py --benchmark --modules perception planning --duration 60
  
  # 仅运行可视化
  python run_system.py --visualization-only
  
  # 运行测试套件
  python run_system.py --run-tests
  
  # 生成系统报告
  python run_system.py --generate-report
        """
    )
    
    # 运行模式
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['simulation', 'nuscenes', 'real_vehicle'],
        default='simulation',
        help='Run mode (default: simulation)'
    )
    
    # 配置
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/system_config.yaml',
        help='Path to configuration file'
    )
    
    # 数据集
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        help='Path to nuScenes dataset root directory'
    )
    
    # 日志级别
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # 可视化
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--visualization-only',
        action='store_true',
        help='Run visualization only'
    )
    
    # 模块测试
    parser.add_argument(
        '--test-module',
        type=str,
        choices=['perception', 'planning', 'sensor', 'visualization', 'all'],
        help='Run specific module test'
    )
    
    # 性能测试
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
    
    # 测试套件
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Run test suite'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    # 报告生成
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate system report'
    )
    
    # 输出
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path'
    )
    
    # 详细输出
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # 跳过检查
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and config checks'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_header("Autonomous Driving System v1.0.0")
    
    # 检查依赖和配置
    if not args.skip_checks:
        if not check_dependencies():
            return 1
        if not check_system_config():
            return 1
    
    # 根据参数执行相应功能
    if args.visualization_only:
        return run_visualization_only(args)
    
    if args.test_module:
        return run_module_test(args.test_module, args)
    
    if args.benchmark:
        return run_benchmark(args)
    
    if args.run_tests:
        return run_tests(args)
    
    if args.generate_report:
        return generate_report(args)
    
    # 根据模式运行
    if args.mode == 'simulation':
        return run_simulation_mode(args)
    elif args.mode == 'nuscenes':
        return run_nuscenes_mode(args)
    elif args.mode == 'real_vehicle':
        return run_real_vehicle_mode(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
