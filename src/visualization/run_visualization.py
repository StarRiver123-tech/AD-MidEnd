#!/usr/bin/env python3
"""
自动驾驶仿真可视化工具启动脚本

使用方法:
    python run_visualization.py [options]

选项:
    --data PATH         加载指定数据文件
    --demo              使用演示数据（默认）
    --fps N             设置播放帧率（默认10）
    --fullscreen        全屏模式
    --help              显示帮助信息

示例:
    python run_visualization.py --demo
    python run_visualization.py --data ./simulation_data.pkl --fps 20
"""

import sys
import os
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from visualization.visualizer import AutonomousDrivingVisualizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='自动驾驶仿真可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
快捷键:
    Space       - 播放/暂停
    ← / →       - 上一帧/下一帧
    Ctrl + S    - 保存截图
    Ctrl + O    - 打开数据文件
    Ctrl + Q    - 退出
        '''
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='加载指定数据文件路径'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        default=True,
        help='使用演示数据（默认）'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='设置播放帧率（默认10）'
    )
    
    parser.add_argument(
        '--fullscreen',
        action='store_true',
        help='以全屏模式启动'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1600,
        help='窗口宽度（默认1600）'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=900,
        help='窗口高度（默认900）'
    )
    
    return parser.parse_args()


def check_dependencies():
    """检查依赖项"""
    missing = []
    
    try:
        import PyQt5
    except ImportError:
        missing.append('PyQt5')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    
    if missing:
        print("错误: 缺少以下依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请使用以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def print_welcome():
    """打印欢迎信息"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         自动驾驶仿真可视化工具 v1.0                          ║
║                                                              ║
║  功能特性:                                                   ║
║    • BEV视角可视化 (自车、车道线、障碍物、Occupancy网格)     ║
║    • 规划轨迹显示 (候选轨迹、选中轨迹、行为决策)             ║
║    • 传感器数据显示 (LiDAR点云BEV/3D视图)                    ║
║    • 播放控制 (播放/暂停/步进、帧率调节)                     ║
║    • 交互功能 (缩放/平移、图层选择、截图保存)                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 打印欢迎信息
    print_welcome()
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序属性
    app.setApplicationName("自动驾驶仿真可视化工具")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = AutonomousDrivingVisualizer()
    
    # 设置窗口大小
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.resize(args.width, args.height)
        window.show()
    
    # 设置帧率
    if args.fps:
        window.set_fps(args.fps)
        print(f"播放帧率: {args.fps} FPS")
    
    # 加载数据
    if args.data:
        if os.path.exists(args.data):
            print(f"正在加载数据文件: {args.data}")
            # TODO: 实现数据文件加载
            print("注意: 数据文件加载功能待实现，使用演示数据")
            window.load_demo_data()
        else:
            print(f"错误: 数据文件不存在: {args.data}")
            sys.exit(1)
    elif args.demo:
        print("使用演示数据")
        window.load_demo_data()
    
    print("\n可视化工具已启动!")
    print("按 Space 键播放/暂停，←/→ 键切换帧\n")
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
