"""
回归测试框架
支持批量测试和生成测试报告
"""

import os
import json
import time
import pytest
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import subprocess
import sys


class TestStatus(Enum):
    """测试状态"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """测试用例"""
    name: str
    module: str
    function: str
    markers: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """测试结果"""
    test_case: TestCase
    status: TestStatus
    duration: float
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'test_case': {
                'name': self.test_case.name,
                'module': self.test_case.module,
                'function': self.test_case.function,
                'markers': self.test_case.markers,
            },
            'status': self.status.value,
            'duration': self.duration,
            'message': self.message,
        }


@dataclass
class TestSuiteResult:
    """测试套件结果"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def total_tests(self) -> int:
        return len(self.results)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def skipped_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
    
    @property
    def error_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'error_tests': self.error_tests,
            'pass_rate': self.pass_rate,
            'duration': self.duration,
            'results': [r.to_dict() for r in self.results],
        }


@dataclass
class RegressionReport:
    """回归测试报告"""
    test_suites: List[TestSuiteResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tests(self) -> int:
        return sum(suite.total_tests for suite in self.test_suites)
    
    @property
    def passed_tests(self) -> int:
        return sum(suite.passed_tests for suite in self.test_suites)
    
    @property
    def failed_tests(self) -> int:
        return sum(suite.failed_tests for suite in self.test_suites)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.pass_rate,
            'duration': self.duration,
            'metadata': self.metadata,
            'test_suites': [suite.to_dict() for suite in self.test_suites],
        }
    
    def to_json(self, filepath: str):
        """导出为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_html(self, filepath: str):
        """导出为HTML报告"""
        html_content = self._generate_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html(self) -> str:
        """生成HTML报告内容"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>回归测试报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #333;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .summary-card.failed .value {{
            color: #f44336;
        }}
        .suite {{
            background-color: white;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .suite-header {{
            background-color: #2196F3;
            color: white;
            padding: 15px;
            cursor: pointer;
        }}
        .suite-content {{
            padding: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        .status-passed {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #f44336;
            font-weight: bold;
        }}
        .status-skipped {{
            color: #FF9800;
            font-weight: bold;
        }}
        .status-error {{
            color: #9C27B0;
            font-weight: bold;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>回归测试报告</h1>
        <p>生成时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>总耗时: {self.duration:.2f}秒</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>总测试数</h3>
            <div class="value">{self.total_tests}</div>
        </div>
        <div class="summary-card">
            <h3>通过</h3>
            <div class="value">{self.passed_tests}</div>
        </div>
        <div class="summary-card failed">
            <h3>失败</h3>
            <div class="value">{self.failed_tests}</div>
        </div>
        <div class="summary-card">
            <h3>通过率</h3>
            <div class="value">{self.pass_rate*100:.1f}%</div>
        </div>
    </div>
    
    <div class="progress-bar">
        <div class="progress-bar-fill" style="width: {self.pass_rate*100}%"></div>
    </div>
"""
        
        for suite in self.test_suites:
            html += f"""
    <div class="suite">
        <div class="suite-header">
            <h3>{suite.name}</h3>
            <p>通过: {suite.passed_tests}/{suite.total_tests} ({suite.pass_rate*100:.1f}%)</p>
        </div>
        <div class="suite-content">
            <table>
                <tr>
                    <th>测试用例</th>
                    <th>状态</th>
                    <th>耗时</th>
                    <th>消息</th>
                </tr>
"""
            for result in suite.results:
                status_class = f"status-{result.status.value}"
                html += f"""
                <tr>
                    <td>{result.test_case.name}</td>
                    <td class="{status_class}">{result.status.value.upper()}</td>
                    <td>{result.duration:.3f}s</td>
                    <td>{result.message}</td>
                </tr>
"""
            
            html += """
            </table>
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


class RegressionTestRunner:
    """回归测试运行器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        """
        初始化回归测试运行器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self.report = RegressionReport()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def run_pytest_tests(
        self,
        test_paths: List[str],
        markers: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None
    ) -> TestSuiteResult:
        """
        运行pytest测试
        
        Args:
            test_paths: 测试文件路径列表
            markers: 要运行的测试标记
            extra_args: 额外的pytest参数
            
        Returns:
            测试结果
        """
        suite_result = TestSuiteResult(name="Pytest Tests")
        suite_result.start_time = time.time()
        
        # 构建pytest命令
        args = ["-v", "--tb=short"]
        
        if markers:
            marker_expr = " or ".join(markers)
            args.extend(["-m", marker_expr])
        
        if extra_args:
            args.extend(extra_args)
        
        args.extend(test_paths)
        
        # 运行pytest
        try:
            exit_code = pytest.main(args)
            
            # 解析结果
            if exit_code == 0:
                suite_result.results.append(TestResult(
                    test_case=TestCase(
                        name="pytest_run",
                        module="pytest",
                        function="main"
                    ),
                    status=TestStatus.PASSED,
                    duration=0.0,
                    message="All tests passed"
                ))
            else:
                suite_result.results.append(TestResult(
                    test_case=TestCase(
                        name="pytest_run",
                        module="pytest",
                        function="main"
                    ),
                    status=TestStatus.FAILED,
                    duration=0.0,
                    message=f"Pytest exit code: {exit_code}"
                ))
        
        except Exception as e:
            suite_result.results.append(TestResult(
                test_case=TestCase(
                    name="pytest_run",
                    module="pytest",
                    function="main"
                ),
                status=TestStatus.ERROR,
                duration=0.0,
                message=str(e)
            ))
        
        suite_result.end_time = time.time()
        self.report.test_suites.append(suite_result)
        
        return suite_result
    
    def run_batch_tests(
        self,
        test_functions: Dict[str, Callable],
        iterations: int = 1
    ) -> TestSuiteResult:
        """
        运行批量测试
        
        Args:
            test_functions: 测试函数字典 {名称: 函数}
            iterations: 迭代次数
            
        Returns:
            测试结果
        """
        suite_result = TestSuiteResult(name="Batch Tests")
        suite_result.start_time = time.time()
        
        for name, test_func in test_functions.items():
            for i in range(iterations):
                test_case = TestCase(
                    name=f"{name}_{i+1}",
                    module="batch",
                    function=name
                )
                
                start = time.time()
                try:
                    test_func()
                    duration = time.time() - start
                    
                    suite_result.results.append(TestResult(
                        test_case=test_case,
                        status=TestStatus.PASSED,
                        duration=duration,
                        message="Test passed"
                    ))
                
                except AssertionError as e:
                    duration = time.time() - start
                    suite_result.results.append(TestResult(
                        test_case=test_case,
                        status=TestStatus.FAILED,
                        duration=duration,
                        message=str(e)
                    ))
                
                except Exception as e:
                    duration = time.time() - start
                    suite_result.results.append(TestResult(
                        test_case=test_case,
                        status=TestStatus.ERROR,
                        duration=duration,
                        message=str(e)
                    ))
        
        suite_result.end_time = time.time()
        self.report.test_suites.append(suite_result)
        
        return suite_result
    
    def compare_with_baseline(
        self,
        baseline_report_path: str
    ) -> Dict[str, Any]:
        """
        与基线报告比较
        
        Args:
            baseline_report_path: 基线报告路径
            
        Returns:
            比较结果
        """
        if not os.path.exists(baseline_report_path):
            return {
                'error': f'Baseline report not found: {baseline_report_path}'
            }
        
        with open(baseline_report_path, 'r') as f:
            baseline = json.load(f)
        
        current = self.report.to_dict()
        
        comparison = {
            'baseline_pass_rate': baseline.get('pass_rate', 0),
            'current_pass_rate': current['pass_rate'],
            'pass_rate_change': current['pass_rate'] - baseline.get('pass_rate', 0),
            'baseline_total_tests': baseline.get('total_tests', 0),
            'current_total_tests': current['total_tests'],
            'new_tests': current['total_tests'] - baseline.get('total_tests', 0),
            'regression_detected': current['pass_rate'] < baseline.get('pass_rate', 1.0),
        }
        
        return comparison
    
    def generate_report(
        self,
        report_name: Optional[str] = None,
        formats: List[str] = ["json", "html"]
    ) -> Dict[str, str]:
        """
        生成测试报告
        
        Args:
            report_name: 报告名称（默认使用时间戳）
            formats: 报告格式列表
            
        Returns:
            生成的报告文件路径
        """
        self.report.end_time = datetime.now()
        
        if report_name is None:
            report_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_files = {}
        
        if "json" in formats:
            json_path = os.path.join(self.output_dir, f"{report_name}.json")
            self.report.to_json(json_path)
            report_files['json'] = json_path
        
        if "html" in formats:
            html_path = os.path.join(self.output_dir, f"{report_name}.html")
            self.report.to_html(html_path)
            report_files['html'] = html_path
        
        return report_files
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("回归测试摘要")
        print("="*60)
        print(f"总测试数: {self.report.total_tests}")
        print(f"通过: {self.report.passed_tests}")
        print(f"失败: {self.report.failed_tests}")
        print(f"跳过: {sum(s.skipped_tests for s in self.report.test_suites)}")
        print(f"错误: {sum(s.error_tests for s in self.report.test_suites)}")
        print(f"通过率: {self.report.pass_rate*100:.2f}%")
        print(f"总耗时: {self.report.duration:.2f}秒")
        print("="*60)
        
        for suite in self.report.test_suites:
            print(f"\n测试套件: {suite.name}")
            print(f"  通过: {suite.passed_tests}/{suite.total_tests}")
            
            if suite.failed_tests > 0:
                print(f"  失败的测试:")
                for result in suite.results:
                    if result.status == TestStatus.FAILED:
                        print(f"    - {result.test_case.name}: {result.message}")


def create_default_test_suite() -> List[TestCase]:
    """创建默认测试套件"""
    return [
        TestCase(
            name="test_object_detection",
            module="unit.test_perception",
            function="TestObjectDetection",
            markers=["unit", "perception"]
        ),
        TestCase(
            name="test_trajectory_generation",
            module="unit.test_planning",
            function="TestTrajectoryGeneration",
            markers=["unit", "planning"]
        ),
        TestCase(
            name="test_lidar_sensor",
            module="unit.test_sensor",
            function="TestLidarSensor",
            markers=["unit", "sensor"]
        ),
        TestCase(
            name="test_perception_simulation",
            module="simulation.test_simulation",
            function="TestPerceptionSimulation",
            markers=["simulation", "perception"]
        ),
        TestCase(
            name="test_planning_simulation",
            module="simulation.test_simulation",
            function="TestPlanningSimulation",
            markers=["simulation", "planning"]
        ),
    ]
