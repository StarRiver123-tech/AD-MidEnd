"""
回归测试用例
测试回归测试框架的功能
"""

import pytest
import os
import json
import tempfile
from datetime import datetime

from regression.regression_framework import (
    TestCase, TestResult, TestSuiteResult, RegressionReport,
    RegressionTestRunner, TestStatus, create_default_test_suite
)


@pytest.mark.regression
class TestTestCase:
    """测试用例单元测试"""
    
    def test_test_case_creation(self):
        """测试测试用例创建"""
        test_case = TestCase(
            name="test_example",
            module="test_module",
            function="test_function",
            markers=["unit", "fast"],
            parameters={"param1": 1, "param2": 2}
        )
        
        assert test_case.name == "test_example"
        assert test_case.module == "test_module"
        assert test_case.function == "test_function"
        assert "unit" in test_case.markers
        assert test_case.parameters["param1"] == 1


@pytest.mark.regression
class TestTestResult:
    """测试结果单元测试"""
    
    def test_test_result_creation(self):
        """测试测试结果创建"""
        test_case = TestCase(
            name="test_example",
            module="test_module",
            function="test_function"
        )
        
        result = TestResult(
            test_case=test_case,
            status=TestStatus.PASSED,
            duration=0.5,
            message="Test passed successfully"
        )
        
        assert result.test_case == test_case
        assert result.status == TestStatus.PASSED
        assert result.duration == 0.5
        assert result.message == "Test passed successfully"
    
    def test_test_result_to_dict(self):
        """测试测试结果转换为字典"""
        test_case = TestCase(
            name="test_example",
            module="test_module",
            function="test_function"
        )
        
        result = TestResult(
            test_case=test_case,
            status=TestStatus.FAILED,
            duration=1.0,
            message="Assertion error"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['status'] == 'failed'
        assert result_dict['duration'] == 1.0
        assert result_dict['test_case']['name'] == 'test_example'


@pytest.mark.regression
class TestTestSuiteResult:
    """测试套件结果单元测试"""
    
    def test_suite_result_creation(self):
        """测试套件结果创建"""
        suite = TestSuiteResult(name="Test Suite 1")
        
        assert suite.name == "Test Suite 1"
        assert suite.total_tests == 0
        assert suite.passed_tests == 0
        assert suite.failed_tests == 0
    
    def test_suite_result_statistics(self):
        """测试套件结果统计"""
        suite = TestSuiteResult(name="Test Suite 1")
        
        # 添加测试结果
        test_cases = [
            TestCase(name=f"test_{i}", module="test", function="test")
            for i in range(5)
        ]
        
        suite.results = [
            TestResult(test_case=test_cases[0], status=TestStatus.PASSED, duration=0.1),
            TestResult(test_case=test_cases[1], status=TestStatus.PASSED, duration=0.2),
            TestResult(test_case=test_cases[2], status=TestStatus.FAILED, duration=0.3),
            TestResult(test_case=test_cases[3], status=TestStatus.SKIPPED, duration=0.0),
            TestResult(test_case=test_cases[4], status=TestStatus.ERROR, duration=0.4),
        ]
        
        assert suite.total_tests == 5
        assert suite.passed_tests == 2
        assert suite.failed_tests == 1
        assert suite.skipped_tests == 1
        assert suite.error_tests == 1
        assert suite.pass_rate == 0.4
    
    def test_suite_result_to_dict(self):
        """测试套件结果转换为字典"""
        suite = TestSuiteResult(name="Test Suite 1")
        suite.start_time = 0.0
        suite.end_time = 10.0
        
        test_case = TestCase(name="test_1", module="test", function="test")
        suite.results = [
            TestResult(test_case=test_case, status=TestStatus.PASSED, duration=0.5)
        ]
        
        suite_dict = suite.to_dict()
        
        assert suite_dict['name'] == "Test Suite 1"
        assert suite_dict['total_tests'] == 1
        assert suite_dict['passed_tests'] == 1
        assert suite_dict['duration'] == 10.0


@pytest.mark.regression
class TestRegressionReport:
    """回归测试报告单元测试"""
    
    def test_report_creation(self):
        """测试报告创建"""
        report = RegressionReport()
        
        assert report.total_tests == 0
        assert report.passed_tests == 0
        assert report.pass_rate == 0.0
        assert report.start_time is not None
    
    def test_report_statistics(self):
        """测试报告统计"""
        report = RegressionReport()
        
        # 创建测试套件
        suite1 = TestSuiteResult(name="Suite 1")
        suite1.results = [
            TestResult(
                test_case=TestCase(name="test1", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.1
            ),
            TestResult(
                test_case=TestCase(name="test2", module="test", function="test"),
                status=TestStatus.FAILED,
                duration=0.2
            ),
        ]
        
        suite2 = TestSuiteResult(name="Suite 2")
        suite2.results = [
            TestResult(
                test_case=TestCase(name="test3", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.3
            ),
        ]
        
        report.test_suites = [suite1, suite2]
        
        assert report.total_tests == 3
        assert report.passed_tests == 2
        assert report.failed_tests == 1
        assert report.pass_rate == 2/3
    
    def test_report_to_json(self, tmp_path):
        """测试报告导出为JSON"""
        report = RegressionReport()
        report.metadata = {'version': '1.0'}
        
        suite = TestSuiteResult(name="Test Suite")
        suite.results = [
            TestResult(
                test_case=TestCase(name="test1", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.5
            )
        ]
        report.test_suites = [suite]
        report.end_time = datetime.now()
        
        json_path = tmp_path / "report.json"
        report.to_json(str(json_path))
        
        assert json_path.exists()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['total_tests'] == 1
        assert data['passed_tests'] == 1
        assert data['metadata']['version'] == '1.0'
    
    def test_report_to_html(self, tmp_path):
        """测试报告导出为HTML"""
        report = RegressionReport()
        
        suite = TestSuiteResult(name="Test Suite")
        suite.results = [
            TestResult(
                test_case=TestCase(name="test1", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.5
            )
        ]
        report.test_suites = [suite]
        report.end_time = datetime.now()
        
        html_path = tmp_path / "report.html"
        report.to_html(str(html_path))
        
        assert html_path.exists()
        
        with open(html_path, 'r') as f:
            content = f.read()
        
        assert "回归测试报告" in content
        assert "test1" in content
        assert "PASSED" in content


@pytest.mark.regression
class TestRegressionTestRunner:
    """回归测试运行器单元测试"""
    
    def test_runner_creation(self, tmp_path):
        """测试运行器创建"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        assert runner.output_dir == str(tmp_path)
        assert os.path.exists(tmp_path)
    
    def test_run_batch_tests(self, tmp_path):
        """测试批量测试运行"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        def passing_test():
            assert True
        
        def failing_test():
            assert False
        
        test_functions = {
            'passing_test': passing_test,
            'failing_test': failing_test,
        }
        
        result = runner.run_batch_tests(test_functions)
        
        assert result.name == "Batch Tests"
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1
    
    def test_run_batch_tests_with_iterations(self, tmp_path):
        """测试带迭代的批量测试"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        counter = [0]
        
        def increment_test():
            counter[0] += 1
        
        test_functions = {
            'increment_test': increment_test,
        }
        
        result = runner.run_batch_tests(test_functions, iterations=5)
        
        assert result.total_tests == 5
        assert counter[0] == 5
    
    def test_generate_report(self, tmp_path):
        """测试报告生成"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        # 添加一些测试结果
        suite = TestSuiteResult(name="Test Suite")
        suite.results = [
            TestResult(
                test_case=TestCase(name="test1", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.5
            )
        ]
        runner.report.test_suites = [suite]
        
        report_files = runner.generate_report(
            report_name="test_report",
            formats=["json", "html"]
        )
        
        assert 'json' in report_files
        assert 'html' in report_files
        assert os.path.exists(report_files['json'])
        assert os.path.exists(report_files['html'])
    
    def test_compare_with_baseline(self, tmp_path):
        """测试与基线比较"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        # 创建基线报告
        baseline_report = {
            'total_tests': 10,
            'passed_tests': 9,
            'pass_rate': 0.9
        }
        
        baseline_path = tmp_path / "baseline.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_report, f)
        
        # 添加当前测试结果
        suite = TestSuiteResult(name="Test Suite")
        suite.results = [
            TestResult(
                test_case=TestCase(name="test", module="test", function="test"),
                status=TestStatus.PASSED,
                duration=0.5
            )
        ]
        runner.report.test_suites = [suite]
        runner.report.end_time = datetime.now()
        
        comparison = runner.compare_with_baseline(str(baseline_path))
        
        assert 'baseline_pass_rate' in comparison
        assert 'current_pass_rate' in comparison
        assert 'pass_rate_change' in comparison
        assert comparison['baseline_pass_rate'] == 0.9


@pytest.mark.regression
class TestDefaultTestSuite:
    """默认测试套件单元测试"""
    
    def test_default_suite_creation(self):
        """测试默认套件创建"""
        suite = create_default_test_suite()
        
        assert len(suite) > 0
        
        for test_case in suite:
            assert test_case.name is not None
            assert test_case.module is not None
            assert test_case.function is not None


@pytest.mark.regression
@pytest.mark.slow
class TestRegressionIntegration:
    """回归测试集成测试"""
    
    def test_full_regression_workflow(self, tmp_path):
        """测试完整回归工作流"""
        runner = RegressionTestRunner(output_dir=str(tmp_path))
        
        # 定义测试函数
        def test_1():
            assert 1 + 1 == 2
        
        def test_2():
            assert "hello" == "hello"
        
        def test_3():
            # 这个测试会失败
            assert 1 == 2
        
        test_functions = {
            'test_1': test_1,
            'test_2': test_2,
            'test_3': test_3,
        }
        
        # 运行批量测试
        runner.run_batch_tests(test_functions, iterations=2)
        
        # 生成报告
        report_files = runner.generate_report(report_name="integration_test")
        
        # 验证结果
        assert runner.report.total_tests == 6
        assert runner.report.passed_tests == 4
        assert runner.report.failed_tests == 2
        
        # 验证报告文件
        assert os.path.exists(report_files['json'])
        assert os.path.exists(report_files['html'])
        
        # 打印摘要
        runner.print_summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
