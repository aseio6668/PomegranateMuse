"""
Test Runner for MyndraComposer
Orchestrates and executes all test suites with comprehensive reporting
"""

import unittest
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import traceback
import concurrent.futures
from io import StringIO

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test result information"""
    test_name: str
    test_class: str
    test_method: str
    status: TestStatus
    duration: float
    message: str = ""
    traceback: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class TestSuiteResult:
    """Test suite execution results"""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    test_results: List[TestResult] = field(default_factory=list)
    setup_errors: List[str] = field(default_factory=list)
    teardown_errors: List[str] = field(default_factory=list)

class TestSuite:
    """Base test suite class"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tests = []
        self.setup_methods = []
        self.teardown_methods = []
        self.logger = logging.getLogger(f"test.{name}")
    
    def add_test(self, test_class: unittest.TestCase):
        """Add test class to suite"""
        self.tests.append(test_class)
    
    def add_setup(self, setup_method):
        """Add setup method"""
        self.setup_methods.append(setup_method)
    
    def add_teardown(self, teardown_method):
        """Add teardown method"""
        self.teardown_methods.append(teardown_method)
    
    def run_setup(self) -> List[str]:
        """Run setup methods"""
        errors = []
        for setup_method in self.setup_methods:
            try:
                setup_method()
            except Exception as e:
                error_msg = f"Setup error in {setup_method.__name__}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        return errors
    
    def run_teardown(self) -> List[str]:
        """Run teardown methods"""
        errors = []
        for teardown_method in self.teardown_methods:
            try:
                teardown_method()
            except Exception as e:
                error_msg = f"Teardown error in {teardown_method.__name__}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        return errors

class CustomTestResult(unittest.TestResult):
    """Custom test result collector"""
    
    def __init__(self):
        super().__init__()
        self.test_results = []
        self.current_test_start = None
    
    def startTest(self, test):
        super().startTest(test)
        self.current_test_start = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.current_test_start
        result = TestResult(
            test_name=test._testMethodName,
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status=TestStatus.PASSED,
            duration=duration,
            message="Test passed"
        )
        self.test_results.append(result)
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.current_test_start
        result = TestResult(
            test_name=test._testMethodName,
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status=TestStatus.ERROR,
            duration=duration,
            message=str(err[1]),
            traceback=''.join(traceback.format_exception(*err))
        )
        self.test_results.append(result)
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.current_test_start
        result = TestResult(
            test_name=test._testMethodName,
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status=TestStatus.FAILED,
            duration=duration,
            message=str(err[1]),
            traceback=''.join(traceback.format_exception(*err))
        )
        self.test_results.append(result)
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.current_test_start
        result = TestResult(
            test_name=test._testMethodName,
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status=TestStatus.SKIPPED,
            duration=duration,
            message=reason
        )
        self.test_results.append(result)

class TestRunner:
    """Main test runner orchestrating all test execution"""
    
    def __init__(self, config_manager=None, output_dir: Optional[Path] = None):
        self.config_manager = config_manager
        self.output_dir = output_dir or Path.cwd() / "test_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.test_suites = {}
        self.results = {}
        
        # Test configuration
        self.parallel_execution = True
        self.max_workers = 4
        self.timeout_per_test = 300  # 5 minutes
        self.continue_on_failure = True
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup test logging"""
        log_file = self.output_dir / "test_execution.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def register_test_suite(self, suite: TestSuite):
        """Register a test suite"""
        self.test_suites[suite.name] = suite
        self.logger.info(f"Registered test suite: {suite.name}")
    
    def run_suite(self, suite_name: str, parallel: bool = None) -> TestSuiteResult:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        parallel = parallel if parallel is not None else self.parallel_execution
        
        self.logger.info(f"Starting test suite: {suite_name}")
        start_time = time.time()
        
        # Run setup
        setup_errors = suite.run_setup()
        
        # Create unittest suite
        unittest_suite = unittest.TestSuite()
        for test_class in suite.tests:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            unittest_suite.addTests(tests)
        
        # Run tests
        if parallel and len(suite.tests) > 1:
            test_results = self._run_parallel_tests(unittest_suite)
        else:
            test_results = self._run_sequential_tests(unittest_suite)
        
        # Run teardown
        teardown_errors = suite.run_teardown()
        
        # Calculate results
        duration = time.time() - start_time
        total_tests = len(test_results)
        passed = len([r for r in test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in test_results if r.status == TestStatus.FAILED])
        skipped = len([r for r in test_results if r.status == TestStatus.SKIPPED])
        errors = len([r for r in test_results if r.status == TestStatus.ERROR])
        
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            test_results=test_results,
            setup_errors=setup_errors,
            teardown_errors=teardown_errors
        )
        
        self.results[suite_name] = suite_result
        self.logger.info(f"Completed test suite: {suite_name} - {passed}/{total_tests} passed")
        
        return suite_result
    
    def _run_sequential_tests(self, unittest_suite) -> List[TestResult]:
        """Run tests sequentially"""
        custom_result = CustomTestResult()
        unittest_suite.run(custom_result)
        return custom_result.test_results
    
    def _run_parallel_tests(self, unittest_suite) -> List[TestResult]:
        """Run tests in parallel"""
        # For simplicity, fall back to sequential for now
        # Real implementation would use multiprocessing or threading
        return self._run_sequential_tests(unittest_suite)
    
    def run_all_suites(self, filter_tags: Optional[List[str]] = None) -> Dict[str, TestSuiteResult]:
        """Run all registered test suites"""
        self.logger.info("Starting execution of all test suites")
        start_time = time.time()
        
        all_results = {}
        
        for suite_name in self.test_suites:
            try:
                result = self.run_suite(suite_name)
                all_results[suite_name] = result
                
                # Stop on failure if configured
                if not self.continue_on_failure and (result.failed > 0 or result.errors > 0):
                    self.logger.warning(f"Stopping execution due to failures in {suite_name}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to run test suite {suite_name}: {e}")
                if not self.continue_on_failure:
                    break
        
        total_duration = time.time() - start_time
        self.logger.info(f"Completed all test suites in {total_duration:.2f} seconds")
        
        # Generate reports
        self._generate_reports(all_results, total_duration)
        
        return all_results
    
    def _generate_reports(self, results: Dict[str, TestSuiteResult], total_duration: float):
        """Generate test reports"""
        # Generate JSON report
        self._generate_json_report(results, total_duration)
        
        # Generate HTML report
        self._generate_html_report(results, total_duration)
        
        # Generate console summary
        self._print_summary(results, total_duration)
    
    def _generate_json_report(self, results: Dict[str, TestSuiteResult], total_duration: float):
        """Generate JSON test report"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "summary": self._calculate_summary(results),
            "suites": {}
        }
        
        for suite_name, result in results.items():
            report_data["suites"][suite_name] = {
                "suite_name": result.suite_name,
                "total_tests": result.total_tests,
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "errors": result.errors,
                "duration": result.duration,
                "setup_errors": result.setup_errors,
                "teardown_errors": result.teardown_errors,
                "tests": [
                    {
                        "test_name": test.test_name,
                        "test_class": test.test_class,
                        "test_method": test.test_method,
                        "status": test.status.value,
                        "duration": test.duration,
                        "message": test.message,
                        "traceback": test.traceback,
                        "timestamp": test.timestamp.isoformat()
                    }
                    for test in result.test_results
                ]
            }
        
        json_file = self.output_dir / "test_results.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"JSON report generated: {json_file}")
    
    def _generate_html_report(self, results: Dict[str, TestSuiteResult], total_duration: float):
        """Generate HTML test report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MyndraComposer Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .suite { margin-bottom: 30px; }
        .suite-header { background: #e9ecef; padding: 10px; border-radius: 5px; }
        .test-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .test-table th, .test-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .test-table th { background-color: #f2f2f2; }
        .status-passed { color: green; font-weight: bold; }
        .status-failed { color: red; font-weight: bold; }
        .status-error { color: orange; font-weight: bold; }
        .status-skipped { color: blue; font-weight: bold; }
        .traceback { font-family: monospace; font-size: 12px; background: #f8f8f8; padding: 10px; }
    </style>
</head>
<body>
    <h1>MyndraComposer Test Results</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Duration:</strong> {total_duration:.2f} seconds</p>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Passed:</strong> <span class="status-passed">{total_passed}</span></p>
        <p><strong>Failed:</strong> <span class="status-failed">{total_failed}</span></p>
        <p><strong>Errors:</strong> <span class="status-error">{total_errors}</span></p>
        <p><strong>Skipped:</strong> <span class="status-skipped">{total_skipped}</span></p>
    </div>
    {suite_reports}
</body>
</html>
        """
        
        summary = self._calculate_summary(results)
        suite_reports = []
        
        for suite_name, result in results.items():
            test_rows = []
            for test in result.test_results:
                status_class = f"status-{test.status.value}"
                test_row = f"""
                <tr>
                    <td>{test.test_class}</td>
                    <td>{test.test_method}</td>
                    <td class="{status_class}">{test.status.value.upper()}</td>
                    <td>{test.duration:.3f}s</td>
                    <td>{test.message}</td>
                </tr>
                """
                test_rows.append(test_row)
            
            suite_report = f"""
            <div class="suite">
                <div class="suite-header">
                    <h3>{suite_name}</h3>
                    <p>Duration: {result.duration:.2f}s | Tests: {result.total_tests} | 
                       Passed: {result.passed} | Failed: {result.failed} | 
                       Errors: {result.errors} | Skipped: {result.skipped}</p>
                </div>
                <table class="test-table">
                    <thead>
                        <tr>
                            <th>Test Class</th>
                            <th>Test Method</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Message</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(test_rows)}
                    </tbody>
                </table>
            </div>
            """
            suite_reports.append(suite_report)
        
        html_content = html_template.format(
            total_duration=total_duration,
            total_tests=summary['total_tests'],
            total_passed=summary['total_passed'],
            total_failed=summary['total_failed'],
            total_errors=summary['total_errors'],
            total_skipped=summary['total_skipped'],
            suite_reports=''.join(suite_reports)
        )
        
        html_file = self.output_dir / "test_results.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {html_file}")
    
    def _print_summary(self, results: Dict[str, TestSuiteResult], total_duration: float):
        """Print test summary to console"""
        summary = self._calculate_summary(results)
        
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Skipped: {summary['total_skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print()
        
        for suite_name, result in results.items():
            status = "PASS" if result.failed == 0 and result.errors == 0 else "FAIL"
            print(f"{suite_name}: {status} ({result.passed}/{result.total_tests})")
        
        print("="*80)
        
        # Exit with error code if tests failed
        if summary['total_failed'] > 0 or summary['total_errors'] > 0:
            sys.exit(1)
    
    def _calculate_summary(self, results: Dict[str, TestSuiteResult]) -> Dict[str, Any]:
        """Calculate overall test summary"""
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_skipped = sum(r.skipped for r in results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'success_rate': success_rate
        }

# Convenience functions
def run_all_tests(config_manager=None, output_dir: Optional[Path] = None) -> Dict[str, TestSuiteResult]:
    """Run all available test suites"""
    runner = TestRunner(config_manager, output_dir)
    
    # Import and register all test suites
    from .unit_tests import register_unit_tests
    from .integration_tests import register_integration_tests
    from .performance_tests import register_performance_tests
    
    register_unit_tests(runner)
    register_integration_tests(runner)
    register_performance_tests(runner)
    
    return runner.run_all_suites()

def run_test_suite(suite_name: str, config_manager=None, output_dir: Optional[Path] = None) -> TestSuiteResult:
    """Run a specific test suite"""
    runner = TestRunner(config_manager, output_dir)
    
    # Import and register all test suites
    from .unit_tests import register_unit_tests
    from .integration_tests import register_integration_tests
    from .performance_tests import register_performance_tests
    
    register_unit_tests(runner)
    register_integration_tests(runner)
    register_performance_tests(runner)
    
    return runner.run_suite(suite_name)