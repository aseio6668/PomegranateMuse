"""
Test Orchestrator for Universal Testing System
Manages comprehensive testing workflows, coverage analysis, and performance profiling
"""

import asyncio
import json
import logging
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .universal_builder import (
    BuildResult, BuildStatus, BuildEnvironment, 
    TestResult, TestType, TestSuite
)
from .language_builders import get_builder_for_language

class TestExecutionMode(Enum):
    """Test execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"

class CoverageType(Enum):
    """Code coverage types"""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"

@dataclass
class TestConfiguration:
    """Comprehensive test configuration"""
    languages: List[str]
    test_types: List[TestType]
    execution_mode: TestExecutionMode = TestExecutionMode.PARALLEL
    timeout: int = 1800  # 30 minutes
    retry_count: int = 3
    coverage_enabled: bool = True
    coverage_types: List[CoverageType] = None
    performance_profiling: bool = True
    generate_reports: bool = True
    environment_matrix: List[BuildEnvironment] = None
    
    def __post_init__(self):
        if self.coverage_types is None:
            self.coverage_types = [CoverageType.LINE, CoverageType.BRANCH]
        if self.environment_matrix is None:
            self.environment_matrix = [BuildEnvironment()]

@dataclass
class CoverageReport:
    """Code coverage report"""
    language: str
    total_lines: int
    covered_lines: int
    line_coverage: float
    total_branches: int = 0
    covered_branches: int = 0
    branch_coverage: float = 0.0
    total_functions: int = 0
    covered_functions: int = 0
    function_coverage: float = 0.0
    files: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.files is None:
            self.files = []

@dataclass
class PerformanceMetrics:
    """Performance profiling metrics"""
    test_name: str
    execution_time: float
    memory_usage: int = 0  # bytes
    cpu_usage: float = 0.0  # percentage
    disk_io: int = 0  # bytes
    network_io: int = 0  # bytes
    heap_size: int = 0  # bytes
    gc_collections: int = 0
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

@dataclass
class TestExecutionReport:
    """Comprehensive test execution report"""
    configuration: TestConfiguration
    start_time: datetime
    end_time: datetime
    total_duration: float
    build_results: List[BuildResult]
    coverage_reports: List[CoverageReport] = None
    performance_reports: List[PerformanceMetrics] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.coverage_reports is None:
            self.coverage_reports = []
        if self.performance_reports is None:
            self.performance_reports = []
        if self.summary is None:
            self.summary = {}

class CoverageAnalyzer:
    """Analyzes code coverage across different languages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Coverage tool configurations for each language
        self.coverage_tools = {
            "pomegranate": {
                "tool": "pomeg",
                "command": ["pomeg", "test", "--coverage"],
                "report_format": "json",
                "report_file": "coverage.json"
            },
            "rust": {
                "tool": "cargo-tarpaulin",
                "command": ["cargo", "tarpaulin", "--out", "Json"],
                "report_format": "json",
                "report_file": "tarpaulin-report.json"
            },
            "go": {
                "tool": "go-cover",
                "command": ["go", "test", "-coverprofile=coverage.out", "./..."],
                "report_format": "go-cover",
                "report_file": "coverage.out"
            },
            "typescript": {
                "tool": "nyc",
                "command": ["nyc", "--reporter=json", "npm", "test"],
                "report_format": "json",
                "report_file": "coverage/coverage-final.json"
            },
            "python": {
                "tool": "coverage.py",
                "command": ["coverage", "run", "-m", "pytest"],
                "report_format": "json",
                "report_file": "coverage.json"
            }
        }
    
    async def analyze_coverage(self, project_path: Path, language: str, 
                             build_result: BuildResult) -> Optional[CoverageReport]:
        """Analyze code coverage for a specific language"""
        if language not in self.coverage_tools:
            self.logger.warning(f"Coverage analysis not supported for {language}")
            return None
            
        config = self.coverage_tools[language]
        
        try:
            # Run coverage analysis
            process = await asyncio.create_subprocess_exec(
                *config["command"],
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Coverage analysis failed for {language}: {stderr.decode()}")
                return None
            
            # Parse coverage report
            report_file = project_path / config["report_file"]
            if not report_file.exists():
                self.logger.error(f"Coverage report file not found: {report_file}")
                return None
                
            return await self._parse_coverage_report(report_file, language, config["report_format"])
            
        except Exception as e:
            self.logger.error(f"Coverage analysis error for {language}: {e}")
            return None
    
    async def _parse_coverage_report(self, report_file: Path, language: str, 
                                   format_type: str) -> CoverageReport:
        """Parse coverage report based on format"""
        if format_type == "json":
            return await self._parse_json_coverage(report_file, language)
        elif format_type == "go-cover":
            return await self._parse_go_coverage(report_file, language)
        else:
            raise ValueError(f"Unsupported coverage format: {format_type}")
    
    async def _parse_json_coverage(self, report_file: Path, language: str) -> CoverageReport:
        """Parse JSON coverage report (common format)"""
        with open(report_file) as f:
            data = json.load(f)
        
        # Handle different JSON structures based on language
        if language == "rust":
            # Tarpaulin format
            total_lines = data.get("covered", 0) + data.get("uncovered", 0)
            covered_lines = data.get("covered", 0)
            
        elif language == "typescript":
            # NYC format
            total = data.get("total", {})
            total_lines = total.get("lines", {}).get("total", 0)
            covered_lines = total.get("lines", {}).get("covered", 0)
            
        elif language == "python":
            # Coverage.py format
            totals = data.get("totals", {})
            total_lines = totals.get("num_statements", 0)
            covered_lines = totals.get("covered_lines", 0)
            
        else:
            # Generic format
            total_lines = data.get("total_lines", 0)
            covered_lines = data.get("covered_lines", 0)
        
        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        
        return CoverageReport(
            language=language,
            total_lines=total_lines,
            covered_lines=covered_lines,
            line_coverage=line_coverage
        )
    
    async def _parse_go_coverage(self, report_file: Path, language: str) -> CoverageReport:
        """Parse Go coverage report format"""
        total_statements = 0
        covered_statements = 0
        
        with open(report_file) as f:
            for line in f:
                if line.startswith("mode:"):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 3:
                    statements = int(parts[1])
                    covered = int(parts[2])
                    
                    total_statements += statements
                    if covered > 0:
                        covered_statements += statements
        
        line_coverage = (covered_statements / total_statements * 100) if total_statements > 0 else 0.0
        
        return CoverageReport(
            language=language,
            total_lines=total_statements,
            covered_lines=covered_statements,
            line_coverage=line_coverage
        )

class PerformanceProfiler:
    """Profiles performance metrics during test execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def profile_test_execution(self, test_name: str, command: List[str], 
                                   cwd: Path, timeout: int = 300) -> PerformanceMetrics:
        """Profile a test execution with performance metrics"""
        import psutil
        import time
        
        start_time = time.time()
        
        try:
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor performance
            initial_memory = psutil.virtual_memory().used
            max_memory = initial_memory
            cpu_samples = []
            
            # Monitor process periodically
            monitor_interval = 0.1  # 100ms
            start_monitor = time.time()
            
            while process.returncode is None:
                try:
                    # Sample CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    cpu_samples.append(cpu_percent)
                    
                    # Sample memory usage
                    current_memory = psutil.virtual_memory().used
                    max_memory = max(max_memory, current_memory)
                    
                    # Check if process is still running
                    try:
                        await asyncio.wait_for(process.wait(), timeout=monitor_interval)
                        break
                    except asyncio.TimeoutError:
                        continue
                        
                except psutil.Error:
                    break
                    
                # Check overall timeout
                if time.time() - start_monitor > timeout:
                    process.kill()
                    break
            
            execution_time = time.time() - start_time
            memory_usage = max_memory - initial_memory
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            
            return PerformanceMetrics(
                test_name=test_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=avg_cpu
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Performance profiling error: {e}")
            
            return PerformanceMetrics(
                test_name=test_name,
                execution_time=execution_time,
                custom_metrics={"error": str(e)}
            )

class TestReportGenerator:
    """Generates comprehensive test reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def generate_comprehensive_report(self, execution_report: TestExecutionReport) -> Dict[str, str]:
        """Generate comprehensive test report in multiple formats"""
        report_files = {}
        
        # Generate JSON report
        json_file = await self._generate_json_report(execution_report)
        report_files["json"] = str(json_file)
        
        # Generate HTML report
        html_file = await self._generate_html_report(execution_report)
        report_files["html"] = str(html_file)
        
        # Generate JUnit XML report
        junit_file = await self._generate_junit_report(execution_report)
        report_files["junit"] = str(junit_file)
        
        # Generate coverage report
        if execution_report.coverage_reports:
            coverage_file = await self._generate_coverage_report(execution_report)
            report_files["coverage"] = str(coverage_file)
        
        # Generate performance report
        if execution_report.performance_reports:
            perf_file = await self._generate_performance_report(execution_report)
            report_files["performance"] = str(perf_file)
        
        return report_files
    
    async def _generate_json_report(self, execution_report: TestExecutionReport) -> Path:
        """Generate JSON test report"""
        report_file = self.output_dir / "test_report.json"
        
        # Convert to serializable format
        report_data = {
            "configuration": asdict(execution_report.configuration),
            "start_time": execution_report.start_time.isoformat(),
            "end_time": execution_report.end_time.isoformat(),
            "total_duration": execution_report.total_duration,
            "build_results": [asdict(result) for result in execution_report.build_results],
            "coverage_reports": [asdict(report) for report in execution_report.coverage_reports],
            "performance_reports": [asdict(report) for report in execution_report.performance_reports],
            "summary": execution_report.summary
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_file
    
    async def _generate_html_report(self, execution_report: TestExecutionReport) -> Path:
        """Generate HTML test report"""
        report_file = self.output_dir / "test_report.html"
        
        # Calculate summary statistics
        total_builds = len(execution_report.build_results)
        successful_builds = len([r for r in execution_report.build_results if r.status == BuildStatus.SUCCESS])
        
        total_tests = sum(len(r.test_results) for r in execution_report.build_results)
        passed_tests = sum(len([t for t in r.test_results if t.status == BuildStatus.SUCCESS]) 
                          for r in execution_report.build_results)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PomegranteMuse Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; text-align: center; }}
        .success {{ background-color: #d4edda; }}
        .failure {{ background-color: #f8d7da; }}
        .warning {{ background-color: #fff3cd; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍎 PomegranteMuse Universal Test Report</h1>
        <p>Generated: {execution_report.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Duration: {execution_report.total_duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <div class="metric success">
            <h3>{successful_builds}</h3>
            <p>Successful Builds</p>
        </div>
        <div class="metric failure">
            <h3>{total_builds - successful_builds}</h3>
            <p>Failed Builds</p>
        </div>
        <div class="metric success">
            <h3>{passed_tests}</h3>
            <p>Passed Tests</p>
        </div>
        <div class="metric failure">
            <h3>{total_tests - passed_tests}</h3>
            <p>Failed Tests</p>
        </div>
    </div>
    
    <h2>Build Results</h2>
    <table>
        <tr>
            <th>Language</th>
            <th>Platform</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Tests</th>
        </tr>
"""
        
        for result in execution_report.build_results:
            status_class = "success" if result.status == BuildStatus.SUCCESS else "failure"
            test_count = len(result.test_results)
            passed_count = len([t for t in result.test_results if t.status == BuildStatus.SUCCESS])
            
            html_content += f"""
        <tr class="{status_class}">
            <td>{result.language}</td>
            <td>{result.platform}</td>
            <td>{result.status.value}</td>
            <td>{result.duration:.2f}s</td>
            <td>{passed_count}/{test_count}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        return report_file
    
    async def _generate_junit_report(self, execution_report: TestExecutionReport) -> Path:
        """Generate JUnit XML test report"""
        report_file = self.output_dir / "junit_report.xml"
        
        root = ET.Element("testsuites")
        root.set("name", "PomegranteMuse Universal Tests")
        root.set("time", str(execution_report.total_duration))
        
        for build_result in execution_report.build_results:
            testsuite = ET.SubElement(root, "testsuite")
            testsuite.set("name", f"{build_result.language}_{build_result.platform}")
            testsuite.set("tests", str(len(build_result.test_results)))
            testsuite.set("time", str(build_result.duration))
            
            failures = len([t for t in build_result.test_results if t.status == BuildStatus.FAILURE])
            testsuite.set("failures", str(failures))
            
            for test_result in build_result.test_results:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", test_result.name)
                testcase.set("time", str(test_result.duration))
                
                if test_result.status == BuildStatus.FAILURE:
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", test_result.error_message)
                    failure.text = test_result.output
        
        tree = ET.ElementTree(root)
        tree.write(report_file, encoding="utf-8", xml_declaration=True)
        
        return report_file
    
    async def _generate_coverage_report(self, execution_report: TestExecutionReport) -> Path:
        """Generate coverage report"""
        report_file = self.output_dir / "coverage_report.json"
        
        coverage_data = {
            "summary": {},
            "by_language": {}
        }
        
        total_lines = 0
        total_covered = 0
        
        for coverage in execution_report.coverage_reports:
            coverage_data["by_language"][coverage.language] = asdict(coverage)
            total_lines += coverage.total_lines
            total_covered += coverage.covered_lines
        
        overall_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0.0
        
        coverage_data["summary"] = {
            "total_lines": total_lines,
            "covered_lines": total_covered,
            "overall_coverage": overall_coverage
        }
        
        with open(report_file, 'w') as f:
            json.dump(coverage_data, f, indent=2)
        
        return report_file
    
    async def _generate_performance_report(self, execution_report: TestExecutionReport) -> Path:
        """Generate performance report"""
        report_file = self.output_dir / "performance_report.json"
        
        perf_data = {
            "summary": {
                "total_execution_time": sum(p.execution_time for p in execution_report.performance_reports),
                "average_execution_time": sum(p.execution_time for p in execution_report.performance_reports) / len(execution_report.performance_reports),
                "total_memory_usage": sum(p.memory_usage for p in execution_report.performance_reports),
                "average_cpu_usage": sum(p.cpu_usage for p in execution_report.performance_reports) / len(execution_report.performance_reports)
            },
            "by_test": [asdict(perf) for perf in execution_report.performance_reports]
        }
        
        with open(report_file, 'w') as f:
            json.dump(perf_data, f, indent=2)
        
        return report_file

class TestOrchestrator:
    """Orchestrates comprehensive testing workflows"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/universal_test")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_analyzer = CoverageAnalyzer()
        self.performance_profiler = PerformanceProfiler()
        self.report_generator = TestReportGenerator(self.cache_dir / "reports")
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_tests(self, project_path: Path, 
                                    config: TestConfiguration) -> TestExecutionReport:
        """Run comprehensive tests based on configuration"""
        start_time = datetime.now()
        
        execution_report = TestExecutionReport(
            configuration=config,
            start_time=start_time,
            end_time=start_time,
            total_duration=0.0,
            build_results=[]
        )
        
        try:
            # Run tests for each language and environment combination
            tasks = []
            
            for language in config.languages:
                for environment in config.environment_matrix:
                    if config.execution_mode == TestExecutionMode.PARALLEL:
                        task = self._run_language_tests(project_path, language, environment, config)
                        tasks.append(task)
                    else:
                        # Sequential execution
                        result = await self._run_language_tests(project_path, language, environment, config)
                        execution_report.build_results.append(result)
            
            # Execute parallel tasks if any
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Test execution error: {result}")
                    else:
                        execution_report.build_results.append(result)
            
            # Generate coverage reports if enabled
            if config.coverage_enabled:
                execution_report.coverage_reports = await self._generate_coverage_reports(
                    project_path, execution_report.build_results
                )
            
            # Generate performance reports if enabled
            if config.performance_profiling:
                execution_report.performance_reports = await self._generate_performance_reports(
                    execution_report.build_results
                )
            
            # Generate summary
            execution_report.summary = self._generate_summary(execution_report)
            
        except Exception as e:
            self.logger.error(f"Test orchestration error: {e}")
            
        finally:
            execution_report.end_time = datetime.now()
            execution_report.total_duration = (execution_report.end_time - execution_report.start_time).total_seconds()
            
            # Generate reports if configured
            if config.generate_reports:
                try:
                    await self.report_generator.generate_comprehensive_report(execution_report)
                except Exception as e:
                    self.logger.error(f"Report generation error: {e}")
        
        return execution_report
    
    async def _run_language_tests(self, project_path: Path, language: str, 
                                 environment: BuildEnvironment, config: TestConfiguration) -> BuildResult:
        """Run tests for a specific language and environment"""
        from .universal_builder import UniversalBuilder
        
        builder = UniversalBuilder()
        
        # Create test suites based on configuration
        test_suites = []
        for test_type in config.test_types:
            test_suite = TestSuite(
                name=f"{language}_{test_type.value}_tests",
                language=language,
                test_files=[],
                test_type=test_type,
                timeout=config.timeout,
                environment=environment
            )
            test_suites.append(test_suite)
        
        # Run build and tests
        result = await builder.build_project(project_path, language, environment, test_suites)
        
        return result
    
    async def _generate_coverage_reports(self, project_path: Path, 
                                       build_results: List[BuildResult]) -> List[CoverageReport]:
        """Generate coverage reports for all languages"""
        coverage_reports = []
        
        for build_result in build_results:
            if build_result.status == BuildStatus.SUCCESS:
                coverage = await self.coverage_analyzer.analyze_coverage(
                    project_path, build_result.language, build_result
                )
                if coverage:
                    coverage_reports.append(coverage)
        
        return coverage_reports
    
    async def _generate_performance_reports(self, build_results: List[BuildResult]) -> List[PerformanceMetrics]:
        """Generate performance reports from build results"""
        performance_reports = []
        
        for build_result in build_results:
            for test_result in build_result.test_results:
                # Extract performance metrics from test results
                metrics = PerformanceMetrics(
                    test_name=test_result.name,
                    execution_time=test_result.duration,
                    custom_metrics=test_result.performance_metrics
                )
                performance_reports.append(metrics)
        
        return performance_reports
    
    def _generate_summary(self, execution_report: TestExecutionReport) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_builds = len(execution_report.build_results)
        successful_builds = len([r for r in execution_report.build_results if r.status == BuildStatus.SUCCESS])
        
        total_tests = sum(len(r.test_results) for r in execution_report.build_results)
        passed_tests = sum(len([t for t in r.test_results if t.status == BuildStatus.SUCCESS]) 
                          for r in execution_report.build_results)
        
        # Coverage summary
        overall_coverage = 0.0
        if execution_report.coverage_reports:
            total_lines = sum(c.total_lines for c in execution_report.coverage_reports)
            covered_lines = sum(c.covered_lines for c in execution_report.coverage_reports)
            overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        
        return {
            "total_builds": total_builds,
            "successful_builds": successful_builds,
            "failed_builds": total_builds - successful_builds,
            "build_success_rate": (successful_builds / total_builds * 100) if total_builds > 0 else 0.0,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "test_success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
            "overall_coverage": overall_coverage,
            "total_duration": execution_report.total_duration,
            "languages_tested": list(set(r.language for r in execution_report.build_results)),
            "platforms_tested": list(set(r.platform for r in execution_report.build_results))
        }

async def run_comprehensive_tests(project_path: Path, config: TestConfiguration) -> TestExecutionReport:
    """Convenience function to run comprehensive tests"""
    orchestrator = TestOrchestrator()
    return await orchestrator.run_comprehensive_tests(project_path, config)