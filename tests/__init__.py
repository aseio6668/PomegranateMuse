"""
Automated Testing Suite for PomegranteMuse
Comprehensive test coverage for the entire platform
"""

from .test_runner import (
    TestRunner,
    TestSuite,
    TestResult,
    run_all_tests,
    run_test_suite
)

from .unit_tests import (
    CoreUnitTests,
    PluginUnitTests,
    ConfigUnitTests,
    run_unit_tests
)

from .integration_tests import (
    SystemIntegrationTests,
    PluginIntegrationTests,
    WebDashboardTests,
    run_integration_tests
)

from .performance_tests import (
    PerformanceTestSuite,
    BenchmarkTests,
    LoadTests,
    run_performance_tests
)

__all__ = [
    # Test runner
    "TestRunner",
    "TestSuite", 
    "TestResult",
    "run_all_tests",
    "run_test_suite",
    
    # Unit tests
    "CoreUnitTests",
    "PluginUnitTests",
    "ConfigUnitTests",
    "run_unit_tests",
    
    # Integration tests
    "SystemIntegrationTests",
    "PluginIntegrationTests", 
    "WebDashboardTests",
    "run_integration_tests",
    
    # Performance tests
    "PerformanceTestSuite",
    "BenchmarkTests",
    "LoadTests",
    "run_performance_tests"
]