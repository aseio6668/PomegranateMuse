"""
Unit Tests for PomegranteMuse
Tests individual components in isolation
"""

from .test_core import CoreUnitTests
from .test_plugins import PluginUnitTests
from .test_config import ConfigUnitTests
from .test_analysis import AnalysisUnitTests
from .test_generation import GenerationUnitTests

def register_unit_tests(test_runner):
    """Register all unit test suites"""
    from ..test_runner import TestSuite
    
    # Core functionality tests
    core_suite = TestSuite("unit_core", "Core functionality unit tests")
    core_suite.add_test(CoreUnitTests)
    test_runner.register_test_suite(core_suite)
    
    # Plugin system tests
    plugin_suite = TestSuite("unit_plugins", "Plugin system unit tests")
    plugin_suite.add_test(PluginUnitTests)
    test_runner.register_test_suite(plugin_suite)
    
    # Configuration management tests
    config_suite = TestSuite("unit_config", "Configuration management unit tests")
    config_suite.add_test(ConfigUnitTests)
    test_runner.register_test_suite(config_suite)
    
    # Analysis engine tests
    analysis_suite = TestSuite("unit_analysis", "Analysis engine unit tests")
    analysis_suite.add_test(AnalysisUnitTests)
    test_runner.register_test_suite(analysis_suite)
    
    # Code generation tests
    generation_suite = TestSuite("unit_generation", "Code generation unit tests")
    generation_suite.add_test(GenerationUnitTests)
    test_runner.register_test_suite(generation_suite)

def run_unit_tests(test_runner):
    """Run all unit tests"""
    register_unit_tests(test_runner)
    
    unit_suites = [
        "unit_core",
        "unit_plugins", 
        "unit_config",
        "unit_analysis",
        "unit_generation"
    ]
    
    results = {}
    for suite_name in unit_suites:
        results[suite_name] = test_runner.run_suite(suite_name)
    
    return results

__all__ = [
    "CoreUnitTests",
    "PluginUnitTests",
    "ConfigUnitTests", 
    "AnalysisUnitTests",
    "GenerationUnitTests",
    "register_unit_tests",
    "run_unit_tests"
]