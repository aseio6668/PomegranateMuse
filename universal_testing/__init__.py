"""
Universal Build Testing System for PomegranteMuse
Provides comprehensive build testing across all supported languages and platforms
"""

from .universal_builder import (
    UniversalBuilder,
    BuildEnvironment,
    BuildResult,
    BuildStatus,
    LanguageConfig,
    PlatformConfig,
    TestSuite,
    TestResult,
    DependencyManager,
    run_universal_build
)

from .language_builders import (
    PomegranateBuilder,
    RustBuilder,
    GoBuilder,
    TypeScriptBuilder,
    PythonBuilder,
    JavaBuilder,
    CppBuilder,
    CSharpBuilder,
    get_builder_for_language
)

from .test_orchestrator import (
    TestOrchestrator,
    TestConfiguration,
    TestReportGenerator,
    CoverageAnalyzer,
    PerformanceProfiler,
    run_comprehensive_tests
)

from .build_matrix import (
    BuildMatrix,
    MatrixConfiguration,
    CrossPlatformTester,
    CompatibilityChecker,
    generate_build_matrix
)

__all__ = [
    # Core universal building
    "UniversalBuilder",
    "BuildEnvironment",
    "BuildResult", 
    "BuildStatus",
    "LanguageConfig",
    "PlatformConfig",
    "TestSuite",
    "TestResult",
    "DependencyManager",
    "run_universal_build",
    
    # Language-specific builders
    "PomegranateBuilder",
    "RustBuilder", 
    "GoBuilder",
    "TypeScriptBuilder",
    "PythonBuilder",
    "JavaBuilder",
    "CppBuilder",
    "CSharpBuilder",
    "get_builder_for_language",
    
    # Test orchestration
    "TestOrchestrator",
    "TestConfiguration",
    "TestReportGenerator",
    "CoverageAnalyzer", 
    "PerformanceProfiler",
    "run_comprehensive_tests",
    
    # Build matrix and cross-platform
    "BuildMatrix",
    "MatrixConfiguration",
    "CrossPlatformTester",
    "CompatibilityChecker",
    "generate_build_matrix"
]