"""
Build Matrix System for Universal Testing
Provides cross-platform testing and compatibility validation
"""

import asyncio
import json
import logging
import platform
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .universal_builder import (
    BuildResult, BuildStatus, BuildEnvironment, 
    UniversalBuilder, run_universal_build
)
from .test_orchestrator import TestConfiguration, TestOrchestrator

class MatrixStrategy(Enum):
    """Build matrix strategies"""
    FULL = "full"  # Test all combinations
    MINIMAL = "minimal"  # Test only essential combinations
    TARGETED = "targeted"  # Test specific combinations
    INCREMENTAL = "incremental"  # Test based on changes

class CompatibilityLevel(Enum):
    """Compatibility levels"""
    COMPATIBLE = "compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"

@dataclass
class MatrixDimension:
    """A dimension in the build matrix"""
    name: str
    values: List[str]
    default: Optional[str] = None
    
    def __post_init__(self):
        if self.default is None and self.values:
            self.default = self.values[0]

@dataclass
class MatrixConfiguration:
    """Build matrix configuration"""
    languages: MatrixDimension
    platforms: MatrixDimension
    architectures: MatrixDimension
    language_versions: Dict[str, List[str]] = None
    custom_dimensions: List[MatrixDimension] = None
    strategy: MatrixStrategy = MatrixStrategy.FULL
    max_parallel_jobs: int = 4
    exclude_combinations: List[Dict[str, str]] = None
    include_combinations: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.language_versions is None:
            self.language_versions = {}
        if self.custom_dimensions is None:
            self.custom_dimensions = []
        if self.exclude_combinations is None:
            self.exclude_combinations = []
        if self.include_combinations is None:
            self.include_combinations = []

@dataclass
class CompatibilityResult:
    """Compatibility test result"""
    combination: Dict[str, str]
    level: CompatibilityLevel
    build_result: Optional[BuildResult]
    issues: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []

@dataclass
class MatrixExecutionResult:
    """Build matrix execution result"""
    configuration: MatrixConfiguration
    start_time: datetime
    end_time: datetime
    total_combinations: int
    executed_combinations: int
    successful_combinations: int
    build_results: List[BuildResult]
    compatibility_results: List[CompatibilityResult]
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.summary is None:
            self.summary = {}

class BuildMatrix:
    """Manages build matrix generation and execution"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/build_matrix")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Default matrix configurations
        self.default_platforms = ["windows", "linux", "darwin"]
        self.default_architectures = ["x86_64", "aarch64"]
        
        # Language compatibility matrix
        self.language_platform_compatibility = {
            "myndra": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "rust": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "go": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "typescript": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "python": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "java": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "cpp": {
                "windows": True,
                "linux": True,
                "darwin": True
            },
            "csharp": {
                "windows": True,
                "linux": True,
                "darwin": False  # Limited .NET Core support
            }
        }
    
    def generate_matrix_combinations(self, config: MatrixConfiguration) -> List[Dict[str, str]]:
        """Generate all possible matrix combinations"""
        combinations = []
        
        # Get base dimensions
        languages = config.languages.values
        platforms = config.platforms.values
        architectures = config.architectures.values
        
        # Generate all combinations
        for language in languages:
            for platform in platforms:
                for architecture in architectures:
                    combination = {
                        "language": language,
                        "platform": platform,
                        "architecture": architecture
                    }
                    
                    # Add language version if specified
                    if language in config.language_versions:
                        versions = config.language_versions[language]
                        for version in versions:
                            version_combination = combination.copy()
                            version_combination["language_version"] = version
                            
                            # Add custom dimensions
                            combinations.extend(self._add_custom_dimensions(
                                version_combination, config.custom_dimensions
                            ))
                    else:
                        # Add custom dimensions
                        combinations.extend(self._add_custom_dimensions(
                            combination, config.custom_dimensions
                        ))
        
        # Filter combinations based on strategy
        combinations = self._filter_combinations(combinations, config)
        
        return combinations
    
    def _add_custom_dimensions(self, base_combination: Dict[str, str], 
                             custom_dimensions: List[MatrixDimension]) -> List[Dict[str, str]]:
        """Add custom dimensions to a base combination"""
        if not custom_dimensions:
            return [base_combination]
        
        combinations = [base_combination]
        
        for dimension in custom_dimensions:
            new_combinations = []
            for combination in combinations:
                for value in dimension.values:
                    new_combination = combination.copy()
                    new_combination[dimension.name] = value
                    new_combinations.append(new_combination)
            combinations = new_combinations
        
        return combinations
    
    def _filter_combinations(self, combinations: List[Dict[str, str]], 
                           config: MatrixConfiguration) -> List[Dict[str, str]]:
        """Filter combinations based on strategy and exclusions"""
        # Apply exclusions
        filtered = []
        for combination in combinations:
            excluded = False
            for exclusion in config.exclude_combinations:
                if all(combination.get(k) == v for k, v in exclusion.items()):
                    excluded = True
                    break
            
            if not excluded:
                # Check compatibility
                language = combination["language"]
                platform = combination["platform"]
                
                if (language in self.language_platform_compatibility and
                    platform in self.language_platform_compatibility[language] and
                    self.language_platform_compatibility[language][platform]):
                    filtered.append(combination)
        
        # Add explicit inclusions
        for inclusion in config.include_combinations:
            if inclusion not in filtered:
                filtered.append(inclusion)
        
        # Apply strategy filtering
        if config.strategy == MatrixStrategy.MINIMAL:
            filtered = self._apply_minimal_strategy(filtered)
        elif config.strategy == MatrixStrategy.TARGETED:
            filtered = self._apply_targeted_strategy(filtered, config)
        
        return filtered
    
    def _apply_minimal_strategy(self, combinations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply minimal testing strategy"""
        minimal = []
        
        # Get current platform for local testing
        current_platform = platform.system().lower()
        current_arch = platform.machine().lower()
        
        # Select one combination per language for current platform
        languages_covered = set()
        for combination in combinations:
            language = combination["language"]
            plat = combination["platform"]
            arch = combination["architecture"]
            
            if (language not in languages_covered and 
                plat == current_platform and 
                arch == current_arch):
                minimal.append(combination)
                languages_covered.add(language)
        
        # If we don't have coverage for all languages, add more combinations
        all_languages = set(c["language"] for c in combinations)
        missing_languages = all_languages - languages_covered
        
        for language in missing_languages:
            # Find any working combination for this language
            for combination in combinations:
                if combination["language"] == language:
                    minimal.append(combination)
                    break
        
        return minimal
    
    def _apply_targeted_strategy(self, combinations: List[Dict[str, str]], 
                               config: MatrixConfiguration) -> List[Dict[str, str]]:
        """Apply targeted testing strategy"""
        # This could be enhanced to target specific combinations based on:
        # - Changed files
        # - Dependencies
        # - Risk assessment
        # For now, return all combinations
        return combinations
    
    async def execute_matrix(self, project_path: Path, config: MatrixConfiguration) -> MatrixExecutionResult:
        """Execute build matrix"""
        start_time = datetime.now()
        
        # Generate combinations
        combinations = self.generate_matrix_combinations(config)
        
        result = MatrixExecutionResult(
            configuration=config,
            start_time=start_time,
            end_time=start_time,
            total_combinations=len(combinations),
            executed_combinations=0,
            successful_combinations=0,
            build_results=[],
            compatibility_results=[]
        )
        
        try:
            self.logger.info(f"Executing build matrix with {len(combinations)} combinations")
            
            # Execute combinations in batches
            batch_size = config.max_parallel_jobs
            for i in range(0, len(combinations), batch_size):
                batch = combinations[i:i + batch_size]
                
                # Execute batch in parallel
                tasks = [
                    self._execute_combination(project_path, combination)
                    for combination in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, batch_result in enumerate(batch_results):
                    combination = batch[j]
                    result.executed_combinations += 1
                    
                    if isinstance(batch_result, Exception):
                        self.logger.error(f"Matrix execution error for {combination}: {batch_result}")
                        
                        # Create error result
                        error_result = BuildResult(
                            language=combination["language"],
                            platform=combination["platform"],
                            status=BuildStatus.ERROR,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            duration=0.0,
                            error_message=str(batch_result)
                        )
                        result.build_results.append(error_result)
                        
                    else:
                        build_result = batch_result
                        result.build_results.append(build_result)
                        
                        if build_result.status == BuildStatus.SUCCESS:
                            result.successful_combinations += 1
                        
                        # Generate compatibility result
                        compatibility = self._analyze_compatibility(combination, build_result)
                        result.compatibility_results.append(compatibility)
                
                # Progress logging
                progress = (result.executed_combinations / len(combinations)) * 100
                self.logger.info(f"Matrix execution progress: {progress:.1f}%")
        
        except Exception as e:
            self.logger.error(f"Matrix execution error: {e}")
        
        finally:
            result.end_time = datetime.now()
            result.summary = self._generate_matrix_summary(result)
            
            # Save results
            await self._save_matrix_results(result)
        
        return result
    
    async def _execute_combination(self, project_path: Path, combination: Dict[str, str]) -> BuildResult:
        """Execute a single matrix combination"""
        # Create build environment from combination
        environment = BuildEnvironment(
            platform=combination["platform"],
            architecture=combination["architecture"],
            language_version=combination.get("language_version", ""),
            environment_vars={}
        )
        
        # Add custom dimension values to environment
        for key, value in combination.items():
            if key not in ["language", "platform", "architecture", "language_version"]:
                environment.environment_vars[key.upper()] = value
        
        # Run build
        builder = UniversalBuilder()
        result = await builder.build_project(
            project_path, 
            combination["language"], 
            environment
        )
        
        return result
    
    def _analyze_compatibility(self, combination: Dict[str, str], 
                             build_result: BuildResult) -> CompatibilityResult:
        """Analyze compatibility for a combination"""
        if build_result.status == BuildStatus.SUCCESS:
            level = CompatibilityLevel.COMPATIBLE
            issues = []
            recommendations = []
        elif build_result.status == BuildStatus.FAILURE:
            level = CompatibilityLevel.INCOMPATIBLE
            issues = [build_result.error_message] if build_result.error_message else []
            recommendations = self._generate_compatibility_recommendations(combination, build_result)
        elif build_result.status in [BuildStatus.WARNING, BuildStatus.TIMEOUT]:
            level = CompatibilityLevel.PARTIALLY_COMPATIBLE
            issues = build_result.warnings if build_result.warnings else []
            recommendations = []
        else:
            level = CompatibilityLevel.UNKNOWN
            issues = []
            recommendations = []
        
        return CompatibilityResult(
            combination=combination,
            level=level,
            build_result=build_result,
            issues=issues,
            recommendations=recommendations
        )
    
    def _generate_compatibility_recommendations(self, combination: Dict[str, str], 
                                              build_result: BuildResult) -> List[str]:
        """Generate compatibility recommendations"""
        recommendations = []
        
        language = combination["language"]
        platform = combination["platform"]
        
        # Language-specific recommendations
        if language == "rust" and "cargo" in build_result.error_message.lower():
            recommendations.append("Install Rust toolchain and Cargo package manager")
        elif language == "go" and "go:" in build_result.error_message.lower():
            recommendations.append("Install Go compiler and set GOPATH environment variable")
        elif language == "typescript" and "npm" in build_result.error_message.lower():
            recommendations.append("Install Node.js and npm package manager")
        elif language == "python" and "python" in build_result.error_message.lower():
            recommendations.append("Install Python interpreter and pip package manager")
        
        # Platform-specific recommendations
        if platform == "windows" and "command not found" in build_result.error_message.lower():
            recommendations.append("Add required tools to Windows PATH environment variable")
        elif platform == "linux" and "permission denied" in build_result.error_message.lower():
            recommendations.append("Check file permissions and executable flags")
        
        return recommendations
    
    def _generate_matrix_summary(self, result: MatrixExecutionResult) -> Dict[str, Any]:
        """Generate matrix execution summary"""
        # Calculate success rates
        success_rate = (result.successful_combinations / result.executed_combinations * 100) if result.executed_combinations > 0 else 0
        
        # Group results by dimensions
        by_language = {}
        by_platform = {}
        
        for build_result in result.build_results:
            lang = build_result.language
            plat = build_result.platform
            
            # By language
            if lang not in by_language:
                by_language[lang] = {"total": 0, "success": 0, "failure": 0}
            by_language[lang]["total"] += 1
            if build_result.status == BuildStatus.SUCCESS:
                by_language[lang]["success"] += 1
            else:
                by_language[lang]["failure"] += 1
            
            # By platform
            if plat not in by_platform:
                by_platform[plat] = {"total": 0, "success": 0, "failure": 0}
            by_platform[plat]["total"] += 1
            if build_result.status == BuildStatus.SUCCESS:
                by_platform[plat]["success"] += 1
            else:
                by_platform[plat]["failure"] += 1
        
        # Compatibility analysis
        compatibility_summary = {
            "compatible": len([c for c in result.compatibility_results if c.level == CompatibilityLevel.COMPATIBLE]),
            "partially_compatible": len([c for c in result.compatibility_results if c.level == CompatibilityLevel.PARTIALLY_COMPATIBLE]),
            "incompatible": len([c for c in result.compatibility_results if c.level == CompatibilityLevel.INCOMPATIBLE]),
            "unknown": len([c for c in result.compatibility_results if c.level == CompatibilityLevel.UNKNOWN])
        }
        
        return {
            "execution_summary": {
                "total_combinations": result.total_combinations,
                "executed_combinations": result.executed_combinations,
                "successful_combinations": result.successful_combinations,
                "success_rate": success_rate,
                "execution_time": (result.end_time - result.start_time).total_seconds()
            },
            "by_language": by_language,
            "by_platform": by_platform,
            "compatibility_summary": compatibility_summary,
            "recommendations": self._generate_overall_recommendations(result)
        }
    
    def _generate_overall_recommendations(self, result: MatrixExecutionResult) -> List[str]:
        """Generate overall recommendations based on matrix results"""
        recommendations = []
        
        # Analyze failure patterns
        failures_by_language = {}
        failures_by_platform = {}
        
        for build_result in result.build_results:
            if build_result.status != BuildStatus.SUCCESS:
                lang = build_result.language
                plat = build_result.platform
                
                if lang not in failures_by_language:
                    failures_by_language[lang] = 0
                failures_by_language[lang] += 1
                
                if plat not in failures_by_platform:
                    failures_by_platform[plat] = 0
                failures_by_platform[plat] += 1
        
        # Generate recommendations based on patterns
        if failures_by_language:
            most_failing_lang = max(failures_by_language.items(), key=lambda x: x[1])
            recommendations.append(f"Consider improving {most_failing_lang[0]} toolchain setup ({most_failing_lang[1]} failures)")
        
        if failures_by_platform:
            most_failing_platform = max(failures_by_platform.items(), key=lambda x: x[1])
            recommendations.append(f"Review {most_failing_platform[0]} environment configuration ({most_failing_platform[1]} failures)")
        
        # Success rate recommendations
        success_rate = (result.successful_combinations / result.executed_combinations * 100) if result.executed_combinations > 0 else 0
        if success_rate < 50:
            recommendations.append("Consider reviewing build configuration and dependencies")
        elif success_rate < 80:
            recommendations.append("Some combinations need attention for better compatibility")
        
        return recommendations
    
    async def _save_matrix_results(self, result: MatrixExecutionResult):
        """Save matrix results to cache"""
        results_file = self.cache_dir / f"matrix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        data = {
            "configuration": asdict(result.configuration),
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_combinations": result.total_combinations,
            "executed_combinations": result.executed_combinations,
            "successful_combinations": result.successful_combinations,
            "build_results": [asdict(br) for br in result.build_results],
            "compatibility_results": [asdict(cr) for cr in result.compatibility_results],
            "summary": result.summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Matrix results saved to {results_file}")

class CrossPlatformTester:
    """Specialized cross-platform testing functionality"""
    
    def __init__(self):
        self.build_matrix = BuildMatrix()
        self.logger = logging.getLogger(__name__)
    
    async def test_cross_platform_compatibility(self, project_path: Path, 
                                              languages: List[str] = None) -> MatrixExecutionResult:
        """Test cross-platform compatibility for specified languages"""
        if languages is None:
            languages = ["myndra", "rust", "go"]
        
        # Create matrix configuration for cross-platform testing
        config = MatrixConfiguration(
            languages=MatrixDimension("language", languages),
            platforms=MatrixDimension("platform", ["windows", "linux", "darwin"]),
            architectures=MatrixDimension("architecture", ["x86_64", "aarch64"]),
            strategy=MatrixStrategy.FULL,
            max_parallel_jobs=2  # Conservative for cross-platform
        )
        
        return await self.build_matrix.execute_matrix(project_path, config)
    
    async def test_language_versions(self, project_path: Path, language: str, 
                                   versions: List[str]) -> MatrixExecutionResult:
        """Test compatibility across different language versions"""
        config = MatrixConfiguration(
            languages=MatrixDimension("language", [language]),
            platforms=MatrixDimension("platform", [platform.system().lower()]),
            architectures=MatrixDimension("architecture", [platform.machine().lower()]),
            language_versions={language: versions},
            strategy=MatrixStrategy.FULL
        )
        
        return await self.build_matrix.execute_matrix(project_path, config)

class CompatibilityChecker:
    """Checks compatibility and provides recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_environment_compatibility(self, language: str, 
                                            environment: BuildEnvironment) -> CompatibilityResult:
        """Check if an environment is compatible with a language"""
        combination = {
            "language": language,
            "platform": environment.platform,
            "architecture": environment.architecture
        }
        
        # Simulate build to check compatibility
        try:
            builder = UniversalBuilder()
            issues = await builder.validate_environment(language, environment)
            
            if not issues:
                level = CompatibilityLevel.COMPATIBLE
            elif len(issues) < 3:
                level = CompatibilityLevel.PARTIALLY_COMPATIBLE
            else:
                level = CompatibilityLevel.INCOMPATIBLE
            
            return CompatibilityResult(
                combination=combination,
                level=level,
                build_result=None,
                issues=issues,
                recommendations=self._generate_environment_recommendations(language, environment, issues)
            )
            
        except Exception as e:
            return CompatibilityResult(
                combination=combination,
                level=CompatibilityLevel.UNKNOWN,
                build_result=None,
                issues=[str(e)],
                recommendations=[]
            )
    
    def _generate_environment_recommendations(self, language: str, environment: BuildEnvironment, 
                                            issues: List[str]) -> List[str]:
        """Generate environment-specific recommendations"""
        recommendations = []
        
        for issue in issues:
            if "not found" in issue.lower():
                if language == "rust":
                    recommendations.append("Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
                elif language == "go":
                    recommendations.append("Install Go: https://golang.org/dl/")
                elif language == "python":
                    recommendations.append("Install Python: https://python.org/downloads/")
                elif language == "typescript":
                    recommendations.append("Install Node.js: https://nodejs.org/")
        
        return recommendations

async def generate_build_matrix(project_path: Path, languages: List[str] = None, 
                              strategy: MatrixStrategy = MatrixStrategy.MINIMAL) -> MatrixExecutionResult:
    """Generate and execute build matrix"""
    if languages is None:
        languages = ["myndra"]
    
    matrix = BuildMatrix()
    
    config = MatrixConfiguration(
        languages=MatrixDimension("language", languages),
        platforms=MatrixDimension("platform", ["windows", "linux", "darwin"]),
        architectures=MatrixDimension("architecture", ["x86_64"]),
        strategy=strategy,
        max_parallel_jobs=3
    )
    
    return await matrix.execute_matrix(project_path, config)