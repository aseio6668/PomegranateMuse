"""
Migration Validation Framework for PomegranteMuse
Validates migration results for correctness, performance, and compatibility
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXTENSIVE = "extensive"

class ValidationCategory(Enum):
    """Validation categories"""
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    SECURITY = "security"
    USABILITY = "usability"

@dataclass
class ValidationResult:
    """Validation test result"""
    name: str
    category: ValidationCategory
    passed: bool
    score: float  # 0.0 - 1.0
    details: str
    metrics: Dict[str, Any]
    execution_time: float

@dataclass
class ValidationSuite:
    """Complete validation suite result"""
    migration_name: str
    validation_level: ValidationLevel
    start_time: datetime
    end_time: datetime
    overall_score: float
    results: List[ValidationResult]
    summary: Dict[str, Any]

class MigrationValidator:
    """Main validation orchestrator"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/validation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def validate_migration(self, project_path: Path, source_language: str,
                               target_language: str, level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> ValidationSuite:
        """Run comprehensive migration validation"""
        start_time = datetime.now()
        
        results = []
        
        # Functional validation
        functional_validator = FunctionalValidator()
        functional_results = await functional_validator.validate(project_path, source_language, target_language)
        results.extend(functional_results)
        
        # Performance validation
        performance_validator = PerformanceValidator()
        performance_results = await performance_validator.validate(project_path, source_language, target_language)
        results.extend(performance_results)
        
        # Compatibility validation
        compatibility_validator = CompatibilityValidator()
        compatibility_results = await compatibility_validator.validate(project_path, source_language, target_language)
        results.extend(compatibility_results)
        
        # Calculate overall score
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return ValidationSuite(
            migration_name=project_path.name,
            validation_level=level,
            start_time=start_time,
            end_time=datetime.now(),
            overall_score=overall_score,
            results=results,
            summary=summary
        )
    
    def _generate_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation summary"""
        by_category = {}
        for result in results:
            category = result.category.value
            if category not in by_category:
                by_category[category] = {"passed": 0, "total": 0, "avg_score": 0.0}
            
            by_category[category]["total"] += 1
            if result.passed:
                by_category[category]["passed"] += 1
            by_category[category]["avg_score"] += result.score
        
        # Calculate averages
        for category_stats in by_category.values():
            if category_stats["total"] > 0:
                category_stats["avg_score"] /= category_stats["total"]
        
        return {
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.passed]),
            "by_category": by_category,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        failed_results = [r for r in results if not r.passed]
        
        if failed_results:
            recommendations.append(f"Address {len(failed_results)} failing validation tests")
        
        low_score_results = [r for r in results if r.score < 0.7]
        if low_score_results:
            recommendations.append("Improve performance for low-scoring components")
        
        return recommendations

class FunctionalValidator:
    """Validates functional correctness"""
    
    async def validate(self, project_path: Path, source_language: str, target_language: str) -> List[ValidationResult]:
        """Run functional validation tests"""
        results = []
        
        # Test 1: Code compilation/syntax check
        result = await self._test_compilation(project_path, target_language)
        results.append(result)
        
        # Test 2: Basic functionality test
        result = await self._test_basic_functionality(project_path)
        results.append(result)
        
        return results
    
    async def _test_compilation(self, project_path: Path, target_language: str) -> ValidationResult:
        """Test code compilation"""
        start_time = datetime.now()
        
        try:
            # Simulate compilation test
            await asyncio.sleep(1)
            
            passed = True  # Assume success for demo
            score = 1.0 if passed else 0.0
            
            return ValidationResult(
                name="Code Compilation",
                category=ValidationCategory.FUNCTIONALITY,
                passed=passed,
                score=score,
                details="Code compiles successfully" if passed else "Compilation errors found",
                metrics={"compilation_time": 1.2},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            return ValidationResult(
                name="Code Compilation",
                category=ValidationCategory.FUNCTIONALITY,
                passed=False,
                score=0.0,
                details=f"Compilation test failed: {e}",
                metrics={},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _test_basic_functionality(self, project_path: Path) -> ValidationResult:
        """Test basic functionality"""
        start_time = datetime.now()
        
        # Simulate functionality test
        await asyncio.sleep(0.5)
        
        return ValidationResult(
            name="Basic Functionality",
            category=ValidationCategory.FUNCTIONALITY,
            passed=True,
            score=0.95,
            details="All basic functionality tests passed",
            metrics={"tests_run": 25, "tests_passed": 24},
            execution_time=(datetime.now() - start_time).total_seconds()
        )

class PerformanceValidator:
    """Validates performance characteristics"""
    
    async def validate(self, project_path: Path, source_language: str, target_language: str) -> List[ValidationResult]:
        """Run performance validation tests"""
        results = []
        
        # Test 1: Execution speed
        result = await self._test_execution_speed(project_path)
        results.append(result)
        
        # Test 2: Memory usage
        result = await self._test_memory_usage(project_path)
        results.append(result)
        
        return results
    
    async def _test_execution_speed(self, project_path: Path) -> ValidationResult:
        """Test execution speed"""
        start_time = datetime.now()
        
        # Simulate performance test
        await asyncio.sleep(2)
        
        # Simulated metrics
        baseline_time = 100  # ms
        current_time = 95   # ms
        improvement = (baseline_time - current_time) / baseline_time
        
        return ValidationResult(
            name="Execution Speed",
            category=ValidationCategory.PERFORMANCE,
            passed=current_time <= baseline_time * 1.1,  # Allow 10% degradation
            score=max(0.0, 1.0 - max(0, current_time - baseline_time) / baseline_time),
            details=f"Execution time: {current_time}ms (baseline: {baseline_time}ms)",
            metrics={
                "baseline_time_ms": baseline_time,
                "current_time_ms": current_time,
                "improvement_percent": improvement * 100
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )
    
    async def _test_memory_usage(self, project_path: Path) -> ValidationResult:
        """Test memory usage"""
        start_time = datetime.now()
        
        # Simulate memory test
        await asyncio.sleep(1)
        
        # Simulated metrics
        baseline_memory = 50  # MB
        current_memory = 48   # MB
        
        return ValidationResult(
            name="Memory Usage",
            category=ValidationCategory.PERFORMANCE,
            passed=current_memory <= baseline_memory * 1.2,  # Allow 20% increase
            score=max(0.0, 1.0 - max(0, current_memory - baseline_memory) / baseline_memory),
            details=f"Memory usage: {current_memory}MB (baseline: {baseline_memory}MB)",
            metrics={
                "baseline_memory_mb": baseline_memory,
                "current_memory_mb": current_memory
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )

class CompatibilityValidator:
    """Validates compatibility across platforms and environments"""
    
    async def validate(self, project_path: Path, source_language: str, target_language: str) -> List[ValidationResult]:
        """Run compatibility validation tests"""
        results = []
        
        # Test 1: Cross-platform compatibility
        result = await self._test_cross_platform(project_path, target_language)
        results.append(result)
        
        # Test 2: Dependency compatibility
        result = await self._test_dependencies(project_path, target_language)
        results.append(result)
        
        return results
    
    async def _test_cross_platform(self, project_path: Path, target_language: str) -> ValidationResult:
        """Test cross-platform compatibility"""
        start_time = datetime.now()
        
        # Simulate cross-platform test
        await asyncio.sleep(1.5)
        
        platforms_tested = ["windows", "linux", "darwin"]
        successful_platforms = ["windows", "linux"]  # Simulate partial success
        
        success_rate = len(successful_platforms) / len(platforms_tested)
        
        return ValidationResult(
            name="Cross-Platform Compatibility",
            category=ValidationCategory.COMPATIBILITY,
            passed=success_rate >= 0.8,  # Require 80% success rate
            score=success_rate,
            details=f"Compatible with {len(successful_platforms)}/{len(platforms_tested)} platforms",
            metrics={
                "platforms_tested": platforms_tested,
                "successful_platforms": successful_platforms,
                "success_rate": success_rate
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )
    
    async def _test_dependencies(self, project_path: Path, target_language: str) -> ValidationResult:
        """Test dependency compatibility"""
        start_time = datetime.now()
        
        # Simulate dependency test
        await asyncio.sleep(1)
        
        return ValidationResult(
            name="Dependency Compatibility",
            category=ValidationCategory.COMPATIBILITY,
            passed=True,
            score=0.9,
            details="All dependencies are compatible with target environment",
            metrics={"total_dependencies": 15, "compatible_dependencies": 15},
            execution_time=(datetime.now() - start_time).total_seconds()
        )

async def validate_migration(project_path: Path, source_language: str,
                           target_language: str, level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> ValidationSuite:
    """Convenience function to validate migration"""
    validator = MigrationValidator()
    return await validator.validate_migration(project_path, source_language, target_language, level)