"""
Benchmarking Integration Module for PomegranteMuse
Integrates performance benchmarking with the main application
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .performance_analyzer import (
    PerformanceBenchmarker, BenchmarkConfig, BenchmarkResult,
    benchmark_pomegranate_compilation, benchmark_full_translation_pipeline
)


@dataclass
class BenchmarkSuite:
    """Definition of a benchmark suite"""
    name: str
    description: str
    benchmarks: List[BenchmarkConfig]
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class BenchmarkIntegration:
    """Main benchmarking integration class"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.benchmarker = PerformanceBenchmarker(
            results_dir=str(self.project_root / ".pomuse" / "benchmarks")
        )
        self.suites_dir = self.project_root / ".pomuse" / "benchmark_suites"
        self.suites_dir.mkdir(parents=True, exist_ok=True)
        self.built_in_suites = self._create_built_in_suites()
    
    def _create_built_in_suites(self) -> Dict[str, BenchmarkSuite]:
        """Create built-in benchmark suites"""
        suites = {}
        
        # Code generation performance suite
        suites["code_generation"] = BenchmarkSuite(
            name="Code Generation Performance",
            description="Benchmark code generation across different languages",
            benchmarks=[],  # Will be populated dynamically
            tags=["generation", "ml", "performance"]
        )
        
        # Build performance suite
        suites["build_performance"] = BenchmarkSuite(
            name="Build Performance",
            description="Benchmark compilation and build processes",
            benchmarks=[],  # Will be populated dynamically
            tags=["build", "compilation", "performance"]
        )
        
        # ML inference suite
        suites["ml_inference"] = BenchmarkSuite(
            name="ML Inference Performance",
            description="Benchmark ML model inference performance",
            benchmarks=[],  # Will be populated dynamically
            tags=["ml", "inference", "performance"]
        )
        
        # End-to-end pipeline suite
        suites["e2e_pipeline"] = BenchmarkSuite(
            name="End-to-End Pipeline",
            description="Benchmark complete translation pipeline",
            benchmarks=[],  # Will be populated dynamically
            tags=["e2e", "pipeline", "integration"]
        )
        
        return suites
    
    async def run_suite(self, suite_name: str, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run a complete benchmark suite"""
        if suite_name not in self.built_in_suites:
            custom_suite = self.load_custom_suite(suite_name)
            if not custom_suite:
                raise ValueError(f"Benchmark suite '{suite_name}' not found")
            return await self._run_custom_suite(custom_suite, **kwargs)
        
        print(f"🏃 Running benchmark suite: {suite_name}")
        
        if suite_name == "code_generation":
            return await self._run_code_generation_suite(**kwargs)
        elif suite_name == "build_performance":
            return await self._run_build_performance_suite(**kwargs)
        elif suite_name == "ml_inference":
            return await self._run_ml_inference_suite(**kwargs)
        elif suite_name == "e2e_pipeline":
            return await self._run_e2e_pipeline_suite(**kwargs)
        else:
            raise ValueError(f"Unknown suite: {suite_name}")
    
    async def _run_code_generation_suite(
        self, 
        languages: List[str] = None, 
        source_dir: str = None,
        iterations: int = 3
    ) -> Dict[str, BenchmarkResult]:
        """Run code generation benchmark suite"""
        if not languages:
            languages = ["rust", "go", "typescript", "python"]
        
        if not source_dir:
            source_dir = str(self.project_root)
        
        results = {}
        
        for language in languages:
            print(f"📊 Benchmarking {language} code generation...")
            try:
                result = await self.benchmarker.benchmark_code_generation(
                    source_files=[source_dir],
                    target_language=language,
                    iterations=iterations
                )
                results[f"generation_{language}"] = result
            except Exception as e:
                print(f"❌ Failed to benchmark {language}: {e}")
                continue
        
        return results
    
    async def _run_build_performance_suite(
        self, 
        build_commands: Dict[str, List[str]] = None,
        iterations: int = 3
    ) -> Dict[str, BenchmarkResult]:
        """Run build performance benchmark suite"""
        if not build_commands:
            # Auto-detect build commands
            build_commands = self._detect_build_commands()
        
        results = {}
        
        for build_name, command in build_commands.items():
            print(f"🔨 Benchmarking {build_name} build...")
            try:
                result = await self.benchmarker.benchmark_build_process(
                    build_command=command,
                    working_dir=str(self.project_root),
                    iterations=iterations
                )
                results[f"build_{build_name}"] = result
            except Exception as e:
                print(f"❌ Failed to benchmark {build_name}: {e}")
                continue
        
        return results
    
    async def _run_ml_inference_suite(
        self, 
        models: List[str] = None,
        prompt: str = None,
        iterations: int = 5
    ) -> Dict[str, BenchmarkResult]:
        """Run ML inference benchmark suite"""
        if not models:
            models = ["codellama:7b", "codellama:13b", "llama2:7b"]
        
        if not prompt:
            prompt = "Convert this Python function to Rust: def add(a, b): return a + b"
        
        results = {}
        
        for model in models:
            print(f"🧠 Benchmarking {model} inference...")
            try:
                result = await self.benchmarker.benchmark_ml_inference(
                    model_name=model,
                    prompt=prompt,
                    iterations=iterations
                )
                results[f"inference_{model.replace(':', '_')}"] = result
            except Exception as e:
                print(f"❌ Failed to benchmark {model}: {e}")
                continue
        
        return results
    
    async def _run_e2e_pipeline_suite(
        self, 
        source_dir: str = None,
        target_languages: List[str] = None,
        iterations: int = 3
    ) -> Dict[str, BenchmarkResult]:
        """Run end-to-end pipeline benchmark suite"""
        if not source_dir:
            source_dir = str(self.project_root)
        
        if not target_languages:
            target_languages = ["rust", "go"]
        
        all_results = {}
        
        for language in target_languages:
            print(f"🔄 Benchmarking E2E pipeline for {language}...")
            try:
                results = await benchmark_full_translation_pipeline(
                    source_dir=source_dir,
                    target_language=language,
                    iterations=iterations
                )
                
                for phase, result in results.items():
                    all_results[f"e2e_{language}_{phase}"] = result
                    
            except Exception as e:
                print(f"❌ Failed to benchmark E2E for {language}: {e}")
                continue
        
        return all_results
    
    def _detect_build_commands(self) -> Dict[str, List[str]]:
        """Auto-detect build commands based on project files"""
        commands = {}
        
        # Check for various build systems
        if (self.project_root / "Cargo.toml").exists():
            commands["rust_cargo"] = ["cargo", "build", "--release"]
        
        if (self.project_root / "go.mod").exists():
            commands["go_build"] = ["go", "build", "-o", "app"]
        
        if (self.project_root / "package.json").exists():
            commands["npm_build"] = ["npm", "run", "build"]
        
        if (self.project_root / "Makefile").exists():
            commands["make"] = ["make"]
        
        if (self.project_root / "CMakeLists.txt").exists():
            commands["cmake"] = ["cmake", "--build", "build"]
        
        # Pomegranate build (if pomegranate.toml exists)
        if (self.project_root / "pomegranate.toml").exists():
            commands["pomegranate"] = ["pomegranate", "build", "--release"]
        
        return commands
    
    async def benchmark_specific_file(
        self, 
        file_path: str, 
        benchmark_type: str = "compilation",
        iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark a specific file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if benchmark_type == "compilation":
            if file_path.suffix == ".pom":
                return await benchmark_pomegranate_compilation(
                    str(file_path), iterations
                )
            else:
                # Try to infer compilation command
                commands = self._get_file_compilation_command(file_path)
                if commands:
                    return await self.benchmarker.benchmark_build_process(
                        build_command=commands,
                        working_dir=str(file_path.parent),
                        iterations=iterations
                    )
                else:
                    raise ValueError(f"Don't know how to compile {file_path.suffix} files")
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    
    def _get_file_compilation_command(self, file_path: Path) -> Optional[List[str]]:
        """Get compilation command for a specific file"""
        suffix = file_path.suffix.lower()
        
        commands = {
            ".rs": ["rustc", str(file_path)],
            ".go": ["go", "build", str(file_path)],
            ".c": ["gcc", "-o", str(file_path.with_suffix(".out")), str(file_path)],
            ".cpp": ["g++", "-o", str(file_path.with_suffix(".out")), str(file_path)],
            ".java": ["javac", str(file_path)],
            ".py": ["python", "-m", "py_compile", str(file_path)]
        }
        
        return commands.get(suffix)
    
    def create_custom_suite(self, suite: BenchmarkSuite) -> bool:
        """Create a custom benchmark suite"""
        try:
            suite_file = self.suites_dir / f"{suite.name.lower().replace(' ', '_')}.json"
            
            suite_data = {
                "name": suite.name,
                "description": suite.description,
                "benchmarks": [
                    {
                        "test_name": b.test_name,
                        "target_executable": b.target_executable,
                        "arguments": b.arguments,
                        "working_directory": b.working_directory,
                        "timeout_seconds": b.timeout_seconds,
                        "warmup_runs": b.warmup_runs,
                        "benchmark_runs": b.benchmark_runs,
                        "environment_variables": b.environment_variables
                    }
                    for b in suite.benchmarks
                ],
                "tags": suite.tags
            }
            
            with open(suite_file, 'w') as f:
                json.dump(suite_data, f, indent=2)
            
            print(f"✅ Created custom benchmark suite: {suite_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create suite: {e}")
            return False
    
    def load_custom_suite(self, suite_name: str) -> Optional[BenchmarkSuite]:
        """Load a custom benchmark suite"""
        suite_file = self.suites_dir / f"{suite_name.lower().replace(' ', '_')}.json"
        
        if not suite_file.exists():
            return None
        
        try:
            with open(suite_file, 'r') as f:
                data = json.load(f)
            
            benchmarks = []
            for b_data in data.get("benchmarks", []):
                config = BenchmarkConfig(
                    test_name=b_data["test_name"],
                    target_executable=b_data["target_executable"],
                    arguments=b_data.get("arguments", []),
                    working_directory=b_data.get("working_directory"),
                    timeout_seconds=b_data.get("timeout_seconds", 300),
                    warmup_runs=b_data.get("warmup_runs", 1),
                    benchmark_runs=b_data.get("benchmark_runs", 3),
                    environment_variables=b_data.get("environment_variables", {})
                )
                benchmarks.append(config)
            
            return BenchmarkSuite(
                name=data["name"],
                description=data["description"],
                benchmarks=benchmarks,
                tags=data.get("tags", [])
            )
            
        except Exception as e:
            print(f"❌ Failed to load suite {suite_name}: {e}")
            return None
    
    async def _run_custom_suite(self, suite: BenchmarkSuite, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run a custom benchmark suite"""
        results = {}
        
        print(f"🏃 Running custom suite: {suite.name}")
        print(f"   Description: {suite.description}")
        
        for config in suite.benchmarks:
            print(f"📊 Running: {config.test_name}")
            try:
                result = await self.benchmarker.run_benchmark(config)
                results[config.test_name] = result
            except Exception as e:
                print(f"❌ Failed to run {config.test_name}: {e}")
                continue
        
        return results
    
    def list_available_suites(self) -> Dict[str, Dict[str, Any]]:
        """List all available benchmark suites"""
        suites = {}
        
        # Built-in suites
        for name, suite in self.built_in_suites.items():
            suites[name] = {
                "name": suite.name,
                "description": suite.description,
                "tags": suite.tags,
                "type": "built-in"
            }
        
        # Custom suites
        for suite_file in self.suites_dir.glob("*.json"):
            try:
                with open(suite_file, 'r') as f:
                    data = json.load(f)
                
                suites[suite_file.stem] = {
                    "name": data["name"],
                    "description": data["description"],
                    "tags": data.get("tags", []),
                    "type": "custom",
                    "file": str(suite_file)
                }
            except Exception:
                continue
        
        return suites
    
    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a performance report for recent benchmarks"""
        results = self.benchmarker.load_benchmark_results()
        
        # Filter recent results
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r for r in results 
            if datetime.fromisoformat(r.get('timestamp', '1970-01-01')) > cutoff_date
        ]
        
        if not recent_results:
            return {"error": f"No benchmark results found in the last {days} days"}
        
        report = {
            "period": f"Last {days} days",
            "total_benchmarks": len(recent_results),
            "test_types": {},
            "performance_trends": {},
            "recommendations": []
        }
        
        # Group by test type
        by_test_type = {}
        for result in recent_results:
            test_name = result.get('config', {}).get('test_name', 'unknown')
            if test_name not in by_test_type:
                by_test_type[test_name] = []
            by_test_type[test_name].append(result)
        
        # Analyze each test type
        for test_name, test_results in by_test_type.items():
            if len(test_results) > 1:
                comparison = self.benchmarker.compare_benchmarks(test_results)
                report["test_types"][test_name] = {
                    "run_count": len(test_results),
                    "performance_trends": comparison.get("performance_trends", {})
                }
                
                # Add to overall trends
                for metric, trend_data in comparison.get("performance_trends", {}).items():
                    if metric not in report["performance_trends"]:
                        report["performance_trends"][metric] = []
                    report["performance_trends"][metric].append({
                        "test": test_name,
                        "trend": trend_data["trend"],
                        "change_percent": trend_data["change_percent"]
                    })
        
        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(report)
        
        return report
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Check for degrading trends
        for metric, trends in report.get("performance_trends", {}).items():
            degrading_tests = [t for t in trends if t["trend"] == "degrading"]
            if degrading_tests:
                recommendations.append(
                    f"⚠️  {metric.title()} is degrading in {len(degrading_tests)} tests. "
                    f"Consider investigating: {', '.join(t['test'] for t in degrading_tests[:3])}"
                )
        
        # Check for high memory usage
        for test_name, data in report.get("test_types", {}).items():
            memory_trends = data.get("performance_trends", {}).get("memory_usage", {})
            if memory_trends.get("change_percent", 0) > 20:
                recommendations.append(
                    f"🧠 Memory usage increased significantly in {test_name}. "
                    f"Consider memory optimization."
                )
        
        # Check for slow performance
        for test_name, data in report.get("test_types", {}).items():
            duration_trends = data.get("performance_trends", {}).get("duration", {})
            if duration_trends.get("change_percent", 0) > 15:
                recommendations.append(
                    f"⏱️  Execution time increased in {test_name}. "
                    f"Consider performance profiling."
                )
        
        if not recommendations:
            recommendations.append("✅ No significant performance issues detected!")
        
        return recommendations


# CLI interface functions
async def run_benchmark_interactive(project_root: str):
    """Interactive benchmark runner"""
    integration = BenchmarkIntegration(project_root)
    
    print("🏃 PomegranteMuse Benchmarking System")
    print("=" * 50)
    
    # List available suites
    suites = integration.list_available_suites()
    print("\nAvailable benchmark suites:")
    for i, (key, info) in enumerate(suites.items(), 1):
        print(f"  {i}. {info['name']} ({info['type']})")
        print(f"     {info['description']}")
    
    print(f"\n  {len(suites) + 1}. Custom benchmark")
    print(f"  {len(suites) + 2}. Performance report")
    
    # Get user selection
    try:
        choice = int(input(f"\nSelect option (1-{len(suites) + 2}): "))
        
        if 1 <= choice <= len(suites):
            suite_key = list(suites.keys())[choice - 1]
            print(f"\n🏃 Running suite: {suites[suite_key]['name']}")
            
            # Get iterations
            iterations = input("Number of iterations [3]: ").strip()
            iterations = int(iterations) if iterations else 3
            
            results = await integration.run_suite(suite_key, iterations=iterations)
            
            print(f"\n✅ Suite completed! {len(results)} benchmarks run.")
            
        elif choice == len(suites) + 1:
            # Custom benchmark
            print("\n🔧 Custom Benchmark")
            executable = input("Executable: ").strip()
            args = input("Arguments (space-separated): ").strip().split()
            iterations = input("Iterations [3]: ").strip()
            iterations = int(iterations) if iterations else 3
            
            config = BenchmarkConfig(
                test_name="custom_benchmark",
                target_executable=executable,
                arguments=args,
                benchmark_runs=iterations
            )
            
            result = await integration.benchmarker.run_benchmark(config)
            print(f"\n✅ Custom benchmark completed: {result.success}")
            
        elif choice == len(suites) + 2:
            # Performance report
            days = input("Report period in days [30]: ").strip()
            days = int(days) if days else 30
            
            report = integration.get_performance_report(days)
            
            if "error" in report:
                print(f"❌ {report['error']}")
            else:
                print(f"\n📊 Performance Report ({report['period']})")
                print(f"Total benchmarks: {report['total_benchmarks']}")
                
                print("\n🔍 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  {rec}")
        
    except (ValueError, KeyboardInterrupt):
        print("\nBenchmark cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_benchmark_interactive(project_path))