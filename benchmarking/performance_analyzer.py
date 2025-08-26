"""
Performance Benchmarking System for MyndraComposer
Provides comprehensive performance analysis and benchmarking capabilities
"""

import os
import time
import json
import psutil
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking"""
    test_name: str
    target_executable: str
    arguments: List[str] = None
    working_directory: str = None
    timeout_seconds: int = 300
    warmup_runs: int = 1
    benchmark_runs: int = 3
    collect_system_metrics: bool = True
    collect_custom_metrics: bool = False
    custom_metric_collectors: List[Callable] = None
    environment_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.arguments is None:
            self.arguments = []
        if self.custom_metric_collectors is None:
            self.custom_metric_collectors = []
        if self.environment_variables is None:
            self.environment_variables = {}


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config: BenchmarkConfig
    runs: List[PerformanceMetrics]
    summary: Dict[str, Any]
    timestamp: str
    success: bool
    error_details: Optional[str] = None


class SystemMonitor:
    """Monitor system resources during benchmark execution"""
    
    def __init__(self, process_pid: Optional[int] = None):
        self.process_pid = process_pid
        self.process = None
        self.monitoring = False
        self.metrics_history = []
        self.start_time = None
        self.initial_disk_io = None
        self.initial_network_io = None
        
    def start_monitoring(self):
        """Start monitoring system metrics"""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics_history = []
        
        # Get initial system state
        if psutil.disk_io_counters():
            self.initial_disk_io = psutil.disk_io_counters()
        if psutil.net_io_counters():
            self.initial_network_io = psutil.net_io_counters()
        
        # If monitoring a specific process
        if self.process_pid:
            try:
                self.process = psutil.Process(self.process_pid)
            except psutil.NoSuchProcess:
                self.process = None
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        if not self.metrics_history:
            return self._get_current_metrics()
        
        # Calculate summary metrics
        duration = time.time() - self.start_time
        cpu_values = [m.cpu_usage_percent for m in self.metrics_history]
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            cpu_usage_percent=sum(cpu_values) / len(cpu_values),
            memory_usage_mb=sum(memory_values) / len(memory_values),
            peak_memory_mb=max(memory_values),
            disk_io_read_mb=self._get_disk_read_mb(),
            disk_io_write_mb=self._get_disk_write_mb(),
            network_io_sent_mb=self._get_network_sent_mb(),
            network_io_recv_mb=self._get_network_recv_mb()
        )
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            self.metrics_history.append(self._get_current_metrics())
            time.sleep(0.1)  # Sample every 100ms
    
    def _get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics"""
        if self.process and self.process.is_running():
            # Process-specific metrics
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_percent = 0
                memory_mb = 0
        else:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            duration_seconds=time.time() - self.start_time if self.start_time else 0,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            peak_memory_mb=memory_mb,  # Will be updated in summary
            disk_io_read_mb=self._get_disk_read_mb(),
            disk_io_write_mb=self._get_disk_write_mb(),
            network_io_sent_mb=self._get_network_sent_mb(),
            network_io_recv_mb=self._get_network_recv_mb()
        )
    
    def _get_disk_read_mb(self) -> float:
        """Get disk read in MB since monitoring started"""
        if not self.initial_disk_io:
            return 0.0
        try:
            current = psutil.disk_io_counters()
            if current:
                return (current.read_bytes - self.initial_disk_io.read_bytes) / 1024 / 1024
        except:
            pass
        return 0.0
    
    def _get_disk_write_mb(self) -> float:
        """Get disk write in MB since monitoring started"""
        if not self.initial_disk_io:
            return 0.0
        try:
            current = psutil.disk_io_counters()
            if current:
                return (current.write_bytes - self.initial_disk_io.write_bytes) / 1024 / 1024
        except:
            pass
        return 0.0
    
    def _get_network_sent_mb(self) -> float:
        """Get network sent in MB since monitoring started"""
        if not self.initial_network_io:
            return 0.0
        try:
            current = psutil.net_io_counters()
            if current:
                return (current.bytes_sent - self.initial_network_io.bytes_sent) / 1024 / 1024
        except:
            pass
        return 0.0
    
    def _get_network_recv_mb(self) -> float:
        """Get network received in MB since monitoring started"""
        if not self.initial_network_io:
            return 0.0
        try:
            current = psutil.net_io_counters()
            if current:
                return (current.bytes_recv - self.initial_network_io.bytes_recv) / 1024 / 1024
        except:
            pass
        return 0.0


class PerformanceBenchmarker:
    """Main performance benchmarking class"""
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir) if results_dir else Path.cwd() / ".pomuse" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a complete benchmark with multiple runs"""
        print(f"🏃 Starting benchmark: {config.test_name}")
        
        try:
            runs = []
            
            # Warmup runs
            if config.warmup_runs > 0:
                print(f"🔥 Running {config.warmup_runs} warmup runs...")
                for i in range(config.warmup_runs):
                    await self._run_single_benchmark(config, is_warmup=True)
            
            # Actual benchmark runs
            print(f"📊 Running {config.benchmark_runs} benchmark runs...")
            for i in range(config.benchmark_runs):
                print(f"  Run {i + 1}/{config.benchmark_runs}")
                metrics = await self._run_single_benchmark(config)
                runs.append(metrics)
                
                # Brief pause between runs
                await asyncio.sleep(1)
            
            # Calculate summary statistics
            summary = self._calculate_summary(runs)
            
            result = BenchmarkResult(
                config=config,
                runs=runs,
                summary=summary,
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            # Save results
            await self._save_benchmark_result(result)
            
            print(f"✅ Benchmark completed: {config.test_name}")
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            error_result = BenchmarkResult(
                config=config,
                runs=[],
                summary={},
                timestamp=datetime.now().isoformat(),
                success=False,
                error_details=str(e)
            )
            print(f"❌ Benchmark failed: {e}")
            return error_result
    
    async def _run_single_benchmark(self, config: BenchmarkConfig, is_warmup: bool = False) -> PerformanceMetrics:
        """Run a single benchmark execution"""
        # Set up environment
        env = os.environ.copy()
        env.update(config.environment_variables)
        
        # Prepare command
        cmd = [config.target_executable] + config.arguments
        working_dir = config.working_directory or os.getcwd()
        
        # Start monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        process = None
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Update monitor to track specific process
            monitor.process_pid = process.pid
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )
            
            return_code = process.returncode
            
        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise Exception(f"Benchmark timed out after {config.timeout_seconds} seconds")
        
        except Exception as e:
            if process:
                process.terminate()
                await process.wait()
            raise e
        
        finally:
            # Stop monitoring and get metrics
            metrics = monitor.stop_monitoring()
            metrics.duration_seconds = time.time() - start_time
            metrics.return_code = return_code if 'return_code' in locals() else None
            
            if 'stderr' in locals() and stderr:
                metrics.error_message = stderr.decode('utf-8', errors='ignore')
            
            # Collect custom metrics if configured
            if config.collect_custom_metrics:
                for collector in config.custom_metric_collectors:
                    try:
                        custom_data = collector()
                        metrics.custom_metrics.update(custom_data)
                    except Exception as e:
                        print(f"Warning: Custom metric collector failed: {e}")
        
        return metrics
    
    def _calculate_summary(self, runs: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark runs"""
        if not runs:
            return {}
        
        durations = [r.duration_seconds for r in runs]
        cpu_usage = [r.cpu_usage_percent for r in runs]
        memory_usage = [r.memory_usage_mb for r in runs]
        peak_memory = [r.peak_memory_mb for r in runs]
        disk_read = [r.disk_io_read_mb for r in runs]
        disk_write = [r.disk_io_write_mb for r in runs]
        
        return {
            "run_count": len(runs),
            "duration": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "std_dev": self._calculate_std_dev(durations)
            },
            "cpu_usage": {
                "mean": sum(cpu_usage) / len(cpu_usage),
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "std_dev": self._calculate_std_dev(cpu_usage)
            },
            "memory_usage": {
                "mean": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage),
                "std_dev": self._calculate_std_dev(memory_usage)
            },
            "peak_memory": {
                "mean": sum(peak_memory) / len(peak_memory),
                "min": min(peak_memory),
                "max": max(peak_memory)
            },
            "disk_io": {
                "read_mb": {
                    "mean": sum(disk_read) / len(disk_read),
                    "total": sum(disk_read)
                },
                "write_mb": {
                    "mean": sum(disk_write) / len(disk_write),
                    "total": sum(disk_write)
                }
            },
            "success_rate": sum(1 for r in runs if r.return_code == 0) / len(runs) * 100
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.test_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        result_dict = {
            "config": asdict(result.config),
            "runs": [asdict(run) for run in result.runs],
            "summary": result.summary,
            "timestamp": result.timestamp,
            "success": result.success,
            "error_details": result.error_details
        }
        
        # Remove non-serializable custom metric collectors
        if "custom_metric_collectors" in result_dict["config"]:
            result_dict["config"]["custom_metric_collectors"] = []
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"📁 Results saved to: {filepath}")
    
    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary to console"""
        summary = result.summary
        
        print("\n📊 Benchmark Summary:")
        print(f"   Test: {result.config.test_name}")
        print(f"   Runs: {summary.get('run_count', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        if 'duration' in summary:
            duration = summary['duration']
            print(f"   Duration: {duration['mean']:.3f}s ± {duration['std_dev']:.3f}s")
        
        if 'memory_usage' in summary:
            memory = summary['memory_usage']
            print(f"   Memory: {memory['mean']:.1f}MB (peak: {summary['peak_memory']['max']:.1f}MB)")
        
        if 'cpu_usage' in summary:
            cpu = summary['cpu_usage']
            print(f"   CPU: {cpu['mean']:.1f}% ± {cpu['std_dev']:.1f}%")
    
    async def benchmark_code_generation(
        self, 
        source_files: List[str], 
        target_language: str,
        iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark code generation performance"""
        config = BenchmarkConfig(
            test_name=f"code_generation_{target_language}",
            target_executable="python",
            arguments=["-c", f"""
import sys
sys.path.append('.')
from language_targets import LanguageRegistry
from ml_providers import MLProviderManager

async def generate_code():
    registry = LanguageRegistry()
    ml_manager = MLProviderManager()
    
    generator = registry.get_generator('{target_language}')
    analysis = {{'files': {source_files}, 'summary': 'Benchmark test'}}
    
    result = await generator.generate(analysis, 'Convert to {target_language}')
    print(f'Generated {{len(result.get("code", ""))}} characters')

import asyncio
asyncio.run(generate_code())
"""],
            benchmark_runs=iterations,
            warmup_runs=1
        )
        
        return await self.run_benchmark(config)
    
    async def benchmark_build_process(
        self, 
        build_command: List[str], 
        working_dir: str = None,
        iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark build process performance"""
        config = BenchmarkConfig(
            test_name="build_process",
            target_executable=build_command[0],
            arguments=build_command[1:],
            working_directory=working_dir,
            benchmark_runs=iterations,
            warmup_runs=1
        )
        
        return await self.run_benchmark(config)
    
    async def benchmark_ml_inference(
        self, 
        model_name: str, 
        prompt: str,
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark ML model inference performance"""
        config = BenchmarkConfig(
            test_name=f"ml_inference_{model_name}",
            target_executable="python",
            arguments=["-c", f"""
import sys
sys.path.append('.')
from ollama_client import OllamaClient

async def test_inference():
    client = OllamaClient()
    await client.initialize()
    
    result = await client.generate_text(
        model='{model_name}',
        prompt='{prompt}'
    )
    print(f'Generated {{len(result)}} characters')

import asyncio
asyncio.run(test_inference())
"""],
            benchmark_runs=iterations,
            warmup_runs=1
        )
        
        return await self.run_benchmark(config)
    
    def load_benchmark_results(self, test_name: str = None) -> List[Dict[str, Any]]:
        """Load previous benchmark results"""
        results = []
        
        for result_file in self.results_dir.glob("*.json"):
            if test_name and test_name not in result_file.name:
                continue
            
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get('timestamp', ''))
        return results
    
    def compare_benchmarks(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple benchmark results"""
        if len(results) < 2:
            return {"error": "Need at least 2 results to compare"}
        
        comparison = {
            "result_count": len(results),
            "time_span": {
                "earliest": min(r.get('timestamp', '') for r in results),
                "latest": max(r.get('timestamp', '') for r in results)
            },
            "performance_trends": {}
        }
        
        # Analyze trends in key metrics
        for metric in ['duration', 'memory_usage', 'cpu_usage']:
            values = []
            timestamps = []
            
            for result in results:
                summary = result.get('summary', {})
                if metric in summary:
                    values.append(summary[metric].get('mean', 0))
                    timestamps.append(result.get('timestamp', ''))
            
            if len(values) > 1:
                # Simple trend analysis
                if values[-1] < values[0]:
                    trend = "improving"
                elif values[-1] > values[0]:
                    trend = "degrading"
                else:
                    trend = "stable"
                
                comparison["performance_trends"][metric] = {
                    "trend": trend,
                    "first_value": values[0],
                    "last_value": values[-1],
                    "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
                }
        
        return comparison


# Convenience functions for common benchmarks
async def benchmark_myndra_compilation(pom_file: str, iterations: int = 3) -> BenchmarkResult:
    """Benchmark Myndra compilation performance"""
    benchmarker = PerformanceBenchmarker()
    
    config = BenchmarkConfig(
        test_name="myndra_compilation",
        target_executable="myndra",
        arguments=["build", pom_file],
        benchmark_runs=iterations
    )
    
    return await benchmarker.run_benchmark(config)


async def benchmark_full_translation_pipeline(
    source_dir: str, 
    target_language: str, 
    iterations: int = 3
) -> Dict[str, BenchmarkResult]:
    """Benchmark the complete translation pipeline"""
    benchmarker = PerformanceBenchmarker()
    results = {}
    
    # Benchmark analysis phase
    analysis_config = BenchmarkConfig(
        test_name=f"analysis_pipeline_{target_language}",
        target_executable="python",
        arguments=["-c", f"""
import sys
sys.path.append('.')
from pomuse import CodeAnalyzer

async def run_analysis():
    analyzer = CodeAnalyzer()
    await analyzer.analyze_directory('{source_dir}')
    print('Analysis completed')

import asyncio
asyncio.run(run_analysis())
"""],
        benchmark_runs=iterations
    )
    
    results["analysis"] = await benchmarker.run_benchmark(analysis_config)
    
    # Benchmark generation phase
    results["generation"] = await benchmarker.benchmark_code_generation(
        [source_dir], target_language, iterations
    )
    
    return results


if __name__ == "__main__":
    async def main():
        benchmarker = PerformanceBenchmarker()
        
        # Example benchmark
        config = BenchmarkConfig(
            test_name="example_test",
            target_executable="python",
            arguments=["-c", "print('Hello, World!')"],
            benchmark_runs=3
        )
        
        result = await benchmarker.run_benchmark(config)
        print(f"Benchmark completed: {result.success}")
    
    asyncio.run(main())