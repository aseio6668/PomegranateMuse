"""
Universal Builder System for MyndraComposer
Provides unified build testing across all supported languages and platforms
"""

import asyncio
import json
import logging
import subprocess
import shutil
import os
import platform
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

class BuildStatus(Enum):
    """Build status enumeration"""
    SUCCESS = "success"
    FAILURE = "failure" 
    WARNING = "warning"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"

class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"

@dataclass
class BuildEnvironment:
    """Build environment configuration"""
    platform: str = ""  # windows, linux, macos
    architecture: str = ""  # x64, arm64, x86
    language_version: str = ""
    dependencies: Dict[str, str] = None
    environment_vars: Dict[str, str] = None
    docker_image: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = {}
        if self.environment_vars is None:
            self.environment_vars = {}
        if not self.platform:
            self.platform = platform.system().lower()
        if not self.architecture:
            self.architecture = platform.machine().lower()

@dataclass
class LanguageConfig:
    """Language-specific configuration"""
    name: str
    file_extensions: List[str]
    build_command: str
    test_command: str
    package_manager: str = ""
    dependency_file: str = ""
    output_directory: str = "build"
    executable_extension: str = ""
    supports_cross_compilation: bool = False
    
    def __post_init__(self):
        if self.platform == "windows" and not self.executable_extension:
            self.executable_extension = ".exe"

@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    name: str
    supported_languages: List[str]
    docker_images: Dict[str, str] = None
    package_managers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.docker_images is None:
            self.docker_images = {}
        if self.package_managers is None:
            self.package_managers = {}

@dataclass
class TestResult:
    """Individual test result"""
    name: str
    test_type: TestType
    status: BuildStatus
    duration: float
    output: str = ""
    error_message: str = ""
    coverage_percentage: float = 0.0
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class BuildResult:
    """Complete build result"""
    language: str
    platform: str
    status: BuildStatus
    start_time: datetime
    end_time: datetime
    duration: float
    build_output: str = ""
    test_results: List[TestResult] = None
    artifacts: List[str] = None
    error_message: str = ""
    warnings: List[str] = None
    dependencies_resolved: bool = True
    coverage_report: Optional[str] = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
        if self.artifacts is None:
            self.artifacts = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    language: str
    test_files: List[str]
    test_type: TestType
    timeout: int = 300  # 5 minutes default
    environment: BuildEnvironment = None
    pre_commands: List[str] = None
    post_commands: List[str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = BuildEnvironment()
        if self.pre_commands is None:
            self.pre_commands = []
        if self.post_commands is None:
            self.post_commands = []

class DependencyManager:
    """Manages dependencies across different languages"""
    
    def __init__(self):
        self.package_managers = {
            "python": {"pip": "requirements.txt", "poetry": "pyproject.toml", "conda": "environment.yml"},
            "rust": {"cargo": "Cargo.toml"},
            "go": {"go": "go.mod"},
            "typescript": {"npm": "package.json", "yarn": "package.json", "pnpm": "package.json"},
            "javascript": {"npm": "package.json", "yarn": "package.json", "pnpm": "package.json"},
            "java": {"maven": "pom.xml", "gradle": "build.gradle"},
            "csharp": {"nuget": "*.csproj", "dotnet": "*.csproj"},
            "cpp": {"conan": "conanfile.txt", "vcpkg": "vcpkg.json", "cmake": "CMakeLists.txt"}
        }
        
    async def detect_dependencies(self, project_path: Path, language: str) -> Dict[str, Any]:
        """Detect dependencies for a project"""
        dependencies = {
            "package_manager": None,
            "dependency_file": None,
            "dependencies": [],
            "dev_dependencies": []
        }
        
        if language not in self.package_managers:
            return dependencies
            
        # Check for dependency files
        for pm, dep_file in self.package_managers[language].items():
            if "*" in dep_file:
                # Handle glob patterns
                pattern = dep_file.replace("*", "**/*")
                matches = list(project_path.glob(pattern))
                if matches:
                    dependencies["package_manager"] = pm
                    dependencies["dependency_file"] = str(matches[0])
                    break
            else:
                dep_path = project_path / dep_file
                if dep_path.exists():
                    dependencies["package_manager"] = pm
                    dependencies["dependency_file"] = str(dep_path)
                    break
        
        # Parse dependencies if file found
        if dependencies["dependency_file"]:
            try:
                await self._parse_dependency_file(dependencies)
            except Exception as e:
                logging.warning(f"Failed to parse dependency file: {e}")
                
        return dependencies
    
    async def _parse_dependency_file(self, dependencies: Dict[str, Any]):
        """Parse dependency file to extract dependencies"""
        dep_file = Path(dependencies["dependency_file"])
        pm = dependencies["package_manager"]
        
        if pm == "pip" and dep_file.name == "requirements.txt":
            with open(dep_file) as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                dependencies["dependencies"] = deps
                
        elif pm in ["npm", "yarn", "pnpm"] and dep_file.name == "package.json":
            with open(dep_file) as f:
                package_data = json.load(f)
                dependencies["dependencies"] = list(package_data.get("dependencies", {}).keys())
                dependencies["dev_dependencies"] = list(package_data.get("devDependencies", {}).keys())
                
        elif pm == "cargo" and dep_file.name == "Cargo.toml":
            # Parse TOML (simplified)
            with open(dep_file) as f:
                content = f.read()
                # Extract dependencies section (simplified parsing)
                deps = []
                in_deps = False
                for line in content.split('\n'):
                    if line.strip() == '[dependencies]':
                        in_deps = True
                        continue
                    elif line.strip().startswith('[') and in_deps:
                        break
                    elif in_deps and '=' in line:
                        dep_name = line.split('=')[0].strip()
                        if dep_name and not dep_name.startswith('#'):
                            deps.append(dep_name)
                dependencies["dependencies"] = deps
    
    async def install_dependencies(self, project_path: Path, dependencies: Dict[str, Any]) -> bool:
        """Install dependencies for a project"""
        pm = dependencies.get("package_manager")
        if not pm:
            return True
            
        install_commands = {
            "pip": ["pip", "install", "-r", "requirements.txt"],
            "npm": ["npm", "install"],
            "yarn": ["yarn", "install"],
            "pnpm": ["pnpm", "install"],
            "cargo": ["cargo", "fetch"],
            "go": ["go", "mod", "download"],
            "maven": ["mvn", "dependency:resolve"],
            "gradle": ["gradle", "dependencies"],
            "dotnet": ["dotnet", "restore"]
        }
        
        if pm not in install_commands:
            return True
            
        try:
            process = await asyncio.create_subprocess_exec(
                *install_commands[pm],
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            return process.returncode == 0
            
        except Exception as e:
            logging.error(f"Failed to install dependencies: {e}")
            return False

class UniversalBuilder:
    """Universal build system supporting multiple languages and platforms"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/universal_build")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dependency_manager = DependencyManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize language configurations
        self.language_configs = self._init_language_configs()
        self.platform_configs = self._init_platform_configs()
        
    def _init_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize language configurations"""
        return {
            "myndra": LanguageConfig(
                name="myndra",
                file_extensions=[".myn"],
                build_command="myndra build",
                test_command="myndra test",
                package_manager="myndra",
                dependency_file="myndra.toml",
                supports_cross_compilation=True
            ),
            "rust": LanguageConfig(
                name="rust", 
                file_extensions=[".rs"],
                build_command="cargo build",
                test_command="cargo test",
                package_manager="cargo",
                dependency_file="Cargo.toml",
                supports_cross_compilation=True
            ),
            "go": LanguageConfig(
                name="go",
                file_extensions=[".go"],
                build_command="go build",
                test_command="go test ./...",
                package_manager="go",
                dependency_file="go.mod",
                supports_cross_compilation=True
            ),
            "typescript": LanguageConfig(
                name="typescript",
                file_extensions=[".ts", ".tsx"],
                build_command="npm run build",
                test_command="npm test",
                package_manager="npm",
                dependency_file="package.json"
            ),
            "python": LanguageConfig(
                name="python",
                file_extensions=[".py"],
                build_command="python -m py_compile",
                test_command="python -m pytest",
                package_manager="pip",
                dependency_file="requirements.txt"
            ),
            "java": LanguageConfig(
                name="java",
                file_extensions=[".java"],
                build_command="mvn compile",
                test_command="mvn test",
                package_manager="maven",
                dependency_file="pom.xml"
            ),
            "cpp": LanguageConfig(
                name="cpp",
                file_extensions=[".cpp", ".cc", ".cxx", ".c"],
                build_command="cmake --build build",
                test_command="ctest",
                package_manager="cmake",
                dependency_file="CMakeLists.txt",
                supports_cross_compilation=True
            ),
            "csharp": LanguageConfig(
                name="csharp",
                file_extensions=[".cs"],
                build_command="dotnet build",
                test_command="dotnet test",
                package_manager="dotnet",
                dependency_file="*.csproj"
            )
        }
    
    def _init_platform_configs(self) -> Dict[str, PlatformConfig]:
        """Initialize platform configurations"""
        return {
            "windows": PlatformConfig(
                name="windows",
                supported_languages=["myndra", "rust", "go", "typescript", "python", "java", "cpp", "csharp"],
                docker_images={
                    "rust": "rust:latest",
                    "go": "golang:latest", 
                    "python": "python:3.11",
                    "java": "openjdk:17",
                    "csharp": "mcr.microsoft.com/dotnet/sdk:7.0"
                }
            ),
            "linux": PlatformConfig(
                name="linux",
                supported_languages=["myndra", "rust", "go", "typescript", "python", "java", "cpp", "csharp"],
                docker_images={
                    "rust": "rust:latest",
                    "go": "golang:latest",
                    "python": "python:3.11-slim",
                    "java": "openjdk:17-slim",
                    "cpp": "gcc:latest",
                    "csharp": "mcr.microsoft.com/dotnet/sdk:7.0"
                }
            ),
            "darwin": PlatformConfig(
                name="darwin",
                supported_languages=["myndra", "rust", "go", "typescript", "python", "java", "cpp", "csharp"],
                docker_images={
                    "rust": "rust:latest",
                    "go": "golang:latest",
                    "python": "python:3.11-slim",
                    "java": "openjdk:17"
                }
            )
        }
    
    async def build_project(self, project_path: Path, language: str, 
                           environment: BuildEnvironment = None,
                           test_suites: List[TestSuite] = None) -> BuildResult:
        """Build a project with comprehensive testing"""
        start_time = datetime.now()
        
        if environment is None:
            environment = BuildEnvironment()
            
        if test_suites is None:
            test_suites = []
            
        # Initialize build result
        result = BuildResult(
            language=language,
            platform=environment.platform,
            status=BuildStatus.ERROR,
            start_time=start_time,
            end_time=start_time,
            duration=0.0
        )
        
        try:
            # Validate language support
            if language not in self.language_configs:
                result.error_message = f"Unsupported language: {language}"
                return result
                
            lang_config = self.language_configs[language]
            
            # Detect and install dependencies
            dependencies = await self.dependency_manager.detect_dependencies(project_path, language)
            if dependencies["package_manager"]:
                self.logger.info(f"Installing dependencies using {dependencies['package_manager']}")
                deps_installed = await self.dependency_manager.install_dependencies(project_path, dependencies)
                result.dependencies_resolved = deps_installed
                
                if not deps_installed:
                    result.warnings.append("Failed to install some dependencies")
            
            # Run build
            build_success = await self._run_build(project_path, lang_config, environment, result)
            
            if build_success:
                # Run tests
                test_results = await self._run_tests(project_path, lang_config, test_suites, environment)
                result.test_results = test_results
                
                # Determine overall status
                failed_tests = [t for t in test_results if t.status == BuildStatus.FAILURE]
                if failed_tests:
                    result.status = BuildStatus.FAILURE
                    result.error_message = f"{len(failed_tests)} tests failed"
                else:
                    result.status = BuildStatus.SUCCESS
            else:
                result.status = BuildStatus.FAILURE
                
        except Exception as e:
            result.status = BuildStatus.ERROR
            result.error_message = str(e)
            self.logger.error(f"Build error: {e}")
            
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
        return result
    
    async def _run_build(self, project_path: Path, lang_config: LanguageConfig,
                        environment: BuildEnvironment, result: BuildResult) -> bool:
        """Run the build process"""
        try:
            # Prepare build command
            build_cmd = lang_config.build_command.split()
            
            # Set environment variables
            env = os.environ.copy()
            env.update(environment.environment_vars)
            
            # Create build process
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                cwd=project_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Run with timeout
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=600)  # 10 minute timeout
                result.build_output = stdout.decode() if stdout else ""
                
                success = process.returncode == 0
                if not success:
                    result.error_message = f"Build failed with exit code {process.returncode}"
                    
                return success
                
            except asyncio.TimeoutError:
                process.kill()
                result.status = BuildStatus.TIMEOUT
                result.error_message = "Build timed out"
                return False
                
        except Exception as e:
            result.error_message = f"Build execution error: {e}"
            return False
    
    async def _run_tests(self, project_path: Path, lang_config: LanguageConfig,
                        test_suites: List[TestSuite], environment: BuildEnvironment) -> List[TestResult]:
        """Run test suites"""
        test_results = []
        
        # If no test suites provided, run default tests
        if not test_suites:
            test_suites = [TestSuite(
                name="default",
                language=lang_config.name,
                test_files=[],
                test_type=TestType.UNIT,
                environment=environment
            )]
        
        for suite in test_suites:
            test_result = await self._run_test_suite(project_path, lang_config, suite)
            test_results.append(test_result)
            
        return test_results
    
    async def _run_test_suite(self, project_path: Path, lang_config: LanguageConfig,
                             suite: TestSuite) -> TestResult:
        """Run a single test suite"""
        start_time = datetime.now()
        
        result = TestResult(
            name=suite.name,
            test_type=suite.test_type,
            status=BuildStatus.ERROR,
            duration=0.0
        )
        
        try:
            # Run pre-commands
            for cmd in suite.pre_commands:
                await self._run_command(cmd.split(), project_path)
            
            # Prepare test command
            test_cmd = lang_config.test_command.split()
            
            # Set environment variables
            env = os.environ.copy()
            env.update(suite.environment.environment_vars)
            
            # Run tests
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                cwd=project_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=suite.timeout)
                result.output = stdout.decode() if stdout else ""
                
                if process.returncode == 0:
                    result.status = BuildStatus.SUCCESS
                else:
                    result.status = BuildStatus.FAILURE
                    result.error_message = f"Tests failed with exit code {process.returncode}"
                    
            except asyncio.TimeoutError:
                process.kill()
                result.status = BuildStatus.TIMEOUT
                result.error_message = "Tests timed out"
            
            # Run post-commands
            for cmd in suite.post_commands:
                await self._run_command(cmd.split(), project_path)
                
        except Exception as e:
            result.status = BuildStatus.ERROR
            result.error_message = str(e)
            
        finally:
            end_time = datetime.now()
            result.duration = (end_time - start_time).total_seconds()
            
        return result
    
    async def _run_command(self, cmd: List[str], cwd: Path) -> Tuple[bool, str]:
        """Run a shell command"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            stdout, _ = await process.communicate()
            output = stdout.decode() if stdout else ""
            
            return process.returncode == 0, output
            
        except Exception as e:
            return False, str(e)
    
    def get_supported_languages(self, platform: str = None) -> List[str]:
        """Get list of supported languages for a platform"""
        if platform is None:
            platform = platform.system().lower()
            
        if platform in self.platform_configs:
            return self.platform_configs[platform].supported_languages
        
        return list(self.language_configs.keys())
    
    def get_language_config(self, language: str) -> Optional[LanguageConfig]:
        """Get configuration for a specific language"""
        return self.language_configs.get(language)
    
    async def validate_environment(self, language: str, environment: BuildEnvironment) -> List[str]:
        """Validate build environment for a language"""
        issues = []
        
        if language not in self.language_configs:
            issues.append(f"Unsupported language: {language}")
            return issues
            
        lang_config = self.language_configs[language]
        
        # Check if language is supported on platform
        platform_config = self.platform_configs.get(environment.platform)
        if platform_config and language not in platform_config.supported_languages:
            issues.append(f"Language {language} not supported on {environment.platform}")
        
        # Check for required tools
        if lang_config.package_manager:
            if not shutil.which(lang_config.package_manager):
                issues.append(f"Package manager {lang_config.package_manager} not found")
        
        # Check build command availability
        build_tool = lang_config.build_command.split()[0]
        if not shutil.which(build_tool):
            issues.append(f"Build tool {build_tool} not found")
        
        return issues
    
    async def generate_build_report(self, results: List[BuildResult]) -> Dict[str, Any]:
        """Generate comprehensive build report"""
        total_builds = len(results)
        successful_builds = len([r for r in results if r.status == BuildStatus.SUCCESS])
        failed_builds = len([r for r in results if r.status == BuildStatus.FAILURE])
        
        # Calculate metrics
        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / total_builds if total_builds > 0 else 0
        
        # Test metrics
        total_tests = sum(len(r.test_results) for r in results)
        passed_tests = sum(len([t for t in r.test_results if t.status == BuildStatus.SUCCESS]) for r in results)
        
        report = {
            "summary": {
                "total_builds": total_builds,
                "successful_builds": successful_builds,
                "failed_builds": failed_builds,
                "success_rate": successful_builds / total_builds if total_builds > 0 else 0,
                "total_duration": total_duration,
                "average_duration": avg_duration,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "by_language": {},
            "by_platform": {},
            "failures": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # Group by language
        for result in results:
            lang = result.language
            if lang not in report["by_language"]:
                report["by_language"][lang] = {
                    "total": 0, "success": 0, "failure": 0, "avg_duration": 0
                }
            
            report["by_language"][lang]["total"] += 1
            if result.status == BuildStatus.SUCCESS:
                report["by_language"][lang]["success"] += 1
            elif result.status == BuildStatus.FAILURE:
                report["by_language"][lang]["failure"] += 1
        
        # Calculate averages
        for lang_stats in report["by_language"].values():
            if lang_stats["total"] > 0:
                lang_results = [r for r in results if r.language == lang]
                lang_stats["avg_duration"] = sum(r.duration for r in lang_results) / len(lang_results)
        
        # Group by platform
        for result in results:
            platform = result.platform
            if platform not in report["by_platform"]:
                report["by_platform"][platform] = {
                    "total": 0, "success": 0, "failure": 0, "avg_duration": 0
                }
            
            report["by_platform"][platform]["total"] += 1
            if result.status == BuildStatus.SUCCESS:
                report["by_platform"][platform]["success"] += 1
            elif result.status == BuildStatus.FAILURE:
                report["by_platform"][platform]["failure"] += 1
        
        # Calculate platform averages
        for platform_stats in report["by_platform"].values():
            if platform_stats["total"] > 0:
                platform_results = [r for r in results if r.platform == platform]
                platform_stats["avg_duration"] = sum(r.duration for r in platform_results) / len(platform_results)
        
        # Collect failures
        for result in results:
            if result.status == BuildStatus.FAILURE:
                report["failures"].append({
                    "language": result.language,
                    "platform": result.platform,
                    "error": result.error_message,
                    "duration": result.duration
                })
        
        return report

async def run_universal_build(project_path: Path, languages: List[str] = None,
                            platforms: List[str] = None, test_suites: List[TestSuite] = None) -> List[BuildResult]:
    """Run universal build across multiple languages and platforms"""
    builder = UniversalBuilder()
    results = []
    
    if languages is None:
        languages = ["myndra"]  # Default to Myndra
        
    if platforms is None:
        platforms = [platform.system().lower()]  # Current platform
        
    for lang in languages:
        for plat in platforms:
            environment = BuildEnvironment(platform=plat)
            
            # Validate environment
            issues = await builder.validate_environment(lang, environment)
            if issues:
                result = BuildResult(
                    language=lang,
                    platform=plat,
                    status=BuildStatus.SKIPPED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration=0.0,
                    error_message=f"Environment validation failed: {'; '.join(issues)}"
                )
                results.append(result)
                continue
            
            # Run build
            result = await builder.build_project(project_path, lang, environment, test_suites)
            results.append(result)
    
    return results