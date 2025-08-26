"""
Settings Management for MyndraComposer
Provides structured settings classes and validation
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MLProviderSettings:
    """ML Provider settings"""
    name: str
    enabled: bool = True
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 4000
    timeout: int = 120
    rate_limit: int = 60  # requests per minute
    custom_headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class LanguageSettings:
    """Language-specific settings"""
    name: str
    enabled: bool = True
    file_extensions: List[str] = field(default_factory=list)
    build_command: str = ""
    test_command: str = ""
    quality_threshold: float = 0.8
    complexity_limit: int = 10

@dataclass
class BuildSettings:
    """Build and testing settings"""
    parallel_jobs: int = 4
    timeout: int = 600
    retry_count: int = 3
    cleanup_artifacts: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    environments: List[str] = field(default_factory=lambda: ["development", "staging", "production"])

@dataclass
class SecuritySettings:
    """Security and compliance settings"""
    scan_enabled: bool = True
    fail_on_critical: bool = True
    fail_on_high: bool = False
    allowed_vulnerabilities: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    security_policies: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationSettings:
    """Team collaboration settings"""
    real_time_enabled: bool = True
    auto_save_interval: int = 30  # seconds
    conflict_resolution: str = "merge"  # merge, overwrite, prompt
    notification_channels: List[str] = field(default_factory=list)
    review_required_roles: List[str] = field(default_factory=lambda: ["senior_developer", "tech_lead"])

@dataclass
class EnterpriseSettings:
    """Enterprise integration settings"""
    sso_enabled: bool = False
    sso_provider: str = ""
    project_management_integration: str = ""
    communication_platform: str = ""
    monitoring_system: str = ""
    audit_logging: bool = True
    data_retention_days: int = 90

@dataclass
class PerformanceSettings:
    """Performance and optimization settings"""
    enable_caching: bool = True
    max_memory_usage: int = 2048  # MB
    max_cpu_usage: int = 80  # percentage
    benchmark_enabled: bool = True
    profiling_enabled: bool = False
    optimization_level: str = "balanced"  # fast, balanced, thorough

@dataclass
class UserPreferences:
    """User preference settings"""
    theme: str = "default"
    editor: str = "vscode"
    auto_save: bool = True
    verbose_output: bool = False
    color_output: bool = True
    keyboard_shortcuts: Dict[str, str] = field(default_factory=dict)
    recent_projects: List[str] = field(default_factory=list)
    max_recent_projects: int = 10

@dataclass
class GlobalSettings:
    """Global system settings"""
    version: str = "1.0.0"
    install_path: str = ""
    data_directory: str = ""
    log_level: str = "INFO"
    log_file: str = ""
    update_check_enabled: bool = True
    telemetry_enabled: bool = False
    crash_reporting: bool = True
    
    ml_providers: Dict[str, MLProviderSettings] = field(default_factory=dict)
    languages: Dict[str, LanguageSettings] = field(default_factory=dict)
    build: BuildSettings = field(default_factory=BuildSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)

@dataclass
class ProjectSettings:
    """Project-specific settings"""
    name: str = ""
    description: str = ""
    source_language: str = ""
    target_language: str = "myndra"
    migration_strategy: str = "incremental"
    
    # Project structure
    source_directories: List[str] = field(default_factory=lambda: ["src", "lib"])
    output_directory: str = "output"
    test_directories: List[str] = field(default_factory=lambda: ["tests", "test"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["node_modules", ".git", "__pycache__"])
    
    # Project-specific overrides
    build: Optional[BuildSettings] = None
    security: Optional[SecuritySettings] = None
    collaboration: Optional[CollaborationSettings] = None
    enterprise: Optional[EnterpriseSettings] = None

@dataclass 
class UserSettings:
    """User-specific settings"""
    user_id: str = ""
    name: str = ""
    email: str = ""
    organization: str = ""
    role: str = "developer"
    
    preferences: UserPreferences = field(default_factory=UserPreferences)
    api_keys: Dict[str, str] = field(default_factory=dict)
    workspace_settings: Dict[str, Any] = field(default_factory=dict)

class DefaultSettings:
    """Default settings factory"""
    
    @staticmethod
    def get_default_ml_providers() -> Dict[str, MLProviderSettings]:
        """Get default ML provider configurations"""
        return {
            "ollama": MLProviderSettings(
                name="ollama",
                enabled=True,
                base_url="http://localhost:11434",
                model="codellama",
                max_tokens=4000,
                timeout=120
            ),
            "openai": MLProviderSettings(
                name="openai",
                enabled=False,
                base_url="https://api.openai.com/v1",
                api_key="",
                model="gpt-4",
                max_tokens=4000,
                timeout=60
            ),
            "anthropic": MLProviderSettings(
                name="anthropic", 
                enabled=False,
                base_url="https://api.anthropic.com",
                api_key="",
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                timeout=60
            ),
            "google": MLProviderSettings(
                name="google",
                enabled=False,
                base_url="https://generativelanguage.googleapis.com",
                api_key="",
                model="gemini-pro",
                max_tokens=4000,
                timeout=60
            )
        }
    
    @staticmethod
    def get_default_languages() -> Dict[str, LanguageSettings]:
        """Get default language configurations"""
        return {
            "python": LanguageSettings(
                name="python",
                enabled=True,
                file_extensions=[".py"],
                build_command="python -m py_compile",
                test_command="python -m pytest",
                quality_threshold=0.8,
                complexity_limit=10
            ),
            "javascript": LanguageSettings(
                name="javascript",
                enabled=True,
                file_extensions=[".js", ".jsx"],
                build_command="npm run build",
                test_command="npm test",
                quality_threshold=0.8,
                complexity_limit=10
            ),
            "typescript": LanguageSettings(
                name="typescript",
                enabled=True,
                file_extensions=[".ts", ".tsx"],
                build_command="npm run build",
                test_command="npm test",
                quality_threshold=0.8,
                complexity_limit=10
            ),
            "rust": LanguageSettings(
                name="rust",
                enabled=True,
                file_extensions=[".rs"],
                build_command="cargo build",
                test_command="cargo test",
                quality_threshold=0.9,
                complexity_limit=8
            ),
            "go": LanguageSettings(
                name="go",
                enabled=True,
                file_extensions=[".go"],
                build_command="go build",
                test_command="go test ./...",
                quality_threshold=0.85,
                complexity_limit=8
            ),
            "java": LanguageSettings(
                name="java",
                enabled=True,
                file_extensions=[".java"],
                build_command="mvn compile",
                test_command="mvn test",
                quality_threshold=0.8,
                complexity_limit=12
            ),
            "cpp": LanguageSettings(
                name="cpp",
                enabled=True,
                file_extensions=[".cpp", ".cc", ".cxx", ".h", ".hpp"],
                build_command="cmake --build build",
                test_command="ctest",
                quality_threshold=0.8,
                complexity_limit=15
            ),
            "csharp": LanguageSettings(
                name="csharp",
                enabled=True,
                file_extensions=[".cs"],
                build_command="dotnet build",
                test_command="dotnet test",
                quality_threshold=0.8,
                complexity_limit=10
            ),
            "myndra": LanguageSettings(
                name="myndra",
                enabled=True,
                file_extensions=[".myn"],
                build_command="myndra build",
                test_command="myndra test",
                quality_threshold=0.9,
                complexity_limit=8
            )
        }
    
    @staticmethod
    def get_default_global_settings() -> GlobalSettings:
        """Get default global settings"""
        return GlobalSettings(
            version="1.0.0",
            install_path=str(Path(__file__).parent.parent),
            data_directory=str(Path.home() / ".myndra"),
            log_level="INFO",
            log_file="myndra.log",
            update_check_enabled=True,
            telemetry_enabled=False,
            crash_reporting=True,
            ml_providers=DefaultSettings.get_default_ml_providers(),
            languages=DefaultSettings.get_default_languages(),
            build=BuildSettings(),
            security=SecuritySettings(),
            performance=PerformanceSettings()
        )
    
    @staticmethod
    def get_default_user_settings() -> UserSettings:
        """Get default user settings"""
        return UserSettings(
            user_id=os.getenv("USER", "unknown"),
            name=os.getenv("USER", "Unknown User"),
            email="",
            organization="",
            role="developer",
            preferences=UserPreferences(),
            api_keys={},
            workspace_settings={}
        )
    
    @staticmethod
    def get_default_project_settings() -> ProjectSettings:
        """Get default project settings"""
        return ProjectSettings(
            name="Untitled Project",
            description="",
            source_language="python",
            target_language="myndra",
            migration_strategy="incremental",
            source_directories=["src", "lib"],
            output_directory="output",
            test_directories=["tests", "test"],
            exclude_patterns=["node_modules", ".git", "__pycache__", "*.pyc", "*.pyo"]
        )

class SettingsValidator:
    """Validates settings configurations"""
    
    @staticmethod
    def validate_ml_provider(provider: MLProviderSettings) -> List[str]:
        """Validate ML provider settings"""
        errors = []
        
        if not provider.name:
            errors.append("ML provider name is required")
        
        if provider.enabled:
            if not provider.base_url:
                errors.append(f"Base URL required for enabled provider: {provider.name}")
            
            if provider.name in ["openai", "anthropic", "google"] and not provider.api_key:
                errors.append(f"API key required for provider: {provider.name}")
            
            if provider.max_tokens <= 0:
                errors.append(f"Max tokens must be positive: {provider.name}")
            
            if provider.timeout <= 0:
                errors.append(f"Timeout must be positive: {provider.name}")
        
        return errors
    
    @staticmethod
    def validate_language(language: LanguageSettings) -> List[str]:
        """Validate language settings"""
        errors = []
        
        if not language.name:
            errors.append("Language name is required")
        
        if not language.file_extensions:
            errors.append(f"File extensions required for language: {language.name}")
        
        if language.quality_threshold < 0 or language.quality_threshold > 1:
            errors.append(f"Quality threshold must be between 0 and 1: {language.name}")
        
        if language.complexity_limit <= 0:
            errors.append(f"Complexity limit must be positive: {language.name}")
        
        return errors
    
    @staticmethod
    def validate_build_settings(build: BuildSettings) -> List[str]:
        """Validate build settings"""
        errors = []
        
        if build.parallel_jobs <= 0:
            errors.append("Parallel jobs must be positive")
        
        if build.timeout <= 0:
            errors.append("Build timeout must be positive")
        
        if build.retry_count < 0:
            errors.append("Retry count cannot be negative")
        
        if build.cache_ttl <= 0:
            errors.append("Cache TTL must be positive")
        
        return errors
    
    @staticmethod
    def validate_security_settings(security: SecuritySettings) -> List[str]:
        """Validate security settings"""
        errors = []
        
        # No specific validation needed for current settings
        # Could add validation for compliance frameworks, policy formats, etc.
        
        return errors
    
    @staticmethod
    def validate_global_settings(settings: GlobalSettings) -> List[str]:
        """Validate global settings"""
        errors = []
        
        # Validate ML providers
        for name, provider in settings.ml_providers.items():
            provider_errors = SettingsValidator.validate_ml_provider(provider)
            errors.extend([f"ML Provider {name}: {error}" for error in provider_errors])
        
        # Validate languages
        for name, language in settings.languages.items():
            language_errors = SettingsValidator.validate_language(language)
            errors.extend([f"Language {name}: {error}" for error in language_errors])
        
        # Validate build settings
        build_errors = SettingsValidator.validate_build_settings(settings.build)
        errors.extend([f"Build: {error}" for error in build_errors])
        
        # Validate security settings
        security_errors = SettingsValidator.validate_security_settings(settings.security)
        errors.extend([f"Security: {error}" for error in security_errors])
        
        return errors

def validate_settings(settings: Union[GlobalSettings, UserSettings, ProjectSettings]) -> List[str]:
    """Validate any settings object"""
    if isinstance(settings, GlobalSettings):
        return SettingsValidator.validate_global_settings(settings)
    elif isinstance(settings, UserSettings):
        # Basic user settings validation
        errors = []
        if not settings.name:
            errors.append("User name is required")
        return errors
    elif isinstance(settings, ProjectSettings):
        # Basic project settings validation
        errors = []
        if not settings.name:
            errors.append("Project name is required")
        if not settings.source_language:
            errors.append("Source language is required")
        if not settings.target_language:
            errors.append("Target language is required")
        return errors
    else:
        return ["Unknown settings type"]