"""
Core Configuration Manager for MyndraComposer
Handles loading, saving, and managing configuration files
"""

import json
import os
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml
import toml

class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"

class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"      # System-wide settings
    USER = "user"          # User-specific settings
    PROJECT = "project"    # Project-specific settings
    RUNTIME = "runtime"    # Runtime/session settings

@dataclass
class ConfigSection:
    """Configuration section"""
    name: str
    description: str
    settings: Dict[str, Any] = field(default_factory=dict)
    readonly: bool = False
    schema: Optional[Dict[str, Any]] = None

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    description: str
    ml_providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    build_settings: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)
    enterprise_settings: Dict[str, Any] = field(default_factory=dict)

class ConfigurationError(Exception):
    """Configuration-related exceptions"""
    pass

class ConfigManager:
    """Main configuration manager"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or Path.home() / ".myndra")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration hierarchy (order matters - later overrides earlier)
        self.config_hierarchy = [
            ConfigScope.GLOBAL,
            ConfigScope.USER, 
            ConfigScope.PROJECT,
            ConfigScope.RUNTIME
        ]
        
        # Loaded configurations by scope
        self.configurations: Dict[ConfigScope, Dict[str, Any]] = {}
        
        # Configuration file paths
        self.config_files = {
            ConfigScope.GLOBAL: self.config_dir / "global.yaml",
            ConfigScope.USER: self.config_dir / "user.yaml",
            ConfigScope.PROJECT: Path.cwd() / ".pomuse" / "config.yaml",
            ConfigScope.RUNTIME: None  # Runtime config not persisted
        }
        
        # Initialize configurations
        self._initialize_configurations()
    
    def _initialize_configurations(self):
        """Initialize configuration hierarchy"""
        # Load configurations in hierarchy order
        for scope in self.config_hierarchy:
            try:
                config = self._load_config_file(scope)
                self.configurations[scope] = config
            except Exception as e:
                self.logger.warning(f"Failed to load {scope.value} config: {e}")
                self.configurations[scope] = {}
        
        # Create default configurations if they don't exist
        self._create_default_configs()
    
    def _load_config_file(self, scope: ConfigScope) -> Dict[str, Any]:
        """Load configuration file for scope"""
        config_file = self.config_files.get(scope)
        
        if not config_file or not config_file.exists():
            return {}
        
        try:
            format_type = self._detect_format(config_file)
            
            with open(config_file, 'r') as f:
                if format_type == ConfigFormat.JSON:
                    return json.load(f)
                elif format_type == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif format_type == ConfigFormat.TOML:
                    return toml.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config format: {config_file}")
                    
        except Exception as e:
            raise ConfigurationError(f"Failed to load config {config_file}: {e}")
    
    def _detect_format(self, config_file: Path) -> ConfigFormat:
        """Detect configuration file format"""
        suffix = config_file.suffix.lower()
        
        if suffix == ".json":
            return ConfigFormat.JSON
        elif suffix in [".yaml", ".yml"]:
            return ConfigFormat.YAML
        elif suffix == ".toml":
            return ConfigFormat.TOML
        else:
            # Default to YAML
            return ConfigFormat.YAML
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist"""
        # Global configuration
        global_config = self.config_files[ConfigScope.GLOBAL]
        if not global_config.exists():
            default_global = self._get_default_global_config()
            self._save_config_file(ConfigScope.GLOBAL, default_global)
        
        # User configuration  
        user_config = self.config_files[ConfigScope.USER]
        if not user_config.exists():
            default_user = self._get_default_user_config()
            self._save_config_file(ConfigScope.USER, default_user)
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "ml_providers": {
                "default": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "codellama",
                    "timeout": 120
                },
                "openai": {
                    "api_key": "",
                    "model": "gpt-4",
                    "max_tokens": 4000
                },
                "anthropic": {
                    "api_key": "",
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 4000
                }
            },
            "languages": {
                "supported_source": ["python", "javascript", "typescript", "java", "cpp", "csharp", "rust", "go"],
                "supported_target": ["myndra", "rust", "go", "typescript", "python"],
                "default_target": "myndra"
            },
            "build": {
                "parallel_jobs": 4,
                "timeout": 600,
                "cleanup_artifacts": true
            },
            "security": {
                "scan_enabled": true,
                "fail_on_critical": true,
                "allowed_vulnerabilities": []
            }
        }
    
    def _get_default_user_config(self) -> Dict[str, Any]:
        """Get default user configuration"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "user": {
                "name": os.getenv("USER", "Unknown"),
                "email": "",
                "preferred_editor": "vscode"
            },
            "preferences": {
                "auto_save": true,
                "verbose_output": false,
                "color_output": true,
                "auto_update_check": true
            },
            "shortcuts": {
                "quick_analyze": "qa",
                "quick_generate": "qg",
                "show_status": "st"
            },
            "recent_projects": []
        }
    
    def _save_config_file(self, scope: ConfigScope, config: Dict[str, Any]):
        """Save configuration to file"""
        config_file = self.config_files.get(scope)
        
        if not config_file:
            return
        
        # Create directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            format_type = self._detect_format(config_file)
            
            with open(config_file, 'w') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, default=str)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False)
                elif format_type == ConfigFormat.TOML:
                    toml.dump(config, f)
                    
            # Update in-memory configuration
            self.configurations[scope] = config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save config {config_file}: {e}")
    
    def get(self, key: str, default: Any = None, scope: Optional[ConfigScope] = None) -> Any:
        """Get configuration value with hierarchy resolution"""
        if scope:
            # Get from specific scope
            config = self.configurations.get(scope, {})
            return self._get_nested_value(config, key, default)
        
        # Search through hierarchy (reverse order - runtime overrides global)
        for scope in reversed(self.config_hierarchy):
            config = self.configurations.get(scope, {})
            value = self._get_nested_value(config, key, None)
            if value is not None:
                return value
        
        return default
    
    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.USER, persist: bool = True):
        """Set configuration value"""
        if scope not in self.configurations:
            self.configurations[scope] = {}
        
        self._set_nested_value(self.configurations[scope], key, value)
        
        if persist and scope != ConfigScope.RUNTIME:
            self._save_config_file(scope, self.configurations[scope])
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def get_section(self, section: str, scope: Optional[ConfigScope] = None) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {}, scope)
    
    def set_section(self, section: str, values: Dict[str, Any], scope: ConfigScope = ConfigScope.USER):
        """Set entire configuration section"""
        self.set(section, values, scope)
    
    def list_keys(self, prefix: str = "", scope: Optional[ConfigScope] = None) -> List[str]:
        """List all configuration keys with optional prefix filter"""
        keys = set()
        
        scopes_to_check = [scope] if scope else self.config_hierarchy
        
        for check_scope in scopes_to_check:
            config = self.configurations.get(check_scope, {})
            keys.update(self._get_all_keys(config, prefix))
        
        return sorted(list(keys))
    
    def _get_all_keys(self, config: Dict[str, Any], prefix: str = "", parent_key: str = "") -> List[str]:
        """Recursively get all configuration keys"""
        keys = []
        
        for key, value in config.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            
            if not prefix or full_key.startswith(prefix):
                keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, prefix, full_key))
        
        return keys
    
    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate configuration against schema"""
        errors = []
        
        # Basic validation - check required ML provider settings
        ml_provider = self.get("ml_providers.default")
        if not ml_provider:
            errors.append("No default ML provider configured")
        else:
            provider_config = self.get(f"ml_providers.{ml_provider}")
            if not provider_config:
                errors.append(f"Configuration missing for ML provider: {ml_provider}")
        
        # Validate language settings
        supported_languages = self.get("languages.supported_source", [])
        if not supported_languages:
            errors.append("No supported source languages configured")
        
        # Validate build settings
        parallel_jobs = self.get("build.parallel_jobs", 1)
        if not isinstance(parallel_jobs, int) or parallel_jobs < 1:
            errors.append("build.parallel_jobs must be a positive integer")
        
        return errors
    
    def export_config(self, scope: Optional[ConfigScope] = None, format_type: ConfigFormat = ConfigFormat.YAML) -> str:
        """Export configuration as string"""
        if scope:
            config = self.configurations.get(scope, {})
        else:
            # Merge all configurations
            config = {}
            for check_scope in self.config_hierarchy:
                scope_config = self.configurations.get(check_scope, {})
                config = self._deep_merge(config, scope_config)
        
        if format_type == ConfigFormat.JSON:
            return json.dumps(config, indent=2, default=str)
        elif format_type == ConfigFormat.YAML:
            return yaml.dump(config, default_flow_style=False)
        elif format_type == ConfigFormat.TOML:
            return toml.dumps(config)
        else:
            raise ConfigurationError(f"Unsupported export format: {format_type}")
    
    def import_config(self, config_str: str, scope: ConfigScope, format_type: ConfigFormat, merge: bool = True):
        """Import configuration from string"""
        try:
            if format_type == ConfigFormat.JSON:
                new_config = json.loads(config_str)
            elif format_type == ConfigFormat.YAML:
                new_config = yaml.safe_load(config_str)
            elif format_type == ConfigFormat.TOML:
                new_config = toml.loads(config_str)
            else:
                raise ConfigurationError(f"Unsupported import format: {format_type}")
            
            if merge:
                existing_config = self.configurations.get(scope, {})
                merged_config = self._deep_merge(existing_config, new_config)
                self.configurations[scope] = merged_config
            else:
                self.configurations[scope] = new_config
            
            # Save to file
            if scope != ConfigScope.RUNTIME:
                self._save_config_file(scope, self.configurations[scope])
                
        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration: {e}")
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def reset_to_defaults(self, scope: ConfigScope):
        """Reset configuration scope to defaults"""
        if scope == ConfigScope.GLOBAL:
            default_config = self._get_default_global_config()
        elif scope == ConfigScope.USER:
            default_config = self._get_default_user_config()
        else:
            default_config = {}
        
        self.configurations[scope] = default_config
        
        if scope != ConfigScope.RUNTIME:
            self._save_config_file(scope, default_config)
    
    def create_backup(self, scope: Optional[ConfigScope] = None) -> str:
        """Create backup of configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        scopes_to_backup = [scope] if scope else [ConfigScope.GLOBAL, ConfigScope.USER]
        
        for backup_scope in scopes_to_backup:
            config_file = self.config_files.get(backup_scope)
            if config_file and config_file.exists():
                backup_file = backup_dir / f"{backup_scope.value}_{timestamp}.yaml"
                
                config = self.configurations.get(backup_scope, {})
                with open(backup_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
        
        return str(backup_dir)

# Convenience functions
def load_config(config_dir: Optional[str] = None) -> ConfigManager:
    """Load configuration manager"""
    return ConfigManager(config_dir)

def save_config(config_manager: ConfigManager):
    """Save all configurations"""
    for scope in [ConfigScope.GLOBAL, ConfigScope.USER, ConfigScope.PROJECT]:
        if scope in config_manager.configurations:
            config_manager._save_config_file(scope, config_manager.configurations[scope])

def get_config_value(key: str, default: Any = None, config_dir: Optional[str] = None) -> Any:
    """Get configuration value"""
    config_manager = ConfigManager(config_dir)
    return config_manager.get(key, default)

def set_config_value(key: str, value: Any, scope: ConfigScope = ConfigScope.USER, 
                    config_dir: Optional[str] = None):
    """Set configuration value"""
    config_manager = ConfigManager(config_dir)
    config_manager.set(key, value, scope)