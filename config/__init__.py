"""
Configuration Management System for MyndraComposer
Provides centralized configuration management with environment-specific settings
"""

from .config_manager import (
    ConfigManager,
    ConfigurationError,
    ConfigSection,
    EnvironmentConfig,
    load_config,
    save_config,
    get_config_value,
    set_config_value
)

from .settings import (
    GlobalSettings,
    ProjectSettings,
    UserSettings,
    DefaultSettings,
    validate_settings
)

from .profiles import (
    ConfigProfile,
    ProfileManager,
    create_profile,
    switch_profile,
    list_profiles
)

__all__ = [
    # Core configuration
    "ConfigManager",
    "ConfigurationError", 
    "ConfigSection",
    "EnvironmentConfig",
    "load_config",
    "save_config",
    "get_config_value",
    "set_config_value",
    
    # Settings management
    "GlobalSettings",
    "ProjectSettings", 
    "UserSettings",
    "DefaultSettings",
    "validate_settings",
    
    # Profile management
    "ConfigProfile",
    "ProfileManager",
    "create_profile",
    "switch_profile", 
    "list_profiles"
]