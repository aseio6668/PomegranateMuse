"""
Plugin Architecture for PomegranteMuse
Provides extensible plugin system for adding custom functionality
"""

from .plugin_manager import (
    PluginManager,
    PluginError,
    PluginType,
    PluginState,
    PluginMetadata,
    BasePlugin,
    load_plugins,
    get_plugin_manager
)

from .plugin_registry import (
    PluginRegistry,
    register_plugin,
    unregister_plugin,
    discover_plugins
)

from .plugin_interface import (
    ILanguagePlugin,
    IMLProviderPlugin,
    IAnalysisPlugin,
    IGeneratorPlugin,
    ITransformPlugin,
    IIntegrationPlugin
)

__all__ = [
    # Core plugin system
    "PluginManager",
    "PluginError",
    "PluginType",
    "PluginState", 
    "PluginMetadata",
    "BasePlugin",
    "load_plugins",
    "get_plugin_manager",
    
    # Plugin registry
    "PluginRegistry",
    "register_plugin",
    "unregister_plugin",
    "discover_plugins",
    
    # Plugin interfaces
    "ILanguagePlugin",
    "IMLProviderPlugin", 
    "IAnalysisPlugin",
    "IGeneratorPlugin",
    "ITransformPlugin",
    "IIntegrationPlugin"
]