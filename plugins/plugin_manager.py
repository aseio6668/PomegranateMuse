"""
Plugin Manager for MyndraComposer
Handles loading, managing, and executing plugins
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
from enum import Enum
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .plugin_interface import BasePluginInterface

class PluginType(Enum):
    """Plugin types"""
    LANGUAGE = "language"
    ML_PROVIDER = "ml_provider"
    ANALYSIS = "analysis"
    GENERATOR = "generator"
    TRANSFORM = "transform"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CUSTOM = "custom"

class PluginState(Enum):
    """Plugin states"""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"

@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    min_pomuse_version: str = "1.0.0"
    max_pomuse_version: str = ""
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    repository: str = ""
    license: str = ""

@dataclass
class LoadedPlugin:
    """Loaded plugin information"""
    metadata: PluginMetadata
    instance: BasePluginInterface
    state: PluginState
    load_time: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    usage_stats: Dict[str, int] = field(default_factory=dict)

class PluginError(Exception):
    """Plugin-related exceptions"""
    pass

class BasePlugin:
    """Base plugin class for easier plugin development"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"plugin.{name}")
        self.config = {}
        self.enabled = False
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled
        }
    
    def get_version(self) -> str:
        return self.version
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return []
    
    def on_load(self) -> bool:
        self.logger.info(f"Plugin {self.name} loaded")
        return True
    
    def on_unload(self) -> bool:
        self.logger.info(f"Plugin {self.name} unloaded")
        return True
    
    def on_enable(self) -> bool:
        self.enabled = True
        self.logger.info(f"Plugin {self.name} enabled")
        return True
    
    def on_disable(self) -> bool:
        self.enabled = False
        self.logger.info(f"Plugin {self.name} disabled")
        return True
    
    def on_config_changed(self, config: Dict[str, Any]) -> bool:
        self.config = config
        return True
    
    def send_message(self, target_plugin: str, message: Dict[str, Any]) -> bool:
        # Implementation would be provided by plugin manager
        return False
    
    def receive_message(self, source_plugin: str, message: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "received"}
    
    def subscribe_to_events(self, event_types: List[str]) -> bool:
        return True
    
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        return True

class PluginManager:
    """Main plugin manager"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Plugin storage
        self.loaded_plugins: Dict[str, LoadedPlugin] = {}
        self.plugin_directories: List[Path] = []
        self.plugin_registry: Dict[str, PluginMetadata] = {}
        
        # Plugin execution
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.plugin_lock = threading.RLock()
        
        # Event system
        self.event_subscribers: Dict[str, List[str]] = {}  # event_type -> plugin_names
        self.message_handlers: Dict[str, callable] = {}
        
        # Initialize default plugin directories
        self._setup_plugin_directories()
        self._load_plugin_registry()
    
    def _setup_plugin_directories(self):
        """Setup default plugin directories"""
        # Built-in plugins
        builtin_dir = Path(__file__).parent / "builtin"
        if builtin_dir.exists():
            self.plugin_directories.append(builtin_dir)
        
        # User plugins
        if self.config_manager:
            user_dir = Path(self.config_manager.config_dir) / "plugins"
        else:
            user_dir = Path.home() / ".myndra" / "plugins"
        user_dir.mkdir(parents=True, exist_ok=True)
        self.plugin_directories.append(user_dir)
        
        # System plugins
        system_dir = Path("/usr/local/share/myndra/plugins")
        if system_dir.exists():
            self.plugin_directories.append(system_dir)
    
    def _load_plugin_registry(self):
        """Load plugin registry from disk"""
        for plugin_dir in self.plugin_directories:
            self._scan_directory(plugin_dir)
    
    def _scan_directory(self, directory: Path):
        """Scan directory for plugins"""
        if not directory.exists():
            return
        
        for plugin_path in directory.iterdir():
            if plugin_path.is_dir():
                manifest_file = plugin_path / "plugin.json"
                if manifest_file.exists():
                    try:
                        metadata = self._load_plugin_metadata(manifest_file)
                        self.plugin_registry[metadata.name] = metadata
                    except Exception as e:
                        self.logger.warning(f"Failed to load plugin metadata from {manifest_file}: {e}")
    
    def _load_plugin_metadata(self, manifest_file: Path) -> PluginMetadata:
        """Load plugin metadata from manifest file"""
        with open(manifest_file) as f:
            data = json.load(f)
        
        # Convert plugin_type string to enum
        plugin_type = PluginType(data.get("plugin_type", "custom"))
        
        return PluginMetadata(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            plugin_type=plugin_type,
            entry_point=data["entry_point"],
            dependencies=data.get("dependencies", []),
            config_schema=data.get("config_schema", {}),
            capabilities=data.get("capabilities", []),
            min_pomuse_version=data.get("min_pomuse_version", "1.0.0"),
            max_pomuse_version=data.get("max_pomuse_version", ""),
            tags=data.get("tags", []),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            license=data.get("license", "")
        )
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin by name"""
        with self.plugin_lock:
            if plugin_name in self.loaded_plugins:
                self.logger.warning(f"Plugin {plugin_name} is already loaded")
                return True
            
            if plugin_name not in self.plugin_registry:
                raise PluginError(f"Plugin {plugin_name} not found in registry")
            
            metadata = self.plugin_registry[plugin_name]
            
            try:
                # Check dependencies
                self._check_dependencies(metadata)
                
                # Load plugin module
                plugin_instance = self._load_plugin_module(metadata)
                
                # Create loaded plugin record
                loaded_plugin = LoadedPlugin(
                    metadata=metadata,
                    instance=plugin_instance,
                    state=PluginState.LOADED,
                    load_time=datetime.now(),
                    config=config or {}
                )
                
                # Initialize plugin
                if config:
                    plugin_instance.on_config_changed(config)
                
                if plugin_instance.on_load():
                    self.loaded_plugins[plugin_name] = loaded_plugin
                    self.logger.info(f"Successfully loaded plugin: {plugin_name}")
                    return True
                else:
                    raise PluginError(f"Plugin {plugin_name} failed to initialize")
                    
            except Exception as e:
                error_msg = f"Failed to load plugin {plugin_name}: {e}"
                self.logger.error(error_msg)
                
                # Store error state
                if plugin_name in self.plugin_registry:
                    error_plugin = LoadedPlugin(
                        metadata=metadata,
                        instance=None,
                        state=PluginState.ERROR,
                        load_time=datetime.now(),
                        error_message=str(e)
                    )
                    self.loaded_plugins[plugin_name] = error_plugin
                
                raise PluginError(error_msg)
    
    def _check_dependencies(self, metadata: PluginMetadata):
        """Check if plugin dependencies are satisfied"""
        for dep in metadata.dependencies:
            if dep not in self.loaded_plugins:
                # Try to load dependency
                if dep in self.plugin_registry:
                    self.load_plugin(dep)
                else:
                    raise PluginError(f"Dependency {dep} not found for plugin {metadata.name}")
    
    def _load_plugin_module(self, metadata: PluginMetadata) -> BasePluginInterface:
        """Load plugin module and create instance"""
        # Find plugin directory
        plugin_dir = None
        for directory in self.plugin_directories:
            candidate = directory / metadata.name
            if candidate.exists():
                plugin_dir = candidate
                break
        
        if not plugin_dir:
            raise PluginError(f"Plugin directory not found for {metadata.name}")
        
        # Load module
        module_path = plugin_dir / f"{metadata.entry_point}.py"
        if not module_path.exists():
            raise PluginError(f"Entry point {metadata.entry_point}.py not found")
        
        spec = importlib.util.spec_from_file_location(metadata.name, module_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add plugin directory to path temporarily
        sys.path.insert(0, str(plugin_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(plugin_dir))
        
        # Get plugin class (assumes class name matches entry point)
        plugin_class_name = metadata.entry_point.split('.')[-1]
        if not hasattr(module, plugin_class_name):
            raise PluginError(f"Plugin class {plugin_class_name} not found in module")
        
        plugin_class = getattr(module, plugin_class_name)
        return plugin_class()
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        with self.plugin_lock:
            if plugin_name not in self.loaded_plugins:
                self.logger.warning(f"Plugin {plugin_name} is not loaded")
                return True
            
            loaded_plugin = self.loaded_plugins[plugin_name]
            
            try:
                # Disable first if enabled
                if loaded_plugin.state == PluginState.ENABLED:
                    self.disable_plugin(plugin_name)
                
                # Call unload hook
                if loaded_plugin.instance and loaded_plugin.instance.on_unload():
                    del self.loaded_plugins[plugin_name]
                    self.logger.info(f"Successfully unloaded plugin: {plugin_name}")
                    return True
                else:
                    raise PluginError(f"Plugin {plugin_name} failed to unload")
                    
            except Exception as e:
                error_msg = f"Failed to unload plugin {plugin_name}: {e}"
                self.logger.error(error_msg)
                raise PluginError(error_msg)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin"""
        with self.plugin_lock:
            if plugin_name not in self.loaded_plugins:
                raise PluginError(f"Plugin {plugin_name} is not loaded")
            
            loaded_plugin = self.loaded_plugins[plugin_name]
            
            if loaded_plugin.state == PluginState.ENABLED:
                return True
            
            if loaded_plugin.state == PluginState.ERROR:
                raise PluginError(f"Plugin {plugin_name} is in error state: {loaded_plugin.error_message}")
            
            try:
                if loaded_plugin.instance.on_enable():
                    loaded_plugin.state = PluginState.ENABLED
                    self.logger.info(f"Enabled plugin: {plugin_name}")
                    return True
                else:
                    raise PluginError(f"Plugin {plugin_name} failed to enable")
                    
            except Exception as e:
                error_msg = f"Failed to enable plugin {plugin_name}: {e}"
                self.logger.error(error_msg)
                loaded_plugin.state = PluginState.ERROR
                loaded_plugin.error_message = str(e)
                raise PluginError(error_msg)
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable an enabled plugin"""
        with self.plugin_lock:
            if plugin_name not in self.loaded_plugins:
                raise PluginError(f"Plugin {plugin_name} is not loaded")
            
            loaded_plugin = self.loaded_plugins[plugin_name]
            
            if loaded_plugin.state != PluginState.ENABLED:
                return True
            
            try:
                if loaded_plugin.instance.on_disable():
                    loaded_plugin.state = PluginState.DISABLED
                    self.logger.info(f"Disabled plugin: {plugin_name}")
                    return True
                else:
                    raise PluginError(f"Plugin {plugin_name} failed to disable")
                    
            except Exception as e:
                error_msg = f"Failed to disable plugin {plugin_name}: {e}"
                self.logger.error(error_msg)
                raise PluginError(error_msg)
    
    def get_plugin(self, plugin_name: str) -> Optional[LoadedPlugin]:
        """Get loaded plugin by name"""
        return self.loaded_plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[LoadedPlugin]:
        """Get all loaded plugins of specific type"""
        return [
            plugin for plugin in self.loaded_plugins.values()
            if plugin.metadata.plugin_type == plugin_type and plugin.state == PluginState.ENABLED
        ]
    
    def list_plugins(self, include_disabled: bool = False) -> List[str]:
        """List all loaded plugins"""
        if include_disabled:
            return list(self.loaded_plugins.keys())
        else:
            return [
                name for name, plugin in self.loaded_plugins.items()
                if plugin.state == PluginState.ENABLED
            ]
    
    def list_available_plugins(self) -> List[str]:
        """List all available plugins in registry"""
        return list(self.plugin_registry.keys())
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        if plugin_name in self.loaded_plugins:
            config = self.loaded_plugins[plugin_name].config
            self.unload_plugin(plugin_name)
            return self.load_plugin(plugin_name, config)
        else:
            return self.load_plugin(plugin_name)
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin"""
        with self.plugin_lock:
            if plugin_name not in self.loaded_plugins:
                raise PluginError(f"Plugin {plugin_name} is not loaded")
            
            loaded_plugin = self.loaded_plugins[plugin_name]
            
            # Validate configuration
            errors = loaded_plugin.instance.validate_config(config)
            if errors:
                raise PluginError(f"Invalid configuration for {plugin_name}: {', '.join(errors)}")
            
            # Apply configuration
            if loaded_plugin.instance.on_config_changed(config):
                loaded_plugin.config = config
                return True
            else:
                raise PluginError(f"Plugin {plugin_name} rejected configuration change")
    
    def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute method on plugin"""
        if plugin_name not in self.loaded_plugins:
            raise PluginError(f"Plugin {plugin_name} is not loaded")
        
        loaded_plugin = self.loaded_plugins[plugin_name]
        
        if loaded_plugin.state != PluginState.ENABLED:
            raise PluginError(f"Plugin {plugin_name} is not enabled")
        
        if not hasattr(loaded_plugin.instance, method_name):
            raise PluginError(f"Plugin {plugin_name} does not have method {method_name}")
        
        method = getattr(loaded_plugin.instance, method_name)
        
        try:
            # Update usage stats
            loaded_plugin.usage_stats[method_name] = loaded_plugin.usage_stats.get(method_name, 0) + 1
            
            return method(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing {method_name} on plugin {plugin_name}: {e}")
            raise
    
    def send_message(self, source_plugin: str, target_plugin: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message between plugins"""
        if target_plugin not in self.loaded_plugins:
            raise PluginError(f"Target plugin {target_plugin} is not loaded")
        
        target = self.loaded_plugins[target_plugin]
        if target.state != PluginState.ENABLED:
            raise PluginError(f"Target plugin {target_plugin} is not enabled")
        
        return target.instance.receive_message(source_plugin, message)
    
    def publish_event(self, event_type: str, data: Dict[str, Any], source_plugin: str = "system"):
        """Publish event to subscribed plugins"""
        if event_type in self.event_subscribers:
            for plugin_name in self.event_subscribers[event_type]:
                if plugin_name in self.loaded_plugins:
                    loaded_plugin = self.loaded_plugins[plugin_name]
                    if loaded_plugin.state == PluginState.ENABLED:
                        try:
                            loaded_plugin.instance.receive_message(source_plugin, {
                                "type": "event",
                                "event_type": event_type,
                                "data": data
                            })
                        except Exception as e:
                            self.logger.error(f"Error sending event to plugin {plugin_name}: {e}")
    
    def subscribe_to_events(self, plugin_name: str, event_types: List[str]) -> bool:
        """Subscribe plugin to events"""
        if plugin_name not in self.loaded_plugins:
            return False
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = []
            
            if plugin_name not in self.event_subscribers[event_type]:
                self.event_subscribers[event_type].append(plugin_name)
        
        return True
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        stats = {
            "total_plugins": len(self.plugin_registry),
            "loaded_plugins": len(self.loaded_plugins),
            "enabled_plugins": len([p for p in self.loaded_plugins.values() if p.state == PluginState.ENABLED]),
            "error_plugins": len([p for p in self.loaded_plugins.values() if p.state == PluginState.ERROR]),
            "plugin_types": {},
            "usage_stats": {}
        }
        
        # Count by type
        for plugin in self.loaded_plugins.values():
            plugin_type = plugin.metadata.plugin_type.value
            stats["plugin_types"][plugin_type] = stats["plugin_types"].get(plugin_type, 0) + 1
        
        # Usage stats
        for name, plugin in self.loaded_plugins.items():
            if plugin.usage_stats:
                stats["usage_stats"][name] = plugin.usage_stats
        
        return stats
    
    def shutdown(self):
        """Shutdown plugin manager"""
        self.logger.info("Shutting down plugin manager")
        
        # Unload all plugins
        plugin_names = list(self.loaded_plugins.keys())
        for plugin_name in plugin_names:
            try:
                self.unload_plugin(plugin_name)
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)

# Global plugin manager instance
_plugin_manager = None

def get_plugin_manager(config_manager=None) -> PluginManager:
    """Get global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(config_manager)
    return _plugin_manager

def load_plugins(plugin_names: List[str], config_manager=None) -> Dict[str, bool]:
    """Load multiple plugins"""
    manager = get_plugin_manager(config_manager)
    results = {}
    
    for plugin_name in plugin_names:
        try:
            results[plugin_name] = manager.load_plugin(plugin_name)
        except Exception as e:
            results[plugin_name] = False
            logging.error(f"Failed to load plugin {plugin_name}: {e}")
    
    return results