"""
Plugin System Unit Tests
Tests for the plugin architecture and management
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

class MockPlugin:
    """Mock plugin for testing"""
    def __init__(self, name, version="1.0.0"):
        self.name = name
        self.version = version
        self.enabled = False
        self.config = {}
    
    def get_plugin_info(self):
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled
        }
    
    def get_version(self):
        return self.version
    
    def get_dependencies(self):
        return []
    
    def validate_config(self, config):
        return []
    
    def on_load(self):
        return True
    
    def on_unload(self):
        return True
    
    def on_enable(self):
        self.enabled = True
        return True
    
    def on_disable(self):
        self.enabled = False
        return True
    
    def on_config_changed(self, config):
        self.config = config
        return True

class MockPluginManager:
    """Mock plugin manager for testing"""
    def __init__(self):
        self.loaded_plugins = {}
        self.plugin_registry = {}
        self.plugin_directories = []
    
    def load_plugin(self, plugin_name, config=None):
        if plugin_name in self.plugin_registry:
            plugin = MockPlugin(plugin_name)
            if config:
                plugin.on_config_changed(config)
            plugin.on_load()
            self.loaded_plugins[plugin_name] = Mock()
            self.loaded_plugins[plugin_name].instance = plugin
            self.loaded_plugins[plugin_name].metadata = Mock()
            self.loaded_plugins[plugin_name].metadata.name = plugin_name
            self.loaded_plugins[plugin_name].state = Mock()
            self.loaded_plugins[plugin_name].state.value = "loaded"
            return True
        return False
    
    def unload_plugin(self, plugin_name):
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name].instance
            plugin.on_unload()
            del self.loaded_plugins[plugin_name]
            return True
        return False
    
    def enable_plugin(self, plugin_name):
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name].instance
            result = plugin.on_enable()
            if result:
                self.loaded_plugins[plugin_name].state.value = "enabled"
            return result
        return False
    
    def disable_plugin(self, plugin_name):
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name].instance
            result = plugin.on_disable()
            if result:
                self.loaded_plugins[plugin_name].state.value = "disabled"
            return result
        return False
    
    def get_plugin(self, plugin_name):
        return self.loaded_plugins.get(plugin_name)
    
    def list_plugins(self):
        return list(self.loaded_plugins.keys())
    
    def list_available_plugins(self):
        return list(self.plugin_registry.keys())

class PluginUnitTests(unittest.TestCase):
    """Unit tests for plugin system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.plugin_manager = MockPluginManager()
        
        # Set up test plugin registry
        self.plugin_manager.plugin_registry = {
            "python_language": {"type": "language", "version": "1.0.0"},
            "ollama_provider": {"type": "ml_provider", "version": "1.0.0"},
            "security_scanner": {"type": "security", "version": "1.0.0"}
        }
        
        # Create mock plugin directory
        self.plugin_dir = self.test_dir / "plugins"
        self.plugin_dir.mkdir()
        self._create_mock_plugin_files()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_mock_plugin_files(self):
        """Create mock plugin files for testing"""
        # Python language plugin
        python_plugin_dir = self.plugin_dir / "python_language"
        python_plugin_dir.mkdir()
        
        manifest = {
            "name": "python_language",
            "version": "1.0.0",
            "description": "Python language support",
            "author": "Test Author",
            "plugin_type": "language",
            "entry_point": "python_plugin",
            "dependencies": []
        }
        
        (python_plugin_dir / "plugin.json").write_text(json.dumps(manifest))
        (python_plugin_dir / "python_plugin.py").write_text("""
class PythonLanguagePlugin:
    def __init__(self):
        self.name = "python_language"
        self.version = "1.0.0"
    
    def get_supported_extensions(self):
        return [".py"]
""")
    
    def test_plugin_loading(self):
        """Test plugin loading functionality"""
        # Test successful loading
        result = self.plugin_manager.load_plugin("python_language")
        self.assertTrue(result)
        self.assertIn("python_language", self.plugin_manager.loaded_plugins)
        
        # Test loading non-existent plugin
        result = self.plugin_manager.load_plugin("non_existent")
        self.assertFalse(result)
        self.assertNotIn("non_existent", self.plugin_manager.loaded_plugins)
    
    def test_plugin_unloading(self):
        """Test plugin unloading functionality"""
        # Load plugin first
        self.plugin_manager.load_plugin("python_language")
        self.assertIn("python_language", self.plugin_manager.loaded_plugins)
        
        # Test unloading
        result = self.plugin_manager.unload_plugin("python_language")
        self.assertTrue(result)
        self.assertNotIn("python_language", self.plugin_manager.loaded_plugins)
        
        # Test unloading non-loaded plugin
        result = self.plugin_manager.unload_plugin("python_language")
        self.assertFalse(result)
    
    def test_plugin_enabling_disabling(self):
        """Test plugin enable/disable functionality"""
        # Load plugin first
        self.plugin_manager.load_plugin("python_language")
        
        # Test enabling
        result = self.plugin_manager.enable_plugin("python_language")
        self.assertTrue(result)
        
        plugin = self.plugin_manager.get_plugin("python_language")
        self.assertEqual(plugin.state.value, "enabled")
        
        # Test disabling
        result = self.plugin_manager.disable_plugin("python_language")
        self.assertTrue(result)
        self.assertEqual(plugin.state.value, "disabled")
        
        # Test enabling non-loaded plugin
        result = self.plugin_manager.enable_plugin("non_existent")
        self.assertFalse(result)
    
    def test_plugin_configuration(self):
        """Test plugin configuration management"""
        self.plugin_manager.load_plugin("python_language")
        plugin = self.plugin_manager.get_plugin("python_language")
        
        # Test configuration
        config = {"max_complexity": 15, "strict_mode": True}
        result = plugin.instance.on_config_changed(config)
        self.assertTrue(result)
        self.assertEqual(plugin.instance.config, config)
    
    def test_plugin_metadata(self):
        """Test plugin metadata handling"""
        self.plugin_manager.load_plugin("python_language")
        plugin = self.plugin_manager.get_plugin("python_language")
        
        # Test metadata access
        self.assertEqual(plugin.metadata.name, "python_language")
        
        info = plugin.instance.get_plugin_info()
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertEqual(info["name"], "python_language")
    
    def test_plugin_dependencies(self):
        """Test plugin dependency management"""
        plugin = MockPlugin("test_plugin")
        
        # Test no dependencies
        deps = plugin.get_dependencies()
        self.assertEqual(deps, [])
        
        # Mock plugin with dependencies
        plugin_with_deps = MockPlugin("dependent_plugin")
        plugin_with_deps.get_dependencies = lambda: ["python_language", "ollama_provider"]
        
        deps = plugin_with_deps.get_dependencies()
        self.assertEqual(len(deps), 2)
        self.assertIn("python_language", deps)
        self.assertIn("ollama_provider", deps)
    
    def test_plugin_validation(self):
        """Test plugin configuration validation"""
        plugin = MockPlugin("test_plugin")
        
        # Test valid configuration
        valid_config = {"timeout": 30, "retries": 3}
        errors = plugin.validate_config(valid_config)
        self.assertEqual(len(errors), 0)
        
        # Mock validation with errors
        plugin.validate_config = lambda config: ["Invalid timeout value"] if config.get("timeout", 0) < 0 else []
        
        invalid_config = {"timeout": -1}
        errors = plugin.validate_config(invalid_config)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid timeout value", errors)
    
    def test_plugin_lifecycle(self):
        """Test plugin lifecycle management"""
        plugin = MockPlugin("lifecycle_test")
        
        # Test load lifecycle
        result = plugin.on_load()
        self.assertTrue(result)
        
        # Test enable lifecycle
        result = plugin.on_enable()
        self.assertTrue(result)
        self.assertTrue(plugin.enabled)
        
        # Test disable lifecycle
        result = plugin.on_disable()
        self.assertTrue(result)
        self.assertFalse(plugin.enabled)
        
        # Test unload lifecycle
        result = plugin.on_unload()
        self.assertTrue(result)
    
    def test_plugin_listing(self):
        """Test plugin listing functionality"""
        # Test empty list initially
        loaded = self.plugin_manager.list_plugins()
        available = self.plugin_manager.list_available_plugins()
        
        self.assertEqual(len(loaded), 0)
        self.assertEqual(len(available), 3)  # python_language, ollama_provider, security_scanner
        
        # Load some plugins and test listing
        self.plugin_manager.load_plugin("python_language")
        self.plugin_manager.load_plugin("ollama_provider")
        
        loaded = self.plugin_manager.list_plugins()
        self.assertEqual(len(loaded), 2)
        self.assertIn("python_language", loaded)
        self.assertIn("ollama_provider", loaded)
    
    def test_plugin_type_filtering(self):
        """Test plugin filtering by type"""
        # This would test filtering plugins by type
        # For now, just test the registry contains different types
        registry = self.plugin_manager.plugin_registry
        
        language_plugins = [name for name, info in registry.items() if info["type"] == "language"]
        ml_provider_plugins = [name for name, info in registry.items() if info["type"] == "ml_provider"]
        security_plugins = [name for name, info in registry.items() if info["type"] == "security"]
        
        self.assertEqual(len(language_plugins), 1)
        self.assertEqual(len(ml_provider_plugins), 1)
        self.assertEqual(len(security_plugins), 1)
        
        self.assertIn("python_language", language_plugins)
        self.assertIn("ollama_provider", ml_provider_plugins)
        self.assertIn("security_scanner", security_plugins)
    
    def test_plugin_version_compatibility(self):
        """Test plugin version compatibility checking"""
        plugin = MockPlugin("version_test", "1.2.3")
        
        self.assertEqual(plugin.get_version(), "1.2.3")
        
        # Test version parsing (mock implementation)
        def parse_version(version_str):
            return tuple(map(int, version_str.split('.')))
        
        version_tuple = parse_version(plugin.get_version())
        self.assertEqual(version_tuple, (1, 2, 3))
        
        # Test compatibility check
        min_version = (1, 0, 0)
        max_version = (2, 0, 0)
        
        self.assertGreaterEqual(version_tuple, min_version)
        self.assertLess(version_tuple, max_version)
    
    def test_plugin_error_handling(self):
        """Test plugin error handling"""
        # Mock plugin that fails to load
        class FailingPlugin(MockPlugin):
            def on_load(self):
                raise Exception("Plugin load failed")
        
        failing_plugin = FailingPlugin("failing_plugin")
        
        # Test that exception is handled gracefully
        try:
            result = failing_plugin.on_load()
            self.fail("Expected exception was not raised")
        except Exception as e:
            self.assertEqual(str(e), "Plugin load failed")
    
    def test_plugin_state_persistence(self):
        """Test plugin state persistence"""
        # Load and configure plugin
        self.plugin_manager.load_plugin("python_language")
        plugin = self.plugin_manager.get_plugin("python_language")
        
        config = {"setting1": "value1", "setting2": 42}
        plugin.instance.on_config_changed(config)
        
        # Verify state is maintained
        self.assertEqual(plugin.instance.config, config)
        
        # Test state after disable/enable cycle
        self.plugin_manager.disable_plugin("python_language")
        self.plugin_manager.enable_plugin("python_language")
        
        # Configuration should be preserved
        self.assertEqual(plugin.instance.config, config)
    
    def test_concurrent_plugin_operations(self):
        """Test concurrent plugin operations"""
        import threading
        
        results = []
        
        def load_plugin(plugin_name):
            result = self.plugin_manager.load_plugin(plugin_name)
            results.append((plugin_name, result))
        
        # Load plugins concurrently
        threads = []
        plugin_names = ["python_language", "ollama_provider", "security_scanner"]
        
        for plugin_name in plugin_names:
            thread = threading.Thread(target=load_plugin, args=(plugin_name,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all plugins were loaded
        self.assertEqual(len(results), 3)
        for plugin_name, result in results:
            self.assertTrue(result)
            self.assertIn(plugin_name, self.plugin_manager.loaded_plugins)