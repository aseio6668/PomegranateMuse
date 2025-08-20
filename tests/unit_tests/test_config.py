"""
Configuration Management Unit Tests
Tests for configuration system, profiles, and settings
"""

import unittest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

class MockConfigManager:
    """Mock configuration manager for testing"""
    def __init__(self):
        self.configurations = {
            "global": {},
            "user": {},
            "project": {},
            "runtime": {}
        }
        self.config_hierarchy = ["global", "user", "project", "runtime"]
    
    def get(self, key, default=None, scope=None):
        if scope:
            config = self.configurations.get(scope, {})
            return self._get_nested_value(config, key, default)
        
        # Search through hierarchy
        for scope_name in reversed(self.config_hierarchy):
            config = self.configurations.get(scope_name, {})
            value = self._get_nested_value(config, key, None)
            if value is not None:
                return value
        return default
    
    def set(self, key, value, scope="user"):
        if scope not in self.configurations:
            self.configurations[scope] = {}
        self._set_nested_value(self.configurations[scope], key, value)
    
    def _get_nested_value(self, config, key, default):
        keys = key.split('.')
        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current
    
    def _set_nested_value(self, config, key, value):
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def list_keys(self, prefix="", scope=None):
        keys = set()
        scopes_to_check = [scope] if scope else self.config_hierarchy
        
        for check_scope in scopes_to_check:
            config = self.configurations.get(check_scope, {})
            keys.update(self._get_all_keys(config, prefix))
        
        return sorted(list(keys))
    
    def _get_all_keys(self, config, prefix="", parent_key=""):
        keys = []
        for key, value in config.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if not prefix or full_key.startswith(prefix):
                keys.append(full_key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, prefix, full_key))
        return keys
    
    def validate_config(self):
        return []
    
    def export_config(self, scope=None):
        if scope:
            return json.dumps(self.configurations.get(scope, {}))
        else:
            merged = {}
            for scope_name in self.config_hierarchy:
                config = self.configurations.get(scope_name, {})
                merged = self._deep_merge(merged, config)
            return json.dumps(merged)
    
    def _deep_merge(self, dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

class MockProfileManager:
    """Mock profile manager for testing"""
    def __init__(self):
        self.profiles = {}
        self.active_profile = None
    
    def create_profile(self, name, description, profile_type="custom"):
        profile = Mock()
        profile.name = name
        profile.description = description
        profile.profile_type = profile_type
        profile.configurations = {}
        self.profiles[name] = profile
        return profile
    
    def switch_profile(self, name):
        if name in self.profiles:
            self.active_profile = name
            return True
        return False
    
    def list_profiles(self):
        return list(self.profiles.values())
    
    def get_active_profile(self):
        return self.profiles.get(self.active_profile) if self.active_profile else None

class ConfigUnitTests(unittest.TestCase):
    """Unit tests for configuration management"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_manager = MockConfigManager()
        self.profile_manager = MockProfileManager()
        
        # Set up initial configuration
        self._setup_test_config()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _setup_test_config(self):
        """Set up test configuration data"""
        # Global configuration
        self.config_manager.configurations["global"] = {
            "ml_providers": {
                "default": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "codellama"
                }
            },
            "languages": {
                "supported_source": ["python", "javascript", "java"],
                "default_target": "pomegranate"
            }
        }
        
        # User configuration
        self.config_manager.configurations["user"] = {
            "preferences": {
                "auto_save": True,
                "verbose_output": False,
                "editor": "vscode"
            },
            "api_keys": {
                "openai": "test_key_123"
            }
        }
        
        # Project configuration
        self.config_manager.configurations["project"] = {
            "build": {
                "parallel_jobs": 4,
                "timeout": 600
            }
        }
    
    def test_config_get_simple(self):
        """Test simple configuration value retrieval"""
        # Test getting from specific scope
        default_provider = self.config_manager.get("ml_providers.default", scope="global")
        self.assertEqual(default_provider, "ollama")
        
        # Test getting with hierarchy resolution
        auto_save = self.config_manager.get("preferences.auto_save")
        self.assertTrue(auto_save)
        
        # Test getting non-existent key with default
        non_existent = self.config_manager.get("non.existent.key", "default_value")
        self.assertEqual(non_existent, "default_value")
    
    def test_config_get_nested(self):
        """Test nested configuration value retrieval"""
        # Test deep nested value
        base_url = self.config_manager.get("ml_providers.ollama.base_url")
        self.assertEqual(base_url, "http://localhost:11434")
        
        # Test nested object
        ollama_config = self.config_manager.get("ml_providers.ollama")
        self.assertIsInstance(ollama_config, dict)
        self.assertIn("base_url", ollama_config)
        self.assertIn("model", ollama_config)
    
    def test_config_set_simple(self):
        """Test simple configuration value setting"""
        # Set new value
        self.config_manager.set("test.new_key", "new_value")
        
        # Verify it was set
        value = self.config_manager.get("test.new_key")
        self.assertEqual(value, "new_value")
        
        # Set in specific scope
        self.config_manager.set("test.scoped_key", "scoped_value", scope="project")
        
        # Verify it's in the correct scope
        value = self.config_manager.get("test.scoped_key", scope="project")
        self.assertEqual(value, "scoped_value")
    
    def test_config_set_nested(self):
        """Test nested configuration value setting"""
        # Set nested value
        self.config_manager.set("new.nested.deep.key", "deep_value")
        
        # Verify nested structure was created
        value = self.config_manager.get("new.nested.deep.key")
        self.assertEqual(value, "deep_value")
        
        # Verify intermediate objects exist
        nested_obj = self.config_manager.get("new.nested")
        self.assertIsInstance(nested_obj, dict)
        self.assertIn("deep", nested_obj)
    
    def test_config_hierarchy(self):
        """Test configuration hierarchy resolution"""
        # Set same key in different scopes
        self.config_manager.set("test.hierarchy", "global_value", scope="global")
        self.config_manager.set("test.hierarchy", "user_value", scope="user")
        self.config_manager.set("test.hierarchy", "project_value", scope="project")
        
        # User scope should override global
        value = self.config_manager.get("test.hierarchy")
        self.assertEqual(value, "project_value")  # Project has highest priority
        
        # Test specific scope access
        global_value = self.config_manager.get("test.hierarchy", scope="global")
        self.assertEqual(global_value, "global_value")
        
        user_value = self.config_manager.get("test.hierarchy", scope="user")
        self.assertEqual(user_value, "user_value")
    
    def test_config_key_listing(self):
        """Test configuration key listing"""
        # List all keys
        all_keys = self.config_manager.list_keys()
        self.assertIn("ml_providers.default", all_keys)
        self.assertIn("preferences.auto_save", all_keys)
        self.assertIn("build.parallel_jobs", all_keys)
        
        # List keys with prefix
        ml_keys = self.config_manager.list_keys("ml_providers")
        self.assertTrue(any(key.startswith("ml_providers") for key in ml_keys))
        
        # List keys for specific scope
        user_keys = self.config_manager.list_keys(scope="user")
        self.assertIn("preferences.auto_save", user_keys)
        self.assertNotIn("ml_providers.default", user_keys)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test validation with valid config
        errors = self.config_manager.validate_config()
        self.assertEqual(len(errors), 0)
        
        # Mock validation with errors
        self.config_manager.validate_config = lambda: ["Invalid ML provider"]
        
        errors = self.config_manager.validate_config()
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid ML provider", errors)
    
    def test_config_export(self):
        """Test configuration export"""
        # Export specific scope
        global_export = self.config_manager.export_config(scope="global")
        global_data = json.loads(global_export)
        self.assertIn("ml_providers", global_data)
        self.assertIn("languages", global_data)
        
        # Export merged configuration
        full_export = self.config_manager.export_config()
        full_data = json.loads(full_export)
        self.assertIn("ml_providers", full_data)
        self.assertIn("preferences", full_data)
        self.assertIn("build", full_data)
    
    def test_profile_creation(self):
        """Test configuration profile creation"""
        profile = self.profile_manager.create_profile(
            "development",
            "Development environment profile",
            "development"
        )
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "development")
        self.assertEqual(profile.description, "Development environment profile")
        self.assertEqual(profile.profile_type, "development")
    
    def test_profile_switching(self):
        """Test configuration profile switching"""
        # Create profiles
        dev_profile = self.profile_manager.create_profile("dev", "Development")
        prod_profile = self.profile_manager.create_profile("prod", "Production")
        
        # Test switching
        result = self.profile_manager.switch_profile("dev")
        self.assertTrue(result)
        self.assertEqual(self.profile_manager.active_profile, "dev")
        
        active = self.profile_manager.get_active_profile()
        self.assertEqual(active.name, "dev")
        
        # Test switching to non-existent profile
        result = self.profile_manager.switch_profile("non_existent")
        self.assertFalse(result)
    
    def test_profile_listing(self):
        """Test configuration profile listing"""
        # Initially empty
        profiles = self.profile_manager.list_profiles()
        self.assertEqual(len(profiles), 0)
        
        # Create profiles
        self.profile_manager.create_profile("profile1", "Description 1")
        self.profile_manager.create_profile("profile2", "Description 2")
        self.profile_manager.create_profile("profile3", "Description 3")
        
        # Test listing
        profiles = self.profile_manager.list_profiles()
        self.assertEqual(len(profiles), 3)
        
        profile_names = [p.name for p in profiles]
        self.assertIn("profile1", profile_names)
        self.assertIn("profile2", profile_names)
        self.assertIn("profile3", profile_names)
    
    def test_config_file_formats(self):
        """Test different configuration file formats"""
        # Test JSON format
        json_config = {
            "setting1": "value1",
            "setting2": {"nested": "value2"}
        }
        json_file = self.test_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(json_config, f)
        
        # Test YAML format
        yaml_config = {
            "setting3": "value3",
            "setting4": {"nested": "value4"}
        }
        yaml_file = self.test_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # Verify files were created
        self.assertTrue(json_file.exists())
        self.assertTrue(yaml_file.exists())
        
        # Test reading back
        with open(json_file) as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json, json_config)
        
        with open(yaml_file) as f:
            loaded_yaml = yaml.safe_load(f)
        self.assertEqual(loaded_yaml, yaml_config)
    
    def test_config_merge_behavior(self):
        """Test configuration merging behavior"""
        # Set up overlapping configurations
        base_config = {
            "section1": {
                "key1": "base_value1",
                "key2": "base_value2"
            },
            "section2": {
                "key3": "base_value3"
            }
        }
        
        override_config = {
            "section1": {
                "key2": "override_value2",  # Override existing
                "key4": "override_value4"   # Add new
            },
            "section3": {
                "key5": "override_value5"   # New section
            }
        }
        
        # Test merge logic
        merged = self.config_manager._deep_merge(base_config, override_config)
        
        # Verify merge results
        self.assertEqual(merged["section1"]["key1"], "base_value1")  # Preserved
        self.assertEqual(merged["section1"]["key2"], "override_value2")  # Overridden
        self.assertEqual(merged["section1"]["key4"], "override_value4")  # Added
        self.assertEqual(merged["section2"]["key3"], "base_value3")  # Preserved
        self.assertEqual(merged["section3"]["key5"], "override_value5")  # New section
    
    def test_config_data_types(self):
        """Test handling of different data types in configuration"""
        # Test various data types
        test_values = {
            "string_value": "test_string",
            "integer_value": 42,
            "float_value": 3.14,
            "boolean_value": True,
            "list_value": [1, 2, 3, "four"],
            "dict_value": {"nested": "dict"},
            "null_value": None
        }
        
        for key, value in test_values.items():
            self.config_manager.set(f"types.{key}", value)
            retrieved_value = self.config_manager.get(f"types.{key}")
            self.assertEqual(retrieved_value, value)
    
    def test_config_environment_overrides(self):
        """Test environment variable configuration overrides"""
        # Mock environment variable override logic
        def get_with_env_override(key, default=None):
            env_key = f"POMUSE_{key.upper().replace('.', '_')}"
            # Mock environment check
            env_overrides = {
                "POMUSE_ML_PROVIDERS_DEFAULT": "openai",
                "POMUSE_PREFERENCES_VERBOSE_OUTPUT": "true"
            }
            
            if env_key in env_overrides:
                env_value = env_overrides[env_key]
                # Simple type conversion
                if env_value.lower() in ('true', 'false'):
                    return env_value.lower() == 'true'
                return env_value
            
            return self.config_manager.get(key, default)
        
        # Test environment override
        value = get_with_env_override("ml_providers.default")
        self.assertEqual(value, "openai")  # Overridden by env var
        
        verbose = get_with_env_override("preferences.verbose_output")
        self.assertTrue(verbose)  # Overridden by env var
        
        # Test non-overridden value
        auto_save = get_with_env_override("preferences.auto_save")
        self.assertTrue(auto_save)  # Original value