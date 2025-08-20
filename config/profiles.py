"""
Configuration Profiles for PomegranteMuse
Allows users to switch between different configuration sets
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .config_manager import ConfigManager, ConfigScope

class ProfileType(Enum):
    """Profile types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CUSTOM = "custom"

@dataclass
class ConfigProfile:
    """Configuration profile"""
    name: str
    description: str
    profile_type: ProfileType
    created_at: datetime
    last_used: Optional[datetime] = None
    configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_used, str):
            self.last_used = datetime.fromisoformat(self.last_used)

class ProfileManager:
    """Manages configuration profiles"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.profiles_dir = self.config_manager.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        self.active_profile_file = self.profiles_dir / "active_profile.json"
        self.logger = logging.getLogger(__name__)
        
        self.profiles: Dict[str, ConfigProfile] = {}
        self.active_profile: Optional[str] = None
        
        self._load_profiles()
        self._load_active_profile()
    
    def _load_profiles(self):
        """Load all profiles from disk"""
        for profile_file in self.profiles_dir.glob("*.json"):
            if profile_file.name == "active_profile.json":
                continue
                
            try:
                with open(profile_file) as f:
                    profile_data = json.load(f)
                
                profile = ConfigProfile(**profile_data)
                self.profiles[profile.name] = profile
                
            except Exception as e:
                self.logger.warning(f"Failed to load profile {profile_file}: {e}")
    
    def _load_active_profile(self):
        """Load active profile name"""
        if self.active_profile_file.exists():
            try:
                with open(self.active_profile_file) as f:
                    data = json.load(f)
                    self.active_profile = data.get("active_profile")
            except Exception as e:
                self.logger.warning(f"Failed to load active profile: {e}")
    
    def _save_profile(self, profile: ConfigProfile):
        """Save profile to disk"""
        profile_file = self.profiles_dir / f"{profile.name}.json"
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2, default=str)
                
        except Exception as e:
            raise Exception(f"Failed to save profile {profile.name}: {e}")
    
    def _save_active_profile(self):
        """Save active profile name"""
        try:
            with open(self.active_profile_file, 'w') as f:
                json.dump({"active_profile": self.active_profile}, f)
        except Exception as e:
            self.logger.warning(f"Failed to save active profile: {e}")
    
    def create_profile(self, name: str, description: str, 
                      profile_type: ProfileType = ProfileType.CUSTOM,
                      copy_current: bool = True, tags: List[str] = None) -> ConfigProfile:
        """Create new configuration profile"""
        if name in self.profiles:
            raise ValueError(f"Profile '{name}' already exists")
        
        profile = ConfigProfile(
            name=name,
            description=description,
            profile_type=profile_type,
            created_at=datetime.now(),
            tags=tags or []
        )
        
        if copy_current:
            # Copy current configuration
            for scope in [ConfigScope.GLOBAL, ConfigScope.USER, ConfigScope.PROJECT]:
                scope_config = self.config_manager.configurations.get(scope, {})
                if scope_config:
                    profile.configurations[scope.value] = scope_config.copy()
        else:
            # Create with defaults
            profile.configurations = self._get_default_profile_config(profile_type)
        
        self.profiles[name] = profile
        self._save_profile(profile)
        
        return profile
    
    def _get_default_profile_config(self, profile_type: ProfileType) -> Dict[str, Dict[str, Any]]:
        """Get default configuration for profile type"""
        if profile_type == ProfileType.DEVELOPMENT:
            return {
                "global": {
                    "ml_providers": {
                        "default": "ollama",
                        "ollama": {"base_url": "http://localhost:11434", "model": "codellama"}
                    },
                    "build": {"parallel_jobs": 2, "timeout": 300},
                    "security": {"fail_on_critical": False}
                },
                "user": {
                    "preferences": {"verbose_output": True, "auto_save": True}
                }
            }
        elif profile_type == ProfileType.PRODUCTION:
            return {
                "global": {
                    "ml_providers": {
                        "default": "openai",
                        "openai": {"model": "gpt-4", "max_tokens": 2000}
                    },
                    "build": {"parallel_jobs": 8, "timeout": 900},
                    "security": {"fail_on_critical": True, "fail_on_high": True}
                },
                "user": {
                    "preferences": {"verbose_output": False, "auto_save": True}
                }
            }
        else:
            return {}
    
    def switch_profile(self, name: str):
        """Switch to a different profile"""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found")
        
        profile = self.profiles[name]
        
        # Apply profile configurations
        for scope_name, scope_config in profile.configurations.items():
            try:
                scope = ConfigScope(scope_name)
                self.config_manager.configurations[scope] = scope_config.copy()
                
                # Save to file if not runtime scope
                if scope != ConfigScope.RUNTIME:
                    self.config_manager._save_config_file(scope, scope_config)
                    
            except ValueError:
                self.logger.warning(f"Unknown configuration scope: {scope_name}")
        
        # Update active profile
        self.active_profile = name
        profile.last_used = datetime.now()
        self._save_profile(profile)
        self._save_active_profile()
        
        self.logger.info(f"Switched to profile: {name}")
    
    def delete_profile(self, name: str, force: bool = False):
        """Delete a profile"""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found")
        
        if name == self.active_profile and not force:
            raise ValueError(f"Cannot delete active profile '{name}' without force=True")
        
        # Remove from memory
        del self.profiles[name]
        
        # Remove file
        profile_file = self.profiles_dir / f"{name}.json"
        if profile_file.exists():
            profile_file.unlink()
        
        # Clear active profile if deleting it
        if name == self.active_profile:
            self.active_profile = None
            self._save_active_profile()
        
        self.logger.info(f"Deleted profile: {name}")
    
    def list_profiles(self) -> List[ConfigProfile]:
        """List all profiles"""
        return list(self.profiles.values())
    
    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """Get specific profile"""
        return self.profiles.get(name)
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get currently active profile"""
        if self.active_profile:
            return self.profiles.get(self.active_profile)
        return None
    
    def export_profile(self, name: str, file_path: str):
        """Export profile to file"""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found")
        
        profile = self.profiles[name]
        
        with open(file_path, 'w') as f:
            json.dump(asdict(profile), f, indent=2, default=str)
    
    def import_profile(self, file_path: str, overwrite: bool = False) -> ConfigProfile:
        """Import profile from file"""
        with open(file_path) as f:
            profile_data = json.load(f)
        
        profile = ConfigProfile(**profile_data)
        
        if profile.name in self.profiles and not overwrite:
            raise ValueError(f"Profile '{profile.name}' already exists")
        
        self.profiles[profile.name] = profile
        self._save_profile(profile)
        
        return profile
    
    def update_profile(self, name: str, **kwargs):
        """Update profile properties"""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found")
        
        profile = self.profiles[name]
        
        # Update allowed fields
        allowed_fields = ["description", "tags"]
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(profile, field, value)
        
        self._save_profile(profile)
    
    def create_preset_profiles(self):
        """Create common preset profiles"""
        presets = [
            {
                "name": "development",
                "description": "Development environment with local tools",
                "profile_type": ProfileType.DEVELOPMENT,
                "tags": ["dev", "local"]
            },
            {
                "name": "staging", 
                "description": "Staging environment for testing",
                "profile_type": ProfileType.STAGING,
                "tags": ["staging", "test"]
            },
            {
                "name": "production",
                "description": "Production environment with strict settings",
                "profile_type": ProfileType.PRODUCTION,
                "tags": ["prod", "strict"]
            }
        ]
        
        for preset in presets:
            if preset["name"] not in self.profiles:
                self.create_profile(
                    name=preset["name"],
                    description=preset["description"],
                    profile_type=preset["profile_type"],
                    copy_current=False,
                    tags=preset["tags"]
                )
    
    def find_profiles(self, tags: List[str] = None, profile_type: ProfileType = None) -> List[ConfigProfile]:
        """Find profiles by tags or type"""
        results = []
        
        for profile in self.profiles.values():
            match = True
            
            if tags:
                if not any(tag in profile.tags for tag in tags):
                    match = False
            
            if profile_type and profile.profile_type != profile_type:
                match = False
            
            if match:
                results.append(profile)
        
        return results
    
    def backup_current_config(self, backup_name: str) -> ConfigProfile:
        """Backup current configuration as a profile"""
        backup_profile = self.create_profile(
            name=backup_name,
            description=f"Backup created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            profile_type=ProfileType.CUSTOM,
            copy_current=True,
            tags=["backup"]
        )
        
        return backup_profile

# Convenience functions
def create_profile(name: str, description: str, config_manager: ConfigManager,
                  profile_type: ProfileType = ProfileType.CUSTOM) -> ConfigProfile:
    """Create a new profile"""
    profile_manager = ProfileManager(config_manager)
    return profile_manager.create_profile(name, description, profile_type)

def switch_profile(name: str, config_manager: ConfigManager):
    """Switch to a profile"""
    profile_manager = ProfileManager(config_manager)
    profile_manager.switch_profile(name)

def list_profiles(config_manager: ConfigManager) -> List[ConfigProfile]:
    """List all profiles"""
    profile_manager = ProfileManager(config_manager)
    return profile_manager.list_profiles()