"""
Plugin Registry for PomegranteMuse
Handles plugin discovery, registration, and metadata management
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import shutil
import zipfile
import tarfile

from .plugin_manager import PluginMetadata, PluginType

@dataclass
class RemotePluginInfo:
    """Remote plugin information"""
    name: str
    version: str
    description: str
    author: str
    download_url: str
    checksum: str
    size_bytes: int
    plugin_type: str
    tags: List[str]
    homepage: str = ""
    repository: str = ""
    
class PluginRepository:
    """Plugin repository for remote plugins"""
    
    def __init__(self, url: str, name: str = ""):
        self.url = url
        self.name = name or urlparse(url).netloc
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, RemotePluginInfo] = {}
        self._last_refresh = None
    
    def refresh(self) -> bool:
        """Refresh plugin list from repository"""
        try:
            response = requests.get(f"{self.url}/plugins.json", timeout=30)
            response.raise_for_status()
            
            plugins_data = response.json()
            self._cache = {}
            
            for plugin_data in plugins_data.get("plugins", []):
                plugin_info = RemotePluginInfo(**plugin_data)
                self._cache[plugin_info.name] = plugin_info
            
            self._last_refresh = True
            self.logger.info(f"Refreshed {len(self._cache)} plugins from repository {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to refresh repository {self.name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[RemotePluginInfo]:
        """Get plugin info by name"""
        return self._cache.get(name)
    
    def list_plugins(self) -> List[RemotePluginInfo]:
        """List all plugins in repository"""
        return list(self._cache.values())
    
    def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None) -> List[RemotePluginInfo]:
        """Search plugins by query and type"""
        results = []
        query_lower = query.lower()
        
        for plugin in self._cache.values():
            # Check if query matches name, description, or tags
            if (query_lower in plugin.name.lower() or 
                query_lower in plugin.description.lower() or
                any(query_lower in tag.lower() for tag in plugin.tags)):
                
                # Filter by type if specified
                if plugin_type is None or plugin.plugin_type == plugin_type.value:
                    results.append(plugin)
        
        return results

class PluginRegistry:
    """Main plugin registry"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Plugin storage
        self.local_plugins: Dict[str, PluginMetadata] = {}
        self.installed_plugins: Dict[str, PluginMetadata] = {}
        self.repositories: Dict[str, PluginRepository] = {}
        
        # Plugin directories
        if config_manager:
            self.plugin_dir = Path(config_manager.config_dir) / "plugins"
        else:
            self.plugin_dir = Path.home() / ".pomegrantemuse" / "plugins"
        
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup default repositories
        self._setup_default_repositories()
        
        # Discover local plugins
        self.discover_plugins()
    
    def _setup_default_repositories(self):
        """Setup default plugin repositories"""
        default_repos = [
            {
                "name": "official",
                "url": "https://plugins.pomegrantemuse.dev"
            },
            {
                "name": "community", 
                "url": "https://community-plugins.pomegrantemuse.dev"
            }
        ]
        
        for repo_info in default_repos:
            try:
                repo = PluginRepository(repo_info["url"], repo_info["name"])
                self.repositories[repo_info["name"]] = repo
            except Exception as e:
                self.logger.warning(f"Failed to setup repository {repo_info['name']}: {e}")
    
    def add_repository(self, name: str, url: str) -> bool:
        """Add a plugin repository"""
        try:
            repo = PluginRepository(url, name)
            if repo.refresh():
                self.repositories[name] = repo
                self.logger.info(f"Added repository: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add repository {name}: {e}")
            return False
    
    def remove_repository(self, name: str) -> bool:
        """Remove a plugin repository"""
        if name in self.repositories:
            del self.repositories[name]
            self.logger.info(f"Removed repository: {name}")
            return True
        return False
    
    def refresh_repositories(self) -> Dict[str, bool]:
        """Refresh all repositories"""
        results = {}
        for name, repo in self.repositories.items():
            results[name] = repo.refresh()
        return results
    
    def discover_plugins(self) -> int:
        """Discover plugins in local directories"""
        discovered = 0
        
        # Scan plugin directory
        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir():
                manifest_file = plugin_path / "plugin.json"
                if manifest_file.exists():
                    try:
                        metadata = self._load_plugin_metadata(manifest_file)
                        self.local_plugins[metadata.name] = metadata
                        self.installed_plugins[metadata.name] = metadata
                        discovered += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load plugin metadata from {manifest_file}: {e}")
        
        self.logger.info(f"Discovered {discovered} local plugins")
        return discovered
    
    def _load_plugin_metadata(self, manifest_file: Path) -> PluginMetadata:
        """Load plugin metadata from manifest file"""
        with open(manifest_file) as f:
            data = json.load(f)
        
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
    
    def register_plugin(self, metadata: PluginMetadata, plugin_path: Path) -> bool:
        """Register a local plugin"""
        try:
            # Create plugin directory
            target_dir = self.plugin_dir / metadata.name
            target_dir.mkdir(exist_ok=True)
            
            # Copy plugin files
            if plugin_path.is_file():
                # Single file plugin
                shutil.copy2(plugin_path, target_dir / f"{metadata.entry_point}.py")
            else:
                # Directory plugin
                shutil.copytree(plugin_path, target_dir, dirs_exist_ok=True)
            
            # Create manifest
            manifest_file = target_dir / "plugin.json"
            with open(manifest_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Register in memory
            self.local_plugins[metadata.name] = metadata
            self.installed_plugins[metadata.name] = metadata
            
            self.logger.info(f"Registered plugin: {metadata.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin {metadata.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a local plugin"""
        try:
            # Remove from memory
            if plugin_name in self.local_plugins:
                del self.local_plugins[plugin_name]
            if plugin_name in self.installed_plugins:
                del self.installed_plugins[plugin_name]
            
            # Remove directory
            plugin_dir = self.plugin_dir / plugin_name
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            
            self.logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def install_plugin(self, plugin_name: str, repository: str = "official", 
                      version: Optional[str] = None) -> bool:
        """Install plugin from repository"""
        if repository not in self.repositories:
            self.logger.error(f"Repository {repository} not found")
            return False
        
        repo = self.repositories[repository]
        plugin_info = repo.get_plugin(plugin_name)
        
        if not plugin_info:
            self.logger.error(f"Plugin {plugin_name} not found in repository {repository}")
            return False
        
        if version and plugin_info.version != version:
            self.logger.error(f"Version {version} not available for plugin {plugin_name}")
            return False
        
        try:
            # Download plugin
            temp_dir = Path(tempfile.mkdtemp())
            download_path = temp_dir / f"{plugin_name}.zip"
            
            self.logger.info(f"Downloading plugin {plugin_name} from {repository}")
            
            response = requests.get(plugin_info.download_url, timeout=300)
            response.raise_for_status()
            
            with open(download_path, 'wb') as f:
                f.write(response.content)
            
            # Verify checksum if provided
            if plugin_info.checksum:
                # Implementation would verify checksum here
                pass
            
            # Extract plugin
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()
            
            if download_path.suffix == '.zip':
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif download_path.suffix in ['.tar.gz', '.tgz']:
                with tarfile.open(download_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                raise ValueError(f"Unsupported archive format: {download_path.suffix}")
            
            # Find plugin manifest
            manifest_file = None
            for root, dirs, files in os.walk(extract_dir):
                if "plugin.json" in files:
                    manifest_file = Path(root) / "plugin.json"
                    break
            
            if not manifest_file:
                raise ValueError("Plugin manifest not found in downloaded archive")
            
            # Load metadata
            metadata = self._load_plugin_metadata(manifest_file)
            
            # Install plugin
            plugin_dir = manifest_file.parent
            success = self.register_plugin(metadata, plugin_dir)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            if success:
                self.logger.info(f"Successfully installed plugin {plugin_name}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to install plugin {plugin_name}: {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin"""
        if plugin_name not in self.installed_plugins:
            self.logger.warning(f"Plugin {plugin_name} is not installed")
            return True
        
        return self.unregister_plugin(plugin_name)
    
    def update_plugin(self, plugin_name: str, repository: str = "official") -> bool:
        """Update an installed plugin"""
        if plugin_name not in self.installed_plugins:
            self.logger.error(f"Plugin {plugin_name} is not installed")
            return False
        
        current_metadata = self.installed_plugins[plugin_name]
        
        # Check for updates
        if repository not in self.repositories:
            self.logger.error(f"Repository {repository} not found")
            return False
        
        repo = self.repositories[repository]
        plugin_info = repo.get_plugin(plugin_name)
        
        if not plugin_info:
            self.logger.error(f"Plugin {plugin_name} not found in repository {repository}")
            return False
        
        if plugin_info.version == current_metadata.version:
            self.logger.info(f"Plugin {plugin_name} is already up to date")
            return True
        
        # Backup current version
        backup_dir = self.plugin_dir / f"{plugin_name}_backup_{current_metadata.version}"
        current_dir = self.plugin_dir / plugin_name
        
        try:
            shutil.copytree(current_dir, backup_dir)
            
            # Uninstall current version
            self.uninstall_plugin(plugin_name)
            
            # Install new version
            if self.install_plugin(plugin_name, repository):
                # Remove backup
                shutil.rmtree(backup_dir)
                self.logger.info(f"Successfully updated plugin {plugin_name} to version {plugin_info.version}")
                return True
            else:
                # Restore backup
                shutil.rmtree(current_dir)
                shutil.move(backup_dir, current_dir)
                self.discover_plugins()  # Re-register restored plugin
                self.logger.error(f"Failed to update plugin {plugin_name}, restored previous version")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update plugin {plugin_name}: {e}")
            return False
    
    def list_installed_plugins(self) -> List[PluginMetadata]:
        """List installed plugins"""
        return list(self.installed_plugins.values())
    
    def list_available_plugins(self, repository: Optional[str] = None) -> List[RemotePluginInfo]:
        """List available plugins from repositories"""
        plugins = []
        
        repos_to_check = [self.repositories[repository]] if repository else self.repositories.values()
        
        for repo in repos_to_check:
            plugins.extend(repo.list_plugins())
        
        return plugins
    
    def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None,
                      repository: Optional[str] = None) -> List[RemotePluginInfo]:
        """Search for plugins"""
        results = []
        
        repos_to_check = [self.repositories[repository]] if repository else self.repositories.values()
        
        for repo in repos_to_check:
            results.extend(repo.search_plugins(query, plugin_type))
        
        return results
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin info by name"""
        return self.installed_plugins.get(plugin_name) or self.local_plugins.get(plugin_name)
    
    def check_updates(self) -> Dict[str, str]:
        """Check for plugin updates"""
        updates = {}
        
        for plugin_name, metadata in self.installed_plugins.items():
            for repo in self.repositories.values():
                plugin_info = repo.get_plugin(plugin_name)
                if plugin_info and plugin_info.version != metadata.version:
                    updates[plugin_name] = plugin_info.version
                    break
        
        return updates
    
    def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """Get plugin dependencies"""
        metadata = self.get_plugin_info(plugin_name)
        return metadata.dependencies if metadata else []
    
    def validate_dependencies(self, plugin_name: str) -> List[str]:
        """Validate plugin dependencies"""
        missing = []
        dependencies = self.get_plugin_dependencies(plugin_name)
        
        for dep in dependencies:
            if dep not in self.installed_plugins:
                missing.append(dep)
        
        return missing
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get plugin registry statistics"""
        total_available = sum(len(repo.list_plugins()) for repo in self.repositories.values())
        
        stats = {
            "installed_plugins": len(self.installed_plugins),
            "local_plugins": len(self.local_plugins),
            "repositories": len(self.repositories),
            "total_available": total_available,
            "plugin_types": {},
            "repositories_info": {}
        }
        
        # Count by type
        for metadata in self.installed_plugins.values():
            plugin_type = metadata.plugin_type.value
            stats["plugin_types"][plugin_type] = stats["plugin_types"].get(plugin_type, 0) + 1
        
        # Repository info
        for name, repo in self.repositories.items():
            stats["repositories_info"][name] = {
                "url": repo.url,
                "plugin_count": len(repo.list_plugins())
            }
        
        return stats

# Global registry instance
_plugin_registry = None

def get_plugin_registry(config_manager=None) -> PluginRegistry:
    """Get global plugin registry instance"""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry(config_manager)
    return _plugin_registry

def register_plugin(metadata: PluginMetadata, plugin_path: Path, config_manager=None) -> bool:
    """Register a plugin"""
    registry = get_plugin_registry(config_manager)
    return registry.register_plugin(metadata, plugin_path)

def unregister_plugin(plugin_name: str, config_manager=None) -> bool:
    """Unregister a plugin"""
    registry = get_plugin_registry(config_manager)
    return registry.unregister_plugin(plugin_name)

def discover_plugins(config_manager=None) -> int:
    """Discover plugins in local directories"""
    registry = get_plugin_registry(config_manager)
    return registry.discover_plugins()