"""
API Routes for MyndraComposer Web Dashboard
RESTful API endpoints for dashboard functionality
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from functools import wraps

def create_response(data=None, message=None, error=None, status_code=200):
    """Create standardized API response"""
    response = {
        "timestamp": datetime.now().isoformat(),
        "status": "success" if error is None else "error"
    }
    
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    if error:
        response["error"] = error
        
    return jsonify(response), status_code

def handle_exceptions(f):
    """Decorator to handle exceptions in API routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"API error in {f.__name__}: {e}")
            return create_response(error=str(e), status_code=500)
    return decorated_function

class APIRoutes:
    """API routes for dashboard"""
    
    def __init__(self, pomuse_manager=None, config_manager=None):
        self.pomuse_manager = pomuse_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Create blueprint
        self.bp = Blueprint('api', __name__, url_prefix='/api/v1')
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes"""
        
        # System routes
        self.bp.add_url_rule('/system/info', 'system_info', self.get_system_info, methods=['GET'])
        self.bp.add_url_rule('/system/stats', 'system_stats', self.get_system_stats, methods=['GET'])
        
        # Project routes
        self.bp.add_url_rule('/projects', 'list_projects', self.list_projects, methods=['GET'])
        self.bp.add_url_rule('/projects', 'create_project', self.create_project, methods=['POST'])
        self.bp.add_url_rule('/projects/<project_id>', 'get_project', self.get_project, methods=['GET'])
        self.bp.add_url_rule('/projects/<project_id>', 'update_project', self.update_project, methods=['PUT'])
        self.bp.add_url_rule('/projects/<project_id>', 'delete_project', self.delete_project, methods=['DELETE'])
        self.bp.add_url_rule('/projects/<project_id>/analyze', 'analyze_project', self.analyze_project, methods=['POST'])
        self.bp.add_url_rule('/projects/<project_id>/generate', 'generate_code', self.generate_code, methods=['POST'])
        self.bp.add_url_rule('/projects/<project_id>/status', 'project_status', self.get_project_status, methods=['GET'])
        
        # Plugin routes
        self.bp.add_url_rule('/plugins', 'list_plugins', self.list_plugins, methods=['GET'])
        self.bp.add_url_rule('/plugins/available', 'available_plugins', self.list_available_plugins, methods=['GET'])
        self.bp.add_url_rule('/plugins/<plugin_name>/load', 'load_plugin', self.load_plugin, methods=['POST'])
        self.bp.add_url_rule('/plugins/<plugin_name>/unload', 'unload_plugin', self.unload_plugin, methods=['POST'])
        self.bp.add_url_rule('/plugins/<plugin_name>/enable', 'enable_plugin', self.enable_plugin, methods=['POST'])
        self.bp.add_url_rule('/plugins/<plugin_name>/disable', 'disable_plugin', self.disable_plugin, methods=['POST'])
        self.bp.add_url_rule('/plugins/<plugin_name>/config', 'plugin_config', self.get_plugin_config, methods=['GET'])
        self.bp.add_url_rule('/plugins/<plugin_name>/config', 'update_plugin_config', self.update_plugin_config, methods=['PUT'])
        
        # Configuration routes
        self.bp.add_url_rule('/config', 'get_config', self.get_config, methods=['GET'])
        self.bp.add_url_rule('/config', 'update_config', self.update_config, methods=['PUT'])
        self.bp.add_url_rule('/config/profiles', 'list_profiles', self.list_profiles, methods=['GET'])
        self.bp.add_url_rule('/config/profiles', 'create_profile', self.create_profile, methods=['POST'])
        self.bp.add_url_rule('/config/profiles/<profile_name>', 'switch_profile', self.switch_profile, methods=['POST'])
        
        # Analytics routes
        self.bp.add_url_rule('/analytics/usage', 'usage_analytics', self.get_usage_analytics, methods=['GET'])
        self.bp.add_url_rule('/analytics/performance', 'performance_analytics', self.get_performance_analytics, methods=['GET'])
        self.bp.add_url_rule('/analytics/errors', 'error_analytics', self.get_error_analytics, methods=['GET'])
        
        # Collaboration routes
        self.bp.add_url_rule('/collaboration/sessions', 'active_sessions', self.get_active_sessions, methods=['GET'])
        self.bp.add_url_rule('/collaboration/users', 'active_users', self.get_active_users, methods=['GET'])
    
    @handle_exceptions
    def get_system_info(self):
        """Get system information"""
        info = {
            "version": "1.0.0",
            "platform": "MyndraComposer",
            "dashboard_version": "1.0.0",
            "python_version": "3.x",
            "server_time": datetime.now().isoformat()
        }
        
        if self.pomuse_manager:
            info["pomuse_status"] = "connected"
            info["plugin_count"] = len(self.pomuse_manager.plugin_manager.loaded_plugins) if hasattr(self.pomuse_manager, 'plugin_manager') else 0
        else:
            info["pomuse_status"] = "disconnected"
        
        return create_response(data=info)
    
    @handle_exceptions
    def get_system_stats(self):
        """Get system statistics"""
        stats = {
            "uptime": "00:00:00",  # Would be calculated from actual start time
            "memory_usage": "0 MB",
            "cpu_usage": "0%",
            "disk_usage": "0%"
        }
        
        if self.pomuse_manager and hasattr(self.pomuse_manager, 'plugin_manager'):
            plugin_stats = self.pomuse_manager.plugin_manager.get_plugin_stats()
            stats.update(plugin_stats)
        
        return create_response(data=stats)
    
    @handle_exceptions
    def list_projects(self):
        """List all projects"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        try:
            projects = self.pomuse_manager.list_projects()
            project_data = []
            
            for project in projects:
                project_info = {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "source_language": project.source_language,
                    "target_language": project.target_language,
                    "created_at": project.created_at.isoformat() if hasattr(project, 'created_at') else None,
                    "status": getattr(project, 'status', 'unknown')
                }
                project_data.append(project_info)
            
            return create_response(data=project_data)
            
        except Exception as e:
            return create_response(error=f"Failed to list projects: {e}", status_code=500)
    
    @handle_exceptions
    def create_project(self):
        """Create new project"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        data = request.get_json()
        if not data:
            return create_response(error="No data provided", status_code=400)
        
        required_fields = ['name', 'source_language', 'target_language']
        for field in required_fields:
            if field not in data:
                return create_response(error=f"Missing required field: {field}", status_code=400)
        
        try:
            project = self.pomuse_manager.create_project(
                name=data['name'],
                description=data.get('description', ''),
                source_language=data['source_language'],
                target_language=data['target_language'],
                source_path=data.get('source_path', ''),
                output_path=data.get('output_path', '')
            )
            
            return create_response(
                data={"project_id": project.id},
                message="Project created successfully",
                status_code=201
            )
            
        except Exception as e:
            return create_response(error=f"Failed to create project: {e}", status_code=500)
    
    @handle_exceptions
    def get_project(self, project_id):
        """Get project details"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        try:
            project = self.pomuse_manager.get_project(project_id)
            if not project:
                return create_response(error="Project not found", status_code=404)
            
            project_data = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "source_language": project.source_language,
                "target_language": project.target_language,
                "source_path": getattr(project, 'source_path', ''),
                "output_path": getattr(project, 'output_path', ''),
                "created_at": project.created_at.isoformat() if hasattr(project, 'created_at') else None,
                "updated_at": project.updated_at.isoformat() if hasattr(project, 'updated_at') else None,
                "status": getattr(project, 'status', 'unknown'),
                "progress": getattr(project, 'progress', 0)
            }
            
            return create_response(data=project_data)
            
        except Exception as e:
            return create_response(error=f"Failed to get project: {e}", status_code=500)
    
    @handle_exceptions
    def update_project(self, project_id):
        """Update project"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        data = request.get_json()
        if not data:
            return create_response(error="No data provided", status_code=400)
        
        try:
            success = self.pomuse_manager.update_project(project_id, **data)
            if success:
                return create_response(message="Project updated successfully")
            else:
                return create_response(error="Failed to update project", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to update project: {e}", status_code=500)
    
    @handle_exceptions
    def delete_project(self, project_id):
        """Delete project"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        try:
            success = self.pomuse_manager.delete_project(project_id)
            if success:
                return create_response(message="Project deleted successfully")
            else:
                return create_response(error="Failed to delete project", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to delete project: {e}", status_code=500)
    
    @handle_exceptions
    def analyze_project(self, project_id):
        """Analyze project code"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        data = request.get_json() or {}
        
        try:
            result = self.pomuse_manager.analyze_project(
                project_id,
                analysis_type=data.get('analysis_type', 'full'),
                options=data.get('options', {})
            )
            
            return create_response(data=result)
            
        except Exception as e:
            return create_response(error=f"Failed to analyze project: {e}", status_code=500)
    
    @handle_exceptions
    def generate_code(self, project_id):
        """Generate code for project"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        data = request.get_json() or {}
        
        try:
            result = self.pomuse_manager.generate_code(
                project_id,
                generation_type=data.get('generation_type', 'full'),
                options=data.get('options', {})
            )
            
            return create_response(data=result)
            
        except Exception as e:
            return create_response(error=f"Failed to generate code: {e}", status_code=500)
    
    @handle_exceptions
    def get_project_status(self, project_id):
        """Get project status"""
        if not self.pomuse_manager:
            return create_response(error="MyndraComposer manager not available", status_code=503)
        
        try:
            status = self.pomuse_manager.get_project_status(project_id)
            return create_response(data=status)
            
        except Exception as e:
            return create_response(error=f"Failed to get project status: {e}", status_code=500)
    
    @handle_exceptions
    def list_plugins(self):
        """List loaded plugins"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            plugins = []
            for name, plugin in self.pomuse_manager.plugin_manager.loaded_plugins.items():
                plugin_info = {
                    "name": name,
                    "version": plugin.metadata.version,
                    "description": plugin.metadata.description,
                    "author": plugin.metadata.author,
                    "type": plugin.metadata.plugin_type.value,
                    "state": plugin.state.value,
                    "load_time": plugin.load_time.isoformat(),
                    "error_message": plugin.error_message
                }
                plugins.append(plugin_info)
            
            return create_response(data=plugins)
            
        except Exception as e:
            return create_response(error=f"Failed to list plugins: {e}", status_code=500)
    
    @handle_exceptions
    def list_available_plugins(self):
        """List available plugins"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            available = self.pomuse_manager.plugin_manager.list_available_plugins()
            return create_response(data=available)
            
        except Exception as e:
            return create_response(error=f"Failed to list available plugins: {e}", status_code=500)
    
    @handle_exceptions
    def load_plugin(self, plugin_name):
        """Load plugin"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        data = request.get_json() or {}
        
        try:
            success = self.pomuse_manager.plugin_manager.load_plugin(
                plugin_name,
                config=data.get('config')
            )
            
            if success:
                return create_response(message=f"Plugin {plugin_name} loaded successfully")
            else:
                return create_response(error=f"Failed to load plugin {plugin_name}", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to load plugin {plugin_name}: {e}", status_code=500)
    
    @handle_exceptions
    def unload_plugin(self, plugin_name):
        """Unload plugin"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            success = self.pomuse_manager.plugin_manager.unload_plugin(plugin_name)
            
            if success:
                return create_response(message=f"Plugin {plugin_name} unloaded successfully")
            else:
                return create_response(error=f"Failed to unload plugin {plugin_name}", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to unload plugin {plugin_name}: {e}", status_code=500)
    
    @handle_exceptions
    def enable_plugin(self, plugin_name):
        """Enable plugin"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            success = self.pomuse_manager.plugin_manager.enable_plugin(plugin_name)
            
            if success:
                return create_response(message=f"Plugin {plugin_name} enabled successfully")
            else:
                return create_response(error=f"Failed to enable plugin {plugin_name}", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to enable plugin {plugin_name}: {e}", status_code=500)
    
    @handle_exceptions
    def disable_plugin(self, plugin_name):
        """Disable plugin"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            success = self.pomuse_manager.plugin_manager.disable_plugin(plugin_name)
            
            if success:
                return create_response(message=f"Plugin {plugin_name} disabled successfully")
            else:
                return create_response(error=f"Failed to disable plugin {plugin_name}", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to disable plugin {plugin_name}: {e}", status_code=500)
    
    @handle_exceptions
    def get_plugin_config(self, plugin_name):
        """Get plugin configuration"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        try:
            plugin = self.pomuse_manager.plugin_manager.get_plugin(plugin_name)
            if not plugin:
                return create_response(error="Plugin not found", status_code=404)
            
            return create_response(data=plugin.config)
            
        except Exception as e:
            return create_response(error=f"Failed to get plugin config: {e}", status_code=500)
    
    @handle_exceptions
    def update_plugin_config(self, plugin_name):
        """Update plugin configuration"""
        if not self.pomuse_manager or not hasattr(self.pomuse_manager, 'plugin_manager'):
            return create_response(error="Plugin manager not available", status_code=503)
        
        data = request.get_json()
        if not data:
            return create_response(error="No configuration data provided", status_code=400)
        
        try:
            success = self.pomuse_manager.plugin_manager.configure_plugin(plugin_name, data)
            
            if success:
                return create_response(message=f"Plugin {plugin_name} configuration updated successfully")
            else:
                return create_response(error=f"Failed to update plugin {plugin_name} configuration", status_code=500)
                
        except Exception as e:
            return create_response(error=f"Failed to update plugin config: {e}", status_code=500)
    
    @handle_exceptions
    def get_config(self):
        """Get current configuration"""
        if not self.config_manager:
            return create_response(error="Configuration manager not available", status_code=503)
        
        try:
            config = {
                "global": self.config_manager.get_section(""),
                "keys": self.config_manager.list_keys()
            }
            return create_response(data=config)
            
        except Exception as e:
            return create_response(error=f"Failed to get configuration: {e}", status_code=500)
    
    @handle_exceptions
    def update_config(self):
        """Update configuration"""
        if not self.config_manager:
            return create_response(error="Configuration manager not available", status_code=503)
        
        data = request.get_json()
        if not data:
            return create_response(error="No configuration data provided", status_code=400)
        
        try:
            for key, value in data.items():
                self.config_manager.set(key, value)
            
            return create_response(message="Configuration updated successfully")
            
        except Exception as e:
            return create_response(error=f"Failed to update configuration: {e}", status_code=500)
    
    @handle_exceptions
    def list_profiles(self):
        """List configuration profiles"""
        # This would integrate with the profile manager
        return create_response(data=[])
    
    @handle_exceptions
    def create_profile(self):
        """Create configuration profile"""
        data = request.get_json()
        # This would integrate with the profile manager
        return create_response(message="Profile created successfully")
    
    @handle_exceptions
    def switch_profile(self, profile_name):
        """Switch configuration profile"""
        # This would integrate with the profile manager
        return create_response(message=f"Switched to profile {profile_name}")
    
    @handle_exceptions
    def get_usage_analytics(self):
        """Get usage analytics"""
        # Placeholder for analytics data
        data = {
            "total_projects": 0,
            "total_analyses": 0,
            "total_generations": 0,
            "plugin_usage": {},
            "language_usage": {}
        }
        return create_response(data=data)
    
    @handle_exceptions
    def get_performance_analytics(self):
        """Get performance analytics"""
        # Placeholder for performance data
        data = {
            "average_analysis_time": "0s",
            "average_generation_time": "0s",
            "memory_usage_trend": [],
            "cpu_usage_trend": []
        }
        return create_response(data=data)
    
    @handle_exceptions
    def get_error_analytics(self):
        """Get error analytics"""
        # Placeholder for error data
        data = {
            "total_errors": 0,
            "error_types": {},
            "error_trend": [],
            "recent_errors": []
        }
        return create_response(data=data)
    
    @handle_exceptions
    def get_active_sessions(self):
        """Get active collaboration sessions"""
        # This would integrate with collaboration system
        return create_response(data=[])
    
    @handle_exceptions
    def get_active_users(self):
        """Get active users"""
        # This would integrate with collaboration system
        return create_response(data=[])

def register_api_routes(app, pomuse_manager=None, config_manager=None):
    """Register API routes with Flask app"""
    api_routes = APIRoutes(pomuse_manager, config_manager)
    app.register_blueprint(api_routes.bp)