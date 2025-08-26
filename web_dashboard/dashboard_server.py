"""
Dashboard Server for MyndraComposer
Main web server providing dashboard interface
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import threading

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import eventlet

from .api_routes import register_api_routes
from .websocket_handler import setup_websocket_routes

class DashboardServer:
    """Web dashboard server for MyndraComposer"""
    
    def __init__(self, pomuse_manager=None, config_manager=None, port=8080):
        self.pomuse_manager = pomuse_manager
        self.config_manager = config_manager
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / "templates"),
                        static_folder=str(Path(__file__).parent / "static"))
        
        self.app.config['SECRET_KEY'] = os.urandom(24)
        
        # Enable CORS for development
        CORS(self.app)
        
        # SocketIO setup
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
        
        # Active connections and rooms
        self.active_connections = {}
        self.project_rooms = {}
        
        # Server state
        self.is_running = False
        self.server_thread = None
        
        # Setup routes
        self._setup_routes()
        register_api_routes(self.app, self.pomuse_manager, self.config_manager)
        setup_websocket_routes(self.socketio, self.pomuse_manager)
    
    def _setup_routes(self):
        """Setup basic dashboard routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/projects')
        def projects():
            """Projects page"""
            return render_template('projects.html')
        
        @self.app.route('/project/<project_id>')
        def project_detail(project_id):
            """Project detail page"""
            return render_template('project_detail.html', project_id=project_id)
        
        @self.app.route('/plugins')
        def plugins():
            """Plugins management page"""
            return render_template('plugins.html')
        
        @self.app.route('/settings')
        def settings():
            """Settings page"""
            return render_template('settings.html')
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics and monitoring page"""
            return render_template('analytics.html')
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        @self.app.route('/api/status')
        def api_status():
            """API status endpoint"""
            status = {
                "server": "running",
                "connections": len(self.active_connections),
                "active_projects": len(self.project_rooms),
                "pomuse_manager": "connected" if self.pomuse_manager else "disconnected",
                "config_manager": "connected" if self.config_manager else "disconnected"
            }
            
            if self.pomuse_manager:
                try:
                    status["active_projects_count"] = len(self.pomuse_manager.list_projects())
                    status["plugin_count"] = len(self.pomuse_manager.plugin_manager.loaded_plugins)
                except Exception as e:
                    status["pomuse_error"] = str(e)
            
            return jsonify(status)
        
        # WebSocket event handlers
        @self.socketio.on('connect')
        def handle_connect():
            session_id = request.sid
            self.active_connections[session_id] = {
                "connected_at": datetime.now(),
                "user_id": None,
                "project_id": None
            }
            
            emit('connection_established', {
                "session_id": session_id,
                "server_time": datetime.now().isoformat()
            })
            
            self.logger.info(f"Client connected: {session_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            session_id = request.sid
            
            # Leave any project rooms
            if session_id in self.active_connections:
                project_id = self.active_connections[session_id].get("project_id")
                if project_id and project_id in self.project_rooms:
                    self.project_rooms[project_id].discard(session_id)
                    if not self.project_rooms[project_id]:
                        del self.project_rooms[project_id]
                
                del self.active_connections[session_id]
            
            self.logger.info(f"Client disconnected: {session_id}")
        
        @self.socketio.on('join_project')
        def handle_join_project(data):
            session_id = request.sid
            project_id = data.get('project_id')
            
            if project_id:
                join_room(project_id)
                
                if project_id not in self.project_rooms:
                    self.project_rooms[project_id] = set()
                self.project_rooms[project_id].add(session_id)
                
                if session_id in self.active_connections:
                    self.active_connections[session_id]["project_id"] = project_id
                
                emit('joined_project', {"project_id": project_id})
                
                # Notify other users in the project
                emit('user_joined', {
                    "session_id": session_id,
                    "project_id": project_id
                }, room=project_id, include_self=False)
        
        @self.socketio.on('leave_project')
        def handle_leave_project(data):
            session_id = request.sid
            project_id = data.get('project_id')
            
            if project_id:
                leave_room(project_id)
                
                if project_id in self.project_rooms:
                    self.project_rooms[project_id].discard(session_id)
                    if not self.project_rooms[project_id]:
                        del self.project_rooms[project_id]
                
                if session_id in self.active_connections:
                    self.active_connections[session_id]["project_id"] = None
                
                emit('left_project', {"project_id": project_id})
                
                # Notify other users in the project
                emit('user_left', {
                    "session_id": session_id,
                    "project_id": project_id
                }, room=project_id)
        
        @self.socketio.on('project_update')
        def handle_project_update(data):
            """Handle real-time project updates"""
            project_id = data.get('project_id')
            update_type = data.get('type')
            update_data = data.get('data', {})
            
            if project_id and project_id in self.project_rooms:
                emit('project_updated', {
                    "project_id": project_id,
                    "type": update_type,
                    "data": update_data,
                    "timestamp": datetime.now().isoformat()
                }, room=project_id, include_self=False)
    
    def start(self, host='localhost', threaded=True):
        """Start the dashboard server"""
        if self.is_running:
            self.logger.warning("Dashboard server is already running")
            return
        
        self.logger.info(f"Starting dashboard server on {host}:{self.port}")
        
        if threaded:
            self.server_thread = threading.Thread(
                target=self._run_server,
                args=(host,),
                daemon=True
            )
            self.server_thread.start()
        else:
            self._run_server(host)
    
    def _run_server(self, host):
        """Run the server"""
        try:
            self.is_running = True
            self.socketio.run(self.app, host=host, port=self.port, debug=False)
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the dashboard server"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping dashboard server")
        self.is_running = False
        
        # Notify all connected clients
        self.socketio.emit('server_shutdown', {
            "message": "Server is shutting down",
            "timestamp": datetime.now().isoformat()
        })
        
        # Close all connections
        self.active_connections.clear()
        self.project_rooms.clear()
    
    def broadcast_update(self, update_type: str, data: Dict[str, Any], 
                        project_id: Optional[str] = None):
        """Broadcast update to connected clients"""
        update_message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if project_id and project_id in self.project_rooms:
            # Send to specific project room
            self.socketio.emit('update', update_message, room=project_id)
        else:
            # Send to all connected clients
            self.socketio.emit('update', update_message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "active_projects": len(self.project_rooms),
            "server_uptime": self._get_uptime(),
            "connections": [
                {
                    "session_id": sid,
                    "connected_at": conn["connected_at"].isoformat(),
                    "project_id": conn.get("project_id")
                }
                for sid, conn in self.active_connections.items()
            ]
        }
    
    def _get_uptime(self) -> str:
        """Get server uptime"""
        # This would be calculated from server start time
        return "0:00:00"  # Placeholder

def create_dashboard_app(pomuse_manager=None, config_manager=None) -> Flask:
    """Create Flask app for dashboard"""
    server = DashboardServer(pomuse_manager, config_manager)
    return server.app

def start_dashboard_server(pomuse_manager=None, config_manager=None, 
                          host='localhost', port=8080, threaded=True) -> DashboardServer:
    """Start dashboard server"""
    server = DashboardServer(pomuse_manager, config_manager, port)
    server.start(host, threaded)
    return server