"""
WebSocket Handler for PomegranteMuse Dashboard
Handles real-time communication and collaboration features
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask_socketio import emit, join_room, leave_room, rooms

class WebSocketHandler:
    """WebSocket event handler for real-time features"""
    
    def __init__(self, socketio, pomuse_manager=None):
        self.socketio = socketio
        self.pomuse_manager = pomuse_manager
        self.logger = logging.getLogger(__name__)
        
        # Active sessions and collaboration state
        self.active_sessions = {}
        self.collaboration_sessions = {}
        self.file_locks = {}
        self.cursor_positions = {}
        
        self._register_events()
    
    def _register_events(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('project_file_open')
        def handle_file_open(data):
            """Handle file open event"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            file_path = data.get('file_path')
            
            if not project_id or not file_path:
                emit('error', {'message': 'Missing project_id or file_path'})
                return
            
            try:
                # Check if file is locked by another user
                lock_key = f"{project_id}:{file_path}"
                if lock_key in self.file_locks and self.file_locks[lock_key] != session_id:
                    emit('file_locked', {
                        'file_path': file_path,
                        'locked_by': self.file_locks[lock_key]
                    })
                    return
                
                # Lock file for this session
                self.file_locks[lock_key] = session_id
                
                # Join file room for real-time collaboration
                file_room = f"{project_id}:{file_path}"
                join_room(file_room)
                
                # Load file content if available
                file_content = self._load_file_content(project_id, file_path)
                
                emit('file_opened', {
                    'file_path': file_path,
                    'content': file_content,
                    'locked': True
                })
                
                # Notify other users in the project
                emit('user_opened_file', {
                    'session_id': session_id,
                    'file_path': file_path,
                    'timestamp': datetime.now().isoformat()
                }, room=project_id, include_self=False)
                
            except Exception as e:
                self.logger.error(f"Error opening file: {e}")
                emit('error', {'message': f'Failed to open file: {e}'})
        
        @self.socketio.on('project_file_close')
        def handle_file_close(data):
            """Handle file close event"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            file_path = data.get('file_path')
            
            if not project_id or not file_path:
                return
            
            try:
                # Release file lock
                lock_key = f"{project_id}:{file_path}"
                if lock_key in self.file_locks and self.file_locks[lock_key] == session_id:
                    del self.file_locks[lock_key]
                
                # Leave file room
                file_room = f"{project_id}:{file_path}"
                leave_room(file_room)
                
                # Clear cursor position
                cursor_key = f"{session_id}:{project_id}:{file_path}"
                if cursor_key in self.cursor_positions:
                    del self.cursor_positions[cursor_key]
                
                emit('file_closed', {'file_path': file_path})
                
                # Notify other users
                emit('user_closed_file', {
                    'session_id': session_id,
                    'file_path': file_path,
                    'timestamp': datetime.now().isoformat()
                }, room=project_id, include_self=False)
                
            except Exception as e:
                self.logger.error(f"Error closing file: {e}")
        
        @self.socketio.on('file_edit')
        def handle_file_edit(data):
            """Handle real-time file editing"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            file_path = data.get('file_path')
            changes = data.get('changes')
            
            if not all([project_id, file_path, changes]):
                return
            
            try:
                # Verify file lock
                lock_key = f"{project_id}:{file_path}"
                if lock_key not in self.file_locks or self.file_locks[lock_key] != session_id:
                    emit('error', {'message': 'File not locked by this session'})
                    return
                
                # Apply changes and broadcast to other users
                file_room = f"{project_id}:{file_path}"
                emit('file_changed', {
                    'file_path': file_path,
                    'changes': changes,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=file_room, include_self=False)
                
                # Auto-save if enabled
                if data.get('auto_save', False):
                    self._save_file_content(project_id, file_path, data.get('content'))
                
            except Exception as e:
                self.logger.error(f"Error handling file edit: {e}")
                emit('error', {'message': f'Failed to process edit: {e}'})
        
        @self.socketio.on('cursor_position')
        def handle_cursor_position(data):
            """Handle cursor position updates"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            file_path = data.get('file_path')
            position = data.get('position')
            
            if not all([project_id, file_path, position]):
                return
            
            # Store cursor position
            cursor_key = f"{session_id}:{project_id}:{file_path}"
            self.cursor_positions[cursor_key] = {
                'position': position,
                'timestamp': datetime.now()
            }
            
            # Broadcast to other users in the file
            file_room = f"{project_id}:{file_path}"
            emit('cursor_moved', {
                'session_id': session_id,
                'position': position,
                'timestamp': datetime.now().isoformat()
            }, room=file_room, include_self=False)
        
        @self.socketio.on('start_analysis')
        def handle_start_analysis(data):
            """Handle analysis start request"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            analysis_type = data.get('analysis_type', 'full')
            
            if not project_id:
                emit('error', {'message': 'Missing project_id'})
                return
            
            try:
                # Start analysis in background
                analysis_id = self._start_analysis_task(project_id, analysis_type, session_id)
                
                emit('analysis_started', {
                    'analysis_id': analysis_id,
                    'project_id': project_id,
                    'analysis_type': analysis_type
                })
                
                # Notify other users in the project
                emit('project_analysis_started', {
                    'analysis_id': analysis_id,
                    'analysis_type': analysis_type,
                    'started_by': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=project_id, include_self=False)
                
            except Exception as e:
                self.logger.error(f"Error starting analysis: {e}")
                emit('error', {'message': f'Failed to start analysis: {e}'})
        
        @self.socketio.on('start_generation')
        def handle_start_generation(data):
            """Handle code generation start request"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            generation_type = data.get('generation_type', 'full')
            target_language = data.get('target_language')
            
            if not project_id:
                emit('error', {'message': 'Missing project_id'})
                return
            
            try:
                # Start generation in background
                generation_id = self._start_generation_task(
                    project_id, generation_type, target_language, session_id
                )
                
                emit('generation_started', {
                    'generation_id': generation_id,
                    'project_id': project_id,
                    'generation_type': generation_type,
                    'target_language': target_language
                })
                
                # Notify other users in the project
                emit('project_generation_started', {
                    'generation_id': generation_id,
                    'generation_type': generation_type,
                    'target_language': target_language,
                    'started_by': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=project_id, include_self=False)
                
            except Exception as e:
                self.logger.error(f"Error starting generation: {e}")
                emit('error', {'message': f'Failed to start generation: {e}'})
        
        @self.socketio.on('chat_message')
        def handle_chat_message(data):
            """Handle chat messages in project"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            message = data.get('message')
            message_type = data.get('type', 'text')
            
            if not project_id or not message:
                return
            
            # Broadcast message to project room
            emit('chat_message', {
                'session_id': session_id,
                'message': message,
                'type': message_type,
                'timestamp': datetime.now().isoformat()
            }, room=project_id)
        
        @self.socketio.on('request_collaboration')
        def handle_collaboration_request(data):
            """Handle collaboration session request"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            target_session = data.get('target_session')
            
            if not all([project_id, target_session]):
                return
            
            # Send collaboration request to target session
            emit('collaboration_request', {
                'from_session': session_id,
                'project_id': project_id,
                'timestamp': datetime.now().isoformat()
            }, room=target_session)
        
        @self.socketio.on('respond_collaboration')
        def handle_collaboration_response(data):
            """Handle collaboration session response"""
            session_id = self._get_session_id()
            project_id = data.get('project_id')
            target_session = data.get('target_session')
            accepted = data.get('accepted', False)
            
            if not all([project_id, target_session]):
                return
            
            if accepted:
                # Create collaboration session
                collab_id = f"{project_id}:{session_id}:{target_session}"
                self.collaboration_sessions[collab_id] = {
                    'project_id': project_id,
                    'participants': [session_id, target_session],
                    'created_at': datetime.now(),
                    'status': 'active'
                }
                
                # Notify both participants
                for participant in [session_id, target_session]:
                    emit('collaboration_started', {
                        'collaboration_id': collab_id,
                        'project_id': project_id,
                        'participants': [session_id, target_session]
                    }, room=participant)
            else:
                # Notify requester of rejection
                emit('collaboration_rejected', {
                    'from_session': session_id,
                    'project_id': project_id
                }, room=target_session)
    
    def _get_session_id(self):
        """Get current session ID"""
        from flask import request
        return request.sid
    
    def _load_file_content(self, project_id: str, file_path: str) -> str:
        """Load file content for editing"""
        try:
            if self.pomuse_manager:
                project = self.pomuse_manager.get_project(project_id)
                if project:
                    # Load file content from project
                    full_path = project.get_file_path(file_path)
                    if full_path and full_path.exists():
                        return full_path.read_text()
            return ""
        except Exception as e:
            self.logger.error(f"Error loading file content: {e}")
            return ""
    
    def _save_file_content(self, project_id: str, file_path: str, content: str):
        """Save file content"""
        try:
            if self.pomuse_manager:
                project = self.pomuse_manager.get_project(project_id)
                if project:
                    project.save_file(file_path, content)
        except Exception as e:
            self.logger.error(f"Error saving file content: {e}")
    
    def _start_analysis_task(self, project_id: str, analysis_type: str, session_id: str) -> str:
        """Start analysis task in background"""
        import uuid
        analysis_id = str(uuid.uuid4())
        
        def analysis_task():
            try:
                if self.pomuse_manager:
                    result = self.pomuse_manager.analyze_project(project_id, analysis_type)
                    
                    # Send results to requester
                    self.socketio.emit('analysis_completed', {
                        'analysis_id': analysis_id,
                        'project_id': project_id,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)
                    
                    # Notify project room
                    self.socketio.emit('project_analysis_completed', {
                        'analysis_id': analysis_id,
                        'analysis_type': analysis_type,
                        'completed_by': session_id,
                        'timestamp': datetime.now().isoformat()
                    }, room=project_id)
                    
            except Exception as e:
                self.logger.error(f"Analysis task error: {e}")
                self.socketio.emit('analysis_failed', {
                    'analysis_id': analysis_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
        
        # Start task in background thread
        import threading
        thread = threading.Thread(target=analysis_task, daemon=True)
        thread.start()
        
        return analysis_id
    
    def _start_generation_task(self, project_id: str, generation_type: str, 
                              target_language: str, session_id: str) -> str:
        """Start code generation task in background"""
        import uuid
        generation_id = str(uuid.uuid4())
        
        def generation_task():
            try:
                if self.pomuse_manager:
                    result = self.pomuse_manager.generate_code(
                        project_id, generation_type, target_language
                    )
                    
                    # Send results to requester
                    self.socketio.emit('generation_completed', {
                        'generation_id': generation_id,
                        'project_id': project_id,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)
                    
                    # Notify project room
                    self.socketio.emit('project_generation_completed', {
                        'generation_id': generation_id,
                        'generation_type': generation_type,
                        'target_language': target_language,
                        'completed_by': session_id,
                        'timestamp': datetime.now().isoformat()
                    }, room=project_id)
                    
            except Exception as e:
                self.logger.error(f"Generation task error: {e}")
                self.socketio.emit('generation_failed', {
                    'generation_id': generation_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
        
        # Start task in background thread
        import threading
        thread = threading.Thread(target=generation_task, daemon=True)
        thread.start()
        
        return generation_id
    
    def broadcast_progress_update(self, task_id: str, progress: float, 
                                 message: str, session_id: str):
        """Broadcast task progress update"""
        self.socketio.emit('progress_update', {
            'task_id': task_id,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }, room=session_id)
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        return {
            'active_sessions': len(self.active_sessions),
            'collaboration_sessions': len(self.collaboration_sessions),
            'file_locks': len(self.file_locks),
            'cursor_positions': len(self.cursor_positions)
        }

def setup_websocket_routes(socketio, pomuse_manager=None):
    """Setup WebSocket routes"""
    handler = WebSocketHandler(socketio, pomuse_manager)
    return handler