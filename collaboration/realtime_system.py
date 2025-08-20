"""
Real-Time Collaboration System for Universal Code Modernization Platform
Enables multiple team members to work together on code analysis and migration projects
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import websockets
import aiohttp
from aiohttp import web, WSMsgType
import aiofiles
import sqlite3
import redis.asyncio as redis


class EventType(Enum):
    """Types of collaboration events"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    CODE_GENERATED = "code_generated"
    COMMENT_ADDED = "comment_added"
    FILE_LOCKED = "file_locked"
    FILE_UNLOCKED = "file_unlocked"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    PROJECT_UPDATED = "project_updated"
    PERMISSION_CHANGED = "permission_changed"
    CHAT_MESSAGE = "chat_message"
    STATUS_UPDATE = "status_update"


class UserRole(Enum):
    """User roles in collaboration"""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    ADMIN = "admin"
    OWNER = "owner"


class ProjectStatus(Enum):
    """Project status states"""
    CREATED = "created"
    ANALYZING = "analyzing"
    READY_FOR_GENERATION = "ready_for_generation"
    GENERATING = "generating"
    READY_FOR_REVIEW = "ready_for_review"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class User:
    """Represents a user in the collaboration system"""
    id: str
    name: str
    email: str
    role: UserRole
    avatar_url: Optional[str] = None
    last_seen: datetime = field(default_factory=datetime.now)
    is_online: bool = False
    current_project: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """Represents a collaborative project"""
    id: str
    name: str
    description: str
    owner_id: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    source_language: str
    target_languages: List[str]
    participants: Dict[str, UserRole] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    analysis_results: Optional[Dict[str, Any]] = None
    generated_code: Dict[str, str] = field(default_factory=dict)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    locked_files: Dict[str, str] = field(default_factory=dict)  # file_path -> user_id
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationEvent:
    """Represents a real-time collaboration event"""
    id: str
    project_id: str
    user_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


@dataclass
class CursorPosition:
    """Represents a user's cursor position"""
    user_id: str
    file_path: str
    line: int
    column: int
    timestamp: datetime


@dataclass
class Comment:
    """Represents a comment on code or analysis"""
    id: str
    project_id: str
    user_id: str
    file_path: Optional[str]
    line_number: Optional[int]
    content: str
    timestamp: datetime
    replies: List[str] = field(default_factory=list)
    resolved: bool = False
    tags: List[str] = field(default_factory=list)


class CollaborationDatabase:
    """Database interface for collaboration data"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        """Initialize database schema"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        await self._create_tables()
    
    async def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                avatar_url TEXT,
                last_seen TIMESTAMP,
                is_online BOOLEAN DEFAULT FALSE,
                current_project TEXT,
                preferences TEXT
            )
        ''')
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                source_language TEXT,
                target_languages TEXT,
                participants TEXT,
                files TEXT,
                analysis_results TEXT,
                generated_code TEXT,
                locked_files TEXT,
                settings TEXT,
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Comments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                file_path TEXT,
                line_number INTEGER,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                replies TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                tags TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.connection.commit()
    
    async def save_user(self, user: User):
        """Save user to database"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (id, name, email, role, avatar_url, last_seen, is_online, current_project, preferences)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user.id, user.name, user.email, user.role.value, user.avatar_url,
            user.last_seen, user.is_online, user.current_project, json.dumps(user.preferences)
        ))
        self.connection.commit()
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            return User(
                id=row['id'],
                name=row['name'],
                email=row['email'],
                role=UserRole(row['role']),
                avatar_url=row['avatar_url'],
                last_seen=datetime.fromisoformat(row['last_seen']) if row['last_seen'] else datetime.now(),
                is_online=bool(row['is_online']),
                current_project=row['current_project'],
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )
        return None
    
    async def save_project(self, project: Project):
        """Save project to database"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO projects 
            (id, name, description, owner_id, status, created_at, updated_at, 
             source_language, target_languages, participants, files, analysis_results, 
             generated_code, locked_files, settings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            project.id, project.name, project.description, project.owner_id,
            project.status.value, project.created_at, project.updated_at,
            project.source_language, json.dumps(project.target_languages),
            json.dumps({k: v.value for k, v in project.participants.items()}),
            json.dumps(project.files), json.dumps(project.analysis_results),
            json.dumps(project.generated_code), json.dumps(project.locked_files),
            json.dumps(project.settings)
        ))
        self.connection.commit()
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        
        if row:
            participants = json.loads(row['participants']) if row['participants'] else {}
            return Project(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                owner_id=row['owner_id'],
                status=ProjectStatus(row['status']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                source_language=row['source_language'],
                target_languages=json.loads(row['target_languages']) if row['target_languages'] else [],
                participants={k: UserRole(v) for k, v in participants.items()},
                files=json.loads(row['files']) if row['files'] else [],
                analysis_results=json.loads(row['analysis_results']) if row['analysis_results'] else None,
                generated_code=json.loads(row['generated_code']) if row['generated_code'] else {},
                locked_files=json.loads(row['locked_files']) if row['locked_files'] else {},
                settings=json.loads(row['settings']) if row['settings'] else {}
            )
        return None
    
    async def save_event(self, event: CollaborationEvent):
        """Save event to database"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO events (id, project_id, user_id, event_type, timestamp, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event.id, event.project_id, event.user_id, event.event_type.value,
            event.timestamp, json.dumps(event.data)
        ))
        self.connection.commit()
    
    async def get_project_events(self, project_id: str, limit: int = 100) -> List[CollaborationEvent]:
        """Get recent events for a project"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM events WHERE project_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (project_id, limit))
        
        events = []
        for row in cursor.fetchall():
            events.append(CollaborationEvent(
                id=row['id'],
                project_id=row['project_id'],
                user_id=row['user_id'],
                event_type=EventType(row['event_type']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                data=json.loads(row['data']) if row['data'] else {}
            ))
        
        return events


class CollaborationManager:
    """Manages real-time collaboration features"""
    
    def __init__(self, db_path: str = ":memory:", redis_url: str = "redis://localhost:6379"):
        self.db = CollaborationDatabase(db_path)
        self.redis_url = redis_url
        self.redis_client = None
        
        # In-memory state for real-time features
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.project_subscribers: Dict[str, Set[str]] = {}  # project_id -> set of user_ids
        self.user_cursors: Dict[str, CursorPosition] = {}  # user_id -> cursor position
        self.file_locks: Dict[str, str] = {}  # file_path -> user_id
        
        # Event callbacks
        self.event_callbacks: Dict[EventType, List[Callable]] = {}
    
    async def initialize(self):
        """Initialize collaboration manager"""
        await self.db.initialize()
        
        # Initialize Redis for pub/sub
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}. Using in-memory pub/sub.")
            self.redis_client = None
    
    async def create_user(self, name: str, email: str, role: UserRole = UserRole.CONTRIBUTOR) -> User:
        """Create a new user"""
        user = User(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            role=role
        )
        await self.db.save_user(user)
        return user
    
    async def create_project(self, name: str, description: str, owner_id: str, 
                           source_language: str, target_languages: List[str]) -> Project:
        """Create a new collaboration project"""
        project = Project(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            owner_id=owner_id,
            status=ProjectStatus.CREATED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_language=source_language,
            target_languages=target_languages,
            participants={owner_id: UserRole.OWNER}
        )
        
        await self.db.save_project(project)
        
        # Emit project created event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project.id,
            user_id=owner_id,
            event_type=EventType.PROJECT_UPDATED,
            timestamp=datetime.now(),
            data={"action": "created", "project_name": name}
        ))
        
        return project
    
    async def join_project(self, user_id: str, project_id: str, role: UserRole = UserRole.CONTRIBUTOR):
        """Add user to project"""
        project = await self.db.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        project.participants[user_id] = role
        project.updated_at = datetime.now()
        await self.db.save_project(project)
        
        # Update user's current project
        user = await self.db.get_user(user_id)
        if user:
            user.current_project = project_id
            await self.db.save_user(user)
        
        # Emit user joined event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.USER_JOINED,
            timestamp=datetime.now(),
            data={"role": role.value, "user_name": user.name if user else "Unknown"}
        ))
    
    async def leave_project(self, user_id: str, project_id: str):
        """Remove user from project"""
        project = await self.db.get_project(project_id)
        if not project:
            return
        
        if user_id in project.participants:
            del project.participants[user_id]
            project.updated_at = datetime.now()
            await self.db.save_project(project)
        
        # Update user's current project
        user = await self.db.get_user(user_id)
        if user and user.current_project == project_id:
            user.current_project = None
            await self.db.save_user(user)
        
        # Remove from subscribers
        if project_id in self.project_subscribers:
            self.project_subscribers[project_id].discard(user_id)
        
        # Emit user left event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.USER_LEFT,
            timestamp=datetime.now(),
            data={"user_name": user.name if user else "Unknown"}
        ))
    
    async def update_project_status(self, project_id: str, status: ProjectStatus, user_id: str):
        """Update project status"""
        project = await self.db.get_project(project_id)
        if not project:
            return
        
        old_status = project.status
        project.status = status
        project.updated_at = datetime.now()
        await self.db.save_project(project)
        
        # Emit status update event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.STATUS_UPDATE,
            timestamp=datetime.now(),
            data={
                "old_status": old_status.value,
                "new_status": status.value
            }
        ))
    
    async def lock_file(self, project_id: str, file_path: str, user_id: str) -> bool:
        """Lock a file for editing"""
        # Check if file is already locked
        if file_path in self.file_locks and self.file_locks[file_path] != user_id:
            return False
        
        self.file_locks[file_path] = user_id
        
        # Update project
        project = await self.db.get_project(project_id)
        if project:
            project.locked_files[file_path] = user_id
            await self.db.save_project(project)
        
        # Emit file locked event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.FILE_LOCKED,
            timestamp=datetime.now(),
            data={"file_path": file_path}
        ))
        
        return True
    
    async def unlock_file(self, project_id: str, file_path: str, user_id: str):
        """Unlock a file"""
        if file_path in self.file_locks and self.file_locks[file_path] == user_id:
            del self.file_locks[file_path]
            
            # Update project
            project = await self.db.get_project(project_id)
            if project and file_path in project.locked_files:
                del project.locked_files[file_path]
                await self.db.save_project(project)
            
            # Emit file unlocked event
            await self.emit_event(CollaborationEvent(
                id=str(uuid.uuid4()),
                project_id=project_id,
                user_id=user_id,
                event_type=EventType.FILE_UNLOCKED,
                timestamp=datetime.now(),
                data={"file_path": file_path}
            ))
    
    async def update_cursor_position(self, user_id: str, project_id: str, file_path: str, 
                                   line: int, column: int):
        """Update user's cursor position"""
        cursor = CursorPosition(
            user_id=user_id,
            file_path=file_path,
            line=line,
            column=column,
            timestamp=datetime.now()
        )
        
        self.user_cursors[user_id] = cursor
        
        # Emit cursor moved event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.CURSOR_MOVED,
            timestamp=datetime.now(),
            data={
                "file_path": file_path,
                "line": line,
                "column": column
            }
        ))
    
    async def add_comment(self, project_id: str, user_id: str, content: str,
                         file_path: Optional[str] = None, line_number: Optional[int] = None) -> Comment:
        """Add a comment to the project"""
        comment = Comment(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            file_path=file_path,
            line_number=line_number,
            content=content,
            timestamp=datetime.now()
        )
        
        # Save to database (implementation would go here)
        
        # Emit comment added event
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.COMMENT_ADDED,
            timestamp=datetime.now(),
            data={
                "comment_id": comment.id,
                "content": content,
                "file_path": file_path,
                "line_number": line_number
            }
        ))
        
        return comment
    
    async def send_chat_message(self, project_id: str, user_id: str, message: str):
        """Send a chat message to project participants"""
        user = await self.db.get_user(user_id)
        
        await self.emit_event(CollaborationEvent(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            event_type=EventType.CHAT_MESSAGE,
            timestamp=datetime.now(),
            data={
                "message": message,
                "user_name": user.name if user else "Unknown"
            }
        ))
    
    async def emit_event(self, event: CollaborationEvent):
        """Emit an event to all subscribers"""
        # Save event to database
        await self.db.save_event(event)
        
        # Notify subscribers via WebSocket
        if event.project_id in self.project_subscribers:
            subscribers = self.project_subscribers[event.project_id].copy()
            for user_id in subscribers:
                if user_id in self.active_connections:
                    try:
                        await self.active_connections[user_id].send(json.dumps(event.to_dict()))
                    except websockets.exceptions.ConnectionClosed:
                        # Remove disconnected user
                        del self.active_connections[user_id]
                        self.project_subscribers[event.project_id].discard(user_id)
        
        # Publish to Redis for scaling across multiple instances
        if self.redis_client:
            try:
                await self.redis_client.publish(f"project:{event.project_id}", json.dumps(event.to_dict()))
            except Exception as e:
                print(f"Redis publish failed: {e}")
        
        # Call registered callbacks
        if event.event_type in self.event_callbacks:
            for callback in self.event_callbacks[event.event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    print(f"Event callback failed: {e}")
    
    async def subscribe_to_project(self, user_id: str, project_id: str):
        """Subscribe user to project events"""
        if project_id not in self.project_subscribers:
            self.project_subscribers[project_id] = set()
        
        self.project_subscribers[project_id].add(user_id)
    
    async def unsubscribe_from_project(self, user_id: str, project_id: str):
        """Unsubscribe user from project events"""
        if project_id in self.project_subscribers:
            self.project_subscribers[project_id].discard(user_id)
    
    def register_event_callback(self, event_type: EventType, callback: Callable):
        """Register a callback for specific event types"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def get_project_activity(self, project_id: str, hours: int = 24) -> List[CollaborationEvent]:
        """Get recent project activity"""
        return await self.db.get_project_events(project_id, limit=100)
    
    async def get_online_users(self, project_id: str) -> List[User]:
        """Get list of online users in a project"""
        if project_id not in self.project_subscribers:
            return []
        
        online_users = []
        for user_id in self.project_subscribers[project_id]:
            if user_id in self.active_connections:
                user = await self.db.get_user(user_id)
                if user:
                    user.is_online = True
                    online_users.append(user)
        
        return online_users


class WebSocketHandler:
    """Handles WebSocket connections for real-time collaboration"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.manager = collaboration_manager
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        user_id = None
        project_id = None
        
        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get("user_id")
            project_id = auth_data.get("project_id")
            
            if not user_id or not project_id:
                await websocket.send(json.dumps({"error": "Missing user_id or project_id"}))
                return
            
            # Verify user has access to project
            project = await self.manager.db.get_project(project_id)
            if not project or user_id not in project.participants:
                await websocket.send(json.dumps({"error": "Access denied"}))
                return
            
            # Register connection
            self.manager.active_connections[user_id] = websocket
            await self.manager.subscribe_to_project(user_id, project_id)
            
            # Send confirmation
            await websocket.send(json.dumps({
                "type": "connected",
                "user_id": user_id,
                "project_id": project_id
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(user_id, project_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Cleanup on disconnect
            if user_id:
                if user_id in self.manager.active_connections:
                    del self.manager.active_connections[user_id]
                if project_id:
                    await self.manager.unsubscribe_from_project(user_id, project_id)
    
    async def handle_message(self, user_id: str, project_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        
        if message_type == "cursor_update":
            await self.manager.update_cursor_position(
                user_id, project_id, 
                data["file_path"], data["line"], data["column"]
            )
        
        elif message_type == "lock_file":
            success = await self.manager.lock_file(project_id, data["file_path"], user_id)
            # Send response back to user
            if user_id in self.manager.active_connections:
                await self.manager.active_connections[user_id].send(json.dumps({
                    "type": "lock_response",
                    "file_path": data["file_path"],
                    "success": success
                }))
        
        elif message_type == "unlock_file":
            await self.manager.unlock_file(project_id, data["file_path"], user_id)
        
        elif message_type == "chat_message":
            await self.manager.send_chat_message(project_id, user_id, data["message"])
        
        elif message_type == "add_comment":
            await self.manager.add_comment(
                project_id, user_id, data["content"],
                data.get("file_path"), data.get("line_number")
            )


class CollaborationAPI:
    """REST API for collaboration features"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.manager = collaboration_manager
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/api/projects/{project_id}', self.get_project)
        self.app.router.add_post('/api/projects', self.create_project)
        self.app.router.add_put('/api/projects/{project_id}/join', self.join_project)
        self.app.router.add_delete('/api/projects/{project_id}/leave', self.leave_project)
        self.app.router.add_get('/api/projects/{project_id}/activity', self.get_project_activity)
        self.app.router.add_get('/api/projects/{project_id}/users', self.get_project_users)
        self.app.router.add_post('/api/users', self.create_user)
        self.app.router.add_get('/api/users/{user_id}', self.get_user)
    
    async def get_project(self, request):
        """Get project information"""
        project_id = request.match_info['project_id']
        project = await self.manager.db.get_project(project_id)
        
        if not project:
            return web.json_response({"error": "Project not found"}, status=404)
        
        return web.json_response(asdict(project), dumps=self.json_serializer)
    
    async def create_project(self, request):
        """Create new project"""
        data = await request.json()
        
        project = await self.manager.create_project(
            name=data["name"],
            description=data.get("description", ""),
            owner_id=data["owner_id"],
            source_language=data["source_language"],
            target_languages=data.get("target_languages", [])
        )
        
        return web.json_response(asdict(project), dumps=self.json_serializer)
    
    async def join_project(self, request):
        """Join project"""
        project_id = request.match_info['project_id']
        data = await request.json()
        
        await self.manager.join_project(
            user_id=data["user_id"],
            project_id=project_id,
            role=UserRole(data.get("role", "contributor"))
        )
        
        return web.json_response({"success": True})
    
    async def leave_project(self, request):
        """Leave project"""
        project_id = request.match_info['project_id']
        data = await request.json()
        
        await self.manager.leave_project(data["user_id"], project_id)
        
        return web.json_response({"success": True})
    
    async def get_project_activity(self, request):
        """Get project activity"""
        project_id = request.match_info['project_id']
        activity = await self.manager.get_project_activity(project_id)
        
        return web.json_response([event.to_dict() for event in activity])
    
    async def get_project_users(self, request):
        """Get online users in project"""
        project_id = request.match_info['project_id']
        users = await self.manager.get_online_users(project_id)
        
        return web.json_response([asdict(user) for user in users], dumps=self.json_serializer)
    
    async def create_user(self, request):
        """Create new user"""
        data = await request.json()
        
        user = await self.manager.create_user(
            name=data["name"],
            email=data["email"],
            role=UserRole(data.get("role", "contributor"))
        )
        
        return web.json_response(asdict(user), dumps=self.json_serializer)
    
    async def get_user(self, request):
        """Get user information"""
        user_id = request.match_info['user_id']
        user = await self.manager.db.get_user(user_id)
        
        if not user:
            return web.json_response({"error": "User not found"}, status=404)
        
        return web.json_response(asdict(user), dumps=self.json_serializer)
    
    def json_serializer(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class CollaborationServer:
    """Main collaboration server that combines WebSocket and HTTP APIs"""
    
    def __init__(self, host: str = "localhost", http_port: int = 8080, ws_port: int = 8081):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        
        self.manager = CollaborationManager()
        self.api = CollaborationAPI(self.manager)
        self.ws_handler = WebSocketHandler(self.manager)
    
    async def start(self):
        """Start the collaboration server"""
        await self.manager.initialize()
        
        print(f"Starting collaboration server...")
        print(f"HTTP API: http://{self.host}:{self.http_port}")
        print(f"WebSocket: ws://{self.host}:{self.ws_port}")
        
        # Start HTTP server
        runner = web.AppRunner(self.api.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()
        
        # Start WebSocket server
        ws_server = await websockets.serve(
            self.ws_handler.handle_connection,
            self.host,
            self.ws_port
        )
        
        return runner, ws_server
    
    async def stop(self, runner, ws_server):
        """Stop the collaboration server"""
        await runner.cleanup()
        ws_server.close()
        await ws_server.wait_closed()


# Example usage and testing
async def test_collaboration_system():
    """Test the collaboration system"""
    print("Testing Real-Time Collaboration System...")
    
    # Create collaboration manager
    manager = CollaborationManager(db_path=":memory:")
    await manager.initialize()
    
    # Create test users
    alice = await manager.create_user("Alice Johnson", "alice@example.com", UserRole.OWNER)
    bob = await manager.create_user("Bob Smith", "bob@example.com", UserRole.CONTRIBUTOR)
    
    # Create test project
    project = await manager.create_project(
        name="Legacy Java Migration",
        description="Migrating legacy Java code to Rust",
        owner_id=alice.id,
        source_language="java",
        target_languages=["rust", "go"]
    )
    
    # Add Bob to project
    await manager.join_project(bob.id, project.id, UserRole.REVIEWER)
    
    # Simulate collaboration activities
    await manager.update_project_status(project.id, ProjectStatus.ANALYZING, alice.id)
    await manager.lock_file(project.id, "src/main.java", alice.id)
    await manager.add_comment(project.id, bob.id, "This method looks complex", "src/main.java", 42)
    await manager.send_chat_message(project.id, alice.id, "Starting analysis of the main class")
    
    # Get project activity
    activity = await manager.get_project_activity(project.id)
    
    print(f"✅ Created project: {project.name}")
    print(f"✅ Added {len(project.participants)} participants")
    print(f"✅ Generated {len(activity)} collaboration events")
    
    return manager


async def run_collaboration_server():
    """Run the collaboration server"""
    server = CollaborationServer()
    runner, ws_server = await server.start()
    
    try:
        # Keep server running
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("Shutting down collaboration server...")
        await server.stop(runner, ws_server)


if __name__ == "__main__":
    # Test mode - run quick test
    if len(asyncio.sys.argv) > 1 and asyncio.sys.argv[1] == "--test":
        asyncio.run(test_collaboration_system())
    else:
        # Server mode - run full server
        asyncio.run(run_collaboration_server())