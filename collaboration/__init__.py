"""
Collaboration System for MyndraComposer
Provides comprehensive collaboration features including real-time communication,
team management, role-based access control, and workflow automation
"""

from .realtime_system import (
    CollaborationServer,
    CollaborationClient,
    User,
    Project,
    FileVersion,
    Comment,
    UserRole as RealtimeUserRole,
    start_collaboration_server,
    connect_to_collaboration
)

from .team_manager import (
    TeamManager,
    TeamMember,
    UserRole,
    PermissionType,
    ActivityType,
    ActivityLog,
    ReviewRequest,
    Team,
    RolePermissionManager,
    display_team_info,
    display_review_dashboard
)

from .team_integration import (
    TeamIntegration,
    WorkflowRule,
    TeamSettings,
    run_team_management_interactive
)

__all__ = [
    # Real-time collaboration
    "CollaborationServer",
    "CollaborationClient", 
    "User",
    "Project",
    "FileVersion",
    "Comment",
    "RealtimeUserRole",
    "start_collaboration_server",
    "connect_to_collaboration",
    
    # Team management
    "TeamManager",
    "TeamMember",
    "UserRole",
    "PermissionType",
    "ActivityType",
    "ActivityLog",
    "ReviewRequest",
    "Team",
    "RolePermissionManager",
    "display_team_info",
    "display_review_dashboard",
    
    # Team integration
    "TeamIntegration",
    "WorkflowRule",
    "TeamSettings",
    "run_team_management_interactive"
]