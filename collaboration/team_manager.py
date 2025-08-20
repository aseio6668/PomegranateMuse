"""
Team Collaboration Manager for PomegranteMuse
Provides comprehensive team management, role-based access control, and collaboration workflows
"""

import os
import json
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import uuid


class UserRole(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class PermissionType(Enum):
    PROJECT_CREATE = "project_create"
    PROJECT_DELETE = "project_delete"
    PROJECT_MODIFY = "project_modify"
    PROJECT_VIEW = "project_view"
    CODE_GENERATE = "code_generate"
    CODE_REVIEW = "code_review"
    CODE_APPROVE = "code_approve"
    SECURITY_SCAN = "security_scan"
    SECURITY_MANAGE = "security_manage"
    COST_VIEW = "cost_view"
    COST_MANAGE = "cost_manage"
    TEAM_INVITE = "team_invite"
    TEAM_MANAGE = "team_manage"
    SETTINGS_MODIFY = "settings_modify"


class ActivityType(Enum):
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ROLE_CHANGED = "role_changed"
    PROJECT_CREATED = "project_created"
    CODE_GENERATED = "code_generated"
    CODE_REVIEWED = "code_reviewed"
    SECURITY_SCAN = "security_scan"
    COST_ANALYSIS = "cost_analysis"
    SETTINGS_CHANGED = "settings_changed"


@dataclass
class TeamMember:
    """Team member information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    joined_at: str
    last_active: str
    permissions: Set[PermissionType] = field(default_factory=set)
    is_active: bool = True
    profile: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.permissions, list):
            self.permissions = set(PermissionType(p) for p in self.permissions)
        elif isinstance(self.permissions, set) and self.permissions:
            # Convert string permissions to PermissionType if needed
            if isinstance(next(iter(self.permissions)), str):
                self.permissions = set(PermissionType(p) for p in self.permissions)


@dataclass
class ActivityLog:
    """Activity log entry"""
    activity_id: str
    user_id: str
    username: str
    activity_type: ActivityType
    timestamp: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    project_id: Optional[str] = None


@dataclass
class ReviewRequest:
    """Code review request"""
    request_id: str
    author_id: str
    reviewer_ids: List[str]
    title: str
    description: str
    generated_code: str
    source_analysis: Dict[str, Any]
    created_at: str
    status: str  # pending, approved, rejected, merged
    comments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Team:
    """Team configuration"""
    team_id: str
    name: str
    description: str
    created_at: str
    owner_id: str
    members: Dict[str, TeamMember] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    activity_log: List[ActivityLog] = field(default_factory=list)
    review_requests: Dict[str, ReviewRequest] = field(default_factory=dict)


class RolePermissionManager:
    """Manages role-based permissions"""
    
    def __init__(self):
        self.role_permissions = self._define_role_permissions()
    
    def _define_role_permissions(self) -> Dict[UserRole, Set[PermissionType]]:
        """Define permissions for each role"""
        return {
            UserRole.OWNER: {
                # All permissions
                PermissionType.PROJECT_CREATE,
                PermissionType.PROJECT_DELETE,
                PermissionType.PROJECT_MODIFY,
                PermissionType.PROJECT_VIEW,
                PermissionType.CODE_GENERATE,
                PermissionType.CODE_REVIEW,
                PermissionType.CODE_APPROVE,
                PermissionType.SECURITY_SCAN,
                PermissionType.SECURITY_MANAGE,
                PermissionType.COST_VIEW,
                PermissionType.COST_MANAGE,
                PermissionType.TEAM_INVITE,
                PermissionType.TEAM_MANAGE,
                PermissionType.SETTINGS_MODIFY
            },
            UserRole.ADMIN: {
                PermissionType.PROJECT_CREATE,
                PermissionType.PROJECT_MODIFY,
                PermissionType.PROJECT_VIEW,
                PermissionType.CODE_GENERATE,
                PermissionType.CODE_REVIEW,
                PermissionType.CODE_APPROVE,
                PermissionType.SECURITY_SCAN,
                PermissionType.SECURITY_MANAGE,
                PermissionType.COST_VIEW,
                PermissionType.COST_MANAGE,
                PermissionType.TEAM_INVITE,
                PermissionType.TEAM_MANAGE,
                PermissionType.SETTINGS_MODIFY
            },
            UserRole.MAINTAINER: {
                PermissionType.PROJECT_MODIFY,
                PermissionType.PROJECT_VIEW,
                PermissionType.CODE_GENERATE,
                PermissionType.CODE_REVIEW,
                PermissionType.CODE_APPROVE,
                PermissionType.SECURITY_SCAN,
                PermissionType.COST_VIEW,
                PermissionType.TEAM_INVITE
            },
            UserRole.DEVELOPER: {
                PermissionType.PROJECT_VIEW,
                PermissionType.CODE_GENERATE,
                PermissionType.CODE_REVIEW,
                PermissionType.SECURITY_SCAN,
                PermissionType.COST_VIEW
            },
            UserRole.REVIEWER: {
                PermissionType.PROJECT_VIEW,
                PermissionType.CODE_REVIEW,
                PermissionType.CODE_APPROVE,
                PermissionType.SECURITY_SCAN,
                PermissionType.COST_VIEW
            },
            UserRole.VIEWER: {
                PermissionType.PROJECT_VIEW,
                PermissionType.COST_VIEW
            }
        }
    
    def get_permissions(self, role: UserRole) -> Set[PermissionType]:
        """Get permissions for a role"""
        return self.role_permissions.get(role, set())
    
    def has_permission(self, user_role: UserRole, permission: PermissionType) -> bool:
        """Check if a role has a specific permission"""
        return permission in self.get_permissions(user_role)
    
    def can_modify_role(self, current_role: UserRole, target_role: UserRole) -> bool:
        """Check if current role can modify target role"""
        role_hierarchy = {
            UserRole.OWNER: 6,
            UserRole.ADMIN: 5,
            UserRole.MAINTAINER: 4,
            UserRole.DEVELOPER: 3,
            UserRole.REVIEWER: 2,
            UserRole.VIEWER: 1
        }
        
        current_level = role_hierarchy.get(current_role, 0)
        target_level = role_hierarchy.get(target_role, 0)
        
        # Can only modify roles at lower levels
        return current_level > target_level


class TeamManager:
    """Main team collaboration manager"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.team_dir = self.project_root / ".pomuse" / "team"
        self.team_dir.mkdir(parents=True, exist_ok=True)
        
        self.team_file = self.team_dir / "team.json"
        self.permission_manager = RolePermissionManager()
        self.current_user_id = self._get_current_user_id()
        
        # Load or create team
        self.team = self._load_or_create_team()
    
    def _get_current_user_id(self) -> str:
        """Get current user ID (simplified implementation)"""
        # In a real implementation, this would integrate with authentication system
        import getpass
        username = getpass.getuser()
        return hashlib.md5(username.encode()).hexdigest()[:8]
    
    def _load_or_create_team(self) -> Team:
        """Load existing team or create new one"""
        if self.team_file.exists():
            try:
                with open(self.team_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to objects
                members = {}
                for member_id, member_data in data.get("members", {}).items():
                    member_data["role"] = UserRole(member_data["role"])
                    member_data["permissions"] = set(
                        PermissionType(p) for p in member_data.get("permissions", [])
                    )
                    members[member_id] = TeamMember(**member_data)
                
                activity_log = []
                for log_data in data.get("activity_log", []):
                    log_data["activity_type"] = ActivityType(log_data["activity_type"])
                    activity_log.append(ActivityLog(**log_data))
                
                review_requests = {}
                for req_id, req_data in data.get("review_requests", {}).items():
                    review_requests[req_id] = ReviewRequest(**req_data)
                
                team = Team(
                    team_id=data["team_id"],
                    name=data["name"],
                    description=data["description"],
                    created_at=data["created_at"],
                    owner_id=data["owner_id"],
                    members=members,
                    settings=data.get("settings", {}),
                    activity_log=activity_log,
                    review_requests=review_requests
                )
                
                return team
                
            except Exception as e:
                print(f"Warning: Could not load team data: {e}")
        
        # Create new team
        return self._create_new_team()
    
    def _create_new_team(self) -> Team:
        """Create a new team with current user as owner"""
        import getpass
        username = getpass.getuser()
        
        team = Team(
            team_id=str(uuid.uuid4())[:8],
            name=f"{self.project_root.name} Team",
            description="PomegranteMuse development team",
            created_at=datetime.now().isoformat(),
            owner_id=self.current_user_id
        )
        
        # Add current user as owner
        owner = TeamMember(
            user_id=self.current_user_id,
            username=username,
            email=f"{username}@local",  # Simplified
            role=UserRole.OWNER,
            joined_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            permissions=self.permission_manager.get_permissions(UserRole.OWNER)
        )
        
        team.members[self.current_user_id] = owner
        
        # Log team creation
        self._log_activity(
            team,
            self.current_user_id,
            username,
            ActivityType.PROJECT_CREATED,
            "Team created"
        )
        
        self._save_team(team)
        return team
    
    def _save_team(self, team: Team):
        """Save team data to file"""
        # Convert to serializable format
        data = {
            "team_id": team.team_id,
            "name": team.name,
            "description": team.description,
            "created_at": team.created_at,
            "owner_id": team.owner_id,
            "settings": team.settings,
            "members": {},
            "activity_log": [],
            "review_requests": {}
        }
        
        # Convert members
        for member_id, member in team.members.items():
            member_data = asdict(member)
            member_data["role"] = member.role.value
            member_data["permissions"] = [p.value for p in member.permissions]
            data["members"][member_id] = member_data
        
        # Convert activity log
        for activity in team.activity_log:
            activity_data = asdict(activity)
            activity_data["activity_type"] = activity.activity_type.value
            data["activity_log"].append(activity_data)
        
        # Convert review requests
        for req_id, request in team.review_requests.items():
            data["review_requests"][req_id] = asdict(request)
        
        with open(self.team_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_member(
        self, 
        username: str, 
        email: str, 
        role: UserRole = UserRole.DEVELOPER,
        requester_id: str = None
    ) -> bool:
        """Add a new team member"""
        requester_id = requester_id or self.current_user_id
        
        # Check permissions
        requester = self.team.members.get(requester_id)
        if not requester or not self.permission_manager.has_permission(
            requester.role, PermissionType.TEAM_INVITE
        ):
            print("❌ Insufficient permissions to invite team members")
            return False
        
        # Check if user already exists
        user_id = hashlib.md5(username.encode()).hexdigest()[:8]
        if user_id in self.team.members:
            print(f"❌ User {username} is already a team member")
            return False
        
        # Create new member
        new_member = TeamMember(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            joined_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            permissions=self.permission_manager.get_permissions(role)
        )
        
        self.team.members[user_id] = new_member
        
        # Log activity
        self._log_activity(
            self.team,
            requester_id,
            requester.username,
            ActivityType.USER_JOINED,
            f"Added {username} as {role.value}"
        )
        
        self._save_team(self.team)
        print(f"✅ Added {username} to team as {role.value}")
        return True
    
    def remove_member(self, user_id: str, requester_id: str = None) -> bool:
        """Remove a team member"""
        requester_id = requester_id or self.current_user_id
        
        # Check permissions
        requester = self.team.members.get(requester_id)
        if not requester or not self.permission_manager.has_permission(
            requester.role, PermissionType.TEAM_MANAGE
        ):
            print("❌ Insufficient permissions to remove team members")
            return False
        
        # Check if member exists
        member = self.team.members.get(user_id)
        if not member:
            print("❌ Member not found")
            return False
        
        # Can't remove owner
        if member.role == UserRole.OWNER:
            print("❌ Cannot remove team owner")
            return False
        
        # Check role hierarchy
        if not self.permission_manager.can_modify_role(requester.role, member.role):
            print("❌ Cannot remove member with equal or higher role")
            return False
        
        username = member.username
        del self.team.members[user_id]
        
        # Log activity
        self._log_activity(
            self.team,
            requester_id,
            requester.username,
            ActivityType.USER_LEFT,
            f"Removed {username} from team"
        )
        
        self._save_team(self.team)
        print(f"✅ Removed {username} from team")
        return True
    
    def change_member_role(
        self, 
        user_id: str, 
        new_role: UserRole, 
        requester_id: str = None
    ) -> bool:
        """Change a team member's role"""
        requester_id = requester_id or self.current_user_id
        
        # Check permissions
        requester = self.team.members.get(requester_id)
        if not requester or not self.permission_manager.has_permission(
            requester.role, PermissionType.TEAM_MANAGE
        ):
            print("❌ Insufficient permissions to change member roles")
            return False
        
        # Check if member exists
        member = self.team.members.get(user_id)
        if not member:
            print("❌ Member not found")
            return False
        
        # Check role hierarchy
        if not self.permission_manager.can_modify_role(requester.role, member.role):
            print("❌ Cannot modify member with equal or higher role")
            return False
        
        if not self.permission_manager.can_modify_role(requester.role, new_role):
            print("❌ Cannot assign role equal to or higher than your own")
            return False
        
        old_role = member.role
        member.role = new_role
        member.permissions = self.permission_manager.get_permissions(new_role)
        
        # Log activity
        self._log_activity(
            self.team,
            requester_id,
            requester.username,
            ActivityType.ROLE_CHANGED,
            f"Changed {member.username} role from {old_role.value} to {new_role.value}"
        )
        
        self._save_team(self.team)
        print(f"✅ Changed {member.username} role to {new_role.value}")
        return True
    
    def create_review_request(
        self, 
        title: str, 
        description: str, 
        generated_code: str,
        source_analysis: Dict[str, Any],
        reviewer_ids: List[str] = None,
        author_id: str = None
    ) -> Optional[str]:
        """Create a code review request"""
        author_id = author_id or self.current_user_id
        
        # Check permissions
        author = self.team.members.get(author_id)
        if not author or not self.permission_manager.has_permission(
            author.role, PermissionType.CODE_GENERATE
        ):
            print("❌ Insufficient permissions to create review requests")
            return None
        
        # Auto-assign reviewers if not specified
        if not reviewer_ids:
            reviewer_ids = self._auto_assign_reviewers(author_id)
        
        # Validate reviewers
        valid_reviewers = []
        for reviewer_id in reviewer_ids:
            reviewer = self.team.members.get(reviewer_id)
            if reviewer and self.permission_manager.has_permission(
                reviewer.role, PermissionType.CODE_REVIEW
            ):
                valid_reviewers.append(reviewer_id)
        
        if not valid_reviewers:
            print("❌ No valid reviewers found")
            return None
        
        # Create review request
        request_id = str(uuid.uuid4())[:8]
        request = ReviewRequest(
            request_id=request_id,
            author_id=author_id,
            reviewer_ids=valid_reviewers,
            title=title,
            description=description,
            generated_code=generated_code,
            source_analysis=source_analysis,
            created_at=datetime.now().isoformat(),
            status="pending"
        )
        
        self.team.review_requests[request_id] = request
        
        # Log activity
        self._log_activity(
            self.team,
            author_id,
            author.username,
            ActivityType.CODE_GENERATED,
            f"Created review request: {title}"
        )
        
        self._save_team(self.team)
        print(f"✅ Created review request: {request_id}")
        return request_id
    
    def add_review_comment(
        self, 
        request_id: str, 
        comment: str, 
        line_number: Optional[int] = None,
        reviewer_id: str = None
    ) -> bool:
        """Add a comment to a review request"""
        reviewer_id = reviewer_id or self.current_user_id
        
        # Check if request exists
        request = self.team.review_requests.get(request_id)
        if not request:
            print("❌ Review request not found")
            return False
        
        # Check permissions
        reviewer = self.team.members.get(reviewer_id)
        if not reviewer or (
            reviewer_id not in request.reviewer_ids and 
            not self.permission_manager.has_permission(reviewer.role, PermissionType.CODE_REVIEW)
        ):
            print("❌ Insufficient permissions to comment on this review")
            return False
        
        # Add comment
        comment_data = {
            "comment_id": str(uuid.uuid4())[:8],
            "reviewer_id": reviewer_id,
            "reviewer_username": reviewer.username,
            "comment": comment,
            "line_number": line_number,
            "timestamp": datetime.now().isoformat()
        }
        
        request.comments.append(comment_data)
        
        # Log activity
        self._log_activity(
            self.team,
            reviewer_id,
            reviewer.username,
            ActivityType.CODE_REVIEWED,
            f"Commented on review request: {request.title}"
        )
        
        self._save_team(self.team)
        print(f"✅ Added comment to review request")
        return True
    
    def approve_review_request(self, request_id: str, reviewer_id: str = None) -> bool:
        """Approve a review request"""
        reviewer_id = reviewer_id or self.current_user_id
        
        # Check if request exists
        request = self.team.review_requests.get(request_id)
        if not request:
            print("❌ Review request not found")
            return False
        
        # Check permissions
        reviewer = self.team.members.get(reviewer_id)
        if not reviewer or not self.permission_manager.has_permission(
            reviewer.role, PermissionType.CODE_APPROVE
        ):
            print("❌ Insufficient permissions to approve reviews")
            return False
        
        request.status = "approved"
        request.metadata["approved_by"] = reviewer_id
        request.metadata["approved_at"] = datetime.now().isoformat()
        
        # Log activity
        self._log_activity(
            self.team,
            reviewer_id,
            reviewer.username,
            ActivityType.CODE_REVIEWED,
            f"Approved review request: {request.title}"
        )
        
        self._save_team(self.team)
        print(f"✅ Approved review request: {request.title}")
        return True
    
    def _auto_assign_reviewers(self, author_id: str) -> List[str]:
        """Auto-assign reviewers based on availability and role"""
        eligible_reviewers = []
        
        for member_id, member in self.team.members.items():
            if (member_id != author_id and 
                member.is_active and
                self.permission_manager.has_permission(member.role, PermissionType.CODE_REVIEW)):
                eligible_reviewers.append(member_id)
        
        # Return up to 2 reviewers, prioritizing higher roles
        role_priority = {
            UserRole.OWNER: 6,
            UserRole.ADMIN: 5,
            UserRole.MAINTAINER: 4,
            UserRole.REVIEWER: 3,
            UserRole.DEVELOPER: 2,
            UserRole.VIEWER: 1
        }
        
        eligible_reviewers.sort(
            key=lambda uid: role_priority.get(self.team.members[uid].role, 0),
            reverse=True
        )
        
        return eligible_reviewers[:2]
    
    def _log_activity(
        self, 
        team: Team, 
        user_id: str, 
        username: str, 
        activity_type: ActivityType, 
        description: str,
        metadata: Dict[str, Any] = None
    ):
        """Log team activity"""
        activity = ActivityLog(
            activity_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            username=username,
            activity_type=activity_type,
            timestamp=datetime.now().isoformat(),
            description=description,
            metadata=metadata or {}
        )
        
        team.activity_log.append(activity)
        
        # Keep only last 100 activities
        if len(team.activity_log) > 100:
            team.activity_log = team.activity_log[-100:]
    
    def get_team_summary(self) -> Dict[str, Any]:
        """Get team summary information"""
        member_count_by_role = {}
        active_members = 0
        
        for member in self.team.members.values():
            role = member.role.value
            member_count_by_role[role] = member_count_by_role.get(role, 0) + 1
            if member.is_active:
                active_members += 1
        
        # Recent activity (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_activities = [
            activity for activity in self.team.activity_log
            if datetime.fromisoformat(activity.timestamp) > recent_cutoff
        ]
        
        # Pending reviews
        pending_reviews = [
            req for req in self.team.review_requests.values()
            if req.status == "pending"
        ]
        
        return {
            "team_id": self.team.team_id,
            "team_name": self.team.name,
            "total_members": len(self.team.members),
            "active_members": active_members,
            "member_count_by_role": member_count_by_role,
            "recent_activities": len(recent_activities),
            "pending_reviews": len(pending_reviews),
            "created_at": self.team.created_at,
            "owner": self.team.members.get(self.team.owner_id, {}).username if self.team.owner_id in self.team.members else "Unknown"
        }
    
    def get_member_activity(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get activity summary for a specific member"""
        member = self.team.members.get(user_id)
        if not member:
            return {"error": "Member not found"}
        
        # Filter activities for this user
        cutoff_date = datetime.now() - timedelta(days=days)
        user_activities = [
            activity for activity in self.team.activity_log
            if (activity.user_id == user_id and 
                datetime.fromisoformat(activity.timestamp) > cutoff_date)
        ]
        
        # Count activities by type
        activity_counts = {}
        for activity in user_activities:
            activity_type = activity.activity_type.value
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # Review participation
        authored_reviews = len([
            req for req in self.team.review_requests.values()
            if req.author_id == user_id
        ])
        
        reviewed_requests = len([
            req for req in self.team.review_requests.values()
            if user_id in req.reviewer_ids
        ])
        
        return {
            "member_info": {
                "username": member.username,
                "role": member.role.value,
                "joined_at": member.joined_at,
                "last_active": member.last_active
            },
            "activity_summary": {
                "total_activities": len(user_activities),
                "activity_counts": activity_counts,
                "authored_reviews": authored_reviews,
                "reviewed_requests": reviewed_requests
            },
            "recent_activities": [
                {
                    "type": activity.activity_type.value,
                    "description": activity.description,
                    "timestamp": activity.timestamp
                }
                for activity in user_activities[-10:]  # Last 10 activities
            ]
        }
    
    def get_review_dashboard(self) -> Dict[str, Any]:
        """Get review dashboard information"""
        reviews_by_status = {}
        reviews_by_author = {}
        
        for request in self.team.review_requests.values():
            # Count by status
            status = request.status
            reviews_by_status[status] = reviews_by_status.get(status, 0) + 1
            
            # Count by author
            author = self.team.members.get(request.author_id, {}).username or "Unknown"
            reviews_by_author[author] = reviews_by_author.get(author, 0) + 1
        
        # Pending reviews with details
        pending_reviews = []
        for request in self.team.review_requests.values():
            if request.status == "pending":
                author = self.team.members.get(request.author_id, {}).username or "Unknown"
                reviewers = [
                    self.team.members.get(rid, {}).username or "Unknown"
                    for rid in request.reviewer_ids
                ]
                
                pending_reviews.append({
                    "request_id": request.request_id,
                    "title": request.title,
                    "author": author,
                    "reviewers": reviewers,
                    "created_at": request.created_at,
                    "comments_count": len(request.comments)
                })
        
        return {
            "total_reviews": len(self.team.review_requests),
            "reviews_by_status": reviews_by_status,
            "reviews_by_author": reviews_by_author,
            "pending_reviews": pending_reviews
        }
    
    def check_permission(self, user_id: str, permission: PermissionType) -> bool:
        """Check if a user has a specific permission"""
        member = self.team.members.get(user_id)
        if not member or not member.is_active:
            return False
        
        return self.permission_manager.has_permission(member.role, permission)
    
    def update_member_activity(self, user_id: str):
        """Update member's last active timestamp"""
        member = self.team.members.get(user_id)
        if member:
            member.last_active = datetime.now().isoformat()
            self._save_team(self.team)


# CLI helper functions
def display_team_info(team_manager: TeamManager):
    """Display team information"""
    summary = team_manager.get_team_summary()
    
    print(f"\n👥 Team: {summary['team_name']}")
    print(f"   Team ID: {summary['team_id']}")
    print(f"   Owner: {summary['owner']}")
    print(f"   Created: {summary['created_at'][:10]}")
    print(f"   Members: {summary['total_members']} ({summary['active_members']} active)")
    
    print("\n🎭 Roles:")
    for role, count in summary['member_count_by_role'].items():
        print(f"   {role.title()}: {count}")
    
    print(f"\n📊 Activity:")
    print(f"   Recent Activities: {summary['recent_activities']}")
    print(f"   Pending Reviews: {summary['pending_reviews']}")


def display_review_dashboard(team_manager: TeamManager):
    """Display review dashboard"""
    dashboard = team_manager.get_review_dashboard()
    
    print(f"\n📋 Review Dashboard")
    print(f"   Total Reviews: {dashboard['total_reviews']}")
    
    print("\n📊 By Status:")
    for status, count in dashboard['reviews_by_status'].items():
        print(f"   {status.title()}: {count}")
    
    if dashboard['pending_reviews']:
        print(f"\n⏳ Pending Reviews ({len(dashboard['pending_reviews'])}):")
        for review in dashboard['pending_reviews'][:5]:  # Show up to 5
            print(f"   • {review['title']} by {review['author']}")
            print(f"     Reviewers: {', '.join(review['reviewers'])}")
            print(f"     Comments: {review['comments_count']}")


if __name__ == "__main__":
    # Example usage
    manager = TeamManager(".")
    
    # Display team info
    display_team_info(manager)
    
    # Display review dashboard
    display_review_dashboard(manager)