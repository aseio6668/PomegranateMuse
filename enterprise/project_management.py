"""
Project Management Integration for MyndraComposer
Integrates with Jira, Azure DevOps, Trello, and other project management systems
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import base64

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class IssueType(Enum):
    """Issue/work item types"""
    STORY = "story"
    TASK = "task"
    BUG = "bug"
    EPIC = "epic"
    FEATURE = "feature"
    SUBTASK = "subtask"
    IMPROVEMENT = "improvement"
    TEST = "test"

class Priority(Enum):
    """Issue priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"

class IssueStatus(Enum):
    """Issue status values"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@dataclass
class WorkItem:
    """Generic work item/issue representation"""
    id: str
    title: str
    description: str
    issue_type: IssueType
    status: IssueStatus
    priority: Priority
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    project_key: str = ""
    labels: List[str] = None
    components: List[str] = None
    epic_link: Optional[str] = None
    story_points: Optional[int] = None
    estimated_hours: Optional[float] = None
    time_spent: Optional[float] = None
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    comments: List[Dict[str, Any]] = None
    attachments: List[Dict[str, Any]] = None
    custom_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.components is None:
            self.components = []
        if self.comments is None:
            self.comments = []
        if self.attachments is None:
            self.attachments = []
        if self.custom_fields is None:
            self.custom_fields = {}
        if self.created_date is None:
            self.created_date = datetime.now()

@dataclass
class ProjectBoard:
    """Project board/dashboard representation"""
    id: str
    name: str
    description: str
    project_key: str
    board_type: str = "scrum"  # scrum, kanban, basic
    columns: List[Dict[str, Any]] = None
    filters: Dict[str, Any] = None
    administrators: List[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.filters is None:
            self.filters = {}
        if self.administrators is None:
            self.administrators = []

class IssueTracker:
    """Base class for issue tracking integrations"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any]):
        self.base_url = base_url.rstrip('/')
        self.auth_config = auth_config
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.logger = logging.getLogger(__name__)
        
    async def authenticate(self) -> bool:
        """Authenticate with the service"""
        raise NotImplementedError
        
    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get list of projects"""
        raise NotImplementedError
        
    async def get_issues(self, project_key: str, filters: Dict[str, Any] = None) -> List[WorkItem]:
        """Get issues from project"""
        raise NotImplementedError
        
    async def create_issue(self, issue: WorkItem) -> str:
        """Create new issue"""
        raise NotImplementedError
        
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing issue"""
        raise NotImplementedError
        
    async def get_issue(self, issue_id: str) -> Optional[WorkItem]:
        """Get specific issue"""
        raise NotImplementedError

class JiraIntegration(IssueTracker):
    """Jira integration implementation"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any]):
        super().__init__(base_url, auth_config)
        self.api_url = f"{base_url}/rest/api/2"
        
    async def authenticate(self) -> bool:
        """Authenticate with Jira"""
        if not REQUESTS_AVAILABLE:
            self.logger.error("Requests library not available")
            return False
            
        auth_type = self.auth_config.get("type", "basic")
        
        try:
            if auth_type == "basic":
                username = self.auth_config["username"]
                password = self.auth_config["password"]
                self.session.auth = (username, password)
                
            elif auth_type == "token":
                email = self.auth_config["email"]
                token = self.auth_config["token"]
                credentials = base64.b64encode(f"{email}:{token}".encode()).decode()
                self.session.headers.update({
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json"
                })
                
            elif auth_type == "oauth":
                # OAuth implementation would go here
                self.logger.warning("OAuth authentication not implemented")
                return False
                
            # Test authentication
            response = self.session.get(f"{self.api_url}/myself")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Jira authentication failed: {e}")
            return False
    
    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get Jira projects"""
        try:
            response = self.session.get(f"{self.api_url}/project")
            response.raise_for_status()
            
            projects = []
            for project in response.json():
                projects.append({
                    "key": project["key"],
                    "name": project["name"],
                    "description": project.get("description", ""),
                    "lead": project.get("lead", {}).get("displayName", ""),
                    "project_type": project.get("projectTypeKey", "")
                })
                
            return projects
            
        except Exception as e:
            self.logger.error(f"Failed to get Jira projects: {e}")
            return []
    
    async def get_issues(self, project_key: str, filters: Dict[str, Any] = None) -> List[WorkItem]:
        """Get issues from Jira project"""
        try:
            jql = f"project = {project_key}"
            
            if filters:
                if filters.get("status"):
                    jql += f" AND status = '{filters['status']}'"
                if filters.get("assignee"):
                    jql += f" AND assignee = '{filters['assignee']}'"
                if filters.get("issue_type"):
                    jql += f" AND issuetype = '{filters['issue_type']}'"
                    
            params = {
                "jql": jql,
                "fields": "summary,description,issuetype,status,priority,assignee,reporter,created,updated,duedate,labels,components,customfield_10016",  # customfield_10016 is often story points
                "maxResults": 1000
            }
            
            response = self.session.get(f"{self.api_url}/search", params=params)
            response.raise_for_status()
            
            issues = []
            for issue_data in response.json()["issues"]:
                issue = self._convert_jira_issue(issue_data)
                if issue:
                    issues.append(issue)
                    
            return issues
            
        except Exception as e:
            self.logger.error(f"Failed to get Jira issues: {e}")
            return []
    
    def _convert_jira_issue(self, issue_data: Dict[str, Any]) -> Optional[WorkItem]:
        """Convert Jira issue to WorkItem"""
        try:
            fields = issue_data["fields"]
            
            # Map Jira issue type to our enum
            jira_type = fields["issuetype"]["name"].lower()
            issue_type_mapping = {
                "story": IssueType.STORY,
                "task": IssueType.TASK,
                "bug": IssueType.BUG,
                "epic": IssueType.EPIC,
                "sub-task": IssueType.SUBTASK,
                "improvement": IssueType.IMPROVEMENT,
                "test": IssueType.TEST
            }
            issue_type = issue_type_mapping.get(jira_type, IssueType.TASK)
            
            # Map Jira status to our enum
            jira_status = fields["status"]["name"].lower()
            status_mapping = {
                "to do": IssueStatus.TODO,
                "in progress": IssueStatus.IN_PROGRESS,
                "in review": IssueStatus.IN_REVIEW,
                "testing": IssueStatus.TESTING,
                "done": IssueStatus.DONE,
                "blocked": IssueStatus.BLOCKED,
                "cancelled": IssueStatus.CANCELLED
            }
            status = status_mapping.get(jira_status, IssueStatus.TODO)
            
            # Map Jira priority to our enum
            jira_priority = fields.get("priority", {}).get("name", "medium").lower()
            priority_mapping = {
                "highest": Priority.CRITICAL,
                "high": Priority.HIGH,
                "medium": Priority.MEDIUM,
                "low": Priority.LOW,
                "lowest": Priority.TRIVIAL
            }
            priority = priority_mapping.get(jira_priority, Priority.MEDIUM)
            
            return WorkItem(
                id=issue_data["key"],
                title=fields["summary"],
                description=fields.get("description", ""),
                issue_type=issue_type,
                status=status,
                priority=priority,
                assignee=fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                reporter=fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
                project_key=issue_data["key"].split("-")[0],
                labels=[label for label in fields.get("labels", [])],
                components=[comp["name"] for comp in fields.get("components", [])],
                story_points=fields.get("customfield_10016"),  # Common story points field
                created_date=datetime.fromisoformat(fields["created"].replace("Z", "+00:00")) if fields.get("created") else None,
                updated_date=datetime.fromisoformat(fields["updated"].replace("Z", "+00:00")) if fields.get("updated") else None,
                due_date=datetime.fromisoformat(fields["duedate"]) if fields.get("duedate") else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to convert Jira issue: {e}")
            return None
    
    async def create_issue(self, issue: WorkItem) -> str:
        """Create new Jira issue"""
        try:
            issue_data = {
                "fields": {
                    "project": {"key": issue.project_key},
                    "summary": issue.title,
                    "description": issue.description,
                    "issuetype": {"name": issue.issue_type.value.title()},
                    "priority": {"name": issue.priority.value.title()}
                }
            }
            
            if issue.assignee:
                issue_data["fields"]["assignee"] = {"name": issue.assignee}
                
            if issue.labels:
                issue_data["fields"]["labels"] = issue.labels
                
            if issue.story_points:
                issue_data["fields"]["customfield_10016"] = issue.story_points
                
            response = self.session.post(f"{self.api_url}/issue", json=issue_data)
            response.raise_for_status()
            
            return response.json()["key"]
            
        except Exception as e:
            self.logger.error(f"Failed to create Jira issue: {e}")
            return ""

class AzureDevOpsIntegration(IssueTracker):
    """Azure DevOps integration implementation"""
    
    def __init__(self, organization: str, auth_config: Dict[str, Any]):
        base_url = f"https://dev.azure.com/{organization}"
        super().__init__(base_url, auth_config)
        self.api_url = f"{base_url}/_apis"
        
    async def authenticate(self) -> bool:
        """Authenticate with Azure DevOps"""
        if not REQUESTS_AVAILABLE:
            return False
            
        try:
            token = self.auth_config.get("personal_access_token")
            if token:
                credentials = base64.b64encode(f":{token}".encode()).decode()
                self.session.headers.update({
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json"
                })
                
            # Test authentication
            response = self.session.get(f"{self.api_url}/projects?api-version=6.0")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Azure DevOps authentication failed: {e}")
            return False
    
    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get Azure DevOps projects"""
        try:
            response = self.session.get(f"{self.api_url}/projects?api-version=6.0")
            response.raise_for_status()
            
            projects = []
            for project in response.json()["value"]:
                projects.append({
                    "key": project["id"],
                    "name": project["name"],
                    "description": project.get("description", ""),
                    "state": project.get("state", ""),
                    "visibility": project.get("visibility", "")
                })
                
            return projects
            
        except Exception as e:
            self.logger.error(f"Failed to get Azure DevOps projects: {e}")
            return []

class TrelloIntegration(IssueTracker):
    """Trello integration implementation"""
    
    def __init__(self, auth_config: Dict[str, Any]):
        super().__init__("https://api.trello.com", auth_config)
        
    async def authenticate(self) -> bool:
        """Authenticate with Trello"""
        if not REQUESTS_AVAILABLE:
            return False
            
        try:
            key = self.auth_config.get("api_key")
            token = self.auth_config.get("token")
            
            if key and token:
                self.session.params.update({
                    "key": key,
                    "token": token
                })
                
            # Test authentication
            response = self.session.get(f"{self.base_url}/1/members/me")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Trello authentication failed: {e}")
            return False

class ProjectManagementIntegration:
    """Main project management integration orchestrator"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/project_mgmt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.integrations: Dict[str, IssueTracker] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_jira_integration(self, name: str, base_url: str, auth_config: Dict[str, Any]):
        """Add Jira integration"""
        self.integrations[name] = JiraIntegration(base_url, auth_config)
        
    def add_azure_devops_integration(self, name: str, organization: str, auth_config: Dict[str, Any]):
        """Add Azure DevOps integration"""
        self.integrations[name] = AzureDevOpsIntegration(organization, auth_config)
        
    def add_trello_integration(self, name: str, auth_config: Dict[str, Any]):
        """Add Trello integration"""
        self.integrations[name] = TrelloIntegration(auth_config)
        
    async def authenticate_all(self) -> Dict[str, bool]:
        """Authenticate all integrations"""
        results = {}
        for name, integration in self.integrations.items():
            try:
                results[name] = await integration.authenticate()
            except Exception as e:
                self.logger.error(f"Authentication failed for {name}: {e}")
                results[name] = False
                
        return results
    
    async def sync_project_data(self, integration_name: str, project_key: str) -> Dict[str, Any]:
        """Sync project data from specific integration"""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration {integration_name} not found")
            
        integration = self.integrations[integration_name]
        
        try:
            # Get project info
            projects = await integration.get_projects()
            project_info = next((p for p in projects if p["key"] == project_key), None)
            
            if not project_info:
                raise ValueError(f"Project {project_key} not found")
                
            # Get issues
            issues = await integration.get_issues(project_key)
            
            # Cache the data
            project_data = {
                "project": project_info,
                "issues": [asdict(issue) for issue in issues],
                "last_sync": datetime.now().isoformat(),
                "integration": integration_name
            }
            
            cache_file = self.cache_dir / f"{integration_name}_{project_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
                
            return project_data
            
        except Exception as e:
            self.logger.error(f"Failed to sync project data: {e}")
            raise
    
    async def create_work_item(self, integration_name: str, work_item: WorkItem) -> str:
        """Create work item in project management system"""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration {integration_name} not found")
            
        integration = self.integrations[integration_name]
        return await integration.create_issue(work_item)
    
    def get_cached_project_data(self, integration_name: str, project_key: str) -> Optional[Dict[str, Any]]:
        """Get cached project data"""
        cache_file = self.cache_dir / f"{integration_name}_{project_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None
    
    async def generate_work_items_for_migration(self, project_key: str, migration_plan: Dict[str, Any]) -> List[WorkItem]:
        """Generate work items for code migration project"""
        work_items = []
        
        # Create epic for overall migration
        epic = WorkItem(
            id="",
            title=f"Code Migration: {migration_plan.get('name', 'Unnamed Project')}",
            description=f"Migrate codebase from {migration_plan.get('source_language')} to {migration_plan.get('target_language')}",
            issue_type=IssueType.EPIC,
            status=IssueStatus.TODO,
            priority=Priority.HIGH,
            project_key=project_key
        )
        work_items.append(epic)
        
        # Create stories for each component
        for component in migration_plan.get("components", []):
            story = WorkItem(
                id="",
                title=f"Migrate {component['name']} component",
                description=f"Migrate {component['name']} from {component.get('current_language')} to {component.get('target_language')}",
                issue_type=IssueType.STORY,
                status=IssueStatus.TODO,
                priority=Priority.MEDIUM,
                project_key=project_key,
                story_points=component.get("complexity", 3),
                labels=["migration", component.get("type", "component")]
            )
            work_items.append(story)
            
            # Create tasks for specific files
            for file_path in component.get("files", []):
                task = WorkItem(
                    id="",
                    title=f"Migrate {Path(file_path).name}",
                    description=f"Migrate file: {file_path}",
                    issue_type=IssueType.TASK,
                    status=IssueStatus.TODO,
                    priority=Priority.LOW,
                    project_key=project_key,
                    labels=["migration", "file"]
                )
                work_items.append(task)
        
        return work_items

async def sync_project_data(integration: ProjectManagementIntegration, 
                          integration_name: str, project_key: str) -> Dict[str, Any]:
    """Convenience function to sync project data"""
    return await integration.sync_project_data(integration_name, project_key)