"""
Enterprise Integration Module for MyndraComposer
Provides comprehensive enterprise system integrations including authentication,
project management, communication platforms, and monitoring systems
"""

from .auth_integration import (
    EnterpriseAuthManager,
    AuthProvider,
    AuthConfig,
    LDAPConfig,
    SAMLConfig,
    OAuthConfig,
    UserProfile,
    AuthenticationError,
    authenticate_user,
    sync_enterprise_users
)

from .project_management import (
    ProjectManagementIntegration,
    JiraIntegration,
    AzureDevOpsIntegration,
    TrelloIntegration,
    IssueTracker,
    WorkItem,
    IssueType,
    Priority,
    ProjectBoard,
    sync_project_data
)

from .communication import (
    CommunicationPlatform,
    SlackIntegration,
    TeamsIntegration,
    EmailNotification,
    NotificationChannel,
    NotificationType,
    send_notification,
    setup_communication_channels
)

from .monitoring import (
    MonitoringIntegration,
    DatadogIntegration,
    NewRelicIntegration,
    PrometheusIntegration,
    MetricsCollector,
    AlertManager,
    MonitoringDashboard,
    setup_monitoring
)

from .enterprise_manager import (
    EnterpriseManager,
    EnterpriseConfig,
    IntegrationType,
    run_enterprise_setup_interactive
)

__all__ = [
    # Authentication
    "EnterpriseAuthManager",
    "AuthProvider",
    "AuthConfig",
    "LDAPConfig", 
    "SAMLConfig",
    "OAuthConfig",
    "UserProfile",
    "AuthenticationError",
    "authenticate_user",
    "sync_enterprise_users",
    
    # Project Management
    "ProjectManagementIntegration",
    "JiraIntegration",
    "AzureDevOpsIntegration", 
    "TrelloIntegration",
    "IssueTracker",
    "WorkItem",
    "IssueType",
    "Priority",
    "ProjectBoard",
    "sync_project_data",
    
    # Communication
    "CommunicationPlatform",
    "SlackIntegration",
    "TeamsIntegration",
    "EmailNotification",
    "NotificationChannel",
    "NotificationType",
    "send_notification",
    "setup_communication_channels",
    
    # Monitoring
    "MonitoringIntegration",
    "DatadogIntegration", 
    "NewRelicIntegration",
    "PrometheusIntegration",
    "MetricsCollector",
    "AlertManager",
    "MonitoringDashboard",
    "setup_monitoring",
    
    # Enterprise Management
    "EnterpriseManager",
    "EnterpriseConfig",
    "IntegrationType",
    "run_enterprise_setup_interactive"
]