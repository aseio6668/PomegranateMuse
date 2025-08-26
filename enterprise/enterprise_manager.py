"""
Enterprise Manager for MyndraComposer
Orchestrates all enterprise integrations and provides unified configuration
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .auth_integration import (
    EnterpriseAuthManager, AuthConfig, AuthProvider,
    LDAPConfig, SAMLConfig, OAuthConfig
)
from .project_management import (
    ProjectManagementIntegration, WorkItem, IssueType, Priority
)
from .communication import (
    CommunicationManager, NotificationType, setup_communication_channels
)
from .monitoring import (
    MonitoringManager, setup_monitoring, MetricType
)

class IntegrationType(Enum):
    """Types of enterprise integrations"""
    AUTHENTICATION = "authentication"
    PROJECT_MANAGEMENT = "project_management"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"
    ALL = "all"

@dataclass
class EnterpriseConfig:
    """Main enterprise configuration"""
    organization_name: str
    domain: str
    authentication: Dict[str, Any] = None
    project_management: Dict[str, Any] = None
    communication: Dict[str, Any] = None
    monitoring: Dict[str, Any] = None
    general_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authentication is None:
            self.authentication = {}
        if self.project_management is None:
            self.project_management = {}
        if self.communication is None:
            self.communication = {}
        if self.monitoring is None:
            self.monitoring = {}
        if self.general_settings is None:
            self.general_settings = {}

class EnterpriseManager:
    """Main enterprise integration manager"""
    
    def __init__(self, config: EnterpriseConfig, cache_dir: Optional[str] = None):
        self.config = config
        self.cache_dir = Path(cache_dir or ".pomuse/enterprise")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.auth_manager: Optional[EnterpriseAuthManager] = None
        self.project_manager: Optional[ProjectManagementIntegration] = None
        self.communication_manager: Optional[CommunicationManager] = None
        self.monitoring_manager: Optional[MonitoringManager] = None
        
        self.logger = logging.getLogger(__name__)
        self.initialized_integrations: List[IntegrationType] = []
        
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all enterprise integrations"""
        results = {}
        
        # Initialize authentication
        if self.config.authentication:
            results["authentication"] = await self.initialize_authentication()
        
        # Initialize project management
        if self.config.project_management:
            results["project_management"] = await self.initialize_project_management()
            
        # Initialize communication
        if self.config.communication:
            results["communication"] = await self.initialize_communication()
            
        # Initialize monitoring
        if self.config.monitoring:
            results["monitoring"] = await self.initialize_monitoring()
            
        return results
    
    async def initialize_authentication(self) -> bool:
        """Initialize authentication integration"""
        try:
            auth_config_data = self.config.authentication
            
            # Create AuthConfig based on provider
            provider = AuthProvider(auth_config_data["provider"])
            
            auth_config = AuthConfig(
                provider=provider,
                session_timeout=auth_config_data.get("session_timeout", 3600),
                require_mfa=auth_config_data.get("require_mfa", False),
                allowed_domains=auth_config_data.get("allowed_domains", [])
            )
            
            # Set provider-specific config
            if provider == AuthProvider.LDAP:
                ldap_data = auth_config_data["ldap"]
                auth_config.ldap_config = LDAPConfig(**ldap_data)
            elif provider == AuthProvider.SAML:
                saml_data = auth_config_data["saml"]
                auth_config.saml_config = SAMLConfig(**saml_data)
            elif provider in [AuthProvider.OAUTH2, AuthProvider.AZURE_AD]:
                oauth_data = auth_config_data["oauth"]
                auth_config.oauth_config = OAuthConfig(**oauth_data)
            
            self.auth_manager = EnterpriseAuthManager(
                auth_config, 
                str(self.cache_dir / "auth")
            )
            
            self.initialized_integrations.append(IntegrationType.AUTHENTICATION)
            self.logger.info("Authentication integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize authentication: {e}")
            return False
    
    async def initialize_project_management(self) -> bool:
        """Initialize project management integration"""
        try:
            pm_config = self.config.project_management
            
            self.project_manager = ProjectManagementIntegration(
                str(self.cache_dir / "project_mgmt")
            )
            
            # Add integrations based on configuration
            for integration_name, integration_config in pm_config.items():
                integration_type = integration_config.get("type")
                
                if integration_type == "jira":
                    self.project_manager.add_jira_integration(
                        integration_name,
                        integration_config["base_url"],
                        integration_config["auth"]
                    )
                elif integration_type == "azure_devops":
                    self.project_manager.add_azure_devops_integration(
                        integration_name,
                        integration_config["organization"],
                        integration_config["auth"]
                    )
                elif integration_type == "trello":
                    self.project_manager.add_trello_integration(
                        integration_name,
                        integration_config["auth"]
                    )
            
            # Test authentication for all integrations
            auth_results = await self.project_manager.authenticate_all()
            failed_auths = [name for name, success in auth_results.items() if not success]
            
            if failed_auths:
                self.logger.warning(f"Authentication failed for: {failed_auths}")
            
            self.initialized_integrations.append(IntegrationType.PROJECT_MANAGEMENT)
            self.logger.info("Project management integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize project management: {e}")
            return False
    
    async def initialize_communication(self) -> bool:
        """Initialize communication integration"""
        try:
            self.communication_manager = await setup_communication_channels(
                self.config.communication
            )
            
            self.initialized_integrations.append(IntegrationType.COMMUNICATION)
            self.logger.info("Communication integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            return False
    
    async def initialize_monitoring(self) -> bool:
        """Initialize monitoring integration"""
        try:
            self.monitoring_manager = await setup_monitoring(
                self.config.monitoring
            )
            
            self.initialized_integrations.append(IntegrationType.MONITORING)
            self.logger.info("Monitoring integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            return False
    
    async def authenticate_user(self, username: str, password: str = None, 
                              token: str = None, **kwargs):
        """Authenticate user through enterprise system"""
        if not self.auth_manager:
            raise ValueError("Authentication not initialized")
            
        return await self.auth_manager.authenticate(username, password, token, **kwargs)
    
    async def create_migration_project(self, project_name: str, 
                                     source_language: str, target_language: str,
                                     file_paths: List[str]) -> Dict[str, Any]:
        """Create comprehensive migration project with all enterprise integrations"""
        project_data = {
            "name": project_name,
            "source_language": source_language,
            "target_language": target_language,
            "files": file_paths,
            "created_at": datetime.now().isoformat(),
            "status": "planning"
        }
        
        results = {}
        
        # Create project management work items
        if self.project_manager:
            try:
                # Generate work items for migration
                migration_plan = {
                    "name": project_name,
                    "source_language": source_language,
                    "target_language": target_language,
                    "components": [
                        {
                            "name": f"Component {i+1}",
                            "files": [file_paths[i]] if i < len(file_paths) else [],
                            "complexity": 3,
                            "type": "migration"
                        }
                        for i in range(min(len(file_paths), 10))  # Limit to 10 components
                    ]
                }
                
                # Get first available project key (simplified)
                projects = []
                for integration_name in self.project_manager.integrations:
                    try:
                        projects = await self.project_manager.integrations[integration_name].get_projects()
                        if projects:
                            break
                    except:
                        continue
                
                if projects:
                    project_key = projects[0]["key"]
                    work_items = await self.project_manager.generate_work_items_for_migration(
                        project_key, migration_plan
                    )
                    
                    results["work_items_generated"] = len(work_items)
                    
            except Exception as e:
                self.logger.error(f"Failed to create project management items: {e}")
                results["project_management_error"] = str(e)
        
        # Send initial notification
        if self.communication_manager:
            try:
                notification_data = {
                    "project": project_name,
                    "source_language": source_language,
                    "target_language": target_language,
                    "files_count": len(file_paths)
                }
                
                await self.communication_manager.send_notification(
                    NotificationType.MIGRATION_STARTED,
                    notification_data
                )
                
                results["notification_sent"] = True
                
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")
                results["notification_error"] = str(e)
        
        # Initialize monitoring
        if self.monitoring_manager:
            try:
                # Track project creation metrics
                self.monitoring_manager.collector.increment_counter(
                    "myndra.projects.created",
                    1,
                    {"source_language": source_language, "target_language": target_language}
                )
                
                self.monitoring_manager.collector.set_gauge(
                    "myndra.project.files_count",
                    len(file_paths),
                    {"project": project_name}
                )
                
                # Send metrics
                await self.monitoring_manager.send_metrics()
                
                results["metrics_recorded"] = True
                
            except Exception as e:
                self.logger.error(f"Failed to record metrics: {e}")
                results["monitoring_error"] = str(e)
        
        # Save project data
        project_file = self.cache_dir / f"project_{project_name.lower().replace(' ', '_')}.json"
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)
            
        results["project_created"] = True
        results["project_file"] = str(project_file)
        
        return results
    
    async def complete_migration_project(self, project_name: str, 
                                       files_migrated: int, errors: int = 0,
                                       duration: float = 0):
        """Complete migration project and update all systems"""
        results = {}
        
        # Send completion notification
        if self.communication_manager:
            try:
                await self.communication_manager.send_migration_notification(
                    project_name, files_migrated, "mixed", "target", 
                    f"{duration:.1f}s" if duration else ""
                )
                results["notification_sent"] = True
            except Exception as e:
                results["notification_error"] = str(e)
        
        # Track completion metrics
        if self.monitoring_manager:
            try:
                self.monitoring_manager.track_migration_metrics(
                    files_migrated, "mixed", "target", duration, errors
                )
                await self.monitoring_manager.send_metrics()
                results["metrics_recorded"] = True
            except Exception as e:
                results["monitoring_error"] = str(e)
        
        return results
    
    async def handle_security_alert(self, project: str, vulnerability: str, 
                                   severity: str, affected_files: List[str]):
        """Handle security alert across all systems"""
        results = {}
        
        # Send security notification
        if self.communication_manager:
            try:
                await self.communication_manager.send_security_alert(
                    project, vulnerability, severity, affected_files
                )
                results["notification_sent"] = True
            except Exception as e:
                results["notification_error"] = str(e)
        
        # Track security metrics
        if self.monitoring_manager:
            try:
                severity_counts = {severity: 1}
                self.monitoring_manager.track_security_metrics(
                    1, severity_counts, project
                )
                await self.monitoring_manager.send_metrics()
                results["metrics_recorded"] = True
            except Exception as e:
                results["monitoring_error"] = str(e)
        
        return results
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        status = {}
        
        for integration_type in IntegrationType:
            if integration_type == IntegrationType.ALL:
                continue
                
            is_initialized = integration_type in self.initialized_integrations
            
            status[integration_type.value] = {
                "initialized": is_initialized,
                "configured": self._is_integration_configured(integration_type),
                "manager": self._get_manager_status(integration_type)
            }
        
        return status
    
    def _is_integration_configured(self, integration_type: IntegrationType) -> bool:
        """Check if integration is configured"""
        if integration_type == IntegrationType.AUTHENTICATION:
            return bool(self.config.authentication)
        elif integration_type == IntegrationType.PROJECT_MANAGEMENT:
            return bool(self.config.project_management)
        elif integration_type == IntegrationType.COMMUNICATION:
            return bool(self.config.communication)
        elif integration_type == IntegrationType.MONITORING:
            return bool(self.config.monitoring)
        return False
    
    def _get_manager_status(self, integration_type: IntegrationType) -> Dict[str, Any]:
        """Get manager-specific status"""
        if integration_type == IntegrationType.AUTHENTICATION:
            if self.auth_manager:
                return {
                    "provider": self.auth_manager.config.provider.value,
                    "session_count": len(self.auth_manager.sessions)
                }
        elif integration_type == IntegrationType.PROJECT_MANAGEMENT:
            if self.project_manager:
                return {
                    "integration_count": len(self.project_manager.integrations)
                }
        elif integration_type == IntegrationType.COMMUNICATION:
            if self.communication_manager:
                return {
                    "platform_count": len(self.communication_manager.platforms),
                    "rule_count": len(self.communication_manager.notification_rules)
                }
        elif integration_type == IntegrationType.MONITORING:
            if self.monitoring_manager:
                return {
                    "integration_count": len(self.monitoring_manager.integrations),
                    "metric_count": len(self.monitoring_manager.collector.get_metrics())
                }
        
        return {"status": "not_initialized"}
    
    def save_config(self):
        """Save enterprise configuration"""
        config_file = self.cache_dir / "enterprise_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, cache_dir: Optional[str] = None) -> Optional['EnterpriseManager']:
        """Load enterprise configuration"""
        cache_path = Path(cache_dir or ".pomuse/enterprise")
        config_file = cache_path / "enterprise_config.json"
        
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
                
            config = EnterpriseConfig(**config_data)
            return cls(config, str(cache_path))
        
        return None

async def run_enterprise_setup_interactive() -> EnterpriseManager:
    """Interactive setup for enterprise integrations"""
    print("🏢 Enterprise Integration Setup")
    print("=" * 50)
    
    # Basic organization info
    org_name = input("Organization name: ").strip()
    domain = input("Organization domain: ").strip()
    
    config = EnterpriseConfig(
        organization_name=org_name,
        domain=domain
    )
    
    # Setup authentication
    setup_auth = input("Setup authentication integration? (y/n): ").lower() == 'y'
    if setup_auth:
        print("\nAuthentication Providers:")
        print("1. LDAP/Active Directory")
        print("2. SAML")
        print("3. OAuth 2.0")
        print("4. Azure AD")
        
        auth_choice = input("Select provider (1-4): ").strip()
        
        if auth_choice == "1":
            config.authentication = {
                "provider": "ldap",
                "ldap": {
                    "server": input("LDAP server: ").strip(),
                    "port": int(input("LDAP port (389): ") or "389"),
                    "use_ssl": input("Use SSL? (y/n): ").lower() == 'y',
                    "base_dn": input("Base DN: ").strip(),
                    "bind_dn": input("Bind DN: ").strip(),
                    "bind_password": input("Bind password: ").strip()
                }
            }
        elif auth_choice == "2":
            config.authentication = {
                "provider": "saml",
                "saml": {
                    "entity_id": input("Entity ID: ").strip(),
                    "sso_url": input("SSO URL: ").strip(),
                    "x509_cert": input("X.509 Certificate path: ").strip()
                }
            }
        elif auth_choice in ["3", "4"]:
            provider = "oauth2" if auth_choice == "3" else "azure_ad"
            config.authentication = {
                "provider": provider,
                "oauth": {
                    "client_id": input("Client ID: ").strip(),
                    "client_secret": input("Client Secret: ").strip(),
                    "authorization_url": input("Authorization URL: ").strip(),
                    "token_url": input("Token URL: ").strip(),
                    "userinfo_url": input("User Info URL: ").strip(),
                    "redirect_uri": input("Redirect URI: ").strip()
                }
            }
    
    # Setup project management
    setup_pm = input("\nSetup project management integration? (y/n): ").lower() == 'y'
    if setup_pm:
        print("\nProject Management Systems:")
        print("1. Jira")
        print("2. Azure DevOps")
        print("3. Trello")
        
        pm_choice = input("Select system (1-3): ").strip()
        
        if pm_choice == "1":
            config.project_management = {
                "jira_main": {
                    "type": "jira",
                    "base_url": input("Jira URL: ").strip(),
                    "auth": {
                        "type": "token",
                        "email": input("Email: ").strip(),
                        "token": input("API Token: ").strip()
                    }
                }
            }
        elif pm_choice == "2":
            config.project_management = {
                "azure_devops": {
                    "type": "azure_devops",
                    "organization": input("Organization: ").strip(),
                    "auth": {
                        "personal_access_token": input("Personal Access Token: ").strip()
                    }
                }
            }
        elif pm_choice == "3":
            config.project_management = {
                "trello": {
                    "type": "trello",
                    "auth": {
                        "api_key": input("API Key: ").strip(),
                        "token": input("Token: ").strip()
                    }
                }
            }
    
    # Setup communication
    setup_comm = input("\nSetup communication integration? (y/n): ").lower() == 'y'
    if setup_comm:
        config.communication = {}
        
        setup_slack = input("Setup Slack? (y/n): ").lower() == 'y'
        if setup_slack:
            config.communication["slack"] = {
                "main": {
                    "webhook_url": input("Slack webhook URL: ").strip()
                }
            }
        
        setup_teams = input("Setup Microsoft Teams? (y/n): ").lower() == 'y'
        if setup_teams:
            config.communication["teams"] = {
                "main": {
                    "webhook_url": input("Teams webhook URL: ").strip()
                }
            }
    
    # Setup monitoring
    setup_monitoring = input("\nSetup monitoring integration? (y/n): ").lower() == 'y'
    if setup_monitoring:
        config.monitoring = {}
        
        print("\nMonitoring Platforms:")
        print("1. Datadog")
        print("2. New Relic")
        print("3. Prometheus")
        
        monitoring_choice = input("Select platform (1-3): ").strip()
        
        if monitoring_choice == "1":
            config.monitoring["datadog"] = {
                "api_key": input("Datadog API Key: ").strip(),
                "app_key": input("Datadog App Key: ").strip()
            }
        elif monitoring_choice == "2":
            config.monitoring["newrelic"] = {
                "api_key": input("New Relic API Key: ").strip()
            }
        elif monitoring_choice == "3":
            config.monitoring["prometheus"] = {
                "pushgateway_url": input("Pushgateway URL (http://localhost:9091): ").strip() or "http://localhost:9091",
                "job_name": "myndra"
            }
    
    # Create and initialize manager
    manager = EnterpriseManager(config)
    
    print("\n🚀 Initializing enterprise integrations...")
    results = await manager.initialize_all()
    
    for integration, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {integration.replace('_', ' ').title()}")
    
    # Save configuration
    manager.save_config()
    print(f"\n💾 Configuration saved to {manager.cache_dir}")
    
    return manager