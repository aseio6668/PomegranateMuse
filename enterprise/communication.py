"""
Communication Platform Integration for MyndraComposer
Integrates with Slack, Microsoft Teams, email, and other communication platforms
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class NotificationType(Enum):
    """Types of notifications"""
    BUILD_SUCCESS = "build_success"
    BUILD_FAILURE = "build_failure"
    DEPLOYMENT_SUCCESS = "deployment_success"
    DEPLOYMENT_FAILURE = "deployment_failure"
    CODE_REVIEW_REQUEST = "code_review_request"
    CODE_REVIEW_APPROVED = "code_review_approved"
    CODE_REVIEW_REJECTED = "code_review_rejected"
    MIGRATION_STARTED = "migration_started"
    MIGRATION_COMPLETED = "migration_completed"
    SECURITY_ALERT = "security_alert"
    COST_ALERT = "cost_alert"
    TEAM_MENTION = "team_mention"
    SYSTEM_ERROR = "system_error"
    CUSTOM = "custom"

class NotificationChannel(Enum):
    """Communication channels"""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"

@dataclass
class NotificationTemplate:
    """Template for notifications"""
    notification_type: NotificationType
    title_template: str
    message_template: str
    color: str = "#007bff"  # Default blue
    emoji: str = "🔔"
    include_attachments: bool = False
    mention_users: List[str] = None
    
    def __post_init__(self):
        if self.mention_users is None:
            self.mention_users = []

@dataclass
class EmailNotification:
    """Email notification configuration"""
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    from_email: str = ""
    from_name: str = "MyndraComposer"

class CommunicationPlatform:
    """Base class for communication platform integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def send_message(self, channel: str, message: str, **kwargs) -> bool:
        """Send message to channel"""
        raise NotImplementedError
        
    async def send_notification(self, notification_type: NotificationType, 
                              data: Dict[str, Any], **kwargs) -> bool:
        """Send structured notification"""
        raise NotImplementedError
        
    async def upload_file(self, channel: str, file_path: str, 
                         comment: str = "") -> bool:
        """Upload file to channel"""
        raise NotImplementedError

class SlackIntegration(CommunicationPlatform):
    """Slack integration implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bot_token = config.get("bot_token")
        self.webhook_url = config.get("webhook_url")
        self.api_url = "https://slack.com/api"
        
        if not REQUESTS_AVAILABLE:
            self.logger.error("Requests library required for Slack integration")
            
    async def send_message(self, channel: str, message: str, **kwargs) -> bool:
        """Send message to Slack channel"""
        if not REQUESTS_AVAILABLE:
            return False
            
        try:
            if self.webhook_url:
                return await self._send_webhook_message(message, **kwargs)
            elif self.bot_token:
                return await self._send_api_message(channel, message, **kwargs)
            else:
                self.logger.error("No Slack token or webhook URL configured")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def _send_webhook_message(self, message: str, **kwargs) -> bool:
        """Send message via Slack webhook"""
        payload = {
            "text": message,
            "username": kwargs.get("username", "MyndraComposer"),
            "icon_emoji": kwargs.get("emoji", ":robot_face:")
        }
        
        if kwargs.get("channel"):
            payload["channel"] = kwargs["channel"]
            
        if kwargs.get("attachments"):
            payload["attachments"] = kwargs["attachments"]
            
        response = requests.post(self.webhook_url, json=payload)
        return response.status_code == 200
    
    async def _send_api_message(self, channel: str, message: str, **kwargs) -> bool:
        """Send message via Slack API"""
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "channel": channel,
            "text": message
        }
        
        if kwargs.get("blocks"):
            payload["blocks"] = kwargs["blocks"]
            
        if kwargs.get("attachments"):
            payload["attachments"] = kwargs["attachments"]
            
        response = requests.post(f"{self.api_url}/chat.postMessage", 
                               headers=headers, json=payload)
        return response.status_code == 200
    
    async def send_notification(self, notification_type: NotificationType,
                              data: Dict[str, Any], **kwargs) -> bool:
        """Send structured notification to Slack"""
        template = self._get_notification_template(notification_type)
        
        # Build Slack-specific message format
        title = template.title_template.format(**data)
        message = template.message_template.format(**data)
        
        # Create Slack attachment
        attachment = {
            "color": template.color,
            "title": title,
            "text": message,
            "timestamp": int(datetime.now().timestamp()),
            "footer": "MyndraComposer",
            "footer_icon": "https://example.com/myndra-icon.png"
        }
        
        # Add fields for structured data
        fields = []
        if data.get("project"):
            fields.append({"title": "Project", "value": data["project"], "short": True})
        if data.get("environment"):
            fields.append({"title": "Environment", "value": data["environment"], "short": True})
        if data.get("duration"):
            fields.append({"title": "Duration", "value": data["duration"], "short": True})
            
        if fields:
            attachment["fields"] = fields
            
        # Send message
        channel = kwargs.get("channel", "#general")
        return await self.send_message(channel, "", attachments=[attachment])
    
    async def upload_file(self, channel: str, file_path: str, comment: str = "") -> bool:
        """Upload file to Slack channel"""
        if not self.bot_token or not REQUESTS_AVAILABLE:
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.bot_token}"}
            
            with open(file_path, 'rb') as file:
                files = {"file": file}
                data = {
                    "channels": channel,
                    "initial_comment": comment,
                    "filename": Path(file_path).name
                }
                
                response = requests.post(f"{self.api_url}/files.upload",
                                       headers=headers, files=files, data=data)
                return response.status_code == 200
                
        except Exception as e:
            self.logger.error(f"Failed to upload file to Slack: {e}")
            return False
    
    def _get_notification_template(self, notification_type: NotificationType) -> NotificationTemplate:
        """Get notification template for type"""
        templates = {
            NotificationType.BUILD_SUCCESS: NotificationTemplate(
                notification_type=notification_type,
                title_template="✅ Build Successful",
                message_template="Build completed successfully for {project} in {duration}",
                color="#28a745",
                emoji="✅"
            ),
            NotificationType.BUILD_FAILURE: NotificationTemplate(
                notification_type=notification_type,
                title_template="❌ Build Failed",
                message_template="Build failed for {project}. Error: {error}",
                color="#dc3545",
                emoji="❌"
            ),
            NotificationType.CODE_REVIEW_REQUEST: NotificationTemplate(
                notification_type=notification_type,
                title_template="👀 Code Review Requested",
                message_template="New code review requested by {author} for {project}",
                color="#007bff",
                emoji="👀"
            ),
            NotificationType.MIGRATION_COMPLETED: NotificationTemplate(
                notification_type=notification_type,
                title_template="🚀 Migration Completed",
                message_template="Code migration completed for {project}. {files_migrated} files processed.",
                color="#28a745",
                emoji="🚀"
            ),
            NotificationType.SECURITY_ALERT: NotificationTemplate(
                notification_type=notification_type,
                title_template="🚨 Security Alert",
                message_template="Security vulnerability detected in {project}: {vulnerability}",
                color="#ffc107",
                emoji="🚨"
            )
        }
        
        return templates.get(notification_type, NotificationTemplate(
            notification_type=notification_type,
            title_template="📢 Notification",
            message_template="{message}",
            color="#007bff",
            emoji="📢"
        ))

class TeamsIntegration(CommunicationPlatform):
    """Microsoft Teams integration implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        
    async def send_message(self, channel: str, message: str, **kwargs) -> bool:
        """Send message to Teams channel"""
        if not REQUESTS_AVAILABLE or not self.webhook_url:
            return False
            
        try:
            payload = {
                "text": message,
                "title": kwargs.get("title", "MyndraComposer Notification")
            }
            
            if kwargs.get("color"):
                payload["themeColor"] = kwargs["color"]
                
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to send Teams message: {e}")
            return False
    
    async def send_notification(self, notification_type: NotificationType,
                              data: Dict[str, Any], **kwargs) -> bool:
        """Send structured notification to Teams"""
        template = self._get_notification_template(notification_type)
        
        title = template.title_template.format(**data)
        message = template.message_template.format(**data)
        
        # Create Teams card format
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": title,
            "themeColor": template.color.replace("#", ""),
            "sections": [{
                "activityTitle": title,
                "activitySubtitle": "MyndraComposer",
                "activityImage": "https://example.com/myndra-icon.png",
                "text": message,
                "facts": []
            }]
        }
        
        # Add facts for structured data
        facts = card["sections"][0]["facts"]
        if data.get("project"):
            facts.append({"name": "Project", "value": data["project"]})
        if data.get("environment"):
            facts.append({"name": "Environment", "value": data["environment"]})
        if data.get("duration"):
            facts.append({"name": "Duration", "value": data["duration"]})
            
        response = requests.post(self.webhook_url, json=card)
        return response.status_code == 200
    
    def _get_notification_template(self, notification_type: NotificationType) -> NotificationTemplate:
        """Get notification template for Teams"""
        # Similar to Slack templates but with Teams-specific formatting
        templates = {
            NotificationType.BUILD_SUCCESS: NotificationTemplate(
                notification_type=notification_type,
                title_template="Build Successful",
                message_template="Build completed successfully for {project} in {duration}",
                color="28a745"
            ),
            NotificationType.BUILD_FAILURE: NotificationTemplate(
                notification_type=notification_type,
                title_template="Build Failed",
                message_template="Build failed for {project}. Error: {error}",
                color="dc3545"
            )
        }
        
        return templates.get(notification_type, NotificationTemplate(
            notification_type=notification_type,
            title_template="Notification",
            message_template="{message}",
            color="007bff"
        ))

class EmailNotificationService:
    """Email notification service"""
    
    def __init__(self, config: EmailNotification):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def send_email(self, to_emails: List[str], subject: str, 
                        body: str, html_body: str = None, 
                        attachments: List[str] = None) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
            msg['To'] = ", ".join(to_emails)
            
            # Add text body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
                
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if Path(file_path).exists():
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {Path(file_path).name}'
                        )
                        msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                    
                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)
                    
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

class CommunicationManager:
    """Main communication manager"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/communication")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.platforms: Dict[str, CommunicationPlatform] = {}
        self.email_service: Optional[EmailNotificationService] = None
        self.notification_rules: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
    def add_slack_integration(self, name: str, config: Dict[str, Any]):
        """Add Slack integration"""
        self.platforms[name] = SlackIntegration(config)
        
    def add_teams_integration(self, name: str, config: Dict[str, Any]):
        """Add Microsoft Teams integration"""
        self.platforms[name] = TeamsIntegration(config)
        
    def setup_email_service(self, config: EmailNotification):
        """Setup email notification service"""
        self.email_service = EmailNotificationService(config)
        
    def add_notification_rule(self, notification_type: NotificationType,
                            channels: List[str], conditions: Dict[str, Any] = None):
        """Add notification routing rule"""
        rule = {
            "notification_type": notification_type,
            "channels": channels,
            "conditions": conditions or {}
        }
        self.notification_rules.append(rule)
        
    async def send_notification(self, notification_type: NotificationType,
                              data: Dict[str, Any], **kwargs) -> Dict[str, bool]:
        """Send notification to appropriate channels"""
        results = {}
        
        # Find matching rules
        matching_rules = []
        for rule in self.notification_rules:
            if rule["notification_type"] == notification_type:
                # Check conditions
                conditions_met = True
                for condition_key, condition_value in rule["conditions"].items():
                    if data.get(condition_key) != condition_value:
                        conditions_met = False
                        break
                        
                if conditions_met:
                    matching_rules.append(rule)
        
        # If no specific rules, use default channels
        if not matching_rules:
            matching_rules = [{"channels": list(self.platforms.keys())}]
            
        # Send notifications
        for rule in matching_rules:
            for channel in rule["channels"]:
                if channel in self.platforms:
                    platform = self.platforms[channel]
                    try:
                        success = await platform.send_notification(
                            notification_type, data, **kwargs
                        )
                        results[channel] = success
                    except Exception as e:
                        self.logger.error(f"Failed to send notification to {channel}: {e}")
                        results[channel] = False
                        
        return results
    
    async def send_build_notification(self, success: bool, project: str,
                                    duration: str = "", error: str = "",
                                    environment: str = "development"):
        """Send build completion notification"""
        notification_type = NotificationType.BUILD_SUCCESS if success else NotificationType.BUILD_FAILURE
        
        data = {
            "project": project,
            "duration": duration,
            "error": error,
            "environment": environment,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return await self.send_notification(notification_type, data)
    
    async def send_migration_notification(self, project: str, files_migrated: int,
                                        source_lang: str, target_lang: str,
                                        duration: str = ""):
        """Send migration completion notification"""
        data = {
            "project": project,
            "files_migrated": files_migrated,
            "source_language": source_lang,
            "target_language": target_lang,
            "duration": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return await self.send_notification(NotificationType.MIGRATION_COMPLETED, data)
    
    async def send_security_alert(self, project: str, vulnerability: str,
                                severity: str = "medium", affected_files: List[str] = None):
        """Send security alert notification"""
        data = {
            "project": project,
            "vulnerability": vulnerability,
            "severity": severity,
            "affected_files": len(affected_files) if affected_files else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return await self.send_notification(NotificationType.SECURITY_ALERT, data)
    
    def load_communication_config(self) -> Dict[str, Any]:
        """Load communication configuration"""
        config_file = self.cache_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        return {}
    
    def save_communication_config(self, config: Dict[str, Any]):
        """Save communication configuration"""
        config_file = self.cache_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

async def send_notification(manager: CommunicationManager, 
                          notification_type: NotificationType,
                          data: Dict[str, Any], **kwargs) -> Dict[str, bool]:
    """Convenience function to send notification"""
    return await manager.send_notification(notification_type, data, **kwargs)

async def setup_communication_channels(config: Dict[str, Any]) -> CommunicationManager:
    """Setup communication channels from configuration"""
    manager = CommunicationManager()
    
    # Setup Slack integrations
    for name, slack_config in config.get("slack", {}).items():
        manager.add_slack_integration(name, slack_config)
        
    # Setup Teams integrations
    for name, teams_config in config.get("teams", {}).items():
        manager.add_teams_integration(name, teams_config)
        
    # Setup email service
    if "email" in config:
        email_config = EmailNotification(**config["email"])
        manager.setup_email_service(email_config)
        
    # Setup notification rules
    for rule in config.get("notification_rules", []):
        manager.add_notification_rule(
            NotificationType(rule["notification_type"]),
            rule["channels"],
            rule.get("conditions", {})
        )
        
    return manager