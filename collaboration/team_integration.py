"""
Team Integration Module for PomegranteMuse
Integrates team collaboration with code generation, security, and project management
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .team_manager import (
    TeamManager, UserRole, PermissionType, ActivityType,
    display_team_info, display_review_dashboard
)


@dataclass
class WorkflowRule:
    """Workflow automation rule"""
    rule_id: str
    name: str
    trigger: str  # code_generated, security_scan_failed, cost_exceeded
    conditions: Dict[str, Any]
    actions: List[str]  # require_review, auto_approve, notify_team
    enabled: bool = True


@dataclass
class TeamSettings:
    """Team-wide settings"""
    require_code_review: bool = True
    min_reviewers: int = 1
    auto_assign_reviewers: bool = True
    enable_security_gates: bool = True
    cost_alert_threshold: float = 100.0
    workflow_rules: List[WorkflowRule] = None
    
    def __post_init__(self):
        if self.workflow_rules is None:
            self.workflow_rules = []


class TeamIntegration:
    """Main team integration class"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.team_manager = TeamManager(project_root)
        self.settings_file = self.project_root / ".pomuse" / "team" / "settings.json"
        self.settings = self._load_team_settings()
    
    def _load_team_settings(self) -> TeamSettings:
        """Load team settings"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                
                # Convert workflow rules
                workflow_rules = []
                for rule_data in data.get("workflow_rules", []):
                    workflow_rules.append(WorkflowRule(**rule_data))
                
                data["workflow_rules"] = workflow_rules
                return TeamSettings(**data)
                
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load team settings: {e}")
        
        return TeamSettings()
    
    def save_team_settings(self):
        """Save team settings"""
        data = asdict(self.settings)
        
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def handle_code_generation(
        self, 
        generated_code: str, 
        source_analysis: Dict[str, Any],
        user_prompt: str,
        author_id: str = None
    ) -> Dict[str, Any]:
        """Handle code generation with team workflow"""
        author_id = author_id or self.team_manager.current_user_id
        
        # Check if user has permission to generate code
        if not self.team_manager.check_permission(author_id, PermissionType.CODE_GENERATE):
            return {
                "success": False,
                "error": "Insufficient permissions to generate code",
                "requires_permission": PermissionType.CODE_GENERATE.value
            }
        
        # Update user activity
        self.team_manager.update_member_activity(author_id)
        
        result = {
            "success": True,
            "generated_code": generated_code,
            "requires_review": False,
            "review_request_id": None,
            "workflow_actions": []
        }
        
        # Check if code review is required
        if self.settings.require_code_review:
            # Create review request
            author = self.team_manager.team.members.get(author_id)
            review_title = f"Code generation: {user_prompt[:50]}..."
            review_description = f"Generated code for: {user_prompt}"
            
            review_request_id = self.team_manager.create_review_request(
                title=review_title,
                description=review_description,
                generated_code=generated_code,
                source_analysis=source_analysis,
                author_id=author_id
            )
            
            if review_request_id:
                result["requires_review"] = True
                result["review_request_id"] = review_request_id
                result["workflow_actions"].append("review_request_created")
                
                # Trigger workflow automation
                await self._trigger_workflow("code_generated", {
                    "author_id": author_id,
                    "review_request_id": review_request_id,
                    "code_length": len(generated_code)
                })
        
        return result
    
    async def handle_security_scan_result(
        self, 
        scan_result: Dict[str, Any], 
        user_id: str = None
    ) -> Dict[str, Any]:
        """Handle security scan results with team workflow"""
        user_id = user_id or self.team_manager.current_user_id
        
        # Check permissions
        if not self.team_manager.check_permission(user_id, PermissionType.SECURITY_SCAN):
            return {
                "success": False,
                "error": "Insufficient permissions to run security scans"
            }
        
        # Update user activity
        self.team_manager.update_member_activity(user_id)
        
        result = {
            "success": True,
            "scan_result": scan_result,
            "workflow_actions": [],
            "notifications": []
        }
        
        # Check for critical vulnerabilities
        if scan_result.get("success"):
            vulnerabilities = scan_result.get("scan_result", {}).get("vulnerabilities", [])
            critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
            
            if critical_vulns and self.settings.enable_security_gates:
                # Trigger workflow for security failures
                await self._trigger_workflow("security_scan_failed", {
                    "user_id": user_id,
                    "critical_vulnerabilities": len(critical_vulns),
                    "total_vulnerabilities": len(vulnerabilities)
                })
                
                result["workflow_actions"].append("security_gate_triggered")
                result["notifications"].append({
                    "type": "security_alert",
                    "message": f"Critical security vulnerabilities detected: {len(critical_vulns)}",
                    "severity": "high"
                })
        
        return result
    
    async def handle_cost_analysis_result(
        self, 
        cost_result: Dict[str, Any], 
        user_id: str = None
    ) -> Dict[str, Any]:
        """Handle cost analysis results with team workflow"""
        user_id = user_id or self.team_manager.current_user_id
        
        # Check permissions
        if not self.team_manager.check_permission(user_id, PermissionType.COST_VIEW):
            return {
                "success": False,
                "error": "Insufficient permissions to view cost analysis"
            }
        
        # Update user activity
        self.team_manager.update_member_activity(user_id)
        
        result = {
            "success": True,
            "cost_result": cost_result,
            "workflow_actions": [],
            "notifications": []
        }
        
        # Check cost thresholds
        if cost_result.get("success"):
            monthly_cost = cost_result.get("summary", {}).get("current_monthly_cost", 0)
            
            if monthly_cost > self.settings.cost_alert_threshold:
                # Trigger cost workflow
                await self._trigger_workflow("cost_exceeded", {
                    "user_id": user_id,
                    "monthly_cost": monthly_cost,
                    "threshold": self.settings.cost_alert_threshold
                })
                
                result["workflow_actions"].append("cost_alert_triggered")
                result["notifications"].append({
                    "type": "cost_alert",
                    "message": f"Monthly cost (${monthly_cost:.2f}) exceeds threshold (${self.settings.cost_alert_threshold:.2f})",
                    "severity": "medium"
                })
        
        return result
    
    async def _trigger_workflow(self, trigger: str, context: Dict[str, Any]):
        """Trigger workflow automation based on events"""
        for rule in self.settings.workflow_rules:
            if not rule.enabled or rule.trigger != trigger:
                continue
            
            # Check conditions
            if self._evaluate_conditions(rule.conditions, context):
                await self._execute_workflow_actions(rule.actions, context)
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate workflow conditions"""
        for condition, expected_value in conditions.items():
            if condition in context:
                actual_value = context[condition]
                
                # Handle different comparison types
                if isinstance(expected_value, dict):
                    operator = expected_value.get("op", "eq")
                    value = expected_value.get("value")
                    
                    if operator == "gt" and actual_value <= value:
                        return False
                    elif operator == "lt" and actual_value >= value:
                        return False
                    elif operator == "eq" and actual_value != value:
                        return False
                else:
                    if actual_value != expected_value:
                        return False
            else:
                return False  # Required condition not met
        
        return True
    
    async def _execute_workflow_actions(self, actions: List[str], context: Dict[str, Any]):
        """Execute workflow actions"""
        for action in actions:
            if action == "require_review":
                await self._action_require_review(context)
            elif action == "auto_approve":
                await self._action_auto_approve(context)
            elif action == "notify_team":
                await self._action_notify_team(context)
            elif action == "block_deployment":
                await self._action_block_deployment(context)
    
    async def _action_require_review(self, context: Dict[str, Any]):
        """Action: Require additional review"""
        review_request_id = context.get("review_request_id")
        if review_request_id:
            # Add additional reviewers or mark as requiring senior review
            print(f"🔒 Additional review required for {review_request_id}")
    
    async def _action_auto_approve(self, context: Dict[str, Any]):
        """Action: Auto-approve under certain conditions"""
        review_request_id = context.get("review_request_id")
        if review_request_id:
            # Auto-approve for simple changes
            print(f"✅ Auto-approved {review_request_id}")
    
    async def _action_notify_team(self, context: Dict[str, Any]):
        """Action: Notify team members"""
        # In a real implementation, this would send notifications
        print(f"📢 Team notification: {context}")
    
    async def _action_block_deployment(self, context: Dict[str, Any]):
        """Action: Block deployment due to issues"""
        print(f"🚫 Deployment blocked due to workflow conditions")
    
    def add_workflow_rule(self, rule: WorkflowRule) -> bool:
        """Add a new workflow rule"""
        try:
            # Check if rule already exists
            for existing_rule in self.settings.workflow_rules:
                if existing_rule.rule_id == rule.rule_id:
                    print(f"❌ Workflow rule {rule.rule_id} already exists")
                    return False
            
            self.settings.workflow_rules.append(rule)
            self.save_team_settings()
            print(f"✅ Added workflow rule: {rule.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add workflow rule: {e}")
            return False
    
    def remove_workflow_rule(self, rule_id: str) -> bool:
        """Remove a workflow rule"""
        try:
            self.settings.workflow_rules = [
                rule for rule in self.settings.workflow_rules 
                if rule.rule_id != rule_id
            ]
            self.save_team_settings()
            print(f"✅ Removed workflow rule: {rule_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to remove workflow rule: {e}")
            return False
    
    def get_team_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive team dashboard"""
        team_summary = self.team_manager.get_team_summary()
        review_dashboard = self.team_manager.get_review_dashboard()
        
        # Recent team activity
        recent_activities = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for activity in self.team_manager.team.activity_log[-20:]:  # Last 20 activities
            if datetime.fromisoformat(activity.timestamp) > cutoff_date:
                recent_activities.append({
                    "type": activity.activity_type.value,
                    "description": activity.description,
                    "username": activity.username,
                    "timestamp": activity.timestamp
                })
        
        # Team productivity metrics
        productivity_metrics = self._calculate_productivity_metrics()
        
        return {
            "team_info": team_summary,
            "review_summary": review_dashboard,
            "recent_activities": recent_activities,
            "productivity_metrics": productivity_metrics,
            "settings": {
                "require_code_review": self.settings.require_code_review,
                "min_reviewers": self.settings.min_reviewers,
                "auto_assign_reviewers": self.settings.auto_assign_reviewers,
                "enable_security_gates": self.settings.enable_security_gates,
                "cost_alert_threshold": self.settings.cost_alert_threshold
            },
            "workflow_rules": len(self.settings.workflow_rules)
        }
    
    def _calculate_productivity_metrics(self) -> Dict[str, Any]:
        """Calculate team productivity metrics"""
        # Activities in last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_activities = [
            activity for activity in self.team_manager.team.activity_log
            if datetime.fromisoformat(activity.timestamp) > cutoff_date
        ]
        
        # Code generation activity
        code_generations = len([
            activity for activity in recent_activities
            if activity.activity_type == ActivityType.CODE_GENERATED
        ])
        
        # Review activity
        reviews_completed = len([
            activity for activity in recent_activities
            if activity.activity_type == ActivityType.CODE_REVIEWED
        ])
        
        # Average review time (simplified calculation)
        completed_reviews = [
            req for req in self.team_manager.team.review_requests.values()
            if req.status in ["approved", "merged"]
        ]
        
        avg_review_time_hours = 0
        if completed_reviews:
            total_review_time = 0
            for review in completed_reviews:
                created = datetime.fromisoformat(review.created_at)
                approved = datetime.fromisoformat(
                    review.metadata.get("approved_at", review.created_at)
                )
                total_review_time += (approved - created).total_seconds() / 3600
            
            avg_review_time_hours = total_review_time / len(completed_reviews)
        
        return {
            "code_generations_last_30_days": code_generations,
            "reviews_completed_last_30_days": reviews_completed,
            "average_review_time_hours": round(avg_review_time_hours, 2),
            "active_members": len([
                member for member in self.team_manager.team.members.values()
                if member.is_active
            ]),
            "productivity_score": self._calculate_productivity_score(
                code_generations, reviews_completed, avg_review_time_hours
            )
        }
    
    def _calculate_productivity_score(
        self, 
        code_generations: int, 
        reviews_completed: int, 
        avg_review_time: float
    ) -> int:
        """Calculate team productivity score (0-100)"""
        score = 50  # Base score
        
        # Adjust based on code generation activity
        if code_generations > 20:
            score += 20
        elif code_generations > 10:
            score += 10
        elif code_generations > 5:
            score += 5
        
        # Adjust based on review activity
        if reviews_completed > 15:
            score += 15
        elif reviews_completed > 10:
            score += 10
        elif reviews_completed > 5:
            score += 5
        
        # Adjust based on review speed
        if avg_review_time < 24:  # Less than 24 hours
            score += 15
        elif avg_review_time < 48:  # Less than 48 hours
            score += 10
        elif avg_review_time > 120:  # More than 5 days
            score -= 20
        
        return max(0, min(100, score))
    
    def configure_team_settings(self, **kwargs) -> bool:
        """Configure team settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                else:
                    print(f"Warning: Unknown setting '{key}'")
            
            self.save_team_settings()
            print("✅ Team settings updated")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update team settings: {e}")
            return False


# CLI interface functions
async def run_team_management_interactive(project_root: str):
    """Interactive team management interface"""
    integration = TeamIntegration(project_root)
    team_manager = integration.team_manager
    
    print("👥 PomegranteMuse Team Management")
    print("=" * 50)
    
    # Management options
    print("\nTeam management options:")
    print("1. View team dashboard")
    print("2. Manage team members")
    print("3. Review management")
    print("4. Team settings")
    print("5. Workflow rules")
    print("6. Activity reports")
    
    try:
        choice = int(input("\nSelect option (1-6): "))
        
        if choice == 1:
            # Team dashboard
            dashboard = integration.get_team_dashboard()
            
            print(f"\n👥 Team Dashboard")
            team_info = dashboard["team_info"]
            print(f"   Team: {team_info['team_name']}")
            print(f"   Members: {team_info['total_members']} ({team_info['active_members']} active)")
            
            # Review summary
            review_summary = dashboard["review_summary"]
            print(f"\n📋 Reviews: {review_summary['total_reviews']}")
            for status, count in review_summary["reviews_by_status"].items():
                print(f"   {status.title()}: {count}")
            
            # Productivity metrics
            metrics = dashboard["productivity_metrics"]
            print(f"\n📊 Productivity (last 30 days):")
            print(f"   Code Generations: {metrics['code_generations_last_30_days']}")
            print(f"   Reviews Completed: {metrics['reviews_completed_last_30_days']}")
            print(f"   Avg Review Time: {metrics['average_review_time_hours']:.1f} hours")
            print(f"   Productivity Score: {metrics['productivity_score']}/100")
            
            # Recent activities
            if dashboard["recent_activities"]:
                print(f"\n🕐 Recent Activities:")
                for activity in dashboard["recent_activities"][-5:]:
                    print(f"   • {activity['username']}: {activity['description']}")
        
        elif choice == 2:
            # Member management
            print("\n👤 Member Management:")
            print("1. Add member")
            print("2. Remove member")
            print("3. Change member role")
            print("4. List members")
            
            sub_choice = int(input("Select action (1-4): "))
            
            if sub_choice == 1:
                username = input("Username: ").strip()
                email = input("Email: ").strip()
                role_input = input("Role [developer/reviewer/maintainer/admin]: ").strip().lower()
                
                role_mapping = {
                    "developer": UserRole.DEVELOPER,
                    "reviewer": UserRole.REVIEWER,
                    "maintainer": UserRole.MAINTAINER,
                    "admin": UserRole.ADMIN
                }
                
                role = role_mapping.get(role_input, UserRole.DEVELOPER)
                success = team_manager.add_member(username, email, role)
                
            elif sub_choice == 2:
                username = input("Username to remove: ").strip()
                # Find user by username
                user_id = None
                for uid, member in team_manager.team.members.items():
                    if member.username == username:
                        user_id = uid
                        break
                
                if user_id:
                    success = team_manager.remove_member(user_id)
                else:
                    print("❌ User not found")
            
            elif sub_choice == 3:
                username = input("Username: ").strip()
                new_role_input = input("New role [developer/reviewer/maintainer/admin]: ").strip().lower()
                
                role_mapping = {
                    "developer": UserRole.DEVELOPER,
                    "reviewer": UserRole.REVIEWER,
                    "maintainer": UserRole.MAINTAINER,
                    "admin": UserRole.ADMIN
                }
                
                new_role = role_mapping.get(new_role_input)
                if not new_role:
                    print("❌ Invalid role")
                    return
                
                # Find user by username
                user_id = None
                for uid, member in team_manager.team.members.items():
                    if member.username == username:
                        user_id = uid
                        break
                
                if user_id:
                    success = team_manager.change_member_role(user_id, new_role)
                else:
                    print("❌ User not found")
            
            elif sub_choice == 4:
                print("\n👥 Team Members:")
                for member in team_manager.team.members.values():
                    status = "active" if member.is_active else "inactive"
                    print(f"   • {member.username} ({member.role.value}) - {status}")
                    print(f"     Email: {member.email}")
                    print(f"     Joined: {member.joined_at[:10]}")
        
        elif choice == 3:
            # Review management
            display_review_dashboard(team_manager)
            
            if team_manager.team.review_requests:
                print("\nReview actions:")
                print("1. Add comment to review")
                print("2. Approve review")
                
                action = int(input("Select action (1-2): "))
                
                if action == 1:
                    request_id = input("Review request ID: ").strip()
                    comment = input("Comment: ").strip()
                    
                    success = team_manager.add_review_comment(request_id, comment)
                
                elif action == 2:
                    request_id = input("Review request ID: ").strip()
                    success = team_manager.approve_review_request(request_id)
        
        elif choice == 4:
            # Team settings
            settings = integration.settings
            print(f"\n⚙️  Current Team Settings:")
            print(f"   Require Code Review: {settings.require_code_review}")
            print(f"   Min Reviewers: {settings.min_reviewers}")
            print(f"   Auto Assign Reviewers: {settings.auto_assign_reviewers}")
            print(f"   Enable Security Gates: {settings.enable_security_gates}")
            print(f"   Cost Alert Threshold: ${settings.cost_alert_threshold:.2f}")
            
            modify = input("\nModify settings? [y/N]: ").strip().lower()
            if modify == 'y':
                new_settings = {}
                
                require_review = input(f"Require code review? [{settings.require_code_review}]: ").strip()
                if require_review.lower() in ['true', 'false']:
                    new_settings['require_code_review'] = require_review.lower() == 'true'
                
                min_reviewers = input(f"Minimum reviewers [{settings.min_reviewers}]: ").strip()
                if min_reviewers.isdigit():
                    new_settings['min_reviewers'] = int(min_reviewers)
                
                cost_threshold = input(f"Cost alert threshold [{settings.cost_alert_threshold}]: ").strip()
                if cost_threshold.replace('.', '').isdigit():
                    new_settings['cost_alert_threshold'] = float(cost_threshold)
                
                if new_settings:
                    integration.configure_team_settings(**new_settings)
        
        elif choice == 5:
            # Workflow rules
            print(f"\n🔄 Workflow Rules ({len(integration.settings.workflow_rules)}):")
            for rule in integration.settings.workflow_rules:
                status = "enabled" if rule.enabled else "disabled"
                print(f"   • {rule.name} ({rule.trigger}) - {status}")
            
            print("\nWorkflow actions:")
            print("1. Add workflow rule")
            print("2. Remove workflow rule")
            
            action = int(input("Select action (1-2): "))
            
            if action == 1:
                name = input("Rule name: ").strip()
                trigger = input("Trigger [code_generated/security_scan_failed/cost_exceeded]: ").strip()
                
                rule = WorkflowRule(
                    rule_id=f"rule_{len(integration.settings.workflow_rules) + 1}",
                    name=name,
                    trigger=trigger,
                    conditions={},
                    actions=["notify_team"]
                )
                
                integration.add_workflow_rule(rule)
            
            elif action == 2:
                rule_id = input("Rule ID to remove: ").strip()
                integration.remove_workflow_rule(rule_id)
        
        elif choice == 6:
            # Activity reports
            print("\n📊 Activity Reports:")
            print("1. Team activity summary")
            print("2. Member activity report")
            
            report_choice = int(input("Select report (1-2): "))
            
            if report_choice == 1:
                dashboard = integration.get_team_dashboard()
                metrics = dashboard["productivity_metrics"]
                
                print(f"\n📈 Team Activity Summary (last 30 days):")
                print(f"   Code Generations: {metrics['code_generations_last_30_days']}")
                print(f"   Reviews Completed: {metrics['reviews_completed_last_30_days']}")
                print(f"   Average Review Time: {metrics['average_review_time_hours']:.1f} hours")
                print(f"   Team Productivity Score: {metrics['productivity_score']}/100")
            
            elif report_choice == 2:
                username = input("Username: ").strip()
                # Find user by username
                user_id = None
                for uid, member in team_manager.team.members.items():
                    if member.username == username:
                        user_id = uid
                        break
                
                if user_id:
                    activity = team_manager.get_member_activity(user_id, 30)
                    
                    print(f"\n👤 Activity Report for {username}:")
                    member_info = activity["member_info"]
                    print(f"   Role: {member_info['role']}")
                    print(f"   Joined: {member_info['joined_at'][:10]}")
                    print(f"   Last Active: {member_info['last_active'][:10]}")
                    
                    summary = activity["activity_summary"]
                    print(f"\n📊 Activity Summary (last 30 days):")
                    print(f"   Total Activities: {summary['total_activities']}")
                    print(f"   Authored Reviews: {summary['authored_reviews']}")
                    print(f"   Reviewed Requests: {summary['reviewed_requests']}")
                    
                    if summary["activity_counts"]:
                        print(f"\n🎯 Activity Breakdown:")
                        for activity_type, count in summary["activity_counts"].items():
                            print(f"   {activity_type.replace('_', ' ').title()}: {count}")
                else:
                    print("❌ User not found")
        
    except (ValueError, KeyboardInterrupt):
        print("\nTeam management cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_team_management_interactive(project_path))