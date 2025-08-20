"""
Progress Tracker for Migration Strategy System
Tracks migration progress, generates reports, and provides dashboards
"""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

class MilestoneStatus(Enum):
    """Milestone status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    BLOCKED = "blocked"

@dataclass
class Milestone:
    """Migration milestone"""
    id: str
    name: str
    description: str
    target_date: datetime
    actual_date: Optional[datetime] = None
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    progress_percentage: float = 0.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class MigrationMetrics:
    """Migration metrics"""
    components_total: int
    components_completed: int
    components_in_progress: int
    components_not_started: int
    lines_of_code_migrated: int
    lines_of_code_total: int
    tests_passing: int
    tests_total: int
    estimated_completion_date: Optional[datetime] = None
    actual_effort_hours: float = 0.0
    estimated_effort_hours: float = 0.0

@dataclass
class ProgressReport:
    """Progress report"""
    project_name: str
    report_date: datetime
    overall_progress: float  # 0.0 - 100.0
    metrics: MigrationMetrics
    milestones: List[Milestone]
    recent_activities: List[str]
    issues: List[str]
    risks: List[str]
    next_steps: List[str]
    team_notes: str = ""

class ProgressTracker:
    """Tracks migration progress"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/progress")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_milestones(self, timeline: Dict[str, datetime]) -> List[Milestone]:
        """Create milestones from timeline"""
        milestones = []
        
        milestone_templates = [
            ("assessment_complete", "Assessment Phase Complete", "Codebase analysis and planning completed"),
            ("planning_complete", "Planning Phase Complete", "Migration strategy and detailed plan finalized"),
            ("preparation_complete", "Preparation Phase Complete", "Environment setup and tooling ready"),
            ("25_percent_complete", "25% Migration Complete", "Quarter of components migrated"),
            ("50_percent_complete", "50% Migration Complete", "Half of components migrated"),
            ("75_percent_complete", "75% Migration Complete", "Three quarters of components migrated"),
            ("migration_complete", "Migration Complete", "All components migrated and tested"),
            ("validation_complete", "Validation Complete", "All validation tests passed"),
            ("deployment_complete", "Deployment Complete", "Production deployment successful")
        ]
        
        base_date = min(timeline.values()) if timeline else datetime.now()
        
        for i, (milestone_id, name, description) in enumerate(milestone_templates):
            target_date = base_date + timedelta(days=i * 14)  # Spread milestones every 2 weeks
            
            milestone = Milestone(
                id=milestone_id,
                name=name,
                description=description,
                target_date=target_date
            )
            milestones.append(milestone)
        
        return milestones
    
    def update_progress(self, project_name: str, metrics: MigrationMetrics,
                       milestones: List[Milestone], activities: List[str] = None,
                       issues: List[str] = None) -> ProgressReport:
        """Update and generate progress report"""
        if activities is None:
            activities = []
        if issues is None:
            issues = []
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress(metrics, milestones)
        
        # Update milestone statuses
        self._update_milestone_statuses(milestones, metrics)
        
        # Identify risks
        risks = self._identify_risks(metrics, milestones)
        
        # Generate next steps
        next_steps = self._generate_next_steps(metrics, milestones)
        
        report = ProgressReport(
            project_name=project_name,
            report_date=datetime.now(),
            overall_progress=overall_progress,
            metrics=metrics,
            milestones=milestones,
            recent_activities=activities[-10:] if activities else [],  # Last 10 activities
            issues=issues,
            risks=risks,
            next_steps=next_steps
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _calculate_overall_progress(self, metrics: MigrationMetrics, milestones: List[Milestone]) -> float:
        """Calculate overall progress percentage"""
        # Weight different aspects of progress
        component_progress = (metrics.components_completed / metrics.components_total * 100) if metrics.components_total > 0 else 0
        code_progress = (metrics.lines_of_code_migrated / metrics.lines_of_code_total * 100) if metrics.lines_of_code_total > 0 else 0
        milestone_progress = (len([m for m in milestones if m.status == MilestoneStatus.COMPLETED]) / len(milestones) * 100) if milestones else 0
        
        # Weighted average
        overall_progress = (component_progress * 0.4 + code_progress * 0.3 + milestone_progress * 0.3)
        
        return min(100.0, max(0.0, overall_progress))
    
    def _update_milestone_statuses(self, milestones: List[Milestone], metrics: MigrationMetrics):
        """Update milestone statuses based on current metrics"""
        now = datetime.now()
        
        for milestone in milestones:
            if milestone.status == MilestoneStatus.COMPLETED:
                continue
            
            # Simple logic to update milestone status based on progress
            if "25_percent" in milestone.id:
                if metrics.components_completed >= metrics.components_total * 0.25:
                    milestone.status = MilestoneStatus.COMPLETED
                    milestone.actual_date = now
                elif metrics.components_in_progress > 0:
                    milestone.status = MilestoneStatus.IN_PROGRESS
                    
            elif "50_percent" in milestone.id:
                if metrics.components_completed >= metrics.components_total * 0.5:
                    milestone.status = MilestoneStatus.COMPLETED
                    milestone.actual_date = now
                elif metrics.components_completed >= metrics.components_total * 0.25:
                    milestone.status = MilestoneStatus.IN_PROGRESS
                    
            elif "75_percent" in milestone.id:
                if metrics.components_completed >= metrics.components_total * 0.75:
                    milestone.status = MilestoneStatus.COMPLETED
                    milestone.actual_date = now
                elif metrics.components_completed >= metrics.components_total * 0.5:
                    milestone.status = MilestoneStatus.IN_PROGRESS
                    
            elif "migration_complete" in milestone.id:
                if metrics.components_completed == metrics.components_total:
                    milestone.status = MilestoneStatus.COMPLETED
                    milestone.actual_date = now
                elif metrics.components_completed >= metrics.components_total * 0.75:
                    milestone.status = MilestoneStatus.IN_PROGRESS
            
            # Check for delays
            if milestone.target_date < now and milestone.status != MilestoneStatus.COMPLETED:
                milestone.status = MilestoneStatus.DELAYED
    
    def _identify_risks(self, metrics: MigrationMetrics, milestones: List[Milestone]) -> List[str]:
        """Identify project risks"""
        risks = []
        
        # Check for delayed milestones
        delayed_milestones = [m for m in milestones if m.status == MilestoneStatus.DELAYED]
        if delayed_milestones:
            risks.append(f"{len(delayed_milestones)} milestones are delayed")
        
        # Check effort variance
        if metrics.actual_effort_hours > metrics.estimated_effort_hours * 1.2:
            risks.append("Actual effort significantly exceeds estimates")
        
        # Check test coverage
        if metrics.tests_total > 0:
            test_pass_rate = metrics.tests_passing / metrics.tests_total
            if test_pass_rate < 0.8:
                risks.append(f"Test pass rate is low ({test_pass_rate:.1%})")
        
        # Check progress rate
        if metrics.components_total > 0:
            completion_rate = metrics.components_completed / metrics.components_total
            if completion_rate < 0.1 and metrics.actual_effort_hours > 40:  # Less than 10% done after 40 hours
                risks.append("Migration progress is slower than expected")
        
        return risks
    
    def _generate_next_steps(self, metrics: MigrationMetrics, milestones: List[Milestone]) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        
        # Find next milestone
        upcoming_milestones = [m for m in milestones if m.status == MilestoneStatus.NOT_STARTED]
        if upcoming_milestones:
            next_milestone = min(upcoming_milestones, key=lambda x: x.target_date)
            next_steps.append(f"Work towards {next_milestone.name} (due {next_milestone.target_date.strftime('%Y-%m-%d')})")
        
        # Check for components in progress
        if metrics.components_in_progress > 0:
            next_steps.append(f"Complete {metrics.components_in_progress} components currently in progress")
        
        # Check for failing tests
        if metrics.tests_total > 0 and metrics.tests_passing < metrics.tests_total:
            failing_tests = metrics.tests_total - metrics.tests_passing
            next_steps.append(f"Fix {failing_tests} failing tests")
        
        # General recommendations
        if not next_steps:
            next_steps.append("Continue with planned migration activities")
        
        return next_steps
    
    def _save_report(self, report: ProgressReport):
        """Save progress report"""
        timestamp = report.report_date.strftime("%Y%m%d_%H%M%S")
        report_file = self.cache_dir / f"progress_report_{report.project_name}_{timestamp}.json"
        
        # Convert to serializable format
        report_data = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Also save as latest
        latest_file = self.cache_dir / f"latest_progress_{report.project_name}.json"
        with open(latest_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

class StatusDashboard:
    """Migration status dashboard"""
    
    def __init__(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker
    
    def generate_dashboard_data(self, project_name: str) -> Dict[str, Any]:
        """Generate dashboard data"""
        # Load latest progress report
        latest_file = self.progress_tracker.cache_dir / f"latest_progress_{project_name}.json"
        
        if not latest_file.exists():
            return {"error": "No progress data available"}
        
        with open(latest_file) as f:
            report_data = json.load(f)
        
        # Calculate dashboard metrics
        metrics = report_data["metrics"]
        milestones = report_data["milestones"]
        
        dashboard_data = {
            "project_name": project_name,
            "last_updated": report_data["report_date"],
            "overall_progress": report_data["overall_progress"],
            "components": {
                "total": metrics["components_total"],
                "completed": metrics["components_completed"],
                "in_progress": metrics["components_in_progress"],
                "not_started": metrics["components_not_started"]
            },
            "code_migration": {
                "total_lines": metrics["lines_of_code_total"],
                "migrated_lines": metrics["lines_of_code_migrated"],
                "percentage": (metrics["lines_of_code_migrated"] / metrics["lines_of_code_total"] * 100) if metrics["lines_of_code_total"] > 0 else 0
            },
            "testing": {
                "total_tests": metrics["tests_total"],
                "passing_tests": metrics["tests_passing"],
                "pass_rate": (metrics["tests_passing"] / metrics["tests_total"] * 100) if metrics["tests_total"] > 0 else 0
            },
            "milestones": {
                "total": len(milestones),
                "completed": len([m for m in milestones if m["status"] == "completed"]),
                "delayed": len([m for m in milestones if m["status"] == "delayed"])
            },
            "recent_activities": report_data["recent_activities"],
            "current_issues": report_data["issues"],
            "identified_risks": report_data["risks"],
            "next_steps": report_data["next_steps"]
        }
        
        return dashboard_data
    
    def display_text_dashboard(self, project_name: str):
        """Display text-based dashboard"""
        data = self.generate_dashboard_data(project_name)
        
        if "error" in data:
            print(f"❌ {data['error']}")
            return
        
        print(f"📊 Migration Dashboard: {data['project_name']}")
        print("=" * 50)
        print(f"Last Updated: {data['last_updated']}")
        print(f"Overall Progress: {data['overall_progress']:.1f}%")
        
        print(f"\n📦 Components:")
        comp = data['components']
        print(f"  ✅ Completed: {comp['completed']}/{comp['total']}")
        print(f"  🔄 In Progress: {comp['in_progress']}")
        print(f"  ⏳ Not Started: {comp['not_started']}")
        
        print(f"\n💻 Code Migration:")
        code = data['code_migration']
        print(f"  Lines Migrated: {code['migrated_lines']:,}/{code['total_lines']:,} ({code['percentage']:.1f}%)")
        
        if data['testing']['total_tests'] > 0:
            print(f"\n🧪 Testing:")
            test = data['testing']
            print(f"  Tests Passing: {test['passing_tests']}/{test['total_tests']} ({test['pass_rate']:.1f}%)")
        
        print(f"\n🎯 Milestones:")
        milestone = data['milestones']
        print(f"  ✅ Completed: {milestone['completed']}/{milestone['total']}")
        if milestone['delayed'] > 0:
            print(f"  ⚠️  Delayed: {milestone['delayed']}")
        
        if data['current_issues']:
            print(f"\n⚠️  Current Issues:")
            for issue in data['current_issues'][:3]:
                print(f"  • {issue}")
        
        if data['identified_risks']:
            print(f"\n🚨 Identified Risks:")
            for risk in data['identified_risks'][:3]:
                print(f"  • {risk}")
        
        print(f"\n🎯 Next Steps:")
        for step in data['next_steps'][:3]:
            print(f"  • {step}")

async def generate_progress_report(project_name: str, metrics: MigrationMetrics,
                                 milestones: List[Milestone]) -> ProgressReport:
    """Convenience function to generate progress report"""
    tracker = ProgressTracker()
    return tracker.update_progress(project_name, metrics, milestones)