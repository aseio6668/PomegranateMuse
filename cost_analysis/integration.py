"""
Cost Analysis Integration Module for MyndraComposer
Integrates cost analysis with project management and deployment tracking
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .cost_analyzer import (
    CostAnalyzer, CostAnalysisResult, OptimizationRecommendation,
    ResourceUsage, ResourceType, CloudProvider
)


@dataclass
class CostBudget:
    """Cost budget configuration"""
    monthly_limit: float
    alert_thresholds: List[float]  # Percentages (e.g., [50, 80, 95])
    category_limits: Dict[str, float]
    auto_actions: Dict[str, str]  # Actions to take when limits exceeded
    currency: str = "USD"


@dataclass
class CostAlert:
    """Cost alert notification"""
    alert_id: str
    timestamp: str
    alert_type: str  # budget_exceeded, trend_warning, optimization_available
    severity: str  # low, medium, high, critical
    message: str
    current_cost: float
    threshold: float
    recommended_actions: List[str]


class CostIntegration:
    """Main cost integration class"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = CostAnalyzer(project_root)
        self.config_file = self.project_root / ".pomuse" / "cost_config.json"
        self.budget = self._load_budget_config()
        self.alerts_file = self.project_root / ".pomuse" / "cost_alerts.json"
        self.tracking_enabled = True
    
    def _load_budget_config(self) -> Optional[CostBudget]:
        """Load cost budget configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    budget_data = config_data.get("budget", {})
                    if budget_data:
                        return CostBudget(**budget_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load cost budget: {e}")
        return None
    
    def save_budget_config(self):
        """Save cost budget configuration"""
        config_data = {}
        if self.budget:
            config_data["budget"] = asdict(self.budget)
        
        config_data["tracking_enabled"] = self.tracking_enabled
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    async def analyze_project_costs(
        self, 
        include_predictions: bool = True,
        optimization_focus: str = "all"  # all, compute, ml, storage
    ) -> Dict[str, Any]:
        """Perform comprehensive project cost analysis"""
        
        print("💰 Analyzing project costs...")
        
        # Detect infrastructure configuration
        infrastructure_config = await self._detect_project_infrastructure()
        
        # Estimate ML usage
        ml_usage = await self._estimate_project_ml_usage()
        
        # Get historical data if available
        historical_data = await self._load_historical_usage()
        
        # Run cost analysis
        analysis_result = await self.analyzer.analyze_project_costs(
            infrastructure_config=infrastructure_config,
            ml_usage_data=ml_usage,
            historical_data=historical_data
        )
        
        if not analysis_result.success:
            return {
                "success": False,
                "error": analysis_result.error_details,
                "analysis_result": analysis_result
            }
        
        # Generate predictions if requested
        predictions = None
        if include_predictions:
            predictions = await self._generate_cost_predictions(analysis_result)
        
        # Filter recommendations by focus area
        filtered_recommendations = self._filter_recommendations(
            analysis_result.recommendations, optimization_focus
        )
        
        # Check budget alerts
        budget_alerts = []
        if self.budget:
            budget_alerts = self._check_budget_alerts(analysis_result)
        
        # Calculate ROI for optimizations
        optimization_roi = self._calculate_optimization_roi(filtered_recommendations)
        
        return {
            "success": True,
            "analysis_result": analysis_result,
            "predictions": predictions,
            "filtered_recommendations": filtered_recommendations,
            "budget_alerts": budget_alerts,
            "optimization_roi": optimization_roi,
            "summary": {
                "current_monthly_cost": analysis_result.total_monthly_cost,
                "optimization_potential": analysis_result.optimization_potential,
                "budget_utilization": self._calculate_budget_utilization(analysis_result),
                "cost_efficiency_score": self._get_efficiency_score(),
                "recommendations_count": len(filtered_recommendations)
            }
        }
    
    async def _detect_project_infrastructure(self) -> Dict[str, Any]:
        """Detect project infrastructure configuration"""
        config = {}
        
        # Check for Docker configuration
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            config["containers"] = {
                "provider": "local",
                "vcpu_hours_per_month": 100,
                "memory_gb_hours_per_month": 200
            }
            
            # Check for multi-stage builds (more expensive)
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    if "FROM" in content and content.count("FROM") > 1:
                        config["containers"]["complexity_multiplier"] = 1.2
            except Exception:
                pass
        
        # Check for Kubernetes configuration
        k8s_dir = self.project_root / "k8s"
        if k8s_dir.exists() or (self.project_root / "kubernetes").exists():
            config["orchestration"] = {
                "provider": "local",
                "monthly_cost": 30.0  # Estimated K8s overhead
            }
        
        # Check for cloud infrastructure as code
        terraform_dir = self.project_root / "terraform"
        if terraform_dir.exists():
            config.update(self._analyze_terraform_costs(terraform_dir))
        
        # Check for serverless configuration
        if (self.project_root / "serverless.yml").exists():
            config["serverless"] = {
                "provider": "aws",
                "estimated_monthly_invocations": 10000,
                "avg_duration_ms": 500
            }
        
        # Check for CI/CD configuration (has costs)
        cicd_files = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "azure-pipelines.yml",
            "Jenkinsfile"
        ]
        
        for cicd_file in cicd_files:
            if (self.project_root / cicd_file).exists():
                config["cicd"] = {
                    "provider": "github" if "github" in cicd_file else "other",
                    "estimated_monthly_cost": 10.0
                }
                break
        
        # Default minimal configuration if nothing detected
        if not config:
            config = {
                "compute": {
                    "provider": "local",
                    "instance_type": "development",
                    "hours_per_month": 160  # 8 hours/day * 20 days
                }
            }
        
        return config
    
    def _analyze_terraform_costs(self, terraform_dir: Path) -> Dict[str, Any]:
        """Analyze Terraform configuration for cost estimation"""
        config = {}
        
        try:
            # Look for terraform files
            tf_files = list(terraform_dir.glob("*.tf"))
            
            has_compute = False
            has_storage = False
            has_database = False
            
            for tf_file in tf_files:
                try:
                    with open(tf_file, 'r') as f:
                        content = f.read().lower()
                        
                        # Check for compute resources
                        if any(resource in content for resource in ["aws_instance", "azurerm_virtual_machine", "google_compute_instance"]):
                            has_compute = True
                        
                        # Check for storage resources
                        if any(resource in content for resource in ["aws_s3_bucket", "azurerm_storage_account", "google_storage_bucket"]):
                            has_storage = True
                        
                        # Check for database resources
                        if any(resource in content for resource in ["aws_rds", "azurerm_sql", "google_sql"]):
                            has_database = True
                
                except Exception:
                    continue
            
            # Estimate costs based on detected resources
            if has_compute:
                config["compute"] = {
                    "provider": "aws",  # Default assumption
                    "instance_type": "t3.medium",
                    "hours_per_month": 730
                }
            
            if has_storage:
                config["storage"] = {
                    "provider": "aws",
                    "size_gb": 100,
                    "class": "standard"
                }
            
            if has_database:
                config["database"] = {
                    "provider": "aws",
                    "instance_type": "db.t3.micro",
                    "monthly_cost": 15.0
                }
        
        except Exception as e:
            print(f"Warning: Could not analyze Terraform configuration: {e}")
        
        return config
    
    async def _estimate_project_ml_usage(self) -> Dict[str, Any]:
        """Estimate ML usage based on project files"""
        usage = {"inference": {"daily_requests": 0}}
        
        # Check for ML-related files
        ml_indicators = [
            "ollama_client.py",
            "ml_providers.py",
            "**/openai*",
            "**/anthropic*",
            "**/ollama*"
        ]
        
        ml_files_found = 0
        for pattern in ml_indicators:
            ml_files_found += len(list(self.project_root.glob(pattern)))
        
        if ml_files_found > 0:
            # Estimate usage based on project complexity
            source_files = (
                len(list(self.project_root.glob("**/*.py"))) +
                len(list(self.project_root.glob("**/*.js"))) +
                len(list(self.project_root.glob("**/*.ts")))
            )
            
            # More source files = potentially more ML usage
            estimated_daily_requests = min(200, max(10, source_files * 2))
            
            usage = {
                "inference": {
                    "provider": "ollama_local",
                    "model": "llama2:7b",
                    "daily_requests": estimated_daily_requests,
                    "avg_input_tokens": 400,
                    "avg_output_tokens": 150,
                    "cache_hit_rate": 20  # Assume low cache hit rate initially
                }
            }
        
        return usage
    
    async def _load_historical_usage(self) -> List[ResourceUsage]:
        """Load historical resource usage data"""
        # In a real implementation, this would load from monitoring systems
        # For now, return empty list or sample data
        return []
    
    async def _generate_cost_predictions(self, analysis_result: CostAnalysisResult) -> Dict[str, Any]:
        """Generate cost predictions based on trends"""
        current_cost = analysis_result.total_monthly_cost
        trends = analysis_result.trends
        
        predictions = {
            "next_month": current_cost,
            "next_quarter": current_cost * 3,
            "next_year": current_cost * 12,
            "confidence_level": "medium"
        }
        
        # Apply trend analysis
        if trends.get("trend") == "increasing":
            growth_rate = trends.get("growth_rate", 0) / 100
            monthly_growth = 1 + (growth_rate / 12)  # Convert annual to monthly
            
            predictions["next_month"] = current_cost * monthly_growth
            predictions["next_quarter"] = current_cost * (monthly_growth ** 3)
            predictions["next_year"] = current_cost * (monthly_growth ** 12)
            predictions["confidence_level"] = "high" if abs(growth_rate) > 0.1 else "medium"
        
        elif trends.get("trend") == "decreasing":
            growth_rate = abs(trends.get("growth_rate", 0)) / 100
            monthly_decline = 1 - (growth_rate / 12)
            
            predictions["next_month"] = current_cost * monthly_decline
            predictions["next_quarter"] = current_cost * (monthly_decline ** 3)
            predictions["next_year"] = current_cost * (monthly_decline ** 12)
        
        # Add scenario predictions
        predictions["scenarios"] = {
            "optimistic": {
                "description": "With all optimizations applied",
                "next_month": max(0, current_cost - analysis_result.optimization_potential),
                "savings": analysis_result.optimization_potential
            },
            "pessimistic": {
                "description": "With 20% cost inflation",
                "next_month": current_cost * 1.2,
                "increase": current_cost * 0.2
            }
        }
        
        return predictions
    
    def _filter_recommendations(
        self, 
        recommendations: List[OptimizationRecommendation], 
        focus: str
    ) -> List[OptimizationRecommendation]:
        """Filter recommendations by focus area"""
        if focus == "all":
            return recommendations
        
        focus_mapping = {
            "compute": ["rightsizing", "reserved_instances", "spot_instances"],
            "ml": ["ml_model_migration", "ml_caching", "ml_optimization"],
            "storage": ["storage_lifecycle", "storage_optimization"],
            "container": ["container_optimization"]
        }
        
        focus_types = focus_mapping.get(focus, [])
        if not focus_types:
            return recommendations
        
        return [
            rec for rec in recommendations 
            if rec.recommendation_type in focus_types
        ]
    
    def _check_budget_alerts(self, analysis_result: CostAnalysisResult) -> List[CostAlert]:
        """Check for budget-related alerts"""
        alerts = []
        
        if not self.budget:
            return alerts
        
        current_cost = analysis_result.total_monthly_cost
        budget_limit = self.budget.monthly_limit
        
        # Check overall budget
        utilization = (current_cost / budget_limit) * 100
        
        for threshold in self.budget.alert_thresholds:
            if utilization >= threshold:
                severity = "critical" if threshold >= 95 else "high" if threshold >= 80 else "medium"
                
                alert = CostAlert(
                    alert_id=f"budget_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    alert_type="budget_exceeded",
                    severity=severity,
                    message=f"Budget utilization at {utilization:.1f}% (${current_cost:.2f} of ${budget_limit:.2f})",
                    current_cost=current_cost,
                    threshold=threshold,
                    recommended_actions=[
                        "Review and implement cost optimization recommendations",
                        "Consider increasing budget or reducing scope",
                        "Enable auto-scaling and rightsizing"
                    ]
                )
                alerts.append(alert)
                break  # Only create one budget alert (highest threshold exceeded)
        
        # Check category budgets
        for category, category_limit in self.budget.category_limits.items():
            category_cost = analysis_result.cost_breakdown.get(category, 0)
            if category_cost > category_limit:
                alert = CostAlert(
                    alert_id=f"category_alert_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    alert_type="category_budget_exceeded",
                    severity="medium",
                    message=f"{category.title()} costs (${category_cost:.2f}) exceed limit (${category_limit:.2f})",
                    current_cost=category_cost,
                    threshold=category_limit,
                    recommended_actions=[f"Optimize {category} usage", "Review resource allocation"]
                )
                alerts.append(alert)
        
        return alerts
    
    def _calculate_optimization_roi(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Calculate ROI for optimization recommendations"""
        if not recommendations:
            return {"total_savings": 0, "implementation_cost": 0, "roi": 0, "payback_months": 0}
        
        total_savings = sum(rec.savings_amount for rec in recommendations)
        
        # Estimate implementation costs based on effort levels
        effort_costs = {"low": 500, "medium": 2000, "high": 5000}
        total_implementation_cost = sum(
            effort_costs.get(rec.implementation_effort, 1000) 
            for rec in recommendations
        )
        
        # Calculate ROI (annual savings vs implementation cost)
        annual_savings = total_savings * 12
        roi = ((annual_savings - total_implementation_cost) / total_implementation_cost * 100) if total_implementation_cost > 0 else 0
        
        # Calculate payback period
        payback_months = (total_implementation_cost / total_savings) if total_savings > 0 else float('inf')
        
        return {
            "total_monthly_savings": total_savings,
            "annual_savings": annual_savings,
            "implementation_cost": total_implementation_cost,
            "roi_percentage": roi,
            "payback_months": payback_months,
            "recommendations_by_effort": {
                "low": [r for r in recommendations if r.implementation_effort == "low"],
                "medium": [r for r in recommendations if r.implementation_effort == "medium"],
                "high": [r for r in recommendations if r.implementation_effort == "high"]
            }
        }
    
    def _calculate_budget_utilization(self, analysis_result: CostAnalysisResult) -> Optional[float]:
        """Calculate budget utilization percentage"""
        if not self.budget:
            return None
        
        return (analysis_result.total_monthly_cost / self.budget.monthly_limit) * 100
    
    def _get_efficiency_score(self) -> int:
        """Get cost efficiency score from latest analysis"""
        recent_results = self.analyzer.load_analysis_results()
        if recent_results:
            report = self.analyzer.generate_cost_report(30)
            return report.get("cost_efficiency_score", 50)
        return 50
    
    def create_budget(
        self, 
        monthly_limit: float, 
        alert_thresholds: List[float] = None,
        category_limits: Dict[str, float] = None
    ) -> bool:
        """Create or update cost budget"""
        try:
            if alert_thresholds is None:
                alert_thresholds = [50, 80, 95]
            
            if category_limits is None:
                category_limits = {}
            
            self.budget = CostBudget(
                monthly_limit=monthly_limit,
                alert_thresholds=alert_thresholds,
                category_limits=category_limits,
                auto_actions={}
            )
            
            self.save_budget_config()
            print(f"✅ Created budget with ${monthly_limit:.2f} monthly limit")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create budget: {e}")
            return False
    
    def get_cost_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive cost dashboard"""
        # Get cost report from analyzer
        cost_report = self.analyzer.generate_cost_report(days)
        
        # Add budget information
        budget_info = None
        if self.budget:
            budget_info = {
                "monthly_limit": self.budget.monthly_limit,
                "alert_thresholds": self.budget.alert_thresholds,
                "category_limits": self.budget.category_limits
            }
        
        # Load recent alerts
        recent_alerts = self._load_recent_alerts(days)
        
        # Calculate key metrics
        latest_analysis = cost_report.get("latest_analysis")
        current_cost = latest_analysis.total_monthly_cost if latest_analysis else 0
        
        dashboard = {
            "period": cost_report.get("period", f"Last {days} days"),
            "current_monthly_cost": current_cost,
            "budget_info": budget_info,
            "budget_utilization": self._calculate_budget_utilization(latest_analysis) if latest_analysis else None,
            "cost_efficiency_score": cost_report.get("cost_efficiency_score", 50),
            "optimization_potential": cost_report.get("total_optimization_potential", 0),
            "category_breakdown": cost_report.get("category_breakdown", {}),
            "optimization_opportunities": cost_report.get("optimization_opportunities", {}),
            "recent_alerts": recent_alerts,
            "trend_analysis": self._get_trend_analysis(cost_report),
            "quick_wins": self._identify_quick_wins(latest_analysis) if latest_analysis else []
        }
        
        return dashboard
    
    def _load_recent_alerts(self, days: int) -> List[CostAlert]:
        """Load recent cost alerts"""
        if not self.alerts_file.exists():
            return []
        
        try:
            with open(self.alerts_file, 'r') as f:
                alerts_data = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_alerts = []
            
            for alert_data in alerts_data:
                alert_date = datetime.fromisoformat(alert_data["timestamp"])
                if alert_date > cutoff_date:
                    recent_alerts.append(CostAlert(**alert_data))
            
            return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception:
            return []
    
    def _save_alert(self, alert: CostAlert):
        """Save cost alert to file"""
        alerts = []
        
        # Load existing alerts
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []
        
        # Add new alert
        alerts.append(asdict(alert))
        
        # Keep only recent alerts (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        alerts = [
            a for a in alerts 
            if datetime.fromisoformat(a["timestamp"]) > cutoff_date
        ]
        
        # Save alerts
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def _get_trend_analysis(self, cost_report: Dict[str, Any]) -> Dict[str, Any]:
        """Get trend analysis from cost report"""
        latest_analysis = cost_report.get("latest_analysis")
        if not latest_analysis:
            return {"status": "no_data"}
        
        trends = latest_analysis.trends
        return {
            "status": trends.get("trend", "stable"),
            "growth_rate": trends.get("growth_rate", 0),
            "monthly_costs": trends.get("monthly_costs", {}),
            "interpretation": self._interpret_trend(trends)
        }
    
    def _interpret_trend(self, trends: Dict[str, Any]) -> str:
        """Interpret cost trend for user"""
        trend = trends.get("trend", "stable")
        growth_rate = trends.get("growth_rate", 0)
        
        if trend == "increasing":
            if growth_rate > 20:
                return "Costs are growing rapidly. Immediate optimization recommended."
            elif growth_rate > 10:
                return "Costs are growing steadily. Monitor and optimize soon."
            else:
                return "Costs are slowly increasing. Normal growth pattern."
        elif trend == "decreasing":
            return "Costs are decreasing. Good cost management!"
        else:
            return "Costs are stable. Consider optimization opportunities."
    
    def _identify_quick_wins(self, analysis_result: CostAnalysisResult) -> List[Dict[str, Any]]:
        """Identify quick cost optimization wins"""
        quick_wins = []
        
        for rec in analysis_result.recommendations:
            if (rec.implementation_effort == "low" and 
                rec.savings_amount > 10 and 
                rec.risk_level == "low"):
                
                quick_wins.append({
                    "title": rec.description,
                    "savings": rec.savings_amount,
                    "effort": rec.implementation_effort,
                    "type": rec.recommendation_type
                })
        
        return sorted(quick_wins, key=lambda x: x["savings"], reverse=True)[:5]


# CLI interface functions
async def run_cost_analysis_interactive(project_root: str):
    """Interactive cost analysis runner"""
    integration = CostIntegration(project_root)
    
    print("💰 MyndraComposer Cost Analysis")
    print("=" * 50)
    
    # Analysis options
    print("\nCost analysis options:")
    print("1. Full cost analysis")
    print("2. Quick cost estimate")
    print("3. Cost dashboard")
    print("4. Set up budget")
    print("5. Optimization recommendations")
    
    try:
        choice = int(input("\nSelect option (1-5): "))
        
        if choice == 1:
            print("\n💰 Running full cost analysis...")
            result = await integration.analyze_project_costs(
                include_predictions=True,
                optimization_focus="all"
            )
            
            if result["success"]:
                summary = result["summary"]
                print(f"\n✅ Cost analysis completed!")
                print(f"   Current Monthly Cost: ${summary['current_monthly_cost']:.2f}")
                print(f"   Optimization Potential: ${summary['optimization_potential']:.2f}")
                print(f"   Cost Efficiency Score: {summary['cost_efficiency_score']}/100")
                
                if result["budget_alerts"]:
                    print(f"\n⚠️  Budget Alerts: {len(result['budget_alerts'])}")
                    for alert in result["budget_alerts"]:
                        print(f"     - {alert.message}")
                
                print(f"\n💡 Recommendations: {summary['recommendations_count']}")
                for i, rec in enumerate(result["filtered_recommendations"][:3], 1):
                    print(f"   {i}. {rec.description} (${rec.savings_amount:.2f}/month)")
            else:
                print(f"❌ Analysis failed: {result['error']}")
        
        elif choice == 2:
            print("\n⚡ Running quick cost estimate...")
            result = await integration.analyze_project_costs(
                include_predictions=False,
                optimization_focus="all"
            )
            
            if result["success"]:
                summary = result["summary"]
                print(f"\n✅ Quick estimate completed!")
                print(f"   Estimated Monthly Cost: ${summary['current_monthly_cost']:.2f}")
                print(f"   Potential Savings: ${summary['optimization_potential']:.2f}")
            else:
                print(f"❌ Estimate failed: {result['error']}")
        
        elif choice == 3:
            days = int(input("Dashboard period in days [30]: ") or "30")
            dashboard = integration.get_cost_dashboard(days)
            
            print(f"\n📊 Cost Dashboard ({dashboard['period']})")
            print(f"   Current Monthly Cost: ${dashboard['current_monthly_cost']:.2f}")
            print(f"   Cost Efficiency Score: {dashboard['cost_efficiency_score']}/100")
            print(f"   Optimization Potential: ${dashboard['optimization_potential']:.2f}")
            
            if dashboard["budget_info"]:
                budget = dashboard["budget_info"]
                utilization = dashboard.get("budget_utilization", 0)
                print(f"   Budget: ${dashboard['current_monthly_cost']:.2f} / ${budget['monthly_limit']:.2f} ({utilization:.1f}%)")
            
            # Show category breakdown
            print("\n💸 Cost Breakdown:")
            for category, cost in dashboard["category_breakdown"].items():
                print(f"   {category.title()}: ${cost:.2f}")
            
            # Show quick wins
            if dashboard["quick_wins"]:
                print("\n🎯 Quick Wins:")
                for win in dashboard["quick_wins"]:
                    print(f"   • {win['title']} (${win['savings']:.2f}/month)")
        
        elif choice == 4:
            print("\n💰 Budget Setup")
            current_budget = integration.budget
            
            if current_budget:
                print(f"Current budget: ${current_budget.monthly_limit:.2f}/month")
                update = input("Update budget? [y/N]: ").strip().lower()
                if update != 'y':
                    return
            
            monthly_limit = float(input("Monthly budget limit ($): "))
            
            # Set up alert thresholds
            thresholds_input = input("Alert thresholds (%) [50,80,95]: ").strip()
            if thresholds_input:
                thresholds = [float(x.strip()) for x in thresholds_input.split(",")]
            else:
                thresholds = [50, 80, 95]
            
            success = integration.create_budget(monthly_limit, thresholds)
            if success:
                print("✅ Budget created successfully!")
            else:
                print("❌ Failed to create budget")
        
        elif choice == 5:
            focus = input("Optimization focus [all/compute/ml/storage]: ").strip() or "all"
            
            print(f"\n🔍 Finding {focus} optimization opportunities...")
            result = await integration.analyze_project_costs(
                include_predictions=False,
                optimization_focus=focus
            )
            
            if result["success"]:
                recommendations = result["filtered_recommendations"]
                roi_info = result["optimization_roi"]
                
                print(f"\n💡 Found {len(recommendations)} optimization opportunities")
                print(f"   Total Monthly Savings: ${roi_info['total_monthly_savings']:.2f}")
                print(f"   Implementation Cost: ${roi_info['implementation_cost']:.2f}")
                print(f"   ROI: {roi_info['roi_percentage']:.1f}%")
                print(f"   Payback Period: {roi_info['payback_months']:.1f} months")
                
                print("\n🎯 Top Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {rec.description}")
                    print(f"      Savings: ${rec.savings_amount:.2f}/month")
                    print(f"      Effort: {rec.implementation_effort}, Risk: {rec.risk_level}")
            else:
                print(f"❌ Analysis failed: {result['error']}")
        
    except (ValueError, KeyboardInterrupt):
        print("\nCost analysis cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_cost_analysis_interactive(project_path))