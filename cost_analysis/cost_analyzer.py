"""
Cost Analysis and Optimization System for MyndraComposer
Provides comprehensive cost analysis for cloud infrastructure, ML usage, and development resources
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import math


class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    ML_INFERENCE = "ml_inference"
    ML_TRAINING = "ml_training"
    DATABASE = "database"
    CONTAINER = "container"
    SERVERLESS = "serverless"


class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    resource_type: ResourceType
    provider: CloudProvider
    region: str
    usage_amount: float
    usage_unit: str
    duration_hours: float
    timestamp: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class CostEstimate:
    """Cost estimate for a resource"""
    resource_usage: ResourceUsage
    unit_cost: float
    total_cost: float
    currency: str = "USD"
    billing_period: str = "monthly"
    discount_applied: float = 0.0
    tax_rate: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    resource_id: str
    current_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percentage: float
    recommendation_type: str
    description: str
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    estimated_impact: str
    action_items: List[str]


@dataclass
class CostAnalysisResult:
    """Results from cost analysis"""
    analysis_id: str
    timestamp: str
    total_monthly_cost: float
    cost_breakdown: Dict[str, float]
    optimization_potential: float
    recommendations: List[OptimizationRecommendation]
    trends: Dict[str, Any]
    success: bool
    error_details: Optional[str] = None


class CloudCostCalculator:
    """Calculate costs for different cloud providers"""
    
    def __init__(self):
        self.pricing_data = self._load_pricing_data()
        self.exchange_rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.75}  # Simplified
    
    def _load_pricing_data(self) -> Dict[str, Dict[str, Any]]:
        """Load pricing data for different cloud providers"""
        return {
            "aws": {
                "ec2": {
                    "t3.micro": {"price_per_hour": 0.0104, "vcpu": 2, "memory_gb": 1},
                    "t3.small": {"price_per_hour": 0.0208, "vcpu": 2, "memory_gb": 2},
                    "t3.medium": {"price_per_hour": 0.0416, "vcpu": 2, "memory_gb": 4},
                    "t3.large": {"price_per_hour": 0.0832, "vcpu": 2, "memory_gb": 8},
                    "m5.large": {"price_per_hour": 0.096, "vcpu": 2, "memory_gb": 8},
                    "m5.xlarge": {"price_per_hour": 0.192, "vcpu": 4, "memory_gb": 16},
                    "c5.large": {"price_per_hour": 0.085, "vcpu": 2, "memory_gb": 4},
                    "r5.large": {"price_per_hour": 0.126, "vcpu": 2, "memory_gb": 16}
                },
                "s3": {
                    "standard": {"price_per_gb_month": 0.023},
                    "ia": {"price_per_gb_month": 0.0125},
                    "glacier": {"price_per_gb_month": 0.004}
                },
                "rds": {
                    "db.t3.micro": {"price_per_hour": 0.017},
                    "db.t3.small": {"price_per_hour": 0.034},
                    "db.r5.large": {"price_per_hour": 0.24}
                },
                "lambda": {
                    "requests": {"price_per_million": 0.20},
                    "duration": {"price_per_gb_second": 0.0000166667}
                },
                "ecs": {
                    "fargate_vcpu": {"price_per_hour": 0.04048},
                    "fargate_memory": {"price_per_gb_hour": 0.004445}
                }
            },
            "azure": {
                "vm": {
                    "B1s": {"price_per_hour": 0.0104, "vcpu": 1, "memory_gb": 1},
                    "B2s": {"price_per_hour": 0.0416, "vcpu": 2, "memory_gb": 4},
                    "D2s_v3": {"price_per_hour": 0.096, "vcpu": 2, "memory_gb": 8}
                },
                "storage": {
                    "standard": {"price_per_gb_month": 0.0208},
                    "premium": {"price_per_gb_month": 0.15}
                },
                "functions": {
                    "consumption": {"price_per_million_executions": 0.20}
                }
            },
            "gcp": {
                "compute": {
                    "e2-micro": {"price_per_hour": 0.008467, "vcpu": 0.25, "memory_gb": 1},
                    "e2-small": {"price_per_hour": 0.016934, "vcpu": 0.5, "memory_gb": 2},
                    "n1-standard-1": {"price_per_hour": 0.0475, "vcpu": 1, "memory_gb": 3.75}
                },
                "storage": {
                    "standard": {"price_per_gb_month": 0.020},
                    "nearline": {"price_per_gb_month": 0.010}
                }
            }
        }
    
    def calculate_compute_cost(
        self, 
        provider: CloudProvider, 
        instance_type: str, 
        hours: float,
        region: str = "us-east-1"
    ) -> float:
        """Calculate compute cost for given usage"""
        try:
            if provider == CloudProvider.AWS:
                pricing = self.pricing_data["aws"]["ec2"]
                if instance_type in pricing:
                    return pricing[instance_type]["price_per_hour"] * hours
            elif provider == CloudProvider.AZURE:
                pricing = self.pricing_data["azure"]["vm"]
                if instance_type in pricing:
                    return pricing[instance_type]["price_per_hour"] * hours
            elif provider == CloudProvider.GCP:
                pricing = self.pricing_data["gcp"]["compute"]
                if instance_type in pricing:
                    return pricing[instance_type]["price_per_hour"] * hours
            
            # Fallback estimate
            return self._estimate_compute_cost(provider, hours)
            
        except Exception:
            return self._estimate_compute_cost(provider, hours)
    
    def calculate_storage_cost(
        self, 
        provider: CloudProvider, 
        storage_gb: float, 
        storage_class: str = "standard"
    ) -> float:
        """Calculate storage cost per month"""
        try:
            if provider == CloudProvider.AWS:
                pricing = self.pricing_data["aws"]["s3"]
                return pricing.get(storage_class, pricing["standard"])["price_per_gb_month"] * storage_gb
            elif provider == CloudProvider.AZURE:
                pricing = self.pricing_data["azure"]["storage"]
                return pricing.get(storage_class, pricing["standard"])["price_per_gb_month"] * storage_gb
            elif provider == CloudProvider.GCP:
                pricing = self.pricing_data["gcp"]["storage"]
                return pricing.get(storage_class, pricing["standard"])["price_per_gb_month"] * storage_gb
            
            # Fallback estimate
            return storage_gb * 0.025  # $0.025 per GB/month average
            
        except Exception:
            return storage_gb * 0.025
    
    def calculate_ml_inference_cost(
        self, 
        provider: CloudProvider, 
        requests_per_month: int, 
        avg_processing_time_ms: float
    ) -> float:
        """Calculate ML inference cost"""
        # Base costs per provider (simplified)
        base_costs = {
            CloudProvider.AWS: 0.0004,  # per 1000 requests
            CloudProvider.AZURE: 0.0005,
            CloudProvider.GCP: 0.0003,
            CloudProvider.LOCAL: 0.0001  # Estimated local hosting cost
        }
        
        base_cost = base_costs.get(provider, 0.0004)
        
        # Factor in processing time (longer processing = higher cost)
        time_factor = max(1.0, avg_processing_time_ms / 1000.0)
        
        return (requests_per_month / 1000.0) * base_cost * time_factor
    
    def calculate_container_cost(
        self, 
        provider: CloudProvider, 
        vcpu_hours: float, 
        memory_gb_hours: float
    ) -> float:
        """Calculate container orchestration cost"""
        if provider == CloudProvider.AWS:
            # ECS Fargate pricing
            vcpu_cost = vcpu_hours * 0.04048
            memory_cost = memory_gb_hours * 0.004445
            return vcpu_cost + memory_cost
        elif provider == CloudProvider.AZURE:
            # Azure Container Instances
            return vcpu_hours * 0.0012 + memory_gb_hours * 0.000115
        elif provider == CloudProvider.GCP:
            # Google Cloud Run
            return vcpu_hours * 0.000024 + memory_gb_hours * 0.0000025
        else:
            # Local Docker/Kubernetes estimate
            return (vcpu_hours + memory_gb_hours) * 0.001
    
    def _estimate_compute_cost(self, provider: CloudProvider, hours: float) -> float:
        """Fallback compute cost estimation"""
        hourly_rates = {
            CloudProvider.AWS: 0.05,
            CloudProvider.AZURE: 0.045,
            CloudProvider.GCP: 0.04,
            CloudProvider.LOCAL: 0.01
        }
        return hours * hourly_rates.get(provider, 0.05)


class MLCostAnalyzer:
    """Analyze ML model usage costs"""
    
    def __init__(self):
        self.model_costs = self._load_model_costs()
    
    def _load_model_costs(self) -> Dict[str, Dict[str, Any]]:
        """Load ML model cost data"""
        return {
            "ollama_local": {
                "base_cost_per_hour": 0.02,  # Local GPU/CPU cost
                "models": {
                    "llama2:7b": {"cost_multiplier": 1.0, "tokens_per_dollar": 50000},
                    "llama2:13b": {"cost_multiplier": 1.5, "tokens_per_dollar": 33000},
                    "codellama:7b": {"cost_multiplier": 1.2, "tokens_per_dollar": 42000},
                    "codellama:13b": {"cost_multiplier": 1.8, "tokens_per_dollar": 28000}
                }
            },
            "openai": {
                "gpt-3.5-turbo": {"input_cost_per_1k": 0.0015, "output_cost_per_1k": 0.002},
                "gpt-4": {"input_cost_per_1k": 0.03, "output_cost_per_1k": 0.06},
                "gpt-4-turbo": {"input_cost_per_1k": 0.01, "output_cost_per_1k": 0.03}
            },
            "anthropic": {
                "claude-3-haiku": {"input_cost_per_1k": 0.00025, "output_cost_per_1k": 0.00125},
                "claude-3-sonnet": {"input_cost_per_1k": 0.003, "output_cost_per_1k": 0.015},
                "claude-3-opus": {"input_cost_per_1k": 0.015, "output_cost_per_1k": 0.075}
            },
            "google": {
                "gemini-pro": {"input_cost_per_1k": 0.0005, "output_cost_per_1k": 0.0015},
                "palm-2": {"input_cost_per_1k": 0.001, "output_cost_per_1k": 0.001}
            }
        }
    
    def calculate_model_usage_cost(
        self, 
        provider: str, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """Calculate cost for ML model usage"""
        try:
            if provider == "ollama_local":
                model_info = self.model_costs[provider]["models"].get(model, {})
                total_tokens = input_tokens + output_tokens
                cost_multiplier = model_info.get("cost_multiplier", 1.0)
                base_cost = self.model_costs[provider]["base_cost_per_hour"]
                
                # Estimate based on tokens and processing time
                estimated_hours = total_tokens / 10000.0  # Rough estimate
                return estimated_hours * base_cost * cost_multiplier
            
            elif provider in ["openai", "anthropic", "google"]:
                model_info = self.model_costs[provider].get(model, {})
                if not model_info:
                    return 0.0
                
                input_cost = (input_tokens / 1000.0) * model_info.get("input_cost_per_1k", 0)
                output_cost = (output_tokens / 1000.0) * model_info.get("output_cost_per_1k", 0)
                return input_cost + output_cost
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def estimate_monthly_ml_cost(
        self, 
        daily_requests: int, 
        avg_input_tokens: int, 
        avg_output_tokens: int,
        provider: str = "ollama_local",
        model: str = "llama2:7b"
    ) -> float:
        """Estimate monthly ML cost based on usage patterns"""
        monthly_requests = daily_requests * 30
        cost_per_request = self.calculate_model_usage_cost(
            provider, model, avg_input_tokens, avg_output_tokens
        )
        return monthly_requests * cost_per_request


class CostOptimizer:
    """Analyze costs and generate optimization recommendations"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load cost optimization rules"""
        return [
            {
                "name": "rightsizing_compute",
                "description": "Right-size compute instances based on actual usage",
                "threshold": {"cpu_utilization": 20, "memory_utilization": 30},
                "potential_savings": 0.3
            },
            {
                "name": "reserved_instances",
                "description": "Use reserved instances for predictable workloads",
                "threshold": {"uptime_percentage": 70},
                "potential_savings": 0.4
            },
            {
                "name": "spot_instances",
                "description": "Use spot instances for fault-tolerant workloads",
                "threshold": {"fault_tolerance": True},
                "potential_savings": 0.7
            },
            {
                "name": "storage_optimization",
                "description": "Optimize storage classes and lifecycle policies",
                "threshold": {"data_access_frequency": "low"},
                "potential_savings": 0.5
            },
            {
                "name": "ml_model_optimization",
                "description": "Optimize ML model selection and caching",
                "threshold": {"model_accuracy_drop": 5},
                "potential_savings": 0.25
            },
            {
                "name": "auto_scaling",
                "description": "Implement auto-scaling for variable workloads",
                "threshold": {"load_variance": 50},
                "potential_savings": 0.35
            }
        ]
    
    def analyze_compute_optimization(
        self, 
        usage_data: List[ResourceUsage]
    ) -> List[OptimizationRecommendation]:
        """Analyze compute cost optimization opportunities"""
        recommendations = []
        
        # Group by resource type and analyze
        compute_resources = [u for u in usage_data if u.resource_type == ResourceType.COMPUTE]
        
        for resource in compute_resources:
            current_cost = resource.usage_amount * 0.05  # Simplified cost calculation
            
            # Right-sizing recommendation
            if resource.usage_amount < 50:  # Low utilization
                optimized_cost = current_cost * 0.7  # 30% savings
                savings = current_cost - optimized_cost
                
                recommendations.append(OptimizationRecommendation(
                    resource_id=f"{resource.provider.value}-{resource.resource_type.value}",
                    current_cost=current_cost,
                    optimized_cost=optimized_cost,
                    savings_amount=savings,
                    savings_percentage=(savings / current_cost) * 100,
                    recommendation_type="rightsizing",
                    description=f"Right-size {resource.resource_type.value} instance due to low utilization",
                    implementation_effort="low",
                    risk_level="low",
                    estimated_impact="Reduce instance size by one tier",
                    action_items=[
                        "Monitor current resource utilization",
                        "Test smaller instance size",
                        "Implement gradual migration"
                    ]
                ))
        
        return recommendations
    
    def analyze_ml_optimization(
        self, 
        ml_usage: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze ML cost optimization opportunities"""
        recommendations = []
        
        current_monthly_cost = ml_usage.get("monthly_cost", 0)
        if current_monthly_cost == 0:
            return recommendations
        
        # Model selection optimization
        if ml_usage.get("provider") == "openai" and current_monthly_cost > 100:
            optimized_cost = current_monthly_cost * 0.6  # Switch to local models
            savings = current_monthly_cost - optimized_cost
            
            recommendations.append(OptimizationRecommendation(
                resource_id="ml-inference",
                current_cost=current_monthly_cost,
                optimized_cost=optimized_cost,
                savings_amount=savings,
                savings_percentage=(savings / current_monthly_cost) * 100,
                recommendation_type="ml_model_migration",
                description="Migrate from cloud ML APIs to local models",
                implementation_effort="medium",
                risk_level="medium",
                estimated_impact="Significant cost reduction with local ML hosting",
                action_items=[
                    "Set up local Ollama deployment",
                    "Test model performance and accuracy",
                    "Implement gradual migration strategy",
                    "Monitor performance metrics"
                ]
            ))
        
        # Caching optimization
        if ml_usage.get("cache_hit_rate", 0) < 30:
            cache_savings = current_monthly_cost * 0.15
            
            recommendations.append(OptimizationRecommendation(
                resource_id="ml-caching",
                current_cost=current_monthly_cost,
                optimized_cost=current_monthly_cost - cache_savings,
                savings_amount=cache_savings,
                savings_percentage=(cache_savings / current_monthly_cost) * 100,
                recommendation_type="ml_caching",
                description="Implement ML response caching",
                implementation_effort="low",
                risk_level="low",
                estimated_impact="Reduce redundant ML API calls",
                action_items=[
                    "Implement Redis-based response caching",
                    "Set appropriate cache TTL values",
                    "Monitor cache hit rates"
                ]
            ))
        
        return recommendations
    
    def analyze_infrastructure_optimization(
        self, 
        infrastructure_costs: Dict[str, float]
    ) -> List[OptimizationRecommendation]:
        """Analyze infrastructure cost optimization"""
        recommendations = []
        
        total_cost = sum(infrastructure_costs.values())
        if total_cost == 0:
            return recommendations
        
        # Container optimization
        container_cost = infrastructure_costs.get("container", 0)
        if container_cost > total_cost * 0.3:  # Containers are >30% of total cost
            optimized_cost = container_cost * 0.75  # 25% savings
            savings = container_cost - optimized_cost
            
            recommendations.append(OptimizationRecommendation(
                resource_id="container-optimization",
                current_cost=container_cost,
                optimized_cost=optimized_cost,
                savings_amount=savings,
                savings_percentage=(savings / container_cost) * 100,
                recommendation_type="container_optimization",
                description="Optimize container resource allocation and scaling",
                implementation_effort="medium",
                risk_level="low",
                estimated_impact="Better resource utilization and auto-scaling",
                action_items=[
                    "Implement horizontal pod autoscaling",
                    "Optimize container resource requests/limits",
                    "Use cluster autoscaling",
                    "Consider serverless containers for variable workloads"
                ]
            ))
        
        # Storage optimization
        storage_cost = infrastructure_costs.get("storage", 0)
        if storage_cost > 50:  # $50+ per month
            storage_savings = storage_cost * 0.4  # 40% potential savings
            
            recommendations.append(OptimizationRecommendation(
                resource_id="storage-optimization",
                current_cost=storage_cost,
                optimized_cost=storage_cost - storage_savings,
                savings_amount=storage_savings,
                savings_percentage=(storage_savings / storage_cost) * 100,
                recommendation_type="storage_lifecycle",
                description="Implement intelligent storage tiering and lifecycle policies",
                implementation_effort="low",
                risk_level="low",
                estimated_impact="Automatic data archiving and cost reduction",
                action_items=[
                    "Analyze data access patterns",
                    "Implement lifecycle policies",
                    "Move cold data to cheaper storage tiers",
                    "Set up automated data archiving"
                ]
            ))
        
        return recommendations


class CostAnalyzer:
    """Main cost analysis engine"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / ".pomuse" / "cost_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.cloud_calculator = CloudCostCalculator()
        self.ml_analyzer = MLCostAnalyzer()
        self.optimizer = CostOptimizer()
    
    async def analyze_project_costs(
        self, 
        infrastructure_config: Dict[str, Any] = None,
        ml_usage_data: Dict[str, Any] = None,
        historical_data: List[ResourceUsage] = None
    ) -> CostAnalysisResult:
        """Perform comprehensive cost analysis"""
        
        analysis_id = f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            print(f"💰 Starting cost analysis: {analysis_id}")
            
            # Analyze infrastructure costs
            infrastructure_costs = await self._analyze_infrastructure_costs(infrastructure_config)
            
            # Analyze ML costs
            ml_costs = await self._analyze_ml_costs(ml_usage_data)
            
            # Analyze historical trends
            trends = await self._analyze_cost_trends(historical_data or [])
            
            # Calculate total costs
            total_monthly_cost = sum(infrastructure_costs.values()) + ml_costs.get("total", 0)
            
            # Generate optimization recommendations
            recommendations = []
            
            # Infrastructure optimization
            infra_recs = self.optimizer.analyze_infrastructure_optimization(infrastructure_costs)
            recommendations.extend(infra_recs)
            
            # ML optimization
            ml_recs = self.optimizer.analyze_ml_optimization(ml_costs)
            recommendations.extend(ml_recs)
            
            # Compute optimization
            if historical_data:
                compute_recs = self.optimizer.analyze_compute_optimization(historical_data)
                recommendations.extend(compute_recs)
            
            # Calculate optimization potential
            optimization_potential = sum(rec.savings_amount for rec in recommendations)
            
            cost_breakdown = {**infrastructure_costs, "ml_inference": ml_costs.get("total", 0)}
            
            result = CostAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                total_monthly_cost=total_monthly_cost,
                cost_breakdown=cost_breakdown,
                optimization_potential=optimization_potential,
                recommendations=recommendations,
                trends=trends,
                success=True
            )
            
            # Save results
            await self._save_analysis_result(result)
            
            print(f"✅ Cost analysis completed!")
            print(f"   Total Monthly Cost: ${total_monthly_cost:.2f}")
            print(f"   Optimization Potential: ${optimization_potential:.2f}")
            print(f"   Recommendations: {len(recommendations)}")
            
            return result
            
        except Exception as e:
            error_result = CostAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                total_monthly_cost=0,
                cost_breakdown={},
                optimization_potential=0,
                recommendations=[],
                trends={},
                success=False,
                error_details=str(e)
            )
            
            print(f"❌ Cost analysis failed: {e}")
            return error_result
    
    async def _analyze_infrastructure_costs(
        self, 
        config: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Analyze infrastructure costs"""
        if not config:
            config = await self._detect_infrastructure_config()
        
        costs = {}
        
        # Compute costs
        compute_config = config.get("compute", {})
        if compute_config:
            provider = CloudProvider(compute_config.get("provider", "local"))
            instance_type = compute_config.get("instance_type", "t3.medium")
            hours_per_month = compute_config.get("hours_per_month", 730)  # 24*30.5
            
            compute_cost = self.cloud_calculator.calculate_compute_cost(
                provider, instance_type, hours_per_month
            )
            costs["compute"] = compute_cost
        
        # Storage costs
        storage_config = config.get("storage", {})
        if storage_config:
            provider = CloudProvider(storage_config.get("provider", "local"))
            storage_gb = storage_config.get("size_gb", 100)
            storage_class = storage_config.get("class", "standard")
            
            storage_cost = self.cloud_calculator.calculate_storage_cost(
                provider, storage_gb, storage_class
            )
            costs["storage"] = storage_cost
        
        # Container costs
        container_config = config.get("containers", {})
        if container_config:
            provider = CloudProvider(container_config.get("provider", "local"))
            vcpu_hours = container_config.get("vcpu_hours_per_month", 100)
            memory_gb_hours = container_config.get("memory_gb_hours_per_month", 200)
            
            container_cost = self.cloud_calculator.calculate_container_cost(
                provider, vcpu_hours, memory_gb_hours
            )
            costs["container"] = container_cost
        
        # Database costs (simplified)
        database_config = config.get("database", {})
        if database_config:
            costs["database"] = database_config.get("monthly_cost", 25.0)
        
        # Network costs (simplified)
        network_config = config.get("network", {})
        if network_config:
            costs["network"] = network_config.get("monthly_cost", 10.0)
        
        return costs
    
    async def _analyze_ml_costs(self, usage_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Analyze ML usage costs"""
        if not usage_data:
            usage_data = await self._estimate_ml_usage()
        
        costs = {}
        
        # Inference costs
        inference_config = usage_data.get("inference", {})
        if inference_config:
            provider = inference_config.get("provider", "ollama_local")
            model = inference_config.get("model", "llama2:7b")
            daily_requests = inference_config.get("daily_requests", 100)
            avg_input_tokens = inference_config.get("avg_input_tokens", 500)
            avg_output_tokens = inference_config.get("avg_output_tokens", 200)
            
            monthly_cost = self.ml_analyzer.estimate_monthly_ml_cost(
                daily_requests, avg_input_tokens, avg_output_tokens, provider, model
            )
            costs["inference"] = monthly_cost
        
        # Training costs (if applicable)
        training_config = usage_data.get("training", {})
        if training_config:
            costs["training"] = training_config.get("monthly_cost", 0)
        
        costs["total"] = sum(costs.values())
        return costs
    
    async def _detect_infrastructure_config(self) -> Dict[str, Any]:
        """Auto-detect infrastructure configuration"""
        config = {}
        
        # Check for Docker/Kubernetes
        if (self.project_root / "Dockerfile").exists():
            config["containers"] = {
                "provider": "local",
                "vcpu_hours_per_month": 50,
                "memory_gb_hours_per_month": 100
            }
        
        # Check for cloud configuration files
        if (self.project_root / "terraform").exists():
            config["compute"] = {
                "provider": "aws",
                "instance_type": "t3.medium",
                "hours_per_month": 730
            }
            config["storage"] = {
                "provider": "aws",
                "size_gb": 100,
                "class": "standard"
            }
        
        # Default minimal configuration
        if not config:
            config = {
                "compute": {
                    "provider": "local",
                    "instance_type": "local",
                    "hours_per_month": 200
                },
                "storage": {
                    "provider": "local",
                    "size_gb": 50,
                    "class": "standard"
                }
            }
        
        return config
    
    async def _estimate_ml_usage(self) -> Dict[str, Any]:
        """Estimate ML usage based on project analysis"""
        # Check project for ML usage indicators
        ml_files = list(self.project_root.rglob("*ollama*")) + list(self.project_root.rglob("*ml*"))
        
        if ml_files or (self.project_root / "ollama_client.py").exists():
            return {
                "inference": {
                    "provider": "ollama_local",
                    "model": "llama2:7b",
                    "daily_requests": 50,
                    "avg_input_tokens": 400,
                    "avg_output_tokens": 150
                }
            }
        else:
            return {"inference": {"daily_requests": 0}}
    
    async def _analyze_cost_trends(self, historical_data: List[ResourceUsage]) -> Dict[str, Any]:
        """Analyze cost trends from historical data"""
        if not historical_data:
            return {"trend": "no_data", "growth_rate": 0}
        
        # Group by month
        monthly_costs = {}
        for usage in historical_data:
            month = usage.timestamp[:7]  # YYYY-MM
            if month not in monthly_costs:
                monthly_costs[month] = 0
            
            # Simplified cost calculation
            estimated_cost = usage.usage_amount * 0.05
            monthly_costs[month] += estimated_cost
        
        # Calculate trend
        months = sorted(monthly_costs.keys())
        if len(months) < 2:
            return {"trend": "insufficient_data", "growth_rate": 0}
        
        first_month_cost = monthly_costs[months[0]]
        last_month_cost = monthly_costs[months[-1]]
        
        if first_month_cost > 0:
            growth_rate = ((last_month_cost - first_month_cost) / first_month_cost) * 100
        else:
            growth_rate = 0
        
        trend = "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
        
        return {
            "trend": trend,
            "growth_rate": growth_rate,
            "monthly_costs": monthly_costs,
            "latest_month_cost": last_month_cost
        }
    
    async def _save_analysis_result(self, result: CostAnalysisResult):
        """Save analysis results to file"""
        filename = f"{result.analysis_id}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        # Convert enums to strings
        for rec in result_dict["recommendations"]:
            # Already primitive types, no conversion needed
            pass
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"📁 Cost analysis results saved to: {filepath}")
    
    def load_analysis_results(self, analysis_id: str = None) -> List[CostAnalysisResult]:
        """Load previous analysis results"""
        results = []
        
        pattern = f"{analysis_id}.json" if analysis_id else "*.json"
        
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to objects
                recommendations = []
                for rec_data in data.get("recommendations", []):
                    recommendations.append(OptimizationRecommendation(**rec_data))
                
                result = CostAnalysisResult(
                    analysis_id=data["analysis_id"],
                    timestamp=data["timestamp"],
                    total_monthly_cost=data["total_monthly_cost"],
                    cost_breakdown=data["cost_breakdown"],
                    optimization_potential=data["optimization_potential"],
                    recommendations=recommendations,
                    trends=data["trends"],
                    success=data["success"],
                    error_details=data.get("error_details")
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def generate_cost_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive cost report"""
        results = self.load_analysis_results()
        
        # Filter recent results
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r for r in results 
            if datetime.fromisoformat(r.timestamp) > cutoff_date
        ]
        
        if not recent_results:
            return {"error": f"No cost analysis results found in the last {days} days"}
        
        # Calculate averages and trends
        avg_monthly_cost = sum(r.total_monthly_cost for r in recent_results) / len(recent_results)
        total_optimization_potential = sum(r.optimization_potential for r in recent_results)
        
        # Category breakdown
        category_costs = {}
        for result in recent_results:
            for category, cost in result.cost_breakdown.items():
                if category not in category_costs:
                    category_costs[category] = []
                category_costs[category].append(cost)
        
        category_averages = {
            category: sum(costs) / len(costs) 
            for category, costs in category_costs.items()
        }
        
        # Top recommendations
        all_recommendations = []
        for result in recent_results:
            all_recommendations.extend(result.recommendations)
        
        # Group by type and calculate total potential savings
        rec_by_type = {}
        for rec in all_recommendations:
            rec_type = rec.recommendation_type
            if rec_type not in rec_by_type:
                rec_by_type[rec_type] = {"count": 0, "total_savings": 0}
            rec_by_type[rec_type]["count"] += 1
            rec_by_type[rec_type]["total_savings"] += rec.savings_amount
        
        return {
            "period": f"Last {days} days",
            "total_analyses": len(recent_results),
            "average_monthly_cost": avg_monthly_cost,
            "total_optimization_potential": total_optimization_potential,
            "category_breakdown": category_averages,
            "optimization_opportunities": rec_by_type,
            "latest_analysis": recent_results[0] if recent_results else None,
            "cost_efficiency_score": self._calculate_efficiency_score(recent_results)
        }
    
    def _calculate_efficiency_score(self, results: List[CostAnalysisResult]) -> int:
        """Calculate cost efficiency score (0-100)"""
        if not results:
            return 50  # Neutral score
        
        latest = results[0]
        
        # Base score
        score = 70
        
        # Adjust based on optimization potential
        if latest.total_monthly_cost > 0:
            optimization_ratio = latest.optimization_potential / latest.total_monthly_cost
            if optimization_ratio < 0.1:  # Less than 10% potential savings
                score += 20
            elif optimization_ratio < 0.2:  # 10-20% potential savings
                score += 10
            elif optimization_ratio > 0.4:  # More than 40% potential savings
                score -= 30
            elif optimization_ratio > 0.3:  # 30-40% potential savings
                score -= 20
        
        # Adjust based on trends
        trend = latest.trends.get("trend", "stable")
        if trend == "decreasing":
            score += 10
        elif trend == "increasing":
            growth_rate = latest.trends.get("growth_rate", 0)
            if growth_rate > 20:  # Rapid cost growth
                score -= 20
            elif growth_rate > 10:
                score -= 10
        
        return max(0, min(100, score))


if __name__ == "__main__":
    async def main():
        analyzer = CostAnalyzer(".")
        
        # Example analysis
        result = await analyzer.analyze_project_costs()
        
        if result.success:
            print(f"\nCost Analysis Summary:")
            print(f"  Total Monthly Cost: ${result.total_monthly_cost:.2f}")
            print(f"  Optimization Potential: ${result.optimization_potential:.2f}")
            print(f"  Recommendations: {len(result.recommendations)}")
        else:
            print(f"Analysis failed: {result.error_details}")
    
    asyncio.run(main())