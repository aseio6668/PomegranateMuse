"""
Monitoring and Observability Integration for PomegranteMuse
Integrates with Datadog, New Relic, Prometheus, and other monitoring platforms
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    DISTRIBUTION = "distribution"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    LOW = "low"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class Alert:
    """Alert configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 0.1"
    threshold: float
    severity: AlertSeverity
    description: str = ""
    tags: Dict[str, str] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.notification_channels is None:
            self.notification_channels = []

@dataclass
class Dashboard:
    """Monitoring dashboard configuration"""
    name: str
    description: str
    widgets: List[Dict[str, Any]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.widgets is None:
            self.widgets = []
        if self.tags is None:
            self.tags = []

class MonitoringIntegration:
    """Base class for monitoring integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def send_metric(self, metric: Metric) -> bool:
        """Send metric to monitoring platform"""
        raise NotImplementedError
        
    async def send_metrics(self, metrics: List[Metric]) -> bool:
        """Send multiple metrics"""
        raise NotImplementedError
        
    async def create_alert(self, alert: Alert) -> str:
        """Create alert rule"""
        raise NotImplementedError
        
    async def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create monitoring dashboard"""
        raise NotImplementedError
        
    async def query_metrics(self, query: str, start_time: datetime, 
                           end_time: datetime) -> List[Dict[str, Any]]:
        """Query metrics from platform"""
        raise NotImplementedError

class DatadogIntegration(MonitoringIntegration):
    """Datadog monitoring integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.app_key = config.get("app_key")
        self.api_url = "https://api.datadoghq.com/api/v1"
        
    async def send_metric(self, metric: Metric) -> bool:
        """Send single metric to Datadog"""
        return await self.send_metrics([metric])
    
    async def send_metrics(self, metrics: List[Metric]) -> bool:
        """Send metrics to Datadog"""
        if not REQUESTS_AVAILABLE or not self.api_key:
            return False
            
        try:
            headers = {
                "Content-Type": "application/json",
                "DD-API-KEY": self.api_key
            }
            
            series = []
            for metric in metrics:
                tags = [f"{k}:{v}" for k, v in metric.tags.items()]
                
                series.append({
                    "metric": metric.name,
                    "points": [[int(metric.timestamp.timestamp()), metric.value]],
                    "type": metric.metric_type.value,
                    "tags": tags
                })
            
            payload = {"series": series}
            
            response = requests.post(
                f"{self.api_url}/series",
                headers=headers,
                json=payload
            )
            
            return response.status_code == 202
            
        except Exception as e:
            self.logger.error(f"Failed to send Datadog metrics: {e}")
            return False
    
    async def create_alert(self, alert: Alert) -> str:
        """Create Datadog monitor"""
        if not REQUESTS_AVAILABLE or not self.api_key or not self.app_key:
            return ""
            
        try:
            headers = {
                "Content-Type": "application/json",
                "DD-API-KEY": self.api_key,
                "DD-APPLICATION-KEY": self.app_key
            }
            
            # Build query based on condition
            query = f"avg(last_5m):{alert.metric_name} {alert.condition} {alert.threshold}"
            
            monitor_data = {
                "name": alert.name,
                "type": "metric alert",
                "query": query,
                "message": alert.description,
                "tags": [f"{k}:{v}" for k, v in alert.tags.items()],
                "options": {
                    "notify_audit": False,
                    "locked": False,
                    "timeout_h": 0,
                    "silenced": {},
                    "include_tags": True,
                    "new_host_delay": 300,
                    "require_full_window": False,
                    "notify_no_data": False,
                    "renotify_interval": 0,
                    "evaluation_delay": 0,
                    "escalation_message": "",
                    "no_data_timeframe": None
                }
            }
            
            response = requests.post(
                f"{self.api_url}/monitor",
                headers=headers,
                json=monitor_data
            )
            
            if response.status_code == 200:
                return str(response.json()["id"])
            else:
                self.logger.error(f"Failed to create Datadog monitor: {response.text}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to create Datadog monitor: {e}")
            return ""
    
    async def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create Datadog dashboard"""
        if not REQUESTS_AVAILABLE or not self.api_key or not self.app_key:
            return ""
            
        try:
            headers = {
                "Content-Type": "application/json",
                "DD-API-KEY": self.api_key,
                "DD-APPLICATION-KEY": self.app_key
            }
            
            dashboard_data = {
                "title": dashboard.name,
                "description": dashboard.description,
                "widgets": dashboard.widgets,
                "layout_type": "ordered",
                "is_read_only": False,
                "notify_list": [],
                "template_variables": []
            }
            
            response = requests.post(
                f"{self.api_url}/dashboard",
                headers=headers,
                json=dashboard_data
            )
            
            if response.status_code == 200:
                return response.json()["id"]
            else:
                self.logger.error(f"Failed to create Datadog dashboard: {response.text}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to create Datadog dashboard: {e}")
            return ""

class NewRelicIntegration(MonitoringIntegration):
    """New Relic monitoring integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.api_url = "https://api.newrelic.com"
        
    async def send_metric(self, metric: Metric) -> bool:
        """Send metric to New Relic"""
        return await self.send_metrics([metric])
    
    async def send_metrics(self, metrics: List[Metric]) -> bool:
        """Send metrics to New Relic"""
        if not REQUESTS_AVAILABLE or not self.api_key:
            return False
            
        try:
            headers = {
                "Content-Type": "application/json",
                "Api-Key": self.api_key
            }
            
            # New Relic uses a different format
            metric_data = []
            for metric in metrics:
                metric_data.append({
                    "name": metric.name,
                    "type": metric.metric_type.value,
                    "value": metric.value,
                    "timestamp": int(metric.timestamp.timestamp()),
                    "attributes": metric.tags
                })
            
            payload = [{"metrics": metric_data}]
            
            response = requests.post(
                f"{self.api_url}/metric/v1",
                headers=headers,
                json=payload
            )
            
            return response.status_code == 202
            
        except Exception as e:
            self.logger.error(f"Failed to send New Relic metrics: {e}")
            return False

class PrometheusIntegration(MonitoringIntegration):
    """Prometheus monitoring integration (via pushgateway)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pushgateway_url = config.get("pushgateway_url", "http://localhost:9091")
        self.job_name = config.get("job_name", "pomegrantemuse")
        
    async def send_metric(self, metric: Metric) -> bool:
        """Send metric to Prometheus pushgateway"""
        return await self.send_metrics([metric])
    
    async def send_metrics(self, metrics: List[Metric]) -> bool:
        """Send metrics to Prometheus pushgateway"""
        if not REQUESTS_AVAILABLE:
            return False
            
        try:
            # Convert metrics to Prometheus format
            prometheus_data = []
            for metric in metrics:
                labels = ",".join([f'{k}="{v}"' for k, v in metric.tags.items()])
                if labels:
                    metric_line = f"{metric.name}{{{labels}}} {metric.value}"
                else:
                    metric_line = f"{metric.name} {metric.value}"
                prometheus_data.append(metric_line)
            
            data = "\n".join(prometheus_data)
            
            # Build URL with job and instance labels
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            if "instance" in self.config:
                url += f"/instance/{self.config['instance']}"
            
            response = requests.post(url, data=data)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to send Prometheus metrics: {e}")
            return False

class MetricsCollector:
    """Metrics collection and aggregation"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.start_time = datetime.now()
        
    def add_metric(self, name: str, value: Union[int, float], 
                  metric_type: MetricType = MetricType.GAUGE,
                  tags: Dict[str, str] = None, unit: str = ""):
        """Add a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.metrics.append(metric)
        
    def increment_counter(self, name: str, value: int = 1, 
                         tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.add_metric(name, value, MetricType.COUNTER, tags)
        
    def set_gauge(self, name: str, value: Union[int, float], 
                 tags: Dict[str, str] = None, unit: str = ""):
        """Set a gauge metric"""
        self.add_metric(name, value, MetricType.GAUGE, tags, unit)
        
    def record_timing(self, name: str, duration: float, 
                     tags: Dict[str, str] = None):
        """Record timing metric"""
        self.add_metric(name, duration, MetricType.HISTOGRAM, tags, "seconds")
        
    def get_metrics(self) -> List[Metric]:
        """Get all collected metrics"""
        return self.metrics.copy()
        
    def clear_metrics(self):
        """Clear collected metrics"""
        self.metrics.clear()

class AlertManager:
    """Alert management and evaluation"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
    def add_alert(self, alert: Alert):
        """Add alert rule"""
        self.alerts.append(alert)
        
    def evaluate_alerts(self, metrics: List[Metric]) -> List[Dict[str, Any]]:
        """Evaluate alert conditions against metrics"""
        triggered_alerts = []
        
        for alert in self.alerts:
            # Find metrics matching the alert
            matching_metrics = [m for m in metrics if m.name == alert.metric_name]
            
            for metric in matching_metrics:
                if self._evaluate_condition(metric.value, alert.condition, alert.threshold):
                    alert_key = f"{alert.name}_{metric.name}"
                    
                    # Check if this is a new alert or existing one
                    if alert_key not in self.active_alerts:
                        triggered_alert = {
                            "alert": alert,
                            "metric": metric,
                            "triggered_at": datetime.now(),
                            "status": "triggered"
                        }
                        
                        self.active_alerts[alert_key] = triggered_alert
                        triggered_alerts.append(triggered_alert)
                        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith(">="):
                return value >= threshold
            elif condition.startswith("<="):
                return value <= threshold
            elif condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith("=="):
                return value == threshold
            elif condition.startswith("!="):
                return value != threshold
            else:
                return False
        except:
            return False
    
    def resolve_alert(self, alert_key: str):
        """Mark alert as resolved"""
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key]["status"] = "resolved"
            self.active_alerts[alert_key]["resolved_at"] = datetime.now()

class MonitoringDashboard:
    """Monitoring dashboard builder"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.widgets = []
        
    def add_timeseries_widget(self, title: str, metrics: List[str], 
                            tags: Dict[str, str] = None):
        """Add timeseries chart widget"""
        widget = {
            "definition": {
                "type": "timeseries",
                "title": title,
                "requests": [
                    {
                        "q": metric,
                        "display_type": "line",
                        "style": {"palette": "dog_classic"}
                    }
                    for metric in metrics
                ],
                "yaxis": {"scale": "linear"},
                "legend": {"show": True}
            }
        }
        self.widgets.append(widget)
        
    def add_query_value_widget(self, title: str, metric: str, 
                             aggregation: str = "avg"):
        """Add single value widget"""
        widget = {
            "definition": {
                "type": "query_value",
                "title": title,
                "requests": [{
                    "q": f"{aggregation}({metric})",
                    "aggregator": aggregation
                }],
                "autoscale": True,
                "precision": 2
            }
        }
        self.widgets.append(widget)
        
    def add_heatmap_widget(self, title: str, metric: str):
        """Add heatmap widget"""
        widget = {
            "definition": {
                "type": "heatmap",
                "title": title,
                "requests": [{
                    "q": metric,
                    "style": {"palette": "dog_classic"}
                }]
            }
        }
        self.widgets.append(widget)
        
    def to_dashboard(self) -> Dashboard:
        """Convert to Dashboard object"""
        return Dashboard(
            name=self.name,
            description=self.description,
            widgets=self.widgets
        )

class MonitoringManager:
    """Main monitoring integration manager"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/monitoring")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.integrations: Dict[str, MonitoringIntegration] = {}
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(__name__)
        
    def add_datadog_integration(self, name: str, config: Dict[str, Any]):
        """Add Datadog integration"""
        self.integrations[name] = DatadogIntegration(config)
        
    def add_newrelic_integration(self, name: str, config: Dict[str, Any]):
        """Add New Relic integration"""
        self.integrations[name] = NewRelicIntegration(config)
        
    def add_prometheus_integration(self, name: str, config: Dict[str, Any]):
        """Add Prometheus integration"""
        self.integrations[name] = PrometheusIntegration(config)
        
    async def send_metrics(self, metrics: List[Metric] = None) -> Dict[str, bool]:
        """Send metrics to all configured integrations"""
        if metrics is None:
            metrics = self.collector.get_metrics()
            
        results = {}
        for name, integration in self.integrations.items():
            try:
                success = await integration.send_metrics(metrics)
                results[name] = success
            except Exception as e:
                self.logger.error(f"Failed to send metrics to {name}: {e}")
                results[name] = False
                
        return results
    
    def track_build_metrics(self, success: bool, duration: float, 
                          project: str, language: str):
        """Track build-related metrics"""
        tags = {"project": project, "language": language}
        
        self.collector.increment_counter("pomegrantemuse.builds.total", 1, tags)
        
        if success:
            self.collector.increment_counter("pomegrantemuse.builds.success", 1, tags)
        else:
            self.collector.increment_counter("pomegrantemuse.builds.failure", 1, tags)
            
        self.collector.record_timing("pomegrantemuse.build.duration", duration, tags)
    
    def track_migration_metrics(self, files_processed: int, 
                              source_lang: str, target_lang: str,
                              duration: float, errors: int = 0):
        """Track migration-related metrics"""
        tags = {
            "source_language": source_lang,
            "target_language": target_lang
        }
        
        self.collector.increment_counter("pomegrantemuse.migrations.total", 1, tags)
        self.collector.set_gauge("pomegrantemuse.migration.files_processed", files_processed, tags)
        self.collector.set_gauge("pomegrantemuse.migration.errors", errors, tags)
        self.collector.record_timing("pomegrantemuse.migration.duration", duration, tags)
    
    def track_security_metrics(self, vulnerabilities_found: int, 
                             severity_counts: Dict[str, int],
                             project: str):
        """Track security-related metrics"""
        base_tags = {"project": project}
        
        self.collector.set_gauge("pomegrantemuse.security.vulnerabilities.total", 
                               vulnerabilities_found, base_tags)
        
        for severity, count in severity_counts.items():
            tags = {**base_tags, "severity": severity}
            self.collector.set_gauge("pomegrantemuse.security.vulnerabilities.by_severity", 
                                   count, tags)
    
    def setup_default_alerts(self):
        """Setup default alert rules"""
        alerts = [
            Alert(
                name="Build Failure Rate High",
                metric_name="pomegrantemuse.builds.failure_rate",
                condition="> 0.2",
                threshold=0.2,
                severity=AlertSeverity.WARNING,
                description="Build failure rate is above 20%"
            ),
            Alert(
                name="Build Duration High",
                metric_name="pomegrantemuse.build.duration",
                condition="> 300",
                threshold=300,
                severity=AlertSeverity.WARNING,
                description="Build duration exceeds 5 minutes"
            ),
            Alert(
                name="Critical Vulnerabilities Found",
                metric_name="pomegrantemuse.security.vulnerabilities.critical",
                condition="> 0",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                description="Critical security vulnerabilities detected"
            )
        ]
        
        for alert in alerts:
            self.alert_manager.add_alert(alert)
    
    def create_default_dashboard(self) -> Dashboard:
        """Create default monitoring dashboard"""
        dashboard = MonitoringDashboard(
            "PomegranteMuse Monitoring",
            "Main monitoring dashboard for PomegranteMuse operations"
        )
        
        # Build metrics
        dashboard.add_timeseries_widget(
            "Build Success Rate",
            ["pomegrantemuse.builds.success_rate"]
        )
        
        dashboard.add_query_value_widget(
            "Total Builds Today",
            "pomegrantemuse.builds.total"
        )
        
        # Migration metrics
        dashboard.add_timeseries_widget(
            "Files Migrated",
            ["pomegrantemuse.migration.files_processed"]
        )
        
        dashboard.add_heatmap_widget(
            "Migration Duration Distribution",
            "pomegrantemuse.migration.duration"
        )
        
        # Security metrics
        dashboard.add_timeseries_widget(
            "Security Vulnerabilities",
            ["pomegrantemuse.security.vulnerabilities.total"]
        )
        
        return dashboard.to_dashboard()

async def setup_monitoring(config: Dict[str, Any]) -> MonitoringManager:
    """Setup monitoring integrations from configuration"""
    manager = MonitoringManager()
    
    # Setup integrations
    if "datadog" in config:
        manager.add_datadog_integration("datadog", config["datadog"])
        
    if "newrelic" in config:
        manager.add_newrelic_integration("newrelic", config["newrelic"])
        
    if "prometheus" in config:
        manager.add_prometheus_integration("prometheus", config["prometheus"])
    
    # Setup default alerts if enabled
    if config.get("enable_default_alerts", True):
        manager.setup_default_alerts()
        
    return manager