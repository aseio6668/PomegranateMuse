"""
Cloud-Native Infrastructure Generation for Universal Code Modernization Platform
Generates Docker, Kubernetes, Terraform, and cloud-specific infrastructure code
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from language_targets import TargetLanguage, CodeGenerationContext


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITAL_OCEAN = "digitalocean"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class InfrastructureType(Enum):
    """Types of infrastructure components"""
    CONTAINER = "container"
    KUBERNETES = "kubernetes"
    SERVERLESS = "serverless"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORKING = "networking"
    MONITORING = "monitoring"
    SECURITY = "security"
    CI_CD = "ci_cd"


@dataclass
class InfrastructureRequirements:
    """Requirements for infrastructure generation"""
    target_language: TargetLanguage
    application_type: str  # "web", "api", "microservice", "batch", "ml"
    cloud_provider: CloudProvider
    deployment_strategy: str  # "rolling", "blue_green", "canary"
    
    # Scaling requirements
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    
    # Storage requirements
    persistent_storage: bool = False
    storage_size: str = "10Gi"
    storage_class: str = "standard"
    
    # Database requirements
    database_type: Optional[str] = None  # "postgres", "mysql", "mongodb", "redis"
    database_ha: bool = False
    
    # Security requirements
    enable_security_scanning: bool = True
    enable_network_policies: bool = True
    enable_rbac: bool = True
    ssl_enabled: bool = True
    
    # Monitoring requirements
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_tracing: bool = False
    
    # Performance requirements
    performance_tier: str = "standard"  # "basic", "standard", "premium"
    
    # Cost optimization
    cost_optimization: bool = True
    spot_instances: bool = False
    
    # Custom configurations
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)


class DockerGenerator:
    """Generates Docker configurations"""
    
    def __init__(self):
        self.base_images = {
            TargetLanguage.RUST: {
                "build": "rust:1.75-alpine",
                "runtime": "alpine:latest"
            },
            TargetLanguage.GO: {
                "build": "golang:1.21-alpine",
                "runtime": "alpine:latest"
            },
            TargetLanguage.TYPESCRIPT: {
                "build": "node:18-alpine",
                "runtime": "node:18-alpine"
            },
            TargetLanguage.PYTHON: {
                "build": "python:3.12-alpine",
                "runtime": "python:3.12-alpine"
            }
        }
    
    def generate_dockerfile(self, requirements: InfrastructureRequirements) -> str:
        """Generate optimized Dockerfile"""
        language = requirements.target_language
        base_images = self.base_images.get(language, self.base_images[TargetLanguage.PYTHON])
        
        if language == TargetLanguage.RUST:
            return self._generate_rust_dockerfile(requirements, base_images)
        elif language == TargetLanguage.GO:
            return self._generate_go_dockerfile(requirements, base_images)
        elif language == TargetLanguage.TYPESCRIPT:
            return self._generate_typescript_dockerfile(requirements, base_images)
        elif language == TargetLanguage.PYTHON:
            return self._generate_python_dockerfile(requirements, base_images)
        else:
            return self._generate_generic_dockerfile(requirements, base_images)
    
    def _generate_rust_dockerfile(self, requirements: InfrastructureRequirements, base_images: Dict[str, str]) -> str:
        """Generate Rust-specific multi-stage Dockerfile"""
        env_vars = self._format_env_vars(requirements.environment_variables)
        
        return f'''# Build stage
FROM {base_images["build"]} AS builder

# Install dependencies
RUN apk add --no-cache musl-dev openssl-dev

WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Create dummy main to build dependencies
RUN mkdir src && echo "fn main() {{}}" > src/main.rs

# Build dependencies (cached layer)
RUN cargo build --release && rm -rf src/

# Copy source code
COPY src/ src/

# Build application
RUN cargo build --release --bin app

# Runtime stage
FROM {base_images["runtime"]} AS runtime

# Install CA certificates for HTTPS
RUN apk add --no-cache ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \\
    adduser -S appuser -u 1001 -G appgroup

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/app ./app

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set environment variables
{env_vars}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD ./app --health-check || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["./app"]
'''
    
    def _generate_go_dockerfile(self, requirements: InfrastructureRequirements, base_images: Dict[str, str]) -> str:
        """Generate Go-specific multi-stage Dockerfile"""
        env_vars = self._format_env_vars(requirements.environment_variables)
        
        return f'''# Build stage
FROM {base_images["build"]} AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build binary with optimizations
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-extldflags "-static"' -o main .

# Runtime stage
FROM {base_images["runtime"]} AS runtime

# Install CA certificates
RUN apk add --no-cache ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \\
    adduser -S appuser -u 1001 -G appgroup

WORKDIR /app

# Copy binary
COPY --from=builder /app/main ./main

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set environment variables
{env_vars}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD ./main -health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["./main"]
'''
    
    def _generate_typescript_dockerfile(self, requirements: InfrastructureRequirements, base_images: Dict[str, str]) -> str:
        """Generate TypeScript/Node.js Dockerfile"""
        env_vars = self._format_env_vars(requirements.environment_variables)
        
        return f'''# Build stage
FROM {base_images["build"]} AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source and build
COPY . .
RUN npm run build

# Runtime stage
FROM {base_images["runtime"]} AS runtime

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \\
    adduser -S nodejs -u 1001 -G nodejs

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application
COPY --from=builder /app/dist ./dist

# Set ownership
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Set environment variables
{env_vars}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD node dist/health-check.js || exit 1

# Expose port
EXPOSE 3000

# Run application
CMD ["node", "dist/index.js"]
'''
    
    def _generate_python_dockerfile(self, requirements: InfrastructureRequirements, base_images: Dict[str, str]) -> str:
        """Generate Python Dockerfile"""
        env_vars = self._format_env_vars(requirements.environment_variables)
        
        return f'''# Build stage
FROM {base_images["build"]} AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM {base_images["runtime"]} AS runtime

# Install runtime dependencies
RUN apk add --no-cache libffi

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \\
    adduser -S appuser -u 1001 -G appgroup

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Set environment variables
{env_vars}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python health_check.py || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
'''
    
    def _generate_generic_dockerfile(self, requirements: InfrastructureRequirements, base_images: Dict[str, str]) -> str:
        """Generate generic Dockerfile"""
        env_vars = self._format_env_vars(requirements.environment_variables)
        
        return f'''FROM {base_images["runtime"]}

WORKDIR /app

# Copy application
COPY . .

# Set environment variables
{env_vars}

# Expose port
EXPOSE 8080

# Run application
CMD ["./app"]
'''
    
    def _format_env_vars(self, env_vars: Dict[str, str]) -> str:
        """Format environment variables for Dockerfile"""
        if not env_vars:
            return "# No environment variables"
        
        formatted = []
        for key, value in env_vars.items():
            formatted.append(f"ENV {key}={value}")
        
        return "\n".join(formatted)
    
    def generate_dockerignore(self, requirements: InfrastructureRequirements) -> str:
        """Generate .dockerignore file"""
        language = requirements.target_language
        
        common_ignores = [
            ".git",
            ".gitignore",
            "README.md",
            "Dockerfile",
            ".dockerignore",
            "node_modules",
            "*.log",
            ".env*",
            ".vscode",
            ".idea",
            "*.tmp",
            "*.swp"
        ]
        
        language_specific = {
            TargetLanguage.RUST: ["target/", "Cargo.lock"],
            TargetLanguage.GO: ["vendor/", "*.exe"],
            TargetLanguage.TYPESCRIPT: ["dist/", "coverage/", "*.tsbuildinfo"],
            TargetLanguage.PYTHON: ["__pycache__/", "*.pyc", ".pytest_cache/", "venv/"]
        }
        
        ignores = common_ignores + language_specific.get(language, [])
        return "\n".join(ignores)
    
    def generate_docker_compose(self, requirements: InfrastructureRequirements) -> str:
        """Generate docker-compose.yml for local development"""
        services = {
            "app": {
                "build": ".",
                "ports": [f"{self._get_default_port(requirements.target_language)}:{self._get_default_port(requirements.target_language)}"],
                "environment": requirements.environment_variables,
                "depends_on": [],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", f"http://localhost:{self._get_default_port(requirements.target_language)}/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "40s"
                }
            }
        }
        
        # Add database if required
        if requirements.database_type:
            db_config = self._get_database_config(requirements.database_type, requirements.database_ha)
            services.update(db_config)
            services["app"]["depends_on"].extend(list(db_config.keys()))
        
        # Add Redis for caching
        if requirements.application_type in ["web", "api"]:
            services["redis"] = {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "5s",
                    "timeout": "3s",
                    "retries": 5
                }
            }
            services["app"]["depends_on"].append("redis")
        
        compose_content = {
            "version": "3.8",
            "services": services,
            "networks": {
                "app-network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Add all services to network
        for service in services:
            services[service]["networks"] = ["app-network"]
        
        if requirements.persistent_storage:
            compose_content["volumes"] = {
                "app-data": {}
            }
            services["app"]["volumes"] = ["app-data:/app/data"]
        
        return yaml.dump(compose_content, default_flow_style=False, indent=2)
    
    def _get_default_port(self, language: TargetLanguage) -> int:
        """Get default port for language"""
        port_mapping = {
            TargetLanguage.RUST: 8080,
            TargetLanguage.GO: 8080,
            TargetLanguage.TYPESCRIPT: 3000,
            TargetLanguage.PYTHON: 8000
        }
        return port_mapping.get(language, 8080)
    
    def _get_database_config(self, db_type: str, ha: bool) -> Dict[str, Any]:
        """Get database configuration for docker-compose"""
        configs = {
            "postgres": {
                "postgres": {
                    "image": "postgres:15-alpine",
                    "environment": {
                        "POSTGRES_DB": "appdb",
                        "POSTGRES_USER": "appuser",
                        "POSTGRES_PASSWORD": "apppassword"
                    },
                    "ports": ["5432:5432"],
                    "volumes": ["postgres_data:/var/lib/postgresql/data"],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U appuser -d appdb"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            },
            "mysql": {
                "mysql": {
                    "image": "mysql:8.0",
                    "environment": {
                        "MYSQL_DATABASE": "appdb",
                        "MYSQL_USER": "appuser",
                        "MYSQL_PASSWORD": "apppassword",
                        "MYSQL_ROOT_PASSWORD": "rootpassword"
                    },
                    "ports": ["3306:3306"],
                    "volumes": ["mysql_data:/var/lib/mysql"],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "mysqladmin", "ping", "-h", "localhost"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            },
            "mongodb": {
                "mongodb": {
                    "image": "mongo:7",
                    "environment": {
                        "MONGO_INITDB_DATABASE": "appdb",
                        "MONGO_INITDB_ROOT_USERNAME": "appuser",
                        "MONGO_INITDB_ROOT_PASSWORD": "apppassword"
                    },
                    "ports": ["27017:27017"],
                    "volumes": ["mongodb_data:/data/db"],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            }
        }
        
        return configs.get(db_type, {})


class KubernetesGenerator:
    """Generates Kubernetes manifests"""
    
    def __init__(self):
        self.api_version = "apps/v1"
    
    def generate_all_manifests(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate all Kubernetes manifests"""
        manifests = {}
        
        # Core application manifests
        manifests["deployment.yaml"] = self.generate_deployment(requirements)
        manifests["service.yaml"] = self.generate_service(requirements)
        
        # Configuration manifests
        if requirements.config_maps:
            manifests["configmap.yaml"] = self.generate_configmap(requirements)
        
        if requirements.secrets:
            manifests["secret.yaml"] = self.generate_secret(requirements)
        
        # Networking manifests
        manifests["ingress.yaml"] = self.generate_ingress(requirements)
        
        if requirements.enable_network_policies:
            manifests["network-policy.yaml"] = self.generate_network_policy(requirements)
        
        # Storage manifests
        if requirements.persistent_storage:
            manifests["pvc.yaml"] = self.generate_pvc(requirements)
        
        # Security manifests
        if requirements.enable_rbac:
            manifests["rbac.yaml"] = self.generate_rbac(requirements)
        
        # Scaling manifests
        manifests["hpa.yaml"] = self.generate_hpa(requirements)
        
        # Monitoring manifests
        if requirements.enable_monitoring:
            manifests["service-monitor.yaml"] = self.generate_service_monitor(requirements)
        
        # Database manifests
        if requirements.database_type:
            db_manifests = self.generate_database_manifests(requirements)
            manifests.update(db_manifests)
        
        return manifests
    
    def generate_deployment(self, requirements: InfrastructureRequirements) -> str:
        """Generate Kubernetes Deployment manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        deployment = {
            "apiVersion": self.api_version,
            "kind": "Deployment",
            "metadata": {
                "name": app_name,
                "labels": {
                    "app": app_name,
                    "version": "v1",
                    "component": "backend",
                    "language": requirements.target_language.value
                }
            },
            "spec": {
                "replicas": requirements.min_replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_name,
                            "version": "v1"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8080",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": app_name,
                        "securityContext": {
                            "fsGroup": 1001,
                            "runAsNonRoot": True,
                            "runAsUser": 1001
                        },
                        "containers": [{
                            "name": app_name,
                            "image": f"{app_name}:latest",
                            "imagePullPolicy": "Always",
                            "ports": [{
                                "containerPort": self._get_container_port(requirements.target_language),
                                "protocol": "TCP"
                            }],
                            "env": self._generate_env_vars(requirements),
                            "resources": {
                                "requests": {
                                    "cpu": requirements.cpu_request,
                                    "memory": requirements.memory_request
                                },
                                "limits": {
                                    "cpu": requirements.cpu_limit,
                                    "memory": requirements.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self._get_container_port(requirements.target_language)
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": self._get_container_port(requirements.target_language)
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {
                                    "drop": ["ALL"]
                                },
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": True,
                                "runAsUser": 1001
                            }
                        }],
                        "restartPolicy": "Always",
                        "terminationGracePeriodSeconds": 30
                    }
                }
            }
        }
        
        # Add volume mounts if persistent storage is required
        if requirements.persistent_storage:
            deployment["spec"]["template"]["spec"]["volumes"] = [{
                "name": "app-storage",
                "persistentVolumeClaim": {
                    "claimName": f"{app_name}-pvc"
                }
            }]
            deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [{
                "name": "app-storage",
                "mountPath": "/app/data"
            }]
        
        return yaml.dump(deployment, default_flow_style=False, indent=2)
    
    def generate_service(self, requirements: InfrastructureRequirements) -> str:
        """Generate Kubernetes Service manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{app_name}-service",
                "labels": {
                    "app": app_name
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{
                    "port": 80,
                    "targetPort": self._get_container_port(requirements.target_language),
                    "protocol": "TCP",
                    "name": "http"
                }],
                "selector": {
                    "app": app_name
                }
            }
        }
        
        return yaml.dump(service, default_flow_style=False, indent=2)
    
    def generate_ingress(self, requirements: InfrastructureRequirements) -> str:
        """Generate Kubernetes Ingress manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{app_name}-ingress",
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true" if requirements.ssl_enabled else "false",
                    "nginx.ingress.kubernetes.io/force-ssl-redirect": "true" if requirements.ssl_enabled else "false",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod" if requirements.ssl_enabled else ""
                }
            },
            "spec": {
                "rules": [{
                    "host": f"{app_name}.example.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{app_name}-service",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if requirements.ssl_enabled:
            ingress["spec"]["tls"] = [{
                "hosts": [f"{app_name}.example.com"],
                "secretName": f"{app_name}-tls"
            }]
        
        return yaml.dump(ingress, default_flow_style=False, indent=2)
    
    def generate_hpa(self, requirements: InfrastructureRequirements) -> str:
        """Generate Horizontal Pod Autoscaler manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{app_name}-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": self.api_version,
                    "kind": "Deployment",
                    "name": app_name
                },
                "minReplicas": requirements.min_replicas,
                "maxReplicas": requirements.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 10,
                            "periodSeconds": 60
                        }]
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 0,
                        "policies": [{
                            "type": "Percent",
                            "value": 50,
                            "periodSeconds": 15
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(hpa, default_flow_style=False, indent=2)
    
    def generate_configmap(self, requirements: InfrastructureRequirements) -> str:
        """Generate ConfigMap manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{app_name}-config"
            },
            "data": {}
        }
        
        # Add all config maps
        for name, config in requirements.config_maps.items():
            configmap["data"].update(config)
        
        return yaml.dump(configmap, default_flow_style=False, indent=2)
    
    def generate_secret(self, requirements: InfrastructureRequirements) -> str:
        """Generate Secret manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{app_name}-secret"
            },
            "type": "Opaque",
            "data": {}
        }
        
        # Add secret placeholders (base64 encoded)
        for secret_name in requirements.secrets:
            # In real implementation, these would be properly encoded
            secret["data"][secret_name] = "Y2hhbmdlLW1l"  # "change-me" in base64
        
        return yaml.dump(secret, default_flow_style=False, indent=2)
    
    def generate_pvc(self, requirements: InfrastructureRequirements) -> str:
        """Generate PersistentVolumeClaim manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{app_name}-pvc"
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": requirements.storage_class,
                "resources": {
                    "requests": {
                        "storage": requirements.storage_size
                    }
                }
            }
        }
        
        return yaml.dump(pvc, default_flow_style=False, indent=2)
    
    def generate_rbac(self, requirements: InfrastructureRequirements) -> str:
        """Generate RBAC manifests"""
        app_name = f"app-{requirements.target_language.value}"
        
        rbac_manifests = []
        
        # ServiceAccount
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": app_name,
                "namespace": "default"
            }
        }
        rbac_manifests.append(service_account)
        
        # Role
        role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": f"{app_name}-role",
                "namespace": "default"
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods", "services", "configmaps", "secrets"],
                    "verbs": ["get", "list", "watch"]
                }
            ]
        }
        rbac_manifests.append(role)
        
        # RoleBinding
        role_binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": f"{app_name}-rolebinding",
                "namespace": "default"
            },
            "subjects": [{
                "kind": "ServiceAccount",
                "name": app_name,
                "namespace": "default"
            }],
            "roleRef": {
                "kind": "Role",
                "name": f"{app_name}-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
        rbac_manifests.append(role_binding)
        
        return yaml.dump_all(rbac_manifests, default_flow_style=False, indent=2)
    
    def generate_network_policy(self, requirements: InfrastructureRequirements) -> str:
        """Generate NetworkPolicy manifest"""
        app_name = f"app-{requirements.target_language.value}"
        
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{app_name}-network-policy"
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [{
                            "protocol": "TCP",
                            "port": self._get_container_port(requirements.target_language)
                        }]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 53},
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 80}
                        ]
                    }
                ]
            }
        }
        
        # Add database egress if database is configured
        if requirements.database_type:
            db_port = self._get_database_port(requirements.database_type)
            network_policy["spec"]["egress"].append({
                "to": [{
                    "podSelector": {
                        "matchLabels": {
                            "app": f"{requirements.database_type}-database"
                        }
                    }
                }],
                "ports": [{
                    "protocol": "TCP",
                    "port": db_port
                }]
            })
        
        return yaml.dump(network_policy, default_flow_style=False, indent=2)
    
    def generate_service_monitor(self, requirements: InfrastructureRequirements) -> str:
        """Generate ServiceMonitor for Prometheus"""
        app_name = f"app-{requirements.target_language.value}"
        
        service_monitor = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{app_name}-metrics",
                "labels": {
                    "app": app_name
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "endpoints": [{
                    "port": "http",
                    "path": "/metrics",
                    "interval": "30s",
                    "scrapeTimeout": "10s"
                }]
            }
        }
        
        return yaml.dump(service_monitor, default_flow_style=False, indent=2)
    
    def generate_database_manifests(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate database manifests"""
        manifests = {}
        db_type = requirements.database_type
        
        if db_type == "postgres":
            manifests.update(self._generate_postgres_manifests(requirements))
        elif db_type == "mysql":
            manifests.update(self._generate_mysql_manifests(requirements))
        elif db_type == "mongodb":
            manifests.update(self._generate_mongodb_manifests(requirements))
        elif db_type == "redis":
            manifests.update(self._generate_redis_manifests(requirements))
        
        return manifests
    
    def _generate_postgres_manifests(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate PostgreSQL manifests"""
        manifests = {}
        
        # StatefulSet for PostgreSQL
        postgres_statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "postgres-database"
            },
            "spec": {
                "serviceName": "postgres-service",
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "postgres-database"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "postgres-database"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "postgres",
                            "image": "postgres:15-alpine",
                            "env": [
                                {"name": "POSTGRES_DB", "value": "appdb"},
                                {"name": "POSTGRES_USER", "value": "appuser"},
                                {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "postgres-secret", "key": "password"}}}
                            ],
                            "ports": [{"containerPort": 5432}],
                            "volumeMounts": [{
                                "name": "postgres-storage",
                                "mountPath": "/var/lib/postgresql/data"
                            }],
                            "livenessProbe": {
                                "exec": {
                                    "command": ["pg_isready", "-U", "appuser", "-d", "appdb"]
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {
                        "name": "postgres-storage"
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": requirements.storage_class,
                        "resources": {
                            "requests": {
                                "storage": "20Gi"
                            }
                        }
                    }
                }]
            }
        }
        
        # Service for PostgreSQL
        postgres_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "postgres-service"
            },
            "spec": {
                "ports": [{
                    "port": 5432,
                    "targetPort": 5432
                }],
                "selector": {
                    "app": "postgres-database"
                },
                "clusterIP": "None"
            }
        }
        
        # Secret for PostgreSQL
        postgres_secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "postgres-secret"
            },
            "type": "Opaque",
            "data": {
                "password": "YXBwcGFzc3dvcmQ="  # "apppassword" in base64
            }
        }
        
        manifests["postgres-statefulset.yaml"] = yaml.dump(postgres_statefulset, default_flow_style=False, indent=2)
        manifests["postgres-service.yaml"] = yaml.dump(postgres_service, default_flow_style=False, indent=2)
        manifests["postgres-secret.yaml"] = yaml.dump(postgres_secret, default_flow_style=False, indent=2)
        
        return manifests
    
    def _generate_env_vars(self, requirements: InfrastructureRequirements) -> List[Dict[str, Any]]:
        """Generate environment variables for deployment"""
        env_vars = []
        
        # Add configured environment variables
        for key, value in requirements.environment_variables.items():
            env_vars.append({"name": key, "value": value})
        
        # Add config map references
        if requirements.config_maps:
            for config_name in requirements.config_maps:
                env_vars.append({
                    "name": f"{config_name.upper()}_CONFIG",
                    "valueFrom": {
                        "configMapKeyRef": {
                            "name": f"app-{requirements.target_language.value}-config",
                            "key": config_name
                        }
                    }
                })
        
        # Add secret references
        for secret_name in requirements.secrets:
            env_vars.append({
                "name": secret_name.upper(),
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"app-{requirements.target_language.value}-secret",
                        "key": secret_name
                    }
                }
            })
        
        return env_vars
    
    def _get_container_port(self, language: TargetLanguage) -> int:
        """Get container port for language"""
        port_mapping = {
            TargetLanguage.RUST: 8080,
            TargetLanguage.GO: 8080,
            TargetLanguage.TYPESCRIPT: 3000,
            TargetLanguage.PYTHON: 8000
        }
        return port_mapping.get(language, 8080)
    
    def _get_database_port(self, db_type: str) -> int:
        """Get database port"""
        port_mapping = {
            "postgres": 5432,
            "mysql": 3306,
            "mongodb": 27017,
            "redis": 6379
        }
        return port_mapping.get(db_type, 5432)


class TerraformGenerator:
    """Generates Terraform infrastructure as code"""
    
    def __init__(self):
        self.supported_providers = {
            CloudProvider.AWS: self._generate_aws_terraform,
            CloudProvider.AZURE: self._generate_azure_terraform,
            CloudProvider.GCP: self._generate_gcp_terraform
        }
    
    def generate_terraform_config(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate Terraform configuration files"""
        provider = requirements.cloud_provider
        
        if provider in self.supported_providers:
            return self.supported_providers[provider](requirements)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def _generate_aws_terraform(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate AWS Terraform configuration"""
        files = {}
        
        # Main configuration
        main_tf = f'''# AWS Infrastructure for {requirements.target_language.value} application
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

data "aws_caller_identity" "current" {{}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "${{var.project_name}}-vpc"
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "main" {{
  vpc_id = aws_vpc.main.id
  
  tags = {{
    Name = "${{var.project_name}}-igw"
  }}
}}

# Public Subnets
resource "aws_subnet" "public" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {{
    Name = "${{var.project_name}}-public-${{count.index + 1}}"
  }}
}}

# Private Subnets
resource "aws_subnet" "private" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 2)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {{
    Name = "${{var.project_name}}-private-${{count.index + 1}}"
  }}
}}

# Route Table for Public Subnets
resource "aws_route_table" "public" {{
  vpc_id = aws_vpc.main.id
  
  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }}
  
  tags = {{
    Name = "${{var.project_name}}-public-rt"
  }}
}}

# Route Table Association for Public Subnets
resource "aws_route_table_association" "public" {{
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}}

# EKS Cluster
resource "aws_eks_cluster" "main" {{
  name     = "${{var.project_name}}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version
  
  vpc_config {{
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.allowed_cidr_blocks
  }}
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy,
  ]
  
  tags = {{
    Name = "${{var.project_name}}-cluster"
  }}
}}

# EKS Node Group
resource "aws_eks_node_group" "main" {{
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${{var.project_name}}-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.private[*].id
  
  capacity_type  = var.node_capacity_type
  instance_types = var.node_instance_types
  
  scaling_config {{
    desired_size = {requirements.min_replicas}
    max_size     = {requirements.max_replicas}
    min_size     = {requirements.min_replicas}
  }}
  
  update_config {{
    max_unavailable = 1
  }}
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_node_group_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.eks_node_group_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.eks_node_group_AmazonEC2ContainerRegistryReadOnly,
  ]
  
  tags = {{
    Name = "${{var.project_name}}-nodes"
  }}
}}

# ECR Repository
resource "aws_ecr_repository" "app" {{
  name                 = "${{var.project_name}}-app"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {{
    scan_on_push = {str(requirements.enable_security_scanning).lower()}
  }}
  
  tags = {{
    Name = "${{var.project_name}}-app"
  }}
}}
'''

        # Variables
        variables_tf = '''variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "myndra-app"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_capacity_type" {
  description = "Capacity type for EKS nodes"
  type        = string
  default     = "ON_DEMAND"
}

variable "node_instance_types" {
  description = "Instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access EKS API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
'''

        # IAM roles
        iam_tf = '''# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster" {
  name = "${var.project_name}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# EKS Node Group IAM Role
resource "aws_iam_role" "eks_node_group" {
  name = "${var.project_name}-eks-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node_group_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "eks_node_group_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "eks_node_group_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group.name
}
'''

        # Outputs
        outputs_tf = '''output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}
'''

        files["main.tf"] = main_tf
        files["variables.tf"] = variables_tf
        files["iam.tf"] = iam_tf
        files["outputs.tf"] = outputs_tf

        # Add database resources if required
        if requirements.database_type:
            files["database.tf"] = self._generate_aws_database_terraform(requirements)

        return files

    def _generate_aws_database_terraform(self, requirements: InfrastructureRequirements) -> str:
        """Generate AWS database Terraform configuration"""
        db_type = requirements.database_type
        
        if db_type == "postgres":
            return '''# RDS PostgreSQL Database
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_eks_cluster.main.vpc_config[0].cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "appdb"
  username = "appuser"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name = "${var.project_name}-postgres"
  }
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}
'''
        
        return ""

    def _generate_azure_terraform(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate Azure Terraform configuration"""
        # Placeholder for Azure implementation
        files = {}
        
        main_tf = f'''# Azure Infrastructure for {requirements.target_language.value} application
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
  }}
}}

provider "azurerm" {{
  features {{}}
}}

# Resource Group
resource "azurerm_resource_group" "main" {{
  name     = "${{var.project_name}}-rg"
  location = var.location
}}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {{
  name                = "${{var.project_name}}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${{var.project_name}}-aks"

  default_node_pool {{
    name       = "default"
    node_count = {requirements.min_replicas}
    vm_size    = "Standard_D2_v2"
  }}

  identity {{
    type = "SystemAssigned"
  }}

  tags = {{
    Environment = "production"
  }}
}}
'''
        
        files["main.tf"] = main_tf
        return files

    def _generate_gcp_terraform(self, requirements: InfrastructureRequirements) -> Dict[str, str]:
        """Generate GCP Terraform configuration"""
        # Placeholder for GCP implementation
        files = {}
        
        main_tf = f'''# GCP Infrastructure for {requirements.target_language.value} application
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = var.project_id
  region  = var.region
}}

# GKE Cluster
resource "google_container_cluster" "primary" {{
  name     = "${{var.project_name}}-gke"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
}}

# Node Pool
resource "google_container_node_pool" "primary_nodes" {{
  name       = "${{google_container_cluster.primary.name}}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = {requirements.min_replicas}

  node_config {{
    preemptible  = {str(requirements.spot_instances).lower()}
    machine_type = "e2-medium"

    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]
  }}
}}
'''
        
        files["main.tf"] = main_tf
        return files


class CloudNativeInfrastructureGenerator:
    """Main class for generating cloud-native infrastructure"""
    
    def __init__(self):
        self.docker_generator = DockerGenerator()
        self.kubernetes_generator = KubernetesGenerator()
        self.terraform_generator = TerraformGenerator()
    
    def generate_complete_infrastructure(self, requirements: InfrastructureRequirements) -> Dict[str, Dict[str, str]]:
        """Generate complete cloud-native infrastructure"""
        infrastructure = {}
        
        # Docker configurations
        infrastructure["docker"] = {
            "Dockerfile": self.docker_generator.generate_dockerfile(requirements),
            ".dockerignore": self.docker_generator.generate_dockerignore(requirements),
            "docker-compose.yml": self.docker_generator.generate_docker_compose(requirements)
        }
        
        # Kubernetes manifests
        infrastructure["kubernetes"] = self.kubernetes_generator.generate_all_manifests(requirements)
        
        # Terraform configurations
        try:
            infrastructure["terraform"] = self.terraform_generator.generate_terraform_config(requirements)
        except ValueError as e:
            print(f"Terraform generation skipped: {e}")
            infrastructure["terraform"] = {}
        
        return infrastructure
    
    def save_infrastructure_files(self, infrastructure: Dict[str, Dict[str, str]], output_dir: Path):
        """Save generated infrastructure files to disk"""
        
        for category, files in infrastructure.items():
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for filename, content in files.items():
                file_path = category_dir / filename
                with open(file_path, 'w') as f:
                    f.write(content)
        
        print(f"Infrastructure files saved to {output_dir}")


# Example usage
async def test_infrastructure_generation():
    """Test cloud-native infrastructure generation"""
    print("Testing Cloud-Native Infrastructure Generation...")
    
    requirements = InfrastructureRequirements(
        target_language=TargetLanguage.RUST,
        application_type="web",
        cloud_provider=CloudProvider.AWS,
        deployment_strategy="rolling",
        min_replicas=2,
        max_replicas=10,
        database_type="postgres",
        enable_monitoring=True,
        enable_security_scanning=True,
        environment_variables={
            "LOG_LEVEL": "info",
            "DATABASE_URL": "postgres://user:pass@db:5432/appdb"
        },
        secrets=["db_password", "api_key"],
        config_maps={
            "app_config": {
                "feature_flags": "true",
                "cache_ttl": "3600"
            }
        }
    )
    
    generator = CloudNativeInfrastructureGenerator()
    infrastructure = generator.generate_complete_infrastructure(requirements)
    
    print(f"Generated infrastructure for {requirements.target_language.value} application:")
    print(f"- Docker files: {len(infrastructure['docker'])}")
    print(f"- Kubernetes manifests: {len(infrastructure['kubernetes'])}")
    print(f"- Terraform files: {len(infrastructure['terraform'])}")
    
    # Save to output directory
    output_dir = Path("./generated_infrastructure")
    generator.save_infrastructure_files(infrastructure, output_dir)
    
    return infrastructure


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_infrastructure_generation())