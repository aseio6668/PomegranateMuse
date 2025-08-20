"""
CI/CD Pipeline Generator for PomegranteMuse
Generates comprehensive CI/CD pipelines for multiple platforms and languages
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class PipelineProvider(Enum):
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    AZURE_DEVOPS = "azure_devops"
    JENKINS = "jenkins"
    BITBUCKET_PIPELINES = "bitbucket_pipelines"
    CIRCLE_CI = "circle_ci"

class LanguageType(Enum):
    RUST = "rust"
    GO = "go"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    POMEGRANATE = "pomegranate"

@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline generation"""
    language: LanguageType
    provider: PipelineProvider
    enable_testing: bool = True
    enable_linting: bool = True
    enable_security_scan: bool = True
    enable_dependency_scan: bool = True
    enable_performance_tests: bool = False
    enable_docker_build: bool = True
    enable_kubernetes_deploy: bool = False
    enable_artifact_publishing: bool = True
    enable_notifications: bool = True
    enable_parallel_jobs: bool = True
    branch_protection: bool = True
    environment_secrets: List[str] = None
    deployment_environments: List[str] = None

class PipelineGenerator:
    """Generates CI/CD pipelines for different providers and languages"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize pipeline templates for different providers"""
        self.templates = {
            PipelineProvider.GITHUB_ACTIONS: self._github_actions_templates(),
            PipelineProvider.GITLAB_CI: self._gitlab_ci_templates(),
            PipelineProvider.AZURE_DEVOPS: self._azure_devops_templates(),
            PipelineProvider.JENKINS: self._jenkins_templates(),
            PipelineProvider.BITBUCKET_PIPELINES: self._bitbucket_templates(),
            PipelineProvider.CIRCLE_CI: self._circle_ci_templates()
        }
    
    def generate_pipeline(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate CI/CD pipeline based on configuration"""
        provider_generator = getattr(self, f"_generate_{config.provider.value}")
        return provider_generator(config, output_dir)
    
    def _generate_github_actions(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate GitHub Actions workflow"""
        workflows_dir = os.path.join(output_dir, ".github", "workflows")
        os.makedirs(workflows_dir, exist_ok=True)
        
        workflows = {}
        
        # Main CI workflow
        ci_workflow = self._create_github_ci_workflow(config)
        ci_path = os.path.join(workflows_dir, "ci.yml")
        with open(ci_path, 'w') as f:
            yaml.dump(ci_workflow, f, default_flow_style=False, sort_keys=False)
        workflows["ci.yml"] = ci_path
        
        # CD workflow
        if config.enable_kubernetes_deploy or config.enable_artifact_publishing:
            cd_workflow = self._create_github_cd_workflow(config)
            cd_path = os.path.join(workflows_dir, "cd.yml")
            with open(cd_path, 'w') as f:
                yaml.dump(cd_workflow, f, default_flow_style=False, sort_keys=False)
            workflows["cd.yml"] = cd_path
        
        # Security scan workflow
        if config.enable_security_scan:
            security_workflow = self._create_github_security_workflow(config)
            security_path = os.path.join(workflows_dir, "security.yml")
            with open(security_path, 'w') as f:
                yaml.dump(security_workflow, f, default_flow_style=False, sort_keys=False)
            workflows["security.yml"] = security_path
        
        return workflows
    
    def _create_github_ci_workflow(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create GitHub Actions CI workflow"""
        workflow = {
            "name": "Continuous Integration",
            "on": {
                "push": {
                    "branches": ["main", "develop", "feature/*"]
                },
                "pull_request": {
                    "branches": ["main", "develop"]
                }
            },
            "env": self._get_environment_variables(config),
            "jobs": {}
        }
        
        # Build and test job
        build_job = {
            "runs-on": "ubuntu-latest",
            "strategy": {
                "matrix": self._get_build_matrix(config.language)
            } if config.enable_parallel_jobs else None,
            "steps": [
                {
                    "name": "Checkout code",
                    "uses": "actions/checkout@v4"
                },
                self._get_language_setup_step(config.language),
                {
                    "name": "Cache dependencies",
                    "uses": "actions/cache@v3",
                    "with": self._get_cache_config(config.language)
                },
                {
                    "name": "Install dependencies",
                    "run": self._get_install_command(config.language)
                }
            ]
        }
        
        # Add linting step
        if config.enable_linting:
            build_job["steps"].append({
                "name": "Run linting",
                "run": self._get_lint_command(config.language)
            })
        
        # Add testing step
        if config.enable_testing:
            build_job["steps"].extend([
                {
                    "name": "Run tests",
                    "run": self._get_test_command(config.language)
                },
                {
                    "name": "Upload coverage reports",
                    "uses": "codecov/codecov-action@v3",
                    "with": {
                        "file": self._get_coverage_file(config.language)
                    }
                }
            ])
        
        # Add Docker build step
        if config.enable_docker_build:
            build_job["steps"].extend([
                {
                    "name": "Set up Docker Buildx",
                    "uses": "docker/setup-buildx-action@v3"
                },
                {
                    "name": "Build Docker image",
                    "uses": "docker/build-push-action@v5",
                    "with": {
                        "context": ".",
                        "push": False,
                        "tags": "app:${{ github.sha }}"
                    }
                }
            ])
        
        workflow["jobs"]["build"] = build_job
        
        # Add security scan job
        if config.enable_security_scan:
            workflow["jobs"]["security"] = {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "Checkout code",
                        "uses": "actions/checkout@v4"
                    },
                    {
                        "name": "Run Trivy vulnerability scanner",
                        "uses": "aquasecurity/trivy-action@master",
                        "with": {
                            "scan-type": "fs",
                            "scan-ref": "."
                        }
                    }
                ]
            }
        
        # Add dependency scan job
        if config.enable_dependency_scan:
            workflow["jobs"]["dependencies"] = {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "Checkout code",
                        "uses": "actions/checkout@v4"
                    },
                    {
                        "name": "Dependency Review",
                        "uses": "actions/dependency-review-action@v3"
                    }
                ]
            }
        
        return workflow
    
    def _create_github_cd_workflow(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create GitHub Actions CD workflow"""
        workflow = {
            "name": "Continuous Deployment",
            "on": {
                "push": {
                    "branches": ["main"]
                },
                "release": {
                    "types": ["published"]
                }
            },
            "env": self._get_environment_variables(config),
            "jobs": {}
        }
        
        # Build and publish artifacts
        if config.enable_artifact_publishing:
            workflow["jobs"]["publish"] = {
                "runs-on": "ubuntu-latest",
                "needs": "build",
                "steps": [
                    {
                        "name": "Checkout code",
                        "uses": "actions/checkout@v4"
                    },
                    self._get_language_setup_step(config.language),
                    {
                        "name": "Build release",
                        "run": self._get_build_command(config.language)
                    },
                    {
                        "name": "Publish artifacts",
                        "run": self._get_publish_command(config.language)
                    }
                ]
            }
        
        # Deploy to environments
        if config.deployment_environments:
            for env in config.deployment_environments:
                workflow["jobs"][f"deploy-{env}"] = {
                    "runs-on": "ubuntu-latest",
                    "needs": "publish" if config.enable_artifact_publishing else "build",
                    "environment": env,
                    "steps": [
                        {
                            "name": "Deploy to " + env,
                            "run": self._get_deploy_command(config.language, env)
                        }
                    ]
                }
        
        return workflow
    
    def _create_github_security_workflow(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create GitHub Actions security workflow"""
        return {
            "name": "Security Analysis",
            "on": {
                "schedule": [{"cron": "0 0 * * 0"}],  # Weekly
                "workflow_dispatch": {}
            },
            "jobs": {
                "codeql": {
                    "runs-on": "ubuntu-latest",
                    "permissions": {
                        "actions": "read",
                        "contents": "read",
                        "security-events": "write"
                    },
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Initialize CodeQL",
                            "uses": "github/codeql-action/init@v2",
                            "with": {
                                "languages": self._get_codeql_language(config.language)
                            }
                        },
                        {
                            "name": "Autobuild",
                            "uses": "github/codeql-action/autobuild@v2"
                        },
                        {
                            "name": "Perform CodeQL Analysis",
                            "uses": "github/codeql-action/analyze@v2"
                        }
                    ]
                }
            }
        }
    
    def _generate_gitlab_ci(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate GitLab CI pipeline"""
        gitlab_ci = self._create_gitlab_ci_config(config)
        ci_path = os.path.join(output_dir, ".gitlab-ci.yml")
        
        with open(ci_path, 'w') as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False, sort_keys=False)
        
        return {".gitlab-ci.yml": ci_path}
    
    def _create_gitlab_ci_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create GitLab CI configuration"""
        gitlab_ci = {
            "stages": ["build", "test", "security", "deploy"],
            "variables": self._get_environment_variables(config),
            "image": self._get_docker_image(config.language)
        }
        
        # Cache configuration
        gitlab_ci["cache"] = {
            "paths": self._get_cache_paths(config.language)
        }
        
        # Build job
        gitlab_ci["build"] = {
            "stage": "build",
            "script": [
                self._get_install_command(config.language),
                self._get_build_command(config.language)
            ],
            "artifacts": {
                "paths": self._get_artifact_paths(config.language),
                "expire_in": "1 week"
            }
        }
        
        # Test job
        if config.enable_testing:
            gitlab_ci["test"] = {
                "stage": "test",
                "script": [
                    self._get_test_command(config.language)
                ],
                "artifacts": {
                    "reports": {
                        "coverage_report": {
                            "coverage_format": "cobertura",
                            "path": self._get_coverage_file(config.language)
                        }
                    }
                }
            }
        
        # Security scan job
        if config.enable_security_scan:
            gitlab_ci["security"] = {
                "stage": "security",
                "image": "registry.gitlab.com/security-products/trivy:latest",
                "script": [
                    "trivy fs --format template --template '@/contrib/gitlab.tpl' -o gl-sast-report.json ."
                ],
                "artifacts": {
                    "reports": {
                        "sast": "gl-sast-report.json"
                    }
                }
            }
        
        # Deploy job
        if config.deployment_environments:
            for env in config.deployment_environments:
                gitlab_ci[f"deploy-{env}"] = {
                    "stage": "deploy",
                    "script": [
                        self._get_deploy_command(config.language, env)
                    ],
                    "environment": {
                        "name": env,
                        "url": f"https://{env}.example.com"
                    },
                    "when": "manual" if env != "staging" else "on_success"
                }
        
        return gitlab_ci
    
    def _generate_azure_devops(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate Azure DevOps pipeline"""
        azure_pipeline = self._create_azure_pipeline_config(config)
        pipeline_path = os.path.join(output_dir, "azure-pipelines.yml")
        
        with open(pipeline_path, 'w') as f:
            yaml.dump(azure_pipeline, f, default_flow_style=False, sort_keys=False)
        
        return {"azure-pipelines.yml": pipeline_path}
    
    def _create_azure_pipeline_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create Azure DevOps pipeline configuration"""
        return {
            "trigger": ["main", "develop"],
            "pr": ["main", "develop"],
            "variables": self._get_environment_variables(config),
            "stages": [
                {
                    "stage": "Build",
                    "jobs": [
                        {
                            "job": "BuildAndTest",
                            "pool": {
                                "vmImage": "ubuntu-latest"
                            },
                            "steps": [
                                self._get_azure_language_setup(config.language),
                                {
                                    "script": self._get_install_command(config.language),
                                    "displayName": "Install dependencies"
                                },
                                {
                                    "script": self._get_build_command(config.language),
                                    "displayName": "Build application"
                                },
                                {
                                    "script": self._get_test_command(config.language),
                                    "displayName": "Run tests"
                                } if config.enable_testing else None
                            ]
                        }
                    ]
                }
            ]
        }
    
    def _generate_jenkins(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate Jenkins pipeline"""
        jenkinsfile = self._create_jenkinsfile(config)
        jenkins_path = os.path.join(output_dir, "Jenkinsfile")
        
        with open(jenkins_path, 'w') as f:
            f.write(jenkinsfile)
        
        return {"Jenkinsfile": jenkins_path}
    
    def _create_jenkinsfile(self, config: PipelineConfig) -> str:
        """Create Jenkinsfile content"""
        stages = []
        
        # Build stage
        stages.append(f"""
        stage('Build') {{
            steps {{
                {self._get_jenkins_language_setup(config.language)}
                sh '{self._get_install_command(config.language)}'
                sh '{self._get_build_command(config.language)}'
            }}
        }}""")
        
        # Test stage
        if config.enable_testing:
            stages.append(f"""
        stage('Test') {{
            steps {{
                sh '{self._get_test_command(config.language)}'
            }}
            post {{
                always {{
                    publishTestResults(testResultsPattern: '{self._get_test_results_pattern(config.language)}')
                }}
            }}
        }}""")
        
        # Security stage
        if config.enable_security_scan:
            stages.append("""
        stage('Security Scan') {
            steps {
                sh 'trivy fs --format table .'
            }
        }""")
        
        # Deploy stage
        if config.deployment_environments:
            deploy_steps = []
            for env in config.deployment_environments:
                deploy_steps.append(f"sh '{self._get_deploy_command(config.language, env)}'")
            
            stages.append(f"""
        stage('Deploy') {{
            steps {{
                {chr(10).join("                " + step for step in deploy_steps)}
            }}
        }}""")
        
        return f"""
pipeline {{
    agent any
    
    environment {{
        {self._get_jenkins_environment_variables(config)}
    }}
    
    stages {{{chr(10).join(stages)}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        success {{
            {self._get_jenkins_success_notification(config) if config.enable_notifications else "echo 'Build successful'"}
        }}
        failure {{
            {self._get_jenkins_failure_notification(config) if config.enable_notifications else "echo 'Build failed'"}
        }}
    }}
}}"""
    
    def _generate_bitbucket_pipelines(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate Bitbucket Pipelines"""
        bitbucket_config = self._create_bitbucket_config(config)
        config_path = os.path.join(output_dir, "bitbucket-pipelines.yml")
        
        with open(config_path, 'w') as f:
            yaml.dump(bitbucket_config, f, default_flow_style=False, sort_keys=False)
        
        return {"bitbucket-pipelines.yml": config_path}
    
    def _create_bitbucket_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create Bitbucket Pipelines configuration"""
        pipelines = {
            "default": [
                {
                    "step": {
                        "name": "Build and Test",
                        "image": self._get_docker_image(config.language),
                        "script": [
                            self._get_install_command(config.language),
                            self._get_build_command(config.language),
                            self._get_test_command(config.language) if config.enable_testing else "echo 'No tests configured'"
                        ]
                    }
                }
            ]
        }
        
        if config.enable_security_scan:
            pipelines["default"].append({
                "step": {
                    "name": "Security Scan",
                    "image": "aquasec/trivy:latest",
                    "script": [
                        "trivy fs --format table ."
                    ]
                }
            })
        
        return {
            "image": self._get_docker_image(config.language),
            "pipelines": pipelines
        }
    
    def _generate_circle_ci(self, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
        """Generate CircleCI configuration"""
        circle_config = self._create_circle_config(config)
        config_dir = os.path.join(output_dir, ".circleci")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.yml")
        
        with open(config_path, 'w') as f:
            yaml.dump(circle_config, f, default_flow_style=False, sort_keys=False)
        
        return {"config.yml": config_path}
    
    def _create_circle_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Create CircleCI configuration"""
        return {
            "version": 2.1,
            "jobs": {
                "build": {
                    "docker": [{"image": self._get_docker_image(config.language)}],
                    "steps": [
                        "checkout",
                        {
                            "run": {
                                "name": "Install dependencies",
                                "command": self._get_install_command(config.language)
                            }
                        },
                        {
                            "run": {
                                "name": "Build application",
                                "command": self._get_build_command(config.language)
                            }
                        },
                        {
                            "run": {
                                "name": "Run tests",
                                "command": self._get_test_command(config.language)
                            }
                        } if config.enable_testing else None
                    ]
                }
            },
            "workflows": {
                "version": 2,
                "build_and_test": {
                    "jobs": ["build"]
                }
            }
        }
    
    # Helper methods for language-specific configurations
    def _get_language_setup_step(self, language: LanguageType) -> Dict[str, Any]:
        """Get language setup step for GitHub Actions"""
        setups = {
            LanguageType.RUST: {
                "name": "Setup Rust",
                "uses": "actions-rs/toolchain@v1",
                "with": {
                    "toolchain": "stable",
                    "override": True
                }
            },
            LanguageType.GO: {
                "name": "Setup Go",
                "uses": "actions/setup-go@v4",
                "with": {
                    "go-version": "1.21"
                }
            },
            LanguageType.TYPESCRIPT: {
                "name": "Setup Node.js",
                "uses": "actions/setup-node@v3",
                "with": {
                    "node-version": "18",
                    "cache": "npm"
                }
            },
            LanguageType.PYTHON: {
                "name": "Setup Python",
                "uses": "actions/setup-python@v4",
                "with": {
                    "python-version": "3.11"
                }
            }
        }
        return setups.get(language, {"name": "Setup", "run": "echo 'No setup required'"})
    
    def _get_install_command(self, language: LanguageType) -> str:
        """Get dependency installation command"""
        commands = {
            LanguageType.RUST: "cargo fetch",
            LanguageType.GO: "go mod download",
            LanguageType.TYPESCRIPT: "npm ci",
            LanguageType.PYTHON: "pip install -r requirements.txt",
            LanguageType.POMEGRANATE: "pomegranate install"
        }
        return commands.get(language, "echo 'No install command'")
    
    def _get_build_command(self, language: LanguageType) -> str:
        """Get build command"""
        commands = {
            LanguageType.RUST: "cargo build --release",
            LanguageType.GO: "go build -o bin/app",
            LanguageType.TYPESCRIPT: "npm run build",
            LanguageType.PYTHON: "python -m build",
            LanguageType.POMEGRANATE: "pomegranate build --release"
        }
        return commands.get(language, "echo 'No build command'")
    
    def _get_test_command(self, language: LanguageType) -> str:
        """Get test command"""
        commands = {
            LanguageType.RUST: "cargo test",
            LanguageType.GO: "go test ./...",
            LanguageType.TYPESCRIPT: "npm test",
            LanguageType.PYTHON: "pytest",
            LanguageType.POMEGRANATE: "pomegranate test"
        }
        return commands.get(language, "echo 'No test command'")
    
    def _get_lint_command(self, language: LanguageType) -> str:
        """Get linting command"""
        commands = {
            LanguageType.RUST: "cargo clippy -- -D warnings",
            LanguageType.GO: "golangci-lint run",
            LanguageType.TYPESCRIPT: "npm run lint",
            LanguageType.PYTHON: "flake8 .",
            LanguageType.POMEGRANATE: "pomegranate lint"
        }
        return commands.get(language, "echo 'No lint command'")
    
    def _get_docker_image(self, language: LanguageType) -> str:
        """Get appropriate Docker image for language"""
        images = {
            LanguageType.RUST: "rust:1.70",
            LanguageType.GO: "golang:1.21",
            LanguageType.TYPESCRIPT: "node:18",
            LanguageType.PYTHON: "python:3.11",
            LanguageType.POMEGRANATE: "ubuntu:22.04"
        }
        return images.get(language, "ubuntu:22.04")
    
    def _get_environment_variables(self, config: PipelineConfig) -> Dict[str, str]:
        """Get environment variables for pipeline"""
        env_vars = {
            "CI": "true",
            "ENVIRONMENT": "ci"
        }
        
        if config.environment_secrets:
            for secret in config.environment_secrets:
                env_vars[secret] = f"${{{{ secrets.{secret} }}}}"
        
        return env_vars
    
    def _get_cache_config(self, language: LanguageType) -> Dict[str, Any]:
        """Get cache configuration"""
        configs = {
            LanguageType.RUST: {
                "path": "~/.cargo/registry\n~/.cargo/git\ntarget/",
                "key": "${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}"
            },
            LanguageType.GO: {
                "path": "~/go/pkg/mod",
                "key": "${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}"
            },
            LanguageType.TYPESCRIPT: {
                "path": "~/.npm",
                "key": "${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}"
            },
            LanguageType.PYTHON: {
                "path": "~/.cache/pip",
                "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}"
            }
        }
        return configs.get(language, {"path": ".", "key": "${{ runner.os }}-cache"})
    
    def _get_build_matrix(self, language: LanguageType) -> Dict[str, List[str]]:
        """Get build matrix for parallel execution"""
        matrices = {
            LanguageType.RUST: {
                "rust": ["stable", "beta"],
                "os": ["ubuntu-latest", "windows-latest", "macos-latest"]
            },
            LanguageType.GO: {
                "go-version": ["1.20", "1.21"],
                "os": ["ubuntu-latest", "windows-latest", "macos-latest"]
            },
            LanguageType.TYPESCRIPT: {
                "node-version": ["16", "18", "20"],
                "os": ["ubuntu-latest", "windows-latest", "macos-latest"]
            },
            LanguageType.PYTHON: {
                "python-version": ["3.9", "3.10", "3.11"],
                "os": ["ubuntu-latest", "windows-latest", "macos-latest"]
            }
        }
        return matrices.get(language, {"os": ["ubuntu-latest"]})
    
    def _github_actions_templates(self) -> Dict[str, Any]:
        """GitHub Actions pipeline templates"""
        return {}
    
    def _gitlab_ci_templates(self) -> Dict[str, Any]:
        """GitLab CI pipeline templates"""
        return {}
    
    def _azure_devops_templates(self) -> Dict[str, Any]:
        """Azure DevOps pipeline templates"""
        return {}
    
    def _jenkins_templates(self) -> Dict[str, Any]:
        """Jenkins pipeline templates"""
        return {}
    
    def _bitbucket_templates(self) -> Dict[str, Any]:
        """Bitbucket pipelines templates"""
        return {}
    
    def _circle_ci_templates(self) -> Dict[str, Any]:
        """CircleCI pipeline templates"""
        return {}
    
    # Additional helper methods
    def _get_coverage_file(self, language: LanguageType) -> str:
        """Get coverage file path"""
        files = {
            LanguageType.RUST: "target/tarpaulin/cobertura.xml",
            LanguageType.GO: "coverage.out",
            LanguageType.TYPESCRIPT: "coverage/lcov.info",
            LanguageType.PYTHON: "coverage.xml"
        }
        return files.get(language, "coverage.xml")
    
    def _get_codeql_language(self, language: LanguageType) -> str:
        """Get CodeQL language identifier"""
        mappings = {
            LanguageType.RUST: "rust",
            LanguageType.GO: "go",
            LanguageType.TYPESCRIPT: "javascript",
            LanguageType.PYTHON: "python"
        }
        return mappings.get(language, "generic")
    
    def _get_publish_command(self, language: LanguageType) -> str:
        """Get artifact publishing command"""
        commands = {
            LanguageType.RUST: "cargo publish",
            LanguageType.GO: "goreleaser release",
            LanguageType.TYPESCRIPT: "npm publish",
            LanguageType.PYTHON: "twine upload dist/*"
        }
        return commands.get(language, "echo 'No publish command'")
    
    def _get_deploy_command(self, language: LanguageType, environment: str) -> str:
        """Get deployment command"""
        return f"kubectl apply -f k8s/{environment}/ && kubectl rollout status deployment/app -n {environment}"
    
    def _get_azure_language_setup(self, language: LanguageType) -> Dict[str, Any]:
        """Get Azure DevOps language setup"""
        setups = {
            LanguageType.RUST: {
                "task": "RustInstaller@1",
                "inputs": {
                    "rustVersion": "stable"
                }
            },
            LanguageType.GO: {
                "task": "GoTool@0",
                "inputs": {
                    "version": "1.21"
                }
            },
            LanguageType.TYPESCRIPT: {
                "task": "NodeTool@0",
                "inputs": {
                    "versionSpec": "18.x"
                }
            }
        }
        return setups.get(language, {"script": "echo 'No setup required'"})
    
    def _get_jenkins_language_setup(self, language: LanguageType) -> str:
        """Get Jenkins language setup"""
        setups = {
            LanguageType.RUST: "tool name: 'Rust', type: 'rust'",
            LanguageType.GO: "tool name: 'Go', type: 'go'",
            LanguageType.TYPESCRIPT: "tool name: 'NodeJS', type: 'nodejs'",
            LanguageType.PYTHON: "tool name: 'Python', type: 'python'"
        }
        return setups.get(language, "echo 'No setup required'")
    
    def _get_jenkins_environment_variables(self, config: PipelineConfig) -> str:
        """Get Jenkins environment variables"""
        vars_list = []
        for key, value in self._get_environment_variables(config).items():
            vars_list.append(f"{key} = '{value}'")
        return "\n        ".join(vars_list)
    
    def _get_jenkins_success_notification(self, config: PipelineConfig) -> str:
        """Get Jenkins success notification"""
        return "slackSend channel: '#ci-cd', color: 'good', message: 'Build successful!'"
    
    def _get_jenkins_failure_notification(self, config: PipelineConfig) -> str:
        """Get Jenkins failure notification"""
        return "slackSend channel: '#ci-cd', color: 'danger', message: 'Build failed!'"
    
    def _get_cache_paths(self, language: LanguageType) -> List[str]:
        """Get cache paths for GitLab CI"""
        paths = {
            LanguageType.RUST: ["target/", "~/.cargo/"],
            LanguageType.GO: ["vendor/"],
            LanguageType.TYPESCRIPT: ["node_modules/"],
            LanguageType.PYTHON: [".pip-cache/"]
        }
        return paths.get(language, [])
    
    def _get_artifact_paths(self, language: LanguageType) -> List[str]:
        """Get artifact paths"""
        paths = {
            LanguageType.RUST: ["target/release/"],
            LanguageType.GO: ["bin/"],
            LanguageType.TYPESCRIPT: ["dist/"],
            LanguageType.PYTHON: ["dist/"]
        }
        return paths.get(language, ["build/"])
    
    def _get_test_results_pattern(self, language: LanguageType) -> str:
        """Get test results pattern for Jenkins"""
        patterns = {
            LanguageType.RUST: "target/test-results.xml",
            LanguageType.GO: "test-results.xml",
            LanguageType.TYPESCRIPT: "test-results.xml",
            LanguageType.PYTHON: "test-results.xml"
        }
        return patterns.get(language, "test-results.xml")

def create_pipeline_config(
    language: str,
    provider: str,
    enable_docker: bool = True,
    enable_k8s: bool = False,
    environments: List[str] = None
) -> PipelineConfig:
    """Create a pipeline configuration with common defaults"""
    return PipelineConfig(
        language=LanguageType(language.lower()),
        provider=PipelineProvider(provider.lower()),
        enable_docker_build=enable_docker,
        enable_kubernetes_deploy=enable_k8s,
        deployment_environments=environments or [],
        environment_secrets=["DOCKER_REGISTRY_TOKEN", "KUBECONFIG"]
    )

# Example usage
if __name__ == "__main__":
    # Create pipeline generator
    generator = PipelineGenerator()
    
    # Create configuration
    config = create_pipeline_config(
        language="rust",
        provider="github_actions",
        enable_docker=True,
        enable_k8s=True,
        environments=["staging", "production"]
    )
    
    # Generate pipeline
    output_dir = "./generated-pipeline"
    os.makedirs(output_dir, exist_ok=True)
    
    pipelines = generator.generate_pipeline(config, output_dir)
    print(f"Generated pipelines: {list(pipelines.keys())}")