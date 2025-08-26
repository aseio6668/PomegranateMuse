"""
CI/CD Integration Module for MyndraComposer
Integrates CI/CD pipeline generation with the main application
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .pipeline_generator import (
    PipelineGenerator, PipelineConfig, PipelineProvider, 
    LanguageType, create_pipeline_config
)


@dataclass
class CICDSettings:
    """CI/CD configuration settings"""
    enabled: bool = True
    default_provider: str = "github_actions"
    auto_generate: bool = True
    include_security_scan: bool = True
    include_performance_tests: bool = False
    deployment_environments: List[str] = None
    notification_channels: List[str] = None
    custom_scripts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.deployment_environments is None:
            self.deployment_environments = ["staging", "production"]
        if self.notification_channels is None:
            self.notification_channels = []
        if self.custom_scripts is None:
            self.custom_scripts = {}


class CICDIntegration:
    """Main CI/CD integration class"""
    
    def __init__(self, project_root: str, settings: Optional[CICDSettings] = None):
        self.project_root = Path(project_root)
        self.settings = settings or CICDSettings()
        self.generator = PipelineGenerator()
        self.config_file = self.project_root / ".pomuse" / "cicd_config.json"
        self._load_config()
    
    def _load_config(self):
        """Load CI/CD configuration from project"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    self.settings = CICDSettings(**config_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load CI/CD config: {e}")
    
    def save_config(self):
        """Save CI/CD configuration to project"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(asdict(self.settings), f, indent=2)
    
    def detect_project_language(self) -> Optional[LanguageType]:
        """Detect the primary language of the project"""
        language_indicators = {
            LanguageType.RUST: ["Cargo.toml", "src/main.rs", "src/lib.rs"],
            LanguageType.GO: ["go.mod", "main.go", "go.sum"],
            LanguageType.TYPESCRIPT: ["package.json", "tsconfig.json", "yarn.lock", "package-lock.json"],
            LanguageType.PYTHON: ["requirements.txt", "setup.py", "pyproject.toml", "__init__.py"],
            LanguageType.MYNDRA: ["myndra.toml", "myndra.config", "*.myn"]
        }
        
        for language, indicators in language_indicators.items():
            for indicator in indicators:
                if indicator.startswith("*"):
                    # Glob pattern
                    pattern = indicator[1:]
                    if list(self.project_root.glob(f"**/*{pattern}")):
                        return language
                else:
                    # Exact file
                    if (self.project_root / indicator).exists():
                        return language
        
        return None
    
    def detect_git_provider(self) -> Optional[PipelineProvider]:
        """Detect the git provider from remote URLs"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "-v"], 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                remotes = result.stdout.lower()
                if "github.com" in remotes:
                    return PipelineProvider.GITHUB_ACTIONS
                elif "gitlab.com" in remotes or "gitlab" in remotes:
                    return PipelineProvider.GITLAB_CI
                elif "bitbucket.org" in remotes:
                    return PipelineProvider.BITBUCKET_PIPELINES
                elif "dev.azure.com" in remotes or "visualstudio.com" in remotes:
                    return PipelineProvider.AZURE_DEVOPS
        except Exception:
            pass
        
        return None
    
    def generate_pipelines(
        self, 
        language: Optional[str] = None, 
        provider: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Generate CI/CD pipelines for the project"""
        
        # Auto-detect language if not provided
        if not language:
            detected_lang = self.detect_project_language()
            if detected_lang:
                language = detected_lang.value
            else:
                raise ValueError("Could not detect project language. Please specify manually.")
        
        # Auto-detect provider if not provided
        if not provider:
            detected_provider = self.detect_git_provider()
            if detected_provider:
                provider = detected_provider.value
            else:
                provider = self.settings.default_provider
        
        # Create pipeline configuration
        config = self._create_pipeline_config(language, provider, custom_config)
        
        # Generate pipelines
        output_dir = str(self.project_root)
        pipelines = self.generator.generate_pipeline(config, output_dir)
        
        # Update project configuration
        self._update_project_config(language, provider, config)
        
        return pipelines
    
    def _create_pipeline_config(
        self, 
        language: str, 
        provider: str, 
        custom_config: Optional[Dict[str, Any]] = None
    ) -> PipelineConfig:
        """Create pipeline configuration based on project settings"""
        
        base_config = {
            "enable_testing": True,
            "enable_linting": True,
            "enable_security_scan": self.settings.include_security_scan,
            "enable_dependency_scan": True,
            "enable_performance_tests": self.settings.include_performance_tests,
            "enable_docker_build": True,
            "enable_kubernetes_deploy": len(self.settings.deployment_environments) > 0,
            "enable_artifact_publishing": True,
            "enable_notifications": len(self.settings.notification_channels) > 0,
            "enable_parallel_jobs": True,
            "branch_protection": True,
            "deployment_environments": self.settings.deployment_environments,
            "environment_secrets": self._get_environment_secrets(language, provider)
        }
        
        # Apply custom configuration overrides
        if custom_config:
            base_config.update(custom_config)
        
        return PipelineConfig(
            language=LanguageType(language),
            provider=PipelineProvider(provider),
            **base_config
        )
    
    def _get_environment_secrets(self, language: str, provider: str) -> List[str]:
        """Get required environment secrets based on language and provider"""
        base_secrets = ["DOCKER_REGISTRY_TOKEN"]
        
        # Language-specific secrets
        language_secrets = {
            "rust": ["CARGO_REGISTRY_TOKEN"],
            "go": ["GOPROXY_TOKEN"],
            "typescript": ["NPM_TOKEN"],
            "python": ["PYPI_TOKEN"],
            "myndra": ["MYNDRA_REGISTRY_TOKEN"]
        }
        
        # Provider-specific secrets
        provider_secrets = {
            "github_actions": ["GITHUB_TOKEN"],
            "gitlab_ci": ["GITLAB_TOKEN"],
            "azure_devops": ["AZURE_DEVOPS_TOKEN"],
            "bitbucket_pipelines": ["BITBUCKET_TOKEN"]
        }
        
        secrets = base_secrets.copy()
        secrets.extend(language_secrets.get(language, []))
        secrets.extend(provider_secrets.get(provider, []))
        
        # Add deployment secrets if environments are configured
        if self.settings.deployment_environments:
            secrets.extend(["KUBECONFIG", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
        
        return secrets
    
    def _update_project_config(self, language: str, provider: str, config: PipelineConfig):
        """Update project configuration with CI/CD settings"""
        project_config_file = self.project_root / ".pomuse" / "project.json"
        
        cicd_info = {
            "language": language,
            "provider": provider,
            "config": asdict(config),
            "generated_files": [],
            "last_updated": None
        }
        
        if project_config_file.exists():
            try:
                with open(project_config_file, 'r') as f:
                    project_config = json.load(f)
            except (json.JSONDecodeError, TypeError):
                project_config = {}
        else:
            project_config = {}
        
        project_config["cicd"] = cicd_info
        
        project_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_file, 'w') as f:
            json.dump(project_config, f, indent=2)
    
    def validate_pipeline_config(self, config_path: str) -> Dict[str, Any]:
        """Validate generated pipeline configuration"""
        validation_results = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            if not os.path.exists(config_path):
                validation_results["errors"].append(f"Pipeline config file not found: {config_path}")
                return validation_results
            
            # Basic file validation
            with open(config_path, 'r') as f:
                if config_path.endswith('.yml') or config_path.endswith('.yaml'):
                    import yaml
                    pipeline_config = yaml.safe_load(f)
                else:
                    pipeline_config = json.load(f)
            
            # Provider-specific validation
            if "github" in config_path:
                validation_results.update(self._validate_github_actions(pipeline_config))
            elif "gitlab" in config_path:
                validation_results.update(self._validate_gitlab_ci(pipeline_config))
            elif "azure" in config_path:
                validation_results.update(self._validate_azure_pipelines(pipeline_config))
            
            # General validation
            if not validation_results["errors"]:
                validation_results["valid"] = True
                validation_results["suggestions"].append("Pipeline configuration looks good!")
        
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_github_actions(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate GitHub Actions workflow"""
        results = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check required fields
        if "on" not in config:
            results["errors"].append("Missing 'on' trigger configuration")
        
        if "jobs" not in config:
            results["errors"].append("Missing 'jobs' configuration")
        elif not config["jobs"]:
            results["errors"].append("No jobs defined")
        
        # Check for security best practices
        for job_name, job in config.get("jobs", {}).items():
            if "permissions" not in job:
                results["warnings"].append(f"Job '{job_name}' missing explicit permissions")
            
            steps = job.get("steps", [])
            for i, step in enumerate(steps):
                if step.get("uses", "").startswith("actions/checkout@v") and not step.get("uses", "").endswith("@v4"):
                    results["suggestions"].append(f"Consider updating checkout action to v4 in job '{job_name}', step {i+1}")
        
        return results
    
    def _validate_gitlab_ci(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate GitLab CI configuration"""
        results = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check for stages
        if "stages" not in config:
            results["warnings"].append("No explicit stages defined")
        
        # Check for image or services
        has_global_image = "image" in config
        has_job_images = any("image" in job for job in config.values() if isinstance(job, dict))
        
        if not has_global_image and not has_job_images:
            results["warnings"].append("No Docker image specified globally or in jobs")
        
        return results
    
    def _validate_azure_pipelines(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate Azure Pipelines configuration"""
        results = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check required fields
        if "stages" not in config and "jobs" not in config and "steps" not in config:
            results["errors"].append("Missing pipeline structure (stages, jobs, or steps)")
        
        return results
    
    def update_pipeline(self, provider: Optional[str] = None) -> Dict[str, str]:
        """Update existing pipeline configuration"""
        project_config_file = self.project_root / ".pomuse" / "project.json"
        
        if not project_config_file.exists():
            raise ValueError("No existing CI/CD configuration found. Generate pipelines first.")
        
        with open(project_config_file, 'r') as f:
            project_config = json.load(f)
        
        cicd_config = project_config.get("cicd", {})
        language = cicd_config.get("language")
        current_provider = cicd_config.get("provider")
        
        if not language:
            raise ValueError("No language configuration found in project")
        
        # Use current provider if none specified
        if not provider:
            provider = current_provider or self.settings.default_provider
        
        return self.generate_pipelines(language, provider)
    
    def list_supported_providers(self) -> List[str]:
        """List all supported CI/CD providers"""
        return [provider.value for provider in PipelineProvider]
    
    def list_supported_languages(self) -> List[str]:
        """List all supported languages"""
        return [language.value for language in LanguageType]
    
    def get_provider_features(self, provider: str) -> Dict[str, Any]:
        """Get features supported by a specific provider"""
        features = {
            "github_actions": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": True,
                "code_scanning": True,
                "dependency_review": True,
                "container_registry": True
            },
            "gitlab_ci": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": True,
                "code_scanning": True,
                "dependency_scanning": True,
                "container_registry": True
            },
            "azure_devops": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": True,
                "code_scanning": False,
                "dependency_scanning": True,
                "container_registry": True
            },
            "jenkins": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": False,
                "code_scanning": False,
                "dependency_scanning": False,
                "container_registry": False
            },
            "bitbucket_pipelines": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": True,
                "code_scanning": False,
                "dependency_scanning": False,
                "container_registry": False
            },
            "circle_ci": {
                "matrix_builds": True,
                "secret_management": True,
                "artifact_storage": True,
                "environment_protection": False,
                "code_scanning": False,
                "dependency_scanning": False,
                "container_registry": False
            }
        }
        
        return features.get(provider, {})


def setup_cicd_for_project(
    project_root: str,
    language: Optional[str] = None,
    provider: Optional[str] = None,
    environments: Optional[List[str]] = None,
    enable_security: bool = True
) -> Dict[str, str]:
    """Convenience function to set up CI/CD for a project"""
    
    settings = CICDSettings(
        deployment_environments=environments or ["staging", "production"],
        include_security_scan=enable_security
    )
    
    integration = CICDIntegration(project_root, settings)
    integration.save_config()
    
    return integration.generate_pipelines(language, provider)


# CLI interface functions
def configure_cicd_interactive(project_root: str) -> Dict[str, str]:
    """Interactive CI/CD configuration"""
    print("🚀 Setting up CI/CD for your project...")
    
    integration = CICDIntegration(project_root)
    
    # Detect current setup
    detected_language = integration.detect_project_language()
    detected_provider = integration.detect_git_provider()
    
    print(f"Detected language: {detected_language.value if detected_language else 'Unknown'}")
    print(f"Detected provider: {detected_provider.value if detected_provider else 'Unknown'}")
    
    # Get user input
    language = input(f"Language [{detected_language.value if detected_language else 'rust'}]: ").strip()
    if not language:
        language = detected_language.value if detected_language else "rust"
    
    provider = input(f"CI/CD Provider [{detected_provider.value if detected_provider else 'github_actions'}]: ").strip()
    if not provider:
        provider = detected_provider.value if detected_provider else "github_actions"
    
    environments_input = input("Deployment environments [staging,production]: ").strip()
    environments = [env.strip() for env in environments_input.split(",")] if environments_input else ["staging", "production"]
    
    security_scan = input("Enable security scanning? [Y/n]: ").strip().lower()
    enable_security = security_scan != "n"
    
    # Generate pipelines
    try:
        pipelines = setup_cicd_for_project(
            project_root=project_root,
            language=language,
            provider=provider,
            environments=environments,
            enable_security=enable_security
        )
        
        print("\n✅ CI/CD pipelines generated successfully!")
        for name, path in pipelines.items():
            print(f"  📄 {name}: {path}")
        
        return pipelines
    
    except Exception as e:
        print(f"\n❌ Error generating pipelines: {e}")
        return {}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "."
    
    configure_cicd_interactive(project_path)