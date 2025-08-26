"""
Multi-target language support for Universal Code Modernization Platform
Extensible architecture for generating code in multiple target languages
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json


class TargetLanguage(Enum):
    """Supported target languages for code generation"""
    MYNDRA = "myndra"
    RUST = "rust"
    GO = "go"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    MODERN_CPP = "modern_cpp"


class ArchitecturePattern(Enum):
    """Common architecture patterns"""
    MONOLITH = "monolith"
    MICROSERVICE = "microservice"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    REACTIVE = "reactive"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"


@dataclass
class LanguageFeatures:
    """Features and capabilities of a target language"""
    name: str
    version: str
    
    # Core language features
    has_generics: bool = False
    has_async_await: bool = False
    has_pattern_matching: bool = False
    has_null_safety: bool = False
    has_memory_safety: bool = False
    has_type_inference: bool = False
    has_macros: bool = False
    has_traits_interfaces: bool = False
    
    # Concurrency features
    concurrency_model: str = "threads"  # threads, async, actors, goroutines, etc.
    has_channels: bool = False
    has_green_threads: bool = False
    
    # Error handling
    error_handling: str = "exceptions"  # exceptions, result_types, optional, etc.
    
    # Package management
    package_manager: str = "none"
    package_file: str = ""
    
    # Build system
    build_system: str = "make"
    build_file: str = "Makefile"
    
    # Common frameworks/libraries
    web_frameworks: List[str] = None
    testing_frameworks: List[str] = None
    orm_libraries: List[str] = None
    
    def __post_init__(self):
        if self.web_frameworks is None:
            self.web_frameworks = []
        if self.testing_frameworks is None:
            self.testing_frameworks = []
        if self.orm_libraries is None:
            self.orm_libraries = []


@dataclass
class MigrationStrategy:
    """Strategy for migrating code to target language"""
    name: str
    description: str
    approach: str  # "rewrite", "transpile", "incremental", "bridge"
    complexity: str  # "low", "medium", "high"
    timeline_estimate: str
    risks: List[str]
    benefits: List[str]
    prerequisites: List[str]
    
    # Technical details
    requires_manual_review: bool = True
    supports_gradual_migration: bool = False
    maintains_performance: bool = True
    preserves_architecture: bool = True


@dataclass
class CodeGenerationContext:
    """Context for code generation"""
    source_language: str
    target_language: TargetLanguage
    architecture_pattern: ArchitecturePattern
    domain: str  # "web", "cli", "library", "service", etc.
    
    # Project characteristics
    project_size: str  # "small", "medium", "large", "enterprise"
    performance_requirements: str  # "low", "medium", "high", "realtime"
    security_requirements: str  # "basic", "enterprise", "defense"
    
    # Migration preferences
    migration_strategy: str = "incremental"
    preserve_existing_apis: bool = True
    modernize_patterns: bool = True
    add_type_safety: bool = True
    improve_error_handling: bool = True
    
    # Additional context
    team_experience: Dict[str, str] = None  # language -> experience_level
    deployment_target: str = "cloud"  # "cloud", "on_premise", "edge", "mobile"
    
    def __post_init__(self):
        if self.team_experience is None:
            self.team_experience = {}


class LanguageGenerator(ABC):
    """Abstract base class for language-specific code generators"""
    
    def __init__(self, features: LanguageFeatures):
        self.features = features
    
    @abstractmethod
    def generate_project_structure(self, context: CodeGenerationContext) -> Dict[str, str]:
        """Generate project structure and configuration files"""
        pass
    
    @abstractmethod
    def generate_function(self, function_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a single function in the target language"""
        pass
    
    @abstractmethod
    def generate_class(self, class_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a class/struct/type in the target language"""
        pass
    
    @abstractmethod
    def generate_module(self, module_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a module/package/namespace in the target language"""
        pass
    
    @abstractmethod
    def generate_error_handling(self, error_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate error handling code"""
        pass
    
    @abstractmethod
    def generate_async_code(self, async_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate asynchronous code"""
        pass
    
    @abstractmethod
    def generate_tests(self, test_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate test code"""
        pass
    
    def get_migration_strategies(self, source_analysis: Dict[str, Any]) -> List[MigrationStrategy]:
        """Get recommended migration strategies for this target language"""
        return []
    
    def estimate_migration_effort(self, source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate migration effort and complexity"""
        return {
            "complexity": "medium",
            "estimated_days": 30,
            "risk_level": "medium",
            "confidence": 0.7
        }
    
    def get_best_practices(self, context: CodeGenerationContext) -> List[str]:
        """Get best practices for the target language in given context"""
        return []
    
    def validate_generated_code(self, code: str, context: CodeGenerationContext) -> Dict[str, Any]:
        """Validate generated code for common issues"""
        return {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }


class LanguageRegistry:
    """Registry for all supported target languages"""
    
    def __init__(self):
        self._generators: Dict[TargetLanguage, LanguageGenerator] = {}
        self._features: Dict[TargetLanguage, LanguageFeatures] = {}
        self._initialize_languages()
    
    def _initialize_languages(self):
        """Initialize supported languages with their features"""
        
        # Myndra (original)
        myndra_features = LanguageFeatures(
            name="Myndra",
            version="1.0",
            has_generics=True,
            has_async_await=True,
            has_pattern_matching=True,
            has_null_safety=True,
            has_memory_safety=True,
            has_type_inference=True,
            has_traits_interfaces=True,
            concurrency_model="reactive",
            has_channels=True,
            error_handling="fallback_strategies",
            package_manager="myndra",
            package_file="myndra.toml",
            build_system="myndra",
            build_file="myndra.toml"
        )
        self._features[TargetLanguage.MYNDRA] = myndra_features
        
        # Rust
        rust_features = LanguageFeatures(
            name="Rust",
            version="1.75",
            has_generics=True,
            has_async_await=True,
            has_pattern_matching=True,
            has_null_safety=True,
            has_memory_safety=True,
            has_type_inference=True,
            has_macros=True,
            has_traits_interfaces=True,
            concurrency_model="async",
            has_channels=True,
            error_handling="result_types",
            package_manager="cargo",
            package_file="Cargo.toml",
            build_system="cargo",
            build_file="Cargo.toml",
            web_frameworks=["axum", "warp", "rocket", "actix-web"],
            testing_frameworks=["built-in", "proptest", "criterion"],
            orm_libraries=["diesel", "sqlx", "sea-orm"]
        )
        self._features[TargetLanguage.RUST] = rust_features
        
        # Go
        go_features = LanguageFeatures(
            name="Go",
            version="1.21",
            has_generics=True,
            has_async_await=False,  # Uses goroutines instead
            has_pattern_matching=False,
            has_null_safety=False,
            has_memory_safety=True,
            has_type_inference=True,
            has_traits_interfaces=True,
            concurrency_model="goroutines",
            has_channels=True,
            has_green_threads=True,
            error_handling="explicit_errors",
            package_manager="go_modules",
            package_file="go.mod",
            build_system="go",
            build_file="go.mod",
            web_frameworks=["gin", "echo", "fiber", "chi"],
            testing_frameworks=["built-in", "testify", "ginkgo"],
            orm_libraries=["gorm", "sqlx", "ent"]
        )
        self._features[TargetLanguage.GO] = go_features
        
        # TypeScript
        typescript_features = LanguageFeatures(
            name="TypeScript",
            version="5.3",
            has_generics=True,
            has_async_await=True,
            has_pattern_matching=False,
            has_null_safety=True,
            has_memory_safety=False,
            has_type_inference=True,
            has_traits_interfaces=True,
            concurrency_model="event_loop",
            error_handling="exceptions",
            package_manager="npm",
            package_file="package.json",
            build_system="tsc",
            build_file="tsconfig.json",
            web_frameworks=["express", "fastify", "nest", "koa"],
            testing_frameworks=["jest", "vitest", "mocha", "playwright"],
            orm_libraries=["prisma", "typeorm", "sequelize", "drizzle"]
        )
        self._features[TargetLanguage.TYPESCRIPT] = typescript_features
        
        # Modern Python
        python_features = LanguageFeatures(
            name="Python",
            version="3.12",
            has_generics=True,
            has_async_await=True,
            has_pattern_matching=True,
            has_null_safety=False,  # Optional with mypy
            has_memory_safety=True,
            has_type_inference=False,  # Optional with mypy
            has_traits_interfaces=True,
            concurrency_model="async",
            error_handling="exceptions",
            package_manager="pip",
            package_file="requirements.txt",
            build_system="setuptools",
            build_file="setup.py",
            web_frameworks=["fastapi", "django", "flask", "starlette"],
            testing_frameworks=["pytest", "unittest", "hypothesis"],
            orm_libraries=["sqlalchemy", "django-orm", "tortoise"]
        )
        self._features[TargetLanguage.PYTHON] = python_features
    
    def register_generator(self, language: TargetLanguage, generator: LanguageGenerator):
        """Register a generator for a target language"""
        self._generators[language] = generator
    
    def get_generator(self, language: TargetLanguage) -> Optional[LanguageGenerator]:
        """Get generator for a target language"""
        return self._generators.get(language)
    
    def get_features(self, language: TargetLanguage) -> Optional[LanguageFeatures]:
        """Get features for a target language"""
        return self._features.get(language)
    
    def get_supported_languages(self) -> List[TargetLanguage]:
        """Get list of all supported target languages"""
        return list(self._features.keys())
    
    def get_language_recommendations(self, source_analysis: Dict[str, Any], context: CodeGenerationContext) -> List[Dict[str, Any]]:
        """Get language recommendations based on source analysis and context"""
        recommendations = []
        
        for language in self.get_supported_languages():
            features = self.get_features(language)
            generator = self.get_generator(language)
            
            if not features:
                continue
                
            # Calculate compatibility score
            score = self._calculate_compatibility_score(source_analysis, features, context)
            
            # Get migration strategies if generator available
            strategies = []
            effort = {"complexity": "unknown", "estimated_days": 0}
            if generator:
                strategies = generator.get_migration_strategies(source_analysis)
                effort = generator.estimate_migration_effort(source_analysis)
            
            recommendations.append({
                "language": language,
                "features": features,
                "compatibility_score": score,
                "migration_strategies": strategies,
                "effort_estimate": effort,
                "pros": self._get_language_pros(features, context),
                "cons": self._get_language_cons(features, context)
            })
        
        # Sort by compatibility score
        recommendations.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return recommendations
    
    def _calculate_compatibility_score(self, source_analysis: Dict[str, Any], features: LanguageFeatures, context: CodeGenerationContext) -> float:
        """Calculate compatibility score between source and target"""
        score = 0.5  # Base score
        
        # Domain compatibility
        domains = source_analysis.get('domains', [])
        if 'web_development' in domains and features.web_frameworks:
            score += 0.2
        if 'data_processing' in domains and features.has_async_await:
            score += 0.1
        
        # Performance requirements
        if context.performance_requirements == "high":
            if features.has_memory_safety and features.name in ["Rust", "Go"]:
                score += 0.2
        
        # Type safety requirements
        if context.add_type_safety and features.has_null_safety:
            score += 0.1
        
        # Error handling modernization
        if context.improve_error_handling and features.error_handling in ["result_types", "fallback_strategies"]:
            score += 0.1
        
        # Team experience
        team_exp = context.team_experience.get(features.name.lower(), "none")
        if team_exp == "expert":
            score += 0.1
        elif team_exp == "intermediate":
            score += 0.05
        elif team_exp == "none":
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _get_language_pros(self, features: LanguageFeatures, context: CodeGenerationContext) -> List[str]:
        """Get pros for a language in given context"""
        pros = []
        
        if features.has_memory_safety:
            pros.append("Memory safety prevents crashes and security vulnerabilities")
        if features.has_null_safety:
            pros.append("Null safety eliminates null pointer exceptions")
        if features.has_async_await:
            pros.append("Modern async/await for concurrent programming")
        if features.has_pattern_matching:
            pros.append("Pattern matching for expressive code")
        if features.concurrency_model in ["goroutines", "reactive"]:
            pros.append("Advanced concurrency model for high performance")
        if features.package_manager != "none":
            pros.append(f"Modern package management with {features.package_manager}")
        
        return pros
    
    def _get_language_cons(self, features: LanguageFeatures, context: CodeGenerationContext) -> List[str]:
        """Get cons for a language in given context"""
        cons = []
        
        if not features.has_generics:
            cons.append("Limited generic programming capabilities")
        if features.error_handling == "exceptions" and context.improve_error_handling:
            cons.append("Traditional exception-based error handling")
        if not features.has_type_inference:
            cons.append("Requires explicit type annotations")
        
        # Team experience concerns
        team_exp = context.team_experience.get(features.name.lower(), "none")
        if team_exp == "none":
            cons.append(f"Team has no experience with {features.name}")
        
        return cons


# Global registry instance
language_registry = LanguageRegistry()