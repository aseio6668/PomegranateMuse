#!/usr/bin/env python3
"""
PomegranteMuse - ML-powered code analysis and translation tool for Pomegranate
A cross-platform tool that uses machine learning to analyze source code files
and translate them into idiomatic Pomegranate programming language code.
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Import our custom modules
from ollama_client import OllamaClient, CodeAnalysisPrompts
from build_tester import BuildTestingIntegration
from cicd import CICDIntegration, setup_cicd_for_project, configure_cicd_interactive
from benchmarking import BenchmarkIntegration, run_benchmark_interactive
from security import SecurityIntegration, run_security_scan_interactive
from cost_analysis import CostIntegration, run_cost_analysis_interactive
from collaboration import TeamIntegration, run_team_management_interactive
from enterprise import EnterpriseManager, run_enterprise_setup_interactive
from universal_testing import (
    UniversalBuilder, TestConfiguration, TestOrchestrator, 
    BuildMatrix, generate_build_matrix, run_comprehensive_tests
)
from migration_strategy import (
    MigrationPlanner, MigrationStrategy, create_migration_plan,
    ProgressTracker, StatusDashboard, LegacyCodeAnalyzer
)

# Version info
__version__ = "0.1.0"
__author__ = "PomegranteMuse Contributors"

class ProjectState:
    """Manages project state and .pomuse directory"""
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.pomuse_dir = working_dir / ".pomuse"
        self.state_file = self.pomuse_dir / "project_state.json"
        self.conversations_dir = self.pomuse_dir / "conversations"
        self.outputs_dir = self.pomuse_dir / "outputs"
        
    def initialize(self):
        """Initialize .pomuse directory structure"""
        self.pomuse_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        if not self.state_file.exists():
            initial_state = {
                "version": __version__,
                "created_at": str(datetime.now()),
                "last_updated": str(datetime.now()),
                "project_name": self.working_dir.name,
                "settings": {
                    "model_provider": "ollama",
                    "default_model": "codellama",
                    "auto_execute_builds": False,
                    "remember_file_permissions": True
                },
                "conversations": [],
                "generated_files": []
            }
            self.save_state(initial_state)
    
    def load_state(self) -> Dict[str, Any]:
        """Load project state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_state(self, state: Dict[str, Any]):
        """Save project state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

class CodeAnalyzer:
    """Analyzes source code files using ML models"""
    
    def __init__(self, model_provider: str = "ollama"):
        self.model_provider = model_provider
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.clj': 'clojure',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.fs': 'fsharp',
            '.jl': 'julia',
            '.r': 'r',
            '.m': 'matlab',
            '.lua': 'lua',
            '.pl': 'perl',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
    
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file and extract semantic information"""
        file_ext = file_path.suffix.lower()
        language = self.supported_extensions.get(file_ext, 'unknown')
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {
                'file_path': str(file_path),
                'language': language,
                'error': f"Failed to read file: {e}",
                'analysis': None
            }
        
        # Basic static analysis
        analysis = {
            'file_path': str(file_path),
            'language': language,
            'size_bytes': len(content.encode('utf-8')),
            'line_count': len(content.split('\n')),
            'has_imports': self._detect_imports(content, language),
            'has_functions': self._detect_functions(content, language),
            'has_classes': self._detect_classes(content, language),
            'complexity_indicators': self._assess_complexity(content, language),
            'domain_hints': self._detect_domain(content, language)
        }
        
        return analysis
    
    def _detect_imports(self, content: str, language: str) -> bool:
        """Detect if file has import/include statements"""
        import_patterns = {
            'python': ['import ', 'from '],
            'javascript': ['import ', 'require(', 'export '],
            'typescript': ['import ', 'export ', 'require('],
            'java': ['import ', 'package '],
            'cpp': ['#include', 'using namespace'],
            'c': ['#include'],
            'csharp': ['using ', 'namespace '],
            'go': ['import ', 'package '],
            'rust': ['use ', 'extern crate'],
            'ruby': ['require ', 'include '],
        }
        
        patterns = import_patterns.get(language, [])
        return any(pattern in content for pattern in patterns)
    
    def _detect_functions(self, content: str, language: str) -> bool:
        """Detect if file contains function definitions"""
        function_patterns = {
            'python': ['def '],
            'javascript': ['function ', '=>', 'async '],
            'typescript': ['function ', '=>', 'async '],
            'java': ['public ', 'private ', 'protected '],
            'cpp': ['void ', 'int ', 'return '],
            'c': ['void ', 'int ', 'return '],
            'csharp': ['public ', 'private ', 'protected '],
            'go': ['func '],
            'rust': ['fn '],
            'ruby': ['def '],
        }
        
        patterns = function_patterns.get(language, [])
        return any(pattern in content for pattern in patterns)
    
    def _detect_classes(self, content: str, language: str) -> bool:
        """Detect if file contains class definitions"""
        class_patterns = {
            'python': ['class '],
            'javascript': ['class '],
            'typescript': ['class ', 'interface '],
            'java': ['class ', 'interface '],
            'cpp': ['class ', 'struct '],
            'c': ['struct '],
            'csharp': ['class ', 'interface ', 'struct '],
            'go': ['type ', 'struct '],
            'rust': ['struct ', 'enum ', 'impl '],
            'ruby': ['class ', 'module '],
        }
        
        patterns = class_patterns.get(language, [])
        return any(pattern in content for pattern in patterns)
    
    def _assess_complexity(self, content: str, language: str) -> Dict[str, int]:
        """Assess code complexity indicators"""
        return {
            'nested_blocks': content.count('{') + content.count('if ') + content.count('for '),
            'conditional_statements': content.count('if ') + content.count('elif ') + content.count('else'),
            'loops': content.count('for ') + content.count('while '),
            'try_catch_blocks': content.count('try') + content.count('catch') + content.count('except'),
        }
    
    def _detect_domain(self, content: str, language: str) -> List[str]:
        """Detect domain-specific hints from content"""
        domains = []
        
        # Math/Scientific computing
        math_keywords = ['numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn', 'tensorflow', 'pytorch', 
                        'math', 'statistics', 'algorithm', 'matrix', 'vector', 'calculation']
        if any(keyword in content.lower() for keyword in math_keywords):
            domains.append('math_scientific')
        
        # Web development
        web_keywords = ['http', 'api', 'server', 'client', 'request', 'response', 'html', 'css', 
                       'javascript', 'react', 'vue', 'angular', 'express', 'django', 'flask']
        if any(keyword in content.lower() for keyword in web_keywords):
            domains.append('web_development')
        
        # Data processing
        data_keywords = ['database', 'sql', 'json', 'csv', 'xml', 'parse', 'serialize', 'query']
        if any(keyword in content.lower() for keyword in data_keywords):
            domains.append('data_processing')
        
        # Game development
        game_keywords = ['game', 'player', 'render', 'graphics', 'animation', 'physics', 'collision']
        if any(keyword in content.lower() for keyword in game_keywords):
            domains.append('game_development')
        
        # Security/Crypto
        security_keywords = ['crypto', 'hash', 'encrypt', 'decrypt', 'security', 'auth', 'token']
        if any(keyword in content.lower() for keyword in security_keywords):
            domains.append('security_crypto')
        
        return domains

class EnhancedOllamaProvider:
    """Enhanced interface to Ollama models for code analysis and translation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "codellama"):
        self.base_url = base_url
        self.default_model = default_model
        self.client = None
        self.available_models = []
    
    async def initialize(self) -> bool:
        """Initialize connection and check available models"""
        self.client = OllamaClient(self.base_url)
        
        # Check connection
        async with self.client as client:
            connected = await client.check_connection()
            if not connected:
                return False
            
            # Get available models
            self.available_models = await client.list_models()
            
            # Ensure we have a suitable model
            suitable_models = [m for m in self.available_models if any(
                keyword in m.lower() for keyword in ['code', 'llama', 'mistral', 'deepseek']
            )]
            
            if not suitable_models and self.available_models:
                self.default_model = self.available_models[0]
            elif suitable_models:
                self.default_model = suitable_models[0]
            
            return True
    
    async def analyze_file_with_ml(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """Use ML model to analyze individual file"""
        if not self.client:
            await self.initialize()
        
        prompt = CodeAnalysisPrompts.analyze_file_content(file_path, content, language)
        
        try:
            async with self.client as client:
                response = await client.generate(
                    model=self.default_model,
                    prompt=prompt,
                    system="You are an expert software engineer. Always respond with valid JSON.",
                    temperature=0.1
                )
                
                # Try to parse JSON response
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    # Fallback to basic analysis if JSON parsing fails
                    return {
                        "primary_purpose": "Code analysis failed - fallback response",
                        "key_functions": [],
                        "complexity_level": "unknown",
                        "domain_classification": ["general"],
                        "translation_approach": "Use basic template"
                    }
        except Exception as e:
            print(f"ML analysis failed for {file_path}: {e}")
            return {
                "primary_purpose": f"Analysis failed: {e}",
                "key_functions": [],
                "complexity_level": "unknown",
                "domain_classification": ["general"],
                "translation_approach": "Use basic template"
            }
    
    async def analyze_code_context(self, files_analysis: List[Dict], user_prompt: str) -> Dict[str, Any]:
        """Use ML model to understand code context and generate translation plan"""
        if not self.client:
            await self.initialize()
        
        prompt = CodeAnalysisPrompts.analyze_project_context(files_analysis, user_prompt)
        
        try:
            async with self.client as client:
                response = await client.generate(
                    model=self.default_model,
                    prompt=prompt,
                    system="You are an expert software architect. Always respond with valid JSON.",
                    temperature=0.2
                )
                
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    # Fallback response
                    return {
                        "project_type": "Unknown project type",
                        "pomegranate_architecture": {
                            "recommended_style": "traditional",
                            "key_modules": ["main"],
                            "security_approach": "basic"
                        },
                        "translation_strategy": {
                            "template_type": "basic_app",
                            "key_transformations": ["direct_translation"],
                            "pomegranate_features_to_use": ["basic_syntax"]
                        },
                        "implementation_plan": ["Create basic structure", "Translate core logic"],
                        "recommendations": ["Start with simple approach"]
                    }
        except Exception as e:
            print(f"Context analysis failed: {e}")
            return {
                "project_type": f"Analysis failed: {e}",
                "translation_strategy": {"template_type": "basic_app"},
                "implementation_plan": ["Manual review required"],
                "recommendations": ["Check Ollama connection"]
            }
    
    async def generate_pomegranate_code(self, context: Dict[str, Any], user_prompt: str) -> str:
        """Generate Pomegranate code using ML model"""
        if not self.client:
            await self.initialize()
        
        template_type = context.get('translation_strategy', {}).get('template_type', 'basic_app')
        prompt = CodeAnalysisPrompts.generate_pomegranate_code(context, user_prompt, template_type)
        
        try:
            async with self.client as client:
                response = await client.generate(
                    model=self.default_model,
                    prompt=prompt,
                    system="You are an expert Pomegranate programmer. Generate clean, idiomatic code.",
                    temperature=0.3
                )
                return response.content
        except Exception as e:
            print(f"Code generation failed: {e}")
            return f"""// Code generation failed: {e}
// Fallback basic Pomegranate template

import std::io with capabilities("write")

fn main() {{
    print("Generated code placeholder - please review manually")
    // TODO: Implement based on user prompt: {user_prompt}
}}
"""

class PomegranateGenerator:
    """Generates Pomegranate code from analysis and ML insights"""
    
    def __init__(self, ollama_provider: EnhancedOllamaProvider):
        self.ollama = ollama_provider
        self.pomegranate_templates = {
            "basic_app": self._basic_app_template,
            "reactive_ui": self._reactive_ui_template,
            "api_client": self._api_client_template,
            "data_processor": self._data_processor_template
        }
    
    async def generate_code(self, analysis: Dict[str, Any], user_prompt: str) -> str:
        """Generate Pomegranate code based on analysis and user intent"""
        
        # Use ML model for code generation if available
        try:
            ml_generated_code = await self.ollama.generate_pomegranate_code(analysis, user_prompt)
            if ml_generated_code and not ml_generated_code.startswith("// Code generation failed"):
                return ml_generated_code
        except Exception as e:
            print(f"ML generation failed, falling back to templates: {e}")
        
        # Fallback to template-based generation
        template_name = self._select_template(analysis)
        template_func = self.pomegranate_templates.get(template_name, self._basic_app_template)
        
        # Generate base code structure
        base_code = template_func(analysis, user_prompt)
        
        return base_code
    
    def _select_template(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate template based on analysis"""
        domains = []
        for file_analysis in analysis.get('files', []):
            domains.extend(file_analysis.get('domain_hints', []))
        
        if 'web_development' in domains:
            return "reactive_ui"
        elif 'data_processing' in domains:
            return "data_processor"
        elif any('api' in domain for domain in domains):
            return "api_client"
        else:
            return "basic_app"
    
    def _basic_app_template(self, analysis: Dict[str, Any], prompt: str) -> str:
        """Basic Pomegranate application template"""
        return f'''// Generated Pomegranate application
// Original prompt: {prompt}

import std::io with capabilities("write")

#tag:app:main
fn main() {{
    log("Starting Pomegranate application") if context == "dev"
    
    // TODO: Implement core functionality based on analysis
    let app_data = initialize_app()
    run_application(app_data)
    
    log("Application completed") if context == "dev"
}}

#tag:core
fn initialize_app() -> AppData {{
    // Initialize application state
    return AppData {{
        // Add fields based on analysis
    }}
}}

#tag:core
fn run_application(data: AppData) {{
    // Main application logic
    print("Pomegranate application running!")
}}
'''
    
    def _reactive_ui_template(self, analysis: Dict[str, Any], prompt: str) -> str:
        """Reactive UI Pomegranate template"""
        return f'''// Generated Pomegranate UI application
// Original prompt: {prompt}

import std::ui with capabilities("render", "event-handling")
import std::reactive with capabilities("observe", "emit")

#tag:app:main
@reactive
capsule main_app {{
    let app_state: evolving<AppState> = AppState::default()
    
    #tag:ui:render
    fn render() -> Element {{
        return ui::div([
            ui::header("Pomegranate App"),
            render_main_content(),
            ui::footer("Generated by PomegranteMuse")
        ])
    }}
    
    #tag:ui:components
    fn render_main_content() -> Element {{
        return ui::div([
            ui::h2("Welcome to your Pomegranate application"),
            ui::p("This UI adapts reactively to state changes")
        ])
    }}
}}

#tag:app:main
fn main() {{
    let app = main_app.render()
    ui::mount(app, "#app")
    log("Reactive UI started") if context == "dev"
}}
'''
    
    def _api_client_template(self, analysis: Dict[str, Any], prompt: str) -> str:
        """API client Pomegranate template"""
        return f'''// Generated Pomegranate API client
// Original prompt: {prompt}

import net::http with capabilities("fetch", "post")
import std::json with capabilities("parse", "stringify")

#tag:api:client
@async
fn api_client() {{
    
    #tag:api:methods
    fn get_data(endpoint: string) -> Promise<ApiResponse>
        fallback retry(3) or return default_response() {{
        
        let response = http::get(base_url + endpoint)
        return ApiResponse::from_json(response.body)
    }}
    
    #tag:api:methods
    fn post_data(endpoint: string, data: ApiData) -> Promise<ApiResponse>
        fallback retry(2) or return error_response() {{
        
        let json_data = data.to_json()
        let response = http::post(base_url + endpoint, json_data)
        return ApiResponse::from_json(response.body)
    }}
}}

#tag:app:main
fn main() {{
    log("API client initialized") if context == "dev"
    
    let client = api_client()
    // Use client for API operations
}}
'''
    
    def _data_processor_template(self, analysis: Dict[str, Any], prompt: str) -> str:
        """Data processing Pomegranate template"""
        return f'''// Generated Pomegranate data processor
// Original prompt: {prompt}

import std::collections with capabilities("iterate", "transform")
import std::io with capabilities("read", "write")

#tag:data:processing
@parallel
fn process_data(input_data: [DataItem]) -> [ProcessedItem] {{
    
    #tag:data:transform
    fn transform_item(item: DataItem) -> ProcessedItem {{
        // Transform individual data items
        return ProcessedItem {{
            id: item.id,
            processed_value: apply_processing_logic(item.value),
            timestamp: now()
        }}
    }}
    
    #tag:data:validate
    fn validate_item(item: DataItem) -> bool {{
        // Validate data before processing
        return item.is_valid() and item.value.is_some()
    }}
    
    // Parallel processing pipeline
    return input_data
        .filter(validate_item)
        .map(transform_item)
}}

#tag:app:main
fn main() {{
    log("Data processor started") if context == "dev"
    
    let input_data = load_input_data()
    let processed = process_data(input_data)
    
    save_processed_data(processed)
    log("Processing complete") if context == "dev"
}}
'''
    
    async def _refine_with_ml(self, base_code: str, analysis: Dict[str, Any], prompt: str) -> str:
        """Use ML model to refine generated code"""
        # This would make an API call to Ollama to refine the code
        # For now, return the base code with some basic refinements
        
        refinement_prompt = f"""
        Refine this Pomegranate code based on the analysis and user requirements:
        
        Base code:
        {base_code}
        
        Analysis:
        {json.dumps(analysis, indent=2)}
        
        User prompt: {prompt}
        
        Please improve the code by:
        1. Adding appropriate error handling
        2. Including relevant domain-specific logic
        3. Optimizing for the detected use case
        4. Following Pomegranate best practices
        """
        
        # Placeholder - would use actual ML model
        return base_code

class InteractiveCLI:
    """Interactive command-line interface for PomegranteMuse"""
    
    def __init__(self):
        self.project_state = None
        self.code_analyzer = CodeAnalyzer()
        self.ollama_provider = EnhancedOllamaProvider()
        self.pomegranate_generator = PomegranateGenerator(self.ollama_provider)
        self.build_tester = BuildTestingIntegration()
        self.cicd_integration = None  # Will be initialized with working directory
        self.enterprise_manager = None  # Will be initialized with working directory
    
    async def start_interactive_session(self, working_dir: Path):
        """Start interactive session"""
        print(f"🍎 Welcome to PomegranteMuse v{__version__}")
        print(f"Working directory: {working_dir}")
        
        # Initialize project
        self.project_state = ProjectState(working_dir)
        self.project_state.initialize()
        
        # Initialize CI/CD integration
        self.cicd_integration = CICDIntegration(str(working_dir))
        
        # Initialize benchmarking integration
        self.benchmark_integration = BenchmarkIntegration(str(working_dir))
        
        # Initialize security integration
        self.security_integration = SecurityIntegration(str(working_dir))
        
        # Initialize cost analysis integration
        self.cost_integration = CostIntegration(str(working_dir))
        
        # Initialize team collaboration integration
        self.team_integration = TeamIntegration(str(working_dir))
        
        # Initialize enterprise integration
        self.enterprise_manager = EnterpriseManager.load_config(str(working_dir / ".pomuse"))
        
        # Initialize universal testing integration
        self.universal_builder = UniversalBuilder(str(working_dir / ".pomuse"))
        self.test_orchestrator = TestOrchestrator(str(working_dir / ".pomuse"))
        self.build_matrix = BuildMatrix(str(working_dir / ".pomuse"))
        
        # Initialize migration strategy integration
        self.migration_planner = MigrationPlanner(str(working_dir / ".pomuse"))
        self.progress_tracker = ProgressTracker(str(working_dir / ".pomuse"))
        self.status_dashboard = StatusDashboard(self.progress_tracker)
        self.legacy_analyzer = LegacyCodeAnalyzer()
        
        # Initialize Ollama connection
        print("Initializing ML provider...")
        try:
            if await self.ollama_provider.initialize():
                print(f"✅ Connected to Ollama (model: {self.ollama_provider.default_model})")
            else:
                print("⚠️  Ollama connection failed - using fallback templates")
        except Exception as e:
            print(f"⚠️  Ollama initialization error: {e}")
        
        print("\nType 'help' for available commands, or 'exit' to quit.")
        
        while True:
            try:
                command = input("\npomuse> ").strip()
                if not command:
                    continue
                
                if command == "exit":
                    print("Goodbye! 🍎")
                    break
                elif command == "help":
                    self._show_help()
                elif command.startswith("analyze"):
                    await self._handle_analyze_command(command)
                elif command.startswith("generate"):
                    await self._handle_generate_command(command)
                elif command.startswith("continue"):
                    await self._handle_continue_command()
                elif command.startswith("status"):
                    self._handle_status_command()
                elif command.startswith("build"):
                    await self._handle_build_command(command)
                elif command.startswith("cicd"):
                    await self._handle_cicd_command(command)
                elif command.startswith("benchmark"):
                    await self._handle_benchmark_command(command)
                elif command.startswith("security"):
                    await self._handle_security_command(command)
                elif command.startswith("cost"):
                    await self._handle_cost_command(command)
                elif command.startswith("team"):
                    await self._handle_team_command(command)
                elif command.startswith("enterprise"):
                    await self._handle_enterprise_command(command)
                elif command.startswith("test"):
                    await self._handle_test_command(command)
                elif command.startswith("migrate"):
                    await self._handle_migrate_command(command)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show available commands"""
        help_text = """
Available commands:

  analyze [path]     - Analyze source code files in the specified path
  generate <prompt>  - Generate Pomegranate code based on analysis and prompt
  continue          - Continue from last session using .pomuse state
  status            - Show current project status
  build test [file]  - Test build a specific generated file
  cicd setup         - Set up CI/CD pipelines interactively
  cicd generate      - Generate CI/CD pipelines with auto-detection
  cicd update        - Update existing CI/CD configuration
  cicd providers     - List supported CI/CD providers and features
  cicd languages     - List supported programming languages
  benchmark run      - Run interactive benchmarking session
  benchmark suite    - Run a specific benchmark suite
  benchmark file     - Benchmark a specific file
  benchmark report   - Generate performance report
  security scan      - Run comprehensive security analysis
  security quick     - Run quick security scan (static only)
  security gate      - Run security gate checks
  security dashboard - View security dashboard
  security policy    - Configure security policy
  cost analyze       - Run comprehensive cost analysis
  cost estimate      - Quick cost estimation
  cost dashboard     - View cost dashboard and trends
  cost budget        - Set up and manage cost budgets
  cost optimize      - Get optimization recommendations
  team dashboard     - View team collaboration dashboard
  team members       - Manage team members and roles
  team reviews       - Manage code review process
  team settings      - Configure team settings and workflows
  team activity      - View team activity and reports
  enterprise setup   - Configure enterprise integrations
  enterprise status  - View integration status
  enterprise auth    - Test authentication
  enterprise project - Create migration project with integrations
  test universal     - Run universal build testing
  test matrix        - Execute build matrix testing
  test coverage      - Run comprehensive coverage analysis
  test compatibility - Check cross-platform compatibility
  migrate plan       - Create comprehensive migration plan
  migrate execute    - Execute migration plan
  migrate status     - View migration progress dashboard
  migrate analyze    - Analyze legacy codebase
  help              - Show this help message
  exit              - Exit PomegranteMuse

Examples:
  analyze ./src
  generate "create a math framework from these files"
  cicd setup
  cicd generate rust github_actions
  benchmark run
  benchmark suite code_generation
  security scan
  security gate pre_commit
  cost analyze
  cost budget 500
  team dashboard
  enterprise setup
  enterprise status
  enterprise project "Legacy Migration"
  test universal rust go typescript
  test matrix --minimal
  test coverage python
  migrate plan python pomegranate
  migrate analyze legacy_project
  migrate status
  team members
  continue
        """
        print(help_text)
    
    async def _handle_analyze_command(self, command: str):
        """Handle analyze command"""
        parts = command.split(maxsplit=1)
        target_path = Path(parts[1]) if len(parts) > 1 else Path(".")
        
        if not target_path.exists():
            print(f"Path does not exist: {target_path}")
            return
        
        print(f"Analyzing files in: {target_path}")
        
        # Find all source files
        source_files = []
        if target_path.is_file():
            source_files = [target_path]
        else:
            for ext in self.code_analyzer.supported_extensions:
                source_files.extend(target_path.rglob(f"*{ext}"))
        
        if not source_files:
            print("No supported source files found.")
            return
        
        print(f"Found {len(source_files)} source files")
        
        # Analyze files
        analyses = []
        for file_path in source_files[:20]:  # Limit for demo
            print(f"  Analyzing: {file_path.name}")
            
            # Basic static analysis
            basic_analysis = await self.code_analyzer.analyze_file(file_path)
            
            # Enhanced ML analysis if available
            if basic_analysis.get('analysis') and not basic_analysis.get('error'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Get ML-enhanced analysis
                    ml_analysis = await self.ollama_provider.analyze_file_with_ml(
                        str(file_path), content, basic_analysis.get('language', 'unknown')
                    )
                    
                    # Merge analyses
                    enhanced_analysis = {**basic_analysis, 'ml_analysis': ml_analysis}
                    analyses.append(enhanced_analysis)
                    
                except Exception as e:
                    print(f"    ML analysis failed for {file_path.name}: {e}")
                    analyses.append(basic_analysis)
            else:
                analyses.append(basic_analysis)
        
        # Store analysis results
        state = self.project_state.load_state()
        state['last_analysis'] = {
            'timestamp': str(datetime.now()),
            'target_path': str(target_path),
            'files_analyzed': len(analyses),
            'analyses': analyses
        }
        self.project_state.save_state(state)
        
        print(f"\nAnalysis complete! Found patterns in {len(analyses)} files.")
        self._show_analysis_summary(analyses)
    
    def _show_analysis_summary(self, analyses: List[Dict]):
        """Show summary of analysis results"""
        languages = {}
        domains = set()
        total_lines = 0
        
        for analysis in analyses:
            lang = analysis.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
            domains.update(analysis.get('domain_hints', []))
            total_lines += analysis.get('line_count', 0)
        
        print("\n📊 Analysis Summary:")
        print(f"   Total lines of code: {total_lines}")
        print(f"   Languages: {', '.join(languages.keys())}")
        if domains:
            print(f"   Detected domains: {', '.join(domains)}")
    
    async def _handle_generate_command(self, command: str):
        """Handle generate command"""
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("Please provide a prompt for code generation.")
            print("Example: generate 'create a math framework from these files'")
            return
        
        user_prompt = parts[1].strip('"\'')
        
        # Load last analysis
        state = self.project_state.load_state()
        last_analysis = state.get('last_analysis')
        
        if not last_analysis:
            print("No analysis found. Please run 'analyze' first.")
            return
        
        print(f"Generating Pomegranate code for: {user_prompt}")
        
        # Get ML analysis
        ml_analysis = await self.ollama_provider.analyze_code_context(
            last_analysis['analyses'], user_prompt
        )
        
        # Generate code
        generated_code = await self.pomegranate_generator.generate_code(
            {'files': last_analysis['analyses'], 'ml_analysis': ml_analysis},
            user_prompt
        )
        
        # Test and improve generated code
        print("\n🔨 Testing generated code...")
        test_result = await self.build_tester.test_and_improve_generated_code(
            generated_code,
            {'files': last_analysis['analyses'], 'ml_analysis': ml_analysis},
            self.ollama_provider
        )
        
        final_code = test_result.get('final_code', generated_code)
        
        # Save generated code
        output_file = self.project_state.outputs_dir / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pom"
        with open(output_file, 'w') as f:
            f.write(final_code)
        
        print(f"\n✅ Generated Pomegranate code saved to: {output_file}")
        
        # Handle team collaboration workflow
        try:
            team_result = await self.team_integration.handle_code_generation(
                generated_code=final_code,
                source_analysis={'files': last_analysis['analyses'], 'ml_analysis': ml_analysis},
                user_prompt=user_prompt
            )
            
            if team_result.get("requires_review"):
                review_id = team_result.get("review_request_id")
                print(f"📋 Code review required - Review ID: {review_id}")
                print(f"   The generated code has been submitted for team review.")
            
            for action in team_result.get("workflow_actions", []):
                if action == "review_request_created":
                    print(f"🔄 Workflow: Review request created automatically")
        
        except Exception as e:
            print(f"Note: Team workflow unavailable: {e}")
        
        # Show test results
        if test_result.get('tested'):
            if test_result.get('success'):
                print(f"🎉 Code builds successfully!")
            else:
                print(f"⚠️  Code has build issues (saved anyway)")
            
            if test_result.get('iterations', 0) > 1:
                print(f"🔧 Applied {test_result.get('iterations')} improvement iterations")
        else:
            reason = test_result.get('reason', 'Unknown')
            print(f"⚠️  Build testing skipped: {reason}")
        
        print("\n📝 Generated code preview:")
        print("=" * 50)
        print(final_code[:500] + "..." if len(final_code) > 500 else final_code)
        print("=" * 50)
        
        # Update state
        state['generated_files'] = state.get('generated_files', [])
        state['generated_files'].append({
            'timestamp': str(datetime.now()),
            'prompt': user_prompt,
            'file_path': str(output_file),
            'ml_analysis': ml_analysis,
            'build_result': test_result
        })
        self.project_state.save_state(state)
    
    async def _handle_continue_command(self):
        """Handle continue command"""
        state = self.project_state.load_state()
        
        if not state:
            print("No previous session found.")
            return
        
        print("📚 Previous session information:")
        print(f"   Project: {state.get('project_name', 'Unknown')}")
        print(f"   Created: {state.get('created_at', 'Unknown')}")
        print(f"   Last updated: {state.get('last_updated', 'Unknown')}")
        
        if 'last_analysis' in state:
            analysis = state['last_analysis']
            print(f"   Last analysis: {analysis['files_analyzed']} files in {analysis['target_path']}")
        
        if 'generated_files' in state:
            files = state['generated_files']
            print(f"   Generated files: {len(files)}")
            for file_info in files[-3:]:  # Show last 3
                print(f"     - {Path(file_info['file_path']).name}: '{file_info['prompt']}'")
        
        print("\nSession restored! You can continue with 'analyze' or 'generate' commands.")
    
    def _handle_status_command(self):
        """Handle status command"""
        state = self.project_state.load_state()
        
        print("📊 Project Status:")
        print(f"   Working directory: {self.project_state.working_dir}")
        print(f"   .pomuse directory: {'✅ exists' if self.project_state.pomuse_dir.exists() else '❌ missing'}")
        
        if state:
            print(f"   Project name: {state.get('project_name', 'Unknown')}")
            print(f"   Model provider: {state.get('settings', {}).get('model_provider', 'ollama')}")
            
            last_analysis = state.get('last_analysis')
            if last_analysis:
                print(f"   Last analysis: {last_analysis['files_analyzed']} files")
            else:
                print("   Last analysis: None")
            
            generated = state.get('generated_files', [])
            print(f"   Generated files: {len(generated)}")
        else:
            print("   No project state found")
    
    async def _handle_build_command(self, command: str):
        """Handle build command"""
        parts = command.split()
        
        if len(parts) < 2 or parts[1] != "test":
            print("Usage: build test [file]")
            print("       build test (tests most recent generated file)")
            return
        
        # Find file to test
        target_file = None
        if len(parts) >= 3:
            # Specific file requested
            target_path = Path(parts[2])
            if target_path.exists():
                target_file = target_path
            else:
                # Try in outputs directory
                outputs_path = self.project_state.outputs_dir / parts[2]
                if outputs_path.exists():
                    target_file = outputs_path
                else:
                    print(f"File not found: {parts[2]}")
                    return
        else:
            # Use most recent generated file
            state = self.project_state.load_state()
            generated_files = state.get('generated_files', [])
            
            if not generated_files:
                print("No generated files found. Use 'generate' first.")
                return
            
            # Get most recent file
            most_recent = generated_files[-1]
            target_file = Path(most_recent['file_path'])
            
            if not target_file.exists():
                print(f"Generated file not found: {target_file}")
                return
        
        print(f"🔨 Testing build for: {target_file.name}")
        
        # Read file content
        try:
            with open(target_file, 'r') as f:
                code_content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        # Test the build
        test_result = await self.build_tester.test_and_improve_generated_code(
            code_content,
            {},  # No context needed for standalone testing
            self.ollama_provider
        )
        
        # Show results
        if test_result.get('tested'):
            if test_result.get('success'):
                print(f"✅ Build successful!")
                print(f"   File: {target_file}")
            else:
                print(f"❌ Build failed")
                
                # Show build details if available
                build_details = test_result.get('build_details', [])
                if build_details:
                    last_build = build_details[-1]['build_result']
                    if last_build.stderr:
                        print(f"   Error: {last_build.stderr[:200]}...")
                    
                    if last_build.error_analysis:
                        analysis = last_build.error_analysis
                        print(f"   {analysis.get('analysis_summary', 'No error summary')}")
            
            iterations = test_result.get('iterations', 0)
            if iterations > 1:
                print(f"   Applied {iterations} improvement iterations")
        else:
            reason = test_result.get('reason', 'Unknown')
            print(f"⚠️  Build testing not available: {reason}")
    
    async def _handle_cicd_command(self, command: str):
        """Handle CI/CD commands"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            print("Usage: cicd <setup|generate|update> [options]")
            return
        
        subcommand = parts[1]
        
        try:
            if subcommand == "setup":
                print("🚀 Setting up CI/CD pipelines...")
                pipelines = configure_cicd_interactive(str(self.project_state.working_dir))
                
                if pipelines:
                    print("\n✅ CI/CD setup completed!")
                    for name, path in pipelines.items():
                        print(f"  📄 {name}: {path}")
                else:
                    print("❌ CI/CD setup failed or was cancelled")
            
            elif subcommand == "generate":
                # Parse additional arguments
                language = None
                provider = None
                
                if len(parts) > 2:
                    args = parts[2].split()
                    if len(args) >= 1:
                        language = args[0]
                    if len(args) >= 2:
                        provider = args[1]
                
                print("🔧 Generating CI/CD pipelines...")
                
                # Auto-detect if not provided
                if not language:
                    detected_lang = self.cicd_integration.detect_project_language()
                    if detected_lang:
                        language = detected_lang.value
                        print(f"Detected language: {language}")
                    else:
                        print("❌ Could not detect project language. Please specify: cicd generate <language> [provider]")
                        print(f"Supported languages: {', '.join(self.cicd_integration.list_supported_languages())}")
                        return
                
                if not provider:
                    detected_provider = self.cicd_integration.detect_git_provider()
                    if detected_provider:
                        provider = detected_provider.value
                        print(f"Detected provider: {provider}")
                    else:
                        provider = "github_actions"
                        print(f"Using default provider: {provider}")
                
                pipelines = self.cicd_integration.generate_pipelines(language, provider)
                
                print(f"\n✅ Generated CI/CD pipelines for {language} on {provider}!")
                for name, path in pipelines.items():
                    print(f"  📄 {name}: {path}")
                    
                    # Validate pipeline
                    validation = self.cicd_integration.validate_pipeline_config(path)
                    if validation["valid"]:
                        print(f"    ✅ Valid configuration")
                    else:
                        print(f"    ⚠️  Validation warnings:")
                        for warning in validation["warnings"]:
                            print(f"      - {warning}")
                        for error in validation["errors"]:
                            print(f"      ❌ {error}")
            
            elif subcommand == "update":
                print("🔄 Updating CI/CD pipelines...")
                
                provider = None
                if len(parts) > 2:
                    provider = parts[2]
                
                pipelines = self.cicd_integration.update_pipeline(provider)
                
                print("\n✅ CI/CD pipelines updated!")
                for name, path in pipelines.items():
                    print(f"  📄 {name}: {path}")
            
            elif subcommand == "providers":
                print("📋 Supported CI/CD providers:")
                for provider in self.cicd_integration.list_supported_providers():
                    features = self.cicd_integration.get_provider_features(provider)
                    print(f"  • {provider}")
                    if features:
                        enabled_features = [k for k, v in features.items() if v]
                        print(f"    Features: {', '.join(enabled_features)}")
            
            elif subcommand == "languages":
                print("📋 Supported languages:")
                for language in self.cicd_integration.list_supported_languages():
                    print(f"  • {language}")
            
            else:
                print(f"Unknown CI/CD subcommand: {subcommand}")
                print("Available subcommands: setup, generate, update, providers, languages")
        
        except Exception as e:
            print(f"❌ CI/CD command failed: {e}")
    
    async def _handle_benchmark_command(self, command: str):
        """Handle benchmarking commands"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            print("Usage: benchmark <run|suite|file|report> [options]")
            return
        
        subcommand = parts[1]
        
        try:
            if subcommand == "run":
                print("🏃 Starting interactive benchmarking session...")
                await run_benchmark_interactive(str(self.project_state.working_dir))
            
            elif subcommand == "suite":
                if len(parts) < 3:
                    # List available suites
                    suites = self.benchmark_integration.list_available_suites()
                    print("📋 Available benchmark suites:")
                    for key, info in suites.items():
                        print(f"  • {key}: {info['name']} ({info['type']})")
                        print(f"    {info['description']}")
                    return
                
                suite_name = parts[2]
                print(f"🏃 Running benchmark suite: {suite_name}")
                
                # Get optional parameters
                iterations = 3
                try:
                    if "iterations=" in command:
                        iterations = int(command.split("iterations=")[1].split()[0])
                except:
                    pass
                
                results = await self.benchmark_integration.run_suite(
                    suite_name, iterations=iterations
                )
                
                print(f"\n✅ Suite completed! {len(results)} benchmarks run.")
                for test_name, result in results.items():
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {test_name}")
            
            elif subcommand == "file":
                if len(parts) < 3:
                    print("Usage: benchmark file <file_path> [iterations]")
                    return
                
                file_path = parts[2]
                iterations = 3
                
                # Check for iterations parameter
                if len(command.split()) > 3:
                    try:
                        iterations = int(command.split()[3])
                    except ValueError:
                        print("Invalid iterations value, using default (3)")
                
                print(f"🔨 Benchmarking file: {file_path}")
                
                result = await self.benchmark_integration.benchmark_specific_file(
                    file_path=file_path,
                    iterations=iterations
                )
                
                if result.success:
                    print(f"✅ File benchmark completed!")
                    summary = result.summary
                    if 'duration' in summary:
                        duration = summary['duration']
                        print(f"   Duration: {duration['mean']:.3f}s ± {duration['std_dev']:.3f}s")
                    if 'memory_usage' in summary:
                        memory = summary['memory_usage']
                        print(f"   Memory: {memory['mean']:.1f}MB")
                else:
                    print(f"❌ File benchmark failed: {result.error_details}")
            
            elif subcommand == "report":
                days = 30
                
                # Check for days parameter
                if len(parts) > 2:
                    try:
                        days = int(parts[2])
                    except ValueError:
                        print("Invalid days value, using default (30)")
                
                print(f"📊 Generating performance report for last {days} days...")
                
                report = self.benchmark_integration.get_performance_report(days)
                
                if "error" in report:
                    print(f"❌ {report['error']}")
                else:
                    print(f"\n📊 Performance Report ({report['period']})")
                    print(f"Total benchmarks: {report['total_benchmarks']}")
                    
                    # Show test types
                    if report['test_types']:
                        print("\n🔍 Test Performance:")
                        for test_name, data in report['test_types'].items():
                            print(f"  • {test_name}: {data['run_count']} runs")
                    
                    # Show recommendations
                    print("\n💡 Recommendations:")
                    for rec in report['recommendations']:
                        print(f"  {rec}")
            
            elif subcommand == "suites":
                suites = self.benchmark_integration.list_available_suites()
                print("📋 Available benchmark suites:")
                for key, info in suites.items():
                    print(f"  • {key}: {info['name']} ({info['type']})")
                    print(f"    {info['description']}")
                    if info['tags']:
                        print(f"    Tags: {', '.join(info['tags'])}")
            
            else:
                print(f"Unknown benchmark subcommand: {subcommand}")
                print("Available subcommands: run, suite, file, report, suites")
        
        except Exception as e:
            print(f"❌ Benchmark command failed: {e}")
    
    async def _handle_security_command(self, command: str):
        """Handle security commands"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            print("Usage: security <scan|quick|gate|dashboard|policy> [options]")
            return
        
        subcommand = parts[1]
        
        try:
            if subcommand == "run":
                print("🔒 Starting interactive security analysis...")
                await run_security_scan_interactive(str(self.project_state.working_dir))
            
            elif subcommand == "scan":
                print("🔍 Running comprehensive security scan...")
                result = await self.security_integration.scan_project_security(
                    include_external=True, 
                    policy_check=True
                )
                
                if result["success"]:
                    summary = result["summary"]
                    print(f"\n✅ Security scan completed!")
                    print(f"   Vulnerabilities: {summary['total_vulnerabilities']}")
                    print(f"   Risk Score: {summary['risk_score']}")
                    print(f"   Policy Compliant: {summary['policy_compliant']}")
                    print(f"   Files Scanned: {summary['files_scanned']}")
                    
                    # Show policy violations if any
                    policy_result = result.get("policy_result")
                    if policy_result and not policy_result["compliant"]:
                        print("\n⚠️  Policy violations:")
                        for violation in policy_result["violations"]:
                            print(f"     - {violation['message']}")
                    
                    # Show recommendations
                    print("\n💡 Recommendations:")
                    for rec in result["recommendations"]:
                        print(f"   {rec}")
                else:
                    print(f"❌ Security scan failed: {result['error']}")
            
            elif subcommand == "quick":
                print("🔍 Running quick security scan...")
                result = await self.security_integration.scan_project_security(
                    include_external=False, 
                    policy_check=True
                )
                
                if result["success"]:
                    summary = result["summary"]
                    print(f"\n✅ Quick scan completed!")
                    print(f"   Vulnerabilities: {summary['total_vulnerabilities']}")
                    print(f"   Risk Score: {summary['risk_score']}")
                    print(f"   Files Scanned: {summary['files_scanned']}")
                else:
                    print(f"❌ Quick scan failed: {result['error']}")
            
            elif subcommand == "gate":
                if len(parts) < 3:
                    # List available gates
                    gates = self.security_integration.list_security_gates()
                    print("🚪 Available security gates:")
                    for gate_name, gate_info in gates.items():
                        status = "enabled" if gate_info["enabled"] else "disabled"
                        print(f"  • {gate_name}: {status}")
                        print(f"    Policy: fail_critical={gate_info['policy']['fail_on_critical']}, "
                              f"fail_high={gate_info['policy']['fail_on_high']}, "
                              f"max_risk={gate_info['policy']['max_risk_score']}")
                    return
                
                gate_name = parts[2]
                print(f"🚪 Running security gate: {gate_name}")
                
                result = await self.security_integration.run_security_gate(gate_name)
                
                if result.get("skipped"):
                    print(f"⏭️  {result['message']}")
                elif result["success"]:
                    print(f"✅ Security gate '{gate_name}' passed!")
                else:
                    print(f"❌ Security gate '{gate_name}' failed!")
                    for violation in result.get("violations", []):
                        print(f"   - {violation['message']}")
                    
                    if result.get("auto_fix_attempted"):
                        auto_fix = result["auto_fix_result"]
                        print(f"🔧 Auto-fix attempted: {auto_fix['fixes_applied']} applied, "
                              f"{auto_fix['fixes_failed']} failed")
            
            elif subcommand == "dashboard":
                days = 30
                
                # Check for days parameter
                if len(parts) > 2:
                    try:
                        days = int(parts[2])
                    except ValueError:
                        print("Invalid days value, using default (30)")
                
                print(f"📊 Generating security dashboard for last {days} days...")
                
                dashboard = self.security_integration.get_security_dashboard(days)
                
                print(f"\n📊 Security Dashboard ({dashboard['period']})")
                print(f"   Total Scans: {dashboard['total_scans']}")
                print(f"   Total Vulnerabilities: {dashboard['total_vulnerabilities']}")
                print(f"   Current Posture: {dashboard['current_posture']}")
                
                # Show security gates
                print("\n🚪 Security Gates:")
                for gate_name, gate_info in dashboard["security_gates"].items():
                    status = "✅" if gate_info["enabled"] else "❌"
                    print(f"   {status} {gate_name}")
                
                # Show risk trend
                if dashboard["risk_trend"]:
                    print("\n📈 Risk Trend (last 5 scans):")
                    for trend in dashboard["risk_trend"][-5:]:
                        print(f"   {trend['date']}: Risk {trend['risk_score']}, "
                              f"Vulns {trend['vulnerability_count']}")
            
            elif subcommand == "policy":
                policy = self.security_integration.policy
                print("⚙️  Current Security Policy:")
                print(f"   Fail on Critical: {policy.fail_on_critical}")
                print(f"   Fail on High: {policy.fail_on_high}")
                print(f"   Max Risk Score: {policy.max_risk_score}")
                print(f"   Excluded Categories: {', '.join(policy.excluded_categories) or 'None'}")
                print(f"   Excluded Files: {', '.join(policy.excluded_files) or 'None'}")
                
                # Allow modification
                modify = input("\nModify policy? [y/N]: ").strip().lower()
                if modify == 'y':
                    fail_critical = input(f"Fail on critical vulnerabilities? [{policy.fail_on_critical}]: ").strip()
                    if fail_critical.lower() in ['true', 'yes', 'y']:
                        policy.fail_on_critical = True
                    elif fail_critical.lower() in ['false', 'no', 'n']:
                        policy.fail_on_critical = False
                    
                    fail_high = input(f"Fail on high severity vulnerabilities? [{policy.fail_on_high}]: ").strip()
                    if fail_high.lower() in ['true', 'yes', 'y']:
                        policy.fail_on_high = True
                    elif fail_high.lower() in ['false', 'no', 'n']:
                        policy.fail_on_high = False
                    
                    max_risk = input(f"Maximum risk score [{policy.max_risk_score}]: ").strip()
                    if max_risk and max_risk.isdigit():
                        policy.max_risk_score = int(max_risk)
                    
                    self.security_integration.save_security_config()
                    print("✅ Security policy updated!")
            
            elif subcommand == "gates":
                gates = self.security_integration.list_security_gates()
                print("🚪 Security Gates Configuration:")
                for gate_name, gate_info in gates.items():
                    status = "enabled" if gate_info["enabled"] else "disabled"
                    print(f"\n  • {gate_name} ({status})")
                    print(f"    Auto-fix: {gate_info['auto_fix']}")
                    print(f"    Notify on failure: {gate_info['notify_on_failure']}")
                    policy = gate_info['policy']
                    print(f"    Policy: critical={policy['fail_on_critical']}, "
                          f"high={policy['fail_on_high']}, "
                          f"max_risk={policy['max_risk_score']}")
            
            else:
                print(f"Unknown security subcommand: {subcommand}")
                print("Available subcommands: run, scan, quick, gate, dashboard, policy, gates")
        
        except Exception as e:
            print(f"❌ Security command failed: {e}")
    
    async def _handle_cost_command(self, command: str):
        """Handle cost analysis commands"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            print("Usage: cost <analyze|estimate|dashboard|budget|optimize> [options]")
            return
        
        subcommand = parts[1]
        
        try:
            if subcommand == "run":
                print("💰 Starting interactive cost analysis...")
                await run_cost_analysis_interactive(str(self.project_state.working_dir))
            
            elif subcommand == "analyze":
                print("💰 Running comprehensive cost analysis...")
                result = await self.cost_integration.analyze_project_costs(
                    include_predictions=True,
                    optimization_focus="all"
                )
                
                if result["success"]:
                    summary = result["summary"]
                    print(f"\n✅ Cost analysis completed!")
                    print(f"   Current Monthly Cost: ${summary['current_monthly_cost']:.2f}")
                    print(f"   Optimization Potential: ${summary['optimization_potential']:.2f}")
                    print(f"   Cost Efficiency Score: {summary['cost_efficiency_score']}/100")
                    
                    # Show budget status
                    if summary.get("budget_utilization"):
                        print(f"   Budget Utilization: {summary['budget_utilization']:.1f}%")
                    
                    # Show predictions
                    predictions = result.get("predictions")
                    if predictions:
                        print(f"\n📈 Cost Predictions:")
                        print(f"   Next Month: ${predictions['next_month']:.2f}")
                        print(f"   Next Quarter: ${predictions['next_quarter']:.2f}")
                        print(f"   Next Year: ${predictions['next_year']:.2f}")
                    
                    # Show top recommendations
                    recommendations = result["filtered_recommendations"][:3]
                    if recommendations:
                        print(f"\n💡 Top Optimization Opportunities:")
                        for i, rec in enumerate(recommendations, 1):
                            print(f"   {i}. {rec.description} (${rec.savings_amount:.2f}/month)")
                    
                    # Show budget alerts
                    if result["budget_alerts"]:
                        print(f"\n⚠️  Budget Alerts:")
                        for alert in result["budget_alerts"]:
                            print(f"   - {alert.message}")
                else:
                    print(f"❌ Cost analysis failed: {result['error']}")
            
            elif subcommand == "estimate":
                focus = "all"
                if len(parts) > 2:
                    focus = parts[2]
                
                print(f"⚡ Running quick cost estimate for {focus}...")
                result = await self.cost_integration.analyze_project_costs(
                    include_predictions=False,
                    optimization_focus=focus
                )
                
                if result["success"]:
                    summary = result["summary"]
                    print(f"\n✅ Cost estimate completed!")
                    print(f"   Estimated Monthly Cost: ${summary['current_monthly_cost']:.2f}")
                    print(f"   Potential Savings: ${summary['optimization_potential']:.2f}")
                    print(f"   Efficiency Score: {summary['cost_efficiency_score']}/100")
                else:
                    print(f"❌ Cost estimate failed: {result['error']}")
            
            elif subcommand == "dashboard":
                days = 30
                if len(parts) > 2:
                    try:
                        days = int(parts[2])
                    except ValueError:
                        print("Invalid days value, using default (30)")
                
                print(f"📊 Generating cost dashboard for last {days} days...")
                dashboard = self.cost_integration.get_cost_dashboard(days)
                
                print(f"\n📊 Cost Dashboard ({dashboard['period']})")
                print(f"   Current Monthly Cost: ${dashboard['current_monthly_cost']:.2f}")
                print(f"   Cost Efficiency Score: {dashboard['cost_efficiency_score']}/100")
                print(f"   Optimization Potential: ${dashboard['optimization_potential']:.2f}")
                
                # Show budget information
                if dashboard["budget_info"]:
                    budget = dashboard["budget_info"]
                    utilization = dashboard.get("budget_utilization", 0)
                    print(f"   Budget: ${dashboard['current_monthly_cost']:.2f} / ${budget['monthly_limit']:.2f} ({utilization:.1f}%)")
                
                # Show cost breakdown
                print("\n💸 Cost Breakdown:")
                for category, cost in dashboard["category_breakdown"].items():
                    percentage = (cost / dashboard['current_monthly_cost'] * 100) if dashboard['current_monthly_cost'] > 0 else 0
                    print(f"   {category.title()}: ${cost:.2f} ({percentage:.1f}%)")
                
                # Show trend analysis
                trend = dashboard.get("trend_analysis", {})
                if trend.get("status") != "no_data":
                    print(f"\n📈 Trend: {trend['interpretation']}")
                
                # Show quick wins
                if dashboard["quick_wins"]:
                    print("\n🎯 Quick Wins:")
                    for win in dashboard["quick_wins"]:
                        print(f"   • {win['title']} (${win['savings']:.2f}/month)")
                
                # Show recent alerts
                if dashboard["recent_alerts"]:
                    print(f"\n⚠️  Recent Alerts ({len(dashboard['recent_alerts'])}):")
                    for alert in dashboard["recent_alerts"][:3]:
                        print(f"   - {alert.message}")
            
            elif subcommand == "budget":
                if len(parts) < 3:
                    # Show current budget
                    budget = self.cost_integration.budget
                    if budget:
                        print("💰 Current Budget Configuration:")
                        print(f"   Monthly Limit: ${budget.monthly_limit:.2f}")
                        print(f"   Alert Thresholds: {budget.alert_thresholds}%")
                        if budget.category_limits:
                            print("   Category Limits:")
                            for category, limit in budget.category_limits.items():
                                print(f"     {category.title()}: ${limit:.2f}")
                    else:
                        print("💰 No budget configured")
                        print("Usage: cost budget <monthly_limit> [alert_thresholds]")
                    return
                
                try:
                    monthly_limit = float(parts[2])
                    
                    # Parse alert thresholds if provided
                    alert_thresholds = [50, 80, 95]  # Default
                    if len(command.split()) > 3:
                        thresholds_str = command.split()[3]
                        if "," in thresholds_str:
                            alert_thresholds = [float(x.strip()) for x in thresholds_str.split(",")]
                    
                    success = self.cost_integration.create_budget(monthly_limit, alert_thresholds)
                    if success:
                        print(f"✅ Budget set to ${monthly_limit:.2f}/month")
                        print(f"   Alert thresholds: {alert_thresholds}%")
                    else:
                        print("❌ Failed to create budget")
                
                except ValueError:
                    print("Invalid budget amount. Usage: cost budget <monthly_limit>")
            
            elif subcommand == "optimize":
                focus = "all"
                if len(parts) > 2:
                    focus = parts[2]
                
                print(f"🔍 Finding {focus} optimization opportunities...")
                result = await self.cost_integration.analyze_project_costs(
                    include_predictions=False,
                    optimization_focus=focus
                )
                
                if result["success"]:
                    recommendations = result["filtered_recommendations"]
                    roi_info = result["optimization_roi"]
                    
                    print(f"\n💡 Found {len(recommendations)} optimization opportunities")
                    print(f"   Total Monthly Savings: ${roi_info['total_monthly_savings']:.2f}")
                    print(f"   Annual Savings: ${roi_info['annual_savings']:.2f}")
                    print(f"   Implementation Cost: ${roi_info['implementation_cost']:.2f}")
                    print(f"   ROI: {roi_info['roi_percentage']:.1f}%")
                    print(f"   Payback Period: {roi_info['payback_months']:.1f} months")
                    
                    # Show recommendations by effort level
                    effort_groups = roi_info["recommendations_by_effort"]
                    for effort in ["low", "medium", "high"]:
                        group_recs = effort_groups.get(effort, [])
                        if group_recs:
                            print(f"\n🎯 {effort.title()} Effort Recommendations:")
                            for i, rec in enumerate(group_recs[:3], 1):
                                print(f"   {i}. {rec.description}")
                                print(f"      Savings: ${rec.savings_amount:.2f}/month")
                                print(f"      Risk: {rec.risk_level}")
                else:
                    print(f"❌ Optimization analysis failed: {result['error']}")
            
            elif subcommand == "report":
                days = 30
                if len(parts) > 2:
                    try:
                        days = int(parts[2])
                    except ValueError:
                        print("Invalid days value, using default (30)")
                
                print(f"📊 Generating cost report for last {days} days...")
                
                # Generate report using the analyzer
                report = self.cost_integration.analyzer.generate_cost_report(days)
                
                if "error" in report:
                    print(f"❌ {report['error']}")
                else:
                    print(f"\n📊 Cost Report ({report['period']})")
                    print(f"   Total Analyses: {report['total_analyses']}")
                    print(f"   Average Monthly Cost: ${report['average_monthly_cost']:.2f}")
                    print(f"   Total Optimization Potential: ${report['total_optimization_potential']:.2f}")
                    print(f"   Cost Efficiency Score: {report['cost_efficiency_score']}/100")
                    
                    # Show category breakdown
                    if report['category_breakdown']:
                        print("\n💸 Average Cost by Category:")
                        for category, cost in report['category_breakdown'].items():
                            print(f"   {category.title()}: ${cost:.2f}")
                    
                    # Show optimization opportunities
                    if report['optimization_opportunities']:
                        print("\n🔍 Optimization Opportunities:")
                        for opp_type, data in report['optimization_opportunities'].items():
                            print(f"   {opp_type.replace('_', ' ').title()}: {data['count']} opportunities, ${data['total_savings']:.2f} potential savings")
            
            else:
                print(f"Unknown cost subcommand: {subcommand}")
                print("Available subcommands: run, analyze, estimate, dashboard, budget, optimize, report")
        
        except Exception as e:
            print(f"❌ Cost command failed: {e}")
    
    async def _handle_team_command(self, command: str):
        """Handle team collaboration commands"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            print("Usage: team <dashboard|members|reviews|settings|activity> [options]")
            return
        
        subcommand = parts[1]
        
        try:
            if subcommand == "run":
                print("👥 Starting interactive team management...")
                await run_team_management_interactive(str(self.project_state.working_dir))
            
            elif subcommand == "dashboard":
                print("👥 Loading team dashboard...")
                dashboard = self.team_integration.get_team_dashboard()
                
                # Team info
                team_info = dashboard["team_info"]
                print(f"\n👥 Team: {team_info['team_name']}")
                print(f"   Members: {team_info['total_members']} ({team_info['active_members']} active)")
                print(f"   Owner: {team_info['owner']}")
                
                # Role breakdown
                print(f"\n🎭 Team Composition:")
                for role, count in team_info['member_count_by_role'].items():
                    print(f"   {role.title()}: {count}")
                
                # Review summary
                review_summary = dashboard["review_summary"]
                print(f"\n📋 Code Reviews:")
                print(f"   Total: {review_summary['total_reviews']}")
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
                        timestamp = activity['timestamp'][:16].replace('T', ' ')
                        print(f"   • {timestamp} - {activity['username']}: {activity['description']}")
            
            elif subcommand == "members":
                team_manager = self.team_integration.team_manager
                
                if len(parts) < 3:
                    # List members
                    print(f"\n👥 Team Members:")
                    for member in team_manager.team.members.values():
                        status = "🟢" if member.is_active else "🔴"
                        print(f"   {status} {member.username} ({member.role.value})")
                        print(f"      Email: {member.email}")
                        print(f"      Joined: {member.joined_at[:10]}")
                        print(f"      Last Active: {member.last_active[:10]}")
                    return
                
                action = parts[2]
                
                if action == "add":
                    username = input("Username: ").strip()
                    email = input("Email: ").strip()
                    role_input = input("Role [developer/reviewer/maintainer/admin]: ").strip().lower()
                    
                    from collaboration import UserRole
                    role_mapping = {
                        "developer": UserRole.DEVELOPER,
                        "reviewer": UserRole.REVIEWER,
                        "maintainer": UserRole.MAINTAINER,
                        "admin": UserRole.ADMIN
                    }
                    
                    role = role_mapping.get(role_input, UserRole.DEVELOPER)
                    success = team_manager.add_member(username, email, role)
                
                elif action == "remove":
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
                
                else:
                    print("Available actions: add, remove")
            
            elif subcommand == "reviews":
                team_manager = self.team_integration.team_manager
                
                if len(parts) < 3:
                    # Show review dashboard
                    dashboard = team_manager.get_review_dashboard()
                    
                    print(f"\n📋 Review Dashboard")
                    print(f"   Total Reviews: {dashboard['total_reviews']}")
                    
                    print(f"\n📊 By Status:")
                    for status, count in dashboard['reviews_by_status'].items():
                        print(f"   {status.title()}: {count}")
                    
                    if dashboard['pending_reviews']:
                        print(f"\n⏳ Pending Reviews:")
                        for review in dashboard['pending_reviews'][:5]:
                            print(f"   • {review['request_id']}: {review['title']}")
                            print(f"     Author: {review['author']}")
                            print(f"     Reviewers: {', '.join(review['reviewers'])}")
                            print(f"     Created: {review['created_at'][:10]}")
                    return
                
                action = parts[2]
                
                if action == "comment":
                    request_id = input("Review request ID: ").strip()
                    comment = input("Comment: ").strip()
                    
                    success = team_manager.add_review_comment(request_id, comment)
                
                elif action == "approve":
                    request_id = input("Review request ID: ").strip()
                    success = team_manager.approve_review_request(request_id)
                
                else:
                    print("Available actions: comment, approve")
            
            elif subcommand == "settings":
                settings = self.team_integration.settings
                
                if len(parts) < 3:
                    # Show current settings
                    print(f"\n⚙️  Team Settings:")
                    print(f"   Require Code Review: {settings.require_code_review}")
                    print(f"   Minimum Reviewers: {settings.min_reviewers}")
                    print(f"   Auto Assign Reviewers: {settings.auto_assign_reviewers}")
                    print(f"   Enable Security Gates: {settings.enable_security_gates}")
                    print(f"   Cost Alert Threshold: ${settings.cost_alert_threshold:.2f}")
                    print(f"   Workflow Rules: {len(settings.workflow_rules)}")
                    return
                
                setting_name = parts[2]
                
                if setting_name == "review":
                    require_review = input("Require code review? [true/false]: ").strip().lower()
                    if require_review in ['true', 'false']:
                        self.team_integration.configure_team_settings(
                            require_code_review=(require_review == 'true')
                        )
                
                elif setting_name == "reviewers":
                    min_reviewers = input("Minimum number of reviewers: ").strip()
                    if min_reviewers.isdigit():
                        self.team_integration.configure_team_settings(
                            min_reviewers=int(min_reviewers)
                        )
                
                elif setting_name == "cost_threshold":
                    threshold = input("Cost alert threshold ($): ").strip()
                    if threshold.replace('.', '').isdigit():
                        self.team_integration.configure_team_settings(
                            cost_alert_threshold=float(threshold)
                        )
                
                else:
                    print("Available settings: review, reviewers, cost_threshold")
            
            elif subcommand == "activity":
                team_manager = self.team_integration.team_manager
                
                if len(parts) < 3:
                    # Show team activity summary
                    dashboard = self.team_integration.get_team_dashboard()
                    metrics = dashboard["productivity_metrics"]
                    
                    print(f"\n📈 Team Activity Summary (last 30 days):")
                    print(f"   Code Generations: {metrics['code_generations_last_30_days']}")
                    print(f"   Reviews Completed: {metrics['reviews_completed_last_30_days']}")
                    print(f"   Average Review Time: {metrics['average_review_time_hours']:.1f} hours")
                    print(f"   Team Productivity Score: {metrics['productivity_score']}/100")
                    
                    # Recent activities
                    if dashboard["recent_activities"]:
                        print(f"\n🕐 Recent Team Activities:")
                        for activity in dashboard["recent_activities"][-10:]:
                            timestamp = activity['timestamp'][:16].replace('T', ' ')
                            print(f"   • {timestamp} - {activity['username']}: {activity['description']}")
                    return
                
                username = parts[2]
                
                # Find user by username
                user_id = None
                for uid, member in team_manager.team.members.items():
                    if member.username == username:
                        user_id = uid
                        break
                
                if user_id:
                    activity_report = team_manager.get_member_activity(user_id, 30)
                    
                    print(f"\n👤 Activity Report for {username}:")
                    member_info = activity_report["member_info"]
                    print(f"   Role: {member_info['role']}")
                    print(f"   Joined: {member_info['joined_at'][:10]}")
                    print(f"   Last Active: {member_info['last_active'][:10]}")
                    
                    summary = activity_report["activity_summary"]
                    print(f"\n📊 Activity Summary (last 30 days):")
                    print(f"   Total Activities: {summary['total_activities']}")
                    print(f"   Authored Reviews: {summary['authored_reviews']}")
                    print(f"   Reviewed Requests: {summary['reviewed_requests']}")
                    
                    if summary["activity_counts"]:
                        print(f"\n🎯 Activity Breakdown:")
                        for activity_type, count in summary["activity_counts"].items():
                            print(f"   {activity_type.replace('_', ' ').title()}: {count}")
                    
                    if activity_report["recent_activities"]:
                        print(f"\n🕐 Recent Activities:")
                        for activity in activity_report["recent_activities"]:
                            timestamp = activity['timestamp'][:16].replace('T', ' ')
                            print(f"   • {timestamp}: {activity['description']}")
                else:
                    print("❌ User not found")
            
            else:
                print(f"Unknown team subcommand: {subcommand}")
                print("Available subcommands: run, dashboard, members, reviews, settings, activity")
        
        except Exception as e:
            print(f"❌ Team command failed: {e}")

    async def _handle_enterprise_command(self, command: str):
        """Handle enterprise integration commands"""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: enterprise <subcommand>")
                print("Available subcommands: setup, status, auth, project")
                return
                
            subcommand = parts[1]
            
            if subcommand == "setup":
                print("🏢 Starting enterprise integration setup...")
                self.enterprise_manager = await run_enterprise_setup_interactive()
                print("✅ Enterprise integration setup completed")
                
            elif subcommand == "status":
                if not self.enterprise_manager:
                    print("❌ Enterprise integration not configured. Run 'enterprise setup' first.")
                    return
                    
                print("🏢 Enterprise Integration Status")
                print("=" * 40)
                
                status = self.enterprise_manager.get_integration_status()
                
                for integration_type, info in status.items():
                    status_icon = "✅" if info["initialized"] else "❌"
                    config_icon = "🔧" if info["configured"] else "⚠️"
                    
                    print(f"{status_icon} {integration_type.replace('_', ' ').title()}")
                    print(f"   Configured: {config_icon}")
                    print(f"   Initialized: {info['initialized']}")
                    
                    if info["manager"]:
                        for key, value in info["manager"].items():
                            if key != "status":
                                print(f"   {key.replace('_', ' ').title()}: {value}")
                    print()
                    
            elif subcommand == "auth":
                if not self.enterprise_manager or not self.enterprise_manager.auth_manager:
                    print("❌ Authentication not configured. Run 'enterprise setup' first.")
                    return
                    
                print("🔐 Testing enterprise authentication...")
                
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                
                try:
                    user_profile = await self.enterprise_manager.authenticate_user(username, password)
                    print(f"✅ Authentication successful!")
                    print(f"   User: {user_profile.full_name} ({user_profile.email})")
                    print(f"   Groups: {', '.join(user_profile.groups)}")
                    print(f"   Roles: {', '.join(user_profile.roles)}")
                    
                except Exception as e:
                    print(f"❌ Authentication failed: {e}")
                    
            elif subcommand == "project":
                if not self.enterprise_manager:
                    print("❌ Enterprise integration not configured. Run 'enterprise setup' first.")
                    return
                    
                if len(parts) < 3:
                    print("Usage: enterprise project <project_name>")
                    return
                    
                project_name = " ".join(parts[2:])
                
                print(f"🚀 Creating enterprise migration project: {project_name}")
                
                # Get project details interactively
                source_lang = input("Source language: ").strip() or "mixed"
                target_lang = input("Target language: ").strip() or "pomegranate"
                
                # Get file paths from current directory
                file_paths = []
                current_dir = Path.cwd()
                for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs']:
                    file_paths.extend([str(p) for p in current_dir.rglob(f"*{ext}")[:5]])  # Limit files
                
                if not file_paths:
                    file_paths = ["example.py", "example.js"]  # Placeholder
                    
                print(f"📁 Including {len(file_paths)} files in migration project")
                
                try:
                    results = await self.enterprise_manager.create_migration_project(
                        project_name, source_lang, target_lang, file_paths
                    )
                    
                    print("✅ Migration project created successfully!")
                    
                    if results.get("work_items_generated"):
                        print(f"   📋 Created {results['work_items_generated']} work items")
                        
                    if results.get("notification_sent"):
                        print("   📢 Team notifications sent")
                        
                    if results.get("metrics_recorded"):
                        print("   📊 Metrics recorded")
                        
                    if results.get("project_file"):
                        print(f"   💾 Project saved to {results['project_file']}")
                        
                    # Display any errors
                    for key, value in results.items():
                        if key.endswith("_error"):
                            print(f"   ⚠️  {key.replace('_', ' ').title()}: {value}")
                            
                except Exception as e:
                    print(f"❌ Failed to create migration project: {e}")
                    
            else:
                print(f"Unknown enterprise subcommand: {subcommand}")
                print("Available subcommands: setup, status, auth, project")
                
        except Exception as e:
            print(f"❌ Enterprise command failed: {e}")

    async def _handle_test_command(self, command: str):
        """Handle universal testing commands"""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: test <subcommand>")
                print("Available subcommands: universal, matrix, coverage, compatibility")
                return
                
            subcommand = parts[1]
            
            if subcommand == "universal":
                await self._handle_universal_test(parts[2:])
                
            elif subcommand == "matrix":
                await self._handle_matrix_test(parts[2:])
                
            elif subcommand == "coverage":
                await self._handle_coverage_test(parts[2:])
                
            elif subcommand == "compatibility":
                await self._handle_compatibility_test(parts[2:])
                
            else:
                print(f"Unknown test subcommand: {subcommand}")
                print("Available subcommands: universal, matrix, coverage, compatibility")
                
        except Exception as e:
            print(f"❌ Test command failed: {e}")
    
    async def _handle_universal_test(self, args: List[str]):
        """Handle universal build testing"""
        from universal_testing import BuildEnvironment, TestType, TestSuite
        
        # Parse languages from arguments
        languages = args if args else ["pomegranate"]
        
        print(f"🧪 Running universal build testing for: {', '.join(languages)}")
        
        project_path = Path.cwd()
        results = []
        
        for language in languages:
            print(f"\n🔨 Testing {language}...")
            
            # Create test environment
            environment = BuildEnvironment()
            
            # Create test suites
            test_suites = [
                TestSuite(
                    name="unit_tests",
                    language=language,
                    test_files=[],
                    test_type=TestType.UNIT,
                    timeout=300
                ),
                TestSuite(
                    name="integration_tests", 
                    language=language,
                    test_files=[],
                    test_type=TestType.INTEGRATION,
                    timeout=600
                )
            ]
            
            # Run build and tests
            result = await self.universal_builder.build_project(
                project_path, language, environment, test_suites
            )
            
            results.append(result)
            
            # Display result
            status_icon = "✅" if result.status.value == "success" else "❌"
            print(f"{status_icon} {language}: {result.status.value} ({result.duration:.1f}s)")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
                
            if result.test_results:
                passed_tests = len([t for t in result.test_results if t.status.value == "success"])
                total_tests = len(result.test_results)
                print(f"   Tests: {passed_tests}/{total_tests} passed")
        
        # Generate summary report
        print(f"\n📊 Universal Test Summary")
        print("=" * 40)
        
        total_builds = len(results)
        successful_builds = len([r for r in results if r.status.value == "success"])
        
        print(f"Builds: {successful_builds}/{total_builds} successful")
        print(f"Total Duration: {sum(r.duration for r in results):.1f}s")
        
        # Save report
        report = await self.universal_builder.generate_build_report(results)
        print(f"\n💾 Report saved to {self.universal_builder.cache_dir}")
    
    async def _handle_matrix_test(self, args: List[str]):
        """Handle build matrix testing"""
        from universal_testing import MatrixStrategy, MatrixConfiguration, MatrixDimension
        
        print("🔬 Running build matrix testing...")
        
        # Parse strategy from arguments
        strategy = MatrixStrategy.MINIMAL
        languages = ["pomegranate", "rust", "go"]
        
        for arg in args:
            if arg == "--full":
                strategy = MatrixStrategy.FULL
            elif arg == "--minimal":
                strategy = MatrixStrategy.MINIMAL
            elif arg == "--targeted":
                strategy = MatrixStrategy.TARGETED
            elif not arg.startswith("--"):
                # Treat as language
                if not languages or languages == ["pomegranate", "rust", "go"]:
                    languages = [arg]
                else:
                    languages.append(arg)
        
        print(f"Strategy: {strategy.value}")
        print(f"Languages: {', '.join(languages)}")
        
        # Create matrix configuration
        config = MatrixConfiguration(
            languages=MatrixDimension("language", languages),
            platforms=MatrixDimension("platform", ["windows", "linux", "darwin"]),
            architectures=MatrixDimension("architecture", ["x86_64"]),
            strategy=strategy,
            max_parallel_jobs=2
        )
        
        # Execute matrix
        project_path = Path.cwd()
        result = await self.build_matrix.execute_matrix(project_path, config)
        
        # Display results
        print(f"\n📊 Build Matrix Results")
        print("=" * 40)
        print(f"Total Combinations: {result.total_combinations}")
        print(f"Executed: {result.executed_combinations}")
        print(f"Successful: {result.successful_combinations}")
        
        success_rate = (result.successful_combinations / result.executed_combinations * 100) if result.executed_combinations > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Show results by language
        print(f"\n📋 Results by Language:")
        for language in languages:
            lang_results = [r for r in result.build_results if r.language == language]
            if lang_results:
                successful = len([r for r in lang_results if r.status.value == "success"])
                total = len(lang_results)
                print(f"  {language}: {successful}/{total} successful")
        
        # Show compatibility issues
        incompatible = [c for c in result.compatibility_results if c.level.value == "incompatible"]
        if incompatible:
            print(f"\n⚠️  Compatibility Issues:")
            for comp in incompatible[:5]:  # Show first 5
                combo = comp.combination
                print(f"  {combo['language']} on {combo['platform']}: {', '.join(comp.issues[:2])}")
        
        print(f"\n💾 Matrix results saved to {self.build_matrix.cache_dir}")
    
    async def _handle_coverage_test(self, args: List[str]):
        """Handle coverage testing"""
        from universal_testing import TestConfiguration, TestType
        
        languages = args if args else ["pomegranate"]
        
        print(f"📈 Running coverage analysis for: {', '.join(languages)}")
        
        # Create test configuration with coverage enabled
        config = TestConfiguration(
            languages=languages,
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            coverage_enabled=True,
            performance_profiling=False,
            generate_reports=True
        )
        
        # Run comprehensive tests
        project_path = Path.cwd()
        result = await self.test_orchestrator.run_comprehensive_tests(project_path, config)
        
        # Display coverage results
        print(f"\n📊 Coverage Results")
        print("=" * 40)
        
        if result.coverage_reports:
            for coverage in result.coverage_reports:
                print(f"{coverage.language}:")
                print(f"  Line Coverage: {coverage.line_coverage:.1f}%")
                print(f"  Lines: {coverage.covered_lines}/{coverage.total_lines}")
                
                if coverage.branch_coverage > 0:
                    print(f"  Branch Coverage: {coverage.branch_coverage:.1f}%")
                    print(f"  Branches: {coverage.covered_branches}/{coverage.total_branches}")
        else:
            print("No coverage data available")
            print("Note: Coverage tools may need to be installed for your languages")
        
        # Show overall summary
        if result.summary:
            print(f"\nOverall Coverage: {result.summary.get('overall_coverage', 0):.1f}%")
            print(f"Test Success Rate: {result.summary.get('test_success_rate', 0):.1f}%")
        
        print(f"\n💾 Coverage reports saved to {self.test_orchestrator.cache_dir}/reports")
    
    async def _handle_compatibility_test(self, args: List[str]):
        """Handle compatibility testing"""
        from universal_testing import CrossPlatformTester, CompatibilityChecker
        
        languages = args if args else ["pomegranate", "rust", "go"]
        
        print(f"🌐 Running cross-platform compatibility testing...")
        print(f"Languages: {', '.join(languages)}")
        
        # Run cross-platform testing
        tester = CrossPlatformTester()
        project_path = Path.cwd()
        
        result = await tester.test_cross_platform_compatibility(project_path, languages)
        
        # Display compatibility results
        print(f"\n🔍 Compatibility Results")
        print("=" * 40)
        
        # Group by compatibility level
        compatibility_summary = {}
        for comp_result in result.compatibility_results:
            level = comp_result.level.value
            if level not in compatibility_summary:
                compatibility_summary[level] = []
            compatibility_summary[level].append(comp_result)
        
        for level, results in compatibility_summary.items():
            icon = "✅" if level == "compatible" else "⚠️" if level == "partially_compatible" else "❌"
            print(f"{icon} {level.replace('_', ' ').title()}: {len(results)} combinations")
            
            # Show examples
            for result_item in results[:3]:  # Show first 3
                combo = result_item.combination
                print(f"   {combo['language']} on {combo['platform']}-{combo['architecture']}")
        
        # Show recommendations
        if result.summary and result.summary.get("recommendations"):
            print(f"\n💡 Recommendations:")
            for rec in result.summary["recommendations"][:5]:
                print(f"  • {rec}")
        
        print(f"\n💾 Compatibility results saved to {tester.build_matrix.cache_dir}")

    async def _handle_migrate_command(self, command: str):
        """Handle migration strategy commands"""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: migrate <subcommand>")
                print("Available subcommands: plan, execute, status, analyze")
                return
                
            subcommand = parts[1]
            
            if subcommand == "plan":
                await self._handle_migration_plan(parts[2:])
                
            elif subcommand == "execute":
                await self._handle_migration_execute(parts[2:])
                
            elif subcommand == "status":
                await self._handle_migration_status(parts[2:])
                
            elif subcommand == "analyze":
                await self._handle_migration_analyze(parts[2:])
                
            else:
                print(f"Unknown migration subcommand: {subcommand}")
                print("Available subcommands: plan, execute, status, analyze")
                
        except Exception as e:
            print(f"❌ Migration command failed: {e}")
    
    async def _handle_migration_plan(self, args: List[str]):
        """Handle migration planning"""
        if len(args) < 2:
            print("Usage: migrate plan <source_language> <target_language> [strategy]")
            print("Strategies: big_bang, strangler_fig, incremental, parallel_run")
            return
        
        source_language = args[0]
        target_language = args[1]
        strategy = None
        
        if len(args) > 2:
            strategy_map = {
                "big_bang": MigrationStrategy.BIG_BANG,
                "strangler_fig": MigrationStrategy.STRANGLER_FIG,
                "incremental": MigrationStrategy.INCREMENTAL,
                "parallel_run": MigrationStrategy.PARALLEL_RUN
            }
            strategy = strategy_map.get(args[2])
        
        print(f"🗺️  Creating migration plan: {source_language} → {target_language}")
        if strategy:
            print(f"Strategy: {strategy.value}")
        
        project_path = Path.cwd()
        
        try:
            # Create migration plan
            plan = await self.migration_planner.create_migration_plan(
                project_path, source_language, target_language, strategy
            )
            
            print(f"\n📋 Migration Plan Created")
            print("=" * 40)
            print(f"Project: {plan.project_name}")
            print(f"Strategy: {plan.strategy.value}")
            print(f"Components: {len(plan.component_analysis)}")
            
            # Show timeline
            if plan.timeline:
                print(f"\n⏰ Timeline:")
                for phase, date in list(plan.timeline.items())[:5]:  # Show first 5 phases
                    print(f"  {phase.replace('_', ' ').title()}: {date.strftime('%Y-%m-%d')}")
            
            # Show resource requirements
            if plan.resource_requirements:
                resources = plan.resource_requirements
                print(f"\n💼 Resource Requirements:")
                print(f"  Developers: {resources.get('developers_needed', 'N/A')}")
                print(f"  Duration: {resources.get('estimated_duration_months', 'N/A')} months")
                print(f"  Effort: {resources.get('total_effort_hours', 'N/A')} hours")
            
            # Show high-risk components
            high_risk_assessments = [r for r in plan.risk_assessments 
                                   if r.risk_level.value in ['high', 'critical']]
            if high_risk_assessments:
                print(f"\n⚠️  High-Risk Components ({len(high_risk_assessments)}):")
                for risk in high_risk_assessments[:3]:  # Show top 3
                    print(f"  • {risk.component}: {risk.risk_level.value} risk")
                    if risk.risk_factors:
                        print(f"    {risk.risk_factors[0]}")
            
            # Show success criteria
            if plan.success_criteria:
                print(f"\n🎯 Success Criteria:")
                for criteria in plan.success_criteria[:3]:  # Show first 3
                    print(f"  • {criteria}")
            
            print(f"\n💾 Migration plan saved to {self.migration_planner.cache_dir}")
            
        except Exception as e:
            print(f"❌ Failed to create migration plan: {e}")
    
    async def _handle_migration_execute(self, args: List[str]):
        """Handle migration execution"""
        print("🚀 Migration execution would begin here")
        print("Note: This would execute the migration plan step by step")
        print("Including code translation, testing, and validation")
        
        # For demo purposes, show what would happen
        project_path = Path.cwd()
        
        print(f"\n📁 Project: {project_path.name}")
        print("🔄 Execution steps would include:")
        print("  1. Component dependency analysis")
        print("  2. Code translation using ML models")
        print("  3. Test migration and validation")
        print("  4. Build system setup")
        print("  5. Performance benchmarking")
        print("  6. Security validation")
        print("  7. Deployment preparation")
        
        print(f"\n⏳ Estimated execution time: Variable based on project size")
        print("Use 'migrate status' to track progress during execution")
    
    async def _handle_migration_status(self, args: List[str]):
        """Handle migration status dashboard"""
        project_name = args[0] if args else Path.cwd().name
        
        print(f"📊 Migration Status Dashboard")
        print("=" * 50)
        
        try:
            # Display text-based dashboard
            self.status_dashboard.display_text_dashboard(project_name)
            
        except Exception as e:
            print(f"❌ No migration data available for {project_name}")
            print("Run 'migrate plan' first to create a migration plan")
            print(f"Error: {e}")
    
    async def _handle_migration_analyze(self, args: List[str]):
        """Handle legacy codebase analysis"""
        language = args[0] if args else "python"
        project_path = Path.cwd()
        
        print(f"🔍 Analyzing legacy codebase: {project_path.name}")
        print(f"Language: {language}")
        
        try:
            # Analyze legacy codebase
            components = await self.legacy_analyzer.analyze_legacy_codebase(project_path, language)
            
            if not components:
                print("❌ No source files found for analysis")
                return
            
            print(f"\n📊 Legacy Analysis Results")
            print("=" * 40)
            print(f"Components analyzed: {len(components)}")
            
            # Show summary statistics
            total_loc = sum(c.lines_of_code for c in components.values())
            total_debt = sum(len(c.technical_debt) for c in components.values())
            avg_maintainability = sum(c.maintainability_score for c in components.values()) / len(components)
            
            print(f"Total lines of code: {total_loc:,}")
            print(f"Technical debt items: {total_debt}")
            print(f"Average maintainability: {avg_maintainability:.1f}/10")
            
            # Show top technical debt items
            all_debt = []
            for component in components.values():
                all_debt.extend(component.technical_debt)
            
            high_severity_debt = [d for d in all_debt if d.severity.value in ['high', 'critical']]
            if high_severity_debt:
                print(f"\n⚠️  High-Severity Technical Debt ({len(high_severity_debt)} items):")
                for debt in high_severity_debt[:5]:  # Show top 5
                    print(f"  • {debt.description} ({debt.severity.value})")
                    print(f"    File: {Path(debt.file_path).name}:{debt.line_number}")
            
            # Show components with lowest maintainability
            low_maintainability = sorted(components.values(), 
                                       key=lambda x: x.maintainability_score)[:3]
            if low_maintainability:
                print(f"\n🔧 Components Needing Attention:")
                for component in low_maintainability:
                    print(f"  • {component.name}: {component.maintainability_score:.1f}/10")
                    print(f"    {len(component.technical_debt)} debt items, {component.lines_of_code} LOC")
            
            # Generate modernization targets
            targets = self.legacy_analyzer.generate_modernization_targets(components)
            if targets:
                print(f"\n🎯 Modernization Targets ({len(targets)}):")
                for target in targets[:3]:  # Show top 3
                    print(f"  • {target.component}")
                    print(f"    Complexity: {target.migration_complexity}")
                    print(f"    Effort: {target.estimated_effort_days} days")
                    print(f"    Business Value: {target.business_value:.1f}/10")
            
            print(f"\n💡 Recommendations:")
            print("  • Focus on high-severity technical debt first")
            print("  • Consider refactoring components with low maintainability scores")
            print("  • Plan migration starting with highest business value targets")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PomegranteMuse - ML-powered code analysis and translation for Pomegranate"
    )
    parser.add_argument(
        "--working-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Working directory for the project"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"PomegranteMuse {__version__}"
    )
    
    args = parser.parse_args()
    
    # Start interactive CLI
    cli = InteractiveCLI()
    
    try:
        asyncio.run(cli.start_interactive_session(args.working_dir))
    except KeyboardInterrupt:
        print("\nExiting PomegranteMuse...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()