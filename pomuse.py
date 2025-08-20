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
    
    async def start_interactive_session(self, working_dir: Path):
        """Start interactive session"""
        print(f"🍎 Welcome to PomegranteMuse v{__version__}")
        print(f"Working directory: {working_dir}")
        
        # Initialize project
        self.project_state = ProjectState(working_dir)
        self.project_state.initialize()
        
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
  help              - Show this help message
  exit              - Exit PomegranteMuse

Examples:
  analyze ./src
  generate "create a math framework from these files"
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