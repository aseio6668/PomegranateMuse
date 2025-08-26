"""
Ollama ML Provider Plugin for MyndraComposer
Provides integration with Ollama for local model inference
"""

import json
import requests
import logging
from typing import Dict, List, Optional, Any

from ...plugin_interface import IMLProviderPlugin, PluginCapability
from ...plugin_manager import BasePlugin

class OllamaProviderPlugin(BasePlugin, IMLProviderPlugin):
    """Ollama ML provider plugin"""
    
    def __init__(self):
        super().__init__("ollama_provider", "1.0.0")
        self.base_url = "http://localhost:11434"
        self.model = "codellama"
        self.timeout = 120
        self.max_tokens = 4000
        self.temperature = 0.1
        self.session = requests.Session()
        self.available_models = []
    
    def get_provider_name(self) -> str:
        """Get provider name"""
        return "ollama"
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize provider with configuration"""
        try:
            self.base_url = config.get("base_url", self.base_url)
            self.model = config.get("model", self.model)
            self.timeout = config.get("timeout", self.timeout)
            self.max_tokens = config.get("max_tokens", self.max_tokens)
            self.temperature = config.get("temperature", self.temperature)
            
            # Test connection
            if self.health_check():
                self._load_available_models()
                self.logger.info(f"Ollama provider initialized with {len(self.available_models)} models")
                return True
            else:
                self.logger.error("Failed to connect to Ollama server")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama provider: {e}")
            return False
    
    def _load_available_models(self):
        """Load list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            self.available_models = [model["name"] for model in data.get("models", [])]
            
        except Exception as e:
            self.logger.warning(f"Failed to load available models: {e}")
            self.available_models = []
    
    def generate_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate code using Ollama model"""
        try:
            # Construct system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Prepare request
            payload = {
                "model": context.get("model", self.model),
                "prompt": f"{system_prompt}\n\nUser Request: {prompt}",
                "stream": False,
                "options": {
                    "temperature": context.get("temperature", self.temperature),
                    "num_predict": context.get("max_tokens", self.max_tokens)
                }
            }
            
            # Make request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            generated_code = data.get("response", "")
            
            # Clean up the response
            return self._clean_generated_code(generated_code)
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return f"Error generating code: {e}"
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt for code generation"""
        source_lang = context.get("source_language", "unknown")
        target_lang = context.get("target_language", "myndra")
        
        prompt = f"""You are an expert programmer specializing in code translation and generation.
Your task is to translate code from {source_lang} to {target_lang}.

Guidelines:
1. Preserve the original functionality and logic
2. Use idiomatic {target_lang} patterns and best practices
3. Add appropriate comments for clarity
4. Handle edge cases and error conditions
5. Ensure the code is production-ready

Context information:
- Source language: {source_lang}
- Target language: {target_lang}
- Code style: {context.get('code_style', 'standard')}
- Quality level: {context.get('quality_level', 'production')}
"""
        
        if context.get("architecture_patterns"):
            prompt += f"\n- Architecture patterns: {', '.join(context['architecture_patterns'])}"
        
        if context.get("dependencies"):
            prompt += f"\n- Available dependencies: {', '.join(context['dependencies'])}"
        
        return prompt
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code"""
        # Remove common prefixes/suffixes
        lines = code.split('\n')
        
        # Remove explanation text before code
        start_idx = 0
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in ['```', 'code:', 'here is', 'here\'s']):
                start_idx = i + 1
                break
        
        # Remove explanation text after code
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if '```' in lines[i]:
                end_idx = i
                break
        
        cleaned_lines = lines[start_idx:end_idx]
        return '\n'.join(cleaned_lines).strip()
    
    def analyze_code(self, code: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze code using Ollama model"""
        try:
            # Prepare analysis prompt
            prompt = self._build_analysis_prompt(code, analysis_type)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for analysis
                    "num_predict": 2000
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            analysis_result = data.get("response", "")
            
            return self._parse_analysis_result(analysis_result, analysis_type)
            
        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return {"error": f"Analysis failed: {e}"}
    
    def _build_analysis_prompt(self, code: str, analysis_type: str) -> str:
        """Build prompt for code analysis"""
        prompts = {
            "complexity": f"""Analyze the complexity of this code:

{code}

Provide:
1. Cyclomatic complexity score
2. Cognitive complexity assessment
3. Areas of high complexity
4. Suggestions for simplification

Format your response as JSON with fields: complexity_score, cognitive_load, complex_areas, suggestions""",
            
            "quality": f"""Assess the quality of this code:

{code}

Evaluate:
1. Code style and formatting
2. Best practices adherence
3. Potential bugs or issues
4. Maintainability score (1-10)
5. Specific recommendations

Format your response as JSON with fields: style_score, best_practices, issues, maintainability, recommendations""",
            
            "security": f"""Perform a security analysis of this code:

{code}

Check for:
1. Common vulnerabilities
2. Input validation issues
3. Authentication/authorization problems
4. Data exposure risks
5. Security recommendations

Format your response as JSON with fields: vulnerabilities, severity_levels, recommendations""",
            
            "performance": f"""Analyze the performance characteristics of this code:

{code}

Assess:
1. Time complexity
2. Space complexity
3. Potential bottlenecks
4. Optimization opportunities
5. Performance recommendations

Format your response as JSON with fields: time_complexity, space_complexity, bottlenecks, optimizations"""
        }
        
        return prompts.get(analysis_type, f"Analyze this code:\n\n{code}")
    
    def _parse_analysis_result(self, result: str, analysis_type: str) -> Dict[str, Any]:
        """Parse analysis result from model response"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to plain text parsing
                return {
                    "analysis_type": analysis_type,
                    "result": result,
                    "format": "text"
                }
                
        except Exception as e:
            return {
                "analysis_type": analysis_type,
                "result": result,
                "parse_error": str(e)
            }
    
    def get_capabilities(self) -> List[PluginCapability]:
        """Get provider capabilities"""
        return [
            PluginCapability(
                name="code_generation",
                description="Generate code in various languages",
                version="1.0.0",
                required=False
            ),
            PluginCapability(
                name="code_analysis",
                description="Analyze code for complexity, quality, security",
                version="1.0.0",
                required=False
            ),
            PluginCapability(
                name="code_completion",
                description="Complete partial code snippets",
                version="1.0.0",
                required=False
            ),
            PluginCapability(
                name="explanation",
                description="Explain code functionality",
                version="1.0.0",
                required=False
            ),
            PluginCapability(
                name="translation",
                description="Translate code between languages",
                version="1.0.0",
                required=False
            )
        ]
    
    def health_check(self) -> bool:
        """Check if Ollama provider is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            payload = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # Model pulling can take time
            )
            
            if response.status_code == 200:
                self._load_available_models()  # Refresh model list
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def on_config_changed(self, config: Dict[str, Any]) -> bool:
        """Handle configuration changes"""
        return self.initialize(config)
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration"""
        errors = []
        
        if "base_url" in config:
            if not isinstance(config["base_url"], str) or not config["base_url"]:
                errors.append("base_url must be a non-empty string")
        
        if "model" in config:
            if not isinstance(config["model"], str) or not config["model"]:
                errors.append("model must be a non-empty string")
        
        if "timeout" in config:
            if not isinstance(config["timeout"], int) or config["timeout"] < 1:
                errors.append("timeout must be a positive integer")
        
        if "max_tokens" in config:
            if not isinstance(config["max_tokens"], int) or config["max_tokens"] < 1:
                errors.append("max_tokens must be a positive integer")
        
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("temperature must be a number between 0 and 2")
        
        return errors

# Plugin entry point
def create_plugin():
    """Create plugin instance"""
    return OllamaProviderPlugin()