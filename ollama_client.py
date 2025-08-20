"""
Ollama API client for PomegranteMuse
Handles communication with Ollama models for code analysis and translation
"""

import aiohttp
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    content: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    return []
        except Exception:
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": model_name}
                async with session.post(
                    f"{self.base_url}/api/pull", 
                    json=payload
                ) as response:
                    if response.status == 200:
                        # Stream the response to monitor progress
                        async for line in response.content:
                            if line:
                                try:
                                    status = json.loads(line.decode('utf-8'))
                                    if status.get("status") == "success":
                                        return True
                                except json.JSONDecodeError:
                                    continue
                    return False
        except Exception:
            return False
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: float = 0.1,
        stream: bool = False
    ) -> OllamaResponse:
        """Generate text using specified model"""
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,  # Max tokens to generate
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                
                if stream:
                    # Handle streaming response
                    full_content = ""
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if chunk.get("response"):
                                    full_content += chunk["response"]
                                if chunk.get("done"):
                                    return OllamaResponse(
                                        content=full_content,
                                        model=chunk.get("model", model),
                                        created_at=chunk.get("created_at", ""),
                                        done=True,
                                        total_duration=chunk.get("total_duration"),
                                        load_duration=chunk.get("load_duration"),
                                        prompt_eval_count=chunk.get("prompt_eval_count"),
                                        eval_count=chunk.get("eval_count")
                                    )
                            except json.JSONDecodeError:
                                continue
                else:
                    # Handle non-streaming response
                    data = await response.json()
                    return OllamaResponse(
                        content=data.get("response", ""),
                        model=data.get("model", model),
                        created_at=data.get("created_at", ""),
                        done=data.get("done", True),
                        total_duration=data.get("total_duration"),
                        load_duration=data.get("load_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count")
                    )
                    
        except Exception as e:
            raise Exception(f"Failed to generate response: {e}")


class CodeAnalysisPrompts:
    """Pre-defined prompts for code analysis tasks"""
    
    @staticmethod
    def analyze_file_content(file_path: str, content: str, language: str) -> str:
        """Generate prompt for analyzing a single file"""
        return f"""You are an expert software engineer analyzing source code for translation to the Pomegranate programming language.

Analyze this {language} file and provide a structured analysis:

File: {file_path}
Language: {language}
Content:
```{language}
{content}
```

Please provide analysis in this JSON format:
{{
    "primary_purpose": "Brief description of what this code does",
    "key_functions": ["list", "of", "main", "functions"],
    "data_structures": ["important", "classes", "or", "types"],
    "dependencies": ["external", "libraries", "used"],
    "complexity_level": "simple|intermediate|complex",
    "design_patterns": ["patterns", "used"],
    "domain_classification": ["web", "math", "data", "game", "etc"],
    "pomegranate_features": ["reactive", "temporal", "security", "etc"],
    "translation_approach": "How to best translate this to Pomegranate",
    "key_challenges": ["challenges", "in", "translation"]
}}

Focus on understanding the code's intent and how it could be idiomatically expressed in Pomegranate."""
    
    @staticmethod
    def analyze_project_context(analyses: List[Dict[str, Any]], user_prompt: str) -> str:
        """Generate prompt for analyzing overall project context"""
        
        files_summary = "\n".join([
            f"- {analysis.get('file_path', 'Unknown')}: {analysis.get('language', 'unknown')} "
            f"({analysis.get('line_count', 0)} lines)"
            for analysis in analyses
        ])
        
        return f"""You are an expert software architect analyzing a codebase for translation to Pomegranate.

User Request: "{user_prompt}"

Codebase Summary:
{files_summary}

Individual File Analyses:
{json.dumps(analyses, indent=2)}

Based on this analysis, provide a comprehensive project assessment in JSON format:

{{
    "project_type": "Type of application/system",
    "architecture_style": "Current architectural approach",
    "main_technologies": ["key", "technologies", "used"],
    "core_functionality": "What the system does",
    "complexity_assessment": "overall complexity level",
    "pomegranate_architecture": {{
        "recommended_style": "reactive|traditional|hybrid",
        "key_modules": ["module", "names"],
        "security_approach": "capability-based recommendations",
        "ui_approach": "if applicable",
        "data_handling": "recommended approach"
    }},
    "translation_strategy": {{
        "priority_order": ["which", "files", "to", "translate", "first"],
        "template_type": "basic_app|reactive_ui|api_client|data_processor",
        "key_transformations": ["major", "changes", "needed"],
        "pomegranate_features_to_use": ["reactive", "temporal", "security", "etc"]
    }},
    "implementation_plan": [
        "Step 1: Description",
        "Step 2: Description",
        "etc"
    ],
    "potential_challenges": ["challenge", "descriptions"],
    "recommendations": ["specific", "recommendations"]
}}

Consider how the user's prompt aligns with the codebase and suggest the best Pomegranate approach."""
    
    @staticmethod
    def generate_pomegranate_code(
        context: Dict[str, Any], 
        user_prompt: str, 
        template_type: str
    ) -> str:
        """Generate prompt for creating Pomegranate code"""
        
        return f"""You are an expert Pomegranate programmer. Generate idiomatic Pomegranate code based on the analysis.

User Request: "{user_prompt}"

Context Analysis:
{json.dumps(context, indent=2)}

Template Type: {template_type}

Generate complete, working Pomegranate code that:

1. Uses appropriate Pomegranate features for the domain
2. Follows Pomegranate best practices and conventions
3. Includes proper error handling with fallback strategies
4. Uses semantic tags for organization
5. Implements capability-based security where appropriate
6. Uses reactive patterns if suitable for the use case
7. Includes temporal types for any time-based functionality
8. Uses appropriate execution models (@async, @parallel, etc.)

Key Pomegranate Features to Consider:
- Context-aware features (dev/prod/test adaptation)
- Live code capsules for modular components
- Reactive programming with observables
- Temporal types for animations/state changes
- Capability-based security model
- Self-healing error handling with fallback strategies
- Semantic navigation with #tag: annotations
- Inline DSL blocks where appropriate

Generate a complete, well-structured .pom file with:
- Proper imports with capabilities
- Clear function and module organization
- Comprehensive error handling
- Documentation comments explaining the approach
- Example usage if applicable

The code should be production-ready and demonstrate best practices for the specific domain and use case."""
    
    @staticmethod
    def refine_generated_code(
        generated_code: str,
        original_context: Dict[str, Any],
        user_feedback: str
    ) -> str:
        """Generate prompt for refining generated code based on feedback"""
        
        return f"""You are refining Pomegranate code based on user feedback and context.

Original Context:
{json.dumps(original_context, indent=2)}

Generated Code:
```pomegranate
{generated_code}
```

User Feedback: "{user_feedback}"

Please refine the code to address the feedback while maintaining Pomegranate best practices:

1. Address specific concerns raised in feedback
2. Improve code quality and idiomaticity
3. Ensure all Pomegranate features are used correctly
4. Maintain consistency with the original intent
5. Add any missing error handling or edge cases
6. Optimize for the specific use case

Return only the refined Pomegranate code, properly formatted and complete."""


async def test_ollama_connection():
    """Test function to verify Ollama connectivity"""
    async with OllamaClient() as client:
        # Check connection
        connected = await client.check_connection()
        print(f"Ollama connection: {'✅ Connected' if connected else '❌ Failed'}")
        
        if connected:
            # List available models
            models = await client.list_models()
            print(f"Available models: {models}")
            
            # Test simple generation if models available
            if models:
                test_model = models[0]
                response = await client.generate(
                    model=test_model,
                    prompt="Say hello in one sentence.",
                    temperature=0.1
                )
                print(f"Test response: {response.content[:100]}...")
                
        return connected


if __name__ == "__main__":
    # Test the client when run directly
    asyncio.run(test_ollama_connection())