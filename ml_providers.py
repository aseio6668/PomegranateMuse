"""
Multi-provider ML system for Universal Code Modernization Platform
Supports multiple AI providers for code analysis and generation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import aiohttp
import os
from datetime import datetime

# Import existing Ollama client
from ollama_client import OllamaClient, CodeAnalysisPrompts


class ProviderType(Enum):
    """Supported ML provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    provider: ProviderType
    capabilities: List[str]  # ["code_generation", "code_analysis", "translation", "debugging"]
    context_length: int
    cost_per_1k_tokens: float = 0.0
    speed_rating: int = 5  # 1-10, 10 being fastest
    quality_rating: int = 5  # 1-10, 10 being highest quality
    specializations: List[str] = None  # ["python", "javascript", "rust", etc.]
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []


@dataclass
class GenerationRequest:
    """Request for code generation or analysis"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    stop_sequences: List[str] = None
    model_preferences: List[str] = None  # Preferred models in order
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.model_preferences is None:
            self.model_preferences = []


@dataclass
class GenerationResponse:
    """Response from ML provider"""
    content: str
    model_used: str
    provider: ProviderType
    tokens_used: int
    cost: float
    generation_time_ms: int
    quality_score: Optional[float] = None  # 0-1, if available
    
    def __post_init__(self):
        if self.cost == 0 and hasattr(self, '_calculate_cost'):
            self.cost = self._calculate_cost()


class MLProvider(ABC):
    """Abstract base class for ML providers"""
    
    def __init__(self, provider_type: ProviderType, config: Dict[str, Any]):
        self.provider_type = provider_type
        self.config = config
        self.available_models: List[ModelInfo] = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider and check availability"""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the provider"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass
    
    def select_best_model(self, request: GenerationRequest, capability: str) -> Optional[str]:
        """Select the best model for a given capability"""
        # Filter models by capability
        capable_models = [
            model for model in self.available_models
            if capability in model.capabilities
        ]
        
        if not capable_models:
            return None
        
        # Check preferences first
        if request.model_preferences:
            for preferred in request.model_preferences:
                for model in capable_models:
                    if preferred.lower() in model.name.lower():
                        return model.name
        
        # Sort by quality rating, then by speed
        capable_models.sort(key=lambda m: (m.quality_rating, m.speed_rating), reverse=True)
        return capable_models[0].name


class OllamaProvider(MLProvider):
    """Ollama ML provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ProviderType.OLLAMA, config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client = OllamaClient(self.base_url)
    
    async def initialize(self) -> bool:
        """Initialize Ollama provider"""
        try:
            async with self.client as client:
                connected = await client.check_connection()
                if connected:
                    models = await client.list_models()
                    self.available_models = self._convert_models(models)
                    return True
                return False
        except Exception:
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Ollama models"""
        return self.available_models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using Ollama"""
        start_time = datetime.now()
        
        # Select model
        model_name = self.select_best_model(request, "code_generation")
        if not model_name and self.available_models:
            model_name = self.available_models[0].name
        
        if not model_name:
            raise Exception("No suitable model available")
        
        try:
            async with self.client as client:
                response = await client.generate(
                    model=model_name,
                    prompt=request.prompt,
                    system=request.system_prompt,
                    temperature=request.temperature
                )
                
                end_time = datetime.now()
                generation_time = int((end_time - start_time).total_seconds() * 1000)
                
                return GenerationResponse(
                    content=response.content,
                    model_used=model_name,
                    provider=ProviderType.OLLAMA,
                    tokens_used=response.eval_count or 0,
                    cost=0.0,  # Ollama is free
                    generation_time_ms=generation_time
                )
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            async with self.client as client:
                return await client.check_connection()
        except:
            return False
    
    def _convert_models(self, model_names: List[str]) -> List[ModelInfo]:
        """Convert Ollama model names to ModelInfo objects"""
        models = []
        for name in model_names:
            # Determine capabilities and specializations based on model name
            capabilities = ["code_generation", "code_analysis"]
            specializations = []
            quality_rating = 7
            speed_rating = 8
            context_length = 4096
            
            name_lower = name.lower()
            if "code" in name_lower or "llama" in name_lower:
                capabilities.extend(["translation", "debugging"])
                quality_rating = 8
                if "code" in name_lower:
                    specializations = ["python", "javascript", "typescript", "rust", "go"]
            
            if "mistral" in name_lower:
                quality_rating = 7
                speed_rating = 9
            
            if "7b" in name_lower:
                context_length = 4096
                speed_rating = 9
            elif "13b" in name_lower:
                context_length = 4096
                speed_rating = 7
                quality_rating = 8
            elif "34b" in name_lower or "70b" in name_lower:
                context_length = 4096
                speed_rating = 5
                quality_rating = 9
            
            models.append(ModelInfo(
                name=name,
                provider=ProviderType.OLLAMA,
                capabilities=capabilities,
                context_length=context_length,
                cost_per_1k_tokens=0.0,
                speed_rating=speed_rating,
                quality_rating=quality_rating,
                specializations=specializations
            ))
        
        return models


class OpenAIProvider(MLProvider):
    """OpenAI ML provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ProviderType.OPENAI, config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def initialize(self) -> bool:
        """Initialize OpenAI provider"""
        try:
            # Define available models (as of knowledge cutoff)
            self.available_models = [
                ModelInfo(
                    name="gpt-4",
                    provider=ProviderType.OPENAI,
                    capabilities=["code_generation", "code_analysis", "translation", "debugging"],
                    context_length=8192,
                    cost_per_1k_tokens=0.03,
                    speed_rating=6,
                    quality_rating=10,
                    specializations=["python", "javascript", "typescript", "rust", "go", "java", "csharp"]
                ),
                ModelInfo(
                    name="gpt-4-turbo-preview",
                    provider=ProviderType.OPENAI,
                    capabilities=["code_generation", "code_analysis", "translation", "debugging"],
                    context_length=128000,
                    cost_per_1k_tokens=0.01,
                    speed_rating=7,
                    quality_rating=10,
                    specializations=["python", "javascript", "typescript", "rust", "go", "java", "csharp"]
                ),
                ModelInfo(
                    name="gpt-3.5-turbo",
                    provider=ProviderType.OPENAI,
                    capabilities=["code_generation", "code_analysis", "translation"],
                    context_length=16384,
                    cost_per_1k_tokens=0.001,
                    speed_rating=9,
                    quality_rating=7,
                    specializations=["python", "javascript", "typescript"]
                ),
            ]
            return await self.health_check()
        except Exception:
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenAI models"""
        return self.available_models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using OpenAI"""
        start_time = datetime.now()
        
        model_name = self.select_best_model(request, "code_generation") or "gpt-3.5-turbo"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        if request.stop_sequences:
            data["stop"] = request.stop_sequences
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", 
                                      headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    end_time = datetime.now()
                    generation_time = int((end_time - start_time).total_seconds() * 1000)
                    
                    content = result["choices"][0]["message"]["content"]
                    tokens_used = result["usage"]["total_tokens"]
                    
                    # Find model info for cost calculation
                    model_info = next((m for m in self.available_models if m.name == model_name), None)
                    cost = (tokens_used / 1000) * (model_info.cost_per_1k_tokens if model_info else 0.001)
                    
                    return GenerationResponse(
                        content=content,
                        model_used=model_name,
                        provider=ProviderType.OPENAI,
                        tokens_used=tokens_used,
                        cost=cost,
                        generation_time_ms=generation_time
                    )
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check OpenAI health"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    return response.status == 200
        except:
            return False


class AnthropicProvider(MLProvider):
    """Anthropic Claude ML provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ProviderType.ANTHROPIC, config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
    
    async def initialize(self) -> bool:
        """Initialize Anthropic provider"""
        try:
            self.available_models = [
                ModelInfo(
                    name="claude-3-opus-20240229",
                    provider=ProviderType.ANTHROPIC,
                    capabilities=["code_generation", "code_analysis", "translation", "debugging"],
                    context_length=200000,
                    cost_per_1k_tokens=0.015,
                    speed_rating=5,
                    quality_rating=10,
                    specializations=["python", "javascript", "typescript", "rust", "go", "java", "csharp"]
                ),
                ModelInfo(
                    name="claude-3-sonnet-20240229",
                    provider=ProviderType.ANTHROPIC,
                    capabilities=["code_generation", "code_analysis", "translation", "debugging"],
                    context_length=200000,
                    cost_per_1k_tokens=0.003,
                    speed_rating=7,
                    quality_rating=9,
                    specializations=["python", "javascript", "typescript", "rust", "go"]
                ),
                ModelInfo(
                    name="claude-3-haiku-20240307",
                    provider=ProviderType.ANTHROPIC,
                    capabilities=["code_generation", "code_analysis", "translation"],
                    context_length=200000,
                    cost_per_1k_tokens=0.00025,
                    speed_rating=9,
                    quality_rating=7,
                    specializations=["python", "javascript", "typescript"]
                ),
            ]
            return await self.health_check()
        except Exception:
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Anthropic models"""
        return self.available_models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using Anthropic Claude"""
        start_time = datetime.now()
        
        model_name = self.select_best_model(request, "code_generation") or "claude-3-sonnet-20240229"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "user", "content": request.prompt}
            ]
        }
        
        if request.system_prompt:
            data["system"] = request.system_prompt
        
        if request.stop_sequences:
            data["stop_sequences"] = request.stop_sequences
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/messages", 
                                      headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    end_time = datetime.now()
                    generation_time = int((end_time - start_time).total_seconds() * 1000)
                    
                    content = result["content"][0]["text"]
                    tokens_used = result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
                    
                    # Find model info for cost calculation
                    model_info = next((m for m in self.available_models if m.name == model_name), None)
                    cost = (tokens_used / 1000) * (model_info.cost_per_1k_tokens if model_info else 0.003)
                    
                    return GenerationResponse(
                        content=content,
                        model_used=model_name,
                        provider=ProviderType.ANTHROPIC,
                        tokens_used=tokens_used,
                        cost=cost,
                        generation_time_ms=generation_time
                    )
        except Exception as e:
            raise Exception(f"Anthropic generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Anthropic health"""
        # Anthropic doesn't have a simple health check endpoint,
        # so we'll just validate the API key format
        return bool(self.api_key and self.api_key.startswith("sk-"))


class GoogleProvider(MLProvider):
    """Google (Gemini) ML provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ProviderType.GOOGLE, config)
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.base_url = config.get("base_url", "https://generativelanguage.googleapis.com/v1")
        
        if not self.api_key:
            raise ValueError("Google API key is required")
    
    async def initialize(self) -> bool:
        """Initialize Google provider"""
        try:
            self.available_models = [
                ModelInfo(
                    name="gemini-pro",
                    provider=ProviderType.GOOGLE,
                    capabilities=["code_generation", "code_analysis", "translation"],
                    context_length=32768,
                    cost_per_1k_tokens=0.0005,
                    speed_rating=8,
                    quality_rating=8,
                    specializations=["python", "javascript", "typescript", "java"]
                ),
                ModelInfo(
                    name="gemini-pro-vision",
                    provider=ProviderType.GOOGLE,
                    capabilities=["code_generation", "code_analysis"],
                    context_length=16384,
                    cost_per_1k_tokens=0.0005,
                    speed_rating=7,
                    quality_rating=8,
                    specializations=["python", "javascript"]
                ),
            ]
            return await self.health_check()
        except Exception:
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Google models"""
        return self.available_models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using Google Gemini"""
        start_time = datetime.now()
        
        model_name = self.select_best_model(request, "code_generation") or "gemini-pro"
        
        # Combine system prompt and user prompt for Gemini
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        
        data = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            }
        }
        
        if request.stop_sequences:
            data["generationConfig"]["stopSequences"] = request.stop_sequences
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Google API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    end_time = datetime.now()
                    generation_time = int((end_time - start_time).total_seconds() * 1000)
                    
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Google doesn't provide token counts in the response for free tier
                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    tokens_used = len(full_prompt + content) // 4
                    
                    # Find model info for cost calculation
                    model_info = next((m for m in self.available_models if m.name == model_name), None)
                    cost = (tokens_used / 1000) * (model_info.cost_per_1k_tokens if model_info else 0.0005)
                    
                    return GenerationResponse(
                        content=content,
                        model_used=model_name,
                        provider=ProviderType.GOOGLE,
                        tokens_used=tokens_used,
                        cost=cost,
                        generation_time_ms=generation_time
                    )
        except Exception as e:
            raise Exception(f"Google generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Google health"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models?key={self.api_key}"
                async with session.get(url) as response:
                    return response.status == 200
        except:
            return False


class MLProviderManager:
    """Manager for multiple ML providers with fallback and load balancing"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, MLProvider] = {}
        self.provider_health: Dict[ProviderType, bool] = {}
        self.usage_stats: Dict[ProviderType, Dict[str, Any]] = {}
        self.fallback_order: List[ProviderType] = [
            ProviderType.OLLAMA,  # Free, local
            ProviderType.OPENAI,  # High quality
            ProviderType.ANTHROPIC,  # High quality, large context
            ProviderType.GOOGLE,  # Fast, low cost
        ]
    
    async def add_provider(self, provider: MLProvider) -> bool:
        """Add a provider to the manager"""
        try:
            if await provider.initialize():
                self.providers[provider.provider_type] = provider
                self.provider_health[provider.provider_type] = True
                self.usage_stats[provider.provider_type] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "average_response_time": 0.0,
                    "errors": 0
                }
                return True
            return False
        except Exception:
            return False
    
    async def remove_provider(self, provider_type: ProviderType):
        """Remove a provider from the manager"""
        if provider_type in self.providers:
            del self.providers[provider_type]
            del self.provider_health[provider_type]
            del self.usage_stats[provider_type]
    
    async def health_check_all(self):
        """Check health of all providers"""
        for provider_type, provider in self.providers.items():
            try:
                self.provider_health[provider_type] = await provider.health_check()
            except:
                self.provider_health[provider_type] = False
    
    async def generate_with_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Generate with automatic fallback to other providers"""
        last_error = None
        
        # Try providers in fallback order, but prioritize healthy ones
        providers_to_try = [
            pt for pt in self.fallback_order 
            if pt in self.providers and self.provider_health.get(pt, False)
        ]
        
        # Add any remaining providers
        for pt in self.providers:
            if pt not in providers_to_try:
                providers_to_try.append(pt)
        
        for provider_type in providers_to_try:
            if provider_type not in self.providers:
                continue
                
            provider = self.providers[provider_type]
            
            try:
                start_time = datetime.now()
                response = await provider.generate(request)
                end_time = datetime.now()
                
                # Update statistics
                stats = self.usage_stats[provider_type]
                stats["requests"] += 1
                stats["total_tokens"] += response.tokens_used
                stats["total_cost"] += response.cost
                
                # Update average response time
                response_time = (end_time - start_time).total_seconds() * 1000
                if stats["average_response_time"] == 0:
                    stats["average_response_time"] = response_time
                else:
                    stats["average_response_time"] = (
                        stats["average_response_time"] * 0.9 + response_time * 0.1
                    )
                
                return response
                
            except Exception as e:
                last_error = e
                self.usage_stats[provider_type]["errors"] += 1
                self.provider_health[provider_type] = False
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    async def get_best_provider_for_task(self, task_type: str, preferences: List[str] = None) -> Optional[ProviderType]:
        """Get the best provider for a specific task"""
        if preferences:
            for pref in preferences:
                provider_type = ProviderType(pref.lower())
                if provider_type in self.providers and self.provider_health.get(provider_type, False):
                    return provider_type
        
        # Score providers based on health, cost, and speed
        best_provider = None
        best_score = -1
        
        for provider_type, provider in self.providers.items():
            if not self.provider_health.get(provider_type, False):
                continue
            
            models = await provider.get_available_models()
            if not models:
                continue
            
            # Find best model for task
            capable_models = [m for m in models if task_type in m.capabilities]
            if not capable_models:
                continue
            
            best_model = max(capable_models, key=lambda m: m.quality_rating)
            
            # Calculate score (higher is better)
            score = 0
            score += best_model.quality_rating * 2  # Quality is most important
            score += best_model.speed_rating  # Speed is secondary
            score -= min(best_model.cost_per_1k_tokens * 100, 10)  # Cost penalty (max 10)
            
            # Prefer local/free providers
            if provider_type == ProviderType.OLLAMA:
                score += 3
            
            # Reliability bonus (fewer errors)
            stats = self.usage_stats[provider_type]
            if stats["requests"] > 0:
                error_rate = stats["errors"] / stats["requests"]
                score += (1 - error_rate) * 2
            
            if score > best_score:
                best_score = score
                best_provider = provider_type
        
        return best_provider
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers"""
        return {
            "providers": dict(self.usage_stats),
            "health_status": dict(self.provider_health),
            "total_requests": sum(stats["requests"] for stats in self.usage_stats.values()),
            "total_tokens": sum(stats["total_tokens"] for stats in self.usage_stats.values()),
            "total_cost": sum(stats["total_cost"] for stats in self.usage_stats.values()),
        }
    
    async def get_all_available_models(self) -> List[ModelInfo]:
        """Get all available models from all providers"""
        all_models = []
        for provider in self.providers.values():
            try:
                models = await provider.get_available_models()
                all_models.extend(models)
            except:
                continue
        return all_models


# Convenience function to create provider manager with common providers
async def create_provider_manager(config: Dict[str, Any]) -> MLProviderManager:
    """Create and initialize a provider manager with configured providers"""
    manager = MLProviderManager()
    
    # Add Ollama provider (always try first as it's free/local)
    if config.get("ollama", {}).get("enabled", True):
        ollama_config = config.get("ollama", {})
        ollama_provider = OllamaProvider(ollama_config)
        await manager.add_provider(ollama_provider)
    
    # Add OpenAI provider if configured
    if config.get("openai", {}).get("api_key"):
        openai_config = config.get("openai", {})
        openai_provider = OpenAIProvider(openai_config)
        await manager.add_provider(openai_provider)
    
    # Add Anthropic provider if configured
    if config.get("anthropic", {}).get("api_key"):
        anthropic_config = config.get("anthropic", {})
        anthropic_provider = AnthropicProvider(anthropic_config)
        await manager.add_provider(anthropic_provider)
    
    # Add Google provider if configured
    if config.get("google", {}).get("api_key"):
        google_config = config.get("google", {})
        google_provider = GoogleProvider(google_config)
        await manager.add_provider(google_provider)
    
    # Initial health check
    await manager.health_check_all()
    
    return manager


# Example usage and testing
async def test_multi_provider_system():
    """Test the multi-provider system"""
    print("Testing Multi-Provider ML System...")
    
    # Configuration
    config = {
        "ollama": {
            "enabled": True,
            "base_url": "http://localhost:11434"
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        },
        "google": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    }
    
    # Create manager
    manager = await create_provider_manager(config)
    
    print(f"Initialized {len(manager.providers)} providers")
    
    # Test generation with fallback
    request = GenerationRequest(
        prompt="Write a simple 'Hello World' function in Python",
        system_prompt="You are a helpful programming assistant. Generate clean, well-documented code.",
        temperature=0.1,
        max_tokens=1000
    )
    
    try:
        response = await manager.generate_with_fallback(request)
        print(f"\nGenerated using {response.provider.value} ({response.model_used}):")
        print(f"Tokens: {response.tokens_used}, Cost: ${response.cost:.4f}, Time: {response.generation_time_ms}ms")
        print("Generated code:")
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
    except Exception as e:
        print(f"Generation failed: {e}")
    
    # Print usage stats
    print("\nUsage Statistics:")
    stats = manager.get_usage_stats()
    for provider, provider_stats in stats["providers"].items():
        print(f"  {provider}: {provider_stats['requests']} requests, ${provider_stats['total_cost']:.4f} cost")


if __name__ == "__main__":
    asyncio.run(test_multi_provider_system())