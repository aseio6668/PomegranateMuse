"""
Plugin Interfaces for PomegranteMuse
Defines abstract interfaces that plugins must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PluginCapability:
    """Plugin capability definition"""
    name: str
    description: str
    version: str
    required: bool = False
    parameters: Dict[str, Any] = None

@dataclass
class AnalysisResult:
    """Analysis result from plugins"""
    plugin_name: str
    success: bool
    data: Dict[str, Any]
    confidence: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None

@dataclass
class GenerationResult:
    """Code generation result from plugins"""
    plugin_name: str
    success: bool
    code: str
    language: str
    metadata: Dict[str, Any] = None
    errors: List[str] = None

class ILanguagePlugin(ABC):
    """Interface for language support plugins"""
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        pass
    
    @abstractmethod
    def parse_code(self, code: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Parse source code and extract structure"""
        pass
    
    @abstractmethod
    def validate_syntax(self, code: str) -> List[str]:
        """Validate code syntax and return errors"""
        pass
    
    @abstractmethod
    def get_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code"""
        pass
    
    @abstractmethod
    def estimate_complexity(self, code: str) -> float:
        """Estimate code complexity (0-1 scale)"""
        pass
    
    @abstractmethod
    def generate_ast(self, code: str) -> Dict[str, Any]:
        """Generate Abstract Syntax Tree"""
        pass

class IMLProviderPlugin(ABC):
    """Interface for ML provider plugins"""
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize provider with configuration"""
        pass
    
    @abstractmethod
    def generate_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate code using ML model"""
        pass
    
    @abstractmethod
    def analyze_code(self, code: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze code using ML model"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[PluginCapability]:
        """Get provider capabilities"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider is available"""
        pass

class IAnalysisPlugin(ABC):
    """Interface for code analysis plugins"""
    
    @abstractmethod
    def get_analysis_types(self) -> List[str]:
        """Get supported analysis types"""
        pass
    
    @abstractmethod
    def analyze(self, code: str, language: str, analysis_type: str, 
               config: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Perform code analysis"""
        pass
    
    @abstractmethod
    def get_metrics(self, code: str, language: str) -> Dict[str, float]:
        """Get code metrics"""
        pass
    
    @abstractmethod
    def detect_patterns(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect code patterns"""
        pass

class IGeneratorPlugin(ABC):
    """Interface for code generation plugins"""
    
    @abstractmethod
    def get_target_languages(self) -> List[str]:
        """Get supported target languages"""
        pass
    
    @abstractmethod
    def generate(self, source_code: str, source_language: str, 
                target_language: str, context: Dict[str, Any]) -> GenerationResult:
        """Generate code in target language"""
        pass
    
    @abstractmethod
    def get_generation_strategies(self) -> List[str]:
        """Get available generation strategies"""
        pass
    
    @abstractmethod
    def validate_output(self, generated_code: str, target_language: str) -> List[str]:
        """Validate generated code"""
        pass

class ITransformPlugin(ABC):
    """Interface for code transformation plugins"""
    
    @abstractmethod
    def get_transform_types(self) -> List[str]:
        """Get supported transformation types"""
        pass
    
    @abstractmethod
    def transform(self, code: str, language: str, transform_type: str,
                 config: Optional[Dict[str, Any]] = None) -> str:
        """Transform code"""
        pass
    
    @abstractmethod
    def can_transform(self, code: str, language: str, transform_type: str) -> bool:
        """Check if transformation is possible"""
        pass

class IIntegrationPlugin(ABC):
    """Interface for external integration plugins"""
    
    @abstractmethod
    def get_integration_type(self) -> str:
        """Get integration type (e.g., 'git', 'jira', 'slack')"""
        pass
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to external service"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from external service"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get connection status"""
        pass
    
    @abstractmethod
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration action"""
        pass

class ISecurityPlugin(ABC):
    """Interface for security analysis plugins"""
    
    @abstractmethod
    def get_scan_types(self) -> List[str]:
        """Get supported security scan types"""
        pass
    
    @abstractmethod
    def scan_code(self, code: str, language: str, scan_type: str) -> Dict[str, Any]:
        """Scan code for security issues"""
        pass
    
    @abstractmethod
    def get_vulnerability_database(self) -> Dict[str, Any]:
        """Get vulnerability database info"""
        pass
    
    @abstractmethod
    def check_dependencies(self, dependencies: List[str], language: str) -> Dict[str, Any]:
        """Check dependencies for vulnerabilities"""
        pass

class IPerformancePlugin(ABC):
    """Interface for performance analysis plugins"""
    
    @abstractmethod
    def profile_code(self, code: str, language: str) -> Dict[str, Any]:
        """Profile code performance"""
        pass
    
    @abstractmethod
    def benchmark(self, code: str, language: str, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark code execution"""
        pass
    
    @abstractmethod
    def optimize_suggestions(self, code: str, language: str) -> List[str]:
        """Get optimization suggestions"""
        pass

class ITestingPlugin(ABC):
    """Interface for testing plugins"""
    
    @abstractmethod
    def generate_tests(self, code: str, language: str, test_type: str) -> str:
        """Generate test code"""
        pass
    
    @abstractmethod
    def run_tests(self, test_code: str, source_code: str, language: str) -> Dict[str, Any]:
        """Run tests and return results"""
        pass
    
    @abstractmethod
    def get_coverage(self, test_code: str, source_code: str, language: str) -> float:
        """Get test coverage percentage"""
        pass

class IDocumentationPlugin(ABC):
    """Interface for documentation generation plugins"""
    
    @abstractmethod
    def generate_docs(self, code: str, language: str, doc_format: str) -> str:
        """Generate documentation"""
        pass
    
    @abstractmethod
    def extract_comments(self, code: str, language: str) -> List[str]:
        """Extract existing comments/documentation"""
        pass
    
    @abstractmethod
    def get_doc_formats(self) -> List[str]:
        """Get supported documentation formats"""
        pass

# Plugin lifecycle hooks
class IPluginLifecycle(ABC):
    """Interface for plugin lifecycle management"""
    
    @abstractmethod
    def on_load(self) -> bool:
        """Called when plugin is loaded"""
        pass
    
    @abstractmethod
    def on_unload(self) -> bool:
        """Called when plugin is unloaded"""
        pass
    
    @abstractmethod
    def on_enable(self) -> bool:
        """Called when plugin is enabled"""
        pass
    
    @abstractmethod
    def on_disable(self) -> bool:
        """Called when plugin is disabled"""
        pass
    
    @abstractmethod
    def on_config_changed(self, config: Dict[str, Any]) -> bool:
        """Called when plugin configuration changes"""
        pass

# Plugin communication interface
class IPluginCommunication(ABC):
    """Interface for inter-plugin communication"""
    
    @abstractmethod
    def send_message(self, target_plugin: str, message: Dict[str, Any]) -> bool:
        """Send message to another plugin"""
        pass
    
    @abstractmethod
    def receive_message(self, source_plugin: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Receive message from another plugin"""
        pass
    
    @abstractmethod
    def subscribe_to_events(self, event_types: List[str]) -> bool:
        """Subscribe to system events"""
        pass
    
    @abstractmethod
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish system event"""
        pass

# Base plugin class combining common interfaces
class BasePluginInterface(IPluginLifecycle, IPluginCommunication):
    """Base interface that all plugins should implement"""
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration"""
        pass