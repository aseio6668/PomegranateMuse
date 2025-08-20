"""
Semantic Architecture Analysis for Universal Code Modernization Platform
Analyzes code structure, patterns, and architecture to provide insights for migration
"""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import networkx as nx


class ArchitecturePattern(Enum):
    """Detected architecture patterns"""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    LAYERED = "layered"
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    HEXAGONAL = "hexagonal"
    EVENT_DRIVEN = "event_driven"
    PLUGIN_ARCHITECTURE = "plugin_architecture"
    SERVICE_ORIENTED = "service_oriented"
    SERVERLESS = "serverless"
    PIPE_AND_FILTER = "pipe_and_filter"
    BROKER = "broker"
    PEER_TO_PEER = "peer_to_peer"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components in the architecture"""
    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    MODEL = "model"
    VIEW = "view"
    UTILITY = "utility"
    CONFIG = "config"
    MIDDLEWARE = "middleware"
    HANDLER = "handler"
    CLIENT = "client"
    ADAPTER = "adapter"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    COMMAND = "command"
    UNKNOWN = "unknown"


@dataclass
class Component:
    """Represents a component in the architecture"""
    name: str
    file_path: str
    component_type: ComponentType
    public_methods: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    coupling_score: float = 0.0
    cohesion_score: float = 0.0
    responsibilities: List[str] = field(default_factory=list)
    design_patterns: List[str] = field(default_factory=list)


@dataclass
class ArchitectureInsight:
    """Insight about the architecture"""
    insight_type: str
    severity: str  # "info", "warning", "critical"
    title: str
    description: str
    affected_components: List[str] = field(default_factory=list)
    suggested_action: str = ""
    migration_impact: str = "low"  # "low", "medium", "high"


@dataclass
class ArchitectureAnalysis:
    """Complete architecture analysis results"""
    detected_patterns: List[ArchitecturePattern]
    components: List[Component]
    dependency_graph: Dict[str, List[str]]
    circular_dependencies: List[List[str]]
    architectural_violations: List[str]
    complexity_metrics: Dict[str, float]
    scalability_score: float
    maintainability_score: float
    testability_score: float
    insights: List[ArchitectureInsight]
    migration_recommendations: List[str]


class PythonArchitectureAnalyzer:
    """Analyzes Python code architecture"""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.dependency_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        
        # Pattern detection rules
        self.pattern_indicators = {
            ArchitecturePattern.MVC: {
                'directories': ['models', 'views', 'controllers'],
                'files': ['model', 'view', 'controller'],
                'imports': ['django', 'flask', 'rails']
            },
            ArchitecturePattern.MICROSERVICES: {
                'directories': ['services', 'microservices'],
                'files': ['service', 'api', 'gateway'],
                'imports': ['flask', 'fastapi', 'tornado', 'aiohttp']
            },
            ArchitecturePattern.LAYERED: {
                'directories': ['presentation', 'business', 'data', 'persistence'],
                'files': ['layer', 'tier'],
                'patterns': ['repository', 'service']
            },
            ArchitecturePattern.EVENT_DRIVEN: {
                'imports': ['celery', 'kafka', 'rabbitmq', 'redis'],
                'patterns': ['publisher', 'subscriber', 'observer', 'listener']
            }
        }
        
        # Component type detection patterns
        self.component_patterns = {
            ComponentType.CONTROLLER: ['controller', 'handler', 'resource', 'endpoint'],
            ComponentType.SERVICE: ['service', 'manager', 'processor', 'engine'],
            ComponentType.REPOSITORY: ['repository', 'dao', 'store', 'persistence'],
            ComponentType.MODEL: ['model', 'entity', 'domain', 'schema'],
            ComponentType.VIEW: ['view', 'template', 'presenter', 'component'],
            ComponentType.UTILITY: ['util', 'helper', 'tool', 'common'],
            ComponentType.CONFIG: ['config', 'setting', 'properties', 'env'],
            ComponentType.MIDDLEWARE: ['middleware', 'filter', 'interceptor'],
            ComponentType.CLIENT: ['client', 'connector', 'gateway', 'proxy'],
            ComponentType.ADAPTER: ['adapter', 'wrapper', 'bridge', 'facade']
        }
    
    def analyze_directory(self, directory: Path) -> ArchitectureAnalysis:
        """Analyze architecture of a Python project directory"""
        
        # Find all Python files
        python_files = list(directory.rglob("*.py"))
        
        # Analyze each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        # Detect architectural patterns
        detected_patterns = self._detect_patterns(directory)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph()
        
        # Find circular dependencies
        circular_deps = self._find_circular_dependencies()
        
        # Calculate metrics
        complexity_metrics = self._calculate_complexity_metrics()
        
        # Generate insights
        insights = self._generate_insights()
        
        # Calculate architecture quality scores
        scalability_score = self._calculate_scalability_score()
        maintainability_score = self._calculate_maintainability_score()
        testability_score = self._calculate_testability_score()
        
        # Generate migration recommendations
        migration_recommendations = self._generate_migration_recommendations(detected_patterns)
        
        return ArchitectureAnalysis(
            detected_patterns=detected_patterns,
            components=list(self.components.values()),
            dependency_graph=dependency_graph,
            circular_dependencies=circular_deps,
            architectural_violations=self._find_violations(),
            complexity_metrics=complexity_metrics,
            scalability_score=scalability_score,
            maintainability_score=maintainability_score,
            testability_score=testability_score,
            insights=insights,
            migration_recommendations=migration_recommendations
        )
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract component information
            component = self._extract_component_info(file_path, tree, content)
            self.components[str(file_path)] = component
            
            # Add to dependency graph
            self.dependency_graph.add_node(str(file_path))
            
            # Analyze imports for dependencies
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._process_import(str(file_path), node)
            
            # Analyze function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._process_call(str(file_path), node)
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def _extract_component_info(self, file_path: Path, tree: ast.AST, content: str) -> Component:
        """Extract component information from AST"""
        
        # Determine component type
        component_type = self._classify_component(file_path.name, content)
        
        # Extract public methods
        public_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                public_methods.append(node.name)
        
        # Calculate complexity metrics
        lines_of_code = len(content.split('\n'))
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        
        # Extract responsibilities (from docstrings, comments, method names)
        responsibilities = self._extract_responsibilities(tree, content)
        
        # Detect design patterns
        design_patterns = self._detect_design_patterns(tree, content)
        
        return Component(
            name=file_path.stem,
            file_path=str(file_path),
            component_type=component_type,
            public_methods=public_methods,
            lines_of_code=lines_of_code,
            cyclomatic_complexity=cyclomatic_complexity,
            responsibilities=responsibilities,
            design_patterns=design_patterns
        )
    
    def _classify_component(self, filename: str, content: str) -> ComponentType:
        """Classify component type based on filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        for component_type, patterns in self.component_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return component_type
                
                # Check in content (class names, function names, etc.)
                if re.search(rf'\b{pattern}\b', content_lower):
                    return component_type
        
        return ComponentType.UNKNOWN
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the module"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _extract_responsibilities(self, tree: ast.AST, content: str) -> List[str]:
        """Extract component responsibilities from code analysis"""
        responsibilities = []
        
        # Extract from docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Simple keyword extraction
                    doc_lower = docstring.lower()
                    if 'handle' in doc_lower or 'process' in doc_lower:
                        responsibilities.append('processing')
                    if 'store' in doc_lower or 'save' in doc_lower or 'persist' in doc_lower:
                        responsibilities.append('data_persistence')
                    if 'validate' in doc_lower or 'check' in doc_lower:
                        responsibilities.append('validation')
                    if 'render' in doc_lower or 'display' in doc_lower:
                        responsibilities.append('presentation')
        
        # Extract from method names
        method_keywords = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for keyword in ['get', 'set', 'create', 'update', 'delete', 'validate', 'process', 'handle']:
                    if keyword in node.name.lower():
                        method_keywords[keyword] += 1
        
        # Add responsibilities based on method patterns
        if method_keywords['get'] > 0 or method_keywords['set'] > 0:
            responsibilities.append('data_access')
        if method_keywords['create'] > 0 or method_keywords['update'] > 0 or method_keywords['delete'] > 0:
            responsibilities.append('data_manipulation')
        if method_keywords['validate'] > 0:
            responsibilities.append('validation')
        if method_keywords['process'] > 0 or method_keywords['handle'] > 0:
            responsibilities.append('business_logic')
        
        return list(set(responsibilities))
    
    def _detect_design_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Detect design patterns in the code"""
        patterns = []
        
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Singleton pattern
        if any('singleton' in name.lower() for name in class_names):
            patterns.append('singleton')
        
        # Factory pattern
        if any('factory' in name.lower() for name in class_names):
            patterns.append('factory')
        
        # Observer pattern
        if any(name.lower().endswith(('observer', 'listener', 'subscriber')) for name in class_names):
            patterns.append('observer')
        
        # Strategy pattern
        if any('strategy' in name.lower() for name in class_names):
            patterns.append('strategy')
        
        # Repository pattern
        if any(name.lower().endswith('repository') for name in class_names):
            patterns.append('repository')
        
        # Adapter pattern
        if any(name.lower().endswith('adapter') for name in class_names):
            patterns.append('adapter')
        
        # Builder pattern
        if any('builder' in name.lower() for name in class_names):
            patterns.append('builder')
        
        # Command pattern
        if any(name.lower().endswith('command') for name in class_names):
            patterns.append('command')
        
        return patterns
    
    def _process_import(self, file_path: str, import_node: ast.AST):
        """Process import statements to build dependencies"""
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                self._add_dependency(file_path, alias.name)
        elif isinstance(import_node, ast.ImportFrom):
            if import_node.module:
                self._add_dependency(file_path, import_node.module)
    
    def _add_dependency(self, from_file: str, to_module: str):
        """Add dependency to the graph"""
        # Only track internal dependencies (not external libraries)
        if not self._is_external_module(to_module):
            self.dependency_graph.add_edge(from_file, to_module)
            
            # Update component dependencies
            if from_file in self.components:
                self.components[from_file].dependencies.append(to_module)
    
    def _is_external_module(self, module_name: str) -> bool:
        """Check if module is external (standard library or third-party)"""
        external_indicators = [
            'os', 'sys', 'json', 'datetime', 'collections', 'typing', 'pathlib',
            'django', 'flask', 'requests', 'numpy', 'pandas', 'matplotlib',
            'tensorflow', 'torch', 'sklearn', 'fastapi', 'aiohttp'
        ]
        
        for indicator in external_indicators:
            if module_name.startswith(indicator):
                return True
        
        return False
    
    def _process_call(self, file_path: str, call_node: ast.Call):
        """Process function calls to build call graph"""
        if isinstance(call_node.func, ast.Attribute):
            # Method call
            if isinstance(call_node.func.value, ast.Name):
                caller = f"{file_path}.{call_node.func.value.id}"
                callee = call_node.func.attr
                self.call_graph.add_edge(caller, callee)
    
    def _detect_patterns(self, directory: Path) -> List[ArchitecturePattern]:
        """Detect architectural patterns from directory structure and code"""
        detected_patterns = []
        
        # Get directory structure
        subdirs = [d.name.lower() for d in directory.iterdir() if d.is_dir()]
        all_files = [f.name.lower() for f in directory.rglob("*") if f.is_file()]
        
        # Check each pattern
        for pattern, indicators in self.pattern_indicators.items():
            score = 0
            
            # Check directories
            if 'directories' in indicators:
                for dir_name in indicators['directories']:
                    if any(dir_name in subdir for subdir in subdirs):
                        score += 2
            
            # Check files
            if 'files' in indicators:
                for file_pattern in indicators['files']:
                    if any(file_pattern in filename for filename in all_files):
                        score += 1
            
            # Check imports (from analyzed components)
            if 'imports' in indicators:
                for component in self.components.values():
                    content = ""
                    try:
                        with open(component.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                    except:
                        continue
                    
                    for import_pattern in indicators['imports']:
                        if import_pattern in content:
                            score += 1
            
            # Check patterns in code
            if 'patterns' in indicators:
                for component in self.components.values():
                    for code_pattern in indicators['patterns']:
                        if code_pattern in component.name.lower() or code_pattern in component.design_patterns:
                            score += 1
            
            # If score is high enough, consider pattern detected
            if score >= 2:
                detected_patterns.append(pattern)
        
        # Default to monolithic if no specific patterns detected
        if not detected_patterns and len(self.components) > 5:
            detected_patterns.append(ArchitecturePattern.MONOLITHIC)
        
        return detected_patterns if detected_patterns else [ArchitecturePattern.UNKNOWN]
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph dictionary"""
        graph_dict = {}
        for node in self.dependency_graph.nodes():
            graph_dict[node] = list(self.dependency_graph.successors(node))
        return graph_dict
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the dependency graph"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except Exception:
            return []
    
    def _calculate_complexity_metrics(self) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        if not self.components:
            return {}
        
        # Average cyclomatic complexity
        avg_complexity = sum(c.cyclomatic_complexity for c in self.components.values()) / len(self.components)
        
        # Average lines of code per component
        avg_loc = sum(c.lines_of_code for c in self.components.values()) / len(self.components)
        
        # Coupling metrics
        avg_coupling = self._calculate_average_coupling()
        
        # Cohesion metrics
        avg_cohesion = self._calculate_average_cohesion()
        
        return {
            'average_cyclomatic_complexity': avg_complexity,
            'average_lines_of_code': avg_loc,
            'average_coupling': avg_coupling,
            'average_cohesion': avg_cohesion,
            'total_components': len(self.components),
            'dependency_graph_density': nx.density(self.dependency_graph) if self.dependency_graph.nodes() else 0
        }
    
    def _calculate_average_coupling(self) -> float:
        """Calculate average coupling between components"""
        if not self.components:
            return 0.0
        
        total_coupling = 0
        for component in self.components.values():
            # Efferent coupling (outgoing dependencies)
            efferent = len(component.dependencies)
            # Afferent coupling (incoming dependencies)
            afferent = len(component.dependents)
            total_coupling += efferent + afferent
        
        return total_coupling / len(self.components)
    
    def _calculate_average_cohesion(self) -> float:
        """Calculate average cohesion within components"""
        if not self.components:
            return 0.0
        
        total_cohesion = 0
        for component in self.components.values():
            # Simple cohesion metric based on responsibilities
            # High cohesion = few, related responsibilities
            if len(component.responsibilities) == 0:
                cohesion = 0.5  # Neutral
            elif len(component.responsibilities) == 1:
                cohesion = 1.0  # High cohesion
            else:
                # Lower cohesion for more responsibilities
                cohesion = 1.0 / len(component.responsibilities)
            
            total_cohesion += cohesion
        
        return total_cohesion / len(self.components)
    
    def _find_violations(self) -> List[str]:
        """Find architectural violations"""
        violations = []
        
        # High coupling violations
        for component in self.components.values():
            if len(component.dependencies) > 10:
                violations.append(f"High coupling in {component.name}: {len(component.dependencies)} dependencies")
        
        # Large component violations
        for component in self.components.values():
            if component.lines_of_code > 1000:
                violations.append(f"Large component {component.name}: {component.lines_of_code} lines of code")
        
        # Complex component violations
        for component in self.components.values():
            if component.cyclomatic_complexity > 20:
                violations.append(f"High complexity in {component.name}: {component.cyclomatic_complexity} cyclomatic complexity")
        
        # Circular dependency violations
        cycles = self._find_circular_dependencies()
        for cycle in cycles:
            violations.append(f"Circular dependency: {' -> '.join(cycle)}")
        
        return violations
    
    def _generate_insights(self) -> List[ArchitectureInsight]:
        """Generate architectural insights and recommendations"""
        insights = []
        
        # Complexity insights
        complex_components = [c for c in self.components.values() if c.cyclomatic_complexity > 15]
        if complex_components:
            insights.append(ArchitectureInsight(
                insight_type="complexity",
                severity="warning",
                title="High Complexity Components Detected",
                description=f"Found {len(complex_components)} components with high cyclomatic complexity",
                affected_components=[c.name for c in complex_components],
                suggested_action="Consider refactoring these components into smaller, more focused units",
                migration_impact="medium"
            ))
        
        # Coupling insights
        highly_coupled = [c for c in self.components.values() if len(c.dependencies) > 8]
        if highly_coupled:
            insights.append(ArchitectureInsight(
                insight_type="coupling",
                severity="warning",
                title="High Coupling Detected",
                description=f"Found {len(highly_coupled)} components with high coupling",
                affected_components=[c.name for c in highly_coupled],
                suggested_action="Reduce dependencies through dependency injection or interface segregation",
                migration_impact="high"
            ))
        
        # Large component insights
        large_components = [c for c in self.components.values() if c.lines_of_code > 500]
        if large_components:
            insights.append(ArchitectureInsight(
                insight_type="size",
                severity="info",
                title="Large Components Found",
                description=f"Found {len(large_components)} components with more than 500 lines of code",
                affected_components=[c.name for c in large_components],
                suggested_action="Consider splitting large components into smaller, cohesive modules",
                migration_impact="low"
            ))
        
        # Circular dependency insights
        cycles = self._find_circular_dependencies()
        if cycles:
            insights.append(ArchitectureInsight(
                insight_type="dependencies",
                severity="critical",
                title="Circular Dependencies Found",
                description=f"Found {len(cycles)} circular dependency cycles",
                affected_components=[],
                suggested_action="Break circular dependencies through dependency inversion or interface extraction",
                migration_impact="high"
            ))
        
        return insights
    
    def _calculate_scalability_score(self) -> float:
        """Calculate architecture scalability score (0-100)"""
        score = 100.0
        
        # Penalize high coupling
        avg_coupling = self._calculate_average_coupling()
        if avg_coupling > 5:
            score -= (avg_coupling - 5) * 5
        
        # Penalize circular dependencies
        cycles = len(self._find_circular_dependencies())
        score -= cycles * 15
        
        # Penalize large components
        large_components = len([c for c in self.components.values() if c.lines_of_code > 1000])
        score -= large_components * 10
        
        return max(0, min(100, score))
    
    def _calculate_maintainability_score(self) -> float:
        """Calculate architecture maintainability score (0-100)"""
        score = 100.0
        
        # Penalize high complexity
        if self.components:
            avg_complexity = sum(c.cyclomatic_complexity for c in self.components.values()) / len(self.components)
            if avg_complexity > 10:
                score -= (avg_complexity - 10) * 3
        
        # Reward good cohesion
        avg_cohesion = self._calculate_average_cohesion()
        if avg_cohesion < 0.5:
            score -= (0.5 - avg_cohesion) * 40
        
        # Penalize violations
        violations = len(self._find_violations())
        score -= violations * 5
        
        return max(0, min(100, score))
    
    def _calculate_testability_score(self) -> float:
        """Calculate architecture testability score (0-100)"""
        score = 100.0
        
        # Penalize high coupling (makes testing harder)
        avg_coupling = self._calculate_average_coupling()
        if avg_coupling > 3:
            score -= (avg_coupling - 3) * 8
        
        # Reward dependency injection patterns
        di_components = len([c for c in self.components.values() if 'dependency_injection' in c.design_patterns])
        score += di_components * 2
        
        # Penalize singleton pattern (harder to test)
        singleton_components = len([c for c in self.components.values() if 'singleton' in c.design_patterns])
        score -= singleton_components * 10
        
        return max(0, min(100, score))
    
    def _generate_migration_recommendations(self, detected_patterns: List[ArchitecturePattern]) -> List[str]:
        """Generate migration recommendations based on detected patterns"""
        recommendations = []
        
        if ArchitecturePattern.MONOLITHIC in detected_patterns:
            recommendations.append("Consider microservices architecture for better scalability")
            recommendations.append("Implement domain-driven design to identify service boundaries")
            recommendations.append("Use event-driven architecture for loose coupling between services")
        
        if ArchitecturePattern.MVC in detected_patterns:
            recommendations.append("Modern MVC frameworks provide better separation of concerns")
            recommendations.append("Consider MVVM pattern for better testability")
            recommendations.append("Implement dependency injection for loose coupling")
        
        if not any(pattern in [ArchitecturePattern.LAYERED, ArchitecturePattern.HEXAGONAL] for pattern in detected_patterns):
            recommendations.append("Implement layered architecture for better organization")
            recommendations.append("Consider hexagonal architecture for better testability and maintainability")
        
        # General recommendations based on insights
        if any(insight.severity == "critical" for insight in self._generate_insights()):
            recommendations.append("Address critical architectural issues before migration")
        
        if self._calculate_testability_score() < 70:
            recommendations.append("Improve testability through dependency injection and interface segregation")
        
        if self._calculate_maintainability_score() < 70:
            recommendations.append("Refactor complex components to improve maintainability")
        
        return recommendations


# Multi-language architecture analyzer factory
class ArchitectureAnalyzerFactory:
    """Factory for creating language-specific architecture analyzers"""
    
    @staticmethod
    def create_analyzer(language: str) -> Optional[PythonArchitectureAnalyzer]:
        """Create analyzer for specific language"""
        if language.lower() == "python":
            return PythonArchitectureAnalyzer()
        # Add more language analyzers here
        # elif language.lower() == "javascript":
        #     return JavaScriptArchitectureAnalyzer()
        # elif language.lower() == "java":
        #     return JavaArchitectureAnalyzer()
        
        return None


async def analyze_project_architecture(project_path: Path, language: str = "python") -> Optional[ArchitectureAnalysis]:
    """Analyze project architecture for a given language"""
    analyzer = ArchitectureAnalyzerFactory.create_analyzer(language)
    if not analyzer:
        return None
    
    return analyzer.analyze_directory(project_path)


# Example usage and testing
async def test_architecture_analysis():
    """Test the architecture analysis system"""
    print("Testing Architecture Analysis System...")
    
    # Analyze current project
    current_dir = Path(__file__).parent
    analysis = await analyze_project_architecture(current_dir, "python")
    
    if not analysis:
        print("No analyzer available for this language")
        return
    
    print(f"\\n📊 Architecture Analysis Results:")
    print(f"Detected Patterns: {[p.value for p in analysis.detected_patterns]}")
    print(f"Components: {len(analysis.components)}")
    print(f"Circular Dependencies: {len(analysis.circular_dependencies)}")
    print(f"Architectural Violations: {len(analysis.architectural_violations)}")
    
    print(f"\\n📈 Quality Scores:")
    print(f"Scalability: {analysis.scalability_score:.1f}/100")
    print(f"Maintainability: {analysis.maintainability_score:.1f}/100")
    print(f"Testability: {analysis.testability_score:.1f}/100")
    
    print(f"\\n💡 Key Insights:")
    for insight in analysis.insights[:3]:  # Show first 3 insights
        print(f"- {insight.title}: {insight.description}")
    
    print(f"\\n🔧 Migration Recommendations:")
    for rec in analysis.migration_recommendations[:3]:  # Show first 3 recommendations
        print(f"- {rec}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_architecture_analysis())