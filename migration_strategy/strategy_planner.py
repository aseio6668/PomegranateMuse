"""
Migration Strategy Planner for PomegranteMuse
Creates comprehensive migration plans based on codebase analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import networkx as nx

class MigrationStrategy(Enum):
    """Migration strategy types"""
    BIG_BANG = "big_bang"  # Complete migration at once
    STRANGLER_FIG = "strangler_fig"  # Gradual replacement
    PARALLEL_RUN = "parallel_run"  # Run both systems in parallel
    INCREMENTAL = "incremental"  # Component-by-component migration
    HYBRID = "hybrid"  # Combination of strategies

class MigrationPhase(Enum):
    """Migration phase types"""
    ASSESSMENT = "assessment"
    PLANNING = "planning"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Component types for migration"""
    CORE_LOGIC = "core_logic"
    DATA_ACCESS = "data_access"
    USER_INTERFACE = "user_interface"
    API_LAYER = "api_layer"
    CONFIGURATION = "configuration"
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class ComponentAnalysis:
    """Analysis of a code component"""
    name: str
    component_type: ComponentType
    file_paths: List[str]
    source_language: str
    target_language: str
    lines_of_code: int
    complexity_score: float
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    migration_effort: int = 0  # in hours
    priority: int = 1  # 1-10 scale
    
@dataclass
class DependencyGraph:
    """Dependency graph for migration planning"""
    components: Dict[str, ComponentAnalysis]
    dependencies: Dict[str, List[str]]  # component -> [dependencies]
    execution_order: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)

@dataclass
class RiskAssessment:
    """Risk assessment for migration"""
    component: str
    risk_level: RiskLevel
    risk_factors: List[str]
    mitigation_strategies: List[str]
    impact_description: str
    probability: float  # 0.0 - 1.0
    impact_score: int  # 1-10 scale
    risk_score: float  # probability * impact_score
    
@dataclass
class MigrationPlan:
    """Comprehensive migration plan"""
    project_name: str
    strategy: MigrationStrategy
    source_language: str
    target_language: str
    phases: List[MigrationPhase]
    component_analysis: Dict[str, ComponentAnalysis]
    dependency_graph: DependencyGraph
    risk_assessments: List[RiskAssessment]
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]
    success_criteria: List[str]
    rollback_plan: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
class MigrationPlanner:
    """Creates comprehensive migration plans"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".pomuse/migration")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Strategy templates
        self.strategy_templates = {
            MigrationStrategy.BIG_BANG: {
                "phases": [
                    MigrationPhase.ASSESSMENT,
                    MigrationPhase.PLANNING,
                    MigrationPhase.PREPARATION,
                    MigrationPhase.EXECUTION,
                    MigrationPhase.VALIDATION,
                    MigrationPhase.DEPLOYMENT
                ],
                "parallel_execution": False,
                "rollback_complexity": "high",
                "recommended_for": ["small_projects", "simple_architecture"]
            },
            MigrationStrategy.STRANGLER_FIG: {
                "phases": [
                    MigrationPhase.ASSESSMENT,
                    MigrationPhase.PLANNING,
                    MigrationPhase.PREPARATION,
                    MigrationPhase.EXECUTION,
                    MigrationPhase.VALIDATION,
                    MigrationPhase.DEPLOYMENT,
                    MigrationPhase.MONITORING,
                    MigrationPhase.OPTIMIZATION
                ],
                "parallel_execution": True,
                "rollback_complexity": "low",
                "recommended_for": ["large_projects", "critical_systems"]
            },
            MigrationStrategy.INCREMENTAL: {
                "phases": [
                    MigrationPhase.ASSESSMENT,
                    MigrationPhase.PLANNING,
                    MigrationPhase.PREPARATION,
                    MigrationPhase.EXECUTION,
                    MigrationPhase.VALIDATION,
                    MigrationPhase.DEPLOYMENT,
                    MigrationPhase.MONITORING
                ],
                "parallel_execution": True,
                "rollback_complexity": "medium",
                "recommended_for": ["modular_architecture", "continuous_delivery"]
            }
        }
        
    async def create_migration_plan(self, project_path: Path, source_language: str,
                                  target_language: str, strategy: MigrationStrategy = None) -> MigrationPlan:
        """Create comprehensive migration plan"""
        self.logger.info(f"Creating migration plan for {project_path}")
        
        # Analyze codebase
        components = await self._analyze_components(project_path, source_language, target_language)
        
        # Build dependency graph
        dependency_graph = await self._build_dependency_graph(components)
        
        # Assess risks
        risk_assessments = await self._assess_risks(components, dependency_graph)
        
        # Select strategy if not provided
        if strategy is None:
            strategy = await self._recommend_strategy(components, dependency_graph, risk_assessments)
        
        # Create timeline
        timeline = await self._create_timeline(components, dependency_graph, strategy)
        
        # Calculate resource requirements
        resource_requirements = await self._calculate_resources(components, dependency_graph)
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(components, target_language)
        
        # Create rollback plan
        rollback_plan = await self._create_rollback_plan(components, strategy)
        
        # Get strategy template
        template = self.strategy_templates.get(strategy, {})
        phases = template.get("phases", list(MigrationPhase))
        
        plan = MigrationPlan(
            project_name=project_path.name,
            strategy=strategy,
            source_language=source_language,
            target_language=target_language,
            phases=phases,
            component_analysis=components,
            dependency_graph=dependency_graph,
            risk_assessments=risk_assessments,
            timeline=timeline,
            resource_requirements=resource_requirements,
            success_criteria=success_criteria,
            rollback_plan=rollback_plan
        )
        
        # Save plan
        await self._save_plan(plan)
        
        return plan
    
    async def _analyze_components(self, project_path: Path, source_language: str,
                                target_language: str) -> Dict[str, ComponentAnalysis]:
        """Analyze codebase components"""
        components = {}
        
        # File extension mapping
        extension_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
            'csharp': ['.cs'],
            'rust': ['.rs'],
            'go': ['.go']
        }
        
        source_extensions = extension_map.get(source_language, [])
        
        # Find source files
        source_files = []
        for ext in source_extensions:
            source_files.extend(project_path.rglob(f"*{ext}"))
        
        # Analyze each file/component
        for file_path in source_files:
            try:
                component = await self._analyze_single_component(
                    file_path, project_path, source_language, target_language
                )
                if component:
                    components[component.name] = component
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Group related files into logical components
        components = await self._group_components(components, project_path)
        
        return components
    
    async def _analyze_single_component(self, file_path: Path, project_path: Path,
                                      source_language: str, target_language: str) -> Optional[ComponentAnalysis]:
        """Analyze a single component/file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            # Determine component type
            component_type = self._determine_component_type(file_path, content)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity(content, source_language)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content, source_language)
            
            # Assess risk factors
            risk_factors = self._assess_component_risks(file_path, content, source_language)
            
            # Estimate migration effort
            migration_effort = self._estimate_migration_effort(
                lines_of_code, complexity_score, component_type, dependencies
            )
            
            # Calculate priority
            priority = self._calculate_priority(component_type, complexity_score, dependencies)
            
            relative_path = str(file_path.relative_to(project_path))
            component_name = self._generate_component_name(file_path, project_path)
            
            return ComponentAnalysis(
                name=component_name,
                component_type=component_type,
                file_paths=[relative_path],
                source_language=source_language,
                target_language=target_language,
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                dependencies=dependencies,
                external_dependencies=[],
                risk_factors=risk_factors,
                migration_effort=migration_effort,
                priority=priority
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _determine_component_type(self, file_path: Path, content: str) -> ComponentType:
        """Determine component type based on file path and content"""
        path_str = str(file_path).lower()
        content_lower = content.lower()
        
        # Check path patterns
        if any(pattern in path_str for pattern in ['test', 'spec', '__test__']):
            return ComponentType.TESTS
        elif any(pattern in path_str for pattern in ['ui', 'view', 'component', 'frontend']):
            return ComponentType.USER_INTERFACE
        elif any(pattern in path_str for pattern in ['api', 'controller', 'endpoint', 'route']):
            return ComponentType.API_LAYER
        elif any(pattern in path_str for pattern in ['dao', 'repository', 'model', 'entity', 'database']):
            return ComponentType.DATA_ACCESS
        elif any(pattern in path_str for pattern in ['config', 'settings', 'properties']):
            return ComponentType.CONFIGURATION
        elif any(pattern in path_str for pattern in ['doc', 'readme', 'documentation']):
            return ComponentType.DOCUMENTATION
        elif any(pattern in path_str for pattern in ['infra', 'deploy', 'docker', 'k8s']):
            return ComponentType.INFRASTRUCTURE
        
        # Check content patterns
        if any(keyword in content_lower for keyword in ['class ', 'function ', 'method ', 'algorithm']):
            return ComponentType.CORE_LOGIC
        elif any(keyword in content_lower for keyword in ['sql', 'database', 'query', 'table']):
            return ComponentType.DATA_ACCESS
        elif any(keyword in content_lower for keyword in ['html', 'css', 'react', 'vue', 'angular']):
            return ComponentType.USER_INTERFACE
        elif any(keyword in content_lower for keyword in ['rest', 'graphql', 'endpoint', 'route']):
            return ComponentType.API_LAYER
        
        return ComponentType.CORE_LOGIC
    
    def _calculate_complexity(self, content: str, language: str) -> float:
        """Calculate complexity score for content"""
        lines = content.split('\n')
        
        # Basic complexity metrics
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Language-specific complexity indicators
        complexity_keywords = {
            'python': ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with', 'class', 'def'],
            'javascript': ['if', 'else', 'for', 'while', 'try', 'catch', 'function', 'class', 'switch'],
            'typescript': ['if', 'else', 'for', 'while', 'try', 'catch', 'function', 'class', 'interface'],
            'java': ['if', 'else', 'for', 'while', 'try', 'catch', 'class', 'interface', 'switch'],
            'cpp': ['if', 'else', 'for', 'while', 'try', 'catch', 'class', 'struct', 'switch'],
            'rust': ['if', 'else', 'for', 'while', 'match', 'fn', 'struct', 'enum', 'impl'],
            'go': ['if', 'else', 'for', 'switch', 'func', 'struct', 'interface', 'select']
        }
        
        keywords = complexity_keywords.get(language, complexity_keywords['python'])
        
        complexity_count = 0
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if f' {keyword} ' in line_lower or line_lower.strip().startswith(keyword):
                    complexity_count += 1
        
        # Normalize complexity score (0.0 - 10.0)
        if code_lines == 0:
            return 0.0
        
        complexity_ratio = complexity_count / code_lines
        complexity_score = min(10.0, complexity_ratio * 10)
        
        return complexity_score
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from content"""
        dependencies = []
        lines = content.split('\n')
        
        # Language-specific import patterns
        import_patterns = {
            'python': ['import ', 'from '],
            'javascript': ['import ', 'require('],
            'typescript': ['import ', 'require('],
            'java': ['import '],
            'cpp': ['#include'],
            'rust': ['use ', 'extern crate'],
            'go': ['import ']
        }
        
        patterns = import_patterns.get(language, ['import '])
        
        for line in lines:
            line_stripped = line.strip()
            for pattern in patterns:
                if line_stripped.startswith(pattern):
                    # Extract dependency name (simplified)
                    dependency = line_stripped.replace(pattern, '').split()[0] if line_stripped.replace(pattern, '').split() else ''
                    if dependency and dependency not in dependencies:
                        dependencies.append(dependency.strip('\'"();'))
        
        return dependencies
    
    def _assess_component_risks(self, file_path: Path, content: str, language: str) -> List[str]:
        """Assess risk factors for component"""
        risks = []
        
        # File size risk
        lines = len(content.split('\n'))
        if lines > 1000:
            risks.append("Large file size (>1000 lines)")
        elif lines > 500:
            risks.append("Medium file size (>500 lines)")
        
        # Complexity risk
        complexity = self._calculate_complexity(content, language)
        if complexity > 7:
            risks.append("High complexity score")
        elif complexity > 5:
            risks.append("Medium complexity score")
        
        # External dependency risk
        if any(pattern in content.lower() for pattern in ['native', 'jni', 'ffi', 'dll']):
            risks.append("Contains native code dependencies")
        
        # Database risk
        if any(pattern in content.lower() for pattern in ['sql', 'database', 'query']):
            risks.append("Contains database operations")
        
        # Concurrency risk
        if any(pattern in content.lower() for pattern in ['thread', 'async', 'await', 'lock', 'mutex']):
            risks.append("Contains concurrency code")
        
        # Legacy patterns risk
        if any(pattern in content.lower() for pattern in ['todo', 'fixme', 'hack', 'deprecated']):
            risks.append("Contains legacy patterns or technical debt")
        
        return risks
    
    def _estimate_migration_effort(self, lines_of_code: int, complexity_score: float,
                                 component_type: ComponentType, dependencies: List[str]) -> int:
        """Estimate migration effort in hours"""
        # Base effort based on lines of code
        base_effort = lines_of_code * 0.1  # 0.1 hours per line of code
        
        # Complexity multiplier
        complexity_multiplier = 1 + (complexity_score / 10)
        
        # Component type multiplier
        type_multipliers = {
            ComponentType.CORE_LOGIC: 1.5,
            ComponentType.DATA_ACCESS: 1.3,
            ComponentType.API_LAYER: 1.2,
            ComponentType.USER_INTERFACE: 1.4,
            ComponentType.CONFIGURATION: 0.8,
            ComponentType.TESTS: 1.1,
            ComponentType.DOCUMENTATION: 0.5,
            ComponentType.INFRASTRUCTURE: 1.6
        }
        
        type_multiplier = type_multipliers.get(component_type, 1.0)
        
        # Dependency multiplier
        dependency_multiplier = 1 + (len(dependencies) * 0.1)
        
        # Calculate total effort
        total_effort = base_effort * complexity_multiplier * type_multiplier * dependency_multiplier
        
        # Round to nearest hour, minimum 1 hour
        return max(1, round(total_effort))
    
    def _calculate_priority(self, component_type: ComponentType, complexity_score: float,
                          dependencies: List[str]) -> int:
        """Calculate component priority (1-10, higher is more important)"""
        # Base priority by component type
        type_priorities = {
            ComponentType.CORE_LOGIC: 8,
            ComponentType.DATA_ACCESS: 7,
            ComponentType.API_LAYER: 6,
            ComponentType.USER_INTERFACE: 5,
            ComponentType.CONFIGURATION: 4,
            ComponentType.TESTS: 3,
            ComponentType.DOCUMENTATION: 2,
            ComponentType.INFRASTRUCTURE: 9
        }
        
        base_priority = type_priorities.get(component_type, 5)
        
        # Adjust for complexity
        if complexity_score > 7:
            base_priority += 1
        elif complexity_score < 3:
            base_priority -= 1
        
        # Adjust for dependencies
        if len(dependencies) > 10:
            base_priority += 1
        elif len(dependencies) == 0:
            base_priority -= 1
        
        return min(10, max(1, base_priority))
    
    def _generate_component_name(self, file_path: Path, project_path: Path) -> str:
        """Generate a component name from file path"""
        relative_path = file_path.relative_to(project_path)
        
        # Remove file extension and convert to component name
        name_parts = list(relative_path.parts[:-1])  # Exclude filename
        filename = relative_path.stem  # Filename without extension
        
        if filename not in ['index', 'main', '__init__']:
            name_parts.append(filename)
        
        return '.'.join(name_parts) if name_parts else filename
    
    async def _group_components(self, components: Dict[str, ComponentAnalysis],
                              project_path: Path) -> Dict[str, ComponentAnalysis]:
        """Group related files into logical components"""
        # For now, return components as-is
        # In the future, this could group related files (e.g., .h and .cpp files)
        return components
    
    async def _build_dependency_graph(self, components: Dict[str, ComponentAnalysis]) -> DependencyGraph:
        """Build dependency graph from components"""
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add nodes
        for component_name in components.keys():
            graph.add_node(component_name)
        
        # Add edges based on dependencies
        dependencies = {}
        for component_name, component in components.items():
            component_deps = []
            
            # Check if dependencies reference other components
            for dep in component.dependencies:
                for other_component in components.keys():
                    if dep in other_component or other_component in dep:
                        component_deps.append(other_component)
                        graph.add_edge(other_component, component_name)
            
            dependencies[component_name] = component_deps
            components[component_name].dependencies = component_deps
        
        # Calculate execution order using topological sort
        try:
            execution_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # If there's a cycle, use a heuristic order
            execution_order = list(components.keys())
            execution_order.sort(key=lambda x: components[x].priority, reverse=True)
        
        # Find critical path
        critical_path = self._find_critical_path(graph, components)
        
        # Group components that can be executed in parallel
        parallel_groups = self._find_parallel_groups(graph, execution_order)
        
        return DependencyGraph(
            components=components,
            dependencies=dependencies,
            execution_order=execution_order,
            critical_path=critical_path,
            parallel_groups=parallel_groups
        )
    
    def _find_critical_path(self, graph: nx.DiGraph, components: Dict[str, ComponentAnalysis]) -> List[str]:
        """Find critical path through dependency graph"""
        # Use longest path algorithm based on migration effort
        try:
            # Create edge weights based on migration effort
            for node in graph.nodes():
                graph.nodes[node]['weight'] = components[node].migration_effort
            
            # Find longest path (critical path)
            if graph.nodes():
                start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
                if start_nodes:
                    # For simplicity, return path from first start node
                    return nx.shortest_path(graph, start_nodes[0], 
                                          max(graph.nodes(), key=lambda x: components[x].migration_effort))
        except:
            pass
        
        # Fallback: return high-priority components
        return sorted(components.keys(), key=lambda x: components[x].priority, reverse=True)[:5]
    
    def _find_parallel_groups(self, graph: nx.DiGraph, execution_order: List[str]) -> List[List[str]]:
        """Find components that can be executed in parallel"""
        parallel_groups = []
        processed = set()
        
        for component in execution_order:
            if component in processed:
                continue
                
            # Find all components with no dependencies on unprocessed components
            group = []
            for candidate in execution_order:
                if candidate in processed:
                    continue
                    
                # Check if all dependencies are already processed
                dependencies = [pred for pred in graph.predecessors(candidate)]
                if all(dep in processed for dep in dependencies):
                    group.append(candidate)
            
            if group:
                parallel_groups.append(group)
                processed.update(group)
        
        return parallel_groups
    
    async def _assess_risks(self, components: Dict[str, ComponentAnalysis],
                          dependency_graph: DependencyGraph) -> List[RiskAssessment]:
        """Assess migration risks"""
        risk_assessments = []
        
        for component_name, component in components.items():
            risks = []
            mitigation_strategies = []
            impact_score = 1
            probability = 0.1
            
            # Assess component-specific risks
            if component.complexity_score > 7:
                risks.append("High complexity component")
                mitigation_strategies.append("Break down into smaller components")
                impact_score += 2
                probability += 0.2
            
            if len(component.dependencies) > 10:
                risks.append("Many dependencies")
                mitigation_strategies.append("Review and reduce dependencies")
                impact_score += 1
                probability += 0.1
            
            if component.lines_of_code > 1000:
                risks.append("Large component size")
                mitigation_strategies.append("Consider splitting into multiple components")
                impact_score += 1
                probability += 0.1
            
            # Assess dependency risks
            if component_name in dependency_graph.critical_path:
                risks.append("Component on critical path")
                mitigation_strategies.append("Prioritize testing and validation")
                impact_score += 3
                probability += 0.1
            
            # Add component-specific risk factors
            risks.extend(component.risk_factors)
            
            # Determine risk level
            risk_score = min(1.0, probability) * min(10, impact_score)
            
            if risk_score >= 7:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 5:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            risk_assessment = RiskAssessment(
                component=component_name,
                risk_level=risk_level,
                risk_factors=risks,
                mitigation_strategies=mitigation_strategies,
                impact_description=f"Migration failure would impact {len(component.dependents)} dependent components",
                probability=min(1.0, probability),
                impact_score=min(10, impact_score),
                risk_score=risk_score
            )
            
            risk_assessments.append(risk_assessment)
        
        return risk_assessments
    
    async def _recommend_strategy(self, components: Dict[str, ComponentAnalysis],
                                dependency_graph: DependencyGraph,
                                risk_assessments: List[RiskAssessment]) -> MigrationStrategy:
        """Recommend migration strategy based on analysis"""
        total_components = len(components)
        total_effort = sum(c.migration_effort for c in components.values())
        high_risk_components = len([r for r in risk_assessments if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        
        # Decision criteria
        if total_components <= 10 and total_effort <= 100 and high_risk_components <= 2:
            return MigrationStrategy.BIG_BANG
        elif high_risk_components > total_components * 0.3:
            return MigrationStrategy.STRANGLER_FIG
        elif len(dependency_graph.parallel_groups) > 3:
            return MigrationStrategy.INCREMENTAL
        else:
            return MigrationStrategy.PARALLEL_RUN
    
    async def _create_timeline(self, components: Dict[str, ComponentAnalysis],
                             dependency_graph: DependencyGraph,
                             strategy: MigrationStrategy) -> Dict[str, datetime]:
        """Create migration timeline"""
        now = datetime.now()
        timeline = {}
        
        # Phase durations based on strategy
        phase_durations = {
            MigrationStrategy.BIG_BANG: {
                MigrationPhase.ASSESSMENT: 7,  # days
                MigrationPhase.PLANNING: 10,
                MigrationPhase.PREPARATION: 14,
                MigrationPhase.EXECUTION: 21,
                MigrationPhase.VALIDATION: 7,
                MigrationPhase.DEPLOYMENT: 3
            },
            MigrationStrategy.STRANGLER_FIG: {
                MigrationPhase.ASSESSMENT: 14,
                MigrationPhase.PLANNING: 21,
                MigrationPhase.PREPARATION: 28,
                MigrationPhase.EXECUTION: 90,
                MigrationPhase.VALIDATION: 14,
                MigrationPhase.DEPLOYMENT: 7,
                MigrationPhase.MONITORING: 30,
                MigrationPhase.OPTIMIZATION: 14
            },
            MigrationStrategy.INCREMENTAL: {
                MigrationPhase.ASSESSMENT: 10,
                MigrationPhase.PLANNING: 14,
                MigrationPhase.PREPARATION: 21,
                MigrationPhase.EXECUTION: 60,
                MigrationPhase.VALIDATION: 10,
                MigrationPhase.DEPLOYMENT: 5,
                MigrationPhase.MONITORING: 21
            }
        }
        
        durations = phase_durations.get(strategy, phase_durations[MigrationStrategy.INCREMENTAL])
        
        current_date = now
        for phase, duration in durations.items():
            timeline[phase.value] = current_date
            current_date += timedelta(days=duration)
        
        return timeline
    
    async def _calculate_resources(self, components: Dict[str, ComponentAnalysis],
                                 dependency_graph: DependencyGraph) -> Dict[str, Any]:
        """Calculate resource requirements"""
        total_effort = sum(c.migration_effort for c in components.values())
        
        # Assume team of developers
        developers_needed = max(2, min(8, total_effort // 160))  # 160 hours per month per developer
        
        # Estimate duration in months
        duration_months = total_effort / (developers_needed * 160)
        
        return {
            "total_effort_hours": total_effort,
            "developers_needed": developers_needed,
            "estimated_duration_months": round(duration_months, 1),
            "budget_estimate": total_effort * 100,  # $100 per hour
            "tools_needed": ["IDE", "Version Control", "Testing Framework", "CI/CD Platform"],
            "skills_required": [
                f"{components[list(components.keys())[0]].source_language} expertise",
                f"{components[list(components.keys())[0]].target_language} expertise",
                "Migration tools",
                "Testing strategies"
            ]
        }
    
    async def _define_success_criteria(self, components: Dict[str, ComponentAnalysis],
                                     target_language: str) -> List[str]:
        """Define success criteria for migration"""
        return [
            "All source code successfully migrated to " + target_language,
            "All existing functionality preserved",
            "All tests pass in new environment",
            "Performance benchmarks meet or exceed original system",
            "No critical security vulnerabilities introduced",
            "Documentation updated for new technology stack",
            "Team trained on new technology stack",
            "Production deployment successful with zero downtime",
            "Monitoring and alerting operational",
            "Rollback plan tested and documented"
        ]
    
    async def _create_rollback_plan(self, components: Dict[str, ComponentAnalysis],
                                  strategy: MigrationStrategy) -> Dict[str, Any]:
        """Create rollback plan"""
        return {
            "rollback_triggers": [
                "Critical functionality failure",
                "Performance degradation > 50%",
                "Security vulnerability discovered",
                "Data loss or corruption",
                "Extended downtime (> 4 hours)"
            ],
            "rollback_steps": [
                "Stop new system deployment",
                "Switch traffic back to original system",
                "Restore database from backup if needed",
                "Verify original system functionality",
                "Communicate status to stakeholders",
                "Analyze failure and plan remediation"
            ],
            "rollback_complexity": self.strategy_templates.get(strategy, {}).get("rollback_complexity", "medium"),
            "estimated_rollback_time": "2-8 hours",
            "data_backup_strategy": "Point-in-time recovery",
            "testing_requirements": [
                "Rollback procedure testing",
                "Data integrity verification",
                "Functionality validation"
            ]
        }
    
    async def _save_plan(self, plan: MigrationPlan):
        """Save migration plan to file"""
        plan_file = self.cache_dir / f"migration_plan_{plan.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        plan_data = asdict(plan)
        
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2, default=str)
        
        self.logger.info(f"Migration plan saved to {plan_file}")

async def create_migration_plan(project_path: Path, source_language: str,
                              target_language: str, strategy: MigrationStrategy = None) -> MigrationPlan:
    """Convenience function to create migration plan"""
    planner = MigrationPlanner()
    return await planner.create_migration_plan(project_path, source_language, target_language, strategy)