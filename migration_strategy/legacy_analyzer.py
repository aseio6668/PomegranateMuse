"""
Legacy Code Analyzer for Migration Strategy System
Analyzes legacy codebases and identifies modernization opportunities
"""

import re
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

class TechnicalDebtLevel(Enum):
    """Technical debt severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RefactoringType(Enum):
    """Types of refactoring opportunities"""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    MOVE_METHOD = "move_method"
    RENAME = "rename"
    REMOVE_DUPLICATION = "remove_duplication"
    SIMPLIFY_CONDITIONALS = "simplify_conditionals"
    MODERNIZE_SYNTAX = "modernize_syntax"
    IMPROVE_NAMING = "improve_naming"

@dataclass
class TechnicalDebt:
    """Technical debt item"""
    file_path: str
    line_number: int
    debt_type: str
    severity: TechnicalDebtLevel
    description: str
    estimated_fix_time: int  # minutes
    impact_score: float  # 0.0 - 10.0

@dataclass
class RefactoringOpportunity:
    """Refactoring opportunity"""
    file_path: str
    line_range: tuple  # (start, end)
    refactoring_type: RefactoringType
    description: str
    current_code: str
    suggested_approach: str
    estimated_effort: int  # minutes
    benefit_score: float  # 0.0 - 10.0

@dataclass
class LegacyComponent:
    """Legacy component analysis"""
    name: str
    file_paths: List[str]
    language: str
    language_version: str
    lines_of_code: int
    cyclomatic_complexity: float
    technical_debt: List[TechnicalDebt]
    refactoring_opportunities: List[RefactoringOpportunity]
    dependencies: List[str]
    last_modified: datetime
    maintainability_score: float  # 0.0 - 10.0

@dataclass
class ModernizationTarget:
    """Modernization target"""
    component: str
    current_technology: str
    target_technology: str
    modernization_benefits: List[str]
    migration_complexity: str  # low, medium, high
    estimated_effort_days: int
    business_value: float  # 0.0 - 10.0

class LegacyCodeAnalyzer:
    """Analyzes legacy codebases for modernization opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Language-specific patterns for technical debt detection
        self.debt_patterns = {
            "python": {
                "code_smells": [
                    (r"#\s*TODO", "TODO comment", TechnicalDebtLevel.LOW),
                    (r"#\s*FIXME", "FIXME comment", TechnicalDebtLevel.MEDIUM),
                    (r"#\s*HACK", "Hack implementation", TechnicalDebtLevel.HIGH),
                    (r"print\s*\(", "Debug print statement", TechnicalDebtLevel.LOW),
                    (r"except\s*:", "Bare except clause", TechnicalDebtLevel.MEDIUM),
                    (r"global\s+\w+", "Global variable usage", TechnicalDebtLevel.MEDIUM)
                ],
                "outdated_patterns": [
                    (r"import\s+__builtin__", "Python 2 import", TechnicalDebtLevel.HIGH),
                    (r"\.has_key\s*\(", "Python 2 dict method", TechnicalDebtLevel.HIGH),
                    (r"xrange\s*\(", "Python 2 xrange", TechnicalDebtLevel.MEDIUM)
                ]
            },
            "javascript": {
                "code_smells": [
                    (r"//\s*TODO", "TODO comment", TechnicalDebtLevel.LOW),
                    (r"//\s*FIXME", "FIXME comment", TechnicalDebtLevel.MEDIUM),
                    (r"console\.log\s*\(", "Console log statement", TechnicalDebtLevel.LOW),
                    (r"var\s+\w+", "var declaration (use let/const)", TechnicalDebtLevel.MEDIUM),
                    (r"==\s*", "Loose equality (use ===)", TechnicalDebtLevel.LOW)
                ],
                "outdated_patterns": [
                    (r"function\s*\([^)]*\)\s*{", "Function declaration (consider arrow functions)", TechnicalDebtLevel.LOW),
                    (r"\.indexOf\s*\([^)]+\)\s*!==\s*-1", "indexOf check (use includes)", TechnicalDebtLevel.LOW)
                ]
            },
            "java": {
                "code_smells": [
                    (r"//\s*TODO", "TODO comment", TechnicalDebtLevel.LOW),
                    (r"//\s*FIXME", "FIXME comment", TechnicalDebtLevel.MEDIUM),
                    (r"System\.out\.print", "System.out usage", TechnicalDebtLevel.LOW),
                    (r"@SuppressWarnings", "Suppressed warnings", TechnicalDebtLevel.MEDIUM),
                    (r"catch\s*\([^)]*Exception[^)]*\)\s*{\s*}", "Empty catch block", TechnicalDebtLevel.HIGH)
                ],
                "outdated_patterns": [
                    (r"Vector\s*<", "Vector usage (use ArrayList)", TechnicalDebtLevel.MEDIUM),
                    (r"Hashtable\s*<", "Hashtable usage (use HashMap)", TechnicalDebtLevel.MEDIUM),
                    (r"StringBuffer\s+", "StringBuffer (use StringBuilder)", TechnicalDebtLevel.LOW)
                ]
            }
        }
        
        # Refactoring opportunity patterns
        self.refactoring_patterns = {
            "long_method": (r"def\s+\w+.*?(?=def|\Z)", 50),  # Methods longer than 50 lines
            "large_class": (r"class\s+\w+.*?(?=class|\Z)", 500),  # Classes longer than 500 lines
            "duplicate_code": None,  # Requires special analysis
            "long_parameter_list": (r"def\s+\w+\s*\([^)]{100,}\)", 0),  # More than 100 chars in params
            "complex_conditionals": (r"if\s+[^:]{50,}:", 0)  # Complex if conditions
        }
    
    async def analyze_legacy_codebase(self, project_path: Path, language: str) -> Dict[str, LegacyComponent]:
        """Analyze legacy codebase for modernization opportunities"""
        self.logger.info(f"Analyzing legacy codebase: {project_path}")
        
        components = {}
        
        # Find source files
        file_extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".cc", ".cxx", ".h"],
            "csharp": [".cs"]
        }
        
        extensions = file_extensions.get(language, [".py"])
        source_files = []
        
        for ext in extensions:
            source_files.extend(project_path.rglob(f"*{ext}"))
        
        # Analyze each file
        for file_path in source_files:
            try:
                component = await self._analyze_file(file_path, project_path, language)
                if component:
                    components[component.name] = component
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return components
    
    async def _analyze_file(self, file_path: Path, project_path: Path, language: str) -> Optional[LegacyComponent]:
        """Analyze a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # Calculate cyclomatic complexity (simplified)
            complexity = self._calculate_cyclomatic_complexity(content, language)
            
            # Detect technical debt
            technical_debt = self._detect_technical_debt(file_path, content, language)
            
            # Find refactoring opportunities
            refactoring_opportunities = self._find_refactoring_opportunities(file_path, content, language)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content, language)
            
            # Calculate maintainability score
            maintainability = self._calculate_maintainability_score(
                lines_of_code, complexity, len(technical_debt), len(refactoring_opportunities)
            )
            
            # Get file modification time
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            relative_path = str(file_path.relative_to(project_path))
            component_name = self._generate_component_name(file_path, project_path)
            
            return LegacyComponent(
                name=component_name,
                file_paths=[relative_path],
                language=language,
                language_version="unknown",  # Would need more sophisticated detection
                lines_of_code=lines_of_code,
                cyclomatic_complexity=complexity,
                technical_debt=technical_debt,
                refactoring_opportunities=refactoring_opportunities,
                dependencies=dependencies,
                last_modified=last_modified,
                maintainability_score=maintainability
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_cyclomatic_complexity(self, content: str, language: str) -> float:
        """Calculate cyclomatic complexity"""
        # Simplified cyclomatic complexity calculation
        complexity_keywords = {
            "python": ["if", "elif", "while", "for", "except", "and", "or"],
            "javascript": ["if", "else if", "while", "for", "catch", "&&", "||", "?"],
            "java": ["if", "else if", "while", "for", "catch", "&&", "||", "?"],
            "cpp": ["if", "else if", "while", "for", "catch", "&&", "||", "?"]
        }
        
        keywords = complexity_keywords.get(language, complexity_keywords["python"])
        
        complexity = 1  # Base complexity
        for keyword in keywords:
            complexity += content.lower().count(keyword)
        
        # Normalize by lines of code
        lines = len(content.split('\n'))
        return complexity / max(1, lines) * 100
    
    def _detect_technical_debt(self, file_path: Path, content: str, language: str) -> List[TechnicalDebt]:
        """Detect technical debt in code"""
        debt_items = []
        
        if language not in self.debt_patterns:
            return debt_items
        
        patterns = self.debt_patterns[language]
        lines = content.split('\n')
        
        # Check for code smells
        for pattern, description, severity in patterns.get("code_smells", []):
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    debt_items.append(TechnicalDebt(
                        file_path=str(file_path),
                        line_number=line_num,
                        debt_type="code_smell",
                        severity=severity,
                        description=description,
                        estimated_fix_time=self._estimate_debt_fix_time(severity),
                        impact_score=self._calculate_debt_impact(severity)
                    ))
        
        # Check for outdated patterns
        for pattern, description, severity in patterns.get("outdated_patterns", []):
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    debt_items.append(TechnicalDebt(
                        file_path=str(file_path),
                        line_number=line_num,
                        debt_type="outdated_pattern",
                        severity=severity,
                        description=description,
                        estimated_fix_time=self._estimate_debt_fix_time(severity),
                        impact_score=self._calculate_debt_impact(severity)
                    ))
        
        return debt_items
    
    def _find_refactoring_opportunities(self, file_path: Path, content: str, language: str) -> List[RefactoringOpportunity]:
        """Find refactoring opportunities"""
        opportunities = []
        lines = content.split('\n')
        
        # Check for long methods
        if language == "python":
            method_pattern = r"def\s+(\w+)\s*\([^)]*\):"
            current_method = None
            method_start = 0
            indent_level = 0
            
            for line_num, line in enumerate(lines):
                match = re.search(method_pattern, line)
                if match:
                    # End previous method if exists
                    if current_method and line_num - method_start > 30:  # Methods longer than 30 lines
                        opportunities.append(RefactoringOpportunity(
                            file_path=str(file_path),
                            line_range=(method_start, line_num - 1),
                            refactoring_type=RefactoringType.EXTRACT_METHOD,
                            description=f"Method '{current_method}' is too long ({line_num - method_start} lines)",
                            current_code="",  # Would extract actual code
                            suggested_approach="Break into smaller, more focused methods",
                            estimated_effort=60,  # 1 hour
                            benefit_score=7.0
                        ))
                    
                    current_method = match.group(1)
                    method_start = line_num
                    indent_level = len(line) - len(line.lstrip())
        
        # Check for complex conditionals
        complex_if_pattern = r"if\s+.{60,}:"  # If statements longer than 60 characters
        for line_num, line in enumerate(lines, 1):
            if re.search(complex_if_pattern, line):
                opportunities.append(RefactoringOpportunity(
                    file_path=str(file_path),
                    line_range=(line_num, line_num),
                    refactoring_type=RefactoringType.SIMPLIFY_CONDITIONALS,
                    description="Complex conditional statement",
                    current_code=line.strip(),
                    suggested_approach="Extract condition to a well-named method",
                    estimated_effort=15,
                    benefit_score=5.0
                ))
        
        return opportunities
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from code"""
        dependencies = []
        
        import_patterns = {
            "python": [r"import\s+(\w+)", r"from\s+(\w+)\s+import"],
            "javascript": [r"import\s+.*from\s+['\"]([^'\"]+)['\"]", r"require\s*\(\s*['\"]([^'\"]+)['\"]"],
            "java": [r"import\s+([\w.]+);"],
            "cpp": [r"#include\s*[<\"]([^>\"]+)[>\"]"]
        }
        
        patterns = import_patterns.get(language, import_patterns["python"])
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _calculate_maintainability_score(self, lines_of_code: int, complexity: float,
                                       debt_count: int, refactoring_count: int) -> float:
        """Calculate maintainability score (0-10)"""
        score = 10.0
        
        # Penalize large files
        if lines_of_code > 500:
            score -= 2.0
        elif lines_of_code > 200:
            score -= 1.0
        
        # Penalize high complexity
        if complexity > 10:
            score -= 3.0
        elif complexity > 5:
            score -= 1.5
        
        # Penalize technical debt
        score -= min(3.0, debt_count * 0.2)
        
        # Penalize refactoring needs
        score -= min(2.0, refactoring_count * 0.1)
        
        return max(0.0, score)
    
    def _estimate_debt_fix_time(self, severity: TechnicalDebtLevel) -> int:
        """Estimate time to fix technical debt (in minutes)"""
        time_estimates = {
            TechnicalDebtLevel.LOW: 5,
            TechnicalDebtLevel.MEDIUM: 15,
            TechnicalDebtLevel.HIGH: 60,
            TechnicalDebtLevel.CRITICAL: 240
        }
        return time_estimates.get(severity, 15)
    
    def _calculate_debt_impact(self, severity: TechnicalDebtLevel) -> float:
        """Calculate impact score for technical debt"""
        impact_scores = {
            TechnicalDebtLevel.LOW: 2.0,
            TechnicalDebtLevel.MEDIUM: 5.0,
            TechnicalDebtLevel.HIGH: 8.0,
            TechnicalDebtLevel.CRITICAL: 10.0
        }
        return impact_scores.get(severity, 5.0)
    
    def _generate_component_name(self, file_path: Path, project_path: Path) -> str:
        """Generate component name from file path"""
        relative_path = file_path.relative_to(project_path)
        name_parts = list(relative_path.parts[:-1])
        filename = relative_path.stem
        
        if filename not in ['index', 'main', '__init__']:
            name_parts.append(filename)
        
        return '.'.join(name_parts) if name_parts else filename
    
    def generate_modernization_targets(self, components: Dict[str, LegacyComponent]) -> List[ModernizationTarget]:
        """Generate modernization targets based on analysis"""
        targets = []
        
        for component_name, component in components.items():
            # High-debt components are good modernization targets
            high_debt_count = len([d for d in component.technical_debt 
                                 if d.severity in [TechnicalDebtLevel.HIGH, TechnicalDebtLevel.CRITICAL]])
            
            if high_debt_count > 3 or component.maintainability_score < 5.0:
                modernization_benefits = []
                
                if component.maintainability_score < 5.0:
                    modernization_benefits.append("Improved maintainability")
                
                if high_debt_count > 0:
                    modernization_benefits.append("Reduced technical debt")
                
                if component.cyclomatic_complexity > 10:
                    modernization_benefits.append("Simplified complexity")
                
                # Estimate migration complexity
                complexity = "low"
                if component.lines_of_code > 1000 or len(component.dependencies) > 10:
                    complexity = "high"
                elif component.lines_of_code > 500 or len(component.dependencies) > 5:
                    complexity = "medium"
                
                # Estimate effort
                effort_days = max(1, component.lines_of_code // 200)  # Rough estimate
                
                # Calculate business value
                business_value = 10.0 - component.maintainability_score
                
                target = ModernizationTarget(
                    component=component_name,
                    current_technology=f"{component.language} (legacy)",
                    target_technology=f"Modern {component.language}",
                    modernization_benefits=modernization_benefits,
                    migration_complexity=complexity,
                    estimated_effort_days=effort_days,
                    business_value=business_value
                )
                
                targets.append(target)
        
        # Sort by business value
        targets.sort(key=lambda x: x.business_value, reverse=True)
        
        return targets

async def analyze_legacy_codebase(project_path: Path, language: str) -> Dict[str, LegacyComponent]:
    """Convenience function to analyze legacy codebase"""
    analyzer = LegacyCodeAnalyzer()
    return await analyzer.analyze_legacy_codebase(project_path, language)