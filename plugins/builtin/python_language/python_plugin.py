"""
Python Language Plugin for MyndraComposer
Provides Python language support with AST parsing and analysis
"""

import ast
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ...plugin_interface import ILanguagePlugin, BasePluginInterface
from ...plugin_manager import BasePlugin

class PythonLanguagePlugin(BasePlugin, ILanguagePlugin):
    """Python language support plugin"""
    
    def __init__(self):
        super().__init__("python_language", "1.0.0")
        self.max_complexity = 10
        self.strict_mode = False
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return [".py", ".pyw", ".py3"]
    
    def parse_code(self, code: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Parse Python code and extract structure"""
        try:
            tree = ast.parse(code)
            
            # Extract classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "bases": [self._get_name(base) for base in node.bases],
                        "decorators": [self._get_name(dec) for dec in node.decorator_list]
                    })
            
            # Extract functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    args = [arg.arg for arg in node.args.args]
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": args,
                        "decorators": [self._get_name(dec) for dec in node.decorator_list],
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    })
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "type": "import"
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append({
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "type": "from_import"
                        })
            
            # Extract variables/constants
            variables = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append({
                                "name": target.id,
                                "line": node.lineno,
                                "type": self._infer_type(node.value)
                            })
            
            return {
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "variables": variables,
                "ast_tree": ast.dump(tree) if file_path else None
            }
            
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "line": getattr(e, 'lineno', 0),
                "offset": getattr(e, 'offset', 0)
            }
        except Exception as e:
            return {"error": f"Parse error: {e}"}
    
    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return "unknown"
    
    def _is_method(self, func_node, tree) -> bool:
        """Check if function is a method inside a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _infer_type(self, value_node) -> str:
        """Infer variable type from assignment value"""
        if isinstance(value_node, ast.Constant):
            return type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            return "list"
        elif isinstance(value_node, ast.Dict):
            return "dict"
        elif isinstance(value_node, ast.Set):
            return "set"
        elif isinstance(value_node, ast.Tuple):
            return "tuple"
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                return value_node.func.id
        return "unknown"
    
    def validate_syntax(self, code: str) -> List[str]:
        """Validate Python code syntax"""
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {e}")
        
        # Additional validation in strict mode
        if self.strict_mode:
            # Check for common issues
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                # Check for tabs vs spaces
                if '\t' in line and '    ' in line:
                    errors.append(f"Mixed tabs and spaces at line {i}")
                
                # Check line length
                if len(line) > 120:
                    errors.append(f"Line too long ({len(line)} chars) at line {i}")
        
        return errors
    
    def get_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from Python code"""
        dependencies = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level module name
                        module = alias.name.split('.')[0]
                        dependencies.add(module)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get top-level module name
                        module = node.module.split('.')[0]
                        dependencies.add(module)
            
            # Filter out standard library modules
            stdlib_modules = {
                'os', 'sys', 'json', 'csv', 'datetime', 'time', 'math', 're',
                'collections', 'itertools', 'functools', 'logging', 'threading',
                'pathlib', 'urllib', 'http', 'email', 'xml', 'html', 'sqlite3',
                'pickle', 'base64', 'hashlib', 'uuid', 'random', 'typing'
            }
            
            external_deps = dependencies - stdlib_modules
            return list(external_deps)
            
        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {e}")
            return []
    
    def estimate_complexity(self, code: str) -> float:
        """Estimate code complexity using cyclomatic complexity"""
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity(tree)
            
            # Normalize to 0-1 scale (assuming max complexity of 20)
            normalized = min(complexity / 20.0, 1.0)
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {e}")
            return 0.0
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1
            elif isinstance(child, ast.DictComp):
                complexity += 1
            elif isinstance(child, ast.SetComp):
                complexity += 1
            elif isinstance(child, ast.GeneratorExp):
                complexity += 1
        
        return complexity
    
    def generate_ast(self, code: str) -> Dict[str, Any]:
        """Generate Abstract Syntax Tree"""
        try:
            tree = ast.parse(code)
            
            return {
                "ast_dump": ast.dump(tree, indent=2),
                "node_count": len(list(ast.walk(tree))),
                "depth": self._calculate_depth(tree),
                "success": True
            }
            
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "line": getattr(e, 'lineno', 0),
                "success": False
            }
        except Exception as e:
            return {
                "error": f"AST generation error: {e}",
                "success": False
            }
    
    def _calculate_depth(self, node, current_depth=0) -> int:
        """Calculate maximum depth of AST"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def on_config_changed(self, config: Dict[str, Any]) -> bool:
        """Handle configuration changes"""
        self.max_complexity = config.get("max_complexity", 10)
        self.strict_mode = config.get("strict_mode", False)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration"""
        errors = []
        
        if "max_complexity" in config:
            if not isinstance(config["max_complexity"], int) or config["max_complexity"] < 1:
                errors.append("max_complexity must be a positive integer")
        
        if "strict_mode" in config:
            if not isinstance(config["strict_mode"], bool):
                errors.append("strict_mode must be a boolean")
        
        return errors
    
    def get_code_metrics(self, code: str) -> Dict[str, Any]:
        """Get detailed code metrics"""
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            # Count various elements
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            
            # Line counts
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            blank_lines = total_lines - code_lines - comment_lines
            
            return {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "complexity": self._calculate_complexity(tree),
                "ast_depth": self._calculate_depth(tree)
            }
            
        except Exception as e:
            return {"error": f"Metrics calculation error: {e}"}

# Plugin entry point
def create_plugin():
    """Create plugin instance"""
    return PythonLanguagePlugin()