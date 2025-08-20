"""
Analysis Engine Unit Tests
Tests for code analysis functionality
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

class MockAnalysisEngine:
    """Mock analysis engine for testing"""
    def __init__(self):
        self.analyzers = {}
        self.results_cache = {}
    
    def register_analyzer(self, language, analyzer):
        if language not in self.analyzers:
            self.analyzers[language] = []
        self.analyzers[language].append(analyzer)
    
    def analyze_code(self, code, language, analysis_type="full"):
        cache_key = f"{hash(code)}_{language}_{analysis_type}"
        
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        result = self._perform_analysis(code, language, analysis_type)
        self.results_cache[cache_key] = result
        return result
    
    def _perform_analysis(self, code, language, analysis_type):
        # Mock analysis results based on code content
        result = {
            "language": language,
            "analysis_type": analysis_type,
            "complexity": self._calculate_complexity(code),
            "quality_score": self._calculate_quality(code),
            "patterns": self._detect_patterns(code),
            "dependencies": self._extract_dependencies(code, language),
            "metrics": self._calculate_metrics(code),
            "issues": self._detect_issues(code, language)
        }
        return result
    
    def _calculate_complexity(self, code):
        # Simple complexity calculation based on control structures
        complexity_keywords = ['if', 'for', 'while', 'try', 'except', 'catch', 'switch']
        lines = code.split('\n')
        complexity = 1  # Base complexity
        
        for line in lines:
            for keyword in complexity_keywords:
                if keyword in line.lower():
                    complexity += 1
        
        return complexity
    
    def _calculate_quality(self, code):
        # Simple quality score calculation
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        if total_lines == 0:
            return 0.0
        
        # Quality factors
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        line_length_score = 1.0 - sum(1 for line in lines if len(line) > 120) / total_lines
        
        # Combined quality score
        quality = (comment_ratio * 0.3 + line_length_score * 0.7)
        return min(quality * 10, 10.0)  # Scale to 0-10
    
    def _detect_patterns(self, code):
        # Detect common patterns
        patterns = []
        
        if 'class' in code.lower():
            patterns.append("object_oriented")
        if 'def ' in code or 'function ' in code:
            patterns.append("functional")
        if 'import ' in code or '#include' in code:
            patterns.append("modular")
        if 'try:' in code or 'try {' in code:
            patterns.append("error_handling")
        
        return patterns
    
    def _extract_dependencies(self, code, language):
        # Extract dependencies based on language
        dependencies = []
        lines = code.split('\n')
        
        if language == "python":
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    module = line.replace('import ', '').split(' as ')[0].strip()
                    dependencies.append(module)
                elif line.startswith('from '):
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        module = parts[1]
                        dependencies.append(module)
        
        elif language == "javascript":
            for line in lines:
                if 'require(' in line or 'import ' in line:
                    # Simple extraction for testing
                    if 'require(' in line:
                        start = line.find("require('") + 9
                        if start > 8:
                            end = line.find("'", start)
                            if end > start:
                                dependencies.append(line[start:end])
        
        return list(set(dependencies))  # Remove duplicates
    
    def _calculate_metrics(self, code):
        # Calculate code metrics
        lines = code.split('\n')
        
        return {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "function_count": code.count('def ') + code.count('function '),
            "class_count": code.count('class ')
        }
    
    def _detect_issues(self, code, language):
        # Detect potential issues
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "line": i,
                    "message": f"Line too long ({len(line)} chars)"
                })
            
            # Check for potential security issues (very basic)
            if 'eval(' in line:
                issues.append({
                    "type": "security",
                    "severity": "high",
                    "line": i,
                    "message": "Use of eval() function"
                })
            
            # Check for TODO comments
            if 'TODO' in line.upper():
                issues.append({
                    "type": "maintenance",
                    "severity": "low",
                    "line": i,
                    "message": "TODO comment found"
                })
        
        return issues

class AnalysisUnitTests(unittest.TestCase):
    """Unit tests for analysis engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.analysis_engine = MockAnalysisEngine()
        
        # Sample code for testing
        self.python_code = """
import os
import sys
from collections import defaultdict

# TODO: Add error handling
def fibonacci(n):
    \"\"\"Calculate fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = defaultdict(list)
    
    def add(self, a, b):
        result = a + b
        self.history['add'].append((a, b, result))
        return result
    
    def divide(self, a, b):
        try:
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        except ValueError as e:
            print(f"Error: {e}")
            return None

# This line is intentionally very long to test line length detection and style checking functionality that should trigger warnings
result = Calculator().add(5, 3)
        """
        
        self.javascript_code = """
const fs = require('fs');
const path = require('path');

// TODO: Optimize this function
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class MathUtils {
    constructor() {
        this.cache = new Map();
    }
    
    power(base, exponent) {
        const key = `${base}_${exponent}`;
        if (this.cache.has(key)) {
            return this.cache.get(key);
        }
        
        const result = Math.pow(base, exponent);
        this.cache.set(key, result);
        return result;
    }
    
    // Dangerous function for testing
    evaluate(expression) {
        return eval(expression);
    }
}
        """
    
    def test_complexity_calculation(self):
        """Test code complexity calculation"""
        # Test Python code complexity
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        complexity = result["complexity"]
        
        # Should detect if, try/except structures
        self.assertGreater(complexity, 1)
        self.assertIsInstance(complexity, int)
        
        # Test simple code with no control structures
        simple_code = "x = 5\ny = 10\nprint(x + y)"
        simple_result = self.analysis_engine.analyze_code(simple_code, "python")
        self.assertEqual(simple_result["complexity"], 1)  # Base complexity
    
    def test_quality_assessment(self):
        """Test code quality assessment"""
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        quality_score = result["quality_score"]
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 10.0)
        
        # Test code with good documentation
        documented_code = '''
"""Module docstring"""

def well_documented_function():
    """Function with proper documentation"""
    # Inline comment
    return "result"
        '''
        
        doc_result = self.analysis_engine.analyze_code(documented_code, "python")
        doc_quality = doc_result["quality_score"]
        
        # Documented code should have higher quality score
        self.assertGreater(doc_quality, 0)
    
    def test_pattern_detection(self):
        """Test pattern detection in code"""
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        patterns = result["patterns"]
        
        self.assertIsInstance(patterns, list)
        self.assertIn("object_oriented", patterns)  # Has class
        self.assertIn("functional", patterns)  # Has functions
        self.assertIn("modular", patterns)  # Has imports
        self.assertIn("error_handling", patterns)  # Has try/except
        
        # Test JavaScript patterns
        js_result = self.analysis_engine.analyze_code(self.javascript_code, "javascript")
        js_patterns = js_result["patterns"]
        
        self.assertIn("object_oriented", js_patterns)  # Has class
        self.assertIn("functional", js_patterns)  # Has functions
    
    def test_dependency_extraction(self):
        """Test dependency extraction"""
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        dependencies = result["dependencies"]
        
        self.assertIsInstance(dependencies, list)
        self.assertIn("os", dependencies)
        self.assertIn("sys", dependencies)
        self.assertIn("collections", dependencies)
        
        # Test JavaScript dependencies
        js_result = self.analysis_engine.analyze_code(self.javascript_code, "javascript")
        js_dependencies = js_result["dependencies"]
        
        self.assertIn("fs", js_dependencies)
        self.assertIn("path", js_dependencies)
    
    def test_metrics_calculation(self):
        """Test code metrics calculation"""
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        metrics = result["metrics"]
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_lines", metrics)
        self.assertIn("code_lines", metrics)
        self.assertIn("comment_lines", metrics)
        self.assertIn("blank_lines", metrics)
        self.assertIn("function_count", metrics)
        self.assertIn("class_count", metrics)
        
        # Verify metrics make sense
        self.assertGreater(metrics["total_lines"], 0)
        self.assertGreater(metrics["code_lines"], 0)
        self.assertGreater(metrics["function_count"], 0)
        self.assertGreater(metrics["class_count"], 0)
        
        # Total lines should be sum of code, comment, and blank lines
        calculated_total = (metrics["code_lines"] + 
                          metrics["comment_lines"] + 
                          metrics["blank_lines"])
        self.assertEqual(metrics["total_lines"], calculated_total)
    
    def test_issue_detection(self):
        """Test issue detection"""
        result = self.analysis_engine.analyze_code(self.python_code, "python")
        issues = result["issues"]
        
        self.assertIsInstance(issues, list)
        
        # Should detect long line
        long_line_issues = [issue for issue in issues if "Line too long" in issue["message"]]
        self.assertGreater(len(long_line_issues), 0)
        
        # Should detect TODO comment
        todo_issues = [issue for issue in issues if "TODO comment" in issue["message"]]
        self.assertGreater(len(todo_issues), 0)
        
        # Test security issue detection
        js_result = self.analysis_engine.analyze_code(self.javascript_code, "javascript")
        js_issues = js_result["issues"]
        
        # Should detect eval() usage
        eval_issues = [issue for issue in js_issues if "eval()" in issue["message"]]
        self.assertGreater(len(eval_issues), 0)
        
        # Verify issue structure
        for issue in issues:
            self.assertIn("type", issue)
            self.assertIn("severity", issue)
            self.assertIn("line", issue)
            self.assertIn("message", issue)
    
    def test_analysis_caching(self):
        """Test analysis result caching"""
        # Analyze same code twice
        result1 = self.analysis_engine.analyze_code(self.python_code, "python")
        result2 = self.analysis_engine.analyze_code(self.python_code, "python")
        
        # Results should be identical (from cache)
        self.assertEqual(result1, result2)
        
        # Verify cache is working
        cache_size_before = len(self.analysis_engine.results_cache)
        self.analysis_engine.analyze_code("new code", "python")
        cache_size_after = len(self.analysis_engine.results_cache)
        
        self.assertEqual(cache_size_after, cache_size_before + 1)
    
    def test_different_analysis_types(self):
        """Test different analysis types"""
        # Test different analysis types
        full_result = self.analysis_engine.analyze_code(self.python_code, "python", "full")
        quick_result = self.analysis_engine.analyze_code(self.python_code, "python", "quick")
        security_result = self.analysis_engine.analyze_code(self.python_code, "python", "security")
        
        # All should return results but may differ in depth
        self.assertIn("analysis_type", full_result)
        self.assertIn("analysis_type", quick_result)
        self.assertIn("analysis_type", security_result)
        
        self.assertEqual(full_result["analysis_type"], "full")
        self.assertEqual(quick_result["analysis_type"], "quick")
        self.assertEqual(security_result["analysis_type"], "security")
    
    def test_empty_code_handling(self):
        """Test handling of empty or invalid code"""
        # Test empty code
        empty_result = self.analysis_engine.analyze_code("", "python")
        self.assertIsInstance(empty_result, dict)
        self.assertEqual(empty_result["complexity"], 1)  # Base complexity
        
        # Test whitespace-only code
        whitespace_result = self.analysis_engine.analyze_code("   \n  \n  ", "python")
        self.assertIsInstance(whitespace_result, dict)
        
        # Test code with only comments
        comment_result = self.analysis_engine.analyze_code("# Just a comment\n# Another comment", "python")
        self.assertIsInstance(comment_result, dict)
        metrics = comment_result["metrics"]
        self.assertEqual(metrics["code_lines"], 0)
        self.assertEqual(metrics["comment_lines"], 2)
    
    def test_language_specific_analysis(self):
        """Test language-specific analysis features"""
        # Python-specific analysis
        python_result = self.analysis_engine.analyze_code(self.python_code, "python")
        python_deps = python_result["dependencies"]
        
        # Should detect Python imports
        self.assertIn("os", python_deps)
        self.assertIn("sys", python_deps)
        
        # JavaScript-specific analysis
        js_result = self.analysis_engine.analyze_code(self.javascript_code, "javascript")
        js_deps = js_result["dependencies"]
        
        # Should detect require statements
        self.assertIn("fs", js_deps)
        self.assertIn("path", js_deps)
    
    def test_analyzer_registration(self):
        """Test analyzer registration system"""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value={"custom": "result"})
        
        # Register analyzer
        self.analysis_engine.register_analyzer("python", mock_analyzer)
        
        # Verify registration
        self.assertIn("python", self.analysis_engine.analyzers)
        self.assertIn(mock_analyzer, self.analysis_engine.analyzers["python"])
    
    def test_concurrent_analysis(self):
        """Test concurrent analysis operations"""
        import threading
        
        results = []
        
        def analyze_code(code_snippet, language):
            result = self.analysis_engine.analyze_code(code_snippet, language)
            results.append(result)
        
        # Run multiple analyses concurrently
        threads = []
        test_codes = [
            ("print('hello')", "python"),
            ("console.log('world')", "javascript"),
            ("def test(): pass", "python")
        ]
        
        for code, lang in test_codes:
            thread = threading.Thread(target=analyze_code, args=(code, lang))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all analyses completed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn("complexity", result)
            self.assertIn("quality_score", result)