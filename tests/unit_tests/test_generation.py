"""
Code Generation Unit Tests
Tests for code generation functionality
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

class MockCodeGenerator:
    """Mock code generator for testing"""
    def __init__(self):
        self.generators = {}
        self.generation_cache = {}
        self.supported_languages = {
            "python": ["pomegranate", "rust", "go", "typescript"],
            "javascript": ["typescript", "python", "pomegranate"],
            "java": ["kotlin", "scala", "python", "pomegranate"],
            "cpp": ["rust", "go", "python", "pomegranate"]
        }
    
    def register_generator(self, source_lang, target_lang, generator):
        key = f"{source_lang}_to_{target_lang}"
        self.generators[key] = generator
    
    def generate_code(self, source_code, source_language, target_language, context=None):
        cache_key = f"{hash(source_code)}_{source_language}_{target_language}"
        
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        result = self._perform_generation(source_code, source_language, target_language, context or {})
        self.generation_cache[cache_key] = result
        return result
    
    def _perform_generation(self, source_code, source_language, target_language, context):
        # Mock generation based on language pairs
        if target_language not in self.supported_languages.get(source_language, []):
            return {
                "success": False,
                "error": f"Translation from {source_language} to {target_language} not supported",
                "generated_code": "",
                "metadata": {}
            }
        
        # Simple mock generation
        generated_code = self._mock_translate(source_code, source_language, target_language)
        
        return {
            "success": True,
            "generated_code": generated_code,
            "source_language": source_language,
            "target_language": target_language,
            "generation_time": 1.5,  # Mock time
            "confidence": self._calculate_confidence(source_code, target_language),
            "metadata": {
                "translation_method": "mock",
                "preserved_functionality": True,
                "optimization_level": context.get("optimization_level", "standard"),
                "style_preferences": context.get("style_preferences", {})
            },
            "warnings": self._generate_warnings(source_code, source_language, target_language),
            "statistics": self._calculate_statistics(source_code, generated_code)
        }
    
    def _mock_translate(self, source_code, source_language, target_language):
        """Mock code translation logic"""
        if source_language == "python" and target_language == "pomegranate":
            return self._python_to_pomegranate(source_code)
        elif source_language == "python" and target_language == "rust":
            return self._python_to_rust(source_code)
        elif source_language == "javascript" and target_language == "typescript":
            return self._javascript_to_typescript(source_code)
        else:
            # Generic translation
            return f"// Translated from {source_language} to {target_language}\n" + source_code
    
    def _python_to_pomegranate(self, code):
        """Mock Python to Pomegranate translation"""
        lines = code.split('\n')
        translated_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                # Convert function definition
                func_name = line.split('(')[0].replace('def ', '')
                params = line.split('(')[1].split(')')[0] if '(' in line else ''
                translated_lines.append(f"fn {func_name}({params}) {{")
            elif line.startswith('class '):
                # Convert class definition
                class_name = line.replace('class ', '').replace(':', '')
                translated_lines.append(f"type {class_name} = {{")
            elif line.startswith('if '):
                # Convert if statement
                condition = line.replace('if ', '').replace(':', '')
                translated_lines.append(f"if {condition} {{")
            elif line.startswith('return '):
                # Convert return statement
                value = line.replace('return ', '')
                translated_lines.append(f"    return {value}")
            elif line.startswith('print('):
                # Convert print statement
                content = line.replace('print(', '').replace(')', '')
                translated_lines.append(f"    log({content})")
            else:
                # Keep other lines as is with some formatting
                if line:
                    translated_lines.append(f"    {line}")
                else:
                    translated_lines.append("")
        
        return '\n'.join(translated_lines)
    
    def _python_to_rust(self, code):
        """Mock Python to Rust translation"""
        lines = code.split('\n')
        translated_lines = ['// Rust translation']
        
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                translated_lines.append(f"fn {func_name}() {{")
            elif line.startswith('print('):
                content = line.replace('print(', '').replace(')', '')
                translated_lines.append(f"    println!({content});")
            elif line:
                translated_lines.append(f"    // {line}")
        
        translated_lines.append("}")
        return '\n'.join(translated_lines)
    
    def _javascript_to_typescript(self, code):
        """Mock JavaScript to TypeScript translation"""
        # Simple mock: add type annotations
        lines = code.split('\n')
        translated_lines = []
        
        for line in lines:
            if 'function ' in line:
                # Add return type annotation
                line = line.replace(')', '): any')
            elif 'const ' in line or 'let ' in line or 'var ' in line:
                # Add type annotation
                if '=' in line:
                    line = line.replace('=', ': any =')
            
            translated_lines.append(line)
        
        return '\n'.join(translated_lines)
    
    def _calculate_confidence(self, source_code, target_language):
        """Calculate generation confidence score"""
        # Mock confidence based on code complexity
        lines = len(source_code.split('\n'))
        complexity_keywords = ['if', 'for', 'while', 'try', 'class', 'function', 'def']
        complexity = sum(1 for line in source_code.split('\n') 
                        for keyword in complexity_keywords if keyword in line.lower())
        
        # Higher complexity = lower confidence
        base_confidence = 0.9
        complexity_penalty = min(complexity * 0.05, 0.3)
        
        return max(base_confidence - complexity_penalty, 0.5)
    
    def _generate_warnings(self, source_code, source_language, target_language):
        """Generate warnings for the translation"""
        warnings = []
        
        # Check for potential issues
        if 'eval(' in source_code:
            warnings.append({
                "type": "security",
                "message": "eval() function detected - manual review required",
                "severity": "high"
            })
        
        if len(source_code.split('\n')) > 100:
            warnings.append({
                "type": "complexity",
                "message": "Large source file - review generated code carefully",
                "severity": "medium"
            })
        
        if source_language == "python" and target_language == "rust":
            warnings.append({
                "type": "paradigm",
                "message": "Memory management patterns may need adjustment",
                "severity": "medium"
            })
        
        return warnings
    
    def _calculate_statistics(self, source_code, generated_code):
        """Calculate generation statistics"""
        source_lines = len([line for line in source_code.split('\n') if line.strip()])
        generated_lines = len([line for line in generated_code.split('\n') if line.strip()])
        
        return {
            "source_lines": source_lines,
            "generated_lines": generated_lines,
            "line_ratio": generated_lines / source_lines if source_lines > 0 else 0,
            "estimated_effort": "low" if source_lines < 50 else "medium" if source_lines < 200 else "high"
        }
    
    def validate_generated_code(self, code, language):
        """Validate generated code syntax"""
        # Mock validation
        if not code.strip():
            return {"valid": False, "errors": ["Empty code generated"]}
        
        # Simple syntax checks
        errors = []
        if language == "python":
            # Check for basic Python syntax
            if code.count('(') != code.count(')'):
                errors.append("Mismatched parentheses")
            if 'def ' in code and not any(line.strip().startswith('def ') for line in code.split('\n')):
                errors.append("Invalid function definition")
        
        elif language == "rust":
            # Check for basic Rust syntax
            if code.count('{') != code.count('}'):
                errors.append("Mismatched braces")
        
        return {"valid": len(errors) == 0, "errors": errors}

class GenerationUnitTests(unittest.TestCase):
    """Unit tests for code generation"""
    
    def setUp(self):
        """Set up test environment"""
        self.generator = MockCodeGenerator()
        
        # Sample source codes for testing
        self.python_function = """
def fibonacci(n):
    \"\"\"Calculate fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
        """
        
        self.python_class = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
        """
        
        self.javascript_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
console.log(doubled);
        """
    
    def test_basic_code_generation(self):
        """Test basic code generation functionality"""
        result = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        
        self.assertTrue(result["success"])
        self.assertIn("generated_code", result)
        self.assertNotEqual(result["generated_code"], "")
        self.assertEqual(result["source_language"], "python")
        self.assertEqual(result["target_language"], "pomegranate")
    
    def test_unsupported_language_pair(self):
        """Test handling of unsupported language pairs"""
        result = self.generator.generate_code(
            self.python_function, "python", "cobol"
        )
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("not supported", result["error"])
    
    def test_python_to_pomegranate_translation(self):
        """Test Python to Pomegranate translation"""
        result = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        
        self.assertTrue(result["success"])
        generated = result["generated_code"]
        
        # Check for Pomegranate syntax elements
        self.assertIn("fn fibonacci", generated)
        self.assertIn("fn factorial", generated)
        self.assertIn("return", generated)
    
    def test_python_to_rust_translation(self):
        """Test Python to Rust translation"""
        result = self.generator.generate_code(
            self.python_function, "python", "rust"
        )
        
        self.assertTrue(result["success"])
        generated = result["generated_code"]
        
        # Check for Rust syntax elements
        self.assertIn("fn ", generated)
        self.assertIn("println!", generated)
        self.assertIn("//", generated)  # Comments
    
    def test_javascript_to_typescript_translation(self):
        """Test JavaScript to TypeScript translation"""
        result = self.generator.generate_code(
            self.javascript_code, "javascript", "typescript"
        )
        
        self.assertTrue(result["success"])
        generated = result["generated_code"]
        
        # Check for TypeScript type annotations
        self.assertIn(": any", generated)
    
    def test_generation_metadata(self):
        """Test generation metadata"""
        result = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        
        self.assertIn("metadata", result)
        metadata = result["metadata"]
        
        self.assertIn("translation_method", metadata)
        self.assertIn("preserved_functionality", metadata)
        self.assertIn("optimization_level", metadata)
        self.assertTrue(metadata["preserved_functionality"])
    
    def test_confidence_scoring(self):
        """Test confidence scoring"""
        # Simple code should have high confidence
        simple_result = self.generator.generate_code(
            "print('hello')", "python", "pomegranate"
        )
        
        # Complex code should have lower confidence
        complex_code = """
def complex_function(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    try:
                        process_item(item)
                    except Exception as e:
                        handle_error(e)
            elif callable(value):
                value()
        """
        
        complex_result = self.generator.generate_code(
            complex_code, "python", "pomegranate"
        )
        
        self.assertIn("confidence", simple_result)
        self.assertIn("confidence", complex_result)
        self.assertGreater(simple_result["confidence"], complex_result["confidence"])
    
    def test_warning_generation(self):
        """Test warning generation for problematic code"""
        # Code with security issue
        security_code = """
def dangerous_function(user_input):
    return eval(user_input)
        """
        
        result = self.generator.generate_code(
            security_code, "python", "pomegranate"
        )
        
        self.assertIn("warnings", result)
        warnings = result["warnings"]
        
        # Should have security warning
        security_warnings = [w for w in warnings if w["type"] == "security"]
        self.assertGreater(len(security_warnings), 0)
        
        # Check warning structure
        for warning in warnings:
            self.assertIn("type", warning)
            self.assertIn("message", warning)
            self.assertIn("severity", warning)
    
    def test_generation_statistics(self):
        """Test generation statistics calculation"""
        result = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        
        self.assertIn("statistics", result)
        stats = result["statistics"]
        
        self.assertIn("source_lines", stats)
        self.assertIn("generated_lines", stats)
        self.assertIn("line_ratio", stats)
        self.assertIn("estimated_effort", stats)
        
        # Verify statistics make sense
        self.assertGreater(stats["source_lines"], 0)
        self.assertGreater(stats["generated_lines"], 0)
        self.assertGreater(stats["line_ratio"], 0)
    
    def test_generation_caching(self):
        """Test generation result caching"""
        # Generate same code twice
        result1 = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        result2 = self.generator.generate_code(
            self.python_function, "python", "pomegranate"
        )
        
        # Results should be identical (from cache)
        self.assertEqual(result1, result2)
        
        # Verify cache is working
        cache_size_before = len(self.generator.generation_cache)
        self.generator.generate_code("new code", "python", "rust")
        cache_size_after = len(self.generator.generation_cache)
        
        self.assertEqual(cache_size_after, cache_size_before + 1)
    
    def test_context_handling(self):
        """Test context parameter handling"""
        context = {
            "optimization_level": "aggressive",
            "style_preferences": {
                "use_tabs": True,
                "max_line_length": 80
            },
            "preserve_comments": True
        }
        
        result = self.generator.generate_code(
            self.python_function, "python", "pomegranate", context
        )
        
        self.assertTrue(result["success"])
        metadata = result["metadata"]
        
        self.assertEqual(metadata["optimization_level"], "aggressive")
        self.assertIn("style_preferences", metadata)
    
    def test_code_validation(self):
        """Test generated code validation"""
        # Valid Python code
        valid_python = "def test():\n    return 42"
        validation_result = self.generator.validate_generated_code(valid_python, "python")
        
        self.assertTrue(validation_result["valid"])
        self.assertEqual(len(validation_result["errors"]), 0)
        
        # Invalid Python code (mismatched parentheses)
        invalid_python = "def test():\n    return func(arg"
        invalid_result = self.generator.validate_generated_code(invalid_python, "python")
        
        self.assertFalse(invalid_result["valid"])
        self.assertGreater(len(invalid_result["errors"]), 0)
        
        # Empty code
        empty_result = self.generator.validate_generated_code("", "python")
        
        self.assertFalse(empty_result["valid"])
        self.assertIn("Empty code generated", invalid_result["errors"][0] if invalid_result["errors"] else "")
    
    def test_class_translation(self):
        """Test class-based code translation"""
        result = self.generator.generate_code(
            self.python_class, "python", "pomegranate"
        )
        
        self.assertTrue(result["success"])
        generated = result["generated_code"]
        
        # Check for class translation
        self.assertIn("type Calculator", generated)
    
    def test_large_code_handling(self):
        """Test handling of large code files"""
        # Create large code sample
        large_code = "def function_{}():\n    return {}\n".format(
            *[(i, i) for i in range(150)]
        )
        
        result = self.generator.generate_code(large_code, "python", "pomegranate")
        
        self.assertTrue(result["success"])
        
        # Should have complexity warning
        warnings = result["warnings"]
        complexity_warnings = [w for w in warnings if w["type"] == "complexity"]
        self.assertGreater(len(complexity_warnings), 0)
        
        # Should have high effort estimate
        stats = result["statistics"]
        self.assertIn(stats["estimated_effort"], ["medium", "high"])
    
    def test_empty_code_handling(self):
        """Test handling of empty or minimal code"""
        # Empty code
        empty_result = self.generator.generate_code("", "python", "pomegranate")
        self.assertTrue(empty_result["success"])
        
        # Whitespace only
        whitespace_result = self.generator.generate_code("   \n  \n", "python", "pomegranate")
        self.assertTrue(whitespace_result["success"])
        
        # Single line
        single_result = self.generator.generate_code("x = 5", "python", "pomegranate")
        self.assertTrue(single_result["success"])
    
    def test_concurrent_generation(self):
        """Test concurrent code generation"""
        import threading
        
        results = []
        
        def generate_code(source, source_lang, target_lang):
            result = self.generator.generate_code(source, source_lang, target_lang)
            results.append(result)
        
        # Run multiple generations concurrently
        threads = []
        test_cases = [
            (self.python_function, "python", "pomegranate"),
            (self.python_class, "python", "rust"),
            (self.javascript_code, "javascript", "typescript")
        ]
        
        for source, source_lang, target_lang in test_cases:
            thread = threading.Thread(
                target=generate_code, 
                args=(source, source_lang, target_lang)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all generations completed successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result["success"])
            self.assertIn("generated_code", result)
    
    def test_generator_registration(self):
        """Test custom generator registration"""
        # Mock custom generator
        mock_generator = Mock()
        mock_generator.generate = Mock(return_value="custom generated code")
        
        # Register generator
        self.generator.register_generator("python", "custom_lang", mock_generator)
        
        # Verify registration
        key = "python_to_custom_lang"
        self.assertIn(key, self.generator.generators)
        self.assertEqual(self.generator.generators[key], mock_generator)