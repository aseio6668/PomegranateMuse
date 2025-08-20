"""
Build testing and error analysis for generated Pomegranate code
Integrates with the Pomegranate compiler to test generated code
"""

import asyncio
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BuildResult:
    """Result of a build attempt"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    build_time_ms: int
    error_analysis: Optional[Dict[str, Any]] = None


@dataclass
class ErrorAnalysis:
    """Analysis of build errors"""
    error_type: str
    error_message: str
    suggested_fix: str
    line_number: Optional[int] = None
    severity: str = "error"  # error, warning, info


class PomegranateBuilder:
    """Interface to Pomegranate compiler for testing generated code"""
    
    def __init__(self, pomegranate_path: Optional[Path] = None):
        # Try to find Pomegranate compiler
        if pomegranate_path:
            self.compiler_path = pomegranate_path
        else:
            # Look for it in the parent directory structure
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir.parent / "Pomegrante2[c]" / "build" / "pomegranate",
                current_dir.parent / "Pomegrante2[c]" / "build" / "pomegranate.exe",
                Path("pomegranate"),  # In PATH
                Path("./pomegranate"),  # Local
            ]
            
            self.compiler_path = None
            for path in possible_paths:
                if path.exists() or self._command_exists(str(path)):
                    self.compiler_path = path
                    break
        
        self.temp_dir = None
    
    def _command_exists(self, command: str) -> bool:
        """Check if command exists in PATH"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, 
                         timeout=5)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def test_code(self, pomegranate_code: str, test_name: str = "generated") -> BuildResult:
        """Test Pomegranate code by attempting to compile it"""
        if not self.compiler_path:
            return BuildResult(
                success=False,
                stdout="",
                stderr="Pomegranate compiler not found",
                exit_code=-1,
                build_time_ms=0,
                error_analysis={
                    "error_type": "compiler_missing",
                    "message": "Pomegranate compiler not available",
                    "suggestion": "Build Pomegranate compiler or add to PATH"
                }
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pom', delete=False) as temp_file:
            temp_file.write(pomegranate_code)
            temp_file_path = Path(temp_file.name)
        
        try:
            start_time = datetime.now()
            
            # Run compiler
            result = subprocess.run(
                [str(self.compiler_path), str(temp_file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            end_time = datetime.now()
            build_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Analyze errors if compilation failed
            error_analysis = None
            if result.returncode != 0:
                error_analysis = self._analyze_errors(result.stderr, pomegranate_code)
            
            return BuildResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                build_time_ms=build_time_ms,
                error_analysis=error_analysis
            )
            
        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False,
                stdout="",
                stderr="Compilation timeout",
                exit_code=-1,
                build_time_ms=30000,
                error_analysis={
                    "error_type": "timeout",
                    "message": "Compilation took too long",
                    "suggestion": "Simplify generated code or check for infinite loops"
                }
            )
        except Exception as e:
            return BuildResult(
                success=False,
                stdout="",
                stderr=f"Build error: {e}",
                exit_code=-1,
                build_time_ms=0,
                error_analysis={
                    "error_type": "build_system_error",
                    "message": str(e),
                    "suggestion": "Check Pomegranate installation and permissions"
                }
            )
        finally:
            # Clean up temporary file
            try:
                temp_file_path.unlink()
            except:
                pass
    
    def _analyze_errors(self, stderr: str, source_code: str) -> Dict[str, Any]:
        """Analyze compilation errors and suggest fixes"""
        errors = []
        
        lines = stderr.split('\n')
        source_lines = source_code.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Parse common error patterns
            error_info = self._parse_error_line(line, source_lines)
            if error_info:
                errors.append(error_info)
        
        return {
            "total_errors": len(errors),
            "errors": errors,
            "analysis_summary": self._summarize_errors(errors)
        }
    
    def _parse_error_line(self, error_line: str, source_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a single error line and extract information"""
        error_line = error_line.strip()
        
        # Common error patterns
        patterns = [
            {
                "pattern": "syntax error",
                "type": "syntax_error",
                "suggestion": "Check syntax around the indicated line"
            },
            {
                "pattern": "undefined function",
                "type": "undefined_function",
                "suggestion": "Define the function or check import statements"
            },
            {
                "pattern": "type mismatch",
                "type": "type_error",
                "suggestion": "Check type annotations and conversions"
            },
            {
                "pattern": "capability",
                "type": "capability_error",
                "suggestion": "Add required capabilities to import statements"
            },
            {
                "pattern": "import",
                "type": "import_error",
                "suggestion": "Check import syntax and module availability"
            }
        ]
        
        for pattern_info in patterns:
            if pattern_info["pattern"] in error_line.lower():
                return {
                    "error_type": pattern_info["type"],
                    "error_message": error_line,
                    "suggested_fix": pattern_info["suggestion"],
                    "severity": "error"
                }
        
        # Generic error
        return {
            "error_type": "unknown_error",
            "error_message": error_line,
            "suggested_fix": "Review the code around this error",
            "severity": "error"
        }
    
    def _summarize_errors(self, errors: List[Dict[str, Any]]) -> str:
        """Summarize error patterns"""
        if not errors:
            return "No errors found"
        
        error_types = {}
        for error in errors:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        summary_parts = []
        for error_type, count in error_types.items():
            readable_type = error_type.replace("_", " ").title()
            summary_parts.append(f"{count} {readable_type}{'s' if count > 1 else ''}")
        
        return f"Found: {', '.join(summary_parts)}"


class CodeIterator:
    """Iteratively improves generated code based on build results"""
    
    def __init__(self, builder: PomegranateBuilder, ollama_provider=None):
        self.builder = builder
        self.ollama = ollama_provider
        self.max_iterations = 3
    
    async def improve_code(self, initial_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Iteratively improve code based on build feedback"""
        current_code = initial_code
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            print(f"  Build iteration {iteration + 1}...")
            
            # Test current code
            build_result = await self.builder.test_code(current_code, f"iteration_{iteration + 1}")
            
            iteration_info = {
                "iteration": iteration + 1,
                "build_result": build_result,
                "code": current_code
            }
            iteration_results.append(iteration_info)
            
            if build_result.success:
                print(f"  ✅ Build succeeded on iteration {iteration + 1}")
                break
            
            print(f"  ❌ Build failed: {build_result.stderr[:100]}...")
            
            # Try to fix errors using ML if available
            if self.ollama and build_result.error_analysis:
                try:
                    improved_code = await self._fix_with_ml(
                        current_code, build_result, context
                    )
                    if improved_code and improved_code != current_code:
                        current_code = improved_code
                        print(f"  🔧 Applied ML-suggested fixes")
                    else:
                        print(f"  ⚠️  No ML improvements available")
                        break
                except Exception as e:
                    print(f"  ⚠️  ML fix failed: {e}")
                    break
            else:
                print(f"  ⚠️  No ML provider available for automated fixes")
                break
        
        final_result = iteration_results[-1]["build_result"] if iteration_results else None
        
        return {
            "final_code": current_code,
            "iterations": iteration_results,
            "success": final_result.success if final_result else False,
            "total_iterations": len(iteration_results)
        }
    
    async def _fix_with_ml(self, code: str, build_result: BuildResult, context: Dict[str, Any]) -> str:
        """Use ML to fix compilation errors"""
        if not self.ollama:
            return code
        
        fix_prompt = f"""Fix the following Pomegranate code compilation errors:

Original Code:
```pomegranate
{code}
```

Compilation Errors:
{build_result.stderr}

Error Analysis:
{json.dumps(build_result.error_analysis, indent=2)}

Please fix the compilation errors while maintaining the original intent and functionality.
Return only the corrected Pomegranate code, properly formatted."""
        
        try:
            # This would use the Ollama client to get fixes
            # For now, return original code (placeholder)
            # In a real implementation, this would call ollama.generate()
            return code
        except Exception:
            return code


# Integration with main PomegranteMuse
class BuildTestingIntegration:
    """Integrates build testing into the main PomegranteMuse workflow"""
    
    def __init__(self, enable_testing: bool = True):
        self.enable_testing = enable_testing
        self.builder = PomegranateBuilder() if enable_testing else None
        self.iterator = CodeIterator(self.builder) if self.builder else None
    
    async def test_and_improve_generated_code(
        self, 
        generated_code: str, 
        context: Dict[str, Any],
        ollama_provider=None
    ) -> Dict[str, Any]:
        """Test generated code and attempt improvements"""
        
        if not self.enable_testing or not self.builder:
            return {
                "tested": False,
                "reason": "Build testing disabled or compiler not available",
                "final_code": generated_code,
                "success": None
            }
        
        print("🔨 Testing generated Pomegranate code...")
        
        # Set up iterator with ML provider
        if ollama_provider:
            self.iterator.ollama = ollama_provider
        
        # Test and improve code
        improvement_result = await self.iterator.improve_code(generated_code, context)
        
        return {
            "tested": True,
            "success": improvement_result["success"],
            "final_code": improvement_result["final_code"],
            "iterations": improvement_result["total_iterations"],
            "build_details": improvement_result["iterations"]
        }


async def test_build_system():
    """Test the build system functionality"""
    print("Testing PomegranteMuse build system...")
    
    # Test with simple valid code
    valid_code = '''
import std::io with capabilities("write")

fn main() {
    print("Hello, Pomegranate!")
}
'''
    
    # Test with invalid code
    invalid_code = '''
import invalid::module with capabilities("nonexistent")

fn main() {
    undefined_function()
    let x: invalid_type = "test"
}
'''
    
    builder = PomegranateBuilder()
    
    print("\n1. Testing valid code...")
    result1 = await builder.test_code(valid_code, "valid_test")
    print(f"   Result: {'✅ Success' if result1.success else '❌ Failed'}")
    if not result1.success:
        print(f"   Error: {result1.stderr}")
    
    print("\n2. Testing invalid code...")
    result2 = await builder.test_code(invalid_code, "invalid_test")
    print(f"   Result: {'✅ Success' if result2.success else '❌ Failed (expected)'}")
    if result2.error_analysis:
        print(f"   Errors found: {result2.error_analysis.get('total_errors', 0)}")
        print(f"   Summary: {result2.error_analysis.get('analysis_summary', 'No summary')}")
    
    print("\n3. Testing code improvement...")
    iterator = CodeIterator(builder)
    improvement = await iterator.improve_code(invalid_code, {})
    print(f"   Iterations: {improvement['total_iterations']}")
    print(f"   Final success: {'✅ Yes' if improvement['success'] else '❌ No'}")


if __name__ == "__main__":
    asyncio.run(test_build_system())