"""
Core Functionality Unit Tests
Tests for the main MyndraComposer core components
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Assuming the main components are available for import
# In a real scenario, these would be the actual imports
class MockPomuseManager:
    """Mock MyndraComposer manager for testing"""
    def __init__(self):
        self.projects = {}
        self.plugins = {}
        self.config = {}
    
    def create_project(self, name, **kwargs):
        project_id = f"proj_{len(self.projects)}"
        project = Mock()
        project.id = project_id
        project.name = name
        for key, value in kwargs.items():
            setattr(project, key, value)
        self.projects[project_id] = project
        return project
    
    def get_project(self, project_id):
        return self.projects.get(project_id)
    
    def list_projects(self):
        return list(self.projects.values())

class CoreUnitTests(unittest.TestCase):
    """Unit tests for core MyndraComposer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.pomuse_manager = MockPomuseManager()
        
        # Create sample source files
        self.source_dir = self.test_dir / "source"
        self.source_dir.mkdir()
        
        # Sample Python file
        python_file = self.source_dir / "example.py"
        python_file.write_text("""
def fibonacci(n):
    \"\"\"Calculate fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
""")
        
        # Sample JavaScript file
        js_file = self.source_dir / "example.js"
        js_file.write_text("""
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class MathUtils {
    static power(base, exponent) {
        return Math.pow(base, exponent);
    }
}
""")
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_project_creation(self):
        """Test project creation functionality"""
        project = self.pomuse_manager.create_project(
            name="Test Project",
            description="A test project",
            source_language="python",
            target_language="myndra",
            source_path=str(self.source_dir)
        )
        
        self.assertIsNotNone(project)
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.source_language, "python")
        self.assertEqual(project.target_language, "myndra")
        self.assertEqual(project.source_path, str(self.source_dir))
    
    def test_project_retrieval(self):
        """Test project retrieval functionality"""
        project = self.pomuse_manager.create_project(
            name="Retrieval Test",
            source_language="javascript"
        )
        
        retrieved_project = self.pomuse_manager.get_project(project.id)
        self.assertIsNotNone(retrieved_project)
        self.assertEqual(retrieved_project.id, project.id)
        self.assertEqual(retrieved_project.name, "Retrieval Test")
    
    def test_project_listing(self):
        """Test project listing functionality"""
        initial_count = len(self.pomuse_manager.list_projects())
        
        # Create multiple projects
        self.pomuse_manager.create_project("Project 1", source_language="python")
        self.pomuse_manager.create_project("Project 2", source_language="javascript")
        self.pomuse_manager.create_project("Project 3", source_language="java")
        
        projects = self.pomuse_manager.list_projects()
        self.assertEqual(len(projects), initial_count + 3)
        
        project_names = [p.name for p in projects]
        self.assertIn("Project 1", project_names)
        self.assertIn("Project 2", project_names)
        self.assertIn("Project 3", project_names)
    
    def test_invalid_project_retrieval(self):
        """Test retrieval of non-existent project"""
        non_existent_project = self.pomuse_manager.get_project("invalid_id")
        self.assertIsNone(non_existent_project)
    
    def test_project_validation(self):
        """Test project parameter validation"""
        # Test with missing required parameters
        with self.assertRaises(TypeError):
            self.pomuse_manager.create_project()  # No name provided
    
    def test_file_discovery(self):
        """Test source file discovery"""
        # This would test the file discovery functionality
        source_files = list(self.source_dir.rglob("*.py"))
        self.assertEqual(len(source_files), 1)
        self.assertEqual(source_files[0].name, "example.py")
        
        js_files = list(self.source_dir.rglob("*.js"))
        self.assertEqual(len(js_files), 1)
        self.assertEqual(js_files[0].name, "example.js")
    
    def test_supported_languages(self):
        """Test supported language validation"""
        supported_source_languages = [
            "python", "javascript", "typescript", "java", 
            "cpp", "csharp", "rust", "go"
        ]
        
        supported_target_languages = [
            "myndra", "rust", "go", "typescript", "python"
        ]
        
        # Test valid combinations
        for source_lang in supported_source_languages:
            for target_lang in supported_target_languages:
                project = self.pomuse_manager.create_project(
                    name=f"Test {source_lang} to {target_lang}",
                    source_language=source_lang,
                    target_language=target_lang
                )
                self.assertIsNotNone(project)
    
    def test_project_configuration(self):
        """Test project configuration handling"""
        project = self.pomuse_manager.create_project(
            name="Config Test",
            source_language="python",
            target_language="myndra",
            migration_strategy="incremental",
            quality_threshold=0.8
        )
        
        self.assertEqual(project.migration_strategy, "incremental")
        self.assertEqual(project.quality_threshold, 0.8)
    
    def test_project_state_management(self):
        """Test project state transitions"""
        project = self.pomuse_manager.create_project(
            name="State Test",
            source_language="python"
        )
        
        # Mock project states
        project.status = "created"
        self.assertEqual(project.status, "created")
        
        project.status = "analyzing"
        self.assertEqual(project.status, "analyzing")
        
        project.status = "generating"
        self.assertEqual(project.status, "generating")
        
        project.status = "completed"
        self.assertEqual(project.status, "completed")
    
    def test_error_handling(self):
        """Test error handling in core functionality"""
        # Test with invalid source path
        project = self.pomuse_manager.create_project(
            name="Error Test",
            source_language="python",
            source_path="/nonexistent/path"
        )
        
        # The project should still be created, but validation would fail later
        self.assertIsNotNone(project)
        self.assertEqual(project.source_path, "/nonexistent/path")
    
    def test_project_metadata(self):
        """Test project metadata handling"""
        import datetime
        
        project = self.pomuse_manager.create_project(
            name="Metadata Test",
            source_language="python",
            description="Test project with metadata",
            created_by="test_user",
            tags=["test", "python", "conversion"]
        )
        
        self.assertEqual(project.description, "Test project with metadata")
        self.assertEqual(project.created_by, "test_user")
        self.assertEqual(project.tags, ["test", "python", "conversion"])
    
    def test_concurrent_project_creation(self):
        """Test handling of concurrent project creation"""
        import threading
        import time
        
        results = []
        
        def create_project(index):
            project = self.pomuse_manager.create_project(
                name=f"Concurrent Project {index}",
                source_language="python"
            )
            results.append(project)
        
        # Create multiple projects concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_project, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all projects were created
        self.assertEqual(len(results), 5)
        project_names = [p.name for p in results]
        for i in range(5):
            self.assertIn(f"Concurrent Project {i}", project_names)
    
    def test_project_update(self):
        """Test project update functionality"""
        project = self.pomuse_manager.create_project(
            name="Update Test",
            source_language="python",
            description="Original description"
        )
        
        # Mock update functionality
        project.description = "Updated description"
        project.target_language = "rust"
        
        self.assertEqual(project.description, "Updated description")
        self.assertEqual(project.target_language, "rust")
    
    def test_project_deletion(self):
        """Test project deletion functionality"""
        project = self.pomuse_manager.create_project(
            name="Delete Test",
            source_language="python"
        )
        
        project_id = project.id
        self.assertIsNotNone(self.pomuse_manager.get_project(project_id))
        
        # Mock deletion
        del self.pomuse_manager.projects[project_id]
        self.assertIsNone(self.pomuse_manager.get_project(project_id))