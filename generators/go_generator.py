"""
Go code generator for Universal Code Modernization Platform
Generates idiomatic Go code with modern patterns and best practices
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language_targets import LanguageGenerator, LanguageFeatures, CodeGenerationContext, MigrationStrategy
from typing import Dict, List, Any
import json
from datetime import datetime


class GoGenerator(LanguageGenerator):
    """Generator for Go programming language"""
    
    def __init__(self):
        features = LanguageFeatures(
            name="Go",
            version="1.21",
            has_generics=True,
            has_async_await=False,  # Uses goroutines
            has_pattern_matching=False,
            has_null_safety=False,
            has_memory_safety=True,
            has_type_inference=True,
            has_traits_interfaces=True,
            concurrency_model="goroutines",
            has_channels=True,
            has_green_threads=True,
            error_handling="explicit_errors",
            package_manager="go_modules",
            package_file="go.mod",
            build_system="go",
            build_file="go.mod",
            web_frameworks=["gin", "echo", "fiber", "chi"],
            testing_frameworks=["built-in", "testify", "ginkgo"],
            orm_libraries=["gorm", "sqlx", "ent"]
        )
        super().__init__(features)
        
        self.type_mappings = {
            "string": "string",
            "str": "string",
            "int": "int",
            "long": "int64",
            "float": "float32",
            "double": "float64",
            "boolean": "bool",
            "void": "",
            "list": "[]",
            "array": "[]",
            "map": "map",
            "dict": "map",
        }
        
        self.common_imports = {
            "web": ["net/http", "github.com/gin-gonic/gin", "encoding/json"],
            "cli": ["flag", "os", "fmt"],
            "library": ["fmt", "errors"],
            "service": ["net/http", "context", "log", "encoding/json"]
        }
    
    def generate_project_structure(self, context: CodeGenerationContext) -> Dict[str, str]:
        """Generate Go project structure with go.mod and main files"""
        
        module_name = self._sanitize_module_name(context.domain)
        
        go_mod = f'''module {module_name}

go 1.21

require (
{self._format_dependencies(context)}
)
'''

        main_go = f'''// Package main is the entry point for the {context.domain} application
// Auto-generated Go code from MyndraComposer
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {{
	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup graceful shutdown
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)

	// Start the application
	if err := run(ctx); err != nil {{
		log.Fatalf("Application failed: %v", err)
	}}

	// Wait for shutdown signal
	<-signalCh
	log.Println("Shutting down gracefully...")

	// Give some time for cleanup
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := shutdown(shutdownCtx); err != nil {{
		log.Printf("Shutdown error: %v", err)
	}}

	log.Println("Application stopped")
}}

// run starts the main application logic
func run(ctx context.Context) error {{
	fmt.Println("Starting {context.domain} application...")
	
	// TODO: Implement main application logic
	
	return nil
}}

// shutdown handles graceful shutdown
func shutdown(ctx context.Context) error {{
	// TODO: Cleanup resources
	return nil
}}
'''

        config_go = '''// Package config handles application configuration
package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config represents application configuration
type Config struct {
	AppName     string `json:"app_name"`
	Version     string `json:"version"`
	Environment string `json:"environment"`
	Port        int    `json:"port"`
	DatabaseURL string `json:"database_url"`
	APIKey      string `json:"api_key"`
}

// Load loads configuration from environment variables and files
func Load() (*Config, error) {
	config := &Config{
		AppName:     getEnv("APP_NAME", "MyndraComposer Generated App"),
		Version:     getEnv("VERSION", "0.1.0"),
		Environment: getEnv("ENVIRONMENT", "development"),
		Port:        getEnvInt("PORT", 8080),
		DatabaseURL: getEnv("DATABASE_URL", ""),
		APIKey:      getEnv("API_KEY", ""),
	}

	return config, nil
}

// LoadFromFile loads configuration from a JSON file
func LoadFromFile(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &config, nil
}

// IsProduction returns true if running in production environment
func (c *Config) IsProduction() bool {
	return c.Environment == "production"
}

// IsDevelopment returns true if running in development environment
func (c *Config) IsDevelopment() bool {
	return c.Environment == "development"
}

// getEnv gets environment variable with default fallback
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt gets integer environment variable with default fallback
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		// In real implementation, parse the integer
		return defaultValue // Simplified for now
	}
	return defaultValue
}
'''

        errors_go = '''// Package errors provides custom error types and utilities
package errors

import (
	"fmt"
)

// ErrorCode represents different types of application errors
type ErrorCode int

const (
	ErrUnknown ErrorCode = iota
	ErrValidation
	ErrNotFound
	ErrUnauthorized
	ErrInternal
	ErrNetwork
	ErrDatabase
	ErrConfiguration
)

// AppError represents a structured application error
type AppError struct {
	Code    ErrorCode `json:"code"`
	Message string    `json:"message"`
	Details string    `json:"details,omitempty"`
	Err     error     `json:"-"`
}

// Error implements the error interface
func (e *AppError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s (%v)", e.Message, e.Details, e.Err)
	}
	if e.Details != "" {
		return fmt.Sprintf("%s: %s", e.Message, e.Details)
	}
	return e.Message
}

// Unwrap returns the underlying error
func (e *AppError) Unwrap() error {
	return e.Err
}

// New creates a new AppError
func New(code ErrorCode, message string) *AppError {
	return &AppError{
		Code:    code,
		Message: message,
	}
}

// Wrap wraps an existing error with additional context
func Wrap(err error, code ErrorCode, message string) *AppError {
	return &AppError{
		Code:    code,
		Message: message,
		Err:     err,
	}
}

// Validation creates a validation error
func Validation(field, message string) *AppError {
	return &AppError{
		Code:    ErrValidation,
		Message: "Validation error",
		Details: fmt.Sprintf("Field '%s': %s", field, message),
	}
}

// NotFound creates a not found error
func NotFound(resource string) *AppError {
	return &AppError{
		Code:    ErrNotFound,
		Message: "Resource not found",
		Details: resource,
	}
}

// Unauthorized creates an unauthorized error
func Unauthorized(message string) *AppError {
	return &AppError{
		Code:    ErrUnauthorized,
		Message: "Unauthorized",
		Details: message,
	}
}

// Internal creates an internal server error
func Internal(err error) *AppError {
	return &AppError{
		Code:    ErrInternal,
		Message: "Internal server error",
		Err:     err,
	}
}

// IsCode checks if error has specific code
func IsCode(err error, code ErrorCode) bool {
	if appErr, ok := err.(*AppError); ok {
		return appErr.Code == code
	}
	return false
}
'''

        utils_go = '''// Package utils provides common utility functions
package utils

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/mail"
	"time"
)

// GenerateID generates a unique ID based on timestamp and random bytes
func GenerateID() string {
	timestamp := time.Now().UnixNano()
	randomBytes := make([]byte, 4)
	rand.Read(randomBytes)
	return fmt.Sprintf("id_%d_%s", timestamp, hex.EncodeToString(randomBytes))
}

// ValidateEmail validates email format
func ValidateEmail(email string) bool {
	_, err := mail.ParseAddress(email)
	return err == nil
}

// RetryWithBackoff executes a function with exponential backoff retry logic
func RetryWithBackoff(operation func() error, maxRetries int, baseDelay time.Duration) error {
	var lastErr error
	delay := baseDelay

	for attempt := 1; attempt <= maxRetries; attempt++ {
		err := operation()
		if err == nil {
			return nil
		}

		lastErr = err

		if attempt == maxRetries {
			break
		}

		time.Sleep(delay)
		delay *= 2 // Exponential backoff
	}

	return fmt.Errorf("operation failed after %d attempts, last error: %w", maxRetries, lastErr)
}

// SafeDivide performs division with zero check
func SafeDivide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, fmt.Errorf("division by zero")
	}
	return a / b, nil
}

// FormatBytes formats bytes as human-readable string
func FormatBytes(bytes uint64) string {
	units := []string{"B", "KB", "MB", "GB", "TB"}
	const threshold = 1024.0

	if bytes == 0 {
		return "0 B"
	}

	size := float64(bytes)
	unitIndex := 0

	for size >= threshold && unitIndex < len(units)-1 {
		size /= threshold
		unitIndex++
	}

	return fmt.Sprintf("%.1f %s", size, units[unitIndex])
}

// Contains checks if slice contains an element
func Contains[T comparable](slice []T, item T) bool {
	for _, element := range slice {
		if element == item {
			return true
		}
	}
	return false
}

// Map applies function to each element in slice
func Map[T, U any](slice []T, fn func(T) U) []U {
	result := make([]U, len(slice))
	for i, item := range slice {
		result[i] = fn(item)
	}
	return result
}

// Filter returns elements that satisfy predicate
func Filter[T any](slice []T, predicate func(T) bool) []T {
	var result []T
	for _, item := range slice {
		if predicate(item) {
			result = append(result, item)
		}
	}
	return result
}

// Reduce applies function against accumulator
func Reduce[T, U any](slice []T, initial U, fn func(U, T) U) U {
	result := initial
	for _, item := range slice {
		result = fn(result, item)
	}
	return result
}
'''

        service_go = '''// Package service contains business logic and service layer
package service

import (
	"context"
	"fmt"
	"sync"
	"time"

	"''' + module_name + '''/internal/config"
	"''' + module_name + '''/internal/errors"
)

// Service represents the main application service
type Service struct {
	config  *config.Config
	metrics map[string]int64
	mu      sync.RWMutex
	started time.Time
}

// New creates a new service instance
func New(cfg *config.Config) *Service {
	return &Service{
		config:  cfg,
		metrics: make(map[string]int64),
		started: time.Now(),
	}
}

// Start initializes and starts the service
func (s *Service) Start(ctx context.Context) error {
	fmt.Printf("Starting %s service (version %s)\\n", s.config.AppName, s.config.Version)
	
	// Initialize service components
	if err := s.initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize service: %w", err)
	}

	s.IncrementMetric("service_starts")
	return nil
}

// Stop gracefully stops the service
func (s *Service) Stop(ctx context.Context) error {
	fmt.Println("Stopping service...")
	
	// Cleanup resources
	if err := s.cleanup(ctx); err != nil {
		return fmt.Errorf("cleanup failed: %w", err)
	}

	s.IncrementMetric("service_stops")
	return nil
}

// GetConfig returns the service configuration
func (s *Service) GetConfig() *config.Config {
	return s.config
}

// GetMetrics returns current metrics snapshot
func (s *Service) GetMetrics() map[string]int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Create a copy to avoid race conditions
	metrics := make(map[string]int64, len(s.metrics))
	for k, v := range s.metrics {
		metrics[k] = v
	}
	return metrics
}

// IncrementMetric increments a metric counter
func (s *Service) IncrementMetric(key string) {
	s.mu.Lock()
	s.metrics[key]++
	s.mu.Unlock()
}

// GetUptime returns how long the service has been running
func (s *Service) GetUptime() time.Duration {
	return time.Since(s.started)
}

// HealthCheck performs a health check
func (s *Service) HealthCheck(ctx context.Context) error {
	// TODO: Add actual health checks (database connectivity, external services, etc.)
	if s.config == nil {
		return errors.Internal(fmt.Errorf("service not properly initialized"))
	}
	
	return nil
}

// ProcessRequest handles a generic request (placeholder)
func (s *Service) ProcessRequest(ctx context.Context, request interface{}) (interface{}, error) {
	s.IncrementMetric("requests_processed")
	
	// TODO: Implement actual request processing logic
	
	return map[string]interface{}{
		"status":    "success",
		"timestamp": time.Now(),
		"uptime":    s.GetUptime().String(),
	}, nil
}

// initialize performs service initialization
func (s *Service) initialize(ctx context.Context) error {
	// TODO: Initialize databases, external clients, etc.
	s.IncrementMetric("initializations")
	return nil
}

// cleanup performs service cleanup
func (s *Service) cleanup(ctx context.Context) error {
	// TODO: Close databases, cleanup resources, etc.
	s.IncrementMetric("cleanups")
	return nil
}
'''

        main_test_go = '''// Package main contains tests for the main application
package main

import (
	"context"
	"testing"
	"time"
)

func TestRun(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	err := run(ctx)
	if err != nil {
		t.Errorf("run() returned error: %v", err)
	}
}

func TestShutdown(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	err := shutdown(ctx)
	if err != nil {
		t.Errorf("shutdown() returned error: %v", err)
	}
}
'''

        utils_test_go = '''// Package utils contains tests for utility functions
package utils

import (
	"strings"
	"testing"
	"time"
)

func TestGenerateID(t *testing.T) {
	id1 := GenerateID()
	id2 := GenerateID()

	if id1 == id2 {
		t.Error("GenerateID should produce unique IDs")
	}

	if !strings.HasPrefix(id1, "id_") {
		t.Error("GenerateID should have 'id_' prefix")
	}
}

func TestValidateEmail(t *testing.T) {
	tests := []struct {
		email string
		valid bool
	}{
		{"user@example.com", true},
		{"test.user@domain.co.uk", true},
		{"invalid_email", false},
		{"@domain.com", false},
		{"user@", false},
		{"", false},
	}

	for _, test := range tests {
		result := ValidateEmail(test.email)
		if result != test.valid {
			t.Errorf("ValidateEmail(%s) = %v, want %v", test.email, result, test.valid)
		}
	}
}

func TestSafeDivide(t *testing.T) {
	// Test normal division
	result, err := SafeDivide(10.0, 2.0)
	if err != nil || result != 5.0 {
		t.Errorf("SafeDivide(10.0, 2.0) = %v, %v; want 5.0, nil", result, err)
	}

	// Test division by zero
	_, err = SafeDivide(10.0, 0.0)
	if err == nil {
		t.Error("SafeDivide(10.0, 0.0) should return error")
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes    uint64
		expected string
	}{
		{0, "0 B"},
		{500, "500.0 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1048576, "1.0 MB"},
	}

	for _, test := range tests {
		result := FormatBytes(test.bytes)
		if result != test.expected {
			t.Errorf("FormatBytes(%d) = %s, want %s", test.bytes, result, test.expected)
		}
	}
}

func TestRetryWithBackoff(t *testing.T) {
	attempts := 0
	operation := func() error {
		attempts++
		if attempts < 3 {
			return fmt.Errorf("temporary error")
		}
		return nil
	}

	err := RetryWithBackoff(operation, 5, 10*time.Millisecond)
	if err != nil {
		t.Errorf("RetryWithBackoff should succeed after retries: %v", err)
	}

	if attempts != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}
}

func TestContains(t *testing.T) {
	slice := []string{"apple", "banana", "cherry"}
	
	if !Contains(slice, "banana") {
		t.Error("Contains should find 'banana'")
	}
	
	if Contains(slice, "grape") {
		t.Error("Contains should not find 'grape'")
	}
}

func TestMap(t *testing.T) {
	numbers := []int{1, 2, 3, 4}
	doubled := Map(numbers, func(n int) int { return n * 2 })
	
	expected := []int{2, 4, 6, 8}
	for i, v := range doubled {
		if v != expected[i] {
			t.Errorf("Map result[%d] = %d, want %d", i, v, expected[i])
		}
	}
}

func TestFilter(t *testing.T) {
	numbers := []int{1, 2, 3, 4, 5, 6}
	evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
	
	expected := []int{2, 4, 6}
	if len(evens) != len(expected) {
		t.Errorf("Filter result length = %d, want %d", len(evens), len(expected))
	}
	
	for i, v := range evens {
		if v != expected[i] {
			t.Errorf("Filter result[%d] = %d, want %d", i, v, expected[i])
		}
	}
}

func TestReduce(t *testing.T) {
	numbers := []int{1, 2, 3, 4}
	sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
	
	if sum != 10 {
		t.Errorf("Reduce sum = %d, want 10", sum)
	}
}
'''

        return {
            "go.mod": go_mod,
            "main.go": main_go,
            "internal/config/config.go": config_go,
            "internal/errors/errors.go": errors_go,
            "internal/utils/utils.go": utils_go,
            "internal/service/service.go": service_go,
            "main_test.go": main_test_go,
            "internal/utils/utils_test.go": utils_test_go,
            ".gitignore": self._generate_gitignore(),
            "README.md": self._generate_readme(context),
            "Dockerfile": self._generate_dockerfile(module_name),
            "Makefile": self._generate_makefile(),
        }
    
    def generate_function(self, function_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a Go function"""
        name = function_info.get("name", "UnnamedFunction")
        params = function_info.get("parameters", [])
        return_type = function_info.get("return_type", "")
        visibility = function_info.get("visibility", "private")
        description = function_info.get("description", "")
        
        # Go uses capitalization for visibility
        if visibility == "public":
            name = name[0].upper() + name[1:] if name else name
        else:
            name = name[0].lower() + name[1:] if name else name
        
        # Convert parameters
        go_params = []
        for param in params:
            param_name = param.get("name", "param")
            param_type = param.get("type", "string")
            go_type = self._convert_type(param_type)
            go_params.append(f"{param_name} {go_type}")
        
        # Convert return type
        go_return_type = ""
        if return_type and return_type != "void":
            go_return_type = self._convert_type(return_type)
            # Add error return for most functions
            if go_return_type:
                go_return_type = f"({go_return_type}, error)"
            else:
                go_return_type = "error"
        
        # Generate documentation
        doc = ""
        if description:
            doc = f"// {name} {description}\n"
        
        # Generate function signature
        params_str = ", ".join(go_params)
        
        # Generate function body
        body = self._generate_function_body(function_info, go_return_type)
        
        return f'''{doc}func {name}({params_str}) {go_return_type} {{
{body}
}}'''
    
    def generate_class(self, class_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a Go struct with methods"""
        name = class_info.get("name", "UnnamedStruct")
        fields = class_info.get("fields", [])
        methods = class_info.get("methods", [])
        description = class_info.get("description", "")
        
        # Generate struct fields
        struct_fields = []
        for field in fields:
            field_name = field.get("name", "field")
            field_type = field.get("type", "string")
            field_vis = field.get("visibility", "private")
            
            # Capitalize for public fields
            if field_vis == "public":
                field_name = field_name[0].upper() + field_name[1:]
            
            go_type = self._convert_type(field_type)
            json_tag = f'`json:"{field_name.lower()}"`'
            
            struct_fields.append(f"\t{field_name} {go_type} {json_tag}")
        
        # Documentation
        doc = ""
        if description:
            doc = f"// {name} {description}\n"
        
        # Struct definition
        struct_def = f'''{doc}type {name} struct {{
{chr(10).join(struct_fields)}
}}'''
        
        # Constructor function
        constructor_params = []
        constructor_assignments = []
        for field in fields:
            field_name = field.get("name", "field")
            field_type = field.get("type", "string")
            go_type = self._convert_type(field_type)
            
            constructor_params.append(f"{field_name} {go_type}")
            
            # Capitalize field name for struct assignment if public
            field_vis = field.get("visibility", "private")
            struct_field_name = field_name[0].upper() + field_name[1:] if field_vis == "public" else field_name
            constructor_assignments.append(f"\t\t{struct_field_name}: {field_name},")
        
        constructor = f'''// New{name} creates a new instance of {name}
func New{name}({", ".join(constructor_params)}) *{name} {{
\treturn &{name}{{
{chr(10).join(constructor_assignments)}
\t}}
}}'''
        
        # Methods
        method_implementations = []
        for method in methods:
            # Add receiver to method
            method_copy = method.copy()
            method_copy["receiver"] = f"*{name}"
            go_method = self.generate_function(method_copy, context)
            method_implementations.append(go_method)
        
        methods_str = "\n\n".join(method_implementations)
        if methods_str:
            methods_str = "\n\n" + methods_str
        
        return f'''{struct_def}

{constructor}{methods_str}'''
    
    def generate_module(self, module_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a Go package"""
        name = module_info.get("name", "unnamed")
        description = module_info.get("description", "")
        functions = module_info.get("functions", [])
        classes = module_info.get("classes", [])
        imports = module_info.get("imports", [])
        
        # Package declaration
        header = f'''// Package {name} {description}
package {name}

'''
        
        # Imports
        imports_section = ""
        if imports:
            import_lines = []
            for imp in imports:
                import_lines.append(f'\t"{self._convert_import(imp)}"')
            imports_section = f'''import (
{chr(10).join(import_lines)}
)

'''
        
        # Generate structs/classes
        classes_section = ""
        for class_info in classes:
            classes_section += self.generate_class(class_info, context) + "\n\n"
        
        # Generate functions
        functions_section = ""
        for func_info in functions:
            functions_section += self.generate_function(func_info, context) + "\n\n"
        
        return header + imports_section + classes_section + functions_section
    
    def generate_error_handling(self, error_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate Go error handling code"""
        error_type = error_info.get("type", "generic")
        message = error_info.get("message", "An error occurred")
        
        if error_type == "validation":
            field = error_info.get("field", "unknown")
            return f'errors.Validation("{field}", "{message}")'
        elif error_type == "not_found":
            resource = error_info.get("resource", "resource")
            return f'errors.NotFound("{resource}")'
        elif error_type == "unauthorized":
            return f'errors.Unauthorized("{message}")'
        else:
            return f'fmt.Errorf("{message}")'
    
    def generate_async_code(self, async_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate Go concurrent code using goroutines"""
        operation = async_info.get("operation", "")
        
        if "http_request" in operation.lower():
            return '''resp, err := http.Get(url)
if err != nil {
    return nil, fmt.Errorf("HTTP request failed: %w", err)
}
defer resp.Body.Close()

body, err := io.ReadAll(resp.Body)
if err != nil {
    return nil, fmt.Errorf("failed to read response: %w", err)
}'''
        elif "concurrent_processing" in operation.lower():
            return '''// Process items concurrently using goroutines
var wg sync.WaitGroup
results := make(chan Result, len(items))

for _, item := range items {
    wg.Add(1)
    go func(item Item) {
        defer wg.Done()
        result := processItem(item)
        results <- result
    }(item)
}

// Wait for all goroutines to complete
go func() {
    wg.Wait()
    close(results)
}()

// Collect results
var allResults []Result
for result := range results {
    allResults = append(allResults, result)
}'''
        else:
            return '''// TODO: Implement concurrent operation using goroutines
go func() {
    // Concurrent operation
}()'''
    
    def generate_tests(self, test_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate Go test code"""
        test_name = test_info.get("name", "TestFunction")
        
        # Ensure test name starts with Test
        if not test_name.startswith("Test"):
            test_name = "Test" + test_name[0].upper() + test_name[1:]
        
        return f'''func {test_name}(t *testing.T) {{
\t// TODO: Implement test
\tif testing.Short() {{
\t\tt.Skip("skipping test in short mode")
\t}}
\t
\t// Add test implementation here
\t// Use t.Error, t.Errorf, t.Fatal, t.Fatalf for test failures
}}'''
    
    def get_migration_strategies(self, source_analysis: Dict[str, Any]) -> List[MigrationStrategy]:
        """Get Go-specific migration strategies"""
        strategies = []
        
        # Microservices migration
        strategies.append(MigrationStrategy(
            name="Microservices with Go",
            description="Break monolith into Go microservices with goroutines",
            approach="incremental",
            complexity="medium",
            timeline_estimate="4-8 months",
            risks=["Service communication complexity", "Distributed system challenges"],
            benefits=["High concurrency", "Simple deployment", "Fast compilation"],
            prerequisites=["Service boundaries identified", "API contracts defined"],
            requires_manual_review=True,
            supports_gradual_migration=True,
            maintains_performance=True,
            preserves_architecture=False
        ))
        
        # Performance optimization
        strategies.append(MigrationStrategy(
            name="Performance-First Migration",
            description="Optimize for high-performance concurrent processing",
            approach="rewrite",
            complexity="medium",
            timeline_estimate="3-6 months",
            risks=["Goroutine management", "Race conditions"],
            benefits=["Excellent concurrency", "Low latency", "Efficient memory usage"],
            prerequisites=["Performance requirements defined", "Load testing setup"],
            requires_manual_review=True,
            supports_gradual_migration=False,
            maintains_performance=False,  # Actually improves it
            preserves_architecture=True
        ))
        
        return strategies
    
    def get_best_practices(self, context: CodeGenerationContext) -> List[str]:
        """Get Go best practices"""
        return [
            "Use explicit error handling instead of exceptions",
            "Prefer composition over inheritance",
            "Use goroutines and channels for concurrency",
            "Keep interfaces small and focused",
            "Use context.Context for cancellation and timeouts",
            "Follow Go naming conventions (camelCase, PascalCase)",
            "Use go fmt to format code consistently",
            "Write table-driven tests",
            "Use sync.WaitGroup for waiting on multiple goroutines",
            "Avoid goroutine leaks by proper cleanup"
        ]
    
    def _convert_type(self, original_type: str) -> str:
        """Convert generic type to Go type"""
        if original_type in self.type_mappings:
            return self.type_mappings[original_type]
        
        # Handle generic types
        if original_type.startswith("List<") or original_type.startswith("list<"):
            inner_type = original_type[5:-1]
            return f"[]{self._convert_type(inner_type)}"
        elif original_type.startswith("Map<") or original_type.startswith("map<"):
            # Extract key and value types
            inner = original_type[4:-1]
            if "," in inner:
                key_type, value_type = inner.split(",", 1)
                return f"map[{self._convert_type(key_type.strip())}]{self._convert_type(value_type.strip())}"
        
        # Default to string for unknown types
        return "string"
    
    def _convert_import(self, import_statement: str) -> str:
        """Convert import statement to Go import"""
        # This is a simplified conversion
        if "http" in import_statement.lower():
            return "net/http"
        elif "json" in import_statement.lower():
            return "encoding/json"
        elif "time" in import_statement.lower():
            return "time"
        elif "fmt" in import_statement.lower():
            return "fmt"
        else:
            return import_statement.replace(".", "/")
    
    def _sanitize_module_name(self, name: str) -> str:
        """Sanitize module name for Go"""
        return name.lower().replace(" ", "-").replace("_", "-")
    
    def _format_dependencies(self, context: CodeGenerationContext) -> str:
        """Format dependencies for go.mod based on context"""
        deps = []
        
        if context.domain == "web":
            deps.extend([
                "\tgithub.com/gin-gonic/gin v1.9.1",
                "\tgithub.com/stretchr/testify v1.8.4"
            ])
        
        if context.domain == "cli":
            deps.extend([
                "\tgithub.com/spf13/cobra v1.7.0"
            ])
        
        # Common dependencies
        deps.extend([
            "\tgithub.com/stretchr/testify v1.8.4"
        ])
        
        return "\n".join(deps)
    
    def _generate_function_body(self, function_info: Dict[str, Any], return_type: str) -> str:
        """Generate function body"""
        if "error" in return_type:
            return '\t// TODO: Implement function logic\n\treturn nil'
        elif return_type and return_type != "":
            # Return zero value and nil error
            zero_val = self._get_zero_value(return_type)
            return f'\t// TODO: Implement function logic\n\treturn {zero_val}, nil'
        else:
            return '\t// TODO: Implement function logic'
    
    def _get_zero_value(self, go_type: str) -> str:
        """Get zero value for Go type"""
        zero_values = {
            "string": '""',
            "int": "0",
            "int64": "0",
            "float32": "0.0",
            "float64": "0.0",
            "bool": "false",
        }
        
        if go_type.startswith("[]"):
            return "nil"
        elif go_type.startswith("map["):
            return "nil"
        elif go_type.startswith("*"):
            return "nil"
        
        return zero_values.get(go_type, "nil")
    
    def _generate_gitignore(self) -> str:
        """Generate .gitignore for Go project"""
        return '''# Binaries for programs and plugins
*.exe
*.exe~
*.dll
*.so
*.dylib

# Test binary, built with `go test -c`
*.test

# Output of the go coverage tool, specifically when used with LiteIDE
*.out

# Dependency directories (remove the comment below to include it)
vendor/

# Go workspace file
go.work

# Environment variables
.env

# IDE files
.vscode/
.idea/
*.swp
*.tmp

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
'''
    
    def _generate_readme(self, context: CodeGenerationContext) -> str:
        """Generate README.md for Go project"""
        module_name = self._sanitize_module_name(context.domain)
        return f'''# {context.domain.title()} - Go Migration

This project was generated by MyndraComposer, migrating from {context.source_language} to Go.

## Features

- High-performance concurrent processing with goroutines
- Explicit error handling
- Simple and efficient design
- Fast compilation and deployment
- Comprehensive testing

## Getting Started

### Prerequisites

- Go 1.21 or later

### Building

```bash
go build
```

### Running

```bash
go run main.go
```

### Testing

```bash
go test ./...
```

### Running with Coverage

```bash
go test -cover ./...
```

### Formatting and Linting

```bash
go fmt ./...
go vet ./...
```

### Building for Production

```bash
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .
```

## Project Structure

- `main.go` - Application entry point
- `internal/config/` - Configuration management
- `internal/service/` - Business logic and services  
- `internal/errors/` - Error handling utilities
- `internal/utils/` - Common utilities
- `*_test.go` - Test files

## Docker

Build Docker image:
```bash
docker build -t {module_name} .
```

Run container:
```bash
docker run -p 8080:8080 {module_name}
```

## Generated by

MyndraComposer - Universal Code Modernization Platform
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
    
    def _generate_dockerfile(self, module_name: str) -> str:
        """Generate Dockerfile for Go project"""
        return f'''# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Expose port
EXPOSE 8080

# Run the binary
CMD ["./main"]
'''
    
    def _generate_makefile(self) -> str:
        """Generate Makefile for Go project"""
        return '''.PHONY: build run test clean fmt vet lint docker

# Build the application
build:
	go build -o bin/app main.go

# Run the application
run:
	go run main.go

# Run tests
test:
	go test -v ./...

# Run tests with coverage
test-coverage:
	go test -v -cover ./...

# Clean build artifacts
clean:
	rm -rf bin/
	go clean

# Format code
fmt:
	go fmt ./...

# Vet code
vet:
	go vet ./...

# Lint code (requires golangci-lint)
lint:
	golangci-lint run

# Build Docker image
docker:
	docker build -t app .

# Run Docker container
docker-run:
	docker run -p 8080:8080 app

# Install dependencies
deps:
	go mod download
	go mod tidy

# Generate mocks (if using gomock)
mocks:
	go generate ./...

# Run benchmarks
bench:
	go test -bench=. ./...

# Security scan (requires gosec)
security:
	gosec ./...
'''