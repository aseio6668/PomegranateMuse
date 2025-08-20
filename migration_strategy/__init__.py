"""
Migration Strategy System for PomegranteMuse
Provides comprehensive migration planning, execution, and validation
"""

from .strategy_planner import (
    MigrationPlanner,
    MigrationStrategy,
    MigrationPhase,
    MigrationPlan,
    ComponentAnalysis,
    DependencyGraph,
    RiskAssessment,
    create_migration_plan
)

from .execution_engine import (
    MigrationExecutor,
    ExecutionResult,
    ExecutionStatus,
    MigrationTask,
    TaskType,
    ParallelExecutor,
    RollbackManager,
    execute_migration_plan
)

from .validation_framework import (
    MigrationValidator,
    ValidationResult,
    ValidationSuite,
    CompatibilityValidator,
    PerformanceValidator,
    FunctionalValidator,
    validate_migration
)

from .progress_tracker import (
    ProgressTracker,
    MigrationMetrics,
    ProgressReport,
    Milestone,
    StatusDashboard,
    generate_progress_report
)

from .legacy_analyzer import (
    LegacyCodeAnalyzer,
    LegacyComponent,
    TechnicalDebt,
    RefactoringOpportunity,
    ModernizationTarget,
    analyze_legacy_codebase
)

__all__ = [
    # Strategy planning
    "MigrationPlanner",
    "MigrationStrategy",
    "MigrationPhase",
    "MigrationPlan",
    "ComponentAnalysis",
    "DependencyGraph",
    "RiskAssessment",
    "create_migration_plan",
    
    # Execution engine
    "MigrationExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "MigrationTask",
    "TaskType",
    "ParallelExecutor",
    "RollbackManager",
    "execute_migration_plan",
    
    # Validation framework
    "MigrationValidator",
    "ValidationResult",
    "ValidationSuite",
    "CompatibilityValidator",
    "PerformanceValidator",
    "FunctionalValidator",
    "validate_migration",
    
    # Progress tracking
    "ProgressTracker",
    "MigrationMetrics",
    "ProgressReport",
    "Milestone",
    "StatusDashboard",
    "generate_progress_report",
    
    # Legacy analysis
    "LegacyCodeAnalyzer",
    "LegacyComponent",
    "TechnicalDebt",
    "RefactoringOpportunity",
    "ModernizationTarget",
    "analyze_legacy_codebase"
]