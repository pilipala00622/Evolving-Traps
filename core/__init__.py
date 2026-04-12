"""
Lightweight exports for the refactored core package.

The previous __init__ eagerly imported legacy modules that no longer exist in
this repository, which made `import core.round_manager` fail before the actual
submodule could load. Keep this package init intentionally small and only
re-export modules that are present in the current codebase.
"""

from core.agent_specs import TaskSpec, TrajectorySpec, VerifierSpec
from core.round_manager import (
    DIFFICULTY_DIMENSIONS,
    FAILURE_MANIFESTATIONS,
    FAILURE_MECHANISMS,
    GENE_SCHEMA_VERSION,
    HALLUSEA_GATES,
    TARGET_ERROR_TYPES,
    TRAP_BOUNDARY_SCOPES,
    TRAP_DISTRACTOR_STYLES,
    TRAP_EVIDENCE_LAYOUTS,
    TRAP_PRESSURE_PATTERNS,
    difficulty_bucket,
    difficulty_score,
    upgrade_gene_schema,
    validate_gene_batch,
    validate_gene_schema,
)
from core.training_readiness import filter_verified_release_candidates

__all__ = [
    "TaskSpec",
    "TrajectorySpec",
    "VerifierSpec",
    "DIFFICULTY_DIMENSIONS",
    "FAILURE_MANIFESTATIONS",
    "FAILURE_MECHANISMS",
    "GENE_SCHEMA_VERSION",
    "HALLUSEA_GATES",
    "TARGET_ERROR_TYPES",
    "TRAP_BOUNDARY_SCOPES",
    "TRAP_DISTRACTOR_STYLES",
    "TRAP_EVIDENCE_LAYOUTS",
    "TRAP_PRESSURE_PATTERNS",
    "difficulty_bucket",
    "difficulty_score",
    "upgrade_gene_schema",
    "validate_gene_batch",
    "validate_gene_schema",
    "filter_verified_release_candidates",
]
