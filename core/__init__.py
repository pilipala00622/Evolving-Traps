# EvoHallu Core
from core.gene import (
    GENE_LAYERS,
    Individual,
    IndividualFactory,
    QueryGene,
    ContextGene,
    TrapGene,
    DifficultyGene,
)
from core.fitness import FitnessEvaluator, FitnessResult
from core.operators import Selection, Crossover, Mutation
from core.llm_interface import LLMInterface
from core.evolution import EvolutionEngine
from core.benchmark_schema import BenchmarkItem, CalibrationStats, HumanReviewState, ValidationStats
from core.benchmark_validation import BenchmarkCalibrator, BenchmarkValidator
from core.plan_workflow import EvolutionPlan, PlanReflection
from core.training_readiness import filter_verified_release_candidates
from core.agent_specs import TaskSpec, TrajectorySpec, VerifierSpec
from core.plan_updater import UpdatedPlan
