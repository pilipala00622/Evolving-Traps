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
