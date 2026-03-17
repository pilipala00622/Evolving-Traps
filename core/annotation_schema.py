"""
Query / response / sentence layered annotation schema.

This module formalizes the shift from:
- query-level "true attribution labels"

to:
- query-level intended failure modes / trigger quality
- response-level observed failure summaries
- sentence-level actual hallucination attribution
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryItem:
    """Prompt-level object: stores trigger intent, not ground-truth attribution."""

    query_id: str
    scenario_type: str
    query: str
    context: str
    intended_failure_mode: str = ""
    query_type: str = ""
    domain: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ResponseRun:
    """One concrete model response to a query."""

    response_id: str
    query_id: str
    model_name: str
    run_id: str = ""
    response_text: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SentenceAnnotation:
    """Sentence/span-level actual attribution; this is the reliable attribution layer."""

    annotation_id: str
    response_id: str
    sentence_id: str
    sentence_text: str
    is_hallucinated: bool
    attribution_type: str = ""
    evidence_support: str = ""
    severity: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ResponseEvaluationSummary:
    """Response-level summary derived from sentence annotations."""

    response_id: str
    dominant_attribution_type: str = ""
    hallucinated_sentence_count: int = 0
    total_sentence_count: int = 0
    attribution_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QueryEvaluationSummary:
    """Query-level summary should only express trigger tendencies, not truth labels."""

    query_id: str
    intended_failure_mode: str = ""
    trigger_success_rate: float = 0.0
    dominant_observed_failure_mode: str = ""
    response_count: int = 0
    model_consistency: float = 0.0
    should_keep_as_trigger: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
