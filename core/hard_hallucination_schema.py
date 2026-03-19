from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HardHallucinationSourceContext:
    trace_id: str
    knowledge_base_category: str
    query_category: str
    source_query: str
    source_question_text: str
    context_text: str
    context_length: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardHallucinationQuestionCard:
    card_id: str
    trace_id: str
    knowledge_base_category: str
    query: str
    hard_hallucination_family: str
    target_failure_mode: str
    target_failure_subtype: str
    trigger_mechanism_label: str
    trigger_mechanism_code: str
    expected_safe_behavior: str
    why_this_is_hallucination: str
    evidence_source_hint: list[str] = field(default_factory=list)
    judge_anchor: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardHallucinationReviewTask:
    review_id: str
    card_id: str
    trace_id: str
    knowledge_base_category: str
    query: str
    context: str
    hard_hallucination_family: str
    target_failure_mode: str
    target_failure_subtype: str
    trigger_mechanism_label: str
    expected_safe_behavior: str
    why_this_is_hallucination: str
    evidence_source_hint: list[str] = field(default_factory=list)
    judge_anchor: str = ""
    review_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
