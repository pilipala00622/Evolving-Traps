from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, Dict

from core.round_manager import INSUFFICIENT_PATTERNS, NO_PATTERNS, YES_PATTERNS


def normalize_carrier(carrier: str | None) -> str:
    value = (carrier or "").strip().lower()
    raw = (carrier or "").strip()
    if value in {"numeric", "numeric_value"} or "numeric" in value or "数值" in raw:
        return "numeric"
    if value in {"entity_set"} or "entity" in value or "实体" in raw:
        return "entity_set"
    if value in {"citation_set"} or "citation" in value or "引用" in raw or "出处" in raw:
        return "citation_set"
    if value in {"boolean", "bool"} or "boolean" in value or "布尔" in raw or "是非" in raw:
        return "boolean"
    return value


def detect_insufficient_evidence(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    return any(pattern in normalized for pattern in INSUFFICIENT_PATTERNS)


def detect_number(answer: str) -> bool:
    return bool(re.search(r"\d", answer))


def detect_entity_list(answer: str) -> bool:
    return answer.count("、") >= 1 or answer.count("\n") >= 2 or bool(re.search(r"[1-9][\.\)]", answer))


def detect_citation(answer: str) -> bool:
    return any(token in answer for token in ("第", "章", "页", "出处", "见", "参考"))


def maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if isinstance(completion.get("content"), str):
            return completion["content"]
        if isinstance(completion.get("text"), str):
            return completion["text"]
        messages = completion.get("messages")
        if isinstance(messages, Iterable):
            parts = []
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    parts.append(message["content"])
            return "\n".join(part for part in parts if part)
    if isinstance(completion, Iterable) and not isinstance(completion, (bytes, bytearray)):
        parts = []
        for item in completion:
            text = completion_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(completion)


def classify_completion(answer: str, reward_spec: Dict[str, Any]) -> Dict[str, Any]:
    answer = (answer or "").strip()
    if not answer:
        return {"label": "unparseable", "judgeable": False, "reason": "empty_answer"}

    if detect_insufficient_evidence(answer):
        return {"label": "correct", "judgeable": True, "reason": "insufficient_evidence_detected"}

    metadata = reward_spec.get("metadata", {})
    carrier = normalize_carrier(metadata.get("answer_carrier", ""))

    if carrier == "numeric":
        if detect_number(answer):
            return {"label": "target_error", "judgeable": True, "reason": "numeric_answer_under_missing_info"}
        return {"label": "non_target_error", "judgeable": True, "reason": "non_numeric_non_abstain"}

    if carrier == "entity_set":
        if detect_entity_list(answer):
            return {"label": "target_error", "judgeable": True, "reason": "entity_overgeneration_under_missing_info"}
        return {"label": "non_target_error", "judgeable": True, "reason": "entity_non_abstain"}

    if carrier == "citation_set":
        if detect_citation(answer):
            return {"label": "target_error", "judgeable": True, "reason": "citation_fabrication_proxy"}
        return {"label": "non_target_error", "judgeable": True, "reason": "citation_non_abstain"}

    if carrier == "boolean":
        short = answer[:24]
        if any(token in short for token in YES_PATTERNS + NO_PATTERNS):
            return {"label": "target_error", "judgeable": True, "reason": "boolean_hard_answer_under_insufficient_evidence"}
        return {"label": "non_target_error", "judgeable": True, "reason": "boolean_non_abstain"}

    return {"label": "unparseable", "judgeable": False, "reason": "unknown_carrier"}


def score_completion(answer: str, reward_spec: Dict[str, Any]) -> Dict[str, Any]:
    result = classify_completion(answer, reward_spec)
    label = result["label"]
    if label == "correct":
        reward = 1.0
    elif label == "target_error":
        reward = -1.0
    elif label == "non_target_error":
        reward = -0.4
    else:
        reward = -0.2
    return {**result, "reward": reward}
