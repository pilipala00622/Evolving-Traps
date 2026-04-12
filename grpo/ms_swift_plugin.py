from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from swift.rewards import ORM, orms
except ImportError:
    try:
        from swift.plugin import ORM, orms
    except ImportError:  # pragma: no cover - local dry-run fallback
        class ORM:  # type: ignore[no-redef]
            pass

        orms = {}

from grpo.reward import completion_to_text, maybe_parse_json, score_completion


def _expand_column(values: Any, target_len: int) -> List[Any]:
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list):
        if not values:
            return [None] * target_len
        if len(values) == target_len:
            return values
        if len(values) == 1:
            return values * target_len
        if target_len % len(values) == 0:
            factor = target_len // len(values)
            expanded: List[Any] = []
            for value in values:
                expanded.extend([value] * factor)
            return expanded
        return values[:target_len] + [values[-1]] * max(0, target_len - len(values))
    return [values] * target_len


def _build_reward_spec(kwargs: Dict[str, List[Any]], index: int) -> Dict[str, Any]:
    def value(name: str) -> Any:
        items = kwargs.get(name) or []
        if not items:
            return None
        if index < len(items):
            return items[index]
        return items[-1]

    reward_spec = maybe_parse_json(value("reward_spec_json")) or {}
    if reward_spec:
        return reward_spec

    return {
        "reward_mode": value("reward_mode") or "binary_outcome",
        "field_rules": maybe_parse_json(value("field_rules_json")) or [],
        "success_criteria": maybe_parse_json(value("success_criteria_json")) or [],
        "failure_reasons": maybe_parse_json(value("failure_reasons_json")) or [],
        "metadata": {
            "answer_carrier": value("answer_carrier") or "",
            "target_error_type": value("target_error_type") or "",
            "task_id": value("task_id") or "",
            "plan_id": value("plan_id") or "",
        },
    }


class EvoHalluOutcomeReward(ORM):
    """
    ms-swift external reward for evidence-boundary / abstention tasks.

    It reuses the repository's existing outcome classifier so evaluation and
    RL rewards stay aligned.
    """

    def __call__(self, completions, **kwargs) -> List[float]:  # type: ignore[override]
        total = len(completions)
        expanded = {
            key: _expand_column(value, total)
            for key, value in kwargs.items()
        }
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            reward_spec = _build_reward_spec(expanded, index)
            answer_text = completion_to_text(completion)
            scored = score_completion(answer_text, reward_spec)
            rewards.append(float(scored["reward"]))
        return rewards


orms["evohallu_outcome"] = EvoHalluOutcomeReward
