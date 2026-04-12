"""
ms-swift GRPO preparation utilities for Qwen2.5-7B integration.
"""

from grpo.config import GRPOConfig
from grpo.dataset_builder import build_grpo_dataset_bundle, build_ms_swift_dataset_bundle
from grpo.reward import classify_completion, completion_to_text, score_completion

__all__ = [
    "GRPOConfig",
    "build_grpo_dataset_bundle",
    "build_ms_swift_dataset_bundle",
    "classify_completion",
    "completion_to_text",
    "score_completion",
]
