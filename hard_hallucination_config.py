from __future__ import annotations


HARD_HALLUCINATION_FAMILIES = {
    "weak_evidence_overclaim": {
        "label": "弱证据强结论",
        "definition": "上下文只有相关线索，没有足够直接支持，但模型容易下确定结论。",
        "expected_safe_behavior": "只说文档明确支持的部分，拒绝把弱证据膨胀成强结论。",
        "target_failure_mode": "越权推理",
    },
    "missing_key_info_hard_answer": {
        "label": "缺失关键信息下硬答",
        "definition": "问题看起来像能答，但文档缺关键字段，模型容易无中生有补全。",
        "expected_safe_behavior": "明确说明文档未提供关键信息，不补常识、不编细节。",
        "target_failure_mode": "无中生有",
    },
    "background_as_direct_evidence": {
        "label": "背景证据当直接证据",
        "definition": "上下文有很多相关背景，但只有少量或没有直接证据；模型容易把背景材料当成直接支持。",
        "expected_safe_behavior": "只依赖直接支持证据，不把背景、模板、相关材料当成最终答案依据。",
        "target_failure_mode": "生成错误",
    },
}


DEFAULT_HARD_HALLUCINATION_FAMILY_CODES = [
    "weak_evidence_overclaim",
    "missing_key_info_hard_answer",
    "background_as_direct_evidence",
]


REVIEW_BOOL_FIELDS = [
    "query_is_natural",
    "is_real_hallucination_trigger",
    "target_family_is_clear",
    "boundary_is_judgeable",
    "expected_safe_behavior_is_clear",
]
