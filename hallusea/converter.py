"""
hallusea/converter.py
=====================
GRIT 格式 → HalluSEA / spec_factory 所需格式的适配层。

GRIT 的输出格式（gene + eval_result）与 core/spec_factory.py 期待的
benchmark_item schema 在字段命名和结构上有差异。
本模块提供无状态的转换函数，使 curriculum.py 能直接复用
spec_factory.benchmark_items_to_training_specs() 而无需修改其接口。

核心映射关系
-----------
GRIT gene 字段          → benchmark_item 字段
────────────────────────────────────────────────
gene_id                 → item_id
seed_id                 → metadata.seed_id
failure_mechanism       → scenario_type（映射见 _MECHANISM_TO_SCENARIO）
target_error_type       → target_error_type
answer_carrier          → metadata.answer_carrier
query（来自 eval_result）→ query
context（来自 eval_result）→ context（若无则留空）
verifier_shape          → metadata.verifier_shape

HalluSEA VerifierSpec 的 success_criteria 由
carrier_rules_to_success_criteria() 根据 CARRIER_RULES 自动生成，
与 evaluate_hard_hallucination_candidates.py 的 auto_label 逻辑保持一致。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.round_manager import CARRIER_RULES, HALLUSEA_GATES


# failure_mechanism → scenario_type 的映射（spec_factory 使用 scenario_type）
_MECHANISM_TO_SCENARIO: Dict[str, str] = {
    "weak_evidence_to_strong_conclusion": "weak_evidence",
    "missing_info_hard_answer":           "missing_info",
    "background_as_direct_evidence":      "background_as_evidence",
}


def carrier_rules_to_success_criteria(answer_carrier: str) -> List[str]:
    """
    根据 CARRIER_RULES 生成 VerifierSpec.success_criteria 列表。

    返回的列表与 evaluate 脚本的 auto_label 规则逻辑保持一致，
    保证训练时奖励函数的判定标准与评测时完全相同。
    """
    rules = CARRIER_RULES.get(answer_carrier, {})
    criteria: List[str] = []
    if rules.get("correct"):
        criteria.append(f"correct条件: {rules['correct']}")
    if rules.get("target_error"):
        criteria.append(f"target_error条件: {rules['target_error']}")
    if rules.get("non_target"):
        criteria.append(f"non_target_error条件: {rules['non_target']}")
    if not criteria:
        criteria.append("unknown carrier – 需人工定义成功判定逻辑")
    return criteria


def grit_gene_to_benchmark_item(
    gene: Dict[str, Any],
    eval_results_for_gene: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    将 GRIT gene 记录（含 metrics / fitness_history）转换为 benchmark_item 格式。

    Parameters
    ----------
    gene : GRIT gene 记录（来自 gene_archive_rN.jsonl）
    eval_results_for_gene : 该基因对应的评测结果列表（来自 model_answers_and_autoeval.jsonl）
                            用于提取代表性 query / context

    Returns
    -------
    benchmark_item 格式的字典，可直接传入 spec_factory.benchmark_item_to_task_spec()
    """
    gene_id = gene.get("gene_id") or gene.get("seed_id", "")
    answer_carrier = gene.get("answer_carrier", "")
    failure_mechanism = gene.get("failure_mechanism", "")
    scenario_type = _MECHANISM_TO_SCENARIO.get(failure_mechanism, failure_mechanism)

    # 优先从 eval_results 中取 query / context（实际诱发结果）
    query = ""
    context = ""
    if eval_results_for_gene:
        rep = eval_results_for_gene[0]
        query = rep.get("query", "")
        context = rep.get("context", "")
    if not query:
        query = gene.get("source_query", "") or gene.get("query", "")

    metrics = gene.get("metrics", {})
    tehr = metrics.get("tehr", 0.0)
    sis  = metrics.get("sis",  0.0)
    purity = metrics.get("purity", 0.0)

    return {
        "item_id": gene_id,
        "target_error_type": gene.get("target_error_type", ""),
        "scenario_type": scenario_type,
        "query": query,
        "context": context,
        "reference_answer": gene.get("expected_safe_behavior", ""),
        "gene_vector": {
            "domain": gene.get("knowledge_base_category", ""),
            "task_type": gene.get("task_frame", ""),
            "failure_mechanism": failure_mechanism,
            "answer_carrier": answer_carrier,
            "trigger_form": gene.get("trigger_form", ""),
            "support_gap_type": gene.get("support_gap_type", ""),
        },
        "metadata": {
            "seed_id": gene.get("seed_id", ""),
            "gene_id": gene_id,
            "round_id": gene.get("round_id", 0),
            "model_version": gene.get("model_version", ""),
            "gene_schema_version": gene.get("gene_schema_version", "v1"),
            "plan_id": f"grit_r{gene.get('round_id', 0)}",
            "answer_carrier": answer_carrier,
            "verifier_shape": gene.get("verifier_shape", ""),
            "abstention_expected": gene.get("abstention_expected", True),
            "plan_summary": {
                "complexity_bucket": _tehr_to_complexity(tehr),
            },
        },
        "evaluation_record": {
            "reference_answer_applicable": True,
            "scoring_mode": "abstention_based",
            "validation_stats": {
                "tehr":            tehr,
                "sis":             sis,
                "purity":          purity,
                "answerability_rate": metrics.get("judgeable_rate", 1.0),
            },
            "calibration_stats": {},
            "fixed_metrics": {
                "benchmark_stability":      round(sis, 4),
                "benchmark_discrimination": round(tehr, 4),
            },
        },
        "human_review": {
            "status": "approved",   # 进入 curriculum 的基因已通过 archive 质量门
            "review_result": {
                "is_single_target_error": True,
                "reward_should_be_verifiable": True,
                "reference_answer_supported": True,
            },
        },
        "release_record": {
            "eligible_for_release": True,
        },
        # HalluSEA 扩展字段：用于生成与 auto_label 一致的 VerifierSpec
        "_hallusea": {
            "answer_carrier": answer_carrier,
            "success_criteria": carrier_rules_to_success_criteria(answer_carrier),
            "fitness_history": gene.get("fitness_history", []),
        },
    }


def grit_eval_result_to_verifier_input(
    eval_result: Dict[str, Any],
    gene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将单条 evaluate_hard_hallucination_candidates 输出直接转换为
    VerifierSpec 构建所需的最小输入字典。

    适用于不走完整 benchmark_item 路径、直接从 eval_result 快速构建
    VerifierSpec 的轻量场景。
    """
    carrier = eval_result.get("answer_carrier", "")
    if not carrier and gene:
        carrier = gene.get("answer_carrier", "")

    return {
        "task_id": eval_result.get("candidate_id", ""),
        "gene_id": eval_result.get("gene_id", ""),
        "answer_carrier": carrier,
        "target_error_type": eval_result.get("target_error_type", ""),
        "auto_label": eval_result.get("auto_label", ""),
        "rule_name": eval_result.get("rule_name", ""),
        "query": eval_result.get("query", ""),
        "model_name": eval_result.get("model_name", ""),
        "round_id": eval_result.get("round_id", 0),
        "model_version": eval_result.get("model_version", ""),
        "success_criteria": carrier_rules_to_success_criteria(carrier),
        "reward_mode": "binary_outcome",
    }


def _tehr_to_complexity(tehr: float) -> str:
    """将 TEHR 映射到 complexity_bucket，用于课程难度标记。"""
    if tehr >= 0.8:
        return "hard"
    if tehr >= 0.5:
        return "medium"
    return "easy"


def filter_archive_for_hallusea(
    population: List[Dict[str, Any]],
    round_id: int,
    prev_tehr_map: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    将当前轮次的种群分为三类：
      - eligible    : 满足 HALLUSEA_GATES 的门槛（进入训练集）
      - solved      : 本轮 TEHR 相比上一轮下降 > 0.10（已学会，少量保留防遗忘）
      - too_noisy   : purity 或 sis 不达标（只归档，不进训练）

    Parameters
    ----------
    population      : build_gene_population() 输出的种群列表
    round_id        : 当前轮次号（0 = 无上一轮 TEHR 参考）
    prev_tehr_map   : {gene_id: prev_round_tehr}，用于检测 solved 状态
    """
    eligible: List[Dict[str, Any]] = []
    solved:   List[Dict[str, Any]] = []
    too_noisy: List[Dict[str, Any]] = []

    min_sis    = HALLUSEA_GATES["min_sis"]
    min_purity = HALLUSEA_GATES["min_purity"]
    min_ans    = HALLUSEA_GATES["min_answerability"]
    min_tehr_new = HALLUSEA_GATES.get("min_tehr_for_new_round", 0.30)

    for gene in population:
        m = gene.get("metrics", {})
        sis    = m.get("sis", 0.0)
        purity = m.get("purity", 0.0)
        tehr   = m.get("tehr", 0.0)
        ans    = m.get("judgeable_rate", 1.0)

        # 检查是否已被上一轮训练"解决"
        if round_id > 0 and prev_tehr_map:
            prev_tehr = prev_tehr_map.get(gene.get("gene_id", ""), None)
            if prev_tehr is not None and tehr < prev_tehr - 0.10:
                solved.append(gene)
                continue

        if sis < min_sis or purity < min_purity or ans < min_ans:
            too_noisy.append(gene)
            continue

        # Round N>0 额外要求：模型仍然失败，否则训练价值不高
        if round_id > 0 and tehr < min_tehr_new:
            solved.append(gene)
            continue

        eligible.append(gene)

    return {"eligible": eligible, "solved": solved, "too_noisy": too_noisy}
