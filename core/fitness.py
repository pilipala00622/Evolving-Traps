"""
适应度评估模块：归因感知的多目标适应度函数

与EvoLLMs的关键区别：
- EvoLLMs用15个通用质量指标，不区分幻觉类型
- 我们的fitness是归因类型敏感的，能评估题目对特定幻觉类型的触发精准度

三维适应度 = α×区分度 + β×归因覆盖度 + γ×有效性 - λ×重复度
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config import GAConfig, ATTRIBUTION_TYPES


@dataclass
class FitnessResult:
    """单个个体的适应度评估结果"""
    discrimination: float = 0.0         # 区分度 [0, 1]
    coverage_bonus: float = 0.0         # 归因覆盖度奖励 [0, 1]
    validity: float = 0.0               # 有效性 [0, 1]
    redundancy: float = 0.0             # 与已有题库的重复度 [0, 1]
    total: float = 0.0                  # 加权总分
    details: Dict = None                # 详细评估信息


class FitnessEvaluator:
    """
    归因感知的适应度评估器
    
    核心设计：适应度不是单一的"好坏分"，而是一个多维评估：
    1. 区分度：这道题能否让不同水平的模型表现出差异
    2. 归因覆盖度：这道题触发的幻觉类型在当前种群中是否稀缺
    3. 有效性：这道题本身是否合理、可作答
    """

    def __init__(self, config: GAConfig):
        self.config = config
        self.population_attribution_dist: Dict[str, int] = {
            attr: 0 for attr in ATTRIBUTION_TYPES
        }
        self._existing_embeddings: List[np.ndarray] = []

    # ===========================================================
    # 维度1：区分度（Discrimination Score）
    # ===========================================================

    def compute_discrimination(
        self,
        model_hallucination_rates: Dict[str, float]
    ) -> float:
        """
        计算这道题在不同模型间的幻觉率方差。
        
        Args:
            model_hallucination_rates: {模型名: 该模型在此题上的幻觉率}
                例: {"gpt-5.2": 0.15, "deepseek-v3.2": 0.35, "ernie-5.0": 0.55}
        
        Returns:
            区分度得分 [0, 1]，方差越大得分越高
        
        设计原理：
            - 所有模型都答对（方差=0）→ 得分0（太简单，没有区分价值）
            - 所有模型都答错（方差=0）→ 得分0（太难，也没有区分价值）
            - 头部答对、尾部答错（方差大）→ 得分高（有区分价值）
        """
        if not model_hallucination_rates:
            return 0.0

        rates = list(model_hallucination_rates.values())
        variance = np.var(rates)

        # 归一化：理论最大方差 = 0.25（当一半是0，一半是1时）
        normalized = min(variance / 0.25, 1.0)

        # 额外奖励：如果头部模型和尾部模型的差距明显
        if len(rates) >= 2:
            sorted_rates = sorted(rates)
            spread = sorted_rates[-1] - sorted_rates[0]
            # 差距>0.3额外加分
            spread_bonus = max(0, (spread - 0.3) * 0.5)
            normalized = min(normalized + spread_bonus, 1.0)

        return float(normalized)

    # ===========================================================
    # 维度2：归因覆盖度（Attribution Coverage Bonus）
    # ===========================================================

    def compute_coverage_bonus(
        self,
        triggered_attributions: Dict[str, float]
    ) -> float:
        """
        计算这道题触发的归因类型在当前种群中的稀缺度。
        
        Args:
            triggered_attributions: {归因类型: 触发强度}
                例: {"错误匹配": 0.8, "缺证断言": 0.3}
        
        Returns:
            覆盖度奖励 [0, 1]
        
        设计原理（与EvoLLMs的核心差异）：
            - 不是每道题都要覆盖多种归因类型
            - 而是让整个种群均衡覆盖所有归因类型
            - 如果当前种群中"确定性膨胀"类题目不足，
              那么一道能触发"确定性膨胀"的题就获得更高的覆盖度奖励
        """
        if not triggered_attributions:
            return 0.0

        # 计算当前种群中各归因类型的分布
        total_count = max(sum(self.population_attribution_dist.values()), 1)
        type_frequencies = {
            attr: count / total_count
            for attr, count in self.population_attribution_dist.items()
        }

        # 理想均匀分布
        ideal_freq = 1.0 / len(ATTRIBUTION_TYPES)

        # 对这道题触发的每种归因类型，计算"稀缺度奖励"
        bonus = 0.0
        for attr_type, trigger_strength in triggered_attributions.items():
            if attr_type not in type_frequencies:
                continue
            current_freq = type_frequencies[attr_type]
            # 越稀缺（当前频率远低于理想频率），奖励越高
            scarcity = max(0, ideal_freq - current_freq) / ideal_freq
            bonus += trigger_strength * scarcity

        # 归一化
        return float(min(bonus / len(triggered_attributions), 1.0)) if triggered_attributions else 0.0

    def update_population_distribution(
        self,
        population_attributions: List[Dict[str, float]]
    ):
        """更新种群层面的归因类型分布（每代开始时调用）"""
        self.population_attribution_dist = {attr: 0 for attr in ATTRIBUTION_TYPES}
        for attrs in population_attributions:
            dominant = max(attrs, key=attrs.get) if attrs else None
            if dominant and dominant in self.population_attribution_dist:
                self.population_attribution_dist[dominant] += 1

    # ===========================================================
    # 维度3：有效性（Validity）
    # ===========================================================

    def compute_validity(
        self,
        has_reference_answer: bool,
        naturalness_score: float,
        is_answerable: bool
    ) -> float:
        """
        评估题目本身的质量。
        
        Args:
            has_reference_answer: 是否有明确的标准答案
            naturalness_score: 自然度评分 (1-5, LLM判断是否像真实用户问题)
            is_answerable: 基于给定context是否可作答
        
        Returns:
            有效性得分 [0, 1]
        """
        score = 0.0

        # 必须有标准答案
        if has_reference_answer:
            score += 0.3

        # 必须可作答
        if is_answerable:
            score += 0.3

        # 自然度 (归一化到0-0.4)
        naturalness_normalized = max(0, (naturalness_score - 1) / 4) * 0.4

        # 低于阈值直接判无效
        if naturalness_score < self.config.validity_threshold:
            return 0.0

        score += naturalness_normalized
        return float(min(score, 1.0))

    # ===========================================================
    # 重复度惩罚
    # ===========================================================

    def compute_redundancy(self, query_embedding: np.ndarray) -> float:
        """
        计算与已有题库的语义重复度。
        
        用cosine similarity衡量新题与已有题的最大相似度。
        超过阈值的会被严重惩罚。
        """
        if not self._existing_embeddings or query_embedding is None:
            return 0.0

        similarities = [
            np.dot(query_embedding, existing) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(existing) + 1e-8
            )
            for existing in self._existing_embeddings
        ]
        max_sim = max(similarities) if similarities else 0.0

        # 超过阈值的急剧惩罚
        if max_sim > self.config.similarity_threshold:
            return 1.0
        # 线性惩罚
        return float(max(0, max_sim - 0.5) / (self.config.similarity_threshold - 0.5))

    def add_to_existing(self, embedding: np.ndarray):
        """将新题的embedding加入已有题库"""
        self._existing_embeddings.append(embedding)

    # ===========================================================
    # 总适应度计算
    # ===========================================================

    def evaluate(
        self,
        model_hallucination_rates: Dict[str, float],
        triggered_attributions: Dict[str, float],
        has_reference_answer: bool,
        naturalness_score: float,
        is_answerable: bool,
        query_embedding: Optional[np.ndarray] = None,
        current_generation: int = 0,
    ) -> FitnessResult:
        """
        计算完整的适应度得分。
        
        分阶段策略：
        - Phase 1 (前N代)：只用区分度 + 有效性，先找到"能用的题"
        - Phase 2 (后续代)：加入归因覆盖度，优化归因分布
        """
        w = self.config.fitness_weights.copy()

        # Phase 1：不考虑归因覆盖度
        if current_generation < self.config.phase1_generations:
            w["coverage"] = 0.0
            # 重新归一化权重
            total_w = w["discrimination"] + w["validity"]
            if total_w > 0:
                w["discrimination"] /= total_w
                w["validity"] /= total_w

        # 计算各维度
        discrimination = self.compute_discrimination(model_hallucination_rates)
        coverage = self.compute_coverage_bonus(triggered_attributions)
        validity = self.compute_validity(has_reference_answer, naturalness_score, is_answerable)
        redundancy = self.compute_redundancy(query_embedding)

        # 有效性为0 → 直接淘汰
        if validity == 0.0:
            return FitnessResult(
                discrimination=discrimination,
                coverage_bonus=coverage,
                validity=0.0,
                redundancy=redundancy,
                total=0.0,
                details={"eliminated": "validity_zero"}
            )

        # 加权求和
        total = (
            w["discrimination"] * discrimination
            + w.get("coverage", 0) * coverage
            + w["validity"] * validity
            - self.config.redundancy_penalty * redundancy
        )

        return FitnessResult(
            discrimination=discrimination,
            coverage_bonus=coverage,
            validity=validity,
            redundancy=redundancy,
            total=max(0.0, total),
            details={
                "weights": w,
                "phase": 1 if current_generation < self.config.phase1_generations else 2,
            }
        )
