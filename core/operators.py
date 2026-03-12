"""
遗传算子模块：选择、交叉、变异

与EvoLLMs的核心差异：
- EvoLLMs没有真正的交叉操作（只有单个QA对的变体生成）
- 我们实现真正的种群级交叉：两个个体交换基因片段
- 变异是归因感知的：可以定向增强某种陷阱配置
"""

import copy
import random
from typing import List, Tuple

from core.gene import (
    GENE_LAYERS,
    Individual,
    apply_mutation_level,
    build_individual_from_layers,
)
from config import GAConfig, ATTRIBUTION_TYPES


class Selection:
    """选择算子：决定哪些个体有资格繁殖"""

    @staticmethod
    def tournament(
        population: List[Individual],
        tournament_size: int = 5
    ) -> Individual:
        """
        锦标赛选择：随机取K个个体，返回fitness最高的。
        
        为什么选锦标赛而不是轮盘赌：
        - 轮盘赌需要fitness值均匀分布，我们的fitness分布可能很不均匀
        - 锦标赛的选择压力可以通过tournament_size控制
        """
        candidates = random.sample(population, min(tournament_size, len(population)))
        return max(candidates, key=lambda ind: ind.total_fitness)

    @staticmethod
    def select_parents(
        population: List[Individual],
        n_pairs: int,
        tournament_size: int = 5
    ) -> List[Tuple[Individual, Individual]]:
        """选择N对亲本"""
        pairs = []
        for _ in range(n_pairs):
            parent_a = Selection.tournament(population, tournament_size)
            parent_b = Selection.tournament(population, tournament_size)
            # 避免自交
            attempts = 0
            while parent_b.id == parent_a.id and attempts < 10:
                parent_b = Selection.tournament(population, tournament_size)
                attempts += 1
            pairs.append((parent_a, parent_b))
        return pairs

    @staticmethod
    def elitism(
        population: List[Individual],
        elite_ratio: float
    ) -> List[Individual]:
        """精英策略：直接保留Top N%"""
        n_elite = max(1, int(len(population) * elite_ratio))
        sorted_pop = sorted(population, key=lambda ind: ind.total_fitness, reverse=True)
        return [copy.deepcopy(ind) for ind in sorted_pop[:n_elite]]


class Crossover:
    """
    交叉算子：两个亲本交换基因片段，产生后代
    
    这是与EvoLLMs最本质的区别：
    EvoLLMs的Model 2只是对单个QA做paraphrase变体
    我们是真正的两个个体间的基因重组
    """

    @staticmethod
    def uniform_crossover(
        parent_a: Individual,
        parent_b: Individual,
        generation: int = 0
    ) -> Tuple[Individual, Individual]:
        """
        均匀交叉：每个基因层独立地从两个亲本中选择一个继承。
        
        例：
          亲本A: [Query_A, Context_A, Trap_A, Diff_A]
          亲本B: [Query_B, Context_B, Trap_B, Diff_B]
          后代1: [Query_A, Context_B, Trap_A, Diff_B]  ← 随机组合
          后代2: [Query_B, Context_A, Trap_B, Diff_A]  ← 互补组合
        """
        child1_layers = {}
        child2_layers = {}
        parent_a_layers = parent_a.gene_layers()
        parent_b_layers = parent_b.gene_layers()

        for layer in GENE_LAYERS:
            if random.random() < 0.5:
                child1_layers[layer] = parent_a_layers[layer]
                child2_layers[layer] = parent_b_layers[layer]
            else:
                child1_layers[layer] = parent_b_layers[layer]
                child2_layers[layer] = parent_a_layers[layer]

        child1 = build_individual_from_layers(
            child1_layers,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
        )
        child2 = build_individual_from_layers(
            child2_layers,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
        )
        return child1, child2

    @staticmethod
    def attribution_guided_crossover(
        parent_a: Individual,
        parent_b: Individual,
        target_attribution: str,
        generation: int = 0
    ) -> Individual:
        """
        归因引导的交叉：从两个亲本中选择对目标归因类型贡献更大的基因。
        
        这是本框架独有的操作 —— 利用归因体系指导进化方向。
        
        例：目标是增强"错误拼接"触发能力
        → 选择 shared_entities 更高的亲本的 ContextGene
        → 选择 cross_doc_overlap 更高的亲本的 TrapGene
        """
        if target_attribution not in ATTRIBUTION_TYPES:
            return Crossover.uniform_crossover(parent_a, parent_b, generation)[0]

        trigger_genes = ATTRIBUTION_TYPES[target_attribution]["trigger_genes"]

        # 对于Context层：比较哪个亲本的相关参数更有利
        a_context_score = sum(
            getattr(parent_a.context_gene, g, 0)
            for g in trigger_genes if hasattr(parent_a.context_gene, g)
        )
        b_context_score = sum(
            getattr(parent_b.context_gene, g, 0)
            for g in trigger_genes if hasattr(parent_b.context_gene, g)
        )
        context = copy.deepcopy(
            parent_a.context_gene if a_context_score >= b_context_score
            else parent_b.context_gene
        )

        # 对于Trap层：选择相关陷阱参数更强的
        a_trap_score = sum(
            getattr(parent_a.trap_gene, g, 0)
            for g in trigger_genes if hasattr(parent_a.trap_gene, g)
        )
        b_trap_score = sum(
            getattr(parent_b.trap_gene, g, 0)
            for g in trigger_genes if hasattr(parent_b.trap_gene, g)
        )
        trap = copy.deepcopy(
            parent_a.trap_gene if a_trap_score >= b_trap_score
            else parent_b.trap_gene
        )
        trap.target_attribution = target_attribution

        layers = {
            "query": copy.deepcopy(random.choice([parent_a.query_gene, parent_b.query_gene])),
            "context": context,
            "trap": trap,
            "difficulty": copy.deepcopy(random.choice([parent_a.difficulty_gene, parent_b.difficulty_gene])),
        }

        return build_individual_from_layers(
            layers,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
        )


class Mutation:
    """
    变异算子：对个体的基因进行随机修改
    
    三级变异强度 + 动态变异率调整
    """

    def __init__(self, config: GAConfig):
        self.config = config
        self._stagnation_counter = 0
        self._current_rates = config.mutation_rates.copy()

    def mutate(self, individual: Individual) -> Individual:
        """对个体执行变异操作"""
        child = copy.deepcopy(individual)

        # 按概率选择变异强度
        r = random.random()
        cumulative = 0.0

        for level, rate in self._current_rates.items():
            cumulative += rate
            if r <= cumulative:
                apply_mutation_level(child, level)
                break

        child.reset_runtime_state(generation=individual.generation, parent_ids=individual.parent_ids)
        return child

    def update_rates(self, fitness_improved: bool):
        """
        动态变异率：连续多代不提升时增加重度变异概率
        
        设计原理：
        - 正常进化：轻度变异为主，精细调整
        - 陷入停滞：提高重度变异概率，跳出局部最优
        - 恢复提升：回到正常比例
        """
        if not self.config.dynamic_mutation:
            return

        if fitness_improved:
            self._stagnation_counter = 0
            self._current_rates = self.config.mutation_rates.copy()
        else:
            self._stagnation_counter += 1
            if self._stagnation_counter >= self.config.mutation_boost_threshold:
                # 提高重度变异概率
                self._current_rates = {
                    "light": 0.30,
                    "medium": 0.35,
                    "heavy": 0.35,
                }

    @property
    def current_rates(self) -> dict:
        return self._current_rates.copy()
